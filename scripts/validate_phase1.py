#!/usr/bin/env python
"""
scripts/validate_phase1.py  –  Phase 1 Validation Suite

Validates the Mist Phase 1 DCT watermark engine against the specification:
    • PSNR > 40 dB
    • SSIM > 0.98
    • Detection confidence > 0.55 on:
        – Clean watermarked image
        – JPEG Q30
        – Resize 0.5× and 2×
        – Crop 30%
        – Brightness +20%
        – Brightness −20%
        – Gaussian blur σ=1.5
    • Confidence ≈ 0.5 on non-watermarked image (FPR sanity)

Usage
-----
    cd g:\\Projects\\Mist
    mist_env\\Scripts\\python scripts\\validate_phase1.py [path/to/image.jpg]

If no image path is given, a synthetic 512×512 test image is generated.
"""

import os
import sys
import io
import hashlib

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.wm_engine import embed, detect, detect_robust, embed_with_prng_payload

# ── Optional metrics imports (graceful fallback) ───────────────────────────
try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim

    def compute_psnr(orig: np.ndarray, wm: np.ndarray) -> float:
        return float(_psnr(orig, wm, data_range=255))

    def compute_ssim(orig: np.ndarray, wm: np.ndarray) -> float:
        return float(_ssim(orig, wm, channel_axis=2, data_range=255))

except ImportError:
    def compute_psnr(orig: np.ndarray, wm: np.ndarray) -> float:
        mse = np.mean((orig.astype(np.float64) - wm.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10(255.0 ** 2 / mse))

    def compute_ssim(orig: np.ndarray, wm: np.ndarray) -> float:  # rough approximation
        mu1 = orig.astype(np.float64).mean()
        mu2 = wm.astype(np.float64).mean()
        s1  = orig.astype(np.float64).std()
        s2  = wm.astype(np.float64).std()
        cov = np.mean((orig.astype(np.float64) - mu1) * (wm.astype(np.float64) - mu2))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        return float(((2*mu1*mu2 + c1)*(2*cov + c2)) / ((mu1**2 + mu2**2 + c1)*(s1**2 + s2**2 + c2)))


# ─────────────────────────────────────────────────────────────────────────────
#  Attack functions
# ─────────────────────────────────────────────────────────────────────────────

def attack_jpeg(img: np.ndarray, quality: int = 30) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def attack_resize(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Scale down then back up (simulates lossy re-encode)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_crop(img: np.ndarray, fraction: float = 0.30) -> np.ndarray:
    """
    Remove `fraction` of the area by cropping from all four edges equally.
    A 30% area removal ≈ removing ~15.5% from each linear dimension.
    """
    h, w = img.shape[:2]
    # Linear crop per edge to remove ~fraction total area
    margin_y = int(h * (1 - (1 - fraction) ** 0.5) / 2)
    margin_x = int(w * (1 - (1 - fraction) ** 0.5) / 2)
    margin_y = max(margin_y, 1)
    margin_x = max(margin_x, 1)
    return img[margin_y: h - margin_y, margin_x: w - margin_x]


def attack_brightness(img: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """Multiply pixel values by `factor`; factor>1 brightens, <1 darkens."""
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def attack_blur(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    ksize = int(6 * sigma + 1) | 1   # ensure odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# ─────────────────────────────────────────────────────────────────────────────
#  Test image loader
# ─────────────────────────────────────────────────────────────────────────────

def load_test_image(path: str | None = None) -> np.ndarray:
    """
    Load image from path, or from dataset/resized/, or generate synthetic.
    Returns a BGR uint8 ndarray.
    """
    # Try explicit path
    if path and os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            print(f"  Loaded: {path}")
            return img

    # Try dataset directory
    resized_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "resized")
    if os.path.isdir(resized_dir):
        candidates = [f for f in os.listdir(resized_dir)
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if candidates:
            p = os.path.join(resized_dir, candidates[0])
            img = cv2.imread(p)
            if img is not None:
                print(f"  Loaded: {p}")
                return img

    # Synthetic fallback — photo-like gradient + noise
    print("  No image found — using synthetic 512×512 test image.")
    rng = np.random.default_rng(42)
    H, W = 512, 512
    xx, yy = np.meshgrid(np.linspace(0, 255, W), np.linspace(0, 255, H))
    base = np.stack([xx, yy, 255 - xx + yy / 2], axis=2).astype(np.float32)
    noise = rng.normal(0, 20, (H, W, 3)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Validation runner
# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 70
PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
WARN = "  ⚠️  WARN"


def check(label: str, value: float, threshold: float, higher_is_better: bool = True) -> bool:
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    status = PASS if ok else FAIL
    print(f"  {label:<45} {value:>8.5f}   [threshold {'>=' if higher_is_better else '<='} {threshold}] {status}")
    return ok


def run_validation(image_path: str | None = None) -> None:
    print()
    print("━" * 70)
    print("  🌫  MIST  —  Phase 1 Validation Suite")
    print("━" * 70)

    # ── 1. Load image ────────────────────────────────────────────────────────
    print("\n[1] Loading test image")
    img = load_test_image(image_path)
    print(f"      Shape: {img.shape}  dtype: {img.dtype}")

    # ── 2. Generate key + bitstream ──────────────────────────────────────────
    key = hashlib.sha256(b"mist-phase1-test-key").digest()   # 32 bytes
    print(f"\n[2] Embedding key (SHA-256 of literal): {key.hex()[:32]}…")

    # ── 3. Embed ─────────────────────────────────────────────────────────────
    print("\n[3] Embedding watermark…")
    watermarked = embed_with_prng_payload(img, key)

    # ── 4. Perceptual quality ─────────────────────────────────────────────────
    print(f"\n[4] Perceptual Quality")
    print(SEP)
    psnr = compute_psnr(img, watermarked)
    ssim = compute_ssim(img, watermarked)
    all_pass = True
    all_pass &= check("PSNR (dB)", psnr, 40.0)
    all_pass &= check("SSIM", ssim, 0.98)
    print()

    # ── 5. Detection on clean watermarked image ───────────────────────────────
    print(f"[5] Detection — Clean Watermarked Image")
    print(SEP)
    result = detect(watermarked, key)
    conf = result["confidence"]
    raw  = result["raw_score"]
    print(f"  raw_score  : {raw:.6f}")
    all_pass &= check("confidence (clean watermarked)", conf, 0.55)
    print()

    # ── 6. Attack robustness ──────────────────────────────────────────────────
    attacks = [
        ("JPEG Q30",           lambda i: attack_jpeg(i, quality=30),      False),
        ("Resize 0.5×→orig",   lambda i: attack_resize(i, scale=0.5),     False),
        ("Resize 2.0×→orig",   lambda i: attack_resize(i, scale=2.0),     False),
        ("Crop 30%",           lambda i: attack_crop(i, fraction=0.30),   True),   # uses detect_robust
        ("Brightness ×1.2",    lambda i: attack_brightness(i, factor=1.2),False),
        ("Brightness ×0.8",    lambda i: attack_brightness(i, factor=0.8),False),
        ("Gaussian blur σ=1.5",lambda i: attack_blur(i, sigma=1.5),       False),
    ]

    print(f"[6] Detection — Attack Robustness")
    print(SEP)
    for attack_name, attack_fn, robust in attacks:
        attacked = attack_fn(watermarked)
        # Crop uses detect_robust (exhaustive grid search) — other attacks use detect()
        res = detect_robust(attacked, key) if robust else detect(attacked, key)
        c = res["confidence"]
        r = res["raw_score"]
        ok = c >= 0.55
        status = PASS if ok else FAIL
        all_pass &= ok
        print(f"  {attack_name:<40} conf={c:.5f}  raw={r:+.5f}  {status}")
    print()

    # ── 7. FPR sanity — non-watermarked image ────────────────────────────────
    print(f"[7] FPR Sanity — Non-Watermarked Image (expect conf ≈ 0.5)")
    print(SEP)
    fpr_result = detect(img, key)
    fc = fpr_result["confidence"]
    print(f"  confidence on original (no watermark): {fc:.6f}")
    fpr_ok = abs(fc - 0.5) < 0.05   # tightened: single-alignment detect() should give near-0.5
    status = PASS if fpr_ok else FAIL
    all_pass &= fpr_ok
    print(f"  |confidence − 0.5| = {abs(fc-0.5):.6f}  [expect < 0.05]  {status}")
    print()

    # ── 8. Wrong key sanity ───────────────────────────────────────────────────
    print(f"[8] Key Isolation — Wrong Key (expect conf ≈ 0.5)")
    print(SEP)
    wrong_key = hashlib.sha256(b"completely-wrong-key").digest()
    wrong_result = detect(watermarked, wrong_key)
    wc = wrong_result["confidence"]
    print(f"  confidence with wrong key: {wc:.6f}")
    key_ok = abs(wc - 0.5) < 0.1
    status = PASS if key_ok else WARN
    print(f"  |confidence − 0.5| = {abs(wc-0.5):.6f}  [expect < 0.10]  {status}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("━" * 70)
    verdict = "✅  ALL CHECKS PASSED  —  Phase 1 SUCCESS" if all_pass \
              else "❌  SOME CHECKS FAILED — review output above"
    print(f"  {verdict}")
    print("━" * 70)
    print()


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_validation(img_path)
