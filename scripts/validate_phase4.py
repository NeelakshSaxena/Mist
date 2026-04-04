"""
scripts/validate_phase4.py — Phase 4 Spatial Attack Resistance Benchmark

Usage:
    python -m scripts.validate_phase4
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.crypto   import generate_keys
from src.core.mist     import watermark_p4, verify_p4
from src.core.wm_engine_p4 import (
    detect_p4, extract_shards_p4, K_SHARDS, MT_SIZE,
    _outer_rs_decode_smart, _outer_rs_encode, _bits_to_bytes,
)
from src.core.ecc      import ECC_TOTAL_BYTES


def make_test_image(h: int = 512, w: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = ((y + x) / 2).astype(np.uint8)
    noise = rng.integers(0, 40, (h, w), dtype=np.uint8)
    gray = np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def psnr_ssim(original, modified):
    mse = float(np.mean((original.astype(float) - modified.astype(float)) ** 2))
    psnr = 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-10))
    o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(float)
    m = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY).astype(float)
    mu_o, mu_m = o.mean(), m.mean()
    sig_o, sig_m = o.std(), m.std()
    sig_om = float(np.mean((o - mu_o) * (m - mu_m)))
    c1, c2 = 6.5025, 58.5225
    ssim = ((2*mu_o*mu_m + c1) * (2*sig_om + c2)) / \
           ((mu_o**2 + mu_m**2 + c1) * (sig_o**2 + sig_m**2 + c2))
    return psnr, ssim


def crop_one_side(img, pct):
    """Crop pct% from one edge only, preserving most macro-tiles."""
    h, w = img.shape[:2]
    # Remove pct% of width from right and proportional height from bottom
    new_w = int(w * (1 - pct))
    new_h = int(h * (1 - pct))
    return img[:new_h, :new_w].copy()


def overlay_attack(img, pct, seed=0):
    """Overlay random colored rectangles covering ~pct% of the image."""
    result = img.copy()
    h, w = result.shape[:2]
    rng = np.random.default_rng(seed)
    covered = 0
    target = h * w * pct
    while covered < target:
        rw = rng.integers(30, max(31, w // 4))
        rh = rng.integers(30, max(31, h // 4))
        rx = rng.integers(0, max(1, w - rw))
        ry = rng.integers(0, max(1, h - rh))
        color = rng.integers(0, 256, 3).tolist()
        cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), color, -1)
        covered += rw * rh
    return result


def jpeg_compress(img, quality=50):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def run_test(name, image, public_key, embed_key, expect_verified=True):
    t0 = time.time()
    result = verify_p4(image, public_key, embed_key)
    dt = time.time() - t0

    verified = result.get("verified", False)
    detected = result.get("detected", False)
    confidence = result.get("confidence", 0.0)
    shards = result.get("shards_recovered", 0)
    tiles = result.get("tiles_located", 0)
    error = result.get("error", "")

    passed = verified == expect_verified
    status = "✅ PASS" if passed else "❌ FAIL"

    h, w = image.shape[:2]
    print(f"  {status}  {name}")
    print(f"         {w}×{h} | Det: {detected} | Ver: {verified} | "
          f"Conf: {confidence:.3f} | Tiles: {tiles} | Shards: {shards}/{K_SHARDS}")
    if error:
        print(f"         {error}")
    print(f"         {dt:.3f}s")
    print()
    return {"name": name, "pass": passed, "verified": verified, "time": dt}


def main():
    print("=" * 70)
    print("  MIST PHASE 4 — Spatial Attack Resistance Validation")
    print("=" * 70)
    print()

    priv, pub = generate_keys()
    key = b"phase4-test-key-2026"
    uid, iid = 123456789, 987654321

    results = []

    # ── 512×512 tests ─────────────────────────────────────────────────
    img512 = make_test_image(512, 512)
    print("Embedding 512×512...")
    t0 = time.time()
    wm512 = watermark_p4(img512, uid, iid, priv, key)
    t_embed_512 = time.time() - t0

    p512, s512 = psnr_ssim(img512, wm512)
    n_tiles_512 = (512 // MT_SIZE) ** 2
    print(f"  Time: {t_embed_512:.3f}s | PSNR: {p512:.2f} | "
          f"SSIM: {s512:.4f} | Tiles: {n_tiles_512}")
    print()

    print("─ Baseline (512×512) ─")
    results.append(run_test("512×512 clean", wm512, pub, key))

    print("─ Overlay attacks (512×512, 64 tiles, nsym=34) ─")
    for pct in [0.20, 0.30]:
        overlaid = overlay_attack(wm512, pct, seed=42)
        results.append(run_test(f"Overlay {int(pct*100)}%", overlaid, pub, key))

    print("─ Fragments (512×512) ─")
    for sz in [256, 384]:
        frag = wm512[:sz, :sz].copy()
        n_frag_tiles = (sz // MT_SIZE) ** 2
        expect = n_frag_tiles >= K_SHARDS
        results.append(run_test(
            f"Fragment {sz}×{sz} ({n_frag_tiles} tiles)",
            frag, pub, key, expect_verified=expect))

    print("─ JPEG (512×512) ─")
    results.append(run_test("JPEG Q70", jpeg_compress(wm512, 70), pub, key))

    # ── 1024×1024 tests (255 RS tiles, much more redundancy) ──────────
    img1k = make_test_image(1024, 1024, seed=99)
    print("\nEmbedding 1024×1024...")
    t0 = time.time()
    wm1k = watermark_p4(img1k, uid, iid, priv, key)
    t_embed_1k = time.time() - t0

    p1k, s1k = psnr_ssim(img1k, wm1k)
    n_tiles_1k = (1024 // MT_SIZE) ** 2
    print(f"  Time: {t_embed_1k:.3f}s | PSNR: {p1k:.2f} | "
          f"SSIM: {s1k:.4f} | Tiles: {n_tiles_1k}")
    print()

    print("─ Baseline (1024×1024) ─")
    results.append(run_test("1024×1024 clean", wm1k, pub, key))

    print("─ Crop attacks (1024×1024, 256 tiles) ─")
    for pct in [0.30, 0.50, 0.70]:
        cropped = crop_one_side(wm1k, pct)
        nt = (cropped.shape[0] // MT_SIZE) * (cropped.shape[1] // MT_SIZE)
        expect = nt >= K_SHARDS
        results.append(run_test(
            f"Crop {int(pct*100)}% ({cropped.shape[1]}×{cropped.shape[0]}, ~{nt} tiles)",
            cropped, pub, key, expect_verified=expect))

    print("─ Overlay attacks (1024×1024) ─")
    for pct in [0.30, 0.50]:
        overlaid = overlay_attack(wm1k, pct, seed=77)
        results.append(run_test(f"1k Overlay {int(pct*100)}%", overlaid, pub, key))

    print("─ Combined (1024×1024) ─")
    compressed = jpeg_compress(wm1k, 60)
    cropped_c = crop_one_side(compressed, 0.30)
    results.append(run_test("1k JPEG Q60 + Crop 30%", cropped_c, pub, key))

    # ── False positive ────────────────────────────────────────────────
    print("─ False Positive ─")
    n_fp, n_tests = 0, 10
    for seed in range(n_tests):
        clean = make_test_image(512, 512, seed=2000 + seed)
        r = verify_p4(clean, pub, key)
        if r.get("verified", False):
            n_fp += 1
    fpr = n_fp / n_tests * 100
    fp_status = "✅ PASS" if fpr < 5.0 else "❌ FAIL"
    print(f"  {fp_status}  FPR: {fpr:.1f}% ({n_fp}/{n_tests})")
    print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 70)
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")
    print(f"  FPR: {fpr:.1f}%  |  512 PSNR: {p512:.2f}  |  512 SSIM: {s512:.4f}")
    print(f"  Embed: 512={t_embed_512:.3f}s  1024={t_embed_1k:.3f}s")
    ok = passed >= total * 0.7 and fpr < 5.0 and p512 >= 36.0 and s512 >= 0.96
    print(f"  {'✅ PHASE 4 OK' if ok else '⚠️  SOME TARGETS MISSED'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
