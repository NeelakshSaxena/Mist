"""
scripts/validate_phase5.py — Phase 5 Geometry-Invariant Detection Benchmark

Validates:
  1. Rotation survival (±5°, ±10°, ±15°)
  2. Scale survival (0.6×, 0.8×, 1.3×, 1.5×)
  3. Rotation + Scale combined
  4. Crop + Rotation combined
  5. Forensic report generation
  6. False positive rate
  7. ROC curve data

Usage:
    python -m scripts.validate_phase5
"""

import os
import sys
import time

# Fix Windows console encoding for unicode output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.crypto   import generate_keys
from src.core.mist     import watermark_p5, verify_p5, forensic_report, ForensicReport
from src.core.wm_engine_p5 import detect_p5, estimate_geometry
from src.core.wm_engine_p4 import K_SHARDS, MT_SIZE
from src.core.forensic import (
    generate_roc_data, calibrate_confidence, ForensicReport as FR,
)
from src.core.p5_profiler import P5Profile
from src.attacks.geometric import (
    rotate, scale, crop_and_resize, perspective_warp,
)


def make_test_image(h: int = 512, w: int = 512, seed: int = 42) -> np.ndarray:
    """Generate a realistic-looking test image with gradients and texture."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = ((y + x) / 2).astype(np.uint8)
    noise = rng.integers(0, 40, (h, w), dtype=np.uint8)
    gray = np.clip(
        base.astype(np.int16) + noise.astype(np.int16), 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def psnr_ssim(original, modified):
    """Compute PSNR and SSIM between two images."""
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


def run_test(name, image, public_key, embed_key, expect_verified=True,
             expect_detected=True, use_p5=True):
    """Run a single detection test and report results."""
    with P5Profile() as prof:
        t0 = time.time()
        if use_p5:
            result = verify_p5(image, public_key, embed_key)
        else:
            from src.core.mist import verify_p4
            result = verify_p4(image, public_key, embed_key)
        dt = time.time() - t0

    verified = result.get("verified", False)
    detected = result.get("detected", False)
    confidence = result.get("confidence", 0.0)
    shards = result.get("shards_recovered", 0)
    tiles = result.get("tiles_located", 0)
    error = result.get("error", "")
    geo = result.get("geometry", {})

    if expect_verified:
        passed = verified
    else:
        passed = detected == expect_detected

    status = "✅ PASS" if passed else "❌ FAIL"

    h, w = image.shape[:2]
    geo_info = ""
    if geo and (abs(geo.get("angle_deg", 0)) > 0.3
                or abs(geo.get("scale_factor", 1) - 1.0) > 0.02):
        geo_info = (f" | Geo: {geo['angle_deg']:.1f}° × "
                    f"{geo['scale_factor']:.2f}× ({geo.get('method', '')})")

    print(f"  {status}  {name}")
    print(f"         {w}×{h} | Det: {detected} | Ver: {verified} | "
          f"Conf: {confidence:.3f} | Tiles: {tiles} | Shards: {shards}/{K_SHARDS}"
          f"{geo_info}")
    if error:
        print(f"         {error}")
    print(f"         {dt:.3f}s | P4 calls: {prof.p4_call_count} "
          f"({prof.p4_total_time:.1f}s) | Candidates: "
          f"{prof.candidates_generated}→{prof.candidates_after_clustering}→"
          f"{prof.candidates_evaluated}")
    print()

    return {"name": name, "pass": passed, "verified": verified,
            "detected": detected, "confidence": confidence, "time": dt}


def run_forensic_test(name, image, public_key, embed_key):
    """Run forensic_report() and display the summary."""
    print(f"  📊 Forensic: {name}")
    t0 = time.time()
    report = forensic_report(image, public_key, embed_key)
    dt = time.time() - t0
    print(report.summary())
    print(f"  Analysis time: {dt:.3f}s")
    print()
    return report


def main():
    print("=" * 70)
    print("  MIST PHASE 5 — Geometry-Invariant Detection + Forensic Engine")
    print("=" * 70)
    print()

    priv, pub = generate_keys()
    key = b"phase5-test-key-2026"
    uid, iid = 123456789, 987654321

    results = []

    # ══════════════════════════════════════════════════════════════════════
    #  512×512 Base Image
    # ══════════════════════════════════════════════════════════════════════
    img512 = make_test_image(512, 512)
    print("Embedding 512×512...")
    t0 = time.time()
    wm512 = watermark_p5(img512, uid, iid, priv, key)
    t_embed = time.time() - t0

    p_val, s_val = psnr_ssim(img512, wm512)
    print(f"  Time: {t_embed:.3f}s | PSNR: {p_val:.2f} dB | SSIM: {s_val:.4f}")
    print()

    # ── Baseline ──────────────────────────────────────────────────────
    print("─ Baseline (no attack) ─")
    results.append(run_test("Clean 512×512", wm512, pub, key))

    # ══════════════════════════════════════════════════════════════════════
    #  ROTATION TESTS
    # ══════════════════════════════════════════════════════════════════════
    print("─ Rotation Attacks ─")
    for angle in [2, 5, 10, 15]:
        for sign in [1, -1]:
            a = angle * sign
            rotated = rotate(wm512, a)
            results.append(run_test(
                f"Rotation {a:+d}°", rotated, pub, key,
                expect_verified=True,
            ))

    # ══════════════════════════════════════════════════════════════════════
    #  SCALE TESTS
    # ══════════════════════════════════════════════════════════════════════
    print("─ Scale Attacks ─")
    for factor in [0.6, 0.8, 1.2, 1.5]:
        scaled = scale(wm512, factor)
        expect_verified = factor >= 0.7  # Very small images lose too many tiles
        results.append(run_test(
            f"Scale {factor:.1f}×", scaled, pub, key,
            expect_verified=expect_verified,
        ))

    # ══════════════════════════════════════════════════════════════════════
    #  ROTATION + SCALE COMBINED
    # ══════════════════════════════════════════════════════════════════════
    print("─ Combined Rotation + Scale ─")
    for angle, factor in [(5, 0.8), (10, 1.3), (-8, 0.9)]:
        attacked = rotate(wm512, angle)
        attacked = scale(attacked, factor)
        results.append(run_test(
            f"Rot {angle:+d}° + Scale {factor}×", attacked, pub, key,
            expect_verified=False,
            expect_detected=True,
        ))

    # ══════════════════════════════════════════════════════════════════════
    #  CROP + ROTATION COMBINED
    # ══════════════════════════════════════════════════════════════════════
    print("─ Crop + Rotation ─")
    for angle, crop_frac in [(5, 0.8), (10, 0.7)]:
        attacked = rotate(wm512, angle)
        attacked = crop_and_resize(attacked, crop_frac, seed=42)
        results.append(run_test(
            f"Rot {angle:+d}° + Crop {int((1-crop_frac)*100)}%",
            attacked, pub, key,
            expect_verified=False,
            expect_detected=True,
        ))

    # ══════════════════════════════════════════════════════════════════════
    #  1024×1024 TESTS (more redundancy)
    # ══════════════════════════════════════════════════════════════════════
    img1k = make_test_image(1024, 1024, seed=99)
    print("\nEmbedding 1024×1024...")
    t0 = time.time()
    wm1k = watermark_p5(img1k, uid, iid, priv, key)
    t_embed_1k = time.time() - t0

    p1k, s1k = psnr_ssim(img1k, wm1k)
    print(f"  Time: {t_embed_1k:.3f}s | PSNR: {p1k:.2f} dB | SSIM: {s1k:.4f}")
    print()

    print("─ Rotation (1024×1024) ─")
    for angle in [5, 10, 15]:
        rotated = rotate(wm1k, angle)
        results.append(run_test(
            f"1k Rotation +{angle}°", rotated, pub, key,
        ))

    print("─ Scale (1024×1024) ─")
    for factor in [0.7, 1.4]:
        scaled = scale(wm1k, factor)
        results.append(run_test(
            f"1k Scale {factor}×", scaled, pub, key,
        ))

    # ══════════════════════════════════════════════════════════════════════
    #  FORENSIC REPORT
    # ══════════════════════════════════════════════════════════════════════
    print("─ Forensic Reports ─")
    run_forensic_test("Clean watermarked", wm512, pub, key)
    rotated_15 = rotate(wm512, 15)
    run_forensic_test("After 15° rotation", rotated_15, pub, key)

    # Forensic on clean (no watermark)
    clean_report = run_forensic_test("Clean (no watermark)", img512, pub, key)
    fp_forensic = clean_report.watermark_detected
    print(f"  Clean forensic detected: {fp_forensic} "
          f"(confidence: {clean_report.confidence_pct:.2f}%)")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  FALSE POSITIVE RATE
    # ══════════════════════════════════════════════════════════════════════
    print("─ False Positive Rate ─")
    n_fp = 0
    n_tests = 10
    fp_scores = []
    for seed in range(n_tests):
        clean = make_test_image(512, 512, seed=2000 + seed)
        r = verify_p5(clean, pub, key)
        if r.get("verified", False):
            n_fp += 1
        fp_scores.append(r.get("confidence", 0.0))

    fpr = n_fp / n_tests * 100
    fp_status = "✅ PASS" if fpr < 5.0 else "❌ FAIL"
    print(f"  {fp_status}  FPR: {fpr:.1f}% ({n_fp}/{n_tests})")
    print(f"         Mean FP confidence: {np.mean(fp_scores):.4f}")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  ROC CURVE DATA
    # ══════════════════════════════════════════════════════════════════════
    print("─ ROC Curve Generation ─")
    # Positive scores: watermarked images under various attacks
    pos_scores = []
    for angle in [0, 5, 10, -5]:
        img = rotate(wm512, angle) if angle != 0 else wm512
        det = detect_p5(img, key)
        pos_scores.append(det["confidence"])

    # Negative scores: clean images
    neg_scores = []
    for seed in range(10):
        clean = make_test_image(512, 512, seed=3000 + seed)
        det = detect_p5(clean, key)
        neg_scores.append(det["confidence"])

    roc = generate_roc_data(pos_scores, neg_scores)
    print(f"  AUC: {roc.auc:.4f}")
    print(f"  Pos: {roc.n_positive} samples | Neg: {roc.n_negative} samples")
    roc_status = "✅ PASS" if roc.auc > 0.85 else "❌ FAIL"
    print(f"  {roc_status}  (target AUC > 0.85)")
    print()

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    detected_count = sum(1 for r in results if r["detected"])
    print(f"  Tests passed:    {passed}/{total}")
    print(f"  Tests detected:  {detected_count}/{total}")
    print(f"  FPR:             {fpr:.1f}%")
    print(f"  ROC AUC:         {roc.auc:.4f}")
    print(f"  PSNR (512):      {p_val:.2f} dB")
    print(f"  SSIM (512):      {s_val:.4f}")
    print(f"  Embed: 512={t_embed:.3f}s  1024={t_embed_1k:.3f}s")

    # Per-category summary
    rotation_tests = [r for r in results if "Rotation" in r["name"]
                      and "Crop" not in r["name"] and "Scale" not in r["name"]]
    scale_tests = [r for r in results if "Scale" in r["name"]
                   and "Rotation" not in r["name"]]
    combo_tests = [r for r in results if "+" in r["name"]]

    if rotation_tests:
        rot_pass = sum(1 for r in rotation_tests if r["detected"])
        print(f"  Rotation survived:  {rot_pass}/{len(rotation_tests)}")
    if scale_tests:
        scl_pass = sum(1 for r in scale_tests if r["detected"])
        print(f"  Scale survived:     {scl_pass}/{len(scale_tests)}")
    if combo_tests:
        cmb_pass = sum(1 for r in combo_tests if r["detected"])
        print(f"  Combined survived:  {cmb_pass}/{len(combo_tests)}")

    ok = (passed >= total * 0.5
          and fpr < 5.0
          and p_val >= 36.0
          and s_val >= 0.96
          and roc.auc > 0.85)
    print()
    print(f"  {'✅ PHASE 5 OK' if ok else '⚠️  SOME TARGETS MISSED'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
