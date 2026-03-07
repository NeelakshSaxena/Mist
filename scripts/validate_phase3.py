#!/usr/bin/env python
"""
scripts/validate_phase3.py  –  Phase 3 Diffusion-Resistant Watermark Validation

Tests the full Phase 3 pipeline against diffusion-simulated attacks and compares
directly against Phase 2 survival rates.

Tests
-----
1.  Round-trip clean image                   → detected=True,  verified=True
2.  Diffusion attack, strength=0.3 (mild)    → detected=True
3.  Diffusion attack, strength=0.5 (moderate)→ detected=True  (or BER ≤ 40%)
4.  Diffusion attack, strength=0.7 (hard)    → survival reported (no hard pass)
5.  JPEG Q50 + diffusion str=0.4 combo       → detected=True
6.  FPR: plain unwatermarked image           → verified=False
7.  Phase 2 vs Phase 3 survival (str=0.5)    → P3 raw_score > P2 raw_score
8.  PSNR quality check                       → PSNR ≥ 38 dB
9.  Scale score breakdown                    → 32×32 > 8×8 (coarser is more robust)

Usage
-----
    cd g:\\Projects\\Mist
    mist_env\\Scripts\\python scripts\\validate_phase3.py [path/to/image.jpg]
"""

import os
import sys
import hashlib

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.mist        import watermark, verify, watermark_p3, verify_p3
from src.core.crypto      import generate_keys
from src.core.ecc         import ECC_TOTAL_BITS
from src.core.wm_engine   import detect   as detect_p2
from src.core.wm_engine_p3 import detect_p3
from src.attacks.diffusion import attack_diffusion_sim
from src.attacks.light     import attack_jpeg


# ── Helpers ───────────────────────────────────────────────────────────────────

SEP  = "─" * 72
PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
INFO = "  ℹ️ "

def pf(ok: bool) -> str:
    return PASS if ok else FAIL


def load_test_image(path: str | None = None) -> np.ndarray:
    if path and os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            return img
    resized_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "resized")
    if os.path.isdir(resized_dir):
        for f in sorted(os.listdir(resized_dir)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(resized_dir, f))
                if img is not None:
                    return img
    # Synthetic fallback: natural-looking gradient with texture
    rng  = np.random.default_rng(42)
    h, w = 512, 512
    base = np.stack(np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h)), axis=2)
    base = np.concatenate([base, 255 - base[:, :, :1]], axis=2).astype(np.float32)
    noise = rng.normal(0, 20, (h, w, 3)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    return cv2.PSNR(original, processed)


# ── Validation runner ─────────────────────────────────────────────────────────

def run_validation(image_path: str | None = None) -> None:
    print()
    print("━" * 72)
    print("  🌫  MIST — Phase 3 Validation Suite")
    print("  Diffusion-Resistant Watermarking")
    print("━" * 72)

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("\n[setup] Generating keys and loading image…")
    private_key, public_key = generate_keys()
    _, wrong_pub            = generate_keys()
    embed_key  = hashlib.sha256(b"mist-phase3-embed-key").digest()

    USER_ID   = 0xDEADBEEF_CAFEBABE
    IMAGE_ID  = 0x0102030405060708
    MODEL_VER = 1

    img = load_test_image(image_path)
    print(f"  Image : {img.shape[1]}×{img.shape[0]} px")

    # Embed P3 watermark once; used across most tests
    print("\n  Embedding Phase 3 watermark (multi-scale + harmonic)…")
    wm3 = watermark_p3(img, USER_ID, IMAGE_ID, private_key, embed_key,
                       model_version=MODEL_VER)
    print("  Done.")

    all_pass = True

    # ── Test 1: Round-trip ────────────────────────────────────────────────────
    print(f"\n[1] Round-trip — clean watermarked image")
    print(SEP)
    res1 = verify_p3(wm3, public_key, embed_key)
    t1a  = res1["detected"]
    t1b  = res1["verified"]
    t1c  = res1["ecc_success"]
    p1   = res1.get("payload") or {}
    t1d  = (p1.get("user_id") == USER_ID and
            p1.get("image_id") == IMAGE_ID and
            p1.get("model_version") == MODEL_VER)
    print(f"  detected                           {pf(t1a)}")
    print(f"  verified (signature valid)         {pf(t1b)}")
    print(f"  ecc_success                        {pf(t1c)}")
    print(f"  payload fields match               {pf(t1d)}")
    ss = res1.get("scale_scores", {})
    print(f"  scale scores:  8×8={ss.get(8,0):.3f}  "
          f"16×16={ss.get(16,0):.3f}  32×32={ss.get(32,0):.3f}")
    print(f"  harmonic score: {res1.get('harmonic_score', 0):.3f}")
    t1 = t1a and t1b and t1c and t1d
    all_pass &= t1

    # ── Test 2: Mild diffusion (str=0.3) ──────────────────────────────────────
    print(f"\n[2] Diffusion attack, strength=0.3 (mild)")
    print(SEP)
    att2 = attack_diffusion_sim(wm3, strength=0.3, seed=1)
    res2 = verify_p3(att2, public_key, embed_key)
    t2 = res2["detected"]
    print(f"  detected after mild diffusion      {pf(t2)}")
    print(f"  verified: {res2['verified']}  |  ecc_ok: {res2['ecc_success']}")
    ss2 = res2.get("scale_scores", {})
    print(f"  scale scores:  8×8={ss2.get(8,0):.3f}  "
          f"16×16={ss2.get(16,0):.3f}  32×32={ss2.get(32,0):.3f}")
    all_pass &= t2

    # ── Test 3: Moderate diffusion (str=0.5) ──────────────────────────────────
    print(f"\n[3] Diffusion attack, strength=0.5 (moderate)")
    print(SEP)
    att3 = attack_diffusion_sim(wm3, strength=0.5, seed=2)
    res3 = verify_p3(att3, public_key, embed_key)
    det3 = detect_p3(att3, embed_key)
    t3   = res3["detected"]
    print(f"  detected after moderate diffusion  {pf(t3)}")
    print(f"  verified: {res3['verified']}  |  ecc_ok: {res3['ecc_success']}")
    ss3 = det3.get("scale_scores", {})
    print(f"  scale scores:  8×8={ss3.get(8,0):.3f}  "
          f"16×16={ss3.get(16,0):.3f}  32×32={ss3.get(32,0):.3f}")
    print(f"  harmonic_score: {det3.get('harmonic_score', 0):.3f}")
    all_pass &= t3

    # ── Test 4: Aggressive diffusion (str=0.7) — informational ───────────────
    print(f"\n[4] Diffusion attack, strength=0.7 (aggressive)  [informational]")
    print(SEP)
    att4 = attack_diffusion_sim(wm3, strength=0.7, seed=3)
    det4 = detect_p3(att4, embed_key)
    ss4  = det4.get("scale_scores", {})
    print(f"  raw_score: {det4['raw_score']:.4f}  |  "
          f"confidence: {det4['confidence']:.4f}  |  detected: {det4['detected']}")
    print(f"  scale scores:  8×8={ss4.get(8,0):.3f}  "
          f"16×16={ss4.get(16,0):.3f}  32×32={ss4.get(32,0):.3f}")
    print(f"  harmonic_score: {det4.get('harmonic_score', 0):.3f}")
    print(f"{INFO} No pass/fail at strength=0.7 (real SD img2img would test this)")

    # ── Test 5: JPEG Q50 + diffusion str=0.4 combo ───────────────────────────
    print(f"\n[5] JPEG Q50 + diffusion str=0.4 combo attack")
    print(SEP)
    att5a = attack_jpeg(wm3, quality=50)
    att5  = attack_diffusion_sim(att5a, strength=0.4, seed=4)
    det5  = detect_p3(att5, embed_key)
    t5    = det5["detected"]
    print(f"  detected after JPEG+diffusion      {pf(t5)}")
    ss5 = det5.get("scale_scores", {})
    print(f"  scale scores:  8×8={ss5.get(8,0):.3f}  "
          f"16×16={ss5.get(16,0):.3f}  32×32={ss5.get(32,0):.3f}")
    all_pass &= t5

    # ── Test 6: FPR — plain image ─────────────────────────────────────────────
    print(f"\n[6] FPR — plain unwatermarked image → not verified")
    print(SEP)
    res6 = verify_p3(img, public_key, embed_key)
    t6   = not res6["verified"]
    print(f"  verified=False on plain image      {pf(t6)}")
    print(f"  ecc_success: {res6['ecc_success']}  |  "
          f"error: {str(res6.get('error', 'none'))[:60]}")
    all_pass &= t6

    # ── Test 7: Phase 2 vs Phase 3 survival at str=0.5 ───────────────────────
    print(f"\n[7] Phase 2 vs Phase 3 survival comparison (diffusion str=0.5)")
    print(SEP)
    # Phase 2 watermark
    p2_embed_key = hashlib.sha256(b"mist-phase2-embed-key").digest()
    wm2    = watermark(img, USER_ID, IMAGE_ID, private_key, p2_embed_key,
                       model_version=MODEL_VER)
    att_p2 = attack_diffusion_sim(wm2, strength=0.5, seed=5)
    att_p3 = attack_diffusion_sim(wm3, strength=0.5, seed=5)

    score_p2 = detect_p2(att_p2, p2_embed_key)["raw_score"]
    score_p3 = detect_p3(att_p3, embed_key)["raw_score"]

    print(f"  Phase 2 raw_score after diffusion: {score_p2:.4f}")
    print(f"  Phase 3 raw_score after diffusion: {score_p3:.4f}")
    t7 = score_p3 > score_p2
    print(f"  P3 survives better than P2         {pf(t7)}")
    all_pass &= t7

    # ── Test 8: PSNR quality check ────────────────────────────────────────────
    print(f"\n[8] Perceptual quality check (PSNR)")
    print(SEP)
    psnr_val = psnr(img, wm3)
    t8 = psnr_val >= 38.0
    print(f"  PSNR: {psnr_val:.2f} dB  (target ≥ 38 dB)   {pf(t8)}")
    all_pass &= t8

    # ── Test 9: Scale score breakdown (32×32 should be the strongest) ─────────
    print(f"\n[9] Scale score breakdown — coarser scale should be more robust")
    print(SEP)
    # Run at strength=0.4 where 8×8 starts weakening but 32×32 holds
    att9 = attack_diffusion_sim(wm3, strength=0.4, seed=6)
    det9 = detect_p3(att9, embed_key)
    ss9  = det9.get("scale_scores", {})
    s8, s16, s32 = ss9.get(8, 0), ss9.get(16, 0), ss9.get(32, 0)
    print(f"  8×8  score: {s8:.4f}")
    print(f"  16×16 score: {s16:.4f}")
    print(f"  32×32 score: {s32:.4f}")
    t9 = s32 >= s8   # 32×32 ≥ 8×8 (coarser is more diffusion-robust)
    print(f"  32×32 score ≥ 8×8 score            {pf(t9)}")
    all_pass &= t9

    # ── Cleanup temp file ─────────────────────────────────────────────────────
    tmp = "_append_p3.py"
    if os.path.exists(tmp):
        os.remove(tmp)
        print(f"\n  [cleanup] Removed {tmp}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("━" * 72)
    verdict = ("✅  ALL CHECKS PASSED  —  Phase 3 SUCCESS"
               if all_pass else "❌  SOME CHECKS FAILED — review above")
    print(f"  {verdict}")
    print("━" * 72)
    print()


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_validation(img_path)
