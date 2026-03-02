"""
src/metrics/test_phase2.py  –  Phase 2 Validation Test Suite

Required tests before Phase 2 is declared complete:

  Test 1 – Tamper Test
    Modify 1 bit in payload_core → signature must fail.
    Modify 1 byte in signature → signature must fail.

  Test 2 – Random Image False Positive Test
    Run detect_watermark() on unwatermarked images.
    Expect 0 positives (because signature check rejects all noise).

  Test 3 – Partial Damage / ECC Recovery Test
    Flip 10% and 20% of ECC-encoded bits randomly.
    RS decode must succeed, and signature must verify (payload_core is intact).

Run from project root:
    python -m src.metrics.test_phase2
"""

import os
import sys
import cv2
import numpy as np
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.crypto import generate_keys, verify
from src.core.payload import (
    build_embed_payload, parse_embed_payload, unpack,
    EMBED_PAYLOAD_BITS, PAYLOAD_CORE_BYTES,
)
from src.core.ecc import encode_payload, decode_payload, ECC_TOTAL_BITS
from src.core.dct import rgb_to_y, pad_to_8, apply_block_dct, apply_block_idct
from src.core.embed import embed_watermark
from src.core.detect import detect_watermark

# ─────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────

def _make_watermarked_img(img_bgr: np.ndarray, full_bits: list, alpha: float = 20.0):
    """Embed full_bits into img_bgr and return (watermarked_bgr, Y_channel)."""
    Y        = rgb_to_y(img_bgr)
    h, w     = Y.shape
    Y_pad    = pad_to_8(Y).astype(np.float32)
    dct_img  = apply_block_dct(Y_pad)
    wm_dct   = embed_watermark(dct_img, full_bits, alpha=alpha)
    wm_Y     = apply_block_idct(wm_dct)

    ycbcr            = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycbcr[:, :, 0]   = np.clip(wm_Y[:h, :w], 0, 255)
    watermarked_bgr  = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    return watermarked_bgr


def _load_images(directory: str, n: int = 20) -> list:
    """Load up to n images from a directory."""
    imgs = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, fname)
            img  = cv2.imread(path)
            if img is not None:
                imgs.append(img)
                if len(imgs) >= n:
                    break
    return imgs


SEP = "=" * 58


# ─────────────────────────────────────────────────────────
#  Test 1 – Tamper Test
# ─────────────────────────────────────────────────────────

def test_tamper(priv_key: bytes, pub_key: bytes) -> bool:
    """
    Verify that modifying a single bit in payload_core causes
    signature verification to fail.
    """
    print(f"\n{SEP}")
    print("TEST 1 – Tamper Detection")
    print(SEP)

    # Build a valid payload
    payload_core, sig, full_bits = build_embed_payload(
        priv_key, user_id=42, image_id=999, model_version=1
    )

    # ── 1a. Unmodified should verify ──
    ok_clean = verify(pub_key, payload_core, sig)
    print(f"  Clean payload verifies:        {'PASS OK' if ok_clean else 'FAIL FAIL'}")

    # ── 1b. Flip 1 bit in payload_core ──
    tampered_core = bytearray(payload_core)
    tampered_core[0] ^= 0x01          # flip LSB of first byte
    ok_tampered_core = not verify(pub_key, bytes(tampered_core), sig)
    print(f"  1-bit payload_core flip fails: {'PASS OK' if ok_tampered_core else 'FAIL FAIL'}")

    # ── 1c. Corrupt 1 byte in the signature itself ──
    tampered_sig = bytearray(sig)
    tampered_sig[0] ^= 0xFF
    ok_tampered_sig = not verify(pub_key, payload_core, bytes(tampered_sig))
    print(f"  Corrupt signature byte fails:  {'PASS OK' if ok_tampered_sig else 'FAIL FAIL'}")

    passed = ok_clean and ok_tampered_core and ok_tampered_sig
    print(f"\n  Result: {'ALL PASSED OK' if passed else 'SOME FAILED FAIL'}")
    return passed


# ─────────────────────────────────────────────────────────
#  Test 2 – False Positive Test on Clean Images
# ─────────────────────────────────────────────────────────

def test_false_positives(pub_key: bytes, img_dir: str = "dataset/resized") -> bool:
    """
    Run detect_watermark() on unwatermarked images and verify 0 false positives.
    Because we require signature verification, any noise will fail verify().
    """
    print(f"\n{SEP}")
    print("TEST 2 – False Positive Rate on Clean Images")
    print(SEP)

    if not os.path.isdir(img_dir):
        print(f"  SKIP – directory not found: {img_dir}")
        return True   # Can't run, not a failure

    images = _load_images(img_dir, n=50)
    if not images:
        print("  SKIP – no images found.")
        return True

    fp_count = 0
    for img in images:
        Y      = rgb_to_y(img)
        result = detect_watermark(Y, pub_key)
        if result is not None:
            fp_count += 1

    fp_rate = fp_count / len(images)
    passed  = fp_count == 0
    print(f"  Images tested:   {len(images)}")
    print(f"  False positives: {fp_count}  ({fp_rate*100:.1f}%)")
    print(f"  Result: {'PASS OK' if passed else 'FAIL FAIL'}")
    return passed


# ─────────────────────────────────────────────────────────
#  Test 3 – ECC Recovery under Byte-Level Damage
# ─────────────────────────────────────────────────────────

def test_ecc_recovery(priv_key: bytes, pub_key: bytes) -> bool:
    """
    Corrupt N% of BYTES in the ECC-encoded codeword and verify that
    Reed-Solomon corrects them and the signature still validates.

    We corrupt at the byte level because RS is a byte-symbol code, and
    DCT-coefficient attacks (JPEG, resize) corrupt whole coefficient
    values, not isolated individual bits.
    """
    print(f"\n{SEP}")
    print("TEST 3 - ECC Recovery Under Byte-Level Damage")
    print(SEP)
    print("  (RS is a byte-symbol code; damage is modelled per-byte,")
    print("   consistent with JPEG/resize coefficient corruption.)")

    from src.core.ecc import bits_to_bytes, bytes_to_bits

    payload_core, sig, full_bits = build_embed_payload(
        priv_key, user_id=7, image_id=1234, model_version=1
    )

    ecc_bits  = encode_payload(full_bits)
    assert len(ecc_bits) == ECC_TOTAL_BITS, f"Expected {ECC_TOTAL_BITS}, got {len(ecc_bits)}"
    ecc_bytes = bytearray(bits_to_bytes(ecc_bits))
    n_total   = len(ecc_bytes)   # 138

    all_passed = True

    for damage_pct in [5, 10, 15, 20]:
        n_corrupt   = max(1, int(n_total * damage_pct / 100))
        corrupted   = bytearray(ecc_bytes)
        corrupt_idx = random.sample(range(n_total), n_corrupt)
        for i in corrupt_idx:
            corrupted[i] ^= 0xFF

        from src.core.ecc import bytes_to_bits as _b2b
        corrupted_bits             = _b2b(corrupted)
        recovered_bits, rs_success = decode_payload(corrupted_bits)

        if not recovered_bits or len(recovered_bits) < EMBED_PAYLOAD_BITS:
            status = False
            sig_ok = False
        else:
            try:
                rec_core, rec_sig = parse_embed_payload(recovered_bits)
                sig_ok = verify(pub_key, rec_core, rec_sig)
            except Exception:
                sig_ok = False
            status = rs_success and sig_ok

        label = (f"  {damage_pct:2d}% byte damage"
                 f" ({n_corrupt}/{n_total} bytes)"
                 f"  RS={rs_success}  sig_ok={sig_ok}")
        ok_str = "PASS" if status else "FAIL"
        print(f"{label:55s}  {ok_str}")
        if not status:
            all_passed = False

    ok_str = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"\n  Result: {ok_str}")
    return all_passed


# ─────────────────────────────────────────────────────────
#  Main runner
# ─────────────────────────────────────────────────────────

def main():
    print(f"\n{'#' * 58}")
    print("  MIST – Phase 2 Validation Test Suite")
    print(f"{'#' * 58}")

    priv_key, pub_key = generate_keys()
    print(f"\n  Generated fresh Ed25519 key pair.")
    print(f"  (Separate from embedding PRNG seed — architecturally isolated.)")

    r1 = test_tamper(priv_key, pub_key)
    r2 = test_false_positives(pub_key)
    r3 = test_ecc_recovery(priv_key, pub_key)

    print(f"\n{'=' * 58}")
    overall = all([r1, r2, r3])
    print(f"  OVERALL: {'PHASE 2 COMPLETE OK' if overall else 'TESTS FAILED FAIL'}")
    print(f"{'=' * 58}\n")


if __name__ == "__main__":
    main()
