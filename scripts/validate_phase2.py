#!/usr/bin/env python
"""
scripts/validate_phase2.py  –  Phase 2 Validation Suite

Validates the full Phase 2 pipeline:
    payload → sign → ECC encode → wm_engine.embed → wm_engine.detect
    → extract_bits → ECC decode → crypto.verify → payload.unpack

Tests
-----
1. Round-trip: payload fields recovered intact after embed/detect/verify
2. Tamper detection: flipping a payload byte → signature fails
3. ECC recovery: injecting byte errors → RS corrects, payload survives
4. Wrong embed key: detect() returns False
5. Wrong public key: signature verification fails
6. FPR: plain unwatermarked image → detected=False, verified=False
7. Bit capacity check: confirm image has enough blocks for 1184 bits

Usage
-----
    cd g:\\Projects\\Mist
    mist_env\\Scripts\\python scripts\\validate_phase2.py [path/to/image.jpg]
"""

import os
import sys
import random
import hashlib

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.mist    import watermark, verify
from src.core.crypto  import generate_keys
from src.core.ecc     import ECC_TOTAL_BITS, decode_payload
from src.core.payload import unpack, EMBED_PAYLOAD_BITS
from src.core.wm_engine import detect, extract_bits


# ── Helpers ───────────────────────────────────────────────────────────────────

SEP  = "─" * 70
PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"

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
    # Synthetic fallback
    rng = np.random.default_rng(42)
    base = np.stack(np.meshgrid(np.linspace(0, 255, 512), np.linspace(0, 255, 512)), axis=2)
    base = np.concatenate([base, 255 - base[:, :, :1]], axis=2).astype(np.float32)
    return np.clip(base + rng.normal(0, 20, (512, 512, 3)), 0, 255).astype(np.uint8)


# ── Validation runner ─────────────────────────────────────────────────────────

def run_validation(image_path: str | None = None) -> None:
    print()
    print("━" * 70)
    print("  🌫  MIST  —  Phase 2 Validation Suite")
    print("━" * 70)

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("\n[setup] Generating keys and loading image…")
    private_key, public_key = generate_keys()
    embed_key  = hashlib.sha256(b"mist-phase2-embed-key").digest()
    wrong_ekey = hashlib.sha256(b"wrong-embed-key").digest()
    _, wrong_pub = generate_keys()

    USER_ID    = 0xDEADBEEF_CAFEBABE
    IMAGE_ID   = 0x0102030405060708
    MODEL_VER  = 1

    img = load_test_image(image_path)
    print(f"  Image: {img.shape[1]}×{img.shape[0]} px")

    h, w    = img.shape[:2]
    bh, bw  = (h + 7) // 8, (w + 7) // 8
    n_blocks = bh * bw
    print(f"  Blocks: {bh}×{bw} = {n_blocks}  (need ≥ {ECC_TOTAL_BITS})")
    cap_ok = n_blocks >= ECC_TOTAL_BITS
    print(f"  Bit capacity          {pf(cap_ok)}")
    if not cap_ok:
        print(f"\n  ❌ Image too small for Phase 2 payload ({n_blocks} < {ECC_TOTAL_BITS}). Aborting.")
        return

    all_pass = True

    # ── Test 1: Round-trip ────────────────────────────────────────────────────
    print(f"\n[1] Round-trip payload recovery")
    print(SEP)
    wm = watermark(img, USER_ID, IMAGE_ID, private_key, embed_key,
                   model_version=MODEL_VER)
    res = verify(wm, public_key, embed_key)

    t1a = res["detected"]
    t1b = res["verified"]
    t1c = res["ecc_success"]
    p   = res.get("payload") or {}
    t1d = (p.get("user_id") == USER_ID and p.get("image_id") == IMAGE_ID
           and p.get("model_version") == MODEL_VER)

    print(f"  detected                         {pf(t1a)}")
    print(f"  verified (signature valid)       {pf(t1b)}")
    print(f"  ecc_success                      {pf(t1c)}")
    print(f"  payload fields match             {pf(t1d)}")
    if res.get("payload"):
        print(f"    user_id       = {p['user_id']:#018x}")
        print(f"    image_id      = {p['image_id']:#018x}")
        print(f"    model_version = {p['model_version']}")
    all_pass &= t1a and t1b and t1c and t1d

    # ── Test 2: Tamper detection ──────────────────────────────────────────────
    print(f"\n[2] Tamper detection — mutate decoded payload field")
    print(SEP)
    # Recover the full clean payload via ECC decode
    raw_bits2  = extract_bits(wm, embed_key, ECC_TOTAL_BITS)
    dec_bits2, _ = decode_payload(raw_bits2)
    from src.core.payload import parse_embed_payload
    pc2, sig2 = parse_embed_payload(dec_bits2)
    # Simulate attacker flipping one byte of user_id in payload_core
    # (bytes 0-7 = user_id in big-endian packed struct)
    tampered_pc = bytearray(pc2)
    tampered_pc[3] ^= 0xFF   # flip 8 bits inside user_id
    from src.core.crypto import verify as cv
    sig_valid = cv(public_key, bytes(tampered_pc), sig2)
    t2 = not sig_valid
    print(f"  Mutated user_id → signature fails   {pf(t2)}")
    all_pass &= t2


    # ── Test 3: ECC recovery ──────────────────────────────────────────────────
    print(f"\n[3] ECC recovery — inject 25 random byte-errors")
    print(SEP)
    raw_bits_ecc = extract_bits(wm, embed_key, ECC_TOTAL_BITS)
    rng_ecc = random.Random(99)
    # Corrupt 25 random byte positions (= 200 bit flips)
    byte_positions = rng_ecc.sample(range(ECC_TOTAL_BITS // 8), 25)
    for bp in byte_positions:
        for bit_i in range(bp * 8, bp * 8 + 8):
            raw_bits_ecc[bit_i] ^= 1
    dec_bits, ecc_ok_3 = decode_payload(raw_bits_ecc)
    t3a = ecc_ok_3
    try:
        pc3, sig3 = parse_embed_payload(dec_bits)
        t3b = crypto_verify_payload(public_key, pc3, sig3)
    except Exception:
        t3b = False
    print(f"  ECC decode succeeded             {pf(t3a)}")
    print(f"  Signature still valid            {pf(t3b)}")
    all_pass &= t3a and t3b

    # ── Test 4: Wrong embed key ───────────────────────────────────────────────
    print(f"\n[4] Wrong embed key → verification fails")
    print(SEP)
    res4 = verify(wm, public_key, wrong_ekey)
    # With wrong embed key, extracted bits are garbage → ECC decode fails or
    # signature is invalid.  verified must be False.
    t4   = not res4["verified"]
    print(f"  verified=False with wrong key    {pf(t4)}")
    print(f"  error: {res4.get('error', 'none')}")
    all_pass &= t4

    # ── Test 5: Wrong public key ──────────────────────────────────────────────
    print(f"\n[5] Wrong public key → signature fails")
    print(SEP)
    res5 = verify(wm, wrong_pub, embed_key)
    t5   = res5["detected"] and not res5["verified"]
    print(f"  detected=True, verified=False    {pf(t5)}")
    if res5.get("error"):
        print(f"  error: {res5['error']}")
    all_pass &= t5

    # ── Test 6: FPR — plain image ─────────────────────────────────────────────
    print(f"\n[6] FPR — plain unwatermarked image → not verified")
    print(SEP)
    res6 = verify(img, public_key, embed_key)
    # A plain image will have garbage bits → ECC will likely fail or signature fails
    t6   = not res6["verified"]
    print(f"  verified=False on plain image    {pf(t6)}")
    print(f"  ecc_success: {res6['ecc_success']},  error: {res6.get('error', 'none')[:60]}")
    all_pass &= t6

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("━" * 70)
    verdict = ("✅  ALL CHECKS PASSED  —  Phase 2 SUCCESS"
               if all_pass else "❌  SOME CHECKS FAILED — review above")
    print(f"  {verdict}")
    print("━" * 70)
    print()


# Helper used in test 3 to avoid import shadowing
def crypto_verify_payload(pub: bytes, core: bytes, sig: bytes) -> bool:
    from src.core.crypto import verify as cv
    return cv(pub, core, sig)


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_validation(img_path)
