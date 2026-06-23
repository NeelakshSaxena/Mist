"""Diagnostic: Can tile extraction work AFTER geometric correction?"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
from src.core.crypto import generate_keys
from src.core.mist import watermark_p5
from src.core.wm_engine_p3 import _to_ycbcr, _harmonic_score
from src.core.wm_engine_p4 import (
    detect_p4, extract_shards_p4, _tile_anchor_bits,
    _extract_tile_bits, _parse_tile_bits, MT_SIZE, K_SHARDS,
    ANCHOR_BITS,
)
from src.attacks.geometric import rotate, scale

# Build test image + embed
rng = np.random.default_rng(42)
y = np.linspace(0, 255, 512, dtype=np.float32)[:, None]
x = np.linspace(0, 255, 512, dtype=np.float32)[None, :]
base = ((y + x) / 2).astype(np.uint8)
noise = rng.integers(0, 40, (512, 512), dtype=np.uint8)
gray = np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

priv, pub = generate_keys()
key = b"phase5-test-key-2026"
wm = watermark_p5(img, 123, 456, priv, key)
_, Y_orig = _to_ycbcr(wm)

print("=== DIAGNOSTIC: Tile extraction after geometric correction ===\n")

# Check: clean watermarked (no attack)
expected_anchor = _tile_anchor_bits(key)
print(f"Expected anchor: {expected_anchor}")
print(f"Original image: {wm.shape[:2]}")

def analyze_tiles(label, image_bgr):
    _, Y = _to_ycbcr(image_bgr)
    h, w = Y.shape
    mt_rows, mt_cols = h // MT_SIZE, w // MT_SIZE
    n_tiles = mt_rows * mt_cols

    anchor_matches = []
    crc_valid_count = 0
    crc_total = 0
    seen_indices = set()

    for tr in range(mt_rows):
        for tc in range(mt_cols):
            y0, x0 = tr * MT_SIZE, tc * MT_SIZE
            region = Y[y0:y0+MT_SIZE, x0:x0+MT_SIZE]
            bits = _extract_tile_bits(region, key)
            if len(bits) < ANCHOR_BITS:
                continue

            anchor_match = sum(a == b for a, b in zip(bits[:ANCHOR_BITS], expected_anchor)) / ANCHOR_BITS
            anchor_matches.append(anchor_match)

            parsed = _parse_tile_bits(bits, expected_anchor)
            if parsed is not None:
                crc_total += 1
                if parsed["crc_valid"]:
                    crc_valid_count += 1
                    if parsed["shard_idx"] not in seen_indices:
                        seen_indices.add(parsed["shard_idx"])

    harm = _harmonic_score(Y, key)
    print(f"\n--- {label} ---")
    print(f"  Size: {w}x{h} | Tiles: {n_tiles}")
    print(f"  Anchor match: mean={np.mean(anchor_matches):.3f} min={np.min(anchor_matches):.3f} max={np.max(anchor_matches):.3f}")
    print(f"  Anchor pass (>0.625): {sum(1 for m in anchor_matches if m >= 0.625)}/{len(anchor_matches)}")
    print(f"  CRC valid: {crc_valid_count}/{crc_total}")
    print(f"  Unique CRC shards: {len(seen_indices)}")
    print(f"  Harmonic score: {harm:.4f}")

# 1. Clean watermarked
analyze_tiles("Clean watermarked (no attack)", wm)

# 2. Rotate +5 and PERFECTLY undo
rot5 = rotate(wm, 5)
print(f"\nRotated +5° image size: {rot5.shape[:2]}")

# Undo rotation manually
h, w = rot5.shape[:2]
cx, cy = w/2.0, h/2.0
M = cv2.getRotationMatrix2D((cx, cy), -5.0, 1.0)  # rotate back by -5°
unrot5 = cv2.warpAffine(rot5, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
analyze_tiles("Rot+5 then undo-rot (-5°)", unrot5)

# 3. Try with grid alignment offsets
_, Y_unrot = _to_ycbcr(unrot5)
print("\n--- Grid offset scan for undo-rot ---")
for dy in range(0, 8, 2):
    for dx in range(0, 8, 2):
        Y2 = Y_unrot[dy:, dx:]
        h2, w2 = Y2.shape
        crc_count = 0
        total = 0
        for tr in range(h2 // MT_SIZE):
            for tc in range(w2 // MT_SIZE):
                y0, x0 = tr * MT_SIZE, tc * MT_SIZE
                region = Y2[y0:y0+MT_SIZE, x0:x0+MT_SIZE]
                bits = _extract_tile_bits(region, key)
                parsed = _parse_tile_bits(bits, expected_anchor)
                if parsed is not None:
                    total += 1
                    if parsed["crc_valid"]:
                        crc_count += 1
        if total > 0:
            print(f"  offset({dx},{dy}): anchor_pass={total} crc_valid={crc_count}")

# 4. Scale 0.8 and undo
sc08 = scale(wm, 0.8)
inv_f = 1.0 / 0.8
new_w = int(sc08.shape[1] * inv_f)
new_h = int(sc08.shape[0] * inv_f)
unsc08 = cv2.resize(sc08, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
analyze_tiles("Scale 0.8x then undo (1.25x)", unsc08)

# 5. Full P4 detection on corrected images
print("\n=== Full P4 detection ===")
for label, img_bgr in [("Undo rot5", unrot5), ("Undo scale0.8", unsc08)]:
    t0 = time.time()
    result = detect_p4(img_bgr, key)
    dt = time.time() - t0
    print(f"  {label}: detected={result['detected']} shards={result['shards_recovered']}/{K_SHARDS} "
          f"inner={'YES' if result.get('inner_codeword') else 'NO'} err={result.get('error','')} ({dt:.1f}s)")

# 6. Scale 1.2 and undo
sc12 = scale(wm, 1.2)
inv_f = 1.0 / 1.2
new_w = int(sc12.shape[1] * inv_f)
new_h = int(sc12.shape[0] * inv_f)
unsc12 = cv2.resize(sc12, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
analyze_tiles("Scale 1.2x then undo (0.833x)", unsc12)
result = detect_p4(unsc12, key)
print(f"  Undo scale1.2: detected={result['detected']} shards={result['shards_recovered']}/{K_SHARDS} "
      f"inner={'YES' if result.get('inner_codeword') else 'NO'} err={result.get('error','')}")
