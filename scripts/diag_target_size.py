"""Check impact of target size on CRC for combined rotation+scale."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2, numpy as np
from src.core.crypto import generate_keys
from src.core.mist import watermark_p5
from src.attacks.geometric import rotate, scale
from src.core.wm_engine_p5 import (
    _score_candidate, _undo_rotation, _undo_scale_to_size, MT_SIZE,
)

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

# First, check what size the rotated-only image is
rotated = rotate(wm, 5)
print(f"Original: {wm.shape}")
print(f"After rotate(5): {rotated.shape}")
attacked = scale(rotated, 0.8)
print(f"After scale(0.8): {attacked.shape}")
print(f"int(round(377/0.8)) = {int(round(377/0.8))}")
print()

# Now test undo_scale to different target sizes + rotation correction
print(f"{'Target':>8} {'CRC':>5} {'Total':>6}  Notes")
print("-" * 50)
for tw in range(466, 480):
    corrected = _undo_scale_to_size(attacked, tw, tw)
    corrected = _undo_rotation(corrected, 5.0)
    crc, total = _score_candidate(corrected, key)
    marker = ""
    if tw == rotated.shape[1]:
        marker = " <<< original rotated size"
    elif tw == int(round(377/0.8)):
        marker = " <<< round(377/0.8)"
    print(f"{tw:>8} {crc:>5} {total:>6}{marker}")
