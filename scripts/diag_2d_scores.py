"""Diagnostic: show _score_candidate at key 2D grid points for combined attack."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2, numpy as np
from src.core.crypto import generate_keys
from src.core.mist import watermark_p5
from src.attacks.geometric import rotate, scale
from src.core.wm_engine_p5 import _score_candidate, _undo_transform

# Build test image
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

# Combined attack: rotate +5, scale 0.8
attacked = scale(rotate(wm, 5), 0.8)
print(f"Attacked image: {attacked.shape}")

# Test key candidate corrections
test_points = [
    (0.0, 1.0, "identity"),
    (5.0, 1.0, "rot-only correct"),
    (0.0, 0.80, "scale-only correct"),
    (5.0, 0.80, "CORRECT combination"),
    (4.0, 0.80, "close: rot=4 scale=0.80"),
    (6.0, 0.80, "close: rot=6 scale=0.80"),
    (5.0, 0.75, "close: rot=5 scale=0.75"),
    (5.0, 0.85, "close: rot=5 scale=0.85"),
    (-20.0, 0.50, "wrong: -20 x 0.50"),
    (-4.0, 0.65, "wrong: -4 x 0.65"),
    (12.0, 1.50, "wrong: 12 x 1.50"),
]

print(f"\n{'Angle':>8} {'Scale':>6} {'CRC':>5} {'Total':>6}  Description")
print("-" * 60)
for angle, scale_f, desc in test_points:
    corrected = _undo_transform(attacked, angle, scale_f)
    crc, total = _score_candidate(corrected, key)
    marker = " <<<" if "CORRECT" in desc else ""
    print(f"{angle:8.1f} {scale_f:6.2f} {crc:5d} {total:6d}  {desc}{marker}")
