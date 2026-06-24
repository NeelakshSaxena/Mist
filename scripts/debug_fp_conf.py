"""Targeted debug: trace detect_p5 confidence for clean images."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.core.wm_engine_p5 import detect_p5, _compute_confidence

key = b"phase5-test-key-2026"

def make_test_image(h=512, w=512, seed=42):
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = ((y + x) / 2).astype(np.uint8)
    noise = rng.integers(0, 40, (h, w), dtype=np.uint8)
    import cv2
    gray = np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

print("Testing detect_p5 for clean images (seeds 2000-2004):")
total = 0
for seed in range(2000, 2005):
    img = make_test_image(seed=seed)
    det = detect_p5(img, key)
    conf = det["confidence"]
    geo = det.get("geometry", {})
    method = geo.get("method", "none") if geo else "none"
    ic = "YES" if det.get("inner_codeword") else "NO"
    print(f"  seed={seed}: conf={conf:.4f}  geo={method}  inner={ic}  "
          f"tiles={det.get('tiles_located',0)}  shards={det.get('shards_recovered',0)}")
    total += conf

print(f"\n  Mean: {total/5:.4f}")

print("\nTesting _compute_confidence edge cases:")
# What if shard_crc_ratio=0.0 but shard_ratio is high?
for crc in [0.0, 0.005, 0.01, 0.10, 0.50, 1.0]:
    c = _compute_confidence(False, False, 1.0, crc, 0.0, 0.0)
    print(f"  crc_ratio={crc:.3f} -> conf={c:.4f}")
