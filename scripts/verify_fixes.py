"""Verify the two critical fixes:
1. CRC-ratio guard suppresses RS false-decodes on clean images
2. signature_verified=False in detect_p5
3. WM 0deg gets conf=0.40 (RS+shard, no sig), clean seed=2003 gets conf≈0
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from src.core.wm_engine_p5 import detect_p5
from src.core.crypto import generate_keys
from src.core.mist import watermark_p5, verify_p5
from src.attacks.geometric import rotate

key = b"phase5-test-key-2026"

def make_test_image(h=512, w=512, seed=42):
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = ((y + x) / 2).astype(np.uint8)
    noise = rng.integers(0, 40, (h, w), dtype=np.uint8)
    gray = np.clip(base.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

priv, pub = generate_keys()
img512 = make_test_image(512, 512, seed=42)
wm512 = watermark_p5(img512, 123456789, 987654321, priv, key)

print("=" * 60)
print("detect_p5 confidence for clean images (should be near 0):")
for seed in [2000, 2001, 2002, 2003, 2004, 2005]:
    img = make_test_image(seed=seed)
    det = detect_p5(img, key)
    inner = "YES" if det.get("inner_codeword") else "NO"
    print(f"  seed={seed}: conf={det['confidence']:.4f}  inner={inner}  "
          f"geo={det.get('geometry', {}).get('method', 'none')[:20] if det.get('geometry') else 'none'}")

print()
print("detect_p5 confidence for watermarked at 0deg (should be ~0.40):")
det = detect_p5(wm512, key)
print(f"  conf={det['confidence']:.4f}  inner={'YES' if det.get('inner_codeword') else 'NO'}")

print()
print("verify_p5 confidence for watermarked at 0deg (should be 1.0):")
r = verify_p5(wm512, pub, key)
print(f"  conf={r['confidence']:.4f}  verified={r['verified']}")

print()
print("Expected ROC scores:")
pos_confs = []
for angle in [0, 5, 10, -5]:
    img = rotate(wm512, angle) if angle != 0 else wm512
    det = detect_p5(img, key)
    pos_confs.append(det["confidence"])
    print(f"  pos angle={angle:+3d}: conf={det['confidence']:.4f}")

print()
neg_total = 0
for seed in range(10):
    img = make_test_image(seed=3000 + seed)
    det = detect_p5(img, key)
    neg_total += det["confidence"]
    if det["confidence"] > 0.01:
        print(f"  HIGH NEG seed={3000+seed}: conf={det['confidence']:.4f}")
print(f"  Neg mean: {neg_total/10:.4f}")

# Simple AUC estimate
wins = 0
pairs = 0
neg_confs = []
for seed in range(10):
    img = make_test_image(seed=3000 + seed)
    det = detect_p5(img, key)
    neg_confs.append(det["confidence"])
for pc in pos_confs:
    for nc in neg_confs:
        pairs += 1
        if pc > nc: wins += 1
        elif pc == nc: wins += 0.5
print(f"\nEstimated AUC: {wins/pairs:.4f}")
