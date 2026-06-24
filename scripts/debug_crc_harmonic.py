"""Debug: trace harmonic_score and p4_direct crc_ratio for WM vs clean images."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from src.core.wm_engine_p4 import detect_p4
from src.core.wm_engine_p5 import detect_p5
from src.core.crypto import generate_keys
from src.core.mist import watermark_p5
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

print("=" * 65)
print("P4 on watermarked image under rotation (p4_direct crc_ratio):")
for angle in [0, 2, 5, 10, 15]:
    t0 = time.time()
    img = rotate(wm512, angle)
    r = detect_p4(img, key)
    elapsed = time.time() - t0
    crc_ratio = r.get("shard_crc_ratio", 0.0)
    harmonic = r.get("harmonic_score", 0.0)
    print(f"  rot={angle:+3d}deg: tiles={r['tiles_located']:3d} "
          f"shards={r['shards_recovered']:3d} crc_ratio={crc_ratio:.4f} "
          f"harmonic={harmonic:.4f} conf={r['confidence']:.4f}  ({elapsed:.1f}s)")

print()
print("P4 on CLEAN images (crc_ratio, harmonic_score):")
for seed in range(2000, 2007):
    img = make_test_image(seed=seed)
    r = detect_p4(img, key)
    crc_ratio = r.get("shard_crc_ratio", 0.0)
    harmonic = r.get("harmonic_score", 0.0)
    inner = "YES" if r.get("inner_codeword") else "NO"
    print(f"  seed={seed}: tiles={r['tiles_located']:3d} "
          f"crc_ratio={crc_ratio:.4f} harmonic={harmonic:.4f} "
          f"conf={r['confidence']:.4f} inner={inner}")
