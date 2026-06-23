"""Quick targeted test for Phase 5 geometric search."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
from src.core.crypto import generate_keys
from src.core.mist import watermark_p5, verify_p5
from src.attacks.geometric import rotate, scale

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
print("Embedded OK")

tests = [
    ("Rotation +5", rotate(wm, 5)),
    ("Scale 0.8x", scale(wm, 0.8)),
    ("Scale 1.2x", scale(wm, 1.2)),
    ("Rot+5 Scale 0.8", scale(rotate(wm, 5), 0.8)),
]

for name, attacked in tests:
    t0 = time.time()
    r = verify_p5(attacked, pub, key)
    dt = time.time() - t0
    geo = r.get("geometry", {})
    geo_str = ""
    if geo:
        geo_str = (f" | Geo: {geo.get('angle_deg', 0):.1f}deg "
                   f"x{geo.get('scale_factor', 1):.2f} "
                   f"({geo.get('method', '')})")
    status = "PASS" if r["verified"] else "FAIL"
    print(f"  [{status}] {name}: ver={r['verified']} det={r['detected']} "
          f"shards={r['shards_recovered']}/{r['shards_needed']}{geo_str} ({dt:.1f}s)")
