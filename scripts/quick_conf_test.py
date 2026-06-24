"""Quick confidence calibration sanity check."""
import numpy as np
from src.core.wm_engine_p5 import _compute_confidence

cases = [
    # name,                        sig,   rs,    shard,   crc,   geo,  corr,  want
    ("Clean (crc=0.004, geo=0)",  False, False, 1.00, 0.004, 0.0, 0.0, "< 0.01"),
    ("False RS decode (crc=.03)", False, True,  1.00, 0.030, 0.0, 0.0, "< 0.05"),  # seed-2003 type
    ("WM partial 26/30 (crc=.95)",False, False, 0.87, 0.95,  0.0, 0.0, "> 0.08"),
    ("WM direct detect (sig=F)",  False, True,  1.00, 1.00,  1.0, 0.0, "~ 0.50"),
    ("WM verify (sig=T, crc=1.0)",True,  True,  1.00, 1.00,  1.0, 0.0, "> 0.90"),  # mist.py sets 1.0
    ("seed-3009 (rs, crc=0.004)", False, True,  1.00, 0.004, 0.0, 0.0, "< 0.01"),  # false RS+no-geo
]

print("\nCalibration check:")
print("-" * 65)
for name, sig, rs, shard, crc, geo, corr, want in cases:
    c = _compute_confidence(sig, rs, shard, crc, geo, corr)
    wval = float(want.split()[1])
    if want.startswith("<"):   ok = c < wval
    elif want.startswith(">"): ok = c > wval
    else:                      ok = abs(c - wval) < 0.15
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {name:<38} conf={c:.4f}  want {want}")

print("\nP4 clean image test:")
from src.core.wm_engine_p4 import detect_p4
img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
r = detect_p4(img, b'test-key')
print(f"  P4 confidence:    {r['confidence']:.4f}  (want ~0)")
print(f"  shard_crc_ratio:  {r.get('shard_crc_ratio', 'MISSING')}")
print(f"  tiles_located:    {r.get('tiles_located', 0)}")
