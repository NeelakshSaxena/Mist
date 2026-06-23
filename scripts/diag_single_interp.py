"""Compare double interpolation (current) vs single affine warp (proposed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2, numpy as np
from src.core.crypto import generate_keys
from src.core.mist import watermark_p5
from src.attacks.geometric import rotate, scale
from src.core.wm_engine_p5 import _score_candidate, _undo_transform, MT_SIZE

# ── Single-interpolation undo ────────────────────────────────────────
def _undo_transform_single(image, angle_deg, scale_factor):
    """Undo rotation + scaling in ONE warpAffine (single interpolation)."""
    h, w = image.shape[:2]
    f = scale_factor
    inv_f = 1.0 / max(f, 0.01)

    new_w = max(MT_SIZE, int(round(w * inv_f)))
    new_h = max(MT_SIZE, int(round(h * inv_f)))

    cx_src = w / 2.0
    cy_src = h / 2.0
    cx_dst = new_w / 2.0
    cy_dst = new_h / 2.0

    a_rad = np.radians(angle_deg)
    cos_a = np.cos(a_rad)
    sin_a = np.sin(a_rad)

    # Inverse mapping matrix (dst pixel → src pixel)
    # dst→center → rotate(+angle) → scale(factor) → translate to src center
    M = np.array([
        [f * cos_a, -f * sin_a, cx_src - f * cos_a * cx_dst + f * sin_a * cy_dst],
        [f * sin_a,  f * cos_a, cy_src - f * sin_a * cx_dst - f * cos_a * cy_dst],
    ], dtype=np.float64)

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


# ── Test setup ───────────────────────────────────────────────────────
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

attacked = scale(rotate(wm, 5), 0.8)
print(f"Attacked: {attacked.shape}")

# ── Compare at correct params ────────────────────────────────────────
test_params = [
    (5.0, 0.80, "CORRECT"),
    (4.0, 0.80, "close: rot=4"),
    (6.0, 0.80, "close: rot=6"),
    (5.0, 0.75, "close: s=0.75"),
    (5.0, 0.85, "close: s=0.85"),
]

print(f"\n{'Params':>20} {'Double CRC':>10} {'Double Tot':>10} {'Single CRC':>10} {'Single Tot':>10}")
print("-" * 70)
for angle, sf, desc in test_params:
    c_double = _undo_transform(attacked, angle, sf)
    c_single = _undo_transform_single(attacked, angle, sf)
    crc_d, tot_d = _score_candidate(c_double, key)
    crc_s, tot_s = _score_candidate(c_single, key)
    print(f"{desc:>20} {crc_d:>10d} {tot_d:>10d} {crc_s:>10d} {tot_s:>10d}")
