"""
src/core/geometry_correction.py  –  Affine Geometry Correction (Agent 2)

Implements inverse affine normalization after geometry synchronization.

The core architectural insight: rather than extracting watermarks from
distorted tiles, we first normalize the image back to its canonical
(pre-attack) geometry, then run the existing P4 detector on the
corrected image.

Pipeline
--------
    1. Compute combined inverse affine matrix (rotation + scale)
    2. Single-pass warpAffine with bicubic interpolation
    3. BORDER_REFLECT_101 edge padding (no ringing artifacts)
    4. Optional luma energy preservation
    5. Deterministic output (same inputs → same output)

Key design decisions
--------------------
    - Single affine warp (not separate scale → rotate) avoids double
      resampling which destroys watermark energy.
    - Bicubic (INTER_CUBIC) instead of Lanczos to balance sharpness
      vs. ringing. Lanczos is sharper but introduces ringing near edges
      which corrupts DCT coefficient relationships.
    - BORDER_REFLECT_101 instead of BORDER_CONSTANT (zero-padding would
      create false edges that corrupt tiles near the border).
    - Output dimensions are computed to match the canonical (pre-attack)
      image size, ensuring P4's macro-tile grid aligns correctly.

Public API
----------
    correct_geometry(image, rotation_deg, scale_factor, ...)
        → corrected_image (np.ndarray)

    correct_geometry_Y(Y, rotation_deg, scale_factor, ...)
        → corrected_Y (np.ndarray, float32)

    compute_inverse_affine(rotation_deg, scale_factor, src_h, src_w, dst_h, dst_w)
        → M (2×3 affine matrix)

    estimate_canonical_size(h, w, scale_factor)
        → (canonical_h, canonical_w)

References
----------
    - Utah State University: "A Robust DCT-Based Digital Watermarking Scheme"
    - De Castro & Morandi 1987: affine normalization after FM registration
    - Standard robust watermarking architecture (affine correction → detector)
"""

import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum dimension for corrected output (must fit at least 1 macro-tile)
_MIN_DIM: int = 256

# Maximum allowed output dimension (prevent memory explosion)
_MAX_DIM: int = 8192

# PSNR threshold for energy preservation check
_ENERGY_PSNR_MIN: float = 25.0

# Interpolation flags
_INTERP_BICUBIC: int = cv2.INTER_CUBIC
_INTERP_AREA: int = cv2.INTER_AREA

# Border mode: reflect without duplicating edge pixel
_BORDER_MODE: int = cv2.BORDER_REFLECT_101


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical Size Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_canonical_size(
    h: int,
    w: int,
    scale_factor: float,
) -> tuple[int, int]:
    """
    Estimate the canonical (pre-attack) image dimensions.

    The attack pipeline applies:  scale(image, factor)
    So the canonical size is:     current_size / scale_factor

    Uses ceil() to match the attack's int() truncation:
        attacked_size = int(original_size * factor)
        => original_size = ceil(attacked_size / factor)

    Parameters
    ----------
    h, w         : current (attacked) image dimensions
    scale_factor : estimated scale factor applied by the attack

    Returns
    -------
    (canonical_h, canonical_w)
    """
    if abs(scale_factor) < 0.01:
        scale_factor = 1.0

    inv_factor = 1.0 / scale_factor
    canonical_h = int(np.ceil(h * inv_factor))
    canonical_w = int(np.ceil(w * inv_factor))

    # Clamp to valid range
    canonical_h = max(_MIN_DIM, min(_MAX_DIM, canonical_h))
    canonical_w = max(_MIN_DIM, min(_MAX_DIM, canonical_w))

    return canonical_h, canonical_w


# ─────────────────────────────────────────────────────────────────────────────
#  Inverse Affine Matrix Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_inverse_affine(
    rotation_deg: float,
    scale_factor: float,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> np.ndarray:
    """
    Compute the 2×3 inverse affine matrix that maps output (corrected)
    coordinates back to input (attacked) coordinates.

    The attack applies:  rotate(image, θ) → scale(result, s)
    To invert, we need:  S⁻¹ · R⁻¹  (first undo scale, then undo rotation)

    But since cv2.warpAffine maps DST→SRC (inverse mapping), we compute
    the forward mapping from DST (corrected) to SRC (attacked):

        For each output pixel (x', y'):
            1. Center on output image
            2. Apply rotation by +θ (undo the -θ attack rotation)
            3. Scale by s (undo the 1/s inverse scaling)
            4. Re-center on input image

    This is combined into a single 2×3 matrix for one-pass warping.

    Parameters
    ----------
    rotation_deg : attack rotation angle in degrees
    scale_factor : attack scale factor
    src_h, src_w : source (attacked) image dimensions
    dst_h, dst_w : destination (corrected) image dimensions

    Returns
    -------
    M : np.ndarray [2, 3] float64 — affine matrix for cv2.warpAffine
    """
    # Centers
    cx_src = src_w / 2.0
    cy_src = src_h / 2.0
    cx_dst = dst_w / 2.0
    cy_dst = dst_h / 2.0

    # Scale ratios: how dst pixels map to src pixels
    # If scale_factor > 1 (image was enlarged), then dst is smaller,
    # so each dst pixel covers more src pixels → sy, sx > 1
    sy = src_h / dst_h
    sx = src_w / dst_w

    # Rotation: to undo rotation by +θ, we rotate the mapping by +θ
    # (since warpAffine uses inverse mapping: dst→src)
    rad = np.radians(rotation_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    # Combined affine: scale + rotate (dst-centered → src-centered)
    # [x_src]   [cos·sx   sin·sy] [x_dst - cx_dst]   [cx_src]
    # [y_src] = [-sin·sx  cos·sy] [y_dst - cy_dst] + [cy_src]
    a11 = cos_a * sx
    a12 = sin_a * sy
    a21 = -sin_a * sx
    a22 = cos_a * sy

    # Translation: ensure centers map correctly
    tx = cx_src - (a11 * cx_dst + a12 * cy_dst)
    ty = cy_src - (a21 * cx_dst + a22 * cy_dst)

    M = np.array([
        [a11, a12, tx],
        [a21, a22, ty],
    ], dtype=np.float64)

    return M


# ─────────────────────────────────────────────────────────────────────────────
#  Core Geometry Correction
# ─────────────────────────────────────────────────────────────────────────────

def correct_geometry(
    image: np.ndarray,
    rotation_deg: float,
    scale_factor: float,
    *,
    canonical_size: tuple[int, int] | None = None,
    preserve_energy: bool = True,
    use_bicubic: bool = True,
) -> np.ndarray:
    """
    Apply inverse affine transform to normalize a geometrically attacked image.

    This is the primary entry point for Agent 2. It computes a single inverse
    affine matrix combining rotation and scale correction, then applies it in
    one pass to minimize resampling degradation.

    Parameters
    ----------
    image          : input image — BGR uint8 [H, W, 3] or grayscale [H, W]
    rotation_deg   : estimated rotation angle (degrees) applied by the attack
    scale_factor   : estimated scale factor applied by the attack
    canonical_size : (h, w) target output size, or None to auto-compute
    preserve_energy: if True, normalize luma mean to preserve watermark energy
    use_bicubic    : if True, use bicubic interpolation; else use area/linear

    Returns
    -------
    corrected : np.ndarray — corrected image, same dtype as input
    """
    if image.ndim == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 0

    # ── Compute target dimensions ─────────────────────────────────────
    if canonical_size is not None:
        dst_h, dst_w = canonical_size
        dst_h = max(_MIN_DIM, min(_MAX_DIM, dst_h))
        dst_w = max(_MIN_DIM, min(_MAX_DIM, dst_w))
    else:
        dst_h, dst_w = estimate_canonical_size(h, w, scale_factor)

    # ── Trivial case: identity transform ──────────────────────────────
    if abs(rotation_deg) < 0.001 and abs(scale_factor - 1.0) < 0.001:
        return image.copy()

    # ── Compute inverse affine matrix ─────────────────────────────────
    M = compute_inverse_affine(
        rotation_deg, scale_factor,
        src_h=h, src_w=w,
        dst_h=dst_h, dst_w=dst_w,
    )

    # ── Select interpolation method ───────────────────────────────────
    # Bicubic: best for rotation correction (smooth, low ringing)
    # Area: best for downscaling (proper anti-aliasing)
    if use_bicubic:
        interp = _INTERP_BICUBIC
    else:
        # Use area interpolation when downscaling, linear when upscaling
        if dst_h * dst_w < h * w:
            interp = _INTERP_AREA
        else:
            interp = cv2.INTER_LINEAR

    # ── Apply single-pass affine warp ─────────────────────────────────
    if c > 0:
        corrected = cv2.warpAffine(
            image, M, (dst_w, dst_h),
            flags=interp + cv2.WARP_INVERSE_MAP,
            borderMode=_BORDER_MODE,
        )
    else:
        corrected = cv2.warpAffine(
            image, M, (dst_w, dst_h),
            flags=interp + cv2.WARP_INVERSE_MAP,
            borderMode=_BORDER_MODE,
        )

    # ── Energy preservation ───────────────────────────────────────────
    if preserve_energy and image.dtype == np.uint8 and c > 0:
        corrected = _preserve_luma_energy(image, corrected)

    return corrected


def correct_geometry_Y(
    Y: np.ndarray,
    rotation_deg: float,
    scale_factor: float,
    *,
    canonical_size: tuple[int, int] | None = None,
    preserve_energy: bool = True,
) -> np.ndarray:
    """
    Apply inverse affine transform on single-channel Y (float32).

    Optimized path for canary scoring and shard extraction where only
    the luminance channel is needed.

    Parameters
    ----------
    Y              : luminance channel — float32 [H, W]
    rotation_deg   : estimated rotation angle (degrees)
    scale_factor   : estimated scale factor
    canonical_size : (h, w) target output size, or None to auto-compute
    preserve_energy: if True, normalize mean luma to preserve energy

    Returns
    -------
    corrected_Y : np.ndarray float32 [dst_H, dst_W]
    """
    h, w = Y.shape

    # Compute target dimensions
    if canonical_size is not None:
        dst_h, dst_w = canonical_size
        dst_h = max(_MIN_DIM, min(_MAX_DIM, dst_h))
        dst_w = max(_MIN_DIM, min(_MAX_DIM, dst_w))
    else:
        dst_h, dst_w = estimate_canonical_size(h, w, scale_factor)

    # Trivial case
    if abs(rotation_deg) < 0.001 and abs(scale_factor - 1.0) < 0.001:
        return Y.copy()

    # Compute inverse affine
    M = compute_inverse_affine(
        rotation_deg, scale_factor,
        src_h=h, src_w=w,
        dst_h=dst_h, dst_w=dst_w,
    )

    # Single-pass warp (bicubic for float32)
    corrected = cv2.warpAffine(
        Y, M, (dst_w, dst_h),
        flags=_INTERP_BICUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=_BORDER_MODE,
    )

    # Energy preservation on Y channel
    if preserve_energy:
        src_mean = float(np.mean(Y))
        dst_mean = float(np.mean(corrected))
        if dst_mean > 1e-6 and src_mean > 1e-6:
            ratio = src_mean / dst_mean
            # Only correct if drift is significant but not extreme
            if 0.8 < ratio < 1.25:
                corrected = corrected * ratio

    return corrected.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Energy Preservation
# ─────────────────────────────────────────────────────────────────────────────

def _preserve_luma_energy(
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """
    Normalize the corrected image's luminance to match the source's mean luma.

    Resampling can shift the average brightness (especially with border
    padding), which reduces the watermark's effective embedding strength.
    This correction ensures the DCT coefficient relationships are preserved.

    Applied in-place on the destination image.
    """
    # Convert to float for luma computation
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float64)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY).astype(np.float64)

    src_mean = np.mean(src_gray)
    dst_mean = np.mean(dst_gray)

    if dst_mean < 1e-6 or src_mean < 1e-6:
        return dst

    ratio = src_mean / dst_mean

    # Only correct if drift is within reasonable bounds
    # Large drift indicates border artifacts, not systematic shift
    if 0.85 < ratio < 1.18:
        corrected = dst.astype(np.float64) * ratio
        return np.clip(corrected, 0, 255).astype(np.uint8)

    return dst


# ─────────────────────────────────────────────────────────────────────────────
#  Correction Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def correction_psnr(
    original: np.ndarray,
    corrected: np.ndarray,
) -> float:
    """
    Compute PSNR between original and corrected images.

    Used for verification: PSNR loss after correction should be < 1.5 dB
    compared to the PSNR between original and directly-transformed image.

    If images have different dimensions, the corrected image is resized
    to match the original for comparison.

    Returns PSNR in dB. Higher = better. Returns float('inf') if identical.
    """
    if original.shape != corrected.shape:
        corrected = cv2.resize(
            corrected,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    if original.ndim == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)
        corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        original_gray = original.astype(np.float64)
        corrected_gray = corrected.astype(np.float64)

    mse = np.mean((original_gray - corrected_gray) ** 2)
    if mse < 1e-10:
        return float('inf')

    return 10.0 * np.log10(255.0 ** 2 / mse)


def verify_grid_alignment(
    corrected: np.ndarray,
    block_size: int = 8,
    tolerance: float = 0.5,
) -> dict:
    """
    Verify that the corrected image's dimensions are compatible with
    the P4 macro-tile grid (256×256 tiles on an 8×8 block grid).

    Parameters
    ----------
    corrected  : corrected image
    block_size : DCT block size (8)
    tolerance  : max allowed fractional pixel misalignment

    Returns
    -------
    dict with:
        grid_aligned  : bool — dimensions are multiples of block_size
        tile_count    : int  — number of full 256×256 tiles
        h_remainder   : int  — pixels lost at bottom
        w_remainder   : int  — pixels lost at right
        phase_ok      : bool — tile phase is consistent
    """
    if corrected.ndim == 3:
        h, w = corrected.shape[:2]
    else:
        h, w = corrected.shape

    mt_size = 256  # macro-tile size

    h_remainder = h % block_size
    w_remainder = w % block_size
    grid_aligned = (h_remainder == 0 and w_remainder == 0)

    mt_rows = h // mt_size
    mt_cols = w // mt_size
    tile_count = mt_rows * mt_cols

    # Phase check: verify that tile boundaries fall on block boundaries
    phase_ok = (h % mt_size % block_size == 0) and (w % mt_size % block_size == 0)

    return {
        "grid_aligned": grid_aligned,
        "tile_count": tile_count,
        "h_remainder": h_remainder,
        "w_remainder": w_remainder,
        "phase_ok": phase_ok,
        "dimensions": (h, w),
    }
