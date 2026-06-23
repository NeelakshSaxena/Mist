"""
src/core/geometry_sync.py  –  Fourier-Mellin Geometry Synchronization

Replaces the brute-force canary candidate search in Phase 5 with a
Fourier-Mellin-based synchronization pipeline:

    FFT magnitude → log-polar remap (cv2.warpPolar) → phase correlation
    → rotation/scale estimate

This is a direct implementation of the Fourier-Mellin Transform (FMT)
registration technique (De Castro & Morandi 1987, Reddy & Chatterji 1996).

Key insight: The magnitude spectrum of an FFT is translation-invariant.
When two images differ by rotation θ and scale s, their log-polar
magnitude spectra differ only by a translation — which phase correlation
can recover in one shot.

Pipeline
--------
    1. Apply Hann window to suppress spectral leakage
    2. FFT2 → magnitude spectrum (shift-invariant)
    3. Log-scale magnitude (compress dynamic range)
    4. cv2.warpPolar with WARP_POLAR_LOG (log-polar remap)
    5. cv2.phaseCorrelate on the two log-polar images
    6. Convert (Δrow, Δcol) shifts → (rotation_deg, scale_factor)

Constraints
-----------
    - Must run under 250ms for 512×512 inputs
    - Must avoid brute-force search (O(1) correlation, not O(N) sweep)
    - Supports: rotation ±20°, scale 0.5×–1.5×

Public API
----------
    estimate_geometry_sync(attacked, reference=None)
        → dict with rotation_deg, scale_factor, confidence, response_peak

    fourier_mellin_register(image1, image2)
        → (rotation_deg, scale_factor, confidence, response_peak)
"""

import time
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Standard analysis size — resized input for deterministic performance
_ANALYSIS_SIZE: int = 512

# Log-polar remap radius (fraction of min dimension / 2)
_LP_RADIUS_FRAC: float = 0.95

# Confidence threshold below which result is unreliable
MIN_CONFIDENCE: float = 0.15

# Supported geometry ranges
MAX_ROTATION_DEG: float = 20.0
MIN_SCALE: float = 0.5
MAX_SCALE: float = 1.5

# Subpixel refinement window for peak localization
_REFINE_WINDOW: int = 5


# ─────────────────────────────────────────────────────────────────────────────
#  Windowing
# ─────────────────────────────────────────────────────────────────────────────

def _apply_hann_window(image: np.ndarray) -> np.ndarray:
    """
    Apply a 2D Hann window to suppress spectral leakage at image borders.

    The Hann window smoothly tapers pixel values to zero at the edges,
    preventing the hard rectangular boundary from injecting high-frequency
    artifacts into the FFT.
    """
    h, w = image.shape
    win_y = np.hanning(h).astype(np.float32)
    win_x = np.hanning(w).astype(np.float32)
    window = np.outer(win_y, win_x)
    return image * window


# ─────────────────────────────────────────────────────────────────────────────
#  FFT Magnitude Spectrum
# ─────────────────────────────────────────────────────────────────────────────

def _fft_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute the centered FFT magnitude spectrum with log scaling.

    Steps:
        1. FFT2 of the windowed image
        2. fftshift to center DC component
        3. Take magnitude (abs)
        4. Log-scale: log(1 + |F|) to compress dynamic range

    The magnitude spectrum is translation-invariant — only rotation and
    scale affect it.
    """
    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)
    # Log-scale to compress dynamic range (DC spike → manageable peak)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  High-pass filter (suppress DC bleed into log-polar)
# ─────────────────────────────────────────────────────────────────────────────

def _highpass_filter(magnitude: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Apply a Gaussian high-pass filter to the magnitude spectrum.

    Removes the dominant DC component and very low frequencies that
    would otherwise dominate the log-polar correlation and reduce
    sensitivity to the rotation/scale signal.
    """
    lowpass = cv2.GaussianBlur(magnitude, (0, 0), sigma)
    return magnitude - lowpass


# ─────────────────────────────────────────────────────────────────────────────
#  Log-Polar Remap (via cv2.warpPolar)
# ─────────────────────────────────────────────────────────────────────────────

def _log_polar_remap(
    magnitude: np.ndarray,
    dsize: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Convert a centered magnitude spectrum to log-polar coordinates
    using cv2.warpPolar with WARP_POLAR_LOG.

    In log-polar space:
        - Rotation becomes a vertical shift (along rows)
        - Scale becomes a horizontal shift (along columns)

    This transforms the rotation+scale estimation problem into a
    pure translation estimation — solvable by phase correlation.

    Parameters
    ----------
    magnitude : 2D float32 array (centered FFT magnitude)
    dsize     : output size (width, height). Defaults to input size.

    Returns
    -------
    log_polar : 2D float32 array in log-polar coordinates
    """
    h, w = magnitude.shape
    center = (w / 2.0, h / 2.0)
    max_radius = min(w, h) * _LP_RADIUS_FRAC / 2.0

    if dsize is None:
        dsize = (w, h)

    flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS
    log_polar = cv2.warpPolar(
        magnitude,
        dsize=dsize,
        center=center,
        maxRadius=max_radius,
        flags=flags,
    )
    return log_polar.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase Correlation
# ─────────────────────────────────────────────────────────────────────────────

def _phase_correlate(
    lp_ref: np.ndarray,
    lp_atk: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute phase correlation between two log-polar images.

    Uses cv2.phaseCorrelate for subpixel-accurate shift estimation.

    Returns
    -------
    shift_y : vertical shift (→ rotation angle)
    shift_x : horizontal shift (→ log-scale)
    response : correlation peak strength [0, 1]
    """
    # cv2.phaseCorrelate expects float64 input
    ref_64 = lp_ref.astype(np.float64)
    atk_64 = lp_atk.astype(np.float64)

    # Apply Hann window to log-polar images to improve correlation quality
    h, w = ref_64.shape
    win_y = cv2.createHanningWindow((w, h), cv2.CV_64F)

    shift, response = cv2.phaseCorrelate(ref_64 * win_y, atk_64 * win_y)

    # cv2.phaseCorrelate returns (dx, dy), response
    shift_x = shift[0]  # horizontal = log-scale
    shift_y = shift[1]  # vertical = rotation

    return float(shift_y), float(shift_x), float(response)


# ─────────────────────────────────────────────────────────────────────────────
#  Shift → Rotation / Scale Conversion
# ─────────────────────────────────────────────────────────────────────────────

def _shifts_to_geometry(
    shift_y: float,
    shift_x: float,
    image_h: int,
    image_w: int,
) -> tuple[float, float]:
    """
    Convert log-polar phase correlation shifts to rotation angle and scale.

    In the warpPolar log-polar mapping:
        - Row index maps to angle: θ = row * 360° / height
        - Column index maps to log-radius: r = maxRadius^(col/width)

    A shift Δrow in correlation → rotation of Δrow * 360° / height
    A shift Δcol in correlation → scale of maxRadius^(Δcol/width)

    Parameters
    ----------
    shift_y : vertical shift from phase correlation (rotation signal)
    shift_x : horizontal shift from phase correlation (scale signal)
    image_h : height of the log-polar image
    image_w : width of the log-polar image

    Returns
    -------
    rotation_deg : estimated rotation in degrees
    scale_factor : estimated scale factor
    """
    # Rotation: each row spans 360°/h degrees
    rotation_deg = -shift_y * 360.0 / image_h

    # Handle angle wrapping — keep in [-180, 180]
    if rotation_deg > 180.0:
        rotation_deg -= 360.0
    elif rotation_deg < -180.0:
        rotation_deg += 360.0

    # Scale: log-polar column shift → scale
    max_radius = min(image_w, image_h) * _LP_RADIUS_FRAC / 2.0
    log_base = np.log(max_radius)
    scale_factor = np.exp(shift_x * log_base / image_w)

    return rotation_deg, scale_factor


# ─────────────────────────────────────────────────────────────────────────────
#  Reference Spectrum Generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_reference_spectrum(h: int, w: int) -> np.ndarray:
    """
    Generate a synthetic reference spectrum for self-correlation mode.

    When we don't have the original image, we use a flat white-noise
    reference. The attacked image's spectrum is correlated against
    a synthetic "ideal" pattern (uniform impulse grid at embedding
    block spacing).

    For Fourier-Mellin, the reference is typically the original image.
    In watermark detection, we can use the image itself as reference
    by correlating the magnitude spectrum against an ideal/flat reference
    and looking for the rotation/scale that maximizes spectral symmetry.
    """
    # Generate an ideal grid pattern at 8-pixel block spacing
    # (matches the watermark embedding block structure)
    ref = np.zeros((h, w), dtype=np.float32)
    ref[::8, ::8] = 1.0

    # Apply same pipeline
    ref_windowed = _apply_hann_window(ref)
    ref_mag = _fft_magnitude(ref_windowed)
    ref_mag = _highpass_filter(ref_mag)
    return ref_mag


# ─────────────────────────────────────────────────────────────────────────────
#  Core Registration Function
# ─────────────────────────────────────────────────────────────────────────────

def fourier_mellin_register(
    image: np.ndarray,
    reference: np.ndarray | None = None,
    image_original_hw: tuple[int, int] | None = None,
    reference_original_hw: tuple[int, int] | None = None,
) -> tuple[float, float, float, float]:
    """
    Fourier-Mellin registration between image and reference.

    Full pipeline:
        1. Resize to analysis size (deterministic timing)
        2. Hann window
        3. FFT2 → magnitude spectrum → log-scale
        4. High-pass filter
        5. cv2.warpPolar log-polar remap
        6. cv2.phaseCorrelate
        7. Convert shifts → (rotation, scale)

    Parameters
    ----------
    image     : grayscale float32 or uint8 image (attacked)
    reference : grayscale float32 or uint8 image (original), or None
                If None, uses a synthetic block-grid reference.
    image_original_hw : (h, w) of attacked image before any resizing
    reference_original_hw : (h, w) of reference image before any resizing

    Returns
    -------
    rotation_deg  : estimated rotation angle in degrees
    scale_factor  : estimated scale factor
    confidence    : phase correlation response peak [0, 1]
    response_peak : raw correlation peak value
    """
    # ── Preprocessing ─────────────────────────────────────────────────
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = image.astype(np.float32)

    # Track original dimensions for scale ambiguity resolution
    if image_original_hw is None:
        image_original_hw = img.shape[:2]

    # Resize to analysis size for deterministic performance
    h_orig, w_orig = img.shape
    if h_orig != _ANALYSIS_SIZE or w_orig != _ANALYSIS_SIZE:
        img = cv2.resize(img, (_ANALYSIS_SIZE, _ANALYSIS_SIZE),
                         interpolation=cv2.INTER_LINEAR)

    # ── Step 1: Hann window ───────────────────────────────────────────
    img_windowed = _apply_hann_window(img)

    # ── Step 2: FFT magnitude spectrum ────────────────────────────────
    img_mag = _fft_magnitude(img_windowed)

    # ── Step 3: High-pass filter ──────────────────────────────────────
    img_mag = _highpass_filter(img_mag)

    # ── Reference processing ──────────────────────────────────────────
    if reference is not None:
        if reference.ndim == 3:
            reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        ref = reference.astype(np.float32)
        if reference_original_hw is None:
            reference_original_hw = ref.shape[:2]
        if ref.shape[0] != _ANALYSIS_SIZE or ref.shape[1] != _ANALYSIS_SIZE:
            ref = cv2.resize(ref, (_ANALYSIS_SIZE, _ANALYSIS_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        ref_windowed = _apply_hann_window(ref)
        ref_mag = _fft_magnitude(ref_windowed)
        ref_mag = _highpass_filter(ref_mag)
    else:
        ref_mag = _generate_reference_spectrum(_ANALYSIS_SIZE, _ANALYSIS_SIZE)
        reference_original_hw = image_original_hw  # self-reference

    # ── Step 4: Log-polar remap ───────────────────────────────────────
    lp_img = _log_polar_remap(img_mag)
    lp_ref = _log_polar_remap(ref_mag)

    # ── Step 5: Phase correlation ─────────────────────────────────────
    shift_y, shift_x, response = _phase_correlate(lp_ref, lp_img)

    # ── Step 6: Convert shifts → geometry ─────────────────────────────
    rotation_deg, scale_factor = _shifts_to_geometry(
        shift_y, shift_x, _ANALYSIS_SIZE, _ANALYSIS_SIZE,
    )

    # ── Resolve 180° ambiguity ────────────────────────────────────────
    # Phase correlation on log-polar has 180° ambiguity.
    # We check both candidates and pick the one in our supported range.
    rotation_alt = rotation_deg + 180.0
    if rotation_alt > 180.0:
        rotation_alt -= 360.0

    # Prefer the angle within ±MAX_ROTATION_DEG
    if abs(rotation_deg) > MAX_ROTATION_DEG and abs(rotation_alt) <= MAX_ROTATION_DEG:
        rotation_deg = rotation_alt
    elif abs(rotation_deg) > MAX_ROTATION_DEG and abs(rotation_alt) > MAX_ROTATION_DEG:
        # Both outside range — pick the smaller one
        if abs(rotation_alt) < abs(rotation_deg):
            rotation_deg = rotation_alt

    # Clamp scale to supported range
    scale_factor = max(MIN_SCALE, min(MAX_SCALE, scale_factor))

    # ── Resolve scale reciprocal ambiguity ────────────────────────────
    # Phase correlation on log-polar spectra has a sign ambiguity for
    # the scale axis. We resolve using actual image dimensions:
    # if image is larger than reference, scale > 1; if smaller, < 1.
    reciprocal_sf = 1.0 / scale_factor if scale_factor > 0.01 else 1.0
    reciprocal_sf = max(MIN_SCALE, min(MAX_SCALE, reciprocal_sf))

    if reference_original_hw is not None and image_original_hw is not None:
        # Use dimension ratio to determine scale direction
        img_area = image_original_hw[0] * image_original_hw[1]
        ref_area = reference_original_hw[0] * reference_original_hw[1]
        dim_ratio = (img_area / ref_area) ** 0.5 if ref_area > 0 else 1.0

        if abs(dim_ratio - 1.0) > 0.03:
            # Dimensions differ significantly — use dimension ratio as
            # the scale estimate directly. This is more reliable than
            # log-polar phase correlation for scale when dimensions
            # are known. The FFT pipeline excels at rotation; scale is
            # trivially computable from pixel dimensions.
            scale_factor = max(MIN_SCALE, min(MAX_SCALE, dim_ratio))
        else:
            # Dimensions similar — pick closer to 1.0
            if abs(reciprocal_sf - 1.0) < abs(scale_factor - 1.0):
                scale_factor = reciprocal_sf
    else:
        # No dimension info — pick closer to 1.0
        if abs(reciprocal_sf - 1.0) < abs(scale_factor - 1.0):
            scale_factor = reciprocal_sf

    # ── Confidence calibration ────────────────────────────────────────
    # cv2.phaseCorrelate response is in [0, ~1] but often much lower.
    # Calibrate to a [0, 1] confidence score.
    confidence = min(1.0, max(0.0, response))

    return rotation_deg, scale_factor, confidence, response


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-hypothesis Registration
# ─────────────────────────────────────────────────────────────────────────────

def _multi_hypothesis_register(
    image: np.ndarray,
    reference: np.ndarray | None = None,
    n_scales: int = 3,
) -> tuple[float, float, float, float]:
    """
    Run Fourier-Mellin at multiple analysis scales and pick the best.

    This improves robustness: aliasing and rounding artifacts in
    warpPolar can be scale-dependent. Trying 3 sizes (448, 512, 576)
    catches cases where one resolution's sampling happens to misalign.

    Returns the hypothesis with highest confidence.
    """
    sizes = [
        max(256, _ANALYSIS_SIZE - 64),
        _ANALYSIS_SIZE,
        _ANALYSIS_SIZE + 64,
    ]

    best = (0.0, 1.0, 0.0, 0.0)  # (rot, scale, conf, peak)

    # Track original dimensions for scale direction resolution
    if image.ndim == 3:
        img_orig_hw = image.shape[:2]
        gray_base = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_orig_hw = image.shape[:2]
        gray_base = image

    ref_orig_hw = img_orig_hw  # default for self-reference
    ref_gray_base = None
    if reference is not None:
        if reference.ndim == 3:
            ref_orig_hw = reference.shape[:2]
            ref_gray_base = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_orig_hw = reference.shape[:2]
            ref_gray_base = reference

    for sz in sizes[:n_scales]:
        img_resized = cv2.resize(
            gray_base.astype(np.float32), (sz, sz),
            interpolation=cv2.INTER_LINEAR,
        )

        ref_resized = None
        if ref_gray_base is not None:
            ref_resized = cv2.resize(
                ref_gray_base.astype(np.float32), (sz, sz),
                interpolation=cv2.INTER_LINEAR,
            )

        # Apply pipeline at this scale
        img_w = _apply_hann_window(img_resized)
        img_mag = _fft_magnitude(img_w)
        img_mag = _highpass_filter(img_mag)

        if ref_resized is not None:
            ref_w = _apply_hann_window(ref_resized)
            ref_mag = _fft_magnitude(ref_w)
            ref_mag = _highpass_filter(ref_mag)
        else:
            ref_mag = _generate_reference_spectrum(sz, sz)

        lp_img = _log_polar_remap(img_mag, dsize=(sz, sz))
        lp_ref = _log_polar_remap(ref_mag, dsize=(sz, sz))

        shift_y, shift_x, response = _phase_correlate(lp_ref, lp_img)
        rot, sf = _shifts_to_geometry(shift_y, shift_x, sz, sz)

        # 180° ambiguity resolution
        rot_alt = rot + 180.0
        if rot_alt > 180.0:
            rot_alt -= 360.0
        if abs(rot) > MAX_ROTATION_DEG and abs(rot_alt) <= MAX_ROTATION_DEG:
            rot = rot_alt

        sf = max(MIN_SCALE, min(MAX_SCALE, sf))

        # Dimension-aware scale resolution
        reciprocal_sf = 1.0 / sf if sf > 0.01 else 1.0
        reciprocal_sf = max(MIN_SCALE, min(MAX_SCALE, reciprocal_sf))

        img_area = img_orig_hw[0] * img_orig_hw[1]
        ref_area = ref_orig_hw[0] * ref_orig_hw[1]
        dim_ratio = (img_area / ref_area) ** 0.5 if ref_area > 0 else 1.0

        if abs(dim_ratio - 1.0) > 0.03:
            # Use dimension ratio as scale when dimensions are known
            sf = max(MIN_SCALE, min(MAX_SCALE, dim_ratio))
        else:
            if abs(reciprocal_sf - 1.0) < abs(sf - 1.0):
                sf = reciprocal_sf

        conf = min(1.0, max(0.0, response))

        if conf > best[2]:
            best = (rot, sf, conf, response)

    return best


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — High-level geometry estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_geometry_sync(
    attacked: np.ndarray,
    reference: np.ndarray | None = None,
    use_multi_hypothesis: bool = True,
) -> dict:
    """
    Estimate geometric transformation (rotation + scale) of an attacked image
    using Fourier-Mellin synchronization.

    This is the primary entry point for the geometry_sync module. It replaces
    the brute-force candidate search in Phase 5's estimate_geometry().

    Parameters
    ----------
    attacked  : grayscale (H, W) float32 or BGR (H, W, 3) uint8 image
    reference : original image (same formats), or None for self-reference mode
    use_multi_hypothesis : if True, run at multiple scales for robustness

    Returns
    -------
    dict with keys:
        rotation_deg  : float — estimated rotation in degrees
        scale_factor  : float — estimated scale factor
        confidence    : float — calibrated confidence [0, 1]
        response_peak : float — raw phase correlation peak
        method        : str   — "fourier_mellin" or "fourier_mellin_multi"
        elapsed_ms    : float — pipeline runtime in milliseconds
    """
    t0 = time.perf_counter()

    if use_multi_hypothesis:
        rot, sf, conf, peak = _multi_hypothesis_register(
            attacked, reference, n_scales=2,
        )
        method = "fourier_mellin_multi"
    else:
        rot, sf, conf, peak = fourier_mellin_register(attacked, reference)
        method = "fourier_mellin"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "rotation_deg": rot,
        "scale_factor": sf,
        "confidence": conf,
        "response_peak": peak,
        "method": method,
        "elapsed_ms": elapsed_ms,
    }
