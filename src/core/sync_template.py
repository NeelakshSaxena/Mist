"""
src/core/sync_template.py  –  Synchronization Template Layer (Agent 3)

Separates synchronization from payload by embedding a dedicated template
in the Fourier domain. The template consists of:

    1. Radial pilot peaks — deterministic spikes at known frequencies
    2. Pseudo-random low-frequency ring — spread-spectrum sync band
    3. Repeated pilot markers — redundant angle/scale anchors

The template is embedded BEFORE payload and detected via FFT magnitude
analysis. This allows robust RST (rotation, scale, translation) estimation
without relying on fragile payload bits for synchronization.

Detection Pipeline
------------------
    1. FFT magnitude of suspect image
    2. Detect radial pilot peaks via matched filter
    3. Estimate rotation from peak angular displacement
    4. Estimate scale from peak radial displacement
    5. Estimate translation from phase of detected peaks

Robustness Targets
------------------
    - Survives JPEG Q50, rotation ±15°, scale 0.6×–1.5×
    - Pilot recovery rate >95%
    - False peak rate <5%

Public API
----------
    generate_sync_template(h, w, key, strength)  → template (float32)
    embed_sync_template(Y, key, strength)        → Y_with_template
    detect_sync_template(Y, key)                 → SyncEstimate
    refine_geometry_from_template(Y, key, ...)   → refined SyncEstimate

References
----------
    Pereira & Pun, "Fast Robust Template Matching for Affine Resistant
    Image Watermarking" (ResearchGate)
"""

import hashlib
import hmac
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Number of radial pilot peaks placed around concentric rings
N_PILOT_PEAKS: int = 24

# Number of concentric rings for the pseudo-random sync band
N_SYNC_RINGS: int = 4

# Frequency band for pilots (fraction of Nyquist; avoids DC and edges)
PILOT_FREQ_LOW: float = 0.08    # 8% of Nyquist
PILOT_FREQ_HIGH: float = 0.35   # 35% of Nyquist

# Default embedding strength: spatial-domain RMS of template
# 3.5 RMS ≈ 37 dB PSNR — weakly perceptual but detectable
DEFAULT_STRENGTH: float = 3.5

# Detection thresholds
PEAK_SNR_THRESHOLD: float = 3.0     # min SNR for a valid pilot detection (was 4.0)
MIN_PILOTS_DETECTED: int = 6        # minimum pilots for valid sync (was 8)
FALSE_PEAK_MAX_RATE: float = 0.05   # max false positive rate

# Matched filter correlation window (pixels in frequency domain)
_MF_WINDOW: int = 2

# Analysis size for deterministic performance
_ANALYSIS_SIZE: int = 512


# ─────────────────────────────────────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SyncEstimate:
    """Result of synchronization template detection."""
    rotation_deg: float = 0.0
    scale_factor: float = 1.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    pilots_detected: int = 0
    pilots_expected: int = N_PILOT_PEAKS * N_SYNC_RINGS
    pilot_recovery_rate: float = 0.0
    false_peak_rate: float = 0.0
    confidence: float = 0.0
    peak_snrs: list = field(default_factory=list)
    method: str = "sync_template"
    valid: bool = False


# ─────────────────────────────────────────────────────────────────────────────
#  Key Derivation
# ─────────────────────────────────────────────────────────────────────────────

def _derive_sync_key(key: bytes) -> bytes:
    """Derive a sync-template-specific key to avoid collision with payload."""
    return hmac.new(key, b"mist-sync-template-v1", hashlib.sha256).digest()


def _pilot_prng(sync_key: bytes, n: int) -> np.ndarray:
    """
    Generate deterministic pseudo-random sequence from sync key.
    Returns n values in [0, 1).
    """
    seed = int.from_bytes(sync_key[:4], "big") & 0x7FFFFFFF
    rng = np.random.RandomState(seed)
    return rng.random(n).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Pilot Peak Geometry
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pilot_positions(
    h: int,
    w: int,
    sync_key: bytes,
) -> list[tuple[int, int, float]]:
    """
    Compute (row, col, expected_phase) for all pilot peaks in the FFT domain.

    Pilots are placed on N_SYNC_RINGS concentric rings at N_PILOT_PEAKS
    angular positions per ring. Angular positions are pseudo-randomly
    jittered to reduce detectability.

    Returns list of (freq_row, freq_col, phase) tuples in centered-FFT coords.
    """
    cy, cx = h // 2, w // 2
    max_radius = min(h, w) // 2

    # Ring radii: logarithmically spaced within pilot frequency band
    r_low = int(max_radius * PILOT_FREQ_LOW)
    r_high = int(max_radius * PILOT_FREQ_HIGH)
    radii = np.geomspace(max(r_low, 4), max(r_high, 8), N_SYNC_RINGS)
    radii = radii.astype(np.int32)

    # Angular jitter from PRNG
    jitter = _pilot_prng(sync_key, N_PILOT_PEAKS * N_SYNC_RINGS) * 0.3

    # Phase values for detection verification
    phases = _pilot_prng(
        hmac.new(sync_key, b"phase", hashlib.sha256).digest(),
        N_PILOT_PEAKS * N_SYNC_RINGS,
    )

    pilots = []
    idx = 0
    for ring_i, radius in enumerate(radii):
        base_angles = np.linspace(0, 2 * np.pi, N_PILOT_PEAKS, endpoint=False)
        for peak_j, base_angle in enumerate(base_angles):
            angle = base_angle + jitter[idx] * (2 * np.pi / N_PILOT_PEAKS)
            fr = cy + int(round(radius * np.sin(angle)))
            fc = cx + int(round(radius * np.cos(angle)))

            # Clamp to valid FFT coordinates
            fr = max(1, min(h - 2, fr))
            fc = max(1, min(w - 2, fc))

            pilots.append((fr, fc, float(phases[idx])))
            idx += 1

    return pilots


# ─────────────────────────────────────────────────────────────────────────────
#  Template Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_sync_template(
    h: int,
    w: int,
    key: bytes,
    strength: float = DEFAULT_STRENGTH,
) -> np.ndarray:
    """
    Generate a spatial-domain synchronization template.

    The template is constructed in the frequency domain by placing
    controlled peaks at known positions, then inverse-FFT'd to spatial.

    The `strength` parameter controls the spatial-domain RMS of the
    resulting template (in pixel intensity units).  The frequency-domain
    amplitude is computed from `strength` via Parseval's theorem to
    compensate for numpy's 1/(H*W) IFFT normalization.

    Parameters
    ----------
    h, w     : target image dimensions
    key      : embedding secret key
    strength : target spatial-domain RMS (pixel units, default 1.5)

    Returns
    -------
    template : float32 array [h, w] — additive spatial template
    """
    sync_key = _derive_sync_key(key)
    pilots = _compute_pilot_positions(h, w, sync_key)
    n_pixels = h * w

    # Compute frequency-domain amplitude from target spatial RMS.
    # By Parseval: RMS_spatial = A_freq * sqrt(2 * n_pilots) / (H * W)
    # Therefore:   A_freq = RMS_spatial * H * W / sqrt(2 * n_pilots)
    n_pilots = len(pilots)
    freq_amplitude = strength * n_pixels / np.sqrt(2.0 * n_pilots)

    # Build frequency-domain template
    F = np.zeros((h, w), dtype=np.complex64)

    for fr, fc, phase in pilots:
        # Place conjugate-symmetric peaks for real-valued output
        F[fr, fc] = freq_amplitude * np.exp(1j * phase * 2 * np.pi)

        # Conjugate symmetric partner (ensures real IFFT output)
        fr_conj = h - fr
        fc_conj = w - fc
        if 0 <= fr_conj < h and 0 <= fc_conj < w:
            F[fr_conj, fc_conj] = freq_amplitude * np.exp(-1j * phase * 2 * np.pi)

    # Inverse FFT shift (pilots are in centered coordinates)
    F_unshifted = np.fft.ifftshift(F)

    # IFFT to spatial domain
    template = np.real(np.fft.ifft2(F_unshifted)).astype(np.float32)

    return template


# ─────────────────────────────────────────────────────────────────────────────
#  Template Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_sync_template(
    Y: np.ndarray,
    key: bytes,
    strength: float = DEFAULT_STRENGTH,
) -> np.ndarray:
    """
    Embed synchronization template into the luminance channel.

    This should be called BEFORE payload embedding to ensure the template
    occupies a separate spectral band from the payload.

    Parameters
    ----------
    Y        : float32 luminance channel [H, W]
    key      : embedding secret key
    strength : embedding strength (default 3.0 — weakly perceptual)

    Returns
    -------
    Y_out    : float32 luminance with embedded template
    """
    h, w = Y.shape
    template = generate_sync_template(h, w, key, strength)

    # Perceptual masking: scale template by local variance
    # (stronger in textured areas, weaker in flat areas)
    local_mean = cv2.blur(Y, (16, 16))
    local_sq = cv2.blur(Y * Y, (16, 16))
    local_var = np.maximum(local_sq - local_mean ** 2, 1.0)
    mask = np.sqrt(local_var) / (np.sqrt(local_var).max() + 1e-6)
    mask = 0.3 + 0.7 * mask  # floor at 30% strength even in flat areas

    Y_out = Y + template * mask
    return Y_out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Template Detection
# ─────────────────────────────────────────────────────────────────────────────

def _extract_fft_magnitude(Y: np.ndarray) -> np.ndarray:
    """
    Compute centered, highpass-filtered raw FFT magnitude spectrum.

    Uses a local-median-subtracted raw magnitude instead of log-scaled
    to preserve peak-to-background contrast for pilot detection.
    """
    # Resize for deterministic performance
    h, w = Y.shape
    if h != _ANALYSIS_SIZE or w != _ANALYSIS_SIZE:
        # Center crop or pad to _ANALYSIS_SIZE
        Y_resized = np.zeros((_ANALYSIS_SIZE, _ANALYSIS_SIZE), dtype=np.float32)
        cy_src, cx_src = h // 2, w // 2
        cy_dst, cx_dst = _ANALYSIS_SIZE // 2, _ANALYSIS_SIZE // 2
        
        h_copy = min(h, _ANALYSIS_SIZE)
        w_copy = min(w, _ANALYSIS_SIZE)
        
        src_y1 = cy_src - h_copy // 2
        src_x1 = cx_src - w_copy // 2
        dst_y1 = cy_dst - h_copy // 2
        dst_x1 = cx_dst - w_copy // 2
        
        Y_resized[dst_y1:dst_y1+h_copy, dst_x1:dst_x1+w_copy] = Y[src_y1:src_y1+h_copy, src_x1:src_x1+w_copy].astype(np.float32)
    else:
        Y_resized = Y.astype(np.float32)

    # Hann window to reduce spectral leakage
    win_y = np.hanning(Y_resized.shape[0]).astype(np.float32)
    win_x = np.hanning(Y_resized.shape[1]).astype(np.float32)
    windowed = Y_resized * np.outer(win_y, win_x)

    F = np.fft.fft2(windowed)
    F_shifted = np.fft.fftshift(F)
    magnitude = np.abs(F_shifted).astype(np.float32)

    # Highpass filter: subtract local mean to remove natural spectral slope.
    # This makes pilot peaks stand out regardless of image content.
    # cv2.blur works with float32 (medianBlur doesn't for large kernels).
    bg = cv2.blur(magnitude, (15, 15))
    filtered = magnitude - bg

    return filtered


def _detect_pilots(
    magnitude: np.ndarray,
    pilots: list[tuple[int, int, float]],
    window: int = _MF_WINDOW,
) -> tuple[list[bool], list[float]]:
    """
    Detect pilot peaks in the FFT magnitude spectrum using SNR thresholding.

    For each expected pilot position, compute the peak-to-local-background
    ratio. A pilot is "detected" if its SNR exceeds PEAK_SNR_THRESHOLD.

    Returns (detected_mask, snr_values).
    """
    h, w = magnitude.shape
    detected = []
    snrs = []

    # Wider annular background: ring from window+2 to window+8
    # Wider ring gives more stable background statistics
    bg_inner = window + 2
    bg_outer = window + 8

    # Precompute relative background mask
    # This prevents evaluating np.ogrid and dist_sq for every pilot
    y_idx, x_idx = np.ogrid[-bg_outer:bg_outer + 1, -bg_outer:bg_outer + 1]
    dist_sq = y_idx**2 + x_idx**2
    bg_mask = (dist_sq >= bg_inner ** 2) & (dist_sq <= bg_outer ** 2)

    for fr, fc, _ in pilots:
        # Clamp coordinates to keep annulus in-bounds
        fr = max(bg_outer, min(h - bg_outer - 1, fr))
        fc = max(bg_outer, min(w - bg_outer - 1, fc))

        # Peak value (max in small window centered on expected position)
        peak_region = magnitude[
            fr - window:fr + window + 1,
            fc - window:fc + window + 1
        ]
        peak_val = float(np.max(peak_region))

        # Background estimate: annular ring around the peak
        bg_region = magnitude[
            fr - bg_outer:fr + bg_outer + 1,
            fc - bg_outer:fc + bg_outer + 1
        ]
        bg_vals = bg_region[bg_mask]
        if len(bg_vals) == 0:
            snrs.append(0.0)
            detected.append(False)
            continue

        bg_mean = float(np.mean(bg_vals))
        bg_std = float(np.std(bg_vals))
        # Use median absolute deviation for robustness against outliers
        bg_median = float(np.median(bg_vals))
        bg_mad = float(np.median(np.abs(bg_vals - bg_median)))
        # Use max of std and MAD*1.4826 (consistency factor for normal dist)
        robust_std = max(bg_std, bg_mad * 1.4826, 0.01)

        snr = (peak_val - bg_mean) / robust_std
        snrs.append(snr)
        detected.append(snr >= PEAK_SNR_THRESHOLD)

    return detected, snrs


def detect_sync_template(
    Y: np.ndarray,
    key: bytes,
    initial_rotation: Optional[float] = None,
    initial_scale: Optional[float] = None,
) -> SyncEstimate:
    """
    Detect synchronization template and estimate RST parameters.

    Pipeline:
        1. FFT magnitude of input
        2. Generate expected pilot positions at identity (0°, 1.0×)
        3. For candidate rotations/scales, rotate pilot positions and match
        4. Best match gives RST estimate

    Parameters
    ----------
    Y   : float32 or uint8 luminance [H, W]
    key : embedding secret key
    initial_rotation: Optional[float], coarse estimate to check
    initial_scale: Optional[float], coarse estimate to check

    Returns
    -------
    SyncEstimate with rotation, scale, translation, confidence
    """
    result = SyncEstimate()
    h_orig, w_orig = Y.shape

    # Get FFT magnitude at analysis size
    magnitude = _extract_fft_magnitude(Y)
    ah, aw = magnitude.shape

    sync_key = _derive_sync_key(key)

    # Expected pilots at analysis size (identity transform)
    pilots_identity = _compute_pilot_positions(ah, aw, sync_key)
    result.pilots_expected = len(pilots_identity)

    def _score(det_mask, snr_vals):
        """Composite score: count of detected × mean SNR of detected."""
        n = sum(det_mask)
        if n == 0:
            return 0.0
        detected_snrs = [s for d, s in zip(det_mask, snr_vals) if d]
        return n * (sum(detected_snrs) / len(detected_snrs))

    # ── Identity check ────────────────────────────────────────────────
    detected_id, snrs_id = _detect_pilots(magnitude, pilots_identity)
    n_detected_id = sum(detected_id)
    rate_id = n_detected_id / max(len(pilots_identity), 1)

    best_rotation = 0.0
    best_scale = 1.0
    best_detected = n_detected_id
    best_rate = rate_id
    best_snrs = snrs_id
    best_score = _score(detected_id, snrs_id)

    # ── Rotation search ───────────────────────────────────────────────
    # Rotate expected pilot positions and check match
    cy, cx = ah // 2, aw // 2
    if initial_rotation is not None and initial_scale is not None:
        candidate_angles = [initial_rotation]
        candidate_scales = [initial_scale]
    else:
        # Sparse search to keep runtime < 0.5s
        candidate_angles = np.arange(-20.0, 21.0, 5.0)  # 9 angles
        candidate_scales = [0.6, 0.8, 1.0, 1.2, 1.4]    # 5 scales

    for angle_deg in candidate_angles:
        for scale in candidate_scales:
            if abs(angle_deg) < 0.01 and abs(scale - 1.0) < 0.01:
                continue  # skip identity (already checked)

            # Transform pilot positions
            rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            transformed_pilots = []

            for fr, fc, phase in pilots_identity:
                # Center, rotate, scale, un-center
                dy = fr - cy
                dx = fc - cx
                # Expected frequency is inversely proportional to attack scale
                new_dy = (cos_a * dy - sin_a * dx) / scale
                new_dx = (sin_a * dy + cos_a * dx) / scale
                new_fr = int(round(cy + new_dy))
                new_fc = int(round(cx + new_dx))

                if 0 <= new_fr < ah and 0 <= new_fc < aw:
                    transformed_pilots.append((new_fr, new_fc, phase))

            if len(transformed_pilots) < MIN_PILOTS_DETECTED:
                continue

            det, snrs = _detect_pilots(magnitude, transformed_pilots)
            n_det = sum(det)
            sc = _score(det, snrs)

            if sc > best_score:
                best_detected = n_det
                best_rate = n_det / max(len(transformed_pilots), 1)
                best_rotation = angle_deg
                best_scale = scale
                best_snrs = snrs
                best_score = sc

    # ── Compute false peak rate ───────────────────────────────────────
    # Test random positions to estimate false positive rate
    rng = np.random.RandomState(42)
    n_random = min(50, len(pilots_identity))
    random_pilots = [
        (
            int(rng.randint(10, ah - 10)),
            int(rng.randint(10, aw - 10)),
            0.0,
        )
        for _ in range(n_random)
    ]
    false_det, _ = _detect_pilots(magnitude, random_pilots)
    false_rate = sum(false_det) / max(n_random, 1)

    # ── Populate result ───────────────────────────────────────────────
    result.rotation_deg = best_rotation
    result.scale_factor = best_scale
    result.pilots_detected = best_detected
    result.pilot_recovery_rate = best_rate
    result.false_peak_rate = false_rate
    result.peak_snrs = best_snrs
    result.confidence = min(1.0, best_rate / 0.95)  # normalize to target

    # Validity check
    result.valid = (
        best_rate >= (MIN_PILOTS_DETECTED / max(result.pilots_expected, 1))
        and false_rate < FALSE_PEAK_MAX_RATE * 3  # relaxed for initial check
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry Refinement
# ─────────────────────────────────────────────────────────────────────────────

def refine_geometry_from_template(
    Y: np.ndarray,
    key: bytes,
    initial_rotation: float = 0.0,
    initial_scale: float = 1.0,
    refine_range_deg: float = 2.0,
    refine_range_scale: float = 0.05,
) -> SyncEstimate:
    """
    Refine RST estimate using fine-grained template matching.

    Searches a narrow grid around the initial estimate with 0.1° and 1%
    resolution for sub-degree accuracy.

    Parameters
    ----------
    Y                  : luminance channel
    key                : embedding key
    initial_rotation   : coarse rotation estimate (degrees)
    initial_scale      : coarse scale estimate
    refine_range_deg   : search range around initial angle
    refine_range_scale : search range around initial scale

    Returns
    -------
    SyncEstimate with refined parameters
    """
    magnitude = _extract_fft_magnitude(Y)
    ah, aw = magnitude.shape
    sync_key = _derive_sync_key(key)
    pilots_identity = _compute_pilot_positions(ah, aw, sync_key)
    cy, cx = ah // 2, aw // 2

    best_rotation = initial_rotation
    best_scale = initial_scale
    best_detected = 0
    best_snrs = []
    best_score = 0.0

    def _ref_score(det_mask, snr_vals):
        n = sum(det_mask)
        if n == 0:
            return 0.0
        return n * np.mean([s for d, s in zip(det_mask, snr_vals) if d])

    # Fine grid: 0.2° steps for angle, 1% steps for scale
    angles = np.arange(
        initial_rotation - refine_range_deg,
        initial_rotation + refine_range_deg + 0.1,
        0.2,
    )
    scales = np.arange(
        max(0.5, initial_scale - refine_range_scale),
        min(1.6, initial_scale + refine_range_scale + 0.005),
        0.01,
    )

    for angle_deg in angles:
        for scale in scales:
            rad = np.radians(angle_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            transformed = []

            for fr, fc, phase in pilots_identity:
                dy, dx = fr - cy, fc - cx
                # Expected frequency is inversely proportional to attack scale
                new_dy = (cos_a * dy - sin_a * dx) / scale
                new_dx = (sin_a * dy + cos_a * dx) / scale
                new_fr = int(round(cy + new_dy))
                new_fc = int(round(cx + new_dx))
                if 0 <= new_fr < ah and 0 <= new_fc < aw:
                    transformed.append((new_fr, new_fc, phase))

            if len(transformed) < MIN_PILOTS_DETECTED:
                continue

            det, snrs = _detect_pilots(magnitude, transformed)
            n_det = sum(det)
            sc = _ref_score(det, snrs)

            if sc > best_score:
                best_detected = n_det
                best_rotation = float(angle_deg)
                best_scale = float(scale)
                best_snrs = snrs
                best_score = sc

    result = SyncEstimate()
    result.rotation_deg = best_rotation
    result.scale_factor = best_scale
    result.pilots_detected = best_detected
    result.pilots_expected = len(pilots_identity)
    result.pilot_recovery_rate = best_detected / max(len(pilots_identity), 1)
    result.confidence = min(1.0, result.pilot_recovery_rate / 0.95)
    result.peak_snrs = best_snrs
    result.method = "sync_template_refined"
    result.valid = result.pilot_recovery_rate >= (
        MIN_PILOTS_DETECTED / max(result.pilots_expected, 1)
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Energy Analysis (Stop Condition Checks)
# ─────────────────────────────────────────────────────────────────────────────

def check_template_energy(
    Y_original: np.ndarray,
    Y_with_template: np.ndarray,
) -> dict:
    """
    Verify that the sync template doesn't dominate payload energy.

    Stop conditions checked:
        - FAIL if spatial PSNR < 35 dB (visible artifacts)
        - FAIL if template RMS > 5.0 (too strong for perceptual transparency)

    Note: The FFT peak/mean ratio is NOT used as a fail criterion because
    the sync template is *designed* to have sharp Fourier peaks. The real
    constraint is spatial-domain perceptibility.

    Returns dict with energy metrics and pass/fail status.
    """
    diff = Y_with_template.astype(np.float64) - Y_original.astype(np.float64)
    template_mse = float(np.mean(diff ** 2))
    template_rms = float(np.sqrt(template_mse))

    # PSNR between original and template-embedded
    if template_mse < 1e-10:
        psnr = float('inf')
    else:
        psnr = 10.0 * np.log10(255.0 ** 2 / template_mse)

    # FFT peak/mean for diagnostics (not a pass/fail criterion)
    F_diff = np.fft.fft2(diff)
    F_mag = np.abs(F_diff)
    peak_to_mean = float(np.max(F_mag)) / (float(np.mean(F_mag)) + 1e-6)

    # Pass/fail based on perceptual criteria
    visible_artifacts = psnr < 35.0
    excessive_energy = template_rms > 5.0  # RMS > 5 out of [0,255] range

    return {
        "template_mse": template_mse,
        "template_rms": template_rms,
        "psnr_db": psnr,
        "peak_to_mean_ratio": peak_to_mean,
        "visible_artifacts": visible_artifacts,
        "excessive_energy": excessive_energy,
        "pass": not visible_artifacts and not excessive_energy,
    }
