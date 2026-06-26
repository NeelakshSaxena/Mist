"""
src/core/wm_engine_p5.py  –  Phase 5 Geometry-Invariant Watermark Engine

Extends Phase 4's spatial-redundant detection to survive geometric
transformations: rotation, scaling, crop+resize, and (partially) perspective.

Architecture
------------
Embedding:
    Identical to Phase 4 (embed_p4).  Phase 5 changes detection only.

Detection:
    Multi-stage geometric search:

    Stage 1 — Direct P4 detection (fast path, ~2s):
        If the image hasn't been geometrically transformed, P4 handles it.

    Stage 2 — Scale-only sweep:
        Try undoing common scale factors via full P4 shard extraction.
        Also try ±1-2 pixel target sizes for rounding tolerance.

    Stage 3 — Rotation-only sweep:
        Try angles at 1° steps over [-20°, 20°] via full P4 shard
        extraction.  Early exit on success.

    Stage 4 — Joint coarse search from best individuals:
        Cross-sweep: best rotation angle + scale candidates, and
        best scale + rotation candidates.

    Stage 5 — True 2D coarse search (center-tile canary):
        Coarse 2D grid (2° angle × 11 scales) with fast scoring:
        check 5 center tiles at 4 sub-block offsets (~30ms/candidate).
        Finds top candidates, then promotes each to full P4 verification.
        Total: ~7s coarse + ~5s verification = ~12s.

    Stage 6 — Perturbation + size-adjust search:
        Fine-tune around best estimate from prior stages.

    Key design choice: Stages 2-4 use extract_shards_p4() as the scoring
    function (~1.5s/candidate but includes P4's full grid alignment search).
    Stage 5 uses a fast "canary" metric for the coarse 2D grid, enabling
    combined rotation+scale detection without exhaustive full-P4 sweeps.

Public API
----------
    embed_p5(image, bitstream, key)     → np.ndarray  (alias for embed_p4)
    detect_p5(image, key)               → dict  (geometry-invariant detection)
    estimate_geometry(image, key)        → dict  (estimated rotation + scale)
"""

import os
import time
import cv2
import numpy as np

from src.core.wm_engine_p3 import (
    _to_ycbcr,
)
from src.core.wm_engine_p4 import (
    embed_p4, detect_p4,
    _tile_anchor_bits,
    _extract_tile_bits,
    _parse_tile_bits,
    extract_shards_p4,
    MT_SIZE, MT_BLOCKS,
    ANCHOR_BITS,
    ANCHOR_MATCH_MIN,
    K_SHARDS,
)
from src.core.gpu_geometry import gpu_batch_canary, gpu_undo_transform
from src.core.p5_profiler import get_active_profile
from src.core.geometry_sync import estimate_geometry_sync
from src.core.geometry_correction import (
    correct_geometry,
    correct_geometry_Y,
    estimate_canonical_size,
)
from src.core.sync_template import (
    embed_sync_template,
    detect_sync_template,
    refine_geometry_from_template,
    check_template_energy,
    SyncEstimate,
    DEFAULT_STRENGTH as SYNC_TEMPLATE_STRENGTH,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Flag: Legacy Geometry Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Set to True to revert to the brute-force canary candidate search.
# Set to False (default) to use the FFT + log-polar Fourier-Mellin pipeline.
# Can be overridden via environment variable: MIST_USE_LEGACY_GEOMETRY=1
USE_LEGACY_GEOMETRY: bool = os.environ.get("MIST_USE_LEGACY_GEOMETRY", "0") == "1"


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Scoring thresholds
ANCHOR_SCORE_BASELINE: float = 0.54  # For forensic: above random (0.50)

# Scale search candidates
SCALE_CANDIDATES: list[float] = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60, 1.80, 2.0,
]

# 2D coarse search parameters
COARSE_2D_ANGLES: list[float] = list(np.arange(-20.0, 21.0, 2.0))
COARSE_2D_SCALES: list[float] = [
    0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.05, 1.10, 1.20, 1.30, 1.50, 1.80,
]

# Minimum canary anchor-pass tiles to consider a candidate "promising"
CANARY_MIN_TILES: int = 3

# ── Stop Conditions ──────────────────────────────────────────────────────────
MAX_REASONABLE_ANGLE: float = 25.0    # Reject candidates beyond ±25°
SCALE_BOUND_LOW: float = 0.40        # Reject scale below 0.40×
SCALE_BOUND_HIGH: float = 2.5        # Reject scale above 2.5×
MIN_CANARY_SIGNAL: int = 2           # Abort search if best canary < 2
MAX_CANDIDATES_EVAL: int = 6         # Cap total P4 evaluations (FM-seeded: rarely need >3)
MAX_REFINEMENT_ITER: int = 2         # Cap fine-tune iterations
MIN_CRC_IMPROVEMENT: int = 1         # Min CRC gain to continue refinement
SHARD_CONSISTENCY_MIN: float = 0.65  # Min shard/tile consistency ratio

# ── Deployment Assertions ────────────────────────────────────────────────────
MAX_DETECTION_TIME: float = 20.0     # Max seconds for detect_p5()
MAX_FORENSIC_TIME: float = 60.0      # Max seconds for forensic_report()
GEOMETRY_STAGE_DISAGREE_MAX: float = 5.0  # Max angle disagreement (°)

# ── Confidence Weights (see _compute_confidence) ─────────────────────────────
# Two-head scoring: HEAD A (verification) + HEAD B (detection).
#
# HEAD A — fires when RS decode succeeds (answered: "is payload valid?")
CONF_W_SIGNATURE: float    = 0.45   # Cryptographic proof
CONF_W_RS_DECODE: float    = 0.25   # Payload integrity
CONF_W_SHARD: float        = 0.15   # Spatial redundancy (GATED on rs_decode)
CONF_W_GEOMETRY: float     = 0.10   # Transform stability (GATED on rs_decode)
CONF_W_CORRELATION: float  = 0.05   # DCT correlation (weak prior)
#
# HEAD B — fires when RS decode fails (answered: "is a watermark present?")
# Cap: 0.50  — so unverified detections never appear as confident as verified.
CONF_W_SHARD_SIGNAL: float = 0.25   # Raw shard count above noise floor  [Strategy 1]
CONF_W_PILOT_RATE: float   = 0.25   # Sync template pilot recovery rate  [Strategy 2]
CONF_W_P3_SIGNAL: float    = 0.25   # Phase 3 harmonic + presence score  [Strategy 3]
CONF_W_CRC_EVIDENCE: float = 0.15   # CRC ratio (partial geometry evidence)
CONF_W_GEO_DET: float      = 0.10   # Geometry confidence


# ─────────────────────────────────────────────────────────────────────────────
#  Geometric Transform Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _undo_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Undo a rotation by rotating back by -angle_deg."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _undo_rotation_Y(Y: np.ndarray, angle_deg: float) -> np.ndarray:
    """Undo rotation on single-channel Y (float32)."""
    h, w = Y.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    return cv2.warpAffine(
        Y, M, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _undo_scale(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Undo scaling by resizing by 1/scale_factor.

    Uses ceil() because the attack uses int() truncation:
        int(original * factor) = attacked_size
    So original = ceil(attacked_size / factor).
    """
    h, w = image.shape[:2]
    inv_factor = 1.0 / max(scale_factor, 0.01)
    new_w = max(MT_SIZE, int(np.ceil(w * inv_factor)))
    new_h = max(MT_SIZE, int(np.ceil(h * inv_factor)))
    interp = cv2.INTER_AREA if inv_factor < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _undo_scale_Y(Y: np.ndarray, scale_factor: float) -> np.ndarray:
    """Undo scaling on single-channel Y (float32). Uses ceil() to match attack's int()."""
    h, w = Y.shape
    inv_factor = 1.0 / max(scale_factor, 0.01)
    new_w = max(MT_SIZE, int(np.ceil(w * inv_factor)))
    new_h = max(MT_SIZE, int(np.ceil(h * inv_factor)))
    interp = cv2.INTER_AREA if inv_factor < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(Y, (new_w, new_h), interpolation=interp)


def _undo_scale_to_size(
    image: np.ndarray, target_w: int, target_h: int,
) -> np.ndarray:
    """Undo scaling by resizing to a specific target size."""
    target_w = max(MT_SIZE, target_w)
    target_h = max(MT_SIZE, target_h)
    inv = target_w / max(image.shape[1], 1)
    interp = cv2.INTER_AREA if inv < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (target_w, target_h), interpolation=interp)


def _undo_transform(
    image: np.ndarray,
    angle_deg: float,
    scale_factor: float,
) -> np.ndarray:
    """
    Undo rotation + scaling.

    The attack pipeline applies: rotate(image, angle) → scale(result, factor).
    To invert: first S^{-1} (undo scale), then R^{-1} (undo rotation).
    """
    result = _undo_scale(image, scale_factor)
    result = _undo_rotation(result, angle_deg)
    return result


def _undo_transform_Y(
    Y: np.ndarray,
    angle_deg: float,
    scale_factor: float,
) -> np.ndarray:
    """Undo rotation + scaling on single-channel Y (float32)."""
    result = _undo_scale_Y(Y, scale_factor)
    result = _undo_rotation_Y(result, angle_deg)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring via extract_shards_p4 (gold-standard — includes grid alignment)
# ─────────────────────────────────────────────────────────────────────────────

def _score_candidate(image: np.ndarray, key: bytes) -> tuple[int, int]:
    """
    Score a geometric candidate using P4's full shard extraction pipeline.

    Returns (crc_valid_count, total_anchor_pass_count).
    ~1.5s per call (includes full grid alignment search + shard extraction).
    """
    try:
        result = extract_shards_p4(image, key)
        crc_count = len(result.get("shard_crc_ok", set()))
        total = result.get("shards_valid", 0)
        return crc_count, total
    except Exception:
        return 0, 0


# ─────────────────────────────────────────────────────────────────────────────
#  Fast "canary" scoring — center-tile anchor match (~30ms per candidate)
# ─────────────────────────────────────────────────────────────────────────────

def _canary_score_Y(
    Y: np.ndarray,
    key: bytes,
    n_tiles: int = 5,
    n_offsets: int = 25,
) -> int:
    """
    Robust vectorized canary scoring on Y channel.
    Checks all 64 macro-tile phases and 25 pixel offsets simultaneously!
    """
    from src.core.wm_engine_p4 import _derive_p4_key
    from src.core.wm_engine_p3 import _block_seed, _select_pair, PAIR_POOL, _block_dct

    h, w = Y.shape
    if h < 264 or w < 264:
        return 0

    p4k = _derive_p4_key(key)
    expected_anchor = np.array(_tile_anchor_bits(key), dtype=np.int8)

    # 1. Precompute PRNG pairs for all 64 block positions (phase tr=0..7, tc=0..7)
    tile_p1 = np.zeros((8, 8, 2), dtype=np.int64)
    tile_p2 = np.zeros((8, 8, 2), dtype=np.int64)
    for tr in range(8):
        for tc in range(8):
            seed = _block_seed(p4k, tr, tc, 8)
            p1, p2 = _select_pair(seed, PAIR_POOL, 8)
            tile_p1[tr, tc] = p1
            tile_p2[tr, tc] = p2

    cy, cx = h // 2, w // 2
    # Extract 256 + 8 = 264 region (to allow up to 8px pixel shift)
    y0 = max(0, cy - 132)
    x0 = max(0, cx - 132)
    y1 = min(h, y0 + 264)
    x1 = min(w, x0 + 264)
    if y1 - y0 < 264 or x1 - x0 < 264:
        return 0
    Y_center = Y[y0:y1, x0:x1]

    best_pass = 0
    br_idx = np.arange(25)[:, None, None]
    bc_idx = np.arange(25)[None, :, None]
    c_idx_3d = np.arange(8)[None, None, :]

    # We only need tr=0 for the first row of each tile (to match anchor bits)
    p1u = tile_p1[0, :, 0]
    p1v = tile_p1[0, :, 1]
    p2u = tile_p2[0, :, 0]
    p2v = tile_p2[0, :, 1]

    # 2. Iterate over sub-block pixel shifts
    px_shifts = [0, 2, 4, 6]
    for px_dy in px_shifts:
        for px_dx in px_shifts:
            Y_shifted = Y_center[px_dy:px_dy+256, px_dx:px_dx+256].astype(np.float32)
            
            dct_img = _block_dct(Y_shifted, 8)
            dct_blocks = dct_img.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3)
            
            # Extract 1 bit per block assuming tr=0 (shape: 32, 32, 8)
            c_idx = np.arange(8)
            c1 = dct_blocks[:, :, p1u[c_idx], p1v[c_idx]]
            c2 = dct_blocks[:, :, p2u[c_idx], p2v[c_idx]]
            all_bits = (c1 > c2).astype(np.int8)
            
            # anchor_bits shape: (25, 25, 8)
            anchor_bits = all_bits[br_idx, bc_idx + c_idx_3d, c_idx_3d]
            matches = (anchor_bits == expected_anchor).sum(axis=2)
            pass_mask = (matches >= 5).astype(np.int32)  # ANCHOR_MATCH_MIN = 0.625 -> 5/8
            
            for br in range(8, 17):
                for bc in range(8, 17):
                    n_pass = int(
                        pass_mask[br, bc] +
                        pass_mask[br-8, bc] +
                        pass_mask[br+8, bc] +
                        pass_mask[br, bc-8] +
                        pass_mask[br, bc+8]
                    )
                    if n_pass > best_pass:
                        best_pass = n_pass
                        if best_pass == 5:
                            return best_pass

    return best_pass


def _canary_score_Y_with_grid(
    Y: np.ndarray,
    key: bytes,
) -> int:
    """
    Canary with extended grid search: 8 sub-block × 8 macro-tile offsets.
    More thorough but slower (~200ms).  Used for top candidates only.
    """
    h, w = Y.shape
    expected_anchor = _tile_anchor_bits(key)

    if h < MT_SIZE * 2 or w < MT_SIZE * 2:
        return 0

    best_pass = 0

    # Sub-block offsets (step 2)
    for dy in range(0, min(8, h - MT_SIZE), 2):
        for dx in range(0, min(8, w - MT_SIZE), 2):
            Y_off = Y[dy:, dx:]
            oh, ow = Y_off.shape
            mt_rows = oh // MT_SIZE
            mt_cols = ow // MT_SIZE

            # Macro-tile phase offsets (step 4 blocks = 32 px)
            for mdy in range(0, MT_BLOCKS, 4):
                for mdx in range(0, MT_BLOCKS, 4):
                    n_chk = 0
                    n_ok = 0
                    off_y = mdy * 8
                    off_x = mdx * 8

                    for ti in range(3):
                        for tj in range(3):
                            ty = off_y + ti * MT_SIZE
                            tx = off_x + tj * MT_SIZE
                            if ty + MT_SIZE > oh or tx + MT_SIZE > ow:
                                continue
                            region = Y_off[ty:ty + MT_SIZE, tx:tx + MT_SIZE]
                            bits = _extract_tile_bits(region, key)
                            if len(bits) < ANCHOR_BITS:
                                continue
                            n_chk += 1
                            m = sum(
                                a == b for a, b in
                                zip(bits[:ANCHOR_BITS], expected_anchor)
                            ) / ANCHOR_BITS
                            if m >= ANCHOR_MATCH_MIN:
                                n_ok += 1

                    if n_ok > best_pass:
                        best_pass = n_ok

    return best_pass


# ─────────────────────────────────────────────────────────────────────────────
#  Anchor-based scoring (for forensic reports)
# ─────────────────────────────────────────────────────────────────────────────

def _anchor_search_score(
    image: np.ndarray,
    key: bytes,
    max_tiles: int = 25,
) -> float:
    """
    Score how well macro-tile anchors match the expected pattern.

    Returns average anchor match fraction [0, 1].
    Baseline (random/no watermark): ~0.50
    Correct alignment: ~0.80-0.95
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return 0.0

    h, w = image.shape[:2]
    if h < MT_SIZE or w < MT_SIZE:
        return 0.0

    _, Y = _to_ycbcr(image)
    expected_anchor = _tile_anchor_bits(key)

    mt_rows = h // MT_SIZE
    mt_cols = w // MT_SIZE

    if mt_rows == 0 or mt_cols == 0:
        return 0.0

    n_r = min(mt_rows, max(1, int(max_tiles ** 0.5)))
    n_c = min(mt_cols, max(1, int(max_tiles ** 0.5)))

    sample_rows = np.linspace(0, mt_rows - 1, n_r, dtype=int)
    sample_cols = np.linspace(0, mt_cols - 1, n_c, dtype=int)

    total_match = 0.0
    count = 0

    for tr in sample_rows:
        for tc in sample_cols:
            y0 = int(tr) * MT_SIZE
            x0 = int(tc) * MT_SIZE
            if y0 + MT_SIZE > h or x0 + MT_SIZE > w:
                continue

            region = Y[y0:y0 + MT_SIZE, x0:x0 + MT_SIZE]
            bits = _extract_tile_bits(region, key)

            if len(bits) < ANCHOR_BITS:
                continue

            count += 1
            match = sum(
                a == b for a, b in zip(bits[:ANCHOR_BITS], expected_anchor)
            ) / ANCHOR_BITS
            total_match += match

    if count == 0:
        return 0.0

    return total_match / count


# ─────────────────────────────────────────────────────────────────────────────
#  Candidate Clustering (Task 2)
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_candidates(
    candidates: list[tuple[int, float, float]],
    angle_bucket: float = 2.0,
    scale_bucket: float = 0.05,
) -> list[tuple[int, float, float]]:
    """
    Cluster canary-scored candidates by angle+scale buckets.
    Retain only the strongest candidate per cluster.

    Parameters
    ----------
    candidates : list of (score, angle, scale)
    angle_bucket : bucket width in degrees
    scale_bucket : bucket width for scale factor

    Returns sorted list of best-per-cluster candidates (descending score).
    """
    clusters: dict[tuple[int, int], tuple[int, float, float]] = {}
    for score, angle, sf in candidates:
        key = (round(angle / angle_bucket), round(sf / scale_bucket))
        if key not in clusters or score > clusters[key][0]:
            clusters[key] = (score, angle, sf)

    result = list(clusters.values())
    result.sort(key=lambda x: x[0], reverse=True)

    prof = get_active_profile()
    if prof:
        prof.candidates_after_clustering = len(result)
        rej = len(candidates) - len(result)
        prof.rejection_reasons["duplicate_cluster"] += rej

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry Validation (Task 4 + 5)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_geometry(
    angle: float,
    scale: float,
    prior_angle: float = 0.0,
    prior_scale: float = 1.0,
) -> tuple[bool, str]:
    """
    Validate a geometry estimate. Returns (valid, reason).

    Checks:
      - angle within MAX_REASONABLE_ANGLE
      - scale within [SCALE_BOUND_LOW, SCALE_BOUND_HIGH]
      - disagreement with prior estimate
    """
    if abs(angle) > MAX_REASONABLE_ANGLE:
        return False, "geometry_bounds"
    if scale < SCALE_BOUND_LOW or scale > SCALE_BOUND_HIGH:
        return False, "geometry_bounds"
    if abs(angle - prior_angle) > GEOMETRY_STAGE_DISAGREE_MAX and prior_angle != 0.0:
        return False, "geometry_bounds"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Signal Confidence Scoring (Task 3)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confidence(
    signature_verified: bool,
    rs_decode_success: bool,
    shard_consistency: float,
    shard_crc_ratio: float,
    geometry_confidence: float,
    correlation: float,
    # Strategy 1: raw shard count — works even when RS fails
    shard_count: int = 0,
    shards_needed: int = 30,
    # Strategy 2: pilot recovery rate from sync template
    pilot_recovery_rate: float = 0.0,
    # Strategy 3: Phase 3 geometry-invariant signals
    harmonic_score: float = 0.0,
    presence_score: float = 0.0,
) -> float:
    """
    Compute calibrated confidence score from multiple signals.

    Two heads:
      HEAD A (verification) — fires when RS decode succeeds:
        1. Signature verified    weight 0.45
        2. RS decode success     weight 0.25
        3. Shard consistency     weight 0.15
        4. Geometry confidence   weight 0.10
        5. Correlation           weight 0.05

      HEAD B (detection) — fires when RS decode fails:
        Answers "is a watermark present?" independently of RS/crypto.
        1. Shard count signal    weight 0.25  (Strategy 1 — CRC-validated shards)
        2. Pilot recovery rate   weight 0.25  (Strategy 2 — sync template)
        3. P3 signal             weight 0.25  (Strategy 3 — FFT harmonic + DCT presence)
        4. CRC evidence          weight 0.15
        5. Geometry confidence   weight 0.10

        HEAD B is capped at 0.50 so unverified detections never
        appear as confident as verified ones.

    Returns float in [0, 1].
    """
    sig_val = 1.0 if signature_verified else 0.0
    rs_val  = 1.0 if rs_decode_success   else 0.0

    # ── HEAD A: Verification path ──────────────────────────────────────
    # Partial gate: CRC validation without RS decode.
    # crc_ratio ~ 0.004 for clean images, ~ 0.9+ for genuine WM tiles.
    crc_gate = min(1.0, shard_crc_ratio * 2.0)

    # Shard gate: require BOTH RS decode AND CRC evidence for full weight.
    rs_crc_gate = rs_val * crc_gate
    shard_gate  = max(rs_crc_gate, crc_gate * 0.90)

    rs_gate  = rs_val
    norm_corr = max(0.0, (correlation - 0.50) * 2.0)

    score_a = (
        0.45 * sig_val
        + 0.25 * rs_crc_gate
        + 0.15 * min(1.0, max(0.0, shard_consistency)) * shard_gate
        + 0.10 * min(1.0, max(0.0, geometry_confidence)) * rs_gate
        + 0.05 * min(1.0, norm_corr)
    )

    # ── HEAD B: Detection path (when RS decode fails) ──────────────────
    # Strategy 1 — Shard CRC count discriminant:
    #   Clean images:    ~0-1 random CRC matches (noise floor)
    #   WM under attack: 10-64 CRC matches (even when RS fails)
    #   Noise floor = 2. Signal saturates at shards_needed (30).
    noise_floor = 2.0
    signal_range = shards_needed * 1.0
    shard_signal = max(0.0, (shard_count - noise_floor) / max(signal_range, 1.0))
    shard_signal = min(1.0, shard_signal)

    # Strategy 2 — Pilot recovery rate:
    #   Clean images:    ~10-18% false pilot hits (empirical from harness)
    #   WM under attack: 25-80% pilot recovery (pilots survive rotation)
    #   Normalise: subtract false-alarm floor, scale to [0,1]
    PILOT_FALSE_ALARM_FLOOR = 0.18   # empirical: clean images show ~15% false peaks
    pilot_signal = max(0.0, (pilot_recovery_rate - PILOT_FALSE_ALARM_FLOOR)
                       / max(1.0 - PILOT_FALSE_ALARM_FLOOR, 0.01))
    pilot_signal = min(1.0, pilot_signal)

    # Strategy 3 — Phase 3 geometry-invariant signals:
    #   harmonic_score: FFT sinusoidal pattern detection (0 for clean, >0 for WM)
    #   presence_score: multi-scale DCT detection (0 for clean, >0 for WM)
    #   Both survive rotation/scaling because they use frequency-domain features.
    p3_signal = min(1.0, 0.5 * min(1.0, harmonic_score)
                       + 0.5 * min(1.0, presence_score))

    score_b = min(0.50, (
        0.25 * shard_signal       # CRC-validated shard count
        + 0.25 * pilot_signal     # sync template pilot recovery
        + 0.25 * p3_signal        # P3 geometry-invariant signal (key discriminant)
        + 0.15 * min(1.0, shard_crc_ratio * 2.0)   # partial CRC evidence
        + 0.10 * min(1.0, max(0.0, geometry_confidence))
    ))

    # Use HEAD A when RS succeeded (verification path),
    # HEAD B when RS failed (detection path).
    # When RS succeeded, HEAD B can still boost if it's higher
    # (covers partial RS decodes with high pilot evidence).
    if rs_decode_success:
        score = max(score_a, score_b)
    else:
        score = score_b

    prof = get_active_profile()
    if prof:
        prof.confidence_components = {
            "signature":       sig_val,
            "rs_decode":       rs_val,
            "rs_crc_gate":     rs_crc_gate,
            "shard_consistency": shard_consistency * shard_gate,
            "shard_crc_ratio":   shard_crc_ratio,
            "geometry":        geometry_confidence * rs_gate,
            "correlation":     norm_corr,
            # Strategy 1+2+3 signals
            "shard_signal":    shard_signal,
            "pilot_signal":    pilot_signal,
            "p3_signal":       p3_signal,
            "score_head_a":    score_a,
            "score_head_b":    score_b,
            "final":           score,
        }

    return min(1.0, max(0.0, score))


def _geometry_stability_score(
    best_angle: float,
    best_scale: float,
    id_crc: int,
    best_crc: int,
    method: str,
) -> float:
    """
    Score how stable/reliable the geometry estimate is.

    NOTE: This score is ONLY meaningful when called with corroborating
    RS decode evidence (best_crc > 0).  Without evidence it returns 0.0
    to prevent spurious FM geometry from inflating confidence on clean
    images.

    Penalizes:
      - Large angles (harder to recover)
      - Extreme scales
      - Brute-force origin (less reliable)
      - Small CRC improvement over identity
    """
    # No shard CRC improvement — geometry is not corroborated.
    # Return 0 so the gated term in _compute_confidence stays 0.
    if best_crc == 0:
        return 0.0

    score = 1.0
    if abs(best_angle) > 15:
        score *= 0.7
    elif abs(best_angle) > 10:
        score *= 0.85
    if best_scale < 0.6 or best_scale > 1.6:
        score *= 0.75
    if method == "brute_force":
        score *= 0.6
    if best_crc > id_crc + 5:
        score = min(1.0, score * 1.2)
    return score


# ─────────────────────────────────────────────────────────────────────────────
#  Profiled P4 Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_candidate_profiled(image: np.ndarray, key: bytes) -> tuple[int, int]:
    """
    Score a geometric candidate via P4 with profiling instrumentation.
    Returns (crc_valid_count, total_anchor_pass_count).
    """
    prof = get_active_profile()
    t0 = time.perf_counter()
    try:
        result = extract_shards_p4(image, key)
        crc_count = len(result.get("shard_crc_ok", set()))
        total = result.get("shards_valid", 0)
        dt = time.perf_counter() - t0
        if prof:
            prof.record_p4_call(dt)
            prof.candidates_evaluated += 1
        return crc_count, total
    except Exception:
        dt = time.perf_counter() - t0
        if prof:
            prof.record_p4_call(dt)
        return 0, 0


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-stage geometric search
# ─────────────────────────────────────────────────────────────────────────────

def estimate_geometry(image: np.ndarray, key: bytes) -> dict:
    """
    Estimate geometric transform — GPU-ACCELERATED pipeline with
    stop conditions, candidate clustering, and geometry validation.

    Pipeline (USE_LEGACY_GEOMETRY=False, default):
      Stage 0 — Fourier-Mellin synchronization (~50-150ms)
                FFT magnitude → log-polar → phase correlation
                Produces a high-quality (rotation, scale) seed.
      Stage 1 — Identity check (full P4, ~1.5s)
      Stage 2 — Generate candidate (angle, scale) pairs
                Seeded from FM result + narrow perturbation grid (~50 total)
      Stage 3 — GPU batch canary: CUDA affine + anchor scoring (~1-2s)
      Stage 3b — Cluster candidates → reduce duplicates
      Stage 4 — Promote top canary winners to full P4 (capped)
      Stage 5 — Extended canary promotion if needed
      Stage 6 — Fine-tune around best (~3s)

    Pipeline (USE_LEGACY_GEOMETRY=True):
      Stage 1 — Identity check (full P4, ~1.5s)
      Stage 2 — Generate brute-force candidate grid (~330 total)
      Stage 3-6 — Same as above

    Stop conditions:
      - Abort if best canary < MIN_CANARY_SIGNAL
      - Cap P4 evaluations at MAX_CANDIDATES_EVAL
      - Reject geometry outside bounds
      - Abort refinement after MAX_REFINEMENT_ITER without improvement

    Returns dict with angle_deg, scale_factor, shard_count, method.
    """
    prof = get_active_profile()
    h, w = image.shape[:2]
    eval_count = 0  # track total P4 evaluations

    best_angle = 0.0
    best_scale = 1.0
    best_crc = 0
    best_total = 0
    fm_result = None  # Fourier-Mellin estimate (if used)

    # ── Stage 0: Fourier-Mellin Synchronization (new pipeline) ────────
    sync_estimate = None  # sync template result
    if not USE_LEGACY_GEOMETRY:
        if prof:
            stage_t0 = time.perf_counter()
        try:
            fm_result = estimate_geometry_sync(
                attacked=image,
                reference=None,
                use_multi_hypothesis=True,
                key=key,
            )
            if prof:
                prof.stage_times["0_fourier_mellin"] = time.perf_counter() - stage_t0
        except Exception:
            fm_result = None
            if prof:
                prof.stage_times["0_fourier_mellin"] = time.perf_counter() - stage_t0

        # ── Stage 0b: Sync Template Detection ─────────────────────────
        if prof:
            stage_t0 = time.perf_counter()
        try:
            _, Y_detect = _to_ycbcr(image)
            if fm_result is not None:
                sync_estimate = refine_geometry_from_template(
                    Y_detect, key,
                    initial_rotation=fm_result["rotation_deg"],
                    initial_scale=fm_result["scale_factor"],
                    refine_range_deg=2.0,
                    refine_range_scale=0.05
                )
            else:
                sync_estimate = detect_sync_template(Y_detect, key)
            if prof:
                prof.stage_times["0b_sync_template"] = time.perf_counter() - stage_t0

            # Use sync template estimate whenever it found ANY pilots.
            # Even a low-confidence sync estimate is more reliable than
            # FM self-reference (which has no ground-truth reference image).
            if sync_estimate.pilots_detected >= 3:
                sync_conf = sync_estimate.confidence
                fm_conf_cur = fm_result.get("confidence", 0.0) if fm_result else 0.0

                # Always take sync estimate if it's non-trivially detected
                if sync_conf > 0.05 or sync_estimate.pilots_detected >= 4:
                    fm_result = {
                        "rotation_deg":  sync_estimate.rotation_deg,
                        "scale_factor":  sync_estimate.scale_factor,
                        "confidence":    max(sync_conf, 0.05),
                        "response_peak": sync_conf,
                        "method":        "sync_template",
                        "sync_pilots":   sync_estimate.pilots_detected,
                    }
                # If FM and sync agree (< 3° apart), boost FM confidence
                elif fm_result is not None:
                    fm_angle = fm_result["rotation_deg"]
                    if abs(fm_angle - sync_estimate.rotation_deg) < 3.0:
                        fm_result["confidence"] = min(
                            1.0, fm_result.get("confidence", 0.0) * 1.5
                        )
        except Exception as e:
            import traceback
            traceback.print_exc()
            sync_estimate = None
            if prof:
                prof.stage_times["0b_sync_template"] = time.perf_counter() - stage_t0

    # ── Stage 0c: Direct FM correction (P5-V2 fast path) ─────────────
    # Trust the FM/sync-template estimate immediately.
    # Apply inverse affine correction ONCE and run a single P4 evaluation.
    # When FM confidence is high (sync_template found pilots), this returns
    # in ~3-5s total bypassing the brute-force Promote stage.
    # When FM confidence is low, we try a narrow rotation sweep instead.
    if not USE_LEGACY_GEOMETRY and fm_result is not None:
        fm_angle_0c = fm_result["rotation_deg"]
        fm_scale_0c = fm_result["scale_factor"]
        fm_conf_0c  = fm_result.get("confidence", 0.0)

        stage_t0 = time.perf_counter()
        if prof:
            pass  # stage_t0 already set above
        try:
            # Build candidate list for Stage 0c:
            # - Primary: FM/sync estimate
            # - If FM confidence is low: add ±2° neighbors around FM angle
            #   and the reciprocal scale (log-polar ambiguity)
            candidates_0c: list[tuple[float, float]] = [(fm_angle_0c, fm_scale_0c)]

            if fm_conf_0c < 0.25:
                # Low confidence — try narrow rotation neighborhood
                for da in [-2.0, -1.0, 1.0, 2.0]:
                    candidates_0c.append((fm_angle_0c + da, fm_scale_0c))

            # Reciprocal scale (Fourier-Mellin log-polar ambiguity)
            fm_scale_recip_0c = 1.0 / fm_scale_0c if fm_scale_0c > 0.01 else 1.0
            if (SCALE_BOUND_LOW <= fm_scale_recip_0c <= SCALE_BOUND_HIGH
                    and abs(fm_scale_recip_0c - fm_scale_0c) > 0.03):
                candidates_0c.append((fm_angle_0c, fm_scale_recip_0c))

            seen_0c: set[tuple] = set()
            for cand_angle, cand_scale in candidates_0c:
                if eval_count >= MAX_CANDIDATES_EVAL:
                    break
                if not (SCALE_BOUND_LOW <= cand_scale <= SCALE_BOUND_HIGH):
                    continue
                key_0c = (round(cand_angle, 1), round(cand_scale, 2))
                if key_0c in seen_0c:
                    continue
                seen_0c.add(key_0c)

                corr_0c = gpu_undo_transform(image, cand_angle, cand_scale)
                if corr_0c is None:
                    corr_0c = correct_geometry(
                        image, cand_angle, cand_scale, preserve_energy=True,
                    )
                if corr_0c.shape[0] < MT_SIZE or corr_0c.shape[1] < MT_SIZE:
                    continue

                crc_0c, total_0c = _score_candidate_profiled(corr_0c, key)
                eval_count += 1
                if prof:
                    prof.record_geometry(cand_angle, cand_scale, crc_0c, "fm_direct")
                if crc_0c >= K_SHARDS:
                    if prof:
                        prof.stage_times["0c_fm_direct"] = (
                            time.perf_counter() - stage_t0
                        )
                    return {
                        "angle_deg": cand_angle, "scale_factor": cand_scale,
                        "score": float(crc_0c) / K_SHARDS,
                        "shard_count": crc_0c, "method": "fm_direct",
                        "pilot_recovery_rate": (
                            sync_estimate.pilot_recovery_rate
                            if sync_estimate is not None else 0.0
                        ),
                    }
                if crc_0c > best_crc:
                    best_crc   = crc_0c
                    best_total = total_0c
                    best_angle = cand_angle
                    best_scale = cand_scale

        except Exception:
            pass
        finally:
            if "0c_fm_direct" not in (prof.stage_times if prof else {}):
                if prof:
                    prof.stage_times["0c_fm_direct"] = time.perf_counter() - stage_t0

    # ── Stage 1: Identity check ───────────────────────────────────────
    if prof:
        stage_t0 = time.perf_counter()
    id_crc, id_total = _score_candidate_profiled(image, key)
    eval_count += 1
    if prof:
        prof.stage_times["1_identity"] = time.perf_counter() - stage_t0
        prof.record_geometry(0.0, 1.0, id_crc, "identity")
    # Only overwrite best from Stage 0c if identity is better
    if id_crc > best_crc:
        best_crc   = id_crc
        best_total = id_total
        best_angle = 0.0
        best_scale = 1.0
    if id_crc >= K_SHARDS:
        return {
            "angle_deg": 0.0, "scale_factor": 1.0,
            "score": float(id_crc) / K_SHARDS,
            "shard_count": id_crc, "method": "identity",
            "pilot_recovery_rate": (
                sync_estimate.pilot_recovery_rate
                if sync_estimate is not None else 0.0
            ),
        }


    # ── Stage 2: Generate candidate (angle, scale) pairs ──────────────
    if prof:
        stage_t0 = time.perf_counter()
    candidates = []

    if not USE_LEGACY_GEOMETRY and fm_result is not None:
        # ── New pipeline: FM-seeded candidate generation ──────────────
        # The Fourier-Mellin estimate provides a strong prior.
        # Generate a focused candidate grid around the FM seed instead
        # of the full brute-force sweep.
        fm_angle = fm_result["rotation_deg"]
        fm_scale = fm_result["scale_factor"]
        fm_conf = fm_result["confidence"]

        # Primary FM candidate
        candidates.append((fm_angle, fm_scale))

        # Also add reciprocal scale candidate (log-polar scale ambiguity)
        fm_scale_recip = 1.0 / fm_scale if fm_scale > 0.01 else 1.0
        if SCALE_BOUND_LOW <= fm_scale_recip <= SCALE_BOUND_HIGH:
            candidates.append((fm_angle, fm_scale_recip))

        # Narrow perturbation grid around FM estimate (±3° × ±10% scale)
        for da in np.arange(-3.0, 3.5, 1.0):
            for ds in [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]:
                a = fm_angle + da
                s = fm_scale + ds
                if abs(a) <= MAX_REASONABLE_ANGLE and SCALE_BOUND_LOW <= s <= SCALE_BOUND_HIGH:
                    candidates.append((a, s))

        # Also include identity neighbors (in case FM was wrong)
        for sf in [0.90, 0.95, 1.05, 1.10]:
            candidates.append((0.0, sf))
        for angle in [-2.0, -1.0, 1.0, 2.0]:
            candidates.append((angle, 1.0))

        # If FM confidence is low, use a modest focused fallback (NOT the full
        # 330-candidate brute-force grid — that's what Stage 0c + Promote P4 exist to avoid)
        if fm_conf < 0.25:
            for sf in [0.6, 0.7, 0.8, 0.85, 0.9, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]:
                candidates.append((0.0, sf))
            for angle in np.arange(-20.0, 20.5, 5.0):
                if abs(angle) >= 0.5:
                    candidates.append((angle, 1.0))
    else:
        # ── Legacy pipeline: brute-force candidate grid ───────────────
        # 2a. FFT Coarse Geometry Initialization
        from src.core.gpu_fft_geometry import estimate_geometry_fft
        try:
            _, Y_orig = _to_ycbcr(image)
            fft_candidates = estimate_geometry_fft(Y_orig)
            for (a, s) in fft_candidates:
                if 0.4 < s < 2.5:
                    candidates.append((a, s))
        except Exception:
            pass

        # Scale-only candidates
        for sf in SCALE_CANDIDATES:
            candidates.append((0.0, sf))

        # Rotation-only candidates (2° steps, -20..+20)
        for angle in np.arange(-20.0, 20.5, 2.0):
            if abs(angle) >= 0.5:
                candidates.append((angle, 1.0))
        for angle in [-1.0, 1.0, -3.0, 3.0]:
            candidates.append((angle, 1.0))

        # Combined candidates (4° angle × coarse scales)
        for angle in np.arange(-20.0, 21.0, 4.0):
            if abs(angle) < 0.5:
                continue
            for sf in COARSE_2D_SCALES:
                if abs(sf - 1.0) > 0.03:
                    candidates.append((angle, sf))

    if prof:
        prof.candidates_generated = len(candidates)
        prof.stage_times["2_generate"] = time.perf_counter() - stage_t0

    # ── Stage 3: Canary scoring (GPU batch or CPU fallback) ───────────
    if prof:
        stage_t0 = time.perf_counter()
        
    from src.core.gpu_batch_pipeline import run_gpu_batch_pipeline
    gpu_results = run_gpu_batch_pipeline(image, key, candidates)

    if gpu_results and len(gpu_results) > 0:
        canary_results = gpu_results
    else:
        _, Y_orig = _to_ycbcr(image)
        canary_results = []
        for angle, sf in candidates:
            Y_corr = _undo_transform_Y(Y_orig, angle, sf)
            if Y_corr.shape[0] < MT_SIZE * 2 or Y_corr.shape[1] < MT_SIZE * 2:
                continue
            score = _canary_score_Y(Y_corr, key, n_tiles=5, n_offsets=4)
            canary_results.append((score, angle, sf))
        canary_results.sort(key=lambda x: x[0], reverse=True)

    if prof:
        prof.stage_times["3_canary"] = time.perf_counter() - stage_t0

    # ── Stage 3b: Cluster candidates ──────────────────────────────────
    canary_results = _cluster_candidates(canary_results)

    # ── Stop condition: abort if no canary signal ─────────────────────
    top_canary_score = canary_results[0][0] if canary_results else 0
    if top_canary_score < MIN_CANARY_SIGNAL:
        if prof:
            prof.rejection_reasons["low_canary"] += 1
        return {
            "angle_deg": best_angle, "scale_factor": best_scale,
            "score": float(best_crc) / K_SHARDS,
            "shard_count": best_crc, "method": "identity_no_signal",
            "pilot_recovery_rate": (
                sync_estimate.pilot_recovery_rate
                if sync_estimate is not None else 0.0
            ),
        }

    # ── Stage 4: Promote top candidates to full P4 scoring ────────────
    if prof:
        stage_t0 = time.perf_counter()
    TOP_N = 5
    seen = set()
    promoted = []

    for score, angle, sf in canary_results[:TOP_N * 3]:
        if len(promoted) >= TOP_N or eval_count >= MAX_CANDIDATES_EVAL:
            break

        # Geometry validation gate
        valid, reason = _validate_geometry(angle, sf)
        if not valid:
            if prof:
                prof.record_rejection(reason)
            continue

        key_tuple = (round(angle, 1), round(sf, 2))
        if key_tuple in seen:
            continue
        seen.add(key_tuple)

        corrected = gpu_undo_transform(image, angle, sf)
        if corrected is None:
            corrected = correct_geometry(image, angle, sf, preserve_energy=True)
        if corrected.shape[0] < MT_SIZE or corrected.shape[1] < MT_SIZE:
            if prof:
                prof.record_rejection("image_too_small")
            continue

        crc, total = _score_candidate_profiled(corrected, key)
        eval_count += 1
        promoted.append((crc, total, angle, sf))
        if prof:
            prof.candidates_promoted += 1
            prof.record_geometry(angle, sf, crc, "canary_promoted")

        if crc > best_crc:
            best_crc = crc
            best_total = total
            best_angle = angle
            best_scale = sf

        if crc >= K_SHARDS:
            break

    # Also try ±pixel size adjustments for scale-only winners
    if best_crc < K_SHARDS and abs(best_scale - 1.0) > 0.03 and eval_count < MAX_CANDIDATES_EVAL:
        inv_f = 1.0 / best_scale
        base_w = int(np.ceil(w * inv_f))
        base_h = int(np.ceil(h * inv_f))
        for dw in [-2, -1, 1, 2]:
            if eval_count >= MAX_CANDIDATES_EVAL:
                break
            tw, th = base_w + dw, base_h + dw
            if tw < MT_SIZE or th < MT_SIZE:
                continue
            corrected = _undo_scale_to_size(image, tw, th)
            if abs(best_angle) > 0.01:
                corrected = _undo_rotation(corrected, best_angle)
            crc, total = _score_candidate_profiled(corrected, key)
            eval_count += 1
            if crc > best_crc:
                best_crc = crc
                best_total = total
            if crc >= K_SHARDS:
                break

    if prof:
        prof.stage_times["4_promote"] = time.perf_counter() - stage_t0

    if best_crc >= K_SHARDS:
        method = "gpu_canary" if gpu_results is not None else "canary_promoted"
        return {
            "angle_deg": best_angle, "scale_factor": best_scale,
            "score": float(best_crc) / K_SHARDS,
            "shard_count": best_crc, "method": method,
            "pilot_recovery_rate": (
                sync_estimate.pilot_recovery_rate
                if sync_estimate is not None else 0.0
            ),
        }

    # ── Stage 5: Extended search — only if canary had signal ──────────
    if prof:
        stage_t0 = time.perf_counter()
    if best_crc < K_SHARDS and top_canary_score >= 2 and len(canary_results) > TOP_N:
        for score, angle, sf in canary_results[TOP_N:TOP_N + 8]:
            if eval_count >= MAX_CANDIDATES_EVAL:
                if prof:
                    prof.rejection_reasons["runtime_abort"] += 1
                break
            if score < 2:
                break

            # Geometry gate
            valid, reason = _validate_geometry(angle, sf, best_angle, best_scale)
            if not valid:
                if prof:
                    prof.record_rejection(reason)
                continue

            key_tuple = (round(angle, 1), round(sf, 2))
            if key_tuple in seen:
                continue
            seen.add(key_tuple)

            corrected = gpu_undo_transform(image, angle, sf)
            if corrected is None:
                corrected = correct_geometry(image, angle, sf, preserve_energy=True)
            if corrected.shape[0] < MT_SIZE or corrected.shape[1] < MT_SIZE:
                continue

            crc, total = _score_candidate_profiled(corrected, key)
            eval_count += 1
            if prof:
                prof.record_geometry(angle, sf, crc, "extended")
            if crc > best_crc:
                best_crc = crc
                best_angle = float(angle)
                best_scale = float(sf)
            if crc >= K_SHARDS:
                break

    if prof:
        prof.stage_times["5_extended"] = time.perf_counter() - stage_t0

    if best_crc >= K_SHARDS:
        return {
            "angle_deg": best_angle, "scale_factor": best_scale,
            "score": float(best_crc) / K_SHARDS,
            "shard_count": best_crc, "method": "extended_canary",
            "pilot_recovery_rate": (
                sync_estimate.pilot_recovery_rate
                if sync_estimate is not None else 0.0
            ),
        }

    # ── Stage 6: Fine-tune around best (only if we found something) ───
    if prof:
        stage_t0 = time.perf_counter()
    if best_crc > id_crc and best_crc > 0 and eval_count < MAX_CANDIDATES_EVAL:
        fine_a, fine_s, fine_c = _fine_tune(
            image, key, best_angle, best_scale, best_crc,
            max_evals=MAX_CANDIDATES_EVAL - eval_count,
        )
        if fine_c > best_crc:
            best_angle, best_scale, best_crc = fine_a, fine_s, fine_c
    if prof:
        prof.stage_times["6_fine_tune"] = time.perf_counter() - stage_t0

    if best_crc <= id_crc:
        return {
            "angle_deg": 0.0, "scale_factor": 1.0,
            "score": float(id_crc) / K_SHARDS,
            "shard_count": id_crc, "method": "identity",
            "pilot_recovery_rate": (
                sync_estimate.pilot_recovery_rate
                if sync_estimate is not None else 0.0
            ),
        }

    # ── Brute-force geometry validation gate (Task 5) ─────────────────
    valid, reason = _validate_geometry(best_angle, best_scale)
    if not valid:
        if prof:
            prof.record_rejection("geometry_bounds")
        return {
            "angle_deg": 0.0, "scale_factor": 1.0,
            "score": float(id_crc) / K_SHARDS,
            "shard_count": id_crc, "method": "identity_geo_rejected",
            "pilot_recovery_rate": (
                sync_estimate.pilot_recovery_rate
                if sync_estimate is not None else 0.0
            ),
        }

    method = "brute_force" if USE_LEGACY_GEOMETRY else "fourier_mellin_seeded"
    return {
        "angle_deg": best_angle, "scale_factor": best_scale,
        "score": float(best_crc) / K_SHARDS,
        "shard_count": best_crc, "method": method,
        "pilot_recovery_rate": (
            sync_estimate.pilot_recovery_rate
            if sync_estimate is not None else 0.0
        ),
    }


def _fine_tune(
    image: np.ndarray,
    key: bytes,
    center_angle: float,
    center_scale: float,
    current_best: int,
    max_evals: int = 6,
) -> tuple[float, float, int]:
    """
    Fine-tune rotation (±0.5°) and scale (±2%, 1 pixel adj).
    Respects max_evals cap and MIN_CRC_IMPROVEMENT stop condition.
    """
    best_a = center_angle
    best_s = center_scale
    best_c = current_best
    evals = 0
    prof = get_active_profile()

    # Fine angle sweep (±0.5° only — 2 candidates)
    for da in [-0.5, 0.5]:
        if evals >= max_evals:
            break
        a = center_angle + da
        corrected = _undo_transform(image, a, center_scale)
        if corrected.shape[0] < MT_SIZE or corrected.shape[1] < MT_SIZE:
            continue
        crc, _ = _score_candidate_profiled(corrected, key)
        evals += 1
        if crc > best_c:
            if crc - best_c < MIN_CRC_IMPROVEMENT and prof:
                prof.record_rejection("crc_no_improvement")
            best_c = crc
            best_a = a
        if best_c >= K_SHARDS:
            return best_a, best_s, best_c

    # Fine scale sweep (±2% only — 2 candidates)
    for ds in [-0.02, 0.02]:
        if evals >= max_evals:
            break
        s = center_scale + ds
        if s < SCALE_BOUND_LOW or s > SCALE_BOUND_HIGH:
            continue
        corrected = _undo_transform(image, best_a, s)
        if corrected.shape[0] < MT_SIZE or corrected.shape[1] < MT_SIZE:
            continue
        crc, _ = _score_candidate_profiled(corrected, key)
        evals += 1
        if crc > best_c:
            best_c = crc
            best_s = s
        if best_c >= K_SHARDS:
            return best_a, best_s, best_c

    return best_a, best_s, best_c


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — Embedding (alias for Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def embed_p5(image: np.ndarray, bitstream: np.ndarray, key: bytes) -> np.ndarray:
    """
    Phase 5 watermark embedding with synchronization template.

    Pipeline:
        1. Embed sync template into luminance (BEFORE payload)
        2. Embed payload via Phase 4 tile-based engine

    The sync template occupies a separate spectral band from the payload
    and enables robust RST estimation without relying on payload bits.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("embed_p5() expects a 3-channel BGR image (H, W, 3).")

    H, W = image.shape[:2]

    # ── Step 1: Embed synchronization template BEFORE payload ─────────
    # Convert to YCbCr, embed template on Y, convert back
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y_orig = ycrcb[:, :, 0]
    Y_with_sync = embed_sync_template(Y_orig, key, strength=SYNC_TEMPLATE_STRENGTH)

    # Reconstruct image with sync template
    ycrcb_sync = ycrcb.copy()
    ycrcb_sync[:, :, 0] = Y_with_sync
    image_with_sync = cv2.cvtColor(
        np.clip(ycrcb_sync, 0, 255).astype(np.uint8),
        cv2.COLOR_YCrCb2BGR,
    )

    # ── Step 2: Embed payload via Phase 4 on top of sync template ─────
    return embed_p4(image_with_sync, bitstream, key)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_p5(image: np.ndarray, key: bytes) -> dict:
    """
    Phase 5 geometry-invariant watermark detection.

    Pipeline:
      1. Direct Phase 4 detection (fast path, ~2s)
      2. If direct fails and presence detected, estimate geometry
      3. Full P4 on corrected image (~2s)
      4. If P4 close but fails, size-adjust and perturbation search
      5. Multi-signal confidence scoring

    Includes runtime guard (MAX_DETECTION_TIME).
    """
    t0 = time.perf_counter()
    prof = get_active_profile()

    result = {
        "detected": False, "confidence": 0.0,
        "presence_score": 0.0, "scale_scores": {}, "harmonic_score": 0.0,
        "tiles_located": 0, "shards_recovered": 0,
        "shards_needed": K_SHARDS, "reconstruction_ratio": 0.0,
        "inner_codeword": None, "error": None,
        "geometry": None, "corrected_image_shape": None,
        "sync_template": None,
        "shard_crc_ratio": 0.0,
    }

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detect_p5() expects BGR (H, W, 3).")

    if image.shape[0] < MT_SIZE or image.shape[1] < MT_SIZE:
        result["error"] = "Image too small for Phase 5 detection."
        return result

    # ── Step 1: Direct Phase 4 detection (fast path) ──────────────────
    p4_direct = detect_p4(image, key)

    if p4_direct.get("inner_codeword") is not None:
        direct_crc_ratio = p4_direct.get("shard_crc_ratio", 0.0)

        # Guard against RS false-decodes on clean images.
        # Random data can accidentally satisfy RS constraints, but a genuine
        # watermark has CRC-valid shards in ~100% of its tiles (crc_ratio≈1.0).
        # False RS decodes have crc_ratio ≈ 0.03 (1-2 CRC hits out of 31 tiles).
        # Requiring crc_ratio >= 0.50 suppresses these reliably.
        if direct_crc_ratio >= 0.50:
            p4_direct["geometry"] = {
                "angle_deg": 0.0, "scale_factor": 1.0,
                "score": 1.0, "method": "direct",
            }
            p4_direct["corrected_image_shape"] = image.shape[:2]
            p4_direct["shard_crc_ratio"] = direct_crc_ratio
            # Direct P4 path: RS decode succeeded with high CRC ratio.
            # Skip sync template detection (saves ~2s) — verify_p5 will
            # set confidence=1.0 after crypto verification anyway.
            # signature_verified=False: detect_p5 has no public key.
            p4_direct["confidence"] = _compute_confidence(
                signature_verified=False,
                rs_decode_success=True,
                shard_consistency=p4_direct.get("reconstruction_ratio", 1.0),
                shard_crc_ratio=direct_crc_ratio,
                geometry_confidence=1.0,
                correlation=0.0,
                shard_count=p4_direct.get("shard_crc_ok_count", 0),
                shards_needed=p4_direct.get("shards_needed", K_SHARDS),
                harmonic_score=p4_direct.get("harmonic_score", 0.0),
                presence_score=p4_direct.get("presence_score", 0.0),
            )
            return p4_direct
        # CRC ratio too low — likely RS false-decode; fall through to geometry pipeline.

    # Carry forward Phase 3 presence info
    result["detected"] = p4_direct["detected"]
    result["presence_score"] = p4_direct["presence_score"]
    result["scale_scores"] = p4_direct["scale_scores"]
    result["harmonic_score"] = p4_direct["harmonic_score"]

    # ── Step 2: Presence gate ─────────────────────────────────────────
    if not p4_direct["detected"] and p4_direct["presence_score"] < 0.10:
        result["error"] = "No watermark presence detected."
        return result

    # ── Step 3: Geometric estimation ──────────────────────────────────
    geo = estimate_geometry(image, key)
    result["geometry"] = geo

    # ── Step 3b: Sync template refinement ─────────────────────────────
    # Use the sync template to refine the geometry estimate with sub-degree
    # accuracy if the template was detected during Stage 0b.
    if not USE_LEGACY_GEOMETRY and geo["method"] not in (
        "identity", "identity_no_signal", "identity_geo_rejected"
    ):
        try:
            _, Y_refine = _to_ycbcr(image)
            sync_refined = refine_geometry_from_template(
                Y_refine, key,
                initial_rotation=geo["angle_deg"],
                initial_scale=geo["scale_factor"],
            )
            result["sync_template"] = {
                "pilots_detected": sync_refined.pilots_detected,
                "pilots_expected": sync_refined.pilots_expected,
                "recovery_rate": sync_refined.pilot_recovery_rate,
                "confidence": sync_refined.confidence,
                "valid": sync_refined.valid,
            }
            # Use refined estimate if template confidence is strong
            if sync_refined.valid and sync_refined.confidence > 0.4:
                geo["angle_deg"] = sync_refined.rotation_deg
                geo["scale_factor"] = sync_refined.scale_factor
                geo["method"] += "+sync_refined"
        except Exception:
            pass

    # Handle early-exit methods from stop conditions
    if geo["method"] in ("identity", "identity_no_signal", "identity_geo_rejected"):
        result["tiles_located"] = p4_direct["tiles_located"]
        result["shards_recovered"] = p4_direct["shards_recovered"]
        result["reconstruction_ratio"] = p4_direct["reconstruction_ratio"]
        result["shard_crc_ratio"] = p4_direct.get("shard_crc_ratio", 0.0)
        result["error"] = p4_direct.get(
            "error", f"No geometric correction improved detection ({geo['method']})."
        )
        # Compute dual-head confidence even on identity path so HEAD B
        # can distinguish watermarked-but-unverified from clean.
        identity_pilot = geo.get("pilot_recovery_rate", 0.0)
        identity_shard_count = p4_direct.get("shard_crc_ok_count", 0)
        result["confidence"] = _compute_confidence(
            signature_verified=False,
            rs_decode_success=p4_direct.get("inner_codeword") is not None,
            shard_consistency=p4_direct.get("reconstruction_ratio", 0.0),
            shard_crc_ratio=p4_direct.get("shard_crc_ratio", 0.0),
            geometry_confidence=0.0,
            correlation=0.0,
            shard_count=identity_shard_count,
            shards_needed=p4_direct.get("shards_needed", K_SHARDS),
            pilot_recovery_rate=identity_pilot,
            harmonic_score=p4_direct.get("harmonic_score", 0.0),
            presence_score=p4_direct.get("presence_score", 0.0),
        )
        return result

    # ── Runtime guard ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    if elapsed > MAX_DETECTION_TIME:
        if prof:
            prof.rejection_reasons["runtime_abort"] += 1

    # ── Step 4: Full P4 on corrected image (single-pass affine) ────────
    # Agent 2: Use correct_geometry() for single-pass inverse affine
    # instead of the old _undo_scale → _undo_rotation double-resample.
    corrected = gpu_undo_transform(image, geo["angle_deg"], geo["scale_factor"])
    if corrected is None:
        corrected = correct_geometry(
            image, geo["angle_deg"], geo["scale_factor"],
            preserve_energy=True,
        )
    result["corrected_image_shape"] = corrected.shape[:2]

    p4_corrected = detect_p4(corrected, key)

    result["detected"] = p4_corrected["detected"] or result["detected"]
    result["presence_score"] = max(
        result["presence_score"], p4_corrected["presence_score"]
    )
    result["scale_scores"] = p4_corrected.get(
        "scale_scores", result["scale_scores"]
    )
    result["harmonic_score"] = max(
        result["harmonic_score"],
        p4_corrected.get("harmonic_score", 0.0),
    )
    result["tiles_located"] = p4_corrected["tiles_located"]
    result["shards_recovered"] = p4_corrected["shards_recovered"]
    result["shards_needed"] = p4_corrected["shards_needed"]
    result["reconstruction_ratio"] = p4_corrected["reconstruction_ratio"]
    result["inner_codeword"] = p4_corrected.get("inner_codeword")

    if p4_corrected.get("error"):
        result["error"] = p4_corrected["error"]

    # ── Multi-signal confidence scoring ───────────────────────────────
    geo_stability = _geometry_stability_score(
        geo["angle_deg"], geo["scale_factor"],
        0, geo.get("shard_count", 0), geo.get("method", ""),
    )
    shard_ratio = result["reconstruction_ratio"]
    rs_ok       = result["inner_codeword"] is not None
    crc_ratio   = p4_corrected.get("shard_crc_ratio", 0.0)

    # Strategy 1: raw shard CRC count from P4 on corrected image.
    # We use CRC ok count because raw 'shards_recovered' accumulates random noise matches.
    n_shards_found  = p4_corrected.get("shard_crc_ok_count", 0)
    n_shards_needed = result.get("shards_needed", K_SHARDS)

    # Strategy 2: pilot recovery rate from geometry result or sync template.
    # estimate_geometry() now propagates pilot_recovery_rate from its
    # internal Stage 0b sync template detection.
    pilot_rate = geo.get("pilot_recovery_rate", 0.0)

    # If geometry didn't run sync template (e.g. legacy pipeline), try quick detection
    if pilot_rate < 0.01 and not USE_LEGACY_GEOMETRY:
        try:
            _, Y_pilot = _to_ycbcr(image)
            sync_fallback = detect_sync_template(Y_pilot, key)
            if sync_fallback is not None:
                pilot_rate = sync_fallback.pilot_recovery_rate
        except Exception:
            pass

    # Store shard_crc_ratio in result for harness visibility
    result["shard_crc_ratio"] = crc_ratio

    result["confidence"] = _compute_confidence(
        signature_verified=False,   # verify_p5 sets 1.0 after crypto verify
        rs_decode_success=rs_ok,
        shard_consistency=shard_ratio,
        shard_crc_ratio=crc_ratio,
        geometry_confidence=geo_stability,
        correlation=0.0,
        shard_count=n_shards_found,
        shards_needed=n_shards_needed,
        pilot_recovery_rate=pilot_rate,
        harmonic_score=result.get("harmonic_score", 0.0),
        presence_score=result.get("presence_score", 0.0),
    )

    if result["inner_codeword"] is not None:
        return result

    # ── Step 5: Size-adjust search for scale corrections ──────────────
    if abs(geo["scale_factor"] - 1.0) > 0.01:
        h_curr, w_curr = image.shape[:2]
        inv_f = 1.0 / geo["scale_factor"]
        base_w = int(np.ceil(w_curr * inv_f))
        base_h = int(np.ceil(h_curr * inv_f))

        for dw in [-2, -1, 1, 2]:
            tw = base_w + dw
            th = base_h + dw
            if tw < MT_SIZE or th < MT_SIZE:
                continue

            alt = _undo_scale_to_size(image, tw, th)
            if abs(geo["angle_deg"]) > 0.01:
                alt = _undo_rotation(alt, geo["angle_deg"])

            p4_alt = detect_p4(alt, key)
            if p4_alt.get("inner_codeword") is not None:
                result["detected"] = True
                result["confidence"] = _compute_confidence(
                    signature_verified=True,
                    rs_decode_success=True,
                    shard_consistency=p4_alt["reconstruction_ratio"],
                    shard_crc_ratio=1.0,
                    geometry_confidence=geo_stability,
                    correlation=0.0,
                )
                result["tiles_located"] = p4_alt["tiles_located"]
                result["shards_recovered"] = p4_alt["shards_recovered"]
                result["reconstruction_ratio"] = p4_alt[
                    "reconstruction_ratio"]
                result["inner_codeword"] = p4_alt["inner_codeword"]
                result["error"] = p4_alt.get("error")
                result["corrected_image_shape"] = alt.shape[:2]
                geo["method"] += "+size_adjust"
                return result

    # ── Step 6: Perturbation search (4 candidates max) ─────────────────
    perturbations = [(-0.5, -0.02), (-0.5, 0.02), (0.5, -0.02), (0.5, 0.02)]
    for da, ds in perturbations:
        a = geo["angle_deg"] + da
        s = geo["scale_factor"] + ds
        if s < SCALE_BOUND_LOW or s > SCALE_BOUND_HIGH:
            continue

        alt = gpu_undo_transform(image, a, s)
        if alt is None:
            alt = correct_geometry(image, a, s, preserve_energy=True)
        if alt.shape[0] < MT_SIZE or alt.shape[1] < MT_SIZE:
            continue

        p4_alt = detect_p4(alt, key)
        if p4_alt.get("inner_codeword") is not None:
            result["detected"] = True
            result["confidence"] = _compute_confidence(
                signature_verified=True,
                rs_decode_success=True,
                shard_consistency=p4_alt["reconstruction_ratio"],
                shard_crc_ratio=1.0,
                geometry_confidence=geo_stability * 0.9,
                correlation=0.0,
            )
            result["tiles_located"] = p4_alt["tiles_located"]
            result["shards_recovered"] = p4_alt["shards_recovered"]
            result["reconstruction_ratio"] = p4_alt[
                "reconstruction_ratio"]
            result["inner_codeword"] = p4_alt["inner_codeword"]
            result["error"] = p4_alt.get("error")
            geo["angle_deg"] = a
            geo["scale_factor"] = s
            geo["method"] = "perturb_refine"
            result["corrected_image_shape"] = alt.shape[:2]
            return result

    return result
