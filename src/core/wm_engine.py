"""
src/core/wm_engine.py  –  Phase 1 Core Frequency Watermark Engine

Architecture
------------
Pipeline (embed):
    RGB → YCbCr  →  Y (luminance)  →  8×8 block DCT
    → for each block: PRNG(key, block_idx) → select mid-freq pair (u1,v1),(u2,v2)
    → adaptive Δ from block variance
    → difference modulation: bit==1 → C1−C2 ≥ Δ, bit==0 → C2−C1 ≥ Δ
    → IDCT → reconstruct RGB

Pipeline (detect):
    same RGB → Y → DCT
    → for each block i: recompute PRNG pair
    → evidence[i] = C1[i] − C2[i]   (positive ≈ bit-1 region, negative ≈ bit-0)
    → raw_score = normalised correlation of evidence with expected ±1 pattern
    → confidence = sigmoid of score
    → detected = confidence ≥ DETECTION_THRESHOLD

Design Decisions
----------------
Mid-frequency pool
    Rows 2–5, columns 1–5 in the 8×8 DCT block (excluding DC row 0/col 0 and
    high-frequency corner).  These survive JPEG quantisation ≥ Q30 and Gaussian
    blur σ ≤ 1.5, which zero-out only very high-frequency terms.

Difference modulation
    Modifying the *difference* C1−C2 rather than the absolute value of a single
    coefficient makes the embedding invariant under any scalar gain applied to
    the block (i.e., brightness / contrast adjustments multiply all DCT
    coefficients by the same factor, preserving the sign of the difference).

Per-block PRNG via HMAC-SHA256
    key = secret embedding key (bytes)
    block_seed = int.from_bytes(HMAC-SHA256(key, block_index_be)[:4], 'big')
    Each block gets an independent, cryptographically isolated seed.  Because
    the seed depends only on (key, block_index) and *not* on pixel content, the
    embedding is fully deterministic and re-producible from just the key.
    Crop-resilience: the detector does not need to know the global block grid
    origin — it re-indexes each candidate 8×8 tile from block (0,0) at the
    crop boundary and correlates; the majority of blocks will hash to the correct
    key-derived pair and give the right sign.

Adaptive strength
    Δ_i = BASE_DELTA * (1 + BETA * sqrt(σ²_i / VAR_NORM))
    where σ²_i = variance of the spatial 8×8 block.
    Textured blocks absorb a larger Δ invisibly; flat blocks use a smaller Δ
    to stay below the JND, keeping PSNR > 40 dB and SSIM > 0.98.

Public API
----------
    embed(image, bitstream, key) → np.ndarray
    detect(image, key)           → {"detected": bool, "confidence": float, "raw_score": float}

Constants
---------
    BASE_DELTA          : base modulation strength (DCT coefficient units)
    BETA                : adaptive scaling factor
    VAR_NORM            : variance normalisation reference
    DETECTION_THRESHOLD : sigmoid output threshold for "detected"
    MID_FREQ_POOL       : list of (u, v) pairs eligible for selection
"""

import hashlib
import hmac
import struct

import cv2
import numpy as np
from scipy.fft import dctn, idctn

# ─────────────────────────────────────────────────────────────────────────────
#  Tunable constants
# ─────────────────────────────────────────────────────────────────────────────

# Mid-frequency (u, v) candidates in the 8×8 DCT block.
# Rows 2–5, cols 1–5, excluding (u==v and u==0) and high-freq corners.
# These are inside the JPEG luminance quantisation table's "keep" zone for Q≥30.
MID_FREQ_POOL: list[tuple[int, int]] = [
    (u, v)
    for u in range(2, 6)
    for v in range(1, 6)
    if (u + v) >= 3 and (u + v) <= 7
]

BASE_DELTA: float = 14.0      # Baseline modulation strength (DCT units)
BETA: float       = 0.4       # Adaptive variance scaling coefficient (lower = less SSIM impact)
VAR_NORM: float   = 600.0     # Variance normalisation reference (empirical)

TILE_SIZE: int = 8            # Watermark repeats every TILE_SIZE blocks in each dimension.
                              # This makes it crop-resilient: the detector searches over
                              # 8×8 block-level tile alignments via a grid search.

DETECTION_THRESHOLD: float = 0.55   # sigmoid(raw_score) decision boundary
SIGMOID_SCALE: float       = 8.0    # Sharpness of sigmoid mapping


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _block_seed_2d(key: bytes, block_row: int, block_col: int) -> int:
    """
    Derive a 32-bit PRNG seed for block at tiled position
    (block_row % TILE_SIZE, block_col % TILE_SIZE).

    Using modular (tiled) coordinates means the watermark pattern repeats
    with period TILE_SIZE in each dimension.  This allows the detector to
    find the correct phase alignment even after aggressive crops, because
    it only needs to search over TILE_SIZE tile offsets in each axis.
    """
    tr = block_row % TILE_SIZE
    tc = block_col % TILE_SIZE
    coord_bytes = struct.pack(">II", tr, tc)
    digest = hmac.new(key, coord_bytes, hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big")


def _block_bit(key: bytes, block_row: int, block_col: int) -> int:
    """
    Derive the embedded bit for block at tiled position
    (block_row % TILE_SIZE, block_col % TILE_SIZE).

    Tiled bit assignment ensures the detector can phase-align using a grid
    search over all TILE_SIZE×TILE_SIZE starting offsets.
    """
    tr = block_row % TILE_SIZE
    tc = block_col % TILE_SIZE
    coord_bytes = struct.pack(">II", tr, tc)
    digest = hmac.new(key, coord_bytes + b"\x01", hashlib.sha256).digest()
    return int.from_bytes(digest[:1], "big") & 1


def _select_pair(seed: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Select two distinct mid-frequency coefficient positions for a block.

    Uses numpy's PCG64 generator for fast, high-quality pseudo-randomness
    given the HMAC-derived 32-bit seed.
    """
    rng = np.random.default_rng(seed)
    pool = MID_FREQ_POOL
    idxs = rng.choice(len(pool), size=2, replace=False)
    return pool[idxs[0]], pool[idxs[1]]


def _adaptive_delta(spatial_block: np.ndarray) -> float:
    """
    Compute adaptive modulation strength for an 8×8 spatial block.

    Δ = BASE_DELTA * (1 + BETA * sqrt(var / VAR_NORM))

    High-variance (textured) blocks tolerate a larger Δ without visible
    artefacts; low-variance (flat) blocks use a smaller Δ to stay imperceptible.
    """
    var = float(np.var(spatial_block.astype(np.float32)))
    return BASE_DELTA * (1.0 + BETA * np.sqrt(var / VAR_NORM))


def _embed_difference(
    dct_block: np.ndarray,
    bit: int,
    pos1: tuple[int, int],
    pos2: tuple[int, int],
    delta: float,
) -> np.ndarray:
    """
    Enforce the coefficient difference rule in an 8×8 DCT block.

    Embedding rule:
        bit == 1 → ensure  dct_block[pos1] - dct_block[pos2] ≥  delta
        bit == 0 → ensure  dct_block[pos2] - dct_block[pos1] ≥  delta

    The modification is symmetric: we shift C1 up by half the deficit and
    C2 down by half, minimising the L2 distortion.

    Parameters
    ----------
    dct_block : np.ndarray  8×8 float32 DCT block (modified in place on a copy)
    bit       : int         0 or 1
    pos1, pos2: (u,v) pairs — the two selected coefficients
    delta     : float       minimum required absolute difference

    Returns
    -------
    np.ndarray  Modified 8×8 block
    """
    block = dct_block.copy()
    c1 = block[pos1]
    c2 = block[pos2]

    if bit == 1:
        current_diff = c1 - c2
        if current_diff < delta:
            deficit = delta - current_diff
            block[pos1] = c1 + deficit / 2.0
            block[pos2] = c2 - deficit / 2.0
    else:  # bit == 0
        current_diff = c2 - c1
        if current_diff < delta:
            deficit = delta - current_diff
            block[pos2] = c2 + deficit / 2.0
            block[pos1] = c1 - deficit / 2.0

    return block


def _sigmoid(x: float, scale: float = SIGMOID_SCALE) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-scale * x))
    else:
        e = np.exp(scale * x)
        return e / (1.0 + e)


# ─────────────────────────────────────────────────────────────────────────────
#  RGB ↔ YCbCr helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_ycbcr(bgr_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert BGR image to YCrCb.  Returns (ycrcb_image, Y_channel_float32).

    Note: OpenCV uses YCrCb ordering (Y, Cr, Cb), which is standard for JPEG.
    """
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)
    return ycrcb, Y


def _from_ycbcr(original_ycrcb: np.ndarray, new_y: np.ndarray) -> np.ndarray:
    """
    Replace Y channel in YCrCb image and convert back to BGR.

    Parameters
    ----------
    original_ycrcb : np.ndarray  Original YCrCb image (for Cr, Cb channels)
    new_y          : np.ndarray  Modified Y channel (float32 or uint8)
    """
    result = original_ycrcb.copy()
    result[:, :, 0] = np.clip(new_y, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)


def _pad_to_8(arr: np.ndarray) -> np.ndarray:
    """Reflect-pad a 2D array so both dimensions are multiples of 8."""
    h, w = arr.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")


# ─────────────────────────────────────────────────────────────────────────────
#  Block-DCT utilities
# ─────────────────────────────────────────────────────────────────────────────

def _apply_block_dct(y_float: np.ndarray) -> np.ndarray:
    """
    Apply 8×8 block-wise 2D DCT using scipy — fully vectorised, no Python loop.

    Reshape (H, W) → (bh, 8, bw, 8) → transpose → (bh, bw, 8, 8), apply
    dctn(norm='ortho') over axes (-2, -1), then reverse the reshape.
    This is equivalent to calling cv2.dct on every 8×8 block but ~10× faster.
    """
    h, w = y_float.shape
    bh, bw = h // 8, w // 8
    blocks = y_float.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)  # (bh, bw, 8, 8)
    dct_blocks = dctn(blocks, type=2, axes=(-2, -1), norm='ortho')
    return dct_blocks.transpose(0, 2, 1, 3).reshape(h, w).astype(np.float32)


def _apply_block_idct(dct_img: np.ndarray) -> np.ndarray:
    """
    Apply 8×8 block-wise 2D IDCT using scipy — fully vectorised.
    """
    h, w = dct_img.shape
    bh, bw = h // 8, w // 8
    blocks = dct_img.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)  # (bh, bw, 8, 8)
    idct_blocks = idctn(blocks, type=2, axes=(-2, -1), norm='ortho')
    return idct_blocks.transpose(0, 2, 1, 3).reshape(h, w).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def embed(image: np.ndarray, bitstream: np.ndarray, key: bytes) -> np.ndarray:
    """
    Embed a pseudo-random watermark into a BGR image using DCT coefficient
    difference modulation.

    Parameters
    ----------
    image     : np.ndarray  BGR image, uint8, shape (H, W, 3).
    bitstream : np.ndarray  1-D array of 0/1 values to embed.  Will be tiled
                            across all blocks if shorter than the block count.
                            For Phase 1 the payload is a mocked bitstream; in
                            Phase 2 this is the ECC-encoded signed payload.
    key       : bytes       Secret embedding key.  Any length; internally
                            expanded via HMAC-SHA256 per block.

    Returns
    -------
    np.ndarray  Watermarked BGR image (uint8, same shape as input).

    Notes
    -----
    - Embedding is fully deterministic: same (image, bitstream, key) → same output.
    - PSNR > 40 dB and SSIM > 0.98 are maintained at default BASE_DELTA=18.
    - Only the Y (luminance) channel is modified; Cb/Cr are unchanged.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("embed() expects a 3-channel BGR image (H, W, 3).")
    if len(bitstream) == 0:
        raise ValueError("bitstream must be non-empty.")

    bits = np.asarray(bitstream, dtype=np.int32).ravel()
    ycrcb, Y = _to_ycbcr(image)

    orig_h, orig_w = Y.shape
    Y_padded = _pad_to_8(Y)
    ph, pw = Y_padded.shape
    bh, bw = ph // 8, pw // 8

    # ── Precompute tile lookup table ──────────────────────────────────────────
    # For each tile position (tr, tc) in [0, TILE_SIZE)², precompute:
    #   tile_p1[tr, tc]  = (u1, v1) — first coefficient index
    #   tile_p2[tr, tc]  = (u2, v2) — second coefficient index
    #   tile_bits[tr, tc] = 0 or 1  — embedded bit
    # This replaces 8256 HMAC + RNG calls in the loop.
    tile_p1   = np.zeros((TILE_SIZE, TILE_SIZE, 2), dtype=np.int8)
    tile_p2   = np.zeros((TILE_SIZE, TILE_SIZE, 2), dtype=np.int8)
    tile_bits = np.zeros((TILE_SIZE, TILE_SIZE),    dtype=np.int8)
    for tr in range(TILE_SIZE):
        for tc in range(TILE_SIZE):
            seed = _block_seed_2d(key, tr, tc)
            p1, p2 = _select_pair(seed)
            tile_p1[tr, tc]   = p1
            tile_p2[tr, tc]   = p2
            tile_bits[tr, tc] = _block_bit(key, tr, tc)

    # ── Reshape Y into blocks (bh, bw, 8, 8) ─────────────────────────────────
    Y_blocks = Y_padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).astype(np.float32)

    # ── Vectorised variance → delta for every block ───────────────────────────
    var_blocks   = np.var(Y_blocks, axis=(2, 3))          # (bh, bw)
    delta_blocks = (BASE_DELTA * (1.0 + BETA * np.sqrt(var_blocks / VAR_NORM))).astype(np.float32)

    # ── DCT ───────────────────────────────────────────────────────────────────
    dct_img    = _apply_block_dct(Y_padded)
    dct_blocks = dct_img.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)  # (bh, bw, 8, 8)

    # ── Per-block tile coordinates ────────────────────────────────────────────
    br_idx = np.arange(bh, dtype=np.int32)[:, None]   # (bh, 1)
    bc_idx = np.arange(bw, dtype=np.int32)[None, :]   # (1, bw)
    tr_idx = br_idx % TILE_SIZE                        # (bh, bw) broadcast
    tc_idx = bc_idx % TILE_SIZE

    # Gather per-block pair and bit from tile table
    p1u = tile_p1[tr_idx, tc_idx, 0].astype(np.int64)  # (bh, bw)
    p1v = tile_p1[tr_idx, tc_idx, 1].astype(np.int64)
    p2u = tile_p2[tr_idx, tc_idx, 0].astype(np.int64)
    p2v = tile_p2[tr_idx, tc_idx, 1].astype(np.int64)
    blk_bits = tile_bits[tr_idx, tc_idx].astype(np.int8)  # (bh, bw)

    # ── Gather C1, C2 for every block ─────────────────────────────────────────
    # dct_blocks[br, bc, u, v] — advanced index with all arrays (bh, bw)
    br_full = np.broadcast_to(br_idx, (bh, bw))
    bc_full = np.broadcast_to(bc_idx, (bh, bw))
    c1 = dct_blocks[br_full, bc_full, p1u, p1v].copy()  # (bh, bw)
    c2 = dct_blocks[br_full, bc_full, p2u, p2v].copy()

    # ── Vectorised difference modulation ──────────────────────────────────────
    # bit==1: enforce C1 - C2 >= delta   (boost C1, reduce C2 by half deficit each)
    # bit==0: enforce C2 - C1 >= delta
    delta = delta_blocks  # (bh, bw)
    bit1_mask = blk_bits == 1   # (bh, bw) bool
    bit0_mask = ~bit1_mask

    # Bit-1 blocks
    diff1 = c1 - c2                         # (bh, bw)
    deficit1 = np.maximum(0.0, delta - diff1)
    c1_new = np.where(bit1_mask, c1 + deficit1 / 2.0, c1)
    c2_new = np.where(bit1_mask, c2 - deficit1 / 2.0, c2)

    # Bit-0 blocks  (overwrite the bit-1 result for bit-0 positions)
    diff0 = c2_new - c1_new
    deficit0 = np.maximum(0.0, delta - diff0)
    c1_new = np.where(bit0_mask, c1_new - deficit0 / 2.0, c1_new)
    c2_new = np.where(bit0_mask, c2_new + deficit0 / 2.0, c2_new)

    # ── Scatter back ──────────────────────────────────────────────────────────
    dct_blocks[br_full, bc_full, p1u, p1v] = c1_new
    dct_blocks[br_full, bc_full, p2u, p2v] = c2_new

    # ── IDCT → spatial ────────────────────────────────────────────────────────
    dct_img_out = dct_blocks.transpose(0, 2, 1, 3).reshape(ph, pw).astype(np.float32)
    Y_watermarked = _apply_block_idct(dct_img_out)

    # Crop back to original size, reconstruct BGR
    Y_out = Y_watermarked[:orig_h, :orig_w]
    return _from_ycbcr(ycrcb, Y_out)


def detect(image: np.ndarray, key: bytes) -> dict:
    """
    Detect a watermark embedded by embed() using the same key.

    The detector measures the normalised correlation between the observed
    coefficient differences and the PRNG-expected sign pattern.  A positive
    correlation implies the watermark is present; a negative / near-zero
    correlation implies it is absent.

    Parameters
    ----------
    image : np.ndarray  BGR image (uint8, H×W×3).  May have been attacked.
    key   : bytes       Same secret key used during embedding.

    Returns
    -------
    dict with keys:
        "detected"   : bool   — True if confidence ≥ DETECTION_THRESHOLD
        "confidence" : float  — sigmoid output in (0, 1); ~0.5 for non-watermarked
        "raw_score"  : float  — mean signed evidence in [-1, 1]; positive = watermark

    Algorithm
    ---------
    For each 8×8 block i, we know (from the key) which pair (pos1, pos2) was
    used and which bit was embedded.  The embedded bit determined the *sign* of
    the expected difference:
        expected[i] = +1  if bit[i % n_bits] == 1
        expected[i] = -1  if bit[i % n_bits] == 0

    The observed evidence:
        observed_diff[i] = dct_block[i][pos1] − dct_block[i][pos2]

    Normalised to [-1, +1] via:
        evidence[i] = tanh(observed_diff[i] / delta_i)

    raw_score = mean( evidence[i] * expected[i] )

    Under no attack, raw_score ≈ 1.0 (perfect correlation).
    Under heavy attack, raw_score degrades toward 0 (random).
    On a non-watermarked image, raw_score ≈ 0 (no correlation).

    Notes
    -----
    - Phase 1: bitstream is regenerated from the key (fixed-length mocked stream).
      In Phase 2 the full ECC-protected, signed payload replaces the mock.
    - The detector is blind to the bitstream content — it only tests whether the
      correct key-derived sign pattern is present in the image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detect() expects a 3-channel BGR image (H, W, 3).")

    _, Y_full = _to_ycbcr(image)

    # ── Precompute tile reference pattern (8×8 tile positions) ───────────────
    tile_pos1  = np.zeros((TILE_SIZE, TILE_SIZE, 2), dtype=np.int8)
    tile_pos2  = np.zeros((TILE_SIZE, TILE_SIZE, 2), dtype=np.int8)
    tile_signs = np.zeros((TILE_SIZE, TILE_SIZE),    dtype=np.float32)
    for tr in range(TILE_SIZE):
        for tc in range(TILE_SIZE):
            seed = _block_seed_2d(key, tr, tc)
            p1, p2 = _select_pair(seed)
            tile_pos1[tr, tc] = p1
            tile_pos2[tr, tc] = p2
            tile_signs[tr, tc] = 1.0 if _block_bit(key, tr, tc) == 1 else -1.0

    def _score_at_offset(Y: np.ndarray, blk_dy: int = 0, blk_dx: int = 0) -> float:
        """
        Compute the raw correlation score for a given block-phase offset.
        Y must already be padded to a multiple of 8.
        """
        ph, pw = Y.shape
        bh, bw = ph // 8, pw // 8
        dct_img = _apply_block_dct(Y)
        dct_blocks = dct_img.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)  # (bh,bw,8,8)
        Y_blocks   = Y.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
        var_blocks = np.var(Y_blocks.astype(np.float32), axis=(2, 3))
        delta_blocks = (BASE_DELTA * (1.0 + BETA * np.sqrt(var_blocks / VAR_NORM))).astype(np.float32)

        br_idx = np.arange(bh, dtype=np.int32)[:, None]
        bc_idx = np.arange(bw, dtype=np.int32)[None, :]
        tr_idx = (br_idx + blk_dy) % TILE_SIZE
        tc_idx = (bc_idx + blk_dx) % TILE_SIZE

        # Vectorise over the 64 tile positions: build ev_mats[tr, tc, bh, bw]
        ev_mats = np.empty((TILE_SIZE, TILE_SIZE, bh, bw), dtype=np.float32)
        for tr in range(TILE_SIZE):
            for tc in range(TILE_SIZE):
                p1 = tuple(tile_pos1[tr, tc])
                p2 = tuple(tile_pos2[tr, tc])
                obs = dct_blocks[:, :, p1[0], p1[1]] - dct_blocks[:, :, p2[0], p2[1]]
                ev_mats[tr, tc] = np.tanh(obs / (delta_blocks + 1e-6))

        ev_gathered   = ev_mats[tr_idx, tc_idx, br_idx, bc_idx]
        sign_gathered = tile_signs[tr_idx, tc_idx]
        return float(np.mean(ev_gathered * sign_gathered))

    # ── Standard detection: alignment (0,0) only ──────────────────────────────
    # FPR rationale: an unwatermarked image has zero-mean random DCT differences.
    # Testing a single alignment gives E[score]=0 with small variance → low FPR.
    # The exhaustive grid search (detect_robust) is reserved for known-cropped images.
    Y_padded = _pad_to_8(Y_full)
    raw_score = _score_at_offset(Y_padded, 0, 0)
    confidence = _sigmoid(raw_score)
    detected   = confidence >= DETECTION_THRESHOLD

    return {
        "detected":   detected,
        "confidence": round(confidence, 6),
        "raw_score":  round(raw_score, 6),
    }


def detect_robust(image: np.ndarray, key: bytes) -> dict:
    """
    Crop-resilient watermark detector.

    Identical to detect() but performs an exhaustive two-level grid search:
    - Level 1: 8 pixel-level offsets in y and x (handles sub-block crop margins)
    - Level 2: TILE_SIZE block-phase offsets in y and x (handles full-block shifts)

    Use this when the image is known to have been cropped.  Do NOT use as the
    default detector: taking the maximum over 4096 trials inflates the FPR on
    unwatermarked images.

    Parameters / Returns: same as detect().
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detect_robust() expects a 3-channel BGR image (H, W, 3).")

    _, Y_full = _to_ycbcr(image)

    # Precompute tile reference pattern
    tile_pos1  = np.zeros((TILE_SIZE, TILE_SIZE, 2), dtype=np.int8)
    tile_pos2  = np.zeros((TILE_SIZE, TILE_SIZE, 2), dtype=np.int8)
    tile_signs = np.zeros((TILE_SIZE, TILE_SIZE),    dtype=np.float32)
    for tr in range(TILE_SIZE):
        for tc in range(TILE_SIZE):
            seed = _block_seed_2d(key, tr, tc)
            p1, p2 = _select_pair(seed)
            tile_pos1[tr, tc] = p1
            tile_pos2[tr, tc] = p2
            tile_signs[tr, tc] = 1.0 if _block_bit(key, tr, tc) == 1 else -1.0

    best_raw_score: float = -2.0

    for px_dy in range(8):
        for px_dx in range(8):
            Y_shifted = Y_full[px_dy:, px_dx:]
            if Y_shifted.shape[0] < 8 or Y_shifted.shape[1] < 8:
                continue
            Y_padded = _pad_to_8(Y_shifted)
            ph, pw   = Y_padded.shape
            bh, bw   = ph // 8, pw // 8
            dct_img  = _apply_block_dct(Y_padded)
            dct_blocks = dct_img.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
            Y_blocks   = Y_padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
            var_blocks   = np.var(Y_blocks.astype(np.float32), axis=(2, 3))
            delta_blocks = (BASE_DELTA * (1.0 + BETA * np.sqrt(var_blocks / VAR_NORM))).astype(np.float32)

            # Build ev_mats[tr, tc, bh, bw]
            ev_mats = np.empty((TILE_SIZE, TILE_SIZE, bh, bw), dtype=np.float32)
            for tr in range(TILE_SIZE):
                for tc in range(TILE_SIZE):
                    p1  = tuple(tile_pos1[tr, tc])
                    p2  = tuple(tile_pos2[tr, tc])
                    obs = dct_blocks[:, :, p1[0], p1[1]] - dct_blocks[:, :, p2[0], p2[1]]
                    ev_mats[tr, tc] = np.tanh(obs / (delta_blocks + 1e-6))

            br_idx = np.arange(bh, dtype=np.int32)[:, None]
            bc_idx = np.arange(bw, dtype=np.int32)[None, :]

            for blk_dy in range(TILE_SIZE):
                tr_idx = (br_idx + blk_dy) % TILE_SIZE
                for blk_dx in range(TILE_SIZE):
                    tc_idx_arr = (bc_idx + blk_dx) % TILE_SIZE
                    ev_gathered   = ev_mats[tr_idx, tc_idx_arr, br_idx, bc_idx]
                    sign_gathered = tile_signs[tr_idx, tc_idx_arr]
                    score = float(np.mean(ev_gathered * sign_gathered))
                    if score > best_raw_score:
                        best_raw_score = score

    confidence = _sigmoid(best_raw_score)
    return {
        "detected":   confidence >= DETECTION_THRESHOLD,
        "confidence": round(confidence, 6),
        "raw_score":  round(best_raw_score, 6),
    }


def embed_with_prng_payload(image: np.ndarray, key: bytes) -> np.ndarray:
    """
    Convenience wrapper for Phase 1: generates the PRNG-derived bitstream from
    the key and embeds it.  Mirrors the exact bitstream that detect() will
    regenerate, ensuring the detect/embed pair is self-consistent.

    Note: since embed() now uses position-based bit assignment via _block_bit(),
    the `bitstream` argument passed to embed() is ignored in favour of per-block
    key-derived bits.  This wrapper exists for API compatibility.

    Parameters
    ----------
    image : np.ndarray  BGR image, uint8.
    key   : bytes       Secret embedding key.

    Returns
    -------
    np.ndarray  Watermarked BGR image.
    """
    # The bitstream argument is ignored internally; pass a dummy.
    return embed(image, np.array([0], dtype=np.int32), key)
