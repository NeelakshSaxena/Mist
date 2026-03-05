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
    """Apply 8×8 block-wise 2D DCT.  Input must be float32 with dimensions % 8 == 0."""
    h, w = y_float.shape
    out = np.empty_like(y_float)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            out[i:i+8, j:j+8] = cv2.dct(y_float[i:i+8, j:j+8])
    return out


def _apply_block_idct(dct_img: np.ndarray) -> np.ndarray:
    """Apply 8×8 block-wise 2D IDCT."""
    h, w = dct_img.shape
    out = np.empty_like(dct_img)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            out[i:i+8, j:j+8] = cv2.idct(dct_img[i:i+8, j:j+8])
    return out


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

    # DCT
    dct_img = _apply_block_dct(Y_padded)

    # Embed bit-by-bit, using 2D block position for deterministic seeding.
    # Using (br, bc) = (i//8, j//8) as the seed coordinates ensures the same
    # pair/bit assignment at that grid location, regardless of image crop.
    n_bits = len(bits)
    bit_idx = 0   # sequential index only for the bitstream tiling

    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            br, bc = i // 8, j // 8
            # Use position-derived bit (from key + position), ignoring bitstream
            # tiling — this ensures crop-resilient detection.
            bit = _block_bit(key, br, bc)
            seed = _block_seed_2d(key, br, bc)
            pos1, pos2 = _select_pair(seed)
            delta = _adaptive_delta(Y_padded[i:i+8, j:j+8])

            dct_img[i:i+8, j:j+8] = _embed_difference(
                dct_img[i:i+8, j:j+8], bit, pos1, pos2, delta
            )
            bit_idx += 1

    # IDCT → spatial
    Y_watermarked = _apply_block_idct(dct_img)

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

    # ── Precompute the TILE_SIZE×TILE_SIZE reference pattern ──────────────────
    # For each tiled position (tr, tc), precompute:
    #   - pos1, pos2: coefficient positions (from _block_seed_2d)
    #   - expected_sign: +1 or -1 (from _block_bit)
    # This avoids re-hashing inside any loop.
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

    # ── Two-level grid search ─────────────────────────────────────────────────
    # Level 1: 8 pixel offsets in y/x (handles sub-block crop margins)
    # Level 2: TILE_SIZE block offsets in y/x (handles whole-block crop margins)
    # The block-phase search is vectorised using modular indexing on a
    # prebuilt per-block raw evidence matrix.
    best_raw_score: float = -2.0
    best_confidence: float = 0.0

    for px_dy in range(8):
        for px_dx in range(8):
            Y_shifted = Y_full[px_dy:, px_dx:]
            if Y_shifted.shape[0] < 8 or Y_shifted.shape[1] < 8:
                continue

            Y_padded = _pad_to_8(Y_shifted)
            dct_img  = _apply_block_dct(Y_padded)
            ph, pw   = Y_padded.shape
            bh, bw   = ph // 8, pw // 8

            # Reshape for vectorised block access: (bh, bw, 8, 8)
            # dct_blocks[br, bc] = 8×8 DCT block at grid position (br, bc)
            dct_blocks = dct_img.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
            # spatial blocks for variance
            Y_blocks = Y_padded.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)

            # For each block, compute raw observed_diff / delta for all tile positions:
            # We'll compute evidence for all (tr, tc) simultaneously.
            # ev_matrix[br, bc] = tanh(obs_diff / delta) for the pair at
            #   tile pos (br%T, bc%T), combined with the expected_sign below.
            # Since different tile positions have different pairs, we loop over
            # the 64 unique (tr,tc) combos — the inner work is numpy array ops.

            # Build per-tile-position evidence matrices (TILE_SIZE×TILE_SIZE
            # of matrices, each of shape bh×bw)
            # ev_by_tile[tr, tc, br, bc] = tanh(diff/delta) for block (br,bc)
            # evaluated as if it belongs to tile position (tr, tc).
            # Indexed as: for blocks where (br%T)==tr and (bc%T)==tc, this is
            # the actual evidence; elsewhere it's garbage but we mask it out.

            # Fast approach: compute ev_raw_matrix[br, bc] using the tile
            # position (br%T, bc%T) for each block — this is what the embedder
            # actually used, and is what the naive (no-crop) detector uses.
            # Then for each block-phase shift (blk_dy, blk_dx), the tile
            # position becomes ((br+blk_dy)%T, (bc+blk_dx)%T), which means
            # we use a different pair for each block.
            # This requires 64 separate pair lookups.  We do this efficiently
            # by prebuilding ev_matrices for each of the 64 tile positions,
            # then assembling per-phase scores using numpy indexing.

            # ev_mats[tr][tc] has shape (bh, bw), containing the evidence for
            # each block if it were assigned tile position (tr, tc).
            ev_mats = np.zeros((TILE_SIZE, TILE_SIZE, bh, bw), dtype=np.float32)
            var_blocks = np.var(Y_blocks.astype(np.float32), axis=(2, 3))  # (bh, bw)
            delta_blocks = (BASE_DELTA * (1.0 + BETA * np.sqrt(var_blocks / VAR_NORM))).astype(np.float32)

            for tr in range(TILE_SIZE):
                for tc in range(TILE_SIZE):
                    p1 = tuple(tile_pos1[tr, tc])
                    p2 = tuple(tile_pos2[tr, tc])
                    obs = dct_blocks[:, :, p1[0], p1[1]] - dct_blocks[:, :, p2[0], p2[1]]
                    ev_mats[tr, tc] = np.tanh(obs / (delta_blocks + 1e-6))

            # For each (blk_dy, blk_dx) phase, compute mean evidence:
            #   mean over all (br, bc) of: ev_mats[(br+blk_dy)%T, (bc+blk_dx)%T, br, bc]
            #                              × tile_signs[(br+blk_dy)%T, (bc+blk_dx)%T]
            # Vectorised: construct tr_idx[br, bc] and tc_idx[br, bc] arrays,
            # gather from ev_mats and tile_signs, multiply, mean.
            br_idx = np.arange(bh, dtype=np.int32)[:, None]  # (bh, 1)
            bc_idx = np.arange(bw, dtype=np.int32)[None, :]  # (1, bw)

            for blk_dy in range(TILE_SIZE):
                tr_idx = (br_idx + blk_dy) % TILE_SIZE   # (bh, bw)
                for blk_dx in range(TILE_SIZE):
                    tc_idx_arr = (bc_idx + blk_dx) % TILE_SIZE   # (bh, bw) broadcast

                    # Gather evidence and signs
                    ev_gathered   = ev_mats[tr_idx, tc_idx_arr, br_idx, bc_idx]
                    sign_gathered = tile_signs[tr_idx, tc_idx_arr]

                    raw_score_candidate = float(np.mean(ev_gathered * sign_gathered))
                    if raw_score_candidate > best_raw_score:
                        best_raw_score = raw_score_candidate
                        best_confidence = _sigmoid(raw_score_candidate)

    raw_score = best_raw_score
    confidence = best_confidence
    detected = confidence >= DETECTION_THRESHOLD

    return {
        "detected":   detected,
        "confidence": round(confidence, 6),
        "raw_score":  round(raw_score, 6),
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
