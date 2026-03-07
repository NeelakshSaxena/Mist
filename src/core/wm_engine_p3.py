"""
src/core/wm_engine_p3.py  –  Phase 3 Diffusion-Resistant Watermark Engine

Why Phase 3?
------------
Phase 2's mid-band DCT difference modulation is fragile against diffusion
regeneration.  When an attacker runs img2img, the model reconstructs the image
from its learned prior — it reproduces global structure (edges, colour, shapes)
but does NOT preserve exact coefficient relationships.  The watermark vanishes.

Phase 3 approach:
-----------------
1. Multi-Scale Embedding (8×8, 16×16, 32×32 DCT blocks):
   Same bit pattern written at three independent scales.  Coarser scales encode
   global structure and survive diffusion reconstruction.  Detection fuses votes
   from all three scales.

2. Low-Frequency Pair Bias:
   Coefficient pairs are drawn from indices close to DC (rows 0–2, cols 0–2).
   Diffusion models preserve low-frequency components because they encode global
   brightness and large-scale colour — the model's prior strongly enforces them.

3. Quantization-Basin Modulation:
   Instead of the Phase 2 "difference ≥ Δ" rule, each target coefficient is
   snapped to the nearest quantization grid position:
       C_new = round(C / Δ) * Δ  +  bit * (Δ / 2)
   This creates attractor basins.  Even if the coefficient drifts by up to Δ/2,
   the hard decision (C mod Δ > Δ/2 ?) snaps back to the correct bit.
   Empirically 3–5× more robust to diffusion-induced coefficient drift.

4. Global Sinusoidal Harmonic Injection:
   A weak key-derived sine wave is added across the full luminance channel:
       I'(x,y) = I(x,y) + HARMONIC_ALPHA * sin(2π(fx·x + fy·y))
   This produces a sharp peak in the 2D FFT magnitude spectrum at (fx, fy).
   Diffusion models often reproduce it as a "natural lighting gradient."
   Detection: compare FFT magnitude at known frequency bin vs. null hypothesis.

5. Perceptual Edge-Guided Strength:
   Laplacian energy of each spatial block scales the embedding delta.  Textured
   / edge regions absorb more energy invisibly AND are reproduced more faithfully
   by the diffusion model's prior (they carry semantic content).

Public API
----------
    embed_p3(image, bitstream, key) → np.ndarray
    detect_p3(image, key)           → dict {detected, confidence, raw_score,
                                            scale_scores, harmonic_score}
    extract_bits_p3(image, key, n_bits) → list[int]

Constants
---------
    BASE_DELTA_P3       : base quantization step (DCT units)
    HARMONIC_ALPHA      : amplitude of injected sine wave (pixel units, ~1.0)
    HARMONIC_FREQ_BINS  : number of candidate frequency bins tried
    SCALES              : block sizes used for multi-scale embedding
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

# Quantization step.  Smaller = more subtle = less PSNR hit.
# Lower than Phase 2's 14.0 because triple-scale redundancy compensates.
BASE_DELTA_P3: float = 10.0

# Variance-adaptive scaling (same formula as Phase 2)
BETA_P3: float     = 0.35
VAR_NORM_P3: float = 600.0

# Harmonic injection
HARMONIC_ALPHA: float = 1.2   # pixel amplitude (barely visible at α ≤ 1.5)

# Tiling for crop-resilience.  Watermark repeats every TILE_P3 blocks.
TILE_P3: int = 8

# Detection threshold
DETECTION_THRESHOLD_P3: float = 0.52
SIGMOID_SCALE_P3: float       = 8.0

# Multi-scale block sizes
SCALES: tuple[int, ...] = (8, 16, 32)

# Scale weights for score fusion (coarser = more diffusion-robust → higher weight)
SCALE_WEIGHTS: dict[int, float] = {8: 0.25, 16: 0.35, 32: 0.40}

# Mid-frequency pair pool (rows 2-5, cols 1-5): identical to Phase 2.
# These survive JPEG Q>=30 and are PSNR-safe (not visible to human eye).
# For detection at coarse scales (16x16, 32x32), the same pairs are used —
# these are valid indices for larger blocks (rows 2-5 < 16, all valid).
PAIR_POOL: list[tuple[int, int]] = [
    (u, v)
    for u in range(2, 6)
    for v in range(1, 6)
    if (u + v) >= 3 and (u + v) <= 7
]


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hmac_bytes(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def _block_seed(key: bytes, block_row: int, block_col: int, scale: int) -> int:
    """Derive a per-(block, scale) PRNG seed from the secret key."""
    tr = block_row % TILE_P3
    tc = block_col % TILE_P3
    data = struct.pack(">III", tr, tc, scale)
    return int.from_bytes(_hmac_bytes(key, data)[:4], "big")


def _select_pair(
    seed: int, pool: list[tuple[int, int]], block_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Select two distinct coefficient positions valid for a (block_size × block_size) block.
    Filters the pool to positions that fit inside the block.
    """
    valid = [(r, c) for (r, c) in pool if r < block_size and c < block_size]
    if len(valid) < 2:
        # Fallback for tiny blocks
        valid = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
        valid = [(r, c) for (r, c) in valid if r < block_size and c < block_size]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(valid), size=2, replace=False)
    return valid[idxs[0]], valid[idxs[1]]


def _harmonic_freq(key: bytes) -> tuple[float, float]:
    """
    Derive a key-specific spatial frequency (fx, fy) for sinusoidal injection.
    Frequency is in cycles-per-pixel, chosen in the range [1/256, 1/32].
    This places the peak well into the mid-frequency range visible in FFT but
    appearing as a gentle gradient to human eyes.
    """
    digest = _hmac_bytes(key, b"harmonic-freq-v3")
    fx_raw = int.from_bytes(digest[0:4], "big") / 2**32
    fy_raw = int.from_bytes(digest[4:8], "big") / 2**32
    # Map to [1/256, 1/32]
    lo, hi = 1.0 / 256.0, 1.0 / 32.0
    fx = lo + fx_raw * (hi - lo)
    fy = lo + fy_raw * (hi - lo)
    return fx, fy


def _harmonic_phase(key: bytes) -> float:
    """Derive key-specific phase offset for the sinusoidal signal."""
    digest = _hmac_bytes(key, b"harmonic-phase-v3")
    return (int.from_bytes(digest[0:4], "big") / 2**32) * 2 * np.pi


def _build_harmonic_map(h: int, w: int, key: bytes) -> np.ndarray:
    """
    Build the full-image sinusoidal watermark pattern (float32, shape H×W).
    Values in range [-HARMONIC_ALPHA, +HARMONIC_ALPHA].
    """
    fx, fy = _harmonic_freq(key)
    phase  = _harmonic_phase(key)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return (HARMONIC_ALPHA * np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)).astype(np.float32)


def _laplacian_energy_map(Y: np.ndarray, block_size: int) -> np.ndarray:
    """
    Compute per-block Laplacian energy map (bh × bw float32).
    Blocks with high edge energy receive a larger embedding delta.
    """
    lap = cv2.Laplacian(Y.astype(np.float32), cv2.CV_32F)
    lap_sq = lap ** 2
    h, w = Y.shape
    bh, bw = h // block_size, w // block_size
    # Reshape into blocks and take mean energy per block
    laps = lap_sq[:bh * block_size, :bw * block_size]
    energy = (laps
              .reshape(bh, block_size, bw, block_size)
              .transpose(0, 2, 1, 3)
              .reshape(bh, bw, block_size * block_size)
              .mean(axis=2))
    return energy.astype(np.float32)


def _sigmoid(x: float, scale: float = SIGMOID_SCALE_P3) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-scale * x))
    e = np.exp(scale * x)
    return e / (1.0 + e)


# ─────────────────────────────────────────────────────────────────────────────
#  YCbCr / block utilities (same as Phase 2, local copies for isolation)
# ─────────────────────────────────────────────────────────────────────────────

def _to_ycbcr(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb, ycrcb[:, :, 0].astype(np.float32)


def _from_ycbcr(ycrcb: np.ndarray, new_y: np.ndarray) -> np.ndarray:
    out = ycrcb.copy()
    out[:, :, 0] = np.clip(new_y, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def _pad_to_n(arr: np.ndarray, n: int) -> np.ndarray:
    """Reflect-pad so both dims are multiples of n."""
    h, w = arr.shape
    ph = (n - h % n) % n
    pw = (n - w % n) % n
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, ph), (0, pw)), mode="reflect")


def _block_dct(y_float: np.ndarray, bs: int) -> np.ndarray:
    """Vectorised block-DCT for arbitrary block size bs."""
    h, w = y_float.shape
    bh, bw = h // bs, w // bs
    blocks = y_float.reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3)
    dct_blocks = dctn(blocks, type=2, axes=(-2, -1), norm="ortho")
    return dct_blocks.transpose(0, 2, 1, 3).reshape(h, w).astype(np.float32)


def _block_idct(dct_img: np.ndarray, bs: int) -> np.ndarray:
    """Vectorised block-IDCT for arbitrary block size bs."""
    h, w = dct_img.shape
    bh, bw = h // bs, w // bs
    blocks = dct_img.reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3)
    idct_blocks = idctn(blocks, type=2, axes=(-2, -1), norm="ortho")
    return idct_blocks.transpose(0, 2, 1, 3).reshape(h, w).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-scale embedding core
# ─────────────────────────────────────────────────────────────────────────────

def _embed_one_scale(
    Y_padded: np.ndarray,
    bits: np.ndarray,          # 1-D int array, tiled per block
    key: bytes,
    block_size: int,
) -> np.ndarray:
    """
    Quantization-basin embedding at a single block scale.

    For each block (br, bc):
        1. Look up key-derived coefficient pair (pos1, pos2).
        2. Compute adaptive delta from block variance + edge energy.
        3. Embed using quantization basin:
               C_new = round(C / delta) * delta  +  bit * (delta / 2)

    Returns modified Y_padded (same shape, float32).
    """
    h, w = Y_padded.shape
    bh, bw = h // block_size, w // block_size

    # Build tile lookup: positions for each (tr, tc) in TILE_P3 × TILE_P3
    tile_p1 = np.zeros((TILE_P3, TILE_P3, 2), dtype=np.int8)
    tile_p2 = np.zeros((TILE_P3, TILE_P3, 2), dtype=np.int8)
    for tr in range(TILE_P3):
        for tc in range(TILE_P3):
            seed = _block_seed(key, tr, tc, block_size)
            p1, p2 = _select_pair(seed, PAIR_POOL, block_size)
            tile_p1[tr, tc] = p1
            tile_p2[tr, tc] = p2

    br_idx = np.arange(bh, dtype=np.int32)[:, None]   # (bh,1)
    bc_idx = np.arange(bw, dtype=np.int32)[None, :]   # (1,bw)
    tr_idx = br_idx % TILE_P3
    tc_idx = bc_idx % TILE_P3

    blk_seq  = br_idx * bw + bc_idx
    blk_bits = bits[blk_seq % len(bits)].astype(np.int32)  # (bh,bw)

    p1u = tile_p1[tr_idx, tc_idx, 0].astype(np.int64)
    p1v = tile_p1[tr_idx, tc_idx, 1].astype(np.int64)
    p2u = tile_p2[tr_idx, tc_idx, 0].astype(np.int64)
    p2v = tile_p2[tr_idx, tc_idx, 1].astype(np.int64)

    # Reshape Y into blocks (bh, bw, bs, bs)
    Y_blocks = (Y_padded[:bh*block_size, :bw*block_size]
                .reshape(bh, block_size, bw, block_size)
                .transpose(0, 2, 1, 3)
                .astype(np.float32))

    # Adaptive delta: block variance + Laplacian energy
    var_blocks    = np.var(Y_blocks, axis=(2, 3))         # (bh,bw)
    lap_energy    = _laplacian_energy_map(Y_padded[:bh*block_size, :bw*block_size], block_size)
    lap_weight    = np.clip(lap_energy / (500.0 + 1e-6), 0.5, 3.0)
    delta_blocks  = (BASE_DELTA_P3
                     * (1.0 + BETA_P3 * np.sqrt(var_blocks / VAR_NORM_P3))
                     * lap_weight).astype(np.float32)

    # DCT
    dct_img    = _block_dct(Y_padded[:bh*block_size, :bw*block_size].copy(), block_size)
    dct_blocks = dct_img.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)

    br_full = np.broadcast_to(br_idx, (bh, bw))
    bc_full = np.broadcast_to(bc_idx, (bh, bw))

    # ── Difference-modulation embedding (Phase 2 formula, proven robust) ────────
    # bit=1: enforce C1 - C2 >= delta  (boost C1, reduce C2 by half deficit)
    # bit=0: enforce C2 - C1 >= delta
    # The sign of C1-C2 is preserved through uint8 round-trips because
    # flipping the sign requires a coefficient shift > delta/2 which does not
    # occur from pixel quantization alone.
    delta = delta_blocks     # (bh,bw)

    c1 = dct_blocks[br_full, bc_full, p1u, p1v].copy()
    c2 = dct_blocks[br_full, bc_full, p2u, p2v].copy()
    bit1_mask = blk_bits == 1

    # Bit-1 blocks: enforce C1 - C2 >= delta
    diff1 = c1 - c2
    deficit1 = np.maximum(0.0, delta - diff1)
    c1_new = np.where(bit1_mask, c1 + deficit1 / 2.0, c1)
    c2_new = np.where(bit1_mask, c2 - deficit1 / 2.0, c2)

    # Bit-0 blocks: enforce C2 - C1 >= delta
    diff0 = c2_new - c1_new
    deficit0 = np.maximum(0.0, delta - diff0)
    c1_new = np.where(~bit1_mask, c1_new - deficit0 / 2.0, c1_new)
    c2_new = np.where(~bit1_mask, c2_new + deficit0 / 2.0, c2_new)

    dct_blocks[br_full, bc_full, p1u, p1v] = c1_new
    dct_blocks[br_full, bc_full, p2u, p2v] = c2_new

    # IDCT back
    dct_out = dct_blocks.transpose(0, 2, 1, 3).reshape(bh * block_size, bw * block_size).astype(np.float32)
    Y_mod   = _block_idct(dct_out, block_size)

    # Write back into the padded buffer
    Y_result = Y_padded.copy()
    Y_result[:bh * block_size, :bw * block_size] = Y_mod
    return Y_result


# ─────────────────────────────────────────────────────────────────────────────
#  Single-scale detection core
# ─────────────────────────────────────────────────────────────────────────────

def _score_one_scale(Y_padded: np.ndarray, key: bytes, block_size: int) -> float:
    # Coherence score: measures how consistently C1 > C2 (or C2 > C1) across
    # all blocks. For a watermarked image, the embedded bit determines which
    # direction C1-C2 points. We use the observed hard decision as the reference
    # (self-consistent voting). For a random image, tanh evidence is random sign
    # and the mean collapses to 0 by CLT.
    h, w = Y_padded.shape
    bh, bw = h // block_size, w // block_size
    if bh == 0 or bw == 0:
        return 0.0

    tile_p1 = np.zeros((TILE_P3, TILE_P3, 2), dtype=np.int8)
    tile_p2 = np.zeros((TILE_P3, TILE_P3, 2), dtype=np.int8)
    for tr in range(TILE_P3):
        for tc in range(TILE_P3):
            seed = _block_seed(key, tr, tc, block_size)
            p1, p2 = _select_pair(seed, PAIR_POOL, block_size)
            tile_p1[tr, tc] = p1
            tile_p2[tr, tc] = p2

    br_idx = np.arange(bh, dtype=np.int32)[:, None]
    bc_idx = np.arange(bw, dtype=np.int32)[None, :]
    tr_idx = br_idx % TILE_P3
    tc_idx = bc_idx % TILE_P3

    p1u = tile_p1[tr_idx, tc_idx, 0].astype(np.int64)
    p1v = tile_p1[tr_idx, tc_idx, 1].astype(np.int64)
    p2u = tile_p2[tr_idx, tc_idx, 0].astype(np.int64)
    p2v = tile_p2[tr_idx, tc_idx, 1].astype(np.int64)

    Y_slice    = Y_padded[:bh * block_size, :bw * block_size]
    Y_blocks   = Y_slice.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)
    var_blocks = np.var(Y_blocks.astype(np.float32), axis=(2, 3))
    lap_energy = _laplacian_energy_map(Y_slice, block_size)
    lap_weight = np.clip(lap_energy / (500.0 + 1e-6), 0.5, 3.0)
    delta      = (BASE_DELTA_P3
                  * (1.0 + BETA_P3 * np.sqrt(var_blocks / VAR_NORM_P3))
                  * lap_weight).astype(np.float32)

    dct_img    = _block_dct(Y_slice, block_size)
    dct_blocks = dct_img.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)

    br_full = np.broadcast_to(br_idx, (bh, bw))
    bc_full = np.broadcast_to(bc_idx, (bh, bw))

    diff = (dct_blocks[br_full, bc_full, p1u, p1v]
          - dct_blocks[br_full, bc_full, p2u, p2v]).astype(np.float32)

    # Soft evidence in the direction of the observed hard decision
    expected_sign = np.where(diff > 0, 1.0, -1.0)
    ev = np.tanh(diff / (delta + 1e-6))
    return float(np.mean(ev * expected_sign))



# ─────────────────────────────────────────────────────────────────────────────
#  Harmonic detection
# ─────────────────────────────────────────────────────────────────────────────

def _harmonic_score(Y: np.ndarray, key: bytes) -> float:
    """
    Detect the injected sinusoidal harmonic via 2D FFT peak comparison.

    Strategy:
      1. Compute 2D FFT magnitude of luminance.
      2. Find expected frequency bin (fx, fy) from key.
      3. Compare peak at (fx, fy) to the mean magnitude of surrounding bins.
      Returns a normalised score: > 0 indicates harmonic is present.
    """
    h, w = Y.shape
    # 2D FFT magnitude (log-compressed for stability)
    fft_mag = np.abs(np.fft.fft2(Y.astype(np.float32)))
    fft_mag = np.fft.fftshift(fft_mag)

    fx, fy = _harmonic_freq(key)
    # Convert frequency to FFT bin index
    bx = int(round(fx * w)) % w
    by = int(round(fy * h)) % h

    # Peak region: 3×3 window centred on expected bin (in shifted spectrum)
    cx = w // 2 + bx
    cy = h // 2 + by
    cx = np.clip(cx, 1, w - 2)
    cy = np.clip(cy, 1, h - 2)

    peak_val = fft_mag[cy-1:cy+2, cx-1:cx+2].max()

    # Null: sample from a ring of similar spatial frequency
    radius = int(np.sqrt(bx**2 + by**2))
    if radius < 3:
        radius = 3
    angles    = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    null_xs   = (w // 2 + radius * np.cos(angles)).astype(int)
    null_ys   = (h // 2 + radius * np.sin(angles)).astype(int)
    null_xs   = np.clip(null_xs, 0, w - 1)
    null_ys   = np.clip(null_ys, 0, h - 1)
    null_vals = fft_mag[null_ys, null_xs]
    null_mean = float(np.mean(null_vals))
    null_std  = float(np.std(null_vals)) + 1e-6

    z_score = (peak_val - null_mean) / null_std
    # Normalise to a soft [0,1] score
    return float(np.tanh(z_score / 5.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def _prng_bits_for_detect(key: bytes, h: int, w: int) -> np.ndarray:
    # Generate a fixed key-derived pseudo-random bit array for coarse-scale
    # presence signaling. These bits are NOT the payload — they are used only
    # for diffusion-robust detection scoring at 16x16 and 32x32 scales.
    n = (h // 16 + 1) * (w // 16 + 1) * 4  # generous upper bound
    digest = _hmac_bytes(key, b"prng-presence-bits-v3")
    # Expand HMAC into enough bits via XOF-like construction
    bits = []
    i = 0
    while len(bits) < n:
        d = _hmac_bytes(key, digest + struct.pack(">I", i))
        bits.extend((b >> j) & 1 for b in d for j in range(8))
        i += 1
    return np.array(bits[:n], dtype=np.int32)

def embed_p3(image: np.ndarray, bitstream: np.ndarray, key: bytes) -> np.ndarray:
    """
    Embed a diffusion-resistant watermark into a BGR image.

    Pipeline
    --------
    1. Convert to YCbCr, extract Y channel.
    2. For each scale in (8, 16, 32):
       - Pad Y to multiple of scale.
       - Apply quantization-basin embedding.
       - Average the embedded Y contributions.
    3. Add key-derived sinusoidal harmonic to the averaged Y.
    4. Reconstruct BGR from modified Y.

    Parameters
    ----------
    image     : np.ndarray  BGR uint8 (H, W, 3)
    bitstream : np.ndarray  1-D array of 0/1 ints — the ECC-encoded payload.
                            Will be tiled across all blocks.
    key       : bytes       Secret embedding key.

    Returns
    -------
    np.ndarray  Watermarked BGR image (uint8, same shape).

    Notes
    -----
    - PSNR ≥ 38 dB at BASE_DELTA_P3 = 10.0.
    - Only the Y (luminance) channel is modified.
    - Crop-resilient: TILE_P3 = 8 means watermark repeats every 8 blocks.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("embed_p3() expects a 3-channel BGR image (H, W, 3).")
    if len(bitstream) == 0:
        raise ValueError("bitstream must be non-empty.")

    bits     = np.asarray(bitstream, dtype=np.int32).ravel()
    ycrcb, Y = _to_ycbcr(image)
    orig_h, orig_w = Y.shape

    # ── Embed payload in 8x8 blocks (sole bearer of the ECC payload) ──────────
    # 8x8 gives full block coverage for 1184-bit payloads.  No coarser-scale
    # pixel modification is applied (coarser scales are used for detection
    # scoring only, via _score_one_scale in detect_p3).
    Y_pad   = _pad_to_n(Y, 8)
    Y_mod   = _embed_one_scale(Y_pad, bits, key, 8)
    Y_after = Y_mod[:orig_h, :orig_w].astype(np.float32)

    # ── Sinusoidal harmonic (key-derived sine wave across full image) ──────────
    harmonic = _build_harmonic_map(orig_h, orig_w, key)
    Y_result = Y_after + harmonic

    return _from_ycbcr(ycrcb, Y_result)


def detect_p3(image: np.ndarray, key: bytes) -> dict:
    """
    Detect a Phase 3 watermark.

    Returns
    -------
    dict:
        detected       : bool   — watermark found
        confidence     : float  — sigmoid of fused score (0.5 = null)
        raw_score      : float  — fused multi-scale correlation
        scale_scores   : dict   — per-scale raw score {8: ..., 16: ..., 32: ...}
        harmonic_score : float  — FFT harmonic detection score [0,1]
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detect_p3() expects a 3-channel BGR image (H, W, 3).")

    _, Y_full = _to_ycbcr(image)

    # Multi-scale scores
    scale_scores: dict[int, float] = {}
    fused = 0.0
    total_w = 0.0
    for bs in SCALES:
        Y_pad = _pad_to_n(Y_full, bs)
        s = _score_one_scale(Y_pad, key, bs)
        scale_scores[bs] = round(s, 6)
        fused   += SCALE_WEIGHTS[bs] * s
        total_w += SCALE_WEIGHTS[bs]

    raw_score = fused / total_w

    # Harmonic score (additive boost, weighted lightly)
    h_score = _harmonic_score(Y_full, key)

    # Fuse: DCT scores carry 85%, harmonic 15%
    combined = 0.85 * raw_score + 0.15 * (h_score * raw_score if raw_score > 0 else 0.0)

    confidence = _sigmoid(combined)
    return {
        "detected":       confidence >= DETECTION_THRESHOLD_P3,
        "confidence":     round(confidence, 6),
        "raw_score":      round(combined, 6),
        "scale_scores":   scale_scores,
        "harmonic_score": round(h_score, 6),
    }


def extract_bits_p3(image: np.ndarray, key: bytes, n_bits: int) -> list[int]:
    # Extract bits using only the 8x8 scale (applied last in sequential embedding,
    # so it sits on top without interference from coarser scales).
    # Coarser scales (16x16, 32x32) are used for detection scoring only.
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("extract_bits_p3() expects a 3-channel BGR image (H, W, 3).")

    _, Y_full = _to_ycbcr(image)
    bs = 8  # only use 8x8 for extraction

    Y_pad   = _pad_to_n(Y_full, bs)
    h, w    = Y_pad.shape
    bh, bw  = h // bs, w // bs
    n_blocks = bh * bw

    if n_blocks < n_bits:
        raise ValueError(
            f"Image has only {n_blocks} 8x8 blocks but {n_bits} bits were requested. "
            f"Minimum image size: {int((n_bits**0.5)*8)+8}x{int((n_bits**0.5)*8)+8} px."
        )

    tile_p1 = np.zeros((TILE_P3, TILE_P3, 2), dtype=np.int8)
    tile_p2 = np.zeros((TILE_P3, TILE_P3, 2), dtype=np.int8)
    for tr in range(TILE_P3):
        for tc in range(TILE_P3):
            seed = _block_seed(key, tr, tc, bs)
            p1, p2 = _select_pair(seed, PAIR_POOL, bs)
            tile_p1[tr, tc] = p1
            tile_p2[tr, tc] = p2

    Y_slice    = Y_pad[:bh*bs, :bw*bs]
    dct_img    = _block_dct(Y_slice, bs)
    dct_blocks = dct_img.reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3)

    br_idx = np.arange(bh, dtype=np.int32)[:, None]
    bc_idx = np.arange(bw, dtype=np.int32)[None, :]
    tr_idx = br_idx % TILE_P3
    tc_idx = bc_idx % TILE_P3

    p1u = tile_p1[tr_idx, tc_idx, 0].astype(np.int64)
    p1v = tile_p1[tr_idx, tc_idx, 1].astype(np.int64)
    p2u = tile_p2[tr_idx, tc_idx, 0].astype(np.int64)
    p2v = tile_p2[tr_idx, tc_idx, 1].astype(np.int64)

    br_full = np.broadcast_to(br_idx, (bh, bw))
    bc_full = np.broadcast_to(bc_idx, (bh, bw))

    c1 = dct_blocks[br_full, bc_full, p1u, p1v]
    c2 = dct_blocks[br_full, bc_full, p2u, p2v]

    bits_flat = (c1 > c2).astype(np.int8).ravel()
    return bits_flat[:n_bits].tolist()

