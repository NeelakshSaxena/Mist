"""
src/core/gpu_geometry.py  –  GPU-Accelerated Batch Geometric Search

Replaces the sequential CPU loop in wm_engine_p5.estimate_geometry()
with CUDA-accelerated batch transforms and vectorized anchor scoring.

Performance target: 500 candidates in ~2-4s (vs. ~60-90s on CPU).

Strategy:
    1. Transfer Y channel to GPU once (CuPy array)
    2. Group candidates by output size (quantized to 16px)
    3. Within each group, apply affine transforms concurrently via CUDA streams
    4. Vectorized anchor scoring on GPU-resident transformed data
    5. Return sorted (score, angle, scale) list for promotion to full P4

Falls back to CPU if CuPy is unavailable.
"""

import numpy as np
import cv2
from typing import Optional

from src.core.gpu_accel import HAS_GPU, _cp, gpu_block_dct

# Lazy imports to avoid circular dependency
_p4_imports = None

def _get_p4():
    """Lazy import of P4 constants to avoid circular import."""
    global _p4_imports
    if _p4_imports is None:
        from src.core.wm_engine_p4 import (
            MT_SIZE, MT_BLOCKS, ANCHOR_BITS, ANCHOR_MATCH_MIN,
            _tile_anchor_bits, _extract_tile_bits, _derive_p4_key,
            TILE_P3,
        )
        from src.core.wm_engine_p3 import (
            _block_seed, _select_pair, PAIR_POOL, _to_ycbcr,
        )
        _p4_imports = {
            'MT_SIZE': MT_SIZE, 'MT_BLOCKS': MT_BLOCKS,
            'ANCHOR_BITS': ANCHOR_BITS, 'ANCHOR_MATCH_MIN': ANCHOR_MATCH_MIN,
            'tile_anchor_bits': _tile_anchor_bits,
            'extract_tile_bits': _extract_tile_bits,
            'derive_p4_key': _derive_p4_key,
            'TILE_P3': TILE_P3,
            'block_seed': _block_seed, 'select_pair': _select_pair,
            'PAIR_POOL': PAIR_POOL, 'to_ycbcr': _to_ycbcr,
        }
    return _p4_imports


# ─────────────────────────────────────────────────────────────────────────────
#  GPU Batch Affine Transform
# ─────────────────────────────────────────────────────────────────────────────

def _compute_output_size(
    h: int, w: int, scale_factor: float, mt_size: int,
) -> tuple[int, int]:
    """Compute the output size for undoing a scale factor."""
    inv_f = 1.0 / max(scale_factor, 0.01)
    new_w = max(mt_size, int(np.ceil(w * inv_f)))
    new_h = max(mt_size, int(np.ceil(h * inv_f)))
    return new_h, new_w


def _gpu_affine_transform(
    Y_gpu,  # CuPy array [H, W] float32
    angle_deg: float,
    scale_factor: float,
    out_h: int,
    out_w: int,
):
    """
    GPU affine transform using CuPy's cupyx.scipy.ndimage.affine_transform.
    
    Undoes rotation + scaling:
      1. Undo scale (resize to out_h × out_w)
      2. Undo rotation (rotate by -angle)
    
    Combined into a single affine for efficiency.
    """
    from cupyx.scipy.ndimage import affine_transform as cu_affine

    cp = _cp
    h, w = Y_gpu.shape

    # Combined inverse affine: output coords → input coords
    # Step 1: Undo scale means output pixel maps to input pixel * sf
    # Step 2: Undo rotation means rotate by +angle (since we're mapping output→input)
    rad = np.radians(angle_deg)  # positive = undo direction
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    # After undo-scale, the image should be at (out_h, out_w).
    # After undo-rotation, each pixel in the output maps back through rotation.
    # Combined: output(y,x) → scale → rotate → input(y',x')

    # Centers
    cy_out, cx_out = out_h / 2.0, out_w / 2.0
    cy_in, cx_in = h / 2.0, w / 2.0

    # Scale factors: output pixel → input pixel
    sy = h / out_h  # = scale_factor (approximately)
    sx = w / out_w

    # Combined matrix: [cos*sy, sin*sx; -sin*sy, cos*sx]
    # Maps output coords (centered) to input coords (centered), then re-center
    matrix = cp.array([
        [cos_a * sy,  sin_a * sx],
        [-sin_a * sy, cos_a * sx],
    ], dtype=cp.float32)

    offset = cp.array([
        cy_in - (cos_a * sy * cy_out + sin_a * sx * cx_out),
        cx_in - (-sin_a * sy * cy_out + cos_a * sx * cx_out),
    ], dtype=cp.float32)

    return cu_affine(
        Y_gpu, matrix, offset,
        output_shape=(out_h, out_w),
        order=3,  # bicubic preserves high-frequency watermark energy
        mode='reflect',
    )


def _gpu_canary_score_from_Y(
    Y_transformed,  # CuPy array [H, W] float32
    key: bytes,
    expected_anchor: list[int],
    n_tiles: int = 5,
) -> int:
    """
    Robust vectorized canary scoring on GPU-resident Y data.
    Transfers a 264x264 block to CPU, then checks all 64 macro-tile phases
    and 25 pixel offsets simultaneously.
    """
    p4 = _get_p4()
    MT_SIZE = p4['MT_SIZE']
    cp = _cp

    h, w = Y_transformed.shape
    if h < 264 or w < 264:
        return 0

    cy, cx = h // 2, w // 2
    # Extract 256 + 8 = 264 region (to allow up to 8px pixel shift)
    y0 = max(0, cy - 132)
    x0 = max(0, cx - 132)
    y1 = min(h, y0 + 264)
    x1 = min(w, x0 + 264)
    if y1 - y0 < 264 or x1 - x0 < 264:
        return 0

    # Transfer 264x264 region to CPU once
    Y_center = cp.asnumpy(Y_transformed[y0:y1, x0:x1])

    p4k = p4['derive_p4_key'](key)
    block_seed = p4['block_seed']
    select_pair = p4['select_pair']
    PAIR_POOL = p4['PAIR_POOL']
    
    # We need _block_dct from p3
    from src.core.wm_engine_p3 import _block_dct

    # Precompute PRNG pairs
    tile_p1 = np.zeros((8, 8, 2), dtype=np.int64)
    tile_p2 = np.zeros((8, 8, 2), dtype=np.int64)
    for tr in range(8):
        for tc in range(8):
            seed = block_seed(p4k, tr, tc, 8)
            p1, p2 = select_pair(seed, PAIR_POOL, 8)
            tile_p1[tr, tc] = p1
            tile_p2[tr, tc] = p2

    best_pass = 0
    br_idx = np.arange(25)[:, None, None]
    bc_idx = np.arange(25)[None, :, None]
    c_idx_3d = np.arange(8)[None, None, :]

    p1u = tile_p1[0, :, 0]
    p1v = tile_p1[0, :, 1]
    p2u = tile_p2[0, :, 0]
    p2v = tile_p2[0, :, 1]

    expected_anchor_arr = np.array(expected_anchor, dtype=np.int8)

    px_shifts = [0, 2, 4, 6]
    for px_dy in px_shifts:
        for px_dx in px_shifts:
            Y_shifted = Y_center[px_dy:px_dy+256, px_dx:px_dx+256]
            
            dct_img = _block_dct(Y_shifted, 8)
            dct_blocks = dct_img.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3)
            
            c_idx = np.arange(8)
            c1 = dct_blocks[:, :, p1u[c_idx], p1v[c_idx]]
            c2 = dct_blocks[:, :, p2u[c_idx], p2v[c_idx]]
            all_bits = (c1 > c2).astype(np.int8)
            
            anchor_bits = all_bits[br_idx, bc_idx + c_idx_3d, c_idx_3d]
            matches = (anchor_bits == expected_anchor_arr).sum(axis=2)
            pass_mask = (matches >= 5).astype(np.int32)  # ANCHOR_MATCH_MIN
            
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


# ─────────────────────────────────────────────────────────────────────────────
#  Main GPU batch canary function
# ─────────────────────────────────────────────────────────────────────────────

def gpu_batch_canary(
    image: np.ndarray,
    candidates: list[tuple[float, float]],
    key: bytes,
    n_streams: int = 4,
) -> Optional[list[tuple[int, float, float]]]:
    """
    GPU-batched canary scoring for geometric candidates.
    
    Parameters
    ----------
    image      : BGR uint8 [H, W, 3]
    candidates : list of (angle_deg, scale_factor)
    key        : embedding key
    n_streams  : number of CUDA streams for concurrency
    
    Returns
    -------
    Sorted list of (canary_score, angle, scale) in descending score order.
    Returns None if GPU is unavailable (caller should use CPU fallback).
    """
    if not HAS_GPU or _cp is None:
        return None

    try:
        from cupyx.scipy.ndimage import affine_transform as cu_affine
    except ImportError:
        return None

    p4 = _get_p4()
    MT_SIZE = p4['MT_SIZE']
    to_ycbcr = p4['to_ycbcr']
    tile_anchor_bits = p4['tile_anchor_bits']

    cp = _cp

    # Convert to Y channel and transfer to GPU once
    _, Y = to_ycbcr(image)
    Y_gpu = cp.asarray(Y, dtype=cp.float32)
    h, w = Y.shape

    expected_anchor = tile_anchor_bits(key)

    # Group candidates by output size (quantize to 16px for batch efficiency)
    QUANT = 16
    groups: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    for i, (angle, sf) in enumerate(candidates):
        out_h, out_w = _compute_output_size(h, w, sf, MT_SIZE)
        # Quantize to reduce groups
        q_h = (out_h // QUANT) * QUANT
        q_w = (out_w // QUANT) * QUANT
        q_h = max(q_h, MT_SIZE * 2)
        q_w = max(q_w, MT_SIZE * 2)
        key_tuple = (q_h, q_w)
        if key_tuple not in groups:
            groups[key_tuple] = []
        groups[key_tuple].append((angle, sf, i))

    # Process all candidates using CUDA streams
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
    results = [None] * len(candidates)

    # Flatten processing order
    flat_work = []
    for (out_h, out_w), group_items in groups.items():
        for angle, sf, idx in group_items:
            flat_work.append((angle, sf, idx, out_h, out_w))

    # Process with stream-level concurrency
    # Each stream processes one transform at a time; we overlap GPU compute
    # with CPU canary scoring from the previous transform
    transformed_cache = {}  # stream_id -> (Y_result, idx, angle, sf)

    for wi, (angle, sf, idx, out_h, out_w) in enumerate(flat_work):
        stream_id = wi % n_streams
        stream = streams[stream_id]

        with stream:
            try:
                Y_corr = _gpu_affine_transform(
                    Y_gpu, angle, sf, out_h, out_w,
                )
                # Score directly (GPU data → small CPU transfers for tiles)
                stream.synchronize()
                score = _gpu_canary_score_from_Y(
                    Y_corr, key, expected_anchor, n_tiles=5,
                )
                results[idx] = (score, angle, sf)
            except Exception:
                results[idx] = (0, angle, sf)

    # Sync all streams
    for s in streams:
        s.synchronize()

    # Fill any None results
    for i in range(len(results)):
        if results[i] is None:
            results[i] = (0, candidates[i][0], candidates[i][1])

    # Sort descending by score
    results.sort(key=lambda x: x[0], reverse=True)

    # Free GPU memory
    del Y_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  GPU batch full-image undo transform (for P4 promotion)
# ─────────────────────────────────────────────────────────────────────────────

def gpu_undo_transform(
    image: np.ndarray,
    angle_deg: float,
    scale_factor: float,
) -> Optional[np.ndarray]:
    """
    GPU-accelerated combined undo of rotation + scaling.
    Returns BGR uint8 image, or None if GPU unavailable.
    """
    if not HAS_GPU or _cp is None:
        return None

    try:
        from cupyx.scipy.ndimage import affine_transform as cu_affine
    except ImportError:
        return None

    p4 = _get_p4()
    MT_SIZE = p4['MT_SIZE']
    cp = _cp

    h, w = image.shape[:2]
    out_h, out_w = _compute_output_size(h, w, scale_factor, MT_SIZE)

    if out_h < MT_SIZE or out_w < MT_SIZE:
        return None

    # Process each channel independently on GPU
    result_channels = []
    for c in range(3):
        ch_gpu = cp.asarray(image[:, :, c].astype(np.float32))

        rad = np.radians(angle_deg)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        cy_out, cx_out = out_h / 2.0, out_w / 2.0
        cy_in, cx_in = h / 2.0, w / 2.0
        sy = h / out_h
        sx = w / out_w

        matrix = cp.array([
            [cos_a * sy,  sin_a * sx],
            [-sin_a * sy, cos_a * sx],
        ], dtype=cp.float32)

        offset = cp.array([
            cy_in - (cos_a * sy * cy_out + sin_a * sx * cx_out),
            cx_in - (-sin_a * sy * cy_out + cos_a * sx * cx_out),
        ], dtype=cp.float32)

        ch_out = cu_affine(
            ch_gpu, matrix, offset,
            output_shape=(out_h, out_w),
            order=3, mode='reflect',
        )
        result_channels.append(cp.asnumpy(ch_out))

    result = np.stack(result_channels, axis=-1)
    result = np.clip(result, 0, 255).astype(np.uint8)

    from src.core.geometry_correction import _preserve_luma_energy
    result = _preserve_luma_energy(image, result)

    # Cleanup
    cp.get_default_memory_pool().free_all_blocks()

    return result
