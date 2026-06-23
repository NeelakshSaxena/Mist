"""
src/core/gpu_batch_pipeline.py  –  End-to-end batched GPU geometry engine

Orchestrates the entire Phase 5 geometric search directly on the GPU.
Features tensorized batch canary scoring.
"""

import time
import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.fft import dctn
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

from src.core.gpu_geometry import _gpu_affine_transform, _compute_output_size
from src.core.gpu_accel import _cp
from src.core.gpu_candidate_ranker import gpu_reduce_candidates
from src.core.wm_engine_p4 import MT_SIZE, _tile_anchor_bits, _derive_p4_key
from src.core.wm_engine_p3 import _block_seed, _select_pair, PAIR_POOL

def _gpu_batch_dct(y_batch_gpu: 'cp.ndarray') -> 'cp.ndarray':
    """
    Batched block DCT on GPU.
    y_batch_gpu: [B, 256, 256]
    Returns: [B, 32, 32, 8, 8] DCT coefficients
    """
    B, H, W = y_batch_gpu.shape
    bh, bw = 32, 32
    bs = 8
    
    blocks = y_batch_gpu.reshape(B, bh, bs, bw, bs).transpose(0, 1, 3, 2, 4)
    # DCT over the last two axes (axes=(-2, -1))
    dct_blocks = dctn(blocks, type=2, axes=(-2, -1), norm="ortho")
    return dct_blocks

def _prepare_p4_pairs(key: bytes):
    """Prepare PRNG pairs for canary scoring on GPU."""
    p4k = _derive_p4_key(key)
    tile_p1 = np.zeros((8, 8, 2), dtype=np.int64)
    tile_p2 = np.zeros((8, 8, 2), dtype=np.int64)
    for tr in range(8):
        for tc in range(8):
            seed = _block_seed(p4k, tr, tc, 8)
            p1, p2 = _select_pair(seed, PAIR_POOL, 8)
            tile_p1[tr, tc] = p1
            tile_p2[tr, tc] = p2
    return tile_p1, tile_p2

def run_gpu_batch_pipeline(
    image: np.ndarray,
    key: bytes,
    candidates: list[tuple[float, float]],
    batch_size: int = 64
) -> list[tuple[int, float, float]]:
    """
    Run the batched GPU pipeline.
    """
    if not HAS_GPU:
        return []

    from src.core.wm_engine_p3 import _to_ycbcr
    _, Y = _to_ycbcr(image)
    h, w = Y.shape
    cy, cx = h // 2, w // 2

    Y_gpu = cp.array(Y, dtype=cp.float32)
    
    expected_anchor = cp.array(_tile_anchor_bits(key), dtype=cp.int8)
    tile_p1, tile_p2 = _prepare_p4_pairs(key)
    
    # We only need tr=0
    p1u = cp.array(tile_p1[0, :, 0], dtype=cp.int64)
    p1v = cp.array(tile_p1[0, :, 1], dtype=cp.int64)
    p2u = cp.array(tile_p2[0, :, 0], dtype=cp.int64)
    p2v = cp.array(tile_p2[0, :, 1], dtype=cp.int64)
    
    c_idx = cp.arange(8)
    
    br_idx = cp.arange(25)[:, None, None]
    bc_idx = cp.arange(25)[None, :, None]
    c_idx_3d = cp.arange(8)[None, None, :]
    
    all_results = []
    
    # To limit memory, we process chunks of candidates
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        B = len(batch)
        
        # 1. Batched transform (using streams to parallelize transform execution)
        # Allocate batch tensor
        Y_batch = cp.zeros((B, 264, 264), dtype=cp.float32)
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(min(B, 16))]
        
        for b_idx, (angle, sf) in enumerate(batch):
            stream = streams[b_idx % len(streams)]
            with stream:
                out_h, out_w = _compute_output_size(h, w, sf, MT_SIZE)
                Y_corr = _gpu_affine_transform(Y_gpu, angle, sf, out_h, out_w)
                
                # Crop center 264x264
                rh, rw = Y_corr.shape
                rcy, rcx = rh // 2, rw // 2
                y0 = max(0, rcy - 132)
                x0 = max(0, rcx - 132)
                y1 = min(rh, y0 + 264)
                x1 = min(rw, x0 + 264)
                
                # Copy to batch
                crop_h = y1 - y0
                crop_w = x1 - x0
                if crop_h > 0 and crop_w > 0:
                    Y_batch[b_idx, :crop_h, :crop_w] = Y_corr[y0:y1, x0:x1]
                    
        for s in streams:
            s.synchronize()
            
        # 2. Tensorized Canary Scoring
        best_pass_batch = cp.zeros(B, dtype=cp.int32)
        
        px_shifts = [0, 2, 4, 6]
        for px_dy in px_shifts:
            for px_dx in px_shifts:
                Y_shifted = Y_batch[:, px_dy:px_dy+256, px_dx:px_dx+256]
                dct_blocks = _gpu_batch_dct(Y_shifted)  # [B, 32, 32, 8, 8]
                
                # Extract bits
                c1 = dct_blocks[:, :, :, p1u, p1v]  # [B, 32, 32, 8]
                c2 = dct_blocks[:, :, :, p2u, p2v]  # [B, 32, 32, 8]
                all_bits = (c1 > c2).astype(cp.int8)
                
                # Anchor matching
                anchor_bits = all_bits[:, br_idx, bc_idx + c_idx_3d, c_idx_3d] # [B, 25, 25, 8]
                matches = (anchor_bits == expected_anchor).sum(axis=3) # [B, 25, 25]
                pass_mask = (matches >= 5).astype(cp.int32)
                
                # Sum over 5 tiles cross
                for br in range(8, 17):
                    for bc in range(8, 17):
                        n_pass = (
                            pass_mask[:, br, bc] +
                            pass_mask[:, br-8, bc] +
                            pass_mask[:, br+8, bc] +
                            pass_mask[:, br, bc-8] +
                            pass_mask[:, br, bc+8]
                        )
                        best_pass_batch = cp.maximum(best_pass_batch, n_pass)
        
        # Retrieve scores
        scores = best_pass_batch.get()
        for b_idx, (angle, sf) in enumerate(batch):
            all_results.append((int(scores[b_idx]), angle, sf))
            
    # Free GPU memory
    del Y_gpu
    del Y_batch
    cp.get_default_memory_pool().free_all_blocks()
    
    # CPU reduction for top-k is fast enough, but we use the ranker logic
    all_results.sort(key=lambda x: x[0], reverse=True)
    return all_results
