"""
src/core/gpu_candidate_ranker.py  –  GPU Top-k Reduction and Scoring

Performs candidate reduction directly on the GPU.
"""

import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None


def gpu_reduce_candidates(
    canary_scores_gpu: 'cp.ndarray',  # [B] array of canary scores
    angles_gpu: 'cp.ndarray',         # [B] array of angles
    scales_gpu: 'cp.ndarray',         # [B] array of scales
    top_k: int = 12
) -> list[tuple[float, float, float]]:
    """
    Selects the top K candidates using GPU-based sorting.
    
    Returns list of (score, angle, scale).
    """
    if not HAS_GPU:
        return []

    # Get sorted indices (descending)
    sorted_idx = cp.argsort(canary_scores_gpu)[::-1]
    
    # Take top K
    top_idx = sorted_idx[:top_k]
    
    scores = canary_scores_gpu[top_idx].get()
    angles = angles_gpu[top_idx].get()
    scales = scales_gpu[top_idx].get()
    
    results = []
    for s, a, sc in zip(scores, angles, scales):
        results.append((float(s), float(a), float(sc)))
        
    return results
