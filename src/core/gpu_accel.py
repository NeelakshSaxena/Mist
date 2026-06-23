"""
src/core/gpu_accel.py  –  GPU Acceleration Utilities for Mist

Provides CUDA-accelerated operations via CuPy with automatic CPU fallback.
If CuPy kernel compilation fails (missing nvrtc), falls back transparently.

Key accelerated operations:
    - Block DCT/IDCT (used in scoring and bit extraction)
    - Batch geometric transforms + canary scoring

Usage:
    from src.core.gpu_accel import gpu_block_dct, HAS_GPU
"""

import warnings
import numpy as np
import cv2
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
#  GPU availability detection
# ─────────────────────────────────────────────────────────────────────────────

HAS_GPU = False
_cp = None
_gpu_dctn = None
_gpu_idctn = None

def _probe_gpu() -> bool:
    """Test that CuPy can actually compile and run kernels, not just import."""
    global HAS_GPU, _cp, _gpu_dctn, _gpu_idctn
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp
            from cupyx.scipy.fft import dctn as gd, idctn as gi
            # Real kernel test — this triggers nvrtc compilation
            t = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
            _ = cp.sum(t * t)  # forces kernel compile
            cp.cuda.Stream.null.synchronize()
            _cp = cp
            _gpu_dctn = gd
            _gpu_idctn = gi
            # Test actual DCT to be safe
            test_block = cp.zeros((8, 8), dtype=cp.float32)
            _ = gd(test_block.reshape(1, 1, 8, 8), type=2, axes=(-2, -1), norm="ortho")
            cp.cuda.Stream.null.synchronize()
            HAS_GPU = True
            pool = cp.get_default_memory_pool()
            pool.set_limit(size=2 * 1024**3)
            return True
    except Exception:
        HAS_GPU = False
        _cp = None
        _gpu_dctn = None
        _gpu_idctn = None
        return False

# Probe on import — sets HAS_GPU
_probe_gpu()


def gpu_available() -> bool:
    return HAS_GPU


# ─────────────────────────────────────────────────────────────────────────────
#  CPU Block DCT / IDCT (always available)
# ─────────────────────────────────────────────────────────────────────────────

def cpu_block_dct(y_float: np.ndarray, bs: int) -> np.ndarray:
    """CPU block DCT (scipy)."""
    from scipy.fft import dctn
    h, w = y_float.shape
    bh, bw = h // bs, w // bs
    blocks = y_float[:bh*bs, :bw*bs].reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3)
    dct_blocks = dctn(blocks, type=2, axes=(-2, -1), norm="ortho")
    return dct_blocks.transpose(0, 2, 1, 3).reshape(bh * bs, bw * bs).astype(np.float32)


def cpu_block_idct(dct_img: np.ndarray, bs: int) -> np.ndarray:
    """CPU block IDCT (scipy)."""
    from scipy.fft import idctn
    h, w = dct_img.shape
    bh, bw = h // bs, w // bs
    blocks = dct_img[:bh*bs, :bw*bs].reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3)
    idct_blocks = idctn(blocks, type=2, axes=(-2, -1), norm="ortho")
    return idct_blocks.transpose(0, 2, 1, 3).reshape(bh * bs, bw * bs).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  GPU Block DCT / IDCT (with automatic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def gpu_block_dct(y_float: np.ndarray, bs: int) -> np.ndarray:
    """
    Block DCT — tries GPU first, falls back to CPU on any failure.
    Input: (H, W) float32 array, H and W must be multiples of bs.
    Returns: (H, W) float32 DCT coefficients.
    """
    global HAS_GPU
    if not HAS_GPU:
        return cpu_block_dct(y_float, bs)
    try:
        h, w = y_float.shape
        bh, bw = h // bs, w // bs
        crop = y_float[:bh * bs, :bw * bs]
        y_gpu = _cp.asarray(crop, dtype=_cp.float32)
        blocks = y_gpu.reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3)
        dct_blocks = _gpu_dctn(blocks, type=2, axes=(-2, -1), norm="ortho")
        result = dct_blocks.transpose(0, 2, 1, 3).reshape(bh * bs, bw * bs)
        return _cp.asnumpy(result).astype(np.float32)
    except Exception:
        HAS_GPU = False  # disable GPU for rest of session
        return cpu_block_dct(y_float, bs)


def gpu_block_idct(dct_img: np.ndarray, bs: int) -> np.ndarray:
    """Block IDCT — tries GPU first, falls back to CPU."""
    global HAS_GPU
    if not HAS_GPU:
        return cpu_block_idct(dct_img, bs)
    try:
        h, w = dct_img.shape
        bh, bw = h // bs, w // bs
        d_gpu = _cp.asarray(dct_img[:bh * bs, :bw * bs], dtype=_cp.float32)
        blocks = d_gpu.reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3)
        idct_blocks = _gpu_idctn(blocks, type=2, axes=(-2, -1), norm="ortho")
        result = idct_blocks.transpose(0, 2, 1, 3).reshape(bh * bs, bw * bs)
        return _cp.asnumpy(result).astype(np.float32)
    except Exception:
        HAS_GPU = False
        return cpu_block_idct(dct_img, bs)


#  Batch GPU scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_score_one_scale_gpu(
    Y_base: np.ndarray,
    key: bytes,
    block_size: int,
    tile_p1: np.ndarray,
    tile_p2: np.ndarray,
    TILE_P3: int,
) -> float:
    """
    GPU-accelerated _score_one_scale equivalent.
    Uses pre-computed pair tables to avoid redundant derivation.
    """
    h, w = Y_base.shape
    bh, bw = h // block_size, w // block_size
    if bh == 0 or bw == 0:
        return 0.0

    br_idx = np.arange(bh, dtype=np.int32)[:, None]
    bc_idx = np.arange(bw, dtype=np.int32)[None, :]
    tr_idx = br_idx % TILE_P3
    tc_idx = bc_idx % TILE_P3

    p1u = tile_p1[tr_idx, tc_idx, 0].astype(np.int64)
    p1v = tile_p1[tr_idx, tc_idx, 1].astype(np.int64)
    p2u = tile_p2[tr_idx, tc_idx, 0].astype(np.int64)
    p2v = tile_p2[tr_idx, tc_idx, 1].astype(np.int64)

    Y_slice = Y_base[:bh * block_size, :bw * block_size]
    dct_img = gpu_block_dct(Y_slice.astype(np.float32), block_size)
    dct_blocks = dct_img.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)

    br_full = np.broadcast_to(br_idx, (bh, bw))
    bc_full = np.broadcast_to(bc_idx, (bh, bw))

    diff = (dct_blocks[br_full, bc_full, p1u, p1v]
            - dct_blocks[br_full, bc_full, p2u, p2v]).astype(np.float32)

    expected_sign = np.where(diff > 0, 1.0, -1.0)
    ev = np.tanh(diff / 10.0)  # approximate delta normalization
    return float(np.mean(ev * expected_sign))


# ─────────────────────────────────────────────────────────────────────────────
#  Fast geometric transform (combined affine)
# ─────────────────────────────────────────────────────────────────────────────

def undo_transform_fast(
    image: np.ndarray,
    angle_deg: float,
    scale_factor: float,
    mt_size: int = 64,
) -> np.ndarray | None:
    """
    Undo rotation + scaling with a single combined affine.
    Returns None if result would be too small.
    """
    h, w = image.shape[:2] if image.ndim == 3 else image.shape
    inv_f = 1.0 / max(scale_factor, 0.01)
    new_w = max(mt_size, int(np.ceil(w * inv_f)))
    new_h = max(mt_size, int(np.ceil(h * inv_f)))

    if new_w < mt_size * 2 or new_h < mt_size * 2:
        return None

    interp = cv2.INTER_AREA if inv_f < 1.0 else cv2.INTER_LANCZOS4
    result = cv2.resize(image, (new_w, new_h), interpolation=interp)

    if abs(angle_deg) > 0.01:
        rh, rw = result.shape[:2] if result.ndim == 3 else result.shape
        cx, cy = rw / 2.0, rh / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
        result = cv2.warpAffine(
            result, M, (rw, rh),
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return result


def sync_gpu():
    """Ensure all GPU operations are complete."""
    if HAS_GPU:
        _cp.cuda.Stream.null.synchronize()
