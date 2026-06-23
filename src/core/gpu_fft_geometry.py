"""
src/core/gpu_fft_geometry.py  –  FFT-based Geometry Estimation

Uses Phase Correlation on Log-Polar transformed FFT magnitude spectrums 
to estimate rotation and scale invariant of translation.
"""

import numpy as np
import cv2

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

from src.core.gpu_accel import _cp


def _logpolar_cv2(image: np.ndarray, center: tuple[float, float], max_radius: float) -> np.ndarray:
    """CPU fallback for log-polar transform."""
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    return cv2.logPolar(image, center, max_radius / np.log(max_radius), flags)


def _logpolar_gpu(image: cp.ndarray, center: tuple[float, float], max_radius: float) -> cp.ndarray:
    """GPU log-polar transform using CuPy."""
    h, w = image.shape
    y, x = cp.mgrid[0:h, 0:w]
    
    # Log-polar grid
    # rho = exp(x / max_radius * log(max_radius))
    # theta = y * 2pi / h
    # This maps the log-polar image back to Cartesian for map_coordinates
    log_base = max_radius / np.log(max_radius)
    
    rho = cp.exp(x / log_base)
    theta = y * 2 * cp.pi / h
    
    map_x = center[0] + rho * cp.cos(theta)
    map_y = center[1] + rho * cp.sin(theta)
    
    coords = cp.stack([map_y, map_x], axis=0)
    return ndi.map_coordinates(image, coords, order=1, mode='constant', cval=0.0)


def _generate_ideal_grid(h: int, w: int, spacing: int = 8) -> np.ndarray:
    """Generate an ideal impulse grid."""
    grid = np.zeros((h, w), dtype=np.float32)
    grid[::spacing, ::spacing] = 1.0
    return grid

def estimate_geometry_fft(
    attacked_y: np.ndarray, 
) -> list[tuple[float, float]]:
    """
    Estimate scale and rotation using FFT phase correlation.
    
    Steps:
      1. Apply Hanning window
      2. Compute FFT magnitude spectrum
      3. Apply high-pass filter (optional, to remove DC and low frequencies)
      4. Log-polar transform
      5. Phase correlation
      6. Extract peak(s) to get (angle, scale)
      
    Returns a list of candidate (angle_deg, scale_factor) tuples.
    """
    if not HAS_GPU:
        return []

    h, w = attacked_y.shape
    reference_y = _generate_ideal_grid(h, w, 8)

    # Move to GPU
    a_gpu = cp.array(attacked_y, dtype=cp.float32)
    r_gpu = cp.array(reference_y, dtype=cp.float32)

    # Window
    hy = cp.hanning(h)
    hx = cp.hanning(w)
    win = hy[:, None] * hx[None, :]

    # FFT Magnitude
    a_fft = cp.abs(cp.fft.fftshift(cp.fft.fft2(a_gpu * win)))
    r_fft = cp.abs(cp.fft.fftshift(cp.fft.fft2(r_gpu * win)))

    # Log Polar
    center = (w / 2.0, h / 2.0)
    max_radius = min(w, h) / 2.0
    
    a_lp = _logpolar_gpu(a_fft, center, max_radius)
    r_lp = _logpolar_gpu(r_fft, center, max_radius)

    # Phase correlation
    a_lp_fft = cp.fft.fft2(a_lp)
    r_lp_fft = cp.fft.fft2(r_lp)
    
    cross_power = r_lp_fft * cp.conj(a_lp_fft)
    cross_power /= cp.maximum(cp.abs(cross_power), 1e-8)
    
    corr = cp.real(cp.fft.ifft2(cross_power))
    
    # Find peaks
    # The shape of corr is (h, w)
    # y-axis is angle, x-axis is log-scale
    
    # We take top N peaks
    N = 3
    flat_idx = cp.argsort(corr.ravel())[-N:][::-1]
    
    candidates = []
    for idx in flat_idx:
        py = int(idx // w)
        px = int(idx % w)
        
        # Calculate angle
        angle_rad = py * 2 * np.pi / h
        angle_deg = np.degrees(angle_rad)
        if angle_deg > 180:
            angle_deg -= 360
            
        # The phase correlation has 180-degree ambiguity
        # We'll return both angle and angle + 180
        angle_deg_alt = angle_deg + 180
        if angle_deg_alt > 180:
            angle_deg_alt -= 360
            
        # Calculate scale
        log_base = max_radius / np.log(max_radius)
        # Shift px to center
        if px > w // 2:
            px_shifted = px - w
        else:
            px_shifted = px
            
        scale_factor = np.exp(px_shifted / log_base)
        
        candidates.append((float(angle_deg), float(scale_factor)))
        candidates.append((float(angle_deg_alt), float(scale_factor)))
        
    return candidates
