"""
src/attacks/diffusion.py  –  Diffusion Model Attack Simulator

Purpose
-------
Approximate the image distortion caused by an img2img diffusion pass without
requiring actual model weights (~8 GB).  The simulator mimics the key effects:

  1. Blur  — destroys fine-scale frequency relationships (hurts mid-band DCT)
  2. Noise — simulates sampling from the model's distribution
  3. Edge sharpening — model reconstructs semantic edges faithfully
  4. Blending — strength parameter controls original vs. reconstructed mix

Interpretation of `strength`
-----------------------------
  strength = 0.0 → no change (identity)
  strength = 0.3 → mild pass (low denoise steps, content mostly preserved)
  strength = 0.5 → moderate pass (typical img2img usage)
  strength = 0.7 → aggressive pass (heavy reconstruction)
  strength = 1.0 → maximum reconstruction (near pure generation)

Public API
----------
    attack_diffusion_sim(img, strength=0.5, seed=42) → np.ndarray
"""

import cv2
import numpy as np


def attack_diffusion_sim(
    img: np.ndarray,
    strength: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate a diffusion img2img attack at the given strength.

    Mathematical model
    ------------------
    The effect of running img2img at noise level `t` (analogous to strength) is:
        x_t  = sqrt(ᾱ_t) * x_0  +  sqrt(1-ᾱ_t) * ε       (forward process)
        x̂_0  = reconstruct(x_t)                             (reverse process)

    We approximate `reconstruct(·)` as:
        blur(σ) → add noise → edge-sharpen → clip

    Then blend:
        output = (1 - strength) * img + strength * x̂_0

    Parameters
    ----------
    img      : np.ndarray  BGR uint8 image (H, W, 3)
    strength : float       Attack intensity in [0, 1]
    seed     : int         RNG seed for reproducible noise

    Returns
    -------
    np.ndarray  Attacked BGR uint8 image (same shape)
    """
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("attack_diffusion_sim() expects a 3-channel BGR image.")

    if strength < 1e-4:
        return img.copy()

    img_f = img.astype(np.float32)
    rng   = np.random.default_rng(seed)

    # ── 1. Gaussian blur (destroys high and mid frequencies) ──────────────────
    sigma  = strength * 5.0  # σ: 0 → 0, 0.5 → 2.5, 1.0 → 5.0
    ksize  = int(2 * round(3 * sigma) + 1) | 1     # nearest odd ≥ 1
    ksize  = max(ksize, 1)
    if sigma > 0.1:
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)
    else:
        blurred = img_f.copy()

    # ── 2. Gaussian noise (simulates re-sampling from distribution) ───────────
    noise_std = strength * 18.0
    noise     = rng.standard_normal(img_f.shape).astype(np.float32) * noise_std
    noised    = blurred + noise

    # ── 3. Unsharp mask (diffusion model reconstructs edges faithfully) ────────
    # Amount scales with strength: aggressive pass → stronger edge reconstruction
    sharpen_amount = strength * 0.9     # 0 → 0, 0.5 → 0.45, 1.0 → 0.9
    blur_for_sharp = cv2.GaussianBlur(noised, (0, 0), sigmaX=2.0)
    sharpened      = noised + sharpen_amount * (noised - blur_for_sharp)

    # ── 4. JPEG-like quality rounding (diffusion output quantized to uint8) ───
    # Models output via VAE decode → clip to valid pixel range
    reconstructed = np.clip(sharpened, 0.0, 255.0)

    # ── 5. Blend original + reconstruction ────────────────────────────────────
    output = (1.0 - strength) * img_f + strength * reconstructed
    return np.clip(output, 0, 255).astype(np.uint8)


def attack_diffusion_strong(img: np.ndarray, seed: int = 42) -> np.ndarray:
    """Convenience: strength=0.7 (aggressive diffusion pass)."""
    return attack_diffusion_sim(img, strength=0.7, seed=seed)


def attack_diffusion_moderate(img: np.ndarray, seed: int = 42) -> np.ndarray:
    """Convenience: strength=0.5 (typical img2img usage)."""
    return attack_diffusion_sim(img, strength=0.5, seed=seed)


def attack_diffusion_mild(img: np.ndarray, seed: int = 42) -> np.ndarray:
    """Convenience: strength=0.3 (low denoising strength)."""
    return attack_diffusion_sim(img, strength=0.3, seed=seed)
