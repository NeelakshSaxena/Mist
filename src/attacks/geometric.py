"""
src/attacks/geometric.py  –  Geometric Attack Functions

Phase 5 attack suite: rotation, scaling, perspective warp, crop+resize combos.

These are used both as test attacks in the validation suite and as the
search space for Phase 5's geometry-invariant detector.

Public API
----------
    rotate(image, angle_deg, *, crop_black=True) → np.ndarray
    scale(image, factor) → np.ndarray
    scale_to(image, target_w, target_h) → np.ndarray
    crop_and_resize(image, crop_frac, target_size=None) → np.ndarray
    perspective_warp(image, strength=0.1, seed=42) → np.ndarray
    random_geometric(image, seed=42) → np.ndarray, dict
"""

import cv2
import numpy as np


def rotate(
    image: np.ndarray,
    angle_deg: float,
    *,
    crop_black: bool = True,
) -> np.ndarray:
    """
    Rotate image by angle_deg degrees (counter-clockwise positive).

    Parameters
    ----------
    image      : np.ndarray  BGR uint8 (H, W, 3)
    angle_deg  : float       Rotation angle in degrees
    crop_black : bool        If True, crop the rotated image to remove black
                             borders (largest inscribed axis-aligned rectangle).
                             If False, return the full rotated canvas.

    Returns
    -------
    np.ndarray  Rotated BGR image (uint8).
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    if not crop_black:
        # Expand canvas to fit entire rotated image
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0
        return cv2.warpAffine(
            image, M, (new_w, new_h),
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # Rotate with same canvas size, then crop the largest inscribed rectangle
    rotated = cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Compute largest inscribed rectangle after rotation
    angle_rad = abs(angle_deg) * np.pi / 180.0
    if angle_rad < 1e-6:
        return rotated

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if w >= h:
        # Landscape or square
        if h * sin_a <= w * cos_a:
            # Common case
            crop_w = int(h / (sin_a + cos_a * (h / w)) * (w / h))
            crop_h = int(h / (sin_a * (w / h) + cos_a))
        else:
            crop_w = int(w / (sin_a * (h / w) + cos_a))
            crop_h = int(w / (sin_a + cos_a * (w / h)) * (h / w))
    else:
        if w * sin_a <= h * cos_a:
            crop_w = int(w / (sin_a * (h / w) + cos_a))
            crop_h = int(w / (sin_a + cos_a * (w / h)) * (h / w))
        else:
            crop_w = int(h / (sin_a + cos_a * (h / w)) * (w / h))
            crop_h = int(h / (sin_a * (w / h) + cos_a))

    # Clamp to valid range
    crop_w = max(8, min(crop_w, w))
    crop_h = max(8, min(crop_h, h))

    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return rotated[y0:y0 + crop_h, x0:x0 + crop_w].copy()


def scale(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Scale an image by a given factor.

    Parameters
    ----------
    image  : np.ndarray  BGR uint8 (H, W, 3)
    factor : float       Scale factor (e.g. 0.5 = half size, 2.0 = double)

    Returns
    -------
    np.ndarray  Scaled BGR image (uint8).
    """
    h, w = image.shape[:2]
    new_w = max(8, int(w * factor))
    new_h = max(8, int(h * factor))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def scale_to(
    image: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Scale image to exact target dimensions."""
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def crop_and_resize(
    image: np.ndarray,
    crop_frac: float,
    target_size: tuple[int, int] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Randomly crop a fraction of the image and optionally resize to target.

    Parameters
    ----------
    image       : np.ndarray  BGR uint8 (H, W, 3)
    crop_frac   : float       Fraction of area to KEEP (e.g. 0.7 = keep 70%)
    target_size : (w, h) or None.  If given, resize cropped region to this size.
    seed        : int         Random seed for crop position.

    Returns
    -------
    np.ndarray  Cropped (and optionally resized) BGR image.
    """
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)

    side_frac = np.sqrt(crop_frac)
    crop_w = max(8, int(w * side_frac))
    crop_h = max(8, int(h * side_frac))

    x0 = rng.integers(0, max(1, w - crop_w + 1))
    y0 = rng.integers(0, max(1, h - crop_h + 1))

    cropped = image[y0:y0 + crop_h, x0:x0 + crop_w].copy()

    if target_size is not None:
        cropped = cv2.resize(
            cropped, target_size, interpolation=cv2.INTER_LANCZOS4
        )

    return cropped


def perspective_warp(
    image: np.ndarray,
    strength: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply a random perspective warp (simulates photographing a screen).

    Parameters
    ----------
    image    : np.ndarray  BGR uint8 (H, W, 3)
    strength : float       Warp magnitude (0.0 = none, 0.2 = significant)
    seed     : int         Random seed for warp parameters.

    Returns
    -------
    np.ndarray  Warped BGR image (same size).
    """
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)

    # Source corners
    src = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ], dtype=np.float32)

    # Perturb destination corners
    perturbation = rng.uniform(-strength, strength, (4, 2)).astype(np.float32)
    perturbation[:, 0] *= w
    perturbation[:, 1] *= h
    dst = src + perturbation

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image, M, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def random_geometric(
    image: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Apply a random combination of geometric transforms.

    Returns the transformed image and a dict of applied parameters.
    """
    rng = np.random.default_rng(seed)
    params = {}

    result = image.copy()

    # Random rotation
    angle = float(rng.uniform(-15, 15))
    params["rotation_deg"] = angle
    result = rotate(result, angle)

    # Random scale
    factor = float(rng.uniform(0.7, 1.3))
    params["scale_factor"] = factor
    result = scale(result, factor)

    # Maybe random crop
    if rng.random() > 0.5:
        crop_frac = float(rng.uniform(0.6, 0.9))
        params["crop_frac"] = crop_frac
        result = crop_and_resize(result, crop_frac, seed=seed + 100)

    return result, params
