import numpy as np
from scipy.fftpack import dct, idct
import cv2

def rgb_to_y(img):
    """Convert RGB (BGR in OpenCV) to Y channel (luminance)."""
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return ycbcr[:, :, 0]

def pad_to_8(img):
    """Pad image dimensions to be multiples of 8."""
    h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')

def dct2(block):
    """Apply 2D Discrete Cosine Transform on a block."""
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    """Apply 2D Inverse Discrete Cosine Transform on a block."""
    return cv2.idct(block.astype(np.float32))

def apply_block_dct(img):
    """Apply 8x8 block-wise 2D DCT on an image."""
    h, w = img.shape
    dct_img = np.zeros_like(img, dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8].astype(np.float32)
            dct_img[i:i+8, j:j+8] = dct2(block)
    return dct_img

def apply_block_idct(dct_img):
    """Apply 8x8 block-wise 2D IDCT on a DCT image."""
    h, w = dct_img.shape
    img = np.zeros_like(dct_img, dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_img[i:i+8, j:j+8]
            img[i:i+8, j:j+8] = idct2(block)
    return img
