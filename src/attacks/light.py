import cv2
import numpy as np

def attack_jpeg(img, quality=50):
    """Apply JPEG compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def attack_resize(img, scale=0.75):
    """Resize image and resize back to original dimensions."""
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Recover original size
    recovered = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return recovered

def attack_crop(img, percent=0.20):
    """Crop from edges."""
    h, w = img.shape[:2]
    
    # Calculate margins to cut
    crop_h = int(h * (percent / 2))
    crop_w = int(w * (percent / 2))
    
    # Crop
    cropped = img[crop_h:h-crop_h, crop_w:w-crop_w]
    return cropped

def attack_brightness(img, percent=15):
    """Increase brightness."""
    # Convert to HSV to nicely adjust brightness without color weirdness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + (percent / 100.0))
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
