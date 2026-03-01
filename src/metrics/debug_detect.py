import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.dct import rgb_to_y, pad_to_8, apply_block_dct
from src.core.embed import embed_watermark
from src.core.detect import extract_watermark_spatial
from src.attacks.light import attack_jpeg

def debug_extraction():
    # Load test image
    img = cv2.imread("dataset/images/000000000139.jpg")
def debug_extraction():
    import glob
    img_paths = glob.glob("dataset/resized/*.jpg")[:100]
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_y = rgb_to_y(img)
        img_padded = pad_to_8(img_y).astype(np.float32)
        dct = apply_block_dct(img_padded)
        
        # Dummy payload
        payload = [1,0,1,1, 0,0,1,0] * 8
        alpha = 20
        
        # Embed
        watermarked_dct = embed_watermark(dct, payload, alpha)
        from src.core.dct import apply_block_idct
        watermarked_y = apply_block_idct(watermarked_dct)
        
        # Reconstruct BGR
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_ycbcr[:, :, 0] = np.clip(watermarked_y[:img.shape[0], :img.shape[1]], 0, 255)
        watermarked_bgr = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2BGR)
        
        # Extract Unattacked Baseline
        recovered_baseline = extract_watermark_spatial(rgb_to_y(watermarked_bgr), num_bits=64, alpha=alpha)
        acc_baseline = sum([1 for a, b in zip(payload, recovered_baseline)]) / 64 * 100 if recovered_baseline else 0
        
        # Attack JPEG
        attacked_bgr = attack_jpeg(watermarked_bgr, quality=50)
        recovered_jpeg = extract_watermark_spatial(rgb_to_y(attacked_bgr), num_bits=64, alpha=alpha)
        acc_jpeg = sum([1 for a, b in zip(payload, recovered_jpeg)]) / 64 * 100 if recovered_jpeg else 0
        
        print(f"{os.path.basename(img_path)} | Base: {acc_baseline}% | JPEG: {acc_jpeg}%")

if __name__ == "__main__":
    debug_extraction()
