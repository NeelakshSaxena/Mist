import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.dct import pad_to_8, apply_block_dct, apply_block_idct, rgb_to_y
from src.core.embed import embed_watermark
from src.core.detect import extract_watermark

def optimize_alpha():
    target_dir = "dataset/resized"
    images = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png'))][:10] # Subset for speed
    
    np.random.seed(42)
    dummy_watermark = np.random.randint(0, 2, 64).tolist()
    
    alphas_to_test = [10, 20, 50, 75, 100, 150, 200]
    
    for alpha in alphas_to_test:
        rates = []
        for img_name in images:
            img = cv2.imread(os.path.join(target_dir, img_name))
            Y = rgb_to_y(img)
            Y_padded = pad_to_8(Y)
            h, w = Y.shape
            
            dct_img = apply_block_dct(Y_padded)
            watermarked_dct = embed_watermark(dct_img, dummy_watermark, alpha=alpha)
            watermarked_padded = apply_block_idct(watermarked_dct)
            
            watermarked_uint8 = np.clip(np.round(watermarked_padded[:h, :w]), 0, 255).astype(np.uint8)
            
            Y_det_padded = pad_to_8(watermarked_uint8).astype(np.float32)
            dct_det = apply_block_dct(Y_det_padded)
            
            extracted_bits = extract_watermark(dct_det, num_bits=64, alpha=alpha)
            correct = sum([1 for a, b in zip(dummy_watermark, extracted_bits) if a == b])
            rates.append(correct / 64.0)
            
        print(f"Alpha {alpha}: Average Recovery = {np.mean(rates)*100:.2f}%")

if __name__ == "__main__":
    optimize_alpha()
