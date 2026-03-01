import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys

# Setup imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.dct import pad_to_8, apply_block_dct, apply_block_idct, rgb_to_y

def validate_dct():
    target_dir = "dataset/resized"
    # Ensure dir exists or let it throw
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        print("Please run data preprocessing first.")
        return

    images = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png'))]
    if len(images) > 100:
        images = images[:100]
        
    if not images:
        print("No images found for validation.")
        return
    
    psnrs = []
    
    print(f"Validating DCT reconstruction over {len(images)} images...")
    
    for img_name in tqdm(images):
        img_path = os.path.join(target_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Step 1: RGB -> Y channel
        Y = rgb_to_y(img)
        
        # Step 2: Pad
        Y_padded = pad_to_8(Y)
        h, w = Y.shape
        
        # Step 3: Block DCT
        dct_img = apply_block_dct(Y_padded)
        
        # Step 4: Block Inverse DCT
        reconstructed_padded = apply_block_idct(dct_img)
        
        # Unpad to original size
        reconstructed = reconstructed_padded[:h, :w]
        
        # Reconvert to uint8 by rounding and clipping
        reconstructed_uint8 = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
        
        # Compute PSNR
        p = psnr(Y, reconstructed_uint8, data_range=255)
        psnrs.append(p)
        
    avg_psnr = np.mean(psnrs)
    print(f"\nAverage PSNR over {len(psnrs)} images: {avg_psnr:.4f} dB")
    
    if avg_psnr > 50:
        print("✅ Validation Passed: Target PSNR > 50 dB achieved.")
    else:
        print("❌ Validation Failed: Target PSNR > 50 dB not achieved.")

if __name__ == "__main__":
    validate_dct()
