import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.dct import pad_to_8, apply_block_dct, apply_block_idct, rgb_to_y
from src.core.embed import embed_watermark
from src.core.detect import extract_watermark_spatial
from src.attacks.light import attack_jpeg, attack_resize, attack_crop, attack_brightness

def run_phase_e_tests():
    target_dir = "dataset/resized"
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    images = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png'))]
    if len(images) > 100:
        images = images[:100]
        
    # Use a fixed dummy payload without altering the global numpy random state 
    # to avoid interfering with any spatial hashing functions in embed/detect.
    dummy_watermark = [1,0,1,1, 0,0,1,0] * 8
    alpha = 20
    
    attacks = {
        'None': lambda img: img,
        'JPEG_50': lambda img: attack_jpeg(img, 50),
        'Resize_0.75': lambda img: attack_resize(img, 0.75),
        'Crop_20': lambda img: attack_crop(img, 0.20),
        'Brightness_15': lambda img: attack_brightness(img, 15),
    }
    
    results = []
    
    print(f"Running Phase E test matrix on {len(images)} images...")
    
    for img_name in tqdm(images):
        img_path = os.path.join(target_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
            
        # Baseline transform & embedding
        Y = rgb_to_y(img)
        Y_padded = pad_to_8(Y)
        h, w = Y.shape
        
        dct_img = apply_block_dct(Y_padded)
        watermarked_dct = embed_watermark(dct_img, dummy_watermark, alpha=alpha)
        watermarked_padded = apply_block_idct(watermarked_dct)
        
        watermarked_Y = watermarked_padded[:h, :w]
        # Reconstruct BGR
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycbcr[:, :, 0] = np.clip(watermarked_Y[:h, :w], 0, 255)
        watermarked_bgr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
        
        for attack_name, attack_fn in attacks.items():
            attacked_bgr = attack_fn(watermarked_bgr)
            
            # Detection process (Spatial extraction natively handles padding, DCT, and grid alignment)
            attacked_Y = rgb_to_y(attacked_bgr)
            extracted_bits = extract_watermark_spatial(attacked_Y, num_bits=64, alpha=alpha)
            
            # Accuracy
            correct = sum([1 for a, b in zip(dummy_watermark, extracted_bits) if a == b])
            accuracy = correct / len(dummy_watermark)
            
            results.append({
                'image': img_name,
                'attack': attack_name,
                'bit_accuracy': accuracy
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(log_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nPhase E Testing Complete! Logs saved to: {csv_path}")
    
    # Print a quick summary
    summary = df.groupby('attack')['bit_accuracy'].mean().reset_index()
    summary['bit_accuracy'] = (summary['bit_accuracy'] * 100).round(2).astype(str) + "%"
    print("\n--- Summary ---")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_phase_e_tests()
