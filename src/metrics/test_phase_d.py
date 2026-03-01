import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Setup imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.dct import pad_to_8, apply_block_dct, apply_block_idct, rgb_to_y
from src.core.embed import embed_watermark
from src.core.detect import extract_watermark

def test_minimal_embedding():
    target_dir = "dataset/resized"
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    images = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png'))]
    if len(images) > 100:
        images = images[:100]
        
    if not images:
        print("No images found for validation.")
        return
        
    # 64-bit dummy watermark
    np.random.seed(42)
    dummy_watermark = np.random.randint(0, 2, 64).tolist()
    print(f"Dummy watermark (first 10 bits): {dummy_watermark[:10]}")
    
    alpha = 20
    recovery_rates = []
    
    print(f"Validating Phase D: Immediate detection over {len(images)} images...")
    
    for img_name in tqdm(images):
        img_path = os.path.join(target_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 1. RGB -> Y
        Y = rgb_to_y(img)
        
        # 2. Pad
        Y_padded = pad_to_8(Y)
        h, w = Y.shape
        
        # 3. Block DCT
        dct_img = apply_block_dct(Y_padded)
        
        # 4. Embed Watermark
        watermarked_dct = embed_watermark(dct_img, dummy_watermark, alpha=alpha)
        
        # 5. Inverse DCT
        watermarked_padded = apply_block_idct(watermarked_dct)
        
        # 6. Unpad and reconstruct (convert to uint8)
        watermarked_img = watermarked_padded[:h, :w]
        watermarked_uint8 = np.clip(np.round(watermarked_img), 0, 255).astype(np.uint8)
        
        ## DETECTION PHASE
        # We need to simulate the receiver side exactly by doing RGB->Y on reconstructed, but here we just have Y.
        # 1. Pad again (assuming sender and receiver know original size, or just pad what we got)
        # Note: watermarked_uint8 is HxW.
        Y_det_padded = pad_to_8(watermarked_uint8).astype(np.float32)
        
        # 2. Block DCT
        dct_det = apply_block_dct(Y_det_padded)
        
        # 3. Extract Watermark
        extracted_bits = extract_watermark(dct_det, num_bits=64, alpha=alpha)
        
        # Compare
        correct = sum([1 for a, b in zip(dummy_watermark, extracted_bits) if a == b])
        accuracy = correct / len(dummy_watermark)
        recovery_rates.append(accuracy)
        
        if accuracy < 1.0:
            print(f"Failed 100% recovery for {img_name}: {accuracy*100}%")
            
    avg_accuracy = np.mean(recovery_rates)
    print(f"\nAverage Bit Recovery Rate: {avg_accuracy*100:.2f}%")
    
    if avg_accuracy == 1.0:
        print("✅ Phase D Validation Passed: 100% immediate bit recovery achieved.")
    else:
        print("❌ Phase D Validation Failed: 100% immediate recovery is required.")

if __name__ == "__main__":
    test_minimal_embedding()
