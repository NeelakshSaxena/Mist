import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.dct import dct2, idct2

def debug_qim():
    np.random.seed(42)
    # Generate random 8x8 block (0-255)
    block = np.random.randint(0, 256, (8, 8), dtype=np.uint8).astype(np.float32)
    dct_block = dct2(block)
    
    alpha = 50
    val = dct_block[3, 2]
    bit = 1
    
    # Standard QIM
    # Quantize to closest multiple of alpha
    q = round(val / alpha)
    
    # If the parity of the quantized value doesn't match the bit, shift it
    if q % 2 != bit:
        if val > q * alpha:
            q += 1
        else:
            q -= 1
            
    # Set the new value
    embedded_val = q * alpha
    dct_block[3, 2] = embedded_val
    print(f"Original val: {val:.2f}, Embedded val: {embedded_val} (q={q})")
    
    # IDCT
    reconstructed_block = idct2(dct_block)
    
    # Simulate saving to image (rounding & clipping to uint8)
    reconstructed_uint8 = np.clip(np.round(reconstructed_block), 0, 255).astype(np.uint8)
    
    # Receiver Side DCT 
    receiver_dct = dct2(reconstructed_uint8.astype(np.float32))
    val_received = receiver_dct[3, 2]
    
    print(f"Received val: {val_received:.2f}")
    
    # Extraction
    q_rx = round(val_received / alpha)
    bit_rx = int(q_rx % 2)
    
    print(f"Extracted bit: {bit_rx}, Expected bit: {bit}")

if __name__ == "__main__":
    for _ in range(5):
        debug_qim()
