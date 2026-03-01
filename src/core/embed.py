import numpy as np
from src.core.ecc import encode_payload

FREQS = [(2,1), (2,2), (3,1), (3,2), (4,1)]
ECC_BITS = 144 # 18 bytes (8 byte payload + 10 byte RS ECC)

# Tile Geometry
# We have 144 bits to embed. Each 8x8 block can hold 5 bits (FREQS).
# So we need ceil(144 / 5) = 29 blocks per tile.
# Let's make a clean rectangular tile: 6x5 blocks = 30 blocks.
# 30 blocks * 5 bits/block = 150 bit capacity per tile. 
# We'll embed the 144 bits, and pad the remaining 6 bits with 0s.
TILE_B_ROWS = 6  # 6 blocks vertically
TILE_B_COLS = 5  # 5 blocks horizontally
TILE_CAPACITY = TILE_B_ROWS * TILE_B_COLS * len(FREQS)

def get_tile_mapping(seed=42):
    """
    Returns a deterministic randomly shuffled mapping of bit indices
    to spatial tile coordinates and frequencies.
    Prevents simple pattern analysis attacks.
    """
    np.random.seed(seed)
    mapping = []
    for br in range(TILE_B_ROWS):
        for bc in range(TILE_B_COLS):
            for f_idx in range(len(FREQS)):
                mapping.append((br, bc, f_idx))
    
    # Shuffle the placement
    np.random.shuffle(mapping)
    return mapping

TILE_MAPPING = get_tile_mapping()

def embed_bit(block, bit, u, v, alpha=20):
    """
    Embeds a single bit into an 8x8 DCT block using Quantization Index Modulation (QIM).
    Modifies the coefficient at index (u,v).
    """
    modified_block = block.copy()
    val = modified_block[u, v]
    
    q = np.floor(val / alpha)
    
    if int(q) % 2 != bit:
        if (val - q * alpha) >= (alpha / 2):
            q += 1
        else:
            q -= 1
            
    modified_block[u, v] = q * alpha + (alpha / 2.0)
    return modified_block

def embed_watermark(dct_img, payload_bits, alpha=20):
    """
    Embed a payload into the DCT image using 2D Tiling and ECC.
    """
    ecc_bits = encode_payload(payload_bits)
    
    # Pad to TILE_CAPACITY
    padded_ecc_bits = ecc_bits + [0] * (TILE_CAPACITY - len(ecc_bits))
    
    h, w = dct_img.shape
    embedded_img = dct_img.copy()
    
    # Iterate over the image in Tile-sized chunks
    tile_h_px = TILE_B_ROWS * 8
    tile_w_px = TILE_B_COLS * 8
    
    for start_r in range(0, h, tile_h_px):
        for start_c in range(0, w, tile_w_px):
            
            # Embed the full padded ECC payload into this tile
            for bit_idx, bit_val in enumerate(padded_ecc_bits):
                br, bc, f_idx = TILE_MAPPING[bit_idx]
                u, v = FREQS[f_idx]
                
                # Absolute pixel coordinates of the block
                r = start_r + br * 8
                c = start_c + bc * 8
                
                # Check if block fits in image (partial tiles at the very edge are just skipped/truncated)
                if r + 8 <= h and c + 8 <= w:
                    block = embedded_img[r:r+8, c:c+8]
                    embedded_img[r:r+8, c:c+8] = embed_bit(block, bit_val, u, v, alpha)
                    
    return embedded_img
