import numpy as np
from src.core.ecc import decode_payload
from src.core.embed import FREQS, ECC_BITS, TILE_B_ROWS, TILE_B_COLS, TILE_CAPACITY, TILE_MAPPING

def extract_bit(block, u, v, alpha=20):
    val = block[u, v]
    q = np.floor(val / alpha)
    return int(q % 2)

def _extract_from_dct(dct_img, num_bits=64, alpha=20):
    """
    Extract payload from a perfectly grid-aligned DCT image containing one or more tiles.
    """
    h, w = dct_img.shape
    
    tile_h_px = TILE_B_ROWS * 8
    tile_w_px = TILE_B_COLS * 8
    
    # We collect votes for each bit position in the tile across all full tiles
    votes = [[] for _ in range(TILE_CAPACITY)]
    
    # Iterate over all possible full tiles
    for start_r in range(0, h - tile_h_px + 1, tile_h_px):
        for start_c in range(0, w - tile_w_px + 1, tile_w_px):
            
            # Extract bits from this specific tile
            for bit_idx in range(TILE_CAPACITY):
                br, bc, f_idx = TILE_MAPPING[bit_idx]
                u, v = FREQS[f_idx]
                
                r = start_r + br * 8
                c = start_c + bc * 8
                
                block = dct_img[r:r+8, c:c+8]
                bit = extract_bit(block, u, v, alpha)
                votes[bit_idx].append(bit)
                
    # Majority vote across surviving tiles
    voted_bits = []
    for v in votes:
        if not v:
            voted_bits.append(0)
        else:
            voted_bits.append(1 if sum(v) > len(v)/2 else 0)
            
    # The ECC payload is just the first ECC_BITS (the rest was zero padding)
    voted_ecc_bits = voted_bits[:ECC_BITS]
    
    payload, success = decode_payload(voted_ecc_bits)
    return payload[:num_bits], success

def extract_watermark(dct_img, num_bits=64, alpha=20):
    """Legacy wrapper for direct DCT extraction."""
    payload, _ = _extract_from_dct(dct_img, num_bits, alpha)
    return payload

def extract_watermark_spatial(y_img, num_bits=64, alpha=20):
    """
    Extract payload from a spatial Y-channel image.
    Highly optimized:
    1. Grid search for proper 8x8 spatial pixel DCT alignment (64 checks)
    2. Extract the best grid into a block matrix.
    3. Grid search for the 6x5 Tile Boundary over the blocks (30 checks).
    """
    from src.core.dct import pad_to_8, apply_block_dct
    
    img_h, img_w = y_img.shape
    best_pixel_conf = -1
    best_dct = None
    
    # --- STEP 1: Find optimal 8x8 Pixel Grid Alignment ---
    for shift_y in range(8):
        for shift_x in range(8):
            if shift_y >= img_h or shift_x >= img_w:
                continue
                
            cropped_y = y_img[shift_y:, shift_x:]
            padded_y = pad_to_8(cropped_y).astype(np.float32)
            dct_img = apply_block_dct(padded_y)
            
            h, w = dct_img.shape
            blocks_h, blocks_w = h // 8, w // 8
            
            # Vectorized quality check
            conf_sum = 0
            block_count = 0
            
            # Sub-sample the DCT coefficients at our FREQS targets across all blocks
            # We reshape to (blocks_h, 8, blocks_w, 8) => swap to (blocks_h, blocks_w, 8, 8)
            dct_blocks = dct_img.reshape(blocks_h, 8, blocks_w, 8).transpose(0, 2, 1, 3)
            
            for u, v in FREQS:
                vals = dct_blocks[:, :, u, v]
                fractions = (vals / alpha) - np.floor(vals / alpha)
                d_to_center = np.abs(fractions - 0.5)
                # Low d_to_center means perfectly aligned
                conf_sum += np.sum(0.5 - d_to_center)
                block_count += vals.size
                
            avg_conf = conf_sum / (block_count + 1e-6)
            
            if avg_conf > best_pixel_conf:
                best_pixel_conf = avg_conf
                best_dct = dct_img
                
    # --- STEP 2: Extract bits from the Best Grid ---
    if best_dct is None:
        return None
        
    h, w = best_dct.shape
    blocks_h = h // 8
    blocks_w = w // 8
    
    bit_matrix = np.zeros((blocks_h, blocks_w, len(FREQS)), dtype=np.uint8)
    best_dct_blocks = best_dct.reshape(blocks_h, 8, blocks_w, 8).transpose(0, 2, 1, 3)
    
    for f_idx, (u, v) in enumerate(FREQS):
        vals = best_dct_blocks[:, :, u, v]
        q = np.floor(vals / alpha)
        bit_matrix[:, :, f_idx] = (q % 2).astype(np.uint8)
                
    # --- STEP 3: Find Optimal Tile Boundary ---
    best_payload = None
    best_tile_conf = -1
    
    # We search the possible block offsets: (0..5, 0..4)
    for t_shift_r in range(TILE_B_ROWS):
        for t_shift_c in range(TILE_B_COLS):
            
            votes = [[] for _ in range(TILE_CAPACITY)]
            
            # Count how many FULL tiles fit if we start at this offset
            tiles_r = (blocks_h - t_shift_r) // TILE_B_ROWS
            tiles_c = (blocks_w - t_shift_c) // TILE_B_COLS
            
            if tiles_r == 0 or tiles_c == 0:
                continue
                
            for tr in range(tiles_r):
                for tc in range(tiles_c):
                    # Absolute block start for this tile
                    start_br = t_shift_r + tr * TILE_B_ROWS
                    start_bc = t_shift_c + tc * TILE_B_COLS
                    
                    # Extract this specific tile
                    for bit_idx in range(TILE_CAPACITY):
                        tile_br, tile_bc, f_idx = TILE_MAPPING[bit_idx]
                        bit = bit_matrix[start_br + tile_br, start_bc + tile_bc, f_idx]
                        votes[bit_idx].append(bit)
                        
            # Calc tile confidence (How unanimously do the tiles agree on the bits?)
            conf = 0
            valid_votes = 0
            for v in votes:
                if len(v) > 0:
                    conf += abs((sum(v) / len(v)) - 0.5)
                    valid_votes += 1
                    
            norm_conf = conf / (valid_votes + 1e-6)
            
            if norm_conf > best_tile_conf:
                best_tile_conf = norm_conf
                
                voted_bits = []
                for v in votes:
                    if not v:
                        voted_bits.append(0)
                    else:
                        voted_bits.append(1 if sum(v) > len(v)/2 else 0)
                        
                voted_ecc_bits = voted_bits[:ECC_BITS]
                payload, success = decode_payload(voted_ecc_bits)
                if success:
                    return payload[:num_bits]
                else:
                    best_payload = payload
                    
    return best_payload[:num_bits] if best_payload else None
