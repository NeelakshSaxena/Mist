"""
src/core/embed.py  –  DCT Watermark Embedding

Phase 2 geometry update
-----------------------
Payload after ECC = 1104 bits (88 data bytes + 50 parity bytes).
Each 8x8 DCT block carries 5 bits (FREQS).
Tile needs >= ceil(1104/5) = 221 blocks.

Chosen tile: 16 rows x 14 cols = 224 blocks x 5 bits = 1120 bit capacity.
The 1104 ECC bits are embedded; the remaining 16 positions are padded to 0.

Tile pixel size: 16x8 = 128 px tall, 14x8 = 112 px wide.
A 512x512 image can fit 4x4 = 16 full tiles -> strong majority-vote redundancy.
"""

import numpy as np
from src.core.ecc import encode_payload, ECC_TOTAL_BITS

FREQS = [(2, 1), (2, 2), (3, 1), (3, 2), (4, 1)]

# Tile geometry
TILE_B_ROWS   = 16   # blocks per tile (vertical)
TILE_B_COLS   = 15   # blocks per tile (horizontal)
TILE_CAPACITY = TILE_B_ROWS * TILE_B_COLS * len(FREQS)   # 1200 bits

# Alias kept for detect.py import
ECC_BITS = ECC_TOTAL_BITS  # 1184


def get_tile_mapping(seed: int = 42) -> list:
    """
    Return a deterministic, shuffled list of (block_row, block_col, freq_idx)
    tuples that maps bit-position → spatial location within a tile.

    The shuffle prevents simple frequency-domain pattern analysis.
    Seed is the *embedding* secret — kept separate from the signing key.
    """
    np.random.seed(seed)
    mapping = [
        (br, bc, f_idx)
        for br in range(TILE_B_ROWS)
        for bc in range(TILE_B_COLS)
        for f_idx in range(len(FREQS))
    ]
    np.random.shuffle(mapping)
    return mapping


TILE_MAPPING = get_tile_mapping()


# ─────────────────────────────────────────────────────────
#  Low-level bit embedding
# ─────────────────────────────────────────────────────────

def embed_bit(block: np.ndarray, bit: int, u: int, v: int, alpha: float = 20.0) -> np.ndarray:
    """
    Embed a single bit into an 8×8 DCT block using Quantization Index Modulation.
    Modifies coefficient (u, v) in-place (on a copy).
    """
    modified = block.copy()
    val = modified[u, v]
    q   = np.floor(val / alpha)

    if int(q) % 2 != bit:
        if (val - q * alpha) >= (alpha / 2):
            q += 1
        else:
            q -= 1

    modified[u, v] = q * alpha + (alpha / 2.0)
    return modified


# ─────────────────────────────────────────────────────────
#  Main embedding function
# ─────────────────────────────────────────────────────────

def embed_watermark(dct_img: np.ndarray, payload_bits: list, alpha: float = 20.0) -> np.ndarray:
    """
    Embed a payload into the DCT image using 2D tiling and ECC.

    Parameters
    ----------
    dct_img      : np.ndarray  Float32 DCT-domain image (Y channel, block-DCT applied).
    payload_bits : list[int]   704 bits — payload_core (192) + signature (512).
    alpha        : float       QIM quantisation step (strength).

    Returns
    -------
    np.ndarray  Modified DCT image with watermark embedded across all full tiles.
    """
    # ECC encode: 704 bits → 1024 bits
    ecc_bits = encode_payload(payload_bits)

    # Pad to TILE_CAPACITY (1050) with zeros
    padded_bits = ecc_bits + [0] * (TILE_CAPACITY - len(ecc_bits))

    h, w = dct_img.shape
    embedded = dct_img.copy()

    tile_h_px = TILE_B_ROWS * 8   # 128 px
    tile_w_px = TILE_B_COLS * 8   # 112 px

    for start_r in range(0, h, tile_h_px):
        for start_c in range(0, w, tile_w_px):
            for bit_idx, bit_val in enumerate(padded_bits):
                br, bc, f_idx = TILE_MAPPING[bit_idx]
                u, v = FREQS[f_idx]

                r = start_r + br * 8
                c = start_c + bc * 8

                # Skip partial edge tiles
                if r + 8 <= h and c + 8 <= w:
                    block = embedded[r:r+8, c:c+8]
                    embedded[r:r+8, c:c+8] = embed_bit(block, bit_val, u, v, alpha)

    return embedded
