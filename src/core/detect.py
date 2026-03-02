"""
src/core/detect.py  –  DCT Watermark Detection + Signature Verification

Phase 2 detection pipeline
---------------------------
1. Grid-search for optimal 8×8 pixel alignment (64 shifts)
2. Extract bit matrix from best-aligned DCT blocks
3. Grid-search for optimal tile boundary (14×15 blocks)
4. Majority-vote across all matching tiles
5. ECC RS-decode 1024 bits → 704 bits
6. Split into payload_core (192 bits / 24 bytes) + signature (512 bits / 64 bytes)
7. Verify: SHA-256(payload_core_bytes) → Ed25519.verify(signature, public_key)
8. If verified, unpack payload_core_bytes into a structured dict

Public surface
--------------
  detect_watermark(y_img, public_key_bytes, alpha=20)
    → dict | None

  extract_watermark_spatial(y_img, alpha=20)   [legacy, returns raw bits]
"""

import numpy as np
from src.core.ecc import decode_payload
from src.core.embed import (
    FREQS, ECC_BITS, TILE_B_ROWS, TILE_B_COLS,
    TILE_CAPACITY, TILE_MAPPING,
)
from src.core.payload import parse_embed_payload, unpack, EMBED_PAYLOAD_BITS
from src.core.crypto import verify


# ─────────────────────────────────────────────────────────
#  Low-level bit extraction
# ─────────────────────────────────────────────────────────

def extract_bit(block: np.ndarray, u: int, v: int, alpha: float = 20.0) -> int:
    val = block[u, v]
    q   = np.floor(val / alpha)
    return int(q % 2)


# ─────────────────────────────────────────────────────────
#  Direct DCT extraction (given perfectly grid-aligned DCT)
# ─────────────────────────────────────────────────────────

def _extract_from_dct(dct_img: np.ndarray, alpha: float = 20.0) -> tuple[list, bool]:
    """
    Extract payload from a grid-aligned DCT image using tiled majority vote.
    Returns (payload_bits: list[int], ecc_success: bool).
    payload_bits is 704 bits on success or best-effort.
    """
    h, w = dct_img.shape

    tile_h_px = TILE_B_ROWS * 8
    tile_w_px = TILE_B_COLS * 8

    votes = [[] for _ in range(TILE_CAPACITY)]

    for start_r in range(0, h - tile_h_px + 1, tile_h_px):
        for start_c in range(0, w - tile_w_px + 1, tile_w_px):
            for bit_idx in range(TILE_CAPACITY):
                br, bc, f_idx = TILE_MAPPING[bit_idx]
                u, v = FREQS[f_idx]
                r = start_r + br * 8
                c = start_c + bc * 8
                block = dct_img[r:r+8, c:c+8]
                votes[bit_idx].append(extract_bit(block, u, v, alpha))

    voted_bits = [
        (1 if sum(v) > len(v) / 2 else 0) if v else 0
        for v in votes
    ]

    voted_ecc_bits = voted_bits[:ECC_BITS]
    return decode_payload(voted_ecc_bits)


# ─────────────────────────────────────────────────────────
#  Spatial detection (handles un-aligned images / attacks)
# ─────────────────────────────────────────────────────────

def _best_aligned_dct(y_img: np.ndarray, alpha: float = 20.0) -> np.ndarray | None:
    """
    Grid-search over all 64 (shift_y, shift_x) pixel offsets to find the
    8×8 DCT grid alignment that maximises QIM confidence.

    Returns the best-aligned full DCT image array, or None.
    """
    from src.core.dct import pad_to_8, apply_block_dct

    img_h, img_w = y_img.shape
    best_conf = -1.0
    best_dct  = None

    for shift_y in range(8):
        for shift_x in range(8):
            if shift_y >= img_h or shift_x >= img_w:
                continue

            cropped = y_img[shift_y:, shift_x:]
            padded  = pad_to_8(cropped).astype(np.float32)
            dct_img = apply_block_dct(padded)

            h_d, w_d = dct_img.shape
            bh, bw   = h_d // 8, w_d // 8

            dct_blocks = dct_img.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)

            conf_sum    = 0.0
            block_count = 0
            for u, v in FREQS:
                vals       = dct_blocks[:, :, u, v]
                fractions  = (vals / alpha) - np.floor(vals / alpha)
                d_center   = np.abs(fractions - 0.5)
                conf_sum  += np.sum(0.5 - d_center)
                block_count += vals.size

            avg_conf = conf_sum / (block_count + 1e-6)
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_dct  = dct_img

    return best_dct


def _extract_spatial_bits(best_dct: np.ndarray, alpha: float = 20.0) -> tuple[list, bool]:
    """
    Given a best-aligned DCT image, search over tile boundary offsets
    (0..TILE_B_ROWS-1, 0..TILE_B_COLS-1) and return the decoded payload bits
    from the highest-confidence tile assignment.
    """
    h, w = best_dct.shape
    bh   = h // 8
    bw   = w // 8

    # Build full bit matrix for all blocks
    bit_matrix = np.zeros((bh, bw, len(FREQS)), dtype=np.uint8)
    dct_blocks = best_dct.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    for f_idx, (u, v) in enumerate(FREQS):
        q = np.floor(dct_blocks[:, :, u, v] / alpha)
        bit_matrix[:, :, f_idx] = (q % 2).astype(np.uint8)

    best_payload_bits = None
    best_ecc_success  = False
    best_tile_conf    = -1.0

    for t_shift_r in range(TILE_B_ROWS):
        for t_shift_c in range(TILE_B_COLS):
            tiles_r = (bh - t_shift_r) // TILE_B_ROWS
            tiles_c = (bw - t_shift_c) // TILE_B_COLS

            if tiles_r == 0 or tiles_c == 0:
                continue

            votes = [[] for _ in range(TILE_CAPACITY)]

            for tr in range(tiles_r):
                for tc in range(tiles_c):
                    start_br = t_shift_r + tr * TILE_B_ROWS
                    start_bc = t_shift_c + tc * TILE_B_COLS
                    for bit_idx in range(TILE_CAPACITY):
                        tile_br, tile_bc, f_idx = TILE_MAPPING[bit_idx]
                        votes[bit_idx].append(
                            int(bit_matrix[start_br + tile_br, start_bc + tile_bc, f_idx])
                        )

            # Confidence: unanimity of tile votes around each bit
            conf      = 0.0
            valid_v   = 0
            for v in votes:
                if v:
                    conf    += abs(sum(v) / len(v) - 0.5)
                    valid_v += 1
            norm_conf = conf / (valid_v + 1e-6)

            if norm_conf > best_tile_conf:
                best_tile_conf = norm_conf

                voted_bits   = [
                    (1 if sum(v) > len(v) / 2 else 0) if v else 0
                    for v in votes
                ]
                ecc_bits     = voted_bits[:ECC_BITS]
                payload_bits, success = decode_payload(ecc_bits)

                if success:
                    # Early exit on clean decode
                    return payload_bits, True

                best_payload_bits = payload_bits
                best_ecc_success  = success

    return (best_payload_bits, best_ecc_success) if best_payload_bits else ([], False)


# ─────────────────────────────────────────────────────────
#  High-level API
# ─────────────────────────────────────────────────────────

def detect_watermark(y_img: np.ndarray, public_key_bytes: bytes, alpha: float = 20.0) -> dict | None:
    """
    Fully authenticated watermark detection.

    Pipeline:
        align → extract → ECC-decode → split payload_core / signature
        → SHA-256(payload_core) → Ed25519.verify(signature, public_key)
        → unpack fields if signature is valid

    Parameters
    ----------
    y_img           : np.ndarray  Spatial-domain Y (luminance) channel (uint8 or float).
    public_key_bytes: bytes        32-byte Ed25519 public key for verification.
    alpha           : float        QIM quantisation step (must match embedder).

    Returns
    -------
    dict  {'user_id', 'image_id', 'timestamp', 'model_version', 'reserved',
           'signature_valid': True, 'ecc_success': bool}
          if watermark is found **and** signature verifies.

    None  if no valid authenticated watermark is found.
    """
    best_dct = _best_aligned_dct(y_img, alpha)
    if best_dct is None:
        return None

    payload_bits, ecc_success = _extract_spatial_bits(best_dct, alpha)
    if not payload_bits or len(payload_bits) < EMBED_PAYLOAD_BITS:
        return None

    try:
        payload_core_bytes, signature_bytes = parse_embed_payload(payload_bits)
    except ValueError:
        return None

    if not verify(public_key_bytes, payload_core_bytes, signature_bytes):
        # Signature check failed — tampered, forged, or wrong key
        return None

    fields = unpack(payload_core_bytes)
    fields["signature_valid"] = True
    fields["ecc_success"]     = ecc_success
    return fields


def extract_watermark_spatial(y_img: np.ndarray, alpha: float = 20.0) -> list | None:
    """
    Legacy / low-level extractor. Returns raw 704 payload bits without
    signature verification. Use detect_watermark() for authenticated detection.
    """
    best_dct = _best_aligned_dct(y_img, alpha)
    if best_dct is None:
        return None

    payload_bits, _ = _extract_spatial_bits(best_dct, alpha)
    return payload_bits[:EMBED_PAYLOAD_BITS] if payload_bits else None
