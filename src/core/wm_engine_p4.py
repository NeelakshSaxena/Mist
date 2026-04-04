"""
src/core/wm_engine_p4.py  –  Phase 4 Spatial Attack Resistance Engine

Builds on Phase 3's multi-scale DCT embedding to add spatial redundancy.
The watermark becomes a distributed signal: even a partial image fragment
can be used to detect and verify ownership.

Architecture
------------
Embedding:
    1. Outer RS encode: split 148-byte inner codeword into N shards (5 bytes each)
    2. Construct per-tile bitstreams: [8-bit anchor | 8-bit index | 40-bit shard | 8-bit CRC]
    3. Assemble full-image bitstream and embed via Phase 3 DCT difference modulation
    4. Add Phase 3 sinusoidal harmonic for presence detection

Detection:
    1. Phase 3 presence detection (multi-scale + harmonic)
    2. Spatial sync: two-stage search for macro-tile grid alignment
    3. Per-tile shard extraction with CRC validation
    4. Outer RS erasure decode → inner RS decode → payload parse → signature verify

Public API
----------
    embed_p4(image, bitstream, key)       → np.ndarray
    detect_p4(image_fragment, key)        → dict
    extract_shards_p4(image, key)         → dict

Constants
---------
    MT_SIZE          : Macro-tile size in pixels (64)
    SHARD_BYTES      : Payload bytes per tile (5)
    K_SHARDS         : Minimum shards for reconstruction (30)
"""

import hashlib
import hmac
import math
import struct

import cv2
import numpy as np
import reedsolo
from scipy.fft import dctn, idctn

from src.core.wm_engine_p3 import (
    _to_ycbcr, _from_ycbcr, _pad_to_n,
    _embed_one_scale, _build_harmonic_map,
    _block_dct, _block_idct,
    _block_seed, _select_pair, _hmac_bytes,
    _score_one_scale,
    detect_p3,
    TILE_P3, PAIR_POOL,
    BASE_DELTA_P3, BETA_P3, VAR_NORM_P3,
)
from src.core.ecc import ECC_TOTAL_BYTES, ECC_TOTAL_BITS

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

MT_SIZE: int       = 64    # Macro-tile size in pixels
MT_BLOCKS: int     = MT_SIZE // 8   # 8 — blocks per macro-tile side
BLOCKS_PER_MT: int = MT_BLOCKS ** 2  # 64 — total 8×8 blocks per macro-tile

# Per-tile bitstream layout
ANCHOR_BITS: int     = 8
INDEX_BITS: int      = 8
CRC_BITS: int        = 8
SHARD_DATA_BITS: int = BLOCKS_PER_MT - ANCHOR_BITS - INDEX_BITS - CRC_BITS  # 40
SHARD_BYTES: int     = SHARD_DATA_BITS // 8  # 5

# Outer RS parameters
K_SHARDS: int          = math.ceil(ECC_TOTAL_BYTES / SHARD_BYTES)  # ceil(148/5) = 30
PADDED_DATA_BYTES: int = K_SHARDS * SHARD_BYTES                    # 150

# Detection thresholds
ANCHOR_MATCH_MIN: float   = 0.625   # 5 of 8 bits
GRID_VALID_MIN: float     = 0.25    # 25% of expected anchors must match
PRESENCE_CUTOFF: float    = 0.50    # Phase 3 confidence gate


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _derive_p4_key(key: bytes) -> bytes:
    """Derive a Phase-4-specific embedding key (avoids pair collision with P3)."""
    return hmac.new(key, b"mist-phase4-v1", hashlib.sha256).digest()


def _tile_anchor_bits(key: bytes) -> list[int]:
    """Derive the 8-bit sync anchor from the embedding key."""
    p4k = _derive_p4_key(key)
    d = hmac.new(p4k, b"tile-anchor", hashlib.sha256).digest()
    return [(d[0] >> (7 - i)) & 1 for i in range(8)]


def _crc8(data: bytes) -> int:
    """CRC-8/MAXIM (polynomial 0x31)."""
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc


def _int_to_bits(val: int, n: int) -> list[int]:
    """Convert integer to n-bit MSB-first bit list."""
    return [(val >> (n - 1 - i)) & 1 for i in range(n)]


def _bits_from_int(bits: list[int]) -> int:
    """Convert MSB-first bit list to integer."""
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def _bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        bv = 0
        for j in range(8):
            if i + j < len(bits):
                bv |= (bits[i + j] << (7 - j))
        out.append(bv)
    return bytes(out)


def _bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


# ─────────────────────────────────────────────────────────────────────────────
#  Outer Reed-Solomon (column-interleaved)
# ─────────────────────────────────────────────────────────────────────────────

def _outer_rs_encode(inner_codeword: bytes, n_tiles: int) -> list[bytes]:
    """
    Distribute 148-byte inner codeword across n_tiles via column-interleaved RS.

    The 148 bytes are padded to 150 and arranged as a 30×5 matrix.
    Each of the 5 columns is independently RS(N, 30)-encoded.
    Any 30 tiles suffice to reconstruct all columns.

    Returns list of n_tiles shards, each SHARD_BYTES (5) bytes.
    """
    padded = inner_codeword + b'\x00' * (PADDED_DATA_BYTES - len(inner_codeword))

    # Data matrix: K_SHARDS rows × SHARD_BYTES cols
    data_matrix = [padded[i * SHARD_BYTES:(i + 1) * SHARD_BYTES]
                   for i in range(K_SHARDS)]

    n_rs = min(n_tiles, 255)
    nsym = n_rs - K_SHARDS

    if nsym <= 0:
        # Tiny image: simple repetition (each tile gets shard i % K_SHARDS)
        return [data_matrix[i % K_SHARDS] for i in range(n_tiles)]

    rs_outer = reedsolo.RSCodec(nsym)

    # Column-interleaved encode
    shards = [bytearray(SHARD_BYTES) for _ in range(n_rs)]
    for col in range(SHARD_BYTES):
        data_col = bytes(data_matrix[row][col] for row in range(K_SHARDS))
        encoded_col = rs_outer.encode(data_col)
        for idx in range(n_rs):
            shards[idx][col] = encoded_col[idx]

    # Tiles beyond n_rs get cyclic copies
    return [bytes(shards[i % n_rs]) for i in range(n_tiles)]


def _outer_rs_decode(shard_map: dict, n_rs: int) -> bytes | None:
    """
    Reconstruct 148-byte inner codeword from available shards.

    shard_map : dict[int, bytes]  — shard_index → shard data (5 bytes)
    n_rs      : int               — RS codeword length (min(n_tiles, 255))

    Returns 148 bytes or None on failure.
    """
    nsym = n_rs - K_SHARDS
    if nsym <= 0:
        # Repetition mode: assemble directly
        result = bytearray(PADDED_DATA_BYTES)
        for row in range(K_SHARDS):
            if row in shard_map:
                for col in range(SHARD_BYTES):
                    result[row * SHARD_BYTES + col] = shard_map[row][col]
            else:
                return None
        return bytes(result[:ECC_TOTAL_BYTES])

    rs_outer = reedsolo.RSCodec(nsym)
    decoded_cols: list[bytes] = []

    for col in range(SHARD_BYTES):
        received = bytearray(n_rs)
        erase_pos = []
        for idx in range(n_rs):
            if idx in shard_map:
                received[idx] = shard_map[idx][col]
            else:
                erase_pos.append(idx)

        if len(erase_pos) > nsym:
            return None

        try:
            decoded, _, _ = rs_outer.decode(received, erase_pos=erase_pos)
            decoded_cols.append(bytes(decoded))
        except reedsolo.ReedSolomonError:
            return None

    # Reassemble: row r, col c → decoded_cols[c][r]
    result = bytearray(PADDED_DATA_BYTES)
    for row in range(K_SHARDS):
        for col in range(SHARD_BYTES):
            result[row * SHARD_BYTES + col] = decoded_cols[col][row]

    return bytes(result[:ECC_TOTAL_BYTES])


def _outer_rs_decode_smart(
    shard_map: dict, crc_ok: set, n_rs: int
) -> bytes | None:
    """
    Smart outer RS decode with multiple fallback strategies.

    Strategy 1: Use only CRC-verified shards; everything else is erasure.
    Strategy 2: Use all shards as data; non-CRC shards are potential errors.
    Strategy 3: Mixed — CRC-ok as data, non-CRC as erasures.
    """
    nsym = n_rs - K_SHARDS
    if nsym <= 0:
        return _outer_rs_decode(shard_map, n_rs)

    rs_outer = reedsolo.RSCodec(nsym)

    # Build the strategies
    strategies = []

    # Strategy 1: CRC-only (all non-CRC = erasure)
    strategies.append("crc_only")
    # Strategy 2: All data, no erasures (RS handles errors)
    strategies.append("all_data")
    # Strategy 3: Non-CRC as erasures
    strategies.append("non_crc_erasure")

    for strategy in strategies:
        decoded_cols: list[bytes] = []
        ok = True

        for col in range(SHARD_BYTES):
            received = bytearray(n_rs)
            erase_pos = []

            for idx in range(n_rs):
                if strategy == "crc_only":
                    if idx in crc_ok:
                        received[idx] = shard_map[idx][col]
                    else:
                        erase_pos.append(idx)
                elif strategy == "all_data":
                    if idx in shard_map:
                        received[idx] = shard_map[idx][col]
                    else:
                        erase_pos.append(idx)
                else:  # non_crc_erasure
                    if idx in shard_map:
                        received[idx] = shard_map[idx][col]
                        if idx not in crc_ok:
                            erase_pos.append(idx)
                    else:
                        erase_pos.append(idx)

            if len(erase_pos) > nsym:
                ok = False
                break

            try:
                decoded, _, _ = rs_outer.decode(received, erase_pos=erase_pos)
                decoded_cols.append(bytes(decoded))
            except reedsolo.ReedSolomonError:
                ok = False
                break

        if ok and len(decoded_cols) == SHARD_BYTES:
            result = bytearray(PADDED_DATA_BYTES)
            for row in range(K_SHARDS):
                for col in range(SHARD_BYTES):
                    result[row * SHARD_BYTES + col] = decoded_cols[col][row]
            return bytes(result[:ECC_TOTAL_BYTES])

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Tile bitstream construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_tile_bitstream(
    anchor: list[int],
    shard_idx: int,
    shard_data: bytes,
) -> list[int]:
    """
    Build a 64-bit tile bitstream: [anchor(8) | index(8) | shard(40) | crc(8)].
    """
    index_bits = _int_to_bits(shard_idx, INDEX_BITS)
    shard_bits = _bytes_to_bits(shard_data)

    # Pad or truncate shard_bits to SHARD_DATA_BITS
    if len(shard_bits) < SHARD_DATA_BITS:
        shard_bits += [0] * (SHARD_DATA_BITS - len(shard_bits))
    shard_bits = shard_bits[:SHARD_DATA_BITS]

    # CRC over [index_byte | shard_bytes]
    crc_input = bytes([shard_idx]) + shard_data
    crc_val = _crc8(crc_input)
    crc_bits = _int_to_bits(crc_val, CRC_BITS)

    return anchor + index_bits + shard_bits + crc_bits


# ─────────────────────────────────────────────────────────────────────────────
#  Tile-level bit extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tile_bits(tile_Y: np.ndarray, key: bytes) -> list[int]:
    """
    Extract 64 hard-decision bits from a 64×64 luminance region.
    Uses Phase-4-derived key for PRNG pair selection.
    """
    p4k = _derive_p4_key(key)
    bs = 8
    h, w = tile_Y.shape[:2]
    bh, bw = h // bs, w // bs
    n_bits = bh * bw

    if n_bits == 0:
        return []

    # Build PRNG pair table (same structure as Phase 3 but with p4 key)
    tile_p1 = np.zeros((TILE_P3, TILE_P3, 2), dtype=np.int8)
    tile_p2 = np.zeros((TILE_P3, TILE_P3, 2), dtype=np.int8)
    for tr in range(min(TILE_P3, bh)):
        for tc in range(min(TILE_P3, bw)):
            seed = _block_seed(p4k, tr, tc, bs)
            p1, p2 = _select_pair(seed, PAIR_POOL, bs)
            tile_p1[tr, tc] = p1
            tile_p2[tr, tc] = p2

    Y_slice = tile_Y[:bh * bs, :bw * bs].astype(np.float32)
    dct_img = _block_dct(Y_slice, bs)
    dct_blocks = dct_img.reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3)

    br_idx = np.arange(bh, dtype=np.int32)[:, None]
    bc_idx = np.arange(bw, dtype=np.int32)[None, :]
    tr_idx = br_idx % TILE_P3
    tc_idx = bc_idx % TILE_P3

    p1u = tile_p1[tr_idx, tc_idx, 0].astype(np.int64)
    p1v = tile_p1[tr_idx, tc_idx, 1].astype(np.int64)
    p2u = tile_p2[tr_idx, tc_idx, 0].astype(np.int64)
    p2v = tile_p2[tr_idx, tc_idx, 1].astype(np.int64)

    br_f = np.broadcast_to(br_idx, (bh, bw))
    bc_f = np.broadcast_to(bc_idx, (bh, bw))

    c1 = dct_blocks[br_f, bc_f, p1u, p1v]
    c2 = dct_blocks[br_f, bc_f, p2u, p2v]

    return (c1 > c2).astype(np.int8).ravel().tolist()


def _parse_tile_bits(
    raw_bits: list[int],
    expected_anchor: list[int],
) -> dict | None:
    """
    Parse a 64-bit tile bitstream. Returns parsed dict or None if anchor fails.
    """
    if len(raw_bits) < BLOCKS_PER_MT:
        return None

    anchor_recv = raw_bits[:ANCHOR_BITS]
    anchor_match = sum(a == b for a, b in zip(anchor_recv, expected_anchor)) / ANCHOR_BITS
    if anchor_match < ANCHOR_MATCH_MIN:
        return None

    index_recv = raw_bits[ANCHOR_BITS:ANCHOR_BITS + INDEX_BITS]
    shard_recv = raw_bits[ANCHOR_BITS + INDEX_BITS:ANCHOR_BITS + INDEX_BITS + SHARD_DATA_BITS]
    crc_recv   = raw_bits[ANCHOR_BITS + INDEX_BITS + SHARD_DATA_BITS:
                          ANCHOR_BITS + INDEX_BITS + SHARD_DATA_BITS + CRC_BITS]

    shard_idx  = _bits_from_int(index_recv)
    shard_data = _bits_to_bytes(shard_recv)
    crc_received = _bits_from_int(crc_recv)
    crc_expected = _crc8(bytes([shard_idx]) + shard_data)

    return {
        "shard_idx": shard_idx,
        "shard_data": shard_data,
        "crc_valid": crc_expected == crc_received,
        "anchor_match": anchor_match,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Spatial synchronization
# ─────────────────────────────────────────────────────────────────────────────

def _find_grid_alignment(Y: np.ndarray, key: bytes) -> tuple[int, int] | None:
    """
    Two-stage search for micro-tile grid alignment in a (possibly cropped) image.

    Stage 1: pixel-level sub-block offset (0..7 × 0..7) via DCT correlation.
    Stage 2: macro-tile phase (0..7 × 0..7 blocks) via anchor scanning.

    Returns (total_dx, total_dy) pixel offset or None if grid not found.
    """
    p4k = _derive_p4_key(key)
    h, w = Y.shape
    expected_anchor = _tile_anchor_bits(key)

    # ── Stage 1: find sub-block pixel offset ─────────────────────────
    best_px_score = -2.0
    best_px = (0, 0)
    for px_dy in range(min(8, h - MT_SIZE)):
        for px_dx in range(min(8, w - MT_SIZE)):
            Y_shifted = Y[px_dy:, px_dx:]
            if Y_shifted.shape[0] < MT_SIZE or Y_shifted.shape[1] < MT_SIZE:
                continue
            Y_pad = _pad_to_n(Y_shifted, 8)
            score = _score_one_scale(Y_pad, p4k, 8)
            if score > best_px_score:
                best_px_score = score
                best_px = (px_dx, px_dy)

    # ── Stage 2: find macro-tile phase ───────────────────────────────
    px_dx, px_dy = best_px
    Y_aligned = Y[px_dy:, px_dx:]
    ah, aw = Y_aligned.shape

    best_mt_score = -1.0
    best_mt = (0, 0)

    for mt_dy in range(MT_BLOCKS):
        for mt_dx in range(MT_BLOCKS):
            off_y = mt_dy * 8
            off_x = mt_dx * 8
            n_checked = 0
            n_matched = 0

            # Check up to 6 tiles at this phase
            for ty_idx in range(3):
                for tx_idx in range(3):
                    tile_y = off_y + ty_idx * MT_SIZE
                    tile_x = off_x + tx_idx * MT_SIZE
                    if tile_y + MT_SIZE > ah or tile_x + MT_SIZE > aw:
                        continue
                    region = Y_aligned[tile_y:tile_y + MT_SIZE,
                                       tile_x:tile_x + MT_SIZE]
                    bits = _extract_tile_bits(region, key)
                    if len(bits) < ANCHOR_BITS:
                        continue
                    n_checked += 1
                    match = sum(a == b for a, b in zip(bits[:ANCHOR_BITS],
                                                       expected_anchor)) / ANCHOR_BITS
                    if match >= ANCHOR_MATCH_MIN:
                        n_matched += 1

            if n_checked > 0:
                score = n_matched / n_checked
                if score > best_mt_score:
                    best_mt_score = score
                    best_mt = (mt_dx, mt_dy)

    if best_mt_score < GRID_VALID_MIN:
        return None

    total_dx = px_dx + best_mt[0] * 8
    total_dy = px_dy + best_mt[1] * 8
    return (total_dx, total_dy)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed_p4(image: np.ndarray, bitstream: np.ndarray, key: bytes) -> np.ndarray:
    """
    Phase 4 spatial-redundant watermark embedding.

    Parameters
    ----------
    image     : np.ndarray  BGR uint8 (H, W, 3).
    bitstream : np.ndarray  1184-bit ECC-encoded payload (from Phase 2 ecc.encode_payload).
    key       : bytes       Secret embedding key.

    Returns
    -------
    np.ndarray  Watermarked BGR image (uint8, same shape).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("embed_p4() expects a 3-channel BGR image (H, W, 3).")
    if len(bitstream) == 0:
        raise ValueError("bitstream must be non-empty.")

    H, W = image.shape[:2]
    ycrcb, Y_orig = _to_ycbcr(image)

    mt_rows = H // MT_SIZE
    mt_cols = W // MT_SIZE
    n_tiles = mt_rows * mt_cols

    if n_tiles < 4:
        raise ValueError(
            f"Image too small for Phase 4: {H}×{W} yields {n_tiles} tiles "
            f"(minimum 4). Need ≥ {MT_SIZE * 2}×{MT_SIZE * 2} px."
        )

    # ── Outer RS encode ───────────────────────────────────────────────
    inner_codeword = _bits_to_bytes(bitstream.ravel().tolist())
    if len(inner_codeword) < ECC_TOTAL_BYTES:
        inner_codeword += b'\x00' * (ECC_TOTAL_BYTES - len(inner_codeword))
    inner_codeword = inner_codeword[:ECC_TOTAL_BYTES]

    shards = _outer_rs_encode(inner_codeword, n_tiles)
    anchor = _tile_anchor_bits(key)

    # ── Build per-tile bitstreams ─────────────────────────────────────
    tile_bitstreams: list[list[int]] = []
    n_rs = min(n_tiles, 255)
    for t in range(n_tiles):
        shard_idx = t % n_rs
        bs = _build_tile_bitstream(anchor, shard_idx, shards[t])
        tile_bitstreams.append(bs)

    # ── Assemble full-image bit array ─────────────────────────────────
    # Pad Y to multiple of MT_SIZE so all blocks belong to a macro-tile
    pad_h = (MT_SIZE - H % MT_SIZE) % MT_SIZE
    pad_w = (MT_SIZE - W % MT_SIZE) % MT_SIZE
    Y_padded = np.pad(Y_orig, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = Y_padded.shape
    bh, bw = ph // 8, pw // 8
    # Recompute tile grid on padded image
    mt_rows_p = ph // MT_SIZE
    mt_cols_p = pw // MT_SIZE

    bits_2d = np.zeros((bh, bw), dtype=np.int32)
    for t_idx in range(n_tiles):
        mt_r = t_idx // mt_cols
        mt_c = t_idx % mt_cols
        br0 = mt_r * MT_BLOCKS
        bc0 = mt_c * MT_BLOCKS
        tile_2d = np.array(tile_bitstreams[t_idx], dtype=np.int32).reshape(
            MT_BLOCKS, MT_BLOCKS)
        bits_2d[br0:br0 + MT_BLOCKS, bc0:bc0 + MT_BLOCKS] = tile_2d

    # Padded-area tiles (outside original image) get repeated data
    for mt_r in range(mt_rows_p):
        for mt_c in range(mt_cols_p):
            if mt_r < mt_rows and mt_c < mt_cols:
                continue  # already filled
            src_t = (mt_r % max(1, mt_rows)) * mt_cols + (mt_c % max(1, mt_cols))
            src_t = min(src_t, n_tiles - 1)
            br0 = mt_r * MT_BLOCKS
            bc0 = mt_c * MT_BLOCKS
            tile_2d = np.array(tile_bitstreams[src_t], dtype=np.int32).reshape(
                MT_BLOCKS, MT_BLOCKS)
            bits_2d[br0:br0 + MT_BLOCKS, bc0:bc0 + MT_BLOCKS] = tile_2d

    bits_flat = bits_2d.ravel()

    # ── Embed via Phase 3 single-scale engine (with P4-derived key) ───
    p4k = _derive_p4_key(key)
    Y_embedded = _embed_one_scale(Y_padded, bits_flat, p4k, 8)

    # Crop back to original size
    Y_result = Y_embedded[:H, :W].astype(np.float32)

    # ── Add sinusoidal harmonic (original key, Phase 3 compat) ────────
    harmonic = _build_harmonic_map(H, W, key)
    Y_result = Y_result + harmonic

    return _from_ycbcr(ycrcb, Y_result)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — Detection
# ─────────────────────────────────────────────────────────────────────────────

def extract_shards_p4(image: np.ndarray, key: bytes) -> dict:
    """
    Extract all shard data from a (possibly cropped) image.

    Returns dict with:
        aligned          : bool
        offset           : (dx, dy) | None
        tiles_located    : int
        shards_valid     : int
        shard_map        : dict[int, bytes]
        tile_details     : list[dict]
        n_rs             : int
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("extract_shards_p4() expects BGR (H, W, 3).")

    _, Y = _to_ycbcr(image)
    H, W = Y.shape
    expected_anchor = _tile_anchor_bits(key)

    result = {
        "aligned": False, "offset": None,
        "tiles_located": 0, "shards_valid": 0,
        "shard_map": {},           # idx → shard_bytes (best effort, all accepted)
        "shard_crc_ok": set(),     # set of indices with CRC-verified shards
        "tile_details": [],
        "n_rs": 0,
    }

    # ── Grid alignment search ─────────────────────────────────────────
    offset = _find_grid_alignment(Y, key)
    if offset is None:
        return result

    dx, dy = offset
    result["aligned"] = True
    result["offset"] = offset

    Y_aligned = Y[dy:, dx:]
    ah, aw = Y_aligned.shape
    mt_rows = ah // MT_SIZE
    mt_cols = aw // MT_SIZE

    # n_rs is inferred after shard extraction from the max shard index found.
    # Placeholder; will be set after extraction.
    result["n_rs"] = 0

    # ── Extract shards from each tile ─────────────────────────────────
    shard_buffer: dict[int, list[tuple[bytes, float]]] = {}

    for tr in range(mt_rows):
        for tc in range(mt_cols):
            y0 = tr * MT_SIZE
            x0 = tc * MT_SIZE
            region = Y_aligned[y0:y0 + MT_SIZE, x0:x0 + MT_SIZE]
            if region.shape[0] < MT_SIZE or region.shape[1] < MT_SIZE:
                continue

            raw_bits = _extract_tile_bits(region, key)
            parsed = _parse_tile_bits(raw_bits, expected_anchor)
            if parsed is None:
                continue

            result["tiles_located"] += 1
            detail = {
                "grid_pos": (tr, tc),
                "shard_idx": parsed["shard_idx"],
                "crc_valid": parsed["crc_valid"],
                "anchor_match": parsed["anchor_match"],
            }
            result["tile_details"].append(detail)

            idx = parsed["shard_idx"]
            if idx not in shard_buffer:
                shard_buffer[idx] = []
            weight = parsed["anchor_match"] * (2.0 if parsed["crc_valid"] else 1.0)
            shard_buffer[idx].append(
                (parsed["shard_data"], weight, parsed["crc_valid"]))

    # ── Shard deduplication: prefer CRC-valid, then majority vote ─────
    for idx, candidates in shard_buffer.items():
        crc_valid = [(sd, w) for sd, w, cv in candidates if cv]
        if crc_valid:
            best = max(crc_valid, key=lambda x: x[1])
            result["shard_map"][idx] = best[0]
            result["shard_crc_ok"].add(idx)
        elif len(candidates) == 1:
            result["shard_map"][idx] = candidates[0][0]
        else:
            n_bits = SHARD_DATA_BITS
            voted = []
            for bit_pos in range(n_bits):
                ones = sum(
                    1 for (sd, _, _) in candidates
                    if (sd[bit_pos // 8] >> (7 - bit_pos % 8)) & 1
                )
                voted.append(1 if ones > len(candidates) / 2 else 0)
            result["shard_map"][idx] = _bits_to_bytes(voted)

    result["shards_valid"] = len(result["shard_map"])

    if result["shard_map"]:
        max_idx = max(result["shard_map"].keys())
        result["n_rs"] = max(max_idx + 1, K_SHARDS + 1)
    else:
        result["n_rs"] = K_SHARDS + 1

    return result


def detect_p4(image_fragment: np.ndarray, key: bytes) -> dict:
    """
    Phase 4 spatial-redundant watermark detection.

    Works on arbitrary image fragments. Does NOT require the full image.

    Returns
    -------
    dict:
        detected              : bool
        confidence            : float  (composite 0..1)
        presence_score        : float  (Phase 3 multi-scale)
        scale_scores          : dict
        harmonic_score        : float
        tiles_located         : int
        shards_recovered      : int
        shards_needed         : int
        reconstruction_ratio  : float
        inner_codeword        : bytes | None
        error                 : str | None
    """
    result = {
        "detected": False, "confidence": 0.0,
        "presence_score": 0.0, "scale_scores": {}, "harmonic_score": 0.0,
        "tiles_located": 0, "shards_recovered": 0,
        "shards_needed": K_SHARDS, "reconstruction_ratio": 0.0,
        "inner_codeword": None, "error": None,
    }

    if image_fragment.ndim != 3 or image_fragment.shape[2] != 3:
        raise ValueError("detect_p4() expects BGR (H, W, 3).")

    # ── Phase 3 presence detection ────────────────────────────────────
    p3 = detect_p3(image_fragment, key)
    result["presence_score"] = p3["confidence"]
    result["scale_scores"] = p3["scale_scores"]
    result["harmonic_score"] = p3["harmonic_score"]

    if p3["confidence"] < PRESENCE_CUTOFF:
        return result

    result["detected"] = True

    # ── Shard extraction ──────────────────────────────────────────────
    shard_result = extract_shards_p4(image_fragment, key)
    result["tiles_located"] = shard_result["tiles_located"]
    result["shards_recovered"] = shard_result["shards_valid"]
    result["reconstruction_ratio"] = (
        shard_result["shards_valid"] / K_SHARDS if K_SHARDS > 0 else 0.0
    )

    if not shard_result["aligned"]:
        result["error"] = "Grid alignment failed; presence-only detection."
        result["confidence"] = 0.15 * result["presence_score"]
        return result

    if shard_result["shards_valid"] < K_SHARDS:
        result["error"] = (
            f"Insufficient shards: {shard_result['shards_valid']}/{K_SHARDS}. "
            f"Need {K_SHARDS - shard_result['shards_valid']} more."
        )
        # Partial confidence
        ratio = shard_result["shards_valid"] / K_SHARDS
        result["confidence"] = (
            0.15 * result["presence_score"] +
            0.30 * min(1.0, ratio)
        )
        return result

    # ── Outer RS decode ───────────────────────────────────────────────
    shard_map = shard_result["shard_map"]
    crc_ok_set = shard_result["shard_crc_ok"]
    max_idx = max(shard_map.keys()) if shard_map else 0
    inner = None

    candidates = sorted(set([
        max(max_idx + 1, K_SHARDS + 1),
        64, 128, 255,
        shard_result["n_rs"],
    ]))
    for n_rs in candidates:
        if n_rs < max_idx + 1 or n_rs < K_SHARDS + 1 or n_rs > 255:
            continue
        # Smart decode: CRC-ok shards are reliable data;
        # non-CRC shards are treated as erasures (2× correction capacity)
        inner = _outer_rs_decode_smart(shard_map, crc_ok_set, n_rs)
        if inner is not None:
            break

    if inner is None:
        result["error"] = "Outer RS decode failed — too many corrupted shards."
        result["confidence"] = 0.35
        return result

    result["inner_codeword"] = inner
    result["confidence"] = 0.85  # High confidence, pending crypto verify
    return result
