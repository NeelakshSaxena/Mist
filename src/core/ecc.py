"""
src/core/ecc.py  –  Reed-Solomon Error Correction

Phase 2 update: expanded for 88-byte (704-bit) payload.
  • Data: 88 bytes (payload_core 24 bytes + Ed25519 signature 64 bytes)
  • ECC:  40 parity bytes  (~31% overhead, corrects up to 20 byte-errors)
  • Total encoded: 128 bytes = 1024 bits

The RS codec is initialised once as a module-level singleton.
All callers use encode_payload() / decode_payload().
"""

import reedsolo

# ─────────────────────────────────────────────────────────
#  Constants (must stay in sync with embed.py / payload.py)
# ─────────────────────────────────────────────────────────
ECC_PAYLOAD_BYTES_DATA  = 88    # payload_core (24) + signature (64)
ECC_PARITY_BYTES        = 60    # corrects up to 30 byte-errors (~20% of 148-byte codeword)
ECC_TOTAL_BYTES         = ECC_PAYLOAD_BYTES_DATA + ECC_PARITY_BYTES  # 148 bytes
ECC_TOTAL_BITS          = ECC_TOTAL_BYTES * 8   # 1184 bits

# Initialise once; RSCodec(nsym) where nsym = parity symbols
rs = reedsolo.RSCodec(ECC_PARITY_BYTES)


# ─────────────────────────────────────────────────────────
#  Bit / byte helpers
# ─────────────────────────────────────────────────────────

def bits_to_bytes(bit_list: list) -> bytearray:
    """Convert a list of 0/1 ints to a bytearray (MSB first)."""
    b = bytearray()
    for i in range(0, len(bit_list), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bit_list):
                byte_val |= (bit_list[i + j] << (7 - j))
        b.append(byte_val)
    return b


def bytes_to_bits(byte_data) -> list:
    """Convert bytes / bytearray to a list of 0/1 ints (MSB first)."""
    bits = []
    for b in byte_data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


# ─────────────────────────────────────────────────────────
#  ECC encode / decode
# ─────────────────────────────────────────────────────────

def encode_payload(bit_list: list) -> list:
    """
    ECC-encode a 704-bit payload bit list.

    Accepts a list of 704 bits (88 bytes).
    Returns 1024 bits (128 bytes: 88 data + 40 parity).
    """
    payload_bytes = bits_to_bytes(bit_list)
    # reedsolo wants exactly ECC_PAYLOAD_BYTES_DATA bytes
    # Pad with zeros if caller provided fewer
    if len(payload_bytes) < ECC_PAYLOAD_BYTES_DATA:
        payload_bytes = payload_bytes + bytearray(ECC_PAYLOAD_BYTES_DATA - len(payload_bytes))
    payload_bytes = payload_bytes[:ECC_PAYLOAD_BYTES_DATA]

    encoded_bytes = rs.encode(payload_bytes)   # returns bytearray of 128 bytes
    return bytes_to_bits(encoded_bytes)


def decode_payload(bit_list: list) -> tuple[list, bool]:
    """
    ECC-decode a 1024-bit encoded bit list.

    Accepts a list of 1024 bits (128 bytes).
    Returns:
        decoded_bits : list[int]  – 704 bits (payload_core + signature)
        success      : bool       – True if RS decoding succeeded without exhaustion
    """
    encoded_bytes = bits_to_bytes(bit_list)

    try:
        decoded_bytes, _, _ = rs.decode(encoded_bytes)
        return bytes_to_bits(decoded_bytes), True
    except reedsolo.ReedSolomonError:
        # Best-effort: return the first 88 bytes as raw (likely corrupted)
        best_effort = bytes_to_bits(encoded_bytes[:ECC_PAYLOAD_BYTES_DATA])
        return best_effort, False
