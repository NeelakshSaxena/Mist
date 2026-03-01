import numpy as np
import reedsolo

# Initialize RS codec. Let's add 10 ECC bytes for an 8-byte payload.
# This makes message length = 18 bytes.
rs = reedsolo.RSCodec(10)

def bits_to_bytes(bit_list):
    """Convert a list of 0s and 1s to a bytearray."""
    b = bytearray()
    for i in range(0, len(bit_list), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bit_list):
                byte_val |= (bit_list[i + j] << (7 - j))
        b.append(byte_val)
    return b

def bytes_to_bits(byte_data):
    """Convert bytearray to a list of 0s and 1s."""
    bits = []
    for b in byte_data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def encode_payload(bit_list):
    """
    Takes 64 bits (8 bytes), encodes with Reed-Solomon,
    returns 144 bits (18 bytes).
    """
    payload_bytes = bits_to_bytes(bit_list)
    encoded_bytes = rs.encode(payload_bytes)
    return bytes_to_bits(encoded_bytes)

def decode_payload(bit_list):
    """
    Takes 144 bits (18 bytes), decodes with Reed-Solomon,
    returns the original 64 bits (8 bytes) and whether it was successful.
    """
    encoded_bytes = bits_to_bytes(bit_list)
    try:
        decoded_bytes, _, _ = rs.decode(encoded_bytes)
        return bytes_to_bits(decoded_bytes), True
    except reedsolo.ReedSolomonError:
        # If it fails to decode, just return the first 64 bits as a best attempt
        best_effort = bytes_to_bits(encoded_bytes[:8])
        return best_effort, False
