"""
src/core/payload.py  –  Phase 2 Payload Schema (Frozen v1.0)

Frozen 704-bit total payload layout (Option 2 – Forensic Grade):

  payload_core  =  192 bits  (24 bytes)
  ┌────────────────┬────────┐
  │ Field          │  Bits  │
  ├────────────────┼────────┤
  │ user_id        │   64   │
  │ image_id       │   64   │
  │ timestamp      │   32   │
  │ model_version  │   16   │
  │ reserved       │   16   │
  └────────────────┴────────┘

  signature  =  512 bits (64 bytes) — Ed25519 over SHA-256(payload_core)

  embed_payload  =  payload_core || signature  =  704 bits (88 bytes)

This schema is intentionally frozen. Do NOT add fields without bumping
model_version and writing a migration note.

Public surface
--------------
  pack(user_id, image_id, timestamp, model_version, reserved)
    → payload_core_bytes (24 bytes)

  unpack(payload_core_bytes)
    → dict{"user_id", "image_id", "timestamp", "model_version", "reserved"}

  build_embed_payload(private_key_bytes, user_id, image_id, timestamp,
                      model_version, reserved)
    → (payload_core_bytes, signature_bytes, full_bits: list[int])

  parse_embed_payload(full_bits: list[int])
    → (payload_core_bytes, signature_bytes)

  bits_to_bytes(bit_list) / bytes_to_bits(byte_data)   # convenience re-exports
"""

import struct
import time
import os
from src.core.crypto import sign, SIGNATURE_BYTES

# ─────────────────────────────────────────────────────────
#  Schema constants (frozen)
# ─────────────────────────────────────────────────────────
SCHEMA_VERSION = 1

PAYLOAD_CORE_BYTES = 24    # 192 bits
PAYLOAD_CORE_BITS  = PAYLOAD_CORE_BYTES * 8  # 192

SIGNATURE_BITS  = SIGNATURE_BYTES * 8  # 512

EMBED_PAYLOAD_BYTES = PAYLOAD_CORE_BYTES + SIGNATURE_BYTES   # 88 bytes
EMBED_PAYLOAD_BITS  = EMBED_PAYLOAD_BYTES * 8                # 704

# Struct format: big-endian  Q Q I H H  = 8+8+4+2+2 = 24 bytes
_STRUCT_FMT = ">QQIHH"


# ─────────────────────────────────────────────────────────
#  Bit / byte helpers (thin wrappers kept here so callers
#  only import from one place)
# ─────────────────────────────────────────────────────────

def bytes_to_bits(byte_data: bytes | bytearray) -> list[int]:
    """Convert bytes → list of 0/1 ints, MSB first."""
    bits = []
    for b in byte_data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def bits_to_bytes(bit_list: list[int]) -> bytes:
    """Convert list of 0/1 ints → bytes, MSB first. Pads on right if needed."""
    out = bytearray()
    for i in range(0, len(bit_list), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bit_list):
                byte_val |= (bit_list[i + j] << (7 - j))
        out.append(byte_val)
    return bytes(out)


# ─────────────────────────────────────────────────────────
#  Schema pack / unpack
# ─────────────────────────────────────────────────────────

def pack(
    user_id: int,
    image_id: int,
    timestamp: int | None = None,
    model_version: int = SCHEMA_VERSION,
    reserved: int = 0,
) -> bytes:
    """
    Pack fields into 24-byte payload_core.

    Parameters
    ----------
    user_id        : int  (uint64)  User / account identifier.
    image_id       : int  (uint64)  Unique image identifier (e.g., hash or DB id).
    timestamp      : int  (uint32)  Unix epoch seconds. Auto-filled if None.
    model_version  : int  (uint16)  Mist model / schema version.
    reserved       : int  (uint16)  Reserved for future use, default 0.
    """
    if timestamp is None:
        timestamp = int(time.time())
    return struct.pack(_STRUCT_FMT, user_id, image_id, timestamp, model_version, reserved)


def unpack(payload_core_bytes: bytes) -> dict:
    """
    Unpack 24-byte payload_core back into a dict.

    Returns
    -------
    dict with keys: user_id, image_id, timestamp, model_version, reserved
    """
    if len(payload_core_bytes) < PAYLOAD_CORE_BYTES:
        raise ValueError(
            f"payload_core must be {PAYLOAD_CORE_BYTES} bytes, got {len(payload_core_bytes)}"
        )
    user_id, image_id, timestamp, model_version, reserved = struct.unpack(
        _STRUCT_FMT, payload_core_bytes[:PAYLOAD_CORE_BYTES]
    )
    return {
        "user_id":       user_id,
        "image_id":      image_id,
        "timestamp":     timestamp,
        "model_version": model_version,
        "reserved":      reserved,
    }


# ─────────────────────────────────────────────────────────
#  Build / parse full embed payload
# ─────────────────────────────────────────────────────────

def build_embed_payload(
    private_key_bytes: bytes,
    user_id: int,
    image_id: int,
    timestamp: int | None = None,
    model_version: int = SCHEMA_VERSION,
    reserved: int = 0,
) -> tuple[bytes, bytes, list[int]]:
    """
    Build the full 704-bit embeddable payload.

    Pipeline:
        pack → payload_core_bytes
        SHA-256(payload_core_bytes) → sign → signature_bytes
        payload_core_bytes || signature_bytes → embed bitstream

    Returns
    -------
    payload_core_bytes : bytes     (24 bytes)
    signature_bytes    : bytes     (64 bytes)
    full_bits          : list[int] (704 bits, ready for ECC encoding)
    """
    payload_core_bytes = pack(user_id, image_id, timestamp, model_version, reserved)
    signature_bytes    = sign(private_key_bytes, payload_core_bytes)

    full_bytes = payload_core_bytes + signature_bytes
    assert len(full_bytes) == EMBED_PAYLOAD_BYTES, \
        f"Expected {EMBED_PAYLOAD_BYTES} bytes, got {len(full_bytes)}"

    return payload_core_bytes, signature_bytes, bytes_to_bits(full_bytes)


def parse_embed_payload(full_bits: list[int]) -> tuple[bytes, bytes]:
    """
    Split a recovered 704-bit list into (payload_core_bytes, signature_bytes).

    Parameters
    ----------
    full_bits : list[int]  704 bits recovered from watermark (after ECC decode).

    Returns
    -------
    payload_core_bytes : bytes  (24 bytes)
    signature_bytes    : bytes  (64 bytes)
    """
    if len(full_bits) < EMBED_PAYLOAD_BITS:
        raise ValueError(
            f"Expected {EMBED_PAYLOAD_BITS} bits, got {len(full_bits)}"
        )
    full_bytes         = bits_to_bytes(full_bits[:EMBED_PAYLOAD_BITS])
    payload_core_bytes = full_bytes[:PAYLOAD_CORE_BYTES]
    signature_bytes    = full_bytes[PAYLOAD_CORE_BYTES:EMBED_PAYLOAD_BYTES]
    return payload_core_bytes, signature_bytes
