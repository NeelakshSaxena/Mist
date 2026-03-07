"""
src/core/mist.py  –  Mist High-Level API  (Phase 2 + Phase 3)

Public surface
--------------
Phase 2 (existing, unchanged):
  watermark(image, user_id, image_id, private_key, embed_key, ...)
  verify(image, public_key, embed_key)

Phase 3 (diffusion-resistant):
  watermark_p3(image, user_id, image_id, private_key, embed_key, ...)
    → np.ndarray  (watermarked BGR image, multi-scale + harmonic)

  verify_p3(image, public_key, embed_key)
    → dict {
        "detected"      : bool   — watermark signal found
        "verified"      : bool   — signature cryptographically valid
        "ecc_success"   : bool   — Reed-Solomon decode succeeded
        "payload"       : dict | None
        "error"         : str | None
        "scale_scores"  : dict   — per-scale DCT score
        "harmonic_score": float  — FFT harmonic detection score
      }

Design (Phase 3)
----------------
Embedding:
    payload.build_embed_payload()  →  704-bit signed payload
    ecc.encode_payload()           →  1184-bit ECC-encoded payload
    wm_engine_p3.embed_p3()        →  multi-scale + harmonic watermarked image

Detection:
    wm_engine_p3.extract_bits_p3() →  majority-voted 1184 bits
    ecc.decode_payload()           →  RS error correction
    payload.parse_embed_payload()  →  payload_core + signature
    crypto.verify()                →  Ed25519 signature check
"""

import numpy as np

from src.core.wm_engine    import embed, detect, extract_bits
from src.core.wm_engine_p3 import embed_p3, detect_p3, extract_bits_p3
from src.core.payload      import build_embed_payload, parse_embed_payload, unpack
from src.core.ecc          import encode_payload, decode_payload, ECC_TOTAL_BITS
from src.core.crypto       import verify as crypto_verify


# ─────────────────────────────────────────────────────────
#  Phase 2 — Embed + Verify
# ─────────────────────────────────────────────────────────

def watermark(
    image:         np.ndarray,
    user_id:       int,
    image_id:      int,
    private_key:   bytes,
    embed_key:     bytes,
    timestamp:     int | None = None,
    model_version: int = 1,
) -> np.ndarray:
    """
    Embed a cryptographically signed, ECC-protected watermark into an image.

    Parameters
    ----------
    image         : np.ndarray  BGR uint8 image.  Must be large enough to hold
                                ECC_TOTAL_BITS (1184) blocks — i.e. ≥ ~390×390 px.
    user_id       : int         uint64 user / account identifier.
    image_id      : int         uint64 unique image identifier (e.g. DB primary key).
    private_key   : bytes       32-byte raw Ed25519 private key seed (kept secret).
    embed_key     : bytes       Secret PRNG seed for coefficient-pair selection.
    timestamp     : int | None  Unix epoch seconds.  Auto-filled if None.
    model_version : int         Mist schema version (default 1).

    Returns
    -------
    np.ndarray  Watermarked BGR image (uint8, same shape as input).
    """
    _, _, full_bits = build_embed_payload(
        private_key, user_id, image_id, timestamp, model_version
    )
    encoded_bits = encode_payload(full_bits)
    bitstream = np.array(encoded_bits, dtype=np.int32)
    return embed(image, bitstream, embed_key)


def verify(
    image:      np.ndarray,
    public_key: bytes,
    embed_key:  bytes,
) -> dict:
    """
    Detect and cryptographically verify a Mist watermark.

    Parameters
    ----------
    image      : np.ndarray  BGR uint8 image (possibly attacked / compressed).
    public_key : bytes       32-byte raw Ed25519 public key.
    embed_key  : bytes       Same secret embed_key used during watermark().

    Returns
    -------
    dict with keys:
        detected    : bool        — Watermark signal found (confidence ≥ 0.55).
        verified    : bool        — Signature valid AND ECC decoded cleanly.
        ecc_success : bool        — Reed-Solomon decode succeeded.
        payload     : dict | None — Recovered fields if verified, else None.
        error       : str | None  — Human-readable failure reason.
    """
    result = {
        "detected":    False,
        "verified":    False,
        "ecc_success": False,
        "payload":     None,
        "error":       None,
    }

    try:
        raw_bits = extract_bits(image, embed_key, ECC_TOTAL_BITS)
    except ValueError as exc:
        result["error"] = f"Image too small for watermark ({exc})"
        return result

    decoded_bits, ecc_ok = decode_payload(raw_bits)
    result["ecc_success"] = ecc_ok

    try:
        payload_core, signature = parse_embed_payload(decoded_bits)
    except (ValueError, Exception) as exc:
        result["error"] = f"Payload parse failed: {exc}"
        return result

    sig_ok = crypto_verify(public_key, payload_core, signature)
    if not sig_ok:
        result["detected"] = True
        result["error"] = "Signature verification failed — payload may be tampered."
        return result

    result["detected"]  = True
    result["verified"]  = True
    result["payload"]   = unpack(payload_core)
    return result


# ─────────────────────────────────────────────────────────
#  Phase 3 — Diffusion-Resistant Embed + Verify
# ─────────────────────────────────────────────────────────

def watermark_p3(
    image:         np.ndarray,
    user_id:       int,
    image_id:      int,
    private_key:   bytes,
    embed_key:     bytes,
    timestamp:     int | None = None,
    model_version: int = 1,
) -> np.ndarray:
    """
    Phase 3 diffusion-resistant watermark embed.
    Parameters identical to watermark(). Uses multi-scale + harmonic engine.
    """
    _, _, full_bits = build_embed_payload(
        private_key, user_id, image_id, timestamp, model_version
    )
    encoded_bits = encode_payload(full_bits)
    bitstream = np.array(encoded_bits, dtype=np.int32)
    return embed_p3(image, bitstream, embed_key)


def verify_p3(
    image:      np.ndarray,
    public_key: bytes,
    embed_key:  bytes,
) -> dict:
    """
    Phase 3 detect + cryptographic verify.

    Returns dict with all keys from verify() plus:
        scale_scores   : dict  — {8: float, 16: float, 32: float}
        harmonic_score : float — FFT sinusoidal detection score [0,1]
    """
    result = {
        "detected":       False,
        "verified":       False,
        "ecc_success":    False,
        "payload":        None,
        "error":          None,
        "scale_scores":   {},
        "harmonic_score": 0.0,
    }

    det = detect_p3(image, embed_key)
    result["scale_scores"]   = det["scale_scores"]
    result["harmonic_score"] = det["harmonic_score"]

    try:
        raw_bits = extract_bits_p3(image, embed_key, ECC_TOTAL_BITS)
    except ValueError as exc:
        result["error"] = f"Image too small for Phase 3 watermark ({exc})"
        return result

    decoded_bits, ecc_ok = decode_payload(raw_bits)
    result["ecc_success"] = ecc_ok

    try:
        payload_core, signature = parse_embed_payload(decoded_bits)
    except Exception as exc:
        result["error"] = f"Payload parse failed: {exc}"
        return result

    sig_ok = crypto_verify(public_key, payload_core, signature)
    if not sig_ok:
        result["detected"] = True
        result["error"] = "Signature verification failed — payload may be tampered."
        return result

    result["detected"] = True
    result["verified"] = True
    result["payload"]  = unpack(payload_core)
    return result
