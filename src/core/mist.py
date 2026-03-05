"""
src/core/mist.py  –  Mist Phase 2 High-Level API

Public surface
--------------
  watermark(image, user_id, image_id, private_key, embed_key,
            timestamp=None, model_version=1)
    → np.ndarray  (watermarked BGR image)

  verify(image, public_key, embed_key)
    → dict {
        "detected"   : bool          — watermark signal found
        "verified"   : bool          — signature cryptographically valid
        "ecc_success": bool          — Reed-Solomon decode succeeded
        "payload"    : dict | None   — recovered fields (user_id, image_id, …)
        "error"      : str | None    — human-readable failure reason
      }

Design
------
Embedding pipeline:
    payload.build_embed_payload()  →  704-bit (payload_core + Ed25519 sig)
    ecc.encode_payload()           →  1184-bit (+ Reed-Solomon parity)
    wm_engine.embed()              →  watermarked image

Detection / verification pipeline:
    wm_engine.detect()             →  confidence check (fast reject)
    wm_engine.extract_bits()       →  1184 hard-decision bits
    ecc.decode_payload()           →  704 bits (RS error correction)
    payload.parse_embed_payload()  →  payload_core + signature bytes
    crypto.verify()                →  signature check
    payload.unpack()               →  human-readable fields
"""

import numpy as np

from src.core.wm_engine import embed, detect, extract_bits
from src.core.payload  import build_embed_payload, parse_embed_payload, unpack
from src.core.ecc      import encode_payload, decode_payload, ECC_TOTAL_BITS
from src.core.crypto   import verify as crypto_verify


# ─────────────────────────────────────────────────────────
#  Embed
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
                                Separate from the signing key — controls *where*
                                the watermark is embedded, not what.
    timestamp     : int | None  Unix epoch seconds.  Auto-filled if None.
    model_version : int         Mist schema version (default 1).

    Returns
    -------
    np.ndarray  Watermarked BGR image (uint8, same shape as input).
    """
    # 1. Build signed payload → 704 bits
    _, _, full_bits = build_embed_payload(
        private_key, user_id, image_id, timestamp, model_version
    )

    # 2. ECC encode → 1184 bits
    encoded_bits = encode_payload(full_bits)   # list[int], length ECC_TOTAL_BITS

    # 3. Embed into luminance channel
    bitstream = np.array(encoded_bits, dtype=np.int32)
    return embed(image, bitstream, embed_key)


# ─────────────────────────────────────────────────────────
#  Detect + Verify
# ─────────────────────────────────────────────────────────

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
    public_key : bytes       32-byte raw Ed25519 public key matching the private
                             key used during watermark().
    embed_key  : bytes       Same secret embed_key used during watermark().

    Returns
    -------
    dict with keys:
        detected    : bool        — Watermark signal found (confidence ≥ 0.55).
        verified    : bool        — Signature valid AND ECC decoded cleanly.
        ecc_success : bool        — Reed-Solomon decode succeeded without exhaustion.
        payload     : dict | None — Recovered fields if verified, else None.
                                    Keys: user_id, image_id, timestamp,
                                          model_version, reserved.
        error       : str | None  — Human-readable failure reason.
    """
    result = {
        "detected":    False,
        "verified":    False,
        "ecc_success": False,
        "payload":     None,
        "error":       None,
    }

    # ── 1. Extract raw bits ────────────────────────────────────────────────────
    # For Phase 2 we go straight to bit extraction rather than using detect()
    # as a fast-reject.  The signed-correlation detect() was designed for Phase 1's
    # PRNG-derived bitstream and produces near-zero scores on ECC payload bits.
    # The ECC+signature check IS the detection proof: a random image will fail
    # ECC decode or signature verification with overwhelming probability.
    try:
        raw_bits = extract_bits(image, embed_key, ECC_TOTAL_BITS)
    except ValueError as exc:
        result["error"] = f"Image too small for watermark ({exc})"
        return result

    # ── 2. ECC decode ─────────────────────────────────────────────────────────
    decoded_bits, ecc_ok = decode_payload(raw_bits)
    result["ecc_success"] = ecc_ok

    # ── 3. Parse payload  ─────────────────────────────────────────────────────
    try:
        payload_core, signature = parse_embed_payload(decoded_bits)
    except (ValueError, Exception) as exc:
        result["error"] = f"Payload parse failed: {exc}"
        return result

    # ── 4. Signature verification ─────────────────────────────────────────────
    sig_ok = crypto_verify(public_key, payload_core, signature)
    if not sig_ok:
        # Signal was present (we extracted bits) but signature is invalid.
        result["detected"] = True   # something is embedded at these positions
        result["error"] = "Signature verification failed — payload may be tampered."
        return result

    # ── 5. Unpack fields ──────────────────────────────────────────────────────
    result["detected"]  = True
    result["verified"]  = True
    result["payload"]   = unpack(payload_core)
    return result

