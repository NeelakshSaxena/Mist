"""
src/core/crypto.py  –  Phase 2 Cryptographic Layer

Key design decisions:
  • Ed25519 (RFC 8032) is used instead of NIST P-256 ECDSA.
    Ed25519 signature size is exactly 512 bits (64 bytes), same as NIST P-256 r+s,
    but far simpler to work with: no nonce management, deterministic, no ASN.1.
  • We explicitly SHA-256 the payload_core ourselves before signing.
    This gives us clear preimage control: Sign(sha256(payload_core)).
  • The signing key is completely separate from the embedding PRNG seed.

Public surface
--------------
  generate_keys()   → (private_key_bytes: bytes, public_key_bytes: bytes)
  sign(private_key_bytes, payload_core_bytes)   → signature_bytes (64 bytes)
  verify(public_key_bytes, payload_core_bytes, signature_bytes) → bool
  sha256_payload(payload_core_bytes) → bytes (32 bytes)
"""

import hashlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
)
from cryptography.exceptions import InvalidSignature

# Ed25519 signature is always exactly 64 bytes = 512 bits.
SIGNATURE_BYTES = 64
SIGNATURE_BITS  = SIGNATURE_BYTES * 8  # 512


def generate_keys() -> tuple[bytes, bytes]:
    """
    Generate a fresh Ed25519 key pair.

    Returns
    -------
    private_key_bytes : bytes  (32-byte raw seed, keep secret)
    public_key_bytes  : bytes  (32-byte raw public key, embed or distribute)
    """
    priv = Ed25519PrivateKey.generate()
    pub  = priv.public_key()

    priv_bytes = priv.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    pub_bytes = pub.public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    return priv_bytes, pub_bytes


def _load_private_key(private_key_bytes: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(private_key_bytes)


def _load_public_key(public_key_bytes: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(public_key_bytes)


def sha256_payload(payload_core_bytes: bytes) -> bytes:
    """
    Compute SHA-256 digest of raw payload_core bytes.

    This is the explicit preimage we control. The digest is
    what gets signed — even though Ed25519 hashes internally,
    we apply SHA-256 here for a clear, auditable preimage.
    """
    return hashlib.sha256(payload_core_bytes).digest()


def sign(private_key_bytes: bytes, payload_core_bytes: bytes) -> bytes:
    """
    Sign a payload.

    Flow: SHA-256(payload_core_bytes) → Ed25519.Sign(digest)

    Parameters
    ----------
    private_key_bytes : bytes   Raw 32-byte Ed25519 private key seed.
    payload_core_bytes : bytes  The raw packed payload (before ECC).

    Returns
    -------
    signature : bytes  64 bytes (512 bits). Embed after payload_core.
    """
    digest = sha256_payload(payload_core_bytes)
    priv   = _load_private_key(private_key_bytes)
    return priv.sign(digest)


def verify(public_key_bytes: bytes, payload_core_bytes: bytes, signature_bytes: bytes) -> bool:
    """
    Verify a payload signature.

    Flow: SHA-256(payload_core_bytes) → Ed25519.Verify(digest, signature)

    Parameters
    ----------
    public_key_bytes   : bytes  32-byte Ed25519 public key.
    payload_core_bytes : bytes  Recovered payload_core (before hashing).
    signature_bytes    : bytes  64-byte signature recovered from watermark.

    Returns
    -------
    bool  True if authentic and untampered, False otherwise.
    """
    try:
        digest = sha256_payload(payload_core_bytes)
        pub    = _load_public_key(public_key_bytes)
        pub.verify(signature_bytes, digest)
        return True
    except (InvalidSignature, Exception):
        return False
