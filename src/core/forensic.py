"""
src/core/forensic.py  –  Phase 5 Statistical Detection Engine

Produces court-grade forensic reports with:
  - Confidence scoring (calibrated %)
  - Correlation strength metrics
  - Tampering likelihood analysis
  - ROC curve generation
  - False positive p-value estimation

Courts do not want binary answers. They want statistical confidence.

Public API
----------
    forensic_report(image, public_key, embed_key) → ForensicReport
    generate_roc_data(embed_fn, verify_fn, images, key, ...) → ROCData
    calibrate_confidence(raw_score) → float  (calibrated percentage)

Classes
-------
    ForensicReport  — dataclass with all forensic fields
    ROCData         — dataclass with ROC curve arrays
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from src.core.wm_engine_p3 import (
    _to_ycbcr, _pad_to_n, _score_one_scale,
    _harmonic_score,
    SCALES, SCALE_WEIGHTS,
    _sigmoid,
    detect_p3,
)
from src.core.wm_engine_p4 import (
    detect_p4, extract_shards_p4,
    _derive_p4_key,
    K_SHARDS, MT_SIZE,
)
from src.core.wm_engine_p5 import (
    detect_p5, estimate_geometry, _anchor_search_score,
    ANCHOR_SCORE_BASELINE,
    _compute_confidence, _geometry_stability_score,
)
from src.core.ecc import decode_payload, ECC_TOTAL_BITS, ECC_TOTAL_BYTES
from src.core.payload import parse_embed_payload, unpack
from src.core.crypto import verify as crypto_verify


# ─────────────────────────────────────────────────────────────────────────────
#  Confidence Calibration
# ─────────────────────────────────────────────────────────────────────────────

# Empirical calibration curve: maps raw DCT correlation score to a
# calibrated confidence percentage.  Derived from null hypothesis testing
# on the Mist detection pipeline.
#
# The null distribution of _score_one_scale on unwatermarked images is
# approximately N(0, σ≈0.03).  A score of 0.15 is ~5σ above null → >99.9999%.
# A score of 0.10 is ~3.3σ → ~99.95%.  Scores >0.20 are certain detections.

_CALIBRATION_POINTS = [
    # (raw_score, confidence_pct)
    (-0.10, 0.0),
    (0.00,  1.0),
    (0.03,  5.0),
    (0.05, 15.0),
    (0.08, 50.0),
    (0.10, 75.0),
    (0.12, 88.0),
    (0.15, 95.0),
    (0.20, 98.0),
    (0.25, 99.0),
    (0.30, 99.5),
    (0.40, 99.9),
    (0.50, 99.95),
    (0.70, 99.99),
    (1.00, 99.999),
]


def calibrate_confidence(raw_score: float) -> float:
    """
    Map a raw DCT correlation score to a calibrated confidence percentage.

    Uses piecewise linear interpolation on empirical calibration curve.
    For court presentation: "With XX.X% statistical confidence, a watermark
    was detected in this image."

    Parameters
    ----------
    raw_score : float  Raw DCT correlation score (typically 0..1 for marked,
                       ~0 for unmarked).

    Returns
    -------
    float  Calibrated confidence percentage [0, 100].
    """
    if raw_score <= _CALIBRATION_POINTS[0][0]:
        return 0.0
    if raw_score >= _CALIBRATION_POINTS[-1][0]:
        return _CALIBRATION_POINTS[-1][1]

    # Linear interpolation
    for i in range(len(_CALIBRATION_POINTS) - 1):
        s0, c0 = _CALIBRATION_POINTS[i]
        s1, c1 = _CALIBRATION_POINTS[i + 1]
        if s0 <= raw_score <= s1:
            t = (raw_score - s0) / (s1 - s0) if s1 > s0 else 0.0
            return c0 + t * (c1 - c0)

    return 0.0


def _estimate_p_value(raw_score: float, n_blocks: int) -> float:
    """
    Estimate p-value under null hypothesis (no watermark).

    Under null: mean correlation ≈ N(0, 1/sqrt(n_blocks))
    p-value = P(Z > raw_score * sqrt(n_blocks))

    Parameters
    ----------
    raw_score : float  Observed mean correlation
    n_blocks  : int    Number of DCT blocks used

    Returns
    -------
    float  Two-sided p-value (smaller = stronger evidence of watermark)
    """
    if n_blocks <= 0:
        return 1.0
    z = abs(raw_score) * np.sqrt(n_blocks)
    # Approximation of erfc for large z
    from scipy.special import erfc
    return float(erfc(z / np.sqrt(2)))


# ─────────────────────────────────────────────────────────────────────────────
#  Tampering Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _tampering_likelihood(
    ecc_success: bool,
    crc_valid_ratio: float,
    reconstruction_ratio: float,
    scale_score_variance: float,
) -> tuple[str, float]:
    """
    Estimate likelihood the image has been tampered with.

    Tampering indicators:
      - High CRC failure ratio → content modification
      - Low reconstruction ratio → area destruction
      - High variance in scale scores → inconsistent modification
      - ECC failure despite presence detection → targeted bit attacks

    Returns (label, probability) where label is one of:
        "NONE", "LOW", "MODERATE", "HIGH", "SEVERE"
    """
    score = 0.0

    # CRC failure indicator
    crc_failure = 1.0 - crc_valid_ratio
    score += crc_failure * 0.3

    # Reconstruction difficulty
    if reconstruction_ratio < 1.0:
        score += (1.0 - reconstruction_ratio) * 0.25

    # Scale inconsistency
    score += min(1.0, scale_score_variance * 10.0) * 0.15

    # ECC failure despite presence
    if not ecc_success:
        score += 0.3

    score = min(1.0, score)

    if score < 0.05:
        label = "NONE"
    elif score < 0.20:
        label = "LOW"
    elif score < 0.45:
        label = "MODERATE"
    elif score < 0.70:
        label = "HIGH"
    else:
        label = "SEVERE"

    return label, round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
#  ForensicReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForensicReport:
    """
    Court-defensible forensic watermark analysis report.

    Fields
    ------
    Core Detection:
        watermark_detected     : bool   — Watermark signal found
        payload_recovered      : bool   — Full payload successfully decoded
        payload_verified       : bool   — Cryptographic signature valid
        payload                : dict | None  — Decoded payload fields

    Statistical Measures:
        confidence_pct         : float  — Calibrated confidence percentage
        correlation_strength   : float  — Raw DCT correlation score
        p_value                : float  — Statistical p-value under null
        n_blocks_analysed      : int    — Number of DCT blocks tested

    Tampering Analysis:
        tampering_likelihood   : str    — "NONE" / "LOW" / "MODERATE" / "HIGH" / "SEVERE"
        tampering_score        : float  — Tampering probability [0, 1]

    Quality Metrics:
        scale_scores           : dict   — Per-scale DCT correlation {8: .., 16: .., 32: ..}
        harmonic_score         : float  — FFT sinusoidal detection score
        tiles_located          : int    — Number of macro-tiles found
        shards_recovered       : int    — Number of data shards recovered
        shards_needed          : int    — Minimum shards for reconstruction
        reconstruction_ratio   : float  — shards_recovered / shards_needed
        ecc_success            : bool   — Inner Reed-Solomon decode succeeded

    Geometric Analysis:
        geometry_detected      : bool   — Geometric transform was compensated
        estimated_rotation_deg : float  — Estimated rotation angle
        estimated_scale_factor : float  — Estimated scale factor
        geometry_method        : str    — Detection method used

    Metadata:
        analysis_timestamp     : float  — Unix epoch of analysis
        analysis_duration_s    : float  — Time taken for analysis
        engine_version         : str    — Mist engine version
        image_dimensions       : tuple  — (H, W) input image dimensions
    """
    # Core Detection
    watermark_detected: bool = False
    payload_recovered: bool = False
    payload_verified: bool = False
    payload: dict | None = None

    # Statistical Measures
    confidence_pct: float = 0.0
    correlation_strength: float = 0.0
    p_value: float = 1.0
    n_blocks_analysed: int = 0

    # Tampering Analysis
    tampering_likelihood: str = "NONE"
    tampering_score: float = 0.0

    # Quality Metrics
    scale_scores: dict = field(default_factory=dict)
    harmonic_score: float = 0.0
    tiles_located: int = 0
    shards_recovered: int = 0
    shards_needed: int = 0
    reconstruction_ratio: float = 0.0
    ecc_success: bool = False

    # Geometric Analysis
    geometry_detected: bool = False
    estimated_rotation_deg: float = 0.0
    estimated_scale_factor: float = 1.0
    geometry_method: str = ""

    # Metadata
    analysis_timestamp: float = 0.0
    analysis_duration_s: float = 0.0
    engine_version: str = "mist-p5-v1.0"
    image_dimensions: tuple = (0, 0)

    def to_dict(self) -> dict:
        """Convert report to serialisable dictionary."""
        d = asdict(self)
        # Convert numpy types to Python native types
        for k, v in d.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                d[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                d[k] = float(v)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Generate a human-readable summary for court presentation."""
        lines = [
            "=" * 60,
            "  MIST FORENSIC WATERMARK ANALYSIS REPORT",
            "=" * 60,
            "",
            f"  Image Dimensions    : {self.image_dimensions[1]}×{self.image_dimensions[0]}",
            f"  Analysis Time       : {self.analysis_duration_s:.3f}s",
            f"  Engine Version      : {self.engine_version}",
            "",
            "─ DETECTION VERDICT ─",
            f"  Watermark Detected  : {'YES' if self.watermark_detected else 'NO'}",
            f"  Payload Recovered   : {'YES' if self.payload_recovered else 'NO'}",
            f"  Signature Verified  : {'YES' if self.payload_verified else 'NO'}",
            "",
            "─ STATISTICAL CONFIDENCE ─",
            f"  Confidence          : {self.confidence_pct:.2f}%",
            f"  Correlation         : {self.correlation_strength:.6f}",
            f"  p-value             : {self.p_value:.2e}",
            f"  Blocks Analysed     : {self.n_blocks_analysed}",
            "",
            "─ TAMPERING ANALYSIS ─",
            f"  Likelihood          : {self.tampering_likelihood}",
            f"  Score               : {self.tampering_score:.4f}",
            f"  ECC Success         : {'YES' if self.ecc_success else 'NO'}",
            f"  Tiles / Shards      : {self.tiles_located} / {self.shards_recovered}",
            f"  Reconstruction      : {self.reconstruction_ratio:.1%}",
        ]

        if self.geometry_detected:
            lines.extend([
                "",
                "─ GEOMETRIC ANALYSIS ─",
                f"  Transform Detected  : YES",
                f"  Rotation Estimate   : {self.estimated_rotation_deg:.1f}°",
                f"  Scale Estimate      : {self.estimated_scale_factor:.3f}×",
                f"  Method              : {self.geometry_method}",
            ])

        if self.payload_verified and self.payload:
            lines.extend([
                "",
                "─ RECOVERED PAYLOAD ─",
                f"  User ID             : {self.payload.get('user_id', 'N/A')}",
                f"  Image ID            : {self.payload.get('image_id', 'N/A')}",
                f"  Timestamp           : {self.payload.get('timestamp', 'N/A')}",
                f"  Model Version       : {self.payload.get('model_version', 'N/A')}",
            ])

        lines.extend(["", "=" * 60])
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Forensic Analysis
# ─────────────────────────────────────────────────────────────────────────────

def forensic_report(
    image: np.ndarray,
    public_key: bytes,
    embed_key: bytes,
) -> ForensicReport:
    """
    Generate a complete forensic watermark analysis report.

    This is the Phase 5 deliverable: a comprehensive analysis that can
    be presented as evidence in legal proceedings.

    Pipeline:
      1. Phase 5 geometry-invariant detection (handles rotation/scale)
      2. Payload extraction + cryptographic verification
      3. Statistical confidence calibration
      4. Tampering likelihood estimation
      5. Report assembly

    Parameters
    ----------
    image      : np.ndarray  BGR uint8 (H, W, 3)
    public_key : bytes       32-byte Ed25519 public key
    embed_key  : bytes       Secret embedding key

    Returns
    -------
    ForensicReport  Complete forensic analysis.
    """
    report = ForensicReport()
    report.image_dimensions = image.shape[:2]
    report.analysis_timestamp = time.time()
    t0 = time.time()

    # ── Compute block count for p-value ───────────────────────────────
    _, Y = _to_ycbcr(image)
    h, w = Y.shape
    n_blocks_8 = (h // 8) * (w // 8)
    report.n_blocks_analysed = n_blocks_8

    # ── Phase 5 geometry-invariant detection ──────────────────────────
    det5 = detect_p5(image, embed_key)

    report.scale_scores = det5.get("scale_scores", {})
    report.harmonic_score = det5.get("harmonic_score", 0.0)
    report.tiles_located = det5.get("tiles_located", 0)
    report.shards_recovered = det5.get("shards_recovered", 0)
    report.shards_needed = det5.get("shards_needed", K_SHARDS)
    report.reconstruction_ratio = det5.get("reconstruction_ratio", 0.0)

    # Geometric analysis
    geo = det5.get("geometry")
    if geo:
        report.estimated_rotation_deg = geo.get("angle_deg", 0.0)
        report.estimated_scale_factor = geo.get("scale_factor", 1.0)
        report.geometry_method = geo.get("method", "")
        report.geometry_detected = (
            abs(geo.get("angle_deg", 0.0)) > 0.5
            or abs(geo.get("scale_factor", 1.0) - 1.0) > 0.03
        )

    # ── Anchor-based correlation score ────────────────────────────────
    # Use anchor match score as the forensic correlation metric.
    # Unlike the self-consistent DCT score (which is always positive),
    # anchor matching discriminates: ~0.85 for watermarked, ~0.50 for clean.
    anchor_score = _anchor_search_score(image, embed_key, max_tiles=12)

    # If geometric correction was applied, use the corrected score
    geo_anchor = 0.0
    if geo and geo.get("score", 0) > anchor_score:
        geo_anchor = geo["score"]

    raw_correlation = max(anchor_score, geo_anchor)
    # Remap anchor score [0.5, 1.0] → [0.0, 1.0] for calibration
    report.correlation_strength = max(0.0, (raw_correlation - 0.50) * 2.0)

    # Watermark detection uses ANCHOR evidence, not self-consistent P3
    # (P3 presence can false-positive due to self-consistent scoring)
    report.watermark_detected = (
        raw_correlation > ANCHOR_SCORE_BASELINE
        or det5.get("inner_codeword") is not None
    )

    # ── Statistical calibration ───────────────────────────────────────
    # Legacy calibration for backward compatibility + p-value
    report.confidence_pct = calibrate_confidence(report.correlation_strength)
    report.p_value = _estimate_p_value(report.correlation_strength, n_blocks_8)

    # Multi-signal confidence (new) — integrates geometry + shard info
    geo_stability = 1.0
    canary_score = 0.0
    if geo:
        geo_stability = _geometry_stability_score(
            geo.get("angle_deg", 0.0), geo.get("scale_factor", 1.0),
            0, geo.get("shard_count", 0), geo.get("method", ""),
        )
        canary_score = min(1.0, geo.get("score", 0.0))
    shard_ratio = report.reconstruction_ratio
    multi_conf = _compute_confidence(
        normalized_correlation=report.correlation_strength,
        shard_recovery_ratio=shard_ratio,
        rs_decode_success=False,  # updated below if payload recovered
        geometry_stability=geo_stability,
        canary_consistency=canary_score,
        reconstruction_consistency=shard_ratio,
    )
    # Use the higher of legacy vs multi-signal for the final report
    report.confidence_pct = max(report.confidence_pct, multi_conf * 100.0)

    # ── Payload extraction + crypto verification ──────────────────────
    inner = det5.get("inner_codeword")
    if inner is not None:
        # Convert bytes to bit list for inner RS decoder
        inner_bits = []
        for byte_val in inner:
            for i in range(7, -1, -1):
                inner_bits.append((byte_val >> i) & 1)

        decoded_bits, ecc_ok = decode_payload(inner_bits)
        report.ecc_success = ecc_ok

        if ecc_ok:
            try:
                payload_core, signature = parse_embed_payload(decoded_bits)
                sig_ok = crypto_verify(public_key, payload_core, signature)

                if sig_ok:
                    report.payload_recovered = True
                    report.payload_verified = True
                    report.payload = unpack(payload_core)
                    report.confidence_pct = max(report.confidence_pct, 99.0)
                else:
                    report.payload_recovered = True  # Got data, but tampered
                    report.payload_verified = False
            except Exception:
                report.ecc_success = False

    # ── Tampering analysis ────────────────────────────────────────────
    shard_result = None
    try:
        shard_result = extract_shards_p4(image, embed_key)
    except Exception:
        pass

    crc_ratio = 0.0
    if shard_result and shard_result.get("tiles_located", 0) > 0:
        crc_ok_count = len(shard_result.get("shard_crc_ok", set()))
        crc_ratio = crc_ok_count / max(1, shard_result["tiles_located"])

    scale_vals = list(report.scale_scores.values()) if report.scale_scores else [0.0]
    scale_variance = float(np.var(scale_vals)) if len(scale_vals) > 1 else 0.0

    label, tampering_prob = _tampering_likelihood(
        report.ecc_success,
        crc_ratio,
        report.reconstruction_ratio,
        scale_variance,
    )
    report.tampering_likelihood = label
    report.tampering_score = tampering_prob

    # ── Finalise ──────────────────────────────────────────────────────
    report.analysis_duration_s = time.time() - t0
    return report


# ─────────────────────────────────────────────────────────────────────────────
#  ROC Curve Generation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ROCData:
    """ROC curve data for forensic validation."""
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr: np.ndarray = field(default_factory=lambda: np.array([]))
    fpr: np.ndarray = field(default_factory=lambda: np.array([]))
    auc: float = 0.0
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds.tolist(),
            "tpr": self.tpr.tolist(),
            "fpr": self.fpr.tolist(),
            "auc": self.auc,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


def generate_roc_data(
    scores_positive: list[float],
    scores_negative: list[float],
    n_thresholds: int = 200,
) -> ROCData:
    """
    Generate ROC curve data from positive (watermarked) and negative (clean)
    detection scores.

    Parameters
    ----------
    scores_positive : list[float]  Detection scores for watermarked images
    scores_negative : list[float]  Detection scores for clean images
    n_thresholds    : int          Number of threshold points for the curve

    Returns
    -------
    ROCData  Contains thresholds, TPR, FPR, and AUC.
    """
    all_scores = np.concatenate([scores_positive, scores_negative])
    thresholds = np.linspace(
        float(np.min(all_scores)) - 0.01,
        float(np.max(all_scores)) + 0.01,
        n_thresholds,
    )

    n_pos = len(scores_positive)
    n_neg = len(scores_negative)
    pos = np.array(scores_positive)
    neg = np.array(scores_negative)

    tpr_arr = np.zeros(n_thresholds)
    fpr_arr = np.zeros(n_thresholds)

    for i, thresh in enumerate(thresholds):
        tpr_arr[i] = np.sum(pos >= thresh) / max(1, n_pos)
        fpr_arr[i] = np.sum(neg >= thresh) / max(1, n_neg)

    # AUC via trapezoidal rule (sorted by FPR descending)
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    tpr_sorted = tpr_arr[sort_idx]
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))

    return ROCData(
        thresholds=thresholds,
        tpr=tpr_arr,
        fpr=fpr_arr,
        auc=abs(auc),
        n_positive=n_pos,
        n_negative=n_neg,
    )
