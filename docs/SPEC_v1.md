# 🌫 Mist Technical Specification v1.0

> **Status:** Active — Engineering Contract / Research Reference / Patent Groundwork  
> **Last Updated:** 2026-03-02  
> **Scope:** Phase 0 Deliverable

---

## 1. System Objective

Mist is a robust, forensic watermarking system designed to:

- Embed **traceable ownership data** inside generated images
- **Survive common real-world transformations**
- Enable **high-confidence attribution**
- Maintain **low perceptual distortion**

### Out of Scope (This Version)

| Concern | Notes |
|---|---|
| Adversarial / targeted removal attacks | Phase 3+ research scope |
| Heavy ML-based purification resistance | Future research |
| Malicious insider threats | Addressed via key management only |

---

## 2. Threat Model

### 2.1 Attacker Capabilities

The attacker is assumed to:

- ✅ Have access to the **final output image**
- ✅ Be able to apply **standard image editing tools** (JPEG, resize, crop, filters)
- ✅ Have full knowledge of the **general embedding approach** (Kerckhoffs's principle)

The attacker is assumed **NOT** to:

- ❌ Know the **secret embedding key `K`**
- ❌ Have access to **server-side infrastructure**
- ❌ Know the **payload signing private key**
- ❌ Be able to modify embedding parameters at generation time

### 2.2 Security Assumptions

| Asset | Location | Access |
|---|---|---|
| Embedding key `K` | Server-side only | Never exposed |
| Payload signing private key | Server-side HSM/secret store | Never exposed |
| Verification public key | Detector / Court system | Publicly available |

---

## 3. Supported Transformation Robustness

Mist **must remain detectable** after each of the following transformations applied independently and in combination:

| Transformation | Specification |
|---|---|
| **JPEG Compression** | Survive at Quality ≥ 30 |
| **Resize** | Detectable after scaling in range `[0.5×, 2×]` |
| **Cropping** | Must survive up to **30% area removal** |
| **Screenshot Simulation** | Re-encoding, subpixel distortions, gamma shift, quantization |
| **Minor Blur** | Gaussian blur, σ ≤ 1.5 |
| **Brightness Adjustment** | ±20% from baseline |
| **Contrast Adjustment** | ±20% from baseline |

---

## 4. Detection Standards

### 4.1 Confidence Threshold

Detection is declared when the computed score exceeds threshold **τ**:

```
confidence(image) ≥ τ
```

where **τ** is empirically calibrated to satisfy:

| Metric | Target |
|---|---|
| False Positive Rate (FPR) | **< 0.5%** (hard limit) |
| False Positive Rate (FPR) | **< 0.2%** (ideal target) |

### 4.2 Evaluation Dataset

| Set | Size | Condition |
|---|---|---|
| Non-watermarked images | ≥ 10,000 | Clean baseline |
| Watermarked images | ≥ 10,000 | All subjected to supported transforms |

### 4.3 Evaluation Metrics

| Metric | Target |
|---|---|
| True Positive Rate (TPR) | Maximized subject to FPR constraint |
| False Positive Rate (FPR) | < 0.5% |
| AUC (ROC Curve) | **≥ 0.98** |
| ROC Curve | Must be generated and archived per release |

---

## 5. Payload Structure

### 5.1 Canonical Payload Schema

```json
{
  "user_id":        "<UUID v4>",
  "timestamp":      "<ISO 8601>",
  "image_id":       "<UUID v4>",
  "model_version":  "<semver string>",
  "digital_signature": "<Ed25519 signature over above fields>"
}
```

### 5.2 Payload Constraints

| Constraint | Specification |
|---|---|
| Signing algorithm | **Ed25519** |
| Signature coverage | All fields except `digital_signature` itself |
| Error correction | **Reed-Solomon ECC** (mandatory) |
| Minimum ECC overhead | Sufficient for ≥ 20% symbol damage recovery |
| Hash integrity | SHA-256 over payload before signing |
| Bit-flip resistance | ECC redundancy required; raw unprotected bits not acceptable |

### 5.3 Encoding Pipeline

```
payload (JSON)
    │
    ▼
SHA-256 hash
    │
    ▼
Ed25519 sign (private key, server-side)
    │
    ▼
Serialize → binary bitstream
    │
    ▼
Reed-Solomon ECC encode
    │
    ▼
Map bits → DCT coefficient modulation
    │
    ▼
Embed into image
```

---

## 6. Failure Conditions

Mist is considered **failed** if any of the following occur:

| Condition | Failure |
|---|---|
| FPR exceeds 0.5% on evaluation dataset | ✗ Statistical failure |
| Detection drops below threshold τ after any **supported** transformation | ✗ Robustness failure |
| Payload corruption causes misattribution to wrong user | ✗ Legal failure |
| Embedding key `K` is recoverable from output images | ✗ Security failure |
| Signing private key is exposed | ✗ Cryptographic failure |

---

## 7. Performance Targets (Baseline)

All benchmarks measured on a **512×512 RGB image**.

| Target | Specification |
|---|---|
| Embedding time | **< 200 ms** |
| Detection time | **< 300 ms** |
| PSNR | **> 38 dB** (perceptual imperceptibility) |
| SSIM | **> 0.98** |

> If humans can visually detect the watermark, the embedding has failed.

---

## 8. Open API Contracts (Phase Deliverables)

The following function signatures constitute the engineering contract across phases:

| Function | Phase | Description |
|---|---|---|
| `embed(image, payload, key)` | Phase 1 | Embeds signed payload into image using key |
| `detect(image, key)` | Phase 1 | Returns detection score and decoded payload |
| `sign_payload(payload)` | Phase 2 | Ed25519 sign; returns full signed payload |
| `verify_signature(payload)` | Phase 2 | Validates signature against public key |
| `ecc_encode(bitstream)` | Phase 2 | Reed-Solomon encodes bitstream |
| `ecc_decode(bitstream)` | Phase 2 | Recovers bitstream with error correction |
| `attack_pipeline(image)` | Phase 3 | Applies all supported transforms for robustness testing |
| `multi_scale_embed(image, payload, key)` | Phase 4 | Embeds at multiple resolution scales |
| `forensic_report(image)` | Phase 5 | Full statistical report: confidence %, TPR, FPR, ROC |

---

## 9. Document Role

This specification serves as:

- 📐 **Engineering Contract** — All implementations must conform to Section 3, 4, 5
- 📚 **Research Reference** — Benchmark baseline for published results
- 📋 **Patent Groundwork** — Claims must be traceable to sections 5 and 6
- 🏛 **Internal Architecture Anchor** — Phase deliverables gate on this specification

---

*Mist Technical Specification v1.0 — End of Document*
