# 🌫 Technical Roadmap

## 🌫 Phase 0 – Threat Model & Formal Specification (1 Week)

### 🎯 Objective
Define exactly what Mist protects against.

### ✅ Supported Edits

Mist must remain detectable after:

- JPEG compression ≥ Quality 30  
- Resize between 0.5× – 2×  
- Crop ≤ 30%  
- Screenshot simulation  
- Brightness/contrast adjustment ±20%  

### 📊 Detection Standards

- Defined detection confidence threshold  
- Acceptable false positive rate: **< 0.5% (ideal target)**  

### 📦 Payload Structure

Each watermark payload must include:

- `user_id`
- `timestamp`
- `image_id`
- `model_version`
- `digital_signature`

### 📄 Deliverable

**Mist Technical Specification v1.0**

> ⚠️ Skipping this phase results in a vague and legally weak system.

---

## 🌫 Phase 1 – Core Frequency Watermark Engine (2–3 Weeks)

### 🎯 Objective

Implement deterministic DCT-based embedding.

### 🏗 Architecture Core

1. Convert RGB → YCbCr  
2. Extract Y (luminance) channel  
3. Divide into 8×8 blocks  
4. Apply DCT  
5. Select mid-frequency coefficient pairs  
6. Embed pseudo-random signal  
7. Apply inverse DCT  
8. Reconstruct image  

### ⚙️ Technical Details

- Pseudo-random generator seeded with secret key  
- Modify coefficient **difference**, not absolute values  
- Adaptive strength scaling based on block variance  

### 📦 Deliverables

- `embed(image, payload, key)`  
- `detect(image, key)`

### 📊 Validation Targets

- PSNR > 40 dB  
- SSIM > 0.98  

> If humans can see it, you failed.

---

## 🌫 Phase 2 – Payload Encoding + Cryptographic Binding (2 Weeks)

### 🎯 Objective

Secure and verifiable watermark payload.

### 🧩 Tasks

- Define payload bit structure (e.g., 256 bits)  
- Apply:
  - SHA-256 hashing  
  - ECC (Reed-Solomon) error correction  
  - Digital signature (RSA or ECC)  
- Compress final bitstream  
- Map bits → DCT modulation  

### 📦 Deliverables

- `sign_payload()`  
- `verify_signature()`  
- `ecc_encode()`  
- `ecc_decode()`

### 📊 Validation

- Tampered payload fails signature verification  
- Partial damage still recovers payload  

> At this point, Mist becomes traceable and court-ready.

---

## 🌫 Phase 3 – Screenshot & Light-Attack Simulation Engine (2 Weeks)

### 🎯 Objective

Train Mist against real-world transformations.

### 🔥 Attacks to Implement

**Compression**
- JPEG quality 30–100  

**Resizing**
- Random scaling  
- Downscale → Upscale  

**Filtering**
- Mild Gaussian blur  

**Cropping**
- Random 10–30%  

**Color & Light**
- Brightness/contrast jitter  

**Screenshot Simulation**
- Resize  
- Gamma shift  
- Quantization  

### 📦 Deliverable

- `attack_pipeline(image)`

### 📊 Validation

Test watermark recovery after each attack.

> This phase separates toy systems from production-grade systems.

---

## 🌫 Phase 4 – Multi-Scale Embedding (2 Weeks)

### 🎯 Objective

Embed watermark at multiple resolutions.

### 🧠 Approach

1. Embed at original resolution  
2. Downscale image to 50%  
3. Embed second scaled watermark  
4. Upscale back  

Watermark exists at:

- Fine scale  
- Coarse scale  

### 📦 Deliverable

- `multi_scale_embed()`

### 📊 Validation

Test robustness under resizing 0.5×–2×.

---

## 🌫 Phase 5 – Statistical Detection Engine (Advanced Layer) (3 Weeks)

### 🎯 Objective

Add confidence scoring and statistical validation.

### 📊 Detector Outputs

- Payload recovered (Yes/No)  
- Correlation strength  
- Confidence %  
- Tampering likelihood  

### 🧮 Implement

- Correlation thresholding  
- False positive measurement  
- ROC curve generation  

### 📦 Deliverable

- `forensic_report(image)`

> Courts require statistical confidence, not binary answers.

---

## 🌫 Phase 6 – Benchmark & Legal Documentation (3–4 Weeks)

### 🎯 Objective

Make Mist legally defensible.

### 🧪 Benchmark Plan

- Collect 10,000 test images  
- Watermark 5,000  
- Leave 5,000 clean  
- Apply randomized attack combinations  

### 📊 Measure

- False positives  
- False negatives  
- ROC curves  
- Confidence distributions  

### 📄 Generate Reports

- Mist Validation Report  
- False Positive Analysis  
- Detection Reliability Report  

> This transforms Mist into forensic-grade technology.

---

## 🌫 Phase 7 – API & Logging Infrastructure (Optional)

### 🎯 Objective

Make Mist production-ready.

### 🧩 Components

- Embed API  
- Detection API  
- Secure log server  
- Timestamp logging  
- Hash storage  
- Key management system  

---

# 🚀 Final Outcome

If all phases are executed, Mist becomes:

- Robust  
- Traceable  
- Statistically validated  
- Legally defensible  
- Product-ready  

---

# 🧪 Phase H Session Log – 2026-03-02

## Status: Debugging Robustness Drop

### 📌 Observation

- **Phase G Result:** 100% JPEG-50 bit accuracy (1D redundancy)  
- **Phase H Result (Initial):** ~50% JPEG-50 accuracy (2D tiling)  

Discrepancy:
A separate diagnostic script (`debug_detect.py`) achieved 100% JPEG-50 accuracy using the same 2D tiling logic that failed in the main test suite.

---

## 🔍 Root Cause Analysis

### 1️⃣ PRNG Desynchronization

The test script was calling:

```python
np.random.seed(42)
```
before every image to generate payloads.

This silently reset the global random state used by `get_tile_mapping()`, causing the detector to search incorrect spatial coordinates.

### 2️⃣ Math Precision Mismatch

- `test_phase_e.py` used `np.round()` during Y-channel reconstruction.
- `debug_detect.py` used implicit truncation.
- This introduced 1-bit quantization noise, occasionally pushing QIM coefficients into the wrong bin centers.

---

## 🛠 Fixes Applied

### ✅ PRNG Sync
- Removed global random seeding from test loop.
- Switched to fixed dummy payload for stable tile mapping.

### ✅ Math Alignment
- Replaced `np.round()` with `np.clip()` + implicit casting.
- Matched reconstruction path to optimized diagnostic logic.

### ✅ Detector Optimization
- Removed early-exit thresholds in spatial grid search.
- Ensured evaluation of all 64 possible pixel shifts.

---

## 📊 Current Metrics (After Fixes)

- **Baseline (No Attack):** 100%
- **Brightness (+15%):** 100%
- **Resize (0.75x):** 100%
- **JPEG (Q50):** ~52%
- **Crop (20%):** ~48%

---

## ⚠️ Observed Anomalies

- Individual image tests in `debug_detect.py` consistently yield 100% recovery.
- Same images sometimes fail inside `test_phase_e.py` batch loop.

This strongly suggests:
- Remaining state leak
- Hidden global mutation
- Environmental difference between single-run and batch contexts

---

## 🔄 Current Action

- Re-running `test_phase_e.py` on 100 images.

### 🎯 Goals
- Restore JPEG-50 to 100%
- Improve `Crop-20` robustness
- Identify remaining state inconsistencies

**Mist Status:** Active Debugging – Robustness Regression Under Investigation

---
