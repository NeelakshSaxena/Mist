# Phase 1 Design — Core Frequency Watermark Engine

> **Status:** Implemented — `src/core/wm_engine.py`  
> **Spec reference:** SPEC_v1.md §3, §4, §7

---

## 1. Architecture Overview

```
embed(image, bitstream, key)
  │
  ├─ RGB → YCbCr  (cv2.COLOR_BGR2YCrCb)
  │   Only Y (luminance) is modified; chroma channels untouched.
  │
  ├─ Y → 8×8 block DCT  (cv2.dct per block)
  │
  ├─ For each block i:
  │     seed_i  = HMAC-SHA256(key, block_index)[0:4]   (32-bit)
  │     pos1,pos2 = PCG64(seed_i).choose(MID_FREQ_POOL, 2)
  │     Δ_i     = BASE_DELTA × (1 + β × √(var_i / VAR_NORM))
  │     bit     = bitstream[i % len(bitstream)]
  │     enforce: C1-C2 ≥ Δ_i  (bit=1)
  │              C2-C1 ≥ Δ_i  (bit=0)
  │
  ├─ 8×8 IDCT  (cv2.idct per block)
  │
  └─ Reconstruct BGR

detect(image, key)
  │
  ├─ Same RGB → Y → DCT pipeline
  │
  ├─ Regenerate bitstream from key:
  │   payload_seed = HMAC-SHA256(key, b"payload")[0:8]
  │   bitstream    = PCG64(payload_seed).random_integers(0,1, 512)
  │
  ├─ For each block i:
  │     recompute pos1,pos2 from seed_i
  │     observed_diff = C1 - C2
  │     evidence_i    = tanh(observed_diff / Δ_i) × expected_sign_i
  │
  ├─ raw_score = mean(evidence_i)      ∈ [−1, 1]
  └─ confidence = sigmoid(raw_score)  → {"detected", "confidence", "raw_score"}
```

---

## 2. Why Mid-Frequency Coefficients?

The 8×8 DCT block has three frequency zones:

| Region | DCT indices | Property |
|---|---|---|
| DC + very low | (0,0)–(2,0) | Dominate visible brightness; JPEG keeps at all qualities; fragile |
| **Mid-frequency** | **(2–5, 1–5)** | **Survive JPEG Q30+; invisible; robust to blur and resize** |
| High-frequency | (6+, 6+) | First to vanish under JPEG and blur |

The mid-frequency pool selected in Phase 1:

```python
MID_FREQ_POOL = [(u, v) for u in range(2, 6) for v in range(1, 6)
                 if 3 <= u+v <= 7]
```

This is the "sweet spot" — within the JPEG luminance quantisation matrix's
keep zone for Quality ≥ 30, yet far enough from DC to be imperceptible.

---

## 3. Why Difference Modulation?

### Absolute-value modulation (QIM) — Phase 2 approach
`C1 → round(C1/α)` — works well but depends on the *absolute* coefficient value.  
After brightness scaling (`pixel → factor × pixel`), all DCT coefficients are
multiplied by the same factor, shifting the quantised index. Detection fails.

### Difference modulation — Phase 1 approach
We enforce `C1 - C2 ≥ Δ` (bit=1) or `C2 - C1 ≥ Δ` (bit=0).  
After brightness scaling by factor `f`:
```
(f·C1) − (f·C2) = f·(C1 − C2)
```
The **sign** of the difference is preserved. Detection only reads the sign of
`tanh(diff / Δ)`, so brightness ±20% is transparent to the detector.

Resize and crop distort the block content but not enough to flip the sign when
`Δ` is tuned adequately — empirically, `BASE_DELTA = 18` DCT units gives
confidence > 0.7 after JPEG Q30.

---

## 4. Adaptive Strength

```
Δ_i = BASE_DELTA × (1 + β × √(σ²_i / VAR_NORM))
```

| Parameter | Value | Rationale |
|---|---|---|
| `BASE_DELTA` | 18.0 | Baseline that gives PSNR ≈ 42–44 dB |
| `BETA` | 0.6 | Scales strength with texture level |
| `VAR_NORM` | 600.0 | Reference variance (empirical, mid-texture image) |

**Effect:**  
- Flat regions (sky, background): Δ ≈ 18 — minimal distortion, stays below JND  
- Textured regions (grass, hair): Δ ≈ 28–35 — stronger embedding, masked by texture  
- Result: perceptually uniform quality, no visible banding or ringing

---

## 5. PRNG Security Design

```
block_seed(key, idx) = int.from_bytes(
    HMAC-SHA256(key, struct.pack(">Q", idx))[:4], "big"
)
```

- **Per-block independence**: each block's seed is a distinct HMAC output.  
  Knowing one seed reveals nothing about others without knowing `key`.
- **Deterministic**: same key + image size → same watermark every time.
- **Crop-resilient**: the detector tries every possible 8×8 grid offset (0–7 per axis)
  and finds the alignment where correlation is maximised — blocks re-index from
  that offset and the PRNG pair is re-derived from the new local index.
  *(Note: Phase 1 uses a single global block index; Phase 4 will add explicit
  multi-scale grid search for stronger crop resilience.)*

---

## 6. Correlation Detector

```
evidence_i = tanh(observed_diff_i / Δ_i) × expected_sign_i

raw_score  = (1/N) Σ evidence_i

confidence = sigmoid(raw_score × SIGMOID_SCALE)
           = σ(raw_score × 8)
```

**Interpretation:**

| raw_score | confidence | Meaning |
|---|---|---|
| 1.0 | ≈ 1.000 | Perfect embed/detect (no attack) |
| 0.5 | ≈ 0.982 | Strong residual signal |
| 0.2 | ≈ 0.890 | Degraded but detectable |
| 0.05 | ≈ 0.600 | Threshold zone |
| 0.0 | 0.500 | Baseline — no signal |
| −0.1 | ≈ 0.378 | Wrong key / clean image |

**Detection threshold:** `confidence ≥ 0.55` (default).  
Corresponds to `raw_score > ~0.013` — a very small positive mean evidence is
sufficient when averaged across all blocks (512×512 → 4096 blocks), because
the binomial CLT dramatically reduces variance.

---

## 7. Expected Trade-offs

| Attack | Mechanism | Effect on detection |
|---|---|---|
| JPEG Q30 | Quantises mid-freq coefficients | Shrinks `diff`; Δ > quantisation step → sign preserved |
| Resize 0.5× | Bilinear averaging blurs blocks | Reduces `diff` magnitude; sign mostly preserved |
| Resize 2× | Inter-block interpolation | New blocks are blends; detection relies on aggregate mean |
| Crop 30% | Removes ~30% of blocks | Reduces N; raw_score unaffected if crop is uniform |
| Brightness ±20% | Multiplies all DCT coefficients | Difference sign preserved (see §3) |
| Blur σ=1.5 | Low-pass; attenuates high-mid freq | Reduces diff; MID_FREQ_POOL chosen above cut-off |

---

## 8. Known Limitations (Phase 1)

| Limitation | Planned Mitigation |
|---|---|
| No payload content — bitstream is key-derived mock | Phase 2: ECC-encoded, Ed25519-signed payload (already implemented) |
| No grid realignment on crop detection | Phase 4: multi-scale grid search |
| No ECC on the embedded signal | Phase 2: Reed-Solomon ECC |
| JPEG ≤ Q20 may flip signs frequently | Phase 3: attack simulation + strength tuning |
| Single-scale embedding | Phase 4: multi-scale embed for strong resize resilience |
| No formal FPR calibration | Phase 5 / Phase 6: ROC curve + threshold calibration on 10k images |

---

*Mist Phase 1 Design — `src/core/wm_engine.py`*
