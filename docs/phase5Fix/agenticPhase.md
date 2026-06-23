Perfect. You’re at the point where “tuning heuristics” is wasting time.
Mist Phase 5 now needs a **real synchronization subsystem**.

Your logs already proved:

- embedding robustness exists
- ECC works
- detection survives attacks
- synchronization collapses

So here’s the cleanest path forward:

---

# Phase 5 Rewrite Strategy

We split implementation into **5 autonomous engineering agents** with:

- scoped responsibilities
- verification gates
- measurable stop conditions
- rollback criteria
- performance budgets

This prevents another “100-second brute-force spaghetti loop” situation.

The papers strongly support this architecture: Fourier-Mellin / log-polar synchronization is the standard fix for RST distortion. ([ResearchGate][1])

---

# MASTER PLAN

```text
P5-V2
├── Agent 1: FFT + Log-Polar Sync Core
├── Agent 2: Affine Geometry Recovery
├── Agent 3: Synchronization Templates
├── Agent 4: Reliability + Confidence Calibration
├── Agent 5: Validation Harness + Auto-Profiling
```

---

# AGENT 1 — FFT + LOG-POLAR SYNCHRONIZATION (complete)

## Goal

Replace:

```python
canary brute-force candidate search
```

with:

```python
FFT magnitude
→ log-polar transform
→ phase correlation
→ rotation/scale estimate
```

This is directly derived from Fourier-Mellin synchronization systems. ([ResearchGate][1])

---

## Files

```text
src/core/geometry_sync.py
src/core/wm_engine_p5.py
```

---

## Implementation Prompt

Implement a new geometry synchronization module for Mist Phase 5 using FFT magnitude spectra, log-polar remapping, and phase correlation.

Requirements:

- Create geometry_sync.py
- Input: grayscale image
- Output:
  rotation_deg
  scale_factor
  confidence
  response_peak

Pipeline:

1. Apply Hann window
2. FFT2
3. Magnitude spectrum
4. Log scaling
5. warpPolar/log-polar remap
6. cv2.phaseCorrelate
7. Convert shifts into:
   - rotation angle
   - scale factor

Constraints:

- Must run under 250ms for 512×512
- Must avoid brute-force search
- Must support:
  rotation ±20°
  scale 0.5×–1.5×

Verification:

- Unit test against synthetic transforms
- Mean angle error < 0.5°
- Mean scale error < 0.03

Stop Conditions:

- PASS if:

  > 90% successful recovery
  > runtime <250ms

- FAIL if:
  phase correlation confidence <0.15 on clean images
  geometry estimate diverges >2°
  Rollback:
- Preserve old P5 path behind feature flag:
  USE_LEGACY_GEOMETRY

---

# AGENT 2 — AFFINE RECOVERY + RESAMPLING (complete)

## Goal

Once geometry estimated:

```text
inverse affine transform
→ normalized image
→ existing P4 detector
```

Your current system extracts directly from distorted tiles.
That is the core architectural mistake.

---

## Papers Supporting This

Affine normalization after synchronization is standard robust watermarking architecture. ([Utah State University][2])

---

## Implementation Prompt

Implement affine geometry correction for Mist Phase 5.

Requirements:

- Add:
  correct_geometry()
- Inputs:
  image
  rotation_deg
  scale_factor
- Outputs:
  corrected_image

Pipeline:

1. Compute inverse affine matrix
2. Rotate around image center
3. Scale back to canonical size
4. Bicubic interpolation
5. Optional edge padding

Requirements:

- Preserve watermark energy
- Avoid ringing artifacts
- Deterministic output

Verification:

- Corrected images must align with original grid
- P4 detection after correction must recover:

  > 90% shard integrity

- PSNR loss after correction:
  <1.5dB

Stop Conditions:

- PASS:
  512 rotation tests decode successfully
- FAIL:
  corrected image dimensions drift
  tile phase mismatch occurs

---

# AGENT 3 — SYNCHRONIZATION TEMPLATE LAYER

## Goal

Separate:

```text
synchronization
≠
payload
```

Right now Mist uses payload bits for sync.

That is fragile.

Research consistently uses synchronization templates. ([ResearchGate][3])

---

## Recommended Design

Embed:

- weak radial Fourier peaks
- low-frequency pseudo-random ring
- repeated pilot markers

NOT payload bits.

---

## Implementation Prompt

Implement synchronization templates for Mist.

Requirements:

- Add synchronization layer separate from payload
- Embed:
  radial pilot peaks
  pseudo-random low-frequency ring
- Strength:
  weakly perceptual
- Embed before payload stage

Detection:

1. FFT magnitude
2. Detect template peaks
3. Estimate:
   rotation
   scale
   translation

Requirements:

- Must survive:
  JPEG 50
  rotation ±15°
  scaling 0.6×–1.5×

Verification:

- Pilot recovery rate >95%
- False peak rate <5%

Stop Conditions:

- PASS:
  synchronization estimates stable across attacks
- FAIL:
  visible Fourier artifacts
  template dominates payload energy

---

# AGENT 4 — RELIABILITY + CONFIDENCE CALIBRATION

## Goal

Fix:

```text
ROC AUC = 0.225
```

That’s catastrophic calibration.

Your current confidence engine is overfitting brute-force geometry noise.

---

## Correct Direction

Confidence should prioritize:

1. signature verification
2. RS decode integrity
3. shard consistency
4. geometry confidence
5. correlation

in that order.

---

## Supporting Research

Robust systems separate:

- synchronization certainty
- payload certainty
- authentication certainty

([ResearchGate][4])

---

## Implementation Prompt

Rewrite Mist forensic confidence calibration.

Requirements:

- Remove heavy dependence on:
  canary score
  brute-force geometry count
- Add weighted scoring:

conf =
0.45 _ signature_verified +
0.25 _ rs*decode_success +
0.15 * shard*consistency +
0.10 * geometry_confidence +
0.05 \* correlation

Outputs:

- calibrated confidence
- forensic likelihood
- confidence breakdown

Verification:

- ROC AUC >0.90
- Clean images:
  confidence <0.15
- Valid watermark:
  confidence >0.95

Stop Conditions:

- PASS:
  false positive rate <1%
- FAIL:
  clean images exceed confidence 0.3

---

# AGENT 5 — VALIDATION + PROFILING HARNESS

## Goal

You need:

- deterministic benchmarking
- runtime profiling
- regression prevention

because right now Phase 5 regresses silently.

---

## Implementation Prompt

Create Phase 5 validation harness with automatic profiling.

Requirements:

- Benchmark:
  geometry estimation
  affine correction
  P4 extraction
  ECC reconstruction

Metrics:

- runtime per stage
- shard accuracy
- geometry error
- ROC AUC
- FP/FN rates

Generate:

- CSV report
- timing histogram
- attack heatmap

Attack suite:

- rotation ±20°
- scaling 0.5×–1.5×
- crop 10–40%
- JPEG 30–95
- combined attacks

Stop Conditions:

- PASS:
  total runtime <15s
  verification >90%
- FAIL:
  any stage exceeds runtime budget
  geometry estimate diverges

---

# CRITICAL ARCHITECTURAL CHANGE

## REMOVE THIS ENTIRE IDEA

```python
for angle in angles:
    for scale in scales:
         brute_force()
```

That architecture does not scale.

The papers repeatedly identify brute-force synchronization as the failure point under geometric distortion. ([ResearchGate][4])

---

# NEW TARGET PIPELINE

## FINAL P5-V2 FLOW

```text
Image
 → FFT sync
 → log-polar registration
 → affine correction
 → synchronization template refinement
 → P4 extraction
 → shard voting
 → RS decode
 → signature verification
 → calibrated forensic confidence
```

---

# EXPECTED RESULTS

If implemented properly:

| Metric              | Current  | Expected |
| ------------------- | -------- | -------- |
| Rotation verify     | 0%       | 85–95%   |
| Runtime             | 100–300s | 5–15s    |
| Candidate count     | 185      | 1–3      |
| ROC AUC             | 0.225    | >0.9     |
| False geometry lock | Massive  | Rare     |

---

# Most Important Insight

Your detector is already robust enough.

The logs prove it.

Mist is not failing because:

- DCT embedding weak
- ECC weak
- shard redundancy weak

It is failing because:

# synchronization is not mathematically invariant.

That’s exactly what Fourier-Mellin + log-polar registration was invented to solve. ([ResearchGate][1])

[1]: https://www.researchgate.net/publication/3308555_RST-Invariant_Digital_Image_Watermarking_based_on_Log-Polar_Mapping_and_Phase_Correlation?utm_source=chatgpt.com "RST-Invariant Digital Image Watermarking based on Log ..."
[2]: https://www.usu.edu/cs/people/XiaojunQi/Promotion/JECS.RobustWM.08.pdf?utm_source=chatgpt.com "A Robust DCT-Based Digital Watermarking Scheme Using ..."
[3]: https://www.researchgate.net/figure/Schematic-view-of-general-template-based-synchronization-Usually-the-template-or-pilot_fig4_3338854?utm_source=chatgpt.com "Schematic view of general template-based synchronization ..."
[4]: https://www.researchgate.net/publication/3338854_Geometric_Attacks_on_Image_Watermarking_Systems?utm_source=chatgpt.com "Geometric Attacks on Image Watermarking Systems"
