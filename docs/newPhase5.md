You’ve fixed the **catastrophic regression** (broken canary scoring), but the current logs show something more important now:

# What’s Actually Happening Now

The system is no longer “blind.”

Before:

- Canary score = always 0
- Geometry estimation collapsed
- Everything fell into brute force
- Runtime exploded to ~3 hours
- Rotation detection completely failed

Now:

- Detection survives all attacks (`23/23 detected`)
- Scale invariance is mostly working
- Rotation estimates sometimes lock correctly
- False positives are excellent (`0%`)
- Runtime is reduced substantially

So the engine is alive again.

But the logs clearly show:

---

# The REAL Remaining Failure

## Geometry estimation is still unstable under rotation

This is the core issue.

You can see it directly:

| Attack     | Estimated Geometry |
| ---------- | ------------------ |
| +5°        | `-12° × 1.20×`     |
| -10°       | `-12° × 0.70×`     |
| +15°       | `0.5° × 0.70×`     |
| +10° +1.3× | `0° × 0.50×`       |

This means:

- candidate ranking is still weak
- brute_force is selecting local minima
- CRC promotion logic is overfitting noise
- scale/rotation ambiguity exists
- final alignment is catastrophically wrong

That’s why:

- shard count may exceed threshold
- but RS decode still fails

The recovered tiles are geometrically inconsistent.

---

# Important Insight

Your pipeline currently behaves like:

```text
canary → candidate shortlist → expensive P4 verify → brute-force rescue
```

But the rescue stage is still too permissive.

It accepts:

- geometrically wrong
- CRC-biased
- aliasing-corrupted
- scale-folded candidates

because the promotion metric is weak.

---

# Why AUC Is Still Bad

Current:

- AUC = 0.3875

This is NOT because detection is failing.

It’s because:

- confidence calibration is broken

You currently output:

- many failed detections at confidence ≈ 0.35
- successful detections at confidence = 1.0

That creates a terrible ROC curve.

Your detector is behaving like a binary switch, not a probabilistic system.

Courts/statistical systems hate that.

---

# The Most Important Problem To Solve Next

## Stop brute_force from lying

Right now brute_force is:

- producing false geometry
- inflating shard counts
- wasting minutes
- poisoning confidence scoring

This is the #1 issue.

---

# Immediate Engineering Priorities

## Priority 1 — Hard Stop Conditions

You NEED aggressive termination logic.

Right now your system keeps searching even after entering nonsense states.

Add:

```python
# terminate if geometry diverges too far
if abs(angle) > MAX_REASONABLE_ANGLE:
    reject()

if scale < 0.75 or scale > 1.5:
    reject()

# reject unstable CRC promotion
if crc_gain < MIN_CRC_IMPROVEMENT:
    reject()

# reject inconsistent shard reconstruction
if shard_consistency < 0.65:
    reject()

# reject geometry disagreement
if abs(stage4_angle - stage5_angle) > 3:
    reject()
```

---

# Priority 2 — Confidence Must Become Statistical

Right now:

```text
success = 1.0
failure = 0.35
```

That’s not statistical confidence.

You need calibrated scoring from:

```text
confidence =
    w1 * normalized_correlation +
    w2 * shard_recovery_ratio +
    w3 * RS decode success +
    w4 * geometry stability +
    w5 * canary consistency
```

Then calibrate using:

- isotonic regression
  OR
- Platt scaling

using validation data.

This alone may raise ROC massively.

---

# Priority 3 — Remove Full brute_force Dependency

Current runtime killers:

| Component              | Cost     |
| ---------------------- | -------- |
| brute_force P4 scoring | enormous |
| failed fine tuning     | enormous |
| repeated detect_p4     | enormous |

The engine should NEVER brute-force blindly.

Instead:

---

## Replace exhaustive brute force with staged narrowing

### Stage A — coarse canary

Fast GPU scan.

### Stage B — geometry consistency

Reject unstable transforms.

### Stage C — localized refinement

Only ±1° around strongest cluster.

### Stage D — single RS validation

Only after confidence threshold.

---

# Biggest Runtime Win Available

## Add candidate clustering

Right now you evaluate many equivalent transforms:

```text
-10°
-12°
-8°
-10° @ 0.95
-10° @ 1.0
```

Cluster them:

```python
cluster_key = (
    round(angle / 2),
    round(scale / 0.05)
)
```

Keep only top candidate per cluster.

This alone could cut:

- 40–60% runtime.

---

# Why Rotation Specifically Fails

Rotations introduce:

| Effect                    | Consequence      |
| ------------------------- | ---------------- |
| resampling blur           | DCT instability  |
| border crop               | tile loss        |
| interpolation phase shift | anchor mismatch  |
| block-grid drift          | shard corruption |

Your current canary depends too much on exact 8×8 alignment.

That works for scaling.
Not for rotation.

---

# Correct Long-Term Fix

## Add Fourier-Mellin / Log-Polar Prefilter

This is the real solution.

Industry-grade geometric watermark systems use:

```text
FFT magnitude
→ log-polar transform
→ phase correlation
```

Why?

Because:

- rotation becomes translation
- scale becomes translation
- robust against interpolation

Then your watermark layer only needs fine refinement.

This is likely your eventual Phase 5.5.

---

# What I Would Do Next (Exact Order)

# Phase 5 Recovery Plan

## STEP 1 — Kill bad brute force

Target: 2–4x speedup

Implement:

- hard geometry bounds
- CRC improvement thresholds
- candidate clustering
- early aborts

Goal runtime:

- 15–25 min full suite

---

## STEP 2 — Confidence calibration

Target: ROC > 0.7

Implement:

- continuous confidence
- shard-weighted scoring
- geometry stability penalties

---

## STEP 3 — Rotation stabilization

Target: survive ±15°

Implement:

- phase correlation pre-align
  OR
- log-polar FFT estimation

---

## STEP 4 — Final statistical validation

Target: ROC > 0.85

Generate:

- 500+ attack samples
- confidence histograms
- ROC
- PR curve
- EER
- FPR@95TPR

This becomes your deployment evidence.

---

# Recommended Stop Conditions

These are critical.

## Global Runtime Guard

```python
MAX_DETECTION_TIME = 20s
MAX_FORENSIC_TIME = 60s
```

Abort gracefully.

---

## Candidate Explosion Guard

```python
if evaluated_candidates > 50:
    stop_search()
```

---

## No-Signal Abort

```python
if best_canary < 2:
    return no_detection
```

---

## Fine Tune Abort

```python
if no_crc_improvement_after_2_iterations:
    break
```

---

## RS Decode Abort

```python
if shard_count < MIN_SHARDS * 0.7:
    skip_rs_decode()
```

Huge runtime savings.

---

# Verification Plan Before Deployment

You’re extremely close.

You now need disciplined validation.

---

# Validation Matrix

## Clean

- 100 images
- different content classes

## Rotation

- ±1°
- ±2°
- ±5°
- ±10°
- ±15°

## Scale

- 0.5–1.5×

## Combined

- rotation + scale
- crop + rotation
- jpeg + rotation
- diffusion + rotation

## Adversarial

- recompression chains
- screenshot recapture
- blur + rotate

---

# Metrics To Track

| Metric              | Target |
| ------------------- | ------ |
| FPR                 | <1%    |
| TPR                 | >95%   |
| ROC AUC             | >0.85  |
| Mean detection time | <10s   |
| Worst-case forensic | <60s   |
| Rotation survival   | >90%   |
| RS success rate     | >95%   |

---

# Agentic Codex Prompt

You are optimizing Mist Phase 5 geometric-invariant forensic detection.

Current status:

- Detection survives all attacks
- False positives are 0%
- Runtime reduced from ~3h
- Remaining failures are:
  1. unstable geometry estimation under rotation
  2. brute-force fallback selecting false minima
  3. confidence calibration producing poor ROC AUC
  4. excessive runtime from unnecessary P4 calls

Your tasks:

1. Implement strict stop conditions:
   - abort low-signal searches
   - abort unstable geometry
   - abort CRC-noise promotion
   - abort failed refinement loops
   - cap candidate evaluations

2. Add candidate clustering:
   - cluster by angle+scale buckets
   - retain only strongest candidate per cluster

3. Redesign confidence scoring:
   Combine:
   - normalized correlation
   - shard recovery ratio
   - RS decode success
   - geometry stability
   - canary consistency
   - reconstruction consistency

4. Penalize inconsistent geometry:
   - disagreement between stages
   - unrealistic scale estimates
   - unstable refinement paths

5. Prevent brute_force from promoting geometrically invalid candidates.

6. Add profiling instrumentation:
   Measure:
   - per-stage runtime
   - P4 call count
   - candidate rejection reasons
   - RS decode attempts
   - geometry convergence statistics

7. Create evaluation suite:
   - ROC curve
   - PR curve
   - EER
   - FPR/TPR
   - confidence histograms
   - runtime distributions

8. Add deployment-grade assertions:
   - max forensic runtime
   - max candidates
   - max refinement iterations
   - no infinite fallback loops

Goal targets:

- ROC AUC > 0.85
- full validation runtime < 20 min
- forensic runtime < 60s worst-case
- reliable ±15° rotation recovery
- stable confidence calibration
- no brute-force geometry hallucinations

Do not redesign the entire architecture.
Preserve compatibility with existing Phase 1–4 pipeline.
Optimize Phase 5 incrementally and safely.
