# Phase 5 Optimization — Implementation Plan

## Files to Modify
| File | Changes |
|------|---------|
| `src/core/wm_engine_p5.py` | Stop conditions, candidate clustering, geometry gates, confidence scoring, profiling hooks, assertions |
| `src/core/forensic.py` | Updated confidence calibration using multi-signal scoring |
| `scripts/validate_phase5.py` | Updated validation with profiling output |

## New Files
| File | Purpose |
|------|---------|
| `src/core/p5_profiler.py` | Profiling instrumentation module |
| `scripts/evaluate_phase5.py` | Full evaluation suite (ROC, PR, EER, histograms, runtime) |

## Implementation Order

### 1. `src/core/p5_profiler.py` — Profiling infrastructure (new file)
- `P5Profile` dataclass with per-stage timings, P4 call count, candidate rejections, RS attempts, geometry stats
- Context manager for stage timing
- Thread-local storage for current profile

### 2. `wm_engine_p5.py` — Core changes (single file, incremental)
- **Constants**: Add stop-condition thresholds (MAX_REASONABLE_ANGLE, SCALE_BOUNDS, MAX_CANDIDATES, MAX_REFINEMENT_ITER, etc.)
- **`_cluster_candidates()`**: New function — bucket by `(round(angle/2), round(scale/0.05))`, keep best per cluster
- **`_compute_confidence()`**: New multi-signal scoring function combining 6 factors
- **`_validate_geometry()`**: New function — penalize inconsistent geometry
- **`estimate_geometry()`**: Inject stop conditions, clustering, profiling, geometry gates
- **`detect_p5()`**: Updated confidence scoring, deployment assertions, runtime guard

### 3. `forensic.py` — Confidence calibration update
- Use multi-signal confidence from `_compute_confidence()` instead of binary switch

### 4. `scripts/evaluate_phase5.py` — Evaluation suite (new file)
- ROC curve generation with matplotlib
- PR curve + EER computation  
- Confidence histograms (positive vs negative)
- Runtime distribution analysis
- Per-attack-type breakdown

### 5. `scripts/validate_phase5.py` — Updated validation
- Enable profiling output
- Print profiling summary per test case
