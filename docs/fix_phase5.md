# Phase 5 Detection — Fix & Optimization Log

## Problem Statement

The Phase 5 geometric-invariant detection engine was:
1. **Taking ~3 hours** for a full validation run
2. **Failing all rotation tests** (0/8 rotation cases passing)
3. **ROC AUC at 0.1875** (target > 0.85)

## Root Cause Analysis

### Critical Bug: Broken Canary Scoring (introduced in previous session)

The `_canary_score_Y()` function in `wm_engine_p5.py` and `_gpu_canary_score_from_Y()` in
`gpu_geometry.py` were rewritten with a sliding-window approach that was **fundamentally incorrect**:

1. **Wrong input to `_extract_tile_bits()`**: The function expects a **64×64 macro-tile** (MT_SIZE=64)
   and returns **64 bits** (one per 8×8 block). The broken code passed a **256×256 region**, which
   returned `32×32 = 1024 bits`.

2. **Wrong bit interpretation**: The 1024 bits were reshaped to `(32, 32)` and sliced as if each row
   represented a spatial tile's anchor. In reality, each bit represents a single DCT coefficient
   comparison from one 8×8 block — there is no tile structure in a flat block grid.

3. **Result**: Canary score returned **0 for every candidate**, causing:
   - All candidates to appear equally bad to the promotion stage
   - Every search falling through to the "brute_force" fallback
   - Brute_force attempting hundreds of full P4 scoring calls (~1.5s each)
   - Wrong geometry estimates propagating to verification → all rotation tests fail

### Performance Issue: Excessive P4 Calls

Even with correct canary scoring, the pipeline was making too many `_score_candidate()` calls:
- **Stage 4**: Promoted top-8 candidates (8 × ~1.5s = 12s)
- **Stage 5**: Extended search promoted 12 more (12 × ~1.5s = 18s)  
- **Fine-tune**: 16+ P4 calls for angle/scale sweep (16 × ~1.5s = 24s)
- **detect_p5 perturbation**: 16 full `detect_p4()` calls (16 × ~1.5s = 24s each)
- **Candidate generation**: 500+ candidates (21 scale × 41 rotation × combined = ~500)

## Fixes Applied

### 1. Canary Scoring Restored (`wm_engine_p5.py`, `gpu_geometry.py`)

Reverted both `_canary_score_Y` and `_gpu_canary_score_from_Y` to the correct **tile-based**
algorithm:
- Checks 5 individual 64×64 macro-tiles near the image center
- 4 sub-block pixel offsets (0,0), (0,4), (4,0), (4,4) for alignment
- Returns count of tiles whose first 8 bits match the expected anchor (≥62.5%)
- Correctly aligned images score 3-5; misaligned score 0-1

### 2. Candidate Generation Trimmed (`estimate_geometry`)

| Parameter | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Rotation-only angles | 41 (1° steps) | 25 (2° steps + key small angles) | 39% |
| Combined angle step | 2° | 4° | 50% |
| Total candidates | ~500 | ~330 | 34% |
| Top-N promotions | 8 | 5 | 38% |
| Extended search | 12 candidates always | 8 only if canary ≥ 2 | conditional |

### 3. Fine-Tune Streamlined (`_fine_tune`)

| Parameter | Before | After |
|-----------|--------|-------|
| Angle candidates | 4 (±1°, 0.5° steps) | 2 (±0.5° only) |
| Scale candidates | 12 (±4%, 2% steps, ±1px) | 2 (±2% only) |
| Max P4 calls | 16 | 4 |
| Early exit on K_SHARDS | No | Yes |

### 4. Perturbation Search Reduced (`detect_p5`)

| Parameter | Before | After |
|-----------|--------|-------|
| Perturbation grid | 4×4 = 16 | 4 (±0.5° × ±0.02) |
| Max P4 calls saved | 12 | per failed detection |

### 5. Stage 6 Gated on Signal

The fine-tune stage now only runs when `best_crc > id_crc` (i.e., geometric correction actually
improved things). Previously, it would fine-tune around noise even when no signal was found.

## Expected Impact

### Runtime
- **Before**: ~3 hours (180 minutes)
- **Expected**: ~15-40 minutes (depends on canary hit rate)
- **Key savings**: Fewer P4 calls per candidate (16→4 fine-tune, 16→4 perturbation), fewer
  candidates (~500→~330), conditional extended search, early exits

### Detection Quality
- **Before**: ROC AUC 0.1875, 0/8 rotation tests passing
- **Expected**: Restored to pre-break performance (canary was working before the broken change)
- **ROC AUC**: Should return to ~0.67+ baseline; further improvements need algorithmic changes

## Remaining Work

1. **ROC AUC > 0.85**: The current canary+promotion pipeline finds the right geometry for
   moderate rotations (±2°, ±5°) but struggles with ±10° and ±15°. This needs:
   - Finer combined rotation+scale grid for large angles
   - Multi-phase macro-tile search during canary scoring
   - Potentially a log-polar or Fourier-Mellin correlation pre-filter

2. **1024×1024 performance**: Larger images generate more candidates and each P4 call is slower.
   Consider downsampling the image during canary scoring for speed.

3. **GPU parallelism**: The CUDA stream concurrency in `gpu_batch_canary` could benefit from true
   asynchronous kernel launches rather than serial `stream.synchronize()` between transform and score.
