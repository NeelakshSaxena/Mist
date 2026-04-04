# Phase 4 Design — Spatial Attack Resistance Engine

> **Status:** Design Complete — Pending Implementation  
> **Spec reference:** SPEC_v1.md §3 (Cropping), §4, §7  
> **Depends on:** Phase 1 (`wm_engine.py`), Phase 2 (`crypto.py`, `ecc.py`, `payload.py`), Phase 3 (`wm_engine_p3.py`)  
> **Deliverable:** `src/core/wm_engine_p4.py`, updated `src/core/mist.py`

---

## 1. Problem Statement

Phases 1–3 embed watermarks that repeat every `TILE_P3 = 8` blocks (64×64 px at 8×8 block size). This provides basic crop resilience — the detector can re-align to the tiling grid. However, the payload is encoded **exactly once** across the full image: block `i` carries bit `i % 1184`. If an adversary crops 40%+ of the image, or occludes large regions with memes/overlays, enough payload bits are destroyed that Reed-Solomon cannot recover the full 1184-bit codeword.

**Phase 4 solves this by treating the watermark as a distributed, redundant signal.**

---

## 2. Threat Model (Phase 4 Specific)

### 2.1 Attacks Defended

| Attack | Description | Target Survival |
|---|---|---|
| **Heavy Cropping** | Remove 30–70% of image area | Payload recovery from remaining 30–70% |
| **Screenshot** | Resolution loss + re-encoding + gamma shift | Handled by Phase 3 + spatial redundancy |
| **Meme / Overlay** | Text, stickers, graphics covering 20–50% of image | Unaffected tiles carry full payload |
| **Recomposition** | Cut-and-paste into collage | Any fragment ≥ minimum tile count → detection |
| **Region Masking** | Blacking out / blurring specific regions | Surviving tiles carry independent payload copies |

### 2.2 Survival Targets

| Metric | Target |
|---|---|
| Minimum image fragment for detection | **128×128 px** (16 × 8×8 blocks) |
| Payload recovery at 50% area loss | **≥ 95% probability** |
| Payload recovery at 70% area loss | **≥ 80% probability** |
| Detection-only (presence) at 80% area loss | **≥ 90% probability** |
| PSNR degradation vs. Phase 3 alone | **< 1.5 dB** |

---

## 3. Architecture Overview

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING PIPELINE                            │
│                                                                 │
│  payload (JSON)                                                 │
│      │                                                          │
│      ▼                                                          │
│  SHA-256 → Ed25519 sign → payload_core ∥ signature (88 bytes)   │
│      │                                                          │
│      ▼                                                          │
│  Reed-Solomon ECC encode → 148 bytes = 1184 bits                │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────────┐               │
│  │  PHASE 4: SPATIAL REDUNDANCY LAYER           │               │
│  │                                              │               │
│  │  1. Divide image into macro-tiles (T×T px)   │               │
│  │  2. Within each macro-tile:                  │               │
│  │     a. Embed SYNC ANCHOR (4-bit marker)      │               │
│  │     b. Embed TILE INDEX (log2(N) bits)       │               │
│  │     c. Embed ECC SHARD (RS-outer fragment)   │               │
│  │  3. Apply Phase 3 multi-scale DCT embedding  │               │
│  │  4. Add Phase 3 sinusoidal harmonic          │               │
│  └──────────────────────────────────────────────┘               │
│      │                                                          │
│      ▼                                                          │
│  Watermarked image                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                            │
│                                                                 │
│  Image fragment (arbitrary crop / partial)                      │
│      │                                                          │
│      ▼                                                          │
│  1. Phase 3 presence detection (multi-scale + harmonic)         │
│      │                                                          │
│      ▼                                                          │
│  2. SPATIAL SYNC: sliding-window scan for anchor patterns       │
│      │ → recover macro-tile boundaries                          │
│      │ → recover tile indices                                   │
│      ▼                                                          │
│  3. Extract ECC shards from each located macro-tile             │
│      │                                                          │
│      ▼                                                          │
│  4. OUTER RS DECODE: reconstruct full 1184-bit payload          │
│      │  from any K-of-N shards (fountain-like)                  │
│      ▼                                                          │
│  5. Inner RS decode (existing Phase 2 ECC)                      │
│      │                                                          │
│      ▼                                                          │
│  6. Parse payload → Ed25519 verify → ownership result           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Diagram (Encoding)

```
                  1184 bits (Phase 2 ECC payload)
                           │
                           ▼
              ┌─────────── OUTER RS ENCODER ──────────┐
              │  Split 1184 bits into K data shards    │
              │  Generate N-K parity shards            │
              │  Total: N shards × S bits each         │
              └────────────────────────────────────────┘
                           │
                    N shards (each S bits)
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
     ┌──────────┐   ┌──────────┐   ┌──────────┐
     │ Tile 0   │   │ Tile 1   │   │ Tile N-1 │
     │          │   │          │   │          │
     │ [ANCHOR] │   │ [ANCHOR] │   │ [ANCHOR] │
     │ [INDEX]  │   │ [INDEX]  │   │ [INDEX]  │
     │ [SHARD]  │   │ [SHARD]  │   │ [SHARD]  │
     └──────────┘   └──────────┘   └──────────┘
           │               │               │
           ▼               ▼               ▼
     Phase 3 DCT embedding per macro-tile
```

---

## 4. Tile-Based Embedding

### 4.1 Macro-Tile Geometry

The image is divided into a regular grid of **macro-tiles**. Each macro-tile is a contiguous rectangular region that carries an independent, self-identifying shard of the payload.

```
┌──────────────────────────────────────┐
│  MT(0,0)  │  MT(0,1)  │  MT(0,2)    │
│           │           │             │
├───────────┼───────────┼─────────────┤
│  MT(1,0)  │  MT(1,1)  │  MT(1,2)    │
│           │           │             │
├───────────┼───────────┼─────────────┤
│  MT(2,0)  │  MT(2,1)  │  MT(2,2)    │
│           │           │             │
└──────────────────────────────────────┘
```

#### Parameters

| Parameter | Symbol | Value | Rationale |
|---|---|---|---|
| Macro-tile size | `MT_SIZE` | 64 px | 8 × 8×8-blocks; sufficient blocks per tile for reliable shard embedding |
| Minimum blocks per tile | — | 64 | 8×8 grid of DCT blocks → can carry ~64 bits reliably |
| Overlap margin | `MT_OVERLAP` | 8 px (1 block) | Overlapping border absorbs misalignment after crop |

For a 512×512 image: `512 / 64 = 8×8 = 64 macro-tiles`.
For a 1024×1024 image: `1024 / 64 = 16×16 = 256 macro-tiles`.

#### Why Fixed Grid (Not Content-Adaptive)?

Content-adaptive tiling (saliency-based) requires the detector to reconstruct the tiling scheme, which is impossible from a partial fragment without the original image. Fixed grids are self-describing: the detector knows the grid period and only needs to find the phase offset.

### 4.2 Per-Tile Payload Structure

Each macro-tile carries a self-contained bitstream embedded in its 8×8 DCT blocks:

```
┌──────────────────────────────────────────────┐
│              MACRO-TILE BITSTREAM             │
├──────────────────────────────────────────────┤
│  SYNC ANCHOR        │  8 bits                │
│  TILE INDEX          │  8 bits                │
│  SHARD DATA          │  S bits (≤ 40 bits)    │
│  SHARD CRC-8         │  8 bits                │
├──────────────────────────────────────────────┤
│  TOTAL per tile      │  ≤ 64 bits             │
└──────────────────────────────────────────────┘
```

#### 4.2.1 Sync Anchor (8 bits)

A fixed, key-derived 8-bit synchronization pattern embedded in the first 8 blocks of each macro-tile. The anchor serves two purposes:

1. **Tile boundary detection**: The detector scans for this pattern with a sliding window to locate macro-tile origins.
2. **Phase alignment**: By finding multiple anchors, the detector infers the grid phase (offset from image origin).

```python
def _tile_anchor(key: bytes) -> int:
    """Derive an 8-bit sync anchor from the embedding key."""
    digest = hmac.new(key, b"tile-anchor-v4", hashlib.sha256).digest()
    return digest[0]  # 8 bits
```

The anchor is **not** a fixed constant — it's key-derived, so an adversary without `K` cannot search for it.

#### 4.2.2 Tile Index (8 bits)

An 8-bit tile index `0..255` that uniquely identifies this tile's position in the outer RS codeword. Without the index, the decoder wouldn't know which shard slot to place the recovered data into.

The index is assigned in row-major order modulo 256:
```
index = (tile_row * tiles_per_row + tile_col) % 256
```

For images with >256 macro-tiles, multiple tiles share the same index, providing additional redundancy.

#### 4.2.3 Shard Data (S bits)

The actual payload fragment assigned to this tile. S is determined by the outer RS code parameters (see §5).

#### 4.2.4 Shard CRC-8 (8 bits)

A CRC-8 checksum over `[TILE_INDEX ∥ SHARD_DATA]`. This allows the detector to verify shard integrity before passing it to the outer RS decoder, preventing corrupted shards from poisoning the reconstruction.

### 4.3 Embedding Within a Macro-Tile

Each macro-tile is an 8×8 grid of 8×8 DCT blocks (64 blocks total). The tile's bitstream (≤ 64 bits) is embedded using the **same Phase 3 difference-modulation scheme** (`_embed_one_scale`), but with a tile-local block indexing:

```python
for block_idx in range(n_blocks_in_tile):
    bit = tile_bitstream[block_idx % len(tile_bitstream)]
    # ... Phase 3 DCT difference modulation ...
```

**Critical**: The PRNG seed for coefficient pair selection within a tile uses a **tile-local coordinate system**:

```python
def _tile_block_seed(key: bytes, tile_idx: int, local_row: int, local_col: int) -> int:
    """PRNG seed = HMAC(key, tile_idx ∥ local_row ∥ local_col)."""
    data = struct.pack(">IHH", tile_idx, local_row, local_col)
    return int.from_bytes(hmac.new(key, data, hashlib.sha256).digest()[:4], "big")
```

This means each tile's embedding is **independent** — the detector can process a single tile in isolation.

---

## 5. Redundant Encoding Across Regions (Outer RS Code)

### 5.1 Two-Layer ECC Architecture

Phase 4 introduces a **two-layer** error correction scheme:

```
Layer 1 (Inner): Phase 2 Reed-Solomon
    Input:  88 bytes (payload_core + signature)
    Output: 148 bytes = 1184 bits
    Corrects: Up to 30 byte-errors (~20% corruption)
    
Layer 2 (Outer): Phase 4 Spatial Reed-Solomon
    Input:  1184 bits = 148 bytes (inner-encoded payload)
    Output: N shards × S bits each
    Corrects: Loss of up to (N - K) entire shards (tiles)
```

### 5.2 Outer RS Code Parameters

The outer RS code operates at the **byte level** over the 148-byte inner codeword:

| Parameter | Symbol | Value | Rationale |
|---|---|---|---|
| Data bytes | `K_OUTER` | 148 | Full inner RS codeword |
| Total shards | `N_OUTER` | Variable (= number of macro-tiles, capped at 255) | GF(2^8) limits RS to 255 symbols max |
| Shard size | `S_OUTER` | Dynamically computed | `ceil(K_OUTER / min(K_OUTER, available_tiles))` bytes |
| Minimum reconstruction shards | `K_SHARDS` | ceil(148 / shard_bytes) | Need this many valid shards to reconstruct |

#### Concrete Example: 512×512 Image

```
Image: 512×512 px
Macro-tiles: 64 px → 8×8 = 64 tiles
Inner codeword: 148 bytes

Strategy: Split 148 bytes across tiles with redundancy
  shard_bytes = ceil(148 / 64) = 3 bytes = 24 bits per tile
  
  But we only have ~40 usable bits per tile (64 blocks - 24 overhead bits).
  shard_bytes = 5 bytes = 40 bits per tile  ← maximum fit
  
  K_SHARDS = ceil(148 / 5) = 30 shards needed for reconstruction
  N = 64 tiles available
  Redundancy factor = 64 / 30 ≈ 2.13×

  Survives loss of 64 - 30 = 34 tiles = 53% area destruction ✓
```

#### Concrete Example: 1024×1024 Image

```
Image: 1024×1024 px
Macro-tiles: 64 px → 16×16 = 256 tiles (capped at 255 by GF(256))
Inner codeword: 148 bytes

  shard_bytes = 1 byte = 8 bits per tile (easily fits in 64 blocks)
  K_SHARDS = 148 shards needed (148 / 1)
  N = 255 tiles available
  Redundancy factor = 255 / 148 ≈ 1.72×

  Survives loss of 255 - 148 = 107 tiles = 42% area destruction ✓
  
  ALTERNATIVE: shard_bytes = 2 bytes per tile
  K_SHARDS = ceil(148 / 2) = 74 shards needed
  N = 255 tiles
  Redundancy factor = 255 / 74 ≈ 3.45×

  Survives loss of 255 - 74 = 181 tiles = 71% area destruction ✓✓
```

### 5.3 Outer RS Encoding Algorithm

```python
def outer_rs_encode(inner_codeword: bytes, n_tiles: int) -> list[bytes]:
    """
    Distribute the 148-byte inner RS codeword across n_tiles using an outer RS code.
    
    Returns a list of n_tiles byte-strings, one shard per tile.
    Each shard is shard_bytes long.
    Any ceil(148 / shard_bytes) shards suffice to reconstruct the full codeword.
    """
    # Determine shard size: maximise redundancy within per-tile bit budget
    max_shard_bits = 40  # 64 blocks - 24 overhead bits
    shard_bytes = max_shard_bits // 8  # = 5 bytes
    
    k_shards = math.ceil(len(inner_codeword) / shard_bytes)
    
    # Pad inner codeword to exact multiple of shard_bytes
    padded = inner_codeword + b'\x00' * (k_shards * shard_bytes - len(inner_codeword))
    
    # Reshape into k_shards × shard_bytes matrix (each row = one data shard)
    data_matrix = [padded[i*shard_bytes : (i+1)*shard_bytes] for i in range(k_shards)]
    
    # Generate parity shards using Reed-Solomon over GF(2^8)
    # Each "symbol" for the outer RS is a shard_bytes-long block
    # We use a systematic RS(n_tiles, k_shards) code:
    #   first k_shards shards = data, remaining = parity
    rs_outer = reedsolo.RSCodec(n_tiles - k_shards, nsize=shard_bytes)
    
    # ... detailed implementation in §8 pseudocode ...
    
    return shards  # list of n_tiles byte-strings
```

### 5.4 Why Not Fountain Codes?

Fountain codes (e.g., LT codes, Raptor codes) are rateless and theoretically elegant, but:

1. **Overhead**: LT codes require ~5–10% overhead above the theoretical minimum K symbols. For our small message sizes (148 bytes), this wastes precious tile capacity.
2. **Decoder complexity**: Raptor codes need a pre-code layer + an LT layer. RS is simpler and well-tested.
3. **Fixed N**: We know the number of tiles at embed time (it's determined by image size), so a fixed-rate RS code is optimal.
4. **Error detection**: RS naturally detects which shards are corrupted (via CRC-8 checks), and can correct erasures at twice the rate of errors. Fountain codes treat all symbols as present-or-absent, losing this advantage.

---

## 6. Partial Detection Logic

### 6.1 Detection Modes

Phase 4 supports two detection modes:

| Mode | Input | Output | Use Case |
|---|---|---|---|
| **Presence Detection** | Any fragment ≥ 128×128 px | `{detected: bool, confidence: float}` | Quick triage: "Is this watermarked?" |
| **Full Verification** | Fragment containing ≥ K_SHARDS intact tiles | `{verified: bool, payload: dict}` | Ownership proof: "Who owns this?" |

### 6.2 Presence Detection (Fast Path)

Uses the existing Phase 3 multi-scale + harmonic detector, which already works on arbitrary-sized inputs. This is the same `detect_p3()` call, applied to whatever fragment is available.

**No changes needed** — Phase 3's tiled DCT correlation naturally works on partial images because the correlation is computed as a **mean** over available blocks.

### 6.3 Full Verification Pipeline (Slow Path)

```
detect_p4(fragment, key)
    │
    ├── 1. SPATIAL SYNC
    │       Sliding window scan for anchor patterns
    │       → Recover macro-tile grid phase (Δx, Δy)
    │       → Locate all intact macro-tiles
    │
    ├── 2. SHARD EXTRACTION
    │       For each located macro-tile:
    │         a. Extract tile bitstream (64 bits)
    │         b. Parse: [anchor | index | shard | crc]
    │         c. Verify CRC-8 → discard if corrupt
    │         d. Add (index, shard) to reconstruction buffer
    │
    ├── 3. SHARD DEDUPLICATION & VOTING
    │       Multiple tiles may share the same index
    │       → Majority vote per bit position
    │       → Select highest-confidence shard per index
    │
    ├── 4. OUTER RS DECODE
    │       Collect K_SHARDS valid (index, shard) pairs
    │       → RS erasure decode → recover 148-byte inner codeword
    │       → If < K_SHARDS available: report partial confidence
    │
    ├── 5. INNER RS DECODE (Phase 2)
    │       Existing ecc.decode_payload() on 1184-bit codeword
    │       → Recover 88-byte payload (payload_core + signature)
    │
    └── 6. CRYPTO VERIFY (Phase 2)
            crypto.verify(public_key, payload_core, signature)
            → Ownership confirmed
```

### 6.4 Confidence Scoring

Detection confidence is a composite of:

```python
def compute_confidence(
    phase3_score: float,        # Multi-scale DCT correlation [0, 1]
    n_tiles_found: int,         # Number of macro-tiles with valid anchors
    n_valid_shards: int,        # Number of CRC-valid shards
    k_shards_needed: int,       # Minimum shards for reconstruction
    ecc_inner_ok: bool,         # Inner RS decode succeeded
    signature_ok: bool,         # Ed25519 verification passed
) -> dict:
    """
    Composite confidence score for Phase 4 detection.
    """
    # Tier 1: Presence detection (no payload needed)
    presence_score = phase3_score
    
    # Tier 2: Structural detection (anchors found)
    structural_score = min(1.0, n_tiles_found / max(4, k_shards_needed * 0.3))
    
    # Tier 3: Payload reconstruction feasibility
    reconstruction_ratio = n_valid_shards / k_shards_needed
    reconstruction_score = min(1.0, reconstruction_ratio)
    
    # Tier 4: Cryptographic verification
    crypto_score = 1.0 if (ecc_inner_ok and signature_ok) else 0.0
    
    # Combined (weighted)
    combined = (
        0.15 * presence_score +
        0.15 * structural_score +
        0.30 * reconstruction_score +
        0.40 * crypto_score
    )
    
    return {
        "confidence": combined,
        "presence_score": presence_score,
        "structural_score": structural_score,
        "reconstruction_score": reconstruction_score,
        "reconstruction_ratio": reconstruction_ratio,
        "crypto_verified": signature_ok,
        "shards_found": n_valid_shards,
        "shards_needed": k_shards_needed,
        "tiles_located": n_tiles_found,
    }
```

#### Confidence Thresholds

| Threshold | Level | Meaning |
|---|---|---|
| `≥ 0.90` | **Verified** | Full payload recovered, signature valid |
| `0.60 – 0.89` | **High Confidence** | Most shards recovered, minor corruption |
| `0.30 – 0.59` | **Probable** | Presence + partial shards, payload not fully recoverable |
| `0.15 – 0.29` | **Possible** | Presence signal only, insufficient shards |
| `< 0.15` | **Not Detected** | No watermark evidence |

### 6.5 Graceful Degradation

When too few tiles are available for full reconstruction:

```
Available tiles:    Action:
────────────────    ───────────────────────────────────────────────
≥ K_SHARDS          Full reconstruction → signature verification
0.5×K to K          Partial reconstruction → report which bytes are
                    uncertain; attempt inner RS with soft info
4 to 0.5×K          Report anchor found + tile indices present;
                    cannot reconstruct payload; high presence confidence
1 to 3              Single-tile detection: report presence only;
                    extract tile index as partial provenance evidence
0                   Fall back to Phase 3 presence-only detection
```

---

## 7. Spatial Synchronization

### 7.1 The Alignment Problem

When an image is cropped, the crop boundary generally does not align with a macro-tile boundary. The detector must determine:

1. **Pixel-level offset** `(Δpx, Δpy)` — sub-tile shift (0..63 in each axis)
2. **Tile-phase offset** `(Δtx, Δty)` — which tile position the fragment starts at

### 7.2 Multi-Resolution Anchor Search

#### Step 1: Coarse Alignment via Phase 3 Harmonic

The Phase 3 sinusoidal harmonic provides **absolute spatial frequency** information. By detecting its phase in the FFT:

```python
def _detect_harmonic_phase(Y: np.ndarray, key: bytes) -> tuple[float, float]:
    """
    Detect the phase of the injected sinusoidal harmonic.
    
    Phase shift = 2π × (fx × Δx + fy × Δy) where (Δx, Δy) is the crop offset.
    If fx and fy are known (from key), we can estimate the fractional crop offset.
    """
    fx, fy = _harmonic_freq(key)
    fft = np.fft.fft2(Y.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    
    # Find the bin corresponding to (fx, fy)
    h, w = Y.shape
    bx = int(round(fx * w))
    by = int(round(fy * h))
    
    # Extract phase at the harmonic frequency
    cx, cy = w // 2 + bx, h // 2 + by
    phase_observed = np.angle(fft_shift[cy, cx])
    phase_expected = _harmonic_phase(key)
    
    delta_phase = (phase_observed - phase_expected) % (2 * np.pi)
    
    # Estimate spatial offset (ambiguous up to period)
    # dx ≈ delta_phase / (2π × fx) mod (1/fx)
    return delta_phase, (fx, fy)
```

This gives a **coarse** estimate of the crop offset, narrowing the search space.

#### Step 2: Fine Alignment via Anchor Scanning

For each candidate pixel offset `(Δpx, Δpy)` in a refined search window (narrowed by Step 1):

```python
def _scan_for_anchors(
    Y: np.ndarray, 
    key: bytes, 
    search_range: int = 64
) -> list[dict]:
    """
    Sliding window scan to locate macro-tile anchors.
    
    For each candidate (dx, dy) offset:
      1. Slice Y starting at (dy, dx)
      2. Extract the first 8 blocks' worth of bits
      3. Compare to expected anchor pattern
      4. Score the correlation
    
    Returns list of candidate tile positions sorted by score.
    """
    anchor_bits = _tile_anchor_bits(key)  # 8-bit pattern → list of 8 ints
    candidates = []
    
    for dy in range(0, search_range, 8):  # step by block size
        for dx in range(0, search_range, 8):
            # Extract bits from assumed first 8 blocks of a tile
            region = Y[dy : dy + 64, dx : dx + 64]
            if region.shape[0] < 64 or region.shape[1] < 64:
                continue
            
            extracted = _extract_tile_bits(region, key, tile_idx=0, n_bits=8)
            
            # Correlation with expected anchor
            match = sum(1 for a, b in zip(extracted, anchor_bits) if a == b) / 8.0
            
            if match >= 0.75:  # 6 of 8 bits match
                candidates.append({
                    "offset": (dx, dy),
                    "anchor_match": match,
                })
    
    return sorted(candidates, key=lambda c: -c["anchor_match"])
```

#### Step 3: Tile Grid Validation

Once a candidate offset is found, validate by checking for **periodic** anchor patterns:

```python
def _validate_grid(Y: np.ndarray, key: bytes, offset: tuple[int, int]) -> float:
    """
    Validate that anchors appear periodically at the expected macro-tile spacing.
    
    Returns a grid confidence score (0.0 to 1.0).
    """
    dx, dy = offset
    h, w = Y.shape
    n_expected = 0
    n_found = 0
    
    for ty in range(dy, h - 63, 64):  # MT_SIZE = 64
        for tx in range(dx, w - 63, 64):
            n_expected += 1
            region = Y[ty : ty + 64, tx : tx + 64]
            extracted = _extract_tile_bits(region, key, tile_idx=0, n_bits=8)
            anchor_expected = _tile_anchor_bits(key)
            match = sum(1 for a, b in zip(extracted, anchor_expected) if a == b) / 8.0
            if match >= 0.625:  # 5 of 8
                n_found += 1
    
    return n_found / max(1, n_expected)
```

### 7.3 Self-Aligning Detection (No Anchor Available)

If the fragment is too small to contain multiple anchors, or if anchors are damaged:

**Fallback: Exhaustive DCT Block Grid Search**

This is the same approach as Phase 1's `detect_robust()`, but extended to macro-tile level:

```
For px_dy in 0..7:       (pixel-level sub-block offset)
  For px_dx in 0..7:
    For mt_dy in 0..7:    (macro-tile phase offset — 8 candidates assuming MT_SIZE/8 = 8)
      For mt_dx in 0..7:
        Score = extract_and_score(Y, offset=(px_dx, px_dy), mt_phase=(mt_dx, mt_dy))
        Track best
```

Total search: `8 × 8 × 8 × 8 = 4096` candidates — same as `detect_robust()`. Each evaluation is O(n_blocks) inner product, so total cost is O(4096 × n_blocks). For a 256×256 fragment: `4096 × 1024 ≈ 4M` operations — well under 300ms.

### 7.4 Scale Invariance

If the image has been resized:

1. Phase 3's multi-scale detection (8, 16, 32 block sizes) naturally accommodates moderate rescaling.
2. For large scale changes, the detector can attempt detection at multiple assumed scales:
   ```
   for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
       rescaled = cv2.resize(fragment, None, fx=1/scale, fy=1/scale)
       result = detect_p4(rescaled, key)
       if result.detected: return result
   ```

---

## 8. Pseudocode

### 8.1 `embed_p4(image, payload, key)`

```python
def embed_p4(
    image: np.ndarray,          # BGR uint8 (H, W, 3)
    bitstream: np.ndarray,      # 1184-bit ECC-encoded payload (from Phase 2)
    key: bytes,                 # Secret embedding key
) -> np.ndarray:
    """
    Phase 4 spatial-redundant watermark embedding.
    
    Pipeline:
    1. Outer RS encode: split 1184-bit payload into N shards
    2. For each macro-tile:
       a. Construct tile bitstream: [anchor | index | shard | crc]
       b. Embed into tile's DCT blocks using Phase 3 engine
    3. Add Phase 3 sinusoidal harmonic (global)
    """
    H, W = image.shape[:2]
    ycrcb, Y_orig = _to_ycbcr(image)
    
    # ── Step 1: Compute macro-tile grid ────────────────────────────
    mt_rows = H // MT_SIZE       # number of complete macro-tile rows
    mt_cols = W // MT_SIZE       # number of complete macro-tile columns
    n_tiles = mt_rows * mt_cols  # total macro-tiles
    
    if n_tiles < 4:
        raise ValueError(
            f"Image too small for Phase 4: {H}×{W} yields {n_tiles} macro-tiles "
            f"(minimum 4). Minimum image size: {MT_SIZE*2}×{MT_SIZE*2} px."
        )
    
    # ── Step 2: Outer RS encode ────────────────────────────────────
    inner_codeword = bits_to_bytes(bitstream.tolist())  # 148 bytes
    
    # Determine shard sizing
    bits_per_tile_payload = BLOCKS_PER_MT - ANCHOR_BITS - INDEX_BITS - CRC_BITS
    shard_bytes = bits_per_tile_payload // 8  # max bytes per shard
    k_shards = math.ceil(len(inner_codeword) / shard_bytes)
    
    # Enforce n_tiles ≤ 255 for GF(2^8) RS
    n_rs = min(n_tiles, 255)
    n_parity = n_rs - k_shards
    
    if n_parity < 1:
        # Not enough tiles for meaningful redundancy — fall back to repetition
        shards = _repetition_encode(inner_codeword, n_tiles, shard_bytes)
    else:
        shards = _outer_rs_encode(inner_codeword, k_shards, n_rs, shard_bytes)
    
    # ── Step 3: Build per-tile bitstreams ──────────────────────────
    anchor = _tile_anchor_bits(key)     # 8 bits
    Y_modified = Y_orig.copy().astype(np.float32)
    
    for tile_idx in range(n_tiles):
        tr = tile_idx // mt_cols
        tc = tile_idx % mt_cols
        
        # Shard for this tile (cyclic if n_tiles > n_rs)
        shard_idx = tile_idx % n_rs
        shard_data = shards[shard_idx]
        shard_bits = bytes_to_bits(shard_data)
        
        # Tile index (8 bits)
        index_bits = [(shard_idx >> (7 - b)) & 1 for b in range(8)]
        
        # CRC-8 over [index ∥ shard]
        crc_input = bytes([shard_idx]) + shard_data
        crc_val = _crc8(crc_input)
        crc_bits = [(crc_val >> (7 - b)) & 1 for b in range(8)]
        
        # Assemble tile bitstream
        tile_bits = anchor + index_bits + shard_bits + crc_bits
        tile_bits = np.array(tile_bits, dtype=np.int32)
        
        # Extract tile region from Y
        y0 = tr * MT_SIZE
        x0 = tc * MT_SIZE
        tile_Y = Y_modified[y0 : y0 + MT_SIZE, x0 : x0 + MT_SIZE].copy()
        
        # Embed using Phase 3 single-scale engine (8×8 blocks)
        tile_padded = _pad_to_n(tile_Y, 8)
        tile_embedded = _embed_one_scale(tile_padded, tile_bits, key, 8)
        
        # Write back
        Y_modified[y0 : y0 + MT_SIZE, x0 : x0 + MT_SIZE] = \
            tile_embedded[:MT_SIZE, :MT_SIZE]
    
    # ── Step 4: Add sinusoidal harmonic (global, Phase 3) ──────────
    harmonic = _build_harmonic_map(H, W, key)
    Y_final = Y_modified + harmonic
    
    return _from_ycbcr(ycrcb, Y_final)
```

### 8.2 `detect_p4(image_fragment, key)`

```python
def detect_p4(
    image_fragment: np.ndarray,   # BGR uint8, arbitrary size
    key: bytes,                   # Secret embedding key
) -> dict:
    """
    Phase 4 spatial-redundant watermark detection.
    
    Works on arbitrary image fragments — does NOT require the full image.
    
    Returns:
        detected:              bool
        verified:              bool   (True only if signature validates)
        confidence:            float  (composite score)
        presence_score:        float  (Phase 3 DCT+harmonic)
        tiles_located:         int
        shards_recovered:      int
        shards_needed:         int
        reconstruction_ratio:  float
        payload:               dict | None
        error:                 str | None
        tile_map:              list[dict]  (per-tile extraction results)
    """
    result = {
        "detected": False, "verified": False, "confidence": 0.0,
        "presence_score": 0.0, "tiles_located": 0, "shards_recovered": 0,
        "shards_needed": 0, "reconstruction_ratio": 0.0,
        "payload": None, "error": None, "tile_map": [],
    }
    
    H, W = image_fragment.shape[:2]
    ycrcb, Y = _to_ycbcr(image_fragment)
    
    # ── Step 1: Phase 3 presence detection ─────────────────────────
    p3_result = detect_p3(image_fragment, key)
    result["presence_score"] = p3_result["confidence"]
    result["scale_scores"] = p3_result["scale_scores"]
    result["harmonic_score"] = p3_result["harmonic_score"]
    
    if p3_result["confidence"] < 0.50:
        # No watermark signal at all
        return result
    
    result["detected"] = True  # Presence detected
    
    # ── Step 2: Spatial synchronization ────────────────────────────
    # Try all pixel-level offsets (0..7) × macro-tile phases (0..7)
    best_offset = None
    best_grid_score = -1.0
    
    for px_dy in range(0, min(8, H - MT_SIZE)):
        for px_dx in range(0, min(8, W - MT_SIZE)):
            Y_shifted = Y[px_dy:, px_dx:]
            score = _validate_grid(Y_shifted, key, (0, 0))
            if score > best_grid_score:
                best_grid_score = score
                best_offset = (px_dx, px_dy)
    
    if best_offset is None or best_grid_score < 0.3:
        # Cannot find grid — report presence only
        result["error"] = "Anchor grid not found; presence-only detection"
        return result
    
    # ── Step 3: Extract shards from located tiles ──────────────────
    dx, dy = best_offset
    Y_aligned = Y[dy:, dx:]
    ah, aw = Y_aligned.shape
    
    mt_rows = ah // MT_SIZE
    mt_cols = aw // MT_SIZE
    
    anchor_expected = _tile_anchor_bits(key)
    shard_buffer = {}  # index → list of (shard_bytes, confidence)
    
    for tr in range(mt_rows):
        for tc in range(mt_cols):
            y0 = tr * MT_SIZE
            x0 = tc * MT_SIZE
            tile_Y = Y_aligned[y0 : y0 + MT_SIZE, x0 : x0 + MT_SIZE]
            
            # Extract all bits from this tile
            tile_bits = _extract_tile_bits(tile_Y, key, tile_idx=0, n_bits=64)
            
            # Parse structure
            anchor_recv   = tile_bits[0:8]
            index_recv    = tile_bits[8:16]
            shard_recv    = tile_bits[16:56]  # 40 bits = 5 bytes
            crc_recv      = tile_bits[56:64]
            
            # Check anchor
            anchor_match = sum(1 for a, b in zip(anchor_recv, anchor_expected) if a == b) / 8.0
            if anchor_match < 0.625:
                continue  # Not a valid tile
            
            # Decode index
            tile_index = 0
            for b in index_recv:
                tile_index = (tile_index << 1) | b
            
            # Decode shard
            shard_data = bits_to_bytes(shard_recv)
            
            # Verify CRC
            crc_expected = _crc8(bytes([tile_index]) + shard_data)
            crc_received = 0
            for b in crc_recv:
                crc_received = (crc_received << 1) | b
            
            crc_ok = (crc_expected == crc_received)
            
            tile_info = {
                "position": (tr, tc),
                "index": tile_index,
                "anchor_match": anchor_match,
                "crc_valid": crc_ok,
            }
            result["tile_map"].append(tile_info)
            
            if crc_ok:
                if tile_index not in shard_buffer:
                    shard_buffer[tile_index] = []
                shard_buffer[tile_index].append((shard_data, anchor_match))
    
    result["tiles_located"] = len(result["tile_map"])
    
    # ── Step 4: Shard deduplication (majority vote per index) ──────
    best_shards = {}
    for idx, candidates in shard_buffer.items():
        if len(candidates) == 1:
            best_shards[idx] = candidates[0][0]
        else:
            # Majority vote: bit-wise voting across candidates
            n_shard_bits = len(candidates[0][0]) * 8
            voted_bits = []
            for bit_pos in range(n_shard_bits):
                ones = sum(1 for c, _ in candidates 
                           if (c[bit_pos // 8] >> (7 - bit_pos % 8)) & 1)
                voted_bits.append(1 if ones > len(candidates) / 2 else 0)
            best_shards[idx] = bits_to_bytes(voted_bits)
    
    n_valid = len(best_shards)
    result["shards_recovered"] = n_valid
    
    # ── Step 5: Outer RS decode ────────────────────────────────────
    shard_bytes = 5  # bytes per shard
    k_shards = math.ceil(148 / shard_bytes)  # = 30
    result["shards_needed"] = k_shards
    result["reconstruction_ratio"] = n_valid / k_shards
    
    if n_valid < k_shards:
        result["error"] = (
            f"Insufficient shards: {n_valid}/{k_shards}. "
            f"Need {k_shards - n_valid} more for reconstruction."
        )
        # Update confidence with partial info
        result["confidence"] = compute_confidence(
            result["presence_score"], result["tiles_located"],
            n_valid, k_shards, False, False
        )["confidence"]
        return result
    
    # Reconstruct using outer RS
    inner_codeword = _outer_rs_decode(best_shards, k_shards, shard_bytes)
    
    if inner_codeword is None:
        result["error"] = "Outer RS decode failed — too many corrupted shards"
        return result
    
    # ── Step 6: Inner RS decode (Phase 2) ──────────────────────────
    inner_bits = bytes_to_bits(inner_codeword)
    decoded_bits, ecc_ok = decode_payload(inner_bits)
    
    if not ecc_ok:
        result["error"] = "Inner RS decode failed"
        result["confidence"] = compute_confidence(
            result["presence_score"], result["tiles_located"],
            n_valid, k_shards, False, False
        )["confidence"]
        return result
    
    # ── Step 7: Parse + verify signature ───────────────────────────
    try:
        payload_core, signature = parse_embed_payload(decoded_bits)
    except Exception as exc:
        result["error"] = f"Payload parse failed: {exc}"
        return result
    
    # Note: public_key must be provided for full verification
    # This function returns the raw payload for external verification
    result["verified"] = True  # Placeholder — actual sig check in mist.verify_p4()
    result["payload_core"] = payload_core
    result["signature"] = signature
    result["confidence"] = 1.0
    
    return result
```

---

## 9. Parameter Recommendations

### 9.1 Core Parameters

| Parameter | Symbol | Recommended | Range | Impact |
|---|---|---|---|---|
| Macro-tile size | `MT_SIZE` | **64 px** | 32–128 | Smaller = more tiles = more redundancy, but fewer bits per tile |
| Blocks per macro-tile (8×8) | `BLOCKS_PER_MT` | **64** | — | Derived: `(MT_SIZE/8)²` |
| Anchor bits | `ANCHOR_BITS` | **8** | 4–16 | More = better sync, fewer payload bits |
| Index bits | `INDEX_BITS` | **8** | 6–10 | 8 bits → 256 max unique indices |
| CRC bits | `CRC_BITS` | **8** | 0–16 | Enables per-shard integrity check |
| Shard data bits | `SHARD_DATA_BITS` | **40** | — | Derived: `BLOCKS_PER_MT - ANCHOR - INDEX - CRC` |
| Shard bytes | `SHARD_BYTES` | **5** | — | `SHARD_DATA_BITS // 8` |

### 9.2 Outer RS Parameters

| Parameter | Formula | 512×512 | 1024×1024 |
|---|---|---|---|
| Total macro-tiles | `(H/64) × (W/64)` | 64 | 256 (capped to 255) |
| K_SHARDS (data) | `ceil(148 / SHARD_BYTES)` | 30 | 30 |
| N_RS (codeword length) | `min(n_tiles, 255)` | 64 | 255 |
| Parity shards | `N_RS - K_SHARDS` | 34 | 225 |
| **Max tile loss** | `N_RS - K_SHARDS` | **34 (53%)** | **225 (88%)** |

### 9.3 Detection Thresholds

| Threshold | Value | Purpose |
|---|---|---|
| Anchor match minimum | **≥ 0.625** (5/8 bits) | Accept tile as valid |
| Grid validation minimum | **≥ 0.30** (30% of expected anchors) | Confirm grid alignment |
| Phase 3 presence cutoff | **≥ 0.50** confidence | Gate for shard extraction |
| Minimum fragment size | **128×128 px** | Need ≥ 4 macro-tiles |
| Verification confidence | **≥ 0.90** | Full ownership proof |

### 9.4 Perceptual Quality Targets

| Metric | Phase 3 Alone | Phase 4 Target | Degradation Budget |
|---|---|---|---|
| PSNR | ≥ 38 dB | **≥ 36.5 dB** | ≤ 1.5 dB |
| SSIM | ≥ 0.98 | **≥ 0.97** | ≤ 0.01 |
| Embedding time (512×512) | < 200 ms | **< 350 ms** | 150 ms for tiling overhead |
| Detection time (512×512) | < 300 ms | **< 500 ms** | 200 ms for spatial sync |

---

## 10. Integration with Previous Phases

### 10.1 Module Dependency Graph

```
mist.py
  ├── watermark_p4()  ←  NEW
  │     ├── payload.build_embed_payload()     [Phase 2]
  │     ├── ecc.encode_payload()              [Phase 2]
  │     ├── wm_engine_p4.embed_p4()           [Phase 4 - NEW]
  │     │     ├── wm_engine_p4._outer_rs_encode()  [Phase 4 - NEW]
  │     │     ├── wm_engine_p3._embed_one_scale()  [Phase 3 - REUSED]
  │     │     └── wm_engine_p3._build_harmonic_map() [Phase 3 - REUSED]
  │     └── (no changes to crypto.py)
  │
  └── verify_p4()  ←  NEW
        ├── wm_engine_p3.detect_p3()          [Phase 3 - REUSED for presence]
        ├── wm_engine_p4.detect_p4()          [Phase 4 - NEW]
        │     ├── wm_engine_p4._scan_for_anchors()  [Phase 4 - NEW]
        │     ├── wm_engine_p4._outer_rs_decode()   [Phase 4 - NEW]
        │     └── wm_engine_p3.extract_bits_p3()    [Phase 3 - REUSED]
        ├── ecc.decode_payload()              [Phase 2]
        ├── payload.parse_embed_payload()     [Phase 2]
        └── crypto.verify()                   [Phase 2]
```

### 10.2 Backward Compatibility

- **Phase 2 payloads**: Phase 4 wraps the exact same 1184-bit inner codeword. The inner RS code is unchanged.
- **Phase 3 engine**: The single-scale embedding function `_embed_one_scale()` is reused as-is for per-tile embedding.
- **Phase 3 harmonic**: Applied globally (full image) as before, providing presence detection on any fragment.
- **Key format**: Same `embed_key` bytes — Phase 4 derives additional tile-specific seeds via HMAC with different salt prefixes.

### 10.3 API Contract Update (SPEC_v1.md §8)

```python
# New Phase 4 functions
def multi_scale_embed(image, payload, key) -> np.ndarray:
    """Phase 4: Tile-based spatially redundant embedding."""
    # Alias: embed_p4()
    
def detect_fragment(image_fragment, key) -> dict:
    """Phase 4: Detection on arbitrary image fragments."""
    # Alias: detect_p4()
    
# Updated mist.py surface
def watermark_p4(image, user_id, image_id, private_key, embed_key, ...) -> np.ndarray
def verify_p4(image, public_key, embed_key) -> dict
```

---

## 11. Failure Modes & Mitigations

### 11.1 Known Failure Modes

| Failure Mode | Cause | Probability | Mitigation |
|---|---|---|---|
| **Anchor collision** | Non-watermarked image accidentally has the 8-bit anchor pattern in DCT coefficients | ~1/256 per tile, but grid periodicity check eliminates false grids | CRC-8 provides secondary validation |
| **Shard corruption below CRC detection** | CRC-8 has only 1/256 collision chance — a corrupted shard might pass CRC | Low (~0.4% per shard) | Inner RS provides second error correction layer |
| **All tiles of a specific index destroyed** | Adversary removes a contiguous band that happens to contain all copies of a particular index | Possible for strip-shaped crops | Index assignment wraps modulo, so tiles with same index are spatially distributed |
| **Scale change breaks grid** | 2× resize changes effective MT_SIZE from 64 to 128 or 32 px | Moderate | Multi-scale detection attempts (§7.4) |
| **Heavy JPEG on small fragment** | Q=30 on 128×128 fragment — only 4 tiles, each with only 64 blocks | Marginal | Presence detection still works via Phase 3 correlation; payload unrecoverable |

### 11.2 Adversary's Best Attack

The strongest attack is **targeted tile destruction**:

1. Adversary knows the system uses a 64px grid (public knowledge per Kerckhoffs's principle)
2. Adversary overlays content at 64px intervals to destroy anchor blocks
3. This requires destroying the **first row of blocks** in each tile

**Mitigation**: The anchor pattern is not at a fixed position within the tile. Its position is HMAC-derived from the key:

```python
def _anchor_offset(key: bytes) -> int:
    """Anchor starts at block offset 0..7 within the tile (key-derived)."""
    return int.from_bytes(
        hmac.new(key, b"anchor-offset-v4", hashlib.sha256).digest()[:1], "big"
    ) % 8
```

Without knowing `K`, the adversary cannot know which blocks carry the anchor.

### 11.3 Residual Weaknesses

| Weakness | Severity | Planned Improvement |
|---|---|---|
| **FPR inflation from grid search** | Medium | Phase 5: calibrate threshold on 10k images with spatial search |
| **64px grid visible in analysis** | Low | Randomize MT_SIZE slightly per key (60–68 px range) |
| **No learning-based alignment** | Low | Phase 7+: train CNN to predict grid offset |
| **Small images (< 256×256)** | High | Not enough tiles for meaningful redundancy; fall back to Phase 3 |

---

## 12. Bonus: Adaptive / Multi-Scale Tiling

### 12.1 Multi-Scale Tile Strategy

In addition to the primary 64px macro-tiles, embed a **coarse summary** at 128px and 256px tile scales:

```
Scale 1 (Primary):   64×64 px tiles  →  Full sharded payload
Scale 2 (Coarse):   128×128 px tiles →  Payload hash (32 bits per tile)
Scale 3 (Ultra):    256×256 px tiles →  Detection-only presence signal
```

The coarse scales provide **supplementary evidence** without carrying the full payload:

- A 128px tile carries a 32-bit hash of the payload → helps validate reconstruction
- A 256px tile carries only a presence signal (key-derived correlation pattern)

This is implemented as three independent passes of `_embed_one_scale()` at block sizes 8, 16, and 32 — reusing the existing Phase 3 multi-scale infrastructure.

### 12.2 Saliency-Guided Strength (Optional Enhancement)

Use a simple gradient-magnitude saliency map to increase embedding strength in textured regions:

```python
def _saliency_weight(tile_Y: np.ndarray) -> float:
    """
    Compute a saliency-based strength multiplier for a tile.
    High-texture tiles get more watermark energy (better masked by content).
    """
    grad_x = cv2.Sobel(tile_Y, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(tile_Y, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    saliency = float(np.mean(mag))
    
    # Map to multiplier [0.8, 1.5]
    return np.clip(0.8 + saliency / 100.0, 0.8, 1.5)
```

This is applied as a per-tile multiplier on `BASE_DELTA_P3`:

```python
delta_tile = BASE_DELTA_P3 * saliency_weight(tile_Y)
```

---

## 13. Implementation File Plan

### 13.1 New Files

| File | Contents |
|---|---|
| `src/core/wm_engine_p4.py` | `embed_p4()`, `detect_p4()`, outer RS encode/decode, anchor/sync logic |
| `scripts/validate_phase4.py` | Benchmark: crop at 30/50/70%, overlay attack, fragment detection |

### 13.2 Modified Files

| File | Change |
|---|---|
| `src/core/mist.py` | Add `watermark_p4()`, `verify_p4()` |
| `docs/PHASES.md` | Update Phase 4 section with spatial redundancy description |
| `docs/SPEC_v1.md` | Add `detect_fragment()` to API contract table |

### 13.3 Dependencies

```
# New (if not already present):
reedsolo      # Already in requirements.txt — used for outer RS as well
crcmod        # For CRC-8 computation (pip install crcmod)
              # Alternative: implement CRC-8 manually (8 lines of code)
```

---

## 14. Validation Plan

### 14.1 Automated Tests

```bash
python scripts/validate_phase4.py
```

| Test | Input | Operation | Pass Criterion |
|---|---|---|---|
| **Crop 30%** | 512×512 watermarked | Random 30% area crop | `verified == True` |
| **Crop 50%** | 512×512 watermarked | Random 50% area crop | `verified == True` |
| **Crop 70%** | 512×512 watermarked | Random 70% area crop | `detected == True`, `reconstruction_ratio ≥ 0.5` |
| **Overlay 30%** | 512×512 watermarked | Random rectangles covering 30% | `verified == True` |
| **Fragment 128px** | 512×512 watermarked | Extract 128×128 corner | `detected == True` |
| **Fragment 256px** | 512×512 watermarked | Extract 256×256 region | `verified == True` OR `reconstruction_ratio ≥ 0.8` |
| **JPEG + Crop** | 512×512 watermarked | Q=50 JPEG then 40% crop | `verified == True` |
| **Screenshot + Crop** | 512×512 watermarked | Screenshot sim then 30% crop | `detected == True` |
| **Meme overlay** | 512×512 watermarked | Add text + sticker covering 40% | `verified == True` |
| **PSNR regression** | 512×512 clean | Embed Phase 4 watermark | `PSNR ≥ 36.5 dB`, `SSIM ≥ 0.97` |
| **False positive** | 1000 unwatermarked | Run `detect_p4()` | `FPR < 0.5%` |

### 14.2 Benchmark Metrics

| Metric | Measurement Method |
|---|---|
| **Survival curve** | Payload recovery rate vs. % area destroyed (10% steps) |
| **Fragment size curve** | Minimum fragment size for detection at 90/95/99% confidence |
| **Embedding overhead** | Wall-clock time increase vs. Phase 3 alone |
| **PSNR/SSIM distribution** | Over 100 test images from dataset/ |

---

*Mist Phase 4 Design — Spatial Attack Resistance Engine*
*Document version: 1.0 — 2026-04-04*
