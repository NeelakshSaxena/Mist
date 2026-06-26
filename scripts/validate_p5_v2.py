"""
scripts/validate_p5_v2.py  –  Phase 5-V2 Validation + Profiling Harness

Pipeline under test (P5-V2):
    Image → FFT sync → log-polar registration → affine correction
         → sync-template refinement → P4 extraction → shard voting
         → RS decode → signature verification → calibrated forensic confidence

Metrics collected per attack:
    • runtime per stage (from P5Profile)
    • shard_crc_ratio / shard accuracy
    • geometry error  (|estimated - true|)
    • detection / verification flags
    • confidence score

Outputs:
    outputs/p5v2_report.csv          – per-case CSV
    outputs/p5v2_timing.png          – runtime histogram per stage
    outputs/p5v2_attack_heatmap.png  – verification rate heatmap by attack

Stop Conditions:
    PASS  total runtime < 15 s  AND  verification rate > 90 %
    FAIL  any single stage > stage budget  OR  geometry diverges (>10° error)

Usage:
    python -m scripts.validate_p5_v2
    python -m scripts.validate_p5_v2 --fast       # 3 seeds, fewer angles
    python -m scripts.validate_p5_v2 --no-plots   # skip matplotlib
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── encoding ──────────────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.crypto import generate_keys
from src.core.mist import watermark_p5, verify_p5, forensic_report
from src.core.wm_engine_p4 import K_SHARDS
from src.core.forensic import generate_roc_data
from src.core.p5_profiler import P5Profile
from src.attacks.geometric import rotate, scale, crop_and_resize


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

# Stage runtime budgets (seconds)
STAGE_BUDGETS: dict[str, float] = {
    "0_fourier_mellin":  0.5,
    "0b_sync_template":  0.5,
    "1_identity":        3.0,
    "2_generate":        0.2,
    "3_canary":          3.0,
    "4_promote":         8.0,
    "5_extended":        4.0,
    "6_refine":          3.0,
}
TOTAL_RUNTIME_BUDGET = 15.0   # seconds — PASS gate
VERIFY_RATE_THRESHOLD = 0.90  # PASS gate
GEO_DIVERGE_DEG = 10.0        # FAIL if |estimated - true| > this


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    attack_type:       str   = ""
    attack_param:      str   = ""
    true_angle:        float = 0.0
    true_scale:        float = 1.0
    detected:          bool  = False
    verified:          bool  = False
    confidence:        float = 0.0
    shards_recovered:  int   = 0
    shard_crc_ratio:   float = 0.0
    estimated_angle:   float = 0.0
    estimated_scale:   float = 1.0
    geo_angle_error:   float = 0.0
    geo_scale_error:   float = 0.0
    runtime_total:     float = 0.0
    # stage times (flat in CSV)
    t_fourier_mellin:  float = 0.0
    t_sync_template:   float = 0.0
    t_identity:        float = 0.0
    t_generate:        float = 0.0
    t_canary:          float = 0.0
    t_promote:         float = 0.0
    # budget violations
    stage_budget_fail: str   = ""
    geo_diverge:       bool  = False

    def csv_row(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
#  JPEG attack helper
# ─────────────────────────────────────────────────────────────────────────────

def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    enc_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode(".jpg", image, enc_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
#  Test image factory
# ─────────────────────────────────────────────────────────────────────────────

def make_test_image(h: int = 512, w: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    base = ((y + x) / 2).astype(np.uint8)
    noise = rng.integers(0, 40, (h, w), dtype=np.uint8)
    gray = np.clip(
        base.astype(np.int16) + noise.astype(np.int16), 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-case runner
# ─────────────────────────────────────────────────────────────────────────────

def run_case(
    attacked: np.ndarray,
    pub_key: bytes,
    embed_key: bytes,
    attack_type: str,
    attack_param: str,
    true_angle: float = 0.0,
    true_scale: float = 1.0,
) -> CaseResult:
    r = CaseResult(
        attack_type=attack_type,
        attack_param=attack_param,
        true_angle=true_angle,
        true_scale=true_scale,
    )

    with P5Profile() as prof:
        t0 = time.perf_counter()
        result = verify_p5(attacked, pub_key, embed_key)
        r.runtime_total = time.perf_counter() - t0

    r.detected         = result.get("detected", False)
    r.verified         = result.get("verified", False)
    r.confidence       = result.get("confidence", 0.0)
    r.shards_recovered = result.get("shards_recovered", 0)

    # shard_crc_ratio: try direct key first (detect_p5 stores it), else p4 dict
    r.shard_crc_ratio  = result.get("shard_crc_ratio", 0.0)

    # geometry
    geo = result.get("geometry") or {}
    r.estimated_angle  = geo.get("angle_deg", 0.0)
    r.estimated_scale  = geo.get("scale_factor", 1.0)
    r.geo_angle_error  = abs(r.estimated_angle - true_angle)
    r.geo_scale_error  = abs(r.estimated_scale - true_scale)
    r.geo_diverge      = r.geo_angle_error > GEO_DIVERGE_DEG

    # stage times
    st = prof.stage_times
    r.t_fourier_mellin = st.get("0_fourier_mellin", 0.0)
    r.t_sync_template  = st.get("0b_sync_template", 0.0)
    r.t_identity       = st.get("1_identity", 0.0)
    r.t_generate       = st.get("2_generate", 0.0)
    r.t_canary         = st.get("3_canary", 0.0)
    # Stage 0c (FM direct) counts against the promote budget when present
    t_0c               = st.get("0c_fm_direct", 0.0)
    r.t_promote        = st.get("4_promote", 0.0) + t_0c

    # budget violations
    violations = []
    for stage, budget in STAGE_BUDGETS.items():
        actual = st.get(stage, 0.0)
        if actual > budget:
            violations.append(f"{stage}={actual:.2f}s>{budget}s")
    r.stage_budget_fail = "; ".join(violations)

    return r


# ─────────────────────────────────────────────────────────────────────────────
#  Attack suite definition
# ─────────────────────────────────────────────────────────────────────────────

def build_attack_suite(wm_image: np.ndarray, fast: bool) -> list[tuple]:
    """
    Returns list of (attacked_image, attack_type, attack_param, true_angle, true_scale).
    """
    suite = []

    # ── Baseline ─────────────────────────────────────────────────────────
    suite.append((wm_image.copy(), "baseline", "none", 0.0, 1.0))

    # ── Rotation ±20° ────────────────────────────────────────────────────
    angles = [-20, -15, -10, -5, 5, 10, 15, 20] if not fast else [-10, 10, 20]
    for a in angles:
        suite.append((rotate(wm_image, a), "rotation", f"{a:+d}deg", float(a), 1.0))

    # ── Scale 0.5×–1.5× ──────────────────────────────────────────────────
    scales_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5] if not fast \
                  else [0.6, 0.8, 1.3, 1.5]
    for s in scales_list:
        suite.append((scale(wm_image, s), "scale", f"{s:.1f}x", 0.0, s))

    # ── Crop 10–40% ──────────────────────────────────────────────────────
    crop_fracs = [0.90, 0.80, 0.70, 0.60] if not fast else [0.80, 0.60]
    for frac in crop_fracs:
        attacked = crop_and_resize(wm_image, frac, seed=42)
        pct = int((1 - frac) * 100)
        suite.append((attacked, "crop", f"{pct}pct", 0.0, 1.0))

    # ── JPEG 30–95 ───────────────────────────────────────────────────────
    jpeg_qualities = [30, 50, 70, 85, 95] if not fast else [50, 85]
    for q in jpeg_qualities:
        suite.append((jpeg_compress(wm_image, q), "jpeg", f"q{q}", 0.0, 1.0))

    # ── Combined attacks ─────────────────────────────────────────────────
    combos = [
        (10,  1.2,  None,  None),
        (-10, 0.8,  None,  None),
        (15,  1.0,  None,  0.80),
        (5,   0.9,  85,    None),
    ] if not fast else [
        (10, 1.2, None, None),
        (-10, 0.8, 85, 0.80),
    ]
    for ang, sc_f, jpeg_q, crop_f in combos:
        img = rotate(wm_image, ang) if ang != 0 else wm_image.copy()
        if sc_f != 1.0:
            img = scale(img, sc_f)
        if crop_f is not None:
            img = crop_and_resize(img, crop_f, seed=7)
        if jpeg_q is not None:
            img = jpeg_compress(img, jpeg_q)
        label = f"r{ang:+d}_s{sc_f}"
        if crop_f:
            label += f"_c{int((1-crop_f)*100)}"
        if jpeg_q:
            label += f"_j{jpeg_q}"
        suite.append((img, "combined", label, float(ang), sc_f))

    return suite


# ─────────────────────────────────────────────────────────────────────────────
#  False-Positive suite (clean images)
# ─────────────────────────────────────────────────────────────────────────────

def run_fp_suite(pub_key, embed_key, n: int = 10) -> list[CaseResult]:
    results = []
    for seed in range(n):
        clean = make_test_image(512, 512, seed=5000 + seed)
        r = run_case(clean, pub_key, embed_key, "clean_fp", f"seed{seed}")
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CSV output
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(results: list[CaseResult], path: Path):
    if not results:
        return
    rows = [r.csv_row() for r in results]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_timing_histogram(results: list[CaseResult], path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    stage_keys = [
        ("t_fourier_mellin", "FM sync"),
        ("t_sync_template",  "Sync template"),
        ("t_identity",       "Identity P4"),
        ("t_generate",       "Generate"),
        ("t_canary",         "Canary (GPU)"),
        ("t_promote",        "Promote P4"),
    ]

    labels  = [lbl for _, lbl in stage_keys]
    medians = []
    p95s    = []
    for key, _ in stage_keys:
        vals = [getattr(r, key) for r in results if getattr(r, key) > 0]
        medians.append(float(np.median(vals)) if vals else 0.0)
        p95s.append(float(np.percentile(vals, 95)) if vals else 0.0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, medians, width, label="Median", color="#4C9BE8")
    ax.bar(x + width / 2, p95s,    width, label="p95",    color="#E8844C")

    # Budget lines
    budget_vals = [
        STAGE_BUDGETS.get(k, 99) for k, _ in stage_keys
    ]
    for xi, bv in zip(x, budget_vals):
        ax.plot([xi - width, xi + width], [bv, bv], "r--", lw=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title("P5-V2 Stage Timing — Median & p95  (red dashed = budget)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Plot → {path}")


def plot_attack_heatmap(results: list[CaseResult], path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    attack_types = ["rotation", "scale", "crop", "jpeg", "combined"]
    metrics = ["detected", "verified"]

    data_det  = []
    data_ver  = []
    row_labels = []

    for at in attack_types:
        sub = [r for r in results if r.attack_type == at]
        if not sub:
            continue
        row_labels.append(at)
        data_det.append(np.mean([1 if r.detected else 0 for r in sub]))
        data_ver.append(np.mean([1 if r.verified else 0 for r in sub]))

    if not row_labels:
        return

    mat = np.array([data_det, data_ver])  # (2, n_attack_types)

    fig, ax = plt.subplots(figsize=(max(6, len(row_labels) * 1.4), 3))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(row_labels)))
    ax.set_xticklabels(row_labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Detected", "Verified"])
    ax.set_title("P5-V2 Attack Survival Heatmap")
    for i in range(2):
        for j in range(len(row_labels)):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center",
                    fontsize=11, color="black")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Plot → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_case(r: CaseResult):
    status = "PASS" if r.verified else ("DET " if r.detected else "FAIL")
    geo_flag = " [GEO!]" if r.geo_diverge else ""
    budget_flag = " [BUDGET!]" if r.stage_budget_fail else ""
    print(
        f"  [{status}] {r.attack_type:10s} {r.attack_param:18s}"
        f"  ver={r.verified!s:5s}  conf={r.confidence:.3f}"
        f"  shards={r.shards_recovered:2d}/{K_SHARDS}"
        f"  crc={r.shard_crc_ratio:.2f}"
        f"  geo_err={r.geo_angle_error:4.1f}deg"
        f"  rt={r.runtime_total:.2f}s"
        f"{geo_flag}{budget_flag}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ROC computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_roc(
    positive_results: list[CaseResult],
    negative_results: list[CaseResult],
) -> float:
    pos_scores = [r.confidence for r in positive_results]
    neg_scores = [r.confidence for r in negative_results]
    if not pos_scores or not neg_scores:
        return 0.0
    roc = generate_roc_data(pos_scores, neg_scores)
    return roc.auc


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P5-V2 Validation Harness")
    parser.add_argument("--fast",     action="store_true", help="Reduced sweep for CI")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib output")
    parser.add_argument("--seeds",    type=int, default=3 if "--fast" in sys.argv else 1,
                        help="Number of image seeds to average over")
    args = parser.parse_args()

    print("=" * 72)
    print("  MIST P5-V2 VALIDATION + PROFILING HARNESS")
    print("=" * 72)

    priv_key, pub_key = generate_keys()
    embed_key = b"mist-p5v2-harness-2026"
    n_seeds = args.seeds

    all_results:  list[CaseResult] = []
    fp_results:   list[CaseResult] = []

    # ── Embed + run attack suite for each seed ────────────────────────────
    for seed in range(n_seeds):
        print(f"\n── Seed {seed} ──────────────────────────────────────────────────")
        img = make_test_image(512, 512, seed=seed)
        print(f"  Embedding 512×512 (seed={seed})…", end=" ", flush=True)
        t_emb = time.perf_counter()
        wm = watermark_p5(img, 1001 + seed, 2001 + seed, priv_key, embed_key)
        print(f"{time.perf_counter() - t_emb:.3f}s")

        suite = build_attack_suite(wm, fast=args.fast)
        print(f"  Running {len(suite)} attack cases (parallel, 2 workers)…")

        # Run attack cases in parallel — each is independent.
        # 2 workers: half the GPU contention vs 4, still 2× throughput.
        def _run_one(args_tuple):
            attacked, atype, aparam, true_a, true_s = args_tuple
            return run_case(attacked, pub_key, embed_key, atype, aparam, true_a, true_s)

        ordered: list[CaseResult] = [None] * len(suite)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_to_idx = {
                pool.submit(_run_one, item): i
                for i, item in enumerate(suite)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ordered[idx] = future.result()
                except Exception as exc:
                    import traceback
                    _, atype, aparam, true_a, true_s = suite[idx]
                    r = CaseResult(attack_type=atype, attack_param=aparam,
                                   true_angle=true_a, true_scale=true_s)
                    r.stage_budget_fail = f"EXCEPTION: {exc}"
                    ordered[idx] = r
                    print(f"  [ERR ] {atype:<10} {aparam:<20} EXCEPTION: {exc}")
                    traceback.print_exc()
                completed += 1
                if completed % 4 == 0 or completed == len(suite):
                    print(f"    [{completed}/{len(suite)} done]", flush=True)

        for r in ordered:
            all_results.append(r)
            print_case(r)

        # FP check (only seed 0 to save time)
        if seed == 0:
            fp_n = 5 if args.fast else 10
            print(f"\n  False-positive check ({fp_n} clean images)…")
            try:
                fp_results = run_fp_suite(pub_key, embed_key, n=fp_n)
                for r in fp_results:
                    print_case(r)
            except Exception as exc:
                import traceback
                print(f"  [ERR ] FP suite crashed: {exc}")
                traceback.print_exc()

    # ── Aggregate metrics ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  AGGREGATE RESULTS")
    print("=" * 72)

    wm_results = [r for r in all_results if r.attack_type != "clean_fp"]

    by_type: dict[str, list[CaseResult]] = {}
    for r in wm_results:
        by_type.setdefault(r.attack_type, []).append(r)

    for atype, group in sorted(by_type.items()):
        det_rate = np.mean([r.detected for r in group])
        ver_rate = np.mean([r.verified for r in group])
        mean_rt  = np.mean([r.runtime_total for r in group])
        mean_geo = np.mean([r.geo_angle_error for r in group])
        n_div    = sum(r.geo_diverge for r in group)
        n_bud    = sum(bool(r.stage_budget_fail) for r in group)
        print(
            f"  {atype:12s}  det={det_rate:.0%}  ver={ver_rate:.0%}"
            f"  rt={mean_rt:.2f}s  geo_err={mean_geo:.1f}deg"
            f"  divg={n_div}  budget_fail={n_bud}"
        )

    total_det = np.mean([r.detected for r in wm_results])
    total_ver = np.mean([r.verified for r in wm_results])
    total_rt  = np.mean([r.runtime_total for r in wm_results])
    fp_rate   = np.mean([r.verified for r in fp_results]) if fp_results else 0.0
    roc_auc   = compute_roc(wm_results, fp_results) if fp_results else 0.0

    fn_rate = 1.0 - total_det  # missed detections on watermarked images
    fp_pct  = fp_rate * 100.0

    print()
    print(f"  Detection rate  : {total_det:.1%}  ({sum(r.detected for r in wm_results)}/{len(wm_results)})")
    print(f"  Verification rate: {total_ver:.1%}  ({sum(r.verified for r in wm_results)}/{len(wm_results)})")
    print(f"  False positive  : {fp_pct:.1f}%  ({sum(r.verified for r in fp_results)}/{len(fp_results)})")
    print(f"  False negative  : {fn_rate:.1%}")
    print(f"  ROC AUC         : {roc_auc:.4f}")
    print(f"  Mean runtime    : {total_rt:.2f}s")

    # ── Stop condition evaluation ──────────────────────────────────────────
    print()
    print("─" * 50)
    print("  STOP CONDITIONS")
    print("─" * 50)

    cond_rt    = total_rt < TOTAL_RUNTIME_BUDGET
    cond_ver   = total_ver >= VERIFY_RATE_THRESHOLD
    cond_geo   = not any(r.geo_diverge for r in wm_results)
    cond_stage = not any(r.stage_budget_fail for r in wm_results)
    cond_roc   = roc_auc > 0.9

    def flag(ok): return "PASS" if ok else "FAIL"

    print(f"  [{flag(cond_rt)}]  Mean runtime < {TOTAL_RUNTIME_BUDGET}s      : {total_rt:.2f}s")
    print(f"  [{flag(cond_ver)}]  Verification > {VERIFY_RATE_THRESHOLD:.0%}          : {total_ver:.1%}")
    print(f"  [{flag(cond_geo)}]  No geometry divergence (>{GEO_DIVERGE_DEG}°)  : {'ok' if cond_geo else 'DIVERGED'}")
    print(f"  [{flag(cond_stage)}]  All stages within budget          : {'ok' if cond_stage else 'OVERRUN'}")
    print(f"  [{flag(cond_roc)}]  ROC AUC > 0.9                     : {roc_auc:.4f}")

    overall_pass = cond_rt and cond_ver and cond_geo
    overall_label = "PASS" if overall_pass else "FAIL"
    print()
    print(f"  >>> OVERALL: {overall_label} <<<")

    # ── Detailed stage timing summary ──────────────────────────────────────
    print()
    print("─" * 50)
    print("  STAGE TIMING (median across all cases)")
    print("─" * 50)
    stage_map = {
        "t_fourier_mellin": ("0_fourier_mellin",  "FM sync          "),
        "t_sync_template":  ("0b_sync_template",  "Sync template    "),
        "t_identity":       ("1_identity",         "Identity P4      "),
        "t_generate":       ("2_generate",         "Generate cands   "),
        "t_canary":         ("3_canary",            "Canary GPU       "),
        "t_promote":        ("4_promote",           "Promote+FM-direct"),
    }
    for attr, (stage_key, label) in stage_map.items():
        vals = [getattr(r, attr) for r in wm_results if getattr(r, attr) > 0]
        if not vals:
            continue
        med = np.median(vals)
        p95 = np.percentile(vals, 95)
        budget = STAGE_BUDGETS.get(stage_key, 99.0)
        status = "PASS" if med < budget else "FAIL"
        print(f"  [{status}]  {label}  med={med:.3f}s  p95={p95:.3f}s  budget={budget}s")

    # ── Write CSV ──────────────────────────────────────────────────────────
    all_csv = all_results + fp_results
    write_csv(all_csv, OUTPUTS / "p5v2_report.csv")

    # ── Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        plot_timing_histogram(wm_results, OUTPUTS / "p5v2_timing.png")
        plot_attack_heatmap(wm_results, OUTPUTS / "p5v2_attack_heatmap.png")

    print()
    print("=" * 72)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
