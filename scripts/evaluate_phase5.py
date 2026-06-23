"""
scripts/evaluate_phase5.py  –  Phase 5 Evaluation Suite

Generates:
  - ROC curve with AUC
  - Precision-Recall curve + EER
  - Confidence histograms (positive vs negative)
  - Runtime distribution analysis
  - Per-attack-type breakdown

Usage:
    python -m scripts.evaluate_phase5 --images-dir data/test --key-file key.bin
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.wm_engine_p5 import detect_p5, embed_p5, K_SHARDS
from src.core.p5_profiler import P5Profile
from src.core.forensic import forensic_report


@dataclass
class EvalResult:
    """Result from evaluating one image."""
    filename: str
    is_watermarked: bool  # ground truth
    detected: bool
    confidence: float
    runtime_s: float
    attack_type: str = "none"
    inner_codeword_found: bool = False
    method: str = ""
    shard_count: int = 0
    p4_calls: int = 0


def evaluate_image(
    image_path: str,
    key: bytes,
    is_watermarked: bool,
    attack_type: str = "none",
) -> EvalResult:
    """Run detection on a single image and return eval result."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    with P5Profile() as prof:
        t0 = time.perf_counter()
        result = detect_p5(image, key)
        runtime = time.perf_counter() - t0

    geo = result.get("geometry", {}) or {}

    return EvalResult(
        filename=os.path.basename(image_path),
        is_watermarked=is_watermarked,
        detected=result["detected"],
        confidence=result["confidence"],
        runtime_s=runtime,
        attack_type=attack_type,
        inner_codeword_found=result.get("inner_codeword") is not None,
        method=geo.get("method", "direct"),
        shard_count=geo.get("shard_count", 0),
        p4_calls=prof.p4_call_count,
    )


def compute_roc(results: list[EvalResult]) -> dict:
    """Compute ROC curve data and AUC."""
    labels = np.array([1 if r.is_watermarked else 0 for r in results])
    scores = np.array([r.confidence for r in results])

    thresholds = np.sort(np.unique(scores))[::-1]
    tpr_list = [0.0]
    fpr_list = [0.0]
    n_pos = max(1, labels.sum())
    n_neg = max(1, len(labels) - labels.sum())

    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # AUC via trapezoidal rule
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    sorted_idx = np.argsort(fpr_arr)
    auc = float(np.trapz(tpr_arr[sorted_idx], fpr_arr[sorted_idx]))

    return {
        "tpr": tpr_arr.tolist(),
        "fpr": fpr_arr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": auc,
    }


def compute_pr(results: list[EvalResult]) -> dict:
    """Compute Precision-Recall curve."""
    labels = np.array([1 if r.is_watermarked else 0 for r in results])
    scores = np.array([r.confidence for r in results])
    thresholds = np.sort(np.unique(scores))[::-1]

    precision_list = []
    recall_list = []
    n_pos = max(1, labels.sum())

    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / n_pos
        precision_list.append(prec)
        recall_list.append(rec)

    return {
        "precision": precision_list,
        "recall": recall_list,
        "thresholds": thresholds.tolist(),
    }


def compute_eer(roc: dict) -> float:
    """Compute Equal Error Rate from ROC data."""
    tpr = np.array(roc["tpr"])
    fpr = np.array(roc["fpr"])
    fnr = 1.0 - tpr
    # EER is where FPR ≈ FNR
    diffs = np.abs(fpr - fnr)
    idx = np.argmin(diffs)
    return float((fpr[idx] + fnr[idx]) / 2.0)


def runtime_stats(results: list[EvalResult]) -> dict:
    """Compute runtime distribution statistics."""
    times = [r.runtime_s for r in results]
    if not times:
        return {}
    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "total": float(np.sum(times)),
    }


def per_attack_breakdown(results: list[EvalResult]) -> dict:
    """Breakdown metrics by attack type."""
    attacks = set(r.attack_type for r in results)
    breakdown = {}
    for atk in sorted(attacks):
        atk_results = [r for r in results if r.attack_type == atk]
        pos = [r for r in atk_results if r.is_watermarked]
        neg = [r for r in atk_results if not r.is_watermarked]
        tp = sum(1 for r in pos if r.inner_codeword_found)
        fn = len(pos) - tp
        fp = sum(1 for r in neg if r.detected)
        tn = len(neg) - fp
        breakdown[atk] = {
            "total": len(atk_results),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": tp / max(1, tp + fn),
            "fpr": fp / max(1, fp + tn),
            "avg_confidence_pos": float(np.mean([r.confidence for r in pos])) if pos else 0.0,
            "avg_confidence_neg": float(np.mean([r.confidence for r in neg])) if neg else 0.0,
            "avg_runtime": float(np.mean([r.runtime_s for r in atk_results])),
            "avg_p4_calls": float(np.mean([r.p4_calls for r in atk_results])),
        }
    return breakdown


def save_report(results: list[EvalResult], output_dir: str):
    """Generate and save full evaluation report."""
    os.makedirs(output_dir, exist_ok=True)

    roc = compute_roc(results)
    pr = compute_pr(results)
    eer = compute_eer(roc)
    rt = runtime_stats(results)
    breakdown = per_attack_breakdown(results)

    report = {
        "summary": {
            "total_images": len(results),
            "watermarked": sum(1 for r in results if r.is_watermarked),
            "clean": sum(1 for r in results if not r.is_watermarked),
            "roc_auc": roc["auc"],
            "eer": eer,
        },
        "runtime": rt,
        "per_attack": breakdown,
    }

    # Save JSON report
    with open(os.path.join(output_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save raw results
    raw = [asdict(r) for r in results]
    with open(os.path.join(output_dir, "eval_raw.json"), "w") as f:
        json.dump(raw, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  Phase 5 Evaluation Report")
    print("=" * 60)
    print(f"  Total images:      {report['summary']['total_images']}")
    print(f"  Watermarked:       {report['summary']['watermarked']}")
    print(f"  Clean:             {report['summary']['clean']}")
    print(f"  ROC AUC:           {roc['auc']:.4f}")
    print(f"  EER:               {eer:.4f}")
    print(f"  Avg runtime:       {rt.get('mean', 0):.2f}s")
    print(f"  P95 runtime:       {rt.get('p95', 0):.2f}s")
    print(f"  Total runtime:     {rt.get('total', 0):.1f}s")
    print()

    for atk, stats in breakdown.items():
        print(f"  [{atk}] TPR={stats['tpr']:.2f}  FPR={stats['fpr']:.2f}  "
              f"avg_conf_pos={stats['avg_confidence_pos']:.3f}  "
              f"avg_p4={stats['avg_p4_calls']:.0f}")
    print("=" * 60)

    # Try to generate plots (optional — depends on matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ROC curve
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(roc["fpr"], roc["tpr"], "b-", linewidth=2,
                label=f"ROC (AUC={roc['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Phase 5 ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
        plt.close(fig)

        # Confidence histogram
        pos_conf = [r.confidence for r in results if r.is_watermarked]
        neg_conf = [r.confidence for r in results if not r.is_watermarked]
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        if pos_conf:
            ax.hist(pos_conf, bins=30, alpha=0.6, label="Watermarked", color="green")
        if neg_conf:
            ax.hist(neg_conf, bins=30, alpha=0.6, label="Clean", color="red")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Distribution")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "confidence_hist.png"), dpi=150)
        plt.close(fig)

        # Runtime distribution
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist([r.runtime_s for r in results], bins=30, alpha=0.7, color="steelblue")
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Count")
        ax.set_title("Runtime Distribution")
        fig.savefig(os.path.join(output_dir, "runtime_hist.png"), dpi=150)
        plt.close(fig)

        print(f"  Plots saved to {output_dir}/")

    except ImportError:
        print("  (matplotlib not available — plots skipped)")

    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Evaluation Suite")
    parser.add_argument("--images-dir", required=True,
                        help="Directory with test images")
    parser.add_argument("--key-file", required=True,
                        help="Path to embedding key file")
    parser.add_argument("--output-dir", default="eval_output",
                        help="Output directory for report")
    parser.add_argument("--watermarked-prefix", default="wm_",
                        help="Filename prefix for watermarked images")
    parser.add_argument("--clean-prefix", default="clean_",
                        help="Filename prefix for clean images")
    args = parser.parse_args()

    with open(args.key_file, "rb") as f:
        key = f.read()

    images_dir = Path(args.images_dir)
    results = []

    for img_path in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")):
        name = img_path.stem
        if name.startswith(args.watermarked_prefix):
            is_wm = True
            # Parse attack type from filename: wm_rotate_10_image.png
            parts = name.replace(args.watermarked_prefix, "").split("_")
            attack = parts[0] if parts else "none"
        elif name.startswith(args.clean_prefix):
            is_wm = False
            attack = "none"
        else:
            continue

        print(f"  Evaluating {img_path.name} (wm={is_wm}, attack={attack})...")
        try:
            r = evaluate_image(str(img_path), key, is_wm, attack)
            results.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")

    if not results:
        print("No images found. Check --images-dir and filename prefixes.")
        return

    save_report(results, args.output_dir)


if __name__ == "__main__":
    main()
