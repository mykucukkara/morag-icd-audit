"""
Script 10: Evaluate prediction results and compute all metrics.

Reads JSONL prediction files from the predictions directory,
computes full metric suites (classification, reliability, hierarchical, cost),
and saves results to the metrics directory.

Metrics are computed over a FIXED Top-N label space (P1-1) so micro/macro-F1 are
comparable across experiments and seeds, and seed statistics are aggregated for ALL
metric groups with sample std (ddof=1) (P1-2).

Usage:
    python scripts/10_evaluate_results.py \\
        --results-dir results/predictions \\
        --output-dir results/metrics \\
        [--top-n 50] [--splits-root data/splits] [--bootstrap] [--split test]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morag_icd.evaluation.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MORAG-ICD prediction results.")
    p.add_argument("--results-dir", default="results/predictions", help="Directory with prediction JSONL files")
    p.add_argument("--output-dir", default="results/metrics", help="Directory to save metrics")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--split", default="test")
    p.add_argument("--splits-root", default="data/splits",
                   help="Root of split files (topN/{train,validation,test}.jsonl) for the fixed label space")
    p.add_argument("--label-set", default=None,
                   help="Optional explicit path to a JSON list of the Top-N codes (overrides derivation)")
    p.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CIs (slow)")
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_or_derive_label_set(top_n: int, splits_root: str, explicit: Optional[str]) -> List[str]:
    """Return the fixed Top-N ICD code vocabulary (PHI-safe: reads only gold_codes).

    Priority: explicit path -> cached topN/label_set.json -> derive from the union of
    gold_codes across the split files (and cache it). The Top-N filter guarantees this
    union equals the Top-N vocabulary.
    """
    if explicit:
        codes = json.loads(Path(explicit).read_text(encoding="utf-8"))
        return sorted({str(c) for c in codes})

    split_dir = Path(splits_root) / f"top{top_n}"
    cache = split_dir / "label_set.json"
    if cache.exists():
        codes = json.loads(cache.read_text(encoding="utf-8"))
        return sorted({str(c) for c in codes})

    labels: set = set()
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        fp = split_dir / name
        if not fp.exists():
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for c in rec.get("gold_codes", []) or []:
                    labels.add(str(c))
    label_set = sorted(labels)
    if label_set:
        try:
            cache.write_text(json.dumps(label_set, indent=2), encoding="utf-8")
        except Exception:
            pass
    return label_set


def _flatten_numeric(metrics: Dict, groups=("classification", "reliability", "hierarchical", "cost", "errors")) -> Dict[str, float]:
    """Flatten numeric leaves of the selected metric groups to 'group.metric' -> value."""
    flat: Dict[str, float] = {}
    for g in groups:
        block = metrics.get(g)
        if not isinstance(block, dict):
            continue
        for k, v in block.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and math.isfinite(v):
                flat[f"{g}.{k}"] = float(v)
    return flat


def aggregate_seeds(flat_per_seed: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """mean / sample-std (ddof=1) / min / max / n per metric across seeds."""
    keys: set = set()
    for d in flat_per_seed:
        keys.update(d.keys())
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(keys):
        vals = [d[k] for d in flat_per_seed if k in d]
        n = len(vals)
        if n == 0:
            continue
        mean = sum(vals) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in vals) / (n - 1)  # sample variance (ddof=1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out[k] = {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "n": n}
    return out


def main():
    args = parse_args()

    results_dir = Path(args.results_dir) / f"top{args.top_n}"
    output_dir = Path(args.output_dir) / f"top{args.top_n}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    label_set = load_or_derive_label_set(args.top_n, args.splits_root, args.label_set)
    print(f"Fixed label space: {len(label_set)} Top-{args.top_n} codes"
          + ("" if label_set else " (WARNING: empty — falling back to per-file inference)"))

    evaluator = Evaluator(output_dir=output_dir)

    pred_pattern = f"**/{args.split}_predictions.jsonl"
    pred_files = sorted(results_dir.glob(pred_pattern))
    if not pred_files:
        print(f"No prediction files found in {results_dir} matching '{pred_pattern}'")
        sys.exit(1)

    print(f"Found {len(pred_files)} prediction files to evaluate")

    all_metrics: dict = {}
    for pred_path in pred_files:
        exp_seed_dir = pred_path.parent.name  # e.g. "E11_seed42"
        print(f"  Evaluating: {exp_seed_dir}...")
        try:
            metrics = evaluator.evaluate(
                predictions_path=pred_path,
                label_set=label_set or None,
                run_bootstrap=args.bootstrap,
                bootstrap_n=args.bootstrap_n,
                seed=args.seed,
            )
            all_metrics[exp_seed_dir] = metrics
        except Exception as e:
            print(f"  Error evaluating {exp_seed_dir}: {e}")
            all_metrics[exp_seed_dir] = {"error": str(e)}

    # Group by experiment across seeds; aggregate ALL metric groups (P1-2).
    exp_groups: dict = {}
    for key, metrics in all_metrics.items():
        exp_id = key.split("_seed")[0]
        if "error" in metrics and len(metrics) == 1:
            continue
        exp_groups.setdefault(exp_id, []).append(_flatten_numeric(metrics))

    seed_stats = {exp_id: aggregate_seeds(flats) for exp_id, flats in exp_groups.items()}

    consolidated = {
        "top_n": args.top_n,
        "label_set_size": len(label_set),
        "fixed_label_space": bool(label_set),
        "individual": all_metrics,
        "seed_statistics": seed_stats,
    }
    consolidated_path = output_dir / "consolidated_metrics.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, default=str)

    print("\n=== Evaluation Complete ===")
    print(f"Evaluated: {len(all_metrics)} runs | fixed label space: {len(label_set)} codes")
    print(f"Consolidated metrics: {consolidated_path}")

    print("\n=== Micro-F1 Summary (mean ± std, sample std) ===")
    for exp_id, stats in sorted(seed_stats.items()):
        mf1 = stats.get("classification.micro_f1", {})
        print(f"  {exp_id:8s}: micro_f1 = {mf1.get('mean', 0):.4f} ± {mf1.get('std', 0):.4f} (n={mf1.get('n', 0)})")


if __name__ == "__main__":
    main()
