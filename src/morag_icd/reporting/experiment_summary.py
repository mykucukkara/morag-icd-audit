"""
Experiment summary utilities.

Aggregates results from multiple runs and generates a human-readable
experiment summary with failure analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def generate_failure_analysis(
    individual_metrics: Dict[str, Dict],
    output_path: str | Path,
) -> None:
    """
    Generate a failure analysis Markdown report.

    Parameters
    ----------
    individual_metrics : dict
        Per-run metrics from consolidated_metrics.json.
    output_path : str | Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Failure Analysis Report",
        "",
        "| Run | Failed Rate | JSON Err | CUDA OOM | Weak Evid | Ambiguous |",
        "|----|------------|----------|----------|-----------|-----------|",
    ]

    for run_id, metrics in sorted(individual_metrics.items()):
        errors = metrics.get("errors", {})
        rel = metrics.get("reliability", {})
        lines.append(
            f"| {run_id} | "
            f"{errors.get('failed_sample_rate', 0):.3f} | "
            f"{errors.get('json_parse_error_rate', 0):.3f} | "
            f"{errors.get('cuda_oom_rate', 0):.3f} | "
            f"{rel.get('weak_evidence_rate', 0):.3f} | "
            f"{rel.get('ambiguous_code_rate', 0):.3f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Failure analysis saved: {output_path}")


def generate_experiment_summary_json(
    seed_stats: Dict[str, Dict],
    output_path: str | Path,
) -> None:
    """
    Write a JSON summary of all experiment metrics for external tools.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    for exp_id, stats in seed_stats.items():
        summary[exp_id] = {
            "micro_f1_mean": stats.get("micro_f1", {}).get("mean"),
            "micro_f1_std": stats.get("micro_f1", {}).get("std"),
            "macro_f1_mean": stats.get("macro_f1", {}).get("mean"),
            "evidence_support_mean": stats.get("evidence_support_rate", {}).get("mean"),
            "unsupported_rate_mean": stats.get("unsupported_code_rate", {}).get("mean"),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Experiment summary JSON saved: {output_path}")
