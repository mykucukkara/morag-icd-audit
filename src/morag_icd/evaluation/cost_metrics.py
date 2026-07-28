"""
Inference cost and efficiency metrics for MORAG-ICD experiments.

Tracks token usage, inference time, GPU memory, and cost-per-correct-code.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_cost_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """
    Compute inference cost and efficiency metrics from prediction records.

    Each prediction dict may contain:
        - runtime_sec (float)
        - token_count_input (int)
        - token_count_output (int)
        - gpu_memory_mb (float)  [optional]

    Returns
    -------
    dict with cost metrics.
    """
    if not predictions:
        return _empty_cost_metrics()

    runtimes = []
    input_tokens = []
    output_tokens = []
    gpu_mems = []
    correct_code_counts = []  # For cost-per-correct-code

    for pred in predictions:
        rt = pred.get("runtime_sec")
        if rt is not None:
            runtimes.append(float(rt))

        ti = pred.get("token_count_input")
        if ti is not None:
            input_tokens.append(int(ti))

        to_ = pred.get("token_count_output")
        if to_ is not None:
            output_tokens.append(int(to_))

        gm = pred.get("gpu_memory_mb")
        if gm is not None:
            gpu_mems.append(float(gm))

        # Count correct codes
        gold_set = set(pred.get("gold_codes", []))
        correct = sum(
            1 for cp in pred.get("predicted_codes", [])
            if cp.get("code") in gold_set
        )
        correct_code_counts.append(correct)

    metrics: Dict[str, float] = {}

    if runtimes:
        metrics["avg_runtime_sec"] = float(np.mean(runtimes))
        metrics["total_runtime_sec"] = float(np.sum(runtimes))
        metrics["p95_runtime_sec"] = float(np.percentile(runtimes, 95))
    else:
        metrics["avg_runtime_sec"] = 0.0
        metrics["total_runtime_sec"] = 0.0
        metrics["p95_runtime_sec"] = 0.0

    if input_tokens:
        metrics["avg_input_tokens"] = float(np.mean(input_tokens))
        metrics["total_input_tokens"] = float(np.sum(input_tokens))
    else:
        metrics["avg_input_tokens"] = 0.0
        metrics["total_input_tokens"] = 0.0

    if output_tokens:
        metrics["avg_output_tokens"] = float(np.mean(output_tokens))
        metrics["total_output_tokens"] = float(np.sum(output_tokens))
    else:
        metrics["avg_output_tokens"] = 0.0
        metrics["total_output_tokens"] = 0.0

    if gpu_mems:
        metrics["avg_gpu_memory_mb"] = float(np.mean(gpu_mems))
        metrics["peak_gpu_memory_mb"] = float(np.max(gpu_mems))
    else:
        metrics["avg_gpu_memory_mb"] = 0.0
        metrics["peak_gpu_memory_mb"] = 0.0

    # Cost per correct code: total runtime / total correct codes
    total_correct = sum(correct_code_counts)
    if total_correct > 0 and runtimes:
        metrics["runtime_per_correct_code"] = float(np.sum(runtimes)) / total_correct
    else:
        metrics["runtime_per_correct_code"] = 0.0

    # Token cost per correct code
    total_tokens = float(np.sum(input_tokens) + np.sum(output_tokens)) if (input_tokens or output_tokens) else 0.0
    if total_correct > 0 and total_tokens > 0:
        metrics["tokens_per_correct_code"] = total_tokens / total_correct
    else:
        metrics["tokens_per_correct_code"] = 0.0

    metrics["n_samples"] = len(predictions)
    return metrics


def compute_error_rate_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """
    Compute error and failure rate metrics from prediction records.

    Looks for error flags in each prediction:
        - 'error' key at sample level
        - 'failed' key at sample level
        - 'cuda_oom' key at sample level
        - 'json_parse_error' key in code-level predictions

    Returns
    -------
    dict with error rates.
    """
    if not predictions:
        return {}

    n = len(predictions)
    failed = sum(1 for p in predictions if p.get("failed") or p.get("error"))
    cuda_oom = sum(1 for p in predictions if p.get("cuda_oom"))
    json_errors = sum(
        1 for p in predictions
        for cp in p.get("predicted_codes", [])
        if cp.get("error") and "json" in str(cp.get("error", "")).lower()
    )
    skipped = sum(1 for p in predictions if p.get("skipped"))

    return {
        "failed_sample_rate": failed / n,
        "cuda_oom_rate": cuda_oom / n,
        "json_parse_error_rate": json_errors / max(n, 1),
        "skipped_rate": skipped / n,
        "success_rate": (n - failed - skipped) / n,
    }


def _empty_cost_metrics() -> Dict[str, float]:
    return {
        "avg_runtime_sec": 0.0, "total_runtime_sec": 0.0, "p95_runtime_sec": 0.0,
        "avg_input_tokens": 0.0, "total_input_tokens": 0.0,
        "avg_output_tokens": 0.0, "total_output_tokens": 0.0,
        "avg_gpu_memory_mb": 0.0, "peak_gpu_memory_mb": 0.0,
        "runtime_per_correct_code": 0.0, "tokens_per_correct_code": 0.0,
        "n_samples": 0,
    }
