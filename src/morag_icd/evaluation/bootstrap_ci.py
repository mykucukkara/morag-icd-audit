"""
Bootstrap confidence intervals for multi-label classification metrics.

Provides non-parametric confidence intervals for micro-F1 and other metrics
by resampling predictions with replacement.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .metrics import compute_classification_metrics


def bootstrap_ci(
    predictions: List[List[str]],
    gold: List[List[str]],
    metric_fn: Optional[Callable] = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    label_set: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for classification metrics.

    Parameters
    ----------
    predictions : list of list of str
        Predicted codes per sample.
    gold : list of list of str
        Gold codes per sample.
    metric_fn : callable, optional
        Custom metric function with signature fn(preds, gold) -> dict.
        Defaults to compute_classification_metrics.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.
    label_set : list of str, optional
        Fixed label universe.

    Returns
    -------
    dict mapping metric_name -> {'mean': float, 'lower': float, 'upper': float, 'std': float}
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)

    if metric_fn is None:
        def metric_fn(p, g):
            return compute_classification_metrics(p, g, label_set=label_set)

    # Collect bootstrap samples
    bootstrap_metrics: Dict[str, List[float]] = {}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_preds = [predictions[i] for i in indices]
        boot_gold = [gold[i] for i in indices]

        try:
            m = metric_fn(boot_preds, boot_gold)
        except Exception:
            continue

        for k, v in m.items():
            if isinstance(v, (int, float)):
                if k not in bootstrap_metrics:
                    bootstrap_metrics[k] = []
                bootstrap_metrics[k].append(float(v))

    alpha = 1.0 - ci_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1.0 - alpha / 2) * 100

    results = {}
    for metric_name, values in bootstrap_metrics.items():
        arr = np.array(values)
        results[metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "lower": float(np.percentile(arr, lower_pct)),
            "upper": float(np.percentile(arr, upper_pct)),
        }

    return results


def compute_seed_statistics(
    metrics_per_seed: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean ± std across multiple seeds.

    Parameters
    ----------
    metrics_per_seed : list of dicts
        One metrics dict per seed run.

    Returns
    -------
    dict mapping metric_name -> {'mean': float, 'std': float, 'min': float, 'max': float}
    """
    if not metrics_per_seed:
        return {}

    all_keys = set()
    for m in metrics_per_seed:
        all_keys.update(k for k, v in m.items() if isinstance(v, (int, float)))

    results = {}
    for k in all_keys:
        values = [m[k] for m in metrics_per_seed if k in m]
        arr = np.array(values, dtype=float)
        results[k] = {
            "mean": float(np.mean(arr)),
            # Sample std (ddof=1): these are 3 seeds drawn from the population of seeds, not
            # the population itself. ddof=0 understates the spread by ~18% at n=3, which is
            # exactly the "mean ± std" that goes into the manuscript tables.
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return results
