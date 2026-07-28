"""
Multi-objective objective function for RAG hyperparameter optimization.

Evaluates a given hyperparameter config on a validation subset and returns
objective values for use by NSGA-II, MOPSO, and Random Search.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Objective directions (True = maximize → negate for minimizer)
#
# NOTE: `unsupported_code_rate` is deliberately NOT an objective. Every predicted code
# increments exactly one of supported_count / unsupported_count (evidence_metrics.py), so
# UCR == 1 - ESR *identically*. Registering both made the Pareto space degenerate: the two
# axes were perfectly anti-correlated, which (a) double-weighted evidence support against
# accuracy and runtime, and (b) pushed the front to 6 objectives, past the point where
# NSGA-II's non-dominated sorting still discriminates. UCR is still computed and reported
# (see evaluate_config below) — it is just not a separate search dimension.
OBJECTIVES = [
    ("micro_f1", True),           # maximize
    ("macro_f1", True),           # maximize
    ("evidence_support_rate", True),  # maximize
    ("similar_code_confusion_rate", False),  # minimize
    ("avg_runtime_sec", False),   # minimize
]


def validate_objective_vector(obj_vector: Any) -> np.ndarray:
    """
    Validate that an objective vector is a finite 1D numeric vector with the expected size.
    """
    arr = np.asarray(obj_vector, dtype=float)
    expected_size = len(OBJECTIVES)
    if arr.ndim != 1:
        raise ValueError(f"Invalid objective vector shape: expected 1D with size {expected_size}, got {arr.shape}")
    if arr.size != expected_size:
        raise ValueError(f"Invalid objective vector size: expected {expected_size}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Invalid objective vector values: non-finite values present")
    return arr


def evaluate_config(
    config: Dict[str, Any],
    pipeline_factory,
    dataset: List[Dict],
    label_set: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run the full pipeline with a given config on a dataset subset and
    compute all objective metrics.

    Parameters
    ----------
    config : dict
        Hyperparameter configuration (from SearchSpace.decode).
    pipeline_factory : callable
        factory(config) -> pipeline object with .process_note(text) method.
    dataset : list of dict
        List of samples with keys 'text', 'gold_codes', 'subject_id', 'hadm_id'.
    label_set : list of str, optional
        Fixed label universe for metric computation.

    Returns
    -------
    dict with objective metric values.
    """
    from ..evaluation.metrics import compute_classification_metrics
    from ..evaluation.evidence_metrics import compute_reliability_metrics, compute_similar_code_confusion_rate
    from ..evaluation.cost_metrics import compute_cost_metrics

    try:
        pipeline = pipeline_factory(config)
    except Exception as e:
        return _failed_objectives(reason=str(e))

    predictions = []
    runtimes = []

    for sample in dataset:
        t0 = time.time()
        try:
            pred_codes = pipeline.process_note(sample["text"])
        except Exception:
            pred_codes = []
        elapsed = time.time() - t0
        runtimes.append(elapsed)

        predictions.append({
            "hadm_id": sample.get("hadm_id", ""),
            "gold_codes": sample.get("gold_codes", []),
            "predicted_codes": pred_codes if isinstance(pred_codes, list) else [],
            "runtime_sec": elapsed,
        })

    # Extract code lists
    pred_code_lists = [
        [cp.get("code", cp) if isinstance(cp, dict) else cp for cp in p["predicted_codes"]]
        for p in predictions
    ]
    gold_code_lists = [p["gold_codes"] for p in predictions]

    # Compute metrics
    clf = compute_classification_metrics(pred_code_lists, gold_code_lists, label_set=label_set)
    rel = compute_reliability_metrics(predictions)

    gold_dicts = [{"hadm_id": p["hadm_id"], "gold_codes": p["gold_codes"]} for p in predictions]
    similar_confusion = compute_similar_code_confusion_rate(predictions, gold_dicts)

    cost = compute_cost_metrics(predictions)

    return {
        "micro_f1": clf.get("micro_f1", 0.0),
        "macro_f1": clf.get("macro_f1", 0.0),
        "precision_at_5": clf.get("precision_at_5", 0.0),
        # ESR/UCR are None for systems that make no evidence claim (never the case for the
        # RAG pipeline being optimized, but coerce so a None can't reach the objective vector).
        # Use explicit None checks, NOT `or`: in evidence-constraint filter mode UCR is a
        # legitimate 0.0 (all surviving codes supported), and `0.0 or 1.0` would flip that
        # best-case value to the worst case in the returned/logged metrics.
        "evidence_support_rate": (lambda v: 0.0 if v is None else v)(rel.get("evidence_support_rate")),
        "unsupported_code_rate": (lambda v: 1.0 if v is None else v)(rel.get("unsupported_code_rate")),
        "similar_code_confusion_rate": similar_confusion,
        "avg_runtime_sec": cost.get("avg_runtime_sec", 0.0),
        "avg_input_tokens": cost.get("avg_input_tokens", 0.0),
    }


def objectives_to_vector(metrics: Dict[str, float]) -> np.ndarray:
    """
    Convert metrics dict to an objective vector for multi-objective optimizers.

    All objectives are framed as minimization (maximized objectives are negated).
    """
    vec = []
    for key, maximize in OBJECTIVES:
        val = metrics.get(key)
        # An objective can be None (e.g. evidence_support_rate is N/A for a system that makes
        # no evidence claim). Treat missing/None as the neutral 0.0 rather than crashing on -None.
        if val is None:
            val = 0.0
        if maximize:
            vec.append(-val)  # negate for minimization
        else:
            vec.append(val)
    return np.array(vec, dtype=float)


def failed_objective_vector(reason: str = "") -> np.ndarray:
    """Return a finite penalty objective vector for failed evaluations."""
    return objectives_to_vector(_failed_objectives(reason))


def _failed_objectives(reason: str = "") -> Dict[str, float]:
    """Return worst-case objectives when pipeline fails."""
    return {
        "micro_f1": 0.0,
        "macro_f1": 0.0,
        "precision_at_5": 0.0,
        "evidence_support_rate": 0.0,
        "unsupported_code_rate": 1.0,
        "similar_code_confusion_rate": 1.0,
        "avg_runtime_sec": 9999.0,
        "avg_input_tokens": 99999.0,
        "_failed": True,
        "_reason": reason,
    }
