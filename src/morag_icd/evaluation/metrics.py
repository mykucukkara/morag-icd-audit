"""
Multi-label classification metrics for ICD-10 code recommendation.

Computes standard metrics (micro/macro F1, Precision@k, Recall@k, Hamming Loss,
Exact Match, Label Coverage, LRAP) from binary label matrices.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    label_ranking_average_precision_score,
    coverage_error,
)


def _to_binary_matrix(
    predictions: List[List[str]],
    gold: List[List[str]],
    label_set: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lists of code strings to binary indicator matrices."""
    if label_set is None:
        all_labels: set = set()
        for codes in predictions + gold:
            all_labels.update(codes)
        label_set = sorted(all_labels)

    label_idx = {label: i for i, label in enumerate(label_set)}
    n = len(predictions)
    m = len(label_set)

    y_true = np.zeros((n, m), dtype=int)
    y_pred = np.zeros((n, m), dtype=int)

    for i, (pred_codes, gold_codes) in enumerate(zip(predictions, gold)):
        for code in pred_codes:
            if code in label_idx:
                y_pred[i, label_idx[code]] = 1
        for code in gold_codes:
            if code in label_idx:
                y_true[i, label_idx[code]] = 1

    return y_true, y_pred, label_set


def precision_at_k(predictions_ranked: List[List[str]], gold: List[List[str]], k: int) -> float:
    """Precision@k: average fraction of top-k predictions that are correct."""
    scores = []
    for pred, g in zip(predictions_ranked, gold):
        gold_set = set(g)
        top_k = pred[:k]
        if len(top_k) == 0:
            scores.append(0.0)
        else:
            hits = sum(1 for c in top_k if c in gold_set)
            scores.append(hits / len(top_k))
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(predictions_ranked: List[List[str]], gold: List[List[str]], k: int) -> float:
    """Recall@k: average fraction of gold codes found in top-k predictions."""
    scores = []
    for pred, g in zip(predictions_ranked, gold):
        gold_set = set(g)
        if len(gold_set) == 0:
            continue
        top_k = pred[:k]
        hits = sum(1 for c in top_k if c in gold_set)
        scores.append(hits / len(gold_set))
    return float(np.mean(scores)) if scores else 0.0


def compute_classification_metrics(
    predictions: List[List[str]],
    gold: List[List[str]],
    label_set: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all standard multi-label classification metrics.

    Parameters
    ----------
    predictions : list of list of str
        Predicted ICD-10 codes per sample (unranked or ranked).
    gold : list of list of str
        Gold-standard ICD-10 codes per sample.
    label_set : list of str, optional
        Fixed universe of labels. Inferred from data if not provided.

    Returns
    -------
    dict with all metrics.
    """
    if not predictions or not gold:
        return _empty_metrics()

    y_true, y_pred, label_set = _to_binary_matrix(predictions, gold, label_set)

    # Guard: all-zero predictions lead to division warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        h_loss = float(hamming_loss(y_true, y_pred))

    # Exact match ratio
    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))

    # Label coverage (fraction of labels with at least one prediction)
    label_coverage = float(np.mean(np.sum(y_pred, axis=0) > 0))

    # Precision@k and Recall@k
    p_at_5 = precision_at_k(predictions, gold, k=5)
    p_at_10 = precision_at_k(predictions, gold, k=10)
    r_at_5 = recall_at_k(predictions, gold, k=5)
    r_at_10 = recall_at_k(predictions, gold, k=10)

    # Label ranking average precision (if applicable)
    try:
        lrap = float(label_ranking_average_precision_score(y_true, y_pred))
    except Exception:
        lrap = 0.0

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "hamming_loss": h_loss,
        "exact_match_ratio": exact_match,
        "label_coverage": label_coverage,
        "precision_at_5": p_at_5,
        "precision_at_10": p_at_10,
        "recall_at_5": r_at_5,
        "recall_at_10": r_at_10,
        "lrap": lrap,
        "n_samples": len(predictions),
        "n_labels": len(label_set),
    }


def _empty_metrics() -> Dict[str, float]:
    """Return zero-valued metrics dict when no data is available."""
    return {
        "micro_f1": 0.0, "macro_f1": 0.0,
        "micro_precision": 0.0, "micro_recall": 0.0,
        "macro_precision": 0.0, "macro_recall": 0.0,
        "hamming_loss": 1.0, "exact_match_ratio": 0.0,
        "label_coverage": 0.0,
        "precision_at_5": 0.0, "precision_at_10": 0.0,
        "recall_at_5": 0.0, "recall_at_10": 0.0,
        "lrap": 0.0, "n_samples": 0, "n_labels": 0,
    }
