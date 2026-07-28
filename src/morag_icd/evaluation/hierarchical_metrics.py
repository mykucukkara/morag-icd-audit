"""
ICD-10 hierarchy-aware evaluation metrics.

These metrics evaluate predictions at the 3-character parent category level
and quantify whether errors stay within the correct ICD category.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from .metrics import compute_classification_metrics


def _to_parent(code: str, level: int = 3) -> str:
    """ICD-10 category (parent) at the given prefix length.

    Normalization is dot-insensitive so both dotted ("I50.9") and MIMIC dot-less
    ("I509") forms map to the same category ("I50"). Shared with the similar-code
    confusion metric so the two are consistent.
    """
    return str(code).replace(".", "").upper()[:level]


def compute_hierarchical_metrics(
    predictions: List[List[str]],
    gold: List[List[str]],
    label_set: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute parent-level (3-character prefix) classification metrics.

    Parameters
    ----------
    predictions : list of list of str
        Predicted ICD-10 codes per sample.
    gold : list of list of str
        Gold ICD-10 codes per sample.

    Returns
    -------
    dict with parent-level metrics and hierarchical error rate.
    """
    # Map predictions and gold to parent codes
    parent_preds = [[_to_parent(c) for c in p] for p in predictions]
    parent_gold = [[_to_parent(c) for c in g] for g in gold]

    # Deduplicate
    parent_preds = [list(set(p)) for p in parent_preds]
    parent_gold = [list(set(g)) for g in parent_gold]

    parent_metrics = compute_classification_metrics(parent_preds, parent_gold, label_set=None)
    parent_metrics = {f"parent_{k}": v for k, v in parent_metrics.items()}

    # Hierarchical error rate: fraction of wrong predictions that share a parent with gold
    hierarchical_error_rate = _hierarchical_error_rate(predictions, gold)
    parent_metrics["hierarchical_error_rate"] = hierarchical_error_rate

    return parent_metrics


def _hierarchical_error_rate(
    predictions: List[List[str]],
    gold: List[List[str]],
) -> float:
    """
    Fraction of incorrect code predictions that are in the same 3-char ICD category
    as at least one gold code (i.e. 'almost right' errors).
    """
    total_wrong = 0
    same_parent_wrong = 0

    for pred_list, gold_list in zip(predictions, gold):
        gold_set = set(gold_list)
        gold_parents = {_to_parent(c) for c in gold_list}

        for code in pred_list:
            if code not in gold_set:
                total_wrong += 1
                if _to_parent(code) in gold_parents:
                    same_parent_wrong += 1

    if total_wrong == 0:
        return 0.0
    return same_parent_wrong / total_wrong
