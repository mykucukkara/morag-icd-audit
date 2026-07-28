"""
Similar-code confusion analysis for ICD-10 predictions.

Identifies common confusion pairs and computes a confusion matrix at the
parent (3-char prefix) level.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def build_confusion_pairs(
    predictions: List[List[str]],
    gold: List[List[str]],
) -> List[Tuple[str, str, int]]:
    """
    Identify (predicted_code, gold_code) confusion pairs where the prediction
    is wrong but shares the same 3-char ICD parent as a gold code.

    Returns
    -------
    list of (pred_code, gold_code, count) sorted by count descending.
    """
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for pred_list, gold_list in zip(predictions, gold):
        gold_set = set(gold_list)
        gold_by_parent: Dict[str, List[str]] = defaultdict(list)
        for g in gold_list:
            gold_by_parent[g[:3]].append(g)

        for code in pred_list:
            if code not in gold_set:
                parent = code[:3]
                if parent in gold_by_parent:
                    for true_code in gold_by_parent[parent]:
                        pair_counts[(code, true_code)] += 1

    result = [(pred, gold_c, cnt) for (pred, gold_c), cnt in pair_counts.items()]
    result.sort(key=lambda x: -x[2])
    return result


def compute_parent_confusion_matrix(
    predictions: List[List[str]],
    gold: List[List[str]],
    top_parents: int = 20,
) -> Dict:
    """
    Build a confusion matrix at the 3-char ICD parent level.

    Returns
    -------
    dict with keys:
        - 'parents': list of parent codes (axis labels)
        - 'matrix': 2D numpy array (true x predicted)
        - 'top_confusions': list of (true_parent, pred_parent, count)
    """
    # Collect all parent codes
    all_parents: set = set()
    for codes in predictions + gold:
        for c in codes:
            all_parents.add(c[:3])

    # Limit to top N by gold frequency
    gold_parent_freq: Dict[str, int] = defaultdict(int)
    for gold_list in gold:
        for c in gold_list:
            gold_parent_freq[c[:3]] += 1

    top_parent_list = sorted(gold_parent_freq, key=lambda x: -gold_parent_freq[x])[:top_parents]
    parent_idx = {p: i for i, p in enumerate(top_parent_list)}
    n = len(top_parent_list)

    matrix = np.zeros((n, n), dtype=int)

    for pred_list, gold_list in zip(predictions, gold):
        true_parents = {c[:3] for c in gold_list if c[:3] in parent_idx}
        pred_parents = {c[:3] for c in pred_list if c[:3] in parent_idx}

        for tp in true_parents:
            for pp in pred_parents:
                matrix[parent_idx[tp], parent_idx[pp]] += 1

    # Extract top confusion pairs (off-diagonal)
    top_confusions = []
    for i, tp in enumerate(top_parent_list):
        for j, pp in enumerate(top_parent_list):
            if i != j and matrix[i, j] > 0:
                top_confusions.append((tp, pp, int(matrix[i, j])))
    top_confusions.sort(key=lambda x: -x[2])

    return {
        "parents": top_parent_list,
        "matrix": matrix,
        "top_confusions": top_confusions[:20],
    }
