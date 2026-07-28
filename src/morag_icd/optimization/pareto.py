"""
Pareto front utilities for multi-objective optimization.

Provides:
- Non-dominated sorting (Pareto front identification)
- Hypervolume approximation (Monte Carlo)
- Knee point detection
- Best compromise solution selection
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution a dominates solution b (all objectives minimized)."""
    return bool(np.all(a <= b) and np.any(a < b))


def find_pareto_front(obj_matrix: np.ndarray) -> List[int]:
    """
    Find the indices of non-dominated solutions (Pareto front) in a minimization context.

    Parameters
    ----------
    obj_matrix : np.ndarray, shape (n_solutions, n_objectives)
        Objective values (all minimized).

    Returns
    -------
    list of int : indices of non-dominated solutions.
    """
    obj_matrix = np.asarray(obj_matrix, dtype=float)
    if obj_matrix.size == 0:
        return []
    if obj_matrix.ndim == 1:
        obj_matrix = obj_matrix.reshape(1, -1)
    if obj_matrix.ndim != 2:
        raise ValueError(f"Invalid objective matrix shape: expected 2D, got {obj_matrix.shape}")

    n = obj_matrix.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if dominates(obj_matrix[j], obj_matrix[i]):
                is_pareto[i] = False
                break

    return [i for i in range(n) if is_pareto[i]]


def fast_non_dominated_sort(obj_matrix: np.ndarray) -> List[List[int]]:
    """
    NSGA-II style fast non-dominated sorting.

    Objective values are interpreted in a minimization sense: lower values are
    better for every objective. Maximization objectives must be negated before
    being passed into this function.

    Returns a list of fronts, each front being a list of solution indices.
    """
    obj_matrix = np.asarray(obj_matrix, dtype=float)
    if obj_matrix.size == 0:
        return []
    if obj_matrix.ndim == 1:
        obj_matrix = obj_matrix.reshape(1, -1)
    if obj_matrix.ndim != 2:
        raise ValueError(f"Invalid objective matrix shape: expected 2D, got {obj_matrix.shape}")

    n = obj_matrix.shape[0]
    domination_count = np.zeros(n, dtype=int)
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(obj_matrix[i], obj_matrix[j]):
                dominated_solutions[i].append(j)
            elif dominates(obj_matrix[j], obj_matrix[i]):
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                if j < 0 or j >= n:
                    raise ValueError(f"Invalid Pareto index generated: idx={j}, n={n}")
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            for idx in next_front:
                if idx < 0 or idx >= n:
                    raise ValueError(f"Invalid Pareto index generated: idx={idx}, n={n}")
            fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(obj_matrix: np.ndarray, front_indices: List[int]) -> np.ndarray:
    """
    Compute crowding distance for solutions in a front.

    Returns array of crowding distances (larger = more diverse).
    """
    n = len(front_indices)
    if n == 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.full(n, np.inf)

    obj_matrix = np.asarray(obj_matrix, dtype=float)
    if obj_matrix.ndim == 1:
        obj_matrix = obj_matrix.reshape(1, -1)
    if obj_matrix.ndim != 2:
        raise ValueError(f"Invalid objective matrix shape: expected 2D, got {obj_matrix.shape}")
    if min(front_indices) < 0 or max(front_indices) >= obj_matrix.shape[0]:
        raise ValueError(f"Invalid Pareto index generated: idx outside [0, {obj_matrix.shape[0]})")

    front_objs = obj_matrix[front_indices]
    if front_objs.shape[0] != n:
        raise ValueError(
            f"Objective dimension mismatch for crowding distance: front={n}, sliced={front_objs.shape[0]}"
        )
    n_obj = front_objs.shape[1]
    if n_obj == 0:
        raise ValueError("Invalid objective matrix shape: zero objective dimensions")
    distances = np.zeros(n)

    for m in range(n_obj):
        sorted_idx = np.argsort(front_objs[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        obj_range = front_objs[sorted_idx[-1], m] - front_objs[sorted_idx[0], m]
        if obj_range == 0:
            continue
        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                front_objs[sorted_idx[i + 1], m] - front_objs[sorted_idx[i - 1], m]
            ) / obj_range

    return distances


def hypervolume_mc(
    obj_matrix: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
    n_samples: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo approximation of hypervolume indicator.

    Parameters
    ----------
    obj_matrix : np.ndarray, shape (n_solutions, n_objectives)
        Non-dominated solutions (all minimized).
    reference_point : np.ndarray, shape (n_objectives,)
        Reference point (should be dominated by all solutions).
        Defaults to 1.1 * max over each objective.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int
        Random seed.

    Returns
    -------
    float : estimated hypervolume.
    """
    if len(obj_matrix) == 0:
        return 0.0

    if reference_point is None:
        reference_point = obj_matrix.max(axis=0) * 1.1

    ideal = obj_matrix.min(axis=0)
    rng = np.random.default_rng(seed)

    # Sample uniformly in [ideal, reference_point]
    ranges = reference_point - ideal
    if np.any(ranges <= 0):
        return 0.0

    samples = rng.uniform(ideal, reference_point, size=(n_samples, obj_matrix.shape[1]))

    # A sample is dominated (i.e., in the hypervolume) if any solution dominates it
    dominated = np.zeros(n_samples, dtype=bool)
    for sol in obj_matrix:
        dominated |= np.all(samples >= sol, axis=1)

    vol = np.prod(ranges) * np.mean(dominated)
    return float(vol)


def detect_knee_point(obj_matrix: np.ndarray, pareto_indices: List[int]) -> int:
    """
    Detect the knee point on the Pareto front using a simple heuristic:
    the solution with the minimum sum of normalized objective values.

    Returns the index into obj_matrix (not into pareto_indices).
    """
    if not pareto_indices:
        return 0

    front = obj_matrix[pareto_indices]
    mins = front.min(axis=0)
    maxs = front.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    normalized = (front - mins) / ranges
    scores = normalized.sum(axis=1)
    best_local = int(np.argmin(scores))
    return pareto_indices[best_local]


def select_best_compromise(
    obj_matrix: np.ndarray,
    pareto_indices: List[int],
) -> int:
    """
    Select the best compromise solution from the Pareto front.
    Same as knee point detection: minimizes sum of normalized objectives.
    """
    return detect_knee_point(obj_matrix, pareto_indices)
