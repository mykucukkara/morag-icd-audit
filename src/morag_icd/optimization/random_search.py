"""
Random Search hyperparameter optimization (E15).

Samples random configurations from the search space, evaluates each on the
validation subset, and returns all results sorted by primary objective (micro_f1).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .search_space import SearchSpace
from .objective import evaluate_config, objectives_to_vector
from .pareto import find_pareto_front


class RandomSearch:
    """
    Random Search over the discrete hyperparameter space.

    For each of n_trials sampled configs, evaluates the pipeline on the
    validation subset and stores results. Returns Pareto-optimal solutions.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 100,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.seed = seed

    def run(
        self,
        pipeline_factory: Callable,
        dataset: List[Dict],
        label_set: Optional[List[str]] = None,
        output_path: Optional[str | Path] = None,
    ) -> Dict:
        """
        Run random search.

        Parameters
        ----------
        pipeline_factory : callable
            factory(config) -> pipeline.
        dataset : list of dict
            Validation subset samples.
        label_set : list of str, optional
        output_path : Path, optional
            Where to save results JSONL.

        Returns
        -------
        dict with keys:
            - 'all_trials': list of {config, metrics, obj_vector}
            - 'pareto_front': list of pareto-optimal trial dicts
            - 'best_compromise': single best compromise config
        """
        configs = self.search_space.sample_random_batch(self.n_trials, seed=self.seed)
        all_trials = []

        print(f"[RandomSearch] Running {self.n_trials} trials...")

        for i, config in enumerate(configs):
            t0 = time.time()
            metrics = evaluate_config(config, pipeline_factory, dataset, label_set)
            obj_vec = objectives_to_vector(metrics).tolist()
            elapsed = time.time() - t0

            trial = {
                "trial_id": i,
                "config": config,
                "config_hash": self.search_space.config_hash(config),
                "metrics": metrics,
                "obj_vector": obj_vec,
                "elapsed_sec": elapsed,
            }
            all_trials.append(trial)

            print(
                f"  Trial {i+1}/{self.n_trials} | "
                f"micro_f1={metrics.get('micro_f1', 0):.4f} | "
                f"unsupported={metrics.get('unsupported_code_rate', 1):.4f} | "
                f"elapsed={elapsed:.1f}s"
            )

        # Find Pareto front
        obj_matrix = np.array([t["obj_vector"] for t in all_trials])
        pareto_indices = find_pareto_front(obj_matrix)
        pareto_trials = [all_trials[i] for i in pareto_indices]

        # Best compromise: minimum sum of normalized objectives
        best_compromise = _select_best_compromise(pareto_trials)

        result = {
            "optimizer": "random_search",
            "n_trials": self.n_trials,
            "seed": self.seed,
            "all_trials": all_trials,
            "pareto_front": pareto_trials,
            "best_compromise": best_compromise,
            "pareto_size": len(pareto_trials),
        }

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

        return result


def _select_best_compromise(pareto_trials: List[Dict]) -> Optional[Dict]:
    """Select the Pareto solution with the minimum normalized objective sum."""
    if not pareto_trials:
        return None

    vecs = np.array([t["obj_vector"] for t in pareto_trials], dtype=float)
    mins = vecs.min(axis=0)
    maxs = vecs.max(axis=0)
    ranges = np.where(maxs - mins > 0, maxs - mins, 1.0)
    normalized = (vecs - mins) / ranges
    scores = normalized.sum(axis=1)
    best_idx = int(np.argmin(scores))
    return pareto_trials[best_idx]
