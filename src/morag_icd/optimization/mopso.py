"""
Multi-Objective Particle Swarm Optimization (MOPSO) for RAG hyperparameter tuning (E16).

Implements MOPSO from scratch using only numpy (no pymoo dependency).
Uses an external archive of non-dominated solutions as the Pareto repository.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .search_space import SearchSpace
from .objective import evaluate_config, objectives_to_vector
from .pareto import find_pareto_front, hypervolume_mc, select_best_compromise


class MOPSO:
    """
    Multi-Objective Particle Swarm Optimization.

    Maintains an external archive of Pareto-optimal solutions.
    Guide selection: random archive member weighted by crowding distance.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        n_particles: int = 20,
        n_iterations: int = 30,
        w: float = 0.4,       # inertia weight
        c1: float = 2.0,      # cognitive coefficient
        c2: float = 2.0,      # social coefficient
        archive_size: int = 50,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.archive_size = archive_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.lb, self.ub = search_space.bounds()
        self.n_dims = search_space.n_dims

    def run(
        self,
        pipeline_factory: Callable,
        dataset: List[Dict],
        label_set: Optional[List[str]] = None,
        output_path: Optional[str | Path] = None,
    ) -> Dict:
        """
        Run MOPSO optimization.

        Returns
        -------
        dict with pareto_front, best_compromise, hypervolume_history, etc.
        """
        # Initialize particles
        pos = self.rng.uniform(self.lb, self.ub, size=(self.n_particles, self.n_dims))
        vel = self.rng.uniform(-0.1, 0.1, size=(self.n_particles, self.n_dims))

        # Evaluate initial population
        configs = [self.search_space.decode(p) for p in pos]
        obj_matrix = self._evaluate_batch(configs, pipeline_factory, dataset, label_set)

        # Personal bests
        pbest_pos = pos.copy()
        pbest_obj = obj_matrix.copy()

        # External archive
        archive_pos = pos.copy()
        archive_obj = obj_matrix.copy()

        hypervolume_history = []
        iteration_logs = []

        print(f"[MOPSO] Starting: n_particles={self.n_particles}, n_iter={self.n_iterations}")

        for iteration in range(self.n_iterations):
            # Select guide for each particle from archive
            guides = self._select_guides(archive_pos, archive_obj)

            # Update velocity and position
            r1 = self.rng.random((self.n_particles, self.n_dims))
            r2 = self.rng.random((self.n_particles, self.n_dims))
            vel = (
                self.w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (guides - pos)
            )
            pos = np.clip(pos + vel, self.lb, self.ub)

            # Evaluate new positions
            configs = [self.search_space.decode(p) for p in pos]
            obj_matrix = self._evaluate_batch(configs, pipeline_factory, dataset, label_set)

            # Update personal bests (domination check)
            for i in range(self.n_particles):
                if self._dominates(obj_matrix[i], pbest_obj[i]):
                    pbest_pos[i] = pos[i].copy()
                    pbest_obj[i] = obj_matrix[i].copy()

            # Update archive
            archive_pos, archive_obj = self._update_archive(
                archive_pos, archive_obj, pos, obj_matrix
            )

            # Compute hypervolume
            pareto_idxs = find_pareto_front(archive_obj)
            hv = hypervolume_mc(archive_obj[pareto_idxs], n_samples=5000, seed=self.seed)
            hypervolume_history.append(hv)

            best_micro_f1 = -archive_obj[:, 0].min()
            print(f"  Iter {iteration+1}/{self.n_iterations} | best_micro_f1={best_micro_f1:.4f} | archive={len(archive_pos)} | HV≈{hv:.4f}")

            iteration_logs.append({
                "iteration": iteration + 1,
                "hypervolume": hv,
                "best_micro_f1": best_micro_f1,
                "archive_size": len(archive_pos),
            })

        # Final Pareto front from archive
        pareto_idxs = find_pareto_front(archive_obj)
        archive_configs = [self.search_space.decode(p) for p in archive_pos]
        pareto_solutions = [
            {
                "config": archive_configs[i],
                "config_hash": self.search_space.config_hash(archive_configs[i]),
                "obj_vector": archive_obj[i].tolist(),
            }
            for i in pareto_idxs
        ]

        best_idx = select_best_compromise(archive_obj, pareto_idxs)
        best_compromise = {
            "config": archive_configs[best_idx],
            "config_hash": self.search_space.config_hash(archive_configs[best_idx]),
            "obj_vector": archive_obj[best_idx].tolist(),
        }

        result = {
            "optimizer": "mopso",
            "n_particles": self.n_particles,
            "n_iterations": self.n_iterations,
            "seed": self.seed,
            "pareto_front": pareto_solutions,
            "pareto_size": len(pareto_solutions),  # parity with random_search's key
            "best_compromise": best_compromise,
            "hypervolume_history": hypervolume_history,
            "iteration_logs": iteration_logs,
            "final_hypervolume": hypervolume_history[-1] if hypervolume_history else 0.0,
        }

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    def _evaluate_batch(self, configs, pipeline_factory, dataset, label_set):
        obj_rows = []
        for config in configs:
            metrics = evaluate_config(config, pipeline_factory, dataset, label_set)
            obj_rows.append(objectives_to_vector(metrics))
        return np.array(obj_rows, dtype=float)

    def _select_guides(self, archive_pos: np.ndarray, archive_obj: np.ndarray) -> np.ndarray:
        """Select one guide per particle from the archive (random or roulette)."""
        n_archive = len(archive_pos)
        if n_archive == 0:
            return self.rng.uniform(self.lb, self.ub, size=(self.n_particles, self.n_dims))
        indices = self.rng.integers(0, n_archive, size=self.n_particles)
        return archive_pos[indices]

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a <= b) and np.any(a < b))

    def _update_archive(
        self,
        archive_pos: np.ndarray,
        archive_obj: np.ndarray,
        new_pos: np.ndarray,
        new_obj: np.ndarray,
    ):
        """Merge new solutions into archive, keep only Pareto-optimal, cap at archive_size."""
        combined_pos = np.vstack([archive_pos, new_pos])
        combined_obj = np.vstack([archive_obj, new_obj])

        pareto_idxs = find_pareto_front(combined_obj)
        pareto_pos = combined_pos[pareto_idxs]
        pareto_obj = combined_obj[pareto_idxs]

        # If archive too large, trim by crowding distance
        if len(pareto_pos) > self.archive_size:
            from .pareto import crowding_distance
            cd = crowding_distance(pareto_obj, list(range(len(pareto_obj))))
            sorted_idx = np.argsort(-cd)[:self.archive_size]
            pareto_pos = pareto_pos[sorted_idx]
            pareto_obj = pareto_obj[sorted_idx]

        return pareto_pos, pareto_obj


def run_mopso(pipeline_factory, dataset, config: Dict, search_space=None, **kwargs) -> Dict:
    """Convenience runner for integration with optimizer_runner."""
    from .search_space import SearchSpace
    if search_space is None:
        search_space = SearchSpace.from_config(config)
    optimizer = MOPSO(
        search_space=search_space,
        n_particles=config.get("mopso_params", {}).get("n_particles", 20),
        n_iterations=config.get("mopso_params", {}).get("n_iterations", 30),
        seed=kwargs.get("seed", 42),
    )
    return optimizer.run(pipeline_factory, dataset, **{k: v for k, v in kwargs.items() if k != "seed"})
