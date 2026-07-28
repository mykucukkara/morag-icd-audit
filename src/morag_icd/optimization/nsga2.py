"""
NSGA-II multi-objective evolutionary algorithm for RAG hyperparameter optimization (E17).

Implements the standard NSGA-II algorithm from scratch using only numpy,
avoiding pymoo dependency issues on TRUBA.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .search_space import SearchSpace
from .objective import evaluate_config, objectives_to_vector, validate_objective_vector, failed_objective_vector
from .pareto import fast_non_dominated_sort, crowding_distance, hypervolume_mc, select_best_compromise


class NSGA2:
    """
    NSGA-II optimizer for multi-objective RAG hyperparameter tuning.

    Operates in the encoded [0,1]^n continuous space and decodes back to
    discrete configs for evaluation.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        pop_size: int = 20,
        n_gen: int = 30,
        crossover_prob: float = 0.9,
        mutation_prob: Optional[float] = None,
        seed: int = 42,
    ):
        self.search_space = search_space
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob or (1.0 / search_space.n_dims)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.lb, self.ub = search_space.bounds()

    def run(
        self,
        pipeline_factory: Callable,
        dataset: List[Dict],
        label_set: Optional[List[str]] = None,
        output_path: Optional[str | Path] = None,
    ) -> Dict:
        """
        Run NSGA-II optimization.

        Returns
        -------
        dict with keys: all_generations, pareto_front, best_compromise,
                        hypervolume_history, final_population
        """
        # Initialize population
        pop = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.search_space.n_dims))
        configs = [self.search_space.decode(x) for x in pop]
        obj_matrix, all_trials = self._evaluate_population(
            configs,
            pipeline_factory,
            dataset,
            label_set,
            generation=0,
            population_role="initial",
        )

        hypervolume_history = []
        generation_logs = []

        print(f"[NSGA-II] Starting: pop_size={self.pop_size}, n_gen={self.n_gen}")

        for gen in range(self.n_gen):
            # Selection, crossover, mutation → offspring
            offspring_pop = self._generate_offspring(pop)
            offspring_configs = [self.search_space.decode(x) for x in offspring_pop]
            offspring_objs, offspring_trials = self._evaluate_population(
                offspring_configs,
                pipeline_factory,
                dataset,
                label_set,
                generation=gen + 1,
                population_role="offspring",
            )
            all_trials.extend(offspring_trials)

            # Combine parent + offspring
            combined_pop = np.vstack([pop, offspring_pop])
            combined_objs = np.vstack([obj_matrix, offspring_objs])
            combined_configs = configs + offspring_configs

            # Non-dominated sorting + crowding distance selection
            pop, obj_matrix, configs = self._select_next_generation(
                combined_pop, combined_objs, combined_configs
            )

            # Compute hypervolume
            pareto_idxs = fast_non_dominated_sort(obj_matrix)[0]
            hv = hypervolume_mc(obj_matrix[pareto_idxs], n_samples=5000, seed=self.seed)
            hypervolume_history.append(hv)

            best_micro_f1 = -obj_matrix[:, 0].min()  # obj[0] = -micro_f1
            print(f"  Gen {gen+1}/{self.n_gen} | best_micro_f1={best_micro_f1:.4f} | HV≈{hv:.4f}")

            generation_logs.append({
                "generation": gen + 1,
                "hypervolume": hv,
                "best_micro_f1": best_micro_f1,
            })

        # Final Pareto front
        fronts = fast_non_dominated_sort(obj_matrix)
        pareto_indices = fronts[0]
        pareto_solutions = [
            {
                "config": configs[i],
                "config_hash": self.search_space.config_hash(configs[i]),
                "obj_vector": obj_matrix[i].tolist(),
                "front": 0,
            }
            for i in pareto_indices
        ]

        # Best compromise
        best_idx = select_best_compromise(obj_matrix, pareto_indices)
        best_compromise = {
            "config": configs[best_idx],
            "config_hash": self.search_space.config_hash(configs[best_idx]),
            "obj_vector": obj_matrix[best_idx].tolist(),
        }

        result = {
            "optimizer": "nsga2",
            "pop_size": self.pop_size,
            "n_gen": self.n_gen,
            "seed": self.seed,
            "n_evaluated": len(all_trials),
            "all_trials": all_trials,
            "pareto_front": pareto_solutions,
            "pareto_size": len(pareto_solutions),  # parity with random_search's key
            "best_compromise": best_compromise,
            "hypervolume_history": hypervolume_history,
            "generation_logs": generation_logs,
            "final_hypervolume": hypervolume_history[-1] if hypervolume_history else 0.0,
        }

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    def _evaluate_population(
        self,
        configs: List[Dict],
        pipeline_factory: Callable,
        dataset: List[Dict],
        label_set,
        generation: int = 0,
        population_role: str = "population",
    ) -> tuple[np.ndarray, List[Dict]]:
        """Evaluate all configs and return objective matrix plus trial records."""
        obj_rows = []
        trials: List[Dict] = []
        for config in configs:
            metrics = evaluate_config(config, pipeline_factory, dataset, label_set)
            raw_obj = objectives_to_vector(metrics)
            trial_status = "passed"
            trial_error = ""
            try:
                obj_vec = validate_objective_vector(raw_obj)
            except Exception as exc:
                trial_status = "failed"
                trial_error = str(exc)
                metrics = dict(metrics)
                metrics["_failed"] = True
                metrics["_reason"] = trial_error
                obj_vec = failed_objective_vector(trial_error)

            obj_rows.append(obj_vec)
            trials.append({
                "generation": generation,
                "population_role": population_role,
                "config": config,
                "config_hash": self.search_space.config_hash(config),
                "metrics": metrics,
                "obj_vector": obj_vec.tolist(),
                "status": trial_status,
                "error": trial_error,
            })

        if not obj_rows:
            return np.empty((0, len(failed_objective_vector())), dtype=float), trials

        obj_matrix = np.vstack(obj_rows)
        if obj_matrix.shape[0] != len(configs):
            raise ValueError(
                f"Objective/population length mismatch: objectives={obj_matrix.shape[0]} configs={len(configs)}"
            )
        return obj_matrix, trials

    def _generate_offspring(self, pop: np.ndarray) -> np.ndarray:
        """SBX crossover + polynomial mutation."""
        n, d = pop.shape
        offspring = np.empty_like(pop)
        indices = self.rng.permutation(n)

        for i in range(0, n, 2):
            p1 = pop[indices[i % n]]
            p2 = pop[indices[(i + 1) % n]]

            if self.rng.random() < self.crossover_prob:
                c1, c2 = self._sbx_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            offspring[i % n] = self._polynomial_mutation(c1)
            if i + 1 < n:
                offspring[(i + 1) % n] = self._polynomial_mutation(c2)

        return np.clip(offspring, self.lb, self.ub)

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray, eta: float = 20.0):
        """Simulated Binary Crossover."""
        u = self.rng.random(len(p1))
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta + 1)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1)),
        )
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return c1, c2

    def _polynomial_mutation(self, x: np.ndarray, eta: float = 20.0) -> np.ndarray:
        """Polynomial mutation."""
        x = x.copy()
        for j in range(len(x)):
            if self.rng.random() < self.mutation_prob:
                u = self.rng.random()
                delta = (
                    (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5
                    else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                )
                x[j] = np.clip(x[j] + delta, self.lb[j], self.ub[j])
        return x

    def _select_next_generation(
        self,
        pop: np.ndarray,
        obj_matrix: np.ndarray,
        configs: List[Dict],
    ):
        """Select next generation using non-dominated rank + crowding distance."""
        fronts = fast_non_dominated_sort(obj_matrix)
        if not fronts:
            return pop[:0], obj_matrix[:0], []
        selected_indices = []

        for front in fronts:
            if len(selected_indices) + len(front) <= self.pop_size:
                selected_indices.extend(front)
            else:
                # Fill remaining slots by crowding distance
                remaining = self.pop_size - len(selected_indices)
                cd = crowding_distance(obj_matrix, front)
                sorted_by_cd = sorted(zip(cd, front), key=lambda x: -x[0])
                selected_indices.extend([idx for _, idx in sorted_by_cd[:remaining]])
                break

        for idx in selected_indices:
            if idx < 0 or idx >= len(configs):
                raise ValueError(f"Invalid Pareto index generated: idx={idx}, n={len(configs)}")

        sel = selected_indices
        return pop[sel], obj_matrix[sel], [configs[i] for i in sel]


def run_nsga2(pipeline_factory, dataset, config: Dict, search_space=None, **kwargs) -> Dict:
    """Convenience runner for integration with optimizer_runner."""
    from .search_space import SearchSpace
    if search_space is None:
        search_space = SearchSpace.from_config(config)
    optimizer = NSGA2(
        search_space=search_space,
        pop_size=config.get("nsga2_params", {}).get("pop_size", 20),
        n_gen=config.get("nsga2_params", {}).get("n_gen", 30),
        seed=kwargs.get("seed", 42),
    )
    return optimizer.run(pipeline_factory, dataset, **{k: v for k, v in kwargs.items() if k != "seed"})
