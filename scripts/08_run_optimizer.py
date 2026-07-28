"""
Script 08: Run hyperparameter optimization (E15/E16/E17).

Usage:
    python scripts/08_run_optimizer.py \\
        --optimizer mopso \\
        --seed 42 \\
        --top-n 50 \\
        --config configs/optimization.yaml \\
        [--n-trials 50]          # for random search
        [--n-subset 500]         # validation subset size
        [--output-dir results/optimization]
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morag_icd.config import load_config
from morag_icd.utils.seed import set_seed
from morag_icd.utils.logging_utils import setup_logger
from morag_icd.utils.io import load_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="Run MORAG-ICD hyperparameter optimization.")
    p.add_argument("--optimizer", required=True, choices=["random_search", "mopso", "nsga2"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--config", default="configs/optimization.yaml", help="Path to optimization.yaml")
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths.yaml")
    p.add_argument("--retrieval", default="configs/retrieval.yaml", help="Path to retrieval.yaml")
    p.add_argument("--models", default="configs/models.yaml", help="Path to models.yaml")
    p.add_argument("--n-trials", type=int, default=None, help="Override n_trials for random search")
    p.add_argument("--max-evaluations", type=int, default=None, help="Alias for n-trials (random search)")
    p.add_argument("--n-subset", type=int, default=500,
                   help="Number of validation samples to use for optimization")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Alias for n-subset")
    p.add_argument("--particles", type=int, default=None, help="Override MOPSO particle count")
    p.add_argument("--iterations", type=int, default=None, help="Override MOPSO iteration count")
    p.add_argument("--population-size", type=int, default=None, help="Override NSGA-II population size")
    p.add_argument("--generations", type=int, default=None, help="Override NSGA-II generations")
    p.add_argument("--output-dir", default="results/optimization")
    p.add_argument("--splits-root", default=None, help="Override split root, e.g. data/splits_real")
    p.add_argument("--indexes-root", default=None, help="Override indexes root, e.g. indexes_real")
    p.add_argument("--runtime-config", default=None,
                   help="Optional YAML merged into model/base config (e.g. real_canary_lowmem.yaml)")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def _optimizer_output_name(optimizer: str) -> str:
    return {
        "random_search": "E15_random_search",
        "mopso": "E16_mopso",
        "nsga2": "E17_nsga2",
    }[optimizer]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _materialize_optimizer_artifacts(
    result: dict,
    output_root: Path,
    optimizer: str,
    seed: int,
    top_n: int,
    run_seconds: float,
    opt_config: dict,
    dataset_size: int,
) -> None:
    pareto_front = result.get("pareto_front", []) or []
    best_compromise = result.get("best_compromise") or {}
    n_evaluated = int(result.get("n_evaluated", len(result.get("all_trials", []))))

    config_payload = {
        "optimizer": optimizer,
        "seed": seed,
        "top_n": top_n,
        "dataset_size": dataset_size,
        "n_evaluated_configs": n_evaluated,
        "optimizer_params": {
            "random_search": {
                "n_trials": result.get("n_trials", opt_config.get("random_search_params", {}).get("n_trials")),
            },
            "mopso": {
                "n_particles": result.get("n_particles", opt_config.get("mopso_params", {}).get("n_particles")),
                "n_iterations": result.get("n_iterations", opt_config.get("mopso_params", {}).get("n_iterations")),
            },
            "nsga2": {
                "pop_size": result.get("pop_size", opt_config.get("nsga2_params", {}).get("pop_size")),
                "n_gen": result.get("n_gen", opt_config.get("nsga2_params", {}).get("n_gen")),
            },
        },
    }
    _write_json(output_root / "optimizer_config.json", config_payload)

    history_rows = result.get("all_trials")
    if history_rows is None:
        history_rows = result.get("iteration_logs") or result.get("generation_logs") or []
    _write_jsonl(output_root / "optimization_history.jsonl", history_rows)

    _write_json(output_root / "pareto_front.json", {
        "optimizer": optimizer,
        "seed": seed,
        "top_n": top_n,
        "pareto_size": len(pareto_front),
        "pareto_front": pareto_front,
    })

    best_cfg = best_compromise.get("config") or {}
    _write_json(output_root / "best_compromise_config.json", {
        "experiment_id": _optimizer_output_name(optimizer).split("_")[0],
        "optimizer": optimizer,
        "seed": seed,
        "top_n": top_n,
        "config": best_cfg,
        "config_hash": best_compromise.get("config_hash", ""),
    })

    _write_json(output_root / "best_compromise_summary.json", {
        "optimizer": optimizer,
        "seed": seed,
        "top_n": top_n,
        "obj_vector": best_compromise.get("obj_vector", []),
        "config_hash": best_compromise.get("config_hash", ""),
        "n_evaluated_configs": n_evaluated,
        "pareto_size": len(pareto_front),
    })

    failed_trials = [
        t for t in (result.get("all_trials") or [])
        if bool((t.get("metrics") or {}).get("_failed", False))
    ]
    _write_jsonl(output_root / "failed_trials.jsonl", failed_trials)

    _write_json(output_root / "runtime_summary.json", {
        "optimizer": optimizer,
        "seed": seed,
        "top_n": top_n,
        "run_seconds": run_seconds,
        "n_evaluated_configs": n_evaluated,
        "pareto_size": len(pareto_front),
        "final_hypervolume": result.get("final_hypervolume", 0.0),
    })

    _write_json(output_root / "optimizer_results.json", result)


def main():
    args = parse_args()
    set_seed(args.seed)
    started = time.time()

    effective_n_subset = args.max_samples if args.max_samples is not None else args.n_subset

    opt_config = load_config(args.config)
    paths_config = load_config(args.paths) if Path(args.paths).exists() else {}
    retrieval_config = load_config(args.retrieval) if Path(args.retrieval).exists() else {}
    model_config = load_config(args.models) if Path(args.models).exists() else {}

    # Real-data roots + runtime overrides (so the optimizer can run on real data)
    if args.indexes_root:
        paths_config["indexes_dir"] = args.indexes_root
    runtime_config = load_config(args.runtime_config) if args.runtime_config and Path(args.runtime_config).exists() else {}
    if runtime_config:
        # gen/truncation params reach the shared LLM via model_config; search-space params
        # in the runtime config are only defaults (each evaluated hp_config overrides them).
        model_config = {**model_config, **runtime_config}
        retrieval_config = {**retrieval_config, **runtime_config}

    logger = setup_logger(
        name=f"optimizer_{args.optimizer}",
        log_dir=paths_config.get("logs_dir", "logs"),
        experiment_id=f"opt_{args.optimizer}",
        seed=args.seed,
    )

    logger.info(f"Starting optimizer={args.optimizer} | seed={args.seed} | top_n={args.top_n}")

    if str(model_config.get("device", "cpu")).lower() == "cuda":
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA unavailable while device=cuda. Failing fast to prevent CPU fallback.")
            sys.exit(1)

    # Override trial counts if specified
    if args.n_trials is not None or args.max_evaluations is not None:
        opt_config.setdefault("random_search_params", {})["n_trials"] = (
            args.max_evaluations if args.max_evaluations is not None else args.n_trials
        )
    if args.particles is not None:
        opt_config.setdefault("mopso_params", {})["n_particles"] = args.particles
    if args.iterations is not None:
        opt_config.setdefault("mopso_params", {})["n_iterations"] = args.iterations
    if args.population_size is not None:
        opt_config.setdefault("nsga2_params", {})["pop_size"] = args.population_size
    if args.generations is not None:
        opt_config.setdefault("nsga2_params", {})["n_gen"] = args.generations

    # Load validation dataset
    if args.smoke_test:
        val_path = Path("data/smoke_test/dataset.jsonl")
    elif args.splits_root:
        val_path = Path(args.splits_root) / f"top{args.top_n}" / "validation.jsonl"
    else:
        split_dir = Path(paths_config.get("project_root", ".")) / "data" / "splits"
        val_path = split_dir / f"top{args.top_n}" / "validation.jsonl"

    if not val_path.exists():
        logger.error(f"Validation data not found: {val_path}")
        sys.exit(1)

    val_data = load_jsonl(val_path)
    val_data = [row for row in val_data if row.get("split", "validation") == "validation"]
    logger.info(f"Loaded {len(val_data)} validation samples")

    # Subsample for optimization speed
    if effective_n_subset and len(val_data) > effective_n_subset:
        import random
        rng = random.Random(args.seed)
        val_data = rng.sample(val_data, effective_n_subset)
        logger.info(f"Subsampled to {effective_n_subset} samples for optimization")

    # Build pipeline factory
    from morag_icd.optimization.optimizer_runner import build_pipeline_factory
    pipeline_factory = build_pipeline_factory(
        base_config={**retrieval_config, **opt_config},
        retrieval_config=retrieval_config,
        model_config=model_config,
        paths_config=paths_config,
        top_n_suffix=str(args.top_n),
    )

    # Output path
    output_dir = Path(args.output_dir) / _optimizer_output_name(args.optimizer)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed{args.seed}_results.json"

    # Run optimizer
    from morag_icd.optimization.optimizer_runner import run_optimizer

    # Fixed Top-N label space (P1-1) so optimizer objectives are on a stable label set.
    label_set = None
    if args.splits_root:
        ls_path = Path(args.splits_root) / f"top{args.top_n}" / "label_set.json"
        if ls_path.exists():
            label_set = sorted({str(c) for c in json.loads(ls_path.read_text(encoding="utf-8"))})
            logger.info(f"Using fixed label space: {len(label_set)} codes")

    result = run_optimizer(
        optimizer_name=args.optimizer,
        pipeline_factory=pipeline_factory,
        dataset=val_data,
        opt_config=opt_config,
        label_set=label_set,
        seed=args.seed,
        output_path=output_path,
    )

    logger.info(
        f"Optimization complete: Pareto front size={result.get('pareto_size', len(result.get('pareto_front', [])))} | "
        f"HV={result.get('final_hypervolume', 0):.4f}"
    )

    if "n_evaluated" not in result:
        if result.get("optimizer") == "random_search":
            result["n_evaluated"] = len(result.get("all_trials", []))
        elif result.get("optimizer") == "mopso":
            result["n_evaluated"] = int(result.get("n_particles", 0)) * int(result.get("n_iterations", 0))
        elif result.get("optimizer") == "nsga2":
            result["n_evaluated"] = int(result.get("pop_size", 0)) * (int(result.get("n_gen", 0)) + 1)

    _materialize_optimizer_artifacts(
        result=result,
        output_root=output_dir,
        optimizer=args.optimizer,
        seed=args.seed,
        top_n=args.top_n,
        run_seconds=(time.time() - started),
        opt_config=opt_config,
        dataset_size=len(val_data),
    )

    best = result.get("best_compromise", {})
    if best:
        print("\n=== Best Compromise Config ===")
        print(json.dumps(best.get("config", {}), indent=2))
        print(f"Objective vector: {best.get('obj_vector', [])}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
