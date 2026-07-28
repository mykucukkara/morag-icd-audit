"""
Script 07: Run a single experiment.

Usage:
    python scripts/07_run_experiment.py \\
        --experiment-id E11 \\
        --seed 42 \\
        --top-n 50 \\
        --config configs/experiments.yaml \\
        [--smoke-test] \\
        [--split test|validation] \\
        [--output-dir results/predictions]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morag_icd.config import load_config
from morag_icd.utils.seed import set_seed
from morag_icd.utils.logging_utils import setup_logger
from morag_icd.utils.io import load_jsonl
from morag_icd.utils.model_readiness import build_experiment_readiness
from morag_icd.utils.hashing import hash_config


OPT_EXPERIMENTS = {"E15", "E16", "E17"}
OPT_DEFAULT_PATHS = {
    "E15": "results/optimization/E15_random_search/best_compromise_config.json",
    "E16": "results/optimization/E16_mopso/best_compromise_config.json",
    "E17": "results/optimization/E17_nsga2/best_compromise_config.json",
}
SUPPORTED_OPTIMIZER_PARAMS = {
    "chunk_size",
    "chunk_overlap",
    "top_k_evidence",
    "top_k_icd_candidates",
    "bm25_dense_alpha",
    "evidence_similarity_threshold",
    "llm_confidence_threshold",
    "max_final_codes",
    "prompt_template_id",
    "section_weights",
}


def _resolve_classifier_model_path(model_config: dict, paths_config: dict) -> tuple[str, bool]:
    raw = str(model_config.get("classifier_model_path", "") or "").strip()
    models_dir = str(paths_config.get("models_dir", "") or "").strip()
    resolved = raw
    if raw and "${models_dir}" in raw and models_dir:
        resolved = raw.replace("${models_dir}", models_dir)
        model_config["classifier_model_path"] = resolved
    exists = bool(resolved) and ("${" not in resolved) and Path(resolved).exists()
    return resolved, exists


def parse_args():
    p = argparse.ArgumentParser(description="Run a single MORAG-ICD experiment.")
    p.add_argument("--experiment-id", required=True, help="Experiment ID, e.g. E11")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--top-n", type=int, default=50, help="Top-N ICD codes (50/100/200)")
    p.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments.yaml")
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths.yaml")
    p.add_argument("--models", default="configs/models.yaml", help="Path to models.yaml")
    p.add_argument("--retrieval", default="configs/retrieval.yaml", help="Path to retrieval.yaml")
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--output-dir", default="results/predictions")
    p.add_argument("--output-root", dest="output_dir", default=argparse.SUPPRESS,
                   help="Alias for --output-dir; useful for real-data outputs")
    p.add_argument("--splits-root", default=None,
                   help="Override split root, e.g. data/splits_real")
    p.add_argument("--indexes-root", default=None,
                   help="Override indexes root, e.g. indexes_real")
    p.add_argument("--icd-kb", default=None,
                   help="Optional ICD KB path recorded in run metadata")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Process only the first N requested split samples (EVAL split only; "
                        "does not limit baseline training data)")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Optionally cap baseline TRAINING samples (default: use full train split)")
    p.add_argument("--shard-index", type=int, default=None,
                   help="0-based shard index for script-level sharding (requires --num-shards)")
    p.add_argument("--num-shards", type=int, default=None,
                   help="Total number of shards for script-level sharding (requires --shard-index)")
    p.add_argument("--data-mode", default="smoke_or_default", choices=["smoke_or_default", "real"],
                   help="Metadata flag for real-data canary runs")
    p.add_argument("--smoke-test", action="store_true",
                   help="Use smoke test data instead of full dataset")
    p.add_argument("--allow-mock-llm", action="store_true",
                   help="Allow deterministic mock LLM for smoke runs")
    p.add_argument("--allow-mock-embedding", action="store_true",
                   help="Allow deterministic mock embeddings for smoke runs")
    p.add_argument("--dry-run", action="store_true",
                   help="Run readiness checks only without inference")
    p.add_argument("--legacy-per-candidate", action="store_true",
                   help="Ablation: score each candidate with a separate LLM call (old design) "
                        "instead of one batched call per note")
    p.add_argument("--optimizer-config", default=None,
                   help="Path to optimizer best_compromise_config.json for E15/E16/E17")
    p.add_argument("--runtime-config", default=None,
                   help="Optional YAML config applied after optimizer and hp overrides")
    p.add_argument("--hp-config", default=None,
                   help="Optional JSON string of hyperparameter overrides")
    return p.parse_args()


def _resolve_shard_args(args):
    """Validate --shard-index/--num-shards and return (is_sharded, shard_index, num_shards).

    Rules:
    - both must be provided together (one-without-the-other is an error)
    - num_shards >= 1
    - 0 <= shard_index < num_shards
    - mutually exclusive with --max-samples
    On invalid input, prints a JSON error and exits(2). Never logs PHI.
    """
    si = args.shard_index
    ns = args.num_shards
    if si is None and ns is None:
        return False, None, None
    if (si is None) != (ns is None):
        _shard_error("shard_args_incomplete: --shard-index and --num-shards must be provided together")
    if ns < 1:
        _shard_error(f"invalid_num_shards: --num-shards must be >= 1 (got {ns})")
    if si < 0:
        _shard_error(f"invalid_shard_index: --shard-index must be >= 0 (got {si})")
    if si >= ns:
        _shard_error(f"invalid_shard_index: --shard-index ({si}) must be < --num-shards ({ns})")
    if args.max_samples is not None:
        _shard_error("mutually_exclusive: --max-samples cannot be combined with sharding args")
    return True, si, ns


def _shard_error(msg: str):
    print(json.dumps({"status": "invalid_shard_args", "error": msg}, indent=2))
    sys.exit(2)


def _make_readiness(args, paths_config, model_config):
    return build_experiment_readiness(
        experiment_id=args.experiment_id,
        top_n=args.top_n,
        smoke_test=bool(args.smoke_test),
        split=args.split,
        paths_config=paths_config,
        model_config=model_config,
        allow_mock_llm=bool(args.allow_mock_llm),
        allow_mock_embedding=bool(args.allow_mock_embedding),
    )


def _write_skip_summary(output_path: Path, args, merged_config: dict, readiness: dict):
    summary = {
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "pipeline_type": "skipped",
        "config_hash": hash_config(merged_config),
        "total_samples": 0,
        "success_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "error_rate": 0.0,
        "status": readiness.get("status", "skipped_missing_model"),
        "reason": "; ".join(readiness.get("errors", [])) or "; ".join(readiness.get("notes", [])),
        "readiness": readiness,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _load_optimizer_payload(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        cfg = payload["config"]
        cfg_hash = str(payload.get("config_hash", "")).strip() or hash_config(cfg)
        return cfg, cfg_hash
    if isinstance(payload, dict):
        return payload, hash_config(payload)
    raise RuntimeError(f"Invalid optimizer config payload: {path}")


def _resolve_optimizer_config(args, logger):
    exp_id = args.experiment_id
    if exp_id not in OPT_EXPERIMENTS:
        if args.optimizer_config:
            logger.warning(
                "Ignoring --optimizer-config for non-optimizer experiment %s",
                exp_id,
            )
        return {
            "optimizer_config_path": "",
            "optimizer_config_loaded": False,
            "optimizer_config_hash": "",
            "optimizer_config_values": {},
            "unused_optimizer_params": [],
        }

    path = Path(args.optimizer_config) if args.optimizer_config else Path(OPT_DEFAULT_PATHS[exp_id])
    if not path.exists():
        raise FileNotFoundError(f"missing_optimizer_config: {path}")

    cfg, cfg_hash = _load_optimizer_payload(path)
    unused = sorted([k for k in cfg.keys() if k not in SUPPORTED_OPTIMIZER_PARAMS])
    return {
        "optimizer_config_path": str(path),
        "optimizer_config_loaded": True,
        "optimizer_config_hash": cfg_hash,
        "optimizer_config_values": cfg,
        "unused_optimizer_params": unused,
    }


def main():
    args = parse_args()
    is_sharded, shard_index, num_shards = _resolve_shard_args(args)
    set_seed(args.seed)

    # Load configs
    exp_config = load_config(args.config)
    paths_config = load_config(args.paths) if Path(args.paths).exists() else {}
    retrieval_config = load_config(args.retrieval) if Path(args.retrieval).exists() else {}
    model_config = load_config(args.models) if Path(args.models).exists() else {}
    if args.splits_root:
        paths_config["splits_root"] = args.splits_root
    if args.indexes_root:
        paths_config["indexes_dir"] = args.indexes_root
    if args.icd_kb:
        paths_config["icd_kb_path"] = args.icd_kb

    logger = setup_logger(
        name=f"run_{args.experiment_id}",
        log_dir=paths_config.get("logs_dir", "logs"),
        experiment_id=args.experiment_id,
        seed=args.seed,
    )

    logger.info(f"Starting experiment {args.experiment_id} | seed={args.seed} | top_n={args.top_n}")

    classifier_model_path, classifier_model_exists = _resolve_classifier_model_path(model_config, paths_config)
    if args.experiment_id == "E3" and not classifier_model_exists:
        err = (
            "missing_classifier_model: E3 requires classifier_model_path with an existing local checkpoint; "
            f"resolved_path={classifier_model_path or '<empty>'}"
        )
        logger.error(err)
        print(json.dumps({"status": "missing_classifier_model", "error": err}, indent=2))
        sys.exit(1)

    try:
        optimizer_meta = _resolve_optimizer_config(args, logger)
    except Exception as e:
        logger.error(str(e))
        print(json.dumps({"status": "missing_optimizer_config", "error": str(e)}, indent=2))
        sys.exit(1)

    # Optional HP override
    hp_config = {}
    if args.hp_config:
        hp_config = json.loads(args.hp_config)
    hp_config = {
        **optimizer_meta["optimizer_config_values"],
        **hp_config,
    }
    runtime_config = {}
    if args.runtime_config:
        runtime_config = load_config(args.runtime_config)
    hp_config = {
        **hp_config,
        **runtime_config,
    }
    hp_config = {
        **hp_config,
        # The pipeline is built from hp_config (+retrieval/model/paths), NOT from merged_config,
        # so top_n_suffix must be threaded here: build_full_model resolves the ICD index and the
        # allowed label set as cfg.get("top_n_suffix", "50"). Without this the retrieval/RAG/full
        # arms silently load the Top-50 index and label space for every --top-n, which is correct
        # by coincidence at Top-50 and wrong for the Top-100/200 scalability runs.
        "top_n_suffix": str(args.top_n),
        "allow_mock_llm": bool(args.allow_mock_llm),
        "allow_mock_embedding": bool(args.allow_mock_embedding),
        "legacy_per_candidate": bool(args.legacy_per_candidate),
    }

    merged_config = {
        **retrieval_config,
        **model_config,
        **exp_config,
        **hp_config,
        "top_n_suffix": str(args.top_n),
        "logs_dir": paths_config.get("logs_dir", "logs"),
        "data_mode": args.data_mode,
        "max_samples": args.max_samples,
        "splits_root": args.splits_root or "",
        "indexes_root": args.indexes_root or "",
        "icd_kb_path": args.icd_kb or "",
        "runtime_config_path": args.runtime_config or "",
        "phi_safe_logging": args.data_mode == "real",
    }

    readiness = _make_readiness(args, paths_config, model_config)
    if args.dry_run:
        if args.experiment_id == "E3":
            from morag_icd.baselines.transformer_classifier import TransformerClassifier

            readiness = {
                **readiness,
                "classifier_model_path": classifier_model_path,
                "classifier_model_available": bool(classifier_model_exists),
                "classifier_load_mode": TransformerClassifier.detect_load_mode(classifier_model_path),
                "classifier_weight_format": TransformerClassifier.detect_weight_format(classifier_model_path),
                "classifier_use_safetensors": bool(model_config.get("classifier_use_safetensors", True)),
            }
        print(json.dumps(readiness, indent=2, default=str))
        return

    output_dir = Path(args.output_dir) / f"top{args.top_n}" / f"{args.experiment_id}_seed{args.seed}"
    if is_sharded:
        output_dir = output_dir / "shards" / f"shard_{shard_index:03d}_of_{num_shards:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_predictions.jsonl"

    if readiness["status"] != "ready":
        logger.warning(f"Readiness check failed: {readiness['errors']}")
        summary = _write_skip_summary(output_path, args, merged_config, readiness)
        classifier_weight_format = ""
        if args.experiment_id == "E3":
            from morag_icd.baselines.transformer_classifier import TransformerClassifier

            classifier_weight_format = TransformerClassifier.detect_weight_format(classifier_model_path)
        summary.update({
            "optimizer_config_path": optimizer_meta["optimizer_config_path"],
            "optimizer_config_loaded": optimizer_meta["optimizer_config_loaded"],
            "optimizer_config_hash": optimizer_meta["optimizer_config_hash"],
            "unused_optimizer_params": optimizer_meta["unused_optimizer_params"],
            "classifier_model_path": classifier_model_path if args.experiment_id == "E3" else "",
            "classifier_model_available": bool(classifier_model_exists) if args.experiment_id == "E3" else False,
            "classifier_weight_format": classifier_weight_format,
            "classifier_use_safetensors": bool(model_config.get("classifier_use_safetensors", True)) if args.experiment_id == "E3" else False,
        })
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
        return

    # Load dataset
    if args.smoke_test:
        data_path = Path(paths_config.get("project_root", ".")) / "data" / "smoke_test" / "dataset.jsonl"
        if not data_path.exists():
            logger.error(f"Smoke test data not found at {data_path}. Run script 00 first.")
            sys.exit(1)
    else:
        split_dir = Path(paths_config.get("splits_root", Path(paths_config.get("project_root", ".")) / "data" / "splits"))
        data_path = split_dir / f"top{args.top_n}" / f"{args.split}.jsonl"
        if not data_path.exists():
            logger.error(f"Dataset not found: {data_path}. Run scripts 01-03 first.")
            sys.exit(1)

    logger.info(f"Loading dataset from {data_path}")
    dataset = load_jsonl(data_path)
    if args.smoke_test and dataset and "split" in dataset[0]:
        dataset = [s for s in dataset if s.get("split") == args.split]
    if args.max_samples is not None:
        if args.max_samples < 1:
            logger.error("--max-samples must be positive when provided")
            sys.exit(1)
        dataset = dataset[: args.max_samples]
    logger.info(f"Loaded {len(dataset)} samples from {args.split} split")

    # Build pipeline
    from morag_icd.pipeline.full_model import build_pipeline_for_experiment
    try:
        pipeline, pipeline_type = build_pipeline_for_experiment(
            experiment_id=args.experiment_id,
            retrieval_config=retrieval_config,
            model_config=model_config,
            paths_config=paths_config,
            hp_config=hp_config,
        )
    except Exception as e:
        logger.error(f"Failed to build pipeline for {args.experiment_id}: {e}")
        failure_status = "failed_missing_model"
        if "mock" in str(e).lower() or "path" in str(e).lower() or "dense" in str(e).lower():
            failure_status = "skipped_missing_model"
        readiness = {
            **readiness,
            "status": failure_status,
            "errors": readiness.get("errors", []) + [str(e)],
        }
        summary = _write_skip_summary(output_path, args, merged_config, readiness)
        if args.experiment_id == "E3":
            from morag_icd.baselines.transformer_classifier import TransformerClassifier

            summary.update({
                "classifier_model_path": classifier_model_path,
                "classifier_model_available": bool(classifier_model_exists),
                "classifier_load_mode": TransformerClassifier.detect_load_mode(classifier_model_path),
                "classifier_weight_format": TransformerClassifier.detect_weight_format(classifier_model_path),
                "classifier_use_safetensors": bool(model_config.get("classifier_use_safetensors", True)),
            })
            summary_path = output_path.parent / f"{output_path.stem}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
        return

    logger.info(f"Pipeline type: {pipeline_type}")

    # For sklearn baselines (E1, E2, E3), train on training split first
    if pipeline_type == "baseline" and hasattr(pipeline, "fit"):
        if args.smoke_test:
            all_data = load_jsonl(data_path)
            train_data = [s for s in all_data if s.get("split") == "train"]
            dataset = [s for s in all_data if s.get("split") == args.split]
        else:
            train_path = data_path.parent / "train.jsonl"
            train_data = load_jsonl(train_path) if train_path.exists() else []
        # NOTE: --max-samples caps the EVAL split only. Previously it also truncated the
        # training set, which silently trained E3 on 1000 of 80,707 notes (1.2%) and made
        # the classifier emit zero predictions. Training size is now capped only by the
        # explicit --max-train-samples.
        if args.max_train_samples is not None:
            train_data = train_data[: args.max_train_samples]
        if train_data:
            logger.info(f"Training baseline on {len(train_data)} samples...")
            if pipeline.__class__.__name__ == "TransformerClassifier":
                x_train = [str(s.get("text", "") or "") for s in train_data]
                label_names = sorted(
                    {
                        str(code)
                        for s in train_data
                        for code in (s.get("gold_codes", []) or [])
                        if str(code).strip()
                    }
                )
                y_train = []
                for s in train_data:
                    gold_codes = {str(c) for c in (s.get("gold_codes", []) or [])}
                    y_train.append([1 if code in gold_codes else 0 for code in label_names])
                pipeline.fit(x_train, y_train, label_names=label_names)
            else:
                pipeline.fit(train_data)
        else:
            logger.warning("No training data found; baseline will run without fitted state.")

    # Script-level sharding: slice the full split deterministically AFTER it is fully loaded
    # (and after any baseline train/test re-derivation). Full split ordering is preserved so
    # global_sample_index is well-defined across shards.
    full_split_sample_count = len(dataset)
    shard_start_index = 0
    shard_end_index = full_split_sample_count
    if is_sharded:
        shard_start_index = full_split_sample_count * shard_index // num_shards
        shard_end_index = full_split_sample_count * (shard_index + 1) // num_shards
        dataset = dataset[shard_start_index:shard_end_index]
        for local_i, sample in enumerate(dataset):
            sample["global_sample_index"] = shard_start_index + local_i
        logger.info(
            f"Sharding: shard {shard_index}/{num_shards} | full_split={full_split_sample_count} "
            f"| range=[{shard_start_index},{shard_end_index}) | shard_samples={len(dataset)}"
        )
    else:
        for local_i, sample in enumerate(dataset):
            sample["global_sample_index"] = local_i

    shard_meta_fields = {
        "is_sharded_run": bool(is_sharded),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_start_index": shard_start_index,
        "shard_end_index": shard_end_index,
        "full_split_sample_count": full_split_sample_count,
        "shard_sample_count": len(dataset),
    }

    # Run experiment
    from morag_icd.pipeline.experiment_runner import ExperimentRunner
    runner = ExperimentRunner(
        pipeline=pipeline,
        config=merged_config,
        experiment_id=args.experiment_id,
        seed=args.seed,
        pipeline_type=pipeline_type,
    )
    # Attach shard metadata AFTER the runner freezes config_hash, so config_hash stays
    # identical across all shards of the same experiment (only the shard slice differs).
    runner.config.update(shard_meta_fields)

    summary = runner.run(dataset=dataset, output_path=output_path)
    summary.update({
        "optimizer_config_path": optimizer_meta["optimizer_config_path"],
        "optimizer_config_loaded": optimizer_meta["optimizer_config_loaded"],
        "optimizer_config_hash": optimizer_meta["optimizer_config_hash"],
        "unused_optimizer_params": optimizer_meta["unused_optimizer_params"],
        "data_mode": args.data_mode,
        "max_samples": args.max_samples,
        "splits_root": args.splits_root or "",
        "indexes_root": args.indexes_root or "",
        "icd_kb_path": args.icd_kb or "",
        "data_path": str(data_path),
        "runtime_config_path": args.runtime_config or "",
        "is_sharded_run": bool(is_sharded),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_start_index": shard_start_index,
        "shard_end_index": shard_end_index,
        "full_split_sample_count": full_split_sample_count,
        "shard_sample_count": len(dataset),
    })
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    logger.info(
        f"Experiment complete: success={summary['success_count']} "
        f"fail={summary['failed_count']} skip={summary['skipped_count']}"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
