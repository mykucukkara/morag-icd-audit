from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable


PHASE2_RAG_EXPERIMENTS = {"E9", "E10", "E11", "E12", "E13", "E14"}
HYBRID_EXPERIMENTS = {"E11", "E12", "E13", "E14"}


def is_resolved_local_path(path_value: str | None) -> bool:
    if not path_value:
        return False
    raw = str(path_value).strip()
    if not raw or "${" in raw:
        return False
    return Path(raw).exists()


def check_model_path(path: str | None, required: bool = True, label: str = "model") -> Dict[str, object]:
    raw_path = "" if path is None else str(path)
    exists = is_resolved_local_path(raw_path)
    if exists:
        status = "available"
        message = f"{label} path is available"
    elif required:
        status = "missing"
        message = f"{label} path is missing or unresolved"
    else:
        status = "not_required"
        message = f"{label} path is not required"
    return {
        "exists": exists,
        "path": raw_path,
        "status": status,
        "message": message,
    }


def check_artifact_paths(
    paths: Iterable[str | Path],
    required: bool = True,
    label: str = "artifact",
) -> Dict[str, object]:
    path_list = [str(Path(path)) for path in paths]
    exists = all(Path(path).exists() for path in path_list)
    if exists:
        status = "available"
        message = f"{label} is available"
    elif required:
        status = "missing"
        message = f"{label} is missing"
    else:
        status = "not_required"
        message = f"{label} is not required"
    return {
        "exists": exists,
        "paths": path_list,
        "status": status,
        "message": message,
    }


def collect_environment_readiness(
    paths_config: Dict,
    model_config: Dict,
    top_n: int,
    smoke_test: bool,
    split: str = "test",
) -> Dict[str, object]:
    project_root = Path(paths_config.get("project_root", "."))
    indexes_dir = Path(paths_config.get("indexes_dir", project_root / "indexes"))

    smoke_dataset = project_root / "data" / "smoke_test" / "dataset.jsonl"
    splits_root = Path(paths_config.get("splits_root", project_root / "data" / "splits"))
    topn_dir = splits_root / f"top{top_n}"
    split_files = [topn_dir / name for name in ["train.jsonl", "validation.jsonl", "test.jsonl"]]

    env = {
        "models": {
            "embedding": check_model_path(model_config.get("embedding_model_path"), required=False, label="embedding model"),
            "llm": check_model_path(model_config.get("llm_model_path"), required=False, label="llm model"),
            "classifier": check_model_path(model_config.get("classifier_model_path"), required=False, label="classifier model"),
        },
        "indexes": {
            "bm25": check_artifact_paths(
                [
                    indexes_dir / "bm25" / f"icd_bm25_{top_n}.pkl",
                    indexes_dir / "bm25" / f"evidence_bm25_{top_n}.pkl",
                ],
                required=True,
                label=f"bm25 indexes top{top_n}",
            ),
            "dense": check_artifact_paths(
                [
                    indexes_dir / "faiss" / f"icd_dense_{top_n}.pkl",
                    indexes_dir / "faiss" / f"evidence_dense_{top_n}.pkl",
                ],
                required=False,
                label=f"dense indexes top{top_n}",
            ),
        },
        "data": {
            "smoke_split": {
                "exists": smoke_dataset.exists(),
                "path": str(smoke_dataset),
                "status": "available" if smoke_dataset.exists() else "missing",
                "message": "smoke dataset is available" if smoke_dataset.exists() else "smoke dataset is missing",
            },
            "topn_split": {
                "exists": all(path.exists() for path in split_files),
                "paths": [str(path) for path in split_files],
                "status": "available" if all(path.exists() for path in split_files) else "missing",
                "message": f"top{top_n} split files are available" if all(path.exists() for path in split_files) else f"top{top_n} split files are missing",
            },
            "requested_split": {
                "exists": smoke_dataset.exists() if smoke_test else (topn_dir / f"{split}.jsonl").exists(),
                "path": str(smoke_dataset if smoke_test else (topn_dir / f"{split}.jsonl")),
                "status": "available" if (smoke_dataset.exists() if smoke_test else (topn_dir / f"{split}.jsonl").exists()) else "missing",
                "message": "requested split is available" if (smoke_dataset.exists() if smoke_test else (topn_dir / f"{split}.jsonl").exists()) else "requested split is missing",
            },
        },
    }
    return env


def build_experiment_readiness(
    experiment_id: str,
    top_n: int,
    smoke_test: bool,
    split: str,
    paths_config: Dict,
    model_config: Dict,
    allow_mock_llm: bool = False,
    allow_mock_embedding: bool = False,
) -> Dict[str, object]:
    env = collect_environment_readiness(paths_config, model_config, top_n, smoke_test, split=split)
    needs_llm = experiment_id in PHASE2_RAG_EXPERIMENTS
    requires_dense = experiment_id == "E10"
    hybrid_dense_optional = experiment_id in HYBRID_EXPERIMENTS

    llm_ready = bool(env["models"]["llm"]["exists"])
    embedding_ready = bool(env["models"]["embedding"]["exists"])
    bm25_ready = bool(env["indexes"]["bm25"]["exists"])
    dense_ready = bool(env["indexes"]["dense"]["exists"])
    data_ready = bool(env["data"]["requested_split"]["exists"])

    errors = []
    notes = []

    if not data_ready:
        errors.append(f"missing data path: {env['data']['requested_split']['path']}")
    if experiment_id in PHASE2_RAG_EXPERIMENTS and not bm25_ready:
        errors.append("missing top50 BM25 indexes")
    if needs_llm and not llm_ready and not allow_mock_llm:
        errors.append("missing llm model path and mock llm disabled")
    if requires_dense:
        if not embedding_ready and not allow_mock_embedding:
            errors.append("missing embedding model path and mock embedding disabled")
        if not dense_ready and not allow_mock_embedding:
            errors.append("missing dense indexes for dense experiment")
    elif hybrid_dense_optional:
        if embedding_ready and dense_ready:
            notes.append("hybrid dense retrieval available")
        elif allow_mock_embedding:
            notes.append("hybrid run may use explicit mock embedding")
        else:
            notes.append("hybrid run will fall back to BM25-only retrieval")

    can_run = not errors
    if can_run:
        if requires_dense:
            mode = "non_mock_dense" if embedding_ready and dense_ready else "mock_dense"
        elif hybrid_dense_optional:
            if embedding_ready and dense_ready:
                mode = "non_mock_hybrid"
            elif allow_mock_embedding:
                mode = "mock_hybrid"
            else:
                mode = "bm25_fallback"
        else:
            mode = "non_mock_bm25"
    else:
        mode = "blocked"

    return {
        "experiment_id": experiment_id,
        "status": "ready" if can_run else "skipped_missing_model",
        "mode": mode,
        "errors": errors,
        "notes": notes,
        "allow_mock_llm": bool(allow_mock_llm),
        "allow_mock_embedding": bool(allow_mock_embedding),
        "models": env["models"],
        "indexes": env["indexes"],
        "data": env["data"],
        "checks": {
            "needs_llm": needs_llm,
            "requires_dense": requires_dense,
            "hybrid_dense_optional": hybrid_dense_optional,
            "llm_model_path_valid": llm_ready,
            "embedding_model_path_valid": embedding_ready,
            "bm25_indexes_exist": bm25_ready,
            "dense_index_exists": dense_ready,
            "data_path_exists": data_ready,
        },
    }


def summarize_phase2_rag_non_mock_readiness(
    paths_config: Dict,
    model_config: Dict,
    top_n: int,
    smoke_test: bool,
    split: str = "test",
) -> Dict[str, object]:
    experiments = {}
    runnable = []
    skipped = []
    for exp_id in ["E9", "E10", "E11", "E12", "E13", "E14"]:
        readiness = build_experiment_readiness(
            experiment_id=exp_id,
            top_n=top_n,
            smoke_test=smoke_test,
            split=split,
            paths_config=paths_config,
            model_config=model_config,
            allow_mock_llm=False,
            allow_mock_embedding=False,
        )
        experiments[exp_id] = readiness
        if readiness["status"] == "ready":
            runnable.append(exp_id)
        else:
            skipped.append({
                "experiment_id": exp_id,
                "reasons": readiness["errors"],
            })
    return {
        "runnable_experiments": runnable,
        "skipped_experiments": skipped,
        "experiments": experiments,
    }
