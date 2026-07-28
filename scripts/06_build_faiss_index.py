import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morag_icd.config import load_config
from morag_icd.retrieval.bm25_index import BM25
from morag_icd.retrieval.dense_index import DenseIndex
from morag_icd.utils.logging_utils import setup_logger
from morag_icd.utils.model_readiness import check_model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--splits-root", default=None)
    parser.add_argument("--icd-kb", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail fast if the requested/selected device is CUDA but CUDA is unavailable.")
    parser.add_argument("--require-faiss", action="store_true",
                        help="Fail fast if FAISS is unavailable instead of writing numpy-only dense indexes.")
    args = parser.parse_args()
    
    paths_cfg = load_config(args.paths)
    models_cfg = load_config(args.models)
    
    indexes_dir = Path(paths_cfg.get("indexes_dir"))
    faiss_dir = Path(args.output_root) if args.output_root else (indexes_dir / "faiss")
    faiss_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(paths_cfg.get("results_dir", "results")) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    log = setup_logger("build_faiss", paths_cfg.get("logs_dir"))
    log.info("Starting FAISS Dense Index Build for Phase 2 retrieval indexes...")

    bm25_root = Path(args.output_root).parent / "bm25" if args.output_root else (indexes_dir / "bm25")
    model_path = args.embedding_model or models_cfg.get("embedding_model_path")
    model_check = check_model_path(model_path, required=True, label="embedding model")
    icd_bm25_path = bm25_root / f"icd_bm25_{args.top_n}.pkl"
    evidence_bm25_path = bm25_root / f"evidence_bm25_{args.top_n}.pkl"

    summary = {
        "top_n": args.top_n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model_check,
        "indexes": {
            "icd_bm25": str(icd_bm25_path),
            "evidence_bm25": str(evidence_bm25_path),
        },
        "inputs": {
            "splits_root": str(args.splits_root) if args.splits_root else None,
            "icd_kb": str(args.icd_kb) if args.icd_kb else None,
            "output_root": str(args.output_root) if args.output_root else None,
            "embedding_model_override": str(args.embedding_model) if args.embedding_model else None,
        },
        "status": "ready",
        "outputs": {},
        "mock_embedding": False,
        "errors": [],
    }

    if not model_check["exists"]:
        summary["status"] = "skipped_missing_model"
        summary["errors"].append(model_check["message"])
    if not icd_bm25_path.exists():
        summary["status"] = "skipped_missing_model"
        summary["errors"].append(f"missing BM25 index: {icd_bm25_path}")
    if not evidence_bm25_path.exists():
        summary["status"] = "skipped_missing_model"
        summary["errors"].append(f"missing BM25 index: {evidence_bm25_path}")
    if args.require_faiss:
        try:
            import faiss  # noqa: F401
        except Exception as exc:
            summary["status"] = "failed_missing_faiss"
            summary["errors"].append(f"FAISS is required but unavailable: {type(exc).__name__}: {exc}")

    summary_path = reports_dir / f"faiss_build_summary_top{args.top_n}.json"
    if summary["status"] != "ready":
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        sys.exit(1)

    device = args.device or models_cfg.get("device", models_cfg.get("fallback_device", "cpu"))
    try:
        import torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            if args.require_cuda:
                raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
            device = models_cfg.get("fallback_device", "cpu")
    except Exception as exc:
        if args.require_cuda:
            summary["status"] = "failed_cuda_unavailable"
            summary["errors"].append(f"CUDA fail-fast check failed: {type(exc).__name__}: {exc}")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
            sys.exit(1)
        if str(device).startswith("cuda"):
            device = models_cfg.get("fallback_device", "cpu")
    summary["device"] = device
    model_path = str(model_path)
    output_specs = [
        (icd_bm25_path, faiss_dir / f"icd_dense_{args.top_n}"),
        (evidence_bm25_path, faiss_dir / f"evidence_dense_{args.top_n}"),
    ]

    for bm25_path, output_path in output_specs:
        bm25 = BM25.load(bm25_path)
        docs = bm25.docs or []
        dense_index = DenseIndex(model_name=model_path, device=device, allow_mock_embedding=False)
        log.info(f"Generating dense index for {bm25_path.name} with {len(docs)} documents...")
        dense_index.fit(docs, text_field="searchable_text")
        dense_index.save(output_path)
        pkl_path = output_path.with_suffix(".pkl")
        index_path = output_path.with_suffix(".index")
        summary["outputs"][output_path.name] = {
            "index_path": str(index_path),
            "pkl_path": str(pkl_path),
            "doc_count": len(docs),
            "embedding_dimension": dense_index.dim,
            "mock_embedding": bool(dense_index.mock_embedding),
            "pkl_size_bytes": pkl_path.stat().st_size if pkl_path.exists() else 0,
            "index_size_bytes": index_path.stat().st_size if index_path.exists() else 0,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Dense index summary saved to {summary_path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
