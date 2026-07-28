import argparse
import json
from pathlib import Path

from morag_icd.config import load_config
from morag_icd.retrieval.bm25_index import BM25
from morag_icd.utils.logging_utils import setup_logger
from morag_icd.utils.io import load_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--top-n", type=int, default=50) # not strictly needed here unless we filter
    parser.add_argument("--splits-root", default=None)
    parser.add_argument("--icd-kb", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    
    paths_cfg = load_config(args.paths)
    project_root = Path(paths_cfg.get("project_root", "."))
    
    processed_dir = Path(paths_cfg.get("processed_dir"))
    indexes_dir = Path(paths_cfg.get("indexes_dir"))
    bm25_dir = Path(args.output_root) if args.output_root else (indexes_dir / "bm25")
    bm25_dir.mkdir(parents=True, exist_ok=True)
    
    log = setup_logger("build_bm25", paths_cfg.get("logs_dir"))
    log.info("Starting BM25 Index Build for ICD Knowledge Base...")
    
    icd_kb_path = Path(args.icd_kb) if args.icd_kb else (processed_dir / "icd_kb.jsonl")
    
    if not icd_kb_path.exists():
        log.error(f"ICD KB not found at {icd_kb_path}. Please run 02_build_icd_kb.py first.")
        return
        
    kb_docs = []
    with open(icd_kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_docs.append(json.loads(line))
            
    # Filter KB to top_n codes if needed, or index all of them. 
    # Usually it's fine to index all and filter later, but let's stick to the full KB.
    log.info(f"Loaded {len(kb_docs)} ICD codes. Fitting BM25...")
    
    bm25 = BM25()
    bm25.fit(kb_docs, text_field="searchable_text")
    
    output_path = bm25_dir / f"icd_bm25_{args.top_n}.pkl"
    bm25.save(output_path)

    # Backward-compatible alias used by older components.
    legacy_output_path = bm25_dir / "icd_kb_bm25.pkl"
    bm25.save(legacy_output_path)

    # Build evidence BM25 from smoke dataset (or fallback to processed note chunks later).
    split_root = Path(args.splits_root) if args.splits_root else (project_root / "data" / "smoke_test")
    evidence_docs = []
    split_dir = split_root / f"top{args.top_n}"
    split_files = [split_dir / name for name in ["train.jsonl", "validation.jsonl", "test.jsonl"]]
    if all(path.exists() for path in split_files):
        for split_path in split_files:
            rows = load_jsonl(split_path)
            split_name = split_path.stem
            for i, row in enumerate(rows):
                text = str(row.get("text", ""))
                evidence_docs.append(
                    {
                        "subject_id": str(row.get("subject_id", "")),
                        "hadm_id": str(row.get("hadm_id", "")),
                        "chunk_id": f"{split_name}_{i}",
                        "section_name": "REAL_NOTE",
                        "text": text,
                        "searchable_text": text,
                        "token_count": len(text.split()),
                    }
                )
    elif (split_root / "dataset.jsonl").exists():
        rows = load_jsonl(split_root / "dataset.jsonl")
        for i, row in enumerate(rows):
            text = str(row.get("text", ""))
            evidence_docs.append(
                {
                    "subject_id": str(row.get("subject_id", "")),
                    "hadm_id": str(row.get("hadm_id", "")),
                    "chunk_id": f"dataset_{i}",
                    "section_name": "REAL_NOTE",
                    "text": text,
                    "searchable_text": text,
                    "token_count": len(text.split()),
                }
            )
    elif split_root.name == "smoke_test" and (split_root / "dataset.jsonl").exists():
        rows = load_jsonl(split_root / "dataset.jsonl")
        for i, row in enumerate(rows):
            text = str(row.get("text", ""))
            evidence_docs.append(
                {
                    "subject_id": str(row.get("subject_id", "")),
                    "hadm_id": str(row.get("hadm_id", "")),
                    "chunk_id": f"smoke_{i}",
                    "section_name": "SMOKE_NOTE",
                    "text": text,
                    "searchable_text": text,
                    "token_count": len(text.split()),
                }
            )

    if evidence_docs:
        evidence_bm25 = BM25()
        evidence_bm25.fit(evidence_docs, text_field="searchable_text")
        evidence_output = bm25_dir / f"evidence_bm25_{args.top_n}.pkl"
        evidence_bm25.save(evidence_output)
        log.info(f"Evidence BM25 Index saved to {evidence_output} ({len(evidence_docs)} docs)")
    else:
        log.warning("No evidence documents found; evidence BM25 was not created.")
    
    log.info(f"BM25 Index saved to {output_path}")

if __name__ == "__main__":
    main()
