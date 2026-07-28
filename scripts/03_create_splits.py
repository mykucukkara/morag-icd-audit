import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from morag_icd.config import load_config
from morag_icd.data.split_builder import create_patient_level_splits
from morag_icd.utils.logging_utils import setup_logger
from morag_icd.utils.io import load_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    args = parser.parse_args()
    
    paths_cfg = load_config(args.paths)
    data_cfg = load_config(args.config)
    
    processed_dir = Path(paths_cfg.get("processed_dir"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(paths_cfg.get("project_root", "."))
    output_root = Path(args.output_root) if args.output_root else (project_root / "data" / "splits")
    splits_dir = output_root / f"top{args.top_n}"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    log = setup_logger("create_splits", paths_cfg.get("logs_dir"))
    log.info(f"Starting Split Generation for Top-{args.top_n} codes...")
    smoke_dataset = project_root / "data" / "smoke_test" / "dataset.jsonl"
    dataset_path = Path(args.dataset) if args.dataset else smoke_dataset

    if dataset_path.exists():
        records = load_jsonl(dataset_path)
        split_df = pd.DataFrame(records)
        if "gold_codes" not in split_df.columns:
            split_df["gold_codes"] = [[] for _ in range(len(split_df))]

        code_counter = Counter()
        for codes in split_df["gold_codes"]:
            if isinstance(codes, list):
                for code in codes:
                    code_counter[str(code)] += 1

        top_codes = {code for code, _ in code_counter.most_common(args.top_n)}
        if top_codes:
            split_df = split_df.copy()
            split_df["gold_codes"] = split_df["gold_codes"].apply(
                lambda codes: [str(code) for code in codes if str(code) in top_codes] if isinstance(codes, list) else []
            )
            split_df = split_df[split_df["gold_codes"].map(len) > 0].copy()

        if "split" not in split_df.columns:
            split_df = create_patient_level_splits(
                split_df,
                train_ratio=args.train_ratio if args.train_ratio is not None else data_cfg.get('train_split', 0.7),
                val_ratio=args.val_ratio if args.val_ratio is not None else data_cfg.get('val_split', 0.15),
                test_ratio=args.test_ratio if args.test_ratio is not None else data_cfg.get('test_split', 0.15),
            )
        log.info(f"Loaded dataset: {len(split_df)} rows from {dataset_path}")
    else:
        log.error(f"Dataset not found at {dataset_path}. Run preprocessing first.")
        return

    output_path = (output_root / f"top{args.top_n}" / f"splits_top_{args.top_n}.csv") if (args.output_root or args.dataset) else (processed_dir / f"splits_top_{args.top_n}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)

    all_codes = set()
    for codes in split_df.get("gold_codes", []):
        if isinstance(codes, list):
            all_codes.update(str(code) for code in codes)

    split_summary = {
        "top_n": args.top_n,
        "dataset_path": str(dataset_path),
        "output_root": str(output_root),
        "rows": int(len(split_df)),
        "unique_subjects": int(split_df["subject_id"].nunique()) if not split_df.empty else 0,
        "unique_hadm_ids": int(split_df["hadm_id"].nunique()) if not split_df.empty else 0,
        "split_counts": split_df["split"].value_counts().to_dict() if "split" in split_df.columns else {},
        "unique_code_count": int(len(all_codes)),
        "output_csv": str(output_path),
    }
    with open(splits_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    for split_name in ["train", "validation", "test"]:
        split_rows = split_df[split_df["split"] == split_name]
        out_path = splits_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in split_rows.iterrows():
                record = {
                    "subject_id": str(row.get("subject_id", "")),
                    "hadm_id": str(row.get("hadm_id", "")),
                    "text": str(row.get("text", "")),
                    "gold_codes": row.get("gold_codes", []) if isinstance(row.get("gold_codes", []), list) else [],
                    "split": split_name,
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        log.info(f"Wrote {len(split_rows)} rows to {out_path}")

    log.info(f"Split sizes:\n{split_df['split'].value_counts()}")
    log.info(f"Saved splits to {output_path}")

if __name__ == "__main__":
    main()
