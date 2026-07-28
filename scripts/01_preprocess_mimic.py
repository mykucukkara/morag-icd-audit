import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from morag_icd.config import load_config
from morag_icd.data.load_mimic import load_mimic_notes, load_mimic_diagnoses
from morag_icd.data.preprocess_notes import process_note
from morag_icd.utils.logging_utils import setup_logger
from morag_icd.utils.timers import Timer


def _run_real_data_preprocess(
    note_path: Path,
    diag_path: Path,
    output_dir: Path,
    dry_run: bool,
    max_notes: int,
    icd_version: int,
    log,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    note_nrows = max_notes if dry_run else None
    diag_nrows = max(max_notes * 20, 20000) if dry_run else None

    with Timer("Load Notes"):
        df_notes = pd.read_csv(note_path, nrows=note_nrows, low_memory=False)
    with Timer("Load Diagnoses"):
        df_diag = pd.read_csv(diag_path, nrows=diag_nrows, low_memory=False)

    required_note_cols = ["subject_id", "hadm_id", "text"]
    required_diag_cols = ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    missing_note_cols = [c for c in required_note_cols if c not in df_notes.columns]
    missing_diag_cols = [c for c in required_diag_cols if c not in df_diag.columns]

    if missing_note_cols or missing_diag_cols:
        summary = {
            "status": "failed_missing_columns",
            "missing_note_columns": missing_note_cols,
            "missing_diag_columns": missing_diag_cols,
            "dry_run": dry_run,
            "max_notes": max_notes,
            "icd_version": icd_version,
            "phi_safe_logging": True,
        }
        (output_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.error("Required columns are missing; wrote preprocess summary and aborted.")
        return 1

    df_notes = df_notes[required_note_cols].copy()
    df_diag = df_diag[required_diag_cols].copy()

    df_notes["subject_id"] = df_notes["subject_id"].astype(str)
    df_notes["hadm_id"] = df_notes["hadm_id"].astype(str)
    df_diag["subject_id"] = df_diag["subject_id"].astype(str)
    df_diag["hadm_id"] = df_diag["hadm_id"].astype(str)
    df_diag["icd_code"] = df_diag["icd_code"].astype(str)
    df_diag["icd_version"] = pd.to_numeric(df_diag["icd_version"], errors="coerce")
    df_diag["seq_num"] = pd.to_numeric(df_diag["seq_num"], errors="coerce")

    df_diag = df_diag[df_diag["icd_version"] == icd_version].copy()
    df_diag = df_diag.sort_values(["subject_id", "hadm_id", "seq_num"], na_position="last")

    code_map: Dict[tuple, List[str]] = (
        df_diag.groupby(["subject_id", "hadm_id"])["icd_code"].apply(list).to_dict()
        if not df_diag.empty
        else {}
    )

    dataset = []
    notes_with_icd = 0
    for _, row in df_notes.iterrows():
        key = (str(row["subject_id"]), str(row["hadm_id"]))
        codes = code_map.get(key, [])
        if not codes:
            continue
        notes_with_icd += 1
        dataset.append(
            {
                "subject_id": key[0],
                "hadm_id": key[1],
                "text": str(row["text"]),
                "gold_codes": codes,
            }
        )

    dataset_path = output_dir / "dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    summary = {
        "status": "ok",
        "dry_run": dry_run,
        "max_notes": max_notes,
        "icd_version": icd_version,
        "notes_loaded": int(len(df_notes)),
        "diagnoses_loaded": int(len(df_diag)),
        "notes_with_icd": int(notes_with_icd),
        "dataset_rows": int(len(dataset)),
        "unique_subjects": int(len({d["subject_id"] for d in dataset})),
        "unique_hadm_ids": int(len({d["hadm_id"] for d in dataset})),
        "output_dataset": str(dataset_path),
        "phi_safe_logging": True,
    }
    (output_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # PHI-safe logging: report only counts and paths.
    log.info(
        "Real-data preprocessing completed: "
        f"rows={summary['dataset_rows']} notes_with_icd={summary['notes_with_icd']} "
        f"output={dataset_path}"
    )
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-notes", type=int, default=1000)
    parser.add_argument("--icd-version", type=int, default=10)
    args = parser.parse_args()
    
    paths_cfg = load_config(args.paths)
    data_cfg = load_config(args.config)
    
    # We need to substitute variables that might come from paths.yaml into data.yaml manually
    # if our config loader didn't cross-reference them.
    # A simple hack for now:
    raw_data_dir = paths_cfg.get("raw_data_dir")
    processed_dir = Path(paths_cfg.get("processed_dir"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    log = setup_logger("preprocess", paths_cfg.get("logs_dir"))
    log.info("Starting MIMIC-IV Preprocessing...")
    
    # Check if files exist
    if args.raw_root:
        raw_root = Path(args.raw_root)
        note_path = raw_root / "mimiciv_note" / "discharge.csv.gz"
        diag_path = raw_root / "mimiciv_hosp" / "diagnoses_icd.csv.gz"
    else:
        note_path = Path(str(data_cfg["mimic_note_path"]).replace("${raw_data_dir}", str(raw_data_dir)))
        diag_path = Path(str(data_cfg["mimic_hosp_diagnoses_path"]).replace("${raw_data_dir}", str(raw_data_dir)))

    if args.output:
        if not note_path.exists() or not diag_path.exists():
            summary = {
                "status": "skipped_missing_raw_files",
                "dry_run": args.dry_run,
                "max_notes": args.max_notes,
                "icd_version": args.icd_version,
                "phi_safe_logging": True,
                "missing_files": [
                    str(note_path) if not note_path.exists() else None,
                    str(diag_path) if not diag_path.exists() else None,
                ],
            }
            summary["missing_files"] = [m for m in summary["missing_files"] if m]
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log.warning("Raw files are missing; wrote preprocess summary and skipped processing.")
            return

        rc = _run_real_data_preprocess(
            note_path=note_path,
            diag_path=diag_path,
            output_dir=Path(args.output),
            dry_run=args.dry_run,
            max_notes=args.max_notes,
            icd_version=args.icd_version,
            log=log,
        )
        if rc != 0:
            raise SystemExit(rc)
        return
    
    if not note_path.exists():
        log.warning(f"Note path {note_path} does not exist. Please run smoke test data generator first if testing locally.")
        # Fallback to smoke test data for easy local testing
        smoke_dir = Path(paths_cfg["project_root"]) / "data" / "smoke_test"
        if smoke_dir.exists():
            log.info("Using smoke test data as fallback.")
            note_path = smoke_dir / "discharge.csv"
            diag_path = smoke_dir / "diagnoses_icd.csv"
        else:
            log.error("No data found. Exiting.")
            return

    with Timer("Load Notes"):
        df_notes = load_mimic_notes(note_path)

    with Timer("Load Diagnoses"):
        _ = load_mimic_diagnoses(diag_path)
    
    with Timer("Process Notes"):
        chunk_size = data_cfg.get("chunk_size", 256)
        chunk_overlap = data_cfg.get("chunk_overlap", 32)
        important_sections = [s.upper() for s in data_cfg.get("important_sections", [])]
        
        processed_data = []
        for idx, row in df_notes.iterrows():
            text = str(row.get("text", ""))
            chunks = process_note(text, chunk_size, chunk_overlap, important_sections)
            
            # Save processed note structure
            processed_data.append({
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "chunks": chunks
            })
            
            if idx > 0 and idx % 1000 == 0:
                log.info(f"Processed {idx} notes...")

    output_path = processed_dir / "processed_notes.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")
            
    log.info(f"Preprocessing completed. Saved to {output_path}")

if __name__ == "__main__":
    main()
