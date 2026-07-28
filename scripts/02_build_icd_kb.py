import argparse
import json
from pathlib import Path

from morag_icd.config import load_config
from morag_icd.data.icd_kb_builder import build_icd_knowledge_base
from morag_icd.data.load_mimic import load_icd_kb
from morag_icd.utils.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--icd-version", type=int, default=10)
    args = parser.parse_args()

    paths_cfg = load_config(args.paths)
    data_cfg = load_config(args.config)

    raw_data_dir = paths_cfg.get("raw_data_dir")
    processed_dir = Path(paths_cfg.get("processed_dir"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger("build_kb", paths_cfg.get("logs_dir"))
    log.info("Starting ICD Knowledge Base Build...")

    diag_path = Path(args.input) if args.input else Path(str(data_cfg["mimic_hosp_d_icd_path"]).replace("${raw_data_dir}", str(raw_data_dir)))

    if not diag_path.exists():
        smoke_dir = Path(paths_cfg["project_root"]) / "data" / "smoke_test"
        if smoke_dir.exists():
            log.info("Using smoke test data as fallback.")
            diag_path = smoke_dir / "d_icd_diagnoses.csv"
        else:
            log.error("No data found. Exiting.")
            return

    df_d_icd = load_icd_kb(diag_path)
    kb = build_icd_knowledge_base(df_d_icd, icd_version=args.icd_version)

    output_path = Path(args.output) if args.output else (processed_dir / "icd_kb.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in kb:
            f.write(json.dumps(item) + "\n")

    log.info(f"Built KB with {len(kb)} ICD-{args.icd_version} codes. Saved to {output_path}")


if __name__ == "__main__":
    main()
