"""
Script 11: Generate final tables, figures, and academic reports.

Reads all metrics from the metrics directory and produces:
- CSV/Markdown/LaTeX tables (Tables 1-6)
- Publication-quality figures (10 plots)
- Markdown and HTML final report
- Reproducibility checklist

Usage:
    python scripts/11_generate_report.py \\
        --metrics-dir results/metrics \\
        --output-dir results/reports \\
        [--top-n 50] \\
        [--no-plots]
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Generate MORAG-ICD final reports.")
    p.add_argument("--metrics-dir", default="results/metrics")
    p.add_argument("--output-dir", default="results/reports")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    p.add_argument("--no-latex", action="store_true", help="Skip LaTeX table export")
    return p.parse_args()


def main():
    args = parse_args()

    metrics_dir = Path(args.metrics_dir) / f"top{args.top_n}"
    output_dir = Path(args.output_dir)
    tables_dir = Path("results/tables")

    for d in [output_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load consolidated metrics
    consolidated_path = metrics_dir / "consolidated_metrics.json"
    if not consolidated_path.exists():
        print(f"Warning: consolidated metrics not found at {consolidated_path}")
        consolidated = {"individual": {}, "seed_statistics": {}}
    else:
        with open(consolidated_path, "r") as f:
            consolidated = json.load(f)

    seed_stats = consolidated.get("seed_statistics", {})
    individual = consolidated.get("individual", {})

    print(f"Loaded metrics for {len(seed_stats)} experiments")

    # Build compact smoke table for E1/E4/E6 only.
    rows = []
    for exp_id in ["E1", "E4", "E6"]:
        stats = seed_stats.get(exp_id, {})
        if not stats:
            continue
        rows.append(
            {
                "experiment_id": exp_id,
                "micro_f1": float(stats.get("micro_f1", {}).get("mean", 0.0)),
                "macro_f1": float(stats.get("macro_f1", {}).get("mean", 0.0)),
                "precision_at_5": float(stats.get("precision_at_5", {}).get("mean", 0.0)),
                "recall_at_10": float(stats.get("recall_at_10", {}).get("mean", 0.0)),
                "hamming_loss": float(stats.get("hamming_loss", {}).get("mean", 1.0)),
                "exact_match": float(stats.get("exact_match_ratio", {}).get("mean", 0.0)),
            }
        )

    csv_out = tables_dir / "baseline_smoke_metrics.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_id",
                "micro_f1",
                "macro_f1",
                "precision_at_5",
                "recall_at_10",
                "hamming_loss",
                "exact_match",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    rag_rows = []
    for exp_id in ["E9", "E10", "E11", "E12", "E13", "E14"]:
        run = _first_run(individual, exp_id)
        if not run:
            continue
        rag_rows.append(
            {
                "experiment_id": exp_id,
                "retrieval_mode": run.get("retrieval_mode", "hybrid"),
                "evidence_constraint": bool(run.get("use_evidence_constraint", False)),
                "contrastive_verifier": bool(run.get("use_contrastive_verifier", False)),
                "micro_f1": run.get("classification", {}).get("micro_f1", 0.0),
                "macro_f1": run.get("classification", {}).get("macro_f1", 0.0),
                "precision_at_5": run.get("classification", {}).get("precision_at_5", 0.0),
                "recall_at_10": run.get("classification", {}).get("recall_at_10", 0.0),
                "evidence_support_rate": run.get("reliability", {}).get("evidence_support_rate", 0.0),
                "unsupported_code_rate": run.get("reliability", {}).get("unsupported_code_rate", 0.0),
                "rationale_coverage_rate": run.get("reliability", {}).get("rationale_coverage_rate", 0.0),
                "weak_evidence_rate": run.get("reliability", {}).get("weak_evidence_rate", 0.0),
                "similar_code_confusion_rate": run.get("reliability", {}).get("similar_code_confusion_rate", 0.0),
                "mock_llm": bool(run.get("mock_llm", False)),
                "mock_embedding": bool(run.get("mock_embedding", False)),
                "skipped_dense": bool(run.get("skipped_dense", False)),
            }
        )

    rag_csv_out = tables_dir / "rag_ablation_smoke_metrics.csv"
    with open(rag_csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_id",
                "retrieval_mode",
                "evidence_constraint",
                "contrastive_verifier",
                "micro_f1",
                "macro_f1",
                "precision_at_5",
                "recall_at_10",
                "evidence_support_rate",
                "unsupported_code_rate",
                "rationale_coverage_rate",
                "weak_evidence_rate",
                "similar_code_confusion_rate",
                "mock_llm",
                "mock_embedding",
                "skipped_dense",
            ],
        )
        writer.writeheader()
        for row in rag_rows:
            writer.writerow(row)

    print("Generating final report...")
    report_path = output_dir / "final_experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MORAG-ICD Smoke Report\n\n")
        f.write("This report summarizes smoke-test runs only. These results are not final scientific results.\n\n")
        f.write(f"Top-N: {args.top_n}\n\n")
        if not rows:
            f.write("no valid metrics available\n")
        else:
            f.write("## Phase 1 Smoke Summary\n\n")
            f.write("| experiment_id | micro_f1 | macro_f1 | precision_at_5 | recall_at_10 | hamming_loss | exact_match |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(
                    f"| {row['experiment_id']} | {row['micro_f1']:.4f} | {row['macro_f1']:.4f} | "
                    f"{row['precision_at_5']:.4f} | {row['recall_at_10']:.4f} | {row['hamming_loss']:.4f} | {row['exact_match']:.4f} |\n"
                )
            f.write("\n## Phase 2 RAG Smoke Test Summary\n\n")
            if not rag_rows:
                f.write("no valid metrics available\n")
            else:
                f.write("| experiment_id | retrieval_mode | evidence_constraint | contrastive_verifier | micro_f1 | macro_f1 | precision_at_5 | recall_at_10 | evidence_support_rate | unsupported_code_rate | rationale_coverage_rate | weak_evidence_rate | similar_code_confusion_rate | mock_llm | mock_embedding | skipped_dense |\n")
                f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                for row in rag_rows:
                    f.write(
                        f"| {row['experiment_id']} | {row['retrieval_mode']} | {str(row['evidence_constraint']).lower()} | {str(row['contrastive_verifier']).lower()} | "
                        f"{_fmt_float(row['micro_f1'])} | {_fmt_float(row['macro_f1'])} | {_fmt_float(row['precision_at_5'])} | {_fmt_float(row['recall_at_10'])} | "
                        f"{_fmt_float(row['evidence_support_rate'])} | {_fmt_float(row['unsupported_code_rate'])} | {_fmt_float(row['rationale_coverage_rate'])} | {_fmt_float(row['weak_evidence_rate'])} | {_fmt_float(row['similar_code_confusion_rate'])} | "
                        f"{str(row['mock_llm']).lower()} | {str(row['mock_embedding']).lower()} | {str(row['skipped_dense']).lower()} |\n"
                    )

    # === Reproducibility Checklist ===
    _generate_reproducibility_checklist(output_dir / "reproducibility_checklist.md", args)

    # === Method Summary ===
    _generate_method_summary(output_dir / "method_summary.md")

    # === Experiment Protocol ===
    _generate_experiment_protocol(output_dir / "experiment_protocol.md")

    print(f"\n=== Report Generation Complete ===")
    print(f"Tables: {tables_dir}")
    print(f"Report: {report_path}")


def _generate_reproducibility_checklist(output_path: Path, args):
    """Generate reproducibility checklist."""
    try:
        py_version = platform.python_version()
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        py_version = sys.version.split()[0]
        git_hash = "N/A"

    content = f"""# Reproducibility Checklist

Generated: {datetime.now(timezone.utc).isoformat()}

## Environment
- Python version: {py_version}
- Git commit: {git_hash}
- Platform: {platform.platform()}

## Seeds
- Training/evaluation seeds: 42, 52, 62

## Data
- Dataset: MIMIC-IV discharge summaries
- Patient-level split: 70% train / 15% validation / 15% test
- Same subject_id never appears in multiple splits
- Top-N codes evaluated: {args.top_n}

## Config Hash
- All configs are hashed per run and stored in prediction files

## Model Paths
- All models loaded from local paths (no internet access required)
- See configs/models.yaml for exact paths

## SLURM Jobs
- See slurm/ directory for SBATCH scripts
- Job manifest: results/job_manifest.csv

## Verification
- Run `python scripts/00_create_smoke_test_data.py` to verify pipeline
- Run `python scripts/09_run_all_experiments.py --smoke-test` for end-to-end test
"""
    output_path.write_text(content, encoding="utf-8")


def _first_run(individual: dict, exp_id: str):
    for run_id, metrics in individual.items():
        if run_id.startswith(exp_id + "_"):
            return metrics
    return None


def _fmt_float(value):
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _generate_method_summary(output_path: Path):
    content = """# MORAG-ICD Method Summary

## Architecture Overview

MORAG-ICD is a hybrid RAG-LLM framework for explainable ICD-10 code recommendation.

## Components

1. **Data Pipeline**: MIMIC-IV preprocessing, section splitting, chunking
2. **ICD Knowledge Base**: BM25 + dense indexed ICD-10 code descriptions
3. **Hybrid Retriever**: alpha-weighted BM25 + FAISS dense retrieval
4. **LLM Code Scorer**: Local LLM evaluates evidence support per ICD candidate
5. **Contrastive Verifier**: Disambiguates similar ICD codes within same family
6. **Multi-objective Optimizer**: MOPSO/NSGA-II tunes RAG hyperparameters

## Optimization Objectives

| Objective | Direction |
|-----------|-----------|
| Micro-F1 | Maximize |
| Macro-F1 | Maximize |
| Evidence Support Rate | Maximize |
| Unsupported Code Rate | Minimize |
| Similar-Code Confusion | Minimize |
| Avg. Inference Time | Minimize |

## Key Innovation

Unlike fixed-parameter RAG systems, MORAG-ICD frames ICD coding as a
multi-objective optimization problem, producing a Pareto front of solutions
that trade off accuracy, explainability, and computational cost.
"""
    output_path.write_text(content, encoding="utf-8")


def _generate_experiment_protocol(output_path: Path):
    content = """# Experiment Protocol

## Dataset
- MIMIC-IV discharge summaries (discharge.csv.gz)
- ICD-10 diagnosis codes (diagnoses_icd.csv.gz, d_icd_diagnoses.csv.gz)

## Splits
- Patient-level: 70% train / 15% validation / 15% test
- Same subject_id never crosses splits

## Experiments (E1–E18)

| ID | Name | Type |
|----|------|------|
| E1 | TF-IDF + Logistic Regression | Baseline |
| E2 | TF-IDF + SVM | Baseline |
| E3 | BioClinicalBERT Classifier | Baseline |
| E4 | BM25 Retrieval-only | Retrieval |
| E5 | Dense Retrieval-only | Retrieval |
| E6 | Hybrid Retrieval-only | Retrieval |
| E7 | LLM Zero-shot | LLM |
| E8 | LLM Few-shot | LLM |
| E9 | BM25-RAG | RAG |
| E10 | Dense-RAG | RAG |
| E11 | Hybrid-RAG | RAG |
| E12 | Hybrid-RAG + Evidence Constraint | Enhanced RAG |
| E13 | Hybrid-RAG + Contrastive Verifier | Enhanced RAG |
| E14 | Full Model (no optimization) | Full Model |
| E15 | Full Model + Random Search | Optimized |
| E16 | Full Model + MOPSO | Optimized |
| E17 | Full Model + NSGA-II | Optimized |
| E18 | Scalability Analysis | Scale Test |

## Seeds: 42, 52, 62
## Top-N: 50 (primary), 100, 200 (E18 scalability)
## Total Runs: 75
"""
    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
