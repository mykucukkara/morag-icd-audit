"""
Script 34: Generate data-driven paper tables from corrected consolidated metrics.

Reads results/metrics/top{N}/consolidated_metrics.json (produced by the corrected
scripts/10_evaluate_results.py: fixed Top-N label space, all-group seed aggregation with
sample std) and emits publication tables as CSV + Markdown + LaTeX. Purely data-driven —
no hardcoded narrative, no always-"—" columns (contrast with the legacy reporting path).

Usage:
    python scripts/34_generate_corrected_paper_tables.py --top-n 50 \\
        --metrics-dir results/metrics --output-dir results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from morag_icd.reporting.provenance_guard import check_row, FixtureContaminationError

EXPERIMENT_ORDER = [f"E{i}" for i in range(1, 18)]

# (column header, "group.metric" key)
MAIN_COLS = [
    ("Micro-F1", "classification.micro_f1"),
    ("Macro-F1", "classification.macro_f1"),
    ("P@5", "classification.precision_at_5"),
    ("R@10", "classification.recall_at_10"),
    ("Hamming", "classification.hamming_loss"),
    ("Evid.Support", "reliability.evidence_support_rate"),
    ("Unsupported", "reliability.unsupported_code_rate"),
]
OPT_COLS = [
    ("Micro-F1", "classification.micro_f1"),
    ("Macro-F1", "classification.macro_f1"),
    ("Evid.Support", "reliability.evidence_support_rate"),
    ("Unsupported", "reliability.unsupported_code_rate"),
    ("SimConfusion", "reliability.similar_code_confusion_rate"),
    ("Avg.Runtime(s)", "cost.avg_runtime_sec"),
]
RELIABILITY_COLS = [
    ("Evid.Support", "reliability.evidence_support_rate"),
    ("Unsupported", "reliability.unsupported_code_rate"),
    ("WeakEvid", "reliability.weak_evidence_rate"),
    ("SimConfusion", "reliability.similar_code_confusion_rate"),
    ("Rejected", "reliability.rejected_similar_codes_rate"),
    ("Contr.Fallback", "reliability.contrastive_fallback_rate"),
    ("LLM.Contr", "reliability.llm_contrastive_rate"),
    ("SchemaInvalid", "reliability.schema_invalid_rate"),
    ("JSONParseErr", "reliability.json_parse_error_rate"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate corrected paper tables.")
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--metrics-dir", default="results/metrics")
    p.add_argument("--output-dir", default="results")
    # A real Top-N test run has thousands of notes; fixtures/canaries have 5-100. Any
    # experiment below this is refused (see --allow-pilot to override for a pilot report).
    p.add_argument("--min-samples", type=int, default=1000,
                   help="Minimum n_samples per experiment for a manuscript table (default 1000).")
    p.add_argument("--allow-pilot", action="store_true",
                   help="Downgrade the provenance guard to warnings (pilot/canary reports only).")
    return p.parse_args()


def experiment_min_samples(individual: Dict) -> Dict[str, int]:
    """Min n_samples per experiment across its seeds, from the `individual` block.

    seed_statistics (what the tables are built from) drops n_samples, so a fixture run
    (n_samples=10) is indistinguishable there from a real one; recover the provenance here.
    """
    out: Dict[str, int] = {}
    for run_key, run in (individual or {}).items():
        if not isinstance(run, dict):
            continue
        exp = str(run.get("experiment_id") or run_key.split("_seed")[0])
        n = run.get("n_samples")
        if isinstance(n, (int, float)):
            out[exp] = min(out.get(exp, int(n)), int(n))
    return out


def fmt(stats: Dict, key: str) -> str:
    v = stats.get(key)
    if not isinstance(v, dict) or v.get("mean") is None:
        return "—"
    mean = v["mean"]
    std = v.get("std", 0.0)
    n = v.get("n", 1)
    return f"{mean:.4f} ± {std:.4f}" if n and n > 1 else f"{mean:.4f}"


def build_table(seed_stats: Dict, experiments: List[str], cols) -> List[List[str]]:
    header = ["Experiment"] + [c[0] for c in cols]
    rows = [header]
    for exp in experiments:
        if exp not in seed_stats:
            continue
        st = seed_stats[exp]
        rows.append([exp] + [fmt(st, key) for _, key in cols])
    return rows


def save_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def save_md(rows, path: Path, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"### {title}", "", "| " + " | ".join(rows[0]) + " |",
             "|" + "|".join(["---"] * len(rows[0])) + "|"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_latex(rows, path: Path, caption: str, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    ncol = len(rows[0])
    out = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\begin{tabular}{l" + "c" * (ncol - 1) + "}", r"\toprule",
           " & ".join(_tex_escape(c) for c in rows[0]) + r" \\", r"\midrule"]
    for r in rows[1:]:
        out.append(" & ".join(_tex_escape(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _tex_escape(s: str) -> str:
    s = str(s).replace("±", r"$\pm$").replace("—", r"--").replace("_", r"\_").replace("@", r"@").replace("%", r"\%")
    return s


def emit(rows, base: Path, stem: str, title: str, label: str, tables_dir: Path, latex_dir: Path):
    save_csv(rows, tables_dir / f"{stem}.csv")
    save_md(rows, tables_dir / f"{stem}.md", title)
    save_latex(rows, latex_dir / f"{stem}.tex", title, label)


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_dir) / f"top{args.top_n}" / "consolidated_metrics.json"
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found. Run scripts/10_evaluate_results.py first.")
        raise SystemExit(1)
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    seed_stats = data.get("seed_statistics", {})
    if not data.get("fixed_label_space"):
        print("WARNING: metrics were NOT computed on a fixed label space; tables may not be comparable.")

    # PROVENANCE GUARD — refuse to emit manuscript tables built from fixture / under-powered
    # runs. This is the choke point the panel flagged: the smoke fixture (all micro_f1 =
    # 0.42105263, success_count=6) must never reach a paper table.
    min_samples = experiment_min_samples(data.get("individual", {}))
    problems = []
    for exp in EXPERIMENT_ORDER:
        if exp not in seed_stats:
            continue
        mf1 = (seed_stats[exp].get("micro_f1") or {}).get("mean")
        reason = check_row(exp, mf1, min_samples.get(exp), args.min_samples, strict=False)
        if reason:
            problems.append(reason)
    if problems:
        banner = "PROVENANCE GUARD: manuscript tables blocked — fixture / under-powered data:"
        print(banner)
        for r in problems:
            print(f"  - {r}")
        if not args.allow_pilot:
            raise FixtureContaminationError(
                banner + " " + " | ".join(problems)
                + "  (re-run at full scale, or pass --allow-pilot for a clearly-labelled pilot report.)"
            )
        print("  --allow-pilot set: continuing with a PILOT-LABELLED report (NOT for the manuscript).")

    is_pilot = bool(problems and args.allow_pilot)
    # A pilot report must never be byte-indistinguishable from — or overwrite — a manuscript
    # table. Route it to a separate `pilot/` subtree and mark captions/labels, so a person or
    # an index script reading results/latex/top{N}/*.tex cannot mistake it for the real thing.
    sub = "pilot" if is_pilot else ""
    tables_dir = Path(args.output_dir) / "tables" / f"top{args.top_n}" / sub
    latex_dir = Path(args.output_dir) / "latex" / f"top{args.top_n}" / sub

    def cap(title: str) -> str:
        return f"[PILOT — under-powered, NOT FOR MANUSCRIPT] {title}" if is_pilot else title

    def lab(label: str) -> str:
        return f"{label}_pilot" if is_pilot else label

    def stem(s: str) -> str:
        return f"{s}_pilot" if is_pilot else s

    emit(build_table(seed_stats, EXPERIMENT_ORDER[:14], MAIN_COLS),
         args.output_dir, stem("table_main_comparison"),
         cap(f"Main comparison (Top-{args.top_n}, mean ± std over seeds)"), lab("tab:main"), tables_dir, latex_dir)
    emit(build_table(seed_stats, ["E14", "E15", "E16", "E17"], OPT_COLS),
         args.output_dir, stem("table_optimizer_comparison"),
         cap(f"Optimizer comparison (Top-{args.top_n})"), lab("tab:opt"), tables_dir, latex_dir)
    emit(build_table(seed_stats, [f"E{i}" for i in range(9, 18)], RELIABILITY_COLS),
         args.output_dir, stem("table_reliability"),
         cap(f"Reliability & contrastive honesty (Top-{args.top_n})"), lab("tab:reliab"), tables_dir, latex_dir)

    print(f"Wrote {'PILOT ' if is_pilot else ''}paper tables to {tables_dir} (CSV+MD) and {latex_dir} (LaTeX)")
    print("Tables: table_main_comparison, table_optimizer_comparison, table_reliability")
    n_exp = sum(1 for e in EXPERIMENT_ORDER if e in seed_stats)
    print(f"Experiments with metrics: {n_exp} | fixed_label_space={data.get('fixed_label_space')} "
          f"| label_set_size={data.get('label_set_size')}")


if __name__ == "__main__":
    main()
