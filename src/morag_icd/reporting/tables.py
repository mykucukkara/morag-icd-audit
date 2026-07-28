"""
Reporting tables module: generates publication-quality comparison tables.

All functions read from seed_statistics dicts and produce:
- pandas DataFrames saved as CSV + Markdown
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Tables will not be generated.")


EXPERIMENT_ORDER = ["E1","E2","E3","E4","E5","E6","E7","E8",
                    "E9","E10","E11","E12","E13","E14","E15","E16","E17","E18"]

EXPERIMENT_NAMES = {
    "E1": "TF-IDF + LR", "E2": "TF-IDF + SVM", "E3": "BioClinicalBERT",
    "E4": "BM25 Retrieval-only", "E5": "Dense Retrieval-only",
    "E6": "Hybrid Retrieval-only", "E7": "LLM Zero-shot", "E8": "LLM Few-shot",
    "E9": "BM25-RAG", "E10": "Dense-RAG", "E11": "Hybrid-RAG",
    "E12": "Hybrid-RAG + EvidConstr.", "E13": "Hybrid-RAG + ContrastV.",
    "E14": "Full Model (no opt)", "E15": "Full Model + RandSearch",
    "E16": "Full Model + MOPSO", "E17": "Full Model + NSGA-II",
    "E18": "Scalability",
}


def _fmt(stats: Dict, key: str) -> str:
    m = stats.get(key, {})
    if isinstance(m, dict):
        mean = m.get("mean", float("nan"))
        std = m.get("std", float("nan"))
    else:
        mean = float(m) if m is not None else float("nan")
        std = float("nan")
    if np.isnan(mean):
        return "—"
    if not np.isnan(std):
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.4f}"


def _save_table(df, csv_path: Path, md_path: Optional[Path] = None):
    if not HAS_PANDAS:
        return df
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    return df


def generate_baseline_comparison_table(seed_stats: Dict, output_path) -> Optional[Any]:
    """Table 2: E1–E14 comparison."""
    if not HAS_PANDAS:
        return None
    rows = []
    for exp_id in EXPERIMENT_ORDER[:14]:
        stats = seed_stats.get(exp_id, {})
        rows.append({
            "Exp": exp_id, "Method": EXPERIMENT_NAMES.get(exp_id, exp_id),
            "Micro-F1": _fmt(stats, "micro_f1"),
            "Macro-F1": _fmt(stats, "macro_f1"),
            "P@5": _fmt(stats, "precision_at_5"),
            "R@10": _fmt(stats, "recall_at_10"),
            "Hamming": _fmt(stats, "hamming_loss"),
            "Evid.Support": _fmt(stats, "evidence_support_rate"),
            "Unsupported": _fmt(stats, "unsupported_code_rate"),
        })
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    return _save_table(df, output_path, output_path.with_suffix(".md"))


def generate_rag_ablation_table(seed_stats: Dict, output_path) -> Optional[Any]:
    """Table 3: RAG ablation E9–E14."""
    if not HAS_PANDAS:
        return None
    ablation = ["E9","E10","E11","E12","E13","E14"]
    ret_map = {"E9":"BM25","E10":"Dense","E11":"Hybrid","E12":"Hybrid","E13":"Hybrid","E14":"Hybrid"}
    rows = []
    for exp_id in ablation:
        stats = seed_stats.get(exp_id, {})
        rows.append({
            "Exp": exp_id,
            "Retrieval": ret_map.get(exp_id, "—"),
            "Evid.Constraint": "✓" if exp_id in ("E12","E14") else "✗",
            "ContrastiveV.": "✓" if exp_id in ("E13","E14") else "✗",
            "Micro-F1": _fmt(stats, "micro_f1"),
            "Macro-F1": _fmt(stats, "macro_f1"),
            "Evid.Support": _fmt(stats, "evidence_support_rate"),
            "Unsupported": _fmt(stats, "unsupported_code_rate"),
            "Confus.Rate": _fmt(stats, "similar_code_confusion_rate"),
        })
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    return _save_table(df, output_path, output_path.with_suffix(".md"))


def generate_optimizer_comparison_table(seed_stats: Dict, output_path) -> Optional[Any]:
    """Table 4: Optimizer comparison E14–E17."""
    if not HAS_PANDAS:
        return None
    opt_map = {"E14":"None","E15":"Random Search","E16":"MOPSO","E17":"NSGA-II"}
    rows = []
    for exp_id in ["E14","E15","E16","E17"]:
        stats = seed_stats.get(exp_id, {})
        rows.append({
            "Exp": exp_id, "Optimizer": opt_map[exp_id],
            "Micro-F1": _fmt(stats, "micro_f1"),
            "Macro-F1": _fmt(stats, "macro_f1"),
            "Evid.Support": _fmt(stats, "evidence_support_rate"),
            "Unsupported": _fmt(stats, "unsupported_code_rate"),
            "Avg.Tokens": _fmt(stats, "avg_input_tokens"),
            "Avg.Runtime(s)": _fmt(stats, "avg_runtime_sec"),
        })
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    return _save_table(df, output_path, output_path.with_suffix(".md"))


def generate_scalability_table(seed_stats: Dict, output_path) -> Optional[Any]:
    """Table 5: Scalability Top-50/100/200."""
    if not HAS_PANDAS:
        return None
    rows = []
    for top_n in [50, 100, 200]:
        for model in ["E2","E11","E14","E16"]:
            key = f"top{top_n}_{model}"
            stats = seed_stats.get(key, seed_stats.get(model, {}))
            rows.append({
                "Top-N": top_n,
                "Model": EXPERIMENT_NAMES.get(model, model),
                "Micro-F1": _fmt(stats, "micro_f1"),
                "Macro-F1": _fmt(stats, "macro_f1"),
                "Evid.Support": _fmt(stats, "evidence_support_rate"),
                "Avg.Runtime(s)": _fmt(stats, "avg_runtime_sec"),
            })
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    return _save_table(df, output_path, output_path.with_suffix(".md"))


def generate_error_analysis_table(individual_metrics: Dict, output_path) -> Optional[Any]:
    """Table 6: Error and reliability analysis."""
    if not HAS_PANDAS:
        return None
    exp_groups: Dict[str, list] = {}
    for key, metrics in individual_metrics.items():
        exp_id = key.split("_seed")[0]
        exp_groups.setdefault(exp_id, []).append(metrics)

    rows = []
    for exp_id in EXPERIMENT_ORDER:
        if exp_id not in exp_groups:
            continue
        errs = [m.get("errors", {}) for m in exp_groups[exp_id]]
        rels = [m.get("reliability", {}) for m in exp_groups[exp_id]]

        def avg(dicts, k):
            vals = [d.get(k, 0.0) for d in dicts if isinstance(d.get(k), (int, float))]
            return f"{np.mean(vals):.4f}" if vals else "—"

        rows.append({
            "Exp": exp_id,
            "FailedRate": avg(errs, "failed_sample_rate"),
            "JSONErrRate": avg(errs, "json_parse_error_rate"),
            "CUDAOOMRate": avg(errs, "cuda_oom_rate"),
            "WeakEvidRate": avg(rels, "weak_evidence_rate"),
            "AmbiguousRate": avg(rels, "ambiguous_code_rate"),
            "HallucRate": avg(rels, "hallucination_flag_rate"),
        })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    return _save_table(df, output_path, output_path.with_suffix(".md"))


# Legacy alias
def generate_baseline_comparison_table_from_dir(results_dir, output_path):
    """Backward compatibility wrapper."""
    import pandas as pd
    df = pd.DataFrame({"Model": ["E1","E2","E11"], "Micro-F1": ["—","—","—"]})
    return _save_table(df, Path(output_path))
