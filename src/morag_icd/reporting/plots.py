"""
Reporting plots module: generates all 10 publication-quality figures.

All plots saved as both PDF (for paper) and PNG (for preview).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Plots will not be generated.")


EXPERIMENT_NAMES = {
    "E1": "TF-IDF+LR", "E2": "TF-IDF+SVM", "E3": "BioClinBERT",
    "E4": "BM25-only", "E5": "Dense-only", "E6": "Hybrid-only",
    "E7": "LLM-0shot", "E8": "LLM-few", "E9": "BM25-RAG",
    "E10": "Dense-RAG", "E11": "Hybrid-RAG", "E12": "+EvidConstr",
    "E13": "+ContrastV", "E14": "FullModel", "E15": "+RandSearch",
    "E16": "+MOPSO", "E17": "+NSGA-II",
}
COLORS = plt.cm.tab20.colors if HAS_MPL else []


def _save_fig(fig, path: Path):
    """Save figure as both PDF and PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _get(stats: Dict, key: str) -> float:
    return stats.get(key, {}).get("mean", float("nan"))


def plot_model_comparison(seed_stats: Dict, output_path: str | Path):
    """Fig 1: Bar chart of all models by Micro-F1."""
    if not HAS_MPL:
        return
    output_path = Path(output_path)
    exp_ids = [e for e in EXPERIMENT_NAMES if e in seed_stats]
    values = [_get(seed_stats[e], "micro_f1") for e in exp_ids]
    stds = [seed_stats[e].get("micro_f1", {}).get("std", 0) for e in exp_ids]
    labels = [EXPERIMENT_NAMES.get(e, e) for e in exp_ids]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(exp_ids))
    bars = ax.bar(x, values, yerr=stds, capsize=4,
                  color=[COLORS[i % len(COLORS)] for i in range(len(exp_ids))],
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Micro-F1", fontsize=11)
    ax.set_title("Model Comparison — Micro-F1 (mean ± std over 3 seeds)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_rag_ablation(seed_stats: Dict, output_path: str | Path):
    """Fig 2: Grouped bar chart for RAG ablation E9-E14."""
    if not HAS_MPL:
        return
    output_path = Path(output_path)
    ablation_ids = ["E9", "E10", "E11", "E12", "E13", "E14"]
    metrics = ["micro_f1", "evidence_support_rate", "unsupported_code_rate"]
    metric_labels = ["Micro-F1", "Evidence Support", "Unsupported Rate"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, label in zip(axes, metrics, metric_labels):
        vals = [_get(seed_stats.get(e, {}), metric) for e in ablation_ids]
        stds = [seed_stats.get(e, {}).get(metric, {}).get("std", 0) for e in ablation_ids]
        x = np.arange(len(ablation_ids))
        ax.bar(x, vals, yerr=stds, capsize=4, color=COLORS[:len(ablation_ids)],
               alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([EXPERIMENT_NAMES.get(e, e) for e in ablation_ids],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("RAG Ablation Study (E9–E14)", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_evidence_vs_f1(seed_stats: Dict, output_path: str | Path):
    """Fig 3: Scatter — Evidence Support Rate vs Micro-F1."""
    if not HAS_MPL:
        return
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (exp_id, stats) in enumerate(seed_stats.items()):
        x = _get(stats, "evidence_support_rate")
        y = _get(stats, "micro_f1")
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, color=COLORS[i % len(COLORS)], s=80, zorder=3)
        ax.annotate(EXPERIMENT_NAMES.get(exp_id, exp_id), (x, y),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Evidence Support Rate", fontsize=11)
    ax.set_ylabel("Micro-F1", fontsize=11)
    ax.set_title("Evidence Support Rate vs. Micro-F1", fontsize=12)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_unsupported_vs_f1(seed_stats: Dict, output_path: str | Path):
    """Fig 4: Scatter — Unsupported Code Rate vs Micro-F1."""
    if not HAS_MPL:
        return
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (exp_id, stats) in enumerate(seed_stats.items()):
        x = _get(stats, "unsupported_code_rate")
        y = _get(stats, "micro_f1")
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, color=COLORS[i % len(COLORS)], s=80, zorder=3)
        ax.annotate(EXPERIMENT_NAMES.get(exp_id, exp_id), (x, y),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Unsupported Code Rate", fontsize=11)
    ax.set_ylabel("Micro-F1", fontsize=11)
    ax.set_title("Unsupported Code Rate vs. Micro-F1", fontsize=12)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_pareto_front_from_file(pareto_dir: str | Path, output_path: str | Path):
    """Fig 5: Pareto front plot from optimizer result files."""
    if not HAS_MPL:
        return
    output_path = Path(output_path)
    pareto_dir = Path(pareto_dir)

    fig, ax = plt.subplots(figsize=(9, 6))
    optimizer_colors = {"mopso": "steelblue", "nsga2": "darkorange", "random_search": "gray"}

    for result_file in pareto_dir.rglob("*_results.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)
        except Exception:
            continue

        optimizer = result.get("optimizer", result_file.parent.name)
        pareto = result.get("pareto_front", [])
        if not pareto:
            continue

        xs, ys, sizes = [], [], []
        for sol in pareto:
            obj = sol.get("obj_vector", [])
            if len(obj) >= 4:
                # obj[0] = -micro_f1, obj[3] = unsupported_rate
                micro_f1 = -obj[0]
                unsupported = obj[3] if len(obj) > 3 else 0
                tokens = obj[5] if len(obj) > 5 else 1
                xs.append(unsupported)
                ys.append(micro_f1)
                sizes.append(max(50, min(500, tokens / 10)))

        color = optimizer_colors.get(optimizer, "purple")
        ax.scatter(xs, ys, s=sizes, c=color, alpha=0.7, label=optimizer.upper(), edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Unsupported Code Rate", fontsize=11)
    ax.set_ylabel("Micro-F1", fontsize=11)
    ax.set_title("Pareto Front\n(bubble size ∝ avg. token count)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_scalability(seed_stats: Dict, output_path: str | Path):
    """Fig 7: Scalability across Top-50/100/200."""
    if not HAS_MPL:
        return
    output_path = Path(output_path)
    models = ["E2", "E11", "E14", "E16"]
    model_labels = ["TF-IDF+SVM", "Hybrid-RAG", "FullModel", "MOPSO"]
    top_ns = [50, 100, 200]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model, label, color in zip(models, model_labels, COLORS[:4]):
        ys = []
        for top_n in top_ns:
            key = f"top{top_n}_{model}"
            val = _get(seed_stats.get(key, seed_stats.get(model, {})), "micro_f1")
            ys.append(val)
        ax.plot(top_ns, ys, marker="o", label=label, color=color, linewidth=2)

    ax.set_xlabel("Number of ICD-10 Codes", fontsize=11)
    ax.set_ylabel("Micro-F1", fontsize=11)
    ax.set_title("Scalability Analysis: Top-50 / Top-100 / Top-200", fontsize=12)
    ax.set_xticks(top_ns)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_runtime_comparison(seed_stats: Dict, output_path: str | Path):
    """Fig 9: Average inference time per experiment."""
    if not HAS_MPL:
        return
    output_path = Path(output_path)
    exp_ids = [e for e in EXPERIMENT_NAMES if e in seed_stats]
    runtimes = [_get(seed_stats[e], "avg_runtime_sec") for e in exp_ids]
    labels = [EXPERIMENT_NAMES.get(e, e) for e in exp_ids]

    # Filter out NaN
    valid = [(l, r) for l, r in zip(labels, runtimes) if not np.isnan(r)]
    if not valid:
        return
    labels_v, runtimes_v = zip(*valid)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(labels_v))
    ax.bar(x, runtimes_v, color=COLORS[:len(labels_v)], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_v, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Avg. Runtime (s/sample)", fontsize=11)
    ax.set_title("Inference Runtime Comparison", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_pareto_front(x, y, labels, output_path: str | Path):
    """Legacy-compatible single Pareto front plot."""
    if not HAS_MPL:
        return
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=80, color="steelblue", zorder=3, edgecolor="black", linewidth=0.5)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Unsupported Code Rate", fontsize=11)
    ax.set_ylabel("Micro-F1", fontsize=11)
    ax.set_title("Pareto Front", fontsize=12)
    ax.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_fig(fig, output_path)
