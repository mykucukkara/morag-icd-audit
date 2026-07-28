#!/usr/bin/env python3
"""
Figure 2 — Component-decomposition ladder (Top-50, n=17,151).
Colorblind-safe (Okabe-Ito), print-ready PDF+PNG. Numbers from scripts/34 output (T1).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

# (id, label, micro_f1, group)  — full-test Top-50 (T1)
DATA = [
    ("E1", "TF-IDF + LR",        0.449, "Classical"),
    ("E2", "TF-IDF + SVM",       0.441, "Classical"),
    ("E3", "Transformer clf.",   0.372, "Classical"),
    ("E6", "Hybrid retrieval",   0.203, "Retrieval-only"),
    ("E4", "BM25 retrieval",     0.196, "Retrieval-only"),
    ("E13","+ Contrastive",      0.195, "Full model"),
    ("E11","Hybrid RAG",         0.186, "RAG"),
    ("E5", "Dense retrieval",    0.185, "Retrieval-only"),
    ("E10","Dense RAG",          0.182, "RAG"),
    ("E9", "BM25 RAG",           0.181, "RAG"),
    ("E12","+ Evidence constr.", 0.134, "Full model"),
    ("E14","Full model",         0.133, "Full model"),
    ("E8", "LLM few-shot",       0.014, "LLM-only"),
    ("E7", "LLM zero-shot",      0.004, "LLM-only"),
]
# Okabe-Ito (published CVD-safe); each group a distinct hue, order fixed.
COLOR = {
    "Classical":      "#0072B2",  # blue
    "Retrieval-only": "#56B4E9",  # sky blue
    "RAG":            "#009E73",  # bluish green
    "Full model":     "#D55E00",  # vermillion
    "LLM-only":       "#999999",  # neutral grey
}
GROUP_ORDER = ["Classical", "Retrieval-only", "RAG", "Full model", "LLM-only"]

fig, ax = plt.subplots(figsize=(3.4, 4.2))  # single-column
ys = list(range(len(DATA)))[::-1]  # top = best
for y, (eid, lab, f1, grp) in zip(ys, DATA):
    ax.barh(y, f1, height=0.66, color=COLOR[grp], edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(f1 + 0.006, y, f"{f1:.3f}", va="center", ha="left", fontsize=7.5, color="#222222")
    ax.text(-0.010, y, f"{eid}  {lab}", va="center", ha="right", fontsize=7.5, color="#222222")

# E1 reference line
ax.axvline(0.449, color="#0072B2", lw=0.8, ls=(0, (4, 3)), zorder=2, alpha=0.7)
ax.text(0.449, len(DATA) - 0.2, "E1 baseline", fontsize=6.8, color="#0072B2",
        ha="center", va="bottom")
# Note-blind floor (E0): a constant predictor that never reads the note. Every retrieval, RAG
# and full-model arm sits to its left — the headline of §4.1b, so it belongs on this figure.
ax.axvline(0.304, color="#CC79A7", lw=1.0, ls=(0, (2, 2)), zorder=2)
# label runs along the line (rotated) so it cannot collide with the E1 label at the top
ax.text(0.298, 4.6, "note-blind floor", fontsize=6.8, color="#CC79A7",
        ha="right", va="center", rotation=90)

ax.set_yticks([])
ax.set_xlim(0, 0.52)
ax.set_xlabel("Micro-F1 (Top-50, n = 17,151)")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(axis="x", color="#DDDDDD", lw=0.5, zorder=0)
ax.set_axisbelow(True)

handles = [Patch(facecolor=COLOR[g], label=g) for g in GROUP_ORDER]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.8,
          handlelength=1.0, handleheight=1.0, borderpad=0.3, labelspacing=0.3)

fig.subplots_adjust(left=0.42, right=0.98, top=0.97, bottom=0.10)
for ext in ("pdf", "png"):
    fig.savefig(f"F2_decomposition_ladder.{ext}", dpi=300)
print("wrote F2_decomposition_ladder.{pdf,png}")
