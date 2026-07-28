#!/usr/bin/env python3
"""
Figure S1 — Scalability: micro-F1 vs label-space size (Top-50/100/200) for the four ladder
representatives. The classical baseline holds while every LLM/retrieval arm degrades, so the
gap widens with scale. Numbers from T5 (results_eurohpc/scalability merge+eval). PDF+PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

X = [50, 100, 200]
# Corrected re-run: the first campaign silently used the Top-50 index/label set at every
# setting (see Results 4.4); these are the numbers after the fix.
SERIES = [
    ("E1 · TF-IDF",        [0.449, 0.466, 0.469], "#0072B2", "o"),
    ("E6 · Hybrid retr.",  [0.203, 0.163, 0.119], "#56B4E9", "s"),
    ("E11 · Hybrid RAG",   [0.186, 0.138, 0.107], "#009E73", "^"),
    ("E14 · Full model",   [0.133, 0.097, 0.070], "#D55E00", "D"),
]

fig, ax = plt.subplots(figsize=(3.6, 3.2))
xs = [0, 1, 2]
for lab, ys, c, mk in SERIES:
    ax.plot(xs, ys, color=c, lw=1.6, marker=mk, ms=5, zorder=3)
    ax.text(xs[-1] + 0.06, ys[-1], lab, va="center", ha="left", fontsize=7.0, color=c)

# widening-gap annotation between E1 and E14 at the ends
for xi, note in [(0, "+0.317"), (2, "+0.399")]:
    y1, y2 = SERIES[0][1][xi], SERIES[3][1][xi]
    ax.annotate("", xy=(xi, y1), xytext=(xi, y2),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=0.8))
    ax.text(xi + (0.04 if xi == 0 else -0.04), (y1 + y2) / 2, note, fontsize=6.6,
            color="#666666", ha="left" if xi == 0 else "right", va="center")

ax.set_xticks(xs); ax.set_xticklabels(["Top-50", "Top-100", "Top-200"])
ax.set_xlim(-0.15, 2.9); ax.set_ylim(0, 0.52)
ax.set_ylabel("Micro-F1")
ax.set_xlabel("Label-space size")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E4E4E4", lw=0.5, zorder=0)
ax.set_axisbelow(True)

fig.subplots_adjust(left=0.15, right=0.99, top=0.97, bottom=0.12)
for ext in ("pdf", "png"):
    fig.savefig(f"S1_scalability.{ext}", dpi=300)
print("wrote S1_scalability.{pdf,png}")
