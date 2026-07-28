#!/usr/bin/env python3
"""
Figure 1 — The RAG-LLM ICD-10 pipeline under study, with the evaluation-integrity safeguards
(Section 3.7) marked at the control points where they apply. Schematic (no data). PDF+PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "pdf.fonttype": 42,
})

BLUE, GREEN, ORANGE, GREY, INK = "#0072B2", "#009E73", "#D55E00", "#8A8A8A", "#222222"

fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

def box(x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2,
            color=INK, zorder=4)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=GREY, zorder=2))

# main pipeline row (left -> right), y-centered
Y, H = 30, 12
xs = [1, 21, 41, 61, 81]
W = 17
labels = [
    "Discharge\nnote",
    "Candidate\nretrieval\n(BM25 / dense /\nhybrid)",
    "Note-local\nevidence\nretrieval",
    "Batched LLM\nscoring\n(1 call / note)",
    "Constraint +\ncontrastive +\nthreshold",
]
fcs = ["#EEEEEE", "#E7F1FA", "#E7F1FA", "#E9F6F1", "#FBEBE1"]
ecs = [GREY, BLUE, BLUE, GREEN, ORANGE]
for x, lab, fc, ec in zip(xs, labels, fcs, ecs):
    box(x, Y, W, H, lab, fc, ec)
for i in range(len(xs) - 1):
    arrow(xs[i] + W, Y + H / 2, xs[i + 1], Y + H / 2)
# output (arrow down from last box to a wrapped label that stays in-bounds)
ax.add_patch(FancyArrowPatch((81 + W / 2, Y), (81 + W / 2, Y - 3.5), arrowstyle="-|>",
                             mutation_scale=10, lw=1.1, color=GREY, zorder=2))
ax.text(81 + W / 2, Y - 4.2, "final codes +\nevidence quote +\nrationale", ha="center",
        va="top", fontsize=7.2, color=INK, style="italic")

# safeguards (below), each pointing up to its control point
def safeguard(cx, text):
    ax.add_patch(FancyBboxPatch((cx - 9.5, 6), 19, 8.5, boxstyle="round,pad=0.5,rounding_size=1.4",
                                linewidth=0.9, edgecolor="#B03A00", facecolor="#FFF6F0", zorder=3))
    ax.text(cx, 10.2, text, ha="center", va="center", fontsize=6.7, color="#7A2800", zorder=4)
    ax.add_patch(FancyArrowPatch((cx, 14.5), (cx, Y - 0.3), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.9, color="#B03A00", ls=(0, (2, 1.6)), zorder=2))

safeguard(41 + W / 2, "Safeguard:\nnote-local evidence\n(no cross-patient leak)")
safeguard(61 + W / 2, "Safeguard:\nschema-robust rank\n(support→conf→retr.)")
ax.text(50, 1.4, "Evaluation-integrity safeguards (§3.7); all result tables provenance-guarded",
        ha="center", va="bottom", fontsize=6.6, color="#7A2800")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
for ext in ("pdf", "png"):
    fig.savefig(f"F1_pipeline.{ext}", dpi=300)
print("wrote F1_pipeline.{pdf,png}")
