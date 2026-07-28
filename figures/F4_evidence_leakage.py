#!/usr/bin/env python3
"""
Figure 4 — Cross-patient evidence leakage in the global-corpus design (T3).
Left: schematic (a code query retrieves the globally best-matching document, almost never the
note's own). Right: the measured rate (100% cross-admission over 10,338 retrieved documents
at the pipeline's own k = 2 setting; identical at k = 5 over 25,845).
Schematic + one measured bar. PDF+PNG, colorblind-safe.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "pdf.fonttype": 42,
})
BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#8A8A8A", "#222222"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.7, 1]})

# ---------- Left: schematic ----------
axL.set_xlim(0, 100); axL.set_ylim(0, 60); axL.axis("off")
# the note being coded
axL.add_patch(FancyBboxPatch((2, 24), 20, 14, boxstyle="round,pad=0.6,rounding_size=1.6",
                             lw=1.1, edgecolor=BLUE, facecolor="#E7F1FA"))
axL.text(12, 31, "Note being\ncoded\n(admission A)", ha="center", va="center", fontsize=7.6, color=INK)
axL.text(2, 46, "query: “ICD-10 E11.9 …”", ha="left", va="center",
         fontsize=6.6, color=GREY, style="italic")
# global index (many admissions)
axL.add_patch(FancyBboxPatch((54, 6), 44, 48, boxstyle="round,pad=0.8,rounding_size=2",
                             lw=1.0, edgecolor=GREY, facecolor="#F5F5F5"))
axL.text(76, 50, "Global evidence index\n(115,103 documents,\none per note, all admissions)", ha="center",
         va="center", fontsize=7.0, color=INK)
# documents: many "other patient" (orange), one "own" (blue)
import itertools
coords = list(itertools.product([60, 68, 76, 84, 92], [12, 20, 28, 36]))
own = (76, 28)
for (cx, cy) in coords:
    is_own = (cx, cy) == own
    axL.add_patch(Rectangle((cx - 2.4, cy - 2.4), 4.8, 4.8, facecolor=(BLUE if is_own else "#FBEBE1"),
                            edgecolor=(BLUE if is_own else ORANGE), lw=1.1 if is_own else 0.7))
axL.text(76, 2.5, "orange = other admission   ■ blue = the note’s own", ha="center", va="center",
         fontsize=6.3, color=GREY)
# retrieval arrow -> lands on an orange (top-ranked) document, NOT the blue own document
axL.add_patch(FancyArrowPatch((22, 31), (57.6, 36), arrowstyle="-|>", mutation_scale=12,
                              lw=1.4, color=ORANGE, zorder=5))
axL.text(39, 37.5, "top-ranked\ndocument", ha="center", va="bottom", fontsize=6.6, color=ORANGE)
axL.set_title("(a) Global retrieval returns another admission’s note", fontsize=8.2)

# ---------- Right: measured rate ----------
axR.barh([1], [100], color=ORANGE, height=0.5, zorder=3)
axR.barh([0], [0.0], color=BLUE, height=0.5, zorder=3)
axR.text(98, 1, "100%", va="center", ha="right", color="white", fontsize=9)
axR.text(2, 0, "0%", va="center", ha="left", color=INK, fontsize=8)
axR.set_yticks([1, 0]); axR.set_yticklabels(["Cross-\nadmission", "Note’s own"])
axR.set_xlim(0, 100); axR.set_xlabel("% of retrieved evidence documents")
axR.set_title("(b) Measured (10,338 documents, k = 2)", fontsize=8.2)
axR.spines[["top", "right", "left"]].set_visible(False)
axR.tick_params(axis="y", length=0)
axR.set_xticks([0, 50, 100])

fig.subplots_adjust(left=0.02, right=0.97, top=0.86, bottom=0.16, wspace=0.35)
for ext in ("pdf", "png"):
    fig.savefig(f"F4_evidence_leakage.{ext}", dpi=300)
print("wrote F4_evidence_leakage.{pdf,png}")
