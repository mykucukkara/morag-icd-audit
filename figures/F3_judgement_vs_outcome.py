#!/usr/bin/env python3
"""
Figure 3 — Capacity ablation (3B vs 7B): evidence judgement improves with scale, end-task F1
does not. Numbers from scripts/37 (T2). Colorblind-safe, print-ready PDF+PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#999999"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 2.9))

# --- Panel A: discriminative lift 3B -> 7B (significant increase) ---
lift = {"3B": 1.02, "7B": 1.49}
axL.plot([0, 1], [lift["3B"], lift["7B"]], color=BLUE, lw=1.4, zorder=2)
axL.scatter([0, 1], [lift["3B"], lift["7B"]], s=48, color=BLUE, zorder=3)
axL.axhline(1.0, color=GREY, lw=0.8, ls=(0, (3, 3)), zorder=1)
axL.text(0.5, 1.0, "no discrimination", fontsize=6.6, color=GREY, ha="center", va="bottom")
for x, k in [(0, "3B"), (1, "7B")]:
    axL.text(x, lift[k] + 0.03, f"{lift[k]:.2f}", ha="center", va="bottom", fontsize=8, color=BLUE)
axL.annotate("", xy=(1, 1.49), xytext=(1, 1.02),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
axL.text(1.06, 1.255, "+0.47\n95% CI\n[+0.10, +0.84]\nsignificant", fontsize=6.6,
         color=ORANGE, ha="left", va="center")
axL.set_xlim(-0.35, 1.75); axL.set_ylim(0.9, 1.65)
axL.set_xticks([0, 1]); axL.set_xticklabels(["Qwen2.5-3B", "Qwen2.5-7B"])
axL.set_ylabel("Evidence-judgement\ndiscriminative lift")
axL.set_title("(a) Judgement improves with scale", fontsize=8.5)
axL.spines[["top", "right"]].set_visible(False)

# --- Panel B: ΔF1 (7B - 3B), CIs span zero (not significant) ---
arms = ["E11 (hybrid RAG)", "E14 (full model)"]
d   = [ 0.008, -0.015]
lo  = [-0.005, -0.048]
hi  = [ 0.021,  0.018]
ys  = [1, 0]
axR.axvline(0.0, color=GREY, lw=0.9, zorder=1)
for y, dd, l, h in zip(ys, d, lo, hi):
    axR.plot([l, h], [y, y], color=BLUE, lw=1.6, zorder=2)
    axR.scatter([dd], [y], s=46, color=BLUE, zorder=3)
    axR.text(dd, y + 0.12, f"{dd:+.3f}", ha="center", va="bottom", fontsize=7.5, color="#222222")
    axR.text(h + 0.004, y, "n.s.", va="center", ha="left", fontsize=7, color=GREY)
axR.set_yticks(ys); axR.set_yticklabels(arms)
axR.set_ylim(-0.6, 1.6); axR.set_xlim(-0.075, 0.075)
axR.set_xlabel(r"$\Delta$ Micro-F1  (7B $-$ 3B), 95% CI")
axR.set_title("(b) End-task F1 does not, at this operating point", fontsize=8.5)
axR.spines[["top", "right", "left"]].set_visible(False)
axR.tick_params(axis="y", length=0)

fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.16, wspace=0.55)
for ext in ("pdf", "png"):
    fig.savefig(f"F3_judgement_vs_outcome.{ext}", dpi=300)
print("wrote F3_judgement_vs_outcome.{pdf,png}")
