#!/usr/bin/env python3
"""
Script 48: the context x capacity interaction, as a single difference-in-differences.

The manuscript claimed that scorer context and model capacity are "jointly necessary and
individually insufficient". A reviewer pointed out that the claim was assembled from cells measured
on different samples: three cells came from a 1,008-note subset and the fourth (7B without the note)
from a separate 200-note capacity run. Differences taken across different samples do not constitute
an interaction.

This computes the contrast properly, on one note set, with all four cells:

    [F1(7B, note) - F1(3B, note)] - [F1(7B, no note) - F1(3B, no note)]

A note-level paired bootstrap resamples notes once per replicate and recomputes all four cells from
the same resample, so the interval reflects the dependence between cells. The randomization test
permutes, per note, which capacity condition each observation belongs to within each context
condition — the exchangeability the null of "no interaction" implies.

Usage:
    python scripts/48_interaction_contrast.py \
        --note-3b results_eurohpc/steelman/top50 --note-7b results_eurohpc/steelman7b/top50 \
        --nonote-3b results_eurohpc/primary_campaign/top50 \
        --nonote-7b results_eurohpc/cap7b_nonote/top50 \
        --arms E11 E14 --out results_eurohpc/primary_campaign/interaction_contrast.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PB = _load("37_paired_bootstrap.py")
S43 = _load("43_steelman7b_contrasts.py")


def f1_from(counts: dict, sample: list[int]) -> float:
    tp = fp = fn = 0
    for i in sample:
        a, b, c = counts[i][:3]
        tp += a; fp += b; fn += c
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note-3b", required=True)
    ap.add_argument("--note-7b", required=True)
    ap.add_argument("--nonote-3b", required=True)
    ap.add_argument("--nonote-7b", required=True)
    ap.add_argument("--arms", nargs="+", default=["E11", "E14"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_eurohpc/primary_campaign/interaction_contrast.json")
    args = ap.parse_args()

    report: dict = {"n_boot": args.n_boot, "n_perm": args.n_perm, "seed": args.seed, "arms": {}}

    for arm in args.arms:
        cells = {}
        for label, root in (("3B_note", args.note_3b), ("7B_note", args.note_7b),
                            ("3B_nonote", args.nonote_3b), ("7B_nonote", args.nonote_7b)):
            p = S43.find_predictions(Path(root), arm)
            if p is None:
                print(f"  {arm}: missing {label} under {root} — skipped")
                cells = {}
                break
            cells[label] = PB.load(p)
        if not cells:
            continue

        idx = sorted(set.intersection(*(set(c) for c in cells.values())))
        if len(idx) < 100:
            print(f"  {arm}: only {len(idx)} notes shared by all four cells — skipped")
            continue
        counts = {k: {i: PB.note_counts(v[i]) for i in idx} for k, v in cells.items()}
        point = {k: f1_from(c, idx) for k, c in counts.items()}

        simple_note = point["7B_note"] - point["3B_note"]
        simple_nonote = point["7B_nonote"] - point["3B_nonote"]
        interaction = simple_note - simple_nonote

        rng = random.Random(args.seed)
        boot_int, boot_note, boot_nonote = [], [], []
        for _ in range(args.n_boot):
            s = [rng.choice(idx) for _ in idx]
            a = f1_from(counts["7B_note"], s) - f1_from(counts["3B_note"], s)
            b = f1_from(counts["7B_nonote"], s) - f1_from(counts["3B_nonote"], s)
            boot_note.append(a); boot_nonote.append(b); boot_int.append(a - b)

        # Randomization: under "no interaction", swapping the 3B and 7B labels within a note is
        # exchangeable in the same direction in both context conditions, so swapping in one
        # condition only is what the null forbids.
        rngp = random.Random(args.seed)
        extreme = 0
        for _ in range(args.n_perm):
            sw_note = {i: rngp.random() < 0.5 for i in idx}
            sw_non = {i: rngp.random() < 0.5 for i in idx}
            a7 = {i: counts["3B_note"][i] if sw_note[i] else counts["7B_note"][i] for i in idx}
            a3 = {i: counts["7B_note"][i] if sw_note[i] else counts["3B_note"][i] for i in idx}
            b7 = {i: counts["3B_nonote"][i] if sw_non[i] else counts["7B_nonote"][i] for i in idx}
            b3 = {i: counts["7B_nonote"][i] if sw_non[i] else counts["3B_nonote"][i] for i in idx}
            perm = (f1_from(a7, idx) - f1_from(a3, idx)) - (f1_from(b7, idx) - f1_from(b3, idx))
            if abs(perm) >= abs(interaction):
                extreme += 1
        p_ar = (extreme + 1) / (args.n_perm + 1)

        lo, hi = PB.ci(boot_int)
        nlo, nhi = PB.ci(boot_note)
        xlo, xhi = PB.ci(boot_nonote)
        report["arms"][arm] = {
            "shared_notes": len(idx),
            "cells_micro_f1": {k: round(v, 4) for k, v in point.items()},
            "simple_effect_of_capacity_with_note": {
                "point": round(simple_note, 4), "ci95": [round(nlo, 4), round(nhi, 4)]},
            "simple_effect_of_capacity_without_note": {
                "point": round(simple_nonote, 4), "ci95": [round(xlo, 4), round(xhi, 4)]},
            "interaction": {
                "point": round(interaction, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "verdict": "SIGNIFICANT (excludes 0)" if (lo > 0 or hi < 0) else "NOT significant (spans 0)",
                "approx_randomization_p": round(p_ar, 4),
                "p_note": f"cannot go below 1/(B+1) = {1 / (args.n_perm + 1):.5f}",
            },
        }
        r = report["arms"][arm]
        print(f"  {arm} on {len(idx)} notes: "
              f"3B/no-note {point['3B_nonote']:.4f}  7B/no-note {point['7B_nonote']:.4f}  "
              f"3B/note {point['3B_note']:.4f}  7B/note {point['7B_note']:.4f}")
        print(f"    capacity effect  with note {simple_note:+.4f}  without note {simple_nonote:+.4f}")
        print(f"    INTERACTION {interaction:+.4f} CI {r['interaction']['ci95']} "
              f"p={r['interaction']['approx_randomization_p']} {r['interaction']['verdict']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
