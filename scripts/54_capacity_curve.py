#!/usr/bin/env python3
"""
Script 54: the capacity axis as a three-point curve, and what the best cell is actually worth.

`scripts/48` established the context x capacity interaction from a 2 x 2 (3B/7B). A reviewer's
objection to that design was that two points are not a trend, so T3-4 added a third capacity point
(Qwen2.5-14B) at both context levels. This script does two things with it.

**(1) The curve.** All six cells on one note set, with the interaction recomputed at each capacity
step. If capacity helps only when the note is supplied, the two slopes should separate and keep
separating.

**(2) The comparison the curve makes necessary.** The 14B-with-note cell is far above anything the
paper reports, which puts the manuscript's central contrast — the pipeline losing several-fold to
TF-IDF — at risk of describing only the 3B-without-note corner. The honest test is not to set that
cell beside the ladder's full-test-set TF-IDF number; the ladder is scored on 17,151 notes and this
cell on 1,008, and comparing across samples is precisely the error this paper is about. So TF-IDF
and the note-blind floor are re-scored on the *same* notes and contrasted with a paired bootstrap,
exactly as every other contrast in the paper is.

The floor is materialized as a prediction file (via `scripts/43`'s writer) so it passes through the
identical test rather than being compared as a bare number.

Usage:
    python scripts/54_capacity_curve.py \
        --note-3b results_eurohpc/steelman/top50 \
        --note-7b results_eurohpc/steelman7b/top50 \
        --note-14b results_eurohpc/cap14b_note/top50 \
        --nonote-3b results_eurohpc/primary_campaign/top50 \
        --nonote-7b results_eurohpc/cap7b_nonote/top50 \
        --nonote-14b results_eurohpc/cap14b_nonote/top50 \
        --baseline results_eurohpc/primary_campaign/top50/E1_seed42 \
        --train data/splits_real/top50/train.jsonl --floor-k 8 \
        --out results_eurohpc/primary_campaign/capacity_curve.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").replace(".", "_"),
                                                  HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PB = _load("37_paired_bootstrap.py")
S43 = _load("43_steelman7b_contrasts.py")

CAPACITIES = ("3B", "7B", "14B")


def f1_from(counts: dict, sample) -> float:
    tp = fp = fn = 0
    for i in sample:
        a, b, c = counts[i][:3]
        tp += a; fp += b; fn += c
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def interaction_between(counts, idx, lo_cap, hi_cap, n_boot, n_perm, seed):
    """Difference-in-differences for one capacity step, same machinery as scripts/48."""
    def d(sample, ctx):
        return f1_from(counts[f"{hi_cap}_{ctx}"], sample) - f1_from(counts[f"{lo_cap}_{ctx}"], sample)

    observed = d(idx, "note") - d(idx, "nonote")
    rng = random.Random(seed)
    boot = [(lambda s: d(s, "note") - d(s, "nonote"))([rng.choice(idx) for _ in idx])
            for _ in range(n_boot)]

    rngp = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = 0.0
        for ctx, sign in (("note", 1), ("nonote", -1)):
            sw = {i: rngp.random() < 0.5 for i in idx}
            hi = {i: counts[f"{lo_cap}_{ctx}"][i] if sw[i] else counts[f"{hi_cap}_{ctx}"][i] for i in idx}
            lo = {i: counts[f"{hi_cap}_{ctx}"][i] if sw[i] else counts[f"{lo_cap}_{ctx}"][i] for i in idx}
            perm += sign * (f1_from(hi, idx) - f1_from(lo, idx))
        if abs(perm) >= abs(observed):
            extreme += 1

    lo_ci, hi_ci = PB.ci(boot)
    return {"step": f"{lo_cap}->{hi_cap}",
            "capacity_effect_with_note": round(d(idx, "note"), 4),
            "capacity_effect_without_note": round(d(idx, "nonote"), 4),
            "interaction": {"point": round(observed, 4), "ci95": [round(lo_ci, 4), round(hi_ci, 4)],
                            "verdict": "SIGNIFICANT (excludes 0)" if lo_ci * hi_ci > 0 else "not significant",
                            "approx_randomization_p": round((extreme + 1) / (n_perm + 1), 5),
                            "p_resolution_floor": round(1.0 / (n_perm + 1), 5)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    for ctx in ("note", "nonote"):
        for cap in CAPACITIES:
            ap.add_argument(f"--{ctx}-{cap.lower()}", required=True)
    ap.add_argument("--baseline", default="results_eurohpc/primary_campaign/top50/E1_seed42")
    ap.add_argument("--train", default="data/splits_real/top50/train.jsonl")
    ap.add_argument("--floor-k", type=int, default=8,
                    help="K of the note-blind floor; 8 is the validation-selected value of §4.1b")
    ap.add_argument("--arms", nargs="+", default=["E11", "E14"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_eurohpc/primary_campaign/capacity_curve.json")
    args = ap.parse_args()

    roots = {f"{cap}_{ctx}": getattr(args, f"{ctx}_{cap.lower()}")
             for ctx in ("note", "nonote") for cap in CAPACITIES}
    report: dict = {"n_boot": args.n_boot, "n_perm": args.n_perm, "seed": args.seed,
                    "floor_K": args.floor_k, "arms": {}}
    floor_codes = S43.top_train_codes(Path(args.train), args.floor_k)
    report["floor_codes"] = floor_codes

    for arm in args.arms:
        paths, missing = {}, []
        for cell, root in roots.items():
            p = S43.find_predictions(Path(root), arm)
            (paths.__setitem__(cell, p) if p is not None else missing.append(f"{cell}:{root}"))
        if missing:
            print(f"  {arm}: missing {', '.join(missing)} — skipped")
            continue

        loaded = {k: PB.load(v) for k, v in paths.items()}
        idx = sorted(set.intersection(*(set(v) for v in loaded.values())))
        if len(idx) < 100:
            print(f"  {arm}: only {len(idx)} notes shared by all six cells — skipped")
            continue
        counts = {k: {i: PB.note_counts(v[i]) for i in idx} for k, v in loaded.items()}

        entry = {
            "shared_notes": len(idx),
            "curve_micro_f1": {ctx: {cap: round(f1_from(counts[f"{cap}_{ctx}"], idx), 4)
                                     for cap in CAPACITIES} for ctx in ("note", "nonote")},
            "steps": [interaction_between(counts, idx, a, b, args.n_boot, args.n_perm, args.seed)
                      for a, b in zip(CAPACITIES, CAPACITIES[1:])],
        }
        entry["steps"].append(
            interaction_between(counts, idx, "3B", "14B", args.n_boot, args.n_perm, args.seed))
        report["arms"][arm] = entry
        print(f"  {arm}: {len(idx)} shared notes")
        for ctx in ("note", "nonote"):
            print(f"      {ctx:7s} " + "  ".join(
                f"{cap}={entry['curve_micro_f1'][ctx][cap]:.4f}" for cap in CAPACITIES))

    # ---- what the best cell is worth, on its own notes -------------------------------------
    best_arm, best_cell = "E14", "14B_note"
    best = S43.find_predictions(Path(roots[best_cell]), best_arm)
    if best is None:
        print(f"  best cell {best_arm}/{best_cell} not found — reference contrasts skipped")
    else:
        out_dir = Path(args.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        report["best_cell"] = {"arm": best_arm, "cell": best_cell}
        report["reference_contrasts"] = {}

        base = (S43.find_predictions(Path(args.baseline).parent, "E1")
                or S43.find_predictions(Path(args.baseline), "E1"))
        if base is not None:
            report["reference_contrasts"]["vs_E1_tfidf"] = S43.contrast(
                base, best, "E1_tfidf", f"{best_arm}_{best_cell}",
                args.n_boot, args.n_perm, args.seed)
            print("  best cell vs TF-IDF: done")
        else:
            print(f"  TF-IDF predictions not found under {args.baseline}")

        floor_path = S43.write_floor(PB.load(best), floor_codes,
                                     out_dir / f"_floor_K{args.floor_k}_capacity_subset.jsonl")
        report["reference_contrasts"]["vs_note_blind_floor"] = S43.contrast(
            floor_path, best, f"floor_K{args.floor_k}", f"{best_arm}_{best_cell}",
            args.n_boot, args.n_perm, args.seed)
        print("  best cell vs note-blind floor: done")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
