#!/usr/bin/env python3
"""
Script 47: the analyses an independent reviewer asked for, all from stored outputs.

Four separate objections, none of which needed new inference:

A. **The note-blind floor's K was selected on the test set.** Choosing the best K over the test
   split and then reporting that K's score is selection on the evaluation data. K is now chosen on
   the training distribution and on validation, and evaluated once on test.

B. **TF-IDF and the full model were compared at different output cardinalities.** A fifteen-code
   cap is not fifteen codes emitted: E1 emits ~15, E14 emits 4.3. Micro-F1 moves with cardinality,
   so the ladder confounds architecture with how many codes each arm chooses to emit. This sweeps
   both arms across matched budgets.

C. **The random-pruning null was not cardinality-matched** (4.0 codes/note against the full model's
   4.26). Re-run so the null emits, per note, exactly as many codes as the full model did.

D. **Stage-wise recall was never measured**, so attributing a share of the loss to "retrieval" or
   "the selector" rested on differences of F1 rather than on where gold codes actually disappear.

Reads predictions and the split files; writes one JSON. Aggregates only.

Usage:
    python scripts/47_reviewer_round2_analyses.py \
        --campaign results_eurohpc/primary_campaign/top50 \
        --splits-root data/splits_real --top-n 50 \
        --out results_eurohpc/primary_campaign/reviewer_round2_analyses.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_37():
    spec = importlib.util.spec_from_file_location("pb37", HERE / "37_paired_bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PB = _load_37()


def find_pred(root: Path, arm: str) -> Path | None:
    for c in (root / f"{arm}_seed42" / "merged" / "test_predictions.jsonl",
              root / f"{arm}_seed42" / "test_predictions.jsonl"):
        if c.exists():
            return c
    return None


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"micro_f1": round(f, 4), "precision": round(p, 4), "recall": round(r, 4)}


def score_pairs(pairs) -> dict:
    """pairs: iterable of (predicted set, gold set)."""
    tp = fp = fn = n = 0
    codes = 0
    for pred, gold in pairs:
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        codes += len(pred)
        n += 1
    out = prf(tp, fp, fn)
    out["n"] = n
    out["codes_per_note"] = round(codes / max(n, 1), 2)
    return out


def load_split(path: Path) -> list[set]:
    golds = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            golds.append({PB._norm(c) for c in (rec.get("gold_codes") or rec.get("labels") or [])})
    return golds


def ranked_codes(rec: dict) -> list[str]:
    """Predicted codes in the order the pipeline ranked them."""
    return [PB._norm(p.get("code")) for p in (rec.get("predicted_codes") or []) if p.get("code")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="results_eurohpc/primary_campaign/top50")
    ap.add_argument("--splits-root", default="data/splits_real")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--scalability", default="results_eurohpc/scalability_v2",
                    help="Root holding top100/ and top200/ prediction directories")
    ap.add_argument("--scalability-v1", default="results_eurohpc/scalability",
                    help="Fallback root: E1 was not affected by the label-space defect and was not re-run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_eurohpc/primary_campaign/reviewer_round2_analyses.json")
    args = ap.parse_args()
    C, S = Path(args.campaign), Path(args.splits_root) / f"top{args.top_n}"
    report: dict = {"top_n": args.top_n, "seed": args.seed}

    # ---------- A. note-blind floor, K chosen off the test set ----------
    train_freq: Counter[str] = Counter()
    for g in load_split(S / "train.jsonl"):
        train_freq.update(g)
    val_golds = load_split(S / "validation.jsonl")
    test_golds = load_split(S / "test.jsonl")

    KS = [3, 5, 8, 10, 12, 15, 20, 25]
    by_k_val = {}
    for K in KS:
        const = {c for c, _ in train_freq.most_common(K)}
        by_k_val[K] = score_pairs((const, g) for g in val_golds)
    best_k = max(by_k_val, key=lambda k: by_k_val[k]["micro_f1"])
    const_best = {c for c, _ in train_freq.most_common(best_k)}
    report["A_floor_selected_on_validation"] = {
        "selection_split": "validation",
        "by_K_validation": {f"K={k}": v for k, v in by_k_val.items()},
        "selected_K": best_k,
        "test_once": score_pairs((const_best, g) for g in test_golds),
        "test_at_matched_budget_K15": score_pairs(
            ({c for c, _ in train_freq.most_common(15)}, g) for g in test_golds),
        "note": ("K is chosen on validation and the selected K is evaluated once on test; the "
                 "earlier figure picked the best K on test itself."),
    }
    print(f"  A. floor: K={best_k} chosen on validation -> test micro-F1 "
          f"{report['A_floor_selected_on_validation']['test_once']['micro_f1']}")

    # ---------- B. matched output cardinality ----------
    arms = {}
    for arm in ("E1", "E11", "E14"):
        p = find_pred(C, arm)
        if p:
            arms[arm] = PB.load(p)
    budgets = [1, 2, 3, 4, 5, 8, 10, 15]
    card = {}
    for arm, runs in arms.items():
        idx = sorted(runs)
        rows = {}
        for b in budgets:
            rows[f"top{b}"] = score_pairs(
                (set(ranked_codes(runs[i])[:b]), {PB._norm(c) for c in (runs[i].get("gold_codes") or [])})
                for i in idx)
        rows["as_run"] = score_pairs(
            (set(ranked_codes(runs[i])), {PB._norm(c) for c in (runs[i].get("gold_codes") or [])})
            for i in idx)
        card[arm] = rows
    report["B_matched_cardinality"] = {
        "by_arm": card,
        "note": ("A fifteen-code cap is a maximum, not an emitted count: E1 emits about fifteen "
                 "codes per note and E14 about four. Comparing the arms at the same budget "
                 "separates the architecture from the number of codes each chooses to emit."),
    }
    if "E1" in card and "E14" in card:
        e14n = card["E14"]["as_run"]["codes_per_note"]
        nearest = min(budgets, key=lambda b: abs(b - e14n))
        print(f"  B. E14 emits {e14n} codes/note; E1 at top{nearest} = "
              f"{card['E1'][f'top{nearest}']['micro_f1']} vs E14 as-run {card['E14']['as_run']['micro_f1']}")

    # ---------- C. cardinality-matched random-pruning null ----------
    if "E11" in arms and "E14" in arms:
        e11, e14 = arms["E11"], arms["E14"]
        shared = sorted(set(e11) & set(e14))
        rng = random.Random(args.seed)
        pairs_rand, pairs_oracle, pairs_full = [], [], []
        for i in shared:
            gold = {PB._norm(c) for c in (e14[i].get("gold_codes") or [])}
            shortlist = ranked_codes(e11[i])
            k = len(ranked_codes(e14[i]))            # per-note budget, not a fixed 4
            pairs_full.append((set(ranked_codes(e14[i])), gold))
            pairs_rand.append((set(rng.sample(shortlist, min(k, len(shortlist)))), gold))
            ordered = [c for c in shortlist if c in gold] + [c for c in shortlist if c not in gold]
            pairs_oracle.append((set(ordered[:k]), gold))
        report["C_null_matched_per_note"] = {
            "shared_notes": len(shared),
            "full_model_E14": score_pairs(pairs_full),
            "random_pruning_null": score_pairs(pairs_rand),
            "oracle_over_shortlist": score_pairs(pairs_oracle),
            "note": ("The null and the oracle now emit exactly as many codes as the full model did "
                     "on that note, so the comparison isolates which codes are kept, not how many. "
                     "The shortlist is E11's emitted ranking, which already reflects LLM scoring: "
                     "the oracle bounds selection over that list, not retrieval coverage."),
        }
        print(f"  C. matched null {report['C_null_matched_per_note']['random_pruning_null']['micro_f1']} "
              f"vs E14 {report['C_null_matched_per_note']['full_model_E14']['micro_f1']} "
              f"vs oracle {report['C_null_matched_per_note']['oracle_over_shortlist']['micro_f1']}")

    # ---------- D. where gold codes are lost, stage by stage ----------
    stages = {}
    for arm, label in (("E6", "retrieval_only_top15"), ("E11", "rag_top15"),
                       ("E12", "plus_evidence_constraint"), ("E13", "plus_contrastive"),
                       ("E14", "full_model")):
        p = find_pred(C, arm)
        if not p:
            continue
        runs = PB.load(p)
        idx = sorted(runs)
        tot_gold = kept = 0
        for i in idx:
            gold = {PB._norm(c) for c in (runs[i].get("gold_codes") or [])}
            tot_gold += len(gold)
            kept += len(gold & set(ranked_codes(runs[i])))
        stages[label] = {"arm": arm, "n": len(idx), "gold_total": tot_gold,
                         "gold_retained": kept, "gold_recall": round(kept / max(tot_gold, 1), 4)}
    report["D_stagewise_gold_recall"] = {
        "by_stage": stages,
        "note": ("Recall of gold codes in each arm's emitted set. These are descriptive retention "
                 "rates along the ladder, not an additive decomposition of the F1 gap."),
    }
    for k, v in stages.items():
        print(f"  D. {k:26s} gold recall {v['gold_recall']}")

    # ---------- E. label-space trend on notes shared between splits ----------
    # The splits are rebuilt per label space (notes with no in-vocabulary code are dropped, and the
    # subject permutation is recomputed), so the Top-50/100/200 test sets are different patient
    # populations: they share only 1,417 of ~9,200 subjects. The across-label-space trend is
    # therefore confounded with a change of cohort. Restricting to notes present in both splits
    # gives a paired check of whether the direction survives.
    scal = Path(args.scalability) if args.scalability else None
    if scal and scal.exists():
        def note_key(rec):
            return rec.get("hadm_id")

        def arm_by_note(path: Path):
            out = {}
            if not path or not path.exists():
                return out
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("status") != "success":
                        continue
                    k = note_key(r)
                    if k is not None:
                        out[k] = r
            return out

        shared_report = {}
        base50 = {a: arm_by_note(find_pred(C, a)) for a in ("E1", "E14")}
        for tn in (100, 200):
            root = scal / f"top{tn}"
            alt = Path(args.scalability_v1) / f"top{tn}" if args.scalability_v1 else None
            other = {}
            for a in ("E1", "E14"):
                pth = find_pred(root, a) or (find_pred(alt, a) if alt and alt.exists() else None)
                other[a] = arm_by_note(pth)
            if not all(other.values()) or not all(base50.values()):
                print(f"  E. Top-{tn}: predictions missing, skipped")
                continue
            keys = sorted(set(base50["E1"]) & set(base50["E14"]) & set(other["E1"]) & set(other["E14"]))
            if len(keys) < 200:
                print(f"  E. Top-{tn}: only {len(keys)} shared notes, skipped")
                continue
            cell = {}
            for label, src in (("top50", base50), (f"top{tn}", other)):
                for arm in ("E1", "E14"):
                    cell[f"{label}_{arm}"] = score_pairs(
                        (set(ranked_codes(src[arm][k])),
                         {PB._norm(c) for c in (src[arm][k].get("gold_codes") or [])})
                        for k in keys)
            g50 = cell["top50_E1"]["micro_f1"] - cell["top50_E14"]["micro_f1"]
            gtn = cell[f"top{tn}_E1"]["micro_f1"] - cell[f"top{tn}_E14"]["micro_f1"]
            shared_report[f"top50_vs_top{tn}"] = {
                "shared_notes": len(keys), "cells": cell,
                "E1_lead_top50": round(g50, 4), f"E1_lead_top{tn}": round(gtn, 4),
                "widens": bool(gtn > g50),
            }
            print(f"  E. Top-50 vs Top-{tn} on {len(keys)} shared notes: "
                  f"lead {g50:.4f} -> {gtn:.4f} ({'widens' if gtn > g50 else 'does not widen'})")
        if shared_report:
            report["E_label_space_on_shared_notes"] = {
                "by_pair": shared_report,
                "note": ("Each label space has its own subject-disjoint partition, so the headline "
                         "across-label-space trend compares different cohorts. These paired "
                         "subsets hold the notes fixed and vary only the label space."),
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
