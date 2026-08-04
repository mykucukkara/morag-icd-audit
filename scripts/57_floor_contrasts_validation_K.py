#!/usr/bin/env python3
"""
Script 57: §4.1b's floor deficits, against the floor §4.1b actually claims.

§4.1b states a note-blind floor whose K is chosen on validation and evaluated once on test
(K = 8, micro-F1 0.3104). The three deficits quoted in the same paragraph — 0.101 / 0.118 / 0.171 —
were not computed against that floor. They are `A_arms_vs_floor.per_arm.*.vs_floor_bestK` from
`revision_round1_analyses.json`, where `best_K` is the K that maximizes the floor *on test*
(K = 10, micro-F1 0.304): the procedure the same paragraph disowns as the previous version's.

Both numbers were individually true of their own artifact, which is why the number-level guard
passed them for three review rounds. Nothing checked that the floor in the sentence and the floor in
the contrast were the same floor.

This recomputes the three contrasts against the validation-selected floor, and it recomputes the
intervals rather than rescaling the point estimates: a deficit is a paired quantity, and shifting the
reference changes the resample distribution, not just its centre.

Two properties are deliberate:

  * K is re-derived here from validation rather than read from the earlier artifact, so the file is
    self-contained evidence that the selection is what the manuscript says it is. The script fails if
    validation does not select 8 — a silent disagreement would be the same class of error it exists
    to correct.
  * The statistics are imported from scripts/37 through scripts/43, not reimplemented, so these
    contrasts pass through the identical machinery as every other contrast in the paper.

It also emits the discriminative lift for the 14B cells (§5.2). The Discussion attributes the
pipeline's movement to a judge that "becomes good enough (lift 3.50)", which is the 7B E11 cell;
the headline configuration is 14B, and no 14B lift existed in any artifact. Where the lift is
undefined the reason is recorded rather than a zero written: the evidence-constrained arms emit only
codes they marked supported, so there is no unsupported group to contrast and the quantity does not
exist for them at any capacity.

Usage:
    python scripts/57_floor_contrasts_validation_K.py \
        --campaign results_eurohpc/primary_campaign/top50 \
        --cap14b-note results_eurohpc/cap14b_note/top50 \
        --cap7b-note results_eurohpc/steelman7b/top50 \
        --train data/splits_real/top50/train.jsonl \
        --validation data/splits_real/top50/validation.jsonl \
        --out results_eurohpc/primary_campaign/floor_contrasts_validation_K.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_43():
    """Import scripts/43 so the floor materialization and the paired test are the same code.

    scripts/43 imports scripts/37 in turn, so the bootstrap, the confidence interval and the
    approximate-randomization test all come from the one implementation the rest of the paper uses.
    """
    spec = importlib.util.spec_from_file_location("steel43", HERE / "43_steelman7b_contrasts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S43 = _load_43()
PB = S43.PB

CANDIDATE_K = [3, 5, 8, 10, 12, 15, 20, 25]


def _gold_sets(path: Path) -> list[set[str]]:
    out: list[set[str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append({PB._norm(c) for c in (rec.get("gold_codes") or rec.get("labels") or [])})
    return out


def constant_predictor_micro_f1(gold: list[set[str]], codes: list[str]) -> dict:
    """Micro-F1 of a constant predictor emitting `codes` for every note."""
    pred = set(codes)
    tp = fp = fn = 0
    for g in gold:
        hit = len(pred & g)
        tp += hit
        fp += len(pred) - hit
        fn += len(g) - hit
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"micro_f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "n": len(gold), "codes_per_note": float(len(pred))}


def select_K_on_validation(train: Path, validation: Path) -> tuple[int, dict]:
    gold = _gold_sets(validation)
    by_K = {}
    for k in CANDIDATE_K:
        by_K[f"K={k}"] = constant_predictor_micro_f1(gold, S43.top_train_codes(train, k))
    best = max(CANDIDATE_K, key=lambda k: by_K[f"K={k}"]["micro_f1"])
    return best, by_K


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True,
                    help="primary campaign top50 tree holding E6/E11/E14")
    ap.add_argument("--cap14b-note", default=None, help="14B with-note tree (for the §5.2 lift)")
    ap.add_argument("--cap7b-note", default=None, help="7B with-note tree (reproduces lift 3.50)")
    ap.add_argument("--steelman-3b", default=None, help="3B with-note tree (the curve's first point)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expect-K", type=int, default=8,
                    help="fail if validation does not select this K; 0 disables the assertion")
    args = ap.parse_args()

    train, validation = Path(args.train), Path(args.validation)
    campaign = Path(args.campaign)

    print("Selecting K on the validation split")
    K, by_K = select_K_on_validation(train, validation)
    print(f"  selected K = {K}  (validation micro-F1 {by_K[f'K={K}']['micro_f1']})")
    if args.expect_K and K != args.expect_K:
        raise SystemExit(
            f"validation selects K={K}, but the manuscript states K={args.expect_K}. "
            "Resolve the disagreement before using either number.")

    floor_codes = S43.top_train_codes(train, K)
    out: dict = {
        "top_n": 50,
        "seed": args.seed,
        "note": ("Deficits of the note-blind floor over each arm, against the floor whose K is "
                 "selected on validation and evaluated once on test. Supersedes the "
                 "`vs_floor_bestK` deltas in revision_round1_analyses.json, which used the K that "
                 "maximizes the floor on test."),
        "floor": {
            "selection_split": "validation",
            "candidate_K": CANDIDATE_K,
            "by_K_validation": by_K,
            "selected_K": K,
        },
        "sign_convention": "delta = floor - arm; positive means the note-blind floor is ahead",
        "contrasts": {},
    }

    tmp = Path(args.out).parent / "_floor_predictions"
    for arm in ("E6", "E11", "E14"):
        print(f"\n{arm} vs note-blind floor (K={K})")
        arm_path = S43.find_predictions(campaign, arm)
        if arm_path is None:
            out["contrasts"][arm] = {"error": f"no predictions found for {arm} under {campaign}"}
            print(f"    MISSING: no predictions for {arm}")
            continue
        floor_path = S43.write_floor(PB.load(arm_path), floor_codes, tmp / f"floor_K{K}_{arm}.jsonl")
        c = S43.contrast(arm_path, floor_path, arm, "note_blind_floor",
                         args.n_boot, args.n_perm, args.seed)
        out["contrasts"][arm] = c
        d = c.get("delta_micro_f1", {})
        print(f"    arm {c.get(arm, {}).get('micro_f1')}  floor "
              f"{c.get('note_blind_floor', {}).get('micro_f1')}  "
              f"deficit {d.get('point')} CI {d.get('ci95')}  {d.get('verdict')}")

    # ---- §5.2: the discriminative lift at each capacity point -----------------------------------
    #  The context condition is part of the key, not left to the reader. The ladder's 3B arms run
    #  with the note withheld on the full test split, while the capacity curve's cells run with the
    #  note supplied on 1,008 notes; a key reading only "E11_3B" invites the two to be compared as
    #  though they were one series, which is the error this round exists to remove.
    sources = [("3B", "note_withheld", campaign),
               ("3B", "note_supplied", Path(args.steelman_3b) if args.steelman_3b else None),
               ("7B", "note_supplied", Path(args.cap7b_note) if args.cap7b_note else None),
               ("14B", "note_supplied", Path(args.cap14b_note) if args.cap14b_note else None)]
    lifts: dict = {}
    for tag, ctx, root in sources:
        if root is None:
            continue
        for arm in ("E11", "E14"):
            p = S43.find_predictions(root, arm)
            if p is None:
                continue
            s = S43.arm_summary(p)
            lifts[f"{arm}_{tag}_{ctx}"] = {
                "capacity": tag, "scorer_context": ctx,
                "n": s["n"], "micro_f1": s["micro_f1"],
                "discriminative_lift": s["discriminative_lift"],
                "undefined_reason": (None if s["discriminative_lift"] else
                                     "the arm emits only codes it marked supported, so there is no "
                                     "unsupported group to contrast; this holds at every capacity"),
            }
    out["discriminative_lift_by_capacity"] = lifts
    out["discriminative_lift_note"] = (
        "The three note_supplied cells are the same 1,008 notes and form the capacity curve quoted "
        "in §5.2. The note_withheld entry is the ladder arm on the full test split and is not a "
        "point on that curve.")
    print("\nDiscriminative lift by capacity")
    for k, v in lifts.items():
        print(f"  {k}: lift {v['discriminative_lift']}  (micro-F1 {v['micro_f1']}, n {v['n']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
