#!/usr/bin/env python3
"""
Paired bootstrap over notes for two systems evaluated on the SAME notes.

The repo previously contained no significance testing of any kind (no t-test, Wilcoxon,
McNemar, permutation or p-value anywhere in src/ or scripts/), so every reported difference
was a bare point estimate. That is not sufficient for the manuscript's central claims, which
are largely NULL claims ("the RAG arms are indistinguishable from retrieval-only", "the
larger model does not help") — and a null claim without an interval is not a finding.

Resampling is at the NOTE level, with replacement, and the SAME resampled note indices are
applied to both systems (paired). Per-note contingency counts are precomputed once, so the
resampling loop is pure integer arithmetic.

Note-level rather than code-level resampling is deliberate: the code decisions inside one
note are not independent (shared candidate set, shared evidence, a shared prediction budget),
so a code-level bootstrap — like McNemar over codes — would understate the variance.

PHI-safe: reads only prediction records and emits aggregate statistics.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _norm(code: str) -> str:
    return str(code or "").replace(".", "").strip().upper()


def load(path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "success":
                continue
            gsi = rec.get("global_sample_index")
            if gsi is not None:
                out[int(gsi)] = rec
    return out


def note_counts(rec: dict) -> Tuple[int, ...]:
    """(tp, fp, fn, supported_gold, supported_total, unsupported_gold, unsupported_total)."""
    gold = {_norm(c) for c in (rec.get("gold_codes") or [])}
    preds = rec.get("predicted_codes") or []
    pset = {_norm(p.get("code")) for p in preds} - {""}
    sg = st = ug = ut = 0
    for p in preds:
        is_gold = _norm(p.get("code")) in gold
        if p.get("supported"):
            st += 1
            sg += int(is_gold)
        else:
            ut += 1
            ug += int(is_gold)
    return (len(pset & gold), len(pset - gold), len(gold - pset), sg, st, ug, ut)


def aggregate(counts: Dict[int, Tuple[int, ...]], sample: List[int]) -> Tuple[float, Optional[float]]:
    tp = fp = fn = sg = st = ug = ut = 0
    for i in sample:
        a, b, c, d, e, f, g = counts[i]
        tp += a; fp += b; fn += c; sg += d; st += e; ug += f; ut += g
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    lift = (sg / st) / (ug / ut) if (st and ut and ug) else None
    return f1, lift


def ci(values: List[float], level: float = 0.95) -> Tuple[float, float]:
    vs = sorted(values)
    n = len(vs)
    lo_i = int(((1 - level) / 2) * n)
    hi_i = min(n - 1, int((1 - (1 - level) / 2) * n))
    return vs[lo_i], vs[hi_i]


def approximate_randomization(
    CA: Dict[int, Tuple[int, ...]],
    CB: Dict[int, Tuple[int, ...]],
    idx: List[int],
    n_perm: int,
    seed: int,
) -> Tuple[float, float]:
    """Two-sided approximate-randomization p-value for the micro-F1 difference.

    Under H0 the two systems are exchangeable within each note, so on each permutation we
    independently swap A<->B per note and recompute the aggregate delta. p = share of
    permutations whose |delta| >= |observed delta| (add-one smoothed). This complements the
    bootstrap CI: the CI states the effect's plausible range, the permutation test states how
    surprising it is under no-difference. Returns (observed_delta, p_value).
    """
    obs = aggregate(CB, idx)[0] - aggregate(CA, idx)[0]
    rng = random.Random(seed)
    at_least_as_extreme = 0
    abs_obs = abs(obs)
    for _ in range(n_perm):
        left: Dict[int, Tuple[int, ...]] = {}
        right: Dict[int, Tuple[int, ...]] = {}
        for i in idx:
            if rng.random() < 0.5:
                left[i], right[i] = CA[i], CB[i]
            else:
                left[i], right[i] = CB[i], CA[i]
        delta = aggregate(right, idx)[0] - aggregate(left, idx)[0]
        if abs(delta) >= abs_obs:
            at_least_as_extreme += 1
    p = (at_least_as_extreme + 1) / (n_perm + 1)
    return obs, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="System A predictions jsonl (baseline)")
    ap.add_argument("--b", required=True, help="System B predictions jsonl (comparison)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    A, B = load(Path(args.a)), load(Path(args.b))
    idx = sorted(set(A) & set(B))
    if not idx:
        raise SystemExit("No shared notes between the two runs; nothing to compare.")

    CA = {i: note_counts(A[i]) for i in idx}
    CB = {i: note_counts(B[i]) for i in idx}
    f1_a, lift_a = aggregate(CA, idx)
    f1_b, lift_b = aggregate(CB, idx)

    rng = random.Random(args.seed)
    d_f1: List[float] = []
    d_lift: List[float] = []
    for _ in range(args.n_boot):
        s = [rng.choice(idx) for _ in idx]
        fa, la = aggregate(CA, s)
        fb, lb = aggregate(CB, s)
        d_f1.append(fb - fa)
        if la is not None and lb is not None:
            d_lift.append(lb - la)

    def verdict(lo: float, hi: float) -> str:
        return "SIGNIFICANT (excludes 0)" if (lo > 0 or hi < 0) else "NOT significant (spans 0)"

    f1_lo, f1_hi = ci(d_f1)
    _, p_ar = approximate_randomization(CA, CB, idx, args.n_perm, args.seed)
    report = {
        "shared_notes": len(idx),
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "seed": args.seed,
        args.label_a: {"micro_f1": round(f1_a, 4), "discriminative_lift": round(lift_a, 3) if lift_a else None},
        args.label_b: {"micro_f1": round(f1_b, 4), "discriminative_lift": round(lift_b, 3) if lift_b else None},
        "delta_micro_f1": {
            "point": round(f1_b - f1_a, 4),
            "ci95": [round(f1_lo, 4), round(f1_hi, 4)],
            "verdict": verdict(f1_lo, f1_hi),
            "approx_randomization_p": round(p_ar, 4),
        },
    }
    if d_lift:
        l_lo, l_hi = ci(d_lift)
        report["delta_lift"] = {
            "point": round(lift_b - lift_a, 3) if (lift_a and lift_b) else None,
            "ci95": [round(l_lo, 3), round(l_hi, 3)],
            "verdict": verdict(l_lo, l_hi),
            "n_resamples_defined": len(d_lift),
        }
    else:
        report["delta_lift"] = {"point": None, "note": "lift undefined in one or both systems"}

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
