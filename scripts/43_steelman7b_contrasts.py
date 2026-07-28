#!/usr/bin/env python3
"""
Script 43: the §4.5 steelman contrasts, as stored artifacts.

The three claims that carry Section 4.5 — 3B → 7B with the note supplied, the best configuration
against the tuned classical baseline, and the best configuration against the note-blind floor — were
computed interactively during revision. That is exactly the provenance the paper argues against, so
they are recomputed here into a file:

  E14: 3B-with-note      vs 7B-with-note      (the interaction effect)
  E14: 7B-with-note      vs E1 TF-IDF         (best configuration vs the classical baseline)
  E14: 7B-with-note      vs note-blind floor  (best configuration vs a predictor that never reads)
  E11: 3B-with-note      vs 7B-with-note      (same interaction in the unconstrained arm)

All four use the identical machinery as every other contrast in the paper: note-level paired
bootstrap plus an approximate-randomization test, imported from scripts/37 rather than reimplemented,
and every contrast is restricted to the notes both systems actually produced.

The note-blind floor is materialized as a prediction file over the same note indices (the K most
frequent TRAINING codes, constant for every note) so that it passes through the same paired test as
a real system. Written to the same output tree as the other analyses; aggregates only.

Usage:
    python scripts/43_steelman7b_contrasts.py \
        --steelman-3b results_eurohpc/steelman/top50 \
        --steelman-7b results_eurohpc/steelman7b/top50 \
        --baseline results_eurohpc/primary_campaign/top50/E1_seed42 \
        --train data/splits_real/top50/train.jsonl \
        --out results_eurohpc/primary_campaign/steelman7b_contrasts.json
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
    """Import scripts/37 as a module so the statistics are the same code, not a copy."""
    spec = importlib.util.spec_from_file_location("pb37", HERE / "37_paired_bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PB = _load_37()


def _count_notes(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def find_predictions(root: Path, arm: str) -> Path | None:
    """Locate an arm's predictions, preferring whichever source covers the most notes.

    Two traps live in this tree. The 7B steelman arms were run as script-level shards and never
    merged. And some arms also have a five-note smoke file sitting at the canonical path, which
    shadows the real 1,008-note shard set — pairing against it silently reduces a contrast to five
    notes and produces a confident-looking null. Counting notes and taking the largest source
    removes both traps, and the choice is printed so it can be audited.
    """
    candidates: list[tuple[int, str, Path]] = []
    for cand in (root / f"{arm}_seed42" / "merged" / "test_predictions.jsonl",
                 root / f"{arm}_seed42" / "test_predictions.jsonl",
                 root / "merged" / "test_predictions.jsonl",
                 root / "test_predictions.jsonl"):
        if cand.exists():
            candidates.append((_count_notes(cand), "single file", cand))

    shards = sorted((root / f"{arm}_seed42" / "shards").glob("shard_*/test_predictions.jsonl"))
    if shards:
        out = root / f"{arm}_seed42" / "merged_from_shards" / "test_predictions.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        seen: set[int] = set()
        with open(out, "w", encoding="utf-8") as fh:
            for sh in shards:
                with open(sh, encoding="utf-8") as src:
                    for line in src:
                        if not line.strip():
                            continue
                        gsi = json.loads(line).get("global_sample_index")
                        # Shards are disjoint by construction; drop any overlap rather than
                        # double-count a note, which would silently narrow every interval.
                        if gsi is not None and gsi in seen:
                            continue
                        if gsi is not None:
                            seen.add(int(gsi))
                        fh.write(line if line.endswith("\n") else line + "\n")
        candidates.append((len(seen), f"{len(shards)} merged shards", out))

    if not candidates:
        return None
    n, how, path = max(candidates, key=lambda c: c[0])
    if len(candidates) > 1:
        others = ", ".join(f"{c[0]} notes via {c[1]}" for c in candidates if c[2] != path)
        print(f"    {arm}: using {n} notes via {how} (rejected: {others})")
    else:
        print(f"    {arm}: using {n} notes via {how}")
    return path


def top_train_codes(train_path: Path, k: int) -> list[str]:
    freq: Counter[str] = Counter()
    with open(train_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            for c in (rec.get("gold_codes") or rec.get("labels") or []):
                freq[PB._norm(c)] += 1
    return [c for c, _ in freq.most_common(k)]


def write_floor(reference: dict[int, dict], codes: list[str], out_path: Path) -> Path:
    """Materialize the constant note-blind predictor over the reference run's notes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for idx, rec in sorted(reference.items()):
            fh.write(json.dumps({
                "global_sample_index": idx,
                "status": "success",
                "gold_codes": rec.get("gold_codes") or [],
                # 'supported' is deliberately absent: the floor makes no evidence judgement, so the
                # evidence metrics must read as not-applicable rather than as a fabricated zero.
                "predicted_codes": [{"code": c} for c in codes],
            }) + "\n")
    return out_path


def contrast(a_path: Path, b_path: Path, label_a: str, label_b: str,
             n_boot: int, n_perm: int, seed: int) -> dict:
    A, B = PB.load(a_path), PB.load(b_path)
    idx = sorted(set(A) & set(B))
    if not idx:
        return {"error": "no shared notes", "a": str(a_path), "b": str(b_path)}
    CA = {i: PB.note_counts(A[i]) for i in idx}
    CB = {i: PB.note_counts(B[i]) for i in idx}
    f1_a, _ = PB.aggregate(CA, idx)
    f1_b, _ = PB.aggregate(CB, idx)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        deltas.append(PB.aggregate(CB, s)[0] - PB.aggregate(CA, s)[0])
    lo, hi = PB.ci(deltas)
    _, p_ar = PB.approximate_randomization(CA, CB, idx, n_perm, seed)
    return {
        "shared_notes": len(idx),
        "n_boot": n_boot, "n_perm": n_perm, "seed": seed,
        label_a: {"micro_f1": round(f1_a, 4)},
        label_b: {"micro_f1": round(f1_b, 4)},
        "delta_micro_f1": {
            "point": round(f1_b - f1_a, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "verdict": "SIGNIFICANT (excludes 0)" if (lo > 0 or hi < 0) else "NOT significant (spans 0)",
            "approx_randomization_p": round(p_ar, 4),
            "p_note": f"approximate randomization cannot go below 1/(B+1) = {1 / (n_perm + 1):.5f}",
        },
    }


def arm_summary(path: Path) -> dict:
    """Per-arm aggregates for the Table 6 rows, from the same counts the contrasts use."""
    runs = PB.load(path)
    idx = sorted(runs)
    counts = {i: PB.note_counts(runs[i]) for i in idx}
    f1, lift = PB.aggregate(counts, idx)
    tp = sum(counts[i][0] for i in idx)
    fp = sum(counts[i][1] for i in idx)
    fn = sum(counts[i][2] for i in idx)
    predicted = sum(len(runs[i].get("predicted_codes") or []) for i in idx)
    return {
        "n": len(idx),
        "micro_f1": round(f1, 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "codes_per_note": round(predicted / len(idx), 2) if idx else 0.0,
        "discriminative_lift": round(lift, 3) if lift else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steelman-3b", default="results_eurohpc/steelman/top50")
    ap.add_argument("--steelman-7b", default="results_eurohpc/steelman7b/top50")
    ap.add_argument("--baseline", default="results_eurohpc/primary_campaign/top50/E1_seed42")
    ap.add_argument("--train", default="data/splits_real/top50/train.jsonl")
    ap.add_argument("--floor-k", type=int, default=10, help="K of the note-blind floor (best K)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_eurohpc/primary_campaign/steelman7b_contrasts.json")
    args = ap.parse_args()

    p3, p7 = Path(args.steelman_3b), Path(args.steelman_7b)
    report: dict = {"floor_K": args.floor_k, "contrasts": {}}

    codes = top_train_codes(Path(args.train), args.floor_k)
    report["floor_codes"] = codes
    report["arms"] = {}

    for arm in ("E11", "E14"):
        a = find_predictions(p3, arm)
        b = find_predictions(p7, arm)
        if not (a and b):
            print(f"  {arm}: missing steelman predictions (3B={a}, 7B={b}) — skipped")
            continue
        report["contrasts"][f"{arm}_3Bnote_vs_7Bnote"] = contrast(
            a, b, f"{arm}_3B_note", f"{arm}_7B_note", args.n_boot, args.n_perm, args.seed)
        # Full per-arm aggregates so Table 6's 7B row is generated rather than transcribed.
        report["arms"][f"{arm}_3B_note"] = arm_summary(a)
        report["arms"][f"{arm}_7B_note"] = arm_summary(b)
        print(f"  {arm}: 3B-note vs 7B-note done")

    b14 = find_predictions(p7, "E14")
    if b14:
        base = find_predictions(Path(args.baseline).parent, "E1") or find_predictions(Path(args.baseline), "E1")
        if base is None:
            bp = Path(args.baseline)
            base = bp / "test_predictions.jsonl" if (bp / "test_predictions.jsonl").exists() else None
        if base:
            report["contrasts"]["E14_7Bnote_vs_E1"] = contrast(
                base, b14, "E1_tfidf", "E14_7B_note", args.n_boot, args.n_perm, args.seed)
            print("  E14: 7B-note vs E1 done")
        else:
            print(f"  E14 vs E1: baseline predictions not found under {args.baseline}")

        floor_path = write_floor(PB.load(b14), codes,
                                 Path(args.out).parent / f"_floor_K{args.floor_k}_steelman_subset.jsonl")
        report["contrasts"]["E14_7Bnote_vs_note_blind_floor"] = contrast(
            floor_path, b14, f"note_blind_floor_K{args.floor_k}", "E14_7B_note",
            args.n_boot, args.n_perm, args.seed)
        print("  E14: 7B-note vs note-blind floor done")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    for k, v in report["contrasts"].items():
        d = v.get("delta_micro_f1", {})
        print(f"  {k}: Δ={d.get('point')} CI={d.get('ci95')} p={d.get('approx_randomization_p')} "
              f"{d.get('verdict','')}")


if __name__ == "__main__":
    main()
