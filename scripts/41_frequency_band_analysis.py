#!/usr/bin/env python3
"""
Per-frequency-band performance on the existing Top-200 runs (CPU, no new inference).

Reviewers objected that the study evaluates only frequent-code subsets — the regime where
discriminative supervised methods are strongest and where RAG/LLM approaches are least likely to
show an advantage. The Top-200 label space already spans a substantial frequency gradient, so we
can test the objection with data in hand: split the 200 codes into frequency bands (by training
prevalence) and report per-band micro-F1 for the classical baseline, retrieval, RAG, and the full
model. If the LLM arms close the gap on the rarest band, the "wrong regime" objection is
substantiated; if the gap persists or widens, the finding generalizes across the available
frequency range.

Metrics are computed per band by restricting BOTH predictions and gold to that band's codes.
PHI-safe: aggregate only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

N = lambda c: str(c or "").replace(".", "").strip().upper()


def load_gold_counts(path):
    freq = Counter()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                for c in (json.loads(line).get("gold_codes") or []):
                    freq[N(c)] += 1
    return freq


def load_preds(path):
    out = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "success":
                continue
            gsi = r.get("global_sample_index")
            gold = {N(c) for c in (r.get("gold_codes") or [])}
            pred = {N(p.get("code")) for p in (r.get("predicted_codes") or [])} - {""}
            out[int(gsi) if gsi is not None else len(out)] = (pred, gold)
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def band_score(preds, band):
    tp = fp = fn = 0
    for pred, gold in preds.values():
        p, g = pred & band, gold & band
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    P, R, F = prf(tp, fp, fn)
    return dict(micro_f1=round(F, 4), precision=round(P, 4), recall=round(R, 4),
                gold_instances=tp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-root", default="data/splits_real")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--scalability-root", default="results_eurohpc/scalability")
    ap.add_argument("--arms", default="E1,E6,E11,E14")
    ap.add_argument("--n-bands", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = Path(args.splits_root) / f"top{args.top_n}"
    freq = load_gold_counts(sd / "train.jsonl")
    labels = [N(c) for c in json.loads((sd / "label_set.json").read_text())]
    ranked = sorted(labels, key=lambda c: -freq.get(c, 0))          # most frequent first
    size = max(1, len(ranked) // args.n_bands)
    bands = []
    for b in range(args.n_bands):
        lo = b * size
        hi = len(ranked) if b == args.n_bands - 1 else (b + 1) * size
        codes = set(ranked[lo:hi])
        counts = [freq.get(c, 0) for c in codes]
        bands.append({"band": b + 1, "rank_range": [lo + 1, hi], "n_codes": len(codes),
                      "train_freq_min": min(counts), "train_freq_max": max(counts),
                      "codes": codes})

    report = {"top_n": args.top_n, "n_bands": args.n_bands,
              "band_definition": [{k: v for k, v in b.items() if k != "codes"} for b in bands],
              "per_arm": {}}

    for arm in args.arms.split(","):
        base = Path(args.scalability_root) / f"top{args.top_n}" / f"{arm}_seed42"
        p = base / "merged" / "test_predictions.jsonl"
        if not p.exists():
            p = base / "test_predictions.jsonl"
        if not p.exists():
            report["per_arm"][arm] = {"error": "missing"}
            continue
        preds = load_preds(p)
        report["per_arm"][arm] = {f"band{b['band']}": band_score(preds, b["codes"]) for b in bands}

    # gap vs the classical baseline, per band
    if "E1" in report["per_arm"] and "error" not in report["per_arm"]["E1"]:
        gaps = {}
        for arm in report["per_arm"]:
            if arm == "E1" or "error" in report["per_arm"][arm]:
                continue
            gaps[arm] = {b: round(report["per_arm"]["E1"][b]["micro_f1"]
                                  - report["per_arm"][arm][b]["micro_f1"], 4)
                         for b in report["per_arm"][arm]}
        report["gap_vs_E1_per_band"] = gaps
        report["interpretation"] = ("if the gap to E1 shrinks toward the rarest band, the "
                                    "'evaluated only in the supervised-friendly regime' objection "
                                    "is substantiated; if it holds or widens, the finding spans "
                                    "the available frequency range")

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
