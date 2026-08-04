#!/usr/bin/env python3
"""
Script 56: is the 1,008-note context x capacity subset like the rest of the test set?

The six cells of §4.5 were run on shards 0-3 of the 68-way split — the *first* 1,008 test notes, not
a random sample. Every contrast between cells is paired on those notes, so the interaction and the
simple effects are internally valid whatever the subset looks like. What the subset does affect is
generalisation: whether 0.426 is what the best configuration would score on the whole test set.

MIMIC row order is not random with respect to admission time, and note length, service and coding
density could travel with it. Rather than assert the subset is typical, this compares it with the
notes it excludes on the quantities that would matter — length, gold codes per note, and how much of
the label distribution it covers — so a reader can see the size of the assumption they are being
asked to make.

Nothing patient-identifying is read or written: note text is measured (length) and discarded, and
only aggregate statistics are stored.

Usage:
    python scripts/56_subset_representativeness.py \
        --test data/splits_real/top50/test.jsonl --subset-size 1008 \
        --out results_eurohpc/primary_campaign/subset_representativeness.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def norm(c: str) -> str:
    return str(c or "").replace(".", "").strip().upper()


def describe(rows: list[dict]) -> dict:
    lens = sorted(len(r["text"]) for r in rows)
    codes = [len(r["gold"]) for r in rows]
    n = len(rows)
    freq: Counter[str] = Counter()
    for r in rows:
        freq.update(r["gold"])
    return {
        "n_notes": n,
        "note_chars_median": lens[n // 2] if n else 0,
        "note_chars_mean": round(sum(lens) / n, 1) if n else 0,
        "gold_codes_per_note_mean": round(sum(codes) / n, 3) if n else 0,
        "distinct_codes_present": len(freq),
        "_freq": freq,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/splits_real/top50/test.jsonl")
    ap.add_argument("--subset-size", type=int, default=1008)
    ap.add_argument("--out", default="results_eurohpc/primary_campaign/subset_representativeness.json")
    args = ap.parse_args()

    rows = []
    with open(args.test, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"text": r.get("text") or "",
                         "gold": [norm(c) for c in (r.get("gold_codes") or [])]})

    subset, rest = rows[:args.subset_size], rows[args.subset_size:]
    a, b = describe(subset), describe(rest)
    fa, fb = a.pop("_freq"), b.pop("_freq")

    # How differently are the codes distributed? Total variation distance over the label set:
    # 0 means identical prevalence, 1 means disjoint. It answers "would a system tuned on one look
    # miscalibrated on the other" more directly than comparing top-code lists by eye.
    ta, tb = sum(fa.values()), sum(fb.values())
    tv = 0.5 * sum(abs(fa[c] / ta - fb[c] / tb) for c in set(fa) | set(fb)) if ta and tb else None

    report = {
        "test_file": str(args.test),
        "selection": f"first {args.subset_size} notes (shards 0-3 of 68), not a random sample",
        "subset": a,
        "remainder": b,
        "differences": {
            "note_chars_median": a["note_chars_median"] - b["note_chars_median"],
            "gold_codes_per_note_mean": round(a["gold_codes_per_note_mean"]
                                              - b["gold_codes_per_note_mean"], 3),
            "label_distribution_total_variation": round(tv, 4) if tv is not None else None,
        },
        "internal_validity_note": (
            "All §4.5 contrasts are paired on these notes, so the interaction and simple effects do "
            "not depend on the subset being typical; only the absolute level does."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  subset : {a}")
    print(f"  rest   : {b}")
    print(f"  diffs  : {report['differences']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
