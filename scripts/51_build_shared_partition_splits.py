#!/usr/bin/env python3
"""
Script 51: rebuild the Top-100/200 splits on the Top-50 subject partition (reviewer item T3-7).

The splits shipped with this study were built independently per label space: notes with no
in-vocabulary code are dropped, and the subject permutation is then recomputed on whatever
survives. The consequence, found while answering a reviewer's question about differing test-set
sizes, is that the three test sets are different patients — Top-50 and Top-200 share only 1,417 of
roughly 9,200 and 9,500 subjects. Comparing systems across label spaces therefore varied the
cohort as well as the vocabulary.

This rebuilds Top-100 and Top-200 so that every subject keeps the split it was given at Top-50.
Subjects that appear only in the larger label spaces (their notes carried no Top-50 code) are
assigned by the same seeded permutation, so the added population is still subject-disjoint across
splits. The result: one partition, three vocabularies, and a scalability comparison that varies
only the thing it claims to vary.

Label sets are still derived per label space, and — unlike the original build — from the training
split alone, which removes the second defect disclosed in §3.1.

Usage:
    python scripts/51_build_shared_partition_splits.py \
        --dataset data/processed_real/dataset.jsonl \
        --reference-split data/splits_real/top50 \
        --out-root data/splits_shared --top-n 100 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def norm(c: str) -> str:
    return str(c or "").replace(".", "").strip().upper()


def read_reference(ref: Path) -> dict[int, str]:
    """subject_id -> split, taken from the reference label space."""
    out: dict[int, str] = {}
    for name in ("train", "validation", "test"):
        p = ref / f"{name}.jsonl"
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                sid = r.get("subject_id")
                if sid is not None:
                    out[int(sid)] = name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed_real/dataset.jsonl")
    ap.add_argument("--reference-split", default="data/splits_real/top50")
    ap.add_argument("--out-root", default="data/splits_shared")
    ap.add_argument("--top-n", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ref = read_reference(Path(args.reference_split))
    print(f"  reference partition: {len(ref):,} subjects "
          f"({sum(v == 'train' for v in ref.values()):,} train / "
          f"{sum(v == 'validation' for v in ref.values()):,} val / "
          f"{sum(v == 'test' for v in ref.values()):,} test)")

    rows = []
    with open(args.dataset, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"subject_id": int(r["subject_id"]), "hadm_id": r.get("hadm_id"),
                         "text": r.get("text") or "",
                         "gold_codes": [norm(c) for c in (r.get("gold_codes") or [])]})
    print(f"  working corpus: {len(rows):,} notes, {len({r['subject_id'] for r in rows}):,} subjects")

    # Subjects absent from the reference get a split from the same seeded permutation, so the
    # added population is partitioned once and identically for every label space.
    extra = sorted({r["subject_id"] for r in rows} - set(ref))
    rng = random.Random(args.seed)
    rng.shuffle(extra)
    n_tr = int(round(len(extra) * args.ratios[0]))
    n_va = int(round(len(extra) * args.ratios[1]))
    for i, sid in enumerate(extra):
        ref[sid] = "train" if i < n_tr else ("validation" if i < n_tr + n_va else "test")
    if extra:
        print(f"  subjects not in the reference: {len(extra):,} (assigned by seeded permutation)")

    for tn in args.top_n:
        # Label set from the TRAINING split only — the original build ranked over the whole corpus.
        freq: Counter[str] = Counter()
        for r in rows:
            if ref.get(r["subject_id"]) == "train":
                freq.update(r["gold_codes"])
        label_set = [c for c, _ in freq.most_common(tn)]
        keep = set(label_set)

        out = Path(args.out_root) / f"top{tn}"
        out.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        handles = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8")
                   for s in ("train", "validation", "test")}
        try:
            for r in rows:
                gold = [c for c in r["gold_codes"] if c in keep]
                if not gold:                       # same filter as the original build
                    continue
                split = ref[r["subject_id"]]
                handles[split].write(json.dumps({
                    "subject_id": r["subject_id"], "hadm_id": r["hadm_id"],
                    "text": r["text"], "gold_codes": gold, "split": split}) + "\n")
                counts[split] += 1
        finally:
            for h in handles.values():
                h.close()

        (out / "label_set.json").write_text(json.dumps(label_set, indent=1), encoding="utf-8")
        summary = {"top_n": tn, "rows": sum(counts.values()), "split_counts": dict(counts),
                   "unique_code_count": len(label_set),
                   "label_set_derived_from": "training split only",
                   "partition": f"shared with {args.reference_split}",
                   "dataset_path": "<local path removed>"}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  Top-{tn}: {sum(counts.values()):,} notes  {dict(counts)}")

    # Confirm the thing this script exists to guarantee.
    print("\n  subject overlap between test sets (should be total containment):")
    sets = {}
    for tn in args.top_n:
        s = set()
        with open(Path(args.out_root) / f"top{tn}" / "test.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    s.add(json.loads(line)["subject_id"])
        sets[tn] = s
    ref_test = {s for s, v in ref.items() if v == "test"}
    for tn, s in sets.items():
        leak = len(s - ref_test)
        print(f"    Top-{tn}: {len(s):,} test subjects, {leak} outside the reference test split "
              f"{'(OK)' if leak == 0 else '(PROBLEM)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
