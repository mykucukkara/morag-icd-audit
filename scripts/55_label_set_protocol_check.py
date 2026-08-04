#!/usr/bin/env python3
"""
Script 55: does the Top-50 label set change if it is derived from training data alone?

§3.1 discloses that the primary Top-50 vocabulary is the N most frequent codes in the working
corpus *as a whole*, not in the training split alone — so validation and test rows participate in
choosing which labels the benchmark contains. The Top-100/200 rebuild (`scripts/51`) uses the
training split only, which leaves the paper running two protocols and a reviewer asking, fairly,
which one the primary benchmark should follow.

The disclosure argues the choice is unlikely to matter because the ranking is dominated by codes
that are frequent everywhere. That is an argument, not a measurement. This measures it.

Three outcomes, and each implies something different:
  * identical sets -> the concern is real in principle and empty in fact; say so with the evidence.
  * same members, different order -> nothing downstream depends on rank, so still no re-run.
  * different members -> the primary campaign is scored on a vocabulary that test data helped
    select, and the affected arms have to be re-run. No amount of prose fixes that.

Usage:
    python scripts/55_label_set_protocol_check.py \
        --dataset data/processed_real/dataset.jsonl \
        --splits-root data/splits_real --top-n 50 \
        --out results_eurohpc/primary_campaign/label_set_protocol_check.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def norm(c: str) -> str:
    return str(c or "").replace(".", "").strip().upper()


def counts_from(path: Path, subject_filter: set[int] | None = None) -> Counter:
    freq: Counter[str] = Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if subject_filter is not None and int(r.get("subject_id", -1)) not in subject_filter:
                continue
            freq.update(norm(c) for c in (r.get("gold_codes") or []))
    return freq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed_real/dataset.jsonl")
    ap.add_argument("--splits-root", default="data/splits_real")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--out", default="results_eurohpc/primary_campaign/label_set_protocol_check.json")
    args = ap.parse_args()

    S = Path(args.splits_root) / f"top{args.top_n}"

    # (a) the set the campaign actually used
    shipped = [norm(c) for c in json.loads((S / "label_set.json").read_text(encoding="utf-8"))]

    # (b) what the training split alone would have chosen
    train_freq = counts_from(S / "train.jsonl")
    training_only = [c for c, _ in train_freq.most_common(args.top_n)]

    # (c) and, for completeness, the whole-corpus ranking §3.1 describes
    whole_freq = counts_from(Path(args.dataset)) if Path(args.dataset).exists() else None
    whole_corpus = [c for c, _ in whole_freq.most_common(args.top_n)] if whole_freq else None

    same_members = set(shipped) == set(training_only)
    same_order = shipped == training_only
    only_shipped = sorted(set(shipped) - set(training_only))
    only_training = sorted(set(training_only) - set(shipped))

    report = {
        "top_n": args.top_n,
        "shipped_label_set_size": len(shipped),
        "training_only_label_set_size": len(training_only),
        "identical_members": same_members,
        "identical_order": same_order,
        "in_shipped_not_in_training_only": only_shipped,
        "in_training_only_not_in_shipped": only_training,
        "n_differing_members": len(only_shipped),
        "verdict": ("identical" if same_order else
                    "same members, different rank order" if same_members else
                    "DIFFERENT MEMBERS — primary campaign vocabulary is test-informed"),
    }
    if whole_corpus is not None:
        report["whole_corpus_matches_shipped"] = set(whole_corpus) == set(shipped)

    # If members differ, say how much of the benchmark is at stake rather than leaving it abstract.
    if not same_members:
        test_freq = counts_from(S / "test.jsonl")
        affected = sum(test_freq[c] for c in only_shipped)
        total = sum(test_freq[c] for c in shipped)
        report["test_gold_instances_under_differing_codes"] = affected
        report["test_gold_instances_total"] = total
        report["share_of_test_gold_affected"] = round(affected / total, 4) if total else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  shipped vs training-only: {report['verdict']}")
    print(f"  members differing: {len(only_shipped)}")
    if only_shipped or only_training:
        print(f"    only in shipped      : {only_shipped}")
        print(f"    only in training-only: {only_training}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
