#!/usr/bin/env python3
"""
Script 52: the copy-instructed control (reviewer item T3-5).

A reviewer proposed that the low exact-quote compliance rate in §4.3 might be an instruction
artefact rather than a grounding failure: the primary prompt asks for "a short quote", never says
the field will be checked by exact string matching, and a model that paraphrases faithfully would
be scored as ungrounded. If that is the explanation, telling the model plainly that the field is
compared character-for-character should raise compliance.

This measures whether it does. The control arm differs from the primary arm in exactly one thing —
`copy_instructed: true`, which swaps `BATCH_CODE_SCORER_PROMPT` for the variant whose only change
is the `"q"` field instruction — so the comparison isolates the instruction.

Both rates are recomputed on the *same* notes. The control ran shards 0-3 of 68 (1,008 notes); the
primary arm ran all 17,151, so its published 0.09 is not the right comparator and using it would
confound the instruction with the note sample.

Nothing patient-identifying is read or written: only the boolean `evidence_quote_verbatim` already
stored per code prediction, and note keys are hashed before use.

Usage:
    python scripts/52_copy_instructed_control.py \
        --control results_eurohpc/copy_instructed/top50/E11_seed42/merged/test_predictions.jsonl \
        --primary results_eurohpc/primary_campaign/top50/E11_seed42/test_predictions.jsonl \
        --out results_eurohpc/copy_instructed/copy_instructed_control.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def note_key(rec: dict) -> str:
    """A stable, non-identifying handle for a note.

    `sample_id` is already a content hash and is the same value in both runs, so it is used when
    present; the subject/admission fallback is hashed so no identifier reaches the output.
    """
    sid = rec.get("sample_id")
    if sid:
        return str(sid)
    raw = f"{rec.get('subject_id')}|{rec.get('hadm_id')}|{rec.get('global_sample_index')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load(path: Path) -> dict[str, list[dict]]:
    """note key -> per-code prediction dicts (the `predicted_codes` entries carry the flag)."""
    out: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            codes = [c for c in (r.get("predicted_codes") or []) if isinstance(c, dict)]
            out[note_key(r)] = codes
    return out


def rate(per_note: dict[str, list[dict]], keys: set[str]) -> dict:
    checked = verbatim = codes = 0
    for k in keys:
        for cp in per_note.get(k, []):
            if cp.get("mock_llm", False):        # a mock copies the quote out; rate is 1.0 by design
                continue
            codes += 1
            v = cp.get("evidence_quote_verbatim")
            if v is None:
                continue
            checked += 1
            verbatim += bool(v)
    return {"notes": len(keys), "codes": codes, "quote_checked_codes": checked,
            "verbatim_codes": verbatim,
            "exact_quote_compliance": round(verbatim / checked, 4) if checked else None}


def per_note_counts(per_note: dict[str, list[dict]], keys: list[str]):
    """(checked, verbatim) per note, so the bootstrap resamples notes rather than codes."""
    out = []
    for k in keys:
        checked = verbatim = 0
        for cp in per_note.get(k, []):
            if cp.get("mock_llm", False):
                continue
            v = cp.get("evidence_quote_verbatim")
            if v is None:
                continue
            checked += 1
            verbatim += bool(v)
        out.append((checked, verbatim))
    return out


def paired_bootstrap(ctl_counts, pri_counts, resamples: int, seed: int) -> dict:
    """Note-level paired bootstrap on the compliance difference.

    The unit is the note, and each resample draws the SAME note indices for both arms, so the
    resampling respects the pairing the design creates. p is two-sided and floored at
    1/(resamples+1), which is the resolution of the test rather than a measured value.
    """
    import random

    def ratio(counts, idx):
        c = sum(counts[i][0] for i in idx)
        v = sum(counts[i][1] for i in idx)
        return v / c if c else None

    n = len(ctl_counts)
    all_idx = list(range(n))
    observed = ratio(ctl_counts, all_idx) - ratio(pri_counts, all_idx)
    rng = random.Random(seed)
    deltas, at_least_as_extreme = [], 0
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a, b = ratio(ctl_counts, idx), ratio(pri_counts, idx)
        if a is None or b is None:
            continue
        d = a - b
        deltas.append(d)
        # Null: the instruction makes no difference, so the centred difference should straddle zero.
        if abs(d - observed) >= abs(observed):
            at_least_as_extreme += 1
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))] if deltas else None
    hi = deltas[int(0.975 * len(deltas)) - 1] if deltas else None
    p = max((at_least_as_extreme + 1) / (resamples + 1), 1.0 / (resamples + 1))
    return {"observed_delta": round(observed, 4),
            "ci95": [round(lo, 4), round(hi, 4)] if deltas else None,
            "p_two_sided": round(p, 4), "resamples": resamples,
            "p_resolution_floor": round(1.0 / (resamples + 1), 5),
            "unit": "note"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_eurohpc/copy_instructed/copy_instructed_control.json")
    args = ap.parse_args()

    ctl = load(Path(args.control))
    pri = load(Path(args.primary))
    shared = set(ctl) & set(pri)
    print(f"  control notes {len(ctl):,} | primary notes {len(pri):,} | shared {len(shared):,}")
    if not shared:
        raise SystemExit("no shared notes — the two runs cannot be compared")

    report = {
        "arm": "E11",
        "difference": "copy_instructed prompt variant; all other settings identical",
        "shared_notes": len(shared),
        "control_copy_instructed": rate(ctl, shared),
        "primary_as_published_prompt": rate(pri, shared),
    }
    c = report["control_copy_instructed"]["exact_quote_compliance"]
    p = report["primary_as_published_prompt"]["exact_quote_compliance"]
    report["delta_control_minus_primary"] = round(c - p, 4) if (c is not None and p is not None) else None
    report["instruction_raised_compliance"] = bool(c is not None and p is not None and c > p)

    # §5.4 item 9 applies to this paper too: the difference gets a paired note-level test rather
    # than being reported as a bare point estimate.
    keys = sorted(shared)
    report["paired_test"] = paired_bootstrap(
        per_note_counts(ctl, keys), per_note_counts(pri, keys), args.resamples, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  copy-instructed  : {c}")
    print(f"  published prompt : {p}")
    print(f"  delta            : {report['delta_control_minus_primary']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
