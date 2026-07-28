#!/usr/bin/env python3
"""
Quantify cross-patient evidence leakage in the ORIGINAL global-corpus evidence design.

Paper 1's most original validity finding: before the note-local fix, "supporting evidence"
for a code was retrieved from a GLOBAL index of every admission's chunks (115k chunks). For
an ICD-code query, BM25 returns the globally best-matching chunk — which is almost never a
chunk of the note being coded. So the displayed "evidence" typically belonged to a DIFFERENT
patient, fabricating the appearance of grounding.

This script measures that directly: for a sample of test notes, query the global evidence
index with the note's gold-code descriptions and report, over the retrieved chunks, the
fraction whose hadm_id != the note's (cross-admission) and subject_id != the note's
(cross-patient). It also reports how often the note's OWN evidence appears in the top-k at
all. No LLM; retrieval only. PHI-safe: emits only aggregate rates + hashed ids never printed.

Usage:
  python scripts/38_measure_evidence_leakage.py --sample 1000 --top-k 5 \
    --evidence-index indexes_real/bm25/evidence_bm25_50.pkl \
    --test data/splits_real/top50/test.jsonl --icd-kb data/processed_real/icd_kb.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from morag_icd.retrieval.bm25_index import BM25


def load_jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _norm(c):
    return str(c or "").replace(".", "").strip().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-index", default="indexes_real/bm25/evidence_bm25_50.pkl")
    ap.add_argument("--test", default="data/splits_real/top50/test.jsonl")
    ap.add_argument("--icd-kb", default="data/processed_real/icd_kb.jsonl")
    ap.add_argument("--sample", type=int, default=1000, help="notes to sample (deterministic prefix)")
    ap.add_argument("--top-k", type=int, default=5, help="evidence chunks retrieved per code query")
    ap.add_argument("--max-codes", type=int, default=10, help="cap gold codes queried per note")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"loading evidence index {args.evidence_index} ...", flush=True)
    idx = BM25.load(args.evidence_index)
    docs = idx.docs
    print(f"  n_chunks={len(docs)}", flush=True)

    kb = {_norm(e.get("code")): (e.get("title") or e.get("long_title") or "")
          for e in load_jsonl(args.icd_kb)}
    notes = load_jsonl(args.test, limit=args.sample)
    print(f"  notes sampled={len(notes)}  top_k={args.top_k}", flush=True)

    retrieved_chunks = 0
    cross_admission = 0          # chunk.hadm_id != note.hadm_id
    cross_patient = 0            # chunk.subject_id != note.subject_id
    queries = 0
    notes_with_any_own = 0       # note had >=1 own-admission chunk retrieved across its codes
    notes_evaluated = 0

    for note in notes:
        nh = str(note.get("hadm_id"))
        ns = str(note.get("subject_id"))
        gold = [c for c in (note.get("gold_codes") or [])][: args.max_codes]
        if not gold:
            continue
        notes_evaluated += 1
        own_hit = False
        for code in gold:
            title = kb.get(_norm(code), "")
            q = f"ICD-10 {code}: {title}".strip()
            scores = idx.get_scores(q)
            if scores is None or len(scores) == 0:
                continue
            order = scores.argsort()[::-1][: args.top_k]
            queries += 1
            for di in order:
                d = docs[int(di)]
                retrieved_chunks += 1
                same_adm = str(d.get("hadm_id")) == nh
                if not same_adm:
                    cross_admission += 1
                else:
                    own_hit = True
                if str(d.get("subject_id")) != ns:
                    cross_patient += 1
        if own_hit:
            notes_with_any_own += 1

    rep = {
        "notes_evaluated": notes_evaluated,
        "code_queries": queries,
        "retrieved_chunks": retrieved_chunks,
        "top_k": args.top_k,
        # THE headline leakage numbers:
        "cross_admission_rate": round(cross_admission / retrieved_chunks, 4) if retrieved_chunks else None,
        "cross_patient_rate": round(cross_patient / retrieved_chunks, 4) if retrieved_chunks else None,
        "same_admission_rate": round(1 - cross_admission / retrieved_chunks, 4) if retrieved_chunks else None,
        # of all notes, how many ever saw a single chunk of their OWN admission in top-k:
        "notes_with_any_own_evidence_rate": round(notes_with_any_own / notes_evaluated, 4) if notes_evaluated else None,
    }
    text = json.dumps(rep, indent=2)
    print("=== GLOBAL-CORPUS EVIDENCE LEAKAGE ===")
    print(text)
    print(f"\nInterpretation: {rep['cross_admission_rate']:.1%} of retrieved 'supporting evidence' "
          f"chunks came from a DIFFERENT admission than the note being coded.")
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
