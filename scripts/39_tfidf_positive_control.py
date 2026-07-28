#!/usr/bin/env python3
"""
Positive control (A): a tuned classical TF-IDF + one-vs-rest logistic-regression coder under
the STANDARD threshold-based protocol, to show the evaluation harness is competently built.

The main campaign scores every arm at a fixed 15-code budget, which depresses absolute micro-F1
uniformly. Published MIMIC Top-50 numbers instead use per-label thresholding with a variable code
count. Here we fit the same TF-IDF+LR family used by E1, tune a decision threshold on the
validation split, and report test micro-F1 under (i) the fixed-15 protocol (= E1), (ii) a tuned
global threshold, and (iii) tuned per-label thresholds. If the tuned classical coder reaches the
published BoW/TF-IDF range (~0.55-0.62 micro-F1), the harness is competent and the LLM collapse is
genuine, not an artefact of a weak setup.

PHI-safe: reads note text only in memory for vectorization; emits only aggregate metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def _norm(c):
    return str(c or "").replace(".", "").strip().upper()


def load(path, label_to_idx, limit=None):
    texts, Y = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            texts.append(r.get("text", ""))
            row = np.zeros(len(label_to_idx), dtype=np.int8)
            for c in r.get("gold_codes", []):
                j = label_to_idx.get(_norm(c))
                if j is not None:
                    row[j] = 1
            Y.append(row)
    return texts, np.array(Y)


def micro_f1_from_pred(P, Y):
    tp = int((P & Y).sum())
    fp = int((P & (1 - Y)).sum())
    fn = int(((1 - P) & Y).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-root", default="data/splits_real")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--max-features", type=int, default=20000)
    ap.add_argument("--fixed-budget", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = Path(args.splits_root) / f"top{args.top_n}"
    labels = json.loads((sd / "label_set.json").read_text())
    labels = [_norm(c) for c in labels]
    l2i = {c: j for j, c in enumerate(labels)}

    print(f"loading splits (Top-{args.top_n}, {len(labels)} labels)...", flush=True)
    Xtr_txt, Ytr = load(sd / "train.jsonl", l2i)
    Xva_txt, Yva = load(sd / "validation.jsonl", l2i)
    Xte_txt, Yte = load(sd / "test.jsonl", l2i)
    print(f"  train={len(Xtr_txt)} val={len(Xva_txt)} test={len(Xte_txt)}", flush=True)

    vec = TfidfVectorizer(max_features=args.max_features, stop_words="english")
    Xtr = vec.fit_transform(Xtr_txt)
    Xva = vec.transform(Xva_txt)
    Xte = vec.transform(Xte_txt)
    print("fitting one-vs-rest logistic regression...", flush=True)
    clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=200), n_jobs=-1)
    clf.fit(Xtr, Ytr)

    Pva = clf.predict_proba(Xva)
    Pte = clf.predict_proba(Xte)

    # (i) fixed-15 budget (matches E1's protocol)
    def topk_pred(P, k):
        out = np.zeros_like(P, dtype=np.int8)
        idx = np.argsort(-P, axis=1)[:, :k]
        for r in range(P.shape[0]):
            out[r, idx[r]] = 1
        return out
    fixed = micro_f1_from_pred(topk_pred(Pte, args.fixed_budget), Yte)

    # (ii) tuned global threshold on validation
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.6, 56):
        _, _, f = micro_f1_from_pred((Pva >= t).astype(np.int8), Yva)
        if f > best_f:
            best_f, best_t = f, t
    glob = micro_f1_from_pred((Pte >= best_t).astype(np.int8), Yte)

    # (iii) tuned per-label thresholds on validation
    per_t = np.full(len(labels), 0.5)
    grid = np.linspace(0.05, 0.7, 66)
    for j in range(len(labels)):
        yv = Yva[:, j]
        if yv.sum() == 0:
            continue
        pv = Pva[:, j]
        bt, bf = 0.5, -1
        for t in grid:
            pred = (pv >= t).astype(np.int8)
            tp = int((pred & yv).sum()); fp = int((pred & (1 - yv)).sum()); fn = int(((1 - pred) & yv).sum())
            pr = tp / (tp + fp) if (tp + fp) else 0.0; rc = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            if f > bf:
                bf, bt = f, t
        per_t[j] = bt
    Pte_pl = (Pte >= per_t[None, :]).astype(np.int8)
    perlabel = micro_f1_from_pred(Pte_pl, Yte)
    avg_codes = float(Pte_pl.sum(axis=1).mean())

    rep = {
        "top_n": args.top_n, "max_features": args.max_features,
        "n_test": len(Xte_txt),
        "fixed_budget_%d" % args.fixed_budget: {"precision": round(fixed[0], 4), "recall": round(fixed[1], 4), "micro_f1": round(fixed[2], 4)},
        "tuned_global_threshold": {"threshold": round(float(best_t), 3), "precision": round(glob[0], 4), "recall": round(glob[1], 4), "micro_f1": round(glob[2], 4)},
        "tuned_per_label_threshold": {"precision": round(perlabel[0], 4), "recall": round(perlabel[1], 4), "micro_f1": round(perlabel[2], 4), "avg_codes_per_note": round(avg_codes, 2)},
    }
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
