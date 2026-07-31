#!/usr/bin/env python3
"""
Script 49: the like-for-like supervised control (reviewer item T3-2).

Trains a label-wise attention coder on the same split as every other arm and scores it under the
same fixed-budget protocol, so the number lands on the paper's own ladder rather than beside it.

Two protocols are reported because they answer different questions:
  * fixed 15-code budget — comparable to E1..E14 in Table 1;
  * tuned global threshold — comparable to the published-protocol positive control (§4.1a).

The encoder is the general-domain one already used for E3. No clinically pretrained weights are
involved, so the result is a lower bound on the label-attention family rather than a reproduction
of PLM-ICD.

Usage:
    python scripts/49_label_attention_control.py \
        --splits-root data/splits_real --top-n 50 \
        --model ${MODELS_ROOT}/classifier_checkpoints/bert-base-uncased_safetensors \
        --epochs 5 --max-length 512 \
        --out results_eurohpc/primary_campaign/label_attention_control.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morag_icd.baselines.label_attention_classifier import (  # noqa: E402
    LabelAttentionClassifier, LabelAttentionConfig,
)


def norm(c: str) -> str:
    return str(c or "").replace(".", "").strip().upper()


def load(path: Path, limit: int | None = None):
    texts, golds = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            texts.append(r.get("text") or "")
            golds.append([norm(c) for c in (r.get("gold_codes") or [])])
            if limit and len(texts) >= limit:
                break
    return texts, golds


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {"micro_f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
            "precision": round(p, 4), "recall": round(r, 4)}


def score(pred_sets, gold_sets) -> dict:
    tp = fp = fn = codes = 0
    for p, g in zip(pred_sets, gold_sets):
        p, g = set(p), set(g)
        tp += len(p & g); fp += len(p - g); fn += len(g - p); codes += len(p)
    out = prf(tp, fp, fn)
    out["n"] = len(gold_sets)
    out["codes_per_note"] = round(codes / max(1, len(gold_sets)), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-root", default="data/splits_real")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-limit", type=int, default=0, help="0 = all; smaller for a smoke run")
    ap.add_argument("--out", default="results_eurohpc/primary_campaign/label_attention_control.json")
    args = ap.parse_args()

    S = Path(args.splits_root) / f"top{args.top_n}"
    tr_x, tr_y = load(S / "train.jsonl", args.train_limit or None)
    va_x, va_y = load(S / "validation.jsonl")
    te_x, te_y = load(S / "test.jsonl")
    print(f"  train {len(tr_x):,} | validation {len(va_x):,} | test {len(te_x):,}")

    # The label set is the ladder's, derived from training frequency so the control is scored
    # against exactly the vocabulary every other arm is scored against.
    freq: Counter[str] = Counter()
    for g in tr_y:
        freq.update(g)
    label_set = [c for c, _ in freq.most_common(args.top_n)]

    cfg = LabelAttentionConfig(
        model_path=args.model, num_labels=len(label_set), max_length=args.max_length,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, head_lr=args.head_lr,
        seed=args.seed, label_set=label_set,
    )
    clf = LabelAttentionClassifier(cfg).fit(tr_x, tr_y)

    report = {
        "arm": "E3b_label_attention",
        "encoder": args.model.split("/")[-1],
        "clinically_pretrained": False,
        "max_length": args.max_length, "epochs": args.epochs, "seed": args.seed,
        "n_labels": len(label_set),
    }

    # (1) the ladder's protocol
    report["fixed_budget"] = score(clf.predict_topk(te_x, k=args.budget), te_y)
    report["fixed_budget"]["budget"] = args.budget
    print(f"  fixed-{args.budget} budget: {report['fixed_budget']}")

    # (2) the published protocol, with the threshold chosen on validation and applied once to test
    va_scores = clf.predict_scores(va_x)
    te_scores = clf.predict_scores(te_x)
    best_t, best_f1 = 0.5, -1.0
    for i in range(1, 60):
        t = i / 100
        pred = [[label_set[j] for j in range(len(label_set)) if row[j] >= t] for row in va_scores]
        f1 = score(pred, va_y)["micro_f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    te_pred = [[label_set[j] for j in range(len(label_set)) if row[j] >= best_t] for row in te_scores]
    report["tuned_threshold"] = score(te_pred, te_y)
    report["tuned_threshold"].update({"threshold": best_t, "selected_on": "validation",
                                      "validation_micro_f1": round(best_f1, 4)})
    print(f"  tuned threshold {best_t}: {report['tuned_threshold']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
