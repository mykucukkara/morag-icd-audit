#!/usr/bin/env python3
"""
Round-1 revision analyses (all CPU, from stored predictions). Produces the numbers the
reviewers required:

  A. NOTE-BLIND FLOOR (E0): a constant predictor that never reads the note (the K most frequent
     training codes). Any system that does not beat this floor extracts no per-note signal.
     Reported at its own optimum and at the ladder's matched 15-code budget, with note-level
     paired bootstrap against each retrieval/RAG/full arm.

  B. LOSS DECOMPOSITION: for the hybrid-RAG shortlist, (i) the shortlist's gold coverage, which
     upper-bounds any downstream selector's recall; (ii) an ORACLE selector over that same
     shortlist at a matched code budget; (iii) a RANDOM-pruning null at the same budget; (iv) the
     actual full model. This replaces the unsupported "retrieval-bound recall" story with an
     accounting of where the F1 is lost.

  C. HONEST OPERATING-POINT TUNING: the campaign never tuned the LLM decision thresholds, so the
     headline contrast was tuned-classical vs untuned-LLM. Only test-split predictions exist for
     the LLM arms, so a validation-selected threshold is impossible post hoc; instead we do
     split-half tuning (select on one half of the notes, evaluate on the held-out half, both
     directions, averaged) and additionally report the test-optimal sweep as an explicit oracle
     upper bound. The split-half number is the honest one.

  D. LEAKAGE CHANCE BASELINE: the global evidence index holds one document per note, so the
     probability that a top-k retrieval returns the coded note's own document by chance is
     k/N. Reporting the measured cross-admission rate without this baseline overstates the
     finding; the defensible claim is about the absence of a mechanism, not a surprising rate.

PHI-safe: aggregate statistics only.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

N = lambda c: str(c or "").replace(".", "").strip().upper()


def load_notes(path, limit=None):
    out = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_preds(path):
    """global_sample_index -> (gold set, [ (code, confidence, supported, rank) ])"""
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
            codes = []
            for p in (r.get("predicted_codes") or []):
                c = N(p.get("code"))
                if not c:
                    continue
                try:
                    conf = float(p.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                codes.append((c, conf, bool(p.get("supported")), int(p.get("candidate_rank") or 0)))
            out[int(gsi) if gsi is not None else len(out)] = (gold, codes)
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def score_sets(items):
    """items: iterable of (pred set, gold set) -> micro P/R/F1 + codes per note"""
    tp = fp = fn = n = npred = 0
    for pred, gold in items:
        n += 1
        npred += len(pred)
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    p, r, f = prf(tp, fp, fn)
    return dict(n=n, precision=round(p, 4), recall=round(r, 4), micro_f1=round(f, 4),
                codes_per_note=round(npred / n, 2) if n else 0.0)


def paired_bootstrap(items_a, items_b, n_boot, seed):
    """items_*: list aligned by note of (pred,gold). Returns delta(b-a) + 95% CI + AR p."""
    idx = list(range(len(items_a)))

    def agg(which, sample):
        tp = fp = fn = 0
        for i in sample:
            pred, gold = which[i]
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
        return prf(tp, fp, fn)[2]

    obs = agg(items_b, idx) - agg(items_a, idx)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        deltas.append(agg(items_b, s) - agg(items_a, s))
    deltas.sort()
    lo, hi = deltas[int(0.025 * n_boot)], deltas[min(n_boot - 1, int(0.975 * n_boot))]
    # approximate randomization (swap per note)
    rng2 = random.Random(seed + 1)
    extreme = 0
    for _ in range(n_boot):
        L, R = [], []
        for i in idx:
            if rng2.random() < 0.5:
                L.append(items_a[i]); R.append(items_b[i])
            else:
                L.append(items_b[i]); R.append(items_a[i])
        d = agg(R, idx) - agg(L, idx)
        if abs(d) >= abs(obs):
            extreme += 1
    return dict(delta=round(obs, 4), ci95=[round(lo, 4), round(hi, 4)],
                p=round((extreme + 1) / (n_boot + 1), 4),
                significant=bool(lo > 0 or hi < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-root", default="data/splits_real")
    ap.add_argument("--campaign", default="results_eurohpc/primary_campaign/eval_input/top50")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = Path(args.splits_root) / f"top{args.top_n}"
    report = {"top_n": args.top_n, "n_boot": args.n_boot, "seed": args.seed}

    # ---------- A. note-blind floor ----------
    freq = Counter()
    for r in load_notes(sd / "train.jsonl"):
        for c in (r.get("gold_codes") or []):
            freq[N(c)] += 1
    test = load_notes(sd / "test.jsonl")
    golds = [{N(c) for c in (r.get("gold_codes") or [])} for r in test]
    floor = {}
    for K in (5, 10, 15, 20):
        S = {c for c, _ in freq.most_common(K)}
        floor[f"K={K}"] = score_sets([(S, g) for g in golds])
    report["A_note_blind_floor"] = {
        "description": "constant predictor: K most frequent TRAIN codes, note never read",
        "by_K": floor,
        "best_K": max(floor, key=lambda k: floor[k]["micro_f1"]),
    }

    # ---------- arms ----------
    ARMS = ["E4", "E5", "E6", "E9", "E10", "E11", "E12", "E13", "E14"]
    arms = {}
    for e in ARMS:
        p = Path(args.campaign) / f"{e}_seed42" / "test_predictions.jsonl"
        if p.exists():
            arms[e] = load_preds(p)

    # align floor vs each arm on shared notes, K=15 (matched budget) and best K
    S15 = {c for c, _ in freq.most_common(15)}
    Sbest = {c for c, _ in freq.most_common(int(report["A_note_blind_floor"]["best_K"].split("=")[1]))}
    cmp_floor = {}
    for e, pr in arms.items():
        keys = sorted(pr)
        arm_items = [({c for c, _, _, _ in pr[k][1]}, pr[k][0]) for k in keys]
        f15_items = [(S15, pr[k][0]) for k in keys]
        fb_items = [(Sbest, pr[k][0]) for k in keys]
        cmp_floor[e] = {
            "arm": score_sets(arm_items),
            "vs_floor_K15": paired_bootstrap(arm_items, f15_items, args.n_boot, args.seed),
            "vs_floor_bestK": paired_bootstrap(arm_items, fb_items, args.n_boot, args.seed),
        }
    report["A_arms_vs_floor"] = {
        "note": "delta = floor - arm; positive & significant => the note-blind floor BEATS the arm",
        "per_arm": cmp_floor,
    }

    # ---------- B. loss decomposition over the hybrid-RAG shortlist ----------
    if "E11" in arms and "E14" in arms:
        e11, e14 = arms["E11"], arms["E14"]
        shared = sorted(set(e11) & set(e14))
        e14_cpn = sum(len(e14[k][1]) for k in shared) / len(shared)
        budget = max(1, int(round(e14_cpn)))
        rng = random.Random(args.seed)
        oracle, rand, actual, full = [], [], [], []
        for k in shared:
            gold, codes = e11[k]
            cs = [c for c, _, _, _ in codes]
            # oracle: gold-first up to budget
            g_in = [c for c in cs if c in gold]
            o = set(g_in[:budget])
            if len(o) < budget:
                o |= set([c for c in cs if c not in gold][: budget - len(o)])
            oracle.append((o, gold))
            rand.append((set(rng.sample(cs, min(budget, len(cs)))), gold))
            actual.append((set(cs), gold))                       # E11 full shortlist (15)
            full.append(({c for c, _, _, _ in e14[k][1]}, gold))  # E14 actual
        report["B_loss_decomposition"] = {
            "shortlist_size_mean": round(sum(len(e11[k][1]) for k in shared) / len(shared), 2),
            "budget_used": budget,
            "shortlist_all_kept_E11": score_sets(actual),
            "oracle_selector_at_budget": score_sets(oracle),
            "random_pruning_null_at_budget": score_sets(rand),
            "actual_full_model_E14": score_sets(full),
            "interpretation": ("oracle - actual = F1 recoverable by a perfect selector on the SAME "
                               "shortlist; actual vs random = how far the LLM filter beats chance"),
        }

    # ---------- C. honest operating-point tuning (split-half) on E11 stored scores ----------
    if "E11" in arms:
        e11 = arms["E11"]
        keys = sorted(e11)
        half = len(keys) // 2
        halves = [keys[:half], keys[half:]]
        THR = [i / 20 for i in range(0, 20)]
        MAXC = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
        SUPP = [False, True]

        def apply_cfg(kset, thr, maxc, supp_only):
            items = []
            for k in kset:
                gold, codes = e11[k]
                cand = [(c, cf, sp) for c, cf, sp, _ in codes]
                if supp_only:
                    cand = [x for x in cand if x[2]]
                cand = [x for x in cand if x[1] >= thr]
                cand.sort(key=lambda x: -x[1])
                items.append(({c for c, _, _ in cand[:maxc]}, gold))
            return items

        def best_cfg(kset):
            best = None
            for thr in THR:
                for maxc in MAXC:
                    for so in SUPP:
                        f = score_sets(apply_cfg(kset, thr, maxc, so))["micro_f1"]
                        if best is None or f > best[0]:
                            best = (f, thr, maxc, so)
            return best

        folds = []
        for tune_i in (0, 1):
            f_t, thr, maxc, so = best_cfg(halves[tune_i])
            held = score_sets(apply_cfg(halves[1 - tune_i], thr, maxc, so))
            folds.append({"tuned_on_half": tune_i, "selected": {"threshold": thr, "max_codes": maxc,
                          "supported_only": so}, "tuning_half_f1": f_t, "heldout": held})
        honest = round(sum(f["heldout"]["micro_f1"] for f in folds) / len(folds), 4)
        f_all, thr_a, maxc_a, so_a = best_cfg(keys)
        report["C_operating_point"] = {
            "untuned_E11_as_run": score_sets([({c for c, _, _, _ in e11[k][1]}, e11[k][0]) for k in keys]),
            "split_half_folds": folds,
            "honest_tuned_micro_f1": honest,
            "test_optimal_oracle": {"micro_f1": f_all, "threshold": thr_a, "max_codes": maxc_a,
                                    "supported_only": so_a,
                                    "caveat": "selected ON test: an upper bound, NOT a tuned result"},
            "note": ("no validation-split LLM predictions exist, so a validation-selected threshold "
                     "is impossible post hoc; the split-half held-out mean is the honest estimate"),
        }

    # ---------- D. leakage chance baseline ----------
    n_train = sum(1 for _ in open(sd / "train.jsonl", encoding="utf-8", errors="ignore"))
    n_val = sum(1 for _ in open(sd / "validation.jsonl", encoding="utf-8", errors="ignore"))
    n_test = sum(1 for _ in open(sd / "test.jsonl", encoding="utf-8", errors="ignore"))
    n_docs = n_train + n_val + n_test
    top_k_ev = 5  # as used in scripts/38
    p_own_chance = min(1.0, top_k_ev / n_docs)
    report["D_leakage_chance"] = {
        "global_index_documents": n_docs,
        "documents_per_note": 1,
        "split_sizes": {"train": n_train, "validation": n_val, "test": n_test},
        "top_k_evidence": top_k_ev,
        "chance_prob_own_document_in_topk": round(p_own_chance, 8),
        "expected_cross_admission_rate_under_chance": round(1 - p_own_chance, 8),
        "measured_cross_admission_rate": 1.0,
        "interpretation": ("with one document per note in a %d-document index, chance alone predicts "
                           "a cross-admission rate of %.6f, so the measured 1.000 is NOT a surprising "
                           "rate. The defensible claim is structural: the global-corpus design contains "
                           "no mechanism that prefers the coded note's own text, so displayed grounding "
                           "is unrelated to the patient being coded." % (n_docs, 1 - p_own_chance)),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
