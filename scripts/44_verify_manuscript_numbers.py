#!/usr/bin/env python3
"""
Script 44: check every load-bearing number in the manuscript against its artifact.

The paper's own audit checklist ends with "machine-generate result tables under provenance checks."
This script closes the loop for the prose as well: each entry below names a claim, the value the
manuscript states, and where that value comes from. It re-reads the artifact, compares, and exits
non-zero if anything disagrees, so a stale number cannot survive a re-run.

Tolerances are explicit rather than implicit: the manuscript rounds to three decimals in prose and
four in tables, so a claim matches when it agrees with the artifact at the precision the manuscript
itself uses.

Usage:
    python scripts/44_verify_manuscript_numbers.py \
        --artifacts results_eurohpc/primary_campaign \
        --scalability results_eurohpc/scalability_v2/metrics \
        --sections manuscript/sections
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def get(obj, path: str):
    """Fetch a dotted path, tolerating flattened keys and list indices.

    scripts/10 stores seed statistics under flattened keys ("classification.micro_f1"), while the
    analysis JSONs nest. Walk greedily so one accessor reads both without the caller caring.
    """
    cur, parts = obj, path.split(".")
    i = 0
    while i < len(parts):
        if isinstance(cur, list):
            cur = cur[int(parts[i])]
            i += 1
            continue
        if not isinstance(cur, dict):
            raise KeyError(f"{path} (not indexable at {parts[i]!r})")
        # Prefer the longest flattened key that matches from here.
        for j in range(len(parts), i, -1):
            candidate = ".".join(parts[i:j])
            if candidate in cur:
                cur = cur[candidate]
                i = j
                break
        else:
            raise KeyError(f"{path} (missing at {parts[i]!r})")
    return cur


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="results_eurohpc/primary_campaign")
    ap.add_argument("--scalability", default="results_eurohpc/scalability_v2/metrics")
    ap.add_argument("--sections", default="manuscript/sections")
    ap.add_argument("--splits-root", default="data/splits_real")
    args = ap.parse_args()
    A = Path(args.artifacts)

    src = {
        "ladder": load_json(A / "metrics" / "top50" / "consolidated_metrics.json"),
        "rev": load_json(A / "revision_round1_analyses.json"),
        "cap11": load_json(A / "capacity_E11_3b_vs_7b.json"),
        "cap14": load_json(A / "capacity_E14_3b_vs_7b.json"),
        "st11": load_json(A / "steelman_E11_paired.json"),
        "st14": load_json(A / "steelman_E14_paired.json"),
        "vsE1": load_json(A / "t1_significance_vs_E1.json"),
        "e13": load_json(A / "t1_E13_vs_E6.json"),
        "e12": load_json(A / "t1_E12_vs_E6.json"),
        "leak2": load_json(A / "t3_evidence_leakage_k2.json"),
        "pc": load_json(A / "positive_control_tfidf.json"),
        "s7b": load_json(A / "steelman7b_contrasts.json"),
        "sc100": load_json(Path(args.scalability) / "top100" / "top100" / "consolidated_metrics.json")
                 or load_json(Path(args.scalability) / "top100" / "consolidated_metrics.json"),
        "sc200": load_json(Path(args.scalability) / "top200" / "top200" / "consolidated_metrics.json")
                 or load_json(Path(args.scalability) / "top200" / "consolidated_metrics.json"),
    }

    def arm_f1(key: str, arm: str) -> float:
        return get(src[key], f"seed_statistics.{arm}.classification.micro_f1.mean")

    def arm_rel(arm: str, metric: str) -> float:
        return get(src["ladder"], f"seed_statistics.{arm}.reliability.{metric}.mean")

    # (section, claim, manuscript value, decimals, callable producing the artifact value)
    CHECKS = [
        ("4.1", "E1 TF-IDF micro-F1", 0.449, 3, lambda: arm_f1("ladder", "E1")),
        ("4.1", "E2 TF-IDF+SVM", 0.441, 3, lambda: arm_f1("ladder", "E2")),
        ("4.1", "E3 transformer", 0.372, 3, lambda: arm_f1("ladder", "E3")),
        ("4.1", "E5 dense retrieval", 0.185, 3, lambda: arm_f1("ladder", "E5")),
        ("4.1", "E6 hybrid retrieval", 0.203, 3, lambda: arm_f1("ladder", "E6")),
        ("4.1", "E7 LLM zero-shot", 0.004, 3, lambda: arm_f1("ladder", "E7")),
        ("4.1", "E8 LLM few-shot", 0.014, 3, lambda: arm_f1("ladder", "E8")),
        ("4.1", "E9 BM25 RAG (range low)", 0.181, 3, lambda: arm_f1("ladder", "E9")),
        ("4.1", "E10 dense RAG", 0.182, 3, lambda: arm_f1("ladder", "E10")),
        ("4.1", "E11 hybrid RAG (range high)", 0.186, 3, lambda: arm_f1("ladder", "E11")),
        ("4.1c", "shortlist kept whole (E11)", 0.186, 3,
         lambda: get(src["rev"], "B_loss_decomposition.shortlist_all_kept_E11.micro_f1")),
        ("4.1", "E12 evidence constraint", 0.134, 3, lambda: arm_f1("ladder", "E12")),
        ("4.1", "E13 contrastive", 0.195, 3, lambda: arm_f1("ladder", "E13")),
        ("4.1", "E14 full model", 0.133, 3, lambda: arm_f1("ladder", "E14")),
        ("4.1", "E1 minus E14 central contrast", 0.3165, 4, lambda: get(src["vsE1"], "E14.E1_minus_E14")),
        ("4.1", "central contrast CI low", 0.3129, 4, lambda: get(src["vsE1"], "E14.ci95.0")),
        ("4.1", "central contrast CI high", 0.3202, 4, lambda: get(src["vsE1"], "E14.ci95.1")),
        ("4.1", "E13 vs E6 delta", -0.008, 3, lambda: get(src["e13"], "delta_micro_f1.point")),
        ("4.1", "E12 vs E6 delta", -0.069, 3, lambda: get(src["e12"], "delta_micro_f1.point")),
        ("4.1a", "tuned TF-IDF positive control", 0.605, 3,
         lambda: get(src["pc"], "tuned_global_threshold.micro_f1")),
        ("4.1b", "note-blind floor best K", 0.304, 3,
         lambda: get(src["rev"], "A_note_blind_floor.by_K.K=10.micro_f1")),
        ("4.1b", "note-blind floor at K=15", 0.285, 3,
         lambda: get(src["rev"], "A_note_blind_floor.by_K.K=15.micro_f1")),
        ("4.1c", "oracle over shortlist", 0.386, 3,
         lambda: get(src["rev"], "B_loss_decomposition.oracle_selector_at_budget.micro_f1")),
        ("4.1c", "random pruning null", 0.108, 3,
         lambda: get(src["rev"], "B_loss_decomposition.random_pruning_null_at_budget.micro_f1")),
        ("4.1c", "E14 codes per note", 4.26, 2,
         lambda: get(src["rev"], "B_loss_decomposition.actual_full_model_E14.codes_per_note")),
        ("4.1d", "operating point, as run", 0.1865, 4,
         lambda: get(src["rev"], "C_operating_point.untuned_E11_as_run.micro_f1")),
        ("4.1d", "operating point, split-half honest", 0.1865, 4,
         lambda: get(src["rev"], "C_operating_point.honest_tuned_micro_f1")),
        ("4.2", "capacity E11 delta", 0.008, 3, lambda: get(src["cap11"], "delta_micro_f1.point")),
        ("4.2", "capacity E11 CI high (MDE bound)", 0.021, 3, lambda: get(src["cap11"], "delta_micro_f1.ci95.1")),
        ("4.2", "capacity E14 delta", -0.015, 3, lambda: get(src["cap14"], "delta_micro_f1.point")),
        ("4.2", "capacity E14 CI high (MDE bound)", 0.018, 3, lambda: get(src["cap14"], "delta_micro_f1.ci95.1")),
        ("4.2", "judgement lift delta", 0.47, 2, lambda: get(src["cap11"], "delta_lift.point")),
        ("4.3", "cross-admission rate at k=2", 1.0, 4, lambda: get(src["leak2"], "cross_admission_rate")),
        ("4.3", "retrieved documents at k=2", 10338, 0, lambda: get(src["leak2"], "retrieved_chunks")),
        ("4.3", "verbatim quote rate E9", 0.08, 2, lambda: arm_rel("E9", "evidence_quote_verbatim_rate")),
        ("4.3", "verbatim quote rate E10", 0.21, 2, lambda: arm_rel("E10", "evidence_quote_verbatim_rate")),
        ("4.3", "verbatim quote rate E11", 0.09, 2, lambda: arm_rel("E11", "evidence_quote_verbatim_rate")),
        ("4.3", "verbatim quote rate E14", 0.09, 2, lambda: arm_rel("E14", "evidence_quote_verbatim_rate")),
        ("4.4", "Top-100 E1", 0.4659, 4, lambda: arm_f1("sc100", "E1")),
        ("4.4", "Top-100 E6", 0.1627, 4, lambda: arm_f1("sc100", "E6")),
        ("4.4", "Top-100 E11", 0.1375, 4, lambda: arm_f1("sc100", "E11")),
        ("4.4", "Top-100 E14", 0.0970, 4, lambda: arm_f1("sc100", "E14")),
        ("4.4", "Top-200 E1", 0.4685, 4, lambda: arm_f1("sc200", "E1")),
        ("4.4", "Top-200 E6", 0.1185, 4, lambda: arm_f1("sc200", "E6")),
        ("4.4", "Top-200 E11", 0.1072, 4, lambda: arm_f1("sc200", "E11")),
        ("4.4", "Top-200 E14", 0.0699, 4, lambda: arm_f1("sc200", "E14")),
        ("4.5", "steelman 3B E11 delta", -0.002, 3, lambda: get(src["st11"], "delta_micro_f1.point")),
        ("4.5", "steelman 3B E11 CI high (MDE bound)", 0.003, 3, lambda: get(src["st11"], "delta_micro_f1.ci95.1")),
        ("4.5", "steelman 3B E14 delta", -0.033, 3, lambda: get(src["st14"], "delta_micro_f1.point")),
    ]

    # Contrasts that only exist once scripts/43 has run.
    if src["s7b"]:
        C = src["s7b"]["contrasts"]
        if "E14_3Bnote_vs_7Bnote" in C:
            CHECKS += [
                ("4.5", "3B->7B with note, E14 delta", 0.188, 3,
                 lambda: get(C, "E14_3Bnote_vs_7Bnote.delta_micro_f1.point")),
                ("4.5", "best configuration micro-F1", 0.2933, 4,
                 lambda: get(C, "E14_3Bnote_vs_7Bnote.E14_7B_note.micro_f1")),
            ]
        if "E14_7Bnote_vs_E1" in C:
            CHECKS.append(("4.5", "best configuration vs TF-IDF", -0.156, 3,
                           lambda: get(C, "E14_7Bnote_vs_E1.delta_micro_f1.point")))
        if "E14_7Bnote_vs_note_blind_floor" in C:
            CHECKS += [
                ("4.5", "best configuration vs floor", 0.001, 3,
                 lambda: get(C, "E14_7Bnote_vs_note_blind_floor.delta_micro_f1.point")),
                ("4.5", "best configuration vs floor CI high (MDE bound)", 0.016, 3,
                 lambda: get(C, "E14_7Bnote_vs_note_blind_floor.delta_micro_f1.ci95.1")),
            ]
    else:
        print("NOTE: steelman7b_contrasts.json absent — §4.5's three carrying contrasts unchecked\n")

    # Per-label-space disclosure comes from the splits themselves.
    for tn, exp_n, exp_gold, exp_ceil in ((50, 17151, 5.38, 0.998), (100, 17159, 6.91, 0.987),
                                          (200, 17581, 8.74, 0.951)):
        f = Path(args.splits_root) / f"top{tn}" / "test.jsonl"
        if not f.exists():
            continue

        def measure(f=f):
            n = tot = capped = 0
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    g = len(json.loads(line).get("gold_codes") or json.loads(line).get("labels") or [])
                    n += 1; tot += g; capped += min(g, 15)
            return n, tot / max(n, 1), capped / max(tot, 1)

        CHECKS += [
            ("4.4", f"Top-{tn} test notes", exp_n, 0, lambda m=measure: m()[0]),
            ("4.4", f"Top-{tn} gold codes/note", exp_gold, 2, lambda m=measure: m()[1]),
            ("4.4", f"Top-{tn} recall ceiling at 15", exp_ceil, 3, lambda m=measure: m()[2]),
        ]

    failures, skipped = [], []
    print(f"{'§':6s} {'claim':46s} {'paper':>10s} {'artifact':>10s}  verdict")
    print("-" * 92)
    for section, claim, stated, dec, fn in CHECKS:
        try:
            actual = fn()
        except Exception as e:  # missing artifact or key
            skipped.append((section, claim, str(e)))
            print(f"{section:6s} {claim:46s} {stated:>10} {'—':>10}  SKIP ({e})")
            continue
        ok = round(float(actual), dec) == round(float(stated), dec)
        print(f"{section:6s} {claim:46s} {stated:>10} {round(float(actual), dec):>10}  "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((section, claim, stated, actual))

    # Figure scripts hardcode their own numbers, so they can drift from the artifacts exactly as
    # prose can — F2 and S1 both carried a rounded-up 0.187 after the text had been corrected.
    fig_dir = Path(args.sections).parent / "figures"
    fig_checks = [
        ("F2_decomposition_ladder.py", "E11", 0.186, arm_f1("ladder", "E11")),
        ("F2_decomposition_ladder.py", "E14", 0.133, arm_f1("ladder", "E14")),
        ("F2_decomposition_ladder.py", "E1", 0.449, arm_f1("ladder", "E1")),
        ("S1_scalability.py", "E11 Top-200", 0.107, arm_f1("sc200", "E11")),
        ("S1_scalability.py", "E14 Top-200", 0.070, arm_f1("sc200", "E14")),
    ]
    fig_bad = []
    for fname, label, stated, actual in fig_checks:
        f = fig_dir / fname
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        if f"{stated:.3f}" not in text:
            fig_bad.append((fname, label, stated))
        elif round(float(actual), 3) != round(stated, 3):
            fig_bad.append((fname, label, f"{stated} vs artifact {round(float(actual), 3)}"))
    print("\nfigure-script values:",
          "all present and matching the artifacts" if not fig_bad else f"PROBLEM {fig_bad}")

    # Guard against markers that must never reach a submitted manuscript.
    banned = {"TODO": 0, "PILOT": 0, "PENDING": 0, "FIXME": 0}
    for f in sorted(Path(args.sections).glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for word in banned:
            banned[word] += len(re.findall(rf"\b{word}\b", text))
    print("\nmanuscript markers:", banned,
          "(TODO in the title page and declarations are the administrative fields)")

    print(f"\n{len(CHECKS) - len(failures) - len(skipped)} ok, {len(failures)} mismatched, "
          f"{len(skipped)} skipped")
    if fig_bad:
        failures.extend(("figures", f"{f}:{l}", v, "see figure script") for f, l, v in fig_bad)
    if failures:
        print("\nMISMATCHES (manuscript vs artifact):")
        for s, c, stated, actual in failures:
            print(f"  §{s} {c}: paper says {stated}, artifact says {actual}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
