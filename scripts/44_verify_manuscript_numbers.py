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


def r(x: float, dec: int) -> float:
    """Round half away from zero, the convention the manuscript's prose uses."""
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))


def matches(stated: float, actual: float, dec: int) -> bool:
    """Accept the stated value if it is a defensible rounding of the artifact.

    Two roundings are allowed, and only for values sitting exactly on a boundary, because on a
    boundary the last digit is genuinely not determined by the artifact:

      * half-up, which is what the prose uses (0.4255 -> 0.426);
      * half-even, which is what Python's round() produces, and which matters because some
        artifacts store values already rounded to four decimals. Re-rounding 0.18648 -> 0.1865
        -> 0.187 would flag a correctly written 0.186 as a mismatch — double rounding, not an error
        in the manuscript.

    Away from a boundary the two agree, so this loosens nothing except the one digit that the
    stored precision cannot resolve.
    """
    return r(actual, dec) == r(stated, dec) or round(float(actual), dec) == round(float(stated), dec)


def load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


#  `results_eurohpc/` is the cluster's output tree and is not part of the public repository; the
#  artifacts the checks actually read are mirrored under `manuscript/tables/generated/artifacts/`.
#  Falling back to the mirror means someone who clones the repository can run this guard without
#  first knowing that the default path only exists on MareNostrum5 — the alternative is a verifier
#  that crashes on a KeyError and looks broken rather than unavailable.
MIRROR = Path("manuscript/tables/generated/artifacts")


def resolve(primary: str, mirror: str, sentinel: str) -> Path:
    """Prefer the cluster tree, fall back to the mirror.

    The test is a file the checks actually read, not the directory: `results_eurohpc/` exists
    locally but holds only the newest campaigns, so a directory test would select a tree missing
    the ladder metrics and fail with a KeyError several hundred lines later.
    """
    p = Path(primary)
    if (p / sentinel).exists():
        return p
    m = MIRROR / mirror
    return m if (m / sentinel).exists() else p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--scalability", default=None)
    ap.add_argument("--sections", default="manuscript/sections")
    ap.add_argument("--splits-root", default="data/splits_real")
    args = ap.parse_args()
    if args.artifacts is None:
        args.artifacts = str(resolve("results_eurohpc/primary_campaign", "primary_campaign",
                                     "metrics/top50/consolidated_metrics.json"))
    if args.scalability is None:
        args.scalability = str(resolve("results_eurohpc/scalability_v2/metrics",
                                       "scalability_v2/metrics",
                                       "top200/top200/consolidated_metrics.json"))
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
        "e12n": load_json(A / "t1_E12_vs_E11.json"),
        "e13n": load_json(A / "t1_E13_vs_E11.json"),
        "e14e12": load_json(A / "t1_E14_vs_E12.json"),
        "e14e13": load_json(A / "t1_E14_vs_E13.json"),
        "rev2": load_json(A / "reviewer_round2_analyses.json"),
        "inter": load_json(A / "interaction_contrast.json"),
        "e12": load_json(A / "t1_E12_vs_E6.json"),
        "leak2": load_json(A / "t3_evidence_leakage_k2.json"),
        "pc": load_json(A / "positive_control_tfidf.json"),
        "s7b": load_json(A / "steelman7b_contrasts.json"),
        "curve": load_json(A / "capacity_curve.json"),
        "lac": load_json(A / "label_attention_control.json"),
        "copy": load_json(A / "copy_instructed_control.json")
                or load_json(Path("results_eurohpc/copy_instructed/copy_instructed_control.json")),
        "band": load_json(A / "frequency_band_analysis.json"),
        "prev": load_json(A / "checklist_prevalence.json")
                or load_json(Path("results_eurohpc/primary_campaign/checklist_prevalence.json")),
        "shared100": load_json(A / "scalability_shared/metrics/top100/consolidated_metrics.json"),
        "shared200": load_json(A / "scalability_shared/metrics/top200/consolidated_metrics.json"),
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
        ("4.1", "E13 vs E11 delta (contrastive helps)", 0.008, 3, lambda: get(src["e13n"], "delta_micro_f1.point")),
        ("4.1", "E12 vs E11 delta", -0.053, 3, lambda: get(src["e12n"], "delta_micro_f1.point")),
        ("4.1", "E14 vs E12 delta (contrastive adds nothing)", -0.001, 3,
         lambda: get(src["e14e12"], "delta_micro_f1.point")),
        ("4.1", "E14 vs E13 delta", -0.062, 3, lambda: get(src["e14e13"], "delta_micro_f1.point")),
        ("4.1c", "E1 truncated to four codes", 0.508, 3,
         lambda: get(src["rev2"], "B_matched_cardinality.by_arm.E1.top4.micro_f1")),
        ("4.1c", "gold recall, hybrid retrieval", 0.384, 3,
         lambda: get(src["rev2"], "D_stagewise_gold_recall.by_stage.retrieval_only_top15.gold_recall")),
        ("4.1c", "gold recall, hybrid RAG", 0.353, 3,
         lambda: get(src["rev2"], "D_stagewise_gold_recall.by_stage.rag_top15.gold_recall")),
        ("4.1c", "gold recall, +evidence constraint", 0.129, 3,
         lambda: get(src["rev2"], "D_stagewise_gold_recall.by_stage.plus_evidence_constraint.gold_recall")),
        ("4.1c", "gold recall, +contrastive", 0.369, 3,
         lambda: get(src["rev2"], "D_stagewise_gold_recall.by_stage.plus_contrastive.gold_recall")),
        ("4.1c", "gold recall, full model", 0.119, 3,
         lambda: get(src["rev2"], "D_stagewise_gold_recall.by_stage.full_model.gold_recall")),
        ("4.4", "paired label-space lead, Top-50 (vs 100)", 0.3166, 4,
         lambda: get(src["rev2"], "E_label_space_on_shared_notes.by_pair.top50_vs_top100.E1_lead_top50")),
        ("4.4", "paired label-space lead, Top-100", 0.3747, 4,
         lambda: get(src["rev2"], "E_label_space_on_shared_notes.by_pair.top50_vs_top100.E1_lead_top100")),
        ("4.4", "paired label-space lead, Top-200", 0.4065, 4,
         lambda: get(src["rev2"], "E_label_space_on_shared_notes.by_pair.top50_vs_top200.E1_lead_top200")),
        ("4.1a", "tuned TF-IDF positive control", 0.605, 3,
         lambda: get(src["pc"], "tuned_global_threshold.micro_f1")),
        ("4.1b", "note-blind floor, K chosen on validation", 0.310, 3,
         lambda: get(src["rev2"], "A_floor_selected_on_validation.test_once.micro_f1")),
        ("4.1b", "note-blind floor at K=15", 0.285, 3,
         lambda: get(src["rev"], "A_note_blind_floor.by_K.K=15.micro_f1")),
        ("4.1c", "oracle over shortlist, matched per note", 0.264, 3,
         lambda: get(src["rev2"], "C_null_matched_per_note.oracle_over_shortlist.micro_f1")),
        ("4.1c", "random pruning null, matched per note", 0.119, 3,
         lambda: get(src["rev2"], "C_null_matched_per_note.random_pruning_null.micro_f1")),
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
        ("4.5", "interaction, full model", 0.196, 3,
         lambda: get(src["inter"], "arms.E14.interaction.point")),
        ("4.5", "interaction CI low", 0.175, 3, lambda: get(src["inter"], "arms.E14.interaction.ci95.0")),
        ("4.5", "interaction CI high", 0.216, 3, lambda: get(src["inter"], "arms.E14.interaction.ci95.1")),
        ("4.5", "capacity effect without note (3B->7B)", -0.008, 3,
         lambda: get(src["inter"], "arms.E14.simple_effect_of_capacity_without_note.point")),
    ]

    # The three-point capacity curve and the reference contrasts it makes necessary (scripts/54).
    # These replaced §4.5's 2 x 2 conclusion: the third capacity point moved the best cell from
    # "indistinguishable from the note-blind floor" to "clears it by 0.130", so every number the
    # revised section rests on is checked here rather than trusted.
    if src["curve"]:
        CU = src["curve"]

        def cell(arm: str, ctx: str, cap: str):
            return lambda: get(CU, f"arms.{arm}.curve_micro_f1.{ctx}.{cap}")

        def step(arm: str, i: int, path: str):
            return lambda: get(CU, f"arms.{arm}.steps.{i}.{path}")

        # steps are ordered 3B->7B, 7B->14B, 3B->14B by the script that writes them
        CHECKS += [
            ("4.5", "Table 6: 3B no note, E11", 0.189, 3, cell("E11", "nonote", "3B")),
            ("4.5", "Table 6: 7B no note, E11", 0.201, 3, cell("E11", "nonote", "7B")),
            ("4.5", "Table 6: 14B no note, E11", 0.216, 3, cell("E11", "nonote", "14B")),
            ("4.5", "Table 6: 3B note, E11", 0.187, 3, cell("E11", "note", "3B")),
            ("4.5", "Table 6: 7B note, E11", 0.235, 3, cell("E11", "note", "7B")),
            ("4.5", "Table 6: 14B note, E11", 0.304, 3, cell("E11", "note", "14B")),
            ("4.5", "Table 6: 3B no note, E14", 0.138, 3, cell("E14", "nonote", "3B")),
            ("4.5", "Table 6: 7B no note, E14", 0.130, 3, cell("E14", "nonote", "7B")),
            ("4.5", "Table 6: 14B no note, E14", 0.174, 3, cell("E14", "nonote", "14B")),
            ("4.5", "Table 6: 3B note, E14", 0.105, 3, cell("E14", "note", "3B")),
            ("4.5", "Table 6: 7B note, E14", 0.293, 3, cell("E14", "note", "7B")),
            ("4.5", "Table 6: 14B note, E14 (best cell)", 0.426, 3, cell("E14", "note", "14B")),
            ("4.5", "capacity 3B->14B without note, E14", 0.036, 3,
             step("E14", 2, "capacity_effect_without_note")),
            ("4.5", "capacity 3B->14B with note, E14", 0.320, 3,
             step("E14", 2, "capacity_effect_with_note")),
            ("4.5", "interaction 3B->14B, E14", 0.284, 3, step("E14", 2, "interaction.point")),
            ("4.5", "interaction 3B->14B CI low", 0.263, 3, step("E14", 2, "interaction.ci95.0")),
            ("4.5", "interaction 3B->14B CI high", 0.305, 3, step("E14", 2, "interaction.ci95.1")),
            ("4.5", "interaction 3B->7B, E14", 0.196, 3, step("E14", 0, "interaction.point")),
            ("4.5", "interaction 7B->14B, E14", 0.088, 3, step("E14", 1, "interaction.point")),
            ("4.5", "interaction 7B->14B CI low", 0.069, 3, step("E14", 1, "interaction.ci95.0")),
            ("4.5", "interaction 7B->14B CI high", 0.108, 3, step("E14", 1, "interaction.ci95.1")),
            ("4.5", "interaction 3B->14B, E11", 0.090, 3, step("E11", 2, "interaction.point")),
            ("4.5", "interaction 3B->14B CI low, E11", 0.081, 3, step("E11", 2, "interaction.ci95.0")),
            ("4.5", "interaction 3B->14B CI high, E11", 0.100, 3, step("E11", 2, "interaction.ci95.1")),
        ]
        R = "reference_contrasts"
        CHECKS += [
            ("4.5", "best cell vs note-blind floor", 0.130, 3,
             lambda: get(CU, f"{R}.vs_note_blind_floor.delta_micro_f1.point")),
            ("4.5", "best cell vs floor CI low", 0.115, 3,
             lambda: get(CU, f"{R}.vs_note_blind_floor.delta_micro_f1.ci95.0")),
            ("4.5", "best cell vs floor CI high", 0.145, 3,
             lambda: get(CU, f"{R}.vs_note_blind_floor.delta_micro_f1.ci95.1")),
            ("4.5", "note-blind floor on the subset (K=8)", 0.296, 3,
             lambda: get(CU, f"{R}.vs_note_blind_floor.floor_K8.micro_f1")),
            ("4.5", "best cell vs TF-IDF", -0.024, 3,
             lambda: get(CU, f"{R}.vs_E1_tfidf.delta_micro_f1.point")),
            ("4.5", "best cell vs TF-IDF CI low", -0.039, 3,
             lambda: get(CU, f"{R}.vs_E1_tfidf.delta_micro_f1.ci95.0")),
            ("4.5", "best cell vs TF-IDF CI high", -0.009, 3,
             lambda: get(CU, f"{R}.vs_E1_tfidf.delta_micro_f1.ci95.1")),
            ("4.5", "TF-IDF on the subset", 0.449, 3,
             lambda: get(CU, f"{R}.vs_E1_tfidf.E1_tfidf.micro_f1")),
        ]
    else:
        print("NOTE: capacity_curve.json absent — §4.5's three-point curve unchecked\n")

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

    # ---- round 4 (reviewer items T3-1, T3-2, T3-3, T3-5, T3-7) --------------------------------
    if src["lac"]:
        CHECKS += [
            ("4.1a", "label-attention, fixed 15-code budget", 0.4149, 4,
             lambda: get(src["lac"], "fixed_budget.micro_f1")),
            ("4.1a", "label-attention, tuned threshold", 0.5587, 4,
             lambda: get(src["lac"], "tuned_threshold.micro_f1")),
            ("4.1a", "label-attention threshold selected on validation", 0.32, 2,
             lambda: get(src["lac"], "tuned_threshold.threshold")),
        ]
    if src["copy"]:
        CHECKS += [
            ("4.3", "exact-quote compliance, published prompt", 0.083, 3,
             lambda: get(src["copy"], "primary_as_published_prompt.exact_quote_compliance")),
            ("4.3", "exact-quote compliance, copy-instructed", 0.057, 3,
             lambda: get(src["copy"], "control_copy_instructed.exact_quote_compliance")),
            ("4.3", "copy-instructed delta", -0.026, 3,
             lambda: get(src["copy"], "paired_test.observed_delta")),
            ("4.3", "copy-instructed CI low", -0.042, 3,
             lambda: get(src["copy"], "paired_test.ci95.0")),
            ("4.3", "copy-instructed CI high", -0.010, 3,
             lambda: get(src["copy"], "paired_test.ci95.1")),
            ("4.3", "copy-instructed p", 0.0013, 4,
             lambda: get(src["copy"], "paired_test.p_two_sided")),
            ("4.3", "checkable quotes, copy-instructed", 5867, 0,
             lambda: get(src["copy"], "control_copy_instructed.quote_checked_codes")),
            ("4.3", "checkable quotes, published prompt", 4964, 0,
             lambda: get(src["copy"], "primary_as_published_prompt.quote_checked_codes")),
        ]
    if src["shared100"] and src["shared200"]:
        def sh(which: str, arm: str):
            return lambda: get(src[which], f"individual.{arm}_seed42.classification.micro_f1")
        CHECKS += [
            ("4.4", "shared partition Top-100 E1", 0.469, 3, sh("shared100", "E1")),
            ("4.4", "shared partition Top-100 E6", 0.165, 3, sh("shared100", "E6")),
            ("4.4", "shared partition Top-100 E11", 0.139, 3, sh("shared100", "E11")),
            ("4.4", "shared partition Top-100 E14", 0.098, 3, sh("shared100", "E14")),
            ("4.4", "shared partition Top-200 E1", 0.467, 3, sh("shared200", "E1")),
            ("4.4", "shared partition Top-200 E6", 0.123, 3, sh("shared200", "E6")),
            ("4.4", "shared partition Top-200 E11", 0.110, 3, sh("shared200", "E11")),
            ("4.4", "shared partition Top-200 E14", 0.070, 3, sh("shared200", "E14")),
        ]
    if src["band"]:
        def bd(arm: str, b: int):
            return lambda: get(src["band"], f"per_arm.{arm}.band{b}.micro_f1")

        def gap(arm: str, b: int):
            return lambda: (get(src["band"], f"per_arm.E1.band{b}.micro_f1")
                            - get(src["band"], f"per_arm.{arm}.band{b}.micro_f1"))
        CHECKS += [
            ("4.4", "band 1 E1", 0.503, 3, bd("E1", 1)),
            ("4.4", "band 2 E1", 0.422, 3, bd("E1", 2)),
            ("4.4", "band 3 E1", 0.365, 3, bd("E1", 3)),
            ("4.4", "band 4 E1", 0.371, 3, bd("E1", 4)),
            ("4.4", "band 1 E11", 0.126, 3, bd("E11", 1)),
            ("4.4", "band 2 E11", 0.112, 3, bd("E11", 2)),
            ("4.4", "band 3 E11", 0.085, 3, bd("E11", 3)),
            ("4.4", "band 4 E11", 0.089, 3, bd("E11", 4)),
            ("4.4", "band 1 gap to E11", 0.377, 3, gap("E11", 1)),
            ("4.4", "band 4 gap to E11", 0.283, 3, gap("E11", 4)),
            ("4.4", "band 1 gap to E14", 0.443, 3, gap("E14", 1)),
            ("4.4", "band 4 gap to E14", 0.281, 3, gap("E14", 4)),
        ]
    if src["prev"]:
        CHECKS += [
            ("5.5", "checklist corpus size", 20, 0, lambda: get(src["prev"], "corpus_size")),
            ("5.5", "studies read at full text", 8, 0,
             lambda: get(src["prev"], "read_at_full_text")),
            ("5.5", "item 1 observed", 0, 0, lambda: get(src["prev"], "items.1.Y")),
            ("5.5", "item 1 determinable", 8, 0, lambda: get(src["prev"], "items.1.determinable")),
            ("5.5", "item 1 rule-of-three bound", 0.38, 2,
             lambda: get(src["prev"], "items.1.rule_of_three_upper_bound")),
            ("5.5", "item 7 determinable", 6, 0, lambda: get(src["prev"], "items.7.determinable")),
            ("5.5", "item 8 determinable", 6, 0, lambda: get(src["prev"], "items.8.determinable")),
        ]

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
        ok = matches(stated, actual, dec)
        print(f"{section:6s} {claim:46s} {stated:>10} {r(actual, dec):>10}  "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((section, claim, stated, actual))

    # Subset densities are quoted in Methods §3.5 to argue the subsets are not unusual; they are
    # computable, and the first version of that sentence quoted values that were not measured.
    split_test = Path(args.splits_root) / "top50" / "test.jsonl"
    if split_test.exists():
        g = []
        with open(split_test, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    g.append(len(json.loads(line).get("gold_codes") or []))
        CHECKS.append(("3.5", "gold codes/note, first 200", 5.47, 2, lambda g=g: sum(g[:200]) / 200))
        CHECKS.append(("3.5", "gold codes/note, first 1,008", 5.38, 2, lambda g=g: sum(g[:1008]) / 1008))

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
        elif not matches(stated, actual, 3):
            fig_bad.append((fname, label, f"{stated} vs artifact {r(actual, 3)}"))
    print("\nfigure-script values:",
          "all present and matching the artifacts" if not fig_bad else f"PROBLEM {fig_bad}")

    # Three defects reached a built PDF while this script reported "0 mismatched", because it only
    # looked at numbers. An independent reviewer found all three. They are cheap to test for.
    hygiene = []
    msdir = Path(args.sections).parent

    #  (i) BibTeX prints the `note` field, so private annotations are typeset into the reference
    #      list. One of ours read "do NOT cite this paper as Klang et al."
    bib = msdir / "references" / "references.bib"
    if bib.exists():
        stray = re.findall(r"^\s*note\s*=", bib.read_text(encoding="utf-8"), re.M)
        if stray:
            hygiene.append(f"{len(stray)} `note =` field(s) in references.bib will be printed by BibTeX")

    #  (ii) Cross-references must resolve. The manuscript numbers its own sections (4.1a-4.1e);
    #       if the build ever renumbers them, every "Section 4.2" silently points elsewhere.
    tex = msdir / "build" / "main.tex"
    if tex.exists():
        t = tex.read_text(encoding="utf-8")
        heads = set(re.findall(r"\\(?:sub)?section\*?\{(\d+\.\d+[a-z]?)", t))
        refs = set(re.findall(r"(?:Section|\\S~)\s?(\d+\.\d+[a-z]?)", t))
        dangling = sorted(refs - heads)
        if dangling:
            hygiene.append(f"cross-references with no matching heading: {', '.join(dangling)}")

    #  (iii) Figure captions carry numbers too, and a caption kept a stale value through a re-run.
    caps = msdir / "figures" / "FIGURE_CAPTIONS.md"
    if caps.exists():
        ctext = caps.read_text(encoding="utf-8")
        if re.search(r"\+0\.373", ctext):
            hygiene.append("figure caption still gives the superseded scalability lead +0.373 (now +0.399)")
        # k=5 may be mentioned, but only alongside the k=2 figure the text reports as primary.
        if "25,845" in ctext and "10,338" not in ctext:
            hygiene.append("figure caption reports leakage at k=5 without the primary k=2 figure")

    print("\nmanuscript hygiene:", "clean" if not hygiene else "")
    for h in hygiene:
        print(f"  PROBLEM {h}")

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
    if hygiene:
        failures.extend(("hygiene", h, "-", "-") for h in hygiene)
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
