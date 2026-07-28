"""
Evidence-quality and reliability metrics for ICD-10 code recommendation.

These metrics evaluate the trustworthiness of predictions beyond classification
accuracy: how well are codes supported by clinical evidence?
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


def _normalize_ws(s: str) -> str:
    """Lowercase + collapse whitespace, for a lenient verbatim substring test.

    Models reproduce a quote's tokens faithfully but often normalize spacing/newlines, so a
    raw substring test understates true grounding; this keeps the check external (still the
    model's own tokens against the source) without being defeated by whitespace.
    """
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def compute_reliability_metrics(predictions: List[Dict]) -> Dict[str, float]:
    """
    Compute evidence support and risk flag metrics from a list of prediction dicts.

    Each prediction dict must contain a 'predicted_codes' key whose value is a list
    of code-level dicts, each with optional fields:
        - supported (bool)
        - risk_flag (str): "none"|"weak_evidence"|"ambiguous"|"possible_hallucination"
        - evidence_quote (str)
        - rationale (str)
        - contrastive_rationale (str)

    Returns
    -------
    dict with reliability metrics.
    """
    total_codes = 0
    evidence_judged_codes = 0             # codes carrying an explicit `supported` judgement
    supported_count = 0
    unsupported_count = 0
    weak_evidence_count = 0
    ambiguous_count = 0
    hallucination_count = 0
    has_rationale_count = 0
    has_evidence_quote_count = 0
    verbatim_quote_count = 0              # evidence_quote found VERBATIM in the note evidence
    quote_checked_count = 0              # codes for which a verbatim check was possible
    has_contrastive_rationale_count = 0
    has_rejected_codes_count = 0          # non-empty rejected_similar_codes
    contrastive_fallback_count = 0        # rejection produced by deterministic fallback, not the LLM
    llm_contrastive_count = 0             # contrastive present AND produced by the LLM (not fallback)
    schema_invalid_count = 0             # LLM output failed schema normalization
    json_parse_error_count = 0          # LLM output could not be parsed at all
    confidence_missing_count = 0        # model omitted `confidence` (NOT the same as 0.0)

    for pred in predictions:
        code_list = pred.get("predicted_codes", [])
        for code_pred in code_list:
            total_codes += 1

            # A baseline (TF-IDF, BM25, plain classifier) makes NO evidence claim: it emits
            # `supported: None` (the key is present but carries no judgement). Counting those
            # codes as "unsupported" would report ESR=0 / UCR=1.00 — a fabricated reliability
            # verdict for a system that never claimed grounding. Only codes carrying an EXPLICIT
            # boolean judgement enter the ESR/UCR base; None means not-applicable.
            has_judgement = code_pred.get("supported") is not None
            if has_judgement:
                evidence_judged_codes += 1
            if code_pred.get("supported", False):
                supported_count += 1
            elif has_judgement:
                unsupported_count += 1

            # Verbatim grounding (external check, not self-report). A supported code should
            # carry an evidence_quote COPIED from the note evidence. The authoritative check
            # is done at inference (rag_pipeline) against the FULL note-local evidence, which
            # is never serialized (PHI), and stored as the boolean `evidence_quote_verbatim`.
            # Fall back to a substring check against the stored preview for older records.
            # A mock LLM copies the quote straight out of the evidence it is shown, so its
            # verbatim rate is ~1.0 by construction and carries no grounding signal — exclude
            # mock codes from this metric (they are excluded from manuscript tables anyway).
            if not code_pred.get("mock_llm", False):
                verbatim = code_pred.get("evidence_quote_verbatim")
                if verbatim is None:
                    quote = str(code_pred.get("evidence_quote", "") or "").strip()
                    source = str(code_pred.get("evidence_preview", "") or "")
                    if quote and source:
                        verbatim = _normalize_ws(quote) in _normalize_ws(source)
                if verbatim is not None:
                    quote_checked_count += 1
                    if verbatim:
                        verbatim_quote_count += 1

            rejected = code_pred.get("rejected_similar_codes")
            has_rejected = isinstance(rejected, list) and len(rejected) > 0
            if has_rejected:
                has_rejected_codes_count += 1
            if code_pred.get("contrastive_fallback_used") is True:
                contrastive_fallback_count += 1
            elif has_rejected or code_pred.get("contrastive_rationale"):
                # contrastive signal that did NOT come from the deterministic fallback
                llm_contrastive_count += 1
            if code_pred.get("schema_valid") is False:
                schema_invalid_count += 1
            if code_pred.get("json_parse_error") is True:
                json_parse_error_count += 1
            if code_pred.get("confidence_present") is False:
                confidence_missing_count += 1

            risk = code_pred.get("risk_flag", "none")
            if risk == "weak_evidence":
                weak_evidence_count += 1
            elif risk == "ambiguous":
                ambiguous_count += 1
            elif risk == "possible_hallucination":
                hallucination_count += 1

            if code_pred.get("rationale", ""):
                has_rationale_count += 1
            if code_pred.get("evidence_quote", "") or code_pred.get("evidence_preview", ""):
                has_evidence_quote_count += 1
            if code_pred.get("contrastive_rationale", ""):
                has_contrastive_rationale_count += 1

    if total_codes == 0:
        return _empty_reliability_metrics()

    # ESR/UCR are defined only over codes that actually carry a judgement. For a baseline
    # with no `supported` field this base is 0 -> the rates are None (N/A), not 0.0/1.0.
    esr = supported_count / evidence_judged_codes if evidence_judged_codes else None
    ucr = unsupported_count / evidence_judged_codes if evidence_judged_codes else None
    verbatim_rate = verbatim_quote_count / quote_checked_count if quote_checked_count else None

    return {
        "total_predicted_codes": total_codes,
        "evidence_judged_codes": evidence_judged_codes,
        "evidence_support_rate": esr,
        "unsupported_code_rate": ucr,
        # External grounding: fraction of emitted quotes that appear verbatim in the note
        # evidence, plus the base it was measured over (None when no quotes were checkable,
        # e.g. baselines). This is the non-circular counterpart to evidence_support_rate.
        "evidence_quote_verbatim_rate": verbatim_rate,
        "evidence_quote_checked_codes": quote_checked_count,
        "weak_evidence_rate": weak_evidence_count / total_codes,
        "ambiguous_code_rate": ambiguous_count / total_codes,
        "hallucination_flag_rate": hallucination_count / total_codes,
        "rationale_coverage_rate": has_rationale_count / total_codes,
        "evidence_quote_rate": has_evidence_quote_count / total_codes,
        "contrastive_rationale_rate": has_contrastive_rationale_count / total_codes,
        "rejected_similar_codes_rate": has_rejected_codes_count / total_codes,
        # Honesty split: how much of the "contrastive verification" is the LLM vs the
        # deterministic fallback. Manuscript claims about LLM disambiguation must use
        # llm_contrastive_rate, not the combined rate.
        "contrastive_fallback_rate": contrastive_fallback_count / total_codes,
        "llm_contrastive_rate": llm_contrastive_count / total_codes,
        # Parse reliability of the LLM output feeding these metrics.
        "schema_invalid_rate": schema_invalid_count / total_codes,
        "json_parse_error_rate": json_parse_error_count / total_codes,
        # Share of codes whose confidence the model never emitted. Any statistic computed
        # over `confidence` (calibration, AUROC, a tuned threshold) is only defined on the
        # complement of this rate, so it must be reported alongside them.
        "confidence_missing_rate": confidence_missing_count / total_codes,
    }


def compute_similar_code_confusion_rate(
    predictions: List[Dict],
    gold: List[Dict],
    icd_kb: Optional[Dict] = None,
) -> float:
    """
    Estimate the rate at which predicted codes are from the same ICD family as
    a gold code but not the exact gold code (i.e. within same 3-char parent).

    Returns the fraction of incorrect predictions that are 'near misses'
    (same 3-character prefix as a gold code).
    """
    from .hierarchical_metrics import _to_parent  # shared, dot-insensitive ICD category

    incorrect_preds = 0
    near_miss_preds = 0

    gold_map = {g["hadm_id"]: set(g.get("gold_codes", [])) for g in gold}

    for pred in predictions:
        hadm_id = pred.get("hadm_id")
        gold_codes = gold_map.get(hadm_id, set())
        gold_parents = {_to_parent(c) for c in gold_codes}

        for code_pred in pred.get("predicted_codes", []):
            code = code_pred.get("code", "")
            if code not in gold_codes:
                incorrect_preds += 1
                if _to_parent(code) in gold_parents:
                    near_miss_preds += 1

    if incorrect_preds == 0:
        return 0.0
    return near_miss_preds / incorrect_preds


def _empty_reliability_metrics() -> Dict[str, float]:
    return {
        "total_predicted_codes": 0,
        "evidence_judged_codes": 0,
        "evidence_support_rate": None,
        "unsupported_code_rate": None,
        "evidence_quote_verbatim_rate": None,
        "evidence_quote_checked_codes": 0,
        "weak_evidence_rate": 0.0,
        "ambiguous_code_rate": 0.0,
        "hallucination_flag_rate": 0.0,
        "rationale_coverage_rate": 0.0,
        "evidence_quote_rate": 0.0,
        "contrastive_rationale_rate": 0.0,
        "rejected_similar_codes_rate": 0.0,
        "contrastive_fallback_rate": 0.0,
        "llm_contrastive_rate": 0.0,
        "schema_invalid_rate": 0.0,
        "json_parse_error_rate": 0.0,
        "confidence_missing_rate": 0.0,
    }
