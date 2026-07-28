from .local_llm import LocalLLM
from .prompts import CODE_SCORER_PROMPT, BATCH_CODE_SCORER_PROMPT, BATCH_CODE_SCORER_PROMPT_WITH_NOTE
from .json_parser import parse_llm_json
import hashlib

VALID_RISK_FLAGS = {"none", "weak_evidence", "ambiguous", "possible_hallucination"}


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1", "supported", "y"}
    return False


def _as_conf(v) -> float:
    try:
        c = float(v)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    return max(0.0, min(1.0, c))


def _expand_compact(item: dict) -> dict:
    """Map the compact batch schema ('q'/'r', fields omitted when unsupported) onto the
    full documented schema. Compact output is what keeps the 50-candidate batched call
    affordable (sequential decode dominates cost)."""
    if not isinstance(item, dict):
        return {}
    out = dict(item)
    if "evidence_quote" not in out and "q" in out:
        out["evidence_quote"] = out.get("q") or ""
    if "rationale" not in out and "r" in out:
        out["rationale"] = out.get("r") or ""
    out.setdefault("evidence_quote", "")
    out.setdefault("rationale", "")
    if "risk_flag" not in out:
        out["risk_flag"] = "none" if _as_bool(out.get("supported", False)) else "ambiguous"
    return out


def _has_confidence(parsed: dict) -> bool:
    """Did the model actually emit a usable confidence, or are we about to invent one?

    Not every model honours the compact batch schema: Qwen2.5-7B omits `confidence` on
    ~90% of items where 3B emits it. Defaulting the absent field to 0.0 silently fabricates
    data AND corrupts downstream ranking (see rag_pipeline ordering), so "absent" has to
    stay distinguishable from "the model said 0.0".
    """
    if "confidence" not in parsed:
        return False
    v = parsed.get("confidence")
    if v is None:
        return False
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def normalize_code_score(parsed: dict, candidate_code: str) -> dict:
    """Coerce a parsed code-scorer response to the documented schema with safe defaults.

    Real LLM output is not guaranteed to be well-formed; without this, missing/malformed
    fields flow silently into the reliability metrics. Adds `schema_valid` (whether the
    required keys were present and typed) without discarding usable values.
    """
    if not isinstance(parsed, dict):
        parsed = {}
    required = ("supported", "confidence", "risk_flag")
    schema_valid = all(k in parsed for k in required)
    confidence_present = _has_confidence(parsed)

    supported = _as_bool(parsed.get("supported", False))
    confidence = _as_conf(parsed.get("confidence", 0.0))
    risk_flag = str(parsed.get("risk_flag", "") or "").strip().lower()
    if risk_flag not in VALID_RISK_FLAGS:
        schema_valid = False
        risk_flag = "none" if supported else "ambiguous"

    return {
        "code": str(parsed.get("code") or candidate_code),
        "supported": supported,
        "confidence": confidence,
        "confidence_present": confidence_present,
        "evidence_quote": str(parsed.get("evidence_quote", "") or ""),
        "rationale": str(parsed.get("rationale", "") or ""),
        "missing_evidence": str(parsed.get("missing_evidence", "") or ""),
        "risk_flag": risk_flag,
        "schema_valid": bool(schema_valid),
        "mock_llm": bool(parsed.get("mock_llm", False)),
    }


class CodeScorer:
    def __init__(self, llm: LocalLLM, config: dict | None = None):
        self.llm = llm
        self.config = config or {}
        
    def score_candidate(self, candidate_code: str, candidate_title: str, candidate_desc: str, evidence: str) -> dict:
        evidence_max = int(self.config.get("evidence_snippet_max_chars", 0) or 0)
        prompt_max = int(self.config.get("prompt_max_chars", 0) or 0)
        if evidence_max:
            evidence = evidence[:evidence_max]
        prompt = CODE_SCORER_PROMPT.format(
            icd_code=candidate_code,
            icd_title=candidate_title,
            icd_description=candidate_desc,
            evidence=evidence
        )
        if prompt_max:
            prompt = prompt[:prompt_max]
        
        return self._score_single(prompt, candidate_code, response=self.llm.generate(prompt))

    def score_candidates_batched(self, candidates: list, note_text: str = "") -> list:
        """Score all candidates for a note in ONE structured LLM call.

        `candidates`: list of {code, title, description, evidence_text}. Returns a list of
        normalized per-code dicts aligned to the input order. This is the publication-grade
        design: ~1 LLM call per note instead of one per candidate. Evidence is the note-local
        evidence provided per candidate.
        """
        if not candidates:
            return []
        snippet_max = int(self.config.get("evidence_snippet_max_chars", 0) or 0)
        prompt_max = int(self.config.get("prompt_max_chars", 0) or 0)
        lines = []
        for i, c in enumerate(candidates, start=1):
            ev = str(c.get("evidence_text", "") or "")
            if snippet_max:
                ev = ev[:snippet_max]
            title = c.get("title") or c.get("description") or ""
            lines.append(f"{i}. {c.get('code','')} ({title})\n   Evidence: {ev}")
        block = "\n".join(lines)
        # Steelman path: show the scorer the note itself, not only per-candidate snippets.
        # The default pipeline gives the LLM `evidence_snippet_max_chars` (200) per candidate and
        # never the note, so it judges codes from fragments; `include_note_in_prompt` lifts that
        # restriction so the failure can be attributed to the design rather than to context
        # starvation. Off by default to keep the primary campaign's behaviour unchanged.
        if bool(self.config.get("include_note_in_prompt", False)) and note_text:
            note_cap = int(self.config.get("prompt_note_max_chars", 6000) or 6000)
            prompt = BATCH_CODE_SCORER_PROMPT_WITH_NOTE.format(
                note=str(note_text)[:note_cap], candidates_block=block)
        else:
            prompt = BATCH_CODE_SCORER_PROMPT.format(candidates_block=block)
        batch_prompt_max = int(self.config.get("batch_prompt_max_chars", 0) or 0) or prompt_max
        if batch_prompt_max:
            prompt = prompt[:batch_prompt_max]

        # The batch response is a JSON array with one object per candidate (~90 tokens each);
        # a fixed small max_new_tokens (e.g. the lowmem 64) would truncate it and zero out
        # every score. Budget dynamically per candidate count, with a hard cap.
        batch_tokens = min(120 + 40 * len(candidates), int(self.config.get("batch_max_new_tokens", 3072)))
        # prompt_max_chars=0: already capped by batch_prompt_max_chars above; the LLM's
        # single-candidate cap would otherwise cut off the trailing JSON instructions.
        response = self.llm.generate(prompt, max_new_tokens=batch_tokens, prompt_max_chars=0)
        try:
            parsed = parse_llm_json(response)
            items = parsed if isinstance(parsed, list) else parsed.get("codes") or parsed.get("results") or []
            by_code = {}
            for it in items if isinstance(items, list) else []:
                if isinstance(it, dict) and it.get("code"):
                    by_code[str(it["code"])] = it
            out = []
            for c in candidates:
                code = str(c.get("code", ""))
                res = normalize_code_score(_expand_compact(by_code.get(code, {})), code)
                res["mock_llm"] = bool(getattr(self.llm, "is_mock", False) or res.get("mock_llm"))
                res["json_parse_error"] = False
                out.append(res)
            return out
        except Exception as e:
            output_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()[:12]
            out = []
            for c in candidates:
                out.append({
                    "code": str(c.get("code", "")), "supported": False, "confidence": 0.0,
                    "confidence_present": False,
                    "evidence_quote": "", "rationale": "", "missing_evidence": "",
                    "risk_flag": "ambiguous", "schema_valid": False, "json_parse_error": True,
                    "parser_stage": "batch_code_scorer", "output_hash": output_hash,
                    "mock_llm": bool(getattr(self.llm, "is_mock", False)), "error_type": type(e).__name__,
                })
            return out

    def _score_single(self, prompt: str, candidate_code: str, response: str) -> dict:
        try:
            parsed = parse_llm_json(response)
            result = normalize_code_score(parsed, candidate_code)
            result["mock_llm"] = bool(getattr(self.llm, "is_mock", False) or result.get("mock_llm"))
            result["json_parse_error"] = False
            return result
        except Exception as e:
            output_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()[:12]
            return {
                "code": candidate_code,
                "supported": False,
                "confidence": 0.0,
                "confidence_present": False,
                "evidence_quote": "",
                "rationale": "",
                "missing_evidence": "",
                "risk_flag": "ambiguous",
                "schema_valid": False,
                "json_parse_error": True,
                "parser_stage": "code_scorer",
                "output_hash": output_hash,
                "mock_llm": bool(getattr(self.llm, "is_mock", False)),
                "error_type": type(e).__name__,
            }
