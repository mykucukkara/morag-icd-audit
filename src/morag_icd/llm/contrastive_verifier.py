from .local_llm import LocalLLM
from .prompts import CONTRASTIVE_VERIFIER_PROMPT
from .json_parser import parse_llm_json
from .code_scorer import _as_conf
import hashlib


def normalize_contrastive(parsed: dict, target_code: str) -> dict:
    """Coerce a parsed contrastive-verifier response to the documented schema."""
    if not isinstance(parsed, dict):
        parsed = {}
    schema_valid = all(k in parsed for k in ("preferred_code", "rejected_codes", "confidence"))
    rejected = []
    raw = parsed.get("rejected_codes")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("code"):
                rejected.append({"code": str(item.get("code")), "reason": str(item.get("reason", "") or "")})
            elif isinstance(item, str) and item.strip():
                rejected.append({"code": item.strip(), "reason": ""})
    else:
        schema_valid = False
    return {
        "preferred_code": str(parsed.get("preferred_code") or target_code),
        "rejected_codes": rejected,
        "contrastive_rationale": str(parsed.get("contrastive_rationale", "") or ""),
        "confidence": _as_conf(parsed.get("confidence", 0.0)),
        "schema_valid": bool(schema_valid),
        "mock_llm": bool(parsed.get("mock_llm", False)),
    }


class ContrastiveVerifier:
    def __init__(self, llm: LocalLLM, config: dict | None = None):
        self.llm = llm
        self.config = config or {}
        
    def verify(self, target_code: str, target_title: str, sibling_codes: list[dict], evidence: str) -> dict:
        max_pairs = int(self.config.get("contrastive_max_pairs", 0) or 0)
        if max_pairs:
            sibling_codes = sibling_codes[:max_pairs]
        evidence_max = int(self.config.get("evidence_snippet_max_chars", 0) or 0)
        prompt_max = int(self.config.get("prompt_max_chars", 0) or 0)
        if evidence_max:
            evidence = evidence[:evidence_max]
        siblings_text = "\n".join([f"- {s.get('code','')}: {s.get('title','')}" for s in sibling_codes])
        prompt = CONTRASTIVE_VERIFIER_PROMPT.format(
            target_code=target_code,
            target_title=target_title,
            sibling_codes=siblings_text,
            evidence=evidence
        )
        if prompt_max:
            prompt = prompt[:prompt_max]
        
        response = self.llm.generate(prompt)
        try:
            parsed = parse_llm_json(response)
            result = normalize_contrastive(parsed, target_code)
            result["mock_llm"] = bool(getattr(self.llm, "is_mock", False) or result.get("mock_llm"))
            result["json_parse_error"] = False
            return result
        except Exception as e:
            output_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()[:12]
            return {
                "preferred_code": target_code,
                "confidence": 0.0,
                "rejected_codes": [],
                "contrastive_rationale": "",
                "schema_valid": False,
                "json_parse_error": True,
                "parser_stage": "contrastive_verifier",
                "output_hash": output_hash,
                "mock_llm": bool(getattr(self.llm, "is_mock", False)),
                "error_type": type(e).__name__,
            }
