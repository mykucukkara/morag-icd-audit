"""
LLM-only baselines for ICD-10 code recommendation (E7, E8).

E7: Zero-shot - prompt the LLM with clinical note, ask for ICD-10 codes.
E8: Few-shot  - include example cases in the prompt.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..llm.local_llm import LocalLLM
from ..llm.json_parser import parse_llm_json


ZERO_SHOT_PROMPT = """You are an expert clinical coder. Given the following clinical discharge summary, identify all applicable ICD-10 diagnosis codes.

DISCHARGE SUMMARY:
{note_text}

Return a JSON object with the following structure:
{{
  "icd10_codes": [
    {{
      "code": "...",
      "title": "...",
      "rationale": "..."
    }}
  ]
}}

Only return valid JSON. List only ICD-10 codes (not ICD-9).
"""

FEW_SHOT_EXAMPLES = """EXAMPLE 1:
Note: "Patient admitted with chest pain and shortness of breath. EKG showed ST elevation. 
Troponin elevated. Treated for acute STEMI. History of type 2 diabetes."
Output: {{"icd10_codes": [{{"code": "I21.9", "title": "Acute myocardial infarction, unspecified", "rationale": "ST elevation and elevated troponin"}}, {{"code": "E11.9", "title": "Type 2 diabetes mellitus without complications", "rationale": "Patient history"}}]}}

EXAMPLE 2:
Note: "Elderly patient with worsening shortness of breath. BNP elevated. Echo shows EF 35%. Diagnosis: congestive heart failure with reduced ejection fraction."
Output: {{"icd10_codes": [{{"code": "I50.22", "title": "Chronic systolic (congestive) heart failure", "rationale": "Reduced EF confirmed by echo"}}]}}

"""

CLOSED_SET_PROMPT = """You are an expert clinical coder. Assign ICD-10 diagnosis codes to the discharge summary below, choosing ONLY from the candidate list provided.

CANDIDATE ICD-10 CODES (choose only from this list):
{label_catalog}

DISCHARGE SUMMARY:
{note_text}

Select every candidate code that the summary supports. Typical admissions carry several codes.

Return ONLY a JSON object of the form:
{{"icd10_codes": [{{"code": "<code exactly as written in the candidate list>", "rationale": "<max 10 words>"}}]}}

Use the code strings exactly as they appear in the candidate list. Do not invent codes outside the list. Only return valid JSON.
"""

FEW_SHOT_PROMPT = """You are an expert clinical coder. Here are some examples, then code the new case.

{examples}

NEW CASE:
DISCHARGE SUMMARY:
{note_text}

Return a JSON object with the following structure:
{{
  "icd10_codes": [
    {{
      "code": "...",
      "title": "...",
      "rationale": "..."
    }}
  ]
}}

Only return valid JSON. List only ICD-10 codes (not ICD-9).
"""


class LLMOnlyBaseline:
    """
    Pure LLM-based ICD-10 code recommender without retrieval (E7, E8).

    E7 = zero-shot: no examples.
    E8 = few-shot: static examples prepended to prompt.
    """

    MAX_NOTE_CHARS = 3000  # Truncate long notes to fit context window

    def __init__(
        self,
        llm_path: str = "",
        device: str = "cpu",
        load_in_4bit: bool = False,
        few_shot: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        allowed_codes=None,
        note_max_chars: int = 0,
        use_fp16: bool = False,
        max_input_tokens=None,
        top_k: int = 15,
        label_catalog=None,
        closed_set: bool = False,
    ):
        self.few_shot = few_shot
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        if note_max_chars:
            self.MAX_NOTE_CHARS = int(note_max_chars)
        # Top-N task alignment: this was the ONLY E1-E14 pipeline never constrained to the
        # benchmark label space. Measured on the pilot: ~99% of its codes were dotted
        # ("I25.9") while gold is dot-less ("I2510"), and ~95% were outside Top-50 even
        # after dot-stripping -> F1 0.003.
        self.allowed_codes = set(allowed_codes) if allowed_codes else None
        # Closed-set mode: show the model the benchmark label space instead of asking it to
        # generate from the full ~16k ICD vocabulary and then discarding whatever falls outside
        # the Top-N set. In open-generation mode 97.4% (E7) and 88.0% (E8) of notes survived
        # label-space filtering with zero codes, so the resulting F1 measured prompt/label-space
        # mismatch rather than coding ability. `label_catalog` is a list of (code, title).
        self.label_catalog = list(label_catalog or [])
        self.closed_set = bool(closed_set and self.label_catalog)
        self.llm = LocalLLM(
            llm_path, device=device, load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False,
            use_fp16=use_fp16, max_input_tokens=max_input_tokens,
        )

    @staticmethod
    def _normalize_code(code: str) -> str:
        """MIMIC ICD-10 codes are dot-less and upper-case."""
        return str(code).strip().upper().replace(".", "").replace(" ", "")

    def process_note(self, note_text: str) -> List[Dict[str, Any]]:
        """
        Generate ICD-10 code predictions for a clinical note.

        Returns
        -------
        list of dicts with keys: code, title, rationale, confidence, supported.
        """
        truncated = note_text[:self.MAX_NOTE_CHARS]

        if self.closed_set:
            catalog = "\n".join(f"- {c}: {t}" for c, t in self.label_catalog)
            prompt = CLOSED_SET_PROMPT.format(label_catalog=catalog, note_text=truncated)
        elif self.few_shot:
            prompt = FEW_SHOT_PROMPT.format(
                examples=FEW_SHOT_EXAMPLES,
                note_text=truncated,
            )
        else:
            prompt = ZERO_SHOT_PROMPT.format(note_text=truncated)

        response = self.llm.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        try:
            parsed = parse_llm_json(response)
            code_list = parsed.get("icd10_codes", [])
            results = []
            seen = set()
            for item in code_list:
                if not isinstance(item, dict):
                    continue
                code = self._normalize_code(item.get("code", ""))
                if not code or code in seen:
                    continue
                # keep only codes inside the benchmark label space
                if self.allowed_codes and code not in self.allowed_codes:
                    continue
                seen.add(code)
                results.append({
                    "code": code,
                    "title": item.get("title", ""),
                    "rationale": item.get("rationale", ""),
                    "confidence": 0.8,  # LLM-only has no explicit confidence
                    "supported": True,
                    "risk_flag": "none",
                    "evidence_quote": "",
                })
            return results[: self.top_k]
        except Exception:
            return []

    def fit(self, X_train, Y_train, **kwargs):
        """No-op: LLM baselines are not trained."""
        pass

    def predict(self, X: List[str]) -> List[List[str]]:
        """Batch predict for evaluation compatibility."""
        results = []
        for text in X:
            preds = self.process_note(text)
            results.append([p["code"] for p in preds])
        return results
