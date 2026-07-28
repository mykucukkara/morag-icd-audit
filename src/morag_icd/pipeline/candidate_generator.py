"""
Candidate ICD-10 code generator.

Wraps the ICD retriever and applies section-weighted scoring to clinical notes
before generating candidate codes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class CandidateGenerator:
    """
    Generates candidate ICD-10 codes from a clinical note using a hybrid retriever.

    Supports section weighting: sections deemed clinically important (e.g.,
    Discharge Diagnosis) contribute more to the retrieval query.
    """

    SECTION_WEIGHT_DEFAULTS: Dict[str, float] = {
        "discharge_diagnosis": 3.0,
        "hospital_course": 2.0,
        "chief_complaint": 1.5,
        "history_of_present_illness": 1.25,
        "past_medical_history": 1.0,
        "other": 0.5,
    }

    def __init__(
        self,
        icd_retriever,
        section_weights: Optional[Dict[str, float]] = None,
        top_k: int = 50,
        allowed_codes: Optional[set] = None,
    ):
        """
        Parameters
        ----------
        icd_retriever : ICDCandidateRetriever
            Retriever for ICD KB.
        section_weights : dict, optional
            Override default section importance weights.
        top_k : int
            Number of candidate codes to return.
        """
        self.icd_retriever = icd_retriever
        self.top_k = top_k
        self.section_weights = {**self.SECTION_WEIGHT_DEFAULTS, **(section_weights or {})}
        # Task alignment: on a Top-N benchmark the candidate space must be the Top-N label
        # set (gold codes are restricted to it; trained baselines predict only within it).
        # Without this filter candidates come from the full ~97k-code KB and mostly can
        # never be correct.
        self.allowed_codes = set(allowed_codes) if allowed_codes else None

    def generate(
        self,
        note_text: str,
        sections: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate ICD-10 codes for a clinical note.

        Parameters
        ----------
        note_text : str
            Full discharge summary text.
        sections : dict, optional
            Pre-split sections {section_name: section_text}.

        Returns
        -------
        list of dicts with keys: code, title, score, searchable_text, ...
        """
        query = self._build_weighted_query(note_text, sections)
        # The retriever now ranks INSIDE the Top-N label space (ICDCandidateRetriever holds
        # the restrict mask), so no over-retrieve-then-filter is needed. The old approach
        # took top-1000 of ~97k codes and then kept the ~1% that were in-label-space,
        # yielding only 3.29 candidates/note and a hard candidate-recall ceiling of 0.107.
        candidates = self.icd_retriever.retrieve_candidates(query, self.top_k)
        if self.allowed_codes:
            # defensive: retriever may be unrestricted (e.g. mask unavailable)
            candidates = [c for c in candidates if c.get("code") in self.allowed_codes]
        return candidates

    def _build_weighted_query(
        self,
        note_text: str,
        sections: Optional[Dict[str, str]],
    ) -> str:
        """
        Build a query string that emphasizes important sections by repetition.
        High-weight sections appear more times in the query, boosting their
        term frequency in BM25 scoring.
        """
        if not sections:
            return note_text[:2000]  # Truncate raw text

        query_parts = []
        for section_name, text in sections.items():
            normalized = section_name.lower().replace(" ", "_").replace("-", "_")
            weight = self.section_weights.get(normalized, 0.5)
            # Repeat section text proportionally to weight
            repeat = max(1, round(weight))
            if text.strip():
                query_parts.extend([text[:500]] * repeat)

        return " ".join(query_parts)

    @classmethod
    def from_config(cls, icd_retriever, config: Dict) -> "CandidateGenerator":
        """Construct from a config dict (retrieval.yaml merged with hp_config)."""
        section_weights = {
            "discharge_diagnosis": config.get("section_weight_discharge_diagnosis", 3.0),
            "hospital_course": config.get("section_weight_hospital_course", 2.0),
            "past_medical_history": config.get("section_weight_past_medical_history", 1.0),
        }
        return cls(
            icd_retriever=icd_retriever,
            section_weights=section_weights,
            top_k=config.get("top_k_icd_candidates", 50),
            allowed_codes=config.get("allowed_codes"),
        )
