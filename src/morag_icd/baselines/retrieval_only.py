"""Retrieval-only baselines (E4, E5, E6): BM25-only, Dense-only, or Hybrid."""
from __future__ import annotations

from typing import List, Dict, Any


class RetrievalOnlyBaseline:
    """
    Predicts ICD-10 codes using retrieval alone (no LLM).
    Compatible with ExperimentRunner's process_note interface.
    """

    def __init__(self, icd_retriever, top_k: int = 10, allowed_codes=None):
        self.icd_retriever = icd_retriever
        self.top_k = top_k
        # Top-N task alignment: restrict predictions to the benchmark's label space.
        self.allowed_codes = set(allowed_codes) if allowed_codes else None

    def process_note(self, note_text: str) -> List[Dict[str, Any]]:
        """Return top-k ICD candidates as prediction dicts."""
        # retriever ranks inside the Top-N label space (mask held by ICDCandidateRetriever)
        candidates = self.icd_retriever.retrieve_candidates(note_text, self.top_k)
        if self.allowed_codes:
            candidates = [c for c in candidates if c.get("code") in self.allowed_codes]
        preds = []
        for c in candidates:
            if not c.get("code"):
                continue
            pred = {
                "code": c.get("code", ""),
                "confidence": float(c.get("score", 1.0)),
                "supported": None,
                "evidence_score": float(c.get("score", 0.0)),
                "icd_description": c.get("title") or c.get("long_title") or c.get("icd_title") or "",
                "evidence_preview": "",
                "rationale": "retrieval-only",
                "risk_flag": "retrieval_only",
            }
            if "skipped_dense" in c:
                pred["skipped_dense"] = bool(c.get("skipped_dense"))
            preds.append(pred)
        return preds

    def fit(self, *args, **kwargs):
        """No training needed."""
        pass

    def predict(self, texts: List[str]) -> List[List[str]]:
        """Batch predict for evaluation compatibility."""
        return [
            [cp["code"] for cp in self.process_note(t)]
            for t in texts
        ]
