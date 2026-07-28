"""
Clinical evidence extractor.

Given a candidate ICD-10 code and a clinical note, retrieves the most relevant
evidence chunks from the note using the evidence retriever. Applies
similarity threshold filtering and returns ranked, scored evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class EvidenceExtractor:
    """
    Extracts relevant clinical evidence chunks for a given ICD-10 candidate code.

    Uses the evidence retriever to find note chunks that support or refute
    the candidate code, filtered by a similarity threshold.
    """

    def __init__(
        self,
        evidence_retriever,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ):
        """
        Parameters
        ----------
        evidence_retriever : ClinicalEvidenceRetriever
        top_k : int
            Number of evidence chunks to retrieve per code.
        similarity_threshold : float
            Minimum retrieval score to accept a chunk. Set 0.0 to disable.
        """
        self.evidence_retriever = evidence_retriever
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def extract(
        self,
        candidate_code: str,
        candidate_title: str,
        candidate_description: str,
        note_chunks: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and filter evidence chunks for a candidate ICD-10 code.

        Parameters
        ----------
        candidate_code : str
            ICD-10 code being evaluated (e.g., "I50.9").
        candidate_title : str
            Code title (e.g., "Heart failure, unspecified").
        candidate_description : str
            Full searchable text for the code.
        note_chunks : list of dict, optional
            Pre-chunked note. If None, evidence retriever will use its index.

        Returns
        -------
        list of dicts with keys: text, score, chunk_id, section_name (if available).
        """
        query = self._build_query(candidate_code, candidate_title, candidate_description)
        raw_chunks = self.evidence_retriever.retrieve_evidence(query, self.top_k)

        # Filter by similarity threshold
        filtered = [
            chunk for chunk in raw_chunks
            if chunk.get("score", 0.0) >= self.similarity_threshold
        ]

        # Ensure chunk_id present
        for i, chunk in enumerate(filtered):
            if "chunk_id" not in chunk:
                chunk["chunk_id"] = f"chunk_{i}"

        return filtered

    def extract_combined_text(
        self,
        candidate_code: str,
        candidate_title: str,
        candidate_description: str,
        max_chars: int = 2000,
    ) -> str:
        """
        Extract evidence and return as a single concatenated string for LLM prompts.
        """
        chunks = self.extract(candidate_code, candidate_title, candidate_description)
        parts = []
        total_chars = 0
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            if total_chars + len(text) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            total_chars += len(text)
        return "\n\n".join(parts)

    @staticmethod
    def _build_query(code: str, title: str, description: str) -> str:
        """Build a retrieval query from ICD code metadata."""
        parts = [f"ICD-10 {code}: {title}"]
        if description and description not in title:
            parts.append(description[:300])
        return " ".join(parts)

    @classmethod
    def from_config(cls, evidence_retriever, config: Dict) -> "EvidenceExtractor":
        """Construct from a config dict."""
        return cls(
            evidence_retriever=evidence_retriever,
            top_k=config.get("top_k_evidence", 5),
            similarity_threshold=config.get("evidence_similarity_threshold", 0.0),
        )
