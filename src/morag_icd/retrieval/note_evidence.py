"""
Note-local clinical evidence retrieval.

Evidence supporting an ICD code assignment for an admission MUST come from THAT
admission's own note (as in MDACE / Code Like Humans and any explainable ICD coder),
not from a global pool of all patients' chunks. This module builds a tiny per-note
retriever over the current note's chunks (BM25 + optional dense via a shared embedder),
which is both scientifically correct and trivially fast (a note has ~5-50 chunks).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .bm25_index import BM25


class NoteLocalEvidenceRetriever:
    """Retrieve supporting evidence for a code query from the current note's own chunks."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embedder: Optional[Any] = None,
        alpha: float = 0.5,
        mode: str = "hybrid",
    ):
        """
        Parameters
        ----------
        chunks : list of {chunk_id, section_name, text}
            The current note's chunks.
        embedder : object with ._encode_texts(list[str]) -> np.ndarray, optional
            Shared embedding backend (e.g. a loaded DenseIndex). None -> BM25-only.
        alpha : float
            Hybrid weight: score = alpha*bm25 + (1-alpha)*dense.
        mode : str
            "bm25" | "dense" | "hybrid".
        """
        self.chunks = chunks or []
        self.mode = mode
        self.alpha = float(alpha)
        self.embedder = embedder
        self.dense_used = False
        self.skipped_dense = False

        self.bm25 = BM25()
        if self.chunks:
            self.bm25.fit([{"searchable_text": c.get("text", "")} | c for c in self.chunks], "searchable_text")

        self._chunk_emb = None
        if mode in ("dense", "hybrid") and self.embedder is not None and self.chunks:
            try:
                emb = np.asarray(
                    self.embedder._encode_texts([c.get("text", "") for c in self.chunks]),
                    dtype=np.float32,
                )
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                self._chunk_emb = emb / np.maximum(norms, 1e-9)
                self.dense_used = True
            except Exception:
                self._chunk_emb = None
                self.skipped_dense = mode == "hybrid"
        elif mode in ("dense", "hybrid"):
            # no embedder available -> hybrid degrades to BM25, dense-only is unavailable
            self.skipped_dense = mode == "hybrid"

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.atleast_1d(np.asarray(v, dtype=np.float32))
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    def retrieve_evidence(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve_evidence_batch([query], top_k)[0]

    def retrieve_evidence_batch(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Retrieve evidence for MANY code queries at once.

        The batched design asks about every Top-N code per note, so the dense side must
        embed all candidate queries in ONE encoder call — doing it per candidate meant 50
        sequential sentence-transformer forwards per note (measured: 33 s/note).
        """
        n = len(self.chunks)
        if n == 0:
            return [[] for _ in queries]

        dense_all = None
        if self.mode != "bm25" and self._chunk_emb is not None:
            q = np.asarray(self.embedder._encode_texts(list(queries)), dtype=np.float32)
            q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
            dense_all = self._chunk_emb @ q.T          # (n_chunks, n_queries)

        results: List[List[Dict[str, Any]]] = []
        for qi, query in enumerate(queries):
            bm = self._norm(self.bm25.get_scores(query) if self.bm25.docs else np.zeros(n))
            if dense_all is None:
                combined = bm
            else:
                dense = self._norm(dense_all[:, qi])
                combined = dense if self.mode == "dense" else self.alpha * bm + (1.0 - self.alpha) * dense

            order = np.argsort(combined)[::-1][: max(1, int(top_k))]
            out: List[Dict[str, Any]] = []
            for idx in order:
                i = int(idx)
                c = self.chunks[i]
                out.append({
                    "text": c.get("text", ""),
                    "score": float(combined[i]),
                    "chunk_id": c.get("chunk_id", f"chunk_{i}"),
                    "section_name": c.get("section_name", ""),
                })
            results.append(out)
        return results
