import numpy as np

from .hybrid_retriever import HybridRetriever


class ICDCandidateRetriever:
    def __init__(self, retriever: HybridRetriever, allowed_codes=None):
        self.retriever = retriever
        self._allowed_codes = set(allowed_codes) if allowed_codes else None
        self._restrict_indices = None
        if self._allowed_codes:
            self._restrict_indices = self._build_restrict_indices()

    def _index_docs(self):
        """Docs backing the ranked score vector (BM25 side unless dense-only)."""
        r = self.retriever
        if getattr(r, "alpha", 0.5) == 0.0 and getattr(r, "dense_index", None) is not None:
            return getattr(r.dense_index, "docs", []) or []
        if getattr(r, "bm25_index", None) is not None and getattr(r.bm25_index, "docs", None):
            return r.bm25_index.docs
        if getattr(r, "dense_index", None) is not None:
            return getattr(r.dense_index, "docs", []) or []
        return []

    def _build_restrict_indices(self):
        """Row indices of the Top-N codes, so ranking happens INSIDE the label space."""
        docs = self._index_docs()
        idx = [i for i, d in enumerate(docs) if d.get("code") in self._allowed_codes]
        return np.asarray(idx, dtype=np.int64) if idx else None

    def retrieve_candidates(self, clinical_text: str, top_k: int) -> list[dict]:
        if self._restrict_indices is not None:
            return self.retriever.retrieve(clinical_text, top_k, restrict_indices=self._restrict_indices)
        return self.retriever.retrieve(clinical_text, top_k)
