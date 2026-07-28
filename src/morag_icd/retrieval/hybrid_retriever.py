import numpy as np
from typing import List, Dict, Any
import logging

from .bm25_index import BM25
from .dense_index import DenseIndex

class HybridRetriever:
    def __init__(self, bm25_index: BM25, dense_index: DenseIndex, alpha: float):
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
        
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores is None or len(scores) == 0:
            return np.array([])
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val == min_val:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def _safe_bm25_scores(self, query: str) -> np.ndarray:
        if self.bm25_index is None or not getattr(self.bm25_index, "docs", None):
            return np.array([])
        return self.bm25_index.get_scores(query)

    def _safe_dense_scores(self, query: str) -> np.ndarray:
        if self.dense_index is None or not getattr(self.dense_index, "docs", None):
            return np.array([])
        return self.dense_index.get_scores(query)

    def retrieve(self, query: str, top_k: int, restrict_indices=None) -> List[Dict[str, Any]]:
        """Rank documents for `query`.

        restrict_indices: optional array of doc indices to rank WITHIN. On a Top-N
        benchmark this must be the Top-N code rows, so ranking happens inside the label
        space instead of over the whole ~97k-code KB and then post-filtering (which
        capped candidate recall at 0.107).
        """
        # alpha==1.0 is BM25-only: do NOT touch the dense index (it needs the embedding
        # model / CUDA and crashed CPU-only E4 on every sample).
        bm25_scores = self._safe_bm25_scores(query) if self.alpha != 0.0 else np.array([])
        dense_scores = self._safe_dense_scores(query) if self.alpha != 1.0 else np.array([])

        if self.alpha == 1.0:
            if len(bm25_scores) == 0:
                self.logger.warning("BM25-only retrieval requested but BM25 index is empty.")
                return []
            hybrid_scores = self._normalize(bm25_scores)
            docs = self.bm25_index.docs
            skipped_dense = True
        elif self.alpha == 0.0:
            if len(dense_scores) == 0:
                self.logger.warning("Dense-only retrieval requested but dense index is unavailable; returning empty.")
                return []
            hybrid_scores = self._normalize(dense_scores)
            docs = self.dense_index.docs
            skipped_dense = False
        else:
            if len(bm25_scores) == 0 and len(dense_scores) == 0:
                self.logger.warning("Hybrid retrieval requested but both BM25 and dense indexes are empty.")
                return []
            if len(dense_scores) == 0:
                self.logger.warning("Dense index unavailable in hybrid mode; using BM25 fallback.")
                hybrid_scores = self._normalize(bm25_scores)
                docs = self.bm25_index.docs
                skipped_dense = True
            elif len(bm25_scores) == 0:
                self.logger.warning("BM25 index unavailable in hybrid mode; using dense-only fallback.")
                hybrid_scores = self._normalize(dense_scores)
                docs = self.dense_index.docs
                skipped_dense = False
            else:
                n = min(len(bm25_scores), len(dense_scores))
                bm25_scores = bm25_scores[:n]
                dense_scores = dense_scores[:n]
                docs = self.bm25_index.docs[:n]
                skipped_dense = False
                norm_bm25 = self._normalize(bm25_scores)
                norm_dense = self._normalize(dense_scores)
                hybrid_scores = self.alpha * norm_bm25 + (1 - self.alpha) * norm_dense

        if len(hybrid_scores) == 0:
            return []

        if restrict_indices is not None:
            cand = np.asarray(restrict_indices, dtype=np.int64)
            cand = cand[cand < len(hybrid_scores)]
            if len(cand) == 0:
                return []
            order = cand[np.argsort(hybrid_scores[cand])[::-1]][:top_k]
            top_indices = order
        else:
            top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = docs[idx].copy()
            doc['score'] = float(hybrid_scores[idx])
            if skipped_dense:
                doc['skipped_dense'] = True
            results.append(doc)
            
        return results
