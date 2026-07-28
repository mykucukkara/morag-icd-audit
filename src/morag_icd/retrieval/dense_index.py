"""
Dense (vector) retrieval index with FAISS and numpy fallback.

Supports:
- FAISS IndexFlatIP (preferred, GPU/CPU)
- numpy cosine similarity (fallback when FAISS is not installed)
- sentence-transformers for embedding generation
"""
from __future__ import annotations

import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..utils.model_readiness import is_resolved_local_path

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class DenseIndex:
    """
    Dense retrieval index using sentence-transformers embeddings.

    Falls back to numpy cosine similarity if FAISS is not available.
    """

    def __init__(self, model_name: str, device: str = "cpu", allow_mock_embedding: bool = False, embedding_dim: int = 64):
        """
        Parameters
        ----------
        model_name : str
            Path to a local sentence-transformers compatible model, or model name.
        device : str
            "cpu" or "cuda".
        """
        self.model_name = model_name
        self.device = device
        self.allow_mock_embedding = allow_mock_embedding
        self.embedding_dim = embedding_dim
        self.index = None          # FAISS index
        self.embeddings = None     # numpy fallback
        self.docs: List[Dict[str, Any]] = []
        self.dim: int = 0
        self._model: Optional[Any] = None
        self.mock_embedding = False

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            if self.allow_mock_embedding:
                self.mock_embedding = True
                return None
            if not is_resolved_local_path(self.model_name):
                raise FileNotFoundError(f"Local embedding model path does not exist: {self.model_name}")
            if not HAS_ST:
                raise ImportError(
                    "sentence-transformers is required for dense retrieval. "
                    "Install it with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _hash_embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        needed = self.embedding_dim * 4
        repeated = (digest * ((needed // len(digest)) + 1))[:needed]
        vec = np.frombuffer(repeated, dtype=np.uint32).astype(np.float32)
        vec = vec[: self.embedding_dim]
        vec = (vec % 10007) / 10007.0
        return vec

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.allow_mock_embedding or self.mock_embedding:
            self.mock_embedding = True
            return np.vstack([self._hash_embed(text) for text in texts]).astype(np.float32)
        if not HAS_ST:
            raise ImportError(
                "sentence-transformers is required for non-mock dense retrieval. "
                "Install it with: pip install sentence-transformers"
            )
        model = self.model
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
        return embs.astype(np.float32)

    def fit(self, docs: List[Dict[str, Any]], text_field: str = "searchable_text") -> None:
        """
        Build the index from a list of document dicts.

        Parameters
        ----------
        docs : list of dict
        text_field : str
            Key in each dict containing the text to embed.
        """
        self.docs = docs
        texts = [d.get(text_field, "") for d in docs]
        if not texts:
            self.embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self.dim = self.embedding_dim
            return

        embs = self._encode_texts(texts)
        self.dim = embs.shape[1]

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.dim)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs_norm = embs / np.maximum(norms, 1e-9)
            self.index.add(embs_norm)
            self.embeddings = embs_norm  # keep for save/load
        else:
            # numpy fallback: store normalized embeddings
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            self.embeddings = embs / np.maximum(norms, 1e-9)

    def get_scores(self, query: str) -> np.ndarray:
        """
        Compute similarity scores between a query and all indexed documents.

        Returns
        -------
        np.ndarray of shape (n_docs,) with cosine similarity scores.
        """
        if not self.docs:
            return np.array([])

        cache = getattr(self, "_score_cache", None)
        if cache is None:
            from collections import OrderedDict
            cache = self._score_cache = OrderedDict()
        key = query
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached.copy()

        q_emb = self._encode_texts([query]).astype(np.float32)
        q_norm = q_emb / np.maximum(np.linalg.norm(q_emb), 1e-9)

        if HAS_FAISS and self.index is not None:
            D, I = self.index.search(q_norm, len(self.docs))
            scores = np.zeros(len(self.docs))
            for i, idx in enumerate(I[0]):
                if 0 <= idx < len(self.docs):
                    scores[idx] = float(D[0][i])
        elif self.embeddings is not None:
            scores = (self.embeddings @ q_norm.T).squeeze()  # normalized -> cosine
        else:
            scores = np.zeros(len(self.docs))

        cache_size = getattr(self, "cache_size", 4096)
        if cache_size:
            cache[key] = scores
            cache.move_to_end(key)
            while len(cache) > cache_size:
                cache.popitem(last=False)
        return scores.copy()

    def save(self, filepath: str | Path) -> None:
        """
        Save the index and documents to disk.

        Saves:
          - <filepath>.index  (FAISS binary index, if available)
          - <filepath>.pkl    (docs list and numpy embeddings)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Store embeddings in the portable .npy format (NOT pickled): a pickled numpy
        # ndarray embeds the numpy module path (e.g. numpy._core) and fails to unpickle
        # across numpy major versions; the .npy format is stable across versions.
        npy_path = filepath.with_suffix(".emb.npy")
        embeddings_in_npy = self.embeddings is not None
        if embeddings_in_npy:
            np.save(npy_path, np.ascontiguousarray(self.embeddings))

        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump({
                "docs": self.docs,
                "embeddings": None,               # kept out of the pickle for portability
                "embeddings_in_npy": embeddings_in_npy,
                "dim": self.dim,
                "mock_embedding": self.mock_embedding,
            }, f)

        # Save FAISS index if available
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, str(filepath.with_suffix(".index")))

    def load(self, filepath: str | Path) -> None:
        """
        Load a saved index from disk.

        Tries to load FAISS index first; falls back to numpy embeddings.
        """
        filepath = Path(filepath)
        pkl_path = filepath.with_suffix(".pkl")

        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.docs = data.get("docs", [])
            self.dim = data.get("dim", 0)
            self.mock_embedding = bool(data.get("mock_embedding", False))
            # Prefer the portable .npy sidecar; fall back to embeddings embedded in the
            # pickle (legacy format, only readable under a compatible numpy version).
            npy_path = filepath.with_suffix(".emb.npy")
            if data.get("embeddings_in_npy") and npy_path.exists():
                self.embeddings = np.load(npy_path)
            else:
                self.embeddings = data.get("embeddings")

        index_path = filepath.with_suffix(".index")
        if HAS_FAISS and index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.index = None
            if not HAS_FAISS and self.embeddings is not None:
                print("Info: FAISS not available; using numpy cosine similarity fallback.")
