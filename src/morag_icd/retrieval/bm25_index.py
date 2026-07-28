from typing import List, Dict, Any
import numpy as np
from collections import Counter, OrderedDict
import math
import pickle
from pathlib import Path

class BM25:
    def __init__(self, k1=1.5, b=0.75, cache_size: int = 4096):
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.docs = []
        self.cache_size = cache_size
        # lazily-built acceleration structures (also rebuilt after unpickling old indexes)
        self._postings = None      # token -> (np.int32 doc indices, np.float32 freqs)
        self._den_norm = None      # per-doc k1*(1 - b + b*doc_len/avgdl)
        self._score_cache = None   # OrderedDict LRU: query str -> scores ndarray
        self._prepared = False

    def fit(self, docs: List[Dict[str, Any]], text_field: str):
        self.docs = docs
        nd = len(docs)
        num_doc = 0
        df = {}
        for d in docs:
            tokens = d[text_field].lower().split()
            self.doc_len.append(len(tokens))
            num_doc += len(tokens)
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                df[word] = df.get(word, 0) + 1

        self.avgdl = num_doc / nd if nd > 0 else 0

        for word, freq in df.items():
            self.idf[word] = math.log(1 + (nd - freq + 0.5) / (freq + 0.5))
        self._postings = None
        self._den_norm = None
        self._score_cache = None

    # Tokens whose IDF is below this contribute negligibly to BM25 (they appear in almost
    # every doc, e.g. "patient"/"the" in a note-chunk corpus). Skipping them in the inverted
    # index makes the build/queries tractable on large corpora with long docs, with a score
    # change < IDF_MIN per token (below ranking resolution).
    IDF_MIN = 1e-4
    # Build the inverted index only when the corpus is cheap enough to invert (short docs,
    # e.g. the ICD KB). For huge/long-doc corpora (e.g. a global note-chunk evidence index)
    # inverting all tokens is prohibitive, so fall back to the original per-query scan.
    BUILD_TOKEN_BUDGET = 8_000_000

    def _ensure_fast(self):
        """Prepare acceleration once (lazily, incl. after unpickle). May choose the scan fallback."""
        if getattr(self, "_prepared", False):
            return
        n = len(self.doc_len)
        avgdl = self.avgdl or 1.0
        self._den_norm = np.array(
            [self.k1 * (1 - self.b + self.b * dl / avgdl) for dl in self.doc_len], dtype=np.float32
        )
        if getattr(self, "_score_cache", None) is None:
            self._score_cache = OrderedDict()

        total_tokens = int(sum(self.doc_len))
        if total_tokens > self.BUILD_TOKEN_BUDGET:
            self._postings = None            # scan fallback (still cached per query)
            self._prepared = True
            return

        keep = {t for t, v in self.idf.items() if v >= self.IDF_MIN}
        idx_lists: Dict[str, list] = {}
        frq_lists: Dict[str, list] = {}
        for i, freqs in enumerate(self.doc_freqs):
            for token, f in freqs.items():
                if token not in keep:
                    continue
                idx_lists.setdefault(token, []).append(i)
                frq_lists.setdefault(token, []).append(f)
        self._postings = {
            token: (np.asarray(idxs, dtype=np.int32), np.asarray(frq_lists[token], dtype=np.float32))
            for token, idxs in idx_lists.items()
        }
        self._prepared = True

    def _scan_scores(self, tokens) -> np.ndarray:
        """Original per-document BM25 scan (fallback for corpora too large to invert)."""
        scores = np.zeros(len(self.docs), dtype=np.float32)
        k1, b, avgdl = self.k1, self.b, (self.avgdl or 1.0)
        toks = [t for t in set(tokens) if self.idf.get(t, 0.0) >= self.IDF_MIN]
        for i in range(len(self.docs)):
            dl = self.doc_len[i]
            fr = self.doc_freqs[i]
            s = 0.0
            for t in toks:
                f = fr.get(t)
                if f:
                    s += self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
            scores[i] = s
        return scores

    def get_scores(self, query: str) -> np.ndarray:
        n = len(self.docs)
        if n == 0:
            return np.zeros(0)
        self._ensure_fast()
        cache = self._score_cache
        key = query.lower()
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached.copy()

        tokens = key.split()
        if self._postings is None:
            scores = self._scan_scores(tokens)          # fallback for large corpora
        else:
            scores = np.zeros(n, dtype=np.float32)
            k1p1 = self.k1 + 1.0
            for token in set(tokens):
                post = self._postings.get(token)
                if post is None:
                    continue
                idf = self.idf.get(token, 0.0)
                if idf < self.IDF_MIN:
                    continue
                idxs, freqs = post
                num = idf * freqs * k1p1
                den = freqs + self._den_norm[idxs]
                np.add.at(scores, idxs, num / den)

        if self.cache_size:
            cache[key] = scores
            cache.move_to_end(key)
            while len(cache) > self.cache_size:
                cache.popitem(last=False)
        return scores.copy()

    def save(self, filepath: str | Path):
        # Do not pickle the (rebuildable) acceleration structures or the cache.
        postings, den_norm, cache = self._postings, self._den_norm, self._score_cache
        self._postings = self._den_norm = self._score_cache = None
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        finally:
            self._postings, self._den_norm, self._score_cache = postings, den_norm, cache

    @classmethod
    def load(cls, filepath: str | Path):
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        # old pickles won't have the new attributes; normalize so _ensure_fast() rebuilds them
        obj._postings = None
        obj._den_norm = None
        obj._score_cache = None
        obj._prepared = False
        if not hasattr(obj, "cache_size"):
            obj.cache_size = 4096
        return obj
