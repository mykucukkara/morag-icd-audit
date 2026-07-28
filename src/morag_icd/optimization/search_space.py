"""
Search space definition and sampling utilities for RAG hyperparameter optimization.

Defines the discrete search space as specified in configs/optimization.yaml and
provides functions to sample, encode, and decode configurations.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Default search space (can be overridden from YAML config)
DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "chunk_size": [128, 256, 384, 512],
    "chunk_overlap": [0, 32, 64, 128],
    "top_k_evidence": [3, 5, 8, 10],
    "top_k_icd_candidates": [20, 30, 50, 75],
    "bm25_dense_alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
    "section_weight_discharge_diagnosis": [1.0, 1.5, 2.0, 3.0],
    "section_weight_hospital_course": [1.0, 1.25, 1.5, 2.0],
    "section_weight_past_medical_history": [0.5, 1.0, 1.25],
    "llm_confidence_threshold": [0.3, 0.5, 0.7],
    "evidence_similarity_threshold": [0.2, 0.35, 0.5],
    "max_final_codes": [5, 10, 15, 20],
    "prompt_template_id": ["p1", "p2", "p3"],
}

# Keys that can be treated as continuous (for PSO/NSGA-II encoding)
CONTINUOUS_KEYS = [
    "bm25_dense_alpha",
    "llm_confidence_threshold",
    "evidence_similarity_threshold",
]


class SearchSpace:
    """Manages the discrete hyperparameter search space."""

    def __init__(self, space: Optional[Dict[str, List[Any]]] = None):
        self.space = space or DEFAULT_SEARCH_SPACE
        self.keys = list(self.space.keys())
        self.sizes = [len(self.space[k]) for k in self.keys]
        self.n_dims = len(self.keys)

    @classmethod
    def from_config(cls, config: Dict) -> "SearchSpace":
        """Build search space from optimization config dict."""
        raw = config.get("search_space", {})
        if not raw:
            return cls()
        return cls(space=raw)

    def sample_random(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Sample one random configuration from the search space."""
        rng = rng or random
        return {k: rng.choice(v) for k, v in self.space.items()}

    def sample_random_batch(self, n: int, seed: int = 42) -> List[Dict[str, Any]]:
        """Sample n random configurations."""
        rng = random.Random(seed)
        return [self.sample_random(rng) for _ in range(n)]

    def encode(self, config: Dict[str, Any]) -> np.ndarray:
        """
        Encode a config dict as a real-valued vector in [0, 1]^n.

        Each dimension is the normalized index of the chosen value in its list.
        This allows PSO/NSGA-II to operate in a continuous space.
        """
        vec = np.zeros(self.n_dims)
        for i, k in enumerate(self.keys):
            vals = self.space[k]
            if config[k] in vals:
                idx = vals.index(config[k])
            else:
                # Find nearest for continuous approximation
                try:
                    dists = [abs(float(v) - float(config[k])) for v in vals]
                    idx = int(np.argmin(dists))
                except (TypeError, ValueError):
                    idx = 0
            # Normalize to [0, 1]
            vec[i] = idx / max(len(vals) - 1, 1)
        return vec

    def decode(self, vec: np.ndarray) -> Dict[str, Any]:
        """
        Decode a real-valued vector back to a config dict.

        Each dimension is mapped to the nearest valid value.
        """
        config = {}
        for i, k in enumerate(self.keys):
            vals = self.space[k]
            # Map [0,1] back to index
            raw_idx = vec[i] * (len(vals) - 1)
            idx = int(np.clip(round(raw_idx), 0, len(vals) - 1))
            config[k] = vals[idx]
        return config

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower_bounds, upper_bounds) arrays for the encoded space."""
        return np.zeros(self.n_dims), np.ones(self.n_dims)

    def config_hash(self, config: Dict[str, Any]) -> str:
        """Produce a short deterministic hash for a config."""
        import hashlib
        s = json_safe_str(config)
        return hashlib.md5(s.encode()).hexdigest()[:8]


def json_safe_str(d: Dict) -> str:
    """Convert dict to a sorted, JSON-safe string for hashing."""
    import json
    return json.dumps(d, sort_keys=True, default=str)
