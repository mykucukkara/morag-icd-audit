"""
Optimizer runner: orchestrates Random Search, MOPSO, and NSGA-II.

Builds the pipeline factory from config, loads the validation subset,
runs the selected optimizer, and saves the Pareto front results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .search_space import SearchSpace
from .random_search import RandomSearch
from .nsga2 import run_nsga2
from .mopso import run_mopso
from .pareto import hypervolume_mc, find_pareto_front


# Hyperparameters that actually influence inference (query-time). Others in the YAML
# search space (chunk_size/chunk_overlap: index-build only; section_weight_*: sections are
# never passed to the candidate generator; prompt_template_id: no template variants exist)
# have NO effect at inference and must not be part of a meaningful optimization.
EFFECTIVE_HP_KEYS = [
    "bm25_dense_alpha",
    "top_k_evidence",
    "top_k_icd_candidates",
    "evidence_similarity_threshold",
    "llm_confidence_threshold",
    "max_final_codes",
    "prompt_template_id",  # only effective once real template variants exist
]


def build_pipeline_factory(
    base_config: Dict,
    retrieval_config: Dict,
    model_config: Dict,
    paths_config: Dict,
    top_n_suffix: str,
) -> Callable:
    """
    Create factory(hp_config) -> RAGPipeline.

    Correctness: the pipeline is cached on the FULL set of query-time-effective
    hyperparameters (EFFECTIVE_HP_KEYS), so each distinct config produces a distinct
    pipeline (previously it was cached on bm25_dense_alpha alone, which silently made
    ~11/12 hyperparameters inert during optimization).

    Efficiency: heavy resources (BM25 + dense indexes and the local LLM) are loaded ONCE
    and shared across every config; only the lightweight per-config wiring (hybrid alpha,
    candidate/evidence top-k and thresholds, scorer/verifier config) is rebuilt.
    """
    import hashlib

    allow_mock_llm = bool(base_config.get("allow_mock_llm", False))
    allow_mock_embedding = bool(base_config.get("allow_mock_embedding", False))

    _resources: Dict[str, Any] = {}
    _pipeline_cache: Dict[str, Any] = {}

    def _resources_loaded():
        if "r" in _resources:
            return _resources["r"]
        _resources["r"] = _load_optimizer_resources(
            model_config=model_config,
            paths_config=paths_config,
            top_n_suffix=top_n_suffix,
            allow_mock_llm=allow_mock_llm,
            allow_mock_embedding=allow_mock_embedding,
        )
        return _resources["r"]

    def factory(hp_config: Dict):
        cfg = {**retrieval_config, **base_config, **hp_config}
        key_dict = {k: cfg.get(k) for k in EFFECTIVE_HP_KEYS}
        cache_hash = hashlib.md5(
            json.dumps(key_dict, sort_keys=True, default=str).encode()
        ).hexdigest()
        if cache_hash not in _pipeline_cache:
            _pipeline_cache[cache_hash] = _assemble_optimizer_pipeline(cfg, _resources_loaded())
        return _pipeline_cache[cache_hash]

    return factory


def _load_optimizer_resources(
    model_config: Dict,
    paths_config: Dict,
    top_n_suffix: str,
    allow_mock_llm: bool,
    allow_mock_embedding: bool,
) -> Dict[str, Any]:
    """Load the heavy, config-independent resources (indexes + LLM) exactly once."""
    from ..retrieval.bm25_index import BM25
    from ..llm.local_llm import LocalLLM
    from ..pipeline.full_model import _load_or_mock_dense, _create_empty_bm25

    indexes_dir = Path(paths_config.get("indexes_dir", "indexes"))
    ev_bm25_path = indexes_dir / "bm25" / f"evidence_bm25_{top_n_suffix}.pkl"
    icd_bm25_path = indexes_dir / "bm25" / f"icd_bm25_{top_n_suffix}.pkl"

    if not allow_mock_llm and not allow_mock_embedding:
        required = [
            ev_bm25_path, icd_bm25_path,
            indexes_dir / "faiss" / f"icd_dense_{top_n_suffix}.pkl",
            indexes_dir / "faiss" / f"evidence_dense_{top_n_suffix}.pkl",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Optimizer requires non-mock BM25+dense indexes for top{top_n_suffix}. "
                f"Missing: {', '.join(missing)}"
            )

    ev_bm25 = BM25.load(ev_bm25_path) if ev_bm25_path.exists() else _create_empty_bm25()
    icd_bm25 = BM25.load(icd_bm25_path) if icd_bm25_path.exists() else _create_empty_bm25()

    embed_path = model_config.get("embedding_model_path", "all-MiniLM-L6-v2")
    device = model_config.get("device", model_config.get("fallback_device", "cpu"))
    ev_dense, ev_mock = _load_or_mock_dense(
        indexes_dir / "faiss" / f"evidence_dense_{top_n_suffix}", ev_bm25.docs,
        embed_path, device, allow_mock_embedding)
    icd_dense, icd_mock = _load_or_mock_dense(
        indexes_dir / "faiss" / f"icd_dense_{top_n_suffix}", icd_bm25.docs,
        embed_path, device, allow_mock_embedding)

    llm = LocalLLM(
        model_config.get("llm_model_path", ""),
        device=device,
        load_in_4bit=model_config.get("load_in_4bit", False),
        allow_mock_llm=allow_mock_llm,
        max_new_tokens=model_config.get("max_new_tokens", 512),
        temperature=model_config.get("temperature", 0.0),
        do_sample=False,
        use_fp16=bool(model_config.get("use_fp16", False)),
    )
    from ..pipeline.full_model import _load_label_space
    return {
        "ev_bm25": ev_bm25, "icd_bm25": icd_bm25,
        "ev_dense": ev_dense, "icd_dense": icd_dense,
        "mock_embedding": bool(ev_mock or icd_mock),
        "dense_available": icd_dense is not None,
        "allowed_codes": _load_label_space(paths_config, str(top_n_suffix)),
        "llm": llm,
    }


def _assemble_optimizer_pipeline(cfg: Dict, res: Dict[str, Any]):
    """Build a per-config RAGPipeline (E14: evidence + contrastive) reusing shared resources."""
    from ..retrieval.hybrid_retriever import HybridRetriever
    from ..retrieval.evidence_retriever import ClinicalEvidenceRetriever
    from ..retrieval.icd_retriever import ICDCandidateRetriever
    from ..llm.code_scorer import CodeScorer
    from ..llm.contrastive_verifier import ContrastiveVerifier
    from ..pipeline.candidate_generator import CandidateGenerator
    from ..pipeline.evidence_extractor import EvidenceExtractor
    from ..pipeline.rag_pipeline import RAGPipeline

    alpha = float(cfg.get("bm25_dense_alpha", 0.5))
    dense_used = bool(res["dense_available"])
    ev_hybrid = HybridRetriever(res["ev_bm25"], res["ev_dense"], alpha=alpha)
    icd_hybrid = HybridRetriever(res["icd_bm25"], res["icd_dense"], alpha=alpha)
    evidence_retriever = ClinicalEvidenceRetriever(ev_hybrid)
    icd_retriever = ICDCandidateRetriever(icd_hybrid, allowed_codes=res.get("allowed_codes"))

    llm = res["llm"]
    code_scorer = CodeScorer(llm, cfg)
    contrastive_verifier = ContrastiveVerifier(llm, cfg)

    pipeline_config = {
        "experiment_id": "E14",
        "retrieval_mode": "hybrid",
        "top_k_icd_candidates": int(cfg.get("top_k_icd_candidates", 50)),
        "top_k_evidence": int(cfg.get("top_k_evidence", 5)),
        "max_final_codes": int(cfg.get("max_final_codes", 10)),
        "evidence_similarity_threshold": float(cfg.get("evidence_similarity_threshold", 0.3)),
        "llm_confidence_threshold": float(cfg.get("llm_confidence_threshold", 0.5)),
        "evidence_constraint_mode": cfg.get("evidence_constraint_mode", "flag"),
        "use_evidence_constraint": True,
        "use_contrastive_verifier": True,
        "prompt_template_id": cfg.get("prompt_template_id"),
        "note_max_chars": cfg.get("note_max_chars"),
        "prompt_max_chars": cfg.get("prompt_max_chars"),
        "evidence_snippet_max_chars": cfg.get("evidence_snippet_max_chars"),
        "bm25_dense_alpha": alpha,
        "chunk_size": cfg.get("chunk_size", 256),
        "chunk_overlap": cfg.get("chunk_overlap", 32),
        "legacy_per_candidate": bool(cfg.get("legacy_per_candidate", False)),
        "scoring_mode": "legacy_per_candidate" if cfg.get("legacy_per_candidate", False) else "batched_per_note",
        "evidence_note_local": True,
        "allowed_codes": res.get("allowed_codes"),
        "mock_llm": bool(getattr(llm, "is_mock", False)),
        "mock_embedding": bool(res["mock_embedding"]),
        "dense_used": dense_used,
        "skipped_dense": not dense_used,
    }
    from ..data.section_splitter import process_note_into_chunks
    cs = int(cfg.get("chunk_size") or 256)
    co = int(cfg.get("chunk_overlap") or 32)

    def _chunker(text: str):
        return process_note_into_chunks(text, chunk_size=cs, chunk_overlap=co)

    candidate_generator = CandidateGenerator.from_config(icd_retriever, pipeline_config)
    evidence_extractor = EvidenceExtractor.from_config(evidence_retriever, pipeline_config)
    pipeline = RAGPipeline(
        evidence_retriever=evidence_retriever,
        icd_retriever=icd_retriever,
        code_scorer=code_scorer,
        candidate_generator=candidate_generator,
        evidence_extractor=evidence_extractor,
        contrastive_verifier=contrastive_verifier,
        config=pipeline_config,
        embedder=res["icd_dense"],
        chunker=_chunker,
    )
    pipeline.run_metadata = {k: v for k, v in pipeline_config.items() if k != "allowed_codes"}
    return pipeline


def run_optimizer(
    optimizer_name: str,
    pipeline_factory: Callable,
    dataset: List[Dict],
    opt_config: Dict,
    label_set: Optional[List[str]] = None,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> Dict:
    """
    Run the specified optimizer.

    Parameters
    ----------
    optimizer_name : str
        One of "random_search", "mopso", "nsga2".
    pipeline_factory : callable
        factory(hp_config) -> pipeline.
    dataset : list of dict
        Validation subset samples.
    opt_config : dict
        Optimizer configuration (from configs/optimization.yaml).
    label_set : list of str, optional
    seed : int
    output_path : str | Path, optional

    Returns
    -------
    dict with optimizer results including pareto_front and best_compromise.
    """
    search_space = SearchSpace.from_config(opt_config)

    if optimizer_name == "random_search":
        rs = RandomSearch(
            search_space=search_space,
            n_trials=opt_config.get("random_search_params", {}).get("n_trials", 100),
            seed=seed,
        )
        return rs.run(pipeline_factory, dataset, label_set=label_set, output_path=output_path)

    elif optimizer_name == "mopso":
        return run_mopso(
            pipeline_factory=pipeline_factory,
            dataset=dataset,
            config=opt_config,
            search_space=search_space,
            seed=seed,
            label_set=label_set,
            output_path=output_path,
        )

    elif optimizer_name == "nsga2":
        return run_nsga2(
            pipeline_factory=pipeline_factory,
            dataset=dataset,
            config=opt_config,
            search_space=search_space,
            seed=seed,
            label_set=label_set,
            output_path=output_path,
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}. Choose from: random_search, mopso, nsga2")
