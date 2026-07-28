"""
Full model factory: assembles all pipeline components from configs.

This module provides the build_full_model() function that constructs the
complete RAGPipeline with evidence retriever, ICD retriever, LLM scorer,
and contrastive verifier — all configured from YAML config dicts.

Also provides experiment-type-specific pipeline builders for E1–E18.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.model_readiness import is_resolved_local_path


def _load_label_space(paths_config: Dict, top_n_suffix: str):
    """Top-N label set (the benchmark's code space) from splits_root/topN/label_set.json."""
    splits_root = paths_config.get("splits_root") or str(
        Path(paths_config.get("project_root", ".")) / "data" / "splits"
    )
    p = Path(splits_root) / f"top{top_n_suffix}" / "label_set.json"
    if p.exists():
        try:
            codes = json.loads(p.read_text(encoding="utf-8"))
            return {str(c) for c in codes} or None
        except Exception:
            return None
    return None


def _load_label_catalog(paths_config: Dict, allowed_codes):
    """[(code, title)] for the benchmark label space, read from the ICD knowledge base.

    Used by the closed-set LLM-only arm so the model can select from the benchmark's codes
    instead of generating freely over the full ICD vocabulary. Falls back to bare codes when
    the KB is unavailable, and preserves the label set's order for a stable prompt.
    """
    if not allowed_codes:
        return None
    kb_path = paths_config.get("icd_kb_path") or ""
    titles = {}
    if kb_path and Path(kb_path).exists():
        try:
            with open(kb_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    c = str(e.get("code", "")).strip()
                    if c:
                        titles[c] = e.get("title") or e.get("long_title") or ""
        except Exception:
            titles = {}
    return [(c, titles.get(c, "")) for c in sorted(allowed_codes)]


def build_full_model(
    retrieval_config: Dict,
    model_config: Dict,
    paths_config: Dict,
    hp_config: Optional[Dict] = None,
    use_contrastive: bool = True,
    use_evidence_constraint: bool = True,
):
    """
    Build the complete RAGPipeline (E14 configuration).

    Parameters
    ----------
    retrieval_config : dict
        From configs/retrieval.yaml.
    model_config : dict
        From configs/models.yaml.
    paths_config : dict
        From configs/paths.yaml.
    hp_config : dict, optional
        Hyperparameter overrides (from optimizer).
    use_contrastive : bool
        Whether to include the ContrastiveVerifier (E13, E14).
    use_evidence_constraint : bool
        Whether to enable evidence constraint filtering (E12, E14).

    Returns
    -------
    RAGPipeline instance.
    """
    return _build_phase2_rag(
        experiment_id="E14",
        retrieval_config=retrieval_config,
        model_config=model_config,
        paths_config=paths_config,
        hp_config=hp_config,
        use_contrastive=use_contrastive,
        use_evidence_constraint=use_evidence_constraint,
    )


def build_pipeline_for_experiment(
    experiment_id: str,
    retrieval_config: Dict,
    model_config: Dict,
    paths_config: Dict,
    data_config: Optional[Dict] = None,
    hp_config: Optional[Dict] = None,
):
    """
    Build the appropriate pipeline for a given experiment ID.

    Returns a (pipeline, pipeline_type) tuple where pipeline_type is a string
    descriptor used in experiment logging.
    """
    exp_type = _get_experiment_type(experiment_id)
    if experiment_id in {"E15", "E16", "E17"} and not hp_config:
        hp_config = _load_optimizer_best_config(experiment_id)
    cfg = {**retrieval_config, **(hp_config or {})}
    max_final_codes = int(cfg.get("max_final_codes", 10))

    if experiment_id in {"E9", "E10", "E11", "E12", "E13", "E14"}:
        pipeline = _build_phase2_rag(
            experiment_id=experiment_id,
            retrieval_config=retrieval_config,
            model_config=model_config,
            paths_config=paths_config,
            hp_config=hp_config,
            use_evidence_constraint=experiment_id in {"E12", "E14"},
            use_contrastive=experiment_id in {"E13", "E14"},
        )
        return pipeline, "rag_phase2"

    if exp_type == "baseline_tfidf_lr":
        from ..baselines.tfidf_lr import TFIDFLRBaseline
        return TFIDFLRBaseline(top_k=max_final_codes), "baseline"

    elif exp_type == "baseline_tfidf_svm":
        from ..baselines.tfidf_svm import TFIDFSVMBaseline
        return TFIDFSVMBaseline(top_k=max_final_codes), "baseline"

    elif exp_type == "baseline_transformer":
        from ..baselines.transformer_classifier import TransformerClassifier
        model_path = model_config.get("classifier_model_path", "")
        return TransformerClassifier(
            model_path=model_path,
            device=model_config.get("device", "cuda"),
            max_length=int(model_config.get("classifier_max_length", 256)),
            batch_size=int(model_config.get("classifier_batch_size", 2)),
            threshold=float(model_config.get("classifier_threshold", 0.5)),
            top_k=int(model_config.get("max_final_codes", 10)),
            epochs=int(model_config.get("classifier_epochs", 1)),
            learning_rate=float(model_config.get("classifier_learning_rate", 2e-5)),
            fp16=bool(model_config.get("classifier_fp16", False)),
            allow_cpu_fallback=bool(model_config.get("classifier_allow_cpu_fallback", False)),
            use_safetensors=bool(model_config.get("classifier_use_safetensors", True)),
            allow_mock=False,
        ), "baseline"

    elif exp_type in ("retrieval_bm25", "retrieval_dense", "retrieval_hybrid"):
        return _build_retrieval_only(exp_type, retrieval_config, model_config, paths_config, hp_config), "retrieval_only"

    elif exp_type in ("llm_zero_shot", "llm_few_shot"):
        from ..baselines.llm_only import LLMOnlyBaseline
        llm_path = model_config.get("llm_model_path", "")
        device = model_config.get("device", "cpu")
        few_shot = exp_type == "llm_few_shot"
        # Forward the runtime config (fp16, decode budget, truncation) and the Top-N label
        # space; previously E7/E8 ignored cfg entirely (fp32 + sampling + 512-token decode
        # made them slower than the full RAG model) and were never label-space constrained.
        top_n_suffix = str(cfg.get("top_n_suffix", "50"))
        _allowed = _load_label_space(paths_config, top_n_suffix)
        # Closed-set variant (llm_only_closed_set): show the model the benchmark label space.
        # Open generation over the full ICD vocabulary left 97% of notes with no in-vocabulary
        # code after filtering, so the arm measured label-space mismatch rather than coding
        # ability; see baselines/llm_only.py.
        _catalog = _load_label_catalog(paths_config, _allowed) if cfg.get("llm_only_closed_set") else None
        return LLMOnlyBaseline(
            llm_path=llm_path,
            device=device,
            few_shot=few_shot,
            max_new_tokens=int(cfg.get("max_new_tokens", model_config.get("max_new_tokens", 256))),
            temperature=float(cfg.get("temperature", 0.0)),
            allowed_codes=_allowed,
            note_max_chars=int(cfg.get("note_max_chars", 0) or 0),
            use_fp16=bool(cfg.get("use_fp16", False)),
            max_input_tokens=cfg.get("max_input_tokens"),
            top_k=max_final_codes,
            label_catalog=_catalog,
            closed_set=bool(cfg.get("llm_only_closed_set", False)),
        ), "llm_only"

    elif exp_type in ("rag_bm25", "rag_dense", "rag_hybrid"):
        return _build_rag(exp_type, retrieval_config, model_config, paths_config, hp_config,
                         use_contrastive=False, use_evidence_constraint=False), "simple_rag"

    else:
        return _build_phase2_rag(
            experiment_id=experiment_id,
            retrieval_config=retrieval_config,
            model_config=model_config,
            paths_config=paths_config,
            hp_config=hp_config,
            use_contrastive=True,
            use_evidence_constraint=True,
        ), "full_model"


def _load_optimizer_best_config(experiment_id: str) -> Dict[str, Any]:
    exp_to_dir = {
        "E15": "E15_random_search",
        "E16": "E16_mopso",
        "E17": "E17_nsga2",
    }
    run_dir = Path("results") / "optimization" / exp_to_dir[experiment_id]
    best_cfg_path = run_dir / "best_compromise_config.json"
    if not best_cfg_path.exists():
        raise FileNotFoundError(
            f"Missing optimizer best config for {experiment_id}: {best_cfg_path}. "
            "Run scripts/08_run_optimizer.py first."
        )
    payload = json.loads(best_cfg_path.read_text(encoding="utf-8"))
    config = payload.get("config")
    if not isinstance(config, dict) or not config:
        raise RuntimeError(f"Invalid best_compromise_config.json for {experiment_id}: missing non-empty config")
    return config


def _get_experiment_type(exp_id: str) -> str:
    mapping = {
        "E1": "baseline_tfidf_lr",
        "E2": "baseline_tfidf_svm",
        "E3": "baseline_transformer",
        "E4": "retrieval_bm25",
        "E5": "retrieval_dense",
        "E6": "retrieval_hybrid",
        "E7": "llm_zero_shot",
        "E8": "llm_few_shot",
        "E9": "rag_bm25",
        "E10": "rag_dense",
        "E11": "rag_hybrid",
        "E12": "rag_evidence_constraint",
        "E13": "rag_contrastive",
        "E14": "full_model",
        "E15": "full_model_opt",
        "E16": "full_model_opt",
        "E17": "full_model_opt",
        "E18": "full_model",
    }
    return mapping.get(exp_id, "full_model")


def _build_retrieval_only(mode, retrieval_config, model_config, paths_config, hp_config):
    from ..retrieval.bm25_index import BM25
    from ..retrieval.dense_index import DenseIndex
    from ..retrieval.hybrid_retriever import HybridRetriever
    from ..retrieval.icd_retriever import ICDCandidateRetriever
    from ..baselines.retrieval_only import RetrievalOnlyBaseline

    cfg = {**retrieval_config, **(hp_config or {})}
    indexes_dir = Path(paths_config.get("indexes_dir", "indexes"))
    top_n = cfg.get("top_n_suffix", "50")
    alpha = 1.0 if mode == "retrieval_bm25" else (0.0 if mode == "retrieval_dense" else cfg.get("bm25_dense_alpha", 0.5))

    icd_bm25_path = indexes_dir / "bm25" / f"icd_bm25_{top_n}.pkl"
    icd_bm25 = BM25.load(icd_bm25_path) if icd_bm25_path.exists() else _create_empty_bm25()
    icd_dense = _try_load_dense(
        indexes_dir / "faiss" / f"icd_dense_{top_n}",
        model_config.get("embedding_model_path", ""),
        model_config.get("device", "cpu"),
    )
    retriever = HybridRetriever(icd_bm25, icd_dense, alpha=alpha)
    _allowed = _load_label_space(paths_config, str(top_n))
    icd_retriever = ICDCandidateRetriever(retriever, allowed_codes=_allowed)
    top_k = int(cfg.get("max_final_codes", 10))
    return RetrievalOnlyBaseline(
        icd_retriever, top_k=top_k,
        allowed_codes=_load_label_space(paths_config, str(top_n)),
    )


def _build_rag(mode, retrieval_config, model_config, paths_config, hp_config,
               use_contrastive, use_evidence_constraint):
    cfg = {**retrieval_config, **(hp_config or {})}
    if mode == "rag_bm25":
        cfg["bm25_dense_alpha"] = 1.0
    elif mode == "rag_dense":
        cfg["bm25_dense_alpha"] = 0.0
    return build_full_model(retrieval_config, model_config, paths_config, cfg,
                            use_contrastive=use_contrastive,
                            use_evidence_constraint=use_evidence_constraint)


def _build_phase2_rag(
    experiment_id: str,
    retrieval_config: Dict,
    model_config: Dict,
    paths_config: Dict,
    hp_config: Optional[Dict] = None,
    use_evidence_constraint: bool = False,
    use_contrastive: bool = False,
):
    from ..retrieval.bm25_index import BM25
    from ..retrieval.hybrid_retriever import HybridRetriever
    from ..retrieval.evidence_retriever import ClinicalEvidenceRetriever
    from ..retrieval.icd_retriever import ICDCandidateRetriever
    from ..llm.local_llm import LocalLLM
    from ..llm.code_scorer import CodeScorer
    from ..llm.contrastive_verifier import ContrastiveVerifier
    from ..pipeline.candidate_generator import CandidateGenerator
    from ..pipeline.evidence_extractor import EvidenceExtractor
    from ..pipeline.rag_pipeline import RAGPipeline

    cfg = {**retrieval_config, **(hp_config or {})}
    indexes_dir = Path(paths_config.get("indexes_dir", "indexes"))
    top_n_suffix = cfg.get("top_n_suffix", "50")
    retrieval_mode = {
        "E9": "bm25",
        "E10": "dense",
        "E11": "hybrid",
        "E12": "hybrid",
        "E13": "hybrid",
        "E14": "hybrid",
    }.get(experiment_id, "hybrid")

    allow_mock_llm = bool(cfg.get("allow_mock_llm", False))
    allow_mock_embedding = bool(cfg.get("allow_mock_embedding", False))

    ev_bm25_path = indexes_dir / "bm25" / f"evidence_bm25_{top_n_suffix}.pkl"
    icd_bm25_path = indexes_dir / "bm25" / f"icd_bm25_{top_n_suffix}.pkl"
    ev_dense_path = indexes_dir / "faiss" / f"evidence_dense_{top_n_suffix}"
    icd_dense_path = indexes_dir / "faiss" / f"icd_dense_{top_n_suffix}"

    embed_path = model_config.get("embedding_model_path", "all-MiniLM-L6-v2")
    # Use the PRIMARY device (cuda). LocalLLM/DenseIndex already self-degrade to CPU when
    # CUDA is unavailable; preferring fallback_device here silently ran the LLM on CPU
    # while the GPU sat idle (root cause of ~162 s/note canaries).
    device = model_config.get("device", model_config.get("fallback_device", "cpu"))
    alpha = 1.0 if retrieval_mode == "bm25" else (0.0 if retrieval_mode == "dense" else cfg.get("bm25_dense_alpha", 0.5))

    # The batched design retrieves evidence from the CURRENT note's own chunks, so the
    # global evidence index (a ~1.3GB corpus-wide chunk index) is only needed by the
    # legacy per-candidate path. Skipping it saves large load time/memory per job.
    _need_global_evidence = bool(cfg.get("legacy_per_candidate", False))
    ev_bm25 = (BM25.load(ev_bm25_path) if (_need_global_evidence and ev_bm25_path.exists())
               else _create_empty_bm25())
    icd_bm25 = BM25.load(icd_bm25_path) if icd_bm25_path.exists() else _create_empty_bm25()

    ev_dense = None
    icd_dense = None
    mock_embedding_used = False
    dense_used = False
    skipped_dense = False

    if retrieval_mode in {"dense", "hybrid"}:
        if _need_global_evidence:
            ev_dense, ev_mock = _load_or_mock_dense(ev_dense_path, ev_bm25.docs, embed_path, device, allow_mock_embedding)
        else:
            ev_dense, ev_mock = None, False
        icd_dense, icd_mock = _load_or_mock_dense(icd_dense_path, icd_bm25.docs, embed_path, device, allow_mock_embedding)
        mock_embedding_used = bool(ev_mock or icd_mock)

        if retrieval_mode == "dense":
            if icd_dense is None:
                raise RuntimeError("Dense index unavailable for E10 and mock embedding is not allowed.")
            dense_used = True
        else:
            if icd_dense is None:
                skipped_dense = True
                dense_used = False
            else:
                dense_used = True

    ev_hybrid = HybridRetriever(ev_bm25, ev_dense, alpha=alpha)
    icd_hybrid = HybridRetriever(icd_bm25, icd_dense, alpha=alpha)
    evidence_retriever = ClinicalEvidenceRetriever(ev_hybrid)
    icd_retriever = ICDCandidateRetriever(icd_hybrid, allowed_codes=_load_label_space(paths_config, top_n_suffix))

    llm_path = model_config.get("llm_model_path", "")
    load_in_4bit = model_config.get("load_in_4bit", False)
    llm = LocalLLM(
        llm_path,
        device=device,
        load_in_4bit=load_in_4bit,
        allow_mock_llm=allow_mock_llm,
        max_new_tokens=cfg.get("max_new_tokens", model_config.get("max_new_tokens", 512)),
        temperature=cfg.get("temperature", model_config.get("temperature", 0.1)),
        do_sample=cfg.get("do_sample", False),
        num_beams=cfg.get("num_beams", 1),
        max_input_tokens=cfg.get("max_input_tokens"),
        prompt_max_chars=cfg.get("prompt_max_chars"),
        use_fp16=cfg.get("use_fp16", False),
        torch_inference_mode=cfg.get("torch_inference_mode", True),
    )
    code_scorer = CodeScorer(llm, cfg)
    contrastive_verifier = ContrastiveVerifier(llm, cfg) if use_contrastive else None

    pipeline_config = {
        "experiment_id": experiment_id,
        "retrieval_mode": retrieval_mode,
        "chunk_size": cfg.get("chunk_size"),
        "chunk_overlap": cfg.get("chunk_overlap"),
        "top_k_icd_candidates": cfg.get("top_k_icd_candidates", 50),
        "top_k_evidence": cfg.get("top_k_evidence", 5),
        "note_max_chars": cfg.get("note_max_chars"),
        "prompt_max_chars": cfg.get("prompt_max_chars"),
        "max_input_tokens": cfg.get("max_input_tokens"),
        "max_new_tokens": cfg.get("max_new_tokens", model_config.get("max_new_tokens", 512)),
        "evidence_snippet_max_chars": cfg.get("evidence_snippet_max_chars"),
        "contrastive_max_pairs": cfg.get("contrastive_max_pairs"),
        "clear_cuda_cache_between_samples": cfg.get("clear_cuda_cache_between_samples", False),
        "prompt_template_id": cfg.get("prompt_template_id"),
        "llm_confidence_threshold": cfg.get("llm_confidence_threshold", 0.5),
        "evidence_similarity_threshold": cfg.get("evidence_similarity_threshold", 0.3),
        "section_weights": cfg.get("section_weights"),
        "evidence_constraint_mode": cfg.get("evidence_constraint_mode", "flag"),
        "max_final_codes": cfg.get("max_final_codes", 10),
        "use_evidence_constraint": use_evidence_constraint,
        "use_contrastive_verifier": use_contrastive,
        "bm25_dense_alpha": alpha,
        "legacy_per_candidate": bool(cfg.get("legacy_per_candidate", False)),
        "scoring_mode": "legacy_per_candidate" if cfg.get("legacy_per_candidate", False) else "batched_per_note",
        "evidence_note_local": True,
        "allowed_codes": _load_label_space(paths_config, top_n_suffix),
        "mock_llm": bool(getattr(llm, "is_mock", False)),
        "mock_embedding": bool(mock_embedding_used),
        "dense_used": bool(dense_used),
        "skipped_dense": bool(skipped_dense),
    }
    pipeline_config["label_space_applied"] = bool(pipeline_config["allowed_codes"])

    from ..data.section_splitter import process_note_into_chunks
    chunk_size = int(cfg.get("chunk_size") or 256)
    chunk_overlap = int(cfg.get("chunk_overlap") or 32)

    def _chunker(text: str):
        return process_note_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Shared embedding backend for note-local evidence (the loaded ICD dense index holds the
    # embedding model; None for BM25-only mode -> note-local evidence uses BM25).
    embedder = icd_dense

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
        embedder=embedder,
        chunker=_chunker,
    )
    # run_metadata is merged into every prediction record/summary — keep it lean
    pipeline.run_metadata = {k: v for k, v in pipeline_config.items() if k != "allowed_codes"}
    return pipeline


def _try_load_dense(path, embed_path, device):
    try:
        from ..retrieval.dense_index import DenseIndex
        index_file = Path(str(path)).with_suffix(".index")
        pkl_file = Path(str(path)).with_suffix(".pkl")
        if (index_file.exists() or pkl_file.exists()) and is_resolved_local_path(embed_path):
            di = DenseIndex(embed_path, device=device)
            di.load(path)
            return di
    except Exception as e:
        print(f"Warning: Dense index load failed ({path}): {e}")
    return None


def _load_or_mock_dense(path, docs, embed_path, device, allow_mock_embedding):
    dense = _try_load_dense(path, embed_path, device)
    if dense is not None:
        return dense, bool(getattr(dense, "mock_embedding", False))
    if allow_mock_embedding:
        from ..retrieval.dense_index import DenseIndex
        dense = DenseIndex(embed_path or "mock", device=device, allow_mock_embedding=True)
        dense.fit(docs or [], text_field="searchable_text")
        return dense, True
    return None, False


def _create_empty_bm25():
    from ..retrieval.bm25_index import BM25
    bm25 = BM25()
    bm25.docs = []
    return bm25
