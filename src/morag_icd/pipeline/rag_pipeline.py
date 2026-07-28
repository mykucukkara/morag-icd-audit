from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional
import re

from .candidate_generator import CandidateGenerator
from .evidence_extractor import EvidenceExtractor
from ..retrieval.evidence_retriever import ClinicalEvidenceRetriever
from ..retrieval.icd_retriever import ICDCandidateRetriever
from ..retrieval.note_evidence import NoteLocalEvidenceRetriever
from ..llm.code_scorer import CodeScorer
from ..llm.contrastive_verifier import ContrastiveVerifier

class RAGPipeline:
    def __init__(
        self, 
        evidence_retriever: ClinicalEvidenceRetriever,
        icd_retriever: ICDCandidateRetriever,
        code_scorer: CodeScorer,
        candidate_generator: CandidateGenerator | None = None,
        evidence_extractor: EvidenceExtractor | None = None,
        contrastive_verifier: ContrastiveVerifier = None,
        config: dict = None,
        embedder=None,
        chunker=None,
    ):
        self.evidence_retriever = evidence_retriever
        self.icd_retriever = icd_retriever
        self.code_scorer = code_scorer
        self.embedder = embedder          # shared embedding backend for note-local evidence
        self.chunker = chunker            # note_text -> list[{chunk_id, section_name, text}]
        self.candidate_generator = candidate_generator or CandidateGenerator.from_config(
            icd_retriever,
            config or {},
        )
        self.evidence_extractor = evidence_extractor or EvidenceExtractor.from_config(
            evidence_retriever,
            config or {},
        )
        self.contrastive_verifier = contrastive_verifier
        self.config = config or {}
        self.run_metadata = {
            "retrieval_mode": self.config.get("retrieval_mode", "hybrid"),
            "use_evidence_constraint": bool(self.config.get("use_evidence_constraint", False)),
            "evidence_constraint_mode": self.config.get("evidence_constraint_mode", "flag"),
            "use_contrastive_verifier": bool(self.config.get("use_contrastive_verifier", False)),
            "mock_llm": bool(self.config.get("mock_llm", False)),
            "mock_embedding": bool(self.config.get("mock_embedding", False)),
            "dense_used": bool(self.config.get("dense_used", False)),
            "skipped_dense": bool(self.config.get("skipped_dense", False)),
        }
        
    def process_note(self, note_text: str) -> list[dict]:
        if self.config.get("legacy_per_candidate"):
            return self._process_legacy(note_text)
        return self._process_batched(note_text)

    # ------------------------------------------------------------------
    # Publication-grade path: note-local evidence + ONE batched LLM call
    # per note + targeted contrastive on borderline same-category families.
    # ------------------------------------------------------------------
    def _process_batched(self, note_text: str) -> list[dict]:
        note_max_chars = int(self.config.get("note_max_chars", 0) or 0)
        if note_max_chars:
            note_text = note_text[:note_max_chars]
        top_k_icd = int(self.config.get("top_k_icd_candidates", 50))
        max_candidate_codes = int(self.config.get("max_candidate_codes", 0) or 0)
        if max_candidate_codes:
            top_k_icd = min(top_k_icd, max_candidate_codes)
        top_k_ev = int(self.config.get("top_k_evidence", 5))
        max_final_codes = int(self.config.get("max_final_codes", 10))
        evidence_threshold = float(self.config.get("evidence_similarity_threshold", 0.3))
        evidence_constraint_mode = self.config.get("evidence_constraint_mode", "flag")
        use_evidence_constraint = bool(self.config.get("use_evidence_constraint", False))
        use_contrastive = bool(self.config.get("use_contrastive_verifier", False)) and self.contrastive_verifier is not None
        snippet_max = int(self.config.get("evidence_snippet_max_chars", 0) or 0)

        candidates = self.candidate_generator.generate(note_text)
        if not candidates:
            return []
        candidates = candidates[:top_k_icd]

        # Note-local evidence: chunk THIS note and retrieve evidence from its own chunks.
        chunks = self.chunker(note_text) if self.chunker else []
        alpha = float(self.config.get("bm25_dense_alpha", 0.5))
        note_ev = NoteLocalEvidenceRetriever(
            chunks, embedder=self.embedder, alpha=alpha,
            mode=self.config.get("retrieval_mode", "hybrid"),
        )

        # Build every candidate's evidence query first, then retrieve in ONE batch (the
        # dense side embeds all queries in a single encoder call).
        valid = []
        for rank, cand in enumerate(candidates, start=1):
            code = cand.get("code", "")
            if not code:
                continue
            title = cand.get("title") or cand.get("long_title") or cand.get("icd_title") or ""
            description = cand.get("searchable_text") or cand.get("description") or title
            valid.append((rank, cand, code, title, description, f"ICD-10 {code}: {title}"))
        ev_batch = note_ev.retrieve_evidence_batch([v[5] for v in valid], top_k_ev) if valid else []

        prepared = []
        for (rank, cand, code, title, description, _q), ev_chunks in zip(valid, ev_batch):
            ev_text = "\n".join(c.get("text", "") for c in ev_chunks[:top_k_ev] if c.get("text"))
            if snippet_max:
                ev_text = ev_text[:snippet_max]
            top_ev = ev_chunks[0] if ev_chunks else {}
            prepared.append({
                "cand": cand, "rank": rank, "code": code, "title": title, "description": description,
                "evidence_text": ev_text,
                "evidence_preview": self._sanitize_excerpt(ev_text),
                "evidence_chunk_id": top_ev.get("chunk_id", "") if top_ev else "",
                "evidence_score": float(top_ev.get("score", 0.0)) if top_ev else 0.0,
            })

        scores = self.code_scorer.score_candidates_batched(
            [{"code": p["code"], "title": p["title"], "description": p["description"],
              "evidence_text": p["evidence_text"]} for p in prepared],
            note_text=note_text,
        )

        scored_results = []
        for p, sc in zip(prepared, scores):
            supported = bool(sc.get("supported", False))
            risk_flag = sc.get("risk_flag", "none")
            evidence_score = p["evidence_score"]
            # Evidence constraint (E12/E14). A code must be grounded in the note's own
            # evidence: either the retrieved evidence is too weak, OR the LLM itself judged
            # it unsupported. In "filter" mode such codes are DROPPED (this is what makes
            # the ablation observable); in "flag" mode they are only marked.
            evidence_too_weak = evidence_score < evidence_threshold
            llm_unsupported = not bool(sc.get("supported", False))
            if use_evidence_constraint and (evidence_too_weak or llm_unsupported):
                if evidence_constraint_mode == "filter":
                    continue
                supported = False
                risk_flag = "weak_evidence" if evidence_too_weak else "ambiguous"
            cand = p["cand"]
            scored_results.append({
                "code": p["code"],
                "confidence": float(sc.get("confidence", 0.0)),
                "confidence_present": bool(sc.get("confidence_present", False)),
                "supported": supported,
                "evidence_quote": sc.get("evidence_quote", p["evidence_preview"]),
                "evidence_quote_verbatim": self._quote_is_verbatim(
                    sc.get("evidence_quote", ""), p["evidence_text"]
                ),
                "evidence_preview": p["evidence_preview"],
                "evidence_chunk_id": p["evidence_chunk_id"],
                "evidence_score": evidence_score,
                "icd_description": p["title"] or p["description"],
                "rationale": sc.get("rationale", ""),
                "risk_flag": risk_flag,
                "schema_valid": bool(sc.get("schema_valid", False)),
                "json_parse_error": bool(sc.get("json_parse_error", False)),
                "error_type": sc.get("error_type", ""),
                "parser_stage": sc.get("parser_stage", ""),
                "mock_llm": bool(sc.get("mock_llm", False)),
                "evidence_constraint_applied": use_evidence_constraint,
                "evidence_constraint_mode": evidence_constraint_mode,
                "evidence_note_local": True,
                "mock_embedding": bool(self.config.get("mock_embedding", False)),
                "candidate_rank": p["rank"],
                "retrieval_score": float(cand.get("score", 0.0) or 0.0),
                "parent_category": str(cand.get("parent") or ""),
                "icd_prefix3": p["code"][:3],
                "kb_sibling_codes": self._extract_kb_sibling_codes(cand.get("siblings")),
            })

        if use_contrastive and scored_results:
            self._targeted_contrastive(scored_results)
            scored_results = [p for p in scored_results if not p.pop("_contrastive_rejected", False)]

        scored_results = sorted(scored_results, key=self._rank_key)

        # Decision threshold (optimizer-tuned). Measured on the pilot: a 0.5 cut removed
        # 51.3% of the retrievable gold codes while carrying no discriminative signal
        # (kept 17.21% gold vs dropped 17.94%). It is therefore OFF by default and, when
        # enabled, is applied with a min_final_codes floor so a note is never left empty
        # purely by thresholding (150/1000 notes previously got zero predictions).
        conf_th = float(self.config.get("llm_confidence_threshold", 0.0) or 0.0)
        min_final = int(self.config.get("min_final_codes", 0) or 0)
        if conf_th > 0:
            # Only judge items that actually carry a confidence; an absent one is not a
            # low one, and thresholding fabricated 0.0s would silently delete every
            # prediction from a model that omits the field.
            kept = [
                p for p in scored_results
                if not p.get("confidence_present", True) or p.get("confidence", 0.0) >= conf_th
            ]
            if len(kept) < min_final:
                kept = scored_results[:min_final]
            scored_results = kept

        return scored_results[:max_final_codes]

    @staticmethod
    def _quote_is_verbatim(quote: str, evidence_text: str) -> Optional[bool]:
        """Did the model's evidence_quote actually come from this note's evidence?

        Computed here, at inference, against the FULL note-local evidence (which is never
        serialized, for PHI reasons) — the stored record keeps only the boolean. Whitespace/
        case are normalized so faithful tokens with reformatted spacing still count; the test
        stays external (the model's own quote must appear in the source it was given).
        Returns None when there is no quote to check (e.g. an unsupported code).
        """
        import re as _re
        q = _re.sub(r"\s+", " ", str(quote or "")).strip().lower()
        src = _re.sub(r"\s+", " ", str(evidence_text or "")).strip().lower()
        if not q or not src:
            return None
        return q in src

    @staticmethod
    def _rank_key(item: dict) -> tuple:
        """Ordering key for the final code list, robust to a model that omits `confidence`.

        Ranking used to be `confidence` alone, with an absent field defaulting to 0.0. That
        is fine while the model always emits it (Qwen2.5-3B: 88% of items) and catastrophic
        when it does not (Qwen2.5-7B: 7%) — the top-`max_final_codes` cut then selects on a
        constant, i.e. on nothing. Ordering here is lexicographic and every component is a
        field the model either emits or that we own:

          1. supported            — the one judgement every model reliably returns
          2. confidence           — only among items that actually carry one; items without
                                    a confidence never outrank items with one at the same
                                    `supported` level, and are ordered among themselves by
                                    retrieval rank rather than by a fabricated 0.0
          3. candidate_rank       — the retriever's own order, a well-defined last resort

        Returns an ascending-sort key (negations make higher-is-better fields sort first).
        """
        supported = bool(item.get("supported", False))
        has_conf = bool(item.get("confidence_present", True))
        try:
            conf = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            rank = int(item.get("candidate_rank", 0) or 0)
        except (TypeError, ValueError):
            rank = 0
        return (not supported, not has_conf, -conf if has_conf else 0.0, rank)

    def _targeted_contrastive(self, scored_results: list[dict], margin: float = 0.2) -> None:
        """Run ONE contrastive call only for borderline same-3-char-category families.

        A family is contrastively verified only when it has >=2 candidates and the top-2
        confidences are within `margin` (genuinely ambiguous). This keeps the contrastive
        step LLM-driven (not a deterministic fallback) and bounded to a few calls per note.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for pred in scored_results:
            groups[pred["code"][:3]].append(pred)
        for prefix, group in groups.items():
            if len(group) < 2:
                continue
            group_sorted = sorted(group, key=self._rank_key)
            # "Borderline" is a statement about confidences. If either of the top two lacks
            # one, the margin is not merely unknown but degenerate: fabricated 0.0s make
            # every family look maximally borderline, which fires an LLM call for EVERY
            # family on models that omit the field. No confidence -> no contrastive call.
            if not (group_sorted[0].get("confidence_present", True)
                    and group_sorted[1].get("confidence_present", True)):
                continue
            if group_sorted[0].get("confidence", 0.0) - group_sorted[1].get("confidence", 0.0) > margin:
                continue  # not borderline -> skip (no LLM call)
            target = group_sorted[0]
            siblings = [{"code": g["code"], "title": g.get("icd_description", "")} for g in group_sorted[1:]]
            evidence_text = "\n".join(g.get("evidence_preview", "") for g in group_sorted)
            cv = self.contrastive_verifier.verify(target["code"], target.get("icd_description", ""), siblings, evidence_text)
            preferred = cv.get("preferred_code", target["code"])
            chosen = next((g for g in group_sorted if g["code"] == preferred), target)
            chosen["contrastive_rationale"] = cv.get("contrastive_rationale", "")
            chosen["rejected_similar_codes"] = self._normalize_rejected_codes(
                cv.get("rejected_codes") or [], chosen.get("code", ""),
                "Rejected after contrastive comparison within the ICD category.",
            )
            chosen["contrastive_confidence"] = float(cv.get("confidence", chosen.get("confidence", 0.0)))
            chosen["contrastive_fallback_used"] = False
            chosen["mock_llm"] = bool(chosen.get("mock_llm", False) or cv.get("mock_llm", False))
            # The verifier must be able to CHANGE the predicted set, otherwise the E13/E14
            # ablation is invisible (measured: previously 0/1000 notes changed). Codes the
            # LLM explicitly rejected within this ICD category are removed.
            rejected_codes = {r.get("code") for r in (chosen.get("rejected_similar_codes") or [])}
            for g in group_sorted:
                if g["code"] != chosen["code"] and g["code"] in rejected_codes:
                    g["_contrastive_rejected"] = True

    def _process_legacy(self, note_text: str) -> list[dict]:
        note_max_chars = int(self.config.get("note_max_chars", 0) or 0)
        if note_max_chars:
            note_text = note_text[:note_max_chars]
        top_k_icd = int(self.config.get("top_k_icd_candidates", 50))
        top_k_ev = int(self.config.get("top_k_evidence", 5))
        max_candidate_codes = int(self.config.get("max_candidate_codes", 0) or 0)
        if max_candidate_codes:
            top_k_icd = min(top_k_icd, max_candidate_codes)
        max_final_codes = int(self.config.get("max_final_codes", 10))
        evidence_threshold = float(self.config.get("evidence_similarity_threshold", 0.3))
        evidence_constraint_mode = self.config.get("evidence_constraint_mode", "flag")
        use_evidence_constraint = bool(self.config.get("use_evidence_constraint", False))
        use_contrastive = bool(self.config.get("use_contrastive_verifier", False)) and self.contrastive_verifier is not None

        candidates = self.candidate_generator.generate(note_text)
        if not candidates:
            return []

        scored_results: List[Dict[str, Any]] = []
        for rank, cand in enumerate(candidates[:top_k_icd], start=1):
            code = cand.get("code", "")
            if not code:
                continue
            title = cand.get("title") or cand.get("long_title") or cand.get("icd_title") or ""
            description = cand.get("searchable_text") or cand.get("description") or title
            evidence_chunks = self.evidence_extractor.extract(code, title, description)
            snippet_max = int(self.config.get("evidence_snippet_max_chars", 0) or 0)
            evidence_parts = []
            for chunk in evidence_chunks[:top_k_ev]:
                text = chunk.get("text", "")
                if snippet_max:
                    text = text[:snippet_max]
                if text:
                    evidence_parts.append(text)
            evidence_text = "\n".join(evidence_parts)
            evidence_preview = self._sanitize_excerpt(evidence_text)
            top_evidence = evidence_chunks[0] if evidence_chunks else {}
            evidence_score = self._normalize_score(float(top_evidence.get("score", 0.0))) if top_evidence else 0.0
            score_res = self.code_scorer.score_candidate(code, title, description, evidence_text)

            supported = bool(score_res.get("supported", False))
            risk_flag = score_res.get("risk_flag", "none")
            if use_evidence_constraint and evidence_score < evidence_threshold:
                if evidence_constraint_mode == "filter":
                    continue
                supported = False
                risk_flag = "weak_evidence"

            pred = {
                "code": code,
                "confidence": float(score_res.get("confidence", 0.0)),
                "supported": supported,
                "evidence_quote": score_res.get("evidence_quote", evidence_preview),
                "evidence_quote_verbatim": self._quote_is_verbatim(
                    score_res.get("evidence_quote", ""), evidence_text
                ),
                "evidence_preview": evidence_preview,
                "evidence_chunk_id": top_evidence.get("chunk_id", "") if top_evidence else "",
                "evidence_score": evidence_score,
                "icd_description": title or description,
                "rationale": score_res.get("rationale", ""),
                "risk_flag": risk_flag,
                "mock_llm": bool(score_res.get("mock_llm", False)),
                "evidence_constraint_applied": use_evidence_constraint,
                "evidence_constraint_mode": evidence_constraint_mode,
                "mock_embedding": bool(self.config.get("mock_embedding", False)),
                "candidate_rank": rank,
                "retrieval_score": float(cand.get("score", 0.0) or 0.0),
                "parent_category": str(cand.get("parent") or ""),
                "icd_prefix3": code[:3],
                "kb_sibling_codes": self._extract_kb_sibling_codes(cand.get("siblings")),
            }
            scored_results.append(pred)

        _FALLBACK_REASON = (
            "Rejected because the retrieved evidence provides weaker support for this code "
            "than for the preferred code."
        )

        if use_contrastive and scored_results:
            all_scored_results: List[Dict[str, Any]] = list(scored_results)
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for pred in scored_results:
                grouped[pred["code"][:3]].append(pred)

            final_results: List[Dict[str, Any]] = []
            for prefix, group in grouped.items():
                group = sorted(group, key=lambda item: item.get("confidence", 0.0), reverse=True)
                target = group[0]
                siblings = [{"code": item["code"], "title": item.get("icd_description", "")} for item in group[1:]]
                if siblings:
                    evidence_text = "\n".join(item.get("evidence_preview", "") for item in group)
                    cv_res = self.contrastive_verifier.verify(
                        target["code"],
                        target.get("icd_description", ""),
                        siblings,
                        evidence_text,
                    )
                    preferred_code = cv_res.get("preferred_code", target["code"])
                    chosen = next((item for item in group if item["code"] == preferred_code), target)
                    chosen["contrastive_rationale"] = cv_res.get("contrastive_rationale", "")
                    llm_rejected = self._normalize_rejected_codes(
                        cv_res.get("rejected_codes") or [],
                        chosen.get("code", ""),
                        _FALLBACK_REASON,
                    )
                    if llm_rejected:
                        chosen["rejected_similar_codes"] = llm_rejected
                        chosen["contrastive_fallback_used"] = False
                    else:
                        fallback_pool = self._build_contrastive_fallback_pool(
                            preferred=chosen,
                            in_group_candidates=group,
                            all_candidates=all_scored_results,
                        )
                        chosen["rejected_similar_codes"] = self._build_rejected_similar_codes(
                            chosen,
                            fallback_pool,
                            _FALLBACK_REASON,
                        )
                        chosen["contrastive_fallback_used"] = True
                    chosen["contrastive_confidence"] = float(cv_res.get("confidence", chosen.get("confidence", 0.0)))
                    chosen["mock_llm"] = bool(chosen.get("mock_llm", False) or cv_res.get("mock_llm", False))
                    final_results.append(chosen)
                else:
                    # No in-prefix siblings — build fallback from other high-confidence candidates
                    fallback_pool = self._build_contrastive_fallback_pool(
                        preferred=target,
                        in_group_candidates=group,
                        all_candidates=all_scored_results,
                    )
                    if fallback_pool:
                        target["rejected_similar_codes"] = self._build_rejected_similar_codes(
                            target,
                            fallback_pool,
                            _FALLBACK_REASON,
                        )
                        target["contrastive_fallback_used"] = True
                        target["contrastive_rationale"] = (
                            "No in-prefix candidates; preferred based on retrieval confidence."
                        )
                    else:
                        target["rejected_similar_codes"] = []
                        target["contrastive_fallback_used"] = False
                        target["contrastive_rationale"] = (
                            "No similar candidates available for contrastive comparison."
                        )
                    target["contrastive_confidence"] = float(target.get("confidence", 0.0))
                    final_results.append(target)

            scored_results = final_results
            for target in scored_results:
                rejected = target.get("rejected_similar_codes")
                if not isinstance(rejected, list) or not rejected:
                    fallback_pool = self._build_contrastive_fallback_pool(
                        preferred=target,
                        in_group_candidates=scored_results,
                        all_candidates=all_scored_results,
                    )
                    target["rejected_similar_codes"] = self._build_rejected_similar_codes(
                        target,
                        fallback_pool,
                        _FALLBACK_REASON,
                    )
                    if target["rejected_similar_codes"]:
                        target["contrastive_fallback_used"] = True
                        target["contrastive_rationale"] = target.get("contrastive_rationale") or (
                            "Preferred code retained after deterministic comparison with high-ranking candidates."
                        )
                target.setdefault("contrastive_fallback_used", False)
                target.setdefault("contrastive_rationale", "")
                target.setdefault("contrastive_confidence", float(target.get("confidence", 0.0)))
                if not str(target.get("contrastive_rationale", "")).strip():
                    target["contrastive_rationale"] = (
                        "Preferred code retained after contrastive verification against similar candidates."
                    )

        conf_th = float(self.config.get("llm_confidence_threshold", 0.0) or 0.0)
        if conf_th > 0:
            scored_results = [p for p in scored_results if p.get("confidence", 0.0) >= conf_th]

        scored_results = sorted(scored_results, key=self._rank_key)
        return scored_results[:max_final_codes]

    @staticmethod
    def _build_rejected_similar_codes(
        preferred: Dict[str, Any],
        candidate_pool: List[Dict[str, Any]],
        reason: str,
        limit: int = 3,
    ) -> List[Dict[str, str]]:
        preferred_code = str(preferred.get("code", ""))
        preferred_prefix = preferred_code[:3]
        seen = {preferred_code}

        def rank_key(item: Dict[str, Any]) -> tuple[int, float]:
            code = str(item.get("code", ""))
            same_prefix = 0 if code[:3] == preferred_prefix else 1
            return same_prefix, -float(item.get("confidence", item.get("score", 0.0)) or 0.0)

        rejected: List[Dict[str, str]] = []
        for item in sorted(candidate_pool, key=rank_key):
            code = str(item.get("code", ""))
            if not code or code in seen:
                continue
            seen.add(code)
            code_reason = str(item.get("reason") or reason).strip() or reason
            rejected.append({
                "code": code,
                "title": str(item.get("icd_description") or item.get("title") or ""),
                "reason": code_reason,
            })
            if len(rejected) >= limit:
                break
        return rejected

    @staticmethod
    def _normalize_rejected_codes(
        rejected_codes: Any,
        preferred_code: str,
        default_reason: str,
    ) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        seen = {preferred_code}
        if not isinstance(rejected_codes, list):
            return normalized

        for item in rejected_codes:
            if isinstance(item, dict):
                code = str(item.get("code", "")).strip()
                title = str(item.get("title", "")).strip()
                reason = str(item.get("reason", "")).strip() or default_reason
            elif isinstance(item, str):
                code = item.strip()
                title = ""
                reason = default_reason
            else:
                continue
            if not code or code in seen:
                continue
            seen.add(code)
            normalized.append({"code": code, "title": title, "reason": reason})

        return normalized

    def _build_contrastive_fallback_pool(
        self,
        preferred: Dict[str, Any],
        in_group_candidates: List[Dict[str, Any]],
        all_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        preferred_code = str(preferred.get("code", ""))
        preferred_parent = str(preferred.get("parent_category", ""))
        preferred_prefix = str(preferred.get("icd_prefix3") or preferred_code[:3])
        preferred_kb_siblings = preferred.get("kb_sibling_codes") or []

        out: List[Dict[str, Any]] = []
        seen = {preferred_code}

        def add_item(item: Dict[str, Any], source: str) -> None:
            code = str(item.get("code", "")).strip()
            if not code or code in seen:
                return
            seen.add(code)
            clone = dict(item)
            clone["contrastive_source"] = source
            out.append(clone)

        # 1) same parent category
        if preferred_parent:
            for item in all_candidates:
                if str(item.get("parent_category", "")).strip() == preferred_parent:
                    add_item(item, "same_parent_category")

        # 2) same 3-character ICD prefix
        if preferred_prefix:
            for item in all_candidates:
                if str(item.get("code", ""))[:3] == preferred_prefix:
                    add_item(item, "same_3char_prefix")

        # 3) ICD KB sibling codes
        sibling_map = {str(item.get("code", "")): item for item in all_candidates}
        for sibling_code in preferred_kb_siblings:
            sibling_item = sibling_map.get(sibling_code)
            if sibling_item:
                add_item(sibling_item, "kb_sibling")
            else:
                add_item(
                    {
                        "code": sibling_code,
                        "icd_description": "",
                        "confidence": 0.0,
                        "retrieval_score": 0.0,
                    },
                    "kb_sibling",
                )

        # 4) current candidate list (other in-group candidates)
        for item in in_group_candidates:
            add_item(item, "current_candidate_list")

        # 5) high retrieval-score candidates not selected
        def retrieval_key(item: Dict[str, Any]) -> float:
            return float(item.get("retrieval_score", item.get("confidence", 0.0)) or 0.0)

        for item in sorted(all_candidates, key=retrieval_key, reverse=True):
            add_item(item, "high_retrieval_unselected")

        return out

    @staticmethod
    def _extract_kb_sibling_codes(raw_siblings: Any) -> List[str]:
        if isinstance(raw_siblings, list):
            out = []
            for item in raw_siblings:
                code = str(item).strip()
                if code:
                    out.append(code)
            return out
        if isinstance(raw_siblings, str):
            return [code for code in re.split(r"[,;\s]+", raw_siblings.strip()) if code]
        return []

    @staticmethod
    def _normalize_score(score: float) -> float:
        if score <= 0:
            return 0.0
        return float(min(1.0, score / (1.0 + score)))

    @staticmethod
    def _sanitize_excerpt(text: str, max_chars: int = 120) -> str:
        cleaned = " ".join(text.split())
        return cleaned[:max_chars]
