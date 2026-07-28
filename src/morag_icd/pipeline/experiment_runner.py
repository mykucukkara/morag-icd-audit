"""
ExperimentRunner: runs a pipeline over a dataset with checkpoint/resume support.

Features:
- Per-sample try/except: one failure does not stop the run
- JSONL streaming output (one line per sample)
- Checkpoint every N samples for resume support
- Resume: skips already-processed samples on restart
- Detailed logging: runtime, token counts, error types
- Experiment summary on completion
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging_utils import setup_logger
from ..utils.hashing import hash_config


class ExperimentRunner:
    """Runs a pipeline over a dataset with checkpoint/resume and full logging."""

    def __init__(
        self,
        pipeline,
        config: Dict,
        experiment_id: str,
        seed: int,
        checkpoint_every: int = 50,
        max_error_rate: float = 0.5,
        pipeline_type: str = "full_model",
    ):
        """
        Parameters
        ----------
        pipeline : pipeline object with .process_note(text) method.
        config : merged config dict.
        experiment_id : str, e.g. "E11".
        seed : int
        checkpoint_every : int
            Write checkpoint every N samples.
        max_error_rate : float
            If error rate exceeds this, stop the run early.
        pipeline_type : str
            Type descriptor for logging ("baseline", "retrieval_only", etc.)
        """
        self.pipeline = pipeline
        self.config = config
        self.experiment_id = experiment_id
        self.seed = seed
        self.checkpoint_every = checkpoint_every
        self.max_error_rate = max_error_rate
        self.pipeline_type = pipeline_type

        logs_dir = config.get("logs_dir", "logs")
        self.logger = setup_logger(
            name=f"{experiment_id}_seed{seed}",
            log_dir=logs_dir,
            experiment_id=experiment_id,
            seed=seed,
        )
        self.config_hash = hash_config(config)

    def run(
        self,
        dataset: List[Dict],
        output_path: str | Path,
        label_set: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run the pipeline over the dataset.

        Parameters
        ----------
        dataset : list of dicts with keys: subject_id, hadm_id, text, gold_codes.
        output_path : str | Path
            JSONL file where predictions are streamed.
        label_set : list of str, optional
            Fixed label universe (for downstream evaluation).

        Returns
        -------
        dict with run summary statistics.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume: load already-processed hadm_ids
        processed_ids = self._load_processed_ids(output_path)
        if processed_ids:
            self.logger.info(f"Resuming: {len(processed_ids)} samples already processed.")

        success_count = 0
        failed_count = 0
        skipped_count = 0
        runtimes = []
        error_log = []
        any_skipped_dense = False

        # Open output file in append mode for streaming
        with open(output_path, "a", encoding="utf-8") as out_f:
            for i, sample in enumerate(dataset):
                hadm_id = str(sample.get("hadm_id", ""))
                subject_id = str(sample.get("subject_id", ""))

                # Resume: skip already-processed samples
                if hadm_id in processed_ids:
                    skipped_count += 1
                    continue

                t0 = time.time()
                try:
                    result = self._process_sample(sample)
                    if any(p.get("skipped_dense", False) for p in result.get("predicted_codes", [])):
                        any_skipped_dense = True
                    elapsed = time.time() - t0
                    runtimes.append(elapsed)
                    result["runtime_sec"] = elapsed
                    result["timestamp"] = datetime.now(timezone.utc).isoformat()

                    # Write prediction to JSONL
                    out_f.write(json.dumps(result, default=str) + "\n")
                    out_f.flush()
                    success_count += 1

                    if (i + 1) % self.checkpoint_every == 0:
                        self.logger.info(
                            f"Checkpoint: {i+1}/{len(dataset)} samples | "
                            f"success={success_count} fail={failed_count}"
                        )

                except Exception as e:
                    elapsed = time.time() - t0
                    failed_count += 1
                    safe_error_type = self._safe_error_type(e)
                    err_info = {
                        "error": self._safe_error_message(e),
                        "error_type": safe_error_type,
                        "stacktrace_hash": hashlib.sha256(traceback.format_exc().encode("utf-8")).hexdigest()[:16],
                        "elapsed_sec": elapsed,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    if self.config.get("data_mode") == "real" or self.config.get("phi_safe_logging", False):
                        err_info["sample_id"] = self._sample_id(subject_id, hadm_id)
                    else:
                        err_info["subject_id"] = subject_id
                        err_info["hadm_id"] = hadm_id
                    error_log.append(err_info)

                    # Write failed sample placeholder
                    failed_record = {
                        "status": "failed",
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "sample_id": self._sample_id(subject_id, hadm_id),
                        "experiment_id": self.experiment_id,
                        "resolved_experiment_id": self.experiment_id,
                        "seed": self.seed,
                        "failed": True,
                        "error_type": safe_error_type,
                        "error_message_safe": self._safe_error_message(e),
                        "predicted_codes": [],
                        "evidence": [],
                        "rationale": "",
                        "confidence": 0.0,
                        "risk_flags": ["oom"] if safe_error_type == "cuda_oom" else ["error"],
                        "gold_codes": sample.get("gold_codes", []),
                        "runtime_sec": elapsed,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "config_hash": self.config_hash,
                        "data_mode": self.config.get("data_mode", ""),
                        "top_n": int(self.config.get("top_n_suffix", 0) or 0),
                        "mock": bool(self.config.get("allow_mock_llm", False) or self.config.get("allow_mock_embedding", False)),
                        "mock_llm": bool(getattr(self.pipeline, "run_metadata", {}).get("mock_llm", False)),
                        "mock_embedding": bool(getattr(self.pipeline, "run_metadata", {}).get("mock_embedding", False)),
                        "phi_safe_logging": True,
                        **self._shard_meta(sample),
                        **getattr(self.pipeline, "run_metadata", {}),
                    }
                    out_f.write(json.dumps(failed_record, default=str) + "\n")
                    out_f.flush()
                    self._clear_cuda_cache_if_needed()

                    self.logger.error(
                        f"Sample failed: sample_id={self._sample_id(subject_id, hadm_id)} | "
                        f"{safe_error_type}: {self._safe_error_message(e)}"
                    )

                # Early stopping if error rate is too high
                total_processed = success_count + failed_count
                if total_processed >= 10 and failed_count / total_processed > self.max_error_rate:
                    self.logger.warning(
                        f"Error rate {failed_count/total_processed:.2%} exceeds threshold "
                        f"{self.max_error_rate:.2%}. Stopping early."
                    )
                    break
                self._clear_cuda_cache_if_needed()

        # Write error log
        if error_log:
            logs_root = Path(self.config.get("logs_dir", "logs"))
            err_dir = logs_root / "errors" / self.experiment_id / f"seed_{self.seed}"
            err_dir.mkdir(parents=True, exist_ok=True)
            err_path = err_dir / "errors.jsonl"
            with open(err_path, "w", encoding="utf-8") as ef:
                for err in error_log:
                    ef.write(json.dumps(err, default=str) + "\n")

        # Summary
        total = success_count + failed_count + skipped_count
        summary = {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "pipeline_type": self.pipeline_type,
            "config_hash": self.config_hash,
            "total_samples": len(dataset),
            "success_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "error_rate": failed_count / max(success_count + failed_count, 1),
            "skipped_dense": any_skipped_dense,
            "avg_runtime_sec": float(sum(runtimes) / len(runtimes)) if runtimes else 0.0,
            "total_runtime_sec": float(sum(runtimes)),
            "output_path": str(output_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        summary.update(getattr(self.pipeline, "run_metadata", {}))

        # Save summary
        summary_path = output_path.parent / f"{output_path.stem}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2, default=str)

        self.logger.info(
            f"Run complete: success={success_count} fail={failed_count} skip={skipped_count}"
        )
        return summary

    def _shard_meta(self, sample: Dict) -> Dict:
        """Return PHI-safe shard metadata for a prediction record.

        global_sample_index identifies the sample's position in the FULL split (set by
        the runner script when sharding is active); shard_* fields describe the shard.
        These are integers, not PHI.
        """
        meta = {"global_sample_index": sample.get("global_sample_index")}
        if self.config.get("is_sharded_run"):
            meta.update({
                "is_sharded_run": True,
                "shard_index": self.config.get("shard_index"),
                "num_shards": self.config.get("num_shards"),
                "shard_start_index": self.config.get("shard_start_index"),
                "shard_end_index": self.config.get("shard_end_index"),
            })
        return meta

    def _process_sample(self, sample: Dict) -> Dict:
        """Process one sample through the pipeline and format prediction record."""
        text = sample.get("text", "")
        gold_codes = sample.get("gold_codes", [])
        subject_id = str(sample.get("subject_id", ""))
        hadm_id = str(sample.get("hadm_id", ""))
        split = sample.get("split", "test")

        # Run pipeline
        predicted_codes = self.pipeline.process_note(text)

        if self.config.get("data_mode") == "real" or self.config.get("phi_safe_logging", False):
            self.logger.info(f"Processed sample sample_id={self._sample_id(subject_id, hadm_id)}")
        else:
            self.logger.info(
                f"Processed sample subject_id={subject_id} hadm_id={hadm_id} text_chars={len(text or '')}"
            )

        # Normalize output (pipeline may return list of dicts or list of strings)
        normalized_preds = []
        if isinstance(predicted_codes, list):
            for cp in predicted_codes:
                if isinstance(cp, dict):
                    normalized_preds.append(cp)
                elif isinstance(cp, str):
                    normalized_preds.append({"code": cp, "confidence": 1.0})

        risk_flags = sorted({
            str(pred.get("risk_flag"))
            for pred in normalized_preds
            if isinstance(pred, dict) and pred.get("risk_flag")
        })
        confidence_values = [
            float(pred.get("confidence", 0.0) or 0.0)
            for pred in normalized_preds
            if isinstance(pred, dict)
        ]
        rationale = next(
            (
                str(pred.get("rationale", ""))
                for pred in normalized_preds
                if isinstance(pred, dict) and str(pred.get("rationale", "")).strip()
            ),
            "",
        )

        return {
            "status": "success",
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "sample_id": self._sample_id(subject_id, hadm_id),
            "split": split,
            "experiment_id": self.experiment_id,
            "resolved_experiment_id": self.experiment_id,
            "seed": self.seed,
            "gold_codes": gold_codes,
            "predicted_codes": normalized_preds,
            "evidence": [],
            "rationale": rationale,
            "confidence": max(confidence_values) if confidence_values else 0.0,
            "risk_flags": risk_flags,
            "config_hash": self.config_hash,
            "data_mode": self.config.get("data_mode", ""),
            "top_n": int(self.config.get("top_n_suffix", 0) or 0),
            "mock": bool(self.config.get("allow_mock_llm", False) or self.config.get("allow_mock_embedding", False)),
            "phi_safe_logging": bool(self.config.get("phi_safe_logging", False)),
            **self._shard_meta(sample),
            **getattr(self.pipeline, "run_metadata", {}),
        }

    def _clear_cuda_cache_if_needed(self) -> None:
        if not self.config.get("clear_cuda_cache_between_samples", False):
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @staticmethod
    def _sample_id(subject_id: str, hadm_id: str) -> str:
        return hashlib.sha256(f"{subject_id}:{hadm_id}".encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _safe_error_type(exc: Exception) -> str:
        raw = f"{type(exc).__name__}: {exc}".lower()
        if "cuda out of memory" in raw or "outofmemory" in raw or "oom" in raw:
            return "cuda_oom"
        return type(exc).__name__

    @staticmethod
    def _safe_error_message(exc: Exception, max_chars: int = 500) -> str:
        message = " ".join(str(exc).split())
        return message[:max_chars]

    @staticmethod
    def _load_processed_ids(output_path: Path) -> set:
        """Load hadm_ids already in the output file for resume support."""
        if not output_path.exists():
            return set()
        ids = set()
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        h = rec.get("hadm_id")
                        if h:
                            ids.add(str(h))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return ids
