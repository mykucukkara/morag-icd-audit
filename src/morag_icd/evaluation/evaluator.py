"""
Central evaluator that reads JSONL predictions and computes all metrics.

Supports:
- Multi-label classification metrics (micro/macro F1, P@k, R@k, etc.)
- Evidence/reliability metrics
- Hierarchical ICD metrics
- Cost/efficiency metrics
- Bootstrap CI (optional)
- Error/failure rate metrics
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import compute_classification_metrics
from .evidence_metrics import compute_reliability_metrics, compute_similar_code_confusion_rate
from .hierarchical_metrics import compute_hierarchical_metrics
from .cost_metrics import compute_cost_metrics, compute_error_rate_metrics


class Evaluator:
    """Evaluate JSONL prediction files and produce comprehensive metrics."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        predictions_path: str | Path,
        label_set: Optional[List[str]] = None,
        run_bootstrap: bool = False,
        bootstrap_n: int = 500,
        seed: int = 42,
    ) -> Dict:
        """
        Load a JSONL predictions file and compute all metrics.

        Parameters
        ----------
        predictions_path : str | Path
            Path to a JSONL file. Each line is one prediction record.
        label_set : list of str, optional
            Fixed label universe. Inferred from data if not provided.
        run_bootstrap : bool
            Whether to compute bootstrap CIs (slow).
        bootstrap_n : int
            Number of bootstrap resamples.
        seed : int
            Random seed for bootstrap.

        Returns
        -------
        dict with all metrics grouped by category.
        """
        t0 = time.time()
        predictions = self._load_predictions(predictions_path)

        if not predictions:
            return {"error": "No predictions loaded", "path": str(predictions_path)}

        # Extract parallel lists for metric computation
        pred_codes: List[List[str]] = [
            [cp.get("code", "") for cp in p.get("predicted_codes", [])]
            for p in predictions
        ]
        gold_codes: List[List[str]] = [p.get("gold_codes", []) for p in predictions]

        # 1. Classification metrics
        clf_metrics = compute_classification_metrics(pred_codes, gold_codes, label_set=label_set)

        # 2. Evidence/reliability metrics
        rel_metrics = compute_reliability_metrics(predictions)

        # 3. Similar-code confusion rate
        similar_confusion = compute_similar_code_confusion_rate(
            predictions, [{"hadm_id": p.get("hadm_id"), "gold_codes": p.get("gold_codes", [])} for p in predictions]
        )
        rel_metrics["similar_code_confusion_rate"] = similar_confusion

        # 4. Hierarchical metrics
        hier_metrics = compute_hierarchical_metrics(pred_codes, gold_codes, label_set=None)

        # 5. Cost metrics
        cost_metrics = compute_cost_metrics(predictions)

        # 6. Error rate metrics
        error_metrics = compute_error_rate_metrics(predictions)

        # 7. Bootstrap CI (optional)
        bootstrap_results: Dict = {}
        if run_bootstrap:
            from .bootstrap_ci import bootstrap_ci
            bootstrap_results = bootstrap_ci(
                pred_codes, gold_codes,
                n_bootstrap=bootstrap_n,
                seed=seed,
                label_set=label_set,
            )

        # Save consolidated metrics
        run_metadata = {}
        if predictions:
            for key in [
                "experiment_id",
                "retrieval_mode",
                "use_evidence_constraint",
                "use_contrastive_verifier",
                "mock_llm",
                "mock_embedding",
                "skipped_dense",
                "evidence_constraint_mode",
            ]:
                if key in predictions[0]:
                    run_metadata[key] = predictions[0].get(key)

        all_metrics = {
            "classification": clf_metrics,
            "reliability": rel_metrics,
            "hierarchical": hier_metrics,
            "cost": cost_metrics,
            "errors": error_metrics,
            "bootstrap_ci": bootstrap_results,
            "eval_runtime_sec": time.time() - t0,
            "predictions_path": str(predictions_path),
            "n_samples": len(predictions),
            **run_metadata,
        }

        # Save metrics
        pred_path = Path(predictions_path)
        exp_id = f"{pred_path.parent.name}_{pred_path.stem}"
        out_path = self.output_dir / f"{exp_id}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, default=str)

        # Also save flat version for easy pandas loading
        flat = self._flatten_metrics(all_metrics)
        flat_path = self.output_dir / f"{exp_id}_metrics_flat.json"
        with open(flat_path, "w", encoding="utf-8") as f:
            json.dump(flat, f, indent=2, default=str)

        return all_metrics

    def evaluate_all(
        self,
        predictions_dir: str | Path,
        pattern: str = "*.jsonl",
        label_set: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate all JSONL files in a directory.

        Returns
        -------
        dict mapping file stem -> metrics dict.
        """
        predictions_dir = Path(predictions_dir)
        all_results = {}
        for p in sorted(predictions_dir.glob(pattern)):
            print(f"Evaluating {p.name}...")
            try:
                result = self.evaluate(p, label_set=label_set)
                all_results[p.stem] = result
            except Exception as e:
                all_results[p.stem] = {"error": str(e)}

        # Save combined summary
        summary_path = self.output_dir / "all_metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)

        return all_results

    @staticmethod
    def _load_predictions(path: str | Path) -> List[Dict]:
        """Load a JSONL predictions file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parse error on line {line_no}: {e}")
        return records

    @staticmethod
    def _flatten_metrics(metrics: Dict, prefix: str = "") -> Dict[str, float]:
        """Flatten nested metrics dict for easy CSV export."""
        flat = {}
        for k, v in metrics.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(Evaluator._flatten_metrics(v, prefix=f"{key}."))
            elif isinstance(v, (int, float, str, bool)) or v is None:
                flat[key] = v
        return flat
