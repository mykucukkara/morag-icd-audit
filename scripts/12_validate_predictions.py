from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args():
    p = argparse.ArgumentParser(description="Validate prediction schema and diagnostics for smoke runs.")
    p.add_argument("--predictions-dir", default="results/predictions")
    p.add_argument("--top-n", type=int, default=50)
    return p.parse_args()


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_schema(predictions_root: Path):
    target_experiments = {"E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17"}
    required_record = [
        "subject_id",
        "hadm_id",
        "experiment_id",
        "seed",
        "gold_codes",
        "predicted_codes",
        "runtime_sec",
        "config_hash",
    ]
    required_code = [
        "code",
        "confidence",
        "supported",
        "evidence_score",
        "icd_description",
        "rationale",
        "risk_flag",
    ]

    result = {
        "status": "passed",
        "files_checked": 0,
        "records_checked": 0,
        "errors": [],
        "by_experiment": {},
    }

    for pred_file in sorted(predictions_root.glob("**/test_predictions.jsonl")):
        exp_dir = pred_file.parent.name
        exp_id = exp_dir.split("_seed")[0]
        if exp_id not in target_experiments:
            continue
        rows = load_jsonl(pred_file)
        result["files_checked"] += 1
        result["records_checked"] += len(rows)
        exp_errors = []

        for i, row in enumerate(rows, start=1):
            for k in required_record:
                if k not in row:
                    exp_errors.append(f"line {i}: missing field {k}")
            preds = row.get("predicted_codes", [])
            if not isinstance(preds, list):
                exp_errors.append(f"line {i}: predicted_codes is not list")
                continue
            for j, cp in enumerate(preds, start=1):
                for k in required_code:
                    if k not in cp:
                        exp_errors.append(f"line {i} pred {j}: missing field {k}")
                if not (cp.get("evidence_preview") or cp.get("evidence_quote")):
                    exp_errors.append(f"line {i} pred {j}: missing evidence_preview/evidence_quote")
                if exp_id in {"E13", "E14"}:
                    required_contrastive_fields = [
                        "contrastive_rationale",
                        "rejected_similar_codes",
                        "contrastive_confidence",
                        "contrastive_fallback_used",
                    ]
                    missing_contrastive = [k for k in required_contrastive_fields if k not in cp]
                    if missing_contrastive:
                        exp_errors.append(
                            f"line {i} pred {j}: missing contrastive fields for {exp_id}: {', '.join(missing_contrastive)}"
                        )

        result["by_experiment"][exp_id] = {
            "file": str(pred_file),
            "records": len(rows),
            "errors": exp_errors,
        }
        if exp_errors:
            result["status"] = "failed"
            result["errors"].extend([f"{exp_id}: {e}" for e in exp_errors[:20]])

    return result


def build_evidence_constraint_diagnostic(predictions_root: Path):
    lines = ["# Evidence Constraint Diagnostic", ""]
    target_exps = ["E12", "E14"]

    for exp in target_exps:
        pred_file = predictions_root / f"{exp}_seed42" / "test_predictions.jsonl"
        lines.append(f"## {exp}")
        if not pred_file.exists():
            lines.append("missing prediction file")
            lines.append("")
            continue

        rows = load_jsonl(pred_file)
        samples = rows[:5]
        lines.append(f"records: {len(rows)}")
        lines.append(f"use_evidence_constraint flags present: {all(bool(r.get('use_evidence_constraint', False)) for r in samples)}")
        threshold_vals = sorted(set(float(r.get("evidence_similarity_threshold", 0.0)) for r in rows))
        lines.append(f"evidence_similarity_threshold values: {threshold_vals}")

        weak_or_unsupported = 0
        with_score = 0
        for row in samples:
            for cp in row.get("predicted_codes", []):
                if cp.get("risk_flag") == "weak_evidence" or cp.get("supported") is False:
                    weak_or_unsupported += 1
                if "evidence_score" in cp:
                    with_score += 1

        lines.append(f"sampled predicted codes with weak/unsupported behavior: {weak_or_unsupported}")
        lines.append(f"sampled predicted codes with evidence_score: {with_score}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_contrastive_diagnostic(predictions_root: Path):
    lines = ["# Contrastive Verifier Diagnostic", ""]
    target_exps = ["E13", "E14"]
    payload: Dict[str, object] = {
        "records_checked": 0,
        "records_with_contrastive_rationale": 0,
        "records_with_rejected_similar_codes": 0,
        "records_with_contrastive_confidence": 0,
        "fallback_used_count": 0,
        "empty_rejected_codes_count": 0,
        "experiments": {},
    }

    for exp in target_exps:
        pred_file = predictions_root / f"{exp}_seed42" / "test_predictions.jsonl"
        lines.append(f"## {exp}")
        if not pred_file.exists():
            lines.append("missing prediction file")
            lines.append("")
            payload["experiments"][exp] = {
                "records": 0,
                "records_checked": 0,
                "records_with_contrastive_rationale": 0,
                "records_with_rejected_similar_codes": 0,
                "records_with_contrastive_confidence": 0,
                "fallback_used_count": 0,
                "empty_rejected_codes_count": 0,
                "missing_prediction_file": True,
            }
            continue

        rows = load_jsonl(pred_file)
        records_checked = 0
        non_empty_rationale = 0
        with_rejected_nonempty = 0
        with_conf = 0
        fallback_used = 0
        empty_rejected = 0
        for row in rows:
            for cp in row.get("predicted_codes", []):
                records_checked += 1
                if str(cp.get("contrastive_rationale", "")).strip():
                    non_empty_rationale += 1
                rej = cp.get("rejected_similar_codes")
                if isinstance(rej, list) and len(rej) > 0:
                    with_rejected_nonempty += 1
                else:
                    empty_rejected += 1
                if "contrastive_confidence" in cp:
                    with_conf += 1
                if cp.get("contrastive_fallback_used") is True:
                    fallback_used += 1

        exp_payload = {
            "records": len(rows),
            "records_checked": records_checked,
            "records_with_contrastive_rationale": non_empty_rationale,
            "records_with_rejected_similar_codes": with_rejected_nonempty,
            "records_with_contrastive_confidence": with_conf,
            "fallback_used_count": fallback_used,
            "empty_rejected_codes_count": empty_rejected,
        }
        payload["experiments"][exp] = exp_payload
        payload["records_checked"] += records_checked
        payload["records_with_contrastive_rationale"] += non_empty_rationale
        payload["records_with_rejected_similar_codes"] += with_rejected_nonempty
        payload["records_with_contrastive_confidence"] += with_conf
        payload["fallback_used_count"] += fallback_used
        payload["empty_rejected_codes_count"] += empty_rejected

        lines.append(f"records: {len(rows)}")
        lines.append(f"records_checked (predicted codes): {records_checked}")
        lines.append(f"records_with_contrastive_rationale: {non_empty_rationale}")
        lines.append(f"records_with_rejected_similar_codes: {with_rejected_nonempty}")
        lines.append(f"empty_rejected_codes_count: {empty_rejected}")
        lines.append(f"records_with_contrastive_confidence: {with_conf}")
        lines.append(f"fallback_used_count: {fallback_used}")
        lines.append("")

    return "\n".join(lines).strip() + "\n", payload


def main():
    args = parse_args()
    pred_root = Path(args.predictions_dir) / f"top{args.top_n}"
    reports_dir = Path("results/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    schema = validate_schema(pred_root)

    schema_json = reports_dir / "prediction_schema_validation.json"
    schema_md = reports_dir / "prediction_schema_validation.md"
    with open(schema_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    with open(schema_md, "w", encoding="utf-8") as f:
        f.write("# Prediction Schema Validation\n\n")
        f.write(f"status: {schema['status']}\n\n")
        f.write(f"files_checked: {schema['files_checked']}\n\n")
        f.write(f"records_checked: {schema['records_checked']}\n\n")
        if schema["errors"]:
            f.write("## Errors\n")
            for err in schema["errors"]:
                f.write(f"- {err}\n")
        else:
            f.write("No schema errors found.\n")

    evidence_diag = build_evidence_constraint_diagnostic(pred_root)
    contrastive_diag, contrastive_payload = build_contrastive_diagnostic(pred_root)

    (reports_dir / "evidence_constraint_diagnostic.md").write_text(evidence_diag, encoding="utf-8")
    (reports_dir / "contrastive_verifier_diagnostic.md").write_text(contrastive_diag, encoding="utf-8")
    (reports_dir / "contrastive_verifier_diagnostic.json").write_text(
        json.dumps(contrastive_payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "schema_status": schema["status"],
        "schema_json": str(schema_json),
        "schema_md": str(schema_md),
        "evidence_constraint_diagnostic": str(reports_dir / "evidence_constraint_diagnostic.md"),
        "contrastive_verifier_diagnostic": str(reports_dir / "contrastive_verifier_diagnostic.md"),
        "contrastive_verifier_diagnostic_json": str(reports_dir / "contrastive_verifier_diagnostic.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
