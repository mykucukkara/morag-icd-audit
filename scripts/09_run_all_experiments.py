"""
Script 09: Run all experiments (E1–E18) across seeds.

Generates the job manifest, then runs each experiment sequentially (or
prints commands for SLURM array job submission).

Usage:
    python scripts/09_run_all_experiments.py \\
        --config configs/experiments.yaml \\
        --top-n 50 \\
        --seeds 42 52 62 \\
        [--smoke-test] \\
        [--dry-run]     # print commands only, don't execute \\
        [--split test|validation]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morag_icd.config import load_config
from morag_icd.slurm.job_manifest import create_job_manifest


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = SCRIPT_DIR / "07_run_experiment.py"


def parse_args():
    p = argparse.ArgumentParser(description="Run all MORAG-ICD experiments.")
    p.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments.yaml")
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths.yaml")
    p.add_argument("--models", default="configs/models.yaml", help="Path to models.yaml")
    p.add_argument("--top-n", type=int, nargs="+", default=[50],
                   help="Top-N codes to run (e.g. 50 100 200)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--output-dir", default="results/predictions")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--allow-mock-llm", action="store_true")
    p.add_argument("--allow-mock-embedding", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--experiments", nargs="+", default=None,
                   help="Run only specific experiment IDs (e.g. E11 E14)")
    return p.parse_args()


def main():
    args = parse_args()
    exp_config = load_config(args.config)
    paths_config = load_config(args.paths) if Path(args.paths).exists() else {}

    experiments = exp_config.get("experiments", [])
    if args.experiments:
        experiments = [e for e in experiments if e["id"] in args.experiments]

    seeds = args.seeds
    top_n_list = args.top_n

    # Create job manifest
    manifest_path = Path("results") / "job_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    create_job_manifest(
        experiments=experiments,
        seeds=seeds,
        top_n_list=top_n_list,
        config_path=str(args.config),
        output_dir=args.output_dir,
        output_path=manifest_path,
    )
    print(f"Job manifest created: {manifest_path}")

    # Build run list
    runs = []
    for top_n in top_n_list:
        for exp in experiments:
            exp_id = exp["id"]
            # E18 (scalability) only runs for specified models on all top_n
            if exp_id == "E18" and top_n == 50:
                continue
            # Non-E18 experiments only run on primary top_n
            if exp_id != "E18" and top_n != top_n_list[0]:
                continue
            for seed in seeds:
                runs.append((exp_id, seed, top_n))

    total = len(runs)
    print(f"\nTotal runs to execute: {total}")
    print(f"Experiments: {[e['id'] for e in experiments]}")
    print(f"Seeds: {seeds}")
    print(f"Top-N: {top_n_list}")
    print()

    if args.smoke_test:
        print("[SMOKE TEST MODE] Using synthetic data\n")

    results_summary = []
    start_time = time.time()

    for i, (exp_id, seed, top_n) in enumerate(runs):
        cmd = [
            sys.executable, str(RUN_SCRIPT),
            "--experiment-id", exp_id,
            "--seed", str(seed),
            "--top-n", str(top_n),
            "--config", args.config,
            "--paths", args.paths,
            "--models", args.models,
            "--split", args.split,
            "--output-dir", args.output_dir,
        ]
        if args.smoke_test:
            cmd.append("--smoke-test")
        if args.allow_mock_llm:
            cmd.append("--allow-mock-llm")
        if args.allow_mock_embedding:
            cmd.append("--allow-mock-embedding")

        print(f"[{i+1}/{total}] {exp_id} | seed={seed} | top_n={top_n}")
        print(f"  Command: {' '.join(cmd)}")

        if args.dry_run:
            dry_cmd = cmd + ["--dry-run"]
            t0 = time.time()
            proc = subprocess.run(dry_cmd, capture_output=True, text=True)
            elapsed = time.time() - t0
            status = "ready" if proc.returncode == 0 else f"dry_run_failed(rc={proc.returncode})"
            print(proc.stdout.strip())
            if proc.stderr.strip():
                print(proc.stderr.strip())
            results_summary.append({
                "exp_id": exp_id, "seed": seed, "top_n": top_n,
                "status": status, "duration_sec": elapsed
            })
            continue

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
            elapsed = time.time() - t0
            status = "success" if proc.returncode == 0 else f"failed(rc={proc.returncode})"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status = "timeout"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"error: {e}"

        print(f"  Status: {status} | Duration: {elapsed:.1f}s\n")
        results_summary.append({
            "exp_id": exp_id, "seed": seed, "top_n": top_n,
            "status": status, "duration_sec": elapsed,
        })

    total_elapsed = time.time() - start_time
    summary = {
        "total_runs": total,
        "completed": sum(1 for r in results_summary if "success" in r["status"]),
        "failed": sum(1 for r in results_summary if "fail" in r["status"] or "error" in r["status"]),
        "dry_run": sum(1 for r in results_summary if r["status"] == "dry_run"),
        "total_runtime_sec": total_elapsed,
        "runs": results_summary,
    }

    summary_path = Path(args.output_dir) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Run Summary ===")
    print(f"Total: {total} | Success: {summary['completed']} | Failed: {summary['failed']}")
    print(f"Total runtime: {total_elapsed:.1f}s")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
