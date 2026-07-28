"""
SLURM job manifest generator with full column set.

Creates a CSV file with all required columns for array job submission:
    array_id, experiment_id, seed, top_n_codes, config_path, output_dir
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List, Optional


def create_job_manifest(
    experiments: List[dict],
    seeds: List[int],
    top_n_list: Optional[List[int]] = None,
    config_path: str = "configs/experiments.yaml",
    output_dir: str = "results/predictions",
    output_path: str | Path = "results/job_manifest.csv",
) -> pd.DataFrame:
    """
    Create a job manifest CSV for SLURM array job submission.

    Parameters
    ----------
    experiments : list of dicts with 'id' key
    seeds : list of ints
    top_n_list : list of ints (default [50])
    config_path : str
    output_dir : str
    output_path : path to save the CSV

    Returns
    -------
    pd.DataFrame with manifest rows.
    """
    if top_n_list is None:
        top_n_list = [50]

    SCALABILITY_EXPS = {"E18"}
    SCALABILITY_MODELS = {"E2", "E11", "E14", "E16"}  # best classical, hybrid-RAG, full, best opt

    jobs = []
    idx = 1

    for top_n in top_n_list:
        for exp in experiments:
            exp_id = exp["id"]
            exp_name = exp.get("name", exp_id)

            # E18 runs on all top_n with only 4 models
            if exp_id in SCALABILITY_EXPS:
                for seed in seeds:
                    for model_id in SCALABILITY_MODELS:
                        jobs.append({
                            "array_id": idx,
                            "experiment_id": f"E18_{model_id}",
                            "experiment_name": f"scalability_{model_id}",
                            "seed": seed,
                            "top_n_codes": top_n,
                            "config_path": config_path,
                            "output_dir": output_dir,
                        })
                        idx += 1
                continue

            # Non-E18: only primary top_n (first in list)
            if top_n != top_n_list[0]:
                continue

            for seed in seeds:
                jobs.append({
                    "array_id": idx,
                    "experiment_id": exp_id,
                    "experiment_name": exp_name,
                    "seed": seed,
                    "top_n_codes": top_n,
                    "config_path": config_path,
                    "output_dir": output_dir,
                })
                idx += 1

    df = pd.DataFrame(jobs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
