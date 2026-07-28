"""Logging utilities with JSONL structured logging and experiment context."""
from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as JSONL for structured experiment logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }
        for attr in ["experiment_id", "seed", "stage", "subject_id", "hadm_id",
                     "error_type", "runtime_sec", "memory_mb", "gpu_memory_mb"]:
            if hasattr(record, attr):
                log_record[attr] = getattr(record, attr)
        if record.exc_info:
            log_record["stacktrace"] = self.formatException(record.exc_info)
        return json.dumps(log_record, default=str)


def setup_logger(
    name: str,
    log_dir: str | Path,
    level: int = logging.INFO,
    experiment_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> logging.Logger:
    """
    Set up a logger with both JSONL file and console handlers.

    Parameters
    ----------
    name : str
        Logger name (unique per experiment+seed).
    log_dir : str | Path
        Directory for log files.
    level : int
        Logging level.
    experiment_id : str, optional
        Experiment ID for log file naming.
    seed : int, optional
        Seed for log file naming.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)
    log_dir = Path(log_dir)

    # Experiment-specific log directory
    if experiment_id and seed is not None:
        exp_log_dir = log_dir / "experiments" / experiment_id / f"seed_{seed}"
    else:
        exp_log_dir = log_dir

    exp_log_dir.mkdir(parents=True, exist_ok=True)

    # JSONL structured log
    jsonl_handler = logging.FileHandler(exp_log_dir / "run.jsonl", encoding="utf-8")
    jsonl_handler.setFormatter(JSONFormatter())
    logger.addHandler(jsonl_handler)

    # Plain text log
    txt_handler = logging.FileHandler(exp_log_dir / "run.log", encoding="utf-8")
    txt_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(txt_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(console_handler)

    return logger
