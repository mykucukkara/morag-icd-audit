"""IO utilities: JSONL, Parquet, and general file helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


def read_jsonl(filepath: str | Path) -> Generator[Dict, None, None]:
    """Yield records from a JSONL file (generator)."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(filepath: str | Path) -> List[Dict]:
    """Load all records from a JSONL file into a list."""
    return list(read_jsonl(filepath))


def write_jsonl(filepath: str | Path, data: List[Dict], mode: str = "w") -> None:
    """Write records to a JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, default=str) + "\n")


def load_parquet_or_jsonl(filepath: str | Path) -> List[Dict]:
    """
    Load data from a Parquet file (preferred) or JSONL fallback.

    Returns
    -------
    list of dicts.
    """
    filepath = Path(filepath)

    # Try parquet first
    parquet_path = filepath.with_suffix(".parquet") if filepath.suffix != ".parquet" else filepath
    if parquet_path.exists():
        try:
            import pandas as pd
            return pd.read_parquet(parquet_path).to_dict(orient="records")
        except ImportError:
            pass

    # Try JSONL
    jsonl_path = filepath.with_suffix(".jsonl") if filepath.suffix != ".jsonl" else filepath
    if jsonl_path.exists():
        return load_jsonl(jsonl_path)

    # Try original path
    if filepath.exists():
        if filepath.suffix == ".jsonl":
            return load_jsonl(filepath)
        elif filepath.suffix in (".parquet", ".pq"):
            import pandas as pd
            return pd.read_parquet(filepath).to_dict(orient="records")

    raise FileNotFoundError(f"No data file found at: {filepath} (tried .parquet and .jsonl)")


def save_json(obj: Any, filepath: str | Path, indent: int = 2) -> None:
    """Save any JSON-serializable object to a file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(filepath: str | Path) -> Any:
    """Load a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
