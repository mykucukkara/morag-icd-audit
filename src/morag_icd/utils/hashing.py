import hashlib
import json
from typing import Any, Dict


def hash_dict(d: Dict) -> str:
    """MD5 hash of a dictionary (sorted keys)."""
    s = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(s).hexdigest()


def hash_text(text: str) -> str:
    """MD5 hash of a string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def hash_config(config: Dict) -> str:
    """Short 8-char hash for a config dict (for reproducibility tracking)."""
    return hash_dict(config)[:8]


def hash_text_short(text: str, length: int = 8) -> str:
    """Short hash of text for log sanitization."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]
