import os
import yaml
from pathlib import Path

def _replace_vars(node, vars_dict):
    if isinstance(node, dict):
        return {k: _replace_vars(v, vars_dict) for k, v in node.items()}
    elif isinstance(node, list):
        return [_replace_vars(v, vars_dict) for v in node]
    elif isinstance(node, str):
        for k, v in vars_dict.items():
            node = node.replace(f"${{{k}}}", str(v))
        return node
    return node

def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    # First extract variables if they exist in the root
    vars_dict = {}
    for k, v in data.items():
        if isinstance(v, str) and not v.startswith("${"):
            vars_dict[k] = v
            
    # Then replace them
    return _replace_vars(data, vars_dict)
