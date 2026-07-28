import hashlib
import json
import re
from typing import Any, Dict, List, Union
from ..utils.exceptions import LLMOutputParsingError


def _salvage_objects(text: str) -> List[Dict[str, Any]]:
    """Extract every complete top-level {...} object from text (string-aware brace scan).

    Recovers the parsed prefix of a truncated JSON array (e.g. when generation hits the
    token budget mid-array) instead of discarding the whole batch response.
    """
    out: List[Dict[str, Any]] = []
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            out.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None
    return out


def parse_llm_json(response_text: str) -> Union[Dict[str, Any], List[Any]]:
    output_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:12]
    json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1).strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Array-aware scan (batched responses are JSON arrays, possibly with leading prose).
    start = response_text.find('[')
    end = response_text.rfind(']')
    if start != -1 and end > start:
        try:
            return json.loads(response_text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Object scan (single-candidate / contrastive responses).
    start = response_text.find('{')
    end = response_text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(response_text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Salvage: complete objects out of a truncated array.
    salvaged = _salvage_objects(response_text)
    if salvaged:
        return salvaged

    raise LLMOutputParsingError(
        f"failed_to_parse_json output_hash={output_hash} parser_stage=fallback_scan"
    )
