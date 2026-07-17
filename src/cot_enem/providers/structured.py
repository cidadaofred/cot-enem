"""Utilities for extracting and minimally validating structured model output."""

import json
import re
from typing import Any

from cot_enem.providers.errors import StructuredResponseError

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _escape_invalid_json_backslashes(value: str) -> str:
    """Preserve JSON escapes and quote LaTeX-style backslashes as literals."""

    repaired: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "\\":
            repaired.append(character)
            index += 1
            continue
        following = value[index + 1] if index + 1 < len(value) else ""
        valid_simple_escape = following in {'"', "\\", "/", "b", "f", "n", "r", "t"}
        valid_unicode_escape = (
            following == "u"
            and index + 5 < len(value)
            and all(char in "0123456789abcdefABCDEF" for char in value[index + 2 : index + 6])
        )
        repaired.append("\\" if valid_simple_escape or valid_unicode_escape else "\\\\")
        index += 1
    return "".join(repaired)


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object, accepting fences or short surrounding model prose."""

    candidate = _FENCE.sub(r"\1", content).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        repaired_candidate = _escape_invalid_json_backslashes(candidate)
        try:
            value = json.loads(repaired_candidate)
        except json.JSONDecodeError:
            value = None
        decoder = json.JSONDecoder()
        if value is None:
            for start, character in enumerate(repaired_candidate):
                if character != "{":
                    continue
                try:
                    decoded, _end = decoder.raw_decode(repaired_candidate, start)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    value = decoded
                    break
        if value is None:
            preview = candidate[:160].replace("\n", " ")
            raise StructuredResponseError(
                "model response is not valid JSON "
                f"(line {exc.lineno}, column {exc.colno}; preview={preview!r})"
            ) from exc
    if not isinstance(value, dict):
        raise StructuredResponseError("structured model response must be a JSON object")
    return value


def require_schema_keys(value: dict[str, Any], schema: dict[str, Any]) -> None:
    """Check required top-level keys; full JSON Schema validation stays optional."""

    required = schema.get("required", [])
    missing = [key for key in required if key not in value]
    if missing:
        raise StructuredResponseError(f"structured response is missing keys: {missing}")
