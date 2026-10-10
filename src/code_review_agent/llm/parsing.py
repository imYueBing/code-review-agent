"""Turning raw model text into a JSON object.

Models reliably wrap JSON in prose or fenced blocks. This module extracts the
first complete JSON value without depending on the whole response being valid
JSON, which is what makes the "retry with repair" path (section 5.3) possible.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import LLMResponseFormatError

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _scan_json_value(text: str, start: int) -> str | None:
    """Return the balanced JSON substring beginning at *start*, if any.

    String literals and escapes are tracked so braces inside strings do not
    unbalance the count.
    """
    opener = text[start]
    if opener not in "{[":
        return None
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from *text*, raising on failure.

    ``json.JSONDecodeError`` is a subclass of ``ValueError`` rather than of our
    error hierarchy, so nothing provider-specific can leak into the workflow.
    """
    if text is None or not text.strip():
        raise LLMResponseFormatError("model returned an empty response")

    stripped = text.strip()

    # 1. Whole response is the object.
    candidates: list[str] = [stripped]

    # 2. Fenced code blocks, most likely first.
    candidates.extend(block.strip() for block in _FENCE_RE.findall(text))

    # 3. Balanced scan from the first brace/bracket of each candidate region.
    for source in (stripped, *_FENCE_RE.findall(text)):
        for index, char in enumerate(source):
            if char in "{[":
                found = _scan_json_value(source, index)
                if found:
                    candidates.append(found)
                break

    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        # A bare array is accepted and wrapped, since the schema expects an
        # object with a "findings" list.
        if isinstance(parsed, list):
            return {"findings": parsed}
        errors.append(f"top-level JSON was {type(parsed).__name__}, expected object")

    preview = stripped[:300].replace("\n", " ")
    detail = errors[0] if errors else "no JSON value found"
    raise LLMResponseFormatError(
        f"could not parse JSON from model response ({detail}); response began: {preview!r}"
    )


def repair_prompt(original_response: str, errors: list[str], expected_schema: dict[str, Any]) -> str:
    """Instruction used for the schema-repair retry."""
    joined = "\n".join(f"- {error}" for error in errors[:10])
    return (
        "Your previous response could not be accepted.\n\n"
        "Validation errors:\n"
        f"{joined}\n\n"
        "Return ONLY a corrected JSON object matching this schema. "
        "No prose, no markdown fences.\n\n"
        f"Schema:\n{json.dumps(expected_schema, indent=2)}\n\n"
        "Previous response:\n"
        f"{original_response[:4000]}"
    )
