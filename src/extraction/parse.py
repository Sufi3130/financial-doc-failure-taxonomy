"""Lenient JSON parsing of raw model output, schema validation, and failure recording.

Every way an output deviates from the schema becomes a Failure entry, so the
failure taxonomy can count them. Repairs applied to get parseable JSON are
recorded separately: a repaired output is usable but not schema-compliant.

Failure types:
    prompt_too_long  prompt + max_tokens exceeded the context window (no generation)
    truncated        generation stopped at max_tokens
    no_json          no '{' in the output
    invalid_json     an object was found but could not be parsed, even after repairs
    not_object       valid JSON, but a list/string/number instead of an object
    missing_field    schema key absent
    null_field       key present but null or empty (allowed by the schema; recorded for analysis)
    wrong_type       value is not a string, e.g. "total": 193.0
    extra_field      key not in the schema (hallucinated field)
"""

import ast
import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from .schema import FIELDS, Receipt

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)(?:```|$)", re.S)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class Failure(BaseModel):
    type: str
    field: str | None = None
    detail: str | None = None


class ParseOutcome(BaseModel):
    parsed: Any = None  # the JSON value as parsed (after repairs), before validation
    repairs: list[str] = []  # code_fence, extra_text, trailing_comma, python_literal
    failures: list[Failure] = []
    schema_valid: bool = False  # parsed object passes the Receipt schema
    strict_valid: bool = False  # schema_valid AND no repairs AND not truncated
    lenient: dict[str, str | None] | None = None  # best usable values for scoring


def _first_object(text):
    """Return the first balanced {...} substring (string-aware), or the
    unclosed tail from the first '{' if it never closes, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def lenient_json(raw):
    """Return (value, repairs, error). value is None when nothing could be parsed;
    error is 'no_json' or an 'invalid_json: ...' message in that case."""
    text = raw.strip()
    repairs = []
    try:
        return json.loads(text), repairs, None
    except json.JSONDecodeError:
        pass

    if (m := _FENCE.search(text)) and "```" in text:
        text = m[1].strip()
        repairs.append("code_fence")
        try:
            return json.loads(text), repairs, None
        except json.JSONDecodeError:
            pass

    candidate = _first_object(text)
    if candidate is None:
        return None, repairs, "no_json"
    if candidate.strip() != text:
        repairs.append("extra_text")

    try:
        return json.loads(candidate), repairs, None
    except json.JSONDecodeError as e:
        error = f"invalid_json: {e.msg} at char {e.pos}"

    fixed = _TRAILING_COMMA.sub(r"\1", candidate)
    if fixed != candidate:
        try:
            return json.loads(fixed), repairs + ["trailing_comma"], None
        except json.JSONDecodeError:
            pass

    try:  # single quotes, None/True: a Python dict literal
        value = ast.literal_eval(fixed)
        if isinstance(value, dict):
            return value, repairs + ["python_literal"], None
    except (ValueError, SyntaxError):
        pass
    return None, repairs, error


def _lenient_value(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return " ".join(x.strip() for x in v if x.strip()) or None  # e.g. address split into lines
    return None


def parse_output(raw, finish_reason=None) -> ParseOutcome:
    out = ParseOutcome()
    if finish_reason == "prompt_too_long":
        out.failures.append(Failure(type="prompt_too_long"))
        return out
    if finish_reason == "length":
        out.failures.append(Failure(type="truncated"))

    value, out.repairs, error = lenient_json(raw or "")
    if value is None:
        out.failures.append(Failure(type="no_json" if error == "no_json" else "invalid_json",
                                    detail=None if error == "no_json" else error))
        return out
    out.parsed = value

    if not isinstance(value, dict):
        out.failures.append(Failure(type="not_object", detail=type(value).__name__))
        dicts = [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []
        if not dicts:
            return out
        value = dicts[0]  # keep going with the first object so fields can still be scored

    try:
        Receipt.model_validate(value)
        out.schema_valid = isinstance(out.parsed, dict)
    except ValidationError as e:
        seen = set()
        for err in e.errors():
            field = str(err["loc"][0]) if err["loc"] else None
            kind = {"missing": "missing_field", "extra_forbidden": "extra_field"}.get(err["type"], "wrong_type")
            if (kind, field) in seen:
                continue
            seen.add((kind, field))
            detail = None if kind == "missing_field" else type(value.get(field)).__name__
            if kind == "extra_field":
                detail = repr(value.get(field))[:80]
            out.failures.append(Failure(type=kind, field=field, detail=detail))

    for field in FIELDS:
        if field in value and (value[field] is None or (isinstance(value[field], str) and not value[field].strip())):
            out.failures.append(Failure(type="null_field", field=field,
                                        detail="null" if value[field] is None else "empty string"))

    out.strict_valid = out.schema_valid and not out.repairs and finish_reason != "length"
    out.lenient = {f: _lenient_value(value.get(f)) for f in FIELDS}
    return out
