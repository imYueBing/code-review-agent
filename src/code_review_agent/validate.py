"""Minimal JSON Schema validation.

The design doc requires that model output be validated against a schema
(section 5.2, 9.1). Rather than take a runtime dependency, this implements the
small subset of JSON Schema the project actually needs. Unsupported keywords
raise loudly instead of silently passing, so a typo in a schema can never make
validation vacuous.
"""

from __future__ import annotations

from typing import Any

from .errors import CodeReviewError

SUPPORTED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "additionalProperties",
}

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


class SchemaError(CodeReviewError):
    """The schema itself is malformed or uses an unsupported keyword."""


def _check_schema(schema: Any, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise SchemaError(f"schema at {path} must be an object")
    for keyword, value in schema.items():
        if keyword not in SUPPORTED_KEYWORDS:
            raise SchemaError(f"unsupported schema keyword {keyword!r} at {path}")
        if keyword == "properties":
            if not isinstance(value, dict):
                raise SchemaError(f"'properties' at {path} must be an object")
            for name, sub in value.items():
                _check_schema(sub, f"{path}.properties.{name}")
        elif keyword == "items":
            _check_schema(value, f"{path}.items")
        elif keyword == "type":
            names = value if isinstance(value, list) else [value]
            for name in names:
                if name not in _TYPE_MAP:
                    raise SchemaError(f"unsupported type {name!r} at {path}")


def _type_ok(value: Any, name: str) -> bool:
    expected = _TYPE_MAP[name]
    if name in {"integer", "number"} and isinstance(value, bool):
        return False  # bool is an int subclass in Python; reject it here.
    return isinstance(value, expected)


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate *instance*; return a list of human-readable error strings."""
    _check_schema(schema)
    return _validate(instance, schema, path)


def _validate(instance: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []

    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(instance, name) for name in names):
            errors.append(f"{path}: expected type {'/'.join(names)}, got {type(instance).__name__}")
            return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} is not one of {schema['enum']}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: string shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: string longer than maxLength {schema['maxLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} is below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} is above maximum {schema['maximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(instance):
                errors.extend(_validate(item, item_schema, f"{path}[{index}]"))

    if isinstance(instance, dict):
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(f"{path}: missing required property {name!r}")
        properties = schema.get("properties", {})
        for name, sub in properties.items():
            if name in instance:
                errors.extend(_validate(instance[name], sub, f"{path}.{name}"))
        if schema.get("additionalProperties") is False:
            for name in instance:
                if name not in properties:
                    errors.append(f"{path}: unexpected property {name!r}")

    return errors
