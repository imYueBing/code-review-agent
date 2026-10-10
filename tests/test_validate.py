"""JSON Schema validator tests (design doc 5.2, 9.1)."""

from __future__ import annotations

import pytest

from code_review_agent.validate import SchemaError, validate


class TestSupportedKeywords:
    def test_type_match(self):
        assert validate("x", {"type": "string"}) == []

    def test_type_mismatch_reports_path(self):
        errors = validate(5, {"type": "string"})
        assert len(errors) == 1
        assert "$" in errors[0] and "string" in errors[0]

    def test_integer_rejects_bool(self):
        # bool is an int subclass in Python; it must not satisfy "integer".
        assert validate(True, {"type": "integer"}) != []

    def test_number_accepts_int_and_float(self):
        assert validate(1, {"type": "number"}) == []
        assert validate(1.5, {"type": "number"}) == []

    def test_union_type(self):
        assert validate(None, {"type": ["string", "null"]}) == []
        assert validate(3, {"type": ["string", "null"]}) != []

    def test_enum(self):
        assert validate("high", {"enum": ["high", "low"]}) == []
        assert validate("nope", {"enum": ["high", "low"]}) != []

    def test_numeric_bounds(self):
        assert validate(1, {"type": "integer", "minimum": 1, "maximum": 3}) == []
        assert validate(0, {"type": "integer", "minimum": 1}) != []
        assert validate(4, {"type": "integer", "maximum": 3}) != []

    def test_string_length(self):
        assert validate("ab", {"type": "string", "minLength": 2}) == []
        assert validate("", {"type": "string", "minLength": 1}) != []

    def test_required_and_nested_path(self):
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "integer"}},
        }
        errors = validate({"a": "text"}, schema)
        assert len(errors) == 1
        assert "$.a" in errors[0]
        assert validate({}, schema)[0].find("missing required") != -1

    def test_additional_properties_false(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": False}
        assert validate({"a": 1}, schema) == []
        assert validate({"a": 1, "b": 2}, schema) != []

    def test_array_items_and_bounds(self):
        schema = {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2}
        assert validate([1, 2], schema) == []
        assert validate([], schema) != []
        assert validate([1, 2, 3], schema) != []
        errors = validate([1, "x"], schema)
        assert len(errors) == 1 and "$[1]" in errors[0]


class TestSchemaSelfCheck:
    def test_unsupported_keyword_raises(self):
        # A typo must fail loudly rather than silently skipping validation.
        with pytest.raises(SchemaError, match="unsupported schema keyword"):
            validate({}, {"type": "object", "requird": ["a"]})

    def test_unknown_type_raises(self):
        with pytest.raises(SchemaError, match="unsupported type"):
            validate({}, {"type": "strng"})

    def test_nested_schema_is_checked(self):
        with pytest.raises(SchemaError):
            validate({}, {"type": "object", "properties": {"a": {"type": "wrong"}}})

    def test_type_errors_short_circuit_nested_checks(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        # A string where an object is expected should not also report the
        # property-level error.
        assert len(validate("nope", schema)) == 1
