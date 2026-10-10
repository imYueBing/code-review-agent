"""Tests for JSON extraction from messy model output (design doc 5.3)."""

from __future__ import annotations

import pytest

from code_review_agent.errors import LLMResponseFormatError
from code_review_agent.llm.parsing import extract_json_object, repair_prompt


class TestExtractJsonObject:
    def test_plain_object(self):
        assert extract_json_object('{"findings": []}') == {"findings": []}

    def test_object_with_surrounding_whitespace(self):
        assert extract_json_object('\n\n  {"a": 1}  \n') == {"a": 1}

    def test_fenced_json_block(self):
        text = 'Here you go:\n```json\n{"findings": [{"title": "x"}]}\n```\nHope that helps!'
        assert extract_json_object(text)["findings"][0]["title"] == "x"

    def test_fenced_block_without_language_tag(self):
        assert extract_json_object("```\n{\"a\": 2}\n```") == {"a": 2}

    def test_prose_before_and_after_object(self):
        text = 'Sure! The result is {"a": 1, "b": [1,2]} and that is all.'
        assert extract_json_object(text) == {"a": 1, "b": [1, 2]}

    def test_braces_inside_strings_do_not_confuse_scanner(self):
        text = 'prefix {"note": "contains } and { braces", "n": 1} suffix'
        assert extract_json_object(text)["n"] == 1

    def test_escaped_quotes_inside_strings(self):
        text = r'{"note": "a \" quote", "n": 2}'
        assert extract_json_object(text)["n"] == 2

    def test_bare_array_is_wrapped_as_findings(self):
        assert extract_json_object('[{"title": "x"}]') == {"findings": [{"title": "x"}]}

    def test_nested_objects(self):
        text = '{"findings": [{"a": {"b": {"c": 1}}}]}'
        assert extract_json_object(text)["findings"][0]["a"]["b"]["c"] == 1

    def test_empty_response_raises(self):
        with pytest.raises(LLMResponseFormatError, match="empty"):
            extract_json_object("")

    def test_whitespace_only_raises(self):
        with pytest.raises(LLMResponseFormatError):
            extract_json_object("   \n  ")

    def test_no_json_at_all_raises(self):
        with pytest.raises(LLMResponseFormatError):
            extract_json_object("I could not review this file, sorry.")

    def test_unterminated_object_raises(self):
        with pytest.raises(LLMResponseFormatError):
            extract_json_object('{"findings": [')

    def test_scalar_json_is_rejected(self):
        with pytest.raises(LLMResponseFormatError):
            extract_json_object('"just a string"')

    def test_error_message_includes_preview(self):
        with pytest.raises(LLMResponseFormatError) as excinfo:
            extract_json_object("totally not json")
        assert "totally not json" in str(excinfo.value)

    def test_unicode_content_survives(self):
        text = '{"title": "错误：空值访问", "n": 1}'
        assert extract_json_object(text)["title"] == "错误：空值访问"


class TestRepairPrompt:
    def test_includes_errors_and_schema(self):
        prompt = repair_prompt('{"findings": "nope"}', ["$.findings: expected array"], {"type": "object"})
        assert "expected array" in prompt
        assert '"type"' in prompt
        assert "findings" in prompt

    def test_truncates_huge_previous_response(self):
        prompt = repair_prompt("x" * 20_000, ["e"], {"type": "object"})
        assert len(prompt) < 20_000

    def test_limits_listed_errors(self):
        prompt = repair_prompt("{}", [f"error-{i}" for i in range(50)], {"type": "object"})
        assert "error-9" in prompt
        assert "error-40" not in prompt
