"""Agent tests: prompt construction, sandboxed tools, review loop (doc 5.2, 8)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from code_review_agent.agent.prompts import SYSTEM_PROMPT, build_file_context, render_tool_results
from code_review_agent.agent.reviewer import ReviewAgent
from code_review_agent.agent.tools import ProjectToolbelt, ToolCall
from code_review_agent.llm.client import BudgetGuard
from code_review_agent.security import PathGuard
from code_review_agent.tools.reader import CodeReader, SourceFile
from code_review_agent.tools.rules import Diagnostic
from code_review_agent.tools.scanner import ScannedFile

from conftest import ScriptedClient, findings_payload, make_finding_json, routed


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "useUser.ts").write_text(
        "export function useUser(user: any) {\n"
        "  const name = user.profile.name;\n"
        "  return { name };\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "src" / "other.ts").write_text(
        "export const helper = 1;\nexport function other() { return helper; }\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def guard(project: Path) -> PathGuard:
    return PathGuard(project)


@pytest.fixture
def reader(guard: PathGuard) -> CodeReader:
    return CodeReader(guard)


@pytest.fixture
def scanned(project: Path) -> list[ScannedFile]:
    return [
        ScannedFile(
            path=project / "src" / "useUser.ts",
            relative_path="src/useUser.ts",
            size_bytes=100,
            line_count=4,
        ),
        ScannedFile(
            path=project / "src" / "other.ts",
            relative_path="src/other.ts",
            size_bytes=60,
            line_count=2,
        ),
    ]


class TestSystemPrompt:
    def test_declares_code_is_untrusted(self):
        assert "UNTRUSTED" in SYSTEM_PROMPT

    def test_states_read_only(self):
        assert "read-only" in SYSTEM_PROMPT.lower()

    def test_forbids_reproducing_secrets(self):
        assert "[REDACTED]" in SYSTEM_PROMPT

    def test_requires_verbatim_evidence(self):
        assert "verbatim" in SYSTEM_PROMPT.lower()

    def test_documents_severity_vocabulary(self):
        for severity in ("critical", "high", "medium", "low", "info"):
            assert severity in SYSTEM_PROMPT

    def test_permits_an_empty_result(self):
        assert "empty findings" in SYSTEM_PROMPT


class TestBuildFileContext:
    def test_includes_file_path_and_numbered_source(self):
        source = SourceFile("src/a.ts", "const a = 1;\nconst b = 2;\n")
        context = build_file_context(source, [], max_chars=10_000)
        assert "src/a.ts" in context
        assert "1 | const a = 1;" in context
        assert "2 | const b = 2;" in context

    def test_fences_source_as_data(self):
        source = SourceFile("src/a.ts", "const a = 1;\n")
        context = build_file_context(source, [], max_chars=10_000)
        assert '<SOURCE_FILE path="src/a.ts">' in context
        assert "</SOURCE_FILE>" in context

    def test_includes_diagnostics_with_line_numbers(self):
        source = SourceFile("src/a.ts", "eval('x');\n")
        diagnostic = Diagnostic(
            file="src/a.ts", line=1, column=1, severity="error",
            rule_id="no-eval", message="dynamic code execution",
        )
        context = build_file_context(source, [diagnostic], max_chars=10_000)
        assert "no-eval" in context
        assert "line 1" in context

    def test_labels_diagnostics_as_untrusted(self):
        source = SourceFile("src/a.ts", "x\n")
        diagnostic = Diagnostic(
            file="src/a.ts", line=1, column=1, severity="warning",
            rule_id="r", message="m",
        )
        assert "untrusted" in build_file_context(source, [diagnostic], max_chars=10_000).lower()

    def test_truncates_to_budget_and_says_so(self):
        big = "\n".join(f"const v{i} = {i};" for i in range(5_000))
        source = SourceFile("big.ts", big)
        context = build_file_context(source, [], max_chars=1_000)
        assert len(context) <= 1_200
        assert "omitted" in context

    def test_notes_when_source_is_truncated(self):
        source = SourceFile("src/a.ts", "a\n", truncated=True)
        assert "truncated" in build_file_context(source, [], max_chars=10_000).lower()

    def test_tiny_budget_still_returns_header(self):
        source = SourceFile("src/a.ts", "x\n")
        context = build_file_context(source, [], max_chars=10)
        assert "src/a.ts" in context


class TestProjectToolbelt:
    def test_lists_available_tools(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        assert set(belt.tool_names) == {"search_symbol", "read_lines", "file_outline"}
        assert len(belt.describe()) == 3

    def test_search_symbol_finds_definition(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("search_symbol", {"query": "useUser"}))
        assert result["ok"] is True
        assert any(hit["file"] == "src/useUser.ts" for hit in result["hits"])

    def test_search_symbol_rejects_empty_query(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("search_symbol", {"query": ""}))
        assert result["ok"] is False

    def test_search_symbol_rejects_non_identifier(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("search_symbol", {"query": "a; rm -rf /"}))
        assert result["ok"] is False
        assert "identifier-like" in result["error"]

    def test_search_is_bounded(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("search_symbol", {"query": "const"}))
        assert len(result["hits"]) <= ProjectToolbelt.MAX_SEARCH_HITS

    def test_read_lines_returns_numbered_content(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(
            ToolCall("read_lines", {"file": "src/useUser.ts", "start_line": 1, "end_line": 2})
        )
        assert result["ok"] is True
        assert "1 |" in result["content"]

    def test_read_lines_rejects_unreviewed_file(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("read_lines", {"file": "src/secret.ts"}))
        assert result["ok"] is False

    def test_read_lines_rejects_escape(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("read_lines", {"file": "/etc/passwd"}))
        assert result["ok"] is False

    def test_read_lines_clamps_window(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(
            ToolCall("read_lines", {"file": "src/useUser.ts", "start_line": 1, "end_line": 100_000})
        )
        assert result["ok"] is True
        assert result["end_line"] - result["start_line"] + 1 <= ProjectToolbelt.MAX_READ_LINES

    def test_read_lines_normalises_reversed_range(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(
            ToolCall("read_lines", {"file": "src/useUser.ts", "start_line": 3, "end_line": 1})
        )
        assert result["start_line"] == 1 and result["end_line"] == 3

    def test_read_lines_rejects_bad_types(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(
            ToolCall("read_lines", {"file": "src/useUser.ts", "start_line": "abc"})
        )
        assert result["ok"] is False

    def test_file_outline_lists_declarations(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("file_outline", {"file": "src/useUser.ts"}))
        assert result["ok"] is True
        assert any(item["name"] == "useUser" for item in result["items"])

    def test_unknown_tool_is_refused(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("run_shell", {"cmd": "rm -rf /"}))
        assert result["ok"] is False
        assert "unknown tool" in result["error"]

    def test_results_are_size_limited(self, guard, reader, scanned, base_config):
        belt = ProjectToolbelt(guard, reader, scanned, base_config)
        result = belt.execute(ToolCall("search_symbol", {"query": "e"}))
        import json

        assert len(json.dumps(result)) <= ProjectToolbelt.MAX_RESULT_CHARS + 500


class TestToolCallParsing:
    def test_parses_tool_calls_list(self):
        calls = ToolCall.parse_many({"tool_calls": [{"tool": "read_lines", "args": {"file": "a"}}]}, limit=5)
        assert calls[0].tool == "read_lines"
        assert calls[0].args == {"file": "a"}

    def test_respects_limit(self):
        payload = {"tool_calls": [{"tool": "t", "args": {}} for _ in range(10)]}
        assert len(ToolCall.parse_many(payload, limit=2)) == 2

    def test_discards_malformed_entries(self):
        payload = {"tool_calls": ["nope", {"no_tool": 1}, {"tool": "ok", "args": {}}]}
        calls = ToolCall.parse_many(payload, limit=5)
        assert [c.tool for c in calls] == ["ok"]

    def test_non_list_payload_is_empty(self):
        assert ToolCall.parse_many({"tool_calls": "bad"}, limit=5) == []
        assert ToolCall.parse_many(None, limit=5) == []

    def test_accepts_name_and_arguments_aliases(self):
        calls = ToolCall.parse_many({"tool_calls": [{"name": "x", "arguments": {"a": 1}}]}, limit=5)
        assert calls[0].tool == "x" and calls[0].args == {"a": 1}


class TestRenderToolResults:
    def test_labels_results_as_untrusted(self):
        assert "untrusted" in render_tool_results([{"ok": True}]).lower()


def build_agent(client, guard, reader, config, scanned, *, toolbelt=False):
    """Build an agent; tool use is enabled exactly when a toolbelt is supplied."""
    effective = replace(config, review=replace(config.review, enable_tool_calls=toolbelt))
    belt = ProjectToolbelt(guard, reader, scanned, effective) if toolbelt else None
    return ReviewAgent(client, reader, effective, BudgetGuard(effective), belt)


class TestReviewAgent:
    def test_accepts_a_valid_finding(self, guard, reader, base_config, scanned):
        # Only useUser.ts yields a finding; other.ts correctly reports none.
        client = routed(
            ("review:src/useUser.ts", findings_payload(make_finding_json())),
            ("review:src/other.ts", '{"findings": []}'),
        )
        agent = build_agent(client, guard, reader, base_config, scanned)
        result = agent.review_files(scanned)
        assert len(result.findings) == 1
        assert result.findings[0].title == "Potential null dereference"
        assert result.files_reviewed == 2

    def test_empty_findings_is_a_valid_outcome(self, guard, reader, base_config, scanned):
        client = ScriptedClient(default='{"findings": []}')
        agent = build_agent(client, guard, reader, base_config, scanned)
        result = agent.review_files(scanned)
        assert result.findings == []
        assert result.files_reviewed == 2

    def test_prompt_contains_numbered_source(self, guard, reader, base_config, scanned):
        client = ScriptedClient(default='{"findings": []}')
        agent = build_agent(client, guard, reader, base_config, scanned)
        agent.review_files(scanned)
        assert "1 | export function useUser" in client.prompts[0]

    def test_malformed_finding_entries_are_dropped(self, guard, reader, base_config, scanned):
        # An unparseable severity cannot be coerced into the vocabulary, so the
        # entry is dropped at parse time while the valid one survives.
        payload = findings_payload(make_finding_json(), {"title": "bad severity", "severity": "nope"})
        client = routed(("review:src/useUser.ts", payload), default='{"findings": []}')
        agent = build_agent(client, guard, reader, base_config, scanned)
        result = agent.review_files(scanned)
        assert len(result.findings) == 1
        assert result.findings[0].title == "Potential null dereference"

    def test_non_object_findings_are_dropped(self, guard, reader, base_config, scanned):
        import json

        client = routed(
            ("review:src/useUser.ts", json.dumps({"findings": ["not an object", 42]})),
            default='{"findings": []}',
        )
        agent = build_agent(client, guard, reader, base_config, scanned)
        result = agent.review_files(scanned)
        assert result.findings == []

    def test_low_confidence_findings_are_filtered(self, guard, reader, base_config, scanned):
        config = replace(base_config, review=replace(base_config.review, min_confidence=0.8))
        client = routed(
            ("review:src/useUser.ts", findings_payload(make_finding_json(confidence=0.3))),
            default='{"findings": []}',
        )
        agent = build_agent(client, guard, reader, config, scanned)
        result = agent.review_files(scanned)
        assert result.findings == []

    def test_findings_per_file_are_capped(self, guard, reader, base_config, scanned):
        config = replace(base_config, review=replace(base_config.review, max_findings_per_file=1))
        payload = findings_payload(
            make_finding_json(title="First"),
            make_finding_json(title="Second", start_line=3, evidence="return { name };"),
        )
        client = routed(("review:src/useUser.ts", payload), default='{"findings": []}')
        agent = build_agent(client, guard, reader, config, scanned)
        assert len(agent.review_files(scanned).findings) == 1

    def test_model_failure_skips_file_without_aborting(self, guard, reader, base_config, scanned):
        from code_review_agent.errors import LLMResponseFormatError

        class Exploding(ScriptedClient):
            def complete_json(self, request):
                raise LLMResponseFormatError("model produced garbage")

        agent = build_agent(Exploding(), guard, reader, base_config, scanned)
        result = agent.review_files(scanned)
        assert result.files_reviewed == 0
        assert len(result.errors) == 2
        assert all(review.skipped_reason == "llm-failure" for review in result.file_reviews)

    def test_unreadable_file_is_skipped(self, guard, reader, base_config, project):
        missing = ScannedFile(
            path=project / "src" / "gone.ts",
            relative_path="src/gone.ts",
            size_bytes=10,
            line_count=1,
        )
        client = ScriptedClient(default='{"findings": []}')
        agent = build_agent(client, guard, reader, base_config, [missing])
        result = agent.review_files([missing])
        assert result.files_reviewed == 0
        assert result.file_reviews[0].skipped_reason == "unreadable"

    def test_second_round_runs_when_first_finds_nothing(self, guard, reader, base_config, scanned):
        config = replace(base_config, review=replace(base_config.review, max_review_rounds_per_file=2))
        client = routed(
            ("review_round2:src/useUser.ts", findings_payload(make_finding_json())),
            ("review_round2:src/other.ts", '{"findings": []}'),
            default='{"findings": []}',
        )
        agent = build_agent(client, guard, reader, config, scanned)
        result = agent.review_files(scanned)
        # Round 2 supplied a finding for useUser.ts only.
        assert len(result.findings) == 1
        assert sum("review_round2" in purpose for purpose in client.purposes) == 2

    def test_no_second_round_when_first_finds_something(self, guard, reader, base_config, scanned):
        config = replace(base_config, review=replace(base_config.review, max_review_rounds_per_file=2))
        client = ScriptedClient(default=findings_payload(make_finding_json()))
        agent = build_agent(client, guard, reader, config, scanned)
        agent.review_files(scanned)
        assert not any("review_round2" in purpose for purpose in client.purposes)

    def test_tool_calls_are_executed_and_fed_back(self, guard, reader, base_config, scanned):
        client = routed(
            (
                "tool_planning",
                '{"tool_calls": [{"tool": "search_symbol", "args": {"query": "helper"}}]}',
            ),
            ("review:src/useUser.ts", findings_payload(make_finding_json())),
            default='{"findings": []}',
        )
        agent = build_agent(client, guard, reader, base_config, scanned, toolbelt=True)
        result = agent.review_files(scanned)
        assert any("TOOL RESULTS" in prompt for prompt in client.prompts)
        assert sum(review.tool_calls for review in result.file_reviews) == 2

    def test_tool_calls_are_capped_per_file(self, guard, reader, base_config, scanned):
        plan = (
            '{"tool_calls": ['
            '{"tool": "search_symbol", "args": {"query": "a"}},'
            '{"tool": "search_symbol", "args": {"query": "b"}},'
            '{"tool": "search_symbol", "args": {"query": "c"}}]}'
        )
        client = routed(("tool_planning", plan), default='{"findings": []}')
        agent = build_agent(client, guard, reader, base_config, scanned, toolbelt=True)
        result = agent.review_files(scanned)
        assert all(
            review.tool_calls <= base_config.review.max_tool_calls_per_file
            for review in result.file_reviews
        )

    def test_tool_planning_failure_does_not_break_review(self, guard, reader, base_config, scanned):
        client = routed(("tool_planning", "not json at all"), default='{"findings": []}')
        agent = build_agent(client, guard, reader, base_config, scanned, toolbelt=True)
        result = agent.review_files(scanned)
        assert result.findings == []
        assert result.files_reviewed == 2

    def test_agent_includes_static_diagnostics_in_prompt(self, guard, reader, base_config, scanned):
        client = ScriptedClient(default='{"findings": []}')
        agent = build_agent(client, guard, reader, base_config, scanned)
        agent.set_diagnostics(
            {
                "src/useUser.ts": [
                    Diagnostic(
                        file="src/useUser.ts", line=2, column=1, severity="warning",
                        rule_id="no-console-debug", message="console logging left in source",
                    )
                ]
            }
        )
        agent.review_files(scanned)
        assert "no-console-debug" in client.prompts[0]

    def test_budget_exhaustion_stops_the_loop(self, guard, reader, base_config, scanned):
        config = replace(base_config, budget=replace(base_config.budget, max_requests=1))
        budget_guard = BudgetGuard(config)
        # The scripted client honours the guard, so the second file cannot be
        # reviewed and the result must report that the review was cut short.
        client = ScriptedClient(default='{"findings": []}', guard=budget_guard)
        agent = ReviewAgent(client, reader, config, budget_guard, None)
        result = agent.review_files(scanned)
        assert result.files_reviewed == 1
        assert result.budget_stopped_reason is not None
        assert "request budget" in result.budget_stopped_reason
        assert result.file_reviews[0].file == "src/useUser.ts"
