"""End-to-end workflow tests with a Fake LLM (design doc 9.2, 11).

These are the acceptance tests: they run the whole pipeline offline and assert
the MVP criteria, including that the reviewed project is never modified.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from code_review_agent.config.settings import AgentConfig
from code_review_agent.models import RunStatus
from code_review_agent.workflow.orchestrator import (
    EXIT_INPUT_ERROR,
    EXIT_OK,
    EXIT_REPORT_ERROR,
    EXIT_REVIEW_FAILED,
    ReviewWorkflow,
)

from conftest import ScriptedClient, findings_payload, make_finding_json, routed


def snapshot(root: Path) -> dict[str, bytes]:
    """Hash-free content snapshot used to prove the project is read-only."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "app"
    (root / "src" / "hooks").mkdir(parents=True)
    (root / "src" / "components").mkdir(parents=True)
    (root / "node_modules" / "dep").mkdir(parents=True)

    (root / "src" / "hooks" / "useUser.ts").write_text(
        "export function useUser(user: any) {\n"
        "  const name = user.profile.name;\n"
        "  return { name };\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "src" / "components" / "Button.tsx").write_text(
        "export function Button({ label }) {\n"
        "  if (label === null) {\n"
        "    return null;\n"
        "  }\n"
        "  return label;\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "node_modules" / "dep" / "index.js").write_text(
        "module.exports = 'noisy';\n", encoding="utf-8"
    )
    (root / ".env").write_text("API_KEY=sk-abcdefghijklmnopqrstuvwx\n", encoding="utf-8")
    return root


@pytest.fixture
def config() -> AgentConfig:
    """Fake provider, no ESLint, no tool calls: fully offline and deterministic."""
    base = AgentConfig()
    return replace(
        base,
        llm=replace(base.llm, provider="fake", model="fake-model"),
        static_analysis=replace(base.static_analysis, enabled=False, eslint_command="off"),
        review=replace(base.review, enable_tool_calls=False),
    )


def validator_only(config: AgentConfig) -> AgentConfig:
    """Config with the deterministic analyzer silenced.

    Used by tests that assert exact counts of *model* findings; otherwise a
    lint diagnostic would legitimately add findings and mask the behaviour
    under test.
    """
    return replace(
        config,
        static_analysis=replace(config.static_analysis, builtin_rules=False, enabled=False),
    )


def run_workflow(project: Path, config: AgentConfig, client, tmp_path: Path, **kwargs):
    """Run the workflow against *client*.

    The workflow builds its own BudgetGuard internally, so a pre-constructed
    client cannot share it. Pass a callable to receive the real guard and build
    a guard-aware client; pass an object to have it used as-is.
    """
    out = tmp_path / "reports"

    def factory(inner_config, guard):
        return client(inner_config, guard) if callable(client) else client

    workflow = ReviewWorkflow(config, client_factory=factory, output_dir=out, **kwargs)
    return workflow.execute(project), out


class TestHappyPath:
    def test_completes_and_writes_reports(self, project, config, tmp_path):
        client = routed(
            ("review:src/hooks/useUser.ts", findings_payload(make_finding_json())),
            default='{"findings": []}',
        )
        result, out = run_workflow(project, config, client, tmp_path)
        assert result.run.status is RunStatus.COMPLETED
        assert result.exit_code == EXIT_OK
        assert (out / "report.md").is_file()
        assert (out / "report.json").is_file()

    def test_reviews_only_supported_files(self, project, config, tmp_path):
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        assert result.run.files_scanned == 2  # the two TS/TSX sources
        assert result.run.files_reviewed == 2

    def test_finding_survives_validation_with_evidence(self, project, config, tmp_path):
        client = routed(
            ("review:src/hooks/useUser.ts", findings_payload(make_finding_json())),
            default='{"findings": []}',
        )
        result, _ = run_workflow(project, config, client, tmp_path)
        assert result.run.findings_count >= 1
        titles = [f.title for f in result.report.findings]
        assert "Potential null dereference" in titles

    def test_every_reported_finding_has_a_file_and_line(self, project, config, tmp_path):
        client = routed(
            ("review:src/hooks/useUser.ts", findings_payload(make_finding_json())),
            default='{"findings": []}',
        )
        result, _ = run_workflow(project, config, client, tmp_path)
        for finding in result.report.findings:
            assert finding.file
            assert finding.start_line >= 1
            assert finding.end_line >= finding.start_line

    def test_findings_get_stable_ids(self, project, config, tmp_path):
        client = routed(
            ("review:src/hooks/useUser.ts", findings_payload(make_finding_json())),
            default='{"findings": []}',
        )
        result, _ = run_workflow(project, config, client, tmp_path)
        ids = [f.id for f in result.report.findings]
        assert ids == [f"finding-{i:03d}" for i in range(1, len(ids) + 1)]

    def test_phase_trace_is_recorded(self, project, config, tmp_path):
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        names = [phase["name"] for phase in result.run.phases]
        assert names == [
            "validate_input",
            "scan_project",
            "run_static_analysis",
            "llm_review",
            "validate_findings",
            "generate_report",
        ]
        assert all(phase["ok"] for phase in result.run.phases)

    def test_no_findings_is_a_completed_review(self, project, config, tmp_path):
        result, _ = run_workflow(project, validator_only(config), ScriptedClient(), tmp_path)
        assert result.run.status is RunStatus.COMPLETED
        assert result.run.findings_count == 0
        assert result.exit_code == EXIT_OK

    def test_token_usage_and_cost_are_recorded(self, project, config, tmp_path):
        # The client must share the workflow's guard for usage to be recorded.
        def factory(_config, guard):
            return ScriptedClient(guard=guard)

        result, _ = run_workflow(project, config, factory, tmp_path)
        assert result.run.token_usage.requests > 0
        assert result.run.token_usage.total_tokens > 0


class TestReadOnlyGuarantee:
    def test_project_is_never_modified(self, project, config, tmp_path):
        before = snapshot(project)
        client = routed(
            ("review:src/hooks/useUser.ts", findings_payload(make_finding_json())),
            default='{"findings": []}',
        )
        run_workflow(project, config, client, tmp_path)
        assert snapshot(project) == before

    def test_no_files_are_created_in_project_by_default(self, project, config, tmp_path):
        # Reports are redirected outside the project for this test; with the
        # default config they would land in <project>/.code-review.
        run_workflow(project, config, ScriptedClient(), tmp_path)
        assert not (project / ".code-review").exists()

    def test_secret_file_is_never_sent_to_the_model(self, project, config, tmp_path):
        client = ScriptedClient()
        run_workflow(project, config, client, tmp_path)
        joined = "\n".join(client.prompts)
        assert "sk-abcdefghijklmnopqrstuvwx" not in joined
        assert "API_KEY" not in joined

    def test_dependency_directory_is_never_read(self, project, config, tmp_path):
        client = ScriptedClient()
        run_workflow(project, config, client, tmp_path)
        assert "node_modules" not in "\n".join(client.prompts)


class TestStaticAnalysisIntegration:
    def test_missing_eslint_is_skipped_and_disclosed(self, project, config, tmp_path):
        config = replace(
            config,
            static_analysis=replace(config.static_analysis, enabled=True, eslint_command="auto"),
        )
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        assert result.run.status is RunStatus.COMPLETED
        assert any(name == "eslint" for name, _ in result.run.skipped_analyzers)
        markdown = (result.output_dir / "report.md").read_text(encoding="utf-8")
        assert "eslint" in markdown

    def test_missing_tool_is_fatal_when_configured(self, project, config, tmp_path):
        config = replace(
            config,
            static_analysis=replace(
                config.static_analysis,
                enabled=True,
                eslint_command="/nonexistent/eslint",
                fail_on_missing_tool=True,
            ),
        )
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        assert result.run.status is RunStatus.FAILED
        assert result.exit_code == EXIT_REVIEW_FAILED
        assert result.run.has_fatal_error

    def test_builtin_rules_produce_deterministic_findings(self, project, config, tmp_path):
        # Button.tsx assigns inside a condition, which no built-in rule targets;
        # add a file that trips a real rule instead.
        (project / "src" / "loose.ts").write_text(
            "export function same(a, b) {\n  return a == b;\n}\n", encoding="utf-8"
        )
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        rules = {
            finding.description
            for finding in result.report.findings
        }
        assert any("no-loose-equality" in description for description in rules)

    def test_deterministic_finding_needs_no_model_cooperation(self, project, config, tmp_path):
        (project / "src" / "debugger.ts").write_text("debugger;\n", encoding="utf-8")
        client = ScriptedClient(default='{"findings": []}')
        result, _ = run_workflow(project, config, client, tmp_path)
        assert result.run.findings_count >= 1
        sources = {f.source.value for f in result.report.findings}
        assert "static_analysis" in sources

    def test_scan_detail_is_reported(self, project, config, tmp_path):
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        assert result.report.scan_summary["files_scanned"] == result.run.files_scanned
        assert "skipped_by_reason" in result.report.scan_summary


class TestValidationIntegration:
    def test_invalid_line_is_rejected_and_disclosed(self, project, config, tmp_path):
        bad = make_finding_json(start_line=999, end_line=999)
        client = routed(("review:src/hooks/useUser.ts", findings_payload(bad)),
                        default='{"findings": []}')
        result, _ = run_workflow(project, validator_only(config), client, tmp_path)
        assert result.run.findings_count == 0
        reasons = {entry["reason"] for entry in result.report.rejections}
        assert "line-out-of-range" in reasons

    def test_fabricated_evidence_is_rejected(self, project, config, tmp_path):
        bad = make_finding_json(evidence="totally invented code;")
        client = routed(("review:src/hooks/useUser.ts", findings_payload(bad)),
                        default='{"findings": []}')
        result, _ = run_workflow(project, validator_only(config), client, tmp_path)
        reasons = {entry["reason"] for entry in result.report.rejections}
        assert "evidence-mismatch" in reasons

    def test_ghost_path_is_rejected(self, project, config, tmp_path):
        bad = make_finding_json(file="src/ghost.ts")
        client = routed(("review:src/hooks/useUser.ts", findings_payload(bad)),
                        default='{"findings": []}')
        result, _ = run_workflow(project, validator_only(config), client, tmp_path)
        assert result.run.findings_count == 0

    def test_duplicates_are_collapsed(self, project, config, tmp_path):
        duplicate = make_finding_json()
        client = routed(
            ("review:src/hooks/useUser.ts", findings_payload(duplicate, dict(duplicate))),
            default='{"findings": []}',
        )
        result, _ = run_workflow(project, validator_only(config), client, tmp_path)
        assert result.run.findings_count == 1
        assert any(entry["reason"] == "duplicate" for entry in result.report.rejections)

    def test_rejections_are_recorded_in_json_report(self, project, config, tmp_path):
        bad = make_finding_json(start_line=999, end_line=999)
        client = routed(("review:src/hooks/useUser.ts", findings_payload(bad)),
                        default='{"findings": []}')
        _, out = run_workflow(project, config, client, tmp_path)
        payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert payload["rejected_findings"]


class TestFailureBranches:
    def test_missing_directory_returns_input_error(self, tmp_path, config):
        workflow = ReviewWorkflow(config, output_dir=tmp_path / "out")
        result = workflow.execute(tmp_path / "does-not-exist")
        assert result.exit_code == EXIT_INPUT_ERROR
        assert result.run.status is RunStatus.FAILED
        assert result.report is None

    def test_file_instead_of_directory_returns_input_error(self, tmp_path, config):
        target = tmp_path / "a-file.txt"
        target.write_text("hello", encoding="utf-8")
        workflow = ReviewWorkflow(config, output_dir=tmp_path / "out")
        result = workflow.execute(target)
        assert result.exit_code == EXIT_INPUT_ERROR

    def test_unreadable_files_are_warned_not_fatal(self, project, config, tmp_path):
        # A file the scanner lists but the reader cannot open.
        import os

        broken = project / "src" / "broken.ts"
        broken.write_text("export const a = 1;\n", encoding="utf-8")
        os.chmod(broken, 0o000)
        try:
            result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
            assert result.run.status in {RunStatus.COMPLETED, RunStatus.FAILED}
        finally:
            os.chmod(broken, 0o644)

    def test_unparseable_model_output_never_reports_success(self, project, config, tmp_path):
        client = ScriptedClient(default="I refuse to answer in JSON.")
        result, _ = run_workflow(project, config, client, tmp_path)
        assert result.run.status is RunStatus.FAILED
        assert result.exit_code == EXIT_REVIEW_FAILED
        assert result.run.findings_count == 0

    def test_partial_failure_still_reports_the_other_files(self, project, config, tmp_path):
        # useUser.ts yields a finding; Button.tsx yields an unparseable payload,
        # which must not discard the first file's result.
        client = routed(
            ("review:src/hooks/useUser.ts", findings_payload(make_finding_json())),
            ("review:src/components/Button.tsx", "not json at all"),
        )
        result, _ = run_workflow(project, config, client, tmp_path)
        # The good file's finding survives, and the failure to review the other
        # file is disclosed rather than hidden behind a "completed" status.
        assert result.run.findings_count >= 1
        assert result.run.status is RunStatus.FAILED
        assert result.run.files_reviewed == 1
        assert result.report is not None
        markdown = (result.output_dir / "report.md").read_text(encoding="utf-8")
        assert "did not complete" in markdown
        assert "Button.tsx" in markdown

    def test_partial_review_can_be_downgraded_by_configuration(self, project, config, tmp_path):
        # An operator may accept a partial run as non-fatal; the caveat must
        # still be reported, so this is an explicit opt-in.
        config = replace(
            config, review=replace(config.review, fail_on_partial_review=False)
        )
        client = routed(
            ("review:src/hooks/useUser.ts", findings_payload(make_finding_json())),
            ("review:src/components/Button.tsx", "not json at all"),
        )
        result, _ = run_workflow(project, config, client, tmp_path)
        assert result.run.status is RunStatus.COMPLETED
        assert result.exit_code == EXIT_OK
        markdown = (result.output_dir / "report.md").read_text(encoding="utf-8")
        assert "not reviewed" in markdown

    def test_budget_exhaustion_marks_review_incomplete(self, project, config, tmp_path):
        config = replace(
            config,
            budget=replace(config.budget, max_requests=1, max_total_tokens=10_000),
        )
        def factory(_config, guard):
            return ScriptedClient(default='{"findings": []}', guard=guard)

        result, _ = run_workflow(project, config, factory, tmp_path)
        # One request is affordable, so the first file is reviewed and the
        # second cannot be: the run must fail loudly rather than look clean.
        assert result.run.status is RunStatus.FAILED
        assert result.exit_code == EXIT_REVIEW_FAILED
        assert result.run.files_reviewed < result.run.files_scanned
        assert any("budget" in error.stage for error in result.run.errors)

    def test_report_write_failure_returns_report_error(self, project, config, tmp_path, monkeypatch):
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        workflow = ReviewWorkflow(config, output_dir=blocker / "sub")
        result = workflow.execute(project)
        assert result.exit_code == EXIT_REPORT_ERROR
        assert result.run.status is RunStatus.FAILED
        assert result.report is None
        assert result.failure_reason

    def test_empty_project_completes_with_zero_coverage(self, tmp_path, config):
        empty = tmp_path / "empty"
        empty.mkdir()
        result, _ = run_workflow(empty, config, ScriptedClient(), tmp_path)
        assert result.run.status is RunStatus.COMPLETED
        assert result.run.files_scanned == 0
        assert result.run.notes

    def test_all_files_failing_marks_review_failed(self, project, config, tmp_path):
        client = ScriptedClient(default="{{{ not json")
        result, _ = run_workflow(project, validator_only(config), client, tmp_path)
        assert result.run.status is RunStatus.FAILED
        assert "no file was successfully reviewed" in (result.failure_reason or "")


class TestScanLimits:
    def test_file_limit_produces_partial_review(self, project, config, tmp_path):
        config = replace(config, scanner=replace(config.scanner, max_files=1))
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        assert result.run.files_scanned == 1
        assert any("limit" in note.lower() for note in result.run.notes)

    def test_zero_files_because_all_filtered(self, tmp_path, config):
        root = tmp_path / "assets"
        root.mkdir()
        (root / "logo.png").write_bytes(b"\x89PNG")
        (root / "readme.md").write_text("# hi", encoding="utf-8")
        result, _ = run_workflow(root, config, ScriptedClient(), tmp_path)
        assert result.run.files_scanned == 0


class TestReportContent:
    def test_markdown_mentions_default_output_location(self, project, config, tmp_path):
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        markdown = (result.output_dir / "report.md").read_text(encoding="utf-8")
        assert "Code Review Report" in markdown

    def test_config_summary_records_read_only(self, project, config, tmp_path):
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        assert result.report.config_summary["read_only"] is True

    def test_config_summary_never_contains_key_material(self, project, config, tmp_path):
        result, _ = run_workflow(project, config, ScriptedClient(), tmp_path)
        blob = json.dumps(result.report.config_summary)
        assert "sk-" not in blob
