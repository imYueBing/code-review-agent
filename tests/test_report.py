"""Report generation tests (design doc 2.1, 6.3, 11)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from code_review_agent.errors import ReportError
from code_review_agent.models import (
    FindingSource,
    ReviewFinding,
    ReviewRun,
    RunStatus,
)
from code_review_agent.reporting.generator import ReportGenerator, ReviewReport


def make_run(**overrides) -> ReviewRun:
    run = ReviewRun(run_id="run-test", project_root="/tmp/project")
    run.files_scanned = 10
    run.files_reviewed = 10
    run.status = RunStatus.COMPLETED
    run.finished_at = run.started_at + timedelta(seconds=1.5)
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


def make_finding(**overrides) -> ReviewFinding:
    payload = {
        "title": "Potential null dereference",
        "severity": "high",
        "category": "bug",
        "file": "src/useUser.ts",
        "start_line": 2,
        "end_line": 2,
        "evidence": "const name = user.profile.name;",
        "description": "user.profile may be undefined.",
        "recommendation": "Guard before access.",
        "confidence": 0.9,
        "source": FindingSource.VALIDATED_LLM,
    }
    payload.update(overrides)
    return ReviewFinding(**payload)


def make_report(findings=None, run=None, **kwargs) -> ReviewReport:
    return ReviewReport(run=run or make_run(), findings=findings or [], **kwargs)


class TestMarkdownRendering:
    def test_includes_run_metadata(self, base_config):
        markdown = ReportGenerator(base_config).render_markdown(make_report())
        assert "run-test" in markdown
        assert "/tmp/project" in markdown
        assert "completed" in markdown

    def test_summary_table_counts_severities(self, base_config):
        findings = [
            make_finding(severity="high"),
            make_finding(severity="medium", title="Other issue", start_line=3,
                         evidence="return { name };"),
        ]
        markdown = ReportGenerator(base_config).render_markdown(make_report(findings))
        assert "| High | 1 |" in markdown
        assert "| Medium | 1 |" in markdown
        assert "| Critical | 0 |" in markdown

    def test_lists_each_finding_with_location(self, base_config):
        markdown = ReportGenerator(base_config).render_markdown(make_report([make_finding()]))
        assert "src/useUser.ts:2" in markdown
        assert "Potential null dereference" in markdown

    def test_renders_line_range_when_multiline(self, base_config):
        finding = make_finding(start_line=1, end_line=3)
        markdown = ReportGenerator(base_config).render_markdown(make_report([finding]))
        assert "src/useUser.ts:1-3" in markdown

    def test_includes_evidence_in_code_block(self, base_config):
        markdown = ReportGenerator(base_config).render_markdown(make_report([make_finding()]))
        assert "const name = user.profile.name;" in markdown
        assert "```" in markdown

    def test_no_findings_message_is_not_a_pass_claim(self, base_config):
        markdown = ReportGenerator(base_config).render_markdown(make_report())
        assert "No findings were reported" in markdown
        assert "does not guarantee" in markdown

    def test_incomplete_review_is_flagged_prominently(self, base_config):
        run = make_run(status=RunStatus.FAILED)
        markdown = ReportGenerator(base_config).render_markdown(make_report(run=run))
        assert "did not complete" in markdown
        assert "not a passing review" in markdown

    def test_partial_coverage_is_a_caveat(self, base_config):
        run = make_run(files_reviewed=4)
        report = make_report(run=run)
        markdown = ReportGenerator(base_config).render_markdown(report)
        assert "## Caveats" in markdown
        assert "6 scanned file(s) were not reviewed" in markdown

    def test_skipped_analyzer_is_a_caveat(self, base_config):
        run = make_run(skipped_analyzers=[("eslint", "not installed")])
        markdown = ReportGenerator(base_config).render_markdown(make_report(run=run))
        assert "eslint" in markdown
        assert "not fully applied" in markdown

    def test_skipped_files_table(self, base_config):
        report = make_report(skipped_files=[{"file": "src/bad.ts", "reason": "unreadable"}])
        markdown = ReportGenerator(base_config).render_markdown(report)
        assert "src/bad.ts" in markdown
        assert "unreadable" in markdown

    def test_rejected_findings_are_disclosed(self, base_config):
        report = make_report(
            rejections=[
                {"file": "src/a.ts", "line": "9", "reason": "evidence-mismatch", "detail": "absent"}
            ]
        )
        markdown = ReportGenerator(base_config).render_markdown(report)
        assert "Discarded Candidate Findings" in markdown
        assert "evidence-mismatch" in markdown

    def test_errors_table(self, base_config):
        run = make_run()
        run.add_error("llm_review", "provider exploded", fatal=True)
        markdown = ReportGenerator(base_config).render_markdown(make_report(run=run))
        assert "provider exploded" in markdown

    def test_secrets_in_errors_are_redacted(self, base_config):
        run = make_run()
        run.add_error("llm", "auth failed with sk-abcdefghijklmnopqrstuvwx")
        markdown = ReportGenerator(base_config).render_markdown(make_report(run=run))
        assert "sk-abcdefghijklmnopqrstuvwx" not in markdown
        assert "[REDACTED]" in markdown

    def test_unknown_cost_is_stated_not_faked(self, base_config):
        markdown = ReportGenerator(base_config).render_markdown(make_report())
        assert "unknown" in markdown

    def test_known_cost_is_rendered(self, base_config):
        run = make_run(estimated_cost=0.0123)
        markdown = ReportGenerator(base_config).render_markdown(make_report(run=run))
        assert "$0.0123" in markdown

    def test_pipe_in_evidence_does_not_break_tables(self, base_config):
        finding = make_finding(title="a | b")
        markdown = ReportGenerator(base_config).render_markdown(make_report([finding]))
        assert markdown.count("# Code Review Report") == 1

    def test_long_evidence_is_clipped(self, base_config):
        from dataclasses import replace as dc_replace

        config = dc_replace(
            base_config, report=dc_replace(base_config.report, max_evidence_chars=20)
        )
        finding = make_finding(evidence="x" * 500)
        markdown = ReportGenerator(config).render_markdown(make_report([finding]))
        assert "evidence truncated" in markdown

    def test_states_read_only_invariant(self, base_config):
        markdown = ReportGenerator(base_config).render_markdown(make_report())
        assert "never modifies the reviewed project" in markdown


class TestJsonRendering:
    def test_payload_is_json_serialisable(self, base_config):
        payload = make_report([make_finding()]).to_dict()
        json.dumps(payload)

    def test_includes_schema_version(self, base_config):
        assert make_report().to_dict()["schema_version"] == "1.0"

    def test_findings_are_serialised(self, base_config):
        payload = make_report([make_finding()]).to_dict()
        assert payload["findings"][0]["severity"] == "high"
        assert payload["findings"][0]["file"] == "src/useUser.ts"

    def test_review_complete_flag_is_explicit(self, base_config):
        payload = make_report().to_dict()
        assert payload["summary"]["review_complete"] is True
        failed = make_report(run=make_run(status=RunStatus.FAILED)).to_dict()
        assert failed["summary"]["review_complete"] is False

    def test_caveats_are_machine_readable(self, base_config):
        run = make_run(files_reviewed=1)
        payload = make_report(run=run).to_dict()
        assert payload["summary"]["caveats"]


class TestWriting:
    def test_writes_both_files(self, base_config, tmp_path):
        markdown_path, json_path = ReportGenerator(base_config).write(make_report(), tmp_path / "out")
        assert markdown_path.is_file()
        assert json_path.is_file()
        assert markdown_path.name == "report.md"
        assert json_path.name == "report.json"

    def test_json_file_is_valid_json(self, base_config, tmp_path):
        _, json_path = ReportGenerator(base_config).write(make_report(), tmp_path / "out")
        json.loads(json_path.read_text(encoding="utf-8"))

    def test_creates_nested_output_directory(self, base_config, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        ReportGenerator(base_config).write(make_report(), target)
        assert (target / "report.md").is_file()

    def test_unwritable_directory_raises_report_error(self, base_config, tmp_path):
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        with pytest.raises(ReportError):
            ReportGenerator(base_config).write(make_report(), blocker / "sub")

    def test_write_failure_is_never_silent(self, base_config, tmp_path, monkeypatch):
        generator = ReportGenerator(base_config)
        output = tmp_path / "out"

        def explode(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", explode)
        with pytest.raises(ReportError, match="cannot write report"):
            generator.write(make_report(), output)
