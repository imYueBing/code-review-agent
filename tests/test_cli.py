"""CLI tests: argument handling, exit codes, output formats (doc 3.2, 11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_review_agent.cli import (
    EXIT_INPUT_ERROR,
    apply_overrides,
    build_parser,
    main,
)
from code_review_agent.config.settings import AgentConfig, load_config


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "app"
    (root / "src").mkdir(parents=True)
    (root / "src" / "useUser.ts").write_text(
        "export function useUser(user: any) {\n"
        "  const name = user.profile.name;\n"
        "  return { name };\n"
        "}\n",
        encoding="utf-8",
    )
    return root


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "code-review.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestParser:
    def test_requires_project_argument(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_accepts_project_path(self):
        assert build_parser().parse_args(["./app"]).project == "./app"

    def test_defaults(self):
        args = build_parser().parse_args(["./app"])
        assert args.format == "text"
        assert args.provider is None
        assert args.verbose is False

    def test_rejects_unknown_provider(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["./app", "--provider", "wishful"])

    def test_rejects_unknown_format(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["./app", "--format", "yaml"])

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--version"])
        assert excinfo.value.code == 0
        assert "0.1.0" in capsys.readouterr().out


class TestOverrides:
    def args(self, *argv):
        return build_parser().parse_args(["./app", *argv])

    def test_provider_and_model(self):
        config = apply_overrides(AgentConfig(), self.args("--provider", "deepseek", "--model", "x"))
        assert config.llm.provider == "deepseek"
        assert config.llm.model == "x"

    def test_budget_flags(self):
        config = apply_overrides(
            AgentConfig(), self.args("--max-files", "3", "--max-requests", "4", "--max-tokens", "5")
        )
        assert config.budget.max_files == 3
        assert config.budget.max_requests == 4
        assert config.budget.max_total_tokens == 5

    def test_eslint_off_disables_static_analysis(self):
        config = apply_overrides(AgentConfig(), self.args("--eslint-command", "off"))
        assert config.static_analysis.enabled is False

    def test_skip_static_analysis_flag(self):
        config = apply_overrides(AgentConfig(), self.args("--skip-static-analysis"))
        assert config.static_analysis.enabled is False

    def test_eslint_command_pin(self):
        config = apply_overrides(AgentConfig(), self.args("--eslint-command", "/usr/bin/eslint"))
        assert config.static_analysis.eslint_command == "/usr/bin/eslint"
        assert config.static_analysis.enabled is True

    def test_fail_on_missing_tool(self):
        config = apply_overrides(AgentConfig(), self.args("--fail-on-missing-tool"))
        assert config.static_analysis.fail_on_missing_tool is True

    def test_no_tool_calls(self):
        config = apply_overrides(AgentConfig(), self.args("--no-tool-calls"))
        assert config.review.enable_tool_calls is False

    def test_output_dir(self):
        config = apply_overrides(AgentConfig(), self.args("--output-dir", "/tmp/out"))
        assert config.report.output_dir == "/tmp/out"

    def test_verbose_sets_debug_logging(self):
        assert apply_overrides(AgentConfig(), self.args("--verbose")).log_level == "DEBUG"

    def test_quiet_sets_error_logging(self):
        assert apply_overrides(AgentConfig(), self.args("--quiet")).log_level == "ERROR"

    def test_cli_overrides_config_file(self, tmp_path):
        path = write_config(tmp_path, '[llm]\nmodel = "from-file"\n')
        config = load_config(path, env={})
        config = apply_overrides(config, self.args("--model", "from-cli"))
        assert config.llm.model == "from-cli"


class TestEndToEnd:
    def test_fake_provider_run_succeeds(self, project, tmp_path, capsys):
        out = tmp_path / "reports"
        code = main(
            [
                str(project),
                "--provider",
                "fake",
                "--skip-static-analysis",
                "--no-tool-calls",
                "--output-dir",
                str(out),
            ]
        )
        assert code == 0
        assert (out / "report.md").is_file()
        assert (out / "report.json").is_file()
        assert "Review completed." in capsys.readouterr().out

    def test_json_format_is_machine_readable(self, project, tmp_path, capsys):
        code = main(
            [
                str(project),
                "--provider",
                "fake",
                "--skip-static-analysis",
                "--no-tool-calls",
                "--output-dir",
                str(tmp_path / "r"),
                "--format",
                "json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["status"] == "completed"
        assert payload["files_reviewed"] == 1
        assert payload["exit_code"] == 0

    def test_missing_project_directory_fails(self, tmp_path, capsys):
        code = main([str(tmp_path / "nope"), "--provider", "fake"])
        assert code == EXIT_INPUT_ERROR
        assert "does not exist" in capsys.readouterr().err

    def test_file_instead_of_directory_fails(self, tmp_path, capsys):
        target = tmp_path / "f.txt"
        target.write_text("x", encoding="utf-8")
        code = main([str(target), "--provider", "fake"])
        assert code == EXIT_INPUT_ERROR
        assert "not a directory" in capsys.readouterr().err

    def test_invalid_config_fails_cleanly(self, project, tmp_path, capsys):
        path = write_config(tmp_path, '[llm]\nprovider = "wishful"\n')
        code = main([str(project), "--config", str(path)])
        assert code == EXIT_INPUT_ERROR
        assert "unsupported provider" in capsys.readouterr().err

    def test_missing_config_file_fails(self, project, tmp_path, capsys):
        code = main([str(project), "--config", str(tmp_path / "nope.toml")])
        assert code == EXIT_INPUT_ERROR
        assert "config file not found" in capsys.readouterr().err

    def test_summary_reports_skipped_analyzer(self, project, tmp_path, capsys):
        code = main(
            [
                str(project),
                "--provider",
                "fake",
                "--eslint-command",
                "auto",
                "--no-tool-calls",
                "--output-dir",
                str(tmp_path / "r"),
            ]
        )
        assert code == 0
        assert "Static analysis skipped" in capsys.readouterr().out

    def test_summary_attributes_findings_to_static_analysis(self, project, tmp_path, capsys):
        # A lint violation in a project the model finds nothing in: the summary
        # must not let "N findings" look like model output.
        (project / "src" / "loose.ts").write_text(
            "export function same(a, b) {\n  return a == b;\n}\n", encoding="utf-8"
        )
        code = main(
            [
                str(project),
                "--provider",
                "fake",
                "--skip-static-analysis",
                "--no-tool-calls",
                "--output-dir",
                str(tmp_path / "r"),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "from built-in static analysis" in out

    def test_zero_reviewed_files_with_findings_is_explained(self, tmp_path, capsys):
        # No credentials and a real provider: no file gets an AI review, but the
        # built-in analyzer still finds real problems. The summary must explain
        # that combination rather than looking self-contradictory.
        root = tmp_path / "app"
        (root / "src").mkdir(parents=True)
        (root / "src" / "debugger.ts").write_text("debugger;\n", encoding="utf-8")
        code = main(
            [
                str(root),
                "--provider",
                "openai",
                "--api-key-env",
                "DEFINITELY_UNSET_CODE_REVIEW_KEY",
                "--output-dir",
                str(tmp_path / "r"),
                "--quiet",
            ]
        )
        out = capsys.readouterr().out
        assert code == 3
        assert "Files reviewed: 0" in out
        assert "no file received an AI review" in out

    def test_cli_never_modifies_the_project(self, project, tmp_path):
        before = {p: p.read_bytes() for p in sorted(project.rglob("*")) if p.is_file()}
        main(
            [
                str(project),
                "--provider",
                "fake",
                "--skip-static-analysis",
                "--no-tool-calls",
                "--output-dir",
                str(tmp_path / "r"),
            ]
        )
        after = {p: p.read_bytes() for p in sorted(project.rglob("*")) if p.is_file()}
        assert before == after

    def test_default_output_dir_is_dot_code_review(self, project):
        # Reports default into the project's own .code-review directory, which
        # the scanner excludes so it can never review its own output.
        code = main(
            [
                str(project),
                "--provider",
                "fake",
                "--skip-static-analysis",
                "--no-tool-calls",
            ]
        )
        assert code == 0
        assert (project / ".code-review" / "report.md").is_file()

    def test_reports_do_not_become_future_review_targets(self, project, tmp_path, capsys):
        main([str(project), "--provider", "fake", "--skip-static-analysis", "--no-tool-calls"])
        capsys.readouterr()
        code = main(
            [str(project), "--provider", "fake", "--skip-static-analysis", "--no-tool-calls",
             "--format", "json"]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        # Still exactly one source file: the generated .code-review is ignored.
        assert payload["files_scanned"] == 1


class TestModuleEntryPoint:
    def test_python_m_invocation(self, project, tmp_path):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_review_agent",
                str(project),
                "--provider",
                "fake",
                "--skip-static-analysis",
                "--no-tool-calls",
                "--output-dir",
                str(tmp_path / "r"),
            ],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PATH": ""},
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Review completed." in result.stdout
