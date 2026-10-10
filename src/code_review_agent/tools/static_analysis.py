"""Static analysis orchestration (design doc 5.2).

Runs the built-in rules always, and ESLint when it is available. A missing tool
is a first-class, recorded outcome rather than a crash, which is exactly the
branch described in section 5.3 ("static check tool not present").
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import AgentConfig
from ..errors import SecurityError, ToolNotAvailableError
from ..security import PathGuard
from ..tools.reader import CodeReader
from ..tools.rules import Diagnostic, analyse_text
from ..tools.scanner import ScannedFile

logger = logging.getLogger(__name__)

#: ESLint reports severity 1 for warnings and 2 for errors.
_ESLINT_SEVERITY = {1: "warning", 2: "error", 0: "info"}


@dataclass
class StaticAnalysisResult:
    diagnostics: dict[str, list[Diagnostic]] = field(default_factory=dict)
    analyzers_run: list[str] = field(default_factory=list)
    skipped_analyzers: list[tuple[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def for_file(self, relative_path: str) -> list[Diagnostic]:
        return self.diagnostics.get(relative_path, [])

    @property
    def total_diagnostics(self) -> int:
        return sum(len(items) for items in self.diagnostics.values())

    @property
    def error_count(self) -> int:
        return sum(1 for items in self.diagnostics.values() for d in items if d.is_error)


class StaticAnalyzer:
    """Coordinates external and built-in checkers over the scanned files."""

    def __init__(self, guard: PathGuard, config: AgentConfig, reader: CodeReader) -> None:
        self.guard = guard
        self.config = config
        self.reader = reader

    # -- public API ------------------------------------------------------
    def run(self, files: list[ScannedFile]) -> StaticAnalysisResult:
        started = time.monotonic()
        result = StaticAnalysisResult()

        if self.config.static_analysis.builtin_rules:
            self._run_builtin(files, result)

        if self.config.static_analysis.enabled:
            try:
                self._run_eslint(files, result)
            except ToolNotAvailableError as exc:
                if self.config.static_analysis.fail_on_missing_tool:
                    raise
                logger.warning("ESLint unavailable: %s", exc)
                result.skipped_analyzers.append(("eslint", str(exc)))
            except Exception as exc:  # noqa: BLE001 - a checker must never abort a review
                logger.warning("ESLint failed: %s", exc)
                result.skipped_analyzers.append(("eslint", f"failed: {exc}"))

        result.duration_seconds = time.monotonic() - started
        return result

    # -- built-in rules --------------------------------------------------
    def _run_builtin(self, files: list[ScannedFile], result: StaticAnalysisResult) -> None:
        count = 0
        for scanned in files:
            source = self.reader.read_safe(scanned.relative_path)
            if source is None:
                continue
            for diagnostic in analyse_text(scanned.relative_path, source.text):
                result.diagnostics.setdefault(scanned.relative_path, []).append(diagnostic)
                count += 1
                if count >= _diagnostic_cap(self.config, len(files)):
                    logger.info("built-in diagnostic cap reached; truncating")
                    result.analyzers_run.append("builtin")
                    return
        result.analyzers_run.append("builtin")

    # -- ESLint ----------------------------------------------------------
    def resolve_eslint(self) -> list[str]:
        """Return the argv prefix to run ESLint, or raise if unavailable."""
        command = self.config.static_analysis.eslint_command
        if command and command != "auto":
            path = Path(command).expanduser()
            if path.is_file():
                return [str(path)]
            found = shutil.which(command)
            if found:
                return [found]
            raise ToolNotAvailableError(f"configured eslint command not found: {command}")

        found = shutil.which("eslint")
        if found:
            return [found]
        # Fall back to a project-local install.
        local = self.guard.root / "node_modules" / ".bin" / "eslint"
        if local.is_file():
            return [str(local)]
        raise ToolNotAvailableError(
            "eslint not found on PATH or in ./node_modules/.bin; "
            "set static_analysis.eslint_command to pin an executable"
        )

    def _run_eslint(self, files: list[ScannedFile], result: StaticAnalysisResult) -> None:
        if not files:
            return
        argv = self.resolve_eslint()
        relative_paths = [scanned.relative_path for scanned in files]

        with tempfile.TemporaryDirectory(prefix="code-review-eslint-") as tmp:
            report_path = Path(tmp) / "eslint.json"
            command = [*argv, "--format", "json", "--output-file", str(report_path), "--no-color"]
            if _has_no_config(self.guard.root):
                # ESLint 9 otherwise refuses to run and emits no report.
                command.append("--no-config-lookup")
            command.extend(relative_paths)
            logger.info("running static analysis: %s", " ".join(command[:4]) + " ...")
            try:
                completed = subprocess.run(  # noqa: S603 - argv list, never a shell string
                    command,
                    cwd=str(self.guard.root),
                    capture_output=True,
                    text=True,
                    timeout=self.config.static_analysis.eslint_timeout_seconds,
                    shell=False,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ToolNotAvailableError(
                    f"eslint exceeded {self.config.static_analysis.eslint_timeout_seconds}s timeout"
                ) from exc
            except OSError as exc:
                raise ToolNotAvailableError(f"could not execute eslint: {exc}") from exc

            if not report_path.is_file():
                detail = (completed.stderr or completed.stdout or "").strip()[:400]
                raise ToolNotAvailableError(
                    f"eslint produced no report (exit {completed.returncode}): {detail}"
                )
            payload = _load_eslint_report(report_path)

        per_file_cap = self.config.static_analysis.max_diagnostics_per_file
        for entry in payload:
            raw_path = str(entry.get("filePath") or "")
            if not raw_path:
                continue
            try:
                relative = self.guard.relative(Path(raw_path))
            except SecurityError:
                logger.warning("eslint reported a file outside the project: %s", raw_path)
                continue
            messages = entry.get("messages") or []
            if not isinstance(messages, list):
                continue
            bucket = result.diagnostics.setdefault(relative, [])
            for message in messages[:per_file_cap]:
                if not isinstance(message, dict):
                    continue
                if message.get("fatal"):
                    continue  # parse errors are not review findings
                bucket.append(
                    Diagnostic(
                        file=relative,
                        line=int(message.get("line") or 1),
                        column=int(message.get("column") or 1),
                        severity=_ESLINT_SEVERITY.get(int(message.get("severity") or 1), "warning"),
                        rule_id=str(message.get("ruleId") or "eslint-parse-error"),
                        message=str(message.get("message") or "").strip(),
                        source="eslint",
                    )
                )
            if not bucket:
                result.diagnostics.pop(relative, None)

        result.analyzers_run.append("eslint")


def _has_no_config(root: Path) -> bool:
    """ESLint 9 needs an explicit config; detect one, else suppress the warning."""
    candidates = (
        "eslint.config.js",
        "eslint.config.mjs",
        "eslint.config.cjs",
        "eslint.config.ts",
        ".eslintrc",
        ".eslintrc.js",
        ".eslintrc.cjs",
        ".eslintrc.json",
        ".eslintrc.yml",
        ".eslintrc.yaml",
    )
    return not any((root / name).exists() for name in candidates)


def _load_eslint_report(path: Path) -> list[dict]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        raise ToolNotAvailableError(f"cannot read eslint report: {exc}") from exc
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolNotAvailableError(f"eslint report was not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ToolNotAvailableError("eslint report was not a JSON array")
    return [entry for entry in payload if isinstance(entry, dict) and "filePath" in entry]


def _diagnostic_cap(config: AgentConfig, file_count: int) -> int:
    """Bound total built-in diagnostics so one messy file cannot dominate."""
    return max(50, min(file_count, config.budget.max_files) * 15)
