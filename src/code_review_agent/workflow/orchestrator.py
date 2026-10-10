"""The explicit review workflow (design doc 4.1, 5).

Every phase records its own timing and outcome on the :class:`ReviewRun`, so a
partial run is visible in the report rather than silently presented as a pass.

Status transitions:

    pending -> running -> completed
                       -> failed       (a required phase failed)
                       -> cancelled    (budget exhausted before any review)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agent.reviewer import AgentResult, ReviewAgent, diagnostic_to_finding
from ..agent.tools import ProjectToolbelt
from ..agent.validator import FindingValidator, ValidationOutcome
from ..config.settings import AgentConfig, estimate_cost
from ..errors import (
    BudgetExceededError,
    CodeReviewError,
    InputError,
    LLMError,
    ReportError,
    SecurityError,
    ToolNotAvailableError,
)
from ..llm.client import BudgetGuard, LLMClient, build_client
from ..models import ReviewFinding, ReviewRun, RunStatus, new_run_id, utcnow
from ..reporting.generator import ReportGenerator, ReviewReport
from ..security import PathGuard
from ..tools.reader import CodeReader
from ..tools.scanner import ProjectScanner, ScanReport
from ..tools.static_analysis import StaticAnalysisResult, StaticAnalyzer

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_REVIEW_FAILED = 3
EXIT_REPORT_ERROR = 4
EXIT_INTERRUPTED = 130


@dataclass
class WorkflowResult:
    run: ReviewRun
    report: ReviewReport | None = None
    markdown_path: Path | None = None
    json_path: Path | None = None
    exit_code: int = EXIT_OK
    output_dir: Path | None = None
    failure_reason: str | None = None


@dataclass
class _PipelineOutcome:
    """Everything the pipeline produced; ``guard`` is None on invalid input."""

    guard: PathGuard | None = None
    scan_report: ScanReport = field(default_factory=ScanReport)
    static_result: StaticAnalysisResult = field(default_factory=StaticAnalysisResult)
    agent_result: AgentResult | None = None
    validation: ValidationOutcome = field(default_factory=ValidationOutcome)
    findings: list[ReviewFinding] = field(default_factory=list)
    fatal: bool = False


class _Phase:
    """Times a workflow phase and appends the result to the run trace."""

    def __init__(self, run: ReviewRun, name: str) -> None:
        self.run = run
        self.name = name
        self.started = 0.0

    def __enter__(self) -> "_Phase":
        self.started = time.monotonic()
        logger.info("phase start: %s", self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = time.monotonic() - self.started
        self.run.phases.append(
            {
                "name": self.name,
                "duration_seconds": round(duration, 4),
                "ok": exc_type is None,
                "error": str(exc) if exc is not None else None,
            }
        )
        logger.info("phase end: %s (%.3fs)", self.name, duration)
        return False


class ReviewWorkflow:
    """Runs scan -> static analysis -> LLM review -> validate -> report."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        client_factory: Callable[[AgentConfig, BudgetGuard], LLMClient] | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._output_dir = output_dir

    # -- public API ------------------------------------------------------
    def execute(self, project_root: Path) -> WorkflowResult:
        run = ReviewRun(run_id=new_run_id(), project_root=str(project_root))
        output_dir = self._resolve_output_dir(project_root)

        outcome = self._run_pipeline(run, project_root)
        if outcome.guard is None:
            run.status = RunStatus.FAILED
            run.finished_at = utcnow()
            message = run.errors[-1].message if run.errors else "invalid input"
            return WorkflowResult(
                run=run,
                exit_code=EXIT_INPUT_ERROR,
                output_dir=output_dir,
                failure_reason=message,
            )

        exit_code, failure_reason = self._finalise_status(run, outcome)

        report = self._build_report(
            run=run,
            findings=outcome.findings,
            scan_report=outcome.scan_report,
            static_result=outcome.static_result,
            agent_result=outcome.agent_result,
            validation=outcome.validation,
            output_dir=output_dir,
        )
        markdown_path: Path | None = None
        json_path: Path | None = None
        with _Phase(run, "generate_report"):
            try:
                markdown_path, json_path = ReportGenerator(self.config).write(report, output_dir)
            except ReportError as exc:
                run.add_error("generate_report", str(exc), fatal=True)
                run.status = RunStatus.FAILED
                exit_code = EXIT_REPORT_ERROR
                failure_reason = str(exc)
                report = None

        run.finished_at = utcnow()
        return WorkflowResult(
            run=run,
            report=report,
            markdown_path=markdown_path,
            json_path=json_path,
            exit_code=exit_code,
            output_dir=output_dir,
            failure_reason=failure_reason,
        )

    # -- pipeline --------------------------------------------------------
    def _run_pipeline(self, run: ReviewRun, project_root: Path) -> _PipelineOutcome:
        outcome = _PipelineOutcome()

        # ---- Phase 1: validate input --------------------------------
        with _Phase(run, "validate_input"):
            try:
                guard = PathGuard(project_root)
            except SecurityError as exc:
                run.add_error("validate_input", str(exc), fatal=True)
                logger.error("invalid project root: %s", exc)
                return outcome
            outcome.guard = guard

            reader = CodeReader(guard, max_bytes=self.config.budget.max_file_bytes)
            scanner = ProjectScanner(guard, self.config)
            try:
                scanner.validate_root()
            except InputError as exc:
                run.add_error("validate_input", str(exc), fatal=True)
                logger.error("input validation failed: %s", exc)
                return outcome

        # ---- Phase 2: scan + filter --------------------------------
        with _Phase(run, "scan_project"):
            try:
                outcome.scan_report = scanner.scan()
            except (InputError, SecurityError) as exc:
                run.add_error("scan_project", str(exc), fatal=True)
                return outcome
        run.files_scanned = len(outcome.scan_report.files)
        if outcome.scan_report.truncated:
            run.notes.append(
                f"File limit reached ({run.files_scanned} of up to "
                f"{self.config.scanner.max_files}); the review covers a subset of the project."
            )
        if run.files_scanned == 0:
            run.notes.append("No reviewable files were found in the project.")

        # ---- Phase 3: static analysis ------------------------------
        analyzer = StaticAnalyzer(guard, self.config, reader)
        with _Phase(run, "run_static_analysis"):
            try:
                outcome.static_result = analyzer.run(outcome.scan_report.files)
            except ToolNotAvailableError as exc:
                # fail_on_missing_tool == True is the only way to get here.
                run.add_error("run_static_analysis", str(exc), fatal=True)
                return outcome
            except CodeReviewError as exc:
                # A checker crash degrades the review; it does not end it.
                run.add_error("run_static_analysis", str(exc), fatal=False)
                logger.warning("static analysis degraded: %s", exc)
        run.skipped_analyzers = list(outcome.static_result.skipped_analyzers)

        # ---- Phase 4: LLM review -----------------------------------
        run.status = RunStatus.RUNNING
        budget_guard = BudgetGuard(self.config)
        client = (
            self._client_factory(self.config, budget_guard)
            if self._client_factory is not None
            else build_client(self.config, budget_guard)
        )

        toolbelt = ProjectToolbelt(guard, reader, outcome.scan_report.files, self.config)
        agent = ReviewAgent(client, reader, self.config, budget_guard, toolbelt)
        agent.set_diagnostics(outcome.static_result.diagnostics)

        agent_result = AgentResult()
        with _Phase(run, "llm_review"):
            try:
                agent_result = agent.review_files(outcome.scan_report.files)
            except BudgetExceededError as exc:
                run.add_error("llm_review", str(exc), fatal=False)
                agent_result = AgentResult(
                    errors=[("budget", str(exc))], budget_stopped_reason=str(exc)
                )
            except LLMError as exc:
                run.add_error("llm_review", str(exc), fatal=True)
                logger.error("review aborted: %s", exc)
                agent_result = AgentResult(errors=[("llm_review", str(exc))])
            except CodeReviewError as exc:
                run.add_error("llm_review", str(exc), fatal=True)
                agent_result = AgentResult(errors=[("llm_review", str(exc))])
        outcome.agent_result = agent_result
        outcome.fatal = run.has_fatal_error

        for stage, message in agent_result.errors:
            if not any(error.stage == stage and error.message == message for error in run.errors):
                run.add_error(stage, message, fatal=False)
        run.files_reviewed = agent_result.files_reviewed

        # ---- Phase 5: validate findings ----------------------------
        # Candidate findings are deliberately *not* de-duplicated here. The
        # validator owns duplicate detection, and collapsing repeats before it
        # runs would hide them from the report's disclosure section.
        candidates: list[ReviewFinding] = list(agent_result.findings)
        candidates.extend(self._deterministic_findings(outcome.static_result, agent_result))

        validator = FindingValidator(guard, reader, self.config)
        with _Phase(run, "validate_findings"):
            outcome.validation = validator.validate(candidates, outcome.scan_report.files)
            ordered = sorted(
                outcome.validation.valid,
                key=lambda f: (f.severity.rank, f.file, f.start_line, f.title),
            )
            for index, finding in enumerate(ordered, start=1):
                finding.id = f"finding-{index:03d}"
            outcome.findings = ordered

        # ---- Bookkeeping -------------------------------------------
        run.output_skipped_files = [
            {"file": review.file, "reason": review.skipped_reason or "skipped"}
            for review in agent_result.file_reviews
            if review.skipped_reason
        ]
        run.token_usage = budget_guard.usage
        run.estimated_cost = estimate_cost(
            self.config, run.token_usage.prompt_tokens, run.token_usage.completion_tokens
        )
        run.findings_count = len(outcome.findings)
        if outcome.validation.rejected or outcome.validation.duplicates:
            run.notes.append(
                f"Discarded {len(outcome.validation.rejected)} candidate(s) with invalid "
                f"evidence or paths, and {len(outcome.validation.duplicates)} duplicate(s)."
            )
        return outcome

    # -- status ----------------------------------------------------------
    def _finalise_status(self, run: ReviewRun, outcome: _PipelineOutcome) -> tuple[int, str | None]:
        agent_result = outcome.agent_result

        if run.has_fatal_error:
            run.status = RunStatus.FAILED
            fatal = next(error for error in run.errors if error.fatal)
            return EXIT_REVIEW_FAILED, f"{fatal.stage}: {fatal.message}"

        if run.files_scanned == 0:
            # Legitimately nothing to review: a completed run with zero coverage.
            run.status = RunStatus.COMPLETED
            return EXIT_OK, None

        if agent_result is None or run.files_reviewed == 0:
            if agent_result is not None and agent_result.budget_stopped_reason:
                run.status = RunStatus.CANCELLED
                run.notes.append(
                    "Budget was exhausted before any file could be reviewed; no review coverage."
                )
                return EXIT_REVIEW_FAILED, agent_result.budget_stopped_reason
            run.status = RunStatus.FAILED
            run.notes.append(
                "No file could be reviewed, so this run provides no review coverage."
            )
            return EXIT_REVIEW_FAILED, "no file was successfully reviewed"

        if agent_result.budget_stopped_reason:
            # Some files were reviewed, but the run is explicitly incomplete.
            run.status = RunStatus.FAILED
            run.notes.append(
                f"Budget stopped the review early: {agent_result.budget_stopped_reason}"
            )
            return EXIT_REVIEW_FAILED, agent_result.budget_stopped_reason

        if run.files_reviewed < run.files_scanned:
            failed_files = [
                review.file for review in agent_result.file_reviews if review.error
            ]
            if failed_files and self.config.review.fail_on_partial_review:
                # Section 5.3: a file whose review failed is not a reviewed file.
                # Reporting "completed" here would imply coverage we do not have.
                run.status = RunStatus.FAILED
                run.notes.append(
                    "Review incomplete: the model failed for "
                    f"{len(failed_files)} file(s) ({', '.join(sorted(failed_files)[:5])}"
                    f"{', ...' if len(failed_files) > 5 else ''})."
                )
                return (
                    EXIT_REVIEW_FAILED,
                    f"LLM review failed for {len(failed_files)} file(s)",
                )
            run.status = RunStatus.COMPLETED
            run.notes.append(
                f"Partial review: {run.files_reviewed}/{run.files_scanned} scanned file(s) reviewed."
            )
            return EXIT_OK, None

        run.status = RunStatus.COMPLETED
        return EXIT_OK, None

    # -- helpers ---------------------------------------------------------
    def _deterministic_findings(
        self, static_result: StaticAnalysisResult, agent_result: AgentResult
    ) -> list[ReviewFinding]:
        """Promote actionable lint diagnostics that the model did not produce.

        These are pre-validated: the line came from a real tool run against the
        real file. They are only suppressed when the model already covered that
        exact location, or when the file was never part of the review at all and
        the run was otherwise complete.
        """
        if not static_result.diagnostics:
            return []

        reviewed_files = {
            review.file for review in agent_result.file_reviews if not review.skipped_reason
        }
        model_locations = {
            (finding.file, finding.start_line) for finding in agent_result.findings
        }

        promoted: list[ReviewFinding] = []
        for relative_path, diagnostics in static_result.diagnostics.items():
            if reviewed_files and relative_path not in reviewed_files:
                # The model never saw this file; its diagnostics belong to a
                # file outside the review scope.
                continue
            for diagnostic in diagnostics:
                if diagnostic.severity == "info":
                    continue
                if (diagnostic.file, diagnostic.line) in model_locations:
                    continue
                promoted.append(diagnostic_to_finding(diagnostic))
        return promoted

    def _build_report(
        self,
        *,
        run: ReviewRun,
        findings: list[ReviewFinding],
        scan_report: ScanReport,
        static_result: StaticAnalysisResult,
        agent_result: AgentResult | None,
        validation: ValidationOutcome,
        output_dir: Path,
    ) -> ReviewReport:
        # A fatal static-analysis failure leaves no agent result, so fall back to
        # whatever the run recorded rather than assuming the agent ran.
        skipped_files = (
            [
                {"file": review.file, "reason": review.skipped_reason or "skipped"}
                for review in agent_result.file_reviews
                if review.skipped_reason
            ]
            if agent_result is not None
            else list(run.output_skipped_files)
        )
        rejections = [
            {
                "file": item.finding.file,
                "line": str(item.finding.start_line),
                "reason": item.reason,
                "detail": item.detail,
            }
            for item in [*validation.rejected, *validation.duplicates]
        ]
        scan_summary = {
            "files_scanned": len(scan_report.files),
            "file_limit": scan_report.limit,
            "truncated": scan_report.truncated,
            "skipped_by_reason": dict(sorted(scan_report.skipped.items())),
            "analyzer_runs": static_result.analyzers_run,
            "analyzer_skips": [
                {"name": name, "reason": reason} for name, reason in static_result.skipped_analyzers
            ],
            "diagnostics_total": static_result.total_diagnostics,
            "diagnostics_errors": static_result.error_count,
            "static_analysis_seconds": round(static_result.duration_seconds, 4),
        }
        validation_summary = {
            "candidate_findings": validation.considered,
            "accepted": len(validation.valid),
            "rejected_invalid": len(validation.rejected),
            "rejected_duplicate": len(validation.duplicates),
            "files_reviewed": run.files_reviewed,
            "model_requests": run.token_usage.requests,
            "prompt_tokens": run.token_usage.prompt_tokens,
            "completion_tokens": run.token_usage.completion_tokens,
            "tool_calls": (
                sum(review.tool_calls for review in agent_result.file_reviews)
                if agent_result is not None
                else 0
            ),
            "review_rounds": (
                sum(review.rounds for review in agent_result.file_reviews)
                if agent_result is not None
                else 0
            ),
            "output_dir": str(output_dir),
        }

        return ReviewReport(
            run=run,
            findings=findings,
            scan_summary=scan_summary,
            validation_summary=validation_summary,
            config_summary=self._config_summary(),
            skipped_files=(
                skipped_files if agent_result is not None else list(run.output_skipped_files)
            ),
            rejections=rejections,
        )

    def _config_summary(self) -> dict[str, Any]:
        return {
            "provider": self.config.llm.provider,
            "model": self.config.llm.model,
            "api_key_env": self.config.llm.api_key_env,
            "api_key_present": bool(self.config.api_key),
            "response_format": self.config.llm.response_format,
            "budget": {
                "max_requests": self.config.budget.max_requests,
                "max_total_tokens": self.config.budget.max_total_tokens,
                "max_cost_usd": self.config.budget.max_cost_usd,
                "max_files": self.config.budget.max_files,
                "max_file_bytes": self.config.budget.max_file_bytes,
                "max_context_chars": self.config.budget.max_context_chars,
            },
            "scanner": {
                "extensions": list(self.config.scanner.extensions),
                "max_files": self.config.scanner.max_files,
                "follow_symlinks": self.config.scanner.follow_symlinks,
            },
            "static_analysis": {
                "enabled": self.config.static_analysis.enabled,
                "eslint_command": self.config.static_analysis.eslint_command,
                "builtin_rules": self.config.static_analysis.builtin_rules,
                "fail_on_missing_tool": self.config.static_analysis.fail_on_missing_tool,
            },
            "review": {
                "max_review_rounds_per_file": self.config.review.max_review_rounds_per_file,
                "min_confidence": self.config.review.min_confidence,
                "enable_tool_calls": self.config.review.enable_tool_calls,
                "max_tool_calls_per_file": self.config.review.max_tool_calls_per_file,
                "fail_on_partial_review": self.config.review.fail_on_partial_review,
            },
            "read_only": True,
        }

    def _resolve_output_dir(self, project_root: Path) -> Path:
        if self._output_dir is not None:
            return Path(self._output_dir).expanduser().resolve()
        base = Path(self.config.report.output_dir)
        if base.is_absolute():
            return base
        return (Path(project_root) / base).resolve()
