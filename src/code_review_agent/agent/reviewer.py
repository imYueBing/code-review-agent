"""The review agent: a bounded, tool-using analysis loop (design doc 4.2, 5.2).

Flow per file:

1. Build a bounded prompt containing the numbered source and static diagnostics.
2. (Optional) let the model request read-only project tools, execute a strictly
   capped number of them, and feed the results back.
3. Ask for a schema-validated ``{"findings": [...]}`` object.

Every model-facing failure is contained: a file that cannot be reviewed is
recorded and skipped, and the caller learns that the review was partial rather
than being told the project is clean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config.settings import AgentConfig
from ..errors import BudgetExceededError, InputError, LLMError, LLMResponseFormatError
from ..llm.client import BudgetGuard, LLMClient, LLMMessage, LLMRequest
from ..models import FINDINGS_RESPONSE_SCHEMA, FindingSource, ReviewFinding
from ..tools.reader import CodeReader
from ..tools.rules import Diagnostic
from ..tools.scanner import ScannedFile
from .prompts import (
    SYSTEM_PROMPT,
    TOOL_PLANNING_PROMPT,
    build_file_context,
    build_review_message,
    render_tool_results,
)
from .tools import ProjectToolbelt, ToolCall

logger = logging.getLogger(__name__)

#: Schema for the optional tool-planning turn. Intentionally loose on the inner
#: objects: a malformed call is answered with an error message by the toolbelt,
#: which teaches the model far more cheaply than a repair round-trip.
TOOL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["tool_calls"],
    "additionalProperties": True,
    "properties": {
        "tool_calls": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "object", "additionalProperties": True},
        }
    },
}


@dataclass
class FileReview:
    """Outcome of reviewing one file."""

    file: str
    findings: list[ReviewFinding] = field(default_factory=list)
    rounds: int = 0
    tool_calls: int = 0
    skipped_reason: str | None = None
    error: str | None = None


@dataclass
class AgentResult:
    findings: list[ReviewFinding] = field(default_factory=list)
    files_reviewed: int = 0
    file_reviews: list[FileReview] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    budget_stopped_reason: str | None = None

    @property
    def files_skipped(self) -> list[FileReview]:
        return [review for review in self.file_reviews if review.skipped_reason]


class ReviewAgent:
    def __init__(
        self,
        client: LLMClient,
        reader: CodeReader,
        config: AgentConfig,
        guard: BudgetGuard,
        toolbelt: ProjectToolbelt | None = None,
    ) -> None:
        self.client = client
        self.reader = reader
        self.config = config
        self.guard = guard
        self.toolbelt = toolbelt
        self._diagnostics: dict[str, list[Diagnostic]] = {}

    # -- public API ------------------------------------------------------
    def review_files(self, files: list[ScannedFile]) -> AgentResult:
        result = AgentResult()
        for scanned in files:
            try:
                self.guard.check()
            except BudgetExceededError as exc:
                self._note_budget_stop(result, exc)
                break

            review = self.review_file(scanned)
            result.file_reviews.append(review)
            if review.error:
                result.errors.append((scanned.relative_path, review.error))
                continue
            if review.skipped_reason:
                continue
            result.files_reviewed += 1
            result.findings.extend(review.findings)

            # The budget may have run out mid-file. Stop cleanly and report
            # partial coverage rather than raising, so the caller can still
            # produce a report that says the review was cut short.
            if self.guard.exhausted:
                self._note_budget_stop(result, BudgetExceededError(self.guard.stopped_reason or ""))
                break

        return result

    @staticmethod
    def _note_budget_stop(result: AgentResult, exc: BudgetExceededError) -> None:
        message = str(exc)
        result.budget_stopped_reason = message
        logger.warning("stopping review loop: %s", message)
        if not any(stage == "budget" for stage, _ in result.errors):
            result.errors.append(("budget", message))

    def review_file(self, scanned: ScannedFile) -> FileReview:
        review = FileReview(file=scanned.relative_path)
        source = self.reader.read_safe(scanned.relative_path)
        if source is None:
            review.skipped_reason = "unreadable"
            return review

        diagnostics: list[Diagnostic] = []
        if self.config.review.include_static_diagnostics:
            diagnostics = self._diagnostics_for(scanned.relative_path)

        context = build_file_context(
            source,
            diagnostics,
            max_chars=self.config.budget.max_context_chars,
        )
        messages = [LLMMessage(role="user", content=build_review_message(context))]

        try:
            messages = self._maybe_run_tools(scanned.relative_path, messages, review)
            findings = self._request_findings(scanned.relative_path, messages, review)
        except BudgetExceededError:
            raise
        except LLMError as exc:
            review.error = str(exc)
            review.skipped_reason = "llm-failure"
            logger.warning("LLM review failed for %s: %s", scanned.relative_path, exc)
            return review

        review.findings = findings
        return review

    # -- internals -------------------------------------------------------
    def _diagnostics_for(self, relative_path: str) -> list[Diagnostic]:
        # Populated by the workflow; the agent only reads it.
        return self._diagnostics.get(relative_path, [])

    def set_diagnostics(self, diagnostics: dict[str, list[Diagnostic]]) -> None:
        self._diagnostics = diagnostics

    def _maybe_run_tools(
        self, relative_path: str, messages: list[LLMMessage], review: FileReview
    ) -> list[LLMMessage]:
        if not self.config.review.enable_tool_calls or self.toolbelt is None:
            return messages
        max_calls = max(0, self.config.review.max_tool_calls_per_file)
        if max_calls == 0:
            return messages

        plan_request = LLMRequest(
            messages=[
                *messages,
                LLMMessage(role="user", content=TOOL_PLANNING_PROMPT),
            ],
            system=SYSTEM_PROMPT,
            schema=TOOL_PLAN_SCHEMA,
            temperature=0.0,
            purpose="tool_planning",
        )
        try:
            plan = self.client.complete_json(plan_request)  # type: ignore[attr-defined]
        except BudgetExceededError:
            raise
        except LLMError as exc:
            # Tool planning is a bonus; failing it must not fail the review.
            logger.info("tool planning unavailable for %s: %s", relative_path, exc)
            return messages

        calls = ToolCall.parse_many(plan, limit=max_calls)
        if not calls:
            return messages

        results: list[dict[str, Any]] = []
        for call in calls:
            try:
                self.guard.check()
            except BudgetExceededError:
                logger.warning("budget exhausted while running tools; continuing without them")
                break
            results.append(self.toolbelt.execute(call))
            review.tool_calls += 1

        if not results:
            return messages
        return [*messages, LLMMessage(role="user", content=render_tool_results(results))]

    def _request_findings(
        self, relative_path: str, messages: list[LLMMessage], review: FileReview
    ) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []

        for round_index in range(max(1, self.config.review.max_review_rounds_per_file)):
            try:
                self.guard.check()
            except BudgetExceededError:
                # No budget for another round: keep the findings we already
                # have. The caller detects guard.exhausted and stops the run.
                break
            request = LLMRequest(
                messages=messages,
                system=SYSTEM_PROMPT,
                schema=FINDINGS_RESPONSE_SCHEMA,
                temperature=0.0,
                purpose=f"review:{relative_path}" if round_index == 0 else f"review_round2:{relative_path}",
            )
            payload = self.client.complete_json(request)  # type: ignore[attr-defined]
            review.rounds += 1

            round_findings = self._parse_findings(payload, relative_path)
            findings.extend(round_findings)

            if round_findings or round_index + 1 >= self.config.review.max_review_rounds_per_file:
                break

            # Second pass only when the first produced nothing: a cheap recall
            # hedge for the "no findings" outcome.
            messages = [
                *messages,
                LLMMessage(
                    role="user",
                    content=(
                        "You reported no findings for this file. Re-examine it once more "
                        "specifically for: null/undefined access, incorrect boundary or "
                        "boolean conditions, unhandled promise rejections, missing cleanup, "
                        "and unsafe input handling. Omit anything you cannot evidence. "
                        "Return the same JSON object."
                    ),
                ),
            ]

        # A round that yields nothing is a valid, common outcome.
        return findings[: self.config.review.max_findings_per_file]

    def _parse_findings(self, payload: dict[str, Any], relative_path: str) -> list[ReviewFinding]:
        raw_findings = payload.get("findings")
        if raw_findings is None:
            raise LLMResponseFormatError("model response had no 'findings' key")
        if not isinstance(raw_findings, list):
            raise LLMResponseFormatError("'findings' must be an array")

        findings: list[ReviewFinding] = []
        for entry in raw_findings:
            if not isinstance(entry, dict):
                logger.info("discarding non-object finding from model for %s", relative_path)
                continue
            # The model may cite another file; allow it, and let the validator
            # confirm the path. Only default to the reviewed file when absent.
            entry = dict(entry)
            entry.setdefault("file", relative_path)
            try:
                finding = ReviewFinding.from_dict(entry, source=FindingSource.VALIDATED_LLM)
            except InputError as exc:
                logger.info("discarding malformed finding for %s: %s", relative_path, exc)
                continue
            confidence = finding.confidence
            if confidence is not None and confidence < self.config.review.min_confidence:
                logger.info(
                    "discarding low-confidence finding (%.2f < %.2f): %s",
                    confidence,
                    self.config.review.min_confidence,
                    finding.title,
                )
                continue
            findings.append(finding)
        return findings


def diagnostic_to_finding(diagnostic: Diagnostic) -> ReviewFinding:
    """Convert a static diagnostic into a finding when the model cannot.

    Section 1.3 ("deterministic first"): a linter error is real evidence even
    with no model available, so it still reaches the report.
    """
    severity = {"error": "high", "warning": "medium", "info": "low"}.get(diagnostic.severity, "low")
    category = "security" if diagnostic.rule_id in _SECURITY_RULES else "bug"
    if diagnostic.rule_id in _MAINTAINABILITY_RULES:
        category = "maintainability"
    return ReviewFinding(
        title=diagnostic.message or diagnostic.rule_id,
        severity=severity,
        category=category,
        file=diagnostic.file,
        start_line=diagnostic.line,
        end_line=diagnostic.line,
        evidence="",
        description=f"Reported by {diagnostic.source} rule `{diagnostic.rule_id}`.",
        recommendation=f"Review and resolve the `{diagnostic.rule_id}` diagnostic.",
        confidence=None,
        source=FindingSource.STATIC_ANALYSIS,
    )


_SECURITY_RULES = {
    "no-eval",
    "no-function-constructor",
    "no-inner-html",
    "no-document-write",
    "no-hardcoded-secret",
}

_MAINTAINABILITY_RULES = {
    "no-var",
    "no-console-debug",
    "timeout-not-cleared",
}
