"""Report generation (design doc 2.1, 6.3, 11).

Two guarantees matter here:

* The summary never implies a clean project. "Review completed" states what was
  reviewed and what was skipped; it is not a verdict.
* A write failure raises :class:`ReportError`. A run that could not produce its
  report is never reported as a success (section 5.3).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import AgentConfig
from ..errors import ReportError
from ..models import ReviewFinding, ReviewRun, RunStatus, Severity
from ..security import redact_secrets

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = "1.0"

_SEVERITY_BADGE = {
    Severity.CRITICAL: "critical",
    Severity.HIGH: "high",
    Severity.MEDIUM: "medium",
    Severity.LOW: "low",
    Severity.INFO: "info",
}


@dataclass
class ReviewReport:
    """Everything needed to render a report, kept free of live objects."""

    run: ReviewRun
    findings: list[ReviewFinding] = field(default_factory=list)
    scan_summary: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    config_summary: dict[str, Any] = field(default_factory=dict)
    skipped_files: list[dict[str, str]] = field(default_factory=list)
    rejections: list[dict[str, str]] = field(default_factory=list)

    @property
    def severity_counts(self) -> dict[str, int]:
        counts = Counter(finding.severity.value for finding in self.findings)
        return {severity.value: counts.get(severity.value, 0) for severity in Severity}

    @property
    def category_counts(self) -> dict[str, int]:
        return dict(Counter(finding.category.value for finding in self.findings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run": self.run.to_dict(),
            "summary": {
                "severity_counts": self.severity_counts,
                "category_counts": self.category_counts,
                "files_scanned": self.run.files_scanned,
                "files_reviewed": self.run.files_reviewed,
                "findings_count": len(self.findings),
                # Explicit so a reader cannot misread the run as "clean".
                "review_complete": self.run.status is RunStatus.COMPLETED,
                "caveats": self.caveats(),
            },
            "scan": self.scan_summary,
            "validation": self.validation_summary,
            "config": self.config_summary,
            "skipped_files": self.skipped_files,
            "rejected_findings": self.rejections,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def caveats(self) -> list[str]:
        notes: list[str] = []
        if self.run.status is not RunStatus.COMPLETED:
            notes.append(
                f"Review status is '{self.run.status.value}'; results are partial and must not be "
                "read as a clean review."
            )
        if self.run.skipped_analyzers:
            names = ", ".join(name for name, _ in self.run.skipped_analyzers)
            notes.append(f"Static analysis was not fully applied; skipped: {names}.")
        if self.run.files_reviewed < self.run.files_scanned:
            notes.append(
                f"{self.run.files_scanned - self.run.files_reviewed} scanned file(s) were not "
                "reviewed; see skipped_files."
            )
        if self.run.errors:
            notes.append(f"{len(self.run.errors)} error(s) were recorded during the run.")
        notes.extend(self.run.notes)
        return notes


class ReportGenerator:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    # -- rendering -------------------------------------------------------
    def render_markdown(self, report: ReviewReport) -> str:
        run = report.run
        lines: list[str] = []
        lines.append("# Code Review Report")
        lines.append("")
        lines.append(f"- **Run ID:** `{run.run_id}`")
        lines.append(f"- **Project:** `{run.project_root}`")
        lines.append(f"- **Status:** `{run.status.value}`")
        lines.append(f"- **Started:** {run.started_at.isoformat()}")
        if run.finished_at:
            lines.append(f"- **Finished:** {run.finished_at.isoformat()}")
        if run.duration_seconds is not None:
            lines.append(f"- **Duration:** {run.duration_seconds:.2f}s")
        lines.append("")

        if run.status is not RunStatus.COMPLETED:
            lines.append(
                "> **This review did not complete.** The findings below are partial. "
                "A partial or failed review is not a passing review."
            )
            lines.append("")

        if not report.findings:
            lines.append(
                "> **No findings were reported.** This means the configured checks and the model "
                "produced no evidence-backed issues. It does not guarantee the code is correct."
            )
            lines.append("")

        counts = report.severity_counts
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Files scanned | {run.files_scanned} |")
        lines.append(f"| Files reviewed | {run.files_reviewed} |")
        lines.append(f"| Findings | {len(report.findings)} |")
        lines.append(f"| Critical | {counts['critical']} |")
        lines.append(f"| High | {counts['high']} |")
        lines.append(f"| Medium | {counts['medium']} |")
        lines.append(f"| Low | {counts['low']} |")
        lines.append(f"| Info | {counts['info']} |")
        if report.category_counts:
            categories = ", ".join(
                f"{name}: {value}" for name, value in sorted(report.category_counts.items())
            )
            lines.append(f"| Categories | {categories} |")
        lines.append(f"| Model requests | {run.token_usage.requests} |")
        lines.append(f"| Total tokens | {run.token_usage.total_tokens} |")
        if run.estimated_cost is not None:
            lines.append(f"| Estimated cost | ${run.estimated_cost:.4f} |")
        else:
            lines.append("| Estimated cost | unknown (no price for this model) |")
        lines.append("")

        caveats = report.caveats()
        if caveats:
            lines.append("## Caveats")
            lines.append("")
            for caveat in caveats:
                lines.append(f"- {caveat}")
            lines.append("")

        if report.findings:
            lines.append("## Findings")
            lines.append("")
            for finding in self._sorted(report.findings):
                lines.append(
                    f"### {finding.id or 'finding'} · {_SEVERITY_BADGE[finding.severity].upper()} · "
                    f"{finding.category.value}"
                )
                lines.append("")
                lines.append(f"**{finding.title}**")
                lines.append("")
                lines.append(
                    f"- **Location:** `{finding.file}:{finding.start_line}"
                    + (f"-{finding.end_line}" if finding.end_line != finding.start_line else "")
                    + "`"
                )
                lines.append(f"- **Source:** `{finding.source.value}`")
                if finding.confidence is not None:
                    lines.append(f"- **Model confidence:** {finding.confidence:.2f} (self-reported)")
                lines.append("")
                lines.append(finding.description.strip() or "_No description provided._")
                lines.append("")
                if finding.evidence.strip():
                    lines.append("Evidence:")
                    lines.append("")
                    lines.append("```")
                    lines.append(self._clip(finding.evidence))
                    lines.append("```")
                    lines.append("")
                if finding.recommendation.strip():
                    lines.append(f"**Recommendation:** {finding.recommendation.strip()}")
                    lines.append("")

        if report.skipped_files:
            lines.append("## Skipped Files")
            lines.append("")
            lines.append("| File | Reason |")
            lines.append("| --- | --- |")
            for entry in report.skipped_files:
                lines.append(f"| `{entry['file']}` | {entry['reason']} |")
            lines.append("")

        if report.rejections:
            lines.append("## Discarded Candidate Findings")
            lines.append("")
            lines.append(
                "These candidates failed validation (bad path, invalid line, unverifiable "
                "evidence, or duplication) and are excluded from the findings above."
            )
            lines.append("")
            lines.append("| File | Line | Reason | Detail |")
            lines.append("| --- | --- | --- | --- |")
            for entry in report.rejections:
                detail = entry.get("detail", "").replace("|", "\\|")
                lines.append(
                    f"| `{entry.get('file', '')}` | {entry.get('line', '')} | "
                    f"{entry.get('reason', '')} | {detail} |"
                )
            lines.append("")

        if report.validation_summary:
            lines.append("## Validation")
            lines.append("")
            for key, value in sorted(report.validation_summary.items()):
                lines.append(f"- {key}: `{value}`")
            lines.append("")

        lines.append("## Environment")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(report.config_summary, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

        if report.scan_summary:
            lines.append("## Scan Detail")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(report.scan_summary, indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")

        if run.errors:
            lines.append("## Errors")
            lines.append("")
            lines.append("| Stage | Fatal | Message |")
            lines.append("| --- | --- | --- |")
            for error in run.errors:
                message = redact_secrets(error.message).replace("|", "\\|")
                lines.append(f"| {error.stage} | {'yes' if error.fatal else 'no'} | {message} |")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(
            f"_Generated by code-review-agent {REPORT_SCHEMA_VERSION}. "
            "This tool reads code and never modifies the reviewed project._"
        )
        lines.append("")
        return "\n".join(lines)

    def _sorted(self, findings: list[ReviewFinding]) -> list[ReviewFinding]:
        return sorted(
            findings,
            key=lambda f: (f.severity.rank, f.file, f.start_line, f.title),
        )

    def _clip(self, text: str) -> str:
        limit = self.config.report.max_evidence_chars
        redacted = redact_secrets(text)
        if len(redacted) <= limit:
            return redacted
        return redacted[:limit] + "\n... (evidence truncated)"

    # -- writing ---------------------------------------------------------
    def write(self, report: ReviewReport, output_dir: Path) -> tuple[Path, Path]:
        """Write Markdown and JSON reports. Raises :class:`ReportError` on failure."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ReportError(f"cannot create report directory {output_dir}: {exc}") from exc

        markdown_path = output_dir / self.config.report.markdown_filename
        json_path = output_dir / self.config.report.json_filename

        markdown = self.render_markdown(report)
        payload = report.to_dict()

        for path, content in ((markdown_path, markdown), (json_path, json.dumps(payload, indent=2) + "\n")):
            try:
                path.write_text(content, encoding="utf-8")
            except OSError as exc:
                raise ReportError(f"cannot write report {path}: {exc}") from exc
            logger.info("wrote report: %s", path)

        return markdown_path, json_path
