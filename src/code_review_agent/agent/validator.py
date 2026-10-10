"""Finding validation: evidence, line numbers, and de-duplication (doc 5.2).

This is the component that makes the review *evidence-based* rather than
trusting. A finding whose file does not exist, whose line numbers are out of
range, or whose quoted evidence is absent from the cited lines is rejected and
recorded, never silently reported as real.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import AgentConfig
from ..models import FindingSource, ReviewFinding
from ..security import PathGuard, redact_secrets
from ..tools.reader import CodeReader
from ..tools.scanner import ScannedFile

logger = logging.getLogger(__name__)


@dataclass
class Rejection:
    finding: ReviewFinding
    reason: str
    detail: str

    def describe(self) -> str:
        location = f"{self.finding.file}:{self.finding.start_line}"
        return f"{self.reason} [{location}] {self.finding.title!r}: {self.detail}"


@dataclass
class ValidationOutcome:
    valid: list[ReviewFinding] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    duplicates: list[Rejection] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def considered(self) -> int:
        return len(self.valid) + len(self.rejected) + len(self.duplicates)


class FindingValidator:
    def __init__(
        self,
        guard: PathGuard,
        reader: CodeReader,
        config: AgentConfig,
        *,
        require_evidence: bool = True,
        near_duplicate_threshold: float = 0.7,
    ) -> None:
        self.guard = guard
        self.reader = reader
        self.config = config
        self.require_evidence = require_evidence
        self.near_duplicate_threshold = near_duplicate_threshold

    def validate(
        self,
        findings: list[ReviewFinding],
        files: list[ScannedFile],
    ) -> ValidationOutcome:
        outcome = ValidationOutcome()
        known_files = {scanned.relative_path: scanned for scanned in files}
        seen_keys: set[tuple[str, int, int, str]] = set()
        # Preserve a canonical title per (file, line-bucket) for fuzzy checks.
        seen_titles: list[tuple[str, set[str]]] = []

        for finding in findings:
            rejection = self._location_rejection(finding, known_files)
            if rejection is not None:
                outcome.rejected.append(rejection)
                continue

            key = finding.dedupe_key
            if key in seen_keys:
                outcome.duplicates.append(
                    Rejection(finding, "duplicate", "an identical finding was already accepted")
                )
                continue

            similar = self._near_duplicate(finding, seen_titles)
            if similar is not None:
                outcome.duplicates.append(
                    Rejection(finding, "near-duplicate", f"overlaps with {similar!r}")
                )
                continue

            evidence_rejection = self._evidence_rejection(finding)
            if evidence_rejection is not None:
                outcome.rejected.append(evidence_rejection)
                continue

            seen_keys.add(key)
            seen_titles.append((finding.file, _title_tokens(finding.title)))
            outcome.valid.append(finding)

        outcome.stats = {
            "considered": outcome.considered,
            "accepted": len(outcome.valid),
            "rejected": len(outcome.rejected),
            "duplicates": len(outcome.duplicates),
        }
        return outcome

    # -- individual checks ----------------------------------------------
    def _escapes_project(self, reported_path: str) -> bool:
        """True when *reported_path* leaves the project, directly or via a link.

        :meth:`PathGuard.resolve` cannot answer this: it follows the symlink and
        then performs the containment check, so a link pointing outside raises
        the same error as a path that does not exist. Here the symlink-aware
        resolution is done explicitly so the two cases stay distinguishable in
        the report.
        """
        root = self.guard.root
        candidate = Path(reported_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        return not (resolved == root or root in resolved.parents)

    def _location_rejection(
        self, finding: ReviewFinding, known_files: dict[str, ScannedFile]
    ) -> Rejection | None:
        if not finding.file:
            return Rejection(finding, "missing-file", "finding has no file path")
        if finding.file not in known_files:
            # Distinguish a genuine escape from a path that simply was not part
            # of this review. The test is symlink-aware resolution of the final
            # component: a link inside the project pointing outside it (or an
            # absolute path elsewhere) is an escape, a typo is not.
            if self._escapes_project(finding.file):
                return Rejection(
                    finding,
                    "path-escape",
                    "path resolves outside the authorised project directory",
                )
            return Rejection(
                finding,
                "unknown-file",
                "file is not among the reviewed project files (may have been filtered out)",
            )

        source = self.reader.read_safe(finding.file)
        if source is None:
            return Rejection(finding, "unreadable-file", "file could not be read")

        if finding.start_line < 1 or finding.end_line < 1:
            return Rejection(
                finding,
                "invalid-line",
                f"line numbers must be >= 1, got {finding.start_line}-{finding.end_line}",
            )
        if not source.is_valid_line(finding.start_line):
            return Rejection(
                finding,
                "line-out-of-range",
                f"start_line {finding.start_line} exceeds file length {source.line_count}",
            )
        if not source.is_valid_line(finding.end_line):
            return Rejection(
                finding,
                "line-out-of-range",
                f"end_line {finding.end_line} exceeds file length {source.line_count}",
            )
        return None

    def _evidence_rejection(self, finding: ReviewFinding) -> Rejection | None:
        evidence = (finding.evidence or "").strip()
        if not evidence:
            # Static-analysis findings cite a diagnostic message and carry no
            # source snippet; for them, line validity is the meaningful check.
            if finding.source is FindingSource.STATIC_ANALYSIS:
                return None
            if self.require_evidence:
                return Rejection(finding, "missing-evidence", "finding quoted no code evidence")
            return None

        redacted = redact_secrets(evidence)
        if redacted != evidence:
            finding.evidence = redacted

        if not self.reader.evidence_matches(
            finding.file, finding.start_line, finding.end_line, redacted
        ):
            # A model that cites a plausible but wrong line is a false positive
            # we can catch for free. Widening the search would defeat the point.
            if finding.source is FindingSource.STATIC_ANALYSIS:
                return None
            return Rejection(
                finding,
                "evidence-mismatch",
                "quoted evidence does not appear at the cited lines",
            )
        return None

    def _near_duplicate(
        self, finding: ReviewFinding, seen_titles: list[tuple[str, set[str]]]
    ) -> str | None:
        tokens = _title_tokens(finding.title)
        if not tokens:
            return None
        for file_path, previous in seen_titles:
            if file_path != finding.file:
                continue
            overlap = _jaccard(tokens, previous)
            if overlap >= self.near_duplicate_threshold:
                return " ".join(sorted(previous))
        return None


def _title_tokens(title: str) -> set[str]:
    import re

    stop = {
        "the", "a", "an", "is", "are", "in", "on", "of", "to", "and", "or", "for",
        "with", "when", "not", "no", "may", "might", "could", "can", "be", "can",
        "this", "that", "it", "its", "as", "at", "by", "from", "into", "than",
    }
    words = re.findall(r"[a-z0-9_]+", title.lower())
    return {word for word in words if word not in stop and len(word) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
