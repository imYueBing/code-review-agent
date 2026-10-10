"""Core data model: findings, runs, and the schema the model must satisfy.

Section 6 of the design doc. ``ReviewFinding`` is the contract between the
agent, the validator, the report generator and the evaluation harness, so the
field names and constraints live in exactly one place.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .errors import InputError
from .validate import validate

# Aliases the model might plausibly emit, mapped onto the canonical vocabulary.
_SEVERITY_ALIASES = {
    "critical": "critical",
    "crit": "critical",
    "blocker": "critical",
    "high": "high",
    "major": "high",
    "medium": "medium",
    "moderate": "medium",
    "med": "medium",
    "low": "low",
    "minor": "low",
    "info": "info",
    "informational": "info",
    "nit": "info",
}

_CATEGORY_ALIASES = {
    "bug": "bug",
    "correctness": "bug",
    "logic": "bug",
    "security": "security",
    "vulnerability": "security",
    "performance": "performance",
    "perf": "performance",
    "maintainability": "maintainability",
    "style": "maintainability",
    "readability": "maintainability",
    "refactor": "maintainability",
    "maintenance": "maintainability",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    return f"run-{utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @classmethod
    def coerce(cls, value: Any) -> "Severity":
        key = str(value or "").strip().lower()
        canonical = _SEVERITY_ALIASES.get(key)
        if canonical is None:
            raise InputError(f"unknown severity: {value!r}")
        return cls(canonical)

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


class Category(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"

    @classmethod
    def coerce(cls, value: Any) -> "Category":
        key = str(value or "").strip().lower()
        canonical = _CATEGORY_ALIASES.get(key)
        if canonical is None:
            raise InputError(f"unknown category: {value!r}")
        return cls(canonical)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FindingSource(str, Enum):
    """Where a finding came from. Reported so users can judge its weight."""

    STATIC_ANALYSIS = "static_analysis"
    LLM = "llm"
    VALIDATED_LLM = "validated_llm"


@dataclass
class ReviewFinding:
    """A single review issue. See design doc section 6.1/6.2."""

    title: str
    severity: Severity
    category: Category
    file: str
    start_line: int
    end_line: int
    evidence: str
    description: str
    recommendation: str = ""
    confidence: float | None = None
    source: FindingSource = FindingSource.LLM
    id: str = ""

    def __post_init__(self) -> None:
        # Accept raw strings so callers (and tests) can construct findings from
        # loosely-typed model output without pre-normalising.
        if not isinstance(self.severity, Severity):
            self.severity = Severity.coerce(self.severity)
        if not isinstance(self.category, Category):
            self.category = Category.coerce(self.category)
        if not isinstance(self.source, FindingSource):
            self.source = FindingSource(str(self.source))
        if self.start_line > self.end_line:
            self.start_line, self.end_line = self.end_line, self.start_line

    @property
    def dedupe_key(self) -> tuple[str, int, int, str]:
        """Identity used to detect duplicate findings (section 5.2).

        Normalising the title means the same defect reported with slightly
        different wording in the same lines still collapses to one entry.
        """
        return (
            self.file,
            self.start_line,
            self.end_line,
            _normalise_title(self.title),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity.value,
            "category": self.category.value,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "evidence": self.evidence,
            "description": self.description,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: FindingSource | None = None) -> "ReviewFinding":
        confidence = data.get("confidence")
        return cls(
            id=str(data.get("id") or ""),
            title=str(data.get("title") or "").strip(),
            severity=Severity.coerce(data.get("severity")),
            category=Category.coerce(data.get("category")),
            file=str(data.get("file") or "").strip(),
            start_line=int(data.get("start_line") or 0),
            end_line=int(data.get("end_line") or data.get("start_line") or 0),
            evidence=str(data.get("evidence") or ""),
            description=str(data.get("description") or ""),
            recommendation=str(data.get("recommendation") or ""),
            confidence=float(confidence) if confidence is not None else None,
            source=source or FindingSource(str(data.get("source") or FindingSource.LLM.value)),
        )


def _normalise_title(title: str) -> str:
    return " ".join(title.lower().split())


@dataclass
class RunError:
    """A non-fatal problem recorded during the run."""

    stage: str
    message: str
    fatal: bool = False
    at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "fatal": self.fatal,
            "at": self.at.isoformat(),
        }


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.requests += 1

    def merge(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.requests += other.requests

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
        }


@dataclass
class ReviewRun:
    """One review invocation. See design doc section 6.3."""

    run_id: str
    project_root: str
    status: RunStatus = RunStatus.PENDING
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    files_scanned: int = 0
    files_reviewed: int = 0
    findings_count: int = 0
    errors: list[RunError] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    estimated_cost: float | None = None
    phases: list[dict[str, Any]] = field(default_factory=list)
    skipped_analyzers: list[str] = field(default_factory=list)
    output_skipped_files: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def has_fatal_error(self) -> bool:
        return any(error.fatal for error in self.errors)

    def add_error(self, stage: str, message: str, *, fatal: bool = False) -> RunError:
        error = RunError(stage=stage, message=message, fatal=fatal)
        self.errors.append(error)
        return error

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project_root": self.project_root,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "files_scanned": self.files_scanned,
            "files_reviewed": self.files_reviewed,
            "findings_count": self.findings_count,
            "errors": [error.to_dict() for error in self.errors],
            "token_usage": self.token_usage.to_dict(),
            "estimated_cost": self.estimated_cost,
            "phases": list(self.phases),
            "skipped_analyzers": list(self.skipped_analyzers),
            "output_skipped_files": list(self.output_skipped_files),
            "notes": list(self.notes),
        }


#: Schema the model's *per-file* response must satisfy. Deliberately strict on
#: the fields the validator depends on, loose on wording.
FINDINGS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["findings"],
    "additionalProperties": True,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "title",
                    "severity",
                    "category",
                    "file",
                    "start_line",
                    "evidence",
                    "description",
                ],
                "additionalProperties": True,
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string", "minLength": 1},
                    "severity": {
                        "type": "string",
                        "enum": [s.value for s in Severity],
                    },
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in Category],
                    },
                    "file": {"type": "string", "minLength": 1},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "evidence": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    },
}


def schema_errors_for_response(payload: Any) -> list[str]:
    return validate(payload, FINDINGS_RESPONSE_SCHEMA)
