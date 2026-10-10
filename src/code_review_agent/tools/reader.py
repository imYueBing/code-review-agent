"""Reading source files with reliable line-number mapping.

The agent must never invent line numbers, and the validator must be able to
check them (design doc 6.2, 9.1). Every read therefore goes through
:class:`SourceFile`, which owns the authoritative 1-based line list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..errors import SecurityError, ToolError
from ..security import PathGuard, redact_secrets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Line:
    number: int  # 1-based, matching editor and linter conventions
    text: str


class SourceFile:
    """An in-memory snapshot of one source file."""

    def __init__(self, relative_path: str, text: str, *, truncated: bool = False) -> None:
        self.relative_path = relative_path
        self.text = text
        self.truncated = truncated
        # splitlines() drops a trailing newline, which is what we want: the
        # resulting indices are exactly the lines a user sees in an editor.
        self._lines: list[str] = text.splitlines()

    @property
    def line_count(self) -> int:
        return len(self._lines)

    def lines(self) -> list[Line]:
        return [Line(number=i + 1, text=value) for i, value in enumerate(self._lines)]

    def line(self, number: int) -> str | None:
        if 1 <= number <= len(self._lines):
            return self._lines[number - 1]
        return None

    def is_valid_line(self, number: int) -> bool:
        return 1 <= number <= len(self._lines)

    def numbered_text(self, *, start: int = 1, end: int | None = None) -> str:
        """Render lines with 1-based numbering, e.g. ``  12 | const x = 1``."""
        end = end if end is not None else len(self._lines)
        start = max(1, start)
        end = min(end, len(self._lines))
        width = len(str(end)) if end else 1
        return "\n".join(
            f"{i:>{width}} | {self._lines[i - 1]}" for i in range(start, end + 1)
        )

    def excerpt(self, start_line: int, end_line: int, *, padding: int = 2) -> str:
        return self.numbered_text(
            start=max(1, start_line - padding), end=min(self.line_count, end_line + padding)
        )


class CodeReader:
    """Reads files strictly inside the authorised project."""

    def __init__(self, guard: PathGuard, *, max_bytes: int = 200_000) -> None:
        self.guard = guard
        self.max_bytes = max_bytes
        self._cache: dict[str, SourceFile] = {}

    def read(self, relative_path: str) -> SourceFile:
        """Read *relative_path*; raises :class:`ToolError` if it is unusable."""
        cached = self._cache.get(relative_path)
        if cached is not None:
            return cached

        try:
            path = self.guard.resolve(relative_path)
        except SecurityError as exc:
            raise ToolError(str(exc)) from exc

        if not path.is_file():
            raise ToolError(f"not a file: {relative_path}")

        try:
            size = path.stat().st_size
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(self.max_bytes + 1)
        except OSError as exc:
            raise ToolError(f"cannot read {relative_path}: {exc}") from exc

        truncated = len(text) > self.max_bytes or size > self.max_bytes
        if truncated:
            logger.warning("truncating %s at %d bytes", relative_path, self.max_bytes)
            text = text[: self.max_bytes]
            # Never leave a half-line behind: it would shift the apparent code.
            if "\n" in text:
                text = text[: text.rfind("\n")]

        source = SourceFile(self.guard.relative(path), text, truncated=truncated)
        self._cache[relative_path] = source
        return source

    def read_safe(self, relative_path: str) -> SourceFile | None:
        """Read, returning ``None`` instead of raising.

        Section 5.3: an unreadable file is a warning that does not abort the
        review of other files.
        """
        try:
            return self.read(relative_path)
        except ToolError as exc:
            logger.warning("skipping unreadable file: %s", exc)
            return None

    def evidence_matches(self, relative_path: str, start_line: int, end_line: int, evidence: str) -> bool:
        """Check *evidence* actually appears in the cited line range.

        Used by the validator to reject findings whose evidence is fabricated.
        Whitespace is normalised, because models routinely re-indent snippets.
        """
        source = self._cache.get(relative_path)
        if source is None:
            return False
        if not evidence.strip():
            return False
        needle = _normalise_whitespace(redact_secrets(evidence))
        if not needle:
            return False
        if not source.is_valid_line(start_line) or not source.is_valid_line(end_line):
            return False
        haystack = _normalise_whitespace(source.excerpt(start_line, end_line, padding=0))
        if needle in haystack:
            return True
        # Multi-line evidence: compare line-by-line ignoring surrounding space.
        if "\n" in evidence:
            for line in evidence.splitlines():
                candidate = _normalise_whitespace(line)
                if candidate and candidate in haystack:
                    return True
        return False


def _normalise_whitespace(text: str) -> str:
    return " ".join(text.split())
