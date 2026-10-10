"""Project scanning and file filtering (design doc 5.2 / 2.1).

The scanner is the only component that walks the filesystem, and it does so
through :class:`PathGuard` so nothing outside the authorised project can ever be
enumerated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import AgentConfig
from ..errors import InputError, SecurityError
from ..security import FileFilter, PathGuard

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScannedFile:
    """A reviewable file discovered by the scanner."""

    path: Path
    relative_path: str
    size_bytes: int
    line_count: int

    @property
    def language(self) -> str:
        return {
            ".ts": "typescript",
            ".tsx": "tsx",
            ".js": "javascript",
            ".jsx": "jsx",
            ".mjs": "javascript",
            ".cjs": "javascript",
        }.get(self.path.suffix.lower(), "unknown")


@dataclass
class ScanReport:
    files: list[ScannedFile] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    limit: int = 0

    @property
    def total_considered(self) -> int:
        return len(self.files) + sum(self.skipped.values())

    def note_skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


class ProjectScanner:
    """Walks a project directory and returns the reviewable files."""

    def __init__(self, guard: PathGuard, config: AgentConfig) -> None:
        self.guard = guard
        self.config = config
        self.filter = FileFilter(
            extensions=tuple(config.scanner.extensions),
            max_file_bytes=config.budget.max_file_bytes,
        )

    def validate_root(self) -> None:
        """Confirm the target exists, is a directory, and is readable."""
        root = self.guard.root
        if not root.is_dir():
            raise InputError(f"not a directory: {root}")
        try:
            # Cheap readability probe: listing the root fails on a dir we cannot read.
            next(iter(root.iterdir()), None)
        except PermissionError as exc:
            raise InputError(f"directory is not readable: {root}") from exc
        except OSError as exc:
            raise InputError(f"cannot read directory {root}: {exc}") from exc

    def scan(self) -> ScanReport:
        report = ScanReport(limit=self.config.scanner.max_files)
        root = self.guard.root
        follow = self.config.scanner.follow_symlinks

        for dirpath, dirnames, filenames in _walk(root, follow_symlinks=follow):
            current = Path(dirpath)
            # Prune excluded directories in place so os.walk never descends.
            kept_dirs = []
            for name in sorted(dirnames):
                if self.filter.is_excluded_dir(name):
                    report.note_skip("excluded-directory")
                    continue
                candidate = current / name
                if candidate.is_symlink() and not follow:
                    report.note_skip("symlink-directory")
                    continue
                kept_dirs.append(name)
            dirnames[:] = kept_dirs

            for filename in sorted(filenames):
                path = current / filename
                reason = self._rejection_reason(path)
                if reason:
                    report.note_skip(reason)
                    continue
                if len(report.files) >= self.config.scanner.max_files:
                    report.truncated = True
                    report.note_skip("over-file-limit")
                    continue
                try:
                    stat = path.stat()
                    text_size = stat.st_size
                    line_count = _count_lines(path)
                except OSError as exc:
                    logger.warning("cannot stat %s: %s", path, exc)
                    report.note_skip("unreadable")
                    continue
                report.files.append(
                    ScannedFile(
                        path=path,
                        relative_path=self.guard.project_relative(path),
                        size_bytes=text_size,
                        line_count=line_count,
                    )
                )

        # Review smaller files first: highest chance of fitting in the budget.
        report.files.sort(key=lambda f: (f.size_bytes, f.relative_path))
        return report

    def _rejection_reason(self, path: Path) -> str | None:
        if path.is_symlink():
            if not self.config.scanner.follow_symlinks:
                return "symlink"
            try:
                # A symlink is only acceptable if it still lands inside the root.
                self.guard.resolve(path)
            except SecurityError:
                return "symlink-outside-root"
        try:
            size = path.stat().st_size
        except OSError:
            return "unreadable"
        # Only the final component is kept un-resolved, so a symlink named
        # alias.ts is reviewed as alias.ts. Containment was already enforced by
        # guard.resolve() above for symlinks, and out-of-root paths are caught
        # here.
        try:
            relative = Path(self.guard.project_relative(path))
        except SecurityError:
            return "outside-root"
        return self.filter.rejection_reason(relative, size)


def _walk(root: Path, *, follow_symlinks: bool):
    import os

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        yield dirpath, dirnames, filenames


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0
