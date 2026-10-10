"""Sandbox primitives: path authorisation, sensitive-file rules, secret redaction.

Design doc section 8. Everything in the tool layer must route file access
through :class:`PathGuard` so the agent can only ever touch the authorised
project directory, and never traverses out via ``..`` or a symlink.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import SecurityError

#: Directory names that are never traversed, per design doc section 5.2.
DEFAULT_EXCLUDED_DIRS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "bower_components",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "out",
    "coverage",
    ".next",
    ".nuxt",
    ".cache",
    ".code-review",
    ".idea",
    ".vscode",
    "vendor",
    "target",
)

#: Glob patterns of files that are excluded because they are binary, generated,
#: lockfiles, or otherwise not useful review targets.
DEFAULT_EXCLUDED_FILE_GLOBS: tuple[str, ...] = (
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.svg",
    "*.pdf",
    "*.zip",
    "*.gz",
    "*.tar",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.mp4",
    "*.mp3",
    "*.wasm",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.pyc",
)

#: Files whose contents must never be sent to the model (design doc 8.4).
SENSITIVE_FILE_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "*.ppk",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "_netrc",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "*.secret",
    ".htpasswd",
    "*.kdbx",
)

#: Credential-ish identifier fragments. Matching is done on normalised word
#: fragments rather than with a word-boundary regex, because ``\b`` does not
#: fire between ``access`` and ``_token`` (underscore is a word character), and
#: because ``apiKey``/``API_KEY``/``api-key`` must all be recognised.
_SECRET_KEY_WORDS = frozenset(
    {
        "apikey",
        "secret",
        "secrets",
        "passwd",
        "password",
        "passwords",
        "token",
        "tokens",
        "credential",
        "credentials",
        "privatekey",
        "clientsecret",
        "accesskey",
        "secretkey",
        "bearer",
        "auth",
        "authorization",
    }
)

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")

# Credential-looking literals. Deliberately conservative to avoid mangling
# ordinary source: requires a long-ish opaque run or a known key prefix.
_SECRET_VALUE_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    # key = "...." / key: '....' style assignments with a long literal
    re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|token|access[_-]?key)\b"
        r"(\s*[:=]\s*)([\"'])([^\"'\n]{8,})\3"
    ),
)

REDACTED = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Replace credential-looking substrings with ``[REDACTED]``.

    Applied to model output, logs and report evidence so that a model echoing
    a secret back never results in that secret landing in a report file.
    """
    if not text:
        return text
    for pattern in _SECRET_VALUE_RES:
        if pattern.groups >= 4:
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(3)}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def looks_sensitive_key(name: str) -> bool:
    """True when a *variable/field name* suggests it holds a credential.

    ``apiKey``, ``API_KEY``, ``api-key`` and ``access_token`` all normalise to
    credential words; ``username`` and ``monkey`` do not.
    """
    if not name:
        return False
    fragments = [part.lower() for part in _TOKEN_SPLIT_RE.split(name) if part]
    if not fragments:
        return False
    # Also test adjacent pairs, so "private" + "key" is caught as "privatekey".
    joined = "".join(fragments)
    if joined in _SECRET_KEY_WORDS:
        return True
    for index in range(len(fragments) - 1):
        if fragments[index] + fragments[index + 1] in _SECRET_KEY_WORDS:
            return True
    return any(fragment in _SECRET_KEY_WORDS for fragment in fragments)


@dataclass(frozen=True)
class FileFilter:
    """Decides which files are reviewable. Pure and independently testable."""

    extensions: tuple[str, ...] = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
    excluded_dirs: tuple[str, ...] = DEFAULT_EXCLUDED_DIRS
    excluded_file_globs: tuple[str, ...] = DEFAULT_EXCLUDED_FILE_GLOBS
    sensitive_file_globs: tuple[str, ...] = SENSITIVE_FILE_GLOBS
    max_file_bytes: int = 200_000

    def is_excluded_dir(self, name: str) -> bool:
        return name in self.excluded_dirs

    def is_sensitive(self, path: Path) -> bool:
        """Sensitive either by filename pattern or by living in a secrets dir."""
        name = path.name
        if any(fnmatch.fnmatch(name, pat) for pat in self.sensitive_file_globs):
            return True
        lowered = {part.lower() for part in path.parts}
        return bool(lowered & {".ssh", ".aws", ".gnupg", "secrets"})

    def is_supported_extension(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def is_excluded_file(self, path: Path) -> bool:
        name = path.name
        if name.endswith(".d.ts"):
            # Type declarations are generated/ambient far more often than not.
            return True
        return any(fnmatch.fnmatch(name, pat) for pat in self.excluded_file_globs)

    def rejection_reason(self, path: Path, size: int | None = None) -> str | None:
        """Return why *path* is not reviewable, or ``None`` if it is."""
        if self.is_sensitive(path):
            return "sensitive"
        if not self.is_supported_extension(path):
            return "unsupported-extension"
        if self.is_excluded_file(path):
            return "excluded-pattern"
        if size is not None and size > self.max_file_bytes:
            return "too-large"
        return None


class PathGuard:
    """Confines all file access to a single authorised root directory."""

    def __init__(self, root: Path) -> None:
        resolved = Path(root).expanduser()
        if not resolved.exists():
            raise SecurityError(f"authorised root does not exist: {resolved}")
        if not resolved.is_dir():
            raise SecurityError(f"authorised root is not a directory: {resolved}")
        self._root = resolved.resolve(strict=True)

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, candidate: str | os.PathLike[str], *, must_exist: bool = True) -> Path:
        """Resolve *candidate* inside the sandbox or raise :class:`SecurityError`.

        Symlinks are resolved first, so a link pointing outside the project is
        rejected even though the link itself lives inside it.
        """
        raw = os.fspath(candidate)
        if not raw:
            raise SecurityError("empty path")
        path = Path(raw)

        if path.is_absolute():
            # An absolute path is accepted only if it genuinely sits inside the
            # root. The check is made against the *resolved parent*, because
            # Path.relative_to is purely lexical: the root may itself be reached
            # through a symlink (/var -> /private/var on macOS, /tmp elsewhere),
            # in which case a string-prefix comparison would reject every
            # legitimate file. Anything outside is refused outright rather than
            # silently re-anchored onto the root.
            try:
                parent = path.parent.resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                raise SecurityError(f"cannot resolve path {raw!r}: {exc}") from exc
            if not (parent == self._root or self._root in parent.parents):
                raise SecurityError(
                    f"absolute path is outside the authorised project directory: {raw!r}"
                )
            candidate_path = path
        else:
            candidate_path = self._root / path
            # Normalise lexically and reject traversal *before* touching the
            # filesystem, so "../x" reports a containment violation rather than
            # a confusing "no such file" from the escaped location.
            normalised = Path(os.path.normpath(candidate_path))
            if not (normalised == self._root or self._root in normalised.parents):
                raise SecurityError(
                    f"path escapes the authorised project directory: {raw!r}"
                )

        try:
            resolved = candidate_path.resolve(strict=must_exist)
        except (OSError, RuntimeError) as exc:
            raise SecurityError(f"cannot resolve path {raw!r}: {exc}") from exc

        self._assert_inside(resolved, raw)
        return resolved

    def _assert_inside(self, resolved: Path, raw: str) -> None:
        if resolved == self._root or self._root in resolved.parents:
            return
        raise SecurityError(
            f"path escapes the authorised project directory: {raw!r} -> {resolved}"
        )

    def project_relative(self, path: Path) -> str:
        """Project-relative path **without** following a final symlink.

        Used by the scanner so that a symlink named ``alias.ts`` is reported as
        ``alias.ts`` rather than as the file it points at. Containment is still
        enforced by the caller resolving the path first.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        try:
            relative = candidate.relative_to(self._root)
        except ValueError as exc:
            raise SecurityError(f"path is not inside the project: {path}") from exc
        return relative.as_posix()

    def relative(self, path: Path) -> str:
        """Project-relative POSIX path, for reporting and schema fields."""
        resolved = path.resolve(strict=False)
        self._assert_inside(resolved, str(path))
        return resolved.relative_to(self._root).as_posix()

    def is_inside(self, candidate: str | os.PathLike[str]) -> bool:
        try:
            self.resolve(candidate, must_exist=False)
        except SecurityError:
            return False
        return True
