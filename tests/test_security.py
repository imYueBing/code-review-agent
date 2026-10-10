"""Security boundary tests (design doc section 8)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from code_review_agent.errors import SecurityError
from code_review_agent.security import (
    FileFilter,
    PathGuard,
    looks_sensitive_key,
    redact_secrets,
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.ts").write_text("export const a = 1;\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("module.exports=1\n", encoding="utf-8")
    (tmp_path / "outside.ts").write_text("outside\n", encoding="utf-8")
    return root


class TestPathGuard:
    def test_resolves_relative_path_inside_root(self, project: Path):
        guard = PathGuard(project)
        assert guard.resolve("src/a.ts").name == "a.ts"

    def test_relative_returns_posix_path(self, project: Path):
        guard = PathGuard(project)
        assert guard.relative(project / "src" / "a.ts") == "src/a.ts"

    def test_rejects_parent_traversal(self, project: Path):
        guard = PathGuard(project)
        with pytest.raises(SecurityError, match="escapes"):
            guard.resolve("../outside.ts")

    def test_rejects_deep_traversal(self, project: Path):
        guard = PathGuard(project)
        with pytest.raises(SecurityError, match="escapes"):
            guard.resolve("src/../../outside.ts")

    def test_rejects_absolute_path_outside_root(self, project: Path):
        guard = PathGuard(project)
        with pytest.raises(SecurityError, match="outside the authorised"):
            guard.resolve("/etc/passwd")

    def test_absolute_path_inside_root_is_accepted(self, project: Path):
        guard = PathGuard(project)
        resolved = guard.resolve(str(project / "src" / "a.ts"))
        assert resolved == (project / "src" / "a.ts").resolve()

    def test_rejects_symlink_pointing_outside_root(self, project: Path, tmp_path: Path):
        link = project / "src" / "escape.ts"
        os.symlink(tmp_path / "outside.ts", link)
        guard = PathGuard(project)
        with pytest.raises(SecurityError, match="escapes"):
            guard.resolve("src/escape.ts")

    def test_missing_path_raises_security_error(self, project: Path):
        guard = PathGuard(project)
        with pytest.raises(SecurityError):
            guard.resolve("src/nope.ts")

    def test_nonexistent_root_rejected(self, tmp_path: Path):
        with pytest.raises(SecurityError):
            PathGuard(tmp_path / "missing")

    def test_file_as_root_rejected(self, project: Path):
        with pytest.raises(SecurityError):
            PathGuard(project / "src" / "a.ts")

    def test_empty_path_rejected(self, project: Path):
        guard = PathGuard(project)
        with pytest.raises(SecurityError):
            guard.resolve("")

    def test_is_inside_helper(self, project: Path):
        guard = PathGuard(project)
        assert guard.is_inside("src/a.ts") is True
        assert guard.is_inside("../outside.ts") is False


class TestFileFilter:
    def test_excludes_sensitive_env_file(self):
        flt = FileFilter()
        assert flt.rejection_reason(Path(".env")) == "sensitive"
        assert flt.rejection_reason(Path("config/.env.production")) == "sensitive"

    def test_excludes_private_keys(self):
        flt = FileFilter()
        assert flt.rejection_reason(Path("certs/server.pem")) == "sensitive"
        assert flt.rejection_reason(Path("id_rsa")) == "sensitive"

    def test_excludes_unsupported_extension(self):
        flt = FileFilter()
        assert flt.rejection_reason(Path("README.md")) == "unsupported-extension"

    def test_excludes_generated_declaration_files(self):
        flt = FileFilter()
        assert flt.rejection_reason(Path("src/types.d.ts")) == "excluded-pattern"

    def test_oversize_file_rejected(self):
        flt = FileFilter(max_file_bytes=10)
        assert flt.rejection_reason(Path("src/big.ts"), size=11) == "too-large"

    def test_accepts_ordinary_source(self):
        flt = FileFilter()
        assert flt.rejection_reason(Path("src/a.tsx"), size=100) is None

    def test_excluded_directory_names(self):
        flt = FileFilter()
        for name in ("node_modules", ".git", "dist", "build", "__pycache__"):
            assert flt.is_excluded_dir(name) is True
        assert flt.is_excluded_dir("src") is False

    def test_secrets_directory_is_sensitive(self):
        flt = FileFilter()
        assert flt.is_sensitive(Path("secrets/prod.json")) is True


class TestRedaction:
    @pytest.mark.parametrize(
        "text",
        [
            "key = sk-abcdefghijklmnopqrstuvwx",
            "token=ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AWS AKIAIOSFODNN7EXAMPLE here",
            'api_key: "abcdefghijklmnop"',
            "password: 'hunter2hunter2'",
        ],
    )
    def test_redacts_credential_looking_values(self, text: str):
        redacted = redact_secrets(text)
        assert "[REDACTED]" in redacted
        assert "sk-abcdefghijklmnopqrstuvwx" not in redacted
        assert "hunter2hunter2" not in redacted
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted

    def test_leaves_ordinary_code_untouched(self):
        text = "const total = price * quantity;"
        assert redact_secrets(text) == text

    def test_empty_string_is_safe(self):
        assert redact_secrets("") == ""

    def test_sensitive_key_detection(self):
        assert looks_sensitive_key("apiKey") is True
        assert looks_sensitive_key("access_token") is True
        assert looks_sensitive_key("username") is False
