"""Scanner and file-filtering tests (design doc 2.1, 5.2)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from code_review_agent.security import PathGuard
from code_review_agent.tools.scanner import ProjectScanner


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "dist").mkdir()
    (root / ".git").mkdir()
    (root / "src" / "a.ts").write_text("export const a = 1;\n", encoding="utf-8")
    (root / "src" / "b.jsx").write_text("export default () => null;\n", encoding="utf-8")
    (root / "src" / "styles.css").write_text("body {}\n", encoding="utf-8")
    (root / "src" / "types.d.ts").write_text("declare const x: number;\n", encoding="utf-8")
    (root / "src" / "bundle.min.js").write_text("var a=1;\n", encoding="utf-8")
    (root / "node_modules" / "dep" / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
    (root / "dist" / "out.js").write_text("var x = 1;\n", encoding="utf-8")
    (root / ".git" / "hook.js").write_text("var y = 2;\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    (root / "package-lock.json").write_text("{}\n", encoding="utf-8")
    (root / "README.md").write_text("# hi\n", encoding="utf-8")
    return root


def scanner_for(root: Path, config) -> ProjectScanner:
    return ProjectScanner(PathGuard(root), config)


class TestScanSelection:
    def test_finds_supported_source_files(self, project: Path, base_config):
        report = scanner_for(project, base_config).scan()
        found = {f.relative_path for f in report.files}
        assert found == {"src/a.ts", "src/b.jsx"}

    def test_excludes_dependency_and_build_directories(self, project: Path, base_config):
        report = scanner_for(project, base_config).scan()
        joined = " ".join(f.relative_path for f in report.files)
        assert "node_modules" not in joined
        assert "dist" not in joined
        assert ".git" not in joined

    def test_excludes_sensitive_files(self, project: Path, base_config):
        report = scanner_for(project, base_config).scan()
        assert not any(f.relative_path == ".env" for f in report.files)
        assert report.skipped.get("sensitive", 0) >= 1

    def test_excludes_declaration_and_minified_files(self, project: Path, base_config):
        found = {f.relative_path for f in scanner_for(project, base_config).scan().files}
        assert "src/types.d.ts" not in found
        assert "src/bundle.min.js" not in found

    def test_excludes_lockfiles_and_docs(self, project: Path, base_config):
        found = {f.relative_path for f in scanner_for(project, base_config).scan().files}
        assert "package-lock.json" not in found
        assert "README.md" not in found

    def test_skip_reasons_are_recorded(self, project: Path, base_config):
        report = scanner_for(project, base_config).scan()
        assert report.skipped
        assert "excluded-directory" in report.skipped
        assert "unsupported-extension" in report.skipped

    def test_relative_paths_use_posix_separators(self, project: Path, base_config):
        for scanned in scanner_for(project, base_config).scan().files:
            assert "\\" not in scanned.relative_path

    def test_line_count_is_accurate(self, project: Path, base_config):
        report = scanner_for(project, base_config).scan()
        by_name = {f.relative_path: f for f in report.files}
        assert by_name["src/a.ts"].line_count == 1
        assert by_name["src/b.jsx"].line_count == 1

    def test_language_detection(self, project: Path, base_config):
        by_name = {f.relative_path: f for f in scanner_for(project, base_config).scan().files}
        assert by_name["src/a.ts"].language == "typescript"
        assert by_name["src/b.jsx"].language == "jsx"


class TestScanLimits:
    def test_max_files_truncates_and_flags(self, project: Path, base_config):
        config = replace(base_config, scanner=replace(base_config.scanner, max_files=1))
        report = scanner_for(project, config).scan()
        assert len(report.files) == 1
        assert report.truncated is True
        assert report.skipped.get("over-file-limit", 0) >= 1

    def test_oversize_file_skipped(self, project: Path, base_config):
        config = replace(base_config, budget=replace(base_config.budget, max_file_bytes=5))
        report = scanner_for(project, config).scan()
        assert report.files == []
        assert report.skipped.get("too-large", 0) >= 1

    def test_smaller_files_are_preferred(self, project: Path, base_config):
        (project / "src" / "tiny.ts").write_text("x\n", encoding="utf-8")
        report = scanner_for(project, base_config).scan()
        assert report.files[0].relative_path == "src/tiny.ts"

    def test_custom_extensions(self, project: Path, base_config):
        config = replace(
            base_config, scanner=replace(base_config.scanner, extensions=(".css",))
        )
        found = {f.relative_path for f in scanner_for(project, config).scan().files}
        assert found == {"src/styles.css"}


class TestSymlinks:
    def test_symlinked_file_is_skipped_by_default(self, project: Path, base_config, tmp_path):
        outside = tmp_path / "outside.ts"
        outside.write_text("export const z = 1;\n", encoding="utf-8")
        os.symlink(outside, project / "src" / "link.ts")
        report = scanner_for(project, base_config).scan()
        assert not any(f.relative_path == "src/link.ts" for f in report.files)
        assert report.skipped.get("symlink", 0) >= 1

    def test_symlinked_file_inside_root_is_allowed_when_following(self, project: Path, base_config):
        os.symlink(project / "src" / "a.ts", project / "src" / "alias.ts")
        config = replace(base_config, scanner=replace(base_config.scanner, follow_symlinks=True))
        found = {f.relative_path for f in scanner_for(project, config).scan().files}
        assert "src/alias.ts" in found

    def test_symlink_escaping_root_is_refused_even_when_following(
        self, project: Path, base_config, tmp_path
    ):
        outside = tmp_path / "secret.ts"
        outside.write_text("export const s = 1;\n", encoding="utf-8")
        os.symlink(outside, project / "src" / "escape.ts")
        config = replace(base_config, scanner=replace(base_config.scanner, follow_symlinks=True))
        report = scanner_for(project, config).scan()
        assert not any(f.relative_path == "src/escape.ts" for f in report.files)
        assert report.skipped.get("symlink-outside-root", 0) >= 1


class TestValidation:
    def test_valid_root_passes(self, project: Path, base_config):
        scanner_for(project, base_config).validate_root()

    def test_empty_directory_is_not_an_error(self, tmp_path: Path, base_config):
        empty = tmp_path / "empty"
        empty.mkdir()
        report = scanner_for(empty, base_config).scan()
        assert report.files == []
