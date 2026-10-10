"""CodeReader and line-mapping tests (design doc 4.2, 9.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_agent.errors import ToolError
from code_review_agent.security import PathGuard
from code_review_agent.tools.reader import CodeReader, SourceFile


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "sample.ts").write_text(
        "// line one\n"
        "const a = 1;\n"
        "\n"
        "function f() {\n"
        "  return a;\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "src" / "empty.ts").write_text("", encoding="utf-8")
    (root / "binary.ts").write_bytes(b"\xff\xfe\x00\x01not utf8")
    return root


@pytest.fixture
def reader(project: Path) -> CodeReader:
    return CodeReader(PathGuard(project))


class TestSourceFile:
    def test_line_count_excludes_trailing_newline(self):
        source = SourceFile("a.ts", "one\ntwo\nthree\n")
        assert source.line_count == 3

    def test_lines_are_one_based(self):
        source = SourceFile("a.ts", "first\nsecond\n")
        assert source.lines()[0].number == 1
        assert source.lines()[0].text == "first"
        assert source.lines()[1].number == 2

    def test_line_lookup(self):
        source = SourceFile("a.ts", "alpha\nbeta\n")
        assert source.line(1) == "alpha"
        assert source.line(2) == "beta"
        assert source.line(0) is None
        assert source.line(3) is None

    def test_is_valid_line(self):
        source = SourceFile("a.ts", "alpha\nbeta\n")
        assert source.is_valid_line(1) is True
        assert source.is_valid_line(2) is True
        assert source.is_valid_line(3) is False
        assert source.is_valid_line(0) is False

    def test_numbered_text_is_aligned(self):
        source = SourceFile("a.ts", "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n")
        rendered = source.numbered_text()
        assert " 1 | a" in rendered
        assert "10 | j" in rendered

    def test_numbered_text_range(self):
        source = SourceFile("a.ts", "a\nb\nc\n")
        assert source.numbered_text(start=2, end=3) == "2 | b\n3 | c"

    def test_excerpt_adds_padding(self):
        source = SourceFile("a.ts", "1\n2\n3\n4\n5\n")
        assert source.excerpt(3, 3, padding=1) == "2 | 2\n3 | 3\n4 | 4"

    def test_empty_file_has_zero_lines(self):
        assert SourceFile("a.ts", "").line_count == 0

    def test_numbered_text_on_empty_file_returns_empty(self):
        assert SourceFile("a.ts", "").numbered_text() == ""


class TestCodeReader:
    def test_reads_file_content(self, reader: CodeReader):
        source = reader.read("src/sample.ts")
        assert source.line_count == 6
        assert source.line(2) == "const a = 1;"

    def test_relative_path_is_posix(self, reader: CodeReader):
        assert reader.read("src/sample.ts").relative_path == "src/sample.ts"

    def test_caches_repeated_reads(self, reader: CodeReader):
        assert reader.read("src/sample.ts") is reader.read("src/sample.ts")

    def test_escape_attempt_raises_tool_error(self, reader: CodeReader):
        with pytest.raises(ToolError, match="outside the authorised|escapes"):
            reader.read("../../../etc/passwd")

    def test_missing_file_raises(self, reader: CodeReader):
        with pytest.raises(ToolError):
            reader.read("src/nope.ts")

    def test_directory_raises(self, reader: CodeReader):
        with pytest.raises(ToolError, match="not a file"):
            reader.read("src")

    def test_read_safe_returns_none_instead_of_raising(self, reader: CodeReader):
        assert reader.read_safe("src/nope.ts") is None
        assert reader.read_safe("src/sample.ts") is not None

    def test_invalid_utf8_is_replaced_not_fatal(self, reader: CodeReader):
        source = reader.read_safe("binary.ts")
        assert source is not None

    def test_truncation_marks_source_and_keeps_whole_lines(self, project: Path):
        reader = CodeReader(PathGuard(project), max_bytes=20)
        source = reader.read("src/sample.ts")
        assert source.truncated is True
        # A partial final line would shift what the model believes is on it.
        for line in source.lines():
            assert not line.text.endswith("\x00")


class TestEvidenceMatching:
    def test_exact_evidence_matches(self, reader: CodeReader):
        reader.read("src/sample.ts")
        assert reader.evidence_matches("src/sample.ts", 2, 2, "const a = 1;") is True

    def test_whitespace_differences_are_tolerated(self, reader: CodeReader):
        reader.read("src/sample.ts")
        assert reader.evidence_matches("src/sample.ts", 2, 2, "  const   a = 1;  ") is True

    def test_multi_line_evidence_matches(self, reader: CodeReader):
        reader.read("src/sample.ts")
        assert reader.evidence_matches("src/sample.ts", 4, 6, "function f() {\n  return a;\n}") is True

    def test_wrong_line_is_rejected(self, reader: CodeReader):
        reader.read("src/sample.ts")
        assert reader.evidence_matches("src/sample.ts", 1, 1, "const a = 1;") is False

    def test_fabricated_evidence_is_rejected(self, reader: CodeReader):
        reader.read("src/sample.ts")
        assert reader.evidence_matches("src/sample.ts", 2, 2, "user.profile.name") is False

    def test_empty_evidence_is_rejected(self, reader: CodeReader):
        reader.read("src/sample.ts")
        assert reader.evidence_matches("src/sample.ts", 2, 2, "   ") is False

    def test_out_of_range_line_is_rejected(self, reader: CodeReader):
        reader.read("src/sample.ts")
        assert reader.evidence_matches("src/sample.ts", 99, 99, "const a = 1;") is False

    def test_unread_file_is_rejected(self, reader: CodeReader):
        # Nothing cached, so no evidence can be confirmed.
        assert reader.evidence_matches("src/sample.ts", 2, 2, "const a = 1;") is False

    def test_secret_in_evidence_is_redacted_before_matching(self, reader: CodeReader):
        reader.read("src/sample.ts")
        # A redacted secret will not be found, which is the safe direction: the
        # finding is discarded rather than the secret being echoed.
        assert reader.evidence_matches("src/sample.ts", 2, 2, "sk-abcdefghijklmnopqrst") is False
