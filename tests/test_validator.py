"""FindingValidator tests (design doc 5.2, 6.2, 11).

These tests encode the acceptance criteria "every formal finding has a file and
line" and "invalid paths, illegal line numbers and duplicates are detected".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_agent.agent.validator import FindingValidator
from code_review_agent.models import FindingSource, ReviewFinding
from code_review_agent.security import PathGuard
from code_review_agent.tools.reader import CodeReader
from code_review_agent.tools.scanner import ScannedFile


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "useUser.ts").write_text(
        "export function useUser(user: any) {\n"   # line 1
        "  const name = user.profile.name;\n"      # line 2
        "  return { name };\n"                     # line 3
        "}\n",                                     # line 4
        encoding="utf-8",
    )
    (root / "src" / "other.ts").write_text("export const x = 1;\n", encoding="utf-8")
    return root


@pytest.fixture
def validator(project: Path, base_config) -> FindingValidator:
    guard = PathGuard(project)
    return FindingValidator(guard, CodeReader(guard), base_config)


@pytest.fixture
def files(project: Path) -> list[ScannedFile]:
    return [
        ScannedFile(
            path=project / "src" / "useUser.ts",
            relative_path="src/useUser.ts",
            size_bytes=100,
            line_count=4,
        ),
        ScannedFile(
            path=project / "src" / "other.ts",
            relative_path="src/other.ts",
            size_bytes=20,
            line_count=1,
        ),
    ]


def finding(**overrides) -> ReviewFinding:
    payload = {
        "title": "Potential null dereference",
        "severity": "high",
        "category": "bug",
        "file": "src/useUser.ts",
        "start_line": 2,
        "end_line": 2,
        "evidence": "const name = user.profile.name;",
        "description": "user.profile may be undefined.",
        "source": FindingSource.VALIDATED_LLM,
    }
    payload.update(overrides)
    return ReviewFinding(**payload)


class TestAcceptance:
    def test_valid_finding_is_accepted(self, validator, files):
        outcome = validator.validate([finding()], files)
        assert len(outcome.valid) == 1
        assert outcome.rejected == []

    def test_stats_summarise_the_batch(self, validator, files):
        outcome = validator.validate([finding()], files)
        assert outcome.stats["accepted"] == 1
        assert outcome.stats["considered"] == 1

    def test_empty_input_is_fine(self, validator, files):
        outcome = validator.validate([], files)
        assert outcome.valid == []
        assert outcome.considered == 0


class TestPathValidation:
    def test_missing_file_field_is_rejected(self, validator, files):
        outcome = validator.validate([finding(file="")], files)
        assert outcome.rejected[0].reason == "missing-file"

    def test_unknown_file_is_rejected(self, validator, files):
        outcome = validator.validate([finding(file="src/ghost.ts")], files)
        assert outcome.rejected[0].reason == "unknown-file"

    def test_filtered_out_file_is_rejected(self, validator, files):
        # node_modules is excluded by the scanner, so it is not a reviewed file.
        outcome = validator.validate([finding(file="node_modules/x.js")], files)
        assert outcome.rejected[0].reason == "unknown-file"

    def test_path_escape_is_rejected(self, validator, files):
        # Traversal out of the project is an escape, not a typo.
        outcome = validator.validate([finding(file="../../etc/passwd")], files)
        assert outcome.rejected[0].reason == "path-escape"

    def test_symlink_escaping_project_is_reported_as_escape(self, project, validator, files, tmp_path):
        # A link that lives inside the project but points outside is an escape.
        outside = tmp_path / "outside.ts"
        outside.write_text("export const o = 1;\n", encoding="utf-8")
        (project / "src" / "link.ts").symlink_to(outside)
        outcome = validator.validate([finding(file="src/link.ts")], files)
        assert outcome.rejected[0].reason == "path-escape"

    def test_absolute_path_outside_project_is_rejected(self, validator, files):
        outcome = validator.validate([finding(file="/etc/hosts")], files)
        assert outcome.rejected[0].reason == "path-escape"


class TestLineValidation:
    def test_zero_line_is_rejected(self, validator, files):
        outcome = validator.validate([finding(start_line=0, end_line=0)], files)
        assert outcome.rejected[0].reason == "invalid-line"

    def test_negative_line_is_rejected(self, validator, files):
        outcome = validator.validate([finding(start_line=-1, end_line=-1)], files)
        assert outcome.rejected[0].reason == "invalid-line"

    def test_start_beyond_eof_is_rejected(self, validator, files):
        outcome = validator.validate([finding(start_line=99, end_line=99)], files)
        assert outcome.rejected[0].reason == "line-out-of-range"

    def test_end_beyond_eof_is_rejected(self, validator, files):
        outcome = validator.validate([finding(start_line=2, end_line=99)], files)
        assert outcome.rejected[0].reason == "line-out-of-range"

    def test_last_valid_line_is_accepted(self, validator, files):
        outcome = validator.validate(
            [finding(start_line=4, end_line=4, evidence="}")], files
        )
        assert len(outcome.valid) == 1

    def test_rejection_detail_is_informative(self, validator, files):
        outcome = validator.validate([finding(start_line=99, end_line=99)], files)
        assert "99" in outcome.rejected[0].detail
        assert "src/useUser.ts:99" in outcome.rejected[0].describe()


class TestEvidenceValidation:
    def test_fabricated_evidence_is_rejected(self, validator, files):
        # A plausible-looking snippet that is not present at the cited lines.
        outcome = validator.validate([finding(evidence="const name = user.account.name;")], files)
        assert outcome.rejected[0].reason == "evidence-mismatch"

    def test_evidence_from_wrong_line_is_rejected(self, validator, files):
        outcome = validator.validate([finding(start_line=3, end_line=3)], files)
        assert outcome.rejected[0].reason == "evidence-mismatch"

    def test_empty_evidence_is_rejected_for_model_findings(self, validator, files):
        outcome = validator.validate([finding(evidence="")], files)
        assert outcome.rejected[0].reason == "missing-evidence"

    def test_empty_evidence_allowed_when_not_required(self, project, files, base_config):
        guard = PathGuard(project)
        lenient = FindingValidator(guard, CodeReader(guard), base_config, require_evidence=False)
        outcome = lenient.validate([finding(evidence="")], files)
        assert len(outcome.valid) == 1

    def test_static_analysis_finding_needs_no_evidence(self, validator, files):
        outcome = validator.validate(
            [finding(evidence="", source=FindingSource.STATIC_ANALYSIS)], files
        )
        assert len(outcome.valid) == 1

    def test_multiline_evidence_is_accepted(self, validator, files):
        outcome = validator.validate(
            [
                finding(
                    start_line=1,
                    end_line=4,
                    evidence="export function useUser(user: any) {\n  const name = user.profile.name;",
                )
            ],
            files,
        )
        assert len(outcome.valid) == 1

    def test_reindented_evidence_is_accepted(self, validator, files):
        outcome = validator.validate(
            [finding(evidence="      const name = user.profile.name;      ")], files
        )
        assert len(outcome.valid) == 1

    def test_secret_in_evidence_is_redacted(self, validator, files):
        # The finding is rejected (a redacted secret is not in the source), and
        # crucially the stored evidence no longer contains the secret.
        item = finding(evidence="const apiKey = 'sk-abcdefghijklmnopqrstuvwx';")
        validator.validate([item], files)
        assert "[REDACTED]" in item.evidence
        assert "sk-abcdefghijklmnopqrstuvwx" not in item.evidence


class TestDeduplication:
    def test_exact_duplicate_is_dropped(self, validator, files):
        outcome = validator.validate([finding(), finding()], files)
        assert len(outcome.valid) == 1
        assert outcome.duplicates[0].reason == "duplicate"

    def test_same_location_different_casing_is_duplicate(self, validator, files):
        outcome = validator.validate(
            [finding(title="Null deref"), finding(title="NULL  DEREF")], files
        )
        assert len(outcome.valid) == 1
        assert outcome.duplicates

    def test_near_duplicate_wording_is_dropped(self, validator, files):
        outcome = validator.validate(
            [
                finding(title="Potential null dereference on user profile"),
                finding(title="Potential null dereference in user profile access"),
            ],
            files,
        )
        assert len(outcome.valid) == 1
        assert outcome.duplicates[0].reason == "near-duplicate"

    def test_distinct_findings_at_different_lines_are_kept(self, validator, files):
        outcome = validator.validate(
            [
                finding(title="Null dereference", start_line=2, end_line=2),
                finding(title="Returned object is unused", start_line=3, end_line=3,
                        evidence="return { name };"),
            ],
            files,
        )
        assert len(outcome.valid) == 2

    def test_genuinely_different_issues_on_same_line_are_kept(self, validator, files):
        outcome = validator.validate(
            [
                finding(title="Null dereference on profile"),
                finding(title="Missing input sanitisation", start_line=2, end_line=2),
            ],
            files,
        )
        assert len(outcome.valid) == 2

    def test_duplicate_across_files_is_kept(self, validator, files):
        outcome = validator.validate(
            [
                finding(file="src/useUser.ts", start_line=2, end_line=2),
                finding(
                    file="src/other.ts",
                    start_line=1,
                    end_line=1,
                    evidence="export const x = 1;",
                ),
            ],
            files,
        )
        assert len(outcome.valid) == 2

    def test_threshold_is_configurable(self, project, files, base_config):
        guard = PathGuard(project)
        strict = FindingValidator(
            guard, CodeReader(guard), base_config, near_duplicate_threshold=0.99
        )
        outcome = strict.validate(
            [
                finding(title="Null dereference of user profile"),
                finding(title="Null dereference of user profile object"),
            ],
            files,
        )
        assert len(outcome.valid) == 2


class TestMixedBatches:
    def test_only_valid_findings_survive(self, validator, files):
        outcome = validator.validate(
            [
                finding(title="Real bug"),
                finding(title="Ghost file", file="src/ghost.ts"),
                finding(title="Bad line", start_line=99, end_line=99),
                finding(title="Made up", evidence="this text is not in the file"),
                finding(title="Real bug"),  # duplicate of the first
            ],
            files,
        )
        assert [f.title for f in outcome.valid] == ["Real bug"]
        assert len(outcome.rejected) == 3
        assert len(outcome.duplicates) == 1
        assert outcome.considered == 5

    def test_rejection_reasons_are_all_specific(self, validator, files):
        outcome = validator.validate(
            [
                finding(title="Ghost", file="src/ghost.ts"),
                finding(title="Line", start_line=99, end_line=99),
                finding(title="Evidence", evidence="nope"),
            ],
            files,
        )
        reasons = {item.reason for item in outcome.rejected}
        assert reasons == {"unknown-file", "line-out-of-range", "evidence-mismatch"}

    def test_path_escape_is_a_distinct_reason(self, validator, files):
        outcome = validator.validate([finding(file="../outside.ts")], files)
        assert outcome.rejected[0].reason == "path-escape"
