"""Model tests: coercion, dedupe keys, run bookkeeping (design doc section 6)."""

from __future__ import annotations

import pytest

from code_review_agent.errors import InputError
from code_review_agent.models import (
    Category,
    FindingSource,
    ReviewFinding,
    ReviewRun,
    RunStatus,
    Severity,
    TokenUsage,
    schema_errors_for_response,
)


class TestSeverity:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("critical", Severity.CRITICAL),
            ("CRIT", Severity.CRITICAL),
            ("blocker", Severity.CRITICAL),
            ("Major", Severity.HIGH),
            ("moderate", Severity.MEDIUM),
            ("minor", Severity.LOW),
            ("nit", Severity.INFO),
        ],
    )
    def test_coerces_aliases(self, raw, expected):
        assert Severity.coerce(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(InputError, match="unknown severity"):
            Severity.coerce("catastrophic")

    def test_none_raises(self):
        with pytest.raises(InputError):
            Severity.coerce(None)

    def test_rank_orders_by_urgency(self):
        assert Severity.CRITICAL.rank < Severity.INFO.rank


class TestCategory:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("bug", Category.BUG),
            ("correctness", Category.BUG),
            ("vulnerability", Category.SECURITY),
            ("perf", Category.PERFORMANCE),
            ("style", Category.MAINTAINABILITY),
            ("readability", Category.MAINTAINABILITY),
        ],
    )
    def test_coerces_aliases(self, raw, expected):
        assert Category.coerce(raw) == expected

    def test_unknown_raises(self):
        with pytest.raises(InputError, match="unknown category"):
            Category.coerce("vibes")


class TestReviewFinding:
    def test_construct_from_loose_types(self):
        finding = ReviewFinding(
            title="t",
            severity="HIGH",
            category="Bug",
            file="a.ts",
            start_line=3,
            end_line=4,
            evidence="x",
            description="d",
        )
        assert finding.severity is Severity.HIGH
        assert finding.category is Category.BUG

    def test_line_order_is_normalised(self):
        finding = ReviewFinding(
            title="t",
            severity="low",
            category="bug",
            file="a.ts",
            start_line=9,
            end_line=4,
            evidence="x",
            description="d",
        )
        assert (finding.start_line, finding.end_line) == (4, 9)

    def test_round_trip_dict(self):
        original = ReviewFinding(
            title="Null deref",
            severity="high",
            category="bug",
            file="src/a.ts",
            start_line=2,
            end_line=3,
            evidence="user.profile.name",
            description="d",
            recommendation="r",
            confidence=0.8,
            source=FindingSource.VALIDATED_LLM,
        )
        restored = ReviewFinding.from_dict(original.to_dict())
        assert restored.title == original.title
        assert restored.severity is original.severity
        assert restored.confidence == original.confidence

    def test_from_dict_defaults_end_line_to_start(self):
        finding = ReviewFinding.from_dict(
            {
                "title": "t",
                "severity": "low",
                "category": "bug",
                "file": "a.ts",
                "start_line": 7,
                "evidence": "e",
                "description": "d",
            }
        )
        assert finding.end_line == 7

    def test_dedupe_key_ignores_title_casing_and_spacing(self):
        left = ReviewFinding(
            title="Potential  NULL Access", severity="high", category="bug",
            file="a.ts", start_line=1, end_line=1, evidence="e", description="d",
        )
        right = ReviewFinding(
            title="potential null access", severity="low", category="bug",
            file="a.ts", start_line=1, end_line=1, evidence="other", description="d",
        )
        assert left.dedupe_key == right.dedupe_key

    def test_dedupe_key_distinguishes_locations(self):
        left = ReviewFinding(
            title="t", severity="high", category="bug",
            file="a.ts", start_line=1, end_line=1, evidence="e", description="d",
        )
        right = ReviewFinding(
            title="t", severity="high", category="bug",
            file="b.ts", start_line=1, end_line=1, evidence="e", description="d",
        )
        assert left.dedupe_key != right.dedupe_key


class TestResponseSchema:
    def test_valid_payload_passes(self):
        payload = {
            "findings": [
                {
                    "title": "t",
                    "severity": "high",
                    "category": "bug",
                    "file": "a.ts",
                    "start_line": 1,
                    "evidence": "code",
                    "description": "d",
                }
            ]
        }
        assert schema_errors_for_response(payload) == []

    def test_empty_findings_is_valid(self):
        assert schema_errors_for_response({"findings": []}) == []

    def test_missing_findings_key_fails(self):
        assert schema_errors_for_response({}) != []

    def test_bad_severity_fails(self):
        payload = {
            "findings": [
                {
                    "title": "t", "severity": "apocalyptic", "category": "bug",
                    "file": "a.ts", "start_line": 1, "evidence": "e", "description": "d",
                }
            ]
        }
        assert schema_errors_for_response(payload) != []

    def test_zero_line_fails(self):
        payload = {
            "findings": [
                {
                    "title": "t", "severity": "high", "category": "bug",
                    "file": "a.ts", "start_line": 0, "evidence": "e", "description": "d",
                }
            ]
        }
        assert schema_errors_for_response(payload) != []

    def test_confidence_out_of_range_fails(self):
        payload = {
            "findings": [
                {
                    "title": "t", "severity": "high", "category": "bug",
                    "file": "a.ts", "start_line": 1, "evidence": "e",
                    "description": "d", "confidence": 1.5,
                }
            ]
        }
        assert schema_errors_for_response(payload) != []

    def test_extra_properties_are_allowed(self):
        payload = {
            "findings": [
                {
                    "title": "t", "severity": "high", "category": "bug",
                    "file": "a.ts", "start_line": 1, "evidence": "e",
                    "description": "d", "rationale": "extra",
                }
            ],
            "summary": "extra top-level",
        }
        assert schema_errors_for_response(payload) == []


class TestReviewRun:
    def test_status_defaults_to_pending(self):
        run = ReviewRun(run_id="r", project_root="/tmp")
        assert run.status is RunStatus.PENDING
        assert run.duration_seconds is None

    def test_fatal_error_tracking(self):
        run = ReviewRun(run_id="r", project_root="/tmp")
        run.add_error("stage", "boom", fatal=True)
        assert run.has_fatal_error is True

    def test_non_fatal_error_does_not_mark_fatal(self):
        run = ReviewRun(run_id="r", project_root="/tmp")
        run.add_error("stage", "warning")
        assert run.has_fatal_error is False

    def test_to_dict_is_json_serialisable(self):
        import json

        run = ReviewRun(run_id="r", project_root="/tmp")
        run.add_error("s", "m")
        json.dumps(run.to_dict())


class TestTokenUsage:
    def test_accumulates(self):
        usage = TokenUsage()
        usage.add(10, 5)
        usage.add(1, 2)
        assert usage.prompt_tokens == 11
        assert usage.completion_tokens == 7
        assert usage.total_tokens == 18
        assert usage.requests == 2

    def test_merge(self):
        left = TokenUsage(prompt_tokens=1, completion_tokens=2, requests=1)
        left.merge(TokenUsage(prompt_tokens=3, completion_tokens=4, requests=1))
        assert left.total_tokens == 10
        assert left.requests == 2
