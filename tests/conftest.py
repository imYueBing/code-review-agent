"""Shared test fixtures and helpers.

Everything here is offline: no network, no real model calls, no cost. That is a
hard requirement from design doc section 9.2.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_review_agent.config.settings import AgentConfig  # noqa: E402
from code_review_agent.llm.client import LLMResponse  # noqa: E402
from code_review_agent.models import FINDINGS_RESPONSE_SCHEMA, TokenUsage  # noqa: E402


@pytest.fixture
def base_config() -> AgentConfig:
    """Config with the fake provider and no external tooling required."""
    config = AgentConfig()
    return replace(
        config,
        llm=replace(config.llm, provider="fake", model="fake-model"),
        static_analysis=replace(config.static_analysis, eslint_command="off", enabled=False),
        review=replace(config.review, enable_tool_calls=False),
    )


@pytest.fixture
def demo_project(tmp_path: Path) -> Path:
    """A small React-ish project with a deliberate, verifiable null-deref bug."""
    root = tmp_path / "demo"
    (root / "src" / "hooks").mkdir(parents=True)
    (root / "src" / "components").mkdir(parents=True)
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / ".git").mkdir(parents=True)

    (root / "src" / "hooks" / "useUser.ts").write_text(
        "export function useUser(user: any) {\n"
        "  const name = user.profile.name;\n"
        "  return { name };\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "src" / "components" / "Button.tsx").write_text(
        "import React from 'react';\n"
        "\n"
        "export function Button({ onClick, label }) {\n"
        "  if (label = null) {\n"
        "    return null;\n"
        "  }\n"
        "  return <button onClick={onClick}>{label}</button>;\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "node_modules" / "left-pad" / "index.js").write_text(
        "module.exports = function () { return 'should never be reviewed'; };\n",
        encoding="utf-8",
    )
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "README.md").write_text("# Demo\n\nNot a source file.\n", encoding="utf-8")
    (root / ".env").write_text("API_KEY=sk-supersecretvalue1234567890\n", encoding="utf-8")
    return root


def findings_payload(*findings: dict) -> str:
    """Serialise a model response containing *findings*."""
    return json.dumps({"findings": list(findings)})


def make_finding_json(
    *,
    title: str = "Potential null dereference",
    severity: str = "high",
    category: str = "bug",
    file: str = "src/hooks/useUser.ts",
    start_line: int = 2,
    end_line: int | None = None,
    evidence: str = "const name = user.profile.name;",
    description: str = "user.profile may be undefined.",
    recommendation: str = "Guard against a missing profile.",
    confidence: float | None = 0.9,
) -> dict:
    payload = {
        "title": title,
        "severity": severity,
        "category": category,
        "file": file,
        "start_line": start_line,
        "end_line": end_line if end_line is not None else start_line,
        "evidence": evidence,
        "description": description,
        "recommendation": recommendation,
    }
    if confidence is not None:
        payload["confidence"] = confidence
    return payload


class ScriptedClient:
    """Fake client whose reply depends on the request purpose.

    ``rules`` is an *ordered* sequence of ``(needle, raw_response)`` pairs; the
    first needle contained in ``LLMRequest.purpose`` wins, and ``default`` is
    used when nothing matches. Ordering is explicit because a keyword such as
    "review" is a substring of both ``review:a.ts`` and ``review_round2:a.ts``,
    and silently matching the wrong rule would make tests lie.

    Raw text (not parsed JSON) is returned deliberately, so tests can drive the
    schema-repair path.
    """

    def __init__(
        self,
        rules: dict[str, str] | list[tuple[str, str]] | None = None,
        default: str = '{"findings": []}',
        *,
        guard=None,
        usage_per_request: tuple[int, int] = (100, 50),
    ):
        items = list(rules.items()) if isinstance(rules, dict) else list(rules or [])
        self._rules = items
        self._default = default
        self._guard = guard
        self._usage = usage_per_request
        self.prompts: list[str] = []
        self.purposes: list[str] = []

    def _text_for(self, purpose: str) -> str:
        for needle, response in self._rules:
            if needle in purpose:
                return response
        return self._default

    def complete(self, request):
        if self._guard is not None:
            self._guard.check()
        self.purposes.append(request.purpose)
        self.prompts.append("\n".join(m.content for m in request.messages))
        if self._guard is not None:
            self._guard.record(*self._usage)
        return LLMResponse(
            text=self._text_for(request.purpose),
            usage=TokenUsage(
                prompt_tokens=self._usage[0], completion_tokens=self._usage[1], requests=1
            ),
            model="scripted",
            finish_reason="stop",
        )

    def complete_json(self, request):
        from code_review_agent.llm.parsing import extract_json_object

        return extract_json_object(self.complete(request).text)


@pytest.fixture
def findings_schema() -> dict:
    return FINDINGS_RESPONSE_SCHEMA


def routed(*rules: tuple[str, str], default: str = '{"findings": []}') -> ScriptedClient:
    """Convenience constructor: later rules do not shadow earlier ones."""
    return ScriptedClient(list(rules), default=default)
