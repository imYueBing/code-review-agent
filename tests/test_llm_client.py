"""LLM client tests: budget guard, retries, schema repair. All offline."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from code_review_agent.config.settings import AgentConfig
from code_review_agent.errors import BudgetExceededError, LLMRequestError, LLMResponseFormatError
from code_review_agent.llm.client import (
    BudgetGuard,
    FakeLLMClient,
    LLMRequest,
    LLMResponse,
    _RetryingClient,
    build_client,
)
from code_review_agent.models import TokenUsage


def request_for(purpose: str = "review") -> LLMRequest:
    return LLMRequest(messages=[], schema={"type": "object", "required": ["findings"]}, purpose=purpose)


class ScriptedRetryClient(_RetryingClient):
    """Transport stub: returns or raises pre-scripted outcomes, no network."""

    def __init__(self, config, outcomes):
        super().__init__(config)
        self._outcomes = list(outcomes)
        self._last = None
        self.calls = 0

    def _send(self, request):
        self.calls += 1
        if self._outcomes:
            self._last = self._outcomes.pop(0)
        outcome = self._last
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response(text: str, prompt: int = 10, completion: int = 5) -> LLMResponse:
    return LLMResponse(
        text=text,
        usage=TokenUsage(prompt_tokens=prompt, completion_tokens=completion, requests=1),
    )


class TestFakeClient:
    def test_returns_scripted_responses_in_order(self):
        client = FakeLLMClient(['{"a": 1}', '{"a": 2}'])
        assert client.complete(request_for()).text == '{"a": 1}'
        assert client.complete(request_for()).text == '{"a": 2}'

    def test_falls_back_to_default(self):
        client = FakeLLMClient(['{"a": 1}'], default='{"findings": []}')
        client.complete(request_for())
        assert client.complete(request_for()).text == '{"findings": []}'

    def test_records_requests(self):
        client = FakeLLMClient()
        client.complete(request_for("first"))
        client.complete(request_for("second"))
        assert [r.purpose for r in client.requests] == ["first", "second"]

    def test_responder_callable(self):
        client = FakeLLMClient(responder=lambda req: json.dumps({"purpose": req.purpose}))
        assert json.loads(client.complete(request_for("x")).text)["purpose"] == "x"

    def test_from_fixture_list(self, tmp_path):
        path = tmp_path / "fixture.json"
        path.write_text(json.dumps(['{"a": 1}']), encoding="utf-8")
        assert FakeLLMClient.from_fixture(str(path)).complete(request_for()).text == '{"a": 1}'

    def test_from_fixture_object_wrapper(self, tmp_path):
        path = tmp_path / "fixture.json"
        path.write_text(json.dumps({"responses": [{"findings": []}]}), encoding="utf-8")
        assert json.loads(FakeLLMClient.from_fixture(str(path)).complete(request_for()).text) == {
            "findings": []
        }

    def test_from_fixture_rejects_non_list(self, tmp_path):
        path = tmp_path / "fixture.json"
        path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
        with pytest.raises(LLMResponseFormatError):
            FakeLLMClient.from_fixture(str(path))


class TestBudgetGuard:
    def test_allows_requests_under_limit(self, base_config):
        config = replace(base_config, budget=replace(base_config.budget, max_requests=2))
        guard = BudgetGuard(config)
        guard.check()
        guard.record(10, 10)
        guard.check()
        assert guard.exhausted is False

    def test_stops_at_request_limit(self, base_config):
        config = replace(base_config, budget=replace(base_config.budget, max_requests=1))
        guard = BudgetGuard(config)
        guard.record(1, 1)
        with pytest.raises(BudgetExceededError, match="request budget"):
            guard.check()
        assert guard.exhausted is True
        assert "1/1" in guard.stopped_reason

    def test_stops_at_token_limit(self, base_config):
        config = replace(
            base_config, budget=replace(base_config.budget, max_requests=100, max_total_tokens=10)
        )
        guard = BudgetGuard(config)
        guard.record(8, 5)
        with pytest.raises(BudgetExceededError, match="token budget"):
            guard.check()

    def test_stops_at_cost_limit(self, base_config):
        # Cost is the binding constraint: raise the other limits out of the way
        # so the reported reason proves the cost ceiling was the one enforced.
        config = replace(
            base_config,
            llm=replace(base_config.llm, model="gpt-4o"),
            budget=replace(
                base_config.budget,
                max_cost_usd=0.000001,
                max_requests=100,
                max_total_tokens=10_000_000,
            ),
        )
        guard = BudgetGuard(config)
        guard.record(100_000, 100_000)
        assert guard.cost_usd > 0
        with pytest.raises(BudgetExceededError, match="cost budget"):
            guard.check()

    def test_repeat_check_keeps_first_reason(self, base_config):
        config = replace(base_config, budget=replace(base_config.budget, max_requests=0))
        guard = BudgetGuard(config)
        with pytest.raises(BudgetExceededError) as first:
            guard.check()
        with pytest.raises(BudgetExceededError) as second:
            guard.check()
        assert str(first.value) == str(second.value)

    def test_unknown_model_yields_no_cost_estimate(self, base_config):
        config = replace(base_config, llm=replace(base_config.llm, model="mystery-model"))
        guard = BudgetGuard(config)
        guard.record(1_000_000, 1_000_000)
        assert guard.cost_usd is None


class TestRetryBehaviour:
    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        """Retry backoff must not slow the suite down."""
        import code_review_agent.llm.client as client_module

        monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)

    def test_retries_then_succeeds(self, base_config):
        config = replace(base_config, llm=replace(base_config.llm, max_retries=2))
        client = ScriptedRetryClient(
            config, [LLMRequestError("boom"), LLMRequestError("boom"), response("ok")]
        )
        assert client.complete(request_for()).text == "ok"
        assert client.calls == 3

    def test_gives_up_after_max_retries(self, base_config):
        config = replace(base_config, llm=replace(base_config.llm, max_retries=1))
        client = ScriptedRetryClient(config, [LLMRequestError("a"), LLMRequestError("b")])
        with pytest.raises(LLMRequestError, match="after 2 attempt"):
            client.complete(request_for())
        assert client.calls == 2

    def test_budget_error_is_not_retried(self, base_config):
        config = replace(base_config, budget=replace(base_config.budget, max_requests=0))
        client = ScriptedRetryClient(config, [response("never")])
        with pytest.raises(BudgetExceededError):
            client.complete(request_for())
        assert client.calls == 0

    def test_usage_is_recorded(self, base_config):
        client = ScriptedRetryClient(base_config, [response("ok", prompt=7, completion=3)])
        client.complete(request_for())
        assert client.guard.usage.total_tokens == 10


class TestSchemaRepair:
    def test_valid_first_response_passes_through(self, base_config):
        client = ScriptedRetryClient(base_config, [response('{"findings": []}')])
        assert client.complete_json(request_for()) == {"findings": []}
        assert client.calls == 1

    def test_prose_wrapped_json_is_recovered_without_repair(self, base_config):
        client = ScriptedRetryClient(base_config, [response('Here: {"findings": []} done')])
        assert client.complete_json(request_for()) == {"findings": []}
        assert client.calls == 1

    def test_schema_violation_triggers_repair(self, base_config):
        config = replace(base_config, llm=replace(base_config.llm, max_format_repairs=1))
        client = ScriptedRetryClient(
            config, [response('{"findings": "not-an-array"}'), response('{"findings": []}')]
        )
        assert client.complete_json(request_for()) == {"findings": []}
        assert client.calls == 2

    def test_unparseable_triggers_repair(self, base_config):
        config = replace(base_config, llm=replace(base_config.llm, max_format_repairs=1))
        client = ScriptedRetryClient(
            config, [response("no json here"), response('{"findings": []}')]
        )
        assert client.complete_json(request_for()) == {"findings": []}

    def test_exhausting_repairs_raises(self, base_config):
        config = replace(base_config, llm=replace(base_config.llm, max_format_repairs=1))
        client = ScriptedRetryClient(
            config, [response("bad"), response("still bad"), response("never reached")]
        )
        with pytest.raises(LLMResponseFormatError, match="did not satisfy"):
            client.complete_json(request_for())

    def test_repair_consumes_budget(self, base_config):
        config = replace(base_config, llm=replace(base_config.llm, max_format_repairs=1))
        client = ScriptedRetryClient(
            config, [response("bad", 10, 5), response('{"findings": []}', 10, 5)]
        )
        client.complete_json(request_for())
        assert client.guard.usage.requests == 2


class TestBuildClient:
    def test_fake_provider(self, base_config):
        assert isinstance(build_client(base_config), FakeLLMClient)

    def test_default_provider_is_openai_compatible(self):
        from code_review_agent.llm.client import OpenAICompatibleClient

        assert isinstance(build_client(AgentConfig()), OpenAICompatibleClient)

    def test_anthropic_provider(self):
        from code_review_agent.llm.client import AnthropicClient

        config = replace(AgentConfig(), llm=replace(AgentConfig().llm, provider="anthropic"))
        assert isinstance(build_client(config), AnthropicClient)


class TestNetworkClientsWithoutKeys:
    def test_openai_client_requires_key(self, base_config):
        from code_review_agent.llm.client import OpenAICompatibleClient

        config = replace(base_config, llm=replace(base_config.llm, provider="openai", api_key_env="DEFINITELY_UNSET_KEY"))
        client = OpenAICompatibleClient(config, api_key=None)
        with pytest.raises(LLMRequestError, match="no API key"):
            client.complete(request_for())

    def test_payload_includes_response_format_for_json_object(self, base_config):
        from code_review_agent.llm.client import OpenAICompatibleClient

        config = replace(
            base_config,
            llm=replace(base_config.llm, provider="openai", response_format="json_object"),
        )
        client = OpenAICompatibleClient(config, api_key="test-key")
        payload = client._payload(request_for())
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["model"] == config.llm.model

    def test_payload_omits_response_format_when_disabled(self, base_config):
        from code_review_agent.llm.client import OpenAICompatibleClient

        config = replace(
            base_config, llm=replace(base_config.llm, response_format="none")
        )
        client = OpenAICompatibleClient(config, api_key="test-key")
        assert "response_format" not in client._payload(request_for())

    def test_parses_openai_envelope(self, base_config):
        from code_review_agent.llm.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(base_config, api_key="k")
        parsed = {
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": '{"findings": []}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
        result = client._to_response(parsed)
        assert result.text == '{"findings": []}'
        assert result.usage.total_tokens == 15

    def test_rejects_envelope_without_choices(self, base_config):
        from code_review_agent.llm.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(base_config, api_key="k")
        with pytest.raises(LLMRequestError, match="no choices"):
            client._to_response({"choices": []})
