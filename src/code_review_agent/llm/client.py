"""LLM clients.

A single narrow interface keeps provider details out of the workflow
(section 7.3). Two implementations ship here:

* :class:`FakeLLMClient` -- offline, deterministic, used by every test so the
  suite never spends money or needs network (section 9.2).
* :class:`OpenAICompatibleClient` -- HTTP against any OpenAI-compatible
  ``/chat/completions`` endpoint (OpenAI, DeepSeek, local servers, ...).
* :class:`AnthropicClient` -- the Messages API, which differs enough in shape
  to warrant its own small adapter.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..config.settings import AgentConfig
from ..errors import BudgetExceededError, LLMRequestError, LLMResponseFormatError
from ..models import TokenUsage
from .parsing import extract_json_object, repair_prompt

logger = logging.getLogger(__name__)

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMRequest:
    messages: list[LLMMessage]
    system: str | None = None
    schema: dict[str, Any] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    purpose: str = "review"


@dataclass
class LLMResponse:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str | None = None


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class BudgetGuard:
    """Enforces token / request / cost ceilings before each request.

    Section 5.3: when the budget is exhausted we stop issuing new model
    requests and keep whatever results already exist.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._budget = config.budget
        self._prices = config.price_for_model()
        self.usage = TokenUsage()
        self.cost_usd: float | None = 0.0 if self._prices else None
        self.stopped_reason: str | None = None

    @property
    def exhausted(self) -> bool:
        return self.stopped_reason is not None

    def check(self) -> None:
        if self.stopped_reason:
            raise BudgetExceededError(self.stopped_reason)
        # Evaluate every limit and report the *most constraining* one, so a tight
        # cost ceiling is never misreported as a token problem.
        breached: list[tuple[float, str]] = []
        if self.usage.requests >= self._budget.max_requests:
            breached.append(
                (
                    self.usage.requests / max(1, self._budget.max_requests),
                    f"request budget exhausted ({self.usage.requests}/{self._budget.max_requests})",
                )
            )
        if self.usage.total_tokens >= self._budget.max_total_tokens:
            breached.append(
                (
                    self.usage.total_tokens / max(1, self._budget.max_total_tokens),
                    f"token budget exhausted ({self.usage.total_tokens}/{self._budget.max_total_tokens})",
                )
            )
        if (
            self._budget.max_cost_usd is not None
            and self.cost_usd is not None
            and self.cost_usd >= self._budget.max_cost_usd
        ):
            breached.append(
                (
                    self.cost_usd / max(self._budget.max_cost_usd, 1e-12),
                    f"cost budget exhausted (${self.cost_usd:.4f}/${self._budget.max_cost_usd:.4f})",
                )
            )
        if breached:
            self.stopped_reason = max(breached, key=lambda item: item[0])[1]
            raise BudgetExceededError(self.stopped_reason)

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage.add(prompt_tokens, completion_tokens)
        if self._prices is not None:
            prompt_rate, completion_rate = self._prices
            increment = (prompt_tokens / 1_000_000) * prompt_rate + (
                completion_tokens / 1_000_000
            ) * completion_rate
            self.cost_usd = (self.cost_usd or 0.0) + increment

    def remaining_headroom(self) -> dict[str, Any]:
        return {
            "requests_used": self.usage.requests,
            "requests_limit": self._budget.max_requests,
            "tokens_used": self.usage.total_tokens,
            "tokens_limit": self._budget.max_total_tokens,
            "cost_usd": self.cost_usd,
            "cost_limit_usd": self._budget.max_cost_usd,
            "stopped_reason": self.stopped_reason,
        }


class FakeLLMClient:
    """Deterministic offline client.

    Responses are supplied as raw text (so JSON-repair behaviour can be
    exercised) via a fixture file, a list, or a callable.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        responder: Callable[[LLMRequest], str] | None = None,
        default: str = '{"findings": []}',
        prompt_tokens_per_request: int = 500,
        completion_tokens_per_request: int = 200,
    ) -> None:
        self._responses = list(responses or [])
        self._responder = responder
        self._default = default
        self._prompt_tokens = prompt_tokens_per_request
        self._completion_tokens = completion_tokens_per_request
        self.requests: list[LLMRequest] = []

    @classmethod
    def from_fixture(cls, path: str) -> "FakeLLMClient":
        """Load a JSON file holding either a list of strings or ``{"responses": [...]}``."""
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if "responses" not in payload:
                raise LLMResponseFormatError(
                    f"fake fixture {path} must be a list or an object with a 'responses' key"
                )
            payload = payload["responses"]
        if not isinstance(payload, list):
            raise LLMResponseFormatError(f"fake fixture {path} must hold a list of responses")
        return cls([item if isinstance(item, str) else json.dumps(item) for item in payload])

    def complete(self, request: LLMRequest) -> LLMResponse:
        index = len(self.requests)
        self.requests.append(request)
        if self._responder is not None:
            text = self._responder(request)
        elif index < len(self._responses):
            text = self._responses[index]
        else:
            text = self._default
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                requests=1,
            ),
            model="fake",
            finish_reason="stop",
        )

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        """Parse and schema-validate the scripted response.

        The fake provider is a first-class provider (``--provider fake``), so it
        must satisfy the full :class:`LLMClient` contract rather than only the
        transport half. Validation failures surface as
        :class:`LLMResponseFormatError`, exactly as a real provider would.
        """
        from ..models import schema_errors_for_response

        response = self.complete(request)
        payload = extract_json_object(response.text)
        if request.schema is not None:
            errors = schema_errors_for_response(payload)
            if errors:
                raise LLMResponseFormatError(
                    "fake response did not satisfy the required schema: " + "; ".join(errors[:5])
                )
        return payload


class _RetryingClient:
    """Shared retry/repair loop for network clients."""

    def __init__(self, config: AgentConfig, guard: BudgetGuard | None = None) -> None:
        self.config = config
        self.guard = guard or BudgetGuard(config)

    # -- transport, overridden by subclasses -----------------------------
    def _send(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - abstract
        raise NotImplementedError

    def complete(self, request: LLMRequest) -> LLMResponse:
        last_error: Exception | None = None
        attempts = self.config.llm.max_retries + 1
        for attempt in range(attempts):
            self.guard.check()
            try:
                response = self._send(request)
            except BudgetExceededError:
                raise
            except LLMRequestError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                # Exponential backoff, capped so tests stay fast.
                delay = min(2.0**attempt * 0.5, 4.0)
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue
            self.guard.record(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response
        raise LLMRequestError(f"LLM request failed after {attempts} attempt(s): {last_error}")

    def complete_json(self, request: LLMRequest) -> dict[str, Any]:
        """Send *request* and return validated JSON, repairing format on failure.

        Raises :class:`LLMResponseFormatError` when the model cannot produce
        schema-valid JSON within ``max_format_repairs`` extra attempts.
        """
        from ..models import schema_errors_for_response

        response = self.complete(request)
        text = response.text
        errors: list[str] = []
        for repair_index in range(self.config.llm.max_format_repairs + 1):
            try:
                payload = extract_json_object(text)
            except LLMResponseFormatError as exc:
                errors = [str(exc)]
            else:
                if request.schema is None:
                    return payload
                errors = schema_errors_for_response(payload)
                if not errors:
                    return payload

            if repair_index >= self.config.llm.max_format_repairs:
                break
            logger.warning(
                "Model output failed validation (%d error(s)); requesting repair %d/%d",
                len(errors),
                repair_index + 1,
                self.config.llm.max_format_repairs,
            )
            repair_request = LLMRequest(
                messages=[
                    *request.messages,
                    LLMMessage(role="assistant", content=text),
                    LLMMessage(
                        role="user",
                        content=repair_prompt(
                            text, errors, request.schema or {"type": "object"}
                        ),
                    ),
                ],
                system=request.system,
                schema=request.schema,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                purpose=f"{request.purpose}:repair",
            )
            text = self.complete(repair_request).text

        raise LLMResponseFormatError(
            "model output did not satisfy the required schema after "
            f"{self.config.llm.max_format_repairs + 1} attempt(s): {'; '.join(errors[:5])}"
        )


class OpenAICompatibleClient(_RetryingClient):
    """Client for any ``/chat/completions``-style endpoint."""

    def __init__(
        self,
        config: AgentConfig,
        api_key: str | None = None,
        guard: BudgetGuard | None = None,
    ) -> None:
        super().__init__(config, guard)
        self.api_key = api_key if api_key is not None else config.api_key
        self.base_url = (
            config.llm.base_url
            or DEFAULT_BASE_URLS.get(config.llm.provider)
            or DEFAULT_BASE_URLS["openai"]
        ).rstrip("/")

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        payload: dict[str, Any] = {
            "model": self.config.llm.model,
            "messages": messages,
            "temperature": (
                request.temperature if request.temperature is not None else self.config.llm.temperature
            ),
            "max_tokens": request.max_output_tokens or self.config.llm.max_output_tokens,
        }
        if request.schema is not None and self.config.llm.response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": request.purpose or "review", "schema": request.schema},
            }
        elif request.schema is not None and self.config.llm.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.config.llm.provider == "deepseek":
            # DeepSeek needs an explicit nudge or it pads JSON with prose.
            messages.append(
                {"role": "system", "content": "Respond with a single JSON object and nothing else."}
            )
        return payload

    def _send(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise LLMRequestError(
                f"no API key found; set ${self.config.llm.api_key_env} or use the fake provider"
            )
        body = json.dumps(self._payload(request)).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.config.llm.timeout_seconds) as handle:
                raw = handle.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # pragma: no cover - best effort diagnostics
                pass
            # 429 and 5xx are worth retrying; 4xx generally is not.
            if exc.code in {408, 409, 425, 429} or exc.code >= 500:
                raise LLMRequestError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
            raise LLMRequestError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMRequestError(f"could not reach {self.base_url}: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMRequestError(f"provider returned non-JSON envelope: {raw[:200]!r}") from exc
        return self._to_response(parsed)

    def _to_response(self, parsed: dict[str, Any]) -> LLMResponse:
        choices = parsed.get("choices") or []
        if not choices:
            raise LLMRequestError(f"provider response contained no choices: {str(parsed)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if text is None:
            # Some providers return a content-part list.
            parts = message.get("content_parts") or []
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not isinstance(text, str):
            raise LLMRequestError("provider response message had no textual content")
        usage = parsed.get("usage") or {}
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                requests=1,
            ),
            model=str(parsed.get("model") or self.config.llm.model),
            finish_reason=choice.get("finish_reason"),
        )


class AnthropicClient(_RetryingClient):
    """Adapter for the Anthropic Messages API."""

    def __init__(
        self,
        config: AgentConfig,
        api_key: str | None = None,
        guard: BudgetGuard | None = None,
    ) -> None:
        super().__init__(config, guard)
        self.api_key = api_key if api_key is not None else config.api_key
        self.base_url = (config.llm.base_url or DEFAULT_BASE_URLS["anthropic"]).rstrip("/")

    def _send(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise LLMRequestError(
                f"no API key found; set ${self.config.llm.api_key_env} or use the fake provider"
            )
        payload: dict[str, Any] = {
            "model": self.config.llm.model,
            "max_tokens": request.max_output_tokens or self.config.llm.max_output_tokens,
            "temperature": (
                request.temperature if request.temperature is not None else self.config.llm.temperature
            ),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system:
            payload["system"] = request.system
        http_request = urllib.request.Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.config.llm.timeout_seconds) as handle:
                parsed = json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # pragma: no cover
                pass
            raise LLMRequestError(f"HTTP {exc.code} from Anthropic: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMRequestError(f"could not reach Anthropic: {exc}") from exc

        blocks = parsed.get("content") or []
        text = "".join(
            block.get("text", "") for block in blocks if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = parsed.get("usage") or {}
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                prompt_tokens=int(usage.get("input_tokens") or 0),
                completion_tokens=int(usage.get("output_tokens") or 0),
                requests=1,
            ),
            model=str(parsed.get("model") or self.config.llm.model),
            finish_reason=parsed.get("stop_reason"),
        )


def build_client(config: AgentConfig, guard: BudgetGuard | None = None) -> LLMClient:
    """Factory honouring ``llm.provider``."""
    provider = config.llm.provider
    if provider == "fake":
        if config.llm.fake_response_path:
            return FakeLLMClient.from_fixture(config.llm.fake_response_path)
        return FakeLLMClient()
    if provider == "anthropic":
        guard = guard or BudgetGuard(config)
        return AnthropicClient(config, guard=guard)
    guard = guard or BudgetGuard(config)
    return OpenAICompatibleClient(config, guard=guard)
