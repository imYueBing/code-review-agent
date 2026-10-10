"""Configuration loading: defaults <- TOML file <- environment variables.

Section 7.3 requires provider/model/auth/cost limits to be configuration, not
hardcoded in the workflow. Secrets are read from the environment only; they are
never written into the config object's serialised form.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..errors import InputError

ENV_PREFIX = "CODE_REVIEW_"

#: USD per 1M tokens, used only for the cost estimate in the run summary.
#: Unknown models yield ``None`` rather than a fabricated number.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o4-mini": (1.10, 4.40),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 4096
    timeout_seconds: float = 90.0
    max_retries: int = 3
    max_format_repairs: int = 2
    #: Response format hint: "json_schema" | "json_object" | "none".
    response_format: str = "json_object"
    #: Optional offline fixture used by the fake provider in tests/demos.
    fake_response_path: str | None = None


@dataclass
class BudgetConfig:
    max_total_tokens: int = 200_000
    max_requests: int = 60
    max_cost_usd: float | None = None
    max_files: int = 40
    max_file_bytes: int = 200_000
    #: Character budget for a single review prompt.
    max_context_chars: int = 120_000
    #: Lines of context printed around a static-analysis diagnostic.
    diagnostic_context_lines: int = 3


@dataclass
class ScannerConfig:
    extensions: tuple[str, ...] = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
    max_files: int = 500
    follow_symlinks: bool = False


@dataclass
class StaticAnalysisConfig:
    enabled: bool = True
    #: "auto" probes for eslint; "off" disables; a path pins an executable.
    eslint_command: str = "auto"
    eslint_timeout_seconds: float = 120.0
    #: When True a missing tool is fatal; when False it is skipped and recorded.
    fail_on_missing_tool: bool = False
    max_diagnostics_per_file: int = 25
    #: Always-on heuristic analyzer requiring no external tooling.
    builtin_rules: bool = True


@dataclass
class ReviewConfig:
    #: Maximum whole-file review rounds for a single file.
    max_review_rounds_per_file: int = 2
    #: Findings below this confidence are dropped before validation.
    min_confidence: float = 0.0
    max_findings_per_file: int = 20
    include_static_diagnostics: bool = True
    #: Allow the agent to request read-only project tools before concluding.
    enable_tool_calls: bool = True
    #: Maximum tool calls honoured per file.
    max_tool_calls_per_file: int = 2
    #: When True (the default), a file whose LLM review failed makes the whole
    #: run FAILED rather than a merely partial "completed" run. This keeps the
    #: section 5.3 rule that an incomplete review is never reported as success.
    fail_on_partial_review: bool = True


@dataclass
class ReportConfig:
    output_dir: str = ".code-review"
    markdown_filename: str = "report.md"
    json_filename: str = "report.json"
    max_evidence_chars: int = 400


@dataclass
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    static_analysis: StaticAnalysisConfig = field(default_factory=StaticAnalysisConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def redacted_dict(self) -> dict[str, Any]:
        """Config safe to embed in a report: no key material is present."""
        return self.to_dict()

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.llm.api_key_env) or None

    def price_for_model(self) -> tuple[float, float] | None:
        return MODEL_PRICES.get(self.llm.model)


_SECTIONS = {
    "llm": LLMConfig,
    "budget": BudgetConfig,
    "scanner": ScannerConfig,
    "static_analysis": StaticAnalysisConfig,
    "review": ReviewConfig,
    "report": ReportConfig,
}

#: ``[agent]`` configures top-level fields rather than a dataclass section.
_TOP_LEVEL_KEYS = frozenset({"log_level"})

_ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    # ENV VAR -> (section or "log_level", field, caster)
    "CODE_REVIEW_PROVIDER": ("llm", "provider", str),
    "CODE_REVIEW_MODEL": ("llm", "model", str),
    "CODE_REVIEW_BASE_URL": ("llm", "base_url", str),
    "CODE_REVIEW_API_KEY_ENV": ("llm", "api_key_env", str),
    "CODE_REVIEW_MAX_REQUESTS": ("budget", "max_requests", int),
    "CODE_REVIEW_MAX_TOKENS": ("budget", "max_total_tokens", int),
    "CODE_REVIEW_MAX_COST_USD": ("budget", "max_cost_usd", float),
    "CODE_REVIEW_MAX_FILES": ("budget", "max_files", int),
    "CODE_REVIEW_OUTPUT_DIR": ("report", "output_dir", str),
    # Log level is a top-level field, not a section; "" marks that special case.
    "CODE_REVIEW_LOG_LEVEL": ("", "log_level", str),
    "CODE_REVIEW_ESLINT_COMMAND": ("static_analysis", "eslint_command", str),
}

# Provider-specific conventional key variables, tried in order.
_PROVIDER_KEY_ENVS = {
    "openai": ("OPENAI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai-compatible": ("OPENAI_API_KEY", "CODE_REVIEW_API_KEY"),
}


def default_config_path() -> Path:
    return Path.cwd() / "code-review.toml"


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> AgentConfig:
    """Build config with precedence defaults < TOML < environment."""
    environ = dict(os.environ if env is None else env)
    config = AgentConfig()

    path: Path | None = None
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise InputError(f"config file not found: {path}")
    elif default_config_path().is_file():
        path = default_config_path()

    if path is not None:
        config = _apply_toml(config, path)

    config = _apply_env(config, environ)
    _check(config)
    return config


def _apply_toml(config: AgentConfig, path: Path) -> AgentConfig:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise InputError(f"cannot read config {path}: {exc}") from exc

    unknown = set(data) - set(_SECTIONS) - _TOP_LEVEL_KEYS - {"agent"}
    if unknown:
        raise InputError(f"unknown config section(s) in {path}: {sorted(unknown)}")

    updates: dict[str, Any] = {}
    for section, values in data.items():
        if section in _TOP_LEVEL_KEYS:
            updates[section] = str(values)
            continue
        if section == "agent":
            if not isinstance(values, dict):
                raise InputError("config section [agent] must be a table")
            unknown_fields = set(values) - _TOP_LEVEL_KEYS
            if unknown_fields:
                raise InputError(
                    f"unknown key(s) in [agent] of {path}: {sorted(unknown_fields)}"
                )
            updates.update({key: str(value) for key, value in values.items()})
            continue
        cls = _SECTIONS[section]
        if not isinstance(values, dict):
            raise InputError(f"config section [{section}] must be a table")
        current = getattr(config, section)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown_fields = set(values) - allowed
        if unknown_fields:
            raise InputError(
                f"unknown key(s) in [{section}] of {path}: {sorted(unknown_fields)}"
            )
        coerced = {key: _coerce(value, getattr(current, key)) for key, value in values.items()}
        updates[section] = replace(current, **coerced)
    return replace(config, **updates)


def _coerce(value: Any, reference: Any) -> Any:
    if isinstance(reference, tuple) and isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(reference, bool) and not isinstance(value, bool):
        raise InputError(f"expected a boolean, got {value!r}")
    if isinstance(reference, int) and not isinstance(reference, bool) and isinstance(value, float):
        return int(value)
    return value


def _apply_env(config: AgentConfig, environ: dict[str, str]) -> AgentConfig:
    updates: dict[str, Any] = {}
    top_level: dict[str, Any] = {}
    for env_name, (section, field_name, caster) in _ENV_OVERRIDES.items():
        raw = environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            value = caster(raw)
        except (TypeError, ValueError) as exc:
            raise InputError(f"invalid value for {env_name}: {raw!r}") from exc
        if not section:
            top_level[field_name] = value
            continue
        current = updates.get(section, getattr(config, section))
        updates[section] = replace(current, **{field_name: value})

    # If the user did not pin a key variable, fall back to a provider convention.
    llm_section = updates.get("llm", config.llm)
    if "CODE_REVIEW_API_KEY_ENV" not in environ and llm_section.provider in _PROVIDER_KEY_ENVS:
        for candidate in _PROVIDER_KEY_ENVS[llm_section.provider]:
            if environ.get(candidate):
                llm_section = replace(llm_section, api_key_env=candidate)
                break
    updates["llm"] = llm_section

    return replace(config, **updates, **top_level)


def _check(config: AgentConfig) -> None:
    if config.llm.provider not in {"openai", "openai-compatible", "deepseek", "anthropic", "fake"}:
        raise InputError(f"unsupported provider: {config.llm.provider}")
    if config.llm.max_retries < 0:
        raise InputError("llm.max_retries must be >= 0")
    if config.llm.max_output_tokens <= 0:
        raise InputError("llm.max_output_tokens must be > 0")
    if config.budget.max_files <= 0:
        raise InputError("budget.max_files must be > 0")
    if config.budget.max_total_tokens <= 0:
        raise InputError("budget.max_total_tokens must be > 0")
    if config.review.min_confidence < 0 or config.review.min_confidence > 1:
        raise InputError("review.min_confidence must be within [0, 1]")
    if config.llm.response_format not in {"json_schema", "json_object", "none"}:
        raise InputError(f"unsupported llm.response_format: {config.llm.response_format}")


def estimate_cost(config: AgentConfig, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = config.price_for_model()
    if price is None:
        return None
    prompt_rate, completion_rate = price
    return (prompt_tokens / 1_000_000) * prompt_rate + (completion_tokens / 1_000_000) * completion_rate
