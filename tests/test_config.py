"""Configuration tests: precedence, validation, secret handling (doc 7.3)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from code_review_agent.config.settings import (
    AgentConfig,
    estimate_cost,
    load_config,
)
from code_review_agent.errors import InputError


class TestDefaults:
    def test_defaults_are_conservative(self):
        config = AgentConfig()
        assert config.llm.provider == "openai"
        assert config.review.enable_tool_calls is True
        assert config.budget.max_files > 0
        assert config.budget.max_requests > 0

    def test_api_key_is_absent_without_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert AgentConfig().api_key is None


class TestTomlLoading:
    def test_loads_all_sections(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text(
            "[llm]\nprovider = \"deepseek\"\nmodel = \"deepseek-chat\"\n\n"
            "[budget]\nmax_files = 5\nmax_requests = 7\n\n"
            "[report]\noutput_dir = \"out\"\n",
            encoding="utf-8",
        )
        config = load_config(path, env={})
        assert config.llm.provider == "deepseek"
        assert config.llm.model == "deepseek-chat"
        assert config.budget.max_files == 5
        assert config.report.output_dir == "out"

    def test_missing_explicit_config_is_an_error(self, tmp_path: Path):
        with pytest.raises(InputError, match="config file not found"):
            load_config(tmp_path / "nope.toml", env={})

    def test_unknown_section_is_rejected(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text("[nonsense]\nx = 1\n", encoding="utf-8")
        with pytest.raises(InputError, match="unknown config section"):
            load_config(path, env={})

    def test_unknown_key_is_rejected(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text("[llm]\nmodle = \"typo\"\n", encoding="utf-8")
        with pytest.raises(InputError, match="unknown key"):
            load_config(path, env={})

    def test_malformed_toml_is_rejected(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text("[llm\nprovider = ", encoding="utf-8")
        with pytest.raises(InputError, match="cannot read config"):
            load_config(path, env={})

    def test_extension_list_from_toml(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text('[scanner]\nextensions = [".ts", ".vue"]\n', encoding="utf-8")
        assert load_config(path, env={}).scanner.extensions == (".ts", ".vue")

    def test_agent_section_sets_log_level(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text('[agent]\nlog_level = "DEBUG"\n', encoding="utf-8")
        assert load_config(path, env={}).log_level == "DEBUG"

    def test_agent_section_rejects_unknown_keys(self, tmp_path: Path):
        # Guards against the easy TOML mistake of nesting a key under the wrong
        # table, where it would otherwise be silently ignored.
        path = tmp_path / "cfg.toml"
        path.write_text('[agent]\nloglevel = "DEBUG"\n', encoding="utf-8")
        with pytest.raises(InputError, match="unknown key"):
            load_config(path, env={})

    def test_shipped_example_config_is_valid(self):
        # The documented example must actually load, or it misleads users.
        example = Path(__file__).resolve().parents[1] / "code-review.example.toml"
        config = load_config(example, env={})
        assert config.llm.provider == "openai"
        assert config.static_analysis.builtin_rules is True
        assert config.review.fail_on_partial_review is True


class TestEnvironmentOverrides:
    def test_env_overrides_toml(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text('[llm]\nmodel = "from-file"\n', encoding="utf-8")
        config = load_config(path, env={"CODE_REVIEW_MODEL": "from-env"})
        assert config.llm.model == "from-env"

    def test_env_numeric_coercion(self):
        config = load_config(None, env={"CODE_REVIEW_MAX_FILES": "12"})
        assert config.budget.max_files == 12
        assert isinstance(config.budget.max_files, int)

    def test_invalid_numeric_env_is_rejected(self):
        with pytest.raises(InputError, match="invalid value"):
            load_config(None, env={"CODE_REVIEW_MAX_FILES": "many"})

    def test_empty_env_value_is_ignored(self):
        config = load_config(None, env={"CODE_REVIEW_MODEL": ""})
        assert config.llm.model == AgentConfig().llm.model

    def test_provider_key_convention_is_detected(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
        config = load_config(
            None, env={"CODE_REVIEW_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "x"}
        )
        assert config.llm.api_key_env == "DEEPSEEK_API_KEY"
        assert config.api_key == "x"

    def test_explicit_api_key_env_wins(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "b")
        config = load_config(
            None,
            env={
                "CODE_REVIEW_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "a",
                "CODE_REVIEW_API_KEY_ENV": "MY_KEY",
                "MY_KEY": "b",
            },
        )
        assert config.llm.api_key_env == "MY_KEY"
        assert config.api_key == "b"

    def test_log_level_override(self):
        assert load_config(None, env={"CODE_REVIEW_LOG_LEVEL": "DEBUG"}).log_level == "DEBUG"


class TestValidation:
    def test_unsupported_provider_rejected(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text('[llm]\nprovider = "wishful"\n', encoding="utf-8")
        with pytest.raises(InputError, match="unsupported provider"):
            load_config(path, env={})

    def test_negative_retries_rejected(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text("[llm]\nmax_retries = -1\n", encoding="utf-8")
        with pytest.raises(InputError, match="max_retries"):
            load_config(path, env={})

    def test_zero_max_files_rejected(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text("[budget]\nmax_files = 0\n", encoding="utf-8")
        with pytest.raises(InputError, match="max_files"):
            load_config(path, env={})

    def test_confidence_out_of_range_rejected(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text("[review]\nmin_confidence = 1.5\n", encoding="utf-8")
        with pytest.raises(InputError, match="min_confidence"):
            load_config(path, env={})

    def test_bad_response_format_rejected(self, tmp_path: Path):
        path = tmp_path / "cfg.toml"
        path.write_text('[llm]\nresponse_format = "yaml"\n', encoding="utf-8")
        with pytest.raises(InputError, match="response_format"):
            load_config(path, env={})


class TestSecretHandling:
    def test_serialised_config_contains_no_key_material(self):
        config = AgentConfig()
        config = replace(config, llm=replace(config.llm, api_key_env="MY_SECRET_KEY"))
        import json

        blob = json.dumps(config.redacted_dict())
        # Only the variable *name* is recorded, never a value.
        assert "MY_SECRET_KEY" in blob
        assert "sk-" not in blob

    def test_api_key_present_flag_reflects_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert AgentConfig().api_key == "sk-test"


class TestCostEstimate:
    def test_known_model_is_priced(self):
        config = replace(AgentConfig(), llm=replace(AgentConfig().llm, model="gpt-4o"))
        cost = estimate_cost(config, prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == pytest.approx(2.50)

    def test_unknown_model_returns_none_not_zero(self):
        config = replace(AgentConfig(), llm=replace(AgentConfig().llm, model="mystery"))
        assert estimate_cost(config, 1_000_000, 1_000_000) is None

    def test_zero_tokens_is_zero_cost(self):
        config = replace(AgentConfig(), llm=replace(AgentConfig().llm, model="gpt-4o"))
        assert estimate_cost(config, 0, 0) == 0.0
