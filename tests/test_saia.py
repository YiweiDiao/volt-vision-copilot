from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from volt_vision.agent.adk_factory import create_incident_copilot_agent
from volt_vision.agent.saia import (
    DEFAULT_SAIA_BASE_URL,
    DEFAULT_SAIA_MAX_TOKENS,
    PROJECT_ROOT_ENV_PATH,
    SaiaSettings,
    create_saia_litellm_model,
    load_saia_settings_from_env,
    resolve_optional_saia_model_from_env,
    safe_saia_configuration_status,
)


def env(**overrides: str) -> dict[str, str]:
    return overrides


def test_missing_key_or_model_is_not_configured() -> None:
    assert load_saia_settings_from_env({}) is None
    assert safe_saia_configuration_status({}) == "not_configured"
    assert load_saia_settings_from_env(env(VOLT_VISION_SAIA_API_KEY="key")) is None
    assert load_saia_settings_from_env(env(VOLT_VISION_SAIA_MODEL="model")) is None


def test_whitespace_values_are_not_configured() -> None:
    values = env(
        VOLT_VISION_SAIA_API_KEY=" ",
        VOLT_VISION_SAIA_MODEL="\t",
    )

    assert load_saia_settings_from_env(values) is None
    assert safe_saia_configuration_status(values) == "not_configured"


def test_defaults_apply_when_key_and_model_exist() -> None:
    settings = load_saia_settings_from_env(
        env(
            VOLT_VISION_SAIA_API_KEY="secret-key",
            VOLT_VISION_SAIA_MODEL="qwen3.6-27b",
        )
    )

    assert settings is not None
    assert settings.base_url == DEFAULT_SAIA_BASE_URL
    assert settings.max_tokens == DEFAULT_SAIA_MAX_TOKENS
    assert settings.raw_model_id == "qwen3.6-27b"
    assert settings.litellm_route == "openai/qwen3.6-27b"


def test_explicit_base_url_and_max_tokens() -> None:
    settings = load_saia_settings_from_env(
        env(
            VOLT_VISION_SAIA_API_KEY="secret-key",
            VOLT_VISION_SAIA_MODEL="model-a",
            VOLT_VISION_SAIA_BASE_URL="https://example.test/v1",
            VOLT_VISION_SAIA_MAX_TOKENS="2048",
        )
    )

    assert settings is not None
    assert settings.base_url == "https://example.test/v1"
    assert settings.max_tokens == 2048


def test_complete_temporary_env_file_resolves_when_process_env_absent(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "VOLT_VISION_SAIA_API_KEY=test-secret-value",
                "VOLT_VISION_SAIA_BASE_URL=https://example.test/v1",
                "VOLT_VISION_SAIA_MODEL=glm-4.7",
                "VOLT_VISION_SAIA_MAX_TOKENS=512",
            )
        ),
        encoding="utf-8",
    )

    settings = load_saia_settings_from_env({}, env_file=env_file)

    assert settings is not None
    assert settings.raw_model_id == "glm-4.7"
    assert settings.base_url == "https://example.test/v1"
    assert settings.max_tokens == 512
    assert safe_saia_configuration_status({}, env_file=env_file) == "configured"


def test_process_env_overrides_conflicting_temporary_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "VOLT_VISION_SAIA_API_KEY=file-secret",
                "VOLT_VISION_SAIA_BASE_URL=https://file.example/v1",
                "VOLT_VISION_SAIA_MODEL=file-model",
                "VOLT_VISION_SAIA_MAX_TOKENS=256",
            )
        ),
        encoding="utf-8",
    )

    settings = load_saia_settings_from_env(
        env(
            VOLT_VISION_SAIA_API_KEY="process-secret",
            VOLT_VISION_SAIA_BASE_URL="https://process.example/v1",
            VOLT_VISION_SAIA_MODEL="process-model",
            VOLT_VISION_SAIA_MAX_TOKENS="1024",
        ),
        env_file=env_file,
    )

    assert settings is not None
    assert settings.api_key.get_secret_value() == "process-secret"
    assert settings.base_url == "https://process.example/v1"
    assert settings.raw_model_id == "process-model"
    assert settings.max_tokens == 1024


def test_incomplete_or_disabled_dotenv_remains_not_configured(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "VOLT_VISION_SAIA_API_KEY=test-secret-value\n",
        encoding="utf-8",
    )

    assert load_saia_settings_from_env({}, env_file=env_file) is None
    assert safe_saia_configuration_status({}, env_file=env_file) == "not_configured"
    assert load_saia_settings_from_env({}, env_file=None) is None
    assert safe_saia_configuration_status({}, env_file=None) == "not_configured"


def test_explicit_mapping_does_not_read_real_project_root_env() -> None:
    settings = load_saia_settings_from_env(
        {
            "VOLT_VISION_SAIA_API_KEY": "mapping-secret",
            "VOLT_VISION_SAIA_MODEL": "mapping-model",
        }
    )

    assert settings is not None
    assert settings.raw_model_id == "mapping-model"
    assert PROJECT_ROOT_ENV_PATH.name == ".env"


def test_invalid_base_url_or_max_tokens_is_invalid_configuration() -> None:
    bad_base = env(
        VOLT_VISION_SAIA_API_KEY="secret-key",
        VOLT_VISION_SAIA_MODEL="model-a",
        VOLT_VISION_SAIA_BASE_URL="not-a-url",
    )
    bad_tokens = env(
        VOLT_VISION_SAIA_API_KEY="secret-key",
        VOLT_VISION_SAIA_MODEL="model-a",
        VOLT_VISION_SAIA_MAX_TOKENS="0",
    )

    assert safe_saia_configuration_status(bad_base) == "invalid_configuration"
    assert safe_saia_configuration_status(bad_tokens) == "invalid_configuration"
    with pytest.raises(ValueError):
        load_saia_settings_from_env(bad_base)


def test_saia_settings_is_immutable_and_redacts_key() -> None:
    settings = load_saia_settings_from_env(
        env(
            VOLT_VISION_SAIA_API_KEY="plain-secret",
            VOLT_VISION_SAIA_MODEL="model-a",
        )
    )
    assert settings is not None

    with pytest.raises(ValidationError):
        settings.raw_model_id = "other"  # type: ignore[misc]

    assert "plain-secret" not in repr(settings)
    assert "plain-secret" not in settings.model_dump_json()


def test_litellm_model_route_and_kwargs_are_constructed_with_fake() -> None:
    settings = load_saia_settings_from_env(
        env(
            VOLT_VISION_SAIA_API_KEY="plain-secret",
            VOLT_VISION_SAIA_MODEL="qwen3.6-27b",
        )
    )
    assert settings is not None
    calls: list[dict[str, object]] = []

    class FakeLiteLlm:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    model = create_saia_litellm_model(settings, lite_llm_cls=FakeLiteLlm)

    assert isinstance(model, FakeLiteLlm)
    assert calls == [
        {
            "model": "openai/qwen3.6-27b",
            "api_key": "plain-secret",
            "base_url": DEFAULT_SAIA_BASE_URL,
            "max_tokens": DEFAULT_SAIA_MAX_TOKENS,
        }
    ]


def test_resolve_optional_saia_model_from_env_uses_model_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[SaiaSettings] = []

    def fake_create(settings: SaiaSettings) -> object:
        created.append(settings)
        return object()

    monkeypatch.setattr("volt_vision.agent.saia.create_saia_litellm_model", fake_create)

    assert resolve_optional_saia_model_from_env({}) is None
    model = resolve_optional_saia_model_from_env(
        env(
            VOLT_VISION_SAIA_API_KEY="plain-secret",
            VOLT_VISION_SAIA_MODEL="model-a",
        )
    )

    assert model is not None
    assert created[0].litellm_route == "openai/model-a"


def test_agent_factory_accepts_model_object_without_raw_saia_string() -> None:
    from google.adk.models.lite_llm import LiteLlm

    model = LiteLlm(
        model="openai/model-a",
        api_key="placeholder",
        base_url=DEFAULT_SAIA_BASE_URL,
        max_tokens=DEFAULT_SAIA_MAX_TOKENS,
    )
    bundle = create_incident_copilot_agent(model=model, history_path="unused.jsonl")
    try:
        assert bundle.agent.model is model
    finally:
        import asyncio

        asyncio.run(bundle.close())


def test_import_safety_without_saia_settings() -> None:
    import volt_vision.agent.saia as saia

    assert saia.load_saia_settings_from_env({}) is None


def test_probe_without_live_does_not_call_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_probe_module()
    import volt_vision.agent.saia as saia_module

    def fail_live(settings: object) -> int:
        raise AssertionError("live probe should not run")

    monkeypatch.setattr(module, "_run_live_probe", fail_live)
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(saia_module, "PROJECT_ROOT_ENV_PATH", tmp_path / "missing.env")

    assert module.main([]) == 0
    output = capsys.readouterr().out
    assert "configuration_status=not_configured" in output
    assert "key_configured=no" in output


def test_probe_uses_dotenv_without_printing_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_probe_module()
    import volt_vision.agent.saia as saia_module

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "VOLT_VISION_SAIA_API_KEY=test-secret-value",
                "VOLT_VISION_SAIA_BASE_URL=https://example.test/v1",
                "VOLT_VISION_SAIA_MODEL=glm-4.7",
                "VOLT_VISION_SAIA_MAX_TOKENS=1024",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.os, "environ", {})
    monkeypatch.setattr(saia_module, "PROJECT_ROOT_ENV_PATH", env_file)

    assert module.main([]) == 0

    output = capsys.readouterr().out
    assert "configuration_status=configured" in output
    assert "selected_raw_model_id=glm-4.7" in output
    assert "base_url_host_path=example.test/v1" in output
    assert "key_configured=yes" in output
    assert "test-secret-value" not in output


def test_live_probe_ready_stop_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_probe_module()
    settings = _probe_settings()

    result = module._run_live_probe(
        settings,
        openai_cls=_fake_openai_cls(content="ready", finish_reason="stop"),
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "live_status=succeeded" in output
    assert "finish_reason=stop" in output
    assert "content_present=True" in output
    assert "content_length=5" in output
    assert "ready" not in output
    assert "plain-secret" not in output


def test_live_probe_empty_length_fails_safely(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_probe_module()

    result = module._run_live_probe(
        _probe_settings(),
        openai_cls=_fake_openai_cls(content="", finish_reason="length"),
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "live_status=response_incomplete_or_unexpected" in output
    assert "finish_reason=length" in output
    assert "content_present=False" in output
    assert "content_length=0" in output


def test_live_probe_unexpected_content_fails_without_printing_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_probe_module()

    result = module._run_live_probe(
        _probe_settings(),
        openai_cls=_fake_openai_cls(content="not ready yet", finish_reason="stop"),
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "live_status=response_incomplete_or_unexpected" in output
    assert "content_present=True" in output
    assert "content_length=13" in output
    assert "not ready yet" not in output


def test_live_probe_network_exception_is_bounded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_probe_module()

    result = module._run_live_probe(
        _probe_settings(),
        openai_cls=_fake_openai_cls(raise_on_list=True),
    )

    output = capsys.readouterr().out
    assert result == 1
    assert output.strip() == "live_status=failed"


def test_gitignore_excludes_env_and_jsonl() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")

    assert ".env" in text
    assert "data/*.jsonl" in text


def _load_probe_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "probe_saia_for_test",
        Path("scripts") / "probe_saia.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _probe_settings() -> SaiaSettings:
    settings = load_saia_settings_from_env(
        env(
            VOLT_VISION_SAIA_API_KEY="plain-secret",
            VOLT_VISION_SAIA_MODEL="model-a",
        )
    )
    assert settings is not None
    return settings


def _fake_openai_cls(
    *,
    content: str = "ready",
    finish_reason: str = "stop",
    raise_on_list: bool = False,
) -> type[object]:
    class FakeModel:
        id = "model-a"

    class FakeModels:
        def list(self) -> object:
            if raise_on_list:
                raise RuntimeError("network secret detail")
            return type("ModelList", (), {"data": [FakeModel()]})()

    class FakeMessage:
        pass

    class FakeChoice:
        pass

    class FakeChatCompletions:
        def create(self, **_: object) -> object:
            message = FakeMessage()
            message.content = content
            choice = FakeChoice()
            choice.message = message
            choice.finish_reason = finish_reason
            return type("Completion", (), {"choices": [choice]})()

    class FakeChat:
        completions = FakeChatCompletions()

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.models = FakeModels()
            self.chat = FakeChat()

    return FakeOpenAI
