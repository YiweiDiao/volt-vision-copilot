"""Optional SAIA OpenAI-compatible provider adapter for ADK LiteLlm."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer

DEFAULT_SAIA_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_SAIA_MAX_TOKENS = 1024
SAIA_API_KEY_ENV_VAR = "VOLT_VISION_SAIA_API_KEY"
SAIA_MODEL_ENV_VAR = "VOLT_VISION_SAIA_MODEL"
SAIA_BASE_URL_ENV_VAR = "VOLT_VISION_SAIA_BASE_URL"
SAIA_MAX_TOKENS_ENV_VAR = "VOLT_VISION_SAIA_MAX_TOKENS"
PROJECT_ROOT_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
_DEFAULT_ENV_FILE = object()

SaiaConfigurationStatus = Literal[
    "not_configured",
    "configured",
    "invalid_configuration",
]


class SaiaSettings(BaseModel):
    """Runtime-only SAIA settings with redacted secret representation."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    raw_model_id: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    max_tokens: int = Field(ge=1)

    @field_serializer("api_key", when_used="json")
    def serialize_api_key(self, value: SecretStr) -> str:
        return value.get_secret_value() and "**********"

    @property
    def litellm_route(self) -> str:
        """Return the LiteLLM OpenAI-compatible route."""

        return f"openai/{self.raw_model_id}"


def load_saia_settings_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    env_file: str | Path | None | object = _DEFAULT_ENV_FILE,
) -> SaiaSettings | None:
    """Load SAIA settings from environment-like mapping without side effects."""

    values = _merged_runtime_values(environ, env_file)
    api_key = _clean_optional(values.get(SAIA_API_KEY_ENV_VAR))
    raw_model_id = _clean_optional(values.get(SAIA_MODEL_ENV_VAR))
    if api_key is None or raw_model_id is None:
        return None

    base_url = _clean_optional(values.get(SAIA_BASE_URL_ENV_VAR)) or DEFAULT_SAIA_BASE_URL
    _validate_base_url(base_url)
    max_tokens = _parse_max_tokens(values.get(SAIA_MAX_TOKENS_ENV_VAR))
    return SaiaSettings(
        api_key=SecretStr(api_key),
        raw_model_id=raw_model_id,
        base_url=base_url,
        max_tokens=max_tokens,
    )


def safe_saia_configuration_status(
    environ: Mapping[str, str] | None = None,
    *,
    env_file: str | Path | None | object = _DEFAULT_ENV_FILE,
) -> SaiaConfigurationStatus:
    """Return only a bounded SAIA configuration status label."""

    try:
        settings = load_saia_settings_from_env(environ, env_file=env_file)
    except ValueError:
        return "invalid_configuration"
    if settings is None:
        return "not_configured"
    return "configured"


def resolve_optional_saia_model_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    env_file: str | Path | None | object = _DEFAULT_ENV_FILE,
) -> object | None:
    """Return a SAIA LiteLlm model only when configuration is fully valid."""

    try:
        settings = load_saia_settings_from_env(environ, env_file=env_file)
    except ValueError as exc:
        raise ValueError("invalid SAIA configuration") from exc
    if settings is None:
        return None
    return create_saia_litellm_model(settings)


def create_saia_litellm_model(
    settings: SaiaSettings,
    *,
    lite_llm_cls: type[Any] | None = None,
) -> object:
    """Create the official ADK LiteLlm wrapper for the SAIA endpoint."""

    if lite_llm_cls is None:
        from google.adk.models.lite_llm import LiteLlm

        lite_llm_cls = LiteLlm
    return lite_llm_cls(
        model=settings.litellm_route,
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
        max_tokens=settings.max_tokens,
    )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _merged_runtime_values(
    environ: Mapping[str, str] | None,
    env_file: str | Path | None | object,
) -> dict[str, str]:
    values: dict[str, str] = {}
    resolved_env_file = _resolve_env_file(environ, env_file)
    if resolved_env_file is not None:
        path = Path(resolved_env_file)
        if path.exists():
            for key, value in dotenv_values(path, encoding="utf-8").items():
                if value is not None:
                    values[key] = value
    source = environ if environ is not None else __import__("os").environ
    for key, value in source.items():
        values[key] = value
    return values


def _resolve_env_file(
    environ: Mapping[str, str] | None,
    env_file: str | Path | None | object,
) -> str | Path | None:
    if env_file is _DEFAULT_ENV_FILE:
        return PROJECT_ROOT_ENV_PATH if environ is None else None
    if env_file is None:
        return None
    if isinstance(env_file, (str, Path)):
        return env_file
    raise TypeError("env_file must be a path or None")


def _parse_max_tokens(value: str | None) -> int:
    cleaned = _clean_optional(value)
    if cleaned is None:
        return DEFAULT_SAIA_MAX_TOKENS
    try:
        max_tokens = int(cleaned)
    except ValueError as exc:
        raise ValueError("invalid SAIA max token setting") from exc
    if max_tokens <= 0:
        raise ValueError("invalid SAIA max token setting")
    return max_tokens


def _validate_base_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("invalid SAIA base URL")
