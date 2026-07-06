"""Configuration and optional live probe for the SAIA provider."""

from __future__ import annotations

import argparse
import os
from urllib.parse import urlparse

from volt_vision.agent.saia import (
    SAIA_API_KEY_ENV_VAR,
    SAIA_BASE_URL_ENV_VAR,
    SAIA_MAX_TOKENS_ENV_VAR,
    SAIA_MODEL_ENV_VAR,
    load_saia_settings_from_env,
    safe_saia_configuration_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe SAIA configuration safely.")
    parser.add_argument("--live", action="store_true", help="perform live SAIA checks")
    args = parser.parse_args(argv)
    status = safe_saia_configuration_status()
    settings = None
    try:
        settings = load_saia_settings_from_env()
    except ValueError:
        settings = None

    _print_configuration(status, settings)
    if not args.live:
        return 0
    if settings is None:
        print("live_status=not_configured")
        return 1
    return _run_live_probe(settings)


def _print_configuration(status: str, settings: object | None) -> None:
    raw_model_id = getattr(settings, "raw_model_id", None) or _clean(
        os.environ.get(SAIA_MODEL_ENV_VAR)
    )
    base_url = getattr(settings, "base_url", None) or _clean(
        os.environ.get(SAIA_BASE_URL_ENV_VAR)
    )
    max_tokens = getattr(settings, "max_tokens", None) or _clean(
        os.environ.get(SAIA_MAX_TOKENS_ENV_VAR)
    )
    parsed = urlparse(base_url or "")
    host_path = f"{parsed.netloc}{parsed.path}" if parsed.netloc else "not_configured"
    print(f"configuration_status={status}")
    print(f"selected_raw_model_id={raw_model_id or 'not_configured'}")
    print(f"base_url_host_path={host_path}")
    print(f"max_tokens={max_tokens or 'not_configured'}")
    key_configured = getattr(settings, "api_key", None) is not None or _clean(
        os.environ.get(SAIA_API_KEY_ENV_VAR)
    )
    print(f"key_configured={'yes' if key_configured else 'no'}")


def _run_live_probe(settings: object, *, openai_cls: object | None = None) -> int:
    try:
        if openai_cls is None:
            from openai import OpenAI

            openai_cls = OpenAI

        client = openai_cls(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
        )
        models = client.models.list()
        model_ids = {item.id for item in models.data}
        model_present = settings.raw_model_id in model_ids
        if not model_present:
            print("live_status=model_not_found")
            print(f"model_id={settings.raw_model_id}")
            return 1
        response = client.chat.completions.create(
            model=settings.raw_model_id,
            messages=[{"role": "user", "content": "Reply with the word ready."}],
            max_tokens=settings.max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        print(f"model_id={settings.raw_model_id}")
        print(f"finish_reason={choice.finish_reason}")
        print(f"content_present={bool(content)}")
        print(f"content_length={len(content)}")
        if choice.finish_reason == "stop" and content.strip().casefold() == "ready":
            print("live_status=succeeded")
            return 0
        print("live_status=response_incomplete_or_unexpected")
        return 1
    except Exception:
        print("live_status=failed")
        return 1


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


if __name__ == "__main__":
    raise SystemExit(main())
