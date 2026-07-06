"""Manual live acceptance smoke for the chat-oriented SAIA copilot path."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from volt_vision.agent.chat import run_chat_copilot, validate_chat_text
from volt_vision.agent.models import CopilotChatResponse
from volt_vision.agent.policy import MANDATORY_CHAT_HEADINGS
from volt_vision.agent.saia import (
    SAIA_API_KEY_ENV_VAR,
    SAIA_MODEL_ENV_VAR,
    load_saia_settings_from_env,
    resolve_optional_saia_model_from_env,
    safe_saia_configuration_status,
)
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)

EXIT_SUCCESS_ADK = 0
EXIT_FAILURE = 1
EXIT_FALLBACK = 2

CURRENT_EVENT_ID = "current_synthetic_smoke"


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Run dry configuration smoke or manually requested live acceptance smoke."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    values = os.environ if environ is None else environ

    if not args.live:
        return _dry_run(values)
    return _live_run(values)


def _dry_run(environ: Mapping[str, str]) -> int:
    print(f"configuration_status={safe_saia_configuration_status(environ)}")
    print(f"selected_model_id={_selected_model_id(environ)}")
    print(f"key_configured={_yes_no(bool(environ.get(SAIA_API_KEY_ENV_VAR, '').strip()))}")
    return EXIT_SUCCESS_ADK


def _live_run(environ: Mapping[str, str]) -> int:
    configuration_status = safe_saia_configuration_status(environ)
    selected_model_id = _selected_model_id(environ)
    if configuration_status != "configured":
        print("live_status=not_configured")
        print(f"execution_mode={None}")
        print(f"fallback_reason={None}")
        print(f"selected_model_id={selected_model_id}")
        print("human_approval_required=no")
        print("assistant_message_character_count=0")
        print("preferred_section_format_observed=no")
        print("chat_safety_validation_passed=no")
        return EXIT_FAILURE

    try:
        settings = load_saia_settings_from_env(environ)
        if settings is None:
            print("live_status=not_configured")
            return EXIT_FAILURE
        model = resolve_optional_saia_model_from_env(environ)
        if model is None:
            print("live_status=not_configured")
            return EXIT_FAILURE
        with TemporaryDirectory() as tmp_dir:
            history_path = Path(tmp_dir) / "history.jsonl"
            _write_synthetic_history(history_path)
            response = run_chat_copilot(
                CURRENT_EVENT_ID,
                history_path=history_path,
                model=model,
            )
        _print_live_response(response, selected_model_id=settings.raw_model_id)
        if response.execution_mode == "adk":
            return EXIT_SUCCESS_ADK
        return EXIT_FALLBACK
    except Exception:
        print("live_status=failed")
        print("execution_mode=None")
        print("fallback_reason=None")
        print(f"selected_model_id={selected_model_id}")
        print("human_approval_required=no")
        print("assistant_message_character_count=0")
        print("preferred_section_format_observed=no")
        print("chat_safety_validation_passed=no")
        return EXIT_FAILURE


def _print_live_response(
    response: CopilotChatResponse,
    *,
    selected_model_id: str,
) -> None:
    preferred_format_observed = _preferred_section_format_observed(
        response.assistant_message
    )
    safety_passed = _chat_safety_validation_passed(
        response.assistant_message,
        event_id=response.event_id,
    )
    print(
        "live_status="
        + ("adk_accepted" if response.execution_mode == "adk" else "deterministic_fallback")
    )
    print(f"execution_mode={response.execution_mode}")
    print(f"fallback_reason={response.fallback_reason}")
    print(f"selected_model_id={selected_model_id}")
    for tool_call in response.tool_trace.tool_calls:
        print(
            "tool_trace="
            f"{tool_call.tool_name},{tool_call.source},"
            f"{tool_call.outcome},{tool_call.error_code}"
        )
    print(f"knowledge_source_ids={','.join(response.knowledge_source_ids)}")
    print(f"human_approval_required={_yes_no(response.human_approval_required)}")
    print(f"assistant_message_character_count={len(response.assistant_message)}")
    print(f"preferred_section_format_observed={_yes_no(preferred_format_observed)}")
    print(f"chat_safety_validation_passed={_yes_no(safety_passed)}")


def _write_synthetic_history(history_path: Path) -> None:
    append_monitoring_event(_synthetic_event("prior_synthetic_smoke_a", 60), history_path)
    append_monitoring_event(_synthetic_event("prior_synthetic_smoke_b", 120), history_path)
    append_monitoring_event(_synthetic_event(CURRENT_EVENT_ID, 180), history_path)


def _synthetic_event(event_id: str, seconds: int) -> MonitoringEvent:
    start = datetime(2026, 1, 1, 8, 0, tzinfo=UTC) + timedelta(seconds=seconds)
    end = start + timedelta(seconds=80)
    return MonitoringEvent(
        event_id=event_id,
        event_type="cycle_screening",
        event_timestamp=end,
        machine_id="CNC_SMOKE",
        candidate_segment_id="candidate",
        reference_segment_id="reference",
        status="suspected_deviation",
        recommended_action="manual_review_required",
        evidence="Normalized DTW distance compared with calibrated threshold.",
        normalized_dtw_distance=0.12,
        threshold=0.10,
        metrics=CycleMetrics(
            cycle_id="candidate",
            machine_id="CNC_SMOKE",
            start_timestamp=start,
            end_timestamp=end,
            duration_seconds=80,
            energy_kwh=0.25,
            average_power_kw=15,
            peak_power_kw=20,
            sample_count=2,
        ),
        indicators=ReferenceRelativeIndicators(
            reference_cycle_id="reference",
            candidate_cycle_id="candidate",
            duration_deviation_pct=12.0,
            energy_deviation_pct=18.0,
            peak_power_deviation_pct=22.0,
        ),
    )


def _preferred_section_format_observed(text: str) -> bool:
    lowered = text.lower()
    return all(heading.lower() in lowered for heading in MANDATORY_CHAT_HEADINGS)


def _chat_safety_validation_passed(text: str, *, event_id: str) -> bool:
    try:
        validate_chat_text(text, event_id=event_id)
    except ValueError:
        return False
    return True


def _selected_model_id(environ: Mapping[str, str]) -> str:
    return environ.get(SAIA_MODEL_ENV_VAR, "").strip() or "unset"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
