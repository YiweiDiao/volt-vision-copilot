"""Bounded live diagnostics for chat-oriented SAIA copilot acceptance."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from volt_vision.agent.adk_execution import (
    ChatDiagnosticExecutionOutcome,
    execute_chat_copilot_diagnostic_bundle,
)
from volt_vision.agent.adk_factory import create_chat_copilot_agent
from volt_vision.agent.chat import (
    ComposerResponseObservation,
    build_chat_composition_context,
    compose_chat_with_model_observed,
)
from volt_vision.agent.models import AgentRunTrace, CopilotChatResponse, ToolCallTrace
from volt_vision.agent.policy import APPROVED_MCP_TOOL_NAMES, MANDATORY_CHAT_HEADINGS
from volt_vision.agent.saia import (
    SAIA_API_KEY_ENV_VAR,
    SAIA_MODEL_ENV_VAR,
    resolve_optional_saia_model_from_env,
    safe_saia_configuration_status,
)
from volt_vision.mcp_server.services import retrieve_maintenance_guidance
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)

DiagnosticStage = Literal[
    "bundle_creation_failed",
    "adk_run_failed",
    "mcp_trace_rejected",
    "composer_not_called",
    "composer_failed",
    "composer_response_length_limited",
    "composer_reasoning_only",
    "composer_visible_text_missing",
    "composer_alternate_text_unhandled",
    "no_composer_text",
    "chat_length_rejected",
    "chat_uncertainty_rejected",
    "chat_internal_reference_rejected",
    "chat_safety_rejected",
    "chat_response_construction_failed",
    "accepted",
]

MAX_CHAT_CHARACTERS = 1800
CURRENT_EVENT_ID = "current_synthetic_smoke"
HEADING_LABELS = (
    ("screening_observed", MANDATORY_CHAT_HEADINGS[0]),
    ("possible_conditions", MANDATORY_CHAT_HEADINGS[1]),
    ("suggested_steps", MANDATORY_CHAT_HEADINGS[2]),
    ("questions_local", MANDATORY_CHAT_HEADINGS[3]),
    ("escalation", MANDATORY_CHAT_HEADINGS[4]),
)
SAFETY_CATEGORY_PATTERNS = {
    "diagnosis": (
        r"\bconfirmed\s+(fault|failure|tool wear|root cause|diagnosis)\b",
        r"\b(root cause is|root cause:|diagnosis:)\b",
        r"\btool wear\s+(is|was|caused|likely|confirmed)\b",
    ),
    "repair_or_replacement": (r"\b(repair|replace|replacement)\b",),
    "shutdown_or_control": (
        r"\b(shutdown|stop|halt)\b",
        r"\b(control|command)\s+the\s+machine\b",
    ),
    "ticket": (r"\b(ticket|maintenance ticket)\b",),
    "tuning": (r"\btuning\b",),
    "plc_scada_mes": (r"\b(plc|scada|mes|opc-ua|modbus)\b",),
}
INTERNAL_REFERENCE_PATTERNS = (
    r"\b(get_event_metrics|retrieve_maintenance_guidance|find_similar_previous_events)\b",
    r"\bsk-[A-Za-z0-9_-]{8,}\b",
    r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b",
    r"\b[A-Z]:\\",
    r"(?<!\w)/(?:home|tmp|var|etc|users?)/",
    r"\b(traceback|stack trace|exception|api key|secret|token)\b",
)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
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
    return 0


def _live_run(environ: Mapping[str, str]) -> int:
    selected_model_id = _selected_model_id(environ)
    if safe_saia_configuration_status(environ) != "configured":
        _print_diagnostic(
            _empty_diagnostic("bundle_creation_failed", selected_model_id),
        )
        return 1

    with TemporaryDirectory() as tmp_dir:
        history_path = Path(tmp_dir) / "history.jsonl"
        _write_synthetic_history(history_path)
        try:
            model = resolve_optional_saia_model_from_env(environ)
            bundle = create_chat_copilot_agent(model=model, history_path=history_path)
        except Exception:
            _print_diagnostic(
                _empty_diagnostic("bundle_creation_failed", selected_model_id),
            )
            return 1

        try:
            outcome = execute_chat_copilot_diagnostic_bundle(bundle, CURRENT_EVENT_ID)
        except Exception:
            _print_diagnostic(
                _empty_diagnostic("adk_run_failed", selected_model_id),
            )
            return 1

        diagnostic = _diagnose_outcome(
            outcome,
            selected_model_id=selected_model_id,
        )
        if diagnostic.stage == "composer_not_called":
            try:
                context = build_chat_composition_context(CURRENT_EVENT_ID, history_path)
                composer_text, observation = compose_chat_with_model_observed(
                    model,
                    context,
                )
            except Exception:
                diagnostic.stage = "composer_failed"
                diagnostic.stage2_composer_called = True
                _print_diagnostic(diagnostic)
                return 1
            diagnostic = _diagnose_composer_text(
                outcome,
                selected_model_id=selected_model_id,
                composer_text=composer_text,
                composer_observation=observation,
                history_path=history_path,
            )
        _print_diagnostic(diagnostic)
        return 0 if diagnostic.stage == "accepted" else 1


class DiagnosticReport:
    def __init__(
        self,
        *,
        stage: DiagnosticStage,
        selected_model_id: str,
        adk_event_count: int = 0,
        tool_calls: tuple[ToolCallTrace, ...] = (),
        composer_text: str | None = None,
        composer_observation: ComposerResponseObservation | None = None,
        stage1_mcp_trace_passed: bool = False,
        stage2_composer_called: bool = False,
        chat_response_constructed: bool = False,
    ) -> None:
        self.stage = stage
        self.selected_model_id = selected_model_id
        self.adk_event_count = adk_event_count
        self.tool_calls = tool_calls
        self.composer_text = composer_text
        self.composer_observation = composer_observation
        self.stage1_mcp_trace_passed = stage1_mcp_trace_passed
        self.stage2_composer_called = stage2_composer_called
        self.chat_response_constructed = chat_response_constructed


def _diagnose_outcome(
    outcome: ChatDiagnosticExecutionOutcome,
    *,
    selected_model_id: str,
) -> DiagnosticReport:
    stage1_passed = _mcp_trace_accepted(outcome.tool_calls)
    report = DiagnosticReport(
        stage="composer_not_called" if stage1_passed else "mcp_trace_rejected",
        selected_model_id=selected_model_id,
        adk_event_count=outcome.adk_event_count,
        tool_calls=outcome.tool_calls,
        stage1_mcp_trace_passed=stage1_passed,
    )
    return report


def _diagnose_composer_text(
    outcome: ChatDiagnosticExecutionOutcome,
    *,
    selected_model_id: str,
    composer_text: str | None,
    composer_observation: ComposerResponseObservation | None = None,
    history_path: Path,
) -> DiagnosticReport:
    observation = composer_observation or _default_observation(composer_text)
    report = DiagnosticReport(
        stage="accepted",
        selected_model_id=selected_model_id,
        adk_event_count=outcome.adk_event_count,
        tool_calls=outcome.tool_calls,
        composer_text=composer_text,
        composer_observation=observation,
        stage1_mcp_trace_passed=True,
        stage2_composer_called=True,
    )
    if composer_text is None or not composer_text.strip():
        report.stage = _empty_composer_stage(observation)
        return report
    if not _chat_length_within_limit(composer_text):
        report.stage = "chat_length_rejected"
        return report
    if not _uncertainty_phrase_present(composer_text):
        report.stage = "chat_uncertainty_rejected"
        return report
    if _internal_reference_detected(composer_text, event_id=CURRENT_EVENT_ID):
        report.stage = "chat_internal_reference_rejected"
        return report
    if _safety_category_hits(composer_text) != ("none",):
        report.stage = "chat_safety_rejected"
        return report
    if not _construct_chat_response(outcome, composer_text, history_path):
        report.stage = "chat_response_construction_failed"
        return report
    report.chat_response_constructed = True
    return report


def _construct_chat_response(
    outcome: ChatDiagnosticExecutionOutcome,
    composer_text: str,
    history_path: Path,
) -> bool:
    try:
        trace = AgentRunTrace(
            event_id=CURRENT_EVENT_ID,
            execution_mode="adk",
            tool_names=tuple(tool_call.tool_name for tool_call in outcome.tool_calls),
            tool_calls=outcome.tool_calls,
            fallback_reason=None,
            completed=True,
        )
        CopilotChatResponse(
            event_id=CURRENT_EVENT_ID,
            execution_mode="adk",
            assistant_message=composer_text.strip(),
            knowledge_source_ids=tuple(
                item.guidance_id
                for item in retrieve_maintenance_guidance(CURRENT_EVENT_ID, history_path)
            ),
            tool_trace=trace,
            human_approval_required=True,
            fallback_reason=None,
        )
    except Exception:
        return False
    return True


def _empty_diagnostic(
    stage: DiagnosticStage,
    selected_model_id: str,
) -> DiagnosticReport:
    return DiagnosticReport(stage=stage, selected_model_id=selected_model_id)


def _empty_composer_stage(
    observation: ComposerResponseObservation,
) -> DiagnosticStage:
    if observation.finish_reason == "length":
        return "composer_response_length_limited"
    if (
        observation.reasoning_content_nonempty == "yes"
        and observation.visible_content_nonempty == "no"
    ):
        return "composer_reasoning_only"
    if (
        observation.alternate_text_field_present == "yes"
        and observation.visible_content_nonempty == "no"
    ):
        return "composer_alternate_text_unhandled"
    if (
        observation.response_received == "yes"
        and observation.visible_content_present == "no"
    ):
        return "composer_visible_text_missing"
    return "no_composer_text"


def _default_observation(text: str | None) -> ComposerResponseObservation:
    return ComposerResponseObservation(
        response_received=_yes_no(text is not None),
        finish_reason="absent_or_unknown",
        visible_content_present=_yes_no(text is not None),
        visible_content_nonempty=_yes_no(bool(text and text.strip())),
        reasoning_content_present="no",
        reasoning_content_nonempty="no",
        completion_token_usage="unavailable",
        alternate_text_field_present="no",
    )


def _print_diagnostic(report: DiagnosticReport) -> None:
    text = report.composer_text
    print(f"diagnostic_stage={report.stage}")
    print(f"selected_model_id={report.selected_model_id}")
    print(f"adk_event_count={report.adk_event_count}")
    print(f"approved_mcp_function_response_count={len(report.tool_calls)}")
    for tool_call in report.tool_calls:
        print(
            "mcp_trace="
            f"{tool_call.tool_name},{tool_call.source},"
            f"{tool_call.outcome},{tool_call.error_code}"
        )
    print(f"stage1_mcp_trace_passed={_yes_no(report.stage1_mcp_trace_passed)}")
    print(f"stage2_composer_called={_yes_no(report.stage2_composer_called)}")
    observation = report.composer_observation or _default_observation(text)
    print(f"response_received={observation.response_received}")
    print(f"finish_reason={observation.finish_reason}")
    print(f"visible_content_present={observation.visible_content_present}")
    print(f"visible_content_nonempty={observation.visible_content_nonempty}")
    print(f"reasoning_content_present={observation.reasoning_content_present}")
    print(f"reasoning_content_nonempty={observation.reasoning_content_nonempty}")
    print(f"completion_token_usage={observation.completion_token_usage}")
    print(f"alternate_text_field_present={observation.alternate_text_field_present}")
    print(f"composer_text_present={_yes_no(text is not None)}")
    print(f"composer_text_nonempty={_yes_no(bool(text and text.strip()))}")
    print(f"chat_length_within_limit={_yes_no(_chat_length_within_limit(text))}")
    print(
        "preferred_section_format_observed="
        f"{_yes_no(_preferred_section_format_observed(text))}"
    )
    for label, present in _heading_presence(text).items():
        print(f"heading_presence:{label}={_yes_no(present)}")
    print(f"heading_order_valid={_yes_no(_heading_order_valid(text))}")
    print(f"uncertainty_phrase_present={_yes_no(_uncertainty_phrase_present(text))}")
    print(
        "internal_reference_detected="
        f"{_yes_no(_internal_reference_detected(text, event_id=CURRENT_EVENT_ID))}"
    )
    print(f"safety_category_hits={','.join(_safety_category_hits(text))}")
    print(f"chat_response_constructed={_yes_no(report.chat_response_constructed)}")


def _mcp_trace_accepted(tool_calls: tuple[ToolCallTrace, ...]) -> bool:
    first_observed: list[str] = []
    succeeded: set[str] = set()
    for tool_call in tool_calls:
        if tool_call.source != "mcp":
            return False
        if tool_call.tool_name not in APPROVED_MCP_TOOL_NAMES:
            return False
        if tool_call.outcome != "succeeded":
            return False
        if tool_call.tool_name not in succeeded:
            first_observed.append(tool_call.tool_name)
        succeeded.add(tool_call.tool_name)
    required = list(APPROVED_MCP_TOOL_NAMES)
    return first_observed[: len(required)] == required and set(required).issubset(
        succeeded
    )


def _chat_length_within_limit(text: str | None) -> bool:
    return text is not None and len(text.strip()) <= MAX_CHAT_CHARACTERS


def _heading_presence(text: str | None) -> dict[str, bool]:
    value = text or ""
    return {
        label: re.search(rf"(^|\n){re.escape(heading)}\s*(\n|$)", value) is not None
        for label, heading in HEADING_LABELS
    }


def _heading_order_valid(text: str | None) -> bool:
    if text is None:
        return False
    positions: list[int] = []
    for _, heading in HEADING_LABELS:
        match = re.search(rf"(^|\n){re.escape(heading)}\s*(\n|$)", text)
        if match is None:
            return False
        positions.append(match.start())
    return positions == sorted(positions)


def _preferred_section_format_observed(text: str | None) -> bool:
    return all(_heading_presence(text).values()) and _heading_order_valid(text)


def _uncertainty_phrase_present(text: str | None) -> bool:
    value = (text or "").lower()
    return "suspected deviation" in value and "not a confirmed diagnosis" in value


def _internal_reference_detected(text: str | None, *, event_id: str) -> bool:
    value = text or ""
    if event_id and re.search(re.escape(event_id), value, flags=re.IGNORECASE):
        return True
    return any(
        re.search(pattern, value, flags=re.IGNORECASE)
        for pattern in INTERNAL_REFERENCE_PATTERNS
    )


def _safety_category_hits(text: str | None) -> tuple[str, ...]:
    value = (text or "").lower().replace("not a confirmed diagnosis", "")
    hits = [
        category
        for category, patterns in SAFETY_CATEGORY_PATTERNS.items()
        if any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)
    ]
    return tuple(hits) if hits else ("none",)


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


def _selected_model_id(environ: Mapping[str, str]) -> str:
    return environ.get(SAIA_MODEL_ENV_VAR, "").strip() or "unset"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
