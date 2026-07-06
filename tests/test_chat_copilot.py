from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from volt_vision.agent import chat as chat_module
from volt_vision.agent.chat import (
    MANDATORY_CHAT_HEADINGS,
    build_chat_composition_context,
    build_deterministic_fallback_chat_response,
    run_chat_copilot,
    validate_chat_text,
)
from volt_vision.agent.models import (
    AgentChatExecutionOutcome,
    AgentRunTrace,
    CopilotChatResponse,
    ToolCallTrace,
)
from volt_vision.monitoring.event_log import append_monitoring_event

from test_mcp_services import make_event


class FakeBundle:
    def __init__(self) -> None:
        self.agent = object()
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


def valid_mcp_tool_calls() -> tuple[ToolCallTrace, ...]:
    return (
        ToolCallTrace(
            tool_name="get_event_metrics",
            source="mcp",
            outcome="succeeded",
            error_code=None,
        ),
        ToolCallTrace(
            tool_name="retrieve_maintenance_guidance",
            source="mcp",
            outcome="succeeded",
            error_code=None,
        ),
        ToolCallTrace(
            tool_name="find_similar_previous_events",
            source="mcp",
            outcome="succeeded",
            error_code=None,
        ),
    )


def valid_chat_text() -> str:
    return "\n".join(
        (
            MANDATORY_CHAT_HEADINGS[0],
            "This is a suspected deviation, not a confirmed diagnosis. Manual inspection recommended.",
            "",
            MANDATORY_CHAT_HEADINGS[1],
            "Possible contributing condition: verify local context and compare reviewed evidence.",
            "",
            MANDATORY_CHAT_HEADINGS[2],
            "Inspect according to local SOP and record observations for authorized review.",
            "",
            MANDATORY_CHAT_HEADINGS[3],
            "Was the selected cycle complete and comparable to the reference?",
            "",
            MANDATORY_CHAT_HEADINGS[4],
            "Escalate when recurrence, production impact, or local procedure requires review.",
        )
    )


def valid_unheaded_chat_text() -> str:
    return (
        "The screening observed a suspected deviation, not a confirmed diagnosis. "
        "The pattern should be reviewed as a possible contributing condition, "
        "with local context compared against the reference cycle. Suggested checks "
        "include inspecting according to local SOP, recording observations, and "
        "comparing recurrence context. Confirm locally whether the selected cycle "
        "was complete and whether material, fixture, or workpiece context changed. "
        "Escalate when recurrence, production impact, or local procedure requires "
        "authorized review."
    )


def test_chat_response_model_is_immutable_and_json_safe() -> None:
    trace = AgentRunTrace(
        event_id="event-1",
        execution_mode="deterministic_fallback",
        tool_names=(),
        tool_calls=(),
        fallback_reason="model_not_configured",
        completed=True,
    )
    response = CopilotChatResponse(
        event_id="event-1",
        execution_mode="deterministic_fallback",
        assistant_message=valid_chat_text(),
        knowledge_source_ids=("power_signature_review",),
        tool_trace=trace,
        human_approval_required=True,
        fallback_reason="model_not_configured",
    )

    with pytest.raises(ValidationError):
        response.assistant_message = "changed"  # type: ignore[misc]

    payload = response.model_dump_json()
    assert "api_key" not in payload.lower()
    assert "traceback" not in payload.lower()
    assert response.human_approval_required is True


def test_model_none_returns_readable_deterministic_fallback_chat(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("prior", seconds=30), history_path)
    append_monitoring_event(make_event("query", seconds=80), history_path)

    response = run_chat_copilot("query", history_path=history_path, model=None)

    assert response.execution_mode == "deterministic_fallback"
    assert response.fallback_reason == "model_not_configured"
    assert response.human_approval_required is True
    assert response.knowledge_source_ids == (
        "power_signature_review",
        "cycle_duration_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    )
    assert "suspected deviation" in response.assistant_message.lower()
    assert "not a confirmed diagnosis" in response.assistant_message.lower()
    assert "query" not in response.assistant_message
    assert tuple(call.source for call in response.tool_trace.tool_calls) == (
        "deterministic_service",
        "deterministic_service",
        "deterministic_service",
    )


def test_within_normal_band_returns_not_triggered_trace_without_live_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(
        make_event("normal", seconds=60, status="within_normal_band"),
        history_path,
    )

    def fail_factory(**_: object) -> object:
        raise AssertionError("ADK factory should not be called")

    monkeypatch.setattr(chat_module, "create_chat_copilot_agent", fail_factory)
    response = run_chat_copilot(
        "normal",
        history_path=history_path,
        model="configured-model",
    )

    assert response.execution_mode == "deterministic_fallback"
    assert response.tool_trace.execution_mode == "not_triggered"
    assert response.tool_trace.tool_calls == ()
    assert response.fallback_reason is None


def test_valid_final_chat_and_ordered_mcp_trace_returns_adk_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    fake_bundle = FakeBundle()
    monkeypatch.setattr(
        chat_module,
        "create_chat_copilot_agent",
        lambda **_: fake_bundle,
    )

    response = run_chat_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: AgentChatExecutionOutcome(
            final_text="ignored stage one text",
            tool_calls=valid_mcp_tool_calls(),
        ),
        chat_composer=lambda *_: valid_chat_text(),
    )

    assert response.execution_mode == "adk"
    assert response.assistant_message == valid_chat_text()
    assert response.knowledge_source_ids == (
        "power_signature_review",
        "cycle_duration_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    )
    assert response.human_approval_required is True
    assert response.tool_trace.tool_names == (
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
    )
    assert fake_bundle.close_count == 1


def test_safe_natural_language_without_headings_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    monkeypatch.setattr(
        chat_module,
        "create_chat_copilot_agent",
        lambda **_: FakeBundle(),
    )

    response = run_chat_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: AgentChatExecutionOutcome(
            final_text="ignored stage one text",
            tool_calls=valid_mcp_tool_calls(),
        ),
        chat_composer=lambda *_: valid_unheaded_chat_text(),
    )

    assert response.execution_mode == "adk"
    assert response.assistant_message == valid_unheaded_chat_text()


@pytest.mark.parametrize(
    "bad_text",
    [
        "",
        "x" * 1801,
        valid_chat_text().replace("suspected deviation, not a confirmed diagnosis", "screening result"),
        valid_chat_text() + "\nget_event_metrics",
        valid_chat_text() + "\nquery",
        valid_chat_text() + "\nsk-testsecretvalue",
        valid_chat_text() + "\nD:\\secret\\path",
        valid_chat_text() + "\nThis is a confirmed fault.",
        valid_chat_text() + "\nReplace the tool.",
        valid_chat_text() + "\nCreate a maintenance ticket.",
        valid_chat_text() + "\nStop the machine.",
        valid_chat_text() + "\nParameter tuning is required.",
    ],
)
def test_invalid_final_chat_falls_back_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_text: str,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    monkeypatch.setattr(
        chat_module,
        "create_chat_copilot_agent",
        lambda **_: FakeBundle(),
    )

    response = run_chat_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: AgentChatExecutionOutcome(
            final_text="ignored stage one text",
            tool_calls=valid_mcp_tool_calls(),
        ),
        chat_composer=lambda *_: bad_text,
    )

    assert response.execution_mode == "deterministic_fallback"
    assert response.fallback_reason == "model_execution_failed"
    assert "query" not in response.assistant_message
    assert all(call.source == "deterministic_service" for call in response.tool_trace.tool_calls)


def test_adk_chat_cannot_override_authoritative_local_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    event = make_event("query", seconds=60)
    append_monitoring_event(event, history_path)
    monkeypatch.setattr(
        chat_module,
        "create_chat_copilot_agent",
        lambda **_: FakeBundle(),
    )

    response = run_chat_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: {
            "final_text": "ignored stage one text",
            "tool_calls": tuple(call.model_dump() for call in valid_mcp_tool_calls()),
            "knowledge_source_ids": ("invented",),
            "human_approval_required": False,
            "status": "within_normal_band",
            "deterministic_evidence": ("invented",),
        },
        chat_composer=lambda *_: valid_chat_text(),
    )

    assert response.execution_mode == "adk"
    assert response.knowledge_source_ids == (
        "power_signature_review",
        "cycle_duration_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    )
    assert response.human_approval_required is True
    assert event.status == "suspected_deviation"


def test_missing_or_bad_mcp_trace_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    monkeypatch.setattr(
        chat_module,
        "create_chat_copilot_agent",
        lambda **_: FakeBundle(),
    )
    wrong_order = (
        valid_mcp_tool_calls()[1],
        valid_mcp_tool_calls()[0],
        valid_mcp_tool_calls()[2],
    )

    response = run_chat_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: AgentChatExecutionOutcome(
            final_text=valid_chat_text(),
            tool_calls=wrong_order,
        ),
        chat_composer=lambda *_: pytest.fail("composer must not be called"),
    )

    assert response.execution_mode == "deterministic_fallback"
    assert response.fallback_reason == "model_execution_failed"


def test_fallback_chat_references_only_safe_local_facts(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)

    response = build_deterministic_fallback_chat_response("query", history_path)
    lowered = response.model_dump_json().lower()

    assert response.knowledge_source_ids
    assert "get_event_metrics" in lowered
    assert "query" not in response.assistant_message
    assert "raw_samples" not in lowered
    assert "csv" not in lowered
    assert str(history_path).lower() not in lowered


def test_validate_chat_text_accepts_safe_reference_text() -> None:
    validate_chat_text(valid_chat_text(), event_id="not-present")


def test_stage_two_not_called_when_mcp_trace_missing_or_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    monkeypatch.setattr(
        chat_module,
        "create_chat_copilot_agent",
        lambda **_: FakeBundle(),
    )
    failed_trace = (
        valid_mcp_tool_calls()[0],
        ToolCallTrace(
            tool_name="retrieve_maintenance_guidance",
            source="mcp",
            outcome="failed",
            error_code="tool_execution_failed",
        ),
        valid_mcp_tool_calls()[2],
    )

    response = run_chat_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: AgentChatExecutionOutcome(
            final_text="ignored stage one text",
            tool_calls=failed_trace,
        ),
        chat_composer=lambda *_: pytest.fail("composer must not be called"),
    )

    assert response.execution_mode == "deterministic_fallback"
    assert response.fallback_reason == "model_execution_failed"


def test_composer_failure_returns_deterministic_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    monkeypatch.setattr(
        chat_module,
        "create_chat_copilot_agent",
        lambda **_: FakeBundle(),
    )

    def fail_composer(*_: object) -> str:
        raise RuntimeError("hidden composer failure")

    response = run_chat_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: AgentChatExecutionOutcome(
            final_text="ignored stage one text",
            tool_calls=valid_mcp_tool_calls(),
        ),
        chat_composer=fail_composer,
    )

    payload = response.model_dump_json()
    assert response.execution_mode == "deterministic_fallback"
    assert response.fallback_reason == "model_execution_failed"
    assert "hidden composer failure" not in payload


def test_composition_context_contains_only_bounded_safe_fields(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("prior", seconds=30), history_path)
    append_monitoring_event(make_event("query", seconds=60), history_path)

    context = build_chat_composition_context("query", history_path)
    payload = context.model_dump_json().lower()

    assert context.screening_status == "suspected_deviation"
    assert context.knowledge_source_ids == (
        "power_signature_review",
        "cycle_duration_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    )
    forbidden = (
        "query",
        "cnc_test",
        "raw_samples",
        "csv",
        str(history_path).lower(),
        "api_key",
        "traceback",
    )
    for fragment in forbidden:
        assert fragment not in payload
