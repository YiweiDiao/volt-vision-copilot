from __future__ import annotations

from pathlib import Path

import pytest

from volt_vision.agent import runner as runner_module
from volt_vision.agent.adk_execution import _tool_traces_from_event
from volt_vision.agent.adk_factory import create_incident_copilot_agent
from volt_vision.agent.fallback import build_deterministic_fallback_result
from volt_vision.agent.models import AgentExecutionOutcome, ToolCallTrace
from volt_vision.agent.policy import CORE_LIMITATIONS
from volt_vision.agent.runner import run_incident_copilot
from volt_vision.mcp_server.services import EVENT_NOT_FOUND_MESSAGE, EventNotFoundError
from volt_vision.monitoring.event_log import append_monitoring_event

from test_mcp_services import make_event


class FakeBundle:
    def __init__(self) -> None:
        self.agent = object()
        self.tool_names = (
            "get_event_metrics",
            "retrieve_maintenance_guidance",
            "find_similar_previous_events",
        )
        self.close_count = 0
        self.fail_close = False

    async def close(self) -> None:
        self.close_count += 1
        if self.fail_close:
            raise RuntimeError("close failed with sensitive details")


class FakeAdkEvent:
    def __init__(self, *responses: object) -> None:
        self._responses = responses

    def get_function_responses(self) -> tuple[object, ...]:
        return self._responses


class FakeFunctionResponse:
    def __init__(self, name: str) -> None:
        self.name = name
        self.response: dict[str, object] = {}


def make_adk_outcome(payload: object, tool_calls: tuple[ToolCallTrace, ...] | None = None) -> AgentExecutionOutcome:
    if hasattr(payload, "model_dump"):
        output_payload = payload.model_dump(mode="python")
    else:
        output_payload = payload
    return AgentExecutionOutcome(
        output_payload=output_payload,
        tool_calls=valid_mcp_tool_calls() if tool_calls is None else tool_calls,
    )


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


def test_within_normal_band_event_returns_not_triggered_without_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    event = make_event("normal", seconds=60, status="within_normal_band")
    append_monitoring_event(event, history_path)

    def fail_executor(*_: object) -> object:
        raise AssertionError("model should not be invoked")

    def fail_factory(**_: object) -> object:
        raise AssertionError("factory should not be invoked")

    monkeypatch.setattr(runner_module, "create_incident_copilot_agent", fail_factory)
    result = run_incident_copilot(
        "normal",
        history_path=history_path,
        model="unused-model",
        agent_executor=fail_executor,
    )

    assert result.trace.execution_mode == "not_triggered"
    assert result.trace.tool_names == ()
    assert result.trace.tool_calls == ()
    assert result.trace.fallback_reason is None
    assert result.recommendation.human_approval_required is True
    assert result.recommendation.guidance_ids == ()


def test_suspected_deviation_without_model_uses_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)

    def fail_factory(**_: object) -> object:
        raise AssertionError("factory should not be invoked")

    monkeypatch.setattr(runner_module, "create_incident_copilot_agent", fail_factory)
    result = run_incident_copilot("query", history_path=history_path)

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_not_configured"
    assert result.recommendation.event_id == "query"


def test_factory_construction_failure_returns_safe_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)

    def failing_factory(**_: object) -> object:
        raise RuntimeError(f"factory leaked path {history_path}")

    monkeypatch.setattr(runner_module, "create_incident_copilot_agent", failing_factory)
    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
    )

    payload = result.model_dump_json()
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"
    assert "factory leaked" not in payload
    assert str(history_path) not in payload


def test_injected_executor_success_produces_adk_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    recommendation = build_deterministic_fallback_result(
        "query",
        history_path,
    ).recommendation
    fake_bundle = FakeBundle()
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: fake_bundle,
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(recommendation),
    )

    assert result.trace.execution_mode == "adk"
    assert result.trace.tool_names == fake_bundle.tool_names
    assert result.trace.tool_calls == valid_mcp_tool_calls()
    assert result.trace.fallback_reason is None
    assert result.recommendation == recommendation
    assert fake_bundle.close_count == 1


def test_model_execution_failure_falls_back_without_exception_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    fake_bundle = FakeBundle()
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: fake_bundle,
    )

    def failing_executor(*_: object) -> object:
        raise RuntimeError(f"secret failure at {history_path}")

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=failing_executor,
    )

    payload = result.model_dump_json()
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"
    assert "secret failure" not in payload
    assert str(history_path) not in payload
    assert "traceback" not in payload.lower()
    assert fake_bundle.close_count == 1


def test_close_failure_after_success_returns_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    recommendation = build_deterministic_fallback_result(
        "query",
        history_path,
    ).recommendation
    fake_bundle = FakeBundle()
    fake_bundle.fail_close = True
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: fake_bundle,
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(recommendation),
    )

    payload = result.model_dump_json()
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"
    assert "close failed" not in payload
    assert fake_bundle.close_count == 1


def test_default_executor_selected_when_model_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    recommendation = build_deterministic_fallback_result(
        "query",
        history_path,
    ).recommendation
    fake_bundle = FakeBundle()
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: fake_bundle,
    )
    calls: list[str] = []

    def fake_default_executor(agent: object, event_id: str) -> object:
        assert agent is fake_bundle.agent
        calls.append(event_id)
        return recommendation

    monkeypatch.setattr(
        runner_module,
        "execute_incident_copilot_bundle",
        lambda bundle, event_id, *, agent_executor=None: make_adk_outcome(
            fake_default_executor(bundle.agent, event_id)
        ),
    )
    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
    )

    assert calls == ["query"]
    assert result.trace.execution_mode == "adk"
    assert fake_bundle.close_count == 0


def test_invalid_adk_output_schema_returns_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome({"event_id": "query"}),
    )

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"


def test_deterministic_evidence_must_match_canonical_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    safe = build_deterministic_fallback_result("query", history_path).recommendation
    payload = safe.model_dump()
    payload["deterministic_evidence"] = (
        *safe.deterministic_evidence,
        "Invented causal interpretation.",
    )
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(payload),
    )

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"


@pytest.mark.parametrize(
    "limitations",
    [
        CORE_LIMITATIONS[:-1],
        (
            CORE_LIMITATIONS[0],
            CORE_LIMITATIONS[1],
            CORE_LIMITATIONS[2],
            "No automatic machine control is authorized.",
        ),
    ],
)
def test_limitations_must_match_canonical_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limitations: tuple[str, ...],
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    safe = build_deterministic_fallback_result("query", history_path).recommendation
    payload = safe.model_dump()
    payload["limitations"] = limitations
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(payload),
    )

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"


@pytest.mark.parametrize(
    "prohibited_phrase",
    [
        "machine has a fault",
        "this is a fault",
        "tool wear is likely",
        "tool wear is the cause",
        "root cause:",
        "root cause of",
        "replace tooling",
        "replace the tool",
        "stop the machine",
        "open a maintenance ticket",
        "create a maintenance ticket",
        "automatic shutdown",
    ],
)
def test_added_prohibited_phrases_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prohibited_phrase: str,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    safe = build_deterministic_fallback_result("query", history_path).recommendation
    payload = safe.model_dump()
    payload["historical_context"] = prohibited_phrase
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(payload),
    )

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"


@pytest.mark.parametrize(
    "field, value",
    [
        ("event_id", "wrong-event"),
        ("screening_status", "within_normal_band"),
        ("guidance_ids", ("invented_guidance",)),
        ("manual_review_checks", ("Replace the spindle immediately.",)),
        ("similar_event_ids", ("invented-similar",)),
        ("headline", "Confirmed fault: repair the machine."),
    ],
)
def test_valid_shaped_but_ungrounded_adk_output_returns_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("prior", seconds=30), history_path)
    append_monitoring_event(make_event("query", seconds=60), history_path)
    safe = build_deterministic_fallback_result("query", history_path).recommendation
    payload = safe.model_dump()
    payload[field] = value
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(payload),
    )

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"
    assert result.recommendation.event_id == "query"


def test_valid_adk_output_with_missing_mcp_trace_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    safe = build_deterministic_fallback_result("query", history_path).recommendation
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(safe, ()),
    )

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"
    assert all(
        tool_call.source == "deterministic_service"
        for tool_call in result.trace.tool_calls
    )


def test_valid_adk_output_with_failed_mcp_trace_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    safe = build_deterministic_fallback_result("query", history_path).recommendation
    failed_calls = (
        valid_mcp_tool_calls()[0],
        ToolCallTrace(
            tool_name="retrieve_maintenance_guidance",
            source="mcp",
            outcome="failed",
            error_code="tool_execution_failed",
        ),
        valid_mcp_tool_calls()[2],
    )
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(safe, failed_calls),
    )

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"


def test_valid_adk_output_with_wrong_tool_order_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    safe = build_deterministic_fallback_result("query", history_path).recommendation
    wrong_order = (
        valid_mcp_tool_calls()[1],
        valid_mcp_tool_calls()[0],
        valid_mcp_tool_calls()[2],
    )
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
        agent_executor=lambda *_: make_adk_outcome(safe, wrong_order),
    )

    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"


def test_unapproved_observed_tool_response_raises_generic_safe_error() -> None:
    event = FakeAdkEvent(FakeFunctionResponse("unapproved_secret_tool"))

    with pytest.raises(ValueError, match="unapproved MCP tool response observed") as exc:
        _tool_traces_from_event(event)

    assert "unapproved_secret_tool" not in str(exc.value)


def test_unapproved_observed_tool_response_causes_safe_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    monkeypatch.setattr(
        runner_module,
        "create_incident_copilot_agent",
        lambda **_: FakeBundle(),
    )

    def failing_executor(*_: object, **__: object) -> object:
        raise ValueError("unapproved MCP tool response observed")

    monkeypatch.setattr(
        runner_module,
        "execute_incident_copilot_bundle",
        failing_executor,
    )

    result = run_incident_copilot(
        "query",
        history_path=history_path,
        model="configured-model",
    )

    payload = result.model_dump_json()
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_execution_failed"
    assert tuple(tool_call.source for tool_call in result.trace.tool_calls) == (
        "deterministic_service",
        "deterministic_service",
        "deterministic_service",
    )
    assert "unapproved_secret_tool" not in payload
    assert "unapproved MCP tool response observed" not in payload


def test_unknown_event_id_raises_existing_safe_domain_error(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("known", seconds=60), history_path)

    with pytest.raises(EventNotFoundError, match=EVENT_NOT_FOUND_MESSAGE):
        run_incident_copilot("unknown", history_path=history_path)


def test_agent_factory_uses_exact_approved_tool_allowlist(tmp_path: Path) -> None:
    bundle = create_incident_copilot_agent(
        model="configured-model",
        history_path=tmp_path / "history.jsonl",
    )
    try:
        toolset = bundle.toolset
        assert tuple(toolset.tool_filter) == (
            "get_event_metrics",
            "retrieve_maintenance_guidance",
            "find_similar_previous_events",
        )
        assert bundle.agent.model == "configured-model"
        assert bundle.tool_names == tuple(toolset.tool_filter)
    finally:
        import asyncio

        asyncio.run(bundle.close())


def test_agent_package_import_is_safe_without_model_configuration() -> None:
    import volt_vision.agent as agent_package

    assert agent_package.run_incident_copilot is run_incident_copilot
