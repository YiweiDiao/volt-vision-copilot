"""Public runner for bounded incident copilot investigation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from volt_vision.agent.adk_execution import execute_incident_copilot_bundle
from volt_vision.agent.adk_factory import create_incident_copilot_agent
from volt_vision.agent.fallback import build_deterministic_fallback_result
from volt_vision.agent.grounding import validate_grounded_recommendation
from volt_vision.agent.models import (
    AgentRunTrace,
    InvestigationRecommendation,
    InvestigationResult,
    ToolCallTrace,
    tool_names_from_trace,
)
from volt_vision.agent.policy import APPROVED_MCP_TOOL_NAMES
from volt_vision.mcp_server.services import (
    DEFAULT_HISTORY_PATH,
    EVENT_NOT_FOUND_MESSAGE,
    EventNotFoundError,
)
from volt_vision.monitoring.event_log import read_monitoring_events
from volt_vision.monitoring.models import MonitoringEvent

AgentExecutor = Callable[[Any, str], Any]


def run_incident_copilot(
    event_id: str,
    *,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    model: str | None = None,
    agent_executor: AgentExecutor | None = None,
) -> InvestigationResult:
    """Run the incident copilot gate, optional ADK path, or fallback."""

    event = _get_persisted_event(event_id, history_path)
    if event.status == "within_normal_band":
        return _not_triggered_result(event)

    if model is None:
        return build_deterministic_fallback_result(
            event_id,
            history_path,
            fallback_reason="model_not_configured",
        )

    try:
        bundle = create_incident_copilot_agent(model=model, history_path=history_path)
        outcome = execute_incident_copilot_bundle(
            bundle,
            event_id,
            agent_executor=agent_executor,
        )
        _validate_required_mcp_trace(outcome.tool_calls)
        recommendation = InvestigationRecommendation.model_validate(
            outcome.output_payload
        )
        validate_grounded_recommendation(recommendation, event_id, history_path)
        return InvestigationResult(
            recommendation=recommendation,
            trace=AgentRunTrace(
                event_id=event.event_id,
                execution_mode="adk",
                tool_names=tool_names_from_trace(outcome.tool_calls),
                tool_calls=outcome.tool_calls,
                fallback_reason=None,
                completed=True,
            ),
        )
    except Exception:
        return build_deterministic_fallback_result(
            event_id,
            history_path,
            fallback_reason="model_execution_failed",
        )


def _get_persisted_event(
    event_id: str,
    history_path: str | Path,
) -> MonitoringEvent:
    for event in reversed(read_monitoring_events(history_path)):
        if event.event_id == event_id:
            return event
    raise EventNotFoundError(EVENT_NOT_FOUND_MESSAGE)


def _not_triggered_result(event: MonitoringEvent) -> InvestigationResult:
    recommendation = InvestigationRecommendation(
        event_id=event.event_id,
        screening_status=event.status,
        headline=(
            "Copilot investigation is available only after a suspected "
            "deviation."
        ),
        deterministic_evidence=(
            "Deterministic screening status: within_normal_band.",
            "No ADK agent, MCP tool, or model call was triggered.",
        ),
        guidance_ids=(),
        manual_review_checks=(),
        similar_event_ids=(),
        historical_context=(
            "No historical comparison was performed because the persisted "
            "event is within the calibrated normal band."
        ),
        limitations=(
            "The deterministic MonitoringEvent status remains authoritative.",
            "Real-world actions require an authorized maintenance or production "
            "representative.",
        ),
        human_approval_required=True,
    )
    trace = AgentRunTrace(
        event_id=event.event_id,
        execution_mode="not_triggered",
        tool_names=(),
        tool_calls=(),
        fallback_reason=None,
        completed=True,
    )
    return InvestigationResult(recommendation=recommendation, trace=trace)


def _validate_required_mcp_trace(tool_calls: tuple[ToolCallTrace, ...]) -> None:
    first_observed: list[str] = []
    succeeded: set[str] = set()
    for tool_call in tool_calls:
        if tool_call.source != "mcp":
            raise ValueError("ADK trace contains non-MCP tool source")
        if tool_call.tool_name not in APPROVED_MCP_TOOL_NAMES:
            raise ValueError("ADK trace contains unapproved tool")
        if tool_call.outcome != "succeeded":
            raise ValueError("ADK MCP tool did not succeed")
        if tool_call.tool_name not in succeeded:
            first_observed.append(tool_call.tool_name)
        succeeded.add(tool_call.tool_name)

    required = list(APPROVED_MCP_TOOL_NAMES)
    if first_observed[: len(required)] != required:
        raise ValueError("ADK MCP tools were missing or out of order")
    if not set(required).issubset(succeeded):
        raise ValueError("ADK MCP tools were incomplete")
