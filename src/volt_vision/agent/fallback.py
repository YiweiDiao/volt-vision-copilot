"""Deterministic no-model fallback for suspected-deviation investigation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from volt_vision.agent.models import (
    AgentRunTrace,
    InvestigationRecommendation,
    InvestigationResult,
    ToolCallTrace,
    successful_trace_for,
    tool_names_from_trace,
)
from volt_vision.agent.policy import CORE_LIMITATIONS
from volt_vision.mcp_server.services import (
    DEFAULT_HISTORY_PATH,
    find_similar_previous_events,
    get_event_metrics,
    retrieve_maintenance_guidance,
)


def build_deterministic_fallback_result(
    event_id: str,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    *,
    fallback_reason: Literal[
        "model_not_configured",
        "model_execution_failed",
    ] = "model_not_configured",
) -> InvestigationResult:
    """Build a bounded recommendation from read-only structured services."""

    tool_calls: list[ToolCallTrace] = []
    metrics = get_event_metrics(event_id, history_path)
    tool_calls.append(successful_trace_for("get_event_metrics", "deterministic_service"))
    guidance = retrieve_maintenance_guidance(event_id, history_path)
    tool_calls.append(
        successful_trace_for(
            "retrieve_maintenance_guidance",
            "deterministic_service",
        )
    )
    similar_events = find_similar_previous_events(
        event_id,
        limit=3,
        history_path=history_path,
    )
    tool_calls.append(
        successful_trace_for(
            "find_similar_previous_events",
            "deterministic_service",
        )
    )
    trace_tool_calls = tuple(tool_calls)
    recommendation = InvestigationRecommendation(
        event_id=metrics.event_id,
        screening_status=metrics.status,
        headline=(
            "Suspected deviation requires manual review by an authorized "
            "maintenance or production representative."
        ),
        deterministic_evidence=build_deterministic_evidence(metrics),
        guidance_ids=tuple(item.guidance_id for item in guidance),
        manual_review_checks=_manual_review_checks(guidance),
        similar_event_ids=tuple(item.event_id for item in similar_events),
        historical_context=_historical_context(similar_events),
        limitations=CORE_LIMITATIONS,
        human_approval_required=True,
    )
    trace = AgentRunTrace(
        event_id=metrics.event_id,
        execution_mode="deterministic_fallback",
        tool_names=tool_names_from_trace(trace_tool_calls),
        tool_calls=trace_tool_calls,
        fallback_reason=fallback_reason,
        completed=True,
    )
    return InvestigationResult(recommendation=recommendation, trace=trace)


def build_deterministic_evidence(metrics: object) -> tuple[str, ...]:
    """Return canonical deterministic evidence text for safe recommendations."""

    status = getattr(metrics, "status")
    indicators = getattr(metrics, "indicators")
    evidence = (
        f"Deterministic screening status: {status}.",
        "Normalized DTW distance "
        f"{getattr(metrics, 'normalized_dtw_distance'):.6g} compared with "
        f"calibrated threshold {getattr(metrics, 'threshold'):.6g}.",
        "Duration deviation percentage: "
        f"{_format_indicator(indicators.duration_deviation_pct)}.",
        "Energy deviation percentage: "
        f"{_format_indicator(indicators.energy_deviation_pct)}.",
        "Peak-power deviation percentage: "
        f"{_format_indicator(indicators.peak_power_deviation_pct)}.",
    )
    return evidence


def _manual_review_checks(guidance: tuple[object, ...]) -> tuple[str, ...]:
    checks: list[str] = []
    for item in guidance:
        checks.extend(getattr(item, "inspection_checks"))
    return tuple(dict.fromkeys(checks))


def _historical_context(similar_events: tuple[object, ...]) -> str:
    if not similar_events:
        return (
            "No earlier same-machine, same-status structured events were found "
            "for recurrence comparison."
        )
    return (
        "Similar historical event IDs are provided as structured recurrence "
        "context only and do not confirm root cause: "
        + ", ".join(getattr(item, "event_id") for item in similar_events)
        + "."
    )


def _format_indicator(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.6g}"
