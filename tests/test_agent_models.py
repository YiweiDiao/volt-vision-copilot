from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from volt_vision.agent.models import (
    AgentRunTrace,
    InvestigationRecommendation,
    InvestigationResult,
    ToolCallTrace,
)


def make_recommendation() -> InvestigationRecommendation:
    return InvestigationRecommendation(
        event_id="event-1",
        screening_status="suspected_deviation",
        headline="Suspected deviation requires manual review.",
        deterministic_evidence=("Deterministic screening status: suspected_deviation.",),
        guidance_ids=("power_signature_review",),
        manual_review_checks=("Check whether similar structured events recur.",),
        similar_event_ids=("prior-1",),
        historical_context="Structured recurrence context only.",
        limitations=("No diagnosis or root-cause confirmation.",),
        human_approval_required=True,
    )


def test_recommendation_model_is_frozen_and_json_safe() -> None:
    recommendation = make_recommendation()

    with pytest.raises(ValidationError):
        recommendation.headline = "changed"  # type: ignore[misc]

    payload = recommendation.model_dump(mode="json")
    assert payload["human_approval_required"] is True
    assert json.loads(recommendation.model_dump_json())["event_id"] == "event-1"


def test_result_model_requires_human_approval_literal_true() -> None:
    tool_trace = ToolCallTrace(
        tool_name="get_event_metrics",
        source="deterministic_service",
        outcome="succeeded",
        error_code=None,
    )
    trace = AgentRunTrace(
        event_id="event-1",
        execution_mode="deterministic_fallback",
        tool_names=("get_event_metrics",),
        tool_calls=(tool_trace,),
        fallback_reason="model_not_configured",
        completed=True,
    )
    result = InvestigationResult(recommendation=make_recommendation(), trace=trace)

    assert result.recommendation.human_approval_required is True

    with pytest.raises(ValidationError):
        InvestigationRecommendation(
            **{
                **make_recommendation().model_dump(),
                "human_approval_required": False,
            }
        )


def test_tool_call_trace_is_frozen_and_json_safe() -> None:
    tool_trace = ToolCallTrace(
        tool_name="find_similar_previous_events",
        source="mcp",
        outcome="failed",
        error_code="tool_execution_failed",
    )

    with pytest.raises(ValidationError):
        tool_trace.outcome = "succeeded"  # type: ignore[misc]

    payload = tool_trace.model_dump(mode="json")
    assert payload == {
        "tool_name": "find_similar_previous_events",
        "source": "mcp",
        "outcome": "failed",
        "error_code": "tool_execution_failed",
    }
    assert "traceback" not in tool_trace.model_dump_json().lower()
