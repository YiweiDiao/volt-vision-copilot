"""Typed immutable payloads for incident copilot results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from volt_vision.agent.policy import APPROVED_MCP_TOOL_NAMES

ApprovedToolName = Literal[
    "get_event_metrics",
    "retrieve_maintenance_guidance",
    "find_similar_previous_events",
]


class ToolCallTrace(BaseModel):
    """Bounded per-tool trace without arguments, payloads, or exception text."""

    model_config = ConfigDict(frozen=True)

    tool_name: ApprovedToolName
    source: Literal["deterministic_service", "mcp"]
    outcome: Literal["succeeded", "failed", "not_called"]
    error_code: Literal[
        "event_not_found",
        "similarity_ranking_unavailable",
        "tool_execution_failed",
        "tool_not_called",
    ] | None


class InvestigationRecommendation(BaseModel):
    """Bounded human-review recommendation produced for one screening event."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    screening_status: Literal["within_normal_band", "suspected_deviation"]
    headline: str
    deterministic_evidence: tuple[str, ...]
    guidance_ids: tuple[str, ...]
    manual_review_checks: tuple[str, ...]
    similar_event_ids: tuple[str, ...]
    historical_context: str
    limitations: tuple[str, ...]
    human_approval_required: Literal[True]


class AgentRunTrace(BaseModel):
    """Safe in-memory execution trace without prompts or raw payloads.

    tool_names is a compact backward-compatible summary derived from tool_calls.
    tool_calls is the authoritative audit record.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    execution_mode: Literal["adk", "deterministic_fallback", "not_triggered"]
    tool_names: tuple[str, ...]
    tool_calls: tuple[ToolCallTrace, ...]
    fallback_reason: Literal["model_not_configured", "model_execution_failed"] | None
    completed: bool


class InvestigationResult(BaseModel):
    """Recommendation plus bounded trace metadata."""

    model_config = ConfigDict(frozen=True)

    recommendation: InvestigationRecommendation
    trace: AgentRunTrace


class CopilotChatResponse(BaseModel):
    """Accepted user-visible chat response with bounded safe metadata."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    execution_mode: Literal["adk", "deterministic_fallback"]
    assistant_message: str
    knowledge_source_ids: tuple[str, ...]
    tool_trace: AgentRunTrace
    human_approval_required: Literal[True]
    fallback_reason: Literal[
        "model_not_configured",
        "model_execution_failed",
    ] | None


class AgentExecutionOutcome(BaseModel):
    """Structured ADK output plus bounded observed tool outcomes."""

    model_config = ConfigDict(frozen=True)

    output_payload: dict[str, object]
    tool_calls: tuple[ToolCallTrace, ...]


class AgentChatExecutionOutcome(BaseModel):
    """Natural-language ADK output plus bounded observed tool outcomes."""

    model_config = ConfigDict(frozen=True)

    final_text: str
    tool_calls: tuple[ToolCallTrace, ...]


def tool_names_from_trace(tool_calls: tuple[ToolCallTrace, ...]) -> tuple[str, ...]:
    """Return compact tool-name summary in trace order."""

    return tuple(tool_call.tool_name for tool_call in tool_calls)


def successful_trace_for(
    tool_name: ApprovedToolName,
    source: Literal["deterministic_service", "mcp"],
) -> ToolCallTrace:
    """Build a successful bounded tool trace."""

    if tool_name not in APPROVED_MCP_TOOL_NAMES:
        raise ValueError("tool name is not approved")
    return ToolCallTrace(
        tool_name=tool_name,
        source=source,
        outcome="succeeded",
        error_code=None,
    )
