"""Chat-oriented safety layer for the SAIA/ADK copilot path."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from volt_vision.agent.adk_execution import (
    execute_chat_copilot_bundle,
    execute_chat_copilot_diagnostic_bundle,
    _run_async,
)
from volt_vision.agent.adk_factory import create_chat_copilot_agent
from volt_vision.agent.models import (
    AgentChatExecutionOutcome,
    AgentRunTrace,
    CopilotChatResponse,
    ToolCallTrace,
    successful_trace_for,
    tool_names_from_trace,
)
from volt_vision.agent.policy import APPROVED_MCP_TOOL_NAMES, MANDATORY_CHAT_HEADINGS
from volt_vision.guidance.retrieval import retrieve_guidance
from volt_vision.mcp_server.services import (
    DEFAULT_HISTORY_PATH,
    EVENT_NOT_FOUND_MESSAGE,
    EventNotFoundError,
    find_similar_previous_events,
    get_event_metrics,
    retrieve_maintenance_guidance,
)
from volt_vision.monitoring.event_log import read_monitoring_events
from volt_vision.monitoring.models import MonitoringEvent

MAX_CHAT_CHARACTERS = 1800
UNCERTAINTY_PHRASES = (
    "suspected deviation",
    "not a confirmed diagnosis",
    "manual inspection recommended",
)
PROHIBITED_CHAT_PATTERNS = (
    r"\bconfirmed\s+(fault|failure|tool wear|root cause|diagnosis)\b",
    r"\b(root cause is|root cause:|diagnosis:)\b",
    r"\b(machine|tool|component)\s+(has|is showing)\s+a\s+fault\b",
    r"\btool wear\s+(is|was|caused|likely|confirmed)\b",
    r"\b(repair|replace|replacement|shutdown|ticket|tuning)\b",
    r"\b(open|create)\s+(a\s+)?maintenance\s+ticket\b",
    r"\b(stop|halt)\s+the\s+machine\b",
    r"\b(control|command)\s+the\s+machine\b",
    r"\b(plc|scada|mes|opc-ua|modbus)\b",
)
SENSITIVE_CHAT_PATTERNS = (
    r"\b(get_event_metrics|retrieve_maintenance_guidance|find_similar_previous_events)\b",
    r"\bsk-[A-Za-z0-9_-]{8,}\b",
    r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b",
    r"\b[A-Z]:\\",
    r"(?<!\w)/(?:home|tmp|var|etc|users?)/",
    r"\b(traceback|stack trace|exception|api key|secret|token)\b",
)

ChatAgentExecutor = Any
ChatComposer = Callable[[object, "ChatCompositionContext"], str]
YesNo = Literal["yes", "no"]
ComposerFinishReason = Literal["stop", "length", "tool_calls", "absent_or_unknown"]
CompletionTokenUsageBucket = Literal[
    "zero",
    "below_half_budget",
    "at_least_half_budget",
    "near_or_at_budget",
    "unavailable",
]


class ChatCompositionContext(BaseModel):
    """Sanitized local facts permitted for Stage 2 chat composition."""

    model_config = ConfigDict(frozen=True)

    screening_status: Literal["suspected_deviation"]
    indicator_summaries: tuple[str, ...]
    guidance_titles: tuple[str, ...]
    possible_contributing_conditions: tuple[str, ...]
    safe_inspection_checks: tuple[str, ...]
    operator_questions: tuple[str, ...]
    escalation_conditions: tuple[str, ...]
    similar_event_summary: str
    knowledge_source_ids: tuple[str, ...]


class ComposerResponseObservation(BaseModel):
    """Bounded composer metadata without model text, reasoning, or raw response."""

    model_config = ConfigDict(frozen=True)

    response_received: YesNo
    finish_reason: ComposerFinishReason
    visible_content_present: YesNo
    visible_content_nonempty: YesNo
    reasoning_content_present: YesNo
    reasoning_content_nonempty: YesNo
    completion_token_usage: CompletionTokenUsageBucket
    alternate_text_field_present: YesNo


def run_chat_copilot(
    event_id: str,
    *,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    model: str | object | None = None,
    agent_executor: ChatAgentExecutor | None = None,
    chat_composer: ChatComposer | None = None,
) -> CopilotChatResponse:
    """Run chat copilot with deterministic fallback and local validation."""

    event = _get_persisted_event(event_id, history_path)
    if event.status == "within_normal_band":
        return _within_normal_band_response(event)

    if model is None:
        return build_deterministic_fallback_chat_response(
            event_id,
            history_path,
            fallback_reason="model_not_configured",
        )

    try:
        bundle = create_chat_copilot_agent(model=model, history_path=history_path)
        if agent_executor is not None:
            outcome = execute_chat_copilot_bundle(
                bundle,
                event_id,
                agent_executor=agent_executor,
            )
        else:
            outcome = execute_chat_copilot_diagnostic_bundle(bundle, event_id)
        _validate_required_mcp_trace(outcome.tool_calls)
        context = build_chat_composition_context(event_id, history_path)
        composer = chat_composer or compose_chat_with_model
        assistant_message = composer(model, context)
        validate_copilot_chat_text(assistant_message, event_id=event_id)
        trace = AgentRunTrace(
            event_id=event.event_id,
            execution_mode="adk",
            tool_names=tool_names_from_trace(outcome.tool_calls),
            tool_calls=outcome.tool_calls,
            fallback_reason=None,
            completed=True,
        )
        return CopilotChatResponse(
            event_id=event.event_id,
            execution_mode="adk",
            assistant_message=assistant_message.strip(),
            knowledge_source_ids=context.knowledge_source_ids,
            tool_trace=trace,
            human_approval_required=True,
            fallback_reason=None,
        )
    except Exception:
        return build_deterministic_fallback_chat_response(
            event_id,
            history_path,
            fallback_reason="model_execution_failed",
        )


def build_chat_composition_context(
    event_id: str,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
) -> ChatCompositionContext:
    """Build the safe local context allowed into the Stage 2 composer."""

    metrics = get_event_metrics(event_id, history_path)
    if metrics.status != "suspected_deviation":
        raise ValueError("chat composition requires a suspected deviation")
    guidance = retrieve_maintenance_guidance(event_id, history_path)
    similar_events = find_similar_previous_events(
        event_id,
        limit=3,
        history_path=history_path,
    )
    return ChatCompositionContext(
        screening_status="suspected_deviation",
        indicator_summaries=_indicator_summaries(metrics),
        guidance_titles=tuple(item.title for item in guidance),
        possible_contributing_conditions=_guidance_field_values(
            guidance,
            "possible_contributing_conditions",
        ),
        safe_inspection_checks=_guidance_field_values(guidance, "safe_inspection_checks"),
        operator_questions=_guidance_field_values(guidance, "operator_questions"),
        escalation_conditions=tuple(
            dict.fromkeys(item.escalation_condition for item in guidance)
        ),
        similar_event_summary=_similar_event_summary(len(similar_events)),
        knowledge_source_ids=tuple(item.guidance_id for item in guidance),
    )


def compose_chat_with_model(
    model: object,
    context: ChatCompositionContext,
) -> str:
    """Compose final chat text with a configured ADK LiteLlm-compatible model."""

    return _run_async(_compose_chat_with_model_async(model, context))


def observe_composer_response(
    response: object | None,
    *,
    max_output_tokens: int = 700,
) -> ComposerResponseObservation:
    """Return bounded metadata about one composer response."""

    return _merge_composer_observations(
        (response,),
        max_output_tokens=max_output_tokens,
    )


def compose_chat_with_model_observed(
    model: object,
    context: ChatCompositionContext,
) -> tuple[str, ComposerResponseObservation]:
    """Compose chat text and return bounded response observation metadata."""

    return _run_async(_compose_chat_with_model_observed_async(model, context))


async def _compose_chat_with_model_async(
    model: object,
    context: ChatCompositionContext,
) -> str:
    text, _ = await _compose_chat_with_model_observed_async(model, context)
    return text


async def _compose_chat_with_model_observed_async(
    model: object,
    context: ChatCompositionContext,
) -> tuple[str, ComposerResponseObservation]:
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    max_output_tokens = 700
    if not hasattr(model, "generate_content_async"):
        raise ValueError("chat composer model is unavailable")
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=_composition_prompt(context))],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=max_output_tokens,
        ),
    )
    text_parts: list[str] = []
    responses: list[object] = []
    async for response in model.generate_content_async(request, stream=False):  # type: ignore[attr-defined]
        responses.append(response)
        content = getattr(response, "content", None)
        parts = getattr(content, "parts", None) or ()
        text_parts.extend(
            part.text
            for part in parts
            if getattr(part, "text", None) and not getattr(part, "thought", False)
        )
    return "\n".join(text_parts).strip(), _merge_composer_observations(
        tuple(responses),
        max_output_tokens=max_output_tokens,
    )


def _merge_composer_observations(
    responses: tuple[object | None, ...],
    *,
    max_output_tokens: int,
) -> ComposerResponseObservation:
    seen = [response for response in responses if response is not None]
    visible_present = False
    visible_nonempty = False
    reasoning_present = False
    reasoning_nonempty = False
    alternate_present = False
    completion_tokens: list[int] = []
    finish_reasons: list[ComposerFinishReason] = []

    for response in seen:
        finish_reasons.append(_bounded_finish_reason(response))
        content = getattr(response, "content", None)
        for part in getattr(content, "parts", None) or ():
            part_text = getattr(part, "text", None)
            is_thought = bool(getattr(part, "thought", False))
            if is_thought:
                if part_text is not None:
                    reasoning_present = True
                if isinstance(part_text, str) and part_text.strip():
                    reasoning_nonempty = True
            else:
                if part_text is not None:
                    visible_present = True
                if isinstance(part_text, str) and part_text.strip():
                    visible_nonempty = True
        reasoning_values = (
            getattr(response, "reasoning_content", None),
            getattr(response, "reasoning", None),
            getattr(response, "thoughts", None),
        )
        if any(value is not None for value in reasoning_values):
            reasoning_present = True
        if any(isinstance(value, str) and value.strip() for value in reasoning_values):
            reasoning_nonempty = True
        alternate_present = alternate_present or _alternate_text_field_present(response)
        token_count = _completion_token_count(response)
        if token_count is not None:
            completion_tokens.append(token_count)

    return ComposerResponseObservation(
        response_received=_yes_no(bool(seen)),
        finish_reason=_merge_finish_reasons(tuple(finish_reasons)),
        visible_content_present=_yes_no(visible_present),
        visible_content_nonempty=_yes_no(visible_nonempty),
        reasoning_content_present=_yes_no(reasoning_present),
        reasoning_content_nonempty=_yes_no(reasoning_nonempty),
        completion_token_usage=_token_usage_bucket(
            sum(completion_tokens) if completion_tokens else None,
            max_output_tokens=max_output_tokens,
        ),
        alternate_text_field_present=_yes_no(alternate_present),
    )


def _bounded_finish_reason(response: object) -> ComposerFinishReason:
    raw_values = (
        getattr(response, "finish_reason", None),
        getattr(response, "turn_complete_reason", None),
    )
    raw = " ".join(str(value).lower() for value in raw_values if value is not None)
    if any(value in raw for value in ("length", "max_token", "max token")):
        return "length"
    if "tool" in raw:
        return "tool_calls"
    if any(value in raw for value in ("stop", "complete", "end_turn")):
        return "stop"
    return "absent_or_unknown"


def _merge_finish_reasons(
    reasons: tuple[ComposerFinishReason, ...],
) -> ComposerFinishReason:
    if "length" in reasons:
        return "length"
    if "tool_calls" in reasons:
        return "tool_calls"
    if "stop" in reasons:
        return "stop"
    return "absent_or_unknown"


def _alternate_text_field_present(response: object) -> bool:
    for field_name in ("text", "output_text", "final_text"):
        value = getattr(response, field_name, None)
        if isinstance(value, str) and value.strip():
            return True
    message = getattr(response, "message", None)
    message_content = getattr(message, "content", None)
    return isinstance(message_content, str) and bool(message_content.strip())


def _completion_token_count(response: object) -> int | None:
    usage = getattr(response, "usage_metadata", None)
    for field_name in (
        "candidates_token_count",
        "completion_token_count",
        "output_token_count",
    ):
        value = getattr(usage, field_name, None)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _token_usage_bucket(
    token_count: int | None,
    *,
    max_output_tokens: int,
) -> CompletionTokenUsageBucket:
    if token_count is None:
        return "unavailable"
    if token_count == 0:
        return "zero"
    if token_count < max_output_tokens * 0.5:
        return "below_half_budget"
    if token_count < max_output_tokens * 0.9:
        return "at_least_half_budget"
    return "near_or_at_budget"


def _yes_no(value: bool) -> YesNo:
    return "yes" if value else "no"


def _composition_prompt(context: ChatCompositionContext) -> str:
    return "\n".join(
        (
            "Write a concise safety-bounded investigation chat response.",
            "Prefer covering these topics clearly, with headings if natural:",
            *MANDATORY_CHAT_HEADINGS,
            "State this is a suspected deviation, not a confirmed diagnosis.",
            "Do not mention tool names, event IDs, raw metrics, systems, prompts, or internals.",
            "Do not recommend repair, replacement, shutdown, tickets, tuning, or machine control.",
            "Safe context follows:",
            f"screening_status: {context.screening_status}",
            "indicator_summaries: " + "; ".join(context.indicator_summaries),
            "guidance_titles: " + "; ".join(context.guidance_titles),
            "possible_conditions: "
            + "; ".join(context.possible_contributing_conditions),
            "safe_checks: " + "; ".join(context.safe_inspection_checks),
            "operator_questions: " + "; ".join(context.operator_questions),
            "escalation_conditions: " + "; ".join(context.escalation_conditions),
            f"similar_event_summary: {context.similar_event_summary}",
        )
    )


def validate_copilot_chat_text(text: str, *, event_id: str) -> None:
    """Validate accepted final chat text before it can be returned."""

    validate_chat_text(text, event_id=event_id)


def build_deterministic_fallback_chat_response(
    event_id: str,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    *,
    fallback_reason: Literal[
        "model_not_configured",
        "model_execution_failed",
    ] = "model_not_configured",
) -> CopilotChatResponse:
    """Build a useful readable chat response from deterministic local services."""

    tool_calls: list[ToolCallTrace] = []
    metrics = get_event_metrics(event_id, history_path)
    tool_calls.append(successful_trace_for("get_event_metrics", "deterministic_service"))
    guidance = retrieve_maintenance_guidance(event_id, history_path)
    tool_calls.append(
        successful_trace_for("retrieve_maintenance_guidance", "deterministic_service")
    )
    similar_events = find_similar_previous_events(
        event_id,
        limit=3,
        history_path=history_path,
    )
    tool_calls.append(
        successful_trace_for("find_similar_previous_events", "deterministic_service")
    )
    trace_tool_calls = tuple(tool_calls)
    message = _fallback_message(metrics, guidance, len(similar_events))
    validate_chat_text(message, event_id=event_id)
    trace = AgentRunTrace(
        event_id=metrics.event_id,
        execution_mode="deterministic_fallback",
        tool_names=tool_names_from_trace(trace_tool_calls),
        tool_calls=trace_tool_calls,
        fallback_reason=fallback_reason,
        completed=True,
    )
    return CopilotChatResponse(
        event_id=metrics.event_id,
        execution_mode="deterministic_fallback",
        assistant_message=message,
        knowledge_source_ids=tuple(item.guidance_id for item in guidance),
        tool_trace=trace,
        human_approval_required=True,
        fallback_reason=fallback_reason,
    )


def validate_chat_text(text: str, *, event_id: str) -> None:
    """Validate accepted final chat text before it can be returned."""

    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped:
        raise ValueError("chat text is empty")
    if len(stripped) > MAX_CHAT_CHARACTERS:
        raise ValueError("chat text is too long")
    lowered = stripped.lower()
    if not all(phrase in lowered for phrase in UNCERTAINTY_PHRASES[:2]):
        raise ValueError("chat text lacks uncertainty wording")
    safety_scan_text = lowered.replace("not a confirmed diagnosis", "")
    for pattern in PROHIBITED_CHAT_PATTERNS:
        if re.search(pattern, safety_scan_text, flags=re.IGNORECASE):
            raise ValueError("chat text contains unsafe operational wording")
    if event_id and re.search(re.escape(event_id), stripped, flags=re.IGNORECASE):
        raise ValueError("chat text contains event identifier")
    for pattern in SENSITIVE_CHAT_PATTERNS:
        if re.search(pattern, stripped, flags=re.IGNORECASE):
            raise ValueError("chat text contains sensitive implementation detail")


def _fallback_message(
    metrics: object,
    guidance: tuple[object, ...],
    similar_count: int,
) -> str:
    condition_lines = _field_lines(guidance, "possible_contributing_conditions", 3)
    check_lines = _field_lines(guidance, "safe_inspection_checks", 4)
    question_lines = _field_lines(guidance, "operator_questions", 3)
    recurrence = (
        "Similar prior structured events were found for comparison."
        if similar_count
        else "No similar prior structured event was found for comparison."
    )
    return "\n".join(
        (
            MANDATORY_CHAT_HEADINGS[0],
            "The deterministic screening marked this as a suspected deviation, "
            "not a confirmed diagnosis. Manual inspection recommended. "
            f"{_safe_status_sentence(metrics)} {recurrence}",
            "",
            MANDATORY_CHAT_HEADINGS[1],
            _sentence_from_lines(condition_lines),
            "",
            MANDATORY_CHAT_HEADINGS[2],
            _sentence_from_lines(check_lines),
            "",
            MANDATORY_CHAT_HEADINGS[3],
            _sentence_from_lines(question_lines),
            "",
            MANDATORY_CHAT_HEADINGS[4],
            "Escalate when recurrence, production impact, or local procedure "
            "requires review. Keep follow-up under authorized human review.",
        )
    )


def _safe_status_sentence(metrics: object) -> str:
    indicators = getattr(metrics, "indicators")
    changed = []
    if _indicator_triggered(indicators.duration_deviation_pct):
        changed.append("duration")
    if _indicator_triggered(indicators.energy_deviation_pct):
        changed.append("energy")
    if _indicator_triggered(indicators.peak_power_deviation_pct):
        changed.append("peak-power")
    if not changed:
        return "No individual reference-relative indicator exceeded the review trigger."
    return "Reference-relative indicators for " + ", ".join(changed) + " need review."


def _indicator_triggered(value: float | None) -> bool:
    return value is not None and abs(value) >= 10.0


def _indicator_summaries(metrics: object) -> tuple[str, ...]:
    indicators = getattr(metrics, "indicators")
    summaries = []
    for label, value in (
        ("duration", indicators.duration_deviation_pct),
        ("energy", indicators.energy_deviation_pct),
        ("peak-power", indicators.peak_power_deviation_pct),
    ):
        if value is None:
            summaries.append(f"{label}: not available")
        elif _indicator_triggered(value):
            direction = "higher" if value > 0 else "lower"
            summaries.append(f"{label}: {direction} than reference review trigger")
        else:
            summaries.append(f"{label}: within review trigger")
    return tuple(summaries)


def _guidance_field_values(
    guidance: tuple[object, ...],
    field_name: str,
) -> tuple[str, ...]:
    values: list[str] = []
    for item in guidance:
        values.extend(getattr(item, field_name, ()))
    return tuple(dict.fromkeys(values))


def _similar_event_summary(count: int) -> str:
    if count <= 0:
        return "No similar prior structured events were found."
    return f"{count} similar prior structured event groups were found."


def _field_lines(
    guidance: tuple[object, ...],
    field_name: str,
    limit: int,
) -> tuple[str, ...]:
    values: list[str] = []
    for item in guidance:
        values.extend(getattr(item, field_name, ()))
    return tuple(dict.fromkeys(values))[:limit]


def _sentence_from_lines(lines: tuple[str, ...]) -> str:
    return " ".join(line.rstrip(".") + "." for line in lines)


def _within_normal_band_response(event: MonitoringEvent) -> CopilotChatResponse:
    guidance = retrieve_guidance(event)
    message = "\n".join(
        (
            MANDATORY_CHAT_HEADINGS[0],
            "The deterministic screening did not mark this cycle as a suspected "
            "deviation. This chat did not start an agent or model run.",
            "",
            MANDATORY_CHAT_HEADINGS[1],
            "No suspected deviation was triggered for this cycle.",
            "",
            MANDATORY_CHAT_HEADINGS[2],
            "Continue normal local review practices and record observations if needed.",
            "",
            MANDATORY_CHAT_HEADINGS[3],
            "Confirm locally whether the selected cycle and reference context are correct.",
            "",
            MANDATORY_CHAT_HEADINGS[4],
            "Escalate only according to local SOP and authorized human review.",
        )
    )
    trace = AgentRunTrace(
        event_id=event.event_id,
        execution_mode="not_triggered",
        tool_names=(),
        tool_calls=(),
        fallback_reason=None,
        completed=True,
    )
    return CopilotChatResponse(
        event_id=event.event_id,
        execution_mode="deterministic_fallback",
        assistant_message=message,
        knowledge_source_ids=tuple(item.guidance_id for item in guidance),
        tool_trace=trace,
        human_approval_required=True,
        fallback_reason=None,
    )


def _knowledge_ids_for_event(
    event_id: str,
    history_path: str | Path,
) -> tuple[str, ...]:
    return tuple(
        item.guidance_id for item in retrieve_maintenance_guidance(event_id, history_path)
    )


def _get_persisted_event(
    event_id: str,
    history_path: str | Path,
) -> MonitoringEvent:
    for event in reversed(read_monitoring_events(history_path)):
        if event.event_id == event_id:
            return event
    raise EventNotFoundError(EVENT_NOT_FOUND_MESSAGE)


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


def make_chat_outcome(
    final_text: str,
    tool_calls: tuple[ToolCallTrace, ...],
) -> AgentChatExecutionOutcome:
    """Small helper for tests and local smoke code."""

    return AgentChatExecutionOutcome(final_text=final_text, tool_calls=tool_calls)
