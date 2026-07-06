"""Synchronous adapter for executing the optional official ADK agent."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
from typing import Any

from volt_vision.agent.models import (
    AgentChatExecutionOutcome,
    AgentExecutionOutcome,
    ToolCallTrace,
)
from volt_vision.agent.policy import APPROVED_MCP_TOOL_NAMES

AgentExecutor = Callable[[Any, str], AgentExecutionOutcome]
ChatAgentExecutor = Callable[[Any, str], AgentChatExecutionOutcome]


@dataclass(frozen=True)
class ChatDiagnosticExecutionOutcome:
    """Raw diagnostic execution summary without prompts or payloads."""

    final_text: str | None
    tool_calls: tuple[ToolCallTrace, ...]
    adk_event_count: int


def execute_incident_copilot_bundle(
    bundle: Any,
    event_id: str,
    *,
    agent_executor: AgentExecutor | None = None,
) -> AgentExecutionOutcome:
    """Run the ADK agent and close its MCP bundle in one async lifecycle."""

    return _run_async(
        _execute_incident_copilot_bundle_async(
            bundle,
            event_id,
            agent_executor=agent_executor,
        )
    )


async def _execute_incident_copilot_bundle_async(
    bundle: Any,
    event_id: str,
    *,
    agent_executor: AgentExecutor | None = None,
) -> AgentExecutionOutcome:
    try:
        if agent_executor is not None:
            return AgentExecutionOutcome.model_validate(
                agent_executor(bundle.agent, event_id)
            )
        return await _execute_incident_copilot_agent_async(bundle.agent, event_id)
    finally:
        await bundle.close()


def execute_incident_copilot_agent(agent: Any, event_id: str) -> AgentExecutionOutcome:
    """Run an ADK agent without owning a bundle lifecycle."""

    return _run_async(_execute_incident_copilot_agent_async(agent, event_id))


def execute_chat_copilot_bundle(
    bundle: Any,
    event_id: str,
    *,
    agent_executor: ChatAgentExecutor | None = None,
) -> AgentChatExecutionOutcome:
    """Run the ADK chat agent and close its MCP bundle in one async lifecycle."""

    return _run_async(
        _execute_chat_copilot_bundle_async(
            bundle,
            event_id,
            agent_executor=agent_executor,
        )
    )


async def _execute_chat_copilot_bundle_async(
    bundle: Any,
    event_id: str,
    *,
    agent_executor: ChatAgentExecutor | None = None,
) -> AgentChatExecutionOutcome:
    try:
        if agent_executor is not None:
            return AgentChatExecutionOutcome.model_validate(
                agent_executor(bundle.agent, event_id)
            )
        return await _execute_chat_copilot_agent_async(bundle.agent, event_id)
    finally:
        await bundle.close()


def execute_chat_copilot_agent(agent: Any, event_id: str) -> AgentChatExecutionOutcome:
    """Run a chat ADK agent without owning a bundle lifecycle."""

    return _run_async(_execute_chat_copilot_agent_async(agent, event_id))


def execute_chat_copilot_diagnostic_bundle(
    bundle: Any,
    event_id: str,
) -> ChatDiagnosticExecutionOutcome:
    """Run the chat agent for bounded diagnostics without validating text."""

    return _run_async(_execute_chat_copilot_diagnostic_bundle_async(bundle, event_id))


async def _execute_chat_copilot_diagnostic_bundle_async(
    bundle: Any,
    event_id: str,
) -> ChatDiagnosticExecutionOutcome:
    try:
        return await _execute_chat_copilot_diagnostic_agent_async(bundle.agent, event_id)
    finally:
        await bundle.close()


async def _execute_chat_copilot_diagnostic_agent_async(
    agent: Any,
    event_id: str,
) -> ChatDiagnosticExecutionOutcome:
    final_output, tool_calls, event_count = await _run_adk_agent_for_diagnostics(
        agent,
        event_id,
    )
    final_text = final_output if isinstance(final_output, str) else None
    return ChatDiagnosticExecutionOutcome(
        final_text=final_text,
        tool_calls=tool_calls,
        adk_event_count=event_count,
    )


async def _execute_chat_copilot_agent_async(
    agent: Any,
    event_id: str,
) -> AgentChatExecutionOutcome:
    text, tool_calls = await _run_adk_agent_for_final_output(agent, event_id)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ADK agent returned no final chat text")
    return AgentChatExecutionOutcome(final_text=text, tool_calls=tool_calls)


async def _execute_incident_copilot_agent_async(
    agent: Any,
    event_id: str,
) -> AgentExecutionOutcome:
    final_output, tool_calls = await _run_adk_agent_for_final_output(agent, event_id)
    if final_output is None:
        raise ValueError("ADK agent returned no final structured output")
    return AgentExecutionOutcome(
        output_payload=_coerce_structured_output(final_output),
        tool_calls=tool_calls,
    )


async def _run_adk_agent_for_final_output(
    agent: Any,
    event_id: str,
) -> tuple[Any, tuple[ToolCallTrace, ...]]:
    final_output, tool_calls, _ = await _run_adk_agent_for_diagnostics(agent, event_id)
    return final_output, tool_calls


async def _run_adk_agent_for_diagnostics(
    agent: Any,
    event_id: str,
) -> tuple[Any, tuple[ToolCallTrace, ...], int]:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    app_name = "volt_vision_incident_copilot"
    user_id = "local_operator"
    session_id = f"incident-{uuid.uuid4().hex}"
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    runner = Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
    )
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=event_id)],
    )
    final_output: Any = None
    tool_calls: list[ToolCallTrace] = []
    event_count = 0
    async for adk_event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        event_count += 1
        tool_calls.extend(_tool_traces_from_event(adk_event))
        if adk_event.is_final_response():
            final_output = _extract_event_output(adk_event)

    return final_output, tuple(tool_calls), event_count


def _tool_traces_from_event(adk_event: Any) -> tuple[ToolCallTrace, ...]:
    traces: list[ToolCallTrace] = []
    for response in adk_event.get_function_responses():
        tool_name = getattr(response, "name", None)
        if tool_name not in APPROVED_MCP_TOOL_NAMES:
            raise ValueError("unapproved MCP tool response observed")
        failed = _function_response_failed(response)
        traces.append(
            ToolCallTrace(
                tool_name=tool_name,
                source="mcp",
                outcome="failed" if failed else "succeeded",
                error_code="tool_execution_failed" if failed else None,
            )
        )
    return tuple(traces)


def _function_response_failed(response: Any) -> bool:
    payload = getattr(response, "response", None)
    if not isinstance(payload, dict):
        return False
    failure_keys = ("error", "isError", "is_error", "errorCode", "error_code")
    return any(bool(payload.get(key)) for key in failure_keys)


def _coerce_structured_output(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if hasattr(output, "model_dump"):
        return output.model_dump(mode="python")
    if isinstance(output, str):
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("ADK agent output was not a structured object")


def _extract_event_output(adk_event: Any) -> Any:
    output = getattr(adk_event, "output", None)
    if output is not None:
        return output

    content = getattr(adk_event, "content", None)
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    text_parts = [
        part.text
        for part in parts
        if getattr(part, "text", None) and not getattr(part, "thought", False)
    ]
    if not text_parts:
        return None
    return "\n".join(text_parts)


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    error: list[BaseException] = []

    def run_in_thread() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - defensive adapter path
            error.append(exc)

    thread = Thread(target=run_in_thread)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]
