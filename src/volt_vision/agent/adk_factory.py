"""Factory for the optional Google ADK Incident Copilot agent."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from volt_vision.agent.models import InvestigationRecommendation
from volt_vision.agent.policy import (
    APPROVED_MCP_TOOL_NAMES,
    CHAT_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
)
from volt_vision.mcp_server.server import HISTORY_PATH_ENV_VAR


@dataclass(frozen=True)
class IncidentCopilotAgentBundle:
    """ADK agent plus MCP toolset lifecycle handle."""

    agent: Any
    toolset: Any
    tool_names: tuple[str, ...]

    async def close(self) -> None:
        """Release the MCP subprocess/session resources if they were opened."""

        await self.toolset.close()


def create_incident_copilot_agent(
    *,
    model: str | Any,
    history_path: str | Path | None = None,
) -> IncidentCopilotAgentBundle:
    """Construct the official ADK LLM agent with local read-only MCP tools."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\[EXPERIMENTAL\] feature FeatureName\.PLUGGABLE_AUTH.*",
            category=UserWarning,
        )
        from google.adk.agents import LlmAgent
        from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
        from mcp import StdioServerParameters

    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=["-m", "volt_vision.mcp_server.server"],
                env=_server_env(history_path),
                cwd=Path.cwd(),
            ),
            timeout=5.0,
        ),
        tool_filter=list(APPROVED_MCP_TOOL_NAMES),
        use_mcp_resources=False,
    )
    agent = LlmAgent(
        name="incident_copilot_agent",
        description=(
            "Summarizes persisted deterministic event evidence and curated "
            "manual-review guidance for suspected deviations."
        ),
        model=model,
        instruction=SYSTEM_INSTRUCTION,
        tools=[toolset],
        output_schema=InvestigationRecommendation,
        include_contents="none",
    )
    return IncidentCopilotAgentBundle(
        agent=agent,
        toolset=toolset,
        tool_names=APPROVED_MCP_TOOL_NAMES,
    )


def create_chat_copilot_agent(
    *,
    model: str | Any,
    history_path: str | Path | None = None,
) -> IncidentCopilotAgentBundle:
    """Construct the ADK chat agent with local read-only MCP tools."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\[EXPERIMENTAL\] feature FeatureName\.PLUGGABLE_AUTH.*",
            category=UserWarning,
        )
        from google.adk.agents import LlmAgent

    toolset = _create_mcp_toolset(history_path)
    agent = LlmAgent(
        name="chat_copilot_agent",
        description=(
            "Provides safety-bounded natural-language investigation chat for "
            "persisted suspected power-signature deviations."
        ),
        model=model,
        instruction=CHAT_SYSTEM_INSTRUCTION,
        tools=[toolset],
        include_contents="none",
    )
    return IncidentCopilotAgentBundle(
        agent=agent,
        toolset=toolset,
        tool_names=APPROVED_MCP_TOOL_NAMES,
    )


def _create_mcp_toolset(history_path: str | Path | None) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\[EXPERIMENTAL\] feature FeatureName\.PLUGGABLE_AUTH.*",
            category=UserWarning,
        )
        from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
        from mcp import StdioServerParameters

    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=["-m", "volt_vision.mcp_server.server"],
                env=_server_env(history_path),
                cwd=Path.cwd(),
            ),
            timeout=5.0,
        ),
        tool_filter=list(APPROVED_MCP_TOOL_NAMES),
        use_mcp_resources=False,
    )


def _server_env(history_path: str | Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if history_path is not None:
        env[HISTORY_PATH_ENV_VAR] = str(history_path)
    return env
