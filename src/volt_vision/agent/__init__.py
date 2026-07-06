"""Incident copilot agent package for bounded event investigation."""

from volt_vision.agent.models import (
    AgentExecutionOutcome,
    AgentRunTrace,
    CopilotChatResponse,
    InvestigationRecommendation,
    InvestigationResult,
    ToolCallTrace,
)
from volt_vision.agent.runner import run_incident_copilot
from volt_vision.agent.chat import run_chat_copilot
from volt_vision.agent.saia import (
    DEFAULT_SAIA_BASE_URL,
    DEFAULT_SAIA_MAX_TOKENS,
    SaiaSettings,
    load_saia_settings_from_env,
    resolve_optional_saia_model_from_env,
    safe_saia_configuration_status,
)

__all__ = [
    "AgentExecutionOutcome",
    "AgentRunTrace",
    "CopilotChatResponse",
    "DEFAULT_SAIA_BASE_URL",
    "DEFAULT_SAIA_MAX_TOKENS",
    "InvestigationRecommendation",
    "InvestigationResult",
    "SaiaSettings",
    "ToolCallTrace",
    "load_saia_settings_from_env",
    "resolve_optional_saia_model_from_env",
    "run_chat_copilot",
    "run_incident_copilot",
    "safe_saia_configuration_status",
]
