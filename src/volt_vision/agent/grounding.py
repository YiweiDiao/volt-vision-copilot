"""Grounding checks for ADK-produced incident recommendations."""

from __future__ import annotations

from pathlib import Path

from volt_vision.agent.fallback import build_deterministic_evidence
from volt_vision.agent.models import InvestigationRecommendation
from volt_vision.agent.policy import CORE_LIMITATIONS
from volt_vision.mcp_server.services import (
    DEFAULT_HISTORY_PATH,
    find_similar_previous_events,
    get_event_metrics,
    retrieve_maintenance_guidance,
)

# Limited explicit prototype guardrail. This is intentionally bounded string
# matching, not a claim of complete semantic safety.
PROHIBITED_TEXT_FRAGMENTS = (
    "confirmed fault",
    "confirm fault",
    "confirmed failure",
    "confirm failure",
    "confirmed tool wear",
    "tool wear confirmed",
    "confirmed root cause",
    "root cause is",
    "root cause:",
    "root cause of",
    "machine has a fault",
    "this is a fault",
    "tool wear is likely",
    "tool wear is the cause",
    "diagnosis is",
    "diagnosed",
    "recommend repair",
    "repair the",
    "replace the",
    "replace tooling",
    "replace the tool",
    "part replacement is required",
    "shutdown the",
    "stop the machine",
    "shut down",
    "automatic shutdown",
    "send plc",
    "perform plc",
    "scada action is required",
    "mes action is required",
    "perform parameter tuning",
    "parameter tuning is required",
    "tune parameter",
    "automatic ticket",
    "create ticket",
    "open a maintenance ticket",
    "create a maintenance ticket",
    "automatic maintenance",
)


def validate_grounded_recommendation(
    candidate: InvestigationRecommendation,
    event_id: str,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
) -> None:
    """Reject valid-shaped ADK output that is not grounded in local evidence."""

    metrics = get_event_metrics(event_id, history_path)
    guidance = retrieve_maintenance_guidance(event_id, history_path)
    similar_events = find_similar_previous_events(
        event_id,
        limit=3,
        history_path=history_path,
    )

    if candidate.event_id != metrics.event_id:
        raise ValueError("ungrounded event ID")
    if candidate.screening_status != metrics.status:
        raise ValueError("ungrounded screening status")
    if candidate.human_approval_required is not True:
        raise ValueError("missing human approval requirement")

    allowed_guidance_ids = {item.guidance_id for item in guidance}
    if not set(candidate.guidance_ids).issubset(allowed_guidance_ids):
        raise ValueError("ungrounded guidance ID")

    allowed_checks = {
        check for item in guidance for check in item.inspection_checks
    }
    if not set(candidate.manual_review_checks).issubset(allowed_checks):
        raise ValueError("ungrounded manual review check")

    allowed_similar_ids = {item.event_id for item in similar_events}
    if not set(candidate.similar_event_ids).issubset(allowed_similar_ids):
        raise ValueError("ungrounded similar event ID")

    if candidate.deterministic_evidence != build_deterministic_evidence(metrics):
        raise ValueError("ungrounded deterministic evidence")

    if candidate.limitations != CORE_LIMITATIONS:
        raise ValueError("missing core safety limitations")

    all_text = "\n".join(
        (
            candidate.headline,
            *candidate.deterministic_evidence,
            *candidate.manual_review_checks,
            candidate.historical_context,
            *candidate.limitations,
        )
    ).lower()
    if any(fragment in all_text for fragment in PROHIBITED_TEXT_FRAGMENTS):
        raise ValueError("prohibited claim or instruction")
