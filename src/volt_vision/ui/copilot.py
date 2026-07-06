"""Pure helpers for Streamlit Copilot investigation and feedback UI."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from volt_vision.agent.models import CopilotChatResponse, InvestigationResult
from volt_vision.feedback.feedback_log import create_copilot_feedback_record
from volt_vision.feedback.models import CopilotFeedbackRecord
from volt_vision.monitoring.models import MonitoringEvent

FeedbackOutcome = str
CopilotUiResult = InvestigationResult | CopilotChatResponse

FEEDBACK_HISTORY_COLUMNS = [
    "recorded_at",
    "event_id",
    "screening_status",
    "execution_mode",
    "feedback_outcome",
    "human_review_acknowledged",
]


def latest_events_by_id(
    events: Sequence[MonitoringEvent],
) -> tuple[MonitoringEvent, ...]:
    """Collapse duplicate event IDs, retaining each latest persisted record."""

    latest: dict[str, MonitoringEvent] = {}
    for event in events:
        latest[event.event_id] = event
    return tuple(latest.values())


def persisted_event_ids(events: Sequence[MonitoringEvent]) -> tuple[str, ...]:
    """Return event IDs eligible for selection from persisted local history."""

    return tuple(event.event_id for event in latest_events_by_id(events))


def can_investigate_event(event: MonitoringEvent | None) -> bool:
    """Return whether Copilot can investigate the persisted event."""

    return event is not None and event.status == "suspected_deviation"


def get_selected_persisted_event(
    events: Sequence[MonitoringEvent],
    event_id: str | None,
) -> MonitoringEvent | None:
    """Return the latest persisted event matching a selected event ID."""

    if event_id is None:
        return None
    for event in latest_events_by_id(events):
        if event.event_id == event_id:
            return event
    return None


def is_current_result_for_event(
    result: CopilotUiResult | None,
    event: MonitoringEvent | None,
) -> bool:
    """Return whether a session result still matches the persisted event."""

    if result is None or event is None:
        return False
    if event.status != "suspected_deviation":
        return False
    if isinstance(result, CopilotChatResponse):
        return (
            result.event_id == event.event_id
            and result.tool_trace.event_id == event.event_id
        )
    return (
        result.recommendation.event_id == event.event_id
        and result.trace.event_id == event.event_id
    )


def can_record_feedback(
    event: MonitoringEvent | None,
    result: CopilotUiResult | None,
    *,
    human_review_acknowledged: bool,
) -> bool:
    """Return whether explicit local feedback can be recorded."""

    return (
        is_current_result_for_event(result, event)
        and result is not None
        and _human_approval_required(result) is True
        and human_review_acknowledged is True
    )


def build_copilot_feedback_record(
    *,
    event: MonitoringEvent,
    result: CopilotUiResult,
    feedback_outcome: FeedbackOutcome,
    human_review_acknowledged: bool,
) -> CopilotFeedbackRecord:
    """Build one bounded feedback record without operational side effects."""

    if not can_record_feedback(
        event,
        result,
        human_review_acknowledged=human_review_acknowledged,
    ):
        raise ValueError("feedback is not eligible for the selected event")

    return create_copilot_feedback_record(
        event_id=event.event_id,
        screening_status=event.status,
        execution_mode=_execution_mode(result),
        feedback_outcome=feedback_outcome,
        human_review_acknowledged=human_review_acknowledged,
    )


def build_feedback_history_frame(
    records: Sequence[CopilotFeedbackRecord],
) -> pd.DataFrame:
    """Build a compact safe feedback-history table."""

    rows = [
        {
            "recorded_at": record.recorded_at.isoformat(),
            "event_id": record.event_id,
            "screening_status": record.screening_status,
            "execution_mode": record.execution_mode,
            "feedback_outcome": record.feedback_outcome,
            "human_review_acknowledged": record.human_review_acknowledged,
        }
        for record in records
    ]
    return pd.DataFrame(rows, columns=FEEDBACK_HISTORY_COLUMNS)


def _execution_mode(result: CopilotUiResult) -> str:
    if isinstance(result, CopilotChatResponse):
        return result.execution_mode
    return result.trace.execution_mode


def _human_approval_required(result: CopilotUiResult) -> bool:
    if isinstance(result, CopilotChatResponse):
        return result.human_approval_required
    return result.recommendation.human_approval_required
