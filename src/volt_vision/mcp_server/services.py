"""Pure read-only services backing the local MCP server."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypeAlias

from volt_vision.guidance import retrieve_guidance as retrieve_event_guidance
from volt_vision.guidance.models import GuidanceItem
from volt_vision.mcp_server.models import (
    CycleMetricsSummary,
    EventMetricsSummary,
    GuidanceItemSummary,
    ReferenceRelativeIndicatorsSummary,
    SimilarEventSummary,
)
from volt_vision.monitoring.event_log import read_monitoring_events
from volt_vision.monitoring.models import MonitoringEvent

HistoryPath: TypeAlias = str | Path

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parents[3] / "data" / (
    "demo_event_history.jsonl"
)
EVENT_NOT_FOUND_MESSAGE = "event not found"
SIMILARITY_RANKING_UNAVAILABLE_MESSAGE = (
    "similarity ranking unavailable for nonzero distance with zero threshold"
)
SIMILARITY_RANKING_NOTE = (
    "Structured historical ranking only; this does not confirm root cause, "
    "tool wear, or a fault."
)


class EventNotFoundError(LookupError):
    """Raised when no persisted MonitoringEvent exists for an event ID."""


class SimilarityRankingError(ValueError):
    """Raised when structured evidence cannot produce a finite ranking score."""


def get_event_metrics(
    event_id: str,
    history_path: HistoryPath = DEFAULT_HISTORY_PATH,
) -> EventMetricsSummary:
    """Return a JSON-safe summary of a persisted event without recomputing it."""

    event = get_monitoring_event(event_id, history_path)
    return _event_metrics_summary(event)


def retrieve_maintenance_guidance(
    event_id: str,
    history_path: HistoryPath = DEFAULT_HISTORY_PATH,
) -> tuple[GuidanceItemSummary, ...]:
    """Return deterministic curated guidance for a persisted event."""

    event = get_monitoring_event(event_id, history_path)
    return tuple(_guidance_summary(item) for item in retrieve_event_guidance(event))


def find_similar_previous_events(
    event_id: str,
    limit: int = 3,
    history_path: HistoryPath = DEFAULT_HISTORY_PATH,
) -> tuple[SimilarEventSummary, ...]:
    """Rank earlier same-machine, same-status persisted events by evidence shape."""

    _validate_limit(limit)
    events = read_monitoring_events(history_path)
    if not events:
        return ()

    latest_by_event_id = _collapse_latest_events(events)
    query_event = latest_by_event_id.get(event_id)
    if query_event is None:
        raise EventNotFoundError(EVENT_NOT_FOUND_MESSAGE)

    candidate_events: list[MonitoringEvent] = []
    for event in latest_by_event_id.values():
        if event.event_id == query_event.event_id:
            continue
        if event.machine_id != query_event.machine_id:
            continue
        if event.status != query_event.status:
            continue
        if event.event_timestamp >= query_event.event_timestamp:
            continue
        candidate_events.append(event)

    ranked = [
        _similar_event_summary(
            event,
            ranking_score=_ranking_score(query_event, event),
        )
        for event in candidate_events
    ]
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                item.ranking_score,
                -item.event_timestamp.timestamp(),
                item.event_id,
            ),
        )[:limit]
    )


def get_monitoring_event(
    event_id: str,
    history_path: HistoryPath = DEFAULT_HISTORY_PATH,
) -> MonitoringEvent:
    """Return the most recently appended event with the requested event ID."""

    events = read_monitoring_events(history_path)
    event = _find_latest_event(event_id, events)
    if event is None:
        raise EventNotFoundError(EVENT_NOT_FOUND_MESSAGE)
    return event


def _find_latest_event(
    event_id: str,
    events: tuple[MonitoringEvent, ...],
) -> MonitoringEvent | None:
    for event in reversed(events):
        if event.event_id == event_id:
            return event
    return None


def _collapse_latest_events(
    events: tuple[MonitoringEvent, ...],
) -> dict[str, MonitoringEvent]:
    latest_by_event_id: dict[str, MonitoringEvent] = {}
    for event in events:
        latest_by_event_id[event.event_id] = event
    return latest_by_event_id


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer from 1 through 5")
    if limit < 1 or limit > 5:
        raise ValueError("limit must be an integer from 1 through 5")


def _ranking_score(query_event: MonitoringEvent, prior_event: MonitoringEvent) -> float:
    components = [
        abs(_dtw_band_ratio(query_event) - _dtw_band_ratio(prior_event)),
    ]
    indicator_pairs = (
        (
            query_event.indicators.duration_deviation_pct,
            prior_event.indicators.duration_deviation_pct,
        ),
        (
            query_event.indicators.energy_deviation_pct,
            prior_event.indicators.energy_deviation_pct,
        ),
        (
            query_event.indicators.peak_power_deviation_pct,
            prior_event.indicators.peak_power_deviation_pct,
        ),
    )
    components.extend(
        abs(query_value - prior_value) / 100
        for query_value, prior_value in indicator_pairs
        if query_value is not None and prior_value is not None
    )
    score = sum(components) / len(components)
    if not math.isfinite(score):
        raise SimilarityRankingError(SIMILARITY_RANKING_UNAVAILABLE_MESSAGE)
    return score


def _dtw_band_ratio(event: MonitoringEvent) -> float:
    if event.threshold == 0:
        if event.normalized_dtw_distance == 0:
            return 0.0
        raise SimilarityRankingError(SIMILARITY_RANKING_UNAVAILABLE_MESSAGE)
    return event.normalized_dtw_distance / event.threshold


def _event_metrics_summary(event: MonitoringEvent) -> EventMetricsSummary:
    return EventMetricsSummary(
        event_id=event.event_id,
        event_type=event.event_type,
        event_timestamp=event.event_timestamp,
        machine_id=event.machine_id,
        candidate_segment_id=event.candidate_segment_id,
        reference_segment_id=event.reference_segment_id,
        status=event.status,
        recommended_action=event.recommended_action,
        evidence=event.evidence,
        normalized_dtw_distance=event.normalized_dtw_distance,
        threshold=event.threshold,
        metrics=CycleMetricsSummary.model_validate(
            event.metrics.model_dump(mode="python")
        ),
        indicators=_indicator_summary(event),
    )


def _guidance_summary(item: GuidanceItem) -> GuidanceItemSummary:
    return GuidanceItemSummary(
        guidance_id=item.guidance_id,
        title=item.title,
        applies_to=item.applies_to,
        inspection_checks=item.inspection_checks,
        evidence_to_record=item.evidence_to_record,
        escalation_condition=item.escalation_condition,
        safety_note=item.safety_note,
    )


def _similar_event_summary(
    event: MonitoringEvent,
    *,
    ranking_score: float,
) -> SimilarEventSummary:
    return SimilarEventSummary(
        event_id=event.event_id,
        event_timestamp=event.event_timestamp,
        machine_id=event.machine_id,
        status=event.status,
        normalized_dtw_distance=event.normalized_dtw_distance,
        threshold=event.threshold,
        indicators=_indicator_summary(event),
        ranking_score=ranking_score,
        ranking_note=SIMILARITY_RANKING_NOTE,
    )


def _indicator_summary(event: MonitoringEvent) -> ReferenceRelativeIndicatorsSummary:
    return ReferenceRelativeIndicatorsSummary.model_validate(
        event.indicators.model_dump(mode="python")
    )
