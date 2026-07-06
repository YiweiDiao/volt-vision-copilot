"""Typed JSON-safe payloads returned by read-only MCP services."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CycleMetricsSummary(BaseModel):
    """Structured cycle metrics without raw power samples."""

    model_config = ConfigDict(frozen=True)

    cycle_id: str
    machine_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float
    energy_kwh: float
    average_power_kw: float
    peak_power_kw: float
    sample_count: int


class ReferenceRelativeIndicatorsSummary(BaseModel):
    """Reference-relative indicators used as explanatory evidence only."""

    model_config = ConfigDict(frozen=True)

    reference_cycle_id: str
    candidate_cycle_id: str
    duration_deviation_pct: float | None
    energy_deviation_pct: float | None
    peak_power_deviation_pct: float | None


class EventMetricsSummary(BaseModel):
    """Persisted deterministic event summary safe for MCP responses."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_type: Literal["cycle_screening"]
    event_timestamp: datetime
    machine_id: str
    candidate_segment_id: str
    reference_segment_id: str
    status: Literal["within_normal_band", "suspected_deviation"]
    recommended_action: Literal["no_automated_action", "manual_review_required"]
    evidence: str
    normalized_dtw_distance: float
    threshold: float
    metrics: CycleMetricsSummary
    indicators: ReferenceRelativeIndicatorsSummary


class GuidanceItemSummary(BaseModel):
    """JSON-safe curated guidance item."""

    model_config = ConfigDict(frozen=True)

    guidance_id: str
    title: str
    applies_to: tuple[str, ...]
    inspection_checks: tuple[str, ...]
    evidence_to_record: tuple[str, ...]
    escalation_condition: str
    safety_note: str


class SimilarEventSummary(BaseModel):
    """Historical structured-event ranking result."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    event_timestamp: datetime
    machine_id: str
    status: Literal["within_normal_band", "suspected_deviation"]
    normalized_dtw_distance: float
    threshold: float
    indicators: ReferenceRelativeIndicatorsSummary
    ranking_score: float = Field(ge=0)
    ranking_note: str
