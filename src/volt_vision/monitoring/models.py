"""Typed monitoring payloads for deterministic power-cycle analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PowerSample(BaseModel):
    """One measured power point for a machine."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    machine_id: str = Field(min_length=1)
    power_kw: float = Field(ge=0)


class CycleMetrics(BaseModel):
    """Deterministic summary metrics for one machine cycle."""

    model_config = ConfigDict(frozen=True)

    cycle_id: str = Field(min_length=1)
    machine_id: str = Field(min_length=1)
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float = Field(ge=0)
    energy_kwh: float = Field(ge=0)
    average_power_kw: float = Field(ge=0)
    peak_power_kw: float = Field(ge=0)
    sample_count: int = Field(ge=1)


class ReferenceRelativeIndicators(BaseModel):
    """Explanatory metrics comparing one candidate cycle to its reference."""

    model_config = ConfigDict(frozen=True)

    reference_cycle_id: str = Field(min_length=1)
    candidate_cycle_id: str = Field(min_length=1)
    duration_deviation_pct: float | None
    energy_deviation_pct: float | None
    peak_power_deviation_pct: float | None


class MonitoringEvent(BaseModel):
    """Structured deterministic monitoring event for bounded review workflows."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(min_length=1)
    event_type: Literal["cycle_screening"]
    event_timestamp: datetime
    machine_id: str = Field(min_length=1)
    candidate_segment_id: str = Field(min_length=1)
    reference_segment_id: str = Field(min_length=1)
    status: Literal["within_normal_band", "suspected_deviation"]
    recommended_action: Literal["no_automated_action", "manual_review_required"]
    evidence: str = Field(min_length=1)
    normalized_dtw_distance: float = Field(ge=0)
    threshold: float = Field(ge=0)
    metrics: CycleMetrics
    indicators: ReferenceRelativeIndicators
