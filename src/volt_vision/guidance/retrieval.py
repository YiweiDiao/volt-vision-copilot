"""Deterministic guidance retrieval from structured monitoring events."""

from __future__ import annotations

from volt_vision.guidance.catalog import (
    CYCLE_DURATION_REVIEW,
    ENERGY_AND_PEAK_REVIEW,
    ESCALATION_AND_RECORDING,
    POWER_SIGNATURE_REVIEW,
)
from volt_vision.guidance.models import GuidanceItem
from volt_vision.monitoring.models import MonitoringEvent

INDICATOR_TRIGGER_PERCENT = 10.0


def retrieve_guidance(event: MonitoringEvent) -> tuple[GuidanceItem, ...]:
    """Return bounded manual-review guidance for a monitoring event."""

    items: list[GuidanceItem] = [POWER_SIGNATURE_REVIEW]

    if event.status == "suspected_deviation":
        if _triggers_specific_playbook(event.indicators.duration_deviation_pct):
            items.append(CYCLE_DURATION_REVIEW)

        if (
            _triggers_specific_playbook(event.indicators.energy_deviation_pct)
            or _triggers_specific_playbook(event.indicators.peak_power_deviation_pct)
        ):
            items.append(ENERGY_AND_PEAK_REVIEW)

    items.append(ESCALATION_AND_RECORDING)
    return tuple(items)


def _triggers_specific_playbook(value: float | None) -> bool:
    return value is not None and abs(value) >= INDICATOR_TRIGGER_PERCENT
