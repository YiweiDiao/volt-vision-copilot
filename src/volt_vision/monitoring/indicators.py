"""Reference-relative explanatory indicators for educational screening.

These percentages compare already-computed candidate metrics against a selected
reference cycle. They are transparent review context only, not a diagnosis or a
separate decision rule.
"""

from __future__ import annotations

import math

from volt_vision.monitoring.models import CycleMetrics, ReferenceRelativeIndicators


def calculate_reference_relative_indicators(
    reference_metrics: CycleMetrics,
    candidate_metrics: CycleMetrics,
) -> ReferenceRelativeIndicators:
    """Calculate signed percentage differences from reference to candidate."""

    if reference_metrics.machine_id != candidate_metrics.machine_id:
        raise ValueError("reference and candidate metrics must use the same machine_id")

    _validate_metrics(reference_metrics, "reference")
    _validate_metrics(candidate_metrics, "candidate")

    return ReferenceRelativeIndicators(
        reference_cycle_id=reference_metrics.cycle_id,
        candidate_cycle_id=candidate_metrics.cycle_id,
        duration_deviation_pct=_relative_deviation_pct(
            reference_metrics.duration_seconds,
            candidate_metrics.duration_seconds,
        ),
        energy_deviation_pct=_relative_deviation_pct(
            reference_metrics.energy_kwh,
            candidate_metrics.energy_kwh,
        ),
        peak_power_deviation_pct=_relative_deviation_pct(
            reference_metrics.peak_power_kw,
            candidate_metrics.peak_power_kw,
        ),
    )


def _relative_deviation_pct(reference_value: float, candidate_value: float) -> float | None:
    if reference_value == 0:
        return None
    deviation = (candidate_value - reference_value) / reference_value * 100.0
    if not math.isfinite(deviation):
        raise ValueError("reference-relative deviation must be finite")
    return deviation


def _validate_metrics(metrics: CycleMetrics, role: str) -> None:
    numeric_values = {
        "duration_seconds": float(metrics.duration_seconds),
        "energy_kwh": float(metrics.energy_kwh),
        "average_power_kw": float(metrics.average_power_kw),
        "peak_power_kw": float(metrics.peak_power_kw),
        "sample_count": float(metrics.sample_count),
    }
    for field_name, value in numeric_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{role} metric {field_name} must be finite")
        if value < 0:
            raise ValueError(f"{role} metric {field_name} must be non-negative")
