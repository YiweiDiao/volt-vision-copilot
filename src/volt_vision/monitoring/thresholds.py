"""Deterministic educational DTW threshold derivation.

This module derives a transparent threshold from known-good calibration cycles.
It is not an industrially validated thresholding method and does not classify
new cycles, score abnormal cycles, or make deviation decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from volt_vision.monitoring.calibration import CalibrationResult
from volt_vision.monitoring.dtw import compute_power_sample_dtw

DEFAULT_ABSOLUTE_MARGIN = 0.02
DEFAULT_RELATIVE_MARGIN = 0.20


@dataclass(frozen=True)
class ReferenceDistance:
    segment_id: str
    normalized_distance: float


@dataclass(frozen=True)
class ThresholdResult:
    reference_segment_id: str
    calibration_reference_distances: tuple[ReferenceDistance, ...]
    max_normalized_distance: float
    absolute_margin: float
    relative_margin: float
    applied_margin: float
    threshold: float


def derive_dtw_threshold(
    calibration: CalibrationResult,
    *,
    absolute_margin: float = DEFAULT_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_RELATIVE_MARGIN,
) -> ThresholdResult:
    """Derive a deterministic DTW threshold from a calibration result."""

    _validate_margin(absolute_margin, "absolute_margin")
    _validate_margin(relative_margin, "relative_margin")

    reference_cycle = calibration.reference_cycle
    reference_distances = tuple(
        ReferenceDistance(
            segment_id=cycle.segment_id,
            normalized_distance=compute_power_sample_dtw(
                reference_cycle.samples,
                cycle.samples,
            ).normalized_distance,
        )
        for cycle in calibration.calibration_cycles
    )
    _validate_reference_distances(reference_distances)

    max_normalized_distance = max(
        distance.normalized_distance for distance in reference_distances
    )
    applied_margin = max(
        absolute_margin,
        relative_margin * max_normalized_distance,
    )
    threshold = max_normalized_distance + applied_margin

    return ThresholdResult(
        reference_segment_id=reference_cycle.segment_id,
        calibration_reference_distances=reference_distances,
        max_normalized_distance=max_normalized_distance,
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
        applied_margin=applied_margin,
        threshold=threshold,
    )


def _validate_margin(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_reference_distances(
    reference_distances: tuple[ReferenceDistance, ...],
) -> None:
    if not reference_distances:
        raise ValueError("calibration reference distances must not be empty")
    for distance in reference_distances:
        if not math.isfinite(distance.normalized_distance):
            raise ValueError("reference distances must be finite")
        if distance.normalized_distance < 0:
            raise ValueError("reference distances must be non-negative")
