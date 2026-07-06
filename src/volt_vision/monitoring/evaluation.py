"""Deterministic educational screening of one cycle against a DTW threshold.

This module reports whether a measured cycle is within a calibrated normal
band or is a suspected deviation. It is not a confirmed industrial diagnosis
and does not claim a fault, failure, tool wear, or root cause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from volt_vision.monitoring.calibration import CalibrationResult
from volt_vision.monitoring.cycles import SelectedCycle
from volt_vision.monitoring.dtw import compute_power_sample_dtw
from volt_vision.monitoring.thresholds import ThresholdResult

EvaluationStatus = Literal["within_normal_band", "suspected_deviation"]


@dataclass(frozen=True)
class CycleEvaluation:
    segment_id: str
    machine_id: str
    reference_segment_id: str
    normalized_dtw_distance: float
    threshold: float
    within_calibrated_normal_band: bool
    status: EvaluationStatus
    evidence: str


def evaluate_cycle_against_threshold(
    cycle: SelectedCycle,
    calibration: CalibrationResult,
    threshold_result: ThresholdResult,
) -> CycleEvaluation:
    """Evaluate one candidate cycle against an existing calibrated threshold."""

    _validate_threshold_reference(calibration, threshold_result)
    _validate_threshold(threshold_result.threshold)

    candidate_machine_id = _single_machine_id(cycle)
    reference_machine_id = _single_machine_id(calibration.reference_cycle)
    if candidate_machine_id != reference_machine_id:
        raise ValueError(
            "candidate cycle machine_id must match calibration reference machine_id"
        )

    normalized_distance = compute_power_sample_dtw(
        calibration.reference_cycle.samples,
        cycle.samples,
    ).normalized_distance
    if not math.isfinite(normalized_distance) or normalized_distance < 0:
        raise ValueError("normalized DTW distance must be finite and non-negative")

    within_band = normalized_distance <= threshold_result.threshold
    status: EvaluationStatus = (
        "within_normal_band" if within_band else "suspected_deviation"
    )
    evidence = _format_evidence(normalized_distance, threshold_result.threshold, within_band)

    return CycleEvaluation(
        segment_id=cycle.segment_id,
        machine_id=candidate_machine_id,
        reference_segment_id=calibration.reference_cycle.segment_id,
        normalized_dtw_distance=normalized_distance,
        threshold=threshold_result.threshold,
        within_calibrated_normal_band=within_band,
        status=status,
        evidence=evidence,
    )


def _validate_threshold_reference(
    calibration: CalibrationResult,
    threshold_result: ThresholdResult,
) -> None:
    if threshold_result.reference_segment_id != calibration.reference_cycle.segment_id:
        raise ValueError(
            "threshold reference_segment_id must match calibration reference cycle"
        )


def _validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")


def _single_machine_id(cycle: SelectedCycle) -> str:
    if not cycle.samples:
        raise ValueError(f"candidate cycle has no samples: {cycle.segment_id}")

    machine_ids = {sample.machine_id for sample in cycle.samples}
    if len(machine_ids) != 1:
        raise ValueError(
            f"candidate cycle samples must use exactly one machine_id: "
            f"{cycle.segment_id}"
        )
    return next(iter(machine_ids))


def _format_evidence(
    normalized_distance: float,
    threshold: float,
    within_band: bool,
) -> str:
    comparison = "is within" if within_band else "exceeds"
    return (
        f"Normalized DTW distance {normalized_distance:.6f} "
        f"{comparison} calibrated threshold {threshold:.6f}."
    )
