"""Deterministic monitoring-event assembly for the educational prototype.

The resulting event is a structured screening payload for review. It is not an
industrial diagnosis and does not claim a confirmed fault, failure, tool wear,
maintenance action, or root cause.
"""

from __future__ import annotations

from typing import Literal

from volt_vision.monitoring.calibration import CalibrationResult
from volt_vision.monitoring.cycles import SelectedCycle
from volt_vision.monitoring.evaluation import evaluate_cycle_against_threshold
from volt_vision.monitoring.indicators import calculate_reference_relative_indicators
from volt_vision.monitoring.metrics import compute_cycle_metrics
from volt_vision.monitoring.models import MonitoringEvent
from volt_vision.monitoring.thresholds import ThresholdResult

RecommendedAction = Literal["no_automated_action", "manual_review_required"]


def build_monitoring_event(
    cycle: SelectedCycle,
    calibration: CalibrationResult,
    threshold_result: ThresholdResult,
) -> MonitoringEvent:
    """Build one deterministic event from cycle metrics and DTW evaluation."""

    evaluation = evaluate_cycle_against_threshold(
        cycle,
        calibration,
        threshold_result,
    )
    metrics = compute_cycle_metrics(cycle.samples, cycle_id=cycle.segment_id)
    reference_metrics = compute_cycle_metrics(
        calibration.reference_cycle.samples,
        cycle_id=calibration.reference_cycle.segment_id,
    )
    indicators = calculate_reference_relative_indicators(reference_metrics, metrics)
    event_timestamp = metrics.end_timestamp
    event_id = (
        f"{evaluation.machine_id}:"
        f"{cycle.segment_id}:"
        f"{calibration.reference_cycle.segment_id}:"
        f"{event_timestamp.isoformat()}"
    )

    return MonitoringEvent(
        event_id=event_id,
        event_type="cycle_screening",
        event_timestamp=event_timestamp,
        machine_id=evaluation.machine_id,
        candidate_segment_id=cycle.segment_id,
        reference_segment_id=calibration.reference_cycle.segment_id,
        status=evaluation.status,
        recommended_action=_recommended_action_for_status(evaluation.status),
        evidence=evaluation.evidence,
        normalized_dtw_distance=evaluation.normalized_dtw_distance,
        threshold=evaluation.threshold,
        metrics=metrics,
        indicators=indicators,
    )


def _recommended_action_for_status(status: str) -> RecommendedAction:
    if status == "within_normal_band":
        return "no_automated_action"
    if status == "suspected_deviation":
        return "manual_review_required"
    raise ValueError(f"unsupported evaluation status: {status}")
