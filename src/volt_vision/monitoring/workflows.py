"""Thin deterministic CSV workflows for the educational monitoring prototype.

These helpers compose local CSV ingestion, DTW calibration, threshold
derivation, and event building. They are intended for safe educational
screening only, not industrial diagnosis or automatic machine action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from volt_vision.monitoring.calibration import (
    CalibrationResult,
    calibrate_reference_template,
)
from volt_vision.monitoring.csv_ingestion import (
    load_calibration_cycles_from_csv,
    load_candidate_cycle_from_csv,
)
from volt_vision.monitoring.cycles import SelectedCycle
from volt_vision.monitoring.events import build_monitoring_event
from volt_vision.monitoring.models import MonitoringEvent
from volt_vision.monitoring.thresholds import (
    DEFAULT_ABSOLUTE_MARGIN,
    DEFAULT_RELATIVE_MARGIN,
    ThresholdResult,
    derive_dtw_threshold,
)


@dataclass(frozen=True)
class CalibrationBundle:
    calibration_cycles: tuple[SelectedCycle, ...]
    calibration_result: CalibrationResult
    threshold_result: ThresholdResult


def build_calibration_bundle_from_csv(
    csv_paths: Sequence[str | Path],
    *,
    calibration_segment_ids: Sequence[str] | None = None,
    absolute_margin: float = DEFAULT_ABSOLUTE_MARGIN,
    relative_margin: float = DEFAULT_RELATIVE_MARGIN,
) -> CalibrationBundle:
    """Load known-good CSV cycles and derive a deterministic DTW threshold."""

    calibration_cycles = load_calibration_cycles_from_csv(
        csv_paths,
        calibration_segment_ids=calibration_segment_ids,
    )
    calibration_result = calibrate_reference_template(calibration_cycles)
    threshold_result = derive_dtw_threshold(
        calibration_result,
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
    )

    return CalibrationBundle(
        calibration_cycles=calibration_cycles,
        calibration_result=calibration_result,
        threshold_result=threshold_result,
    )


def build_monitoring_event_from_candidate_csv(
    csv_path: str | Path,
    calibration_bundle: CalibrationBundle,
    *,
    candidate_segment_id: str = "uploaded_cycle",
) -> MonitoringEvent:
    """Load one candidate CSV cycle and build a structured monitoring event."""

    candidate_cycle = load_candidate_cycle_from_csv(
        csv_path,
        candidate_segment_id=candidate_segment_id,
    )
    return build_monitoring_event(
        candidate_cycle,
        calibration_bundle.calibration_result,
        calibration_bundle.threshold_result,
    )
