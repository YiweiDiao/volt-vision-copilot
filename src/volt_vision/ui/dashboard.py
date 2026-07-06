"""Pure dashboard helpers for deterministic power-cycle screening.

The UI layer prepares display objects from existing monitoring services. It
does not diagnose faults, infer root causes, call models, or control machines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from volt_vision.monitoring.calibration import (
    CalibrationResult,
    calibrate_reference_template,
)
from volt_vision.monitoring.cycles import (
    SelectedCycle,
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import DemoTimeline, generate_demo_timeline
from volt_vision.monitoring.events import build_monitoring_event
from volt_vision.monitoring.models import MonitoringEvent
from volt_vision.monitoring.thresholds import ThresholdResult, derive_dtw_threshold

SyntheticCandidate = Literal["normal", "changed"]
DemoOperatingState = Literal["Idle", "Processing"]


@dataclass(frozen=True)
class DashboardResult:
    reference_cycle: SelectedCycle
    candidate_cycle: SelectedCycle
    calibration_cycles: tuple[SelectedCycle, ...]
    calibration_result: CalibrationResult
    threshold_result: ThresholdResult
    event: MonitoringEvent


@dataclass(frozen=True)
class DemoStateInterval:
    start_elapsed_seconds: float
    end_elapsed_seconds: float
    state: DemoOperatingState


@dataclass(frozen=True)
class DemoReplayData:
    samples: pd.DataFrame
    state_intervals: tuple[DemoStateInterval, ...]
    max_elapsed_seconds: float


def run_synthetic_demo_workflow(
    *,
    candidate: SyntheticCandidate,
) -> DashboardResult:
    """Run the deterministic synthetic demo workflow for one candidate choice."""

    timeline = generate_demo_timeline()
    calibration_cycles = select_calibration_cycles(timeline)
    calibration_result = calibrate_reference_template(calibration_cycles)
    threshold_result = derive_dtw_threshold(calibration_result)

    if candidate == "normal":
        candidate_cycle = calibration_cycles[0]
    elif candidate == "changed":
        candidate_cycle = select_abnormal_evaluation_cycle(timeline)
    else:
        raise ValueError("candidate must be 'normal' or 'changed'")

    event = build_monitoring_event(
        candidate_cycle,
        calibration_result,
        threshold_result,
    )

    return DashboardResult(
        reference_cycle=calibration_result.reference_cycle,
        candidate_cycle=candidate_cycle,
        calibration_cycles=calibration_cycles,
        calibration_result=calibration_result,
        threshold_result=threshold_result,
        event=event,
    )


def build_demo_replay_data(timeline: DemoTimeline | None = None) -> DemoReplayData:
    """Build raw synthetic timeline data and neutral state intervals.

    Segment boundaries are inclusive. A cursor exactly at a segment start belongs
    to the segment beginning there; a cursor exactly at a segment end belongs to
    the segment ending there.
    """

    demo_timeline = timeline if timeline is not None else generate_demo_timeline()
    if not demo_timeline.samples:
        raise ValueError("demo timeline must contain samples")

    timeline_start = demo_timeline.samples[0].timestamp
    sample_rows = [
        {
            "elapsed_seconds": (sample.timestamp - timeline_start).total_seconds(),
            "power_kw": sample.power_kw,
        }
        for sample in demo_timeline.samples
    ]
    intervals = tuple(
        DemoStateInterval(
            start_elapsed_seconds=(
                segment.start_timestamp - timeline_start
            ).total_seconds(),
            end_elapsed_seconds=(segment.end_timestamp - timeline_start).total_seconds(),
            state=_display_state_for_segment_type(segment.segment_type),
        )
        for segment in demo_timeline.segments
    )

    return DemoReplayData(
        samples=pd.DataFrame(sample_rows, columns=["elapsed_seconds", "power_kw"]),
        state_intervals=intervals,
        max_elapsed_seconds=sample_rows[-1]["elapsed_seconds"],
    )


def get_demo_operating_state_at_elapsed_seconds(
    replay_data: DemoReplayData,
    elapsed_seconds: float,
) -> DemoOperatingState:
    """Return the neutral synthetic state at a replay cursor position."""

    for interval in replay_data.state_intervals:
        if (
            interval.start_elapsed_seconds
            <= elapsed_seconds
            <= interval.end_elapsed_seconds
        ):
            return interval.state
    raise ValueError("elapsed_seconds is outside the synthetic demo timeline")


def build_power_comparison_frame(
    reference_cycle: SelectedCycle,
    candidate_cycle: SelectedCycle,
    *,
    reference_label: str = "Reference",
    candidate_label: str = "Candidate",
) -> pd.DataFrame:
    """Return raw power samples with elapsed seconds for comparison plotting."""

    rows = []
    for cycle, series_label in (
        (reference_cycle, reference_label),
        (candidate_cycle, candidate_label),
    ):
        if not cycle.samples:
            raise ValueError(f"cycle has no samples: {cycle.segment_id}")
        start_timestamp = cycle.samples[0].timestamp
        rows.extend(
            {
                "elapsed_seconds": (
                    sample.timestamp - start_timestamp
                ).total_seconds(),
                "power_kw": sample.power_kw,
                "series": series_label,
            }
            for sample in cycle.samples
        )
    return pd.DataFrame(rows, columns=["elapsed_seconds", "power_kw", "series"])


def format_indicator_percentage(value: float | None) -> str:
    """Format an optional reference-relative percentage for display."""

    if value is None:
        return "Not available (reference value is zero)"
    return f"{value:+.2f}%"


def recommended_follow_up_text(recommended_action: str) -> str:
    """Return bounded user-facing follow-up text for an event action code."""

    if recommended_action == "no_automated_action":
        return "Recommended follow-up: No automated action is proposed by this prototype."
    if recommended_action == "manual_review_required":
        return "Recommended follow-up: Manual inspection required before any action."
    raise ValueError(f"unsupported recommended_action: {recommended_action}")


EVENT_HISTORY_COLUMNS = [
    "Event timestamp",
    "Event ID",
    "Machine ID",
    "Status",
    "Recommended action",
    "Normalized DTW distance",
    "Threshold",
    "Duration deviation percentage",
    "Energy deviation percentage",
    "Peak-power deviation percentage",
]


def build_event_history_frame(events: Sequence[MonitoringEvent]) -> pd.DataFrame:
    """Build a compact display table for local demo event history."""

    rows = [
        {
            "Event timestamp": event.event_timestamp.isoformat(),
            "Event ID": event.event_id,
            "Machine ID": event.machine_id,
            "Status": event.status,
            "Recommended action": event.recommended_action,
            "Normalized DTW distance": event.normalized_dtw_distance,
            "Threshold": event.threshold,
            "Duration deviation percentage": format_indicator_percentage(
                event.indicators.duration_deviation_pct
            ),
            "Energy deviation percentage": format_indicator_percentage(
                event.indicators.energy_deviation_pct
            ),
            "Peak-power deviation percentage": format_indicator_percentage(
                event.indicators.peak_power_deviation_pct
            ),
        }
        for event in events
    ]
    return pd.DataFrame(rows, columns=EVENT_HISTORY_COLUMNS)


def _display_state_for_segment_type(segment_type: str) -> DemoOperatingState:
    if segment_type == "idle":
        return "Idle"
    if segment_type in ("normal_cycle", "abnormal_cycle"):
        return "Processing"
    raise ValueError(f"unsupported synthetic segment type: {segment_type}")
