"""Metadata-backed demo cycle extraction helpers.

These helpers use DemoTimeline segment metadata to prepare deterministic demo
calibration and evaluation inputs. They are not automatic power-based
segmentation for real machine data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from volt_vision.monitoring.demo_data import DemoSegment, DemoTimeline
from volt_vision.monitoring.models import PowerSample

CycleSegmentType = Literal["normal_cycle", "abnormal_cycle", "candidate_cycle"]


@dataclass(frozen=True)
class SelectedCycle:
    segment_id: str
    segment_type: CycleSegmentType
    samples: tuple[PowerSample, ...]


def get_segment_by_id(timeline: DemoTimeline, segment_id: str) -> DemoSegment:
    """Return one metadata segment by ID."""

    for segment in timeline.segments:
        if segment.segment_id == segment_id:
            return segment
    raise ValueError(f"unknown demo segment_id: {segment_id}")


def extract_segment_samples(
    timeline: DemoTimeline,
    segment_id: str,
) -> tuple[PowerSample, ...]:
    """Return samples whose timestamps fall within a metadata segment."""

    segment = get_segment_by_id(timeline, segment_id)
    samples = tuple(
        sample
        for sample in timeline.samples
        if segment.start_timestamp <= sample.timestamp <= segment.end_timestamp
    )
    _validate_extracted_samples(segment, samples)
    return samples


def select_calibration_cycles(
    timeline: DemoTimeline,
    *,
    count: int = 3,
) -> tuple[SelectedCycle, ...]:
    """Return normal demo cycles in chronological order for later calibration."""

    if count <= 0:
        raise ValueError("calibration cycle count must be positive")

    normal_segments = sorted(
        (
            segment
            for segment in timeline.segments
            if segment.segment_type == "normal_cycle"
        ),
        key=lambda segment: segment.start_timestamp,
    )
    if len(normal_segments) < count:
        raise ValueError(
            f"requested {count} calibration cycles, but only "
            f"{len(normal_segments)} normal cycles are available"
        )

    return tuple(_select_cycle(timeline, segment) for segment in normal_segments[:count])


def select_abnormal_evaluation_cycle(timeline: DemoTimeline) -> SelectedCycle:
    """Return the single abnormal demo cycle for later evaluation or replay."""

    abnormal_segments = [
        segment
        for segment in timeline.segments
        if segment.segment_type == "abnormal_cycle"
    ]
    if len(abnormal_segments) != 1:
        raise ValueError(
            "demo timeline must contain exactly one abnormal_cycle segment; "
            f"found {len(abnormal_segments)}"
        )
    return _select_cycle(timeline, abnormal_segments[0])


def _select_cycle(timeline: DemoTimeline, segment: DemoSegment) -> SelectedCycle:
    if segment.segment_type not in ("normal_cycle", "abnormal_cycle"):
        raise ValueError(f"segment is not a selectable cycle: {segment.segment_id}")
    return SelectedCycle(
        segment_id=segment.segment_id,
        segment_type=segment.segment_type,
        samples=extract_segment_samples(timeline, segment.segment_id),
    )


def _validate_extracted_samples(
    segment: DemoSegment,
    samples: tuple[PowerSample, ...],
) -> None:
    if not samples:
        raise ValueError(f"segment contains no samples: {segment.segment_id}")
    if samples[0].timestamp != segment.start_timestamp:
        raise ValueError(
            f"first sample timestamp does not match segment start: {segment.segment_id}"
        )
    if samples[-1].timestamp != segment.end_timestamp:
        raise ValueError(
            f"last sample timestamp does not match segment end: {segment.segment_id}"
        )

    machine_id = samples[0].machine_id
    previous_timestamp = samples[0].timestamp
    for sample in samples[1:]:
        if sample.machine_id != machine_id:
            raise ValueError(
                f"segment samples must belong to exactly one machine: "
                f"{segment.segment_id}"
            )
        if sample.timestamp <= previous_timestamp:
            raise ValueError(
                f"segment sample timestamps must be strictly increasing: "
                f"{segment.segment_id}"
            )
        previous_timestamp = sample.timestamp
