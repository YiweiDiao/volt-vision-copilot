from datetime import UTC, datetime

import pytest

from volt_vision.monitoring.demo_data import (
    DEFAULT_MACHINE_ID,
    DemoSegment,
    generate_demo_timeline,
)
from volt_vision.monitoring.metrics import compute_cycle_metrics
from volt_vision.monitoring.models import PowerSample


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def samples_for_segment(
    samples: list[PowerSample],
    segment: DemoSegment,
) -> list[PowerSample]:
    return [
        sample
        for sample in samples
        if segment.start_timestamp <= sample.timestamp <= segment.end_timestamp
    ]


def test_demo_timeline_is_deterministic_for_same_seed() -> None:
    first = generate_demo_timeline(seed=42, start_timestamp=START)
    second = generate_demo_timeline(seed=42, start_timestamp=START)

    assert first == second


def test_demo_timeline_changes_some_power_values_for_different_seed() -> None:
    first = generate_demo_timeline(seed=42, start_timestamp=START)
    second = generate_demo_timeline(seed=43, start_timestamp=START)

    first_powers = [sample.power_kw for sample in first.samples]
    second_powers = [sample.power_kw for sample in second.samples]
    assert first_powers != second_powers


def test_timestamps_are_strictly_increasing_unique_and_utc() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    timestamps = [sample.timestamp for sample in timeline.samples]

    assert timestamps == sorted(timestamps)
    assert len(timestamps) == len(set(timestamps))
    assert all(timestamp.tzinfo is UTC for timestamp in timestamps)
    assert all(
        later > earlier for earlier, later in zip(timestamps, timestamps[1:])
    )


def test_power_values_and_default_machine_id_are_valid() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)

    assert all(sample.machine_id == DEFAULT_MACHINE_ID for sample in timeline.samples)
    assert all(sample.power_kw >= 0 for sample in timeline.samples)


def test_segment_structure_and_metadata_boundaries() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    segment_types = [segment.segment_type for segment in timeline.segments]

    assert segment_types == [
        "idle",
        "normal_cycle",
        "idle",
        "normal_cycle",
        "idle",
        "normal_cycle",
        "idle",
        "abnormal_cycle",
        "idle",
    ]
    assert segment_types.count("normal_cycle") == 3
    assert segment_types.count("abnormal_cycle") == 1

    for segment in timeline.segments:
        segment_samples = samples_for_segment(timeline.samples, segment)
        assert segment_samples[0].timestamp == segment.start_timestamp
        assert segment_samples[-1].timestamp == segment.end_timestamp

    assert all(
        later.start_timestamp > earlier.end_timestamp
        for earlier, later in zip(timeline.segments, timeline.segments[1:])
    )


def test_normal_cycles_are_similar_but_not_identical() -> None:
    timeline = generate_demo_timeline(seed=42, start_timestamp=START)
    normal_segments = [
        segment for segment in timeline.segments if segment.segment_type == "normal_cycle"
    ]
    normal_cycles = [
        samples_for_segment(timeline.samples, segment) for segment in normal_segments
    ]

    power_shapes = [
        tuple(sample.power_kw for sample in normal_cycle)
        for normal_cycle in normal_cycles
    ]
    durations = [
        (segment.end_timestamp - segment.start_timestamp).total_seconds()
        for segment in normal_segments
    ]

    assert len(set(power_shapes)) == 3
    assert max(durations) - min(durations) <= 3
    assert all(50 <= duration <= 60 for duration in durations)


def test_abnormal_cycle_is_longer_higher_energy_and_has_spike() -> None:
    timeline = generate_demo_timeline(seed=42, start_timestamp=START)
    normal_segments = [
        segment for segment in timeline.segments if segment.segment_type == "normal_cycle"
    ]
    abnormal_segment = next(
        segment for segment in timeline.segments if segment.segment_type == "abnormal_cycle"
    )

    normal_metrics = [
        compute_cycle_metrics(
            samples_for_segment(timeline.samples, segment),
            cycle_id=segment.segment_id,
        )
        for segment in normal_segments
    ]
    abnormal_metrics = compute_cycle_metrics(
        samples_for_segment(timeline.samples, abnormal_segment),
        cycle_id=abnormal_segment.segment_id,
    )

    assert all(
        abnormal_metrics.duration_seconds > metrics.duration_seconds
        for metrics in normal_metrics
    )
    assert all(
        abnormal_metrics.energy_kwh > metrics.energy_kwh
        for metrics in normal_metrics
    )
    assert abnormal_metrics.peak_power_kw > max(
        metrics.peak_power_kw for metrics in normal_metrics
    ) + 1.0


def test_metrics_accept_generated_normal_and_abnormal_segments() -> None:
    timeline = generate_demo_timeline(seed=42, start_timestamp=START)
    normal_segment = next(
        segment for segment in timeline.segments if segment.segment_type == "normal_cycle"
    )
    abnormal_segment = next(
        segment for segment in timeline.segments if segment.segment_type == "abnormal_cycle"
    )

    normal_metrics = compute_cycle_metrics(
        samples_for_segment(timeline.samples, normal_segment),
        cycle_id=normal_segment.segment_id,
    )
    abnormal_metrics = compute_cycle_metrics(
        samples_for_segment(timeline.samples, abnormal_segment),
        cycle_id=abnormal_segment.segment_id,
    )

    assert normal_metrics.machine_id == DEFAULT_MACHINE_ID
    assert abnormal_metrics.machine_id == DEFAULT_MACHINE_ID


def test_generate_demo_timeline_requires_utc_start_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        generate_demo_timeline(start_timestamp=datetime(2026, 1, 1, 8, 0))
