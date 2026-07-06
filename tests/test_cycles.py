from datetime import UTC, datetime

import pytest

from volt_vision.monitoring.cycles import (
    extract_segment_samples,
    get_segment_by_id,
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.metrics import compute_cycle_metrics


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def test_get_segment_by_id_returns_expected_metadata() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)

    segment = get_segment_by_id(timeline, "normal_cycle_1")

    assert segment.segment_id == "normal_cycle_1"
    assert segment.segment_type == "normal_cycle"


def test_get_segment_by_id_unknown_segment_raises_clear_error() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)

    with pytest.raises(ValueError, match="unknown demo segment_id"):
        get_segment_by_id(timeline, "missing")


def test_extract_normal_cycle_samples_match_metadata_boundaries() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    segment = get_segment_by_id(timeline, "normal_cycle_1")

    samples = extract_segment_samples(timeline, segment.segment_id)

    assert samples
    assert samples[0].timestamp == segment.start_timestamp
    assert samples[-1].timestamp == segment.end_timestamp


def test_extract_normal_cycle_samples_exclude_neighboring_idle_samples() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    segment = get_segment_by_id(timeline, "normal_cycle_1")

    samples = extract_segment_samples(timeline, segment.segment_id)
    segment_indices = [timeline.samples.index(sample) for sample in samples]

    assert segment_indices == list(range(segment_indices[0], segment_indices[-1] + 1))
    assert timeline.samples[segment_indices[0] - 1].timestamp < segment.start_timestamp
    assert timeline.samples[segment_indices[-1] + 1].timestamp > segment.end_timestamp


def test_select_calibration_cycles_returns_three_normal_cycles_in_order() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)

    cycles = select_calibration_cycles(timeline)

    assert len(cycles) == 3
    assert [cycle.segment_id for cycle in cycles] == [
        "normal_cycle_1",
        "normal_cycle_2",
        "normal_cycle_3",
    ]
    assert all(cycle.segment_type == "normal_cycle" for cycle in cycles)
    assert all(cycle.samples for cycle in cycles)
    assert [cycle.samples[0].timestamp for cycle in cycles] == sorted(
        cycle.samples[0].timestamp for cycle in cycles
    )


def test_calibration_samples_use_one_machine_and_increasing_timestamps() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)

    cycles = select_calibration_cycles(timeline)

    for cycle in cycles:
        machine_ids = {sample.machine_id for sample in cycle.samples}
        timestamps = [sample.timestamp for sample in cycle.samples]
        assert len(machine_ids) == 1
        assert all(
            later > earlier for earlier, later in zip(timestamps, timestamps[1:])
        )


def test_select_abnormal_evaluation_cycle_returns_non_empty_abnormal_cycle() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)

    cycle = select_abnormal_evaluation_cycle(timeline)

    assert cycle.segment_id == "abnormal_cycle_1"
    assert cycle.segment_type == "abnormal_cycle"
    assert cycle.samples


def test_metrics_accept_selected_calibration_and_abnormal_cycles() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    calibration_cycle = select_calibration_cycles(timeline)[0]
    abnormal_cycle = select_abnormal_evaluation_cycle(timeline)

    calibration_metrics = compute_cycle_metrics(
        calibration_cycle.samples,
        cycle_id=calibration_cycle.segment_id,
    )
    abnormal_metrics = compute_cycle_metrics(
        abnormal_cycle.samples,
        cycle_id=abnormal_cycle.segment_id,
    )

    assert calibration_metrics.sample_count == len(calibration_cycle.samples)
    assert abnormal_metrics.sample_count == len(abnormal_cycle.samples)


def test_select_calibration_cycles_rejects_non_positive_count() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)

    with pytest.raises(ValueError, match="count must be positive"):
        select_calibration_cycles(timeline, count=0)


def test_select_calibration_cycles_rejects_count_larger_than_available() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)

    with pytest.raises(ValueError, match="only 3 normal cycles are available"):
        select_calibration_cycles(timeline, count=4)


def test_custom_machine_id_extracts_and_selects_correctly() -> None:
    timeline = generate_demo_timeline(
        start_timestamp=START,
        machine_id="CNC_CUSTOM",
    )

    calibration_cycles = select_calibration_cycles(timeline)
    abnormal_cycle = select_abnormal_evaluation_cycle(timeline)

    all_selected_samples = [
        sample
        for cycle in (*calibration_cycles, abnormal_cycle)
        for sample in cycle.samples
    ]
    assert all(sample.machine_id == "CNC_CUSTOM" for sample in all_selected_samples)
