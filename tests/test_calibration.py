import math
from datetime import UTC, datetime, timedelta

import pytest

from volt_vision.monitoring.calibration import calibrate_reference_template
from volt_vision.monitoring.cycles import (
    SelectedCycle,
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.models import PowerSample


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def make_cycle(
    segment_id: str,
    values: list[float],
    *,
    machine_id: str = "CNC_TEST",
    segment_type: str = "normal_cycle",
) -> SelectedCycle:
    samples = tuple(
        PowerSample(
            timestamp=START + timedelta(seconds=index),
            machine_id=machine_id,
            power_kw=value,
        )
        for index, value in enumerate(values)
    )
    return SelectedCycle(
        segment_id=segment_id,
        segment_type=segment_type,  # type: ignore[arg-type]
        samples=samples,
    )


def test_demo_calibration_cycles_produce_valid_result() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    cycles = select_calibration_cycles(timeline)

    result = calibrate_reference_template(cycles)

    assert result.calibration_cycles == cycles
    assert result.reference_cycle in cycles
    assert len(result.pairwise_distances) == 3
    assert set(result.mean_normalized_distances) == {
        cycle.segment_id for cycle in cycles
    }


def test_pairwise_distance_count_is_three_for_three_cycles() -> None:
    result = calibrate_reference_template(
        select_calibration_cycles(generate_demo_timeline(start_timestamp=START))
    )

    assert len(result.pairwise_distances) == 3


def test_pairwise_distances_are_finite_non_negative_and_unique() -> None:
    result = calibrate_reference_template(
        select_calibration_cycles(generate_demo_timeline(start_timestamp=START))
    )

    pairs = {
        (distance.left_segment_id, distance.right_segment_id)
        for distance in result.pairwise_distances
    }
    assert len(pairs) == len(result.pairwise_distances)
    for distance in result.pairwise_distances:
        assert distance.left_segment_id != distance.right_segment_id
        assert math.isfinite(distance.normalized_distance)
        assert distance.normalized_distance >= 0


def test_mean_distances_are_finite_and_non_negative() -> None:
    result = calibrate_reference_template(
        select_calibration_cycles(generate_demo_timeline(start_timestamp=START))
    )

    for mean_distance in result.mean_normalized_distances.values():
        assert math.isfinite(mean_distance)
        assert mean_distance >= 0


def test_reference_cycle_has_lowest_mean_distance() -> None:
    result = calibrate_reference_template(
        select_calibration_cycles(generate_demo_timeline(start_timestamp=START))
    )

    reference_mean = result.mean_normalized_distances[result.reference_cycle.segment_id]
    assert reference_mean == min(result.mean_normalized_distances.values())


def test_calibration_preserves_supplied_cycle_order() -> None:
    cycles = select_calibration_cycles(generate_demo_timeline(start_timestamp=START))
    reordered_cycles = (cycles[2], cycles[0], cycles[1])

    result = calibrate_reference_template(reordered_cycles)

    assert result.calibration_cycles == reordered_cycles


def test_fewer_than_three_cycles_raises_clear_error() -> None:
    cycles = select_calibration_cycles(generate_demo_timeline(start_timestamp=START))

    with pytest.raises(ValueError, match="at least three"):
        calibrate_reference_template(cycles[:2])


def test_including_abnormal_cycle_raises_clear_error() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    cycles = select_calibration_cycles(timeline)
    abnormal_cycle = select_abnormal_evaluation_cycle(timeline)

    with pytest.raises(ValueError, match="segment_type == 'normal_cycle'"):
        calibrate_reference_template((cycles[0], cycles[1], abnormal_cycle))


def test_duplicate_segment_ids_raise_clear_error() -> None:
    first = make_cycle("cycle_1", [1.0, 2.0])
    duplicate = make_cycle("cycle_1", [1.1, 2.1])
    third = make_cycle("cycle_3", [1.2, 2.2])

    with pytest.raises(ValueError, match="unique segment IDs"):
        calibrate_reference_template((first, duplicate, third))


def test_cycles_from_different_machine_ids_raise_clear_error() -> None:
    cycles = (
        make_cycle("cycle_1", [1.0, 2.0], machine_id="CNC_A"),
        make_cycle("cycle_2", [1.1, 2.1], machine_id="CNC_A"),
        make_cycle("cycle_3", [1.2, 2.2], machine_id="CNC_B"),
    )

    with pytest.raises(ValueError, match="one machine_id"):
        calibrate_reference_template(cycles)


def test_empty_cycle_samples_raise_clear_error() -> None:
    cycles = (
        make_cycle("cycle_1", [1.0, 2.0]),
        make_cycle("cycle_2", [1.1, 2.1]),
        SelectedCycle(segment_id="cycle_3", segment_type="normal_cycle", samples=()),
    )

    with pytest.raises(ValueError, match="no samples"):
        calibrate_reference_template(cycles)


def test_tie_case_chooses_earliest_input_cycle() -> None:
    cycles = (
        make_cycle("cycle_1", [2.0, 2.0]),
        make_cycle("cycle_2", [2.0, 2.0]),
        make_cycle("cycle_3", [2.0, 2.0]),
    )

    result = calibrate_reference_template(cycles)

    assert result.mean_normalized_distances == {
        "cycle_1": 0.0,
        "cycle_2": 0.0,
        "cycle_3": 0.0,
    }
    assert result.reference_cycle == cycles[0]


def test_calibration_uses_only_supplied_normal_cycles() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    normal_cycles = select_calibration_cycles(timeline)

    result = calibrate_reference_template(normal_cycles)

    selected_ids = {cycle.segment_id for cycle in result.calibration_cycles}
    assert selected_ids == {"normal_cycle_1", "normal_cycle_2", "normal_cycle_3"}
    assert all("abnormal" not in segment_id for segment_id in selected_ids)
