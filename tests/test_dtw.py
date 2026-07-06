import math

import pytest

from volt_vision.monitoring.cycles import select_calibration_cycles
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.dtw import compute_dtw, compute_power_sample_dtw


def test_identical_numeric_sequences_return_zero_distance() -> None:
    result = compute_dtw([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert result.raw_distance == 0
    assert result.normalized_distance == 0


def test_simple_known_case_has_expected_distance_and_path_length() -> None:
    result = compute_dtw([1.0, 2.0, 3.0], [1.0, 3.0])

    assert result.raw_distance == pytest.approx(1.0)
    assert result.warping_path_length == 3
    assert result.normalized_distance == pytest.approx(1.0 / 3)


def test_dtw_is_symmetric_for_non_trivial_sequences() -> None:
    first = compute_dtw([1.0, 3.0, 2.0, 4.0], [1.0, 2.0, 4.0])
    second = compute_dtw([1.0, 2.0, 4.0], [1.0, 3.0, 2.0, 4.0])

    assert first.raw_distance == pytest.approx(second.raw_distance)
    assert first.normalized_distance == pytest.approx(second.normalized_distance)


def test_different_sequence_lengths_are_supported() -> None:
    result = compute_dtw([0.0, 1.0, 2.0, 3.0], [0.0, 2.0])

    assert result.warping_path_length >= 4
    assert result.raw_distance >= 0


def test_one_element_sequences_are_supported() -> None:
    result = compute_dtw([1.5], [4.0])

    assert result.raw_distance == pytest.approx(2.5)
    assert result.normalized_distance == pytest.approx(2.5)
    assert result.warping_path_length == 1


def test_empty_first_sequence_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="values_a must not be empty"):
        compute_dtw([], [1.0])


def test_empty_second_sequence_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="values_b must not be empty"):
        compute_dtw([1.0], [])


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nan_and_infinity_raise_clear_error(bad_value: float) -> None:
    with pytest.raises(ValueError, match="finite numeric values"):
        compute_dtw([1.0, bad_value], [1.0, 2.0])


def test_distances_are_finite_and_non_negative() -> None:
    result = compute_dtw([3.0, 1.0, 4.0], [2.0, 4.0])

    assert math.isfinite(result.raw_distance)
    assert math.isfinite(result.normalized_distance)
    assert result.raw_distance >= 0
    assert result.normalized_distance >= 0
    assert result.warping_path_length > 0


def test_inputs_are_not_mutated() -> None:
    values_a = [1.0, 2.0, 3.0]
    values_b = [1.0, 3.0]

    compute_dtw(values_a, values_b)

    assert values_a == [1.0, 2.0, 3.0]
    assert values_b == [1.0, 3.0]


def test_power_sample_dtw_works_on_extracted_demo_calibration_cycles() -> None:
    timeline = generate_demo_timeline()
    first_cycle, second_cycle, *_ = select_calibration_cycles(timeline)

    result = compute_power_sample_dtw(first_cycle.samples, second_cycle.samples)

    assert math.isfinite(result.raw_distance)
    assert math.isfinite(result.normalized_distance)
    assert result.raw_distance >= 0
    assert result.normalized_distance >= 0
    assert result.warping_path_length > 0


def test_power_sample_dtw_does_not_require_demo_metadata_or_labels() -> None:
    timeline = generate_demo_timeline()
    first_cycle, second_cycle, *_ = select_calibration_cycles(timeline)

    result = compute_power_sample_dtw(first_cycle.samples, second_cycle.samples)

    assert result == compute_dtw(
        [sample.power_kw for sample in first_cycle.samples],
        [sample.power_kw for sample in second_cycle.samples],
    )
