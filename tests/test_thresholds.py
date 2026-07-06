import math
from datetime import UTC, datetime, timedelta

import pytest

from volt_vision.monitoring.calibration import calibrate_reference_template
from volt_vision.monitoring.cycles import select_calibration_cycles
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.models import PowerSample
from volt_vision.monitoring.thresholds import derive_dtw_threshold
from volt_vision.monitoring.cycles import SelectedCycle


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def make_cycle(segment_id: str, values: list[float]) -> SelectedCycle:
    return SelectedCycle(
        segment_id=segment_id,
        segment_type="normal_cycle",
        samples=tuple(
            PowerSample(
                timestamp=START + timedelta(seconds=index),
                machine_id="CNC_TEST",
                power_kw=value,
            )
            for index, value in enumerate(values)
        ),
    )


def make_demo_calibration():
    timeline = generate_demo_timeline(start_timestamp=START)
    return calibrate_reference_template(select_calibration_cycles(timeline))


def test_demo_calibration_produces_valid_threshold_result() -> None:
    calibration = make_demo_calibration()

    result = derive_dtw_threshold(calibration)

    assert result.reference_segment_id == calibration.reference_cycle.segment_id
    assert len(result.calibration_reference_distances) == len(
        calibration.calibration_cycles
    )
    assert result.threshold >= result.max_normalized_distance


def test_reference_distances_preserve_calibration_cycle_order() -> None:
    calibration = make_demo_calibration()

    result = derive_dtw_threshold(calibration)

    assert [
        distance.segment_id for distance in result.calibration_reference_distances
    ] == [cycle.segment_id for cycle in calibration.calibration_cycles]


def test_reference_segment_is_present_with_zero_distance() -> None:
    calibration = make_demo_calibration()

    result = derive_dtw_threshold(calibration)

    reference_distances = [
        distance
        for distance in result.calibration_reference_distances
        if distance.segment_id == calibration.reference_cycle.segment_id
    ]
    assert len(reference_distances) == 1
    assert reference_distances[0].normalized_distance == 0


def test_all_threshold_result_values_are_finite_and_non_negative() -> None:
    result = derive_dtw_threshold(make_demo_calibration())
    numeric_values = [
        result.max_normalized_distance,
        result.absolute_margin,
        result.relative_margin,
        result.applied_margin,
        result.threshold,
        *[
            distance.normalized_distance
            for distance in result.calibration_reference_distances
        ],
    ]

    assert all(math.isfinite(value) for value in numeric_values)
    assert all(value >= 0 for value in numeric_values)


def test_max_normalized_distance_equals_max_stored_reference_distance() -> None:
    result = derive_dtw_threshold(make_demo_calibration())

    assert result.max_normalized_distance == max(
        distance.normalized_distance
        for distance in result.calibration_reference_distances
    )


def test_threshold_equals_max_distance_plus_applied_margin() -> None:
    result = derive_dtw_threshold(make_demo_calibration())

    assert result.threshold == pytest.approx(
        result.max_normalized_distance + result.applied_margin
    )


def test_applied_margin_uses_larger_absolute_or_relative_margin() -> None:
    result = derive_dtw_threshold(
        make_demo_calibration(),
        absolute_margin=0.03,
        relative_margin=0.25,
    )

    assert result.applied_margin == pytest.approx(
        max(0.03, 0.25 * result.max_normalized_distance)
    )


def test_absolute_margin_dominates_when_distances_are_zero() -> None:
    calibration = calibrate_reference_template(
        (
            make_cycle("cycle_1", [2.0, 2.0]),
            make_cycle("cycle_2", [2.0, 2.0]),
            make_cycle("cycle_3", [2.0, 2.0]),
        )
    )

    result = derive_dtw_threshold(
        calibration,
        absolute_margin=0.02,
        relative_margin=0.20,
    )

    assert result.max_normalized_distance == 0
    assert result.applied_margin == pytest.approx(0.02)
    assert result.threshold == pytest.approx(0.02)


def test_relative_margin_dominates_when_distances_are_large() -> None:
    calibration = calibrate_reference_template(
        (
            make_cycle("cycle_1", [0.0, 0.0]),
            make_cycle("cycle_2", [10.0, 10.0]),
            make_cycle("cycle_3", [20.0, 20.0]),
        )
    )

    result = derive_dtw_threshold(
        calibration,
        absolute_margin=0.02,
        relative_margin=0.20,
    )

    assert result.max_normalized_distance == pytest.approx(10.0)
    assert result.applied_margin == pytest.approx(2.0)
    assert result.threshold == pytest.approx(12.0)


def test_negative_absolute_margin_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="absolute_margin must be non-negative"):
        derive_dtw_threshold(make_demo_calibration(), absolute_margin=-0.01)


def test_negative_relative_margin_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="relative_margin must be non-negative"):
        derive_dtw_threshold(make_demo_calibration(), relative_margin=-0.01)


@pytest.mark.parametrize("bad_margin", [math.nan, math.inf, -math.inf])
def test_non_finite_absolute_margin_raises_clear_error(bad_margin: float) -> None:
    with pytest.raises(ValueError, match="absolute_margin must be finite"):
        derive_dtw_threshold(make_demo_calibration(), absolute_margin=bad_margin)


@pytest.mark.parametrize("bad_margin", [math.nan, math.inf, -math.inf])
def test_non_finite_relative_margin_raises_clear_error(bad_margin: float) -> None:
    with pytest.raises(ValueError, match="relative_margin must be finite"):
        derive_dtw_threshold(make_demo_calibration(), relative_margin=bad_margin)


def test_threshold_uses_only_supplied_calibration_cycles() -> None:
    calibration = make_demo_calibration()

    result = derive_dtw_threshold(calibration)

    selected_ids = {
        distance.segment_id for distance in result.calibration_reference_distances
    }
    assert selected_ids == {
        cycle.segment_id for cycle in calibration.calibration_cycles
    }
    assert all("abnormal" not in segment_id for segment_id in selected_ids)
