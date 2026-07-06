import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from volt_vision.monitoring.calibration import calibrate_reference_template
from volt_vision.monitoring.cycles import (
    SelectedCycle,
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.evaluation import evaluate_cycle_against_threshold
from volt_vision.monitoring.models import PowerSample
from volt_vision.monitoring.thresholds import derive_dtw_threshold


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def make_cycle(
    segment_id: str,
    values: list[float],
    *,
    machine_id: str = "CNC_TEST",
    segment_type: str = "normal_cycle",
) -> SelectedCycle:
    return SelectedCycle(
        segment_id=segment_id,
        segment_type=segment_type,  # type: ignore[arg-type]
        samples=tuple(
            PowerSample(
                timestamp=START + timedelta(seconds=index),
                machine_id=machine_id,
                power_kw=value,
            )
            for index, value in enumerate(values)
        ),
    )


def make_demo_pipeline():
    timeline = generate_demo_timeline(start_timestamp=START)
    calibration_cycles = select_calibration_cycles(timeline)
    calibration = calibrate_reference_template(calibration_cycles)
    threshold_result = derive_dtw_threshold(calibration)
    abnormal_cycle = select_abnormal_evaluation_cycle(timeline)
    return calibration_cycles, calibration, threshold_result, abnormal_cycle


def test_calibration_normal_cycles_evaluate_within_normal_band() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()

    evaluations = [
        evaluate_cycle_against_threshold(cycle, calibration, threshold_result)
        for cycle in calibration_cycles
    ]

    assert all(evaluation.status == "within_normal_band" for evaluation in evaluations)
    assert all(
        evaluation.within_calibrated_normal_band for evaluation in evaluations
    )
    assert all(
        evaluation.normalized_dtw_distance <= evaluation.threshold
        for evaluation in evaluations
    )


def test_abnormal_demo_cycle_evaluates_as_suspected_deviation() -> None:
    _, calibration, threshold_result, abnormal_cycle = make_demo_pipeline()

    evaluation = evaluate_cycle_against_threshold(
        abnormal_cycle,
        calibration,
        threshold_result,
    )

    assert evaluation.status == "suspected_deviation"
    assert evaluation.within_calibrated_normal_band is False
    assert evaluation.normalized_dtw_distance > evaluation.threshold


def test_evaluation_result_contains_expected_identifiers() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()

    evaluation = evaluate_cycle_against_threshold(
        calibration_cycles[0],
        calibration,
        threshold_result,
    )

    assert evaluation.segment_id == calibration_cycles[0].segment_id
    assert evaluation.machine_id == calibration_cycles[0].samples[0].machine_id
    assert evaluation.reference_segment_id == calibration.reference_cycle.segment_id


def test_reference_cycle_evaluates_to_zero_and_within_band() -> None:
    _, calibration, threshold_result, _ = make_demo_pipeline()

    evaluation = evaluate_cycle_against_threshold(
        calibration.reference_cycle,
        calibration,
        threshold_result,
    )

    assert evaluation.normalized_dtw_distance == 0
    assert evaluation.status == "within_normal_band"


def test_distance_equal_to_threshold_is_within_normal_band() -> None:
    cycles = (
        make_cycle("cycle_1", [2.0, 2.0]),
        make_cycle("cycle_2", [2.0, 2.0]),
        make_cycle("cycle_3", [2.0, 2.0]),
    )
    calibration = calibrate_reference_template(cycles)
    threshold_result = derive_dtw_threshold(
        calibration,
        absolute_margin=0.0,
        relative_margin=0.0,
    )

    evaluation = evaluate_cycle_against_threshold(
        cycles[0],
        calibration,
        threshold_result,
    )

    assert evaluation.normalized_dtw_distance == evaluation.threshold == 0
    assert evaluation.status == "within_normal_band"


def test_mismatched_threshold_reference_raises_clear_error() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    mismatched_threshold = replace(
        threshold_result,
        reference_segment_id="not_the_reference",
    )

    with pytest.raises(ValueError, match="reference_segment_id must match"):
        evaluate_cycle_against_threshold(
            calibration_cycles[0],
            calibration,
            mismatched_threshold,
        )


def test_empty_candidate_samples_raise_clear_error() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    empty_cycle = SelectedCycle(
        segment_id="empty",
        segment_type=calibration_cycles[0].segment_type,
        samples=(),
    )

    with pytest.raises(ValueError, match="no samples"):
        evaluate_cycle_against_threshold(empty_cycle, calibration, threshold_result)


def test_candidate_samples_from_multiple_machines_raise_clear_error() -> None:
    _, calibration, threshold_result, _ = make_demo_pipeline()
    mixed_machine_cycle = SelectedCycle(
        segment_id="mixed_machine",
        segment_type="normal_cycle",
        samples=(
            PowerSample(timestamp=START, machine_id="CNC_01", power_kw=1.0),
            PowerSample(
                timestamp=START + timedelta(seconds=1),
                machine_id="CNC_02",
                power_kw=1.1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="exactly one machine_id"):
        evaluate_cycle_against_threshold(
            mixed_machine_cycle,
            calibration,
            threshold_result,
        )


def test_candidate_from_different_single_machine_raises_clear_error() -> None:
    _, calibration, threshold_result, _ = make_demo_pipeline()
    different_machine_cycle = make_cycle(
        "different_machine",
        [1.0, 1.1, 1.2],
        machine_id="CNC_OTHER",
    )

    with pytest.raises(ValueError, match="must match calibration reference"):
        evaluate_cycle_against_threshold(
            different_machine_cycle,
            calibration,
            threshold_result,
        )


@pytest.mark.parametrize("bad_threshold", [math.nan, math.inf, -math.inf])
def test_non_finite_threshold_raises_clear_error(bad_threshold: float) -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    invalid_threshold = replace(threshold_result, threshold=bad_threshold)

    with pytest.raises(ValueError, match="threshold must be finite"):
        evaluate_cycle_against_threshold(
            calibration_cycles[0],
            calibration,
            invalid_threshold,
        )


def test_negative_threshold_raises_clear_error() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    invalid_threshold = replace(threshold_result, threshold=-0.001)

    with pytest.raises(ValueError, match="threshold must be non-negative"):
        evaluate_cycle_against_threshold(
            calibration_cycles[0],
            calibration,
            invalid_threshold,
        )


def test_evaluation_does_not_use_segment_type_or_root_cause_wording() -> None:
    _, calibration, threshold_result, abnormal_cycle = make_demo_pipeline()

    evaluation = evaluate_cycle_against_threshold(
        abnormal_cycle,
        calibration,
        threshold_result,
    )

    assert evaluation.status == "suspected_deviation"
    forbidden_terms = ["fault", "failure", "tool wear", "root cause"]
    assert all(term not in evaluation.evidence.lower() for term in forbidden_terms)
    assert "abnormal" not in evaluation.evidence.lower()
