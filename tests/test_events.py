from dataclasses import replace
from datetime import UTC, datetime

import pytest

from volt_vision.monitoring.calibration import calibrate_reference_template
from volt_vision.monitoring.cycles import (
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.evaluation import evaluate_cycle_against_threshold
from volt_vision.monitoring.events import build_monitoring_event
from volt_vision.monitoring.indicators import calculate_reference_relative_indicators
from volt_vision.monitoring.metrics import compute_cycle_metrics
from volt_vision.monitoring.models import CycleMetrics
from volt_vision.monitoring.thresholds import derive_dtw_threshold


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def make_demo_pipeline(*, machine_id: str = "CNC_01"):
    timeline = generate_demo_timeline(start_timestamp=START, machine_id=machine_id)
    calibration_cycles = select_calibration_cycles(timeline)
    calibration = calibrate_reference_template(calibration_cycles)
    threshold_result = derive_dtw_threshold(calibration)
    abnormal_cycle = select_abnormal_evaluation_cycle(timeline)
    return calibration_cycles, calibration, threshold_result, abnormal_cycle


def test_end_to_end_normal_event() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    candidate = calibration_cycles[0]

    event = build_monitoring_event(candidate, calibration, threshold_result)

    assert event.status == "within_normal_band"
    assert event.event_type == "cycle_screening"
    assert event.recommended_action == "no_automated_action"
    assert event.event_timestamp == candidate.samples[-1].timestamp
    assert event.machine_id == candidate.samples[0].machine_id
    assert event.candidate_segment_id == candidate.segment_id
    assert event.reference_segment_id == calibration.reference_cycle.segment_id
    assert event.indicators.reference_cycle_id == calibration.reference_cycle.segment_id
    assert event.indicators.candidate_cycle_id == candidate.segment_id
    assert isinstance(event.indicators.duration_deviation_pct, float)
    assert isinstance(event.indicators.energy_deviation_pct, float)
    assert isinstance(event.indicators.peak_power_deviation_pct, float)


def test_reference_cycle_event_has_zero_reference_relative_indicators() -> None:
    _, calibration, threshold_result, _ = make_demo_pipeline()
    reference_cycle = calibration.reference_cycle

    event = build_monitoring_event(reference_cycle, calibration, threshold_result)

    assert event.status == "within_normal_band"
    assert event.recommended_action == "no_automated_action"
    assert event.indicators.duration_deviation_pct == pytest.approx(0.0)
    assert event.indicators.energy_deviation_pct == pytest.approx(0.0)
    assert event.indicators.peak_power_deviation_pct == pytest.approx(0.0)


def test_end_to_end_suspected_deviation_event_uses_cautious_evidence() -> None:
    _, calibration, threshold_result, abnormal_cycle = make_demo_pipeline()

    event = build_monitoring_event(abnormal_cycle, calibration, threshold_result)

    assert event.event_type == "cycle_screening"
    assert event.status == "suspected_deviation"
    assert event.recommended_action == "manual_review_required"
    assert event.normalized_dtw_distance > event.threshold
    assert event.indicators.duration_deviation_pct is not None
    assert event.indicators.energy_deviation_pct is not None
    assert event.indicators.peak_power_deviation_pct is not None
    assert event.indicators.duration_deviation_pct > 0
    assert event.indicators.energy_deviation_pct > 0
    assert event.indicators.peak_power_deviation_pct > 0
    assert "Normalized DTW distance" in event.evidence
    assert "calibrated threshold" in event.evidence
    forbidden_terms = ["fault", "failure", "tool", "wear", "cause", "diagnosis", "maintenance"]
    assert all(term not in event.evidence.lower() for term in forbidden_terms)


def test_event_metrics_match_existing_cycle_metrics() -> None:
    _, calibration, threshold_result, abnormal_cycle = make_demo_pipeline()

    event = build_monitoring_event(abnormal_cycle, calibration, threshold_result)
    expected_metrics = compute_cycle_metrics(
        abnormal_cycle.samples,
        cycle_id=abnormal_cycle.segment_id,
    )

    assert event.metrics == expected_metrics
    assert event.metrics.duration_seconds == expected_metrics.duration_seconds
    assert event.metrics.energy_kwh == expected_metrics.energy_kwh
    assert event.metrics.average_power_kw == expected_metrics.average_power_kw
    assert event.metrics.peak_power_kw == expected_metrics.peak_power_kw
    assert event.event_timestamp == event.metrics.end_timestamp


def test_event_builder_is_deterministic() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    candidate = calibration_cycles[1]

    first = build_monitoring_event(candidate, calibration, threshold_result)
    second = build_monitoring_event(candidate, calibration, threshold_result)

    assert first.event_id == second.event_id
    assert first == second
    assert first.model_dump() == second.model_dump()


def test_event_status_matches_cycle_evaluation() -> None:
    calibration_cycles, calibration, threshold_result, abnormal_cycle = make_demo_pipeline()

    for candidate in (*calibration_cycles, abnormal_cycle):
        event = build_monitoring_event(candidate, calibration, threshold_result)
        evaluation = evaluate_cycle_against_threshold(
            candidate,
            calibration,
            threshold_result,
        )
        assert event.status == evaluation.status
        assert event.evidence == evaluation.evidence
        assert event.normalized_dtw_distance == evaluation.normalized_dtw_distance
        assert event.threshold == evaluation.threshold
        expected_action = (
            "no_automated_action"
            if evaluation.status == "within_normal_band"
            else "manual_review_required"
        )
        assert event.recommended_action == expected_action


def test_event_payload_excludes_labels_and_root_cause_fields() -> None:
    _, calibration, threshold_result, abnormal_cycle = make_demo_pipeline()

    event = build_monitoring_event(abnormal_cycle, calibration, threshold_result)
    payload = event.model_dump()

    forbidden_keys = {
        "expected_label",
        "segment_type",
        "root_cause",
        "fault",
        "failure",
        "tool_wear",
        "maintenance_action",
    }
    assert forbidden_keys.isdisjoint(payload)
    assert forbidden_keys.isdisjoint(payload["metrics"])
    assert forbidden_keys.isdisjoint(payload["indicators"])


def test_recommended_action_is_not_indicator_dependent() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    candidate = calibration_cycles[0]

    event = build_monitoring_event(candidate, calibration, threshold_result)

    assert event.status == "within_normal_band"
    assert event.indicators.duration_deviation_pct != 0
    assert event.indicators.energy_deviation_pct != 0
    assert event.indicators.peak_power_deviation_pct != 0
    assert event.recommended_action == "no_automated_action"


def test_zero_reference_baseline_indicator_is_event_compatible() -> None:
    reference_metrics = CycleMetrics(
        cycle_id="reference",
        machine_id="CNC_TEST",
        start_timestamp=START,
        end_timestamp=START,
        duration_seconds=0,
        energy_kwh=10,
        average_power_kw=2,
        peak_power_kw=5,
        sample_count=1,
    )
    candidate_metrics = CycleMetrics(
        cycle_id="candidate",
        machine_id="CNC_TEST",
        start_timestamp=START,
        end_timestamp=START,
        duration_seconds=5,
        energy_kwh=12,
        average_power_kw=2.4,
        peak_power_kw=6,
        sample_count=2,
    )

    indicators = calculate_reference_relative_indicators(
        reference_metrics,
        candidate_metrics,
    )

    assert indicators.duration_deviation_pct is None
    assert indicators.energy_deviation_pct == pytest.approx(20.0)
    assert indicators.peak_power_deviation_pct == pytest.approx(20.0)


def test_custom_machine_id_event_uses_custom_machine_id() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline(
        machine_id="CNC_CUSTOM",
    )

    event = build_monitoring_event(
        calibration_cycles[0],
        calibration,
        threshold_result,
    )

    assert event.machine_id == "CNC_CUSTOM"
    assert event.event_id.startswith("CNC_CUSTOM:")


def test_inconsistent_calibration_and_threshold_raise_from_evaluation() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    mismatched_threshold = replace(
        threshold_result,
        reference_segment_id="not_the_reference",
    )

    with pytest.raises(ValueError, match="reference_segment_id must match"):
        build_monitoring_event(
            calibration_cycles[0],
            calibration,
            mismatched_threshold,
        )


def test_monitoring_event_serializes_with_pydantic() -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()

    event = build_monitoring_event(
        calibration_cycles[0],
        calibration,
        threshold_result,
    )

    dumped = event.model_dump()
    dumped_json_mode = event.model_dump(mode="json")
    json_payload = event.model_dump_json()
    assert dumped["event_id"] == event.event_id
    assert dumped_json_mode["event_type"] == "cycle_screening"
    assert dumped_json_mode["recommended_action"] == "no_automated_action"
    assert dumped_json_mode["indicators"]["reference_cycle_id"] == (
        calibration.reference_cycle.segment_id
    )
    assert "expected_label" not in dumped_json_mode
    assert "segment_type" not in dumped_json_mode
    assert event.candidate_segment_id in json_payload
