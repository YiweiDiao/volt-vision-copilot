from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from volt_vision.monitoring.cycles import (
    SelectedCycle,
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.models import MonitoringEvent
from volt_vision.monitoring.event_log import read_monitoring_events
from volt_vision.monitoring.workflows import (
    CalibrationBundle,
    build_calibration_bundle_from_csv,
    build_monitoring_event_from_candidate_csv,
)


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def write_cycle_csv(path: Path, cycle: SelectedCycle) -> None:
    lines = ["timestamp,machine_id,power_kw"]
    lines.extend(
        f"{sample.timestamp.isoformat()},{sample.machine_id},{sample.power_kw}"
        for sample in cycle.samples
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_demo_cycles(*, machine_id: str = "CNC_01"):
    timeline = generate_demo_timeline(start_timestamp=START, machine_id=machine_id)
    calibration_cycles = select_calibration_cycles(timeline)
    abnormal_cycle = select_abnormal_evaluation_cycle(timeline)
    return calibration_cycles, abnormal_cycle


def write_calibration_csvs(
    tmp_path: Path,
    cycles: tuple[SelectedCycle, ...],
) -> list[Path]:
    paths = []
    for index, cycle in enumerate(cycles, start=1):
        csv_path = tmp_path / f"known_good_{index}.csv"
        write_cycle_csv(csv_path, cycle)
        paths.append(csv_path)
    return paths


def make_calibration_bundle(tmp_path: Path) -> CalibrationBundle:
    calibration_cycles, _ = make_demo_cycles()
    return build_calibration_bundle_from_csv(
        write_calibration_csvs(tmp_path, calibration_cycles)
    )


def test_build_calibration_bundle_from_three_valid_csvs(tmp_path: Path) -> None:
    demo_calibration_cycles, _ = make_demo_cycles()
    csv_paths = write_calibration_csvs(tmp_path, demo_calibration_cycles)

    bundle = build_calibration_bundle_from_csv(csv_paths)

    assert isinstance(bundle, CalibrationBundle)
    assert len(bundle.calibration_cycles) == 3
    assert [cycle.segment_id for cycle in bundle.calibration_cycles] == [
        "calibration_cycle_1",
        "calibration_cycle_2",
        "calibration_cycle_3",
    ]
    assert all(
        cycle.segment_type == "normal_cycle" for cycle in bundle.calibration_cycles
    )
    assert bundle.calibration_result.reference_cycle in bundle.calibration_cycles
    assert (
        bundle.threshold_result.reference_segment_id
        == bundle.calibration_result.reference_cycle.segment_id
    )
    assert math.isfinite(bundle.threshold_result.threshold)
    assert bundle.threshold_result.threshold >= 0


def test_custom_calibration_segment_ids_are_preserved(tmp_path: Path) -> None:
    demo_calibration_cycles, _ = make_demo_cycles()
    csv_paths = write_calibration_csvs(tmp_path, demo_calibration_cycles)

    bundle = build_calibration_bundle_from_csv(
        csv_paths,
        calibration_segment_ids=["known_good_a", "known_good_b", "known_good_c"],
    )

    assert [cycle.segment_id for cycle in bundle.calibration_cycles] == [
        "known_good_a",
        "known_good_b",
        "known_good_c",
    ]
    assert [
        cycle.segment_id for cycle in bundle.calibration_result.calibration_cycles
    ] == ["known_good_a", "known_good_b", "known_good_c"]


def test_custom_margins_are_passed_to_threshold_derivation(tmp_path: Path) -> None:
    demo_calibration_cycles, _ = make_demo_cycles()
    csv_paths = write_calibration_csvs(tmp_path, demo_calibration_cycles)

    bundle = build_calibration_bundle_from_csv(
        csv_paths,
        absolute_margin=0.05,
        relative_margin=0.30,
    )

    assert bundle.threshold_result.absolute_margin == pytest.approx(0.05)
    assert bundle.threshold_result.relative_margin == pytest.approx(0.30)


def test_build_normal_monitoring_event_from_candidate_csv(tmp_path: Path) -> None:
    demo_calibration_cycles, _ = make_demo_cycles()
    bundle = build_calibration_bundle_from_csv(
        write_calibration_csvs(tmp_path, demo_calibration_cycles)
    )
    candidate_path = tmp_path / "normal_candidate.csv"
    write_cycle_csv(candidate_path, demo_calibration_cycles[0])

    event = build_monitoring_event_from_candidate_csv(
        candidate_path,
        bundle,
        candidate_segment_id="normal_upload",
    )

    assert isinstance(event, MonitoringEvent)
    assert event.status == "within_normal_band"
    assert event.candidate_segment_id == "normal_upload"
    assert event.machine_id == demo_calibration_cycles[0].samples[0].machine_id
    assert event.model_dump()["event_id"] == event.event_id


def test_build_abnormal_monitoring_event_from_candidate_csv(tmp_path: Path) -> None:
    demo_calibration_cycles, abnormal_cycle = make_demo_cycles()
    bundle = build_calibration_bundle_from_csv(
        write_calibration_csvs(tmp_path, demo_calibration_cycles)
    )
    candidate_path = tmp_path / "abnormal_candidate.csv"
    write_cycle_csv(candidate_path, abnormal_cycle)

    event = build_monitoring_event_from_candidate_csv(
        candidate_path,
        bundle,
        candidate_segment_id="abnormal_upload",
    )

    assert event.status == "suspected_deviation"
    assert event.normalized_dtw_distance > event.threshold
    forbidden_terms = [
        "fault",
        "failure",
        "tool wear",
        "root cause",
        "diagnosis",
        "maintenance",
    ]
    assert all(term not in event.evidence.lower() for term in forbidden_terms)


def test_monitoring_event_workflow_is_deterministic(tmp_path: Path) -> None:
    demo_calibration_cycles, _ = make_demo_cycles()
    bundle = build_calibration_bundle_from_csv(
        write_calibration_csvs(tmp_path, demo_calibration_cycles)
    )
    candidate_path = tmp_path / "repeat_candidate.csv"
    write_cycle_csv(candidate_path, demo_calibration_cycles[1])

    first = build_monitoring_event_from_candidate_csv(candidate_path, bundle)
    second = build_monitoring_event_from_candidate_csv(candidate_path, bundle)

    assert first.event_id == second.event_id
    assert first.status == second.status


def test_candidate_screening_does_not_automatically_persist_event(
    tmp_path: Path,
) -> None:
    demo_calibration_cycles, abnormal_cycle = make_demo_cycles()
    bundle = build_calibration_bundle_from_csv(
        write_calibration_csvs(tmp_path, demo_calibration_cycles)
    )
    candidate_path = tmp_path / "changed_candidate.csv"
    event_history_path = tmp_path / "event_history.jsonl"
    write_cycle_csv(candidate_path, abnormal_cycle)

    event = build_monitoring_event_from_candidate_csv(candidate_path, bundle)

    assert event.status == "suspected_deviation"
    assert read_monitoring_events(event_history_path) == ()
    assert not event_history_path.exists()


def test_fewer_than_three_calibration_csvs_raise_existing_validation(
    tmp_path: Path,
) -> None:
    demo_calibration_cycles, _ = make_demo_cycles()
    csv_paths = write_calibration_csvs(tmp_path, demo_calibration_cycles[:2])

    with pytest.raises(ValueError, match="at least three"):
        build_calibration_bundle_from_csv(csv_paths)


def test_mixed_calibration_machine_ids_raise_existing_validation(
    tmp_path: Path,
) -> None:
    first_cycles, _ = make_demo_cycles(machine_id="CNC_01")
    second_cycles, _ = make_demo_cycles(machine_id="CNC_OTHER")
    csv_paths = write_calibration_csvs(
        tmp_path,
        (first_cycles[0], first_cycles[1], second_cycles[2]),
    )

    with pytest.raises(ValueError, match="all calibration cycles must use one machine_id"):
        build_calibration_bundle_from_csv(csv_paths)


def test_invalid_margin_raises_existing_validation(tmp_path: Path) -> None:
    demo_calibration_cycles, _ = make_demo_cycles()
    csv_paths = write_calibration_csvs(tmp_path, demo_calibration_cycles)

    with pytest.raises(ValueError, match="absolute_margin must be non-negative"):
        build_calibration_bundle_from_csv(csv_paths, absolute_margin=-0.01)


def test_candidate_machine_mismatch_raises_existing_validation(tmp_path: Path) -> None:
    demo_calibration_cycles, _ = make_demo_cycles(machine_id="CNC_01")
    _, other_machine_abnormal = make_demo_cycles(machine_id="CNC_OTHER")
    bundle = build_calibration_bundle_from_csv(
        write_calibration_csvs(tmp_path, demo_calibration_cycles)
    )
    candidate_path = tmp_path / "other_machine_candidate.csv"
    write_cycle_csv(candidate_path, other_machine_abnormal)

    with pytest.raises(ValueError, match="must match calibration reference"):
        build_monitoring_event_from_candidate_csv(candidate_path, bundle)
