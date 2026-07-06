from __future__ import annotations

import math
import importlib.util
from pathlib import Path

from volt_vision.monitoring.csv_ingestion import (
    load_calibration_cycles_from_csv,
    load_candidate_cycle_from_csv,
)
from volt_vision.monitoring.workflows import (
    build_calibration_bundle_from_csv,
    build_monitoring_event_from_candidate_csv,
)

EXPECTED_CSV_FILES = (
    "calibration_1.csv",
    "calibration_2.csv",
    "calibration_3.csv",
    "candidate_normal.csv",
    "candidate_changed.csv",
)
CALIBRATION_FILES = EXPECTED_CSV_FILES[:3]
EVIDENCE_FORBIDDEN_TERMS = (
    "fault",
    "failure",
    "tool wear",
    "root cause",
    "diagnosis",
    "maintenance",
)


def generate_demo_csv_bundle(output_directory: Path) -> tuple[Path, ...]:
    script_path = Path("scripts/generate_demo_csv_bundle.py")
    spec = importlib.util.spec_from_file_location(
        "generate_demo_csv_bundle",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_demo_csv_bundle(output_directory)


def test_generator_is_deterministic(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    generate_demo_csv_bundle(first_directory)
    generate_demo_csv_bundle(second_directory)

    for filename in EXPECTED_CSV_FILES:
        assert (first_directory / filename).read_bytes() == (
            second_directory / filename
        ).read_bytes()


def test_generator_writes_expected_csv_files(tmp_path: Path) -> None:
    generate_demo_csv_bundle(tmp_path)

    generated_csv_files = sorted(path.name for path in tmp_path.glob("*.csv"))

    assert generated_csv_files == sorted(EXPECTED_CSV_FILES)


def test_generated_files_load_through_strict_csv_ingestion(tmp_path: Path) -> None:
    generate_demo_csv_bundle(tmp_path)

    calibration_cycles = load_calibration_cycles_from_csv(
        [tmp_path / filename for filename in CALIBRATION_FILES]
    )
    normal_candidate = load_candidate_cycle_from_csv(
        tmp_path / "candidate_normal.csv"
    )
    changed_candidate = load_candidate_cycle_from_csv(
        tmp_path / "candidate_changed.csv"
    )

    assert len(calibration_cycles) == 3
    assert normal_candidate.segment_type == "candidate_cycle"
    assert changed_candidate.segment_type == "candidate_cycle"


def test_generated_bundle_integrates_with_calibration_workflow(
    tmp_path: Path,
) -> None:
    generate_demo_csv_bundle(tmp_path)

    bundle = build_calibration_bundle_from_csv(
        [tmp_path / filename for filename in CALIBRATION_FILES]
    )

    assert all(cycle.segment_type == "normal_cycle" for cycle in bundle.calibration_cycles)
    assert len({cycle.samples[0].machine_id for cycle in bundle.calibration_cycles}) == 1
    assert bundle.calibration_result.reference_cycle in bundle.calibration_cycles
    assert math.isfinite(bundle.threshold_result.threshold)
    assert bundle.threshold_result.threshold >= 0


def test_generated_normal_candidate_is_within_normal_band(tmp_path: Path) -> None:
    generate_demo_csv_bundle(tmp_path)
    bundle = build_calibration_bundle_from_csv(
        [tmp_path / filename for filename in CALIBRATION_FILES]
    )

    event = build_monitoring_event_from_candidate_csv(
        tmp_path / "candidate_normal.csv",
        bundle,
    )

    assert event.status == "within_normal_band"


def test_generated_changed_candidate_is_suspected_deviation(tmp_path: Path) -> None:
    generate_demo_csv_bundle(tmp_path)
    bundle = build_calibration_bundle_from_csv(
        [tmp_path / filename for filename in CALIBRATION_FILES]
    )

    event = build_monitoring_event_from_candidate_csv(
        tmp_path / "candidate_changed.csv",
        bundle,
    )

    assert event.status == "suspected_deviation"
    assert event.normalized_dtw_distance > event.threshold
    assert all(term not in event.evidence.lower() for term in EVIDENCE_FORBIDDEN_TERMS)


def test_static_committed_bundle_loads_and_matches_generated_outcomes(
    tmp_path: Path,
) -> None:
    bundle_directory = Path("examples/csv_demo")
    generated_directory = tmp_path / "generated"
    generate_demo_csv_bundle(generated_directory)

    for filename in EXPECTED_CSV_FILES:
        assert (bundle_directory / filename).exists()
        assert (bundle_directory / filename).read_bytes() == (
            generated_directory / filename
        ).read_bytes()

    calibration_paths = [bundle_directory / filename for filename in CALIBRATION_FILES]
    load_calibration_cycles_from_csv(calibration_paths)
    load_candidate_cycle_from_csv(bundle_directory / "candidate_normal.csv")
    load_candidate_cycle_from_csv(bundle_directory / "candidate_changed.csv")

    bundle = build_calibration_bundle_from_csv(calibration_paths)
    normal_event = build_monitoring_event_from_candidate_csv(
        bundle_directory / "candidate_normal.csv",
        bundle,
    )
    changed_event = build_monitoring_event_from_candidate_csv(
        bundle_directory / "candidate_changed.csv",
        bundle,
    )

    assert normal_event.status == "within_normal_band"
    assert changed_event.status == "suspected_deviation"
    assert changed_event.normalized_dtw_distance > changed_event.threshold
