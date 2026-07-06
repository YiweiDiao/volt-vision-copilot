from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from volt_vision.monitoring.calibration import calibrate_reference_template
from volt_vision.monitoring.csv_ingestion import (
    load_calibration_cycles_from_csv,
    load_candidate_cycle_from_csv,
)
from volt_vision.monitoring.cycles import (
    SelectedCycle,
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.evaluation import evaluate_cycle_against_threshold
from volt_vision.monitoring.events import build_monitoring_event
from volt_vision.monitoring.metrics import compute_cycle_metrics
from volt_vision.monitoring.models import PowerSample
from volt_vision.monitoring.thresholds import derive_dtw_threshold


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str] | None = None) -> None:
    headers = headers or ["timestamp", "machine_id", "power_kw"]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(row.get(header, "") for header in headers))
    path.write_text("\n".join(lines), encoding="utf-8")


def csv_rows_from_cycle(cycle: SelectedCycle) -> list[dict[str, str]]:
    return [
        {
            "timestamp": sample.timestamp.isoformat(),
            "machine_id": sample.machine_id,
            "power_kw": str(sample.power_kw),
        }
        for sample in cycle.samples
    ]


def make_demo_pipeline():
    timeline = generate_demo_timeline(start_timestamp=START)
    calibration_cycles = select_calibration_cycles(timeline)
    calibration = calibrate_reference_template(calibration_cycles)
    threshold_result = derive_dtw_threshold(calibration)
    abnormal_cycle = select_abnormal_evaluation_cycle(timeline)
    return calibration_cycles, calibration, threshold_result, abnormal_cycle


def test_valid_csv_loads_candidate_cycle(tmp_path: Path) -> None:
    csv_path = tmp_path / "candidate.csv"
    write_csv(
        csv_path,
        [
            {"timestamp": "2026-01-01T08:00:00Z", "machine_id": "CNC_01", "power_kw": "1.0"},
            {"timestamp": "2026-01-01T08:00:01Z", "machine_id": "CNC_01", "power_kw": "1.2"},
        ],
        headers=["power_kw", "timestamp", "machine_id"],
    )

    cycle = load_candidate_cycle_from_csv(
        csv_path,
        candidate_segment_id="csv_candidate",
    )

    assert isinstance(cycle, SelectedCycle)
    assert cycle.segment_type == "candidate_cycle"
    assert cycle.segment_id == "csv_candidate"
    assert all(isinstance(sample, PowerSample) for sample in cycle.samples)
    assert [sample.power_kw for sample in cycle.samples] == [1.0, 1.2]
    assert all(sample.timestamp.tzinfo is UTC for sample in cycle.samples)
    assert [sample.timestamp.second for sample in cycle.samples] == [0, 1]


def test_offset_timestamps_are_normalized_and_order_checked_in_utc(tmp_path: Path) -> None:
    csv_path = tmp_path / "offset.csv"
    write_csv(
        csv_path,
        [
            {
                "timestamp": "2026-01-01T09:00:00+01:00",
                "machine_id": "CNC_01",
                "power_kw": "1.0",
            },
            {
                "timestamp": "2026-01-01T09:00:01+01:00",
                "machine_id": "CNC_01",
                "power_kw": "1.1",
            },
        ],
    )

    cycle = load_candidate_cycle_from_csv(csv_path)

    assert cycle.samples[0].timestamp == datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    assert cycle.samples[1].timestamp == datetime(2026, 1, 1, 8, 0, 1, tzinfo=UTC)


def test_missing_required_column_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    write_csv(csv_path, [], headers=["timestamp", "machine_id"])

    with pytest.raises(ValueError, match="schema requires exactly"):
        load_candidate_cycle_from_csv(csv_path)


def test_extra_unexpected_column_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "extra.csv"
    write_csv(csv_path, [], headers=["timestamp", "machine_id", "power_kw", "label"])

    with pytest.raises(ValueError, match="unexpected columns"):
        load_candidate_cycle_from_csv(csv_path)


def test_duplicate_headers_raise_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicate_headers.csv"
    csv_path.write_text(
        "timestamp,machine_id,power_kw,timestamp\n"
        "2026-01-01T08:00:00Z,CNC_01,1.0,2026-01-01T08:00:00Z",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate headers are not allowed"):
        load_candidate_cycle_from_csv(csv_path)


def test_empty_file_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV file is empty"):
        load_candidate_cycle_from_csv(csv_path)


def test_header_only_file_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "header_only.csv"
    write_csv(csv_path, [])

    with pytest.raises(ValueError, match="at least one data row"):
        load_candidate_cycle_from_csv(csv_path)


def test_blank_data_row_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "blank_row.csv"
    csv_path.write_text("timestamp,machine_id,power_kw\n,,", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV row 2: blank data row"):
        load_candidate_cycle_from_csv(csv_path)


def test_blank_physical_line_between_valid_rows_raises_clear_error(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "blank_physical_line.csv"
    csv_path.write_text(
        "timestamp,machine_id,power_kw\n"
        "2026-01-01T08:00:00Z,CNC_01,1.0\n"
        "\n"
        "2026-01-01T08:00:01Z,CNC_01,1.1",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="physical line 3: blank data row"):
        load_candidate_cycle_from_csv(csv_path)


def test_missing_required_value_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_value.csv"
    csv_path.write_text(
        "timestamp,machine_id,power_kw\n2026-01-01T08:00:00Z,CNC_01",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV row 2: missing value"):
        load_candidate_cycle_from_csv(csv_path)


def test_extra_unlabelled_value_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "too_many_values.csv"
    csv_path.write_text(
        "timestamp,machine_id,power_kw\n2026-01-01T08:00:00Z,CNC_01,1.0,extra",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV row 2: too many values"):
        load_candidate_cycle_from_csv(csv_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamp", "not-a-time", "timestamp must be ISO-8601"),
        ("timestamp", "2026-01-01T08:00:00", "timestamp must be timezone-aware"),
        ("machine_id", "   ", "machine_id must be non-empty"),
        ("power_kw", "not-a-number", "power_kw must be numeric"),
        ("power_kw", "-0.1", "power_kw must be non-negative"),
        ("power_kw", "nan", "power_kw must be finite"),
        ("power_kw", "inf", "power_kw must be finite"),
    ],
)
def test_row_validation_errors_include_row_number(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    csv_path = tmp_path / "bad_row.csv"
    row = {
        "timestamp": "2026-01-01T08:00:00Z",
        "machine_id": "CNC_01",
        "power_kw": "1.0",
    }
    row[field] = value
    write_csv(csv_path, [row])

    with pytest.raises(ValueError, match=f"CSV row 2: .*{message}"):
        load_candidate_cycle_from_csv(csv_path)


def test_mixed_machine_ids_raise_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed_machine.csv"
    write_csv(
        csv_path,
        [
            {"timestamp": "2026-01-01T08:00:00Z", "machine_id": "CNC_01", "power_kw": "1.0"},
            {"timestamp": "2026-01-01T08:00:01Z", "machine_id": "CNC_02", "power_kw": "1.1"},
        ],
    )

    with pytest.raises(ValueError, match="all rows must use one machine_id"):
        load_candidate_cycle_from_csv(csv_path)


def test_duplicate_timestamps_raise_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicate_time.csv"
    write_csv(
        csv_path,
        [
            {"timestamp": "2026-01-01T08:00:00Z", "machine_id": "CNC_01", "power_kw": "1.0"},
            {"timestamp": "2026-01-01T08:00:00+00:00", "machine_id": "CNC_01", "power_kw": "1.1"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate timestamp"):
        load_candidate_cycle_from_csv(csv_path)


def test_decreasing_timestamps_raise_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "decreasing.csv"
    write_csv(
        csv_path,
        [
            {"timestamp": "2026-01-01T08:00:01Z", "machine_id": "CNC_01", "power_kw": "1.0"},
            {"timestamp": "2026-01-01T08:00:00Z", "machine_id": "CNC_01", "power_kw": "1.1"},
        ],
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_candidate_cycle_from_csv(csv_path)


def test_blank_candidate_segment_id_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "candidate.csv"
    write_csv(
        csv_path,
        [{"timestamp": "2026-01-01T08:00:00Z", "machine_id": "CNC_01", "power_kw": "1.0"}],
    )

    with pytest.raises(ValueError, match="candidate_segment_id must be non-empty"):
        load_candidate_cycle_from_csv(csv_path, candidate_segment_id="   ")


def test_loaded_normal_demo_csv_evaluates_within_normal_band(tmp_path: Path) -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    csv_path = tmp_path / "normal.csv"
    write_csv(csv_path, csv_rows_from_cycle(calibration_cycles[0]))

    cycle = load_candidate_cycle_from_csv(csv_path, candidate_segment_id="csv_normal")
    evaluation = evaluate_cycle_against_threshold(cycle, calibration, threshold_result)

    assert evaluation.status == "within_normal_band"


def test_loaded_abnormal_demo_csv_evaluates_as_suspected_deviation(tmp_path: Path) -> None:
    _, calibration, threshold_result, abnormal_cycle = make_demo_pipeline()
    csv_path = tmp_path / "candidate.csv"
    write_csv(csv_path, csv_rows_from_cycle(abnormal_cycle))

    cycle = load_candidate_cycle_from_csv(csv_path, candidate_segment_id="csv_candidate")
    evaluation = evaluate_cycle_against_threshold(cycle, calibration, threshold_result)

    assert evaluation.status == "suspected_deviation"


def test_loaded_candidate_builds_monitoring_event(tmp_path: Path) -> None:
    calibration_cycles, calibration, threshold_result, _ = make_demo_pipeline()
    csv_path = tmp_path / "event_candidate.csv"
    write_csv(csv_path, csv_rows_from_cycle(calibration_cycles[1]))

    cycle = load_candidate_cycle_from_csv(csv_path, candidate_segment_id="csv_event")
    event = build_monitoring_event(cycle, calibration, threshold_result)
    expected_metrics = compute_cycle_metrics(cycle.samples, cycle_id=cycle.segment_id)

    assert event.machine_id == cycle.samples[0].machine_id
    assert event.candidate_segment_id == "csv_event"
    assert event.event_timestamp == cycle.samples[-1].timestamp
    assert event.metrics == expected_metrics
    assert event.model_dump_json()


def test_calibration_rejects_candidate_cycle_inputs(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_path = tmp_path / "candidate.csv"
    write_csv(csv_path, csv_rows_from_cycle(calibration_cycles[0]))
    candidate = load_candidate_cycle_from_csv(csv_path)

    with pytest.raises(ValueError, match="segment_type == 'normal_cycle'"):
        calibrate_reference_template((calibration_cycles[0], calibration_cycles[1], candidate))


def test_one_sample_candidate_is_allowed(tmp_path: Path) -> None:
    csv_path = tmp_path / "one_sample.csv"
    write_csv(
        csv_path,
        [{"timestamp": "2026-01-01T08:00:00Z", "machine_id": "CNC_01", "power_kw": "1.0"}],
    )

    cycle = load_candidate_cycle_from_csv(csv_path)
    metrics = compute_cycle_metrics(cycle.samples, cycle_id=cycle.segment_id)

    assert len(cycle.samples) == 1
    assert metrics.duration_seconds == 0


def test_valid_calibration_csvs_load_in_order_with_default_ids(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"calibration_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)

    loaded_cycles = load_calibration_cycles_from_csv(csv_paths)

    assert len(loaded_cycles) == 3
    assert [cycle.segment_id for cycle in loaded_cycles] == [
        "calibration_cycle_1",
        "calibration_cycle_2",
        "calibration_cycle_3",
    ]
    assert all(cycle.segment_type == "normal_cycle" for cycle in loaded_cycles)
    assert [cycle.samples[0].timestamp for cycle in loaded_cycles] == [
        cycle.samples[0].timestamp for cycle in calibration_cycles
    ]
    assert {cycle.samples[0].machine_id for cycle in loaded_cycles} == {"CNC_01"}


def test_calibration_csvs_preserve_custom_ids_in_order(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"known_good_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)

    loaded_cycles = load_calibration_cycles_from_csv(
        csv_paths,
        calibration_segment_ids=[" first ", "second", " third "],
    )

    assert [cycle.segment_id for cycle in loaded_cycles] == [
        "first",
        "second",
        "third",
    ]


def test_string_calibration_ids_argument_raises_clear_error(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"known_good_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)

    with pytest.raises(ValueError, match="sequence of IDs, not one string"):
        load_calibration_cycles_from_csv(
            csv_paths,
            calibration_segment_ids="abc",
        )


def test_duplicate_calibration_ids_raise_clear_error(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"cycle_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)

    with pytest.raises(ValueError, match="calibration_segment_ids must be unique"):
        load_calibration_cycles_from_csv(
            csv_paths,
            calibration_segment_ids=["one", "one", "three"],
        )


def test_blank_calibration_id_raises_clear_error(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"cycle_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)

    with pytest.raises(ValueError, match="calibration_segment_ids must be non-empty"):
        load_calibration_cycles_from_csv(
            csv_paths,
            calibration_segment_ids=["one", "   ", "three"],
        )


def test_mismatched_calibration_id_count_raises_clear_error(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"cycle_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)

    with pytest.raises(ValueError, match="must match csv_paths length"):
        load_calibration_cycles_from_csv(
            csv_paths,
            calibration_segment_ids=["one", "two"],
        )


def test_too_few_calibration_csv_paths_raise_clear_error(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles[:2], start=1):
        csv_path = tmp_path / f"cycle_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)

    with pytest.raises(ValueError, match="at least three"):
        load_calibration_cycles_from_csv(csv_paths)


@pytest.mark.parametrize("bad_paths", ["not-a-sequence.csv", Path("cycle.csv")])
def test_single_path_argument_for_calibration_raises_clear_error(bad_paths) -> None:
    with pytest.raises(ValueError, match="sequence of local CSV paths"):
        load_calibration_cycles_from_csv(bad_paths)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("timestamp,machine_id\n2026-01-01T08:00:00Z,CNC_01", "schema requires exactly"),
        (
            "timestamp,machine_id,power_kw\nnot-a-time,CNC_01,1.0",
            "timestamp must be ISO-8601",
        ),
        (
            "timestamp,machine_id,power_kw\n2026-01-01T08:00:00Z,CNC_01,1.0\n\n",
            "blank data row",
        ),
        (
            "timestamp,machine_id,power_kw\n2026-01-01T08:00:00Z,CNC_01,nan",
            "power_kw must be finite",
        ),
    ],
)
def test_calibration_csvs_reuse_strict_csv_validation(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"cycle_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)
    csv_paths[1].write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_calibration_cycles_from_csv(csv_paths)


def test_calibration_csvs_with_mixed_machine_ids_raise_clear_error(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"cycle_{index}.csv"
        rows = csv_rows_from_cycle(cycle)
        if index == 3:
            rows = [{**row, "machine_id": "CNC_OTHER"} for row in rows]
        write_csv(csv_path, rows)
        csv_paths.append(csv_path)

    with pytest.raises(ValueError, match="all calibration cycles must use one machine_id"):
        load_calibration_cycles_from_csv(csv_paths)


def test_loaded_calibration_csvs_integrate_with_calibration_and_threshold(
    tmp_path: Path,
) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_paths = []
    for index, cycle in enumerate(calibration_cycles, start=1):
        csv_path = tmp_path / f"cycle_{index}.csv"
        write_csv(csv_path, csv_rows_from_cycle(cycle))
        csv_paths.append(csv_path)

    loaded_cycles = load_calibration_cycles_from_csv(csv_paths)
    calibration = calibrate_reference_template(loaded_cycles)
    threshold_result = derive_dtw_threshold(calibration)

    assert calibration.reference_cycle in loaded_cycles
    assert threshold_result.reference_segment_id == calibration.reference_cycle.segment_id
    for cycle in loaded_cycles:
        assert compute_cycle_metrics(cycle.samples, cycle_id=cycle.segment_id)


def test_candidate_loader_still_returns_candidate_cycle(tmp_path: Path) -> None:
    calibration_cycles, _, _, _ = make_demo_pipeline()
    csv_path = tmp_path / "candidate.csv"
    write_csv(csv_path, csv_rows_from_cycle(calibration_cycles[0]))

    candidate = load_candidate_cycle_from_csv(csv_path)

    assert candidate.segment_type == "candidate_cycle"
