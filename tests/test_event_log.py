from __future__ import annotations

import json
from pathlib import Path

import pytest

from volt_vision.monitoring.calibration import calibrate_reference_template
from volt_vision.monitoring.cycles import (
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.event_log import (
    append_monitoring_event,
    clear_monitoring_event_log_for_demo,
    read_monitoring_events,
)
from volt_vision.monitoring.events import build_monitoring_event
from volt_vision.monitoring.thresholds import derive_dtw_threshold


def make_demo_events():
    timeline = generate_demo_timeline()
    calibration_cycles = select_calibration_cycles(timeline)
    calibration = calibrate_reference_template(calibration_cycles)
    threshold_result = derive_dtw_threshold(calibration)
    normal_event = build_monitoring_event(
        calibration_cycles[0],
        calibration,
        threshold_result,
    )
    second_normal_event = build_monitoring_event(
        calibration_cycles[1],
        calibration,
        threshold_result,
    )
    changed_event = build_monitoring_event(
        select_abnormal_evaluation_cycle(timeline),
        calibration,
        threshold_result,
    )
    return normal_event, second_normal_event, changed_event


def test_append_and_read_one_event(tmp_path: Path) -> None:
    normal_event, _, _ = make_demo_events()
    log_path = tmp_path / "events.jsonl"

    append_monitoring_event(normal_event, log_path)

    assert read_monitoring_events(log_path) == (normal_event,)


def test_append_order_is_preserved(tmp_path: Path) -> None:
    normal_event, _, changed_event = make_demo_events()
    log_path = tmp_path / "events.jsonl"

    append_monitoring_event(normal_event, log_path)
    append_monitoring_event(changed_event, log_path)

    assert read_monitoring_events(log_path) == (normal_event, changed_event)


def test_read_limit_returns_most_recent_events_in_file_order(tmp_path: Path) -> None:
    first_event, second_event, third_event = make_demo_events()
    log_path = tmp_path / "events.jsonl"
    for event in (first_event, second_event, third_event):
        append_monitoring_event(event, log_path)

    assert read_monitoring_events(log_path, limit=2) == (second_event, third_event)


@pytest.mark.parametrize("bad_limit", [0, -1])
def test_read_limit_must_be_positive(tmp_path: Path, bad_limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        read_monitoring_events(tmp_path / "events.jsonl", limit=bad_limit)


def test_missing_file_returns_empty_tuple(tmp_path: Path) -> None:
    assert read_monitoring_events(tmp_path / "missing.jsonl") == ()


def test_append_creates_parent_directory(tmp_path: Path) -> None:
    normal_event, _, _ = make_demo_events()
    log_path = tmp_path / "nested" / "history" / "events.jsonl"

    append_monitoring_event(normal_event, log_path)

    assert log_path.exists()
    assert read_monitoring_events(log_path) == (normal_event,)


def test_jsonl_shape_is_one_json_object_per_line(tmp_path: Path) -> None:
    normal_event, _, changed_event = make_demo_events()
    log_path = tmp_path / "events.jsonl"

    append_monitoring_event(normal_event, log_path)
    append_monitoring_event(changed_event, log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert isinstance(payload, dict)
        assert payload["event_type"] == "cycle_screening"
        assert payload["recommended_action"] in {
            "no_automated_action",
            "manual_review_required",
        }
        assert "indicators" in payload


def test_malformed_json_line_raises_with_line_number(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log_path.write_text('{"not": "closed"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        read_monitoring_events(log_path)


def test_schema_invalid_json_object_raises_with_line_number(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log_path.write_text('{"event_id": "missing-required-fields"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        read_monitoring_events(log_path)


def test_non_object_json_line_raises_with_line_number(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    log_path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        read_monitoring_events(log_path)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    normal_event, _, changed_event = make_demo_events()
    log_path = tmp_path / "events.jsonl"
    append_monitoring_event(normal_event, log_path)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n   \n")
    append_monitoring_event(changed_event, log_path)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n")

    assert read_monitoring_events(log_path) == (normal_event, changed_event)


def test_clear_demo_log_removes_history(tmp_path: Path) -> None:
    normal_event, _, _ = make_demo_events()
    log_path = tmp_path / "events.jsonl"
    append_monitoring_event(normal_event, log_path)

    clear_monitoring_event_log_for_demo(log_path)

    assert read_monitoring_events(log_path) == ()


def test_jsonl_does_not_persist_raw_data_or_demo_labels(tmp_path: Path) -> None:
    normal_event, _, changed_event = make_demo_events()
    log_path = tmp_path / "events.jsonl"
    append_monitoring_event(normal_event, log_path)
    append_monitoring_event(changed_event, log_path)

    log_text = log_path.read_text(encoding="utf-8")

    forbidden_terms = [
        "expected_label",
        "segment_type",
        "samples",
        "power_samples",
        "candidate_normal.csv",
        "candidate_changed.csv",
    ]
    assert all(term not in log_text for term in forbidden_terms)
