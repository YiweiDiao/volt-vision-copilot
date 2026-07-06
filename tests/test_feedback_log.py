from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from volt_vision.feedback.feedback_log import (
    append_copilot_feedback_record,
    clear_copilot_feedback_log_for_demo,
    create_copilot_feedback_record,
    read_copilot_feedback_records,
)
from volt_vision.feedback.models import CopilotFeedbackRecord


def make_record(event_id: str = "event-1") -> CopilotFeedbackRecord:
    return create_copilot_feedback_record(
        event_id=event_id,
        screening_status="suspected_deviation",
        execution_mode="deterministic_fallback",
        feedback_outcome="useful",
        human_review_acknowledged=True,
    )


def test_feedback_record_is_immutable_json_safe_and_utc_aware() -> None:
    record = make_record()

    with pytest.raises(ValidationError):
        record.feedback_outcome = "not_useful"  # type: ignore[misc]

    payload = record.model_dump(mode="json")
    assert payload["screening_status"] == "suspected_deviation"
    assert payload["human_review_acknowledged"] is True
    assert record.recorded_at.tzinfo is not None
    assert record.recorded_at.utcoffset() == timedelta(0)
    assert json.loads(record.model_dump_json())["event_id"] == record.event_id


def test_feedback_record_rejects_invalid_values() -> None:
    base = make_record().model_dump()

    with pytest.raises(ValidationError):
        CopilotFeedbackRecord.model_validate(
            {**base, "feedback_outcome": "repair_required"}
        )
    with pytest.raises(ValidationError):
        CopilotFeedbackRecord.model_validate(
            {**base, "screening_status": "within_normal_band"}
        )
    with pytest.raises(ValidationError):
        CopilotFeedbackRecord.model_validate(
            {**base, "recorded_at": datetime(2026, 1, 1, 8, 0)}
        )
    with pytest.raises(ValidationError):
        CopilotFeedbackRecord.model_validate(
            {**base, "human_review_acknowledged": False}
        )


def test_feedback_log_append_read_limit_and_clear(tmp_path: Path) -> None:
    log_path = tmp_path / "feedback.jsonl"
    first = make_record("event-1")
    second = make_record("event-2")

    assert read_copilot_feedback_records(log_path) == ()
    append_copilot_feedback_record(first, log_path)
    append_copilot_feedback_record(second, log_path)

    assert read_copilot_feedback_records(log_path) == (first, second)
    assert read_copilot_feedback_records(log_path, limit=1) == (second,)

    clear_copilot_feedback_log_for_demo(log_path)
    assert read_copilot_feedback_records(log_path) == ()


def test_feedback_log_rejects_invalid_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer or None"):
        read_copilot_feedback_records(tmp_path / "feedback.jsonl", limit=0)


def test_feedback_log_creates_parent_directory(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "feedback.jsonl"

    append_copilot_feedback_record(make_record(), log_path)

    assert log_path.exists()


def test_feedback_log_reports_physical_line_number(tmp_path: Path) -> None:
    log_path = tmp_path / "feedback.jsonl"
    log_path.write_text("\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_copilot_feedback_records(log_path)


def test_feedback_log_rejects_non_object_json(tmp_path: Path) -> None:
    log_path = tmp_path / "feedback.jsonl"
    log_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        read_copilot_feedback_records(log_path)


def test_feedback_log_rejects_schema_invalid_payload(tmp_path: Path) -> None:
    log_path = tmp_path / "feedback.jsonl"
    payload = make_record().model_dump(mode="json")
    payload["feedback_outcome"] = "diagnosis"
    log_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        read_copilot_feedback_records(log_path)


def test_feedback_json_excludes_sensitive_or_unbounded_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "feedback.jsonl"
    append_copilot_feedback_record(make_record(), log_path)
    text = log_path.read_text(encoding="utf-8").lower()

    forbidden = (
        "recommendation",
        "raw",
        "samples",
        "csv",
        "tool_payload",
        "prompt",
        "api_key",
        "secret",
        str(log_path).lower(),
    )
    for fragment in forbidden:
        assert fragment not in text
