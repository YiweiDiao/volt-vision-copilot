from __future__ import annotations

from pathlib import Path

from volt_vision.agent import run_incident_copilot
from volt_vision.feedback.models import CopilotFeedbackRecord
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.ui.copilot import (
    build_copilot_feedback_record,
    build_feedback_history_frame,
    can_investigate_event,
    can_record_feedback,
    get_selected_persisted_event,
    is_current_result_for_event,
    latest_events_by_id,
)

from test_mcp_services import make_event


def test_latest_duplicate_persisted_event_id_keeps_last_record() -> None:
    old = make_event("duplicate", seconds=30, normalized_dtw_distance=0.11)
    latest = make_event("duplicate", seconds=60, normalized_dtw_distance=0.22)

    collapsed = latest_events_by_id((old, latest))

    assert collapsed == (latest,)


def test_investigation_eligibility_rejects_unsaved_and_normal_events() -> None:
    normal = make_event("normal", seconds=60, status="within_normal_band")

    assert can_investigate_event(None) is False
    assert can_investigate_event(normal) is False


def test_investigation_eligibility_accepts_persisted_suspected_event() -> None:
    event = make_event("query", seconds=60)

    assert can_investigate_event(event) is True


def test_selected_event_returns_none_after_log_clear_semantics() -> None:
    event = make_event("query", seconds=60)

    assert get_selected_persisted_event((event,), "query") == event
    assert get_selected_persisted_event((), "query") is None


def test_feedback_eligibility_rejects_mismatches_and_missing_ack(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "events.jsonl"
    event = make_event("query", seconds=60)
    append_monitoring_event(event, log_path)
    result = run_incident_copilot("query", history_path=log_path, model=None)
    other_event = make_event("other", seconds=90)

    assert can_record_feedback(event, result, human_review_acknowledged=False) is False
    assert can_record_feedback(other_event, result, human_review_acknowledged=True) is False
    assert is_current_result_for_event(result, event) is True
    assert is_current_result_for_event(result, other_event) is False


def test_feedback_record_builds_for_acknowledged_valid_result(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    event = make_event("query", seconds=60)
    append_monitoring_event(event, log_path)
    result = run_incident_copilot("query", history_path=log_path, model=None)

    record = build_copilot_feedback_record(
        event=event,
        result=result,
        feedback_outcome="useful",
        human_review_acknowledged=True,
    )

    assert isinstance(record, CopilotFeedbackRecord)
    assert record.event_id == event.event_id
    assert record.screening_status == "suspected_deviation"
    assert record.execution_mode == "deterministic_fallback"
    assert record.feedback_outcome == "useful"
    assert record.human_review_acknowledged is True
    assert event.status == "suspected_deviation"
    assert result.recommendation.event_id == event.event_id


def test_feedback_builder_rejects_missing_acknowledgement(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    event = make_event("query", seconds=60)
    append_monitoring_event(event, log_path)
    result = run_incident_copilot("query", history_path=log_path, model=None)

    try:
        build_copilot_feedback_record(
            event=event,
            result=result,
            feedback_outcome="useful",
            human_review_acknowledged=False,
        )
    except ValueError as exc:
        assert "feedback is not eligible" in str(exc)
    else:
        raise AssertionError("missing acknowledgement should be rejected")


def test_feedback_history_frame_uses_safe_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    event = make_event("query", seconds=60)
    append_monitoring_event(event, log_path)
    result = run_incident_copilot("query", history_path=log_path, model=None)
    record = build_copilot_feedback_record(
        event=event,
        result=result,
        feedback_outcome="needs_follow_up",
        human_review_acknowledged=True,
    )

    frame = build_feedback_history_frame((record,))

    assert frame.columns.tolist() == [
        "recorded_at",
        "event_id",
        "screening_status",
        "execution_mode",
        "feedback_outcome",
        "human_review_acknowledged",
    ]
    frame_text = " ".join(frame.astype(str).to_numpy().ravel()).lower()
    for forbidden in ("raw", "samples", "csv", "prompt", "secret", "tool_payload"):
        assert forbidden not in frame_text


def test_helper_selection_and_eligibility_do_not_write_files(tmp_path: Path) -> None:
    event = make_event("query", seconds=60)
    untouched_path = tmp_path / "feedback.jsonl"

    latest_events_by_id((event,))
    get_selected_persisted_event((event,), event.event_id)
    can_investigate_event(event)

    assert not untouched_path.exists()
