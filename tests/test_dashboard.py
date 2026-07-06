from __future__ import annotations

import importlib
import inspect
import math
from pathlib import Path

from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.event_log import (
    append_monitoring_event,
    clear_monitoring_event_log_for_demo,
    read_monitoring_events,
)
from volt_vision.monitoring.models import ReferenceRelativeIndicators
from volt_vision.ui import app as app_module
from volt_vision.ui.dashboard import (
    EVENT_HISTORY_COLUMNS,
    build_event_history_frame,
    build_demo_replay_data,
    build_power_comparison_frame,
    format_indicator_percentage,
    get_demo_operating_state_at_elapsed_seconds,
    recommended_follow_up_text,
    run_synthetic_demo_workflow,
)


FORBIDDEN_STATE_TERMS = (
    "abnormal",
    "fault",
    "failure",
    "tool",
    "wear",
    "root cause",
    "diagnosis",
)


def test_synthetic_normal_demo_flow() -> None:
    result = run_synthetic_demo_workflow(candidate="normal")
    event = result.event

    assert event.status == "within_normal_band"
    assert math.isfinite(event.normalized_dtw_distance)
    assert event.normalized_dtw_distance >= 0
    assert math.isfinite(event.threshold)
    assert event.threshold >= 0
    assert event.machine_id == "CNC_01"
    assert event.event_timestamp == event.metrics.end_timestamp
    assert event.recommended_action == "no_automated_action"


def test_synthetic_changed_demo_flow_uses_cautious_evidence() -> None:
    result = run_synthetic_demo_workflow(candidate="changed")
    event = result.event

    assert event.status == "suspected_deviation"
    assert event.normalized_dtw_distance > event.threshold
    assert event.recommended_action == "manual_review_required"
    forbidden_terms = [
        "fault",
        "failure",
        "tool wear",
        "root cause",
        "diagnosis",
        "maintenance",
    ]
    assert all(term not in event.evidence.lower() for term in forbidden_terms)


def test_power_comparison_frame_preserves_raw_samples() -> None:
    result = run_synthetic_demo_workflow(candidate="changed")

    frame = build_power_comparison_frame(
        result.reference_cycle,
        result.candidate_cycle,
        reference_label="Reference cycle",
        candidate_label="Candidate cycle",
    )

    reference_rows = frame[frame["series"] == "Reference cycle"]
    candidate_rows = frame[frame["series"] == "Candidate cycle"]
    assert reference_rows.iloc[0]["elapsed_seconds"] == 0
    assert candidate_rows.iloc[0]["elapsed_seconds"] == 0
    assert reference_rows["power_kw"].tolist() == [
        sample.power_kw for sample in result.reference_cycle.samples
    ]
    assert candidate_rows["power_kw"].tolist() == [
        sample.power_kw for sample in result.candidate_cycle.samples
    ]
    assert set(frame.columns) == {"elapsed_seconds", "power_kw", "series"}
    assert "expected_label" not in frame.columns
    assert "segment_type" not in frame.columns


def test_demo_replay_data_preserves_raw_timeline_samples() -> None:
    timeline = generate_demo_timeline()

    replay_data = build_demo_replay_data(timeline)

    assert replay_data.samples.iloc[0]["elapsed_seconds"] == 0
    assert replay_data.samples["elapsed_seconds"].is_monotonic_increasing
    assert replay_data.samples["power_kw"].tolist() == [
        sample.power_kw for sample in timeline.samples
    ]
    assert set(replay_data.samples.columns) == {"elapsed_seconds", "power_kw"}
    assert "expected_label" not in replay_data.samples.columns
    assert "segment_type" not in replay_data.samples.columns


def test_demo_replay_state_mapping_uses_neutral_labels() -> None:
    timeline = generate_demo_timeline()
    replay_data = build_demo_replay_data(timeline)

    idle_state = get_demo_operating_state_at_elapsed_seconds(replay_data, 1)
    normal_state = get_demo_operating_state_at_elapsed_seconds(replay_data, 20)
    changed_state = get_demo_operating_state_at_elapsed_seconds(replay_data, 241)

    assert idle_state == "Idle"
    assert normal_state == "Processing"
    assert changed_state == "Processing"
    for state in (idle_state, normal_state, changed_state):
        assert all(term not in state.lower() for term in FORBIDDEN_STATE_TERMS)


def test_demo_replay_boundary_behavior_is_deterministic() -> None:
    timeline = generate_demo_timeline()
    replay_data = build_demo_replay_data(timeline)
    timeline_start = timeline.samples[0].timestamp
    first_segment = timeline.segments[0]
    second_segment = timeline.segments[1]

    first_end_elapsed = (
        first_segment.end_timestamp - timeline_start
    ).total_seconds()
    second_start_elapsed = (
        second_segment.start_timestamp - timeline_start
    ).total_seconds()

    # Segment start and end timestamps are inclusive. The exact start of a new
    # segment belongs to the segment beginning there.
    assert get_demo_operating_state_at_elapsed_seconds(
        replay_data,
        first_end_elapsed,
    ) == "Idle"
    assert get_demo_operating_state_at_elapsed_seconds(
        replay_data,
        second_start_elapsed,
    ) == "Processing"


def test_demo_replay_helpers_do_not_expose_expected_labels() -> None:
    replay_data = build_demo_replay_data()

    assert "expected_label" not in replay_data.samples.columns
    assert all(
        not hasattr(interval, "expected_label")
        for interval in replay_data.state_intervals
    )
    assert {interval.state for interval in replay_data.state_intervals} == {
        "Idle",
        "Processing",
    }


def test_synthetic_dashboard_workflow_is_deterministic() -> None:
    first = run_synthetic_demo_workflow(candidate="changed")
    second = run_synthetic_demo_workflow(candidate="changed")

    assert first.event.event_id == second.event.event_id
    assert first.event.status == second.event.status
    assert first.event.threshold == second.event.threshold
    assert first.event.normalized_dtw_distance == second.event.normalized_dtw_distance


def test_dashboard_indicator_percentage_formatting() -> None:
    assert format_indicator_percentage(12.0) == "+12.00%"
    assert format_indicator_percentage(-20.0) == "-20.00%"
    assert format_indicator_percentage(None) == (
        "Not available (reference value is zero)"
    )


def test_dashboard_recommended_follow_up_text_is_bounded() -> None:
    normal_text = recommended_follow_up_text("no_automated_action")
    changed_text = recommended_follow_up_text("manual_review_required")

    assert normal_text == (
        "Recommended follow-up: No automated action is proposed by this prototype."
    )
    assert changed_text == (
        "Recommended follow-up: Manual inspection required before any action."
    )
    forbidden_terms = [
        "fault",
        "failure",
        "tool wear",
        "root cause",
        "diagnosis",
        "maintenance",
    ]
    assert all(term not in normal_text.lower() for term in forbidden_terms)
    assert all(term not in changed_text.lower() for term in forbidden_terms)


def test_event_history_frame_uses_compact_allowed_fields() -> None:
    normal = run_synthetic_demo_workflow(candidate="normal").event
    changed = run_synthetic_demo_workflow(candidate="changed").event

    frame = build_event_history_frame((normal, changed))

    assert frame.shape[0] == 2
    assert frame.columns.tolist() == EVENT_HISTORY_COLUMNS
    assert frame["Event ID"].tolist() == [normal.event_id, changed.event_id]
    assert frame["Status"].tolist() == ["within_normal_band", "suspected_deviation"]
    assert frame["Recommended action"].tolist() == [
        "no_automated_action",
        "manual_review_required",
    ]
    assert frame.iloc[0]["Duration deviation percentage"].startswith("-")
    assert frame.iloc[1]["Duration deviation percentage"].startswith("+")
    frame_text = " ".join(frame.astype(str).to_numpy().ravel())
    forbidden_terms = [
        "samples",
        "expected_label",
        "segment_type",
        ".csv",
        "fault",
        "failure",
        "tool wear",
        "root cause",
        "diagnosis",
    ]
    assert all(term not in frame_text for term in forbidden_terms)


def test_event_history_frame_handles_empty_history() -> None:
    frame = build_event_history_frame(())

    assert frame.empty
    assert frame.columns.tolist() == EVENT_HISTORY_COLUMNS


def test_event_history_frame_formats_none_indicator_values() -> None:
    normal = run_synthetic_demo_workflow(candidate="normal").event
    event_with_none = normal.model_copy(
        update={
            "indicators": ReferenceRelativeIndicators(
                reference_cycle_id=normal.indicators.reference_cycle_id,
                candidate_cycle_id=normal.indicators.candidate_cycle_id,
                duration_deviation_pct=None,
                energy_deviation_pct=12.0,
                peak_power_deviation_pct=-20.0,
            )
        }
    )

    frame = build_event_history_frame((event_with_none,))

    assert frame.iloc[0]["Duration deviation percentage"] == (
        "Not available (reference value is zero)"
    )
    assert frame.iloc[0]["Energy deviation percentage"] == "+12.00%"
    assert frame.iloc[0]["Peak-power deviation percentage"] == "-20.00%"


def test_event_history_log_integration_uses_saved_event_order(tmp_path: Path) -> None:
    normal = run_synthetic_demo_workflow(candidate="normal").event
    changed = run_synthetic_demo_workflow(candidate="changed").event
    log_path = tmp_path / "events.jsonl"

    append_monitoring_event(normal, log_path)
    append_monitoring_event(changed, log_path)
    frame = build_event_history_frame(read_monitoring_events(log_path))

    assert frame["Event ID"].tolist() == [normal.event_id, changed.event_id]
    assert frame["Status"].tolist() == ["within_normal_band", "suspected_deviation"]


def test_clear_local_demo_history_supports_empty_history(tmp_path: Path) -> None:
    normal = run_synthetic_demo_workflow(candidate="normal").event
    log_path = tmp_path / "events.jsonl"
    append_monitoring_event(normal, log_path)

    clear_monitoring_event_log_for_demo(log_path)

    assert read_monitoring_events(log_path) == ()
    assert build_event_history_frame(read_monitoring_events(log_path)).empty


def test_explicit_save_is_only_app_append_call_site() -> None:
    app_source = inspect.getsource(app_module)
    synthetic_source = inspect.getsource(run_synthetic_demo_workflow)
    csv_source = inspect.getsource(app_module.run_csv_workflow_from_uploads)

    assert app_source.count("append_monitoring_event(") == 1
    assert "append_monitoring_event" in inspect.getsource(
        app_module.render_event_save_controls
    )
    assert "append_monitoring_event" not in synthetic_source
    assert "append_monitoring_event" not in csv_source


def test_rendered_monitoring_result_is_session_state_safe() -> None:
    result = run_synthetic_demo_workflow(candidate="normal")
    render_context = app_module.RenderedMonitoringResult(
        mode="synthetic",
        event=result.event,
        calibration_cycle_count=len(result.calibration_cycles),
        reference_cycle=result.reference_cycle,
        candidate_cycle=result.candidate_cycle,
        reference_cycle_id=result.calibration_result.reference_cycle.segment_id,
        threshold_result=result.threshold_result,
    )

    assert render_context.event == result.event
    assert render_context.reference_cycle == result.reference_cycle
    assert render_context.candidate_cycle == result.candidate_cycle
    assert not hasattr(render_context, "uploaded_file")
    assert not hasattr(render_context, "csv_bytes")


def test_app_import_does_not_launch_streamlit_or_run_monitoring() -> None:
    app = importlib.import_module("volt_vision.ui.app")

    assert callable(app.main)
