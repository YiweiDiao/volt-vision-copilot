"""Smoke test Copilot fallback investigation plus local feedback recording."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from volt_vision.agent import run_incident_copilot
from volt_vision.feedback.feedback_log import (
    append_copilot_feedback_record,
    read_copilot_feedback_records,
)
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)
from volt_vision.ui.copilot import build_copilot_feedback_record


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        event_history_path = tmp_path / "events.jsonl"
        feedback_history_path = tmp_path / "feedback.jsonl"
        event = _make_event()
        append_monitoring_event(event, event_history_path)
        result = run_incident_copilot(
            event.event_id,
            history_path=event_history_path,
            model=None,
        )
        record = build_copilot_feedback_record(
            event=event,
            result=result,
            feedback_outcome="useful",
            human_review_acknowledged=True,
        )
        append_copilot_feedback_record(record, feedback_history_path)
        records = read_copilot_feedback_records(feedback_history_path)

    print(f"event_id={record.event_id}")
    print(f"execution_mode={record.execution_mode}")
    print(f"feedback_outcome={record.feedback_outcome}")
    print(f"human_review_acknowledged={record.human_review_acknowledged}")
    print(f"feedback_record_count={len(records)}")
    print(f"trace_tool_names={','.join(result.trace.tool_names)}")

def _make_event() -> MonitoringEvent:
    start = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    end = start + timedelta(seconds=80)
    return MonitoringEvent(
        event_id="smoke-event-001",
        event_type="cycle_screening",
        event_timestamp=end,
        machine_id="CNC_SMOKE",
        candidate_segment_id="candidate",
        reference_segment_id="reference",
        status="suspected_deviation",
        recommended_action="manual_review_required",
        evidence="Normalized DTW distance compared with calibrated threshold.",
        normalized_dtw_distance=0.24,
        threshold=0.10,
        metrics=CycleMetrics(
            cycle_id="candidate",
            machine_id="CNC_SMOKE",
            start_timestamp=start,
            end_timestamp=end,
            duration_seconds=80,
            energy_kwh=0.35,
            average_power_kw=15.75,
            peak_power_kw=22.0,
            sample_count=2,
        ),
        indicators=ReferenceRelativeIndicators(
            reference_cycle_id="reference",
            candidate_cycle_id="candidate",
            duration_deviation_pct=12.0,
            energy_deviation_pct=18.0,
            peak_power_deviation_pct=9.0,
        ),
    )


if __name__ == "__main__":
    main()
