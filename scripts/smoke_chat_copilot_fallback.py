"""Non-live smoke check for deterministic chat copilot fallback."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from tempfile import TemporaryDirectory
from pathlib import Path

from volt_vision.agent.chat import run_chat_copilot
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        history_path = Path(tmp_dir) / "history.jsonl"
        event = _synthetic_event()
        append_monitoring_event(event, history_path)
        response = run_chat_copilot(
            event.event_id,
            history_path=history_path,
            model=None,
        )
        print(f"event ID: {response.event_id}")
        print(f"execution mode: {response.execution_mode}")
        print(f"fallback reason: {response.fallback_reason}")
        print(f"knowledge source IDs: {', '.join(response.knowledge_source_ids)}")
        print(f"human approval required: {response.human_approval_required}")
        print(f"tool trace names: {', '.join(response.tool_trace.tool_names)}")
        print(f"accepted message character count: {len(response.assistant_message)}")


def _synthetic_event() -> MonitoringEvent:
    start = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    end = start + timedelta(seconds=80)
    return MonitoringEvent(
        event_id="synthetic_chat_smoke_event",
        event_type="cycle_screening",
        event_timestamp=end,
        machine_id="CNC_DEMO",
        candidate_segment_id="candidate",
        reference_segment_id="reference",
        status="suspected_deviation",
        recommended_action="manual_review_required",
        evidence="Normalized DTW distance compared with calibrated threshold.",
        normalized_dtw_distance=0.12,
        threshold=0.10,
        metrics=CycleMetrics(
            cycle_id="candidate",
            machine_id="CNC_DEMO",
            start_timestamp=start,
            end_timestamp=end,
            duration_seconds=80,
            energy_kwh=0.25,
            average_power_kw=15,
            peak_power_kw=20,
            sample_count=2,
        ),
        indicators=ReferenceRelativeIndicators(
            reference_cycle_id="reference",
            candidate_cycle_id="candidate",
            duration_deviation_pct=12.0,
            energy_deviation_pct=18.0,
            peak_power_deviation_pct=22.0,
        ),
    )


if __name__ == "__main__":
    main()
