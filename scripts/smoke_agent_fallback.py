"""Smoke test the no-model incident copilot fallback."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from volt_vision.agent import run_incident_copilot
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        history_path = Path(tmp_dir) / "history.jsonl"
        event = _make_event()
        append_monitoring_event(event, history_path)
        result = run_incident_copilot(event.event_id, history_path=history_path)

    print(f"execution_mode={result.trace.execution_mode}")
    print(f"event_id={result.recommendation.event_id}")
    print(f"screening_status={result.recommendation.screening_status}")
    print(f"guidance_ids={','.join(result.recommendation.guidance_ids)}")
    print(f"trace_tool_names={','.join(result.trace.tool_names)}")
    print(
        "trace_tool_calls="
        + ",".join(
            f"{item.tool_name}:{item.source}:{item.outcome}"
            for item in result.trace.tool_calls
        )
    )
    print(
        "human_approval_required="
        f"{result.recommendation.human_approval_required}"
    )


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
