from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from volt_vision.mcp_server.server import HISTORY_PATH_ENV_VAR, create_mcp_server
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)


def make_event(event_id: str) -> MonitoringEvent:
    from datetime import UTC, datetime, timedelta

    start = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    event_timestamp = start + timedelta(seconds=60)
    return MonitoringEvent(
        event_id=event_id,
        event_type="cycle_screening",
        event_timestamp=event_timestamp,
        machine_id="CNC_TEST",
        candidate_segment_id="candidate",
        reference_segment_id="reference",
        status="suspected_deviation",
        recommended_action="manual_review_required",
        evidence="Normalized DTW distance compared with calibrated threshold.",
        normalized_dtw_distance=0.12,
        threshold=0.10,
        metrics=CycleMetrics(
            cycle_id="candidate",
            machine_id="CNC_TEST",
            start_timestamp=start,
            end_timestamp=event_timestamp,
            duration_seconds=60,
            energy_kwh=0.25,
            average_power_kw=15,
            peak_power_kw=20,
            sample_count=2,
        ),
        indicators=ReferenceRelativeIndicators(
            reference_cycle_id="reference",
            candidate_cycle_id="candidate",
            duration_deviation_pct=10.0,
            energy_deviation_pct=20.0,
            peak_power_deviation_pct=30.0,
        ),
    )


@pytest.mark.anyio
async def test_mcp_server_exposes_exactly_three_read_only_tools() -> None:
    server = create_mcp_server(history_path="unused.jsonl")

    tools = await server.list_tools()
    tool_names = {tool.name for tool in tools}

    assert tool_names == {
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
    }
    assert all("write" not in tool.name.lower() for tool in tools)
    assert all("read-only" in (tool.description or "").lower() for tool in tools)


@pytest.mark.anyio
async def test_mcp_protocol_smoke_with_official_stdio_client(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    event = make_event("query")
    append_monitoring_event(event, log_path)
    env = os.environ.copy()
    env[HISTORY_PATH_ENV_VAR] = str(log_path)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "volt_vision.mcp_server.server"],
        env=env,
        cwd=Path.cwd(),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}

            metrics = await session.call_tool(
                "get_event_metrics",
                {"event_id": "query"},
            )
            guidance = await session.call_tool(
                "retrieve_maintenance_guidance",
                {"event_id": "query"},
            )

    assert tool_names == {
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
    }
    metrics_payload = _extract_payload(metrics)
    guidance_payload = _extract_payload(guidance)
    assert metrics_payload["event_id"] == "query"
    assert metrics_payload["status"] == "suspected_deviation"
    assert [item["guidance_id"] for item in guidance_payload] == [
        "power_signature_review",
        "cycle_duration_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    ]


def _extract_payload(result: object) -> object:
    structured_content = getattr(result, "structuredContent", None)
    if structured_content and "result" in structured_content:
        return structured_content["result"]
    content = getattr(result, "content", ())
    assert content
    text = getattr(content[0], "text", None)
    assert text is not None
    return json.loads(text)
