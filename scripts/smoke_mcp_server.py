from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from volt_vision.mcp_server.server import HISTORY_PATH_ENV_VAR
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)


async def main() -> None:
    with TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "history.jsonl"
        append_monitoring_event(_make_event(), log_path)
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
                metrics = await session.call_tool(
                    "get_event_metrics",
                    {"event_id": "smoke-event"},
                )
                guidance = await session.call_tool(
                    "retrieve_maintenance_guidance",
                    {"event_id": "smoke-event"},
                )

    metrics_payload = _extract_payload(metrics)
    guidance_payload = _extract_payload(guidance)
    print("initialize: OK")
    print(f"tools: {','.join(tool.name for tool in tools.tools)}")
    print(f"get_event_metrics.event_id: {metrics_payload['event_id']}")
    print(
        "retrieve_maintenance_guidance.ids: "
        + ",".join(item["guidance_id"] for item in guidance_payload)
    )


def _make_event() -> MonitoringEvent:
    start = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    end = start + timedelta(seconds=60)
    return MonitoringEvent(
        event_id="smoke-event",
        event_type="cycle_screening",
        event_timestamp=end,
        machine_id="CNC_SMOKE",
        candidate_segment_id="candidate",
        reference_segment_id="reference",
        status="suspected_deviation",
        recommended_action="manual_review_required",
        evidence="Normalized DTW distance compared with calibrated threshold.",
        normalized_dtw_distance=0.12,
        threshold=0.10,
        metrics=CycleMetrics(
            cycle_id="candidate",
            machine_id="CNC_SMOKE",
            start_timestamp=start,
            end_timestamp=end,
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


def _extract_payload(result: object) -> Any:
    structured_content = getattr(result, "structuredContent", None)
    if structured_content and "result" in structured_content:
        return structured_content["result"]
    content = getattr(result, "content", ())
    if not content:
        raise ValueError("MCP tool result did not include content")
    text = getattr(content[0], "text", None)
    if text is None:
        raise ValueError("MCP tool result content was not text")
    return json.loads(text)


if __name__ == "__main__":
    asyncio.run(main())
