"""Thin official-SDK MCP wrapper for local read-only monitoring tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from volt_vision.mcp_server.models import (
    EventMetricsSummary,
    GuidanceItemSummary,
    SimilarEventSummary,
)
from volt_vision.mcp_server.services import (
    DEFAULT_HISTORY_PATH,
    find_similar_previous_events as find_similar_previous_events_service,
    get_event_metrics as get_event_metrics_service,
    retrieve_maintenance_guidance as retrieve_maintenance_guidance_service,
)

HISTORY_PATH_ENV_VAR = "VOLT_VISION_EVENT_HISTORY_PATH"


def create_mcp_server(history_path: str | Path | None = None) -> FastMCP:
    """Create the import-safe local MCP server without starting transport."""

    resolved_history_path = _resolve_history_path(history_path)
    mcp = FastMCP(
        "volt-vision-readonly",
        log_level="ERROR",
        instructions=(
            "Local stdio-only read-only tools for persisted deterministic "
            "MonitoringEvent records. Tools return structured event evidence, "
            "derived metrics, indicators, and curated manual-review guidance "
            "only; they do not diagnose root cause, confirm faults, recommend "
            "repairs, control machines, or write data."
        ),
    )

    @mcp.tool(
        description=(
            "Read-only. Return structured persisted MonitoringEvent metrics for "
            "manual review. Does not return raw traces, recompute status, "
            "diagnose faults, or control equipment."
        ),
        structured_output=True,
    )
    def get_event_metrics(event_id: str) -> dict[str, Any]:
        result: EventMetricsSummary = get_event_metrics_service(
            event_id,
            resolved_history_path,
        )
        return result.model_dump(mode="json")

    @mcp.tool(
        description=(
            "Read-only. Retrieve deterministic curated maintenance guidance for "
            "a persisted event. Guidance is limited to manual inspection and "
            "evidence recording; it does not confirm root cause or recommend "
            "repair, shutdown, tuning, tickets, or machine control."
        ),
        structured_output=True,
    )
    def retrieve_maintenance_guidance(event_id: str) -> list[dict[str, Any]]:
        results: tuple[GuidanceItemSummary, ...] = retrieve_maintenance_guidance_service(
            event_id,
            resolved_history_path,
        )
        return [item.model_dump(mode="json") for item in results]

    @mcp.tool(
        description=(
            "Read-only. Rank earlier same-machine, same-status persisted events "
            "using structured screening evidence only. This is a historical "
            "ranking heuristic, not root-cause confirmation, diagnosis, repair "
            "advice, or machine-control authorization."
        ),
        structured_output=True,
    )
    def find_similar_previous_events(
        event_id: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        results: tuple[SimilarEventSummary, ...] = find_similar_previous_events_service(
            event_id,
            limit,
            resolved_history_path,
        )
        return [item.model_dump(mode="json") for item in results]

    return mcp


def main() -> None:
    """Run the local MCP server over stdio."""

    create_mcp_server().run(transport="stdio")


def _resolve_history_path(history_path: str | Path | None) -> Path:
    if history_path is not None:
        return Path(history_path)
    env_value = os.environ.get(HISTORY_PATH_ENV_VAR)
    if env_value:
        return Path(env_value)
    return DEFAULT_HISTORY_PATH


if __name__ == "__main__":
    main()
