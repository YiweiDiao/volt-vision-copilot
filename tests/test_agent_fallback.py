from __future__ import annotations

from pathlib import Path

import pytest

from volt_vision.agent.fallback import build_deterministic_fallback_result
from volt_vision.agent.policy import APPROVED_MCP_TOOL_NAMES
from volt_vision.guidance.retrieval import retrieve_guidance
from volt_vision.mcp_server.services import EVENT_NOT_FOUND_MESSAGE, EventNotFoundError
from volt_vision.monitoring.event_log import append_monitoring_event

from test_mcp_services import make_event


def test_suspected_deviation_without_model_uses_grounded_fallback(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.jsonl"
    prior = make_event("prior", seconds=30)
    event = make_event("query", seconds=80)
    append_monitoring_event(prior, history_path)
    append_monitoring_event(event, history_path)

    result = build_deterministic_fallback_result("query", history_path)

    expected_guidance = retrieve_guidance(event)
    expected_checks = tuple(
        dict.fromkeys(
            check
            for item in expected_guidance
            for check in item.inspection_checks
        )
    )
    assert result.recommendation.event_id == "query"
    assert result.recommendation.screening_status == "suspected_deviation"
    assert result.recommendation.guidance_ids == tuple(
        item.guidance_id for item in expected_guidance
    )
    assert result.recommendation.manual_review_checks == expected_checks
    assert result.recommendation.similar_event_ids == ("prior",)
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.fallback_reason == "model_not_configured"
    assert result.trace.tool_names == APPROVED_MCP_TOOL_NAMES
    assert tuple(tool_call.tool_name for tool_call in result.trace.tool_calls) == (
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
    )
    assert all(
        tool_call.source == "deterministic_service"
        and tool_call.outcome == "succeeded"
        and tool_call.error_code is None
        for tool_call in result.trace.tool_calls
    )


def test_fallback_trace_names_are_stable_and_exact(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)

    result = build_deterministic_fallback_result("query", history_path)

    assert result.trace.tool_names == (
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
    )
    assert tuple(tool_call.tool_name for tool_call in result.trace.tool_calls) == (
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
    )


def test_fallback_does_not_mutate_jsonl_history(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)
    before = history_path.read_bytes()

    build_deterministic_fallback_result("query", history_path)

    assert history_path.read_bytes() == before


def test_unknown_event_id_raises_existing_safe_domain_error(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("known", seconds=60), history_path)

    with pytest.raises(EventNotFoundError, match=EVENT_NOT_FOUND_MESSAGE):
        build_deterministic_fallback_result("unknown", history_path)


def test_fallback_result_excludes_forbidden_sensitive_content(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    append_monitoring_event(make_event("query", seconds=60), history_path)

    result_text = build_deterministic_fallback_result(
        "query",
        history_path,
    ).model_dump_json()

    forbidden_fragments = (
        "raw_samples",
        "power_samples",
        "csv",
        ".pdf",
        str(history_path),
        "api_key",
        "chain-of-thought",
        "traceback",
    )
    for fragment in forbidden_fragments:
        assert fragment.lower() not in result_text.lower()
