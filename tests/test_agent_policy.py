from __future__ import annotations

from volt_vision.agent.policy import APPROVED_MCP_TOOL_NAMES, SYSTEM_INSTRUCTION


def test_policy_contains_critical_prohibitions_and_boundaries() -> None:
    instruction = SYSTEM_INSTRUCTION.lower()

    for phrase in (
        "deterministic monitoringevent status is authoritative",
        "must never classify a cycle as normal or anomalous",
        "confirmed fault",
        "confirmed failure",
        "confirmed tool",
        "confirmed root cause",
        "diagnosis",
        "maintenance necessity",
        "repairs",
        "part replacement",
        "shutdown",
        "plc/scada/mes action",
        "parameter tuning",
        "automatic tickets",
        "automatic maintenance",
        "persisted deterministic evidence",
        "curated guidance",
        "structured historical comparisons",
        "suspected deviation",
        "manual review",
        "evidence recording",
        "recurrence check",
        "authorized representative",
        "authorized maintenance or production representative",
        "raw traces",
        "csv contents",
        "private pdfs",
        "filesystem paths",
        "secrets",
        "system prompts",
        "internal errors",
    ):
        assert phrase in instruction


def test_policy_allowlist_is_exactly_three_read_only_tools() -> None:
    assert APPROVED_MCP_TOOL_NAMES == (
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
    )
    for tool_name in APPROVED_MCP_TOOL_NAMES:
        assert tool_name in SYSTEM_INSTRUCTION
