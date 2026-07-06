"""Typed guidance payloads for local deterministic retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceItem:
    """One conservative manual-review guidance item."""

    guidance_id: str
    title: str
    applies_to: tuple[str, ...]
    inspection_checks: tuple[str, ...]
    evidence_to_record: tuple[str, ...]
    escalation_condition: str
    safety_note: str
    possible_contributing_conditions: tuple[str, ...]
    operator_questions: tuple[str, ...]
    safe_inspection_checks: tuple[str, ...]
    do_not_conclude: tuple[str, ...]
    reviewer_label: str
    knowledge_version: str
