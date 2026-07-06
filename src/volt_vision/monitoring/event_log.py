"""Local JSONL MonitoringEvent history for educational demo review.

This module provides append-only local event history for the prototype. It is
not an industrial audit log, compliance system, diagnostic system, or action
trigger.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from volt_vision.monitoring.models import MonitoringEvent


def append_monitoring_event(
    event: MonitoringEvent,
    log_path: str | Path,
) -> None:
    """Append one structured MonitoringEvent as a single JSONL record."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = event.model_dump(mode="json")
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as log_file:
        log_file.write(f"{line}\n")


def read_monitoring_events(
    log_path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[MonitoringEvent, ...]:
    """Read structured MonitoringEvent records from a local JSONL file.

    Whitespace-only blank lines are ignored anywhere in the file. Corrupted
    non-empty lines are rejected with their physical line number.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer or None")

    path = Path(log_path)
    if not path.exists():
        return ()

    events: list[MonitoringEvent] = []
    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            events.append(_parse_event_line(line, line_number))

    if limit is None:
        return tuple(events)
    return tuple(events[-limit:])


def clear_monitoring_event_log_for_demo(log_path: str | Path) -> None:
    """Remove the local demo event log so a demo/test can start cleanly."""

    path = Path(log_path)
    if path.exists():
        path.unlink()


def _parse_event_line(line: str, line_number: int) -> MonitoringEvent:
    try:
        payload: Any = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL event at line {line_number}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"JSONL event at line {line_number} must be a JSON object")

    try:
        return MonitoringEvent.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"invalid MonitoringEvent payload at line {line_number}"
        ) from exc
