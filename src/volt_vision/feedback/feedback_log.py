"""Append-only local JSONL feedback history for Copilot review UX."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from volt_vision.feedback.models import CopilotFeedbackRecord

DEFAULT_FEEDBACK_HISTORY_PATH = Path(__file__).resolve().parents[3] / "data" / (
    "demo_copilot_feedback.jsonl"
)


def create_copilot_feedback_record(
    *,
    event_id: str,
    screening_status: str,
    execution_mode: str,
    feedback_outcome: str,
    human_review_acknowledged: bool,
) -> CopilotFeedbackRecord:
    """Create one safe local feedback record with generated local metadata."""

    return CopilotFeedbackRecord(
        feedback_id=str(uuid4()),
        recorded_at=datetime.now(UTC),
        event_id=event_id,
        screening_status=screening_status,
        execution_mode=execution_mode,
        feedback_outcome=feedback_outcome,
        human_review_acknowledged=human_review_acknowledged,
    )


def append_copilot_feedback_record(
    record: CopilotFeedbackRecord,
    log_path: str | Path,
) -> None:
    """Append one feedback record as a single JSONL object."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as log_file:
        log_file.write(f"{line}\n")


def read_copilot_feedback_records(
    log_path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[CopilotFeedbackRecord, ...]:
    """Read local feedback records from JSONL with safe line-number errors."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer or None")

    path = Path(log_path)
    if not path.exists():
        return ()

    records: list[CopilotFeedbackRecord] = []
    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            records.append(_parse_feedback_line(line, line_number))

    if limit is None:
        return tuple(records)
    return tuple(records[-limit:])


def clear_copilot_feedback_log_for_demo(log_path: str | Path) -> None:
    """Remove the local feedback JSONL history for an explicit demo reset."""

    path = Path(log_path)
    if path.exists():
        path.unlink()


def _parse_feedback_line(line: str, line_number: int) -> CopilotFeedbackRecord:
    try:
        payload: Any = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL feedback at line {line_number}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"JSONL feedback at line {line_number} must be a JSON object")

    try:
        return CopilotFeedbackRecord.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"invalid CopilotFeedbackRecord payload at line {line_number}"
        ) from exc
