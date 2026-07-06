"""Typed feedback payloads for explicit local human review records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CopilotFeedbackRecord(BaseModel):
    """One explicit local UX feedback record for a suspected-deviation review."""

    model_config = ConfigDict(frozen=True)

    feedback_id: str = Field(min_length=1)
    recorded_at: datetime
    event_id: str = Field(min_length=1)
    screening_status: Literal["suspected_deviation"]
    execution_mode: Literal["adk", "deterministic_fallback"]
    feedback_outcome: Literal["useful", "needs_follow_up", "not_useful"]
    human_review_acknowledged: Literal[True]

    @field_validator("recorded_at")
    @classmethod
    def recorded_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware UTC")
        if value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("recorded_at must be timezone-aware UTC")
        return value
