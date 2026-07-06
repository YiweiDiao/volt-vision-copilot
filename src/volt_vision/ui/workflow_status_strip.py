"""UI-safe display text for the workflow status strip."""

from __future__ import annotations

from dataclasses import dataclass

from volt_vision.ui.workflow import WorkflowSnapshot

WORKFLOW_PATH_LABEL = "Data ready -> Screen -> Save -> Investigate -> Feedback"
SAFETY_REMINDER = (
    "Copilot guidance supports human review and does not confirm a fault or "
    "root cause."
)
MODEL_ASSISTANCE_LABEL = "Model-assisted investigation available"
LOCAL_FALLBACK_LABEL = "Local deterministic fallback available"
ACTION_READY_LABEL = "Action available in the workflow controls below."
PREREQUISITE_REQUIRED_LABEL = (
    "Complete the prerequisite in the instruction before this action is available."
)


@dataclass(frozen=True)
class WorkflowStatusStripText:
    """Bounded display strings for the Streamlit workflow status strip."""

    path_label: str
    current_state_label: str
    primary_action_label: str
    short_instruction: str
    action_availability_label: str
    assistance_availability_label: str
    safety_reminder: str


def build_workflow_status_strip_text(
    snapshot: WorkflowSnapshot,
) -> WorkflowStatusStripText:
    """Map a workflow snapshot to compact user-facing status-strip text."""

    return WorkflowStatusStripText(
        path_label=WORKFLOW_PATH_LABEL,
        current_state_label=_current_state_label(snapshot),
        primary_action_label=snapshot.primary_action_label,
        short_instruction=snapshot.short_instruction,
        action_availability_label=(
            ACTION_READY_LABEL
            if snapshot.primary_action_enabled
            else PREREQUISITE_REQUIRED_LABEL
        ),
        assistance_availability_label=(
            MODEL_ASSISTANCE_LABEL
            if snapshot.model_assistance_available
            else LOCAL_FALLBACK_LABEL
        ),
        safety_reminder=SAFETY_REMINDER,
    )


def _current_state_label(snapshot: WorkflowSnapshot) -> str:
    if snapshot.investigation == "investigating":
        return "Copilot investigating"
    if snapshot.investigation == "investigation_available":
        return "Guidance available"
    if snapshot.investigation == "fallback_guidance_available":
        return "Fallback guidance available"
    if snapshot.screening == "within_normal_band":
        return "Within normal band"
    if snapshot.persistence == "save_required":
        return "Save required"
    if snapshot.investigation == "ready_to_investigate":
        return "Ready to investigate"
    if snapshot.screening == "data_ready" and snapshot.primary_action_enabled:
        return "Ready to screen"
    if snapshot.screening == "data_ready":
        return "Candidate needed"
    return "Review screening"


__all__ = [
    "ACTION_READY_LABEL",
    "LOCAL_FALLBACK_LABEL",
    "MODEL_ASSISTANCE_LABEL",
    "PREREQUISITE_REQUIRED_LABEL",
    "SAFETY_REMINDER",
    "WORKFLOW_PATH_LABEL",
    "WorkflowStatusStripText",
    "build_workflow_status_strip_text",
]
