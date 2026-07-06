"""Pure workflow-state derivation for the Streamlit UI.

This module accepts only UI-safe facts that are already known by the app. It
does not import Streamlit, read files, call models, or inspect configuration
secrets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ScreeningCheckpoint = Literal[
    "data_ready",
    "within_normal_band",
    "suspected_deviation",
]
PersistenceCheckpoint = Literal["not_applicable", "save_required", "saved"]
InvestigationCheckpoint = Literal[
    "not_available",
    "ready_to_investigate",
    "investigating",
    "investigation_available",
    "fallback_guidance_available",
]
PrimaryAction = Literal[
    "run_screening",
    "review_screening",
    "save_event",
    "investigate_with_copilot",
    "record_feedback",
    "continue_monitoring",
    "none",
]
ScreeningStatus = Literal["within_normal_band", "suspected_deviation"]
CopilotExecutionMode = Literal["adk", "deterministic_fallback"]
SaiaConfigurationStatus = Literal[
    "configured",
    "not_configured",
    "invalid_configuration",
]


@dataclass(frozen=True)
class WorkflowInputs:
    """Minimal UI-safe facts used to derive the displayed workflow state."""

    has_candidate_data: bool
    screening_status: ScreeningStatus | None
    current_result_saved: bool
    selected_saved_event_status: ScreeningStatus | None
    copilot_in_progress: bool
    copilot_execution_mode: CopilotExecutionMode | None
    has_copilot_result: bool
    human_review_acknowledged: bool
    saia_configuration_status: SaiaConfigurationStatus


@dataclass(frozen=True)
class WorkflowSnapshot:
    """Bounded user-facing workflow state for the UI layout layer."""

    screening: ScreeningCheckpoint
    persistence: PersistenceCheckpoint
    investigation: InvestigationCheckpoint
    primary_action: PrimaryAction
    primary_action_label: str
    short_instruction: str
    copilot_available: bool
    feedback_available: bool
    primary_action_enabled: bool
    feedback_recording_enabled: bool
    human_review_acknowledgement_required: bool
    model_assistance_available: bool


_ACTION_LABELS: dict[PrimaryAction, str] = {
    "run_screening": "Run deterministic screening",
    "review_screening": "Review screening result",
    "save_event": "Save event locally",
    "investigate_with_copilot": "Investigate with Copilot",
    "record_feedback": "Record feedback locally",
    "continue_monitoring": "Continue monitoring",
    "none": "No action available",
}


def derive_workflow_snapshot(inputs: WorkflowInputs) -> WorkflowSnapshot:
    """Derive a least-permissive workflow snapshot from UI-safe facts."""

    current_status = _known_screening_status(inputs.screening_status)
    selected_status = _known_screening_status(inputs.selected_saved_event_status)
    execution_mode = _known_execution_mode(inputs.copilot_execution_mode)
    configuration_status = _known_configuration_status(
        inputs.saia_configuration_status,
    )

    if selected_status == "within_normal_band":
        return _within_normal_band_snapshot()

    if selected_status == "suspected_deviation":
        return _derive_saved_suspected_snapshot(
            inputs=inputs,
            execution_mode=execution_mode,
            configuration_status=configuration_status,
        )

    if current_status == "within_normal_band":
        return _within_normal_band_snapshot()

    if current_status == "suspected_deviation":
        if not inputs.current_result_saved:
            return _snapshot(
                screening="suspected_deviation",
                persistence="save_required",
                investigation="not_available",
                primary_action="save_event",
                instruction=(
                    "Review deterministic evidence and save the suspected "
                    "deviation before investigation."
                ),
                primary_action_enabled=True,
            )
        return _snapshot(
            screening="suspected_deviation",
            persistence="saved",
            investigation="not_available",
            primary_action="review_screening",
            instruction=(
                "Select the saved suspected deviation before starting a "
                "Copilot investigation."
            ),
        )

    if inputs.has_candidate_data:
        return _data_ready_snapshot(has_candidate_data=True)

    return _data_ready_snapshot(has_candidate_data=False)


def _derive_saved_suspected_snapshot(
    *,
    inputs: WorkflowInputs,
    execution_mode: CopilotExecutionMode | None,
    configuration_status: SaiaConfigurationStatus,
) -> WorkflowSnapshot:
    if inputs.copilot_in_progress:
        return _snapshot(
            screening="suspected_deviation",
            persistence="saved",
            investigation="investigating",
            primary_action="none",
            instruction=(
                "Copilot is reviewing saved screening evidence and approved "
                "guidance. No automatic machine action is being taken."
            ),
            copilot_available=True,
            model_assistance_available=configuration_status == "configured",
        )

    if inputs.has_copilot_result:
        feedback_recording_enabled = inputs.human_review_acknowledged
        acknowledgement_required = not inputs.human_review_acknowledged
        if execution_mode == "adk":
            instruction = (
                "Guidance supports human review and is not a confirmed "
                "diagnosis. Feedback can now be recorded locally after review."
            )
            if acknowledgement_required:
                instruction = (
                    "Guidance supports human review and is not a confirmed "
                    "diagnosis. Acknowledge human review before recording "
                    "local feedback."
                )
            return _snapshot(
                screening="suspected_deviation",
                persistence="saved",
                investigation="investigation_available",
                primary_action="record_feedback",
                instruction=instruction,
                copilot_available=True,
                feedback_available=True,
                primary_action_enabled=feedback_recording_enabled,
                feedback_recording_enabled=feedback_recording_enabled,
                human_review_acknowledgement_required=acknowledgement_required,
                model_assistance_available=configuration_status == "configured",
            )
        if execution_mode == "deterministic_fallback":
            instruction = (
                "Local deterministic guidance is available and human review "
                "remains required. Feedback can now be recorded locally after "
                "review."
            )
            if acknowledgement_required:
                instruction = (
                    "Local deterministic guidance is available and human "
                    "review remains required. Acknowledge human review before "
                    "recording local feedback."
                )
            return _snapshot(
                screening="suspected_deviation",
                persistence="saved",
                investigation="fallback_guidance_available",
                primary_action="record_feedback",
                instruction=instruction,
                copilot_available=True,
                feedback_available=True,
                primary_action_enabled=feedback_recording_enabled,
                feedback_recording_enabled=feedback_recording_enabled,
                human_review_acknowledgement_required=acknowledgement_required,
                model_assistance_available=False,
            )
        return _snapshot(
            screening="suspected_deviation",
            persistence="saved",
            investigation="not_available",
            primary_action="review_screening",
            instruction=(
                "Review the saved suspected deviation before starting another "
                "Copilot investigation."
            ),
        )

    if execution_mode is not None:
        return _snapshot(
            screening="suspected_deviation",
            persistence="saved",
            investigation="not_available",
            primary_action="review_screening",
            instruction=(
                "Review the saved suspected deviation before starting another "
                "Copilot investigation."
            ),
        )

    instruction = (
        "Investigate the saved suspected deviation. Local fallback guidance "
        "remains available if model assistance is unavailable."
    )
    return _snapshot(
        screening="suspected_deviation",
        persistence="saved",
        investigation="ready_to_investigate",
        primary_action="investigate_with_copilot",
        instruction=instruction,
        copilot_available=True,
        primary_action_enabled=True,
        model_assistance_available=configuration_status == "configured",
    )


def _data_ready_snapshot(*, has_candidate_data: bool) -> WorkflowSnapshot:
    instruction = (
        "Prepare a candidate cycle and run deterministic screening."
        if has_candidate_data
        else (
            "Select or upload a candidate cycle before running deterministic "
            "screening."
        )
    )
    return _snapshot(
        screening="data_ready",
        persistence="not_applicable",
        investigation="not_available",
        primary_action="run_screening",
        instruction=instruction,
        primary_action_enabled=has_candidate_data,
    )


def _within_normal_band_snapshot() -> WorkflowSnapshot:
    return _snapshot(
        screening="within_normal_band",
        persistence="not_applicable",
        investigation="not_available",
        primary_action="continue_monitoring",
        instruction=(
            "The cycle is within the calibrated normal band. Copilot "
            "investigation is not needed."
        ),
    )


def _snapshot(
    *,
    screening: ScreeningCheckpoint,
    persistence: PersistenceCheckpoint,
    investigation: InvestigationCheckpoint,
    primary_action: PrimaryAction,
    instruction: str,
    copilot_available: bool = False,
    feedback_available: bool = False,
    primary_action_enabled: bool | None = None,
    feedback_recording_enabled: bool = False,
    human_review_acknowledgement_required: bool = False,
    model_assistance_available: bool = False,
) -> WorkflowSnapshot:
    action_enabled = False if primary_action_enabled is None else primary_action_enabled
    return WorkflowSnapshot(
        screening=screening,
        persistence=persistence,
        investigation=investigation,
        primary_action=primary_action,
        primary_action_label=_ACTION_LABELS[primary_action],
        short_instruction=instruction,
        copilot_available=copilot_available,
        feedback_available=feedback_available,
        primary_action_enabled=action_enabled,
        feedback_recording_enabled=feedback_recording_enabled,
        human_review_acknowledgement_required=human_review_acknowledgement_required,
        model_assistance_available=model_assistance_available,
    )


def _known_screening_status(value: object) -> ScreeningStatus | None:
    if value in ("within_normal_band", "suspected_deviation"):
        return value  # type: ignore[return-value]
    return None


def _known_execution_mode(value: object) -> CopilotExecutionMode | None:
    if value in ("adk", "deterministic_fallback"):
        return value  # type: ignore[return-value]
    return None


def _known_configuration_status(value: object) -> SaiaConfigurationStatus:
    if value in ("configured", "not_configured", "invalid_configuration"):
        return value  # type: ignore[return-value]
    return "invalid_configuration"


__all__ = [
    "CopilotExecutionMode",
    "InvestigationCheckpoint",
    "PersistenceCheckpoint",
    "PrimaryAction",
    "SaiaConfigurationStatus",
    "ScreeningCheckpoint",
    "ScreeningStatus",
    "WorkflowInputs",
    "WorkflowSnapshot",
    "derive_workflow_snapshot",
]
