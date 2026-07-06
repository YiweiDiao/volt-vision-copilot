from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from volt_vision.ui.workflow import (
    ScreeningCheckpoint,
    WorkflowInputs,
    WorkflowSnapshot,
    derive_workflow_snapshot,
)


def make_inputs(**overrides: object) -> WorkflowInputs:
    values = {
        "has_candidate_data": False,
        "screening_status": None,
        "current_result_saved": False,
        "selected_saved_event_status": None,
        "copilot_in_progress": False,
        "copilot_execution_mode": None,
        "has_copilot_result": False,
        "human_review_acknowledged": False,
        "saia_configuration_status": "not_configured",
    }
    values.update(overrides)
    return WorkflowInputs(**values)  # type: ignore[arg-type]


def test_no_candidate_no_result_prompts_screening() -> None:
    snapshot = derive_workflow_snapshot(make_inputs())

    assert snapshot.screening == "data_ready"
    assert snapshot.persistence == "not_applicable"
    assert snapshot.investigation == "not_available"
    assert snapshot.primary_action == "run_screening"
    assert snapshot.primary_action_enabled is False
    assert snapshot.copilot_available is False
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False
    assert "Select or upload a candidate cycle" in snapshot.short_instruction


def test_candidate_ready_without_result_prompts_screening() -> None:
    snapshot = derive_workflow_snapshot(make_inputs(has_candidate_data=True))

    assert snapshot.screening == "data_ready"
    assert snapshot.primary_action == "run_screening"
    assert snapshot.primary_action_enabled is True
    assert snapshot.copilot_available is False
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False


def test_screening_checkpoint_has_no_unreachable_screening_complete_literal() -> None:
    assert "screening_complete" not in get_args(ScreeningCheckpoint)


def test_within_normal_band_disables_copilot() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            has_candidate_data=True,
            screening_status="within_normal_band",
        )
    )

    assert snapshot.screening == "within_normal_band"
    assert snapshot.persistence == "not_applicable"
    assert snapshot.investigation == "not_available"
    assert snapshot.primary_action == "continue_monitoring"
    assert snapshot.primary_action_enabled is False
    assert snapshot.copilot_available is False
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False
    assert "within the calibrated normal band" in snapshot.short_instruction
    assert "not needed" in snapshot.short_instruction


def test_suspected_deviation_unsaved_requires_save() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            has_candidate_data=True,
            screening_status="suspected_deviation",
            current_result_saved=False,
        )
    )

    assert snapshot.screening == "suspected_deviation"
    assert snapshot.persistence == "save_required"
    assert snapshot.investigation == "not_available"
    assert snapshot.primary_action == "save_event"
    assert snapshot.primary_action_enabled is True
    assert snapshot.copilot_available is False
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False
    assert "save the suspected deviation" in snapshot.short_instruction


def test_saved_suspected_deviation_with_configured_model_is_ready() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            current_result_saved=True,
            selected_saved_event_status="suspected_deviation",
            saia_configuration_status="configured",
        )
    )

    assert snapshot.screening == "suspected_deviation"
    assert snapshot.persistence == "saved"
    assert snapshot.investigation == "ready_to_investigate"
    assert snapshot.primary_action == "investigate_with_copilot"
    assert snapshot.primary_action_enabled is True
    assert snapshot.copilot_available is True
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False
    assert snapshot.model_assistance_available is True
    assert "Local fallback guidance remains available" in snapshot.short_instruction


def test_saved_suspected_deviation_without_configured_model_keeps_fallback_ready() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            current_result_saved=True,
            selected_saved_event_status="suspected_deviation",
            saia_configuration_status="not_configured",
        )
    )

    assert snapshot.investigation == "ready_to_investigate"
    assert snapshot.primary_action == "investigate_with_copilot"
    assert snapshot.primary_action_enabled is True
    assert snapshot.copilot_available is True
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False
    assert snapshot.model_assistance_available is False
    assert "fallback guidance" in snapshot.short_instruction


def test_copilot_investigating_disables_feedback() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            copilot_in_progress=True,
            saia_configuration_status="configured",
        )
    )

    assert snapshot.investigation == "investigating"
    assert snapshot.primary_action == "none"
    assert snapshot.primary_action_enabled is False
    assert snapshot.copilot_available is True
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False
    assert snapshot.model_assistance_available is True
    assert "No automatic machine action" in snapshot.short_instruction


def test_accepted_adk_guidance_without_acknowledgement_requires_review_ack() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode="adk",
            human_review_acknowledged=False,
            saia_configuration_status="configured",
        )
    )

    assert snapshot.investigation == "investigation_available"
    assert snapshot.primary_action == "record_feedback"
    assert snapshot.primary_action_enabled is False
    assert snapshot.copilot_available is True
    assert snapshot.feedback_available is True
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is True
    assert snapshot.model_assistance_available is True
    assert "not a confirmed diagnosis" in snapshot.short_instruction
    assert "Acknowledge human review" in snapshot.short_instruction


def test_accepted_adk_guidance_with_acknowledgement_enables_feedback_recording() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode="adk",
            human_review_acknowledged=True,
            saia_configuration_status="configured",
        )
    )

    assert snapshot.investigation == "investigation_available"
    assert snapshot.primary_action == "record_feedback"
    assert snapshot.primary_action_enabled is True
    assert snapshot.copilot_available is True
    assert snapshot.feedback_available is True
    assert snapshot.feedback_recording_enabled is True
    assert snapshot.human_review_acknowledgement_required is False
    assert snapshot.model_assistance_available is True
    assert "not a confirmed diagnosis" in snapshot.short_instruction
    assert "Feedback can now be recorded locally" in snapshot.short_instruction


def test_deterministic_fallback_without_acknowledgement_requires_review_ack() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode="deterministic_fallback",
            human_review_acknowledged=False,
            saia_configuration_status="configured",
        )
    )

    assert snapshot.investigation == "fallback_guidance_available"
    assert snapshot.primary_action == "record_feedback"
    assert snapshot.primary_action_enabled is False
    assert snapshot.copilot_available is True
    assert snapshot.feedback_available is True
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is True
    assert snapshot.model_assistance_available is False
    assert "Local deterministic guidance" in snapshot.short_instruction
    assert "Acknowledge human review" in snapshot.short_instruction


def test_deterministic_fallback_with_acknowledgement_enables_feedback_recording() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode="deterministic_fallback",
            human_review_acknowledged=True,
            saia_configuration_status="configured",
        )
    )

    assert snapshot.investigation == "fallback_guidance_available"
    assert snapshot.primary_action == "record_feedback"
    assert snapshot.primary_action_enabled is True
    assert snapshot.copilot_available is True
    assert snapshot.feedback_available is True
    assert snapshot.feedback_recording_enabled is True
    assert snapshot.human_review_acknowledgement_required is False
    assert snapshot.model_assistance_available is False
    assert "Local deterministic guidance" in snapshot.short_instruction
    assert "Feedback can now be recorded locally" in snapshot.short_instruction


def test_no_copilot_result_despite_acknowledgement_remains_feedback_ineligible() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=False,
            human_review_acknowledged=True,
        )
    )

    assert snapshot.investigation == "ready_to_investigate"
    assert snapshot.primary_action == "investigate_with_copilot"
    assert snapshot.primary_action_enabled is True
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False


def test_selected_saved_normal_event_is_never_eligible() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            selected_saved_event_status="within_normal_band",
            copilot_in_progress=True,
            has_copilot_result=True,
            copilot_execution_mode="adk",
            human_review_acknowledged=True,
            saia_configuration_status="configured",
        )
    )

    assert snapshot.screening == "within_normal_band"
    assert snapshot.investigation == "not_available"
    assert snapshot.primary_action == "continue_monitoring"
    assert snapshot.primary_action_enabled is False
    assert snapshot.copilot_available is False
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False
    assert snapshot.model_assistance_available is False


def test_saved_but_not_selected_review_screening_is_disabled() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            screening_status="suspected_deviation",
            current_result_saved=True,
        )
    )

    assert snapshot.persistence == "saved"
    assert snapshot.investigation == "not_available"
    assert snapshot.primary_action == "review_screening"
    assert snapshot.primary_action_enabled is False
    assert snapshot.feedback_available is False


@pytest.mark.parametrize(
    "inputs",
    [
        make_inputs(
            screening_status="suspected_deviation",
            current_result_saved=False,
            has_copilot_result=True,
            copilot_execution_mode="adk",
        ),
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode=None,
        ),
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=False,
            copilot_execution_mode="adk",
        ),
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode="unexpected",
            saia_configuration_status="unexpected",
        ),
    ],
)
def test_impossible_combinations_fail_closed(inputs: WorkflowInputs) -> None:
    snapshot = derive_workflow_snapshot(inputs)

    assert snapshot.copilot_available is False or snapshot.feedback_available is False
    assert snapshot.feedback_available is False
    assert snapshot.feedback_recording_enabled is False
    assert snapshot.human_review_acknowledgement_required is False
    assert snapshot.investigation != "investigation_available"
    assert snapshot.investigation != "fallback_guidance_available"
    assert snapshot.primary_action in {
        "review_screening",
        "save_event",
        "continue_monitoring",
    }


def test_every_snapshot_is_immutable() -> None:
    snapshot = derive_workflow_snapshot(make_inputs())

    with pytest.raises(FrozenInstanceError):
        snapshot.primary_action = "none"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        inputs = make_inputs()
        inputs.has_candidate_data = True  # type: ignore[misc]


def test_primary_action_none_is_never_enabled() -> None:
    snapshot = derive_workflow_snapshot(
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            copilot_in_progress=True,
        )
    )

    assert snapshot.primary_action == "none"
    assert snapshot.primary_action_enabled is False


def test_user_facing_labels_and_instructions_are_bounded() -> None:
    snapshots = [
        derive_workflow_snapshot(make_inputs()),
        derive_workflow_snapshot(make_inputs(has_candidate_data=True)),
        derive_workflow_snapshot(
            make_inputs(screening_status="within_normal_band")
        ),
        derive_workflow_snapshot(
            make_inputs(screening_status="suspected_deviation")
        ),
        derive_workflow_snapshot(
            make_inputs(selected_saved_event_status="suspected_deviation")
        ),
        derive_workflow_snapshot(
            make_inputs(
                selected_saved_event_status="suspected_deviation",
                copilot_in_progress=True,
            )
        ),
        derive_workflow_snapshot(
            make_inputs(
                selected_saved_event_status="suspected_deviation",
                has_copilot_result=True,
                copilot_execution_mode="adk",
            )
        ),
        derive_workflow_snapshot(
            make_inputs(
                selected_saved_event_status="suspected_deviation",
                has_copilot_result=True,
                copilot_execution_mode="deterministic_fallback",
            )
        ),
    ]
    forbidden_fragments = (
        "saia",
        "adk",
        "mcp",
        "api key",
        "api_key",
        "secret",
        "token",
        "traceback",
        "exception",
        "stack",
        "payload",
        "prompt",
        "tool_name",
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
        "event_id",
        "d:\\",
        "/tmp/",
    )

    for snapshot in snapshots:
        assert isinstance(snapshot, WorkflowSnapshot)
        assert snapshot.primary_action_label
        assert snapshot.short_instruction
        combined = (
            f"{snapshot.primary_action_label} {snapshot.short_instruction}".lower()
        )
        for fragment in forbidden_fragments:
            assert fragment not in combined
