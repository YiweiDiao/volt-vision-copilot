from __future__ import annotations

import pytest

from volt_vision.ui.workflow import WorkflowInputs, derive_workflow_snapshot
from volt_vision.ui.workflow_status_strip import (
    LOCAL_FALLBACK_LABEL,
    MODEL_ASSISTANCE_LABEL,
    SAFETY_REMINDER,
    WORKFLOW_PATH_LABEL,
    WorkflowStatusStripText,
    build_workflow_status_strip_text,
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


@pytest.mark.parametrize(
    ("inputs", "state_label", "assistance_label"),
    [
        (make_inputs(), "Candidate needed", LOCAL_FALLBACK_LABEL),
        (
            make_inputs(has_candidate_data=True),
            "Ready to screen",
            LOCAL_FALLBACK_LABEL,
        ),
        (
            make_inputs(
                has_candidate_data=True,
                screening_status="within_normal_band",
            ),
            "Within normal band",
            LOCAL_FALLBACK_LABEL,
        ),
        (
            make_inputs(
                has_candidate_data=True,
                screening_status="suspected_deviation",
                current_result_saved=False,
            ),
            "Save required",
            LOCAL_FALLBACK_LABEL,
        ),
        (
            make_inputs(
                current_result_saved=True,
                selected_saved_event_status="suspected_deviation",
                saia_configuration_status="configured",
            ),
            "Ready to investigate",
            MODEL_ASSISTANCE_LABEL,
        ),
        (
            make_inputs(
                selected_saved_event_status="suspected_deviation",
                copilot_in_progress=True,
                saia_configuration_status="configured",
            ),
            "Copilot investigating",
            MODEL_ASSISTANCE_LABEL,
        ),
        (
            make_inputs(
                selected_saved_event_status="suspected_deviation",
                has_copilot_result=True,
                copilot_execution_mode="adk",
                human_review_acknowledged=True,
                saia_configuration_status="configured",
            ),
            "Guidance available",
            MODEL_ASSISTANCE_LABEL,
        ),
        (
            make_inputs(
                selected_saved_event_status="suspected_deviation",
                has_copilot_result=True,
                copilot_execution_mode="deterministic_fallback",
                human_review_acknowledged=True,
                saia_configuration_status="configured",
            ),
            "Fallback guidance available",
            LOCAL_FALLBACK_LABEL,
        ),
        (
            make_inputs(
                selected_saved_event_status="suspected_deviation",
                has_copilot_result=True,
                copilot_execution_mode="adk",
                human_review_acknowledged=False,
                saia_configuration_status="configured",
            ),
            "Guidance available",
            MODEL_ASSISTANCE_LABEL,
        ),
    ],
)
def test_status_strip_text_for_representative_snapshots(
    inputs: WorkflowInputs,
    state_label: str,
    assistance_label: str,
) -> None:
    snapshot = derive_workflow_snapshot(inputs)

    text = build_workflow_status_strip_text(snapshot)

    assert isinstance(text, WorkflowStatusStripText)
    assert text.path_label == WORKFLOW_PATH_LABEL
    assert text.current_state_label == state_label
    assert text.primary_action_label == snapshot.primary_action_label
    assert text.short_instruction == snapshot.short_instruction
    assert text.assistance_availability_label == assistance_label
    assert text.safety_reminder == SAFETY_REMINDER


def test_disabled_primary_action_mentions_prerequisite_without_fake_button() -> None:
    snapshot = derive_workflow_snapshot(make_inputs())

    text = build_workflow_status_strip_text(snapshot)

    assert snapshot.primary_action_enabled is False
    assert "Complete the prerequisite" in text.action_availability_label


def test_status_strip_text_does_not_expose_provider_or_model_names() -> None:
    texts = _representative_status_texts()
    forbidden_fragments = (
        "saia",
        "adk",
        "mcp",
        "qwen",
        "gemini",
        "gpt",
        "litellm",
        "provider",
        "configured-model",
        "deterministic_fallback",
    )

    for text in texts:
        combined = _combined_text(text).lower()
        for fragment in forbidden_fragments:
            assert fragment not in combined


def test_status_strip_text_does_not_expose_sensitive_or_internal_details() -> None:
    texts = _representative_status_texts()
    forbidden_fragments = (
        "evt-123",
        "event_id",
        "tool",
        "get_event_metrics",
        "retrieve_maintenance_guidance",
        "find_similar_previous_events",
        "secret",
        "api key",
        "api_key",
        "token",
        "password",
        "error",
        "exception",
        "traceback",
        "d:\\",
        "c:\\",
        "/tmp/",
        ".env",
        "raw",
        "metric",
        "prompt",
    )

    for text in texts:
        combined = _combined_text(text).lower()
        for fragment in forbidden_fragments:
            assert fragment not in combined


def _representative_status_texts() -> list[WorkflowStatusStripText]:
    inputs = [
        make_inputs(),
        make_inputs(has_candidate_data=True),
        make_inputs(
            has_candidate_data=True,
            screening_status="within_normal_band",
        ),
        make_inputs(
            has_candidate_data=True,
            screening_status="suspected_deviation",
            current_result_saved=False,
        ),
        make_inputs(
            current_result_saved=True,
            selected_saved_event_status="suspected_deviation",
            saia_configuration_status="configured",
        ),
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            copilot_in_progress=True,
            saia_configuration_status="configured",
        ),
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode="adk",
            human_review_acknowledged=True,
            saia_configuration_status="configured",
        ),
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode="deterministic_fallback",
            human_review_acknowledged=True,
            saia_configuration_status="configured",
        ),
        make_inputs(
            selected_saved_event_status="suspected_deviation",
            has_copilot_result=True,
            copilot_execution_mode="adk",
            human_review_acknowledged=False,
            saia_configuration_status="configured",
        ),
    ]
    return [
        build_workflow_status_strip_text(derive_workflow_snapshot(item))
        for item in inputs
    ]


def _combined_text(text: WorkflowStatusStripText) -> str:
    return " ".join(
        (
            text.path_label,
            text.current_state_label,
            text.primary_action_label,
            text.short_instruction,
            text.action_availability_label,
            text.assistance_availability_label,
            text.safety_reminder,
        )
    )
