from __future__ import annotations

from pathlib import Path

import pytest

from volt_vision.agent.models import AgentRunTrace, CopilotChatResponse, ToolCallTrace
from volt_vision.agent.policy import MANDATORY_CHAT_HEADINGS
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.ui import app as app_module
from volt_vision.ui.copilot import build_copilot_feedback_record, can_record_feedback

from test_mcp_services import make_event


class FakeModel:
    secret = "secret-model-object"


def valid_chat_text() -> str:
    return "\n".join(
        (
            MANDATORY_CHAT_HEADINGS[0],
            "This is a suspected deviation, not a confirmed diagnosis. Manual inspection recommended.",
            "",
            MANDATORY_CHAT_HEADINGS[1],
            "Possible contributing condition: verify context and compare reviewed evidence.",
            "",
            MANDATORY_CHAT_HEADINGS[2],
            "Inspect according to local SOP and record observations.",
            "",
            MANDATORY_CHAT_HEADINGS[3],
            "Was the selected cycle complete and comparable?",
            "",
            MANDATORY_CHAT_HEADINGS[4],
            "Escalate when recurrence, production impact, or local procedure requires review.",
        )
    )


def make_chat_response(
    event_id: str = "query",
    *,
    execution_mode: str = "adk",
    knowledge_source_ids: tuple[str, ...] = ("power_signature_review",),
) -> CopilotChatResponse:
    source = "mcp" if execution_mode == "adk" else "deterministic_service"
    fallback_reason = None if execution_mode == "adk" else "model_not_configured"
    tool_calls = (
        ToolCallTrace(
            tool_name="get_event_metrics",
            source=source,
            outcome="succeeded",
            error_code=None,
        ),
        ToolCallTrace(
            tool_name="retrieve_maintenance_guidance",
            source=source,
            outcome="succeeded",
            error_code=None,
        ),
        ToolCallTrace(
            tool_name="find_similar_previous_events",
            source=source,
            outcome="succeeded",
            error_code=None,
        ),
    )
    trace = AgentRunTrace(
        event_id=event_id,
        execution_mode=execution_mode,
        tool_names=tuple(call.tool_name for call in tool_calls),
        tool_calls=tool_calls,
        fallback_reason=fallback_reason,
        completed=True,
    )
    return CopilotChatResponse(
        event_id=event_id,
        execution_mode=execution_mode,
        assistant_message=valid_chat_text(),
        knowledge_source_ids=knowledge_source_ids,
        tool_trace=trace,
        human_approval_required=True,
        fallback_reason=fallback_reason,
    )


def make_rendered_result(event_id: str = "query", *, status: str = "suspected_deviation") -> app_module.RenderedMonitoringResult:
    event = make_event(event_id, seconds=60, status=status)
    return app_module.RenderedMonitoringResult(
        mode="synthetic",
        event=event,
        calibration_cycle_count=3,
        reference_cycle=object(),  # type: ignore[arg-type]
        candidate_cycle=object(),  # type: ignore[arg-type]
        reference_cycle_id="reference",
        threshold_result=object(),  # type: ignore[arg-type]
    )


def test_configured_model_resolution_is_lazy_until_investigation_action(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def resolver() -> object:
        calls.append("resolver")
        return FakeModel()

    def runner(*_: object, **kwargs: object) -> CopilotChatResponse:
        calls.append("runner")
        assert isinstance(kwargs["model"], FakeModel)
        return make_chat_response()

    assert calls == []

    result = app_module.run_copilot_chat_for_ui(
        "query",
        history_path=tmp_path / "history.jsonl",
        configuration_status_reader=lambda: "configured",
        model_resolver=resolver,
        chat_runner=runner,
    )

    assert result.execution_mode == "adk"
    assert calls == ["resolver", "runner"]


@pytest.mark.parametrize("status", ["not_configured", "invalid_configuration"])
def test_unconfigured_or_invalid_saia_uses_public_chat_with_model_none(
    tmp_path: Path,
    status: str,
) -> None:
    calls: list[object | None] = []

    def runner(*_: object, **kwargs: object) -> CopilotChatResponse:
        calls.append(kwargs["model"])
        return make_chat_response(execution_mode="deterministic_fallback")

    result = app_module.run_copilot_chat_for_ui(
        "query",
        history_path=tmp_path / "history.jsonl",
        configuration_status_reader=lambda: status,
        model_resolver=lambda: pytest.fail("resolver must stay lazy"),
        chat_runner=runner,
    )

    assert result.execution_mode == "deterministic_fallback"
    assert calls == [None]


def test_resolver_failure_safely_falls_back_with_model_none(tmp_path: Path) -> None:
    calls: list[object | None] = []

    def runner(*_: object, **kwargs: object) -> CopilotChatResponse:
        calls.append(kwargs["model"])
        return make_chat_response(execution_mode="deterministic_fallback")

    result = app_module.run_copilot_chat_for_ui(
        "query",
        history_path=tmp_path / "history.jsonl",
        configuration_status_reader=lambda: "configured",
        model_resolver=lambda: (_ for _ in ()).throw(
            RuntimeError("secret resolver failure")
        ),
        chat_runner=runner,
    )

    assert result.execution_mode == "deterministic_fallback"
    assert calls == [None]
    assert "secret resolver failure" not in result.model_dump_json()


def test_no_model_is_resolved_during_import_or_render_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    monkeypatch.setattr(
        app_module,
        "resolve_optional_saia_model_from_env",
        lambda: pytest.fail("model resolver must not run during preparation"),
    )
    monkeypatch.setattr(
        app_module,
        "safe_saia_configuration_status",
        lambda: "configured",
    )

    inputs = app_module.build_workflow_inputs_from_app_state(
        event_log_path=tmp_path / "missing.jsonl",
    )

    assert inputs.has_candidate_data is False
    assert inputs.saia_configuration_status == "configured"


def test_status_strip_is_populated_after_mode_workflow_establishes_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    monkeypatch.setattr(
        app_module.st,
        "set_page_config",
        lambda **_: calls.append("page_config"),
    )
    monkeypatch.setattr(app_module, "render_header", lambda: calls.append("header"))
    monkeypatch.setattr(
        app_module.st,
        "empty",
        lambda: calls.append("placeholder") or object(),
    )
    monkeypatch.setattr(
        app_module.st,
        "radio",
        lambda *_args, **_kwargs: calls.append("mode_controls") or "Synthetic demo",
    )

    def fake_synthetic_mode() -> None:
        calls.append("mode_workflow")
        app_module.st.session_state[app_module.SESSION_RESULT_KEY] = make_rendered_result(
            "current",
            status="suspected_deviation",
        )

    def fake_status_strip(**kwargs: object) -> None:
        calls.append("status_strip")
        assert kwargs["placeholder"] is not None
        inputs = app_module.build_workflow_inputs_from_app_state(
            event_log_path=Path("unused.jsonl"),
        )
        assert inputs.screening_status == "suspected_deviation"

    monkeypatch.setattr(app_module, "render_synthetic_demo_mode", fake_synthetic_mode)
    monkeypatch.setattr(app_module, "render_workflow_status_strip", fake_status_strip)

    app_module.main()

    assert calls == [
        "page_config",
        "header",
        "placeholder",
        "mode_controls",
        "mode_workflow",
        "status_strip",
    ]


def test_synthetic_candidate_change_uses_current_result_not_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    app_module.st.session_state[app_module.SESSION_RESULT_KEY] = make_rendered_result(
        "prior",
        status="within_normal_band",
    )
    app_module.st.session_state[app_module.SESSION_RESULT_KEY] = make_rendered_result(
        "current",
        status="suspected_deviation",
    )

    inputs = app_module.build_workflow_inputs_from_app_state(
        event_log_path=tmp_path / "events.jsonl",
    )

    assert inputs.screening_status == "suspected_deviation"


def test_selected_saved_suspected_deviation_can_receive_adk_result(
    tmp_path: Path,
) -> None:
    result = app_module.run_copilot_chat_for_ui(
        "query",
        history_path=tmp_path / "history.jsonl",
        configuration_status_reader=lambda: "configured",
        model_resolver=lambda: FakeModel(),
        chat_runner=lambda *_args, **_kwargs: make_chat_response("query"),
    )

    assert result.execution_mode == "adk"
    assert result.event_id == "query"


def test_within_normal_band_never_starts_copilot_or_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    event = make_event("normal", seconds=60, status="within_normal_band")

    assert app_module._can_start_copilot_investigation(event) is False


def test_model_run_failure_falls_back_without_exception_text_and_clears_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    calls: list[object | None] = []

    def runner(*_: object, **kwargs: object) -> CopilotChatResponse:
        model = kwargs["model"]
        calls.append(model)
        if model is not None:
            raise RuntimeError("secret exception text")
        return make_chat_response(execution_mode="deterministic_fallback")

    app_module._set_copilot_in_progress("query")
    try:
        result = app_module.run_copilot_chat_for_ui(
            "query",
            history_path=tmp_path / "history.jsonl",
            configuration_status_reader=lambda: "configured",
            model_resolver=lambda: FakeModel(),
            chat_runner=runner,
        )
    finally:
        app_module._clear_copilot_in_progress("query")

    assert result.execution_mode == "deterministic_fallback"
    assert "secret exception text" not in result.model_dump_json().lower()
    assert app_module._is_copilot_in_progress("query") is False
    assert isinstance(calls[0], FakeModel)
    assert calls[1] is None


def test_duplicate_in_progress_event_does_not_start_second_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    event = make_event("query", seconds=60)

    assert app_module._can_start_copilot_investigation(event) is True
    app_module._set_copilot_in_progress(event.event_id)

    assert app_module._can_start_copilot_investigation(event) is False


def test_result_storage_contains_only_public_response_not_model_or_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    response = make_chat_response("query")

    app_module._copilot_result_store()["query"] = response

    stored = app_module.st.session_state[app_module.COPILOT_RESULTS_KEY]["query"]
    assert stored == response
    payload = stored.model_dump_json().lower()
    assert "secret-model-object" not in payload
    assert "api_key" not in payload
    assert "prompt" not in payload


@pytest.mark.parametrize("mode", ["adk", "deterministic_fallback"])
def test_chat_results_remain_usable_by_feedback_logic(mode: str) -> None:
    event = make_event("query", seconds=60)
    response = make_chat_response("query", execution_mode=mode)

    assert can_record_feedback(
        event,
        response,
        human_review_acknowledged=True,
    )


def test_layout_context_does_not_save_event_implicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    log_path = tmp_path / "events.jsonl"

    context = app_module.build_investigation_workspace_context(
        result=make_rendered_result("query"),
        event_log_path=log_path,
    )

    assert context.current_result_saved is False
    assert not log_path.exists()


def test_successful_save_appends_once_and_selects_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    event = make_event("query", seconds=60)
    calls: list[str] = []

    def fake_append(appended_event: object, log_path: Path) -> None:
        calls.append(str(log_path.name))
        assert appended_event == event

    app_module.save_event_for_investigation(
        event,
        tmp_path / "events.jsonl",
        append_event=fake_append,
    )

    assert calls == ["events.jsonl"]
    assert app_module.st.session_state[app_module.COPILOT_EVENT_SELECTION_KEY] == "query"
    assert (
        app_module.st.session_state[app_module.LAST_SAVED_EVENT_CONFIRMATION_KEY]
        == "query"
    )


def test_explicit_save_selects_saved_suspected_deviation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    log_path = tmp_path / "events.jsonl"
    rendered = make_rendered_result("query")
    append_monitoring_event(rendered.event, log_path)

    app_module.select_saved_event_for_investigation(rendered.event)
    context = app_module.build_investigation_workspace_context(
        result=rendered,
        event_log_path=log_path,
    )

    assert context.current_result_saved is True
    assert context.selected_event == rendered.event
    assert app_module.st.session_state[app_module.COPILOT_EVENT_SELECTION_KEY] == "query"


def test_next_render_after_save_sees_saved_event_as_copilot_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    log_path = tmp_path / "events.jsonl"
    rendered = make_rendered_result("query")
    append_monitoring_event(rendered.event, log_path)
    app_module.save_event_for_investigation(rendered.event, log_path)

    context = app_module.build_investigation_workspace_context(
        result=rendered,
        event_log_path=log_path,
    )
    text = app_module.build_copilot_panel_text(context.workflow_snapshot)

    assert context.current_result_saved is True
    assert context.selected_event == rendered.event
    assert text.target_label == "Saved suspected deviation selected"
    assert text.eligibility_label == "Ready to investigate the saved suspected deviation."
    assert app_module._can_start_copilot_investigation(context.selected_event)


def test_save_does_not_auto_invoke_copilot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    monkeypatch.setattr(
        app_module,
        "run_copilot_chat_for_ui",
        lambda *_args, **_kwargs: pytest.fail("save must not run Copilot"),
    )

    app_module.save_event_for_investigation(
        make_event("query", seconds=60),
        tmp_path / "events.jsonl",
    )


def test_state_helpers_do_not_resolve_model_or_call_live_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(app_module.st, "session_state", {}, raising=False)
    monkeypatch.setattr(
        app_module,
        "resolve_optional_saia_model_from_env",
        lambda: pytest.fail("model resolver must not be called"),
    )
    monkeypatch.setattr(
        app_module,
        "run_copilot_chat_for_ui",
        lambda *_args, **_kwargs: pytest.fail("Copilot must not be called"),
    )
    rendered = make_rendered_result("query")

    context = app_module.build_investigation_workspace_context(
        result=rendered,
        event_log_path=tmp_path / "events.jsonl",
    )

    assert context.current_result_saved is False


def test_within_normal_band_panel_text_is_not_copilot_eligible() -> None:
    snapshot = app_module.derive_workflow_snapshot(
        app_module.WorkflowInputs(
            has_candidate_data=True,
            screening_status="within_normal_band",
            current_result_saved=False,
            selected_saved_event_status=None,
            copilot_in_progress=False,
            copilot_execution_mode=None,
            has_copilot_result=False,
            human_review_acknowledged=False,
            saia_configuration_status="configured",
        )
    )

    text = app_module.build_copilot_panel_text(snapshot)

    assert "not available" in text.eligibility_label
    assert "within the normal band" in text.eligibility_label


def test_unsaved_suspected_deviation_panel_text_requires_save() -> None:
    snapshot = app_module.derive_workflow_snapshot(
        app_module.WorkflowInputs(
            has_candidate_data=True,
            screening_status="suspected_deviation",
            current_result_saved=False,
            selected_saved_event_status=None,
            copilot_in_progress=False,
            copilot_execution_mode=None,
            has_copilot_result=False,
            human_review_acknowledged=False,
            saia_configuration_status="configured",
        )
    )

    text = app_module.build_copilot_panel_text(snapshot)

    assert "Save the suspected deviation" in text.eligibility_label
    assert "save control" in text.action_label


def test_saved_suspected_deviation_panel_text_is_ready() -> None:
    snapshot = app_module.derive_workflow_snapshot(
        app_module.WorkflowInputs(
            has_candidate_data=True,
            screening_status="suspected_deviation",
            current_result_saved=True,
            selected_saved_event_status="suspected_deviation",
            copilot_in_progress=False,
            copilot_execution_mode=None,
            has_copilot_result=False,
            human_review_acknowledged=False,
            saia_configuration_status="configured",
        )
    )

    text = app_module.build_copilot_panel_text(snapshot)

    assert text.eligibility_label == "Ready to investigate the saved suspected deviation."
    assert text.action_label == "Use the Investigate with Copilot button when ready."


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("adk", "Model-assisted guidance"),
        ("deterministic_fallback", "Local deterministic fallback guidance"),
    ],
)
def test_guidance_type_labels_are_bounded(mode: str, expected: str) -> None:
    response = make_chat_response("query", execution_mode=mode)

    label = app_module.guidance_type_label(response)

    assert label == expected


def test_knowledge_source_labels_are_bounded_and_hide_internals() -> None:
    response = make_chat_response(
        "query",
        knowledge_source_ids=(
            "power_signature_review",
            "raw_prompt_payload",
            "get_event_metrics",
            "C:\\private\\path",
        ),
    )

    labels = app_module.build_knowledge_source_labels(response)
    combined = " ".join(labels).lower()

    assert labels == ("Knowledge source: power_signature_review",)
    for forbidden in ("raw", "prompt", "get_event_metrics", "private", "path", "tool"):
        assert forbidden not in combined


def test_feedback_remains_explicit_and_requires_acknowledgement() -> None:
    event = make_event("query", seconds=60)
    response = make_chat_response("query")

    assert not can_record_feedback(event, response, human_review_acknowledged=False)
    assert can_record_feedback(event, response, human_review_acknowledged=True)
    record = build_copilot_feedback_record(
        event=event,
        result=response,
        feedback_outcome="useful",
        human_review_acknowledged=True,
    )

    assert record.execution_mode == "adk"
    assert record.human_review_acknowledged is True


def test_copilot_panel_text_does_not_expose_sensitive_details() -> None:
    snapshots = [
        app_module.derive_workflow_snapshot(
            app_module.WorkflowInputs(
                has_candidate_data=True,
                screening_status="within_normal_band",
                current_result_saved=False,
                selected_saved_event_status=None,
                copilot_in_progress=False,
                copilot_execution_mode=None,
                has_copilot_result=False,
                human_review_acknowledged=False,
                saia_configuration_status="configured",
            )
        ),
        app_module.derive_workflow_snapshot(
            app_module.WorkflowInputs(
                has_candidate_data=True,
                screening_status="suspected_deviation",
                current_result_saved=False,
                selected_saved_event_status=None,
                copilot_in_progress=False,
                copilot_execution_mode=None,
                has_copilot_result=False,
                human_review_acknowledged=False,
                saia_configuration_status="configured",
            )
        ),
        app_module.derive_workflow_snapshot(
            app_module.WorkflowInputs(
                has_candidate_data=True,
                screening_status="suspected_deviation",
                current_result_saved=True,
                selected_saved_event_status="suspected_deviation",
                copilot_in_progress=False,
                copilot_execution_mode="adk",
                has_copilot_result=True,
                human_review_acknowledged=False,
                saia_configuration_status="configured",
            )
        ),
    ]
    forbidden = (
        "saia",
        "glm",
        "qwen",
        "api key",
        "c:\\",
        "d:\\",
        "exception",
        "traceback",
        "prompt",
        "raw",
        "payload",
        "event_id",
    )

    for snapshot in snapshots:
        text = app_module.build_copilot_panel_text(snapshot)
        combined = " ".join(
            (
                text.target_label,
                text.eligibility_label,
                text.action_label,
                text.feedback_cta,
            )
        ).lower()
        for fragment in forbidden:
            assert fragment not in combined
