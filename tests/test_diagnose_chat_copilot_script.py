from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from volt_vision.agent.chat import observe_composer_response
from volt_vision.agent.models import ToolCallTrace
from volt_vision.agent.policy import MANDATORY_CHAT_HEADINGS


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_chat_copilot.py"
ALLOWED_SAFETY_LABELS = {
    "diagnosis",
    "repair_or_replacement",
    "shutdown_or_control",
    "ticket",
    "tuning",
    "plc_scada_mes",
    "none",
}


def load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "diagnose_chat_copilot_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def valid_unheaded_chat_text() -> str:
    return (
        "The screening observed a suspected deviation, not a confirmed diagnosis. "
        "Possible contributing conditions should be investigated by comparing "
        "local context and reviewed evidence. Suggested checks include inspection "
        "according to local SOP and recording observations. Confirm locally whether "
        "the selected cycle was complete and comparable. Escalate when recurrence "
        "or production impact requires authorized review."
    )


def valid_tool_calls() -> tuple[ToolCallTrace, ...]:
    return (
        ToolCallTrace(
            tool_name="get_event_metrics",
            source="mcp",
            outcome="succeeded",
            error_code=None,
        ),
        ToolCallTrace(
            tool_name="retrieve_maintenance_guidance",
            source="mcp",
            outcome="succeeded",
            error_code=None,
        ),
        ToolCallTrace(
            tool_name="find_similar_previous_events",
            source="mcp",
            outcome="succeeded",
            error_code=None,
        ),
    )


def outcome(module: ModuleType, text: str | None) -> object:
    return module.ChatDiagnosticExecutionOutcome(
        final_text=text,
        tool_calls=valid_tool_calls(),
        adk_event_count=4,
    )


class FakePart:
    def __init__(self, text: str | None, *, thought: bool = False) -> None:
        self.text = text
        self.thought = thought


class FakeContent:
    def __init__(self, *parts: FakePart) -> None:
        self.parts = parts


class FakeUsage:
    def __init__(self, candidates_token_count: int | None = None) -> None:
        self.candidates_token_count = candidates_token_count


class FakeResponse:
    def __init__(
        self,
        *,
        finish_reason: str | None = None,
        content: FakeContent | None = None,
        usage_metadata: FakeUsage | None = None,
        output_text: str | None = None,
    ) -> None:
        self.finish_reason = finish_reason
        self.content = content
        self.usage_metadata = usage_metadata
        self.output_text = output_text


def configured_env() -> dict[str, str]:
    return {
        "VOLT_VISION_SAIA_API_KEY": "secret-diagnostic-key",
        "VOLT_VISION_SAIA_MODEL": "qwen3.6-27b",
    }


@pytest.mark.parametrize(
    "response, expected",
    [
        (
            FakeResponse(
                finish_reason="STOP",
                content=FakeContent(FakePart("visible hidden text")),
                usage_metadata=FakeUsage(100),
            ),
            {
                "finish_reason": "stop",
                "visible_content_nonempty": "yes",
                "reasoning_content_nonempty": "no",
                "completion_token_usage": "below_half_budget",
            },
        ),
        (
            FakeResponse(
                finish_reason="MAX_TOKENS",
                content=FakeContent(FakePart("")),
                usage_metadata=FakeUsage(700),
            ),
            {
                "finish_reason": "length",
                "visible_content_present": "yes",
                "visible_content_nonempty": "no",
                "completion_token_usage": "near_or_at_budget",
            },
        ),
        (
            FakeResponse(
                finish_reason="STOP",
                content=FakeContent(FakePart("hidden reasoning", thought=True)),
                usage_metadata=FakeUsage(50),
            ),
            {
                "finish_reason": "stop",
                "visible_content_nonempty": "no",
                "reasoning_content_nonempty": "yes",
            },
        ),
        (
            FakeResponse(
                finish_reason="STOP",
                content=FakeContent(FakePart("")),
                output_text="alternate hidden text",
            ),
            {
                "finish_reason": "stop",
                "visible_content_nonempty": "no",
                "alternate_text_field_present": "yes",
            },
        ),
        (
            FakeResponse(
                finish_reason="STOP",
                content=FakeContent(FakePart("")),
                usage_metadata=FakeUsage(0),
            ),
            {
                "finish_reason": "stop",
                "visible_content_present": "yes",
                "visible_content_nonempty": "no",
                "completion_token_usage": "zero",
            },
        ),
    ],
)
def test_composer_response_observation_is_bounded(
    response: FakeResponse,
    expected: dict[str, str],
) -> None:
    observation = observe_composer_response(response, max_output_tokens=700)
    payload = observation.model_dump_json()

    for field, value in expected.items():
        assert getattr(observation, field) == value
    assert "hidden" not in payload
    assert "700" not in payload
    assert "100" not in payload


@pytest.mark.parametrize(
    "text, expected_stage",
    [
        (None, "no_composer_text"),
        ("", "no_composer_text"),
        ("x" * 1801, "chat_length_rejected"),
        (valid_unheaded_chat_text(), "accepted"),
        (
            valid_chat_text().replace(
                "suspected deviation, not a confirmed diagnosis",
                "screening result",
            ),
            "chat_uncertainty_rejected",
        ),
        (valid_chat_text() + "\nget_event_metrics", "chat_internal_reference_rejected"),
        (valid_chat_text() + "\ncurrent_synthetic_smoke", "chat_internal_reference_rejected"),
        (valid_chat_text() + "\nD:\\private\\path", "chat_internal_reference_rejected"),
        (valid_chat_text() + "\nsk-testsecretvalue", "chat_internal_reference_rejected"),
        (valid_chat_text() + "\nThis is a confirmed fault.", "chat_safety_rejected"),
        (valid_chat_text(), "accepted"),
    ],
)
def test_bounded_rejection_stages_map_from_mocked_outcomes(
    tmp_path: Path,
    text: str | None,
    expected_stage: str,
) -> None:
    module = load_script_module()
    history_path = tmp_path / "history.jsonl"
    module._write_synthetic_history(history_path)

    report = module._diagnose_composer_text(
        outcome(module, text),
        selected_model_id="qwen3.6-27b",
        composer_text=text,
        history_path=history_path,
    )

    assert report.stage == expected_stage
    assert report.stage1_mcp_trace_passed is True
    assert report.stage2_composer_called is True


@pytest.mark.parametrize(
    "observation_kwargs, expected_stage",
    [
        (
            {
                "response_received": "yes",
                "finish_reason": "length",
                "visible_content_present": "yes",
                "visible_content_nonempty": "no",
                "reasoning_content_present": "no",
                "reasoning_content_nonempty": "no",
                "completion_token_usage": "near_or_at_budget",
                "alternate_text_field_present": "no",
            },
            "composer_response_length_limited",
        ),
        (
            {
                "response_received": "yes",
                "finish_reason": "stop",
                "visible_content_present": "no",
                "visible_content_nonempty": "no",
                "reasoning_content_present": "yes",
                "reasoning_content_nonempty": "yes",
                "completion_token_usage": "below_half_budget",
                "alternate_text_field_present": "no",
            },
            "composer_reasoning_only",
        ),
        (
            {
                "response_received": "yes",
                "finish_reason": "stop",
                "visible_content_present": "no",
                "visible_content_nonempty": "no",
                "reasoning_content_present": "no",
                "reasoning_content_nonempty": "no",
                "completion_token_usage": "below_half_budget",
                "alternate_text_field_present": "yes",
            },
            "composer_alternate_text_unhandled",
        ),
        (
            {
                "response_received": "yes",
                "finish_reason": "stop",
                "visible_content_present": "no",
                "visible_content_nonempty": "no",
                "reasoning_content_present": "no",
                "reasoning_content_nonempty": "no",
                "completion_token_usage": "below_half_budget",
                "alternate_text_field_present": "no",
            },
            "composer_visible_text_missing",
        ),
        (
            {
                "response_received": "yes",
                "finish_reason": "stop",
                "visible_content_present": "yes",
                "visible_content_nonempty": "no",
                "reasoning_content_present": "no",
                "reasoning_content_nonempty": "no",
                "completion_token_usage": "zero",
                "alternate_text_field_present": "no",
            },
            "no_composer_text",
        ),
    ],
)
def test_empty_composer_text_refines_stage_from_bounded_observation(
    tmp_path: Path,
    observation_kwargs: dict[str, str],
    expected_stage: str,
) -> None:
    module = load_script_module()
    history_path = tmp_path / "history.jsonl"
    module._write_synthetic_history(history_path)
    observation = module.ComposerResponseObservation(**observation_kwargs)

    report = module._diagnose_composer_text(
        outcome(module, ""),
        selected_model_id="qwen3.6-27b",
        composer_text="",
        composer_observation=observation,
        history_path=history_path,
    )

    assert report.stage == expected_stage


def test_mcp_trace_rejected_stage_maps_from_bad_trace(tmp_path: Path) -> None:
    module = load_script_module()
    history_path = tmp_path / "history.jsonl"
    module._write_synthetic_history(history_path)
    bad_outcome = module.ChatDiagnosticExecutionOutcome(
        final_text=valid_chat_text(),
        tool_calls=(valid_tool_calls()[1], valid_tool_calls()[0], valid_tool_calls()[2]),
        adk_event_count=4,
    )

    report = module._diagnose_outcome(
        bad_outcome,
        selected_model_id="qwen3.6-27b",
    )

    assert report.stage == "mcp_trace_rejected"
    assert report.stage1_mcp_trace_passed is False
    assert report.stage2_composer_called is False


def test_valid_stage_one_reports_composer_not_called(tmp_path: Path) -> None:
    module = load_script_module()
    history_path = tmp_path / "history.jsonl"
    module._write_synthetic_history(history_path)

    report = module._diagnose_outcome(
        outcome(module, None),
        selected_model_id="qwen3.6-27b",
    )

    assert report.stage == "composer_not_called"
    assert report.stage1_mcp_trace_passed is True
    assert report.stage2_composer_called is False


def test_construction_failure_stage_maps_from_mocked_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    history_path = tmp_path / "history.jsonl"
    module._write_synthetic_history(history_path)
    monkeypatch.setattr(
        module,
        "retrieve_maintenance_guidance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("hidden path")),
    )

    report = module._diagnose_composer_text(
        outcome(module, valid_chat_text()),
        selected_model_id="qwen3.6-27b",
        composer_text=valid_chat_text(),
        history_path=history_path,
    )

    assert report.stage == "chat_response_construction_failed"
    assert report.chat_response_constructed is False


def test_bundle_creation_and_adk_run_failures_are_bounded(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(module, "resolve_optional_saia_model_from_env", lambda *_: object())

    def fail_create(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("secret create failure")

    monkeypatch.setattr(module, "create_chat_copilot_agent", fail_create)
    assert module.main(["--live"], environ=configured_env()) == 1
    output = capsys.readouterr().out
    assert "diagnostic_stage=bundle_creation_failed" in output
    assert "secret create failure" not in output

    monkeypatch.setattr(module, "create_chat_copilot_agent", lambda **_kwargs: object())

    def fail_execute(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("secret execute failure")

    monkeypatch.setattr(module, "execute_chat_copilot_diagnostic_bundle", fail_execute)
    assert module.main(["--live"], environ=configured_env()) == 1
    output = capsys.readouterr().out
    assert "diagnostic_stage=adk_run_failed" in output
    assert "secret execute failure" not in output


def test_composer_failure_is_bounded(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(module, "resolve_optional_saia_model_from_env", lambda *_: object())
    monkeypatch.setattr(module, "create_chat_copilot_agent", lambda **_kwargs: object())
    monkeypatch.setattr(
        module,
        "execute_chat_copilot_diagnostic_bundle",
        lambda *_args, **_kwargs: outcome(module, None),
    )

    def fail_composer(*_args: object, **_kwargs: object) -> tuple[str, object]:
        raise RuntimeError("secret composer failure")

    monkeypatch.setattr(module, "compose_chat_with_model_observed", fail_composer)

    assert module.main(["--live"], environ=configured_env()) == 1
    output = capsys.readouterr().out
    assert "diagnostic_stage=composer_failed" in output
    assert "stage2_composer_called=yes" in output
    assert "secret composer failure" not in output


def test_heading_flags_and_safety_categories_are_bounded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script_module()
    report = module.DiagnosticReport(
        stage="chat_safety_rejected",
        selected_model_id="qwen3.6-27b",
        adk_event_count=4,
        tool_calls=valid_tool_calls(),
        composer_text=(
            valid_chat_text()
            + "\nConfirmed fault. Replace part. Stop the machine. Create ticket. "
            "Parameter tuning. PLC."
        ),
    )

    module._print_diagnostic(report)

    output = capsys.readouterr().out
    labels_line = next(
        line for line in output.splitlines() if line.startswith("safety_category_hits=")
    )
    labels = set(labels_line.split("=", 1)[1].split(","))
    assert labels <= ALLOWED_SAFETY_LABELS
    assert "heading_presence:screening_observed=yes" in output
    assert "heading_presence:possible_conditions=yes" in output
    assert "heading_presence:suggested_steps=yes" in output
    assert "heading_presence:questions_local=yes" in output
    assert "heading_presence:escalation=yes" in output


def test_preferred_section_format_is_informative_not_acceptance_gate(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    module = load_script_module()
    history_path = tmp_path / "history.jsonl"
    module._write_synthetic_history(history_path)

    report = module._diagnose_composer_text(
        outcome(module, valid_unheaded_chat_text()),
        selected_model_id="qwen3.6-27b",
        composer_text=valid_unheaded_chat_text(),
        history_path=history_path,
    )
    module._print_diagnostic(report)

    output = capsys.readouterr().out
    assert report.stage == "accepted"
    assert "preferred_section_format_observed=no" in output
    assert "chat_response_constructed=yes" in output


def test_diagnostic_output_excludes_sensitive_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_script_module()
    report = module.DiagnosticReport(
        stage="chat_internal_reference_rejected",
        selected_model_id="qwen3.6-27b",
        adk_event_count=4,
        tool_calls=valid_tool_calls(),
        composer_text=(
            valid_chat_text()
            + "\nPrompt payload current_synthetic_smoke CNC_SMOKE "
            "D:\\private\\history.jsonl secret-diagnostic-key traceback"
        ),
    )

    module._print_diagnostic(report)

    output = capsys.readouterr().out.lower()
    forbidden = (
        "prompt payload",
        "current_synthetic_smoke",
        "cnc_smoke",
        "private",
        "history.jsonl",
        "secret-diagnostic-key",
        "traceback",
        "this is a suspected deviation",
        "manual inspection recommended",
    )
    for fragment in forbidden:
        assert fragment not in output


def test_dry_run_has_no_model_network_mcp_or_temp_history_side_effects(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(
        module,
        "resolve_optional_saia_model_from_env",
        lambda *_: pytest.fail("model resolver must not be called"),
    )
    monkeypatch.setattr(
        module,
        "create_chat_copilot_agent",
        lambda *_args, **_kwargs: pytest.fail("bundle must not be created"),
    )
    monkeypatch.setattr(
        module,
        "execute_chat_copilot_diagnostic_bundle",
        lambda *_args, **_kwargs: pytest.fail("ADK must not run"),
    )
    monkeypatch.setattr(
        module,
        "TemporaryDirectory",
        lambda: pytest.fail("temporary history must not be created"),
    )

    exit_code = module.main([], environ=configured_env())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.splitlines() == [
        "configuration_status=configured",
        "selected_model_id=qwen3.6-27b",
        "key_configured=yes",
    ]


def test_mocked_accepted_live_chat_reports_accepted(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(module, "resolve_optional_saia_model_from_env", lambda *_: object())
    monkeypatch.setattr(module, "create_chat_copilot_agent", lambda **_kwargs: object())
    monkeypatch.setattr(
        module,
        "execute_chat_copilot_diagnostic_bundle",
        lambda *_args, **_kwargs: outcome(module, None),
    )
    monkeypatch.setattr(
        module,
        "compose_chat_with_model_observed",
        lambda *_args, **_kwargs: (
            valid_chat_text(),
            module._default_observation(valid_chat_text()),
        ),
    )

    exit_code = module.main(["--live"], environ=configured_env())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "diagnostic_stage=accepted" in output
    assert "stage1_mcp_trace_passed=yes" in output
    assert "stage2_composer_called=yes" in output
    assert "chat_response_constructed=yes" in output


def test_missing_configuration_reports_bundle_creation_failed(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(
        module,
        "resolve_optional_saia_model_from_env",
        lambda *_: pytest.fail("model resolver must not be called"),
    )

    exit_code = module.main(["--live"], environ={})

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "diagnostic_stage=bundle_creation_failed" in output
    assert "selected_model_id=unset" in output
