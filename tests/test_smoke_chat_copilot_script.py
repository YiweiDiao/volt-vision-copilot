from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from volt_vision.agent.models import AgentRunTrace, CopilotChatResponse, ToolCallTrace
from volt_vision.agent.policy import MANDATORY_CHAT_HEADINGS


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_chat_copilot.py"


def load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_chat_copilot_test", SCRIPT_PATH)
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


def valid_tool_calls(source: str = "mcp") -> tuple[ToolCallTrace, ...]:
    return (
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


def make_response(execution_mode: str = "adk") -> CopilotChatResponse:
    fallback_reason = None if execution_mode == "adk" else "model_execution_failed"
    trace = AgentRunTrace(
        event_id="current_synthetic_smoke",
        execution_mode=execution_mode,
        tool_names=tuple(call.tool_name for call in valid_tool_calls()),
        tool_calls=valid_tool_calls(
            "mcp" if execution_mode == "adk" else "deterministic_service"
        ),
        fallback_reason=fallback_reason,
        completed=True,
    )
    return CopilotChatResponse(
        event_id="current_synthetic_smoke",
        execution_mode=execution_mode,
        assistant_message=valid_chat_text(),
        knowledge_source_ids=(
            "power_signature_review",
            "cycle_duration_review",
            "energy_and_peak_review",
            "escalation_and_recording",
        ),
        tool_trace=trace,
        human_approval_required=True,
        fallback_reason=fallback_reason,
    )


def configured_env() -> dict[str, str]:
    return {
        "VOLT_VISION_SAIA_API_KEY": "secret-smoke-key",
        "VOLT_VISION_SAIA_MODEL": "qwen3.6-27b",
    }


def test_dry_run_makes_no_model_temp_history_or_mcp_calls(
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
        "run_chat_copilot",
        lambda *_args, **_kwargs: pytest.fail("chat runner must not be called"),
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


def test_mocked_accepted_adk_chat_returns_exit_zero(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(module, "resolve_optional_saia_model_from_env", lambda *_: object())
    monkeypatch.setattr(module, "run_chat_copilot", lambda *_args, **_kwargs: make_response())

    exit_code = module.main(["--live"], environ=configured_env())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "live_status=adk_accepted" in output
    assert "execution_mode=adk" in output
    assert "tool_trace=get_event_metrics,mcp,succeeded,None" in output
    assert "tool_trace=retrieve_maintenance_guidance,mcp,succeeded,None" in output
    assert "tool_trace=find_similar_previous_events,mcp,succeeded,None" in output
    assert "preferred_section_format_observed=yes" in output
    assert "chat_safety_validation_passed=yes" in output


def test_mocked_deterministic_fallback_returns_exit_two(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(module, "resolve_optional_saia_model_from_env", lambda *_: object())
    monkeypatch.setattr(
        module,
        "run_chat_copilot",
        lambda *_args, **_kwargs: make_response("deterministic_fallback"),
    )

    exit_code = module.main(["--live"], environ=configured_env())

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "live_status=deterministic_fallback" in output
    assert "execution_mode=deterministic_fallback" in output
    assert "fallback_reason=model_execution_failed" in output


def test_missing_configuration_returns_exit_one(
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
    assert "live_status=not_configured" in output
    assert "selected_model_id=unset" in output


def test_output_filtering_excludes_sensitive_content(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(module, "resolve_optional_saia_model_from_env", lambda *_: object())
    monkeypatch.setattr(module, "run_chat_copilot", lambda *_args, **_kwargs: make_response())

    module.main(["--live"], environ=configured_env())

    output = capsys.readouterr().out.lower()
    forbidden = (
        "this is a suspected deviation",
        "manual inspection recommended",
        "prompt",
        "current_synthetic_smoke",
        "normalized dtw",
        "duration_seconds",
        "normalized_dtw_distance",
        "evidence",
        "payload",
        "history.jsonl",
        "secret-smoke-key",
        "exception",
        "traceback",
        "sk-testsecretvalue",
    )
    for fragment in forbidden:
        assert fragment not in output


def test_script_level_failure_does_not_print_exception_text(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    monkeypatch.setattr(module, "resolve_optional_saia_model_from_env", lambda *_: object())

    def fail_chat(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("secret exception text with C:\\private\\path")

    monkeypatch.setattr(module, "run_chat_copilot", fail_chat)

    exit_code = module.main(["--live"], environ=configured_env())

    output = capsys.readouterr().out.lower()
    assert exit_code == 1
    assert "live_status=failed" in output
    assert "secret exception" not in output
    assert "private" not in output


def test_automated_live_tests_use_mocked_model_and_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script_module()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "resolve_optional_saia_model_from_env",
        lambda *_: calls.append("model") or object(),
    )
    monkeypatch.setattr(
        module,
        "run_chat_copilot",
        lambda *_args, **_kwargs: calls.append("runner") or make_response(),
    )

    exit_code = module.main(["--live"], environ=configured_env())

    assert exit_code == 0
    assert calls == ["model", "runner"]
