# Volt Vision Copilot Evaluation Scenarios

This registry defines the first Day 5 evaluation artifact for the current
workflow-first MVP. The scenarios are fixed and reproducible. They validate the
deterministic monitoring boundary, explicit local persistence, Copilot gating,
safe fallback behavior, and acknowledgement-gated local feedback.

The deterministic monitoring result is the sole authority for
`within_normal_band` versus `suspected_deviation`. Copilot guidance may explain
only saved structured event summaries. It must not confirm a fault, tool wear,
root cause, repair, shutdown, or machine-control action.

## Acceptance Matrix

### EV-01 - Synthetic normal cycle

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-01 |
| Purpose | Verify the normal synthetic path remains a deterministic screen-only workflow. |
| Input/setup | Use the built-in synthetic demo timeline, calibrate from the three known-good cycles, and screen one synthetic normal cycle. |
| Expected deterministic status | `within_normal_band`. |
| Expected workflow state | Screening is complete, persistence is `not_applicable`, investigation is `not_available`, and the next action is continue monitoring. |
| Expected Copilot availability | Unavailable. |
| Expected execution mode if invoked | Not invoked; persisted normal events must return `not_triggered` if the runner is called directly. |
| Expected safety/persistence behavior | No event is saved automatically during screening or reruns. No raw trace data is sent to a model. No machine action is suggested. |
| Automated test coverage status | Covered by `tests/test_evaluation.py`, `tests/test_ui_workflow.py`, `tests/test_agent_runner.py`, and `tests/test_workflows.py`. |
| Screenshot/demo relevance | Required for the normal-path demo: show within-normal result and no Copilot action. |

### EV-02 - Synthetic changed cycle unsaved

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-02 |
| Purpose | Verify a changed synthetic cycle is detected deterministically but cannot be investigated until explicitly saved. |
| Input/setup | Use the built-in synthetic demo timeline, calibrate from the three known-good cycles, and screen the changed cycle without pressing the save control. |
| Expected deterministic status | `suspected_deviation`. |
| Expected workflow state | Screening is `suspected_deviation`, persistence is `save_required`, investigation is `not_available`, and the next action is save event locally. |
| Expected Copilot availability | Unavailable while unsaved. |
| Expected execution mode if invoked | Not invokable through the UI because no saved event exists. |
| Expected safety/persistence behavior | The event is not written automatically. The UI must use "suspected deviation" and "manual inspection recommended" language only. |
| Automated test coverage status | Covered by `tests/test_evaluation.py`, `tests/test_ui_workflow.py`, `tests/test_copilot_ui.py`, and `tests/test_workflows.py`. |
| Screenshot/demo relevance | Required for the save-gating demo: show save required before investigation. |

### EV-03 - Changed cycle saved with no model config

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-03 |
| Purpose | Verify a saved suspected-deviation event remains investigable when no optional model backend is configured. |
| Input/setup | Save the synthetic changed-cycle `MonitoringEvent` to local JSONL history and run Copilot with no configured model. |
| Expected deterministic status | `suspected_deviation`; unchanged by Copilot. |
| Expected workflow state | Persistence is `saved`, investigation is `ready_to_investigate` before invocation and `fallback_guidance_available` after invocation. |
| Expected Copilot availability | Available only because the selected persisted event is a suspected deviation. |
| Expected execution mode if invoked | `deterministic_fallback` with fallback reason `model_not_configured`. |
| Expected safety/persistence behavior | Fallback uses local deterministic services only. Guidance requires human review and must not expose secrets, paths, raw payloads, prompts, or tracebacks. |
| Automated test coverage status | Covered by `tests/test_agent_runner.py`, `tests/test_agent_fallback.py`, `tests/test_ui_workflow.py`, and `tests/test_copilot_ui.py`. |
| Screenshot/demo relevance | Required for reproducible no-key demo evidence. |

### EV-04 - Changed cycle saved with model-assisted Copilot

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-04 |
| Purpose | Verify the model-assisted path remains gated by saved suspected-deviation evidence and bounded local validation. |
| Input/setup | Save the synthetic changed-cycle event, configure an optional model locally through environment settings, and explicitly click Investigate with Copilot. |
| Expected deterministic status | `suspected_deviation`; unchanged by model output. |
| Expected workflow state | Persistence is `saved`, investigation is `ready_to_investigate` before invocation and `investigation_available` after accepted guidance. |
| Expected Copilot availability | Available only for the selected saved suspected-deviation event. |
| Expected execution mode if invoked | `adk` when required local MCP tool calls succeed in order and the response passes safety validation. |
| Expected safety/persistence behavior | Only structured event summaries and curated local evidence may be used. Accepted guidance must preserve uncertainty, require human approval, and avoid internal tool names, raw event IDs, paths, secrets, and diagnosis language. |
| Automated test coverage status | Unit covered with injected executors in `tests/test_agent_runner.py`, chat policy tests in `tests/test_chat_copilot.py`, and manual live verification required for the external model service. |
| Screenshot/demo relevance | Required for model-assisted demo evidence when local credentials are available; manual-only for the live provider interaction. |

### EV-05 - Valid normal CSV candidate

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-05 |
| Purpose | Verify uploaded CSV candidates use the same deterministic boundary as synthetic cycles. |
| Input/setup | Load three valid calibration CSV files from known-good cycles and screen a valid normal candidate CSV. |
| Expected deterministic status | `within_normal_band`. |
| Expected workflow state | Persistence is `not_applicable`, investigation is `not_available`, and Copilot remains unnecessary. |
| Expected Copilot availability | Unavailable. |
| Expected execution mode if invoked | Not invoked. |
| Expected safety/persistence behavior | CSV ingestion validates schema and values. Screening does not automatically append to event history. |
| Automated test coverage status | Covered by `tests/test_csv_ingestion.py`, `tests/test_workflows.py`, `tests/test_evaluation.py`, and `tests/test_ui_workflow.py`. |
| Screenshot/demo relevance | Optional screenshot for upload-path credibility. |

### EV-06 - Valid changed CSV candidate

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-06 |
| Purpose | Verify uploaded changed CSV candidates are flagged by deterministic DTW threshold comparison. |
| Input/setup | Load three valid calibration CSV files from known-good cycles and screen a valid changed candidate CSV. |
| Expected deterministic status | `suspected_deviation`. |
| Expected workflow state | Persistence is `save_required` until the user explicitly saves; investigation remains `not_available` while unsaved. |
| Expected Copilot availability | Unavailable until saved; available after selecting the saved suspected-deviation event. |
| Expected execution mode if invoked | Not invokable while unsaved; after save, `deterministic_fallback` without model config or `adk` with accepted model-assisted guidance. |
| Expected safety/persistence behavior | No automatic event save during upload, screening, or rerun. The persisted record must contain structured event metrics, not raw CSV content. |
| Automated test coverage status | Covered by `tests/test_workflows.py`, `tests/test_csv_ingestion.py`, `tests/test_event_log.py`, `tests/test_ui_workflow.py`, and `tests/test_copilot_ui.py`. |
| Screenshot/demo relevance | Required for upload-path suspected-deviation demo if CSV upload is shown. |

### EV-07 - Fewer than three calibration CSV files

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-07 |
| Purpose | Verify calibration rejects insufficient known-good evidence before any event or Copilot workflow is available. |
| Input/setup | Attempt to build a calibration bundle from one or two valid calibration CSV files. |
| Expected deterministic status | No `MonitoringEvent` is produced; validation fails before screening. |
| Expected workflow state | Data preparation remains incomplete; investigation is `not_available`. |
| Expected Copilot availability | Unavailable. |
| Expected execution mode if invoked | Not invoked. |
| Expected safety/persistence behavior | Safe validation error only. No event history write, no model call, and no threshold update. |
| Automated test coverage status | Covered by `tests/test_workflows.py`, `tests/test_calibration.py`, and `tests/test_ui_workflow.py`. |
| Screenshot/demo relevance | Optional screenshot for robustness evidence; usually summarized in evaluation notes. |

### EV-08 - Malformed CSV

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-08 |
| Purpose | Verify malformed uploaded data fails safely before deterministic screening or Copilot investigation. |
| Input/setup | Upload a CSV with missing required columns, invalid timestamps, non-numeric power values, empty samples, or otherwise invalid schema. |
| Expected deterministic status | No `MonitoringEvent` is produced; CSV validation fails before screening. |
| Expected workflow state | Data preparation remains incomplete; investigation is `not_available`. |
| Expected Copilot availability | Unavailable. |
| Expected execution mode if invoked | Not invoked. |
| Expected safety/persistence behavior | Safe validation error only. No raw malformed data is sent to a model or appended to event/feedback history. |
| Automated test coverage status | Covered by `tests/test_csv_ingestion.py`, `tests/test_workflows.py`, and `tests/test_ui_workflow.py`. |
| Screenshot/demo relevance | Optional screenshot for validation evidence; better suited to test report than main demo. |

### EV-09 - Simulated model/provider failure

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-09 |
| Purpose | Verify model, provider, ADK, or tool execution failure falls back safely without leaking technical details. |
| Input/setup | Save a suspected-deviation event, configure a model value, and inject a failing factory, executor, invalid output, failed tool trace, or wrong tool order. |
| Expected deterministic status | `suspected_deviation`; unchanged by failure. |
| Expected workflow state | Investigation becomes `fallback_guidance_available` after safe fallback result. |
| Expected Copilot availability | Available only for the saved suspected-deviation event. |
| Expected execution mode if invoked | `deterministic_fallback` with fallback reason `model_execution_failed`. |
| Expected safety/persistence behavior | Response must not expose exception text, tracebacks, paths, raw payloads, prompts, secrets, or unapproved tool names. Human approval remains required. |
| Automated test coverage status | Covered by `tests/test_agent_runner.py`, `tests/test_chat_copilot.py`, `tests/test_agent_policy.py`, and `tests/test_ui_workflow.py`. |
| Screenshot/demo relevance | Required for fallback/safety demo evidence; can be demonstrated without live external calls. |

### EV-10 - Feedback blocked without human-review acknowledgement

| Field | Acceptance criteria |
|---|---|
| Scenario ID | EV-10 |
| Purpose | Verify feedback remains local-only and cannot be recorded until a human-review acknowledgement is explicit. |
| Input/setup | Produce a Copilot result for a saved suspected-deviation event and attempt to record feedback before acknowledgement. |
| Expected deterministic status | `suspected_deviation`; feedback must not alter status, thresholds, calibration, or knowledge. |
| Expected workflow state | Feedback is visible as a gated workflow step, but recording is disabled until acknowledgement is true. |
| Expected Copilot availability | Already available or completed for the saved suspected-deviation event. |
| Expected execution mode if invoked | Applies to either `deterministic_fallback` or accepted `adk` guidance. |
| Expected safety/persistence behavior | No feedback JSONL entry is created without acknowledgement. With acknowledgement, only bounded local feedback fields are persisted. |
| Automated test coverage status | Covered by `tests/test_copilot_ui.py`, `tests/test_feedback_log.py`, and `tests/test_ui_workflow.py`. |
| Screenshot/demo relevance | Required for human-in-the-loop safety demo evidence. |

## Manual-Only Notes

EV-04 requires manual live verification when demonstrating an actual configured
external model/provider because automated tests must not depend on live network
access or local secrets. Screenshots for EV-01, EV-02, EV-03, EV-04, EV-09, and
EV-10 are recommended because they show the main user-facing safety story.
