"""Streamlit dashboard for Volt Vision Copilot.

Manual launch:
    uv run streamlit run src/volt_vision/ui/app.py
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Literal, Protocol

import pandas as pd
import plotly.express as px
import streamlit as st

from volt_vision.monitoring.cycles import SelectedCycle
from volt_vision.monitoring.csv_ingestion import load_candidate_cycle_from_csv
from volt_vision.feedback.feedback_log import (
    DEFAULT_FEEDBACK_HISTORY_PATH,
    append_copilot_feedback_record,
    clear_copilot_feedback_log_for_demo,
    read_copilot_feedback_records,
)
from volt_vision.agent import (
    resolve_optional_saia_model_from_env,
    run_chat_copilot,
    safe_saia_configuration_status,
)
from volt_vision.agent.models import CopilotChatResponse
from volt_vision.monitoring.event_log import (
    append_monitoring_event,
    clear_monitoring_event_log_for_demo,
    read_monitoring_events,
)
from volt_vision.monitoring.models import MonitoringEvent
from volt_vision.monitoring.thresholds import ThresholdResult
from volt_vision.monitoring.workflows import (
    CalibrationBundle,
    build_calibration_bundle_from_csv,
    build_monitoring_event_from_candidate_csv,
)
from volt_vision.ui.dashboard import (
    DemoReplayData,
    build_event_history_frame,
    build_demo_replay_data,
    build_power_comparison_frame,
    format_indicator_percentage,
    get_demo_operating_state_at_elapsed_seconds,
    recommended_follow_up_text,
    run_synthetic_demo_workflow,
)
from volt_vision.ui.copilot import (
    build_copilot_feedback_record,
    build_feedback_history_frame,
    can_investigate_event,
    can_record_feedback,
    CopilotUiResult,
    get_selected_persisted_event,
    is_current_result_for_event,
    latest_events_by_id,
)
from volt_vision.ui.workflow import (
    WorkflowInputs,
    WorkflowSnapshot,
    derive_workflow_snapshot,
)
from volt_vision.ui.workflow_status_strip import (
    WorkflowStatusStripText,
    build_workflow_status_strip_text,
)

SAFETY_NOTE = (
    "This educational prototype screens power-signature deviations. "
    "A suspected deviation is not a confirmed fault or root-cause diagnosis. "
    "Manual inspection is recommended before any action."
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMO_EVENT_LOG_PATH = PROJECT_ROOT / "data" / "demo_event_history.jsonl"
DEMO_FEEDBACK_LOG_PATH = DEFAULT_FEEDBACK_HISTORY_PATH
SESSION_RESULT_KEY = "latest_monitoring_result"
COPILOT_RESULTS_KEY = "copilot_results_by_event_id"
COPILOT_EVENT_SELECTION_KEY = "copilot_selected_event_id"
HUMAN_REVIEW_ACKNOWLEDGED_KEY = "copilot_human_review_acknowledged"
COPILOT_IN_PROGRESS_EVENT_IDS_KEY = "copilot_in_progress_event_ids"
LAST_SAVED_EVENT_CONFIRMATION_KEY = "last_saved_event_confirmation"
COPILOT_SPINNER_MESSAGE = (
    "Copilot is reviewing saved screening evidence and approved guidance. "
    "No automatic machine action is being taken."
)
COPILOT_REVIEW_REMINDER = (
    "Guidance supports human review and does not confirm a fault or root cause."
)
KNOWLEDGE_PENDING_MESSAGE = (
    "Knowledge sources appear after a Copilot investigation is available."
)
FEEDBACK_PENDING_MESSAGE = (
    "Run Copilot investigation before recording local feedback."
)
FORBIDDEN_UI_TEXT_FRAGMENTS = (
    "api key",
    "api_key",
    "exception",
    "get_event_metrics",
    "prompt",
    "raw",
    "secret",
    "stack trace",
    "token",
    "tool",
    "traceback",
)

ConfigurationStatusReader = Callable[[], str]
ModelResolver = Callable[[], object | None]
ChatCopilotRunner = Callable[..., CopilotChatResponse]


class UploadedFileLike(Protocol):
    def getvalue(self) -> bytes:
        ...


@dataclass(frozen=True)
class CsvWorkflowResult:
    bundle: CalibrationBundle
    event: MonitoringEvent
    reference_cycle: SelectedCycle
    candidate_cycle: SelectedCycle


@dataclass(frozen=True)
class RenderedMonitoringResult:
    mode: Literal["synthetic", "csv"]
    event: MonitoringEvent
    calibration_cycle_count: int
    reference_cycle: SelectedCycle
    candidate_cycle: SelectedCycle
    reference_cycle_id: str
    threshold_result: ThresholdResult


@dataclass(frozen=True)
class InvestigationWorkspaceContext:
    result: RenderedMonitoringResult
    events: tuple[MonitoringEvent, ...]
    latest_events: tuple[MonitoringEvent, ...]
    current_result_saved: bool
    selected_event: MonitoringEvent | None
    copilot_result: CopilotChatResponse | None
    workflow_snapshot: object


@dataclass(frozen=True)
class CopilotPanelText:
    target_label: str
    eligibility_label: str
    action_label: str
    feedback_cta: str


def main() -> None:
    st.set_page_config(page_title="Volt Vision Copilot", layout="wide")
    render_header()
    workflow_status_placeholder = st.empty()

    mode = st.radio(
        "Mode",
        ["Synthetic demo", "CSV workflow"],
        horizontal=True,
    )

    if mode == "Synthetic demo":
        render_synthetic_demo_mode()
    else:
        render_csv_workflow_mode()
    render_workflow_status_strip(
        event_log_path=DEMO_EVENT_LOG_PATH,
        placeholder=workflow_status_placeholder,
    )


def render_header() -> None:
    st.title("Volt Vision Copilot")
    st.caption("Patent-inspired deterministic power-cycle monitoring prototype")
    st.warning(SAFETY_NOTE)


def render_workflow_status_strip(
    *,
    event_log_path: Path,
    placeholder: object | None = None,
) -> None:
    inputs = build_workflow_inputs_from_app_state(event_log_path=event_log_path)
    snapshot = derive_workflow_snapshot(inputs)
    text = build_workflow_status_strip_text(snapshot)

    if placeholder is not None:
        with placeholder.container():  # type: ignore[attr-defined]
            render_workflow_status_strip_text(snapshot, text)
        return
    render_workflow_status_strip_text(snapshot, text)


def render_workflow_status_strip_text(
    snapshot: WorkflowSnapshot,
    text: WorkflowStatusStripText,
) -> None:
    with st.container(border=True):
        st.caption(text.path_label)
        state_column, action_column, assistance_column = st.columns([1, 2, 1])
        with state_column:
            st.markdown(f"**Current state:** {text.current_state_label}")
        with action_column:
            st.markdown(f"**Next safe action:** {text.primary_action_label}")
            st.caption(text.short_instruction)
            if not snapshot.primary_action_enabled:
                st.caption(text.action_availability_label)
        with assistance_column:
            st.markdown(f"**Assistance:** {text.assistance_availability_label}")
        st.info(text.safety_reminder)


def build_workflow_inputs_from_app_state(*, event_log_path: Path) -> WorkflowInputs:
    latest_result = st.session_state.get(SESSION_RESULT_KEY)
    result = (
        latest_result
        if isinstance(latest_result, RenderedMonitoringResult)
        else None
    )
    events = _read_events_for_workflow_status(event_log_path)
    latest_events = latest_events_by_id(events)
    saved_event_ids = {event.event_id for event in latest_events}
    selected_event = _selected_event_for_workflow_status(latest_events)
    copilot_result = _copilot_result_for_workflow_status(selected_event)

    return WorkflowInputs(
        has_candidate_data=result is not None,
        screening_status=result.event.status if result is not None else None,
        current_result_saved=(
            result is not None and result.event.event_id in saved_event_ids
        ),
        selected_saved_event_status=(
            selected_event.status if selected_event is not None else None
        ),
        copilot_in_progress=(
            selected_event is not None
            and _is_copilot_in_progress(selected_event.event_id)
        ),
        copilot_execution_mode=(
            _copilot_execution_mode(copilot_result)
            if copilot_result is not None
            and _copilot_execution_mode(copilot_result)
            in {"adk", "deterministic_fallback"}
            else None
        ),
        has_copilot_result=copilot_result is not None,
        human_review_acknowledged=bool(
            st.session_state.get(HUMAN_REVIEW_ACKNOWLEDGED_KEY, False)
        ),
        saia_configuration_status=safe_saia_configuration_status(),
    )


def _read_events_for_workflow_status(log_path: Path) -> tuple[MonitoringEvent, ...]:
    try:
        return tuple(read_monitoring_events(log_path))
    except ValueError:
        return ()


def _selected_event_for_workflow_status(
    latest_events: tuple[MonitoringEvent, ...],
) -> MonitoringEvent | None:
    if not latest_events:
        return None

    selected_event_id = st.session_state.get(COPILOT_EVENT_SELECTION_KEY)
    if isinstance(selected_event_id, str):
        selected_event = get_selected_persisted_event(latest_events, selected_event_id)
        if selected_event is not None:
            return selected_event
    return latest_events[0]


def _selected_event_for_workspace(
    current_event: MonitoringEvent,
    latest_events: tuple[MonitoringEvent, ...],
) -> MonitoringEvent | None:
    if not latest_events:
        return None

    selected_event_id = st.session_state.get(COPILOT_EVENT_SELECTION_KEY)
    if isinstance(selected_event_id, str):
        selected_event = get_selected_persisted_event(latest_events, selected_event_id)
        if selected_event is not None:
            return selected_event

    if _is_event_saved(current_event, latest_events):
        return get_selected_persisted_event(latest_events, current_event.event_id)
    return latest_events[0]


def _is_event_saved(
    event: MonitoringEvent,
    latest_events: tuple[MonitoringEvent, ...],
) -> bool:
    return any(saved_event.event_id == event.event_id for saved_event in latest_events)


def select_saved_event_for_investigation(event: MonitoringEvent) -> None:
    st.session_state[COPILOT_EVENT_SELECTION_KEY] = event.event_id


def _copilot_result_for_workflow_status(
    selected_event: MonitoringEvent | None,
) -> CopilotChatResponse | None:
    if selected_event is None:
        return None

    result_store = st.session_state.get(COPILOT_RESULTS_KEY)
    if not isinstance(result_store, dict):
        return None

    result = result_store.get(selected_event.event_id)
    if isinstance(result, CopilotChatResponse) and is_current_result_for_event(
        result,
        selected_event,
    ):
        return result
    return None


def render_synthetic_demo_mode() -> None:
    choice = st.selectbox(
        "Demo candidate",
        ["Normal demo candidate", "Changed demo candidate"],
    )
    synthetic_choice = "normal" if choice == "Normal demo candidate" else "changed"
    result = run_synthetic_demo_workflow(candidate=synthetic_choice)
    replay_data = build_demo_replay_data()
    render_demo_timeline_replay(replay_data)
    render_context = RenderedMonitoringResult(
        mode="synthetic",
        event=result.event,
        calibration_cycle_count=len(result.calibration_cycles),
        reference_cycle=result.reference_cycle,
        candidate_cycle=result.candidate_cycle,
        reference_cycle_id=result.calibration_result.reference_cycle.segment_id,
        threshold_result=result.threshold_result,
    )
    st.session_state[SESSION_RESULT_KEY] = render_context
    render_result(render_context)


def render_csv_workflow_mode() -> None:
    calibration_uploads = st.file_uploader(
        "Known-good calibration CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )
    candidate_upload = st.file_uploader(
        "Candidate CSV file",
        type=["csv"],
        accept_multiple_files=False,
    )

    if st.button("Run deterministic screening", type="primary"):
        if len(calibration_uploads) < 3:
            st.error("At least three known-good calibration CSV files are required.")
            return
        if candidate_upload is None:
            st.error("Exactly one candidate CSV file is required.")
            return

        try:
            result = run_csv_workflow_from_uploads(
                calibration_uploads,
                candidate_upload,
            )
        except ValueError as error:
            st.error(str(error))
            return

        st.session_state[SESSION_RESULT_KEY] = RenderedMonitoringResult(
            mode="csv",
            event=result.event,
            calibration_cycle_count=len(result.bundle.calibration_cycles),
            reference_cycle=result.reference_cycle,
            candidate_cycle=result.candidate_cycle,
            reference_cycle_id=(
                result.bundle.calibration_result.reference_cycle.segment_id
            ),
            threshold_result=result.bundle.threshold_result,
        )
    stored_result = st.session_state.get(SESSION_RESULT_KEY)
    if (
        isinstance(stored_result, RenderedMonitoringResult)
        and stored_result.mode == "csv"
    ):
        render_result(stored_result)


def render_demo_timeline_replay(replay_data: DemoReplayData) -> None:
    st.subheader("Demo Timeline Replay")
    replay_position = st.slider(
        "Replay position",
        min_value=0.0,
        max_value=float(replay_data.max_elapsed_seconds),
        value=0.0,
        step=1.0,
    )
    current_state = get_demo_operating_state_at_elapsed_seconds(
        replay_data,
        replay_position,
    )
    st.write(
        f"Current demo operating state: {current_state} "
        f"at {replay_position:.0f} elapsed seconds"
    )

    figure = px.line(
        replay_data.samples,
        x="elapsed_seconds",
        y="power_kw",
        title="Synthetic power timeline replay",
        labels={
            "elapsed_seconds": "Elapsed time in seconds",
            "power_kw": "Power in kW",
        },
    )
    figure.add_vline(
        x=replay_position,
        line_width=3,
        line_dash="dash",
        line_color="red",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.info(
        "Replay state labels are supplied by synthetic demo metadata for "
        "illustration only. Uploaded CSV workflows do not perform automatic "
        "idle/processing segmentation in this prototype."
    )


def run_csv_workflow_from_uploads(
    calibration_uploads: list[UploadedFileLike],
    candidate_upload: UploadedFileLike,
) -> CsvWorkflowResult:
    """Run one CSV workflow using temporary generated file names only."""

    with TemporaryDirectory() as temporary_directory:
        run_directory = Path(temporary_directory)
        calibration_paths = []
        for index, upload in enumerate(calibration_uploads, start=1):
            csv_path = run_directory / f"calibration_{index}.csv"
            csv_path.write_bytes(upload.getvalue())
            calibration_paths.append(csv_path)

        candidate_path = run_directory / "candidate.csv"
        candidate_path.write_bytes(candidate_upload.getvalue())

        bundle = build_calibration_bundle_from_csv(calibration_paths)
        event = build_monitoring_event_from_candidate_csv(
            candidate_path,
            bundle,
            candidate_segment_id="uploaded_cycle",
        )
        candidate_cycle = load_candidate_cycle_from_csv(
            candidate_path,
            candidate_segment_id="uploaded_cycle",
        )
        return CsvWorkflowResult(
            bundle=bundle,
            event=event,
            reference_cycle=bundle.calibration_result.reference_cycle,
            candidate_cycle=candidate_cycle,
        )


def render_result(result: RenderedMonitoringResult) -> None:
    render_investigation_workspace(
        result=result,
        event_log_path=DEMO_EVENT_LOG_PATH,
        feedback_log_path=DEMO_FEEDBACK_LOG_PATH,
    )


def render_investigation_workspace(
    *,
    result: RenderedMonitoringResult,
    event_log_path: Path,
    feedback_log_path: Path,
) -> None:
    context = build_investigation_workspace_context(
        result=result,
        event_log_path=event_log_path,
    )
    st.subheader("Investigation workspace")
    left_column, right_column = st.columns([3, 2])
    with left_column:
        render_evidence_column(context)
    with right_column:
        render_copilot_panel(context, event_log_path=event_log_path)
    render_workspace_tabs(context, event_log_path, feedback_log_path)


def build_investigation_workspace_context(
    *,
    result: RenderedMonitoringResult,
    event_log_path: Path,
) -> InvestigationWorkspaceContext:
    events = _read_events_for_workflow_status(event_log_path)
    latest_events = latest_events_by_id(events)
    current_result_saved = _is_event_saved(result.event, latest_events)
    selected_event = _selected_event_for_workspace(result.event, latest_events)
    copilot_result = _copilot_result_for_workflow_status(selected_event)
    workflow_inputs = _build_workflow_inputs_from_safe_facts(
        result=result,
        latest_events=latest_events,
        current_result_saved=current_result_saved,
        selected_event=selected_event,
        copilot_result=copilot_result,
    )
    return InvestigationWorkspaceContext(
        result=result,
        events=events,
        latest_events=latest_events,
        current_result_saved=current_result_saved,
        selected_event=selected_event,
        copilot_result=copilot_result,
        workflow_snapshot=derive_workflow_snapshot(workflow_inputs),
    )


def _build_workflow_inputs_from_safe_facts(
    *,
    result: RenderedMonitoringResult | None,
    latest_events: tuple[MonitoringEvent, ...],
    current_result_saved: bool,
    selected_event: MonitoringEvent | None,
    copilot_result: CopilotChatResponse | None,
) -> WorkflowInputs:
    return WorkflowInputs(
        has_candidate_data=result is not None,
        screening_status=result.event.status if result is not None else None,
        current_result_saved=current_result_saved,
        selected_saved_event_status=(
            selected_event.status if selected_event is not None else None
        ),
        copilot_in_progress=(
            selected_event is not None
            and _is_copilot_in_progress(selected_event.event_id)
        ),
        copilot_execution_mode=(
            _copilot_execution_mode(copilot_result)
            if copilot_result is not None
            and _copilot_execution_mode(copilot_result)
            in {"adk", "deterministic_fallback"}
            else None
        ),
        has_copilot_result=copilot_result is not None,
        human_review_acknowledged=bool(
            st.session_state.get(HUMAN_REVIEW_ACKNOWLEDGED_KEY, False)
        ),
        saia_configuration_status=safe_saia_configuration_status(),
    )


def render_evidence_column(context: InvestigationWorkspaceContext) -> None:
    result = context.result
    render_screening_result_summary(result.event)
    render_power_shape_chart(result.reference_cycle, result.candidate_cycle)
    render_event_save_controls(
        result.event,
        DEMO_EVENT_LOG_PATH,
        already_saved=context.current_result_saved,
    )
    render_compact_deterministic_evidence(result.event)


def render_event_save_controls(
    event: MonitoringEvent,
    log_path: Path,
    *,
    already_saved: bool = False,
) -> None:
    st.subheader("Save event")
    st.caption(
        "Saving records a local demo screening event. It does not create a "
        "maintenance ticket or trigger machine action."
    )
    if already_saved:
        if _consume_saved_event_confirmation(event):
            st.success("Saved local demo event for Copilot review.")
        else:
            st.success("This screening event is saved for Copilot review.")
        return
    if st.button("Save event to local demo history"):
        try:
            append_monitoring_event(event, log_path)
            save_event_for_investigation(event, log_path)
        except ValueError as error:
            st.error(str(error))
            return
        st.rerun()


def save_event_for_investigation(
    event: MonitoringEvent,
    log_path: Path,
    *,
    append_event: Callable[[MonitoringEvent, Path], None] | None = None,
) -> None:
    if append_event is not None:
        append_event(event, log_path)
    select_saved_event_for_investigation(event)
    st.session_state[LAST_SAVED_EVENT_CONFIRMATION_KEY] = event.event_id


def _consume_saved_event_confirmation(event: MonitoringEvent) -> bool:
    if st.session_state.get(LAST_SAVED_EVENT_CONFIRMATION_KEY) != event.event_id:
        return False
    del st.session_state[LAST_SAVED_EVENT_CONFIRMATION_KEY]
    return True


def render_compact_deterministic_evidence(event: MonitoringEvent) -> None:
    st.subheader("Deterministic evidence")
    indicators = event.indicators
    st.write(
        "Duration deviation: "
        f"{format_indicator_percentage(indicators.duration_deviation_pct)}"
    )
    st.write(
        "Energy deviation: "
        f"{format_indicator_percentage(indicators.energy_deviation_pct)}"
    )
    st.write(
        "Peak-power deviation: "
        f"{format_indicator_percentage(indicators.peak_power_deviation_pct)}"
    )
    st.write(recommended_follow_up_text(event.recommended_action))


def render_copilot_panel(
    context: InvestigationWorkspaceContext,
    *,
    event_log_path: Path,
) -> None:
    st.subheader("Copilot guidance")
    selected_event = render_saved_event_selector(context.latest_events)
    if selected_event is not None and selected_event != context.selected_event:
        context = build_investigation_workspace_context(
            result=context.result,
            event_log_path=event_log_path,
        )
        selected_event = context.selected_event

    panel_text = build_copilot_panel_text(context.workflow_snapshot)
    st.markdown(f"**Investigation target:** {panel_text.target_label}")
    st.info(panel_text.eligibility_label)

    if can_investigate_event(selected_event):
        can_start = _can_start_copilot_investigation(selected_event)
        if st.button(
            "Investigate with Copilot",
            disabled=not can_start,
        ):
            if not _can_start_copilot_investigation(selected_event):
                st.info(COPILOT_SPINNER_MESSAGE)
                return
            _set_copilot_in_progress(selected_event.event_id)
            try:
                with st.spinner(COPILOT_SPINNER_MESSAGE):
                    result = run_copilot_chat_for_ui(
                        selected_event.event_id,
                        history_path=event_log_path,
                    )
            except Exception:
                st.error(
                    "Copilot investigation could not be prepared from the current "
                    "local history."
                )
                return
            finally:
                _clear_copilot_in_progress(selected_event.event_id)
            _copilot_result_store()[selected_event.event_id] = result
            context = build_investigation_workspace_context(
                result=context.result,
                event_log_path=event_log_path,
            )

    if context.copilot_result is not None:
        render_copilot_result(context.copilot_result)
    else:
        st.caption(panel_text.action_label)
    st.caption(panel_text.feedback_cta)


def render_saved_event_selector(
    latest_events: tuple[MonitoringEvent, ...],
) -> MonitoringEvent | None:
    if not latest_events:
        return None
    labels = {
        event.event_id: (
            f"{event.event_timestamp.isoformat()} | {event.machine_id} | "
            f"{event.status}"
        )
        for event in latest_events
    }
    selected_event_id = st.selectbox(
        "Saved event for Copilot review",
        [event.event_id for event in latest_events],
        format_func=lambda event_id: labels[event_id],
        key=COPILOT_EVENT_SELECTION_KEY,
    )
    return get_selected_persisted_event(latest_events, selected_event_id)


def build_copilot_panel_text(snapshot: object) -> CopilotPanelText:
    screening = getattr(snapshot, "screening", None)
    persistence = getattr(snapshot, "persistence", None)
    investigation = getattr(snapshot, "investigation", None)
    feedback_available = bool(getattr(snapshot, "feedback_available", False))
    feedback_enabled = bool(getattr(snapshot, "feedback_recording_enabled", False))

    if screening == "within_normal_band":
        eligibility = "Copilot is not available for a cycle within the normal band."
        target = "No suspected deviation selected"
        action = "Continue monitoring; no Copilot investigation is needed."
    elif persistence == "save_required":
        eligibility = "Save the suspected deviation before using Copilot."
        target = "Current suspected deviation is not saved"
        action = "Use the save control in the evidence column first."
    elif investigation == "ready_to_investigate":
        eligibility = "Ready to investigate the saved suspected deviation."
        target = "Saved suspected deviation selected"
        action = "Use the Investigate with Copilot button when ready."
    elif investigation == "investigating":
        eligibility = COPILOT_SPINNER_MESSAGE
        target = "Saved suspected deviation selected"
        action = "Copilot investigation is in progress."
    elif investigation == "investigation_available":
        eligibility = "Model-assisted guidance is available for human review."
        target = "Saved suspected deviation selected"
        action = "Review guidance before recording feedback."
    elif investigation == "fallback_guidance_available":
        eligibility = "Local deterministic fallback guidance is available."
        target = "Saved suspected deviation selected"
        action = "Review guidance before recording feedback."
    else:
        eligibility = "Save a suspected deviation before using Copilot."
        target = "No saved suspected deviation selected"
        action = "Complete the prerequisite before investigation."

    feedback_cta = (
        "Feedback is ready to record locally."
        if feedback_enabled
        else "Acknowledge human review before recording feedback."
        if feedback_available
        else "Feedback becomes available after Copilot guidance."
    )
    return CopilotPanelText(
        target_label=target,
        eligibility_label=eligibility,
        action_label=action,
        feedback_cta=feedback_cta,
    )


def render_workspace_tabs(
    context: InvestigationWorkspaceContext,
    event_log_path: Path,
    feedback_log_path: Path,
) -> None:
    evidence_tab, history_tab, knowledge_tab, feedback_tab = st.tabs(
        [
            "Evidence",
            "Similar events / Local history",
            "Knowledge sources",
            "Feedback",
        ]
    )
    with evidence_tab:
        render_calibration_summary(
            calibration_cycle_count=context.result.calibration_cycle_count,
            reference_cycle_id=context.result.reference_cycle_id,
            threshold_result=context.result.threshold_result,
        )
        render_cycle_metrics(context.result.event)
        render_reference_relative_indicators(context.result.event)
        render_technical_details(context.result.event)
    with history_tab:
        render_local_event_history_section(event_log_path)
    with knowledge_tab:
        render_knowledge_sources_tab(context.copilot_result)
    with feedback_tab:
        if context.selected_event is not None and context.copilot_result is not None:
            render_feedback_controls(
                event=context.selected_event,
                result=context.copilot_result,
                feedback_log_path=feedback_log_path,
            )
        else:
            st.info(FEEDBACK_PENDING_MESSAGE)
        render_recent_feedback_section(feedback_log_path)


def render_knowledge_sources_tab(result: CopilotChatResponse | None) -> None:
    source_labels = build_knowledge_source_labels(result)
    if not source_labels:
        st.info(KNOWLEDGE_PENDING_MESSAGE)
        return
    for source_label in source_labels:
        st.write(source_label)


def build_knowledge_source_labels(
    result: CopilotChatResponse | None,
) -> tuple[str, ...]:
    if result is None:
        return ()
    return tuple(
        f"Knowledge source: {source_id}"
        for source_id in result.knowledge_source_ids
        if _is_safe_knowledge_source_id(source_id)
    )


def _is_safe_knowledge_source_id(source_id: str) -> bool:
    lowered = source_id.lower()
    if any(fragment in lowered for fragment in FORBIDDEN_UI_TEXT_FRAGMENTS):
        return False
    return re.fullmatch(r"[a-z0-9_-]{1,80}", source_id) is not None


def render_local_event_history_section(log_path: Path) -> None:
    st.subheader("Recent local demo events")
    try:
        events = read_monitoring_events(log_path, limit=20)
    except ValueError as error:
        st.error(str(error))
        return

    if not events:
        st.info("No local demo events have been saved yet.")
    else:
        st.dataframe(
            build_event_history_frame(events),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Clearing only deletes the local demo JSONL history. It does not affect "
        "calibration, uploaded files, source code, or machines."
    )
    confirm_clear = st.checkbox("Confirm clearing local demo history")
    if st.button("Clear local demo history"):
        if not confirm_clear:
            st.warning("Confirm clearing local demo history before deleting it.")
            return
        clear_monitoring_event_log_for_demo(log_path)
        st.success("Cleared local demo event history.")


def run_copilot_chat_for_ui(
    event_id: str,
    *,
    history_path: Path,
    configuration_status_reader: ConfigurationStatusReader = safe_saia_configuration_status,
    model_resolver: ModelResolver = resolve_optional_saia_model_from_env,
    chat_runner: ChatCopilotRunner = run_chat_copilot,
) -> CopilotChatResponse:
    model = _resolve_optional_copilot_model_for_ui(
        configuration_status_reader=configuration_status_reader,
        model_resolver=model_resolver,
    )
    try:
        return chat_runner(event_id, history_path=history_path, model=model)
    except Exception:
        if model is None:
            raise
        return chat_runner(event_id, history_path=history_path, model=None)


def _resolve_optional_copilot_model_for_ui(
    *,
    configuration_status_reader: ConfigurationStatusReader,
    model_resolver: ModelResolver,
) -> object | None:
    if configuration_status_reader() != "configured":
        return None
    try:
        return model_resolver()
    except Exception:
        return None


def render_copilot_result(result: CopilotChatResponse) -> None:
    st.markdown("**Screening status:** Suspected deviation")
    st.markdown(f"**Guidance type:** {guidance_type_label(result)}")
    st.markdown(result.assistant_message)
    st.info("Human approval required: True")
    st.warning(COPILOT_REVIEW_REMINDER)


def guidance_type_label(result: CopilotChatResponse) -> str:
    if result.execution_mode == "adk":
        return "Model-assisted guidance"
    return "Local deterministic fallback guidance"


def render_feedback_controls(
    *,
    event: MonitoringEvent,
    result: CopilotUiResult,
    feedback_log_path: Path,
) -> None:
    st.markdown("**Human feedback**")
    acknowledged = st.checkbox(
        "I have reviewed this bounded screening summary. No automatic action is taken.",
        key=HUMAN_REVIEW_ACKNOWLEDGED_KEY,
    )
    label_to_value = {
        "Useful": "useful",
        "Needs follow-up": "needs_follow_up",
        "Not useful": "not_useful",
    }
    selected_label = st.radio(
        "Feedback outcome",
        list(label_to_value.keys()),
        horizontal=True,
    )
    if st.button("Record feedback locally"):
        if not can_record_feedback(
            event,
            result,
            human_review_acknowledged=acknowledged,
        ):
            st.warning("Human review acknowledgement is required before feedback.")
            return
        try:
            record = build_copilot_feedback_record(
                event=event,
                result=result,
                feedback_outcome=label_to_value[selected_label],
                human_review_acknowledged=acknowledged,
            )
            append_copilot_feedback_record(record, feedback_log_path)
        except ValueError:
            st.error("Feedback could not be recorded from the current local review.")
            return
        st.success(
            "Feedback recorded locally. It does not alter screening status or "
            "trigger any action."
        )


def render_recent_feedback_section(feedback_log_path: Path) -> None:
    st.markdown("**Recent local Copilot feedback**")
    try:
        feedback_records = read_copilot_feedback_records(feedback_log_path, limit=20)
    except ValueError:
        st.error("Local feedback history could not be read.")
        return

    if feedback_records:
        st.dataframe(
            build_feedback_history_frame(feedback_records),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No local Copilot feedback has been recorded yet.")

    confirm_clear = st.checkbox("Confirm clearing local feedback history")
    if st.button("Clear local feedback history"):
        if not confirm_clear:
            st.warning("Confirm clearing local feedback history before deleting it.")
            return
        clear_copilot_feedback_log_for_demo(feedback_log_path)
        st.success("Cleared local Copilot feedback history.")


def _copilot_result_store() -> dict[str, CopilotChatResponse]:
    existing = st.session_state.get(COPILOT_RESULTS_KEY)
    if not isinstance(existing, dict):
        existing = {}
        st.session_state[COPILOT_RESULTS_KEY] = existing
    return existing


def _copilot_execution_mode(result: CopilotUiResult) -> str:
    if isinstance(result, CopilotChatResponse):
        return result.execution_mode
    return result.trace.execution_mode


def _copilot_in_progress_event_ids() -> set[str]:
    existing = st.session_state.get(COPILOT_IN_PROGRESS_EVENT_IDS_KEY)
    if isinstance(existing, set):
        return existing
    if isinstance(existing, (list, tuple)):
        return {item for item in existing if isinstance(item, str)}
    return set()


def _is_copilot_in_progress(event_id: str) -> bool:
    return event_id in _copilot_in_progress_event_ids()


def _can_start_copilot_investigation(event: MonitoringEvent | None) -> bool:
    return (
        can_investigate_event(event)
        and event is not None
        and not _is_copilot_in_progress(event.event_id)
    )


def _set_copilot_in_progress(event_id: str) -> None:
    event_ids = _copilot_in_progress_event_ids()
    event_ids.add(event_id)
    st.session_state[COPILOT_IN_PROGRESS_EVENT_IDS_KEY] = event_ids


def _clear_copilot_in_progress(event_id: str) -> None:
    event_ids = _copilot_in_progress_event_ids()
    event_ids.discard(event_id)
    st.session_state[COPILOT_IN_PROGRESS_EVENT_IDS_KEY] = event_ids


def render_calibration_summary(
    *,
    calibration_cycle_count: int,
    reference_cycle_id: str,
    threshold_result: ThresholdResult,
) -> None:
    st.subheader("Calibration Summary")
    summary = pd.DataFrame(
        [
            ("Calibration cycle count", calibration_cycle_count),
            ("Reference cycle ID", reference_cycle_id),
            ("Threshold", threshold_result.threshold),
            ("Absolute margin", threshold_result.absolute_margin),
            ("Relative margin", threshold_result.relative_margin),
            ("Applied margin", threshold_result.applied_margin),
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)


def render_screening_result_summary(event: MonitoringEvent) -> None:
    st.subheader("Screening Result")
    status_label = (
        "Within calibrated normal band"
        if event.status == "within_normal_band"
        else "Suspected deviation"
    )
    if event.status == "within_normal_band":
        st.success(status_label)
    else:
        st.warning(status_label)
    metric_column, threshold_column = st.columns(2)
    metric_column.metric(
        "Normalized DTW distance",
        f"{event.normalized_dtw_distance:.6f}",
    )
    threshold_column.metric("Calibrated threshold", f"{event.threshold:.6f}")
    st.write(event.evidence)


def render_screening_result(event: MonitoringEvent) -> None:
    st.subheader("Screening Result")
    status_label = (
        "Within calibrated normal band"
        if event.status == "within_normal_band"
        else "Suspected deviation"
    )
    if event.status == "within_normal_band":
        st.success(status_label)
    else:
        st.warning(status_label)
    st.metric("Normalized DTW distance", f"{event.normalized_dtw_distance:.6f}")
    st.metric("Calibrated threshold", f"{event.threshold:.6f}")
    st.write(event.evidence)


def render_cycle_metrics(event: MonitoringEvent) -> None:
    st.subheader("Cycle Metrics")
    metrics = event.metrics
    table = pd.DataFrame(
        [
            ("Duration seconds", metrics.duration_seconds),
            ("Energy kWh", metrics.energy_kwh),
            ("Average power kW", metrics.average_power_kw),
            ("Peak power kW", metrics.peak_power_kw),
            ("Sample count", metrics.sample_count),
        ],
        columns=["Metric", "Value"],
    )
    st.dataframe(table, hide_index=True, use_container_width=True)


def render_reference_relative_indicators(event: MonitoringEvent) -> None:
    st.subheader("Reference-relative indicators")
    indicators = event.indicators
    table = pd.DataFrame(
        [
            (
                "Duration deviation",
                format_indicator_percentage(indicators.duration_deviation_pct),
            ),
            (
                "Energy deviation",
                format_indicator_percentage(indicators.energy_deviation_pct),
            ),
            (
                "Peak power deviation",
                format_indicator_percentage(indicators.peak_power_deviation_pct),
            ),
        ],
        columns=["Indicator", "Value"],
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.write(recommended_follow_up_text(event.recommended_action))


def render_power_shape_chart(
    reference_cycle: SelectedCycle,
    candidate_cycle: SelectedCycle,
) -> None:
    st.subheader("Power-Shape Comparison")
    chart_data = build_power_comparison_frame(reference_cycle, candidate_cycle)
    figure = px.line(
        chart_data,
        x="elapsed_seconds",
        y="power_kw",
        color="series",
        title="Reference vs candidate power signature",
        labels={
            "elapsed_seconds": "Elapsed time in seconds",
            "power_kw": "Power in kW",
            "series": "Series",
        },
    )
    st.plotly_chart(figure, use_container_width=True)


def render_technical_details(event: MonitoringEvent) -> None:
    with st.expander("Technical details"):
        st.json(event.model_dump(mode="json"))


if __name__ == "__main__":
    main()
