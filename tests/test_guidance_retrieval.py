from __future__ import annotations

from datetime import UTC, datetime, timedelta
from dataclasses import FrozenInstanceError

import pytest
from volt_vision.guidance.catalog import GUIDANCE_CATALOG
from volt_vision.guidance.retrieval import retrieve_guidance
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def make_event(
    *,
    status: str = "within_normal_band",
    duration_deviation_pct: float | None = 0.0,
    energy_deviation_pct: float | None = 0.0,
    peak_power_deviation_pct: float | None = 0.0,
) -> MonitoringEvent:
    return MonitoringEvent(
        event_id="event-1",
        event_type="cycle_screening",
        event_timestamp=START + timedelta(seconds=60),
        machine_id="CNC_TEST",
        candidate_segment_id="candidate",
        reference_segment_id="reference",
        status=status,
        recommended_action=(
            "no_automated_action"
            if status == "within_normal_band"
            else "manual_review_required"
        ),
        evidence="Normalized DTW distance compared with calibrated threshold.",
        normalized_dtw_distance=0.05,
        threshold=0.10,
        metrics=CycleMetrics(
            cycle_id="candidate",
            machine_id="CNC_TEST",
            start_timestamp=START,
            end_timestamp=START + timedelta(seconds=60),
            duration_seconds=60,
            energy_kwh=0.25,
            average_power_kw=15,
            peak_power_kw=20,
            sample_count=2,
        ),
        indicators=ReferenceRelativeIndicators(
            reference_cycle_id="reference",
            candidate_cycle_id="candidate",
            duration_deviation_pct=duration_deviation_pct,
            energy_deviation_pct=energy_deviation_pct,
            peak_power_deviation_pct=peak_power_deviation_pct,
        ),
    )


def guidance_ids(event: MonitoringEvent) -> tuple[str, ...]:
    return tuple(item.guidance_id for item in retrieve_guidance(event))


def test_within_normal_band_returns_general_guidance_in_stable_order() -> None:
    event = make_event(
        status="within_normal_band",
        duration_deviation_pct=100.0,
        energy_deviation_pct=100.0,
        peak_power_deviation_pct=100.0,
    )

    assert guidance_ids(event) == (
        "power_signature_review",
        "escalation_and_recording",
    )


def test_suspected_deviation_without_triggered_indicators_returns_general_items() -> None:
    event = make_event(
        status="suspected_deviation",
        duration_deviation_pct=9.999,
        energy_deviation_pct=-9.999,
        peak_power_deviation_pct=0.0,
    )

    assert guidance_ids(event) == (
        "power_signature_review",
        "escalation_and_recording",
    )


def test_duration_boundary_below_ten_does_not_trigger() -> None:
    event = make_event(status="suspected_deviation", duration_deviation_pct=9.999)

    assert "cycle_duration_review" not in guidance_ids(event)


def test_duration_boundary_positive_ten_triggers() -> None:
    event = make_event(status="suspected_deviation", duration_deviation_pct=10.0)

    assert guidance_ids(event) == (
        "power_signature_review",
        "cycle_duration_review",
        "escalation_and_recording",
    )


def test_duration_boundary_negative_ten_triggers() -> None:
    event = make_event(status="suspected_deviation", duration_deviation_pct=-10.0)

    assert guidance_ids(event) == (
        "power_signature_review",
        "cycle_duration_review",
        "escalation_and_recording",
    )


def test_energy_boundary_positive_ten_triggers_energy_and_peak_review() -> None:
    event = make_event(status="suspected_deviation", energy_deviation_pct=10.0)

    assert guidance_ids(event) == (
        "power_signature_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    )


def test_peak_boundary_negative_ten_triggers_energy_and_peak_review() -> None:
    event = make_event(status="suspected_deviation", peak_power_deviation_pct=-10.0)

    assert guidance_ids(event) == (
        "power_signature_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    )


def test_energy_and_peak_together_return_energy_and_peak_review_once() -> None:
    event = make_event(
        status="suspected_deviation",
        energy_deviation_pct=10.0,
        peak_power_deviation_pct=-10.0,
    )

    ids = guidance_ids(event)

    assert ids.count("energy_and_peak_review") == 1
    assert ids == (
        "power_signature_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    )


def test_none_values_do_not_trigger_specialized_guidance() -> None:
    event = make_event(
        status="suspected_deviation",
        duration_deviation_pct=None,
        energy_deviation_pct=None,
        peak_power_deviation_pct=None,
    )

    assert guidance_ids(event) == (
        "power_signature_review",
        "escalation_and_recording",
    )


def test_final_ordering_is_stable_when_all_specialized_guidance_triggers() -> None:
    event = make_event(
        status="suspected_deviation",
        duration_deviation_pct=-10.0,
        energy_deviation_pct=10.0,
        peak_power_deviation_pct=10.0,
    )

    assert guidance_ids(event) == (
        "power_signature_review",
        "cycle_duration_review",
        "energy_and_peak_review",
        "escalation_and_recording",
    )


def test_retrieval_does_not_mutate_input_event() -> None:
    event = make_event(status="suspected_deviation", duration_deviation_pct=10.0)
    before = event.model_dump()

    retrieve_guidance(event)

    assert event.model_dump() == before


def test_catalog_ids_are_unique() -> None:
    ids = [item.guidance_id for item in GUIDANCE_CATALOG]

    assert len(ids) == len(set(ids))


def test_all_guidance_items_have_non_empty_required_fields() -> None:
    for item in GUIDANCE_CATALOG:
        assert item.guidance_id
        assert item.title
        assert item.applies_to
        assert all(value for value in item.applies_to)
        assert item.inspection_checks
        assert all(value for value in item.inspection_checks)
        assert item.evidence_to_record
        assert all(value for value in item.evidence_to_record)
        assert item.escalation_condition
        assert item.safety_note
        assert item.possible_contributing_conditions
        assert all(value for value in item.possible_contributing_conditions)
        assert item.operator_questions
        assert all(value for value in item.operator_questions)
        assert item.safe_inspection_checks
        assert all(value for value in item.safe_inspection_checks)
        assert item.do_not_conclude
        assert all(value for value in item.do_not_conclude)
        assert item.reviewer_label
        assert item.knowledge_version


def test_guidance_items_are_immutable() -> None:
    item = GUIDANCE_CATALOG[0]

    with pytest.raises(FrozenInstanceError):
        item.reviewer_label = "changed"  # type: ignore[misc]


def test_extended_guidance_fields_use_safe_non_diagnostic_language() -> None:
    unsafe_fragments = (
        "confirmed fault",
        "confirmed failure",
        "root cause is",
        "replace",
        "repair",
        "shutdown",
        "parameter tuning",
        "open a ticket",
        "create a ticket",
        "machine control",
    )
    safe_required_fragments = (
        "verify",
        "compare",
        "inspect",
        "record",
        "escalate",
        "may need",
    )
    guidance_text = " ".join(
        " ".join(
            (
                *item.possible_contributing_conditions,
                *item.operator_questions,
                *item.safe_inspection_checks,
            )
        ).lower()
        for item in GUIDANCE_CATALOG
    )
    do_not_conclude_text = " ".join(
        " ".join(item.do_not_conclude).lower() for item in GUIDANCE_CATALOG
    )

    for fragment in unsafe_fragments:
        assert fragment not in guidance_text
    for fragment in safe_required_fragments:
        assert fragment in guidance_text
    for item in GUIDANCE_CATALOG:
        assert all(value.startswith("Do not conclude") for value in item.do_not_conclude)
    assert "maintenance necessity" in do_not_conclude_text
