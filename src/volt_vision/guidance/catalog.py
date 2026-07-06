"""Stable local catalog of curated safety-bounded guidance."""

from __future__ import annotations

from volt_vision.guidance.models import GuidanceItem


POWER_SIGNATURE_REVIEW = GuidanceItem(
    guidance_id="power_signature_review",
    title="Power Signature Review",
    applies_to=("all cycle screening events",),
    inspection_checks=(
        "Compare candidate and reference power signatures.",
        "Confirm the selected cycle, timestamps, and machine identifier are plausible.",
        "Check whether similar structured events recur.",
    ),
    evidence_to_record=(
        "normalized DTW distance",
        "calibrated threshold",
        "screening status",
        "recurrence observations",
    ),
    escalation_condition="Escalate only under local procedures after manual review.",
    safety_note="This review does not identify root cause or confirm a fault.",
    possible_contributing_conditions=(
        "Cycle selection, timestamp alignment, or reference comparison may need verification.",
        "Process context may differ from the reviewed reference cycle.",
        "A recurring power-signature difference may need authorized local review.",
    ),
    operator_questions=(
        "Was the intended complete cycle selected for comparison?",
        "Did the workpiece, material, fixture, or operator context differ locally?",
        "Have similar suspected deviations been recorded for the same machine?",
    ),
    safe_inspection_checks=(
        "Verify the selected cycle, timestamps, and machine identifier.",
        "Inspect the selected cycle context according to local SOP.",
        "Compare the candidate and reference power signatures using reviewed evidence.",
        "Record local observations according to local SOP.",
    ),
    do_not_conclude=(
        "Do not conclude a confirmed fault.",
        "Do not conclude a confirmed root cause.",
        "Do not conclude maintenance necessity from the power signature alone.",
    ),
    reviewer_label="Expert-reviewed demo knowledge",
    knowledge_version="2026-07-05",
)

CYCLE_DURATION_REVIEW = GuidanceItem(
    guidance_id="cycle_duration_review",
    title="Cycle Duration Review",
    applies_to=("suspected deviations with duration deviation at or above 10 percent",),
    inspection_checks=(
        "Review cycle boundaries and waiting or idle time.",
        "Review material, workpiece batch, and operating context.",
        "Check whether the duration pattern recurs.",
    ),
    evidence_to_record=(
        "duration deviation percentage",
        "human observations",
        "production context",
        "recurrence observations",
    ),
    escalation_condition="Escalate when recurrence or production impact requires local review.",
    safety_note="Duration evidence alone does not show that a component is faulty.",
    possible_contributing_conditions=(
        "Cycle boundary selection or waiting time may need verification.",
        "Workpiece, batch, or operator context may differ from the reference.",
        "A recurring duration shift may need authorized local review.",
    ),
    operator_questions=(
        "Was there waiting, rework, or a pause during the reviewed cycle?",
        "Did the material, batch, or process plan differ locally?",
        "Has the duration pattern repeated in structured event history?",
    ),
    safe_inspection_checks=(
        "Verify cycle boundaries and timestamp coverage.",
        "Compare duration evidence with local production context.",
        "Record duration observations according to local SOP.",
    ),
    do_not_conclude=(
        "Do not conclude a faulty component from duration evidence.",
        "Do not conclude tool wear from duration evidence.",
        "Do not conclude maintenance necessity from duration evidence.",
    ),
    reviewer_label="Expert-reviewed demo knowledge",
    knowledge_version="2026-07-05",
)

ENERGY_AND_PEAK_REVIEW = GuidanceItem(
    guidance_id="energy_and_peak_review",
    title="Energy and Peak Review",
    applies_to=(
        "suspected deviations with energy or peak-power deviation at or above 10 percent",
    ),
    inspection_checks=(
        "Review load, material, workpiece, and fixturing context.",
        "Review auxiliary loads and operating context.",
        "Check whether the energy or peak-power pattern recurs.",
    ),
    evidence_to_record=(
        "energy deviation percentage",
        "peak-power deviation percentage",
        "human observations",
        "recurrence observations",
    ),
    escalation_condition="Escalate when recurrence or production impact requires local review.",
    safety_note="Energy and peak evidence do not confirm tool wear, fault, or root cause.",
    possible_contributing_conditions=(
        "Load, material, workpiece, or fixturing context may need verification.",
        "Auxiliary loads or operating context may differ from the reference.",
        "A recurring energy or peak-power pattern may need authorized local review.",
    ),
    operator_questions=(
        "Did load, material, workpiece, or fixturing context differ locally?",
        "Were auxiliary loads or operating conditions different during the cycle?",
        "Has a similar energy or peak-power pattern been recorded before?",
    ),
    safe_inspection_checks=(
        "Verify load and workpiece context according to local SOP.",
        "Compare energy and peak-power evidence with reference-relative indicators.",
        "Record observations and recurrence context for authorized review.",
    ),
    do_not_conclude=(
        "Do not conclude confirmed tool wear.",
        "Do not conclude a confirmed equipment defect.",
        "Do not conclude maintenance necessity.",
    ),
    reviewer_label="Expert-reviewed demo knowledge",
    knowledge_version="2026-07-05",
)

ESCALATION_AND_RECORDING = GuidanceItem(
    guidance_id="escalation_and_recording",
    title="Escalation and Recording",
    applies_to=("all cycle screening events",),
    inspection_checks=(
        "Record structured event details and human observations.",
        "Escalate only when recurrence, production impact, or local procedure requires it.",
        "Keep any follow-up under authorized human control.",
    ),
    evidence_to_record=(
        "event ID",
        "event timestamp",
        "status",
        "normalized DTW distance",
        "threshold",
        "duration, energy, and peak-power deviations",
        "human observations",
    ),
    escalation_condition="Do not open tickets automatically.",
    safety_note="Only authorized representatives may decide real-world actions.",
    possible_contributing_conditions=(
        "Recurrence, production impact, or local procedure may affect escalation.",
        "Missing local observations may limit review quality.",
        "A suspected deviation may need comparison with recent structured events.",
    ),
    operator_questions=(
        "Is there recurring structured evidence for the same machine and status?",
        "Is there production impact that local procedures require staff to review?",
        "What observations should be recorded before authorized escalation?",
    ),
    safe_inspection_checks=(
        "Record structured event details and local observations.",
        "Compare recurrence context before escalation.",
        "Escalate only according to local SOP and authorized human review.",
    ),
    do_not_conclude=(
        "Do not conclude that a ticket is required automatically.",
        "Do not conclude that machine action is authorized.",
        "Do not conclude diagnosis or maintenance necessity.",
    ),
    reviewer_label="Expert-reviewed demo knowledge",
    knowledge_version="2026-07-05",
)

GUIDANCE_CATALOG = (
    POWER_SIGNATURE_REVIEW,
    CYCLE_DURATION_REVIEW,
    ENERGY_AND_PEAK_REVIEW,
    ESCALATION_AND_RECORDING,
)

GUIDANCE_BY_ID = {item.guidance_id: item for item in GUIDANCE_CATALOG}

__all__ = [
    "ESCALATION_AND_RECORDING",
    "ENERGY_AND_PEAK_REVIEW",
    "CYCLE_DURATION_REVIEW",
    "GUIDANCE_BY_ID",
    "GUIDANCE_CATALOG",
    "POWER_SIGNATURE_REVIEW",
]
