from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from volt_vision.guidance.retrieval import retrieve_guidance
from volt_vision.mcp_server.services import (
    EVENT_NOT_FOUND_MESSAGE,
    SIMILARITY_RANKING_UNAVAILABLE_MESSAGE,
    EventNotFoundError,
    SimilarityRankingError,
    find_similar_previous_events,
    get_event_metrics,
    get_monitoring_event,
    retrieve_maintenance_guidance,
)
from volt_vision.monitoring.event_log import append_monitoring_event
from volt_vision.monitoring.models import (
    CycleMetrics,
    MonitoringEvent,
    ReferenceRelativeIndicators,
)


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def make_event(
    event_id: str,
    *,
    seconds: int,
    machine_id: str = "CNC_TEST",
    status: str = "suspected_deviation",
    normalized_dtw_distance: float = 0.12,
    threshold: float = 0.10,
    duration_deviation_pct: float | None = 10.0,
    energy_deviation_pct: float | None = 20.0,
    peak_power_deviation_pct: float | None = 30.0,
) -> MonitoringEvent:
    event_timestamp = START + timedelta(seconds=seconds)
    action = (
        "no_automated_action"
        if status == "within_normal_band"
        else "manual_review_required"
    )
    return MonitoringEvent(
        event_id=event_id,
        event_type="cycle_screening",
        event_timestamp=event_timestamp,
        machine_id=machine_id,
        candidate_segment_id=f"{event_id}-candidate",
        reference_segment_id="reference",
        status=status,
        recommended_action=action,
        evidence="Normalized DTW distance compared with calibrated threshold.",
        normalized_dtw_distance=normalized_dtw_distance,
        threshold=threshold,
        metrics=CycleMetrics(
            cycle_id=f"{event_id}-candidate",
            machine_id=machine_id,
            start_timestamp=event_timestamp - timedelta(seconds=60),
            end_timestamp=event_timestamp,
            duration_seconds=60,
            energy_kwh=0.25,
            average_power_kw=15,
            peak_power_kw=20,
            sample_count=2,
        ),
        indicators=ReferenceRelativeIndicators(
            reference_cycle_id="reference",
            candidate_cycle_id=f"{event_id}-candidate",
            duration_deviation_pct=duration_deviation_pct,
            energy_deviation_pct=energy_deviation_pct,
            peak_power_deviation_pct=peak_power_deviation_pct,
        ),
    )


def write_events(log_path: Path, *events: MonitoringEvent) -> None:
    for event in events:
        append_monitoring_event(event, log_path)


def test_event_lookup_returns_latest_appended_duplicate_event_id(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    old = make_event("duplicate", seconds=60, normalized_dtw_distance=0.11)
    latest = make_event("duplicate", seconds=120, normalized_dtw_distance=0.22)
    write_events(log_path, old, latest)

    found = get_monitoring_event("duplicate", log_path)

    assert found == latest


def test_unknown_event_id_raises_safe_domain_error(tmp_path: Path) -> None:
    log_path = tmp_path / "history.jsonl"
    write_events(log_path, make_event("known", seconds=60))

    with pytest.raises(EventNotFoundError, match=EVENT_NOT_FOUND_MESSAGE):
        get_event_metrics("unknown", log_path)


def test_get_event_metrics_returns_structured_data_without_raw_samples(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    event = make_event("query", seconds=60)
    write_events(log_path, event)

    result = get_event_metrics("query", log_path).model_dump(mode="json")

    assert result["event_id"] == "query"
    assert result["status"] == "suspected_deviation"
    assert result["recommended_action"] == "manual_review_required"
    assert result["normalized_dtw_distance"] == event.normalized_dtw_distance
    assert result["threshold"] == event.threshold
    assert result["metrics"]["cycle_id"] == event.metrics.cycle_id
    assert result["indicators"]["duration_deviation_pct"] == 10.0
    assert "samples" not in result
    assert "power_samples" not in result
    assert "raw" not in result


def test_retrieve_maintenance_guidance_matches_stable_day_3_order(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    event = make_event("query", seconds=60)
    write_events(log_path, event)

    service_ids = tuple(
        item.guidance_id for item in retrieve_maintenance_guidance("query", log_path)
    )
    direct_ids = tuple(item.guidance_id for item in retrieve_guidance(event))

    assert service_ids == direct_ids


def test_similar_events_filter_same_machine_same_status_and_earlier_only(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    prior_match = make_event("prior-match", seconds=40)
    same_machine_other_status = make_event(
        "other-status",
        seconds=45,
        status="within_normal_band",
    )
    other_machine = make_event("other-machine", seconds=50, machine_id="CNC_OTHER")
    later = make_event("later", seconds=90)
    query = make_event("query", seconds=80)
    write_events(log_path, prior_match, same_machine_other_status, other_machine, query, later)

    results = find_similar_previous_events("query", history_path=log_path)

    assert tuple(item.event_id for item in results) == ("prior-match",)


def test_query_event_is_excluded_from_similar_results(tmp_path: Path) -> None:
    log_path = tmp_path / "history.jsonl"
    prior = make_event("prior", seconds=40)
    query_old_duplicate = make_event("query", seconds=50)
    query = make_event("query", seconds=80)
    write_events(log_path, prior, query_old_duplicate, query)

    results = find_similar_previous_events("query", history_path=log_path)

    assert tuple(item.event_id for item in results) == ("prior",)


def test_duplicate_historical_ids_are_collapsed_before_ranking(tmp_path: Path) -> None:
    log_path = tmp_path / "history.jsonl"
    duplicate_old = make_event("historical", seconds=20, normalized_dtw_distance=0.10)
    duplicate_latest = make_event("historical", seconds=40, normalized_dtw_distance=0.12)
    query = make_event("query", seconds=80, normalized_dtw_distance=0.12)
    write_events(log_path, duplicate_old, duplicate_latest, query)

    results = find_similar_previous_events("query", history_path=log_path)

    assert len(results) == 1
    assert results[0].event_id == "historical"
    assert results[0].event_timestamp == duplicate_latest.event_timestamp


def test_latest_historical_duplicate_after_query_excludes_event_id(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    included = make_event("included", seconds=30)
    historical_old = make_event("historical", seconds=40)
    query = make_event("query", seconds=80)
    historical_latest_after_query = make_event("historical", seconds=90)
    write_events(
        log_path,
        historical_old,
        included,
        query,
        historical_latest_after_query,
    )

    results = find_similar_previous_events("query", limit=5, history_path=log_path)

    assert tuple(item.event_id for item in results) == ("included",)


def test_latest_historical_duplicate_before_query_is_ranked_and_returned(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    historical_old = make_event(
        "historical",
        seconds=20,
        normalized_dtw_distance=0.10,
    )
    historical_latest = make_event(
        "historical",
        seconds=50,
        normalized_dtw_distance=0.11,
        energy_deviation_pct=21.0,
    )
    query = make_event("query", seconds=80)
    write_events(log_path, historical_old, historical_latest, query)

    results = find_similar_previous_events("query", history_path=log_path)

    assert len(results) == 1
    assert results[0].event_id == "historical"
    assert results[0].event_timestamp == historical_latest.event_timestamp
    assert results[0].normalized_dtw_distance == 0.11
    assert results[0].indicators.energy_deviation_pct == 21.0


def test_latest_query_duplicate_is_used_for_filtering_and_ranking(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    prior = make_event("prior", seconds=70, machine_id="CNC_LATEST")
    query_old = make_event("query", seconds=80, machine_id="CNC_OLD")
    old_machine_prior = make_event("old-machine-prior", seconds=90, machine_id="CNC_OLD")
    query_latest = make_event("query", seconds=100, machine_id="CNC_LATEST")
    write_events(log_path, prior, query_old, old_machine_prior, query_latest)

    results = find_similar_previous_events("query", limit=5, history_path=log_path)

    assert tuple(item.event_id for item in results) == ("prior",)


def test_zero_threshold_with_nonzero_distance_raises_safe_similarity_error(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "history.jsonl"
    prior = make_event(
        "prior",
        seconds=40,
        normalized_dtw_distance=0.01,
        threshold=0.0,
    )
    query = make_event("query", seconds=80)
    write_events(log_path, prior, query)

    with pytest.raises(
        SimilarityRankingError,
        match=SIMILARITY_RANKING_UNAVAILABLE_MESSAGE,
    ):
        find_similar_previous_events("query", history_path=log_path)


def test_ranking_score_behavior_is_deterministic(tmp_path: Path) -> None:
    log_path = tmp_path / "history.jsonl"
    prior = make_event(
        "prior",
        seconds=40,
        normalized_dtw_distance=0.10,
        threshold=0.10,
        duration_deviation_pct=20.0,
        energy_deviation_pct=25.0,
        peak_power_deviation_pct=None,
    )
    query = make_event(
        "query",
        seconds=80,
        normalized_dtw_distance=0.12,
        threshold=0.10,
        duration_deviation_pct=10.0,
        energy_deviation_pct=20.0,
        peak_power_deviation_pct=None,
    )
    write_events(log_path, prior, query)

    result = find_similar_previous_events("query", history_path=log_path)[0]

    expected = ((1.2 - 1.0) + (10.0 / 100) + (5.0 / 100)) / 3
    assert result.ranking_score == pytest.approx(expected)
    assert "does not confirm root cause" in result.ranking_note


def test_stable_tie_breaking_uses_score_recency_then_event_id(tmp_path: Path) -> None:
    log_path = tmp_path / "history.jsonl"
    query = make_event("query", seconds=100)
    newest_equal = make_event("b-newest", seconds=70, energy_deviation_pct=21.0)
    older_a = make_event("a-older", seconds=50, energy_deviation_pct=21.0)
    older_b = make_event("b-older", seconds=50, energy_deviation_pct=21.0)
    closest = make_event("z-closest", seconds=20)
    write_events(log_path, older_b, newest_equal, closest, older_a, query)

    results = find_similar_previous_events("query", limit=5, history_path=log_path)

    assert tuple(item.event_id for item in results) == (
        "z-closest",
        "b-newest",
        "a-older",
        "b-older",
    )


def test_missing_history_returns_empty_similar_results(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"

    assert find_similar_previous_events("query", history_path=missing_path) == ()


@pytest.mark.parametrize("limit", [1, 5])
def test_limit_accepts_one_through_five(tmp_path: Path, limit: int) -> None:
    log_path = tmp_path / "history.jsonl"
    write_events(log_path, make_event("query", seconds=60))

    find_similar_previous_events("query", limit=limit, history_path=log_path)


@pytest.mark.parametrize("limit", [0, 6, 1.5, "3", None, True, False])
def test_limit_rejects_out_of_range_and_non_integer_values(
    tmp_path: Path,
    limit: object,
) -> None:
    log_path = tmp_path / "history.jsonl"
    write_events(log_path, make_event("query", seconds=60))

    with pytest.raises(ValueError, match="limit must be an integer from 1 through 5"):
        find_similar_previous_events("query", limit=limit, history_path=log_path)  # type: ignore[arg-type]


def test_service_functions_do_not_modify_history_file(tmp_path: Path) -> None:
    log_path = tmp_path / "history.jsonl"
    prior = make_event("prior", seconds=40)
    query = make_event("query", seconds=80)
    write_events(log_path, prior, query)
    before = log_path.read_bytes()

    get_event_metrics("query", log_path)
    retrieve_maintenance_guidance("query", log_path)
    find_similar_previous_events("query", history_path=log_path)

    assert log_path.read_bytes() == before
