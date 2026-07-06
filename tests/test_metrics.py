from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from volt_vision.monitoring.metrics import compute_cycle_metrics
from volt_vision.monitoring.models import PowerSample


BASE_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def sample(offset_seconds: int, power_kw: float, machine_id: str = "cnc-1") -> PowerSample:
    return PowerSample(
        timestamp=BASE_TIME + timedelta(seconds=offset_seconds),
        machine_id=machine_id,
        power_kw=power_kw,
    )


def test_regular_one_second_samples_compute_metrics() -> None:
    metrics = compute_cycle_metrics(
        [sample(0, 1.0), sample(1, 3.0), sample(2, 5.0)],
        cycle_id="cycle-1",
    )

    assert metrics.cycle_id == "cycle-1"
    assert metrics.machine_id == "cnc-1"
    assert metrics.start_timestamp == BASE_TIME
    assert metrics.end_timestamp == BASE_TIME + timedelta(seconds=2)
    assert metrics.duration_seconds == 2
    assert metrics.sample_count == 3


def test_irregular_intervals_use_expected_trapezoidal_energy() -> None:
    metrics = compute_cycle_metrics(
        [sample(0, 2.0), sample(10, 4.0), sample(40, 8.0)],
        cycle_id="cycle-irregular",
    )

    expected_kwh = ((2.0 + 4.0) / 2 * (10 / 3600)) + (
        (4.0 + 8.0) / 2 * (30 / 3600)
    )
    assert metrics.duration_seconds == 40
    assert metrics.energy_kwh == pytest.approx(expected_kwh)


def test_peak_and_time_weighted_average_power() -> None:
    metrics = compute_cycle_metrics(
        [sample(0, 2.0), sample(10, 6.0), sample(20, 2.0)],
        cycle_id="cycle-average",
    )

    assert metrics.peak_power_kw == 6.0
    assert metrics.average_power_kw == pytest.approx(4.0)


def test_empty_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_cycle_metrics([], cycle_id="empty")


def test_mixed_machine_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one machine"):
        compute_cycle_metrics(
            [sample(0, 1.0, "cnc-1"), sample(1, 1.0, "cnc-2")],
            cycle_id="mixed",
        )


def test_duplicate_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate timestamps"):
        compute_cycle_metrics([sample(0, 1.0), sample(0, 2.0)], cycle_id="duplicate")


def test_non_increasing_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_cycle_metrics([sample(2, 1.0), sample(1, 2.0)], cycle_id="backward")


def test_negative_power_is_rejected_by_model() -> None:
    with pytest.raises(ValidationError):
        sample(0, -0.1)


def test_negative_power_is_rejected_by_metrics_function() -> None:
    bad_sample = PowerSample.model_construct(
        timestamp=BASE_TIME,
        machine_id="cnc-1",
        power_kw=-0.1,
    )

    with pytest.raises(ValueError, match="non-negative"):
        compute_cycle_metrics([bad_sample], cycle_id="negative")
