from __future__ import annotations

import inspect
import math
from datetime import UTC, datetime, timedelta

import pytest

import volt_vision.monitoring.indicators as indicators_module
from volt_vision.monitoring.calibration import calibrate_reference_template
from volt_vision.monitoring.cycles import (
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline
from volt_vision.monitoring.indicators import (
    calculate_reference_relative_indicators,
)
from volt_vision.monitoring.metrics import compute_cycle_metrics
from volt_vision.monitoring.models import CycleMetrics, ReferenceRelativeIndicators


START = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def make_metrics(
    cycle_id: str,
    *,
    machine_id: str = "CNC_TEST",
    duration_seconds: float = 100.0,
    energy_kwh: float = 10.0,
    average_power_kw: float = 0.36,
    peak_power_kw: float = 5.0,
    sample_count: int = 2,
) -> CycleMetrics:
    return CycleMetrics(
        cycle_id=cycle_id,
        machine_id=machine_id,
        start_timestamp=START,
        end_timestamp=START + timedelta(seconds=duration_seconds),
        duration_seconds=duration_seconds,
        energy_kwh=energy_kwh,
        average_power_kw=average_power_kw,
        peak_power_kw=peak_power_kw,
        sample_count=sample_count,
    )


def test_constructed_metric_case_preserves_signed_percentages() -> None:
    reference_metrics = make_metrics("reference", duration_seconds=100, energy_kwh=10, peak_power_kw=5)
    candidate_metrics = make_metrics("candidate", duration_seconds=112, energy_kwh=8, peak_power_kw=6)

    result = calculate_reference_relative_indicators(reference_metrics, candidate_metrics)

    assert isinstance(result, ReferenceRelativeIndicators)
    assert result.reference_cycle_id == "reference"
    assert result.candidate_cycle_id == "candidate"
    assert result.duration_deviation_pct == pytest.approx(12.0)
    assert result.energy_deviation_pct == pytest.approx(-20.0)
    assert result.peak_power_deviation_pct == pytest.approx(20.0)


def test_same_metric_values_produce_zero_deviations() -> None:
    reference_metrics = make_metrics("reference")
    candidate_metrics = make_metrics("candidate")

    result = calculate_reference_relative_indicators(reference_metrics, candidate_metrics)

    assert result.duration_deviation_pct == pytest.approx(0.0)
    assert result.energy_deviation_pct == pytest.approx(0.0)
    assert result.peak_power_deviation_pct == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("zero_field", "expected_none_field"),
    [
        ("duration_seconds", "duration_deviation_pct"),
        ("energy_kwh", "energy_deviation_pct"),
        ("peak_power_kw", "peak_power_deviation_pct"),
    ],
)
def test_zero_reference_baselines_return_none(
    zero_field: str,
    expected_none_field: str,
) -> None:
    reference_values = {
        "duration_seconds": 100.0,
        "energy_kwh": 10.0,
        "peak_power_kw": 5.0,
    }
    reference_values[zero_field] = 0.0
    reference_metrics = make_metrics("reference", **reference_values)
    candidate_metrics = make_metrics(
        "candidate",
        duration_seconds=112,
        energy_kwh=8,
        peak_power_kw=6,
    )

    result = calculate_reference_relative_indicators(reference_metrics, candidate_metrics)

    assert getattr(result, expected_none_field) is None
    for field_name in (
        "duration_deviation_pct",
        "energy_deviation_pct",
        "peak_power_deviation_pct",
    ):
        value = getattr(result, field_name)
        if field_name != expected_none_field:
            assert value is not None
            assert math.isfinite(value)


def test_different_machine_ids_raise_clear_error() -> None:
    reference_metrics = make_metrics("reference", machine_id="CNC_A")
    candidate_metrics = make_metrics("candidate", machine_id="CNC_B")

    with pytest.raises(ValueError, match="same machine_id"):
        calculate_reference_relative_indicators(reference_metrics, candidate_metrics)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_source_metric_values_are_rejected(bad_value: float) -> None:
    reference_metrics = make_metrics("reference")
    candidate_metric_values = make_metrics("candidate").model_dump()
    candidate_metric_values["energy_kwh"] = bad_value
    candidate_metrics = CycleMetrics.model_construct(**candidate_metric_values)

    with pytest.raises(ValueError, match="candidate metric energy_kwh must be finite"):
        calculate_reference_relative_indicators(reference_metrics, candidate_metrics)


def test_synthetic_integration_produces_explanatory_indicators() -> None:
    timeline = generate_demo_timeline(start_timestamp=START)
    calibration_cycles = select_calibration_cycles(timeline)
    calibration = calibrate_reference_template(calibration_cycles)
    normal_candidate = calibration_cycles[0]
    changed_candidate = select_abnormal_evaluation_cycle(timeline)

    reference_metrics = compute_cycle_metrics(
        calibration.reference_cycle.samples,
        cycle_id=calibration.reference_cycle.segment_id,
    )
    normal_metrics = compute_cycle_metrics(
        normal_candidate.samples,
        cycle_id=normal_candidate.segment_id,
    )
    changed_metrics = compute_cycle_metrics(
        changed_candidate.samples,
        cycle_id=changed_candidate.segment_id,
    )

    normal_indicators = calculate_reference_relative_indicators(
        reference_metrics,
        normal_metrics,
    )
    changed_indicators = calculate_reference_relative_indicators(
        reference_metrics,
        changed_metrics,
    )

    for result in (normal_indicators, changed_indicators):
        for value in result.model_dump().values():
            if isinstance(value, float):
                assert math.isfinite(value)
    assert changed_indicators.duration_deviation_pct is not None
    assert changed_indicators.energy_deviation_pct is not None
    assert changed_indicators.peak_power_deviation_pct is not None
    assert changed_indicators.duration_deviation_pct > 0
    assert changed_indicators.energy_deviation_pct > 0
    assert changed_indicators.peak_power_deviation_pct > 0


def test_indicator_module_has_no_decision_coupling() -> None:
    source = inspect.getsource(indicators_module)

    assert "ThresholdResult" not in source
    assert "CycleEvaluation" not in source
    assert "status" not in source
    assert "expected_label" not in source
    assert "anomaly" not in source
