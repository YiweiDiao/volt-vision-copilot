"""Deterministic power-cycle metric calculations."""

from __future__ import annotations

from collections.abc import Sequence

from volt_vision.monitoring.models import CycleMetrics, PowerSample


def compute_cycle_metrics(
    samples: Sequence[PowerSample],
    *,
    cycle_id: str,
) -> CycleMetrics:
    """Return deterministic metrics for one ordered machine cycle.

    Power is measured in kilowatts (kW). Energy is measured in kilowatt-hours
    (kWh), so each timestamp delta is converted from seconds to hours.
    Timestamp-aware trapezoidal integration is used because real sensor samples
    may arrive at irregular intervals; assuming one-second spacing would distort
    energy and average-power calculations.
    """

    if not samples:
        raise ValueError("cycle samples must not be empty")

    machine_id = samples[0].machine_id
    previous_sample = samples[0]
    previous_timestamp = previous_sample.timestamp
    peak_power_kw = samples[0].power_kw
    energy_kwh = 0.0

    if samples[0].power_kw < 0:
        raise ValueError("power_kw must be non-negative")

    for sample in samples[1:]:
        if sample.machine_id != machine_id:
            raise ValueError("cycle samples must belong to exactly one machine")
        if sample.power_kw < 0:
            raise ValueError("power_kw must be non-negative")
        if sample.timestamp == previous_timestamp:
            raise ValueError("cycle samples must not contain duplicate timestamps")
        if sample.timestamp < previous_timestamp:
            raise ValueError("cycle sample timestamps must be strictly increasing")

        delta_hours = (sample.timestamp - previous_sample.timestamp).total_seconds()
        delta_hours /= 3600

        # Trapezoids approximate energy between samples using the average power
        # across the interval, preserving physical units: kW * hours = kWh.
        interval_average_power_kw = (
            previous_sample.power_kw + sample.power_kw
        ) / 2
        energy_kwh += interval_average_power_kw * delta_hours
        peak_power_kw = max(peak_power_kw, sample.power_kw)
        previous_sample = sample
        previous_timestamp = sample.timestamp

    start_timestamp = samples[0].timestamp
    end_timestamp = samples[-1].timestamp
    duration_seconds = (end_timestamp - start_timestamp).total_seconds()
    average_power_kw = (
        energy_kwh / (duration_seconds / 3600) if duration_seconds else peak_power_kw
    )

    return CycleMetrics(
        cycle_id=cycle_id,
        machine_id=machine_id,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        duration_seconds=duration_seconds,
        energy_kwh=energy_kwh,
        average_power_kw=average_power_kw,
        peak_power_kw=peak_power_kw,
        sample_count=len(samples),
    )
