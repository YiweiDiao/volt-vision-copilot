"""Synthetic CNC-inspired power data for the educational demo prototype.

The generated signal is deterministic demonstration data only. It is not
industrially validated fault data and must not be treated as evidence of a
confirmed machine fault, tool wear, or root cause.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from volt_vision.monitoring.models import PowerSample

DEFAULT_MACHINE_ID = "CNC_01"
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1
DEFAULT_START_TIMESTAMP = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

SegmentType = Literal["idle", "normal_cycle", "abnormal_cycle"]
ExpectedLabel = Literal["idle", "normal", "abnormal"]


@dataclass(frozen=True)
class DemoSegment:
    segment_id: str
    segment_type: SegmentType
    start_timestamp: datetime
    end_timestamp: datetime
    expected_label: ExpectedLabel


@dataclass(frozen=True)
class DemoTimeline:
    samples: list[PowerSample]
    segments: list[DemoSegment]


def generate_idle_segment(
    start_timestamp: datetime,
    *,
    duration_seconds: int = 20,
    sample_interval_seconds: int = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    machine_id: str = DEFAULT_MACHINE_ID,
    rng: random.Random | None = None,
) -> list[PowerSample]:
    """Generate low non-negative idle baseline power samples."""

    generator = rng if rng is not None else random.Random(0)
    _validate_generation_inputs(start_timestamp, duration_seconds, sample_interval_seconds)

    samples: list[PowerSample] = []
    for index, timestamp in enumerate(
        _iter_timestamps(start_timestamp, duration_seconds, sample_interval_seconds)
    ):
        slow_drift_kw = 0.03 * math.sin(index / 5)
        bounded_noise_kw = generator.uniform(-0.025, 0.025)
        power_kw = max(0.0, 0.45 + slow_drift_kw + bounded_noise_kw)
        samples.append(_sample(timestamp, machine_id, power_kw))
    return samples


def generate_normal_cycle(
    start_timestamp: datetime,
    *,
    duration_seconds: int = 56,
    sample_interval_seconds: int = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    machine_id: str = DEFAULT_MACHINE_ID,
    rng: random.Random | None = None,
) -> list[PowerSample]:
    """Generate one normal CNC-style machining cycle."""

    generator = rng if rng is not None else random.Random(0)
    _validate_generation_inputs(start_timestamp, duration_seconds, sample_interval_seconds)

    plateau_kw = generator.uniform(3.1, 3.45)
    ramp_peak_kw = plateau_kw + generator.uniform(0.1, 0.25)
    samples: list[PowerSample] = []

    for timestamp in _iter_timestamps(
        start_timestamp, duration_seconds, sample_interval_seconds
    ):
        elapsed_seconds = (timestamp - start_timestamp).total_seconds()
        progress = elapsed_seconds / duration_seconds
        power_kw = _normal_cycle_power(progress, plateau_kw, ramp_peak_kw, generator)
        samples.append(_sample(timestamp, machine_id, power_kw))
    return samples


def generate_abnormal_cycle(
    start_timestamp: datetime,
    *,
    duration_seconds: int = 88,
    sample_interval_seconds: int = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    machine_id: str = DEFAULT_MACHINE_ID,
    rng: random.Random | None = None,
) -> list[PowerSample]:
    """Generate one visibly deviating synthetic machining cycle."""

    generator = rng if rng is not None else random.Random(0)
    _validate_generation_inputs(start_timestamp, duration_seconds, sample_interval_seconds)

    plateau_kw = generator.uniform(3.9, 4.25)
    samples: list[PowerSample] = []

    for timestamp in _iter_timestamps(
        start_timestamp, duration_seconds, sample_interval_seconds
    ):
        elapsed_seconds = (timestamp - start_timestamp).total_seconds()
        progress = elapsed_seconds / duration_seconds
        power_kw = _abnormal_cycle_power(progress, plateau_kw, generator)
        samples.append(_sample(timestamp, machine_id, power_kw))
    return samples


def generate_demo_timeline(
    *,
    seed: int = 7,
    start_timestamp: datetime = DEFAULT_START_TIMESTAMP,
    machine_id: str = DEFAULT_MACHINE_ID,
    sample_interval_seconds: int = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> DemoTimeline:
    """Return the full deterministic demo timeline for one machine."""

    _validate_generation_inputs(start_timestamp, 1, sample_interval_seconds)
    generator = random.Random(seed)
    current_timestamp = start_timestamp
    all_samples: list[PowerSample] = []
    segments: list[DemoSegment] = []

    plan: tuple[tuple[SegmentType, int], ...] = (
        ("idle", 18),
        ("normal_cycle", 56),
        ("idle", 14),
        ("normal_cycle", 58),
        ("idle", 16),
        ("normal_cycle", 55),
        ("idle", 15),
        ("abnormal_cycle", 88),
        ("idle", 20),
    )

    counters = {"idle": 0, "normal_cycle": 0, "abnormal_cycle": 0}
    for segment_type, duration_seconds in plan:
        counters[segment_type] += 1
        segment_samples = _generate_segment_samples(
            segment_type=segment_type,
            start_timestamp=current_timestamp,
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            machine_id=machine_id,
            rng=generator,
        )
        all_samples.extend(segment_samples)
        segments.append(
            DemoSegment(
                segment_id=f"{segment_type}_{counters[segment_type]}",
                segment_type=segment_type,
                start_timestamp=segment_samples[0].timestamp,
                end_timestamp=segment_samples[-1].timestamp,
                expected_label=_expected_label(segment_type),
            )
        )
        current_timestamp = segment_samples[-1].timestamp + timedelta(
            seconds=sample_interval_seconds
        )

    return DemoTimeline(samples=all_samples, segments=segments)


def _generate_segment_samples(
    *,
    segment_type: SegmentType,
    start_timestamp: datetime,
    duration_seconds: int,
    sample_interval_seconds: int,
    machine_id: str,
    rng: random.Random,
) -> list[PowerSample]:
    if segment_type == "idle":
        return generate_idle_segment(
            start_timestamp,
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            machine_id=machine_id,
            rng=rng,
        )
    if segment_type == "normal_cycle":
        return generate_normal_cycle(
            start_timestamp,
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            machine_id=machine_id,
            rng=rng,
        )
    return generate_abnormal_cycle(
        start_timestamp,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        machine_id=machine_id,
        rng=rng,
    )


def _normal_cycle_power(
    progress: float,
    plateau_kw: float,
    ramp_peak_kw: float,
    rng: random.Random,
) -> float:
    if progress < 0.12:
        power_kw = 0.55 + progress * 4.0
    elif progress < 0.28:
        phase = (progress - 0.12) / 0.16
        power_kw = 0.95 + phase * (ramp_peak_kw - 0.95)
    elif progress < 0.78:
        ripple_kw = 0.12 * math.sin(progress * math.tau * 7)
        power_kw = plateau_kw + ripple_kw
    else:
        phase = (progress - 0.78) / 0.22
        power_kw = plateau_kw * (1 - phase) + 0.62 * phase
    return max(0.0, power_kw + rng.uniform(-0.08, 0.08))


def _abnormal_cycle_power(
    progress: float,
    plateau_kw: float,
    rng: random.Random,
) -> float:
    if progress < 0.10:
        power_kw = 0.55 + progress * 5.0
    elif progress < 0.24:
        phase = (progress - 0.10) / 0.14
        power_kw = 1.1 + phase * (plateau_kw - 1.1)
    elif progress < 0.72:
        ripple_kw = 0.22 * math.sin(progress * math.tau * 5)
        load_bulge_kw = 0.45 if 0.42 <= progress <= 0.58 else 0.0
        power_kw = plateau_kw + ripple_kw + load_bulge_kw
    else:
        phase = (progress - 0.72) / 0.28
        power_kw = plateau_kw * (1 - phase) + 0.65 * phase

    spike_kw = 2.1 * math.exp(-((progress - 0.63) / 0.025) ** 2)
    return max(0.0, power_kw + spike_kw + rng.uniform(-0.1, 0.1))


def _iter_timestamps(
    start_timestamp: datetime,
    duration_seconds: int,
    sample_interval_seconds: int,
) -> list[datetime]:
    sample_count = (duration_seconds // sample_interval_seconds) + 1
    return [
        start_timestamp + timedelta(seconds=index * sample_interval_seconds)
        for index in range(sample_count)
    ]


def _sample(timestamp: datetime, machine_id: str, power_kw: float) -> PowerSample:
    return PowerSample(
        timestamp=timestamp,
        machine_id=machine_id,
        power_kw=round(power_kw, 4),
    )


def _expected_label(segment_type: SegmentType) -> ExpectedLabel:
    if segment_type == "normal_cycle":
        return "normal"
    if segment_type == "abnormal_cycle":
        return "abnormal"
    return "idle"


def _validate_generation_inputs(
    start_timestamp: datetime,
    duration_seconds: int,
    sample_interval_seconds: int,
) -> None:
    if start_timestamp.tzinfo is None or start_timestamp.utcoffset() != timedelta(0):
        raise ValueError("start_timestamp must be timezone-aware UTC")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if duration_seconds % sample_interval_seconds != 0:
        raise ValueError("duration_seconds must align to sample_interval_seconds")
