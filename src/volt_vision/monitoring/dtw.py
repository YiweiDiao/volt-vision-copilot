"""Deterministic univariate DTW for educational power-shape comparison.

This module compares one-dimensional power sequences only. It does not perform
calibration, thresholding, anomaly detection, or time-aware alignment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from volt_vision.monitoring.models import PowerSample


@dataclass(frozen=True)
class DtwResult:
    raw_distance: float
    normalized_distance: float
    warping_path_length: int


def compute_dtw(values_a: Sequence[float], values_b: Sequence[float]) -> DtwResult:
    """Compute full dynamic-programming DTW distance for two numeric sequences.

    Local cost is absolute difference. When predecessor costs tie, the stable
    tie-break is: lower accumulated cost, shorter path length, then diagonal,
    up, left. The final normalized distance is raw cost divided by selected
    warping path length.
    """

    clean_a = _validate_values(values_a, "values_a")
    clean_b = _validate_values(values_b, "values_b")

    row_count = len(clean_a) + 1
    column_count = len(clean_b) + 1
    costs = [[math.inf] * column_count for _ in range(row_count)]
    lengths = [[0] * column_count for _ in range(row_count)]
    costs[0][0] = 0.0

    for row_index, value_a in enumerate(clean_a, start=1):
        for column_index, value_b in enumerate(clean_b, start=1):
            local_cost = abs(value_a - value_b)
            previous_cost, previous_length = min(
                (
                    (costs[row_index - 1][column_index - 1], lengths[row_index - 1][column_index - 1], 0),
                    (costs[row_index - 1][column_index], lengths[row_index - 1][column_index], 1),
                    (costs[row_index][column_index - 1], lengths[row_index][column_index - 1], 2),
                ),
                key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
            )[:2]
            costs[row_index][column_index] = local_cost + previous_cost
            lengths[row_index][column_index] = previous_length + 1

    raw_distance = costs[-1][-1]
    warping_path_length = lengths[-1][-1]
    return DtwResult(
        raw_distance=raw_distance,
        normalized_distance=raw_distance / warping_path_length,
        warping_path_length=warping_path_length,
    )


def compute_power_sample_dtw(
    samples_a: Sequence[PowerSample],
    samples_b: Sequence[PowerSample],
) -> DtwResult:
    """Compute DTW using the power_kw field from PowerSample objects."""

    return compute_dtw(
        [sample.power_kw for sample in samples_a],
        [sample.power_kw for sample in samples_b],
    )


def _validate_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")

    clean_values = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in clean_values):
        raise ValueError(f"{name} must contain only finite numeric values")
    return clean_values
