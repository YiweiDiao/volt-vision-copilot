"""Deterministic DTW calibration from known-good demo normal cycles.

Calibration here means selecting one real input cycle as a medoid reference
template. This module does not derive thresholds, classify deviations, or use
abnormal-cycle data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from volt_vision.monitoring.cycles import SelectedCycle
from volt_vision.monitoring.dtw import compute_power_sample_dtw


@dataclass(frozen=True)
class PairwiseDtwDistance:
    left_segment_id: str
    right_segment_id: str
    normalized_distance: float


@dataclass(frozen=True)
class CalibrationResult:
    reference_cycle: SelectedCycle
    calibration_cycles: tuple[SelectedCycle, ...]
    pairwise_distances: tuple[PairwiseDtwDistance, ...]
    mean_normalized_distances: dict[str, float]


def calibrate_reference_template(
    cycles: Sequence[SelectedCycle],
) -> CalibrationResult:
    """Select the normal-cycle medoid with the lowest mean DTW distance.

    Mean distances are compared exactly as floats. If two means are exactly
    equal, Python's stable ``min`` selection preserves the supplied input order.
    """

    calibration_cycles = tuple(cycles)
    _validate_calibration_cycles(calibration_cycles)

    pairwise_distances = _compute_pairwise_distances(calibration_cycles)
    mean_normalized_distances = _compute_mean_distances(
        calibration_cycles,
        pairwise_distances,
    )
    reference_cycle = min(
        calibration_cycles,
        key=lambda cycle: mean_normalized_distances[cycle.segment_id],
    )

    return CalibrationResult(
        reference_cycle=reference_cycle,
        calibration_cycles=calibration_cycles,
        pairwise_distances=pairwise_distances,
        mean_normalized_distances=mean_normalized_distances,
    )


def _validate_calibration_cycles(cycles: tuple[SelectedCycle, ...]) -> None:
    if len(cycles) < 3:
        raise ValueError("at least three normal calibration cycles are required")

    segment_ids = [cycle.segment_id for cycle in cycles]
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("calibration cycles must have unique segment IDs")

    machine_ids: set[str] = set()
    for cycle in cycles:
        if cycle.segment_type != "normal_cycle":
            raise ValueError(
                "calibration cycles must all have segment_type == 'normal_cycle'"
            )
        if not cycle.samples:
            raise ValueError(f"calibration cycle has no samples: {cycle.segment_id}")

        cycle_machine_ids = {sample.machine_id for sample in cycle.samples}
        if len(cycle_machine_ids) != 1:
            raise ValueError(
                f"calibration cycle samples must use one machine_id: "
                f"{cycle.segment_id}"
            )
        machine_ids.update(cycle_machine_ids)

    if len(machine_ids) != 1:
        raise ValueError("calibration cycles must all come from one machine_id")


def _compute_pairwise_distances(
    cycles: tuple[SelectedCycle, ...],
) -> tuple[PairwiseDtwDistance, ...]:
    distances: list[PairwiseDtwDistance] = []
    for left_index, left_cycle in enumerate(cycles[:-1]):
        for right_cycle in cycles[left_index + 1 :]:
            dtw_result = compute_power_sample_dtw(
                left_cycle.samples,
                right_cycle.samples,
            )
            normalized_distance = dtw_result.normalized_distance
            if not math.isfinite(normalized_distance) or normalized_distance < 0:
                raise ValueError("pairwise DTW distance must be finite and non-negative")
            distances.append(
                PairwiseDtwDistance(
                    left_segment_id=left_cycle.segment_id,
                    right_segment_id=right_cycle.segment_id,
                    normalized_distance=normalized_distance,
                )
            )
    return tuple(distances)


def _compute_mean_distances(
    cycles: tuple[SelectedCycle, ...],
    pairwise_distances: tuple[PairwiseDtwDistance, ...],
) -> dict[str, float]:
    distances_by_segment_id = {cycle.segment_id: [] for cycle in cycles}
    for distance in pairwise_distances:
        distances_by_segment_id[distance.left_segment_id].append(
            distance.normalized_distance
        )
        distances_by_segment_id[distance.right_segment_id].append(
            distance.normalized_distance
        )

    return {
        segment_id: sum(distances) / len(distances)
        for segment_id, distances in distances_by_segment_id.items()
    }
