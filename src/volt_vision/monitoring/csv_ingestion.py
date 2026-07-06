"""Strict local CSV ingestion for candidate and calibration power cycles.

CSV ingestion only creates user-declared cycles for deterministic screening or
calibration. Calibration CSVs are treated as known-good because the user
declares them so; ingestion does not independently verify normality, infer
labels, diagnose a fault, or infer a root cause.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from volt_vision.monitoring.cycles import SelectedCycle
from volt_vision.monitoring.models import PowerSample

REQUIRED_HEADERS = frozenset({"timestamp", "machine_id", "power_kw"})


def load_candidate_cycle_from_csv(
    csv_path: str | Path,
    *,
    candidate_segment_id: str = "uploaded_cycle",
) -> SelectedCycle:
    """Load one complete, already-selected candidate cycle from a local CSV."""

    segment_id = candidate_segment_id.strip()
    if not segment_id:
        raise ValueError("candidate_segment_id must be non-empty")

    return SelectedCycle(
        segment_id=segment_id,
        segment_type="candidate_cycle",
        samples=_load_samples_from_csv(csv_path),
    )


def load_calibration_cycles_from_csv(
    csv_paths: Sequence[str | Path],
    *,
    calibration_segment_ids: Sequence[str] | None = None,
) -> tuple[SelectedCycle, ...]:
    """Load user-declared known-good calibration cycles from local CSV files."""

    paths = _validate_csv_paths(csv_paths)
    segment_ids = _calibration_segment_ids(paths, calibration_segment_ids)
    cycles = tuple(
        SelectedCycle(
            segment_id=segment_id,
            segment_type="normal_cycle",
            samples=_load_samples_from_csv(path),
        )
        for path, segment_id in zip(paths, segment_ids)
    )
    _validate_calibration_machine_consistency(cycles)
    return cycles


def _load_samples_from_csv(csv_path: str | Path) -> tuple[PowerSample, ...]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        csv_text = csv_file.read()
    _reject_blank_physical_data_rows(csv_text)

    reader = csv.DictReader(io.StringIO(csv_text))
    _validate_headers(reader.fieldnames)
    samples = tuple(_parse_rows(reader))
    _validate_samples(samples)
    return samples


def _validate_csv_paths(csv_paths: Sequence[str | Path]) -> tuple[str | Path, ...]:
    if isinstance(csv_paths, (str, Path)):
        raise ValueError("csv_paths must be a sequence of local CSV paths")

    paths = tuple(csv_paths)
    if len(paths) < 3:
        raise ValueError("at least three calibration CSV paths are required")
    return paths


def _calibration_segment_ids(
    csv_paths: tuple[str | Path, ...],
    calibration_segment_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if calibration_segment_ids is None:
        return tuple(
            f"calibration_cycle_{index}"
            for index in range(1, len(csv_paths) + 1)
        )
    if isinstance(calibration_segment_ids, str):
        raise ValueError(
            "calibration_segment_ids must be a sequence of IDs, not one string"
        )

    segment_ids = tuple(segment_id.strip() for segment_id in calibration_segment_ids)
    if len(segment_ids) != len(csv_paths):
        raise ValueError("calibration_segment_ids must match csv_paths length")
    if any(not segment_id for segment_id in segment_ids):
        raise ValueError("calibration_segment_ids must be non-empty")
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("calibration_segment_ids must be unique")
    return segment_ids


def _validate_calibration_machine_consistency(
    cycles: tuple[SelectedCycle, ...],
) -> None:
    machine_ids = {cycle.samples[0].machine_id for cycle in cycles}
    if len(machine_ids) != 1:
        raise ValueError("all calibration cycles must use one machine_id")


def _validate_headers(fieldnames: Sequence[str] | None) -> None:
    if fieldnames is None:
        raise ValueError(_schema_error("CSV file is empty"))
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError(_schema_error("duplicate headers are not allowed"))

    header_set = set(fieldnames)
    missing_headers = REQUIRED_HEADERS - header_set
    extra_headers = header_set - REQUIRED_HEADERS
    if missing_headers:
        raise ValueError(_schema_error(f"missing columns: {sorted(missing_headers)}"))
    if extra_headers:
        raise ValueError(_schema_error(f"unexpected columns: {sorted(extra_headers)}"))


def _reject_blank_physical_data_rows(csv_text: str) -> None:
    for line_number, line in enumerate(csv_text.splitlines()[1:], start=2):
        if line.strip() == "":
            raise ValueError(
                f"CSV physical line {line_number}: blank data row is not allowed"
            )


def _parse_rows(reader: csv.DictReader[str]) -> list[PowerSample]:
    samples: list[PowerSample] = []
    for row_number, row in enumerate(reader, start=2):
        _validate_row_shape(row, row_number)
        timestamp = _parse_timestamp(row["timestamp"], row_number)
        machine_id = row["machine_id"].strip()
        if not machine_id:
            raise ValueError(f"CSV row {row_number}: machine_id must be non-empty")
        power_kw = _parse_power_kw(row["power_kw"], row_number)
        samples.append(
            PowerSample(
                timestamp=timestamp,
                machine_id=machine_id,
                power_kw=power_kw,
            )
        )
    return samples


def _validate_row_shape(
    row: Mapping[str | None, str | list[str] | None],
    row_number: int,
) -> None:
    if None in row:
        raise ValueError(f"CSV row {row_number}: too many values for CSV v1 schema")
    if all((not isinstance(value, str) or value.strip() == "") for value in row.values()):
        raise ValueError(f"CSV row {row_number}: blank data row is not allowed")
    missing_fields = []
    for header in sorted(REQUIRED_HEADERS):
        value = row.get(header)
        if value is None:
            missing_fields.append(header)
    if missing_fields:
        raise ValueError(
            f"CSV row {row_number}: missing value for required columns "
            f"{missing_fields}"
        )


def _parse_timestamp(value: str, row_number: int) -> datetime:
    raw_value = value.strip()
    if raw_value.endswith("Z"):
        raw_value = f"{raw_value[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {row_number}: timestamp must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"CSV row {row_number}: timestamp must be timezone-aware"
        )
    return parsed.astimezone(timezone.utc)


def _parse_power_kw(value: str, row_number: int) -> float:
    try:
        power_kw = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"CSV row {row_number}: power_kw must be numeric") from exc
    if not math.isfinite(power_kw):
        raise ValueError(f"CSV row {row_number}: power_kw must be finite")
    if power_kw < 0:
        raise ValueError(f"CSV row {row_number}: power_kw must be non-negative")
    return power_kw


def _validate_samples(samples: tuple[PowerSample, ...]) -> None:
    if not samples:
        raise ValueError("CSV file must contain at least one data row")

    machine_id = samples[0].machine_id
    previous_timestamp = samples[0].timestamp
    for index, sample in enumerate(samples[1:], start=2):
        row_number = index + 1
        if sample.machine_id != machine_id:
            raise ValueError(f"CSV row {row_number}: all rows must use one machine_id")
        if sample.timestamp == previous_timestamp:
            raise ValueError(f"CSV row {row_number}: duplicate timestamp")
        if sample.timestamp < previous_timestamp:
            raise ValueError(
                f"CSV row {row_number}: timestamps must be strictly increasing"
            )
        previous_timestamp = sample.timestamp


def _schema_error(reason: str) -> str:
    return (
        f"{reason}; CSV v1 schema requires exactly columns "
        f"{sorted(REQUIRED_HEADERS)}"
    )
