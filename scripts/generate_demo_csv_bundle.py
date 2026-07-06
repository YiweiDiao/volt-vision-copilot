"""Generate public synthetic CSV inputs for the Volt Vision dashboard demo."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import UTC
from pathlib import Path

from volt_vision.monitoring.cycles import (
    SelectedCycle,
    select_abnormal_evaluation_cycle,
    select_calibration_cycles,
)
from volt_vision.monitoring.demo_data import generate_demo_timeline

DEFAULT_OUTPUT_DIRECTORY = Path("examples/csv_demo")
CSV_HEADER = "timestamp,machine_id,power_kw"


def generate_demo_csv_bundle(output_directory: str | Path) -> tuple[Path, ...]:
    """Write the deterministic synthetic CSV demo bundle."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    cycles_by_filename = _demo_cycles_by_filename()
    written_paths = []
    for filename, cycle in cycles_by_filename.items():
        csv_path = output_path / filename
        csv_path.write_text(_cycle_to_csv_text(cycle), encoding="utf-8")
        written_paths.append(csv_path)
    return tuple(written_paths)


def _demo_cycles_by_filename() -> Mapping[str, SelectedCycle]:
    timeline = generate_demo_timeline()
    calibration_cycles = select_calibration_cycles(timeline)
    changed_candidate = select_abnormal_evaluation_cycle(timeline)
    return {
        "calibration_1.csv": calibration_cycles[0],
        "calibration_2.csv": calibration_cycles[1],
        "calibration_3.csv": calibration_cycles[2],
        "candidate_normal.csv": calibration_cycles[0],
        "candidate_changed.csv": changed_candidate,
    }


def _cycle_to_csv_text(cycle: SelectedCycle) -> str:
    rows = [CSV_HEADER]
    rows.extend(_sample_to_csv_row(sample) for sample in cycle.samples)
    return "\n".join(rows) + "\n"


def _sample_to_csv_row(sample) -> str:
    timestamp = sample.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
    power_kw = f"{sample.power_kw:.4f}"
    return f"{timestamp},{sample.machine_id},{power_kw}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CSV files for the Volt Vision demo."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIRECTORY,
        type=Path,
        help="Directory where the CSV demo bundle should be written.",
    )
    args = parser.parse_args()
    generate_demo_csv_bundle(args.output)


if __name__ == "__main__":
    main()
