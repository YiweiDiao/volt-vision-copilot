# Volt Vision Copilot CSV Demo Bundle

These files are synthetic, public demonstration inputs for the local Volt
Vision Copilot CSV workflow.

They are not real CNC data, customer production data, or evidence that this
prototype validates industrial fault-diagnosis performance.

## Files

- `calibration_1.csv`
- `calibration_2.csv`
- `calibration_3.csv`
- `candidate_normal.csv`
- `candidate_changed.csv`

## How To Use

1. Launch the dashboard:
   `uv run streamlit run src/volt_vision/ui/app.py`
2. Select `CSV workflow`.
3. Upload `calibration_1.csv`, `calibration_2.csv`, and `calibration_3.csv`
   as known-good calibration files.
4. Upload `candidate_normal.csv` to demonstrate a result within the calibrated
   normal band.
5. Upload `candidate_changed.csv` to demonstrate a suspected deviation.

The dashboard does not use filenames to infer status. Filenames only help a
human select demo inputs. The result comes from CSV contents, calibration, DTW,
threshold logic, and deterministic event construction.

A suspected deviation is not a confirmed fault or root-cause diagnosis. Manual
inspection is required before any action.

## CSV Schema

All files use strict CSV v1 with exactly these columns:

```text
timestamp,machine_id,power_kw
```

No labels, expected outcomes, diagnoses, or extra columns are included.
