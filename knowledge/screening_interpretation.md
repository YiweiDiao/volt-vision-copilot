# Screening Interpretation

`within_normal_band` means the normalized DTW distance did not exceed the current calibration-derived threshold. It does not prove the machine is healthy.

`suspected_deviation` means the candidate power pattern exceeded the calibrated normal band. It does not confirm a fault.

Duration, energy, and peak-power deviations are explanatory evidence only. They do not decide event status.

Calibration cycles are user-declared known-good cycles. The prototype treats them as the local reference basis for screening, not as proof of universal normal operation.
