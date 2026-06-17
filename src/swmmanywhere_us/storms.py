"""Synthetic NRCS (SCS) design-storm hyetographs for SWMManywhere-US.

Distributes a total rainfall depth (e.g. the NOAA Atlas 14 24-hour point
estimate) into a rainfall time series using the standard SCS dimensionless
cumulative rainfall distributions, and writes it as a SWMM ``.dat`` rain file
(``Format=INTENSITY``) that :mod:`post_processing` wires to the model's single
rain gage.  This module is pure and offline -- the NOAA depth lookup lives in
:func:`prepare_data.get_design_precipitation` and is passed in as a number.

NRCS rainfall-distribution type (NRCS TR-55, Appendix B):

* **Type II** -- most of the continental US (the nationally "common" choice).
* **Type III** -- Gulf of Mexico and Atlantic coastal areas, including the
  Florida peninsula, where tropical systems drop large 24-hour totals.

Florida-focused models should use **Type III**; the type is configurable.

The dimensionless cumulative fractions are the classic TP-149 / TR-55 24-hour
ordinates shipped by HEC-HMS and HydroCAD (Type II reaches 0.663 of the
24-hour depth by hour 12).  Do NOT "correct" the hour-12 value to the NEH-630
0.476 variant -- they are different published tabulations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

StormType = Literal["II", "III"]

_MM_PER_IN = 25.4

# (hour, cumulative fraction Pt/P24); monotone, 0 at t=0, 1 at t=24h.
_SCS_24H: dict[str, list[tuple[float, float]]] = {
    "II": [
        (0.0, 0.000), (1.0, 0.011), (2.0, 0.022), (3.0, 0.035), (4.0, 0.048),
        (5.0, 0.063), (6.0, 0.080), (7.0, 0.099), (8.0, 0.120), (9.0, 0.147),
        (10.0, 0.181), (11.0, 0.235), (11.5, 0.283), (11.75, 0.357),
        (12.0, 0.663), (12.5, 0.735), (13.0, 0.772), (14.0, 0.820),
        (15.0, 0.854), (16.0, 0.880), (18.0, 0.921), (20.0, 0.952),
        (22.0, 0.977), (24.0, 1.000),
    ],
    "III": [
        (0.0, 0.000), (1.0, 0.010), (2.0, 0.020), (3.0, 0.031), (4.0, 0.043),
        (5.0, 0.057), (6.0, 0.072), (7.0, 0.089), (8.0, 0.115), (9.0, 0.148),
        (10.0, 0.189), (11.0, 0.250), (11.5, 0.298), (11.75, 0.339),
        (12.0, 0.500), (12.5, 0.702), (13.0, 0.751), (14.0, 0.811),
        (15.0, 0.849), (16.0, 0.886), (18.0, 0.927), (20.0, 0.957),
        (22.0, 0.980), (24.0, 1.000),
    ],
}


def duration_hours(duration: str) -> float:
    """Parse an Atlas-14 duration label ('24-hr', '60-min', '6-hr') to hours."""
    value, _, unit = duration.partition("-")
    hours = float(value)
    if unit.startswith("min"):
        return hours / 60.0
    if unit.startswith(("hr", "hour")):
        return hours
    if unit.startswith("day"):
        return hours * 24.0
    msg = f"Unrecognized duration label: {duration!r}"
    raise ValueError(msg)


def build_nrcs_hyetograph(
    total_depth_mm: float,
    *,
    storm_type: StormType = "III",
    duration: str = "24-hr",
    dt_min: int = 6,
    start: str | pd.Timestamp | None = None,
    rain_dat_unit: Literal["IN", "MM"] = "MM",
) -> pd.DataFrame:
    """Distribute a total depth into an SCS design-storm hyetograph.

    The SCS Type II/III curves are defined over a 24-hour storm; for a
    non-24-hour ``duration`` the dimensionless curve is stretched to that
    span (exact for the standard ``"24-hr"`` case).

    Args:
        total_depth_mm: Total storm depth (mm), e.g. the Atlas-14 point estimate.
        storm_type: SCS distribution, ``"II"`` (national) or ``"III"`` (coastal/FL).
        duration: Storm duration label; sets the span the curve is stretched over.
        dt_min: Hyetograph time step (minutes).
        start: Timestamp anchoring the series (defaults to 2000-01-01 00:00).
        rain_dat_unit: Units for the returned ``value`` column ("MM" or "IN").

    Returns:
        DataFrame ``[rain_gage, date, value]`` where ``value`` is rainfall
        INTENSITY (depth-unit per hour) over each ``dt_min`` interval, with a
        trailing zero so rainfall stops cleanly.  The total depth is conserved.
    """
    if storm_type not in _SCS_24H:
        msg = f"storm_type must be 'II' or 'III', got {storm_type!r}"
        raise ValueError(msg)
    start = pd.Timestamp("2000-01-01") if start is None else pd.Timestamp(start)
    dur_h = duration_hours(duration)
    dur_min = dur_h * 60.0
    if dt_min <= 0 or not float(dur_min / dt_min).is_integer():
        msg = (
            f"timestep_min ({dt_min}) must be a positive divisor of the storm duration "
            f"({dur_min:g} min); e.g. 5, 6, 10, 15 for a 24-hr storm"
        )
        raise ValueError(msg)
    n = round(dur_min / dt_min)

    curve = _SCS_24H[storm_type]
    curve_h = np.array([h for h, _ in curve])
    curve_f = np.array([f for _, f in curve])
    grid_h = curve_h / 24.0 * dur_h  # stretch the 24-h curve onto this span

    depth_unit = total_depth_mm if rain_dat_unit == "MM" else total_depth_mm / _MM_PER_IN
    # linspace lands the last edge exactly on dur_h (no float drift); since dt_min
    # divides dur_min, the step is exactly dt_min minutes.
    edges_h = np.linspace(0.0, dur_h, n + 1)
    cum = np.interp(edges_h, grid_h, curve_f) * depth_unit  # cumulative depth
    incr = np.diff(cum)  # per-interval depth (>= 0; curve is monotone)
    intensity = incr / (dt_min / 60.0)  # depth-unit per hour

    total = float(incr.sum())
    if not np.isclose(total, depth_unit, rtol=1e-9, atol=1e-9):
        msg = f"hyetograph depth {total} != target {depth_unit}; distribution is non-conserving"
        raise AssertionError(msg)

    # One row per interval start, plus a trailing zero at the storm end.
    times = [start + pd.Timedelta(minutes=dt_min * i) for i in range(n + 1)]
    values = [*intensity.tolist(), 0.0]
    return pd.DataFrame({"rain_gage": 1, "date": times, "value": values})


def write_rain_dat(df: pd.DataFrame, path: str | Path, *, comment: str = "") -> Path:
    """Write a rain DataFrame to a SWMM ``.dat`` file (gage Y M D H M value)."""
    path = Path(path)
    lines = [f";{comment}"] if comment else []
    lines.extend(
        f"{int(r.rain_gage)}   {r.date.year} {r.date.month:02d} {r.date.day:02d} "
        f"{r.date.hour:02d} {r.date.minute:02d}    {r.value:.4f}"
        for r in df.itertuples(index=False)
    )
    path.write_text("\n".join(lines) + "\n")
    return path
