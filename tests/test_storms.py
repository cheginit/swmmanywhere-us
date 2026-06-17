"""Tests for NRCS design-storm synthesis and the rain-derived simulation window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swmmanywhere_us.storms import (
    build_nrcs_hyetograph,
    duration_hours,
    write_rain_dat,
)
from swmmanywhere_us.swmm_defaults import SWMMOptions


def _total_depth(df: pd.DataFrame, dt_min: int) -> float:
    """Reconstruct total depth from an INTENSITY hyetograph (value * dt)."""
    return float((df["value"] * (dt_min / 60.0)).sum())


@pytest.mark.parametrize(("unit", "expected"), [("MM", 144.0), ("IN", 144.0 / 25.4)])
def test_hyetograph_conserves_depth(unit: str, expected: float):
    """Differencing the mass curve to INTENSITY must conserve the total depth."""
    dt = 6
    df = build_nrcs_hyetograph(144.0, storm_type="III", dt_min=dt, rain_dat_unit=unit)
    assert _total_depth(df, dt) == pytest.approx(expected, rel=1e-9)


def test_hyetograph_is_monotone_and_nonnegative():
    """A valid hyetograph has no negative intensities (curve is monotone)."""
    df = build_nrcs_hyetograph(120.0, storm_type="II", dt_min=6)
    assert (df["value"] >= 0).all()
    # Cumulative depth is non-decreasing.
    cum = np.cumsum(df["value"].to_numpy())
    assert np.all(np.diff(cum) >= 0)


def test_type_ii_front_loads_peak_more_than_type_iii():
    """By hour 12, Type II has dropped ~66% of depth; Type III is symmetric (~50%)."""
    depth, dt = 144.0, 6
    n_to_12h = int(12 * 60 / dt)  # interval index at t = 12 h
    cum_ii = (build_nrcs_hyetograph(depth, storm_type="II", dt_min=dt)["value"][:n_to_12h]).sum() * (
        dt / 60.0
    )
    cum_iii = (
        build_nrcs_hyetograph(depth, storm_type="III", dt_min=dt)["value"][:n_to_12h]
    ).sum() * (dt / 60.0)
    assert cum_iii == pytest.approx(0.5 * depth, abs=0.02 * depth)
    assert cum_ii > 0.6 * depth
    assert cum_ii > cum_iii


def test_hyetograph_anchor_and_trailing_zero():
    """Series starts at the anchor and ends with a zero so rain stops cleanly."""
    df = build_nrcs_hyetograph(100.0, storm_type="III", dt_min=6, start="2008-06-01 00:00:00")
    assert df["date"].iloc[0] == pd.Timestamp("2008-06-01 00:00:00")
    assert df["date"].iloc[-1] == pd.Timestamp("2008-06-02 00:00:00")  # +24 h
    assert df["value"].iloc[-1] == 0.0


def test_invalid_storm_type_raises():
    with pytest.raises(ValueError, match="storm_type"):
        build_nrcs_hyetograph(100.0, storm_type="IV")  # type: ignore[arg-type]


def test_nondividing_timestep_raises_clear_error():
    """A timestep that doesn't divide the duration is a clear config error, not a crash."""
    with pytest.raises(ValueError, match="divisor of the storm duration"):
        build_nrcs_hyetograph(100.0, duration="60-min", dt_min=7)


def test_nondividing_timestep_still_conserves_when_valid():
    """A valid divisor (e.g. 10-min over 60-min) lands exactly on the duration."""
    df = build_nrcs_hyetograph(50.0, storm_type="II", duration="60-min", dt_min=10)
    assert df["date"].iloc[-1] - df["date"].iloc[0] == pd.Timedelta(hours=1)
    assert (df["value"][:-1] * (10 / 60.0)).sum() == pytest.approx(50.0, rel=1e-9)


@pytest.mark.parametrize(
    "bad",
    [
        {"return_period": 20},  # not an Atlas-14 ARI column
        {"duration": "90-min"},  # not a known duration label
        {"timestep_min": 0},  # must be > 0
        {"tail_hours": -1.0},  # must be >= 0
        {"storm_type": "IV"},  # only II / III
    ],
)
def test_nrcs_storm_config_rejects_bad_values(bad: dict):
    """Bad nrcs_storm values fail at config validation, not silently as a NOAA fallback."""
    from pydantic import ValidationError

    from swmmanywhere_us.swmmanywhere import NrcsStorm

    with pytest.raises(ValidationError):
        NrcsStorm(**bad)


@pytest.mark.parametrize(
    ("label", "hours"), [("24-hr", 24.0), ("6-hr", 6.0), ("60-min", 1.0), ("30-min", 0.5)]
)
def test_duration_hours(label: str, hours: float):
    assert duration_hours(label) == pytest.approx(hours)


def test_write_rain_dat_roundtrips(tmp_path):
    """The written .dat parses back to the same gage/date/value rows."""
    df = build_nrcs_hyetograph(120.0, storm_type="III", dt_min=6, start="2000-01-01 00:00:00")
    path = write_rain_dat(df, tmp_path / "storm.dat", comment="test")
    parsed = pd.read_csv(
        path, sep=r"\s+", comment=";", names=["g", "y", "mo", "d", "h", "mi", "value"]
    )
    assert len(parsed) == len(df)
    assert parsed["value"].to_numpy() == pytest.approx(df["value"].to_numpy(), abs=1e-4)


def test_from_rain_data_tail_extends_end():
    """sim_tail_hours pushes END past the last rain timestamp for drawdown."""
    rain = pd.DataFrame({"date": pd.to_datetime(["2000-01-01 00:00", "2000-01-01 02:00"])})
    opts = SWMMOptions.from_rain_data(rain, sim_tail_hours=24.0)
    # Last rain is 02:00 on 01/01; +24 h -> 02:00 on 01/02.
    assert opts.end_date == "01/02/2000"
    assert opts.end_time == "02:00:00"


def test_from_rain_data_overlap_guard_raises():
    """An explicit window that misses user-supplied rain fails loudly."""
    rain = pd.DataFrame({"date": pd.to_datetime(["2000-01-01 00:00", "2000-01-01 02:00"])})
    with pytest.raises(ValueError, match="do not overlap"):
        SWMMOptions.from_rain_data(
            rain, validate_overlap=True, start_date="01/01/2008", end_date="01/01/2008"
        )


def test_from_rain_data_overlap_guard_passes_when_aligned():
    """The guard stays dormant when the explicit window covers the rain."""
    rain = pd.DataFrame({"date": pd.to_datetime(["2000-01-01 00:00", "2000-01-01 02:00"])})
    opts = SWMMOptions.from_rain_data(
        rain, validate_overlap=True, start_date="01/01/2000", end_date="01/01/2000"
    )
    assert opts.start_date == "01/01/2000"
