"""Tests for SWMM post-processing / INP emission."""

from __future__ import annotations

import pytest

from swmmanywhere_us.post_processing import _weir_discharge_coeff
from swmmanywhere_us.swmm_defaults import WeirDefaults


def test_weir_discharge_coeff_unit_conversion():
    """US-customary weir coefficient is converted to SI only for metric flow units."""
    # US flow units -> unchanged.
    assert _weir_discharge_coeff(3.33, "CFS") == 3.33
    assert _weir_discharge_coeff(3.0, "GPM") == 3.0
    assert _weir_discharge_coeff(3.0, "MGD") == 3.0
    # Metric flow units -> scaled by sqrt(0.3048) ~ 0.5521 (US 3.33 -> SI ~1.84).
    assert _weir_discharge_coeff(3.33, "LPS") == pytest.approx(1.838, abs=0.005)
    for u in ("LPS", "CMS", "MLD"):
        assert _weir_discharge_coeff(3.0, u) == pytest.approx(3.0 * 0.3048**0.5)


def test_weir_inp_row_reads_disch_coeff_key():
    """The key post_processing writes ('disch_coeff') is the one to_inp_row honours."""
    # DischCoeff is column index 5: Name InletNode OutletNode WeirType CrestHeight DischCoeff ...
    row = WeirDefaults().to_inp_row(
        "W1", disch_coeff=2.5, InletNode="A", OutletNode="B", crest_height=1.0
    )
    assert float(row.split()[5]) == pytest.approx(2.5)

    # The old, broken 'Cd' key is ignored -> falls back to the dataclass default.
    row_bad = WeirDefaults().to_inp_row("W2", Cd=99.0, InletNode="A", OutletNode="B")
    assert float(row_bad.split()[5]) == pytest.approx(WeirDefaults.disch_coeff)
    assert "99" not in row_bad
