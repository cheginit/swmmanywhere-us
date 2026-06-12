"""Tests to verify function signatures after refactoring."""

from __future__ import annotations

import inspect

from swmmanywhere_us.geospatial_utilities import compute_flow_direction, derive_subcatchments


def test_compute_flow_direction_signature():
    """Verify compute_flow_direction has required parameters."""
    sig = inspect.signature(compute_flow_direction)
    assert "fid" in sig.parameters
    assert "fdir_path" in sig.parameters
    assert "slope_path" in sig.parameters


def test_derive_subcatchments_no_method_param():
    """Verify derive_subcatchments has no method parameter."""
    sig = inspect.signature(derive_subcatchments)
    assert "method" not in sig.parameters
