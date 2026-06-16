from __future__ import annotations

from swmmanywhere_us.parameters import (
    HydraulicDesign,
    OutfallDerivation,
    SubcatchmentDerivation,
    TopologyDerivation,
    get_full_parameters,
)


def test_get_full_parameters_keys():
    """Verify get_full_parameters returns all expected parameter groups."""
    params = get_full_parameters()
    assert set(params.keys()) == {
        "subcatchment_derivation",
        "outfall_derivation",
        "topology_derivation",
        "trunk_inference",
        "hydraulic_design",
        "channel_design",
        "pond_design",
        "dual_drainage",
        "simplification",
    }


def test_get_full_parameters_types():
    """Verify parameter values are of the correct Pydantic model type."""
    params = get_full_parameters()
    assert isinstance(params["subcatchment_derivation"], SubcatchmentDerivation)
    assert isinstance(params["outfall_derivation"], OutfallDerivation)
    assert isinstance(params["topology_derivation"], TopologyDerivation)
    assert isinstance(params["hydraulic_design"], HydraulicDesign)


def test_get_full_parameters_returns_copy():
    """Verify get_full_parameters returns a new dict each time."""
    p1 = get_full_parameters()
    p2 = get_full_parameters()
    assert p1 is not p2


def test_subcatchment_derivation_defaults():
    """Verify SubcatchmentDerivation default values are preserved."""
    sd = SubcatchmentDerivation()
    assert sd.min_drainage_area_m2 == 100_000
    assert sd.lane_width == 3.5
    assert sd.max_street_length == 60.0
    assert sd.dem_resolution == 10
    assert sd.lulc_year == 2019
    assert sd.buffer_size_local == 5.0
    assert sd.min_hole_areasqm_local == 1000.0
    assert sd.buffer_size_major == 15.0
    assert sd.min_hole_areasqm_major == 20000.0


def test_outfall_derivation_defaults():
    """Verify OutfallDerivation default values are preserved."""
    od = OutfallDerivation()
    assert od.river_buffer_distance == 150.0
    assert od.outfall_clustering_factor == 1.0


def test_topology_derivation_defaults():
    """Verify TopologyDerivation default values are preserved."""
    td = TopologyDerivation()
    assert td.omit_edges == ["corridor", "track", "footway", "path"]
    assert td.weights == ["chahinian_slope", "length", "contributing_area"]
    assert td.chahinian_slope_scaling == 1
    assert td.chahinian_angle_scaling == 0.3
    assert td.length_scaling == 0.1
    assert td.contributing_area_scaling == 0.1


def test_hydraulic_design_defaults():
    """Verify HydraulicDesign default values are preserved."""
    hd = HydraulicDesign()
    assert hd.max_fr == 0.8
    assert hd.min_shear == 2
    assert hd.min_v == 0.61
    assert hd.max_v == 3.05
    assert hd.min_depth == 0.5
    assert hd.max_depth == 5
    assert hd.precipitation == 0.006
    assert hd.depth_nbins == 10
    assert len(hd.diameters) > 0


def test_parameter_override():
    """Verify parameter overrides work correctly."""
    params = get_full_parameters()
    params["outfall_derivation"].outfall_clustering_factor = 2.0
    params["outfall_derivation"].river_buffer_distance = 30
    assert params["outfall_derivation"].outfall_clustering_factor == 2.0
    assert params["outfall_derivation"].river_buffer_distance == 30
