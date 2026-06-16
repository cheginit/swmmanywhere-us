"""Tests for graph functions to ensure refactoring doesn't change behavior."""

from __future__ import annotations

from typing import Any

import networkx as nx
import shapely

from swmmanywhere_us.graph_functions import (
    assign_id,
    calculate_weights,
    derive_topology,
    double_directed,
    enforce_outfall_slope,
    fix_geometries,
    identify_outfalls,
    pipe_by_pipe,
    remove_non_pipe_allowable_links,
    remove_river_crossing_pipes,
    set_chahinian_slope,
    set_surface_slope,
    split_long_edges,
    to_undirected,
)
from swmmanywhere_us.graph_functions.design import _partial_flow_hydraulics
from swmmanywhere_us.graph_functions.simplification import (
    _aggregate_subcatchments,
    _consolidate_chain,
    _find_degree2_chains,
    _find_protected_nodes,
    _remove_dangling_leaves,
    simplify_network,
)
from swmmanywhere_us.parameters import (
    HydraulicDesign,
    OutfallDerivation,
    SimplificationParams,
    SubcatchmentDerivation,
    TopologyDerivation,
)


def _make_street_graph() -> nx.MultiDiGraph[Any]:
    """Create a small street graph for testing."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    nodes = {
        0: {"x": 0.0, "y": 0.0},
        1: {"x": 100.0, "y": 0.0},
        2: {"x": 200.0, "y": 0.0},
        3: {"x": 100.0, "y": 100.0},
    }
    for n, attrs in nodes.items():
        G.add_node(n, **attrs)

    edges = [
        (
            0,
            1,
            {
                "geometry": shapely.LineString([(0, 0), (100, 0)]),
                "length": 100.0,
                "highway": "residential",
                "lanes": 2,
                "edge_type": "pipe",
                "osmid": "a",
            },
        ),
        (
            1,
            2,
            {
                "geometry": shapely.LineString([(100, 0), (200, 0)]),
                "length": 100.0,
                "highway": "residential",
                "lanes": 2,
                "edge_type": "pipe",
                "osmid": "b",
            },
        ),
        (
            1,
            3,
            {
                "geometry": shapely.LineString([(100, 0), (100, 100)]),
                "length": 100.0,
                "highway": "footway",
                "lanes": 0,
                "edge_type": "pipe",
                "osmid": "c",
            },
        ),
    ]
    for u, v, d in edges:
        G.add_edge(u, v, **d)
    return G


def test_assign_id():
    """Test assign_id assigns IDs in format 'u-v'."""
    G = _make_street_graph()
    result = assign_id(G)
    for u, v, d in result.edges(data=True):
        assert "id" in d
        assert d["id"] == f"{u}-{v}"


def test_assign_id_removes_duplicates():
    """Test assign_id removes duplicate edges."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, geometry=shapely.LineString([(0, 0), (1, 0)]))
    G.add_edge(0, 1, geometry=shapely.LineString([(0, 0), (1, 0)]))
    assert G.number_of_edges() == 2
    result = assign_id(G)
    assert result.number_of_edges() == 1


def test_remove_non_pipe_allowable_links():
    """Test that non-pipe-allowable links are removed."""
    G = _make_street_graph()
    td = TopologyDerivation()  # omit_edges includes "footway"
    result = remove_non_pipe_allowable_links(G, topology_derivation=td)
    highways = [d.get("highway") for _, _, d in result.edges(data=True)]
    assert "footway" not in highways
    assert result.number_of_edges() == 2


def test_to_undirected():
    """Test converting directed to undirected graph."""
    G = _make_street_graph()
    result = to_undirected(G)
    assert not result.is_directed()
    assert result.number_of_edges() == G.number_of_edges()


def test_double_directed():
    """Test creating double directed graph."""
    G = nx.MultiGraph()
    G.add_edge(0, 1, id="0-1", geometry=shapely.LineString([(0, 0), (1, 0)]), edge_type="pipe")
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=1.0, y=0.0)
    result = double_directed(G)
    assert result.is_directed()
    assert result.has_edge(0, 1)
    assert result.has_edge(1, 0)
    # Reverse edge gets a natural "v-u" id with no suffix.
    fwd_data = result.get_edge_data(0, 1, 0)
    rev_data = result.get_edge_data(1, 0, 0)
    assert fwd_data["id"] == "0-1"
    assert rev_data["id"] == "1-0"


def test_fix_geometries():
    """Test fix_geometries rebuilds geometry from node coords."""
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=10.0, y=0.0)
    # Geometry doesn't match node coords
    G.add_edge(0, 1, geometry=shapely.LineString([(5, 5), (15, 5)]))
    result = fix_geometries(G)
    geom = result.get_edge_data(0, 1, 0)["geometry"]
    assert geom.coords[0] == (0.0, 0.0)
    assert geom.coords[-1] == (10.0, 0.0)


def test_split_long_edges():
    """Test split_long_edges splits edges exceeding max length."""
    G = nx.MultiGraph()
    G.graph["crs"] = "EPSG:32617"
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=200.0, y=0.0)
    G.add_edge(
        0,
        1,
        id="0-1",
        geometry=shapely.LineString([(0, 0), (200, 0)]),
        length=200.0,
        highway="residential",
    )
    sd = SubcatchmentDerivation(max_street_length=60.0)
    result = split_long_edges(G, subcatchment_derivation=sd)
    # 200m split into ~60m segments = at least 3 edges
    assert result.number_of_edges() >= 3
    for _, _, d in result.edges(data=True):
        assert d["length"] <= 60.0 + 1e-6


def test_set_surface_slope():
    """Test surface slope calculation."""
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0, surface_elevation=10.0)
    G.add_node(1, x=100.0, y=0.0, surface_elevation=5.0)
    G.add_edge(0, 1, length=100.0)
    result = set_surface_slope(G)
    slope = result.get_edge_data(0, 1, 0)["surface_slope"]
    assert abs(slope - 0.05) < 1e-10


def test_set_chahinian_slope():
    """Test Chahinian slope weighting."""
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=100.0, y=0.0)
    # Slope of 0.3% -> weight should be 0 (preferred)
    G.add_edge(0, 1, surface_slope=0.003, length=100.0)
    result = set_chahinian_slope(G)
    weight = result.get_edge_data(0, 1, 0)["chahinian_slope"]
    assert weight == 0.0


def test_pipe_by_pipe_no_adverse_slopes():
    """Pass 5 carving repairs adverse residue where it is feasible to do so.

    Terrain rises a modest 1.0 m over the first pipe, so Step C leaves the
    downstream invert mildly adverse; Pass 5 must carve the downstream path, 
    within the max_depth budget, until every street pipe is strictly positive.
    """
    G = nx.MultiDiGraph()
    surfaces = {1: 0.0, 2: 1.0, 3: 1.0, 4: 0.5}
    for n, se in surfaces.items():
        G.add_node(n, x=float(n) * 100.0, y=0.0, surface_elevation=se, contributing_area=500.0)
    for u, v in [(1, 2), (2, 3), (3, 4)]:
        G.add_edge(u, v, edge_type="pipe", length=100.0, id=f"{u}-{v}")

    result = pipe_by_pipe(G, hydraulic_design=HydraulicDesign())

    cf = nx.get_node_attributes(result, "chamber_floor_elevation")
    for u, v in [(1, 2), (2, 3), (3, 4)]:
        slope = (cf[u] - cf[v]) / 100.0
        assert slope > 0, f"pipe {u}->{v} adverse: slope={slope:.6f}"


def test_pipe_by_pipe_respects_max_depth_on_infeasible_terrain():
    """On terrain too steep for gravity, Pass 5 never excavates past max_depth.

    A 10 m rise over the first pipe cannot be made non-adverse within the 5 m
    excavation limit (it would need a pump).  The fixed Pass 5 leaves a bounded
    residual adverse pipe rather than carving an infeasible >5 m burial depth.
    """
    hd = HydraulicDesign()
    G = nx.MultiDiGraph()
    surfaces = {1: 0.0, 2: 10.0, 3: 10.0, 4: 9.5}
    for n, se in surfaces.items():
        G.add_node(n, x=float(n) * 100.0, y=0.0, surface_elevation=se, contributing_area=500.0)
    for u, v in [(1, 2), (2, 3), (3, 4)]:
        G.add_edge(u, v, edge_type="pipe", length=100.0, id=f"{u}-{v}")

    result = pipe_by_pipe(G, hydraulic_design=hd)

    cf = nx.get_node_attributes(result, "chamber_floor_elevation")
    se = nx.get_node_attributes(result, "surface_elevation")
    # No node is buried deeper than max_depth (the invariant the old carve broke).
    for n in cf:
        if n in se:
            assert se[n] - cf[n] <= hd.max_depth + 1e-6, f"node {n} exceeds max_depth"


def test_pond_volume_below():
    """_volume_below integrates the stage-storage curve up to a partial depth."""
    from swmmanywhere_us.graph_functions.water_bodies import (
        _curve_volume,
        _stage_area_curve,
        _volume_below,
    )

    curve = _stage_area_curve(5000.0, 2.0, 30.0, 0.55, 5)
    total = _curve_volume(curve)
    assert _volume_below(curve, 0.0) == 0.0
    assert abs(_volume_below(curve, 2.0) - total) < 1e-6  # full depth == total
    # Monotonic and strictly between for partial depths (top holds more volume).
    assert 0 < _volume_below(curve, 1.0) < _volume_below(curve, 1.5) < total
    # Weir activation ratio at 0.9*Dmax is a sensible fraction (<1).
    assert 0.5 < _volume_below(curve, 1.8) / total < 1.0


def test_partial_flow_hydraulics():
    """Design-depth partial-flow geometry differs from the R=D/4 full-pipe surrogate."""
    import math

    diam, slope = 0.5, 0.005
    r_full = diam / 4.0
    v_full = (slope**0.5) * r_full ** (2 / 3) / 0.012
    q_full = v_full * (math.pi * diam**2 / 4)

    # Low flow -> shallow depth: slower and a smaller hydraulic radius than full.
    fr, _a, r, v = _partial_flow_hydraulics(0.2 * q_full, diam, slope)
    assert fr < 0.5
    assert v < v_full
    assert r < r_full

    # Half-full (theta = pi): hydraulic radius and velocity coincide with full-pipe.
    fr2, _a2, r2, v2 = _partial_flow_hydraulics(0.5 * q_full, diam, slope)
    assert abs(fr2 - 0.5) < 0.02
    assert abs(r2 - r_full) < 1e-3
    assert abs(v2 - v_full) < 1e-3


def test_calculate_weights():
    """Test weight calculation."""
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=100.0, y=0.0)
    G.add_node(2, x=200.0, y=0.0)
    G.add_edge(0, 1, chahinian_slope=0.5, length=100.0, contributing_area=50.0)
    G.add_edge(1, 2, chahinian_slope=0.2, length=200.0, contributing_area=100.0)
    td = TopologyDerivation()
    result = calculate_weights(G, topology_derivation=td)
    for _, _, d in result.edges(data=True):
        assert "weight" in d
        assert d["weight"] >= 0


# ── Outfall / topology derivation tests ────────────────────────────


def test_derive_topology_reattaches_river_anchored_pond_connector():
    """Pond connectors anchored to river nodes survive derive_topology.

    The network-side endpoint of a pond_connector can be a river node
    (e.g. a canal-adjacent pond).  River nodes are extracted before the
    street-only shortest-path pass and re-added afterward, so the
    reattach test must run against the post-re-add graph, testing the
    pre-re-add survivor set silently drops every river-anchored pond.
    """
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    nodes = {
        1: {"x": 0.0, "y": 0.0},
        2: {"x": 100.0, "y": 0.0},
        3: {"x": 200.0, "y": 0.0},
        4: {"x": 250.0, "y": 0.0},  # river node, outfall target
        5: {"x": 350.0, "y": 0.0},  # river node, pond anchor
        6: {"x": 220.0, "y": 40.0, "node_type": "river_outfall"},  # snapped sink
        10: {"x": 150.0, "y": 80.0, "node_type": "water_body"},
        11: {"x": 50.0, "y": 80.0, "node_type": "water_body"},
        12: {"x": 220.0, "y": 80.0, "node_type": "water_body"},
    }
    for n, attrs in nodes.items():
        G.add_node(n, **attrs)
    for u, v in [(1, 2), (2, 3)]:
        G.add_edge(u, v, edge_type="pipe", weight=1.0, length=100.0)
    G.add_edge(3, 4, edge_type="outfall", weight=5.0, length=5.0)
    G.add_edge(4, 5, edge_type="river", weight=0.0, length=100.0)
    G.add_edge(10, 5, edge_type="pond_connector", length=80.0)  # river anchor
    G.add_edge(11, 2, edge_type="pond_connector", length=80.0)  # street anchor
    G.add_edge(12, 6, edge_type="pond_connector", length=40.0)  # canal-snapped sink

    result = derive_topology(G, topology_derivation=TopologyDerivation())

    assert result.has_edge(10, 5), "river-anchored pond connector was dropped"
    assert result.has_edge(11, 2), "street-anchored pond connector was dropped"
    assert result.has_edge(12, 6), "canal-snapped pond connector was dropped"


def test_identify_outfalls_pairs_to_river_centerline():
    """Street nodes near a river centerline pair to snapped river_outfall sinks.

    River features are noded only at their endpoints, so pairing must
    measure distance to the centerline geometry: here the streets are
    20 m from the canal but ~500 m from either endpoint node.
    """
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    G.add_node(100, x=0.0, y=0.0)
    G.add_node(101, x=1000.0, y=0.0)
    G.add_edge(
        100,
        101,
        edge_type="river",
        length=1000.0,
        weight=0.0,
        geometry=shapely.LineString([(0, 0), (1000, 0)]),
    )
    street = {1: (480.0, 20.0), 2: (540.0, 20.0), 3: (600.0, 20.0)}
    for n, (x, y) in street.items():
        G.add_node(n, x=x, y=y, surface_elevation=5.0)
    for u, v in [(1, 2), (2, 3)]:
        geom = shapely.LineString([street[u], street[v]])
        G.add_edge(u, v, edge_type="pipe", length=60.0, weight=1.0, geometry=geom)
        G.add_edge(v, u, edge_type="pipe", length=60.0, weight=1.0, geometry=geom.reverse())

    od = OutfallDerivation(river_buffer_distance=30.0, outfall_clustering_factor=0.0)
    result = identify_outfalls(G, outfall_derivation=od)

    sinks = [n for n, d in result.nodes(data=True) if d.get("node_type") == "river_outfall"]
    assert sinks, "expected river_outfall sinks for streets near the centerline"
    for s in sinks:
        # Sinks snap onto the centerline (y=0) between the streets, not to
        # the distant feature endpoints.
        assert abs(result.nodes[s]["y"]) < 1e-6
        assert 400.0 < result.nodes[s]["x"] < 700.0
    assert not [n for n, d in result.nodes(data=True) if d.get("node_type") == "dummy_river"]
    outfall_street_sides = {
        u for u, _, d in result.edges(data=True) if d.get("edge_type") == "outfall"
    }
    assert outfall_street_sides <= set(street)


def test_identify_outfalls_conduit_length_is_geometric_not_cost():
    """Retained outfall conduits take the real street->water distance, not the cost.

    Streets sit 20 m from the river centerline, so each river_outfall conduit is
    ~20 m long whatever the clustering penalty works out to, that penalty is
    only the MST selection cost (carried on the edge ``weight``).
    """
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    G.add_node(100, x=0.0, y=0.0)
    G.add_node(101, x=1000.0, y=0.0)
    G.add_edge(
        100,
        101,
        edge_type="river",
        length=1000.0,
        weight=0.0,
        geometry=shapely.LineString([(0, 0), (1000, 0)]),
    )
    street = {1: (480.0, 20.0), 2: (540.0, 20.0), 3: (600.0, 20.0)}
    for n, (x, y) in street.items():
        G.add_node(n, x=x, y=y, surface_elevation=5.0)
    for u, v in [(1, 2), (2, 3)]:
        geom = shapely.LineString([street[u], street[v]])
        G.add_edge(u, v, edge_type="pipe", length=60.0, weight=1.0, geometry=geom)
        G.add_edge(v, u, edge_type="pipe", length=60.0, weight=1.0, geometry=geom.reverse())

    # factor=2 -> penalty = 2 x median pipe length (60) = 120, unrelated to 20 m.
    od = OutfallDerivation(river_buffer_distance=30.0, outfall_clustering_factor=2.0)
    result = identify_outfalls(G, outfall_derivation=od)

    outfall_edges = [
        (u, v, d) for u, v, d in result.edges(data=True) if d.get("edge_type") == "outfall"
    ]
    assert outfall_edges
    for _, _, d in outfall_edges:
        assert abs(d["length"] - d["geometry"].length) < 1e-9  # length tracks geometry
        assert abs(d["length"] - 20.0) < 1.0  # ~20 m to the river, not the cost
        assert d["weight"] == 120.0  # weight carries the scaled clustering cost
        assert d["length"] != d["weight"]  # decoupled from the selection cost


def test_identify_outfalls_clustering_is_scale_invariant():
    """Clustering depends on the dimensionless factor, not the absolute pipe length.

    The same factor yields the same outfall count whether junctions are 20 m or
    200 m apart, because the penalty scales with the network's median pipe length.
    """

    def n_outfalls(spacing: float, factor: float) -> int:
        g = nx.MultiDiGraph()
        g.graph["crs"] = "EPSG:32617"
        g.add_node(900, x=0.0, y=0.0)
        g.add_node(901, x=spacing * 6, y=0.0)
        g.add_edge(
            900,
            901,
            edge_type="river",
            length=spacing * 6,
            weight=0.0,
            geometry=shapely.LineString([(0, 0), (spacing * 6, 0)]),
        )
        coords = {i: (spacing * i, 20.0) for i in range(6)}
        for node, (x, y) in coords.items():
            g.add_node(node, x=x, y=y, surface_elevation=5.0)
        for u in range(5):
            geom = shapely.LineString([coords[u], coords[u + 1]])
            g.add_edge(u, u + 1, edge_type="pipe", length=spacing, weight=1.0, geometry=geom)
            g.add_edge(
                u + 1, u, edge_type="pipe", length=spacing, weight=1.0, geometry=geom.reverse()
            )
        od = OutfallDerivation(river_buffer_distance=30.0, outfall_clustering_factor=factor)
        res = identify_outfalls(g, outfall_derivation=od)
        return sum(1 for _, _, d in res.edges(data=True) if d.get("edge_type") == "outfall")

    # Same factor -> same outfall count at very different absolute scales.
    assert n_outfalls(20.0, 0.5) == n_outfalls(200.0, 0.5)
    assert n_outfalls(20.0, 1.5) == n_outfalls(200.0, 1.5)
    # A larger factor consolidates more (fewer outfalls) at any given scale.
    assert n_outfalls(50.0, 1.5) < n_outfalls(50.0, 0.5)


def test_enforce_outfall_slope_fixes_adverse_outfall():
    """An adverse outfall (sink invert >= street invert) is made gravity-feasible.

    The receiving-water sink is lowered just below the street invert so the
    conduit drains downhill; pond-intake sinks (fixed by pond design) are left
    untouched.
    """
    G = nx.MultiDiGraph()
    # River outfall: street invert 2 m, sink pinned at the water surface 6 m
    # (adverse, the water sits above the street).
    G.add_node(1, chamber_floor_elevation=2.0)
    G.add_node(900, chamber_floor_elevation=6.0, node_type="river_outfall")
    G.add_edge(1, 900, edge_type="outfall", length=20.0)
    # A pond intake that is also adverse, must NOT be touched.
    G.add_node(2, chamber_floor_elevation=2.0)
    G.add_node(950, chamber_floor_elevation=6.0, node_type="water_body")
    G.add_edge(2, 950, edge_type="outfall", pond_intake=True, length=20.0)

    result = enforce_outfall_slope(G, hydraulic_design=HydraulicDesign())

    # River sink lowered to give a small positive slope: 2.0 - 1e-3 * 20 = 1.98.
    new_sink = result.nodes[900]["chamber_floor_elevation"]
    assert abs(new_sink - (2.0 - 1e-3 * 20.0)) < 1e-9
    assert result.nodes[1]["chamber_floor_elevation"] - new_sink > 0  # drains downhill now
    # Pond storage invert untouched.
    assert result.nodes[950]["chamber_floor_elevation"] == 6.0


def test_identify_outfalls_dummy_fallback_when_river_far():
    """Components with no river within the buffer still get a dummy outfall."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    G.add_node(100, x=0.0, y=0.0)
    G.add_node(101, x=1000.0, y=0.0)
    G.add_edge(
        100,
        101,
        edge_type="river",
        length=1000.0,
        weight=0.0,
        geometry=shapely.LineString([(0, 0), (1000, 0)]),
    )
    street = {1: (480.0, 200.0), 2: (540.0, 200.0)}
    for n, (x, y) in street.items():
        G.add_node(n, x=x, y=y, surface_elevation=5.0)
    geom = shapely.LineString([street[1], street[2]])
    G.add_edge(1, 2, edge_type="pipe", length=60.0, weight=1.0, geometry=geom)
    G.add_edge(2, 1, edge_type="pipe", length=60.0, weight=1.0, geometry=geom.reverse())

    od = OutfallDerivation(river_buffer_distance=30.0, outfall_clustering_factor=0.0)
    result = identify_outfalls(G, outfall_derivation=od)

    assert [n for n, d in result.nodes(data=True) if d.get("node_type") == "dummy_river"]
    assert not [n for n, d in result.nodes(data=True) if d.get("node_type") == "river_outfall"]


# ── Simplification tests ───────────────────────────────────────────


def _make_pipe_network() -> nx.MultiDiGraph[Any]:
    """Create a pipe network with degree-2 chains for simplification tests.

    Topology::

        0 --> 1 --> 2 --> 3 --> 4  (main trunk, 1-2-3 are degree-2)
                               |
                               v
                               5  (outfall via river edge)

    Nodes 0 and 4 are branching/terminal, 1-2-3 are degree-2 candidates.
    """
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    nodes = {
        0: {
            "x": 0.0,
            "y": 0.0,
            "surface_elevation": 10.0,
            "chamber_floor_elevation": 8.0,
            "contributing_area": 100.0,
        },
        1: {
            "x": 50.0,
            "y": 0.0,
            "surface_elevation": 9.5,
            "chamber_floor_elevation": 7.5,
            "contributing_area": 80.0,
        },
        2: {
            "x": 100.0,
            "y": 0.0,
            "surface_elevation": 9.0,
            "chamber_floor_elevation": 7.0,
            "contributing_area": 60.0,
        },
        3: {
            "x": 150.0,
            "y": 0.0,
            "surface_elevation": 8.5,
            "chamber_floor_elevation": 6.5,
            "contributing_area": 40.0,
        },
        4: {
            "x": 200.0,
            "y": 0.0,
            "surface_elevation": 8.0,
            "chamber_floor_elevation": 6.0,
            "contributing_area": 20.0,
        },
        5: {
            "x": 200.0,
            "y": -50.0,
            "surface_elevation": 7.0,
            "chamber_floor_elevation": 5.0,
            "contributing_area": 0.0,
        },
    }
    for n, attrs in nodes.items():
        G.add_node(n, **attrs)

    edges = [
        (
            0,
            1,
            {
                "geometry": shapely.LineString([(0, 0), (50, 0)]),
                "length": 50.0,
                "edge_type": "pipe",
                "diameter": 0.3,
                "roughness": 0.012,
                "contributing_area": 100.0,
                "in_offset": 0,
                "out_offset": 0,
                "id": "0-1",
            },
        ),
        (
            1,
            2,
            {
                "geometry": shapely.LineString([(50, 0), (100, 0)]),
                "length": 50.0,
                "edge_type": "pipe",
                "diameter": 0.45,
                "roughness": 0.011,
                "contributing_area": 80.0,
                "in_offset": 0,
                "out_offset": 0,
                "id": "1-2",
            },
        ),
        (
            2,
            3,
            {
                "geometry": shapely.LineString([(100, 0), (150, 0)]),
                "length": 50.0,
                "edge_type": "pipe",
                "diameter": 0.6,
                "roughness": 0.010,
                "contributing_area": 60.0,
                "in_offset": 0,
                "out_offset": 0,
                "id": "2-3",
            },
        ),
        (
            3,
            4,
            {
                "geometry": shapely.LineString([(150, 0), (200, 0)]),
                "length": 50.0,
                "edge_type": "pipe",
                "diameter": 0.9,
                "roughness": 0.010,
                "contributing_area": 40.0,
                "in_offset": 0,
                "out_offset": 0,
                "id": "3-4",
            },
        ),
        (
            4,
            5,
            {
                "geometry": shapely.LineString([(200, 0), (200, -50)]),
                "length": 50.0,
                "edge_type": "river",
                "channel_width": 2.0,
                "channel_depth": 1.0,
                "roughness": 0.035,
                "id": "4-5",
            },
        ),
    ]
    for u, v, d in edges:
        G.add_edge(u, v, **d)
    return G


def test_find_protected_nodes():
    """Protected nodes include river-connected, branching, and terminal nodes."""
    G = _make_pipe_network()
    protected = _find_protected_nodes(G)
    # Node 0: street in-degree=0 (terminal start) -> protected
    assert 0 in protected
    # Node 4: connected to river edge -> protected
    assert 4 in protected
    # Node 5: connected to river edge -> protected
    assert 5 in protected
    # Nodes 1, 2, 3: degree-2 in street subnetwork -> NOT protected
    assert 1 not in protected
    assert 2 not in protected
    assert 3 not in protected


def test_find_degree2_chains():
    """Degree-2 chain detection on a simple linear pipe network."""
    G = _make_pipe_network()
    protected = _find_protected_nodes(G)
    chains = _find_degree2_chains(G, protected)
    assert len(chains) == 1
    chain = chains[0]
    # Chain should be [0, 1, 2, 3, 4]
    assert chain[0] == 0
    assert chain[-1] == 4
    assert set(chain[1:-1]) == {1, 2, 3}


def test_consolidate_chain():
    """Consolidating a degree-2 chain merges into one edge."""
    G = _make_pipe_network()
    protected = _find_protected_nodes(G)
    chains = _find_degree2_chains(G, protected)
    chain = chains[0]

    mapping = _consolidate_chain(G, chain, max_length=500.0)

    # Interior nodes 1, 2, 3 removed
    assert 1 not in G
    assert 2 not in G
    assert 3 not in G
    # Endpoints preserved
    assert 0 in G
    assert 4 in G
    # Single replacement edge 0->4
    assert G.has_edge(0, 4)
    edata = list(G.edges(0, data=True))
    pipe_edges = [d for _, _, d in edata if d.get("edge_type") == "pipe"]
    assert len(pipe_edges) == 1
    merged = pipe_edges[0]
    # Length = 4 * 50 = 200
    assert merged["length"] == 200.0
    # Diameter = max (governing): the merged conduit carries the summed
    # contributing area, so it inherits the largest (downstream-most) segment.
    assert merged["diameter"] == 0.9
    # Contributing area = sum
    assert merged["contributing_area"] == 100.0 + 80.0 + 60.0 + 40.0
    # River edge still intact
    assert G.has_edge(4, 5)
    # Mapping
    assert mapping == {1: 4, 2: 4, 3: 4}


def test_consolidate_chain_splits_long():
    """Chains longer than max_conduit_length are split into segments."""
    G = _make_pipe_network()
    protected = _find_protected_nodes(G)
    chains = _find_degree2_chains(G, protected)
    chain = chains[0]

    # max_length=100 -> 200m chain should split into 2 segments
    _consolidate_chain(G, chain, max_length=100.0)

    # Should have 2 street edges total between surviving nodes
    pipe_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("edge_type") == "pipe"]
    assert len(pipe_edges) == 2
    total_len = sum(d["length"] for _, _, d in pipe_edges)
    assert abs(total_len - 200.0) < 1e-6


def test_remove_dangling_leaves():
    """Dangling leaf with small contributing area is removed."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    # Leaf: 10 -> 11 -> 12 (branching node)
    # Main: 12 -> 13 -> 14, plus 15 -> 12 (makes 12 a branch)
    nodes = {
        10: {"x": 0.0, "y": 50.0, "contributing_area": 5.0},
        11: {"x": 50.0, "y": 50.0, "contributing_area": 3.0},
        12: {"x": 100.0, "y": 0.0, "contributing_area": 200.0},
        13: {"x": 150.0, "y": 0.0, "contributing_area": 100.0},
        14: {"x": 200.0, "y": -50.0, "contributing_area": 0.0},
        15: {"x": 50.0, "y": -50.0, "contributing_area": 150.0},
    }
    for n, attrs in nodes.items():
        G.add_node(n, **attrs)
    G.add_edge(
        10, 11, edge_type="pipe", length=50.0, geometry=shapely.LineString([(0, 50), (50, 50)])
    )
    G.add_edge(
        11, 12, edge_type="pipe", length=70.0, geometry=shapely.LineString([(50, 50), (100, 0)])
    )
    G.add_edge(
        15, 12, edge_type="pipe", length=70.0, geometry=shapely.LineString([(50, -50), (100, 0)])
    )
    G.add_edge(
        12, 13, edge_type="pipe", length=50.0, geometry=shapely.LineString([(100, 0), (150, 0)])
    )
    G.add_edge(
        13, 14, edge_type="river", length=50.0, geometry=shapely.LineString([(150, 0), (200, -50)])
    )

    G, removed = _remove_dangling_leaves(G, min_area=10.0)

    # Leaf chain 10->11 has total CA = 5+3 = 8 < 10, should be removed
    assert 10 not in G
    assert 11 not in G
    assert removed[10] == 12
    assert removed[11] == 12
    # Main chain preserved
    assert 12 in G
    assert 13 in G


def test_simplify_network_disabled():
    """When disabled, simplify_network returns the graph unchanged."""
    G = _make_pipe_network()
    params = SimplificationParams(enabled=False)
    # addresses not used when disabled, pass None
    result = simplify_network(G, addresses=None, simplification=params)
    assert result.number_of_nodes() == G.number_of_nodes()
    assert result.number_of_edges() == G.number_of_edges()


def test_aggregate_subcatchments(tmp_path):
    """Subcatchments for removed nodes are merged to surviving outlet."""
    import geopandas as gpd

    subs = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "geometry": [
                shapely.box(0, 0, 10, 10),
                shapely.box(10, 0, 20, 10),
                shapely.box(20, 0, 30, 10),
            ],
            "area": [100.0, 100.0, 100.0],
            "slope": [1.0, 2.0, 3.0],
            "width": [5.0, 5.0, 5.0],
            "rc": [30.0, 50.0, 70.0],
            "impervious_area": [30.0, 50.0, 70.0],
        },
        crs="EPSG:32617",
    )
    subs_path = tmp_path / "subs.parquet"
    subs.to_parquet(subs_path)

    # Nodes 1 and 2 merged into node 3
    node_mapping = {1: 3, 2: 3}
    _aggregate_subcatchments(subs_path, node_mapping)

    result = gpd.read_parquet(subs_path)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["id"] == 3
    assert row["area"] == 300.0
    assert row["impervious_area"] == 150.0
    # RC: area-weighted average = (30*100 + 50*100 + 70*100) / 300 = 50
    assert abs(row["rc"] - 50.0) < 1e-6


def test_reroute_subs_to_isolated_ponds(tmp_path):
    """Isolated pond gets nearby orphan subs; pipe-fed pond is left alone."""
    import geopandas as gpd

    from swmmanywhere_us.graph_functions.water_bodies import (
        reroute_subs_to_isolated_ponds,
    )
    from swmmanywhere_us.parameters import PondDesign

    # Two ponds:
    #   pond A (id=100) is isolated, no pipes terminate at its ds_node (101)
    #   pond B (id=200) has a pipe feeder via pond_inflow into it
    #
    #     orphan_sub(0,0) ── (nearest pond A) ── pond A storage (10,10)
    #                                          orifice/pond_outflow chain
    #     pipe_sub(110,10) ── pipe ─→ ds_node B(110,0) ─ pipe_inflow ─→ pond B (120,0)
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    # Pond A, isolated
    G.add_node(
        100,
        x=10.0,
        y=10.0,
        node_type="water_body",
        wb_volume_m3=1000.0,
        contributing_area=0.0,
    )
    # Pond A outlet junction + ds_node (no incoming pipes)
    G.add_node(101, x=11.0, y=10.0, node_type="outlet_junction", contributing_area=0.0)
    G.add_node(102, x=20.0, y=10.0, contributing_area=0.0)  # pond A ds_node
    G.add_edge(100, 101, edge_type="orifice")
    G.add_edge(101, 102, edge_type="pond_outflow")

    # Pond B, has pipe inflow (so it is NOT isolated)
    G.add_node(
        200,
        x=120.0,
        y=0.0,
        node_type="water_body",
        wb_volume_m3=1000.0,
        contributing_area=0.0,
    )
    G.add_node(201, x=121.0, y=0.0, node_type="outlet_junction", contributing_area=0.0)
    G.add_node(202, x=130.0, y=0.0, contributing_area=0.0)  # pond B ds_node
    G.add_edge(200, 201, edge_type="orifice")
    G.add_edge(201, 202, edge_type="pond_outflow")
    # Pipe feeding pond B's storage (route_pipes_into_ponds output)
    G.add_node(203, x=110.0, y=10.0, contributing_area=50.0)  # pipe_sub outlet
    G.add_edge(203, 200, edge_type="pond_inflow", length=10.0)

    # Subs file:
    #   sub 999 is the orphan (outlet 999 not on graph and not in any pondshed)
    #   sub 203 drains via pond_inflow to pond B, must NOT be rerouted
    G.add_node(999, x=0.0, y=0.0, contributing_area=42.0)  # orphan outlet on graph
    subs = gpd.GeoDataFrame(
        {
            "id": [999, 203],
            "geometry": [
                shapely.box(-5, -5, 5, 5),  # near pond A
                shapely.box(105, 5, 115, 15),  # already routed to pond B
            ],
            "area": [100.0, 80.0],
        },
        crs="EPSG:32617",
    )
    subs_path = tmp_path / "subs.parquet"
    subs.to_parquet(subs_path)

    # Stub addresses: only needs model_paths.subcatchments + bbox_paths.basins.
    from types import SimpleNamespace

    addresses = SimpleNamespace(
        model_paths=SimpleNamespace(subcatchments=subs_path),
        bbox_paths=SimpleNamespace(basins=tmp_path / "no_basins.parquet"),
    )
    result = reroute_subs_to_isolated_ponds(G, addresses=addresses, pond_design=PondDesign())

    out = gpd.read_parquet(subs_path)
    by_orig = dict(zip(subs["id"], subs["geometry"]))
    rerouted = {row.id for row in out.itertuples() if row.id == 100}
    untouched = {row.id for row in out.itertuples() if row.id == 203}
    assert rerouted == {100}, "orphan sub should be rerouted to isolated pond A (100)"
    assert untouched == {203}, "pipe-fed pond B's sub must stay on its existing outlet"
    # contributing_area transferred from node 999 to pond 100
    assert result.nodes[100]["contributing_area"] == 42.0
    assert result.nodes[999]["contributing_area"] == 0.0
    assert by_orig  # silence unused-var lint


def test_reroute_subs_to_isolated_ponds_capacity_cap(tmp_path):
    """Per-pond capacity cap skips subs over the FDOT 1-inch threshold."""
    import geopandas as gpd

    from swmmanywhere_us.graph_functions.water_bodies import (
        reroute_subs_to_isolated_ponds,
    )
    from swmmanywhere_us.parameters import PondDesign

    # One isolated pond with deliberately tiny capacity:
    # wb_volume_m3 = 1 → cap = 2.0 * 1 / (0.0254 * 0.5) = 157 m²
    # Two candidate orphan subs of 100 m² each, only one fits the cap.
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    G.add_node(
        100,
        x=10.0,
        y=10.0,
        node_type="water_body",
        wb_volume_m3=1.0,  # tiny
        contributing_area=0.0,
    )
    G.add_node(101, x=11.0, y=10.0, node_type="outlet_junction", contributing_area=0.0)
    G.add_node(102, x=20.0, y=10.0, contributing_area=0.0)
    G.add_edge(100, 101, edge_type="orifice")
    G.add_edge(101, 102, edge_type="pond_outflow")
    # Two orphan-sub outlet nodes; sub_near (id=901) is closer to pond, picked first.
    G.add_node(901, x=5.0, y=5.0, contributing_area=33.0)
    G.add_node(902, x=15.0, y=20.0, contributing_area=33.0)

    subs = gpd.GeoDataFrame(
        {
            "id": [901, 902],
            "geometry": [
                shapely.box(0, 0, 10, 10),  # near pond, sub_near
                shapely.box(10, 15, 20, 25),  # farther, sub_far
            ],
            "area": [100.0, 100.0],
        },
        crs="EPSG:32617",
    )
    subs_path = tmp_path / "subs.parquet"
    subs.to_parquet(subs_path)

    from types import SimpleNamespace

    addresses = SimpleNamespace(
        model_paths=SimpleNamespace(subcatchments=subs_path),
        bbox_paths=SimpleNamespace(basins=tmp_path / "no_basins.parquet"),
    )
    reroute_subs_to_isolated_ponds(G, addresses=addresses, pond_design=PondDesign())

    out = gpd.read_parquet(subs_path)
    by_id = dict(zip(subs["id"], out["id"]))
    # Nearest sub should be rerouted; farther sub blocked by capacity cap.
    assert by_id[901] == 100, "nearest sub fits the cap and is rerouted"
    assert by_id[902] == 902, "second sub exceeds pond capacity and stays put"


def test_remove_river_crossing_pipes():
    """Pipe edges crossing a river centerline are removed; others kept."""
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:32617"
    G.add_node(100, x=0.0, y=0.0)
    G.add_node(101, x=1000.0, y=0.0)
    G.add_edge(
        100,
        101,
        edge_type="river",
        length=1000.0,
        geometry=shapely.LineString([(0, 0), (1000, 0)]),
    )
    # Pipe crossing the river (bridge street) and a parallel one that doesn't.
    G.add_node(1, x=500.0, y=50.0)
    G.add_node(2, x=500.0, y=-50.0)
    G.add_node(3, x=600.0, y=50.0)
    G.add_edge(
        1, 2, edge_type="pipe", length=100.0, geometry=shapely.LineString([(500, 50), (500, -50)])
    )
    G.add_edge(
        1, 3, edge_type="pipe", length=100.0, geometry=shapely.LineString([(500, 50), (600, 50)])
    )

    result = remove_river_crossing_pipes(G)

    assert not result.has_edge(1, 2), "crossing pipe should be removed"
    assert result.has_edge(1, 3), "non-crossing pipe should be kept"
    assert result.has_edge(100, 101), "river edge untouched"
    assert 2 in result.nodes, "nodes are kept"


def test_pond_has_viable_outfall_canal_stage_exemption():
    """Stage-carrying terminals need only driving head; others need 2 m.

    A pond 0.3 m above the adjacent canal's water surface drains by
    gravity (stage terminal: 0.1 m margin + min-slope head over the
    path), while the same 0.3 m against a derived-invert terminal stays
    below the 2.0 m criterion.
    """
    from swmmanywhere_us.graph_functions.water_bodies import _pond_has_viable_outfall

    def make_chain(terminal_type, terminal_cfe, path_len=100.0):
        G = nx.MultiDiGraph()
        G.add_node(1, node_type="water_body", surface_elevation=5.0, wb_max_depth=1.0)
        G.add_node(2, node_type="outlet_junction", x=0.0, y=0.0)
        G.add_node(3, node_type=terminal_type, chamber_floor_elevation=terminal_cfe)
        G.add_edge(1, 2, edge_type="orifice", length=0.0)
        G.add_edge(2, 3, edge_type="pond_outflow", length=path_len)
        return G

    # Canal stage 0.3 m below pond max WSE (5.0): required = 0.1 + 0.001*100 = 0.2
    assert _pond_has_viable_outfall(make_chain("river_outfall", 4.7), 1, 2.0)
    # Same head against a derived-invert terminal: 0.3 < 2.0 -> closed basin
    assert not _pond_has_viable_outfall(make_chain(None, 4.7), 1, 2.0)
    # Stage terminal but pond barely above the water surface: 0.05 < 0.2
    assert not _pond_has_viable_outfall(make_chain("river_outfall", 4.95), 1, 2.0)
    # Deep derived-invert terminal still passes the 2 m rule
    assert _pond_has_viable_outfall(make_chain(None, 2.5), 1, 2.0)
