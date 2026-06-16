"""Tests for outfall graph-function helpers."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import networkx as nx
import shapely
from shapely.geometry import box

from swmmanywhere_us.graph_functions.outfall import (
    _pair_rivers,
    _pond_intake_pairs,
    _prune_uphill_matches,
)

if TYPE_CHECKING:
    from pathlib import Path

CRS = "EPSG:32618"


def _basins(tmp_path: Path) -> Path:
    """Two disjoint basin polygons with centroids at (0,0) and (100,0)."""
    gdf = gpd.GeoDataFrame(geometry=[box(-10, -10, 10, 10), box(90, -10, 110, 10)], crs=CRS)
    path = tmp_path / "basins.parquet"
    gdf.to_parquet(path)
    return path


def _addresses(basins_path: Path) -> Any:
    return types.SimpleNamespace(bbox_paths=types.SimpleNamespace(basins=basins_path))


def test_pond_intake_pairs_matches_nearest_basin(tmp_path: Path) -> None:
    """Each pond storage node maps to the basin it sits in, in node order."""
    addresses = _addresses(_basins(tmp_path))
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph.add_node(7, x=0.0, y=0.0, node_type="pipe")  # not a pond -> ignored
    graph.add_node(1000, x=0.0, y=0.0, node_type="water_body", wb_area_m2=5000.0)
    graph.add_node(1001, x=100.0, y=0.0, node_type="water_body", wb_area_m2=5000.0)

    pairs = _pond_intake_pairs(graph, addresses, CRS)

    assert [n for n, _ in pairs] == [1000, 1001]
    assert pairs[0][1].equals(box(-10, -10, 10, 10))
    assert pairs[1][1].equals(box(90, -10, 110, 10))


def test_pond_intake_pairs_drops_distant_storage(tmp_path: Path) -> None:
    """A storage node farther than the default 5 m from any basin is off-line."""
    addresses = _addresses(_basins(tmp_path))
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    # 15.0 from origin -> 5 m from polyA's edge (x=10): kept (boundary inclusive).
    graph.add_node(1, x=15.0, y=0.0, node_type="water_body", wb_area_m2=5000.0)
    # 15.001 -> just over 5 m: dropped.
    graph.add_node(2, x=15.001, y=0.0, node_type="water_body", wb_area_m2=5000.0)
    # Far away: dropped.
    graph.add_node(3, x=500.0, y=500.0, node_type="water_body", wb_area_m2=5000.0)

    pairs = _pond_intake_pairs(graph, addresses, CRS)

    assert [n for n, _ in pairs] == [1]


def test_pond_intake_pairs_footprint_match_is_tunable(tmp_path: Path) -> None:
    """footprint_match_m widens/narrows the storage-to-basin match tolerance."""
    addresses = _addresses(_basins(tmp_path))
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    # 20 m from polyA's edge (x=10): dropped at default 5 m, kept at 25 m.
    graph.add_node(1, x=30.0, y=0.0, node_type="water_body", wb_area_m2=5000.0)

    assert _pond_intake_pairs(graph, addresses, CRS, footprint_match_m=5.0) == []
    assert [n for n, _ in _pond_intake_pairs(graph, addresses, CRS, footprint_match_m=25.0)] == [1]


def test_pond_intake_pairs_respects_min_area(tmp_path: Path) -> None:
    """Ponds below ``min_area_m2`` are excluded entirely."""
    addresses = _addresses(_basins(tmp_path))
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph.add_node(1000, x=0.0, y=0.0, node_type="water_body", wb_area_m2=5000.0)
    graph.add_node(1001, x=100.0, y=0.0, node_type="water_body", wb_area_m2=50.0)

    pairs = _pond_intake_pairs(graph, addresses, CRS, min_area_m2=100.0)

    assert [n for n, _ in pairs] == [1000]


def test_pond_intake_pairs_empty_without_ponds(tmp_path: Path) -> None:
    """No pond nodes -> empty result (basins file untouched)."""
    addresses = _addresses(_basins(tmp_path))
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0, node_type="pipe")

    assert _pond_intake_pairs(graph, addresses, CRS) == []


def test_pair_rivers_matches_nearest_water_body() -> None:
    """Street nodes near a water body pair to a shoreline sink at the real distance.

    Covers the vectorized water-body matching path in _pair_rivers (otherwise
    only exercised end-to-end behind include_water_body_outfalls).
    """
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.graph["crs"] = CRS
    poly = box(0.0, 0.0, 100.0, 100.0)  # shoreline at y=100
    coords = {1: (50.0, 120.0), 2: (50.0, 180.0)}  # 20 m and 80 m from the shore
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y, surface_elevation=5.0, node_type="pipe")
    g.add_edge(
        1,
        2,
        edge_type="pipe",
        length=60.0,
        weight=1.0,
        geometry=shapely.LineString([coords[1], coords[2]]),
    )
    g.add_edge(
        2,
        1,
        edge_type="pipe",
        length=60.0,
        weight=1.0,
        geometry=shapely.LineString([coords[2], coords[1]]),
    )
    pipe_points = {n: shapely.Point(x, y) for n, (x, y) in coords.items()}

    result = _pair_rivers(g, {}, pipe_points, 150.0, 40.0, water_body_polys=[poly])

    wb = [
        d
        for _, v, d in result.edges(data=True)
        if d.get("edge_type") == "outfall"
        and result.nodes[v].get("node_type") == "water_body_outfall"
    ]
    assert len(wb) == 2
    for d in wb:
        assert d["weight"] == 40.0  # clustering cost (selection), not a length
        assert abs(d["length"] - d["geometry"].length) < 1e-9  # length is real distance
    assert sorted(round(d["length"], 1) for d in wb) == [20.0, 80.0]


def test_pair_rivers_pond_intake_edges() -> None:
    """Eligible pipe nodes near a pond pair to its storage node (vectorized path).

    No river is present, so the lowest-elevation node becomes a dummy outfall
    and is excluded from intake; the others pair to the pond storage with an
    ``outfall`` edge whose length is the real pipe->storage distance.
    """
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.graph["crs"] = CRS
    poly = box(0.0, 0.0, 100.0, 100.0)
    g.add_node(900, x=50.0, y=50.0, node_type="water_body")  # pond storage (centroid)
    coords = {1: (50.0, 130.0), 2: (50.0, 160.0), 3: (50.0, 200.0)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y, surface_elevation=float(n), node_type="pipe")
    for u, v in [(1, 2), (2, 3), (3, 900)]:
        line = shapely.LineString(
            [(g.nodes[u]["x"], g.nodes[u]["y"]), (g.nodes[v]["x"], g.nodes[v]["y"])]
        )
        g.add_edge(u, v, edge_type="pipe", length=line.length, weight=1.0, geometry=line)
        g.add_edge(v, u, edge_type="pipe", length=line.length, weight=1.0, geometry=line.reverse())
    pipe_points = {n: shapely.Point(x, y) for n, (x, y) in coords.items()}

    result = _pair_rivers(
        g,
        {},
        pipe_points,
        150.0,
        40.0,
        pond_intake_pairs=[(900, poly)],
        pond_intake_buffer=200.0,
    )

    intakes = {(u, v): d for u, v, d in result.edges(data=True) if d.get("pond_intake")}
    # Node 1 is the lowest-elevation dummy outfall -> not an intake; 2 and 3 are.
    assert set(intakes) == {(2, 900), (3, 900)}
    assert intakes[(2, 900)]["length"] == 110.0  # (50,160) -> (50,50)
    assert intakes[(3, 900)]["length"] == 150.0  # (50,200) -> (50,50)
    assert all(d["weight"] == 40.0 for d in intakes.values())


def test_prune_uphill_matches(tmp_path: "Path") -> None:
    """Matches to receiving water above the street surface are dropped; below are kept."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    # DEM: left half (x < 300) elevation 0, right half (x >= 300) elevation 100.
    n, res = 60, 10.0
    dem = np.zeros((n, n), dtype="float32")
    dem[:, 30:] = 100.0
    dem_path = tmp_path / "dem.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
        crs=CRS, transform=from_origin(0.0, n * res, res, res), nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)

    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_node(1, x=50.0, y=300.0, surface_elevation=5.0, node_type="pipe")
    pipe_points = {1: Point(50.0, 300.0)}

    # Water body in the HIGH half (shoreline DEM ~100 > street 5) -> pruned.
    high_wb = box(400.0, 290.0, 450.0, 310.0)
    kept, pruned = _prune_uphill_matches(
        {1: 0}, [high_wb], pipe_points, g, dem_path, to_exterior=True
    )
    assert kept == {} and pruned == 1

    # Water body in the LOW half (shoreline DEM ~0 < street 5) -> kept.
    low_wb = box(100.0, 290.0, 150.0, 310.0)
    kept2, pruned2 = _prune_uphill_matches(
        {1: 0}, [low_wb], pipe_points, g, dem_path, to_exterior=True
    )
    assert kept2 == {1: 0} and pruned2 == 0
