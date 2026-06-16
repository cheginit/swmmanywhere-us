"""End-to-end tests for the WhiteboxWorkflows (wbw) geospatial pipeline.

Exercises the runtime behavior of ``compute_flow_direction`` and
``delineate_catchment`` after the migration off ``pywbt``, the prior
test suite only checked their signatures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from swmmanywhere_us.geospatial_utilities import (
    compute_flow_direction,
    delineate_catchment,
)

if TYPE_CHECKING:
    from pathlib import Path

CRS = "EPSG:32618"  # projected UTM (meters), delineate_catchment requires this
RES = 10.0
N = 60
ORIGIN_X = 500_000.0
ORIGIN_Y = 4_500_000.0
# WhiteboxTools D8 pointer encoding: powers of two for the 8 directions.
_WBT_PNTR_VALUES = {0, 1, 2, 4, 8, 16, 32, 64, 128}


@pytest.fixture
def dem_path(tmp_path: Path) -> Path:
    """A synthetic DEM: a V-shaped valley draining south."""
    yy, xx = np.mgrid[0:N, 0:N].astype("float64")
    dem = (100.0 + yy * 0.5 + (xx - N / 2) ** 2 * 0.02).astype("float32")
    transform = from_origin(ORIGIN_X, ORIGIN_Y + N * RES, RES, RES)
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=N,
        width=N,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)
    return path


def _assert_valid_fdir(fdir_path: Path) -> None:
    with rasterio.open(fdir_path) as src:
        vals = set(np.unique(src.read(1)).tolist())
    assert vals <= _WBT_PNTR_VALUES, f"unexpected pointer values: {vals - _WBT_PNTR_VALUES}"
    assert vals & {1, 2, 4, 8, 16, 32, 64, 128}, "no flow directions encoded"


def test_compute_flow_direction_no_graph(dem_path: Path, tmp_path: Path) -> None:
    """Without a graph: breach -> D8 pointer, plus slope on the raw DEM."""
    fdir_path = tmp_path / "fdir.tif"
    slope_path = tmp_path / "slope.tif"

    compute_flow_direction(dem_path, fdir_path, slope_path)

    assert fdir_path.exists()
    assert slope_path.exists()
    _assert_valid_fdir(fdir_path)
    # CRS and geotransform must survive the wbw read/write round-trip.
    with rasterio.open(fdir_path) as src:
        assert src.crs.to_epsg() == 32618
        assert src.transform.a == pytest.approx(RES)
    with rasterio.open(slope_path) as src:
        slope = src.read(1)
        finite = slope[np.isfinite(slope)]
        assert finite.size > 0
        assert finite.min() >= 0.0  # percent slope is non-negative


def test_compute_flow_direction_with_burn(dem_path: Path, tmp_path: Path) -> None:
    """With a graph: streets are burned into the DEM (FillBurn) first."""
    fdir_path = tmp_path / "fdir.tif"
    slope_path = tmp_path / "slope.tif"

    graph: nx.Graph = nx.Graph()
    graph.graph["crs"] = CRS
    # A street roughly along the valley centerline.
    cx = ORIGIN_X + (N / 2) * RES
    graph.add_edge(
        0,
        1,
        geometry=LineString([(cx, ORIGIN_Y + 5 * RES), (cx, ORIGIN_Y + 55 * RES)]),
    )

    compute_flow_direction(dem_path, fdir_path, slope_path, graph=graph, rail_path=None)

    assert fdir_path.exists()
    assert slope_path.exists()
    _assert_valid_fdir(fdir_path)


def test_delineate_catchment_maps_node_ids(dem_path: Path, tmp_path: Path) -> None:
    """delineate_catchment delineates basins and persists snapped pour points.

    The snapped shapefile must keep the ``node_id`` attribute and feature
    order so derive_subcatchments can map watershed raster IDs back to nodes.
    """
    fdir_path = tmp_path / "fdir.tif"
    slope_path = tmp_path / "slope.tif"
    compute_flow_direction(dem_path, fdir_path, slope_path)

    cx = ORIGIN_X + (N / 2) * RES
    pour_points = gpd.GeoDataFrame(
        {"node_id": [101, 202, 303]},
        geometry=[
            Point(cx, ORIGIN_Y + 8 * RES),
            Point(cx, ORIGIN_Y + 30 * RES),
            Point(cx, ORIGIN_Y + 52 * RES),
        ],
        crs=CRS,
    )

    subs = delineate_catchment(fdir_path, pour_points, min_drainage_area_m2=2_000)

    assert isinstance(subs, gpd.GeoDataFrame)
    assert "watershed" in subs.columns
    assert len(subs) > 0
    assert subs.geometry.is_valid.all()

    # The snapped pour points are written for derive_subcatchments to read.
    snapped_path = fdir_path.parent / "pour_pts_snapped.shp"
    assert snapped_path.exists()
    snapped = gpd.read_file(snapped_path)
    assert "node_id" in snapped.columns
    assert snapped["node_id"].tolist() == [101, 202, 303]
