"""Geospatial utilities module for SWMManywhere.

A module containing functions to perform a variety of geospatial operations,
such as reprojecting coordinates and handling raster data.
"""

from __future__ import annotations

import math
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, overload

import geopandas as gpd
import networkx as nx
import numpy as np
import orjson as json
import pandas as pd
import pyproj
import rasterio as rst
import rasterio.features as rio_features
import rioxarray as rxr
import shapely
import whitebox_workflows as wb
import xarray as xr
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj.exceptions import CRSError
from pyproj.transformer import Transformer
from rasterio import features
from scipy.ndimage import distance_transform_edt
from shapely import geometry as sgeom
from shapely import ops
from shapely.strtree import STRtree

from swmmanywhere_us.logging import logger

# Sanity bound for percent slope.  WBT Slope with --units=percent returns
# tan(theta) * 100.  Real terrain rarely exceeds 200% (63 deg) outside
# cliffs/quarries, so 500% is a generous ceiling: any cell above it
# indicates a DEM conditioning artifact (stream burn, inlet dig, nodata
# leakage) that should surface as a warning.
_SLOPE_SANITY_MAX_PCT = 500.0

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from shapely import LineString, MultiLineString, MultiPolygon, Polygon
    from shapely.geometry.base import BaseGeometry

    GeomType = TypeVar("GeomType", LineString, MultiLineString, Polygon, MultiPolygon, BaseGeometry)
    GeomArray = NDArray[GeomType]  # pyright:ignore[reportInvalidTypeForm]

    FloatArray = NDArray[np.floating]
    IntArray = NDArray[np.integer]

TransformerFromCRS = lru_cache(Transformer.from_crs)


def get_utm_epsg(
    x: float,
    y: float,
    crs: str | int | pyproj.CRS = "EPSG:4326",
    datum_name: str = "WGS 84",
) -> str:
    """Get the UTM CRS code for a given coordinate.

    Note, this function is taken from GeoPandas and modified to use
    for getting the UTM CRS code for a given coordinate.

    Args:
        x (float): Longitude in crs
        y (float): Latitude in crs
        crs (str | int | pyproj.CRS, optional): The CRS of the input
            coordinates. Defaults to 'EPSG:4326'.
        datum_name (str, optional): The datum name to use for the UTM CRS

    Returns:
        str: Formatted EPSG code for the UTM zone.

    Example:
        >>> get_utm_epsg(-0.1276, 51.5074)
        'EPSG:32630'
    """
    if not isinstance(x, float) or not isinstance(y, float):
        msg = "x and y must be floats"
        raise TypeError(msg)

    try:
        crs = pyproj.CRS(crs)
    except CRSError as exc:
        msg = "Invalid CRS"
        raise ValueError(msg) from exc

    # ensure using geographic coordinates
    if crs.is_geographic:
        lon = x
        lat = y
    else:
        transformer = TransformerFromCRS(crs, 4326, always_xy=True)
        lon, lat = transformer.transform(x, y)
    utm_crs_list = query_utm_crs_info(
        datum_name=datum_name,
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    return f"{utm_crs_list[0].auth_name}:{utm_crs_list[0].code}"


def nearest_node_buffer(
    points1: dict[str, sgeom.Point], points2: dict[str, sgeom.Point], threshold: float
) -> dict[str, str]:
    """Find the nearest node within a given buffer threshold.

    Args:
        points1 (dict): A dictionary where keys are labels and values are
            Shapely points geometries.
        points2 (dict): A dictionary where keys are labels and values are
            Shapely points geometries.
        threshold (float): The maximum distance for a node to be considered
            'nearest'. If no nodes are within this distance, the node is not
            included in the output.

    Returns:
        dict: A dictionary where keys are labels from points1 and values are
            labels from points2 of the nearest nodes within the threshold.
    """
    if not points1 or not points2:
        return {}

    # Convert the keys of points2 to a list
    labels2 = list(points2.keys())

    # Create a spatial index
    tree = STRtree(list(points2.values()))

    # Initialize an empty dictionary to store the matching nodes
    matching = {}

    # Iterate over points1
    for key, geom in points1.items():
        # Find the nearest node in the spatial index to the current geometry
        nearest = tree.nearest(geom)
        nearest_geom = points2[labels2[nearest]]

        # If the nearest node is within the threshold, add it to the
        # matching dictionary
        if geom.buffer(threshold).intersects(nearest_geom):
            matching[key] = labels2[nearest]

    # Return the matching dictionary
    return matching


def raster_to_geodf(raster_path: str | Path) -> gpd.GeoDataFrame:
    """Vectorize a raster file to a GeoDataFrame.

    Each raster cell becomes a polygon vertex; connected regions of the
    same integer value become a single polygon with its raster value
    preserved in a ``watershed`` column.  Polygons from disconnected
    parts of the same watershed are dissolved into a single
    (multi)polygon, and any polygon strictly contained within another
    is dropped as vectorization noise.

    Args:
        raster_path: Path to the raster file.

    Returns:
        GeoDataFrame with a ``watershed`` column holding the raster value
        and clean polygon geometries.
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        msg = f"Raster file not found: {raster_path}"
        raise FileNotFoundError(msg)

    da = rxr.open_rasterio(raster_path)
    if not isinstance(da, xr.DataArray):
        msg = f"Expected a single-band raster, got: {da}"
        raise TypeError(msg)
    da = da.squeeze(drop=True)
    # 4-connectivity: 8-connectivity merges watershed cells touching only
    # at a corner into one polygon pinched at that point, an invalid
    # (ring self-intersection) geometry.  4-connectivity keeps them as
    # separate parts, recombined per watershed by the dissolve below and
    # cleaned by _repair_disconnected_subcatchments.
    shapes = rio_features.shapes(
        source=da.to_numpy(),
        transform=da.rio.transform(),
        connectivity=4,
    )
    geojsons, values = zip(*shapes)
    gdf = gpd.GeoDataFrame(
        {"watershed": values},
        geometry=[shapely.geometry.shape(g) for g in geojsons],
        crs=da.rio.crs,
    )
    gdf = gdf[gdf["watershed"] != da.rio.nodata].copy()

    # Dissolve by watershed value first so each watershed ends up as a
    # single (multi)polygon carrying its raster ID.
    gdf = gdf.dissolve(by="watershed", as_index=False)

    # Drop polygons whose geometry is strictly contained within another
    # (vectorization sometimes produces thin slivers inside larger
    # watersheds).  The ``contains_properly`` test is run at the polygon
    # level, preserving each surviving watershed's ID.
    _, contained_idx = gdf.geometry.sindex.query(gdf.geometry, predicate="contains_properly")
    return gdf.loc[~gdf.index.isin(contained_idx)].reset_index(drop=True)


class Grid:
    """A class to represent a grid."""

    def __init__(
        self,
        affine: rst.Affine,
        shape: tuple[int, int],
        crs: int,
        bbox: tuple[float, float, float, float],
    ):
        """Initialize the Grid class.

        Args:
            affine (rst.Affine): The affine transformation.
            shape (tuple): The shape of the grid.
            crs (int): The CRS of the grid.
            bbox (tuple): The bounding box of the grid.
        """
        self.affine = affine
        self.shape = shape
        self.crs = crs
        self.bbox = bbox


def calculate_slope(
    polys_gdf: gpd.GeoDataFrame, grid: Grid, cell_slopes: np.ndarray
) -> gpd.GeoDataFrame:
    """Calculate the average slope of each polygon.

    Args:
        polys_gdf (gpd.GeoDataFrame): A GeoDataFrame containing polygons with
            columns: 'geometry', 'area', and 'id'.
        grid (Grid): Information of the raster (affine, shape, crs, bbox)
        cell_slopes (np.ndarray): The slopes of each cell in the grid.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with an added
            'slope' column.
    """
    finite_cells = np.isfinite(cell_slopes)
    slopes: dict[Any, float] = {}
    for idx, geom in polys_gdf.geometry.items():
        mask = features.geometry_mask([geom], grid.shape, grid.affine, invert=True)
        vals = cell_slopes[mask & finite_cells]
        # A polygon smaller than one raster cell (e.g. a tiny Voronoi
        # sliver) can cover no finite cell, leave it NaN here and fill
        # it from the median below, so SWMM never receives a NaN slope.
        slopes[idx] = max(float(vals.mean()), 0.0) if vals.size else float("nan")
    slope_series = pd.Series(slopes)
    finite = slope_series[np.isfinite(slope_series)]
    fill = float(finite.median()) if not finite.empty else 0.5
    polys_gdf["slope"] = slope_series.fillna(fill)
    return polys_gdf


def _fill_watershed_gaps(watershed_path: Path) -> None:
    """Fill nodata gaps in a watershed raster with nearest valid watershed ID.

    Cells not assigned to any watershed by WhiteboxTools are filled
    by propagating the nearest assigned cell's watershed ID using
    a distance transform.
    """
    with rst.open(watershed_path) as src:
        data = src.read(1)
        nodata = src.nodata
        meta = src.meta.copy()

    if nodata is None:
        return

    mask = data == nodata
    if not mask.any():
        return

    _, nearest_idx = distance_transform_edt(mask, return_indices=True)  # pyright: ignore[reportGeneralTypeIssues]
    data[mask] = data[nearest_idx[0][mask], nearest_idx[1][mask]]

    with rst.open(watershed_path, "w", **meta) as dst:
        dst.write(data, 1)


def delineate_catchment(
    fdir_path: Path,
    pour_points: gpd.GeoDataFrame,
    min_drainage_area_m2: float = 100_000,
) -> gpd.GeoDataFrame:
    """Delineate subcatchments from a flow direction raster and pour points.

    Uses WhiteboxTools to extract streams, snap pour points (street inlet
    locations) to the stream network, and delineate watersheds. Cells
    whose flow paths don't reach any pour point are filled by nearest-
    neighbor propagation to ensure full domain coverage.

    Args:
        fdir_path: Path to the D8 flow direction raster (WhiteboxTools format).
        pour_points: GeoDataFrame of pour point locations.
        min_drainage_area_m2: Minimum upstream drainage area in square meters
            for a cell to be classified as a stream. Controls basin granularity.

    Returns:
        GeoDataFrame with subcatchment polygon geometries.
    """
    fdir = Path(fdir_path)
    if not fdir.exists():
        msg = f"Flow direction raster not found: {fdir}"
        raise FileNotFoundError(msg)

    da = rxr.open_rasterio(fdir)
    if not isinstance(da, xr.DataArray):
        msg = f"Expected a single-band raster, got: {da}"
        raise TypeError(msg)
    da = da.squeeze(drop=True)
    crs = da.rio.crs
    if not crs.is_projected:
        msg = f"Flow direction raster CRS must be projected, got: {crs}"
        raise ValueError(msg)
    units_per_meter = crs.linear_units_factor[1]
    cell_size_m = math.ceil(abs(da.rio.resolution()[0]) * units_per_meter)
    snap_dist = cell_size_m
    cell_area = snap_dist**2
    stream_accum_threshold = math.ceil(min_drainage_area_m2 / cell_area)
    da.close()

    pour_pts_file = fdir.parent / "pour_points.shp"
    pour_points.to_file(pour_pts_file)

    # Memory-first WhiteboxWorkflows chain: flow accumulation from the D8
    # pointer -> stream extraction -> snap pour points onto the stream
    # network -> watershed delineation.  The snapped pour points, the
    # watershed raster, and the flow-accumulation raster are written to
    # disk; the stream raster stays in memory.  flow_accum.tif must be
    # persisted because ``assign_channel_geometry`` runs as a separate,
    # later pipeline step and reads it back to size river channels from
    # true upstream watershed area -- without it, channels silently fall
    # back to the (tiny) impervious-area default.
    #
    # No post-Watershed MajorityFilter.  A 3x3 majority filter on the
    # watershed-id raster overwrites every watershed smaller than ~5
    # cells with its surroundings, erasing it, which orphaned ~18% of
    # pour points here.  Those orphans were then Voronoi-split back into
    # neighboring catchments, slicing them with long straight lines.
    # Raster speckle is instead repaired, watershed-aware, by
    # _repair_disconnected_subcatchments (longest-shared-border merge).
    wbe = wb.WbEnvironment()
    d8 = wbe.read_raster(str(fdir))
    flow_accum = wbe.hydrology.d8_flow_accum(input=d8, input_is_pointer=True, out_type="cells")
    streams = wbe.streams.extract_streams(
        flow_accumulation=flow_accum, threshold=stream_accum_threshold
    )
    snapped = wbe.hydrology.jenson_snap_pour_points(
        pour_pts=wbe.read_vector(str(pour_pts_file)),
        streams=streams,
        snap_dist=snap_dist,
    )
    # Persist the snapped pour points: derive_subcatchments reads this file
    # back to recover the node_id -> watershed-id mapping.  Watershed IDs
    # are 1-based in pour-point feature order, so the same ``snapped`` object
    # feeds Watershed to keep the raster IDs and shapefile rows aligned.
    wbe.write_vector(snapped, str(fdir.parent / "pour_pts_snapped.shp"))
    watershed = wbe.hydrology.watershed(d8_pntr=d8, pour_pts=snapped)

    # Persist the flow-accumulation raster (cell counts) for the later
    # assign_channel_geometry step (graph_functions/design.py).
    wbe.write_raster(flow_accum, str(fdir.parent / "flow_accum.tif"))

    watershed_path = fdir.parent / "watershed.tif"
    wbe.write_raster(watershed, str(watershed_path))

    # Fill gaps: cells whose D8 flow paths don't reach any pour point
    # are assigned to the nearest delineated watershed.
    _fill_watershed_gaps(watershed_path)

    return raster_to_geodf(watershed_path)


def compute_flow_direction(
    fid: Path,
    fdir_path: Path,
    slope_path: Path,
    graph: nx.Graph[Any] | None = None,
    rail_path: Path | None = None,
) -> None:
    """Condition a DEM and compute flow direction and slope using WhiteboxTools.

    Optionally burns a street network and rail lines into the DEM before
    computing flow direction (D8 pointer).

    Slope is computed on the ORIGINAL hydroflattened DEM (not the
    stream-burned / breached DEM).  Stream burning (Hellweger 1997 AGREE;
    Saunders 1999 FillBurn) drops stream cells by tens to hundreds of
    meters by construction.  Differentiating across those discontinuities
    yields slopes of 10,000%+, artifacts of flow-routing conditioning,
    not terrain.  Callow et al. 2007 (J. Hydrol. 332:30-39) measured that
    stream burning caused "the largest increase in individual cell slope"
    of any tested conditioning method; Lindsay 2016 (Earth Surf. Proc.
    Landforms 41:658-668) documents the artifact and recommends
    topology-preserving variants explicitly to avoid terrain-derivative
    contamination.  Maidment 2002 (Arc Hydro) treats slope and flow
    direction as independent derived products: the raw DEM feeds the
    slope grid, only the AGREE/burned DEM feeds flow direction.  Running
    Slope here on the pre-conditioning DEM is the Arc Hydro convention.

    Note: the "inlet depression digging" DEM conditioning of Si et al.
    (2024) is intentionally not applied.  The physical drop-to-inlet is
    represented hydraulically by the ``add_manhole_drops`` pipeline step
    (SWMM OutOffset), which does not perturb watershed delineation.

    Args:
        fid: Filepath to the DEM.
        fdir_path: Filepath to save the flow direction raster.
        slope_path: Filepath to save the slope raster.
        graph: The input graph with edges containing 'geometry' for
            burning into the DEM. If None, no burning is performed.
        rail_path: Filepath to the rail lines for burning into
            the DEM. If None, no rail burning is performed.
    """
    wbe = wb.WbEnvironment()
    dem = wbe.read_raster(str(fid))

    # Slope is always computed on the original hydroflattened DEM.
    slope = wbe.terrain.slope(input=dem, units="percent")
    wbe.write_raster(slope, str(slope_path))

    if graph is None:
        conditioned = dem
    else:
        # Step 1: Burn streets/rail into the DEM (stream burning, AGREE).
        # FillBurn takes the burn lines as a vector, so write them to a
        # temporary shapefile and read them back as a WbW vector.  The burned
        # DEM returned by FillBurn is a memory object, so it outlives the
        # temporary directory.
        with tempfile.TemporaryDirectory(dir=str(fid.parent)) as temp_dir:
            busn_path = Path(temp_dir) / "busn.shp"
            busn_gdf = gpd.GeoDataFrame(
                geometry=list(nx.get_edge_attributes(graph, "geometry").values()),
                crs=graph.graph["crs"],
            )
            if rail_path is not None and Path(rail_path).exists():
                rail_lines = ops.linemerge(
                    shapely.node(gpd.read_parquet(rail_path).geometry.union_all())
                )
                rail_lines = shapely.get_parts(rail_lines)
                rail_lines = shapely.set_precision(rail_lines, 1e-3)
                all_geoms = ops.linemerge(
                    shapely.node(shapely.union_all([*busn_gdf.geometry.to_numpy(), *rail_lines]))
                )
                all_geoms = shapely.get_parts(all_geoms)
                gpd.GeoDataFrame(geometry=all_geoms, crs=graph.graph["crs"]).to_file(busn_path)
            else:
                busn_gdf[["geometry"]].to_file(busn_path)
            conditioned = wbe.hydrology.fill_burn(dem=dem, streams=wbe.read_vector(str(busn_path)))

    # Breach remaining depressions (--dist=150 --min_dist --fill), then
    # compute the D8 flow-direction pointer.
    breached = wbe.hydrology.breach_depressions_least_cost(
        dem=conditioned, max_dist=150, minimize_dist=True, fill_deps=True
    )
    fdir = wbe.hydrology.d8_pointer(dem=breached)
    wbe.write_raster(fdir, str(fdir_path))

    if not Path(fdir_path).exists():
        msg = "Flow direction raster not created."
        raise ValueError(msg)

    if not Path(slope_path).exists():
        msg = "Slope raster not created."
        raise ValueError(msg)
    with rst.open(slope_path) as src:
        slope_data = src.read(1)
        slope_valid = slope_data[np.isfinite(slope_data)]
        if slope_valid.size > 0:
            slope_max = float(slope_valid.max())
            if slope_max > _SLOPE_SANITY_MAX_PCT:
                logger.warning(
                    f"Slope raster max = {slope_max:.0f}%, implausibly high "
                    f"(> {_SLOPE_SANITY_MAX_PCT}%).  Slope should be computed on "
                    "a natural DEM; check for elevation artifacts."
                )


def _repair_disconnected_subcatchments(
    polys_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Reassign detached sub parts to the adjacent sub (preserve area).

    For every subcatchment whose geometry is a MultiPolygon, the
    largest part keeps the subcatchment's ``id``; each smaller detached
    part is merged into the nearest *other* subcatchment.  Total area
    is conserved.  Returns a GeoDataFrame with one contiguous geometry
    per ``id``.

    Operates on numpy/shapely arrays (not pandas iteration) so the
    geometry typing stays concrete.
    """
    src_crs = polys_gdf.crs
    ids = np.asarray(polys_gdf["id"].to_numpy())
    geoms = np.asarray(polys_gdf.geometry.to_numpy())
    parts, part_index = shapely.get_parts(geoms, return_index=True)
    if len(parts) == len(geoms):
        return polys_gdf  # every subcatchment already single-part

    part_ids = ids[part_index]
    part_areas = shapely.area(parts)

    # Largest part of each id is the keeper; smaller parts are orphans.
    order = np.argsort(-part_areas)
    seen: set[Any] = set()
    keep_idx: list[int] = []
    orphan_idx: list[int] = []
    for i in order.tolist():
        pid = part_ids[i]
        if pid in seen:
            orphan_idx.append(i)
        else:
            seen.add(pid)
            keep_idx.append(i)
    if not orphan_idx:
        return polys_gdf

    keeper_ids = part_ids[keep_idx]
    keeper_geoms = parts[keep_idx]
    tree = shapely.STRtree(keeper_geoms)
    new_geoms: dict[Any, list[Any]] = {
        kid: [g] for kid, g in zip(keeper_ids.tolist(), keeper_geoms)
    }
    for i in orphan_idx:
        og = parts[i]
        own = part_ids[i]
        cand = np.atleast_1d(tree.query_nearest(og))
        target_id = own  # default: truly isolated sliver keeps its id
        best_d = float("inf")
        for pos in cand.tolist():
            kid = keeper_ids[pos]
            if kid == own:
                continue
            d = float(shapely.distance(og, keeper_geoms[pos]))
            if d < best_d:
                best_d = d
                target_id = kid
        new_geoms.setdefault(target_id, []).append(og)

    out_ids = list(new_geoms.keys())
    out_geoms = [shapely.unary_union(gs) for gs in new_geoms.values()]
    return gpd.GeoDataFrame({"id": out_ids}, geometry=out_geoms, crs=src_crs)


def _merge_nested_subcatchments(  # noqa: C901 - enclosure scan + chain resolution + attribute-conserving merge is one coherent pass
    polys_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Dissolve fully-enclosed (nested) subcatchments into their container.

    WBT can delineate a small watershed completely surrounded by a
    larger one, a pocket, and the dissolve/rehome merge steps can
    likewise leave one subcatchment enclosing another.  Each
    fully-enclosed subcatchment is dissolved into its smallest enclosing
    neighbor: ``area`` and ``impervious_area`` are summed, ``rc`` and
    ``slope`` become area-weighted means, ``width`` = sqrt(area / pi).
    Columns absent from the input are simply not produced.
    """
    n = len(polys_gdf)
    if n < 2:
        return polys_gdf
    geoms = list(polys_gdf.geometry.to_numpy())
    ids = list(polys_gdf["id"].to_numpy())

    def _footprint(g: Any) -> Any:
        parts = g.geoms if g.geom_type == "MultiPolygon" else [g]
        return shapely.unary_union([shapely.Polygon(p.exterior) for p in parts])

    filled = [_footprint(g) for g in geoms]
    areas = np.array([float(shapely.area(g)) for g in geoms])
    tree = shapely.STRtree(np.asarray(filled, dtype=object))

    # container[i] = index of the smallest subcatchment that encloses i.
    container = [-1] * n
    for i in range(n):
        best, best_area = -1, float("inf")
        for j in tree.query(geoms[i], predicate="within").tolist():
            if j != i and areas[i] < areas[j] < best_area:
                best, best_area = j, float(areas[j])
        container[i] = best
    if all(c < 0 for c in container):
        return polys_gdf

    # Follow chains (a pocket inside a pocket) to a non-enclosed root.
    # ``container`` points to strictly larger areas, so this terminates.
    def _root(i: int) -> int:
        while container[i] >= 0:
            i = container[i]
        return i

    grouped: dict[int, list[int]] = {}
    for i in range(n):
        grouped.setdefault(_root(i), []).append(i)

    has = {c: c in polys_gdf.columns for c in ("area", "width", "rc", "slope", "impervious_area")}
    rc_src = polys_gdf["rc"].to_numpy(dtype=float) if has["rc"] else None
    slope_src = polys_gdf["slope"].to_numpy(dtype=float) if has["slope"] else None
    imp_src = polys_gdf["impervious_area"].to_numpy(dtype=float) if has["impervious_area"] else None

    out_rows: list[dict[str, Any]] = []
    for r, members in grouped.items():
        w = areas[members]
        tot = float(w.sum())
        rec: dict[str, Any] = {
            "id": ids[r],
            "geometry": shapely.unary_union([geoms[m] for m in members]),
        }
        if has["area"]:
            rec["area"] = tot
        if has["width"]:
            rec["width"] = float(np.sqrt(tot / np.pi)) if tot > 0 else 0.0
        if rc_src is not None:
            rec["rc"] = float(np.average(rc_src[members], weights=w)) if tot > 0 else 0.0
        if slope_src is not None:
            rec["slope"] = float(np.average(slope_src[members], weights=w)) if tot > 0 else 0.0
        if imp_src is not None:
            rec["impervious_area"] = float(imp_src[members].sum())
        out_rows.append(rec)

    logger.info(
        f"_merge_nested_subcatchments: dissolved {n - len(out_rows)} enclosed "
        f"subcatchment(s); {n} -> {len(out_rows)}."
    )
    cols = [
        "id",
        *[c for c in ("area", "slope", "width", "rc", "impervious_area") if has[c]],
        "geometry",
    ]
    out = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=polys_gdf.crs)
    return out[cols]


def _components_within_tol(parts: np.ndarray, tol_m: float) -> list[np.ndarray]:
    """Group part geometries whose ``tol_m`` buffers overlap.

    Returns a list of index arrays, one per connected component.
    """
    n = int(parts.size)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return int(a)

    buf = shapely.buffer(parts, tol_m)
    tree = shapely.STRtree(buf)
    # predicate="intersects" refines past the STRtree bbox prefilter so
    # parts are only merged when their tol buffers actually overlap
    # (bbox overlap alone would fuse genuinely-separated parts).
    for i in range(n):
        for j in tree.query(buf[i], predicate="intersects").tolist():
            if j > i:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(v) for v in groups.values()]


def _explode_into_components(
    geoms: np.ndarray,
    ids: np.ndarray,
    tol_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Explode each row into tol-merged connected components.

    Returns ``(comp_geom, comp_area, comp_row, comp_rid, resolved)`` where
    ``comp_rid`` is the working outlet id (each row's largest component
    is the immovable keeper, ``resolved=True``; smaller detached
    components start unresolved).
    """
    cg: list[Any] = []
    ca: list[float] = []
    cr: list[int] = []
    keep: list[bool] = []
    for r in range(len(geoms)):
        g = geoms[r]
        if g is None or g.is_empty:
            continue
        parts = shapely.get_parts(np.asarray([g], dtype=object))
        if parts.size == 1:
            merged = [parts[0]]
        else:
            merged = [
                shapely.unary_union(parts[idx]) if idx.size > 1 else parts[idx[0]]
                for idx in _components_within_tol(parts, tol_m)
            ]
        areas = [float(shapely.area(m)) for m in merged]
        keeper = int(np.argmax(areas))
        for k, (m, a) in enumerate(zip(merged, areas)):
            cg.append(m)
            ca.append(a)
            cr.append(r)
            keep.append(k == keeper)
    comp_geom = np.asarray(cg, dtype=object)
    comp_row = np.asarray(cr, dtype="int64")
    comp_rid = ids[comp_row].copy() if comp_row.size else np.asarray([], dtype=ids.dtype)
    return (
        comp_geom,
        np.asarray(ca, dtype=float),
        comp_row,
        comp_rid,
        np.asarray(keep, dtype=bool),
    )


def _floodfill_component_ids(
    comp_geom: np.ndarray,
    comp_rid: np.ndarray,
    resolved: np.ndarray,
    tol_m: float,
) -> None:
    """In place: detached components inherit a bordering keeper's id.

    Iterative flood-fill, an unresolved component adopts the resolved
    neighbor it shares the longest buffered border with, so chains of
    detached parts resolve back through space they actually touch.  True
    islands (no bordering resolved component) fall back to the nearest
    resolved component by polygon distance.
    """
    progress = True
    while progress and not resolved.all():
        progress = False
        ridx = np.flatnonzero(resolved)
        tree = shapely.STRtree(comp_geom[ridx])
        for i in np.flatnonzero(~resolved):
            gi = comp_geom[i]
            cand = tree.query(shapely.buffer(gi, tol_m))
            if len(cand) == 0:
                continue
            best_j, best_ov = -1, -1.0
            for c in np.atleast_1d(cand).tolist():
                j = int(ridx[c])
                ov = float(
                    shapely.area(
                        shapely.intersection(
                            shapely.buffer(gi, tol_m),
                            shapely.buffer(comp_geom[j], tol_m),
                        )
                    )
                )
                if ov > best_ov:
                    best_ov, best_j = ov, j
            # require a real (non-zero) shared border, not just a
            # bbox-prefilter hit, before adopting the neighbor's id,
            # otherwise a component could glue to a sub it doesn't touch
            # and stay disconnected.  Untouched comps wait for a later
            # iteration (a closer comp resolves) or the island fallback.
            if best_j >= 0 and best_ov > 0.0:
                comp_rid[i] = comp_rid[best_j]
                resolved[i] = True
                progress = True
    if not resolved.all():
        ridx = np.flatnonzero(resolved)
        tree = shapely.STRtree(comp_geom[ridx])
        for i in np.flatnonzero(~resolved):
            nn = np.atleast_1d(tree.query_nearest(comp_geom[i]))
            comp_rid[i] = comp_rid[int(ridx[int(nn[0])])]
            resolved[i] = True


def rehome_detached_components(
    gdf: gpd.GeoDataFrame,
    tol_m: float = 1.0,
) -> gpd.GeoDataFrame:
    """Make every subcatchment a single contiguous polygon (mass-conserving).

    Merge steps (``_aggregate_subcatchments``,
    ``cleanup_orphan_subcatchments``) regroup subs onto one outlet via
    the *pipe* node mapping, so ``unary_union`` yields a disconnected
    MultiPolygon when the grouped subs are spatially separated, which
    propagates straight into the pondsheds built from them.  Here each
    subcatchment's largest connected component (parts within ``tol_m``
    count as one component) keeps its ``id``; every detached component
    is flood-filled into the neighboring subcatchment it shares the
    longest border with.  Because each re-homed component physically
    borders the component it joins, every resulting subcatchment is
    contiguous.

    ``area`` and ``impervious_area`` are conserved (extensive, summed
    from each component's source-row impervious density); ``rc`` and
    ``slope`` are recombined as area-weighted means; ``width`` =
    sqrt(area / pi).  Finally any subcatchment fully enclosed by another
    is dissolved into it (``_merge_nested_subcatchments``) so the merge
    leaves no subcatchment carrying a hole.
    """
    if gdf.empty:
        return gdf
    geoms = gdf.geometry.to_numpy()
    ids = np.asarray(gdf["id"].to_numpy())

    comp_geom, comp_area, comp_row, comp_rid, resolved = _explode_into_components(geoms, ids, tol_m)
    if comp_geom.size == 0 or bool(resolved.all()):
        # Nothing detached, but a sub can still enclose another.
        return _merge_nested_subcatchments(gdf)

    _floodfill_component_ids(comp_geom, comp_rid, resolved, tol_m)

    has = {c: c in gdf.columns for c in ("area", "width", "rc", "slope", "impervious_area")}
    n_rows = len(gdf)
    row_area = np.zeros(n_rows)
    np.add.at(row_area, comp_row, comp_area)
    rc_src = gdf["rc"].to_numpy(dtype=float) if has["rc"] else None
    slope_src = gdf["slope"].to_numpy(dtype=float) if has["slope"] else None
    imp_src = gdf["impervious_area"].to_numpy(dtype=float) if has["impervious_area"] else None

    out_rows: list[dict[str, Any]] = []
    for tid in dict.fromkeys(comp_rid.tolist()):
        sel = np.flatnonzero(comp_rid == tid)
        w = comp_area[sel]
        srows = comp_row[sel]
        tot = float(w.sum())
        rec: dict[str, Any] = {
            "id": tid,
            "geometry": shapely.unary_union(comp_geom[sel]),
        }
        if has["area"]:
            rec["area"] = tot
        if has["width"]:
            rec["width"] = float(np.sqrt(tot / np.pi)) if tot > 0 else 0.0
        if rc_src is not None:
            rec["rc"] = float(np.average(rc_src[srows], weights=w)) if tot > 0 else 0.0
        if slope_src is not None:
            rec["slope"] = float(np.average(slope_src[srows], weights=w)) if tot > 0 else 0.0
        if imp_src is not None:
            dens = np.divide(
                imp_src[srows],
                row_area[srows],
                out=np.zeros(len(srows)),
                where=row_area[srows] > 0,
            )
            rec["impervious_area"] = float((dens * w).sum())
        out_rows.append(rec)

    cols = [
        "id",
        *[c for c in ("area", "slope", "width", "rc", "impervious_area") if has[c]],
        "geometry",
    ]
    out = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=gdf.crs)
    return _merge_nested_subcatchments(out[cols])


def _clean_polygon(geom: Any) -> Any | None:
    """Return a valid polygonal geometry, or ``None`` if it has no area.

    ``shapely.intersection`` of a Voronoi cell with a subcatchment can
    yield an invalid (self-touching) polygon, or, when they meet only
    along an edge, a line/point/GeometryCollection.  A subcatchment
    geometry must be a valid Polygon/MultiPolygon, otherwise downstream
    raster sampling (``calculate_slope``) produces NaN and SWMM runoff
    blows up.  This repairs the geometry and keeps only its area parts.
    """
    if geom is None or shapely.is_empty(geom):
        return None
    if not shapely.is_valid(geom):
        geom = shapely.make_valid(geom)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom if not shapely.is_empty(geom) else None
    # GeometryCollection / line / point leftovers, keep polygonal parts.
    polys = [
        g
        for g in shapely.get_parts(np.asarray([geom], dtype=object))
        if g.geom_type in ("Polygon", "MultiPolygon")
    ]
    if not polys:
        return None
    out = shapely.union_all(polys)
    return out if not shapely.is_empty(out) else None


def _voronoi_pieces(
    poly: Any,
    pts: list[Any],
    node_ids: list[int],
) -> list[tuple[int, Any]]:
    """Voronoi-partition ``poly`` among ``pts``.

    Returns ``(node_id, piece)`` pairs, each piece is a Voronoi cell
    clipped to ``poly`` and repaired to a valid polygon.  The cells tile
    ``poly`` exactly, so the split conserves area.
    """
    if not shapely.is_valid(poly):
        poly = shapely.make_valid(poly)
    mp = shapely.multipoints(np.asarray(pts, dtype=object))
    cells = shapely.get_parts(shapely.voronoi_polygons(mp, extend_to=poly))
    if cells.size == 0:
        cleaned = _clean_polygon(poly)
        return [(node_ids[0], cleaned)] if cleaned is not None else []
    tree = shapely.STRtree(cells)
    pieces: list[tuple[int, Any]] = []
    for pt, nid in zip(pts, node_ids):
        hit = tree.query(pt, predicate="within")
        idx = int(hit[0]) if len(hit) else int(np.atleast_1d(tree.query_nearest(pt))[0])
        piece = _clean_polygon(shapely.intersection(cells[idx], poly))
        if piece is not None and shapely.area(piece) > 0.0:
            pieces.append((nid, piece))
    return pieces


def _split_merged_watersheds(
    polys_gdf: gpd.GeoDataFrame,
    pour_points: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Give every pour point its own subcatchment.

    WBT ``Watershed`` produces one polygon per *snapped* pour-point
    cell, so graph nodes that ``JensonSnapPourPoints`` snapped into a
    shared cell all fall inside one polygon and only one of them carries
    an ``id``, the rest get no subcatchment.  Every polygon that
    geometrically contains more than one pour point is Voronoi-split
    among those points (clipped to the polygon, area conserved), so each
    junction ends up with its own contiguous subcatchment.  Pour points
    inside no polygon (e.g. over water bodies) are left without one.
    """
    poly_ids = [int(p) for p in polys_gdf["id"].to_numpy().tolist()]
    poly_by_id = dict(zip(poly_ids, polys_gdf.geometry.to_numpy()))
    pp_geom = {
        int(n): g
        for n, g in zip(pour_points["node_id"].to_numpy(), pour_points.geometry.to_numpy())
    }

    # Seed each polygon with its own outlet, then absorb the subless
    # graph nodes (not themselves a polygon id) that fall inside it.
    group: dict[int, list[int]] = {p: [p] for p in poly_ids}
    subless = pour_points.loc[~pour_points["node_id"].isin(set(poly_ids)), ["node_id", "geometry"]]
    if not subless.empty:
        joined = gpd.sjoin(
            subless, polys_gdf[["id", "geometry"]], predicate="intersects", how="inner"
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        for nid, poly_id in zip(joined["node_id"].to_numpy(), joined["id"].to_numpy()):
            group[int(poly_id)].append(int(nid))

    out_ids: list[int] = []
    out_geoms: list[Any] = []
    n_split = 0
    dropped = 0
    for pid, members in group.items():
        pts = [pp_geom[m] for m in members if m in pp_geom]
        mids = [m for m in members if m in pp_geom]
        if len(pts) < 2:
            out_ids.append(pid)
            out_geoms.append(poly_by_id[pid])
            continue
        pieces = _voronoi_pieces(poly_by_id[pid], pts, mids)
        for nid, piece in pieces:
            out_ids.append(nid)
            out_geoms.append(piece)
        dropped += len(mids) - len({nid for nid, _ in pieces})
        n_split += 1

    logger.info(
        f"derive_subcatchments: Voronoi-split {n_split} merged watershed(s); "
        f"{len(polys_gdf)} -> {len(out_ids)} subcatchments."
    )
    if dropped:
        logger.warning(
            f"derive_subcatchments: {dropped} pour point(s) lost their subcatchment to "
            f"degenerate Voronoi slivers; their runoff is rerouted by "
            f"cleanup_orphan_subcatchments."
        )
    return gpd.GeoDataFrame({"id": out_ids}, geometry=out_geoms, crs=polys_gdf.crs)


def derive_subcatchments(
    graph: nx.Graph[Any],
    fdir_path: Path,
    slope_path: Path,
    min_drainage_area_m2: float = 100_000,
) -> gpd.GeoDataFrame:
    """Derive subcatchments from graph nodes and pre-computed rasters.

    Expects flow direction and slope rasters to already exist on disk
    (produced by ``compute_flow_direction`` in a prior pipeline step).

    Args:
        graph: The input graph with nodes containing 'x' and 'y'.
        fdir_path: Path to the D8 flow direction raster.
        slope_path: Path to the slope raster.
        min_drainage_area_m2: Minimum upstream drainage area (m²) for
            stream extraction. Controls subcatchment granularity.

    Returns:
        GeoDataFrame with columns: 'geometry', 'area', 'id', 'width', 'slope'.
    """
    # Build pour points from every graph node with coordinates (see below).
    with rst.open(fdir_path) as src:
        bbox = sgeom.box(*src.bounds)
        grid = Grid(src.transform, (src.height, src.width), src.crs, src.bounds)

    x_attr = nx.get_node_attributes(graph, "x")
    y_attr = nx.get_node_attributes(graph, "y")
    # Every graph node with coordinates is a pour point.  Using all nodes
    # (not just degree>2 intersections) gives each node its own small
    # watershed.  That eliminates the "multiple manholes in one subcatchment"
    # ambiguity and aligns the surface routing with the pipe-network
    # topology, so pipe sizing (which accumulates per-node contributing
    # area along the pipe tree) matches SWMM's subcatchment-to-Outlet
    # routing exactly.
    pour_data = [
        (u, float(x_attr[u]), float(y_attr[u]))
        for u in graph.nodes
        if u in x_attr
        and u in y_attr
        and shapely.contains_xy(bbox, float(x_attr[u]), float(y_attr[u]))
    ]
    if not pour_data:
        msg = "No pour points found (no graph nodes with coordinates within raster bbox)."
        raise ValueError(msg)
    node_ids, xs, ys = zip(*pour_data)
    # Attach node_id as an attribute so WBT's JensonSnapPourPoints preserves
    # the mapping through snapping, the snapped shapefile retains this
    # column, letting us look up the outlet by spatial ``contains`` rather
    # than (unreliable) nearest-neighbor.
    pour_points = gpd.GeoDataFrame(
        {"node_id": list(node_ids)},
        geometry=gpd.points_from_xy(xs, ys),
        crs=graph.graph["crs"],
    )

    # Delineate catchments.  The returned polygons carry a ``watershed``
    # column whose value matches the 1-based row index (FID+1) of the
    # corresponding pour point in the snapped shapefile, exactly the
    # mapping WBT's Watershed tool writes into its output raster.
    polys_gdf = delineate_catchment(fdir_path, pour_points, min_drainage_area_m2)

    snapped_path = Path(fdir_path).parent / "pour_pts_snapped.shp"
    snapped_pour = gpd.read_file(snapped_path)
    if snapped_pour.crs is None:
        snapped_pour.set_crs(graph.graph["crs"], inplace=True)
    elif str(snapped_pour.crs) != str(graph.graph["crs"]):
        snapped_pour = snapped_pour.to_crs(graph.graph["crs"])

    # Raster value V corresponds to the (V-1)-th row of the snapped
    # pour-points shapefile, which carries the ``node_id`` attribute we
    # injected upstream.  Using this direct lookup is unambiguous even
    # when several pour-points fall inside one watershed polygon (which
    # happens when intersections sit on a shared flow path and the
    # upstream ones end up with no watershed cells of their own).
    snap_node_ids = snapped_pour["node_id"].to_numpy()
    ws_values = polys_gdf["watershed"].astype("int64").to_numpy()
    fid_indices = ws_values - 1
    # Clip any out-of-range values defensively; they should always be valid.
    valid = (fid_indices >= 0) & (fid_indices < len(snap_node_ids))
    polys_gdf = polys_gdf.loc[valid].reset_index(drop=True)
    polys_gdf["id"] = snap_node_ids[fid_indices[valid]].astype("int64")
    polys_gdf = polys_gdf.drop(columns=["watershed"])

    # Merge polygons that share a node ID (shouldn't happen now, but harmless cleanup)
    polys_gdf = polys_gdf.dissolve(by="id").reset_index()

    # Subdivide merged watersheds.  WBT's JensonSnapPourPoints snaps
    # several nearby graph nodes (median spacing ~40 m) onto one stream
    # cell, so WBT Watershed delineates a single polygon for them and
    # all but one node end up with no subcatchment.  Voronoi-split every
    # polygon that contains more than one pour point so each junction
    # gets its own contiguous subcatchment (area conserved).  Seed the
    # split with the SNAPPED pour points (where WBT actually placed each
    # outlet) rather than the original coordinates, so a polygon's own
    # outlet seed lies inside it and is not dropped by the partition.
    polys_gdf = _split_merged_watersheds(polys_gdf, snapped_pour)

    # Repair disconnected (MultiPolygon) subcatchments.  WBT's raster
    # Watershed occasionally assigns a detached sliver to a pour point
    # whose main body is elsewhere (D8 divide / raster-resolution
    # artifact).  A subcatchment must be a single contiguous polygon,
    # otherwise any pondshed built by unioning subs inherits the
    # disconnection.  Each detached part is reassigned to the adjacent
    # subcatchment it shares the longest boundary with (falls back to
    # nearest-centroid), so total area is preserved (mass balance) and
    # the sliver's runoff is attributed to the sub it is actually
    # embedded in.
    polys_gdf = _repair_disconnected_subcatchments(polys_gdf)

    # Dissolve fully-enclosed (nested) subcatchments into their container
    # so no subcatchment carries a hole filled by another.
    polys_gdf = _merge_nested_subcatchments(polys_gdf)

    # Calculate area, slope, and width
    polys_gdf["area"] = polys_gdf.geometry.area

    with rst.open(slope_path) as src:
        cell_slopes = src.read(1).astype(float)

    polys_gdf = calculate_slope(polys_gdf, grid, cell_slopes)
    polys_gdf["width"] = polys_gdf["area"].div(np.pi).pow(0.5)
    return polys_gdf


def derive_rc(
    subcatchments: gpd.GeoDataFrame,
    imperviousness_path: Path,
) -> gpd.GeoDataFrame:
    """Derive the Runoff Coefficient (RC) of each subcatchment.

    The runoff coefficient is computed as the mean fractional imperviousness
    (0-100) within each subcatchment polygon, using the NLCD imperviousness
    raster.

    Args:
        subcatchments (gpd.GeoDataFrame): A GeoDataFrame containing polygons that
            represent subcatchments with columns: 'geometry', 'area', and 'id'.
        imperviousness_path (Path): Path to a GeoTIFF of fractional
            imperviousness (0-100 scale).

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', 'id', 'impervious_area', and 'rc'.

    Author:
        @cheginit, @barneydobson
    """
    with rst.open(imperviousness_path) as src:
        imperv_data = src.read(1).astype(float)
        nodata = src.nodata
        transform = src.transform

    if nodata is not None:
        imperv_data[imperv_data == nodata] = np.nan

    # NaN marks "no impervious cell captured" so the median fallback below can
    # distinguish a genuine 0 % from an unsampled subcatchment.
    subcatchments["rc"] = np.nan
    for idx, geom in zip(subcatchments.index, subcatchments.geometry):
        mask = features.rasterize(
            [(geom, 1)],
            out_shape=imperv_data.shape,
            transform=transform,
            fill=0,
            # all_touched: a Voronoi sliver smaller than one NLCD cell still
            # samples the cell(s) it overlaps instead of capturing nothing.
            all_touched=True,
            dtype="uint8",
        )
        if mask is None:
            continue
        mask = mask.astype(bool)
        values = imperv_data[mask]
        valid = values[~np.isnan(values)]
        if valid.size > 0:
            subcatchments.loc[idx, "rc"] = float(np.mean(valid))

    # Any subcatchment that still captured no valid cell would otherwise get
    # rc=0 -> impervious_area=0, silently dropping its area from the mass
    # balance.  Fill it with the median rc of the sampled subcatchments
    # (mirrors calculate_slope's NaN->median fill) and warn.
    sampled = subcatchments["rc"].notna()
    n_missing = int((~sampled).sum())
    if n_missing:
        fallback = float(subcatchments.loc[sampled, "rc"].median()) if sampled.any() else 0.0
        logger.warning(
            f"derive_rc: {n_missing} subcatchment(s) captured no impervious cell; "
            f"filled rc with median {fallback:.1f}%."
        )
        subcatchments["rc"] = subcatchments["rc"].fillna(fallback)

    subcatchments["impervious_area"] = subcatchments["rc"] / 100 * subcatchments["area"]
    return subcatchments


def nodes_to_features(graph: nx.Graph[Any]):
    """Convert a graph to a GeoJSON node feature collection.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        dict: A GeoJSON feature collection.
    """
    features = []
    for node, data in graph.nodes(data=True):
        geom = sgeom.mapping(sgeom.Point(data["x"], data["y"]))
        props = {
            k: ",".join(map(str, v)) if pd.api.types.is_list_like(v) else v for k, v in data.items()
        }
        feature = {"type": "Feature", "geometry": geom, "properties": {"id": str(node), **props}}
        features.append(feature)
    return features


def edges_to_features(graph: nx.Graph[Any]):
    """Convert a graph to a GeoJSON edge feature collection.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        dict: A GeoJSON feature collection.
    """
    features = []
    for u, v, data in graph.edges(data=True):
        if "geometry" not in data:
            geom = None
        else:
            geom = sgeom.mapping(data["geometry"])
            del data["geometry"]
        props = {
            k: ",".join(map(str, val)) if pd.api.types.is_list_like(val) else val
            for k, val in data.items()
        }
        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": {"u": str(u), "v": str(v), **props},
        }
        features.append(feature)
    return features


def graph_to_geojson(graph: nx.Graph[Any], fid_nodes: Path, fid_edges: Path, crs: str) -> None:
    """Write a graph to a GeoJSON file.

    Args:
        graph (nx.Graph): The input graph.
        fid_nodes (Path): The filepath to save the nodes GeoJSON file.
        fid_edges (Path): The filepath to save the edges GeoJSON file.
        crs (str): The CRS of the graph.
    """
    graph = graph.copy()
    nodes = nodes_to_features(graph)
    edges = edges_to_features(graph)

    for iterable, fid in zip([nodes, edges], [fid_nodes, fid_edges]):
        geojson = {
            "type": "FeatureCollection",
            "features": iterable,
            "crs": {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:{crs.replace(':', '::')}"},
            },
        }
        fid.write_text(json.dumps(geojson, option=json.OPT_SERIALIZE_NUMPY).decode())


@overload
def simplify_geometry[GeomType: (LineString, MultiLineString, Polygon, MultiPolygon, BaseGeometry)](
    geometry: GeomType,
    tol_init: float,
    threshold_factor: float = 1.5,
    max_iter: int = 100,
) -> GeomType: ...


@overload
def simplify_geometry(
    geometry: GeomArray,
    tol_init: float,
    threshold_factor: float = 1.5,
    max_iter: int = 100,
) -> GeomArray: ...


def simplify_geometry[GeomType: (LineString, MultiLineString, Polygon, MultiPolygon, BaseGeometry)](
    geometry: GeomType | GeomArray,
    tol_init: float,
    threshold_factor: float = 1.5,
    max_iter: int = 100,
) -> GeomType | GeomArray:
    """Simplify a LineString using the Hausdorff distance as a measure of shape preservation.

    Parameters
    ----------
    geometry : any shapely geometry or an array of geometries
        The input geometry to be simplified.
    tol_init : float, optional
        The starting tolerance for simplification (default is 0.01).
    threshold_factor : float, optional
        Factor to determine when the shape has changed significantly (default is 1.5).
        A lower value will result in less simplification but better shape preservation.
    max_iter : int, optional
        Maximum number of iterations to avoid infinite loops (default is 100).

    Returns:
    -------
    same as input
        The simplified geometry or array of geometries.
    """
    tol_new = tol_init
    parts = shapely.get_parts(geometry)
    dist_init = shapely.hausdorff_distance(shapely.simplify(parts, tol_new), parts)
    dist_prev = dist_init
    tol_step = 0.5 * tol_init

    n_iter = 0
    while n_iter <= max_iter:
        n_iter += 1
        dist_new = shapely.hausdorff_distance(shapely.simplify(parts, tol_new), parts)
        if np.all(dist_new > threshold_factor * dist_init):
            # If Hausdorff distance has increased significantly, stop
            break
        if np.all(dist_new > dist_prev * threshold_factor):
            # If there's a sudden jump in Hausdorff distance, stop
            break
        if np.allclose(dist_new, dist_prev):
            # If there's no change in Hausdorff distance, stop
            break

        dist_prev = dist_new
        tol_new += tol_step

    # Return the second-to-last result (last result before significant change)
    return shapely.simplify(geometry, tol_new - tol_step)
