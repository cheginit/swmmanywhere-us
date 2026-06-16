"""Module for integrating water bodies as SWMM storage units with outlet structures.

Implements the FDOT/Opti CMAC pond modeling methodology:
- Geometric characterization with side-slope stage-area curves
- Primary orifice sizing from Opti CMAC volume table
- Emergency weir (spillway) sizing
- Correct SWMM topology: subcatchment -> storage -> orifice/weir -> outlet junction -> network

Integration happens in two phases so that subcatchment delineation can route
runoff directly to ponds (matching the calibrated reference-model pattern):

1. :func:`insert_pond_nodes` runs BEFORE subcatchment delineation.  It adds
   pond STORAGE nodes at the pond centroid with a single undirected
   ``pond_connector`` edge to the nearest pipe-network node, so the
   watershed-based subcatchment routine can treat the pond as a pour point.

2. :func:`finalize_pond_outlets` runs AFTER topology derivation and
   pipe-by-pipe sizing.  By that point the ``pond_connector`` has been
   oriented downstream by Dijkstra, so we can safely replace it with the
   canonical SWMM outlet structure: ``storage --orifice/weir--> outlet_junction
   --conduit--> downstream_network_node``.
"""

from __future__ import annotations

import collections
import itertools
import math
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    from swmmanywhere_us.filepaths import FilePaths
    from swmmanywhere_us.parameters import HydraulicDesign, PondDesign

# Unit conversions
_M2_TO_SQFT = 10.7639
_M3_TO_CUFT = 35.3147
_M_TO_FT = 3.28084
_FT_TO_M = 0.3048
_IN_TO_M = 0.0254


def _fdot_volume_m3(area_m2: float, intercept: float, slope: float) -> float:
    """Estimate pond volume using FDOT empirical V-A relationship.

    V_cuft = intercept + slope * A_sqft   (FDOT Drainage Design Guide)

    Returns volume in m^3.
    """
    area_sqft = area_m2 * _M2_TO_SQFT
    vol_cuft = intercept + slope * area_sqft
    return max(vol_cuft / _M3_TO_CUFT, 0.0)


def _compute_bottom_area(
    surface_area_m2: float,
    max_depth_m: float,
    side_slope_pct: float,
    min_ratio: float,
) -> float:
    """Compute bottom area assuming square planform with uniform side slopes."""
    hv_ratio = 1.0 / (side_slope_pct / 100.0)
    l_surface = math.sqrt(surface_area_m2)
    delta_l = 2.0 * hv_ratio * max_depth_m
    l_bottom = max(l_surface - delta_l, min_ratio * l_surface)
    return l_bottom**2


def _stage_area_curve(
    surface_area_m2: float,
    max_depth_m: float,
    side_slope_pct: float,
    min_ratio: float,
    n_points: int,
) -> list[tuple[float, float]]:
    """Generate a multi-point stage-area curve with side-slope geometry."""
    hv_ratio = 1.0 / (side_slope_pct / 100.0)
    l_surface = math.sqrt(surface_area_m2)
    min_l = min_ratio * l_surface

    curve = []
    for i in range(n_points):
        depth = i * max_depth_m / (n_points - 1)
        delta_l = 2.0 * hv_ratio * (max_depth_m - depth)
        l_at_depth = max(l_surface - delta_l, min_l)
        area_at_depth = l_at_depth**2
        curve.append((depth, area_at_depth))
    return curve


def _curve_volume(curve: list[tuple[float, float]]) -> float:
    """Trapezoidal integration of a stage-area curve."""
    if len(curve) < 2:
        return 0.0
    return sum(
        (curve[i][1] + curve[i + 1][1]) / 2 * (curve[i + 1][0] - curve[i][0])
        for i in range(len(curve) - 1)
    )


def _solve_max_depth(
    surface_area_m2: float,
    target_volume_m3: float,
    side_slope_pct: float,
    min_ratio: float,
    n_points: int,
    max_depth_cap: float,
) -> float:
    """Bisection search for the max_depth matching target volume.

    Returns the mid depth that produced the best trapezoidal-integrated
    volume match.  Returning ``(lo + hi) / 2`` on early termination would
    collapse to a value halfway between the bracketed limits, which is
    NOT the mid that satisfied the tolerance and can miss the target by
    >20%.
    """
    mean_depth = target_volume_m3 / max(surface_area_m2, 1.0)
    if mean_depth >= max_depth_cap:
        return max_depth_cap

    lo, hi = max(mean_depth * 0.5, 0.1), min(mean_depth * 3.0, max_depth_cap)
    best_mid = (lo + hi) / 2
    best_err = float("inf")
    for _ in range(30):
        mid = (lo + hi) / 2
        curve = _stage_area_curve(surface_area_m2, mid, side_slope_pct, min_ratio, n_points)
        vol = _curve_volume(curve)
        err = abs(vol - target_volume_m3) / max(target_volume_m3, 1.0)
        if err < best_err:
            best_err = err
            best_mid = mid
        if vol < target_volume_m3:
            lo = mid
        else:
            hi = mid
        if err < 0.05:
            break
    return best_mid


def _validate_curve_volume(curve: list[tuple[float, float]], target_volume_m3: float) -> float:
    """Percent error between curve-integrated volume and target."""
    if len(curve) < 2 or target_volume_m3 <= 0:
        return 0.0
    volume = sum(
        (curve[i][1] + curve[i + 1][1]) / 2 * (curve[i + 1][0] - curve[i][0])
        for i in range(len(curve) - 1)
    )
    return abs(volume - target_volume_m3) / target_volume_m3 * 100


def _volume_below(curve: list[tuple[float, float]], depth_limit: float) -> float:
    """Trapezoidal stage-storage up to ``depth_limit`` (linear area interpolation)."""
    if len(curve) < 2 or depth_limit <= curve[0][0]:
        return 0.0
    vol = 0.0
    for (d0, a0), (d1, a1) in itertools.pairwise(curve):
        if d1 <= depth_limit:
            vol += (a0 + a1) / 2.0 * (d1 - d0)
        else:  # partial segment up to depth_limit
            if d0 < depth_limit and d1 > d0:
                frac = (depth_limit - d0) / (d1 - d0)
                a_mid = a0 + frac * (a1 - a0)
                vol += (a0 + a_mid) / 2.0 * (depth_limit - d0)
            break
    return vol


def _pond_qa_warnings(
    idx: Any,
    curve: list[tuple[float, float]],
    max_depth: float,
    bottom_area: float,
    area_m2: float,
    orifice_diam: float,
    weir_crest: float,
    pond_design: PondDesign,
) -> None:
    """Log spec S11 QA diagnostics (bottom-area band, weir activation volume, drawdown time).

    Pure post-hoc checks over the already-sized geometry, they emit warnings so
    a user can spot ponds outside the design envelope, and never change the model.
    """
    # S11.1, bottom-area ratio bands (>=0.30 pass, 0.20-0.30 warn, <0.20 fail).
    if area_m2 > 0:
        ratio = bottom_area / area_m2
        if ratio < 0.20:
            logger.warning(f"pond {idx}: bottom-area ratio {ratio:.2f} < 0.20 (S11.1 fail).")
        elif ratio < 0.30:
            logger.warning(f"pond {idx}: bottom-area ratio {ratio:.2f} in 0.20-0.30 (S11.1 warn).")

    # S11.3, weir activation volume: V(weir_crest)/V_total should be >= 0.85,
    # so the spillway activates only near-full (extreme events).
    v_total = _curve_volume(curve)
    if v_total > 0:
        activation = _volume_below(curve, weir_crest) / v_total
        if activation < 0.85:
            logger.warning(
                f"pond {idx}: weir activation volume ratio {activation:.2f} < 0.85 (S11.3), "
                f"spillway may engage during design (not just extreme) storms."
            )

    # S11.2, orifice drawdown time t = V / (Cd*A*sqrt(2*g*0.5*Dmax)) should be 24-72 h.
    orifice_area = math.pi * orifice_diam**2 / 4.0
    head = 0.5 * max_depth
    if orifice_area > 0 and head > 0:
        q_avg = pond_design.orifice_cd * orifice_area * math.sqrt(2.0 * 9.81 * head)
        if q_avg > 0:
            t_hr = v_total / (q_avg * 3600.0)
            if t_hr < 24.0 or t_hr > 72.0:
                logger.warning(
                    f"pond {idx}: orifice drawdown time {t_hr:.0f} h outside 24-72 h (S11.2)."
                )


def _orifice_diameter_m(volume_m3: float) -> float:
    """Select orifice diameter from Opti CMAC table based on controllable volume."""
    vol_cuft = volume_m3 * _M3_TO_CUFT
    if vol_cuft < 50_000:
        return 8 * _IN_TO_M
    if vol_cuft < 200_000:
        return 12 * _IN_TO_M
    if vol_cuft < 400_000:
        return 18 * _IN_TO_M
    return 24 * _IN_TO_M


def _weir_length_m(surface_area_m2: float, default_length_m: float) -> float:
    """Scale weir length by pond dimension (10% of side length, bounded 3-10 m)."""
    l_surface = math.sqrt(surface_area_m2)
    return min(max(3.0, 0.10 * l_surface), 10.0) if l_surface > 0 else default_length_m


def _apply_pond_geometry(
    graph: nx.MultiDiGraph[Any],
    storage_id: Any,
    pond_design: PondDesign,
    area_m2: float,
    max_depth: float,
    invert_elev: float,
) -> None:
    """Persist a pond's depth-dependent geometry to its STORAGE node and outlets.

    Recomputes the FDOT volume, stage-area curve, orifice diameter, weir length
    and weir crest for ``max_depth`` and writes them (with the invert) onto the
    node, then mirrors the orifice/weir values onto the pond's already-created
    outlet EDGES so the stored curve top AND the emitted outlet structure follow
    the (possibly deepened) MaxDepth.  The outlet edges are the channel the INP
    writer actually reads: without the edge update a deepened pond would keep a
    weir crest sized for its pre-deepen depth.  Call this whenever
    ``chamber_floor_elevation`` / ``wb_max_depth`` change after the initial
    sizing (it is used by ``route_pipes_into_ponds``, after the outlet edges
    exist).
    """
    volume_m3 = _fdot_volume_m3(
        area_m2, pond_design.fdot_volume_intercept, pond_design.fdot_volume_slope
    )
    node = graph.nodes[storage_id]
    node["wb_max_depth"] = max_depth
    node["chamber_floor_elevation"] = invert_elev
    node["wb_stage_area_curve"] = _stage_area_curve(
        area_m2,
        max_depth,
        pond_design.side_slope_pct,
        pond_design.bottom_area_min_ratio,
        pond_design.n_curve_points,
    )
    node["wb_orifice_diam_m"] = _orifice_diameter_m(volume_m3)
    node["wb_weir_length_m"] = _weir_length_m(area_m2, pond_design.weir_length_m)
    node["wb_weir_crest_m"] = pond_design.weir_crest_ratio * max_depth

    # Mirror onto the outlet structure edges the INP writer reads (created by
    # finalize_pond_outlets before this runs); otherwise these node writes are
    # inert and the deepened pond keeps its pre-deepen weir crest.
    for _u, _v, _k, ed in graph.out_edges(storage_id, keys=True, data=True):
        if ed.get("edge_type") == "orifice":
            ed["orifice_diam_m"] = node["wb_orifice_diam_m"]
        elif ed.get("edge_type") == "weir":
            ed["weir_crest_m"] = node["wb_weir_crest_m"]
            ed["weir_length_m"] = node["wb_weir_length_m"]


def _classify_water_body(
    row: pd.Series[Any],
    pond_design: PondDesign,
) -> str:
    """Classify a water-body polygon as ``pond`` or ``lake``.

    Classification rule (see PondDesign references):

    1. Explicit lake-family OSM tag (``water`` in ``lake_osm_tags``):
       always ``lake``, respects explicit natural-feature tagging.
    2. Otherwise, apply the limnological area threshold (Richardson
       et al. 2022): area <= ``max_pond_area_m2`` -> ``pond``,
       otherwise ``lake``.

    OSM ``water=pond`` and ``basin=detention|retention|...`` tags are
    **informative** but do **not** override the size rule.  The FDOT
    V-A regression (V=0.643+2.59*A) and Opti-CMAC orifice table were
    fitted on Florida subdivision detention ponds <= 2 ha, 3-6 ft deep
    (Harper & Baker 2007); applying them to 10+ ha basins is
    extrapolation outside the validated envelope, regardless of how
    the feature is tagged in OSM.  Large designed basins may be
    legitimate engineered facilities, but their stage-storage-outlet
    behavior needs site-specific data, not a subdivision-pond
    regression.  For those cases users can raise ``max_pond_area_m2``
    in PondDesign explicitly.

    NLCD class-11 polygons carry no OSM tags and fall into rule 2.
    """
    max_pond_area = float(pond_design.max_pond_area_m2)
    lake_tags = {t.lower() for t in pond_design.lake_osm_tags}

    water_tag = str(row.get("water") or "").strip().lower()

    # Rule 1: explicit natural-water tag.
    if water_tag in lake_tags:
        return "lake"

    # Rule 2: area-only (limnological threshold + FDOT calibration envelope).
    return "pond" if float(row["area_m2"]) <= max_pond_area else "lake"


def _read_wb_layer(path: Path, target_crs: Any, source: str) -> gpd.GeoDataFrame | None:
    """Read one water-body parquet layer in *target_crs*; None when absent or empty."""
    if not path.exists():
        return None
    wb = gpd.read_parquet(path)
    if wb.empty:
        return None
    if target_crs and wb.crs and str(wb.crs) != str(target_crs):
        wb = wb.to_crs(target_crs)
    wb["wb_source"] = source
    return wb


def _load_water_body_polygons(
    addresses: FilePaths,
    pond_design: PondDesign,
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """Load, classify, and deduplicate water-body polygons.

    Returns a GeoDataFrame with an added ``wb_class`` column (``pond`` or
    ``lake``).  Downstream, ponds are modeled as SWMM STORAGE with FDOT-sized
    orifice + weir outlets; lakes are modeled as fixed-stage OUTFALL
    boundaries (receiving waters) without active outlet structures.

    Classification blends NLCD (area-only) and OSM (tag + area), see
    :func:`_classify_water_body` for the rule.
    """
    layers = (
        _read_wb_layer(addresses.bbox_paths.water_bodies, target_crs, "nlcd"),
        _read_wb_layer(addresses.bbox_paths.basins, target_crs, "osm_basin"),
    )
    wb_gdfs: list[gpd.GeoDataFrame] = [wb for wb in layers if wb is not None]

    if not wb_gdfs:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=target_crs))

    all_wb = gpd.GeoDataFrame(pd.concat(wb_gdfs, ignore_index=True))
    all_wb = all_wb[all_wb.geometry.notna() & ~all_wb.geometry.is_empty].copy()
    all_wb["area_m2"] = all_wb.geometry.area
    all_wb = all_wb[all_wb["area_m2"] >= pond_design.min_area_m2].reset_index(drop=True)
    if all_wb.empty:
        return all_wb

    # Deduplicate overlapping polygons (keep larger).
    all_wb = all_wb.sort_values("area_m2", ascending=False).reset_index(drop=True)
    keep = np.ones(len(all_wb), dtype=bool)
    tree = all_wb.sindex
    for i, geom in enumerate(all_wb.geometry):
        if not keep[i]:
            continue
        candidates = tree.query(geom, predicate="intersects")
        for j in candidates:
            if j <= i or not keep[j]:
                continue
            if geom.intersection(all_wb.geometry.iloc[j]).area > 0.5 * all_wb.iloc[j]["area_m2"]:
                keep[j] = False
    all_wb = all_wb[keep].reset_index(drop=True)

    # Classification.
    all_wb["wb_class"] = all_wb.apply(lambda r: _classify_water_body(r, pond_design), axis=1)
    n_ponds = int((all_wb["wb_class"] == "pond").sum())
    n_lakes = int((all_wb["wb_class"] == "lake").sum())
    logger.info(
        f"Water-body classification: {n_ponds} pond(s), {n_lakes} lake(s); "
        f"lake threshold = {pond_design.max_pond_area_m2 / 1e4:.1f} ha "
        "(Richardson et al. 2022)."
    )
    return all_wb


def _network_nodes_gdf(graph: nx.Graph[Any]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of pipe/river nodes (excluding already-inserted ponds)."""
    crs = graph.graph.get("crs")
    rows = []
    for n, d in graph.nodes(data=True):
        if "x" not in d or "y" not in d:
            continue
        if d.get("node_type") == "water_body":
            # Skip ponds we've already inserted, otherwise a second pond
            # could snap onto the first.
            continue
        is_river = any(
            graph.edges[u, v, k].get("edge_type") == "river"  # pyright: ignore[reportArgumentType]
            for u, v, k in graph.edges(n, keys=True)  # pyright: ignore[reportCallIssue]
        )
        rows.append({"node": n, "x": d["x"], "y": d["y"], "is_river": is_river})
    if not rows:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs))
    return gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]),
        crs=crs,
    )


def _overtopping_crest_elev(
    geom: Any,
    src: Any,
    ring_inner_m: float = 5.0,
    ring_outer_m: float = 20.0,
) -> float:
    """Estimate a pond's overtopping-crest elevation from the DEM.

    Samples the (non-hydroflattened) terrain on a ring just outside the water
    polygon, the surrounding embankment / grade that pond water must rise
    above to spill onto the street.  Returns the 25th-percentile ring
    elevation (closer to the low point of the rim, where overtopping actually
    begins, rather than the mean grade), or NaN if no valid pixels.  Used to
    extend the SWMM ``MaxDepth`` from the FDOT treatment depth up to the
    embankment so the pond carries real flood-detention storage.
    """
    ring = geom.buffer(ring_outer_m).difference(geom.buffer(ring_inner_m))
    if ring.is_empty:
        return float("nan")
    minx, miny, maxx, maxy = ring.bounds
    xs = np.linspace(minx, maxx, 12)
    ys = np.linspace(miny, maxy, 12)
    pts = [(float(x), float(y)) for x in xs for y in ys if ring.contains(shapely.Point(x, y))]
    if not pts:
        rp = ring.representative_point()
        pts = [(rp.x, rp.y)]
    nodata = src.nodata
    vals = [
        float(v[0])
        for v in src.sample(pts)
        if v[0] is not None
        and not np.isnan(v[0])
        and (nodata is None or v[0] != nodata)
        and abs(v[0]) > 1e-6
    ]
    if not vals:
        return float("nan")
    return float(np.percentile(vals, 25))


def insert_pond_nodes(  # noqa: C901, PLR0912, PLR0915 - subcatchment reroute + fallback snap kept in one routine
    graph: nx.MultiGraph[Any] | nx.MultiDiGraph[Any],
    addresses: FilePaths,
    pond_design: PondDesign,
    **kwargs: Any,
) -> nx.MultiGraph[Any] | nx.MultiDiGraph[Any]:
    """Insert pond STORAGE nodes and reroute containing subcatchments.

    Runs AFTER :func:`calculate_contributing_area` so that the naturally-
    delineated subcatchments (pour points = street intersections only) are
    available.  For each pond polygon, sorted from largest to smallest
    area:

    1. Find the subcatchment whose polygon contains the pond centroid.
       If one exists and hasn't already been claimed by a larger pond in
       the same subcatchment, that subcatchment's ``Outlet`` is rewritten
       to the new pond STORAGE id, so the subcatchment's surface runoff
       drains directly into the pond, matching the calibrated reference-model pattern
       (``SUB.Outlet = STORAGE``).  The pond's ``pond_connector`` edge is
       anchored to the subcatchment's ORIGINAL pour point, so the pond's
       outflow rejoins the pipe network exactly where the subcatchment
       used to discharge.
    2. Ponds that don't fall inside any subcatchment (or whose subcatchment
       is already claimed) fall back to the ``nearest network node``
       behavior, the pond is still integrated, just without rerouting a
       subcatchment to it.

    The updated subcatchments parquet is written back so downstream steps
    (``post_processing.synthetic_write``) pick up the new Outlet assignment.

    Args:
        graph: The pipeline graph (undirected at this stage).
        addresses: File path manager.
        pond_design: Pond geometry and outlet sizing parameters.
        **kwargs: Ignored.

    Returns:
        The graph with pond storage nodes and pond_connector edges added.
    """
    crs = graph.graph.get("crs")
    all_wb = _load_water_body_polygons(addresses, pond_design, crs)
    if all_wb.empty:
        logger.info("No water body polygons found; skipping pond insertion.")
        return graph

    # Only designed-pond class bodies get FDOT V-A sizing + Opti-CMAC
    # orifice/weir outlets.  Larger lakes / reservoirs stay in the
    # hydroflattened DEM so surface runoff still routes toward them, but
    # they are not modeled as detention ponds, that would apply the
    # Florida subdivision calibration envelope (Harper & Baker 2007)
    # miles outside its validated range.
    n_total = len(all_wb)
    all_wb = all_wb[all_wb["wb_class"] == "pond"].reset_index(drop=True)
    n_skipped = n_total - len(all_wb)
    if n_skipped:
        logger.info(
            f"Skipping {n_skipped} lake/reservoir polygon(s); not sized as "
            "detention ponds (see PondDesign.max_pond_area_m2)."
        )
    if all_wb.empty:
        logger.info("No pond-class water bodies after classification; skipping insertion.")
        return graph

    nodes_gdf = _network_nodes_gdf(graph)
    if nodes_gdf.empty:
        logger.warning("No network nodes with coordinates; skipping pond insertion.")
        return graph

    # Subcatchments parquet is optional, if it doesn't exist (e.g. bbox
    # had no usable DEM), fall back to nearest-node snapping for every pond.
    subs_path = addresses.model_paths.subcatchments
    subs_gdf: gpd.GeoDataFrame | None = None
    if subs_path.exists():
        subs_gdf = gpd.read_parquet(subs_path)
        if crs and subs_gdf.crs and str(subs_gdf.crs) != str(crs):
            subs_gdf = subs_gdf.to_crs(crs)

    # Pond node IDs continue past the max existing int node ID so they
    # cannot collide with pipe/river nodes.
    existing_int_nodes = [n for n in graph.nodes if isinstance(n, int)]
    next_node_id = (max(existing_int_nodes) + 1) if existing_int_nodes else 0

    # River centerlines for the canal-anchor override below: when the canal
    # is closer to a pond than the anchor chosen by the preferences, the
    # pond discharges to a sink snapped onto the canal instead of crossing
    # the street grid to a distant pour point.
    river_geoms = [
        d["geometry"]
        for _, _, d in graph.edges(data=True)
        if d.get("edge_type") == "river" and d.get("geometry") is not None
    ]
    river_tree = shapely.STRtree(river_geoms) if river_geoms else None

    # Track which subcatchment ids have already been claimed by a pond so
    # a second, smaller pond in the same subcatchment doesn't also steal
    # its outlet.  Keyed by the subcatchment's current ``id`` column value.
    claimed_sub_ids: set[Any] = set()
    # Subcatchments gdf gets mutated in-place (we rewrite ``id`` for each
    # rerouted sub); we keep a running dict of those changes so we can
    # write once at the end instead of once per pond.
    id_updates: dict[Any, Any] = {}

    # Process ponds largest-first so the biggest pond in a subcatchment
    # wins the Outlet slot.
    all_wb = all_wb.sort_values("area_m2", ascending=False).reset_index(drop=True)

    # Overtopping-crest elevation per pond (DEM rim just outside the polygon).
    # post_processing extends the pond's SWMM MaxDepth + stage-area curve from
    # the FDOT treatment depth up to this crest, giving real flood-detention
    # storage above the permanent pool.
    embankment_elev: list[float] = [float("nan")] * len(all_wb)
    elev_path = addresses.bbox_paths.elevation
    if elev_path.exists():
        import rasterio

        with rasterio.open(elev_path) as dem_src:
            embankment_elev = [_overtopping_crest_elev(g, dem_src) for g in all_wb.geometry]

    inserted = 0
    rerouted = 0
    canal_anchored = 0
    for idx, (centroid, wb_row) in enumerate(
        zip(all_wb.geometry.centroid, all_wb.itertuples(index=False))
    ):
        area_m2 = float(wb_row.area_m2)  # pyright: ignore[reportArgumentType]
        cx, cy = centroid.x, centroid.y  # pyright: ignore[reportAttributeAccessIssue]

        # --- Decide the pond's downstream-network anchor -----------------
        # Preference 1: the pond sits inside a subcatchment that no larger
        # pond has already claimed, anchor to that sub's original pour
        # point and reroute the sub to the pond.
        anchor_node: Any | None = None
        sub_row = None
        if subs_gdf is not None and not subs_gdf.empty:
            containing = subs_gdf[subs_gdf.geometry.contains(centroid)]
            for _, candidate in containing.iterrows():
                cand_id = candidate["id"]
                if cand_id in claimed_sub_ids:
                    continue
                if cand_id not in graph.nodes:
                    continue
                sub_row = candidate
                anchor_node = cand_id
                break

        # Preference 2: nearest network node, preferring rivers.
        if anchor_node is None:
            distances = nodes_gdf.geometry.distance(centroid)
            river_mask = nodes_gdf["is_river"]
            nearest_river_idx = distances[river_mask].idxmin() if river_mask.any() else None
            nearest_any_idx = distances.idxmin()
            if (
                nearest_river_idx is not None
                and distances[nearest_river_idx] <= pond_design.max_snap_distance_m
            ):
                anchor_node = nodes_gdf.loc[nearest_river_idx, "node"]
            elif distances[nearest_any_idx] <= pond_design.max_snap_distance_m:
                anchor_node = nodes_gdf.loc[nearest_any_idx, "node"]
            else:
                continue  # pond is too far from any network to be wired in

        # --- Canal-anchor override ----------------------------------------
        # Real subdivision ponds discharge to the adjacent canal, not across
        # the street grid: when the nearest river centerline is closer to
        # the pond polygon than the anchor chosen above (and within the snap
        # radius), replace the anchor with a fresh sink snapped onto the
        # canal at the nearest point.  Snapping onto the centerline (rather
        # than wiring to a river graph node) matters because river features
        # are noded only at their endpoints, often hundreds of meters away.
        # The sink reuses ``node_type="river_outfall"`` so identify_outfalls
        # samples the receiving water's DEM stage for it and post_processing
        # routes it to [OUTFALLS]; derive_topology preserves it alongside
        # the pond storage.  The subcatchment claim below (Outlet rewrite +
        # contributing_area transfer) is unaffected, only WHERE the pond
        # discharges changes.
        intercept_node = None
        if river_tree is not None:
            poly: Any = wb_row.geometry  # GeoDataFrame row .geometry is a shapely geom at runtime
            near_idx = int(river_tree.nearest(poly))
            d_canal = shapely.distance(poly, river_geoms[near_idx])
            anchor_pt = shapely.Point(graph.nodes[anchor_node]["x"], graph.nodes[anchor_node]["y"])
            if (
                d_canal < shapely.distance(poly, anchor_pt)
                and d_canal <= pond_design.max_snap_distance_m
            ):
                snap_line = shapely.shortest_line(centroid, river_geoms[near_idx])
                snap_pt = shapely.Point(snap_line.coords[-1])
                sink_id = next_node_id
                next_node_id += 1
                graph.add_node(sink_id, x=snap_pt.x, y=snap_pt.y, node_type="river_outfall")
                # The displaced anchor stays the pond's street-flow
                # INTERCEPTION point: route_pipes_into_ponds (via
                # partition_pond_network) reroutes the pipes arriving
                # there through the pond, so the subdivision's drainage
                # still passes through detention before the pond
                # releases to the canal.
                intercept_node = anchor_node
                anchor_node = sink_id
                canal_anchored += 1

        # --- Pond geometry / sizing --------------------------------------
        volume_m3 = _fdot_volume_m3(
            area_m2, pond_design.fdot_volume_intercept, pond_design.fdot_volume_slope
        )
        max_depth = _solve_max_depth(
            area_m2,
            volume_m3,
            pond_design.side_slope_pct,
            pond_design.bottom_area_min_ratio,
            pond_design.n_curve_points,
            pond_design.max_depth_m,
        )
        curve = _stage_area_curve(
            area_m2,
            max_depth,
            pond_design.side_slope_pct,
            pond_design.bottom_area_min_ratio,
            pond_design.n_curve_points,
        )
        vol_error = _validate_curve_volume(curve, volume_m3)
        if vol_error > 10:
            logger.warning(
                f"water body {idx}: stage-area curve volume error {vol_error:.0f}% "
                f"(target {volume_m3:.0f} m3, depth {max_depth:.2f} m)"
            )
        bottom_area = _compute_bottom_area(
            area_m2, max_depth, pond_design.side_slope_pct, pond_design.bottom_area_min_ratio
        )
        orifice_diam = _orifice_diameter_m(volume_m3)
        weir_length = _weir_length_m(area_m2, pond_design.weir_length_m)
        weir_crest = pond_design.weir_crest_ratio * max_depth

        # Spec S11 QA diagnostics (warnings only; do not change the model).
        _pond_qa_warnings(
            idx, curve, max_depth, bottom_area, area_m2, orifice_diam, weir_crest, pond_design
        )

        # --- Add pond STORAGE node ---------------------------------------
        storage_id = next_node_id
        next_node_id += 1
        graph.add_node(
            storage_id,
            x=cx,
            y=cy,
            node_type="water_body",
            wb_index=idx,
            wb_area_m2=area_m2,
            wb_max_depth=max_depth,
            wb_volume_m3=volume_m3,
            wb_bottom_area_m2=bottom_area,
            wb_stage_area_curve=curve,
            wb_orifice_diam_m=orifice_diam,
            wb_weir_length_m=weir_length,
            wb_weir_crest_m=weir_crest,
            wb_snapped_to=anchor_node,
            wb_embankment_elev=embankment_elev[idx],
        )
        if intercept_node is not None:
            graph.nodes[storage_id]["wb_intercept_node"] = intercept_node

        # Reroute the claimed subcatchment's Outlet to the new pond.  Both
        # the claimed set and the pending id_updates dict are keyed by the
        # subcatchment's original id so we can write them back in one pass.
        if sub_row is not None:
            orig_sub_id = sub_row["id"]
            claimed_sub_ids.add(orig_sub_id)
            id_updates[orig_sub_id] = storage_id
            rerouted += 1
            # Transfer the subcatchment's effective contributing_area to the
            # pond storage node so downstream pipe-sizing (pipe_by_pipe and
            # resize_street_pipes_for_pond_routing) sees the pond's inflow
            # rather than the original (now-orphaned) pour point.  Graph
            # node attributes are the ground-truth input for rational-method
            # flow accumulation; without this transfer pond storages have
            # ``contributing_area = 0`` and their downstream pipes are
            # designed for zero flow.
            orig_node_data = graph.nodes.get(orig_sub_id, {})
            transferred_area = float(orig_node_data.get("contributing_area", 0.0) or 0.0)
            if transferred_area > 0:
                current_pond_ca = float(
                    graph.nodes[storage_id].get("contributing_area", 0.0) or 0.0
                )
                graph.nodes[storage_id]["contributing_area"] = current_pond_ca + transferred_area
                graph.nodes[orig_sub_id]["contributing_area"] = 0.0

        # --- Add the pond_connector edge (pond -> anchor) ---------------
        # ``contributing_area`` is zeroed so the edge is a no-op in
        # topology weight calculation, subcatchment runoff is routed
        # directly to the pond STORAGE node via SWMM Outlet, so the
        # connector is not a pipe-flow path that needs sizing.
        ax = graph.nodes[anchor_node]["x"]
        ay = graph.nodes[anchor_node]["y"]
        connector_geom = shapely.LineString([(cx, cy), (ax, ay)])
        graph.add_edge(
            storage_id,
            anchor_node,
            geometry=connector_geom,
            length=max(shapely.length(connector_geom), 1.0),
            edge_type="pond_connector",
            id=f"{storage_id}-{anchor_node}-pond_connector",
            roughness=0.035,
            channel_width=max(area_m2**0.5 * 0.1, 1.0),
            channel_depth=max_depth,
            diameter=0.0,
            contributing_area=0.0,
        )
        inserted += 1

    # Persist the subcatchment Outlet rewrites, if any.  Cast to int64 after
    # mapping because the source column is sometimes float64 (pandas widens
    # to float when any upstream operation produced a NaN), a stringified
    # "4562.0" wouldn't match the integer node IDs in the graph.
    if id_updates and subs_gdf is not None:
        subs_gdf = subs_gdf.copy()
        subs_gdf["id"] = subs_gdf["id"].map(lambda v: id_updates.get(v, v)).astype("int64")
        subs_gdf.to_parquet(subs_path)

    logger.info(
        f"Inserted {inserted} pond storage node(s) with connector edges "
        f"(from {len(all_wb)} candidates); {rerouted} subcatchment(s) rerouted "
        f"to drain directly into a pond; {canal_anchored} pond(s) anchored to "
        "a sink snapped onto the nearest canal."
    )
    return graph


def _pond_has_viable_outfall(  # noqa: C901 - single BFS with in-loop terminal head test
    graph: nx.MultiDiGraph[Any],
    pond_storage_id: Any,
    min_head_drop_m: float,
) -> bool:
    """True iff some SWMM outfall is reachable from the pond with adequate head.

    BFS forward from the pond storage through orifice/weir → outlet_junction
    → pond_outflow → downstream pipes/rivers/outfall edges.  A pond is
    "gravity-viable" iff at least one reachable terminal sink (node with
    out-degree 0 in the directed flow graph, SWMM outfall) clears the
    head requirement below the pond's max water surface elevation:

    - Terminals carrying a real receiving-water stage
      (``node_type`` ``river_outfall`` / ``water_body_outfall``, whose
      invert is the hydroflattened DEM stage of the canal) need only a
      small driving head, 0.1 m plus the minimum-slope drop accumulated
      over the path.  A pond whose max WSE sits above the adjacent
      canal's water surface drains by gravity; demanding the flat
      ``min_head_drop_m`` (sized for derived-invert terminals) against a
      near-ground canal stage marked nearly every canal-adjacent pond on
      flat terrain as a closed basin.
    - All other terminals (dummy rivers and pipe-derived inverts) keep
      the full ``min_head_drop_m`` criterion.

    Ponds that reach only perched terminals (head drop too small to
    drive sustained flow through the path's flat conduits) return
    ``False`` and should be modeled as closed-basin retention with
    Green-Ampt exfiltration.
    """
    if pond_storage_id not in graph.nodes:
        return False
    pdata = graph.nodes[pond_storage_id]
    pond_max_wse = pdata.get("surface_elevation")
    if pond_max_wse is None:
        invert = pdata.get("chamber_floor_elevation")
        depth = pdata.get("wb_max_depth", 0)
        if invert is None:
            return True  # can't decide; default to "viable" (preserve current behavior)
        pond_max_wse = float(invert) + float(depth)
    pond_max_wse = float(pond_max_wse)

    min_slope = 1e-3  # SWMM minimum positive slope (matches finalize_pond_outlets)
    stage_margin_m = 0.1  # driving head over the receiving water's stage

    visited: set[Any] = {pond_storage_id}
    # (node, cumulative path length in m), BFS first-visit length is a
    # good-enough proxy for the min-slope head allowance.
    queue: list[tuple[Any, float]] = [(pond_storage_id, 0.0)]
    while queue:
        node, cum_m = queue.pop(0)
        out_edges = list(graph.out_edges(node, keys=True))
        if not out_edges and node != pond_storage_id:
            # Terminal SWMM outfall, check head drop.
            v_invert = graph.nodes[node].get("chamber_floor_elevation")
            if v_invert is None:
                continue
            if graph.nodes[node].get("node_type") in (
                "river_outfall",
                "water_body_outfall",
            ):
                required = stage_margin_m + min_slope * cum_m
            else:
                required = min_head_drop_m
            if pond_max_wse - float(v_invert) >= required:
                return True
            continue  # this terminal is too perched; keep searching others
        for _, v, k in out_edges:
            if v in visited:
                continue
            visited.add(v)
            edge_len = float(graph.edges[node, v, k].get("length", 0) or 0)
            queue.append((v, cum_m + edge_len))
    return False


def _drains_to_outfall_closure(graph: nx.MultiDiGraph[Any]) -> set[Any]:
    """Nodes whose flow eventually reaches a SWMM outfall.

    Sinks are the targets of ``outfall`` edges plus the synthetic
    ``river_outfall`` / ``water_body_outfall`` sinks that post_processing
    routes to [OUTFALLS] by node_type alone; the closure is their reverse
    (upstream) reachability, computed once per finalize pass.
    """
    sinks: set[Any] = {v for _, v, d in graph.edges(data=True) if d.get("edge_type") == "outfall"}
    sinks |= {
        n
        for n, d in graph.nodes(data=True)
        if d.get("node_type") in ("river_outfall", "water_body_outfall")
    }
    closure: set[Any] = set(sinks)
    queue: collections.deque[Any] = collections.deque(sinks)
    while queue:
        node = queue.popleft()
        for pred in graph.predecessors(node):
            if pred not in closure:
                closure.add(pred)
                queue.append(pred)
    return closure


def _nearest_drainable_anchor(
    graph: nx.MultiDiGraph[Any],
    storage_id: Any,
    cx: float,
    cy: float,
    pond_max_wse: float,
    max_distance_m: float,
    drains_to_outfall: set[Any],
    river_geoms: list[Any] | None = None,
    river_tree: Any = None,
    min_slope: float = 1e-3,
) -> tuple[Any, float] | None:
    """Find the euclidean-nearest junction the pond can gravity-drain to.

    A candidate must (a) carry a ``chamber_floor_elevation`` with
    ``cfe + min_slope * distance < pond_max_wse``, the slope allowance
    guarantees the in_offset logic downstream places the conduit inlet
    strictly below the pond's max WSE (no perched/dead outlets, which the
    old downstream-DAG walk produced by accepting inverts that cleared the
    WSE by centimetres); (b) drain to a SWMM outfall (``drains_to_outfall``
    closure, canal-stub fragments in other components qualify, dead-end
    junctions do not); (c) not be upstream of the pond itself (cycle
    guard); (d) not be a pond storage or outlet_junction; (e) not lie on
    the far side of a river, an outfall conduit does not cross a canal,
    so candidates whose straight segment from the pond crosses a river
    centerline are rejected (the caller offers the canal itself as the
    alternative).  Geometric nearness directly minimizes the conduit
    length, unlike the old walk which searched only the downstream cone
    of an uphill anchor and routinely returned 300-500 m ridge-crossing
    targets.

    Returns ``(node_id, distance_m)`` or ``None`` (caller falls through to
    the canal candidate and the deepen / closed-basin tiers).
    """
    upstream = nx.ancestors(graph, storage_id)
    pond_pt = shapely.Point(cx, cy)
    candidates: list[tuple[float, Any]] = []
    for n, ndata in graph.nodes(data=True):
        if n == storage_id or n in upstream or n not in drains_to_outfall:
            continue
        if ndata.get("node_type") in ("water_body", "outlet_junction"):
            continue
        cfe = ndata.get("chamber_floor_elevation")
        if cfe is None or "x" not in ndata or "y" not in ndata:
            continue
        dist = math.hypot(ndata["x"] - cx, ndata["y"] - cy)
        if dist > max_distance_m:
            continue
        if float(cfe) + min_slope * dist >= pond_max_wse:
            continue
        candidates.append((dist, n))
    candidates.sort()
    for dist, n in candidates:
        if river_tree is not None:
            seg = shapely.LineString(
                [pond_pt, shapely.Point(graph.nodes[n]["x"], graph.nodes[n]["y"])]
            )
            hits = river_tree.query(seg)
            # ``crosses`` (interiors intersect) rather than ``intersects``
            # so a sink sitting ON the canal centerline (segment endpoint
            # touching the line) is not rejected.
            if any(seg.crosses(river_geoms[int(h)]) for h in hits):  # pyright: ignore[reportOptionalSubscript]
                continue
        return n, dist
    return None


def finalize_pond_outlets(  # noqa: C901, PLR0912, PLR0915 - replaces every pond's connector with a full outlet structure in one pass
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    pond_design: PondDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Replace each pond's ``pond_connector`` edge with a proper SWMM outlet.

    Runs AFTER ``derive_topology`` and ``pipe_by_pipe`` so the pond_connector
    edge has already been oriented downstream (``storage -> network``) and
    every downstream node has a pipe_by_pipe-assigned
    ``chamber_floor_elevation``.  For each surviving pond STORAGE node
    this function:

    1. Finds the surviving pond_connector edge(s) out of the storage node,
       identifies the provisional downstream network node, and removes the
       edge.
    2. **Validates that the downstream node is physically drainable**,
       its invert must sit below the pond's max water surface elevation
       (``pond_invert + wb_max_depth``), otherwise gravity flow is
       impossible.  Three-tier rescue:

       a. Find the geometrically nearest junction (within
          ``downstream_search_max_m``) whose invert plus a min-slope
          allowance sits below the pond's max WSE and which drains to a
          SWMM outfall (see :func:`_nearest_drainable_anchor`).
       b. If no downstream anchor exists, **deepen the pond** so its
          invert sits below the provisional anchor: set
          ``pond_invert = ds_invert - min_slope * L``, capped at
          ``pond_design.max_depth_m`` (FDOT 3.81 m).  Stage-area curve,
          orifice diameter, and weir crest are rebuilt for the new depth.
       c. If deepening past the FDOT cap would be needed, the pond is
          a true closed basin; log a warning and leave it attached.  The
          caller treats these as closed basins (step 3 of the improvement
          plan).
    3. Creates an ``outlet_junction`` node 10% of the way from the pond
       centroid toward the (possibly rerouted) downstream node.
    4. Adds an ORIFICE + WEIR pair (``storage -> outlet``).
    5. Adds a CONDUIT (``outlet -> downstream_network_node``) with an
       ``in_offset`` enforcing positive slope when the rerouted anchor is
       still slightly above the pond invert (common in flat terrain).

    Pond storage nodes that lost their connector during topology derivation
    are dropped at the end of this function (orphan cleanup).
    """
    graph = graph.copy()

    # Collect connector edges per pond so we can mutate the graph safely.
    connectors: dict[int, list[tuple[Any, Any, int, dict[str, Any]]]] = {}
    for u, v, k, d in list(graph.edges(data=True, keys=True)):
        if d.get("edge_type") != "pond_connector":
            continue
        # After topology derivation the connector is directed storage -> network.
        if graph.nodes[u].get("node_type") == "water_body":
            storage_id = u
        elif graph.nodes[v].get("node_type") == "water_body":
            # Unexpected: reversed during topology. Still wire the pond,
            # but log it so we notice.
            logger.warning(
                f"pond_connector {u}->{v} has the pond as target; "
                "topology may have inverted the intended direction"
            )
            storage_id = v
        else:
            continue
        connectors.setdefault(storage_id, []).append((u, v, k, d.copy()))

    existing_int_nodes = [n for n in graph.nodes if isinstance(n, int)]
    next_node_id = (max(existing_int_nodes) + 1) if existing_int_nodes else 0

    # Outfall-reachability closure for the drainable-anchor rescue: a
    # rescue target must actually drain somewhere, but need not share a
    # component with the pond (canal-stub fragments are ideal targets).
    drains_to_outfall = _drains_to_outfall_closure(graph)

    # River centerlines: junction candidates must not lie across a canal
    # (an outfall conduit does not cross a river), and the canal itself is
    # a first-class rescue target, when it is nearer than any drainable
    # junction and the pond clears its (hydroflattened DEM) stage, the
    # pond discharges to a sink snapped onto the canal instead.
    rescue_river_geoms = [
        d["geometry"]
        for _, _, d in graph.edges(data=True)
        if d.get("edge_type") == "river" and d.get("geometry") is not None
    ]
    rescue_river_tree = shapely.STRtree(rescue_river_geoms) if rescue_river_geoms else None
    dem_src = None
    if rescue_river_tree is not None:
        elev_path = addresses.bbox_paths.elevation
        if elev_path.exists():
            import rasterio

            dem_src = rasterio.open(elev_path)

    finalized = 0
    rerouted_to_downstream = 0
    deepened = 0
    stuck_on_uphill = 0
    for storage_id, conn_list in connectors.items():
        storage_data = graph.nodes[storage_id]
        max_depth = storage_data.get("wb_max_depth", pond_design.max_depth_m)
        area_m2 = storage_data.get("wb_area_m2", 500.0)
        orifice_diam = storage_data.get(
            "wb_orifice_diam_m", _orifice_diameter_m(storage_data.get("wb_volume_m3", 0.0))
        )
        weir_length = storage_data.get("wb_weir_length_m", pond_design.weir_length_m)
        weir_crest = storage_data.get("wb_weir_crest_m", pond_design.weir_crest_ratio * max_depth)

        storage_se = storage_data.get("surface_elevation")
        if storage_se is None:
            # Fall back to the downstream node's elevation if the pond's
            # DEM lookup failed (rare; happens on the raster edge).
            ds_node = conn_list[0][1] if conn_list[0][0] == storage_id else conn_list[0][0]
            storage_se = graph.nodes[ds_node].get("surface_elevation", 0.0)
        invert_elev = storage_se - max_depth
        pond_max_wse = invert_elev + max_depth  # = storage_se, pond's top WSE
        graph.nodes[storage_id]["surface_elevation"] = storage_se
        graph.nodes[storage_id]["chamber_floor_elevation"] = invert_elev

        # Pond centroid and a representative downstream node for placement.
        cx = storage_data["x"]
        cy = storage_data["y"]

        for u, v, k, d in conn_list:
            # Downstream node is whichever end of the connector isn't the pond.
            downstream_node = v if u == storage_id else u
            if not graph.has_edge(u, v, key=k):
                continue
            graph.remove_edge(u, v, key=k)

            # --- Drainable-anchor validation ---------------------------------
            # pipe_by_pipe has set chamber_floor_elevation on every pipe node.
            # Check whether the provisional downstream node is actually
            # below the pond's max WSE; if not, find the geometrically
            # nearest junction that is (with a slope allowance so the
            # rescued conduit inlet lands strictly below the WSE).  We
            # only reroute if the search finds a usable anchor,
            # otherwise we leave the pond attached to the provisional
            # node and let the pipe in_offset logic below either raise
            # the pipe (if the mismatch is small) or accept a
            # non-draining pond (closed-basin behavior handled by step
            # 3 of the improvement plan).  Canal-anchored ponds (sink is
            # a river_outfall) are never rescued away from their canal.
            ds_invert_raw = graph.nodes[downstream_node].get("chamber_floor_elevation")
            ds_invert_initial: float | None = (
                float(ds_invert_raw) if ds_invert_raw is not None else None
            )
            needs_reroute = (
                ds_invert_initial is not None
                and ds_invert_initial >= pond_max_wse
                and graph.nodes[downstream_node].get("node_type") != "river_outfall"
            )
            if needs_reroute and ds_invert_initial is not None:
                hit = _nearest_drainable_anchor(
                    graph,
                    storage_id,
                    cx,
                    cy,
                    pond_max_wse,
                    max_distance_m=pond_design.downstream_search_max_m,
                    drains_to_outfall=drains_to_outfall,
                    river_geoms=rescue_river_geoms,
                    river_tree=rescue_river_tree,
                )
                # Canal candidate: when the canal centerline is nearer than
                # the best drainable junction and the pond clears the
                # canal's water stage (hydroflattened DEM at the snap
                # point) with the same slope allowance, discharge to a
                # sink snapped onto the canal, the FDOT answer for a
                # canal-adjacent pond, and the only crossing-free target
                # when every near junction sits on the far bank.  The
                # displaced anchor keeps the interception role.
                if rescue_river_tree is not None and dem_src is not None:
                    pond_pt = shapely.Point(cx, cy)
                    near_idx = int(rescue_river_tree.nearest(pond_pt))
                    d_canal = pond_pt.distance(rescue_river_geoms[near_idx])
                    if d_canal <= pond_design.downstream_search_max_m and (
                        hit is None or d_canal < hit[1]
                    ):
                        snap_line = shapely.shortest_line(pond_pt, rescue_river_geoms[near_idx])
                        snap_pt = shapely.Point(snap_line.coords[-1])
                        stage = next(iter(dem_src.sample([(snap_pt.x, snap_pt.y)])))[0]
                        nodata = dem_src.nodata
                        stage_ok = (
                            not np.isnan(stage)
                            and (nodata is None or stage != nodata)
                            and float(stage) + 1e-3 * d_canal < pond_max_wse
                        )
                        if stage_ok:
                            sink_id = next_node_id
                            next_node_id += 1
                            graph.add_node(
                                sink_id,
                                x=snap_pt.x,
                                y=snap_pt.y,
                                node_type="river_outfall",
                                surface_elevation=float(stage),
                                chamber_floor_elevation=float(stage),
                            )
                            if graph.nodes[storage_id].get("wb_intercept_node") is None:
                                graph.nodes[storage_id]["wb_intercept_node"] = downstream_node
                            hit = (sink_id, d_canal)
                if hit is not None:
                    better_node, dist_m = hit
                    better_invert = float(graph.nodes[better_node]["chamber_floor_elevation"])
                    logger.info(
                        f"pond {storage_id}: rerouting outflow from "
                        f"{downstream_node} "
                        f"(invert {ds_invert_initial:.2f} m, "
                        f"above pond WSE {pond_max_wse:.2f} m) to "
                        f"{better_node} "
                        f"(invert {better_invert:.2f} m, "
                        f"{dist_m:.0f} m away)"
                    )
                    downstream_node = better_node
                    rerouted_to_downstream += 1
                else:
                    # STEP 2: try to deepen the pond.  The pond top
                    # (storage_se) is fixed by the DEM; we can only
                    # *lower* the invert (excavate) to make the pond
                    # deeper.  Two gates must pass for deepening to
                    # produce gravity drainage:
                    #
                    # (a) pond top must sit above the downstream pipe
                    #     invert plus minimum-slope head, otherwise
                    #     water physically cannot reach the pipe inlet
                    #     no matter how deep we dig (closed basin).
                    # (b) the required depth must be within the FDOT
                    #     design cap (``max_depth_m``, 3.81 m default);
                    #     deeper excavations are unrealistic for the
                    #     subdivision-pond class of structures we are
                    #     modeling.
                    dx_tmp = graph.nodes[downstream_node]["x"]
                    dy_tmp = graph.nodes[downstream_node]["y"]
                    est_pipe_length = max(
                        ((dx_tmp - cx) ** 2 + (dy_tmp - cy) ** 2) ** 0.5 * 0.9,
                        1.0,
                    )
                    slope_head = 0.001 * est_pipe_length
                    pond_top_above_pipe = storage_se > (ds_invert_initial + slope_head)
                    target_invert = ds_invert_initial - slope_head
                    target_depth = storage_se - target_invert
                    can_deepen_in_cap = (
                        target_depth > max_depth and target_depth <= pond_design.max_depth_m
                    )
                    if pond_top_above_pipe and can_deepen_in_cap:
                        logger.info(
                            f"pond {storage_id}: deepening from "
                            f"{max_depth:.2f} m to {target_depth:.2f} m "
                            f"(pond WSE {pond_max_wse:.2f} m, pipe invert "
                            f"{ds_invert_initial:.2f} m, L~{est_pipe_length:.0f} m); "
                            "rebuilding stage-area curve + outlet sizing."
                        )
                        # Recompute geometric state for the deeper pond.
                        # FDOT volume depends only on area (unchanged), so
                        # Opti-CMAC orifice diameter does not change; what
                        # changes is the frustum depth and its stage-area
                        # curve plus the weir crest (always 0.9 * depth).
                        max_depth = target_depth
                        invert_elev = target_invert
                        volume_m3 = _fdot_volume_m3(
                            area_m2,
                            pond_design.fdot_volume_intercept,
                            pond_design.fdot_volume_slope,
                        )
                        new_curve = _stage_area_curve(
                            area_m2,
                            max_depth,
                            pond_design.side_slope_pct,
                            pond_design.bottom_area_min_ratio,
                            pond_design.n_curve_points,
                        )
                        orifice_diam = _orifice_diameter_m(volume_m3)
                        weir_length = _weir_length_m(area_m2, pond_design.weir_length_m)
                        weir_crest = pond_design.weir_crest_ratio * max_depth
                        # Persist to the graph node so post_processing and
                        # later steps see the new geometry.
                        graph.nodes[storage_id]["wb_max_depth"] = max_depth
                        graph.nodes[storage_id]["chamber_floor_elevation"] = invert_elev
                        graph.nodes[storage_id]["wb_stage_area_curve"] = new_curve
                        graph.nodes[storage_id]["wb_orifice_diam_m"] = orifice_diam
                        graph.nodes[storage_id]["wb_weir_length_m"] = weir_length
                        graph.nodes[storage_id]["wb_weir_crest_m"] = weir_crest
                        deepened += 1
                    else:
                        stuck_on_uphill += 1
                        # Mark the pond as a closed basin so post_processing
                        # can emit a SurDepth + seepage row (step 3): water
                        # overtopping MaxDepth accumulates on the surface
                        # instead of disappearing into SWMM's flooding term,
                        # and the pond slowly drains via exfiltration
                        # through its bottom at a rate matching typical
                        # Florida retention-pond design values.
                        graph.nodes[storage_id]["wb_closed_basin"] = True
                        if not pond_top_above_pipe:
                            reason = (
                                f"pond top ({storage_se:.2f} m) sits below "
                                f"the provisional pipe invert "
                                f"({ds_invert_initial:.2f} m), gravity "
                                "drainage impossible regardless of depth"
                            )
                        else:
                            reason = (
                                f"would need to deepen to {target_depth:.2f} m, "
                                f"exceeding the "
                                f"{pond_design.max_depth_m:.2f} m FDOT cap"
                            )
                        logger.warning(
                            f"pond {storage_id}: no drainable downstream node "
                            f"within {pond_design.downstream_search_max_m:.0f} m "
                            f"and {reason}; modeling as closed basin with "
                            "surface ponding + seepage exfiltration."
                        )

            dx = graph.nodes[downstream_node]["x"]
            dy = graph.nodes[downstream_node]["y"]

            outlet_id = next_node_id
            next_node_id += 1
            # Place outlet 10% of the way from the pond toward the downstream node.
            ox = cx + (dx - cx) * 0.1
            oy = cy + (dy - cy) * 0.1
            graph.add_node(
                outlet_id,
                x=ox,
                y=oy,
                surface_elevation=storage_se,
                chamber_floor_elevation=invert_elev,
                node_type="outlet_junction",
                wb_index=storage_data.get("wb_index"),
            )

            orifice_geom = shapely.LineString([(cx, cy), (ox, oy)])
            graph.add_edge(
                storage_id,
                outlet_id,
                geometry=orifice_geom,
                length=max(shapely.length(orifice_geom), 0.1),
                edge_type="orifice",
                id=f"{storage_id}-{outlet_id}-orifice",
                orifice_type="SIDE",
                orifice_diam_m=orifice_diam,
                orifice_cd=pond_design.orifice_cd,
                # Crest offset above pond invert eliminates per-step
                # free/submerged regime flipping in DYNWAVE; default 0.10 m
                # cuts non-convergence dramatically on flat-terrain ponds.
                # See PondDesign.orifice_crest_offset_m.
                orifice_offset=pond_design.orifice_crest_offset_m,
            )
            graph.add_edge(
                storage_id,
                outlet_id,
                geometry=orifice_geom,
                length=max(shapely.length(orifice_geom), 0.1),
                edge_type="weir",
                id=f"{storage_id}-{outlet_id}-weir",
                weir_type="TRANSVERSE",
                weir_crest_m=weir_crest,
                weir_length_m=weir_length,
                weir_cd=pond_design.weir_cw,
            )

            # Conduit: outlet_junction -> downstream_network_node.
            outflow_geom = shapely.LineString([(ox, oy), (dx, dy)])
            length = max(shapely.length(outflow_geom), 1.0)
            ds_invert = graph.nodes[downstream_node].get("chamber_floor_elevation", invert_elev)
            min_slope = 0.001
            in_offset = 0.0
            if invert_elev < ds_invert + min_slope * length:
                in_offset = (ds_invert - invert_elev) + min_slope * length
            graph.add_edge(
                outlet_id,
                downstream_node,
                geometry=outflow_geom,
                length=length,
                edge_type="pond_outflow",
                id=f"{outlet_id}-{downstream_node}",
                roughness=d.get("roughness", 0.035),
                channel_width=d.get("channel_width", max(area_m2**0.5 * 0.1, 1.0)),
                channel_depth=max_depth,
                diameter=0.0,
                in_offset=in_offset,
                out_offset=0.0,
            )

        finalized += 1

    if dem_src is not None:
        dem_src.close()

    # Drop pond storage nodes that lost their pond_connector during topology
    # derivation (e.g. because the connector's downstream endpoint was pruned).
    # Such ponds have no orifice, no weir, no outflow conduit and would land
    # in the .inp as disconnected STORAGE nodes, water can enter via
    # subcatchment Outlet routing but can never leave, producing spurious
    # continuity error.  We also reroute any subcatchments that were pointed at the pond
    # back to the pond's original ``wb_snapped_to`` anchor so their runoff
    # isn't lost to a dead-end storage.
    orphan_ponds = [
        n
        for n, d in graph.nodes(data=True)
        if d.get("node_type") == "water_body"
        and not any(
            graph.edges[n, v, k].get("edge_type") == "orifice"
            for v, k in [(v_, k_) for _, v_, k_ in graph.out_edges(n, keys=True)]
        )
    ]
    if orphan_ponds:
        subs_path = addresses.model_paths.subcatchments
        subs_gdf: gpd.GeoDataFrame | None = None
        if subs_path.exists():
            subs_gdf = gpd.read_parquet(subs_path)
        reroute_updates: dict[Any, Any] = {}
        for pid in orphan_ponds:
            pdata = graph.nodes[pid]
            anchor = pdata.get("wb_snapped_to")
            if anchor is not None and anchor in graph.nodes:
                reroute_updates[pid] = anchor
            graph.remove_node(pid)
        if subs_gdf is not None and reroute_updates and "id" in subs_gdf.columns:
            mask = subs_gdf["id"].isin(reroute_updates)
            if mask.any():
                subs_gdf.loc[mask, "id"] = (
                    subs_gdf.loc[mask, "id"].map(reroute_updates).astype("int64")
                )
                subs_gdf.to_parquet(subs_path)
        logger.info(
            f"Removed {len(orphan_ponds)} orphan pond storage node(s) that lost "
            "their outlet during topology derivation; subcatchments rerouted to "
            "the original snap anchor."
        )

    # Post-finalize viability check: for every pond not yet marked
    # closed-basin, verify that at least one reachable SWMM outfall sits
    # at least ``min_outfall_head_drop_m`` below the pond's max water
    # surface elevation.  Ponds whose outflow chain ends at a dummy
    # OSM-river terminus that's only fractions of a meter below the pond
    # (e.g. on the test catchment at invert 5.1-5.4 m vs pond max WSE 4-6 m,
    # giving 0-1 m of head) can't physically convey their design release
    # through the 1-2 km flat dummy-river segment.  In that case we model
    # the pond as closed-basin retention with Green-Ampt exfiltration
    # (SJRWMD/SFWMD 72-hour-recovery rule, FDOT flatwoods practice).
    # ``post_processing.synthetic_write`` reads ``wb_closed_basin`` and
    # emits the ``SurDepth`` + Green-Ampt seepage tail; the orifice /
    # weir / pond_outflow chain stays in the graph as a backup overflow
    # path.
    min_head_drop = float(pond_design.min_outfall_head_drop_m)
    converted = 0
    if min_head_drop > 0:
        for pid, pdata in graph.nodes(data=True):
            if pdata.get("node_type") != "water_body":
                continue
            if pdata.get("wb_closed_basin"):
                continue
            if not _pond_has_viable_outfall(graph, pid, min_head_drop):
                graph.nodes[pid]["wb_closed_basin"] = True
                converted += 1
                logger.info(
                    f"pond {pid}: marking closed-basin, no SWMM outfall "
                    f"reachable with >= {min_head_drop:.1f} m head drop below "
                    f"pond max WSE on any forward path; will rely on "
                    "Green-Ampt seepage for drainage."
                )

    logger.info(
        f"Finalized outlets for {finalized} pond(s); each storage now feeds "
        "an outlet junction via orifice + weir, then to the network. "
        f"Rerouted {rerouted_to_downstream}; deepened {deepened}; "
        f"{stuck_on_uphill + converted} treated as closed basins "
        f"({stuck_on_uphill} no-gravity-anchor, {converted} no-viable-outfall-path)."
    )
    return graph


# --------------------------------------------------------------------------- #
# Shared partition: pond-outlet sinks + node->pond assignment
# --------------------------------------------------------------------------- #


def partition_pond_network(  # noqa: C901, PLR0912, PLR0915 - storage/ds_node-mode BFS partition kept as one delineation-consistent routine
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    keep_edge_types: frozenset[str] = frozenset(
        {
            "pipe",
            "street_channel",
            "orifice",
            "weir",
            "pond_inflow",
            "pond_outflow",
            "outfall",
        }
    ),
    pond_buffer_m: float = 50.0,
    pond_sink_mode: Literal["ds_node", "storage"] = "ds_node",
) -> tuple[
    nx.MultiDiGraph[Any],
    dict[Any, list[Any]],
    set[Any],
    dict[Any, Any],
]:
    """Compute the shared pond-routing partition of a graph.

    ``pond_sink_mode`` controls what the upstream BFS treats as the
    pond sink:

    - ``"ds_node"`` (default, used by :func:`route_pipes_into_ponds`):
      the sink is the pond's downstream junction (target of the
      ``pond_outflow`` edge).  BFS collects everything upstream of that
      junction.  Correct for *pipe rerouting* (we want every feeder
      that lands at the ds_node), but for *pondshed delineation* it
      over-collects, it sweeps in subs that bypass the pond and merely
      share its downstream junction, producing spatially-disconnected
      pondsheds.

    - ``"storage"`` (used by :func:`post_processing.generate_pondsheds`):
      the sink is the pond **storage** node itself.  BFS goes upstream
      only through the catchment that drains *through* the pond
      (``pond_inflow`` conduits + their upstream pipes + subs whose
      ``Outlet`` is the storage).  This is the hydrologically-correct
      pondshed: the area whose runoff the pond actually manages.  The
      orifice / weir / pond_outflow edges (pond → downstream) are NOT
      traversed, so a pond's pondshed never leaks past its own outlet.

    Used by both :func:`route_pipes_into_ponds` (pipe-rerouting step) and
    :func:`post_processing.generate_pondsheds` (visualization output) so that
    pipes and subcatchments end up consistently assigned to the same ponds.

    Algorithm:

    1. Build a filtered drainage network containing only the edge types we
       care about for pond routing, by default pipes, surface dual-drainage
       channels, and pond outlet/outflow connectors (drops natural
       ``river`` edges).
    2. Identify two kinds of sinks:

       - **Pond outlets**: target nodes of every ``pond_outflow`` edge.
       - **Lake outlets**: outlet nodes of subcatchments that are
         dominated by an OSM water polygon (``basins.parquet``) which
         was *not* matched to any pond storage (within ``pond_buffer_m``
         of the polygon).  These are the natural water bodies the pond
         classifier rejected.

    3. Multi-source BFS upstream from every sink in the filtered graph,
       blocked at other sinks.  Each non-sink node is assigned to its
       nearest downstream sink.
    4. For ds_nodes shared by multiple ponds, split their captured
       upstream area by spatial Voronoi: each upstream node is assigned
       to the pond storage whose centroid is geographically closest.

    Returns:
        ``(pipe_graph, pond_outlets, lake_outlets, node_to_pond)`` where:

        - ``pipe_graph`` (``nx.MultiDiGraph``): filtered network used for
          BFS.
        - ``pond_outlets`` (``dict[ds_node, list[pond_storage_id]]``):
          maps each pond outlet sink to the pond(s) discharging there.
        - ``lake_outlets`` (``set[node_id]``): the additional sinks added
          to keep lake catchments out of pondsheds.
        - ``node_to_pond`` (``dict[node_id, pond_storage_id]``): every
          node whose drainage path leads to a pond outlet sink (after
          Voronoi for multi-pond ds_nodes).  Nodes whose drainage path
          leads to a lake outlet (or to nothing) are absent.
    """
    pond_set: set[Any] = {
        n for n, d in graph.nodes(data=True) if d.get("node_type") == "water_body"
    }

    # Step 1: filtered network ---------------------------------------------
    pipe_graph: nx.MultiDiGraph[Any] = nx.MultiDiGraph()
    pipe_graph.add_nodes_from(graph.nodes(data=True))
    for u, v, k, d in graph.edges(keys=True, data=True):
        if d.get("edge_type") in keep_edge_types:
            pipe_graph.add_edge(u, v, key=k, **d)

    # Step 2a: pond outlets via pond_outflow edges
    pond_outlets: dict[Any, list[Any]] = {}
    for u, v, _k, d in graph.edges(keys=True, data=True):
        if d.get("edge_type") != "pond_outflow":
            continue
        for pp in graph.predecessors(u):
            if graph.nodes[pp].get("node_type") == "water_body":
                pond_outlets.setdefault(v, []).append(pp)

    # Step 2a': canal-anchored ponds discharge to a snapped sink on the
    # river centerline, where no pipes exist to intercept.  Their street
    # interception point is the pour-point anchor displaced at insert time
    # (``wb_intercept_node``), so in ds_node mode the pond is also
    # registered there, feeder pipes arriving at that junction reroute
    # through the pond (detention) before it releases to the canal.
    if pond_sink_mode == "ds_node":
        for p in pond_set:
            intercept = graph.nodes[p].get("wb_intercept_node")
            if intercept is not None and intercept in graph.nodes:
                ponds_here = pond_outlets.setdefault(intercept, [])
                if p not in ponds_here:
                    ponds_here.append(p)

    # Step 2b: lake outlets via basins.parquet ------------------------------
    lake_outlets: set[Any] = set()
    if pond_set:
        try:
            basins_path = addresses.bbox_paths.basins
        except Exception:  # noqa: BLE001 - addresses can be a stub in tests
            basins_path = None
        if basins_path is not None and basins_path.exists():
            try:
                basins = gpd.read_parquet(basins_path)
            except Exception:  # noqa: BLE001 - missing or corrupt parquet, skip lakes
                basins = gpd.GeoDataFrame()
            crs = graph.graph.get("crs")
            if not basins.empty and crs is not None:
                basins_proj = basins.to_crs(crs)
                pond_pts = [
                    shapely.Point(graph.nodes[p]["x"], graph.nodes[p]["y"]) for p in pond_set
                ]
                pond_buf = shapely.unary_union([pt.buffer(pond_buffer_m) for pt in pond_pts])
                is_pond_basin = basins_proj.geometry.intersects(pond_buf)
                lake_polys_gdf = basins_proj.loc[~is_pond_basin, ["geometry"]]
                if not lake_polys_gdf.empty:
                    lake_mask = shapely.unary_union(lake_polys_gdf.geometry.to_numpy())
                    subs_path = addresses.model_paths.subcatchments
                    if subs_path.exists():
                        subs = gpd.read_parquet(subs_path)
                        if not subs.empty:
                            cand: set[Any] = set()
                            centroids = subs.geometry.centroid
                            in_lake = centroids.within(lake_mask)
                            cand |= set(subs.loc[in_lake, "id"].astype("int64").tolist())
                            for lake_geom in lake_polys_gdf.geometry:
                                inter = subs.geometry.intersection(lake_geom).area
                                contains_majority = inter > 0.5 * lake_geom.area
                                cand |= set(
                                    subs.loc[contains_majority, "id"].astype("int64").tolist()
                                )
                            lake_outlets = {
                                sid
                                for sid in cand
                                if sid in pipe_graph.nodes and sid not in pond_set
                            }

    # Step 3: multi-source BFS upstream from sinks --------------------------
    if pond_sink_mode == "storage":
        # Sinks are the pond STORAGE nodes themselves.  Upstream of a
        # storage node (via predecessors) is exactly its pond_inflow
        # feeders + their pipe catchment + subs whose Outlet is the
        # storage.  Orifice / weir / pond_outflow are storage
        # *successors*, so a predecessor-BFS never crosses them, the
        # pondshed cannot leak past the pond's own outlet.  Each storage
        # is its own sink, so there is no multi-pond tiebreak.
        pond_sinks: set[Any] = set(pond_set)
    else:
        # ds_node mode (pipe-rerouting): sink is the downstream junction.
        pond_sinks = set(pond_outlets)
    sinks = pond_sinks | lake_outlets
    assignment: dict[Any, Any] = {s: s for s in sinks}
    queue: collections.deque[Any] = collections.deque(sinks)
    while queue:
        node = queue.popleft()
        sink = assignment[node]
        for pred in pipe_graph.predecessors(node):
            if pred in assignment:
                continue
            assignment[pred] = sink
            queue.append(pred)

    # Step 4: per-node pond assignment --------------------------------------
    node_to_pond: dict[Any, Any] = {}
    if pond_sink_mode == "storage":
        # The sink IS the pond storage id; map directly (lake-outlet
        # sinks are not in pond_set, so they're skipped).
        node_to_pond = {node: sink for node, sink in assignment.items() if sink in pond_set}
    else:
        # ds_node mode: resolve each sink's pond(s), Voronoi tiebreak
        # when several ponds share one downstream junction.
        for node, sink in assignment.items():
            if sink not in pond_outlets:
                continue  # lake-outlet partition; not a pond
            ponds_here = pond_outlets[sink]
            if len(ponds_here) == 1:
                node_to_pond[node] = ponds_here[0]
                continue
            # Multi-pond ds_node: closest pond by Euclidean distance
            # between the node and each candidate pond's storage centroid.
            nx_data = graph.nodes[node]
            nx_pt = shapely.Point(nx_data["x"], nx_data["y"])
            best_pond = ponds_here[0]
            best_d = float("inf")
            for pid in ponds_here:
                pond_data = graph.nodes[pid]
                dist = nx_pt.distance(shapely.Point(pond_data["x"], pond_data["y"]))
                if dist < best_d:
                    best_d = dist
                    best_pond = pid
            node_to_pond[node] = best_pond

    return pipe_graph, pond_outlets, lake_outlets, node_to_pond


def _compute_pond_outflow_carriers(
    pipe_graph: nx.MultiDiGraph[Any], pond_ds_nodes: set[Any]
) -> set[Any]:
    """Forward-BFS from every pond ds_node; returns the union of visited nodes.

    Used by :func:`route_pipes_into_ponds` to skip pipes whose upstream
    end is downstream of another pond's outflow, those pipes carry pond
    A's released water and must not be rerouted into pond B (per the
    spec: pond B's controls only see its own pondshed).  The set
    INCLUDES the source ds_nodes themselves so the rule also catches
    pipes whose ``u`` IS another pond's ds_node directly.
    """
    visited: set[Any] = set()
    for ds in pond_ds_nodes:
        if ds not in pipe_graph.nodes or ds in visited:
            visited.add(ds)
        queue: collections.deque[Any] = collections.deque([ds])
        local_seen = {ds}
        while queue:
            n = queue.popleft()
            for succ in pipe_graph.successors(n):
                if succ in local_seen:
                    continue
                local_seen.add(succ)
                queue.append(succ)
        visited |= local_seen
    return visited


def route_pipes_into_ponds(  # noqa: C901, PLR0912, PLR0915 - canonical EPA reroute step has many branches per pipe
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    pond_design: PondDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Reroute upstream pipes to terminate at pond storage nodes.

    Implements the canonical SWMM detention-pond connectivity pattern
    documented in the EPA SWMM 5 Applications Manual (Rossman 2009),
    Example 3, Detention Pond Design:

        "The storage unit is first connected to the rest of the
        drainage system. This can be done by changing culvert C11's
        outlet to SU1... Culvert C11 is given a downstream offset of
        1 ft so that for minor storms it has no backwater but still
        has its crown below the top of the storage pond."

    For each pipe ``(u, v, k)`` where ``v`` is a pond outlet sink
    (target of a ``pond_outflow`` edge), the pipe is removed and re-
    added as ``(u, target_pond, k)`` with ``edge_type="pond_inflow"``.
    The target pond is selected by the shared partition's
    ``node_to_pond`` map, which uses a Voronoi tiebreaker when the
    sink hosts a cluster of ponds.

    Pipes whose ``u`` is downstream of *any* pond's outflow are
    skipped, they carry that upstream pond's released water and must
    not feed back into another pond (per the spec: each pond's
    controls only see its own pondshed; cascading pond outflows merge
    passively at their shared downstream junction, not through
    another pond's storage).  This is enforced via
    :func:`_compute_pond_outflow_carriers`.

    Adverse-slope guarantee: the rerouter computes
    ``in_offset = (D_invert - P_invert) + pond_inflow_offset_m`` so the
    pipe's downstream end stays at (or above) the elevation it had at
    the original ds_node, preserving the pipe's original positive
    slope.  Because :func:`finalize_pond_outlets` already ensured
    ``P_invert <= D_invert`` for every gravity-drained pond, the
    slope-preserving offset is non-negative by construction.  For
    closed-basin ponds where ``finalize_pond_outlets`` had to leave
    ``P_invert > D_invert``, the pond is deepened *just enough* so the
    invariant holds for every feeder pipe being rerouted; if that
    deepening would exceed ``pond_design.max_depth_m``, the affected
    feeder gets an ``out_offset`` at its upstream end so its u-side
    invert is lifted above the pond invert plus a min-slope buffer.
    No pipe ever ends up adverse-slope.

    Args:
        graph: Graph after :func:`finalize_pond_outlets`.  Must already
            have ``chamber_floor_elevation`` set on every node.
        addresses: File path manager (used to read basins.parquet for
            the lake-outlet detection in the partition).
        pond_design: Provides ``pond_inflow_offset_m`` (default 0.30 m
            per EPA C11 1-ft pattern) and ``max_depth_m`` (FDOT cap).
        **kwargs: Ignored (pipeline framework).

    Returns:
        The graph with rerouted pipes.  Number of pond_inflow edges
        equals the count of pond outlet sinks' inbound pipes that
        passed the carrier filter (logged on completion).
    """
    graph = graph.copy()

    pipe_graph, pond_outlets, _lake_outlets, node_to_pond = partition_pond_network(graph, addresses)
    if not pond_outlets:
        logger.info("route_pipes_into_ponds: no pond_outflow edges; nothing to do.")
        return graph

    pond_ds_nodes = set(pond_outlets)
    carriers = _compute_pond_outflow_carriers(pipe_graph, pond_ds_nodes)

    inflow_offset = float(pond_design.pond_inflow_offset_m)
    max_depth = float(pond_design.max_depth_m)
    min_slope = 1e-3  # SWMM minimum positive slope (matches finalize_pond_outlets)

    # Collect candidate edges first so we don't mutate the graph during
    # iteration.  Each candidate is (u, v=ds_node, k, attrs, target_pond).
    candidates: list[tuple[Any, Any, int, dict[str, Any], Any]] = []
    skipped_carrier = 0
    skipped_no_target = 0
    for u, v, k, d in list(graph.edges(keys=True, data=True)):
        if d.get("edge_type") != "pipe":
            continue
        if v not in pond_outlets:
            continue
        if u in carriers:
            skipped_carrier += 1
            continue
        target_pond = node_to_pond.get(u)
        if target_pond is None:
            # u has no partition assignment (e.g. detached subgraph),
            # fall back to the closest pond at this ds_node.
            ponds_here = pond_outlets[v]
            ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
            best_pond = ponds_here[0]
            best_d = float("inf")
            for pid in ponds_here:
                pd_data = graph.nodes[pid]
                dist = ((pd_data["x"] - ux) ** 2 + (pd_data["y"] - uy) ** 2) ** 0.5
                if dist < best_d:
                    best_d = dist
                    best_pond = pid
            target_pond = best_pond
        if target_pond is None:
            skipped_no_target += 1
            continue
        candidates.append((u, v, k, dict(d), target_pond))

    if not candidates:
        logger.info(
            f"route_pipes_into_ponds: no eligible pipes "
            f"(skipped {skipped_carrier} carriers, "
            f"{skipped_no_target} no-target)."
        )
        return graph

    # Tier-1 / Tier-2 / Tier-3 elevation handling --------------------------
    # First pass: per pond, find the lowest required pond invert across all
    # of its rerouted feeders.  If the existing pond invert is already at
    # or below that, Tier 1 (no deepening).  Otherwise deepen (Tier 2).
    pond_to_feeders: dict[Any, list[tuple[Any, Any, int, dict[str, Any]]]] = {}
    for u, v, k, attrs, target_pond in candidates:
        pond_to_feeders.setdefault(target_pond, []).append((u, v, k, attrs))

    tier_counts = {1: 0, 2: 0, 3: 0}
    deepened_ponds = 0
    raised_upstreams = 0
    for pond_id, feeders in pond_to_feeders.items():
        pdata = graph.nodes[pond_id]
        p_invert = float(pdata["chamber_floor_elevation"])
        p_surface = float(pdata.get("surface_elevation", p_invert + max_depth))
        # The shared ds_node for this pond (any feeder works; they all share it).
        ds_node = feeders[0][1]
        d_invert = float(graph.nodes[ds_node].get("chamber_floor_elevation", p_invert))

        # The slope-preserving offset target is D_invert (so the pipe's
        # downstream end stays where it used to be).  We need the pipe's
        # upstream end to satisfy ``u_invert >= D_invert + min_slope * length``
        # to keep at least the SWMM minimum slope on the rerouted pipe.
        # Tier 2 needs P_invert <= D_invert so that in_offset >= 0.
        if p_invert > d_invert:
            # Deepen to D_invert (capped at max_depth).
            new_p_invert = max(d_invert, p_surface - max_depth)
            if new_p_invert < p_invert:
                # Successful deepen, rebuild the full depth-dependent geometry
                # (stage-area curve, weir crest, orifice) so the stored curve
                # top and weir crest follow the new, deeper MaxDepth instead of
                # lagging at the pre-deepen depth.
                _apply_pond_geometry(
                    graph,
                    pond_id,
                    pond_design,
                    float(pdata.get("wb_area_m2", 0.0)),
                    p_surface - new_p_invert,
                    new_p_invert,
                )
                p_invert = new_p_invert
                deepened_ponds += 1

        for u, v, k, attrs in feeders:
            length = float(attrs.get("length", 0) or 0)
            length = max(length, 1.0)  # avoid div-by-zero / SWMM L>=D rules
            u_invert = float(graph.nodes[u].get("chamber_floor_elevation", d_invert))

            # Slope-preserving target downstream invert = D_invert; the pipe's
            # downstream end sits at P_invert + in_offset.  Add the EPA
            # backwater offset on top so the crown stays just below the pond
            # top in minor storms.
            target_ds_invert = d_invert
            in_offset = max(0.0, target_ds_invert - p_invert) + inflow_offset

            # Pipe slope check: u_invert + out_offset >= P_invert + in_offset + min_slope * L
            out_offset = float(attrs.get("out_offset", 0) or 0)
            current_slope = ((u_invert + out_offset) - (p_invert + in_offset)) / length

            tier = 1
            if current_slope < min_slope:
                # Tier 3: lift the pipe's upstream end so the slope is satisfied.
                needed_lift = (p_invert + in_offset + min_slope * length) - (u_invert + out_offset)
                out_offset += needed_lift
                raised_upstreams += 1
                tier = 3
            elif p_invert < float(pdata["chamber_floor_elevation"]) - 1e-9:
                tier = 2  # this pond was deepened above
            tier_counts[tier] += 1

            # Replace the edge: (u, ds_node) -> (u, pond_id) with new attrs.
            graph.remove_edge(u, v, key=k)
            new_attrs = dict(attrs)
            new_attrs["edge_type"] = "pond_inflow"
            new_attrs["in_offset"] = in_offset
            new_attrs["out_offset"] = out_offset
            new_attrs["length"] = length
            # Update geometry to a straight 2-point line ending at the pond.
            ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
            px, py = graph.nodes[pond_id]["x"], graph.nodes[pond_id]["y"]
            new_attrs["geometry"] = shapely.LineString([(ux, uy), (px, py)])
            new_attrs["length"] = float(shapely.length(new_attrs["geometry"]))
            new_attrs["id"] = f"{u}-{pond_id}-pond_inflow"
            graph.add_edge(u, pond_id, key=k, **new_attrs)

    n_ponds_with_inflow = len(pond_to_feeders)
    logger.info(
        f"route_pipes_into_ponds: rerouted {len(candidates)} pipe(s) into "
        f"{n_ponds_with_inflow}/{len(pond_outlets)} pond outlet(s).  "
        f"Tier 1 (slope-preserving) = {tier_counts[1]}, "
        f"Tier 2 (pond-deepened) = {tier_counts[2]}, "
        f"Tier 3 (upstream lifted) = {tier_counts[3]}.  "
        f"Skipped {skipped_carrier} pipes carrying upstream pond outflow."
    )
    return graph


def reroute_subs_to_isolated_ponds(  # noqa: C901, PLR0912, PLR0915 - Voronoi assignment + per-pond capacity cap is one coherent pass
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    pond_design: PondDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Reroute orphan subcatchments to their nearest pond with capacity.

    After :func:`route_pipes_into_ponds`, many subs still drain to street
    outfalls without passing any pond: carrier-blocked feeders, ponds
    whose ds_node has no pipe predecessors, and rescue chains that land
    ponds on each other's anchors all leave detention dormant while the
    surrounding blocks bypass it.  Physically these subdivision ponds ARE
    the detention for their neighboring blocks (SJRWMD/FDOT permitting),
    so this step changes ``Sub.Outlet`` from the current network node to
    the pond storage node for orphan subs in each pond's natural drainage
    neighborhood, subject to a per-pond catchment cap derived from the
    pond's physical storage capacity (FDOT 1-inch water-quality rule,
    Florida ERP design practice).

    Eligibility (a sub is rerouted to a pond if all are true):

    1. The sub is an orphan under the DELINEATION-CONSISTENT storage-mode
       :func:`partition_pond_network` map (the same partition
       :func:`post_processing.generate_pondsheds` uses), i.e. its runoff
       does not already transit any pond storage.  The earlier ds_node-mode
       check over-collected (it attributed subs to ponds merely sharing a
       downstream junction) and, combined with an isolation gate that a
       single carrier pipe could veto, produced rings of detention-less
       subs around starved ponds.
    2. The pond is the sub's nearest pond by Euclidean distance between
       the sub centroid and the pond storage centroid.
    3. That distance is ``≤ pond_design.sub_reroute_max_distance_m``, the
       sub is not river-fronting (within a street-block ~60 m of a river
       centerline that is nearer than the pond, those subs drain to the
       canal directly), and the straight line from the sub to the pond
       does not cross a river centerline (capture never reaches across a
       canal, the source of far-bank MultiPolygon pondshed fragments).
    4. The pond has not yet exhausted its design catchment cap, computed
       as ``wb_volume_m3 / (water_quality_depth_m * runoff_coeff)``,
       i.e. the catchment area such that 1 inch of runoff fills the
       pond's design volume (Florida ERP/FDOT water-quality treatment
       rule; SJRWMD AH Vol. II §10.2), where the cap is first DEBITED
       by the area the pond's storage-mode pondshed already captures
       through the pipe network.  Subs are added in nearest-first order
       until cumulative captured area reaches the cap; remaining subs
       stay on their original outlets even if they meet (1)-(3).  The
       multiplier ``pond_design.sub_reroute_capacity_multiplier`` scales
       this cap (default 2.0 to account for the storm-vs-WQ-design
       intensity gap noted in the handoff §4a).

    Side effects:

    - ``subs.id`` (the SWMM ``Sub.Outlet`` value) is rewritten to the pond
      storage node for each eligible sub.
    - ``contributing_area`` is transferred from the original outlet node
      to the pond storage node on the graph, so
      :func:`resize_street_pipes_for_pond_routing` (the next pipeline
      step) sees the new topology.
    - The subcatchments parquet is rewritten in-place.

    Pondshed delineation by :func:`post_processing.generate_pondsheds`
    automatically picks up the new assignments because it maps each
    sub's ``id`` to its pond via ``partition_pond_network``'s
    ``node_to_pond``, where pond storages map to themselves.

    Set ``pond_design.sub_reroute_max_distance_m = 0`` to disable.
    """
    max_dist = float(pond_design.sub_reroute_max_distance_m)
    if max_dist <= 0:
        logger.info("reroute_subs_to_isolated_ponds: disabled (max_distance_m=0).")
        return graph

    ponds: set[Any] = {n for n, d in graph.nodes(data=True) if d.get("node_type") == "water_body"}
    if not ponds:
        return graph

    subs_path = addresses.model_paths.subcatchments
    if not subs_path.exists():
        logger.info("reroute_subs_to_isolated_ponds: no subcatchments file; skipping.")
        return graph
    subs_gdf = gpd.read_parquet(subs_path)
    if subs_gdf.empty:
        return graph

    # Delineation-consistent partition (storage mode, piped-only keep set,
    # matches generate_pondsheds): a sub counts as "already captured" only
    # when its runoff actually transits a pond storage, not when it merely
    # shares a downstream junction with one.
    keep = frozenset({"pipe", "orifice", "weir", "pond_inflow", "pond_outflow", "outfall"})
    _pipe_graph, _pond_outlets, _lake_outlets, node_to_pond = partition_pond_network(
        graph, addresses, keep_edge_types=keep, pond_sink_mode="storage"
    )

    # Coordinates of every pond.  A sub is only rerouted to its nearest
    # pond *overall*, never pulled overland past a closer pond to a
    # farther one, which is what produced spatially-disconnected
    # pondsheds before the nearest-pond rule.
    all_pond_coords = {
        pid: (float(graph.nodes[pid]["x"]), float(graph.nodes[pid]["y"])) for pid in ponds
    }
    # Per-pond catchment cap (m²), FDOT 1-inch water-quality rule scaled
    # by the multiplier.  Falls back to a generous default if the pond's
    # design volume attribute is missing (shouldn't happen post-
    # insert_pond_nodes but defensive).
    wq_depth_m = 0.0254  # 1 inch
    runoff_coeff = 0.5  # Florida residential mix
    cap_mult = float(pond_design.sub_reroute_capacity_multiplier)
    pond_cap_m2: dict[Any, float] = {}
    for pid in ponds:
        vol_m3 = float(graph.nodes[pid].get("wb_volume_m3", 0.0) or 0.0)
        pond_cap_m2[pid] = cap_mult * vol_m3 / (wq_depth_m * runoff_coeff)

    # Debit each pond's cap by the catchment its storage-mode pondshed
    # already captures through the pipe network, so overland capture tops
    # up detention instead of stacking on top of pipe-fed catchment.
    captured_m2: dict[Any, float] = dict.fromkeys(ponds, 0.0)
    for _, sub_row in subs_gdf.iterrows():
        try:
            pid = node_to_pond.get(int(sub_row["id"]))
        except (TypeError, ValueError):
            continue
        if pid in captured_m2:
            captured_m2[pid] += float(sub_row["area"])

    # River centerlines: a sub whose nearest receiving water is the canal
    # itself must not be pulled overland into a pondshed, that is what
    # painted river corridors as pondsheds and created cross-river
    # MultiPolygon fragments (if the straight line from a sub to a pond
    # crosses the canal, the canal is by construction the closer water).
    river_geoms = [
        d["geometry"]
        for _, _, d in graph.edges(data=True)
        if d.get("edge_type") == "river" and d.get("geometry") is not None
    ]
    river_tree = shapely.STRtree(river_geoms) if river_geoms else None

    # First pass: enumerate candidates (sub_id, target_pond, distance_sq,
    # area).  Sort by distance ascending so each pond fills up with its
    # closest subs first; per-pond cumulative area is enforced in pass 2.
    max_dist_sq = max_dist * max_dist
    candidates: list[tuple[int, Any, float, float]] = []
    skipped_already_in_pondshed = 0
    skipped_too_far = 0
    skipped_river_closer = 0
    for _, sub_row in subs_gdf.iterrows():
        sub_id_raw = sub_row["id"]
        try:
            sub_id = int(sub_id_raw)
        except (TypeError, ValueError):
            continue
        if sub_id in node_to_pond:
            skipped_already_in_pondshed += 1
            continue
        centroid = sub_row.geometry.centroid
        cx, cy = float(centroid.x), float(centroid.y)
        # Nearest pond among ALL ponds: each orphan sub goes to its truly
        # nearest pond (capacity permitting), never pulled overland past a
        # closer pond to a farther one.
        best_pond: Any = None
        best_d_sq = float("inf")
        for pid, (px, py) in all_pond_coords.items():
            d_sq = (px - cx) ** 2 + (py - cy) ** 2
            if d_sq < best_d_sq:
                best_d_sq = d_sq
                best_pond = pid
        if best_pond is None:
            continue
        if best_d_sq > max_dist_sq:
            skipped_too_far += 1
            continue
        if river_tree is not None:
            # River-frontage band: a sub within a street-block of the
            # canal (and nearer to it than to the pond) drains to the
            # canal directly, these are the corridor subs that painted
            # the rivers as pondsheds.  Mid-block subs farther from the
            # water stay with their subdivision pond even when the canal
            # is geometrically closer.
            d_river = centroid.distance(river_geoms[int(river_tree.nearest(centroid))])
            if d_river <= 60.0 and d_river * d_river < best_d_sq:
                skipped_river_closer += 1
                continue
            # Cross-river guard: capture never reaches across a canal
            # (this is what produced MultiPolygon pondshed fragments on
            # the far bank).
            px, py = all_pond_coords[best_pond]
            seg = shapely.LineString([(cx, cy), (px, py)])
            if any(seg.crosses(river_geoms[int(h)]) for h in river_tree.query(seg)):
                skipped_river_closer += 1
                continue
        candidates.append((sub_id, best_pond, best_d_sq, float(sub_row["area"])))

    candidates.sort(key=lambda c: c[2])

    pond_filled_m2: dict[Any, float] = {pid: captured_m2[pid] for pid in ponds}
    reroutes: dict[int, Any] = {}
    skipped_capacity = 0
    rerouted_m2 = 0.0
    for sub_id, pond_id, _d_sq, area_m2 in candidates:
        cap = pond_cap_m2.get(pond_id, 0.0)
        if cap > 0 and pond_filled_m2[pond_id] + area_m2 > cap:
            skipped_capacity += 1
            continue
        reroutes[sub_id] = pond_id
        pond_filled_m2[pond_id] += area_m2
        rerouted_m2 += area_m2

    if not reroutes:
        logger.info(
            f"reroute_subs_to_isolated_ponds: {len(ponds)} pond(s) but no "
            f"eligible subs to reroute "
            f"(skipped {skipped_already_in_pondshed} already transiting a pond, "
            f"{skipped_too_far} beyond {max_dist:.0f} m, "
            f"{skipped_river_closer} nearer a river, "
            f"{skipped_capacity} over pond capacity)."
        )
        return graph

    graph = graph.copy()

    # Transfer contributing_area from the original outlet node to the pond
    # storage so resize_street_pipes_for_pond_routing sees the new topology.
    for orig_id, pond_id in reroutes.items():
        orig_data = graph.nodes.get(orig_id, {})
        transferred = float(orig_data.get("contributing_area", 0.0) or 0.0)
        if transferred > 0:
            current = float(graph.nodes[pond_id].get("contributing_area", 0.0) or 0.0)
            graph.nodes[pond_id]["contributing_area"] = current + transferred
            graph.nodes[orig_id]["contributing_area"] = 0.0

    # Apply reroutes to subs.parquet.
    subs_gdf = subs_gdf.copy()
    subs_gdf["id"] = subs_gdf["id"].map(lambda v: reroutes.get(int(v), v)).astype("int64")  # pyright: ignore[reportArgumentType]
    subs_gdf.to_parquet(subs_path)

    n_ponds_helped = len(set(reroutes.values()))
    logger.info(
        f"reroute_subs_to_isolated_ponds: rerouted {len(reroutes)} orphan sub(s) "
        f"({rerouted_m2 / 1e4:.1f} ha) to {n_ponds_helped}/{len(ponds)} "
        f"pond(s) (capacity-debited cap {cap_mult:.1f}x FDOT 1-inch, "
        f"within {max_dist:.0f} m).  Skipped {skipped_capacity} over capacity, "
        f"{skipped_river_closer} nearer a river, "
        f"{skipped_already_in_pondshed} already transiting a pond."
    )
    return graph


def _enclosed_gap_reroutes(
    subs_gdf: gpd.GeoDataFrame, pond_ids: set[Any], close: float
) -> dict[int, Any]:
    """Map enclosed outfall-bound sub ids to the pond whose gap they sit in.

    Expects ``_sid`` (int sub id) and ``_pp`` (assigned pond, NaN when
    outfall-bound) columns on *subs_gdf*.
    """
    centroids = subs_gdf.geometry.centroid
    outfall_bound = subs_gdf["_pp"].isna()

    reroutes: dict[int, Any] = {}
    for pid in pond_ids:
        psub = subs_gdf[subs_gdf["_pp"] == pid]
        if len(psub) < 2:
            continue
        shed = shapely.union_all(psub.geometry.to_numpy())
        if shed.geom_type != "MultiPolygon":
            continue
        gap_region = shed.buffer(close).buffer(-close).difference(shed)
        if gap_region.is_empty:
            continue
        in_gap = outfall_bound & centroids.within(gap_region)
        for sid in subs_gdf.loc[in_gap, "_sid"]:
            reroutes.setdefault(int(sid), pid)
    return reroutes


def reroute_enclosed_gap_subs(
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    pond_design: PondDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Reroute outfall-bound subcatchments enclosed by a pond's catchment.

    After ``route_pipes_into_ponds`` / ``reroute_subs_to_isolated_ponds`` a
    pond's piped catchment can be a *disconnected* MultiPolygon: the
    shortest-path storm network routes a strip of subs PAST the pond to a
    distant outfall, geographically splitting the pond's captured catchment
    into pieces.  Where such outfall-bound subs are spatially **enclosed** by
    the pond's own catchment (they fall in the gap between the pond's pieces),
    they belong to the pond's natural drainage neighborhood, the long route
    to a distant outfall is a shortest-path artifact, not a real divide.

    For each pond whose catchment is a MultiPolygon, the gap region is the
    morphological close ``buffer(+R).buffer(-R)`` minus the catchment itself
    (``R = pond_design.enclosed_gap_close_m``).  Every subcatchment whose
    centroid lies in that gap region **and** currently drains to a SWMM
    outfall (not into any pond) has its ``Outlet`` rewritten to the pond
    storage, so the pond captures its contiguous catchment.  A conservative
    ``R`` keeps this to truly-enclosed gaps; wider outfall corridors (only
    partly bounded by the pond) are left alone to avoid the over-reach that
    re-fragments neighboring pondsheds.  ``enclosed_gap_close_m = 0`` disables
    the step.
    """
    close = float(pond_design.enclosed_gap_close_m)
    if close <= 0:
        logger.info("reroute_enclosed_gap_subs: disabled (enclosed_gap_close_m=0).")
        return graph

    subs_path = addresses.model_paths.subcatchments
    if not subs_path.exists():
        return graph
    subs_gdf = gpd.read_parquet(subs_path)
    if subs_gdf.empty:
        return graph
    crs = graph.graph.get("crs")
    if crs and subs_gdf.crs and str(subs_gdf.crs) != str(crs):
        subs_gdf = subs_gdf.to_crs(crs)

    # Delineation-consistent partition (storage mode, piped-only keep set,
    # matches generate_pondsheds, so the gaps we close are the ones that show
    # up as MultiPolygon pondsheds).
    keep = frozenset({"pipe", "orifice", "weir", "pond_inflow", "pond_outflow", "outfall"})
    _pg, _po, _lo, node_to_pond = partition_pond_network(
        graph, addresses, keep_edge_types=keep, pond_sink_mode="storage"
    )
    pond_ids = {n for n, d in graph.nodes(data=True) if d.get("node_type") == "water_body"}

    subs_gdf["_sid"] = subs_gdf["id"].astype("int64")
    subs_gdf["_pp"] = subs_gdf["_sid"].map(node_to_pond)

    reroutes = _enclosed_gap_reroutes(subs_gdf, pond_ids, close)
    if not reroutes:
        logger.info("reroute_enclosed_gap_subs: no enclosed outfall-bound subs found.")
        return graph

    graph = graph.copy()
    # Transfer contributing_area from each rerouted sub's original outlet to
    # the pond storage so resize_pond_orifices sees the added inflow.
    for orig_id, pid in reroutes.items():
        ca = float(graph.nodes.get(orig_id, {}).get("contributing_area", 0.0) or 0.0)
        if ca > 0 and pid in graph.nodes:
            graph.nodes[pid]["contributing_area"] = (
                float(graph.nodes[pid].get("contributing_area", 0.0) or 0.0) + ca
            )
            graph.nodes[orig_id]["contributing_area"] = 0.0

    subs_gdf["id"] = subs_gdf["_sid"].map(lambda v: reroutes.get(int(v), v)).astype("int64")  # pyright: ignore[reportArgumentType]
    subs_gdf = subs_gdf.drop(columns=["_sid", "_pp"])
    subs_gdf.to_parquet(subs_path)
    logger.info(
        f"reroute_enclosed_gap_subs: rerouted {len(reroutes)} enclosed outfall-bound "
        f"sub(s) into {len(set(reroutes.values()))} pond(s) (close {close:.0f} m) to "
        "merge split pondsheds."
    )
    return graph


_ORIFICE_PIPE_MANNINGS_N = 0.012  # concrete-lined circular pipe, same as pipe_by_pipe


def _full_circular_pipe_capacity_m3s(diam: float, slope: float) -> float:
    """Manning's full-pipe Q for a circular concrete pipe (m^3/s)."""
    if diam <= 0 or slope <= 0:
        return 0.0
    A = math.pi * diam**2 / 4.0
    R = diam / 4.0
    v = (slope**0.5) * (R ** (2 / 3)) / _ORIFICE_PIPE_MANNINGS_N
    return v * A


def _downstream_street_pipe_capacity(graph: nx.MultiDiGraph[Any], downstream_node: Any) -> float:
    """Full-flow capacity of the smallest-capacity street pipe leaving ``downstream_node``.

    Returns 0 if no street pipe exits the node (the pond drains to a
    terminal / outfall-only node, so the orifice is not capacity-limited
    by a pipe).
    """
    candidates: list[float] = []
    for _, v, k in graph.out_edges(downstream_node, keys=True):
        d = graph.edges[downstream_node, v, k]
        if d.get("edge_type") != "pipe":
            continue
        diam = d.get("diameter")
        length = float(d.get("length", 0) or 0)
        if diam is None or length <= 0:
            continue
        u_cfe = graph.nodes[downstream_node].get("chamber_floor_elevation")
        v_cfe = graph.nodes[v].get("chamber_floor_elevation")
        if u_cfe is None or v_cfe is None:
            continue
        in_off = float(d.get("in_offset", 0) or 0)
        out_off = float(d.get("out_offset", 0) or 0)
        slope = ((float(u_cfe) + in_off) - (float(v_cfe) + out_off)) / length
        slope = max(slope, 0.001)
        candidates.append(_full_circular_pipe_capacity_m3s(float(diam), slope))
    if not candidates:
        return 0.0
    return min(candidates)


def _pipe_cap_downstream_of_orifice(graph: nx.MultiDiGraph[Any], outlet_junction_id: Any) -> float:
    """Return the full-flow capacity of the first street pipe downstream of the outlet junction.

    Walks outlet_junction -> (pond_outflow edge) -> downstream_node -> (street edge).
    The pond_outflow conduit itself is RECT_OPEN and always oversized, so the
    binding hydraulic constraint is the first street pipe at the network
    re-entry point.  Returns 0 if no such pipe exists (pond drains directly
    to a terminal / outfall).
    """
    caps: list[float] = []
    for _, ds_node, k in graph.out_edges(outlet_junction_id, keys=True):
        e = graph.edges[outlet_junction_id, ds_node, k]
        if e.get("edge_type") != "pond_outflow":
            continue
        cap = _downstream_street_pipe_capacity(graph, ds_node)
        if cap > 0:
            caps.append(cap)
    if not caps:
        return 0.0
    return max(caps)


def resize_pond_orifices(  # noqa: C901, PLR0912, PLR0915 - FDOT orifice re-sizing ladder (peak inflow, pipe cap, standard sizes) kept as one auditable routine
    graph: nx.MultiDiGraph[Any],
    pond_design: PondDesign,
    hydraulic_design: HydraulicDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Bump pond orifice diameters to match design inflow and pipe capacity.

    The Opti-CMAC orifice table that ``finalize_pond_outlets`` applies is
    calibrated for the *controllable treatment volume* of a subdivision
    detention pond (< 2 ha, < 2 m deep).  When a pond's subcatchment is
    larger than the calibration envelope, or when ``finalize_pond_outlets``
    reroutes the pond to a substantially deeper anchor, the Opti-CMAC
    diameter is undersized for the pond's actual peak inflow, the pond
    fills faster than the orifice can drain, overtops, and contributes to
    the simulated flooding loss even after ``resize_street_pipes_for_pond_routing``
    has made the downstream pipes large enough.

    This step resizes orifices with two guardrails:

    - **Design criterion cap**, diameter never exceeds
      ``pond_design.max_orifice_diameter_m`` (default 0.9144 m = 36 in,
      FDOT / SJRWMD subdivision-class limit).  Standard sizes (6 / 8 / 12
      / 18 / 24 / 30 / 36 / 48 in) come from
      ``pond_design.orifice_standard_diameters_m``.
    - **Hydraulic feasibility**, the target orifice flow is capped at
      ``orifice_sizing_pipe_fraction`` x Q_full of the first downstream
      street pipe, so the new orifice does NOT move the hydraulic
      bottleneck downstream to the pipe junction and create a new
      junction flood.

    Target orifice Q is
    ``min(Q_peak_inflow * peak_margin, Q_pipe_full * pipe_fraction)``,
    where ``Q_peak_inflow`` comes from re-accumulating cumulative area
    through the current topology x design precipitation (rational method).
    Orifice area is then back-solved from the sharp-edged free-orifice
    equation ``Q = Cd * A * sqrt(2 g H)`` with H = pond max depth
    (plus closed-basin SurDepth where applicable), and the smallest
    standard diameter that meets or exceeds the requirement is picked.
    The orifice is never shrunk, Opti-CMAC's table stays the lower bound.

    References:
        FDOT Drainage Design Guide Ch. 9 (orifice / riser sizing).
        SJRWMD ERP Applicant's Handbook Vol. II Part X (discharge control).
        Opti-CMAC Controllable Volume Implementation Guidelines.
    """
    graph = graph.copy()

    # Rational-method peak flow at each node, accumulated through the
    # current topology, mirrors pipe_by_pipe / resize_street_pipes Pass 1.
    topological_order = list(nx.topological_sort(graph))
    cumulative_area: dict[Any, float] = {}
    for node in topological_order:
        area = float(graph.nodes[node].get("contributing_area", 0.0) or 0.0)
        for pred in graph.predecessors(node):
            area += cumulative_area.get(pred, 0.0)
        cumulative_area[node] = area
    precip = hydraulic_design.precipitation  # m/hr
    peak_inflow = {n: cumulative_area[n] * precip / 3600.0 for n in graph}

    cd = pond_design.orifice_cd
    peak_margin = pond_design.orifice_sizing_peak_margin
    pipe_fraction = pond_design.orifice_sizing_pipe_fraction
    max_diam = pond_design.max_orifice_diameter_m
    std_diams = sorted(pond_design.orifice_standard_diameters_m)

    resized = 0
    details: list[tuple[Any, float, float, float, float]] = []
    for u, v, _k, d in list(graph.edges(data=True, keys=True)):
        if d.get("edge_type") != "orifice":
            continue
        storage_id = u
        outlet_junction_id = v
        storage_data = graph.nodes[storage_id]
        if storage_data.get("node_type") != "water_body":
            continue
        current_diam = float(d.get("orifice_diam_m", 0) or 0)
        if current_diam <= 0:
            continue

        # Effective design head = pond depth (+ closed-basin SurDepth).
        max_depth = float(storage_data.get("wb_max_depth", pond_design.max_depth_m))
        if storage_data.get("wb_closed_basin"):
            max_depth += float(pond_design.closed_basin_sur_depth_m)
        head = max(max_depth, 0.1)

        # Pond peak inflow: rational-method design flow at the storage.
        q_peak = peak_inflow.get(storage_id, 0.0) * peak_margin

        # Hydraulic-feasibility ceiling = first-downstream-street-pipe full flow.
        # Walk: storage -> orifice -> outlet_junction -> pond_outflow ->
        # downstream_node -> street pipe.  If no street pipe downstream
        # (terminal / direct outfall), no pipe cap applies.
        pipe_cap = _pipe_cap_downstream_of_orifice(graph, outlet_junction_id)
        q_target = min(q_peak, pipe_cap * pipe_fraction) if pipe_cap > 0 else q_peak

        # Current orifice Q at design head, for comparison.
        current_a = math.pi * current_diam**2 / 4.0
        current_q = cd * current_a * math.sqrt(2.0 * 9.81 * head)
        if q_target <= current_q:
            # Already sufficient, Opti-CMAC's bin is enough.
            continue

        # Back-solve required area, then diameter.
        req_a = q_target / (cd * math.sqrt(2.0 * 9.81 * head))
        req_diam = math.sqrt(4.0 * req_a / math.pi)
        req_diam = min(req_diam, max_diam)

        # Round UP to next standard diameter that meets the requirement,
        # but never shrink below the current Opti-CMAC diameter.
        new_diam = current_diam
        for cand in std_diams:
            if cand <= current_diam:
                continue
            if cand > max_diam + 1e-9:
                break
            if cand >= req_diam:
                new_diam = cand
                break
        else:
            # Cap at the largest-at-or-below max_diam.
            for cand in reversed(std_diams):
                if cand <= max_diam + 1e-9 and cand > current_diam:
                    new_diam = cand
                    break

        if new_diam > current_diam + 1e-9:
            d["orifice_diam_m"] = new_diam
            storage_data["wb_orifice_diam_m"] = new_diam
            resized += 1
            details.append((storage_id, current_diam, new_diam, q_target, pipe_cap))

    if resized:
        logger.info(
            f"resize_pond_orifices: bumped {resized} pond orifice(s) to "
            "match peak inflow + downstream pipe capacity."
        )
        for sid, old, new, qt, pc in details[:10]:
            logger.info(
                f"  pond {sid}: orifice {old * 39.37:.0f} in -> "
                f"{new * 39.37:.0f} in  "
                f"(Q_target={qt * 1000:.0f} L/s, pipe_cap={pc * 1000:.0f} L/s)"
            )
    else:
        logger.info("resize_pond_orifices: Opti-CMAC sizing already sufficient; no bumps.")

    # Every pond keeps its emergency-spillway weir (conventional detention
    # design).  Dropping weirs whose orifice passes the rational design peak
    # was tested and rejected: the rational peak (duration-averaged) is below
    # the dynamic storm peak, so those ponds still overtop in simulation and
    # without a weir the excess is booked as flood loss instead of routed
    # downstream.  ``Surcharge=YES`` handles the parallel orifice+weir
    # DYNWAVE regime-flutter.  Verified on the test catchment: keeping weirs left
    # flood unchanged while improving routing continuity (+0.95% -> +0.50%)
    # and convergence (5.97% -> 5.71%).
    return graph
