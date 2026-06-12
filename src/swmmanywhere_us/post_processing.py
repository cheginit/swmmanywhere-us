"""Post processing module for SWMManywhere.

A module containing functions to format and write processed data into SWMM .inp
files.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from swmmanywhere_us.logging import logger

from .swmm_defaults import SWMMOptions
from .swmm_inp_generator import SwmmInputGenerator

if TYPE_CHECKING:
    import networkx as nx

    from swmmanywhere_us.filepaths import FilePaths
    from swmmanywhere_us.parameters import PondDesign


def _read_rain_dat_file(dat_file: str | Path | None, base_dir: Path) -> tuple[Path, pd.DataFrame]:
    """Read the rain data from a DAT file.

    Args:
        dat_file (Path): The path to the DAT file. If not provided,
        a storm event will be created with the following parameters:

            ;File: "storm.dat"
            1   2000 01 01 00 00    0.0
            1   2000 01 01 00 05    28
            1   2000 01 01 00 10    32
            1   2000 01 01 00 15    3
        base_dir (Path): The base directory to copy the DAT file to.
        This option is only used if ``dat_file`` is not provided.


    Returns:
        pd.DataFrame: The rain data as a DataFrame.
    """
    if dat_file is None:
        dat_file = base_dir / "storm.dat"
        file_content = f""";File: {dat_file}
1   2000 01 01 00 00    0.0
1   2000 01 01 00 05    28
1   2000 01 01 00 10    32
1   2000 01 01 00 15    3
"""
        dat_file.write_text(file_content)
        return dat_file, pd.DataFrame(
            {
                "rain_gage": [1, 1, 1, 1],
                "date": pd.to_datetime(
                    ["2000-01-01 00:00", "2000-01-01 00:05", "2000-01-01 00:10", "2000-01-01 00:15"]
                ),
                "value": [0.0, 28, 32, 3],
            }
        )

    try:
        df = pd.read_csv(
            dat_file,
            sep=r"\s+",
            comment=";",
            names=["rain_gage", "year", "month", "day", "hour", "minute", "value"],
        )
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        msg = f"Input is not a correct DAT file: {exc}"
        raise ValueError(msg) from exc

    df["date"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]])
    df = df[["rain_gage", "date", "value"]].copy()
    return Path(dat_file), df


def _extend_stage_area_curve(
    curve: list[Any], new_top_depth: float, side_slope_pct: float
) -> list[Any]:
    """Extend a stage-area curve upward to ``new_top_depth``.

    Appends points above the curve's current top, expanding the surface area
    via the pond side slope (the bowl keeps widening above the treatment pool,
    same geometry as :func:`water_bodies._stage_area_curve`).  Used to give the
    pond flood-detention storage between the permanent-pool top and the
    embankment crest.  Returns the curve unchanged if ``new_top_depth`` is not
    above the current top.
    """
    top_depth, top_area = float(curve[-1][0]), float(curve[-1][1])
    if new_top_depth <= top_depth:
        return curve
    hv = 1.0 / (side_slope_pct / 100.0)  # horizontal run per unit rise
    half_l_top = float(np.sqrt(max(top_area, 0.0)))
    extended = list(curve)
    n_extra = 4
    for i in range(1, n_extra + 1):
        d = top_depth + (new_top_depth - top_depth) * i / n_extra
        half_l = half_l_top + 2.0 * hv * (d - top_depth)
        extended.append((d, half_l * half_l))
    return extended


def _drop_small_detached_parts(geom: Any, min_part_m2: float) -> Any:
    """Drop negligible detached parts from a (Multi)Polygon pondshed.

    Keeps the largest connected component plus any other component at least
    ``min_part_m2`` in area; drops the rest.  This trims the trivial slivers a
    pondshed picks up when the storm network pipes a single far subcatchment to
    the pond via a long winding path (a few hundred m² stranded hundreds of m
    away) — visual noise, not a meaningful split.  Genuine large detached
    pieces (distant clusters fed by a trunk) are preserved.  Polygons and
    geometries with one component are returned unchanged.
    """
    if geom.geom_type != "MultiPolygon":
        return geom
    parts = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
    kept = [parts[0]] + [p for p in parts[1:] if p.area >= min_part_m2]
    return kept[0] if len(kept) == 1 else shapely.MultiPolygon(kept)


def synthetic_write(  # noqa: C901, PLR0912, PLR0915 - end-to-end graph -> SWMM .inp translation with many branches per SWMM element type
    addresses: FilePaths,
    rain_dat_path: str | Path | None = None,
    rain_dat_unit: Literal["IN", "MM"] = "MM",
    inp_options: dict[str, Any] | None = None,
    pond_design: PondDesign | None = None,
):
    """Load synthetic data and write to SWMM input file.

    Produces a SWMM model with both belowground pipes (street edges as
    CIRCULAR conduits) and surface drainage (river edges as RECT_OPEN
    conduits, water bodies as STORAGE nodes).  Outfall edges become
    direct conduit connections between the pipe and channel networks.

    Args:
        addresses (FilePaths): A dictionary of file paths.
        rain_dat_path (str | Path | None): Optional path to the rain data file.
        rain_dat_unit: Rain data units, either "IN" or "MM".
        inp_options (dict[str, Any] | None): Optional SWMM options overrides.
        pond_design: Pond-design parameters.  Used to apply closed-basin
            surface-ponding depth + Green-Ampt seepage to ponds that
            ``finalize_pond_outlets`` marked with ``wb_closed_basin=True``.
    """
    if rain_dat_unit not in ["IN", "MM"]:
        msg = "Invalid rain_dat_unit. Must be 'IN' or 'MM'."
        raise ValueError(msg)

    from swmmanywhere_us.graph_utilities import load_graph

    nodes = gpd.read_file(addresses.model_paths.nodes)
    edges = gpd.read_file(addresses.model_paths.edges)
    subs = gpd.read_parquet(addresses.model_paths.subcatchments)

    # Load graph for reliable access to complex node attributes (e.g., stage-area curves)
    # that get mangled during GeoJSON round-trip serialization.
    full_graph = load_graph(addresses.model_paths.graph)

    rain_dat, rain = _read_rain_dat_file(rain_dat_path, addresses.model_paths.model)

    gage_id = rain["rain_gage"].unique()
    if len(gage_id) != 1:
        msg = "There must be exactly one rain gage."
        raise ValueError(msg)
    gage_id = str(gage_id[0])

    # --- Node preparation ---
    node_cols = ["id", "x", "y", "chamber_floor_elevation", "surface_elevation"]
    optional_cols = ["node_type", "wb_area_m2", "wb_max_depth"]
    node_cols.extend(col for col in optional_cols if col in nodes.columns)
    nodes = nodes[node_cols].copy()
    nodes["id"] = nodes["id"].astype(str)

    # Manhole MaxDepth: use the actual surface-to-invert burial depth
    # per node (matches the calibrated UWO Swiss sewer reference where
    # junctions have MaxDepth 2-8 m distributed; p50 = 2.5 m, p99 = 5.4 m).
    # Previously we clipped to a 3 m floor, which under-represents
    # deep-buried manholes on pipes that pipe_by_pipe pushed below
    # normal burial for slope / max-depth enforcement.  A 3 m FLOOR is
    # still applied only when the computed depth is implausibly small
    # (< 0.5 m) or NaN, to give shallow-trench pipes a reasonable
    # surcharge buffer rather than zero.
    # NOTE: water-body (STORAGE) nodes have their MaxDepth resolved from
    # ``wb_max_depth`` in the node loop below — this manhole default is
    # not applied to ponds.
    raw_depth = nodes.surface_elevation - nodes.chamber_floor_elevation
    max_depth = raw_depth.where(raw_depth >= 0.5, 3.0).fillna(3.0)

    # Physical constraint: a manhole MUST be at least as deep as the
    # largest pipe passing through it — otherwise the pipe's crown sits
    # above ground and water runs out the top at zero surcharge.  Collect
    # the max adjacent pipe diameter (CIRCULAR) or channel depth
    # (RECT_OPEN) per node from the raw edges frame BEFORE it's trimmed.
    # On the test catchment this pattern (burial < pipe D) caused 65 % of street
    # junctions to be shallower than their own pipes and dominated
    # DYNWAVE non-convergence on 5 of the 5 most-frequent-nonconverging
    # nodes.  No tunable parameter: the constraint uses the pipe's own
    # geometry already assigned by ``pipe_by_pipe`` / ``assign_channel_geometry``.
    _max_adj_pipe_d: dict[str, float] = {}
    for _, e in edges.iterrows():
        d = float(e.get("diameter", 0) or 0)
        cd = float(e.get("channel_depth", 0) or 0)
        pipe_height = max(d, cd)
        if pipe_height <= 0:
            continue
        for nid in (str(e["u"]), str(e["v"])):
            prev = _max_adj_pipe_d.get(nid, 0.0)
            if pipe_height > prev:
                _max_adj_pipe_d[nid] = pipe_height
    pipe_floor = nodes["id"].map(_max_adj_pipe_d).fillna(0.0)
    nodes["max_depth"] = np.maximum(max_depth, pipe_floor)

    # --- Edge preparation ---
    edge_cols = ["id", "u", "v", "length"]
    # Pond outlet attributes (orifice_*, weir_*) are kept so that per-pond
    # diameters / crest heights / weir lengths computed in
    # :func:`finalize_pond_outlets` survive the GeoJSON round-trip into the
    # XSECTIONS / ORIFICES / WEIRS blocks.  Without them every pond falls
    # back to the fallback defaults below and outlet sizing is identical
    # across ponds (hydraulically meaningless).
    optional_edge_cols = [
        "edge_type",
        "diameter",
        "roughness",
        "channel_width",
        "channel_depth",
        "in_offset",
        "out_offset",
        "orifice_type",
        "orifice_diam_m",
        "orifice_cd",
        "orifice_offset",
        "flap_gate",
        "weir_type",
        "weir_crest_m",
        "weir_length_m",
        "weir_cd",
    ]
    edge_cols.extend(col for col in optional_edge_cols if col in edges.columns)
    edges = edges[edge_cols].copy()
    edges["u"] = edges["u"].astype(str)
    edges["v"] = edges["v"].astype(str)
    edges["id"] = edges["id"].astype(str)

    if "edge_type" not in edges.columns:
        edges["edge_type"] = "pipe"
    if "diameter" not in edges.columns:
        edges["diameter"] = 0.3
    if "roughness" not in edges.columns:
        edges["roughness"] = 0.01

    # --- Classify nodes by role ---
    # Determine which nodes are river-only, pipe-only, water body, etc.
    edge_types_by_node: dict[str, set[str]] = {}
    for _, row in edges.iterrows():
        for n in (row["u"], row["v"]):
            edge_types_by_node.setdefault(n, set()).add(row.get("edge_type", "pipe"))

    river_node_ids = {
        n for n, types in edge_types_by_node.items() if "river" in types and "pipe" not in types
    }
    wb_node_ids = set()
    outlet_junction_ids = set()
    dummy_river_ids: set[str] = set()
    water_body_outfall_ids: set[str] = set()
    river_outfall_ids: set[str] = set()
    if "node_type" in nodes.columns:
        wb_node_ids = set(nodes.loc[nodes["node_type"] == "water_body", "id"])
        outlet_junction_ids = set(nodes.loc[nodes["node_type"] == "outlet_junction", "id"])
        # dummy_river, river_outfall, and water_body_outfall nodes are all
        # synthetic outfall surrogates created by :func:`identify_outfalls`.
        # dummy_river marks a subgraph with no natural river terminus (sink at
        # the lowest node); river_outfall marks a discharge snapped onto a
        # river centerline; water_body_outfall marks a discharge into a
        # water-body polygon.  All are connected only by an ``outfall`` edge
        # (not a ``river`` edge), so they miss the ``river_node_ids`` set
        # below and would otherwise land in JUNCTIONS — where SWMM treats
        # them as dead-end sinks that pond water up to MaxDepth.  Route all
        # to OUTFALLS directly.  They differ only in invert handling below:
        # dummy_river nodes have no DEM elevation and get an invert derived
        # from the upstream pipe, whereas river_outfall and
        # water_body_outfall nodes already carry the receiving water's DEM
        # surface elevation (set in identify_outfalls) and keep it.
        dummy_river_ids = set(nodes.loc[nodes["node_type"] == "dummy_river", "id"])
        water_body_outfall_ids = set(nodes.loc[nodes["node_type"] == "water_body_outfall", "id"])
        river_outfall_ids = set(nodes.loc[nodes["node_type"] == "river_outfall", "id"])

    # Terminal river nodes (no outgoing river edges) become SWMM outfalls
    river_edges = edges[edges["edge_type"] == "river"]
    river_upstream = set(river_edges["u"])
    river_downstream = set(river_edges["v"])
    terminal_river_ids = (river_downstream - river_upstream) & river_node_ids
    # If no river edges exist, fall back to pipe terminal nodes
    if not terminal_river_ids:
        all_u = set(edges["u"])
        terminal_river_ids = set(nodes["id"]) - all_u
    # Always include synthetic dummy-river, river, and water-body outfalls.
    terminal_river_ids |= dummy_river_ids
    terminal_river_ids |= water_body_outfall_ids
    terminal_river_ids |= river_outfall_ids

    # Fix D-1: dummy_river inverts get set by ``set_elevation`` as
    # ``surface - 3 m`` (manhole default), regardless of how deep the
    # upstream network pipe actually is.  When the upstream pipe is
    # shallow-buried (common in flat Florida terrain after pipe_by_pipe),
    # this produces an outfall invert 4-7 m below the incoming pipe -- a
    # degenerate drop that the ``enforce_outfall_slope`` step compensates
    # for by stretching the pipe length 100-200 m.  That's a workaround;
    # the proper fix is to anchor the dummy-river invert one-pipe-
    # diameter below its feeding pipe invert (standard receiving-water
    # drop in storm-drain design).  No new parameter: we use the
    # incoming outfall edge's own diameter.
    outfall_edges_to_dummy = edges[
        (edges["edge_type"] == "outfall") & (edges["v"].isin(dummy_river_ids))
    ]
    if not outfall_edges_to_dummy.empty:
        cfe_by_id = dict(zip(nodes["id"], nodes["chamber_floor_elevation"]))
        dummy_cfe_updates: dict[str, float] = {}
        for _, e in outfall_edges_to_dummy.iterrows():
            upstream_cfe = cfe_by_id.get(str(e["u"]))
            if upstream_cfe is None or pd.isna(upstream_cfe):
                continue
            drop = max(float(e.get("diameter", 0) or 0), 0.3)
            dummy_cfe_updates[str(e["v"])] = float(upstream_cfe) - drop
        if dummy_cfe_updates:
            nodes["chamber_floor_elevation"] = nodes.apply(
                lambda r: dummy_cfe_updates.get(r["id"], r["chamber_floor_elevation"]),
                axis=1,
            )
            # Recompute max_depth since we just changed some CFEs.
            raw_depth = nodes.surface_elevation - nodes.chamber_floor_elevation
            max_depth = raw_depth.where(raw_depth >= 0.5, 3.0).fillna(3.0)
            pipe_floor = nodes["id"].map(_max_adj_pipe_d).fillna(0.0)
            nodes["max_depth"] = np.maximum(max_depth, pipe_floor)

    # --- Subcatchments ---
    subs["id"] = subs["id"].astype(str)
    subs = subs.loc[subs.id.isin(nodes.id), ["id", "geometry", "area", "slope", "width", "rc"]]
    # Unique subcatchment name per row.  Many subcatchments can share one
    # Outlet node (every pipe junction keeps its own fine subcatchment, and a
    # pond storage receives dozens of them), so the bare ``{id}-sub`` name
    # would collide in ``subcatchments_dict``/``polygons_dict`` and silently
    # drop all but one.  A per-id counter keeps each sub a distinct SWMM
    # subcatchment draining to the shared Outlet.
    subs["subcatchment"] = subs["id"] + "-" + subs.groupby("id").cumcount().astype(str) + "-sub"
    subs["rain_gage"] = gage_id
    subs["area"] /= 10000  # convert to ha

    # --- Rain frequency ---
    freq_str = pd.infer_freq(rain["date"])
    freq_td = pd.to_timedelta(freq_str) if freq_str is not None else None
    if pd.notna(freq_td):
        freq_seconds = freq_td.total_seconds()
    else:
        diffs = rain["date"].diff().dt.total_seconds().dropna()
        mode_val = diffs.mode()
        freq_seconds = (
            mode_val.iloc[0] if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]) else diffs.min()
        )
    hours, remainder = divmod(freq_seconds, 3600)
    minutes = remainder // 60
    frequency = f"{int(hours):02d}:{int(minutes):02d}"

    raingages_dict = {
        gage_id: {
            "Format": "INTENSITY",
            "Interval": frequency,
            "SCF": 1.0,
            "Source": "FILE",
            "filename": str(rain_dat.resolve()),
            "station_id": gage_id,
            "rain_units": rain_dat_unit,
        }
    }

    # --- Subcatchments dict ---
    subcatchments_dict = {}
    polygons_dict = {}
    cols = ["subcatchment", "id", "rain_gage", "area", "rc", "width", "slope", "geometry"]
    for sub_id, nid, gage, area, rc, width, slope, poly in subs[cols].itertuples(
        index=False, name=None
    ):
        subcatchments_dict[sub_id] = {
            "Raingage": gage,
            "Outlet": nid,
            "Area": area,
            "PercImperv": rc,
            "Width": width,
            "PercSlope": slope,
        }
        if poly.geom_type == "MultiPolygon":
            largest = max(poly.geoms, key=lambda g: g.area)
        else:
            largest = poly
        polygons_dict[sub_id] = shapely.get_coordinates(largest.exterior).tolist()

    # --- Build SWMM node dicts ---
    storage_dict = {}
    junctions_dict = {}
    outfalls_dict = {}
    coordinates_dict = {}
    curves_dict: dict[str, dict[str, Any]] = {}

    for row in nodes.itertuples(index=False):
        nid = str(row.id)
        x, y = row.x, row.y
        cfe_raw = row.chamber_floor_elevation
        # Dummy-river outfalls can have no DEM-derived elevation. Use a
        # sentinel below any plausible receiving-water stage so gravity flow
        # to the outfall is preserved.
        elev = -1.0 if pd.isna(cfe_raw) else float(cfe_raw)  # pyright: ignore[reportArgumentType]
        max_depth = float(row.max_depth)  # pyright: ignore[reportArgumentType]
        coordinates_dict[nid] = {"X": x, "Y": y}

        if nid in terminal_river_ids:
            # Terminal river / dummy-river nodes -> SWMM OUTFALL
            outfalls_dict[nid] = {
                "InvertElev": elev - 1.0,
                "OutfallType": "FREE",
            }
        elif nid in outlet_junction_ids:
            # Pond outlet junctions -> SWMM JUNCTION with manhole-class surge
            # headroom.  MaxDepth = 0 (the prior default) gives the junction
            # zero storage, so any peak inflow beyond instantaneous downstream
            # capacity floods immediately.  This is exactly the failure mode
            # observed on the test catchment, where 4 ponds plus 3
            # street pipes converge: even a one-second imbalance produces
            # multi-thousand-L/s flood spikes at the junction.  A 3 m manhole-
            # equivalent depth (matching what we set for ordinary pipe
            # junctions below) gives DYNWAVE a few seconds of buffering before
            # spilling, which absorbs short transient peaks while still
            # surfacing real persistent overflow as flooding.
            junctions_dict[nid] = {
                "InvertElev": elev,
                "MaxDepth": 3.0,
                "InitDepth": 0,
                "SurchargeDepth": 0,
                "PondedArea": 0,
            }
        elif nid in river_node_ids:
            # River channel nodes -> SWMM JUNCTION (open-top)
            junctions_dict[nid] = {
                "InvertElev": elev,
                "MaxDepth": max(max_depth, 0.5),
                "InitDepth": 0,
                "SurchargeDepth": 0,
                "PondedArea": 0,
            }
        elif nid in wb_node_ids:
            # Water body nodes -> SWMM STORAGE with TABULAR stage-area curve.
            # Read the curve from the graph (not GeoJSON) because nested lists
            # get mangled during GeoJSON round-trip serialization.  ``nid`` is
            # a stringified integer here; the graph keys are plain ints.
            graph_node_data: dict[str, Any] = {}
            with contextlib.suppress(ValueError, KeyError):
                graph_node_data = dict(full_graph.nodes[int(nid)])
            wb_curve = graph_node_data.get("wb_stage_area_curve")
            # The storage's MaxDepth must match the top of its stage-area
            # curve — SWMM treats depths above the last curve point as
            # flooded (area capped at the last Y-value), so a MaxDepth
            # greater than the curve's last X would create "phantom"
            # storage in a band with no physical geometry.  Prefer the
            # curve's top depth, fall back to the pond's designed
            # ``wb_max_depth``, and only then to the 1 m floor.
            wb_max_depth_attr = graph_node_data.get("wb_max_depth")
            if wb_curve and isinstance(wb_curve, list) and len(wb_curve) >= 2:
                curve_top_depth = float(wb_curve[-1][0])
            else:
                curve_top_depth = 0.0
            if wb_max_depth_attr is not None:
                storage_max_depth = max(float(wb_max_depth_attr), curve_top_depth, 1.0)
            elif curve_top_depth > 0:
                storage_max_depth = max(curve_top_depth, 1.0)
            else:
                storage_max_depth = 1.0

            # Flood-detention storage: extend MaxDepth + the stage-area curve
            # up to the pond's overtopping crest (embankment grade just outside
            # the polygon, sampled at insertion).  Conventional detention ponds
            # store water from the invert up to the embankment crest, not just
            # the FDOT treatment pool — without this the pond floods the instant
            # it fills the permanent pool.  ``elev`` is the pond invert.  Capped
            # at +2 m of added depth so a high surrounding grade can't
            # extrapolate a small FDOT pond into an unrealistic basin.
            emb_elev = graph_node_data.get("wb_embankment_elev")
            if (
                pond_design is not None
                and emb_elev is not None
                and not np.isnan(float(emb_elev))
                and wb_curve
                and isinstance(wb_curve, list)
                and len(wb_curve) >= 2
            ):
                emb_depth = min(float(emb_elev) - float(elev), storage_max_depth + 2.0)
                if emb_depth > storage_max_depth + 0.05:
                    wb_curve = _extend_stage_area_curve(
                        wb_curve, emb_depth, float(pond_design.side_slope_pct)
                    )
                    storage_max_depth = emb_depth
                    curve_top_depth = emb_depth

            # Closed-basin adjustment: ponds that finalize_pond_outlets
            # could not drain by gravity get
            #   (a) ``SurDepth`` so overtop water stays on the surface as
            #       virtual storage rather than being lost to SWMM's
            #       flooding-loss term — matches how the calibrated reference
            #       model handles every storage (SurDepth = 99 ft), and
            #   (b) Green-Ampt seepage (Psi/Ksat/IMD) so the pond depletes
            #       by percolation through its bottom at a rate
            #       consistent with Florida retention-pond drawdown
            #       practice (SJRWMD/SFWMD 72-hour recovery).
            is_closed_basin = bool(graph_node_data.get("wb_closed_basin", False))
            if pond_design is not None:
                if is_closed_basin:
                    # Closed basin: virtual surface ponding + exfiltration
                    # (percolation through the pond bottom).  Drains
                    # entirely by seepage since there is no gravity outlet.
                    sur_depth = float(pond_design.closed_basin_sur_depth_m)
                    psi = float(pond_design.closed_basin_psi_mm)
                    ksat = float(pond_design.closed_basin_ksat_mm_hr)
                    imd = float(pond_design.closed_basin_imd)
                else:
                    # Gravity-drained (rerouted / original-anchor) pond: add
                    # the same virtual-surface-pond headroom so that the
                    # design-storm peak does not disappear into SWMM's
                    # flooding-loss term during the brief overtopping
                    # window.  The extra head over the orifice also
                    # increases outlet Q ~ sqrt(H), so the pond recovers
                    # faster once inflow falls.  No seepage tail — these
                    # ponds rely on the gravity outlet, not exfiltration.
                    sur_depth = float(pond_design.open_basin_sur_depth_m)
                    psi = ksat = imd = 0.0
            else:
                sur_depth = 0.0
                psi = ksat = imd = 0.0

            if wb_curve and isinstance(wb_curve, list) and len(wb_curve) >= 2:
                curve_name = f"{nid}_curve"
                curves_dict[curve_name] = {
                    "Type": "STORAGE",
                    "X-Values": [pt[0] for pt in wb_curve],
                    "Y-Values": [pt[1] for pt in wb_curve],
                }
                storage_dict[nid] = {
                    "InvertElev": elev,
                    "MaxDepth": storage_max_depth,
                    "InitDepth": 0,
                    "StorageCurve": "TABULAR",
                    "Coefficient": curve_name,
                    "Exponent": 0,
                    "Constant": 0,
                    "PondedArea": sur_depth,
                    "EvapFrac": 0,
                    "Psi": psi,
                    "Ksat": ksat,
                    "IMD": imd,
                }
            else:
                wb_area = getattr(row, "wb_area_m2", 500.0) if hasattr(row, "wb_area_m2") else 500.0
                storage_dict[nid] = {
                    "InvertElev": elev,
                    "MaxDepth": storage_max_depth,
                    "InitDepth": 0,
                    "StorageCurve": "FUNCTIONAL",
                    "Coefficient": wb_area,
                    "Exponent": 0,
                    "Constant": 0,
                    "PondedArea": sur_depth,
                    "EvapFrac": 0,
                    "Psi": psi,
                    "Ksat": ksat,
                    "IMD": imd,
                }
        else:
            # Pipe network nodes -> SWMM JUNCTION (manhole).  Junctions
            # support MaxDepth-based surcharge and have no numerical issues
            # with zero-volume routing, unlike small STORAGE units.  The
            # reference model also uses junctions for manholes.
            junctions_dict[nid] = {
                "InvertElev": elev,
                "MaxDepth": max(max_depth, 0.5),
                "InitDepth": 0,
                "SurchargeDepth": 0,
                "PondedArea": 0,
            }

    # --- Build node elevation lookup for offset calculations ---
    node_elev: dict[str, float] = {}
    for nid in storage_dict:
        node_elev[nid] = storage_dict[nid]["InvertElev"]
    for nid in junctions_dict:
        node_elev[nid] = junctions_dict[nid]["InvertElev"]
    for nid in outfalls_dict:
        node_elev[nid] = outfalls_dict[nid]["InvertElev"]

    # --- Build SWMM conduit, orifice, weir, and xsection dicts ---
    conduits_dict = {}
    orifices_dict: dict[str, dict[str, Any]] = {}
    weirs_dict: dict[str, dict[str, Any]] = {}
    xsections_dict = {}
    # [LOSSES] flap gates: a pond_outflow conduit is the pond's one-way
    # discharge to the network.  On flat terrain the downstream junction
    # surcharges to the outlet-junction level and reverse-flows back through
    # this near-zero-slope conduit (and DYNWAVE sloshes on it).  A flap gate
    # (standard pond-outfall backflow preventer) clips that reverse flow.
    losses_dict: dict[str, dict[str, Any]] = {}

    # Precompute the largest pipe diameter at each node, over BOTH
    # endpoints of every pipe edge, so an outfall pipe can match (or
    # exceed) the biggest pipe at its upstream node — a sudden taper
    # right before the outfall makes DYNWAVE iterate on the area change.
    # Scanning both endpoints (not just the downstream one) is what
    # catches a synthetic trunk graph-directed *out* of the node: such
    # a trunk still reverse-flows the node's discharge in DYNWAVE, and
    # keying only on the downstream node would leave the outfall straw
    # at the 0.3 m minimum.  Reads only existing diameters; no parameter.
    _max_pipe_d_at_node: dict[str, float] = {}
    # Sum of pipe cross-sectional areas (∝ d²) arriving at each node, keyed by
    # the pipe's DOWNSTREAM end.  An on-line pond intake (a street paired to a
    # pond by online_pond_intake) is a flow SINK: every pipe in its
    # intercepted catchment drains into it, so the single intake conduit must
    # pass the COMBINED inflow, not just the largest feeder — otherwise it
    # bottlenecks and the intake junction surcharges.  The area-equivalent
    # diameter √(Σ dᵢ²) sizes the intake to carry the lot.
    _sum_in_d2_at_node: dict[str, float] = {}
    for _e in edges.itertuples(index=False):
        if getattr(_e, "edge_type", "pipe") != "pipe":
            continue
        d_in = float(getattr(_e, "diameter", 0) or 0)
        if d_in <= 0:
            continue
        for _n in (str(_e.u), str(_e.v)):
            if d_in > _max_pipe_d_at_node.get(_n, 0.0):
                _max_pipe_d_at_node[_n] = d_in
        _sum_in_d2_at_node[str(_e.v)] = _sum_in_d2_at_node.get(str(_e.v), 0.0) + d_in * d_in

    for row in edges.itertuples(index=False):
        eid = str(row.id)
        u, v = str(row.u), str(row.v)
        length = float(row.length)  # pyright: ignore[reportArgumentType]
        edge_type = getattr(row, "edge_type", "pipe")

        # Skip edges where both endpoints are missing from nodes
        if u not in node_elev and v not in node_elev:
            continue
        # Add missing terminal nodes as outfalls
        for n in (u, v):
            if n not in node_elev:
                node_elev[n] = 0.0
                outfalls_dict[n] = {"InvertElev": 0.0, "OutfallType": "FREE"}
                coordinates_dict[n] = {"X": 0, "Y": 0}

        # Use pipe-level offsets from assign_channel_geometry / pipe_by_pipe
        # when available (these enforce positive slope on adverse terrain).
        # Otherwise default to 0 (pipe connects at node InvertElev).
        in_offset = float(getattr(row, "in_offset", 0.0) or 0.0)
        out_offset = float(getattr(row, "out_offset", 0.0) or 0.0)
        # NaN offsets from edges that didn't go through pipe design
        if np.isnan(in_offset):
            in_offset = 0.0
        if np.isnan(out_offset):
            out_offset = 0.0

        roughness = getattr(row, "roughness", 0.01)

        if edge_type == "orifice":
            # Orifice link — either a pond primary outlet (SIDE, one-way
            # flap gate) or a dual-drainage catchbasin grate (BOTTOM, two-
            # way).  Type and flap gate are read from the edge so both
            # kinds emit correctly; pond orifices that set neither fall
            # back to the SIDE / one-way default below.
            orifice_diam = (
                getattr(row, "orifice_diam_m", 0.3) if hasattr(row, "orifice_diam_m") else 0.3
            )
            orifice_cd = getattr(row, "orifice_cd", 0.65) if hasattr(row, "orifice_cd") else 0.65
            orifice_offset = (
                getattr(row, "orifice_offset", 0.0) if hasattr(row, "orifice_offset") else 0.0
            )
            o_type = getattr(row, "orifice_type", None) if hasattr(row, "orifice_type") else None
            orifice_type = o_type if isinstance(o_type, str) and o_type else "SIDE"
            o_flap = getattr(row, "flap_gate", None) if hasattr(row, "flap_gate") else None
            # FlapGate=YES makes a pond orifice one-way (pond -> network).
            # SIDE orifices without a flap gate run reversed whenever the
            # downstream junction surcharges above the pond water level
            # (observed on the test catchment at multi-pond junctions).  FDOT / SJRWMD /
            # ASCE MOP 77 detention practice assumes one-way controlled
            # release, so YES is the pond default.  Catchbasin grates set
            # flap_gate="NO" so sewer surcharge can reverse onto the street.
            flap_gate = o_flap if isinstance(o_flap, str) and o_flap else "YES"
            orifices_dict[eid] = {
                "InletNode": u,
                "OutletNode": v,
                "orifice_type": orifice_type,
                "crest_height": orifice_offset,
                "disch_coeff": orifice_cd,
                "flap_gate": flap_gate,
            }
            xsections_dict[eid] = {
                "Shape": "CIRCULAR",
                "Geom1": orifice_diam,
            }
        elif edge_type == "weir":
            # Pond emergency spillway weir (storage -> outlet junction)
            weir_crest = getattr(row, "weir_crest_m", 1.0) if hasattr(row, "weir_crest_m") else 1.0
            weir_length = (
                getattr(row, "weir_length_m", 4.6) if hasattr(row, "weir_length_m") else 4.6
            )
            weir_cd = getattr(row, "weir_cd", 3.0) if hasattr(row, "weir_cd") else 3.0
            weirs_dict[eid] = {
                "InletNode": u,
                "OutletNode": v,
                "WeirType": "TRANSVERSE",
                "CrestHeight": weir_crest,
                "Cd": weir_cd,
                # FlapGate=YES on the emergency spillway weir matches the
                # one-way physical interpretation: water flows OVER the
                # pond embankment crest from pond to network, never the
                # other way (gravity prohibits it).  Without FlapGate=YES
                # SWMM allows reverse flow when the downstream junction
                # surcharges above the weir crest — same mechanism that
                # causes orifice backflow on the test catchment.
                "flap_gate": "YES",
                "EndCon": 0,
                "EndCoeff": 0,
            }
            xsections_dict[eid] = {
                "Shape": "RECT_OPEN",
                "Geom1": weir_crest * 0.1,
                "Geom2": weir_length,
            }
        elif edge_type in {"river", "pond_outflow", "street_channel"}:
            # Open channel conduit
            ch_width = getattr(row, "channel_width", 2.0) if hasattr(row, "channel_width") else 2.0
            ch_depth = getattr(row, "channel_depth", 1.0) if hasattr(row, "channel_depth") else 1.0
            roughness = roughness if roughness > 0.02 else 0.035
            # Physical constraint: an open channel can't be shorter than
            # its own hydraulic depth (it would just be a drop structure).
            # Clamp Length >= channel_depth to prevent degenerate
            # Manning's-n routing in SWMM.
            effective_length = max(length, float(ch_depth), 1.0)

            conduits_dict[eid] = {
                "InletNode": u,
                "OutletNode": v,
                "Length": effective_length,
                "Roughness": roughness,
                "InOffset": in_offset,
                "OutOffset": out_offset,
                "InitFlow": 0,
                "MaxFlow": 0,
            }
            xsections_dict[eid] = {
                "Shape": "RECT_OPEN",
                "Geom1": ch_depth,
                "Geom2": ch_width,
            }
            if edge_type == "pond_outflow":
                # One-way pond -> network outfall: block downstream backwater.
                losses_dict[eid] = {"FlapGate": "YES"}
        else:
            # Pipe conduit (street or outfall connector)
            diam = getattr(row, "diameter", 0.3) if hasattr(row, "diameter") else 0.3
            diam = max(float(diam), 0.15)
            # Fix B: outfall pipes should not taper down from the pipes
            # at their upstream node — abrupt section changes at the
            # outfall point make DYNWAVE iterate on the area mismatch.
            # Bump the outfall to match the largest pipe at its upstream
            # node.  ``_max_pipe_d_at_node`` spans every pipe at the
            # node — incoming feeders and any trunk graph-directed
            # outward — so a reverse-flowing trunk can't push its
            # discharge through a 0.3 m straw.  No new parameter.
            if edge_type == "outfall":
                node_pipe_d = _max_pipe_d_at_node.get(u, 0.0)
                diam = max(diam, node_pipe_d)
                # On-line pond intake (outfall edge into a STORAGE node): the
                # street is a flow sink for its whole intercepted catchment, so
                # size the intake conduit to the COMBINED inflow (area-
                # equivalent √Σdᵢ²) rather than the largest single feeder —
                # prevents the intake junction surcharging behind an undersized
                # straw.
                if v in wb_node_ids:
                    diam = max(diam, float(np.sqrt(_sum_in_d2_at_node.get(u, 0.0))))
            # Physical constraint: a circular pipe can't be shorter than
            # its own diameter (it would literally be a hole in the wall).
            # Clamp Length >= Diameter for every emitted pipe so SWMM
            # doesn't see L/D < 1 -- that ratio corresponds to a pipe
            # fitting / manhole junction, not a conveyance pipe, and it
            # destabilises DYNWAVE convergence (the pipe alternately
            # surcharges and drains on every routing step).  1.0 m is a
            # final floor so pipes buried in very short graph segments
            # still get a physically meaningful routing length.
            #
            # Outfall pipes get a tighter L >= 5*D rule.  Outfalls are
            # at the system boundary and any short-pipe instability
            # propagates upstream as the dominant Time-Step Critical
            # Element in DYNWAVE.  L/D = 5 is the standard culvert /
            # short-pipe stability threshold (Chow's Open-Channel
            # Hydraulics; matches our merger-conduit sizing).  Adds at
            # most ~12 m to a 3-m outfall pipe — negligible for the
            # short stub at the network outflow boundary.
            l_over_d_min = 5.0 if edge_type == "outfall" else 1.0
            effective_length = max(length, l_over_d_min * diam, 1.0)

            conduits_dict[eid] = {
                "InletNode": u,
                "OutletNode": v,
                "Length": effective_length,
                "Roughness": roughness if roughness < 0.02 else 0.01,
                "InOffset": in_offset,
                "OutOffset": out_offset,
                "InitFlow": 0,
                "MaxFlow": 1e10,
            }
            xsections_dict[eid] = {
                "Shape": "CIRCULAR",
                "Geom1": diam,
            }

    # --- Fix outfall constraints (SWMM: exactly 1 inlet, 0 outlets per outfall) ---
    # Count incoming and outgoing links per outfall node
    outfall_inlet_count: dict[str, int] = dict.fromkeys(outfalls_dict, 0)
    outfall_outlet_count: dict[str, int] = dict.fromkeys(outfalls_dict, 0)
    all_link_dicts = [conduits_dict, orifices_dict, weirs_dict]
    for link_dict in all_link_dicts:
        for ldata in link_dict.values():
            out_node = ldata.get("OutletNode", "")
            in_node = ldata.get("InletNode", "")
            if out_node in outfall_inlet_count:
                outfall_inlet_count[out_node] += 1
            if in_node in outfall_outlet_count:
                outfall_outlet_count[in_node] += 1

    # Outfalls that need fixing: >1 inlet OR any outlet
    needs_fix = {
        oid
        for oid in outfalls_dict
        if outfall_inlet_count.get(oid, 0) > 1 or outfall_outlet_count.get(oid, 0) > 0
    }

    # Pre-compute the COMBINED cross-sectional area of every conduit
    # terminating at each merged outfall, so the merger->outfall stub can
    # carry all of them at once.  A merger junction receives N feeders
    # whose peak flows roughly coincide under a design storm; sizing the
    # stub to the single largest feeder (the previous behaviour) leaves
    # it badly undersized whenever several conduits converge — on the test catchment a
    # 0.9 m stub fed by 5 conduits ran at 26 m/s / 264 % full and became
    # the dominant DYNWAVE time-step-critical element.  Open channels
    # (RECT_OPEN) were also skipped entirely; both shapes are summed here
    # and converted back to an equivalent circular diameter.  No parameter.
    feeder_area_sum: dict[str, float] = {}
    for cid, cdata in conduits_dict.items():
        out_node = cdata.get("OutletNode", "")
        if out_node not in needs_fix:
            continue
        xs = xsections_dict.get(cid, {})
        shape = xs.get("Shape")
        if shape == "CIRCULAR":
            d = float(xs.get("Geom1", 0) or 0)
            feeder_area_sum[out_node] = feeder_area_sum.get(out_node, 0.0) + (np.pi * d * d / 4.0)
        elif shape == "RECT_OPEN":
            geom1 = float(xs.get("Geom1", 0) or 0)
            geom2 = float(xs.get("Geom2", 0) or 0)
            feeder_area_sum[out_node] = feeder_area_sum.get(out_node, 0.0) + geom1 * geom2

    for oid in needs_fix:
        # Insert a merger junction upstream of this outfall
        merger_id = f"{oid}_merger"
        of_data = outfalls_dict[oid]
        of_elev = of_data["InvertElev"]
        of_coord = coordinates_dict.get(oid, {"X": 0, "Y": 0})

        # Lower the outfall invert by 0.5 m so the merger -> outfall
        # connector has a clear positive slope, and so any pipe draining to
        # the merger has headroom before the outfall.
        of_data["InvertElev"] = of_elev - 0.5

        # Merger junction sits AT the original outfall invert so incoming
        # pipes drain freely without backing up.
        junctions_dict[merger_id] = {
            "InvertElev": of_elev,
            "MaxDepth": 3.0,
            "InitDepth": 0,
            "SurchargeDepth": 0,
            "PondedArea": 0,
        }
        coordinates_dict[merger_id] = {
            "X": of_coord["X"] + 1,
            "Y": of_coord["Y"] + 1,
        }

        # Reroute all links that drain to or originate from this outfall
        for link_dict in all_link_dicts:
            for ldata in link_dict.values():
                if ldata.get("OutletNode") == oid:
                    ldata["OutletNode"] = merger_id
                if ldata.get("InletNode") == oid:
                    ldata["InletNode"] = merger_id

        # CIRCULAR merger->outfall stub sized so its area equals the
        # combined feeder area (so it never bottlenecks the inflow), with
        # length = max(5*D, 5 m).  L >= 5*D keeps L/D well above SWMM's
        # short-pipe instability threshold (1) and matches typical
        # outfall culvert proportions.  n = 0.012 (concrete) instead of
        # 0.01 to match the rest of the pipe network and dampen the
        # advection wave.
        area_sum = feeder_area_sum.get(oid, 0.0)
        feeder_d = float(np.sqrt(4.0 * area_sum / np.pi)) if area_sum > 0 else 0.0
        merger_diam = max(feeder_d, 0.45)  # 18-in floor (typical outfall)
        merger_length = max(5.0 * merger_diam, 5.0)
        merger_link_id = f"{merger_id}-{oid}"
        conduits_dict[merger_link_id] = {
            "InletNode": merger_id,
            "OutletNode": oid,
            "Length": merger_length,
            "Roughness": 0.012,
            "InOffset": 0,
            "OutOffset": 0,
            "InitFlow": 0,
            "MaxFlow": 0,
        }
        xsections_dict[merger_link_id] = {
            "Shape": "CIRCULAR",
            "Geom1": merger_diam,
        }

    # --- Map settings ---
    map_settings = {
        "DIMENSIONS": [
            nodes.x.min(),
            nodes.y.min(),
            nodes.x.max(),
            nodes.y.max(),
        ]
    }

    inp_options = {} if inp_options is None else inp_options
    options = SWMMOptions.from_rain_data(rain, **inp_options)
    generator = SwmmInputGenerator(
        title="SWMManywhere Generated Model",
        options=options,
        raingages=raingages_dict,
        subcatchments=subcatchments_dict,
        junctions=junctions_dict,
        storage=storage_dict,
        outfalls=outfalls_dict,
        conduits=conduits_dict,
        orifices=orifices_dict,
        weirs=weirs_dict,
        losses=losses_dict,
        xsections=xsections_dict,
        coordinates=coordinates_dict,
        polygons=polygons_dict,
        curves=curves_dict,
        map_settings=map_settings,
    )

    generator.generate(addresses.model_paths.inp)


def _clip_pondsheds_to_aoi(
    rows: list[dict[str, Any]],
    clip_bbox_4326: tuple[float, float, float, float],
    target_crs: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    """Keep only pondshed polygon components intersecting the AOI.

    A pondshed is delineated over the buffered processing area, so a
    pond just outside the user's AOI can collect spatially-scattered
    subcatchments through the unified pipe/trunk network, producing a
    disconnected MultiPolygon whose far-flung parts are buffer-zone
    artifacts.  We explode each pondshed into its connected polygon
    components and keep only those that intersect the AOI; a pond with
    no component touching the AOI is dropped entirely.

    Returns ``(clipped_rows, n_dropped_parts, n_dropped_ponds)``.
    """
    aoi = gpd.GeoSeries([shapely.box(*clip_bbox_4326)], crs="EPSG:4326").to_crs(target_crs).iloc[0]
    clipped_rows: list[dict[str, Any]] = []
    n_dropped_parts = 0
    n_dropped_ponds = 0
    for rec in rows:
        parts = list(shapely.get_parts(rec["geometry"]))
        keep = [p for p in parts if p.intersects(aoi)]
        n_dropped_parts += len(parts) - len(keep)
        if not keep:
            n_dropped_ponds += 1
            continue
        new_geom = shapely.unary_union(keep)
        clipped_rows.append(
            {
                "outlet_node_id": rec["outlet_node_id"],
                "pond_id": rec["pond_id"],
                "n_subcatchments": rec["n_subcatchments"],
                "area_m2": float(shapely.area(new_geom)),
                "geometry": new_geom,
            }
        )
    return clipped_rows, n_dropped_parts, n_dropped_ponds


def generate_pondsheds(  # noqa: C901 - per-outlet BFS partition with pond + lake sinks, then trimmed to pond outlets
    addresses: FilePaths,
    clip_bbox_4326: tuple[float, float, float, float] | None = None,
) -> None:
    """Write a per-outlet drainage-area polygon file for visualization.

    A pondshed is the full upstream watershed that arrives at a pond's
    network outlet through the urban dual-drainage system, *excluding*
    any catchment that first drains into another pond outlet OR into a
    natural lake (an OSM water body that wasn't ingested as a managed
    pond).  The result is a Voronoi-like partition: pondsheds nest
    cleanly without overlap, and a downstream pond holds only the
    *incremental* area it adds beyond what its upstream ponds and
    lakes already drained.

    Algorithm:

    1. Build the filtered drainage network — stormdrain pipes
       (``edge_type="pipe"``), surface dual-drainage channels
       (``street_channel``), pond outlet structures (``orifice``,
       ``weir``), pond outflow connectors (``pond_outflow``) and
       outfalls (``outfall``).  Natural rivers (``river``) are dropped.
    2. Identify two kinds of sinks:

       - **Pond outlets**: nodes targeted by a ``pond_outflow`` edge —
         the network junctions where managed-pond outflow joins the
         drainage system.
       - **Lake outlets**: outlet nodes of subcatchments whose
         centroid lies inside an OSM water polygon
         (``basins.parquet``) that wasn't matched to any pond storage
         (a 50 m buffer absorbs the snap-to-junction offset).  These
         are the natural water bodies the pond classifier rejected
         (above ``max_pond_area_m2`` cutoff or filtered out by tag).

       ``basins.parquet`` is used here as a *topological* signal —
       it tells us which graph nodes anchor lake-bound flow — not as
       a polygon mask to subtract from output geometries.
    3. Multi-source BFS upstream from every sink in the filtered
       graph, blocked at other sinks.  Each non-sink node is assigned
       to its nearest downstream sink (the next thing its runoff hits
       — pond outlet, lake outlet, or nothing if it terminates at the
       boundary).
    4. Map each subcatchment to its outlet's sink, group, union
       geometries.
    5. Keep only pond-outlet sinks in the output.  Lake-outlet
       partitions hold the lake's catchment and are silently dropped;
       pond-outlet partitions hold every upstream sub draining there
       through the pipe-and-channel network.

    Pond storage nodes have in-degree 0 in this graph (no
    ``pond_inflow`` edges), so they aren't sinks themselves; however
    BFS from a pond's outlet traverses the orifice/weir backwards
    into the storage node, so the pond's own subcatchment
    (``Sub.Outlet == pond_storage_id``) is correctly captured.

    When several ponds share one outlet (cluster reroutes to one
    downstream junction), the merged catchment is split among the
    ponds by spatial Voronoi assignment: each subcatchment in the
    shared catchment is attributed to the pond whose storage centroid
    is nearest to the sub centroid.  This guarantees one pondshed row
    per ``pond_outflow`` edge in the graph — the function logs a
    warning if the produced count diverges from the expected count
    (e.g. a pond whose upstream catchment is empty in the pipe graph).

    Output: GeoDataFrame parquet at ``addresses.model_paths.pondsheds``
    with one row per ``pond_outflow`` edge: columns ``outlet_node_id``
    (the downstream junction the pond discharges to), ``pond_id`` (the
    pond storage node), ``n_subcatchments``, ``area_m2``, ``geometry``.
    SWMM does not consume this file.

    The partition logic (filtered graph, sinks, BFS, Voronoi tiebreak)
    is shared with the pipeline step :func:`route_pipes_into_ponds` via
    :func:`graph_functions.water_bodies._partition_pond_network`, so
    every subcatchment and every rerouted pipe end up consistently
    attributed to the same pond.

    Design invariant — a pondshed may legitimately be a *disconnected*
    MultiPolygon and that is **not** a defect.  Every subcatchment is
    guaranteed contiguous (``geospatial_utilities.rehome_detached_components``
    runs after both subcatchment-merge steps), so any disconnection here
    is purely the pipe-BFS faithfully grouping spatially-separated
    subcatchments that the storm-drain network routes to one pond —
    pipes cross natural drainage divides by design.  The polygon is
    therefore a true picture of what the modelled network delivers to
    the pond; it is intentionally not forced contiguous.
    """
    from swmmanywhere_us.graph_functions.water_bodies import partition_pond_network
    from swmmanywhere_us.graph_utilities import load_graph

    graph_path = addresses.model_paths.graph
    subs_path = addresses.model_paths.subcatchments
    if not graph_path.exists() or not subs_path.exists():
        logger.warning("generate_pondsheds: missing graph or subcatchments file; skipping.")
        return

    graph: nx.MultiDiGraph[Any] = load_graph(graph_path)  # type: ignore[assignment]
    subs = gpd.read_parquet(subs_path)
    if subs.empty:
        logger.info("generate_pondsheds: empty subcatchments; nothing to write.")
        return

    # ``pond_sink_mode="storage"``: the pondshed is the catchment that
    # drains *through* the pond (upstream of the storage node), not the
    # over-collected catchment of the pond's downstream junction.  This
    # is what makes pondsheds spatially coherent — see
    # ``partition_pond_network`` docstring.
    #
    # Drop ``street_channel`` from the partition graph for delineation: the
    # surface (dual-drainage) chain creates *parallel* routes (each surface
    # node splits into a street_channel overland edge and a grate orifice down
    # into the pipe), which makes a sub reachable by several ponds and forces
    # the BFS to break the tie by hop-count.  Following only the piped path
    # (grate-captured flow → pipes → pond) is each sub's dominant flow route,
    # so the assignment is deterministic and matches where the captured runoff
    # actually goes (the surface channel only carries exceedance overflow).
    # ``route_pipes_into_ponds`` keeps the full edge set — this override is
    # delineation-only.
    pondshed_keep = frozenset({"pipe", "orifice", "weir", "pond_inflow", "pond_outflow", "outfall"})
    _pipe_graph, pond_outlets, lake_outlets, node_to_pond = partition_pond_network(
        graph, addresses, keep_edge_types=pondshed_keep, pond_sink_mode="storage"
    )
    if not pond_outlets:
        logger.warning("generate_pondsheds: no pond_outflow edges; nothing to write.")
        return
    if lake_outlets:
        logger.info(f"generate_pondsheds: blocking BFS at {len(lake_outlets)} lake outlet(s).")

    # Map subs to ponds via Sub.Outlet (= subs.id), using the shared
    # partition's per-node pond assignment (Voronoi already applied
    # there for multi-pond sinks).
    subs = subs[["id", "geometry", "area"]].copy()
    subs["id"] = subs["id"].astype("int64")
    subs["pond"] = subs["id"].map(node_to_pond)

    subs_with_pond = subs.dropna(subset=["pond"])
    if subs_with_pond.empty:
        logger.warning("generate_pondsheds: no subcatchments mapped to any pond.")
        return

    # Each pond's ds_node (where its pond_outflow terminates), for the
    # ``outlet_node_id`` column.
    pond_to_ds: dict[Any, Any] = {}
    for ds_node, pond_ids in pond_outlets.items():
        for pid in pond_ids:
            pond_to_ds[pid] = ds_node

    def _node_id(x: Any) -> int | str:
        """Coerce a graph node id (may arrive as numpy float from pandas groupby)
        to int when integer-valued, else fall back to str for safety.
        """
        try:
            return int(x)
        except (TypeError, ValueError):
            return str(x)

    # Group subs by their assigned pond.  Each resulting row is one
    # pondshed: the union of subcatchments whose runoff arrives at that
    # specific pond (after Voronoi tiebreak for multi-pond sinks).
    rows: list[dict[str, Any]] = []
    for pond_key, group in subs_with_pond.groupby("pond"):
        pond_id = _node_id(pond_key)
        ds_node = pond_to_ds.get(pond_key, pond_key)
        polygon = shapely.unary_union(group.geometry.to_numpy())
        # Trim trivial detached slivers (< 0.1 ha) so a pondshed isn't reported
        # as a MultiPolygon just because the network pipes one far sub to it via
        # a long path.  area_m2/n_subcatchments stay the true catchment totals
        # (those subs still drain to the pond in SWMM); only the polygon is
        # trimmed for a clean visual.
        polygon = _drop_small_detached_parts(polygon, 1000.0)
        rows.append(
            {
                "outlet_node_id": _node_id(ds_node),
                "pond_id": pond_id,
                "n_subcatchments": len(group),
                "area_m2": float(group["area"].sum()),
                "geometry": polygon,
            }
        )

    if not rows:
        logger.warning("generate_pondsheds: no pondsheds for pond outlets.")
        return

    # Validation: # pondsheds should equal # pond_outflow edges.
    n_pond_outflows = sum(
        1
        for _, _, _, d in graph.edges(keys=True, data=True)
        if d.get("edge_type") == "pond_outflow"
    )
    if len(rows) != n_pond_outflows:
        logger.warning(
            f"generate_pondsheds: produced {len(rows)} pondshed(s) but graph has "
            f"{n_pond_outflows} pond_outflow edges — {n_pond_outflows - len(rows)} "
            "pond(s) have no upstream catchment in the pipe network."
        )

    pondsheds_gdf = gpd.GeoDataFrame(rows, crs=subs.crs)

    # Clip to the original (unbuffered) AOI: a pondshed is delineated
    # over the buffered processing area, so a pond just outside the AOI
    # can collect spatially-scattered subcatchments through the unified
    # pipe/trunk network — producing disconnected MultiPolygon
    # "pondsheds" whose far-flung parts are buffer-zone artifacts.  We
    # explode each pondshed into its connected polygon components and
    # keep only the components that intersect the AOI; a pond with no
    # component touching the AOI is dropped entirely.  This delivers
    # spatially-coherent pondsheds for the area the user actually asked
    # for while still benefiting from the full-topology buffered run.
    if clip_bbox_4326 is not None and not pondsheds_gdf.empty:
        target_crs = subs.crs if subs.crs is not None else "EPSG:4326"
        clipped_rows, n_dropped_parts, n_dropped_ponds = _clip_pondsheds_to_aoi(
            rows, clip_bbox_4326, target_crs
        )
        pondsheds_gdf = gpd.GeoDataFrame(clipped_rows, crs=target_crs)
        logger.info(
            f"generate_pondsheds: clipped to AOI — dropped {n_dropped_parts} "
            f"disconnected component(s) and {n_dropped_ponds} pond(s) "
            "entirely outside the original bbox."
        )

    pondsheds_gdf.to_parquet(addresses.model_paths.pondsheds)
    logger.info(
        f"generate_pondsheds: wrote {len(pondsheds_gdf)} pondshed(s) "
        f"({n_pond_outflows} pond_outflow edge(s) in graph), covering "
        f"{pondsheds_gdf['area_m2'].sum() / 1e4:.1f} ha total to "
        f"{addresses.model_paths.pondsheds.name}"
    )
