"""Dual drainage: minor (pipe) + major (street) systems linked by catchbasin grates.

Implements the dual-drainage modelling approach assessed in:

- Senior, M.; Scheckenberger, R.; Bishop, B. (2018), *"Modeling
  Catchbasins and Inlets in SWMM."* Journal of Water Management
  Modeling 26:C435. https://doi.org/10.14796/JWMM.C435

Senior et al. compare five ways to link the minor (storm sewer) and
major (overland/street) systems and recommend **Method 4**: the
catchbasin grate modelled as an *orifice* connecting a separate surface
(roadway) node to the buried storm-sewer node.  Connecting the two
systems directly at one shared node (their Method 1A) makes the DYNWAVE
head equation stiff — SWMM has to split flow between a closed pipe and
an open channel at a single shared head — which destabilises the solver
and overestimates sewer inflow.  An orifice linkage is a well-behaved
hydraulic element: it meters inflow with head, permits reverse flow
(sewer surcharge back onto the street), and numerically decouples the
two systems.

On top of the buried pipe network this module builds:

1. a **surface (roadway) node** above every pipe junction, its invert
   one curb-depth below grade (the gutter);
2. **street-channel** conduits between the surface nodes, mirroring the
   pipe topology — the major-system overland conveyance;
3. a **catchbasin grate orifice** dropping each surface node into its
   pipe junction (two-way, so sewer surcharge reverses onto the street);
4. **surface outfalls** so the street system discharges overland to the
   receiving water in parallel with the pipes.

Subcatchment runoff is rerouted to the surface node — it lands on the
road, enters the sewer through the grate, and the street carries the
excess once the sewer is full.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import shapely

from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    import networkx as nx

    from swmmanywhere_us.filepaths import FilePaths
    from swmmanywhere_us.parameters import DualDrainage, SubcatchmentDerivation

# Catchbasin grate hydraulics — EPA SWMM dual-drainage Method 4
# (Senior et al. 2018, JWMM C435).  A standard Ontario bottom-draw grate
# (OPSD 400.010 / 400.100) has an opening area of ~0.125 m^2; 0.62 is the
# standard sharp-edged orifice discharge coefficient.
_GRATE_AREA_M2 = 0.125
_GRATE_CD = 0.62


def _street_width_m(data: dict[str, Any], default_lanes: float, lane_width: float) -> float:
    """Infer street curb-to-curb width in metres from edge data.

    Prefers an explicit ``lanes`` attribute (int, float or comma-separated
    string from OSM).  Falls back to ``default_lanes`` when the value is
    missing or unparsable.
    """
    lanes_raw = data.get("lanes", default_lanes)
    try:
        if isinstance(lanes_raw, str):
            lanes = float(lanes_raw.split(",")[0])
        elif isinstance(lanes_raw, list):
            lanes = sum(float(str(x).split(",")[0]) for x in lanes_raw)
        else:
            lanes = float(lanes_raw)
    except (ValueError, TypeError):
        lanes = float(default_lanes)
    return max(lanes, 1.0) * lane_width


def _add_surface_nodes(
    graph: nx.MultiDiGraph[Any],
    pipe_nodes: set[Any],
    curb_depth: float,
    next_id: int,
) -> tuple[dict[Any, int], int]:
    """Add a surface (roadway) node above each pipe junction.

    The surface node's invert sits one curb-depth below grade (the
    gutter), so the SWMM junction MaxDepth = curb_depth: the street ponds
    up to curb height, then floods.  Returns the junction -> surface-node
    mapping and the next free integer node id.
    """
    surf: dict[Any, int] = {}
    for j in pipe_nodes:
        jd = graph.nodes[j]
        if "x" not in jd or jd.get("surface_elevation") is None:
            continue
        js = next_id
        next_id += 1
        surf[j] = js
        road = float(jd["surface_elevation"])
        graph.add_node(
            js,
            x=jd["x"],
            y=jd["y"],
            surface_elevation=road,
            chamber_floor_elevation=road - curb_depth,
            node_type="surface",
        )
    return surf, next_id


def add_dual_drainage(
    graph: nx.MultiDiGraph[Any],
    dual_drainage: DualDrainage,
    subcatchment_derivation: SubcatchmentDerivation,
    addresses: FilePaths,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Overlay the major (street) system on the pipe network — Method 4.

    Runs after ``pipe_by_pipe`` so pipe inverts and junction depths are
    known.  Adds a surface (roadway) node above every pipe junction,
    street-channel conduits between the surface nodes, a catchbasin grate
    orifice from each surface node down to its pipe junction, and surface
    outfalls; then reroutes every subcatchment's ``Outlet`` to the
    surface node.

    Args:
        graph: The pipeline graph after pipe design.  Pipe nodes must
            carry ``x``, ``y``, ``surface_elevation`` and
            ``chamber_floor_elevation``.
        dual_drainage: Channel geometry + roughness parameters.
        subcatchment_derivation: Provides ``lane_width`` for street-width
            inference from OSM ``lanes`` tags.
        addresses: File path manager — used to reroute the subcatchments
            parquet to the new surface nodes.
        **kwargs: Ignored.

    Returns:
        Graph with the surface (major) system added.  When disabled, the
        graph is returned unchanged.
    """
    if not dual_drainage.enabled:
        logger.info("add_dual_drainage: disabled by config; skipping.")
        return graph

    graph = graph.copy()
    curb_depth = dual_drainage.curb_depth_m
    roughness = dual_drainage.channel_roughness
    default_lanes = dual_drainage.default_lanes
    lane_width = subcatchment_derivation.lane_width
    grate_diam = math.sqrt(4.0 * _GRATE_AREA_M2 / math.pi)

    pipe_edges = [
        (u, v, k, d)
        for u, v, k, d in graph.edges(keys=True, data=True)
        if d.get("edge_type") == "pipe"
    ]
    pipe_nodes = {n for u, v, _k, _d in pipe_edges for n in (u, v)}
    if not pipe_nodes:
        logger.info("add_dual_drainage: no pipe edges; skipping.")
        return graph

    # Running id counter for new surface nodes — guaranteed not to
    # collide with any existing integer node id.
    int_ids = [n for n in graph.nodes if isinstance(n, int)]
    next_id = (max(int_ids) + 1) if int_ids else 1

    # 1. Surface (roadway) node above each pipe junction.
    surf, next_id = _add_surface_nodes(graph, pipe_nodes, curb_depth, next_id)

    # 2. Street-channel conduits between the surface nodes (mirror the
    #    pipe topology) — the major-system overland conveyance.
    n_channels = 0
    for u, v, _k, d in pipe_edges:
        if u not in surf or v not in surf:
            continue
        us, vs = surf[u], surf[v]
        graph.add_edge(
            us,
            vs,
            key=0,
            edge_type="street_channel",
            id=f"{us}-{vs}-channel",
            geometry=d.get("geometry"),
            length=d["length"],
            channel_width=_street_width_m(d, default_lanes, lane_width),
            channel_depth=curb_depth,
            roughness=roughness,
            in_offset=0.0,
            out_offset=0.0,
            diameter=0.0,
            contributing_area=0.0,
        )
        n_channels += 1

    # 3. Catchbasin grate — a two-way BOTTOM orifice dropping each
    #    surface node into its pipe junction.  Two-way (no flap gate) so
    #    sewer surcharge reverses back onto the street.
    for j, js in surf.items():
        jd = graph.nodes[j]
        # The grate is a vertical structure (surface node directly above
        # the pipe junction); a short horizontal stub keeps the GIS
        # geometry non-degenerate.
        graph.add_edge(
            js,
            j,
            key=0,
            edge_type="orifice",
            id=f"{js}-{j}-grate",
            geometry=shapely.LineString([(jd["x"], jd["y"]), (jd["x"] + 0.5, jd["y"])]),
            length=1.0,
            diameter=0.0,
            orifice_type="BOTTOM",
            orifice_diam_m=grate_diam,
            orifice_cd=_GRATE_CD,
            orifice_offset=0.0,
            flap_gate="NO",
        )

    # 4. Surface outfalls — let the street system discharge overland to
    #    the receiving water alongside the pipes.  A SWMM outfall accepts
    #    only one link, so each gets its own surrogate outfall node.
    n_outfalls = 0
    for u, v, _k, d in list(graph.edges(keys=True, data=True)):
        if d.get("edge_type") != "outfall" or u not in surf:
            continue
        us = surf[u]
        os_id = next_id
        next_id += 1
        od = graph.nodes[v]
        graph.add_node(
            os_id,
            x=od.get("x", graph.nodes[us]["x"]),
            y=od.get("y", graph.nodes[us]["y"]),
            surface_elevation=od.get("surface_elevation"),
            chamber_floor_elevation=od.get("chamber_floor_elevation"),
            node_type="dummy_river",
        )
        graph.add_edge(
            us,
            os_id,
            key=0,
            edge_type="street_channel",
            id=f"{us}-{os_id}-outfall",
            geometry=d.get("geometry"),
            length=max(float(d.get("length", 0.0) or 0.0), 1.0),
            channel_width=default_lanes * lane_width,
            channel_depth=curb_depth,
            roughness=roughness,
            in_offset=0.0,
            out_offset=0.0,
            diameter=0.0,
            contributing_area=0.0,
        )
        n_outfalls += 1

    # 5. Reroute each subcatchment's Outlet from the pipe junction to its
    #    surface node — runoff lands on the road, then enters the sewer
    #    through the grate.
    subs_path = addresses.model_paths.subcatchments
    n_rerouted = 0
    if subs_path.exists():
        subs = gpd.read_parquet(subs_path)
        if not subs.empty and "id" in subs.columns:
            mapped = subs["id"].map(lambda x: surf.get(x, x))
            n_rerouted = int((mapped != subs["id"]).sum())
            subs["id"] = mapped
            subs.to_parquet(subs_path)

    logger.info(
        f"add_dual_drainage (Method 4): {len(surf)} surface node(s), "
        f"{n_channels} street channel(s), {len(surf)} grate orifice(s), "
        f"{n_outfalls} surface outfall(s); rerouted {n_rerouted} subcatchment(s)."
    )
    return graph
