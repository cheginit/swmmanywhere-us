"""Module for graphfcns that identify outfalls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from swmmanywhere_us import parameters
from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from swmmanywhere_us.filepaths import FilePaths


def _get_points(
    graph: nx.MultiDiGraph[Any],
) -> tuple[dict[Any, shapely.Point], dict[Any, shapely.Point]]:
    """Get the river and street points from the graph.

    A river point are the start and end nodes of an edge with `edge_type = river`.
    A street point are the start and end nodes of an edge with
    `edge_type = street`.

    Args:
        graph (nx.Graph): A graph

    Returns:
        river_points (dict): A dictionary of river points
        pipe_points (dict): A dictionary of street points
    """
    # Get edge types, convert to nx.Graph to remove keys
    etypes = nx.get_edge_attributes(nx.Graph(graph), "edge_type")

    # Get river and street points as a dict
    n_types = (
        pd.DataFrame(etypes.items(), columns=["key", "type"])
        .explode("key")
        .reset_index(drop=True)
        .groupby("type")["key"]
        .apply(list)
        .to_dict()
    )
    river_points = {
        n: shapely.Point(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in n_types.get("river", {})
    }
    pipe_points = {
        n: shapely.Point(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in n_types["pipe"]
    }

    return river_points, pipe_points


def _load_outfall_water_bodies(
    addresses: FilePaths,
    target_crs: Any,
    min_area_m2: float,
) -> list[Any]:
    """Load water-body polygons to use as outfall (discharge) candidates.

    Reads the NLCD ``water_bodies`` parquet and the OSM ``basins`` parquet,
    reprojects both to ``target_crs``, drops empty/below-threshold polygons,
    and returns the combined polygon geometries as a list (empty when no
    qualifying polygons are available).

    Unlike :func:`water_bodies._load_water_body_polygons`, this does no
    pond/lake classification — every water body above ``min_area_m2`` is a
    plausible receiving water, and the downstream MST / shortest-path outfall
    selection prunes which are actually used.
    """
    geoms: list[gpd.GeoSeries] = []
    for path in (addresses.bbox_paths.water_bodies, addresses.bbox_paths.basins):
        if not path.exists():
            continue
        gdf = gpd.read_parquet(path)
        if gdf.empty:
            continue
        if target_crs and gdf.crs and str(gdf.crs) != str(target_crs):
            gdf = gdf.to_crs(target_crs)
        geoms.append(gdf.geometry)

    if not geoms:
        return []

    polys = gpd.GeoSeries(pd.concat(geoms, ignore_index=True), crs=target_crs)
    valid = polys.notna() & ~polys.is_empty & (polys.area >= min_area_m2)
    return list(polys[valid])


def _pond_intake_pairs(
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    target_crs: Any,
    min_area_m2: float = 0.0,
) -> list[tuple[Any, Any]]:
    """Match each pond STORAGE node to its basin polygon (for on-line intake).

    ``insert_pond_nodes`` places each pond's STORAGE node at the basin
    polygon's centroid, so the nearest basin polygon to a storage node is its
    own footprint.  Returns ``[(storage_node_id, polygon), ...]`` for use by
    :func:`_pair_rivers` to pair surrounding pipe nodes to the pond.  Ponds
    smaller than ``min_area_m2`` are excluded (left off-line).  Empty when
    there are no pond storages or no basins file.
    """
    pond_nodes = [
        (n, d)
        for n, d in graph.nodes(data=True)
        if d.get("node_type") == "water_body" and d.get("wb_area_m2", 0.0) >= min_area_m2
    ]
    if not pond_nodes or not addresses.bbox_paths.basins.exists():
        return []
    basins = gpd.read_parquet(addresses.bbox_paths.basins)
    if basins.empty:
        return []
    if target_crs and basins.crs and str(basins.crs) != str(target_crs):
        basins = basins.to_crs(target_crs)
    geoms = list(basins.geometry)
    tree = shapely.STRtree(geoms)
    pairs: list[tuple[Any, Any]] = []
    for n, d in pond_nodes:
        pt = shapely.Point(d["x"], d["y"])
        poly = geoms[int(tree.nearest(pt))]
        # Accept only when the storage really sits in/at this basin (a pond's
        # own footprint), so a stray storage doesn't grab a distant polygon.
        if poly.distance(pt) <= 5.0:
            pairs.append((n, poly))
    return pairs


def _pair_rivers(  # noqa: C901, PLR0912, PLR0915 - river/water-body/pond pairing with per-subgraph fallback ladder is one pass
    graph: nx.MultiDiGraph[Any],
    river_points: dict[Any, shapely.Point],
    pipe_points: dict[Any, shapely.Point],
    river_buffer_distance: float,
    outfall_length: float,
    water_body_polys: list[Any] | None = None,
    pond_intake_pairs: list[tuple[Any, Any]] | None = None,
    pond_intake_buffer: float = 30.0,
) -> nx.MultiDiGraph[Any]:
    """Pair river and street nodes.

    Street nodes within ``river_buffer_distance`` of a river *centerline*
    are paired to a synthetic ``river_outfall`` sink snapped onto that
    centerline.  Distance is measured to the river edge geometry, not to
    river graph nodes: river features are noded only at their endpoints
    (often hundreds of metres apart), so point-to-node pairing misses most
    streets that actually border the water.  When ``water_body_polys`` is
    supplied, street nodes within the same buffer of a water-body polygon
    are likewise paired to a synthetic ``water_body_outfall`` sink on the
    shoreline.  If a subgraph has no river *and* no water-body match, a
    dummy river node is created and the lowest elevation node is used as
    the outfall.

    Args:
        graph: A directed multi-graph.
        river_points: A dictionary of river points (used to place dummy
            river sinks).
        pipe_points: A dictionary of street points.
        river_buffer_distance: The distance within which a river and
            street node can be paired.
        outfall_length: The length of the outfall.
        water_body_polys: Optional water-body polygon geometries to treat as
            outfall candidates alongside rivers.

    Returns:
        The graph with paired river-street outfalls.
    """
    # street_node -> dummy river sink (components with no river/water-body
    # match).  Edges for these are added after the per-component loop.
    matched_outfalls = {}
    # street_node -> river edge geometry index (one snapped sink node is
    # created per pairing after the per-component loop).
    river_pairings: dict[Any, int] = {}
    # street_node -> water-body polygon index (one sink node is created per
    # matched polygon after the per-component loop).
    wb_pairings: dict[Any, int] = {}

    # Build a spatial index over the river centerlines so each street node
    # can be matched to the nearest point on the river within the buffer
    # distance.
    river_lines: list[Any] = [
        d["geometry"]
        for _, _, d in graph.edges(data=True)
        if d.get("edge_type") == "river" and d.get("geometry") is not None
    ]
    river_tree: shapely.STRtree | None = shapely.STRtree(river_lines) if river_lines else None

    # Build a spatial index over the water-body polygons so each street node
    # can be matched to the nearest shoreline within the buffer distance.
    wb_geoms: list[Any] = list(water_body_polys) if water_body_polys else []
    wb_tree: shapely.STRtree | None = shapely.STRtree(wb_geoms) if wb_geoms else None

    # Allocate fresh integer IDs for dummy river sentinels so node IDs stay
    # homogeneous across the graph.  Semantic info lives in ``node_type``.
    existing_ints = [n for n in graph.nodes if isinstance(n, int)]
    next_dummy_id = (max(existing_ints) + 1) if existing_ints else 0

    # Insert a dummy river node and use lowest elevation node as outfall
    # for each subgraph with no matched outfalls
    subgraphs = []
    for component in nx.weakly_connected_components(graph):
        sg = graph.subgraph(component).copy()
        subgraphs.append(sg)

        # Pair up the river and street nodes for each subgraph
        pipe_points_ = {k: v for k, v in pipe_points.items() if k in sg.nodes}

        # Subgraphs without any street nodes (rivers and/or pond_connector-
        # only components) don't need outfall pairing — rivers drain via
        # their inherent direction and pond connectors are rewired later
        # by finalize_pond_outlets.
        if not pipe_points_:
            continue

        # Pair street nodes to nearby rivers using the centerline geometry:
        # each street node within river_buffer_distance of a river edge is
        # matched to that edge (the nearest one if several are in range).
        subgraph_rivers: dict[Any, int] = {}
        if river_tree is not None:
            for node, pt in pipe_points_.items():
                hits = river_tree.query_nearest(pt, max_distance=river_buffer_distance)
                if len(hits):
                    subgraph_rivers[node] = int(hits[0])

        # Pair street nodes to nearby water bodies using the polygon geometry:
        # each street node within river_buffer_distance of a polygon shoreline
        # is matched to that polygon (the nearest one if several are in range).
        subgraph_wb: dict[Any, int] = {}
        if wb_tree is not None:
            for node, pt in pipe_points_.items():
                hits = wb_tree.query_nearest(pt, max_distance=river_buffer_distance)
                if len(hits):
                    subgraph_wb[node] = int(hits[0])

        # Check if there are any matched outfalls (river or water body)
        if subgraph_rivers or subgraph_wb:
            river_pairings.update(subgraph_rivers)
            wb_pairings.update(subgraph_wb)
            continue

        # In cases of e.g., an area with no rivers/water bodies to discharge
        # into or too small a buffer

        # Identify the lowest elevation node among the street-side nodes.
        # Water-body STORAGE nodes are excluded (they're sources, not
        # discharge points) and any non-street nodes lack the coordinates
        # needed for the downstream outfall-edge geometry.
        candidate_nodes = [
            n for n in sg.nodes if n in pipe_points and sg.nodes[n].get("node_type") != "water_body"
        ]
        if not candidate_nodes:
            continue
        lowest_elevation_node = min(
            candidate_nodes,
            key=lambda x: sg.nodes[x]["surface_elevation"],
        )

        # Create a dummy river to discharge into.  Use a fresh int ID so
        # the whole graph keeps a homogeneous ID type.
        dummy_id = next_dummy_id
        next_dummy_id += 1
        x = graph.nodes[lowest_elevation_node]["x"] + 1
        y = graph.nodes[lowest_elevation_node]["y"] + 1
        sg.add_node(dummy_id)
        nx.set_node_attributes(sg, {dummy_id: {"x": x, "y": y, "node_type": "dummy_river"}})

        # Update function's dicts
        matched_outfalls[lowest_elevation_node] = dummy_id
        river_points[dummy_id] = shapely.Point(x, y)

        logger.warning(
            f"""No outfalls found for subgraph containing
                        {lowest_elevation_node}, using this node as outfall."""
        )

    graph = nx.compose_all(subgraphs)

    # Add edges between the dummy river sinks and their street nodes
    for street_id, river_id in matched_outfalls.items():
        # The fixed weight is intentional: it acts as the MST clustering
        # radius for outfall selection — distance-based weights were tested
        # and degrade outfall placement.
        graph.add_edge(
            street_id,
            river_id,
            length=outfall_length,
            weight=outfall_length,
            edge_type="outfall",
            geometry=shapely.LineString([pipe_points[street_id], river_points[river_id]]),
            id=f"{street_id}-{river_id}-outfall",
        )

    # Create a synthetic outfall sink for each river-paired street node,
    # placed at the point on the river centerline nearest that street, then
    # connect the two with an ``outfall`` edge.  Snapping onto the centerline
    # (rather than wiring to a river graph node) keeps the discharge at the
    # closest point on the receiving water — river nodes exist only at
    # feature endpoints, often hundreds of metres away — and gives the sink
    # a location where ``identify_outfalls`` can sample the receiving
    # water's surface elevation from the (hydroflattened) DEM.  These
    # ``river_outfall`` sinks are structurally identical to the dummy-river
    # and ``water_body_outfall`` sinks (single incoming ``outfall`` edge, no
    # outgoing edges).
    for street_id, line_idx in river_pairings.items():
        sink_id = next_dummy_id
        next_dummy_id += 1
        street_pt = pipe_points[street_id]
        line = shapely.shortest_line(street_pt, river_lines[line_idx])
        snap_pt = shapely.Point(line.coords[-1])
        graph.add_node(sink_id, x=snap_pt.x, y=snap_pt.y, node_type="river_outfall")
        graph.add_edge(
            street_id,
            sink_id,
            length=outfall_length,
            weight=outfall_length,
            edge_type="outfall",
            geometry=shapely.LineString([street_pt, snap_pt]),
            id=f"{street_id}-{sink_id}-outfall",
        )

    # Create a synthetic outfall sink for each paired street node, placed at
    # the point on the water-body shoreline (polygon exterior) nearest that
    # street, then connect the two with an ``outfall`` edge.  Snapping to the
    # shoreline (rather than the centroid) keeps the outfall conduit at the
    # water's edge instead of crossing open water, and gives the sink the
    # receiving water's edge location so ``identify_outfalls`` can sample its
    # surface elevation from the DEM.  These ``water_body_outfall`` sinks are
    # otherwise structurally identical to the dummy-river sinks above (single
    # incoming ``outfall`` edge, no outgoing edges).
    for street_id, poly_idx in wb_pairings.items():
        sink_id = next_dummy_id
        next_dummy_id += 1
        street_pt = pipe_points[street_id]
        line = shapely.shortest_line(street_pt, wb_geoms[poly_idx].exterior)
        shore_pt = shapely.Point(line.coords[-1])
        graph.add_node(sink_id, x=shore_pt.x, y=shore_pt.y, node_type="water_body_outfall")
        graph.add_edge(
            street_id,
            sink_id,
            length=outfall_length,
            weight=outfall_length,
            edge_type="outfall",
            geometry=shapely.LineString([street_pt, shore_pt]),
            id=f"{street_id}-{sink_id}-outfall",
        )

    # On-line pond intakes: pair pipe nodes near a pond's footprint to that
    # pond's STORAGE node with an ``outfall`` edge, so derive_topology treats
    # the pipe node as a drain point and routes the surrounding catchment into
    # the pond (which discharges via finalize_pond_outlets).  Constraints:
    #   * pond precedence is yielded to rivers — a node already paired to a
    #     river/dummy outfall is skipped, so no junction ends up with two
    #     competing downstream conduits;
    #   * only nodes in the SAME connected component as the pond storage are
    #     paired (an outfall edge across components would wrongly merge them);
    #   * water-body STORAGE nodes themselves are never paired.
    if pond_intake_pairs:
        component_of: dict[Any, int] = {}
        for ci, comp in enumerate(nx.weakly_connected_components(graph)):
            for n in comp:
                component_of[n] = ci
        pond_geoms = [poly for _, poly in pond_intake_pairs]
        pond_storage_ids = [sid for sid, _ in pond_intake_pairs]
        pond_tree = shapely.STRtree(pond_geoms)
        n_intakes = 0
        for node, pt in pipe_points.items():
            if node not in graph.nodes or node in matched_outfalls or node in river_pairings:
                continue
            if graph.nodes[node].get("node_type") in {"water_body", "water_body_outfall"}:
                continue
            hits = pond_tree.query_nearest(pt, max_distance=pond_intake_buffer)
            if not len(hits):
                continue
            storage_id = pond_storage_ids[int(hits[0])]
            if node == storage_id or component_of.get(node) != component_of.get(storage_id):
                continue
            storage_pt = shapely.Point(graph.nodes[storage_id]["x"], graph.nodes[storage_id]["y"])
            graph.add_edge(
                node,
                storage_id,
                length=outfall_length,
                weight=outfall_length,
                edge_type="outfall",
                pond_intake=True,
                geometry=shapely.LineString([pt, storage_pt]),
                id=f"{node}-{storage_id}-pondintake",
            )
            n_intakes += 1
        logger.info(f"online_pond_intake: added {n_intakes} pond intake edge(s).")

    return graph


def _root_nodes(graph: nx.MultiDiGraph[Any]) -> nx.MultiDiGraph[Any]:
    """Root nodes with a waste node.

    Connect all nodes that have nowhere to flow to to a waste node, i.e., the
    root of the entire graph.

    Args:
        graph (nx.Graph): A graph

    Returns:
        graph (nx.Graph): A graph
    """
    graph_ = graph.copy()

    # Allocate a fresh integer ID for the waste sentinel node.  Semantic
    # information lives in ``node_type`` so downstream code can locate it
    # without relying on a string identifier.
    existing_ints = [n for n in graph.nodes if isinstance(n, int)]
    waste_id = (max(existing_ints) + 1) if existing_ints else 0
    graph.add_node(waste_id, node_type="waste")

    for node in graph_.nodes:
        if graph.out_degree(node) == 0:
            # Location of the waste node doesn't matter - so if there
            # are multiple river nodes with out_degree 0 - that's fine.
            graph.nodes[waste_id]["x"] = graph.nodes[node]["x"] + 1
            graph.nodes[waste_id]["y"] = graph.nodes[node]["y"] + 1
            graph.add_edge(
                node,
                waste_id,
                length=0,
                weight=0,
                edge_type="waste-outfall",
                id=f"{node}-waste-outfall",
            )
    return graph


def _connect_mst_outfalls(
    paired_G: nx.MultiDiGraph[Any], raw_G: nx.MultiDiGraph[Any]
) -> nx.MultiDiGraph[Any]:
    """Connect outfalls to a waste node.

    Run a minimum spanning tree (MST) on the paired graph to identify the
    'efficient' outfalls. These outfalls are inserted into the original graph.

    Args:
        paired_G (nx.Graph): A graph where streets and rivers are paired with
            outfalls.
        raw_G (nx.Graph): A graph where streets and rivers are separated

    Returns:
        (nx.Graph): A graph
    """
    # Find shortest path to identify only 'efficient' outfalls. The MST
    # makes sense here over shortest path as each node is only allowed to
    # be visited once - thus encouraging fewer outfalls. In shortest path
    # nodes near rivers will always just pick their nearest river node.
    T = nx.minimum_spanning_tree(paired_G.to_undirected(), weight="length")

    # Retain the shortest path outfalls in the original graph
    for u, v, d in T.edges(data=True):
        u_waste = paired_G.nodes[u].get("node_type") == "waste"
        v_waste = paired_G.nodes[v].get("node_type") == "waste"
        if d["edge_type"] == "outfall" and not u_waste and not v_waste:
            if u not in raw_G.nodes():
                raw_G.add_node(u, **paired_G.nodes[u])
            elif v not in raw_G.nodes():
                raw_G.add_node(v, **paired_G.nodes[v])

            # Need to check both directions since T is undirected
            if (u, v) in paired_G.edges():
                raw_G.add_edge(u, v, **d)
            elif (v, u) in paired_G.edges():
                raw_G.add_edge(v, u, **d)
            else:
                msg = f"Edge {u}-{v} not found in paired_G"
                raise ValueError(msg)

    return raw_G


def _flag_iqr_outlier_outfalls(graph: nx.MultiDiGraph[Any]) -> list[Any]:
    """Flag outfall street nodes whose elevation is an IQR outlier.

    After outfall pairing, compute the interquartile range of street-side
    outfall node elevations and flag any that exceed Q3 + 1.5*IQR as
    outliers.  These are typically "fake" outfalls created in flat
    sub-basins that cannot actually drain to the flagged elevation.

    Returns a list of outlier street node IDs (for logging).  The filter
    is applied per weakly connected component so that topographically
    distinct drainage basins are not compared against each other.
    """
    outliers: list[Any] = []

    # Group outfall edges by weakly connected component
    # (pass the directed graph directly; weakly_connected_components
    #  computes WCCs by ignoring edge direction internally)
    wccs = list(nx.weakly_connected_components(graph))
    for wcc in wccs:
        outfall_street_nodes = []
        for u, _v, d in graph.edges(data=True):
            if d.get("edge_type") != "outfall":
                continue
            if u not in wcc:
                continue
            if "surface_elevation" not in graph.nodes[u]:
                continue
            outfall_street_nodes.append((u, float(graph.nodes[u]["surface_elevation"])))

        if len(outfall_street_nodes) < 4:
            continue  # IQR needs at least 4 samples to be meaningful

        elevations = np.array([e for _, e in outfall_street_nodes])
        q1, q3 = np.percentile(elevations, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        for node, elev in outfall_street_nodes:
            if elev > upper_bound:
                outliers.append(node)

    return outliers


def _set_water_body_outfall_elevations(graph: nx.MultiDiGraph[Any], addresses: FilePaths) -> None:
    """Sample the DEM at each synthetic outfall sink to set its elevation.

    River and water-body outfall sinks are created in :func:`_pair_rivers`
    after ``set_elevation`` has already run, so they carry no
    ``surface_elevation``.  Left unset, ``pipe_by_pipe`` falls back to
    ``0 - min_depth`` for them — a deep sentinel invert that produces a
    multi-metre drop over the short outfall conduit, which
    ``enforce_outfall_slope`` then "fixes" by stretching the conduit
    50-150 m.  Sampling the (hydroflattened) DEM at the snapped sink gives
    the receiving water's surface elevation, so the outfall sits at a
    realistic stage and the conduit keeps a sane slope.
    """
    import rasterio

    sinks = [
        (n, d["x"], d["y"])
        for n, d in graph.nodes(data=True)
        if d.get("node_type") in {"water_body_outfall", "river_outfall"}
    ]
    if not sinks:
        return
    coords = [(x, y) for _, x, y in sinks]
    with rasterio.open(addresses.bbox_paths.elevation) as src:
        nodata = src.nodata
        sampled = [float(v[0]) for v in src.sample(coords)]
    for (n, _, _), elev in zip(sinks, sampled):
        if np.isnan(elev) or (nodata is not None and elev == nodata):
            continue
        graph.nodes[n]["surface_elevation"] = elev
        graph.nodes[n]["chamber_floor_elevation"] = elev


def break_pond_intake_cycles(
    graph: nx.MultiDiGraph[Any],
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Drop on-line pond intake edges that close a directed cycle.

    An intake edge ``street -> pond_storage`` (added by ``identify_outfalls``
    when ``online_pond_intake`` is set) creates a cycle whenever the street is
    DOWNSTREAM of the pond: the pond discharges (via its connector, oriented
    ``storage -> anchor`` by ``derive_topology``) down to that street, which
    the intake would then feed back into the pond
    (``street -> storage -> anchor -> ... -> street``).  Such streets are
    genuinely below the pond and must drain on to the regional outfall, so
    their intake is removed; the upstream/surrounding intakes (the large
    majority) are kept, leaving the pond on-line for its true catchment.

    Runs after ``connect_pipe_components`` (so synthetic trunk pipes are
    accounted for) and before ``pipe_by_pipe`` (which needs an acyclic graph
    for its topological sort).  A no-op when there are no intake edges; the
    pre-intake graph is already a DAG, so every remaining cycle contains at
    least one intake edge.
    """
    if not any(d.get("pond_intake") for _, _, d in graph.edges(data=True)):
        return graph

    graph = graph.copy()
    removed = 0
    while not nx.is_directed_acyclic_graph(graph):
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:  # pragma: no cover - guarded by the while
            break
        cut = next(
            ((u, v, k) for u, v, k, _ in cycle if graph.edges[u, v, k].get("pond_intake")),
            None,
        )
        if cut is None:
            logger.warning(
                "break_pond_intake_cycles: found a cycle with no pond intake edge; "
                "leaving it for pipe_by_pipe to surface."
            )
            break
        graph.remove_edge(*cut)
        removed += 1

    if removed:
        logger.info(f"break_pond_intake_cycles: removed {removed} cyclic pond intake edge(s).")
    return graph


def _outfall_water_body_polys(
    graph: nx.MultiDiGraph[Any],
    outfall_derivation: parameters.OutfallDerivation,
    addresses: FilePaths | None,
) -> list[Any]:
    """Water-body polygons to pair as outfall candidates ([] when disabled)."""
    if not outfall_derivation.include_water_body_outfalls:
        return []
    if addresses is None:
        logger.warning(
            "include_water_body_outfalls is set but no addresses were "
            "provided; skipping water-body outfall pairing."
        )
        return []
    water_body_polys = _load_outfall_water_bodies(
        addresses,
        graph.graph.get("crs"),
        outfall_derivation.water_body_min_area_m2,
    )
    logger.info(f"Loaded {len(water_body_polys)} water-body polygon(s) as outfall candidates.")
    return water_body_polys


def identify_outfalls(
    graph: nx.MultiDiGraph[Any],
    outfall_derivation: parameters.OutfallDerivation,
    addresses: FilePaths | None = None,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Identify outfalls in a combined river-street graph.

    This function identifies outfalls in a combined river-street graph. An
    outfall is a node that is connected to a river and a street. Each street
    node within outfall_derivation.river_buffer_distance of a river
    centerline is paired with a synthetic ``river_outfall`` sink snapped
    onto that centerline - this provides a large set of plausible outfalls
    even where river features are noded sparsely. When
    ``outfall_derivation.include_water_body_outfalls`` is set, water-body
    polygons are also treated as outfall candidates (paired by polygon
    geometry). If there are no plausible outfalls for an entire subgraph, then
    a dummy river node is created and the lowest elevation node is paired with
    it. Any street->river/outfall link is given a `weight` and `length` of
    outfall_derivation.outfall_length, this is to ensure some penalty on the
    total number of outfalls selected.

    The retained outfalls are those selected by the minimum spanning tree
    (MST) of the combined street-river graph: rivers and the waste root
    fuse at zero cost, so ``outfall_length`` acts as a clustering radius —
    one outfall survives per street cluster connected by edges shorter
    than it.

    Args:
        graph: A directed multi-graph.
        outfall_derivation: An OutfallDerivation parameter object.
        addresses: File path manager, required to load water-body polygons
            when ``include_water_body_outfalls`` is set.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with outfall edges added.
    """
    graph = graph.copy()

    river_points, pipe_points = _get_points(graph)

    water_body_polys = _outfall_water_body_polys(graph, outfall_derivation, addresses)

    # On-line pond intakes: match each pond STORAGE node to its basin polygon
    # so _pair_rivers can pair the surrounding pipe nodes to the pond (routing
    # the catchment INTO the pond rather than past it to a regional outfall).
    pond_intake_pairs: list[tuple[Any, Any]] = []
    if outfall_derivation.online_pond_intake and addresses is not None:
        pond_intake_pairs = _pond_intake_pairs(
            graph,
            addresses,
            graph.graph.get("crs"),
            min_area_m2=outfall_derivation.pond_intake_min_area_m2,
        )
        if pond_intake_pairs:
            logger.info(
                f"online_pond_intake: pairing pipe nodes to {len(pond_intake_pairs)} "
                f"pond(s) within {outfall_derivation.pond_intake_buffer_m:.0f} m."
            )

    graph_ = _pair_rivers(
        graph,
        river_points,
        pipe_points,
        outfall_derivation.river_buffer_distance,
        outfall_derivation.outfall_length,
        water_body_polys=water_body_polys,
        pond_intake_pairs=pond_intake_pairs,
        pond_intake_buffer=outfall_derivation.pond_intake_buffer_m,
    )

    # Give river/water-body outfall sinks the receiving water's surface
    # elevation from the DEM (they're created after set_elevation, so
    # otherwise carry none — see _set_water_body_outfall_elevations).
    if addresses is not None:
        _set_water_body_outfall_elevations(graph_, addresses)

    # Set the length of the river edges to 0 - from a design perspective
    # once water is in the river we don't care about the length - since it
    # costs nothing
    for _, _, d in graph_.edges(data=True):
        if d["edge_type"] == "river":
            d["length"] = 0
            d["weight"] = 0

    # Add edges from the river nodes to a waste node
    graph_ = _root_nodes(graph_)

    result = _connect_mst_outfalls(graph_, graph)

    # IQR filter: flag outfalls whose street-side elevation is an outlier
    # relative to other outfalls in the same weakly connected component.
    # These are often "fake" dummy outfalls created in flat sub-basins
    # that can't realistically drain to that high point.
    outliers = _flag_iqr_outlier_outfalls(result)
    if outliers:
        logger.warning(
            f"{len(outliers)} outfall(s) flagged as elevation IQR outliers: "
            f"{outliers[:5]}{'...' if len(outliers) > 5 else ''}"
        )

    return result
