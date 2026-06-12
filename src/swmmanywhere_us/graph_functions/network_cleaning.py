"""Module for graphfcns that change the plausible pipe location network."""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import networkx as nx
import shapely

from swmmanywhere_us.graph_functions.street import street_network_cleanup
from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from swmmanywhere_us.filepaths import FilePaths
    from swmmanywhere_us.parameters import SubcatchmentDerivation, TopologyDerivation


def assign_id(
    graph: nx.MultiGraph[Any] | nx.MultiDiGraph[Any], **kwargs: Any
) -> nx.MultiGraph[Any] | nx.MultiDiGraph[Any]:
    """Assign a string ID to each edge of the form ``"u-v"``.

    Existing unique IDs are preserved (for example the ``"{storage}-orifice"``
    and ``"{storage}-weir"`` IDs set during water-body integration).  Parallel
    edges that share ``(u, v)`` are disambiguated with an ``edge_type``
    suffix; anything that still collides is dropped as a duplicate.

    Args:
        graph: A multi-graph (directed or undirected).
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The same graph with an ID assigned to each edge.
    """
    edge_ids: set[str] = set()
    edges_to_remove = []
    for u, v, key, data in graph.edges(data=True, keys=True):
        existing_id = data.get("id")
        if existing_id:
            existing_id = str(existing_id)
            data["id"] = existing_id
            if existing_id not in edge_ids:
                edge_ids.add(existing_id)
                continue

        candidate = f"{u}-{v}"
        if candidate in edge_ids:
            edge_type = data.get("edge_type")
            if edge_type:
                candidate = f"{u}-{v}-{edge_type}"
        if candidate in edge_ids:
            edges_to_remove.append((u, v, key))
        else:
            data["id"] = candidate
            edge_ids.add(candidate)
    for u, v, key in edges_to_remove:
        graph.remove_edge(u, v, key)
    return graph


def remove_non_pipe_allowable_links(
    graph: nx.MultiDiGraph[Any], topology_derivation: TopologyDerivation, **kwargs: Any
) -> nx.MultiDiGraph[Any]:
    """Remove non-pipe allowable links.

    This function removes links that are not allowable for pipes. The non-
    allowable links are specified in the `omit_edges` attribute of the
    topology_derivation parameter. There two cases handled:

    1. The `highway` property of the edge. From OSM, `highway` is a category
        that contains the road type, e.g., motorway, trunk, primary. If the
        edge contains a value in the `highway` property that is in `omit_edges`,
        the edge is removed.

    2. Any other properties of the edge that are in `omit_edges`. If the
        property is not null in the edge data, the edge is removed. e.g.,
        if `bridge` is in `omit_edges` and the `bridge` entry of the edge
        is NULL, then the edge is retained, if it is something like 'yes',
        or 'viaduct' then the edge is removed.

    Args:
        graph: A directed multi-graph.
        topology_derivation: A TopologyDerivation parameter object.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with non-pipe allowable links removed.
    """
    edges_to_remove = set()
    for u, v, keys, data in graph.edges(data=True, keys=True):
        for omit in topology_derivation.omit_edges:
            if data.get("highway", None) == omit:
                edges_to_remove.add((u, v, keys))
    for edges in edges_to_remove:
        graph.remove_edge(*edges)
    return graph


def sum_over_delimiter(s: str | float) -> float:
    """Sum over a delimiter.

    This function takes a value, if it is not a string it is casted as a
    float, otherwise it sums over the numbers in the string. The
    numbers are separated by a delimiter. The delimiter is any non-numeric
    character. If the input is not a string, the function returns the input.

    Args:
        s (int | str | float): The input.

    Returns:
        float: The sum of the numbers in the string
    """
    if not isinstance(s, str):
        return float(s)
    return float(sum([int(num) for num in re.split(r"\D+", s) if num]))


def calculate_streetcover(
    graph: nx.MultiGraph[Any],
    subcatchment_derivation: SubcatchmentDerivation,
    addresses: FilePaths,
    **kwargs: Any,
) -> nx.MultiGraph[Any]:
    """Format the lanes attribute of each edge and calculates width.

    Args:
        graph: An undirected multi-graph.
        subcatchment_derivation: A SubcatchmentDerivation parameter object.
        addresses: A FilePaths parameter object.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph (unchanged, streetcover saved to file).
    """
    graph = graph.copy()
    lines = []
    for u, v, data in graph.edges(data=True):
        if data.get("edge_type", "pipe") != "pipe":
            continue
        lanes = data.get("lanes", 1)
        if isinstance(lanes, list):
            lanes = sum([sum_over_delimiter(x) for x in lanes])
        else:
            lanes = sum_over_delimiter(lanes)
        lines.append(
            {
                "geometry": data["geometry"].buffer(
                    lanes * subcatchment_derivation.lane_width,
                    cap_style=2,
                    join_style=2,
                ),
                "u": u,
                "v": v,
            }
        )
    gpd.GeoDataFrame(lines, crs=graph.graph["crs"]).to_parquet(addresses.model_paths.streetcover)
    return graph


def _flip_edge(u: Any, v: Any, data: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Reverse an edge's direction, reversing its geometry when present."""
    if "geometry" in data:
        data = {**data, "geometry": shapely.LineString(data["geometry"].coords[::-1])}
    return v, u, data


def double_directed(
    graph: nx.Graph[Any] | nx.MultiGraph[Any], **kwargs: Any
) -> nx.MultiDiGraph[Any]:
    """Convert an undirected graph to a directed graph with both flow directions.

    Street edges are doubled: for each undirected edge ``(u, v)`` the
    directed graph receives both ``u -> v`` (original geometry) and
    ``v -> u`` (reversed geometry).  This is essential because the
    Chahinian slope weight is asymmetric — downhill flow is preferred
    over uphill — so the topology derivation needs both candidate
    directions to choose the optimal one.

    River edges are single-directed and oriented downhill by node
    surface elevation: the upstream ``to_undirected`` step discards the
    source flow direction, so the (u, v) order arriving here is
    arbitrary and must be re-derived from the DEM.

    Args:
        graph: An undirected (multi-)graph.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        A directed multi-graph with doubled street edges.
    """
    G_new = nx.MultiDiGraph()
    G_new.graph = graph.graph.copy()

    # Copy all nodes
    for node, data in graph.nodes(data=True):
        G_new.add_node(node, **data)

    for src, dst, edge_data in graph.edges(data=True):
        edge_type = edge_data.get("edge_type", "pipe")
        u, v, data = src, dst, edge_data

        # Pond connectors must point storage -> network so the pond is an
        # ancestor of the outfall during topology derivation.  Flip when
        # the undirected iteration order happened to put the network side
        # first.
        if edge_type == "pond_connector" and graph.nodes[v].get("node_type") == "water_body":
            u, v, data = _flip_edge(u, v, data)

        # Rivers: orient downhill by node surface elevation.  The
        # to_undirected step upstream discarded the source flow
        # direction, so the (u, v) order here is arbitrary — re-derive
        # it from the DEM so the conduit flows the way water does.
        if edge_type == "river":
            eu = graph.nodes[u].get("surface_elevation")
            ev = graph.nodes[v].get("surface_elevation")
            if eu is not None and ev is not None and ev > eu:
                u, v, data = _flip_edge(u, v, data)

        # Forward edge (original geometry and direction)
        G_new.add_edge(u, v, **data)

        if edge_type != "pipe":
            # Rivers and other non-street edges are single-directed.
            continue

        # Reverse edge for streets (reversed geometry, natural "v-u" id).
        reverse_data = data.copy()
        reverse_data["id"] = f"{v}-{u}"
        if "geometry" in data:
            reverse_data["geometry"] = shapely.LineString(data["geometry"].coords[::-1])
        G_new.add_edge(v, u, **reverse_data)

    return G_new


def to_undirected(
    graph: nx.MultiGraph[Any] | nx.MultiDiGraph[Any], **kwargs: Any
) -> nx.MultiGraph[Any]:
    """Convert the graph to an undirected graph.

    Args:
        graph: A multi-graph (directed or undirected).
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        An undirected multi-graph.
    """
    # Don't use nx.MultiGraph.to_undirected! It enables multigraph if the geometries
    # are different, but we have already saved the street cover so don't
    # want this!
    return graph.to_undirected()


def _collapse_tiny_vertices(
    coords: list[tuple[float, ...]], min_len: float
) -> list[tuple[float, ...]]:
    """Drop intermediate vertices whose incoming segment is shorter than ``min_len``.

    Preserves the first and last vertex so the overall geometry endpoints
    don't move.  When shapely.segmentize is applied to a LineString that
    already contains a sub-``min_len`` sliver (e.g. a 60.0 m + 0.01 m
    two-vertex street inherited from OSM precision), segmentize keeps the
    sliver and we end up with near-zero-length SWMM conduits that produce
    thousands-of-percent slope artefacts once pipe_by_pipe assigns even a
    small invert drop.  Snapping the sliver to its neighbour avoids that
    entirely.
    """
    if len(coords) < 2:
        return list(coords)
    cleaned = [coords[0]]
    for i in range(1, len(coords) - 1):
        dx = coords[i][0] - cleaned[-1][0]
        dy = coords[i][1] - cleaned[-1][1]
        if (dx * dx + dy * dy) ** 0.5 >= min_len:
            cleaned.append(coords[i])
    # Final vertex: if the last-kept vertex is too close to the end, drop it
    # so the closing segment carries length from two-back instead of producing
    # another sliver.
    last = coords[-1]
    dx = last[0] - cleaned[-1][0]
    dy = last[1] - cleaned[-1][1]
    if (dx * dx + dy * dy) ** 0.5 < min_len and len(cleaned) > 1:
        cleaned.pop()
    cleaned.append(last)
    return cleaned


def split_long_edges(
    graph: nx.MultiGraph[Any],
    subcatchment_derivation: SubcatchmentDerivation,
    **kwargs: Any,
) -> nx.MultiGraph[Any]:
    """Split long edges into shorter edges.

    This function splits long edges into shorter edges. The edges are split
    into segments of length 'max_street_length'. The 'geometry' of the
    original edge must be a LineString. Intended to follow up with call of
    `merge_nodes`.

    Args:
        graph (nx.MultiGraph): A graph
        subcatchment_derivation (parameters.SubcatchmentDerivation): A
            SubcatchmentDerivation parameter object
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        graph (nx.MultiGraph): A graph
    """
    max_length = subcatchment_derivation.max_street_length
    # Segments shorter than this become degenerate SWMM conduits with
    # near-vertical slopes after pipe_by_pipe; collapse their bordering
    # vertex so no 0.01 m sliver survives segmentize.
    min_segment_len = 1.0

    # Separate pipe edges (to be split) from non-pipe edges (preserved as-is).
    # Edges without an edge_type attribute are treated as pipes.
    pipe_edges = [
        (u, v, d) for u, v, d in graph.edges(data=True) if d.get("edge_type", "pipe") == "pipe"
    ]
    other_edges = [
        (u, v, d) for u, v, d in graph.edges(data=True) if d.get("edge_type", "pipe") != "pipe"
    ]

    new_linestrings = shapely.segmentize([d["geometry"] for _, _, d in pipe_edges], max_length)

    new_edges = {}
    for new_linestring, (_, _, d) in zip(new_linestrings, pipe_edges):
        raw_coords = list(new_linestring.coords)
        coords = _collapse_tiny_vertices(raw_coords, min_segment_len)
        for start, end in itertools.pairwise(coords):
            if start == end:
                # Closed-loop LineStrings with only one survive vertex after
                # collapse produce self-loops with zero length — dropping
                # them matches ``_gdf_to_graph``'s closed-loop filter.
                continue
            geom = shapely.LineString([start, end])
            new_edges[(start, end, 0)] = {**d, "length": geom.length}

    # Preserve non-street edges
    for u, v, d in other_edges:
        geom = d.get("geometry")
        if geom:
            start = geom.coords[0]
            end = geom.coords[-1]
        else:
            start = (graph.nodes[u]["x"], graph.nodes[u]["y"])
            end = (graph.nodes[v]["x"], graph.nodes[v]["y"])
        new_edges[(start, end, 0)] = {**d}

    # Create new graph
    new_graph = nx.MultiGraph()
    new_graph.graph = graph.graph.copy()
    new_graph.add_edges_from(new_edges)
    nx.set_edge_attributes(new_graph, new_edges)
    all_nodes = set()
    for start, end, _ in new_edges:
        all_nodes.add(start)
        all_nodes.add(end)
    nx.set_node_attributes(new_graph, {node: {"x": node[0], "y": node[1]} for node in all_nodes})
    return nx.relabel_nodes(new_graph, {node: ix for ix, node in enumerate(new_graph.nodes)})


# Two pipe-edge endpoint nodes closer than this are a degenerate
# digitisation / edge-split artifact — real storm-drain reaches are tens
# of metres.  Such edges survive split_long_edges and then needlessly
# constrain the SWMM routing time step.
_MIN_PIPE_LENGTH_M = 5.0


def _short_pipe_pairs(graph: nx.MultiGraph[Any]) -> list[tuple[Any, Any]]:
    """Pipe edges whose endpoint nodes sit closer than ``_MIN_PIPE_LENGTH_M``."""
    pairs: list[tuple[Any, Any]] = []
    for u, v, d in graph.edges(data=True):
        if d.get("edge_type", "pipe") != "pipe":
            continue
        nu, nv = graph.nodes[u], graph.nodes[v]
        if "x" not in nu or "x" not in nv:
            continue
        if ((nu["x"] - nv["x"]) ** 2 + (nu["y"] - nv["y"]) ** 2) ** 0.5 < _MIN_PIPE_LENGTH_M:
            pairs.append((u, v))
    return pairs


def _cluster_extent_m(graph: nx.MultiGraph[Any], members: list[Any]) -> float | None:
    """Max pairwise node distance in a cluster, or None when a node lacks x/y."""
    coords: list[tuple[float, float]] = []
    for n in members:
        x, y = graph.nodes[n].get("x"), graph.nodes[n].get("y")
        if x is None or y is None:
            return None
        coords.append((float(x), float(y)))
    return max(
        ((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5
        for i in range(len(coords))
        for j in range(i + 1, len(coords))
    )


def merge_short_edges(graph: nx.MultiGraph[Any], **kwargs: Any) -> nx.MultiGraph[Any]:
    """Collapse degenerate sub-threshold pipe edges by contracting nodes.

    ``split_long_edges`` leaves behind pipe edges whose two endpoint
    nodes are only a few metres apart — split remainders and inherently
    short OSM street slivers.  These are not real storm-drain reaches
    and they constrain the SWMM routing time step.  Each connected
    cluster of pipe edges whose endpoints are closer than
    ``_MIN_PIPE_LENGTH_M`` is contracted to a single node.  A cluster
    whose nodes span more than three times the threshold is left alone:
    that is a genuine short chain, not a point artifact.  This is the
    ``merge_nodes`` follow-up that ``split_long_edges`` anticipates.

    Edge length is measured node-to-node: ``geometry`` still carries the
    un-split parent line until ``fix_geometries`` runs.

    Args:
        graph: An undirected multi-graph (post ``split_long_edges``).
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with sub-threshold pipe-edge clusters contracted.
    """
    graph = graph.copy()
    short_pairs = _short_pipe_pairs(graph)
    if not short_pairs:
        logger.info("merge_short_edges: no sub-threshold pipe edges found.")
        return graph

    clusters: nx.Graph[Any] = nx.Graph()
    clusters.add_edges_from(short_pairs)
    n_clusters = n_removed = n_skipped = 0
    for comp in list(nx.connected_components(clusters)):
        members = list(comp)
        extent = _cluster_extent_m(graph, members)
        if extent is None:
            continue
        if extent > 3.0 * _MIN_PIPE_LENGTH_M:
            n_skipped += 1
            continue
        rep = max(members, key=graph.degree)
        for m in members:
            if m != rep:
                nx.contracted_nodes(graph, rep, m, self_loops=False, copy=False)
                n_removed += 1
        n_clusters += 1

    # contracted_nodes stashes each merged-away node's data under a
    # "contraction" attribute on the survivor — drop it to keep the graph
    # cleanly serialisable.
    for _n, d in graph.nodes(data=True):
        d.pop("contraction", None)

    logger.info(
        f"merge_short_edges: contracted {n_clusters} cluster(s) of short pipe "
        f"edges, removed {n_removed} node(s); skipped {n_skipped} long chain(s)."
    )
    return graph


def divide_and_conquer(
    graph: nx.MultiDiGraph[Any],
    subcatchment_derivation: SubcatchmentDerivation,
    addresses: FilePaths,
    **kwargs: Any,
) -> nx.MultiGraph[Any]:
    """Divide the street network into major and local streets and clean up.

    Non-street edges (rivers) are preserved through the cleanup and
    reattached to the cleaned street graph.

    Args:
        graph: A directed multi-graph.
        subcatchment_derivation: A SubcatchmentDerivation parameter object.
        addresses: A FilePaths parameter object.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        An undirected multi-graph of cleaned streets plus preserved river edges.
    """
    # Extract non-street edges to preserve through cleanup
    river_edges = [
        (u, v, d.copy()) for u, v, d in graph.edges(data=True) if d.get("edge_type") != "pipe"
    ]
    river_nodes = {}
    for u, v, _d in river_edges:
        for n in (u, v):
            if n not in river_nodes:
                river_nodes[n] = dict(graph.nodes[n])

    # Clean only street edges
    street_edge_data = [
        {"geometry": d["geometry"], **{k: v for k, v in d.items() if k != "geometry"}}
        for _, _, d in graph.edges(data=True)
        if d.get("edge_type") == "pipe"
    ]
    if not street_edge_data:
        return graph.to_undirected()

    edges = gpd.GeoDataFrame(street_edge_data, crs=graph.graph["crs"])
    cleaned = street_network_cleanup(
        edges,
        subcatchment_derivation.buffer_size_local,
        subcatchment_derivation.min_hole_areasqm_local,
        subcatchment_derivation.buffer_size_major,
        subcatchment_derivation.min_hole_areasqm_major,
        subcatchment_derivation.dem_resolution,
        addresses.model_paths.model,
    )

    # Re-add river edges and their nodes
    for node, attrs in river_nodes.items():
        cleaned.add_node(node, **attrs)
    for u, v, d in river_edges:
        cleaned.add_edge(u, v, **d)

    return cleaned


def fix_geometries(
    graph: nx.MultiGraph[Any] | nx.MultiDiGraph[Any], **kwargs: Any
) -> nx.MultiGraph[Any] | nx.MultiDiGraph[Any]:
    """Fix the geometries of the edges.

    This function fixes the geometries of the edges. The geometries are
    recalculated from the node coordinates.

    Args:
        graph: A multi-graph (directed or undirected).
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with fixed edge geometries.
    """
    graph = graph.copy()
    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry", None)

        start_point_node = (graph.nodes[u]["x"], graph.nodes[u]["y"])
        end_point_node = (graph.nodes[v]["x"], graph.nodes[v]["y"])
        if not geom:
            start_point_edge = (None, None)
            end_point_edge = (None, None)
        else:
            start_point_edge = data["geometry"].coords[0]
            end_point_edge = data["geometry"].coords[-1]

        if (start_point_edge == end_point_node) & (end_point_edge == start_point_node):
            data["geometry"] = data["geometry"].reverse()
        elif (start_point_edge != start_point_node) | (end_point_edge != end_point_node):
            data["geometry"] = shapely.LineString([start_point_node, end_point_node])
    return graph


def remove_river_crossing_pipes(graph: nx.MultiDiGraph[Any], **kwargs: Any) -> nx.MultiDiGraph[Any]:
    """Remove pipe candidacy from street edges that cross a river centerline.

    Gravity storm mains do not cross canals — each bank drains to its own
    outfalls, and bridge streets carry traffic, not pipes.  Left in place,
    these crossings let the shortest-path topology route whole
    neighbourhoods over the water to the far bank (and pondshed
    delineation follows the pipes), producing cross-river pondsheds and
    outfalls on the wrong side.  Severing them splits the street graph at
    every canal; each bank then pairs to its own canal-snapped outfall
    sinks (or a dummy river) in ``identify_outfalls``.

    Nodes are kept (they remain subcatchment pour points); only the
    crossing pipe edges are dropped.  Edges without a geometry attribute
    are tested with the straight chord between their endpoints.

    Args:
        graph: A directed multi-graph containing pipe and river edges.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with river-crossing pipe edges removed.
    """
    river_geoms = [
        d["geometry"]
        for _, _, d in graph.edges(data=True)
        if d.get("edge_type") == "river" and d.get("geometry") is not None
    ]
    if not river_geoms:
        return graph
    river_tree = shapely.STRtree(river_geoms)

    graph = graph.copy()
    to_remove = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        if d.get("edge_type") != "pipe":
            continue
        geom = d.get("geometry")
        if geom is None:
            geom = shapely.LineString(
                [
                    (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                    (graph.nodes[v]["x"], graph.nodes[v]["y"]),
                ]
            )
        if any(geom.crosses(river_geoms[int(h)]) for h in river_tree.query(geom)):
            to_remove.append((u, v, k))
    graph.remove_edges_from(to_remove)
    if to_remove:
        logger.info(
            f"remove_river_crossing_pipes: removed {len(to_remove)} pipe "
            "edge(s) crossing a river centerline."
        )
    return graph
