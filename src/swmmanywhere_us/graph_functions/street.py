"""Module for cleaning up street network geometries."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Literal, overload

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely import ops

from swmmanywhere_us.geospatial_utilities import simplify_geometry
from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from shapely import LineString, MultiLineString
    from shapely.geometry.base import BaseGeometry

    LineArray = NDArray[LineString]  # pyright: ignore[reportInvalidTypeForm]


__all__ = ["street_network_cleanup"]


def _collapse_lines(lines: list[LineString]) -> LineString:
    """Collapse multiple LineStrings into a single LineString by averaging coordinates."""
    n_pts = shapely.get_num_coordinates(lines).max()
    dist = np.linspace(0, 1, n_pts)
    coords = np.mean(
        [
            shapely.get_coordinates(shapely.line_interpolate_point(line, dist, normalized=True))
            for line in lines
        ],
        axis=0,
    )
    return shapely.LineString(coords)


@overload
def _lines_to_multigraph(
    multiline: LineString | MultiLineString | BaseGeometry | LineArray,
    edge_type: str,
    *,
    directed: Literal[False] = False,
    integer_labels: bool = ...,
    precision: float = 1e-3,
) -> nx.MultiGraph[Any]: ...


@overload
def _lines_to_multigraph(
    multiline: LineString | MultiLineString | BaseGeometry | LineArray,
    edge_type: str,
    *,
    directed: Literal[True],
    integer_labels: bool = ...,
    precision: float = 1e-3,
) -> nx.MultiDiGraph[Any]: ...


def _lines_to_multigraph(
    multiline: LineString | MultiLineString | BaseGeometry | LineArray,
    edge_type: str,
    *,
    directed: bool = False,
    integer_labels: bool = False,
    precision: float = 1e-3,
) -> nx.MultiGraph[Any] | nx.MultiDiGraph[Any]:
    """Convert a MultiLineString to an undirected multi-graph."""
    if not isinstance(multiline, shapely.LineString | shapely.MultiLineString | np.ndarray):
        msg = "multiline must be LineString or MultiLineString or array of LineString"
        raise TypeError(msg)

    geom = shapely.get_parts(multiline) if not isinstance(multiline, np.ndarray) else multiline

    if not (0 < precision < 1):
        msg = "precision must be 0 < precision < 1"
        raise ValueError(msg)
    geom = shapely.set_precision(geom, precision)
    geom = geom[~shapely.is_empty(geom)]
    graph = nx.MultiDiGraph()
    k = {}
    for line in geom:
        edge = (line.coords[0], line.coords[-1])
        k[edge] = k.get(edge, 0) + 1
        graph.add_node(edge[0], x=edge[0][0], y=edge[0][1])
        graph.add_node(edge[1], x=edge[1][0], y=edge[1][1])
        graph.add_edge(
            *edge, key=k[edge] - 1, geometry=line, length=line.length, edge_type=edge_type
        )

    if directed:
        if integer_labels:
            graph = nx.relabel.convert_node_labels_to_integers(graph)
        return graph  # pyright: ignore[reportReturnType]

    mgraph = graph.to_undirected()
    leaves = (n for n, d in nx.degree(mgraph) if d == 1)

    # ensure that for all leaves, the first point of their geometry is the leaf node
    edges_to_reverse = (
        (u, v, k, g)
        for n in leaves
        for u, v, k, g in mgraph.edges(n, keys=True, data="geometry")
        if not np.allclose(g.coords[0], n)
    )

    for u, v, k, g in edges_to_reverse:
        mgraph.edges[u, v, k]["geometry"] = g.reverse()

    if integer_labels:
        mgraph = nx.relabel.convert_node_labels_to_integers(mgraph)
    return mgraph  # pyright: ignore[reportReturnType]


def _lines_to_graph(lines: list[LineString] | LineArray) -> nx.Graph[Any]:
    """Convert a list of LineString to an undirected graph."""
    graph = nx.MultiDiGraph()
    k = {}
    for line in lines:
        edge = (line.coords[0], line.coords[-1])
        k[edge] = k.get(edge, 0) + 1
        graph.add_edge(*edge, key=k[edge] - 1, geometry=line)
    multi_edges: defaultdict[tuple[Any, Any], list[LineString]] = defaultdict(list)
    for (u, v, _), g in nx.get_edge_attributes(graph, "geometry").items():
        multi_edges[(u, v)].append(g)
    graph = nx.DiGraph()
    graph.add_edges_from(
        (u, v, {"geometry": _collapse_lines(g)}) for (u, v), g in multi_edges.items()
    )
    graph = graph.to_undirected()
    leaves = (n for n, d in nx.degree(graph) if d == 1)

    # ensure that for all leaves, the first point of their geometry is the leaf node
    edges_to_reverse = (
        (u, v, g)
        for n in leaves
        for u, v, g in graph.edges(n, data="geometry")
        if not np.allclose(g.coords[0], n)
    )

    for u, v, g in edges_to_reverse:
        graph.edges[u, v]["geometry"] = g.reverse()
    return graph


def _harmonize_direction(graph: nx.Graph[Any]) -> nx.DiGraph[Any]:
    """Convert an undirected graph to a directed graph.

    Notes:
    -----
    This function takes the first leaf node as the source and orients the edges
    in the graph so all other leaf nodes become terminals. The resulting graph
    is a directed graph with the same edges as the original graph.
    """
    if "geometry" not in list(next(iter(graph.edges(data=True)))[-1]):
        msg = "Input graph must have a 'geometry' attribute"
        raise ValueError(msg)

    source = next((node for node, degree in nx.degree(graph) if degree == 1), None)
    if source is None:
        msg = "Input graph must have at least one leaf node"
        raise ValueError(msg)

    digraph = nx.DiGraph()
    visited = set()
    stack = deque([source])
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    edata = graph.get_edge_data(current, neighbor).copy()
                    if not np.allclose(edata["geometry"].coords[0], current):
                        edata["geometry"] = edata["geometry"].reverse()
                    digraph.add_edge(current, neighbor, **edata)
                    stack.append(neighbor)
    return digraph


def _get_lines(graph: nx.Graph[Any]) -> LineArray:
    """Get geometry attributes of a graph and convert to a LineString or MultiLineString."""
    return np.array(list(nx.get_edge_attributes(graph, "geometry").values()))


def _street_line_cleanup(
    street_lines: list[LineString] | LineArray, buff_size: float, min_hole_area: float
) -> LineArray:
    """Clean up a street network by generating centerlines using Voronoi polygons."""
    if np.isclose([buff_size, min_hole_area], [0, 0]).any():
        return np.array(street_lines)

    street_lines = shapely.get_parts(ops.linemerge(shapely.union_all(street_lines)))  # pyright: ignore[reportArgumentType]
    polys = shapely.get_parts(shapely.union_all(shapely.buffer(street_lines, buff_size)))
    n_rings = shapely.get_num_interior_rings(polys)

    def _filter_holes(poly: shapely.Polygon, n: int) -> shapely.Polygon:
        holes = shapely.get_interior_ring(poly, range(n))
        holes = holes[shapely.area(shapely.polygons(holes)) >= min_hole_area]
        return shapely.Polygon(shapely.get_exterior_ring(poly), holes)

    buff = shapely.union_all([_filter_holes(p, n) for p, n in zip(polys, n_rings)])
    # We snap coordinates to a 1 cm grid to reduce tiny slivers and
    # near-coincident vertices introduced by buffer/union operations.
    # Purpose is to help prevent degenerate boundary rings that can cause
    # GEOS Voronoi construction to fail.
    buff = shapely.set_precision(buff, 1e-2)

    # Voronoi requires good enough number of points along the boundary
    # so we segmentize the boundary to have 4 times the number of points
    # along the boundaries
    distance = shapely.length(buff) / (shapely.get_num_coordinates(buff) * 4)
    poly = shapely.segmentize(buff, distance)
    voronoi_edges = shapely.voronoi_polygons(poly, only_edges=True)
    cl = shapely.get_parts(ops.linemerge(gpd.GeoSeries(voronoi_edges).clip(poly).union_all()))  # pyright: ignore[reportArgumentType]

    cl_nx = _harmonize_direction(_lines_to_graph(cl))
    terminals = {n for n, d in nx.degree(cl_nx) if d == 1}
    cl_nx.remove_nodes_from(terminals.difference(nx.dag_longest_path(cl_nx)))

    def _is_short(n: int, nb: int, threshold: float) -> bool:
        if cl_nx.has_edge(n, nb):
            return shapely.length(cl_nx.edges[(n, nb)]["geometry"]) < threshold
        return shapely.length(cl_nx.edges[(nb, n)]["geometry"]) < threshold

    cl_geoms = shapely.get_parts(ops.linemerge(shapely.union_all(_get_lines(cl_nx))))  # pyright: ignore[reportArgumentType]
    cl_nx = _harmonize_direction(_lines_to_graph(cl_geoms))
    threshold = 5 * buff_size
    short_terminals = ((n, next(nx.all_neighbors(cl_nx, n))) for n, d in nx.degree(cl_nx) if d == 1)
    short_terminals = [n for n, nb in short_terminals if _is_short(n, nb, threshold)]
    cl_nx.remove_nodes_from(short_terminals)
    return shapely.get_parts(ops.linemerge(_get_lines(cl_nx)))


def _attach_end_point(
    line: shapely.LineString,
    main_line: shapely.MultiLineString | shapely.LineString,
    is_start: bool,
) -> shapely.LineString:
    """Attach end point of a line to the nearest line in a multiline."""
    if is_start:
        _, proj = ops.nearest_points(shapely.get_point(line, 0), main_line)
        return shapely.LineString([proj.coords[0], *line.coords])
    _, proj = ops.nearest_points(shapely.get_point(line, -1), main_line)
    return shapely.LineString([*line.coords, proj.coords[0]])


def _clean_and_merge_lines(
    lines: LineArray, buffer_size_local: float, min_hole_areasqm_local: float
) -> LineArray:
    # Empty input (e.g. a small rural bbox that has no major-highway edges)
    # — return an empty array so downstream ops.union_all treats it as a no-op.
    if len(lines) == 0:
        return np.array([], dtype=object)
    graph = _lines_to_multigraph(ops.linemerge(lines), "pipe", integer_labels=True)
    components = [_get_lines(graph.subgraph(c)) for c in nx.connected_components(graph)]
    cleaned = [
        c
        if shapely.get_type_id(ops.linemerge(c)) == 1
        else _street_line_cleanup(c, buffer_size_local, min_hole_areasqm_local)
        for c in components
        if len(c) > 2
    ]
    if not cleaned:
        return np.array([], dtype=object)
    return np.concat(cleaned)


def _get_geometry(gdf: gpd.GeoDataFrame | gpd.GeoSeries) -> LineArray:
    lines = ops.linemerge(shapely.node(gdf.geometry.union_all()))
    lines = shapely.set_precision(shapely.get_parts(lines), 1e-3)
    return lines[~shapely.is_empty(lines)]


def _point_to_key(point: shapely.Point, precision: int = 3) -> tuple[float, float]:
    """Convert point to hashable key with given precision.

    Parameters
    ----------
    point : Point
        Point to convert
    precision : int, default 10
        Number of decimal places for rounding

    Returns:
    -------
    tuple of (float, float)
        Hashable coordinate tuple
    """
    return (round(point.x, precision), round(point.y, precision))


def _consolidate_parallel_lines(
    lines: list[shapely.LineString],
) -> list[shapely.LineString]:
    """Consolidate lines that share the same endpoints, keeping the longest.

    Parameters
    ----------
    lines : list of LineString
        Line segments to consolidate

    Returns:
    -------
    list of LineString
        Consolidated line segments
    """
    # Group lines by their endpoints (unordered)
    line_groups = {}

    for line in lines:
        start = _point_to_key(shapely.Point(line.coords[0]))
        end = _point_to_key(shapely.Point(line.coords[-1]))
        key = tuple(sorted([start, end]))

        if key not in line_groups:
            line_groups[key] = []
        line_groups[key].append(line)

    result = []
    for group in line_groups.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            best = max(group, key=lambda ln: ln.length)  # pyright: ignore[reportUnknownLambdaType]
            result.append(best)

    return result


def _merge_close_endpoints(  # noqa: C901, PLR0915 - union-find endpoint merge with nested find/union closures
    multilinestring: MultiLineString,
    distance_threshold: float,
) -> MultiLineString:
    """Merge endpoints that are closer than threshold.

    When endpoints are merged, all line segments using those endpoints are updated
    to use the merged point. If this creates multiple line segments between the
    same pair of points, only the longest is kept.

    Parameters
    ----------
    multilinestring : MultiLineString
        Input geometry with line segments
    distance_threshold : float
        Distance threshold for merging endpoints

    Returns:
    -------
    MultiLineString
        New MultiLineString with merged endpoints

    Notes:
    -----
    - Preserves network connectivity
    - Coordinates are snapped to merged locations
    - Line segments may be reoriented but geometry is preserved
    - Uses STRtree spatial index for efficient nearest neighbor search
    """
    lines = list(multilinestring.geoms)

    # Collect all unique endpoints
    endpoint_to_lines = {}  # point_key -> list of (line_idx, is_start, Point)
    unique_points = []
    point_to_key = {}

    for idx, line in enumerate(lines):
        start = shapely.Point(line.coords[0])
        end = shapely.Point(line.coords[-1])

        for point, is_start in [(start, True), (end, False)]:
            key = _point_to_key(point)

            if key not in endpoint_to_lines:
                endpoint_to_lines[key] = []
                unique_points.append(point)
                point_to_key[point] = key

            endpoint_to_lines[key].append((idx, is_start, point))

    tree = shapely.STRtree(unique_points)
    # Union-Find data structure for merging
    parent = {key: key for key in endpoint_to_lines}

    def find(key: tuple[float, float]) -> tuple[float, float]:
        """Find root with path compression."""
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(key1: tuple[float, float], key2: tuple[float, float]) -> None:
        """Union two sets."""
        root1 = find(key1)
        root2 = find(key2)
        if root1 != root2:
            parent[root2] = root1

    processed = set()
    for point in unique_points:
        key = point_to_key[point]
        if key in processed:
            continue

        nearby_indices = tree.query(point, predicate="dwithin", distance=distance_threshold)

        for idx in nearby_indices:
            nearby_point = unique_points[idx]
            nearby_key = point_to_key[nearby_point]

            # Skip self
            if key == nearby_key:
                continue

            if point.distance(nearby_point) < distance_threshold:
                union(key, nearby_key)
                processed.add(nearby_key)

        processed.add(key)

    merge_map = {}  # old_key -> canonical_key
    canonical_points = {}  # canonical_key -> Point

    for key in endpoint_to_lines:
        root = find(key)
        merge_map[key] = root

        # Use the first point as canonical
        if root not in canonical_points:
            canonical_points[root] = endpoint_to_lines[root][0][2]

    new_lines = []
    for line in lines:
        coords = list(line.coords)

        start_key = _point_to_key(shapely.Point(coords[0]))
        merged_key = merge_map[start_key]
        merged_point = canonical_points[merged_key]
        coords[0] = (merged_point.x, merged_point.y)

        end_key = _point_to_key(shapely.Point(coords[-1]))
        merged_key = merge_map[end_key]
        merged_point = canonical_points[merged_key]
        coords[-1] = (merged_point.x, merged_point.y)

        new_lines.append(shapely.LineString(coords))
    new_lines = _consolidate_parallel_lines(new_lines)
    return shapely.MultiLineString(new_lines)


def street_network_cleanup(
    edges: gpd.GeoDataFrame,
    buffer_size_local: float,
    min_hole_areasqm_local: float,
    buffer_size_highway: float,
    min_hole_areasqm_highway: float,
    merge_dist: float,
    save_dir: Path,
) -> nx.MultiGraph[Any]:
    """Clean up a street network.

    Parameters
    ----------
    edges : gpd.GeoDataFrame
        The input street edges with ``highway`` and ``geometry`` columns.
    buffer_size_local : float
        Buffer size for local streets in the same units as the graph's CRS.
    min_hole_areasqm_local : float
        Minimum hole area to fill for local streets in square meters.
    buffer_size_highway : float
        Buffer size for highways in the same units as the graph's CRS.
    min_hole_areasqm_highway : float
        Minimum hole area to fill for highways in square meters.
    merge_dist : float
        Distance threshold to merge nearby street endpoints in the same units as the graph's CRS.

    Returns:
    -------
    nx.Graph
        The cleaned street network graph.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cleaning up street network ...")
    edges = edges[["highway", "geometry"]].copy()

    list_idx = edges.highway.apply(pd.api.types.is_list_like)
    edges.loc[list_idx, "highway"] = edges.loc[list_idx, "highway"].apply(",".join)
    edges["highway"] = edges.highway.astype(str)
    raw_path = save_dir / "street_edges_raw.parquet"
    edges.to_parquet(raw_path)
    logger.info(f"Saved raw street edges to {raw_path}")

    logger.info("Removing non-drivable roads ...")
    edges = edges[~edges.highway.isin(["track", "footway", "path"])].copy()

    filtered_path = save_dir / "street_edges_filtered.parquet"
    edges.to_parquet(filtered_path)
    logger.info(f"Saved filtered street edges to {filtered_path}")

    logger.info("Separating local and major roads ...")
    main_msk = edges.highway.str.contains("motorway|trunk|primary|secondary")
    e_nonmain = edges.loc[~main_msk].geometry
    local_path = save_dir / "street_edges_local.parquet"
    gpd.GeoDataFrame(geometry=e_nonmain).to_parquet(local_path)
    logger.info(f"Saved local street edges to {local_path}")
    logger.info("Cleaning local roads ...")
    e_nonmain = _get_geometry(e_nonmain)
    local_street = _clean_and_merge_lines(e_nonmain, buffer_size_local, min_hole_areasqm_local)

    major_path = save_dir / "street_edges_major.parquet"
    e_main = edges.loc[main_msk].geometry
    gpd.GeoDataFrame(geometry=e_main).to_parquet(major_path)
    logger.info(f"Saved major street edges to {major_path}")
    logger.info("Cleaning major roads ...")
    e_main = _get_geometry(e_main)
    major_streets = _clean_and_merge_lines(e_main, buffer_size_highway, min_hole_areasqm_highway)
    major_streets = ops.linemerge(major_streets)

    logger.info("Merging local and major roads ...")
    msk = shapely.distance(shapely.get_point(local_street, 0), major_streets) < merge_dist
    local_street[msk] = [_attach_end_point(line, major_streets, True) for line in local_street[msk]]
    msk = shapely.distance(shapely.get_point(local_street, -1), major_streets) < merge_dist
    local_street[msk] = [
        _attach_end_point(line, major_streets, False) for line in local_street[msk]
    ]
    all_lines = ops.linemerge(shapely.node(shapely.union_all([*local_street, major_streets])))
    if isinstance(all_lines, shapely.LineString):
        all_lines = shapely.MultiLineString([all_lines])

    all_lines = simplify_geometry(all_lines, merge_dist)
    all_lines = _merge_close_endpoints(all_lines, merge_dist)

    clean_path = save_dir / "street_edges_cleaned.parquet"
    gpd.GeoDataFrame(geometry=shapely.get_parts(all_lines), crs=edges.crs).to_parquet(clean_path)
    logger.info(f"Saved cleaned street edges to {clean_path}")

    graph = _lines_to_multigraph(all_lines, "pipe", integer_labels=True)
    if edges.crs is None:
        msg = "edges GeoDataFrame must have a CRS set"
        raise ValueError(msg)
    graph.graph["crs"] = edges.crs.to_string()
    return graph
