"""Module for graphfcns that change subcatchments."""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import islice
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
import rasterio
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window
from scipy.interpolate import NearestNDInterpolator

from swmmanywhere_us import parameters, shortest_path_utils
from swmmanywhere_us.graph_utilities import filter_edges
from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from numpy.typing import NDArray
    from rasterio.io import DatasetReader
    from shapely import LineString

    from swmmanywhere_us.filepaths import FilePaths

    FloatArray = NDArray[np.float64]
    LineArray = NDArray[LineString]  # pyright: ignore[reportInvalidTypeForm]


def _transform_xy(
    dataset: DatasetReader, xy: Iterable[tuple[float, float]]
) -> Generator[tuple[float, float], None, None]:
    """Transform x, y coordinates to FRACTIONAL row, col pixel indices.

    Using ``op=lambda x: x`` preserves sub-pixel offsets so that two
    nearby query points resolve to distinct interpolated elevations
    instead of collapsing onto the same integer pixel.  Critical for
    accurate slope computation between closely-spaced graph nodes.
    """
    dt = dataset.transform
    _xy = iter(xy)
    while True:
        buf = tuple(islice(_xy, 0, 256))
        if not buf:
            break
        rows, cols = rowcol(dt, *zip(*buf), op=lambda v: v)
        yield from zip(rows, cols)


def sample_window(
    dataset: DatasetReader,
    xy: Iterable[tuple[float, float]],
    window: int = 5,
    indexes: int | list[int] | None = None,
    masked: bool = False,
    resampling: int = 1,
) -> Generator[FloatArray, None, None]:
    """Interpolate pixel values at given coordinates by interpolation.

    .. note::

        This function is adapted from
        the ``rasterio.sample.sample_gen`` function of
        `RasterIO <https://rasterio.readthedocs.io/en/latest/api/rasterio.sample.html#rasterio.sample.sample_gen>`__.

    Parameters
    ----------
    dataset : rasterio.DatasetReader
        Opened in ``"r"`` mode.
    xy : iterable
        Pairs of x, y coordinates in the dataset's reference system.
    window : int, optional
        Size of the window to read around each point. Must be odd.
        Default is 5.
    indexes : int or list of int, optional
        Indexes of dataset bands to sample, defaults to all bands.
    masked : bool, optional
        Whether to mask samples that fall outside the extent of the dataset.
        Default is ``False``.
    resampling : int, optional
        Resampling method to use. See rasterio.enums.Resampling for options.
        Default is 1, i.e., ``Resampling.bilinear``.

    Yields:
    ------
    numpy.array
        An array of length equal to the number of specified indexes
        containing the interpolated values for the bands corresponding to those indexes.
    """
    height = dataset.height
    width = dataset.width
    if indexes is None:
        indexes = dataset.indexes
    elif isinstance(indexes, int):
        indexes = [indexes]
    indexes = cast("list[int]", indexes)
    nodata = np.full(len(indexes), (dataset.nodata or 0), dtype=dataset.dtypes[0])
    if masked:
        mask_flags = [set(dataset.mask_flag_enums[i - 1]) for i in indexes]
        dataset_is_masked = any(
            {MaskFlags.alpha, MaskFlags.per_dataset, MaskFlags.nodata} & enums
            for enums in mask_flags
        )
        mask = [not (dataset_is_masked and enums == {MaskFlags.all_valid}) for enums in mask_flags]
        nodata = np.ma.array(nodata, mask=mask)

    if window % 2 == 0:
        msg = "window must be an odd integer"
        raise TypeError(msg)

    half_window = window // 2

    for row, col in _transform_xy(dataset, xy):
        if 0 <= row < height and 0 <= col < width:
            # Use float window offsets so rasterio's bilinear resampling
            # interpolates at the sub-pixel location, giving each query
            # point a distinct value even within the same pixel.
            col_start = max(0.0, col - half_window)
            row_start = max(0.0, row - half_window)
            data = dataset.read(
                indexes,
                window=Window(col_start, row_start, window, window),  # pyright: ignore[reportCallIssue]
                out_shape=(len(indexes), 1, 1),
                resampling=Resampling(resampling),
                masked=masked,
            )

            yield data[:, 0, 0]
        else:
            yield nodata


def _interpolate_na(x: FloatArray, y: FloatArray, z: FloatArray, k: int) -> FloatArray:
    """Interpolate NaN values in Z coordinates using k-nearest neighbors."""
    valid_mask = ~np.isnan(z)
    if not np.any(~valid_mask):
        return z

    if np.sum(valid_mask) < k:
        msg = f"Need at least {k} valid points for k-nearest interpolation"
        raise ValueError(msg)

    nan_mask = ~valid_mask
    interp = NearestNDInterpolator(np.c_[x[valid_mask], y[valid_mask]], z[valid_mask])
    z[nan_mask] = interp(np.c_[x[nan_mask], y[nan_mask]])
    return z


def set_elevation(
    graph: nx.MultiGraph[Any], addresses: FilePaths, **kwargs: Any
) -> nx.MultiGraph[Any]:
    """Set the elevation for each node.

    This function sets the elevation for each node. The elevation is
    calculated from the elevation data.

    Args:
        graph: An undirected multi-graph.
        addresses: A FilePaths parameter object.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with surface elevation set on nodes.
    """
    nodes, xys = zip(*[(n, (d["x"], d["y"])) for n, d in graph.nodes(data=True)])

    with rasterio.open(addresses.bbox_paths.elevation) as src:
        elevations = np.array(list(sample_window(src, xys))).ravel()
        bounds = src.bounds
        nodata = src.nodata

    # ``sample_window`` returns the DEM's nodata fill — or 0 when nodata is
    # unset — for points outside the raster.  The buffered street network can
    # extend past the downloaded DEM tile, so peripheral nodes would otherwise
    # keep a 0 m elevation that reads as a deep pit (forcing uphill flow and
    # distorting pipe inverts).  Mark out-of-bounds and true-nodata samples as
    # NaN so ``_interpolate_na`` fills them from the nearest valid neighbours.
    xy_arr = np.asarray(xys, dtype=float)
    out_of_bounds = (
        (xy_arr[:, 0] < bounds.left)
        | (xy_arr[:, 0] > bounds.right)
        | (xy_arr[:, 1] < bounds.bottom)
        | (xy_arr[:, 1] > bounds.top)
    )
    elevations[out_of_bounds] = np.nan
    if nodata is not None:
        elevations[elevations == nodata] = np.nan

    elev_res = dict(zip(nodes, np.c_[np.atleast_2d(xys), elevations]))
    x, y, z = zip(*elev_res.values(), strict=True)
    z = _interpolate_na(
        np.asarray(x),
        np.asarray(y),
        np.asarray(z),
        k=3,
    )
    elevations_dict = dict(zip(elev_res, z.tolist()))
    nx.set_node_attributes(graph, elevations_dict, "surface_elevation")

    # Default chamber_floor_elevation to surface - 3 m for all nodes.
    # pipe_by_pipe and assign_channel_geometry will overwrite this for
    # nodes they process (pipe junctions and river nodes).  Nodes that
    # survive to the .inp without being designed (e.g., outfall source
    # nodes with no street edges) keep this default instead of NaN.
    cfe = {n: z_val - 3.0 for n, z_val in elevations_dict.items()}
    nx.set_node_attributes(graph, cfe, "chamber_floor_elevation")
    return graph


def set_surface_slope(graph: nx.MultiDiGraph[Any], **kwargs: Any) -> nx.MultiDiGraph[Any]:
    """Set the surface slope for each edge.

    This function sets the surface slope for each edge. The surface slope is
    calculated from the elevation data.

    Args:
        graph: A directed multi-graph with surface elevation on nodes.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with surface slope set on edges.
    """
    graph = graph.copy()
    # Compute the slope for each edge
    slope_dict = {
        (u, v, k): (graph.nodes[u]["surface_elevation"] - graph.nodes[v]["surface_elevation"])
        / d["length"]
        for u, v, k, d in graph.edges(data=True, keys=True)
    }

    # Set the 'surface_slope' attribute for all edges
    nx.set_edge_attributes(graph, slope_dict, "surface_slope")
    return graph


def set_chahinian_slope(graph: nx.MultiDiGraph[Any], **kwargs: Any) -> nx.MultiDiGraph[Any]:
    """Set the Chahinian slope for each edge.

    This function sets the Chahinian slope for each edge. The Chahinian slope is
    calculated from the surface slope and weighted according to the slope
    (based on: https://doi.org/10.1016/j.compenvurbsys.2019.101370)

    Args:
        graph: A directed multi-graph with surface slope on edges.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with Chahinian slope set on edges.
    """
    graph = graph.copy()

    # Values where the weight of the slope can be matched to the values
    # in weights - e.g., a slope of 0.3% has 0 weight (preferred), while
    # a slope of <=-1% has a weight of 1 (not preferred)
    slope_points = [-1, 0.3, 0.7, 10]
    weights = [1, 0, 0, 1]

    # Calculate weights
    slope = nx.get_edge_attributes(graph, "surface_slope")
    weights = np.interp(
        np.asarray(list(slope.values())) * 100,
        slope_points,
        weights,
        left=1,
        right=1,
    )
    nx.set_edge_attributes(graph, dict(zip(slope, weights)), "chahinian_slope")

    return graph


def calculate_weights(
    graph: nx.MultiDiGraph[Any], topology_derivation: parameters.TopologyDerivation, **kwargs: Any
) -> nx.MultiDiGraph[Any]:
    """Calculate the weights for each edge.

    This function calculates the weights for each edge. The weights are
    calculated from the edge attributes.

    Args:
        graph: A directed multi-graph.
        topology_derivation: A TopologyDerivation parameter object.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        The graph with weights set on edges.
    """
    # Calculate bounds to normalise between
    bounds: dict[Any, list[float]] = defaultdict(lambda: [np.inf, -np.inf])

    for w in topology_derivation.weights:
        bounds[w][0] = min(nx.get_edge_attributes(graph, w).values())  # lower bound
        bounds[w][1] = max(nx.get_edge_attributes(graph, w).values())  # upper bound

    # Avoid division by zero
    bounds = {w: [b[0], b[1]] for w, b in bounds.items() if b[0] != b[1]}

    graph = graph.copy()
    eps = np.finfo(float).eps
    for _, _, d in graph.edges(data=True):
        total_weight = 0
        for attr, bds in bounds.items():
            # Normalise
            weight = max((d[attr] - bds[0]) / (bds[1] - bds[0]), eps)
            # Exponent
            weight = weight ** getattr(topology_derivation, f"{attr}_exponent")
            # Scaling
            weight = weight * getattr(topology_derivation, f"{attr}_scaling")
            # Sum
            total_weight += weight
        # Set
        d["weight"] = total_weight
    return graph


def _extract_non_street(
    graph: nx.MultiDiGraph[Any],
) -> tuple[dict[str, Any], dict[Any, Any], dict[str, Any]]:
    """Extract river, outfall, and pond-connector data before street-only filtering.

    Returns:
        river_data: ``{nodes: {node: attrs}, edges: {(u,v,k): attrs}}`` for
            all river edges and their endpoint nodes.
        outfall_data: ``{(u,v,k): attrs}`` for all outfall edges, keyed by
            ``(street_node, river_node, key)``.
        pond_data: ``{nodes: {node: attrs}, edges: {(u,v,k): attrs}}`` for
            pond_connector edges and any water-body STORAGE or canal-snapped
            ``river_outfall`` sink nodes they reference.  Both are preserved
            out-of-band the same way river nodes are, so topology derivation
            can't prune them.
    """
    river_nodes: dict[Any, Any] = {}
    river_edges: dict[Any, Any] = {}
    outfall_edges: dict[Any, Any] = {}
    pond_nodes: dict[Any, Any] = {}
    pond_edges: dict[Any, Any] = {}

    for u, v, k, d in graph.edges(data=True, keys=True):
        et = d["edge_type"]
        if et == "river":
            river_edges[(u, v, k)] = d.copy()
            for n in (u, v):
                if n not in river_nodes:
                    river_nodes[n] = dict(graph.nodes[n])
        elif et == "outfall":
            outfall_edges[(u, v, k)] = d.copy()
            # Preserve the river-side node attributes so dummy-river nodes
            # created for basins with no nearby river can be restored after
            # the street-only Dijkstra pass.  Without this, outfall edges
            # get dropped for small inland bboxes that have no real river
            # network.
            if v not in river_nodes:
                river_nodes[v] = dict(graph.nodes[v])
        elif et == "pond_connector":
            pond_edges[(u, v, k)] = d.copy()
            for n in (u, v):
                # Preserve the pond storage and, for canal-anchored ponds,
                # the snapped sink on the river centerline — neither lies on
                # a pipe path, so both would otherwise be pruned.
                if (
                    graph.nodes[n].get("node_type") in ("water_body", "river_outfall")
                    and n not in pond_nodes
                ):
                    pond_nodes[n] = dict(graph.nodes[n])

    return (
        {"nodes": river_nodes, "edges": river_edges},
        outfall_edges,
        {"nodes": pond_nodes, "edges": pond_edges},
    )


def _reattach_rivers(  # noqa: C901 - order-dependent sequential re-add of river/outfall/pond edges; splitting hides the coupling
    graph: nx.MultiDiGraph[Any],
    river_data: dict[str, Any],
    outfall_edges: dict[Any, Any],
    pond_data: dict[str, Any],
) -> nx.MultiDiGraph[Any]:
    """Re-add river, outfall, and pond_connector edges after topology derivation.

    Only outfall edges whose street-side node (``u``) survived topology
    derivation are reattached, linking the pipe network to the river
    network.  Pond_connector edges are reattached whenever their
    ``downstream_network`` endpoint is present in the reattached graph —
    a street node that survived derivation or a re-added river node; the
    pond storage node itself is preserved in ``pond_data`` and re-added
    here.
    """
    surviving_nodes = set(graph.nodes)

    # Re-add river nodes and edges
    for node, attrs in river_data["nodes"].items():
        graph.add_node(node, **attrs)
    for (u, v, k), attrs in river_data["edges"].items():
        graph.add_edge(u, v, key=k, **attrs)

    # Re-add outfall edges where the street-side node survived
    for (u, v, k), attrs in outfall_edges.items():
        if u in surviving_nodes:
            if v not in graph.nodes and v in river_data["nodes"]:
                graph.add_node(v, **river_data["nodes"][v])
            if v in graph.nodes:
                graph.add_edge(u, v, key=k, **attrs)

    # Re-add pond storage nodes, then reattach pond_connector edges where
    # the network-side endpoint is present in the reattached graph.  The
    # test must run against ``graph.nodes`` (street survivors plus the
    # rivers re-added above), not ``surviving_nodes``: that snapshot
    # predates the river re-add, so a river-anchored connector could
    # never pass it and every pond anchored to a river was silently
    # dropped.  Street anchors pruned by derivation were removed from the
    # graph entirely and still fail this test.  The pond node itself is a
    # source, so we always bring it back.
    for node, attrs in pond_data["nodes"].items():
        if node not in graph.nodes:
            graph.add_node(node, **attrs)
    for (u, v, k), attrs in pond_data["edges"].items():
        pond_end = (
            u if attrs.get("edge_type") == "pond_connector" and u in pond_data["nodes"] else v
        )
        net_end = v if pond_end == u else u
        if net_end in graph.nodes:
            graph.add_edge(u, v, key=k, **attrs)

    return graph


def derive_topology(
    graph: nx.MultiDiGraph[Any],
    topology_derivation: parameters.TopologyDerivation,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Derive the topology of a graph.

    Derives the network topology based on the weighted graph of potential
    pipe carrying edges in the graph.  River edges are extracted before
    topology derivation (they already have correct topology from OSM) and
    reattached afterward together with the outfall edges that connect the
    pipe network to the river network.

    Topology is a Dijkstra shortest-path forest: each node follows its
    least-``weight`` path to the nearest outfall selected by
    :func:`identify_outfalls`.  When
    ``topology_derivation.chahinian_angle_scaling`` > 0, a turn-angle
    transition cost (Chahinian et al. 2019, Eq. 4) is added as the
    forest grows, favouring straight-through and right-angle junctions
    over acute turns — see :func:`shortest_path_utils.dijkstra_pq`.

    Args:
        graph: A directed multi-graph with weights on edges.
        topology_derivation: A TopologyDerivation parameter object
            (supplies the turn-angle cost weight).
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        A directed multi-graph with the derived topology, including
        retained river and outfall edges.
    """
    graph = graph.copy()

    # Extract river, outfall, and pond_connector subgraphs before filtering
    river_data, outfall_edges, pond_data = _extract_non_street(graph)

    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(original_limit, len(graph.nodes)))

    outfalls = [u for u, _, d in graph.edges(data=True) if d["edge_type"] == "outfall"]
    visited: set[Any] = set(outfalls)
    for outfall in outfalls:
        visited = visited | set(nx.ancestors(graph, outfall))
    graph.remove_nodes_from(set(graph.nodes) - visited)
    graph = filter_edges(graph, frozenset({"pipe"}))

    if nx.negative_edge_cycle(graph, weight="weight"):
        logger.warning("Graph contains negative cycle")

    graph = shortest_path_utils.dijkstra_pq(
        graph, outfalls, angle_scaling=topology_derivation.chahinian_angle_scaling
    )

    sys.setrecursionlimit(original_limit)

    # Reattach river, outfall, and pond_connector edges
    graph = _reattach_rivers(graph, river_data, outfall_edges, pond_data)

    pipe_weight = sum(
        d.get("weight", 0) for _, _, d in graph.edges(data=True) if d["edge_type"] == "pipe"
    )
    logger.info(f"Total pipe graph weight {pipe_weight:.2f}.")
    logger.info(
        f"Retained {len(river_data['edges'])} river edges, "
        f"{sum(1 for u, _, _ in outfall_edges if u in graph.nodes)} outfall edges."
    )

    return graph
