"""Graph utilities module for SWMManywhere-US."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import networkx as nx
import orjson as json
import shapely

if TYPE_CHECKING:
    GraphType = nx.Graph[Any] | nx.DiGraph[Any] | nx.MultiDiGraph[Any]
    GraphTypeVar = TypeVar("GraphTypeVar", nx.Graph[Any], nx.DiGraph[Any], nx.MultiDiGraph[Any])


def load_graph(fid: Path | str) -> GraphType:
    """Load a graph from a file saved with save_graph."""
    fid = Path(fid)
    if not fid.exists():
        msg = f"File {fid} does not exist!"
        raise FileNotFoundError(msg)
    graph = nx.node_link_graph(json.loads(fid.read_bytes()), directed=True, edges="edges")
    geom_dict = nx.get_edge_attributes(graph, "geometry")
    nx.set_edge_attributes(
        graph,
        dict(zip(geom_dict, shapely.from_wkt(list(geom_dict.values())))),
        "geometry",
    )
    return graph


def save_graph(graph: GraphType, fid: Path) -> None:
    """Save a graph to a file."""
    graph = graph.copy()
    geom_dict = nx.get_edge_attributes(graph, "geometry")
    nx.set_edge_attributes(
        graph,
        dict(zip(geom_dict, shapely.to_wkt(list(geom_dict.values())))),
        "geometry",
    )
    fid.write_bytes(
        json.dumps(nx.node_link_data(graph, edges="edges"), option=json.OPT_SERIALIZE_NUMPY)
    )


def filter_edges[GraphTypeVar: (nx.Graph[Any], nx.DiGraph[Any], nx.MultiDiGraph[Any])](
    graph: GraphTypeVar,
    keep_types: frozenset[str] = frozenset({"pipe"}),
) -> GraphTypeVar:
    """Filter a graph to keep only edges of specified types.

    Edges whose ``edge_type`` is not in *keep_types* are removed together
    with any nodes that become isolated.

    Outfall edges are treated specially: only the downstream node (``v``)
    is removed so the upstream street node survives.  Pond_connector
    edges are handled symmetrically on the other side: only the pond
    STORAGE node (``u`` by construction) is removed so the downstream
    network node survives.

    Args:
        graph: The input graph.
        keep_types: Edge types to retain.

    Returns:
        A copy of the graph with only the requested edge types.
    """
    graph = graph.copy()  # pyright: ignore[reportAssignmentType]
    nodes_to_remove: list[Any] = []
    edges_to_remove: list[tuple[Any, Any, Any]] = []
    for u, v, k, d in graph.edges(data=True, keys=True):  # pyright: ignore[reportCallIssue]
        if d["edge_type"] in keep_types:
            continue
        if d["edge_type"] == "outfall":
            nodes_to_remove.append(v)
        elif d["edge_type"] == "pond_connector":
            # Pond is the upstream (u) side; network node (v) must survive.
            pond_side = u if graph.nodes[u].get("node_type") == "water_body" else v
            nodes_to_remove.append(pond_side)
        elif d["edge_type"] == "street_channel":
            # Street channels are overlays parallel to street pipes and share
            # both endpoints with a surviving street edge.  Drop just the
            # channel edge; the endpoints are still part of the real pipe
            # network.
            edges_to_remove.append((u, v, k))
        else:
            nodes_to_remove.extend((u, v))
    for u, v, k in edges_to_remove:
        if graph.has_edge(u, v, key=k):  # pyright: ignore[reportCallIssue]
            graph.remove_edge(u, v, key=k)  # pyright: ignore[reportCallIssue]
    graph.remove_nodes_from(nodes_to_remove)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph
