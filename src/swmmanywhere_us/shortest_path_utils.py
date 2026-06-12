"""Utility functions for shortest path algorithms."""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Hashable

    import networkx as nx


def chahinian_angle_cost(theta_deg: float) -> float:
    """Turn-angle cost of Chahinian et al. (2019), Eq. (4).

    ``theta_deg`` is the angle at a junction between the upstream and
    downstream pipe, folded onto [0, 180] (Eq. 4 is symmetric about
    180).  Straight-through flow (180 deg) is free; street-grid right
    angles (90 deg) carry a mild cost of 0.2; acute angles (< 30 deg,
    near-reversals) get the full penalty of 1.  The bimodal preference
    for 180 and 90 deg is intentional — real street-aligned sewer
    junctions cluster at right angles and straight-throughs.

    Reference:
        Chahinian N, Delenne C, Commandre B, Derras M, Deruelle L,
        Bailly J-S (2019). Automatic mapping of urban wastewater
        networks based on manhole cover locations. Computers,
        Environment and Urban Systems 78:101370.
        https://doi.org/10.1016/j.compenvurbsys.2019.101370
    """
    if theta_deg < 30:
        return 1.0
    d90 = abs(90.0 - theta_deg)
    d180 = abs(180.0 - theta_deg)
    if d90 < d180:  # right-angle regime: 30 <= theta < 135
        return 0.8 * d90 / 60.0 + 0.2
    return 0.4 * d180 / 90.0  # straight-through regime: 135 <= theta <= 180


def _turn_angle_deg(
    graph: nx.MultiDiGraph[Any], upstream: Any, node: Any, downstream: Any
) -> float | None:
    """Angle at *node* between pipes (upstream -> node) and (node -> downstream).

    Computed from node coordinates — pipes are treated as straight
    between junctions, as in Chahinian et al. (2019) where manholes are
    connected by straight Delaunay edges.  180 deg = flow continues
    straight through; 0 deg = full reversal.  Returns None when a
    coordinate is missing or two points coincide.
    """
    d_up = graph.nodes[upstream]
    d_node = graph.nodes[node]
    d_down = graph.nodes[downstream]
    if any("x" not in d or "y" not in d for d in (d_up, d_node, d_down)):
        return None
    v1 = (d_up["x"] - d_node["x"], d_up["y"] - d_node["y"])
    v2 = (d_down["x"] - d_node["x"], d_down["y"] - d_node["y"])
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return None
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)
    return math.degrees(math.acos(min(max(cos_a, -1.0), 1.0)))


def dijkstra_pq(  # noqa: C901, PLR0912 - classical Dijkstra + turn-cost relaxation + path-trimming; easier to follow as one routine
    graph: nx.MultiDiGraph[Any],
    outfalls: list[Any],
    weight_attr: str = "weight",
    angle_scaling: float = 0.0,
) -> nx.MultiDiGraph[Any]:
    """Dijkstra's algorithm for shortest paths to outfalls.

    This function calculates the shortest paths from each node in the graph to
    the nearest outfall. The graph is modified to include the outfall
    and the shortest path length.

    When ``angle_scaling`` > 0, a turn-angle transition cost (Chahinian
    et al. 2019, Eqs. 4-5) is added at relaxation time: extending the
    forest with pipe (neighbor -> node) costs an extra
    ``angle_scaling * C_theta`` where ``C_theta`` is evaluated against
    the angle between the candidate pipe and node's already-finalized
    downstream pipe.  Per the paper (Fig. 2), the angle cost "cannot be
    computed directly on the initial graph" — it only exists relative to
    the chosen continuation.  Evaluating it against the finalized tree
    successor preserves the one-out-edge-per-node forest contract that
    ``pipe_by_pipe`` requires (a full line-graph search would not).  The
    paper's secondary term — angles to other upstream branches already
    attached at the junction (their Algorithm 2) — is omitted: in a
    label-setting search no sibling branch is final when a junction's
    label is fixed.

    Args:
        graph (nx.MultiDiGraph): The input graph.
        outfalls (list): A list of outfall nodes.
        weight_attr (str): The name of the edge attribute containing the edge
            weights. Defaults to 'weight'.
        angle_scaling (float): Weight of the turn-angle transition cost
            (the paper's alpha_theta). 0 disables it.

    Returns:
        nx.MultiDiGraph[Any]: The graph with the shortest paths to outfalls.
    """
    graph = graph.copy()
    # Initialize the dictionary with infinity for all nodes
    shortest_paths = {node: float("inf") for node in graph.nodes}

    # Initialize the dictionary to store the paths
    paths: dict[Hashable, list[Any]] = {node: [] for node in graph.nodes}

    # Set the shortest path length to 0 for outfalls
    for outfall in outfalls:
        shortest_paths[outfall] = 0
        paths[outfall] = [outfall]

    # Initialize a min-heap with (distance, node) tuples
    heap = [(0, outfall) for outfall in outfalls]
    while heap:
        # Pop the node with the smallest distance
        dist, node = heapq.heappop(heap)
        if dist > shortest_paths[node]:
            continue  # stale heap entry; node already finalized cheaper

        # Downstream continuation of node's (final) path, for the
        # turn-angle transition cost.  Outfall nodes have none — the
        # first pipe into an outfall carries no turn cost.
        downstream = paths[node][-2] if len(paths[node]) >= 2 else None

        # For each neighbor of the current node
        for neighbor, _, edge_data in graph.in_edges(node, data=True):
            # Calculate the distance through the current node
            alt_dist = dist + edge_data[weight_attr]
            if angle_scaling > 0 and downstream is not None:
                theta = _turn_angle_deg(graph, neighbor, node, downstream)
                if theta is not None:
                    alt_dist += angle_scaling * chahinian_angle_cost(theta)
            # If the alternative distance is shorter

            if alt_dist >= shortest_paths[neighbor]:
                continue

            # Update the shortest path length
            shortest_paths[neighbor] = alt_dist
            # Update the path
            paths[neighbor] = paths[node] + [neighbor]
            # Push the neighbor to the heap
            heapq.heappush(heap, (alt_dist, neighbor))

    # Remove nodes with no path to an outfall
    for node in [node for node, path in paths.items() if not path]:
        graph.remove_node(node)
        del paths[node], shortest_paths[node]

    if len(graph.nodes) == 0:
        msg = """No nodes with path to outfall, """
        raise ValueError(msg)

    edges_to_keep: set[Any] = set()

    for path in paths.values():
        # Assign outfall
        outfall = path[0]
        for node in path:
            graph.nodes[node]["outfall"] = outfall
            graph.nodes[node]["shortest_path"] = shortest_paths[node]

        # Store path
        edges_to_keep.update(zip(path[1:], path[:-1]))

    # Remove edges not on paths
    new_graph = graph.copy()
    for u, v in graph.edges():
        if (u, v) not in edges_to_keep:
            new_graph.remove_edge(u, v)

    return new_graph
