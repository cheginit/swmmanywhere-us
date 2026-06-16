"""Utility functions for shortest path algorithms."""

from __future__ import annotations

import heapq
import math
from itertools import count
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
    for 180 and 90 deg is intentional, real street-aligned sewer
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
    # The two regimes meet at theta=135 with a DELIBERATE discontinuity (jump
    # from ~0.787 down to 0.20): Eq. 4 of Chahinian et al. (2019) is bimodal,
    # preferring both 90 deg (right-angle, cost 0.2) and 180 deg (straight,
    # cost 0) junctions, so the cost is not continuous across the regime split.
    if d90 < d180:  # right-angle regime: 30 <= theta < 135
        return 0.8 * d90 / 60.0 + 0.2
    return 0.4 * d180 / 90.0  # straight-through regime: 135 <= theta <= 180


def _turn_angle_deg(
    graph: nx.MultiDiGraph[Any], upstream: Any, node: Any, downstream: Any
) -> float | None:
    """Angle at *node* between pipes (upstream -> node) and (node -> downstream).

    Computed from node coordinates, pipes are treated as straight
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


# classical Dijkstra + turn-cost relaxation + path-trimming; easier to follow as one routine
def dijkstra_pq(  # noqa: C901
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
    computed directly on the initial graph", it only exists relative to
    the chosen continuation.  Evaluating it against the finalized tree
    successor preserves the one-out-edge-per-node forest contract that
    ``pipe_by_pipe`` requires (a full line-graph search would not).  The
    paper's secondary term, angles to other upstream branches already
    attached at the junction (their Algorithm 2), is omitted: in a
    label-setting search no sibling branch is final when a junction's
    label is fixed.

    NetworkX's Dijkstra variants (``multi_source_dijkstra`` et al.) are
    deliberately not used: their ``weight`` callable is memoryless,
    receiving only ``(u, v, data)`` for the single edge being relaxed, so
    it cannot see a node's already-finalized downstream successor and
    therefore cannot express the turn-angle transition cost.  Encoding
    turn costs for a stock search would require a line-graph expansion,
    which breaks the one-out-edge-per-node forest contract (see above).
    With ``angle_scaling == 0`` the search reduces to plain additive
    weights and ``multi_source_dijkstra`` on the reversed graph would
    suffice, but splitting into two code paths is not worth it.

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
    # Tentative shortest distance to the nearest outfall for every node.
    # Doubles as the on-pop staleness check and the no-path marker: a node
    # left at infinity never reached an outfall.
    shortest_paths = {node: float("inf") for node in graph.nodes}

    # Forest links toward the outfall.  Storing one back-pointer per node
    # (rather than the full path) keeps each relaxation O(1) instead of
    # copying a growing list; the path is implicit in the chain of preds.
    # ``root`` carries each node's owning outfall down the tree so it need
    # not be re-derived afterward.
    pred: dict[Hashable, Any] = {}
    root: dict[Hashable, Any] = {}

    # Set the shortest path length to 0 for outfalls
    for outfall in outfalls:
        shortest_paths[outfall] = 0
        root[outfall] = outfall

    # Predecessor adjacency, read straight off the graph for speed:
    # NetworkX's own Dijkstra reads ``G._adj`` for the same reason, whereas
    # ``in_edges(..., data=True)`` rebuilds an edge view on every pop.
    pred_adj = graph._pred  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]

    # Min-heap of (distance, tiebreak, node).  The monotonic counter breaks
    # distance ties so the heap never has to compare node objects, which
    # are not guaranteed to be orderable.
    counter = count()
    heap = [(0, next(counter), outfall) for outfall in outfalls]
    while heap:
        # Pop the node with the smallest distance
        dist, _, node = heapq.heappop(heap)
        if dist > shortest_paths[node]:
            continue  # stale heap entry; node already finalized cheaper

        # Downstream continuation of node's (final) path, for the
        # turn-angle transition cost.  Outfall nodes have none, the
        # first pipe into an outfall carries no turn cost.
        downstream = pred.get(node)

        # For each upstream neighbor; parallel edges collapse to their
        # cheapest, matching the original per-edge relaxation.
        for neighbor, keydict in pred_adj[node].items():
            # Calculate the distance through the current node
            alt_dist = dist + min(d[weight_attr] for d in keydict.values())
            if angle_scaling > 0 and downstream is not None:
                theta = _turn_angle_deg(graph, neighbor, node, downstream)
                if theta is not None:
                    alt_dist += angle_scaling * chahinian_angle_cost(theta)
            # If the alternative distance is shorter

            if alt_dist >= shortest_paths[neighbor]:
                continue

            # Update the shortest path length and forest links
            shortest_paths[neighbor] = alt_dist
            pred[neighbor] = node
            root[neighbor] = root[node]
            # Push the neighbor to the heap
            heapq.heappush(heap, (alt_dist, next(counter), neighbor))

    # Remove nodes with no path to an outfall (still at infinity)
    for node in [n for n, d in shortest_paths.items() if d == float("inf")]:
        graph.remove_node(node)
        del shortest_paths[node]

    if len(graph.nodes) == 0:
        msg = """No nodes with path to outfall, """
        raise ValueError(msg)

    # Annotate the surviving (reachable) nodes
    for node, dist in shortest_paths.items():
        graph.nodes[node]["outfall"] = root[node]
        graph.nodes[node]["shortest_path"] = dist

    # The forest edges are exactly the back-pointers: each non-outfall node
    # contributes its single (node -> predecessor) edge toward the outfall.
    edges_to_keep: set[Any] = set(pred.items())

    # Remove edges not on paths
    new_graph = graph.copy()
    for u, v in graph.edges():
        if (u, v) not in edges_to_keep:
            new_graph.remove_edge(u, v)

    return new_graph
