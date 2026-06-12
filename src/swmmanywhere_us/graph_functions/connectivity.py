"""Connect disconnected pipe components via synthetic trunk pipes.

After :func:`derive_topology` builds a shortest-path forest, the pipe-only
subgraph in flat, OSM-fragmented terrain (e.g. coastal Florida flatwoods) typically breaks
into many weakly-connected components: each subdivision is one component,
and OSM doesn't include the trunk drainage that physically connects them
to the master canal.  The literature solution is a Steiner-tree inference
that proposes synthetic conduits to span the gaps (Haydar et al. 2019,
2026; Li et al. 2025; Chegini & Li 2022).

We use a *practical greedy-MST simplification* of that idea — Prim-style
augmentation: for each disconnected component, add one minimum-length
straight-line synthetic trunk to the nearest already-connected component,
seeded from the component containing the lowest-elevation real outfall
(typically the master canal's exit).

The synthetic trunks:

- get ``edge_type = "pipe"`` so the existing ``pipe_by_pipe`` sizing
  takes over,
- carry ``is_trunk = True`` so they can be identified for inspection,
- have direction set so flow runs from higher to lower
  ``surface_elevation`` (gravity),
- are skipped if the only candidate bridge exceeds
  ``TrunkInference.max_trunk_length_m`` — those components stay
  disconnected and rely on closed-basin pond treatment downstream
  (per the test catchment's coastal flat-terrain physical model).

References:
    Haydar S, Chahinian N, Pasquier P, Wittner C (2019). Optimal urban
        sewer layout design using Steiner tree problems. Engineering
        Optimization 51(11):1980.
    Haydar S, Chahinian N, Pasquier P (2026). Reconstructing Sewer
        Network Topology Using Graph Theory. Water 18(2):222.
    Chegini T, Li H-Y (2022). An algorithm for deriving the topology of
        belowground urban stormwater networks. HESS 26:4279.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import shapely

from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from swmmanywhere_us.filepaths import FilePaths
    from swmmanywhere_us.parameters import TrunkInference


def _pipe_components(graph: nx.MultiDiGraph[Any]) -> list[set[Any]]:
    """Weakly-connected components of the pipe-only subgraph, sized ≥ 1
    (singletons are dropped — they have no pipe edges and aren't part of
    a "drainage component" in the trunk-inference sense).
    """
    G = nx.MultiDiGraph()
    for u, v, k, d in graph.edges(keys=True, data=True):
        if d.get("edge_type") == "pipe":
            G.add_edge(u, v, key=k)
    return [c for c in nx.weakly_connected_components(G) if len(c) >= 2]


def _node_xy(graph: nx.MultiDiGraph[Any], n: Any) -> tuple[float, float] | None:
    x, y = graph.nodes[n].get("x"), graph.nodes[n].get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def _node_elev(graph: nx.MultiDiGraph[Any], n: Any) -> float | None:
    e = graph.nodes[n].get("surface_elevation")
    if e is None:
        e = graph.nodes[n].get("chamber_floor_elevation")
    if e is None:
        return None
    try:
        return float(e)
    except (TypeError, ValueError):
        return None


def _component_min_elev_node(
    graph: nx.MultiDiGraph[Any], comp: set[Any]
) -> tuple[Any, float] | None:
    """Pick the lowest-``surface_elevation`` node in a component (its
    natural drainage exit point and the right bridge attachment)."""
    best = None
    best_e = float("inf")
    for n in comp:
        e = _node_elev(graph, n)
        if e is None or e >= best_e:
            continue
        best_e = e
        best = n
    if best is None:
        return None
    return best, best_e


def _seed_component(  # noqa: C901, PLR0912 - three-priority short-circuit seed selection reads as one sequential routine
    graph: nx.MultiDiGraph[Any], components: list[set[Any]]
) -> int | None:
    """Pick the seed (already-canal-connected) component.

    Outfall sinks themselves are NOT inside pipe components — they're
    connected via ``outfall``-type edges from a pipe-network street node
    (the ``u`` of the outfall edge).  So we identify the seed by
    locating an outfall edge whose target sits at the canal's actual
    boundary exit, then return the pipe component containing its source.

    Priority (each step short-circuits if it finds a unique component):
      1. Outfall edges whose target is a ``node_type == "dummy_river"``
         sentinel.  ``_pair_rivers`` plants these at the *boundary* low
         point of subgraphs where no nearby river was matched — on the
         test catchment the district canal exits the bbox at exactly these locations
         (1249 / 1250, inverts set to ~ -0.5 m by ``post_processing``).
         If multiple dummy_river targets exist, prefer the source whose
         upstream surface elevation is lowest (most-downstream-like).
      2. Outfall edges whose target has the lowest set
         ``chamber_floor_elevation`` (real river-paired outfalls
         get their invert from ``set_elevation``).
      3. Otherwise, the largest pipe component (its outfall, wherever it
         goes, is the main drainage exit by Dijkstra topology).

    Returns the index into ``components`` or ``None`` if no seed.
    """
    # Pass 1: outfall edges to dummy_river sinks.
    dummy_candidates: list[tuple[Any, float]] = []
    for u, v, d in graph.edges(data=True):
        if d.get("edge_type") != "outfall":
            continue
        if graph.nodes[v].get("node_type") != "dummy_river":
            continue
        # rank by source's surface elevation (lowest = most likely the
        # actual canal exit at the bbox boundary).
        s = graph.nodes[u].get("surface_elevation")
        try:
            dummy_candidates.append((u, float(s) if s is not None else float("inf")))
        except (TypeError, ValueError):
            dummy_candidates.append((u, float("inf")))
    dummy_candidates.sort(key=lambda t: t[1])
    for street_node, _s in dummy_candidates:
        for i, comp in enumerate(components):
            if street_node in comp:
                return i

    # Pass 2: outfall edges to real-river sinks ranked by lowest invert.
    real_candidates: list[tuple[Any, float]] = []
    for u, v, d in graph.edges(data=True):
        if d.get("edge_type") != "outfall":
            continue
        if graph.nodes[v].get("node_type") == "dummy_river":
            continue
        inv_raw = graph.nodes[v].get("chamber_floor_elevation")
        if inv_raw is None:
            continue
        try:
            real_candidates.append((u, float(inv_raw)))
        except (TypeError, ValueError):
            continue
    real_candidates.sort(key=lambda t: t[1])
    for street_node, _inv in real_candidates:
        for i, comp in enumerate(components):
            if street_node in comp:
                return i

    # Pass 3: largest pipe component (most likely the main drainage area).
    if components:
        return max(range(len(components)), key=lambda i: len(components[i]))
    return None


def _component_outflow_nodes(graph: nx.MultiDiGraph[Any], comp: set[Any]) -> list[Any]:
    """Return the nodes in ``comp`` that are the local "outflow" — the
    sources of ``outfall``-type edges within the component.  These are
    the nodes where the component's pipe-by-pipe Dijkstra tree converges:
    *every* pipe in the component routes water to one of these.  Adding
    a trunk at any other node would skip the local Dijkstra topology and
    leave most of the component's water flowing to the original
    (perched) outfall instead of the trunk.
    """
    sources: list[Any] = []
    for u, _v, d in graph.edges(data=True):
        if d.get("edge_type") == "outfall" and u in comp:
            sources.append(u)
    return sources


def _bridge_candidates(
    graph: nx.MultiDiGraph[Any],
    src_candidates: list[Any],
    dst_nodes: set[Any],
    max_d_sq: float,
) -> list[tuple[float, Any, Any]]:
    """(d_sq, src, dst) pairs within range, for nodes that carry coordinates."""
    candidates: list[tuple[float, Any, Any]] = []
    for u in src_candidates:
        u_xy = _node_xy(graph, u)
        if u_xy is None:
            continue
        ux, uy = u_xy
        for v in dst_nodes:
            v_xy = _node_xy(graph, v)
            if v_xy is None:
                continue
            vx, vy = v_xy
            d_sq = (ux - vx) ** 2 + (uy - vy) ** 2
            if d_sq > max_d_sq:
                continue
            candidates.append((d_sq, u, v))
    return candidates


def _nearest_bridge(
    graph: nx.MultiDiGraph[Any],
    src_candidates: list[Any],
    dst_nodes: set[Any],
    max_length_m: float,
    river_geoms: list[Any] | None = None,
    river_tree: Any = None,
) -> tuple[Any, Any, float] | None:
    """Find the shortest Euclidean bridge from any node in
    ``src_candidates`` to any node in ``dst_nodes``, within
    ``max_length_m``.  ``src_candidates`` is intentionally a *restricted*
    set (typically the component's outfall street nodes) so the trunk
    attaches where the component's directed flow already converges.
    Bridges whose straight segment crosses a river centerline are
    rejected — a synthetic trunk must not carry drainage across a canal
    (each bank drains to its own outfalls).
    """
    candidates = _bridge_candidates(graph, src_candidates, dst_nodes, max_length_m * max_length_m)
    candidates.sort(key=lambda c: c[0])
    for d_sq, u, v in candidates:
        if river_tree is not None and river_geoms is not None:
            u_xy = _node_xy(graph, u)
            if u_xy is None:
                continue
            v_xy = _node_xy(graph, v)
            if v_xy is None:
                continue
            seg = shapely.LineString([u_xy, v_xy])
            if any(seg.crosses(river_geoms[int(h)]) for h in river_tree.query(seg)):
                continue
        return u, v, d_sq**0.5
    return None


def _orient_bridge(
    graph: nx.MultiDiGraph[Any], remaining_node: Any, connected_node: Any
) -> tuple[Any, Any]:
    """Orient the trunk so flow runs downhill by ``surface_elevation``.

    A trunk bridges a disconnected drainage cell to the canal-connected
    network; which end is hydraulically upstream is decided by gravity,
    not by which component happened to be bridged first.  The
    higher-ground node becomes the trunk source.  This honours the
    module-docstring contract — "direction set so flow runs from higher
    to lower ``surface_elevation`` (gravity)" — that the previous
    unconditional (remaining → connected) orientation silently broke:
    a trunk pointing uphill is reverse-flowed by DYNWAVE, dumping the
    whole component's discharge backward into the now-downstream node
    and out through its undersized outfall straw.

    Falls back to (remaining → connected) when either elevation is
    missing or the two are equal.  ``pipe_by_pipe``'s positive-slope
    enforcement still adjusts inverts if the local order is inverted.
    """
    e_remaining = _node_elev(graph, remaining_node)
    e_connected = _node_elev(graph, connected_node)
    if e_remaining is not None and e_connected is not None and e_connected > e_remaining:
        return connected_node, remaining_node
    return remaining_node, connected_node


def connect_pipe_components(  # noqa: C901, PLR0912, PLR0915 - greedy Prim-style augmentation loop with skip diagnostics is one algorithm
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    trunk_inference: TrunkInference,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Add synthetic trunk pipes to bridge disconnected pipe components.

    Runs after :func:`derive_topology` (so all pipe components are known)
    and before :func:`pipe_by_pipe` (so the new trunk pipes get sized
    through the normal design path).

    Greedy Prim-style augmentation: starting from the component that
    contains the lowest-elevation real outfall (the canal exit), in
    each round, attach the most cost-effective other component via a
    single straight-line synthetic pipe — the shortest Euclidean bridge
    from any node of the unconnected component to any already-connected
    node, capped by ``trunk_inference.max_trunk_length_m``.  Components
    farther than that cap stay disconnected — by physical design, those
    are isolated drainage cells whose ponds should remain closed-basin
    (handled later in ``finalize_pond_outlets``).

    The new edges are flagged ``is_trunk=True`` so they can be
    identified for inspection.  Diameter is left as a placeholder
    (``trunk_inference.placeholder_diameter_m``) — ``pipe_by_pipe``
    re-sizes them according to accumulated contributing area.
    """
    if not trunk_inference.enabled:
        logger.info("connect_pipe_components: disabled.")
        return graph

    components = _pipe_components(graph)
    if len(components) <= 1:
        logger.info(
            f"connect_pipe_components: pipe graph already has "
            f"{len(components)} component(s); nothing to bridge."
        )
        return graph

    seed_idx = _seed_component(graph, components)
    if seed_idx is None:
        logger.warning(
            "connect_pipe_components: could not pick a seed component "
            "(no real outfall and no elevations); skipping."
        )
        return graph

    graph = graph.copy()
    connected: set[Any] = set(components[seed_idx])
    remaining = [c for i, c in enumerate(components) if i != seed_idx]

    max_len = float(trunk_inference.max_trunk_length_m)

    # River centerlines: synthetic trunks must not carry drainage across a
    # canal, so bridge candidates whose segment crosses a river are skipped.
    river_geoms = [
        d["geometry"]
        for _, _, d in graph.edges(data=True)
        if d.get("edge_type") == "river" and d.get("geometry") is not None
    ]
    river_tree = shapely.STRtree(river_geoms) if river_geoms else None
    placeholder_diam = float(trunk_inference.placeholder_diameter_m)

    bridges_added = 0
    skipped_too_far = 0
    skipped_no_candidate = 0
    bridge_lengths_m: list[float] = []

    # Prim's-style: in each round, pick the comp closest to ``connected``.
    # The trunk's *source* in each remaining component is restricted to
    # the component's outfall street nodes — that's where the component's
    # internal Dijkstra tree routes water (the node where every pipe in
    # the component's downstream-path tree converges).  Attaching the
    # trunk elsewhere would create an island branch that no pipe routes
    # water toward.
    while remaining:
        best_round: tuple[int, Any, Any, float] | None = None
        for i, comp in enumerate(remaining):
            src_candidates = _component_outflow_nodes(graph, comp)
            if not src_candidates:
                # No outfall edge inside this component — use lowest-elev
                # node as the "natural" downstream attachment (fallback).
                info = _component_min_elev_node(graph, comp)
                src_candidates = [info[0]] if info is not None else list(comp)
            hit = _nearest_bridge(
                graph, src_candidates, connected, max_len, river_geoms, river_tree
            )
            if hit is None:
                continue
            u, v, length = hit
            if best_round is None or length < best_round[3]:
                best_round = (i, u, v, length)
        if best_round is None:
            # No comp can be bridged within max_len; we're done.
            for comp in remaining:
                src_candidates = _component_outflow_nodes(graph, comp) or list(comp)
                if (
                    _nearest_bridge(
                        graph, src_candidates, connected, max_len, river_geoms, river_tree
                    )
                    is None
                ):
                    if (
                        _nearest_bridge(
                            graph,
                            src_candidates,
                            connected,
                            float("inf"),
                            river_geoms,
                            river_tree,
                        )
                        is None
                    ):
                        skipped_no_candidate += 1
                    else:
                        skipped_too_far += 1
            break

        i_picked, u_node, v_node, length = best_round
        up_node, down_node = _orient_bridge(graph, u_node, v_node)

        # Synthesize the trunk pipe.  Geometry is a straight line; the
        # pipe_by_pipe step will set in/out offsets and pick diameters
        # based on accumulated contributing area.
        # Bridge endpoints always carry coordinates (_nearest_bridge only
        # considers nodes with x/y), so _node_xy cannot return None here.
        up_xy = cast("tuple[float, float]", _node_xy(graph, up_node))
        down_xy = cast("tuple[float, float]", _node_xy(graph, down_node))
        edge_id = f"{up_node}-{down_node}-trunk"
        graph.add_edge(
            up_node,
            down_node,
            edge_type="pipe",
            id=edge_id,
            is_trunk=True,
            length=max(length, 1.0),
            geometry=shapely.LineString([up_xy, down_xy]),
            diameter=placeholder_diam,
            contributing_area=0.0,
            weight=length,  # for any downstream weight-based logic
        )
        bridges_added += 1
        bridge_lengths_m.append(length)

        # Merge the newly-bridged component into the connected set.
        connected = connected | set(remaining[i_picked])
        remaining.pop(i_picked)

    if bridges_added:
        avg_len = sum(bridge_lengths_m) / len(bridge_lengths_m)
        max_b = max(bridge_lengths_m)
        logger.info(
            f"connect_pipe_components: added {bridges_added} synthetic trunk "
            f"pipe(s) (avg {avg_len:.0f} m, max {max_b:.0f} m); "
            f"{skipped_too_far} comp(s) skipped (> {max_len:.0f} m), "
            f"{skipped_no_candidate} comp(s) skipped (no coordinates)."
        )
    else:
        logger.info(
            f"connect_pipe_components: no bridges added "
            f"({skipped_too_far} comps > {max_len:.0f} m, "
            f"{skipped_no_candidate} no coords)."
        )
    return graph
