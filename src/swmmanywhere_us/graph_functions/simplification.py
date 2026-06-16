"""Network simplification for reducing SWMM model complexity.

Consolidates degree-2 pipe chains and removes small dangling leaves to
reduce element count while preserving network topology and hydraulic
capacity.  Based on the approach described in:

    Pichler et al. (2024), "Fully automated simplification of urban
    drainage models on a city scale", Water Science & Technology 90(9).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import shapely

from swmmanywhere_us import geospatial_utilities as go
from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    import networkx as nx

    from swmmanywhere_us.filepaths import FilePaths
    from swmmanywhere_us.parameters import SimplificationParams


_PIPE_TYPES = frozenset({"pipe"})
_NON_PIPE_TYPES = frozenset({"river", "outfall", "orifice", "weir", "pond_inflow", "pond_outflow"})
_SPECIAL_NODE_TYPES = frozenset({"water_body", "outlet_junction"})


# ── Protected node detection ────────────────────────────────────────


def _find_protected_nodes(graph: nx.MultiDiGraph[Any]) -> set[Any]:
    """Return the set of nodes that must not be removed."""
    protected: set[Any] = set()

    for u, v, d in graph.edges(data=True):
        etype = d.get("edge_type", "pipe")
        if etype in _NON_PIPE_TYPES:
            protected.add(u)
            protected.add(v)

    for n, d in graph.nodes(data=True):
        if d.get("node_type") in _SPECIAL_NODE_TYPES:
            protected.add(n)

    # Branching / merging points in the street subnetwork
    for n in graph.nodes():
        if n in protected:
            continue
        street_in = sum(
            1
            for _, _, d in graph.in_edges(n, data=True)
            if d.get("edge_type", "pipe") in _PIPE_TYPES
        )
        street_out = sum(
            1
            for _, _, d in graph.out_edges(n, data=True)
            if d.get("edge_type", "pipe") in _PIPE_TYPES
        )
        if street_in != 1 or street_out != 1:
            protected.add(n)

    return protected


# ── Dangling leaf removal ───────────────────────────────────────────


def _is_structurally_protected(graph: nx.MultiDiGraph[Any], node: Any) -> bool:
    """Check whether a node is structurally important (not removable).

    A node is structurally protected if it is connected to a non-street
    edge, has a special ``node_type``, or is a branching/merging point
    (street in-degree > 1 or street out-degree > 1).  Unlike
    ``_find_protected_nodes`` this does **not** protect leaf start nodes
    (street in-degree == 0) so that dangling leaves can be pruned.
    """
    for _, _, d in graph.in_edges(node, data=True):
        if d.get("edge_type", "pipe") in _NON_PIPE_TYPES:
            return True
    for _, _, d in graph.out_edges(node, data=True):
        if d.get("edge_type", "pipe") in _NON_PIPE_TYPES:
            return True
    if graph.nodes[node].get("node_type") in _SPECIAL_NODE_TYPES:
        return True
    street_in = sum(
        1
        for _, _, d in graph.in_edges(node, data=True)
        if d.get("edge_type", "pipe") in _PIPE_TYPES
    )
    street_out = sum(
        1
        for _, _, d in graph.out_edges(node, data=True)
        if d.get("edge_type", "pipe") in _PIPE_TYPES
    )
    return bool(street_in > 1 or street_out > 1)


def _walk_leaf_chain(graph: nx.MultiDiGraph[Any], leaf: Any) -> list[Any]:
    """Follow the single street out-edge from *leaf* downstream.

    Stops at a structurally protected node, a branching point, or a node
    with more than one street in-edge.  Returns the removable chain
    (leaf first).
    """
    chain = [leaf]
    current = leaf
    while True:
        street_out = [
            (v, k, d)
            for _, v, k, d in graph.out_edges(current, keys=True, data=True)
            if d.get("edge_type", "pipe") in _PIPE_TYPES
        ]
        if len(street_out) != 1:
            break
        nxt = street_out[0][0]
        if _is_structurally_protected(graph, nxt):
            break
        nxt_street_in = sum(
            1
            for _, _, d in graph.in_edges(nxt, data=True)
            if d.get("edge_type", "pipe") in _PIPE_TYPES
        )
        if nxt_street_in != 1:
            break
        chain.append(nxt)
        current = nxt
    return chain


def _remove_dangling_leaves(
    graph: nx.MultiDiGraph[Any],
    min_area: float,
) -> tuple[nx.MultiDiGraph[Any], dict[Any, Any]]:
    """Remove leaf chains whose total contributing area is below *min_area*.

    A leaf chain starts at a node with zero street in-edges and follows
    the single outgoing street edge downstream until reaching a
    structurally important node (branching point, river-connected, etc.).

    Returns the modified graph and a mapping from each removed node to
    the surviving downstream node it was redirected to.
    """
    removed_map: dict[Any, Any] = {}
    changed = True
    while changed:
        changed = False
        leaves = []
        for n in list(graph.nodes()):
            if n not in graph:
                continue
            street_in = sum(
                1
                for _, _, d in graph.in_edges(n, data=True)
                if d.get("edge_type", "pipe") in _PIPE_TYPES
            )
            street_out_list = [
                (v, k, d)
                for _, v, k, d in graph.out_edges(n, keys=True, data=True)
                if d.get("edge_type", "pipe") in _PIPE_TYPES
            ]
            if street_in == 0 and len(street_out_list) == 1:
                leaves.append(n)

        for leaf in leaves:
            if leaf not in graph:
                continue
            chain = _walk_leaf_chain(graph, leaf)

            # Find surviving downstream node
            street_out = [
                (v, k, d)
                for _, v, k, d in graph.out_edges(chain[-1], keys=True, data=True)
                if d.get("edge_type", "pipe") in _PIPE_TYPES
            ]
            if not street_out:
                continue
            surviving = street_out[0][0]

            total_ca = sum(graph.nodes[n].get("contributing_area", 0.0) for n in chain)
            if total_ca >= min_area:
                continue

            for n in chain:
                removed_map[n] = surviving
                graph.remove_node(n)
            changed = True

    return graph, removed_map


# ── Degree-2 chain detection ───────────────────────────────────────


def _find_degree2_chains(graph: nx.MultiDiGraph[Any], protected: set[Any]) -> list[list[Any]]:
    """Find all maximal chains of degree-2 street nodes.

    Each chain is a list ``[P_start, n1, n2, ..., nk, P_end]`` where
    the first and last elements are protected endpoints and the middle
    elements are unprotected degree-2 nodes to be consolidated.
    """
    visited: set[Any] = set()
    chains: list[list[Any]] = []

    # Candidates: non-protected nodes with exactly 1 street in + 1 street out
    candidates = set(graph.nodes()) - protected

    for start_protected in list(protected):
        # Walk each outgoing street edge into a potential chain
        for _, first, _, d in graph.out_edges(start_protected, keys=True, data=True):
            if d.get("edge_type", "pipe") not in _PIPE_TYPES:
                continue
            if first not in candidates or first in visited:
                continue

            chain = [start_protected, first]
            visited.add(first)
            current = first

            while True:
                street_out = [
                    (v, k, ed)
                    for _, v, k, ed in graph.out_edges(current, keys=True, data=True)
                    if ed.get("edge_type", "pipe") in _PIPE_TYPES
                ]
                if len(street_out) != 1:
                    break
                nxt = street_out[0][0]
                if nxt in protected:
                    chain.append(nxt)
                    break
                if nxt in visited or nxt not in candidates:
                    chain.append(nxt)
                    break
                visited.add(nxt)
                chain.append(nxt)
                current = nxt

            # Valid chain: at least one interior node to remove
            if len(chain) >= 3 and chain[-1] in protected:
                chains.append(chain)

    return chains


def _merge_edge_properties(edges: list[dict[str, Any]], total_length: float) -> dict[str, Any]:
    """Compute merged properties for a consolidated chain."""
    lengths = [float(d.get("length", 0)) for d in edges]

    # Diameter: the governing (largest) segment.  A degree-2 chain is
    # diameter-monotonic (non-decreasing downstream) and the merged conduit
    # carries the chain's full SUMMED contributing area, so it must inherit the
    # downstream-most (largest) diameter; min() would undersize it.
    diameters = [float(d.get("diameter", 0.3)) for d in edges]
    diameter = max(diameters)

    # Roughness: length-weighted average
    roughnesses = [float(d.get("roughness", 0.01)) for d in edges]
    roughness = sum(n * ln for n, ln in zip(roughnesses, lengths)) / total_length

    # Contributing area: sum
    ca = sum(float(d.get("contributing_area", 0)) for d in edges)

    # Offsets: first edge's in_offset, last edge's out_offset
    in_offset = float(edges[0].get("in_offset", 0))
    out_offset = float(edges[-1].get("out_offset", 0))

    return {
        "length": total_length,
        "diameter": diameter,
        "roughness": roughness,
        "contributing_area": ca,
        "in_offset": in_offset,
        "out_offset": out_offset,
        "edge_type": "pipe",
    }


def _merge_geometries(geoms: list[shapely.LineString]) -> shapely.LineString:
    """Merge a sequence of LineStrings into a single LineString."""
    coords: list[Any] = []
    for i, g in enumerate(geoms):
        c = list(g.coords)
        if i == 0:
            coords.extend(c)
        else:
            # Skip the first point (shared with previous segment's last)
            coords.extend(c[1:])
    return shapely.LineString(coords)


def _remove_chain_edges(graph: nx.MultiDiGraph[Any], chain: list[Any]) -> None:
    """Remove the street edges linking consecutive chain nodes."""
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i + 1]
        if graph.has_node(u) and graph.has_node(v):
            for k in list(graph[u].get(v, {}).keys()):
                if graph[u][v][k].get("edge_type", "pipe") in _PIPE_TYPES:
                    graph.remove_edge(u, v, k)


def _add_split_edges(
    graph: nx.MultiDiGraph[Any],
    chain: list[Any],
    edge_data_list: list[dict[str, Any]],
    merged_props: dict[str, Any],
    max_length: float,
) -> set[Any]:
    """Split a long merged chain into segments <= *max_length*.

    Retains enough interior nodes from the original chain to serve as
    segment endpoints.  Returns the set of retained interior node IDs
    (caller must remove the rest).
    """
    total_length = merged_props["length"]
    n_segments = int(np.ceil(total_length / max_length))
    target_seg_len = total_length / n_segments

    # Walk along the chain accumulating length, pick split points
    cumulative = 0.0
    split_indices: list[int] = []
    next_threshold = target_seg_len

    for i, ed in enumerate(edge_data_list[:-1]):
        cumulative += float(ed.get("length", 0))
        if cumulative >= next_threshold - 1e-6:
            split_indices.append(i + 1)
            next_threshold += target_seg_len

    p_start = chain[0]
    p_end = chain[-1]
    segment_nodes = [p_start]
    segment_nodes.extend(chain[idx] for idx in split_indices if chain[idx] != p_end)
    segment_nodes.append(p_end)

    retained: set[Any] = set(segment_nodes) - {p_start, p_end}

    # Remove old edges between all nodes in the chain
    _remove_chain_edges(graph, chain)

    # Add merged segment edges
    for seg_i in range(len(segment_nodes) - 1):
        seg_start = segment_nodes[seg_i]
        seg_end = segment_nodes[seg_i + 1]

        start_chain_idx = chain.index(seg_start)
        end_chain_idx = chain.index(seg_end)
        seg_edges = edge_data_list[start_chain_idx:end_chain_idx]

        if not seg_edges:
            continue

        seg_length = sum(float(d.get("length", 0)) for d in seg_edges)
        seg_props = _merge_edge_properties(seg_edges, seg_length)

        geoms = [d["geometry"] for d in seg_edges if "geometry" in d]
        if geoms:
            seg_props["geometry"] = _merge_geometries(geoms)

        seg_props["id"] = f"{seg_start}-{seg_end}"
        graph.add_edge(seg_start, seg_end, **seg_props)

    return retained


# ── Chain consolidation ─────────────────────────────────────────────


def _chain_edge_data(graph: nx.MultiDiGraph[Any], chain: list[Any]) -> list[dict[str, Any]]:
    """Street-edge data for each consecutive chain pair (chain[i] -> chain[i+1])."""
    edge_data_list: list[dict[str, Any]] = []
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i + 1]
        # The chain edge is the pipe edge from u to its chain SUCCESSOR v, not
        # merely u's first pipe out-edge, those differ when u branches (e.g.
        # P_start), which otherwise captured the wrong length/diameter/
        # contributing_area/geometry for the chain's first segment.
        for d in (graph.get_edge_data(u, v) or {}).values():
            if d.get("edge_type", "pipe") in _PIPE_TYPES:
                edge_data_list.append(d)
                break
    return edge_data_list


def _consolidate_chain(
    graph: nx.MultiDiGraph[Any],
    chain: list[Any],
    max_length: float,
) -> dict[Any, Any]:
    """Merge a degree-2 chain into one (or more) replacement edges.

    Returns a mapping ``{removed_node: surviving_downstream_node}``
    for every interior node deleted.
    """
    p_start = chain[0]
    p_end = chain[-1]
    interior = chain[1:-1]

    edge_data_list = _chain_edge_data(graph, chain)
    if not edge_data_list:
        return {}

    total_length = sum(float(d.get("length", 0)) for d in edge_data_list)
    if total_length == 0:
        return {}

    # Merge properties
    merged = _merge_edge_properties(edge_data_list, total_length)

    # Merge geometries
    geoms = [d["geometry"] for d in edge_data_list if "geometry" in d]
    if geoms:
        merged["geometry"] = _merge_geometries(geoms)

    if total_length <= max_length:
        # Simple case: merge all into one edge
        for n in interior:
            graph.remove_node(n)
        # Remove any pre-existing street edge between endpoints
        for k in list(graph[p_start].get(p_end, {}).keys()):
            edata = graph[p_start][p_end][k]
            if edata.get("edge_type", "pipe") in _PIPE_TYPES:
                graph.remove_edge(p_start, p_end, k)
        merged["id"] = f"{p_start}-{p_end}"
        graph.add_edge(p_start, p_end, **merged)
        return dict.fromkeys(interior, p_end)

    # Long chain: split into segments, keeping some interior nodes
    retained = _add_split_edges(graph, chain, edge_data_list, merged, max_length)
    removed = [n for n in interior if n not in retained]
    for n in removed:
        graph.remove_node(n)
    return dict.fromkeys(removed, p_end)


# ── Contributing area redistribution ────────────────────────────────


def _resolve_transitive(node_mapping: dict[Any, Any]) -> dict[Any, Any]:
    """Resolve transitive mappings (a -> b -> c => a -> c)."""
    resolved: dict[Any, Any] = {}
    for n, target in node_mapping.items():
        final = target
        while final in node_mapping:
            final = node_mapping[final]
        resolved[n] = final
    return resolved


def _redistribute_contributing_area(
    graph: nx.MultiDiGraph[Any], node_mapping: dict[Any, Any], own_ca: dict[Any, float]
) -> None:
    """Move each removed node's OWN contributing area onto its surviving target.

    ``own_ca`` is the per-node own subcatchment area snapshotted *before* any
    removal.  Matching calculate_contributing_area's semantics, each surviving
    node's contributing_area becomes its own area plus the own area of every
    removed node that drains into it, and each edge carries its upstream node's
    own area, rather than the previous max(incoming-edge sum, own) conflation.
    """
    resolved = _resolve_transitive(node_mapping)
    merged_own: dict[Any, float] = {}
    for node, area in own_ca.items():
        target = resolved.get(node, node)  # removed -> survivor; survivor -> self
        merged_own[target] = merged_own.get(target, 0.0) + float(area)

    for n in graph.nodes():
        if n in merged_own:
            graph.nodes[n]["contributing_area"] = merged_own[n]

    # Edge contributing_area = upstream node's own area (as set by
    # calculate_contributing_area at geospatial_utilities edge assignment).
    for u, _v, _k, d in graph.edges(keys=True, data=True):
        d["contributing_area"] = graph.nodes[u].get("contributing_area", 0)


# ── Subcatchment aggregation ────────────────────────────────────────


def _aggregate_subcatchments(subs_path: Any, node_mapping: dict[Any, Any]) -> None:
    """Reroute and merge subcatchments for removed nodes.

    Reads the subcatchment parquet, remaps outlet IDs, dissolves
    subcatchments that now share an outlet, and writes back.
    """
    subs = gpd.read_parquet(subs_path)
    if subs.empty:
        return

    resolved = _resolve_transitive(node_mapping)

    # Remap subcatchment outlet IDs
    subs["id"] = subs["id"].map(lambda x: resolved.get(x, x))

    # Dissolve subcatchments sharing the same outlet
    has_rc = "rc" in subs.columns
    has_slope = "slope" in subs.columns
    has_imperv = "impervious_area" in subs.columns

    records = []
    for outlet_id, group in subs.groupby("id"):
        if len(group) == 1:
            records.append(group.iloc[0].to_dict())
            continue

        total_area = float(group["area"].sum())
        areas = group["area"].to_numpy(dtype=float)
        weights = areas / total_area if total_area > 0 else np.ones(len(areas)) / len(areas)

        rec: dict[str, Any] = {"id": outlet_id, "area": total_area}
        rec["geometry"] = shapely.unary_union(group.geometry.to_numpy())
        rec["width"] = float(np.sqrt(total_area / np.pi))

        if has_rc:
            rec["rc"] = float(np.average(group["rc"].to_numpy(dtype=float), weights=weights))
        if has_slope:
            rec["slope"] = float(np.average(group["slope"].to_numpy(dtype=float), weights=weights))
        if has_imperv:
            rec["impervious_area"] = float(group["impervious_area"].sum())

        records.append(rec)

    result = gpd.GeoDataFrame(records, crs=subs.crs)
    # Transitive node_mapping collapses chains, so a group can union
    # spatially-disjoint subs into a disconnected MultiPolygon.  Re-home
    # the detached parts to the subcatchment they physically border
    # (mass-conserving) so every subcatchment stays contiguous.
    result = go.rehome_detached_components(result)
    result.to_parquet(subs_path)


def simplify_network(
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    simplification: SimplificationParams,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Simplify the pipe network by consolidating degree-2 chains.

    Merges series of degree-2 pipe nodes into single conduits and
    optionally removes small dangling leaf branches.  Subcatchments
    whose outlet node is removed are rerouted to the nearest surviving
    downstream node.

    The function is a no-op when ``simplification.enabled`` is False.

    Args:
        graph: Directed multi-graph after hydraulic design.
        addresses: File paths (used to read/write subcatchments).
        simplification: Simplification parameters.
        **kwargs: Ignored.

    Returns:
        The (possibly simplified) graph.
    """
    if not simplification.enabled:
        return graph

    graph = graph.copy()
    # Snapshot each node's OWN contributing area before any removal, so the
    # redistribution can move a removed node's own area onto its survivor.
    own_ca = {n: float(graph.nodes[n].get("contributing_area", 0.0)) for n in graph.nodes}
    n_nodes_before = graph.number_of_nodes()
    n_edges_before = graph.number_of_edges()

    protected = _find_protected_nodes(graph)

    # Step 1 -- remove dangling leaves with small contributing area
    removed_leaf_map: dict[Any, Any] = {}
    if simplification.min_contributing_area_m2 > 0:
        graph, removed_leaf_map = _remove_dangling_leaves(
            graph, simplification.min_contributing_area_m2
        )
        # Recompute protected set after removals
        protected = _find_protected_nodes(graph)

    # Step 2 -- consolidate degree-2 chains
    chains = _find_degree2_chains(graph, protected)
    removed_chain_map: dict[Any, Any] = {}
    for chain in chains:
        mapping = _consolidate_chain(graph, chain, simplification.max_conduit_length)
        removed_chain_map.update(mapping)

    # Build full node mapping: removed_node -> surviving_downstream_node
    node_mapping = {**removed_leaf_map, **removed_chain_map}

    # Step 3 -- reassign contributing_area on surviving nodes
    _redistribute_contributing_area(graph, node_mapping, own_ca)

    # Step 4 -- aggregate subcatchments
    subs_path = addresses.model_paths.subcatchments
    if subs_path.exists() and node_mapping:
        _aggregate_subcatchments(subs_path, node_mapping)

    n_nodes_after = graph.number_of_nodes()
    n_edges_after = graph.number_of_edges()
    logger.info(
        f"simplify_network: {n_nodes_before} -> {n_nodes_after} nodes "
        f"({n_edges_before} -> {n_edges_after} edges, "
        f"{len(node_mapping)} nodes removed)"
    )
    return graph
