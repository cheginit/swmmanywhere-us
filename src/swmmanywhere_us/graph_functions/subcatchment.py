"""Module for graphfcns that change subcatchments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely

from swmmanywhere_us import geospatial_utilities as go
from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from swmmanywhere_us.filepaths import FilePaths
    from swmmanywhere_us.parameters import SubcatchmentDerivation


def calculate_contributing_area(
    graph: nx.MultiGraph[Any],
    addresses: FilePaths,
    subcatchment_derivation: SubcatchmentDerivation,
    **kwargs: Any,
) -> nx.MultiGraph[Any]:
    """Calculate the contributing area for each edge.

    Conditions the DEM, computes flow direction and slope, delineates
    subcatchments, and calculates the runoff coefficient.

    Args:
        graph: A graph
        addresses: A FilePaths parameter object
        subcatchment_derivation: A SubcatchmentDerivation parameter object
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        graph: A graph
    """
    graph = graph.copy()

    # Compute flow direction and slope from DEM
    go.compute_flow_direction(
        fid=addresses.bbox_paths.elevation,
        fdir_path=addresses.model_paths.flow_direction,
        slope_path=addresses.model_paths.slope,
        graph=graph,
        rail_path=addresses.bbox_paths.tiger_rail,
        wbt_zip_path=addresses.project_paths.whiteboxtools_binaries_zip,
    )

    subs_gdf = go.derive_subcatchments(
        graph,
        fdir_path=addresses.model_paths.flow_direction,
        slope_path=addresses.model_paths.slope,
        min_drainage_area_m2=subcatchment_derivation.min_drainage_area_m2,
    )
    # Calculate runoff coefficient (RC) from imperviousness raster
    subs_rc = go.derive_rc(subs_gdf, addresses.bbox_paths.imperviousness)

    # Write subs
    subs_rc.to_parquet(addresses.model_paths.subcatchments)

    # --- Per-node subcatchment assignment ------------------------------
    # ``derive_subcatchments`` now runs WBT's Watershed with every graph
    # node as a pour point, so each subcatchment polygon corresponds to
    # exactly one graph node (its ``id``).  Each node receives the full
    # impervious area of its own subcatchment — no distribution across
    # multiple nodes, no nearest-node fallback — which makes the
    # node-level ``contributing_area`` match how SWMM actually routes the
    # subcatchment's runoff (subcatchment.Outlet = this node).  Pipe-sizing
    # flow accumulation and SWMM simulation therefore see the same
    # distribution.
    ca_per_node: dict[Any, float] = dict.fromkeys(graph.nodes, 0.0)
    for sub_id, impervious in zip(subs_rc["id"], subs_rc["impervious_area"]):
        nid = int(sub_id)
        if nid in ca_per_node:
            ca_per_node[nid] = float(impervious)

    total_assigned = sum(ca_per_node.values())
    total_source = float(subs_rc["impervious_area"].sum())
    n_with_area = sum(1 for v in ca_per_node.values() if v > 0)
    logger.info(
        f"contributing_area: assigned {total_assigned:,.0f} m^2 to "
        f"{n_with_area:,} nodes (source total {total_source:,.0f} m^2, "
        f"{len(subs_rc)} subcatchments)"
    )

    nx.set_node_attributes(graph, ca_per_node, "contributing_area")

    # Edge-level attribute: inherit from upstream node (unchanged behaviour)
    edge_attributes = {edge: graph.nodes[edge[0]]["contributing_area"] for edge in graph.edges}
    nx.set_edge_attributes(graph, edge_attributes, "contributing_area")
    return graph


def _best_touching_neighbor(
    g: Any,
    geoms: Any,
    resolved_idx: Any,
    cand: Any,
    touch_tol_m: float,
) -> int:
    """Index of the resolved neighbour with the longest shared border, or -1."""
    best_j = -1
    best_overlap = -1.0
    for c in np.atleast_1d(cand).tolist():
        j = int(resolved_idx[c])
        overlap = shapely.area(
            shapely.intersection(
                shapely.buffer(g, touch_tol_m),
                shapely.buffer(geoms[j], touch_tol_m),
            )
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_j = j
    return best_j


def _reassign_orphans_by_adjacency(
    subs: gpd.GeoDataFrame,
    orphan_mask: Any,
    touch_tol_m: float = 1.0,
) -> dict[int, int]:
    """Map each orphan subcatchment's id to a *spatially adjacent*
    non-orphan subcatchment's id.

    Iterative flood-fill: orphans that touch a non-orphan (or an
    already-resolved orphan) inherit that neighbor's outlet id, so a
    chain of orphans resolves back to the nearest reachable
    subcatchment *through space they actually border*.  This keeps the
    post-merge ``groupby(id).unary_union`` contiguous instead of
    collapsing scattered orphans onto one far node.  Orphans with no
    spatial neighbor at all (true islands) fall back to the nearest
    resolved subcatchment by polygon distance.

    Returns ``{orphan_original_id: target_id}``.
    """
    geoms = subs.geometry.to_numpy()
    ids = subs["id"].to_numpy()
    is_orphan = orphan_mask.to_numpy()

    # current_id[i] is each sub's working outlet id; resolved[i] True
    # once it carries a reachable (or chained-to-reachable) id.
    current_id = ids.astype("int64").copy()
    resolved = ~is_orphan
    reassign: dict[int, int] = {}

    progress = True
    while progress and not resolved.all():
        progress = False
        resolved_idx = np.flatnonzero(resolved)
        if resolved_idx.size == 0:
            break
        tree = shapely.STRtree(geoms[resolved_idx])
        for i in np.flatnonzero(~resolved):
            g = geoms[i]
            cand = tree.query(shapely.buffer(g, touch_tol_m))
            if len(cand) == 0:
                continue
            # pick the resolved neighbour with the longest shared border
            best_j = _best_touching_neighbor(g, geoms, resolved_idx, cand, touch_tol_m)
            if best_j >= 0:
                current_id[i] = current_id[best_j]
                reassign[int(ids[i])] = int(current_id[best_j])
                resolved[i] = True
                progress = True

    # Fallback for true islands: nearest resolved sub by polygon distance.
    if not resolved.all():
        resolved_idx = np.flatnonzero(resolved)
        if resolved_idx.size:
            tree = shapely.STRtree(geoms[resolved_idx])
            for i in np.flatnonzero(~resolved):
                nearest = tree.query_nearest(geoms[i])
                j = int(resolved_idx[int(np.atleast_1d(nearest)[0])])
                reassign[int(ids[i])] = int(current_id[j])
    return reassign


def cleanup_orphan_subcatchments(
    graph: nx.MultiDiGraph[Any],
    addresses: FilePaths,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Reroute subcatchments whose outlet has no pipe path to an outfall.

    Runs at the end of the pipeline (after ``derive_topology``,
    ``finalize_pond_outlets`` and ``simplify_network``) so we know which
    nodes survived and which can actually route water to a SWMM outfall.
    For each subcatchment whose outlet is either missing from the final
    graph or sits in a disconnected component with no path to an outfall,
    the subcatchment is reassigned to the nearest **reachable** node and
    its polygon is merged with any existing subcatchment that shares the
    new outlet.  The total catchment area is preserved (mass balance), so
    the SWMM model sees every drop of rainfall routed to some outfall
    instead of silently dumping orphan-subcatchment runoff into the
    flooding loss term.
    """
    subs_path = addresses.model_paths.subcatchments
    if not subs_path.exists():
        return graph

    subs = gpd.read_parquet(subs_path)
    if subs.empty:
        return graph

    # Identify SWMM outfalls — the river-side node (v) of every
    # ``outfall`` edge, plus the synthetic sinks that post_processing
    # routes to [OUTFALLS] by node_type alone.  Canal-anchored pond
    # chains terminate at ``river_outfall`` sinks reached via a
    # ``pond_outflow`` edge (no ``outfall`` edge), so without the
    # node_type union every such pond's catchment is flagged orphan
    # and its Outlet=storage assignments get silently undone.
    swmm_outfalls = {v for _, v, d in graph.edges(data=True) if d.get("edge_type") == "outfall"}
    swmm_outfalls |= {
        n
        for n, d in graph.nodes(data=True)
        if d.get("node_type") in ("river_outfall", "water_body_outfall")
    }
    if not swmm_outfalls:
        logger.warning("cleanup_orphan_subcatchments: no outfall edges found; skipping.")
        return graph

    # Reachable = every node that can reach an outfall via the directed
    # pipe/river topology.
    reachable: set[Any] = set(swmm_outfalls)
    for ofl in swmm_outfalls:
        reachable |= nx.ancestors(graph, ofl)

    orphan_mask = ~subs["id"].isin(reachable)
    n_orphans = int(orphan_mask.sum())
    if n_orphans == 0:
        logger.info("cleanup_orphan_subcatchments: every subcatchment outlet reaches an outfall.")
        return graph

    # Reassign each orphan to a *spatially adjacent* reachable
    # subcatchment via iterative flood-fill (longest shared border wins
    # ties) rather than to the nearest reachable graph node by centroid.
    # A non-orphan subcatchment's ``id`` is by definition a node that
    # reaches an outfall, so gluing orphans to bordering non-orphans
    # restores connectivity *and* keeps the post-merge
    # ``groupby(id).unary_union`` contiguous instead of collapsing
    # spatially scattered orphans onto one far node — the root cause of
    # disconnected MultiPolygon pondsheds.
    reassign = _reassign_orphans_by_adjacency(subs, orphan_mask)
    if not reassign:
        logger.warning(
            "cleanup_orphan_subcatchments: no reachable subcatchment to glue "
            f"{n_orphans} orphan(s) to; skipping."
        )
        return graph

    # Apply the reassignment.  Each orphan now carries a reachable outlet id
    # (a spatially-adjacent sub's), so its runoff routes to a SWMM outfall.
    subs.loc[orphan_mask, "id"] = (
        subs.loc[orphan_mask, "id"].map(reassign).fillna(subs.loc[orphan_mask, "id"])
    )
    subs["id"] = subs["id"].astype("int64")

    # Do NOT dissolve subcatchments that now share an outlet id.  SWMM fully
    # supports many subcatchments draining to one node, and dissolving by
    # outlet collapses the fine per-pipe subcatchments into mega-subcatchments
    # — in pond models dozens of subs route to one pond storage and would
    # merge into a single 100+ ha polygon (the small pipe subcatchments
    # "disappearing").  Each sub keeps its own polygon and area; unique SWMM
    # names are assigned at write time (``synthetic_write``).  Pondshed
    # contiguity is preserved because each orphan inherited a *spatially
    # adjacent* sub's id, so the per-pond union in ``generate_pondsheds``
    # stays connected without a subcatchment-level merge.
    subs.to_parquet(subs_path)

    logger.info(
        f"cleanup_orphan_subcatchments: rerouted {n_orphans} orphan subcatchment(s) "
        f"to an adjacent reachable subcatchment; {len(subs)} fine subcatchments retained "
        "(no outlet dissolve)."
    )
    return graph
