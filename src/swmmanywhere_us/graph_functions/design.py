"""Module for graphfcns that design the pipe inverts and diameters.

Improved pipe-by-pipe implementation following Duque et al. (2022).

Correctness improvements over the original:
    1. Slope derived from Manning's equation (paper Eq. 3), not geometry
    2. Diameter monotonicity enforced (downstream >= upstream)
    3. Depth-dependent max filling ratio (paper Table 1), evaluated at the
       true normal-depth y/D, not the Q/Q_full capacity ratio
    4. Configurable max velocity (HydraulicDesign.max_v) enforced consistently
       with the Manning slope bound
    5. Shear stress constraint (tau >= 2 Pa for d >= 0.45m)

Efficiency improvements:
    6. O(N) cumulative flow accumulation (replaces O(N²) nx.ancestors)
    7. First-feasible diameter scan (replaces 390-combo grid search)
    8. Scalar math only (no per-pipe DataFrame creation)

Reference
---------
Duque, N. et al. (2022). A Simplified Sanitary Sewer System Generator
for Exploratory Modelling at City-Scale. Water Research, 209, 117903.
https://doi.org/10.1016/j.watres.2021.117903
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from collections.abc import Hashable

    from swmmanywhere_us.parameters import ChannelDesign, HydraulicDesign

# ---------------------------------------------------------------------------
# Constants (paper Table 1 and Section 2.2.2.2)
# ---------------------------------------------------------------------------
MANNINGS_N = 0.012
WATER_DENSITY = 1000.0  # kg/m³
GRAVITY = 9.81  # m/s²
MIN_SHEAR = 2.0  # Pa (paper Table 1, constraint 3)

# Minimum soil cover above a pipe crown (m).  pipe_by_pipe's slope
# enforcement raises inverts toward the surface on flat terrain; without
# this floor the crown ends up above grade, leaving no room for the
# dual-drainage curb channel and destabilizing DYNWAVE.  0.5 m clears a
# standard 0.3 m curb channel with margin.
_MIN_PIPE_COVER_M = 0.5


# ---------------------------------------------------------------------------
# Cost function (same as original, from https://doi.org/10.2166/hydro.2016.105)
# ---------------------------------------------------------------------------
def _calculate_cost(V: float, diam: float) -> float:
    """Calculate the cost of the pipe.

    Args:
        V: The excavation volume of the pipe.
        diam: The diameter of the pipe.

    Returns:
        The cost of the pipe in USD.
    """
    return 1.32 / 2000 * (9579.31 * diam**0.5737 + 1163.77 * V**1.31)


# ---------------------------------------------------------------------------
# Hydraulic helpers
# ---------------------------------------------------------------------------
def _max_filling_ratio(diam: float) -> float:
    """Depth-dependent max filling ratio (paper Table 1, constraint 2)."""
    if diam <= 0.6:
        return 0.7
    if diam <= 1.5:
        return 0.8
    return 0.85


def _max_velocity(hydraulic_design: HydraulicDesign) -> float:
    """Maximum allowable pipe velocity (m/s).

    Uses the configurable ``HydraulicDesign.max_v`` (default 3.05 m/s). The
    Manning slope upper bound (``s_max_hydraulic``) caps the full-pipe velocity
    at this value, but the explicit check applies it at the true partial-flow
    (normal-depth) velocity, which near capacity exceeds the full-pipe velocity,
    so the check is intentionally stricter than the slope bound alone. The
    paper's looser slope-dependent 5/10 m/s limit (Table 1, c4-5) is superseded
    by this stricter, user-tunable value.
    """
    return float(hydraulic_design.max_v)


def _slope_from_velocity(v: float, R: float) -> float:
    """Compute slope from Manning's equation (paper Eq. 3)."""
    return (v * MANNINGS_N / R ** (2 / 3)) ** 2


def _velocity_from_slope(slope: float, R: float) -> float:
    """Compute velocity from Manning's equation."""
    return (slope**0.5) * (R ** (2 / 3)) / MANNINGS_N


def _shear_stress(R: float, slope: float) -> float:
    """Wall shear stress: tau = rho * g * R * S."""
    return WATER_DENSITY * GRAVITY * R * slope


# Central angle (rad) where partial-flow Manning discharge peaks (~308 deg,
# Q ~ 1.076 Q_full); the subcritical normal depth is the root below this.
_PARTIAL_FLOW_Q_PEAK_THETA = 5.3784


def _partial_flow_hydraulics(
    q: float, diam: float, slope: float, mannings_n: float = MANNINGS_N
) -> tuple[float, float, float, float]:
    """Normal-depth partial-flow geometry for design flow ``q`` in a circular pipe.

    Solves Manning's Q = (1/n) A(theta) R(theta)^(2/3) sqrt(S) for the central
    angle theta (subcritical/lower root where Q is monotonic) and returns
    ``(fill_ratio y/D, flow area A, hydraulic radius R, velocity v = Q/A)`` at
    that depth, the actual values the pipe runs at, not the full-pipe surrogate
    (R = D/4).  Flows at or above the partial-flow capacity peak return full-pipe
    geometry (y/D = 1).
    """
    if q <= 0 or diam <= 0 or slope <= 0:
        return 0.0, 0.0, 0.0, 0.0
    sqrt_s = slope**0.5

    def q_of_theta(theta: float) -> float:
        area = (diam**2 / 8.0) * (theta - math.sin(theta))
        rad = (diam / 4.0) * (1.0 - math.sin(theta) / theta)
        return (1.0 / mannings_n) * area * rad ** (2.0 / 3.0) * sqrt_s

    if q >= q_of_theta(_PARTIAL_FLOW_Q_PEAK_THETA):
        theta = 2.0 * math.pi  # surcharged -> cap at full-pipe geometry
    else:
        lo, hi = 1e-9, _PARTIAL_FLOW_Q_PEAK_THETA
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if q_of_theta(mid) < q:
                lo = mid
            else:
                hi = mid
        theta = 0.5 * (lo + hi)

    area = (diam**2 / 8.0) * (theta - math.sin(theta))
    perim = diam * theta / 2.0
    rad = area / perim if perim > 0 else 0.0
    fill_ratio = (1.0 - math.cos(theta / 2.0)) / 2.0
    velocity = q / area if area > 0 else 0.0
    return fill_ratio, area, rad, velocity


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------
# core multi-pass hydraulic design algorithm; splitting would obscure the pass structure
def pipe_by_pipe(  # noqa: C901, PLR0912, PLR0915
    graph: nx.MultiDiGraph[Any],
    hydraulic_design: HydraulicDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Pipe by pipe hydraulic design following Duque et al. (2022).

    Two-pass algorithm:
    1. Cumulative flow accumulation in topological order, O(N)
    2. For each pipe, scan diameters from smallest feasible and accept
       the first that satisfies all hydraulic constraints.

    Args:
        graph: Directed graph with ``surface_elevation`` and
            ``contributing_area`` node attributes, and ``length``
            edge attribute.
        hydraulic_design: Hydraulic design parameters.
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        Graph with pipe design attributes on edges and
        ``chamber_floor_elevation`` on nodes.
    """
    graph = graph.copy()
    surface_elevations = nx.get_node_attributes(graph, "surface_elevation")
    topological_order = list(nx.topological_sort(graph))
    diameters = sorted(hydraulic_design.diameters)

    # --- Pass 1: Cumulative flow accumulation O(N) ---
    cumulative_area: dict[Hashable, float] = {}
    for node in topological_order:
        area = graph.nodes[node].get("contributing_area", 0.0)
        for pred in graph.predecessors(node):
            area += cumulative_area[pred]
        cumulative_area[node] = area

    M3_PER_HR_TO_M3_PER_S = 1.0 / 3600.0
    precip = hydraulic_design.precipitation
    flow = {n: cumulative_area[n] * precip * M3_PER_HR_TO_M3_PER_S for n in graph}

    # --- Pass 2: Pipe design ---
    chamber_floor: dict[Hashable, float] = {}
    upstream_max_diam: dict[Hashable, float] = {}

    params = [
        "diameter",
        "depth",
        "slope",
        "velocity",
        "fr",
        "tau",
        "cost_usd",
        "velocity_feasibility",
        "fr_feasibility",
        "surcharge_feasibility",
    ]
    edge_designs: dict[str, dict[Any, Any]] = {p: {} for p in params}
    fallback_count = 0

    logger.info(f"Running pipe-by-pipe design ({len(graph)} nodes)")
    # A node is a "pipe source" if it has no incoming street edge, even if
    # it has pond_connector / river / outfall predecessors.  Without this
    # broadening, a pipe-network node fed only by a pond ends up with no
    # chamber_floor and breaks the us_invert lookup below.
    for node in topological_order:
        has_pipe_pred = any(
            d.get("edge_type") == "pipe" for _, _, d in graph.in_edges(node, data=True)
        )
        if not has_pipe_pred:
            has_pipe_succ = any(
                d.get("edge_type") == "pipe" for _, _, d in graph.out_edges(node, data=True)
            )
            if has_pipe_succ or node not in surface_elevations:
                chamber_floor[node] = surface_elevations.get(node, 0) - hydraulic_design.min_depth

        Q = flow[node]

        for ds_node in graph.successors(node):
            edge = graph.get_edge_data(node, ds_node, 0)
            # Skip non-pipe edges -- rivers and outfalls get separate geometry,
            # and pond_connector edges are replaced by finalize_pond_outlets.
            if edge.get("edge_type") in (
                "river",
                "outfall",
                "water_body_link",
                "pond_connector",
            ):
                continue
            length = edge["length"]
            us_elev = surface_elevations[node]
            ds_elev = surface_elevations[ds_node]
            us_invert = max(chamber_floor[node], us_elev - hydraulic_design.max_depth)

            # Enforce positive slope: if the upstream invert is already below
            # the downstream surface, we cannot create positive flow.  Raise
            # the upstream invert to ensure a positive slope is achievable.
            min_us_invert = ds_elev + hydraulic_design.min_positive_slope * length
            if us_invert < min_us_invert:
                # Raise upstream invert, but don't go above surface
                us_invert = min(min_us_invert, us_elev - 0.01)
                # Propagate back to chamber_floor so downstream pipes see it
                chamber_floor[node] = max(chamber_floor[node], us_invert)

            # Diameter monotonicity (paper Section 2.2.2.2)
            min_diam = upstream_max_diam.get(node, diameters[0])
            start_idx = 0
            for i, d in enumerate(diameters):
                if d >= min_diam:
                    start_idx = i
                    break

            # First-feasible diameter scan
            accepted = False
            best: dict[str, float] = {}

            for diam in diameters[start_idx:]:
                A = np.pi * diam**2 / 4
                R = diam / 4.0

                # Slope bounds from Manning's (paper Eq. 3)
                s_min = _slope_from_velocity(hydraulic_design.min_v, R)
                s_max_hydraulic = _slope_from_velocity(hydraulic_design.max_v, R)

                # Depth-dependent filling ratio (paper Table 1)
                max_fr = _max_filling_ratio(diam)

                # Required velocity for design flow at max filling
                if Q > 0 and (max_fr * A) > 0:
                    v_required = Q / (max_fr * A)
                    s_required = _slope_from_velocity(v_required, R)
                else:
                    s_required = s_min

                # Clamp slope to feasible range
                slope = max(s_min, min(s_required, s_max_hydraulic))

                # Derive downstream invert from slope
                invert_ds = us_invert - slope * length
                depth_ds = ds_elev - invert_ds

                # Adjust slope if depth is out of bounds (flat terrain)
                depth_constrained = False
                if depth_ds > hydraulic_design.max_depth:
                    slope_adj = (us_invert - (ds_elev - hydraulic_design.max_depth)) / length
                    if slope_adj <= 0:
                        continue
                    slope = slope_adj
                    depth_ds = hydraulic_design.max_depth
                    depth_constrained = True
                elif depth_ds < hydraulic_design.min_depth:
                    slope_adj = (us_invert - (ds_elev - hydraulic_design.min_depth)) / length
                    if slope_adj <= 0:
                        continue
                    slope = slope_adj
                    depth_ds = hydraulic_design.min_depth
                    depth_constrained = True

                if depth_ds < hydraulic_design.min_depth or depth_ds > hydraulic_design.max_depth:
                    continue

                # Partial-flow geometry at the design (normal) depth, the
                # actual velocity, hydraulic radius and filling ratio the pipe
                # runs at, plus the full-pipe capacity ratio for the
                # flat-terrain margin:
                #  - fill_ratio = true y/D, checked against the Table-1 max;
                #  - v / r_pf = design-depth velocity & hydraulic radius for the
                #    self-cleansing-velocity and shear checks (not R = D/4);
                #  - capacity_ratio = Q/Q_full (full-pipe), for the margin.
                fill_ratio, _a_pf, r_pf, v = _partial_flow_hydraulics(Q, diam, slope)
                q_full = _velocity_from_slope(slope, R) * A
                capacity_ratio = Q / q_full if q_full > 0 else 0.0

                # Velocity check at the design depth (relaxed when depth-constrained).
                # Skipped for zero-flow pipes (Q == 0, e.g. a fully pervious
                # headwater): the self-cleansing minimum velocity is meaningless
                # with no flow, and the partial-flow solver returns v = 0, which
                # would otherwise reject every diameter and force the fallback.
                max_v_local = _max_velocity(hydraulic_design)
                if (
                    not depth_constrained
                    and Q > 0
                    and (v < hydraulic_design.min_v or v > max_v_local)
                ):
                    continue

                # Filling ratio check (true normal-depth y/D vs Table 1, c2)
                if fill_ratio > max_fr:
                    continue

                # Flat-terrain capacity margin: when the slope is limited by
                # depth (not hydraulics), Duque's accept-first scan picks the
                # smallest diameter clearing max_fr -- no margin for peak flows
                # during dynamic-wave simulation.  Require a configurable
                # pipe-full-capacity margin (Q_full >= factor * Q, i.e.
                # capacity_ratio <= 1/factor) for these pipes.  See Hesarkazzazi
                # et al. (2022) and Saldarriaga et al. (2024) on the flat-terrain
                # failure mode of pipe-series accept-first-feasible sizing.
                if (
                    depth_constrained
                    and hydraulic_design.flat_terrain_capacity_factor > 1.0
                    and capacity_ratio > 1.0 / hydraulic_design.flat_terrain_capacity_factor
                ):
                    continue

                # Shear stress at the design depth (relaxed when depth-constrained,
                # and skipped for zero-flow pipes: no flow, no scour to resist).
                tau = _shear_stress(r_pf, slope)
                if not depth_constrained and Q > 0 and diam >= 0.45 and tau < MIN_SHEAR:
                    continue

                # Excavation cost
                up_depth = us_elev - us_invert
                avg_depth = (up_depth + depth_ds) / 2
                V = length * (diam + 0.3) * (avg_depth + 0.1)
                cost = _calculate_cost(max(V, 0), diam)

                v_feas = max(hydraulic_design.min_v - v, 0) + max(v - max_v_local, 0)
                fr_feas = max(fill_ratio - max_fr, 0)

                best = {
                    "diameter": diam,
                    "depth": depth_ds,
                    "slope": slope,
                    "velocity": v,
                    "fr": fill_ratio,
                    "tau": tau,
                    "cost_usd": cost,
                    "velocity_feasibility": v_feas,
                    "fr_feasibility": fr_feas,
                    "surcharge_feasibility": 0.0,
                }
                accepted = True
                break

            # Fallback: largest diameter, guarantee positive slope
            if not accepted:
                fallback_count += 1
                diam = diameters[-1]
                r = diam / 4.0
                s_min = _slope_from_velocity(hydraulic_design.min_v, r)
                min_slope = hydraulic_design.min_positive_slope

                # Compute the feasible slope range within depth constraints
                slope_at_max = (us_invert - (ds_elev - hydraulic_design.max_depth)) / length
                slope_at_min = (us_invert - (ds_elev - hydraulic_design.min_depth)) / length

                if slope_at_max > 0:
                    # Normal case: positive slope is achievable
                    slope = max(min(s_min, slope_at_max), slope_at_min, min_slope)
                else:
                    # us_invert should have been raised in the pre-loop check,
                    # but if we still can't get positive slope, force it by
                    # setting downstream invert to us_invert - min_slope * length
                    slope = min_slope

                invert_ds = us_invert - slope * length
                depth_ds = ds_elev - invert_ds
                # Allow depth to go below min_depth when enforcing positive slope
                # (alternative would be to raise us_invert further, done in pre-loop)
                depth_ds = min(depth_ds, hydraulic_design.max_depth)
                invert_ds = ds_elev - depth_ds

                # Recompute final slope from actual inverts
                slope = max((us_invert - invert_ds) / length, min_slope)

                fill_ratio, _a_pf, r_pf, v = _partial_flow_hydraulics(Q, diam, slope)
                tau = _shear_stress(r_pf, slope)
                up_depth = us_elev - us_invert
                avg_depth = (up_depth + depth_ds) / 2
                vol = length * (diam + 0.3) * (max(avg_depth, 0) + 0.1)
                cost = _calculate_cost(max(vol, 0), diam)

                max_v_local = _max_velocity(hydraulic_design)
                best = {
                    "diameter": diam,
                    "depth": depth_ds,
                    "slope": slope,
                    "velocity": v,
                    "fr": fill_ratio,
                    "tau": tau,
                    "cost_usd": cost,
                    "velocity_feasibility": max(hydraulic_design.min_v - v, 0)
                    + max(v - max_v_local, 0),
                    "fr_feasibility": max(fill_ratio - _max_filling_ratio(diam), 0),
                    "surcharge_feasibility": 0.0,
                }

            # Store design
            key = (node, ds_node, 0)
            for p in params:
                edge_designs[p][key] = best[p]

            # Propagate downstream chamber floor.  Use the actual us_invert
            # from the design (which may have been raised for positive-slope
            # enforcement), not the raw chamber_floor[node].
            # When multiple pipes converge, shallowest (highest invert) wins.
            invert_ds = us_invert - best["slope"] * length
            invert_ds = max(invert_ds, ds_elev - hydraulic_design.max_depth)
            chamber_floor[ds_node] = max(invert_ds, chamber_floor.get(ds_node, -np.inf))

            # Diameter monotonicity propagation
            upstream_max_diam[ds_node] = max(
                best["diameter"],
                upstream_max_diam.get(ds_node, 0),
            )

    if fallback_count > 0:
        logger.warning(f"{fallback_count} pipes used fallback (relaxed constraints)")

    # --- Pass 3: Condition chamber_floor for monotonic positive slope ---
    # Pass 2 designs pipes one-at-a-time and can produce adverse slopes when
    # the terrain is uphill.  This pass conditions the chamber_floor values
    # by (a) filling local sinks, (b) breaking equal-elevation ties with an
    # epsilon nudge, and (c) enforcing min and max slope bounds along every
    # pipe edge in topological order.  Adapted from the graph-conditioning
    # algorithm in hyriver/busn_estimator.
    min_slope_global = hydraulic_design.min_positive_slope
    max_slope_global = 0.10  # 10% max pipe grade (engineering limit)
    max_depth = hydraulic_design.max_depth
    epsilon = 1e-6

    # Build a street-pipe-only subgraph view for sink detection and
    # neighbor iteration.  Each edge must have a length attribute.
    pipe_edges: list[tuple[Any, Any, float]] = []
    for u, v, d in graph.edges(data=True):
        if d.get("edge_type") != "pipe":
            continue
        if u not in chamber_floor or v not in chamber_floor:
            continue
        length = d.get("length", 0)
        if length > 0:
            pipe_edges.append((u, v, length))

    # Pipe-network nodes (participating in chamber_floor conditioning)
    pipe_nodes = {n for u, v, _ in pipe_edges for n in (u, v)}

    # --- Step A: fill chamber_floor sinks ---
    # A sink is a pipe node whose chamber_floor is LOWER than every
    # connected pipe neighbor (via either direction).  Fill it to the
    # minimum neighbor's chamber_floor.  Process highest to lowest so
    # chained sinks resolve in one pass.
    neighbors_of: dict[Any, set[Any]] = {n: set() for n in pipe_nodes}
    for u, v, _ in pipe_edges:
        neighbors_of[u].add(v)
        neighbors_of[v].add(u)

    sink_fills = 0
    sorted_nodes = sorted(pipe_nodes, key=lambda n: -chamber_floor[n])
    for node in sorted_nodes:
        nbrs = neighbors_of.get(node, set())
        if len(nbrs) < 2:
            continue
        node_cf = chamber_floor[node]
        all_higher = all(chamber_floor[nb] > node_cf for nb in nbrs)
        if all_higher:
            chamber_floor[node] = min(chamber_floor[nb] for nb in nbrs)
            sink_fills += 1
    if sink_fills > 0:
        logger.info(f"Pass 3: filled {sink_fills} chamber_floor sink nodes")

    # --- Step B: break equal-elevation ties with epsilon nudging ---
    # When two connected nodes have the same chamber_floor, the
    # downstream one is nudged slightly lower so that slope
    # enforcement has an unambiguous direction.
    ties_broken = 0
    for node in topological_order:
        if node not in pipe_nodes:
            continue
        node_cf = chamber_floor[node]
        succ_ordered = [
            s
            for s in graph.successors(node)
            if s in pipe_nodes and graph.get_edge_data(node, s, 0).get("edge_type") == "pipe"
        ]
        for i, ds_node in enumerate(succ_ordered):
            if abs(chamber_floor[ds_node] - node_cf) < epsilon:
                chamber_floor[ds_node] = node_cf - (i + 1) * epsilon
                ties_broken += 1
    if ties_broken > 0:
        logger.info(f"Pass 3: broke {ties_broken} chamber_floor ties via epsilon nudge")

    # --- Step C: enforce min/max slope along each pipe ---
    # Uses fixed slope bounds (``min_positive_slope`` to 10 %) and the
    # simple "lower ds_cf to required slope, raise only above surface"
    # logic.  Pass 3 produces shallow burials (often < min_depth on
    # flat terrain) but those shallow pipes are what keep DYNWAVE
    # stable on the broader network, verified empirically that
    # forcing min_depth at the Pass 3 layer regresses routing
    # continuity from -3.8 % to -36 %, even when slope bounds stay
    # fixed.  (A per-pipe velocity-derived bound variant was tested
    # and regressed continuity to -95 % on flat terrain; it was
    # removed.)
    slope_corrections = 0
    max_slope_clamps = 0
    for node in topological_order:
        if node not in chamber_floor:
            continue
        for ds_node in graph.successors(node):
            edge = graph.get_edge_data(node, ds_node, 0)
            if edge is None:  # pyright: ignore[reportUnnecessaryComparison]
                continue
            if edge.get("edge_type") != "pipe":
                continue
            if ds_node not in chamber_floor:
                continue
            length = edge["length"]
            if length <= 0:
                continue

            us_cf = chamber_floor[node]
            ds_cf = chamber_floor[ds_node]

            # Simple slope clamp: lower ds_cf if too high for min_slope,
            # raise (cap at surface - 0.1) if too low for max_slope.
            # Does NOT enforce min_depth, shallow burials are tolerated
            # to keep network gradients low for SWMM stability.
            required_ds = us_cf - min_slope_global * length
            # Clamp the min-slope deepening to max_depth.  On uphill
            # chains the unclamped lowering spirals the invert 15-20 m
            # below grade, far past max_depth, producing 300%-slope
            # pipes the DYNWAVE solver cannot converge on.  Capping at
            # max_depth accepts a flatter (even mildly adverse) pipe
            # instead, which DYNWAVE handles far better.
            if ds_node in surface_elevations:
                required_ds = max(required_ds, surface_elevations[ds_node] - max_depth)
            min_ds_cf = us_cf - max_slope_global * length
            if ds_cf > required_ds:
                chamber_floor[ds_node] = required_ds
                slope_corrections += 1
            elif ds_cf < min_ds_cf:
                if ds_node in surface_elevations:
                    min_ds_cf = min(min_ds_cf, surface_elevations[ds_node] - 0.1)
                chamber_floor[ds_node] = min_ds_cf
                max_slope_clamps += 1

    if slope_corrections > 0:
        logger.info(f"Pass 3: lowered {slope_corrections} downstream inverts for min slope")
    if max_slope_clamps > 0:
        logger.info(f"Pass 3: raised {max_slope_clamps} downstream inverts for max slope")

    # --- Pass 4: enforce minimum pipe cover -------------------------------
    # Passes 2-3 raise inverts toward the surface to force positive slopes,
    # which on flat terrain buries pipes shallower than their own diameter
    # (crowns above grade).  That leaves no room for the dual-drainage curb
    # channel, so add_dual_drainage emits degenerate channels and DYNWAVE
    # diverges.  Deepen any too-shallow node so every connected pipe's crown
    # clears grade by ``_MIN_PIPE_COVER_M``.  Diameter monotonicity means a
    # downstream node is lowered at least as much as its upstream node, so
    # this only ever adds positive slope, it never creates an adverse pipe.
    node_max_diam: dict[Hashable, float] = {}
    for (pu, pv, _pk), diam in edge_designs["diameter"].items():
        d = float(diam)
        node_max_diam[pu] = max(node_max_diam.get(pu, 0.0), d)
        node_max_diam[pv] = max(node_max_diam.get(pv, 0.0), d)
    deepened = 0
    for nid, max_diam in node_max_diam.items():
        if nid not in chamber_floor or nid not in surface_elevations:
            continue
        required_cf = surface_elevations[nid] - max_diam - _MIN_PIPE_COVER_M
        if chamber_floor[nid] > required_cf:
            chamber_floor[nid] = required_cf
            deepened += 1
    if deepened:
        logger.info(
            f"Pass 4: deepened {deepened} node(s) so pipe crowns clear grade "
            f"by >= {_MIN_PIPE_COVER_M} m"
        )

    # --- Pass 5: zero-adverse profile carving ("breach, don't fill") ------
    # Step C tolerates mildly adverse pipes where its min-slope lowering
    # would exceed max_depth (the alternative, unclamped lowering, spirals
    # inverts on uphill chains).  Gravity sewers must not run uphill
    # (feasibility constraint in sewer-design optimization; adverse reaches
    # require pumps/siphons), so repair the residue here.  This is the
    # invert analog of least-cost depression BREACHING vs FILLING
    # (Lindsay 2016): walking the designed forest from the outfalls
    # upstream, an adverse pipe first lifts its upstream invert toward the
    # surface (bounded by crown cover, so Pass 4 is not undone); any
    # remaining deficit is pushed onto the downstream path by uniformly
    # lowering every node from the pipe's downstream end to the path
    # terminal.  The uniform shift preserves every already-conditioned
    # slope on the carved stretch, and sibling feeders only gain head, so
    # no new adverse pipe can appear.  The eps target keeps every pipe
    # strictly positive; the full ``min_positive_slope`` ambition stays
    # Step C's job where burial depth affords it (velocity-derived bounds
    # were tested and break DYNWAVE on flat terrain, see Step C note).
    eps_slope = 1e-4
    succ_pipe: dict[Hashable, tuple[Hashable, float]] = {}
    for pu, pv, pd in graph.edges(data=True):
        if pd.get("edge_type") == "pipe" and float(pd.get("length", 0) or 0) > 0:
            succ_pipe[pu] = (pv, float(pd["length"]))
    lifted = carved = 0
    max_carve = 0.0
    residual_count = 0
    max_residual = 0.0
    for node in reversed(topological_order):
        if node not in chamber_floor or node not in succ_pipe:
            continue
        ds_node, length = succ_pipe[node]
        if ds_node not in chamber_floor:
            continue
        floor_u = chamber_floor[ds_node] + eps_slope * length
        if chamber_floor[node] >= floor_u:
            continue
        deficit = floor_u - chamber_floor[node]
        # Remedy 1: lift the upstream invert, keeping the crown buried.
        ceiling = (
            surface_elevations[node] - node_max_diam.get(node, 0.0) - _MIN_PIPE_COVER_M
            if node in surface_elevations
            else chamber_floor[node]
        )
        lift = min(deficit, max(ceiling - chamber_floor[node], 0.0))
        if lift > 0:
            chamber_floor[node] += lift
            deficit -= lift
            lifted += 1
        if deficit <= 1e-12:
            continue
        # Remedy 2: carve the downstream path by the remaining deficit.  The
        # carve lowers the whole downstream chain UNIFORMLY (preserving every
        # already-conditioned slope), so the admissible shift is bounded by the
        # SHALLOWEST max-depth headroom on that chain, carving further would
        # excavate past the 5 m limit Step C and Pass 4 respect.  Any deficit
        # beyond that headroom is left as a bounded residual adverse pipe (the
        # same mild-adverse tolerance Step C already accepts) rather than
        # producing an infeasible burial depth.
        path: list[Hashable] = []
        cur: Hashable | None = ds_node
        on_path: set[Hashable] = set()
        while cur is not None and cur not in on_path:
            on_path.add(cur)
            path.append(cur)
            nxt = succ_pipe.get(cur)
            cur = nxt[0] if nxt is not None else None
        headrooms = [
            chamber_floor[n] - (surface_elevations[n] - max_depth)
            for n in path
            if n in chamber_floor and n in surface_elevations
        ]
        applied = min(deficit, max(min(headrooms), 0.0)) if headrooms else deficit
        if applied > 0:
            for n in path:
                if n in chamber_floor:
                    chamber_floor[n] -= applied
            max_carve = max(max_carve, applied)
            carved += 1
        residual = deficit - applied
        if residual > 1e-9:
            residual_count += 1
            max_residual = max(max_residual, residual)
    if lifted or carved:
        logger.info(
            f"Pass 5: repaired adverse pipes, lifted {lifted} upstream "
            f"invert(s), carved {carved} downstream path(s) "
            f"(max carve {max_carve:.3f} m)."
        )
    if residual_count:
        logger.warning(
            f"Pass 5: {residual_count} adverse pipe(s) left with a bounded residual "
            f"(max {max_residual:.3f} m) to respect the {max_depth} m excavation limit."
        )

    # Report nodes exceeding max burial depth as a warning (after all invert
    # conditioning, so it reflects the final design).
    deep_count = sum(
        1
        for node, floor in chamber_floor.items()
        if node in surface_elevations and surface_elevations[node] - floor > max_depth
    )
    if deep_count > 0:
        logger.warning(
            f"{deep_count} node(s) exceed max_depth ({max_depth} m) after invert "
            f"conditioning, terrain requires deeper pipes"
        )

    # Refresh the slope design record to match the conditioned inverts.
    for su, sv, sk in list(edge_designs["slope"]):
        if su in chamber_floor and sv in chamber_floor:
            edata = graph.get_edge_data(su, sv, 0) or {}
            seg_len = max(float(edata.get("length", 0.0) or 0.0), 1.0)
            edge_designs["slope"][(su, sv, sk)] = (chamber_floor[su] - chamber_floor[sv]) / seg_len

    for parameter in hydraulic_design.edge_design_parameters:
        if parameter in edge_designs:
            nx.set_edge_attributes(graph, edge_designs[parameter], parameter)

    nx.set_node_attributes(graph, chamber_floor, "chamber_floor_elevation")
    return graph


def add_manhole_drops(
    graph: nx.MultiDiGraph[Any],
    hydraulic_design: HydraulicDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Apply a small hydraulic drop at each manhole by setting street
    pipes' ``out_offset``, mirroring the pattern seen in the calibrated
    UWO Swiss sewer reference (every street pipe has OutOffset
    0.01-0.27 m, median ~0.03 m).

    The drop is a standard sanitary / storm-drain design detail
    (ASCE MOP 37 Sec. 5.4): at each manhole the pipe's downstream invert
    sits ``OutOffset`` meters ABOVE the junction invert, so water exits
    the pipe and falls into the manhole.  Physically, this:

    1. Dissipates flow energy (prevents scour at the inlet to the next
       pipe, and reduces surge propagation).
    2. Gives DYNWAVE a non-zero gradient even at otherwise-flat
       junctions, improves numerical convergence on flat terrain.
    3. Accommodates pipe-size mismatches: crowns typically align or
       the larger pipe's crown is higher, producing a drop at the
       spring-line / invert.

    Per-pipe drop is conservatively capped at half the pipe's actual
    hydraulic drop so the pipe itself never becomes adverse.  Pipes
    already shallower than the drop budget get a proportional (half-of-
    actual-drop) offset rather than the full target value.  This runs
    AFTER ``pipe_by_pipe`` so inverts are already fixed; only the
    ``out_offset`` edge attribute is added.

    Note: this offset supersedes the artificial DEM "inlet depression"
    dig of Si et al. (2024), both target the same physical phenomenon
    (flow "dropping into" an inlet), but the pipe-level offset is the
    correct modeling layer because it only affects the hydraulic SWMM
    model and does not perturb watershed delineation (Callow 2007).
    """
    graph = graph.copy()
    drop_m = float(hydraulic_design.manhole_drop_m)
    if drop_m <= 0:
        logger.info("add_manhole_drops: disabled (manhole_drop_m = 0).")
        return graph
    applied = 0
    clamped = 0
    skipped = 0
    for u, v, _k, d in graph.edges(data=True, keys=True):
        if d.get("edge_type") != "pipe":
            continue
        u_cfe = graph.nodes[u].get("chamber_floor_elevation")
        v_cfe = graph.nodes[v].get("chamber_floor_elevation")
        if u_cfe is None or v_cfe is None:
            skipped += 1
            continue
        in_off = float(d.get("in_offset", 0) or 0)
        # Current effective hydraulic drop = upstream pipe invert
        # (= junction_u invert + in_offset) minus downstream junction
        # invert (out_offset is still 0 at this point).
        actual_drop = (float(u_cfe) + in_off) - float(v_cfe)
        if actual_drop <= 0:
            # Already flat or adverse, don't make it worse.
            skipped += 1
            continue
        # Cap at half the hydraulic drop so pipe slope stays at least
        # half of its pre-drop value.
        out_off = min(drop_m, 0.5 * actual_drop)
        if out_off < drop_m - 1e-9:
            clamped += 1
        d["out_offset"] = out_off
        applied += 1
    logger.info(
        f"add_manhole_drops: set OutOffset on {applied} street pipe(s) "
        f"(target {drop_m * 100:.1f} cm; {clamped} clamped to preserve "
        f"pipe slope, {skipped} skipped as already flat/adverse)."
    )
    return graph


def enforce_outfall_slope(  # noqa: C901 - one pass over outfall edges, two feasibility fixes
    graph: nx.MultiDiGraph[Any],
    hydraulic_design: HydraulicDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Make outfall conduits gravity-feasible: stretch over-steep, fix adverse.

    ``identify_outfalls`` picks the nearest river point (or creates a
    dummy-river sentinel) and pairs it to the downstream-most street
    node; the outfall conduit length is the real street-to-receiving-water
    distance (dummy-river sinks fall back to the clustering penalty,
    ``outfall_clustering_factor * median pipe length``, as a nominal length).
    After ``pipe_by_pipe`` sets real inverts, a short outfall that drops
    5-10 m becomes a 100-200 % slope pipe, DYNWAVE spends the bulk of its
    iterations on that single link and drives overall non-convergence
    to > 50 %.  On the test catchment the specific offender is the dummy-river
    outfall whose upstream node (CFE ~6 m) connects to a dummy at
    CFE -1 m over a few meters of pipe.

    Following the user-proposed principle ("outfall location is a
    hint for WHICH water body; actual attachment point can shift"),
    this step keeps the matched receiving water but **stretches the
    outfall pipe** to whatever length keeps its slope at
    ``max_outfall_slope``.  SWMM uses ``length`` for hydraulic routing,
    not geometry, so increasing length is equivalent to having walked
    along the river (or moved the dummy farther out) to a point where
    a reasonable culvert gradient is physically attainable.  For the
    over-steep case only ``length`` changes.

    The mirror-image problem is the **adverse** outfall: the street is
    paired (by horizontal distance, elevation-blind) to a receiving water
    at or above its invert, so the conduit would run uphill.  This arises
    because ``_set_water_body_outfall_elevations`` pins the sink invert to
    the water *surface*, which is unphysically high, a real outfall
    discharges below the waterline.  Here the receiving-water sink invert
    is lowered just below the street invert so flow drains by gravity (a
    submerged / free outfall), again keeping the matched water but shifting
    the attachment depth.  Pond-storage sinks are excluded: their invert is
    fixed by the pond design and one storage may carry several intakes.

    References:
        FDOT Drainage Design Guide Chapter 6 (Culverts), 10 % slope
            upper bound for free-flowing circular culverts.
        ASCE Manual 37 §4.2.3, outfall pipe design criteria.
    """
    graph = graph.copy()
    max_slope = float(hydraulic_design.max_outfall_slope)
    if max_slope <= 0:
        return graph

    # Minimal positive slope used when restoring an adverse outfall, gentle
    # enough not to over-bury the sink, steep enough to drain by gravity.
    feasible_slope = 1e-3

    stretched = 0
    lowered = 0
    worst_before = 0.0
    worst_after = 0.0
    max_lower = 0.0
    for u, v, _k, d in graph.edges(data=True, keys=True):
        if d.get("edge_type") != "outfall":
            continue
        u_cfe = graph.nodes[u].get("chamber_floor_elevation")
        v_cfe = graph.nodes[v].get("chamber_floor_elevation")
        if u_cfe is None or v_cfe is None:
            continue
        length = float(d.get("length", 0) or 0)
        if length <= 0:
            continue
        in_off = float(d.get("in_offset", 0) or 0)
        out_off = float(d.get("out_offset", 0) or 0)
        drop = (float(u_cfe) + in_off) - (float(v_cfe) + out_off)
        if drop <= 0:
            # Adverse/flat outfall: lower the receiving-water sink invert just
            # below the street invert so the conduit drains by gravity.  Pond
            # storages are excluded (invert fixed by pond design; may carry
            # several intakes).
            if d.get("pond_intake"):
                continue
            new_v_cfe = (float(u_cfe) + in_off) - out_off - feasible_slope * length
            graph.nodes[v]["chamber_floor_elevation"] = new_v_cfe
            lowered += 1
            max_lower = max(max_lower, float(v_cfe) - new_v_cfe)
            continue
        slope = drop / length
        if slope <= max_slope + 1e-9:
            continue
        required_length = drop / max_slope
        d["length"] = required_length
        # Keep the shapely geometry but set its parametric length to
        # match so downstream consumers (INP writer, visualizations)
        # see a consistent number.  We do NOT move node coordinates, 
        # the new length is an effective / hydraulic length for the
        # "walked" outfall.
        stretched += 1
        worst_before = max(worst_before, slope)
        new_slope = drop / required_length
        worst_after = max(worst_after, new_slope)
    msgs = []
    if stretched:
        msgs.append(
            f"stretched {stretched} over-steep pipe(s) "
            f"(worst {worst_before * 100:.0f} % -> {worst_after * 100:.1f} %)"
        )
    if lowered:
        msgs.append(f"lowered {lowered} adverse sink invert(s) (max {max_lower:.2f} m)")
    if msgs:
        logger.info(f"enforce_outfall_slope: {'; '.join(msgs)}.")
    else:
        logger.info("enforce_outfall_slope: all outfall conduits already gravity-feasible.")
    return graph


def _full_pipe_capacity_m3s(diam: float, slope: float, mannings_n: float) -> float:
    """Manning full-pipe capacity (m^3/s) for a circular pipe."""
    if diam <= 0 or slope <= 0:
        return 0.0
    A = np.pi * diam**2 / 4.0
    R = diam / 4.0
    v = (slope**0.5) * (R ** (2 / 3)) / mannings_n
    return v * A


def pond_release_capacity_m3s(pond_data: dict[str, Any]) -> float:
    """Design controlled-release rate (m^3/s): orifice + weir at full pond.

    Sum of the orifice's free-flow capacity at MaxDepth plus the weir's
    spillway capacity at design freeboard (capped at 0.5 m of head over
    the weir crest so closed-basin ponds with crest << MaxDepth don't
    explode the weir term).

    .. note::
        Originally introduced for a "decoupled-subbasin" pondshed-aware
        pipe sizing scheme (FDOT 2026 Drainage Manual Ch. 5; ASCE MOP 77;
        HEC-HMS Reservoir Routing).  The substitution was tested on the
        flat-terrain test catchment and rolled back, under-sized downstream pipes caused backwater
        that destabilized the orifice convergence Phase 1 had fixed.
        Helper retained for downstream analysis (e.g. computing
        pond-by-pond design release for visualization or
        post-processing); see :func:`resize_street_pipes_for_pond_routing`.
    """
    diam = float(pond_data.get("wb_orifice_diam_m", 0) or 0)
    max_depth = float(pond_data.get("wb_max_depth", 0) or 0)
    if diam <= 0 or max_depth <= 0:
        return 0.0
    cd_orifice = 0.65  # default Cd; pond_design isn't available in this scope
    area = np.pi * diam**2 / 4.0
    q_orifice = cd_orifice * area * np.sqrt(2.0 * GRAVITY * max_depth)

    weir_length = float(pond_data.get("wb_weir_length_m", 0) or 0)
    weir_crest = float(pond_data.get("wb_weir_crest_m", 0) or 0)
    cw = 3.0  # default rectangular weir coefficient
    head_weir = max(0.0, min(max_depth - weir_crest, 0.5))
    q_weir = cw * weir_length * head_weir**1.5 if head_weir > 0 else 0.0

    return q_orifice + q_weir


# single area re-accumulation + monotone diameter bump pass mirroring pipe_by_pipe
def resize_street_pipes_for_pond_routing(  # noqa: C901, PLR0912, PLR0915
    graph: nx.MultiDiGraph[Any],
    hydraulic_design: HydraulicDesign,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Bump street-pipe diameters to handle pond outlet contributions.

    ``pipe_by_pipe`` runs BEFORE ``finalize_pond_outlets`` and sizes street
    pipes assuming each pond drains via its provisional ``pond_connector``
    anchor.  When ``finalize_pond_outlets`` reroutes a pond to a truly
    downstream node (step 1), often dropping it 1-3 m below the original
    anchor, that pond's subcatchment area now flows through a different
    branch of the network than ``pipe_by_pipe`` accounted for, leaving the
    new branch's street pipes catastrophically undersized.  The
    canonical symptom is a single street junction receiving
    4 pond outflows + 3 street pipes that produce a peak inflow ~4x the
    downstream pipe's full capacity.

    This step:

    1. Re-accumulates ``contributing_area`` through the FINAL topology
       (orifice / weir / pond_outflow predecessors are followed for area
       propagation, identical to ``pipe_by_pipe`` Pass 1).
    2. Bumps every street edge whose existing diameter cannot carry the
       new cumulative design flow at the pipe's existing slope, to the
       smallest standard diameter that can.  Pipe inverts and slopes are
       NOT touched, so the pond outlet decisions made in
       ``finalize_pond_outlets`` (in_offset, conduit inverts) remain
       valid.  Diameter monotonicity is enforced, once a pipe is
       resized, its successor's minimum diameter is at least as large.

    **Note on pondshed-aware sizing**: the FDOT decoupled-subbasin
    pattern (substituting the pond's design release rate for the
    upstream catchment area at each pond outlet) was implemented and
    tested but rolled back, see :func:`_pond_release_capacity_m3s`
    docstring.  The substitution makes downstream pipes smaller per
    design-storm logic, but the test storms used to verify our generator
    routinely exceed the 10-yr design intensity, so the smaller pipes
    cause backwater that destabilizes the orifice convergence the crest
    offset had fixed.  Until pipe-by-pipe is restructured to size against
    a check-storm intensity, area substitution is left disabled here;
    the helper is retained for downstream analysis tools.
    """
    graph = graph.copy()
    diameters = sorted(hydraulic_design.diameters)
    precip = hydraulic_design.precipitation
    safety_factor = hydraulic_design.flat_terrain_capacity_factor
    M3_PER_HR_TO_M3_PER_S = 1.0 / 3600.0
    min_slope = hydraulic_design.min_positive_slope

    topological_order = list(nx.topological_sort(graph))

    # Re-accumulate area through the final topology.  Predecessors are
    # iterated regardless of edge type so pond storage -> outlet_junction
    # -> downstream_node fully propagates the pond's subcatchment area.
    cumulative_area: dict[Hashable, float] = {}
    for node in topological_order:
        area = float(graph.nodes[node].get("contributing_area", 0.0) or 0.0)
        for pred in graph.predecessors(node):
            area += cumulative_area.get(pred, 0.0)
        cumulative_area[node] = area
    flow = {n: cumulative_area[n] * precip * M3_PER_HR_TO_M3_PER_S for n in graph}

    # Track the resized diameter at each node so successor pipes inherit
    # at least the upstream-most diameter (monotonicity).
    upstream_max_diam: dict[Hashable, float] = {}
    bumped = 0
    for node in topological_order:
        for ds_node in graph.successors(node):
            edge = graph.get_edge_data(node, ds_node, 0)
            # Resize both regular pipes and pond_inflow conduits, the
            # latter are pipes that route_pipes_into_ponds rerouted to
            # terminate at a pond storage instead of the original ds_node,
            # but they remain CIRCULAR conduits that need diameter sizing.
            if not edge or edge.get("edge_type") not in {"pipe", "pond_inflow"}:
                continue
            curr_diam = edge.get("diameter")
            if curr_diam is None:
                continue
            length = float(edge.get("length", 0) or 0)
            if length <= 0:
                continue
            in_off = float(edge.get("in_offset", 0) or 0)
            out_off = float(edge.get("out_offset", 0) or 0)
            u_cfe = graph.nodes[node].get("chamber_floor_elevation")
            v_cfe = graph.nodes[ds_node].get("chamber_floor_elevation")
            if u_cfe is None or v_cfe is None:
                continue
            slope = ((float(u_cfe) + in_off) - (float(v_cfe) + out_off)) / length
            slope = max(slope, min_slope)
            Q_design = flow[node] * safety_factor
            Q_full_curr = _full_pipe_capacity_m3s(float(curr_diam), slope, MANNINGS_N)
            min_required = max(
                float(curr_diam),
                upstream_max_diam.get(node, 0.0),
            )
            if Q_full_curr >= Q_design and float(curr_diam) >= min_required:
                upstream_max_diam[ds_node] = max(
                    upstream_max_diam.get(ds_node, 0.0), float(curr_diam)
                )
                continue
            # Find smallest standard diameter that handles the new flow.
            new_diam = float(curr_diam)
            for cand in diameters:
                if cand < min_required:
                    continue
                Q_full_cand = _full_pipe_capacity_m3s(cand, slope, MANNINGS_N)
                if Q_full_cand >= Q_design:
                    new_diam = cand
                    break
            else:
                # No standard diameter handles the design flow, use the
                # largest available and let SWMM surcharge.
                new_diam = diameters[-1]
            if new_diam > float(curr_diam):
                edge["diameter"] = new_diam
                bumped += 1
            upstream_max_diam[ds_node] = max(upstream_max_diam.get(ds_node, 0.0), new_diam)

    if bumped:
        logger.info(
            f"resize_street_pipes_for_pond_routing: bumped {bumped} street pipe "
            f"diameter(s) to absorb pond-outlet inflow after rerouting."
        )
    else:
        logger.info("resize_street_pipes_for_pond_routing: no street pipes needed bumping.")
    return graph


# ---------------------------------------------------------------------------
# Channel geometry
# ---------------------------------------------------------------------------
def _sample_flow_accumulation(
    graph: nx.MultiDiGraph[Any],
    flow_accum_path: str | Path,
) -> dict[Hashable, float]:
    """Sample the D8 flow accumulation raster at river node locations.

    Returns the upstream drainage area (m^2) at each river node by
    multiplying the cell count by the cell area.
    """
    import rasterio

    flow_accum_path = Path(flow_accum_path)
    if not flow_accum_path.exists():
        return {}

    river_nodes = {
        n for u, v, d in graph.edges(data=True) if d.get("edge_type") == "river" for n in (u, v)
    }
    if not river_nodes:
        return {}

    xs = nx.get_node_attributes(graph, "x")
    ys = nx.get_node_attributes(graph, "y")
    node_xy = [(n, xs[n], ys[n]) for n in river_nodes if n in xs and n in ys]
    if not node_xy:
        return {}

    with rasterio.open(flow_accum_path) as src:
        cell_area = abs(src.transform.a * src.transform.e)
        coords = [(x, y) for _, x, y in node_xy]
        values = list(src.sample(coords))

    result: dict[Hashable, float] = {}
    for (node, _, _), val in zip(node_xy, values):
        cell_count = float(val[0])
        if cell_count > 0:
            result[node] = cell_count * cell_area
    return result


def _river_flow_accum_areas(graph: nx.MultiDiGraph[Any], addresses: Any) -> dict[Hashable, float]:
    """Load true watershed area at river nodes from the flow-accumulation raster.

    Returns ``{}`` (so the caller falls back to accumulated impervious area)
    when no raster is configured or it cannot be sampled, warning loudly if a
    path is configured but the file is missing.
    """
    if addresses is None:
        return {}
    flow_accum_path = getattr(getattr(addresses, "model_paths", None), "flow_accumulation", None)
    if not flow_accum_path:
        return {}
    flow_accum_areas = _sample_flow_accumulation(graph, flow_accum_path)
    if flow_accum_areas:
        logger.info(
            f"Using flow accumulation raster for {len(flow_accum_areas)} river nodes "
            f"(true watershed area)"
        )
    elif not Path(flow_accum_path).exists():
        logger.warning(
            f"Flow accumulation raster {flow_accum_path} is missing; river channel "
            f"geometry falls back to accumulated impervious area and channels may be "
            f"undersized. derive_subcatchments should persist flow_accum.tif."
        )
    return flow_accum_areas


def _river_drainage_areas(
    graph: nx.MultiDiGraph[Any], addresses: Any
) -> tuple[dict[Hashable, float], dict[Hashable, float], set[Any]]:
    """Upstream drainage area (m^2) at each river node.

    Returns ``(cumulative_river_area, flow_accum_areas, river_nodes)``.
    Priority 1: the flow accumulation raster (true watershed area).
    Priority 2: impervious contributing area accumulated via pipe
    outfalls, propagated downstream along river edges.
    """
    flow_accum_areas = _river_flow_accum_areas(graph, addresses)

    contributing_areas = nx.get_node_attributes(graph, "contributing_area")
    river_inflow: dict[Hashable, float] = {}
    for u, v, d in graph.edges(data=True):
        if d.get("edge_type") == "outfall":
            river_inflow[v] = river_inflow.get(v, 0) + contributing_areas.get(u, 0)

    # Propagate area downstream along river edges (topological order)
    river_nodes = {
        n for u, v, d in graph.edges(data=True) if d.get("edge_type") == "river" for n in (u, v)
    }
    river_subgraph = graph.subgraph(river_nodes).copy()
    try:
        topo_order = list(nx.topological_sort(river_subgraph))
    except nx.NetworkXUnfeasible:
        topo_order = list(river_nodes)

    cumulative_river_area: dict[Hashable, float] = {}
    for node in topo_order:
        if node in flow_accum_areas:
            # Use the true watershed area from flow accumulation
            cumulative_river_area[node] = flow_accum_areas[node]
        else:
            # Fallback: accumulate from outfall inflows + upstream river nodes
            area = river_inflow.get(node, 0)
            for pred in river_subgraph.predecessors(node):
                area += cumulative_river_area.get(pred, 0)
            cumulative_river_area[node] = area
    return cumulative_river_area, flow_accum_areas, river_nodes


def _set_outfall_connector_geometry(graph: nx.MultiDiGraph[Any]) -> int:
    """Set diameter/roughness/offsets on outfall connector edges.

    Outfall pipes connect deep pipe nodes to shallower river nodes.  Uses
    in_offset to raise the pipe entrance so the conduit has a positive
    slope (water flows downhill from pipe network to river even when the
    river channel invert is higher than the buried pipe invert).  Returns
    the number of adverse-slope conduits fixed.
    """
    min_slope = 0.001
    adverse_fixed = 0
    for u, v, _k, d in graph.edges(data=True, keys=True):
        if d.get("edge_type") != "outfall":
            continue
        upstream_diam = 0.0
        for pred in graph.predecessors(u):
            edge_data = graph.get_edge_data(pred, u, 0)
            if edge_data and "diameter" in edge_data:
                upstream_diam = max(upstream_diam, edge_data["diameter"])
        d["diameter"] = max(upstream_diam, 0.3)
        d["roughness"] = MANNINGS_N

        # Enforce positive slope via in_offset (DEPTH mode)
        u_invert = graph.nodes[u].get("chamber_floor_elevation", 0)
        v_invert = graph.nodes[v].get("chamber_floor_elevation", 0)
        length = d.get("length", 1)
        if length > 0 and u_invert < v_invert + min_slope * length:
            d["in_offset"] = (v_invert - u_invert) + min_slope * length
            d["out_offset"] = 0.0
            adverse_fixed += 1
        else:
            d["in_offset"] = 0.0
            d["out_offset"] = 0.0
    return adverse_fixed


def assign_channel_geometry(
    graph: nx.MultiDiGraph[Any],
    channel_design: ChannelDesign,
    addresses: Any = None,
    **kwargs: Any,
) -> nx.MultiDiGraph[Any]:
    """Assign open channel geometry to river edges.

    Uses Leopold-Maddock hydraulic geometry relations to estimate
    channel width from upstream drainage area.  When a flow accumulation
    raster is available (produced during subcatchment delineation), the
    **true watershed area** at each river node is used.  Otherwise falls
    back to the impervious contributing area accumulated via pipe outfalls.

    Also assigns ``chamber_floor_elevation`` to river nodes and sets
    geometry on outfall connector edges.

    Args:
        graph: Directed graph after pipe_by_pipe design.
        channel_design: Channel geometry parameters.
        addresses: FilePaths object (optional). When provided, the flow
            accumulation raster is read for true watershed area.
        **kwargs: Ignored.

    Returns:
        Graph with ``channel_width``, ``channel_depth``, ``roughness``
        on river edges and ``chamber_floor_elevation`` on river nodes.
    """
    graph = graph.copy()
    surface_elevations = nx.get_node_attributes(graph, "surface_elevation")

    # --- Determine upstream area at each river node ---
    cumulative_river_area, flow_accum_areas, river_nodes = _river_drainage_areas(graph, addresses)

    # --- Helper: compute width and depth from area ---
    M2_TO_KM2 = 1e-6

    def _width_depth(area_m2: float) -> tuple[float, float]:
        area_km2 = max(area_m2 * M2_TO_KM2, 0.01)
        w = channel_design.width_coeff * (area_km2**channel_design.width_exponent)
        w = max(channel_design.min_width, min(w, channel_design.max_width))
        d = w * channel_design.depth_ratio
        d = max(channel_design.min_depth, min(d, channel_design.max_depth))
        return w, d

    # --- Assign geometry to each river edge ---
    for u, _v, _k, d in graph.edges(data=True, keys=True):
        if d.get("edge_type") != "river":
            continue
        width, depth = _width_depth(cumulative_river_area.get(u, 0))
        d["channel_width"] = width
        d["channel_depth"] = depth
        d["roughness"] = channel_design.mannings_n
        d["diameter"] = 0.0

    # Set chamber_floor_elevation for river nodes
    for node in river_nodes:
        se = surface_elevations.get(node, 0.0)
        _, depth = _width_depth(cumulative_river_area.get(node, 0))
        graph.nodes[node]["chamber_floor_elevation"] = se - depth

    # Set geometry on outfall connector edges.
    adverse_fixed = _set_outfall_connector_geometry(graph)
    if adverse_fixed > 0:
        logger.info(f"Fixed {adverse_fixed} outfall conduits with adverse slopes via in_offset")

    n_river = sum(1 for _, _, d in graph.edges(data=True) if d.get("edge_type") == "river")
    n_accum = len(flow_accum_areas)
    logger.info(
        f"Assigned channel geometry to {n_river} river edges "
        f"({n_accum} nodes from flow accumulation, {len(river_nodes) - n_accum} from fallback)"
    )
    return graph
