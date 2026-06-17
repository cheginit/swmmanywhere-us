"""Tests for the turn-angle-aware Dijkstra forest (Chahinian et al. 2019)."""

from __future__ import annotations

import networkx as nx
import pytest

from swmmanywhere_us.shortest_path_utils import chahinian_angle_cost, dijkstra_pq


def test_chahinian_angle_cost_eq4():
    """Verify Eq. (4) of Chahinian et al. (2019) at its anchor points."""
    # Straight through is free.
    assert chahinian_angle_cost(180.0) == 0.0
    # Street-grid right angle carries the mild 0.2 cost.
    assert chahinian_angle_cost(90.0) == pytest.approx(0.2)
    # Acute angles (< 30 deg) get the full penalty.
    assert chahinian_angle_cost(0.0) == 1.0
    assert chahinian_angle_cost(29.9) == 1.0
    # Continuous at the 30-deg cap: 0.8 * 60/60 + 0.2 = 1.0.
    assert chahinian_angle_cost(30.0) == pytest.approx(1.0)
    # Right-angle regime, halfway between 90 and 135.
    assert chahinian_angle_cost(120.0) == pytest.approx(0.6)
    # At 135 deg Eq. 4 switches branch: 0.4 * 45/90 = 0.2.
    assert chahinian_angle_cost(135.0) == pytest.approx(0.2)
    # Straight-through regime decays linearly to 0 at 180.
    assert chahinian_angle_cost(157.5) == pytest.approx(0.1)


def test_dijkstra_pq_turn_cost_prefers_straight_path():
    """A cheaper dogleg loses to a straight route once turn costs apply.

    Layout (flow is toward outfall O):

        T(20,0) -> A(10,0) -> O(0,0)     straight, weight 1.0 + 1.0
        T(20,0) -> B(10,10) -> O(0,0)    90-deg turn at B, weight 0.95 + 0.95
    """
    G = nx.MultiDiGraph()
    G.add_node("O", x=0.0, y=0.0)
    G.add_node("A", x=10.0, y=0.0)
    G.add_node("B", x=10.0, y=10.0)
    G.add_node("T", x=20.0, y=0.0)
    G.add_edge("A", "O", weight=1.0)
    G.add_edge("B", "O", weight=0.95)
    G.add_edge("T", "A", weight=1.0)
    G.add_edge("T", "B", weight=0.95)

    # Without turn costs the slightly cheaper dogleg wins.
    no_angle = dijkstra_pq(G, ["O"])
    assert no_angle.has_edge("T", "B")
    assert not no_angle.has_edge("T", "A")

    # With turn costs, the 90-deg turn at B (0.2 * 1.0) makes the dogleg
    # 2.1 vs 2.0 for the straight route through A (180 deg, free).
    with_angle = dijkstra_pq(G, ["O"], angle_scaling=1.0)
    assert with_angle.has_edge("T", "A")
    assert not with_angle.has_edge("T", "B")


def test_dijkstra_pq_turn_cost_skips_nodes_without_coords():
    """Nodes lacking x/y simply contribute no turn cost (no crash)."""
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    result = dijkstra_pq(G, [3], angle_scaling=0.3)
    assert result.has_edge(1, 2)
    assert result.has_edge(2, 3)


def test_dijkstra_pq_skips_outfalls_absent_from_graph():
    """A stray outfall (e.g. a pond dropped by edge filtering) is ignored.

    derive_topology builds the outfalls list before filtering the graph to
    pipe edges; a pond that is both an outfall and a pond_connector endpoint
    is then removed, leaving its id in the list.  Seeding the search from it
    must not raise ``KeyError`` deep in the relaxation loop.
    """
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    # 99 is not a node in G; it must be silently skipped, real outfall still wins.
    result = dijkstra_pq(G, [3, 99])
    assert result.has_edge(1, 2)
    assert result.has_edge(2, 3)
    assert 99 not in result


def test_dijkstra_pq_handles_non_orderable_nodes_on_ties():
    """Equal-distance heap entries must not force comparing node objects.

    The heap carries a monotonic tie-breaker, so two equally-distant,
    non-orderable nodes never get compared directly (which would raise
    ``TypeError: '<' not supported``).
    """
    outfall, a, b = object(), object(), object()
    G = nx.MultiDiGraph()
    G.add_edge(a, outfall, weight=1.0)
    G.add_edge(b, outfall, weight=1.0)
    result = dijkstra_pq(G, [outfall])
    assert result.has_edge(a, outfall)
    assert result.has_edge(b, outfall)
