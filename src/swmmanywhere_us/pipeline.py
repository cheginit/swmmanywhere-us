"""Typed pipeline framework for sequential graph transformations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from swmmanywhere_us.graph_functions import (
    add_dual_drainage,
    add_manhole_drops,
    assign_channel_geometry,
    assign_id,
    break_pond_intake_cycles,
    calculate_contributing_area,
    calculate_streetcover,
    calculate_weights,
    cleanup_orphan_subcatchments,
    connect_pipe_components,
    derive_topology,
    divide_and_conquer,
    double_directed,
    enforce_outfall_slope,
    finalize_pond_outlets,
    fix_geometries,
    identify_outfalls,
    insert_pond_nodes,
    merge_short_edges,
    pipe_by_pipe,
    remove_non_pipe_allowable_links,
    remove_river_crossing_pipes,
    reroute_enclosed_gap_subs,
    reroute_subs_to_isolated_ponds,
    resize_pond_orifices,
    resize_street_pipes_for_pond_routing,
    route_pipes_into_ponds,
    set_chahinian_slope,
    set_elevation,
    set_surface_slope,
    simplify_network,
    split_long_edges,
    to_undirected,
)
from swmmanywhere_us.graph_utilities import filter_edges, load_graph, save_graph
from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    import networkx as nx

    from swmmanywhere_us.parameters import ParametersDict

GraphType: TypeAlias = "nx.MultiGraph[Any] | nx.MultiDiGraph[Any]"
GraphFunction: TypeAlias = "Callable[..., nx.MultiGraph[Any] | nx.MultiDiGraph[Any]]"


@dataclass(frozen=True, kw_only=True)
class Step:
    """A single graph transformation step in the pipeline.

    Args:
        name: Unique identifier for this step (used for logging and checkpoints).
        func: The graph transformation function to execute.
        params: Keys from ``ParametersDict`` that this step requires.
        needs_addresses: Whether ``addresses`` (FilePaths) should be passed.
        checkpoint: Whether to save the graph state after this step.
    """

    name: str
    func: GraphFunction
    params: tuple[str, ...]
    needs_addresses: bool
    checkpoint: bool


class Pipeline:
    """An ordered sequence of graph transformation steps.

    Validates step name uniqueness at construction time and provides
    checkpoint/resume capability during execution.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        names = [s.name for s in steps]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            msg = f"Duplicate step names: {duplicates}"
            raise ValueError(msg)
        self.steps = list(steps)
        self._name_to_idx = {s.name: i for i, s in enumerate(self.steps)}

    def run(
        self,
        graph: GraphType,
        params: ParametersDict,
        addresses: Any,
        *,
        resume_after: str | None = None,
        checkpoint_dir: Path | None = None,
    ) -> GraphType:
        """Execute the pipeline steps sequentially.

        Args:
            graph: The input graph.
            params: Parameter dictionary (keys match ``Step.params``).
            addresses: FilePaths object passed to steps that need it.
            resume_after: If set, load the checkpoint for this step name
                and skip all steps up to and including it.
            checkpoint_dir: Directory for saving/loading graph checkpoints.
                When ``None``, checkpointing is disabled.

        Returns:
            The transformed graph after all steps (or early exit).
        """
        start_idx = 0
        if resume_after is not None:
            if resume_after not in self._name_to_idx:
                msg = f"Unknown step '{resume_after}'. Available: {list(self._name_to_idx)}"
                raise ValueError(msg)
            if checkpoint_dir is None:
                msg = "checkpoint_dir is required when using resume_after"
                raise ValueError(msg)
            cp_path = checkpoint_dir / f"{resume_after}_graph.json"
            if not cp_path.exists():
                msg = f"No checkpoint found at {cp_path}"
                raise FileNotFoundError(msg)
            graph = cast("GraphType", load_graph(cp_path))
            start_idx = self._name_to_idx[resume_after] + 1
            logger.info(f"Resuming pipeline after '{resume_after}'")

        for step in self.steps[start_idx:]:
            kwargs = {k: params[k] for k in step.params}
            if step.needs_addresses:
                kwargs["addresses"] = addresses

            graph = step.func(graph, **kwargs)

            if len(filter_edges(graph, frozenset({"pipe", "river"})).edges) == 0:
                logger.warning(f"step: {step.name} removed all edges, stopping.")
                return graph

            logger.info(f"step: {step.name} completed.")

            if checkpoint_dir and step.checkpoint:
                save_graph(graph, checkpoint_dir / f"{step.name}_graph.json")

        return graph

    def __getitem__(self, name: str) -> Step:
        return self.steps[self._name_to_idx[name]]

    def __len__(self) -> int:
        return len(self.steps)


_PIPELINE_STEPS: list[Step] = [
    Step(name="assign_id_1", func=assign_id, params=(), needs_addresses=False, checkpoint=True),
    Step(
        name="remove_non_pipe_allowable_links",
        func=remove_non_pipe_allowable_links,
        params=("topology_derivation",),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="divide_and_conquer",
        func=divide_and_conquer,
        params=("subcatchment_derivation",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="calculate_streetcover",
        func=calculate_streetcover,
        params=("subcatchment_derivation",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="to_undirected", func=to_undirected, params=(), needs_addresses=False, checkpoint=True
    ),
    Step(
        name="split_long_edges",
        func=split_long_edges,
        params=("subcatchment_derivation",),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="merge_short_edges",
        func=merge_short_edges,
        params=(),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(name="assign_id_2", func=assign_id, params=(), needs_addresses=False, checkpoint=True),
    Step(
        name="calculate_contributing_area",
        func=calculate_contributing_area,
        params=("subcatchment_derivation",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="insert_pond_nodes",
        func=insert_pond_nodes,
        params=("pond_design",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="set_elevation", func=set_elevation, params=(), needs_addresses=True, checkpoint=True
    ),
    Step(
        name="double_directed",
        func=double_directed,
        params=(),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="fix_geometries_1",
        func=fix_geometries,
        params=(),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="remove_river_crossing_pipes",
        func=remove_river_crossing_pipes,
        params=(),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="set_surface_slope",
        func=set_surface_slope,
        params=(),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="set_chahinian_slope",
        func=set_chahinian_slope,
        params=(),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="calculate_weights",
        func=calculate_weights,
        params=("topology_derivation",),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="identify_outfalls",
        func=identify_outfalls,
        params=("outfall_derivation",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="derive_topology",
        func=derive_topology,
        params=("topology_derivation",),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="connect_pipe_components",
        func=connect_pipe_components,
        params=("trunk_inference",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="break_pond_intake_cycles",
        func=break_pond_intake_cycles,
        params=(),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="pipe_by_pipe",
        func=pipe_by_pipe,
        params=("hydraulic_design",),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="add_manhole_drops",
        func=add_manhole_drops,
        params=("hydraulic_design",),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="assign_channel_geometry",
        func=assign_channel_geometry,
        params=("channel_design",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="enforce_outfall_slope",
        func=enforce_outfall_slope,
        params=("hydraulic_design",),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="finalize_pond_outlets",
        func=finalize_pond_outlets,
        params=("pond_design",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="route_pipes_into_ponds",
        func=route_pipes_into_ponds,
        params=("pond_design",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="reroute_subs_to_isolated_ponds",
        func=reroute_subs_to_isolated_ponds,
        params=("pond_design",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="reroute_enclosed_gap_subs",
        func=reroute_enclosed_gap_subs,
        params=("pond_design",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="resize_street_pipes_for_pond_routing",
        func=resize_street_pipes_for_pond_routing,
        params=("hydraulic_design",),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="resize_pond_orifices",
        func=resize_pond_orifices,
        params=("pond_design", "hydraulic_design"),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(
        name="fix_geometries_2",
        func=fix_geometries,
        params=(),
        needs_addresses=False,
        checkpoint=True,
    ),
    Step(name="assign_id_3", func=assign_id, params=(), needs_addresses=False, checkpoint=True),
    Step(
        name="simplify_network",
        func=simplify_network,
        params=("simplification",),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="cleanup_orphan_subcatchments",
        func=cleanup_orphan_subcatchments,
        params=(),
        needs_addresses=True,
        checkpoint=True,
    ),
    Step(
        name="add_dual_drainage",
        func=add_dual_drainage,
        params=("dual_drainage", "subcatchment_derivation"),
        needs_addresses=True,
        checkpoint=True,
    ),
]

# Pond-subsystem steps — omitted when ``add_pondsheds`` is False so the
# pipeline produces a plain dual-drainage stormwater network.
_POND_STEP_NAMES: frozenset[str] = frozenset(
    {
        "insert_pond_nodes",
        "break_pond_intake_cycles",
        "finalize_pond_outlets",
        "route_pipes_into_ponds",
        "reroute_subs_to_isolated_ponds",
        "reroute_enclosed_gap_subs",
        "resize_street_pipes_for_pond_routing",
        "resize_pond_orifices",
    }
)


# Add-on steps stripped in ``bare`` mode — everything beyond core
# topology derivation and pipe sizing.  Leaves the bare buried storm-
# drain network (pipes, junctions, outfalls, subcatchments) for
# diagnosing flow direction and pipe-sizing in isolation.
_COMPLEXITY_STEP_NAMES: frozenset[str] = frozenset(
    {
        "connect_pipe_components",
        "add_manhole_drops",
        "simplify_network",
        "add_dual_drainage",
    }
)


def build_pipeline(add_pondsheds: bool = False, bare: bool = False) -> Pipeline:
    """Build the graph-transformation pipeline.

    Args:
        add_pondsheds: When True, include the pond subsystem (pond
            insertion, outlet structures, pipe rerouting into ponds,
            orifice/weir sizing).  When False (default), those steps are
            omitted and the pipeline produces a plain dual-drainage
            stormwater network with per-junction subcatchments only.
        bare: When True, strip the pipeline to the bare buried storm-
            drain network — core topology derivation and pipe sizing
            only.  Omits the pond subsystem, trunk inference, manhole
            drops, network simplification, the dual-drainage surface
            network, and the subcatchment coverage pass.  For diagnosing
            whether flow directions and pipe sizes are sane in isolation.

    Returns:
        The configured :class:`Pipeline`.
    """
    skip: set[str] = set()
    if bare or not add_pondsheds:
        skip |= _POND_STEP_NAMES
    if bare:
        skip |= _COMPLEXITY_STEP_NAMES
    steps = [s for s in _PIPELINE_STEPS if s.name not in skip]
    return Pipeline(steps)


DEFAULT_PIPELINE = build_pipeline(add_pondsheds=True)
