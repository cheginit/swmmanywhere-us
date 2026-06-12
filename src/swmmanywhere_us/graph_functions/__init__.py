"""Graph transformation functions for the SWMManywhere-US pipeline."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swmmanywhere_us.graph_functions.connectivity import connect_pipe_components
    from swmmanywhere_us.graph_functions.design import (
        add_manhole_drops,
        assign_channel_geometry,
        enforce_outfall_slope,
        pipe_by_pipe,
        resize_street_pipes_for_pond_routing,
    )
    from swmmanywhere_us.graph_functions.dual_drainage import add_dual_drainage
    from swmmanywhere_us.graph_functions.network_cleaning import (
        assign_id,
        calculate_streetcover,
        divide_and_conquer,
        double_directed,
        fix_geometries,
        merge_short_edges,
        remove_non_pipe_allowable_links,
        remove_river_crossing_pipes,
        split_long_edges,
        to_undirected,
    )
    from swmmanywhere_us.graph_functions.outfall import (
        break_pond_intake_cycles,
        identify_outfalls,
    )
    from swmmanywhere_us.graph_functions.simplification import simplify_network
    from swmmanywhere_us.graph_functions.subcatchment import (
        calculate_contributing_area,
        cleanup_orphan_subcatchments,
    )
    from swmmanywhere_us.graph_functions.topology import (
        calculate_weights,
        derive_topology,
        set_chahinian_slope,
        set_elevation,
        set_surface_slope,
    )
    from swmmanywhere_us.graph_functions.water_bodies import (
        finalize_pond_outlets,
        insert_pond_nodes,
        reroute_enclosed_gap_subs,
        reroute_subs_to_isolated_ponds,
        resize_pond_orifices,
        route_pipes_into_ponds,
    )

# ---------------------------------------------------------------------------
# Lazy public API: heavy imports are deferred until first access.  The
# TYPE_CHECKING block above gives pyright/mypy full visibility without
# executing any imports at runtime.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "connect_pipe_components": (
        "swmmanywhere_us.graph_functions.connectivity",
        "connect_pipe_components",
    ),
    "add_manhole_drops": ("swmmanywhere_us.graph_functions.design", "add_manhole_drops"),
    "assign_channel_geometry": (
        "swmmanywhere_us.graph_functions.design",
        "assign_channel_geometry",
    ),
    "enforce_outfall_slope": ("swmmanywhere_us.graph_functions.design", "enforce_outfall_slope"),
    "pipe_by_pipe": ("swmmanywhere_us.graph_functions.design", "pipe_by_pipe"),
    "resize_street_pipes_for_pond_routing": (
        "swmmanywhere_us.graph_functions.design",
        "resize_street_pipes_for_pond_routing",
    ),
    "add_dual_drainage": ("swmmanywhere_us.graph_functions.dual_drainage", "add_dual_drainage"),
    "assign_id": ("swmmanywhere_us.graph_functions.network_cleaning", "assign_id"),
    "calculate_streetcover": (
        "swmmanywhere_us.graph_functions.network_cleaning",
        "calculate_streetcover",
    ),
    "divide_and_conquer": (
        "swmmanywhere_us.graph_functions.network_cleaning",
        "divide_and_conquer",
    ),
    "double_directed": ("swmmanywhere_us.graph_functions.network_cleaning", "double_directed"),
    "fix_geometries": ("swmmanywhere_us.graph_functions.network_cleaning", "fix_geometries"),
    "merge_short_edges": ("swmmanywhere_us.graph_functions.network_cleaning", "merge_short_edges"),
    "remove_non_pipe_allowable_links": (
        "swmmanywhere_us.graph_functions.network_cleaning",
        "remove_non_pipe_allowable_links",
    ),
    "remove_river_crossing_pipes": (
        "swmmanywhere_us.graph_functions.network_cleaning",
        "remove_river_crossing_pipes",
    ),
    "split_long_edges": ("swmmanywhere_us.graph_functions.network_cleaning", "split_long_edges"),
    "to_undirected": ("swmmanywhere_us.graph_functions.network_cleaning", "to_undirected"),
    "break_pond_intake_cycles": (
        "swmmanywhere_us.graph_functions.outfall",
        "break_pond_intake_cycles",
    ),
    "identify_outfalls": ("swmmanywhere_us.graph_functions.outfall", "identify_outfalls"),
    "simplify_network": ("swmmanywhere_us.graph_functions.simplification", "simplify_network"),
    "calculate_contributing_area": (
        "swmmanywhere_us.graph_functions.subcatchment",
        "calculate_contributing_area",
    ),
    "cleanup_orphan_subcatchments": (
        "swmmanywhere_us.graph_functions.subcatchment",
        "cleanup_orphan_subcatchments",
    ),
    "calculate_weights": ("swmmanywhere_us.graph_functions.topology", "calculate_weights"),
    "derive_topology": ("swmmanywhere_us.graph_functions.topology", "derive_topology"),
    "set_chahinian_slope": ("swmmanywhere_us.graph_functions.topology", "set_chahinian_slope"),
    "set_elevation": ("swmmanywhere_us.graph_functions.topology", "set_elevation"),
    "set_surface_slope": ("swmmanywhere_us.graph_functions.topology", "set_surface_slope"),
    "finalize_pond_outlets": (
        "swmmanywhere_us.graph_functions.water_bodies",
        "finalize_pond_outlets",
    ),
    "insert_pond_nodes": ("swmmanywhere_us.graph_functions.water_bodies", "insert_pond_nodes"),
    "reroute_enclosed_gap_subs": (
        "swmmanywhere_us.graph_functions.water_bodies",
        "reroute_enclosed_gap_subs",
    ),
    "reroute_subs_to_isolated_ponds": (
        "swmmanywhere_us.graph_functions.water_bodies",
        "reroute_subs_to_isolated_ponds",
    ),
    "resize_pond_orifices": (
        "swmmanywhere_us.graph_functions.water_bodies",
        "resize_pond_orifices",
    ),
    "route_pipes_into_ponds": (
        "swmmanywhere_us.graph_functions.water_bodies",
        "route_pipes_into_ponds",
    ),
}

__all__ = [
    "add_dual_drainage",
    "add_manhole_drops",
    "assign_channel_geometry",
    "assign_id",
    "break_pond_intake_cycles",
    "calculate_contributing_area",
    "calculate_streetcover",
    "calculate_weights",
    "cleanup_orphan_subcatchments",
    "connect_pipe_components",
    "derive_topology",
    "divide_and_conquer",
    "double_directed",
    "enforce_outfall_slope",
    "finalize_pond_outlets",
    "fix_geometries",
    "identify_outfalls",
    "insert_pond_nodes",
    "merge_short_edges",
    "pipe_by_pipe",
    "remove_non_pipe_allowable_links",
    "remove_river_crossing_pipes",
    "reroute_enclosed_gap_subs",
    "reroute_subs_to_isolated_ponds",
    "resize_pond_orifices",
    "resize_street_pipes_for_pond_routing",
    "route_pipes_into_ponds",
    "set_chahinian_slope",
    "set_elevation",
    "set_surface_slope",
    "simplify_network",
    "split_long_edges",
    "to_undirected",
]


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = mod if attr is None else getattr(mod, attr)
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# ---------------------------------------------------------------------------
# Eager-import override: set EAGER_IMPORT=1 (any non-"0"/non-empty value) to
# load all lazy members immediately.  Useful in CI and for profiling.
# ---------------------------------------------------------------------------
if os.environ.get("EAGER_IMPORT", "") not in ("", "0"):
    for _name in _LAZY_IMPORTS:
        __getattr__(_name)
