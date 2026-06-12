"""Write the final synthetic network to a single GeoPackage for GIS review.

Model generation calls :func:`write_geopackage` as its last step, so every
run produces a self-contained ``network.gpkg`` — one descriptively-named
layer per element class, the graph CRS embedded in each — that opens and
styles in QGIS without hunting through the model directory's loose files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import shapely

from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    import networkx as nx

# One line layer per edge class.  Grate orifices are excluded here and
# written instead as a point layer (see write_geopackage).
_EDGE_LAYERS: dict[str, set[str]] = {
    "edges_pipe": {"pipe"},
    "edges_street_channel": {"street_channel"},
    "edges_river": {"river"},
    "edges_outfall": {"outfall", "waste-outfall"},
    "edges_pond_inflow": {"pond_inflow"},
    "edges_pond_outflow": {"pond_outflow", "pond_connector"},
    "edges_orifice_weir": {"orifice", "weir"},
}
_EDGE_ATTRS = ("id", "edge_type", "diameter", "length", "is_trunk", "contributing_area")
_GRATE_ATTRS = ("id", "edge_type", "orifice_type", "orifice_diam_m", "orifice_cd", "flap_gate")
_NODE_ATTRS = (
    "node_type",
    "surface_elevation",
    "chamber_floor_elevation",
    "wb_closed_basin",
    "contributing_area",
    "max_depth",
)


def _is_grate(d: dict[str, Any]) -> bool:
    """A catchbasin grate orifice — drawn as a point, not a line."""
    return d.get("edge_type") == "orifice" and str(d.get("id", "")).endswith("-grate")


def _edge_geometry(nodes: dict[Any, dict[str, Any]], u: Any, v: Any, d: dict[str, Any]) -> Any:
    """Best-effort LineString for an edge — its stored geometry, else node-to-node."""
    g = d.get("geometry")
    if isinstance(g, shapely.geometry.base.BaseGeometry):
        return g if not g.is_empty else None
    if isinstance(g, dict) and g.get("type") == "LineString":
        return shapely.LineString(g["coordinates"])
    if isinstance(g, str):
        try:
            geo = shapely.from_wkt(g)
        except Exception:  # noqa: BLE001
            geo = None
        if geo is not None and not geo.is_empty:
            return geo
    su, tv = nodes.get(u), nodes.get(v)
    if su and tv and "x" in su and "x" in tv:
        return shapely.LineString([(su["x"], su["y"]), (tv["x"], tv["y"])])
    return None


def write_geopackage(  # noqa: C901, PLR0912, PLR0915 - one coherent graph -> GeoPackage translation, one section per layer class
    graph: nx.Graph[Any], model_dir: Path
) -> Path:
    """Write the final network to ``<model_dir>/network.gpkg``.

    A single GeoPackage with one descriptively-named layer per element
    class (``edges_pipe``, ``edges_river``, ``nodes_junction``,
    ``subcatchments`` ...) and the graph CRS embedded in every layer.

    Catchbasin grate orifices are written as a ``grates`` *point* layer:
    a grate is a vertical structure linking a surface node to the pipe
    junction directly below it (same x, y), so a line geometry would
    only ever be a degenerate sub-metre stub.

    Args:
        graph: The final pipeline graph.
        model_dir: The model output directory.

    Returns:
        Path to the written GeoPackage.
    """
    crs = graph.graph.get("crs")
    out = Path(model_dir) / "network.gpkg"
    if out.exists():
        out.unlink()
    nodes: dict[Any, dict[str, Any]] = dict(graph.nodes(data=True))
    edges = list(graph.edges(data=True))
    n_layers = 0

    # --- edge line layers ---
    for layer, types in _EDGE_LAYERS.items():
        recs: list[dict[str, Any]] = []
        for u, v, d in edges:
            if d.get("edge_type") not in types or _is_grate(d):
                continue
            geom = _edge_geometry(nodes, u, v, d)
            if geom is None:
                continue
            rec = {a: d.get(a) for a in _EDGE_ATTRS}
            rec["source"], rec["target"], rec["geometry"] = u, v, geom
            recs.append(rec)
        if recs:
            gpd.GeoDataFrame(recs, geometry="geometry", crs=crs).to_file(
                out, layer=layer, driver="GPKG"
            )
            n_layers += 1

    # --- catchbasin grate orifices, as points ---
    grate_recs: list[dict[str, Any]] = []
    for u, v, d in edges:
        if not _is_grate(d):
            continue
        node_data = nodes.get(v) or nodes.get(u) or {}
        if "x" not in node_data or "y" not in node_data:
            continue
        rec = {a: d.get(a) for a in _GRATE_ATTRS}
        rec["source"], rec["target"] = u, v
        rec["geometry"] = shapely.Point(node_data["x"], node_data["y"])
        grate_recs.append(rec)
    if grate_recs:
        gpd.GeoDataFrame(grate_recs, geometry="geometry", crs=crs).to_file(
            out, layer="grates", driver="GPKG"
        )
        n_layers += 1

    # --- node point layers ---
    node_layers: dict[str, list[dict[str, Any]]] = {
        "nodes_junction": [],
        "nodes_pond": [],
        "nodes_outlet_junction": [],
        "nodes_outfall": [],
    }
    has_outgoing = {u for u, _v in graph.edges()}
    for nid, n in nodes.items():
        if "x" not in n or "y" not in n:
            continue
        nt = n.get("node_type")
        rec = {a: n.get(a) for a in _NODE_ATTRS}
        rec["id"] = nid
        rec["geometry"] = shapely.Point(n["x"], n["y"])
        if nt == "water_body":
            node_layers["nodes_pond"].append(rec)
        elif nt == "outlet_junction":
            node_layers["nodes_outlet_junction"].append(rec)
        elif nt in ("dummy_river", "outfall", "waste") or nid not in has_outgoing:
            # Explicitly-typed outfalls plus any untyped sink (out-degree 0).
            node_layers["nodes_outfall"].append(rec)
        else:
            node_layers["nodes_junction"].append(rec)
    for layer, recs in node_layers.items():
        if recs:
            gpd.GeoDataFrame(recs, geometry="geometry", crs=crs).to_file(
                out, layer=layer, driver="GPKG"
            )
            n_layers += 1

    # --- subcatchment / pondshed polygon layers ---
    for name in ("subcatchments", "pondsheds"):
        p = Path(model_dir) / f"{name}.parquet"
        if not p.exists():
            continue
        poly = gpd.read_parquet(p)
        if poly.crs is None and crs is not None:
            poly = poly.set_crs(crs, allow_override=True)
        poly.to_file(out, layer=name, driver="GPKG")
        n_layers += 1

    logger.info(f"write_geopackage: wrote {n_layers} layer(s) -> {out}")
    return out
