"""Preprocessing module for SWMManywhere.

A module to call downloads, preprocess these downloads into formats suitable
for graphfcns, and some other utilities (such as creating a project folder
structure or create the starting graph from rivers/streets).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely
from rasterio.warp import transform_bounds

from swmmanywhere_us import geospatial_utilities as go
from swmmanywhere_us import prepare_data
from swmmanywhere_us.logging import logger

if TYPE_CHECKING:
    from swmmanywhere_us.filepaths import FilePaths


def prepare_street(
    bbox: tuple[float, float, float, float],
    addresses: FilePaths,
    target_crs: str,
):
    """Download street network as a GeoDataFrame and save as parquet.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in
            the format (minx, miny, maxx, maxy) in EPSG:4326.
        addresses (FilePaths): Class containing the addresses of the directories.
        target_crs (str): Target CRS to reproject the data to.
    """
    if addresses.bbox_paths.street.exists():
        logger.info(
            f"Street network already exists at {addresses.bbox_paths.street}, skipping download"
        )
        return
    logger.info(f"Downloading network to {addresses.bbox_paths.street}")
    streets = prepare_data.download_street(bbox)
    streets = streets.to_crs(target_crs)

    # OSM returns the full geometry of every way intersecting the bbox, so
    # boundary-crossing roads sprawl ~1-2 km past the AOI.  Those out-of-extent
    # nodes fall outside the downloaded DEM (clipped to the bbox) and pick up a
    # 0 m nodata elevation that reads as a deep pit, forcing uphill flow.
    # Truncate ways at the same bbox the DEM uses so the street network
    # respects the declared extent.
    bbox_utm = transform_bounds(4326, target_crs, *bbox)
    streets["geometry"] = streets.geometry.clip_by_rect(*bbox_utm)
    streets = streets[~streets.geometry.is_empty].explode(index_parts=False)
    streets = streets[streets.geometry.geom_type == "LineString"].reset_index(drop=True)

    streets.to_parquet(addresses.bbox_paths.street)


def prepare_river(
    bbox: tuple[float, float, float, float],
    addresses: FilePaths,
    target_crs: str,
):
    """Download river network as a GeoDataFrame and save as parquet."""
    if addresses.bbox_paths.river.exists():
        logger.info(
            f"River network already exists at {addresses.bbox_paths.river}, skipping download."
        )
        return
    logger.info(f"Downloading river network to {addresses.bbox_paths.river}")
    rivers = prepare_data.download_river(bbox)
    if not rivers.empty:
        rivers = rivers.to_crs(target_crs)
    rivers.to_parquet(addresses.bbox_paths.river)


def prepare_tiger_rail(
    bbox: tuple[float, float, float, float],
    addresses: FilePaths,
    target_crs: str,
) -> None:
    """Download TIGER rail data in bbox to file.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in
            the format (minx, miny, maxx, maxy) in EPSG:4326.
        addresses (FilePaths): Class containing the addresses of the directories.
        target_crs (str): Target CRS to reproject the graph to.
        source_crs (str): Source CRS of the graph.
    """
    if addresses.bbox_paths.tiger_rail.exists():
        logger.info(
            f"Rail data already exists at {addresses.bbox_paths.tiger_rail}, skipping download."
        )
        return

    if addresses.project_paths.national_rail.exists():
        gdf = gpd.read_parquet(addresses.project_paths.national_rail)
    else:
        logger.info(
            f"Downloading national TIGER 2025 rail data to {addresses.project_paths.national_rail}"
        )
        url = "https://www2.census.gov/geo/tiger/TIGER2025/RAILS/tl_2025_us_rails.zip"
        gdf = gpd.read_file(url).to_crs(4326)
        gdf.to_parquet(addresses.project_paths.national_rail)
    logger.info(f"Clipping national TIGER 2025 rail data to {addresses.bbox_paths.tiger_rail}")
    gdf["geometry"] = gdf.clip_by_rect(*bbox)
    if gdf.geometry.is_empty.all():
        logger.warning("No rail data found in the bounding box.")
        return
    gdf.to_crs(target_crs).to_parquet(addresses.bbox_paths.tiger_rail)


def prepare_basins(
    bbox: tuple[float, float, float, float],
    addresses: FilePaths,
    target_crs: str,
) -> None:
    """Download OSM detention/retention basins and save as parquet."""
    if addresses.bbox_paths.basins.exists():
        logger.info(
            f"Basin data already exists at {addresses.bbox_paths.basins}, skipping download."
        )
        return
    logger.info(f"Downloading basin data to {addresses.bbox_paths.basins}")
    basins = prepare_data.download_basins(bbox)
    if not basins.empty:
        basins = basins.to_crs(target_crs)
    basins.to_parquet(addresses.bbox_paths.basins)


def run_downloads(
    bbox: tuple[float, float, float, float],
    dem_res: int,
    lulc_year: int,
    addresses: FilePaths,
):
    """Run the data downloads.

    Run the precipitation, elevation, building, street and river network
    downloads. If the data already exists, do not download it again. Reprojects
    data to the UTM zone.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in
            the format (minx, miny, maxx, maxy) in EPSG:4326.
        dem_res (int): Resolution of the DEM to download (in meters).
        lulc_year (int): Year of the land use land cover data to use.
        addresses (FilePaths): Class containing the addresses of the directories.
    """
    target_crs = go.get_utm_epsg(bbox[0], bbox[1])

    prepare_data.download_lulc(
        addresses.bbox_paths.lulc,
        bbox,
        target_crs,
        lulc_year,
    )
    prepare_data.download_imperviousness(
        addresses.bbox_paths.imperviousness,
        bbox,
        target_crs,
        lulc_year,
    )
    prepare_data.download_elevation(
        addresses.bbox_paths.elevation,
        bbox,
        dem_res,
        addresses.bbox_paths.lulc,
        addresses.bbox_paths.water_bodies,
        addresses.bbox_paths.extra_waterbody,
        target_crs,
    )
    prepare_street(bbox, addresses, target_crs)
    prepare_river(bbox, addresses, target_crs)
    prepare_basins(bbox, addresses, target_crs)
    prepare_tiger_rail(bbox, addresses, target_crs)


def _gdf_to_graph(gdf: gpd.GeoDataFrame, edge_type: str) -> nx.MultiDiGraph[Any]:
    """Convert a GeoDataFrame of LineStrings to a networkx graph.

    Args:
        gdf: GeoDataFrame with LineString geometries.
        edge_type: Value for the ``edge_type`` edge attribute.

    Returns:
        nx.MultiDiGraph with integer-labeled nodes carrying ``x``/``y`` and
        edges carrying ``geometry``, ``length``, ``edge_type``, plus all other
        GDF columns.
    """
    graph = nx.MultiDiGraph()
    gdf = gdf.copy()
    if gdf.crs is not None:
        graph.graph["crs"] = gdf.crs.to_wkt()
        # Round to ~10cm precision to avoid floating point issues with OSM
        # data in lat/lon. For projected CRS, 2 decimal places is sufficient
        # to avoid issues while keeping more precision.
        if gdf.crs.is_geographic:
            gdf["geometry"] = gdf.geometry.set_precision(6)
        else:
            gdf["geometry"] = gdf.geometry.set_precision(2)
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    else:
        graph.graph["crs"] = None
    cols = gdf.columns[gdf.columns != "geometry"]
    if "lanes" in cols:
        gdf["lanes"] = gdf["lanes"].fillna(1).astype(int)
    # Map each unique (x, y) endpoint to a sequential integer node ID.
    coord_to_id: dict[tuple[float, float], int] = {}
    k: dict[tuple[int, int], int] = {}
    for geom, *row in gdf[["geometry", *cols]].itertuples(index=False, name=None):
        u_coord = geom.coords[0]
        v_coord = geom.coords[-1]
        # Skip closed-loop LineStrings (first coord == last coord).  These
        # carry no flow-routing information and would produce self-loop edges
        # that break topological sort downstream.
        if u_coord == v_coord:
            continue
        for coord in (u_coord, v_coord):
            if coord not in coord_to_id:
                nid = len(coord_to_id)
                coord_to_id[coord] = nid
                graph.add_node(nid, x=coord[0], y=coord[1])
        u_id = coord_to_id[u_coord]
        v_id = coord_to_id[v_coord]
        k[(u_id, v_id)] = k.get((u_id, v_id), 0) + 1
        attrs = {"geometry": geom, "length": geom.length, "edge_type": edge_type}
        attrs.update(zip(cols, row))
        graph.add_edge(u_id, v_id, key=k[(u_id, v_id)] - 1, **attrs)
    return graph


def create_starting_graph(addresses: FilePaths) -> nx.MultiDiGraph[Any]:
    """Create the starting graph.

    Create the starting graph by combining the street and river networks.
    Both networks share a single integer node ID space keyed by endpoint
    coordinates, so geometrically coincident street/river endpoints collapse
    into the same node.

    Args:
        addresses (FilePaths): Class containing the addresses of the directories.

    Returns:
        nx.Graph[Any]: Combined street and river network.
    """
    street_gdf = gpd.read_parquet(addresses.bbox_paths.street)
    street_crs = street_gdf.crs
    if street_crs is None:
        msg = "Street GeoDataFrame must have a CRS, but got None"
        raise ValueError(msg)
    street_gdf = street_gdf.copy()
    street_gdf["edge_type"] = "pipe"

    river_gdf = gpd.read_parquet(addresses.bbox_paths.river)
    if not river_gdf.empty:
        river_gdf = river_gdf.to_crs(street_crs)
        river_gdf["geometry"] = river_gdf.geometry.clip_by_rect(*street_gdf.total_bounds)
        river_gdf = gpd.GeoDataFrame(
            geometry=shapely.get_parts(river_gdf.geometry.to_numpy()), crs=river_gdf.crs
        )
        river_gdf = river_gdf[~river_gdf.geometry.is_empty].reset_index(drop=True)
        river_gdf["edge_type"] = "river"
        combined = gpd.GeoDataFrame(
            pd.concat([street_gdf, river_gdf], ignore_index=True),
            crs=street_crs,
        )
    else:
        combined = street_gdf

    return _gdf_to_graph(combined, edge_type="pipe")
