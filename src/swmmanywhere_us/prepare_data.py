"""Prepare data module for SWMManywhere.

A module to download data needed for SWMManywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import rioxarray
import seamless_3dep as s3dep
import shapely
import tiny_osm
import tiny_retriever as terry
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from scipy.ndimage import binary_dilation

from swmmanywhere_us.logging import logger
from swmmanywhere_us.nlcd import download_nlcd

if TYPE_CHECKING:
    from pyproj import CRS


_POND_MIN_AREA_M2 = 200.0
_POND_MAX_AREA_M2 = 500_000.0


def _fetch_osm_utm(
    bbox: tuple[float, float, float, float],
    osm_filter: str | tuple[str, ...],
) -> gpd.GeoDataFrame:
    """Fetch OSM features within a bbox and reproject to local UTM."""
    fc = tiny_osm.fetch(*bbox, osm_filter=osm_filter)
    features = fc["features"]
    if not features:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=4326))
    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    return gdf.to_crs(gdf.estimate_utm_crs())


def download_street(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Get street network within a bounding box from OpenStreetMap.

    [CREDIT: Taher Cheghini busn_estimator package]

    Args:
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form
            of (west, south, east, north) at EPSG:4326.

    Returns:
        gpd.GeoDataFrame: Street network as a GeoDataFrame projected to UTM.
    """
    return _fetch_osm_utm(bbox, tiny_osm.OSMFilters.HIGHWAY)


def download_river(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Get water network within a bounding box from OpenStreetMap.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form
            of (west, south, east, north) at EPSG:4326.

    Returns:
        gpd.GeoDataFrame: River network as a GeoDataFrame projected to UTM.
    """
    gdf = _fetch_osm_utm(bbox, tiny_osm.OSMFilters.WATERWAY)
    if gdf.empty:
        logger.warning("No water network found within the bounding box.")
    return gdf


def download_lulc(
    lulc_path: Path, bbox: tuple[float, float, float, float], target_utm: str | CRS, cover_year: int
) -> None:
    """Download land use land cover (LULC) data from NLCD for the given bounding box.

    Args:
        lulc_path (Path): File path to save the downloaded NLCD data.
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form
            of (west, south, east, north) at EPSG:4326.
        target_utm (str): Target UTM EPSG code for reprojection.
        cover_year (int): Year of NLCD land cover data to download.
    """
    if lulc_path.exists():
        logger.info(f"LULC data already exists at {lulc_path}, skipping download.")
        return

    bbox_utm = transform_bounds(4326, target_utm, *bbox)
    nlcd_res = 30
    bbox_buff_utm = shapely.box(*bbox_utm).buffer(50 * nlcd_res).bounds
    bbox_buff = transform_bounds(target_utm, 4326, *bbox_buff_utm)

    tiffs = download_nlcd("land_cover", bbox_buff, lulc_path.parent, year=cover_year, res=nlcd_res)
    cover = s3dep.tiffs_to_da(tiffs, bbox_buff).rio.reproject(target_utm).rio.clip_box(*bbox_utm)
    cover.rio.to_raster(lulc_path)


def download_imperviousness(
    imperviousness_path: Path,
    bbox: tuple[float, float, float, float],
    target_utm: str | CRS,
    year: int = 2021,
) -> None:
    """Download fractional imperviousness data from NLCD for the given bounding box.

    Args:
        imperviousness_path (Path): File path to save the downloaded data.
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form
            of (west, south, east, north) at EPSG:4326.
        target_utm (str): Target UTM EPSG code for reprojection.
        year (int): Year of NLCD imperviousness data to download.
    """
    if imperviousness_path.exists():
        logger.info(
            f"Imperviousness data already exists at {imperviousness_path}, skipping download."
        )
        return

    bbox_utm = transform_bounds(4326, target_utm, *bbox)
    nlcd_res = 30
    bbox_buff_utm = shapely.box(*bbox_utm).buffer(50 * nlcd_res).bounds
    bbox_buff = transform_bounds(target_utm, 4326, *bbox_buff_utm)

    tiffs = download_nlcd(
        "imperviousness", bbox_buff, imperviousness_path.parent, year=year, res=nlcd_res
    )
    imperv = (
        s3dep.tiffs_to_da(tiffs, bbox_buff)
        .rio.reproject(target_utm, resampling=Resampling.bilinear)
        .rio.clip_box(*bbox_utm)
    )
    imperv.rio.to_raster(imperviousness_path)


def _hydroflatten_dem(  # noqa: C901, PLR0912, PLR0915 - tight inner chunk-processing loop is clearer kept together than split
    dem_path: Path,
    waterbodies_gds: gpd.GeoSeries,
    output_path: str | Path,
    chunk_size: int = 2048,
) -> None:
    """Hydroflatten a DEM using water body polygons with rioxarray and chunked processing.

    This function sets all water surface elevations to a constant value determined
    by the minimum elevation at each water body's boundary. This function uses
    ``rioxarray`` for reading the DEM and ``rasterio`` for chunked writing,
    which is more memory efficient for large DEMs.

    Args:
        dem_path (Path): Path to the input DEM raster file.
        waterbodies_gdf (gpd.GeoSeries): GeoSeries of water body polygons.
        output_path (pathlib.Path): Path for output hydroflattened DEM.
        chunk_size (int): Size of processing chunks in pixels. Larger values use more
            memory but may be faster. Defaults to 2048.
    """
    logger.info("Hydroflattening DEM using water bodies")

    dem = rioxarray.open_rasterio(dem_path, masked=True)
    if not isinstance(dem, xr.DataArray):
        msg = "Expected DEM to be a single-band raster, but got multiple bands."
        raise TypeError(msg)
    dem = dem.squeeze("band", drop=True)
    if waterbodies_gds.crs != dem.rio.crs:
        waterbodies_gds = waterbodies_gds.to_crs(dem.rio.crs)

    logger.info("Pre-computing flattening elevation for each water body")
    # Pre-compute flattening elevation for each water body
    wb_gdf = gpd.GeoDataFrame(geometry=waterbodies_gds)
    wb_gdf["flatten_elev"] = np.nan
    for idx, water_body in wb_gdf.geometry.items():
        clipped = dem.rio.clip_box(*water_body.bounds, auto_expand=True)
        if clipped.size == 0:
            continue

        data = clipped.to_numpy()
        water_mask = rasterize(
            [(water_body, 1)],
            out_shape=data.shape,
            transform=clipped.rio.transform(),
            fill=0,
            dtype="uint8",
        )
        if water_mask is None:
            msg = "Rasterization of water body returned None"
            raise ValueError(msg)
        water_mask = water_mask.astype(bool)

        if not water_mask.any():
            continue

        boundary_mask = binary_dilation(water_mask, iterations=1) & ~water_mask
        boundary_vals = data[boundary_mask]

        # Handle masked arrays from rioxarray
        if np.ma.is_masked(boundary_vals):
            boundary_vals = boundary_vals.compressed()

        if boundary_vals.size > 0:
            wb_gdf.loc[idx, "flatten_elev"] = np.nanmin(boundary_vals)
        else:
            water_vals = data[water_mask]
            if np.ma.is_masked(water_vals):
                water_vals = water_vals.compressed()
            if water_vals.size > 0:
                wb_gdf.loc[idx, "flatten_elev"] = np.nanmin(water_vals)
    dem.close()

    wb_gdf = wb_gdf[wb_gdf["flatten_elev"].notna()]
    wb_sidx = wb_gdf.sindex

    logger.info(f"Applying hydroflattening to DEM in chunks of size {chunk_size}x{chunk_size}")
    # Use rasterio for chunked writing (more efficient than rioxarray)
    with (
        rasterio.open(dem_path) as src,
        rasterio.open(output_path, "w", **src.profile) as dst,
    ):
        for row_start in range(0, src.height, chunk_size):
            for col_start in range(0, src.width, chunk_size):
                height = min(chunk_size, src.height - row_start)
                width = min(chunk_size, src.width - col_start)
                window = Window(col_start, row_start, width, height)  # pyright: ignore[reportCallIssue]

                chunk = src.read(1, window=window)
                chunk_transform = src.window_transform(window)
                chunk_bounds = rasterio.windows.bounds(window, src.transform)
                chunk_bbox = shapely.box(*chunk_bounds)

                possible_idx = list(wb_sidx.intersection(chunk_bounds))
                if not possible_idx:
                    dst.write(chunk, 1, window=window)
                    continue

                intersecting = wb_gdf.iloc[list(map(int, possible_idx))]
                intersecting = intersecting[intersecting.intersects(chunk_bbox)]

                if len(intersecting) == 0:
                    dst.write(chunk, 1, window=window)
                    continue

                flattened = chunk.copy()
                for water_body, elev in intersecting[["geometry", "flatten_elev"]].itertuples(
                    index=False, name=None
                ):
                    water_mask = rasterize(
                        [(water_body, 1)],
                        out_shape=chunk.shape,
                        transform=chunk_transform,
                        fill=0,
                        dtype="uint8",
                    )
                    if water_mask is None:
                        msg = "Rasterization of water body returned None"
                        raise ValueError(msg)
                    water_mask = water_mask.astype(bool)

                    if water_mask.any():
                        flattened[water_mask] = elev
                dst.write(flattened, 1, window=window)


def download_basins(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Get OSM water-body polygons within a bounding box.

    Applies a broad area filter only; pond-vs-lake classification happens
    downstream in ``water_bodies._load_water_body_polygons`` so it can use
    the OSM tags preserved here (``natural``, ``water``, ``basin``,
    ``landuse``) alongside size-based rules.

    Args:
        bbox: Bounding box as (west, south, east, north) in EPSG:4326.

    Returns:
        Polygon GeoDataFrame projected to UTM with all OSM classification
        tags preserved.
    """
    gdf = _fetch_osm_utm(bbox, tiny_osm.OSMFilters.WATER_BODY)
    if not gdf.empty:
        area = gdf.geometry.area
        gdf = gdf[(area >= _POND_MIN_AREA_M2) & (area <= _POND_MAX_AREA_M2)].reset_index(drop=True)
        gdf["geometry"] = shapely.polygons(  # pyright: ignore[reportCallIssue, reportArgumentType]
            shapely.get_exterior_ring(gdf.geometry.to_numpy())
        )
    if gdf.empty:
        logger.warning("No basins found within the bounding box.")
    return gdf


def download_elevation(
    fid: Path,
    bbox: tuple[float, float, float, float],
    dem_res: int,
    lulc_path: Path,
    water_bodies_path: Path,
    extra_waterbody: Path | None,
    utm: str,
) -> None:
    """Download elevation data from USGS 3DEP and hydroflatten using waterbody polygons.

    Args:
        fid (Path): File path to save the downloaded elevation data.
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form
            of (west, south, east, north) at EPSG:4326.
        dem_res (int): Desired DEM resolution in meters.
        lulc_path (Path): Path to land use land cover file.
        extra_waterbody (Path | None): Path to extra waterbody file (if any).
        utm (str): UTM EPSG code for the area of interest.

    Author:
        cheginit
    """
    if fid.exists():
        logger.info(f"Elevation data already exists at {fid}, skipping download.")
        return
    logger.info(f"Downloading elevation data at {dem_res} m resolution from 3DEP")
    bbox_utm = transform_bounds(4326, utm, *bbox)
    bbox_buff_utm = shapely.box(*bbox_utm).buffer(50 * dem_res).bounds
    bbox_buff = transform_bounds(utm, 4326, *bbox_buff_utm)

    tiff_files = s3dep.get_map("DEM", bbox_buff, fid.parent, res=dem_res)
    # Reproject with bilinear resampling.  rioxarray's default is
    # nearest-neighbour, which stair-steps a continuous DEM and corrupts
    # the derived slope and D8 flow direction on flat terrain; bilinear
    # interpolates a smooth surface.
    dem = (
        s3dep.tiffs_to_da(tiff_files, bbox_buff)
        .rio.reproject(utm, resampling=Resampling.bilinear)
        .rio.clip_box(*bbox_utm)
    )
    dem_file_raw = fid.with_stem(fid.stem + "_raw")
    dem.rio.to_raster(dem_file_raw)

    logger.info("Extracting water bodies from NLCD data for hydroflattening")
    with rasterio.open(lulc_path) as src:
        cover_data = src.read(1)
        cover_transform = src.transform
        cover_crs = src.crs
        lulc_res = abs(cover_transform.a)

    water_mask = (cover_data == 11).astype(np.uint8)
    min_waterbody_areasqm = 9 * lulc_res**2
    water_polys = [
        shapely.geometry.shape(geom)
        for geom, val in shapes(water_mask, transform=cover_transform)
        if val == 1
    ]
    water_bodies = gpd.GeoSeries(water_polys, crs=cover_crs)
    water_bodies = water_bodies.loc[water_bodies.area >= min_waterbody_areasqm]
    water_bodies = water_bodies.buffer(lulc_res).buffer(-lulc_res)

    if extra_waterbody:
        logger.info("Merging provided water bodies with NLCD water bodies")
        extra_wb = gpd.read_file(extra_waterbody)
        extra_wb = extra_wb.to_crs(utm).clip_by_rect(*bbox_utm)
        extra_wb = extra_wb[~extra_wb.geometry.is_empty].force_2d()  # pyright: ignore[reportCallIssue]
        water_bodies = gpd.GeoSeries(pd.concat([extra_wb, water_bodies], ignore_index=True))

    gpd.GeoDataFrame(geometry=water_bodies).to_parquet(water_bodies_path)
    logger.info(f"Saved water bodies to {water_bodies_path}")

    _hydroflatten_dem(dem_file_raw, water_bodies, fid)


def get_design_precipitation(
    lat: float,
    lon: float,
    return_period: int = 10,
    duration: str = "60-min",
    cache_dir: Path | None = None,
) -> float:
    """Query NOAA Atlas 14 for design precipitation intensity.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees (negative for western hemisphere).
        return_period: Design storm return period in years (1-1000).
        duration: Storm duration string matching Atlas 14 row labels
            (e.g., "5-min", "10-min", "15-min", "30-min", "60-min",
            "2-hr", "3-hr", "6-hr", "12-hr", "24-hr").
        cache_dir: Directory for caching API responses. Defaults to .cache/.

    Returns:
        Design precipitation intensity in m/hr.

    Raises:
        ValueError: If the requested duration or return period is not found
            in the API response.
    """
    if cache_dir is None:
        cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True, parents=True)

    cache_file = cache_dir / f"noaa_atlas14_{lat:.4f}_{lon:.4f}.csv"
    if cache_file.exists():
        text = cache_file.read_text()
    else:
        url = "https://hdsc.nws.noaa.gov/cgi-bin/new/fe_text_mean.csv"
        params = {
            "lat": lat,
            "lon": lon,
            "data": "intensity",
            "series": "pds",
            "units": "metric",
        }
        text = terry.fetch(url, "text", request_kwargs={"params": params})
        cache_file.write_text(text)
        logger.info(f"Cached NOAA Atlas 14 data to {cache_file}")

    # Parse the CSV to extract the requested value.
    # Format:
    #   ...header lines...
    #   by duration for ARI (years):, 1,2,5,10,25,50,100,200,500,1000
    #   5-min:, 169,195,236,269,315,349,383,417,461,494
    #   60-min:, 49,56,68,78,91,101,111,121,134,144
    #   ...
    ari_cols: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("by duration for ARI"):
            ari_cols = [int(x.strip()) for x in line.split(",")[1:] if x.strip()]
            continue
        if not ari_cols or ":" not in line:
            continue
        label, _, values_str = line.partition(",")
        label = label.strip().rstrip(":")
        if label.lower() != duration.lower():
            continue
        values = [float(x.strip()) for x in values_str.split(",") if x.strip()]
        if return_period not in ari_cols:
            msg = f"Return period {return_period} not in Atlas 14 columns: {ari_cols}"
            raise ValueError(msg)
        idx = ari_cols.index(return_period)
        intensity_mm_hr = values[idx]
        intensity_m_hr = intensity_mm_hr / 1000
        logger.info(
            f"NOAA Atlas 14 ({lat:.2f}, {lon:.2f}): "
            f"{return_period}-yr {duration} intensity = {intensity_mm_hr:.0f} mm/hr"
        )
        return intensity_m_hr

    msg = f"Duration '{duration}' not found in Atlas 14 response for ({lat}, {lon})"
    raise ValueError(msg)
