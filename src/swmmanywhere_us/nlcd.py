"""Download NLCD data from USGS MRLC via WMS.

Provides access to NLCD land cover and fractional imperviousness
datasets using GeoServer WMS endpoints, following the same tiling
and download pattern as ``seamless_3dep.get_map``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import tiny_retriever as terry
from seamless_3dep import decompose_bbox

from swmmanywhere_us.logging import logger

__all__ = ["download_nlcd"]

MAX_PIXELS = 8_000_000

_WMS_CONFIGS: dict[str, dict[str, str]] = {
    "imperviousness": {
        "workspace": "mrlc_Factional-Impervious-Surface-Native_conus_year_data",
        "layer": "Factional-Impervious-Surface-Native_conus_year_data",
        "prefix": "imperviousness",
    },
    "land_cover": {
        "workspace": "mrlc_Land-Cover-Native_conus_year_data",
        "layer": "Land-Cover-Native_conus_year_data",
        "prefix": "land_cover",
    },
}


def _create_hash(box: tuple[float, float, float, float], res: int) -> str:
    return hashlib.sha256(",".join(map(str, [*box, res])).encode()).hexdigest()


def download_nlcd(
    dataset: Literal["imperviousness", "land_cover"],
    bbox: Sequence[float],
    save_dir: str | Path,
    year: int = 2021,
    res: int = 30,
    pixel_max: int | None = MAX_PIXELS,
) -> list[Path]:
    """Download NLCD data via WMS from USGS MRLC GeoServer.

    Parameters
    ----------
    dataset : {"imperviousness", "land_cover"}
        Type of NLCD dataset to download.
    bbox : tuple
        Bounding box in decimal degrees (west, south, east, north).
    save_dir : str or Path
        Directory to save downloaded GeoTIFF files.
    year : int, optional
        NLCD data year, by default 2021.
    res : int, optional
        Resolution in meters, by default 30 (NLCD native).
    pixel_max : int or None, optional
        Maximum pixels per tile for domain decomposition, by default 8 million.

    Returns
    -------
    list of Path
        Downloaded GeoTIFF files.
    """
    if dataset not in _WMS_CONFIGS:
        msg = f"`dataset` must be one of {list(_WMS_CONFIGS)}"
        raise ValueError(msg)

    if pixel_max is not None and pixel_max > MAX_PIXELS:
        msg = f"`pixel_max` must be less than {MAX_PIXELS}."
        raise ValueError(msg)

    cfg = _WMS_CONFIGS[dataset]
    workspace = cfg["workspace"]
    layer = cfg["layer"]
    prefix = cfg["prefix"]

    bbox = tuple(bbox)
    bbox_list, sub_width, sub_height = decompose_bbox(bbox, res, pixel_max)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tiff_list = [save_dir / f"{prefix}_{_create_hash(box, res)}.tiff" for box in bbox_list]
    if all(t.exists() for t in tiff_list):
        return tiff_list

    logger.info(f"Downloading NLCD {dataset} ({year}) at {res} m resolution via WMS")

    base_url = f"https://dmsdata.cr.usgs.gov/geoserver/{workspace}/wms"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "SRS": "EPSG:4326",
        "FORMAT": "image/geotiff",
        "WIDTH": str(sub_width),
        "HEIGHT": str(sub_height),
    }
    qs = urlencode(params)
    urls = [f"{base_url}?{qs}&BBOX={box[0]},{box[1]},{box[2]},{box[3]}" for box in bbox_list]
    terry.download(urls, tiff_list)
    return tiff_list
