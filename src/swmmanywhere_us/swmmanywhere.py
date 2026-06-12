"""The main SWMManywhere-US module to generate a synthetic SWMM network."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import shapely
from pydantic import BaseModel, ConfigDict, Field
from rasterio.warp import transform_bounds

import swmmanywhere_us.geospatial_utilities as go
from swmmanywhere_us import filepaths, parameters, prepare_data, preprocessing
from swmmanywhere_us.geopackage import write_geopackage
from swmmanywhere_us.graph_utilities import filter_edges, save_graph
from swmmanywhere_us.logging import logger
from swmmanywhere_us.parameters import HydraulicDesign
from swmmanywhere_us.pipeline import build_pipeline
from swmmanywhere_us.post_processing import generate_pondsheds, synthetic_write
from swmmanywhere_us.swmm_defaults import SWMMOptions
from swmmanywhere_us.utilities import yaml_dump


def _get_original_bbox_4326(
    bbox_config: dict[str, float | int],
) -> tuple[float, float, float, float]:
    """The user's *unbuffered* area-of-interest bbox, in EPSG:4326.

    Captured before :func:`_get_extended_bbox` widens it by the
    processing halo, so final outputs (pondsheds, network) can be
    clipped back to the true AOI — discarding buffer-zone artifacts.
    """
    bbox = (
        float(bbox_config["xmin"]),
        float(bbox_config["ymin"]),
        float(bbox_config["xmax"]),
        float(bbox_config["ymax"]),
    )
    if int(bbox_config.get("crs_epsg", 4326)) != 4326:
        bbox = transform_bounds(bbox_config["crs_epsg"], 4326, *bbox)
        bbox = cast("tuple[float, float, float, float]", bbox)
    return cast("tuple[float, float, float, float]", tuple(round(c, 6) for c in bbox))


def _get_extended_bbox(
    bbox_config: dict[str, float | int],
) -> tuple[float, float, float, float]:
    """Get bounding box in EPSG:4326, optionally extended by a km buffer."""
    bbox = (
        float(bbox_config["xmin"]),
        float(bbox_config["ymin"]),
        float(bbox_config["xmax"]),
        float(bbox_config["ymax"]),
    )
    if int(bbox_config.get("crs_epsg", 4326)) != 4326:
        bbox = transform_bounds(bbox_config["crs_epsg"], 4326, *bbox)
        bbox = cast("tuple[float, float, float, float]", bbox)

    buffer_km = float(bbox_config.get("buffer_km", 0))
    if buffer_km <= 0:
        return cast("tuple[float, float, float, float]", tuple(round(c, 6) for c in bbox))

    logger.info(f"Extending bbox by {buffer_km} km buffer")
    center_x, center_y = shapely.box(*bbox).centroid.xy
    utm = go.get_utm_epsg(center_x[0], center_y[0])
    bbox_extended = transform_bounds(4326, utm, *bbox)
    bbox_extended = shapely.box(*bbox_extended).buffer(buffer_km * 1000).bounds
    bbox_extended = transform_bounds(utm, 4326, *bbox_extended)
    return cast("tuple[float, float, float, float]", tuple(round(c, 6) for c in bbox_extended))


class BboxConfig(BaseModel):
    """Bounding box configuration."""

    model_config = ConfigDict(extra="forbid")

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs_epsg: int = 4326
    buffer_km: float = 1


class SwmmSettings(BaseModel):
    """SWMM simulation settings."""

    model_config = ConfigDict(extra="forbid")

    rain_dat_path: Path | None = None
    rain_dat_unit: Literal["IN", "MM"] = "MM"
    inp_options: SWMMOptions = Field(default_factory=SWMMOptions)


class Config(BaseModel):
    """Top-level swmmanywhere-us run configuration."""

    model_config = ConfigDict(extra="forbid")

    base_dir: Path
    project: str
    bbox: BboxConfig
    bbox_number: int | None = None
    model_number: int | None = None
    add_pondsheds: bool = False
    bare_network: bool = False
    swmm_settings: SwmmSettings = Field(default_factory=SwmmSettings)
    paths_overrides: dict[str, Path] = Field(default_factory=dict)
    params_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration dictionary and apply defaults via pydantic."""
    validated = Config.model_validate(config)
    result = validated.model_dump(mode="python")

    result["base_dir"] = Path(result["base_dir"])
    result["base_dir"].mkdir(parents=True, exist_ok=True)

    for key, path in result["paths_overrides"].items():
        if not Path(path).exists():
            msg = f"{key} not found at {path}"
            raise FileNotFoundError(msg)
        result["paths_overrides"][key] = Path(path)

    rain_dat_path = result["swmm_settings"].get("rain_dat_path")
    if rain_dat_path is not None:
        result["swmm_settings"]["rain_dat_path"] = Path(rain_dat_path)

    params = parameters.get_full_parameters()
    for category, cat_overrides in result["params_overrides"].items():
        if category not in params:
            msg = f"{category} not a category of parameter. Must be one of {list(params)}."
            raise ValueError(msg)
        cat_properties = params[category].model_json_schema()["properties"]
        for key in cat_overrides:
            if key not in cat_properties:
                msg = f"{key} not found in {category}."
                raise ValueError(msg)

    return result


def swmmanywhere(config: dict[str, Any]) -> Path:
    """Run SWMManywhere-US to generate a synthetic SWMM network.

    Args:
        config: Configuration dictionary with keys: base_dir, project, bbox,
            and optionally swmm_settings, params_overrides, paths_overrides.

    Returns:
        Path to the generated .inp file.
    """
    config = _validate_config(config)

    # Capture the user's true area-of-interest BEFORE the buffer widens
    # it, so final outputs can be clipped back to it (edge-artifact
    # removal: process a halo, deliver only the AOI).
    original_bbox_4326 = _get_original_bbox_4326(config["bbox"])
    config["bbox"] = _get_extended_bbox(config["bbox"])
    logger.info("Creating project structure.")

    paths_overrides = config.get("paths_overrides", {})
    for key, path in paths_overrides.items():
        logger.info(f"Overriding path {key} with {path}")
    swmm_settings: dict[str, Any] = config.get("swmm_settings", {})

    addresses = filepaths.FilePaths(
        config["base_dir"],
        config["project"],
        config["bbox"],
        bbox_number=config.get("bbox_number", None),
        model_number=config.get("model_number", None),
        **paths_overrides,
    )

    logger.info(f"Project structure created at {addresses.project_paths.base_dir}")
    logger.info(f"Project name: {config['project']}")
    logger.info(f"Bounding box: {config['bbox']}, number: {addresses.bbox_paths.bbox_number}")
    logger.info(f"Model number: {addresses.model_paths.model_number}")

    yaml_dump(config, (addresses.model_paths.model / "config.yml").open("w"))

    logger.info("Loading and setting parameters.")
    params = parameters.get_full_parameters()
    for category, overrides in config.get("params_overrides", {}).items():
        for key, val in overrides.items():
            logger.info(f"Setting {category} {key} to {val}")
            setattr(params[category], key, val)

    # Query NOAA Atlas 14 for design precipitation if not explicitly set
    hd = params["hydraulic_design"]
    if hd.precipitation == HydraulicDesign().precipitation:
        bbox = config["bbox"]
        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        try:
            hd.precipitation = prepare_data.get_design_precipitation(
                center_lat,
                center_lon,
                return_period=hd.design_return_period,
                duration=hd.design_duration,
                cache_dir=config["base_dir"] / ".cache",
            )
        except Exception:  # noqa: BLE001 - best-effort: network errors, CSV parse errors, etc. fall back to default precip
            logger.warning(
                f"Failed to query NOAA Atlas 14, using default precipitation: "
                f"{hd.precipitation * 1000:.1f} mm/hr"
            )

    logger.info("Downloading required input data...")
    preprocessing.run_downloads(
        config["bbox"],
        params["subcatchment_derivation"].dem_resolution,
        params["subcatchment_derivation"].lulc_year,
        addresses,
    )

    logger.info("Creating starting graph...")
    graph = preprocessing.create_starting_graph(addresses)

    add_pondsheds = bool(config.get("add_pondsheds", False))
    bare_network = bool(config.get("bare_network", False))
    logger.info(
        f"Running graph transformation pipeline "
        f"(add_pondsheds={add_pondsheds}, bare_network={bare_network})..."
    )
    graph = build_pipeline(add_pondsheds, bare=bare_network).run(
        graph, params, addresses, checkpoint_dir=addresses.model_paths.model
    )

    logger.info("Saving final graph and writing inp file...")
    go.graph_to_geojson(
        graph, addresses.model_paths.nodes, addresses.model_paths.edges, graph.graph["crs"]
    )
    save_graph(graph, addresses.model_paths.graph)

    if len(filter_edges(graph, frozenset({"pipe", "river"})).edges) == 0:
        logger.warning("No edges in graph, returning graph file...")
        return addresses.model_paths.graph

    synthetic_write(
        addresses,
        rain_dat_path=swmm_settings.get("rain_dat_path"),
        rain_dat_unit=swmm_settings.get("rain_dat_unit", "MM"),
        inp_options=swmm_settings.get("inp_options"),
        pond_design=params["pond_design"],
    )

    # Visualization aid: per-pond drainage-area polygons.  Only produced
    # when the pond subsystem is active (``add_pondsheds=True``); a plain
    # dual-drainage network has no ponds and hence no pondsheds.
    #
    # AOI clipping (``clip_bbox_4326``) is intentionally OFF — the full
    # buffered extent is kept for diagnostics.  ``original_bbox_4326`` is
    # retained for if/when AOI clipping of final deliverables is desired.
    _ = original_bbox_4326  # reserved for optional AOI clipping
    if add_pondsheds and not bare_network:
        generate_pondsheds(addresses)

    # Single-file GIS deliverable: every element class as its own layer.
    # A failure here must not lose the run — the .inp is the authoritative
    # output, the GeoPackage is only a review aid.
    try:
        write_geopackage(graph, addresses.model_paths.model)
    except Exception:  # noqa: BLE001
        logger.warning("write_geopackage failed; .inp output is unaffected.", exc_info=True)

    return addresses.model_paths.inp
