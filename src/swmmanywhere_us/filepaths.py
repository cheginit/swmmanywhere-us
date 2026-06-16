"""File paths module for SWMMAnywhere."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import orjson as json

from swmmanywhere_us.logging import logger


def next_directory(keyword: str, directory: Path) -> int:
    """Find the next directory number.

    Find the next directory number within a directory with a <keyword>_ in its
    name.

    Args:
        keyword (str): Keyword to search for in the directory name.
        directory (Path): Path to the directory to search within.

    Returns:
        int: Next directory number.
    """
    existing_dirs = [int(d.name.split("_")[-1]) for d in directory.glob(f"{keyword}_*")]
    return 1 if not existing_dirs else max(existing_dirs) + 1


def check_bboxes(bbox: tuple[float, float, float, float], data_dir: Path) -> int | bool:
    """Find the bounding box number.

    Check if the bounding box coordinates match any existing bounding box
    directories within data_dir.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in
            the format (minx, miny, maxx, maxy).
        data_dir (Path): Path to the data directory.

    Returns:
        int: Bounding box number if the coordinates match, else False.
    """
    # Find all bounding_box_info.json files
    info_fids = data_dir.glob("*/*bounding_box_info.json")

    # Iterate over info files
    for info_fid in info_fids:
        # Read bounding_box_info.json
        bounding_info = json.loads(info_fid.read_text())
        # Check if the bounding box coordinates match
        if Counter(bounding_info.get("bbox")) == Counter(bbox):
            bbox_full_dir = info_fid.parent
            bbox_dir = bbox_full_dir.name
            return int(bbox_dir.replace("bbox_", ""))

    return False


def get_next_bbox_number(bbox: tuple[float, float, float, float], data_dir: Path) -> int:
    """Get the next bounding box number.

    If there are existing bounding box directories, check within them to see if
    any have the same bounding box, otherwise find the next number. If
    there are no existing bounding box directories, return 1.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in
            the format (minx, miny, maxx, maxy).
        data_dir (Path): Path to the data directory.

    Returns:
        int: Next bounding box number.
    """
    bbox_number = check_bboxes(bbox, data_dir)
    if not bbox_number:
        return next_directory("bbox", data_dir)
    return bbox_number


def get_overrides(klass: type, overrides: dict[str, Path]) -> dict[str, Path]:
    """Get overrides for a class."""
    out = {}
    for p in overrides.copy():
        if not hasattr(klass, p):
            continue
        out[p] = overrides.pop(p)
    return out


class ProjectPaths:
    """Paths for the project folder (within base_dir)."""

    def __init__(self, base_dir: Path, project_name: str, **kwargs: Path):
        """Initialize the project paths.

        Args:
            base_dir (Path): The base directory.
            project_name (str): The name of the project.
            **kwargs: Additional file paths to override.
        """
        self.project_name = project_name
        self.base_dir = Path(base_dir)
        self.overrides: dict[str, Path] = get_overrides(ProjectPaths, kwargs)

        self.project.mkdir(exist_ok=True, parents=True)
        self.national.mkdir(exist_ok=True, parents=True)

    @property
    def project(self) -> Path:
        """The project folder (sits in the base_dir)."""
        return self.overrides.get("project", self.base_dir / self.project_name)

    @property
    def national(self) -> Path:
        """The national folder (for national scale downloads)."""
        return self.overrides.get("national", self.project / "national")

    @property
    def national_rail(self) -> Path:
        """The national scale rail file."""
        return self.overrides.get("national_rail", self.national / "rail.parquet")


class BBoxPaths:
    """Paths for the bounding box folder (within project folder)."""

    def __init__(
        self,
        project_paths: ProjectPaths,
        bbox_bounds: tuple[float, float, float, float],
        bbox_number: int | None = None,
        **kwargs: Path,
    ):
        """Initialize the bounding box paths.

        Args:
            project_paths (ProjectPaths): The project paths.
            bbox_bounds (tuple[float, float, float, float]): Bounding box coordinates.
            bbox_number (int, optional): Bounding box number or auto-detected.
            **kwargs: Additional file paths to override.
        """
        self.bbox_bounds = bbox_bounds
        if not bbox_number:
            bbox_number = get_next_bbox_number(self.bbox_bounds, project_paths.project)
        self.base_dir = project_paths.project
        self.bbox_number = bbox_number
        self.overrides: dict[str, Path] = get_overrides(BBoxPaths, kwargs)

        self.bbox.mkdir(exist_ok=True, parents=True)
        self.download.mkdir(exist_ok=True, parents=True)

        bounding_box_info = {"bbox": self.bbox_bounds, "project": project_paths.project_name}

        bbox_info_file = self.bbox / "bounding_box_info.json"
        if not bbox_info_file.exists():
            bbox_info_file.write_text(json.dumps(bounding_box_info).decode())

    @property
    def bbox(self) -> Path:
        """The bounding box folder (specific to a bounding box)."""
        return self.overrides.get("bbox", self.base_dir / f"bbox_{self.bbox_number}")

    @property
    def download(self) -> Path:
        """The download folder (for bbox specific downloaded data)."""
        return self.overrides.get("download", self.bbox / "download")

    @property
    def river(self) -> Path:
        """The river data for the bounding box."""
        return self.overrides.get("river", self.download / "river.parquet")

    @property
    def street(self) -> Path:
        """The street network data for the bounding box."""
        return self.overrides.get("street", self.download / "street_network.parquet")

    @property
    def elevation(self) -> Path:
        """The elevation file for the bounding box."""
        return self.overrides.get("elevation", self.download / "elevation.tif")

    @property
    def lulc(self) -> Path:
        """The land use land cover file for the bounding box."""
        return self.overrides.get("lulc", self.download / "lulc.tif")

    @property
    def imperviousness(self) -> Path:
        """The fractional imperviousness file for the bounding box."""
        return self.overrides.get("imperviousness", self.download / "imperviousness.tif")

    @property
    def water_bodies(self) -> Path:
        """The water bodies file for the bounding box."""
        return self.overrides.get("water_bodies", self.download / "water_bodies.parquet")

    @property
    def extra_waterbody(self) -> Path | None:
        """The extra waterbody file for the bounding box (e.g. from local authority)."""
        fid = self.overrides.get("extra_waterbody")
        if fid:
            fid = Path(fid)
            if fid.exists():
                return fid
            msg = f"Extra waterbody file {fid} does not exist."
            raise FileNotFoundError(msg)
        return None

    @property
    def basins(self) -> Path:
        """The OSM basin/detention/retention data for the bounding box."""
        return self.overrides.get("basins", self.download / "basins.parquet")

    @property
    def tiger_rail(self) -> Path:
        """The tiger rail file for the bounding box (clipped from national scale)."""
        return self.overrides.get("tiger_rail", self.download / "rail.parquet")


class ModelPaths:
    """Paths for the model folder (within bbox folder)."""

    def __init__(
        self,
        bbox_paths: BBoxPaths,
        model_number: int | None = None,
        **kwargs: Path,
    ):
        """Initialize the model paths.

        Args:
            bbox_paths (BBoxPaths): The bounding box paths.
            model_number (int, optional): Model number or next available.
            **kwargs: Additional file paths to override.
        """
        if model_number is None:
            model_number = next_directory("model", bbox_paths.bbox)
        self.base_dir = bbox_paths.bbox
        self.model_number = model_number
        self.overrides: dict[str, Path] = get_overrides(ModelPaths, kwargs)

        self.model.mkdir(exist_ok=True, parents=True)

    @property
    def model(self) -> Path:
        """The model folder (one specific synthesized model)."""
        return self.overrides.get("model", self.base_dir / f"model_{self.model_number}")

    @property
    def inp(self) -> Path:
        """The synthesized SWMM input file for the model."""
        return self.overrides.get("inp", self.model / f"model_{self.model_number}.inp")

    @property
    def subcatchments(self) -> Path:
        """The subcatchments file for the model."""
        return self.overrides.get("subcatchments", self.model / "subcatchments.parquet")

    @property
    def pondsheds(self) -> Path:
        """The pondsheds file for the model.

        One polygon per pond storage node, representing the union of all
        subcatchments whose runoff drains to the pond's downstream junction
        through the SWMM pipe network.  Generated as a visualization aid;
        SWMM itself does not consume this file.
        """
        return self.overrides.get("pondsheds", self.model / "pondsheds.parquet")

    @property
    def graph(self) -> Path:
        """The graph file for the model."""
        return self.overrides.get("graph", self.model / "graph.json")

    @property
    def nodes(self) -> Path:
        """The nodes file for the model."""
        return self.overrides.get("nodes", self.model / "nodes.json")

    @property
    def edges(self) -> Path:
        """The edges file for the model."""
        return self.overrides.get("edges", self.model / "edges.json")

    @property
    def streetcover(self) -> Path:
        """The street cover file for the model."""
        return self.overrides.get("streetcover", self.model / "streetcover.parquet")

    @property
    def flow_direction(self) -> Path:
        """The flow direction raster for the model."""
        return self.overrides.get("flow_direction", self.model / "flow_direction.tif")

    @property
    def flow_accumulation(self) -> Path:
        """The D8 flow accumulation raster for the model.

        Produced by WhiteboxTools during subcatchment delineation.
        Cell values represent the number of upstream cells draining
        through each cell.
        """
        return self.overrides.get("flow_accumulation", self.model / "flow_accum.tif")

    @property
    def slope(self) -> Path:
        """The slope raster for the model."""
        return self.overrides.get("slope", self.model / "slope.tif")


class FilePaths:
    """File paths class (manager for project, bbox and model)."""

    def __init__(
        self,
        base_dir: Path,
        project_name: str,
        bbox_bounds: tuple[float, float, float, float],
        bbox_number: int | None = None,
        model_number: int | None = None,
        **kwargs: Path,
    ):
        """Initialize the file paths.

        Args:
            base_dir (Path): Base directory.
            project_name (str): Project name.
            bbox_bounds (tuple): Bounding box (minx, miny, maxx, maxy).
            bbox_number (int, optional): Override bbox number.
            model_number (int, optional): Override model number.
            **kwargs: Additional file paths (overrides).
        """
        # Validate overrides and convert to paths
        for p, value in kwargs.items():
            path = Path(value)
            if not path.exists():
                logger.warning(f"Override path for {p}, {path} does not yet exist.")
            kwargs[p] = path

        # Create project paths and apply overrides
        self.project_paths = ProjectPaths(base_dir, project_name, **kwargs)

        # Create bbox paths and apply overrides
        self.bbox_paths = BBoxPaths(self.project_paths, bbox_bounds, bbox_number, **kwargs)

        # Create model paths and apply overrides
        self.model_paths = ModelPaths(self.bbox_paths, model_number, **kwargs)

        self._overrides = kwargs
