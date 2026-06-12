"""Standalone SWMM Input File Generator.

This module generates SWMM input files (.inp) based on user-provided options,
with default values for any sections not specified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .swmm_defaults import (
    AquiferDefaults,
    BuildupDefaults,
    ConduitDefaults,
    CoordinateDefaults,
    EvaporationOptions,
    GroundwaterDefaults,
    InfiltrationCurveNumberDefaults,
    InfiltrationGreenAmptDefaults,
    InfiltrationHortonDefaults,
    InflowDefaults,
    JunctionDefaults,
    LanduseDefaults,
    LossDefaults,
    OrificeDefaults,
    OutfallDefaults,
    OutletDefaults,
    PollutantDefaults,
    PumpDefaults,
    RaingageDefaults,
    ReportOptions,
    StorageDefaults,
    SubareaDefaults,
    SubcatchmentDefaults,
    SWMMOptions,
    WashoffDefaults,
    WeirDefaults,
    XSectionDefaults,
)

if TYPE_CHECKING:
    from io import TextIOWrapper

    from .swmm_defaults import InfiltrationType


def _write_section_header(
    f: TextIOWrapper, section_name: str, columns: list[str] | None = None
) -> None:
    """Write a section header with optional column names."""
    f.write(f"\n[{section_name}]\n")
    if columns:
        # Format: ;;Name        Column1      Column2      Column3
        # First column gets special treatment (no padding before it)
        header_parts = [f";;{columns[0]}", *(f"{col}" for col in columns[1:])]
        f.write(" ".join(f"{part:<16}" for part in header_parts).rstrip() + "\n")


@dataclass
class SwmmInputGenerator:
    """Generator for SWMM input files (.inp).

    Allows users to specify any section data, with defaults provided for
    unspecified values.
    """

    title: str = "SWMM Model"
    options: SWMMOptions | None = None
    evaporation: EvaporationOptions | None = None
    report: ReportOptions | None = None

    # Precipitation
    raingages: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Subcatchments
    subcatchments: dict[str, dict[str, Any]] = field(default_factory=dict)
    subareas: dict[str, dict[str, Any]] = field(default_factory=dict)
    infiltration: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Nodes
    junctions: dict[str, dict[str, Any]] = field(default_factory=dict)
    outfalls: dict[str, dict[str, Any]] = field(default_factory=dict)
    storage: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Links
    conduits: dict[str, dict[str, Any]] = field(default_factory=dict)
    pumps: dict[str, dict[str, Any]] = field(default_factory=dict)
    orifices: dict[str, dict[str, Any]] = field(default_factory=dict)
    weirs: dict[str, dict[str, Any]] = field(default_factory=dict)
    outlets: dict[str, dict[str, Any]] = field(default_factory=dict)
    xsections: dict[str, dict[str, Any]] = field(default_factory=dict)
    losses: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Inflows
    inflows: dict[str, dict[str, Any]] = field(default_factory=dict)
    dwf: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Curves and time series
    curves: dict[str, dict[str, Any]] = field(default_factory=dict)
    timeseries: dict[str, Any] = field(default_factory=dict)
    patterns: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Coordinates
    coordinates: dict[str, dict[str, float]] = field(default_factory=dict)
    vertices: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    polygons: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    # Tags
    tags: dict[str, dict[str, str]] = field(default_factory=dict)

    # Controls
    controls: str = ""

    # Water quality
    pollutants: dict[str, dict[str, Any]] = field(default_factory=dict)
    landuses: dict[str, dict[str, Any]] = field(default_factory=dict)
    coverages: dict[str, dict[str, float]] = field(default_factory=dict)
    loadings: dict[str, dict[str, float]] = field(default_factory=dict)
    buildup: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    washoff: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    # Groundwater
    aquifers: dict[str, dict[str, Any]] = field(default_factory=dict)
    groundwater: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Map settings
    map_settings: dict[str, Any] = field(default_factory=dict)

    def _get_options(self) -> SWMMOptions:
        """Get options dataclass."""
        return self.options if self.options is not None else SWMMOptions()

    def _get_evaporation(self) -> EvaporationOptions:
        """Get evaporation options dataclass."""
        return self.evaporation if self.evaporation is not None else EvaporationOptions()

    def _get_report(self) -> ReportOptions:
        """Get report options dataclass."""
        return self.report if self.report is not None else ReportOptions()

    def _get_infiltration_type(self) -> InfiltrationType:
        """Get the infiltration type from options."""
        return self._get_options().infiltration

    def _get_infiltration_defaults(
        self,
    ) -> (
        InfiltrationHortonDefaults | InfiltrationGreenAmptDefaults | InfiltrationCurveNumberDefaults
    ):
        """Get default infiltration values based on type."""
        inf_type = self._get_infiltration_type()
        if inf_type in ("GREEN_AMPT", "MODIFIED_GREEN_AMPT"):
            return InfiltrationGreenAmptDefaults()
        if inf_type == "CURVE_NUMBER":
            return InfiltrationCurveNumberDefaults()
        return InfiltrationHortonDefaults()

    def _write_title(self, f: TextIOWrapper) -> None:
        """Write TITLE section."""
        f.write("[TITLE]\n")
        f.write(";;Project Title/Notes\n")
        f.write(f"{self.title}\n")

    def _write_options(self, f: TextIOWrapper) -> None:
        """Write OPTIONS section."""
        f.write("\n[OPTIONS]\n")
        f.write(";;Option             Value\n")
        f.write(self._get_options().to_inp_section() + "\n")

    def _write_evaporation(self, f: TextIOWrapper) -> None:
        """Write EVAPORATION section."""
        f.write("\n[EVAPORATION]\n")
        f.write(";;Data Source    Parameters\n")
        f.write(";;-------------- ----------------\n")
        f.write(self._get_evaporation().to_inp_section() + "\n")

    def _write_report(self, f: TextIOWrapper) -> None:
        """Write REPORT section."""
        f.write("\n[REPORT]\n")
        f.write(";;Reporting Options\n")
        f.write(self._get_report().to_inp_section() + "\n")

    def _write_raingages(self, f: TextIOWrapper) -> None:
        """Write RAINGAGES section."""
        if not self.raingages:
            return
        defaults = RaingageDefaults()
        _write_section_header(f, "RAINGAGES", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.raingages.items()
        )

    def _write_subcatchments(self, f: TextIOWrapper) -> None:
        """Write SUBCATCHMENTS section."""
        if not self.subcatchments:
            return
        defaults = SubcatchmentDefaults()
        _write_section_header(f, "SUBCATCHMENTS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.subcatchments.items()
        )

    def _write_subareas(self, f: TextIOWrapper) -> None:
        """Write SUBAREAS section."""
        if not self.subcatchments:
            return
        defaults = SubareaDefaults()
        _write_section_header(f, "SUBAREAS", defaults.columns())
        for name in self.subcatchments:
            props = self.subareas.get(name, {})
            f.write(defaults.to_inp_row(name, **props) + "\n")

    def _write_infiltration(self, f: TextIOWrapper) -> None:
        """Write INFILTRATION section."""
        if not self.subcatchments:
            return
        defaults = self._get_infiltration_defaults()
        _write_section_header(f, "INFILTRATION", defaults.columns())
        for name in self.subcatchments:
            props = self.infiltration.get(name, {})
            f.write(defaults.to_inp_row(name, **props) + "\n")

    def _write_junctions(self, f: TextIOWrapper) -> None:
        """Write JUNCTIONS section."""
        if not self.junctions:
            return
        defaults = JunctionDefaults()
        _write_section_header(f, "JUNCTIONS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.junctions.items()
        )

    def _write_outfalls(self, f: TextIOWrapper) -> None:
        """Write OUTFALLS section."""
        if not self.outfalls:
            return
        defaults = OutfallDefaults()
        _write_section_header(f, "OUTFALLS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.outfalls.items()
        )

    def _write_storage(self, f: TextIOWrapper) -> None:
        """Write STORAGE section."""
        if not self.storage:
            return
        defaults = StorageDefaults()
        _write_section_header(f, "STORAGE", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.storage.items()
        )

    def _write_conduits(self, f: TextIOWrapper) -> None:
        """Write CONDUITS section."""
        if not self.conduits:
            return
        defaults = ConduitDefaults()
        _write_section_header(f, "CONDUITS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.conduits.items()
        )

    def _write_pumps(self, f: TextIOWrapper) -> None:
        """Write PUMPS section."""
        if not self.pumps:
            return
        defaults = PumpDefaults()
        _write_section_header(f, "PUMPS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.pumps.items()
        )

    def _write_orifices(self, f: TextIOWrapper) -> None:
        """Write ORIFICES section."""
        if not self.orifices:
            return
        defaults = OrificeDefaults()
        _write_section_header(f, "ORIFICES", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.orifices.items()
        )

    def _write_weirs(self, f: TextIOWrapper) -> None:
        """Write WEIRS section."""
        if not self.weirs:
            return
        defaults = WeirDefaults()
        _write_section_header(f, "WEIRS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.weirs.items()
        )

    def _write_outlets(self, f: TextIOWrapper) -> None:
        """Write OUTLETS section."""
        if not self.outlets:
            return
        defaults = OutletDefaults()
        _write_section_header(f, "OUTLETS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.outlets.items()
        )

    def _write_xsections(self, f: TextIOWrapper) -> None:
        """Write XSECTIONS section."""
        all_links = set(self.conduits.keys()) | set(self.weirs.keys()) | set(self.orifices.keys())
        if not all_links and not self.xsections:
            return
        defaults = XSectionDefaults()
        _write_section_header(f, "XSECTIONS", defaults.columns())
        for link in all_links:
            props = self.xsections.get(link, {})
            f.write(defaults.to_inp_row(link, **props) + "\n")

    def _write_losses(self, f: TextIOWrapper) -> None:
        """Write LOSSES section."""
        if not self.losses:
            return
        defaults = LossDefaults()
        _write_section_header(f, "LOSSES", defaults.columns())
        f.writelines(
            defaults.to_inp_row(link, **props) + "\n" for link, props in self.losses.items()
        )

    def _write_inflows(self, f: TextIOWrapper) -> None:
        """Write INFLOWS section."""
        if not self.inflows:
            return
        defaults = InflowDefaults()
        _write_section_header(f, "INFLOWS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(node, **props) + "\n" for node, props in self.inflows.items()
        )

    def _write_dwf(self, f: TextIOWrapper) -> None:
        """Write DWF section."""
        if not self.dwf:
            return
        f.write("\n[DWF]\n")
        f.write(";;Node           Constituent      Baseline   Patterns\n")
        for node, props in self.dwf.items():
            constituent = props.get("Constituent", "FLOW")
            baseline = props.get("Baseline", 0)
            patterns = props.get("Patterns", "")
            f.write(f"{node:<16} {constituent:<16} {baseline:<10} {patterns}".rstrip() + "\n")

    def _write_curves(self, f: TextIOWrapper) -> None:
        """Write CURVES section."""
        if not self.curves:
            return
        f.write("\n[CURVES]\n")
        f.write(";;Name           Type       X-Value    Y-Value\n")
        for name, props in self.curves.items():
            curve_type = props.get("Type", "")
            x_values = props.get("X-Values", props.get("x_values", []))
            y_values = props.get("Y-Values", props.get("y_values", []))
            for i, (x, y) in enumerate(zip(x_values, y_values)):
                if i == 0:
                    f.write(f"{name:<16} {curve_type:<10} {x:<10} {y}\n")
                else:
                    f.write(f"{name:<16} {'':<10} {x:<10} {y}\n")

    def _write_timeseries(self, f: TextIOWrapper) -> None:
        """Write TIMESERIES section."""
        if not self.timeseries:
            return
        f.write("\n[TIMESERIES]\n")
        f.write(";;Name           Date       Time       Value\n")
        for name, data in self.timeseries.items():
            if isinstance(data, dict):
                # Check if this is a FILE reference
                if "File" in data or "file" in data:
                    filename = data.get("File", data.get("file", ""))
                    f.write(f'{name:<16} FILE "{filename}"\n')
                    continue

                times = data.get("Times", data.get("times", []))
                values = data.get("Values", data.get("values", []))
                f.writelines(f"{name:<20} {t:<10} {v}\n" for t, v in zip(times, values))
            elif isinstance(data, list):
                for item in data:
                    if len(item) == 2:
                        t, v = item
                        f.write(f"{name:<20} {t:<10} {v}\n")
                    else:
                        d, t, v = item
                        f.write(f"{name:<20} {d:<10} {t:<10} {v}\n")

    def _write_patterns(self, f: TextIOWrapper) -> None:
        """Write PATTERNS section."""
        if not self.patterns:
            return
        f.write("\n[PATTERNS]\n")
        f.write(";;Name           Type       Multipliers\n")
        for name, props in self.patterns.items():
            pattern_type = props.get("Type", "HOURLY")
            multipliers: list[float] = props.get("Multipliers", [])
            if multipliers:
                mult_str = "  ".join(str(m) for m in multipliers[:6])
                f.write(f"{name:<16} {pattern_type:<10} {mult_str}".rstrip() + "\n")
                for i in range(6, len(multipliers), 6):
                    mult_str = "  ".join(str(m) for m in multipliers[i : i + 6])
                    f.write(f"{name:<16} {'':<10} {mult_str}".rstrip() + "\n")

    def _write_coordinates(self, f: TextIOWrapper) -> None:
        """Write COORDINATES section."""
        if not self.coordinates:
            return
        defaults = CoordinateDefaults()
        _write_section_header(f, "COORDINATES", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **coords) + "\n" for name, coords in self.coordinates.items()
        )

    def _write_vertices(self, f: TextIOWrapper) -> None:
        """Write VERTICES section."""
        if not self.vertices:
            return
        defaults = CoordinateDefaults()
        _write_section_header(f, "VERTICES", defaults.columns())
        for link, vertex_list in self.vertices.items():
            f.writelines(defaults.to_inp_row(link, X=x, Y=y) + "\n" for x, y in vertex_list)

    def _write_polygons(self, f: TextIOWrapper) -> None:
        """Write Polygons section."""
        if not self.polygons:
            return
        defaults = CoordinateDefaults()
        _write_section_header(f, "Polygons", defaults.columns())
        for name, vertex_list in self.polygons.items():
            f.writelines(defaults.to_inp_row(name, X=x, Y=y) + "\n" for x, y in vertex_list)

    def _write_tags(self, f: TextIOWrapper) -> None:
        """Write TAGS section."""
        if not self.tags:
            return
        f.write("\n[TAGS]\n")
        for name, props in self.tags.items():
            elem_type = props.get("ElementType", "Node")
            tag = props.get("Tag", "")
            f.write(f"{elem_type:<10} {name:<16} {tag}".rstrip() + "\n")

    def _write_controls(self, f: TextIOWrapper) -> None:
        """Write CONTROLS section."""
        if not self.controls:
            return
        f.write("\n[CONTROLS]\n")
        f.write(self.controls)
        if not self.controls.endswith("\n"):
            f.write("\n")

    def _write_pollutants(self, f: TextIOWrapper) -> None:
        """Write POLLUTANTS section."""
        if not self.pollutants:
            return
        defaults = PollutantDefaults()
        _write_section_header(f, "POLLUTANTS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.pollutants.items()
        )

    def _write_landuses(self, f: TextIOWrapper) -> None:
        """Write LANDUSES section."""
        if not self.landuses:
            return
        defaults = LanduseDefaults()
        _write_section_header(f, "LANDUSES", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.landuses.items()
        )

    def _write_coverages(self, f: TextIOWrapper) -> None:
        """Write COVERAGES section."""
        if not self.coverages:
            return
        f.write("\n[COVERAGES]\n")
        f.write(";;Subcatchment   Land Use         Percent\n")
        for subcatch, landuse_dict in self.coverages.items():
            f.writelines(
                f"{subcatch:<16} {landuse:<16} {percent}\n"
                for landuse, percent in landuse_dict.items()
            )

    def _write_loadings(self, f: TextIOWrapper) -> None:
        """Write LOADINGS section."""
        if not self.loadings:
            return
        f.write("\n[LOADINGS]\n")
        f.write(";;Subcatchment   Pollutant        Initial Buildup\n")
        for subcatch, pollutant_dict in self.loadings.items():
            f.writelines(
                f"{subcatch:<16} {pollutant:<16} {loading}\n"
                for pollutant, loading in pollutant_dict.items()
            )

    def _write_buildup(self, f: TextIOWrapper) -> None:
        """Write BUILDUP section."""
        if not self.buildup:
            return
        defaults = BuildupDefaults()
        _write_section_header(f, "BUILDUP", defaults.columns())
        for landuse, pollutant_dict in self.buildup.items():
            f.writelines(
                defaults.to_inp_row(landuse, pollutant, **props) + "\n"
                for pollutant, props in pollutant_dict.items()
            )

    def _write_washoff(self, f: TextIOWrapper) -> None:
        """Write WASHOFF section."""
        if not self.washoff:
            return
        defaults = WashoffDefaults()
        _write_section_header(f, "WASHOFF", defaults.columns())
        for landuse, pollutant_dict in self.washoff.items():
            f.writelines(
                defaults.to_inp_row(landuse, pollutant, **props) + "\n"
                for pollutant, props in pollutant_dict.items()
            )

    def _write_aquifers(self, f: TextIOWrapper) -> None:
        """Write AQUIFERS section."""
        if not self.aquifers:
            return
        defaults = AquiferDefaults()
        _write_section_header(f, "AQUIFERS", defaults.columns())
        f.writelines(
            defaults.to_inp_row(name, **props) + "\n" for name, props in self.aquifers.items()
        )

    def _write_groundwater(self, f: TextIOWrapper) -> None:
        """Write GROUNDWATER section."""
        if not self.groundwater:
            return
        defaults = GroundwaterDefaults()
        _write_section_header(f, "GROUNDWATER", defaults.columns())
        f.writelines(
            defaults.to_inp_row(subcatch, **props) + "\n"
            for subcatch, props in self.groundwater.items()
        )

    def _write_map(self, f: TextIOWrapper) -> None:
        """Write MAP section."""
        if not self.map_settings:
            return
        f.write("\n[MAP]\n")
        if "DIMENSIONS" in self.map_settings:
            dims = self.map_settings["DIMENSIONS"]
            f.write(f"DIMENSIONS {dims[0]:.3f} {dims[1]:.3f} {dims[2]:.3f} {dims[3]:.3f}\n")
        if "Units" in self.map_settings:
            f.write(f"Units      {self.map_settings['Units']}\n")

    def generate(self, filepath: str | Path) -> Path:
        """Generate a SWMM input file.

        Parameters
        ----------
        filepath : str | Path
            Path to the output .inp file

        Returns:
        -------
        Path
            Path to the generated file
        """
        filepath = Path(filepath)

        with filepath.open("w") as f:
            self._write_title(f)
            self._write_options(f)
            self._write_evaporation(f)
            self._write_report(f)
            self._write_raingages(f)
            self._write_subcatchments(f)
            self._write_subareas(f)
            self._write_infiltration(f)
            self._write_junctions(f)
            self._write_outfalls(f)
            self._write_storage(f)
            self._write_conduits(f)
            self._write_pumps(f)
            self._write_orifices(f)
            self._write_weirs(f)
            self._write_outlets(f)
            self._write_xsections(f)
            self._write_losses(f)
            self._write_inflows(f)
            self._write_dwf(f)
            self._write_curves(f)
            self._write_timeseries(f)
            self._write_patterns(f)
            self._write_pollutants(f)
            self._write_landuses(f)
            self._write_coverages(f)
            self._write_loadings(f)
            self._write_buildup(f)
            self._write_washoff(f)
            self._write_aquifers(f)
            self._write_groundwater(f)
            self._write_controls(f)
            self._write_tags(f)
            self._write_map(f)
            self._write_coordinates(f)
            self._write_vertices(f)
            self._write_polygons(f)

        return filepath
