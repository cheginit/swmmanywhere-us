"""Default values for SWMM input file elements.

This module contains dataclasses with default values for all SWMM elements
including nodes, links, subcatchments, and other model components.

Each dataclass provides:
- Default values for all fields
- columns(): Class method returning column names for the INP section header
- to_inp_row(name, **overrides): Method to generate a formatted INP file row
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    import pandas as pd

# Type aliases for SWMM options
FlowUnits = Literal["CFS", "GPM", "MGD", "CMS", "LPS", "MLD"]
InfiltrationType = Literal[
    "HORTON", "MODIFIED_HORTON", "GREEN_AMPT", "MODIFIED_GREEN_AMPT", "CURVE_NUMBER"
]
FlowRouting = Literal["STEADY", "KINWAVE", "DYNWAVE"]
LinkOffsets = Literal["DEPTH", "ELEVATION"]
InertialDamping = Literal["NONE", "PARTIAL", "FULL"]
NormalFlowLimited = Literal["SLOPE", "FROUDE", "BOTH"]
ForceMainEquation = Literal["H-W", "D-W"]
SurchargeMethod = Literal["EXTRAN", "SLOT"]
YesNo = Literal["YES", "NO"]

# Element type literals
OutfallType = Literal["FREE", "NORMAL", "FIXED", "TIDAL", "TIMESERIES"]
StorageCurveType = Literal["FUNCTIONAL", "TABULAR"]
OrificeType = Literal["SIDE", "BOTTOM"]
WeirType = Literal["TRANSVERSE", "SIDEFLOW", "V-NOTCH", "TRAPEZOIDAL", "ROADWAY"]
PumpStatus = Literal["ON", "OFF"]
XSectionShape = Literal[
    "CIRCULAR",
    "FORCE_MAIN",
    "FILLED_CIRCULAR",
    "RECT_CLOSED",
    "RECT_OPEN",
    "TRAPEZOIDAL",
    "TRIANGULAR",
    "HORIZ_ELLIPSE",
    "VERT_ELLIPSE",
    "ARCH",
    "PARABOLIC",
    "POWER",
    "RECT_TRIANGULAR",
    "RECT_ROUND",
    "MODBASKETHANDLE",
    "EGG",
    "HORSESHOE",
    "GOTHIC",
    "CATENARY",
    "SEMIELLIPTICAL",
    "BASKETHANDLE",
    "SEMICIRCULAR",
    "IRREGULAR",
    "CUSTOM",
]
RainFormat = Literal["INTENSITY", "VOLUME", "CUMULATIVE"]
RainSource = Literal["TIMESERIES", "FILE"]
RainUnits = Literal["IN", "MM"]
SubareaRouteTo = Literal["IMPERVIOUS", "PERVIOUS", "OUTLET"]
BuildupFunction = Literal["NONE", "POW", "EXP", "SAT"]
WashoffFunction = Literal["NONE", "EXP", "RC", "EMC"]
NormalizerType = Literal["AREA", "CURBLENGTH"]


def _format_value(value: Any) -> str:
    """Format a value for INP file output."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class SWMMOptions(BaseModel):
    """SWMM simulation options with defaults.

    All parameters have sensible defaults and can be overridden as needed.
    Dates may be unset at construction time; ``from_rain_data`` fills them
    from the rain timeseries before INP generation.
    """

    model_config = ConfigDict(extra="forbid")

    flow_units: FlowUnits = "LPS"
    infiltration: InfiltrationType = "HORTON"
    flow_routing: FlowRouting = "DYNWAVE"
    link_offsets: LinkOffsets = "DEPTH"
    # MIN_SLOPE enforces a minimum conduit slope; SWMM overrides computed
    # slopes below this threshold.  A non-zero value stabilizes flat-terrain
    # routing by preventing numerical issues with near-horizontal pipes.
    # Matches the value used by the site's calibrated reference model.
    min_slope: float = 0.001
    allow_ponding: bool = False
    skip_steady_state: bool = False
    start_date: str | None = None
    start_time: str | None = None
    report_start_date: str | None = None
    report_start_time: str | None = None
    end_date: str | None = None
    end_time: str | None = None
    sweep_start: str = "01/01"
    sweep_end: str = "12/31"
    dry_days: int = 0
    report_step: str = "00:15:00"
    wet_step: str = "00:05:00"
    dry_step: str = "01:00:00"
    routing_step: str = "00:00:05"
    rule_step: str | None = "00:00:00"
    inertial_damping: InertialDamping = "PARTIAL"
    normal_flow_limited: NormalFlowLimited = "BOTH"
    force_main_equation: ForceMainEquation = "H-W"
    variable_step: float = 0.75
    # SWMM short-conduit lengthening for stability: any conduit shorter
    # than v · lengthening_step is internally lengthened.  Tested on
    # the test catchment at 5 s and 30 s, both let the time step grow but
    # DYNWAVE then can't iterate enough per step, producing 45-87 %
    # non-convergence and -10 % routing continuity.  Keeping at 0
    # (SWMM default) trades runtime for accuracy: ~12 min on the test catchment with
    # -3.8 % routing continuity, vs 1.7 min with -10 % continuity.
    # Accuracy wins for a generator that produces a single design
    # model rather than running batches.
    lengthening_step: float = 0.0
    # SWMM-recommended dummy surface area at junctions (SWMM manual Sec.
    # A.5), stabilizes DYNWAVE at tiny/shallow manholes that would
    # otherwise see rapid level changes and fail iteration convergence.
    # 1.167 m^2 = 12.57 sq ft, the EPA SWMM reference value.
    min_surfarea: float = 1.167
    # Bump iteration cap from 8 to 20 to reduce "max trials exceeded"
    # continuity errors at junctions with parallel pond orifice+weir links
    # and street-channel overlays.
    max_trials: int = 20
    head_tolerance: float = 0.0
    sys_flow_tol: float = 5.0
    lat_flow_tol: float = 5.0
    minimum_step: float = 0.5
    threads: int = 1
    # SLOT = Preissmann slot method for pressurized pipe flow, far more
    # stable than the default EXTRAN surcharge solver when pipes surcharge
    # (which is the whole point of dual-drainage overlays).  Dobson et al.
    # (2025) also flagged EXTRAN instability in flat-terrain SWMManywhere
    # models.
    surcharge_method: SurchargeMethod | None = "SLOT"

    _field_to_swmm: ClassVar[dict[str, str]] = {
        "flow_units": "FLOW_UNITS",
        "infiltration": "INFILTRATION",
        "flow_routing": "FLOW_ROUTING",
        "link_offsets": "LINK_OFFSETS",
        "min_slope": "MIN_SLOPE",
        "allow_ponding": "ALLOW_PONDING",
        "skip_steady_state": "SKIP_STEADY_STATE",
        "start_date": "START_DATE",
        "start_time": "START_TIME",
        "report_start_date": "REPORT_START_DATE",
        "report_start_time": "REPORT_START_TIME",
        "end_date": "END_DATE",
        "end_time": "END_TIME",
        "sweep_start": "SWEEP_START",
        "sweep_end": "SWEEP_END",
        "dry_days": "DRY_DAYS",
        "report_step": "REPORT_STEP",
        "wet_step": "WET_STEP",
        "dry_step": "DRY_STEP",
        "routing_step": "ROUTING_STEP",
        "rule_step": "RULE_STEP",
        "inertial_damping": "INERTIAL_DAMPING",
        "normal_flow_limited": "NORMAL_FLOW_LIMITED",
        "force_main_equation": "FORCE_MAIN_EQUATION",
        "variable_step": "VARIABLE_STEP",
        "lengthening_step": "LENGTHENING_STEP",
        "min_surfarea": "MIN_SURFAREA",
        "max_trials": "MAX_TRIALS",
        "head_tolerance": "HEAD_TOLERANCE",
        "sys_flow_tol": "SYS_FLOW_TOL",
        "lat_flow_tol": "LAT_FLOW_TOL",
        "minimum_step": "MINIMUM_STEP",
        "threads": "THREADS",
        "surcharge_method": "SURCHARGE_METHOD",
    }

    @model_validator(mode="after")
    def _apply_time_defaults(self) -> Self:
        if self.start_date is not None and self.start_time is None:
            self.start_time = "00:00:00"
        if self.end_date is not None and self.end_time is None:
            self.end_time = "23:59:00"
        if self.report_start_date is None:
            self.report_start_date = self.start_date
        if self.report_start_time is None:
            self.report_start_time = self.start_time
        return self

    @classmethod
    def from_rain_data(
        cls,
        rain_df: pd.DataFrame,
        *,
        sim_tail_hours: float = 0.0,
        validate_overlap: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Create SWMMOptions with the simulation window derived from rain data.

        START/END default to the rain's first/last timestamp; any ``start_date``/
        ``end_date`` in ``kwargs`` overrides (filled only when ``None``).  The
        end is extended by ``sim_tail_hours`` so a post-storm recession (e.g.
        pond drawdown) is captured instead of truncated at the last rain step.

        Args:
            rain_df: DataFrame with a 'date' column.
            sim_tail_hours: Hours to extend END past the last rain timestamp.
            validate_overlap: If True, raise when an explicit start/end window is
                supplied that does not overlap the rain (avoids silent zero-rain
                runs for a user-provided .dat).  Keyword-only; never travels
                through ``inp_options`` (SWMMOptions forbids extra keys).
            **kwargs: Override any default option values.
        """
        import pandas as pd

        start_dt = rain_df["date"].min()
        end_dt = rain_df["date"].max() + pd.Timedelta(hours=sim_tail_hours)

        if validate_overlap and (kwargs.get("start_date") or kwargs.get("end_date")):
            win_start = (
                pd.to_datetime(f"{kwargs['start_date']} {kwargs.get('start_time') or '00:00:00'}")
                if kwargs.get("start_date")
                else start_dt
            )
            win_end = (
                pd.to_datetime(f"{kwargs['end_date']} {kwargs.get('end_time') or '23:59:00'}")
                if kwargs.get("end_date")
                else end_dt
            )
            rain_lo, rain_hi = rain_df["date"].min(), rain_df["date"].max()
            if rain_hi < win_start or rain_lo > win_end:
                msg = (
                    f"Rain data spans [{rain_lo}, {rain_hi}] but the simulation window is "
                    f"[{win_start}, {win_end}] -- they do not overlap, so the run would see zero "
                    f"rainfall. Align swmm_settings.inp_options start/end dates with the rain .dat, "
                    f"or drop the date overrides to derive the window from the rain."
                )
                raise ValueError(msg)

        defaults = {
            "start_date": start_dt.strftime("%m/%d/%Y"),
            "start_time": start_dt.strftime("%H:%M:%S"),
            "end_date": end_dt.strftime("%m/%d/%Y"),
            "end_time": end_dt.strftime("%H:%M:%S"),
        }
        for k, v in defaults.items():
            if kwargs.get(k) is None:
                kwargs[k] = v
        return cls(**kwargs)

    def to_inp_section(self) -> str:
        """Generate OPTIONS section content."""
        if self.start_date is None or self.end_date is None:
            msg = "Start and end dates must be set before rendering INP section."
            raise ValueError(msg)
        lines = []
        data = self.model_dump()
        for py_name, swmm_name in self._field_to_swmm.items():
            value = data.get(py_name)
            if value is not None:
                lines.append(f"{swmm_name:<20} {_format_value(value)}")
        return "\n".join(lines)


@dataclass
class EvaporationOptions:
    """SWMM evaporation options."""

    constant: float = 0.0
    dry_only: YesNo = "NO"

    def to_inp_section(self) -> str:
        """Generate EVAPORATION section content."""
        return f"CONSTANT             {self.constant}\nDRY_ONLY             {self.dry_only}"


@dataclass
class ReportOptions:
    """SWMM report options."""

    input_report: YesNo = "YES"
    controls: YesNo = "YES"
    # Accepts "ALL", "NONE", or a space-separated list of element IDs.
    subcatchments: str = "ALL"
    nodes: str = "ALL"
    links: str = "ALL"

    def to_inp_section(self) -> str:
        """Generate REPORT section content."""
        lines = [
            f"INPUT                {self.input_report}",
            f"CONTROLS             {self.controls}",
            f"SUBCATCHMENTS        {self.subcatchments}",
            f"NODES                {self.nodes}",
            f"LINKS                {self.links}",
        ]
        return "\n".join(lines)


# --- Node Defaults ---


@dataclass
class JunctionDefaults:
    """Default values for junction nodes."""

    invert_elev: float = 0.0
    max_depth: float = 0.0
    init_depth: float = 0.0
    surcharge_depth: float = 0.0
    ponded_area: float = 0.0

    _columns: ClassVar[list[str]] = [
        "Name",
        "InvertElev",
        "MaxDepth",
        "InitDepth",
        "SurchargeDepth",
        "PondedArea",
    ]
    _widths: ClassVar[list[int]] = [16, 12, 12, 12, 16, 12]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a junction."""
        data = {
            "InvertElev": overrides.get(
                "InvertElev", overrides.get("invert_elev", self.invert_elev)
            ),
            "MaxDepth": overrides.get("MaxDepth", overrides.get("max_depth", self.max_depth)),
            "InitDepth": overrides.get("InitDepth", overrides.get("init_depth", self.init_depth)),
            "SurchargeDepth": overrides.get(
                "SurchargeDepth", overrides.get("surcharge_depth", self.surcharge_depth)
            ),
            "PondedArea": overrides.get(
                "PondedArea", overrides.get("ponded_area", self.ponded_area)
            ),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['InvertElev']):<{w[1]}} "
            f"{_format_value(data['MaxDepth']):<{w[2]}} {_format_value(data['InitDepth']):<{w[3]}} "
            f"{_format_value(data['SurchargeDepth']):<{w[4]}} {_format_value(data['PondedArea'])}"
        )


@dataclass
class OutfallDefaults:
    """Default values for outfall nodes."""

    invert_elev: float = 0.0
    outfall_type: OutfallType = "FREE"
    stage_or_timeseries: str = ""
    tide_gate: YesNo = "NO"
    route_to: str = ""

    _columns: ClassVar[list[str]] = [
        "Name",
        "InvertElev",
        "OutfallType",
        "StageOrTimeseries",
        "TideGate",
        "RouteTo",
    ]
    _widths: ClassVar[list[int]] = [16, 10, 10, 16, 8, 16]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for an outfall."""
        data = {
            "InvertElev": overrides.get(
                "InvertElev", overrides.get("invert_elev", self.invert_elev)
            ),
            "OutfallType": overrides.get(
                "OutfallType", overrides.get("outfall_type", self.outfall_type)
            ),
            "StageOrTimeseries": overrides.get(
                "StageOrTimeseries", overrides.get("stage_or_timeseries", self.stage_or_timeseries)
            ),
            "TideGate": overrides.get("TideGate", overrides.get("tide_gate", self.tide_gate)),
            "RouteTo": overrides.get("RouteTo", overrides.get("route_to", self.route_to)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['InvertElev']):<{w[1]}} "
            f"{data['OutfallType']:<{w[2]}} {data['StageOrTimeseries']:<{w[3]}} "
            f"{data['TideGate']:<{w[4]}} {data['RouteTo']}"
        ).rstrip()


@dataclass
class StorageDefaults:
    """Default values for storage units.

    The field historically named ``ponded_area`` maps to SWMM's
    **SurDepth** column (additional depth above ``MaxDepth`` where water
    can pond on the surface before being lost to the flooding-loss
    term).  The optional trailing Green-Ampt columns (``psi``, ``ksat``,
    ``imd``) describe exfiltration through the pond bottom, used for
    closed-basin retention ponds that depend on percolation rather than
    gravity outflow to drain (common in flat coastal Florida).  When any
    of these are non-zero, the row is extended with their values.
    """

    invert_elev: float = 0.0
    max_depth: float = 10.0
    init_depth: float = 0.0
    storage_curve: StorageCurveType = "FUNCTIONAL"
    coefficient: float = 1000.0
    exponent: float = 0.0
    constant: float = 0.0
    ponded_area: float = 0.0  # SWMM SurDepth, above-MaxDepth ponding (not a planar area)
    evap_frac: float = 0.0
    psi: float = 0.0  # Green-Ampt suction head (mm or inches)
    ksat: float = 0.0  # Green-Ampt saturated conductivity (mm/hr or in/hr)
    imd: float = 0.0  # Green-Ampt initial moisture deficit (dimensionless)

    _columns: ClassVar[list[str]] = [
        "Name",
        "InvertElev",
        "MaxDepth",
        "InitDepth",
        "StorageCurve",
        "Coefficient",
        "Exponent",
        "Constant",
        "SurDepth",
        "EvapFrac",
        "Psi",
        "Ksat",
        "IMD",
    ]
    _widths: ClassVar[list[int]] = [16, 8, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 8]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a storage unit.

        SWMM uses different field layouts depending on the curve type:
        - FUNCTIONAL: Name Elev MaxD InitD FUNCTIONAL Coeff Exp Const SurDepth Fevap (Psi Ksat IMD)
        - TABULAR:    Name Elev MaxD InitD TABULAR CurveName SurDepth Fevap (Psi Ksat IMD)

        The Green-Ampt tail (Psi/Ksat/IMD) is omitted when all three are
        zero, so junctions that don't use exfiltration produce the same
        rows as before.
        """
        data = {
            "InvertElev": overrides.get(
                "InvertElev", overrides.get("invert_elev", self.invert_elev)
            ),
            "MaxDepth": overrides.get("MaxDepth", overrides.get("max_depth", self.max_depth)),
            "InitDepth": overrides.get("InitDepth", overrides.get("init_depth", self.init_depth)),
            "StorageCurve": overrides.get(
                "StorageCurve", overrides.get("storage_curve", self.storage_curve)
            ),
            "Coefficient": overrides.get(
                "Coefficient", overrides.get("coefficient", self.coefficient)
            ),
            "Exponent": overrides.get("Exponent", overrides.get("exponent", self.exponent)),
            "Constant": overrides.get("Constant", overrides.get("constant", self.constant)),
            "PondedArea": overrides.get(
                "PondedArea", overrides.get("ponded_area", self.ponded_area)
            ),
            "EvapFrac": overrides.get("EvapFrac", overrides.get("evap_frac", self.evap_frac)),
            "Psi": overrides.get("Psi", overrides.get("psi", self.psi)),
            "Ksat": overrides.get("Ksat", overrides.get("ksat", self.ksat)),
            "IMD": overrides.get("IMD", overrides.get("imd", self.imd)),
        }
        w = self._widths
        common = (
            f"{name:<{w[0]}} {_format_value(data['InvertElev']):<{w[1]}} "
            f"{_format_value(data['MaxDepth']):<{w[2]}} {_format_value(data['InitDepth']):<{w[3]}} "
        )
        seepage_tail = ""
        if float(data["Ksat"]) > 0:
            seepage_tail = (
                f" {_format_value(data['Psi']):<{w[10]}} "
                f"{_format_value(data['Ksat']):<{w[11]}} "
                f"{_format_value(data['IMD'])}"
            )
        if data["StorageCurve"] == "TABULAR":
            # TABULAR: CurveName replaces Coeff/Exp/Const
            return (
                f"{common}{data['StorageCurve']:<{w[4]}} {data['Coefficient']:<{w[5]}} "
                f"{_format_value(data['PondedArea']):<{w[8]}} "
                f"{_format_value(data['EvapFrac'])}{seepage_tail}"
            )
        return (
            f"{common}{data['StorageCurve']:<{w[4]}} {_format_value(data['Coefficient']):<{w[5]}} "
            f"{_format_value(data['Exponent']):<{w[6]}} {_format_value(data['Constant']):<{w[7]}} "
            f"{_format_value(data['PondedArea']):<{w[8]}} "
            f"{_format_value(data['EvapFrac'])}{seepage_tail}"
        )


# --- Link Defaults ---


@dataclass
class ConduitDefaults:
    """Default values for conduit links."""

    length: float = 0.0
    roughness: float = 0.01
    in_offset: float = 0.0001
    out_offset: float = 0.0001
    init_flow: float = 0.0
    max_flow: float = 10000000000.0

    _columns: ClassVar[list[str]] = [
        "Name",
        "InletNode",
        "OutletNode",
        "Length",
        "Roughness",
        "InOffset",
        "OutOffset",
        "InitFlow",
        "MaxFlow",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 16, 10, 10, 10, 10, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a conduit."""
        inlet = overrides.get("InletNode", overrides.get("inlet_node", ""))
        outlet = overrides.get("OutletNode", overrides.get("outlet_node", ""))
        data = {
            "Length": overrides.get("Length", overrides.get("length", self.length)),
            "Roughness": overrides.get("Roughness", overrides.get("roughness", self.roughness)),
            "InOffset": overrides.get("InOffset", overrides.get("in_offset", self.in_offset)),
            "OutOffset": overrides.get("OutOffset", overrides.get("out_offset", self.out_offset)),
            "InitFlow": overrides.get("InitFlow", overrides.get("init_flow", self.init_flow)),
            "MaxFlow": overrides.get("MaxFlow", overrides.get("max_flow", self.max_flow)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {inlet:<{w[1]}} {outlet:<{w[2]}} "
            f"{_format_value(data['Length']):<{w[3]}} {_format_value(data['Roughness']):<{w[4]}} "
            f"{_format_value(data['InOffset']):<{w[5]}} {_format_value(data['OutOffset']):<{w[6]}} "
            f"{_format_value(data['InitFlow']):<{w[7]}} {_format_value(data['MaxFlow'])}"
        )


@dataclass
class PumpDefaults:
    """Default values for pump links."""

    init_status: PumpStatus = "ON"
    depth: float = 0.0
    shutoff_depth: float = 0.0

    _columns: ClassVar[list[str]] = [
        "Name",
        "InletNode",
        "OutletNode",
        "PumpCurve",
        "InitStatus",
        "Depth",
        "ShutoffDepth",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 16, 16, 8, 8, 12]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a pump."""
        inlet = overrides.get("InletNode", overrides.get("inlet_node", ""))
        outlet = overrides.get("OutletNode", overrides.get("outlet_node", ""))
        curve = overrides.get("PumpCurve", overrides.get("pump_curve", ""))
        data = {
            "InitStatus": overrides.get(
                "InitStatus", overrides.get("init_status", self.init_status)
            ),
            "Depth": overrides.get("Depth", overrides.get("depth", self.depth)),
            "ShutoffDepth": overrides.get(
                "ShutoffDepth", overrides.get("shutoff_depth", self.shutoff_depth)
            ),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {inlet:<{w[1]}} {outlet:<{w[2]}} {curve:<{w[3]}} "
            f"{data['InitStatus']:<{w[4]}} {_format_value(data['Depth']):<{w[5]}} "
            f"{_format_value(data['ShutoffDepth'])}"
        )


@dataclass
class OrificeDefaults:
    """Default values for orifice links.

    ``flap_gate`` is "YES" by default: every pond orifice we emit is the
    primary controlled release of a detention basin, and standard FDOT /
    SJRWMD / ASCE MOP 77 practice is for these outlets to be one-way
    only (gate, check valve, or flapped culvert).  With the SWMM SIDE
    orifice + ``FlapGate=NO``, when the downstream junction surcharges
    above the pond's water level the orifice runs reversed and water
    flows from the network *back* into the pond, exactly the
    behavior observed on the test catchment when multiple ponds share a ds_node and
    one of them peaks (pond 1226's 57 ha pondshed surcharges junction
    196 to 1.8 m, reversing the 1226 and 1238 orifices for ~5 hours).
    ``FlapGate=YES`` forces SWMM to clip negative flow to zero,
    restoring detention-basin behavior.
    """

    orifice_type: OrificeType = "SIDE"
    crest_height: float = 0.0
    disch_coeff: float = 0.65
    flap_gate: YesNo = "YES"
    open_close_time: float = 0.0

    _columns: ClassVar[list[str]] = [
        "Name",
        "InletNode",
        "OutletNode",
        "OrificeType",
        "CrestHeight",
        "DischCoeff",
        "FlapGate",
        "OpenCloseTime",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 16, 12, 10, 10, 8, 12]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for an orifice."""
        inlet = overrides.get("InletNode", overrides.get("inlet_node", ""))
        outlet = overrides.get("OutletNode", overrides.get("outlet_node", ""))
        data = {
            "OrificeType": overrides.get(
                "OrificeType", overrides.get("orifice_type", self.orifice_type)
            ),
            "CrestHeight": overrides.get(
                "CrestHeight", overrides.get("crest_height", self.crest_height)
            ),
            "DischCoeff": overrides.get(
                "DischCoeff", overrides.get("disch_coeff", self.disch_coeff)
            ),
            "FlapGate": overrides.get("FlapGate", overrides.get("flap_gate", self.flap_gate)),
            "OpenCloseTime": overrides.get(
                "OpenCloseTime", overrides.get("open_close_time", self.open_close_time)
            ),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {inlet:<{w[1]}} {outlet:<{w[2]}} {data['OrificeType']:<{w[3]}} "
            f"{_format_value(data['CrestHeight']):<{w[4]}} {_format_value(data['DischCoeff']):<{w[5]}} "
            f"{data['FlapGate']:<{w[6]}} {_format_value(data['OpenCloseTime'])}"
        )


@dataclass
class WeirDefaults:
    """Default values for weir links.

    ``flap_gate`` defaults to "YES": every weir we emit is the
    emergency spillway on top of a detention pond's embankment crest.
    Water flowing over the crest is physically one-way (gravity), and
    SWMM with ``FlapGate=NO`` would allow reverse flow whenever the
    downstream junction surcharges above the weir crest, which is
    unphysical for an embankment weir.
    """

    weir_type: WeirType = "TRANSVERSE"
    crest_height: float = 0.0
    # SI sharp-crested transverse-weir coefficient (matches the default
    # metric/LPS flow units).  This is only a fallback: post_processing emits
    # an explicit flow-unit-converted ``disch_coeff`` for every pond weir, so
    # the value here is used only if a weir is written without one.
    disch_coeff: float = 1.84
    flap_gate: YesNo = "YES"
    end_con: float = 0.0
    end_coeff: float = 0.0
    # Surcharge = YES tells SWMM to switch the weir to the orifice flow
    # equation (Q = Cd*A*sqrt(2gH)) once it becomes fully submerged,
    # instead of capping discharge at the free-weir rate.  Without this
    # flag, a weir drowned by a parallel orifice oscillates across the
    # crest every routing step because DYNWAVE sees an artificial flow
    # plateau at the submerged weir.  Both calibrated reference models
    # (a calibrated Florida model and the UWO Swiss sewer) universally
    # set Surcharge = YES, and EPA SWMM Applications Manual §7.3
    # (Rossman 2017) identifies this as the standard fix for parallel
    # orifice+weir structures sharing the same node pair.
    surcharge: YesNo = "YES"

    _columns: ClassVar[list[str]] = [
        "Name",
        "InletNode",
        "OutletNode",
        "WeirType",
        "CrestHeight",
        "DischCoeff",
        "FlapGate",
        "EndCon",
        "EndCoeff",
        "Surcharge",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 16, 12, 10, 10, 8, 8, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a weir."""
        inlet = overrides.get("InletNode", overrides.get("inlet_node", ""))
        outlet = overrides.get("OutletNode", overrides.get("outlet_node", ""))
        data = {
            "WeirType": overrides.get("WeirType", overrides.get("weir_type", self.weir_type)),
            "CrestHeight": overrides.get(
                "CrestHeight", overrides.get("crest_height", self.crest_height)
            ),
            "DischCoeff": overrides.get(
                "DischCoeff", overrides.get("disch_coeff", self.disch_coeff)
            ),
            "FlapGate": overrides.get("FlapGate", overrides.get("flap_gate", self.flap_gate)),
            "EndCon": overrides.get("EndCon", overrides.get("end_con", self.end_con)),
            "EndCoeff": overrides.get("EndCoeff", overrides.get("end_coeff", self.end_coeff)),
            "Surcharge": overrides.get("Surcharge", overrides.get("surcharge", self.surcharge)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {inlet:<{w[1]}} {outlet:<{w[2]}} {data['WeirType']:<{w[3]}} "
            f"{_format_value(data['CrestHeight']):<{w[4]}} {_format_value(data['DischCoeff']):<{w[5]}} "
            f"{data['FlapGate']:<{w[6]}} {_format_value(data['EndCon']):<{w[7]}} "
            f"{_format_value(data['EndCoeff']):<{w[8]}} {data['Surcharge']}"
        )


@dataclass
class OutletDefaults:
    """Default values for outlet links."""

    outflow_height: float = 0.0
    outlet_type: str = "FUNCTIONAL/DEPTH"
    qcoeff: float = 0.0
    qexpon: float = 1.0
    flap_gate: YesNo = "NO"

    _columns: ClassVar[list[str]] = [
        "Name",
        "InletNode",
        "OutletNode",
        "OutflowHeight",
        "OutletType",
        "Qcoeff",
        "Qexpon",
        "FlapGate",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 16, 10, 20, 10, 8, 8]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for an outlet."""
        inlet = overrides.get("InletNode", overrides.get("inlet_node", ""))
        outlet = overrides.get("OutletNode", overrides.get("outlet_node", ""))
        data = {
            "OutflowHeight": overrides.get(
                "OutflowHeight", overrides.get("outflow_height", self.outflow_height)
            ),
            "OutletType": overrides.get(
                "OutletType", overrides.get("outlet_type", self.outlet_type)
            ),
            "Qcoeff": overrides.get("Qcoeff", overrides.get("qcoeff", self.qcoeff)),
            "Qexpon": overrides.get("Qexpon", overrides.get("qexpon", self.qexpon)),
            "FlapGate": overrides.get("FlapGate", overrides.get("flap_gate", self.flap_gate)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {inlet:<{w[1]}} {outlet:<{w[2]}} "
            f"{_format_value(data['OutflowHeight']):<{w[3]}} {data['OutletType']:<{w[4]}} "
            f"{_format_value(data['Qcoeff']):<{w[5]}} {_format_value(data['Qexpon']):<{w[6]}} "
            f"{data['FlapGate']}"
        )


@dataclass
class XSectionDefaults:
    """Default values for cross-sections."""

    shape: XSectionShape = "CIRCULAR"
    geom1: float = 1.0
    geom2: float = 0.0
    geom3: float = 0.0
    geom4: float = 0.0
    barrels: int = 1
    culvert: str = ""

    _columns: ClassVar[list[str]] = [
        "Link",
        "Shape",
        "Geom1",
        "Geom2",
        "Geom3",
        "Geom4",
        "Barrels",
        "Culvert",
    ]
    _widths: ClassVar[list[int]] = [16, 12, 16, 10, 10, 10, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a cross-section."""
        data = {
            "Shape": overrides.get("Shape", overrides.get("shape", self.shape)),
            "Geom1": overrides.get("Geom1", overrides.get("geom1", self.geom1)),
            "Geom2": overrides.get("Geom2", overrides.get("geom2", self.geom2)),
            "Geom3": overrides.get("Geom3", overrides.get("geom3", self.geom3)),
            "Geom4": overrides.get("Geom4", overrides.get("geom4", self.geom4)),
            "Barrels": overrides.get("Barrels", overrides.get("barrels", self.barrels)),
            "Culvert": overrides.get("Culvert", overrides.get("culvert", self.culvert)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {data['Shape']:<{w[1]}} {_format_value(data['Geom1']):<{w[2]}} "
            f"{_format_value(data['Geom2']):<{w[3]}} {_format_value(data['Geom3']):<{w[4]}} "
            f"{_format_value(data['Geom4']):<{w[5]}} {_format_value(data['Barrels']):<{w[6]}} "
            f"{data['Culvert']}"
        ).rstrip()


@dataclass
class LossDefaults:
    """Default values for link losses."""

    inlet: float = 0.0
    outlet: float = 0.0
    average: float = 0.0
    flap_gate: YesNo = "NO"
    seepage_rate: float = 0.0

    _columns: ClassVar[list[str]] = [
        "Link",
        "Inlet",
        "Outlet",
        "Average",
        "FlapGate",
        "SeepageRate",
    ]
    _widths: ClassVar[list[int]] = [16, 10, 10, 10, 10, 12]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for losses."""
        data = {
            "Inlet": overrides.get("Inlet", overrides.get("inlet", self.inlet)),
            "Outlet": overrides.get("Outlet", overrides.get("outlet", self.outlet)),
            "Average": overrides.get("Average", overrides.get("average", self.average)),
            "FlapGate": overrides.get("FlapGate", overrides.get("flap_gate", self.flap_gate)),
            "SeepageRate": overrides.get(
                "SeepageRate", overrides.get("seepage_rate", self.seepage_rate)
            ),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['Inlet']):<{w[1]}} {_format_value(data['Outlet']):<{w[2]}} "
            f"{_format_value(data['Average']):<{w[3]}} {data['FlapGate']:<{w[4]}} "
            f"{_format_value(data['SeepageRate'])}"
        )


# --- Subcatchment Defaults ---


@dataclass
class SubcatchmentDefaults:
    """Default values for subcatchments."""

    area: float = 0.0
    perc_imperv: float = 0.0
    width: float = 0.0
    perc_slope: float = 0.0
    curb_length: float = 0.0
    snow_pack: str = ""

    _columns: ClassVar[list[str]] = [
        "Name",
        "Raingage",
        "Outlet",
        "Area",
        "PercImperv",
        "Width",
        "PercSlope",
        "CurbLength",
        "SnowPack",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 16, 8, 8, 8, 8, 8, 16]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a subcatchment."""
        raingage = overrides.get("Raingage", overrides.get("raingage", ""))
        outlet = overrides.get("Outlet", overrides.get("outlet", ""))
        data = {
            "Area": overrides.get("Area", overrides.get("area", self.area)),
            "PercImperv": overrides.get(
                "PercImperv", overrides.get("perc_imperv", self.perc_imperv)
            ),
            "Width": overrides.get("Width", overrides.get("width", self.width)),
            "PercSlope": overrides.get("PercSlope", overrides.get("perc_slope", self.perc_slope)),
            "CurbLength": overrides.get(
                "CurbLength", overrides.get("curb_length", self.curb_length)
            ),
            "SnowPack": overrides.get("SnowPack", overrides.get("snow_pack", self.snow_pack)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {raingage:<{w[1]}} {outlet:<{w[2]}} "
            f"{_format_value(data['Area']):<{w[3]}} {_format_value(data['PercImperv']):<{w[4]}} "
            f"{_format_value(data['Width']):<{w[5]}} {_format_value(data['PercSlope']):<{w[6]}} "
            f"{_format_value(data['CurbLength']):<{w[7]}} {data['SnowPack']}"
        ).rstrip()


@dataclass
class SubareaDefaults:
    """Default values for subareas.

    Units depend on SWMM FLOW_UNITS setting:
    - US units: inches for depression storage, Manning's n dimensionless
    - SI units: mm for depression storage

    Defaults from SWMM 5 Reference Manual Vol. I, Table 4-2.
    """

    n_imperv: float = 0.015  # Manning's n for impervious (typical concrete/asphalt)
    n_perv: float = 0.15  # Manning's n for pervious (grass/turf)
    s_imperv: float = 0.05  # impervious depression storage (in)
    s_perv: float = 0.10  # pervious depression storage (in)
    pct_zero: float = 25.0  # percent of impervious with no depression storage
    route_to: SubareaRouteTo = "OUTLET"
    pct_routed: str = ""

    _columns: ClassVar[list[str]] = [
        "Name",
        "N-Imperv",
        "N-Perv",
        "S-Imperv",
        "S-Perv",
        "PctZero",
        "RouteTo",
        "PctRouted",
    ]
    _widths: ClassVar[list[int]] = [16, 10, 10, 10, 10, 10, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a subarea."""
        data = {
            "N-Imperv": overrides.get("N-Imperv", overrides.get("n_imperv", self.n_imperv)),
            "N-Perv": overrides.get("N-Perv", overrides.get("n_perv", self.n_perv)),
            "S-Imperv": overrides.get("S-Imperv", overrides.get("s_imperv", self.s_imperv)),
            "S-Perv": overrides.get("S-Perv", overrides.get("s_perv", self.s_perv)),
            "PctZero": overrides.get("PctZero", overrides.get("pct_zero", self.pct_zero)),
            "RouteTo": overrides.get("RouteTo", overrides.get("route_to", self.route_to)),
            "PctRouted": overrides.get("PctRouted", overrides.get("pct_routed", self.pct_routed)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['N-Imperv']):<{w[1]}} {_format_value(data['N-Perv']):<{w[2]}} "
            f"{_format_value(data['S-Imperv']):<{w[3]}} {_format_value(data['S-Perv']):<{w[4]}} "
            f"{_format_value(data['PctZero']):<{w[5]}} {data['RouteTo']:<{w[6]}} "
            f"{_format_value(data['PctRouted'])}"
        ).rstrip()


@dataclass
class InfiltrationHortonDefaults:
    """Default values for Horton infiltration.

    SWMM interprets units according to the FLOW_UNITS setting:
    - US units (CFS/GPM/MGD): in/hr for rates, inches for max_infil
    - SI units (CMS/LPS/MLD): mm/hr for rates, mm for max_infil

    Defaults represent typical sandy/silty soils with moderate infiltration.
    Based on SWMM 5 Reference Manual Vol. I, Table 4-9.
    """

    max_rate: float = 3.0  # initial infiltration rate (in/hr)
    min_rate: float = 0.5  # minimum steady-state rate (in/hr)
    decay: float = 4.0  # decay constant (1/hr)
    dry_time: float = 7.0  # time to fully dry out (days)
    max_infil: float = 0.0  # maximum infiltration volume (0 = unlimited)

    _columns: ClassVar[list[str]] = [
        "Subcatchment",
        "MaxRate",
        "MinRate",
        "Decay",
        "DryTime",
        "MaxInfil",
    ]
    _widths: ClassVar[list[int]] = [16, 10, 10, 10, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for Horton infiltration."""
        data = {
            "MaxRate": overrides.get("MaxRate", overrides.get("max_rate", self.max_rate)),
            "MinRate": overrides.get("MinRate", overrides.get("min_rate", self.min_rate)),
            "Decay": overrides.get("Decay", overrides.get("decay", self.decay)),
            "DryTime": overrides.get("DryTime", overrides.get("dry_time", self.dry_time)),
            "MaxInfil": overrides.get("MaxInfil", overrides.get("max_infil", self.max_infil)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['MaxRate']):<{w[1]}} {_format_value(data['MinRate']):<{w[2]}} "
            f"{_format_value(data['Decay']):<{w[3]}} {_format_value(data['DryTime']):<{w[4]}} "
            f"{_format_value(data['MaxInfil'])}"
        )


@dataclass
class InfiltrationGreenAmptDefaults:
    """Default values for Green-Ampt infiltration."""

    suction: float = 3.5
    hyd_con: float = 0.5
    imd_max: float = 0.25

    _columns: ClassVar[list[str]] = ["Subcatchment", "Suction", "HydCon", "IMDmax"]
    _widths: ClassVar[list[int]] = [16, 10, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for Green-Ampt infiltration."""
        data = {
            "Suction": overrides.get("Suction", overrides.get("suction", self.suction)),
            "HydCon": overrides.get("HydCon", overrides.get("hyd_con", self.hyd_con)),
            "IMDmax": overrides.get("IMDmax", overrides.get("imd_max", self.imd_max)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['Suction']):<{w[1]}} "
            f"{_format_value(data['HydCon']):<{w[2]}} {_format_value(data['IMDmax'])}"
        )


@dataclass
class InfiltrationCurveNumberDefaults:
    """Default values for Curve Number infiltration."""

    curve_num: float = 75.0
    hyd_con: float = 0.5
    dry_time: float = 7.0

    _columns: ClassVar[list[str]] = ["Subcatchment", "CurveNum", "HydCon", "DryTime"]
    _widths: ClassVar[list[int]] = [16, 10, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for Curve Number infiltration."""
        data = {
            "CurveNum": overrides.get("CurveNum", overrides.get("curve_num", self.curve_num)),
            "HydCon": overrides.get("HydCon", overrides.get("hyd_con", self.hyd_con)),
            "DryTime": overrides.get("DryTime", overrides.get("dry_time", self.dry_time)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['CurveNum']):<{w[1]}} "
            f"{_format_value(data['HydCon']):<{w[2]}} {_format_value(data['DryTime'])}"
        )


# --- Precipitation Defaults ---


@dataclass
class RaingageDefaults:
    """Default values for rain gages.

    Supports both FILE and TIMESERIES data sources:
    - FILE (default): Reads rainfall from external file
    - TIMESERIES: Uses time series defined in [TIMESERIES] section

    For FILE source, provide: filename, station_id, rain_units
    For TIMESERIES source, provide: timeseries (name of the time series)

    Example FILE usage:
        raingages={"RG1": {"filename": "rainfall.dat", "station_id": "12345", "rain_units": "IN"}}

    Example TIMESERIES usage:
        raingages={"RG1": {"source": "TIMESERIES", "timeseries": "TS1"}}
    """

    format: RainFormat = "VOLUME"
    interval: str = "1:00"
    scf: float = 1.0
    source: RainSource = "FILE"
    # FILE source fields
    filename: str = ""
    station_id: str = ""
    rain_units: RainUnits = "IN"
    # TIMESERIES source field
    timeseries: str = ""

    _columns: ClassVar[list[str]] = [
        "Name",
        "Format",
        "Interval",
        "SCF",
        "Source",
        "SourceName/File",
    ]
    _widths: ClassVar[list[int]] = [16, 9, 8, 8, 10, 20]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a rain gage.

        For FILE source, outputs: Name Format Interval SCF FILE "filename" station_id units
        For TIMESERIES source, outputs: Name Format Interval SCF TIMESERIES series_name
        """
        data = {
            "Format": overrides.get("Format", overrides.get("format", self.format)),
            "Interval": overrides.get("Interval", overrides.get("interval", self.interval)),
            "SCF": overrides.get("SCF", overrides.get("scf", self.scf)),
            "Source": overrides.get("Source", overrides.get("source", self.source)),
        }
        w = self._widths
        base = (
            f"{name:<{w[0]}} {data['Format']:<{w[1]}} {data['Interval']:<{w[2]}} "
            f"{_format_value(data['SCF']):<{w[3]}} {data['Source']:<{w[4]}}"
        )

        if data["Source"] == "FILE":
            filename = overrides.get("filename", overrides.get("Filename", self.filename))
            station_id = overrides.get("station_id", overrides.get("StationID", self.station_id))
            rain_units = overrides.get("rain_units", overrides.get("RainUnits", self.rain_units))
            # Quote the filename
            quoted_filename = f'"{filename}"' if not filename.startswith('"') else filename
            return f"{base} {quoted_filename}  {station_id} {rain_units}".rstrip()
        # TIMESERIES source
        ts_name = overrides.get(
            "timeseries",
            overrides.get("Timeseries", overrides.get("SourceName", self.timeseries or name)),
        )
        return f"{base} {ts_name}".rstrip()


# --- Inflow Defaults ---


@dataclass
class InflowDefaults:
    """Default values for external inflows."""

    constituent: str = "FLOW"
    time_series: str = '""'
    inflow_type: str = "FLOW"
    mfactor: float = 1.0
    sfactor: float = 1.0
    baseline: float = 0.0
    pattern: str = ""

    _columns: ClassVar[list[str]] = [
        "Node",
        "Constituent",
        "TimeSeries",
        "Type",
        "Mfactor",
        "Sfactor",
        "Baseline",
        "Pattern",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 16, 8, 8, 8, 8, 16]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for an inflow."""
        data = {
            "Constituent": overrides.get(
                "Constituent", overrides.get("constituent", self.constituent)
            ),
            "TimeSeries": overrides.get(
                "TimeSeries", overrides.get("time_series", self.time_series)
            ),
            "Type": overrides.get("Type", overrides.get("inflow_type", self.inflow_type)),
            "Mfactor": overrides.get("Mfactor", overrides.get("mfactor", self.mfactor)),
            "Sfactor": overrides.get("Sfactor", overrides.get("sfactor", self.sfactor)),
            "Baseline": overrides.get("Baseline", overrides.get("baseline", self.baseline)),
            "Pattern": overrides.get("Pattern", overrides.get("pattern", self.pattern)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {data['Constituent']:<{w[1]}} {data['TimeSeries']:<{w[2]}} "
            f"{data['Type']:<{w[3]}} {_format_value(data['Mfactor']):<{w[4]}} "
            f"{_format_value(data['Sfactor']):<{w[5]}} {_format_value(data['Baseline']):<{w[6]}} "
            f"{data['Pattern']}"
        ).rstrip()


# --- Water Quality Defaults ---


@dataclass
class PollutantDefaults:
    """Default values for pollutants."""

    mass_units: str = "MG/L"
    rain_concen: float = 0.0
    gw_concen: float = 0.0
    ii_concen: float = 0.0
    decay_coeff: float = 0.0
    snow_only: YesNo = "NO"
    co_pollutant: str = "*"
    co_fraction: float = 0.0
    dwf_concen: float = 0.0
    init_concen: float = 0.0

    _columns: ClassVar[list[str]] = [
        "Name",
        "MassUnits",
        "RainConcen",
        "GWConcen",
        "IIConcen",
        "DecayCoeff",
        "SnowOnly",
        "CoPollutant",
        "CoFraction",
        "DWFConcen",
        "InitConcen",
    ]
    _widths: ClassVar[list[int]] = [16, 8, 10, 10, 10, 10, 8, 12, 10, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a pollutant."""
        data = {
            "MassUnits": overrides.get("MassUnits", overrides.get("mass_units", self.mass_units)),
            "RainConcen": overrides.get(
                "RainConcen", overrides.get("rain_concen", self.rain_concen)
            ),
            "GWConcen": overrides.get("GWConcen", overrides.get("gw_concen", self.gw_concen)),
            "IIConcen": overrides.get("IIConcen", overrides.get("ii_concen", self.ii_concen)),
            "DecayCoeff": overrides.get(
                "DecayCoeff", overrides.get("decay_coeff", self.decay_coeff)
            ),
            "SnowOnly": overrides.get("SnowOnly", overrides.get("snow_only", self.snow_only)),
            "CoPollutant": overrides.get(
                "CoPollutant", overrides.get("co_pollutant", self.co_pollutant)
            ),
            "CoFraction": overrides.get(
                "CoFraction", overrides.get("co_fraction", self.co_fraction)
            ),
            "DWFConcen": overrides.get("DWFConcen", overrides.get("dwf_concen", self.dwf_concen)),
            "InitConcen": overrides.get(
                "InitConcen", overrides.get("init_concen", self.init_concen)
            ),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {data['MassUnits']:<{w[1]}} {_format_value(data['RainConcen']):<{w[2]}} "
            f"{_format_value(data['GWConcen']):<{w[3]}} {_format_value(data['IIConcen']):<{w[4]}} "
            f"{_format_value(data['DecayCoeff']):<{w[5]}} {data['SnowOnly']:<{w[6]}} "
            f"{data['CoPollutant']:<{w[7]}} {_format_value(data['CoFraction']):<{w[8]}} "
            f"{_format_value(data['DWFConcen']):<{w[9]}} {_format_value(data['InitConcen'])}"
        )


@dataclass
class LanduseDefaults:
    """Default values for land uses."""

    cleaning_interval: float = 0.0
    fraction_available: float = 0.0
    last_cleaned: float = 0.0

    _columns: ClassVar[list[str]] = ["Name", "CleaningInterval", "FractionAvailable", "LastCleaned"]
    _widths: ClassVar[list[int]] = [16, 16, 16, 12]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for a land use."""
        data = {
            "CleaningInterval": overrides.get(
                "CleaningInterval", overrides.get("cleaning_interval", self.cleaning_interval)
            ),
            "FractionAvailable": overrides.get(
                "FractionAvailable", overrides.get("fraction_available", self.fraction_available)
            ),
            "LastCleaned": overrides.get(
                "LastCleaned", overrides.get("last_cleaned", self.last_cleaned)
            ),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['CleaningInterval']):<{w[1]}} "
            f"{_format_value(data['FractionAvailable']):<{w[2]}} {_format_value(data['LastCleaned'])}"
        )


@dataclass
class BuildupDefaults:
    """Default values for pollutant buildup."""

    function: BuildupFunction = "NONE"
    coeff1: float = 0.0
    coeff2: float = 0.0
    coeff3: float = 0.0
    normalizer: NormalizerType = "AREA"

    _columns: ClassVar[list[str]] = [
        "LandUse",
        "Pollutant",
        "Function",
        "Coeff1",
        "Coeff2",
        "Coeff3",
        "Normalizer",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 10, 10, 10, 10, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, landuse: str, pollutant: str, **overrides: Any) -> str:
        """Generate INP file row for buildup."""
        data = {
            "Function": overrides.get("Function", overrides.get("function", self.function)),
            "Coeff1": overrides.get("Coeff1", overrides.get("coeff1", self.coeff1)),
            "Coeff2": overrides.get("Coeff2", overrides.get("coeff2", self.coeff2)),
            "Coeff3": overrides.get("Coeff3", overrides.get("coeff3", self.coeff3)),
            "Normalizer": overrides.get("Normalizer", overrides.get("normalizer", self.normalizer)),
        }
        w = self._widths
        return (
            f"{landuse:<{w[0]}} {pollutant:<{w[1]}} {data['Function']:<{w[2]}} "
            f"{_format_value(data['Coeff1']):<{w[3]}} {_format_value(data['Coeff2']):<{w[4]}} "
            f"{_format_value(data['Coeff3']):<{w[5]}} {data['Normalizer']}"
        )


@dataclass
class WashoffDefaults:
    """Default values for pollutant washoff."""

    function: WashoffFunction = "NONE"
    coeff1: float = 0.0
    coeff2: float = 0.0
    cleaning_effic: float = 0.0
    bmp_effic: float = 0.0

    _columns: ClassVar[list[str]] = [
        "LandUse",
        "Pollutant",
        "Function",
        "Coeff1",
        "Coeff2",
        "CleaningEffic",
        "BMPEffic",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 10, 10, 10, 12, 10]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, landuse: str, pollutant: str, **overrides: Any) -> str:
        """Generate INP file row for washoff."""
        data = {
            "Function": overrides.get("Function", overrides.get("function", self.function)),
            "Coeff1": overrides.get("Coeff1", overrides.get("coeff1", self.coeff1)),
            "Coeff2": overrides.get("Coeff2", overrides.get("coeff2", self.coeff2)),
            "CleaningEffic": overrides.get(
                "CleaningEffic", overrides.get("cleaning_effic", self.cleaning_effic)
            ),
            "BMPEffic": overrides.get("BMPEffic", overrides.get("bmp_effic", self.bmp_effic)),
        }
        w = self._widths
        return (
            f"{landuse:<{w[0]}} {pollutant:<{w[1]}} {data['Function']:<{w[2]}} "
            f"{_format_value(data['Coeff1']):<{w[3]}} {_format_value(data['Coeff2']):<{w[4]}} "
            f"{_format_value(data['CleaningEffic']):<{w[5]}} {_format_value(data['BMPEffic'])}"
        )


# --- Groundwater Defaults ---


@dataclass
class AquiferDefaults:
    """Default values for aquifers."""

    por: float = 0.5
    wp: float = 0.15
    fc: float = 0.30
    ksat: float = 0.1
    kslope: float = 5.0
    tslope: float = 15.0
    etu: float = 0.35
    ets: float = 14.0
    seep: float = 0.002
    ebot: float = 0.0
    egw: float = 10.0
    umc: float = 0.30
    etupat: str = ""

    _columns: ClassVar[list[str]] = [
        "Name",
        "Por",
        "WP",
        "FC",
        "Ksat",
        "Kslope",
        "Tslope",
        "ETu",
        "ETs",
        "Seep",
        "Ebot",
        "Egw",
        "Umc",
        "ETupat",
    ]
    _widths: ClassVar[list[int]] = [16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 12]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for an aquifer."""
        data = {
            "Por": overrides.get("Por", overrides.get("por", self.por)),
            "WP": overrides.get("WP", overrides.get("wp", self.wp)),
            "FC": overrides.get("FC", overrides.get("fc", self.fc)),
            "Ksat": overrides.get("Ksat", overrides.get("ksat", self.ksat)),
            "Kslope": overrides.get("Kslope", overrides.get("kslope", self.kslope)),
            "Tslope": overrides.get("Tslope", overrides.get("tslope", self.tslope)),
            "ETu": overrides.get("ETu", overrides.get("etu", self.etu)),
            "ETs": overrides.get("ETs", overrides.get("ets", self.ets)),
            "Seep": overrides.get("Seep", overrides.get("seep", self.seep)),
            "Ebot": overrides.get("Ebot", overrides.get("ebot", self.ebot)),
            "Egw": overrides.get("Egw", overrides.get("egw", self.egw)),
            "Umc": overrides.get("Umc", overrides.get("umc", self.umc)),
            "ETupat": overrides.get("ETupat", overrides.get("etupat", self.etupat)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {_format_value(data['Por']):<{w[1]}} {_format_value(data['WP']):<{w[2]}} "
            f"{_format_value(data['FC']):<{w[3]}} {_format_value(data['Ksat']):<{w[4]}} "
            f"{_format_value(data['Kslope']):<{w[5]}} {_format_value(data['Tslope']):<{w[6]}} "
            f"{_format_value(data['ETu']):<{w[7]}} {_format_value(data['ETs']):<{w[8]}} "
            f"{_format_value(data['Seep']):<{w[9]}} {_format_value(data['Ebot']):<{w[10]}} "
            f"{_format_value(data['Egw']):<{w[11]}} {_format_value(data['Umc']):<{w[12]}} "
            f"{data['ETupat']}"
        ).rstrip()


@dataclass
class GroundwaterDefaults:
    """Default values for groundwater flow."""

    aquifer: str = ""
    node: str = ""
    esurf: float = 0.0
    a1: float = 0.0
    b1: float = 0.0
    a2: float = 0.0
    b2: float = 0.0
    a3: float = 0.0
    dsw: float = 0.0
    egwt: str = ""
    ebot: str = ""
    wgr: str = ""
    umc: str = ""

    _columns: ClassVar[list[str]] = [
        "Subcatchment",
        "Aquifer",
        "Node",
        "Esurf",
        "A1",
        "B1",
        "A2",
        "B2",
        "A3",
        "Dsw",
        "Egwt",
        "Ebot",
        "Wgr",
        "Umc",
    ]
    _widths: ClassVar[list[int]] = [16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for groundwater."""
        data = {
            "Aquifer": overrides.get("Aquifer", overrides.get("aquifer", self.aquifer)),
            "Node": overrides.get("Node", overrides.get("node", self.node)),
            "Esurf": overrides.get("Esurf", overrides.get("esurf", self.esurf)),
            "A1": overrides.get("A1", overrides.get("a1", self.a1)),
            "B1": overrides.get("B1", overrides.get("b1", self.b1)),
            "A2": overrides.get("A2", overrides.get("a2", self.a2)),
            "B2": overrides.get("B2", overrides.get("b2", self.b2)),
            "A3": overrides.get("A3", overrides.get("a3", self.a3)),
            "Dsw": overrides.get("Dsw", overrides.get("dsw", self.dsw)),
            "Egwt": overrides.get("Egwt", overrides.get("egwt", self.egwt)),
            "Ebot": overrides.get("Ebot", overrides.get("ebot", self.ebot)),
            "Wgr": overrides.get("Wgr", overrides.get("wgr", self.wgr)),
            "Umc": overrides.get("Umc", overrides.get("umc", self.umc)),
        }
        w = self._widths
        return (
            f"{name:<{w[0]}} {data['Aquifer']:<{w[1]}} {data['Node']:<{w[2]}} "
            f"{_format_value(data['Esurf']):<{w[3]}} {_format_value(data['A1']):<{w[4]}} "
            f"{_format_value(data['B1']):<{w[5]}} {_format_value(data['A2']):<{w[6]}} "
            f"{_format_value(data['B2']):<{w[7]}} {_format_value(data['A3']):<{w[8]}} "
            f"{_format_value(data['Dsw']):<{w[9]}} {data['Egwt']:<{w[10]}} "
            f"{data['Ebot']:<{w[11]}} {data['Wgr']:<{w[12]}} {data['Umc']}"
        ).rstrip()


# --- Coordinate Defaults ---


@dataclass
class CoordinateDefaults:
    """Default values for coordinates."""

    x: float = 0.0
    y: float = 0.0

    _columns: ClassVar[list[str]] = ["Name", "X", "Y"]
    _widths: ClassVar[list[int]] = [16, 18, 18]

    @classmethod
    def columns(cls) -> list[str]:
        return cls._columns

    def to_inp_row(self, name: str, **overrides: Any) -> str:
        """Generate INP file row for coordinates."""
        x = overrides.get("X", overrides.get("x", self.x))
        y = overrides.get("Y", overrides.get("y", self.y))
        w = self._widths
        return f"{name:<{w[0]}} {_format_value(x):<{w[1]}} {_format_value(y)}"
