"""Run a SWMM model and read its results, without pyswmm / swmm-toolkit.

SWMManywhere-US writes an EPA SWMM ``.inp``; this module runs it with the
from-source ``runswmm`` engine (built by ``swmm-solver`` via pixi-build, see the
``swmm`` pixi environment) and reads the binary ``.out`` directly from the EPA
SWMM output C library (``libswmm-output``) through ctypes.  No compiled Python
bindings are involved, so it avoids the swmm-toolkit wheel / macOS code-signing
problems entirely.

The attribute enums and the by-name element lookup mirror the official EPA
``epaswmm`` package's ``Output`` API (which wraps this same C library via
Cython); the only deliberate difference is that the series accessors return
numpy arrays aligned to ``times`` rather than ``{datetime: value}`` maps.

The results are exposed per element *by name*, so downstream code can pull the
series it needs for a specific node or link, e.g. to visualize a pondshed
outlet, a pond's stored volume, or its inflow / outflow::

    from swmm_results import SwmmResults, run_swmm, NodeAttr, LinkAttr

    rpt, out = run_swmm("model.inp")
    with SwmmResults(out) as res:
        t = res.times_hours                          # report-step time axis
        vol = res.node_series("Pond3", NodeAttr.VOLUME)        # pond storage (m3)
        qin = res.node_series("Pond3", NodeAttr.TOTAL_INFLOW)  # pond inflow
        qout = res.link_series("Pond3-orifice", LinkAttr.FLOW) # pond outflow
        qshed = res.node_series("Outfall7", NodeAttr.TOTAL_INFLOW)  # pondshed outlet

Run it directly for a quick summary + a system-outfall plot::

    .../.pixi/envs/swmm/bin/python docs/swmm_results.py model_dir_or_inp
"""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Self

import numpy as np

# SWMM stores dates as days since this epoch (the Delphi/Excel-1900 convention).
_SWMM_EPOCH = datetime(1899, 12, 30)  # noqa: DTZ001 - SWMM dates are naive/local


class NodeAttr(IntEnum):
    """Node result attributes (``SMO_nodeAttribute``)."""

    DEPTH = 0  # water depth above invert (ft or m)
    HEAD = 1  # hydraulic head / water-surface elevation (ft or m)
    VOLUME = 2  # stored + ponded volume (ft3 or m3) -- pond storage
    LATERAL_INFLOW = 3  # runoff + external inflow (flow units)
    TOTAL_INFLOW = 4  # lateral + upstream inflow (flow units) -- node/pond inflow
    FLOODING = 5  # surface flooding / overflow (flow units)


class LinkAttr(IntEnum):
    """Link result attributes (``SMO_linkAttribute``)."""

    FLOW = 0  # flow rate (flow units) -- e.g. pond outlet discharge
    DEPTH = 1  # flow depth (ft or m)
    VELOCITY = 2  # flow velocity (ft/s or m/s)
    VOLUME = 3  # stored volume (ft3 or m3)
    CAPACITY = 4  # fraction of conduit filled (-)


class SubcatchAttr(IntEnum):
    """Subcatchment result attributes (``SMO_subcatchAttribute``)."""

    RAINFALL = 0
    SNOW_DEPTH = 1
    EVAP_LOSS = 2
    INFIL_LOSS = 3
    RUNOFF = 4  # runoff flow (flow units)
    GW_OUTFLOW = 5
    GW_ELEV = 6
    SOIL_MOISTURE = 7


class SystemAttr(IntEnum):
    """System-wide result attributes (``SMO_systemAttribute``)."""

    AIR_TEMP = 0
    RAINFALL = 1
    SNOW_DEPTH = 2
    EVAP_INFIL_LOSS = 3
    RUNOFF = 4
    DRY_WEATHER_INFLOW = 5
    GROUNDWATER_INFLOW = 6
    RDII_INFLOW = 7
    DIRECT_INFLOW = 8
    TOTAL_LATERAL_INFLOW = 9
    FLOODING = 10  # total flooding across all nodes (flow units)
    OUTFALL_FLOW = 11  # total flow leaving via outfalls (flow units)
    STORAGE = 12  # total stored volume (ft3 or m3)
    EVAP_RATE = 13


# SMO_elementType
_SUBCATCH, _NODE, _LINK = 0, 1, 2
# SMO_time
_REPORT_STEP, _NUM_PERIODS = 0, 1

_FLOW_UNITS = ("CFS", "GPM", "MGD", "CMS", "LPS", "MLD")  # SMO_flowUnits order


def find_engine() -> Path:
    """Locate the ``runswmm`` executable (on PATH in the ``swmm`` pixi env)."""
    exe = shutil.which("runswmm")
    if exe is None:
        msg = "runswmm not found on PATH; run inside the `swmm` pixi env (pixi run -e swmm ...)"
        raise FileNotFoundError(msg)
    return Path(exe)


def find_output_lib(engine: Path | None = None) -> Path:
    """Locate ``libswmm-output`` (sits in ``../lib`` next to ``runswmm``)."""
    engine = engine or find_engine()
    libdir = engine.parent.parent / "lib"
    for pattern in ("libswmm-output.dylib", "libswmm-output.so", "swmm-output.dll"):
        hits = list(libdir.glob(pattern))
        if hits:
            return hits[0]
    msg = f"libswmm-output not found under {libdir}"
    raise FileNotFoundError(msg)


def run_swmm(
    inp: str | Path,
    rpt: str | Path | None = None,
    out: str | Path | None = None,
    *,
    engine: str | Path | None = None,
    quiet: bool = True,
) -> tuple[Path, Path]:
    """Run a SWMM ``.inp`` with ``runswmm``; return the ``(report, output)`` paths.

    ``rpt``/``out`` default to the input stem with ``.rpt``/``.out`` suffixes.
    """
    inp = Path(inp)
    rpt = Path(rpt) if rpt else inp.with_suffix(".rpt")
    out = Path(out) if out else inp.with_suffix(".out")
    engine = Path(engine) if engine else find_engine()
    proc = subprocess.run(  # noqa: S603 - engine path is resolved, args are file paths
        [str(engine), str(inp), str(rpt), str(out)],
        check=False,
        capture_output=quiet,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-2000:] if quiet else ""
        msg = f"runswmm failed (rc={proc.returncode}) on {inp}\n{tail}"
        raise RuntimeError(msg)
    return rpt, out


class SwmmResults:
    """Read a SWMM binary ``.out`` file via ``libswmm-output`` (ctypes).

    Use as a context manager so the file handle is always closed::

        with SwmmResults(out_path) as res:
            series = res.node_series("J1", NodeAttr.TOTAL_INFLOW)
    """

    # -- internals -------------------------------------------------------
    def _bind(self) -> None:
        lib = self._lib
        p_int, p_float, p_char = (
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_char_p),
        )
        lib.SMO_init.argtypes = [ct.POINTER(ct.c_void_p)]
        lib.SMO_open.argtypes = [ct.c_void_p, ct.c_char_p]
        lib.SMO_close.argtypes = [ct.c_void_p]
        lib.SMO_getProjectSize.argtypes = [ct.c_void_p, ct.POINTER(p_int), p_int]
        lib.SMO_getUnits.argtypes = [ct.c_void_p, ct.POINTER(p_int), p_int]
        lib.SMO_getTimes.argtypes = [ct.c_void_p, ct.c_int, p_int]
        lib.SMO_getStartDate.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
        lib.SMO_getElementName.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, p_char, p_int]
        # (handle, elementIndex, attr, startPeriod, endPeriod, float**, int*)
        series_sig = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.POINTER(p_float),
            p_int,
        ]
        lib.SMO_getNodeSeries.argtypes = series_sig
        lib.SMO_getLinkSeries.argtypes = series_sig
        lib.SMO_getSubcatchSeries.argtypes = series_sig
        lib.SMO_getSystemSeries.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.POINTER(p_float),
            p_int,
        ]
        lib.SMO_freeMemory.argtypes = [ct.c_void_p]

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            msg = f"{what} failed (rc={rc}) for {self._path}"
            raise RuntimeError(msg)

    def _int_array(self, fn: object) -> list[int]:
        arr, length = ct.POINTER(ct.c_int)(), ct.c_int()
        self._check(
            fn(self._handle, ct.byref(arr), ct.byref(length)), getattr(fn, "__name__", "fn")
        )
        vals = [int(arr[i]) for i in range(length.value)]
        self._lib.SMO_freeMemory(arr)
        return vals

    def __init__(self, out_path: str | Path, output_lib: str | Path | None = None) -> None:
        self._path = Path(out_path)
        self._lib = ct.CDLL(str(output_lib or find_output_lib()))
        self._bind()
        self._handle = ct.c_void_p()
        self._check(self._lib.SMO_init(ct.byref(self._handle)), "SMO_init")
        self._check(self._lib.SMO_open(self._handle, str(self._path).encode()), "SMO_open")
        sizes = self._int_array(self._lib.SMO_getProjectSize)
        self.n_subcatch, self.n_node, self.n_link = sizes[0], sizes[1], sizes[2]
        units = self._int_array(self._lib.SMO_getUnits)
        self.flow_units = _FLOW_UNITS[units[1]] if len(units) > 1 else "?"
        step = ct.c_int()
        self._lib.SMO_getTimes(self._handle, _REPORT_STEP, ct.byref(step))
        self.report_step = step.value  # seconds between reporting periods
        nper = ct.c_int()
        self._lib.SMO_getTimes(self._handle, _NUM_PERIODS, ct.byref(nper))
        self.n_periods = nper.value
        date = ct.c_double()
        self._lib.SMO_getStartDate(self._handle, ct.byref(date))
        self.start_date = _SWMM_EPOCH + timedelta(days=date.value)
        self._names: dict[int, dict[str, int]] = {}

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._lib.SMO_close(self._handle)
            self._handle = None

    # -- time axis -------------------------------------------------------
    @property
    def times_hours(self) -> np.ndarray:
        """Reporting times as hours from the simulation start."""
        return np.arange(self.n_periods) * (self.report_step / 3600.0)

    @property
    def times(self) -> list[datetime]:
        """Reporting times as datetimes."""
        return [
            self.start_date + timedelta(seconds=i * self.report_step) for i in range(self.n_periods)
        ]

    def _name_index(self, element_type: int, count: int) -> dict[str, int]:
        if element_type not in self._names:
            mapping: dict[str, int] = {}
            for i in range(count):
                name_p, size = ct.c_char_p(), ct.c_int()
                self._check(
                    self._lib.SMO_getElementName(
                        self._handle, element_type, i, ct.byref(name_p), ct.byref(size)
                    ),
                    "SMO_getElementName",
                )
                mapping[name_p.value.decode()] = i
                self._lib.SMO_freeMemory(ct.cast(name_p, ct.c_void_p))
            self._names[element_type] = mapping
        return self._names[element_type]

    # -- element names ---------------------------------------------------
    @property
    def node_names(self) -> list[str]:
        return list(self._name_index(_NODE, self.n_node))

    @property
    def link_names(self) -> list[str]:
        return list(self._name_index(_LINK, self.n_link))

    @property
    def subcatchment_names(self) -> list[str]:
        return list(self._name_index(_SUBCATCH, self.n_subcatch))

    def _to_array(self, arr: object, length: int) -> np.ndarray:
        vals = np.ctypeslib.as_array(arr, shape=(length,)).astype(np.float64)
        self._lib.SMO_freeMemory(arr)
        return vals

    def _series(self, fn: object, element_type: int, name: str, attr: int) -> np.ndarray:
        index = self._name_index(
            element_type, getattr(self, ("n_subcatch", "n_node", "n_link")[element_type])
        )
        if name not in index:
            kind = ("subcatchment", "node", "link")[element_type]
            msg = f"{kind} {name!r} not found in {self._path.name}"
            raise KeyError(msg)
        arr, length = ct.POINTER(ct.c_float)(), ct.c_int()
        self._check(
            fn(
                self._handle,
                index[name],
                attr,
                0,
                self.n_periods - 1,
                ct.byref(arr),
                ct.byref(length),
            ),
            getattr(fn, "__name__", "series"),
        )
        return self._to_array(arr, length.value)

    # -- series by name --------------------------------------------------
    def node_series(self, name: str, attr: NodeAttr) -> np.ndarray:
        """Time series of ``attr`` for node ``name`` (length == n_periods)."""
        return self._series(self._lib.SMO_getNodeSeries, _NODE, name, int(attr))

    def link_series(self, name: str, attr: LinkAttr) -> np.ndarray:
        """Time series of ``attr`` for link ``name``."""
        return self._series(self._lib.SMO_getLinkSeries, _LINK, name, int(attr))

    def subcatchment_series(self, name: str, attr: SubcatchAttr) -> np.ndarray:
        """Time series of ``attr`` for subcatchment ``name``."""
        return self._series(self._lib.SMO_getSubcatchSeries, _SUBCATCH, name, int(attr))

    def system_series(self, attr: SystemAttr) -> np.ndarray:
        """Time series of a system-wide ``attr`` (e.g. total outfall flow)."""
        arr, length = ct.POINTER(ct.c_float)(), ct.c_int()
        self._check(
            self._lib.SMO_getSystemSeries(
                self._handle, int(attr), 0, self.n_periods - 1, ct.byref(arr), ct.byref(length)
            ),
            "SMO_getSystemSeries",
        )
        return self._to_array(arr, length.value)

    # -- pond / pondshed convenience ------------------------------------
    def pond_volume(self, node: str) -> np.ndarray:
        """Stored volume time series for a pond / storage node (flow-unit volume units)."""
        return self.node_series(node, NodeAttr.VOLUME)

    def node_inflow(self, node: str) -> np.ndarray:
        """Total inflow (lateral + upstream) to a node, e.g. pond or pondshed outlet."""
        return self.node_series(node, NodeAttr.TOTAL_INFLOW)

    def node_flooding(self, node: str) -> np.ndarray:
        """Surface flooding / overflow rate at a node."""
        return self.node_series(node, NodeAttr.FLOODING)

    def link_flow(self, link: str) -> np.ndarray:
        """Flow rate through a link, e.g. a pond's orifice / weir outlet (outflow)."""
        return self.link_series(link, LinkAttr.FLOW)

    # -- context manager -------------------------------------------------
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _demo(target: str) -> None:
    """Run a model (dir or .inp) and print a node/link/system summary + plot outfalls."""
    target = Path(target)
    if target.is_dir():
        sim = target / f"{target.name}_sim.inp"
        target = sim if sim.exists() else target / f"{target.name}.inp"
    rpt, out = run_swmm(target)
    print(f"ran {target.name} -> {rpt.name}, {out.name}")
    with SwmmResults(out) as res:
        print(
            f"{res.n_node} nodes, {res.n_link} links, {res.n_subcatch} subcatchments; "
            f"{res.n_periods} periods @ {res.report_step}s; flow units {res.flow_units}"
        )
        q = res.system_series(SystemAttr.OUTFALL_FLOW)
        print(
            f"system outfall flow: peak {q.max():.0f} {res.flow_units}, "
            f"first node {res.node_names[0]!r}, first link {res.link_names[0]!r}"
        )
        try:
            import matplotlib as mpl

            mpl.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(res.times_hours, q, color="#c1272d", lw=1.6)
            ax.set(
                xlabel="time (hours)",
                ylabel=f"outfall flow ({res.flow_units})",
                title=f"{target.stem}: system outfall hydrograph (runswmm)",
            )
            ax.grid(alpha=0.3)
            png = Path(target).with_name(f"{Path(target).stem}_outfall.png")
            fig.tight_layout()
            fig.savefig(png, dpi=130)
            print(f"wrote {png}")
        except ImportError:
            print("(matplotlib not available; skipped plot)")


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1] if len(sys.argv) > 1 else ".")
