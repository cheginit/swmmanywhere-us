"""Minimal SWMManywhere-US example.

Generates a synthetic SWMM model for a small residential area from
public US datasets (3DEP DEM, NLCD, OpenStreetMap, NOAA Atlas 14).
All inputs are downloaded automatically into ``base_dir`` on first run
and cached for subsequent runs.

Run::

    python docs/example.py
"""

from __future__ import annotations

from swmmanywhere_us import configure_logger, swmmanywhere

configure_logger(level="INFO")

config = {
    # Downloads and generated models are written here (created if missing).
    "base_dir": "data",
    "project": "demo",
    # Area of interest in EPSG:4326.  Processing extends the box by
    # ``buffer_km`` so the drainage topology around the AOI is complete.
    "bbox": {
        "xmin": -88.162,
        "ymin": 41.772,
        "xmax": -88.150,
        "ymax": 41.780,
        "buffer_km": 1,
    },
    # --- optional subsystems -------------------------------------------
    # Detention-pond subsystem (pond insertion, outlet structures, pipe
    # rerouting into ponds, pondshed delineation).  Off by default.
    # "add_pondsheds": True,
    #
    # The dual-drainage surface overlay (street channels + inlet grates)
    # is ON by default; disable it for a pipes-only network:
    # "params_overrides": {"dual_drainage": {"enabled": False}},
    #
    # Any parameter group can be tuned the same way, e.g.:
    # "params_overrides": {"hydraulic_design": {"design_return_period": "25"}},
}

inp_path = swmmanywhere(config)
