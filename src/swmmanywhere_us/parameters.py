"""Parameters module for SWMManywhere-US."""

from __future__ import annotations

from typing import Self, TypedDict

import numpy as np
from pydantic import BaseModel, Field, model_validator


class SubcatchmentDerivation(BaseModel):
    """Parameters for subcatchment derivation."""

    min_drainage_area_m2: float = Field(
        default=100_000,
        ge=0,
        description="Minimum upstream drainage area (m²) for stream extraction.",
    )
    lane_width: float = Field(default=3.5, ge=2.0, le=5.0, description="Width of a road lane.")
    max_street_length: float = Field(
        default=60.0, ge=40.0, le=100.0, description="Distance to split streets into segments."
    )
    dem_resolution: int = Field(default=10, ge=0, description="Resolution of the DEM to use.")
    lulc_year: int = Field(
        default=2019, ge=1985, le=2024, description="Year of the LULC data to use."
    )
    buffer_size_local: float = Field(
        default=5.0, ge=0.0, le=100.0, description="Buffer size for local street cleanup."
    )
    min_hole_areasqm_local: float = Field(
        default=1000.0, ge=100.0, le=10000.0, description="Minimum hole area for local streets."
    )
    buffer_size_major: float = Field(
        default=15.0, ge=0.0, le=200.0, description="Buffer size for major street cleanup."
    )
    min_hole_areasqm_major: float = Field(
        default=20000.0, ge=1000.0, le=100000.0, description="Minimum hole area for major streets."
    )


class OutfallDerivation(BaseModel):
    """Parameters for outfall derivation."""

    river_buffer_distance: float = Field(
        default=150.0, ge=10.0, le=500.0, description="Buffer distance to link rivers to streets."
    )
    outfall_clustering_factor: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description=(
            "Outfall consolidation strength, as a multiple of the network's median "
            "pipe length.  identify_outfalls penalizes each candidate street->outfall "
            "link in the minimum spanning tree by ``factor * median_pipe_length``, so "
            "a cluster of streets collapses onto one shared outfall when they are "
            "linked by pipes shorter than that penalty.  Expressing it relative to "
            "pipe length (rather than an absolute distance) keeps outfall density "
            "consistent across dense and sparse networks: factor < 1 keeps outfalls "
            "local (little consolidation), factor > 1 merges aggressively, 0 disables "
            "consolidation entirely.  This is a selection weight only, the retained "
            "outfall conduit's length is the real street-to-receiving-water distance."
        ),
    )
    include_water_body_outfalls: bool = Field(
        default=False,
        description=(
            "Treat water-body polygons (basins/water_bodies parquet) as outfall "
            "candidates in addition to rivers. A street node within "
            "river_buffer_distance of a polygon's boundary is paired to a synthetic "
            "outfall sink at that water body. Off by default so the pond subsystem "
            "(which models water bodies as storage sources) is unaffected; enable "
            "for pond-free / bare networks where water bodies are receiving waters."
        ),
    )
    water_body_min_area_m2: float = Field(
        default=200.0,
        ge=0.0,
        description="Minimum water-body polygon area to qualify as an outfall candidate.",
    )
    online_pond_intake: bool = Field(
        default=False,
        description=(
            "Make detention ponds on-line sinks: pipe nodes within "
            "pond_intake_buffer_m of a pond's footprint are paired to that "
            "pond's STORAGE node with an outfall edge, so derive_topology routes "
            "the surrounding catchment INTO the pond (which then discharges via "
            "its orifice/weir). Without this a pond only captures the catchment "
            "that happens to converge at its single outlet junction, leaving the "
            "surrounding area to drain past it to the regional outfall. Off by "
            "default; raises pond inflow (and overtopping of under-sized ponds)."
        ),
    )
    pond_intake_buffer_m: float = Field(
        default=30.0,
        ge=0.0,
        le=300.0,
        description=(
            "Distance from a pond's footprint within which pipe nodes are paired "
            "to the pond as on-line intakes (only used when online_pond_intake)."
        ),
    )
    pond_intake_min_area_m2: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum pond footprint area to receive on-line intakes (only used "
            "when online_pond_intake).  Ponds below this are left off-line so a "
            "large catchment isn't concentrated onto a small basin and its "
            "feeder trunks.  0 (default) intakes every pond."
        ),
    )
    pond_footprint_match_m: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Maximum distance between a pond's STORAGE node and a basin polygon "
            "for that polygon to be taken as the pond's own footprint (only used "
            "when online_pond_intake).  insert_pond_nodes places the storage node "
            "at the basin centroid, so the match is near-exact; this small "
            "tolerance stops a stray storage node from grabbing a distant polygon. "
            "Storage nodes with no basin within this distance are left off-line."
        ),
    )


class TopologyDerivation(BaseModel):
    """Parameters for topology derivation."""

    omit_edges: list[str] = Field(
        default=["corridor", "track", "footway", "path"],
        min_length=1,
        description="OSM paths pipes are not allowed under.",
    )
    weights: list[str] = Field(
        default=["chahinian_slope", "length", "contributing_area"],
        min_length=1,
        description="Weights for topo derivation.",
    )
    chahinian_slope_scaling: float = Field(default=1, le=1, ge=0)
    chahinian_angle_scaling: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description=(
            "Weight of the turn-angle transition cost (Chahinian et al. 2019, "
            "Eqs. 4-5) applied while the Dijkstra forest is grown in "
            "derive_topology: extending the forest with a pipe costs an extra "
            "scaling * C_theta(angle between the pipe and the junction's "
            "already-chosen downstream pipe).  C_theta favors straight-"
            "through junctions (180 deg, cost 0) and street-grid right angles "
            "(90 deg, cost 0.2) and penalizes acute turns (cost 1).  Unlike "
            "the entries in `weights`, this is a transition cost, it cannot "
            "be precomputed per edge.  Default 0.3 is the paper's best-fit "
            "alpha_theta on Prades-le-Lez.  Set 0 to disable."
        ),
    )
    length_scaling: float = Field(default=0.1, le=1, ge=0)
    contributing_area_scaling: float = Field(default=0.1, le=1, ge=0)
    chahinian_slope_exponent: float = Field(default=1, le=2, ge=0)
    length_exponent: float = Field(default=1, le=2, ge=0)
    contributing_area_exponent: float = Field(default=1, le=2, ge=0)

    @model_validator(mode="after")
    def check_weights(self) -> Self:
        """Check that weights have associated scaling and exponents."""
        for weight in self.weights:
            if not hasattr(self, f"{weight}_scaling"):
                msg = f"Missing {weight}_scaling"
                raise ValueError(msg)
            if not hasattr(self, f"{weight}_exponent"):
                msg = f"Missing {weight}_exponent"
                raise ValueError(msg)
        return self


class ChannelDesign(BaseModel):
    """Parameters for open channel geometry estimation.

    Channel width is estimated from upstream contributing area using
    Leopold-Maddock hydraulic geometry: W = width_coeff * A_km2^width_exponent.
    Depth is derived as depth_ratio * width.

    References:
        Leopold, L.B. & Maddock, T. (1953). The hydraulic geometry of stream
        channels and some physiographic implications. USGS Professional Paper 252.
    """

    mannings_n: float = Field(
        default=0.035, ge=0.01, le=0.2, description="Manning's n for natural open channels."
    )
    width_coeff: float = Field(
        default=2.7,
        ge=0.1,
        le=20.0,
        description="Leopold-Maddock coefficient: W = coeff * A_km2^exp.",
    )
    width_exponent: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Leopold-Maddock exponent."
    )
    depth_ratio: float = Field(
        default=0.4,
        ge=0.1,
        le=1.0,
        description="Depth-to-width ratio for rectangular approximation.",
    )
    min_width: float = Field(default=1.0, ge=0.3, le=10.0, description="Minimum channel width (m).")
    max_width: float = Field(
        default=50.0, ge=5.0, le=200.0, description="Maximum channel width (m)."
    )
    min_depth: float = Field(default=0.3, ge=0.1, le=2.0, description="Minimum channel depth (m).")
    max_depth: float = Field(default=5.0, ge=1.0, le=20.0, description="Maximum channel depth (m).")


class PondDesign(BaseModel):
    """Parameters for stormwater pond modeling.

    Based on FDOT Drainage Design Guide and Opti CMAC guidelines.
    Controls geometric characterization, outlet structure sizing,
    and emergency spillway design for detention ponds modeled as
    SWMM storage units.

    The FDOT volume-area regression and Opti CMAC orifice table were
    calibrated on Florida subdivision detention ponds typically < 2 ha
    and 3-6 ft deep (Harper & Baker 2007).  Larger natural lakes and
    wetlands should not be modeled as detention ponds, see
    ``max_pond_area_m2`` and ``lake_tags`` below for the classification
    rule.

    References:
        FDOT Drainage Design Guide, Chapter 9.
        Opti CMAC Controllable Volume Implementation Guidelines.
        Richardson et al. 2022, *Scientific Reports* 12:10472, 
            functional definition separating ponds (<=5 ha, <=5 m, <30%
            emergent vegetation) from lakes and wetlands.
        Harper & Baker 2007, *Evaluation of Current Stormwater Design
            Criteria within the State of Florida*, FDEP final report, 
            Florida stormwater-pond calibration envelope (<=2 ha, 3-6 ft).
        Wetzel 2001, *Limnology: Lake and River Ecosystems* (3rd ed.), 
            limnological pond threshold (~2 ha, <3 m).
    """

    side_slope_pct: float = Field(
        default=25.0,
        ge=10.0,
        le=50.0,
        description="Side slope as percentage (V/H * 100). 25% -> H:V = 4:1.",
    )
    max_depth_m: float = Field(
        default=3.81,
        ge=1.0,
        le=6.0,
        description="Maximum design depth (m). FDOT default 12.5 ft = 3.81 m.",
    )
    n_curve_points: int = Field(
        default=5,
        ge=3,
        le=9,
        description="Number of stage-area curve points.",
    )
    bottom_area_min_ratio: float = Field(
        default=0.30,
        ge=0.1,
        le=0.9,
        description="Minimum bottom-to-surface area ratio (prevents degenerate geometry).",
    )
    orifice_cd: float = Field(
        default=0.65,
        ge=0.5,
        le=0.8,
        description="Orifice discharge coefficient.",
    )
    orifice_standard_diameters_m: list[float] = Field(
        default=[
            0.1524,  # 6 in
            0.2032,  # 8 in (Opti-CMAC small)
            0.3048,  # 12 in (Opti-CMAC medium)
            0.4572,  # 18 in (Opti-CMAC large)
            0.6096,  # 24 in (Opti-CMAC extra-large / FDOT subdivision cap)
            0.7620,  # 30 in (regional)
            0.9144,  # 36 in (regional)
            1.2192,  # 48 in (regional/reservoir)
        ],
        description=(
            "Standard orifice / riser diameters (m) available for pond "
            "outlet sizing.  Values are whole inches (6 / 8 / 12 / 18 / "
            "24 / 30 / 36 / 48 in) per FDOT Drainage Design Guide "
            "Chapter 9 and Opti-CMAC catalog.  Rounded up to the "
            "nearest in resize_pond_orifices."
        ),
    )
    max_orifice_diameter_m: float = Field(
        default=0.9144,
        ge=0.15,
        le=2.0,
        description=(
            "Hard cap on pond orifice diameter (m).  Default 0.9144 m = "
            "36 in matches FDOT / SJRWMD subdivision-class retention-pond "
            "practice; regional ponds go up to 60-72 in but require "
            "site-specific design."
        ),
    )
    orifice_sizing_peak_margin: float = Field(
        default=1.1,
        ge=0.5,
        le=2.0,
        description=(
            "Safety factor applied to the pond's peak design inflow when "
            "picking the target orifice flow: Q_target >= margin * "
            "Q_peak_inflow so the orifice can pass the rational-method "
            "peak without surcharging the pond."
        ),
    )
    orifice_sizing_pipe_fraction: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description=(
            "Maximum fraction of the first downstream street pipe's "
            "Manning full-flow capacity that the orifice may pass.  "
            "Keeps the orifice from moving the hydraulic bottleneck "
            "downstream to the pipe and producing a new flood at the "
            "pipe junction.  Default 0.8 reserves 20 % for in-street "
            "runoff that also uses the same downstream pipe."
        ),
    )
    weir_cw: float = Field(
        default=3.0,
        ge=2.0,
        le=4.0,
        description=(
            "Sharp-crested (transverse) weir discharge coefficient in "
            "US-customary units (FDOT/Opti-CMAC; ~3.0-3.33).  post_processing "
            "converts it to the model's flow-unit system before writing the "
            "INP (x sqrt(0.3048) ~= 1.84 for metric LPS/CMS/MLD), so configure "
            "it as a US value regardless of the chosen flow_units."
        ),
    )
    weir_crest_ratio: float = Field(
        default=0.90,
        ge=0.80,
        le=0.95,
        description="Weir crest as fraction of max depth (remainder is freeboard).",
    )
    weir_length_m: float = Field(
        default=4.6,
        ge=3.0,
        le=10.0,
        description="Default emergency weir length (m). FDOT default ~15 ft.",
    )
    fdot_volume_intercept: float = Field(
        default=0.6431,
        description="FDOT V-A intercept: V_cuft = intercept + slope * A_sqft.",
    )
    fdot_volume_slope: float = Field(
        default=2.5921,
        description="FDOT V-A slope: V_cuft = intercept + slope * A_sqft.",
    )
    min_area_m2: float = Field(
        default=200.0,
        ge=50.0,
        le=5000.0,
        description="Minimum water body surface area to model (m^2).",
    )
    max_pond_area_m2: float = Field(
        default=50_000.0,
        ge=10_000.0,
        le=500_000.0,
        description=(
            "Upper area (m^2) for a water body to be treated as a designed "
            "stormwater pond suitable for FDOT V-A sizing + Opti-CMAC outlet "
            "structures.  Default 5 ha matches the Richardson et al. 2022 "
            "functional upper bound and stays within the Florida calibration "
            "envelope (Harper & Baker 2007, ~2 ha).  Polygons above this cut "
            "are classified as natural lakes/reservoirs and modeled as "
            "fixed-stage OUTFALL nodes instead of detention STORAGE."
        ),
    )
    lake_osm_tags: list[str] = Field(
        default=["lake", "reservoir", "lagoon", "oxbow"],
        description=(
            "OSM ``water=*`` values that force a polygon to be treated as a "
            "natural lake / reservoir (i.e. OUTFALL boundary), regardless of "
            "area.  Basin / pond / retention / detention tags remain eligible "
            "for designed-pond modeling."
        ),
    )
    max_snap_distance_m: float = Field(
        default=200.0,
        ge=50.0,
        le=1000.0,
        description="Maximum distance to snap a water body to the network (m).",
    )
    downstream_search_hops: int = Field(
        default=20,
        ge=1,
        le=200,
        description=(
            "Unused by the current geometric drainable-anchor rescue "
            "(kept for config compatibility).  The pipe-DAG walk it bounded "
            "was replaced by a euclidean nearest-drainable-node search "
            "limited by ``downstream_search_max_m``."
        ),
    )
    downstream_search_max_m: float = Field(
        default=750.0,
        ge=50.0,
        le=5000.0,
        description=(
            "Euclidean search radius (m) for the geometric drainable-anchor "
            "rescue in ``finalize_pond_outlets``, used when a pond's "
            "provisional anchor invert is above the pond's max water surface "
            "elevation; without it every pond in a hollow would be a closed "
            "basin.  Larger values allow long culverts but risk creating "
            "non-physical shortcuts.  750 m is roughly the scale of a "
            "residential block cluster."
        ),
    )
    min_outfall_head_drop_m: float = Field(
        default=2.0,
        ge=0.0,
        le=20.0,
        description=(
            "Minimum head differential (m) between a pond's max water "
            "surface elevation and at least one reachable SWMM outfall "
            "invert, required to consider the pond gravity-drainable.  "
            "When no reachable outfall sits this far below the pond, "
            "the pond is reclassified as closed-basin and drained via "
            "the Green-Ampt exfiltration tail (SJRWMD/SFWMD 72-hour-"
            "recovery rule, FDOT flatwoods practice).  This catches "
            "ponds whose outflow path leads to a dummy OSM-river "
            "terminus that sits only fractions of a meter below the "
            "pond, head differential too small to drive any sustained "
            "gravity flow.  Default 2 m: head differentials below 2 m "
            "would imply <~0.1 % path slope on the typical Florida "
            "subdivision-scale 1-2 km outfall reach, which can't "
            "physically convey pond design release rates.  Terminals "
            "that carry a real receiving-water stage (river_outfall / "
            "water_body_outfall sinks, invert = hydroflattened DEM "
            "stage) are exempt from this criterion, they instead need "
            "0.1 m of driving head plus the minimum-slope drop over the "
            "path, since a pond whose max WSE clears the adjacent "
            "canal's water surface drains by gravity regardless of the "
            "2 m rule.  Set to 0 to disable the check."
        ),
    )
    closed_basin_sur_depth_m: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description=(
            "SWMM ``SurDepth`` (surface ponding depth above ``MaxDepth``) "
            "applied to ponds classified as closed basins.  When the pond "
            "fills past ``MaxDepth``, water accumulates on the surface in "
            "a virtual pond of this extra depth before being lost to the "
            "flooding-loss term.  2 m matches typical Florida freeboard + "
            "yard-ponding practice (SFWMD BMP Manual)."
        ),
    )
    open_basin_sur_depth_m: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description=(
            "SWMM ``SurDepth`` applied to gravity-drained (open) pond "
            "storages.  Default 0 keeps gravity-drained ponds strictly "
            "at ``MaxDepth``, any overtopping is booked as SWMM flood "
            "loss so mass balance stays clean.  Setting this > 0 adds a "
            "virtual-surface-pond buffer above MaxDepth (same mechanism "
            "as for closed basins), which reduces pond-overtopping flood "
            "at the cost of DYNWAVE stability: the added head accelerates "
            "orifice + weir peaks (Q ~ sqrt(H)), raises downstream pipe "
            "surge rates, and can push routing-continuity error well "
            "above the 5 % target.  Empirical test-catchment trade-off at 140 mm / "
            "6 hr SCS-II: 0 m -> 495 M L flood / +5 % continuity; 0.5 m "
            "-> 390 M L flood / +13 %; 2 m -> 171 M L flood / +31 %.  "
            "Tune upward only when pond overtopping dominates real flood "
            "volume AND the user accepts higher continuity error.  No "
            "Green-Ampt seepage is emitted for open ponds, they rely on "
            "the gravity outlet, not exfiltration."
        ),
    )
    closed_basin_ksat_mm_hr: float = Field(
        default=12.5,
        ge=0.0,
        le=250.0,
        description=(
            "Green-Ampt saturated hydraulic conductivity (mm/hr) at the "
            "pond bottom for closed-basin exfiltration.  Default 12.5 mm/hr "
            "(~0.5 in/hr = 0.3 m/day) is the SJRWMD-SAM-consistent design "
            "rate: Id = (2/3) * Kvs / FS with Kvs ~= 3 ft/day for Myakka / "
            "Immokalee coastal sands and FS = 2 per Indian River County "
            "Code 930.08; the resulting 0.9 m pond drains in ~3 days, "
            "matching the SFWMD/SJRWMD 72-hour recovery rule (SJRWMD ERP "
            "Applicant's Handbook Vol. II; Harper & Baker 2007).  NRCS raw "
            "Ksat for these series is 100-250 mm/hr in the A horizon, "
            "we discount to ~10% to account for the spodic Bh layer and "
            "long-term pond clogging."
        ),
    )
    closed_basin_psi_mm: float = Field(
        default=50.0,
        ge=0.0,
        le=500.0,
        description=(
            "Green-Ampt suction head (mm) at the pond bottom.  ~50 mm is "
            "typical for HSG A sandy soils; 100-200 mm for HSG B sandy "
            "loams (Rawls et al. 1983, SWMM Reference Manual Vol. I)."
        ),
    )
    closed_basin_imd: float = Field(
        default=0.3,
        ge=0.0,
        le=0.5,
        description=(
            "Green-Ampt initial moisture deficit (dimensionless, 0-1).  "
            "0.3 assumes a moderately dry pond bottom between storms; "
            "0.0 for saturated / wet retention, 0.4-0.5 for drought "
            "conditions."
        ),
    )
    pond_inflow_offset_m: float = Field(
        default=0.30,
        ge=0.0,
        le=2.0,
        description=(
            "Backwater offset (m) added when an upstream pipe is rerouted "
            "to terminate at a pond storage node (``route_pipes_into_ponds``). "
            "Per EPA SWMM Applications Manual Example 3 (Rossman 2009) the "
            "downstream end of the inflow pipe is offset above the pond "
            "invert so that for minor storms the pipe has no backwater but "
            "its crown still sits below the pond's max water surface, "
            "default 0.30 m matches the manual's 1 ft recommendation. "
            "Stacked on top of the slope-preserving offset the rerouter "
            "computes from the original pipe geometry."
        ),
    )
    orifice_crest_offset_m: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description=(
            "Crest height (m) of the pond's primary orifice above the pond "
            "invert.  When set to 0, the orifice opening sits exactly at the "
            "pond floor and SWMM's free-flow / submerged-flow regime "
            "boundary coincides with quiescent depth, tiny depth changes "
            "flip the regime each routing step, producing flow-instability "
            "indices > 100 and DYNWAVE non-convergence at every pond.  A "
            "small positive offset (~0.10 m / 4 in) puts the orifice into "
            "free-flow regime up to that depth, eliminating per-step regime "
            "transitions and cutting non-convergence on the test catchment from ~68 % to "
            "single-digit %.  Values 0.05-0.30 m are standard SWMM detention-"
            "pond practice; FDOT 'low-flow orifice' invert offsets typically "
            "fall in this range."
        ),
    )
    sub_reroute_max_distance_m: float = Field(
        default=500.0,
        ge=0.0,
        le=5000.0,
        description=(
            "Max centroid-to-pond distance (m) for subcatchments rerouted "
            "into an isolated pond by ``reroute_subs_to_isolated_ponds``.  "
            "A pond is 'isolated' if it has zero ``pond_inflow`` edges after "
            "``route_pipes_into_ponds``, typically a head-of-network pond "
            "whose ds_node has no street-pipe predecessors.  This step "
            "rewires nearby subs' ``Outlet`` directly to the pond storage "
            "(calibrated-reference-model pattern: ``Sub.Outlet = pond_id``) so "
            "the pond's controls can manage its pondshed flooding.  "
            "Default 500 m covers most FDOT subdivision-pond catchment "
            "radii (typical sub linear scale 200-400 m).  Set to 0 to "
            "disable sub-rerouting entirely."
        ),
    )
    sub_reroute_capacity_multiplier: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description=(
            "Multiplier on the FDOT 1-inch water-quality treatment volume "
            "when computing how much catchment area an isolated pond can "
            "absorb via ``reroute_subs_to_isolated_ponds``.  The base cap "
            "is ``pond_volume_m3 / (0.0254 * 0.5)`` (1 inch of runoff at "
            "50 % runoff coefficient) per SJRWMD/SFWMD ERP Applicant's "
            "Handbook Vol. II §10, the catchment size that fills the "
            "pond's design volume with 1 inch.  Default 2.0x allows the "
            "pond to absorb roughly the runoff from a 5 inch design storm "
            "(close to Florida 10-yr 6-hr Atlas-14, ~5.5 inch).  Higher "
            "values let the pond manage more pondshed but risk overflow "
            "during the test storm; lower values leave more subs on the "
            "pipe network."
        ),
    )
    enclosed_gap_close_m: float = Field(
        default=75.0,
        ge=0.0,
        le=300.0,
        description=(
            "Morphological-close radius for ``reroute_enclosed_gap_subs``.  A "
            "pond's piped catchment can be a disconnected MultiPolygon because "
            "the shortest-path storm network routes a strip of subs PAST the "
            "pond to a distant outfall, splitting its catchment.  Where such "
            "outfall-bound subs are spatially ENCLOSED by the pond's own "
            "catchment (they fall in the gap region of a buffer(+R)/buffer(-R) "
            "close), they belong to the pond's natural drainage and are "
            "rerouted to it (the distant-outfall route is a routing artifact). "
            "A conservative R keeps it to truly-enclosed gaps; wider outfall "
            "corridors are left alone to avoid over-reach.  0 disables."
        ),
    )


class HydraulicDesign(BaseModel):
    """Parameters for hydraulic design."""

    diameters: list[float] = Field(
        default=np.linspace(0.15, 3, int((3 - 0.15) / 0.075) + 1).tolist(), min_length=1
    )
    max_fr: float = Field(default=0.8, le=1, ge=0)
    min_shear: float = Field(default=2, le=3, ge=0)
    min_v: float = Field(
        default=0.61,
        le=2,
        ge=0.3,
        description="Minimum permissible full-pipe velocity (m/s). Storm drain "
        "standards typically require 2 fps (0.61 m/s) to prevent "
        "sediment deposition.",
    )
    max_v: float = Field(
        default=3.05,
        le=10,
        ge=2.0,
        description="Maximum permissible full-pipe velocity (m/s). Storm drain "
        "standards typically limit to 10 fps (3.05 m/s) to prevent "
        "erosion.",
    )
    min_depth: float = Field(default=0.5, le=1, ge=0)
    max_depth: float = Field(default=5, le=10, ge=2)
    precipitation: float = Field(default=0.006, le=1.0, ge=0.001)
    design_return_period: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Design storm return period (years) for NOAA Atlas 14 lookup.",
    )
    design_duration: str = Field(
        default="60-min",
        description="Design storm duration for NOAA Atlas 14 lookup (e.g., '60-min', '30-min').",
    )
    depth_nbins: int = Field(default=10, ge=1)
    edge_design_parameters: list[str] = Field(default=["diameter", "cost_usd"], min_length=1)
    min_positive_slope: float = Field(
        default=0.001,
        ge=1e-6,
        le=0.01,
        description="Minimum positive pipe slope (m/m) enforced during design.",
    )
    flat_terrain_capacity_factor: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description=(
            "Safety factor for pipe-full Manning capacity over design peak flow "
            "on pipes where the minimum-slope constraint is binding. "
            "Addresses the known flat-terrain failure mode of the Duque 2016 "
            "pipe-by-pipe method (Hesarkazzazi et al. 2022, Saldarriaga et al. "
            "2024): when slope is locked to the minimum rather than driven by "
            "hydraulics, the accept-first-feasible scan picks the smallest "
            "diameter clearing min-slope, which has no capacity margin for "
            "dynamic-wave peak flows.  For each binding-slope pipe, the "
            "diameter is bumped up until Q_full >= factor * Q_design.  "
            "Set to 1.0 to disable."
        ),
    )
    manhole_drop_m: float = Field(
        default=0.03,
        ge=0.0,
        le=0.3,
        description=(
            "Hydraulic drop (m) applied at each manhole by setting the "
            "street pipe's SWMM ``OutOffset``, matches the pattern seen "
            "in the calibrated UWO sewer reference model where every "
            "street pipe has a small drop (0.01-0.27 m, median ~0.03 m) "
            "at its downstream junction.  Standard sanitary-sewer / "
            "storm-drain design (ASCE MOP 37 Sec. 5.4): a small drop at "
            "each manhole dissipates flow energy, prevents backwater "
            "surges, and gives DYNWAVE a non-zero gradient even at "
            "otherwise-flat junctions.  Default 0.03 m (~1 in) is the "
            "low end of typical Florida FDOT design practice (3-9 cm). "
            "Per-pipe drop is capped at half the pipe's actual hydraulic "
            "drop so slope stays positive; pipes shallower than "
            "``2*manhole_drop_m`` get a proportional drop rather than "
            "the full value.  Set to 0 to disable."
        ),
    )
    max_outfall_slope: float = Field(
        default=0.05,
        ge=0.01,
        le=1.0,
        description=(
            "Upper bound on pipe slope (m/m) for an outfall conduit "
            "(street / river node -> receiving water / dummy river).  "
            "If the elevation drop over the outfall conduit's length "
            "exceeds this slope, the pipe is stretched so that slope = "
            "max_outfall_slope, conceptually, the outfall moves "
            "`along the river / water body` to a farther attachment "
            "point where the culvert gradient becomes physically "
            "reasonable.  Default 5 % keeps DYNWAVE convergence clean "
            "(10 % pipes still churn in DYNWAVE non-convergence even "
            "though they're within FDOT's culvert criterion).  FDOT "
            "Drainage Design Guide Ch. 6 and ASCE MOP 37 §4.2.3 allow "
            "up to 10 % for free-flowing culverts, but 5 % is typical "
            "for larger trunk outfalls to avoid supercritical flow and "
            "the high-velocity scour that implies.  Without this "
            "constraint, short outfall conduits (especially dummy-river "
            "outfalls for sub-basins with no natural receiving water) "
            "dropping several meters become 100-200 % slope links, which "
            "destroy DYNWAVE convergence on that single link."
        ),
    )


class DualDrainage(BaseModel):
    """Parameters for the dual-drainage (surface + subsurface) overlay.

    Adds an open-channel surface conduit parallel to every street pipe so
    that flow which exceeds pipe capacity is routed overland along the
    street to the next junction instead of disappearing into SWMM's
    flooding loss term.  Recommended for flat terrain (e.g. Gulf Coast,
    South Florida, Mississippi Delta) where the Duque 2016 pipe-series
    sizing produces pipes that saturate under design storms.

    References: Reyes-Silva et al. 2022 (Water 15(1):46), susdrain
    "Designing for Exceedance" guidance.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Enable the dual-drainage overlay.  When False the step is a "
            "no-op and only the subsurface pipe network is modeled."
        ),
    )
    curb_depth_m: float = Field(
        default=0.3,
        ge=0.05,
        le=1.0,
        description=(
            "Effective channel height (m), sets the invert offset above "
            "the pipe invert.  Matches typical curb heights (0.15-0.30 m) "
            "plus some overflow allowance so SWMM reports surcharge on the "
            "street rather than catastrophic flooding."
        ),
    )
    channel_roughness: float = Field(
        default=0.016,
        ge=0.010,
        le=0.050,
        description=(
            "Manning's n for the surface channel.  0.016 is typical for "
            "asphalt/concrete gutters; use 0.020+ if modeling rough "
            "pavement or frequent parked vehicles."
        ),
    )
    default_lanes: float = Field(
        default=2.0,
        ge=1.0,
        le=8.0,
        description=(
            "Fallback lane count when OSM edges lack a ``lanes`` attribute. "
            "Channel width = default_lanes * subcatchment_derivation.lane_width."
        ),
    )


class SimplificationParams(BaseModel):
    """Parameters for optional network simplification.

    When enabled, consolidates degree-2 pipe chains and removes small
    dangling leaves to reduce SWMM model element count.  Based on:

        Pichler et al. (2024), "Fully automated simplification of urban
        drainage models on a city scale", Water Science & Technology 90(9).
    """

    enabled: bool = Field(
        default=False,
        description="Enable network simplification. When False the step is a no-op.",
    )
    min_contributing_area_m2: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Dangling leaf chains whose total contributing area (m^2) "
            "is below this threshold are removed. Set 0 to skip leaf removal."
        ),
    )
    max_conduit_length: float = Field(
        default=500.0,
        ge=50.0,
        le=5000.0,
        description="Maximum merged conduit length (m). Longer chains are split.",
    )


class TrunkInference(BaseModel):
    """Parameters for synthetic trunk-pipe inference.

    OSM-derived pipe networks are typically fragmented (e.g. 26
    disconnected components on the test catchment, each subdivision is its own
    component because OSM doesn't include the trunk drainage that
    physically connects them to the master canal).  This step bridges
    those gaps with synthetic straight-line trunk pipes via a greedy
    Prim-style MST augmentation seeded at the canal's exit outfall.

    References:
        Haydar S, Chahinian N, Pasquier P, Wittner C (2019). Optimal
            urban sewer layout design using Steiner tree problems.
            Engineering Optimization 51(11):1980.
        Haydar S, Chahinian N, Pasquier P (2026). Reconstructing Sewer
            Network Topology Using Graph Theory. Water 18(2):222.
        Chegini T, Li H-Y (2022). An algorithm for deriving the topology
            of belowground urban stormwater networks. HESS 26:4279.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Enable trunk-pipe inference between disconnected pipe "
            "components.  When False the pipe network keeps whatever "
            "fragmentation ``derive_topology`` produces."
        ),
    )
    max_trunk_length_m: float = Field(
        default=2000.0,
        ge=0.0,
        le=20000.0,
        description=(
            "Maximum length (m) of any single synthetic trunk pipe.  "
            "Components whose nearest already-connected neighbor is "
            "farther than this stay disconnected and rely on closed-"
            "basin pond treatment downstream.  Default 2000 m allows "
            "neighborhood-scale shortcuts but rejects cross-bbox "
            "transcontinental trunks that would be unphysical."
        ),
    )
    placeholder_diameter_m: float = Field(
        default=0.30,
        ge=0.10,
        le=3.0,
        description=(
            "Initial trunk-pipe diameter (m) before ``pipe_by_pipe`` "
            "resizes based on accumulated contributing area.  Default "
            "0.30 m (12 in), the standard pipe-by-pipe minimum that "
            "the design step will grow as needed."
        ),
    )


class ParametersDict(TypedDict):
    """Type definition for the parameters dictionary."""

    subcatchment_derivation: SubcatchmentDerivation
    outfall_derivation: OutfallDerivation
    topology_derivation: TopologyDerivation
    trunk_inference: TrunkInference
    hydraulic_design: HydraulicDesign
    channel_design: ChannelDesign
    pond_design: PondDesign
    dual_drainage: DualDrainage
    simplification: SimplificationParams


def get_full_parameters() -> ParametersDict:
    """Get the full set of parameters."""
    return {
        "subcatchment_derivation": SubcatchmentDerivation(),
        "outfall_derivation": OutfallDerivation(),
        "topology_derivation": TopologyDerivation(),
        "trunk_inference": TrunkInference(),
        "hydraulic_design": HydraulicDesign(),
        "channel_design": ChannelDesign(),
        "pond_design": PondDesign(),
        "dual_drainage": DualDrainage(),
        "simplification": SimplificationParams(),
    }
