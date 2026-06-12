# SWMManywhere-US

Synthesize urban drainage network models (EPA SWMM) anywhere in the United States from
public national datasets.

Given only a bounding box, SWMManywhere-US downloads the required input data, derives a
plausible storm-drain network, sizes it hydraulically, and writes a ready-to-run SWMM
`.inp` model:

- **Elevation**: USGS 3DEP DEM (10 m, configurable) with water-body hydroflattening
- **Streets and rivers**: OpenStreetMap (direct Overpass queries)
- **Imperviousness / land cover**: USGS MRLC NLCD
- **Design rainfall**: NOAA Atlas 14 intensity-duration-frequency data
- **Rail barriers**: US Census TIGER

The generation pipeline derives flow topology with a turn-aware Dijkstra forest
(Chahinian et al. 2019 weights), sizes pipes with an improved Duque et al. (2022)
pipe-by-pipe method (diameter monotonicity, zero-adverse invert conditioning), and can
optionally add a dual-drainage surface network and a detention-pond subsystem
(FDOT-style orifice/weir outlet structures, pondshed delineation).

This project started as a US-focused adaptation of
[SWMManywhere](https://github.com/ImperialCollegeLondon/SWMManywhere) (Dobson et al.,
2025, *Environmental Modelling & Software*,
[doi:10.1016/j.envsoft.2025.106358](https://doi.org/10.1016/j.envsoft.2025.106358)) and
has since replaced the data acquisition, topology derivation, subcatchment delineation,
and hydraulic design stages.

> **Status**: pre-release. The API is functional but unstable, and the package is not
> yet published on PyPI.

## Installation

```bash
pip install git+https://github.com/cheginit/swmmanywhere-us.git
```

For development, the repository is managed with [pixi](https://pixi.sh):

```bash
git clone https://github.com/cheginit/swmmanywhere-us.git
cd swmmanywhere-us
pixi r -e test314 test   # run the test suite
pixi r lint              # pre-commit lint
pixi r typecheck         # pyright
```

## Quick start

See [docs/example.py](docs/example.py):

```python
from swmmanywhere_us import configure_logger, swmmanywhere

configure_logger(level="INFO")

config = {
    "base_dir": "data",  # downloads and models are written here
    "project": "demo",
    # A small residential area (EPSG:4326); processing extends it by buffer_km
    "bbox": {
        "xmin": -88.162,
        "ymin": 41.772,
        "xmax": -88.150,
        "ymax": 41.780,
        "buffer_km": 1,
    },
    # Optional subsystems:
    # "add_pondsheds": True,  # detention-pond subsystem (off by default)
    # "params_overrides": {"dual_drainage": {"enabled": False}},  # surface overlay (on by default)
}

inp_path = swmmanywhere(config)
print(f"SWMM model written to: {inp_path}")
```

The generated model lands under `<base_dir>/<project>/bbox_<N>/model_<M>/model_<M>.inp`
together with the intermediate graphs, subcatchments, and a GeoPackage of the network
for GIS inspection.

## License

[BSD 3-Clause](LICENSE)
