"""Utilities for YAML save/load."""

from __future__ import annotations

import functools
from pathlib import Path, PosixPath, WindowsPath
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from yaml import ScalarNode


class PathDumper(getattr(yaml, "CSafeDumper", yaml.SafeDumper)):  # pyright: ignore[reportGeneralTypeIssues, reportUntypedBaseClass]
    """Create a custom YAML dumper that handles Path objects."""


def _path_representer(dumper: PathDumper, data: Path | PosixPath | WindowsPath) -> ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.as_posix())


PathDumper.add_representer(Path, _path_representer)
PathDumper.add_representer(PosixPath, _path_representer)
PathDumper.add_representer(WindowsPath, _path_representer)

yaml_load = functools.partial(yaml.load, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))

yaml_dump = functools.partial(
    yaml.dump,
    Dumper=PathDumper,
    default_flow_style=False,
    indent=2,
    sort_keys=False,
)
