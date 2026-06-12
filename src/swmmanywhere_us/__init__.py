"""SWMManywhere-US: Generate synthetic SWMM networks for US locations."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swmmanywhere_us.logging import configure_logger
    from swmmanywhere_us.swmmanywhere import swmmanywhere

try:
    __version__ = version("swmmanywhere_us")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["configure_logger", "swmmanywhere"]

# ---------------------------------------------------------------------------
# Lazy public API: heavy imports are deferred until first access.  The
# TYPE_CHECKING block above gives pyright/mypy full visibility without
# executing any imports at runtime.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "configure_logger": ("swmmanywhere_us.logging", "configure_logger"),
    "swmmanywhere": ("swmmanywhere_us.swmmanywhere", "swmmanywhere"),
}


__all__ = [
    "__version__",
    "configure_logger",
    "swmmanywhere",
]


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        val = mod if attr is None else getattr(mod, attr)
        globals()[name] = val
        return val
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# ---------------------------------------------------------------------------
# Eager-import override: set EAGER_IMPORT=1 (any non-"0"/non-empty value) to
# load all lazy members immediately.  Useful in CI and for profiling.
# ---------------------------------------------------------------------------
if os.environ.get("EAGER_IMPORT", "") not in ("", "0"):
    for _name in _LAZY_IMPORTS:
        __getattr__(_name)
