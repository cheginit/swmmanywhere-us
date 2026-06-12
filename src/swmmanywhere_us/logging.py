"""Logging utilities for swmmanywhere_us.

Example:
-------
>>> from swmmanywhere_us import configure_logger, logger
>>> configure_logger(verbose=True)          # console shows DEBUG+
>>> configure_logger(file="run.log")        # also log to file
>>> logger.info("Starting pipeline")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import IO, Literal

__all__ = [
    "configure_logger",
    "generate_log_path",
    "get_log_file_path",
    "logger",
]

logger = logging.getLogger("swmmanywhere_us")
logger.setLevel(logging.DEBUG)
logger.propagate = False

_file_handler: logging.FileHandler | None = None
_console_handler: logging.StreamHandler[IO[str]] | None = None

_handlers = logger.handlers
if not _handlers:
    _sh: logging.StreamHandler[IO[str]] = logging.StreamHandler(sys.stderr)
    _sh.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    _sh.setLevel(logging.WARNING)
    logger.addHandler(_sh)
    _console_handler = _sh


def generate_log_path(work_dir: Path, prefix: str = "swmmanywhere_us") -> Path:
    """Generate a timestamped log file path.

    Parameters
    ----------
    work_dir : Path
        Directory where the log file will be created.
    prefix : str, optional
        Prefix for the log file name. Defaults to ``"swmmanywhere_us"``.

    Returns
    -------
    Path
        Path to the log file (e.g., ``<work_dir>/swmmanywhere_us-20260206-140112.log``).
    """
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return work_dir / f"{prefix}-{timestamp}.log"


def get_log_file_path() -> Path | None:
    """Get the current log file path if file logging is enabled."""
    if _file_handler is not None:
        return Path(_file_handler.baseFilename)
    return None


def _validate_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int) -> int:
    if isinstance(level, str):
        level_upper = level.upper()
        if level_upper not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            msg = f"Invalid log level: {level!r}. Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
            raise ValueError(msg)
        return getattr(logging, level_upper)

    if level not in (
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ):
        msg = f"Invalid log level: {level!r}. Must be a valid logging level constant."
        raise ValueError(msg)
    return level


def configure_logger(  # noqa: C901 - branchy but linear flag/handler resolution
    *,
    verbose: bool | None = None,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int | None = None,
    file: str | Path | None = None,
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int | None = None,
    file_mode: Literal["a", "w"] = "a",
    file_only: bool = False,
) -> None:
    """Configure logging settings.

    Parameters
    ----------
    verbose : bool, optional
        Shortcut: ``True`` sets console to DEBUG, ``False`` to WARNING.
        If both ``level`` and ``verbose`` are given, ``level`` wins.
    level : str or int, optional
        Console logging level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
    file : str or Path, optional
        Enable file logging at this path. Pass ``None`` to disable file logging.
    file_level : str or int, optional
        File handler level. Defaults to ``DEBUG``.
    file_mode : {'a', 'w'}, optional
        Append or overwrite the log file. Defaults to ``'a'``.
    file_only : bool, optional
        If ``True``, disable console logging. Requires ``file`` to be set.
    """
    global _file_handler  # noqa: PLW0603

    if level is not None:
        level_int = _validate_level(level)
        if _console_handler is not None:
            _console_handler.setLevel(level_int)
    elif verbose is not None:
        console_level = logging.DEBUG if verbose else logging.WARNING
        if _console_handler is not None:
            _console_handler.setLevel(console_level)

    if file is not None:
        if _file_handler is not None:
            logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None

        file_level_int = _validate_level(file_level) if file_level is not None else logging.DEBUG

        filepath = Path(file)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if file_mode not in ("a", "w"):
            msg = f"Invalid file_mode: {file_mode!r}. Must be 'a' or 'w'."
            raise ValueError(msg)

        _file_handler = logging.FileHandler(filepath, mode=file_mode)
        _file_handler.setLevel(file_level_int)
        _file_handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%Y/%m/%d %H:%M:%S",
            )
        )
        logger.addHandler(_file_handler)

        if _file_handler.stream is not None:  # pyright: ignore[reportUnnecessaryComparison]
            _file_handler.stream.reconfigure(line_buffering=True)

        if file_only and _console_handler is not None:
            logger.removeHandler(_console_handler)

    elif _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None
