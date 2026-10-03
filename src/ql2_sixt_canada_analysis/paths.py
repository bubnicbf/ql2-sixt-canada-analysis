"""Authoritative repository and data-directory paths.

All paths derive from :data:`PROJECT_ROOT`, which is computed from this file's
location in the source checkout (``<root>/src/ql2_sixt_canada_analysis/``),
so it never depends on the current working directory. Importing this module
only constructs :class:`pathlib.Path` values: it does not touch the filesystem
or create directories.

These defaults assume the package runs from the source checkout (an editable
install or ``pythonpath = ["src"]``). Functions that read data accept an
explicit directory argument for other layouts and for tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = [
    "DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
]

PROJECT_ROOT: Final[Path] = Path(__file__).parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
