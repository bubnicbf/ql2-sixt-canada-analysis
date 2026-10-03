"""Authoritative repository and data-directory paths.

All paths derive from :data:`PROJECT_ROOT`, which is computed from this file's
location in the source checkout (``<root>/src/ql2_sixt_canada_analysis/``),
so it never depends on the current working directory. Importing this module
only constructs :class:`pathlib.Path` values: it does not touch the filesystem
or create directories.

These defaults assume the package runs from the source checkout (an editable
install or ``pythonpath = ["src"]``). Functions that read data accept an
explicit directory argument for other layouts and for tests.

Automated notebook validation cannot pass an argument into a notebook, so
:func:`resolve_raw_data_dir` also honours the :data:`RAW_DATA_DIR_ENV_VAR`
environment variable. Developers never need to set it; the test harness sets
it to a temporary directory of synthetic CSVs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

__all__ = [
    "DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "RAW_DATA_DIR_ENV_VAR",
    "resolve_raw_data_dir",
]

PROJECT_ROOT: Final[Path] = Path(__file__).parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

#: Environment variable that overrides the raw-data directory. Intended for
#: automated validation (notebook execution against synthetic inputs), not
#: for normal use.
RAW_DATA_DIR_ENV_VAR: Final[str] = "QL2_SIXT_RAW_DATA_DIR"


def resolve_raw_data_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the raw-data directory to use.

    Precedence: an explicit ``override`` argument, then a non-empty
    :data:`RAW_DATA_DIR_ENV_VAR` environment variable, then
    :data:`RAW_DATA_DIR`. The result is a :class:`pathlib.Path`; nothing is
    read, created or resolved on the filesystem.
    """
    if override is not None:
        return Path(override)
    from_env = os.environ.get(RAW_DATA_DIR_ENV_VAR, "")
    if from_env:
        return Path(from_env)
    return RAW_DATA_DIR
