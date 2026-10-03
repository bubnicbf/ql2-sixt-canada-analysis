"""Discover and load the two raw CSV datasets (``jobs`` and ``cars``).

The loader performs ingestion only: it reads each file once with
:func:`pandas.read_csv` and returns the DataFrames unchanged. It never cleans,
renames, coerces, deduplicates, or writes data. Importing this module performs
no filesystem access.

Filename discovery
------------------
Only regular ``*.csv`` files directly inside the raw-data directory are
considered (case-insensitive extension, no recursion). Each file stem is split
into lowercase alphanumeric tokens, and the file's role is the **last** token
equal to ``jobs`` or ``cars``. Using the last token tolerates arbitrary
prefixes, numeric ranges, spaces, parentheses and download suffixes, including
a shared prefix that itself mentions the other role. Files with no role token
are ignored. Missing or ambiguous roles raise an error instead of guessing.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

__all__ = [
    "AmbiguousDatasetError",
    "DatasetNotFoundError",
    "IngestionError",
    "NotARegularFileError",
    "RawDataDirectoryError",
    "RawDataDiscoveryError",
    "RawDataLoadError",
    "RawDatasetPaths",
    "RawDatasets",
    "default_raw_dir",
    "discover_raw_csvs",
    "load_raw_datasets",
]

Role = Literal["jobs", "cars"]
ROLES: Final[tuple[Role, ...]] = ("jobs", "cars")

_TOKEN_SPLIT: Final = re.compile(r"[^a-z0-9]+")
# read_csv options that would stop it returning one complete DataFrame
# from the discovered file.
_FORBIDDEN_READ_OPTIONS: Final = frozenset({"filepath_or_buffer", "chunksize", "iterator"})

StrPath = str | os.PathLike[str]


# --------------------------------------------------------------------- exceptions


class IngestionError(Exception):
    """Base class for raw-data discovery and loading failures."""


class RawDataDiscoveryError(IngestionError):
    """The raw-data directory or its CSV files do not meet expectations."""


class RawDataDirectoryError(RawDataDiscoveryError):
    """The raw-data directory is missing or is not a directory."""

    def __init__(self, message: str, path: Path) -> None:
        super().__init__(message)
        self.path = path


class DatasetNotFoundError(RawDataDiscoveryError):
    """No CSV file was found for a logical dataset role."""

    def __init__(self, role: Role) -> None:
        super().__init__(f"No raw CSV found for the '{role}' dataset.")
        self.role = role


class AmbiguousDatasetError(RawDataDiscoveryError):
    """More than one CSV file matched a logical dataset role."""

    def __init__(self, role: Role, candidates: list[Path]) -> None:
        super().__init__(
            f"Found {len(candidates)} candidate CSVs for the '{role}' dataset; "
            "expected exactly one."
        )
        self.role = role
        self.candidates = candidates


class NotARegularFileError(RawDataDiscoveryError):
    """A path matched a dataset role but is not a regular file."""

    def __init__(self, role: Role, path: Path) -> None:
        super().__init__(f"The '{role}' CSV candidate is not a regular file.")
        self.role = role
        self.path = path


class RawDataLoadError(IngestionError):
    """pandas could not read a discovered CSV. The original error is the cause."""

    def __init__(self, role: Role, path: Path) -> None:
        super().__init__(f"Failed to load the '{role}' raw CSV with pandas.")
        self.role = role
        self.path = path


# ----------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class RawDatasetPaths:
    """Discovered file paths for each logical raw dataset."""

    jobs: Path
    cars: Path


@dataclass(frozen=True, slots=True)
class RawDatasets:
    """Raw DataFrames for each logical dataset, exactly as read by pandas."""

    jobs: pd.DataFrame
    cars: pd.DataFrame


# ------------------------------------------------------------------- public API


def default_raw_dir() -> Path:
    """Return ``data/raw`` of the source checkout containing this package.

    Resolved from this file's location (``<root>/src/<package>/``), so it does
    not depend on the current working directory. Raises
    :class:`RawDataDirectoryError` if the package is not running from a
    source checkout (for example a non-editable install); pass ``raw_dir``
    explicitly in that case.
    """
    project_root = Path(__file__).resolve().parents[2]
    if not (project_root / "pyproject.toml").is_file():
        raise RawDataDirectoryError(
            "Cannot locate the project root from the installed package; "
            "pass raw_dir explicitly.",
            project_root,
        )
    return project_root / "data" / "raw"


def discover_raw_csvs(raw_dir: StrPath | None = None) -> RawDatasetPaths:
    """Find exactly one ``jobs`` CSV and one ``cars`` CSV in ``raw_dir``.

    Args:
        raw_dir: Directory containing the raw CSVs. Defaults to
            :func:`default_raw_dir`.

    Raises:
        RawDataDirectoryError: ``raw_dir`` is missing or not a directory.
        DatasetNotFoundError: No CSV matches a role.
        AmbiguousDatasetError: More than one CSV matches a role.
        NotARegularFileError: A matching path is not a regular file.
    """
    directory = default_raw_dir() if raw_dir is None else Path(raw_dir)
    if not directory.exists():
        raise RawDataDirectoryError("The raw-data directory does not exist.", directory)
    if not directory.is_dir():
        raise RawDataDirectoryError("The raw-data path is not a directory.", directory)

    candidates: dict[Role, list[Path]] = {role: [] for role in ROLES}
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() != ".csv":
            continue
        role = _classify(entry.stem)
        if role is not None:
            candidates[role].append(entry)

    selected: dict[Role, Path] = {}
    for role in ROLES:
        matches = candidates[role]
        if not matches:
            raise DatasetNotFoundError(role)
        if len(matches) > 1:
            raise AmbiguousDatasetError(role, matches)
        if not matches[0].is_file():
            raise NotARegularFileError(role, matches[0])
        selected[role] = matches[0]
    return RawDatasetPaths(jobs=selected["jobs"], cars=selected["cars"])


def load_raw_datasets(
    raw_dir: StrPath | None = None,
    *,
    read_csv_options: Mapping[str, Any] | None = None,
) -> RawDatasets:
    """Discover and load the ``jobs`` and ``cars`` raw CSVs.

    Each file is read once with :func:`pandas.read_csv` using pandas defaults
    plus any ``read_csv_options`` (applied to both files, e.g. ``dtype=str``
    or ``low_memory=False``). No cleaning or type coercion is applied and
    nothing is written to disk.

    Args:
        raw_dir: Directory containing the raw CSVs. Defaults to
            :func:`default_raw_dir`.
        read_csv_options: Extra keyword arguments for :func:`pandas.read_csv`.
            ``filepath_or_buffer``, ``chunksize`` and ``iterator`` are not
            allowed because the loader must return complete DataFrames.

    Raises:
        RawDataDiscoveryError: Discovery failed (see :func:`discover_raw_csvs`).
        RawDataLoadError: pandas could not read a file; the original exception
            is available as ``__cause__``.
        ValueError: ``read_csv_options`` contains a forbidden option.
    """
    options = dict(read_csv_options or {})
    forbidden = sorted(_FORBIDDEN_READ_OPTIONS.intersection(options))
    if forbidden:
        raise ValueError(f"Unsupported read_csv option(s): {', '.join(forbidden)}")

    paths = discover_raw_csvs(raw_dir)
    return RawDatasets(
        jobs=_read_csv("jobs", paths.jobs, options),
        cars=_read_csv("cars", paths.cars, options),
    )


# ---------------------------------------------------------------------- helpers


def _classify(stem: str) -> Role | None:
    """Return the role named by the last ``jobs``/``cars`` token in ``stem``."""
    role_tokens = [t for t in _TOKEN_SPLIT.split(stem.lower()) if t in ROLES]
    return role_tokens[-1] if role_tokens else None  # type: ignore[return-value]


def _read_csv(role: Role, path: Path, options: Mapping[str, Any]) -> pd.DataFrame:
    """Single entry point for reading a raw CSV with pandas."""
    try:
        return pd.read_csv(path, **options)
    except (OSError, ValueError) as exc:  # pandas parser/empty/decode errors are ValueErrors
        raise RawDataLoadError(role, path) from exc
