"""Discover, validate and load the raw CSV datasets.

Paths come from :mod:`ql2_sixt_canada_analysis.paths` and dataset keys,
filename tokens and column contracts from
:mod:`ql2_sixt_canada_analysis.schemas`; this module defines none of them.

The loader performs ingestion only: each file's header is checked against its
column contract, then the file is read once with :func:`pandas.read_csv` and
returned unchanged. It never cleans, renames, coerces, deduplicates, or writes
data. Importing this module performs no filesystem access.

Filename discovery
------------------
Only regular ``*.csv`` files directly inside the raw-data directory are
considered (case-insensitive extension, no recursion). Each file stem is split
into lowercase alphanumeric tokens, and the file belongs to the dataset whose
filename token appears **last**. This tolerates arbitrary prefixes, numeric
ranges, spaces, parentheses and download suffixes, including a shared prefix
that mentions another dataset. Files with no dataset token are ignored.
Missing or ambiguous datasets raise an error instead of guessing.

Header validation
-----------------
The header row is read with :mod:`csv` (no data rows) so duplicate column
names are visible before pandas renames them. It must equal the dataset's
``columns`` exactly, including order.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from ql2_sixt_canada_analysis.paths import RAW_DATA_DIR
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    DatasetDefinition,
    DatasetKey,
)

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
    "SourceSchemaError",
    "discover_raw_csvs",
    "load_raw_datasets",
    "read_csv_header",
    "validate_header",
]

_TOKEN_SPLIT: Final = re.compile(r"[^a-z0-9]+")
# read_csv options that would stop the loader returning one complete
# DataFrame whose columns are exactly the validated header.
_FORBIDDEN_READ_OPTIONS: Final = frozenset(
    {
        "filepath_or_buffer", "chunksize", "iterator",
        "header", "names", "usecols", "index_col", "skiprows", "comment",
    }
)

StrPath = str | os.PathLike[str]


# --------------------------------------------------------------------- exceptions


class IngestionError(Exception):
    """Base class for raw-data discovery, validation and loading failures."""


class RawDataDiscoveryError(IngestionError):
    """The raw-data directory or its CSV files do not meet expectations."""


class RawDataDirectoryError(RawDataDiscoveryError):
    """The raw-data directory is missing or is not a directory."""

    def __init__(self, message: str, path: Path) -> None:
        super().__init__(message)
        self.path = path


class DatasetNotFoundError(RawDataDiscoveryError):
    """No CSV file was found for a logical dataset."""

    def __init__(self, key: DatasetKey) -> None:
        super().__init__(f"No raw CSV found for the '{key}' dataset.")
        self.role = key


class AmbiguousDatasetError(RawDataDiscoveryError):
    """More than one CSV file matched a logical dataset."""

    def __init__(self, key: DatasetKey, candidates: list[Path]) -> None:
        super().__init__(
            f"Found {len(candidates)} candidate CSVs for the '{key}' dataset; "
            "expected exactly one."
        )
        self.role = key
        self.candidates = candidates


class NotARegularFileError(RawDataDiscoveryError):
    """A path matched a logical dataset but is not a regular file."""

    def __init__(self, key: DatasetKey, path: Path) -> None:
        super().__init__(f"The '{key}' CSV candidate is not a regular file.")
        self.role = key
        self.path = path


class RawDataLoadError(IngestionError):
    """A discovered CSV could not be read. Any original error is the cause."""

    def __init__(self, key: DatasetKey, path: Path, reason: str = "could not be read") -> None:
        super().__init__(f"The '{key}' raw CSV {reason}.")
        self.role = key
        self.path = path


class SourceSchemaError(IngestionError):
    """A CSV header does not match its dataset's column contract.

    The message reports counts only; the offending column names are available
    on the ``missing``, ``unexpected`` and ``duplicated`` attributes.
    """

    def __init__(
        self,
        key: DatasetKey,
        *,
        missing: tuple[str, ...],
        unexpected: tuple[str, ...],
        duplicated: tuple[str, ...],
        order_mismatch: bool,
    ) -> None:
        problems = []
        if missing:
            problems.append(f"{len(missing)} missing")
        if unexpected:
            problems.append(f"{len(unexpected)} unexpected")
        if duplicated:
            problems.append(f"{len(duplicated)} duplicated")
        if order_mismatch:
            problems.append("columns out of order")
        super().__init__(
            f"The '{key}' raw CSV header does not match its column contract: "
            + ", ".join(problems) + "."
        )
        self.role = key
        self.missing = missing
        self.unexpected = unexpected
        self.duplicated = duplicated
        self.order_mismatch = order_mismatch


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


def discover_raw_csvs(raw_dir: StrPath | None = None) -> RawDatasetPaths:
    """Find exactly one CSV per logical dataset in ``raw_dir``.

    Args:
        raw_dir: Directory containing the raw CSVs. Defaults to
            :data:`ql2_sixt_canada_analysis.paths.RAW_DATA_DIR`.

    Raises:
        RawDataDirectoryError: ``raw_dir`` is missing or not a directory.
        DatasetNotFoundError: No CSV matches a dataset.
        AmbiguousDatasetError: More than one CSV matches a dataset.
        NotARegularFileError: A matching path is not a regular file.
    """
    directory = RAW_DATA_DIR if raw_dir is None else Path(raw_dir)
    if not directory.exists():
        raise RawDataDirectoryError("The raw-data directory does not exist.", directory)
    if not directory.is_dir():
        raise RawDataDirectoryError("The raw-data path is not a directory.", directory)

    candidates: dict[DatasetKey, list[Path]] = {key: [] for key in DATASET_DEFINITIONS}
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() != ".csv":
            continue
        key = _classify(entry.stem)
        if key is not None:
            candidates[key].append(entry)

    selected: dict[DatasetKey, Path] = {}
    for key, matches in candidates.items():
        if not matches:
            raise DatasetNotFoundError(key)
        if len(matches) > 1:
            raise AmbiguousDatasetError(key, matches)
        if not matches[0].is_file():
            raise NotARegularFileError(key, matches[0])
        selected[key] = matches[0]
    return RawDatasetPaths(jobs=selected[DatasetKey.JOBS], cars=selected[DatasetKey.CARS])


def load_raw_datasets(
    raw_dir: StrPath | None = None,
    *,
    read_csv_options: Mapping[str, Any] | None = None,
) -> RawDatasets:
    """Discover, header-validate and load the ``jobs`` and ``cars`` raw CSVs.

    Each header is validated against its :class:`DatasetDefinition` before
    the file is read once with :func:`pandas.read_csv` using pandas defaults
    plus any ``read_csv_options`` (applied to both files, e.g. ``dtype=str``
    or ``low_memory=False``). No cleaning or type coercion is applied and
    nothing is written to disk.

    Args:
        raw_dir: Directory containing the raw CSVs. Defaults to
            :data:`ql2_sixt_canada_analysis.paths.RAW_DATA_DIR`.
        read_csv_options: Extra keyword arguments for :func:`pandas.read_csv`.
            Options that change which rows or columns form the header or
            result (``header``, ``names``, ``usecols``, ``index_col``,
            ``skiprows``, ``comment``, ``chunksize``, ``iterator``,
            ``filepath_or_buffer``) are rejected.

    Raises:
        RawDataDiscoveryError: Discovery failed (see :func:`discover_raw_csvs`).
        SourceSchemaError: A header does not match its column contract.
        RawDataLoadError: A file could not be read; the original exception
            is available as ``__cause__``.
        ValueError: ``read_csv_options`` contains a rejected option.
    """
    options = dict(read_csv_options or {})
    forbidden = sorted(_FORBIDDEN_READ_OPTIONS.intersection(options))
    if forbidden:
        raise ValueError(f"Unsupported read_csv option(s): {', '.join(forbidden)}")

    paths = discover_raw_csvs(raw_dir)
    return RawDatasets(
        jobs=_load(DATASET_DEFINITIONS[DatasetKey.JOBS], paths.jobs, options),
        cars=_load(DATASET_DEFINITIONS[DatasetKey.CARS], paths.cars, options),
    )


def read_csv_header(
    path: StrPath, key: DatasetKey, *, encoding: str | None = None, sep: str = ","
) -> tuple[str, ...]:
    """Read only the header row of a CSV, without reading any data rows.

    Raises:
        RawDataLoadError: The file cannot be opened or decoded, or is empty.
    """
    path = Path(path)
    # pandas strips a UTF-8 byte-order mark by default; match that behaviour.
    if encoding is None or encoding.lower().replace("_", "-") in {"utf-8", "utf8"}:
        encoding = "utf-8-sig"
    try:
        with path.open(newline="", encoding=encoding) as handle:
            header = next(csv.reader(handle, delimiter=sep), None)
    except (OSError, ValueError, csv.Error) as exc:  # decode errors are ValueErrors
        raise RawDataLoadError(key, path) from exc
    if not header:
        raise RawDataLoadError(key, path, "has no header row")
    return tuple(header)


def validate_header(definition: DatasetDefinition, header: tuple[str, ...]) -> None:
    """Check ``header`` equals ``definition.columns`` exactly, including order.

    Raises:
        SourceSchemaError: Columns are missing, unexpected, duplicated, or
            out of order.
    """
    if header == definition.columns:
        return
    expected, actual = set(definition.columns), set(header)
    raise SourceSchemaError(
        definition.key,
        missing=tuple(c for c in definition.columns if c not in actual),
        unexpected=tuple(dict.fromkeys(c for c in header if c not in expected)),
        duplicated=tuple(dict.fromkeys(c for c in header if header.count(c) > 1)),
        order_mismatch=len(header) == len(definition.columns) and actual == expected,
    )


# ---------------------------------------------------------------------- helpers


def _classify(stem: str) -> DatasetKey | None:
    """Return the dataset whose filename token appears last in ``stem``."""
    token_to_key = {
        token: definition.key
        for definition in DATASET_DEFINITIONS.values()
        for token in definition.filename_tokens
    }
    matches = [token_to_key[t] for t in _TOKEN_SPLIT.split(stem.lower()) if t in token_to_key]
    return matches[-1] if matches else None


def _load(definition: DatasetDefinition, path: Path, options: Mapping[str, Any]) -> pd.DataFrame:
    """Validate one file's header, then read it once with pandas."""
    sep = options.get("sep", options.get("delimiter", ","))
    if not isinstance(sep, str) or len(sep) != 1:
        raise ValueError("read_csv_options 'sep' must be a single character")
    header = read_csv_header(path, definition.key, encoding=options.get("encoding"), sep=sep)
    validate_header(definition, header)
    try:
        frame = pd.read_csv(path, **options)
    except (OSError, ValueError) as exc:  # pandas parser/empty/decode errors are ValueErrors
        raise RawDataLoadError(definition.key, path) from exc
    validate_header(definition, tuple(frame.columns))
    return frame
