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
The header row is read by :func:`pandas.read_csv` itself (``header=None``,
``nrows=1``, all values as text) with the caller's tokenizer options, so it is
split exactly as the full read will split it, while duplicate column names
stay visible (pandas only renames duplicates when it builds a header). It
must equal the dataset's ``columns`` exactly, including order.

``read_csv_options`` are classified explicitly: tokenizer options are applied
to both the header read and the full read, value-interpretation options only
to the full read, options that reshape the header or result are rejected, and
any option not listed here is rejected rather than risk the two reads
disagreeing.
"""

from __future__ import annotations

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
# Options that change how text is decoded or split into fields. They are
# forwarded to the header read as well as the full read so both tokenize the
# file identically.
_TOKENIZER_READ_OPTIONS: Final = frozenset(
    {
        "sep", "delimiter", "delim_whitespace", "engine", "dialect",
        "quotechar", "quoting", "doublequote", "escapechar", "skipinitialspace",
        "lineterminator", "skip_blank_lines", "on_bad_lines",
        "encoding", "encoding_errors", "compression", "storage_options", "memory_map",
    }
)
# Options that only interpret or limit data values. They cannot affect the
# header row, so they are applied to the full read only.
_DATA_READ_OPTIONS: Final = frozenset(
    {
        "dtype", "converters", "true_values", "false_values",
        "na_values", "keep_default_na", "na_filter",
        "parse_dates", "date_format", "dayfirst", "cache_dates",
        "keep_date_col", "date_parser", "infer_datetime_format",
        "thousands", "decimal", "float_precision", "dtype_backend",
        "low_memory", "nrows", "skipfooter", "verbose",
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
            Tokenizer options (e.g. ``sep``, ``quotechar``, ``escapechar``,
            ``doublequote``, ``skipinitialspace``, ``dialect``, ``encoding``,
            ``compression``) also govern the header check; value options
            (e.g. ``dtype``, ``na_values``) apply to the full read only.
            Options that change which rows or columns form the header or
            result (``header``, ``names``, ``usecols``, ``index_col``,
            ``skiprows``, ``comment``, ``chunksize``, ``iterator``,
            ``filepath_or_buffer``) and unrecognised options are rejected.

    Raises:
        RawDataDiscoveryError: Discovery failed (see :func:`discover_raw_csvs`).
        SourceSchemaError: A header does not match its column contract.
        RawDataLoadError: A file could not be read; the original exception
            is available as ``__cause__``.
        ValueError: ``read_csv_options`` contains a rejected or unrecognised
            option.
    """
    options = dict(read_csv_options or {})
    _check_read_options(options)

    paths = discover_raw_csvs(raw_dir)
    return RawDatasets(
        jobs=_load(DATASET_DEFINITIONS[DatasetKey.JOBS], paths.jobs, options),
        cars=_load(DATASET_DEFINITIONS[DatasetKey.CARS], paths.cars, options),
    )


def read_csv_header(
    path: StrPath,
    key: DatasetKey,
    *,
    read_csv_options: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Read only the header row of a CSV, tokenized as pandas will tokenize it.

    Uses :func:`pandas.read_csv` with ``header=None`` and ``nrows=1`` plus the
    tokenizer subset of ``read_csv_options``, returning the raw header cells
    as strings (duplicates preserved, no renaming).

    Raises:
        RawDataLoadError: The file cannot be opened, decoded or parsed, or has
            no header row; the original exception is the cause.
        ValueError: ``read_csv_options`` contains a rejected or unrecognised
            option.
    """
    path = Path(path)
    options = dict(read_csv_options or {})
    _check_read_options(options)
    header_options = {k: v for k, v in options.items() if k in _TOKENIZER_READ_OPTIONS}
    if header_options.get("engine") == "pyarrow":
        # The pyarrow engine does not support nrows. Its supported dialect
        # options are a subset of the C engine's, which tokenizes them alike.
        header_options["engine"] = "c"
    try:
        frame = pd.read_csv(
            path, header=None, nrows=1, dtype=str, na_filter=False, **header_options
        )
    except (OSError, ValueError) as exc:  # parser/empty/decode errors are ValueErrors
        raise RawDataLoadError(key, path) from exc
    if frame.empty:
        raise RawDataLoadError(key, path, "has no header row")
    return tuple(str(cell) for cell in frame.iloc[0].tolist())


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


def _check_read_options(options: Mapping[str, Any]) -> None:
    """Reject options that reshape the result or are not classified above."""
    forbidden = sorted(_FORBIDDEN_READ_OPTIONS.intersection(options))
    if forbidden:
        raise ValueError(f"Unsupported read_csv option(s): {', '.join(forbidden)}")
    known = _TOKENIZER_READ_OPTIONS | _DATA_READ_OPTIONS
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(
            f"Unrecognised read_csv option(s): {', '.join(unknown)}; the header "
            "check cannot guarantee it tokenizes the file like the full read"
        )


def _load(definition: DatasetDefinition, path: Path, options: Mapping[str, Any]) -> pd.DataFrame:
    """Validate one file's header, then read it fully once with pandas."""
    header = read_csv_header(path, definition.key, read_csv_options=options)
    validate_header(definition, header)
    try:
        frame = pd.read_csv(path, **options)
    except (OSError, ValueError) as exc:  # pandas parser/empty/decode errors are ValueErrors
        raise RawDataLoadError(definition.key, path) from exc
    validate_header(definition, tuple(frame.columns))
    return frame
