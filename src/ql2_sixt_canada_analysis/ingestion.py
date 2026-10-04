"""Discover, validate and load the raw CSV datasets.

Paths come from :mod:`ql2_sixt_canada_analysis.paths` and dataset keys,
filename tokens and column contracts from
:mod:`ql2_sixt_canada_analysis.schemas`; this module defines none of them.

The loader performs ingestion only: each file's header is checked against its
column contract, then the file is read once with :func:`pandas.read_csv` and
returned unchanged. It never cleans, renames, coerces, deduplicates, or writes
data. Importing this module performs no filesystem access.

Blank physical lines
--------------------
:func:`pandas.read_csv` silently drops completely blank lines by default
(``skip_blank_lines=True``), which would make them impossible to audit. The
raw loader therefore reads with the project policy :data:`RAW_CSV_READ_DEFAULTS`
(``skip_blank_lines=False``): every physical line after the header becomes a
DataFrame row, so an empty line, a delimiter-only line and a whitespace-only
line all arrive as rows (all-missing, all-missing, and a first field holding
the whitespace with the rest missing, respectively) for
:mod:`ql2_sixt_canada_analysis.quality` to classify, remove and count. The
loader itself removes nothing. Two consequences are intentional:

* A file whose header is preceded by blank lines is rejected
  (:class:`RawDataLoadError`), because with blank lines preserved the first
  physical line *is* the header. The raw files start with their header.
* Each blank line that follows the final record, if any, is also a row.

The policy lives here only. ``read_csv_options`` may not contain
``skip_blank_lines``; the single documented way to opt out is
``preserve_blank_lines=False`` on :func:`load_raw_datasets` /
:func:`read_csv_header`, after which blank-row counts are not meaningful.

Identifier columns
------------------
Each dataset's ``identifier_columns`` (from its :class:`DatasetDefinition`)
are read with ``dtype=IDENTIFIER_DTYPE`` (pandas nullable string) in the same
single :func:`pandas.read_csv` call, so they are never parsed as numbers:
leading zeros, long digit strings and ``"0"`` survive exactly, and empty
fields stay ``pd.NA``. Other columns keep normal pandas inference. After the
read, :func:`~ql2_sixt_canada_analysis.identifiers.validate_identifier_dtypes`
confirms the guarantee held.

Caller options are merged per dataset, without mutating the caller's objects:

* ``dtype`` mapping: entries for non-identifier columns are kept; an entry for
  an identifier column is accepted only if it is a nullable string dtype (it
  is replaced by the project dtype); anything else raises
  :class:`~ql2_sixt_canada_analysis.identifiers.IdentifierTypeConflictError`.
* ``dtype`` scalar: a string-like scalar (``str``, ``object``, ``"string"``)
  is an explicit request to read every column as text; it is applied to the
  non-identifier columns and identifiers still get the project dtype. A
  non-string scalar (e.g. ``int``) would retype identifiers and is rejected.
* ``converters`` or ``parse_dates`` naming an identifier column (by name or
  position) are rejected; for other columns they pass through unchanged.
* ``engine="pyarrow"`` is rejected for datasets with identifiers: that engine
  infers types first and casts to the requested dtype afterwards, which
  silently drops leading zeros even though the result is a string column.
  The default C engine and the python engine honour the mapping while
  parsing. (``dtype_backend="pyarrow"`` with those engines is fine.)

Conflicts are detected for both datasets before any file is read.

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

Complete-source policy
----------------------
The loader returns **every** record of each file or fails. Options that can
sample, skip, filter, project or stream part of a source are rejected before
any file is opened with :class:`IncompleteSourceOptionError` (naming the
option and the reason), never silently dropped: ``nrows`` (row limit),
``skiprows`` (leading/selected rows), ``skipfooter`` (trailing rows),
``comment`` (truncates lines and drops comment-only lines), ``chunksize`` /
``iterator`` (partial reads unless every chunk is consumed), ``usecols``
(column projection), ``header`` / ``names`` / ``index_col`` (reshape the
validated header) and ``on_bad_lines`` other than ``"error"`` (``"skip"``,
``"warn"`` or a callable discard malformed records; ``"error"`` - the parser
default - is allowed, so malformed rows fail fast). Remaining options only
change tokenisation or value interpretation and cannot remove records.
:class:`RawDatasets` from :func:`load_raw_datasets` carries
``complete_source=True`` only when this policy held *and* blank physical
lines were preserved.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import pandas as pd

from ql2_sixt_canada_analysis.identifiers import (
    IdentifierTypeConflictError,
    is_identifier_dtype,
    validate_identifier_dtypes,
)
from ql2_sixt_canada_analysis.paths import RAW_DATA_DIR
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    DatasetDefinition,
    DatasetKey,
)

__all__ = [
    "RAW_CSV_READ_DEFAULTS",
    "SAFE_ON_BAD_LINES",
    "IncompleteSourceOptionError",
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
#: Authoritative :func:`pandas.read_csv` settings for every raw CSV read (header
#: check and full load alike). ``skip_blank_lines=False`` keeps blank physical
#: lines as rows so the blank-row quality step can measure them. Defined once;
#: other modules must not restate it. Callers cannot override these through
#: ``read_csv_options``; see ``preserve_blank_lines``.
RAW_CSV_READ_DEFAULTS: Final[Mapping[str, Any]] = MappingProxyType({"skip_blank_lines": False})
# read_csv options governed by project policy rather than by callers.
_POLICY_READ_OPTIONS: Final = frozenset(RAW_CSV_READ_DEFAULTS)
# Complete-source policy: read_csv options that would stop the loader
# returning one complete DataFrame of every source record whose columns are
# exactly the validated header, with the reason reported to the caller.
_INCOMPLETE_SOURCE_OPTIONS: Final[Mapping[str, str]] = MappingProxyType({
    "filepath_or_buffer": "the source path is chosen by discovery",
    "nrows": "limits the number of records read",
    "skiprows": "omits source records",
    "skipfooter": "omits trailing source records",
    "comment": "truncates lines and drops comment-only records",
    "chunksize": "returns a partial, chunked reader instead of every record",
    "iterator": "returns an iterator that may be consumed partially",
    "usecols": "projects away contract columns",
    "header": "reshapes the validated header",
    "names": "replaces the validated header",
    "index_col": "moves contract columns into the index",
})
_FORBIDDEN_READ_OPTIONS: Final = frozenset(_INCOMPLETE_SOURCE_OPTIONS)
#: The only accepted ``on_bad_lines`` value: malformed records fail the load.
SAFE_ON_BAD_LINES: Final = "error"
# Options that change how text is decoded or split into fields. They are
# forwarded to the header read as well as the full read so both tokenize the
# file identically.
_TOKENIZER_READ_OPTIONS: Final = frozenset(
    {
        "sep", "delimiter", "delim_whitespace", "engine", "dialect",
        "quotechar", "quoting", "doublequote", "escapechar", "skipinitialspace",
        "lineterminator", "on_bad_lines",
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
        "low_memory", "verbose",
    }
)

StrPath = str | os.PathLike[str]


# --------------------------------------------------------------------- exceptions


class IngestionError(Exception):
    """Base class for raw-data discovery, validation and loading failures."""


class IncompleteSourceOptionError(IngestionError, ValueError):
    """A ``read_csv`` option could omit source records (complete-source policy).

    ``option`` names the prohibited option; ``reason`` says why.
    """

    def __init__(self, option: str, reason: str) -> None:
        super().__init__(f"read_csv option '{option}' is not allowed: it {reason} "
                         "(the raw loader must return every source record).")
        self.option = option
        self.reason = reason


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
    """Raw DataFrames for each logical dataset, exactly as read by pandas.

    Identifier columns are nullable strings (see the module docstring); all
    other columns are as pandas inferred them.
    """

    jobs: pd.DataFrame
    cars: pd.DataFrame
    #: True only when :func:`load_raw_datasets` read every source record under
    #: the complete-source policy with blank lines preserved. False (not
    #: proven) for frames assembled any other way.
    complete_source: bool = False


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
    preserve_blank_lines: bool = True,
) -> RawDatasets:
    """Discover, header-validate and load the ``jobs`` and ``cars`` raw CSVs.

    Each header is validated against its :class:`DatasetDefinition` before
    the file is read once with :func:`pandas.read_csv` using pandas defaults,
    the project policy :data:`RAW_CSV_READ_DEFAULTS` (blank physical lines are
    kept as rows) and any ``read_csv_options`` (applied to both files, e.g.
    ``dtype=str`` or ``low_memory=False``). No cleaning or type coercion is
    applied and nothing is written to disk; completely blank rows are removed
    and counted later by :mod:`ql2_sixt_canada_analysis.quality`.
    Each dataset's identifier columns are read as the nullable string dtype
    (see "Identifier columns" in the module docstring for the merge rules).

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
            ``filepath_or_buffer``), the policy option ``skip_blank_lines``
            and unrecognised options are rejected.
        preserve_blank_lines: ``True`` (the project policy) keeps every blank
            physical line as an all-missing row so it can be audited. Pass
            ``False`` only as a deliberate, documented decision to let pandas
            drop blank lines; blank-row counts are then not meaningful.

    Raises:
        RawDataDiscoveryError: Discovery failed (see :func:`discover_raw_csvs`).
        SourceSchemaError: A header does not match its column contract.
        RawDataLoadError: A file could not be read; the original exception
            is available as ``__cause__``.
        IdentifierTypeConflictError: ``read_csv_options`` would parse an
            identifier column unsafely (a ``ValueError`` subclass).
        IdentifierDtypeError: An identifier column was not read as the
            nullable string dtype (e.g. an engine ignored the mapping).
        ValueError: ``read_csv_options`` contains a rejected, policy-governed
            or unrecognised option.
    """
    options = _resolve_read_options(read_csv_options, preserve_blank_lines)
    per_dataset = {
        key: _with_identifier_dtypes(definition, options)
        for key, definition in DATASET_DEFINITIONS.items()
    }

    paths = discover_raw_csvs(raw_dir)
    return RawDatasets(
        jobs=_load(DATASET_DEFINITIONS[DatasetKey.JOBS], paths.jobs, per_dataset[DatasetKey.JOBS]),
        cars=_load(DATASET_DEFINITIONS[DatasetKey.CARS], paths.cars, per_dataset[DatasetKey.CARS]),
        complete_source=bool(preserve_blank_lines),
    )


def read_csv_header(
    path: StrPath,
    key: DatasetKey,
    *,
    read_csv_options: Mapping[str, Any] | None = None,
    preserve_blank_lines: bool = True,
) -> tuple[str, ...]:
    """Read only the header row of a CSV, tokenized as pandas will tokenize it.

    Uses :func:`pandas.read_csv` with ``header=None`` and ``nrows=1`` plus the
    project policy :data:`RAW_CSV_READ_DEFAULTS` and the tokenizer subset of
    ``read_csv_options``, returning the raw header cells as strings
    (duplicates preserved, no renaming). With blank lines preserved (the
    default) the header must be the first physical line of the file.

    Raises:
        RawDataLoadError: The file cannot be opened, decoded or parsed, has
            no header row, or starts with a blank line; the original
            exception is the cause.
        ValueError: ``read_csv_options`` contains a rejected, policy-governed
            or unrecognised option.
    """
    path = Path(path)
    options = _resolve_read_options(read_csv_options, preserve_blank_lines)
    header_options = {
        k: v for k, v in options.items() if k in _TOKENIZER_READ_OPTIONS | _POLICY_READ_OPTIONS
    }
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


def _resolve_read_options(
    read_csv_options: Mapping[str, Any] | None, preserve_blank_lines: bool
) -> dict[str, Any]:
    """Validate caller options and merge them with the project read policy.

    The policy option (``skip_blank_lines``) is set here from
    ``preserve_blank_lines`` and cannot be supplied in ``read_csv_options``,
    so it is impossible to bypass the blank-line policy by accident.
    """
    options = dict(read_csv_options or {})
    _check_read_options(options)
    if preserve_blank_lines:
        options.update(RAW_CSV_READ_DEFAULTS)
    else:
        options["skip_blank_lines"] = True
    return options


def _with_identifier_dtypes(
    definition: DatasetDefinition, options: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a new options dict whose ``dtype`` reads identifiers as nullable strings.

    Implements the conflict policy in the module docstring. Neither
    ``options`` nor any mapping inside it is modified.
    """
    identifiers = definition.identifier_columns
    merged = dict(options)
    if identifiers and options.get("engine") == "pyarrow":
        raise IdentifierTypeConflictError(definition.key, "engine", identifiers)

    def column_of(key: object) -> object:
        """Resolve a positional key to its contract column name."""
        if isinstance(key, int) and not isinstance(key, bool) and 0 <= key < len(definition.columns):
            return definition.columns[key]
        return key

    for option in ("converters", "parse_dates"):
        value = options.get(option)
        if isinstance(value, Mapping):
            keys = [column_of(k) for k in value]
            keys += [column_of(v) for vs in value.values() if isinstance(vs, (list, tuple)) for v in vs]
        elif isinstance(value, (list, tuple)):
            keys = [column_of(v) for item in value
                    for v in (item if isinstance(item, (list, tuple)) else [item])]
        else:
            keys = []
        clashing = [c for c in identifiers if c in keys]
        if clashing:
            raise IdentifierTypeConflictError(definition.key, option, clashing)

    caller_dtype = options.get("dtype")
    if caller_dtype is None:
        dtype: dict[Any, Any] = {}
    elif isinstance(caller_dtype, Mapping):
        dtype = {}
        clashing = []
        for key, value in caller_dtype.items():
            column = column_of(key)
            if column in identifiers:
                if not _is_nullable_string(value):
                    clashing.append(column)
                continue  # replaced by the project dtype below
            dtype[key] = value
        if clashing:
            raise IdentifierTypeConflictError(definition.key, "dtype", clashing)
    else:
        if identifiers and not _is_text_dtype(caller_dtype):
            raise IdentifierTypeConflictError(definition.key, "dtype", identifiers)
        dtype = {c: caller_dtype for c in definition.columns if c not in identifiers}
    dtype.update(definition.identifier_dtypes)
    merged["dtype"] = dtype
    return merged


def _is_nullable_string(value: object) -> bool:
    try:
        return is_identifier_dtype(pd.api.types.pandas_dtype(value))
    except (TypeError, ValueError):
        return False


def _is_text_dtype(value: object) -> bool:
    try:
        return pd.api.types.is_string_dtype(pd.api.types.pandas_dtype(value))
    except (TypeError, ValueError):
        return False


def _check_read_options(options: Mapping[str, Any]) -> None:
    """Reject options that could omit records, reshape the result or are unclassified."""
    for option in sorted(_FORBIDDEN_READ_OPTIONS.intersection(options)):
        raise IncompleteSourceOptionError(option, _INCOMPLETE_SOURCE_OPTIONS[option])
    if "on_bad_lines" in options and not (isinstance(options["on_bad_lines"], str)
                                          and options["on_bad_lines"] == SAFE_ON_BAD_LINES):
        raise IncompleteSourceOptionError("on_bad_lines", "would skip or reroute malformed records")
    policy = sorted(_POLICY_READ_OPTIONS.intersection(options))
    if policy:
        raise ValueError(
            f"Policy-governed read_csv option(s): {', '.join(policy)}; blank-line "
            "handling is fixed by RAW_CSV_READ_DEFAULTS. Pass preserve_blank_lines=False "
            "to opt out deliberately."
        )
    known = _TOKENIZER_READ_OPTIONS | _DATA_READ_OPTIONS
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(
            f"Unrecognised read_csv option(s): {', '.join(unknown)}; the header "
            "check cannot guarantee it tokenizes the file like the full read"
        )


def _load(definition: DatasetDefinition, path: Path, options: Mapping[str, Any]) -> pd.DataFrame:
    """Validate one file's header, then read it fully once with pandas."""
    # ``options`` are already resolved (policy applied); pass them through the
    # header read unchanged so both reads tokenize the file identically.
    header_options = {k: v for k, v in options.items() if k not in _POLICY_READ_OPTIONS}
    header = read_csv_header(
        path, definition.key, read_csv_options=header_options,
        preserve_blank_lines=not options["skip_blank_lines"],
    )
    validate_header(definition, header)
    try:
        frame = pd.read_csv(path, **options)
    except (OSError, ValueError) as exc:  # pandas parser/empty/decode errors are ValueErrors
        raise RawDataLoadError(definition.key, path) from exc
    validate_header(definition, tuple(frame.columns))
    validate_identifier_dtypes(frame, definition)
    return frame
