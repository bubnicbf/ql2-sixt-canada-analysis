"""Safe representation of identifier columns as pandas nullable strings.

Identifier columns are defined once per dataset in
:mod:`ql2_sixt_canada_analysis.schemas` (``DatasetDefinition.identifier_columns``)
and represented with :data:`~ql2_sixt_canada_analysis.schemas.IDENTIFIER_DTYPE`,
pandas' nullable ``"string"`` dtype whose missing value is ``pd.NA``.

* **Read time (primary protection).** The standard loader,
  :func:`ql2_sixt_canada_analysis.ingestion.load_raw_datasets`, passes the
  identifier dtype mapping to :func:`pandas.read_csv`, so identifiers are
  never parsed as numbers: leading zeros, long digit strings and zero-like
  values survive exactly as written, and empty fields stay missing.
* **Post-load helper.** :func:`cast_identifier_fields` and
  :func:`cast_identifiers_for_raw_datasets` cast DataFrames that were built
  elsewhere. They **cannot restore** leading zeros, digits or formatting
  already lost if the frame was parsed numerically before it got here.
* **Validation.** :func:`validate_identifier_dtypes` and
  :func:`validate_raw_dataset_identifier_dtypes` are silent on success and
  raise :class:`IdentifierDtypeError` otherwise.

Nothing here strips, re-cases, pads, parses, fills or validates identifier
*content*; only the type changes. Error messages carry the dataset key and
counts only; offending column names are on error attributes and identifier
values never appear. Nothing is printed, logged or written.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import TYPE_CHECKING

import pandas as pd

from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    IDENTIFIER_DTYPE,
    DatasetDefinition,
    DatasetKey,
)

if TYPE_CHECKING:  # ingestion imports this module; avoid a runtime cycle
    from ql2_sixt_canada_analysis.ingestion import RawDatasets

__all__ = [
    "IdentifierDtypeError",
    "IdentifierError",
    "IdentifierTypeConflictError",
    "MissingIdentifierColumnError",
    "cast_identifier_fields",
    "cast_identifiers_for_raw_datasets",
    "is_identifier_dtype",
    "validate_identifier_dtypes",
    "validate_raw_dataset_identifier_dtypes",
]


# --------------------------------------------------------------------- exceptions


class IdentifierError(Exception):
    """Base class for identifier typing failures."""

    def __init__(self, message: str, key: DatasetKey, columns: Iterable[str]) -> None:
        super().__init__(message)
        self.role = key
        self.columns = tuple(columns)


class MissingIdentifierColumnError(IdentifierError, KeyError):
    """A configured identifier column is absent from a DataFrame."""

    def __init__(self, key: DatasetKey, columns: Iterable[str]) -> None:
        columns = tuple(columns)
        super().__init__(
            f"The '{key}' frame lacks {len(columns)} configured identifier column(s).", key, columns
        )

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0])


class IdentifierDtypeError(IdentifierError, TypeError):
    """Configured identifier columns do not use the nullable string dtype."""

    def __init__(self, key: DatasetKey, columns: Iterable[str]) -> None:
        columns = tuple(columns)
        super().__init__(
            f"The '{key}' frame has {len(columns)} identifier column(s) not stored as the "
            "nullable string dtype.", key, columns,
        )


class IdentifierTypeConflictError(IdentifierError, ValueError):
    """Caller read options would parse an identifier column unsafely.

    Attributes:
        option: The ``read_csv`` option that conflicts (e.g. ``"dtype"``).
    """

    def __init__(self, key: DatasetKey, option: str, columns: Iterable[str]) -> None:
        columns = tuple(columns)
        super().__init__(
            f"read_csv option '{option}' conflicts with {len(columns)} '{key}' identifier "
            "column(s); identifiers are always read as the nullable string dtype.",
            key, columns,
        )
        self.option = option


# ------------------------------------------------------------------- public API


def is_identifier_dtype(dtype: object) -> bool:
    """True if ``dtype`` is pandas' nullable string dtype (missing value ``pd.NA``).

    Storage (``python`` or ``pyarrow``) is irrelevant. ``object``, NumPy
    unicode and the NaN-backed ``str`` dtype are *not* accepted.
    """
    return isinstance(dtype, pd.StringDtype) and dtype.na_value is pd.NA


def cast_identifier_fields(frame: pd.DataFrame, definition: DatasetDefinition) -> pd.DataFrame:
    """Return a new frame with ``definition``'s identifier columns as nullable strings.

    Only the configured identifier columns change type; column order, row
    order, index, and every other column's values and dtype are unchanged and
    ``frame`` is not mutated. Missing values become ``pd.NA`` (never the text
    ``"nan"``/``"None"``). Idempotent.

    This cannot recover representation lost *before* the call: an identifier
    that was already parsed as a number has lost its leading zeros or digits.
    The standard loader avoids that by typing identifiers at read time.

    Raises:
        TypeError: ``frame`` is not a DataFrame or ``definition`` is not a
            :class:`DatasetDefinition`.
        MissingIdentifierColumnError: A configured identifier column is absent.
    """
    _require(frame, definition)
    missing = [c for c in definition.identifier_columns if c not in frame.columns]
    if missing:
        raise MissingIdentifierColumnError(definition.key, missing)
    if not definition.identifier_columns:
        return frame.copy()
    # ``astype`` with a mapping returns a new frame and leaves other columns'
    # data and dtypes untouched.
    return frame.astype(dict(definition.identifier_dtypes))


def cast_identifiers_for_raw_datasets(datasets: RawDatasets) -> RawDatasets:
    """Apply :func:`cast_identifier_fields` to ``jobs`` and ``cars`` with their own definitions."""
    _require_raw_datasets(datasets)
    return dataclasses.replace(
        datasets,
        jobs=cast_identifier_fields(datasets.jobs, DATASET_DEFINITIONS[DatasetKey.JOBS]),
        cars=cast_identifier_fields(datasets.cars, DATASET_DEFINITIONS[DatasetKey.CARS]),
    )


def validate_identifier_dtypes(frame: pd.DataFrame, definition: DatasetDefinition) -> None:
    """Check every configured identifier column is present and nullable-string typed.

    Silent on success; never modifies ``frame``.

    Raises:
        TypeError: Invalid argument types.
        MissingIdentifierColumnError: A configured identifier column is absent.
        IdentifierDtypeError: An identifier column has another dtype.
    """
    _require(frame, definition)
    missing = [c for c in definition.identifier_columns if c not in frame.columns]
    if missing:
        raise MissingIdentifierColumnError(definition.key, missing)
    wrong = [c for c in definition.identifier_columns if not is_identifier_dtype(frame[c].dtype)]
    if wrong:
        raise IdentifierDtypeError(definition.key, wrong)


def validate_raw_dataset_identifier_dtypes(datasets: RawDatasets) -> None:
    """Run :func:`validate_identifier_dtypes` on ``jobs`` and ``cars``. Silent on success."""
    _require_raw_datasets(datasets)
    validate_identifier_dtypes(datasets.jobs, DATASET_DEFINITIONS[DatasetKey.JOBS])
    validate_identifier_dtypes(datasets.cars, DATASET_DEFINITIONS[DatasetKey.CARS])


# ---------------------------------------------------------------------- helpers


def _require(frame: object, definition: object) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(frame).__name__}")
    if not isinstance(definition, DatasetDefinition):
        raise TypeError(f"expected a DatasetDefinition, got {type(definition).__name__}")


def _require_raw_datasets(datasets: object) -> None:
    from ql2_sixt_canada_analysis.ingestion import RawDatasets  # noqa: PLC0415 (cycle)

    if not isinstance(datasets, RawDatasets):
        raise TypeError(f"expected RawDatasets, got {type(datasets).__name__}")


# Programmer invariant: the project-wide identifier dtype satisfies the predicate.
assert is_identifier_dtype(IDENTIFIER_DTYPE)
