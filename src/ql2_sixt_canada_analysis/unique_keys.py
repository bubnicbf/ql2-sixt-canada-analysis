"""Assess and strictly validate each dataset's centrally defined unique key.

The key contract for a dataset is ``DatasetDefinition.unique_key_columns``
(see :mod:`ql2_sixt_canada_analysis.schemas`, which also documents each
dataset's row grain). A key is valid only when it is both:

* **complete** - every row has a non-missing value for every component, and
* **unique** - no two complete rows share the full component tuple.

Semantics
---------
* A component is missing only when pandas reports it missing (``pd.NA``,
  ``NaN``, ``None``, ``NaT``). Non-empty text such as ``"0"``, ``"False"``,
  ``"N/A"``, ``"None"``, ``"null"`` or ``"nan"`` is a value, and empty or
  whitespace-only strings are compared verbatim: keys are never stripped or
  normalised here. Semantic identifier-content checks are a separate control.
* A row with *any* missing component is a missing-key row. Missing-key rows
  are reported separately and never form duplicate groups.
* Uniqueness is evaluated only among complete rows, using all components as a
  tuple via :meth:`pandas.DataFrame.duplicated` (``keep=False``). Keys are never
  concatenated into strings, so delimiter-like characters cannot collide.
* Every member of a duplicate group counts as a duplicate row; groups are
  counted separately.
* **Empty frames** are vacuously valid: there are no missing-key rows and no
  duplicates. Whether data was expected at all belongs to a separate
  presence/volume control.

Assessment (:func:`assess_unique_key`) always returns a
:class:`UniqueKeyReport` and only raises for configuration problems
(:class:`~ql2_sixt_canada_analysis.schemas.KeyConfigurationError`). Strict
validation (:func:`validate_unique_key`) raises :class:`UniqueKeyViolationError`
when the contract fails. Neither modifies, sorts, removes or repairs rows.

Reports and errors hold counts and column names only - never key values,
rows or data samples - and nothing is printed, logged or written. Counts from
the proprietary data are proprietary: keep them in memory.

Pipeline position: load (blank lines kept, identifiers typed) -> validate
source columns -> remove completely blank rows -> validate identifier dtypes ->
assess unique keys on the cleaned frames, so a fully blank physical line is
counted by the blank-row control and not again as a missing key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    DatasetDefinition,
    DatasetKey,
    KeyConfigurationError,
)

__all__ = [
    "KeyConfigurationError",
    "RawDatasetUniqueKeyReports",
    "UniqueKeyReport",
    "UniqueKeyViolationError",
    "assess_raw_dataset_unique_keys",
    "assess_unique_key",
    "validate_raw_dataset_unique_keys",
    "validate_unique_key",
]


# ----------------------------------------------------------------------- reports


@dataclass(frozen=True, slots=True)
class UniqueKeyReport:
    """Completeness and uniqueness metrics for one dataset's key (no values).

    Attributes:
        dataset: Logical dataset assessed.
        key_columns: The key components, in contract order.
        total_row_count: Rows assessed.
        complete_key_row_count: Rows with every key component present.
        missing_key_row_count: Rows with at least one missing component.
        duplicate_key_row_count: Complete rows whose key tuple occurs more than
            once (every member of every duplicate group).
        duplicate_key_group_count: Distinct key tuples that occur more than once.
        distinct_complete_key_count: Distinct key tuples among complete rows.
    """

    dataset: DatasetKey
    key_columns: tuple[str, ...]
    total_row_count: int
    complete_key_row_count: int
    missing_key_row_count: int
    duplicate_key_row_count: int
    duplicate_key_group_count: int
    distinct_complete_key_count: int

    def __post_init__(self) -> None:
        # Programmer invariants; a violation is a bug in this module.
        counts = (self.total_row_count, self.complete_key_row_count, self.missing_key_row_count,
                  self.duplicate_key_row_count, self.duplicate_key_group_count,
                  self.distinct_complete_key_count)
        assert all(isinstance(c, int) and c >= 0 for c in counts)
        assert self.total_row_count == self.complete_key_row_count + self.missing_key_row_count
        assert self.duplicate_key_row_count <= self.complete_key_row_count
        assert (self.duplicate_key_group_count == 0) == (self.duplicate_key_row_count == 0)
        assert self.duplicate_key_row_count >= 2 * self.duplicate_key_group_count
        assert self.distinct_complete_key_count <= self.complete_key_row_count
        # Each group of size n contributes n rows but one distinct key.
        assert self.distinct_complete_key_count == (
            self.complete_key_row_count - self.duplicate_key_row_count
            + self.duplicate_key_group_count
        )

    @property
    def is_complete(self) -> bool:
        """No row is missing a key component."""
        return self.missing_key_row_count == 0

    @property
    def is_unique(self) -> bool:
        """No complete key tuple repeats (missing-key rows are excluded)."""
        return self.duplicate_key_row_count == 0

    @property
    def is_valid(self) -> bool:
        """The key contract holds: complete *and* unique."""
        return self.is_complete and self.is_unique

    @property
    def violations(self) -> tuple[str, ...]:
        """Safe violation categories: ``"missing_key"`` and/or ``"duplicate_key"``."""
        return tuple(
            name for name, failed in (("missing_key", not self.is_complete),
                                      ("duplicate_key", not self.is_unique)) if failed
        )


@dataclass(frozen=True, slots=True)
class RawDatasetUniqueKeyReports:
    """Unique-key reports for ``jobs`` and ``cars``, assessed independently."""

    jobs: UniqueKeyReport
    cars: UniqueKeyReport

    @property
    def by_key(self) -> Mapping[DatasetKey, UniqueKeyReport]:
        return MappingProxyType({DatasetKey.JOBS: self.jobs, DatasetKey.CARS: self.cars})

    @property
    def all_valid(self) -> bool:
        return self.jobs.is_valid and self.cars.is_valid

    @property
    def total_missing_key_row_count(self) -> int:
        return self.jobs.missing_key_row_count + self.cars.missing_key_row_count

    @property
    def total_duplicate_key_row_count(self) -> int:
        return self.jobs.duplicate_key_row_count + self.cars.duplicate_key_row_count


# -------------------------------------------------------------------- exceptions


class UniqueKeyViolationError(Exception):
    """Strict validation found missing or duplicate keys in source data.

    The message names the dataset(s) and violation categories only. The
    in-memory reports (counts, no values) are on ``reports``; do not log them
    for proprietary data.
    """

    def __init__(self, reports: tuple[UniqueKeyReport, ...]) -> None:
        self.reports = tuple(reports)
        parts = [f"'{r.dataset}' ({', '.join(r.violations)})" for r in self.reports]
        super().__init__("Unique-key contract failed for " + "; ".join(parts) + ".")

    @property
    def roles(self) -> tuple[DatasetKey, ...]:
        return tuple(r.dataset for r in self.reports)


# ------------------------------------------------------------------- public API


def assess_unique_key(frame: pd.DataFrame, definition: DatasetDefinition) -> UniqueKeyReport:
    """Measure key completeness and uniqueness for ``frame`` (never raises on violations).

    ``frame`` is not modified, sorted or copied in full; only the key columns
    are read.

    Raises:
        TypeError: Invalid argument types.
        KeyConfigurationError: The definition declares no key, repeats a
            component, or a component is absent from ``frame``.
    """
    key = _key_columns(frame, definition)
    keys = frame.loc[:, list(key)]                      # key columns only
    complete = keys.notna().all(axis=1).to_numpy()      # vectorised, per row
    complete_keys = keys.loc[complete]
    duplicated = complete_keys.duplicated(keep=False).to_numpy()   # all group members
    duplicate_rows = int(duplicated.sum())
    groups = int(len(complete_keys.loc[duplicated].drop_duplicates())) if duplicate_rows else 0
    distinct = int(len(complete_keys.drop_duplicates()))
    complete_count = int(complete.sum())
    return UniqueKeyReport(
        dataset=definition.key,
        key_columns=key,
        total_row_count=len(frame),
        complete_key_row_count=complete_count,
        missing_key_row_count=len(frame) - complete_count,
        duplicate_key_row_count=duplicate_rows,
        duplicate_key_group_count=groups,
        distinct_complete_key_count=distinct,
    )


def assess_raw_dataset_unique_keys(datasets: RawDatasets) -> RawDatasetUniqueKeyReports:
    """Assess ``jobs`` and ``cars`` against their own centralized key contracts."""
    if not isinstance(datasets, RawDatasets):
        raise TypeError(f"expected RawDatasets, got {type(datasets).__name__}")
    return RawDatasetUniqueKeyReports(
        jobs=assess_unique_key(datasets.jobs, DATASET_DEFINITIONS[DatasetKey.JOBS]),
        cars=assess_unique_key(datasets.cars, DATASET_DEFINITIONS[DatasetKey.CARS]),
    )


def validate_unique_key(frame: pd.DataFrame, definition: DatasetDefinition) -> UniqueKeyReport:
    """Assess, then return the report if valid or raise :class:`UniqueKeyViolationError`."""
    report = assess_unique_key(frame, definition)
    if not report.is_valid:
        raise UniqueKeyViolationError((report,))
    return report


def validate_raw_dataset_unique_keys(datasets: RawDatasets) -> RawDatasetUniqueKeyReports:
    """Assess both datasets; raise one :class:`UniqueKeyViolationError` listing every failure."""
    reports = assess_raw_dataset_unique_keys(datasets)
    failed = tuple(r for r in (reports.jobs, reports.cars) if not r.is_valid)
    if failed:
        raise UniqueKeyViolationError(failed)
    return reports


# ---------------------------------------------------------------------- helpers


def _key_columns(frame: object, definition: object) -> tuple[str, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(frame).__name__}")
    if not isinstance(definition, DatasetDefinition):
        raise TypeError(f"expected a DatasetDefinition, got {type(definition).__name__}")
    key = definition.unique_key_columns
    if not key:
        raise KeyConfigurationError(f"{definition.key}: no unique key is defined", definition.key)
    if len(set(key)) != len(key):  # guards definitions altered after construction
        raise KeyConfigurationError(
            f"{definition.key}: unique key repeats a component", definition.key,
            tuple(dict.fromkeys(c for c in key if key.count(c) > 1)),
        )
    absent = tuple(c for c in key if c not in frame.columns)
    if absent:
        raise KeyConfigurationError(
            f"The '{definition.key}' frame lacks {len(absent)} unique-key column(s).",
            definition.key, absent,
        )
    return tuple(key)
