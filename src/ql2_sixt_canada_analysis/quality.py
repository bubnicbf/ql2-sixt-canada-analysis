"""Data-quality step: identify, remove and quantify completely blank rows.

Definition (the single project-wide definition of a completely blank row)
-------------------------------------------------------------------------
A row is **completely blank** when *every* field in it is one of:

* a pandas-recognised missing value (``NaN``, ``NaT``, ``pd.NA``),
* ``None``,
* an empty string ``""``, or
* a string containing only whitespace (``str.strip() == ""``).

A row with at least one other value is **not** blank and is kept unchanged.
In particular ``0``, ``0.0``, ``False``, the strings ``"0"`` and ``"False"``,
any non-empty text, timestamps and every other non-missing scalar are
meaningful. Partially blank rows are retained in full.

Guarantees
----------
* The caller's DataFrame is never mutated; results hold a new frame.
* Column order, the relative order of retained rows and every retained cell
  value (including surrounding whitespace) are preserved exactly.
* **Index policy:** the original index labels are preserved (no reset), so a
  retained row keeps the label it had in the source frame and can be traced
  back to it. Callers who want a positional index call ``reset_index`` on
  the result themselves.
* Nothing is written to disk and nothing is logged or printed; counts are
  returned programmatically. Metrics computed from the proprietary data must
  never be committed.

:class:`pandas.DataFrame` objects are mutable even when held by a frozen
dataclass; the result types freeze their *fields*, not the frame contents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype

from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.schemas import DatasetKey

__all__ = [
    "BlankRowResult",
    "RawDatasetBlankRowResults",
    "completely_blank_row_mask",
    "remove_blank_rows_from_raw_datasets",
    "remove_completely_blank_rows",
]


# ----------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class BlankRowResult:
    """Outcome of removing completely blank rows from one DataFrame.

    Attributes:
        cleaned: The retained rows (a new frame; original index preserved).
        original_row_count: Rows in the input frame.
        retained_row_count: Rows in ``cleaned``.
        removed_blank_row_count: Completely blank rows that were removed.
    """

    cleaned: pd.DataFrame
    original_row_count: int
    retained_row_count: int
    removed_blank_row_count: int

    def __post_init__(self) -> None:
        # Programmer invariants: a result that violates these is a bug here.
        assert self.removed_blank_row_count >= 0
        assert self.retained_row_count >= 0
        assert self.retained_row_count == len(self.cleaned)
        assert self.original_row_count == self.retained_row_count + self.removed_blank_row_count


@dataclass(frozen=True, slots=True)
class RawDatasetBlankRowResults:
    """Blank-row results for both logical raw datasets, processed independently.

    Attributes:
        jobs: Result for the ``jobs`` dataset.
        cars: Result for the ``cars`` dataset.
    """

    jobs: BlankRowResult
    cars: BlankRowResult

    @property
    def by_key(self) -> Mapping[DatasetKey, BlankRowResult]:
        """The per-dataset results keyed by :class:`DatasetKey`."""
        return MappingProxyType({DatasetKey.JOBS: self.jobs, DatasetKey.CARS: self.cars})

    @property
    def total_removed_blank_row_count(self) -> int:
        """Completely blank rows removed across both datasets."""
        return self.jobs.removed_blank_row_count + self.cars.removed_blank_row_count

    @property
    def cleaned(self) -> RawDatasets:
        """Both cleaned frames in the ingestion container type."""
        return RawDatasets(jobs=self.jobs.cleaned, cars=self.cars.cleaned)


# ------------------------------------------------------------------- public API


def completely_blank_row_mask(frame: pd.DataFrame) -> pd.Series:
    """Return a boolean Series (aligned to ``frame.index``) marking blank rows.

    Implements the module definition exactly, vectorised: a cell is blank when
    it is missing (``isna``) or is a string whose ``strip()`` is empty. Only
    string values are inspected for whitespace, so ``0``, ``False`` and other
    scalars are never blank. ``frame`` is not modified.

    Raises:
        TypeError: ``frame`` is not a :class:`pandas.DataFrame`.
    """
    _require_dataframe(frame)
    if frame.shape[1] == 0:
        # No fields at all: nothing can be "entirely blank" in a meaningful
        # sense, and every row is kept.
        return pd.Series(False, index=frame.index, dtype=bool)

    blank = frame.isna().to_numpy(dtype=bool, copy=True)  # rows x columns
    for position, (_, column) in enumerate(frame.items()):
        values = _string_bearing_values(column)
        if values is not None:
            # ``str.strip()`` leaves a missing value for non-string cells, and
            # ``== ""`` is False for those, so only genuine whitespace-only or
            # empty strings are flagged here (missing cells come from isna).
            blank[:, position] |= (values.str.strip() == "").to_numpy(dtype=bool)
    return pd.Series(blank.all(axis=1), index=frame.index, dtype=bool)


def remove_completely_blank_rows(frame: pd.DataFrame) -> BlankRowResult:
    """Remove rows that are completely blank and count them.

    See the module docstring for the definition and guarantees (no mutation,
    original index preserved, column order, row order and values unchanged).

    Raises:
        TypeError: ``frame`` is not a :class:`pandas.DataFrame`.
    """
    mask = completely_blank_row_mask(frame)
    removed = int(mask.sum())
    if removed == 0:
        cleaned = frame.copy()  # one copy guarantees the caller's frame is independent
    else:
        cleaned = frame.loc[~mask.to_numpy()]  # boolean take: already a new frame
    return BlankRowResult(
        cleaned=cleaned,
        original_row_count=len(frame),
        retained_row_count=len(cleaned),
        removed_blank_row_count=removed,
    )


def remove_blank_rows_from_raw_datasets(datasets: RawDatasets) -> RawDatasetBlankRowResults:
    """Apply :func:`remove_completely_blank_rows` to ``jobs`` and ``cars`` separately.

    Raises:
        TypeError: ``datasets`` is not a :class:`RawDatasets` or a member is
            not a :class:`pandas.DataFrame`.
    """
    if not isinstance(datasets, RawDatasets):
        raise TypeError(f"expected RawDatasets, got {type(datasets).__name__}")
    return RawDatasetBlankRowResults(
        jobs=remove_completely_blank_rows(datasets.jobs),
        cars=remove_completely_blank_rows(datasets.cars),
    )


# ---------------------------------------------------------------------- helpers


def _require_dataframe(frame: object) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(frame).__name__}")


# ``infer_dtype`` results under which a column may contain Python strings.
_STRING_BEARING_INFERENCES: Final = frozenset({"string", "mixed", "mixed-integer"})


def _string_bearing_values(column: pd.Series) -> pd.Series | None:
    """Return ``column`` as object values if it may contain strings, else None.

    Numeric, boolean, datetime and other non-object dtypes cannot hold
    strings, and an object column whose non-missing values are e.g. all
    integers has no strings either; both are skipped so ``.str`` is only used
    where pandas allows it.
    """
    dtype = column.dtype
    if not (dtype == np.dtype(object) or isinstance(dtype, (pd.StringDtype, pd.CategoricalDtype))):
        return None
    values = column.astype(object)
    if infer_dtype(values, skipna=True) not in _STRING_BEARING_INFERENCES:
        return None
    return values
