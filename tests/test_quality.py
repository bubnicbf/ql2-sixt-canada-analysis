"""Tests for the completely-blank-row quality step.

Every input is a small, obviously synthetic DataFrame or a synthetic CSV
generated in ``tmp_path`` from the centralized column contracts. The
proprietary files are never read and no real counts appear anywhere.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import contract_columns, write_synthetic_csv

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis import quality
from ql2_sixt_canada_analysis.ingestion import RawDatasets, load_raw_datasets
from ql2_sixt_canada_analysis.quality import (
    BlankRowResult,
    RawDatasetBlankRowResults,
    completely_blank_row_mask,
    remove_blank_rows_from_raw_datasets,
    remove_completely_blank_rows,
)
from ql2_sixt_canada_analysis.schemas import DatasetKey

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS


def _frame(rows: list[list[object]], columns: tuple[str, ...] = ("alpha", "beta", "gamma")) -> pd.DataFrame:
    """Object-dtype frame so that any mix of Python values survives as written."""
    return pd.DataFrame(rows, columns=list(columns), dtype=object)


def _assert_invariants(source: pd.DataFrame, result: BlankRowResult) -> None:
    """The documented count, order, value and non-mutation guarantees."""
    assert isinstance(result, BlankRowResult)
    assert result.removed_blank_row_count >= 0
    assert result.retained_row_count >= 0
    assert result.retained_row_count == len(result.cleaned)
    assert result.original_row_count == len(source)
    assert result.original_row_count == result.retained_row_count + result.removed_blank_row_count
    assert list(result.cleaned.columns) == list(source.columns)
    assert not completely_blank_row_mask(result.cleaned).any()
    # Retained rows: exactly the non-blank source rows, in order, labels kept.
    mask = completely_blank_row_mask(source)
    expected = source.loc[~mask.to_numpy()]
    assert list(result.cleaned.index) == list(expected.index)
    pd.testing.assert_frame_equal(result.cleaned, expected)
    assert result.cleaned is not source


# ------------------------------------------------------------- classification


BLANK_ROWS = [
    pytest.param([None, None, None], id="all-None"),
    pytest.param([pd.NA, pd.NA, pd.NaT], id="all-pandas-missing"),
    pytest.param([np.nan, np.nan, np.nan], id="all-numpy-nan"),
    pytest.param(["", "", ""], id="all-empty-strings"),
    pytest.param(["   ", "\t", " \t \n"], id="all-whitespace-strings"),
    pytest.param([np.nan, "", "  "], id="mixed-missing-empty-whitespace"),
    pytest.param([None, pd.NA, " "], id="unicode-whitespace"),
]

NONBLANK_ROWS = [
    pytest.param([0, None, None], id="integer-zero"),
    pytest.param([None, 0.0, None], id="float-zero"),
    pytest.param([None, None, False], id="False"),
    pytest.param(["0", "", np.nan], id="string-zero"),
    pytest.param(["", "False", "  "], id="string-False"),
    pytest.param([np.nan, "", "synthetic"], id="one-meaningful-field"),
    pytest.param(["synthetic_a", "synthetic_b", "synthetic_c"], id="full-row"),
    pytest.param([pd.Timestamp("2001-02-03"), None, ""], id="timestamp"),
    pytest.param([" padded ", None, None], id="padded-text"),
    pytest.param([b"", None, None], id="empty-bytes-is-not-a-string"),
]

ANCHOR = ["synthetic_x", "synthetic_y", "synthetic_z"]


@pytest.mark.parametrize("row", BLANK_ROWS)
def test_completely_blank_row_is_removed(row: list[object]) -> None:
    source = _frame([ANCHOR, row, ANCHOR])
    result = remove_completely_blank_rows(source)
    assert result.removed_blank_row_count == 1
    assert result.retained_row_count == 2
    assert list(result.cleaned.index) == [0, 2]
    _assert_invariants(source, result)


@pytest.mark.parametrize("row", NONBLANK_ROWS)
def test_row_with_a_meaningful_value_is_retained(row: list[object]) -> None:
    source = _frame([ANCHOR, row])
    result = remove_completely_blank_rows(source)
    assert result.removed_blank_row_count == 0
    assert result.retained_row_count == 2
    # The retained row is unchanged, including its blank-looking fields.
    retained = result.cleaned.iloc[1].tolist()
    for got, expected in zip(retained, row, strict=True):
        assert (got is expected) or (got == expected) or (pd.isna(got) and pd.isna(expected))
    _assert_invariants(source, result)


def test_mask_marks_exactly_the_blank_rows() -> None:
    source = _frame([ANCHOR, [None, "", " "], [0, None, None], ["", "", ""]])
    mask = completely_blank_row_mask(source)
    assert mask.dtype == bool
    assert list(mask.index) == list(source.index)
    assert mask.tolist() == [False, True, False, True]


# -------------------------------------------------------- behaviour/invariants


def test_source_frame_is_not_mutated() -> None:
    source = _frame([ANCHOR, [None, "", " "], ["0", None, None]])
    snapshot = source.copy(deep=True)
    result = remove_completely_blank_rows(source)
    pd.testing.assert_frame_equal(source, snapshot)
    assert source.index.equals(snapshot.index)
    result.cleaned.iloc[0, 0] = "mutated_in_result"  # must not leak back
    pd.testing.assert_frame_equal(source, snapshot)


def test_column_order_is_preserved() -> None:
    columns = ("zeta", "alpha", "mid", "beta")
    source = _frame([["s", None, 1, False], [None, None, None, None]], columns)
    assert list(remove_completely_blank_rows(source).cleaned.columns) == list(columns)


def test_retained_row_order_and_values_are_preserved() -> None:
    rows = [["b", 2, " y "], [None, None, None], ["a", 1, " x "], ["", "", ""], ["c", 3, ""]]
    source = _frame(rows)
    result = remove_completely_blank_rows(source)
    assert result.cleaned.values.tolist() == [rows[0], rows[2], rows[4]]
    assert result.cleaned.iloc[0, 2] == " y "  # whitespace not stripped


def test_original_index_labels_are_preserved_not_reset() -> None:
    source = _frame([ANCHOR, [None, None, None], ANCHOR, ["", " ", None], ANCHOR])
    source.index = pd.Index([10, 20, 30, 40, 50], name="synthetic_id")
    result = remove_completely_blank_rows(source)
    assert list(result.cleaned.index) == [10, 30, 50]
    assert result.cleaned.index.name == "synthetic_id"


def test_counts_reconcile_and_no_blank_rows_remain() -> None:
    source = _frame([ANCHOR, [None, None, None], ["", "", ""], ANCHOR, [" ", None, ""], ANCHOR])
    result = remove_completely_blank_rows(source)
    assert (result.original_row_count, result.retained_row_count, result.removed_blank_row_count) == (6, 3, 3)
    assert not completely_blank_row_mask(result.cleaned).any()
    _assert_invariants(source, result)


def test_empty_frame_returns_zero_counts() -> None:
    source = pd.DataFrame(columns=["alpha", "beta"])
    result = remove_completely_blank_rows(source)
    assert (result.original_row_count, result.retained_row_count, result.removed_blank_row_count) == (0, 0, 0)
    assert result.cleaned.empty and list(result.cleaned.columns) == ["alpha", "beta"]
    _assert_invariants(source, result)


def test_frame_without_columns_keeps_its_rows() -> None:
    source = pd.DataFrame(index=[1, 2, 3])
    result = remove_completely_blank_rows(source)
    assert result.removed_blank_row_count == 0 and result.retained_row_count == 3


def test_frame_with_no_blank_rows_returns_zero_removed() -> None:
    source = _frame([ANCHOR, ["0", 0, False], [" ", None, 0.0]])
    result = remove_completely_blank_rows(source)
    assert result.removed_blank_row_count == 0
    pd.testing.assert_frame_equal(result.cleaned, source)
    _assert_invariants(source, result)


def test_frame_with_only_blank_rows_becomes_empty() -> None:
    source = _frame([[None, None, None], ["", "", ""], [" ", np.nan, ""]])
    result = remove_completely_blank_rows(source)
    assert (result.original_row_count, result.retained_row_count, result.removed_blank_row_count) == (3, 0, 3)
    assert result.cleaned.empty and list(result.cleaned.columns) == list(source.columns)
    _assert_invariants(source, result)


def test_mixed_dtypes_including_string_category_and_datetime() -> None:
    source = pd.DataFrame(
        {
            "num": [1.5, np.nan, 0.0, np.nan],
            "text": pd.array(["s", None, None, ""], dtype="string"),
            "cat": pd.Categorical(["k", None, None, " "]),
            "when": pd.to_datetime(["2001-01-01", None, None, None]),
            "flag": pd.array([True, None, False, None], dtype="boolean"),
        }
    )
    result = remove_completely_blank_rows(source)
    assert list(result.cleaned.index) == [0, 2]
    assert result.removed_blank_row_count == 2
    assert result.cleaned.dtypes.equals(source.dtypes)
    _assert_invariants(source, result)


def test_numeric_only_frame() -> None:
    source = pd.DataFrame({"a": [0, np.nan, 2], "b": [np.nan, np.nan, 0.0]})
    result = remove_completely_blank_rows(source)
    assert list(result.cleaned.index) == [0, 2]


def test_duplicate_nonblank_rows_are_not_deduplicated() -> None:
    source = _frame([ANCHOR, ANCHOR, [None, None, None], ANCHOR])
    result = remove_completely_blank_rows(source)
    assert result.retained_row_count == 3
    assert result.cleaned.values.tolist() == [ANCHOR, ANCHOR, ANCHOR]


def test_blank_looking_field_does_not_remove_row_with_meaningful_field() -> None:
    source = _frame([["   ", "", "synthetic"], ["", None, 0]])
    result = remove_completely_blank_rows(source)
    assert result.removed_blank_row_count == 0
    assert result.cleaned.iloc[0, 0] == "   "  # kept verbatim


def test_result_is_frozen_and_typed() -> None:
    result = remove_completely_blank_rows(_frame([ANCHOR]))
    assert dataclasses.is_dataclass(result)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.removed_blank_row_count = 5  # type: ignore[misc]
    assert all(isinstance(getattr(result, f), int) for f in
               ("original_row_count", "retained_row_count", "removed_blank_row_count"))


def test_inconsistent_result_is_rejected_as_programmer_error() -> None:
    with pytest.raises(AssertionError):
        BlankRowResult(cleaned=_frame([ANCHOR]), original_row_count=1,
                       retained_row_count=1, removed_blank_row_count=1)


@pytest.mark.parametrize("bad", [None, [], {}, "synthetic", pd.Series([1, 2])])
def test_non_dataframe_input_raises_type_error(bad: object) -> None:
    with pytest.raises(TypeError):
        remove_completely_blank_rows(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        completely_blank_row_mask(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------- both datasets


def _synthetic_datasets() -> RawDatasets:
    jobs_cols, cars_cols = contract_columns(JOBS), contract_columns(CARS)
    jobs = _frame(
        [[f"j{c}" for c in range(len(jobs_cols))], [None] * len(jobs_cols), ["0"] + [None] * (len(jobs_cols) - 1)],
        jobs_cols,
    )
    cars = _frame(
        [[""] * len(cars_cols), [f"c{c}" for c in range(len(cars_cols))], [" "] * len(cars_cols), [None] * len(cars_cols)],
        cars_cols,
    )
    return RawDatasets(jobs=jobs, cars=cars)


def test_both_datasets_are_processed_independently() -> None:
    datasets = _synthetic_datasets()
    results = remove_blank_rows_from_raw_datasets(datasets)
    assert isinstance(results, RawDatasetBlankRowResults)
    assert results.jobs is not results.cars
    assert results.jobs.removed_blank_row_count == 1
    assert results.cars.removed_blank_row_count == 3
    assert results.total_removed_blank_row_count == 4
    assert results.total_removed_blank_row_count == (
        results.jobs.removed_blank_row_count + results.cars.removed_blank_row_count
    )
    assert results.by_key[JOBS] is results.jobs and results.by_key[CARS] is results.cars
    assert set(results.by_key) == set(DatasetKey)
    _assert_invariants(datasets.jobs, results.jobs)
    _assert_invariants(datasets.cars, results.cars)


def test_cleaned_datasets_keep_their_contract_columns() -> None:
    results = remove_blank_rows_from_raw_datasets(_synthetic_datasets())
    assert tuple(results.jobs.cleaned.columns) == contract_columns(JOBS)
    assert tuple(results.cars.cleaned.columns) == contract_columns(CARS)
    cleaned = results.cleaned
    assert isinstance(cleaned, RawDatasets)
    assert cleaned.jobs is results.jobs.cleaned and cleaned.cars is results.cars.cleaned


def test_neither_source_frame_is_mutated() -> None:
    datasets = _synthetic_datasets()
    jobs_before, cars_before = datasets.jobs.copy(deep=True), datasets.cars.copy(deep=True)
    remove_blank_rows_from_raw_datasets(datasets)
    pd.testing.assert_frame_equal(datasets.jobs, jobs_before)
    pd.testing.assert_frame_equal(datasets.cars, cars_before)


def test_non_container_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        remove_blank_rows_from_raw_datasets((_frame([ANCHOR]), _frame([ANCHOR])))  # type: ignore[arg-type]


# ------------------------------------------------- ingestion keeps blank rows


def _write_with_blank_lines(directory: Path, key: DatasetKey, lines: list[str]) -> Path:
    """Write header + ``lines`` for ``key``. Line endings are explicit ``\\n``."""
    columns = contract_columns(key)
    path = directory / f"synthetic_{key}.csv"
    text = ",".join(columns) + "\n" + "".join(line + "\n" for line in lines)
    path.write_bytes(text.encode("utf-8"))  # bytes: no platform newline translation
    return path


def _record(key: DatasetKey, tag: str) -> str:
    return ",".join(f"synthetic_{tag}_{c}" for c in range(len(contract_columns(key))))


def test_ingestion_preserves_blank_lines_and_quality_step_removes_them(tmp_path: Path) -> None:
    for key in DatasetKey:
        n = len(contract_columns(key))
        _write_with_blank_lines(tmp_path, key, [
            _record(key, "first"),
            "",                       # physically empty line
            "," * (n - 1),            # delimiter-only line
            "   ",                    # whitespace-only line (first field)
            " \t ," + " ," * (n - 2) + " ",  # whitespace in every field
            _record(key, "last"),
        ])
    raw = load_raw_datasets(tmp_path)
    for key, frame in ((JOBS, raw.jobs), (CARS, raw.cars)):
        assert len(frame) == 6, "ingestion must keep every physical line as a row"
        assert tuple(frame.columns) == contract_columns(key)  # schema still validated
        assert frame.iloc[0, 0] == "synthetic_first_0" and frame.iloc[5, 0] == "synthetic_last_0"
        assert frame.iloc[1].isna().all()           # empty line -> all missing
        assert frame.iloc[2].isna().all()           # delimiter-only -> all missing
        assert frame.iloc[3, 0] == "   " and frame.iloc[3, 1:].isna().all()
        assert frame.iloc[4, 0] == " \t "
        mask = completely_blank_row_mask(frame)
        assert mask.tolist() == [False, True, True, True, True, False]

    results = remove_blank_rows_from_raw_datasets(raw)
    for key, result in results.by_key.items():
        assert result.original_row_count == 6
        assert result.removed_blank_row_count == 4
        assert result.retained_row_count == 2
        assert list(result.cleaned.index) == [0, 5]  # original labels kept
        assert result.cleaned.iloc[:, 0].tolist() == ["synthetic_first_0", "synthetic_last_0"]
    assert results.total_removed_blank_row_count == 8


def test_trailing_blank_lines_are_rows_too(tmp_path: Path) -> None:
    for key in DatasetKey:
        _write_with_blank_lines(tmp_path, key, [_record(key, "only"), "", ""])
    raw = load_raw_datasets(tmp_path)
    assert len(raw.jobs) == 3 and len(raw.cars) == 3
    results = remove_blank_rows_from_raw_datasets(raw)
    assert results.jobs.removed_blank_row_count == 2
    assert results.cars.removed_blank_row_count == 2
    assert results.total_removed_blank_row_count == 4


def test_synthetic_files_without_blank_lines_remove_nothing(tmp_path: Path) -> None:
    for key in DatasetKey:
        write_synthetic_csv(tmp_path / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    results = remove_blank_rows_from_raw_datasets(load_raw_datasets(tmp_path))
    assert results.total_removed_blank_row_count == 0
    assert results.jobs.retained_row_count == 3 and results.cars.retained_row_count == 3


def test_quality_step_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for key in DatasetKey:
        _write_with_blank_lines(tmp_path, key, [_record(key, "a"), "", _record(key, "b")])
    before = sorted(p.name for p in tmp_path.rglob("*"))
    remove_blank_rows_from_raw_datasets(load_raw_datasets(tmp_path))
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


# ------------------------------------------------------------------- packaging


def test_package_exposes_quality_api() -> None:
    assert ql2_sixt_canada_analysis.remove_completely_blank_rows is remove_completely_blank_rows
    assert ql2_sixt_canada_analysis.remove_blank_rows_from_raw_datasets is remove_blank_rows_from_raw_datasets
    assert ql2_sixt_canada_analysis.BlankRowResult is BlankRowResult
    assert ql2_sixt_canada_analysis.RawDatasetBlankRowResults is RawDatasetBlankRowResults


def test_module_documents_the_single_definition() -> None:
    doc = quality.__doc__ or ""
    for phrase in ("completely blank", "whitespace", "0.0", "False", "index"):
        assert phrase in doc
