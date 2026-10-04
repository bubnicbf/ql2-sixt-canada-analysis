"""Tests for centralized identifier definitions and safe identifier typing.

All identifier values are fabricated (``000123``, ``SYNTHETIC-ID-001``, ...)
and every CSV is generated in ``tmp_path`` from the centralized contracts.
The proprietary files are never read.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from conftest import contract_columns

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis import ingestion
from ql2_sixt_canada_analysis.identifiers import (
    IdentifierDtypeError,
    IdentifierError,
    IdentifierTypeConflictError,
    MissingIdentifierColumnError,
    cast_identifier_fields,
    cast_identifiers_for_raw_datasets,
    is_identifier_dtype,
    validate_identifier_dtypes,
    validate_raw_dataset_identifier_dtypes,
)
from ql2_sixt_canada_analysis.ingestion import RawDatasets, SourceSchemaError, load_raw_datasets
from ql2_sixt_canada_analysis.quality import (
    completely_blank_row_mask,
    remove_blank_rows_from_raw_datasets,
)
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    IDENTIFIER_DTYPE,
    SHARED_IDENTIFIER_COLUMNS,
    DatasetDefinition,
    DatasetKey,
)

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
DEFINITIONS = list(DATASET_DEFINITIONS.values())

# Fabricated identifiers exercising risky representations ("" = missing).
RISKY_IDS = [
    "000123",
    "000000",
    "123456789012345678901234567890",
    "SYNTHETIC-ID-001",
    "A-001-B",
    "0",
    "",
]
PRESERVED_IDS = RISKY_IDS[:-1]


def _identifiers(key: DatasetKey) -> tuple[str, ...]:
    return DATASET_DEFINITIONS[key].identifier_columns


def _filler(position: int) -> str:
    """Neutral placeholder: numeric text in even columns, plain text in odd."""
    return str(position * 10 + 7) if position % 2 == 0 else f"synthetic_text_{position}"


def _row(key: DatasetKey, identifier: str) -> list[str]:
    ids = set(_identifiers(key))
    return [identifier if c in ids else _filler(i) for i, c in enumerate(contract_columns(key))]


def _write(directory: Path, key: DatasetKey, lines: list[str]) -> Path:
    path = directory / f"synthetic_{key}.csv"
    text = ",".join(contract_columns(key)) + "\n" + "".join(line + "\n" for line in lines)
    path.write_bytes(text.encode("utf-8"))  # explicit "\n" on every platform
    return path


@pytest.fixture
def id_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "raw"
    directory.mkdir()
    for key in DatasetKey:
        _write(directory, key, [",".join(_row(key, v)) for v in RISKY_IDS])
    return directory


def _frame(key: DatasetKey, identifiers: list[object]) -> pd.DataFrame:
    """In-memory frame with contract columns; identifiers as given, others mixed."""
    columns = contract_columns(key)
    ids = set(_identifiers(key))
    data = {}
    for i, column in enumerate(columns):
        if column in ids:
            data[column] = pd.Series(identifiers, dtype=object)
        elif i % 2 == 0:
            data[column] = pd.Series([float(i)] * len(identifiers), dtype="float64")
        else:
            data[column] = pd.Series([f"t{i}"] * len(identifiers), dtype=object)
    return pd.DataFrame(data, columns=list(columns))


# ------------------------------------------------------- centralized definitions


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: str(d.key))
def test_identifier_collections_are_immutable_unique_known_strings(definition: DatasetDefinition) -> None:
    ids = definition.identifier_columns
    assert isinstance(ids, tuple) and ids, "both datasets carry an identifier"
    assert all(isinstance(c, str) and c.strip() for c in ids)
    assert len(set(ids)) == len(ids)
    assert set(ids) <= set(definition.columns)
    assert ids == tuple(c for c in definition.columns if c in ids), "contract order"
    assert len(ids) < len(definition.columns), "not every column is an identifier"
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.identifier_columns = ()  # type: ignore[misc]


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: str(d.key))
def test_identifier_dtype_mapping_is_read_only_and_safe(definition: DatasetDefinition) -> None:
    mapping = definition.identifier_dtypes
    assert tuple(mapping) == definition.identifier_columns
    assert all(is_identifier_dtype(d) for d in mapping.values())
    with pytest.raises(TypeError):
        mapping["synthetic"] = IDENTIFIER_DTYPE  # type: ignore[index]


def test_identifier_dtype_is_nullable_string() -> None:
    assert isinstance(IDENTIFIER_DTYPE, pd.StringDtype)
    assert IDENTIFIER_DTYPE.na_value is pd.NA


def test_shared_identifiers_use_the_same_dtype_in_both_datasets() -> None:
    assert SHARED_IDENTIFIER_COLUMNS, "jobs and cars share the scrape-job identity"
    for column in SHARED_IDENTIFIER_COLUMNS:
        jobs_dtype = DATASET_DEFINITIONS[JOBS].identifier_dtypes[column]
        cars_dtype = DATASET_DEFINITIONS[CARS].identifier_dtypes[column]
        assert jobs_dtype == cars_dtype


@pytest.mark.parametrize(
    "identifiers",
    [("a", "a"), ("missing",), ("",), ["a"], ("b", "a")],
    ids=["duplicate", "unknown", "empty", "list", "out-of-order"],
)
def test_invalid_identifier_definitions_are_rejected(identifiers) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        DatasetDefinition(key=JOBS, filename_tokens=("x",), columns=("a", "b"),
                          identifier_columns=identifiers)


# ---------------------------------------------------- ingestion-time preservation


@pytest.mark.parametrize("key", list(DatasetKey))
def test_risky_identifiers_are_preserved_exactly(id_dir: Path, key: DatasetKey) -> None:
    frame = getattr(load_raw_datasets(id_dir), key.value)
    for column in _identifiers(key):
        series = frame[column]
        assert is_identifier_dtype(series.dtype)
        assert series.iloc[: len(PRESERVED_IDS)].tolist() == PRESERVED_IDS  # zeros, long, alnum, hyphen, "0"
        assert series.iloc[-1] is pd.NA                                      # empty -> missing
        assert not series.isin(["nan", "None", "<NA>", "NaN", ""]).any()


def test_long_identifier_is_not_rounded(id_dir: Path) -> None:
    jobs = load_raw_datasets(id_dir).jobs
    column = _identifiers(JOBS)[0]
    assert "123456789012345678901234567890" in set(jobs[column].dropna())
    assert not jobs[column].dropna().str.contains("e+", regex=False).any()


def test_non_identifier_columns_keep_normal_inference(id_dir: Path) -> None:
    datasets = load_raw_datasets(id_dir)
    for key, frame in ((JOBS, datasets.jobs), (CARS, datasets.cars)):
        # Oracle: plain pandas with the same blank-line policy, no dtype mapping.
        oracle = pd.read_csv(id_dir / f"synthetic_{key}.csv", **ingestion.RAW_CSV_READ_DEFAULTS)
        others = [c for c in contract_columns(key) if c not in _identifiers(key)]
        pd.testing.assert_frame_equal(frame[others], oracle[others])
        numeric = [c for i, c in enumerate(contract_columns(key)) if c in others and i % 2 == 0]
        assert numeric and all(pd.api.types.is_numeric_dtype(frame[c]) for c in numeric)
        assert not all(is_identifier_dtype(frame[c].dtype) for c in others)


def test_each_dataset_uses_its_own_identifier_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Give each dataset a different (synthetic) identifier choice; each frame
    # must follow its own definition only.
    jobs_cols, cars_cols = contract_columns(JOBS), contract_columns(CARS)
    patched = MappingProxyType({
        JOBS: dataclasses.replace(DATASET_DEFINITIONS[JOBS], identifier_columns=(jobs_cols[-1],)),
        CARS: dataclasses.replace(DATASET_DEFINITIONS[CARS], identifier_columns=(cars_cols[-2],)),
    })
    monkeypatch.setattr(ingestion, "DATASET_DEFINITIONS", patched)
    for key, cols in ((JOBS, jobs_cols), (CARS, cars_cols)):
        _write(tmp_path, key, [",".join(["000123"] * len(cols))])
    datasets = load_raw_datasets(tmp_path)
    assert datasets.jobs[jobs_cols[-1]].tolist() == ["000123"]
    assert is_identifier_dtype(datasets.cars[cars_cols[-2]].dtype)
    assert datasets.cars[cars_cols[-2]].tolist() == ["000123"]
    assert pd.api.types.is_numeric_dtype(datasets.jobs[jobs_cols[0]])  # not an identifier there


def test_shared_identifier_is_consistent_across_datasets(id_dir: Path) -> None:
    datasets = load_raw_datasets(id_dir)
    for column in SHARED_IDENTIFIER_COLUMNS:
        assert datasets.jobs[column].dtype == datasets.cars[column].dtype
        pd.testing.assert_series_equal(datasets.jobs[column], datasets.cars[column])


def test_python_engine_also_preserves_identifiers(id_dir: Path) -> None:
    jobs = load_raw_datasets(id_dir, read_csv_options={"engine": "python"}).jobs
    assert jobs[_identifiers(JOBS)[0]].iloc[:2].tolist() == ["000123", "000000"]


def test_pyarrow_dtype_backend_still_preserves_identifiers(id_dir: Path) -> None:
    jobs = load_raw_datasets(id_dir, read_csv_options={"dtype_backend": "pyarrow"}).jobs
    column = _identifiers(JOBS)[0]
    assert is_identifier_dtype(jobs[column].dtype)
    assert jobs[column].iloc[:2].tolist() == ["000123", "000000"]


# ------------------------------------------------- blank-row interaction


def test_blank_rows_survive_typing_and_are_removed(tmp_path: Path) -> None:
    for key in DatasetKey:
        n = len(contract_columns(key))
        _write(tmp_path, key, [
            ",".join(_row(key, "000123")),
            "",                                   # empty physical line
            "," * (n - 1),                        # delimiter-only line
            ",".join(_row(key, "")),              # missing id, other fields set
            ",".join(_row(key, "SYNTHETIC-ID-001")),
        ])
    raw = load_raw_datasets(tmp_path)
    for key in DatasetKey:
        frame = getattr(raw, key.value)
        column = _identifiers(key)[0]
        assert len(frame) == 5, "blank physical lines remain observable"
        assert frame[column].iloc[1] is pd.NA and frame[column].iloc[2] is pd.NA
        assert completely_blank_row_mask(frame).tolist() == [False, True, True, False, False]

    results = remove_blank_rows_from_raw_datasets(raw)
    validate_raw_dataset_identifier_dtypes(results.cleaned)  # types survive cleaning
    for key, result in results.by_key.items():
        column = _identifiers(key)[0]
        assert result.removed_blank_row_count == 2
        assert list(result.cleaned.index) == [0, 3, 4]  # source order kept
        values = result.cleaned[column].tolist()
        assert values[0] == "000123" and values[1] is pd.NA and values[2] == "SYNTHETIC-ID-001"


def test_post_load_cast_does_not_hide_blank_rows() -> None:
    frame = _frame(JOBS, ["000123", None])
    frame.iloc[1] = None  # completely blank row
    cast = cast_identifier_fields(frame, DATASET_DEFINITIONS[JOBS])
    assert completely_blank_row_mask(cast).tolist() == [False, True]


# ------------------------------------------------------------ post-load casting


@pytest.mark.parametrize("key", list(DatasetKey))
def test_cast_changes_only_identifier_columns(key: DatasetKey) -> None:
    definition = DATASET_DEFINITIONS[key]
    source = _frame(key, ["000123", None, np.nan, 0, "A-001-B"])
    source.index = pd.Index([50, 40, 30, 20, 10], name="synthetic_label")
    snapshot = source.copy(deep=True)

    cast = cast_identifier_fields(source, definition)

    pd.testing.assert_frame_equal(source, snapshot)                 # not mutated
    assert cast is not source
    assert list(cast.columns) == list(source.columns)               # column order
    assert cast.index.equals(source.index)                          # index + row order
    for column in definition.identifier_columns:
        assert is_identifier_dtype(cast[column].dtype)
        values = cast[column].tolist()
        assert values[0] == "000123" and values[3] == "0" and values[4] == "A-001-B"
        assert values[1] is pd.NA and values[2] is pd.NA            # missing stays missing
    others = [c for c in source.columns if c not in definition.identifier_columns]
    pd.testing.assert_frame_equal(cast[others], source[others])     # values and dtypes


def test_cast_is_idempotent() -> None:
    definition = DATASET_DEFINITIONS[CARS]
    once = cast_identifier_fields(_frame(CARS, ["000123", None]), definition)
    pd.testing.assert_frame_equal(cast_identifier_fields(once, definition), once)


def test_cast_cannot_restore_representation_lost_earlier() -> None:
    # Documented limitation: a frame already parsed numerically has lost zeros.
    definition = DATASET_DEFINITIONS[JOBS]
    column = definition.identifier_columns[0]
    lost = _frame(JOBS, [123, None]).astype({column: "float64"})
    assert cast_identifier_fields(lost, definition)[column].iloc[0] != "000123"


def test_cast_missing_identifier_column_raises_schema_error() -> None:
    definition = DATASET_DEFINITIONS[JOBS]
    frame = _frame(JOBS, ["000123"]).drop(columns=list(definition.identifier_columns))
    with pytest.raises(MissingIdentifierColumnError) as info:
        cast_identifier_fields(frame, definition)
    assert info.value.role == JOBS
    assert info.value.columns == definition.identifier_columns
    assert isinstance(info.value, (IdentifierError, KeyError))


@pytest.mark.parametrize("key", list(DatasetKey))
def test_cast_handles_empty_frame_with_expected_columns(key: DatasetKey) -> None:
    empty = pd.DataFrame(columns=list(contract_columns(key)))
    cast = cast_identifier_fields(empty, DATASET_DEFINITIONS[key])
    assert cast.empty and list(cast.columns) == list(contract_columns(key))
    validate_identifier_dtypes(cast, DATASET_DEFINITIONS[key])


def test_cast_for_raw_datasets_uses_each_definition() -> None:
    raw = RawDatasets(jobs=_frame(JOBS, ["000123"]), cars=_frame(CARS, ["000000"]))
    cast = cast_identifiers_for_raw_datasets(raw)
    assert isinstance(cast, RawDatasets) and cast.jobs is not raw.jobs
    validate_raw_dataset_identifier_dtypes(cast)
    with pytest.raises(IdentifierDtypeError):
        validate_raw_dataset_identifier_dtypes(raw)  # originals untouched


@pytest.mark.parametrize("bad", [None, [], "synthetic"])
def test_cast_rejects_invalid_arguments(bad: object) -> None:
    with pytest.raises(TypeError):
        cast_identifier_fields(bad, DATASET_DEFINITIONS[JOBS])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cast_identifier_fields(_frame(JOBS, ["x"]), bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        cast_identifiers_for_raw_datasets(bad)  # type: ignore[arg-type]


# --------------------------------------------------------- read-option conflicts


@pytest.mark.parametrize("safe", ["string", "string[python]", pd.StringDtype(), IDENTIFIER_DTYPE])
def test_safe_identifier_dtype_from_caller_is_accepted(id_dir: Path, safe: object) -> None:
    dtype = {c: safe for c in _identifiers(JOBS)}
    jobs = load_raw_datasets(id_dir, read_csv_options={"dtype": dtype}).jobs
    assert jobs[_identifiers(JOBS)[0]].iloc[0] == "000123"
    validate_identifier_dtypes(jobs, DATASET_DEFINITIONS[JOBS])


UNSAFE_OPTIONS = [
    pytest.param(lambda c: {"dtype": {c: "int64"}}, "dtype", id="dtype-int"),
    pytest.param(lambda c: {"dtype": {c: float}}, "dtype", id="dtype-float"),
    pytest.param(lambda c: {"dtype": {c: object}}, "dtype", id="dtype-object"),
    pytest.param(lambda c: {"dtype": {c: str}}, "dtype", id="dtype-str-nan-backed"),
    pytest.param(lambda c: {"dtype": {c: "Int64"}}, "dtype", id="dtype-nullable-int"),
    pytest.param(lambda c: {"dtype": {contract_columns(JOBS).index(c): "int64"}}, "dtype",
                 id="dtype-positional-key"),
    pytest.param(lambda c: {"dtype": int}, "dtype", id="dtype-scalar-int"),
    pytest.param(lambda c: {"converters": {c: lambda v: v.lstrip("0")}}, "converters", id="converter"),
    pytest.param(lambda c: {"parse_dates": [c]}, "parse_dates", id="parse-dates"),
    pytest.param(lambda c: {"engine": "pyarrow"}, "engine", id="pyarrow-engine"),
]


@pytest.mark.parametrize(("make", "option"), UNSAFE_OPTIONS)
def test_unsafe_identifier_overrides_are_rejected(id_dir: Path, make, option: str) -> None:  # type: ignore[no-untyped-def]
    options = make(_identifiers(JOBS)[0])
    with pytest.raises(IdentifierTypeConflictError) as info:
        load_raw_datasets(id_dir, read_csv_options=options)
    assert info.value.option == option
    assert info.value.role in set(DatasetKey)
    assert set(info.value.columns) <= set(_identifiers(info.value.role))
    message = str(info.value)
    assert not any(v in message for v in PRESERVED_IDS if len(v) > 1), "message leaks a value"
    assert str(id_dir) not in message
    assert isinstance(info.value, ValueError)


def test_positional_converter_for_identifier_is_rejected(id_dir: Path) -> None:
    position = contract_columns(JOBS).index(_identifiers(JOBS)[0])
    with pytest.raises(IdentifierTypeConflictError):
        load_raw_datasets(id_dir, read_csv_options={"converters": {position: str}})


def test_conflicts_are_detected_before_any_file_is_read(
    id_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingestion.pd, "read_csv", lambda *a, **k: pytest.fail("file read"))
    with pytest.raises(IdentifierTypeConflictError):
        load_raw_datasets(id_dir, read_csv_options={"dtype": {_identifiers(CARS)[0]: "float64"}})


def test_caller_dtype_for_non_identifier_is_preserved(id_dir: Path) -> None:
    other = next(c for i, c in enumerate(contract_columns(JOBS))
                 if c not in _identifiers(JOBS) and i % 2 == 0)
    jobs = load_raw_datasets(id_dir, read_csv_options={"dtype": {other: "string"}}).jobs
    assert is_identifier_dtype(jobs[other].dtype)  # caller's choice honoured
    validate_identifier_dtypes(jobs, DATASET_DEFINITIONS[JOBS])


def test_text_scalar_dtype_keeps_identifier_guarantee(id_dir: Path) -> None:
    jobs = load_raw_datasets(id_dir, read_csv_options={"dtype": str}).jobs
    column = _identifiers(JOBS)[0]
    assert is_identifier_dtype(jobs[column].dtype)
    assert jobs[column].iloc[-1] is pd.NA


def test_caller_options_are_not_mutated(id_dir: Path) -> None:
    first, second = [c for c in contract_columns(JOBS) if c not in _identifiers(JOBS)][:2]
    dtype = {first: "string", _identifiers(JOBS)[0]: "string"}
    converters = {second: str}
    options = {"dtype": dtype, "converters": converters, "na_values": ["synthetic_na"]}
    snapshot = copy.deepcopy({k: v for k, v in options.items() if k != "converters"})
    load_raw_datasets(id_dir, read_csv_options=options)
    assert options["dtype"] is dtype and dtype == snapshot["dtype"]
    assert options["converters"] is converters and converters == {second: str}
    assert options["na_values"] == snapshot["na_values"] and set(options) == {"dtype", "converters", "na_values"}


# --------------------------------------------------------------- type validation


def test_validator_passes_silently_for_string_identifiers(
    id_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    datasets = load_raw_datasets(id_dir)
    snapshot = datasets.jobs.copy(deep=True)
    assert validate_identifier_dtypes(datasets.jobs, DATASET_DEFINITIONS[JOBS]) is None
    assert validate_raw_dataset_identifier_dtypes(datasets) is None
    pd.testing.assert_frame_equal(datasets.jobs, snapshot)  # not modified
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("dtype", ["int64", "float64", object, "Int64", "str"],
                         ids=["int", "float", "object", "nullable-int", "nan-backed-str"])
def test_validator_rejects_non_nullable_string_identifiers(dtype: object) -> None:
    definition = DATASET_DEFINITIONS[CARS]
    column = definition.identifier_columns[0]
    frame = cast_identifier_fields(_frame(CARS, ["123", "456"]), definition).astype({column: dtype})
    snapshot = frame.copy(deep=True)
    with pytest.raises(IdentifierDtypeError) as info:
        validate_identifier_dtypes(frame, definition)
    assert info.value.role == CARS and info.value.columns == (column,)
    assert "123" not in str(info.value) and "456" not in str(info.value)
    pd.testing.assert_frame_equal(frame, snapshot)


def test_validator_reports_missing_identifier_column() -> None:
    definition = DATASET_DEFINITIONS[JOBS]
    frame = _frame(JOBS, ["x"]).drop(columns=list(definition.identifier_columns))
    with pytest.raises(MissingIdentifierColumnError):
        validate_identifier_dtypes(frame, definition)


def test_loader_rejects_a_read_that_bypassed_the_mapping(
    id_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = pd.read_csv

    def ignore_dtype(path, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("header", "infer") is not None:
            kwargs.pop("dtype", None)  # simulate an engine ignoring the mapping
        return real(path, **kwargs)

    monkeypatch.setattr(ingestion.pd, "read_csv", ignore_dtype)
    with pytest.raises(IdentifierDtypeError):
        load_raw_datasets(id_dir)


def test_header_contract_is_still_enforced(tmp_path: Path) -> None:
    for key in DatasetKey:
        _write(tmp_path, key, [])
    (tmp_path / f"synthetic_{JOBS}.csv").write_text("synthetic_wrong\n000123\n", encoding="utf-8")
    with pytest.raises(SourceSchemaError):
        load_raw_datasets(tmp_path)


def test_package_exposes_identifier_api() -> None:
    for name in ("cast_identifier_fields", "cast_identifiers_for_raw_datasets",
                 "validate_identifier_dtypes", "validate_raw_dataset_identifier_dtypes",
                 "IDENTIFIER_DTYPE", "SHARED_IDENTIFIER_COLUMNS", "IdentifierTypeConflictError"):
        assert name in ql2_sixt_canada_analysis.__all__
        assert hasattr(ql2_sixt_canada_analysis, name)
