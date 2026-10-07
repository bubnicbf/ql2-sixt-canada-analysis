"""Tests for centralized unique-key contracts and their assessment/validation.

Every key value is fabricated (``SYNTH-JOB-001``, ``000001``, ...). Frames are
built from the centralized column, identifier and key definitions; synthetic
CSVs are written to ``tmp_path``. The proprietary files are never read.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import contract_columns

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.identifiers import (
    cast_identifier_fields,
    is_identifier_dtype,
    validate_raw_dataset_identifier_dtypes,
)
from ql2_sixt_canada_analysis.ingestion import RawDatasets, load_raw_datasets
from ql2_sixt_canada_analysis.quality import remove_blank_rows_from_raw_datasets
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    IDENTIFIER_DTYPE,
    SHARED_IDENTIFIER_COLUMNS,
    DatasetDefinition,
    DatasetKey,
    KeyConfigurationError,
)
from ql2_sixt_canada_analysis.unique_keys import (
    RawDatasetUniqueKeyReports,
    UniqueKeyReport,
    UniqueKeyViolationError,
    assess_raw_dataset_unique_keys,
    assess_unique_key,
    validate_raw_dataset_unique_keys,
    validate_unique_key,
)

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
DEFINITIONS = list(DATASET_DEFINITIONS.values())
MISSING = object()  # sentinel: write a missing value into a key component

# A synthetic two-component definition for composite-key behaviour.
COMPOSITE = DatasetDefinition(
    key=CARS, filename_tokens=("synthetic",),
    columns=("synth_parent", "synth_position", "synth_note"),
    identifier_columns=("synth_parent", "synth_position"),
    unique_key_columns=("synth_parent", "synth_position"),
)
SINGLE = DatasetDefinition(
    key=JOBS, filename_tokens=("synthetic",),
    columns=("synth_id", "synth_measure", "synth_note"),
    identifier_columns=("synth_id",),
    unique_key_columns=("synth_id",),
)


def _frame(definition: DatasetDefinition, keys: list[object]) -> pd.DataFrame:
    """Frame with ``definition.columns``; ``keys`` fill the key components.

    Each element of ``keys`` is a scalar (single-column key) or a tuple (one
    value per component). Identifier components use the nullable string
    dtype; other key components are numeric; non-key columns get neutral
    placeholders (repeating, so non-key duplication is common).
    """
    key_cols = definition.unique_key_columns
    rows = [k if isinstance(k, tuple) else (k,) for k in keys]
    data: dict[str, object] = {}
    for i, column in enumerate(definition.columns):
        if column in key_cols:
            position = key_cols.index(column)
            values = [None if r[position] is MISSING else r[position] for r in rows]
            if column in definition.identifier_columns:
                data[column] = pd.array(values, dtype=IDENTIFIER_DTYPE)
            else:
                data[column] = pd.array([np.nan if v is None else v for v in values], dtype="float64")
        elif i % 2:
            data[column] = pd.array(["synthetic_placeholder"] * len(rows), dtype=object)
        else:
            data[column] = pd.array([1.5] * len(rows), dtype="float64")
    return pd.DataFrame(data, columns=list(definition.columns))


def _key_for(definition: DatasetDefinition, n: int) -> tuple[object, ...]:
    """A fabricated complete key for ``definition`` (identifier text / numeric ordinal)."""
    return tuple(
        f"SYNTH-{definition.key.value.upper()}-{n:03d}" if c in definition.identifier_columns else float(n)
        for c in definition.unique_key_columns
    )


def _counts(report: UniqueKeyReport) -> tuple[int, ...]:
    return (report.total_row_count, report.complete_key_row_count, report.missing_key_row_count,
            report.duplicate_key_row_count, report.duplicate_key_group_count,
            report.distinct_complete_key_count)


def _check_invariants(report: UniqueKeyReport) -> None:
    assert report.total_row_count == report.complete_key_row_count + report.missing_key_row_count
    assert report.duplicate_key_row_count <= report.complete_key_row_count
    assert (report.duplicate_key_group_count == 0) == (report.duplicate_key_row_count == 0)
    assert report.is_valid == (report.is_complete and report.is_unique)
    if report.is_valid:
        assert report.missing_key_row_count == 0 and report.duplicate_key_row_count == 0


# ------------------------------------------------------- centralized definitions


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: str(d.key))
def test_key_definitions_are_non_empty_immutable_and_schema_valid(definition: DatasetDefinition) -> None:
    key = definition.unique_key_columns
    assert isinstance(key, tuple) and key
    assert all(isinstance(c, str) and c.strip() for c in key)
    assert len(set(key)) == len(key)
    assert set(key) <= set(definition.columns)
    assert len(key) < len(definition.columns), "key must not be the whole row"
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.unique_key_columns = ()  # type: ignore[misc]


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: str(d.key))
def test_key_starts_with_identifiers_and_exceptions_are_explicit(definition: DatasetDefinition) -> None:
    key = definition.unique_key_columns
    identifier_part = tuple(c for c in key if c in definition.identifier_columns)
    assert identifier_part, "every key is anchored on an identifier"
    assert definition.non_identifier_key_columns == tuple(c for c in key if c not in identifier_part)
    # At most one documented non-identifier component (the cars ordinal).
    assert len(definition.non_identifier_key_columns) <= 1


def test_jobs_key_is_its_identifier_and_cars_key_extends_the_shared_job_key() -> None:
    jobs, cars = DATASET_DEFINITIONS[JOBS], DATASET_DEFINITIONS[CARS]
    assert jobs.non_identifier_key_columns == ()
    assert jobs.unique_key_columns == SHARED_IDENTIFIER_COLUMNS
    assert cars.unique_key_columns[: len(SHARED_IDENTIFIER_COLUMNS)] == SHARED_IDENTIFIER_COLUMNS
    assert len(cars.unique_key_columns) == len(SHARED_IDENTIFIER_COLUMNS) + 1


def test_shared_key_components_have_the_same_dtype_policy() -> None:
    for column in SHARED_IDENTIFIER_COLUMNS:
        assert all(column in d.unique_key_columns for d in DEFINITIONS)
        assert DATASET_DEFINITIONS[JOBS].identifier_dtypes[column] == \
            DATASET_DEFINITIONS[CARS].identifier_dtypes[column]


@pytest.mark.parametrize(
    "key", [("a", "a"), ("missing",), ("",), ["a"], (1,)],
    ids=["duplicate", "unknown", "empty-name", "list", "non-string"],
)
def test_invalid_key_definitions_raise_configuration_error(key) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(KeyConfigurationError):
        DatasetDefinition(key=JOBS, filename_tokens=("x",), columns=("a", "b"), unique_key_columns=key)
    assert issubclass(KeyConfigurationError, ValueError)


def test_empty_key_definition_raises_configuration_error_on_assessment() -> None:
    no_key = dataclasses.replace(SINGLE, unique_key_columns=())
    with pytest.raises(KeyConfigurationError):
        assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001"]), no_key)


def test_duplicate_component_slipped_past_construction_is_rejected() -> None:
    broken = dataclasses.replace(SINGLE)
    object.__setattr__(broken, "unique_key_columns", ("synth_id", "synth_id"))
    with pytest.raises(KeyConfigurationError):
        assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001"]), broken)


def test_absent_key_column_raises_configuration_error() -> None:
    frame = _frame(COMPOSITE, [("SYNTH-JOB-001", "1")]).drop(columns=["synth_position"])
    with pytest.raises(KeyConfigurationError) as info:
        assess_unique_key(frame, COMPOSITE)
    assert info.value.columns == ("synth_position",) and info.value.role == CARS
    assert not isinstance(info.value, UniqueKeyViolationError)


# ------------------------------------------------------------ single-column key


def test_unique_complete_single_key_passes() -> None:
    report = assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001", "SYNTH-JOB-002", "000001"]), SINGLE)
    assert report.is_complete and report.is_unique and report.is_valid
    assert report.violations == ()
    assert report.distinct_complete_key_count == report.complete_key_row_count == 3
    assert report.key_columns == SINGLE.unique_key_columns and report.dataset == JOBS
    _check_invariants(report)


def test_duplicate_pair_counts_both_rows() -> None:
    report = assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001", "SYNTH-JOB-001", "SYNTH-JOB-002"]), SINGLE)
    assert not report.is_unique and report.is_complete and not report.is_valid
    assert (report.duplicate_key_row_count, report.duplicate_key_group_count) == (2, 1)
    assert report.violations == ("duplicate_key",)
    _check_invariants(report)


def test_triple_is_three_rows_one_group() -> None:
    report = assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001"] * 3 + ["SYNTH-JOB-002"]), SINGLE)
    assert (report.duplicate_key_row_count, report.duplicate_key_group_count) == (3, 1)
    assert report.distinct_complete_key_count == 2


def test_non_adjacent_duplicates_are_detected_without_reordering() -> None:
    keys = ["SYNTH-JOB-002", "SYNTH-JOB-001", "SYNTH-JOB-003", "SYNTH-JOB-002", "SYNTH-JOB-004", "SYNTH-JOB-001"]
    frame = _frame(SINGLE, keys)
    frame.index = pd.Index([9, 3, 7, 1, 5, 2])
    snapshot = frame.copy(deep=True)
    report = assess_unique_key(frame, SINGLE)
    assert (report.duplicate_key_row_count, report.duplicate_key_group_count) == (4, 2)
    pd.testing.assert_frame_equal(frame, snapshot)  # order, index, values, dtypes


# ---------------------------------------------------------------- composite key


def test_repeating_one_component_is_still_unique() -> None:
    report = assess_unique_key(
        _frame(COMPOSITE, [("SYNTH-JOB-001", "1"), ("SYNTH-JOB-001", "2"), ("SYNTH-JOB-002", "1")]),
        COMPOSITE,
    )
    assert report.is_valid and report.distinct_complete_key_count == 3


def test_repeating_full_combination_is_detected() -> None:
    report = assess_unique_key(
        _frame(COMPOSITE, [("SYNTH-JOB-001", "1"), ("SYNTH-JOB-002", "1"), ("SYNTH-JOB-001", "1")]),
        COMPOSITE,
    )
    assert (report.duplicate_key_row_count, report.duplicate_key_group_count) == (2, 1)


def test_component_order_is_deterministic() -> None:
    assert assess_unique_key(_frame(COMPOSITE, []), COMPOSITE).key_columns == COMPOSITE.unique_key_columns
    assert DATASET_DEFINITIONS[CARS].unique_key_columns == tuple(DATASET_DEFINITIONS[CARS].unique_key_columns)


@pytest.mark.parametrize(
    "pair",
    [
        (("A|B", "C"), ("A", "B|C")),
        (("A-B", "C"), ("A", "B-C")),
        (("A,B", "C"), ("A", "B,C")),
        (("A", "BC"), ("AB", "C")),
        (("A\tB", ""), ("A", "\tB")),
        (("SYNTH", "JOB-001"), ("SYNTH-JOB", "001")),
    ],
)
def test_delimiter_like_values_never_collide(pair) -> None:  # type: ignore[no-untyped-def]
    report = assess_unique_key(_frame(COMPOSITE, list(pair)), COMPOSITE)
    assert report.is_unique and report.distinct_complete_key_count == 2


# ------------------------------------------------------------------ missing keys


def test_missing_single_key_is_counted() -> None:
    report = assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001", MISSING]), SINGLE)
    assert (report.missing_key_row_count, report.complete_key_row_count) == (1, 1)
    assert not report.is_complete and report.is_unique and not report.is_valid
    assert report.violations == ("missing_key",)


@pytest.mark.parametrize("row", [("SYNTH-JOB-001", MISSING), (MISSING, "1"), (MISSING, MISSING)],
                         ids=["second", "first", "all"])
def test_partially_or_fully_missing_composite_counts_once(row) -> None:  # type: ignore[no-untyped-def]
    report = assess_unique_key(_frame(COMPOSITE, [("SYNTH-JOB-002", "1"), row]), COMPOSITE)
    assert report.missing_key_row_count == 1 and report.total_row_count == 2


def test_multiple_missing_keys_are_not_a_duplicate_group() -> None:
    report = assess_unique_key(_frame(SINGLE, [MISSING, MISSING, MISSING, "SYNTH-JOB-001"]), SINGLE)
    assert report.missing_key_row_count == 3
    assert (report.duplicate_key_row_count, report.duplicate_key_group_count) == (0, 0)
    assert report.is_unique and not report.is_valid  # missing keys still fail the contract


def test_uniqueness_is_assessed_correctly_alongside_missing_keys() -> None:
    report = assess_unique_key(
        _frame(COMPOSITE, [("SYNTH-JOB-001", "1"), (MISSING, "1"), ("SYNTH-JOB-001", "1"), ("SYNTH-JOB-001", MISSING)]),
        COMPOSITE,
    )
    assert _counts(report) == (4, 2, 2, 2, 1, 1)
    assert report.violations == ("missing_key", "duplicate_key")


@pytest.mark.parametrize("missing", [pd.NA, None, np.nan], ids=["NA", "None", "nan"])
def test_pandas_missing_values_are_missing(missing: object) -> None:
    frame = pd.DataFrame({"synth_id": pd.array([missing, "x"], dtype=object),
                          "synth_measure": [1, 2], "synth_note": ["a", "b"]})
    assert assess_unique_key(frame, SINGLE).missing_key_row_count == 1


# --------------------------------------------------------- identifier semantics


def test_leading_zeros_are_distinct_from_unpadded_values() -> None:
    report = assess_unique_key(_frame(SINGLE, ["000001", "1", "01", "000002", "2"]), SINGLE)
    assert report.is_valid and report.distinct_complete_key_count == 5


def test_long_identifiers_compare_exactly() -> None:
    long_a = "123456789012345678901234567890"
    long_b = "123456789012345678901234567891"  # equal as a float, distinct as text
    assert float(long_a) == float(long_b)
    assert assess_unique_key(_frame(SINGLE, [long_a, long_b]), SINGLE).is_valid
    assert not assess_unique_key(_frame(SINGLE, [long_a, long_a]), SINGLE).is_unique


def test_alphanumeric_identifiers_validate() -> None:
    assert assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001", "SYNTH-CAR-001", "A-001-B"]), SINGLE).is_valid


@pytest.mark.parametrize("value", ["0", "False", "N/A", "None", "null", "nan", "", " "])
def test_text_that_looks_missing_is_a_value(value: str) -> None:
    report = assess_unique_key(_frame(SINGLE, [value, "SYNTH-JOB-001"]), SINGLE)
    assert report.is_complete and report.is_valid
    assert not assess_unique_key(_frame(SINGLE, [value, value]), SINGLE).is_unique


def test_keys_are_not_stripped_before_comparison() -> None:
    assert assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001", " SYNTH-JOB-001"]), SINGLE).is_valid


# --------------------------------------------------- real contracts end to end


def _write(directory: Path, key: DatasetKey, lines: list[str]) -> None:
    text = ",".join(contract_columns(key)) + "\n" + "".join(line + "\n" for line in lines)
    (directory / f"synthetic_{key}.csv").write_bytes(text.encode("utf-8"))


def _line(key: DatasetKey, values: dict[str, str]) -> str:
    return ",".join(values.get(c, f"synthetic_{i}") for i, c in enumerate(contract_columns(key)))


def _keyed(key: DatasetKey, n: int | None, *, ordinal: str | None = None) -> str:
    """CSV line whose key components carry fabricated values (``None`` -> empty)."""
    definition = DATASET_DEFINITIONS[key]
    values = {}
    for c in definition.unique_key_columns:
        if c in definition.identifier_columns:
            values[c] = "" if n is None else f"SYNTH-JOB-{n:03d}"
        else:
            values[c] = ordinal if ordinal is not None else "1"
    return _line(key, values)


def test_blank_rows_are_removed_before_key_assessment(tmp_path: Path) -> None:
    for key in DatasetKey:
        n = len(contract_columns(key))
        _write(tmp_path, key, [
            _keyed(key, 1),
            "",                      # completely blank physical line
            "," * (n - 1),           # delimiter-only line (also completely blank)
            _keyed(key, None),       # partially populated, missing key
            _keyed(key, 2),
        ])
    raw = load_raw_datasets(tmp_path)
    blank = remove_blank_rows_from_raw_datasets(raw)
    validate_raw_dataset_identifier_dtypes(blank.cleaned)
    reports = assess_raw_dataset_unique_keys(blank.cleaned)
    for key, report in reports.by_key.items():
        assert blank.by_key[key].removed_blank_row_count == 2      # counted by blank-row control
        assert report.total_row_count == blank.by_key[key].retained_row_count == 3
        assert report.missing_key_row_count == 1                    # not double-counted
        assert report.is_unique and not report.is_valid
    assert reports.total_missing_key_row_count == 2
    # Assessing raw (uncleaned) data would double-count the blank lines:
    assert assess_raw_dataset_unique_keys(raw).jobs.missing_key_row_count == 3


def test_contracts_detect_duplicates_from_csv(tmp_path: Path) -> None:
    _write(tmp_path, JOBS, [_keyed(JOBS, 1), _keyed(JOBS, 2), _keyed(JOBS, 1)])
    _write(tmp_path, CARS, [_keyed(CARS, 1, ordinal="1"), _keyed(CARS, 1, ordinal="2"),
                            _keyed(CARS, 2, ordinal="1")])
    reports = assess_raw_dataset_unique_keys(load_raw_datasets(tmp_path))
    assert not reports.jobs.is_valid and reports.jobs.duplicate_key_row_count == 2
    assert reports.cars.is_valid   # same job, different ordinal: distinct offers
    assert not reports.all_valid


def test_leading_zero_csv_keys_survive_ingestion_and_stay_distinct(tmp_path: Path) -> None:
    jobs_key = DATASET_DEFINITIONS[JOBS].unique_key_columns[0]
    _write(tmp_path, JOBS, [_line(JOBS, {jobs_key: "000001"}), _line(JOBS, {jobs_key: "1"})])
    _write(tmp_path, CARS, [_keyed(CARS, 1)])
    jobs = load_raw_datasets(tmp_path).jobs
    assert is_identifier_dtype(jobs[jobs_key].dtype)
    assert assess_unique_key(jobs, DATASET_DEFINITIONS[JOBS]).is_valid


# ------------------------------------------------------------- report invariants


def _synthetic_raw() -> RawDatasets:
    jobs = DATASET_DEFINITIONS[JOBS]
    cars = DATASET_DEFINITIONS[CARS]
    return RawDatasets(
        jobs=_frame(jobs, [_key_for(jobs, 1), _key_for(jobs, 1), tuple(MISSING for _ in jobs.unique_key_columns)]),
        cars=_frame(cars, [_key_for(cars, 1), _key_for(cars, 2), _key_for(cars, 2), _key_for(cars, 2)]),
    )


def test_combined_reports_are_distinct_and_aggregate_correctly() -> None:
    raw = _synthetic_raw()
    reports = assess_raw_dataset_unique_keys(raw)
    assert isinstance(reports, RawDatasetUniqueKeyReports)
    assert reports.jobs.dataset == JOBS and reports.cars.dataset == CARS
    assert reports.jobs.key_columns == DATASET_DEFINITIONS[JOBS].unique_key_columns
    assert reports.cars.key_columns == DATASET_DEFINITIONS[CARS].unique_key_columns
    assert _counts(reports.jobs) == (3, 2, 1, 2, 1, 1)
    assert _counts(reports.cars) == (4, 4, 0, 3, 1, 2)
    assert reports.total_missing_key_row_count == 1 == sum(r.missing_key_row_count for r in reports.by_key.values())
    assert reports.total_duplicate_key_row_count == 5 == sum(r.duplicate_key_row_count for r in reports.by_key.values())
    assert not reports.all_valid
    for report in reports.by_key.values():
        _check_invariants(report)


def test_report_is_frozen_and_holds_no_data() -> None:
    report = assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001", "SYNTH-JOB-001"]), SINGLE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.duplicate_key_row_count = 0  # type: ignore[misc]
    fields = {f.name: getattr(report, f.name) for f in dataclasses.fields(report)}
    assert not any(isinstance(v, (pd.DataFrame, pd.Series, list, set, dict)) for v in fields.values())
    assert "SYNTH-JOB-001" not in repr(report)


def test_inconsistent_report_is_a_programmer_error() -> None:
    with pytest.raises(AssertionError):
        UniqueKeyReport(dataset=JOBS, key_columns=("k",), total_row_count=2, complete_key_row_count=2,
                        missing_key_row_count=1, duplicate_key_row_count=0,
                        duplicate_key_group_count=0, distinct_complete_key_count=2)


# --------------------------------------------------------------------- edge cases


@pytest.mark.parametrize("definition", [SINGLE, COMPOSITE, *DEFINITIONS], ids=lambda d: str(d.unique_key_columns))
def test_empty_frame_is_vacuously_valid(definition: DatasetDefinition) -> None:
    report = assess_unique_key(pd.DataFrame(columns=list(definition.columns)), definition)
    assert _counts(report) == (0, 0, 0, 0, 0, 0)
    assert report.is_complete and report.is_unique and report.is_valid
    validate_unique_key(pd.DataFrame(columns=list(definition.columns)), definition)


def test_single_complete_row_passes() -> None:
    assert validate_unique_key(_frame(SINGLE, ["SYNTH-JOB-001"]), SINGLE).is_valid


def test_duplicate_non_key_values_with_unique_keys_pass() -> None:
    frame = _frame(SINGLE, ["SYNTH-JOB-001", "SYNTH-JOB-002"])
    assert frame.drop(columns=["synth_id"]).duplicated().any()
    assert validate_unique_key(frame, SINGLE).is_valid


def test_exact_duplicate_rows_are_detected_not_removed() -> None:
    frame = _frame(SINGLE, ["SYNTH-JOB-001", "SYNTH-JOB-001"])
    assert frame.duplicated(keep=False).all()  # the two rows are identical
    snapshot = frame.copy(deep=True)
    report = assess_unique_key(frame, SINGLE)
    assert report.duplicate_key_row_count == 2 and len(frame) == 2
    pd.testing.assert_frame_equal(frame, snapshot)


def test_assessment_is_idempotent_and_does_not_mutate() -> None:
    raw = _synthetic_raw()
    snapshots = (raw.jobs.copy(deep=True), raw.cars.copy(deep=True))
    first = assess_raw_dataset_unique_keys(raw)
    second = assess_raw_dataset_unique_keys(raw)
    assert first == second
    for frame, snapshot in zip((raw.jobs, raw.cars), snapshots, strict=True):
        pd.testing.assert_frame_equal(frame, snapshot)
        assert frame.index.equals(snapshot.index) and frame.dtypes.equals(snapshot.dtypes)


def test_assessment_works_on_cast_frames() -> None:
    cars = DATASET_DEFINITIONS[CARS]
    frame = cast_identifier_fields(_frame(cars, [_key_for(cars, 1), _key_for(cars, 2)]), cars)
    assert assess_unique_key(frame, cars).is_valid


# ------------------------------------------------------------ strict validation


def test_assessment_returns_reports_for_source_violations() -> None:
    report = assess_unique_key(_frame(SINGLE, ["SYNTH-JOB-001", "SYNTH-JOB-001", MISSING]), SINGLE)
    assert isinstance(report, UniqueKeyReport) and not report.is_valid


def test_strict_validation_raises_safe_typed_error() -> None:
    secret = "SYNTH-JOB-SECRET-123456789"
    frame = _frame(SINGLE, [secret, secret, MISSING])
    with pytest.raises(UniqueKeyViolationError) as info:
        validate_unique_key(frame, SINGLE)
    error = info.value
    assert error.roles == (JOBS,)
    assert error.reports[0] == assess_unique_key(frame, SINGLE)
    message = str(error)
    assert "'jobs'" in message and "missing_key" in message and "duplicate_key" in message
    assert secret not in message and "synth_id" not in message
    assert not any(ch.isdigit() for ch in message), "no counts or values in the message"
    assert not isinstance(error, (KeyConfigurationError, ValueError))


def test_strict_validation_of_both_datasets_lists_every_failure() -> None:
    with pytest.raises(UniqueKeyViolationError) as info:
        validate_raw_dataset_unique_keys(_synthetic_raw())
    assert info.value.roles == (JOBS, CARS)
    jobs, cars = DATASET_DEFINITIONS[JOBS], DATASET_DEFINITIONS[CARS]
    valid = RawDatasets(jobs=_frame(jobs, [_key_for(jobs, 1)]), cars=_frame(cars, [_key_for(cars, 1)]))
    assert validate_raw_dataset_unique_keys(valid).all_valid


@pytest.mark.parametrize("bad", [None, [], "synthetic"])
def test_invalid_argument_types_raise_type_error(bad: object) -> None:
    with pytest.raises(TypeError):
        assess_unique_key(bad, SINGLE)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess_unique_key(_frame(SINGLE, ["x"]), bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess_raw_dataset_unique_keys(bad)  # type: ignore[arg-type]


def test_package_exposes_unique_key_api() -> None:
    for name in ("assess_unique_key", "assess_raw_dataset_unique_keys", "validate_unique_key",
                 "validate_raw_dataset_unique_keys", "UniqueKeyReport", "RawDatasetUniqueKeyReports",
                 "UniqueKeyViolationError", "KeyConfigurationError"):
        assert name in ql2_sixt_canada_analysis.__all__ and hasattr(ql2_sixt_canada_analysis, name)


def test_analysis_definitions_key_on_the_derived_columns() -> None:
    from conftest import link

    from ql2_sixt_canada_analysis.ingestion import RawDatasets
    from ql2_sixt_canada_analysis.schemas import (
        ANALYSIS_DATASET_DEFINITIONS, CARS_DEFINITION, JOB_LINKAGE_KEY_COLUMN, JOBS_DEFINITION,
        OFFER_POSITION_KEY_COLUMN,
    )
    from ql2_sixt_canada_analysis.unique_keys import assess_raw_dataset_unique_keys

    jobs = pd.DataFrame([{c: ("0007" if c == "job_id" else "SYNTH") for c in JOBS_DEFINITION.columns}]).astype(
        dict(JOBS_DEFINITION.identifier_dtypes))
    cars = pd.DataFrame([{c: ("0007.0" if c == "job_id" else p if c == "row_index" else "SYNTH")
                          for c in CARS_DEFINITION.columns} for p in ("0.0", "0")]).astype(
        dict(CARS_DEFINITION.identifier_dtypes))
    raw = assess_raw_dataset_unique_keys(RawDatasets(jobs=jobs, cars=cars))
    assert raw.cars.is_valid                             # raw text: "0.0" and "0" differ, yet ...
    result = link(jobs, cars)
    analysis = assess_raw_dataset_unique_keys(result.datasets(), ANALYSIS_DATASET_DEFINITIONS)
    assert analysis.jobs.is_valid and not analysis.cars.is_valid    # ... the derived positions collide
    assert ANALYSIS_DATASET_DEFINITIONS[DatasetKey.CARS].unique_key_columns == (JOB_LINKAGE_KEY_COLUMN,
                                                                                OFFER_POSITION_KEY_COLUMN)
