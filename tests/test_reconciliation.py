"""Tests for the job-to-detail relationship and count reconciliation.

All identifiers and counts are fabricated (``SYNTH-JOB-001``, ``000001``,
0/1/2/3). Frames are built from the centralized column, identifier, key and
relationship definitions; CSVs are written to ``tmp_path`` only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from conftest import contract_columns

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.identifiers import is_identifier_dtype, validate_raw_dataset_identifier_dtypes
from ql2_sixt_canada_analysis.ingestion import load_raw_datasets
from ql2_sixt_canada_analysis.quality import remove_blank_rows_from_raw_datasets
from ql2_sixt_canada_analysis.reconciliation import (
    JobDetailReconciliationError,
    JobDetailReconciliationReport,
    ReconciliationPreconditionError,
    assess_job_detail_reconciliation,
    validate_job_detail_reconciliation,
)
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    IDENTIFIER_DTYPE,
    JOB_DETAIL_RELATIONSHIP,
    DatasetDefinition,
    DatasetKey,
    JobDetailRelationshipDefinition,
    RelationshipConfigurationError,
)
from ql2_sixt_canada_analysis.unique_keys import assess_raw_dataset_unique_keys

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
REL = JOB_DETAIL_RELATIONSHIP
MISSING = object()

# Synthetic composite relationship (two-component key) for composite tests.
_SYNTH_PARENT = DatasetDefinition(
    key=JOBS, filename_tokens=("synthparent",),
    columns=("synth_a", "synth_b", "synth_expected", "synth_note"),
    identifier_columns=("synth_a", "synth_b"), unique_key_columns=("synth_a", "synth_b"),
)
_SYNTH_DETAIL = DatasetDefinition(
    key=CARS, filename_tokens=("synthdetail",),
    columns=("synth_note", "synth_fa", "synth_fb"),
    identifier_columns=("synth_fa", "synth_fb"), unique_key_columns=(),
)
COMPOSITE = JobDetailRelationshipDefinition(
    parent=JOBS, detail=CARS,
    parent_key_columns=("synth_a", "synth_b"), detail_key_columns=("synth_fa", "synth_fb"),
    expected_detail_count_column="synth_expected",
    definitions=MappingProxyType({JOBS: _SYNTH_PARENT, CARS: _SYNTH_DETAIL}),
)


def _jobs(keys: list[object], expected: list[object], rel: JobDetailRelationshipDefinition = REL,
          count_dtype: object = None) -> pd.DataFrame:
    """Parent frame: keys -> parent key columns, expected -> count column, placeholders elsewhere."""
    definition = rel.parent_definition
    rows = [k if isinstance(k, tuple) else (k,) for k in keys]
    data: dict[str, object] = {}
    for i, column in enumerate(definition.columns):
        if column in rel.parent_key_columns:
            pos = rel.parent_key_columns.index(column)
            data[column] = pd.array([None if r[pos] is MISSING else r[pos] for r in rows], dtype=IDENTIFIER_DTYPE)
        elif column in rel.expected_detail_count_columns:
            # Every declared-count column gets the same values (they must agree);
            # tests that need them to differ build frames explicitly.
            values = [None if v is MISSING else v for v in expected]
            data[column] = pd.Series(values, dtype=count_dtype if count_dtype is not None else
                                     ("int64" if all(type(v) is int for v in values) else object))
        elif column in definition.identifier_columns:
            data[column] = pd.array([f"SYNTH-OTHER-{i}"] * len(rows), dtype=IDENTIFIER_DTYPE)
        else:
            data[column] = pd.Series(["synthetic_placeholder"] * len(rows), dtype=object)
    return pd.DataFrame(data, columns=list(definition.columns))


def _cars(keys: list[object], rel: JobDetailRelationshipDefinition = REL) -> pd.DataFrame:
    """Detail frame: keys -> foreign-key columns, neutral non-blank placeholders elsewhere."""
    definition = rel.detail_definition
    rows = [k if isinstance(k, tuple) else (k,) for k in keys]
    data: dict[str, object] = {}
    for i, column in enumerate(definition.columns):
        if column in rel.detail_key_columns:
            pos = rel.detail_key_columns.index(column)
            data[column] = pd.array([None if r[pos] is MISSING else r[pos] for r in rows], dtype=IDENTIFIER_DTYPE)
        elif column in definition.identifier_columns:
            data[column] = pd.array([f"SYNTH-DETAIL-{n:03d}" for n in range(len(rows))], dtype=IDENTIFIER_DTYPE)
        else:
            data[column] = pd.Series([float(i)] * len(rows), dtype="float64")
    return pd.DataFrame(data, columns=list(definition.columns))


def _assess(jobs: pd.DataFrame, cars: pd.DataFrame, rel: JobDetailRelationshipDefinition = REL) -> JobDetailReconciliationReport:
    report = assess_job_detail_reconciliation(jobs, cars, rel)
    _check_invariants(report)
    return report


def _check_invariants(r: JobDetailReconciliationReport) -> None:
    assert r.job_count == r.valid_expected_count_job_count + r.missing_expected_count_job_count + r.invalid_expected_count_job_count
    assert r.valid_expected_count_job_count == r.matched_job_count + r.under_counted_job_count + r.over_counted_job_count
    assert r.detail_row_count == r.linked_detail_row_count + r.missing_link_detail_row_count + r.orphan_detail_row_count
    assert 0 <= r.distinct_orphan_key_count <= r.orphan_detail_row_count
    assert r.jobs_without_linked_details_count <= r.job_count
    assert r.absolute_discrepancy_total >= 0 and abs(r.net_discrepancy) <= r.absolute_discrepancy_total
    if r.is_reconciled:
        assert r.valid_expected_detail_total == r.detail_row_count == r.linked_detail_row_count
        assert r.violations == ()
    else:
        assert r.violations


J1, J2, J3 = "SYNTH-JOB-001", "SYNTH-JOB-002", "SYNTH-JOB-003"


# ----------------------------------------------------- relationship configuration


def test_relationship_datasets_and_registry() -> None:
    assert (REL.parent, REL.detail) == (JOBS, CARS)
    assert REL.parent_definition is DATASET_DEFINITIONS[JOBS]
    assert REL.detail_definition is DATASET_DEFINITIONS[CARS]


def test_relationship_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        REL.parent_key_columns = ()  # type: ignore[misc]
    assert isinstance(REL.parent_key_columns, tuple) and isinstance(REL.detail_key_columns, tuple)


def test_relationship_columns_are_schema_valid_and_compatible() -> None:
    parent, detail = REL.parent_definition, REL.detail_definition
    assert REL.parent_key_columns and len(REL.parent_key_columns) == len(REL.detail_key_columns)
    assert set(REL.parent_key_columns) <= set(parent.columns)
    assert set(REL.detail_key_columns) <= set(detail.columns)
    assert REL.expected_detail_count_column in parent.columns
    assert REL.expected_detail_count_column not in parent.identifier_columns
    assert REL.parent_key_columns == parent.unique_key_columns
    for p, d in zip(REL.parent_key_columns, REL.detail_key_columns, strict=True):
        assert parent.identifier_dtypes[p] == detail.identifier_dtypes[d]


@pytest.mark.parametrize(
    "change",
    [
        {"parent_key_columns": ()},
        {"detail_key_columns": ()},
        {"detail_key_columns": ("synth_fa",)},                     # unequal length
        {"parent_key_columns": ("synth_b", "synth_a")},           # not the parent unique key
        {"detail_key_columns": ("synth_fa", "synth_note")},       # not an identifier
        {"detail_key_columns": ("synth_fa", "synth_missing")},    # unknown column
        {"detail_key_columns": ("synth_fa", "synth_fa")},         # duplicate
        {"expected_detail_count_column": "synth_unknown"},
        {"expected_detail_count_column": "synth_a"},              # a key, not a measure
        {"detail": JOBS},                                         # same dataset
    ],
)
def test_invalid_relationships_raise_configuration_error(change: dict) -> None:
    with pytest.raises(RelationshipConfigurationError):
        dataclasses.replace(COMPOSITE, **change)
    assert issubclass(RelationshipConfigurationError, ValueError)


# ------------------------------------------------------------------- exact match


def test_one_job_exact_match_passes() -> None:
    report = _assess(_jobs([J1], [2]), _cars([J1, J1]))
    assert report.is_reconciled and report.matched_job_count == 1


def test_multiple_jobs_exact_and_zero_expected_pass() -> None:
    report = _assess(_jobs([J1, J2, J3], [1, 3, 0]), _cars([J2, J1, J2, J2]))
    assert report.is_reconciled
    assert (report.matched_job_count, report.jobs_without_linked_details_count) == (3, 1)
    assert report.valid_expected_detail_total == report.detail_row_count == 4
    assert (report.absolute_discrepancy_total, report.net_discrepancy) == (0, 0)


def test_one_job_one_detail() -> None:
    assert validate_job_detail_reconciliation(_jobs([J1], [1]), _cars([J1])).is_reconciled


# --------------------------------------------------------- under and over counts


def test_under_counts() -> None:
    report = _assess(_jobs([J1, J2], [3, 2]), _cars([J1]))
    assert (report.under_counted_job_count, report.matched_job_count) == (2, 0)
    assert report.jobs_without_linked_details_count == 1          # J2: positive expected, no details
    assert (report.net_discrepancy, report.absolute_discrepancy_total) == (-4, 4)
    assert "under_count" in report.violations and not report.is_reconciled


def test_over_counts() -> None:
    report = _assess(_jobs([J1, J2], [1, 0]), _cars([J1, J1, J1, J2]))
    assert report.over_counted_job_count == 2                      # J2: expected zero, one detail
    assert (report.net_discrepancy, report.absolute_discrepancy_total) == (3, 3)
    assert "over_count" in report.violations


def test_offsetting_errors_do_not_pass() -> None:
    report = _assess(_jobs([J1, J2], [2, 2]), _cars([J1, J1, J1, J2]))
    assert report.valid_expected_detail_total == report.detail_row_count == 4   # totals equal
    assert report.net_discrepancy == 0 and report.absolute_discrepancy_total == 2
    assert (report.under_counted_job_count, report.over_counted_job_count) == (1, 1)
    assert not report.is_reconciled
    with pytest.raises(JobDetailReconciliationError):
        validate_job_detail_reconciliation(_jobs([J1, J2], [2, 2]), _cars([J1, J1, J1, J2]))


# ---------------------------------------------------------- expected-count policy


@pytest.mark.parametrize(
    ("value", "field"),
    [
        (MISSING, "missing_expected_count_job_count"),
        (np.nan, "missing_expected_count_job_count"),
        (pd.NA, "missing_expected_count_job_count"),
        ("", "missing_expected_count_job_count"),
        ("   ", "missing_expected_count_job_count"),
        ("abc", "non_numeric_expected_count_job_count"),
        ("nan", "non_numeric_expected_count_job_count"),
        (True, "non_numeric_expected_count_job_count"),
        (False, "non_numeric_expected_count_job_count"),
        (float("inf"), "non_finite_expected_count_job_count"),
        (float("-inf"), "non_finite_expected_count_job_count"),
        ("inf", "non_finite_expected_count_job_count"),
        (1.5, "fractional_expected_count_job_count"),
        ("2.5", "fractional_expected_count_job_count"),
        (-1, "negative_expected_count_job_count"),
        ("-2", "negative_expected_count_job_count"),
    ],
)
def test_invalid_expected_counts_are_categorised_not_repaired(value: object, field: str) -> None:
    jobs = _jobs([J1, J2], [1, value], count_dtype=object)
    snapshot = jobs.copy(deep=True)
    report = _assess(jobs, _cars([J1]))
    assert getattr(report, field) == 1
    assert report.valid_expected_count_job_count == 1
    assert report.matched_job_count == 1 and report.under_counted_job_count == 0  # not treated as 0
    assert report.jobs_without_linked_details_count == 1
    assert not report.is_reconciled and not report.all_expected_counts_valid
    pd.testing.assert_frame_equal(jobs, snapshot)


@pytest.mark.parametrize("value", ["2", "2.0", 2.0, np.int64(2), np.float64(2.0)])
def test_whole_numeric_values_and_strings_are_accepted(value: object) -> None:
    report = _assess(_jobs([J1], [value], count_dtype=object), _cars([J1, J1]))
    assert report.is_reconciled and report.valid_expected_detail_total == 2


@pytest.mark.parametrize("dtype", ["float64", "Int64", "Float64", "string"])
def test_typed_count_columns(dtype: str) -> None:
    values = ["1", None] if dtype == "string" else [1, None]
    report = _assess(_jobs([J1, J2], values, count_dtype=dtype), _cars([J1]))
    assert report.missing_expected_count_job_count == 1 and report.matched_job_count == 1


def test_boolean_count_column_is_invalid() -> None:
    report = _assess(_jobs([J1], [True], count_dtype="bool"), _cars([J1]))
    assert report.non_numeric_expected_count_job_count == 1 and not report.is_reconciled


# ------------------------------------------------------------ detail categories


def test_missing_link_and_orphans_are_distinct() -> None:
    report = _assess(_jobs([J1], [1]), _cars([J1, MISSING, MISSING, "SYNTH-JOB-404", "SYNTH-JOB-404", "SYNTH-JOB-405"]))
    assert report.linked_detail_row_count == 1
    assert report.missing_link_detail_row_count == 2
    assert (report.orphan_detail_row_count, report.distinct_orphan_key_count) == (3, 2)
    assert report.matched_job_count == 1  # job matched, but details are not all linked
    assert set(report.violations) == {"missing_link", "orphan_detail"} and not report.is_reconciled


@pytest.mark.parametrize(("jobs_key", "cars_key"), [("000001", "1"), ("1", "000001"), ("000001", "01")])
def test_leading_zero_identifiers_stay_distinct(jobs_key: str, cars_key: str) -> None:
    report = _assess(_jobs([jobs_key], [1]), _cars([cars_key]))
    assert report.orphan_detail_row_count == 1 and report.under_counted_job_count == 1


def test_long_and_alphanumeric_identifiers_reconcile_exactly() -> None:
    long_a = "123456789012345678901234567890"
    long_b = "123456789012345678901234567891"  # same float value, different identifier
    report = _assess(_jobs([long_a, long_b, "A-001-B"], [2, 1, 1]), _cars([long_a, long_b, long_a, "A-001-B"]))
    assert report.is_reconciled


def test_identifiers_are_compared_verbatim() -> None:
    report = _assess(_jobs([J1], [1]), _cars([" " + J1]))
    assert report.orphan_detail_row_count == 1


# ----------------------------------------------------------------- composite keys


def test_composite_keys_reconcile_by_all_components() -> None:
    jobs = _jobs([("SYNTH-JOB-001", "000001"), ("SYNTH-JOB-001", "000002")], [2, 1], COMPOSITE)
    cars = _cars([("SYNTH-JOB-001", "000001"), ("SYNTH-JOB-001", "000002"), ("SYNTH-JOB-001", "000001")], COMPOSITE)
    assert _assess(jobs, cars, COMPOSITE).is_reconciled


def test_matching_one_component_is_not_a_link() -> None:
    jobs = _jobs([("SYNTH-JOB-001", "000001")], [1], COMPOSITE)
    cars = _cars([("SYNTH-JOB-001", "000002"), ("SYNTH-JOB-002", "000001"), ("SYNTH-JOB-001", MISSING)], COMPOSITE)
    report = _assess(jobs, cars, COMPOSITE)
    assert report.linked_detail_row_count == 0
    assert (report.orphan_detail_row_count, report.missing_link_detail_row_count) == (2, 1)


@pytest.mark.parametrize("pair", [(("A|B", "C"), ("A", "B|C")), (("A", "BC"), ("AB", "C")), (("A,B", ""), ("A", ",B"))])
def test_delimiter_like_values_never_collide(pair) -> None:  # type: ignore[no-untyped-def]
    parent, other = pair
    report = _assess(_jobs([parent], [1], COMPOSITE), _cars([other], COMPOSITE), COMPOSITE)
    assert report.orphan_detail_row_count == 1 and report.linked_detail_row_count == 0


def test_component_order_is_deterministic() -> None:
    assert COMPOSITE.parent_key_columns == _SYNTH_PARENT.unique_key_columns
    swapped = _cars([("000001", "SYNTH-JOB-001")], COMPOSITE)  # components in the wrong order
    report = _assess(_jobs([("SYNTH-JOB-001", "000001")], [1], COMPOSITE), swapped, COMPOSITE)
    assert report.orphan_detail_row_count == 1


# --------------------------------------------------------------- preconditions


def test_duplicate_parent_keys_raise_precondition_error() -> None:
    with pytest.raises(ReconciliationPreconditionError) as info:
        assess_job_detail_reconciliation(_jobs([J1, J1], [1, 1]), _cars([J1, J1]))
    assert info.value.reason == "parent_key_duplicate" and info.value.role == JOBS
    assert info.value.unique_key_report.duplicate_key_row_count == 2
    assert J1 not in str(info.value)


def test_missing_parent_key_raises_precondition_error() -> None:
    with pytest.raises(ReconciliationPreconditionError) as info:
        assess_job_detail_reconciliation(_jobs([J1, MISSING], [1, 1]), _cars([J1]))
    assert info.value.reason == "parent_key_missing"


def test_non_identifier_key_dtype_raises_precondition_error() -> None:
    cars = _cars([J1]).astype({REL.detail_key_columns[0]: object})
    with pytest.raises(ReconciliationPreconditionError) as info:
        assess_job_detail_reconciliation(_jobs([J1], [1]), cars)
    assert (info.value.reason, info.value.role) == ("identifier_dtype", CARS)


def test_uncleaned_blank_rows_raise_precondition_error() -> None:
    cars = _cars([J1, J1])
    cars.iloc[1] = None  # a completely blank row
    with pytest.raises(ReconciliationPreconditionError) as info:
        assess_job_detail_reconciliation(_jobs([J1], [1]), cars)
    assert info.value.reason == "blank_rows_present"


def test_absent_columns_raise_configuration_error() -> None:
    with pytest.raises(RelationshipConfigurationError) as info:
        assess_job_detail_reconciliation(_jobs([J1], [1]).drop(columns=[REL.expected_detail_count_column]), _cars([J1]))
    assert info.value.columns == (REL.expected_detail_count_column,)
    with pytest.raises(RelationshipConfigurationError):
        assess_job_detail_reconciliation(_jobs([J1], [1]), _cars([J1]).drop(columns=list(REL.detail_key_columns)))


def test_duplicate_detail_rows_are_counted_not_removed() -> None:
    cars = _cars([J1, J1])
    cars.iloc[1] = cars.iloc[0]  # exact duplicate detail rows
    assert cars.duplicated(keep=False).all()
    report = _assess(_jobs([J1], [1]), cars)
    assert report.over_counted_job_count == 1 and report.linked_detail_row_count == 2
    assert len(cars) == 2


@pytest.mark.parametrize("bad", [None, [], "synthetic"])
def test_invalid_argument_types(bad: object) -> None:
    with pytest.raises(TypeError):
        assess_job_detail_reconciliation(bad, _cars([J1]))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess_job_detail_reconciliation(_jobs([J1], [1]), _cars([J1]), bad)  # type: ignore[arg-type]


# -------------------------------------------------------------- report contents


def test_report_holds_plain_aggregate_scalars_only() -> None:
    report = _assess(_jobs([J1, J2], [1, 2]), _cars([J1, "SYNTH-JOB-404"]))
    for f in dataclasses.fields(report):
        value = getattr(report, f.name)
        if f.name == "count_fields":                    # per-declaration aggregates (contract names, ints)
            assert all(all(type(getattr(x, g.name)) is (str if g.name == "column" else int)
                           for g in dataclasses.fields(x)) for x in value)
            continue
        assert type(value) in (int, bool), f.name
    assert J1 not in repr(report) and "SYNTH-JOB-404" not in repr(report)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.job_count = 0  # type: ignore[misc]


def test_inconsistent_report_is_a_programmer_error() -> None:
    good = _assess(_jobs([J1], [1]), _cars([J1]))
    with pytest.raises(AssertionError):
        dataclasses.replace(good, linked_detail_row_count=5)


# ------------------------------------------------------------------ non-mutation


def test_sources_are_unchanged_and_assessment_is_idempotent() -> None:
    jobs = _jobs([J2, J1, J3], [1, "2", 0], count_dtype=object)
    jobs.index = pd.Index([30, 10, 20], name="synthetic_label")
    cars = _cars([J1, "SYNTH-JOB-404", J2, MISSING, J1])
    cars.index = pd.Index([5, 4, 3, 2, 1])
    snapshots = (jobs.copy(deep=True), cars.copy(deep=True))
    first = assess_job_detail_reconciliation(jobs, cars)
    second = assess_job_detail_reconciliation(jobs, cars)
    assert first == second
    for frame, snapshot in zip((jobs, cars), snapshots, strict=True):
        pd.testing.assert_frame_equal(frame, snapshot)
        assert frame.index.equals(snapshot.index) and list(frame.columns) == list(snapshot.columns)
        assert frame.dtypes.equals(snapshot.dtypes)


# --------------------------------------------------------------------- edge cases


def test_empty_jobs_and_details_are_vacuously_reconciled() -> None:
    report = _assess(_jobs([], []), _cars([]))
    assert report.is_reconciled and report.job_count == report.detail_row_count == 0


def test_empty_jobs_with_details_fail_as_orphans() -> None:
    report = _assess(_jobs([], []), _cars([J1, MISSING]))
    assert (report.orphan_detail_row_count, report.missing_link_detail_row_count) == (1, 1)
    assert not report.is_reconciled


def test_jobs_without_details_pass_only_when_all_expect_zero() -> None:
    assert _assess(_jobs([J1, J2], [0, 0]), _cars([])).is_reconciled
    assert not _assess(_jobs([J1, J2], [0, 1]), _cars([])).is_reconciled
    assert not _assess(_jobs([J1, J2], [0, MISSING], count_dtype=object), _cars([])).is_reconciled


# ------------------------------------------------------------- strict validation


def test_strict_validation_passes_and_returns_report() -> None:
    report = validate_job_detail_reconciliation(_jobs([J1], [2]), _cars([J1, J1]))
    assert isinstance(report, JobDetailReconciliationReport) and report.is_reconciled


@pytest.mark.parametrize(
    ("jobs_args", "cars_keys", "category"),
    [
        (([J1], [2]), [J1], "under_count"),
        (([J1], [1]), [J1, MISSING], "missing_link"),
        (([J1], [1]), [J1, "SYNTH-JOB-404"], "orphan_detail"),
        (([J1], ["abc"]), [J1], "invalid_expected_count"),
    ],
)
def test_strict_validation_raises_safe_error(jobs_args, cars_keys, category: str) -> None:  # type: ignore[no-untyped-def]
    jobs, cars = _jobs(*jobs_args, count_dtype=object), _cars(cars_keys)
    assessed = assess_job_detail_reconciliation(jobs, cars)  # assessment does not raise
    with pytest.raises(JobDetailReconciliationError) as info:
        validate_job_detail_reconciliation(jobs, cars)
    message = str(info.value)
    assert category in message and info.value.report == assessed
    assert "SYNTH" not in message and "abc" not in message
    assert not any(ch.isdigit() for ch in message)
    assert not isinstance(info.value, (ReconciliationPreconditionError, RelationshipConfigurationError))


# ----------------------------------------------------------- pipeline interaction


def _csv(directory: Path, key: DatasetKey, lines: list[str]) -> None:
    text = ",".join(contract_columns(key)) + "\n" + "".join(line + "\n" for line in lines)
    (directory / f"synthetic_{key}.csv").write_bytes(text.encode("utf-8"))


def _row(key: DatasetKey, values: dict[str, str]) -> str:
    return ",".join(values.get(c, f"synthetic_{i}") for i, c in enumerate(contract_columns(key)))


def test_pipeline_blank_rows_removed_before_reconciliation(tmp_path: Path) -> None:
    pk, dk, count = REL.parent_key_columns[0], REL.detail_key_columns[0], REL.expected_detail_count_column
    cars_ordinal = REL.detail_definition.non_identifier_key_columns[0]
    _csv(tmp_path, JOBS, [_row(JOBS, {pk: "000001", count: "2"}), _row(JOBS, {pk: "000002", count: "0"}), ""])
    n_cars = len(contract_columns(CARS))
    _csv(tmp_path, CARS, [
        _row(CARS, {dk: "000001", cars_ordinal: "1"}),
        "",                                   # completely blank: must not count
        "," * (n_cars - 1),                   # completely blank: must not count
        _row(CARS, {dk: "000001", cars_ordinal: "2"}),
        _row(CARS, {dk: "", cars_ordinal: "3"}),   # partially populated, missing link
    ])
    blank = remove_blank_rows_from_raw_datasets(load_raw_datasets(tmp_path))
    cleaned = blank.cleaned
    validate_raw_dataset_identifier_dtypes(cleaned)
    key_reports = assess_raw_dataset_unique_keys(cleaned)
    report = _assess(cleaned.jobs, cleaned.cars)
    assert blank.cars.removed_blank_row_count == 2
    assert report.detail_row_count == 3 and report.linked_detail_row_count == 2
    assert report.missing_link_detail_row_count == 1
    assert report.matched_job_count == 2  # leading-zero keys preserved end to end
    validate_raw_dataset_identifier_dtypes(cleaned)  # still valid after reconciliation
    assert assess_raw_dataset_unique_keys(cleaned) == key_reports  # independent of reconciliation
    assert is_identifier_dtype(cleaned.cars[dk].dtype)


def test_package_exposes_reconciliation_api() -> None:
    for name in ("assess_job_detail_reconciliation", "validate_job_detail_reconciliation",
                 "JobDetailReconciliationReport", "JobDetailReconciliationError",
                 "ReconciliationPreconditionError", "RelationshipConfigurationError",
                 "JOB_DETAIL_RELATIONSHIP", "JobDetailRelationshipDefinition"):
        assert name in ql2_sixt_canada_analysis.__all__ and hasattr(ql2_sixt_canada_analysis, name)
