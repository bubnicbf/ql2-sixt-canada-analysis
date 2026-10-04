"""Tests for the jobs-to-cars one-to-many relationship control and trusted join.

All identifiers are fabricated (``SYNTH-JOB-001``, ``000001``, ...). Frames
are built from the centralized column, identifier, key and relationship
definitions; CSVs are written to ``tmp_path`` only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest
from conftest import contract_columns
from pandas.errors import MergeError

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis import relationships
from ql2_sixt_canada_analysis.identifiers import is_identifier_dtype, validate_identifier_dtypes
from ql2_sixt_canada_analysis.ingestion import load_raw_datasets
from ql2_sixt_canada_analysis.quality import remove_blank_rows_from_raw_datasets
from ql2_sixt_canada_analysis.reconciliation import (
    ReconciliationPreconditionError,
    assess_job_detail_reconciliation,
)
from ql2_sixt_canada_analysis.relationships import (
    OneToManyJoinReport,
    OneToManyRelationshipError,
    RelationshipPreconditionError,
    ValidatedJoinError,
    ValidatedJoinResult,
    assess_one_to_many_join,
    join_jobs_to_details,
    validate_one_to_many_join,
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
J1, J2, J3 = "SYNTH-JOB-001", "SYNTH-JOB-002", "SYNTH-JOB-003"

# Synthetic composite relationship with differently named keys and a
# colliding non-key column ("synth_note") for collision tests.
_SYNTH_PARENT = DatasetDefinition(
    key=JOBS, filename_tokens=("synthparent",),
    columns=("synth_a", "synth_b", "synth_expected", "synth_note", "synth_parent_only"),
    identifier_columns=("synth_a", "synth_b"), unique_key_columns=("synth_a", "synth_b"),
)
_SYNTH_DETAIL = DatasetDefinition(
    key=CARS, filename_tokens=("synthdetail",),
    columns=("synth_fa", "synth_fb", "synth_note", "synth_detail_only"),
    identifier_columns=("synth_fa", "synth_fb"), unique_key_columns=(),
)
COMPOSITE = JobDetailRelationshipDefinition(
    parent=JOBS, detail=CARS,
    parent_key_columns=("synth_a", "synth_b"), detail_key_columns=("synth_fa", "synth_fb"),
    expected_detail_count_column="synth_expected",
    definitions=MappingProxyType({JOBS: _SYNTH_PARENT, CARS: _SYNTH_DETAIL}),
)


def _frame(definition: DatasetDefinition, key_columns: tuple[str, ...], keys: list[object], tag: str) -> pd.DataFrame:
    rows = [k if isinstance(k, tuple) else (k,) for k in keys]
    data: dict[str, object] = {}
    for i, column in enumerate(definition.columns):
        if column in key_columns:
            pos = key_columns.index(column)
            data[column] = pd.array([None if r[pos] is MISSING else r[pos] for r in rows], dtype=IDENTIFIER_DTYPE)
        elif column in definition.identifier_columns:
            data[column] = pd.array([f"SYNTH-{tag}-ID-{n:03d}" for n in range(len(rows))], dtype=IDENTIFIER_DTYPE)
        else:
            data[column] = pd.Series([f"{tag}-{column}-{n}" for n in range(len(rows))], dtype=object)
    return pd.DataFrame(data, columns=list(definition.columns))


def _jobs(keys: list[object], rel: JobDetailRelationshipDefinition = REL) -> pd.DataFrame:
    return _frame(rel.parent_definition, rel.parent_key_columns, keys, "JOB")


def _cars(keys: list[object], rel: JobDetailRelationshipDefinition = REL) -> pd.DataFrame:
    return _frame(rel.detail_definition, rel.detail_key_columns, keys, "DETAIL")


def _assess(jobs: pd.DataFrame, cars: pd.DataFrame, rel: JobDetailRelationshipDefinition = REL) -> OneToManyJoinReport:
    report = assess_one_to_many_join(jobs, cars, rel)
    assert report.detail_row_count == (report.linked_detail_row_count + report.missing_link_detail_row_count
                                       + report.orphan_detail_row_count)
    assert report.expected_left_join_row_count == report.linked_detail_row_count + report.parents_without_details_count
    assert report.parent_row_count == report.parents_with_details_count + report.parents_without_details_count
    if report.is_valid:
        assert report.actual_left_join_row_count == report.expected_left_join_row_count
        assert report.violations == ()
    return report


# ----------------------------------------------------- relationship configuration


def test_relationship_roles_keys_and_dtypes() -> None:
    assert (REL.parent, REL.detail) == (JOBS, CARS)
    assert REL.parent_key_columns and len(REL.parent_key_columns) == len(REL.detail_key_columns)
    assert set(REL.parent_key_columns) <= set(REL.parent_definition.columns)
    assert set(REL.detail_key_columns) <= set(REL.detail_definition.columns)
    assert REL.parent_key_columns == DATASET_DEFINITIONS[JOBS].unique_key_columns
    for p, d in zip(REL.parent_key_columns, REL.detail_key_columns, strict=True):
        assert REL.parent_definition.identifier_dtypes[p] == REL.detail_definition.identifier_dtypes[d]
    assert (REL.parent_suffix, REL.detail_suffix) == ("_job", "_detail")


def test_relationship_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        REL.detail_key_columns = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        REL.parent_suffix = "_x"  # type: ignore[misc]


@pytest.mark.parametrize("change", [{"parent_suffix": ""}, {"detail_suffix": "_job"}, {"detail_key_columns": ("synth_fa",)}])
def test_invalid_relationship_configuration(change: dict) -> None:
    with pytest.raises(RelationshipConfigurationError):
        dataclasses.replace(COMPOSITE, **change)


# ------------------------------------------------------------- valid relationships


@pytest.mark.parametrize(
    ("jobs_keys", "cars_keys", "with_details", "rows"),
    [
        ([J1], [J1], 1, 1),
        ([J1], [J1, J1, J1], 1, 3),
        ([J1, J2, J3], [J2, J1, J2, J2], 2, 5),       # 3 for J2, 1 for J1, J3 placeholder
        ([J1, J2], [J1], 1, 2),
        ([J1, J2, J3], [], 0, 3),
    ],
    ids=["one-one", "one-many", "varied", "zero-detail-parent", "all-zero"],
)
def test_valid_one_to_many_relationships(jobs_keys, cars_keys, with_details: int, rows: int) -> None:  # type: ignore[no-untyped-def]
    jobs, cars = _jobs(jobs_keys), _cars(cars_keys)
    report = _assess(jobs, cars)
    assert report.is_valid and report.cardinality_validated and report.row_conservation_holds
    assert report.parents_with_details_count == with_details
    assert report.expected_left_join_row_count == report.actual_left_join_row_count == rows
    result = join_jobs_to_details(jobs, cars)
    assert isinstance(result, ValidatedJoinResult) and result.report == report
    assert len(result.joined) == rows


def test_each_linked_detail_and_each_empty_parent_appears_once() -> None:
    jobs, cars = _jobs([J1, J2, J3]), _cars([J3, J1, J3])
    joined = join_jobs_to_details(jobs, cars).joined
    detail_marker = next(  # a detail-only, non-identifier column: unique per detail row here
        c for c in REL.detail_definition.columns
        if c not in REL.parent_definition.columns and c not in REL.detail_definition.identifier_columns)
    linked = joined[detail_marker].dropna()
    assert sorted(linked) == sorted(cars[detail_marker])               # every detail exactly once
    pk = REL.parent_key_columns[0]
    assert joined[pk].tolist() == [J1, J2, J3, J3]                      # parent order, J2 once
    assert joined.loc[joined[pk] == J2, detail_marker].isna().all()
    assert joined.loc[joined[pk] == J3, detail_marker].tolist() == [cars[detail_marker][0], cars[detail_marker][2]]


# ----------------------------------------------------------- invalid parent keys


def test_duplicate_parent_keys_fail_preconditions_without_multiplying_rows() -> None:
    jobs, cars = _jobs([J1, J1]), _cars([J1, J1, J1])
    for call in (assess_one_to_many_join, validate_one_to_many_join, join_jobs_to_details):
        with pytest.raises(RelationshipPreconditionError) as info:
            call(jobs, cars)
        assert info.value.reason == "parent_key_duplicate" and info.value.role == JOBS
        assert J1 not in str(info.value) and not any(ch.isdigit() for ch in str(info.value))


def test_missing_parent_key_fails_preconditions() -> None:
    with pytest.raises(RelationshipPreconditionError) as info:
        join_jobs_to_details(_jobs([J1, MISSING]), _cars([J1]))
    assert info.value.reason == "parent_key_missing"


def test_identifier_dtype_and_blank_row_preconditions() -> None:
    cars = _cars([J1]).astype({REL.detail_key_columns[0]: object})
    with pytest.raises(RelationshipPreconditionError) as info:
        assess_one_to_many_join(_jobs([J1]), cars)
    assert info.value.reason == "identifier_dtype"
    blank = _cars([J1, J1])
    blank.iloc[1] = None
    with pytest.raises(RelationshipPreconditionError) as info:
        assess_one_to_many_join(_jobs([J1]), blank)
    assert info.value.reason == "blank_rows_present"


def test_absent_relationship_column_is_a_configuration_error() -> None:
    with pytest.raises(RelationshipConfigurationError):
        assess_one_to_many_join(_jobs([J1]), _cars([J1]).drop(columns=list(REL.detail_key_columns)))


def test_reconciliation_precondition_error_shares_the_hierarchy() -> None:
    assert issubclass(ReconciliationPreconditionError, RelationshipPreconditionError)
    with pytest.raises(ReconciliationPreconditionError):
        assess_job_detail_reconciliation(_jobs([J1, J1]), _cars([J1]))


# ------------------------------------------------------------ missing and orphan


def test_missing_link_and_orphans_are_classified_separately() -> None:
    jobs = _jobs([J1])
    cars = _cars([J1, MISSING, "SYNTH-JOB-404", "SYNTH-JOB-404", "SYNTH-JOB-405", MISSING])
    report = _assess(jobs, cars)
    assert (report.linked_detail_row_count, report.missing_link_detail_row_count) == (1, 2)
    assert (report.orphan_detail_row_count, report.distinct_orphan_key_count) == (3, 2)
    assert report.cardinality_validated and report.row_conservation_holds  # join itself is sound
    assert not report.is_valid and report.violations == ("missing_link", "orphan_detail")
    assert report.expected_left_join_row_count == report.actual_left_join_row_count == 1


@pytest.mark.parametrize("bad_key", [MISSING, "SYNTH-JOB-404"], ids=["missing-link", "orphan"])
def test_strict_validation_and_join_refuse_unlinked_details(bad_key: object) -> None:
    jobs, cars = _jobs([J1]), _cars([J1, bad_key])
    snapshot = cars.copy(deep=True)
    assessed = assess_one_to_many_join(jobs, cars)
    for call in (validate_one_to_many_join, join_jobs_to_details):
        with pytest.raises(OneToManyRelationshipError) as info:
            call(jobs, cars)
        assert info.value.report == assessed
        message = str(info.value)
        assert "SYNTH" not in message and not any(ch.isdigit() for ch in message)
    pd.testing.assert_frame_equal(cars, snapshot)  # nothing dropped from the source


def test_leading_zero_and_long_identifiers() -> None:
    long_a, long_b = "123456789012345678901234567890", "123456789012345678901234567891"
    report = _assess(_jobs(["000001", long_a]), _cars(["1", "000001", long_b]))
    assert report.linked_detail_row_count == 1 and report.orphan_detail_row_count == 2


# ----------------------------------------------------------------- composite keys


def test_composite_keys_require_all_components() -> None:
    jobs = _jobs([("SYNTH-JOB-001", "000001"), ("SYNTH-JOB-001", "000002")], COMPOSITE)
    cars = _cars([("SYNTH-JOB-001", "000002"), ("SYNTH-JOB-001", "000001"), ("SYNTH-JOB-001", "000002")], COMPOSITE)
    report = _assess(jobs, cars, COMPOSITE)
    assert report.is_valid and report.actual_left_join_row_count == 3
    bad = _cars([("SYNTH-JOB-001", "000003"), ("SYNTH-JOB-002", "000001")], COMPOSITE)
    assert _assess(jobs, bad, COMPOSITE).orphan_detail_row_count == 2


@pytest.mark.parametrize("pair", [(("A|B", "C"), ("A", "B|C")), (("A", "BC"), ("AB", "C")), (("A-B", ""), ("A", "-B"))])
def test_delimiter_like_values_never_match(pair) -> None:  # type: ignore[no-untyped-def]
    parent, other = pair
    report = _assess(_jobs([parent], COMPOSITE), _cars([other], COMPOSITE), COMPOSITE)
    assert report.linked_detail_row_count == 0 and report.orphan_detail_row_count == 1


def test_component_order_is_deterministic() -> None:
    report = _assess(_jobs([("X1", "Y1")], COMPOSITE), _cars([("Y1", "X1")], COMPOSITE), COMPOSITE)
    assert report.orphan_detail_row_count == 1


# --------------------------------------------------------------- column collisions


def test_collisions_get_stable_suffixes_and_keys_are_explicit() -> None:
    jobs = _jobs([("SYNTH-JOB-001", "000001"), ("SYNTH-JOB-002", "000001")], COMPOSITE)
    cars = _cars([("SYNTH-JOB-001", "000001")], COMPOSITE)
    jobs_columns, cars_columns = list(jobs.columns), list(cars.columns)
    joined = join_jobs_to_details(jobs, cars, COMPOSITE).joined
    assert "synth_note_job" in joined and "synth_note_detail" in joined and "synth_note" not in joined
    assert joined["synth_note_job"].tolist() == list(jobs["synth_note"])          # neither overwritten
    assert joined["synth_note_detail"].iloc[0] == cars["synth_note"].iloc[0]
    assert pd.isna(joined["synth_note_detail"].iloc[1])
    assert {"synth_parent_only", "synth_detail_only", "synth_expected"} <= set(joined.columns)  # unsuffixed
    # Differently named keys are both kept; detail keys are missing for detail-less parents.
    assert {"synth_a", "synth_b", "synth_fa", "synth_fb"} <= set(joined.columns)
    assert pd.isna(joined["synth_fa"].iloc[1]) and joined["synth_a"].iloc[1] == "SYNTH-JOB-002"
    assert joined.columns.is_unique
    assert list(jobs.columns) == jobs_columns and list(cars.columns) == cars_columns


def test_same_named_key_appears_once_and_keeps_identifier_dtype() -> None:
    joined = join_jobs_to_details(_jobs([J1, J2]), _cars([J1])).joined
    pk = REL.parent_key_columns[0]
    assert list(joined.columns).count(pk) == 1 and joined.columns.is_unique
    assert is_identifier_dtype(joined[pk].dtype)
    assert f"{pk}_job" not in joined and f"{pk}_detail" not in joined
    assert not any(c.startswith("__ql2") or c == "_merge" for c in joined.columns)


def test_suffixes_that_would_collide_are_rejected() -> None:
    jobs = _jobs([("A", "B")], COMPOSITE).assign(synth_note_job="synthetic")
    with pytest.raises(RelationshipConfigurationError):
        join_jobs_to_details(jobs, _cars([("A", "B")], COMPOSITE), COMPOSITE)


# ------------------------------------------------------------- order, mutation


def test_inputs_unchanged_and_results_repeatable() -> None:
    jobs = _jobs([J3, J1, J2])
    jobs.index = pd.Index([30, 10, 20], name="synthetic_label")
    cars = _cars([J2, J3, J2, J1])
    cars.index = pd.Index([4, 3, 2, 1])
    snapshots = (jobs.copy(deep=True), cars.copy(deep=True))
    first, second = assess_one_to_many_join(jobs, cars), assess_one_to_many_join(jobs, cars)
    assert first == second
    joined_a = join_jobs_to_details(jobs, cars).joined
    joined_b = join_jobs_to_details(jobs, cars).joined
    pd.testing.assert_frame_equal(joined_a, joined_b)
    assert joined_a[REL.parent_key_columns[0]].tolist() == [J3, J1, J2, J2]   # parent order kept
    assert isinstance(joined_a.index, pd.RangeIndex)
    for frame, snapshot in zip((jobs, cars), snapshots, strict=True):
        pd.testing.assert_frame_equal(frame, snapshot)
        assert frame.index.equals(snapshot.index) and frame.dtypes.equals(snapshot.dtypes)


# ------------------------------------------------------------ row conservation


def test_empty_data_policy() -> None:
    both_empty = _assess(_jobs([]), _cars([]))
    assert both_empty.is_valid and both_empty.expected_left_join_row_count == 0
    assert len(join_jobs_to_details(_jobs([]), _cars([])).joined) == 0
    orphans = _assess(_jobs([]), _cars([J1, MISSING]))
    assert not orphans.is_valid and (orphans.orphan_detail_row_count, orphans.missing_link_detail_row_count) == (1, 1)
    no_details = _assess(_jobs([J1, J2]), _cars([]))
    assert no_details.is_valid and no_details.actual_left_join_row_count == 2


def test_equal_global_totals_do_not_pass() -> None:
    # Two jobs, two details - but both details belong to an unknown job.
    report = _assess(_jobs([J1, J2]), _cars(["SYNTH-JOB-404", "SYNTH-JOB-404"]))
    assert report.parent_row_count == report.detail_row_count == 2
    assert not report.is_valid and report.orphan_detail_row_count == 2


def test_report_requires_every_check_for_validity() -> None:
    good = _assess(_jobs([J1]), _cars([J1]))
    assert good.is_valid
    for change in ({"cardinality_validated": False}, {"row_conservation_holds": False},
                   {"parent_key_unique": False}, {"relationship_dtypes_compatible": False}):
        assert not dataclasses.replace(good, **change).is_valid
    with pytest.raises(AssertionError):
        dataclasses.replace(good, actual_left_join_row_count=-1)


def test_report_holds_only_aggregate_scalars() -> None:
    report = _assess(_jobs([J1, J2]), _cars([J1, "SYNTH-JOB-404"]))
    for f in dataclasses.fields(report):
        assert type(getattr(report, f.name)) in (int, bool), f.name
    assert "SYNTH" not in repr(report)


# ------------------------------------------------------ pandas cardinality checks


def test_join_uses_one_to_many_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    real = pd.merge

    def recording(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(relationships.pd, "merge", recording)
    join_jobs_to_details(_jobs([J1]), _cars([J1, J1]))
    assert calls and all(c["validate"] == "one_to_many" and c["how"] == "left" and c["sort"] is False
                         for c in calls)


def test_forced_invalid_merge_becomes_validated_join_error(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs, cars = _jobs([J1, J1]), _cars([J1])  # duplicate parents
    fake = _assess(_jobs([J1]), _cars([J1]))
    fake = dataclasses.replace(fake, parent_row_count=2, parents_without_details_count=1,
                               expected_left_join_row_count=2, actual_left_join_row_count=2)
    monkeypatch.setattr(relationships, "validate_one_to_many_join", lambda *a, **k: fake)  # bypass checks
    with pytest.raises(ValidatedJoinError) as info:
        join_jobs_to_details(jobs, cars)
    assert isinstance(info.value.__cause__, MergeError)
    assert "SYNTH" not in str(info.value) and not any(ch.isdigit() for ch in str(info.value))


def test_probe_reports_cardinality_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_merge(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise MergeError("synthetic failure")

    monkeypatch.setattr(relationships.pd, "merge", failing_merge)
    report = assess_one_to_many_join(_jobs([J1]), _cars([J1]))
    assert not report.cardinality_validated and not report.is_valid
    assert "cardinality" in report.violations


# ----------------------------------------------------- interaction with pipeline


def _csv(directory: Path, key: DatasetKey, lines: list[str]) -> None:
    text = ",".join(contract_columns(key)) + "\n" + "".join(line + "\n" for line in lines)
    (directory / f"synthetic_{key}.csv").write_bytes(text.encode("utf-8"))


def _row(key: DatasetKey, values: dict[str, str]) -> str:
    return ",".join(values.get(c, f"synthetic_{i}") for i, c in enumerate(contract_columns(key)))


def test_pipeline_interaction(tmp_path: Path) -> None:
    pk, dk, count = REL.parent_key_columns[0], REL.detail_key_columns[0], REL.expected_detail_count_column
    ordinal = REL.detail_definition.non_identifier_key_columns[0]
    _csv(tmp_path, JOBS, [_row(JOBS, {pk: "000001", count: "3"}), _row(JOBS, {pk: "000002", count: "0"})])
    n = len(contract_columns(CARS))
    _csv(tmp_path, CARS, [
        _row(CARS, {dk: "000001", ordinal: "1"}),
        "",                                          # blank line -> removed, not unlinked
        "," * (n - 1),
        _row(CARS, {dk: "000001", ordinal: "1"}),    # duplicate detail key: kept
        _row(CARS, {dk: "", ordinal: "2"}),          # partially populated, missing link
    ])
    cleaned = remove_blank_rows_from_raw_datasets(load_raw_datasets(tmp_path)).cleaned
    keys = assess_raw_dataset_unique_keys(cleaned)
    reconciliation = assess_job_detail_reconciliation(cleaned.jobs, cleaned.cars)
    report = _assess(cleaned.jobs, cleaned.cars)
    assert report.detail_row_count == 3 and report.missing_link_detail_row_count == 1
    assert report.linked_detail_row_count == 2                         # duplicate detail rows both counted
    assert not keys.cars.is_valid                                      # unique-key control: distinct signal
    assert reconciliation.under_counted_job_count == 1                 # count control: distinct signal
    assert not report.is_valid and report.violations == ("missing_link",)
    with pytest.raises(OneToManyRelationshipError):
        join_jobs_to_details(cleaned.jobs, cleaned.cars)
    # Without the unlinked row the join is trusted; identifiers keep their dtype.
    linked_only = cleaned.cars[cleaned.cars[dk].notna()]
    joined = join_jobs_to_details(cleaned.jobs, linked_only).joined
    assert len(joined) == 3 and is_identifier_dtype(joined[pk].dtype)
    validate_identifier_dtypes(cleaned.jobs, DATASET_DEFINITIONS[JOBS])
    assert len(cleaned.cars) == 3                                      # nothing removed by the controls


def test_package_exposes_relationship_api() -> None:
    for name in ("assess_one_to_many_join", "validate_one_to_many_join", "join_jobs_to_details",
                 "OneToManyJoinReport", "ValidatedJoinResult", "OneToManyRelationshipError",
                 "ValidatedJoinError", "RelationshipPreconditionError"):
        assert name in ql2_sixt_canada_analysis.__all__ and hasattr(ql2_sixt_canada_analysis, name)
