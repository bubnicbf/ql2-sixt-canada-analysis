"""Tests for the independent expected-location coverage contract and control.

Every location value is fabricated (``SYNTH-CITY-A``, ``SYNTH-AIRPORT``, ...).
The real contract is used only for its column definition and its
fail-closed (unconfigured) state; expectations in tests are synthetic.
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
from ql2_sixt_canada_analysis.coverage import (
    LocationCoverageError,
    assess_dataset_location_coverage,
    LocationCoverageReport,
    assess_expected_location_coverage,
    validate_expected_location_coverage,
)
from ql2_sixt_canada_analysis.ingestion import load_raw_datasets
from ql2_sixt_canada_analysis.quality import remove_blank_rows_from_raw_datasets
from ql2_sixt_canada_analysis.reconciliation import assess_job_detail_reconciliation
from ql2_sixt_canada_analysis.relationships import assess_one_to_many_join
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    EXPECTED_LOCATION_COVERAGE,
    INVESTIGATED_LOCATION_STREAM,
    JOB_DETAIL_RELATIONSHIP,
    DatasetDefinition,
    DatasetKey,
    LocationCoverageConfigurationError,
    LocationCoverageDefinition,
    LocationCoverageMode,
)

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
EXH, MIN = LocationCoverageMode.EXHAUSTIVE, LocationCoverageMode.MINIMUM_REQUIRED
A, B, C, X = "SYNTH-CITY-A", "SYNTH-CITY-B", "SYNTH-CITY-C", "SYNTH-CITY-X"
LOCATION_COLUMN = EXPECTED_LOCATION_COVERAGE.location_columns[0]


def _single(expected: list[str], mode: LocationCoverageMode = EXH) -> LocationCoverageDefinition:
    """The real jobs location column with a synthetic expected set."""
    return dataclasses.replace(EXPECTED_LOCATION_COVERAGE, expected_locations=tuple((e,) for e in expected), mode=mode)


# Synthetic composite contract (city + site) over a synthetic jobs schema.
_SYNTH_JOBS = DatasetDefinition(
    key=JOBS, filename_tokens=("synthjobs",), columns=("synth_job", "synth_city", "synth_site", "synth_note"),
    identifier_columns=("synth_job",), unique_key_columns=("synth_job",),
)
_REGISTRY = MappingProxyType({JOBS: _SYNTH_JOBS, CARS: DATASET_DEFINITIONS[CARS]})


def _composite(expected: list[tuple[str, str]], mode: LocationCoverageMode = EXH) -> LocationCoverageDefinition:
    return LocationCoverageDefinition(dataset=JOBS, location_columns=("synth_city", "synth_site"),
                                      expected_locations=tuple(expected), mode=mode, definitions=_REGISTRY)


def _jobs(locations: list[object], coverage: LocationCoverageDefinition | None = None, dtype: object = "string") -> pd.DataFrame:
    """Jobs frame with the coverage columns set; neutral placeholders elsewhere."""
    coverage = coverage or EXPECTED_LOCATION_COVERAGE
    rows = [loc if isinstance(loc, tuple) else (loc,) for loc in locations]
    data: dict[str, object] = {}
    for i, column in enumerate(coverage.source_definition.columns):
        if column in coverage.location_columns:
            pos = coverage.location_columns.index(column)
            data[column] = pd.Series([r[pos] for r in rows], dtype=dtype)
        elif column in coverage.source_definition.identifier_columns:
            data[column] = pd.array([f"SYNTH-JOB-{n:03d}" for n in range(len(rows))], dtype="string")
        else:
            data[column] = pd.Series([f"placeholder_{i}"] * len(rows), dtype=object)
    return pd.DataFrame(data, columns=list(coverage.source_definition.columns))


def _assess(jobs: pd.DataFrame, coverage: LocationCoverageDefinition) -> LocationCoverageReport:
    r = assess_expected_location_coverage(jobs, coverage)
    assert r.covered_expected_location_count + r.missing_expected_location_count == r.expected_location_count
    assert 0.0 <= r.coverage_ratio <= 1.0
    assert r.coverage_ratio == r.covered_expected_location_count / r.expected_location_count
    assert (r.missing_expected_location_count == 0) == r.all_expected_covered
    assert r.observed_location_count == r.covered_expected_location_count + r.unexpected_location_count
    expected_valid = r.all_expected_covered and r.all_rows_assigned and (r.mode is MIN or r.unexpected_location_count == 0)
    assert r.is_valid == expected_valid and bool(r.violations) != r.is_valid
    return r


# ------------------------------------------------------------- configuration


def test_real_contract_is_branch_level_minimum_with_the_investigated_stream() -> None:
    cov = EXPECTED_LOCATION_COVERAGE
    # Branch-level locations exist only in the detail dataset.
    assert cov.dataset == CARS and isinstance(cov.location_columns, tuple) and cov.location_columns
    assert set(cov.location_columns) <= set(DATASET_DEFINITIONS[CARS].columns)
    assert not set(cov.location_columns) & set(DATASET_DEFINITIONS[JOBS].columns)
    assert cov.is_configured and cov.mode is MIN          # authority covers a minimum, not a universe
    assert INVESTIGATED_LOCATION_STREAM in cov.expected_locations
    assert len(INVESTIGATED_LOCATION_STREAM) == len(cov.location_columns)
    assert dict(cov.aliases) == {}                        # no alias is authoritatively confirmed
    assert set(cov.stream_scope_columns) <= set(DATASET_DEFINITIONS[CARS].columns)


def test_unconfigured_contract_fails_closed() -> None:
    unconfigured = dataclasses.replace(EXPECTED_LOCATION_COVERAGE, expected_locations=None, mode=None)
    for call in (assess_expected_location_coverage, validate_expected_location_coverage):
        with pytest.raises(LocationCoverageConfigurationError) as info:
            call(_jobs([A]), unconfigured)
        assert A not in str(info.value)
    assert issubclass(LocationCoverageConfigurationError, ValueError)


def test_definition_is_immutable_and_well_formed() -> None:
    cov = _composite([("SYNTH-CITY-A", "SYNTH-AIRPORT"), ("SYNTH-CITY-A", "SYNTH-DOWNTOWN")])
    with pytest.raises(dataclasses.FrozenInstanceError):
        cov.expected_locations = ()  # type: ignore[misc]
    assert isinstance(cov.expected_locations, tuple) and all(isinstance(k, tuple) for k in cov.expected_locations)
    assert all(len(k) == len(cov.location_columns) for k in cov.expected_locations)
    assert len(set(cov.expected_locations)) == len(cov.expected_locations)
    assert all(v and v.strip() for k in cov.expected_locations for v in k)
    assert cov.mode in set(LocationCoverageMode) and cov.is_configured


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_locations": (), "mode": EXH},                               # empty set
        {"expected_locations": (("A",), ("A",)), "mode": EXH},                 # duplicate
        {"expected_locations": (("A", "B"),), "mode": EXH},                    # wrong arity
        {"expected_locations": ("A",), "mode": EXH},                           # not a tuple key
        {"expected_locations": ((None,),), "mode": EXH},                       # missing component
        {"expected_locations": (("  ",),), "mode": EXH},                       # blank component
        {"expected_locations": (("A",),), "mode": None},                       # mode missing
        {"expected_locations": None, "mode": EXH},                             # mode without set
        {"expected_locations": (("A",),), "mode": "exhaustive"},               # not the enum
        {"location_columns": ()},
        {"location_columns": ("synthetic_unknown_column",)},
        {"location_columns": (LOCATION_COLUMN, LOCATION_COLUMN)},
    ],
)
def test_invalid_contracts_raise_configuration_error(kwargs: dict) -> None:
    with pytest.raises(LocationCoverageConfigurationError):
        dataclasses.replace(EXPECTED_LOCATION_COVERAGE, **kwargs)


def test_malformed_composite_key_is_rejected() -> None:
    with pytest.raises(LocationCoverageConfigurationError):
        _composite([("SYNTH-CITY-A",)])  # type: ignore[list-item]


def test_absent_location_column_is_a_configuration_error() -> None:
    with pytest.raises(LocationCoverageConfigurationError) as info:
        assess_expected_location_coverage(_jobs([A]).drop(columns=[LOCATION_COLUMN]), _single([A]))
    assert info.value.columns == (LOCATION_COLUMN,)


# ------------------------------------------------- expectations are independent


def test_expectations_are_independent_of_observed_rows() -> None:
    cov = _single([A, B])
    snapshot = copy.deepcopy(cov.expected_locations)
    assert _assess(_jobs([A, B]), cov).is_valid
    removed = _assess(_jobs([A]), cov)                        # drop an expected location
    assert removed.missing_expected_location_count == 1 and not removed.is_valid
    added = _assess(_jobs([A, B, X]), cov)                    # add an unexpected one
    assert added.expected_location_count == 2 and added.unexpected_location_count == 1
    for _ in range(3):
        _assess(_jobs([X, X]), cov)
    assert cov.expected_locations == snapshot                 # never redefined by data
    public = [n for n in dir(ql2_sixt_canada_analysis) if "location" in n.lower()]
    assert not any(n.startswith(("build_", "derive_", "infer_")) for n in public)


# ----------------------------------------------------------------- full coverage


def test_complete_coverage_with_repeats_and_any_order() -> None:
    cov = _single([A, B, C])
    for locations in ([A, B, C], [C, C, B, A, A, A], [B, A, C, A]):
        r = _assess(_jobs(locations), cov)
        assert r.is_valid and r.coverage_ratio == 1.0
        assert r.covered_expected_location_count == r.observed_location_count == 3


def test_one_location_one_job_passes() -> None:
    assert validate_expected_location_coverage(_jobs([A]), _single([A])).is_valid


# ------------------------------------------------------ missing expected locations


def test_missing_expected_locations_are_detected_and_not_compensated() -> None:
    cov = _single([A, B, C])
    one = _assess(_jobs([A] * 50 + [B] * 50), cov)
    assert one.missing_expected_location_count == 1 and one.coverage_ratio == pytest.approx(2 / 3)
    two = _assess(_jobs([A] * 100), cov)
    assert two.missing_expected_location_count == 2 and two.coverage_ratio == pytest.approx(1 / 3)
    assert not one.is_valid and one.violations == ("missing_expected_location",)


def test_strict_validation_raises_safe_error() -> None:
    cov = _single([A, B])
    with pytest.raises(LocationCoverageError) as info:
        validate_expected_location_coverage(_jobs([A, X, None]), cov)
    message = str(info.value)
    assert "missing_expected_location" in message and "missing_location_assignment" in message
    assert message == ("Expected-location coverage contract failed: missing_expected_location, "
                       "unexpected_location, missing_location_assignment.")   # fixed template, no values
    for value in (A, B, X, "SYNTH"):
        assert value not in message
    assert not any(ch.isdigit() for ch in message)
    assert info.value.report == assess_expected_location_coverage(_jobs([A, X, None]), cov)
    assert not isinstance(info.value, LocationCoverageConfigurationError)


# ------------------------------------------------------------ unexpected locations


def test_unexpected_locations_by_mode() -> None:
    jobs = _jobs([A, B, X, X, X, "SYNTH-CITY-Y"])
    exhaustive = _assess(jobs, _single([A, B], EXH))
    minimum = _assess(jobs, _single([A, B], MIN))
    for r in (exhaustive, minimum):
        assert r.unexpected_location_count == 2 and r.covered_expected_location_count == 2
    assert not exhaustive.is_valid and exhaustive.violations == ("unexpected_location",)
    assert minimum.is_valid and minimum.unexpected_locations_acceptable
    with pytest.raises(LocationCoverageError):
        validate_expected_location_coverage(jobs, _single([A, B], EXH))
    validate_expected_location_coverage(jobs, _single([A, B], MIN))


def test_minimum_mode_still_fails_on_missing_expected_and_assignments() -> None:
    assert not _assess(_jobs([A, X]), _single([A, B], MIN)).is_valid
    assert not _assess(_jobs([A, B, None]), _single([A, B], MIN)).is_valid


# ------------------------------------------------------ missing location assignment


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, "", " ", "\t \n"], ids=["None", "nan", "NA", "empty", "space", "mixed-ws"])
@pytest.mark.parametrize("dtype", [object, "string"])
def test_missing_location_components_are_unassigned(missing: object, dtype: object) -> None:
    jobs = _jobs([A, missing, missing], dtype=dtype)
    snapshot = jobs.copy(deep=True)
    r = _assess(jobs, _single([A]))
    assert r.missing_location_row_count == 2 and r.observed_location_count == 1
    assert not r.all_rows_assigned and r.violations == ("missing_location_assignment",)
    with pytest.raises(LocationCoverageError):
        validate_expected_location_coverage(jobs, _single([A]))
    pd.testing.assert_frame_equal(jobs, snapshot)  # whitespace values untouched


def test_only_unassigned_jobs_fail() -> None:
    r = _assess(_jobs([None, " "]), _single([A]))
    assert r.observed_location_count == 0 and r.coverage_ratio == 0.0 and not r.is_valid


# ------------------------------------------------------------- exact comparison


@pytest.mark.parametrize("observed", ["synth-city-a", " SYNTH-CITY-A", "SYNTH-CITY-A ", "SYNTH CITY A",
                                      "SYNTH-CITY-A.", "SYNTH-CTY-A", "SYNTHCITYA"])
def test_near_matches_are_not_matched(observed: str) -> None:
    r = _assess(_jobs([observed]), _single([A]))
    assert r.covered_expected_location_count == 0 and r.unexpected_location_count == 1


def test_string_zero_is_a_valid_location() -> None:
    r = _assess(_jobs(["0", "0"]), _single(["0"]))
    assert r.is_valid and r.missing_location_row_count == 0


def test_non_text_values_never_match_text_expectations() -> None:
    r = _assess(_jobs([0, 1.5], dtype=object), _single(["0"]))
    assert r.covered_expected_location_count == 0 and r.unexpected_location_count == 2


# ------------------------------------------------------------------- composite


def test_composite_requires_all_components() -> None:
    cov = _composite([("SYNTH-CITY-A", "SYNTH-AIRPORT"), ("SYNTH-CITY-B", "SYNTH-AIRPORT")])
    r = _assess(_jobs([("SYNTH-CITY-A", "SYNTH-AIRPORT"), ("SYNTH-CITY-A", "SYNTH-DOWNTOWN"),
                       ("SYNTH-CITY-C", "SYNTH-AIRPORT")], cov), cov)
    assert r.covered_expected_location_count == 1 and r.unexpected_location_count == 2
    # Same site label in different cities is distinct.
    assert r.missing_expected_location_count == 1


def test_composite_missing_component_and_order() -> None:
    cov = _composite([("SYNTH-CITY-A", "SYNTH-AIRPORT")])
    r = _assess(_jobs([("SYNTH-CITY-A", None), ("SYNTH-CITY-A", " "), ("SYNTH-AIRPORT", "SYNTH-CITY-A")], cov), cov)
    assert r.missing_location_row_count == 2 and r.unexpected_location_count == 1
    assert r.covered_expected_location_count == 0


@pytest.mark.parametrize("pair", [(("A|B", "C"), ("A", "B|C")), (("A", "BC"), ("AB", "C")), (("A-B", "C"), ("A", "B-C"))])
def test_composite_delimiter_like_values_never_collide(pair) -> None:  # type: ignore[no-untyped-def]
    expected, observed = pair
    cov = _composite([expected])
    r = _assess(_jobs([observed], cov), cov)
    assert r.covered_expected_location_count == 0 and r.unexpected_location_count == 1


# ----------------------------------------------------------- report and mutation


def test_report_holds_only_aggregates() -> None:
    r = _assess(_jobs([A, X, None]), _single([A, B]))
    for f in dataclasses.fields(r):
        value = getattr(r, f.name)
        assert type(value) is int or isinstance(value, LocationCoverageMode), f.name
    assert isinstance(r.coverage_ratio, float)
    text = repr(r)
    assert A not in text and X not in text and "SYNTH" not in text
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.row_count = 0  # type: ignore[misc]
    with pytest.raises(AssertionError):
        dataclasses.replace(r, covered_expected_location_count=r.expected_location_count + 1)


def test_source_unchanged_and_idempotent() -> None:
    jobs = _jobs([B, A, " ", X, A])
    jobs.index = pd.Index([50, 40, 30, 20, 10], name="synthetic_label")
    snapshot = jobs.copy(deep=True)
    cov = _single([A, B, C])
    first, second = _assess(jobs, cov), _assess(jobs, cov)
    assert first == second
    pd.testing.assert_frame_equal(jobs, snapshot)
    assert jobs.index.equals(snapshot.index) and list(jobs.columns) == list(snapshot.columns)
    assert jobs.dtypes.equals(snapshot.dtypes)


def test_empty_jobs_fail_with_zero_coverage() -> None:
    r = _assess(_jobs([]), _single([A, B]))
    assert r.row_count == 0 and r.coverage_ratio == 0.0 and r.missing_expected_location_count == 2
    assert not r.is_valid


def test_duplicate_jobs_are_not_removed_or_double_counted() -> None:
    jobs = _jobs([A, A])
    jobs.iloc[1] = jobs.iloc[0]   # exact duplicate job rows (unique-key control's concern)
    r = _assess(jobs, _single([A]))
    assert r.covered_expected_location_count == 1 and r.row_count == 2 and len(jobs) == 2


@pytest.mark.parametrize("bad", [None, [], "synthetic"])
def test_invalid_argument_types(bad: object) -> None:
    with pytest.raises(TypeError):
        assess_expected_location_coverage(bad, _single([A]))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess_expected_location_coverage(_jobs([A]), bad)  # type: ignore[arg-type]


# ------------------------------------------------------------ pipeline integration


def test_pipeline_blank_rows_removed_and_coverage_before_join(tmp_path: Path) -> None:
    rel = JOB_DETAIL_RELATIONSHIP
    pk, count, dk = rel.parent_key_columns[0], rel.expected_detail_count_column, rel.detail_key_columns[0]
    cov = _single([A])                      # branch-level contract on the detail dataset
    assert cov.dataset == rel.detail

    def write(key: DatasetKey, lines: list[str]) -> None:
        text = ",".join(contract_columns(key)) + "\n" + "".join(line + "\n" for line in lines)
        (tmp_path / f"synthetic_{key}.csv").write_bytes(text.encode("utf-8"))

    def row(key: DatasetKey, values: dict[str, str]) -> str:
        return ",".join(values.get(c, f"synthetic_{i}") for i, c in enumerate(contract_columns(key)))

    n_cars = len(contract_columns(CARS))
    write(JOBS, [row(JOBS, {pk: "SYNTH-JOB-001", count: "3"}), row(JOBS, {pk: "SYNTH-JOB-002", count: "0"})])
    write(CARS, [
        row(CARS, {dk: "SYNTH-JOB-001", LOCATION_COLUMN: A}),
        "",                                                       # blank -> removed, not unassigned
        "," * (n_cars - 1),
        row(CARS, {dk: "SYNTH-JOB-001", LOCATION_COLUMN: A}),     # repeat: still one location
        row(CARS, {dk: "SYNTH-JOB-001", LOCATION_COLUMN: ""}),    # partial row, unassigned
    ])
    blank = remove_blank_rows_from_raw_datasets(load_raw_datasets(tmp_path))
    cleaned = blank.cleaned
    r = assess_dataset_location_coverage(cleaned, cov)
    assert blank.cars.removed_blank_row_count == 2
    assert r.row_count == 3 and r.missing_location_row_count == 1
    assert r.covered_expected_location_count == r.observed_location_count == 1
    # Coverage is measured on cleaned rows before any join; other controls stay independent.
    relationship = assess_one_to_many_join(cleaned.jobs, cleaned.cars)
    assert relationship.actual_left_join_row_count == 4
    reconciliation = assess_job_detail_reconciliation(cleaned.jobs, cleaned.cars)
    assert reconciliation.matched_job_count == 2
    assert len(cleaned.cars) == 3                              # nothing removed by coverage


def test_alias_counts_only_through_the_central_definition() -> None:
    plain = _single([A], MIN)
    aliased = dataclasses.replace(plain, aliases={(A,): (("SYNTH-CITY-A-ALT",),)})
    jobs = _jobs(["SYNTH-CITY-A-ALT"])
    assert _assess(jobs, plain).covered_expected_location_count == 0            # not applied implicitly
    r = assess_expected_location_coverage(jobs, aliased)
    assert r.covered_expected_location_count == 1 and r.unexpected_location_count == 0
    assert jobs[LOCATION_COLUMN].tolist() == ["SYNTH-CITY-A-ALT"]                # source unchanged
    with pytest.raises(dataclasses.FrozenInstanceError):
        aliased.aliases = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        aliased.aliases[(B,)] = ((X,),)  # type: ignore[index]


@pytest.mark.parametrize(
    "aliases",
    [{("SYNTH-UNKNOWN",): (("X",),)}, {(A,): ((B,),)}, {(A,): (("X", "Y"),)}, {(A,): ((" ",),)}, {(A,): ()}],
    ids=["not-expected", "alias-is-expected", "arity", "blank", "empty"],
)
def test_invalid_aliases_are_rejected(aliases: dict) -> None:
    with pytest.raises(LocationCoverageConfigurationError):
        dataclasses.replace(_single([A, B]), aliases=aliases)


def test_package_exposes_coverage_api() -> None:
    for name in ("assess_expected_location_coverage", "validate_expected_location_coverage",
                 "LocationCoverageReport", "LocationCoverageError", "LocationCoverageConfigurationError",
                 "LocationCoverageDefinition", "LocationCoverageMode", "EXPECTED_LOCATION_COVERAGE"):
        assert name in ql2_sixt_canada_analysis.__all__ and hasattr(ql2_sixt_canada_analysis, name)
