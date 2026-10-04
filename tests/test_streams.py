"""Tests for the generic expected-location stream investigation.

All locations, identifiers and timestamps are fabricated (``SYNTH-DOWNTOWN``,
``SYNTH-JOB-001``, ``2025-01-01T00:00:00Z``, ...). Column positions come from
the central contract and relationship definitions; the real target appears
only through ``INVESTIGATED_LOCATION_STREAM``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
from conftest import contract_columns

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.coverage import assess_expected_location_coverage
from ql2_sixt_canada_analysis.ingestion import RawDatasets, load_raw_datasets
from ql2_sixt_canada_analysis.quality import remove_blank_rows_from_raw_datasets
from ql2_sixt_canada_analysis.schemas import (
    COLLECTION_SCHEDULE,
    EXPECTED_LOCATION_COVERAGE,
    IDENTIFIER_DTYPE,
    INVESTIGATED_LOCATION_STREAM,
    JOB_DETAIL_RELATIONSHIP,
    CollectionScheduleDefinition,
    DatasetKey,
    LocationCoverageConfigurationError,
    LocationCoverageMode,
)
from ql2_sixt_canada_analysis.streams import (
    LocationStreamError,
    LocationStreamInvestigationReport,
    LocationStreamStatus as S,
    PipelineStage as P,
    StreamContinuity,
    TimeCoverageStatus,
    investigate_location_stream,
    resolve_expected_location,
    validate_location_stream,
)

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
REL = JOB_DETAIL_RELATIONSHIP
PK, DK, COUNT = REL.parent_key_columns[0], REL.detail_key_columns[0], REL.expected_detail_count_column
LOC = EXPECTED_LOCATION_COVERAGE.location_columns[0]
SCOPE = EXPECTED_LOCATION_COVERAGE.stream_scope_columns[0]
DOWNTOWN, AIRPORT, CITY = "SYNTH-DOWNTOWN", "SYNTH-AIRPORT", "SYNTH-CITY-A"
J1, J2 = "SYNTH-JOB-001", "SYNTH-JOB-002"
TARGET = (DOWNTOWN,)
COV = dataclasses.replace(EXPECTED_LOCATION_COVERAGE, expected_locations=((DOWNTOWN,), (AIRPORT,)),
                          mode=LocationCoverageMode.MINIMUM_REQUIRED)
# A jobs-level column used as the location for jobs-level synthetic contracts.
JOB_LOCATION = next(c for c in contract_columns(JOBS)
                    if c not in (PK, COUNT) and c not in REL.parent_definition.identifier_columns)
TS = JOB_LOCATION  # a jobs column reused as the synthetic schedule's timestamp column


def _jobs(rows: list[dict]) -> pd.DataFrame:
    data = {c: [r.get(c, f"jobs-{c}") for r in rows] for c in contract_columns(JOBS)}
    frame = pd.DataFrame(data, columns=list(contract_columns(JOBS)))
    frame[COUNT] = pd.Series([r.get(COUNT, 0) for r in rows], dtype="int64")
    return frame.astype(dict(REL.parent_definition.identifier_dtypes))


def _cars(rows: list[tuple[object, str, str]]) -> pd.DataFrame:
    """rows: (job key, location, scope)."""
    data = {c: [f"cars-{c}"] * len(rows) for c in contract_columns(CARS)}
    data[DK] = [r[0] for r in rows]
    data[LOC] = [r[1] for r in rows]
    data[SCOPE] = [r[2] for r in rows]
    frame = pd.DataFrame(data, columns=list(contract_columns(CARS)))
    return frame.astype(dict(REL.detail_definition.identifier_dtypes))


def _healthy_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = _jobs([{PK: J1, COUNT: 2}, {PK: J2, COUNT: 2}])
    cars = _cars([(J1, DOWNTOWN, CITY), (J1, AIRPORT, CITY), (J2, AIRPORT, CITY), (J2, DOWNTOWN, CITY)])
    return jobs, cars


def _investigate(jobs: pd.DataFrame, cars: pd.DataFrame, **kwargs: object) -> LocationStreamInvestigationReport:
    kwargs.setdefault("coverage", COV)
    report = investigate_location_stream(jobs, cars, TARGET, **kwargs)  # type: ignore[arg-type]
    assert report.earliest_failing_stage == (report.failing_stages[0] if report.failing_stages else None)
    assert report.is_healthy == (report.earliest_failing_stage is None)
    return report


# --------------------------------------------------------------- target config


def test_investigated_target_is_a_central_expected_key() -> None:
    target = resolve_expected_location(INVESTIGATED_LOCATION_STREAM)
    assert target is INVESTIGATED_LOCATION_STREAM
    assert target in EXPECTED_LOCATION_COVERAGE.expected_locations
    assert len(target) == len(EXPECTED_LOCATION_COVERAGE.location_columns)


@pytest.mark.parametrize("target", [("SYNTH-UNKNOWN",), (DOWNTOWN, "extra"), DOWNTOWN, (1,), ("synth-downtown",)])
def test_unknown_or_malformed_targets_are_rejected_safely(target: object) -> None:
    with pytest.raises(LocationCoverageConfigurationError) as info:
        resolve_expected_location(target, COV)  # type: ignore[arg-type]
    message = str(info.value)
    assert DOWNTOWN not in message and AIRPORT not in message
    assert not any(part in message for part in INVESTIGATED_LOCATION_STREAM)


def test_resolution_needs_a_configured_contract_and_no_data() -> None:
    unconfigured = dataclasses.replace(EXPECTED_LOCATION_COVERAGE, expected_locations=None, mode=None)
    with pytest.raises(LocationCoverageConfigurationError):
        resolve_expected_location(INVESTIGATED_LOCATION_STREAM, unconfigured)
    jobs, cars = _healthy_inputs()
    report = investigate_location_stream(jobs, cars, TARGET, coverage=unconfigured)
    assert report.status is S.EXPECTATION_NOT_CONFIGURED and report.earliest_failing_stage is P.CONFIGURATION
    assert report.repository_fix_required and not report.target_configured


# ------------------------------------------------------------- classifications


def test_healthy_stream() -> None:
    report = _investigate(*_healthy_inputs())
    assert report.status is S.STREAM_PRESENT_AND_HEALTHY and report.failing_stages == ()
    assert report.stream_continuity is StreamContinuity.COMPLETE
    assert report.reconciliation_passes and report.relationship_passes and report.target_details_all_linked
    assert validate_location_stream(*_healthy_inputs(), TARGET, coverage=COV).is_healthy


def test_absent_stream_is_upstream_and_coverage_still_fails() -> None:
    jobs = _jobs([{PK: J1, COUNT: 1}])
    cars = _cars([(J1, AIRPORT, CITY)])               # airport present, downtown never
    report = _investigate(jobs, cars)
    assert (report.status, report.earliest_failing_stage) == (S.RAW_STREAM_ABSENT, P.RAW_SOURCE)
    assert report.upstream_issue_indicated and not report.repository_fix_required
    assert report.unverified_representation_variant is False   # airport is not a downtown variant
    assert not assess_expected_location_coverage(cars, COV).all_expected_covered
    with pytest.raises(LocationStreamError) as info:
        validate_location_stream(jobs, cars, TARGET, coverage=COV)
    assert DOWNTOWN not in str(info.value) and "raw_stream_absent" in str(info.value)


def test_partial_stream_across_collection_events() -> None:
    jobs = _jobs([{PK: J1, COUNT: 2}, {PK: J2, COUNT: 1}])
    cars = _cars([(J1, DOWNTOWN, CITY), (J1, AIRPORT, CITY), (J2, AIRPORT, CITY)])
    report = _investigate(jobs, cars)
    assert (report.status, report.earliest_failing_stage) == (S.RAW_STREAM_PARTIAL, P.SOURCE_CONTINUITY)
    assert report.stream_continuity is StreamContinuity.PARTIAL
    assert report.reconciliation_passes                  # declared counts match what was returned
    assert report.upstream_issue_indicated


def test_other_scopes_do_not_affect_continuity() -> None:
    jobs = _jobs([{PK: J1, COUNT: 1}, {PK: J2, COUNT: 1}])
    cars = _cars([(J1, DOWNTOWN, CITY), (J2, AIRPORT, "SYNTH-CITY-B")])
    assert _investigate(jobs, cars).stream_continuity is StreamContinuity.COMPLETE


def test_ingestion_and_cleaning_exclusions(tmp_path: Path) -> None:
    jobs, cars = _healthy_inputs()
    without = cars[cars[LOC] != DOWNTOWN]
    raw_csv = tmp_path / "synthetic_raw.csv"
    cars.to_csv(raw_csv, index=False)               # synthetic raw source containing the target
    lost_in_ingestion = _investigate(jobs, without, loaded=RawDatasets(jobs=jobs, cars=without), raw_source=raw_csv)
    assert (lost_in_ingestion.status, lost_in_ingestion.earliest_failing_stage) == (S.INGESTION_EXCLUSION, P.INGESTION)
    lost_in_cleaning = _investigate(jobs, without, loaded=RawDatasets(jobs=jobs, cars=cars), raw_source=raw_csv)
    assert (lost_in_cleaning.status, lost_in_cleaning.earliest_failing_stage) == (S.CLEANING_EXCLUSION, P.CLEANING)
    for report in (lost_in_ingestion, lost_in_cleaning):
        assert report.repository_fix_required and report.present_in_raw_source


def test_parent_key_violation_blocks_reconciliation() -> None:
    jobs = _jobs([{PK: J1, COUNT: 1}, {PK: J1, COUNT: 1}])
    cars = _cars([(J1, DOWNTOWN, CITY)])
    report = _investigate(jobs, cars)
    assert (report.status, report.earliest_failing_stage) == (S.PARENT_KEY_VIOLATION, P.PARENT_KEYS)
    assert report.parent_keys_valid is False and report.reconciliation_passes is None


def test_identifier_type_mismatch() -> None:
    jobs, cars = _healthy_inputs()
    report = _investigate(jobs, cars.astype({DK: object}))
    assert (report.status, report.earliest_failing_stage) == (S.IDENTIFIER_TYPE_MISMATCH, P.IDENTIFIER_TYPES)
    assert report.repository_fix_required


def test_relationship_link_failure_is_distinct_from_absence() -> None:
    jobs = _jobs([{PK: J1, COUNT: 1}])
    cars = _cars([(J1, DOWNTOWN, CITY), ("SYNTH-JOB-404", DOWNTOWN, CITY)])
    report = _investigate(jobs, cars)
    assert (report.status, report.earliest_failing_stage) == (S.RELATIONSHIP_LINK_FAILURE, P.RELATIONSHIP)
    assert report.present_after_cleaning and report.target_details_all_linked is False


def test_count_mismatch() -> None:
    jobs = _jobs([{PK: J1, COUNT: 5}, {PK: J2, COUNT: 2}])
    cars = _cars([(J1, DOWNTOWN, CITY), (J1, AIRPORT, CITY), (J2, AIRPORT, CITY), (J2, DOWNTOWN, CITY)])
    report = _investigate(jobs, cars)
    assert (report.status, report.earliest_failing_stage) == (S.JOB_DETAIL_COUNT_MISMATCH, P.RECONCILIATION)
    assert report.reconciliation_passes is False


def test_jobs_present_details_absent_for_jobs_level_stream() -> None:
    jobs_cov = dataclasses.replace(COV, dataset=JOBS, location_columns=(JOB_LOCATION,), stream_scope_columns=())
    jobs = _jobs([{PK: J1, JOB_LOCATION: DOWNTOWN, COUNT: 2}])
    report = investigate_location_stream(jobs, _cars([]), TARGET, coverage=jobs_cov)
    assert (report.status, report.earliest_failing_stage) == (S.JOBS_PRESENT_DETAILS_ABSENT, P.DETAIL_PRESENCE)
    assert report.target_details_present is False and report.present_after_cleaning
    assert report.stream_continuity is StreamContinuity.NOT_APPLICABLE
    absent = investigate_location_stream(_jobs([{PK: J1, JOB_LOCATION: AIRPORT}]), _cars([]), TARGET, coverage=jobs_cov)
    assert absent.status is S.RAW_STREAM_ABSENT        # jobs absent is not the same as details absent


# ---------------------------------------------------------------- time coverage


def _schedule(periods: tuple[str, ...]) -> CollectionScheduleDefinition:
    return CollectionScheduleDefinition(dataset=JOBS, timestamp_column=TS, expected_periods=periods, period="h")


def test_no_authoritative_schedule_is_explicit() -> None:
    assert COLLECTION_SCHEDULE is None
    report = _investigate(*_healthy_inputs())
    assert not report.schedule_available and report.time_coverage is TimeCoverageStatus.NOT_ASSESSED


def test_partial_time_coverage_under_an_authoritative_schedule() -> None:
    jobs = _jobs([{PK: J1, TS: "2025-01-01T00:00:00Z", COUNT: 2}, {PK: J2, TS: "2025-01-01T01:00:00Z", COUNT: 2}])
    _, cars = _healthy_inputs()
    complete = _investigate(jobs, cars, schedule=_schedule(("2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z")))
    assert complete.time_coverage is TimeCoverageStatus.COMPLETE and complete.is_healthy
    partial = _investigate(jobs, cars, schedule=_schedule(
        ("2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z")))
    assert partial.time_coverage is TimeCoverageStatus.PARTIAL
    assert (partial.status, partial.earliest_failing_stage) == (S.RAW_STREAM_PARTIAL, P.TIME_COVERAGE)


def test_invalid_schedule_is_rejected() -> None:
    with pytest.raises(LocationCoverageConfigurationError):
        _schedule(())
    with pytest.raises(LocationCoverageConfigurationError):
        _schedule(("not-a-timestamp",))


# --------------------------------------------------------- representation issues


@pytest.mark.parametrize("variant", ["synth-downtown", " SYNTH-DOWNTOWN", "SYNTH-DOWNTOWN ", "SYNTH DOWNTOWN",
                                     "SYNTH.DOWNTOWN"])
def test_variants_are_flagged_not_applied(variant: str) -> None:
    jobs = _jobs([{PK: J1, COUNT: 1}])
    cars = _cars([(J1, variant, CITY)])
    report = _investigate(jobs, cars)
    assert (report.status, report.earliest_failing_stage) == (S.UNVERIFIED_ALIAS, P.LOCATION_MATCHING)
    assert report.unverified_representation_variant and report.authoritative_mapping_required
    assert not report.repository_fix_required and report.present_after_cleaning is False
    assert cars[LOC].tolist() == [variant]                                  # source untouched


def test_exact_match_recognised_and_airport_downtown_distinct() -> None:
    report = _investigate(*_healthy_inputs())
    assert report.present_after_cleaning and report.matched_via_authoritative_alias is False
    airport_only = _investigate(_jobs([{PK: J1, COUNT: 1}]), _cars([(J1, AIRPORT, CITY)]))
    assert airport_only.status is S.RAW_STREAM_ABSENT


def test_component_order_error_is_not_a_match() -> None:
    composite = dataclasses.replace(COV, location_columns=(SCOPE, LOC),
                                    expected_locations=((CITY, DOWNTOWN),), stream_scope_columns=())
    cars = _cars([(J1, CITY, DOWNTOWN)])        # components swapped between the two columns
    report = investigate_location_stream(_jobs([{PK: J1, COUNT: 1}]), cars, (CITY, DOWNTOWN), coverage=composite)
    assert report.status is S.UNVERIFIED_ALIAS and report.present_after_cleaning is False


def test_authoritative_alias_only_through_central_definition() -> None:
    alias = "SYNTH-DOWNTOWN-BRANCH"
    jobs = _jobs([{PK: J1, COUNT: 1}])
    cars = _cars([(J1, alias, CITY)])
    assert _investigate(jobs, cars).status is S.RAW_STREAM_ABSENT          # not applied implicitly
    aliased = dataclasses.replace(COV, aliases={TARGET: ((alias,),)})
    report = _investigate(jobs, cars, coverage=aliased)
    assert report.is_healthy and report.matched_via_authoritative_alias
    assert cars[LOC].tolist() == [alias]


# ------------------------------------------------------------ pipeline behaviour


def test_pipeline_blank_rows_and_partial_rows(tmp_path: Path) -> None:
    def write(key: DatasetKey, lines: list[str]) -> None:
        text = ",".join(contract_columns(key)) + "\n" + "".join(line + "\n" for line in lines)
        (tmp_path / f"synthetic_{key}.csv").write_bytes(text.encode("utf-8"))

    def row(key: DatasetKey, values: dict[str, str]) -> str:
        return ",".join(values.get(c, "") for c in contract_columns(key))

    write(JOBS, [row(JOBS, {PK: "000001", COUNT: "2"}), ""])
    write(CARS, [
        row(CARS, {DK: "000001", LOC: DOWNTOWN, SCOPE: CITY}),    # mostly empty, but not blank
        "", "," * (len(contract_columns(CARS)) - 1),               # completely blank -> removed
        row(CARS, {DK: "000001", LOC: AIRPORT, SCOPE: CITY}),
    ])
    loaded = load_raw_datasets(tmp_path)
    cleaned = remove_blank_rows_from_raw_datasets(loaded).cleaned
    report = investigate_location_stream(cleaned.jobs, cleaned.cars, TARGET, coverage=COV, loaded=loaded,
                                         raw_source=tmp_path / f"synthetic_{CARS}.csv")
    assert report.present_in_raw_source and report.present_after_ingestion and report.present_after_cleaning
    assert report.identifier_dtypes_valid and report.is_healthy           # leading-zero keys still link
    assert len(cleaned.cars) == 2


def test_absent_target_does_not_need_relationship_checks() -> None:
    report = _investigate(_jobs([{PK: J1, COUNT: 1}, {PK: J1, COUNT: 1}]), _cars([(J1, AIRPORT, CITY)]))
    assert report.status is S.RAW_STREAM_ABSENT and report.parent_keys_valid is None


# --------------------------------------------------------- safety and mutation


def test_report_is_categorical_and_safe() -> None:
    report = _investigate(*_healthy_inputs())
    for field in dataclasses.fields(report):
        value = getattr(report, field.name)
        assert value is None or isinstance(value, (bool, S, P, StreamContinuity, TimeCoverageStatus, tuple)), field.name
        if isinstance(value, tuple):
            assert all(isinstance(v, P) for v in value)
    text = repr(report)
    for raw in (DOWNTOWN, AIRPORT, CITY, J1, J2, "SYNTH"):
        assert raw not in text
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.status = S.RAW_STREAM_ABSENT  # type: ignore[misc]


def test_inputs_unchanged_idempotent_and_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    jobs, cars = _healthy_inputs()
    jobs.index = pd.Index([20, 10], name="synthetic_label")
    cars.index = pd.Index([4, 3, 2, 1])
    snapshots = (jobs.copy(deep=True), cars.copy(deep=True))
    first, second = _investigate(jobs, cars), _investigate(jobs, cars)
    assert first == second
    for frame, snapshot in zip((jobs, cars), snapshots, strict=True):
        pd.testing.assert_frame_equal(frame, snapshot)
        assert frame.index.equals(snapshot.index) and frame.dtypes.equals(snapshot.dtypes)
    assert list(tmp_path.iterdir()) == []


def test_package_exposes_stream_api() -> None:
    for name in ("investigate_location_stream", "validate_location_stream", "resolve_expected_location",
                 "LocationStreamInvestigationReport", "LocationStreamStatus", "PipelineStage",
                 "INVESTIGATED_LOCATION_STREAM", "COLLECTION_SCHEDULE"):
        assert name in ql2_sixt_canada_analysis.__all__ and hasattr(ql2_sixt_canada_analysis, name)
    assert IDENTIFIER_DTYPE is not None
