"""Tests for the central temporal contract, parsing and reconciliation.

All timestamps, dates and identifiers are fabricated. Field references come
from ``TEMPORAL_RECONCILIATION``; synthetic rules (ordering, reporting dates,
time zones) are configured per test because the real contract leaves them
unavailable until an authority defines them.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import contract_columns

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.ingestion import load_raw_datasets
from ql2_sixt_canada_analysis.quality import remove_blank_rows_from_raw_datasets
from ql2_sixt_canada_analysis.reconciliation import assess_job_detail_reconciliation
from ql2_sixt_canada_analysis.relationships import RelationshipPreconditionError, assess_one_to_many_join
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    JOB_DETAIL_RELATIONSHIP,
    TEMPORAL_RECONCILIATION as T,
    DatasetKey,
    ReportingDateRule,
    TemporalAwareness as A,
    TemporalConfigurationError,
    TemporalDateCheck,
    TemporalFieldDefinition,
    TemporalKind as K,
    TemporalReconciliationDefinition,
    TimestampOrderingRule,
)
from ql2_sixt_canada_analysis.temporal import (
    RuleStatus,
    TemporalParseError,
    TemporalPreconditionError,
    TemporalReconciliationError,
    TemporalReconciliationReport,
    assess_temporal_reconciliation,
    parse_temporal_field,
    validate_temporal_reconciliation,
)
from ql2_sixt_canada_analysis.unique_keys import assess_raw_dataset_unique_keys

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
REL = JOB_DETAIL_RELATIONSHIP
PK, DK = REL.parent_key_columns[0], REL.detail_key_columns[0]
FIN = T.replications[0].source                     # job finish time (jobs)
JFIN = T.replications[0].replica                   # its copy on detail rows
SCR = next(f.ref for f in T.fields if f.awareness is A.DESIGNATOR)   # per-detail scrape time
JDATE, CDATE, CLEAN = (c.target for c in T.date_checks)               # jobs/cars dates
J1, J2 = "SYNTH-JOB-001", "SYNTH-JOB-002"
EDMONTON = "America/Edmonton"


def _offset(ref: tuple[DatasetKey, str], kind: K = K.TIMESTAMP) -> TemporalFieldDefinition:
    if kind is K.DATE:
        return TemporalFieldDefinition(ref[0], ref[1], K.DATE, True, "%Y-%m-%d", A.NOT_APPLICABLE)
    return TemporalFieldDefinition(ref[0], ref[1], K.TIMESTAMP, True, "ISO8601", A.OFFSET)


def _defn(*, ordering: TimestampOrderingRule | None = None, reporting_tz: str | None = None,
          fields: tuple[TemporalFieldDefinition, ...] | None = None) -> TemporalReconciliationDefinition:
    """Synthetic contract: offset-aware timestamps; optional ordering/date rules."""
    fields = fields or (_offset(FIN), _offset(JFIN), _offset(SCR), _offset(JDATE, K.DATE),
                        _offset(CDATE, K.DATE), _offset(CLEAN, K.DATE))
    if reporting_tz is None:
        checks = tuple(TemporalDateCheck(c.target, None) for c in T.date_checks)
    else:
        checks = (TemporalDateCheck(JDATE, ReportingDateRule(FIN, reporting_tz)),
                  TemporalDateCheck(CDATE, ReportingDateRule(SCR, reporting_tz)),
                  TemporalDateCheck(CLEAN, ReportingDateRule(FIN, reporting_tz)))
    return dataclasses.replace(T, fields=fields, ordering=ordering, date_checks=checks)


ORDER = TimestampOrderingRule(earlier=SCR, later=FIN)   # synthetic: scrape before finish


def _frames(jobs: list[dict], cars: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    def build(key: DatasetKey, rows: list[dict]) -> pd.DataFrame:
        cols = contract_columns(key)
        frame = pd.DataFrame({c: pd.Series([r.get(c, f"synthetic-{c}") for r in rows], dtype=object)
                              for c in cols}, columns=list(cols))
        return frame.astype(dict(DATASET_DEFINITIONS[key].identifier_dtypes))
    return build(JOBS, jobs), build(CARS, cars)


def _job(key: str, finished: object, date: object = "2025-01-15") -> dict:
    return {PK: key, FIN[1]: finished, JDATE[1]: date}


def _car(key: object, scraped: object, finished: object, date: object = "2025-01-15", clean: object = "2025-01-15") -> dict:
    return {DK: key, SCR[1]: scraped, JFIN[1]: finished, CDATE[1]: date, CLEAN[1]: clean}


def _valid_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _frames([_job(J1, "2025-01-15T12:00:00Z")],
                   [_car(J1, "2025-01-15T11:00:00Z", "2025-01-15T12:00:00Z"),
                    _car(J1, "2025-01-15T06:30:00-05:00", "2025-01-15T12:00:00Z")])


def _assess(jobs: pd.DataFrame, cars: pd.DataFrame, definition: TemporalReconciliationDefinition) -> TemporalReconciliationReport:
    r = assess_temporal_reconciliation(jobs, cars, definition)
    for f in r.field_reports:
        assert f.row_count == f.valid_count + f.missing_count + f.invalid_count
    for rule in (r.ordering, *r.date_checks, *r.replications):
        assert rule.row_count == rule.passed + rule.failed + rule.unassessable
    return r


# ----------------------------------------------------------------- configuration


def test_real_contract_fields_and_roles() -> None:
    refs = {f.ref for f in T.fields}
    names = {f.column for f in T.fields}
    for wanted in ("finished_at", "scraped_at", "scrape_date", "date_clean"):    # the four authorised names
        assert wanted in names
    for field in T.fields:
        assert field.column in DATASET_DEFINITIONS[field.dataset].columns
        assert isinstance(field.kind, K) and isinstance(field.awareness, A)
        assert (field.kind is K.DATE) == (field.awareness is A.NOT_APPLICABLE)
    assert {r for r in refs if T.field(r).kind is K.DATE} == {JDATE, CDATE, CLEAN}
    assert T.canonical_timezone == "UTC"
    assert not T.field(FIN).resolvable_to_instant          # naive, no authoritative zone
    assert T.field(SCR).resolvable_to_instant              # explicit source designator
    assert T.ordering is None and all(c.rule is None for c in T.date_checks)
    assert set(T.unavailable_rules) == {"timestamp_ordering", *(f"date_derivation:{d}.{c}" for d, c in (JDATE, CDATE, CLEAN))}
    assert all(offset >= -dt.timedelta(hours=14) for offset in T.field(SCR).designator_offsets.values())
    with pytest.raises(dataclasses.FrozenInstanceError):
        T.ordering = ORDER  # type: ignore[misc]
    with pytest.raises(TypeError):
        T.field(SCR).designator_offsets["XYZ"] = dt.timedelta(0)  # type: ignore[index]


@pytest.mark.parametrize("make", [
    lambda: TimestampOrderingRule(SCR, FIN, tolerance=dt.timedelta(seconds=-1)),
    lambda: TimestampOrderingRule(SCR, SCR),
    lambda: ReportingDateRule(FIN, "Not/AZone"),
    lambda: TemporalFieldDefinition(JOBS, FIN[1], K.TIMESTAMP, True, "%Y", A.NOT_APPLICABLE),
    lambda: TemporalFieldDefinition(JOBS, FIN[1], K.DATE, True, "%Y", A.NAIVE),
    lambda: TemporalFieldDefinition(JOBS, FIN[1], K.TIMESTAMP, True, "%Y", A.OFFSET, source_timezone="UTC"),
    lambda: TemporalFieldDefinition(JOBS, FIN[1], K.TIMESTAMP, True, "%Y", A.DESIGNATOR),
    lambda: dataclasses.replace(T, fields=()),
    lambda: dataclasses.replace(T, canonical_timezone="Mars/Olympus"),
    lambda: dataclasses.replace(T, fields=(*T.fields, TemporalFieldDefinition(JOBS, "synthetic_missing_column", K.DATE, True, "%Y-%m-%d", A.NOT_APPLICABLE))),
    lambda: dataclasses.replace(T, ordering=TimestampOrderingRule(JDATE, FIN)),            # a date is not a timestamp
    lambda: dataclasses.replace(T, date_checks=(TemporalDateCheck(FIN, None),)),           # target not a date
    lambda: dataclasses.replace(T, date_checks=(TemporalDateCheck(CLEAN, ReportingDateRule(JDATE, "UTC")),)),
])
def test_invalid_configuration_raises_typed_error(make) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TemporalConfigurationError):
        make()
    assert issubclass(TemporalConfigurationError, ValueError)


def test_tolerance_is_explicit_configuration_not_data() -> None:
    rule = TimestampOrderingRule(SCR, FIN)
    assert rule.tolerance == dt.timedelta(0) and rule.inclusive
    jobs, cars = _valid_pair()
    _assess(jobs, cars, _defn(ordering=rule))
    assert rule.tolerance == dt.timedelta(0)                # unchanged by assessment


# ------------------------------------------------------------------- parsing


def test_offset_parsing_missing_invalid_and_equivalent_instants() -> None:
    field = _offset(SCR)
    s = pd.Series(["2025-01-15T12:00:00Z", "2025-01-15T07:00:00-05:00", None, "", "   ", "not-a-time",
                   "2025-02-30T00:00:00Z", 7, float("inf"), np.nan], dtype=object)
    snapshot = s.copy()
    r = parse_temporal_field(s, field)
    assert r.instants.iloc[0] == r.instants.iloc[1] == pd.Timestamp("2025-01-15T12:00:00Z")
    assert r.missing.tolist() == [False, False, True, True, True, False, False, False, False, True]
    assert r.invalid.tolist() == [False, False, False, False, False, True, True, True, True, False]
    assert not r.instants[r.missing].notna().any()                  # no sentinels
    assert "nan" not in r.wall.astype(str).str.lower().replace("nat", "").tolist()
    pd.testing.assert_series_equal(s, snapshot)


def test_mixed_awareness_is_unresolved_not_guessed() -> None:
    r = parse_temporal_field(pd.Series(["2025-01-15T12:00:00Z", "2025-01-15T12:00:00"]), _offset(SCR))
    assert r.unresolved.tolist() == [False, True] and pd.isna(r.instants.iloc[1])


def test_naive_without_authority_fails_closed_with_authority_resolves() -> None:
    naive = TemporalFieldDefinition(JOBS, FIN[1], K.TIMESTAMP, True, "%Y-%m-%d %H:%M:%S", A.NAIVE)
    r = parse_temporal_field(pd.Series(["2025-01-15 12:00:00"]), naive)
    assert r.unresolved.tolist() == [True] and pd.isna(r.instants.iloc[0])
    zoned = dataclasses.replace(naive, source_timezone=EDMONTON)
    r = parse_temporal_field(pd.Series(["2025-01-15 05:00:00"]), zoned)
    assert r.instants.iloc[0] == pd.Timestamp("2025-01-15T12:00:00Z")


def test_designator_mapping_is_explicit() -> None:
    field = T.field(SCR)
    r = parse_temporal_field(pd.Series(["2025-01-15 05:00:00 MST", "2025-01-15 05:00:00 PST", "2025-01-15 05:00:00"]), field)
    assert r.instants.iloc[0] == pd.Timestamp("2025-01-15T12:00:00Z")
    assert r.unresolved.tolist() == [False, True, False] and r.invalid.tolist() == [False, False, True]


def test_machine_timezone_does_not_change_results(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs, cars = _valid_pair()
    definition = _defn(ordering=ORDER, reporting_tz=EDMONTON)
    results = []
    for zone in ("UTC", "Asia/Tokyo", "America/Los_Angeles"):
        monkeypatch.setenv("TZ", zone)
        if hasattr(time, "tzset"):
            time.tzset()
        results.append(assess_temporal_reconciliation(jobs, cars, definition))
    monkeypatch.delenv("TZ", raising=False)
    if hasattr(time, "tzset"):
        time.tzset()
    assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------- ordering


@pytest.mark.parametrize(
    ("scraped", "inclusive", "tolerance", "passes"),
    [
        ("2025-01-15T11:00:00Z", True, 0, True),
        ("2025-01-15T12:00:00Z", True, 0, True),        # equality, inclusive
        ("2025-01-15T12:00:00Z", False, 0, False),      # equality, exclusive
        ("2025-01-15T12:00:01Z", True, 0, False),       # wrong direction
        ("2025-01-15T12:05:00Z", True, 300, True),      # exactly at tolerance
        ("2025-01-15T12:05:01Z", True, 300, False),     # just outside tolerance
    ],
)
def test_ordering_direction_equality_and_tolerance(scraped: str, inclusive: bool, tolerance: int, passes: bool) -> None:
    rule = TimestampOrderingRule(SCR, FIN, inclusive=inclusive, tolerance=dt.timedelta(seconds=tolerance))
    jobs, cars = _frames([_job(J1, "2025-01-15T12:00:00Z")], [_car(J1, scraped, "2025-01-15T12:00:00Z")])
    r = _assess(jobs, cars, _defn(ordering=rule))
    assert (r.ordering.passed, r.ordering.failed) == ((1, 0) if passes else (0, 1))


def test_each_detail_is_assessed_and_one_violation_is_not_hidden() -> None:
    jobs, cars = _frames([_job(J1, "2025-01-15T12:00:00Z")],
                         [_car(J1, "2025-01-15T10:00:00Z", "2025-01-15T12:00:00Z")] * 4
                         + [_car(J1, "2025-01-15T13:00:00Z", "2025-01-15T12:00:00Z")])
    r = _assess(jobs, cars, _defn(ordering=ORDER))
    assert (r.ordering.passed, r.ordering.failed) == (4, 1) and not r.timestamp_ordering_valid
    assert "ordering" in r.violations


def test_missing_invalid_and_unlinked_are_unassessable() -> None:
    jobs, cars = _frames([_job(J1, "2025-01-15T12:00:00Z")],
                         [_car(J1, None, "2025-01-15T12:00:00Z"), _car(J1, "garbage", "2025-01-15T12:00:00Z"),
                          _car(None, "2025-01-15T11:00:00Z", "x"), _car("SYNTH-JOB-404", "2025-01-15T11:00:00Z", "x"),
                          _car(J1, "2025-01-15T11:00:00Z", "2025-01-15T12:00:00Z")])
    r = _assess(jobs, cars, _defn(ordering=ORDER))
    assert (r.ordering.passed, r.ordering.failed, r.ordering.unassessable) == (1, 0, 4)
    assert r.unlinked_detail_row_count == 2 and not r.is_valid
    scr = next(f for f in r.field_reports if f.column == SCR[1])
    assert (scr.missing_count, scr.invalid_count) == (1, 1)


# ---------------------------------------------------------------- date derivation


def test_reporting_date_after_timezone_conversion_and_boundaries() -> None:
    # 2025-01-16T03:00Z is still 2025-01-15 in Edmonton (UTC-7 in winter).
    jobs, cars = _frames([_job(J1, "2025-01-16T03:00:00Z", "2025-01-15")],
                         [_car(J1, "2025-01-16T03:00:00Z", "2025-01-16T03:00:00Z", "2025-01-15", "2025-01-15")])
    r = _assess(jobs, cars, _defn(reporting_tz=EDMONTON))
    for check in r.date_checks:
        assert (check.passed, check.failed, check.boundary_crossing) == (1, 0, 1)
    utc = _assess(jobs, cars, _defn(reporting_tz="UTC"))                # UTC basis gives another date
    assert all(c.failed == 1 for c in utc.date_checks)


@pytest.mark.parametrize(
    ("instant", "date", "zone"),
    [
        ("2025-01-15T00:00:00Z", "2025-01-15", "UTC"),                 # UTC midnight
        ("2025-01-15T07:00:00Z", "2025-01-15", EDMONTON),              # local midnight
        ("2025-01-31T23:59:59Z", "2025-01-31", "UTC"),                 # month end
        ("2026-01-01T06:59:59Z", "2025-12-31", EDMONTON),              # year end, local
        ("2026-01-01T00:00:00Z", "2026-01-01", "UTC"),
        ("2024-02-29T12:00:00Z", "2024-02-29", "UTC"),                 # leap day
        ("2025-03-09T09:30:00Z", "2025-03-09", EDMONTON),              # after spring-forward
        ("2025-11-02T08:30:00Z", "2025-11-02", EDMONTON),              # fall-back repeated hour
    ],
)
def test_calendar_boundaries(instant: str, date: str, zone: str) -> None:
    jobs, cars = _frames([_job(J1, instant, date)], [_car(J1, instant, instant, date, date)])
    r = _assess(jobs, cars, _defn(reporting_tz=zone))
    assert all(c.passed == 1 for c in r.date_checks) and r.date_derivation_valid


def test_mismatching_dates_detected_and_missing_vs_invalid() -> None:
    jobs, cars = _frames([_job(J1, "2025-01-15T12:00:00Z", "2025-01-14")],
                         [_car(J1, "2025-01-15T12:00:00Z", "2025-01-15T12:00:00Z", "2025-02-30", None)])
    r = _assess(jobs, cars, _defn(reporting_tz="UTC"))
    by_name = {c.name: c for c in r.date_checks}
    assert by_name[f"date_derivation:{JDATE[0]}.{JDATE[1]}"].failed == 1
    assert by_name[f"date_derivation:{CDATE[0]}.{CDATE[1]}"].unassessable == 1   # invalid date
    assert by_name[f"date_derivation:{CLEAN[0]}.{CLEAN[1]}"].unassessable == 1   # missing date
    fields = {f.ref: f for f in r.field_reports}
    assert fields[CDATE].invalid_count == 1 and fields[CLEAN].missing_count == 1
    assert "date_derivation" in r.violations and "field_parse" in r.violations


@pytest.mark.parametrize(("text", "valid"), [("2025-1-15", True), ("2025-01-15", True),
                                             ("2025-01-15 00:00:00", False), ("15/01/2025", False)])
def test_date_format_policy(text: str, valid: bool) -> None:
    r = parse_temporal_field(pd.Series([text]), _offset(CLEAN, K.DATE))
    assert bool(r.valid[0]) is valid
    if valid:
        assert r.wall.iloc[0] == pd.Timestamp("2025-01-15")


# -------------------------------------------------------------- daylight saving


def test_daylight_saving_naive_policy() -> None:
    field = TemporalFieldDefinition(JOBS, FIN[1], K.TIMESTAMP, True, "%Y-%m-%d %H:%M:%S", A.NAIVE, EDMONTON)
    r = parse_temporal_field(pd.Series(["2025-11-02 01:30:00", "2025-03-09 02:30:00", "2025-03-09 03:30:00"]), field)
    assert r.unresolved.tolist() == [True, True, False]           # ambiguous, nonexistent -> unresolved
    assert r.instants.iloc[2] == pd.Timestamp("2025-03-09T09:30:00Z")
    aware = parse_temporal_field(pd.Series(["2025-11-02T01:30:00-06:00", "2025-11-02T01:30:00-07:00"]), _offset(SCR))
    assert aware.instants.iloc[1] - aware.instants.iloc[0] == pd.Timedelta(hours=1)


# ----------------------------------------------------------------- relationships


def test_duplicate_parent_keys_raise_precondition_error() -> None:
    jobs, cars = _frames([_job(J1, "2025-01-15T12:00:00Z"), _job(J1, "2025-01-15T12:00:00Z")],
                         [_car(J1, "2025-01-15T11:00:00Z", "2025-01-15T12:00:00Z")])
    with pytest.raises(TemporalPreconditionError) as info:
        assess_temporal_reconciliation(jobs, cars, _defn(ordering=ORDER))
    assert isinstance(info.value, RelationshipPreconditionError) and J1 not in str(info.value)


def test_relationship_keys_are_tuples_not_concatenations() -> None:
    jobs, cars = _frames([_job("A-B", "2025-01-15T12:00:00Z"), _job("A", "2025-01-15T12:00:00Z")],
                         [_car("A-B", "2025-01-15T11:00:00Z", "2025-01-15T12:00:00Z")])
    r = _assess(jobs, cars, _defn(ordering=ORDER))
    assert r.unlinked_detail_row_count == 0 and r.ordering.passed == 1


def test_composite_relationship_links_by_all_components(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import MappingProxyType
    ordinal = REL.detail_definition.non_identifier_key_columns[0]
    parent = dataclasses.replace(REL.parent_definition, columns=(*REL.parent_definition.columns, "synth_part"),
                                 identifier_columns=(PK, "synth_part"), unique_key_columns=(PK, "synth_part"))
    detail = dataclasses.replace(REL.detail_definition, columns=(*REL.detail_definition.columns, "synth_part"),
                                 identifier_columns=(DK, "synth_part"), unique_key_columns=(DK, "synth_part", ordinal))
    registry = MappingProxyType({JOBS: parent, CARS: detail})
    rel = dataclasses.replace(REL, parent_key_columns=(PK, "synth_part"), detail_key_columns=(DK, "synth_part"),
                              definitions=registry)
    definition = dataclasses.replace(_defn(ordering=ORDER), relationship=rel)
    jobs, cars = _frames([_job(J1, "2025-01-15T12:00:00Z")],
                         [_car(J1, "2025-01-15T11:00:00Z", "2025-01-15T12:00:00Z"),
                          _car(J1, "2025-01-15T11:00:00Z", "2025-01-15T12:00:00Z")])
    jobs = jobs.assign(synth_part=pd.array(["P1"], dtype="string"))
    cars = cars.assign(synth_part=pd.array(["P1", "P2"], dtype="string"))
    r = assess_temporal_reconciliation(jobs, cars, definition)
    assert r.unlinked_detail_row_count == 1 and r.ordering.passed == 1


def test_replication_compares_copies_to_their_parent() -> None:
    jobs, cars = _frames([_job(J1, "2025-01-15 12:00:00.000")],
                         [_car(J1, "2025-01-15 05:00:00 MST", "2025-01-15 12:00:00.000"),
                          _car(J1, "2025-01-15 05:00:00 MST", "2025-01-15 12:00:00.001")])
    r = _assess(jobs, cars, T)                                    # real contract: naive wall-time copies
    assert (r.replications[0].passed, r.replications[0].failed) == (1, 1)
    assert "replication" in r.violations


# --------------------------------------------------------------- report invariants


def test_unavailable_rules_fail_closed_and_validation_types() -> None:
    jobs, cars = _valid_pair()
    configured = _defn(ordering=ORDER, reporting_tz="UTC")
    assert _assess(jobs, cars, configured).is_valid
    assert validate_temporal_reconciliation(jobs, cars, configured).is_valid
    open_rules = _assess(jobs, cars, _defn())
    assert not open_rules.is_valid and "rule_unavailable" in open_rules.violations
    assert open_rules.ordering.status is RuleStatus.UNAVAILABLE and open_rules.ordering.passed == 0
    with pytest.raises(TemporalReconciliationError) as info:
        validate_temporal_reconciliation(jobs, cars, _defn())
    assert not isinstance(info.value, TemporalParseError) and "rule_unavailable" in info.value.violations
    bad_jobs, bad_cars = _frames([_job(J1, "garbage")], [_car(J1, "2025-01-15T11:00:00Z", "2025-01-15T12:00:00Z")])
    with pytest.raises(TemporalParseError) as parse_info:
        validate_temporal_reconciliation(bad_jobs, bad_cars, configured)
    message = str(parse_info.value)
    assert "garbage" not in message and "2025" not in message and J1 not in message


def test_report_holds_aggregates_only() -> None:
    jobs, cars = _valid_pair()
    r = _assess(jobs, cars, _defn(ordering=ORDER, reporting_tz=EDMONTON))
    def walk(obj: object) -> None:
        assert not isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray, pd.Timestamp, list, set, dict))
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                walk(getattr(obj, f.name))
        elif isinstance(obj, tuple):
            for item in obj:
                walk(item)
    walk(r)
    assert "2025" not in repr(r) and J1 not in repr(r)


def test_empty_frames_policy() -> None:
    jobs, cars = _frames([], [])
    assert _assess(jobs, cars, _defn(ordering=ORDER, reporting_tz="UTC")).is_valid
    assert not _assess(jobs, cars, T).is_valid                    # unavailable rules still fail closed


# ------------------------------------------------------------------ non-mutation


def test_sources_unchanged_idempotent_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    jobs, cars = _valid_pair()
    jobs.index = pd.Index([7], name="synthetic_label")
    cars.index = pd.Index([9, 8])
    snapshots = (jobs.copy(deep=True), cars.copy(deep=True))
    definition = _defn(ordering=ORDER, reporting_tz=EDMONTON)
    assert assess_temporal_reconciliation(jobs, cars, definition) == assess_temporal_reconciliation(jobs, cars, definition)
    for frame, snapshot in zip((jobs, cars), snapshots, strict=True):
        pd.testing.assert_frame_equal(frame, snapshot)
        assert frame.index.equals(snapshot.index) and frame.dtypes.equals(snapshot.dtypes)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------- pipeline


def test_pipeline_blank_rows_and_partial_rows(tmp_path: Path) -> None:
    def write(key: DatasetKey, rows: list[dict], blanks: int) -> None:
        cols = contract_columns(key)
        lines = [",".join(str(r.get(c, f"synthetic-{i}")) for i, c in enumerate(cols)) for r in rows]
        lines += [""] * blanks + ["," * (len(cols) - 1)]
        (tmp_path / f"synthetic_{key}.csv").write_bytes((",".join(cols) + "\n" + "\n".join(lines) + "\n").encode())

    write(JOBS, [{PK: "000001", FIN[1]: "2025-01-15 12:00:00.000", JDATE[1]: "2025-01-15"}], 1)
    write(CARS, [{DK: "000001", SCR[1]: "2025-01-15 05:00:00 MST", JFIN[1]: "2025-01-15 12:00:00.000",
                  CDATE[1]: "2025-01-15", CLEAN[1]: "2025-01-15"},
                 {DK: "000001", SCR[1]: "", JFIN[1]: "2025-01-15 12:00:00.000"}], 2)   # partial row
    cleaned = remove_blank_rows_from_raw_datasets(load_raw_datasets(tmp_path)).cleaned
    keys = assess_raw_dataset_unique_keys(cleaned)
    counts = assess_job_detail_reconciliation(cleaned.jobs, cleaned.cars)
    relationship = assess_one_to_many_join(cleaned.jobs, cleaned.cars)
    r = assess_temporal_reconciliation(cleaned.jobs, cleaned.cars)
    scr = next(f for f in r.field_reports if f.ref == SCR)
    assert r.detail_row_count == 2 and scr.missing_count == 1          # blank rows gone, partial row kept
    assert r.replications[0].passed == 2 and r.unlinked_detail_row_count == 0
    assert str(cleaned.jobs[PK].dtype) == str(DATASET_DEFINITIONS[JOBS].identifier_dtypes[PK])
    assert relationship.is_valid and keys.jobs.is_valid and counts is not None   # separate controls
    assert not r.is_valid and "rule_unavailable" in r.violations       # no downstream trust yet


def test_package_exposes_temporal_api() -> None:
    for name in ("assess_temporal_reconciliation", "validate_temporal_reconciliation", "parse_temporal_field",
                 "TemporalReconciliationReport", "TemporalReconciliationError", "TemporalParseError",
                 "TEMPORAL_RECONCILIATION", "TemporalConfigurationError"):
        assert name in ql2_sixt_canada_analysis.__all__ and hasattr(ql2_sixt_canada_analysis, name)
    assert os.sep  # (no filesystem use)
