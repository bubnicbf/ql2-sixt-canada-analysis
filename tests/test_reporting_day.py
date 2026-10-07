"""Approved reporting day, strict scrape dates and the retired cleaned date (pricing-authorities-v8).

Records are read as data and mutated in memory for negative cases. Frames are
fabricated (``SYNTH-JOB-*``); dates and times are synthetic test inputs, never
source extracts. The only real values are approved configuration (cities and
zones).
"""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from test_completeness import CARS, JOBS, frame

from ql2_sixt_canada_analysis.authority_decisions import (
    DecisionId,
    DecisionRecordError,
    DecisionStatus,
    load_decision_record,
    parse_decision_record,
)
from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
from ql2_sixt_canada_analysis.schemas import (
    FINISHED_AT_TIMEZONE_SELECTOR,
    ISO_8601_DATE_FORMAT,
    TEMPORAL_RECONCILIATION,
    CityTimezoneMap,
    ReportingDateRule,
    TemporalConfigurationError,
)
from ql2_sixt_canada_analysis.temporal import (
    ScrapeDateStatus as SDS,
    assess_temporal_reconciliation,
    derive_reporting_days,
)
from ql2_sixt_canada_analysis.temporal_authority import (
    DATE_CLEAN_FIELD,
    SCRAPE_DATE_FIELDS,
    TemporalAuthorityBlocker as TB,
    TemporalDecisionStatus as TS,
    current_temporal_authority,
    temporal_authority_from_record,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V7, V8 = RECORD_DIR / "v7.toml", RECORD_DIR / "v8.toml"
GOVERNANCE = "docs/decisions/governance/reporting-day-and-source-date-governance-v1-2026-10-06.md"
D = DecisionId
ZONES = {"calgary": "America/Edmonton", "toronto": "America/Toronto", "vancouver": "America/Vancouver"}
CONTRACT = current_expected_stream_contract()
AUTHORITY = temporal_authority_from_record(load_decision_record(V8), TEMPORAL_RECONCILIATION, CONTRACT)
DEF = AUTHORITY.definition                                  # raw-key contract under the v8 authority
SCRAPED = "2026-01-01 00:00:00 MST"                         # always before the synthetic finish times


def v8() -> dict:
    return tomllib.loads(V8.read_text(encoding="utf-8"))


def entry(data: dict, decision: DecisionId) -> dict:
    return next(e for e in data["decisions"] if e["id"] == decision.value)


def finish(data: dict) -> dict:
    statuses = [e["status"] for e in data["decisions"]]
    data["summary"] = {s.value.lower(): statuses.count(s.value) for s in DecisionStatus}
    data["external_inputs"] = [e["id"] for e in data["decisions"] if e["blocking_external_input"]]
    return data


def fails(data: dict) -> str:
    with pytest.raises(DecisionRecordError) as info:
        parse_decision_record(finish(data))
    return str(info.value)


def frames(jobs, details):  # type: ignore[no-untyped-def]
    """jobs: (job, city, finished_at, scrape_date); details: (job, city, scrape_date, date_clean[, job_finished])."""
    fin = {j: f for j, _, f, _ in jobs}
    job_rows = [{"job_id": j, "city": c, "finished_at": f, "scrape_date": s, "record_count": "1",
                 "actual_car_rows": "1"} for j, c, f, s in jobs]
    car_rows = [{"job_id": d[0], "row_index": str(i), "city": d[1], "scrape_date": d[2], "date_clean": d[3],
                 "job_finished_at": d[4] if len(d) > 4 else fin.get(d[0], "2026-07-15 12:00:00.000"),
                 "scraped_at": SCRAPED} for i, d in enumerate(details)]
    return frame(JOBS, job_rows), frame(CARS, car_rows)


# ======================================================================= authority


def test_v8_approves_the_four_date_decisions_as_supplied() -> None:
    record = load_decision_record(V8)
    assert dict(record.approved_resolution(D.REPORTING_DAY_SOURCE)) == {"field": "jobs.finished_at"}
    zone = record.approved_resolution(D.REPORTING_DAY_TIMEZONE)
    assert zone["mode"] == "PARENT_CITY" and {z["city"]: z["timezone"] for z in zone["city_timezones"]} == ZONES
    assert record.approved_resolution(D.SCRAPE_DATE_SEMANTICS)["derivation"] == "REPORTING_DAY"
    assert record.approved_resolution(D.DATE_CLEAN_SEMANTICS)["derivation"] == "RETIRED_FROM_PRICING"
    for decision, kinds in ((D.REPORTING_DAY_SOURCE, {"BUSINESS_OWNER"}), (D.REPORTING_DAY_TIMEZONE, {"BUSINESS_OWNER"}),
                            (D.SCRAPE_DATE_SEMANTICS, {"COLLECTION_OWNER"}),
                            (D.DATE_CLEAN_SEMANTICS, {"COLLECTION_OWNER"})):
        item = record.decision(decision)
        assert {a.kind.value for a in item.authority} == kinds and {a.reference for a in item.authority} == {GOVERNANCE}
    v7 = load_decision_record(V7)
    assert all(v7.decision(d).status is DecisionStatus.PROPOSED for d in (
        D.REPORTING_DAY_SOURCE, D.REPORTING_DAY_TIMEZONE, D.SCRAPE_DATE_SEMANTICS, D.DATE_CLEAN_SEMANTICS))


@pytest.mark.parametrize("decision, change, needle", [
    (D.REPORTING_DAY_SOURCE, {"field": "cars.scraped_at"}, "parent finish time"),       # never the scrape time
    (D.REPORTING_DAY_SOURCE, {"field": "cars.job_finished_at"}, "parent finish time"),
    (D.REPORTING_DAY_SOURCE, {"field": "jobs.scrape_date"}, "REPORTING_DAY_SOURCE"),
    (D.REPORTING_DAY_TIMEZONE, {"mode": "FIXED_ZONE"}, "unsupported value"),
    (D.REPORTING_DAY_TIMEZONE, {"timezone": "America/Toronto"}, "exactly the required fields"),
    (D.SCRAPE_DATE_SEMANTICS, {"derivation": "RETIRED_FROM_PRICING"}, "unsupported value"),
    (D.DATE_CLEAN_SEMANTICS, {"derivation": "CLEANED"}, "unsupported value"),
    (D.DATE_CLEAN_SEMANTICS, {"meaning": "2026-07-15 12:00 SYNTH"}, "source-level"),
])
def test_date_decision_shapes_fail_closed(decision, change, needle) -> None:
    data = v8()
    entry(data, decision)["resolution"].update(change)
    message = fails(data)
    assert needle in message or (needle == "source-level" and "DATE_CLEAN_SEMANTICS" in message)


@pytest.mark.parametrize("mutate", [
    lambda z: z.pop(),                                                                  # a city missing
    lambda z: z.append({"city": "montreal", "timezone": "America/Toronto"}),
    lambda z: z[0].update(city="Calgary"),
    lambda z: z[0].update(timezone="UTC"),
    lambda z: z[0].update(timezone="-07:00"),
])
def test_the_reporting_day_city_map_must_be_exact_and_exhaustive(mutate) -> None:
    data = v8()
    mutate(entry(data, D.REPORTING_DAY_TIMEZONE)["resolution"]["city_timezones"])
    assert "REPORTING_DAY_TIMEZONE" in fails(data)


def test_scrape_dates_need_the_approved_reporting_day() -> None:
    data = v8()
    item = entry(data, D.REPORTING_DAY_SOURCE)
    item.update(status="PROPOSED", blocking_external_input=True)
    item.pop("authority"), item.pop("resolution")
    assert "requires REPORTING_DAY_SOURCE" in fails(data)


def test_the_v8_temporal_contract_configures_both_scrape_dates_and_retires_date_clean() -> None:
    assert current_temporal_authority().record_id == "pricing-authorities-v8"
    assert (AUTHORITY.reporting_day_status, AUTHORITY.date_semantics_status) == (TS.APPROVED, TS.APPROVED)
    assert AUTHORITY.blocking_reasons == () and AUTHORITY.reporting_day_available and AUTHORITY.date_clean_retired
    assert AUTHORITY.pricing_date_fields == ("jobs.finished_at",) and "cars.date_clean" not in AUTHORITY.pricing_date_fields
    assert dict(AUTHORITY.reporting_day_timezones.entries) == ZONES
    assert [c.target for c in DEF.date_checks] == list(SCRAPE_DATE_FIELDS)
    for check in DEF.date_checks:
        rule = check.rule
        assert rule.source == (JOBS, "finished_at") and rule.city_local and rule.reporting_timezone is None
        assert rule.timezone_selector == FINISHED_AT_TIMEZONE_SELECTOR
    assert DEF.retired_fields == (DATE_CLEAN_FIELD,)
    assert DEF.unavailable_rules == ()                       # no date_derivation:cars.date_clean any more
    for ref in (*SCRAPE_DATE_FIELDS, DATE_CLEAN_FIELD):
        assert DEF.field(ref).source_format == ISO_8601_DATE_FORMAT
    assert GOVERNANCE in AUTHORITY.references


@pytest.mark.parametrize("decision, derivation", [
    (D.DATE_CLEAN_SEMANTICS, "REPORTING_DAY"), (D.DATE_CLEAN_SEMANTICS, "SOURCE_SUPPLIED_UNDERIVED"),
    (D.SCRAPE_DATE_SEMANTICS, "SOURCE_SUPPLIED_UNDERIVED")])
def test_unimplemented_date_semantics_are_invalid_not_guessed(decision, derivation) -> None:
    data = v8()
    entry(data, decision)["resolution"]["derivation"] = derivation
    authority = temporal_authority_from_record(parse_decision_record(finish(data)), TEMPORAL_RECONCILIATION, CONTRACT)
    assert (authority.reporting_day_status, authority.date_semantics_status) == (TS.INVALID, TS.INVALID)
    assert {TB.REPORTING_DAY_UNRESOLVED, TB.DATE_SEMANTICS_UNRESOLVED} <= set(authority.blocking_reasons)
    assert all(c.rule is None for c in authority.definition.date_checks) and authority.pricing_date_fields == ()


def test_reporting_date_rules_have_exactly_one_zone_basis() -> None:
    zones = CityTimezoneMap(tuple(ZONES.items()))
    ReportingDateRule((JOBS, "finished_at"), "America/Toronto")
    ReportingDateRule((JOBS, "finished_at"), city_timezones=zones, timezone_selector=FINISHED_AT_TIMEZONE_SELECTOR)
    for kwargs in ({}, {"reporting_timezone": "America/Toronto", "city_timezones": zones,
                        "timezone_selector": FINISHED_AT_TIMEZONE_SELECTOR},
                   {"city_timezones": zones}, {"reporting_timezone": "Not/AZone"}):
        with pytest.raises(TemporalConfigurationError):
            ReportingDateRule((JOBS, "finished_at"), **kwargs)


# =================================================================== derivation


@pytest.mark.parametrize("city, finished, day, utc_day", [
    ("toronto", "2026-07-15 21:30:00.000", "2026-07-15", "2026-07-16"),     # EDT: the UTC date is the next day
    ("vancouver", "2026-07-15 23:59:59.999", "2026-07-15", "2026-07-16"),
    ("calgary", "2026-01-15 18:00:00.000", "2026-01-15", "2026-01-16"),     # MST, winter
    ("calgary", "2026-07-15 00:00:00.000", "2026-07-15", "2026-07-15"),     # local midnight
    ("toronto", "2026-11-01 00:30:00.000", "2026-11-01", "2026-11-01"),     # fall-back day, unambiguous hour
])
def test_the_reporting_day_is_the_parent_city_local_date_of_the_finish_instant(city, finished, day, utc_day) -> None:
    j, c = frames([("SYNTH-JOB-1", city, finished, day)], [("SYNTH-JOB-1", city, day, "2001-01-01")])
    derived = derive_reporting_days(j, c, DEF)
    row = derived.jobs.iloc[0]
    assert row["reporting_day"] == dt.date.fromisoformat(day) and row["reporting_timezone"] == ZONES[city]
    assert row["finished_at_raw"] == finished and row["scrape_date_raw"] == day
    assert row["finished_at_utc"].date() == dt.date.fromisoformat(utc_day)
    assert row["scrape_date_status"] == SDS.AGREES.value and row["reporting_day_provenance"] == "jobs.finished_at@jobs.city"
    detail = derived.cars.iloc[0]
    assert (detail["reporting_day"], detail["scrape_date_status"]) == (dt.date.fromisoformat(day), SDS.AGREES.value)
    assert detail["reporting_day_provenance"].startswith("linked_parent:")
    report = assess_temporal_reconciliation(j, c, DEF)
    assert report.is_valid and report.date_derivation_valid
    crossing = int(day != utc_day)
    assert [(r.passed, r.failed, r.boundary_crossing) for r in report.date_checks] == [(1, 0, crossing)] * 2
    if day != utc_day:                                     # the UTC date is never the reporting day
        j2, c2 = frames([("SYNTH-JOB-1", city, finished, utc_day)], [("SYNTH-JOB-1", city, utc_day, utc_day)])
        assert derive_reporting_days(j2, c2, DEF).jobs["scrape_date_status"].tolist() == [SDS.MISMATCH.value]
        assert not assess_temporal_reconciliation(j2, c2, DEF).is_valid


def test_the_city_selects_the_zone_for_the_same_wall_time() -> None:
    finished = "2026-07-15 22:30:00.000"
    j, c = frames([("SYNTH-JOB-T", "toronto", finished, "2026-07-15"),
                   ("SYNTH-JOB-V", "vancouver", finished, "2026-07-15")],
                  [("SYNTH-JOB-T", "toronto", "2026-07-15", "2026-07-15"),
                   ("SYNTH-JOB-V", "vancouver", "2026-07-15", "2026-07-15")])
    derived = derive_reporting_days(j, c, DEF).jobs
    assert derived["reporting_timezone"].tolist() == ["America/Toronto", "America/Vancouver"]
    assert derived["finished_at_utc"].iloc[0] != derived["finished_at_utc"].iloc[1]
    assert derived["scrape_date_status"].tolist() == [SDS.AGREES.value] * 2


@pytest.mark.parametrize("value, status", [
    ("2026-7-15", SDS.INVALID), (" 2026-07-15", SDS.INVALID), ("2026-07-15 ", SDS.INVALID),
    ("2026-07-15T00:00:00", SDS.INVALID), ("2026/07/15", SDS.INVALID), ("15 Jul 2026", SDS.INVALID),
    ("2026-02-30", SDS.INVALID), ("2026-07-15Z", SDS.INVALID), ("２０２６-07-15", SDS.INVALID),
    ("", SDS.MISSING), ("   ", SDS.MISSING), (None, SDS.MISSING), ("2026-07-14", SDS.MISMATCH),
])
def test_scrape_dates_are_parsed_strictly_and_never_repaired(value, status) -> None:
    j, c = frames([("SYNTH-JOB-1", "toronto", "2026-07-15 12:00:00.000", value)],
                  [("SYNTH-JOB-1", "toronto", value, "2026-07-15")])
    before = (j.copy(deep=True), c.copy(deep=True))
    derived = derive_reporting_days(j, c, DEF)
    assert derived.jobs["scrape_date_status"].tolist() == [status.value]
    assert derived.cars["scrape_date_status"].tolist() == [status.value]
    raw = derived.jobs["scrape_date_raw"].iloc[0]
    assert raw == value or (value is None and pd.isna(raw))           # the raw value is preserved
    assert derived.jobs["scrape_date_parsed"].iloc[0] is None if status is not SDS.MISMATCH else True
    pd.testing.assert_frame_equal(j, before[0]), pd.testing.assert_frame_equal(c, before[1])
    assert not assess_temporal_reconciliation(j, c, DEF).is_valid


def test_detail_rows_need_a_trusted_linked_parent() -> None:
    j, c = frames([("SYNTH-JOB-1", "toronto", "2026-07-15 12:00:00.000", "2026-07-15")],
                  [("SYNTH-JOB-1", "toronto", "2026-07-15", "2026-07-15"),
                   ("SYNTH-JOB-1", "calgary", "2026-07-15", "2026-07-15"),          # city disagrees with parent
                   ("SYNTH-JOB-ORPHAN", "toronto", "2026-07-15", "2026-07-15")])    # no parent
    derived = derive_reporting_days(j, c, DEF)
    assert derived.cars["scrape_date_status"].tolist() == [SDS.AGREES.value, SDS.UNTRUSTED.value,
                                                            SDS.UNLINKED.value]
    assert derived.cars["reporting_day"].tolist()[1:] == [None, None]           # nothing inherited untrusted
    report = assess_temporal_reconciliation(j, c, DEF)
    cars_check = report.date_checks[1]
    assert (cars_check.passed, cars_check.failed, cars_check.unassessable) == (1, 0, 2) and not report.is_valid


def test_an_unknown_parent_city_has_no_reporting_day() -> None:
    j, c = frames([("SYNTH-JOB-1", "Toronto", "2026-07-15 12:00:00.000", "2026-07-15")],
                  [("SYNTH-JOB-1", "Toronto", "2026-07-15", "2026-07-15")])
    derived = derive_reporting_days(j, c, DEF)
    assert derived.jobs["scrape_date_status"].tolist() == [SDS.UNRESOLVABLE.value]
    assert derived.jobs["reporting_timezone"].tolist() == [None]                # never a default zone
    assert derived.cars["scrape_date_status"].tolist() == [SDS.UNRESOLVABLE.value]


def test_the_scrape_time_never_decides_the_reporting_day() -> None:
    j, c = frames([("SYNTH-JOB-1", "toronto", "2026-07-15 12:00:00.000", "2026-07-15")],
                  [("SYNTH-JOB-1", "toronto", "2026-07-15", "2026-07-15")])
    c = c.assign(scraped_at="2026-07-14 01:00:00 MST")                            # a different calendar date
    derived = derive_reporting_days(j, c, DEF)
    assert derived.cars["reporting_day"].tolist() == [dt.date(2026, 7, 15)]
    assert derived.cars["scrape_date_status"].tolist() == [SDS.AGREES.value]


@pytest.mark.parametrize("date_clean", ["2026-07-14", "SYNTH-not-a-date", "", None])
def test_date_clean_is_retired_never_required_to_agree_and_never_blocking(date_clean) -> None:
    j, c = frames([("SYNTH-JOB-1", "toronto", "2026-07-15 12:00:00.000", "2026-07-15")],
                  [("SYNTH-JOB-1", "toronto", "2026-07-15", date_clean)])
    report = assess_temporal_reconciliation(j, c, DEF)
    assert report.is_valid and "date_derivation:cars.date_clean" not in report.unavailable_rules
    clean = next(f for f in report.field_reports if f.ref == DATE_CLEAN_FIELD)
    assert clean.retired and clean.parses                                      # reported, never blocking
    assert (clean.valid_count, clean.invalid_count + clean.missing_count) == (
        (1, 0) if date_clean == "2026-07-14" else (0, 1))
    assert clean.parse_quality_clean is (date_clean == "2026-07-14")
    derived = derive_reporting_days(j, c, DEF)
    assert "date_clean" not in set(derived.cars.columns) | set(derived.jobs.columns)
    assert derived.cars["scrape_date_status"].tolist() == [SDS.AGREES.value]   # date_clean never overrides
    assert c["date_clean"].iloc[0] == date_clean or (date_clean is None and pd.isna(c["date_clean"].iloc[0]))


def test_without_the_approved_rule_there_is_no_reporting_day() -> None:
    v7 = temporal_authority_from_record(load_decision_record(V7), TEMPORAL_RECONCILIATION, CONTRACT)
    j, c = frames([("SYNTH-JOB-1", "toronto", "2026-07-15 12:00:00.000", "2026-07-15")],
                  [("SYNTH-JOB-1", "toronto", "2026-07-15", "2026-07-15")])
    with pytest.raises(TemporalConfigurationError):
        derive_reporting_days(j, c, v7.definition)
    assert "date_derivation:cars.date_clean" in v7.definition.unavailable_rules     # history unchanged


def test_status_counts_are_aggregates_only() -> None:
    j, c = frames([("SYNTH-JOB-1", "toronto", "2026-07-15 12:00:00.000", "2026-07-15"),
                   ("SYNTH-JOB-2", "vancouver", "2026-07-15 12:00:00.000", "2026-07-14")],
                  [("SYNTH-JOB-1", "toronto", "2026-07-15", "2026-07-15"),
                   ("SYNTH-JOB-2", "vancouver", "2026-07-15", "2026-07-15")])
    derived = derive_reporting_days(j, c, DEF)
    assert derived.status_counts(JOBS) == {"agrees": 1, "mismatch": 1, "missing": 0, "invalid": 0,
                                           "unresolvable": 0, "unlinked": 0, "untrusted": 0}
    assert derived.eligible(CARS).tolist() == [True, True]
    assert "SYNTH" not in repr(derived.rule)


# ===================================================================== governance


def test_governance_document_records_the_supplied_decisions_only() -> None:
    raw = (ROOT / GOVERNANCE).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in ("2026-10-06", "business owner", "collection owner", "`jobs.finished_at`", "PARENT_CITY",
                   "REPORTING_DAY", "RETIRED_FROM_PRICING", "ISO_8601_DATE", "trusted linked parent",
                   "never derived from `cars.scraped_at`", "Raw values are preserved", "Nothing is repaired",
                   "never overrides `scrape_date`", "never used for grouping", "never rewritten",
                   "date_derivation:cars.date_clean", "pricing-authorities-v8", "Not supplied"):
        assert phrase in text, phrase
    for city, zone in ZONES.items():
        assert f"| `{city}` | `{zone}` |" in raw
    assert "SYNTH" not in raw and "@" not in raw
