"""Authority-backed city-local finish times, replication and scrape/finish ordering (pricing-authorities-v6).

The committed records are read as data; negative cases mutate parsed copies
in memory. Frames are fabricated (``SYNTH-JOB-*``); the only real values are
the approved configuration (cities and zones) read from the record. Synthetic
timestamps are test inputs, never source extracts.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import re
import time
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from stream_contract_fixtures import synthetic_contract
from test_completeness import CARS, JOBS, frame
from test_readiness import GATES

from ql2_sixt_canada_analysis.authority_decisions import (
    CURRENT_RECORD_PATH,
    AuthorityKind,
    DecisionId,
    DecisionRecordError,
    DecisionStatus,
    load_current_decision_record,
    load_decision_record,
    parse_decision_record,
)
from ql2_sixt_canada_analysis.collection_schedule import current_per_stream_schedule
from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
from ql2_sixt_canada_analysis.readiness import (
    PricingBlocker as PB,
    assess_location_policy,
    assess_pricing_readiness,
)
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    EXPECTED_LOCATION_COVERAGE as COV,
    TEMPORAL_RECONCILIATION,
    CityTimezoneMap,
    TemporalConfigurationError,
    TemporalFieldDefinition,
)
from ql2_sixt_canada_analysis.temporal import (
    assess_temporal_reconciliation,
    canonical_utc_text,
    derive_utc_timestamps,
    parse_temporal_field,
)
from ql2_sixt_canada_analysis.temporal_authority import (
    TemporalAuthorityBlocker as TB,
    TemporalDecisionStatus as TS,
    current_temporal_authority,
    current_temporal_reconciliation,
    temporal_authority_from_record,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V5, V6 = RECORD_DIR / "v5.toml", RECORD_DIR / "v6.toml"
GOVERNANCE = "docs/decisions/governance/finished-at-timezone-and-scrape-ordering-governance-v1-2026-10-06.md"
HISTORY_SHA256 = {
    "v1.toml": "b13881b099885130e853207f872e62bfde9820f438e34859f3ddef0ebdbf1bd7",
    "v2.toml": "899c20e289d932868b1d430b8fd70edca6ebb2900ca963c4c040cc6f5840cb4c",
    "v3.toml": "431ec36a8d60b058a9d2ede401ab0274dea0bc525f8b025f2b04cbdff0aa2d79",
    "v4.toml": "28a412352884a28c7ebf6874aa1bfc25182c13f210357861933001886a25d552",
    "v5.toml": "1958fca6f38a03be5beea12d920346c9f07d1be9dde689c4c5be4527331861ac",
}
D = DecisionId
#: The decisions exactly as supplied (test oracle).
ZONES = {"calgary": "America/Edmonton", "toronto": "America/Toronto", "vancouver": "America/Vancouver"}
UNRESOLVED = (D.REPORTING_DAY_SOURCE, D.REPORTING_DAY_TIMEZONE, D.SCRAPE_DATE_SEMANTICS, D.DATE_CLEAN_SEMANTICS)
AUTHORITY = temporal_authority_from_record(load_decision_record(V6), TEMPORAL_RECONCILIATION,
                                           current_expected_stream_contract())
DEF = AUTHORITY.definition                         # the raw-key contract under the v6 authority
FINISHED = DEF.field((JOBS, "finished_at"))
UTC = dt.timezone.utc


def v6() -> dict:
    return tomllib.loads(V6.read_text(encoding="utf-8"))


def entry(data: dict, decision: DecisionId) -> dict:
    return next(e for e in data["decisions"] if e["id"] == decision.value)


def finish(data: dict) -> dict:
    statuses = [e["status"] for e in data["decisions"]]
    data["summary"] = {s.value.lower(): statuses.count(s.value) for s in DecisionStatus}
    data["external_inputs"] = [e["id"] for e in data["decisions"] if e["blocking_external_input"]]
    return data


def unapprove(data: dict, *decisions: DecisionId) -> dict:
    for decision in decisions:
        item = entry(data, decision)
        item.update(status="PROPOSED", blocking_external_input=True)
        item.pop("authority", None), item.pop("resolution", None)
    return finish(data)


def fails(data: dict) -> str:
    with pytest.raises(DecisionRecordError) as info:
        parse_decision_record(finish(data))
    return str(info.value)


# ======================================================================= authority


def test_history_is_unchanged_and_valid() -> None:
    for name, digest in HISTORY_SHA256.items():
        assert hashlib.sha256((RECORD_DIR / name).read_bytes()).hexdigest() == digest, name
        load_decision_record(RECORD_DIR / name)


def test_v6_adds_exactly_the_ordering_decisions_and_is_carried_forward() -> None:
    record, v5 = load_decision_record(V6), load_decision_record(V5)
    assert (record.schema_version, record.record_version, record.record_id, record.supersedes) == (
        3, 6, "pricing-authorities-v6", "pricing-authorities-v5")
    assert CURRENT_RECORD_PATH.name == "v7.toml"
    current = load_current_decision_record()                  # v7 keeps every v6 decision except the rental pair
    for decision in record.decisions:
        if decision.id not in (D.RENTAL_DATE_VALIDITY, D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS):
            assert current.decision(decision.id) == decision, decision.id
    counts = record.counts()
    assert (counts[DecisionStatus.APPROVED], counts[DecisionStatus.PROPOSED], counts[DecisionStatus.REJECTED]) == (
        16, 6, 0)
    before = {d.id for d in v5.decisions if d.is_approved}
    after = {d.id for d in record.decisions if d.is_approved}
    assert after - before == {D.SCRAPED_FINISHED_ORDERING, D.SCRAPED_FINISHED_TOLERANCE}
    for decision in before:                                  # every earlier approval is preserved unchanged
        assert record.decision(decision).resolution == v5.decision(decision).resolution, decision
        assert set(v5.decision(decision).authority) <= set(record.decision(decision).authority), decision
    for decision in UNRESOLVED:
        assert record.decision(decision).status is DecisionStatus.PROPOSED
        assert record.decision(decision).blocking_external_input


def test_v6_resolutions_and_authorities_are_the_supplied_decisions() -> None:
    record = load_decision_record(V6)
    zones = record.approved_resolution(D.FINISHED_AT_TIMEZONE)["city_timezones"]
    assert [(z["city"], z["timezone"]) for z in zones] == sorted(ZONES.items())
    assert dict(record.approved_resolution(D.SCRAPED_FINISHED_ORDERING)) == {
        "earlier": "cars.scraped_at", "later": "jobs.finished_at", "equal_allowed": True}
    assert dict(record.approved_resolution(D.SCRAPED_FINISHED_TOLERANCE)) == {"tolerance": 0, "unit": "SECONDS"}
    assert [a.kind for a in record.decision(D.SCRAPED_FINISHED_ORDERING).authority] == [AuthorityKind.COLLECTION_OWNER]
    assert [a.kind for a in record.decision(D.SCRAPED_FINISHED_TOLERANCE).authority] == [
        AuthorityKind.COLLECTION_OWNER, AuthorityKind.BUSINESS_OWNER]
    assert record.decision(D.SCRAPED_FINISHED_TOLERANCE).joint
    for decision in (D.FINISHED_AT_TIMEZONE, D.SCRAPED_FINISHED_ORDERING, D.SCRAPED_FINISHED_TOLERANCE):
        assert GOVERNANCE in {a.reference for a in record.decision(decision).authority}
    assert (ROOT / GOVERNANCE).is_file()


@pytest.mark.parametrize("mutate, needle", [
    (lambda z: z.pop(), "exactly the approved cities"),                             # missing city
    (lambda z: z.append({"city": "montreal", "timezone": "America/Toronto"}), "exactly the approved cities"),
    (lambda z: z[0].update(city="Calgary"), "exactly the approved cities"),         # case difference
    (lambda z: z[0].update(city="calgary "), "non-blank"),                          # whitespace difference
    (lambda z: z[0].update(timezone="-07:00"), "region IANA zone"),                 # fixed offset
    (lambda z: z[0].update(timezone="Etc/GMT+7"), "region IANA zone"),
    (lambda z: z[0].update(timezone="UTC"), "region IANA zone"),
    (lambda z: z[0].update(timezone="MST"), "region IANA zone"),                    # abbreviations
    (lambda z: z[0].update(timezone="EDT"), "region IANA zone"),
    (lambda z: z[0].update(timezone="US/Eastern"), "region IANA zone"),             # legacy link
    (lambda z: z[0].update(timezone="America/Synthville"), "region IANA zone"),     # invalid IANA name
])
def test_city_timezone_map_must_be_exact_and_exhaustive(mutate, needle) -> None:
    data = v6()
    mutate(entry(data, D.FINISHED_AT_TIMEZONE)["resolution"]["city_timezones"])
    assert needle in fails(data)


@pytest.mark.parametrize("change, needle", [
    (dict(earlier="jobs.finished_at", later="cars.scraped_at"), "SCRAPED_FINISHED_ORDERING"),   # reversed
    (dict(earlier="cars.job_finished_at"), "SCRAPED_FINISHED_ORDERING"),
    (dict(equal_allowed="yes"), "boolean"),
])
def test_ordering_shape_is_scrape_not_after_finish(change, needle) -> None:
    data = v6()
    entry(data, D.SCRAPED_FINISHED_ORDERING)["resolution"].update(change)
    assert needle in fails(data)


@pytest.mark.parametrize("change, needle", [(dict(tolerance=-1), "non-negative"), (dict(tolerance=0.5), "non-negative"),
                                            (dict(unit="DAYS"), "unsupported value")])
def test_tolerance_shape_fails_closed(change, needle) -> None:
    data = v6()
    entry(data, D.SCRAPED_FINISHED_TOLERANCE)["resolution"].update(change)
    assert needle in fails(data)


def test_ordering_needs_the_city_map_and_tolerance_needs_the_ordering() -> None:
    data = unapprove(v6(), D.FINISHED_AT_TIMEZONE, D.SCHEDULE_EXPECTED_PERIODS, D.SCHEDULE_SHARING_MODEL)
    assert "FINISHED_AT_TIMEZONE" in fails(data)
    assert "requires SCRAPED_FINISHED_ORDERING" in fails(unapprove(v6(), D.SCRAPED_FINISHED_ORDERING))


# ========================================================== authority sufficiency


def test_current_policy_is_available_with_zero_tolerance() -> None:
    authority = current_temporal_authority()
    assert authority is current_temporal_authority() and authority.record_id == "pricing-authorities-v7"
    assert (authority.timezone_status, authority.ordering_status, authority.tolerance_status) == (
        TS.APPROVED, TS.APPROVED, TS.APPROVED)
    assert dict(authority.city_timezones.entries) == ZONES
    assert authority.city_timezones == current_per_stream_schedule().timezones      # one map, one mechanism
    rule = authority.ordering
    assert (rule.earlier, rule.later, rule.inclusive, rule.tolerance) == (
        (CARS, "scraped_at"), (JOBS, "finished_at"), True, dt.timedelta(0))
    assert authority.tolerance_seconds == 0 and current_temporal_reconciliation().ordering == rule
    assert authority.blocking_reasons == (TB.REPORTING_DAY_UNRESOLVED, TB.DATE_SEMANTICS_UNRESOLVED)
    assert authority.pricing_date_fields == ()
    for ref in ((JOBS, "finished_at"), (CARS, "job_finished_at")):
        field = authority.definition.field(ref)
        assert field.city_timezones == authority.city_timezones and field.timezone_selector == (JOBS, "city")
        assert field.source_timezone is None
    assert authority.definition.field((CARS, "scraped_at")).designator_offsets == {"MST": dt.timedelta(hours=-7)}


def test_timezone_is_authority_sufficient_only_with_the_complete_valid_map() -> None:
    v5 = temporal_authority_from_record(load_decision_record(V5), TEMPORAL_RECONCILIATION)
    assert v5.timezone_status is TS.APPROVED and v5.ordering_status is TS.NOT_APPROVED
    other = synthetic_contract(dataclasses.replace(COV, expected_locations=(("calgary", "Calgary Downtown"),)))
    invalid = temporal_authority_from_record(load_decision_record(V6), TEMPORAL_RECONCILIATION, other)
    assert invalid.timezone_status is TS.INVALID and invalid.city_timezones is None
    assert invalid.ordering is None and TB.FINISHED_AT_TIMEZONE_UNAVAILABLE in invalid.blocking_reasons
    v4 = temporal_authority_from_record(load_decision_record(RECORD_DIR / "v4.toml"), TEMPORAL_RECONCILIATION)
    assert v4.timezone_status is TS.NOT_APPROVED and v4.definition.field((JOBS, "finished_at")).city_timezones is None
    none = temporal_authority_from_record(None)
    assert none.timezone_status is TS.RECORD_UNAVAILABLE and none.ordering is None


def test_ordering_is_available_only_when_ordering_and_tolerance_are_approved() -> None:
    record = parse_decision_record(unapprove(v6(), D.SCRAPED_FINISHED_TOLERANCE))
    authority = temporal_authority_from_record(record, TEMPORAL_RECONCILIATION)
    assert authority.ordering_status is TS.NOT_APPROVED and authority.ordering is None
    assert authority.definition.ordering is None and TB.TIMESTAMP_ORDERING_UNAVAILABLE in authority.blocking_reasons
    record = parse_decision_record(unapprove(v6(), D.SCRAPED_FINISHED_TOLERANCE, D.SCRAPED_FINISHED_ORDERING))
    assert temporal_authority_from_record(record, TEMPORAL_RECONCILIATION).ordering is None


# ==================================================================== localization


def parse(values, cities, context=None):  # type: ignore[no-untyped-def]
    return parse_temporal_field(pd.Series(values, dtype=object), FINISHED, city_values=pd.Series(cities, dtype=object),
                                context_available=context)


@pytest.mark.parametrize("city, local, utc", [
    ("toronto", "2026-01-15 12:00:00.000", "20260115T170000Z"),     # EST, winter
    ("toronto", "2026-07-15 12:00:00.000", "20260715T160000Z"),     # EDT, summer
    ("calgary", "2026-01-15 12:00:00.000", "20260115T190000Z"),     # MST, winter
    ("calgary", "2026-07-15 12:00:00.000", "20260715T180000Z"),     # MDT, summer
    ("vancouver", "2026-01-15 12:00:00.000", "20260115T200000Z"),   # PST, winter
    ("vancouver", "2026-07-15 12:00:00.000", "20260715T190000Z"),   # PDT, summer
])
def test_ordinary_city_local_times_resolve_to_utc(city, local, utc) -> None:
    result = parse([local], [city])
    assert result.resolved.all() and not result.unresolved.any()
    assert canonical_utc_text(result.instants).tolist() == [utc]
    assert str(result.instants.dtype) == "datetime64[ns, UTC]"


def test_the_city_selects_the_zone_not_a_fixed_offset() -> None:
    result = parse(["2026-01-15 12:00:00.000", "2026-07-15 12:00:00.000"] * 3,
                   ["toronto", "toronto", "calgary", "calgary", "vancouver", "vancouver"])
    offsets = [(i.tz_convert(ZONES[c]).utcoffset()) for i, c in zip(result.instants, ["toronto", "toronto",
                                                                                        "calgary", "calgary",
                                                                                        "vancouver", "vancouver"])]
    assert offsets[0] != offsets[1] and offsets[2] != offsets[3] and offsets[4] != offsets[5]   # DST applies
    assert len({str(o) for o in offsets[::2]}) == 3                                            # no global zone


@pytest.mark.parametrize("city", [None, "", "   ", "Toronto", "toronto ", " toronto", "TORONTO", "montreal", 1])
def test_unapproved_cities_fail_closed(city) -> None:
    result = parse(["2026-01-15 12:00:00.000"], [city])
    assert result.unknown_city.tolist() == [True] and result.instants.isna().all() and result.unresolved.all()


def test_ambiguous_and_nonexistent_local_times_fail_closed() -> None:
    result = parse(["2026-11-01 01:30:00.000", "2026-03-08 02:30:00.000", "2026-03-08 02:30:00.000",
                    "2026-11-01 00:30:00.000"], ["toronto", "toronto", "vancouver", "toronto"])
    assert result.ambiguous.tolist() == [True, False, False, False]
    assert result.nonexistent.tolist() == [False, True, True, False]
    assert result.instants.isna().tolist() == [True, True, True, False]           # nothing chosen or shifted


def test_no_machine_local_fallback(monkeypatch) -> None:
    baseline = canonical_utc_text(parse(["2026-01-15 12:00:00.000"], ["calgary"]).instants).tolist()
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        assert canonical_utc_text(parse(["2026-01-15 12:00:00.000"], ["calgary"]).instants).tolist() == baseline
        assert parse(["2026-01-15 12:00:00.000"], [None]).instants.isna().all()
        no_context = parse_temporal_field(pd.Series(["2026-01-15 12:00:00.000"]), FINISHED)
        assert no_context.context_unavailable.all() and no_context.instants.isna().all()
    finally:
        monkeypatch.delenv("TZ")
        if hasattr(time, "tzset"):
            time.tzset()


def test_full_precision_is_kept_and_canonical_text_is_presentation_only() -> None:
    result = parse(["2026-01-15 12:00:00.123456", "2026-01-15 12:00:00.100", "2026-01-15 12:00:00.900"],
                   ["toronto"] * 3)
    first, low, high = result.instants.tolist()
    assert first.microsecond == 123456
    texts = canonical_utc_text(result.instants).tolist()
    assert texts == ["20260115T170000Z"] * 3                                       # same serialized second
    assert low != high and high - low == pd.Timedelta(milliseconds=800)           # distinct instants internally
    with pytest.raises(TypeError):
        canonical_utc_text(pd.Series(["20260115T170000Z"]))


def test_city_map_fields_are_validated() -> None:
    zones = CityTimezoneMap(tuple(ZONES.items()))
    with pytest.raises(TemporalConfigurationError):
        dataclasses.replace(FINISHED, city_timezones=None)                         # selector without map
    with pytest.raises(TemporalConfigurationError):
        dataclasses.replace(FINISHED, source_timezone="America/Toronto")          # two zone bases
    with pytest.raises(TemporalConfigurationError):
        dataclasses.replace(DEF.field((CARS, "scraped_at")), city_timezones=zones,
                            timezone_selector=(JOBS, "city"))                       # designator field
    with pytest.raises(TemporalConfigurationError):
        dataclasses.replace(DEF, fields=tuple(dataclasses.replace(f, timezone_selector=(CARS, "city"))
                                              if f.city_timezones is not None else f for f in DEF.fields))
    assert isinstance(FINISHED, TemporalFieldDefinition)


# ============================================================ frames and reconciliation

FIN = "2026-07-15 12:00:00.000"             # synthetic local finish time


def frames(rows):  # type: ignore[no-untyped-def]
    """rows: (job, parent city, finished_at, detail city, job_finished_at, scraped_at) - one detail row each.

    A job value of ``None`` on the detail side is written as a missing key.
    """
    jobs, cars, seen = [], [], set()
    for job, city, fin, dcity, jfin, scraped in rows:
        if job not in seen and not str(job).startswith("ORPHAN"):
            seen.add(job)
            jobs.append({"job_id": job, "city": city, "finished_at": fin, "scrape_date": "2026-07-15",
                         "record_count": "1", "actual_car_rows": "1"})
        cars.append({"job_id": None if str(job).startswith("ORPHAN-NOKEY") else job, "row_index": str(len(cars)),
                     "city": dcity, "job_finished_at": jfin, "scraped_at": scraped, "scrape_date": "2026-07-15",
                     "date_clean": "2026-07-15"})
    return frame(JOBS, jobs), frame(CARS, cars)


def row(job="SYNTH-JOB-1", city="toronto", fin=FIN, dcity=None, jfin=None, scraped="2026-07-15 08:00:00 MST"):  # type: ignore[no-untyped-def]
    return (job, city, fin, city if dcity is None else dcity, fin if jfin is None else jfin, scraped)


def reconcile(*rows):  # type: ignore[no-untyped-def]
    j, c = frames(rows)
    return assess_temporal_reconciliation(j, c, DEF)


def fields_of(report) -> dict:  # type: ignore[no-untyped-def]
    return {f"{f.dataset.value}.{f.column}": f for f in report.field_reports}


def rep(report):  # type: ignore[no-untyped-def]
    (r,) = report.replications
    return (r.passed, r.failed, r.unassessable)


def order(report):  # type: ignore[no-untyped-def]
    return (report.ordering.passed, report.ordering.failed, report.ordering.unassessable)


# ===================================================================== replication


def test_matching_parent_and_detail_finish_times_replicate() -> None:
    report = reconcile(row(), row(job="SYNTH-JOB-2", city="calgary"), row(job="SYNTH-JOB-3", city="vancouver"))
    assert rep(report) == (3, 0, 0) and report.replication_failed_count == 0
    assert fields_of(report)["cars.job_finished_at"].resolved_count == 3
    assert report.city_mismatch_detail_row_count == 0


@pytest.mark.parametrize("copy_value", ["2026-07-15 12:00:01.000",          # different wall time
                                        "2026-07-15 12:00:00.900"])         # same canonical second, other instant
def test_mismatched_copies_fail_replication(copy_value) -> None:
    report = reconcile(row(fin="2026-07-15 12:00:00.100", jfin=copy_value))
    assert rep(report) == (0, 1, 0) and "replication" in report.violations and not report.is_valid
    assert order(report) == (0, 0, 1)                                     # ordering is not assessed on it


def test_a_detail_city_mismatch_breaks_trust() -> None:
    report = reconcile(row(dcity="calgary"), row(job="SYNTH-JOB-2"))
    assert report.city_mismatch_detail_row_count == 1 and "city_mismatch" in report.violations
    assert rep(report) == (1, 0, 1) and order(report) == (1, 0, 1)
    # The copy is still resolved through the parent's city, never the detail row's own city.
    j, c = frames([row(city="vancouver", dcity="toronto")])
    derived = derive_utc_timestamps(j, c, DEF)
    assert derived.cars["job_finished_at_canonical"].tolist() == ["20260715T190000Z"]


@pytest.mark.parametrize("which", ["parent", "copy"])
def test_missing_finish_times_are_counted_and_unassessable(which) -> None:
    report = reconcile(row(fin="" if which == "parent" else FIN, jfin="" if which == "copy" else None))
    field = "jobs.finished_at" if which == "parent" else "cars.job_finished_at"
    assert fields_of(report)[field].missing_count == 1 and rep(report) == (0, 0, 1)
    assert not report.is_valid and "field_parse" in report.violations


@pytest.mark.parametrize("orphan", ["ORPHAN-SYNTH-JOB", "ORPHAN-NOKEY"])
def test_orphan_and_keyless_details_are_unlinked_not_dropped(orphan) -> None:
    report = reconcile(row(), row(job=orphan))
    assert report.unlinked_detail_row_count == 1 and rep(report) == (1, 0, 1) and order(report) == (1, 0, 1)
    assert fields_of(report)["cars.job_finished_at"].context_unavailable_count == 1
    assert report.detail_row_count == 2 and not report.is_valid


# ======================================================================== ordering


@pytest.mark.parametrize("scraped, passed", [
    ("2026-07-15 08:00:00 MST", True),                # 15:00Z before the 16:00Z finish (EDT)
    ("2026-07-15 09:00:00 MST", True),                # exactly equal: allowed
    ("2026-07-15 09:00:01 MST", False),               # one second after: fails (no tolerance)
])
def test_scrape_must_not_be_after_finish(scraped, passed) -> None:
    report = reconcile(row(scraped=scraped))
    assert order(report) == ((1, 0, 0) if passed else (0, 1, 0))
    assert ("ordering" in report.violations) is (not passed)


def test_one_microsecond_after_the_finish_fails() -> None:
    # The finish carries a fraction; the scrape (whole seconds) equals it only without the fraction.
    report = reconcile(row(fin="2026-07-15 11:59:59.999999", scraped="2026-07-15 09:00:00 MST"))
    assert order(report) == (0, 1, 0)                                       # 1 microsecond late
    report = reconcile(row(fin="2026-07-15 12:00:00.000001", scraped="2026-07-15 09:00:00 MST"))
    assert order(report) == (1, 0, 0)
    assert DEF.ordering.tolerance == dt.timedelta(0) and DEF.ordering.inclusive


def test_comparison_uses_utc_instants_not_wall_clocks() -> None:
    # Vancouver winter finish 10:00 PST = 18:00Z; scrape "10:30 MST" = 17:30Z is earlier although its
    # wall clock reads later; a Calgary summer finish 12:00 MDT = 18:00Z with a scrape "11:30 MST" = 18:30Z
    # is later although its wall clock reads earlier.
    report = reconcile(row(city="vancouver", fin="2026-01-15 10:00:00.000", scraped="2026-01-15 10:30:00 MST"),
                       row(job="SYNTH-JOB-2", city="calgary", fin="2026-07-15 12:00:00.000",
                           scraped="2026-07-15 11:30:00 MST"))
    assert order(report) == (1, 1, 0)


@pytest.mark.parametrize("scraped, category", [("", "missing"), ("SYNTH", "invalid"),
                                               ("2026-07-15 08:00:00 XYZ", "unresolved")])
def test_missing_or_invalid_scrape_times_are_unassessable(scraped, category) -> None:
    report = reconcile(row(scraped=scraped), row(job="SYNTH-JOB-2"))
    field = fields_of(report)["cars.scraped_at"]
    assert {"missing": field.missing_count, "invalid": field.invalid_count,
            "unresolved": field.unresolved_count}[category] == 1
    assert order(report) == (1, 0, 1) and not report.is_valid


@pytest.mark.parametrize("fin, attribute", [("2026-11-01 01:30:00.000", "ambiguous_count"),
                                            ("2026-03-08 02:30:00.000", "nonexistent_count")])
def test_unresolved_finish_times_block_ordering_and_trust(fin, attribute) -> None:
    report = reconcile(row(fin=fin, scraped="2026-01-01 00:00:00 MST"), row(job="SYNTH-JOB-2"))
    assert getattr(fields_of(report)["jobs.finished_at"], attribute) == 1
    assert getattr(fields_of(report)["cars.job_finished_at"], attribute) == 1
    assert order(report) == (1, 0, 1) and "time_unresolved" in report.violations and not report.is_valid
    assert report.unresolved_time_count == 2


def test_an_unknown_parent_city_blocks_its_rows_only() -> None:
    report = reconcile(row(city="Toronto"), row(job="SYNTH-JOB-2"), row(job="SYNTH-JOB-3", city="calgary"))
    assert fields_of(report)["jobs.finished_at"].unknown_city_count == 1
    assert fields_of(report)["cars.job_finished_at"].unknown_city_count == 1
    assert order(report) == (2, 0, 1) and rep(report) == (2, 0, 1)
    assert report.ordering.row_count == report.detail_row_count == 3        # nothing dropped


def test_unrelated_rows_are_neither_dropped_nor_affected() -> None:
    report = reconcile(row(), row(job="SYNTH-JOB-2", scraped="2026-07-15 09:30:00 MST"),
                       row(job="SYNTH-JOB-3", jfin="2026-07-15 12:00:05.000"), row(job="SYNTH-JOB-4", city="vancouver"))
    assert order(report) == (2, 1, 1) and rep(report) == (3, 1, 0)
    assert report.ordering.row_count == 4 and report.parent_row_count == 4


def test_a_clean_dataset_still_waits_for_the_reporting_day() -> None:
    report = reconcile(row(), row(job="SYNTH-JOB-2", city="calgary"))
    assert rep(report) == (2, 0, 0) and order(report) == (2, 0, 0)
    assert report.violations == ("rule_unavailable",) and not report.is_valid
    assert set(report.unavailable_rules) == {"date_derivation:jobs.scrape_date", "date_derivation:cars.scrape_date",
                                             "date_derivation:cars.date_clean"}


# ======================================================================= readiness


def pricing(temporal_ok: bool):  # type: ignore[no-untyped-def]
    from test_readiness import DISTINCT

    return assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                    **(GATES | {"temporal_fields_trusted": temporal_ok}))


@pytest.mark.parametrize("rows", [
    [row(scraped="2026-07-15 09:00:01 MST")],                              # ordering violation
    [row(fin="2026-11-01 01:30:00.000", scraped="2026-01-01 00:00:00 MST")],  # ambiguous
    [row(fin="2026-03-08 02:30:00.000", scraped="2026-01-01 00:00:00 MST")],  # nonexistent
    [row(jfin="2026-07-15 12:00:00.500")],                                  # parent/detail mismatch
    [row(dcity="calgary")],                                                 # city mismatch
    [row()],                                                                # clean, but reporting day unresolved
])
def test_temporal_trust_and_pricing_stay_blocked(rows) -> None:
    report = reconcile(*rows)
    assert not report.is_valid
    readiness = pricing(bool(report.is_valid))
    assert PB.TEMPORAL_FIELDS_UNTRUSTED in readiness.blocking_reasons and not readiness.ready


def test_reporting_day_and_date_semantics_remain_blockers() -> None:
    authority = current_temporal_authority()
    assert (authority.reporting_day_status, authority.date_semantics_status) == (TS.NOT_APPROVED, TS.NOT_APPROVED)
    assert {TB.REPORTING_DAY_UNRESOLVED, TB.DATE_SEMANTICS_UNRESOLVED} <= set(authority.blocking_reasons)
    assert all(check.rule is None for check in authority.definition.date_checks)


# ====================================================================== date_clean


def test_date_clean_is_preserved_and_never_a_pricing_date() -> None:
    assert "date_clean" in DATASET_DEFINITIONS[CARS].columns               # still ingested raw
    j, c = frames([row()])
    before = (j.copy(deep=True), c.copy(deep=True))
    report = assess_temporal_reconciliation(j, c, DEF)
    derived = derive_utc_timestamps(j, c, DEF)
    pd.testing.assert_frame_equal(j, before[0]), pd.testing.assert_frame_equal(c, before[1])
    assert "date_clean" not in set(derived.cars.columns) | set(derived.jobs.columns)
    assert fields_of(report)["cars.date_clean"].valid_count == 1           # parsed for traceability only
    # Observed equality with scrape_date (identical above) creates no authority and no trust.
    assert "date_derivation:cars.date_clean" in report.unavailable_rules
    assert current_temporal_authority().pricing_date_fields == ()


def test_an_approved_reporting_day_is_never_overridden_by_date_clean() -> None:
    data = v6()
    authority_entry = [{"kind": "BUSINESS_OWNER", "source": "Business owner: SYNTH test decision",
                        "reference": GOVERNANCE}]
    for decision, resolution in ((D.REPORTING_DAY_SOURCE, {"field": "jobs.scrape_date"}),
                                 (D.REPORTING_DAY_TIMEZONE, {"timezone": "America/Toronto"})):
        entry(data, decision).update(status="APPROVED", blocking_external_input=False,
                                     authority=copy.deepcopy(authority_entry), resolution=resolution)
    authority = temporal_authority_from_record(parse_decision_record(finish(data)), TEMPORAL_RECONCILIATION)
    assert authority.reporting_day_status is TS.APPROVED
    assert authority.pricing_date_fields == ("jobs.scrape_date",) and "cars.date_clean" not in authority.pricing_date_fields
    assert TB.REPORTING_DAY_UNRESOLVED not in authority.blocking_reasons
    assert TB.DATE_SEMANTICS_UNRESOLVED in authority.blocking_reasons     # date_clean semantics still unresolved


# ===================================================================== derived fields


def test_derived_utc_fields_are_separate_and_full_precision() -> None:
    j, c = frames([row(fin="2026-07-15 12:00:00.250", scraped="2026-07-15 08:00:00 MST")])
    derived = derive_utc_timestamps(j, c, DEF)
    assert list(derived.jobs.columns) == ["finished_at_utc", "finished_at_canonical"]
    assert list(derived.cars.columns) == ["job_finished_at_utc", "job_finished_at_canonical", "scraped_at_utc",
                                          "scraped_at_canonical"]
    assert derived.jobs["finished_at_utc"].iloc[0] == pd.Timestamp("2026-07-15T16:00:00.250", tz="UTC")
    assert derived.jobs["finished_at_canonical"].iloc[0] == "20260715T160000Z"
    assert derived.cars["scraped_at_canonical"].iloc[0] == "20260715T150000Z"
    assert j["finished_at"].iloc[0] == "2026-07-15 12:00:00.250"              # raw value unchanged


# ===================================================================== baseline and docs


def test_baseline_reports_the_temporal_policy_in_aggregate() -> None:
    import json

    from test_collection_schedule import pricing_with, real_frames, real_report

    from ql2_sixt_canada_analysis.pricing_baseline import build_pricing_baseline, render_baseline_markdown
    from ql2_sixt_canada_analysis.stability import assess_vehicle_attribute_stability

    j, c = real_frames()
    _, readiness = pricing_with(j, c, real_report(j, c))
    temporal = assess_temporal_reconciliation(j, c, DEF)
    stable = assess_vehicle_attribute_stability(c.assign(car_name="SYNTH Vehicle"))
    baseline = build_pricing_baseline(pricing=readiness, jobs=j, cars=c, temporal=temporal, vehicle_stability=stable,
                                      temporal_contract=DEF, temporal_authority=AUTHORITY)
    summary = baseline.temporal
    assert (summary.timezone_status, summary.ordering_status, summary.tolerance_status, summary.tolerance_seconds) == (
        "approved", "approved", "approved", 0)
    assert summary.authority_blockers == ("reporting_day_unresolved", "date_semantics_unresolved")
    assert summary.pricing_date_fields == () and summary.temporal_fields_trusted is False
    fields_ = {f.field: f for f in summary.fields}
    assert fields_["jobs.finished_at"].resolved == 270 and fields_["cars.job_finished_at"].resolved == len(c)
    assert summary.replication_passed == len(c) and summary.ordering_rule_status == "configured"
    markdown = render_baseline_markdown(baseline, commit="abc1234", date="2026-10-06")
    assert "Temporal policy" in markdown and "(0 seconds)" in markdown and "**NOT PRICING READY**" in markdown
    assert not re.search(r"\d{8}T\d{6}Z|\d{4}-\d{2}-\d{2} \d{2}:|SYNTH", markdown)
    json.dumps(baseline.to_dict())


def test_governance_document_records_the_supplied_decisions_only() -> None:
    raw = (ROOT / GOVERNANCE).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in ("2026-10-06", "collection owner", "business owner", "jobs.city", "parent job city",
                   "repeated copies", "UTC", "YYYYMMDDTHHMMSSZ", "Raw values are preserved", "Full precision",
                   "Ambiguous local times", "Nonexistent local times", "earlier than or equal to",
                   "Equality is allowed", "zero seconds", "Parent/detail replication", "City integrity is a prerequisite",
                   "current dataset and subsequent collections until superseded", "new authority-record version",
                   "fixed UTC-07:00", "Not supplied", "pricing-authorities-v6", "remain `PROPOSED`"):
        assert phrase in text, phrase
    for city, zone in ZONES.items():
        assert f"| `{city}` | `{zone}` |" in raw
    assert "@" not in raw and not re.search(r"#\d|\b(?!UTC-)[A-Z]{2,}-\d+", raw)
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:|\d{6,}|\d+\.\d{2}\b", raw) and "SYNTH" not in raw


def test_documentation_describes_the_temporal_policy() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    records = " ".join((RECORD_DIR / "README.md").read_text(encoding="utf-8").split())
    for phrase in ("America/Edmonton", "America/Toronto", "America/Vancouver", "parent job", "YYYYMMDDTHHMMSSZ",
                   "full precision", "cars.job_finished_at", "earlier than or equal to", "zero", "ambiguous",
                   "nonexistent", "date_clean", "temporal_fields_untrusted", Path(GOVERNANCE).name):
        assert phrase in readme, phrase
    assert "No authoritative time zone is documented" not in readme
    for phrase in ("v6.toml", "SCRAPED_FINISHED_ORDERING", "SCRAPED_FINISHED_TOLERANCE", "zero seconds"):
        assert phrase in records, phrase
    assert "The dataset is **not** pricing ready" in readme
