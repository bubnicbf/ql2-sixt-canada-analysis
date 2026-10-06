"""Authority-backed per-stream hourly collection schedule (pricing-authorities-v5, schema 3).

The committed records are read as data; negative cases mutate parsed copies
in memory. Observed-assignment tests use fabricated frames (``SYNTH-JOB-*``)
and synthetic cities and branches; the only real values are the approved
configuration (stream keys, zones and window) read from the record.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from stream_contract_fixtures import synthetic_contract
from test_completeness import CARS, JOBS, frame
from test_expected_stream_contract import DISPLAY_SPELLINGS, SUPPLIED, completeness_of
from test_readiness import gates_for

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
from ql2_sixt_canada_analysis.collection_schedule import (
    CityTimezoneMap,
    ExceptionsModel,
    JobAssignmentFailure as JF,
    PerStreamSchedule,
    PerStreamScheduledCoverageReport,
    ScheduleAuthorityStatus as SS,
    ScheduleConfigurationError,
    ScheduleCoverageBlocker as SB,
    ScheduleExceptions,
    ScheduleFailureKind as FK,
    SharingMode,
    StreamSchedule,
    StreamScheduleException,
    assess_per_stream_scheduled_coverage,
    current_per_stream_schedule,
    format_utc_instant,
    materialize_local_hours,
    schedule_from_record,
)
from ql2_sixt_canada_analysis.expected_stream_contract import (
    current_expected_stream_contract,
    expected_stream_contract_from_record,
)
from ql2_sixt_canada_analysis.location_authority import current_location_authority
from ql2_sixt_canada_analysis.readiness import (
    CompletenessBlocker as CMP,
    PricingBlocker as PB,
    apply_location_policy,
    assess_location_policy,
    assess_pricing_readiness,
)
from ql2_sixt_canada_analysis.schemas import (
    EXPECTED_LOCATION_COVERAGE as COV,
    JOB_DETAIL_RELATIONSHIP as REL,
    VANCOUVER_LOCATION_POLICY,
    LocationCoverageMode,
    TemporalConfigurationError,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V4, V5 = RECORD_DIR / "v4.toml", RECORD_DIR / "v5.toml"
GOVERNANCE = "docs/decisions/governance/collection-schedule-governance-v1-2026-10-06.md"
D = DecisionId
SCHEDULE_DECISIONS = (D.SCHEDULE_CAPTURE_TIMESTAMP, D.SCHEDULE_EXPECTED_PERIODS, D.SCHEDULE_SHARING_MODEL,
                      D.SCHEDULE_EXCEPTIONS, D.FINISHED_AT_TIMEZONE)
#: The decisions exactly as supplied (test oracle).
ZONES = {"calgary": "America/Edmonton", "toronto": "America/Toronto", "vancouver": "America/Vancouver"}
LOCAL_START, LOCAL_END = dt.datetime(2026, 8, 27, 22), dt.datetime(2026, 8, 31, 15)
CALGARY_QUESTION = ("For the one Calgary hourly job that produced Calgary Airport rows but no Calgary Downtown rows, "
                    "was the Downtown branch attempted, and what was the outcome?")
CAL_DOWN, CAL_AIR = SUPPLIED[0], SUPPLIED[1]
FINISH_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def v5() -> dict:
    return tomllib.loads(V5.read_text(encoding="utf-8"))


def entry(data: dict, decision: DecisionId) -> dict:
    return next(e for e in data["decisions"] if e["id"] == decision.value)


def resolution(data: dict, decision: DecisionId) -> dict:
    return entry(data, decision)["resolution"]


def finish(data: dict) -> dict:
    statuses = [e["status"] for e in data["decisions"]]
    data["summary"] = {s.value.lower(): statuses.count(s.value) for s in DecisionStatus}
    data["external_inputs"] = [e["id"] for e in data["decisions"] if e["blocking_external_input"]]
    return data


def unapprove(data: dict, decision: DecisionId) -> dict:
    item = entry(data, decision)
    item.update(status="PROPOSED", blocking_external_input=True)
    item.pop("authority", None), item.pop("resolution", None)
    return finish(data)


def fails(data: dict) -> str:
    with pytest.raises(DecisionRecordError) as info:
        parse_decision_record(finish(data))
    return str(info.value)


# ================================================================= authority (v5)


def test_v5_is_a_schema_3_record_superseding_v4_and_carried_into_the_current_record() -> None:
    record = load_decision_record(V5)
    assert (record.schema_version, record.record_version, record.record_id, record.supersedes) == (
        3, 5, "pricing-authorities-v5", "pricing-authorities-v4")
    assert CURRENT_RECORD_PATH.as_posix() == "docs/decisions/pricing_authorities/v6.toml"
    current = load_current_decision_record()
    assert current.record_id == "pricing-authorities-v6" and current.supersedes == "pricing-authorities-v5"
    for decision in SCHEDULE_DECISIONS:                   # v6 keeps every schedule decision unchanged
        assert current.decision(decision).resolution == record.decision(decision).resolution
        assert set(record.decision(decision).authority) <= set(current.decision(decision).authority)
    counts = record.counts()
    assert (counts[DecisionStatus.APPROVED], counts[DecisionStatus.PROPOSED], counts[DecisionStatus.REJECTED]) == (
        14, 8, 0)
    v4 = load_decision_record(V4)
    newly = {d.id for d in record.decisions if d.is_approved} - {d.id for d in v4.decisions if d.is_approved}
    assert newly == set(SCHEDULE_DECISIONS)
    for decision in record.decisions:                     # every other temporal decision stays PROPOSED
        if not decision.is_approved:
            assert decision.status is DecisionStatus.PROPOSED and decision.blocking_external_input
            assert decision.id not in SCHEDULE_DECISIONS
    for decision in (D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE,
                     D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS, D.JOB_ID_RAW_AND_LINKAGE_PRESERVATION):
        assert record.decision(decision).resolution == v4.decision(decision).resolution     # preserved
        assert record.decision(decision).authority == v4.decision(decision).authority


def test_v5_resolutions_are_exactly_the_supplied_schedule_decisions() -> None:
    record = load_decision_record(V5)
    assert dict(record.approved_resolution(D.SCHEDULE_CAPTURE_TIMESTAMP)) == {
        "field": "jobs.finished_at", "detail_copy": "cars.job_finished_at", "detail_observation": "cars.scraped_at"}
    zones = record.approved_resolution(D.FINISHED_AT_TIMEZONE)["city_timezones"]
    assert {z["city"]: z["timezone"] for z in zones} == ZONES and len(zones) == 3
    periods = record.approved_resolution(D.SCHEDULE_EXPECTED_PERIODS)
    assert (periods["cadence"], periods["phase"], periods["parent_jobs_per_city_period"]) == (
        "PT1H", "LOCAL_TOP_OF_HOUR", 1)
    streams = {tuple(s["stream"]): s for s in periods["streams"]}
    assert tuple(streams) == SUPPLIED                      # exactly one schedule per approved stream
    for item in streams.values():
        assert (item["local_start"], item["local_end"], item["end_inclusive"]) == (
            "2026-08-27T22:00:00", "2026-08-31T15:00:00", True)
    assert dict(record.approved_resolution(D.SCHEDULE_SHARING_MODEL)) == {"mode": "PER_STREAM"}
    assert dict(record.approved_resolution(D.SCHEDULE_EXCEPTIONS)) == {"model": "NO_EXCEPTIONS"}


def test_v5_authority_roles_and_governance_reference() -> None:
    record = load_decision_record(V5)
    for decision in SCHEDULE_DECISIONS:
        kinds = [a.kind for a in record.decision(decision).authority]
        expected = ([AuthorityKind.COLLECTION_OWNER, AuthorityKind.BUSINESS_OWNER]
                    if decision is D.SCHEDULE_EXCEPTIONS else [AuthorityKind.COLLECTION_OWNER])
        assert kinds == expected, decision
        assert {a.reference for a in record.decision(decision).authority} == {GOVERNANCE}
        assert all(e.kind.value != "BEHAVIORAL_ANALYSIS" for e in record.decision(decision).evidence)
    assert record.decision(D.SCHEDULE_EXCEPTIONS).joint is True
    assert (ROOT / GOVERNANCE).is_file()


def test_v5_corrects_the_source_keys_and_keeps_roles_pairs_and_alias() -> None:
    record = load_decision_record(V5)
    for decision in (D.EXPECTED_STREAM_UNIVERSE, D.EXPECTED_STREAM_SOURCE_SPELLING):
        assert tuple(map(tuple, record.approved_resolution(decision)["streams"])) == SUPPLIED
    assert tuple(map(tuple, load_decision_record(V4).approved_resolution(D.EXPECTED_STREAM_UNIVERSE)["streams"])) == \
        DISPLAY_SPELLINGS                                   # history unchanged
    roles = {tuple(a["stream"]): a["role"] for a in record.approved_resolution(D.LOCATION_ROLE_ASSIGNMENTS)["assignments"]}
    assert roles == {k: ("AIRPORT" if "Int Airport" in k[1] else "DOWNTOWN") for k in SUPPLIED}
    pairs = [(tuple(p["airport"]), tuple(p["downtown"]))
             for p in record.approved_resolution(D.VALID_LOCATION_COMPARISON_PAIRS)["pairs"]]
    assert pairs == [(SUPPLIED[1], SUPPLIED[0]), (SUPPLIED[3], SUPPLIED[2]), (SUPPLIED[5], SUPPLIED[4])]
    identity = record.approved_resolution(D.VANCOUVER_LOCATION_IDENTITY)
    assert identity["state"] == "CONFIRMED_ALIAS" and tuple(identity["canonical_location"]) == SUPPLIED[4]
    assert tuple(map(tuple, identity["governed_locations"])) == (SUPPLIED[4], SUPPLIED[6])


def test_governance_document_records_the_supplied_decisions_only() -> None:
    raw = (ROOT / GOVERNANCE).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in ("2026-10-06", "collection owner", "business owner", "jobs.finished_at", "cars.job_finished_at",
                   "cars.scraped_at", "observation time only", "PT1H", "start of every local clock hour",
                   "one parent job per city per local hour", "PER_STREAM", "`2026-08-27T22:00:00`",
                   "`2026-08-31T15:00:00`", "end boundary inclusive", "YYYYMMDDTHHMMSSZ", "never simply appended",
                   "nonexistent", "skipped", "never deduplicated", "NO_EXCEPTIONS", "remains a real blocker",
                   "supersede", "current dataset and subsequent schedule versions until superseded",
                   "requires a new versioned schedule decision", "Not supplied", "pricing-authorities-v5",
                   "remain `PROPOSED`", "does not make the dataset pricing ready"):
        assert phrase in text, phrase
    assert CALGARY_QUESTION in text
    for city, zone in ZONES.items():
        assert f"| `{city}` | `{zone}` |" in raw
    for city, location in SUPPLIED:
        assert f"| `{city}` | `{location}` |" in raw
    assert "@" not in raw and not re.search(r"#\d|[A-Z]{2,}-\d+", raw)             # no emails or tickets
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}|\d{6,}|\d+\.\d{2}\b", raw)  # no raw timestamps, ids, prices
    assert "job_id" not in raw and "SYNTH" not in raw


def test_only_the_v5_record_backs_the_schedule() -> None:
    assert schedule_from_record(load_decision_record(V4), current_expected_stream_contract()).status is SS.NOT_APPROVED
    assert schedule_from_record(None, current_expected_stream_contract()).status is SS.RECORD_UNAVAILABLE
    dependants = {D.FINISHED_AT_TIMEZONE: (D.SCHEDULE_EXPECTED_PERIODS, D.SCHEDULE_SHARING_MODEL),
                  D.SCHEDULE_EXPECTED_PERIODS: (D.SCHEDULE_SHARING_MODEL,)}
    for decision in SCHEDULE_DECISIONS:
        data = unapprove(v5(), decision)
        for dependant in dependants.get(decision, ()):          # a dependant cannot stay approved without it
            data = unapprove(data, dependant)
        record = parse_decision_record(data)
        schedule = schedule_from_record(record, current_expected_stream_contract())
        assert schedule.status is SS.NOT_APPROVED and not schedule.available, decision
        assert schedule.blocking_reasons == (SB.COLLECTION_SCHEDULE_UNAVAILABLE,)


@pytest.mark.parametrize("field", ["cars.scraped_at", "cars.job_finished_at", "jobs.scrape_date", "jobs.synth"])
def test_capture_anchor_must_be_the_parent_finish_time(field: str) -> None:
    data = v5()
    resolution(data, D.SCHEDULE_CAPTURE_TIMESTAMP)["field"] = field
    assert "SCHEDULE_CAPTURE_TIMESTAMP" in fails(data)
    data = v5()
    del resolution(data, D.SCHEDULE_CAPTURE_TIMESTAMP)["detail_copy"]
    assert "required fields" in fails(data)


@pytest.mark.parametrize("mutate, needle", [
    (lambda z: z.pop(), "exactly the approved cities"),                                     # missing city
    (lambda z: z.append({"city": "synth", "timezone": "America/Regina"}), "exactly the approved cities"),
    (lambda z: z[0].update(city="Calgary"), "exactly the approved cities"),                 # misspelled city
    (lambda z: z[0].update(city=""), "non-blank"),
    (lambda z: z[0].update(city=" calgary"), "non-blank"),
    (lambda z: z[0].update(timezone="UTC"), "region IANA zone"),
    (lambda z: z[0].update(timezone="Etc/GMT+7"), "region IANA zone"),
    (lambda z: z[0].update(timezone="MST"), "region IANA zone"),
    (lambda z: z[0].update(timezone="-07:00"), "region IANA zone"),
    (lambda z: z[0].update(timezone="America/Synthville"), "region IANA zone"),
    (lambda z: z.append(dict(z[0])), "duplicate city"),
])
def test_city_timezone_map_is_exhaustive_exact_and_region_based(mutate, needle) -> None:
    data = v5()
    mutate(resolution(data, D.FINISHED_AT_TIMEZONE)["city_timezones"])
    assert needle in fails(data)


def _first_stream(data: dict) -> dict:
    return resolution(data, D.SCHEDULE_EXPECTED_PERIODS)["streams"][0]


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r["streams"].pop(), "exactly one schedule"),                                # missing stream
    (lambda r: r["streams"].append(copy.deepcopy(r["streams"][0])), "exactly one schedule"),   # duplicate
    (lambda r: r["streams"][0].update(stream=["calgary", "Calgary Synth"]), "outside the approved universe"),
    (lambda r: r["streams"][0].update(stream=["Calgary", "Downtown"]), "outside the approved universe"),
    (lambda r: r["streams"][0].update(local_start="2026-08-27T22:00:00-06:00"), "naive local"),
    (lambda r: r["streams"][0].update(local_end="2026-08-31T15:00:00Z"), "naive local"),
    (lambda r: r["streams"][0].update(local_start="2026-08-27 22:00:00"), "naive local"),
    (lambda r: r["streams"][0].update(local_start="2026-08-27T22:30:00"), "top of a local hour"),
    (lambda r: r["streams"][0].update(local_end="2026-08-27T21:00:00"), "precedes"),
    (lambda r: r["streams"][0].update(end_inclusive="yes"), "boolean"),
    (lambda r: r["streams"][0].pop("end_inclusive"), "required fields"),
    (lambda r: r.update(cadence="PT2H"), "PT1H"),
    (lambda r: r.update(cadence="1h"), "PT1H"),
    (lambda r: r.update(phase="FIRST_OBSERVED_JOB"), "unsupported value"),
    (lambda r: r.update(parent_jobs_per_city_period=2), "one parent job"),
    (lambda r: r.update(schedule_version="Version 1"), "short code"),
    (lambda r: r.update(period_starts=["2026-08-28T04:00:00Z"]), "required fields"),       # no global instant list
])
def test_expected_periods_are_explicit_per_stream_definitions(mutate, needle) -> None:
    data = v5()
    mutate(resolution(data, D.SCHEDULE_EXPECTED_PERIODS))
    assert needle in fails(data)


def test_schema_2_shapes_are_refused_in_a_schema_3_record() -> None:
    data = v5()
    entry(data, D.SCHEDULE_EXPECTED_PERIODS)["resolution"] = {
        "period": "PT1H", "period_starts": ["2026-08-28T04:00:00+00:00"]}
    assert "required fields" in fails(data)
    data = v5()
    entry(data, D.FINISHED_AT_TIMEZONE)["resolution"] = {"timezone": "America/Edmonton"}
    assert "FINISHED_AT_TIMEZONE" in fails(data)


def test_per_stream_sharing_and_dependencies_fail_closed() -> None:
    data = v5()
    resolution(data, D.SCHEDULE_SHARING_MODEL)["mode"] = "GLOBAL"
    assert "unsupported value" in fails(data)
    data = unapprove(unapprove(v5(), D.SCHEDULE_SHARING_MODEL), D.SCHEDULE_EXPECTED_PERIODS)
    entry(data, D.SCHEDULE_SHARING_MODEL).update(
        status="APPROVED", blocking_external_input=False, resolution={"mode": "PER_STREAM"},
        authority=copy.deepcopy(entry(v5(), D.SCHEDULE_SHARING_MODEL)["authority"]))
    assert "requires SCHEDULE_EXPECTED_PERIODS" in fails(data)
    data = unapprove(v5(), D.FINISHED_AT_TIMEZONE)                       # periods need the zone map
    assert "FINISHED_AT_TIMEZONE" in fails(data)


def _listed(**overrides) -> dict:
    item = {"stream": list(CAL_DOWN), "period_start_utc": "20260828T040000Z", "failure": "STREAM_ABSENT_FROM_CAPTURE",
            "reason": "SYNTH documented operational outage", "authority_kind": "COLLECTION_OWNER",
            "reference": GOVERNANCE, "schedule_version": "per_stream_hourly_v1"}
    return {"model": "LISTED_EXCEPTIONS", "exceptions": [item | overrides]}


@pytest.mark.parametrize("value, needle", [
    ({"model": "NO_EXCEPTIONS", "exceptions": []}, "required fields"),     # explicit, never an empty list
    ({"model": "LISTED_EXCEPTIONS", "exceptions": []}, "non-empty"),
    ({"model": "NONE"}, "unsupported value"),
    (_listed(schedule_version="synth_v9"), "approved schedule version"),
    (_listed(stream=["calgary", "Calgary Synth"]), "outside the approved universe"),
    (_listed(period_start_utc="2026-08-28T04:00:00Z"), "YYYYMMDDTHHMMSSZ"),
    (_listed(failure="ANY"), "unsupported value"),
    (_listed(reference="SYNTH-TICKET"), "durable governance reference"),
    (_listed(authority_kind="ANALYST"), "unsupported value"),
    (_listed(reason=""), "non-blank"),
])
def test_schedule_exceptions_are_explicit_and_fully_typed(value, needle) -> None:
    data = v5()
    entry(data, D.SCHEDULE_EXCEPTIONS)["resolution"] = value
    assert needle in fails(data)


def test_a_listed_exception_record_validates_and_types_every_field() -> None:
    data = v5()
    entry(data, D.SCHEDULE_EXCEPTIONS)["resolution"] = _listed()
    record = parse_decision_record(finish(data))
    schedule = schedule_from_record(record, current_expected_stream_contract())
    assert schedule.available and schedule.exceptions.model is ExceptionsModel.LISTED_EXCEPTIONS
    (exc,) = schedule.exceptions.exceptions
    assert (exc.stream, exc.period_start_utc, exc.local_start, exc.utc_offset, exc.failure, exc.authority_kind,
            exc.schedule_version) == (CAL_DOWN, "20260828T040000Z", LOCAL_START, dt.timedelta(hours=-6),
                                      FK.STREAM_ABSENT_FROM_CAPTURE, AuthorityKind.COLLECTION_OWNER,
                                      "per_stream_hourly_v1")
    entry(data, D.SCHEDULE_EXCEPTIONS)["resolution"] = _listed(period_start_utc="20260828T043000Z")
    off_period = schedule_from_record(parse_decision_record(finish(data)), current_expected_stream_contract())
    assert off_period.status is SS.INVALID                                  # names no expected period


def test_vancouver_identity_names_its_governed_keys_in_schema_3() -> None:
    data = v5()
    del resolution(data, D.VANCOUVER_LOCATION_IDENTITY)["governed_locations"]
    assert "governed_locations" in fails(data)
    data = v5()
    resolution(data, D.VANCOUVER_LOCATION_IDENTITY)["governed_locations"] = [list(SUPPLIED[4])]
    assert "exactly two keys" in fails(data)
    data = v5()
    resolution(data, D.VANCOUVER_LOCATION_IDENTITY)["governed_locations"] = [list(SUPPLIED[4]), list(CAL_DOWN)]
    message = fails(data)
    assert "one city" in message or "canonical locations" in message
    data = v5()                                                    # the superseded spelling is not a governed key
    resolution(data, D.VANCOUVER_LOCATION_IDENTITY)["governed_locations"] = [["Vancouver", "Downtown"],
                                                                             ["Vancouver", "Thurlow"]]
    assert "VANCOUVER_LOCATION_IDENTITY" in fails(data)


def test_a_contract_that_contradicts_the_schedule_is_invalid() -> None:
    record = load_decision_record(V5)
    other = synthetic_contract(dataclasses.replace(COV, expected_locations=SUPPLIED[:-1]))
    assert schedule_from_record(record, other).status is SS.INVALID
    assert schedule_from_record(record, other).blocking_reasons == (SB.COLLECTION_SCHEDULE_INVALID,)
    unapproved = expected_stream_contract_from_record(None)
    assert schedule_from_record(record, unapproved).status is SS.NOT_APPROVED


# ============================================================ project schedule


def test_project_schedule_is_per_stream_and_computed_from_the_definitions() -> None:
    schedule = current_per_stream_schedule()
    assert schedule is current_per_stream_schedule()                    # resolved once
    assert schedule.status is SS.AVAILABLE and schedule.blocking_reasons == ()
    assert schedule.record_id == "pricing-authorities-v6" and schedule.schedule_version == "per_stream_hourly_v1"
    assert schedule.sharing_mode is SharingMode.PER_STREAM and schedule.capture_field == "jobs.finished_at"
    assert (schedule.detail_copy_field, schedule.detail_observation_field) == ("cars.job_finished_at",
                                                                              "cars.scraped_at")
    assert schedule.exceptions == ScheduleExceptions.none() and schedule.exceptions.model is ExceptionsModel.NO_EXCEPTIONS
    assert schedule.excused_period_count == 0 and schedule.references == (
        GOVERNANCE, "docs/decisions/governance/finished-at-timezone-and-scrape-ordering-governance-v1-2026-10-06.md")
    assert schedule.expected_streams == SUPPLIED == tuple(s.stream for s in schedule.schedules)
    assert dict(schedule.timezones.entries) == ZONES
    hours = int((LOCAL_END - LOCAL_START) / dt.timedelta(hours=1)) + 1          # inclusive end, no DST in window
    assert hours == 90
    assert dict(schedule.period_count_by_stream) == {k: hours for k in SUPPLIED}
    assert dict(schedule.period_count_by_city) == {"calgary": 2 * hours, "toronto": 2 * hours, "vancouver": 3 * hours}
    assert schedule.total_period_count == 7 * hours == 630
    for s in schedule.schedules:
        assert (s.timezone, s.local_start, s.local_end, s.end_inclusive, s.cadence) == (
            ZONES[s.city], LOCAL_START, LOCAL_END, True, "PT1H")
        assert s.record_id == "pricing-authorities-v6" and s.capture_field == "jobs.finished_at"


@pytest.mark.parametrize("city, first, last", [
    ("calgary", "20260828T040000Z", "20260831T210000Z"),     # MDT (UTC-06:00)
    ("toronto", "20260828T020000Z", "20260831T190000Z"),     # EDT (UTC-04:00)
    ("vancouver", "20260828T050000Z", "20260831T220000Z"),   # PDT (UTC-07:00)
])
def test_materialized_periods_are_local_hours_converted_with_the_city_zone(city, first, last) -> None:
    schedule = current_per_stream_schedule()
    for stream in (s for s in schedule.schedules if s.city == city):
        periods = stream.periods
        assert (periods[0].utc_text, periods[-1].utc_text) == (first, last)
        assert periods[0].local_start == LOCAL_START and periods[-1].local_start == LOCAL_END
        assert periods[0].utc_text != LOCAL_START.strftime("%Y%m%dT%H%M%SZ")    # never local time + "Z"
        assert all(re.fullmatch(r"\d{8}T\d{6}Z", p.utc_text) for p in periods)
        assert all(b.utc_start - a.utc_start == dt.timedelta(hours=1) for a, b in zip(periods, periods[1:]))
        assert all(p.fold == 0 and p.stream == stream.stream for p in periods)
        assert periods[0].local_text.endswith(("-06:00", "-04:00", "-07:00"))


def test_materialization_is_deterministic_and_never_reads_observations() -> None:
    schedule = current_per_stream_schedule()
    s = schedule.schedules[0]
    rebuilt = StreamSchedule(**{f.name: getattr(s, f.name) for f in dataclasses.fields(s)})
    assert rebuilt.materialize() == s.materialize() == s.periods
    assert materialize_local_hours(s.stream, s.timezone, s.local_start, s.local_end, True) == s.periods
    half_open = materialize_local_hours(s.stream, s.timezone, s.local_start, s.local_end, False)
    assert len(half_open) == len(s.periods) - 1 and half_open == s.periods[:-1]


# ======================================================================== DST


def test_spring_forward_skips_the_nonexistent_local_hour() -> None:
    periods = materialize_local_hours(("synth", "SYNTH Branch"), "America/Toronto", dt.datetime(2026, 3, 8, 0),
                                      dt.datetime(2026, 3, 8, 4), True)
    assert [p.local_start.hour for p in periods] == [0, 1, 3, 4]                   # 02:00 does not exist
    assert [p.utc_offset for p in periods] == [dt.timedelta(hours=-5)] * 2 + [dt.timedelta(hours=-4)] * 2
    assert all(b.utc_start - a.utc_start == dt.timedelta(hours=1) for a, b in zip(periods, periods[1:]))


@pytest.mark.parametrize("zone, standard", [("America/Toronto", -5), ("America/Winnipeg", -6),
                                            ("America/Halifax", -4)])
def test_fall_back_keeps_both_offset_distinct_repeated_hours(zone, standard) -> None:
    periods = materialize_local_hours(("synth", "SYNTH Branch"), zone, dt.datetime(2026, 11, 1, 0),
                                      dt.datetime(2026, 11, 1, 3), True)
    assert [p.local_start.hour for p in periods] == [0, 1, 1, 2, 3]
    repeated = [p for p in periods if p.local_start.hour == 1]
    assert [p.fold for p in repeated] == [0, 1]
    assert [p.utc_offset for p in repeated] == [dt.timedelta(hours=standard + 1), dt.timedelta(hours=standard)]
    assert repeated[0].utc_text != repeated[1].utc_text                             # never deduplicated
    assert len({p.utc_start for p in periods}) == 5
    assert all(b.utc_start - a.utc_start == dt.timedelta(hours=1) for a, b in zip(periods, periods[1:]))


@pytest.mark.parametrize("zone", sorted(set(ZONES.values())))
@pytest.mark.parametrize("start, end", [(dt.datetime(2026, 3, 7, 22), dt.datetime(2026, 3, 8, 6)),
                                        (dt.datetime(2026, 10, 31, 22), dt.datetime(2026, 11, 1, 6))])
def test_approved_zones_follow_the_installed_iana_database(zone, start, end) -> None:
    # Independent oracle: every UTC hour whose local wall clock is a top-of-hour inside the window. Whatever
    # transitions the database defines for a zone (including none) are honoured; no offset is assumed.
    from zoneinfo import ZoneInfo

    tz, utc = ZoneInfo(zone), dt.timezone.utc
    probe = (start - dt.timedelta(days=1)).replace(tzinfo=utc)
    oracle = []
    while probe <= (end + dt.timedelta(days=1)).replace(tzinfo=utc):
        wall = probe.astimezone(tz)
        if wall.minute == 0 and start <= wall.replace(tzinfo=None) <= end:
            oracle.append((wall.replace(tzinfo=None), wall.utcoffset(), probe))
        probe += dt.timedelta(minutes=30)
    periods = materialize_local_hours(("synth", "SYNTH Branch"), zone, start, end, True)
    assert [(p.local_start, p.utc_offset, p.utc_start) for p in periods] == oracle


def test_utc_text_needs_an_aware_instant() -> None:
    with pytest.raises(ScheduleConfigurationError):
        format_utc_instant(dt.datetime(2026, 1, 1))
    aware = dt.datetime(2026, 1, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=-7)))
    assert format_utc_instant(aware) == "20260101T080000Z"


# =========================================================== typed model guards


def test_city_timezone_map_fails_closed() -> None:
    zones = CityTimezoneMap(tuple(ZONES.items()))
    assert zones.zone_name("calgary") == "America/Edmonton"
    assert CityTimezoneMap(tuple(reversed(tuple(ZONES.items())))) == zones          # deterministic order
    for zone in ("US/Mountain", "Canada/Eastern", "EST5EDT", "MDT"):              # links and abbreviations
        with pytest.raises(TemporalConfigurationError):
            CityTimezoneMap((("calgary", zone),))
    for city in ("Calgary", " calgary", "", None, "synth", 1):
        with pytest.raises(TemporalConfigurationError):
            zones.zone_name(city)
    for entries in ((), (("calgary", "UTC"),), (("calgary", "Etc/GMT+7"),), (("calgary", "America/Synthville"),),
                    (("calgary", "America/Edmonton"), ("calgary", "America/Edmonton")), (("", "America/Edmonton"),)):
        with pytest.raises(TemporalConfigurationError):
            CityTimezoneMap(entries)


def _stream_fields(**overrides) -> dict:
    s = current_per_stream_schedule().schedules[0]
    return {f.name: getattr(s, f.name) for f in dataclasses.fields(s)} | overrides


@pytest.mark.parametrize("overrides", [
    dict(city="toronto"),                                           # city must be the stream's city
    dict(local_start=LOCAL_START.replace(tzinfo=dt.timezone.utc)),  # aware boundary
    dict(local_start=LOCAL_START.replace(minute=30)),
    dict(local_end=LOCAL_START - dt.timedelta(hours=1)),
    dict(cadence="PT2H"), dict(phase="ANY"),
    dict(capture_field="cars.scraped_at"), dict(capture_field="cars.job_finished_at"),
    dict(timezone="UTC"), dict(end_inclusive=1), dict(stream=("calgary", " Calgary Downtown")),
])
def test_stream_schedule_rejects_inconsistent_definitions(overrides) -> None:
    with pytest.raises(ScheduleConfigurationError):
        StreamSchedule(**_stream_fields(**overrides))


def _schedule_fields(**overrides) -> dict:
    s = current_per_stream_schedule()
    return {f.name: getattr(s, f.name) for f in dataclasses.fields(s)} | overrides


def test_exactly_one_schedule_per_expected_stream() -> None:
    schedules = current_per_stream_schedule().schedules
    unknown = StreamSchedule(**_stream_fields(stream=("calgary", "Calgary Synth")))
    for bad in (schedules[:-1], (*schedules, schedules[0]), (*schedules[:-1], unknown), ()):
        with pytest.raises(ScheduleConfigurationError):
            PerStreamSchedule(**_schedule_fields(schedules=bad))
    with pytest.raises(ScheduleConfigurationError):
        PerStreamSchedule(**_schedule_fields(sharing_mode=SharingMode.SHARED))
    with pytest.raises(ScheduleConfigurationError):
        PerStreamSchedule(**_schedule_fields(exceptions=None))
    with pytest.raises(ScheduleConfigurationError):
        PerStreamSchedule(**_schedule_fields(timezones=CityTimezoneMap(tuple(ZONES.items())[:2])))
    with pytest.raises(ScheduleConfigurationError):
        PerStreamSchedule(status=SS.NOT_APPROVED, schedules=schedules)


def test_exceptions_model_is_explicit_and_immutable() -> None:
    none = ScheduleExceptions.none()
    assert none.model is ExceptionsModel.NO_EXCEPTIONS and none.exceptions == ()
    exc = StreamScheduleException(stream=CAL_DOWN, period_start_utc="20260828T040000Z", local_start=LOCAL_START,
                                  utc_offset=dt.timedelta(hours=-6), failure=FK.STREAM_ABSENT_FROM_CAPTURE,
                                  reason="SYNTH reason", authority_kind=AuthorityKind.COLLECTION_OWNER,
                                  reference=GOVERNANCE, schedule_version="per_stream_hourly_v1")
    with pytest.raises(ScheduleConfigurationError):
        ScheduleExceptions(model=ExceptionsModel.NO_EXCEPTIONS, exceptions=(exc,))
    with pytest.raises(ScheduleConfigurationError):
        ScheduleExceptions(model=ExceptionsModel.LISTED_EXCEPTIONS)
    with pytest.raises(ScheduleConfigurationError):
        ScheduleExceptions(model=ExceptionsModel.LISTED_EXCEPTIONS, exceptions=(exc, exc))
    for failure in (FK.PARENT_JOB_AMBIGUOUS, FK.PARENT_JOB_INVALID):          # never excusable
        with pytest.raises(ScheduleConfigurationError):
            dataclasses.replace(exc, failure=failure)
    with pytest.raises(dataclasses.FrozenInstanceError):
        exc.reason = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        none.model = ExceptionsModel.LISTED_EXCEPTIONS  # type: ignore[misc]


# ========================================================= observed assignment

ALPHA_AIR, ALPHA_DOWN, BETA_DOWN = ("alpha", "Alpha Airport"), ("alpha", "Alpha Downtown"), ("beta", "Beta Downtown")
SYN_STREAMS = (ALPHA_AIR, ALPHA_DOWN, BETA_DOWN)
SYN_ZONES = {"alpha": "America/Edmonton", "beta": "America/Toronto"}
SYN_START, SYN_END = dt.datetime(2026, 8, 27, 22), dt.datetime(2026, 8, 28, 0)
SYN_COV = dataclasses.replace(COV, expected_locations=SYN_STREAMS, mode=LocationCoverageMode.EXHAUSTIVE)
SYN_CONTRACT = synthetic_contract(SYN_COV)


def synth_schedule(start=SYN_START, end=SYN_END, zones=None, exceptions=None, version="synth_v1"):  # type: ignore[no-untyped-def]
    zones = zones or SYN_ZONES
    schedules = tuple(StreamSchedule(
        schedule_version=version, stream=s, city=s[0], timezone=zones[s[0]], local_start=start, local_end=end,
        end_inclusive=True, cadence="PT1H", phase="LOCAL_TOP_OF_HOUR", capture_field="jobs.finished_at",
        record_id="pricing-authorities-synthetic", references=("SYNTH-REFERENCE",)) for s in SYN_STREAMS)
    return PerStreamSchedule(
        status=SS.AVAILABLE, expected_streams=SYN_STREAMS, record_id="pricing-authorities-synthetic",
        schedule_version=version, capture_field="jobs.finished_at", detail_copy_field="cars.job_finished_at",
        detail_observation_field="cars.scraped_at", sharing_mode=SharingMode.PER_STREAM,
        timezones=CityTimezoneMap(tuple(zones.items())), schedules=schedules,
        exceptions=exceptions or ScheduleExceptions.none(), references=("SYNTH-REFERENCE",))


SCHEDULE = synth_schedule()


def local(hour: int, minute: int = 7, start=SYN_START) -> str:  # type: ignore[no-untyped-def]
    return (start + dt.timedelta(hours=hour, minutes=minute, seconds=13)).strftime(FINISH_FORMAT)[:-3]


def synth_frames(hours=3, skip=(), jobs_override=None, extra_jobs=(), extra_cars=()):  # type: ignore[no-untyped-def]
    """One job per city and local hour; each carries a detail row for every stream of its city."""
    jobs, cars = [], []
    for city in ("alpha", "beta"):
        for h in range(hours):
            job = f"SYNTH-JOB-{city}-{h}"
            finished = (jobs_override or {}).get((city, h), local(h))
            jobs.append({"job_id": job, "city": city, "finished_at": finished})
            for stream in SYN_STREAMS:
                if stream[0] == city and (stream, h) not in skip:
                    cars.append({"job_id": job, "city": stream[0], "location": stream[1],
                                 "job_finished_at": finished, "scraped_at": "SYNTH-OBSERVATION"})
    jobs += list(extra_jobs)
    cars += list(extra_cars)
    return pd.DataFrame(jobs), pd.DataFrame(cars)


def assess(j, c, schedule=SCHEDULE, contract=SYN_CONTRACT):  # type: ignore[no-untyped-def]
    return assess_per_stream_scheduled_coverage(j, c, schedule=schedule, contract=contract, relationship=REL)


def by_stream(report: PerStreamScheduledCoverageReport) -> dict:
    return {c.stream: c for c in report.streams}


def test_complete_synthetic_world_passes_and_frames_are_untouched() -> None:
    j, c = synth_frames()
    before = (j.copy(deep=True), c.copy(deep=True))
    report = assess(j, c)
    assert report.blocking_reasons == () and report.all_streams_complete and report.is_valid
    assert (report.jobs_assessed, report.jobs_assigned, report.jobs_not_assigned) == (6, 6, 0)
    assert all((s.expected, s.covered, s.unexcused_missing) == (3, 3, 0) for s in report.streams)
    assert dict(report.expected_city_periods) == {"alpha": 3, "beta": 3}
    pd.testing.assert_frame_equal(j, before[0]), pd.testing.assert_frame_equal(c, before[1])


def test_scraped_at_is_never_read() -> None:
    j, c = synth_frames()
    report = assess(j, c)
    assert assess(j, c.assign(scraped_at="2000-01-01 00:00:00 MST")) == report
    assert assess(j, c.drop(columns=["scraped_at"])) == report


def test_one_missing_stream_period_blocks_only_that_stream() -> None:
    j, c = synth_frames(skip={(ALPHA_DOWN, 1)})
    report = assess(j, c)
    streams = by_stream(report)
    assert (streams[ALPHA_DOWN].covered, streams[ALPHA_DOWN].unexcused_missing) == (2, 1)
    (missing,) = streams[ALPHA_DOWN].missing
    assert missing.failure is FK.STREAM_ABSENT_FROM_CAPTURE and not missing.excused
    assert missing.period.local_start == SYN_START + dt.timedelta(hours=1)
    assert streams[ALPHA_AIR].complete and streams[BETA_DOWN].complete    # airport never satisfies downtown
    assert report.blocking_reasons == (SB.SCHEDULED_COVERAGE_INCOMPLETE,)
    assert len(report.streams) == 3                                       # missing reports never disappear


def test_a_stream_never_observed_still_reports_every_period() -> None:
    j, c = synth_frames(skip={(BETA_DOWN, h) for h in range(3)})
    streams = by_stream(assess(j, c))
    assert (streams[BETA_DOWN].covered, streams[BETA_DOWN].unexcused_missing) == (0, 3)
    assert streams[ALPHA_AIR].complete and streams[ALPHA_DOWN].complete


@pytest.mark.parametrize("variant", [("alpha", "alpha downtown"), ("Alpha", "Alpha Downtown"),
                                     ("alpha", "Alpha Downtown "), (" alpha", "Alpha Downtown"),
                                     ("alpha", "Alpha  Downtown"), ("beta", "Alpha Downtown")])
def test_only_the_exact_raw_key_covers_a_stream_period(variant) -> None:
    extra = {"job_id": "SYNTH-JOB-alpha-1", "city": variant[0], "location": variant[1],
             "job_finished_at": local(1), "scraped_at": "SYNTH-OBSERVATION"}
    j, c = synth_frames(skip={(ALPHA_DOWN, 1)}, extra_cars=(extra,))
    streams = by_stream(assess(j, c))
    assert streams[ALPHA_DOWN].unexcused_missing == 1 and streams[ALPHA_DOWN].covered == 2


def test_a_row_of_another_city_never_covers_a_stream() -> None:
    extra = {"job_id": "SYNTH-JOB-alpha-1", "city": "beta", "location": "Beta Downtown",
             "job_finished_at": local(1), "scraped_at": "SYNTH-OBSERVATION"}
    j, c = synth_frames(skip={(BETA_DOWN, 1)}, extra_cars=(extra,))
    streams = by_stream(assess(j, c))
    assert streams[BETA_DOWN].unexcused_missing == 1                  # the alpha job is alpha's period only


def test_a_missing_city_period_job_makes_every_stream_of_that_city_missing() -> None:
    j, c = synth_frames()
    j = j[j["job_id"] != "SYNTH-JOB-alpha-2"]
    c = c[c["job_id"] != "SYNTH-JOB-alpha-2"]
    report = assess(j, c)
    streams = by_stream(report)
    for stream in (ALPHA_AIR, ALPHA_DOWN):
        assert [m.failure for m in streams[stream].missing] == [FK.PARENT_JOB_ABSENT]
    assert streams[BETA_DOWN].complete and dict(report.missing_city_periods) == {"alpha": 1, "beta": 0}
    assert SB.SCHEDULED_COVERAGE_INCOMPLETE in report.blocking_reasons


@pytest.mark.parametrize("city, failure", [
    (None, JF.MISSING_CITY), ("", JF.MISSING_CITY), ("   ", JF.MISSING_CITY),
    ("Alpha", JF.UNKNOWN_CITY), ("alpha ", JF.UNKNOWN_CITY), ("synth", JF.UNKNOWN_CITY),
])
def test_city_must_be_exact_and_approved(city, failure) -> None:
    extra = {"job_id": "SYNTH-JOB-extra", "city": city, "finished_at": local(1)}
    report = assess(*synth_frames(extra_jobs=(extra,)))
    assert dict(report.job_failure_counts) == {failure.value: 1}
    assert SB.SCHEDULED_JOB_ASSIGNMENT_FAILED in report.blocking_reasons


@pytest.mark.parametrize("finished, failure", [
    (None, JF.MISSING_FINISHED_AT), ("", JF.MISSING_FINISHED_AT),
    ("SYNTH", JF.INVALID_FINISHED_AT), ("2026-08-27T23:07:13Z", JF.INVALID_FINISHED_AT),
    ("2026-08-27 23:07:13.000Z", JF.INVALID_FINISHED_AT), ("2026-08-27 23:07:13 MST", JF.INVALID_FINISHED_AT),
    ("2026-02-30 23:07:13.000", JF.INVALID_FINISHED_AT),
    ("2026-08-27 20:07:13.000", JF.OUTSIDE_SCHEDULE_WINDOW), ("2026-08-28 01:07:13.000", JF.OUTSIDE_SCHEDULE_WINDOW),
])
def test_finish_time_failures_fail_closed(finished, failure) -> None:
    extra = {"job_id": "SYNTH-JOB-extra", "city": "alpha", "finished_at": finished}
    report = assess(*synth_frames(extra_jobs=(extra,)))
    assert dict(report.job_failure_counts) == {failure.value: 1}
    assert report.jobs_not_assigned == 1 and SB.SCHEDULED_JOB_ASSIGNMENT_FAILED in report.blocking_reasons
    assert all(s.complete for s in report.streams)                   # other periods are judged as before


def test_two_jobs_in_one_city_period_are_never_resolved_by_choice() -> None:
    extra_job = {"job_id": "SYNTH-JOB-dup", "city": "alpha", "finished_at": local(1, minute=40)}
    extra_cars = [{"job_id": "SYNTH-JOB-dup", "city": "alpha", "location": s[1], "job_finished_at": local(1, minute=40),
                   "scraped_at": "SYNTH-OBSERVATION"} for s in (ALPHA_AIR, ALPHA_DOWN)]
    report = assess(*synth_frames(extra_jobs=(extra_job,), extra_cars=extra_cars))
    assert dict(report.job_failure_counts) == {JF.DUPLICATE_CITY_PERIOD.value: 2}
    streams = by_stream(report)
    for stream in (ALPHA_AIR, ALPHA_DOWN):
        assert [m.failure for m in streams[stream].missing] == [FK.PARENT_JOB_AMBIGUOUS]
    assert streams[BETA_DOWN].complete
    assert {SB.SCHEDULED_JOB_ASSIGNMENT_FAILED, SB.SCHEDULED_COVERAGE_INCOMPLETE} <= set(report.blocking_reasons)


def test_detail_copy_must_agree_with_its_parent() -> None:
    j, c = synth_frames()
    c = c.copy()
    row = c.index[(c["job_id"] == "SYNTH-JOB-beta-0")][0]
    c.loc[row, "job_finished_at"] = local(0, minute=8)
    report = assess(j, c)
    assert report.detail_copy_mismatches == 1 and dict(report.job_failure_counts) == {
        JF.DETAIL_COPY_MISMATCH.value: 1}
    assert [m.failure for m in by_stream(report)[BETA_DOWN].missing] == [FK.PARENT_JOB_INVALID]
    assert {SB.SCHEDULED_DETAIL_COPY_MISMATCH, SB.SCHEDULED_JOB_ASSIGNMENT_FAILED,
            SB.SCHEDULED_COVERAGE_INCOMPLETE} <= set(report.blocking_reasons)


def test_observed_daylight_saving_ambiguity_and_gaps_fail_closed() -> None:
    fall = synth_schedule(start=dt.datetime(2026, 11, 1, 0), end=dt.datetime(2026, 11, 1, 3),
                          zones={"alpha": "America/Toronto", "beta": "America/Toronto"})
    j = pd.DataFrame([{"job_id": "SYNTH-JOB-1", "city": "alpha", "finished_at": "2026-11-01 01:30:00.000"}])
    c = pd.DataFrame(columns=["job_id", "city", "location", "job_finished_at"])
    assert dict(assess(j, c, schedule=fall).job_failure_counts) == {JF.AMBIGUOUS_LOCAL_TIME.value: 1}
    spring = synth_schedule(start=dt.datetime(2026, 3, 8, 0), end=dt.datetime(2026, 3, 8, 4))
    j = pd.DataFrame([{"job_id": "SYNTH-JOB-1", "city": "beta", "finished_at": "2026-03-08 02:30:00.000"}])
    assert dict(assess(j, c, schedule=spring).job_failure_counts) == {JF.NONEXISTENT_LOCAL_TIME.value: 1}
    # The repeated fall-back hour is two expected periods; an unambiguous job never fills both.
    assert by_stream(assess(j.iloc[:0], c, schedule=fall))[ALPHA_DOWN].expected == 5


def test_unavailable_or_invalid_schedule_and_missing_columns_fail_closed() -> None:
    j, c = synth_frames()
    for status, blocker in ((SS.NOT_APPROVED, SB.COLLECTION_SCHEDULE_UNAVAILABLE),
                            (SS.RECORD_UNAVAILABLE, SB.COLLECTION_SCHEDULE_UNAVAILABLE),
                            (SS.INVALID, SB.COLLECTION_SCHEDULE_INVALID)):
        report = assess(j, c, schedule=PerStreamSchedule(status=status))
        assert report.blocking_reasons == (blocker,) and report.streams == () and not report.all_streams_complete
    with pytest.raises(ScheduleConfigurationError):
        assess(j.drop(columns=["finished_at"]), c)
    with pytest.raises(ScheduleConfigurationError):
        assess(j, c.drop(columns=["job_finished_at"]))
    with pytest.raises(TypeError):
        assess_per_stream_scheduled_coverage(j, c, schedule=None, contract=SYN_CONTRACT, relationship=REL)


def test_schedule_streams_must_equal_the_contract() -> None:
    j, c = synth_frames()
    other = synthetic_contract(dataclasses.replace(COV, expected_locations=SYN_STREAMS[:2],
                                                   mode=LocationCoverageMode.EXHAUSTIVE))
    assert SB.SCHEDULED_COVERAGE_STREAMS_NOT_EXACT in assess(j, c, contract=other).blocking_reasons


# ================================================================== exceptions


def _exception(stream=ALPHA_DOWN, hour=1, failure=FK.STREAM_ABSENT_FROM_CAPTURE, version="synth_v1"):  # type: ignore[no-untyped-def]
    period = next(p for p in SCHEDULE.schedule_for(stream).periods if p.local_start == SYN_START + dt.timedelta(hours=hour))
    return StreamScheduleException(stream=stream, period_start_utc=period.utc_text, local_start=period.local_start,
                                   utc_offset=period.utc_offset, failure=failure, reason="SYNTH documented outage",
                                   authority_kind=AuthorityKind.COLLECTION_OWNER,
                                   reference="docs/decisions/governance/synth.md", schedule_version=version)


def _with(*exceptions):  # type: ignore[no-untyped-def]
    return synth_schedule(exceptions=ScheduleExceptions(model=ExceptionsModel.LISTED_EXCEPTIONS,
                                                        exceptions=tuple(exceptions)))


def test_an_exception_excuses_exactly_its_stream_period_and_failure() -> None:
    j, c = synth_frames(skip={(ALPHA_DOWN, 1)})
    report = assess(j, c, schedule=_with(_exception()))
    streams = by_stream(report)
    assert (streams[ALPHA_DOWN].excused, streams[ALPHA_DOWN].unexcused_missing) == (1, 0)
    assert report.blocking_reasons == () and report.excused_total == 1
    # Never across streams, periods or failure kinds.
    for exc in (_exception(stream=ALPHA_AIR), _exception(hour=2), _exception(failure=FK.PARENT_JOB_ABSENT)):
        report = assess(j, c, schedule=_with(exc))
        assert by_stream(report)[ALPHA_DOWN].unexcused_missing == 1
        assert SB.SCHEDULED_COVERAGE_INCOMPLETE in report.blocking_reasons


def test_an_exception_never_applies_across_schedule_versions_or_periods() -> None:
    with pytest.raises(ScheduleConfigurationError):
        _with(_exception(version="synth_v2"))
    bogus = dataclasses.replace(_exception(), period_start_utc="20260828T043000Z")
    with pytest.raises(ScheduleConfigurationError):
        _with(bogus)
    shifted = dataclasses.replace(_exception(), utc_offset=dt.timedelta(hours=-7))
    with pytest.raises(ScheduleConfigurationError):
        _with(shifted)


# ================================================== real configuration, readiness


def real_frames(skip=()):  # type: ignore[no-untyped-def]
    """Synthetic jobs at every approved city-hour carrying every approved stream (real keys are configuration)."""
    job_rows, car_rows = [], []
    for city in ZONES:
        for h in range(90):
            job = f"SYNTH-JOB-{city}-{h:03d}"
            finished = local(h, start=LOCAL_START)
            streams = [k for k in SUPPLIED if k[0] == city and (k, h) not in skip]
            job_rows.append({"job_id": job, "record_count": str(len(streams)), "actual_car_rows": str(len(streams)),
                             "city": city, "finished_at": finished})
            car_rows += [{"job_id": job, "row_index": str(i), "city": k[0], "location": k[1],
                          "job_finished_at": finished} for i, k in enumerate(streams)]
    return frame(JOBS, job_rows), frame(CARS, car_rows)


def pricing_with(j, c, scheduled, **gates):  # type: ignore[no-untyped-def]
    report = completeness_of(j, c)
    policy = VANCOUVER_LOCATION_POLICY
    merged = gates_for(j, c, COV, report) | {"expected_stream_contract": current_expected_stream_contract(),
                                              "location_authority": current_location_authority(),
                                              "scheduled_coverage": scheduled} | gates
    return report, assess_pricing_readiness(
        location_policy=assess_location_policy(policy, None, apply_location_policy(c, policy)), **merged)


def real_report(j, c):  # type: ignore[no-untyped-def]
    return assess_per_stream_scheduled_coverage(j, c, schedule=current_per_stream_schedule(),
                                                contract=current_expected_stream_contract(), relationship=REL)


def test_readiness_clears_schedule_unavailability_with_the_approved_schedule() -> None:
    j, c = real_frames()
    scheduled = real_report(j, c)
    assert scheduled.blocking_reasons == () and scheduled.jobs_assigned == 270
    assert sum(s.covered for s in scheduled.streams) == 630
    completeness, pricing = pricing_with(j, c, scheduled)
    # The corrected keys match exactly: no spelling or unexpected-stream blocker; only the offer combination remains.
    assert not {CMP.SOURCE_SPELLING_MISMATCH, CMP.UNEXPECTED_PAIRS} & set(completeness.blocking_reasons)
    assert pricing.blocking_reasons == (PB.CANONICAL_OFFER_COMBINATION_UNRESOLVED,) and not pricing.ready
    assert pricing.schedule_available and pricing.scheduled_coverage_complete


def test_the_calgary_downtown_gap_stays_an_unexcused_blocker() -> None:
    j, c = real_frames(skip={(CAL_DOWN, 40)})
    scheduled = real_report(j, c)
    streams = by_stream(scheduled)
    assert (streams[CAL_DOWN].covered, streams[CAL_DOWN].unexcused_missing, streams[CAL_DOWN].excused) == (89, 1, 0)
    assert [m.failure for m in streams[CAL_DOWN].missing] == [FK.STREAM_ABSENT_FROM_CAPTURE]
    assert streams[CAL_AIR].complete                                        # airport never satisfies downtown
    assert all(s.complete for k, s in streams.items() if k != CAL_DOWN)
    assert scheduled.blocking_reasons == (SB.SCHEDULED_COVERAGE_INCOMPLETE,)
    _, pricing = pricing_with(j, c, scheduled, temporal_fields_trusted=False)
    blockers = set(pricing.blocking_reasons)
    assert PB.SCHEDULED_COVERAGE_INCOMPLETE in blockers and PB.COLLECTION_SCHEDULE_UNAVAILABLE not in blockers
    assert {PB.TEMPORAL_FIELDS_UNTRUSTED, PB.CANONICAL_OFFER_COMBINATION_UNRESOLVED} <= blockers   # unrelated stay
    assert not {PB.SOURCE_SPELLING_MISMATCH, PB.UNEXPECTED_SOURCE_STREAMS} & blockers
    assert not pricing.ready and not pricing.scheduled_coverage_complete


def test_readiness_reports_a_missing_unavailable_or_mismatched_schedule() -> None:
    j, c = real_frames()
    _, pricing = pricing_with(j, c, None)
    assert PB.SCHEDULED_COVERAGE_ASSESSMENT_MISSING in pricing.blocking_reasons and not pricing.ready
    unavailable = assess_per_stream_scheduled_coverage(
        j, c, schedule=PerStreamSchedule(status=SS.NOT_APPROVED), contract=current_expected_stream_contract(),
        relationship=REL)
    _, pricing = pricing_with(j, c, unavailable)
    assert PB.COLLECTION_SCHEDULE_UNAVAILABLE in pricing.blocking_reasons and not pricing.ready
    mismatched = assess(*synth_frames())                                     # assessed for another contract
    _, pricing = pricing_with(j, c, mismatched)
    assert PB.SCHEDULED_COVERAGE_CONTRACT_MISMATCH in pricing.blocking_reasons and not pricing.ready
    assert {b.value for b in SB} <= {b.value for b in PB}


def test_the_display_spellings_no_longer_establish_coverage() -> None:
    j, c = real_frames()
    display = dict(zip(SUPPLIED, DISPLAY_SPELLINGS))
    c = c.copy()
    keys = [display[(a, b)] for a, b in zip(c["city"], c["location"])]
    c["city"], c["location"] = [k[0] for k in keys], [k[1] for k in keys]
    streams = real_report(j, c).streams
    assert all(s.covered == 0 and s.unexcused_missing == 90 for s in streams)


# ============================================================= baseline and docs


def test_baseline_reports_the_schedule_in_aggregate_without_source_values() -> None:
    import json

    from ql2_sixt_canada_analysis.pricing_baseline import build_pricing_baseline, render_baseline_markdown
    from ql2_sixt_canada_analysis.stability import assess_vehicle_attribute_stability

    j, c = real_frames(skip={(CAL_DOWN, 40)})
    _, pricing = pricing_with(j, c, real_report(j, c))
    stable = assess_vehicle_attribute_stability(c.assign(car_name="SYNTH Vehicle"))
    baseline = build_pricing_baseline(pricing=pricing, jobs=j, cars=c, temporal=None, vehicle_stability=stable)
    summary = baseline.collection_schedule
    assert (summary.status, summary.schedule_version, summary.sharing_model, summary.capture_field,
            summary.exceptions_model, summary.excused_period_count) == (
        "available", "per_stream_hourly_v1", "per_stream", "jobs.finished_at", "no_exceptions", 0)
    assert (summary.schedule_count, summary.total_periods, summary.unexcused_missing_periods) == (7, 630, 1)
    assert dict(summary.city_periods) == {"calgary": 180, "toronto": 180, "vancouver": 270}
    assert (summary.jobs_assessed, summary.jobs_assigned, summary.job_failures) == (270, 270, ())
    rows = {s.stream: s for s in summary.streams}
    assert (rows[CAL_DOWN].expected_periods, rows[CAL_DOWN].covered_periods,
            rows[CAL_DOWN].missing_by_failure) == (90, 89, (("stream_absent_from_capture", 1),))
    markdown = render_baseline_markdown(baseline, commit="abc1234", date="2026-10-06")
    assert "Collection schedule (authority-backed, per stream)" in markdown and "**NOT PRICING READY**" in markdown
    assert "| calgary / Calgary Downtown | 90 | 89 | 1 | 0 |" in markdown
    assert "SYNTH" not in markdown and not re.search(r"\d{8}T\d{6}Z|\d{4}-\d{2}-\d{2} \d{2}:", markdown)
    json.dumps(baseline.to_dict())


def test_documentation_describes_the_per_stream_schedule() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    records = " ".join((RECORD_DIR / "README.md").read_text(encoding="utf-8").split())
    for phrase in ("PER_STREAM", "jobs.finished_at", "cars.scraped_at", "NO_EXCEPTIONS", "America/Edmonton",
                   "YYYYMMDDTHHMMSSZ", "630", "calgary / Calgary Downtown", "scheduled_coverage_incomplete",
                   "collection-schedule-governance-v1-2026-10-06.md"):
        assert phrase in readme, phrase
    for phrase in ("v5.toml", "schema 3", "PER_STREAM", "NO_EXCEPTIONS", "supersede"):
        assert phrase in records, phrase
    for text in (readme, records):
        assert "global shared schedule" not in text.lower() or "not a global shared schedule" in text.lower()
        assert "Calgary Downtown gap is excused" not in text
    assert "The dataset is **not** pricing ready" in readme
