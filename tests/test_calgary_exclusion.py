"""The governed INCOMPLETE_PARENT_CAPTURE exclusion (pricing-authorities-v8, schema 4).

Records are read as data; negative cases mutate parsed copies in memory.
Frames are fabricated (``SYNTH-JOB-*``); the only real values are approved
configuration (cities, stream keys, the governed period) read from the record.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from test_collection_schedule import (
    ALPHA_AIR,
    ALPHA_DOWN,
    BETA_DOWN,
    CAL_AIR,
    CAL_DOWN,
    CALGARY_EXCLUSION_DOC,
    EXCLUDED_HOUR,
    SCHEDULE,
    SYN_START,
    assess,
    by_stream,
    real_frames,
    real_report,
    synth_frames,
    synth_schedule,
)
from test_expected_stream_contract import COV, SUPPLIED

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
    CaptureExclusionSet,
    ExceptionsModel,
    ParentCaptureExclusion,
    ScheduleConfigurationError,
    ScheduleCoverageBlocker as SB,
    ScheduleExceptions,
    ScheduleFailureKind as FK,
    StreamScheduleException,
    current_per_stream_schedule,
)
from ql2_sixt_canada_analysis.readiness import PricingBlocker as PB
from ql2_sixt_canada_analysis.streams import StreamContinuity, assess_expected_location_streams

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V7, V8 = RECORD_DIR / "v7.toml", RECORD_DIR / "v8.toml"
D = DecisionId
HISTORY_SHA256 = {
    "v6.toml": "92dcb8e10faf963fd19c36ab52d1f626f7c30cea7742003644e555d44e795148",
    "v7.toml": "18d9da1caea4444db131e398b4cf216950269077d366c584ac5719d05d0b3cf7",
}
GOVERNED_PERIOD = "20260828T170000Z"


def v8() -> dict:
    return tomllib.loads(V8.read_text(encoding="utf-8"))


def entry(data: dict, decision: DecisionId) -> dict:
    return next(e for e in data["decisions"] if e["id"] == decision.value)


def exclusion_item(data: dict) -> dict:
    item, = entry(data, D.SCHEDULE_EXCEPTIONS)["resolution"]["exceptions"]
    return item


def fails(data: dict) -> str:
    statuses = [e["status"] for e in data["decisions"]]
    data["summary"] = {s.value.lower(): statuses.count(s.value) for s in DecisionStatus}
    with pytest.raises(DecisionRecordError) as info:
        parse_decision_record(data)
    return str(info.value)


# ======================================================================= authority


def test_v8_is_current_supersedes_v7_and_history_is_unchanged() -> None:
    record = load_decision_record(V8)
    assert (record.schema_version, record.record_version, record.record_id, record.supersedes) == (
        4, 8, "pricing-authorities-v8", "pricing-authorities-v7")
    assert CURRENT_RECORD_PATH.name == "v8.toml" and load_current_decision_record() == record
    for name, digest in HISTORY_SHA256.items():
        assert hashlib.sha256((RECORD_DIR / name).read_bytes()).hexdigest() == digest, name
    v7 = load_decision_record(V7)
    assert v7.approved_resolution(D.SCHEDULE_EXCEPTIONS) == {"model": "NO_EXCEPTIONS"}   # history, superseded
    counts = record.counts()
    assert (counts[DecisionStatus.APPROVED], counts[DecisionStatus.PROPOSED], counts[DecisionStatus.REJECTED]) == (
        23, 0, 0) and record.external_inputs == ()
    # Every earlier approval is preserved unchanged, except the superseded exceptions model.
    for decision in v7.decisions:
        if decision.is_approved and decision.id is not D.SCHEDULE_EXCEPTIONS:
            assert record.decision(decision.id) == decision, decision.id


def test_the_exclusion_is_the_supplied_decision_and_never_a_job_identifier() -> None:
    record = load_decision_record(V8)
    entry_ = record.decision(D.SCHEDULE_EXCEPTIONS)
    assert entry_.joint and {a.kind for a in entry_.authority} == {AuthorityKind.COLLECTION_OWNER,
                                                                   AuthorityKind.BUSINESS_OWNER}
    assert {a.reference for a in entry_.authority} == {CALGARY_EXCLUSION_DOC}
    resolution = record.approved_resolution(D.SCHEDULE_EXCEPTIONS)
    assert resolution["model"] == "LISTED_EXCEPTIONS"
    item, = resolution["exceptions"]
    assert set(item) == {"city", "streams", "period_start_utc", "failure", "reason", "authority_kind",
                         "reference", "schedule_version"}                         # no job identifier key
    assert (item["city"], tuple(map(tuple, item["streams"])), item["period_start_utc"], item["failure"],
            item["schedule_version"], item["reference"]) == (
        "calgary", (CAL_DOWN, CAL_AIR), GOVERNED_PERIOD, "INCOMPLETE_PARENT_CAPTURE", "per_stream_hourly_v1",
        CALGARY_EXCLUSION_DOC)
    raw = V8.read_text(encoding="utf-8")
    block = raw.split('id = "SCHEDULE_EXCEPTIONS"')[1].split("[[decisions]]")[0]
    assert "job_id" not in block and not re.search(r"\d{6,}", block.replace(GOVERNED_PERIOD, ""))


@pytest.mark.parametrize("mutate, needle", [
    (lambda i: i.update(streams=[list(CAL_DOWN)]), "exactly all its streams"),               # one stream only
    (lambda i: i.update(streams=[list(CAL_DOWN), list(CAL_AIR), ["calgary", "SYNTH Branch"]]),
     "exactly all its streams"),
    (lambda i: i.update(city="Calgary"), "exactly all its streams"),                         # exact city only
    (lambda i: i.update(city="toronto"), "exactly all its streams"),
    (lambda i: i.update(period_start_utc="2026-08-28T17:00:00Z"), "YYYYMMDDTHHMMSSZ"),
    (lambda i: i.update(schedule_version="per_stream_hourly_v2"), "schedule version"),
    (lambda i: i.update(reference="SYNTH-TICKET"), "durable governance reference"),
    (lambda i: i.update(failure="OTHER_FAILURE"), "required fields"),           # never a partial-stream excuse
    (lambda i: i.update(job_id="SYNTH-JOB-1"), "exactly the required fields"),             # never a job key
    (lambda i: i.pop("reason"), "exactly the required fields"),
])
def test_exclusion_shape_fails_closed(mutate, needle) -> None:
    data = v8()
    mutate(exclusion_item(data))
    assert needle in fails(data)


def test_duplicate_or_overlapping_exceptions_fail_closed() -> None:
    data = v8()
    items = entry(data, D.SCHEDULE_EXCEPTIONS)["resolution"]["exceptions"]
    items.append(dict(items[0]))
    assert "duplicate exception" in fails(data)
    data = v8()
    items = entry(data, D.SCHEDULE_EXCEPTIONS)["resolution"]["exceptions"]
    items.append({"stream": list(CAL_DOWN), "period_start_utc": GOVERNED_PERIOD,
                  "failure": "STREAM_ABSENT_FROM_CAPTURE", "reason": "SYNTH reason",
                  "authority_kind": "COLLECTION_OWNER", "reference": CALGARY_EXCLUSION_DOC,
                  "schedule_version": "per_stream_hourly_v1"})
    assert "both excused and excluded" in fails(data)


def test_the_project_schedule_carries_exactly_one_typed_exclusion() -> None:
    schedule = current_per_stream_schedule()
    assert schedule.exceptions.model is ExceptionsModel.LISTED_EXCEPTIONS
    exclusion, = schedule.exceptions.parent_capture_exclusions
    assert exclusion.failure is FK.INCOMPLETE_PARENT_CAPTURE and exclusion.streams == (CAL_DOWN, CAL_AIR)
    period = next(p for p in schedule.schedule_for(CAL_DOWN).periods if p.utc_text == GOVERNED_PERIOD)
    for stream in (CAL_DOWN, CAL_AIR):
        assert schedule.exceptions.excluding(stream, period, "per_stream_hourly_v1") is exclusion
        assert not schedule.exceptions.excuses(stream, period, FK.STREAM_ABSENT_FROM_CAPTURE,
                                               "per_stream_hourly_v1")       # never misused as an excuse
    other = next(p for p in schedule.schedule_for(CAL_DOWN).periods if p.utc_text != GOVERNED_PERIOD)
    assert schedule.exceptions.excluding(CAL_DOWN, other, "per_stream_hourly_v1") is None   # no general rule
    for stream in SUPPLIED[2:]:
        p = schedule.schedule_for(stream).periods[0]
        assert schedule.exceptions.excluding(stream, p, "per_stream_hourly_v1") is None


# ================================================================= typed model


def _exclusion(hour=1, city="alpha", streams=(ALPHA_AIR, ALPHA_DOWN), version="synth_v1"):  # type: ignore[no-untyped-def]
    period = next(p for p in SCHEDULE.schedule_for(streams[0]).periods
                  if p.local_start == SYN_START + dt.timedelta(hours=hour))
    return ParentCaptureExclusion(city=city, streams=tuple(streams), period_start_utc=period.utc_text,
                                  local_start=period.local_start, utc_offset=period.utc_offset,
                                  reason="SYNTH documented incomplete capture",
                                  authority_kind=AuthorityKind.COLLECTION_OWNER,
                                  reference="docs/decisions/governance/synth.md", schedule_version=version)


def _with(*exclusions, exceptions=()):  # type: ignore[no-untyped-def]
    return synth_schedule(exceptions=ScheduleExceptions(model=ExceptionsModel.LISTED_EXCEPTIONS,
                                                        exceptions=tuple(exceptions),
                                                        parent_capture_exclusions=tuple(exclusions)))


def test_exclusions_must_name_every_stream_of_their_city_and_a_real_period() -> None:
    _with(_exclusion())
    with pytest.raises(ScheduleConfigurationError):
        _with(_exclusion(streams=(ALPHA_DOWN,)))                       # a partial capture exclusion
    with pytest.raises(ScheduleConfigurationError):
        _with(_exclusion(version="synth_v2"))
    with pytest.raises(ScheduleConfigurationError):
        _with(dataclasses.replace(_exclusion(), period_start_utc="20260828T043000Z"))
    with pytest.raises(ScheduleConfigurationError):
        _with(_exclusion(), _exclusion())                              # duplicate
    with pytest.raises(ScheduleConfigurationError):
        ScheduleExceptions(model=ExceptionsModel.NO_EXCEPTIONS, parent_capture_exclusions=(_exclusion(),))
    period = next(p for p in SCHEDULE.schedule_for(ALPHA_DOWN).periods
                  if p.local_start == SYN_START + dt.timedelta(hours=1))
    excuse = StreamScheduleException(stream=ALPHA_DOWN, period_start_utc=period.utc_text,
                                     local_start=period.local_start, utc_offset=period.utc_offset,
                                     failure=FK.STREAM_ABSENT_FROM_CAPTURE, reason="SYNTH reason",
                                     authority_kind=AuthorityKind.COLLECTION_OWNER,
                                     reference="docs/decisions/governance/synth.md", schedule_version="synth_v1")
    with pytest.raises(ScheduleConfigurationError):
        _with(_exclusion(), exceptions=(excuse,))                      # excused and excluded at once
    with pytest.raises(dataclasses.FrozenInstanceError):
        _exclusion().city = "beta"  # type: ignore[misc]


def test_a_matched_exclusion_makes_both_streams_analytically_null_and_keeps_raw_rows() -> None:
    j, c = synth_frames(skip={(ALPHA_DOWN, 1)})                       # airport rows only for that capture
    before = (j.copy(deep=True), c.copy(deep=True))
    report = assess(j, c, schedule=_with(_exclusion()))
    streams = by_stream(report)
    for stream in (ALPHA_AIR, ALPHA_DOWN):
        assert (streams[stream].expected, len(streams[stream].excluded), streams[stream].required,
                streams[stream].covered, streams[stream].unexcused_missing) == (3, 1, 2, 2, 0)
    assert (streams[BETA_DOWN].covered, streams[BETA_DOWN].excluded) == (3, ())   # other cities untouched
    assert report.blocking_reasons == () and report.is_valid
    assert (report.nominal_periods, report.excluded_periods, report.required_periods, report.covered_periods,
            report.excluded_parent_captures, report.unmatched_exclusions) == (9, 2, 7, 7, 1, 0)
    pd.testing.assert_frame_equal(j, before[0]), pd.testing.assert_frame_equal(c, before[1])
    excluded = report.capture_exclusions
    assert isinstance(excluded, CaptureExclusionSet) and "SYNTH-JOB" not in repr(report)   # keys stay hidden
    assert excluded.parent_mask(j).tolist() == [False, True, False, False, False, False]
    assert int(excluded.detail_mask(c).sum()) == 1                    # the remaining airport row, still present
    assert not excluded.parent_mask(j, BETA_DOWN).any()


def test_without_the_exclusion_the_gap_stays_an_unexcused_blocker() -> None:
    j, c = synth_frames(skip={(ALPHA_DOWN, 1)})
    report = assess(j, c)
    assert by_stream(report)[ALPHA_DOWN].unexcused_missing == 1
    assert SB.SCHEDULED_COVERAGE_INCOMPLETE in report.blocking_reasons
    other = assess(j, c, schedule=_with(_exclusion(hour=2)))          # another period: no general drop rule
    assert by_stream(other)[ALPHA_DOWN].unexcused_missing == 1 and SB.SCHEDULED_COVERAGE_INCOMPLETE in \
        other.blocking_reasons


@pytest.mark.parametrize("case", ["no_capture", "two_captures", "copy_mismatch"])
def test_an_exclusion_matching_zero_or_several_captures_fails_closed(case) -> None:
    if case == "no_capture":
        j, c = synth_frames()
        j, c = j[j["job_id"] != "SYNTH-JOB-alpha-1"], c[c["job_id"] != "SYNTH-JOB-alpha-1"]
    elif case == "two_captures":
        j, c = synth_frames(extra_jobs=[{"job_id": "SYNTH-JOB-alpha-1b", "city": "alpha",
                                         "finished_at": synth_frames()[0]["finished_at"].iloc[1]}])
    else:
        j, c = synth_frames()
        c = c.copy()
        c.loc[c["job_id"] == "SYNTH-JOB-alpha-1", "job_finished_at"] = "2026-08-27 23:59:59.000"
    report = assess(j, c, schedule=_with(_exclusion()))
    assert report.unmatched_exclusions == 1 and report.excluded_parent_captures == 0
    assert SB.SCHEDULE_EXCLUSION_UNMATCHED in report.blocking_reasons and not report.is_valid
    assert all(not s.excluded for s in report.streams)               # nothing silently excluded
    assert not report.capture_exclusions.parent_mask(j).any()
    assert PB.SCHEDULE_EXCLUSION_UNMATCHED.value == SB.SCHEDULE_EXCLUSION_UNMATCHED.value


# ======================================================= project counts and continuity


def test_project_counts_and_continuity_understand_the_exclusion() -> None:
    j, c = real_frames(skip={(CAL_DOWN, EXCLUDED_HOUR)})           # the decided situation, synthetic rows
    scheduled = real_report(j, c)
    assert (scheduled.nominal_periods, scheduled.excluded_periods, scheduled.required_periods,
            scheduled.covered_periods, scheduled.unexcused_missing_total) == (630, 2, 628, 628, 0)
    rows = {s.stream: (s.covered, len(s.excluded)) for s in scheduled.streams}
    assert rows == {k: ((89, 1) if k[0] == "calgary" else (90, 0)) for k in SUPPLIED}
    governed = assess_expected_location_streams(j, c, coverage=COV,
                                                capture_exclusions=scheduled.capture_exclusions)
    report = governed.reports[CAL_DOWN]
    assert report.stream_continuity is StreamContinuity.COMPLETE and governed.all_expected_streams_healthy
    accounting = report.event_accounting
    assert (accounting.in_scope_jobs, accounting.governed_excluded_jobs, accounting.jobs_with_target_details) == (
        90, 1, 89)
    ungoverned = assess_expected_location_streams(j, c, coverage=COV)
    assert ungoverned.reports[CAL_DOWN].stream_continuity is not StreamContinuity.COMPLETE
    assert not ungoverned.all_expected_streams_healthy
    with pytest.raises(TypeError):
        assess_expected_location_streams(j, c, coverage=COV, capture_exclusions={"SYNTH": 1})


# ===================================================================== governance


def test_governance_document_records_the_decision_only() -> None:
    raw = (ROOT / CALGARY_EXCLUSION_DOC).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in ("2026-10-06", "Collection owner and business owner (joint)", "per_stream_hourly_v1",
                   "`calgary`", "`Calgary Downtown`", "`Calgary Int Airport`", "INCOMPLETE_PARENT_CAPTURE",
                   GOVERNED_PERIOD, "Analytically null", "Both", "Raw data preserved", "nothing else is covered",
                   "new, versioned authority decision", "Supersedes", "NO_EXCEPTIONS", "not deleted",
                   "STREAM_ABSENT_FROM_CAPTURE", "never by a job identifier", "Not supplied"):
        assert phrase.lower() in text.lower(), phrase
    assert "job_id" not in raw and "SYNTH" not in raw and "@" not in raw
    assert not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:|\$\s?\d|\b\d+\.\d{2}\b|\d{9,}", raw)
