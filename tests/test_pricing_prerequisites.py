"""Pricing readiness requires an authoritative schedule, complete scheduled coverage of every
expected stream and a trusted job-detail join - directly, as typed inputs.

Fabricated frames and a synthetic schedule (test configuration only; the project
``COLLECTION_SCHEDULE`` stays ``None``). Real assessments produce every input.
"""

from __future__ import annotations

import dataclasses
import inspect

import pandas as pd
import pytest
from conftest import linked_join
from test_city_integrity import COV, PROJECT_GATES, completeness as project_completeness, cross_city, healthy
from test_completeness import SYNTH_COV
from test_readiness import (
    DISTINCT, GATES, JOIN_OK, SCHEDULE, SCHEDULED_OK, captured, scheduled_coverage, scheduled_frames,
)

from ql2_sixt_canada_analysis import join_readiness
from ql2_sixt_canada_analysis.join_readiness import JobDetailJoinBlocker as JB
from ql2_sixt_canada_analysis.readiness import (
    PricingBlocker as B,
    PricingNotReadyError,
    assess_location_policy,
    assess_pricing_readiness,
    validate_pricing_readiness,
)
from ql2_sixt_canada_analysis.relationships import ValidatedJoinError
from ql2_sixt_canada_analysis.schemas import COLLECTION_SCHEDULE
from ql2_sixt_canada_analysis.streams import (
    CollectionScheduleAssessment,
    CollectionScheduleStatus as CS,
    ScheduledCoverageBlocker as SB,
    TimeCoverageStatus as T,
    assess_collection_schedule,
    assess_expected_location_streams,
    assess_scheduled_time_coverage,
)

POLICY = assess_location_policy(DISTINCT)          # synthetic resolved policy: only the tested gates block


def ready(**changes):  # type: ignore[no-untyped-def]
    join = changes.get("job_detail_join")
    if "job_linkage" not in changes and join is not None and hasattr(join, "job_linkage_report"):
        changes["job_linkage"] = join.job_linkage_report        # the report the join was assessed with
    return assess_pricing_readiness(location_policy=POLICY, **(GATES | changes))


def streams_with(statuses: dict, schedule=SCHEDULE):  # type: ignore[no-untyped-def]
    """The real scheduled aggregate with chosen per-stream time coverage (by contract position)."""
    agg = SCHEDULED_OK.expected_streams
    results = tuple(dataclasses.replace(r, report=dataclasses.replace(r.report, time_coverage=statuses[i]))
                    if i in statuses else r for i, r in enumerate(agg.results))
    return dataclasses.replace(agg, results=results, schedule=schedule)


def coverage_of(streams, schedule=SCHEDULE):  # type: ignore[no-untyped-def]
    return assess_scheduled_time_coverage(assess_collection_schedule(schedule), streams)


def project_scheduled(statuses: dict | None = None, results=None):  # type: ignore[no-untyped-def]
    """Project three-stream contract: real scheduled aggregate, optionally altered."""
    j, c = healthy()
    agg = assess_expected_location_streams(j, captured(c), coverage=COV, schedule=SCHEDULE)
    if statuses:
        agg = dataclasses.replace(agg, results=tuple(
            dataclasses.replace(r, report=dataclasses.replace(r.report, time_coverage=statuses[i]))
            if i in statuses else r for i, r in enumerate(agg.results)))
    if results is not None:
        agg = dataclasses.replace(agg, results=results(agg.results))
    return coverage_of(agg)


def project_ready(scheduled, join=None):  # type: ignore[no-untyped-def]
    j, c = healthy()
    gates = PROJECT_GATES(project_completeness(j, c)) | {"scheduled_coverage": scheduled}
    if join is not None:
        gates["job_detail_join"] = join
    return assess_pricing_readiness(location_policy=POLICY, **gates)


# ------------------------------------------------------------- the passing baseline


def test_fixtures_pass_every_gate():
    assert SCHEDULED_OK.is_valid and SCHEDULED_OK.all_streams_complete
    assert [e.time_coverage for e in SCHEDULED_OK.stream_coverage] == [T.COMPLETE]
    assert JOIN_OK.join_ready
    report = ready()
    assert report.ready and report.schedule_available and report.scheduled_coverage_complete
    assert report.trusted_join_ready and report.scheduled_coverage is SCHEDULED_OK and report.job_detail_join is JOIN_OK


def test_project_three_streams_with_complete_coverage_pass():
    scheduled = project_scheduled()
    assert [e.target for e in scheduled.stream_coverage] == list(COV.expected_locations)
    assert all(e.time_coverage is T.COMPLETE for e in scheduled.stream_coverage)
    assert project_ready(scheduled).ready


# ---------------------------------------------------- schedule availability


def test_project_schedule_is_none_and_unavailable():
    assert COLLECTION_SCHEDULE is None
    assessment = assess_collection_schedule()
    assert assessment.status is CS.UNAVAILABLE and not assessment.available and assessment.schedule is None


def test_unavailable_schedule_blocks_pricing():
    report = ready(scheduled_coverage=scheduled_coverage(schedule=None))
    assert not report.ready and not report.schedule_available
    assert report.blocking_reasons == (B.COLLECTION_SCHEDULE_UNAVAILABLE, B.SCHEDULED_COVERAGE_INCOMPLETE)


def test_missing_schedule_assessment_blocks_pricing():
    report = ready(scheduled_coverage=None)
    assert report.blocking_reasons == (B.SCHEDULED_COVERAGE_ASSESSMENT_MISSING,)
    missing_streams = assess_scheduled_time_coverage(assess_collection_schedule(SCHEDULE), None)
    assert ready(scheduled_coverage=missing_streams).blocking_reasons == (B.SCHEDULED_COVERAGE_STREAMS_UNAVAILABLE,)


@pytest.mark.parametrize("make", [
    lambda: "hourly",                                                              # not a schedule
    lambda: _bypassed(timestamp_column="synth_not_a_field"),                        # outside the contracts
    lambda: _bypassed(expected_periods=()),                                         # structurally incomplete
])
def test_invalid_schedule_blocks_pricing(make):
    assessment = assess_collection_schedule(make())
    assert assessment.status is CS.INVALID and assessment.schedule is None
    report = ready(scheduled_coverage=assess_scheduled_time_coverage(assessment, SCHEDULED_OK.expected_streams))
    assert not report.ready and B.COLLECTION_SCHEDULE_INVALID in report.blocking_reasons


def _bypassed(**fields):  # type: ignore[no-untyped-def]
    schedule = dataclasses.replace(SCHEDULE)
    for name, value in fields.items():
        object.__setattr__(schedule, name, value)
    return schedule


def test_schedule_assessment_cannot_be_fabricated():
    with pytest.raises(ValueError):
        CollectionScheduleAssessment(CS.AVAILABLE)                     # available without a schedule
    with pytest.raises(ValueError):
        CollectionScheduleAssessment(CS.UNAVAILABLE, SCHEDULE)
    with pytest.raises(TypeError):
        CollectionScheduleAssessment("available", SCHEDULE)  # type: ignore[arg-type]


# --------------------------------------------------- per-stream coverage status


@pytest.mark.parametrize("status", [T.NOT_ASSESSED, T.NEVER_PRESENT, T.PARTIAL, T.UNASSESSABLE,
                                    "complete", "complete_ish", None])
def test_every_non_complete_status_fails(status):
    scheduled = coverage_of(streams_with({0: status}))
    assert scheduled.blocking_reasons == (SB.SCHEDULED_COVERAGE_INCOMPLETE,)
    report = ready(scheduled_coverage=scheduled)
    assert not report.ready and report.blocking_reasons == (B.SCHEDULED_COVERAGE_INCOMPLETE,)
    assert not report.scheduled_coverage_complete


def test_complete_calgary_does_not_mask_incomplete_vancouver():
    scheduled = project_scheduled({1: T.NOT_ASSESSED, 2: T.PARTIAL})
    assert [e.complete for e in scheduled.stream_coverage] == [True, False, False]
    assert not project_ready(scheduled).ready
    assert not project_ready(project_scheduled({2: T.NEVER_PRESENT})).ready        # two of three


def test_duplicate_or_unexpected_reports_do_not_replace_a_missing_stream():
    duplicate = project_scheduled(results=lambda rs: (rs[0], rs[0], rs[1]))        # third stream missing
    assert SB.STREAM_POPULATION_NOT_EXACT in duplicate.blocking_reasons
    assert duplicate.stream_coverage[0].time_coverage is None                       # duplicated: no single report
    assert duplicate.stream_coverage[2].time_coverage is None                       # missing
    extra = project_scheduled(results=lambda rs: (rs[0], rs[1], dataclasses.replace(
        rs[2], target=("SYNTH-CITY-9", "SYNTH-BRANCH-Z"))))
    assert SB.STREAM_POPULATION_NOT_EXACT in extra.blocking_reasons
    for scheduled in (duplicate, extra):
        report = project_ready(scheduled)
        assert not report.ready and B.SCHEDULED_COVERAGE_STREAMS_NOT_EXACT in report.blocking_reasons


def test_coverage_must_be_assessed_against_the_assessed_schedule():
    other = dataclasses.replace(SCHEDULE, expected_periods=("2025-01-16T12:00:00Z",))
    stale = coverage_of(dataclasses.replace(SCHEDULED_OK.expected_streams, schedule=None))
    assert ready(scheduled_coverage=stale).blocking_reasons == (B.SCHEDULED_COVERAGE_SCHEDULE_NOT_APPLIED,)
    mismatched = coverage_of(SCHEDULED_OK.expected_streams, schedule=other)
    assert B.SCHEDULED_COVERAGE_SCHEDULE_NOT_APPLIED in ready(scheduled_coverage=mismatched).blocking_reasons


def test_coverage_for_another_contract_is_rejected():
    j, c = healthy()
    foreign = scheduled_coverage(frames=(j, captured(c)), coverage=COV)            # three-stream contract
    report = ready(scheduled_coverage=foreign)                                       # completeness is SYNTH_COV
    assert B.SCHEDULED_COVERAGE_CONTRACT_MISMATCH in report.blocking_reasons and not report.ready


def test_stream_health_and_temporal_trust_do_not_override_missing_coverage():
    j, c = scheduled_frames()
    unscheduled = assess_expected_location_streams(j, c, coverage=SYNTH_COV, schedule=None)
    assert unscheduled.all_expected_streams_healthy                                  # healthy, but not assessed
    assert [r.report.time_coverage for r in unscheduled.results] == [T.NOT_ASSESSED]
    report = ready(scheduled_coverage=coverage_of(unscheduled, schedule=None), temporal_fields_trusted=True)
    assert not report.ready and B.COLLECTION_SCHEDULE_UNAVAILABLE in report.blocking_reasons
    report = ready(scheduled_coverage=coverage_of(unscheduled))                     # schedule given, not applied
    assert {B.SCHEDULED_COVERAGE_SCHEDULE_NOT_APPLIED, B.SCHEDULED_COVERAGE_INCOMPLETE} <= set(report.blocking_reasons)


# ------------------------------------------------------------------ trusted join


def construction_failure(monkeypatch):  # type: ignore[no-untyped-def]
    """A real join assessment whose cardinality-validated merge fails (JOIN_CONSTRUCTION_FAILED)."""
    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValidatedJoinError("synthetic construction failure")
    monkeypatch.setattr(join_readiness, "join_jobs_to_details", fail)
    join = linked_join(*scheduled_frames())
    monkeypatch.undo()
    return join


def test_join_construction_failure_is_an_explicit_pricing_blocker(monkeypatch):
    join = construction_failure(monkeypatch)
    assert join.blocking_reasons == (JB.JOIN_CONSTRUCTION_FAILED,) and not join.join_ready
    report = ready(job_detail_join=join)
    assert not report.ready and not report.trusted_join_ready
    assert report.blocking_reasons == (B.TRUSTED_JOIN_NOT_READY, B.JOIN_CONSTRUCTION_FAILED)
    with pytest.raises(PricingNotReadyError) as info:
        validate_pricing_readiness(location_policy=POLICY, **(GATES | {"job_detail_join": join,
                                                                        "job_linkage": join.job_linkage_report}))
    assert "join_construction_failed" in str(info.value) and "SYNTH" not in str(info.value)


def test_missing_join_assessment_blocks():
    assert ready(job_detail_join=None).blocking_reasons == (B.TRUSTED_JOIN_ASSESSMENT_MISSING,)


def test_failed_business_key_blocks_despite_a_valid_relationship():
    j, c = scheduled_frames()
    c = pd.concat([c, c.iloc[[0]]], ignore_index=True)                              # duplicate detail key
    j = j.copy()
    j.loc[0, ["record_count", "actual_car_rows"]] = 2
    join = linked_join(j, c)
    assert join.relationship_contract_valid and not join.details_key_contract_valid
    assert join.diagnostic_jobs_with_details is not None                            # a frame exists...
    report = ready(job_detail_join=join)
    assert not report.ready and B.JOIN_DETAILS_KEY_CONTRACT_FAILED in report.blocking_reasons


def test_unreconciled_counts_block_despite_a_valid_relationship():
    j, c = scheduled_frames()
    j = j.copy()
    j.loc[0, "actual_car_rows"] = 9
    join = linked_join(j, c)
    assert join.relationship_contract_valid and not join.declared_counts_reconciled
    assert B.JOIN_DECLARED_COUNTS_NOT_RECONCILED in ready(job_detail_join=join).blocking_reasons


def test_cross_city_row_keeps_join_and_pricing_blocked():
    join = linked_join(*cross_city())
    assert not join.join_ready and join.trusted_jobs_with_details is None
    report = ready(job_detail_join=join)
    assert B.JOIN_PARENT_DETAIL_CITY_MISMATCH in report.blocking_reasons and not report.ready


def test_a_non_none_joined_frame_is_not_readiness(monkeypatch):
    join = construction_failure(monkeypatch)
    j, c = scheduled_frames()
    duplicate = linked_join(j, pd.concat([c, c.iloc[[0]]], ignore_index=True))
    assert duplicate.diagnostic_jobs_with_details is not None
    assert not ready(job_detail_join=duplicate).ready and not ready(job_detail_join=join).ready


# --------------------------------------------------------------- combined behaviour


def test_both_reported_failures_accumulate(monkeypatch):
    join = construction_failure(monkeypatch)
    report = ready(scheduled_coverage=scheduled_coverage(schedule=None), job_detail_join=join,
                   key_contracts_valid=False)
    assert report.blocking_reasons == (B.KEY_CONTRACTS_INVALID, B.COLLECTION_SCHEDULE_UNAVAILABLE,
                                       B.SCHEDULED_COVERAGE_INCOMPLETE, B.TRUSTED_JOIN_NOT_READY,
                                       B.JOIN_CONSTRUCTION_FAILED)


def test_each_new_prerequisite_is_independent(monkeypatch):
    join = construction_failure(monkeypatch)
    assert ready(job_detail_join=join).scheduled_coverage_complete              # valid schedule, invalid join
    unscheduled = ready(scheduled_coverage=scheduled_coverage(schedule=None))
    assert unscheduled.trusted_join_ready and not unscheduled.ready              # valid join, no schedule


def test_ordering_is_deterministic_under_reordered_stream_reports():
    forward = project_scheduled({1: T.PARTIAL})
    agg = forward.expected_streams
    backward = coverage_of(dataclasses.replace(agg, results=tuple(reversed(agg.results))))
    assert backward.stream_coverage == forward.stream_coverage
    assert backward.blocking_reasons == forward.blocking_reasons
    assert project_ready(backward).blocking_reasons == project_ready(forward).blocking_reasons


def test_new_inputs_cannot_be_omitted():
    parameters = inspect.signature(assess_pricing_readiness).parameters
    for name in ("scheduled_coverage", "job_detail_join", "job_linkage"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty
    legacy = {k: v for k, v in GATES.items() if k not in ("scheduled_coverage", "job_detail_join", "job_linkage")}
    with pytest.raises(TypeError):
        assess_pricing_readiness(location_policy=POLICY, **legacy)  # type: ignore[call-arg]
    for name, value in (("scheduled_coverage", True), ("job_detail_join", True),
                        ("scheduled_coverage", JOIN_OK), ("job_detail_join", SCHEDULED_OK),
                        ("job_linkage", True), ("job_linkage", JOIN_OK)):
        with pytest.raises(TypeError):
            ready(**{name: value})


# --------------------------------------------------------- full regressions


def test_regression_no_schedule_with_healthy_streams_is_not_ready():
    # Reported: COLLECTION_SCHEDULE=None gave NOT_ASSESSED coverage while streams were
    # healthy, and nothing in pricing readiness required a schedule.
    j, c = scheduled_frames()
    streams = assess_expected_location_streams(j, c, coverage=SYNTH_COV, schedule=COLLECTION_SCHEDULE)
    assert streams.all_expected_streams_healthy
    assert all(r.report.time_coverage is T.NOT_ASSESSED for r in streams.results)
    scheduled = assess_scheduled_time_coverage(assess_collection_schedule(COLLECTION_SCHEDULE), streams)
    report = ready(scheduled_coverage=scheduled)
    assert report.ready is False
    assert report.blocking_reasons == (B.COLLECTION_SCHEDULE_UNAVAILABLE, B.SCHEDULED_COVERAGE_INCOMPLETE)


def test_regression_join_construction_failure_with_every_other_gate_passing(monkeypatch):
    # Reported: job_detail_join_ready was never passed to pricing readiness, so a modeled
    # JOIN_CONSTRUCTION_FAILED did not block pricing.
    join = construction_failure(monkeypatch)
    assert ready().ready                                                        # everything else passes
    report = ready(job_detail_join=join)
    assert report.ready is False and B.JOIN_CONSTRUCTION_FAILED in report.blocking_reasons
