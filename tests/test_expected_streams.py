"""Completeness and pricing need every configured expected stream, exactly once.

Synthetic contracts with fabricated (city, branch) pairs; the expected set
always comes from the contract, never from the data.
"""

from __future__ import annotations

import dataclasses

import pytest
from test_completeness import CITY, J1, J2, J3, cars, jobs, reconcile
from test_readiness import GATES, assess_location_policy, gates_for

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.city_integrity import assess_city_integrity
from ql2_sixt_canada_analysis.coverage import assess_expected_location_coverage
from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.readiness import (
    CompletenessBlocker as CB,
    CompletenessReport,
    PricingBlocker as PB,
    PricingNotReadyError,
    assess_completeness,
    assess_pricing_readiness,
    validate_pricing_readiness,
)
from ql2_sixt_canada_analysis.schemas import (
    COMPARED_LOCATION_STREAMS,
    EXPECTED_LOCATION_COVERAGE,
    INVESTIGATED_LOCATION_STREAM,
    LocationCoverageConfigurationError,
    LocationCoverageMode,
)
from ql2_sixt_canada_analysis.streams import (
    ExpectedLocationStreamsReport,
    ExpectedStreamBlocker as EB,
    ExpectedStreamResult,
    StreamContinuity,
    assess_expected_location_streams,
    investigate_location_stream,
)

CITY2 = "SYNTH-CITY-2"
A, B, C, D = "SYNTH-BRANCH-A", "SYNTH-BRANCH-B", "SYNTH-BRANCH-C", "SYNTH-BRANCH-D"
TA, TB, TC = (CITY, A), (CITY2, B), (CITY2, C)
COV3 = dataclasses.replace(EXPECTED_LOCATION_COVERAGE, expected_locations=(TA, TB, TC),
                           mode=LocationCoverageMode.EXHAUSTIVE)


def healthy():  # type: ignore[no-untyped-def]
    """J1 in CITY with A; J2/J3 in CITY2 each with B and C (all declarations match)."""
    j = jobs((J1, 1, 1), (J2, 2, 2, CITY2), (J3, 2, 2, CITY2))
    c = cars((J1, A), (J2, B, CITY2), (J2, C, CITY2), (J3, B, CITY2), (J3, C, CITY2))
    return j, c


def completeness(j, c, streams, coverage=COV3):  # type: ignore[no-untyped-def]
    return assess_completeness(datasets=RawDatasets(jobs=j, cars=c, complete_source=True),
                               coverage=assess_expected_location_coverage(c, coverage),
                               streams=streams, reconciliation=reconcile(j, c),
                               city_integrity=assess_city_integrity(j, c, coverage=coverage), expected_coverage=coverage)


def pricing(report):  # type: ignore[no-untyped-def]
    return assess_pricing_readiness(location_policy=assess_location_policy(), **gates_for(*healthy(), COV3, report))


# ------------------------------------------------------------------ exact population


def test_exact_expected_population_is_assessed_once_and_can_complete():
    j, c = healthy()
    agg = assess_expected_location_streams(j, c, coverage=COV3)
    assert (agg.expected_stream_count, agg.assessed_stream_count) == (3, 3)
    assert [r.target for r in agg.results] == [TA, TB, TC]                  # contract order
    assert agg.all_expected_assessed and agg.assessed_exactly_once and agg.all_expected_streams_healthy
    assert agg.all_continuity_complete and agg.blocking_reasons == () and agg.is_valid
    assert set(agg.reports) == {TA, TB, TC}
    report = completeness(j, c, agg)
    assert report.complete and report.expected_streams is agg
    readiness = pricing(report)
    # Other pricing gates still apply (here the unresolved location policy).
    assert readiness.completeness is report and PB.DATA_INCOMPLETE not in readiness.blocking_reasons
    assert readiness.blocking_reasons == (PB.LOCATION_POLICY_UNRESOLVED,)


def test_omitted_expected_stream_blocks_completeness_and_pricing():
    j, c = healthy()
    full = assess_expected_location_streams(j, c, coverage=COV3)
    partial = ExpectedLocationStreamsReport.from_reports(COV3, {k: v for k, v in full.reports.items() if k != TC})
    assert (partial.assessed_stream_count, partial.missing_stream_count) == (2, 1)
    assert not partial.all_expected_assessed and not partial.all_expected_streams_healthy
    assert partial.blocking_reasons == (EB.EXPECTED_STREAM_REPORT_MISSING,)
    report = completeness(j, c, partial)
    assert not report.complete and CB.EXPECTED_STREAM_REPORT_MISSING in report.blocking_reasons
    assert {PB.DATA_INCOMPLETE, PB.EXPECTED_STREAMS_NOT_PROVEN} <= set(pricing(report).blocking_reasons)


def test_duplicate_report_does_not_masquerade_as_full_coverage():
    j, c = healthy()
    full = assess_expected_location_streams(j, c, coverage=COV3)
    dup = ExpectedLocationStreamsReport.from_reports(COV3, [(TA, full.reports[TA]), (TA, full.reports[TA]),
                                                            (TB, full.reports[TB])])
    assert dup.assessed_stream_count == 3 and not dup.assessed_exactly_once
    assert dup.blocking_reasons == (EB.EXPECTED_STREAM_REPORT_MISSING, EB.DUPLICATE_STREAM_REPORT)
    assert TA not in dup.reports                                            # ambiguous keys are not exposed
    assert not completeness(j, c, dup).complete


def test_unexpected_stream_report_is_rejected():
    j, c = healthy()
    full = assess_expected_location_streams(j, c, coverage=COV3)
    extra = ExpectedLocationStreamsReport.from_reports(
        COV3, [*full.reports.items(), ((CITY2, D), full.reports[TB])])
    assert extra.unexpected_stream_count == 1 and EB.UNEXPECTED_STREAM_REPORT in extra.blocking_reasons
    report = completeness(j, c, extra)
    assert not report.complete and CB.UNEXPECTED_STREAM_REPORT in report.blocking_reasons


def test_unavailable_and_empty_reports_fail_closed():
    j, c = healthy()
    full = assess_expected_location_streams(j, c, coverage=COV3)
    unavailable = ExpectedLocationStreamsReport.from_reports(COV3, {**full.reports, TC: None})
    assert unavailable.blocking_reasons == (EB.EXPECTED_STREAM_REPORT_UNAVAILABLE,)
    empty = ExpectedLocationStreamsReport.from_reports(COV3, [])
    assert empty.assessed_stream_count == 0 and empty.missing_stream_count == 3
    for agg in (unavailable, empty):
        assert not completeness(j, c, agg).complete


# ------------------------------------------------------------ original false passes


def test_calgary_healthy_and_one_vancouver_like_stream_partial_is_not_complete():
    # Regression: only the first stream was supplied, so completeness passed.
    j = jobs((J1, 1, 1), (J2, 2, 2, CITY2), (J3, 1, 1, CITY2))
    c = cars((J1, A), (J2, B, CITY2), (J2, C, CITY2), (J3, B, CITY2))          # C missing from J3
    agg = assess_expected_location_streams(j, c, coverage=COV3)
    assert agg.assessed_exactly_once and agg.reports[TA].is_healthy and agg.reports[TB].is_healthy
    assert agg.reports[TC].stream_continuity is StreamContinuity.PARTIAL       # still visible
    assert not agg.all_expected_streams_healthy
    assert agg.blocking_reasons == (EB.STREAM_CONTINUITY_PARTIAL, EB.EXPECTED_STREAM_UNHEALTHY)
    report = completeness(j, c, agg)
    assert not report.complete
    assert {CB.STREAM_CONTINUITY_PARTIAL, CB.STREAM_UNHEALTHY} <= set(report.blocking_reasons)
    readiness = pricing(report)
    assert not readiness.ready and PB.EXPECTED_STREAMS_NOT_PROVEN in readiness.blocking_reasons
    with pytest.raises(PricingNotReadyError):
        validate_pricing_readiness(location_policy=assess_location_policy(), **gates_for(*healthy(), COV3, report))


def test_detail_pairs_without_a_parent_job_in_their_city_are_not_complete():
    # Regression: rows claim the second-city pairs but no job of that city exists.
    j = jobs((J1, 3, 3))
    c = cars((J1, A), (J1, B, CITY2), (J1, C, CITY2))
    coverage_report = assess_expected_location_coverage(c, COV3)
    assert coverage_report.is_valid                                          # rows cover every pair...
    agg = assess_expected_location_streams(j, c, coverage=COV3)
    # The CITY job carries CITY2 rows: a cross-city parent/detail assignment, so even
    # the first stream fails (it used to pass on its own).
    assert agg.reports[TA].parent_detail_scope_agrees is False and not agg.reports[TA].is_healthy
    assert EB.STREAM_SCOPE_MISMATCH in agg.blocking_reasons
    for key in (TB, TC):                                                     # ...but no job-level stream
        assert agg.reports[key].stream_continuity is StreamContinuity.UNASSESSABLE
        assert agg.reports[key].event_accounting.in_scope_jobs == 0
    report = completeness(j, c, agg)
    assert not report.complete and CB.STREAM_CONTINUITY_UNASSESSABLE in report.blocking_reasons
    assert not pricing(report).ready


# ------------------------------------------------------------- contract handling


def test_missing_aggregate_blocks_completeness_and_pricing():
    j, c = healthy()
    report = completeness(j, c, None)
    assert CB.EXPECTED_STREAM_ASSESSMENT_UNAVAILABLE in report.blocking_reasons
    assert PB.COMPLETENESS_UNAVAILABLE in pricing(None).blocking_reasons


def test_aggregate_for_another_contract_blocks():
    j, c = healthy()
    two = dataclasses.replace(COV3, expected_locations=(TA, TB))
    agg = assess_expected_location_streams(j, c, coverage=two)
    assert agg.is_valid                                                      # valid for its own contract
    report = completeness(j, c, agg)                                         # ...not for COV3
    assert not report.complete and CB.STREAM_CONTRACT_MISMATCH in report.blocking_reasons


def test_unconfigured_contract_and_wrong_types_raise():
    unconfigured = dataclasses.replace(COV3, expected_locations=None, mode=None)
    j, c = healthy()
    with pytest.raises(LocationCoverageConfigurationError):
        assess_expected_location_streams(j, c, coverage=unconfigured)
    with pytest.raises(LocationCoverageConfigurationError):
        ExpectedLocationStreamsReport(coverage=unconfigured, results=())
    with pytest.raises(TypeError):
        ExpectedLocationStreamsReport(coverage=COV3, results=("not a result",))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        completeness(j, c, (investigate_location_stream(j, c, TA, coverage=COV3),))  # a plain tuple


def test_complete_report_cannot_be_fabricated_without_a_valid_aggregate():
    with pytest.raises(ValueError):
        CompletenessReport(blocking_reasons=())
    j, c = healthy()
    partial = ExpectedLocationStreamsReport.from_reports(COV3, [])
    with pytest.raises(ValueError):
        CompletenessReport(blocking_reasons=(), expected_streams=partial)


def test_manual_aggregation_is_order_independent():
    j, c = healthy()
    full = assess_expected_location_streams(j, c, coverage=COV3)
    items = list(full.reports.items())
    forward = ExpectedLocationStreamsReport.from_reports(COV3, items)
    backward = ExpectedLocationStreamsReport.from_reports(COV3, items[::-1] + [((CITY2, D), None)])
    assert forward == full and [r.target for r in backward.results] == [TA, TB, TC, (CITY2, D)]
    assert backward.blocking_reasons == (EB.UNEXPECTED_STREAM_REPORT,)


def test_adding_a_pair_to_the_contract_expands_the_assessment():
    j, c = healthy()
    four = dataclasses.replace(COV3, expected_locations=(TA, TB, TC, (CITY2, D)))
    agg = assess_expected_location_streams(j, c, coverage=four)
    assert (agg.expected_stream_count, agg.assessed_stream_count) == (4, 4)
    assert agg.reports[(CITY2, D)] is not None and not agg.all_expected_streams_healthy   # D never observed
    assert not completeness(j, c, agg, coverage=four).complete


def test_results_and_reports_are_immutable():
    j, c = healthy()
    agg = assess_expected_location_streams(j, c, coverage=COV3)
    with pytest.raises(TypeError):
        agg.reports[TA] = None  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        agg.results = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        agg.results[0].report = None  # type: ignore[misc]
    assert isinstance(agg.results, tuple) and agg.is_valid


def test_blocker_categories_carry_no_source_values():
    j = jobs((J1, 1, 1), (J2, 2, 2, CITY2), (J3, 1, 1, CITY2))
    c = cars((J1, A), (J2, B, CITY2), (J2, C, CITY2), (J3, B, CITY2))
    report = completeness(j, c, assess_expected_location_streams(j, c, coverage=COV3))
    text = " ".join(b.value for b in report.blocking_reasons) + " ".join(
        b.value for b in pricing(report).blocking_reasons)
    assert "SYNTH" not in text and not any(ch.isdigit() for ch in text)
    assert not any("SYNTH" in b.value for b in EB)


def test_project_contract_is_the_approved_exhaustive_universe_with_the_designated_streams():
    keys = EXPECTED_LOCATION_COVERAGE.expected_locations
    assert EXPECTED_LOCATION_COVERAGE.mode is LocationCoverageMode.EXHAUSTIVE and len(keys) == 7
    assert INVESTIGATED_LOCATION_STREAM in keys and set(COMPARED_LOCATION_STREAMS) <= set(keys)
    assert keys != (INVESTIGATED_LOCATION_STREAM, *COMPARED_LOCATION_STREAMS)   # not the former three-stream minimum


def test_package_exports():
    for name in ("ExpectedLocationStreamsReport", "ExpectedStreamBlocker", "ExpectedStreamResult",
                 "assess_expected_location_streams"):
        assert name in ql2_sixt_canada_analysis.__all__
    assert ExpectedStreamResult((CITY, A), None).report is None
