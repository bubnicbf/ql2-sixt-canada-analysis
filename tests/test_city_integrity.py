"""City (collection-scope) integrity: assignable job city and parent/detail agreement.

Fabricated jobs and rows only (``SYNTH-JOB-*``). The two cities and three
branch labels are the configured expected-location contract (repository
configuration), never source values. The invariant under test: an
unassignable job city or a cross-city parent/detail row blocks continuity,
reconciliation, the trusted join, completeness and pricing readiness.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from conftest import linked_join, require_linked_join
from test_completeness import COV, J1, J2, J3, cars, jobs, reconcile
from test_readiness import DISTINCT, GATES, gates_for

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.city_integrity import (
    CITY_INTEGRITY_SAMPLE_LIMIT,
    CityIntegrityBlocker as CIB,
    CityIntegrityError,
    CityIntegrityReport,
    ScopeMismatchSample,
    assess_city_integrity,
    unassignable_scope_mask,
    validate_city_integrity,
)
from ql2_sixt_canada_analysis.coverage import assess_expected_location_coverage
from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.join_readiness import (
    JobDetailJoinBlocker as JB,
    UntrustedJoinError,
)
from ql2_sixt_canada_analysis.readiness import (
    CompletenessBlocker as CB,
    CompletenessReport,
    PricingBlocker as PB,
    PricingNotReadyError,
    assess_completeness,
    assess_location_policy,
    assess_pricing_readiness,
    validate_pricing_readiness,
)
from ql2_sixt_canada_analysis.schemas import (
    COMPARED_LOCATION_STREAMS,
    INVESTIGATED_LOCATION_STREAM,
    JOB_DETAIL_RELATIONSHIP as REL,
    VANCOUVER_LOCATION_POLICY,
    LocationCoverageConfigurationError,
    LocationPolicyState,
    RelationshipConfigurationError,
)
from ql2_sixt_canada_analysis.streams import (
    ExpectedStreamBlocker as EB,
    LocationStreamStatus as S,
    PipelineStage as P,
    StreamContinuity,
    assess_expected_location_streams,
    investigate_location_stream,
)

#: Synthetic three-stream contract over approved keys (the investigated Calgary stream and the two
#: governed Vancouver streams) - test configuration only; the project contract is the approved universe.
COV = dataclasses.replace(COV, expected_locations=(INVESTIGATED_LOCATION_STREAM, *COMPARED_LOCATION_STREAMS))
CAL, L1 = INVESTIGATED_LOCATION_STREAM
(VAN, L2), (_, L3) = COMPARED_LOCATION_STREAMS
PARENT_CITY, DETAIL_CITY = REL.scope_agreement_columns[0]
J4 = "SYNTH-JOB-004"
UNASSIGNABLE = [None, "", "   ", "\t"]


def healthy():  # type: ignore[no-untyped-def]
    """Three expected streams, every linked row in its job's city."""
    return (jobs((J1, 1, 1, CAL), (J2, 2, 2, VAN)),
            cars((J1, L1, CAL), (J2, L2, VAN), (J2, L3, VAN)))


def cross_city():  # type: ignore[no-untyped-def]
    """The reported defect: a CAL job carries a VAN row; coverage, counts and (formerly) streams pass."""
    return (jobs((J1, 1, 1, CAL), (J2, 2, 2, VAN), (J3, 2, 2, CAL)),
            cars((J1, L1, CAL), (J2, L2, VAN), (J2, L3, VAN), (J3, L1, CAL), (J3, L2, VAN)))


def with_unassignable_job(value):  # type: ignore[no-untyped-def]
    """Healthy frames plus a zero-offer job whose city is unassignable (it used to vanish from every scope)."""
    j, c = healthy()
    return pd.concat([j, jobs((J4, 0, 0, value))], ignore_index=True), c


def completeness(j, c, **overrides):  # type: ignore[no-untyped-def]
    inputs = dict(datasets=RawDatasets(jobs=j, cars=c, complete_source=True),
                  coverage=assess_expected_location_coverage(c, COV),
                  streams=assess_expected_location_streams(j, c, coverage=COV),
                  reconciliation=reconcile(j, c), city_integrity=assess_city_integrity(j, c, coverage=COV),
                  expected_coverage=COV)
    return assess_completeness(**(inputs | overrides))


def PROJECT_GATES(report):  # type: ignore[no-untyped-def]  # noqa: N802
    """Every non-completeness gate passing for the project contract (healthy frames, synthetic schedule)."""
    return gates_for(*healthy(), COV, report)


def pricing(report):  # type: ignore[no-untyped-def]
    """Every other gate passes (synthetic resolved policy), so only completeness can block."""
    return assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                    **PROJECT_GATES(report))


# ------------------------------------------------------------- the single rule


def test_project_scope_contract_is_one_definition():
    assert REL.scope_agreement_columns == tuple(zip(COV.parent_scope_columns, COV.stream_scope_columns))
    assert REL.scope_agreement_columns == ((PARENT_CITY, DETAIL_CITY),)
    assert CAL != VAN


@pytest.mark.parametrize("value, unassignable", [
    (None, True), (np.nan, True), (pd.NA, True), ("", True), ("   ", True), ("\t\n", True),
    (" " + CAL, True), (CAL + " ", True), (7, True), (CAL, False), (VAN, False),
])
def test_unassignable_scope_rule(value, unassignable):
    frame = pd.DataFrame({PARENT_CITY: pd.Series([value, CAL], dtype=object)})
    before = frame.copy(deep=True)
    assert unassignable_scope_mask(frame, (PARENT_CITY,)).tolist() == [unassignable, False]
    pd.testing.assert_frame_equal(frame, before)                 # nothing is trimmed or repaired


def test_healthy_frames_pass_every_control():
    j, c = healthy()
    report = assess_city_integrity(j, c, coverage=COV)
    assert report.is_valid and report.blocking_reasons == ()
    assert (report.job_count, report.scope_unassignable_job_count, report.linked_detail_row_count,
            report.scope_mismatch_detail_row_count) == (2, 0, 3, 0)
    assert validate_city_integrity(j, c, coverage=COV) == report
    assert reconcile(j, c).is_reconciled
    assert linked_join(j, c).trusted_jobs_with_details is not None
    assert completeness(j, c).complete and pricing(completeness(j, c)).ready


# ------------------------------------------------------- unassignable job city


@pytest.mark.parametrize("value", UNASSIGNABLE)
def test_unassignable_job_city_fails_continuity_of_every_stream(value):
    j, c = with_unassignable_job(value)
    agg = assess_expected_location_streams(j, c, coverage=COV)
    for report in agg.reports.values():
        assert report.event_accounting.scope_unassignable_jobs == 1        # diagnostic kept
        assert report.stream_continuity is StreamContinuity.SCOPE_UNASSIGNABLE
        assert report.status is S.SCOPE_UNASSIGNABLE and report.earliest_failing_stage is P.SOURCE_CONTINUITY
        assert not report.is_healthy and report.upstream_issue_indicated
    # The job did not silently drop out: the aggregate fails with a typed, distinct reason.
    assert not agg.all_expected_streams_healthy and EB.STREAM_SCOPE_UNASSIGNABLE in agg.blocking_reasons
    assert EB.STREAM_CONTINUITY_PARTIAL not in agg.blocking_reasons


@pytest.mark.parametrize("value", UNASSIGNABLE)
def test_unassignable_job_city_blocks_integrity_completeness_and_pricing(value):
    j, c = with_unassignable_job(value)
    report = assess_city_integrity(j, c, coverage=COV)
    assert report.scope_unassignable_job_count == 1 and report.scope_unassignable_job_sample == ((J4,),)
    assert report.blocking_reasons == (CIB.CITY_SCOPE_UNASSIGNABLE,)
    with pytest.raises(CityIntegrityError) as info:
        validate_city_integrity(j, c, coverage=COV)
    assert "SYNTH" not in str(info.value) and CIB.CITY_SCOPE_UNASSIGNABLE.value in str(info.value)
    complete = completeness(j, c)
    assert not complete.complete
    assert {CB.CITY_SCOPE_UNASSIGNABLE, CB.STREAM_SCOPE_UNASSIGNABLE} <= set(complete.blocking_reasons)
    readiness = pricing(complete)
    assert not readiness.ready
    assert readiness.blocking_reasons == (PB.DATA_INCOMPLETE, PB.EXPECTED_STREAMS_NOT_PROVEN,
                                          PB.SCOPE_INTEGRITY_NOT_PROVEN)
    with pytest.raises(PricingNotReadyError):
        validate_pricing_readiness(location_policy=assess_location_policy(DISTINCT), **PROJECT_GATES(complete))
    join = linked_join(j, c)
    assert join.trusted_jobs_with_details is None and JB.CITY_SCOPE_UNASSIGNABLE in join.blocking_reasons


def test_padded_city_is_non_canonical_and_never_trimmed():
    # Policy: no city normalisation contract exists, so surrounding whitespace is not
    # trimmed into validity - the value is unassignable on the job side and never equal
    # to the canonical value on the detail side.
    j, c = with_unassignable_job(" " + CAL)
    assert assess_city_integrity(j, c, coverage=COV).scope_unassignable_job_count == 1
    j, c = healthy()
    c.loc[0, DETAIL_CITY] = CAL + " "
    report = assess_city_integrity(j, c, coverage=COV)
    assert report.scope_mismatch_detail_row_count == 1
    assert report.scope_mismatch_sample[0].detail_scope == (CAL + " ",)       # raw value, unmodified


def test_healthy_streams_supplied_by_a_caller_cannot_hide_unassignable_scope():
    # Each stream assessed on frames without the malformed job looks healthy; the
    # required city-integrity result for the real frames still blocks.
    clean_j, clean_c = healthy()
    j, c = with_unassignable_job("")
    healthy_streams = assess_expected_location_streams(clean_j, clean_c, coverage=COV)
    assert healthy_streams.is_valid
    report = completeness(j, c, streams=healthy_streams)
    assert not report.complete and report.blocking_reasons == (CB.CITY_SCOPE_UNASSIGNABLE,)
    assert PB.SCOPE_INTEGRITY_NOT_PROVEN in pricing(report).blocking_reasons


# ------------------------------------------------- parent/detail city agreement


@pytest.mark.parametrize("job_city, detail_city", [(CAL, VAN), (VAN, CAL)])
def test_cross_city_row_fails_agreement_both_directions(job_city, detail_city):
    j = jobs((J1, 1, 1, job_city))
    c = cars((J1, L1, detail_city))
    report = assess_city_integrity(j, c, coverage=COV)
    assert report.blocking_reasons == (CIB.PARENT_DETAIL_CITY_MISMATCH,)
    assert (report.scope_mismatch_detail_row_count, report.scope_mismatch_job_count) == (1, 1)
    assert report.scope_mismatch_sample == (ScopeMismatchSample(
        detail_key=(J1, 0), parent_key=(J1,), detail_scope=(detail_city,), parent_scope=(job_city,)),)


@pytest.mark.parametrize("value", UNASSIGNABLE)
def test_missing_or_blank_detail_city_fails_agreement(value):
    j, c = healthy()
    c[DETAIL_CITY] = c[DETAIL_CITY].astype(object)
    c.loc[1, DETAIL_CITY] = value
    report = assess_city_integrity(j, c, coverage=COV)
    assert report.scope_mismatch_detail_row_count == 1 and report.job_scope_assignable
    assert report.scope_mismatch_sample[0].detail_scope == (None if value is None else value,)
    assert not reconcile(j, c).is_reconciled


@pytest.mark.parametrize("value", UNASSIGNABLE)
def test_missing_or_blank_job_city_fails_agreement_and_scope(value):
    j, c = healthy()
    j[PARENT_CITY] = j[PARENT_CITY].astype(object)
    j.loc[1, PARENT_CITY] = value                                   # J2 has two linked rows
    report = assess_city_integrity(j, c, coverage=COV)
    assert report.blocking_reasons == (CIB.CITY_SCOPE_UNASSIGNABLE, CIB.PARENT_DETAIL_CITY_MISMATCH)
    assert report.scope_unassignable_job_sample == ((J2,),)
    assert (report.scope_mismatch_detail_row_count, report.scope_mismatch_job_count) == (2, 1)


def test_rows_linked_to_a_duplicated_parent_key_are_checked_against_each_parent():
    j = pd.concat([jobs((J1, 1, 1, CAL)), jobs((J1, 1, 1, VAN))], ignore_index=True)
    report = assess_city_integrity(j, cars((J1, L1, CAL)), coverage=COV)
    assert report.scope_mismatch_detail_row_count == 1                    # disagrees with one parent
    # Orphans are left to the relationship controls.
    orphan = assess_city_integrity(jobs((J1, 1, 1, CAL)), cars((J1, L1, CAL), ("SYNTH-JOB-999", L1, VAN)),
                                   coverage=COV)
    assert orphan.is_valid and orphan.linked_detail_row_count == 1


def test_cross_city_row_prevents_reconciliation():
    j, c = cross_city()
    report = reconcile(j, c)
    assert report.declared_counts_reconciled                          # counts alone match...
    assert not report.parent_detail_scope_agrees and not report.is_reconciled
    assert (report.scope_mismatch_detail_row_count, report.scope_mismatch_job_count) == (1, 1)
    assert "parent_detail_scope_mismatch" in report.violations


def test_cross_city_row_withholds_the_trusted_join():
    j, c = cross_city()
    join = linked_join(j, c)
    assert join.trusted_jobs_with_details is None and not join.join_ready
    assert JB.PARENT_DETAIL_CITY_MISMATCH in join.blocking_reasons
    assert JB.DECLARED_COUNTS_NOT_RECONCILED not in join.blocking_reasons   # the counts did match
    assert not join.city_integrity_valid and join.relationship_contract_valid
    with pytest.raises(UntrustedJoinError) as info:
        require_linked_join(j, c)
    assert "SYNTH" not in str(info.value)


def test_cross_city_row_fails_the_streams_it_concerns():
    j, c = cross_city()
    agg = assess_expected_location_streams(j, c, coverage=COV)
    for target in ((CAL, L1), (VAN, L2)):                 # the parent's stream and the row's own
        report = agg.reports[target]
        assert report.parent_detail_scope_agrees is False
        assert report.status is S.PARENT_DETAIL_SCOPE_MISMATCH and P.SCOPE_AGREEMENT in report.failing_stages
    assert EB.STREAM_SCOPE_MISMATCH in agg.blocking_reasons


def test_reported_false_pass_now_fails_closed():
    # Regression: expected coverage, all stream inputs, counts and an absent schedule
    # all looked complete while a detail row sat under another city than its job.
    j, c = cross_city()
    assert assess_expected_location_coverage(c, COV).is_valid
    report = completeness(j, c)
    assert not report.complete
    assert {CB.PARENT_DETAIL_CITY_MISMATCH, CB.STREAM_SCOPE_MISMATCH} <= set(report.blocking_reasons)
    assert linked_join(j, c).trusted_jobs_with_details is None
    readiness = pricing(report)
    assert not readiness.ready and PB.SCOPE_INTEGRITY_NOT_PROVEN in readiness.blocking_reasons


def test_healthy_coverage_streams_and_counts_cannot_override_a_mismatch():
    j, c = cross_city()
    clean_j, clean_c = healthy()
    report = completeness(j, c, streams=assess_expected_location_streams(clean_j, clean_c, coverage=COV),
                          reconciliation=reconcile(clean_j, clean_c))
    assert not report.complete and report.blocking_reasons == (CB.PARENT_DETAIL_CITY_MISMATCH,)
    assert pricing(report).blocking_reasons == (PB.DATA_INCOMPLETE, PB.SCOPE_INTEGRITY_NOT_PROVEN)


# ------------------------------------------------ required input and contract


def test_missing_or_foreign_city_integrity_blocks_completeness():
    j, c = healthy()
    assert completeness(j, c, city_integrity=None).blocking_reasons == (CB.CITY_INTEGRITY_UNAVAILABLE,)
    unbound = assess_city_integrity(j, c, coverage=None)
    assert completeness(j, c, city_integrity=unbound).blocking_reasons == (CB.CITY_INTEGRITY_CONTRACT_MISMATCH,)
    with pytest.raises(TypeError):
        completeness(j, c, city_integrity=True)
    streams = assess_expected_location_streams(j, c, coverage=COV)
    with pytest.raises(ValueError):                       # a complete report cannot omit city integrity
        CompletenessReport(blocking_reasons=(), expected_streams=streams)
    bad = assess_city_integrity(*with_unassignable_job(""), coverage=COV)
    with pytest.raises(ValueError):
        CompletenessReport(blocking_reasons=(), expected_streams=streams, city_integrity=bad)


def test_inconsistent_scope_configuration_is_rejected():
    j, c = healthy()
    with pytest.raises(LocationCoverageConfigurationError):
        assess_city_integrity(j, c, coverage=dataclasses.replace(COV, parent_scope_columns=("mode",)))
    no_scope = dataclasses.replace(REL, scope_agreement_columns=())
    with pytest.raises(LocationCoverageConfigurationError):
        investigate_location_stream(j, c, (CAL, L1), relationship=no_scope)
    with pytest.raises(RelationshipConfigurationError):
        assess_city_integrity(j, c, relationship=no_scope, coverage=None)
    with pytest.raises(RelationshipConfigurationError):
        assess_city_integrity(j.drop(columns=[PARENT_CITY]), c, coverage=COV)
    for pairs in ((("synth_unknown", DETAIL_CITY),), ((PARENT_CITY, REL.detail_key_columns[0]),), "city"):
        with pytest.raises(RelationshipConfigurationError):
            dataclasses.replace(REL, scope_agreement_columns=pairs)


# ---------------------------------------------- combined blockers and determinism


def test_simultaneous_blockers_are_all_preserved():
    j, c = cross_city()
    j = pd.concat([j, jobs((J4, 0, 0, ""))], ignore_index=True)
    j.loc[0, REL.expected_detail_count_columns[1]] = 9                 # unrelated declared-count failure
    report = completeness(j, c, datasets=RawDatasets(jobs=j, cars=c, complete_source=False))
    assert {CB.SOURCE_NOT_COMPLETE, CB.DECLARED_COUNT_UNRECONCILED, CB.CITY_SCOPE_UNASSIGNABLE,
            CB.PARENT_DETAIL_CITY_MISMATCH, CB.STREAM_SCOPE_UNASSIGNABLE} <= set(report.blocking_reasons)
    assert report.blocking_reasons == tuple(dict.fromkeys(report.blocking_reasons))
    join = linked_join(j, c)
    assert {JB.DECLARED_COUNTS_NOT_RECONCILED, JB.CITY_SCOPE_UNASSIGNABLE,
            JB.PARENT_DETAIL_CITY_MISMATCH} <= set(join.blocking_reasons)


def test_results_and_samples_are_independent_of_row_order():
    j, c = cross_city()
    j = pd.concat([j, jobs((J4, 0, 0, ""), ("SYNTH-JOB-005", 0, 0, None))], ignore_index=True)
    first = assess_city_integrity(j, c, coverage=COV)
    shuffled = assess_city_integrity(j.iloc[::-1], c.iloc[[4, 2, 0, 3, 1]], coverage=COV)
    assert first == shuffled
    assert first.scope_unassignable_job_sample == ((J4,), ("SYNTH-JOB-005",))
    assert completeness(j, c).blocking_reasons == completeness(j.iloc[::-1], c.iloc[::-1]).blocking_reasons


def test_samples_are_bounded_sorted_and_kept_out_of_repr():
    n = CITY_INTEGRITY_SAMPLE_LIMIT + 5
    keys = [f"SYNTH-JOB-{i:03d}" for i in range(n, 0, -1)]
    j = jobs(*[(k, 1, 1, CAL) for k in keys])
    c = cars(*[(k, L1, VAN) for k in keys])
    report = assess_city_integrity(j, c, coverage=COV)
    assert report.scope_mismatch_detail_row_count == n
    assert len(report.scope_mismatch_sample) == CITY_INTEGRITY_SAMPLE_LIMIT
    assert [s.parent_key for s in report.scope_mismatch_sample] == sorted(s.parent_key
                                                                         for s in report.scope_mismatch_sample)
    assert report.scope_mismatch_sample[0].parent_key == ("SYNTH-JOB-001",)
    assert "SYNTH" not in repr(report)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.scope_mismatch_detail_row_count = 0  # type: ignore[misc]
    with pytest.raises(ValueError):                                      # counts and samples must agree
        dataclasses.replace(report, scope_mismatch_detail_row_count=0, scope_mismatch_job_count=0)


def test_inputs_are_not_modified():
    j, c = cross_city()
    before = (j.copy(deep=True), c.copy(deep=True))
    assess_city_integrity(j, c, coverage=COV)
    assess_expected_location_streams(j, c, coverage=COV)
    reconcile(j, c)
    pd.testing.assert_frame_equal(j, before[0])
    pd.testing.assert_frame_equal(c, before[1])


# ---------------------------------------------------- Vancouver policy untouched


def test_city_agreement_does_not_decide_vancouver_identity():
    # Both Vancouver labels sharing their job's city says nothing about whether they
    # are one pickup location: the policy stays unresolved and blocks pricing.
    j, c = healthy()
    assert assess_city_integrity(j, c, coverage=COV).is_valid
    assert VANCOUVER_LOCATION_POLICY.state is LocationPolicyState.UNRESOLVED
    policy = assess_location_policy()
    assert not policy.locations_are_aliases and not policy.locations_comparable_independently
    readiness = assess_pricing_readiness(location_policy=policy, **PROJECT_GATES(completeness(j, c)))
    assert readiness.blocking_reasons == (PB.LOCATION_POLICY_UNRESOLVED,)


def test_blocker_values_name_no_source_columns_and_exports():
    from conftest import contract_columns
    from ql2_sixt_canada_analysis.schemas import DatasetKey

    columns = {col for key in DatasetKey for col in contract_columns(key)}
    for enum in (CIB, CB, PB, EB, JB, S):
        assert not any(col in member.value for member in enum for col in columns), enum
    for name in ("assess_city_integrity", "validate_city_integrity", "CityIntegrityReport", "CityIntegrityBlocker",
                 "CityIntegrityError", "unassignable_scope_mask", "ScopeMismatchSample", "CITY_INTEGRITY_SAMPLE_LIMIT"):
        assert name in ql2_sixt_canada_analysis.__all__
    assert isinstance(assess_city_integrity(*healthy(), coverage=COV), CityIntegrityReport)
