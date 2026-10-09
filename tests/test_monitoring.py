"""Monitoring and actionability (data-plan Section 6): catalog, evaluations, pipeline binding and presentation.

Every observation is fabricated: ``SYNTH-*`` jobs and products, synthetic 2030
capture periods, prices, counts and policies. The only committed values read
are approved configuration (stream keys, time zones, location roles, the
temporal authority and the canonical-offer policy) through the shared
synthetic pipeline world. The proprietary raw files are never read.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import os
import re
import socket
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest
from conftest import contract_columns, link, write_synthetic_csv
from test_price_change_analysis import merge, path
from test_price_change_events import (
    AUTHORITY,
    CAL_AIR,
    CAL_DOWN,
    CONTRACT,
    STREAMS,
    TOR_AIR,
    TOR_DOWN,
    VAN_DOWN,
    VAN_THUR,
    pipeline_result,
    synthetic_world,
)
from test_reconciliation import MISSING, _cars, _jobs
from test_visible_assortment import synthetic_floor_rule

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis import monitoring as mon
from ql2_sixt_canada_analysis.assortment_contract import AnomalyPolicyStatus as APS, UnusualDropPolicy
from ql2_sixt_canada_analysis.authority_decisions import load_current_decision_record
from ql2_sixt_canada_analysis.collection_schedule import JobAssignmentFailure
from ql2_sixt_canada_analysis.comparison import LocationStreamComparisonStatus as LCS
from ql2_sixt_canada_analysis.coverage import assess_expected_location_coverage
from ql2_sixt_canada_analysis.monitoring import (
    DATA_PLAN_RECONCILIATION,
    DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY,
    MONITORING_CONTROLS,
    MONITORING_TABLE_COLUMNS,
    PRODUCTION_CALIBRATION_REQUIREMENTS,
    SEVERITY_DESCRIPTIONS,
    STATUS_DESCRIPTIONS,
    CalibrationStatus,
    ControlEvaluation,
    ControlStatus as ST,
    EvidenceGap as G,
    MonitoringBlocker,
    MonitoringContractError,
    MonitoringControl,
    MonitoringControlId as C,
    MonitoringEvidence,
    MonitoringFinding as F,
    MonitoringNote as N,
    MonitoringReport,
    MonitoringReportStatus,
    MonitoringResult,
    Severity,
    SynchronizedMovementPolicy,
    blocked_monitoring_report,
    evaluate_monitoring_controls,
    monitoring_control,
    monitoring_control_table,
    monitoring_from_pipeline,
    monitoring_summary_lines,
    run_monitoring,
    severity_scale_table,
    status_legend_table,
    validate_monitoring_table,
)
from ql2_sixt_canada_analysis.pricing_pipeline import bind_pipeline_evidence
from ql2_sixt_canada_analysis.pricing_population import frame_binding
from ql2_sixt_canada_analysis.readiness import LocationPolicyReport, PricingBlocker
from ql2_sixt_canada_analysis.reconciliation import assess_job_detail_reconciliation
from ql2_sixt_canada_analysis.schemas import (
    ANALYSIS_TEMPORAL_RECONCILIATION,
    JOB_DETAIL_RELATIONSHIP,
    DatasetKey,
    TemporalKind,
)
from ql2_sixt_canada_analysis.stability import VehicleStabilityReport, VehicleStabilityStatus
from ql2_sixt_canada_analysis.temporal import RuleStatus, TemporalFieldReport, TemporalReconciliationReport, \
    TemporalRuleReport
from ql2_sixt_canada_analysis.temporal_authority import temporal_authority_from_record

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "ql2_sixt_canada_analysis" / "monitoring.py"
ORDER = (C.MISSING_EXPECTED_LOCATIONS, C.JOB_DETAIL_COUNT_MISMATCHES, C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS,
         C.UNEXPECTED_TIMESTAMP_OFFSETS, C.INVALID_OR_CHANGING_PRODUCT_ATTRIBUTES, C.ABRUPT_ASSORTMENT_CHANGES,
         C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS, C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES)
J1, J2, J3 = "SYNTH-JOB-001", "SYNTH-JOB-002", "SYNTH-JOB-003"

#: Toronto Downtown: two products rise together at hour 1 (a synchronized increase); Vancouver Downtown and its
#: Thurlow alias carry the same two products, which fall together at hour 1 (one canonical synchronized decrease).
SYNC = merge(path(TOR_DOWN, (50.0, 55.0, 55.0)), path(TOR_DOWN, (80.0, 88.0, 88.0), "SYNTH Car B"),
             path(VAN_DOWN, (60.0, 54.0, 54.0)), path(VAN_DOWN, (70.0, 63.0, 63.0), "SYNTH Car B"),
             path(VAN_THUR, (60.0, 54.0, 54.0)), path(VAN_THUR, (70.0, 63.0, 63.0), "SYNTH Car B"))
#: A product that rises only in the final interval (right censored) and a product dropped at the final capture.
FINAL = merge(path(TOR_DOWN, (50.0, 50.0, 55.0)), path(TOR_DOWN, (80.0, 88.0, 99.0), "SYNTH Car B"),
              path(TOR_AIR, (40.0, 40.0, None), "SYNTH Car C"))
#: Nothing changes: only the constant filler and one steady product.
QUIET = merge(*(path(s, (30.0, 30.0, 30.0), "SYNTH Car Q") for s in STREAMS))


# ============================================================================ synthetic evidence


def temporal_report(**overrides) -> TemporalReconciliationReport:  # type: ignore[no-untyped-def]
    """A fabricated temporal report (counts only); keyword overrides replace whole parts."""
    field_kw = overrides.pop("field", {})
    fields = (TemporalFieldReport(**{"dataset": DatasetKey.JOBS, "column": "finished_at",
                                     "kind": TemporalKind.TIMESTAMP, "required": True, "row_count": 3,
                                     "valid_count": 3, "missing_count": 0, "invalid_count": 0,
                                     "unresolved_count": 0, "resolvable": True, **field_kw}),)
    base = dict(field_reports=fields, ordering=TemporalRuleReport("ordering", RuleStatus.CONFIGURED, 3, 3, 0, 0),
                date_checks=(TemporalRuleReport("reporting_day", RuleStatus.CONFIGURED, 3, 3, 0, 0),),
                replications=(TemporalRuleReport("replication", RuleStatus.CONFIGURED, 3, 3, 0, 0),),
                parent_row_count=3, detail_row_count=3, unlinked_detail_row_count=0)
    return TemporalReconciliationReport(**{**base, **overrides})


def stability_report(**overrides) -> VehicleStabilityReport:  # type: ignore[no-untyped-def]
    base = dict(status=VehicleStabilityStatus.PASSED, observations_assessed=6, distinct_entities=2,
                complete_identity_entities=2, incomplete_identity_entities=0, incomplete_identity_observations=0,
                temporally_unassessable_entities=0, sufficient_history_entities=2, insufficient_history_entities=0,
                fully_stable_entities=2, value_unstable_only_entities=0, presence_unstable_only_entities=0,
                value_and_presence_unstable_entities=0, entities_with_value_conflicts=0,
                entities_with_presence_instability=0, same_capture_conflict_entities=0, attributes=())
    return VehicleStabilityReport(**{**base, **overrides})


def reconciliation(keys=(J1, J2), declared=(1, 1), details=(J1, J2)):  # type: ignore[no-untyped-def]
    return assess_job_detail_reconciliation(_jobs(list(keys), list(declared)), _cars(list(details)),
                                            JOB_DETAIL_RELATIONSHIP)


TEMPORAL_AUTHORITY = temporal_authority_from_record(load_current_decision_record(), ANALYSIS_TEMPORAL_RECONCILIATION,
                                                    CONTRACT)
LINKAGE = link(_jobs([J1, J2], [1, 1]), _cars([J1, J2])).report


def full_run(world: dict, **replace):  # type: ignore[no-untyped-def]
    """The synthetic pipeline result plus every foundational report the monitoring layer reads."""
    run = pipeline_result(world)
    extra = dict(coverage=assess_expected_location_coverage(world["cars"], CONTRACT.coverage),
                 reconciliation=reconciliation(), job_linkage=LINKAGE, temporal_authority=TEMPORAL_AUTHORITY,
                 temporal=temporal_report(), vehicle_stability=stability_report())
    # Built as one run: its evidence manifest is captured after every report is in place.
    return bind_pipeline_evidence(dataclasses.replace(run, **{**extra, **replace}))


def evaluate(world: dict, *, movement_policy=DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY, assortment_policy=None,  # type: ignore[no-untyped-def]
             **replace) -> MonitoringResult:
    kwargs = {"assortment_policy": assortment_policy} if assortment_policy is not None else {}
    return monitoring_from_pipeline(full_run(world, **replace), movement_policy=movement_policy, **kwargs)


def evidence_for(world: dict, **replace) -> MonitoringEvidence:  # type: ignore[no-untyped-def]
    result = evaluate(world)
    assert result.evidence is not None
    return dataclasses.replace(result.evidence, **replace)


def one(evidence: MonitoringEvidence, control: C, **kwargs) -> ControlEvaluation:  # type: ignore[no-untyped-def]
    return evaluate_monitoring_controls(evidence, **kwargs).evaluation(control)


def approved_movement(**params) -> SynchronizedMovementPolicy:  # type: ignore[no-untyped-def]
    base = dict(minimum_changed_offers=2, minimum_changed_share=Fraction(1, 2),
                minimum_median_abs_change_percent=Fraction(5), minimum_locations=1)
    return SynchronizedMovementPolicy(APS.APPROVED, "SYNTH-RECORD", "SYNTH-REFERENCE", **{**base, **params})


@pytest.fixture(scope="module")
def sync_world():  # type: ignore[no-untyped-def]
    return synthetic_world(products=SYNC)


@pytest.fixture(scope="module")
def final_world():  # type: ignore[no-untyped-def]
    return synthetic_world(products=FINAL)


@pytest.fixture(scope="module")
def quiet_world():  # type: ignore[no-untyped-def]
    return synthetic_world(products=QUIET)


# ============================================================================ catalog


def test_the_catalog_holds_exactly_the_eight_required_controls_in_data_plan_order() -> None:
    assert tuple(c.control_id for c in MONITORING_CONTROLS) == ORDER == tuple(C)
    assert [c.name for c in MONITORING_CONTROLS] == [
        "Missing expected locations", "Job/detail count mismatches", "Duplicate or aliased location feeds",
        "Unexpected timestamp offsets", "Invalid or changing product attributes", "Abrupt assortment changes",
        "Large synchronized price movements", "Unconfirmed anomalies at the end of a collection window"]
    assert all(monitoring_control(c) is MONITORING_CONTROLS[i] for i, c in enumerate(ORDER))
    assert monitoring_control("abrupt_assortment_changes").control_id is C.ABRUPT_ASSORTMENT_CHANGES


def test_every_definition_is_complete_and_typed() -> None:
    for control in MONITORING_CONTROLS:
        assert isinstance(control.severity, Severity) and isinstance(control.calibration_status, CalibrationStatus)
        for text in (control.name, control.condition, control.business_impact, control.recommended_response,
                     control.evidence_source):
            assert isinstance(text, str) and len(text.split()) >= 3
        assert not re.search(r"\d", control.condition + control.business_impact + control.recommended_response), \
            "no unsupported numeric threshold in a definition"
    severities = {c.control_id: c.severity for c in MONITORING_CONTROLS}
    assert severities[C.MISSING_EXPECTED_LOCATIONS] is Severity.CRITICAL
    assert severities[C.ABRUPT_ASSORTMENT_CHANGES] is severities[C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES] \
        is Severity.MEDIUM
    assert all(severities[c] is Severity.HIGH for c in ORDER[1:5] + (C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS,))
    candidates = {c.control_id for c in MONITORING_CONTROLS
                  if c.calibration_status is CalibrationStatus.CANDIDATE_POLICY_UNAPPROVED}
    assert candidates == {C.ABRUPT_ASSORTMENT_CHANGES, C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS}
    assert set(SEVERITY_DESCRIPTIONS) == set(Severity) and set(STATUS_DESCRIPTIONS) == set(ST)
    assert set(DATA_PLAN_RECONCILIATION) == set(C) and len(PRODUCTION_CALIBRATION_REQUIREMENTS) >= 4
    assert "Vancouver Downtown and Thurlow" in monitoring_control(C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS) \
        .recommended_response
    assert "never establish an alias" in monitoring_control(C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS).condition


def test_severity_and_status_legends_cover_every_member_without_confidence_language() -> None:
    scale, legend = severity_scale_table(), status_legend_table()
    assert scale["severity"].tolist() == [s.value for s in Severity]
    assert legend["evaluation_status"].tolist() == [s.value for s in ST]
    assert not re.search(r"confiden|significan|p-value|probab", " ".join(scale.to_numpy().ravel()), re.I)
    assert [s.rank for s in Severity] == [0, 1, 2]


def test_definitions_policies_evaluations_and_reports_are_immutable() -> None:
    control = MONITORING_CONTROLS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        control.severity = Severity.MEDIUM  # type: ignore[misc]
    with pytest.raises(TypeError):
        mon.DATA_PLAN_RECONCILIATION[C.MISSING_EXPECTED_LOCATIONS] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        mon.SEVERITY_DESCRIPTIONS[Severity.HIGH] = ("x", "y")  # type: ignore[index]
    report = blocked_monitoring_report([MonitoringBlocker.PIPELINE_EVIDENCE_UNAVAILABLE])
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.status = MonitoringReportStatus.EVALUATED  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.evaluations[0].status = ST.PASSED  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY.minimum_locations = 1  # type: ignore[misc]
    assert isinstance(MONITORING_CONTROLS, tuple) and isinstance(report.evaluations, tuple)


def test_malformed_definitions_are_refused() -> None:
    base = dataclasses.asdict(MONITORING_CONTROLS[0])
    with pytest.raises(MonitoringContractError):
        MonitoringControl(**{**base, "condition": "  "})
    with pytest.raises(MonitoringContractError):
        MonitoringControl(**{**base, "severity": "critical"})
    with pytest.raises(MonitoringContractError):
        MonitoringControl(**{**base, "calibration_status": "calibrated"})


# ============================================================================ evaluation status contract


@pytest.mark.parametrize("kwargs", [
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.PASSED, findings=(F.EXPECTED_LOCATION_MISSING,)),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.PASSED, unavailable_evidence=(G.RECONCILIATION_UNAVAILABLE,)),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.TRIGGERED),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.NOT_ASSESSABLE),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.NOT_ASSESSABLE, findings=(F.EXPECTED_LOCATION_MISSING,),
         unavailable_evidence=(G.LOCATION_COVERAGE_UNAVAILABLE,)),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.CANDIDATE_ONLY),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.CONFIRMATION_REQUIRED,
         findings=(F.EXPECTED_LOCATION_MISSING,)),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.TRIGGERED, findings=(F.EXPECTED_LOCATION_MISSING,),
         effective_severity=Severity.MEDIUM),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.PASSED, policy_status=APS.UNAVAILABLE),
    dict(control_id=C.ABRUPT_ASSORTMENT_CHANGES, status=ST.PASSED, policy_status=APS.UNAVAILABLE),
    dict(control_id=C.ABRUPT_ASSORTMENT_CHANGES, status=ST.TRIGGERED, policy_status=APS.PROPOSED,
         findings=(F.OBSERVED_DROP_REVIEW_CANDIDATE,)),
    dict(control_id=C.ABRUPT_ASSORTMENT_CHANGES, status=ST.CANDIDATE_ONLY, policy_status=APS.APPROVED),
    dict(control_id=C.ABRUPT_ASSORTMENT_CHANGES, status=ST.CANDIDATE_ONLY),
    dict(control_id=C.ABRUPT_ASSORTMENT_CHANGES, status=ST.CANDIDATE_ONLY, policy_status=APS.UNAVAILABLE,
         effective_severity=Severity.HIGH),
    dict(control_id=C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES, status=ST.TRIGGERED,
         findings=(F.RIGHT_CENSORED_PRICE_CHANGE,)),
    dict(control_id=C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES, status=ST.CONFIRMATION_REQUIRED),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.TRIGGERED, findings=("expected_location_missing",)),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status=ST.TRIGGERED,
         findings=(F.EXPECTED_LOCATION_MISSING, F.EXPECTED_LOCATION_MISSING)),
    dict(control_id="missing_expected_locations", status=ST.PASSED),
    dict(control_id=C.MISSING_EXPECTED_LOCATIONS, status="passed"),
])
def test_an_evaluation_inconsistent_with_its_single_status_is_refused(kwargs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(MonitoringContractError):
        ControlEvaluation(**kwargs)


def test_consistent_evaluations_have_exactly_one_typed_status() -> None:
    passed = ControlEvaluation(C.MISSING_EXPECTED_LOCATIONS, ST.PASSED)
    assert passed.effective_severity is Severity.CRITICAL and passed.control is MONITORING_CONTROLS[0]
    escalated = ControlEvaluation(C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES, ST.CONFIRMATION_REQUIRED,
                                  findings=(F.RIGHT_CENSORED_PRICE_CHANGE,), effective_severity=Severity.HIGH)
    assert escalated.effective_severity is Severity.HIGH
    candidate = ControlEvaluation(C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS, ST.CANDIDATE_ONLY,
                                  policy_status=APS.UNAVAILABLE)
    assert candidate.status is ST.CANDIDATE_ONLY and isinstance(candidate.status, ST)


def test_report_contract_requires_eight_ordered_evaluations_and_consistent_status() -> None:
    blocked = blocked_monitoring_report([MonitoringBlocker.EVIDENCE_BINDING_MISMATCH], ["pricing_not_ready"])
    assert blocked.blocked and blocked.upstream_blockers == ("pricing_not_ready",)
    assert all(e.status is ST.NOT_ASSESSABLE for e in blocked.evaluations)
    with pytest.raises(MonitoringContractError):
        MonitoringReport(MonitoringReportStatus.BLOCKED, blocked.evaluations[:7],
                         blockers=(MonitoringBlocker.EVIDENCE_BINDING_MISMATCH,))
    with pytest.raises(MonitoringContractError):
        MonitoringReport(MonitoringReportStatus.BLOCKED, tuple(reversed(blocked.evaluations)),
                         blockers=(MonitoringBlocker.EVIDENCE_BINDING_MISMATCH,))
    with pytest.raises(MonitoringContractError):
        MonitoringReport(MonitoringReportStatus.BLOCKED, blocked.evaluations)            # no blocker
    with pytest.raises(MonitoringContractError):
        MonitoringReport(MonitoringReportStatus.EVALUATED, blocked.evaluations)          # not assessable controls
    with pytest.raises(MonitoringContractError):
        MonitoringReport(MonitoringReportStatus.PARTIALLY_EVALUATED, blocked.evaluations,
                         upstream_blockers=(1,))                                          # type: ignore[arg-type]
    with pytest.raises(MonitoringContractError):
        MonitoringReport(MonitoringReportStatus.BLOCKED, blocked.evaluations,
                         blockers=(MonitoringBlocker.EVIDENCE_BINDING_MISMATCH,),
                         assortment_policy_status=APS.APPROVED)                            # policy disagrees


# ============================================================================ 1. missing expected locations


def test_missing_expected_location_is_detected_from_the_configured_universe(quiet_world) -> None:  # type: ignore[no-untyped-def]
    cars = quiet_world["cars"]
    missing = cars[(cars["city"] != TOR_AIR[0]) | (cars["location"] != TOR_AIR[1])]
    coverage = assess_expected_location_coverage(missing, CONTRACT.coverage)
    ev = MonitoringEvidence(contract=CONTRACT, coverage=coverage, scheduled=quiet_world["scheduled"])
    result = one(ev, C.MISSING_EXPECTED_LOCATIONS)
    assert result.status is ST.TRIGGERED and F.EXPECTED_LOCATION_MISSING in result.findings
    # The observed locations never define the expected universe: a contract built from them is a mismatch.
    observed = tuple(dict.fromkeys(zip(missing["city"], missing["location"])))
    narrowed = dataclasses.replace(CONTRACT.coverage, expected_locations=observed)
    report = assess_expected_location_coverage(missing, narrowed)
    assert report.is_valid                                           # it would "pass" against the observed set
    evaluation = one(MonitoringEvidence(contract=CONTRACT, coverage=report, scheduled=quiet_world["scheduled"]),
                     C.MISSING_EXPECTED_LOCATIONS)
    assert evaluation.status is ST.TRIGGERED and F.COVERAGE_CONTRACT_MISMATCH in evaluation.findings


def test_a_repeated_location_cannot_compensate_for_a_missing_expected_location(quiet_world) -> None:  # type: ignore[no-untyped-def]
    cars = quiet_world["cars"]
    keep = cars[(cars["city"] != TOR_AIR[0]) | (cars["location"] != TOR_AIR[1])]
    repeated = keep[(keep["city"] == TOR_DOWN[0]) & (keep["location"] == TOR_DOWN[1])]
    padded = pd.concat([keep, repeated.iloc[: len(cars) - len(keep)]], ignore_index=True)
    assert len(padded) == len(cars)                                   # same row count, one location repeated
    coverage = assess_expected_location_coverage(padded, CONTRACT.coverage)
    evaluation = one(MonitoringEvidence(contract=CONTRACT, coverage=coverage, scheduled=quiet_world["scheduled"]),
                     C.MISSING_EXPECTED_LOCATIONS)
    assert evaluation.status is ST.TRIGGERED and evaluation.findings == (F.EXPECTED_LOCATION_MISSING,)


def test_an_unexcused_missing_stream_period_triggers_even_when_the_location_appears_elsewhere() -> None:
    world = synthetic_world(products=QUIET, drop_streams={(TOR_AIR, 1)})
    assert not world["scheduled"].is_valid
    ev = MonitoringEvidence(contract=CONTRACT, coverage=assess_expected_location_coverage(world["cars"],
                                                                                          CONTRACT.coverage),
                            scheduled=world["scheduled"])
    evaluation = one(ev, C.MISSING_EXPECTED_LOCATIONS)
    assert evaluation.status is ST.TRIGGERED and evaluation.findings == (F.UNEXCUSED_MISSING_STREAM_PERIOD,)


@pytest.mark.parametrize("world_kwargs, note", [
    (dict(excluded=("calgary", 1), drop_streams={(CAL_DOWN, 1)}), N.GOVERNED_EXCLUSION_APPLIED),
    (dict(excused=("toronto", 1), absent_jobs={("toronto", 1)}), N.GOVERNED_EXCEPTION_EXCUSED),
])
def test_a_governed_exclusion_or_exception_is_not_an_unexplained_missing_stream(world_kwargs, note) -> None:  # type: ignore[no-untyped-def]
    world = synthetic_world(products=QUIET, **world_kwargs)
    assert world["scheduled"].is_valid
    evaluation = evaluate(world).report.evaluation(C.MISSING_EXPECTED_LOCATIONS)
    assert evaluation.status is ST.PASSED and note in evaluation.notes and not evaluation.findings


def test_missing_contract_coverage_or_schedule_is_not_assessable_never_passed(quiet_world) -> None:  # type: ignore[no-untyped-def]
    coverage = assess_expected_location_coverage(quiet_world["cars"], CONTRACT.coverage)
    for ev, gap in ((MonitoringEvidence(coverage=coverage, scheduled=quiet_world["scheduled"]),
                     G.EXPECTED_STREAM_CONTRACT_UNAVAILABLE),
                    (MonitoringEvidence(contract=CONTRACT, scheduled=quiet_world["scheduled"]),
                     G.LOCATION_COVERAGE_UNAVAILABLE),
                    (MonitoringEvidence(contract=CONTRACT, coverage=coverage), G.SCHEDULED_COVERAGE_UNAVAILABLE)):
        evaluation = one(ev, C.MISSING_EXPECTED_LOCATIONS)
        assert evaluation.status is ST.NOT_ASSESSABLE and evaluation.unavailable_evidence == (gap,)
    passed = one(MonitoringEvidence(contract=CONTRACT, coverage=coverage, scheduled=quiet_world["scheduled"]),
                 C.MISSING_EXPECTED_LOCATIONS)
    assert passed.status is ST.PASSED


# ============================================================================ 2. job/detail count mismatches


def test_offsetting_over_and_under_counts_are_judged_per_job_and_never_cancel() -> None:
    report = reconciliation(keys=(J1, J2), declared=(2, 1), details=(J1, J2, J2))
    assert report.net_discrepancy == 0                         # the aggregate would hide both defects
    evaluation = one(MonitoringEvidence(reconciliation=report, job_linkage=LINKAGE), C.JOB_DETAIL_COUNT_MISMATCHES)
    assert evaluation.status is ST.TRIGGERED
    assert {F.DETAIL_ROWS_BELOW_DECLARED_COUNT, F.DETAIL_ROWS_ABOVE_DECLARED_COUNT} <= set(evaluation.findings)


@pytest.mark.parametrize("declared, finding", [
    ((MISSING, 1), F.DECLARED_COUNT_MISSING),
    ((-1, 1), F.DECLARED_COUNT_INVALID),
    ((1.5, 1), F.DECLARED_COUNT_INVALID),
    (("SYNTH-NOT-A-COUNT", 1), F.DECLARED_COUNT_INVALID),
])
def test_missing_or_invalid_declared_counts_never_pass(declared, finding) -> None:  # type: ignore[no-untyped-def]
    evaluation = one(MonitoringEvidence(reconciliation=reconciliation(declared=declared), job_linkage=LINKAGE),
                     C.JOB_DETAIL_COUNT_MISMATCHES)
    assert evaluation.status is ST.TRIGGERED and finding in evaluation.findings


def test_orphan_details_trigger_and_reconciled_jobs_pass() -> None:
    orphan = one(MonitoringEvidence(reconciliation=reconciliation(details=(J1, J2, J3)), job_linkage=LINKAGE),
                 C.JOB_DETAIL_COUNT_MISMATCHES)
    assert orphan.status is ST.TRIGGERED and F.ORPHAN_DETAIL_ROWS in orphan.findings
    assert one(MonitoringEvidence(reconciliation=reconciliation(), job_linkage=LINKAGE),
               C.JOB_DETAIL_COUNT_MISMATCHES).status is ST.PASSED
    missing = one(MonitoringEvidence(job_linkage=LINKAGE), C.JOB_DETAIL_COUNT_MISMATCHES)
    assert missing.status is ST.NOT_ASSESSABLE and missing.unavailable_evidence == (G.RECONCILIATION_UNAVAILABLE,)
    unlinked = one(MonitoringEvidence(reconciliation=reconciliation()), C.JOB_DETAIL_COUNT_MISMATCHES)
    assert unlinked.status is ST.NOT_ASSESSABLE and unlinked.unavailable_evidence == (G.JOB_LINKAGE_UNAVAILABLE,)


def test_job_order_does_not_change_the_reconciliation_evaluation() -> None:
    forward = one(MonitoringEvidence(reconciliation=reconciliation((J1, J2, J3), (2, 1, 1), (J1, J2, J2, J3)),
                                     job_linkage=LINKAGE), C.JOB_DETAIL_COUNT_MISMATCHES)
    shuffled = one(MonitoringEvidence(reconciliation=reconciliation((J3, J2, J1), (1, 1, 2), (J2, J3, J1, J2)),
                                      job_linkage=LINKAGE), C.JOB_DETAIL_COUNT_MISMATCHES)
    assert forward == shuffled


# ============================================================================ 3. duplicate or aliased location feeds


def test_the_approved_vancouver_alias_passes_and_is_one_canonical_location(sync_world) -> None:  # type: ignore[no-untyped-def]
    result = evaluate(sync_world)
    evaluation = result.report.evaluation(C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)
    assert evaluation.status is ST.PASSED and N.APPROVED_ALIAS_CANONICALIZED in evaluation.notes
    analysis, assortment = result.evidence.price_changes, result.evidence.assortment
    locations = [s.canonical_location for s in analysis.report.locations]
    assert VAN_DOWN in locations and VAN_THUR not in locations and len(locations) == len(set(locations)) == 6
    assert VAN_THUR not in assortment.report.approved_locations
    vancouver = analysis.report.location(VAN_DOWN)
    assert vancouver.price_change_count == 2                    # the alias never doubles the two decreases


def test_behavioural_similarity_alone_never_establishes_an_alias(quiet_world) -> None:  # type: ignore[no-untyped-def]
    readiness = quiet_world["readiness"]
    for evidence in (LCS.LIKELY_DUPLICATE_STREAMS, LCS.COMPARISON_INCONCLUSIVE):
        unresolved = LocationPolicyReport(state=type(readiness.location_policy.state).UNRESOLVED, authority=None,
                                          behavioral_evidence=evidence, identity_evidence_conflict=False,
                                          canonicalization_applied=False, scope=readiness.location_policy.scope)
        ev = evidence_for(quiet_world, pricing=dataclasses.replace(readiness, location_policy=unresolved))
        evaluation = one(ev, C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)
        assert evaluation.status is ST.NOT_ASSESSABLE
        assert G.LOCATION_IDENTITY_UNRESOLVED in evaluation.unavailable_evidence
        assert N.BEHAVIORAL_SIMILARITY_IS_NOT_AUTHORITY in evaluation.notes
        assert N.APPROVED_ALIAS_CANONICALIZED not in evaluation.notes


def test_contradicted_identity_decision_and_unapplied_alias_trigger(quiet_world) -> None:  # type: ignore[no-untyped-def]
    readiness = quiet_world["readiness"]
    policy = readiness.location_policy
    conflict = dataclasses.replace(policy, behavioral_evidence=LCS.LOCATION_MAPPING_DEFECT,
                                   identity_evidence_conflict=True)
    evaluation = one(evidence_for(quiet_world, pricing=dataclasses.replace(readiness, location_policy=conflict)),
                     C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)
    assert evaluation.status is ST.TRIGGERED
    assert {F.IDENTITY_EVIDENCE_CONFLICT, F.LOCATION_MAPPING_DEFECT} <= set(evaluation.findings)
    unapplied = dataclasses.replace(policy, canonicalization_applied=False)
    evaluation = one(evidence_for(quiet_world, pricing=dataclasses.replace(readiness, location_policy=unapplied)),
                     C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)
    assert evaluation.status is ST.TRIGGERED and F.ALIAS_CANONICALIZATION_NOT_APPLIED in evaluation.findings
    mismatch = dataclasses.replace(readiness, blocking_reasons=(PricingBlocker.CANONICAL_OFFER_POLICY_MISMATCH,))
    evaluation = one(evidence_for(quiet_world, pricing=mismatch), C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)
    assert evaluation.status is ST.TRIGGERED and F.CANONICAL_OFFER_POLICY_MISMATCH in evaluation.findings


def test_unapproved_or_misspelled_location_feeds_trigger(quiet_world) -> None:  # type: ignore[no-untyped-def]
    cars = quiet_world["cars"]
    variant = cars[(cars["city"] == TOR_DOWN[0]) & (cars["location"] == TOR_DOWN[1])].head(1).assign(
        location=TOR_DOWN[1].upper())
    coverage = assess_expected_location_coverage(pd.concat([cars, variant], ignore_index=True), CONTRACT.coverage)
    evaluation = one(evidence_for(quiet_world, coverage=coverage), C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)
    assert evaluation.status is ST.TRIGGERED
    assert {F.UNAPPROVED_LOCATION_STREAM, F.SOURCE_SPELLING_VARIANT} <= set(evaluation.findings)


def test_missing_location_authority_or_offer_combination_is_not_assessable(quiet_world) -> None:  # type: ignore[no-untyped-def]
    no_authority = one(evidence_for(quiet_world, location_authority=None), C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)
    assert no_authority.status is ST.NOT_ASSESSABLE
    assert G.LOCATION_AUTHORITY_UNAVAILABLE in no_authority.unavailable_evidence
    no_offers = one(evidence_for(quiet_world, canonical_offers=None), C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)
    assert AUTHORITY.canonicalization_merges_streams
    assert no_offers.status is ST.NOT_ASSESSABLE
    assert no_offers.unavailable_evidence == (G.CANONICAL_OFFER_COMBINATION_UNAVAILABLE,)


# ============================================================================ 4. unexpected timestamp offsets


@pytest.mark.parametrize("zone", ["UTC", "Asia/Kolkata", "America/St_Johns", "Pacific/Kiritimati"])
def test_timestamp_evaluation_follows_city_zones_not_the_machine_zone(zone, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    reference = evaluate(synthetic_world(products=QUIET)).report.evaluation(C.UNEXPECTED_TIMESTAMP_OFFSETS)
    monkeypatch.setenv("TZ", zone)
    time.tzset()
    try:
        world = synthetic_world(products=QUIET)
        evaluation = evaluate(world).report.evaluation(C.UNEXPECTED_TIMESTAMP_OFFSETS)
        periods = [p.utc_text for s in world["scheduled"].schedule.schedules for p in s.periods]
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
    assert evaluation == reference and evaluation.status is ST.PASSED
    assert TEMPORAL_AUTHORITY.city_timezones is not None and not TEMPORAL_AUTHORITY.blocking_reasons
    assert periods == [p.utc_text for s in synthetic_world(products=QUIET)["scheduled"].schedule.schedules
                       for p in s.periods]


@pytest.mark.parametrize("temporal, finding", [
    (dict(field=dict(valid_count=2, invalid_count=1)), F.TIMESTAMP_PARSE_FAILURE),
    (dict(field=dict(unresolved_count=1, unknown_city_count=1)), F.TIMESTAMP_UNRESOLVED),
    (dict(replications=(TemporalRuleReport("replication", RuleStatus.CONFIGURED, 3, 2, 1, 0),)),
     F.PARENT_DETAIL_TIMESTAMP_MISMATCH),
    (dict(ordering=TemporalRuleReport("ordering", RuleStatus.CONFIGURED, 3, 2, 1, 0)),
     F.TIMESTAMP_ORDERING_VIOLATION),
    (dict(date_checks=(TemporalRuleReport("reporting_day", RuleStatus.CONFIGURED, 3, 2, 1, 0),)),
     F.REPORTING_DATE_MISMATCH),
    (dict(city_mismatch_detail_row_count=1), F.TIMESTAMP_CITY_ZONE_UNKNOWN),
])
def test_invalid_unresolved_or_inconsistent_timestamps_trigger(quiet_world, temporal, finding) -> None:  # type: ignore[no-untyped-def]
    evaluation = one(evidence_for(quiet_world, temporal=temporal_report(**temporal)), C.UNEXPECTED_TIMESTAMP_OFFSETS)
    assert evaluation.status is ST.TRIGGERED and finding in evaluation.findings


@pytest.mark.parametrize("failure, finding", [
    (JobAssignmentFailure.NONEXISTENT_LOCAL_TIME, F.FINISH_TIME_UNRESOLVABLE),
    (JobAssignmentFailure.AMBIGUOUS_LOCAL_TIME, F.FINISH_TIME_UNRESOLVABLE),
    (JobAssignmentFailure.INVALID_FINISHED_AT, F.FINISH_TIME_UNRESOLVABLE),
    (JobAssignmentFailure.OUTSIDE_SCHEDULE_WINDOW, F.FINISH_TIME_OFF_SCHEDULE),
    (JobAssignmentFailure.DUPLICATE_CITY_PERIOD, F.CAPTURE_PERIOD_COLLISION),
    (JobAssignmentFailure.DETAIL_COPY_MISMATCH, F.PARENT_DETAIL_TIMESTAMP_MISMATCH),
    (JobAssignmentFailure.UNKNOWN_CITY, F.TIMESTAMP_CITY_ZONE_UNKNOWN),
])
def test_schedule_assignment_failures_are_timestamp_findings(quiet_world, failure, finding) -> None:  # type: ignore[no-untyped-def]
    scheduled = dataclasses.replace(quiet_world["scheduled"], job_failures=((failure, 1),))
    evaluation = one(evidence_for(quiet_world, scheduled=scheduled), C.UNEXPECTED_TIMESTAMP_OFFSETS)
    assert evaluation.status is ST.TRIGGERED and evaluation.findings == (finding,)


def test_unavailable_temporal_authority_or_rules_are_not_assessable(quiet_world) -> None:  # type: ignore[no-untyped-def]
    no_authority = one(evidence_for(quiet_world, temporal_authority=None), C.UNEXPECTED_TIMESTAMP_OFFSETS)
    assert no_authority.status is ST.NOT_ASSESSABLE
    assert no_authority.unavailable_evidence == (G.TEMPORAL_AUTHORITY_UNAVAILABLE,)
    unavailable = temporal_report(ordering=TemporalRuleReport("ordering", RuleStatus.UNAVAILABLE, 3, 0, 0, 3))
    rule = one(evidence_for(quiet_world, temporal=unavailable), C.UNEXPECTED_TIMESTAMP_OFFSETS)
    assert rule.status is ST.NOT_ASSESSABLE and G.TEMPORAL_RULE_UNAVAILABLE in rule.unavailable_evidence
    unlinked = one(evidence_for(quiet_world, temporal=temporal_report(unlinked_detail_row_count=1)),
                   C.UNEXPECTED_TIMESTAMP_OFFSETS)
    assert unlinked.status is ST.NOT_ASSESSABLE and G.TIMESTAMP_ROWS_UNASSESSABLE in unlinked.unavailable_evidence


# ============================================================================ 5. product attributes


@pytest.mark.parametrize("overrides, status, finding, gap", [
    (dict(status=VehicleStabilityStatus.VIOLATIONS, entities_with_value_conflicts=1, fully_stable_entities=1,
          value_unstable_only_entities=1), ST.TRIGGERED, F.ATTRIBUTE_VALUE_CONFLICT, None),
    (dict(status=VehicleStabilityStatus.VIOLATIONS, same_capture_conflict_entities=1), ST.TRIGGERED,
     F.SAME_CAPTURE_ATTRIBUTE_CONFLICT, None),
    (dict(status=VehicleStabilityStatus.VIOLATIONS, incomplete_identity_entities=1,
          incomplete_identity_observations=1, distinct_entities=3), ST.TRIGGERED,
     F.REQUIRED_IDENTITY_VALUE_MISSING, None),
    (dict(status=VehicleStabilityStatus.PARTIALLY_ASSESSABLE, insufficient_history_entities=1,
          sufficient_history_entities=1, fully_stable_entities=1), ST.NOT_ASSESSABLE, None,
     G.PRODUCT_HISTORY_INSUFFICIENT),
    (dict(status=VehicleStabilityStatus.VIOLATIONS, temporally_unassessable_entities=1,
          sufficient_history_entities=1, fully_stable_entities=1), ST.NOT_ASSESSABLE, None,
     G.PRODUCT_HISTORY_TEMPORALLY_UNASSESSABLE),
    (dict(status=VehicleStabilityStatus.UNASSESSABLE, distinct_entities=0, complete_identity_entities=0,
          sufficient_history_entities=0, fully_stable_entities=0), ST.NOT_ASSESSABLE, None,
     G.PRODUCT_POPULATION_EMPTY),
])
def test_missing_values_insufficient_history_and_proven_conflicts_stay_distinct(overrides, status, finding, gap  # type: ignore[no-untyped-def]
                                                                                ) -> None:
    evaluation = one(MonitoringEvidence(vehicle_stability=stability_report(**overrides)),
                     C.INVALID_OR_CHANGING_PRODUCT_ATTRIBUTES)
    assert evaluation.status is status
    assert evaluation.findings == ((finding,) if finding else ())
    assert evaluation.unavailable_evidence == ((gap,) if gap else ())


def test_product_attributes_pass_only_on_a_passed_stability_report() -> None:
    assert one(MonitoringEvidence(vehicle_stability=stability_report()),
               C.INVALID_OR_CHANGING_PRODUCT_ATTRIBUTES).status is ST.PASSED
    assert one(MonitoringEvidence(), C.INVALID_OR_CHANGING_PRODUCT_ATTRIBUTES).unavailable_evidence == (
        G.VEHICLE_STABILITY_UNAVAILABLE,)


# ============================================================================ 6. abrupt assortment changes


def test_abrupt_assortment_changes_stay_candidate_only_without_an_approved_policy(final_world) -> None:  # type: ignore[no-untyped-def]
    result = evaluate(final_world)
    evaluation = result.report.evaluation(C.ABRUPT_ASSORTMENT_CHANGES)
    assert result.evidence.assortment.report.overall.drop_intervals > 0
    assert evaluation.status is ST.CANDIDATE_ONLY and evaluation.policy_status is APS.UNAVAILABLE
    assert evaluation.findings == (F.OBSERVED_DROP_REVIEW_CANDIDATE,)
    assert N.OPERATIONAL_THRESHOLD_NOT_APPROVED in evaluation.notes
    assert evaluation.effective_severity is Severity.MEDIUM
    assert result.evidence.assortment.report.unusual_drop_intervals is None


def test_no_observed_drop_is_still_candidate_only_and_never_a_pass(quiet_world) -> None:  # type: ignore[no-untyped-def]
    evaluation = evaluate(quiet_world).report.evaluation(C.ABRUPT_ASSORTMENT_CHANGES)
    assert evaluation.status is ST.CANDIDATE_ONLY and not evaluation.findings
    assert N.NO_REVIEW_CANDIDATES_OBSERVED in evaluation.notes


def test_a_proposed_unusual_drop_policy_never_raises_an_alert(final_world) -> None:  # type: ignore[no-untyped-def]
    proposed = UnusualDropPolicy(APS.PROPOSED, description="SYNTH proposal")
    evaluation = evaluate(final_world, assortment_policy=proposed).report.evaluation(C.ABRUPT_ASSORTMENT_CHANGES)
    assert evaluation.status is ST.CANDIDATE_ONLY and evaluation.policy_status is APS.PROPOSED


def test_an_explicitly_approved_unusual_drop_policy_is_evaluated_and_escalates_on_suspect_completeness() -> None:
    world = synthetic_world(products=FINAL, withheld={(TOR_AIR, 1)})   # an eligible empty capture
    approved = UnusualDropPolicy(APS.APPROVED, record_id="SYNTH-RECORD", reference="SYNTH-REFERENCE",
                                 rule=synthetic_floor_rule(1, 0.5))
    result = evaluate(world, assortment_policy=approved)
    evaluation = result.report.evaluation(C.ABRUPT_ASSORTMENT_CHANGES)
    assert result.report.assortment_policy_status is APS.APPROVED
    assert evaluation.status is ST.TRIGGERED and evaluation.findings == (F.APPROVED_UNUSUAL_DROP_RULE_MET,)
    assert N.COLLECTION_COMPLETENESS_SUSPECT in evaluation.notes and evaluation.effective_severity is Severity.HIGH
    strict = UnusualDropPolicy(APS.APPROVED, record_id="SYNTH-RECORD", reference="SYNTH-REFERENCE",
                               rule=synthetic_floor_rule(99, 1.0))
    below = evaluate(world, assortment_policy=strict).report.evaluation(C.ABRUPT_ASSORTMENT_CHANGES)
    assert below.status is ST.PASSED and N.OBSERVED_DROPS_BELOW_APPROVED_RULE in below.notes
    assert below.effective_severity is Severity.MEDIUM


# ============================================================================ 7. synchronized price movements


def test_synchronized_movement_is_descriptive_and_increases_and_decreases_never_cancel(sync_world) -> None:  # type: ignore[no-untyped-def]
    result = evaluate(sync_world)
    evaluation = result.report.evaluation(C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS)
    assert evaluation.status is ST.CANDIDATE_ONLY and evaluation.policy_status is APS.UNAVAILABLE
    assert evaluation.findings == (F.SYNCHRONIZED_INCREASE_REVIEW_CANDIDATE, F.SYNCHRONIZED_DECREASE_REVIEW_CANDIDATE)
    assert N.OPERATIONAL_THRESHOLD_NOT_APPROVED in evaluation.notes


def test_an_explicitly_approved_movement_policy_evaluates_each_direction_separately(sync_world) -> None:  # type: ignore[no-untyped-def]
    triggered = evaluate(sync_world, movement_policy=approved_movement()).report
    evaluation = triggered.evaluation(C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS)
    assert triggered.movement_policy_status is APS.APPROVED and evaluation.status is ST.TRIGGERED
    assert evaluation.findings == (F.APPROVED_SYNCHRONIZED_INCREASE_RULE_MET,
                                   F.APPROVED_SYNCHRONIZED_DECREASE_RULE_MET)
    # One increasing and one decreasing location never add up to two locations moving together.
    two = evaluate(sync_world, movement_policy=approved_movement(minimum_locations=2)).report
    below = two.evaluation(C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS)
    assert below.status is ST.PASSED and N.SYNCHRONIZED_MOVEMENTS_BELOW_APPROVED_RULE in below.notes
    for params in (dict(minimum_changed_offers=3), dict(minimum_changed_share=Fraction(1)),
                   dict(minimum_median_abs_change_percent=Fraction(11))):
        assert evaluate(sync_world, movement_policy=approved_movement(**params)).report.evaluation(
            C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS).status is ST.PASSED


def test_aliased_streams_cannot_duplicate_a_synchronized_event() -> None:
    vancouver_only = synthetic_world(products=merge(
        path(VAN_DOWN, (60.0, 54.0, 54.0)), path(VAN_DOWN, (70.0, 63.0, 63.0), "SYNTH Car B"),
        path(VAN_THUR, (60.0, 54.0, 54.0)), path(VAN_THUR, (70.0, 63.0, 63.0), "SYNTH Car B")))
    policy = approved_movement(minimum_locations=2)
    assert evaluate(vancouver_only, movement_policy=policy).report.evaluation(
        C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS).status is ST.PASSED          # Downtown and Thurlow are one location
    # Two distinct canonical locations of one city, same scheduled period: they do count as two.
    toronto = synthetic_world(products=merge(
        path(TOR_DOWN, (50.0, 45.0, 45.0)), path(TOR_DOWN, (80.0, 72.0, 72.0), "SYNTH Car B"),
        path(TOR_AIR, (50.0, 45.0, 45.0)), path(TOR_AIR, (80.0, 72.0, 72.0), "SYNTH Car B")))
    evaluation = evaluate(toronto, movement_policy=policy).report.evaluation(C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS)
    assert evaluation.status is ST.TRIGGERED and evaluation.findings == (F.APPROVED_SYNCHRONIZED_DECREASE_RULE_MET,)


@pytest.mark.parametrize("kwargs", [
    dict(status=APS.APPROVED, minimum_changed_offers=2, minimum_changed_share=Fraction(1, 2),
         minimum_median_abs_change_percent=Fraction(5), minimum_locations=1),             # no authority
    dict(status=APS.UNAVAILABLE, minimum_locations=1),                                    # parameters without approval
    dict(status=APS.PROPOSED, minimum_changed_share=Fraction(1, 2)),
    dict(status=APS.APPROVED, record_id="SYNTH", reference="SYNTH", minimum_changed_offers=2,
         minimum_changed_share=0.5, minimum_median_abs_change_percent=Fraction(5), minimum_locations=1),
    dict(status=APS.APPROVED, record_id="SYNTH", reference="SYNTH", minimum_changed_offers=0,
         minimum_changed_share=Fraction(1, 2), minimum_median_abs_change_percent=Fraction(5), minimum_locations=1),
    dict(status=APS.APPROVED, record_id="SYNTH", reference="SYNTH", minimum_changed_offers=2,
         minimum_changed_share=Fraction(3, 2), minimum_median_abs_change_percent=Fraction(5), minimum_locations=1),
    dict(status=APS.APPROVED, record_id="SYNTH", reference="SYNTH", minimum_changed_offers=True,
         minimum_changed_share=Fraction(1, 2), minimum_median_abs_change_percent=Fraction(5), minimum_locations=1),
    dict(status="approved"),
])
def test_movement_policies_require_explicit_approval_and_valid_parameters(kwargs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(MonitoringContractError):
        SynchronizedMovementPolicy(**kwargs)


def test_the_default_movement_policy_is_unavailable_and_has_no_threshold() -> None:
    p = DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY
    assert p.status is APS.UNAVAILABLE and not p.executable
    assert (p.minimum_changed_offers, p.minimum_changed_share, p.minimum_median_abs_change_percent,
            p.minimum_locations) == (None, None, None, None)
    assert mon.DEFAULT_UNUSUAL_DROP_POLICY.status is APS.UNAVAILABLE


@pytest.mark.parametrize("world_name", ["sync_world", "final_world", "quiet_world"])
def test_no_approved_threshold_never_yields_a_production_alert_or_a_false_pass(world_name, request) -> None:  # type: ignore[no-untyped-def]
    report = evaluate(request.getfixturevalue(world_name)).report
    for control in (C.ABRUPT_ASSORTMENT_CHANGES, C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS):
        evaluation = report.evaluation(control)
        assert evaluation.status is ST.CANDIDATE_ONLY and evaluation.policy_status is APS.UNAVAILABLE
        assert not any(f.value.startswith("approved_") for f in evaluation.findings)
    assert all(e.status is not ST.TRIGGERED for e in report.evaluations)


# ============================================================================ 8. end-of-window confirmation


def test_final_window_events_require_confirmation_and_are_never_called_persistent(final_world) -> None:  # type: ignore[no-untyped-def]
    result = evaluate(final_world)
    evaluation = result.report.evaluation(C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES)
    assert evaluation.status is ST.CONFIRMATION_REQUIRED
    assert evaluation.findings == (F.RIGHT_CENSORED_PRICE_CHANGE, F.RIGHT_CENSORED_ASSORTMENT_DROP)
    persistence = result.evidence.price_changes.persistence
    final = persistence[persistence["not_testable_reason"] == "right_censored_final_capture"]
    assert len(final) and set(final["persistence"]) == {"not_testable"}
    table = monitoring_control_table(result.report)
    row = table[table["control_id"] == C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES.value].iloc[0]
    assert row["evaluation_status"] == "confirmation_required" and "persistent" not in row["findings"]
    assert row["observation"].startswith("Right-censored")


def test_a_right_censored_synchronized_movement_escalates_the_confirmation_severity() -> None:
    world = synthetic_world(products=merge(path(TOR_DOWN, (50.0, 50.0, 55.0)),
                                           path(TOR_DOWN, (80.0, 80.0, 88.0), "SYNTH Car B")))
    evaluation = evaluate(world).report.evaluation(C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES)
    assert evaluation.status is ST.CONFIRMATION_REQUIRED and evaluation.effective_severity is Severity.HIGH
    assert N.RIGHT_CENSORED_SYNCHRONIZED_MOVEMENT in evaluation.notes


@pytest.mark.parametrize("world_kwargs, stream", [
    (dict(hours=4, excluded=("calgary", 2), drop_streams={(CAL_DOWN, 2)}), CAL_AIR),
    (dict(hours=4, excused=("toronto", 2), absent_jobs={("toronto", 2)}), TOR_DOWN),
])
def test_hard_breaks_and_missing_captures_are_never_bridged(world_kwargs, stream) -> None:  # type: ignore[no-untyped-def]
    products = merge(path(stream, (50.0, 55.0, 55.0, 55.0)), path(stream, (70.0, 70.0, 70.0, None), "SYNTH Car D"),
                     path(stream, (90.0, 60.0, 60.0, 60.0), "SYNTH Car E"))
    world = synthetic_world(products=products, **world_kwargs)
    result = evaluate(world)
    evaluation = result.report.evaluation(C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES)
    persistence = result.evidence.price_changes.persistence
    assert set(persistence["persistence"]) == {"not_testable"}
    assert "right_censored_final_capture" not in set(persistence["not_testable_reason"])
    assert evaluation.status is ST.PASSED and N.PERSISTENCE_NOT_TESTABLE_ACROSS_BREAK in evaluation.notes
    assert F.RIGHT_CENSORED_PRICE_CHANGE not in evaluation.findings


def test_an_observed_drop_before_a_break_is_not_bridged_or_confirmed() -> None:
    world = synthetic_world(hours=4, excused=("toronto", 2), absent_jobs={("toronto", 2)},
                            products=merge(path(TOR_DOWN, (50.0, None, None, None), "SYNTH Car D")))
    evaluation = evaluate(world).report.evaluation(C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES)
    assert N.DROP_FOLLOW_UP_BLOCKED_BY_BREAK in evaluation.notes
    assert F.RIGHT_CENSORED_ASSORTMENT_DROP not in evaluation.findings


# ============================================================================ blocked and partial evidence


def test_blocked_prerequisite_evidence_is_not_assessable_and_never_passes() -> None:
    report = evaluate_monitoring_controls(MonitoringEvidence())
    assert report.status is MonitoringReportStatus.PARTIALLY_EVALUATED and not report.blocked
    assert all(e.status is ST.NOT_ASSESSABLE for e in report.evaluations)
    assert not report.controls_with_status(ST.PASSED)


def test_a_pipeline_without_readiness_blocks_every_control(quiet_world) -> None:  # type: ignore[no-untyped-def]
    result = monitoring_from_pipeline(dataclasses.replace(full_run(quiet_world), pricing=None))
    assert result.report.blocked and result.evidence is None
    assert result.report.blockers == (MonitoringBlocker.PIPELINE_EVIDENCE_UNAVAILABLE,)
    assert all(e.status is ST.NOT_ASSESSABLE and e.unavailable_evidence == (G.MONITORING_EVIDENCE_BLOCKED,)
               for e in result.report.evaluations)
    table = monitoring_control_table(result.report)
    assert len(table) == 8 and set(table["evaluation_status"]) == {"not_assessable"}
    assert table["condition"].tolist() == [c.condition for c in MONITORING_CONTROLS]


def test_pricing_not_ready_keeps_structural_controls_and_makes_downstream_controls_not_assessable(quiet_world  # type: ignore[no-untyped-def]
                                                                                                  ) -> None:
    readiness = dataclasses.replace(quiet_world["readiness"],
                                    blocking_reasons=(PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE,))
    report = evaluate(quiet_world, pricing=readiness).report
    assert report.upstream_blockers == ("scheduled_coverage_incomplete",)
    assert report.evaluation(C.MISSING_EXPECTED_LOCATIONS).status is ST.PASSED
    for control in ORDER[5:]:
        evaluation = report.evaluation(control)
        assert evaluation.status is ST.NOT_ASSESSABLE and evaluation.unavailable_evidence
    assert report.status is MonitoringReportStatus.PARTIALLY_EVALUATED


def test_an_evidence_binding_mismatch_blocks_every_control(quiet_world, sync_world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    run = full_run(quiet_world)
    other = full_run(sync_world)
    mixed = dataclasses.replace(run, scheduled=other.scheduled)       # readiness was decided on another schedule
    assert monitoring_from_pipeline(mixed).report.blockers == (MonitoringBlocker.EVIDENCE_BINDING_MISMATCH,)
    from ql2_sixt_canada_analysis import price_change_analysis

    foreign = price_change_analysis.price_change_analysis_from_pipeline(other)
    monkeypatch.setattr(price_change_analysis, "price_change_analysis_from_pipeline", lambda r: foreign)
    assert monitoring_from_pipeline(run).report.blockers == (MonitoringBlocker.EVIDENCE_BINDING_MISMATCH,)


def test_a_downstream_contract_error_blocks_instead_of_passing(quiet_world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis import visible_assortment
    from ql2_sixt_canada_analysis.assortment_contract import AssortmentContractError

    def broken(run, policy=None):  # type: ignore[no-untyped-def]
        raise AssortmentContractError("SYNTH contract failure")

    monkeypatch.setattr(visible_assortment, "visible_assortment_from_pipeline", broken)
    report = monitoring_from_pipeline(full_run(quiet_world)).report
    assert report.blockers == (MonitoringBlocker.DOWNSTREAM_EVIDENCE_INVALID,)


def test_wrong_argument_types_are_refused(quiet_world) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TypeError):
        monitoring_from_pipeline(object())
    with pytest.raises(TypeError):
        evaluate_monitoring_controls(object())                                  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MonitoringEvidence(coverage=object())
    with pytest.raises(TypeError):
        monitoring_from_pipeline(full_run(quiet_world), movement_policy=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        run_monitoring(assortment_policy=object())                             # type: ignore[arg-type]


# ============================================================================ one pipeline run, binding, determinism


def test_run_monitoring_runs_the_pipeline_exactly_once_and_binds_every_evidence(sync_world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis import price_change_analysis, pricing_pipeline, visible_assortment

    run = full_run(sync_world)
    calls: list[object] = []
    seen: list[object] = []
    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline", lambda raw_dir=None: calls.append(raw_dir) or run)
    for module, name in ((price_change_analysis, "price_change_analysis_from_pipeline"),
                         (visible_assortment, "visible_assortment_from_pipeline")):
        original = getattr(module, name)
        monkeypatch.setattr(module, name, lambda r, *a, _f=original, **k: seen.append(r) or _f(r, *a, **k))
    result = run_monitoring("SYNTH-RAW-DIRECTORY")
    assert calls == ["SYNTH-RAW-DIRECTORY"] and len(seen) == 2 and all(r is run for r in seen)
    evidence = result.evidence
    binding = frame_binding(run.jobs, run.cars)
    assert evidence.price_changes.events.binding == evidence.assortment.binding == binding
    assert evidence.assortment.price_changes.binding == binding
    assert evidence.price_changes.location_authority is evidence.assortment.location_authority \
        is run.location_authority
    assert evidence.coverage is run.coverage and evidence.reconciliation is run.reconciliation
    assert evidence.scheduled is run.scheduled and evidence.pricing is run.pricing


def test_evaluation_is_deterministic_idempotent_and_never_mutates_inputs(final_world) -> None:  # type: ignore[no-untyped-def]
    run = full_run(final_world)
    jobs, cars = run.jobs.copy(deep=True), run.cars.copy(deep=True)
    first = monitoring_from_pipeline(run)
    analysis_table = first.evidence.price_changes.event_table.copy(deep=True)
    timeline = first.evidence.assortment.timeline.copy(deep=True)
    again = evaluate_monitoring_controls(first.evidence)
    second = monitoring_from_pipeline(run)
    assert first.report == second.report == again
    pd.testing.assert_frame_equal(monitoring_control_table(first.report), monitoring_control_table(again))
    assert monitoring_summary_lines(first.report) == monitoring_summary_lines(second.report)
    pd.testing.assert_frame_equal(run.jobs, jobs)
    pd.testing.assert_frame_equal(run.cars, cars)
    pd.testing.assert_frame_equal(first.evidence.price_changes.event_table, analysis_table)
    pd.testing.assert_frame_equal(first.evidence.assortment.timeline, timeline)


def test_row_order_of_the_source_frames_does_not_change_coverage_evaluation(quiet_world) -> None:  # type: ignore[no-untyped-def]
    cars = quiet_world["cars"]
    forward = assess_expected_location_coverage(cars, CONTRACT.coverage)
    shuffled = assess_expected_location_coverage(cars.sample(frac=1.0, random_state=7), CONTRACT.coverage)
    evaluations = [one(MonitoringEvidence(contract=CONTRACT, coverage=c, scheduled=quiet_world["scheduled"]),
                       control) for c in (forward, shuffled)
                   for control in (C.MISSING_EXPECTED_LOCATIONS, C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)]
    assert evaluations[:2] == evaluations[2:]


# ============================================================================ presentation and confidentiality

FORBIDDEN_COLUMNS = ("job_id", "car_name", "car_type", "price", "price_num", "price_per_day", "pickup_date",
                     "return_date", "city", "location", "scraped_at", "finished_at", "canonical_city",
                     "canonical_location", "change_cents", "source_location_labels", "path")


def test_the_public_table_is_sanitized_and_holds_one_row_per_control(final_world) -> None:  # type: ignore[no-untyped-def]
    report = evaluate(final_world).report
    table = monitoring_control_table(report)
    assert tuple(table.columns) == MONITORING_TABLE_COLUMNS and len(table) == 8
    assert table["control_id"].tolist() == [c.value for c in ORDER]
    for required in ("condition", "severity", "likely_business_impact", "recommended_response",
                     "evaluation_status", "calibration_status"):
        assert table[required].map(bool).all(), required
    assert not set(FORBIDDEN_COLUMNS) & set(table.columns)
    text = table.to_csv(index=False)
    assert "SYNTH" not in text and "$" not in text and "2030" not in text and not re.search(r"\d", text)
    for line in monitoring_summary_lines(report):
        assert "SYNTH" not in line and not re.search(r"\d", line)


@pytest.mark.parametrize("mutate", [
    lambda t: t.assign(job_id="SYNTH-JOB-001"),
    lambda t: t.drop(columns=["notes"]),
    lambda t: t.iloc[::-1].reset_index(drop=True),
    lambda t: t.iloc[:7],
    lambda t: t.assign(condition=t["condition"].str.replace("approved", "observed")),
    lambda t: t.assign(findings="SYNTH Car A"),
    lambda t: t.assign(evaluation_status="alerted"),
    lambda t: t.assign(notes="hard_breaks_not_bridged, hard_breaks_not_bridged"),
    lambda t: t.assign(observation=STATUS_DESCRIPTIONS[ST.PASSED]),
    lambda t: t.assign(severity="low"),
])
def test_the_table_validator_refuses_anything_but_the_sanitized_table(final_world, mutate) -> None:  # type: ignore[no-untyped-def]
    table = monitoring_control_table(evaluate(final_world).report)
    with pytest.raises(MonitoringContractError):
        validate_monitoring_table(mutate(table))
    with pytest.raises(MonitoringContractError):
        validate_monitoring_table(table.to_dict())


def test_reports_and_results_keep_evidence_out_of_their_representation(final_world) -> None:  # type: ignore[no-untyped-def]
    result = evaluate(final_world)
    text = repr(result) + repr(result.report) + repr(result.evidence)
    assert "SYNTH" not in text and "DataFrame" not in text and "2030" not in text
    with pytest.raises(MonitoringContractError):
        MonitoringResult(result.report)                                  # a completed report keeps its evidence


def test_the_core_module_has_no_printing_writing_plotting_environment_or_network_calls() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {"print", "open", "write_text", "write_bytes", "to_csv", "to_parquet", "to_json", "to_excel",
                 "savefig", "pyplot", "environ", "getenv", "putenv", "system", "subprocess", "socket", "urlopen",
                 "requests", "mkdir", "makedirs", "unlink", "remove", "rename", "chdir", "display"}
    assert not names & forbidden
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names} | {n.module.split(".")[0] for n in ast.walk(tree)
                                     if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"os", "sys", "subprocess", "socket", "urllib", "requests", "matplotlib", "shutil",
                           "tempfile", "logging"}


def test_evaluation_performs_no_io_and_changes_no_environment(final_world, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = full_run(final_world)

    def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("monitoring must not perform I/O")

    before_env, before_cwd = dict(os.environ), os.getcwd()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(builtins, "print", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    result = monitoring_from_pipeline(run)
    monitoring_control_table(result.report), monitoring_summary_lines(result.report)
    severity_scale_table(), status_legend_table()
    monkeypatch.undo()
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd and list(tmp_path.iterdir()) == []


def test_importing_monitoring_performs_no_io_or_pipeline_execution() -> None:
    code = ("import builtins, io, os, sys\n"
            f"ROOT = {str(ROOT)!r}\n"
            "real = builtins.open\n"
            "def guarded(file, mode='r', *a, **k):\n"
            "    path = os.path.abspath(os.fspath(file)) if isinstance(file, (str, bytes, os.PathLike)) else ''\n"
            "    if any(c in mode for c in 'wax+') or str(path).startswith(ROOT):\n"
            "        raise AssertionError('I/O during import')\n"
            "    return real(file, mode, *a, **k)\n"
            "builtins.open = io.open = guarded\n"
            "import ql2_sixt_canada_analysis.monitoring as m\n"
            "print('ql2_sixt_canada_analysis.pricing_pipeline' in sys.modules, 'matplotlib' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_public_package_exports_are_complete() -> None:
    public = {n for n in vars(mon) if not n.startswith("_") and getattr(getattr(mon, n), "__module__", None)
              == mon.__name__}
    assert public <= set(mon.__all__) and len(mon.__all__) == len(set(mon.__all__))
    for name in mon.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(mon, name)


def test_the_pipeline_result_retains_the_reports_monitoring_reads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis.coverage import LocationCoverageReport
    from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult, run_pricing_pipeline

    names = {f.name for f in dataclasses.fields(PricingPipelineResult)}
    assert {"coverage", "reconciliation"} <= names
    directory = tmp_path / "synthetic_raw"
    directory.mkdir()
    for key in DatasetKey:
        write_synthetic_csv(directory / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    run = run_pricing_pipeline(directory)
    assert isinstance(run.coverage, LocationCoverageReport)
    result = monitoring_from_pipeline(run)
    report = result.report
    assert not report.blocked and report.status is MonitoringReportStatus.PARTIALLY_EVALUATED
    missing = report.evaluation(C.MISSING_EXPECTED_LOCATIONS)
    assert missing.status is ST.TRIGGERED and F.EXPECTED_LOCATION_MISSING in missing.findings
    assert not report.controls_with_status(ST.PASSED) or C.MISSING_EXPECTED_LOCATIONS not in \
        report.controls_with_status(ST.PASSED)
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == sorted(
        f"synthetic_{k}.csv" for k in DatasetKey)


# ============================================================================ documentation


def test_readme_documents_section_six_and_its_data_plan_reconciliation() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    start = readme.index("## Monitoring and actionability")
    section = readme[start:]
    for control in MONITORING_CONTROLS:
        assert control.name in section
        assert DATA_PLAN_RECONCILIATION[control.control_id] in section
    for phrase in ("Condition", "Severity", "Likely business impact", "Recommended response", "contract-based",
                   "candidate-only", "confirmation_required", "Right-censored", "exactly once", "fail closed",
                   "No files", "Before production monitoring", "python -m pytest tests/test_monitoring.py",
                   "05_monitoring_actionability.ipynb", "roughly 90 hours", "not a scheduler"):
        assert phrase in section, phrase
    for requirement in PRODUCTION_CALIBRATION_REQUIREMENTS:
        assert requirement in section
    assert "## QL2 controls\n\n- Expected/location completeness" not in (ROOT / "README.md").read_text("utf-8")


def test_notebook_readme_lists_notebook_05_with_its_behaviour_and_limits() -> None:
    text = " ".join((ROOT / "notebooks" / "README.md").read_text(encoding="utf-8").split())
    assert "05_monitoring_actionability.ipynb" in text
    for phrase in ("run_monitoring", "exactly once", "candidate-only", "confirmation", "not_assessable",
                   "No files are written", "no alert"):
        assert phrase in text, phrase


# ============================================================================ exact policy boundaries

#: Every approved minimum is permissive except the one a test probes (fabricated policy values).
TINY = Fraction(1, 10**9)


def boundary_policy(**params) -> SynchronizedMovementPolicy:  # type: ignore[no-untyped-def]
    return approved_movement(**{"minimum_changed_offers": 2, "minimum_changed_share": TINY,
                                "minimum_median_abs_change_percent": TINY, "minimum_locations": 1, **params})


def synchronized_world(rising: tuple[tuple[float, float], ...], steady: int = 0, stream=TOR_DOWN):  # type: ignore[no-untyped-def]
    """``rising`` products move from the first to the second price at hour 1; ``steady`` products never move.

    The constant filler product is always present and unchanged, so the comparable denominator is
    ``len(rising) + steady + 1``.
    """
    parts = [path(stream, (a, b, b), f"SYNTH Car R{i}") for i, (a, b) in enumerate(rising)]
    parts += [path(stream, (20.0, 20.0, 20.0), f"SYNTH Car S{i}") for i in range(steady)]
    return synthetic_world(products=merge(*parts))


def movement_status(world: dict, policy: SynchronizedMovementPolicy) -> ControlEvaluation:  # type: ignore[no-untyped-def]
    return evaluate(world, movement_policy=policy).report.evaluation(C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS)


@pytest.mark.parametrize("steady, share", [(3, Fraction(1, 3)), (0, Fraction(2, 3))])
def test_an_exact_changed_share_meets_an_equal_inclusive_minimum(steady, share) -> None:  # type: ignore[no-untyped-def]
    world = synthetic_world(products=merge(*(
        [path(TOR_DOWN, (50.0, 55.0, 55.0), "SYNTH Car R0"), path(TOR_DOWN, (80.0, 88.0, 88.0), "SYNTH Car R1")]
        + [path(TOR_DOWN, (20.0, 20.0, 20.0), f"SYNTH Car S{i}") for i in range(steady)])))
    table = evaluate(world).evidence.price_changes.event_table
    row = table[(table["canonical_location"] == TOR_DOWN[1]) & (table["price_change_count"] == 2)].iloc[0]
    assert Fraction(row["price_change_count"], row["comparable"]) == share
    assert Fraction(row["changed_share_of_comparable"]) < share     # the float summary falls below the boundary
    at = movement_status(world, boundary_policy(minimum_changed_share=share))
    assert at.status is ST.TRIGGERED and at.findings == (F.APPROVED_SYNCHRONIZED_INCREASE_RULE_MET,)
    above = movement_status(world, boundary_policy(minimum_changed_share=share + TINY))
    assert above.status is ST.PASSED and N.SYNCHRONIZED_MOVEMENTS_BELOW_APPROVED_RULE in above.notes


@pytest.mark.parametrize("cents, median", [(1, Fraction(1, 3)), (2, Fraction(2, 3))])
def test_an_exact_median_percentage_meets_an_equal_inclusive_minimum(cents, median) -> None:  # type: ignore[no-untyped-def]
    after = 3.0 + cents / 100                                        # 100 * cents / 300 percent
    world = synchronized_world(((3.0, after), (3.0, after)))
    table = evaluate(world).evidence.price_changes.event_table
    row = table[(table["canonical_location"] == TOR_DOWN[1]) & (table["price_change_count"] == 2)].iloc[0]
    assert Fraction(row["median_abs_change_percent"]) != median     # the float summary is not the exact value
    at = movement_status(world, boundary_policy(minimum_median_abs_change_percent=median))
    assert at.status is ST.TRIGGERED and at.findings == (F.APPROVED_SYNCHRONIZED_INCREASE_RULE_MET,)
    above = movement_status(world, boundary_policy(minimum_median_abs_change_percent=median + TINY))
    assert above.status is ST.PASSED


def test_an_even_sized_set_uses_the_exact_rational_median() -> None:
    # Decreases of one, two, three and five cents from three dollars: one third, two thirds, one and five thirds
    # percent. The exact median of the middle pair is five sixths.
    world = synchronized_world(((3.0, 2.99), (3.0, 2.98), (3.0, 2.97), (3.0, 2.95)))
    expected = Fraction(5, 6)
    import statistics as stats
    assert stats.median([Fraction(1, 3), Fraction(2, 3), Fraction(1), Fraction(5, 3)]) == expected
    analysis = evaluate(world).evidence.price_changes
    exact = mon._exact_intervals(analysis)
    measures = [m for m in exact.values() if m.decrease == 4]
    assert len(measures) == 1 and measures[0].median_abs_change_percent == expected
    assert measures[0].increase == 0 and measures[0].changed_share == Fraction(4, 5)
    at = movement_status(world, boundary_policy(minimum_median_abs_change_percent=expected))
    assert at.status is ST.TRIGGERED and at.findings == (F.APPROVED_SYNCHRONIZED_DECREASE_RULE_MET,)
    assert movement_status(world, boundary_policy(minimum_median_abs_change_percent=expected + TINY)).status \
        is ST.PASSED


def _candidate(outcome: str, previous: object = None, change: object = None) -> dict:
    return {"outcome": outcome, "previous_price_cents": previous, "change_cents": change}


def test_exact_interval_measures_exclude_non_changes_and_invalid_denominators() -> None:
    rows = [_candidate("increase", 300, 1), _candidate("increase", 0, 5), _candidate("increase", None, 5),
            _candidate("unchanged", 300, 0), _candidate("appeared"), _candidate("disappeared", 300),
            _candidate("ambiguous")]
    measures = mon._exact_interval(rows)
    assert (measures.increase, measures.decrease, measures.comparable) == (3, 0, 4)
    assert measures.changed_share == Fraction(3, 4)
    assert measures.median_abs_change_percent == Fraction(1, 3)    # zero and missing denominators never enter
    only_invalid = mon._exact_interval([_candidate("decrease", 0, -5), _candidate("decrease", None, -5)])
    assert only_invalid.median_abs_change_percent is None
    no_comparable = mon._exact_interval([_candidate("appeared"), _candidate("ambiguous")])
    assert no_comparable.comparable == 0 and no_comparable.changed_share is None
    with pytest.raises(MonitoringContractError):
        mon._exact_interval([_candidate("increase", 300.0, 1)])     # an approximate cent value is refused
    with pytest.raises(MonitoringContractError):
        mon._exact_interval([_candidate("increase", 300, None)])


def test_the_exact_comparison_never_qualifies_missing_values_and_refuses_floats() -> None:
    assert mon._exact_at_least(Fraction(1, 3), Fraction(1, 3)) and mon._exact_at_least(1, Fraction(1))
    assert not mon._exact_at_least(Fraction(1, 3) - TINY, Fraction(1, 3))
    assert not mon._exact_at_least(None, TINY)                      # zero or missing denominators never qualify
    for approximate in (1 / 3, float("nan"), True):
        with pytest.raises(TypeError):
            mon._exact_at_least(approximate, Fraction(1, 3))      # type: ignore[arg-type]
    with pytest.raises(TypeError):
        mon._exact_at_least(Fraction(1), 0.5)                     # type: ignore[arg-type]
    assert not hasattr(mon, "_at_least")


def test_stale_or_inconsistent_candidate_evidence_is_refused_not_replaced_by_the_float_summary(sync_world) -> None:  # type: ignore[no-untyped-def]
    evidence = evaluate(sync_world).evidence
    analysis = evidence.price_changes
    candidates = analysis.events.candidates
    stale_events = dataclasses.replace(analysis.events)
    object.__setattr__(stale_events, "candidates", candidates[candidates["outcome"] != "increase"])
    stale = object.__new__(type(analysis))
    for f in dataclasses.fields(analysis):
        object.__setattr__(stale, f.name, getattr(analysis, f.name))
    object.__setattr__(stale, "events", stale_events)
    evaluation = one(dataclasses.replace(evidence, price_changes=stale), C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS,
                     movement_policy=approved_movement())
    assert evaluation.status is ST.NOT_ASSESSABLE
    assert evaluation.unavailable_evidence == (G.PRICE_CHANGE_EVIDENCE_INCONSISTENT,)
    assert candidates.equals(analysis.events.candidates)            # the bound evidence itself is unchanged


def test_mixed_direction_intervals_never_become_synchronized_under_an_approved_policy() -> None:
    world = synchronized_world(((3.0, 3.03), (3.0, 2.97)))
    evaluation = movement_status(world, boundary_policy())
    assert evaluation.status is ST.PASSED and not evaluation.findings
    assert N.MIXED_DIRECTION_INTERVALS_NOT_SYNCHRONIZED in evaluation.notes


def test_exact_boundaries_keep_directions_aliases_and_location_counts(sync_world) -> None:  # type: ignore[no-untyped-def]
    # SYNC: Toronto Downtown rises by ten percent and Vancouver (Downtown with its Thurlow alias) falls by ten
    # percent, each with a two-thirds changed share. The exact boundaries hold for both directions separately.
    policy = boundary_policy(minimum_changed_share=Fraction(2, 3), minimum_median_abs_change_percent=Fraction(10))
    evaluation = movement_status(sync_world, policy)
    assert evaluation.findings == (F.APPROVED_SYNCHRONIZED_INCREASE_RULE_MET,
                                   F.APPROVED_SYNCHRONIZED_DECREASE_RULE_MET)
    two = movement_status(sync_world, approved_movement(minimum_changed_share=Fraction(2, 3),
                                                        minimum_median_abs_change_percent=Fraction(10),
                                                        minimum_locations=2))
    assert two.status is ST.PASSED                                   # opposite directions never combine
    exact = mon._exact_intervals(evaluate(sync_world).evidence.price_changes)
    vancouver = [m for (city, location, *_), m in exact.items() if (city, location) == VAN_DOWN and m.decrease]
    assert len(vancouver) == 1 and vancouver[0].decrease == 2       # the alias never doubles the candidates
    assert not any(location == VAN_THUR[1] for _, location, *_ in exact)


def test_unavailable_and_proposed_policies_stay_candidate_only_at_exact_boundaries() -> None:
    world = synchronized_world(((3.0, 3.01), (3.0, 3.01)), steady=3)   # share 1/3, median 1/3
    for policy in (DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY, SynchronizedMovementPolicy(APS.PROPOSED)):
        evaluation = movement_status(world, policy)
        assert evaluation.status is ST.CANDIDATE_ONLY
        assert evaluation.findings == (F.SYNCHRONIZED_INCREASE_REVIEW_CANDIDATE,)


def test_exact_policy_evaluation_is_deterministic_sanitized_and_side_effect_free(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    world = synchronized_world(((3.0, 3.01), (3.0, 3.01)), steady=3)   # share 1/3, median 1/3
    run = full_run(world)
    policy = boundary_policy(minimum_changed_share=Fraction(1, 3), minimum_median_abs_change_percent=Fraction(1, 3))
    candidates = monitoring_from_pipeline(run).evidence.price_changes.events.candidates.copy(deep=True)
    cars = run.cars.copy(deep=True)

    def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("monitoring must not perform I/O")

    before_env = dict(os.environ)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(builtins, "print", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    first = monitoring_from_pipeline(run, movement_policy=policy)
    second = monitoring_from_pipeline(run, movement_policy=policy)
    again = evaluate_monitoring_controls(first.evidence, movement_policy=policy)
    table = monitoring_control_table(first.report)
    monkeypatch.undo()
    assert first.report == second.report == again
    assert first.report.evaluation(C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS).status is ST.TRIGGERED
    assert tuple(table.columns) == MONITORING_TABLE_COLUMNS and len(table) == 8
    validate_monitoring_table(table)
    assert "SYNTH" not in table.to_csv(index=False) and not re.search(r"\d", table.to_csv(index=False))
    pd.testing.assert_frame_equal(first.evidence.price_changes.events.candidates, candidates)
    pd.testing.assert_frame_equal(run.cars, cars)
    assert dict(os.environ) == before_env and list(tmp_path.iterdir()) == []


# ============================================================================ same-run evidence binding

from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract  # noqa: E402
from ql2_sixt_canada_analysis.pricing_pipeline import (  # noqa: E402
    BOUND_PIPELINE_REPORTS,
    PipelineEvidenceManifest,
    PricingPipelineResult,
    pipeline_evidence_bound,
    run_pricing_pipeline,
)


def assert_binding_blocked(result: MonitoringResult) -> None:
    """The fail-closed contract of a binding mismatch: blocked, no trusted evidence, eight not-assessable rows."""
    report = result.report
    assert report.blocked and report.blockers == (MonitoringBlocker.EVIDENCE_BINDING_MISMATCH,)
    assert result.evidence is None
    assert all(e.status is ST.NOT_ASSESSABLE and e.unavailable_evidence == (G.MONITORING_EVIDENCE_BLOCKED,)
               for e in report.evaluations)
    table = monitoring_control_table(report)
    assert table["control_id"].tolist() == [c.value for c in ORDER]
    assert table["condition"].tolist() == [c.condition for c in MONITORING_CONTROLS]
    assert set(table["findings"]) == set(table["notes"]) == {""}
    text = table.to_csv(index=False) + "\n".join(monitoring_summary_lines(report))
    assert "SYNTH" not in text and not re.search(r"\d", text)


@pytest.fixture(scope="module")
def twin_runs():  # type: ignore[no-untyped-def]
    """Two separately built runs over equal fabricated frames: equal report values, distinct report objects."""
    return full_run(synthetic_world(products=QUIET)), full_run(synthetic_world(products=QUIET))


def test_an_intact_run_is_bound_and_evaluated(twin_runs) -> None:  # type: ignore[no-untyped-def]
    run, _ = twin_runs
    assert isinstance(run.evidence, PipelineEvidenceManifest) and pipeline_evidence_bound(run)
    result = monitoring_from_pipeline(run)
    assert result.report.status is MonitoringReportStatus.EVALUATED and result.evidence is not None
    assert result.evidence.coverage is run.coverage and result.evidence.temporal is run.temporal


def test_the_reported_coverage_substitution_fails_closed_against_a_false_trigger(quiet_world) -> None:  # type: ignore[no-untyped-def]
    run = full_run(quiet_world)
    cars = quiet_world["cars"]
    foreign = assess_expected_location_coverage(
        cars[(cars["city"] != TOR_AIR[0]) | (cars["location"] != TOR_AIR[1])], CONTRACT.coverage)
    # The foreign report alone would trigger the control on a run whose own coverage is complete.
    assert one(MonitoringEvidence(contract=CONTRACT, coverage=foreign, scheduled=run.scheduled),
               C.MISSING_EXPECTED_LOCATIONS).status is ST.TRIGGERED
    assert monitoring_from_pipeline(run).report.evaluation(C.MISSING_EXPECTED_LOCATIONS).status is ST.PASSED
    assert_binding_blocked(monitoring_from_pipeline(dataclasses.replace(run, coverage=foreign)))


def test_a_coverage_substitution_fails_closed_against_a_false_pass() -> None:
    world = synthetic_world(products=QUIET)
    cars = world["cars"]
    missing = cars[(cars["city"] != TOR_AIR[0]) | (cars["location"] != TOR_AIR[1])]
    run = full_run(world, coverage=assess_expected_location_coverage(missing, CONTRACT.coverage))
    assert monitoring_from_pipeline(run).report.evaluation(C.MISSING_EXPECTED_LOCATIONS).status is ST.TRIGGERED
    clean = assess_expected_location_coverage(cars, CONTRACT.coverage)
    assert_binding_blocked(monitoring_from_pipeline(dataclasses.replace(run, coverage=clean)))


def _foreign(name: str, other: PricingPipelineResult):  # type: ignore[no-untyped-def]
    """A structurally valid replacement for one bound report: another run's object or a fresh equal copy."""
    fresh = {
        "contract": lambda: dataclasses.replace(current_expected_stream_contract()),
        "reconciliation": lambda: reconciliation(),
        "job_linkage": lambda: link(_jobs([J1, J2], [1, 1]), _cars([J1, J2])).report,
        "temporal_authority": lambda: temporal_authority_from_record(
            load_current_decision_record(), ANALYSIS_TEMPORAL_RECONCILIATION, CONTRACT),
        "temporal": lambda: temporal_report(),
        "vehicle_stability": lambda: stability_report(),
        "location_authority": lambda: dataclasses.replace(AUTHORITY),
    }
    return fresh[name]() if name in fresh else getattr(other, name)


@pytest.mark.parametrize("name", ["contract", "coverage", "reconciliation", "job_linkage", "scheduled",
                                  "temporal_authority", "temporal", "vehicle_stability", "population",
                                  "canonical_offers", "location_authority", "pricing"])
def test_every_foundational_report_from_another_run_is_rejected_even_when_equal(twin_runs, name) -> None:  # type: ignore[no-untyped-def]
    run, other = twin_runs
    foreign = _foreign(name, other)
    assert foreign is not getattr(run, name)
    if name in ("coverage", "reconciliation", "temporal", "vehicle_stability", "scheduled", "job_linkage"):
        assert foreign == getattr(run, name)          # equal public values never prove same-run provenance
    mixed = dataclasses.replace(run, **{name: foreign})
    assert not pipeline_evidence_bound(mixed)
    assert_binding_blocked(monitoring_from_pipeline(mixed))
    assert pipeline_evidence_bound(run)                # the original run is untouched


def test_a_foreign_vehicle_stability_population_and_both_temporal_reports_are_rejected(twin_runs) -> None:  # type: ignore[no-untyped-def]
    run, _ = twin_runs
    different = stability_report(status=VehicleStabilityStatus.VIOLATIONS, entities_with_value_conflicts=1,
                                 fully_stable_entities=1, value_unstable_only_entities=1)
    assert_binding_blocked(monitoring_from_pipeline(dataclasses.replace(run, vehicle_stability=different)))
    both = dataclasses.replace(run, temporal=temporal_report(), temporal_authority=_foreign("temporal_authority", run))
    assert_binding_blocked(monitoring_from_pipeline(both))


def test_substituted_frames_manifests_or_missing_provenance_are_rejected(twin_runs, sync_world) -> None:  # type: ignore[no-untyped-def]
    run, other = twin_runs
    shuffled = dataclasses.replace(run, cars=run.cars.iloc[::-1])
    assert_binding_blocked(monitoring_from_pipeline(shuffled))                       # frames differ from the binding
    swapped = dataclasses.replace(run, coverage=other.coverage, evidence=other.evidence)
    assert_binding_blocked(monitoring_from_pipeline(swapped))                         # another run's manifest
    assert_binding_blocked(monitoring_from_pipeline(dataclasses.replace(run, evidence=None)))
    elsewhere = full_run(sync_world)
    assert_binding_blocked(monitoring_from_pipeline(
        dataclasses.replace(elsewhere, jobs=run.jobs, cars=run.cars)))                # frames of another run


def test_runs_of_the_real_pipeline_are_bound_and_cannot_be_mixed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    directory = tmp_path / "synthetic_raw"
    directory.mkdir()
    for key in DatasetKey:
        write_synthetic_csv(directory / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    first, second = run_pricing_pipeline(directory), run_pricing_pipeline(directory)
    assert pipeline_evidence_bound(first) and pipeline_evidence_bound(second)
    assert not first.pricing_analysis_ready                    # pricing is blocked on these fabricated rows ...
    report = monitoring_from_pipeline(first).report
    assert not report.blocked                                 # ... yet the bound structural evidence is assessed
    assert report.evaluation(C.MISSING_EXPECTED_LOCATIONS).status is ST.TRIGGERED
    assert report.evaluation(C.JOB_DETAIL_COUNT_MISMATCHES).status is not ST.NOT_ASSESSABLE
    for name in ("coverage", "reconciliation", "temporal", "temporal_authority", "pricing"):
        assert getattr(second, name) == getattr(first, name) or name in ("pricing", "temporal_authority")
        assert_binding_blocked(monitoring_from_pipeline(dataclasses.replace(first, **{name: getattr(second, name)})))


def test_binding_validation_never_reassesses_retained_reports(twin_runs, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import ql2_sixt_canada_analysis as pkg
    from ql2_sixt_canada_analysis import coverage, reconciliation as rec, stability, temporal

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("retained evidence must not be reassessed")

    for module, name in ((coverage, "assess_expected_location_coverage"),
                         (coverage, "assess_dataset_location_coverage"),
                         (rec, "assess_job_detail_reconciliation"), (temporal, "assess_temporal_reconciliation"),
                         (stability, "assess_vehicle_attribute_stability")):
        monkeypatch.setattr(module, name, forbidden)
        if name in pkg.__all__:
            monkeypatch.setattr(pkg, name, forbidden)
    run, other = twin_runs
    assert not monitoring_from_pipeline(run).report.blocked
    assert_binding_blocked(monitoring_from_pipeline(dataclasses.replace(run, reconciliation=other.reconciliation)))


def test_binding_is_deterministic_immutable_and_never_mutates_its_inputs(quiet_world) -> None:  # type: ignore[no-untyped-def]
    run = full_run(quiet_world)
    jobs, cars = run.jobs.copy(deep=True), run.cars.copy(deep=True)
    unbound = dataclasses.replace(run, evidence=None)
    rebound = bind_pipeline_evidence(unbound)
    assert unbound.evidence is None and rebound.evidence is not unbound.evidence
    assert all(getattr(rebound, n) is getattr(run, n) for n, _ in BOUND_PIPELINE_REPORTS)
    assert [pipeline_evidence_bound(run) for _ in range(3)] == [True] * 3
    assert monitoring_from_pipeline(run).report == monitoring_from_pipeline(run).report
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.evidence.reports = ()                                                # type: ignore[misc]
    with pytest.raises(ValueError):
        PipelineEvidenceManifest(frames=run.evidence.frames, eligible=None, reports=run.evidence.reports[:-1])
    pd.testing.assert_frame_equal(run.jobs, jobs)
    pd.testing.assert_frame_equal(run.cars, cars)


def test_provenance_never_appears_in_representations(quiet_world) -> None:  # type: ignore[no-untyped-def]
    run = full_run(quiet_world)
    text = repr(run.evidence) + repr(run)
    assert text.startswith("PipelineEvidenceManifest()")
    assert "SYNTH" not in text and "digest" not in text and "DataFrame" not in text and not re.search(r"\d", text)
    assert "evidence" not in repr(run)
