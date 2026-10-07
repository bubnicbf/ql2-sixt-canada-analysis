"""Sanitized pricing-readiness baseline: a reporting layer over the existing typed assessments.

Synthetic inputs only (fabricated ``SYNTH-...`` rows built by the readiness and
city-integrity fixtures); the real source files are never read here.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pandas as pd
import pytest
from stream_contract_fixtures import synthetic_contract
from test_city_integrity import COV, PROJECT_GATES, completeness as project_completeness, healthy
from test_completeness import SYNTH_COV
from test_readiness import DISTINCT, GATES, STABLE, scheduled_coverage, scheduled_frames

from ql2_sixt_canada_analysis import pricing_baseline as pb
from ql2_sixt_canada_analysis.authority_decisions import AuthorityKind, AuthorityReference
from ql2_sixt_canada_analysis.pricing_baseline import (
    BaselineInputError,
    ContinuityFinding,
    PlanReadinessGap as G,
    PricingReadinessBaseline,
    StreamPopulation,
    UnsafeBaselineValueError,
    build_pricing_baseline,
    render_baseline_markdown,
)
from ql2_sixt_canada_analysis.readiness import PricingBlocker as B, assess_location_policy, assess_pricing_readiness
from ql2_sixt_canada_analysis.identifiers import is_identifier_dtype
from ql2_sixt_canada_analysis.schemas import (
    JOB_DETAIL_RELATIONSHIP as REL,
    INVESTIGATED_LOCATION_STREAM,
    TEMPORAL_RECONCILIATION,
    DatasetKey,
    LocationCoverageMode,
)
from ql2_sixt_canada_analysis.coverage import location_pair_evidence
from ql2_sixt_canada_analysis.temporal import assess_temporal_reconciliation

SYNTH_TARGET = SYNTH_COV.expected_locations[0]


def synth_pricing(**changes):  # type: ignore[no-untyped-def]
    return assess_pricing_readiness(location_policy=assess_location_policy(), **(GATES | changes))


def synth_baseline(pricing=None, cars=None, **kwargs):  # type: ignore[no-untyped-def]
    j, c = scheduled_frames()
    return build_pricing_baseline(
        pricing=pricing if pricing is not None else synth_pricing(), jobs=kwargs.pop("jobs", j),
        cars=cars if cars is not None else c,
        temporal=kwargs.pop("temporal", None), vehicle_stability=kwargs.pop("vehicle_stability", STABLE),
        coverage=kwargs.pop("coverage", SYNTH_COV), investigated_stream=kwargs.pop("investigated_stream", SYNTH_TARGET),
        **kwargs)


def project_baseline(cars=None, **gate_changes):  # type: ignore[no-untyped-def]
    j, c = healthy()
    gates = PROJECT_GATES(project_completeness(j, c)) | gate_changes
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(), **gates)
    return build_pricing_baseline(pricing=pricing, jobs=j, cars=c if cars is None else cars, temporal=None,
                                  vehicle_stability=STABLE)


# ------------------------------------------------------------ active blockers


def test_active_blockers_are_the_typed_report_values():
    pricing = synth_pricing(scheduled_coverage=scheduled_coverage(schedule=None), job_detail_join=None)
    baseline = synth_baseline(pricing)
    assert baseline.pricing_blockers == tuple(b.value for b in pricing.blocking_reasons)
    sub = dict(baseline.subordinate_blockers)
    assert sub["scheduled_coverage"] == tuple(b.value for b in pricing.scheduled_coverage.blocking_reasons)
    assert sub["trusted_join"] == ("trusted_join_assessment_missing",)
    assert sub["location_policy"] == tuple(b.value for b in pricing.location_policy.blocking_reasons)
    assert ("collection_schedule", "unavailable") in baseline.statuses
    assert not baseline.pricing_ready


def test_temporal_report_values_are_copied_not_reconstructed():
    j, c = scheduled_frames()
    temporal = assess_temporal_reconciliation(j, c, TEMPORAL_RECONCILIATION)
    baseline = synth_baseline(temporal=temporal)
    sub = dict(baseline.subordinate_blockers)
    assert sub["temporal"] == temporal.violations
    assert sub["temporal_unavailable_rules"] == temporal.unavailable_rules
    assert ("temporal_fields_trusted", "true" if temporal.is_valid else "false") in baseline.statuses
    assert dict(synth_baseline().subordinate_blockers)["temporal"] == ("temporal_report_missing",)


def test_active_blockers_and_plan_gaps_are_separate():
    baseline = project_baseline()
    gap_values = {g.value for g in G}
    assert not gap_values & set(baseline.pricing_blockers)
    assert not any(gap_values & set(codes) for _, codes in baseline.subordinate_blockers)
    assert not gap_values & {b.value for b in B}                  # gaps are not PricingBlocker values
    assert baseline.plan_gaps == ()                                # rental rules ran under an approved policy
    assert project_baseline(rental_dates=None).plan_gaps == (G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE,)
    assert not any("exhaustive" in g.value for g in G)     # exhaustiveness is a central blocker, not a gap


def test_non_exhaustive_contract_is_a_central_blocker_not_a_plan_gap():
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT), **GATES)
    baseline = synth_baseline(pricing=pricing)                          # every central gate passes
    assert baseline.pricing_ready and baseline.pricing_blockers == ()
    assert baseline.expected_population.authority == "authoritative_exhaustive"
    minimum = dataclasses.replace(SYNTH_COV, mode=LocationCoverageMode.MINIMUM_REQUIRED)
    blocked = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                       **(GATES | {"expected_stream_contract": synthetic_contract(minimum)}))
    assert B.EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE in blocked.blocking_reasons
    assert B.EXPECTED_STREAM_CONTRACT_MISMATCH in blocked.blocking_reasons      # completeness used another contract
    report = synth_baseline(pricing=blocked, coverage=minimum)
    assert "expected_stream_universe_not_exhaustive" in report.pricing_blockers
    assert report.expected_population.authority == "authoritative_minimum_required"
    assert set(report.plan_gaps) <= set(G)


AUTHORITY = AuthorityReference(kind=AuthorityKind.BUSINESS_OWNER, source="SYNTH-AUTHORITY",
                               reference="SYNTH-DECISION-003")


def test_role_map_is_a_central_gate_not_a_plan_gap():
    assert not any("role" in g.value for g in G) and tuple(G) == (G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE,)
    missing = synth_baseline(pricing=synth_pricing(location_authority=None))
    assert {"branch_role_authority_unavailable", "comparison_pair_authority_unavailable"} <= set(
        missing.pricing_blockers)
    assert dict(missing.subordinate_blockers)["location_authority"] == (
        "branch_role_authority_unavailable", "comparison_pair_authority_unavailable")


J, C_ = DatasetKey.JOBS, DatasetKey.CARS


def test_project_distinguishes_six_rental_date_fields():
    assert pb.rental_date_fields(REL) == (
        (J, "pickup_date"), (J, "return_date"), (C_, "job_pickup_date"), (C_, "job_return_date"),
        (C_, "pickup_date"), (C_, "return_date"))


def test_rental_date_gap_follows_the_central_rental_assessment():
    from stream_contract_fixtures import synthetic_rental_policy

    from ql2_sixt_canada_analysis.rental_dates import RentalDatePolicy, RentalDateReport, RentalPolicyStatus

    assert project_baseline().plan_gaps == ()
    unapproved = RentalDateReport(policy=RentalDatePolicy(status=RentalPolicyStatus.NOT_APPROVED))
    baseline = project_baseline(rental_dates=unapproved)
    assert baseline.plan_gaps == (G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE,)
    assert "rental_date_rules_unavailable" in baseline.pricing_blockers         # and a central blocker
    missing = project_baseline(rental_dates=None)
    assert "rental_date_assessment_missing" in missing.pricing_blockers
    assert missing.rental_dates.policy_status == "report_missing"
    assert synthetic_rental_policy().available


def project_baseline_with(**kwargs):  # type: ignore[no-untyped-def]
    j, c = healthy()
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(),
                                       **PROJECT_GATES(project_completeness(j, c)))
    return build_pricing_baseline(pricing=pricing, jobs=j, cars=c, temporal=None, vehicle_stability=STABLE,
                                  **kwargs)


# --------------------------------------------------------------- populations


def test_expected_and_observed_populations_stay_separate():
    j, c = healthy()
    extra = pd.concat([c, c.iloc[[0]].assign(**{COV.label_column: "SYNTH Airport"})], ignore_index=True)
    baseline = project_baseline(cars=extra)
    assert baseline.expected_population.keys == tuple(COV.expected_locations)          # contract order
    assert baseline.expected_population.count == 3 and baseline.observed_population.count == 4
    observed = baseline.observed_population
    assert observed.authority == "observed_not_authoritative"
    assert (observed.exact_expected_count, observed.unexpected_count, observed.spelling_variant_count,
            observed.expected_missing_count) == (3, 1, 0, 0)
    # Only observed keys that equal approved keys are named; the extra observed key is a count only.
    assert observed.keys == tuple(COV.expected_locations)
    assert (INVESTIGATED_LOCATION_STREAM[0], "SYNTH Airport") not in observed.keys
    assert (INVESTIGATED_LOCATION_STREAM[0], "SYNTH Airport") not in baseline.expected_population.keys


def test_ordering_is_deterministic_and_deduplicated():
    j, c = healthy()
    doubled = pd.concat([c, c], ignore_index=True)
    forward = project_baseline(cars=doubled)
    backward = project_baseline(cars=doubled.iloc[::-1])
    assert forward == backward
    keys = forward.observed_population.keys
    assert len(keys) == len(set(keys)) and list(keys) == [k for k in COV.expected_locations if k in keys]
    assert len(forward.pricing_blockers) == len(set(forward.pricing_blockers))


# ------------------------------------------------------------- continuity


def test_continuity_finding_is_aggregate_only():
    j, c = healthy()
    (parent_key,), (detail_key,) = REL.parent_key_columns, REL.detail_key_columns
    # Restore the contract identifier dtypes after concatenation (pandas may widen them),
    # so the reconciliation preconditions hold on every pandas version.
    j2 = pd.concat([j, j.iloc[[0]].assign(**{parent_key: "SYNTH-JOB-009"})], ignore_index=True).astype(
        dict(REL.parent_definition.identifier_dtypes))
    # A Calgary capture event that carries another Calgary branch but not the investigated stream.
    extra = c.iloc[[0]].assign(**{detail_key: "SYNTH-JOB-009", COV.label_column: "SYNTH Airport"})
    cars = pd.concat([c, extra], ignore_index=True).astype(dict(REL.detail_definition.identifier_dtypes))
    assert all(is_identifier_dtype(j2[col].dtype) for col in REL.parent_key_columns)
    assert all(is_identifier_dtype(cars[col].dtype) for col in REL.detail_key_columns)
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(),
                                       **(PROJECT_GATES(project_completeness(j2, cars))
                                          | {"scheduled_coverage": scheduled_coverage(schedule=None)}))
    baseline = build_pricing_baseline(pricing=pricing, jobs=j2, cars=cars, temporal=None, vehicle_stability=STABLE)
    finding = baseline.continuity
    assert isinstance(finding, ContinuityFinding) and finding.stream == INVESTIGATED_LOCATION_STREAM
    assert (finding.in_scope_capture_events, finding.capture_events_lacking_stream) == (2, 1)
    assert all(type(getattr(finding, f.name)) in (int, bool, str, tuple) for f in dataclasses.fields(finding))
    text = json.dumps(baseline.to_dict())
    assert "SYNTH-JOB" not in text
    markdown = render_baseline_markdown(baseline, commit="abc1234", date="2026-10-04")
    assert "1 of 2 in-scope capture events" in markdown and "not called scheduled" in markdown
    assert "SYNTH-JOB" not in markdown


# -------------------------------------------------------------- sanitization


def test_serialized_output_contains_no_source_material():
    j, c = healthy()
    c = c.copy()
    c["car_name"] = "SYNTH Vehicle Name"
    c["price_num"] = 12.34
    c["scraped_at"] = "2025-01-15 05:00:00 MST"
    baseline = project_baseline(cars=c)
    text = json.dumps(baseline.to_dict()) + render_baseline_markdown(baseline, commit="abc1234", date="2026-10-04")
    for forbidden in ("SYNTH-JOB", "SYNTH Vehicle Name", "12.34", "2025-01-15", "05:00:00", "MST"):
        assert forbidden not in text, forbidden


@pytest.mark.parametrize("field, value", [
    ("pricing_blockers", ("SYNTH-JOB-001",)),            # identifier
    ("pricing_blockers", ("2025-01-15 05:00:00",)),     # timestamp
    ("pricing_blockers", ("12.50",)),                    # price
    ("pricing_blockers", (pd.DataFrame({"x": [1]}),)),  # DataFrame
    ("pricing_blockers", (1.5,)),                        # float
    ("statuses", (("vehicle", "Corolla Hybrid 2024"),)),
    ("subordinate_blockers", (("sample", (object(),)),)),
])
def test_unsafe_material_is_rejected(field, value):
    baseline = dataclasses.replace(project_baseline(), **{field: value})
    with pytest.raises(UnsafeBaselineValueError) as info:
        baseline.to_dict()
    assert "SYNTH" not in str(info.value) and "Corolla" not in str(info.value)
    with pytest.raises(UnsafeBaselineValueError):
        render_baseline_markdown(baseline, commit="abc1234", date="2026-10-04")


def test_unsafe_location_labels_and_render_arguments_are_rejected():
    base = project_baseline()
    bad = dataclasses.replace(base, observed_population=dataclasses.replace(
        base.observed_population, keys=(("calgary", "Branch 42"),)))
    with pytest.raises(UnsafeBaselineValueError):
        bad.to_dict()
    for commit, date in (("not a hash", "2026-10-04"), ("abc1234", "04/10/2026")):
        with pytest.raises(UnsafeBaselineValueError):
            render_baseline_markdown(base, commit=commit, date=date)


def test_synthetic_safe_baseline_renders():
    markdown = render_baseline_markdown(project_baseline(), commit="abc1234", date="2026-10-04")
    assert "**NOT PRICING READY**" in markdown
    assert "### A. Active blockers" in markdown and "### B. Plan-level prerequisites" in markdown
    assert "Configured expected streams: 3" in markdown and "Observations do not establish authority" in markdown
    assert markdown == render_baseline_markdown(project_baseline(), commit="abc1234", date="2026-10-04")


# ------------------------------------------------------------ fail closed


def test_missing_or_malformed_inputs_fail_closed():
    j, c = scheduled_frames()
    pricing = synth_pricing()
    for kwargs in (dict(pricing=None), dict(pricing=True), dict(cars=c.iloc[0:0]), dict(cars=None),
                   dict(cars=c.drop(columns=[SYNTH_COV.label_column])), dict(temporal="trusted"),
                   dict(vehicle_stability=True), dict(investigated_stream=("SYNTH-NOT", "EXPECTED")),
                   dict(jobs=None), dict(jobs=j.iloc[0:0]), dict(coverage=COV)):     # COV: not the pricing contract
        inputs = dict(pricing=pricing, jobs=j, cars=c, temporal=None, vehicle_stability=STABLE, coverage=SYNTH_COV,
                      investigated_stream=SYNTH_TARGET) | kwargs
        with pytest.raises(BaselineInputError):
            build_pricing_baseline(**inputs)
    with pytest.raises(BaselineInputError):
        render_baseline_markdown({"pricing_ready": True}, commit="abc1234", date="2026-10-04")  # type: ignore[arg-type]


def test_unavailable_assessments_are_reported_not_passed():
    baseline = synth_baseline(pricing=synth_pricing(scheduled_coverage=None, job_detail_join=None),
                              vehicle_stability=None)
    sub = dict(baseline.subordinate_blockers)
    assert sub["scheduled_coverage"] == ("scheduled_coverage_assessment_missing",)
    assert ("vehicle_stability", "unavailable") in baseline.statuses and not baseline.pricing_ready


def test_inputs_are_not_mutated():
    j, c = scheduled_frames()
    pricing = synth_pricing()
    before_frame, before_pricing = c.copy(deep=True), copy.deepcopy(pricing.blocking_reasons)
    synth_baseline(pricing=pricing, cars=c)
    pd.testing.assert_frame_equal(c, before_frame)
    assert pricing.blocking_reasons == before_pricing


def test_readiness_semantics_are_unchanged():
    report = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT), **GATES)
    assert report.ready and report.blocking_reasons == ()
    assert not {g.value for g in G} & {b.value for b in B}
    assert isinstance(synth_baseline(), PricingReadinessBaseline)
    assert isinstance(synth_baseline().expected_population, StreamPopulation)
    import ql2_sixt_canada_analysis
    assert not set(pb.__all__) & set(ql2_sixt_canada_analysis.__all__)   # not re-exported from the package
