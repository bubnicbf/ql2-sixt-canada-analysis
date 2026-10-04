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
from test_city_integrity import PROJECT_GATES, completeness as project_completeness, healthy
from test_completeness import SYNTH_COV
from test_readiness import DISTINCT, GATES, STABLE, scheduled_coverage, scheduled_frames

from ql2_sixt_canada_analysis import pricing_baseline as pb
from ql2_sixt_canada_analysis.pricing_baseline import (
    LocationRole,
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
    EXPECTED_LOCATION_COVERAGE as COV,
    INVESTIGATED_LOCATION_STREAM,
    TEMPORAL_RECONCILIATION,
    DatasetKey,
    LocationCoverageMode,
    LocationPolicyAuthority,
    TemporalAwareness,
    TemporalFieldDefinition,
    TemporalKind,
    TemporalReplicationRule,
)
from ql2_sixt_canada_analysis.temporal import assess_temporal_reconciliation

SYNTH_TARGET = SYNTH_COV.expected_locations[0]


def synth_pricing(**changes):  # type: ignore[no-untyped-def]
    return assess_pricing_readiness(location_policy=assess_location_policy(), **(GATES | changes))


def synth_baseline(pricing=None, cars=None, **kwargs):  # type: ignore[no-untyped-def]
    j, c = scheduled_frames()
    return build_pricing_baseline(
        pricing=pricing if pricing is not None else synth_pricing(), cars=cars if cars is not None else c,
        temporal=kwargs.pop("temporal", None), vehicle_stability=kwargs.pop("vehicle_stability", STABLE),
        coverage=kwargs.pop("coverage", SYNTH_COV), investigated_stream=kwargs.pop("investigated_stream", SYNTH_TARGET),
        **kwargs)


def project_baseline(cars=None, **gate_changes):  # type: ignore[no-untyped-def]
    j, c = healthy()
    gates = PROJECT_GATES(project_completeness(j, c)) | gate_changes
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(), **gates)
    return build_pricing_baseline(pricing=pricing, cars=c if cars is None else cars, temporal=None,
                                  vehicle_stability=STABLE)


# ------------------------------------------------------------ active blockers


def test_active_blockers_are_the_typed_report_values():
    pricing = synth_pricing(scheduled_coverage=scheduled_coverage(schedule=None), job_detail_join=None)
    baseline = synth_baseline(pricing)
    assert baseline.pricing_blockers == tuple(b.value for b in pricing.blocking_reasons)
    sub = dict(baseline.subordinate_blockers)
    assert sub["scheduled_coverage"] == tuple(b.value for b in pricing.scheduled_coverage.blocking_reasons)
    assert sub["trusted_join"] == ("trusted_join_assessment_missing",)
    assert sub["location_policy"] == ("vancouver_policy_unresolved",)
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
    assert baseline.plan_gaps == (G.EXPECTED_STREAMS_NOT_EXHAUSTIVE, G.LOCATION_ROLE_MAP_UNAVAILABLE,
                                  G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE)


def test_minimum_required_contract_is_a_plan_gap_not_a_pricing_blocker():
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT), **GATES)
    baseline = synth_baseline(pricing=pricing)                          # every central gate passes
    assert baseline.pricing_ready and baseline.pricing_blockers == ()
    assert G.EXPECTED_STREAMS_NOT_EXHAUSTIVE in baseline.plan_gaps
    assert baseline.expected_population.authority == "authoritative_minimum_required"
    exhaustive = dataclasses.replace(SYNTH_COV, mode=LocationCoverageMode.EXHAUSTIVE)
    assert G.EXPECTED_STREAMS_NOT_EXHAUSTIVE not in synth_baseline(coverage=exhaustive).plan_gaps


AUTHORITY = LocationPolicyAuthority(source="SYNTH-AUTHORITY", reference="SYNTH-DECISION-003")


def full_role_map(baseline):  # type: ignore[no-untyped-def]
    keys = set(baseline.expected_population.keys) | set(baseline.observed_population.keys)
    return {k: LocationRole.DOWNTOWN for k in sorted(keys)}


def test_missing_role_map_is_a_plan_gap():
    assert G.LOCATION_ROLE_MAP_UNAVAILABLE in synth_baseline().plan_gaps


def test_role_map_gap_closes_only_with_complete_typed_authority_backed_roles():
    j, c = healthy()
    # One extra observed (not expected) stream, so expected-only maps are incomplete.
    c = pd.concat([c, c.iloc[[0]].assign(**{COV.label_column: "SYNTH Airport"})], ignore_index=True).astype(
        dict(REL.detail_definition.identifier_dtypes))
    complete = full_role_map(project_baseline(cars=c))
    assert len(complete) == 4
    one_key = dict(list(complete.items())[:1])

    def gaps(**kwargs):  # type: ignore[no-untyped-def]
        pricing = assess_pricing_readiness(location_policy=assess_location_policy(),
                                           **PROJECT_GATES(project_completeness(j, c)))
        return build_pricing_baseline(pricing=pricing, cars=c, temporal=None, vehicle_stability=STABLE,
                                      **kwargs).plan_gaps

    still_open = [
        dict(location_role_map=complete),                                            # no authority
        dict(location_role_map=one_key, location_role_authority=AUTHORITY),           # partial map
        dict(location_role_map={k: "SYNTH-ROLE" for k in complete}, location_role_authority=AUTHORITY),
        dict(location_role_map={k: "airport" for k in complete}, location_role_authority=AUTHORITY),  # untyped
        dict(location_role_map={**complete, ("SYNTH",): LocationRole.AIRPORT}, location_role_authority=AUTHORITY),
        dict(location_role_map={}, location_role_authority=AUTHORITY),
        dict(location_role_map=complete, location_role_authority="SYNTH-AUTHORITY"),
    ]
    for kwargs in still_open:
        assert G.LOCATION_ROLE_MAP_UNAVAILABLE in gaps(**kwargs), kwargs
    # A map covering the expected streams only misses observed ones: still a gap.
    expected_only = {k: LocationRole.DOWNTOWN for k in COV.expected_locations}
    assert G.LOCATION_ROLE_MAP_UNAVAILABLE in gaps(location_role_map=expected_only, location_role_authority=AUTHORITY)
    assert G.LOCATION_ROLE_MAP_UNAVAILABLE not in gaps(location_role_map=complete, location_role_authority=AUTHORITY)


def _rental_contract(*, date_fields=True, required=True, replications=True):  # type: ignore[no-untyped-def]
    T = TEMPORAL_RECONCILIATION
    fields = list(T.fields)
    reps = list(T.replications)
    if date_fields:
        for column in pb.RENTAL_PERIOD_COLUMNS:
            for dataset in (DatasetKey.JOBS, DatasetKey.CARS):
                fields.append(TemporalFieldDefinition(dataset, column, TemporalKind.DATE, required,
                                                      "%Y-%m-%d", TemporalAwareness.NOT_APPLICABLE))
            if replications:
                reps.append(TemporalReplicationRule((DatasetKey.JOBS, column), (DatasetKey.CARS, column)))
    return dataclasses.replace(T, fields=tuple(fields), replications=tuple(reps))


def test_rental_date_gap_closes_only_with_validity_agreement_and_authority():
    def gaps(contract, authority=AUTHORITY):  # type: ignore[no-untyped-def]
        return project_baseline_with(temporal_contract=contract, rental_period_rule_authority=authority).plan_gaps

    for contract, authority in ((_rental_contract(), None),                         # fields+rules, no authority
                                (_rental_contract(replications=False), AUTHORITY),  # no parent/detail agreement
                                (_rental_contract(required=False), AUTHORITY),      # validity not required
                                (_rental_contract(date_fields=False), AUTHORITY),   # not modeled at all
                                (TEMPORAL_RECONCILIATION, AUTHORITY)):
        assert G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE in gaps(contract, authority)
    assert G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE not in gaps(_rental_contract())


def project_baseline_with(**kwargs):  # type: ignore[no-untyped-def]
    j, c = healthy()
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(),
                                       **PROJECT_GATES(project_completeness(j, c)))
    return build_pricing_baseline(pricing=pricing, cars=c, temporal=None, vehicle_stability=STABLE, **kwargs)


# --------------------------------------------------------------- populations


def test_expected_and_observed_populations_stay_separate():
    j, c = healthy()
    extra = pd.concat([c, c.iloc[[0]].assign(**{COV.label_column: "SYNTH Airport"})], ignore_index=True)
    baseline = project_baseline(cars=extra)
    assert baseline.expected_population.keys == tuple(sorted(COV.expected_locations))
    assert baseline.expected_population.count == 3 and baseline.observed_population.count == 4
    assert baseline.observed_population.authority == "observed_not_authoritative"
    assert (INVESTIGATED_LOCATION_STREAM[0], "SYNTH Airport") in baseline.observed_population.keys
    assert (INVESTIGATED_LOCATION_STREAM[0], "SYNTH Airport") not in baseline.expected_population.keys


def test_ordering_is_deterministic_and_deduplicated():
    j, c = healthy()
    doubled = pd.concat([c, c], ignore_index=True)
    forward = project_baseline(cars=doubled)
    backward = project_baseline(cars=doubled.iloc[::-1])
    assert forward == backward
    keys = forward.observed_population.keys
    assert list(keys) == sorted(set(keys))
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
    baseline = build_pricing_baseline(pricing=pricing, cars=cars, temporal=None, vehicle_stability=STABLE)
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
                   dict(vehicle_stability=True), dict(investigated_stream=("SYNTH-NOT", "EXPECTED"))):
        inputs = dict(pricing=pricing, cars=c, temporal=None, vehicle_stability=STABLE, coverage=SYNTH_COV,
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
