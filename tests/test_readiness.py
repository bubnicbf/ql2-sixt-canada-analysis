"""Tests for the location identity policy gate and fail-closed pricing readiness.

Authority configuration (synthetic ``LocationIdentityPolicy`` objects) and
behavioural evidence (comparison reports) are built separately. All labels
and authority metadata are fabricated (``SYNTH-BRANCH-A``, ``SYNTH-AUTHORITY``).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from test_comparison import COV, DEF, _jobs, same_both
from test_vehicle_stability import T, V1, V2, frame as stability_frame, obs, two

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.comparison import (
    IdentityEvidence,
    LocationStreamComparisonReport,
    LocationStreamComparisonStatus as CS,
    OfferSetResult,
    ScopeBaseline,
    TemporalOverlap,
    compare_location_streams,
)
from ql2_sixt_canada_analysis.readiness import (
    AnalyticalLocationKeys,
    LocationPolicyReport,
    PricingBlocker as B,
    PricingNotReadyError,
    PricingReadinessReport,
    apply_location_policy,
    assess_location_policy,
    assess_pricing_readiness,
    validate_pricing_readiness,
)
from ql2_sixt_canada_analysis.stability import VehicleStabilityStatus, assess_vehicle_attribute_stability
from ql2_sixt_canada_analysis.schemas import (
    VEHICLE_ATTRIBUTE_STABILITY as V,
    COMPARED_LOCATION_STREAMS,
    EXPECTED_LOCATION_COVERAGE,
    LOCATION_STREAM_COMPARISON,
    VANCOUVER_LOCATION_POLICY,
    LocationIdentityPolicy,
    LocationPolicyAuthority,
    LocationPolicyConfigurationError,
    LocationPolicyState as PS,
)

A, B_, C = DEF.first, DEF.second, ("SYNTH-BRANCH-C",)
LOC = COV.location_columns[0]
CANONICAL = ("SYNTH-CANONICAL-BRANCH",)
AUTHORITY = LocationPolicyAuthority(source="SYNTH-AUTHORITY", reference="SYNTH-DECISION-001",
                                    note="Fabricated decision for tests.")
UNRESOLVED = LocationIdentityPolicy(first=A, second=B_, coverage=COV)
DISTINCT = dataclasses.replace(UNRESOLVED, state=PS.CONFIRMED_DISTINCT, authority=AUTHORITY)
ALIAS = dataclasses.replace(UNRESOLVED, state=PS.CONFIRMED_ALIAS, authority=AUTHORITY, canonical_location=CANONICAL)
# Vehicle-stability evidence comes from the real assessment on fabricated vehicles.
STABLE = assess_vehicle_attribute_stability(two())                                   # full population passes
UNSTABLE = assess_vehicle_attribute_stability(two(**{V.attribute_columns[0]: "SYNTH-CLASS-B"}))
PARTIAL = assess_vehicle_attribute_stability(stability_frame([obs(V1, T[0]), obs(V1, T[1]), obs(V2, T[0])]))
UNSTABLE_AND_PARTIAL = assess_vehicle_attribute_stability(stability_frame([
    obs(V1, T[0]), obs(V1, T[1], **{V.attribute_columns[0]: "SYNTH-CLASS-B"}), obs(V2, T[0])]))
GATES = dict(key_contracts_valid=True, expected_coverage_passed=True, expected_stream_healthy=True,
             job_detail_counts_reconciled=True, one_to_many_contract_valid=True,
             temporal_fields_trusted=True, vehicle_stability=STABLE)
FAILING = {gate: False for gate in GATES} | {"vehicle_stability": UNSTABLE}


def evidence(status: CS) -> LocationStreamComparisonReport:
    """A behavioural comparison report with the given status (evidence only)."""
    return LocationStreamComparisonReport(
        status=status, targets_configured=True, first_present=True, second_present=True,
        first_details_linked=True, second_details_linked=True, identity_evidence=IdentityEvidence.UNAVAILABLE,
        shares_collection_events=True, temporal_overlap=TemporalOverlap.COMPLETE, ambiguous_pairing=False,
        comparable_captures_exist=True, product_sets=OfferSetResult.IDENTICAL,
        price_aware_offers=OfferSetResult.IDENTICAL, synchronized_prices=True,
        scope_baseline=ScopeBaseline.DISCRIMINATIVE)


def frame(labels: list[tuple[str, ...]]) -> pd.DataFrame:
    return pd.DataFrame({LOC: [k[0] for k in labels], "synth_other": range(len(labels))},
                        index=[f"SYNTH-ROW-{i}" for i in range(len(labels))])


BEHAVIOURAL = [CS.LIKELY_DUPLICATE_STREAMS, CS.LIKELY_DISTINCT_STREAMS, CS.COMPARISON_INCONCLUSIVE,
               CS.INSUFFICIENT_COMPARABLE_CAPTURES, CS.ONE_STREAM_ABSENT, CS.BOTH_STREAMS_ABSENT]


# --------------------------------------------------------- default unresolved


def test_project_vancouver_policy_defaults_to_unresolved_without_authority():
    assert VANCOUVER_LOCATION_POLICY.state is PS.UNRESOLVED
    assert VANCOUVER_LOCATION_POLICY.authority is None and VANCOUVER_LOCATION_POLICY.canonical_location is None
    assert (VANCOUVER_LOCATION_POLICY.first, VANCOUVER_LOCATION_POLICY.second) == COMPARED_LOCATION_STREAMS
    assert (LOCATION_STREAM_COMPARISON.first, LOCATION_STREAM_COMPARISON.second) == COMPARED_LOCATION_STREAMS
    assert VANCOUVER_LOCATION_POLICY.coverage is EXPECTED_LOCATION_COVERAGE
    assert not VANCOUVER_LOCATION_POLICY.resolved and dict(VANCOUVER_LOCATION_POLICY.alias_mapping) == {}


def test_unresolved_policy_grants_no_permission_and_blocks_pricing():
    report = assess_location_policy()                      # project default
    assert report.state is PS.UNRESOLVED and report.authority is None
    assert report.location_policy_resolved is False
    assert report.location_policy_authority_sufficient is False
    assert report.locations_are_aliases is False
    assert report.locations_comparable_independently is False
    assert report.canonicalization_required is False and report.canonicalization_applied is False
    assert report.blocking_reasons == (B.LOCATION_POLICY_UNRESOLVED,)
    readiness = assess_pricing_readiness(location_policy=report, **GATES)
    assert readiness.ready is False and readiness.blocking_reasons == (B.LOCATION_POLICY_UNRESOLVED,)


def test_alias_false_is_not_permission_to_compare_independently():
    # The former boolean handoff (alias confirmed = False) was read as "distinct";
    # under an unresolved policy neither permission is granted.
    report = assess_location_policy(UNRESOLVED)
    assert not report.locations_are_aliases and not report.locations_comparable_independently
    with pytest.raises(PricingNotReadyError) as info:
        validate_pricing_readiness(location_policy=report, **GATES)
    assert info.value.blocking_reasons == (B.LOCATION_POLICY_UNRESOLVED,)
    assert "SYNTH" not in str(info.value)


# --------------------------------------------- behavioural evidence is not authority


@pytest.mark.parametrize("status", BEHAVIOURAL)
def test_behavioural_evidence_never_resolves_the_policy(status):
    report = assess_location_policy(UNRESOLVED, evidence(status))
    assert report.behavioral_evidence is status and report.state is PS.UNRESOLVED
    assert not report.location_policy_resolved and not report.locations_are_aliases
    assert not report.locations_comparable_independently
    readiness = assess_pricing_readiness(location_policy=report, **GATES)
    assert not readiness.ready and B.LOCATION_POLICY_UNRESOLVED in readiness.blocking_reasons


def test_likely_duplicate_from_the_comparison_api_keeps_policy_unresolved():
    comparison = compare_location_streams(_jobs(), same_both(), DEF)
    assert comparison.status is CS.LIKELY_DUPLICATE_STREAMS          # evidence produced by the real API
    report = assess_location_policy(UNRESOLVED, comparison)
    assert report.state is PS.UNRESOLVED and not report.locations_are_aliases
    assert not assess_pricing_readiness(location_policy=report, **GATES).ready


def test_even_identity_metadata_evidence_does_not_set_the_policy():
    report = assess_location_policy(UNRESOLVED, evidence(CS.CONFIRMED_ALIAS))
    assert report.state is PS.UNRESOLVED and not report.locations_are_aliases


# --------------------------------------------------------- confirmed distinct


def test_confirmed_distinct_allows_independent_comparison_and_keeps_authority():
    report = assess_location_policy(DISTINCT, evidence(CS.LIKELY_DUPLICATE_STREAMS))
    assert report.state is PS.CONFIRMED_DISTINCT and report.location_policy_resolved
    assert report.location_policy_authority_sufficient
    assert report.locations_are_aliases is False and report.locations_comparable_independently is True
    assert report.authority == AUTHORITY and report.authority.reference == "SYNTH-DECISION-001"
    assert report.canonicalization_required is False and report.blocking_reasons == ()
    assert assess_pricing_readiness(location_policy=report, **GATES).ready


def test_confirmed_distinct_keeps_source_labels_as_analytical_keys():
    keys = apply_location_policy(frame([A, B_]), DISTINCT)
    assert keys.analytical_keys.tolist() == [A, B_] and not keys.alias_mapping_applied


# ------------------------------------------------------------ confirmed alias


def test_confirmed_alias_requires_and_uses_canonical_grouping():
    df = frame([A, B_, C])
    before = df.copy(deep=True)
    keys = apply_location_policy(df, ALIAS)
    assert keys.alias_mapping_applied
    assert keys.analytical_keys.tolist() == [CANONICAL, CANONICAL, C]
    assert keys.source_keys.tolist() == [A, B_, C]                   # lineage preserved
    assert keys.analytical_keys.index.equals(df.index)
    pd.testing.assert_frame_equal(df, before)                         # raw labels untouched
    report = assess_location_policy(ALIAS, analytical_keys=keys)
    assert report.location_policy_resolved and report.locations_are_aliases is True
    assert report.locations_comparable_independently is False
    assert report.canonicalization_required and report.canonicalization_applied
    assert assess_pricing_readiness(location_policy=report, **GATES).ready


def test_confirmed_alias_without_applied_canonicalization_blocks_pricing():
    report = assess_location_policy(ALIAS)
    assert report.locations_are_aliases and not report.locations_comparable_independently
    readiness = assess_pricing_readiness(location_policy=report, **GATES)
    assert not readiness.ready and readiness.blocking_reasons == (B.ALIAS_CANONICALIZATION_NOT_APPLIED,)


def test_keys_built_under_another_policy_do_not_count_as_canonicalised():
    other = apply_location_policy(frame([A, B_]), UNRESOLVED)
    report = assess_location_policy(ALIAS, analytical_keys=other)
    assert not report.canonicalization_applied
    assert B.ALIAS_CANONICALIZATION_NOT_APPLIED in report.blocking_reasons


def test_unresolved_policy_never_merges_labels():
    keys = apply_location_policy(frame([A, B_]), UNRESOLVED)
    assert keys.analytical_keys.tolist() == [A, B_] and not keys.alias_mapping_applied
    assert not assess_location_policy(UNRESOLVED, analytical_keys=keys).canonicalization_applied


# ------------------------------------------------- invalid / contradictory config


@pytest.mark.parametrize("changes", [
    {"state": PS.CONFIRMED_ALIAS, "authority": AUTHORITY},                          # alias without canonical
    {"state": PS.CONFIRMED_ALIAS, "authority": AUTHORITY, "canonical_location": ("",)},
    {"state": PS.CONFIRMED_ALIAS, "authority": AUTHORITY, "canonical_location": ("SYNTH", "EXTRA")},
    {"state": PS.CONFIRMED_ALIAS, "canonical_location": CANONICAL},                 # alias without authority
    {"state": PS.CONFIRMED_DISTINCT},                                               # distinct without authority
    {"state": PS.CONFIRMED_DISTINCT, "authority": AUTHORITY, "canonical_location": CANONICAL},  # merges distinct
    {"authority": AUTHORITY},                                                       # unresolved with a decision
    {"canonical_location": CANONICAL},                                              # unresolved with a mapping
    {"state": "confirmed_distinct", "authority": AUTHORITY},
    {"authority": "SYNTH-AUTHORITY", "state": PS.CONFIRMED_DISTINCT},
    {"second": A},
    {"first": ("SYNTH-NOT-EXPECTED",)},
    {"coverage": dataclasses.replace(COV, expected_locations=None, mode=None)},
])
def test_incomplete_or_contradictory_policy_is_rejected(changes):
    with pytest.raises(LocationPolicyConfigurationError):
        dataclasses.replace(UNRESOLVED, **changes)


@pytest.mark.parametrize("kwargs", [{"source": ""}, {"source": "   "}, {"source": None},
                                    {"source": "SYNTH-AUTHORITY", "reference": ""},
                                    {"source": "SYNTH-AUTHORITY", "note": " "}])
def test_authority_metadata_must_be_complete(kwargs):
    with pytest.raises(LocationPolicyConfigurationError):
        LocationPolicyAuthority(**kwargs)


@pytest.mark.parametrize("policy, status", [(ALIAS, CS.CONFIRMED_DISTINCT_LOCATIONS),
                                            (DISTINCT, CS.CONFIRMED_ALIAS),
                                            (DISTINCT, CS.DUPLICATED_COLLECTION_CONFIGURATION)])
def test_identity_metadata_contradicting_policy_blocks_pricing(policy, status):
    keys = apply_location_policy(frame([A, B_]), policy)
    report = assess_location_policy(policy, evidence(status), keys)
    assert report.identity_evidence_conflict and not report.location_policy_authority_sufficient
    assert not report.locations_are_aliases and not report.locations_comparable_independently
    readiness = assess_pricing_readiness(location_policy=report, **GATES)
    assert not readiness.ready and B.IDENTITY_EVIDENCE_CONFLICT in readiness.blocking_reasons


# --------------------------------------------------- other gates stay effective


def test_resolved_policy_does_not_override_other_gates():
    report = assess_location_policy(DISTINCT)
    readiness = assess_pricing_readiness(location_policy=report, **(GATES | {
        "temporal_fields_trusted": False, "vehicle_stability": UNSTABLE}))
    assert readiness.ready is False
    assert readiness.blocking_reasons == (B.TEMPORAL_FIELDS_UNTRUSTED, B.VEHICLE_ATTRIBUTES_UNSTABLE)


@pytest.mark.parametrize("gate", sorted(GATES))
def test_each_foundational_gate_blocks_alone(gate):
    readiness = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                         **(GATES | {gate: FAILING[gate]}))
    assert not readiness.ready and len(readiness.blocking_reasons) == 1


def test_all_failures_are_reported_together():
    readiness = assess_pricing_readiness(location_policy=assess_location_policy(),
                                         **(FAILING | {"vehicle_stability": UNSTABLE_AND_PARTIAL}))
    assert set(readiness.blocking_reasons) == set(B) - {B.ALIAS_CANONICALIZATION_NOT_APPLIED,
                                                        B.IDENTITY_EVIDENCE_CONFLICT,
                                                        B.VEHICLE_STABILITY_UNAVAILABLE}
    assert readiness.blocking_reasons[-1] is B.LOCATION_POLICY_UNRESOLVED


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_missing_gate_results_cannot_pass(value):
    with pytest.raises(TypeError):
        assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                 **(GATES | {"temporal_fields_trusted": value}))
    with pytest.raises(TypeError):
        assess_pricing_readiness(location_policy=None, **GATES)  # type: ignore[arg-type]


# ------------------------------------------------ full-population vehicle stability


def test_stability_fixtures_cover_each_population_state():
    assert (STABLE.status, UNSTABLE.status, PARTIAL.status, UNSTABLE_AND_PARTIAL.status) == (
        VehicleStabilityStatus.PASSED, VehicleStabilityStatus.VIOLATIONS,
        VehicleStabilityStatus.PARTIALLY_ASSESSABLE, VehicleStabilityStatus.VIOLATIONS)


def test_partially_assessed_product_history_blocks_pricing():
    # Regression: one vehicle assessed, one under-observed, no violations - formerly PASSED.
    readiness = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                         **(GATES | {"vehicle_stability": PARTIAL}))
    assert readiness.ready is False and readiness.blocking_reasons == (B.VEHICLE_HISTORY_INSUFFICIENT,)


@pytest.mark.parametrize("rows, expected", [
    ([obs(V1, T[0]), obs(V2, T[0])], (B.VEHICLE_HISTORY_INSUFFICIENT,)),     # entirely unassessable
    ([], (B.VEHICLE_HISTORY_INSUFFICIENT,)),                                  # empty population
])
def test_unassessed_product_population_blocks_pricing(rows, expected):
    stability = assess_vehicle_attribute_stability(stability_frame(rows))
    readiness = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                         **(GATES | {"vehicle_stability": stability}))
    assert readiness.blocking_reasons == expected


def test_violations_and_insufficient_history_are_both_reported():
    readiness = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                         **(GATES | {"vehicle_stability": UNSTABLE_AND_PARTIAL}))
    assert readiness.blocking_reasons == (B.VEHICLE_ATTRIBUTES_UNSTABLE, B.VEHICLE_HISTORY_INSUFFICIENT)


def test_missing_stability_report_blocks_and_wrong_types_are_rejected():
    readiness = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                         **(GATES | {"vehicle_stability": None}))
    assert readiness.blocking_reasons == (B.VEHICLE_STABILITY_UNAVAILABLE,)
    for value in (True, "passed", STABLE.status):
        with pytest.raises(TypeError):
            assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                     **(GATES | {"vehicle_stability": value}))


def test_blocker_values_name_no_source_columns():
    from conftest import contract_columns
    from ql2_sixt_canada_analysis.schemas import DatasetKey

    columns = {c for key in DatasetKey for c in contract_columns(key)}
    assert not any(c in b.value for b in B for c in columns)


# ---------------------------------------------------- immutability / determinism


def test_reports_policies_and_keys_are_immutable_and_deterministic():
    report = assess_location_policy(DISTINCT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.state = PS.CONFIRMED_ALIAS  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.authority.source = "SYNTH-OTHER"  # type: ignore[misc, union-attr]
    with pytest.raises(dataclasses.FrozenInstanceError):
        VANCOUVER_LOCATION_POLICY.state = PS.CONFIRMED_DISTINCT  # type: ignore[misc]
    with pytest.raises(TypeError):
        ALIAS.alias_mapping[A] = A  # type: ignore[index]
    readiness = assess_pricing_readiness(location_policy=report, **GATES)
    with pytest.raises(dataclasses.FrozenInstanceError):
        readiness.blocking_reasons = ()  # type: ignore[misc]
    assert assess_location_policy(DISTINCT) == report
    assert assess_pricing_readiness(location_policy=report, **GATES) == readiness
    keys = apply_location_policy(frame([A, B_]), ALIAS)
    leaked = keys.analytical_keys
    leaked.iloc[0] = C
    assert keys.analytical_keys.tolist() == [CANONICAL, CANONICAL]
    source = keys.source_keys
    source.iloc[0] = C
    assert keys.source_keys.tolist() == [A, B_]


def test_type_errors_and_missing_location_columns():
    with pytest.raises(TypeError):
        assess_location_policy(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess_location_policy(UNRESOLVED, comparison=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        apply_location_policy([], UNRESOLVED)  # type: ignore[arg-type]
    with pytest.raises(LocationPolicyConfigurationError) as info:
        apply_location_policy(frame([A]).drop(columns=[LOC]), UNRESOLVED)
    assert "SYNTH" not in str(info.value)


def test_package_exports():
    for name in ("VANCOUVER_LOCATION_POLICY", "LocationPolicyState", "LocationIdentityPolicy",
                 "LocationPolicyAuthority", "assess_location_policy", "assess_pricing_readiness",
                 "validate_pricing_readiness", "apply_location_policy", "PricingBlocker",
                 "PricingReadinessReport", "LocationPolicyReport", "AnalyticalLocationKeys"):
        assert name in ql2_sixt_canada_analysis.__all__
    assert isinstance(assess_pricing_readiness(location_policy=assess_location_policy(), **GATES),
                      PricingReadinessReport)
    assert isinstance(assess_location_policy(), LocationPolicyReport)
    assert isinstance(apply_location_policy(frame([A]), UNRESOLVED), AnalyticalLocationKeys)
