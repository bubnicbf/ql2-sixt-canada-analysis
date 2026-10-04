"""Governed scope of the Vancouver location policy: an alias may never leave its city.

The policy keys, cities and labels are the configured project contract
(repository configuration); authority metadata, jobs and rows are fabricated
(``SYNTH-...``). No Vancouver identity decision is made here - every
resolved policy below is a synthetic test configuration.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from test_city_integrity import completeness, healthy
from test_readiness import GATES, evidence

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.comparison import LocationStreamComparisonStatus as CS
from ql2_sixt_canada_analysis.coverage import assess_expected_location_coverage
from ql2_sixt_canada_analysis.readiness import (
    LocationPolicyReport,
    PricingBlocker as B,
    PricingNotReadyError,
    apply_location_policy,
    assess_location_policy,
    assess_pricing_readiness,
    validate_pricing_readiness,
)
from ql2_sixt_canada_analysis.schemas import (
    COMPARED_LOCATION_STREAMS,
    EXPECTED_LOCATION_COVERAGE as COV,
    INVESTIGATED_LOCATION_STREAM,
    VANCOUVER_LOCATION_POLICY as VAN_POLICY,
    LocationPolicyAuthority,
    LocationPolicyConfigurationError,
    LocationPolicyScope,
    LocationPolicyScopeDefect as D,
    LocationPolicyState as PS,
    assess_location_policy_scope,
)
from ql2_sixt_canada_analysis.streams import assess_expected_location_streams

CAL_KEY = INVESTIGATED_LOCATION_STREAM                     # Calgary Downtown
DOWNTOWN, THURLOW = COMPARED_LOCATION_STREAMS              # the two governed Vancouver keys
CAL, VAN = CAL_KEY[0], DOWNTOWN[0]
CITY_COL, LABEL_COL = COV.location_columns
AUTHORITY = LocationPolicyAuthority(source="SYNTH-AUTHORITY", reference="SYNTH-DECISION-002")


def alias(canonical: tuple[object, ...]):  # type: ignore[no-untyped-def]
    return dataclasses.replace(VAN_POLICY, state=PS.CONFIRMED_ALIAS, authority=AUTHORITY,
                               canonical_location=canonical)


DISTINCT = dataclasses.replace(VAN_POLICY, state=PS.CONFIRMED_DISTINCT, authority=AUTHORITY)
VALID_ALIAS = alias(DOWNTOWN)
TO_CALGARY = alias(CAL_KEY)                                 # the reported defect
CROSS_CITY = (D.CANONICAL_LOCATION_CITY_MISMATCH, D.CANONICAL_LOCATION_IMPERSONATES_STREAM,
              D.CANONICAL_LOCATION_SCOPE_MISMATCH)


def keys_frame(keys: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({CITY_COL: [k[0] for k in keys], LABEL_COL: [k[1] for k in keys]},
                        index=[f"SYNTH-ROW-{i}" for i in range(len(keys))])


def bypass(policy, **fields):  # type: ignore[no-untyped-def]
    """A policy object whose construction-time validation was bypassed."""
    clone = dataclasses.replace(policy)
    for name, value in fields.items():
        object.__setattr__(clone, name, value)
    return clone


# ------------------------------------------------------- the central scope rule


def test_project_policy_governs_one_vancouver_scope():
    scope = VAN_POLICY.scope
    assert isinstance(scope, LocationPolicyScope) and scope.is_valid
    assert scope.governed_locations == (DOWNTOWN, THURLOW) and scope.governed_scope == (VAN,)
    assert scope.scope_columns == COV.stream_scope_columns and scope.canonical_location is None
    assert CAL != VAN and VAN_POLICY.state is PS.UNRESOLVED


def test_calgary_canonical_key_is_rejected_with_typed_defects():
    scope = assess_location_policy_scope(TO_CALGARY)
    assert scope.defects == CROSS_CITY and not scope.is_valid
    assert scope.governed_scope == (VAN,) and scope.canonical_location == CAL_KEY
    assert scope.canonical_scope == (CAL,)                       # the rejected city, for audit
    # Shape and non-blank values were never the problem: tuple validation alone is insufficient.
    assert isinstance(CAL_KEY, tuple) and len(CAL_KEY) == len(COV.location_columns)
    assert all(isinstance(v, str) and v.strip() for v in CAL_KEY)


@pytest.mark.parametrize("canonical, expected", [
    ((CAL, "SYNTH-BRANCH-Z"), (D.CANONICAL_LOCATION_CITY_MISMATCH, D.CANONICAL_LOCATION_SCOPE_MISMATCH)),
    (("SYNTH-CITY-9", DOWNTOWN[1]), (D.CANONICAL_LOCATION_CITY_MISMATCH, D.CANONICAL_LOCATION_SCOPE_MISMATCH)),
    ((VAN, "SYNTH-UNAUTHORISED-BRANCH"), (D.CANONICAL_LOCATION_SCOPE_MISMATCH,)),   # right city, unauthorised branch
    ((VAN, CAL_KEY[1]), (D.CANONICAL_LOCATION_SCOPE_MISMATCH,)),
])
def test_out_of_scope_canonical_keys_are_rejected(canonical, expected):
    policy = alias(canonical)
    assert policy.state is PS.CONFIRMED_ALIAS                     # declared state kept for audit
    assert policy.scope.defects == expected and dict(policy.alias_mapping) == {}


@pytest.mark.parametrize("bad", [None, "", "   ", "\t", f" {VAN}"])
def test_blank_or_padded_canonical_components_fail(bad):
    for canonical in ((bad, DOWNTOWN[1]), (VAN, bad)):
        if bad is not None and bad.strip():                        # padded: constructible, rejected by scope
            assert alias(canonical).scope.defects[0] is D.CANONICAL_LOCATION_MALFORMED
        else:
            with pytest.raises(LocationPolicyConfigurationError):   # structurally malformed: never constructed
                alias(canonical)
        assert assess_location_policy_scope(bypass(VALID_ALIAS, canonical_location=canonical)).defects[0] \
            is D.CANONICAL_LOCATION_MALFORMED


def test_governed_keys_spanning_two_cities_have_no_governed_scope():
    for state in (PS.UNRESOLVED, PS.CONFIRMED_DISTINCT, PS.CONFIRMED_ALIAS):
        changes = dict(second=CAL_KEY, state=state,
                       authority=None if state is PS.UNRESOLVED else AUTHORITY,
                       canonical_location=DOWNTOWN if state is PS.CONFIRMED_ALIAS else None)
        policy = dataclasses.replace(VAN_POLICY, **changes)
        assert policy.scope.governed_scope is None
        assert D.GOVERNED_SCOPE_AMBIGUOUS in policy.scope.defects
        report = assess_location_policy(policy)
        assert not report.location_policy_authority_sufficient
        assert not report.locations_are_aliases and not report.locations_comparable_independently
        assert B.GOVERNED_SCOPE_AMBIGUOUS in assess_pricing_readiness(location_policy=report, **GATES).blocking_reasons


def test_missing_scope_information_fails_closed():
    no_scope = dataclasses.replace(COV, stream_scope_columns=(), parent_scope_columns=())
    policy = dataclasses.replace(DISTINCT, coverage=no_scope)
    assert policy.scope.defects == (D.GOVERNED_SCOPE_UNAVAILABLE,)
    assert not assess_location_policy(policy).locations_comparable_independently
    assert assess_location_policy_scope(object()).defects[0] is D.GOVERNED_SCOPE_UNAVAILABLE


# ------------------------------------------------------ malformed policy result


def test_calgary_alias_is_recorded_but_grants_no_permission():
    keys = apply_location_policy(keys_frame([DOWNTOWN, THURLOW, CAL_KEY]), TO_CALGARY)
    report = assess_location_policy(TO_CALGARY, None, keys)
    assert report.state is PS.CONFIRMED_ALIAS and report.authority == AUTHORITY
    assert report.location_policy_resolved is True                 # a decision was configured...
    assert report.location_policy_authority_sufficient is False     # ...but it is not usable
    assert report.scope_valid is False and report.scope_defects == CROSS_CITY
    assert report.canonicalization_permitted is False and report.locations_are_aliases is False
    assert report.locations_comparable_independently is False
    assert report.canonicalization_applied is False
    assert report.blocking_reasons == tuple(B(d.value) for d in CROSS_CITY) + (B.ALIAS_CANONICALIZATION_NOT_APPLIED,)
    readiness = assess_pricing_readiness(location_policy=report, **GATES)      # every other gate passes
    assert readiness.ready is False
    assert B.CANONICAL_LOCATION_CITY_MISMATCH in readiness.blocking_reasons
    with pytest.raises(PricingNotReadyError) as info:
        validate_pricing_readiness(location_policy=report, **GATES)
    assert "canonical_key_crosses_governed_scope" in str(info.value)
    assert CAL not in str(info.value) and VAN not in str(info.value)


def test_no_vancouver_row_is_rewritten_and_nothing_is_partially_mapped():
    df = keys_frame([DOWNTOWN, THURLOW, CAL_KEY, DOWNTOWN])
    before = df.copy(deep=True)
    keys = apply_location_policy(df, TO_CALGARY)
    assert keys.canonicalization_refused and not keys.alias_mapping_applied
    assert keys.scope_defects == CROSS_CITY
    assert keys.analytical_keys.tolist() == keys.source_keys.tolist() == [DOWNTOWN, THURLOW, CAL_KEY, DOWNTOWN]
    assert keys.analytical_keys.tolist().count(CAL_KEY) == 1          # Calgary gains nothing
    pd.testing.assert_frame_equal(df, before)


def test_bypassed_policy_is_refused_at_the_application_boundary():
    # Construction would accept only well-formed values; force an invalid key in anyway.
    forged = bypass(VALID_ALIAS, canonical_location=CAL_KEY)
    assert dict(forged.alias_mapping) == {}
    keys = apply_location_policy(keys_frame([DOWNTOWN, THURLOW]), forged)
    assert keys.canonicalization_refused and keys.analytical_keys.tolist() == [DOWNTOWN, THURLOW]
    blank = bypass(VALID_ALIAS, canonical_location=(VAN, ""))
    assert apply_location_policy(keys_frame([DOWNTOWN]), blank).scope_defects == (D.CANONICAL_LOCATION_MALFORMED,)
    spanning = bypass(VALID_ALIAS, second=CAL_KEY)
    refused = apply_location_policy(keys_frame([DOWNTOWN, CAL_KEY]), spanning)
    assert refused.canonicalization_refused and refused.analytical_keys.tolist() == [DOWNTOWN, CAL_KEY]
    with pytest.raises(LocationPolicyConfigurationError):
        apply_location_policy(keys_frame([DOWNTOWN]), bypass(VALID_ALIAS, coverage=None))
    with pytest.raises(ValueError):                    # a report cannot claim applied canonicalisation
        LocationPolicyReport(state=PS.CONFIRMED_ALIAS, authority=AUTHORITY, behavioral_evidence=None,
                             identity_evidence_conflict=False, canonicalization_applied=True,
                             scope=forged.scope)


def test_a_report_without_scope_assessment_is_not_sufficient():
    report = LocationPolicyReport(state=PS.CONFIRMED_DISTINCT, authority=AUTHORITY, behavioral_evidence=None,
                                  identity_evidence_conflict=False, canonicalization_applied=False)
    assert report.scope is None and not report.location_policy_authority_sufficient
    assert report.blocking_reasons == (B.GOVERNED_SCOPE_UNAVAILABLE,)


def test_scope_mismatch_and_mapping_defect_are_both_reported():
    report = assess_location_policy(TO_CALGARY, evidence(CS.LOCATION_MAPPING_DEFECT))
    assert report.identity_evidence_conflict
    assert report.blocking_reasons == ((B.IDENTITY_EVIDENCE_CONFLICT,) + tuple(B(d.value) for d in CROSS_CITY)
                                       + (B.ALIAS_CANONICALIZATION_NOT_APPLIED,))
    distinct = assess_location_policy(bypass(DISTINCT, canonical_location=CAL_KEY), evidence(CS.LOCATION_MAPPING_DEFECT))
    assert distinct.blocking_reasons == (B.IDENTITY_EVIDENCE_CONFLICT, B.CANONICAL_LOCATION_NOT_PERMITTED)
    assert not distinct.locations_comparable_independently


# -------------------------------------------- downstream populations unchanged


def test_invalid_alias_changes_no_coverage_stream_or_completeness_population():
    j, c = healthy()                       # Calgary Downtown + both Vancouver streams, all healthy
    coverage = assess_expected_location_coverage(c, COV)
    streams = assess_expected_location_streams(j, c, coverage=COV)
    apply_location_policy(c, TO_CALGARY)   # refused; inputs untouched
    assert assess_expected_location_coverage(c, COV) == coverage
    again = assess_expected_location_streams(j, c, coverage=COV)
    assert again == streams and again.assessed_exactly_once
    assert [r.target for r in again.results] == [CAL_KEY, DOWNTOWN, THURLOW]   # Vancouver streams stay
    calgary = again.reports[CAL_KEY].event_accounting
    assert calgary.jobs_with_target_details == streams.reports[CAL_KEY].event_accounting.jobs_with_target_details
    report = completeness(j, c)
    assert report.complete and report.expected_streams.expected_stream_count == 3
    # Completeness is decided on source labels; pricing still needs a usable identity decision.
    readiness = assess_pricing_readiness(location_policy=assess_location_policy(TO_CALGARY),
                                         **(GATES | {"completeness": report}))
    assert not readiness.ready and B.CANONICAL_LOCATION_CITY_MISMATCH in readiness.blocking_reasons


def test_reported_issue_end_to_end():
    # Regression: a confirmed Vancouver alias with Calgary Downtown as canonical key used to
    # map both Vancouver labels to Calgary, stay authority-sufficient and allow pricing.
    j, c = healthy()
    keys = apply_location_policy(c, TO_CALGARY)
    assert CAL_KEY not in keys.analytical_keys.tolist()[1:]           # the Vancouver rows stay Vancouver
    assert keys.analytical_keys.tolist() == keys.source_keys.tolist()
    policy = assess_location_policy(TO_CALGARY, None, keys)
    assert not policy.location_policy_authority_sufficient and not policy.canonicalization_permitted
    complete = completeness(j, c)
    assert [r.target for r in complete.expected_streams.results] == [CAL_KEY, DOWNTOWN, THURLOW]
    readiness = assess_pricing_readiness(location_policy=policy, **(GATES | {"completeness": complete}))
    assert readiness.ready is False
    assert set(readiness.blocking_reasons) == {B(d.value) for d in CROSS_CITY} | {B.ALIAS_CANONICALIZATION_NOT_APPLIED}


# ------------------------------------------------------- valid states preserved


def test_valid_vancouver_alias_maps_both_labels_and_leaves_calgary_alone():
    for canonical in (DOWNTOWN, THURLOW):
        policy = alias(canonical)
        assert policy.scope.is_valid and dict(policy.alias_mapping) == {DOWNTOWN: canonical, THURLOW: canonical}
        keys = apply_location_policy(keys_frame([DOWNTOWN, CAL_KEY, THURLOW]), policy)
        assert keys.alias_mapping_applied and not keys.canonicalization_refused
        assert keys.analytical_keys.tolist() == [canonical, CAL_KEY, canonical]
        report = assess_location_policy(policy, None, keys)
        assert report.location_policy_authority_sufficient and report.canonicalization_permitted
        assert report.blocking_reasons == ()
        assert assess_pricing_readiness(location_policy=report, **GATES).ready


def test_valid_distinct_policy_keeps_streams_separate():
    keys = apply_location_policy(keys_frame([DOWNTOWN, THURLOW]), DISTINCT)
    assert not keys.alias_mapping_applied and keys.analytical_keys.tolist() == [DOWNTOWN, THURLOW]
    report = assess_location_policy(DISTINCT, None, keys)
    assert report.scope_valid and report.locations_comparable_independently and not report.canonicalization_permitted
    assert assess_pricing_readiness(location_policy=report, **GATES).ready


def test_unresolved_policy_remains_blocked():
    report = assess_location_policy()
    assert report.scope_valid and not report.location_policy_resolved
    assert report.blocking_reasons == (B.LOCATION_POLICY_UNRESOLVED,)
    assert not assess_pricing_readiness(location_policy=report, **GATES).ready


def test_distinct_policy_may_not_carry_a_canonical_key():
    with pytest.raises(LocationPolicyConfigurationError):
        dataclasses.replace(DISTINCT, canonical_location=DOWNTOWN)
    forged = bypass(DISTINCT, canonical_location=CAL_KEY)
    report = assess_location_policy(forged)
    assert report.scope_defects == (D.CANONICAL_LOCATION_NOT_PERMITTED,)
    assert not report.locations_comparable_independently
    assert not apply_location_policy(keys_frame([DOWNTOWN]), forged).alias_mapping_applied


# ------------------------------------------------------ determinism and safety


def test_diagnostics_are_deterministic_and_order_independent():
    first = apply_location_policy(keys_frame([DOWNTOWN, THURLOW, CAL_KEY]), TO_CALGARY)
    second = apply_location_policy(keys_frame([CAL_KEY, THURLOW, DOWNTOWN]), TO_CALGARY)
    assert first.scope_defects == second.scope_defects == CROSS_CITY
    assert TO_CALGARY.scope == TO_CALGARY.scope == assess_location_policy_scope(TO_CALGARY)
    assert assess_location_policy(TO_CALGARY) == assess_location_policy(TO_CALGARY)
    with pytest.raises(dataclasses.FrozenInstanceError):
        TO_CALGARY.scope.defects = ()  # type: ignore[misc]


def test_defect_values_name_no_columns_and_are_exported():
    from conftest import contract_columns
    from ql2_sixt_canada_analysis.schemas import DatasetKey

    columns = {col for key in DatasetKey for col in contract_columns(key)}
    assert not any(col in d.value for d in D for col in columns)
    assert all(B(d.value).name == d.name for d in D)                 # one typed blocker per defect
    for name in ("LocationPolicyScope", "LocationPolicyScopeDefect", "assess_location_policy_scope"):
        assert name in ql2_sixt_canada_analysis.__all__
