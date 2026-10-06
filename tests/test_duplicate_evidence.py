"""Duplicate-stream inference counts only valid evidence, for the target pair and the baseline.

Fabricated rows only (``SYNTH-...``), built with the comparison fixtures.
Two missing prices are not equal evidence; a baseline must meet the same
evidence standard as the target pair. Behavioural results stay evidence:
nothing here resolves the Vancouver policy.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from test_comparison import (
    A, B, C, D, DEF, J1, J2, J3, PRICE, PRODUCT, SCOPE, _cars, _jobs, offer, same_both,
)
from test_readiness import GATES

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.comparison import (
    INVALID_OFFER_SAMPLE_LIMIT,
    DuplicateInferenceBlocker as DB,
    LocationStreamComparisonStatus as S,
    OfferDefect,
    OfferSetResult as O,
    OfferStreamRole as R,
    ScopeBaseline,
    TemporalOverlap,
    compare_location_streams,
    offer_signature,
)
from ql2_sixt_canada_analysis.readiness import apply_location_policy, assess_location_policy, assess_pricing_readiness
from ql2_sixt_canada_analysis.schemas import (
    LOCATION_STREAM_COMPARISON,
    VANCOUVER_LOCATION_POLICY,
    LocationCoverageConfigurationError,
    LocationPolicyState,
)

NUMERIC = DEF.numeric_columns[0]                 # the numeric price (project: price_num)
TEXT_PRICE = next(c for c in PRICE if c != NUMERIC)
J4 = "SYNTH-JOB-004"


def comparator(jobs, car: str = "SYNTH-CAR-Y", **fields):  # type: ignore[no-untyped-def]
    """Branch C rows that differ from the targets (a discriminating comparator)."""
    return [(j, C, offer(car) | fields) for j in jobs]


def run(rows, definition=DEF):  # type: ignore[no-untyped-def]
    return compare_location_streams(_jobs((J1, J2, J3, J4)), _cars(rows), definition)


def targets(jobs=(J1, J2), a=None, b=None):  # type: ignore[no-untyped-def]
    a, b = a or offer(), b or offer()
    return [(j, label, o) for j in jobs for label, o in ((A, a), (B, b))]


MISSING_PRICE = offer() | {c: None for c in PRICE}


def assert_not_duplicate(r):  # type: ignore[no-untyped-def]
    assert r.status is not S.LIKELY_DUPLICATE_STREAMS
    assert not r.duplicate_inference_permitted and not r.duplicate_evidence_sufficient


# ------------------------------------------------------------- offer signature


@pytest.mark.parametrize("column", [*PRODUCT, *PRICE])
@pytest.mark.parametrize("value, reason", [
    (None, OfferDefect.MISSING_VALUE), (np.nan, OfferDefect.MISSING_VALUE), (pd.NA, OfferDefect.MISSING_VALUE),
    (pd.NaT, OfferDefect.MISSING_VALUE), ("", OfferDefect.BLANK_TEXT), ("   ", OfferDefect.BLANK_TEXT),
])
def test_missing_or_blank_signature_fields_are_invalid(column, value, reason):
    sig = offer_signature(offer() | {column: value}, DEF)
    assert not sig.is_valid and sig.values is None
    assert sig.invalid_fields == (column,) and sig.reasons == (reason,)


@pytest.mark.parametrize("value", ["abc", "1O.00", " 10.00", "-1", "nan", "inf", float("inf"), -0.5, True, object()])
def test_malformed_numeric_price_is_invalid(value):
    sig = offer_signature(offer() | {NUMERIC: value}, DEF)
    assert sig.invalid_fields == (NUMERIC,) and sig.reasons == (OfferDefect.INVALID_NUMBER,)


@pytest.mark.parametrize("value", [0, 0.0, "0", "0.00", 10, 12.5, np.float64(3.0), "45.10"])
def test_valid_numbers_including_zero_are_accepted(value):
    sig = offer_signature(offer() | {NUMERIC: value}, DEF)
    assert sig.is_valid and sig.values is not None and NUMERIC not in sig.invalid_fields
    assert offer_signature(offer() | {TEXT_PRICE: "0"}, DEF).is_valid      # textual zero is non-blank text


def test_two_none_prices_are_not_matching_signatures():
    first, second = offer_signature(MISSING_PRICE, DEF), offer_signature(MISSING_PRICE, DEF)
    assert first.values is None and second.values is None              # no comparable value exists
    assert first.invalid_fields == second.invalid_fields == PRICE
    both = offer_signature(offer() | {PRODUCT[0]: None, NUMERIC: None}, DEF)
    assert both.invalid_fields == (PRODUCT[0], NUMERIC)                 # every reason kept


def test_project_signature_contract():
    assert LOCATION_STREAM_COMPARISON.numeric_columns == ("price_num",)
    assert set(LOCATION_STREAM_COMPARISON.numeric_columns) <= set(LOCATION_STREAM_COMPARISON.price_columns)
    for bad in (("synth_unknown",), ("price_num", "price_num"), "price_num"):
        with pytest.raises(LocationCoverageConfigurationError):
            dataclasses.replace(LOCATION_STREAM_COMPARISON, numeric_columns=bad)


# ------------------------------------------------- capture-level fail-closed rule


def test_missing_prices_do_not_make_captures_identical():
    r = run(targets(a=MISSING_PRICE, b=MISSING_PRICE) + comparator((J1, J2)))
    assert_not_duplicate(r)
    assert r.status is S.COMPARISON_UNASSESSABLE
    assert (r.paired_capture_count, r.eligible_paired_capture_count, r.invalid_paired_capture_count) == (2, 0, 2)
    assert r.matching_paired_capture_count == r.differing_paired_capture_count == 0
    assert DB.INVALID_OFFER_EVIDENCE in r.duplicate_inference_blockers
    assert DB.INSUFFICIENT_PAIRED_CAPTURES in r.duplicate_inference_blockers
    assert r.invalid_offer_fields == PRICE and r.invalid_offer_reasons == (OfferDefect.MISSING_VALUE,)
    assert r.price_aware_offers is O.UNAVAILABLE and r.synchronized_prices is None


@pytest.mark.parametrize("bad", [{PRODUCT[0]: None}, {PRODUCT[1]: "  "}, {NUMERIC: np.nan},
                                 {NUMERIC: "n/a"}, {PRODUCT[0]: None, NUMERIC: None}])
def test_invalid_product_or_price_never_supports_a_duplicate(bad):
    r = run(targets(a=offer() | bad, b=offer() | bad) + comparator((J1, J2)))
    assert_not_duplicate(r)
    assert r.eligible_paired_capture_count == 0 and not r.offer_validity_sufficient


def test_capture_with_only_invalid_offers_is_not_an_empty_identical_capture():
    r = run([(J1, A, MISSING_PRICE), (J1, B, MISSING_PRICE), (J2, A, MISSING_PRICE), (J2, B, MISSING_PRICE)])
    assert r.status is S.COMPARISON_UNASSESSABLE and r.matching_paired_capture_count == 0
    assert r.product_sets is O.UNAVAILABLE


def test_invalid_offers_are_not_dropped_to_leave_identical_subsets():
    # Valid offers agree; one extra invalid offer on one side must not vanish.
    rows = targets(jobs=(J1, J2, J3)) + [(J3, A, MISSING_PRICE)] + comparator((J1, J2, J3))
    r = run(rows)
    assert_not_duplicate(r)
    assert (r.eligible_paired_capture_count, r.invalid_paired_capture_count) == (2, 1)
    assert (r.first_side_invalid_pair_count, r.second_side_invalid_pair_count) == (1, 0)
    assert r.invalid_evidence_streams == (R.FIRST,)
    assert DB.INVALID_OFFER_EVIDENCE in r.duplicate_inference_blockers
    assert r.status is S.COMPARISON_INCONCLUSIVE


def test_one_side_invalid_makes_the_pair_ineligible_without_imputation():
    r = run([(J1, A, offer()), (J1, B, MISSING_PRICE), (J2, A, offer()), (J2, B, offer())] + comparator((J1, J2)))
    assert (r.eligible_paired_capture_count, r.invalid_paired_capture_count) == (1, 1)
    assert (r.first_side_invalid_pair_count, r.second_side_invalid_pair_count) == (0, 1)
    assert r.invalid_evidence_streams == (R.SECOND,)
    assert_not_duplicate(r)


def test_invalid_captures_do_not_count_toward_the_minimum():
    # Three temporal pairs (>= minimum of two) but only one eligible pair.
    rows = targets(jobs=(J1,)) + targets(jobs=(J2, J3), a=MISSING_PRICE, b=MISSING_PRICE) + comparator((J1, J2, J3))
    r = run(rows)
    assert r.paired_capture_count == 3 >= r.minimum_paired_captures
    assert r.eligible_paired_capture_count == 1 and not r.target_threshold_passed
    assert DB.INSUFFICIENT_PAIRED_CAPTURES in r.duplicate_inference_blockers
    assert_not_duplicate(r)


def test_invalid_captures_are_not_discriminatory_mismatches():
    # Products differ only through a missing value: no differing evidence, so not "distinct".
    rows = [(J1, A, offer()), (J1, B, offer() | {PRODUCT[0]: None}),
            (J2, A, offer()), (J2, B, offer() | {PRODUCT[0]: None})]
    r = run(rows)
    assert r.differing_paired_capture_count == 0 and r.status is S.COMPARISON_UNASSESSABLE
    assert r.status is not S.LIKELY_DISTINCT_STREAMS


def test_multiple_invalid_reasons_are_all_reported():
    rows = targets(a=offer() | {PRODUCT[0]: None, NUMERIC: "abc"}, b=offer() | {PRODUCT[1]: " "}) + comparator((J1, J2))
    r = run(rows)
    assert r.invalid_offer_fields == (PRODUCT[0], PRODUCT[1], NUMERIC)
    assert r.invalid_offer_reasons == (OfferDefect.MISSING_VALUE, OfferDefect.BLANK_TEXT, OfferDefect.INVALID_NUMBER)
    assert r.invalid_evidence_streams == (R.FIRST, R.SECOND)
    assert {s.role for s in r.invalid_offer_sample} == {R.FIRST, R.SECOND}


# ----------------------------------------------------------- the baseline standard


def test_valid_target_and_qualified_baseline_still_infers_likely_duplicate():
    r = run(targets() + comparator((J1, J2)))
    assert r.status is S.LIKELY_DUPLICATE_STREAMS and r.duplicate_inference_permitted
    assert r.target_threshold_passed and r.baseline_threshold_passed and r.baseline_discriminative
    assert r.temporal_pairing_sufficient and r.offer_validity_sufficient
    ev = r.baseline_evidence
    assert ev.qualified_comparator_pair_count >= 1 and ev.minimum_paired_captures == r.minimum_paired_captures
    assert ev.invalid_paired_capture_count == 0 and ev.discriminatory_eligible_capture_count >= 2


def test_one_partially_overlapping_comparator_capture_cannot_establish_the_baseline():
    r = run(targets() + comparator((J1,)))
    assert r.scope_baseline is ScopeBaseline.INSUFFICIENT and not r.baseline_threshold_passed
    assert r.duplicate_inference_blockers == (DB.BASELINE_EVIDENCE_INSUFFICIENT,)
    assert_not_duplicate(r)


def test_one_fully_overlapping_comparator_capture_is_below_the_minimum():
    # Every stream has one capture; the comparator overlaps completely but has one pair.
    r = run(targets(jobs=(J1,)) + comparator((J1,)))
    assert r.scope_baseline is ScopeBaseline.INSUFFICIENT
    assert r.baseline_evidence.paired_capture_count >= 1 and r.baseline_evidence.qualified_comparator_pair_count == 0
    assert_not_duplicate(r)


def test_baseline_with_enough_pairs_but_too_few_valid_ones_is_insufficient():
    rows = targets() + [(J1, C, offer("SYNTH-CAR-Y")), (J2, C, offer("SYNTH-CAR-Y") | {NUMERIC: None})]
    r = run(rows)
    assert r.scope_baseline is ScopeBaseline.INSUFFICIENT
    assert r.baseline_evidence.invalid_paired_capture_count >= 1
    assert R.BASELINE in r.invalid_evidence_streams and R.FIRST not in r.invalid_evidence_streams
    assert {s.role for s in r.invalid_offer_sample} == {R.BASELINE}       # diagnosed as baseline invalidity
    assert r.first_invalid_offer_row_count == r.second_invalid_offer_row_count == 0
    assert r.baseline_invalid_offer_row_count == 1
    assert_not_duplicate(r)


def test_baseline_with_enough_valid_pairs_but_incomplete_overlap_is_insufficient():
    # C shares J1 and J2 with A/B, but A/B also have J3 that C lacks (partial overlap for every comparator).
    rows = targets(jobs=(J1, J2, J3)) + comparator((J1, J2))
    r = run(rows)
    assert r.baseline_evidence.eligible_paired_capture_count >= 2
    assert r.scope_baseline is ScopeBaseline.INSUFFICIENT and not r.baseline_threshold_passed
    assert_not_duplicate(r)


def test_qualified_comparator_with_one_differing_capture_is_discriminative():
    rows = targets() + [(J1, C, offer()), (J2, C, offer("SYNTH-CAR-Y"))]    # identical once, differs once
    r = run(rows)
    assert r.scope_baseline is ScopeBaseline.DISCRIMINATIVE and r.status is S.LIKELY_DUPLICATE_STREAMS


def test_target_passing_with_failing_baseline_is_not_duplicate():
    r = run(targets() + comparator((J1, J2), **{NUMERIC: None}))
    assert r.target_threshold_passed and r.offer_validity_sufficient
    assert not r.baseline_threshold_passed and r.scope_baseline is ScopeBaseline.INSUFFICIENT
    assert_not_duplicate(r)


def test_baseline_passing_with_failing_target_is_not_duplicate():
    r = run(targets(jobs=(J1,)) + targets(jobs=(J2,), a=MISSING_PRICE) + comparator((J1, J2)))
    assert r.baseline_threshold_passed and r.scope_baseline is ScopeBaseline.DISCRIMINATIVE
    assert not r.target_threshold_passed
    assert_not_duplicate(r)


def test_identical_comparator_still_marks_the_scope_non_discriminative():
    r = run(targets() + [(J1, C, offer()), (J2, C, offer())])
    assert r.scope_baseline is ScopeBaseline.NON_DISCRIMINATIVE
    assert_not_duplicate(r)


def test_existing_distinct_scenario_is_unchanged():
    r = run([(J1, A, offer("SYNTH-CAR-X")), (J1, B, offer("SYNTH-CAR-Y")),
             (J2, A, offer("SYNTH-CAR-X")), (J2, B, offer("SYNTH-CAR-Z"))])
    assert r.status is S.LIKELY_DISTINCT_STREAMS and r.eligible_paired_capture_count == 2


# ------------------------------------------------------------- determinism


def test_results_and_diagnostics_do_not_depend_on_row_order():
    rows = (targets(jobs=(J1, J2, J3)) + [(J3, A, MISSING_PRICE), (J2, B, offer() | {PRODUCT[0]: " "})]
            + comparator((J1, J2, J3), **{NUMERIC: "x"}))
    forward = run(rows)
    backward = run(list(reversed(rows)))
    for name in ("status", "duplicate_inference_blockers", "scope_baseline", "invalid_paired_capture_count",
                 "invalid_offer_fields", "invalid_offer_reasons", "baseline_evidence"):
        assert getattr(forward, name) == getattr(backward, name), name
    assert [(s.role, s.invalid_fields) for s in forward.invalid_offer_sample] == [
        (s.role, s.invalid_fields) for s in backward.invalid_offer_sample]
    order = [list(R).index(s.role) for s in forward.invalid_offer_sample]
    assert order == sorted(order)


def test_invalid_offer_sample_is_bounded_and_kept_out_of_repr():
    jobs = tuple(f"SYNTH-JOB-{i:03d}" for i in range(1, INVALID_OFFER_SAMPLE_LIMIT + 4))
    rows = [(j, label, MISSING_PRICE) for j in jobs for label in (A, B)]
    r = compare_location_streams(_jobs(jobs), _cars(rows), DEF)
    assert len(r.invalid_offer_sample) == INVALID_OFFER_SAMPLE_LIMIT
    assert r.first_invalid_offer_row_count == len(jobs)
    assert "SYNTH" not in repr(r)


# --------------------------------------------------------- full regressions


def test_reported_failure_missing_prices_with_weak_baseline():
    # Regression: two fully paired captures with missing prices plus one partially
    # overlapping comparator capture used to return LIKELY_DUPLICATE_STREAMS.
    r = run(targets(a=MISSING_PRICE, b=MISSING_PRICE) + comparator((J1,)))
    assert r.temporal_overlap is TemporalOverlap.COMPLETE and r.paired_capture_count == 2
    assert_not_duplicate(r)
    assert r.eligible_paired_capture_count == 0 and r.scope_baseline is not ScopeBaseline.DISCRIMINATIVE


def test_reported_failure_valid_target_with_one_partial_comparator_capture():
    # Regression: identical valid target captures plus one partially overlapping comparator
    # capture made the baseline DISCRIMINATIVE and the result LIKELY_DUPLICATE_STREAMS.
    r = run(targets() + comparator((J1,)))
    assert r.target_threshold_passed and r.matching_paired_capture_count == 2
    assert r.scope_baseline is ScopeBaseline.INSUFFICIENT
    assert_not_duplicate(r)


# ------------------------------------------- evidence never becomes authority


def test_insufficient_or_invalid_duplicate_evidence_resolves_nothing():
    for rows in (targets(a=MISSING_PRICE, b=MISSING_PRICE) + comparator((J1,)), targets() + comparator((J1,))):
        r = run(rows)
        undecided = dataclasses.replace(VANCOUVER_LOCATION_POLICY, state=LocationPolicyState.UNRESOLVED, authority=None,
                                    canonical_location=None)
        policy = assess_location_policy(undecided, r)               # evidence cannot resolve an undecided policy
        assert policy.state is LocationPolicyState.UNRESOLVED and not policy.location_policy_resolved
        approved = assess_location_policy(VANCOUVER_LOCATION_POLICY, r)   # nor change the approved decision
        assert approved.state is VANCOUVER_LOCATION_POLICY.state and approved.authority == VANCOUVER_LOCATION_POLICY.authority
        assert not policy.location_policy_authority_sufficient and not policy.canonicalization_permitted
        keys = apply_location_policy(pd.DataFrame({c: [k] for c, k in zip(
            VANCOUVER_LOCATION_POLICY.coverage.location_columns, VANCOUVER_LOCATION_POLICY.first)}), undecided)
        assert not keys.alias_mapping_applied
        assert not assess_pricing_readiness(location_policy=policy, **GATES).ready


def test_new_values_name_no_columns_and_are_exported():
    from conftest import contract_columns
    from ql2_sixt_canada_analysis.schemas import DatasetKey

    columns = {c for key in DatasetKey for c in contract_columns(key)}
    for enum in (DB, OfferDefect, R, ScopeBaseline):
        assert not any(c in m.value for m in enum for c in columns), enum
    for name in ("offer_signature", "OfferSignature", "OfferDefect", "OfferStreamRole", "InvalidOfferSample",
                 "BaselineEvidence", "INVALID_OFFER_SAMPLE_LIMIT"):
        assert name in ql2_sixt_canada_analysis.__all__
