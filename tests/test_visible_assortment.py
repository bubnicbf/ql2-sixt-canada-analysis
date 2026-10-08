"""Visible assortment: the calculation engine of data-plan Section 5.

Every observation is fabricated: ``SYNTH-*`` products, synthetic branches,
synthetic 2030 capture periods, rental dates, prices and policies. The only
committed values read are approved configuration (stream keys, roles, schedule
zones and the canonical-offer policy) through the shared synthetic pipeline
world of ``test_price_change_events``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import math
import os
import random
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_price_change_events import (
    AUTHORITY,
    CAL_AIR,
    CAL_DOWN,
    ELSEWHERE,
    LOC,
    OTHER,
    POLICY,
    STREAMS,
    TOR_DOWN,
    VAN_DOWN,
    VAN_THUR,
    P,
    at,
    classify,
    frame,
    offer,
    pipeline_result,
    readiness_for,
    synthetic_world,
    timeline,
)

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis import price_change_events as pce
from ql2_sixt_canada_analysis import visible_assortment as va
from ql2_sixt_canada_analysis.assortment_contract import (
    ASSORTMENT_BREAK_REASONS,
    ASSORTMENT_TIMELINE_COLUMNS,
    DEFAULT_ASSORTMENT_DEFINITION as D,
    DEFAULT_UNUSUAL_DROP_POLICY,
    AnomalyPolicyStatus as APS,
    AnomalyPolicyUnavailableError,
    AssessabilityStatus as AS,
    AssortmentBlocker as B,
    AssortmentComparison,
    AssortmentContractError,
    AssortmentInterpretation,
    DenominatorStatus as DS,
    UnusualDropPolicy,
)
from ql2_sixt_canada_analysis.price_change_events import (
    CANDIDATE_COLUMNS,
    EVENT_INTERVAL_COLUMNS,
    FORBIDDEN_TIMESTAMP_SOURCES,
    CaptureState as CS,
    IntervalBreak as IB,
    PriceChangeBlocker as PB,
    PriceChangeContractError,
)
from ql2_sixt_canada_analysis.pricing_population import DetailEligibility as E
from ql2_sixt_canada_analysis.readiness import PricingBlocker
from ql2_sixt_canada_analysis.schemas import CONFIDENTIAL_TECHNICAL_COLUMNS
from ql2_sixt_canada_analysis.visible_assortment import (
    MEMBERSHIP_COLUMNS,
    RENTAL_CONTEXT_CHANGED,
    AssortmentCaptureError,
    AssortmentCounts,
    AssortmentIdentityError,
    AssortmentLocationError,
    AssortmentPriceEvidenceError,
    AssortmentReconciliationError,
    AssortmentReport,
    AssortmentStatus,
    LocationAssortmentSummary,
    ProductMembership as M,
    RentalContextError,
    VisibleAssortmentResult,
    assess_visible_assortment,
    calculate_visible_assortment,
    run_visible_assortment,
    validate_price_coincidence,
    visible_assortment_from_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
PREV, CUR = EVENT_INTERVAL_COLUMNS
SYNTH_STREAM = ("synth-city", "SYNTH Second Stream")


# ============================================================================ pure-engine helpers


def car(h: int, name: str, cents: int = 5000, loc=LOC, **changes) -> dict:  # type: ignore[no-untyped-def]
    return offer(P(h), cents, loc=loc, name=name, **changes)


def sets(spec: dict[int, str], loc=LOC) -> list[dict]:  # type: ignore[no-untyped-def]
    """``{hour: "ABC"}`` -> one offer per letter (products ``SYNTH Car <letter>``) at that hour."""
    return [car(h, f"SYNTH Car {letter}", loc=loc) for h, letters in spec.items() for letter in letters]


def unusual_count(timeline_frame: pd.DataFrame, policy: UnusualDropPolicy) -> int | None:
    if policy.status is not APS.APPROVED:
        return None
    return int(timeline_frame["unusual_drop"].fillna(False).astype(bool).sum())


def build(rows, timelines, locations=None, policy=DEFAULT_UNUSUAL_DROP_POLICY, offers=None):  # type: ignore[no-untyped-def]
    """The pure engine on synthetic offers, wrapped in a fully validated completed result."""
    prices = classify(rows, timelines, locations)
    tl, membership, summaries = calculate_visible_assortment(frame(rows) if offers is None else offers, prices,
                                                             policy=policy)
    report = AssortmentReport(status=AssortmentStatus.COMPLETED, offers_assessed=len(rows), locations=summaries,
                              overall=sum((s.counts for s in summaries), AssortmentCounts()),
                              anomaly_policy_status=policy.status, unusual_drop_intervals=unusual_count(tl, policy))
    return VisibleAssortmentResult(report, tl, membership, tuple(timelines), prices)


def row(result: VisibleAssortmentResult, h: int, loc=LOC) -> dict:  # type: ignore[no-untyped-def]
    t = result.timeline
    mine = t[(t["canonical_city"] == loc[0]) & (t["canonical_location"] == loc[1])
             & (t["scheduled_capture_period"] == P(h))]
    assert len(mine) == 1
    return {k: (None if v is pd.NA else v) for k, v in mine.iloc[0].astype(object).to_dict().items()}


def members(result: VisibleAssortmentResult, h: int, loc=LOC) -> dict[str, str]:  # type: ignore[no-untyped-def]
    m = result.membership
    mine = m[(m["canonical_location"] == loc[1]) & (m[CUR] == P(h))]
    return dict(zip(mine["car_name"], mine["membership"]))


def counts(r: dict) -> tuple:
    return (r["previous_product_count"], r["returned_product_count"], r["retained_count"], r["addition_count"],
            r["removal_count"])


def approved(rule, record="SYNTH-record", reference="SYNTH-reference") -> UnusualDropPolicy:  # type: ignore[no-untyped-def]
    return UnusualDropPolicy(APS.APPROVED, record, reference, "SYNTH test policy (not authority)", rule)


def synthetic_floor_rule(minimum_absolute_drop: int, minimum_drop_rate: float):  # type: ignore[no-untyped-def]
    """A fabricated test rule (never a project default or real authority)."""
    if isinstance(minimum_absolute_drop, bool) or not isinstance(minimum_absolute_drop, int) \
            or minimum_absolute_drop < 1:
        raise ValueError("the synthetic absolute floor is a positive int")
    if not isinstance(minimum_drop_rate, float) or not 0.0 < minimum_drop_rate <= 1.0:
        raise ValueError("the synthetic rate floor lies in (0, 1]")

    def rule(c: AssortmentComparison) -> bool:
        return (c.absolute_drop >= minimum_absolute_drop and c.drop_rate is not None
                and c.drop_rate >= minimum_drop_rate)
    return rule


# ============================================================================ contract and input validation


def test_the_engine_consumes_the_prompt1_contract_without_redefinition() -> None:
    assert (va._LOCATION, va._CAPTURE, va._CONTEXT, va._PRODUCT) == (
        D.location_columns, D.capture_column, D.context_columns, D.product_columns)
    assert MEMBERSHIP_COLUMNS == (*D.location_columns, *D.context_columns, *D.product_columns, PREV, CUR,
                                  "membership")
    assert RENTAL_CONTEXT_CHANGED in ASSORTMENT_BREAK_REASONS and RENTAL_CONTEXT_CHANGED not in {b.value for b in IB}
    assert va._PRICE_CHANGES == frozenset(D.price_change_outcomes) == {"increase", "decrease"}
    result = build(sets({0: "A", 1: "A"}), [timeline([0, 1])])
    assert tuple(result.timeline.columns) == ASSORTMENT_TIMELINE_COLUMNS == D.timeline_columns
    assert not set(MEMBERSHIP_COLUMNS) & D.prohibited_columns
    assert not {"price_cents", "currency", "price_basis"} & set(MEMBERSHIP_COLUMNS)


@pytest.mark.parametrize("column", ["car_name", "seats", "pickup_date", "canonical_location",
                                    "scheduled_capture_period"])
def test_missing_required_columns_fail_closed(column) -> None:  # type: ignore[no-untyped-def]
    rows = sets({0: "A", 1: "A"})
    prices = classify(rows, [timeline([0, 1])])
    with pytest.raises(AssortmentIdentityError):
        calculate_visible_assortment(frame(rows).drop(columns=[column]), prices)


@pytest.mark.parametrize(("column", "value"), [
    ("car_name", None), ("car_name", ""), ("car_name", " SYNTH Car A"), ("car_type", float("nan")),
    ("seats", 5), ("bags", pd.NA), ("canonical_city", None), ("pickup_date", "2030-04-01"),
    ("return_date", pd.Timestamp("2030-04-03")), ("return_date", None)])
def test_missing_or_malformed_identity_fails_closed_without_repair(column, value) -> None:  # type: ignore[no-untyped-def]
    rows = sets({0: "AB", 1: "AB"})
    prices = classify(rows, [timeline([0, 1])])
    bad = frame(rows).astype(object)
    bad.at[0, column] = value
    with pytest.raises(AssortmentIdentityError):
        calculate_visible_assortment(bad, prices)


def test_identifiers_source_labels_prices_and_row_order_never_affect_the_result() -> None:
    rows = sets({0: "ABC", 1: "BCD", 2: "D"})
    base = build(rows, [timeline([0, 1, 2])])
    noisy = frame(rows).assign(job_id=[f"SYNTH-JOB-{i}" for i in range(len(rows))],
                               row_index=list(range(len(rows)))[::-1], scrape_date="SYNTH",
                               source_location_labels="SYNTH Other Label", observation_count=7)
    shuffled = noisy.sample(frac=1.0, random_state=11)
    other = build(rows, [timeline([0, 1, 2])], offers=shuffled)
    pd.testing.assert_frame_equal(base.timeline, other.timeline)
    pd.testing.assert_frame_equal(base.membership, other.membership)
    assert base.report == other.report


def test_inputs_are_never_mutated() -> None:
    rows = sets({0: "AB", 1: "BC"})
    prices = classify(rows, [timeline([0, 1])])
    offers = frame(rows)
    before_offers, before_candidates = offers.copy(deep=True), prices.candidates.copy(deep=True)
    calculate_visible_assortment(offers, prices)
    pd.testing.assert_frame_equal(offers, before_offers)
    pd.testing.assert_frame_equal(prices.candidates, before_candidates)


def test_unknown_canonical_locations_fail_closed() -> None:
    rows = sets({0: "A", 1: "A"})
    prices = classify(rows, [timeline([0, 1])])
    with pytest.raises(AssortmentLocationError):
        calculate_visible_assortment(frame([*rows, car(0, "SYNTH Car Z", loc=ELSEWHERE)]), prices)


@pytest.mark.parametrize("bad", [car(5, "SYNTH Car A"), car(1, "SYNTH Car A", **{"scheduled_capture_period": "x"})])
def test_offers_outside_an_eligible_capture_fail_closed(bad) -> None:  # type: ignore[no-untyped-def]
    rows = sets({0: "A", 2: "A"})
    prices = classify(rows, [timeline([0, 1, 2], states={1: CS.MISSING_CAPTURE})])
    with pytest.raises(AssortmentCaptureError):
        calculate_visible_assortment(frame([*rows, bad]), prices)
    with pytest.raises(AssortmentCaptureError):                                # a missing capture holds no set
        calculate_visible_assortment(frame([*rows, car(1, "SYNTH Car A")]), prices)


def test_duplicate_timeline_locations_and_malformed_grids_fail_closed() -> None:
    rows = sets({0: "A", 1: "A"})
    prices = classify(rows, [timeline([0, 1])])
    object.__setattr__(prices, "timelines", (prices.timelines[0], prices.timelines[0]))
    with pytest.raises(AssortmentCaptureError):
        calculate_visible_assortment(frame(rows), prices)
    with pytest.raises(PriceChangeContractError):                             # duplicate scheduled periods
        timeline([0, 0, 1])
    with pytest.raises(PriceChangeContractError):                             # malformed scheduled period
        pce.ScheduledCapture("2030-03-04 00:00", CS.ELIGIBLE, (LOC,))
    with pytest.raises(TypeError):
        calculate_visible_assortment(frame(rows), object())                   # type: ignore[arg-type]
    with pytest.raises(TypeError):
        calculate_visible_assortment(rows, classify(rows, [timeline([0, 1])]))  # type: ignore[arg-type]


def test_more_than_one_rental_context_in_a_capture_fails_closed() -> None:
    rows = [*sets({0: "A", 1: "A"}), car(1, "SYNTH Car B", ret=dt.date(2030, 4, 9))]
    prices = classify(rows, [timeline([0, 1])])
    with pytest.raises(RentalContextError):
        calculate_visible_assortment(frame(rows), prices)


# ============================================================================ product counting


def test_returned_products_are_distinct_product_identities_not_offer_rows() -> None:
    rows = [car(0, "SYNTH Car A"),
            car(1, "SYNTH Car A", 5000), car(1, "SYNTH Car A", 6100),                  # two prices: one product
            car(1, "SYNTH Car B", currency="CA$"), car(1, "SYNTH Car B", currency="US$"),  # two units: one product
            car(1, "SYNTH Car C", basis="week")]
    result = build(rows, [timeline([0, 1])])
    assert row(result, 0)["returned_product_count"] == 1
    assert counts(row(result, 1)) == (1, 3, 1, 2, 0)
    duplicated = pd.concat([frame(rows), frame(rows).iloc[[0, 1]]], ignore_index=True)   # exact duplicate rows
    again = build(rows, [timeline([0, 1])], offers=duplicated)
    pd.testing.assert_frame_equal(result.timeline, again.timeline)


def test_ambiguous_price_variants_count_once_and_are_never_price_changes() -> None:
    rows = [car(0, "SYNTH Car A", 5000), car(1, "SYNTH Car A", 5000), car(1, "SYNTH Car A", 9000)]
    result = build(rows, [timeline([0, 1])])
    assert set(result.price_changes.candidates["outcome"]) == {"ambiguous"}
    r = row(result, 1)
    assert counts(r) == (1, 1, 1, 0, 0) and (r["price_increase_count"], r["price_decrease_count"]) == (0, 0)
    assert r["price_change"] is False and r["assortment_change"] is False


def test_products_at_different_locations_are_isolated() -> None:
    rows = [*sets({0: "AB", 1: "AB"}), *sets({0: "C", 1: "CD"}, loc=OTHER)]
    result = build(rows, [timeline([0, 1]), timeline([0, 1], loc=OTHER)])
    assert counts(row(result, 1)) == (2, 2, 2, 0, 0)
    assert counts(row(result, 1, OTHER)) == (1, 2, 1, 1, 0)
    assert members(result, 1, OTHER) == {"SYNTH Car C": "retained", "SYNTH Car D": "added"}


def test_an_eligible_zero_offer_capture_is_an_empty_set_with_removals_then_additions() -> None:
    result = build(sets({0: "AB", 2: "A"}), [timeline([0, 1, 2])])
    empty, after = row(result, 1), row(result, 2)
    assert empty["capture_state"] == "eligible" and empty["assessability_status"] == AS.ASSESSED.value
    assert counts(empty) == (2, 0, 0, 0, 2) and empty["retention"] == 0.0 and empty["jaccard_similarity"] == 0.0
    assert counts(after) == (0, 1, 0, 1, 0)
    assert after["retention"] is None and after["retention_denominator_status"] == DS.ZERO_DENOMINATOR.value
    assert members(result, 1) == {"SYNTH Car A": "removed", "SYNTH Car B": "removed"}
    assert members(result, 2) == {"SYNTH Car A": "added"}
    assert result.report.overall.empty_captures == 1


def test_the_first_capture_is_a_seed_without_fabricated_changes() -> None:
    result = build(sets({0: "ABC", 1: "ABC"}), [timeline([0, 1])])
    seed = row(result, 0)
    assert seed["assessability_status"] == AS.SEED_CAPTURE.value and seed["returned_product_count"] == 3
    assert seed["has_previous_interval"] is False and seed[PREV] is None and seed["interval_break_reason"] is None
    assert all(seed[c] is None for c in va._INTERVAL_FIELDS)
    assert {seed[c] for c in ("retention_denominator_status", "jaccard_denominator_status",
                              "drop_rate_denominator_status")} == {DS.NOT_ASSESSABLE.value}
    assert members(result, 0) == {}


def test_canonical_alias_provenance_never_splits_a_product() -> None:
    rows = [car(0, "SYNTH Car A", labels="SYNTH Downtown|SYNTH Alias"), car(1, "SYNTH Car A", labels="SYNTH Alias")]
    result = build(rows, [timeline([0, 1])])
    assert counts(row(result, 1)) == (1, 1, 1, 0, 0) and members(result, 1) == {"SYNTH Car A": "retained"}


# ============================================================================ additions, removals, retention, Jaccard


@pytest.mark.parametrize(("before", "after", "expected", "retention", "jaccard"), [
    ("AB", "AB", (2, 2, 2, 0, 0), 1.0, 1.0),                       # stable
    ("A", "AB", (1, 2, 1, 1, 0), 1.0, 0.5),                        # pure addition
    ("AB", "A", (2, 1, 1, 0, 1), 0.5, 0.5),                        # pure removal
    ("ABC", "BCDE", (3, 4, 2, 2, 1), 2 / 3, 2 / 5),                # simultaneous additions and removals
    ("A", "B", (1, 1, 0, 1, 1), 0.0, 0.0),                         # complete replacement
    ("AB", "ACD", (2, 3, 1, 2, 1), 0.5, 0.25),                     # retention uses |P|, Jaccard |P | C|
])
def test_interval_set_differences_and_ratios_are_exact(before, after, expected, retention, jaccard) -> None:  # type: ignore[no-untyped-def]
    result = build(sets({0: before, 1: after}), [timeline([0, 1])])
    r = row(result, 1)
    assert counts(r) == expected and r["retention"] == retention and r["jaccard_similarity"] == jaccard
    p, c, kept, added, removed = expected
    assert r["net_change"] == added - removed == c - p and r["absolute_drop"] == max(p - c, 0)
    assert r["drop_rate"] == max(p - c, 0) / p and r["assortment_change"] is (added + removed > 0)
    assert r["retention"] != kept / c or kept == p or c == p                    # never the current-count share
    assert sorted(members(result, 1).values()) == sorted(
        ["retained"] * kept + ["added"] * added + ["removed"] * removed)


def test_a_product_that_disappears_and_returns_changes_only_across_valid_adjacent_intervals() -> None:
    adjacent = build(sets({0: "AB", 1: "B", 2: "AB"}), [timeline([0, 1, 2])])
    assert members(adjacent, 1)["SYNTH Car A"] == "removed" and members(adjacent, 2)["SYNTH Car A"] == "added"
    broken = build(sets({0: "AB", 2: "AB", 3: "AB"}), [timeline([0, 1, 2, 3], states={1: CS.MISSING_CAPTURE})])
    assert members(broken, 1) == members(broken, 2) == {}
    assert row(broken, 2)["assessability_status"] == AS.INTERVAL_BREAK.value
    assert members(broken, 3) == {"SYNTH Car A": "retained", "SYNTH Car B": "retained"}


def test_empty_previous_current_and_both_sets_have_explicit_denominator_states() -> None:
    result = build(sets({1: "A", 3: "A"}), [timeline([0, 1, 2, 3, 4, 5])])
    first_add, emptied, refill, both_empty = row(result, 1), row(result, 2), row(result, 3), row(result, 5)
    assert (first_add["retention"], first_add["retention_denominator_status"]) == (None, DS.ZERO_DENOMINATOR.value)
    assert (first_add["jaccard_similarity"], first_add["jaccard_denominator_status"]) == (0.0, DS.DEFINED.value)
    assert (first_add["drop_rate"], first_add["drop_rate_denominator_status"]) == (None, DS.ZERO_DENOMINATOR.value)
    assert (emptied["retention"], emptied["jaccard_similarity"], emptied["drop_rate"]) == (0.0, 0.0, 1.0)
    assert refill["retention"] is None and refill["jaccard_similarity"] == 0.0
    assert (both_empty["retention"], both_empty["jaccard_similarity"], both_empty["drop_rate"]) == (None, None, None)
    assert {both_empty[c] for c in ("retention_denominator_status", "jaccard_denominator_status",
                                    "drop_rate_denominator_status")} == {DS.ZERO_DENOMINATOR.value}
    assert both_empty["net_change"] == 0 and both_empty["assortment_change"] is False
    o = result.report.overall
    assert (o.retention_zero_denominator, o.jaccard_zero_denominator) == (3, 1)   # into hours 1, 3, 5; into 5


def test_jaccard_is_symmetric_and_retention_is_directional() -> None:
    result = build(sets({0: "AB", 1: "ACD", 2: "AB"}), [timeline([0, 1, 2])])
    forward, backward = row(result, 1), row(result, 2)
    assert forward["jaccard_similarity"] == backward["jaccard_similarity"] == 0.25
    assert (forward["retention"], backward["retention"]) == (0.5, 1 / 3)


def test_property_style_set_formulas_on_many_deterministic_synthetic_pairs() -> None:
    rng = random.Random(20301004)
    letters = "ABCDEFGH"
    hours = list(range(12))
    spec = {h: "".join(sorted(rng.sample(letters, rng.randint(0, 5)))) for h in hours}
    result = build(sets(spec), [timeline(hours)])
    for h in hours[1:]:
        p, c = set(spec[h - 1]), set(spec[h])
        r = row(result, h)
        assert counts(r) == (len(p), len(c), len(p & c), len(c - p), len(p - c))
        for value, num, den in ((r["retention"], len(p & c), len(p)), (r["jaccard_similarity"], len(p & c),
                                                                       len(p | c)),
                                (r["drop_rate"], max(len(p) - len(c), 0), len(p))):
            if den:
                assert value == float(Fraction(num, den)) and math.isfinite(value) and 0.0 <= value <= 1.0
            else:
                assert value is None
    floats = result.timeline[list(va._NULLABLE_FLOAT)].astype("float64").to_numpy()
    assert not np.isinf(floats).any()


# ============================================================================ schedule behaviour


@pytest.mark.parametrize(("state", "reason"), [(CS.MISSING_CAPTURE, IB.MISSING_CAPTURE),
                                               (CS.GOVERNED_EXCLUSION, IB.GOVERNED_EXCLUSION)])
def test_ineligible_captures_are_hard_breaks_without_any_attribution(state, reason) -> None:  # type: ignore[no-untyped-def]
    result = build(sets({0: "AB", 2: "C"}), [timeline([0, 1, 2], states={1: state})])
    gap, after = row(result, 1), row(result, 2)
    assert (gap["capture_state"], gap["assessability_status"], gap["interval_break_reason"]) == (
        state.value, AS.CAPTURE_NOT_ELIGIBLE.value, reason.value)
    assert gap["returned_product_count"] is None
    assert (after["assessability_status"], after["interval_break_reason"]) == (AS.INTERVAL_BREAK.value, reason.value)
    assert after["returned_product_count"] == 1
    for r in (gap, after):
        assert all(r[c] is None for c in va._INTERVAL_FIELDS) and r["has_previous_interval"] is False
    assert result.membership.empty
    s = result.report.location(LOC)
    assert s.breaks == ((reason.value, 2),) and s.counts.assessed_intervals == 0


def test_non_hourly_adjacency_and_changed_streams_are_breaks() -> None:
    gap = build(sets({0: "A", 2: "A"}), [timeline([0, 2])])
    assert row(gap, 2)["interval_break_reason"] == IB.NOT_ONE_HOUR.value
    streams = {1: tuple(sorted((LOC, SYNTH_STREAM)))}
    changed = build(sets({0: "A", 1: "A", 2: "A"}), [timeline([0, 1, 2], streams=streams)])
    assert [row(changed, h)["interval_break_reason"] for h in (1, 2)] == [IB.SOURCE_STREAMS_CHANGED.value] * 2
    assert row(changed, 1)["contributing_stream_count"] == 2
    assert changed.membership.empty and changed.report.overall.assessed_intervals == 0


def test_the_first_capture_after_a_break_starts_a_new_run() -> None:
    result = build(sets({0: "ABCD", 2: "A", 3: "AB"}), [timeline([0, 1, 2, 3], states={1: CS.MISSING_CAPTURE})])
    restart, following = row(result, 2), row(result, 3)
    assert restart["assessability_status"] == AS.INTERVAL_BREAK.value and restart["absolute_drop"] is None
    assert following[PREV] == P(2) and counts(following) == (1, 2, 1, 1, 0)


def test_a_rental_context_change_is_a_typed_break_not_a_full_turnover() -> None:
    rows = [*sets({0: "AB"}), car(1, "SYNTH Car A", ret=dt.date(2030, 4, 9)),
            car(1, "SYNTH Car B", ret=dt.date(2030, 4, 9)), car(2, "SYNTH Car A", ret=dt.date(2030, 4, 9))]
    result = build(rows, [timeline([0, 1, 2])])
    changed = row(result, 1)
    assert (changed["assessability_status"], changed["interval_break_reason"]) == (
        AS.INTERVAL_BREAK.value, RENTAL_CONTEXT_CHANGED)
    assert all(changed[c] is None for c in va._INTERVAL_FIELDS) and members(result, 1) == {}
    assert counts(row(result, 2)) == (2, 1, 1, 0, 1)
    assert result.report.location(LOC).breaks == ((RENTAL_CONTEXT_CHANGED, 1),)
    assert set(result.price_changes.candidates["outcome"]) >= {"appeared", "disappeared"}   # price engine view


# ============================================================================ drop metrics and anomaly policy


def test_the_default_unresolved_policy_never_emits_false_certainty() -> None:
    result = build(sets({0: "ABCD", 1: "A"}), [timeline([0, 1])])
    assert result.timeline["unusual_drop"].isna().all() and str(result.timeline["unusual_drop"].dtype) == "boolean"
    assert set(result.timeline["anomaly_policy_status"]) == {APS.UNAVAILABLE.value}
    assert result.report.anomaly_policy_status is APS.UNAVAILABLE and result.report.unusual_drop_intervals is None
    assert row(result, 1)["absolute_drop"] == 3 and row(result, 1)["drop_rate"] == 0.75
    proposed = build(sets({0: "ABCD", 1: "A"}), [timeline([0, 1])],
                     policy=UnusualDropPolicy(APS.PROPOSED, description="SYNTH candidate"))
    assert proposed.timeline["unusual_drop"].isna().all()
    assert set(proposed.timeline["anomaly_policy_status"]) == {APS.PROPOSED.value}


def test_an_explicit_approved_synthetic_policy_flags_boundaries_exactly() -> None:
    rule = synthetic_floor_rule(2, 0.5)
    spec = {0: "ABCD", 1: "AB", 2: "A", 3: "A", 4: ""}             # drops 2 (0.5), 1 (0.5), 0, 1 (1.0)
    result = build(sets(spec), [timeline([0, 1, 2, 3, 4])], policy=approved(rule))
    assert [row(result, h)["unusual_drop"] for h in range(5)] == [None, True, False, False, False]
    assert result.report.unusual_drop_intervals == 1 and result.report.anomaly_policy_status is APS.APPROVED
    tighter = build(sets(spec), [timeline([0, 1, 2, 3, 4])], policy=approved(synthetic_floor_rule(1, 1.0)))
    assert [row(tighter, h)["unusual_drop"] for h in range(1, 5)] == [False, False, False, True]
    for bad in ((0, 0.5), (2, 0.0), (2, 1.5), (True, 0.5), (2, 1)):
        with pytest.raises(ValueError):
            synthetic_floor_rule(*bad)
    assert DEFAULT_UNUSUAL_DROP_POLICY.status is APS.UNAVAILABLE and not DEFAULT_UNUSUAL_DROP_POLICY.executable


def test_an_approved_policy_never_classifies_breaks_seeds_or_non_drops() -> None:
    result = build(sets({0: "AB", 2: "A", 3: ""}), [timeline([0, 1, 2, 3], states={1: CS.GOVERNED_EXCLUSION})],
                   policy=approved(lambda c: True))
    assert [row(result, h)["unusual_drop"] for h in range(4)] == [None, None, None, True]
    with pytest.raises(AssortmentReconciliationError):                         # only an observed drop is unusual
        build(sets({0: "A", 1: "AB"}), [timeline([0, 1])], policy=approved(lambda c: True))


def test_approved_policies_need_an_executable_boolean_rule() -> None:
    with pytest.raises(AnomalyPolicyUnavailableError):
        build(sets({0: "AB", 1: "A"}), [timeline([0, 1])], policy=UnusualDropPolicy(APS.APPROVED, "SYNTH", "SYNTH"))
    with pytest.raises(AssortmentContractError):
        build(sets({0: "AB", 1: "A"}), [timeline([0, 1])], policy=approved(lambda c: 1))
    with pytest.raises(AssortmentContractError):
        UnusualDropPolicy(APS.APPROVED, "SYNTH", "SYNTH", rule="not callable")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        calculate_visible_assortment(frame(sets({0: "A"})), classify(sets({0: "A"}), [timeline([0])]),
                                     policy="SYNTH")                            # type: ignore[arg-type]


def test_the_engine_never_infers_collection_failure_or_supplier_withdrawal() -> None:
    result = build(sets({0: "ABCD", 1: ""}), [timeline([0, 1])])
    text = " ".join(str(v) for v in result.timeline.astype(object).to_numpy().ravel())
    text += repr(result.report)
    for claim in (AssortmentInterpretation.STATISTICALLY_UNUSUAL_DROP, AssortmentInterpretation.SUSPECTED_COLLECTION_FAILURE,
                  AssortmentInterpretation.SUPPLIER_ASSORTMENT_WITHDRAWAL):
        assert claim.value not in text
    source = (ROOT / "src" / "ql2_sixt_canada_analysis" / "visible_assortment.py").read_text(encoding="utf-8")
    assert "AssortmentInterpretation" not in source and "quantile" not in source and ".std(" not in source


# ============================================================================ price-change coincidence


def test_increases_and_decreases_in_the_same_interval_are_counted() -> None:
    rows = [car(0, "SYNTH Car A", 5000), car(0, "SYNTH Car B", 5000), car(0, "SYNTH Car C", 5000),
            car(1, "SYNTH Car A", 5500), car(1, "SYNTH Car B", 4000), car(1, "SYNTH Car D", 1)]
    r = row(build(rows, [timeline([0, 1])]), 1)
    assert (r["price_increase_count"], r["price_decrease_count"]) == (1, 1)
    assert r["price_change"] is True and r["assortment_change"] is True and r["assortment_price_coincidence"] is True
    assert r["falling_assortment_with_price_increase"] is False                 # net change is zero


@pytest.mark.parametrize(("rows", "outcome"), [
    ([car(0, "SYNTH Car A"), car(1, "SYNTH Car A")], "unchanged"),
    ([car(0, "SYNTH Car A"), car(1, "SYNTH Car A"), car(1, "SYNTH Car B")], "appeared"),
    ([car(0, "SYNTH Car A"), car(0, "SYNTH Car B"), car(1, "SYNTH Car A")], "disappeared"),
    ([car(0, "SYNTH Car A", 1), car(0, "SYNTH Car A", 2), car(1, "SYNTH Car A", 3)], "ambiguous"),
    ([car(0, "SYNTH Car A", currency="CA$"), car(1, "SYNTH Car A", currency="US$")], "appeared"),
])
def test_other_outcomes_are_never_price_changes(rows, outcome) -> None:  # type: ignore[no-untyped-def]
    result = build(rows, [timeline([0, 1])])
    assert outcome in set(result.price_changes.candidates["outcome"])
    r = row(result, 1)
    assert (r["price_increase_count"], r["price_decrease_count"], r["price_change"]) == (0, 0, False)
    assert r["assortment_price_coincidence"] is False


def test_changes_in_different_intervals_or_locations_never_coincide() -> None:
    rows = [car(0, "SYNTH Car A", 5000), car(1, "SYNTH Car A", 6000), car(2, "SYNTH Car A", 6000),
            car(2, "SYNTH Car B"),
            car(0, "SYNTH Car A", 1, loc=OTHER), car(1, "SYNTH Car A", 1, loc=OTHER), car(1, "SYNTH Car B", 1, loc=OTHER),
            car(2, "SYNTH Car A", 9, loc=OTHER), car(2, "SYNTH Car B", 1, loc=OTHER)]
    result = build(rows, [timeline([0, 1, 2]), timeline([0, 1, 2], loc=OTHER)])
    first, second = row(result, 1), row(result, 2)
    assert (first["price_change"], first["assortment_change"], first["assortment_price_coincidence"]) == (
        True, False, False)
    assert (second["price_change"], second["assortment_change"], second["assortment_price_coincidence"]) == (
        False, True, False)
    other_first, other_second = row(result, 1, OTHER), row(result, 2, OTHER)
    assert (other_first["assortment_change"], other_first["price_change"]) == (True, False)
    assert (other_second["assortment_change"], other_second["price_change"]) == (False, True)
    assert result.report.overall.coincident_intervals == 0


def test_a_falling_assortment_with_a_price_increase_is_flagged_descriptively() -> None:
    rows = [car(0, "SYNTH Car A", 5000), car(0, "SYNTH Car B", 5000), car(1, "SYNTH Car A", 5100)]
    r = row(build(rows, [timeline([0, 1])]), 1)
    assert (r["net_change"], r["price_increase_count"]) == (-1, 1)
    assert r["falling_assortment_with_price_increase"] is True and r["assortment_price_coincidence"] is True


def test_duplicate_stale_or_mismatched_price_evidence_fails_closed() -> None:
    rows = sets({0: "AB", 1: "AB"})
    prices = classify(rows, [timeline([0, 1])])
    duplicated = classify(rows, [timeline([0, 1])])
    object.__setattr__(duplicated, "candidates", pd.concat([duplicated.candidates, duplicated.candidates.iloc[[0]]]))
    with pytest.raises(AssortmentPriceEvidenceError):
        calculate_visible_assortment(frame(rows), duplicated)
    stale = classify(sets({0: "A", 1: "A"}), [timeline([0, 1])])                # built from other offers
    with pytest.raises(AssortmentPriceEvidenceError):
        calculate_visible_assortment(frame(rows), stale)
    shifted = classify(rows, [timeline([0, 1])])
    moved = shifted.candidates.copy()
    moved[PREV], moved[CUR] = P(5), P(6)
    object.__setattr__(shifted, "candidates", moved)
    with pytest.raises(AssortmentPriceEvidenceError):                          # an interval outside the grid
        calculate_visible_assortment(frame(rows), shifted)
    forged = classify(sets({0: "A", 1: "AB"}), [timeline([0, 1])])
    table = forged.candidates.copy()
    table.loc[table["outcome"] == "appeared", "outcome"] = "increase"           # an added product cannot change price
    object.__setattr__(forged, "candidates", table)
    with pytest.raises(AssortmentPriceEvidenceError):
        calculate_visible_assortment(frame(sets({0: "A", 1: "AB"})), forged)
    blocked = pce._blocked([PB.PRICING_NOT_READY])
    with pytest.raises(AssortmentPriceEvidenceError):
        calculate_visible_assortment(frame(rows), blocked)
    assert prices.completed


UNIT_VARIANTS = pytest.mark.parametrize("unit", [
    ({"currency": "CA$"}, {"currency": "US$"}),                                  # two currencies
    ({"basis": "day"}, {"basis": "week"}),                                      # two price bases
], ids=["currency", "price_basis"])


@UNIT_VARIANTS
def test_one_retained_product_may_carry_several_unit_specific_price_increases(unit) -> None:  # type: ignore[no-untyped-def]
    """Regression: price counts are candidate-level and may exceed the retained visible-product count."""
    first, second = unit
    rows = [car(0, "SYNTH Car A", 5000, **first), car(0, "SYNTH Car A", 4000, **second),
            car(1, "SYNTH Car A", 5500, **first), car(1, "SYNTH Car A", 4400, **second)]
    result = build(rows, [timeline([0, 1])])                                    # validated on construction
    assert result.completed
    seed, r = row(result, 0), row(result, 1)
    assert seed["returned_product_count"] == r["returned_product_count"] == 1    # units never inflate products
    assert counts(r) == (1, 1, 1, 0, 0) and (r["retention"], r["jaccard_similarity"]) == (1.0, 1.0)
    assert (r["price_increase_count"], r["price_decrease_count"]) == (2, 0)
    assert r["price_increase_count"] + r["price_decrease_count"] > r["retained_count"]
    assert r["price_change"] is True and r["assortment_change"] is False
    assert r["assortment_price_coincidence"] is False and r["falling_assortment_with_price_increase"] is False
    assert len(result.membership) == 1 and members(result, 1) == {"SYNTH Car A": "retained"}
    candidates = result.price_changes.candidates
    assert (candidates["outcome"] == "increase").sum() == 2
    o = result.report.overall
    assert (o.price_increases, o.price_decreases, o.price_changes, o.retained) == (2, 0, 2, 1)
    assert (o.price_change_intervals, o.assortment_change_intervals, o.coincident_intervals) == (1, 0, 0)
    assert result.report.location(LOC).counts.price_increases == 2


@UNIT_VARIANTS
def test_unit_specific_changes_keep_directions_separate_on_one_product(unit) -> None:  # type: ignore[no-untyped-def]
    first, second = unit
    rows = [car(0, "SYNTH Car A", 5000, **first), car(0, "SYNTH Car A", 4000, **second), car(0, "SYNTH Car B"),
            car(1, "SYNTH Car A", 5500, **first), car(1, "SYNTH Car A", 3900, **second)]
    r = row(build(rows, [timeline([0, 1])]), 1)
    assert counts(r) == (2, 1, 1, 0, 1) and (r["price_increase_count"], r["price_decrease_count"]) == (1, 1)
    assert r["assortment_price_coincidence"] is True and r["falling_assortment_with_price_increase"] is True


def test_assessed_rows_are_not_rejected_for_price_counts_above_retained_products() -> None:
    """The aggregate row check allows candidate counts above ``retained_count`` but not changes without one."""
    result = build([car(0, "SYNTH Car A", 1, currency="CA$"), car(0, "SYNTH Car A", 2, currency="US$"),
                    car(1, "SYNTH Car A", 3, currency="CA$"), car(1, "SYNTH Car A", 4, currency="US$")],
                   [timeline([0, 1])])
    assessed = row(result, 1)
    va._check_assessed(assessed)                                                 # 2 changes, 1 retained product
    orphan = {**assessed, "previous_product_count": 1, "returned_product_count": 1, "retained_count": 0,
              "addition_count": 1, "removal_count": 1, "retention": 0.0, "jaccard_similarity": 0.0,
              "assortment_change": True, "assortment_price_coincidence": True}
    with pytest.raises(AssortmentReconciliationError, match="only retained products can change price"):
        va._check_assessed(orphan)


def _forged(result: VisibleAssortmentResult, change) -> VisibleAssortmentResult:  # type: ignore[no-untyped-def]
    """Attach a copy of the result's price evidence whose candidate frame was altered by ``change``."""
    forged = dataclasses.replace(result.price_changes)
    table = result.price_changes.candidates.copy()
    change(table)
    object.__setattr__(forged, "candidates", table)
    return VisibleAssortmentResult(result.report, result.timeline, result.membership, result.timelines, forged)


def test_completed_results_prove_price_counts_from_candidate_identities() -> None:
    rows = [car(0, "SYNTH Car A", 5000), car(0, "SYNTH Car B"), car(1, "SYNTH Car A", 5500)]
    result = build(rows, [timeline([0, 1])])
    increase = result.price_changes.candidates["outcome"] == "increase"

    def to_removed(t: pd.DataFrame) -> None:                                    # projects to a removed product
        t.loc[increase, "car_name"] = "SYNTH Car B"

    def to_unknown(t: pd.DataFrame) -> None:                                    # projects to no product at all
        t.loc[increase, "car_type"] = "SYNTH Other Type"

    def to_other_interval(t: pd.DataFrame) -> None:
        t.loc[increase, [PREV, CUR]] = [P(5), P(6)]

    def recount(t: pd.DataFrame) -> None:                                       # counts no longer reconcile
        t.loc[increase, "outcome"] = "unchanged"

    def extra_unit(t: pd.DataFrame) -> None:                                    # a second, uncounted increase
        t.loc[len(t)] = t.loc[increase].iloc[0].to_dict() | {"currency": "US$"}

    for change in (to_removed, to_unknown, to_other_interval, recount, extra_unit):
        with pytest.raises(AssortmentReconciliationError):
            _forged(result, change)
    with pytest.raises(AssortmentReconciliationError, match="lack a contract column"):
        _forged(result, lambda t: t.drop(columns=["currency"], inplace=True))
    with pytest.raises(AssortmentReconciliationError):
        validate_price_coincidence(result.timeline, result.membership.iloc[0:0], result.price_changes)
    validate_price_coincidence(result.timeline, result.membership, result.price_changes)


@UNIT_VARIANTS
def test_a_unit_specific_change_on_a_non_retained_product_still_fails_closed(unit) -> None:  # type: ignore[no-untyped-def]
    first, second = unit
    rows = [car(0, "SYNTH Car A", **first), car(1, "SYNTH Car A", **first), car(1, "SYNTH Car B", **second)]
    forged = classify(rows, [timeline([0, 1])])
    table = forged.candidates.copy()
    table.loc[table["car_name"] == "SYNTH Car B", "outcome"] = "increase"        # an added product cannot change
    object.__setattr__(forged, "candidates", table)
    with pytest.raises(AssortmentPriceEvidenceError, match="only a retained product can change price"):
        calculate_visible_assortment(frame(rows), forged)


def test_membership_detail_never_assigns_prices_to_additions_or_removals() -> None:
    rows = [car(0, "SYNTH Car A", 5000), car(0, "SYNTH Car B", 1), car(1, "SYNTH Car A", 5500), car(1, "SYNTH Car C")]
    result = build(rows, [timeline([0, 1])])
    assert members(result, 1) == {"SYNTH Car A": "retained", "SYNTH Car B": "removed", "SYNTH Car C": "added"}
    assert not any("price" in c or "cents" in c or "change" in c for c in result.membership.columns)
    assert row(result, 1)["price_increase_count"] == 1 <= row(result, 1)["retained_count"]


# ============================================================================ timeline and reconciliation


def test_the_timeline_has_one_unique_row_per_location_and_capture_in_authority_order() -> None:
    rows = [*sets({0: "A", 1: "AB"}), *sets({1: "C"}, loc=OTHER)]
    tls = [timeline([0, 1, 2]), timeline([0, 1], loc=OTHER)]
    result = build(rows, tls, locations=[OTHER, LOC])
    t = result.timeline
    assert list(zip(t["canonical_location"], t["scheduled_capture_period"])) == [
        (OTHER[1], P(0)), (OTHER[1], P(1)), (LOC[1], P(0)), (LOC[1], P(1)), (LOC[1], P(2))]
    assert not t.duplicated(["canonical_city", "canonical_location", "scheduled_capture_period"]).any()
    assert result.report.approved_locations == (OTHER, LOC)
    again = build(list(reversed(rows)), list(reversed(tls)), locations=[OTHER, LOC])
    pd.testing.assert_frame_equal(t, again.timeline)
    pd.testing.assert_frame_equal(result.membership, again.membership)


def test_aggregate_totals_reconcile_to_rows_locations_and_breaks() -> None:
    rows = [*sets({0: "AB", 1: "BC", 3: "C", 4: "CD"}), *sets({0: "X", 1: "", 2: "XY"}, loc=OTHER)]
    tls = [timeline([0, 1, 2, 3, 4], states={2: CS.MISSING_CAPTURE}), timeline([0, 1, 2], loc=OTHER)]
    result = build(rows, tls)
    t, o = result.timeline, result.report.overall
    assessed = t[t["assessability_status"] == AS.ASSESSED.value]
    assert o.assessed_intervals == len(assessed) == len(result.membership.groupby([PREV, CUR, "canonical_location"]))
    assert o.retained == int(assessed["retained_count"].sum()) == (result.membership["membership"] == "retained").sum()
    assert o.additions == (result.membership["membership"] == "added").sum()
    assert o.removals == (result.membership["membership"] == "removed").sum()
    assert o.returned_products == int(t["returned_product_count"].sum())
    assert o.scheduled_captures == len(t) and o.eligible_captures == int((t["capture_state"] == "eligible").sum())
    assert sum((s.counts for s in result.report.locations), AssortmentCounts()) == o
    for s in result.report.locations:
        assert s.counts.assessed_intervals + sum(n for _, n in s.breaks) == s.counts.scheduled_captures - 1
    assert result.report.location(LOC).breaks == ((IB.MISSING_CAPTURE.value, 2),)


def test_the_timeline_holds_no_prohibited_fields_and_has_fixed_dtypes() -> None:
    result = build(sets({0: "AB", 1: "B"}), [timeline([0, 1])])
    t = result.timeline
    assert not set(t.columns) & (D.prohibited_columns | set(D.product_columns) | set(D.context_columns)
                                 | set(D.unit_columns) | set(FORBIDDEN_TIMESTAMP_SOURCES)
                                 | set(CONFIDENTIAL_TECHNICAL_COLUMNS))
    assert not set(t.columns) & (set(CANDIDATE_COLUMNS) - {"canonical_city", "canonical_location", PREV})
    assert {c: str(t[c].dtype) for c in ("returned_product_count", "retention", "unusual_drop",
                                         "contributing_stream_count", "has_previous_interval")} == {
        "returned_product_count": "Int64", "retention": "Float64", "unusual_drop": "boolean",
        "contributing_stream_count": "int64", "has_previous_interval": "bool"}
    shown = t.to_string()
    assert "SYNTH Car" not in shown and "2030-04" not in shown and "5000" not in shown


def _tamper(result: VisibleAssortmentResult, change) -> None:  # type: ignore[no-untyped-def]
    t = result.timeline.copy()
    change(t)
    VisibleAssortmentResult(result.report, t, result.membership, result.timelines, result.price_changes)


@pytest.mark.parametrize("change", [
    lambda t: t.__setitem__("retained_count", t["retained_count"] + 1),
    lambda t: t.__setitem__("retention", t["retention"] * 0.5),
    lambda t: t.__setitem__("jaccard_similarity", t["jaccard_similarity"] / 2),
    lambda t: t.__setitem__("assortment_change", ~t["assortment_change"]),
    lambda t: t.__setitem__("unusual_drop", t["unusual_drop"].fillna(False)),
    lambda t: t.__setitem__("previous_product_count", t["previous_product_count"] + 1),
    lambda t: t.__setitem__("price_increase_count", t["price_increase_count"] + 1),
    lambda t: t.__setitem__("interval_break_reason", "not_one_hour"),
    lambda t: t.__setitem__("assessability_status", AS.SEED_CAPTURE.value),
    lambda t: t.__setitem__("anomaly_policy_status", APS.APPROVED.value),
    lambda t: t.__setitem__("retention_denominator_status", DS.DEFINED.value),
    lambda t: t.drop(t.index[-1], inplace=True),
    lambda t: t.sort_values("scheduled_capture_period", ascending=False, inplace=True),
    lambda t: t.insert(len(t.columns), "price_cents", 1),
])
def test_malformed_timelines_cannot_construct_a_completed_result(change) -> None:  # type: ignore[no-untyped-def]
    result = build(sets({0: "ABC", 1: "AB", 2: ""}), [timeline([0, 1, 2])])
    with pytest.raises(AssortmentReconciliationError):
        _tamper(result, change)


def test_malformed_membership_reports_or_counts_cannot_construct_a_completed_result() -> None:
    result = build(sets({0: "ABC", 1: "ABD"}), [timeline([0, 1])])
    for membership in (result.membership.iloc[1:], result.membership.assign(membership="added"),
                       pd.concat([result.membership, result.membership.iloc[[0]]]),
                       result.membership.drop(columns=["car_name"])):
        with pytest.raises(AssortmentReconciliationError):
            VisibleAssortmentResult(result.report, result.timeline, membership, result.timelines,
                                    result.price_changes)
    summary = result.report.locations[0]
    wrong = dataclasses.replace(summary, counts=dataclasses.replace(summary.counts, empty_captures=1))
    with pytest.raises(AssortmentReconciliationError):
        VisibleAssortmentResult(dataclasses.replace(result.report, locations=(wrong,), overall=wrong.counts),
                                result.timeline, result.membership, result.timelines, result.price_changes)
    with pytest.raises(AssortmentReconciliationError):
        dataclasses.replace(result.report, overall=AssortmentCounts())
    with pytest.raises(AssortmentReconciliationError):
        AssortmentCounts(scheduled_captures=2, eligible_captures=1)
    with pytest.raises(AssortmentReconciliationError):
        AssortmentCounts(previous_products=2, retained=1)
    with pytest.raises(AssortmentReconciliationError):
        LocationAssortmentSummary(LOC, (("SYNTH-break", 1),), AssortmentCounts())
    with pytest.raises(AssortmentReconciliationError):
        LocationAssortmentSummary(LOC, (), AssortmentCounts(scheduled_captures=2, eligible_captures=2,
                                                            seed_captures=1, break_captures=1))


def test_blocked_results_cannot_masquerade_as_completed() -> None:
    result = build(sets({0: "A", 1: "A"}), [timeline([0, 1])])
    with pytest.raises(AssortmentReconciliationError):
        AssortmentReport(AssortmentStatus.BLOCKED)                               # a blocked report has blockers
    blocked = AssortmentReport(AssortmentStatus.BLOCKED, blockers=(B.PRICING_NOT_READY,))
    with pytest.raises(AssortmentReconciliationError):
        VisibleAssortmentResult(blocked, result.timeline)
    with pytest.raises(AssortmentReconciliationError):
        AssortmentReport(AssortmentStatus.BLOCKED, blockers=(B.PRICING_NOT_READY,), overall=AssortmentCounts())
    with pytest.raises(AssortmentReconciliationError):
        dataclasses.replace(result.report, blockers=(B.PRICING_NOT_READY,))
    with pytest.raises(AssortmentReconciliationError):
        dataclasses.replace(result.report, unusual_drop_intervals=0)            # no policy, no classification
    with pytest.raises(AssortmentReconciliationError):
        VisibleAssortmentResult(result.report, result.timeline, result.membership, None, result.price_changes)
    assert not VisibleAssortmentResult(blocked).completed and result.completed


# ============================================================================ gated assessment and pipeline


def assess(w: dict, **overrides) -> va.VisibleAssortmentResult:  # type: ignore[no-untyped-def]
    args = {**w, **overrides}
    if "price_changes" not in overrides:
        args["price_changes"] = pce.assess_price_change_candidates(
            args["jobs"], args["cars"], readiness=args["readiness"], population=args["population"],
            scheduled=args["scheduled"], canonical_offers=args["canonical_offers"],
            location_authority=args["location_authority"])
    return assess_visible_assortment(args.pop("jobs"), args.pop("cars"), **args)


def grow(name: str, *hours: int, streams=(TOR_DOWN,)) -> dict:  # type: ignore[no-untyped-def]
    return {(s, h): [(name, 50.0 + h)] for s in streams for h in hours}


def location_row(result: VisibleAssortmentResult, key, h: int) -> dict:  # type: ignore[no-untyped-def]
    t = result.timeline
    mine = t[(t["canonical_city"] == key[0]) & (t["canonical_location"] == key[1])
             & (t["scheduled_capture_period"] == at(key[0], h))]
    assert len(mine) == 1
    return {k: (None if v is pd.NA else v) for k, v in mine.iloc[0].astype(object).to_dict().items()}


def test_the_pipeline_world_completes_with_every_location_and_does_not_mutate_inputs() -> None:
    products = grow("SYNTH Car A", 0, 1, 2)
    products[(TOR_DOWN, 1)] = [("SYNTH Car A", 51.0), ("SYNTH Car B", 70.0)]
    w = synthetic_world(products=products)
    before = (w["jobs"].copy(deep=True), w["cars"].copy(deep=True))
    result = visible_assortment_from_pipeline(pipeline_result(w))
    assert result.completed and result.report.blockers == ()
    approved_keys = pce.approved_canonical_locations(AUTHORITY, POLICY)
    assert result.report.approved_locations == approved_keys and VAN_THUR not in approved_keys
    assert len(result.timeline) == 3 * len(approved_keys)
    assert counts(location_row(result, TOR_DOWN, 1)) == (2, 3, 2, 1, 0)
    assert counts(location_row(result, TOR_DOWN, 2)) == (3, 2, 2, 0, 1)
    tor = location_row(result, TOR_DOWN, 1)
    assert (tor["price_increase_count"], tor["assortment_price_coincidence"]) == (1, True)
    assert result.price_changes.binding == result.binding and result.location_authority is AUTHORITY
    pd.testing.assert_frame_equal(w["jobs"], before[0]), pd.testing.assert_frame_equal(w["cars"], before[1])


def test_vancouver_alias_offers_are_one_canonical_product() -> None:
    result = assess(synthetic_world(products=grow("SYNTH Car V", 0, 1, 2, streams=(VAN_DOWN, VAN_THUR))))
    assert counts(location_row(result, VAN_DOWN, 1)) == (2, 2, 2, 0, 0)       # filler and car V, never doubled
    assert location_row(result, VAN_DOWN, 1)["contributing_stream_count"] == 2
    assert not (result.timeline["canonical_location"] == VAN_THUR[1]).any()


def test_governed_exclusions_excused_missing_and_empty_captures_end_to_end() -> None:
    products = {(s, h): [("SYNTH Car A", 50.0), ("SYNTH Car B", 80.0)] for s in STREAMS for h in (0, 2)}
    w = synthetic_world(products=products, excluded=("calgary", 1), drop_streams={(CAL_DOWN, 1)})
    result = assess(w)
    assert result.completed
    for key in (CAL_DOWN, CAL_AIR):
        excluded = location_row(result, key, 1)
        assert (excluded["capture_state"], excluded["assessability_status"]) == (
            CS.GOVERNED_EXCLUSION.value, AS.CAPTURE_NOT_ELIGIBLE.value)
        assert location_row(result, key, 2)["interval_break_reason"] == IB.GOVERNED_EXCLUSION.value
        assert result.report.location(key).counts.assessed_intervals == 0
    missing = assess(synthetic_world(hours=4, products=grow("SYNTH Car A", 0, 2, 3), absent_jobs={("toronto", 1)},
                                     excused=("toronto", 1)))
    assert location_row(missing, TOR_DOWN, 2)["interval_break_reason"] == IB.MISSING_CAPTURE.value
    empty = assess(synthetic_world(products=grow("SYNTH Car A", 0, 1, 2), withheld={(TOR_DOWN, 1)}))
    hollow = location_row(empty, TOR_DOWN, 1)
    assert counts(hollow) == (2, 0, 0, 0, 2) and hollow["capture_state"] == CS.ELIGIBLE.value
    assert counts(location_row(empty, TOR_DOWN, 2)) == (0, 2, 0, 2, 0)


def test_run_visible_assortment_uses_one_pipeline_result_and_writes_nothing(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis import pricing_pipeline

    w = synthetic_world(products=grow("SYNTH Car A", 0, 1))
    calls = []

    def once(raw_dir=None):  # type: ignore[no-untyped-def]
        calls.append(raw_dir)
        return pipeline_result(w)

    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline", once)
    monkeypatch.chdir(tmp_path)
    result = run_visible_assortment(tmp_path)
    assert result.completed and calls == [tmp_path] and os.listdir(tmp_path) == []


def test_blocked_central_readiness_propagates_typed_blockers() -> None:
    w = synthetic_world()
    not_ready = readiness_for(w["cars"], w["scheduled"], w["canonical_offers"],
                              (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE,))
    for blocked in (visible_assortment_from_pipeline(dataclasses.replace(pipeline_result(w), pricing=not_ready)),
                    assess(w, readiness=not_ready)):
        assert blocked.report.blockers == (B.PRICING_NOT_READY,) and blocked.timeline is None
        assert blocked.report.upstream_blockers == (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE.value,)
        assert blocked.report.overall is None and blocked.membership is None and blocked.timelines is None
    missing = visible_assortment_from_pipeline(dataclasses.replace(pipeline_result(w), population=None))
    assert missing.report.upstream_blockers == ("required_assessment_unavailable",)
    with pytest.raises(TypeError):
        visible_assortment_from_pipeline(object())
    with pytest.raises(TypeError):
        visible_assortment_from_pipeline(dataclasses.replace(pipeline_result(w), pricing=None))


def test_stale_or_mismatched_evidence_fails_closed() -> None:
    w, other = synthetic_world(), synthetic_world(hours=2)
    assert B.EVIDENCE_BINDING_MISMATCH in assess(w, population=other["population"]).report.blockers
    assert B.EVIDENCE_BINDING_MISMATCH in assess(w, jobs=other["jobs"]).report.blockers
    assert B.EVIDENCE_BINDING_MISMATCH in assess(w, canonical_offers=other["canonical_offers"]).report.blockers
    foreign = pce.assess_price_change_candidates(
        other["jobs"], other["cars"], readiness=other["readiness"], population=other["population"],
        scheduled=other["scheduled"], canonical_offers=other["canonical_offers"],
        location_authority=other["location_authority"])
    assert assess(w, price_changes=foreign).report.blockers == (B.PRICE_CHANGE_EVIDENCE_INVALID,)
    blocked_prices = pce._blocked([PB.CAPTURE_EVIDENCE_INCONSISTENT])
    stopped = assess(w, price_changes=blocked_prices)
    assert stopped.report.blockers == (B.PRICE_CHANGE_EVIDENCE_INVALID,)
    assert stopped.report.upstream_blockers == (PB.CAPTURE_EVIDENCE_INCONSISTENT.value,)
    run = pipeline_result(w)
    assert B.EVIDENCE_BINDING_MISMATCH in visible_assortment_from_pipeline(
        dataclasses.replace(run, population=other["population"])).report.blockers
    with pytest.raises(TypeError):
        assess(w, price_changes=None)
    with pytest.raises(TypeError):
        assess(w, policy=None)


def test_invalid_schedules_offers_and_unavailable_authority_fail_closed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    invalid = synthetic_world(absent_jobs={("toronto", 1)})                     # unexcused missing capture
    assert B.SCHEDULE_EVIDENCE_INVALID in assess(invalid, price_changes=pce._blocked(
        [PB.SCHEDULE_EVIDENCE_INVALID])).report.blockers
    w = synthetic_world()
    scheduled = dataclasses.replace(w["scheduled"], capture_exclusions=None)
    readiness = readiness_for(w["cars"], scheduled, w["canonical_offers"])
    assert B.SCHEDULE_EVIDENCE_INVALID in assess(w, scheduled=scheduled, readiness=readiness).report.blockers

    def unavailable(authority, policy):  # type: ignore[no-untyped-def]
        raise pce.UnknownCanonicalLocationError("synthetic")

    prices = assess(w).price_changes
    monkeypatch.setattr(pce, "approved_canonical_locations", unavailable)
    assert B.LOCATION_AUTHORITY_UNAVAILABLE in assess(w, price_changes=prices).report.blockers


def test_offer_level_contract_failures_map_to_typed_blockers() -> None:
    w = synthetic_world(products=grow("SYNTH Car A", 1))
    offers = w["canonical_offers"]
    table = offers.offers.copy()
    i = table.index[table["car_name"] == "SYNTH Car A"][0]
    table.at[i, "return_date"] = dt.date(2030, 4, 9)                          # a second search context
    tampered = dataclasses.replace(offers, offers=table)
    readiness = readiness_for(w["cars"], w["scheduled"], tampered)
    world = {**w, "canonical_offers": tampered, "readiness": readiness}
    assert assess(world).report.blockers == (B.MULTIPLE_RENTAL_CONTEXTS,)
    run = dataclasses.replace(pipeline_result(w), canonical_offers=tampered, pricing=readiness)
    assert visible_assortment_from_pipeline(run).report.blockers == (B.MULTIPLE_RENTAL_CONTEXTS,)   # Prompt 1 gate
    prices = assess(w).price_changes
    renamed = offers.offers.copy()
    renamed.at[i, "car_name"] = "SYNTH Car Renamed"                            # stale offers vs price evidence
    extra = dataclasses.replace(offers, offers=renamed)
    mismatched = {**w, "canonical_offers": extra, "readiness": readiness_for(w["cars"], w["scheduled"], extra)}
    assert assess(mismatched, price_changes=prices).report.blockers == (B.PRICE_CHANGE_EVIDENCE_INVALID,)


def test_population_disagreement_reaches_the_price_engine_and_blocks() -> None:
    w = synthetic_world()
    statuses = list(w["population"].parent_status)
    statuses[0] = E.REPORTING_DAY_FAILED.value
    population = dataclasses.replace(w["population"], parent_status=tuple(statuses))
    result = assess(w, population=population)
    assert result.report.blockers == (B.PRICE_CHANGE_EVIDENCE_INVALID,)
    assert result.report.upstream_blockers == (PB.CAPTURE_EVIDENCE_INCONSISTENT.value,)


def test_run_visible_assortment_fails_closed_on_unready_synthetic_raw_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from conftest import contract_columns, write_synthetic_csv

    from ql2_sixt_canada_analysis.schemas import DatasetKey

    for key in DatasetKey:
        write_synthetic_csv(tmp_path / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    before = sorted(os.listdir(tmp_path))
    result = run_visible_assortment(tmp_path)
    assert result.report.blockers == (B.PRICING_NOT_READY,) and result.report.upstream_blockers
    assert result.timeline is None and result.membership is None
    assert sorted(os.listdir(tmp_path)) == before
    assert str(tmp_path) not in repr(result)


# ============================================================================ confidentiality and packaging


def test_proprietary_detail_stays_out_of_repr_reports_and_files(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    result = assess(synthetic_world(products=grow("SYNTH Car A", 0, 1)))
    assert result.completed and len(result.membership)
    shown = repr(result) + repr(result.report) + str(result.report)
    assert "SYNTH Car" not in shown and "SYNTH Filler" not in shown and "SYNTH-JOB" not in shown
    assert not re.search(r"2030\d{4}T\d{6}Z|2030-0[34]-\d\d", shown) and "DataFrame" not in shown
    assert os.listdir(tmp_path) == []
    fields = {f.name: f for f in dataclasses.fields(VisibleAssortmentResult)}
    for name in ("timeline", "membership", "timelines", "price_changes", "binding", "location_authority"):
        assert not fields[name].repr and not fields[name].compare
    assert not set(result.membership.columns) & (set(CONFIDENTIAL_TECHNICAL_COLUMNS)
                                                 | set(FORBIDDEN_TIMESTAMP_SOURCES) | {"source_location_labels"})


def test_the_module_never_writes_or_prints() -> None:
    source = (ROOT / "src" / "ql2_sixt_canada_analysis" / "visible_assortment.py").read_text(encoding="utf-8")
    for forbidden in ("to_csv", "to_parquet", "to_json", "savefig", "open(", "print(", "write_text", "write_bytes"):
        assert forbidden not in source


def test_importing_the_module_performs_no_io_pipeline_or_plotting() -> None:
    code = ("import builtins, io, os, sys\n"
            f"ROOT = {str(ROOT)!r}\n"
            "real = builtins.open\n"
            "def guarded(file, mode='r', *a, **k):\n"
            "    path = os.path.abspath(os.fspath(file)) if isinstance(file, (str, bytes, os.PathLike)) else ''\n"
            "    if any(c in mode for c in 'wax+') or str(path).startswith(ROOT):\n"
            "        raise AssertionError('I/O during import')\n"
            "    return real(file, mode, *a, **k)\n"
            "builtins.open = io.open = guarded\n"
            "import ql2_sixt_canada_analysis.visible_assortment as m\n"
            "help(m.run_visible_assortment)\n"
            "print('RESULT', 'ql2_sixt_canada_analysis.pricing_pipeline' in sys.modules, 'matplotlib' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1].split() == ["RESULT", "False", "False"]


def test_public_package_exports_are_complete() -> None:
    public = {n for n in vars(va) if not n.startswith("_") and getattr(getattr(va, n), "__module__", None)
              == va.__name__}
    assert public <= set(va.__all__) and len(va.__all__) == len(set(va.__all__))
    for name in va.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(va, name)
    assert {m.value for m in M} == {"retained", "added", "removed"}
    assert set(itertools.chain(B)) >= {B.PRICE_CHANGE_EVIDENCE_INVALID, B.RECONCILIATION_FAILED,
                                       B.UNKNOWN_CANONICAL_LOCATION, B.CAPTURE_EVIDENCE_INCONSISTENT}


def test_the_readme_documents_the_engine_and_the_data_plan_reconciliation() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    for phrase in ("visible_assortment.py", "run_visible_assortment", "visible_assortment_from_pipeline",
                   "assess_visible_assortment", "calculate_visible_assortment",
                   "Count returned products by location and capture: implemented as distinct product-set cardinality",
                   "Identify additions and removals: implemented as consecutive-set differences",
                   "Calculate consecutive-capture retention: implemented with the previous set as denominator",
                   "Calculate Jaccard similarity: implemented using intersection divided by union",
                   "Identify unusual assortment drops: drop measurements implemented; final flag gated by approved policy",
                   "Check whether assortment changes coincide with price changes: implemented by exact canonical "
                   "location and interval reconciliation",
                   "Produce an assortment timeline: implemented as the fixed-schema in-memory aggregate timeline"):
        assert phrase in readme, phrase
    assert "Production monitoring and assortment drop alerts are not established" in readme
