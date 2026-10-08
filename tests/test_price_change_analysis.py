"""Higher-order price-change analysis: synchronization, cross-location movement, persistence, final decrease.

Every observation is fabricated (``SYNTH-*`` jobs and products, synthetic 2030
capture periods and prices). The only committed values read are approved
configuration: source-stream keys, time zones, location roles and pairs, and
the canonical-offer policy of the current authority record.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest
from stream_contract_fixtures import synthetic_location_authority
from test_price_change_events import (
    AUTHORITY,
    CAL_AIR,
    CAL_DOWN,
    CONTRACT,
    PREV,
    CUR,
    TOR_AIR,
    TOR_DOWN,
    VAN_DOWN,
    VAN_THUR,
    at,
    pipeline_result,
    readiness_for,
    run,
    synthetic_world,
)

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis.readiness import PricingBlocker
from ql2_sixt_canada_analysis import price_change_analysis as pca
from ql2_sixt_canada_analysis.price_change_analysis import (
    CROSS_LOCATION_PRODUCT_COLUMNS,
    EVENT_TABLE_COLUMNS,
    PERSISTENCE_COLUMNS,
    CrossLocationOutcome as X,
    CrossLocationStatus,
    FinalDecreaseIndicator as FI,
    FinalDecreaseStatus as FS,
    IntervalFlag,
    MovementClass as M,
    NotTestableReason as NT,
    PersistenceOutcome as PO,
    PriceChangeAnalysisBlocker as AB,
    PriceChangeAnalysisResult,
    PriceChangeReconciliationError,
    analyze_price_change_events,
    event_heatmap_source,
    exact_change_percent,
    plot_price_change_heatmap,
    price_change_analysis_from_pipeline,
    run_price_change_analysis,
    write_price_change_analysis_outputs,
)
from ql2_sixt_canada_analysis.price_change_events import (
    EVENT_IDENTITY_COLUMNS,
    OutcomeCounts,
    PriceChangeCandidateResult,
    TerminalOutcome as T,
    change_percent,
)

VAN_AIR = ("vancouver", "Vancouver Int Airport")


def analyze(products=None, **world) -> PriceChangeAnalysisResult:  # type: ignore[no-untyped-def]
    w = synthetic_world(products=products, **world)
    events = run(w)
    assert events.completed, events.report.blockers
    result = analyze_price_change_events(events, location_authority=w["location_authority"])
    assert result.completed, result.report.blockers
    return result


def path(stream, prices, name="SYNTH Car A", start=0):  # type: ignore[no-untyped-def]
    """``{(stream, hour): [(name, price)]}`` for a price path; ``None`` = absent; a tuple = several offers."""
    out = {}
    for h, price in enumerate(prices, start):
        if price is None:
            continue
        for p in (price if isinstance(price, tuple) else (price,)):
            out.setdefault((stream, h), []).append((name, p))
    return out


def merge(*parts):  # type: ignore[no-untyped-def]
    out: dict = {}
    for part in parts:
        for k, v in part.items():
            out.setdefault(k, []).extend(v)
    return out


def interval(result: PriceChangeAnalysisResult, stream, h: int) -> dict:
    """The event-table row of ``stream`` for the interval ending at local hour ``h``."""
    t = result.event_table
    rows = t[(t["canonical_city"] == stream[0]) & (t["canonical_location"] == stream[1]) & (t[CUR] == at(stream[0], h))]
    assert len(rows) == 1
    return rows.iloc[0].to_dict()


def persistence(result: PriceChangeAnalysisResult, name="SYNTH Car A", stream=TOR_DOWN, h=1) -> list[dict]:
    """Persistence records of one product for changes in the interval ending at local hour ``h``."""
    p = result.persistence
    rows = p[(p["car_name"] == name) & (p["canonical_location"] == stream[1]) & (p[CUR] == at(stream[0], h))]
    return [r.to_dict() for _, r in rows.iterrows()]


def cross(result: PriceChangeAnalysisResult, name="SYNTH Car A", city="toronto", h=1) -> list[dict]:
    """Matched airport/downtown rows of one product for the interval ending at local hour ``h``."""
    c = result.cross_location
    mask = (c["car_name"] == name) & (c["canonical_city"] == city) & (c[CUR] == at(city, h))
    return [r.to_dict() for _, r in c[mask].iterrows()]


def check_reconciled(result: PriceChangeAnalysisResult) -> None:
    report, overall, table = result.report, result.events.report.overall, result.event_table
    assert report.price_change_count == overall.changed == int(table["price_change_count"].sum())
    assert report.assortment_event_count == overall.appeared + overall.disappeared
    assert report.intervals == overall.intervals == len(table)
    assert (table["candidates"] == table[[o.value for o in T]].sum(axis=1)).all()
    assert report.persistence.changed_events == len(result.persistence) == overall.changed
    assert tuple(table.columns) == EVENT_TABLE_COLUMNS and tuple(result.persistence.columns) == PERSISTENCE_COLUMNS
    s = report.persistence
    assert s.held + s.continued + s.reverted + s.disappeared + s.ambiguous + s.not_testable == s.changed_events


# ============================================================================ within-location synchronization


def test_no_movement_interval_stays_visible() -> None:
    result = analyze()
    check_reconciled(result)
    row = interval(result, TOR_DOWN, 1)
    assert (row["movement_class"], row["interval_flag"], row["comparable"], row["candidates"]) == (
        M.NO_PRICE_MOVEMENT.value, IntervalFlag.QUIET.value, 1, 1)
    assert not row["direction_synchronized"] and row["changed_share_of_comparable"] == 0.0
    assert result.report.intervals == 6 * 2                                     # every eligible interval, quiet too


@pytest.mark.parametrize(("prices", "cls"), [((50.0, 55.0), M.ISOLATED_INCREASE), ((50.0, 45.0), M.ISOLATED_DECREASE)])
def test_a_single_change_is_isolated(prices, cls) -> None:  # type: ignore[no-untyped-def]
    row = interval(analyze(path(TOR_DOWN, prices)), TOR_DOWN, 1)
    assert row["movement_class"] == cls.value and row["price_change_count"] == 1
    assert not row["direction_synchronized"] and not row["exact_cent_synchronized"]
    assert row["largest_same_cent_cohort"] == 1 and row["changed_share_of_comparable"] == 0.5


def test_several_increases_with_different_cents_and_percentages() -> None:
    row = interval(analyze(merge(path(TOR_DOWN, (50.0, 55.0)), path(TOR_DOWN, (60.0, 70.0), "SYNTH Car B"))),
                   TOR_DOWN, 1)
    assert row["movement_class"] == M.SYNCHRONIZED_INCREASE.value and row["direction_synchronized"]
    assert not row["exact_cent_synchronized"] and not row["exact_percent_synchronized"]
    assert (row["largest_same_cent_cohort"], row["largest_same_percent_cohort"]) == (1, 1)
    assert (row["min_change_cents"], row["max_change_cents"]) == (500, 1000)
    assert row["comparable"] == 3 and row["changed_share_of_comparable"] == float(Fraction(2, 3))


def test_several_decreases_with_identical_cents() -> None:
    row = interval(analyze(merge(path(TOR_DOWN, (50.0, 45.0)), path(TOR_DOWN, (80.0, 75.0), "SYNTH Car B"))),
                   TOR_DOWN, 1)
    assert row["movement_class"] == M.SYNCHRONIZED_DECREASE.value and row["direction_synchronized"]
    assert row["exact_cent_synchronized"] and not row["exact_percent_synchronized"]   # -10% vs -6.25%
    assert row["largest_same_cent_cohort"] == 2 and row["largest_same_percent_cohort"] == 1


def test_exact_percentage_synchronization_uses_reduced_rationals() -> None:
    row = interval(analyze(merge(path(TOR_DOWN, (50.0, 55.0)), path(TOR_DOWN, (80.0, 88.0), "SYNTH Car B"))),
                   TOR_DOWN, 1)
    assert row["exact_percent_synchronized"] and not row["exact_cent_synchronized"]
    assert row["largest_same_percent_cohort"] == 2
    assert exact_change_percent(5000, 500) == exact_change_percent(8000, 800) == Fraction(10)
    assert exact_change_percent(3, 1) == Fraction(100, 3) and exact_change_percent(0, 5) is None


def test_mixed_direction_interval() -> None:
    row = interval(analyze(merge(path(TOR_DOWN, (50.0, 55.0)), path(TOR_DOWN, (80.0, 75.0), "SYNTH Car B"))),
                   TOR_DOWN, 1)
    assert row["movement_class"] == M.MIXED_DIRECTION.value and not row["direction_synchronized"]
    assert row["exact_cent_synchronized"] is False


def test_zero_previous_price_is_excluded_from_exact_percentage_synchronization() -> None:
    row = interval(analyze(merge(path(TOR_DOWN, (0.0, 5.0)), path(TOR_DOWN, (0.0, 5.0), "SYNTH Car B"))),
                   TOR_DOWN, 1)
    assert row["direction_synchronized"] and row["exact_cent_synchronized"]
    assert not row["exact_percent_synchronized"] and row["largest_same_percent_cohort"] == 0
    assert row["zero_denominator"] == 2 and row["median_abs_change_percent"] is None


def test_assortment_events_and_ambiguity_never_count_as_price_movement() -> None:
    products = merge(path(TOR_DOWN, (None, 50.0)), path(TOR_DOWN, (60.0, None), "SYNTH Car B"),
                     path(TOR_DOWN, (70.0, 77.0), "SYNTH Car C"), path(TOR_DOWN, ((40.0, 41.0), 42.0), "SYNTH Car D"))
    result = analyze(products)
    check_reconciled(result)
    row = interval(result, TOR_DOWN, 1)
    assert (row["appeared"], row["disappeared"], row["ambiguous"], row["increase"]) == (1, 1, 1, 1)
    assert row["movement_class"] == M.ISOLATED_INCREASE.value and row["price_change_count"] == 1
    assert row["assortment_event_count"] == 2 and row["candidates"] == 5
    assert row["interval_flag"] == IntervalFlag.AMBIGUITY_PRESENT.value


@pytest.mark.parametrize(("a", "b", "column"), [((None, 50.0), (None, 60.0), "appeared"),
                                                ((50.0, None), (60.0, None), "disappeared")])
def test_simultaneous_assortment_events(a, b, column) -> None:  # type: ignore[no-untyped-def]
    row = interval(analyze(merge(path(TOR_DOWN, a), path(TOR_DOWN, b, "SYNTH Car B"))), TOR_DOWN, 1)
    assert row[column] == 2 and row["assortment_event_count"] == 2 and row["price_change_count"] == 0
    assert row["movement_class"] == M.NO_PRICE_MOVEMENT.value
    assert row["interval_flag"] == IntervalFlag.ASSORTMENT_ONLY.value


# ============================================================================ cross-location


def both(a_prices, d_prices, name="SYNTH Car A", city=(TOR_AIR, TOR_DOWN)):  # type: ignore[no-untyped-def]
    return merge(path(city[0], a_prices, name), path(city[1], d_prices, name))


@pytest.mark.parametrize(("a", "d", "outcome", "flags"), [
    ((50.0, 55.0), (80.0, 90.0), X.SAME_DIRECTION, (True, False, False)),      # same direction, different cent/pct
    ((50.0, 55.0), (80.0, 75.0), X.OPPOSITE_DIRECTION, (False, False, False)),
    ((50.0, 55.0), (100.0, 105.0), X.SAME_DIRECTION, (True, True, False)),     # exact cent only
    ((50.0, 55.0), (80.0, 88.0), X.SAME_DIRECTION, (True, False, True)),       # exact percentage only
    ((50.0, 55.0), (50.0, 55.0), X.SAME_DIRECTION, (True, True, True)),
    ((50.0, 55.0), (80.0, 80.0), X.AIRPORT_ONLY_CHANGE, (False, False, False)),
    ((50.0, 50.0), (80.0, 70.0), X.DOWNTOWN_ONLY_CHANGE, (False, False, False)),
    ((50.0, 50.0), (80.0, 80.0), X.BOTH_UNCHANGED, (False, False, False)),
    ((None, 50.0), (None, 80.0), X.SIMULTANEOUS_APPEARANCE, (False, False, False)),
    ((50.0, None), (80.0, None), X.SIMULTANEOUS_DISAPPEARANCE, (False, False, False)),
    ((50.0, None), (80.0, 81.0), X.ONE_SIDED_ASSORTMENT, (False, False, False)),
    ((None, 50.0), (80.0, None), X.MIXED_ASSORTMENT, (False, False, False)),
    (((50.0, 51.0), 52.0), (80.0, 81.0), X.AMBIGUOUS, (False, False, False)),
])
def test_cross_location_outcomes_and_separate_exact_flags(a, d, outcome, flags) -> None:  # type: ignore[no-untyped-def]
    result = analyze(both(a, d))
    check_reconciled(result)
    row, = cross(result)
    assert row["cross_outcome"] == outcome.value
    assert (row["same_direction"], row["same_cent_change"], row["same_percent_change"]) == flags
    assert (row["airport_location"], row["downtown_location"]) == (TOR_AIR[1], TOR_DOWN[1])
    toronto = next(s for s in result.report.cross_location if s.canonical_city == "toronto")
    assert toronto.status is CrossLocationStatus.AVAILABLE and dict(toronto.outcomes)[outcome.value] >= 1


def test_a_unit_change_is_never_a_cross_location_price_comparison() -> None:
    products = merge(path(TOR_AIR, (50.0, 55.0)),
                     {(TOR_DOWN, 0): [("SYNTH Car A", 80.0)], (TOR_DOWN, 1): [("SYNTH Car A", 88.0, ("currency", "US$"))]})
    result = analyze(products)
    rows = cross(result)
    assert [r["cross_outcome"] for r in rows] == [X.ONE_SIDED_ASSORTMENT.value]       # CA$: airport up, downtown gone
    assert not any(r["same_direction"] for r in rows)
    toronto = next(s for s in result.report.cross_location if s.canonical_city == "toronto")
    assert toronto.downtown_only_products == 2                                  # the US$ unit, both intervals


def test_the_cross_location_product_key_excludes_city_and_location_only() -> None:
    assert CROSS_LOCATION_PRODUCT_COLUMNS == tuple(c for c in EVENT_IDENTITY_COLUMNS
                                                   if c not in ("canonical_city", "canonical_location"))
    assert {"pickup_date", "return_date", "currency", "price_basis", "car_name"} <= set(CROSS_LOCATION_PRODUCT_COLUMNS)
    result = analyze(merge(both((50.0, 55.0), (80.0, 88.0)), path(TOR_AIR, (50.0, 55.0), "SYNTH Car Z")))
    assert len(cross(result)) == 1 and not cross(result, "SYNTH Car Z")          # one-sided products never match
    assert len(result.cross_location[result.cross_location["canonical_city"] == "calgary"]) == 2    # fillers


def test_roles_come_from_the_location_authority_never_from_labels() -> None:
    roles = {k: r for k, r in AUTHORITY.role_map.assignments}
    roles[TOR_AIR], roles[TOR_DOWN] = roles[TOR_DOWN], roles[TOR_AIR]           # labels now contradict roles
    pairs = [(p.airport, p.downtown) for p in AUTHORITY.effective_pairs if p.airport[0] != "toronto"]
    authority = synthetic_location_authority(CONTRACT, roles, (*pairs, (TOR_DOWN, TOR_AIR)))
    result = analyze(path(TOR_AIR, (50.0, 55.0)), authority=authority)
    assert interval(result, TOR_AIR, 1)["role"] == "DOWNTOWN" and interval(result, TOR_DOWN, 1)["role"] == "AIRPORT"
    toronto = next(s for s in result.report.cross_location if s.canonical_city == "toronto")
    assert (toronto.airport, toronto.downtown) == (TOR_DOWN, TOR_AIR)
    assert toronto.downtown_only_products == 2 and not cross(result)
    changed = analyze(both((50.0, 55.0), (80.0, 80.0)), authority=authority)    # "Int Airport" moved
    assert cross(changed)[0]["cross_outcome"] == X.DOWNTOWN_ONLY_CHANGE.value


def test_a_city_without_both_roles_is_unavailable_never_inferred() -> None:
    roles = {k: r for k, r in AUTHORITY.role_map.assignments}
    roles[TOR_AIR] = "OTHER"
    pairs = [(p.airport, p.downtown) for p in AUTHORITY.effective_pairs if p.airport[0] != "toronto"]
    authority = synthetic_location_authority(CONTRACT, roles, tuple(pairs))
    result = analyze(both((50.0, 55.0), (80.0, 88.0)), authority=authority)
    toronto = next(s for s in result.report.cross_location if s.canonical_city == "toronto")
    assert toronto.status is CrossLocationStatus.ROLE_UNAVAILABLE and toronto.matched_products == 0
    assert not cross(result) and interval(result, TOR_AIR, 1)["role"] == "OTHER"


# ============================================================================ Vancouver alias


def test_dual_vancouver_provenance_is_one_canonical_event() -> None:
    products = merge(path(VAN_DOWN, (50.0, 45.0)), path(VAN_THUR, (50.0, 45.0)),
                     path(VAN_DOWN, (60.0, 54.0), "SYNTH Car B"), path(VAN_THUR, (60.0, 54.0), "SYNTH Car B"),
                     path(VAN_AIR, (50.0, 45.0)))
    result = analyze(products)
    check_reconciled(result)
    row = interval(result, VAN_DOWN, 1)
    assert row["decrease"] == 2 and row["price_change_count"] == 2              # never four
    assert row["largest_same_percent_cohort"] == 2 and row["exact_percent_synchronized"]
    assert row["multi_source_candidates"] == row["candidates"] == 3             # filler, car A, car B
    assert len(persistence(result, stream=VAN_DOWN)) == 1
    assert [r["cross_outcome"] for r in cross(result, city="vancouver")] == [X.SAME_DIRECTION.value]
    summary = result.report.location(VAN_DOWN)
    assert summary.price_change_count == 2 and not any(s.canonical_location == VAN_THUR
                                                       for s in result.report.locations)


def test_provenance_change_is_reported_but_is_not_a_price_movement() -> None:
    products = merge(path(VAN_DOWN, (50.0, None)), path(VAN_THUR, (None, 50.0)))
    row = interval(analyze(products), VAN_DOWN, 1)
    assert row["unchanged"] == 2 and row["price_change_count"] == 0 and row["provenance_changed_candidates"] == 1


# ============================================================================ persistence


@pytest.mark.parametrize(("prices", "outcome", "returned", "overshot"), [
    ((50.0, 55.0, 55.0), PO.HELD, False, False),
    ((50.0, 55.0, 60.0), PO.CONTINUED, False, False),
    ((50.0, 45.0, 40.0), PO.CONTINUED, False, False),
    ((50.0, 60.0, 55.0), PO.REVERTED, False, False),                            # partial reversal
    ((50.0, 60.0, 50.0), PO.REVERTED, True, False),                             # full return
    ((50.0, 60.0, 45.0), PO.REVERTED, False, True),                             # overshoot
    ((50.0, 40.0, 55.0), PO.REVERTED, False, True),
    ((50.0, 55.0, None), PO.DISAPPEARED, False, False),
    ((50.0, 55.0, (56.0, 57.0)), PO.AMBIGUOUS, False, False),
])
def test_persistence_outcomes(prices, outcome, returned, overshot) -> None:  # type: ignore[no-untyped-def]
    result = analyze(path(TOR_DOWN, prices))
    check_reconciled(result)
    record, = persistence(result)
    assert record["persistence"] == outcome.value and record["not_testable_reason"] is None
    assert (record["returned_to_prior_price"], record["overshot_prior_price"]) == (returned, overshot)
    assert record["next_current_period"] == at("toronto", 2) and record[CUR] == at("toronto", 1)
    assert interval(result, TOR_DOWN, 1)[f"persistence_{outcome.value}"] == 1


def test_a_change_at_the_final_capture_is_right_censored() -> None:
    result = analyze(path(TOR_DOWN, (50.0, 50.0, 55.0)))
    record, = persistence(result, h=2)
    assert (record["persistence"], record["not_testable_reason"]) == (
        PO.NOT_TESTABLE.value, NT.RIGHT_CENSORED_FINAL_CAPTURE.value)
    s = result.report.persistence
    assert s.not_testable == 1 and s.with_following_interval == 0 and s.rates()["held_of_testable"] is None


def test_the_governed_exclusion_makes_persistence_not_testable() -> None:
    result = analyze(path(CAL_AIR, (50.0, 55.0, 999.0, 60.0)), hours=4, excluded=("calgary", 2),
                     drop_streams={(CAL_DOWN, 2)})
    record, = persistence(result, stream=CAL_AIR)
    assert (record["persistence"], record["not_testable_reason"]) == (
        PO.NOT_TESTABLE.value, NT.GOVERNED_EXCLUSION_BREAK.value)
    assert record["next_current_period"] is None


def test_a_missing_capture_makes_persistence_not_testable() -> None:
    result = analyze(path(TOR_DOWN, (50.0, 55.0, None, 60.0)), hours=4, absent_jobs={("toronto", 2)},
                     excused=("toronto", 2))
    record, = persistence(result)
    assert record["not_testable_reason"] == NT.MISSING_CAPTURE_BREAK.value


def test_a_source_stream_change_makes_persistence_not_testable() -> None:
    result = analyze(path(VAN_DOWN, (50.0, 55.0, 60.0)), short={VAN_THUR})
    record, = persistence(result, stream=VAN_DOWN)
    assert record["not_testable_reason"] == NT.SOURCE_STREAMS_CHANGED.value
    assert result.events.report.location(VAN_DOWN).intervals == 1


def test_a_valid_empty_capture_produces_disappeared_persistence() -> None:
    result = analyze(path(TOR_DOWN, (50.0, 55.0, 60.0)), withheld={(TOR_DOWN, 2)})
    record, = persistence(result)
    assert record["persistence"] == PO.DISAPPEARED.value
    assert interval(result, TOR_DOWN, 2)["interval_flag"] == IntervalFlag.EMPTY_ENDPOINT.value
    assert result.report.location(TOR_DOWN).empty_endpoint_intervals == 1


def _rebuild(events: PriceChangeCandidateResult, frame: pd.DataFrame) -> PriceChangeCandidateResult:
    """A self-consistent event result for a modified candidate frame (counts recomputed per location)."""
    summaries = []
    for s in events.report.locations:
        rows = frame[(frame["canonical_city"] == s.canonical_location[0])
                     & (frame["canonical_location"] == s.canonical_location[1])]
        counts = OutcomeCounts.of([T(o) for o in rows["outcome"]], intervals=s.intervals,
                                  zero_denominator=int(rows["zero_denominator"].sum()))
        summaries.append(dataclasses.replace(s, counts=counts))
    report = dataclasses.replace(events.report, locations=tuple(summaries),
                                 overall=sum((s.counts for s in summaries), OutcomeCounts()))
    return PriceChangeCandidateResult(report, frame.reset_index(drop=True), events.timelines, events.binding,
                                      events.location_authority)


def test_a_missing_following_candidate_fails_reconciliation() -> None:
    w = synthetic_world(products=path(TOR_DOWN, (50.0, 55.0, None)))
    events = run(w)
    f = events.candidates
    drop = (f["car_name"] == "SYNTH Car A") & (f["outcome"] == "disappeared")
    tampered = _rebuild(events, f[~drop])
    with pytest.raises(PriceChangeReconciliationError):
        analyze_price_change_events(tampered, location_authority=w["location_authority"])
    blocked = pca._analyze_bound(pipeline_result(w), tampered)
    assert blocked.report.blockers == (AB.RECONCILIATION_FAILED,) and blocked.event_table is None


def test_a_broken_price_chain_fails_reconciliation() -> None:
    w = synthetic_world(products=path(TOR_DOWN, (50.0, 55.0, 60.0)))
    events = run(w)
    f = events.candidates.copy()
    i = f.index[(f["car_name"] == "SYNTH Car A") & (f[PREV] == at("toronto", 1))][0]
    p, c = 5400, f.at[i, "current_price_cents"]                                 # follow-up starts from 54.00
    f.at[i, "previous_price_cents"], f.at[i, "previous_price"] = p, p / 100
    f.at[i, "change_cents"], f.at[i, "change_dollars"] = c - p, (c - p) / 100
    f.at[i, "change_percent"] = change_percent(p, c)
    tampered = _rebuild(events, f)
    with pytest.raises(PriceChangeReconciliationError):
        analyze_price_change_events(tampered, location_authority=w["location_authority"])


# ============================================================================ final Vancouver decrease


def test_the_final_vancouver_decrease_is_derived_and_right_censored() -> None:
    products = merge(path(VAN_DOWN, (60.0, 60.0, 54.0)), path(VAN_THUR, (60.0, 60.0, 54.0)),
                     path(VAN_DOWN, (70.0, 70.0, 63.0), "SYNTH Car B"), path(VAN_AIR, (90.0, 90.0, 81.0)))
    result = analyze(products)
    case = result.report.final_decrease
    assert case.status is FS.DERIVED and case.canonical_city == "vancouver"
    assert (case.previous_period, case.current_period) == (at("vancouver", 1), at("vancouver", 2))
    assert {k for k, _, _ in case.locations} == {VAN_DOWN, VAN_AIR} and all(final for _, _, final in case.locations)
    assert {(k, role) for k, role, _ in case.locations} == {(VAN_DOWN, "DOWNTOWN"), (VAN_AIR, "AIRPORT")}
    counts = dict(case.counts)
    assert (counts["decrease"], counts["unchanged"], counts["increase"]) == (3, 2, 0)
    assert case.price_change_count == 3 and case.comparable == 5
    assert case.changed_share_of_comparable == 0.6 and case.direction_synchronized
    assert case.exact_percent_synchronized and not case.exact_cent_synchronized
    assert dict(case.decrease_cents)["min"] == -900.0 and dict(case.decrease_percent)["max"] == -10.0
    assert dict(case.persistence) == {PO.NOT_TESTABLE.value: 3} and not case.persistence_testable
    assert dict(case.not_testable_reasons) == {NT.RIGHT_CENSORED_FINAL_CAPTURE.value: 3}
    assert FI.PERSISTENCE_RIGHT_CENSORED in case.indicators and FI.STABLE_ASSORTMENT in case.indicators
    provenance = dict(case.provenance)                                          # source labels: provenance only
    assert (provenance["Vancouver Downtown|Vancouver Thurlow"], provenance["Vancouver Downtown"],
            provenance["Vancouver Int Airport"]) == (2, 1, 2)                   # filler + car A, car B, airport
    assert "not proof" in case.describe() and "right censoring" in case.describe()
    assert "2030" not in repr(case)


def test_no_vancouver_decrease_is_reported_explicitly() -> None:
    case = analyze(path(VAN_DOWN, (50.0, 55.0))).report.final_decrease
    assert case.status is FS.NO_DECREASE and case.current_period is None and case.persistence == ()


# ============================================================================ reconciliation, tampering, order


def test_aggregate_tables_reconcile_to_the_event_report() -> None:
    products = merge(path(TOR_DOWN, (50.0, 55.0, 50.0)), path(TOR_AIR, (None, 70.0, 70.0)),
                     path(CAL_DOWN, (40.0, (41.0, 42.0), 43.0)), path(VAN_DOWN, (60.0, 54.0, None)))
    result = analyze(products)
    check_reconciled(result)
    assert sum(s.price_change_count for s in result.report.locations) == result.report.price_change_count
    assert sum(n for _, n in result.report.movement_classes) == result.report.intervals


def test_input_row_order_and_inputs_never_change() -> None:
    w = synthetic_world(products=merge(path(TOR_DOWN, (50.0, 55.0, 50.0)), path(TOR_AIR, (60.0, 66.0, 66.0))))
    events = run(w)
    before = events.candidates.copy(deep=True)
    reference = analyze_price_change_events(events, location_authority=w["location_authority"])
    pd.testing.assert_frame_equal(events.candidates, before)
    shuffled = PriceChangeCandidateResult(events.report, events.candidates.sample(frac=1.0, random_state=3)
                                          .reset_index(drop=True), events.timelines, events.binding,
                                          events.location_authority)
    again = analyze_price_change_events(shuffled, location_authority=w["location_authority"])
    assert again.report == reference.report
    for name in ("event_table", "cross_location", "persistence"):
        pd.testing.assert_frame_equal(getattr(again, name), getattr(reference, name))


@pytest.mark.parametrize("target, tamper", [
    ("event_table", lambda f: f.assign(increase=f["increase"] + 1)),
    ("event_table", lambda f: f.iloc[1:]),
    ("persistence", lambda f: f.assign(persistence="held")),
    ("cross_location", lambda f: pd.concat([f, f])),
])
def test_tampered_higher_order_tables_are_rejected(target, tamper) -> None:  # type: ignore[no-untyped-def]
    good = analyze(path(TOR_DOWN, (50.0, 55.0, 60.0)))
    frames = {n: getattr(good, n) for n in ("event_table", "cross_location", "persistence")}
    frames[target] = tamper(frames[target].copy())
    with pytest.raises(PriceChangeReconciliationError):
        PriceChangeAnalysisResult(good.report, good.events, good.location_authority, **frames)
    with pytest.raises(PriceChangeReconciliationError):
        PriceChangeAnalysisResult(dataclasses.replace(good.report, price_change_count=0), good.events,
                                  good.location_authority, **{n: getattr(good, n) for n in frames})


def test_heatmap_source_reconciles_and_masks_breaks() -> None:
    result = analyze(merge(path(TOR_DOWN, (50.0, 55.0, 50.0, 45.0)), path(CAL_AIR, (50.0, 55.0, 99.0, 60.0))),
                     hours=4, excluded=("calgary", 2), drop_streams={(CAL_DOWN, 2)})
    source = event_heatmap_source(result)
    assert int(pd.Series(source["increase"].ravel()).sum()) + int(pd.Series(source["decrease"].ravel()).sum()) \
        == result.report.price_change_count
    assert (source["state"] == "interval").sum() == result.report.intervals
    cal = source["locations"].index(CAL_AIR)
    assert (source["state"][cal] == "break").sum() == 3                         # first capture + both sides of exclusion
    assert pd.isna(source["increase"][cal][source["state"][cal] != "interval"]).all()
    tampered = analyze(path(TOR_DOWN, (50.0, 55.0)))
    object.__setattr__(tampered, "event_table", tampered.event_table.assign(increase=0))
    with pytest.raises(PriceChangeReconciliationError):
        event_heatmap_source(tampered)


def test_outputs_are_written_only_to_the_requested_directory(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    result = analyze(merge(path(TOR_DOWN, (50.0, 55.0, 50.0)), path(VAN_DOWN, (60.0, 54.0, 54.0))))
    fig, _ = plot_price_change_heatmap(result)                                  # in memory only
    assert os.listdir(cwd) == [] and not list(tmp_path.glob("out*"))
    table_path, figure_path = write_price_change_analysis_outputs(result, tmp_path / "out")
    assert sorted(os.listdir(tmp_path / "out")) == ["price_change_event_heatmap.png", "price_change_event_table.csv"]
    assert figure_path.read_bytes()[:4] == b"\x89PNG" and os.listdir(cwd) == []
    written = pd.read_csv(table_path)
    assert tuple(written.columns) == EVENT_TABLE_COLUMNS and len(written) == result.report.intervals
    text = table_path.read_text(encoding="utf-8")
    assert "SYNTH" not in text and "2030-04" not in text                        # no product values or rental dates
    with pytest.raises(TypeError):
        write_price_change_analysis_outputs(result, None)  # type: ignore[arg-type]


def test_frames_stay_out_of_repr_and_equality() -> None:
    result = analyze(path(TOR_DOWN, (50.0, 55.0, 60.0)))
    shown = repr(result) + repr(result.report) + str(result.report)
    assert "SYNTH" not in shown and "DataFrame" not in shown and not re.search(r"2030\d{4}T\d{6}Z", shown)
    assert "5500" not in shown and "55.0" not in shown
    fields = {f.name: f for f in dataclasses.fields(PriceChangeAnalysisResult)}
    for name in ("events", "event_table", "cross_location", "persistence", "location_authority"):
        assert not fields[name].repr and not fields[name].compare


# ============================================================================ evidence binding and orchestration


def test_mismatched_or_unbound_evidence_fails_closed() -> None:
    w = synthetic_world(products=path(TOR_DOWN, (50.0, 55.0)))
    events = run(w)
    other = synthetic_location_authority(CONTRACT, dict(AUTHORITY.role_map.assignments),
                                         tuple((p.airport, p.downtown) for p in AUTHORITY.effective_pairs))
    assert analyze_price_change_events(events, location_authority=other).report.blockers == (AB.EVIDENCE_MISMATCH,)
    unbound = PriceChangeCandidateResult(events.report, events.candidates, events.timelines)
    assert analyze_price_change_events(unbound, location_authority=AUTHORITY).report.blockers == (
        AB.EVENT_EVIDENCE_UNBOUND,)
    stale = pca._analyze_bound(pipeline_result(synthetic_world()), events)      # events of another run
    assert stale.report.blockers == (AB.EVIDENCE_MISMATCH,) and stale.event_table is None
    mixed = dataclasses.replace(pipeline_result(w), location_authority=other)
    blocked = price_change_analysis_from_pipeline(mixed)
    assert blocked.report.blockers == (AB.EVENTS_NOT_COMPLETED,) and blocked.report.event_blockers
    not_ready = dataclasses.replace(pipeline_result(w), pricing=readiness_for(
        w["cars"], w["scheduled"], w["canonical_offers"], (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE,)))
    assert price_change_analysis_from_pipeline(not_ready).report.blockers == (AB.EVENTS_NOT_COMPLETED,)
    with pytest.raises(TypeError):
        analyze_price_change_events(events, location_authority=None)


def test_the_top_level_run_calls_the_pricing_pipeline_exactly_once(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis import pricing_pipeline

    w = synthetic_world(products=path(TOR_DOWN, (50.0, 55.0)))
    calls = []

    def fake(raw_dir=None):  # type: ignore[no-untyped-def]
        calls.append(raw_dir)
        return pipeline_result(w)

    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline", fake)
    result = run_price_change_analysis("SYNTH-RAW-DIR")
    assert result.completed and calls == ["SYNTH-RAW-DIR"]
    assert result.report.price_change_count == 1


def test_run_fails_closed_on_unready_synthetic_raw_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from conftest import contract_columns, write_synthetic_csv

    from ql2_sixt_canada_analysis.schemas import DatasetKey

    for key in DatasetKey:
        write_synthetic_csv(tmp_path / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    result = run_price_change_analysis(tmp_path)
    assert result.report.blockers == (AB.EVENTS_NOT_COMPLETED,) and result.event_table is None
    assert str(tmp_path) not in repr(result)


def test_importing_the_module_performs_no_io_pipeline_or_plotting() -> None:
    root = str(Path(__file__).resolve().parents[1])
    code = ("import builtins, io, os, sys\n"
            f"ROOT = {root!r}\n"
            "real = builtins.open\n"
            "def guarded(file, mode='r', *a, **k):\n"
            "    path = os.path.abspath(os.fspath(file)) if isinstance(file, (str, bytes, os.PathLike)) else ''\n"
            "    if any(c in mode for c in 'wax+') or str(path).startswith(ROOT):\n"
            "        raise AssertionError('I/O during import')\n"
            "    return real(file, mode, *a, **k)\n"
            "builtins.open = io.open = guarded\n"
            "before = set(os.listdir('.'))\n"
            "import ql2_sixt_canada_analysis.price_change_analysis as m\n"
            "print('matplotlib' in sys.modules, 'ql2_sixt_canada_analysis.pricing_pipeline' in sys.modules,\n"
            "      set(os.listdir('.')) == before)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False", "True"]


def test_public_package_exports_are_complete() -> None:
    assert len(pca.__all__) == len(set(pca.__all__))
    public = {n for n in vars(pca) if not n.startswith("_") and getattr(getattr(pca, n), "__module__", None)
              == pca.__name__}
    assert public <= set(pca.__all__)
    for name in pca.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(pca, name)
