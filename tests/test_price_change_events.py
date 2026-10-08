"""The price-change event contract: observed price-change candidates between adjacent scheduled captures.

Every value is fabricated: ``SYNTH-*`` jobs and products, synthetic cities and
branches (``alpha``/``beta``), synthetic 2030 capture periods, rental dates and
prices. The only committed configuration read is the canonical-offer policy of
the current authority record (it leaves the synthetic streams unaliased).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from test_collection_schedule import ALPHA_AIR, ALPHA_DOWN, BETA_DOWN, SYN_CONTRACT, SYN_STREAMS, synth_schedule

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis import price_change_events as pce
from ql2_sixt_canada_analysis.authority_decisions import AuthorityKind
from ql2_sixt_canada_analysis.canonical_offers import (
    APPROVED_PRODUCT_COLUMNS,
    assess_canonical_offers,
    current_canonical_offer_policy,
)
from ql2_sixt_canada_analysis.collection_schedule import (
    ExceptionsModel,
    ParentCaptureExclusion,
    ScheduleExceptions,
    ScheduleFailureKind,
    StreamScheduleException,
    assess_per_stream_scheduled_coverage,
    format_utc_instant,
)
from ql2_sixt_canada_analysis.price_change_events import (
    CANDIDATE_COLUMNS,
    CAPTURE_STEP,
    EVENT_IDENTITY_COLUMNS,
    EVENT_KEY_COLUMNS,
    EVENT_TIMESTAMP_COLUMN,
    FORBIDDEN_TIMESTAMP_SOURCES,
    CaptureEvidenceError,
    CaptureInterval,
    CaptureState as CS,
    IntervalBreak as IB,
    LocationCaptureTimeline,
    OutcomeCounts,
    PriceChangeBlocker as PB,
    PriceChangeCandidateReport,
    PriceChangeCandidateResult,
    PriceChangeContractError,
    PriceChangeStatus,
    ScheduledCapture,
    TerminalOutcome as T,
    assess_price_change_candidates,
    capture_timelines,
    classify_endpoint_offers,
    classify_price_change_candidates,
    parse_scheduled_period,
    price_change_candidates_from_pipeline,
    price_change_cents,
)
from ql2_sixt_canada_analysis.pricing_population import DetailEligibility as E, PricingPopulation, frame_binding
from ql2_sixt_canada_analysis.readiness import (
    PricingBlocker,
    PricingReadinessReport,
    apply_location_policy,
    assess_location_policy,
)
from ql2_sixt_canada_analysis.schemas import JOB_DETAIL_RELATIONSHIP as REL, VANCOUVER_LOCATION_POLICY

POLICY = current_canonical_offer_policy()
LOC, OTHER = ("synth-city", "SYNTH Downtown"), ("synth-city", "SYNTH Airport")
PICKUP, RETURN = dt.date(2030, 4, 1), dt.date(2030, 4, 3)


def P(hour: int) -> str:
    """A synthetic scheduled period (UTC start)."""
    return (dt.datetime(2030, 3, 4, 0, tzinfo=dt.timezone.utc) + dt.timedelta(hours=hour)).strftime("%Y%m%dT%H%M%SZ")


# ============================================================================ pure-contract fixtures


def offer(period: str, cents: int = 5000, loc=LOC, name="SYNTH Car A", pickup=PICKUP, ret=RETURN,  # type: ignore[no-untyped-def]
          currency="CA$", basis="day", labels=None, **changes) -> dict:
    row = {"canonical_city": loc[0], "canonical_location": loc[1], EVENT_TIMESTAMP_COLUMN: period,
           "pickup_date": pickup, "return_date": ret, "car_name": name, "car_type": "SYNTH Compact",
           "transmission": "SYNTH Automatic", "seats": "5", "bags": "2", "price_cents": cents, "price_basis": basis,
           "currency": currency, "source_location_labels": labels if labels is not None else loc[1],
           "observation_count": 1, "provenance": "unique", "price_variation": False}
    row.update(changes)
    return row


def frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def timeline(hours, loc=LOC, states=None, streams=None) -> LocationCaptureTimeline:  # type: ignore[no-untyped-def]
    states = states or {}
    streams = streams or {}
    return LocationCaptureTimeline(loc, tuple(
        ScheduledCapture(P(h), states.get(h, CS.ELIGIBLE), streams.get(h, (loc,))) for h in hours))


def filler(hours, loc=LOC) -> list[dict]:  # type: ignore[no-untyped-def]
    """A constant product present at every eligible capture (keeps every capture non-empty)."""
    return [offer(P(h), 1000, loc=loc, name="SYNTH Filler") for h in hours]


def classify(rows, timelines):  # type: ignore[no-untyped-def]
    candidates, summaries = classify_price_change_candidates(frame(rows), timelines)
    overall = sum((s.counts for s in summaries), OutcomeCounts())
    report = PriceChangeCandidateReport(status=PriceChangeStatus.COMPLETED, offers_assessed=len(rows),
                                        locations=summaries, overall=overall)
    result = PriceChangeCandidateResult(report, candidates, tuple(timelines))   # every invariant enforced
    return result


def product(result: PriceChangeCandidateResult, name: str = "SYNTH Car A") -> pd.DataFrame:
    return result.candidates[result.candidates["car_name"] == name].reset_index(drop=True)


def outcomes(result: PriceChangeCandidateResult, name: str = "SYNTH Car A") -> list[tuple[str, str, str]]:
    rows = product(result, name)
    return list(zip(rows["previous_period"], rows["current_period"], rows["outcome"]))


def check_invariants(result: PriceChangeCandidateResult) -> None:
    c, frame_ = result.report.overall, result.candidates
    assert c.candidates == len(frame_) == sum(getattr(c, o.value) for o in T)
    assert frame_["outcome"].value_counts().to_dict() == {o.value: getattr(c, o.value) for o in T if getattr(c, o.value)}
    assert sum((s.counts for s in result.report.locations), OutcomeCounts()) == c
    assert not frame_.duplicated(list(EVENT_KEY_COLUMNS)).any()
    for s in result.report.locations:
        assert s.intervals + sum(n for _, n in s.breaks) == max(s.scheduled_periods - 1, 0)


# ============================================================================ terminal outcomes


@pytest.mark.parametrize(("before", "after", "expected", "change"), [
    (5000, 5000, T.UNCHANGED, 0), (5000, 5001, T.INCREASE, 1), (5000, 4250, T.DECREASE, -750)])
def test_one_offer_at_both_endpoints_compares_exact_cents(before, after, expected, change) -> None:  # type: ignore[no-untyped-def]
    result = classify([offer(P(0), before), offer(P(1), after)], [timeline([0, 1])])
    check_invariants(result)
    row = result.candidates.iloc[0]
    assert len(result.candidates) == 1 and row["outcome"] == expected.value
    assert (row["previous_price_cents"], row["current_price_cents"], row["change_cents"]) == (before, after, change)
    assert (row["previous_offer_count"], row["current_offer_count"]) == (1, 1)
    assert classify_endpoint_offers([before], [after]) is expected and price_change_cents(before, after) == change


def test_a_one_cent_difference_is_never_rounded_away() -> None:
    result = classify([offer(P(0), 1999), offer(P(1), 2000)], [timeline([0, 1])])
    assert outcomes(result) == [(P(0), P(1), "increase")] and result.candidates.loc[0, "change_cents"] == 1


def test_missing_prior_identity_is_an_appearance_and_missing_current_is_a_disappearance() -> None:
    rows = [*filler([0, 1, 2]), offer(P(1), 4000)]
    result = classify(rows, [timeline([0, 1, 2])])
    check_invariants(result)
    assert outcomes(result) == [(P(0), P(1), "appeared"), (P(1), P(2), "disappeared")]
    appeared, disappeared = product(result).iloc[0], product(result).iloc[1]
    assert appeared["previous_price_cents"] is None and appeared["current_price_cents"] == 4000
    assert appeared["change_cents"] is None and (appeared["previous_offer_count"], appeared["current_offer_count"]) == (0, 1)
    assert disappeared["current_price_cents"] is None and disappeared["previous_price_cents"] == 4000
    assert disappeared["change_cents"] is None


@pytest.mark.parametrize(("before", "after"), [((5000, 5100), (5000,)), ((5000,), (4900, 5000)),
                                               ((5000, 5100), (5200, 5300)), ((5000, 5100), ()), ((), (1, 2, 3))])
def test_several_offers_at_either_endpoint_are_ambiguous_without_expansion(before, after) -> None:  # type: ignore[no-untyped-def]
    rows = [*filler([0, 1]), *(offer(P(0), c) for c in before), *(offer(P(1), c) for c in after)]
    result = classify(rows, [timeline([0, 1])])
    check_invariants(result)
    rows_ = product(result)
    assert len(rows_) == 1 and rows_.loc[0, "outcome"] == "ambiguous"                # one candidate, no product
    assert (rows_.loc[0, "previous_offer_count"], rows_.loc[0, "current_offer_count"]) == (len(before), len(after))
    assert rows_.loc[0, ["previous_price_cents", "current_price_cents", "change_cents"]].tolist() == [None] * 3
    assert not rows_.loc[0, "zero_baseline"] and result.report.overall.ambiguous == 1
    assert classify_endpoint_offers(before, after) is T.AMBIGUOUS


@pytest.mark.parametrize(("field", "other"), [
    ("car_name", "SYNTH Car B"), ("car_type", "SYNTH Fullsize"), ("transmission", "SYNTH Manual"), ("seats", "7"),
    ("bags", "3"), ("pickup_date", dt.date(2030, 4, 2)), ("return_date", dt.date(2030, 4, 4))])
def test_different_products_or_rental_dates_never_compare(field, other) -> None:  # type: ignore[no-untyped-def]
    result = classify([*filler([0, 1]), offer(P(0), 5000), offer(P(1), 9000, **{field: other})], [timeline([0, 1])])
    check_invariants(result)
    c = result.report.overall
    assert (c.appeared, c.disappeared, c.increase, c.decrease, c.unchanged) == (1, 1, 0, 0, 1)   # filler unchanged


@pytest.mark.parametrize(("field", "other"), [("seats", "5 "), ("car_name", "synth car a")])
def test_identity_values_are_never_trimmed_or_recased(field, other) -> None:  # type: ignore[no-untyped-def]
    if other != other.strip():
        with pytest.raises(PriceChangeContractError):                            # untrimmed text is malformed
            classify([offer(P(0)), offer(P(1), **{field: other})], [timeline([0, 1])])
        return
    result = classify([*filler([0, 1]), offer(P(0)), offer(P(1), **{field: other})], [timeline([0, 1])])
    assert (result.report.overall.appeared, result.report.overall.disappeared) == (1, 1)


def test_different_canonical_locations_never_compare() -> None:
    rows = [*filler([0, 1]), *filler([0, 1], OTHER), offer(P(0), 5000, loc=LOC), offer(P(1), 5000, loc=OTHER)]
    result = classify(rows, [timeline([0, 1]), timeline([0, 1], loc=OTHER)])
    check_invariants(result)
    assert result.report.overall.priced == 2                                     # the two fillers only
    by_location = {s.canonical_location: s.counts for s in result.report.locations}
    assert (by_location[LOC].disappeared, by_location[LOC].appeared) == (1, 0)
    assert (by_location[OTHER].appeared, by_location[OTHER].disappeared) == (1, 0)


@pytest.mark.parametrize(("field", "before", "after"), [("currency", "CA$", "US$"), ("price_basis", "day", "week")])
def test_unit_changes_are_a_disappearance_and_an_appearance_never_a_price_movement(field, before, after) -> None:  # type: ignore[no-untyped-def]
    for cents in ((5000, 5000), (5000, 7000), (5000, 3000)):
        rows = [*filler([0, 1]), offer(P(0), cents[0], **{field: before}), offer(P(1), cents[1], **{field: after})]
        result = classify(rows, [timeline([0, 1])])
        check_invariants(result)
        moved = product(result)
        assert sorted(moved["outcome"]) == ["appeared", "disappeared"]
        assert moved["change_cents"].tolist() == [None, None]
        assert result.report.overall.priced == 1                                 # the filler only


def test_price_is_not_part_of_the_identity_but_currency_and_basis_are() -> None:
    assert "price_cents" not in EVENT_IDENTITY_COLUMNS
    assert {"currency", "price_basis", "canonical_city", "canonical_location", "pickup_date", "return_date",
            *APPROVED_PRODUCT_COLUMNS} == set(EVENT_IDENTITY_COLUMNS)
    assert EVENT_KEY_COLUMNS == (*EVENT_IDENTITY_COLUMNS, "previous_period", "current_period")
    result = classify([offer(P(0), 100), offer(P(1), 99_999)], [timeline([0, 1])])
    assert outcomes(result) == [(P(0), P(1), "increase")]


def test_source_labels_are_provenance_and_never_split_the_identity() -> None:
    rows = [offer(P(0), 5000, labels="SYNTH Downtown"), offer(P(1), 5000, labels="SYNTH Downtown|SYNTH Thurlow")]
    result = classify(rows, [timeline([0, 1])])
    row = result.candidates.iloc[0]
    assert row["outcome"] == "unchanged"
    assert (row["previous_source_labels"], row["current_source_labels"]) == (
        "SYNTH Downtown", "SYNTH Downtown|SYNTH Thurlow")


def test_a_zero_previous_price_is_counted_separately_and_never_infinite() -> None:
    result = classify([*filler([0, 1]), offer(P(0), 0), offer(P(1), 2500)], [timeline([0, 1])])
    check_invariants(result)
    row = product(result).iloc[0]
    assert row["outcome"] == "increase" and bool(row["zero_baseline"]) and row["change_cents"] == 2500
    assert result.report.overall.zero_baseline == 1
    assert not any(isinstance(v, float) and np.isinf(v) for v in result.candidates.to_numpy().ravel())
    zero_both = classify([offer(P(0), 0), offer(P(1), 0)], [timeline([0, 1])])
    assert outcomes(zero_both) == [(P(0), P(1), "unchanged")] and zero_both.report.overall.zero_baseline == 1


# ============================================================================ adjacency


def test_exact_one_hour_scheduled_periods_compare() -> None:
    t = timeline([0, 1, 2])
    assert [(i.previous_period, i.current_period) for i in t.intervals] == [(P(0), P(1)), (P(1), P(2))]
    assert parse_scheduled_period(P(1)) - parse_scheduled_period(P(0)) == CAPTURE_STEP == dt.timedelta(hours=1)
    result = classify([offer(P(h), 5000 + h) for h in (0, 1, 2)], [t])
    assert outcomes(result) == [(P(0), P(1), "increase"), (P(1), P(2), "increase")]


def test_a_two_hour_gap_never_compares() -> None:
    with pytest.raises(PriceChangeContractError):
        CaptureInterval(LOC, P(0), P(2))
    for bad in ((P(1), P(0)), (P(0), P(0))):
        with pytest.raises(PriceChangeContractError):
            CaptureInterval(LOC, *bad)
    t = timeline([0, 2])                                                         # nothing scheduled between them
    assert t.intervals == () and t.break_counts == ((IB.NOT_ONE_HOUR.value, 1),)
    result = classify([offer(P(0), 5000), offer(P(2), 9000)], [t])
    assert result.candidates.empty and result.report.overall == OutcomeCounts()


def test_previous_means_the_preceding_capture_not_the_last_observation() -> None:
    rows = [*filler([0, 1, 2]), offer(P(0), 5000), offer(P(2), 6000)]               # absent at t-1
    result = classify(rows, [timeline([0, 1, 2])])
    check_invariants(result)
    assert outcomes(result) == [(P(0), P(1), "disappeared"), (P(1), P(2), "appeared")]
    assert product(result)["change_cents"].tolist() == [None, None]
    assert not ((result.candidates["previous_period"] == P(0)) & (result.candidates["current_period"] == P(2))).any()


@pytest.mark.parametrize("state, reason", [(CS.GOVERNED_EXCLUSION, IB.GOVERNED_EXCLUSION),
                                           (CS.MISSING_CAPTURE, IB.MISSING_CAPTURE)])
def test_an_ineligible_middle_capture_is_a_hard_break(state, reason) -> None:  # type: ignore[no-untyped-def]
    t = timeline([0, 1, 2], states={1: state})
    assert t.intervals == () and t.break_counts == ((reason.value, 2),)
    result = classify([offer(P(0), 5000), offer(P(2), 7000)], [t])
    assert result.candidates.empty
    with pytest.raises(CaptureEvidenceError):                                   # no offer may sit on a break
        classify([offer(P(0)), offer(P(1)), offer(P(2))], [t])


def test_a_change_of_contributing_source_streams_is_a_break() -> None:
    a, b = ("synth-city", "SYNTH Downtown"), ("synth-city", "SYNTH Thurlow")
    t = timeline([0, 1, 2], streams={0: (a,), 1: (a, b), 2: (a, b)})
    assert [(i.previous_period, i.current_period) for i in t.intervals] == [(P(1), P(2))]
    assert t.break_counts == ((IB.SOURCE_STREAMS_CHANGED.value, 1),)


def test_adjacency_never_depends_on_product_presence() -> None:
    t = timeline([0, 1, 2, 3])
    sparse = classify([*filler([0, 1, 2, 3]), offer(P(3), 5000)], [t])
    dense = classify([*filler([0, 1, 2, 3]), *(offer(P(h), 5000) for h in range(4))], [t])
    assert sparse.timelines[0].intervals == dense.timelines[0].intervals == t.intervals
    assert outcomes(sparse) == [(P(2), P(3), "appeared")]


@pytest.mark.parametrize("value", [None, np.nan, 20300304080000, "", "2030-03-04T08:00:00Z", " 20300304T080000Z",
                                   "20300304T080000Z ", "20300304T080000z", "20300304T250000Z", "20300230T080000Z",
                                   "20300304T0800Z", "20300304 080000Z", "２０３００３０４T080000Z",
                                   pd.Timestamp("2030-03-04 08:00", tz="UTC"),
                                   dt.datetime(2030, 3, 4, 8, tzinfo=dt.timezone.utc)])
def test_invalid_or_missing_scheduled_periods_fail_closed(value) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(PriceChangeContractError):
        parse_scheduled_period(value)
    with pytest.raises(PriceChangeContractError):
        ScheduledCapture(value, CS.ELIGIBLE, (LOC,))
    with pytest.raises(PriceChangeContractError):
        classify([offer(P(0)), offer(value)], [timeline([0, 1])])


def test_the_canonical_period_parses_to_an_aware_utc_instant() -> None:
    instant = parse_scheduled_period("20300304T080000Z")
    assert instant == dt.datetime(2030, 3, 4, 8, tzinfo=dt.timezone.utc) and instant.utcoffset() == dt.timedelta(0)


def test_timelines_and_captures_reject_malformed_structure() -> None:
    with pytest.raises(PriceChangeContractError):
        timeline([1, 0])                                                        # out of order
    with pytest.raises(PriceChangeContractError):
        timeline([0, 0])                                                        # duplicate
    with pytest.raises(PriceChangeContractError):
        ScheduledCapture(P(0), CS.ELIGIBLE, ())
    with pytest.raises(PriceChangeContractError):
        ScheduledCapture(P(0), "eligible", (LOC,))  # type: ignore[arg-type]
    with pytest.raises(PriceChangeContractError):
        timeline([0], streams={0: (("another-city", "SYNTH Branch"),)})
    with pytest.raises(PriceChangeContractError):
        LocationCaptureTimeline(("synth-city", " SYNTH Downtown"), ())


# ============================================================================ offers and evidence


def test_every_eligible_capture_needs_offers_and_offers_need_an_eligible_capture() -> None:
    with pytest.raises(CaptureEvidenceError):
        classify([offer(P(0))], [timeline([0, 1])])                              # eligible capture with no offers
    with pytest.raises(CaptureEvidenceError):
        classify([offer(P(0)), offer(P(1)), offer(P(5))], [timeline([0, 1])])    # outside the schedule
    with pytest.raises(CaptureEvidenceError):
        classify([offer(P(0)), offer(P(1)), offer(P(0), loc=OTHER)], [timeline([0, 1])])


@pytest.mark.parametrize("changes", [{"price_cents": -1}, {"price_cents": 50.0}, {"price_cents": True},
                                     {"price_cents": None}, {"car_name": None}, {"car_name": ""},
                                     {"seats": 5}, {"currency": np.nan}, {"pickup_date": "2030-04-01"},
                                     {"pickup_date": pd.Timestamp("2030-04-01")}])
def test_malformed_offers_fail_closed(changes) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(PriceChangeContractError):
        classify([offer(P(0)), offer(P(1), **changes)], [timeline([0, 1])])


def test_duplicate_canonical_offers_and_missing_columns_fail_closed() -> None:
    with pytest.raises(PriceChangeContractError):
        classify([offer(P(0)), offer(P(0)), offer(P(1))], [timeline([0, 1])])
    with pytest.raises(PriceChangeContractError):
        classify_price_change_candidates(frame([offer(P(0)), offer(P(1))]).drop(columns="currency"),
                                         [timeline([0, 1])])
    with pytest.raises(PriceChangeContractError):
        classify([offer(P(0)), offer(P(1))], [timeline([0, 1]), timeline([0, 1])])


def test_input_row_order_never_changes_the_result() -> None:
    rows = [*filler(range(4)), *filler(range(4), OTHER), offer(P(0), 5000), offer(P(1), 5000), offer(P(1), 5100),
            offer(P(2), 4000), offer(P(3), 4500), offer(P(1), 800, name="SYNTH Car B"),
            offer(P(2), 800, name="SYNTH Car B", currency="US$"), offer(P(3), 1200, loc=OTHER, name="SYNTH Car C")]
    timelines = [timeline(range(4)), timeline(range(4), loc=OTHER)]
    reference = classify(rows, timelines)
    for seed in range(5):
        shuffled = frame(rows).sample(frac=1.0, random_state=seed)
        candidates, summaries = classify_price_change_candidates(shuffled, list(reversed(timelines)))
        pd.testing.assert_frame_equal(candidates, reference.candidates)
        assert summaries == reference.report.locations
    check_invariants(reference)


def test_input_frames_are_never_modified() -> None:
    offers = frame([*filler([0, 1]), offer(P(0), 5000), offer(P(1), 5100)])
    before = offers.copy(deep=True)
    classify_price_change_candidates(offers, [timeline([0, 1])])
    pd.testing.assert_frame_equal(offers, before)


# ============================================================================ result invariants


def test_candidate_and_terminal_counts_reconcile_and_keys_are_unique() -> None:
    rows = [*filler(range(5)), offer(P(0), 5000), offer(P(1), 5100), offer(P(2), 5100), offer(P(3), 4000),
            offer(P(1), 10, name="SYNTH Car B"), offer(P(1), 11, name="SYNTH Car B"), offer(P(2), 12, name="SYNTH Car B"),
            offer(P(4), 0, name="SYNTH Car C")]
    result = classify(rows, [timeline(range(5))])
    check_invariants(result)
    c = result.report.overall
    # filler: 4 unchanged; Car A: increase, unchanged, decrease, disappeared;
    # Car B: ambiguous, ambiguous, disappeared; Car C: appeared.
    assert (c.candidates, c.unchanged, c.increase, c.decrease, c.appeared, c.disappeared, c.ambiguous) == (
        12, 5, 1, 1, 1, 2, 2)


def test_outcome_counts_and_summaries_enforce_accounting() -> None:
    with pytest.raises(PriceChangeContractError):
        OutcomeCounts(candidates=2, unchanged=1)
    with pytest.raises(PriceChangeContractError):
        OutcomeCounts(candidates=1, appeared=1, zero_baseline=1)
    with pytest.raises(PriceChangeContractError):
        OutcomeCounts(candidates=-1)
    assert OutcomeCounts.of([T.INCREASE, T.APPEARED]) + OutcomeCounts.of([T.AMBIGUOUS]) == OutcomeCounts(
        3, increase=1, appeared=1, ambiguous=1)
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateReport(status=PriceChangeStatus.BLOCKED)
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateReport(status=PriceChangeStatus.COMPLETED, blockers=(PB.PRICING_NOT_READY,),
                                   overall=OutcomeCounts())


@pytest.mark.parametrize("tamper", [
    lambda f: f.assign(outcome="increase"),                                    # wrong outcome for the prices
    lambda f: f.assign(previous_price_cents=[None] * len(f)),
    lambda f: f.assign(current_period=P(9)),                                   # not one hour / not an interval
    lambda f: f.assign(change_cents=0),
    lambda f: f.assign(zero_baseline=True),
    lambda f: pd.concat([f, f], ignore_index=True),                            # duplicate event keys
    lambda f: f.drop(columns="zero_baseline"),
    lambda f: f.assign(car_name=None),
])
def test_tampered_candidate_frames_are_refused(tamper) -> None:  # type: ignore[no-untyped-def]
    good = classify([offer(P(0), 5000), offer(P(1), 4000)], [timeline([0, 1])])
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateResult(good.report, tamper(good.candidates.copy()), good.timelines)


def test_ambiguous_candidates_never_expose_a_selected_price() -> None:
    good = classify([offer(P(0), 5000), offer(P(0), 5100), offer(P(1), 4000)], [timeline([0, 1])])
    assert good.candidates.loc[0, "outcome"] == "ambiguous"
    for column in ("previous_price_cents", "current_price_cents", "change_cents"):
        tampered = good.candidates.copy()
        tampered.loc[0, column] = 5000
        with pytest.raises(PriceChangeContractError):
            PriceChangeCandidateResult(good.report, tampered, good.timelines)


def test_a_candidate_outside_the_timelines_is_refused() -> None:
    good = classify([offer(P(0), 5000), offer(P(1), 4000)], [timeline([0, 1])])
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateResult(good.report, good.candidates, (timeline([0, 1], states={1: CS.MISSING_CAPTURE}),))


# ============================================================================ end to end (scheduled machinery)


START = dt.datetime(2030, 3, 4, 8)          # synthetic local start (no DST transition nearby)
SYN_ZONES = {"alpha": "America/Edmonton", "beta": "America/Toronto"}


def local_hour(city: str, h: int) -> tuple[dt.datetime, dt.timedelta, str]:
    local = START + dt.timedelta(hours=h)
    aware = local.replace(tzinfo=ZoneInfo(SYN_ZONES[city]))
    return local, aware.utcoffset(), format_utc_instant(aware)


def car_row(stream, job, h, name="SYNTH Car A", price=50.0, currency="CA$", basis="day"):  # type: ignore[no-untyped-def]
    finished = (START + dt.timedelta(hours=h, minutes=7, seconds=13)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return {"job_id": job, "city": stream[0], "location": stream[1], "job_finished_at": finished,
            "scraped_at": "SYNTH-OBSERVATION", "car_name": name, "car_type": "SYNTH Compact",
            "transmission": "SYNTH Automatic", "seats": "5", "bags": "2", "pickup_date": "2030-04-01",
            "return_date": "2030-04-03", "price_num": price, "price_per_day": f"{currency}{price:,.2f}/{basis}"}


def synthetic_world(hours=3, products=None, absent_jobs=(), excluded=None, excused=None, drop_streams=()):  # type: ignore[no-untyped-def]
    """Fabricated jobs and cars run through the real schedule, exclusion, population and canonical-offer machinery.

    ``products`` maps ``(stream, hour)`` to extra ``(name, price)`` offers; every
    present stream-period also carries a constant filler product.
    """
    jobs, cars = [], []
    for city in ("alpha", "beta"):
        for h in range(hours):
            if (city, h) in absent_jobs:
                continue
            job = f"SYNTH-JOB-{city}-{h}"
            finished = (START + dt.timedelta(hours=h, minutes=7, seconds=13)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            jobs.append({"job_id": job, "city": city, "finished_at": finished})
            for stream in SYN_STREAMS:
                if stream[0] != city or (stream, h) in drop_streams:
                    continue
                cars.append(car_row(stream, job, h, name="SYNTH Filler", price=10.0))
                for name, price, *unit in (products or {}).get((stream, h), ()):
                    cars.append(car_row(stream, job, h, name=name, price=price, **dict(unit)))
    exclusions, stream_exceptions = (), ()
    if excluded is not None:
        city, h = excluded
        local, offset, text = local_hour(city, h)
        exclusions = (ParentCaptureExclusion(
            city=city, streams=tuple(s for s in SYN_STREAMS if s[0] == city), period_start_utc=text,
            local_start=local, utc_offset=offset, reason="SYNTH incomplete capture",
            authority_kind=AuthorityKind.COLLECTION_OWNER, reference="SYNTH-REFERENCE", schedule_version="synth_v1"),)
    if excused is not None:
        city, h = excused
        local, offset, text = local_hour(city, h)
        stream_exceptions = tuple(StreamScheduleException(
            stream=s, period_start_utc=text, local_start=local, utc_offset=offset,
            failure=ScheduleFailureKind.PARENT_JOB_ABSENT, reason="SYNTH outage",
            authority_kind=AuthorityKind.COLLECTION_OWNER, reference="SYNTH-REFERENCE", schedule_version="synth_v1")
            for s in SYN_STREAMS if s[0] == city)
    exceptions = (ScheduleExceptions(model=ExceptionsModel.LISTED_EXCEPTIONS, exceptions=stream_exceptions,
                                     parent_capture_exclusions=exclusions)
                  if exclusions or stream_exceptions else None)
    schedule = synth_schedule(start=START, end=START + dt.timedelta(hours=hours - 1), exceptions=exceptions)
    jobs_df, cars_df = pd.DataFrame(jobs), pd.DataFrame(cars)
    scheduled = assess_per_stream_scheduled_coverage(jobs_df, cars_df, schedule=schedule, contract=SYN_CONTRACT,
                                                     relationship=REL)

    def status(frame_: pd.DataFrame, excluded_mask: np.ndarray, assigned: pd.Series) -> tuple[str, ...]:
        return tuple(E.GOVERNED_EXCLUSION.value if x else E.ELIGIBLE.value if a else E.CAPTURE_PERIOD_UNASSIGNED.value
                     for x, a in zip(excluded_mask, assigned.notna()))

    population = PricingPopulation(
        binding=frame_binding(jobs_df, cars_df),
        parent_status=status(jobs_df, scheduled.capture_exclusions.parent_mask(jobs_df),
                             scheduled.capture_periods.parent_periods(jobs_df)),
        detail_status=status(cars_df, scheduled.capture_exclusions.detail_mask(cars_df),
                             scheduled.capture_periods.detail_periods(cars_df)))
    offers = assess_canonical_offers(jobs_df, cars_df, population=population, scheduled=scheduled, policy=POLICY)
    readiness = PricingReadinessReport(
        blocking_reasons=(), location_policy=assess_location_policy(
            VANCOUVER_LOCATION_POLICY, None, apply_location_policy(cars_df, VANCOUVER_LOCATION_POLICY)),
        scheduled_coverage=scheduled, canonical_offers=offers)
    return dict(jobs=jobs_df, cars=cars_df, readiness=readiness, scheduled=scheduled, canonical_offers=offers)


def run(w: dict, **overrides) -> PriceChangeCandidateResult:  # type: ignore[no-untyped-def]
    args = {**w, **overrides}
    return assess_price_change_candidates(args.pop("jobs"), args.pop("cars"), **args)


def summary(result: PriceChangeCandidateResult, stream):  # type: ignore[no-untyped-def]
    return next(s for s in result.report.locations if s.canonical_location == stream)


def test_end_to_end_synthetic_world_completes_from_canonical_offers() -> None:
    w = synthetic_world(products={(ALPHA_DOWN, 0): [("SYNTH Car A", 50.0)], (ALPHA_DOWN, 1): [("SYNTH Car A", 55.0)],
                                  (ALPHA_DOWN, 2): [("SYNTH Car A", 52.5)]})
    before = (w["jobs"].copy(deep=True), w["cars"].copy(deep=True))
    result = run(w)
    assert result.completed and result.report.blockers == ()
    check_invariants(result)
    assert outcomes(result) == [(local_hour("alpha", 0)[2], local_hour("alpha", 1)[2], "increase"),
                                (local_hour("alpha", 1)[2], local_hour("alpha", 2)[2], "decrease")]
    assert product(result)["change_cents"].tolist() == [500, -250]
    assert [s.canonical_location for s in result.report.locations] == sorted(SYN_STREAMS)
    assert all((s.scheduled_periods, s.eligible_periods, s.intervals) == (3, 3, 2) for s in result.report.locations)
    pd.testing.assert_frame_equal(w["jobs"], before[0]), pd.testing.assert_frame_equal(w["cars"], before[1])


def test_governed_excluded_middle_capture_is_a_hard_break_without_mass_turnover() -> None:
    """An eligible capture, the governed INCOMPLETE_PARENT_CAPTURE, a later eligible capture."""
    products = {(s, h): [("SYNTH Car A", 50.0 + h), ("SYNTH Car B", 80.0)] for s in SYN_STREAMS for h in (0, 2)}
    products[(ALPHA_AIR, 1)] = [("SYNTH Car A", 999.0)]                         # the incomplete capture's rows
    w = synthetic_world(products=products, excluded=("alpha", 1), drop_streams={(ALPHA_DOWN, 1)})
    scheduled = w["scheduled"]
    assert scheduled.is_valid and scheduled.excluded_periods == 2 and scheduled.excluded_parent_captures == 1
    assert w["canonical_offers"].out_of_scope_rows > 0                          # excluded rows never enter
    result = run(w)
    assert result.completed
    check_invariants(result)
    excluded_period = local_hour("alpha", 1)[2]
    for stream in (ALPHA_AIR, ALPHA_DOWN):
        s = summary(result, stream)
        assert (s.scheduled_periods, s.eligible_periods, s.excluded_periods, s.intervals) == (3, 2, 1, 0)
        assert s.breaks == ((IB.GOVERNED_EXCLUSION.value, 2),)
        assert s.counts == OutcomeCounts()                                       # no appearances or disappearances
    alpha = result.candidates[result.candidates["canonical_city"] == "alpha"]
    assert alpha.empty
    for t in (t for t in result.timelines if t.canonical_location[0] == "alpha"):
        assert t.periods(CS.GOVERNED_EXCLUSION) == (excluded_period,)
    assert summary(result, BETA_DOWN).counts.candidates > 0                      # the other city is unaffected
    for t in (t for t in result.timelines if t.canonical_location[0] == "alpha"):
        assert all(excluded_period not in (i.previous_period, i.current_period) for i in t.intervals)


def test_an_excused_missing_capture_breaks_and_a_product_back_after_it_is_never_a_change() -> None:
    products = {(ALPHA_DOWN, 0): [("SYNTH Car A", 50.0)], (ALPHA_DOWN, 2): [("SYNTH Car A", 70.0)],
                (ALPHA_DOWN, 3): [("SYNTH Car A", 70.0)]}
    w = synthetic_world(hours=4, products=products, absent_jobs={("alpha", 1)}, excused=("alpha", 1))
    assert w["scheduled"].is_valid
    result = run(w)
    check_invariants(result)
    s = summary(result, ALPHA_DOWN)
    assert (s.missing_periods, s.intervals) == (1, 1) and s.breaks == ((IB.MISSING_CAPTURE.value, 2),)
    assert outcomes(result) == [(local_hour("alpha", 2)[2], local_hour("alpha", 3)[2], "unchanged")]


def test_currency_change_end_to_end_is_never_a_price_movement() -> None:
    products = {(BETA_DOWN, 0): [("SYNTH Car A", 50.0, ("currency", "CA$"))],
                (BETA_DOWN, 1): [("SYNTH Car A", 50.0, ("currency", "US$"))]}
    result = run(synthetic_world(hours=2, products=products))
    assert sorted(product(result)["outcome"]) == ["appeared", "disappeared"]


def test_end_to_end_blocks_fail_closed() -> None:
    w = synthetic_world()
    not_ready = dataclasses.replace(w["readiness"], blocking_reasons=(PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE,))
    blocked = run(w, readiness=not_ready)
    assert blocked.report.blockers == (PB.PRICING_NOT_READY,) and blocked.candidates is None
    assert blocked.report.readiness_blockers == (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE.value,)
    other = synthetic_world(hours=2)
    assert PB.READINESS_EVIDENCE_MISMATCH in run(w, canonical_offers=other["canonical_offers"]).report.blockers
    assert PB.FRAME_BINDING_MISMATCH in run(w, cars=w["cars"].assign(price_num=1.0)).report.blockers
    invalid = synthetic_world(absent_jobs={("alpha", 1)})                       # unexcused missing capture
    assert not invalid["scheduled"].is_valid
    assert PB.SCHEDULE_EVIDENCE_INVALID in run(invalid).report.blockers
    with pytest.raises(TypeError):
        run(w, scheduled=None)


def test_inconsistent_capture_evidence_blocks() -> None:
    w = synthetic_world()
    scheduled = w["scheduled"]
    stream = scheduled.streams[0]
    tampered = dataclasses.replace(scheduled, streams=(dataclasses.replace(stream, covered=stream.covered - 1),
                                                       *scheduled.streams[1:]))
    with pytest.raises(CaptureEvidenceError):
        capture_timelines(tampered, POLICY)
    with pytest.raises(TypeError):
        capture_timelines(scheduled, None)


def test_the_pipeline_entry_point_fails_closed_when_pricing_is_not_ready(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from conftest import contract_columns, write_synthetic_csv

    from ql2_sixt_canada_analysis.pricing_pipeline import run_pricing_pipeline
    from ql2_sixt_canada_analysis.schemas import DatasetKey

    for key in DatasetKey:
        write_synthetic_csv(tmp_path / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    result = price_change_candidates_from_pipeline(run_pricing_pipeline(tmp_path))
    assert result.report.blockers == (PB.PRICING_NOT_READY,) and result.report.readiness_blockers
    assert result.candidates is None and result.timelines is None
    with pytest.raises(TypeError):
        price_change_candidates_from_pipeline(object())


# ============================================================================ confidentiality and packaging


def test_no_raw_job_identifier_is_required_or_exposed() -> None:
    assert not set(FORBIDDEN_TIMESTAMP_SOURCES) & set(CANDIDATE_COLUMNS)
    rows = frame([offer(P(0)), offer(P(1))])
    assert "job_id" not in rows.columns
    candidates, _ = classify_price_change_candidates(rows, [timeline([0, 1])])
    assert len(candidates) == 1
    result = run(synthetic_world(products={(ALPHA_DOWN, 1): [("SYNTH Car A", 50.0)]}))
    shown = result.candidates.to_string() + repr(result)
    assert "SYNTH-JOB" not in shown and "job_id" not in result.candidates.columns


def test_event_level_frames_stay_out_of_repr_and_are_never_written(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    w = synthetic_world(products={(ALPHA_DOWN, 0): [("SYNTH Car A", 50.0)], (ALPHA_DOWN, 1): [("SYNTH Car A", 61.0)]})
    result = run(w)
    assert result.completed and len(result.candidates)
    shown = repr(result) + repr(result.report) + str(result.report) + repr(result.timelines)
    assert "SYNTH Car" not in shown and "SYNTH Filler" not in shown and "SYNTH-JOB" not in shown
    assert not re.search(r"2030\d{4}T\d{6}Z|2030-0[34]-\d\d", shown)            # no periods or rental dates
    assert "DataFrame" not in repr(result) and "price_cents" not in repr(result) and "6100" not in shown
    assert os.listdir(tmp_path) == []


def test_importing_the_module_performs_no_io_or_plotting() -> None:
    """No project file is read, nothing is written and no plotting or statistics library is loaded."""
    root = str(Path(__file__).resolve().parents[1])
    code = ("import builtins, io, os, pathlib, sys\n"
            f"ROOT = {root!r}\n"
            "real = builtins.open\n"
            "def guarded(file, mode='r', *a, **k):\n"
            "    path = os.path.abspath(os.fspath(file)) if isinstance(file, (str, bytes, os.PathLike)) else ''\n"
            "    if any(c in mode for c in 'wax+') or str(path).startswith(ROOT):\n"
            "        raise AssertionError('I/O during import')\n"
            "    return real(file, mode, *a, **k)\n"
            "builtins.open = io.open = guarded\n"
            "import ql2_sixt_canada_analysis.price_change_events as m\n"
            "print('matplotlib' in sys.modules, 'scipy' in sys.modules, 'seaborn' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False", "False"]


def test_public_package_exports_are_complete() -> None:
    assert len(pce.__all__) == len(set(pce.__all__))
    public = {n for n in vars(pce) if not n.startswith("_") and getattr(getattr(pce, n), "__module__", None)
              == pce.__name__}
    assert public <= set(pce.__all__)
    for name in pce.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(pce, name)
    assert {o.value for o in T} == {"unchanged", "increase", "decrease", "appeared", "disappeared", "ambiguous"}
