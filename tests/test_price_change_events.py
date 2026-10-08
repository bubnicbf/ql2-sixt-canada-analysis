"""Price-change events: the locked event contract and the event-construction engine.

Every observation is fabricated: ``SYNTH-*`` jobs and products, synthetic
branches (``synth-city``), synthetic 2030 capture periods, rental dates and
prices. The only committed values read are approved configuration (the
source-stream keys, city time zones, location authority and canonical-offer
policy of the current authority record).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis import price_change_events as pce
from ql2_sixt_canada_analysis.authority_decisions import AuthorityKind
from ql2_sixt_canada_analysis.canonical_offers import (
    APPROVED_PRODUCT_COLUMNS,
    CanonicalOfferPolicy,
    CanonicalOfferStatus,
    assess_canonical_offers,
    current_canonical_offer_policy,
)
from ql2_sixt_canada_analysis.collection_schedule import (
    CityTimezoneMap,
    ExceptionsModel,
    ParentCaptureExclusion,
    PerStreamSchedule,
    ScheduleAuthorityStatus,
    ScheduleExceptions,
    ScheduleFailureKind,
    SharingMode,
    StreamSchedule,
    StreamScheduleException,
    assess_per_stream_scheduled_coverage,
    current_per_stream_schedule,
    format_utc_instant,
)
from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
from ql2_sixt_canada_analysis.location_authority import current_location_authority
from ql2_sixt_canada_analysis.price_change_events import (
    CANDIDATE_COLUMNS,
    CAPTURE_STEP,
    EVENT_IDENTITY_COLUMNS,
    EVENT_INTERVAL_COLUMNS,
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
    UnknownCanonicalLocationError,
    approved_canonical_locations,
    assess_price_change_candidates,
    capture_timelines,
    change_percent,
    classify_endpoint_offers,
    classify_price_change_candidates,
    parse_scheduled_period,
    price_change_candidates_from_pipeline,
    price_change_cents,
    run_price_change_events,
)
from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult
from ql2_sixt_canada_analysis.pricing_population import DetailEligibility as E, PricingPopulation, frame_binding
from ql2_sixt_canada_analysis.readiness import (
    PricingBlocker,
    PricingReadinessReport,
    apply_location_policy,
    assess_location_policy,
)
from ql2_sixt_canada_analysis.schemas import JOB_DETAIL_RELATIONSHIP as REL, VANCOUVER_LOCATION_POLICY

PREV, CUR = EVENT_INTERVAL_COLUMNS
POLICY = current_canonical_offer_policy()
LOC, OTHER = ("synth-city", "SYNTH Downtown"), ("synth-city", "SYNTH Airport")
ELSEWHERE = ("synth-town", "SYNTH Downtown")
PICKUP, RETURN = dt.date(2030, 4, 1), dt.date(2030, 4, 3)


def P(hour: int) -> str:
    """A synthetic scheduled period (UTC start)."""
    return (dt.datetime(2030, 3, 4, 0, tzinfo=dt.timezone.utc) + dt.timedelta(hours=hour)).strftime("%Y%m%dT%H%M%SZ")


# ============================================================================ pure-engine fixtures


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
    return pd.DataFrame(rows, columns=list(offer(P(0)))) if not rows else pd.DataFrame(rows)


def timeline(hours, loc=LOC, states=None, streams=None) -> LocationCaptureTimeline:  # type: ignore[no-untyped-def]
    states = states or {}
    streams = streams or {}
    return LocationCaptureTimeline(loc, tuple(
        ScheduledCapture(P(h), states.get(h, CS.ELIGIBLE), streams.get(h, (loc,))) for h in hours))


def filler(hours, loc=LOC) -> list[dict]:  # type: ignore[no-untyped-def]
    """A constant product present at every listed capture."""
    return [offer(P(h), 1000, loc=loc, name="SYNTH Filler") for h in hours]


def classify(rows, timelines, locations=None) -> PriceChangeCandidateResult:  # type: ignore[no-untyped-def]
    candidates, summaries = classify_price_change_candidates(frame(rows), timelines, locations)
    report = PriceChangeCandidateReport(status=PriceChangeStatus.COMPLETED, offers_assessed=len(rows),
                                        locations=summaries, overall=sum((s.counts for s in summaries),
                                                                         OutcomeCounts()))
    return PriceChangeCandidateResult(report, candidates, tuple(timelines))     # every invariant enforced


def product(result: PriceChangeCandidateResult, name: str = "SYNTH Car A") -> pd.DataFrame:
    return result.candidates[result.candidates["car_name"] == name].reset_index(drop=True)


def outcomes(result: PriceChangeCandidateResult, name: str = "SYNTH Car A") -> list[tuple[str, str, str]]:
    rows = product(result, name)
    return list(zip(rows[PREV], rows[CUR], rows["outcome"]))


def check_invariants(result: PriceChangeCandidateResult) -> None:
    """The strict accounting every completed result must satisfy."""
    c, rows = result.report.overall, result.candidates
    assert c.candidates == len(rows) == sum(getattr(c, o.value) for o in T)
    assert c.comparable == c.unchanged + c.increase + c.decrease == c.percent_valid + c.zero_denominator
    assert c.changed == c.increase + c.decrease
    assert dict(rows["outcome"].value_counts()) == {o.value: getattr(c, o.value) for o in T if getattr(c, o.value)}
    assert int(rows["percent_valid"].sum()) == c.percent_valid and int(rows["zero_denominator"].sum()) == \
        c.zero_denominator
    assert not (rows["percent_valid"] & rows["zero_denominator"]).any()
    assert sum((s.counts for s in result.report.locations), OutcomeCounts()) == c
    assert not rows.duplicated(list(EVENT_KEY_COLUMNS)).any()
    assert tuple(rows.columns) == CANDIDATE_COLUMNS
    for value in rows["change_percent"]:
        assert value is None or (isinstance(value, float) and math.isfinite(value))
    for s in result.report.locations:
        assert s.intervals + sum(n for _, n in s.breaks) == max(s.scheduled_periods - 1, 0)
    for a, b in zip(rows[PREV], rows[CUR]):
        assert parse_scheduled_period(b) - parse_scheduled_period(a) == CAPTURE_STEP


# ============================================================================ outcomes and exact metrics


@pytest.mark.parametrize(("before", "after", "expected", "change", "percent"), [
    (5000, 5000, T.UNCHANGED, 0, 0.0),
    (5000, 6250, T.INCREASE, 1250, 25.0),
    (8000, 6000, T.DECREASE, -2000, -25.0)])
def test_one_offer_at_both_endpoints_compares_exact_cents(before, after, expected, change, percent) -> None:  # type: ignore[no-untyped-def]
    result = classify([offer(P(0), before), offer(P(1), after)], [timeline([0, 1])])
    check_invariants(result)
    row = result.candidates.iloc[0]
    assert len(result.candidates) == 1 and row["outcome"] == expected.value
    assert (row["previous_price_cents"], row["current_price_cents"], row["change_cents"]) == (before, after, change)
    assert (row["previous_price"], row["current_price"], row["change_dollars"]) == (
        before / 100, after / 100, change / 100)
    assert row["change_percent"] == percent and bool(row["percent_valid"]) and not row["zero_denominator"]
    assert (row["previous_offer_count"], row["current_offer_count"]) == (1, 1)
    assert classify_endpoint_offers([before], [after]) is expected and price_change_cents(before, after) == change


def test_the_percentage_denominator_is_the_previous_price() -> None:
    up, down = classify([offer(P(0), 4000), offer(P(1), 5000)], [timeline([0, 1])]), \
        classify([offer(P(0), 5000), offer(P(1), 4000)], [timeline([0, 1])])
    assert up.candidates.loc[0, "change_percent"] == 25.0                       # 1000 / 4000, not / 5000
    assert down.candidates.loc[0, "change_percent"] == -20.0                    # -1000 / 5000, not / 4000


def test_a_non_round_percentage_is_exact_from_cents_not_display_dollars() -> None:
    result = classify([offer(P(0), 3333), offer(P(1), 10001)], [timeline([0, 1])])
    row = result.candidates.iloc[0]
    exact = float(Fraction(100 * (10001 - 3333), 3333))
    assert row["change_percent"] == exact == change_percent(3333, 10001)
    assert row["change_percent"] != round(100 * (100.01 - 33.33) / 33.33, 2)    # no display rounding
    assert row["change_cents"] == 6668 and row["change_dollars"] == 66.68
    tiny = classify([offer(P(0), 1999), offer(P(1), 2000)], [timeline([0, 1])])
    assert tiny.candidates.loc[0, "outcome"] == "increase" and tiny.candidates.loc[0, "change_cents"] == 1


@pytest.mark.parametrize(("after", "expected"), [(2500, T.INCREASE), (0, T.UNCHANGED)])
def test_a_zero_previous_price_keeps_cents_but_has_no_percentage(after, expected) -> None:  # type: ignore[no-untyped-def]
    result = classify([*filler([0, 1]), offer(P(0), 0), offer(P(1), after)], [timeline([0, 1])])
    check_invariants(result)
    row = product(result).iloc[0]
    assert row["outcome"] == expected.value and row["change_cents"] == after and row["change_dollars"] == after / 100
    assert row["change_percent"] is None and bool(row["zero_denominator"]) and not row["percent_valid"]
    assert result.report.overall.zero_denominator == 1 and result.report.overall.percent_valid == 1   # filler
    assert change_percent(0, after) is None


def test_no_valid_result_ever_contains_an_infinite_or_nan_percentage() -> None:
    rows = [offer(P(0), c, name=f"SYNTH Car {i}") for i, c in enumerate((0, 1, 7, 99_999_999))]
    rows += [offer(P(1), c, name=f"SYNTH Car {i}") for i, c in enumerate((99_999_999, 0, 7, 1))]
    result = classify(rows, [timeline([0, 1])])
    check_invariants(result)
    values = [v for v in result.candidates["change_percent"] if v is not None]
    assert len(values) == 3 and all(math.isfinite(v) for v in values)


def test_missing_prior_identity_is_an_appearance_and_missing_current_is_a_disappearance() -> None:
    result = classify([*filler([0, 1, 2]), offer(P(1), 4000)], [timeline([0, 1, 2])])
    check_invariants(result)
    assert outcomes(result) == [(P(0), P(1), "appeared"), (P(1), P(2), "disappeared")]
    appeared, disappeared = product(result).iloc[0], product(result).iloc[1]
    assert (appeared["previous_offer_count"], appeared["current_offer_count"]) == (0, 1)
    assert appeared["previous_price_cents"] is None and appeared["previous_price"] is None
    assert appeared["current_price_cents"] == 4000 and appeared["current_price"] == 40.0
    assert (disappeared["previous_offer_count"], disappeared["current_offer_count"]) == (1, 0)
    assert disappeared["current_price_cents"] is None and disappeared["previous_price_cents"] == 4000
    for row in (appeared, disappeared):
        assert [row[c] for c in ("change_cents", "change_dollars", "change_percent")] == [None] * 3
        assert not row["percent_valid"] and not row["zero_denominator"]


def test_the_first_capture_is_a_baseline_without_appearances() -> None:
    result = classify([offer(P(0), 5000), offer(P(0), 900, name="SYNTH Car B"), offer(P(1), 5000),
                       offer(P(1), 900, name="SYNTH Car B")], [timeline([0, 1])])
    assert result.report.overall.appeared == 0 and result.report.overall.unchanged == 2
    assert set(result.candidates[PREV]) == {P(0)}                                 # nothing ends at the first capture
    single = classify([offer(P(0))], [timeline([0])])
    assert single.candidates.empty and single.report.overall == OutcomeCounts()


@pytest.mark.parametrize(("before", "after"), [((5000, 5100), (5000,)), ((5000,), (4900, 5000)),
                                               ((5000, 5100), (5200, 5300)), ((5000, 5100), ()), ((), (1, 2, 3))])
def test_several_offers_at_either_endpoint_are_ambiguous_without_expansion(before, after) -> None:  # type: ignore[no-untyped-def]
    rows = [*filler([0, 1]), *(offer(P(0), c) for c in before), *(offer(P(1), c) for c in after)]
    result = classify(rows, [timeline([0, 1])])
    check_invariants(result)
    rows_ = product(result)
    assert len(rows_) == 1 and rows_.loc[0, "outcome"] == "ambiguous"                # one candidate, no product
    assert (rows_.loc[0, "previous_offer_count"], rows_.loc[0, "current_offer_count"]) == (len(before), len(after))
    selected = ("previous_price_cents", "current_price_cents", "change_cents", "previous_price", "current_price",
                "change_dollars", "change_percent")
    assert rows_.loc[0, list(selected)].tolist() == [None] * len(selected)
    assert not rows_.loc[0, "zero_denominator"] and not rows_.loc[0, "percent_valid"]
    assert result.report.overall.ambiguous == 1 and classify_endpoint_offers(before, after) is T.AMBIGUOUS


@pytest.mark.parametrize(("field", "other"), [
    ("car_name", "SYNTH Car B"), ("car_type", "SYNTH Fullsize"), ("transmission", "SYNTH Manual"), ("seats", "7"),
    ("bags", "3"), ("pickup_date", dt.date(2030, 4, 2)), ("return_date", dt.date(2030, 4, 4)),
    ("car_name", "synth car a")])
def test_different_products_or_rental_periods_never_compare(field, other) -> None:  # type: ignore[no-untyped-def]
    result = classify([*filler([0, 1]), offer(P(0), 5000), offer(P(1), 9000, **{field: other})], [timeline([0, 1])])
    check_invariants(result)
    c = result.report.overall
    assert (c.appeared, c.disappeared, c.changed, c.comparable) == (1, 1, 0, 1)    # filler unchanged only


def test_untrimmed_identity_text_is_malformed_never_trimmed() -> None:
    with pytest.raises(PriceChangeContractError):
        classify([offer(P(0)), offer(P(1), seats="5 ")], [timeline([0, 1])])


@pytest.mark.parametrize("other", [OTHER, ELSEWHERE])
def test_different_canonical_locations_or_cities_never_compare(other) -> None:  # type: ignore[no-untyped-def]
    rows = [*filler([0, 1]), *filler([0, 1], other), offer(P(0), 5000, loc=LOC), offer(P(1), 7000, loc=other)]
    result = classify(rows, [timeline([0, 1]), timeline([0, 1], loc=other)])
    check_invariants(result)
    assert result.report.overall.comparable == 2 and result.report.overall.changed == 0     # the fillers only
    assert (result.report.location(LOC).counts.disappeared, result.report.location(other).counts.appeared) == (1, 1)
    assert not (result.candidates["canonical_city"] + result.candidates["canonical_location"]).duplicated().all()


@pytest.mark.parametrize(("field", "before", "after"), [("currency", "CA$", "US$"), ("price_basis", "day", "week")])
def test_unit_changes_are_a_disappearance_and_an_appearance_never_a_price_movement(field, before, after) -> None:  # type: ignore[no-untyped-def]
    for cents in ((5000, 5000), (5000, 7000), (5000, 3000)):
        rows = [*filler([0, 1]), offer(P(0), cents[0], **{field: before}), offer(P(1), cents[1], **{field: after})]
        result = classify(rows, [timeline([0, 1])])
        check_invariants(result)
        moved = product(result)
        assert sorted(moved["outcome"]) == ["appeared", "disappeared"]
        assert moved["change_cents"].tolist() == [None, None] and result.report.overall.comparable == 1


def test_mixed_units_across_identities_never_block_valid_events() -> None:
    rows = [offer(P(0), 5000), offer(P(1), 5500), offer(P(0), 700, name="SYNTH Car B", currency="US$"),
            offer(P(1), 650, name="SYNTH Car B", currency="US$"), offer(P(0), 300, name="SYNTH Car C", basis="week"),
            offer(P(1), 300, name="SYNTH Car C", basis="week")]
    result = classify(rows, [timeline([0, 1])])
    check_invariants(result)
    c = result.report.overall
    assert (c.increase, c.decrease, c.unchanged, c.appeared, c.disappeared) == (1, 1, 1, 0, 0)


def test_price_is_not_part_of_the_identity_but_currency_and_basis_are() -> None:
    assert "price_cents" not in EVENT_IDENTITY_COLUMNS
    assert EVENT_IDENTITY_COLUMNS == ("canonical_city", "canonical_location", "pickup_date", "return_date",
                                      *APPROVED_PRODUCT_COLUMNS, "currency", "price_basis")
    assert EVENT_KEY_COLUMNS == (*EVENT_IDENTITY_COLUMNS, "previous_scheduled_capture_period",
                                 "current_scheduled_capture_period")
    result = classify([offer(P(0), 100), offer(P(1), 99_999)], [timeline([0, 1])])
    assert outcomes(result) == [(P(0), P(1), "increase")]


def test_source_labels_are_provenance_and_never_split_the_identity() -> None:
    rows = [offer(P(0), 5000, labels="SYNTH Downtown"), offer(P(1), 5000, labels="SYNTH Downtown|SYNTH Thurlow")]
    row = classify(rows, [timeline([0, 1])]).candidates.iloc[0]
    assert row["outcome"] == "unchanged"
    assert (row["previous_source_labels"], row["current_source_labels"]) == (
        "SYNTH Downtown", "SYNTH Downtown|SYNTH Thurlow")


# ============================================================================ interval grid and adjacency


def test_exact_one_hour_scheduled_periods_compare() -> None:
    t = timeline([0, 1, 2])
    assert [(i.previous_period, i.current_period) for i in t.intervals] == [(P(0), P(1)), (P(1), P(2))]
    assert parse_scheduled_period(P(1)) - parse_scheduled_period(P(0)) == CAPTURE_STEP == dt.timedelta(hours=1)
    result = classify([offer(P(h), 5000 + h) for h in (0, 1, 2)], [t])
    assert outcomes(result) == [(P(0), P(1), "increase"), (P(1), P(2), "increase")]


def test_a_two_hour_gap_never_compares() -> None:
    for bad in ((P(0), P(2)), (P(1), P(0)), (P(0), P(0))):
        with pytest.raises(PriceChangeContractError):
            CaptureInterval(LOC, *bad)
    t = timeline([0, 2])                                                         # nothing scheduled between them
    assert t.intervals == () and t.break_counts == ((IB.NOT_ONE_HOUR.value, 1),)
    result = classify([offer(P(0), 5000), offer(P(2), 9000)], [t])
    assert result.candidates.empty and result.report.overall == OutcomeCounts()


def test_previous_means_the_preceding_capture_not_the_last_observation() -> None:
    result = classify([*filler([0, 1, 2]), offer(P(0), 5000), offer(P(2), 6000)], [timeline([0, 1, 2])])
    check_invariants(result)
    assert outcomes(result) == [(P(0), P(1), "disappeared"), (P(1), P(2), "appeared")]
    assert product(result)["change_cents"].tolist() == [None, None]
    assert not ((result.candidates[PREV] == P(0)) & (result.candidates[CUR] == P(2))).any()


def test_an_entirely_empty_valid_capture_creates_turnover_only_across_its_adjacent_intervals() -> None:
    rows = [offer(P(0), 5000), offer(P(0), 900, name="SYNTH Car B"), offer(P(2), 5100),
            offer(P(2), 900, name="SYNTH Car B")]                                 # P(1) is valid but empty
    result = classify(rows, [timeline([0, 1, 2])])
    check_invariants(result)
    assert outcomes(result) == [(P(0), P(1), "disappeared"), (P(1), P(2), "appeared")]
    assert outcomes(result, "SYNTH Car B") == [(P(0), P(1), "disappeared"), (P(1), P(2), "appeared")]
    assert result.report.overall.comparable == 0


@pytest.mark.parametrize("state, reason", [(CS.GOVERNED_EXCLUSION, IB.GOVERNED_EXCLUSION),
                                           (CS.MISSING_CAPTURE, IB.MISSING_CAPTURE)])
def test_an_ineligible_middle_capture_is_a_hard_break(state, reason) -> None:  # type: ignore[no-untyped-def]
    t = timeline([0, 1, 2], states={1: state})
    assert t.intervals == () and t.break_counts == ((reason.value, 2),)
    result = classify([offer(P(0), 5000), offer(P(2), 7000)], [t])
    assert result.candidates.empty and result.report.overall.intervals == 0
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


def test_timelines_and_captures_reject_malformed_structure() -> None:
    for hours in ([1, 0], [0, 0]):
        with pytest.raises(PriceChangeContractError):
            timeline(hours)
    with pytest.raises(PriceChangeContractError):
        ScheduledCapture(P(0), CS.ELIGIBLE, ())
    with pytest.raises(PriceChangeContractError):
        ScheduledCapture(P(0), "eligible", (LOC,))  # type: ignore[arg-type]
    with pytest.raises(PriceChangeContractError):
        timeline([0], streams={0: (("another-city", "SYNTH Branch"),)})
    with pytest.raises(PriceChangeContractError):
        LocationCaptureTimeline(("synth-city", " SYNTH Downtown"), ())


# ============================================================================ offers, locations and empties


def test_offers_need_an_eligible_capture_of_an_approved_location() -> None:
    with pytest.raises(CaptureEvidenceError):
        classify([offer(P(0)), offer(P(1)), offer(P(5))], [timeline([0, 1])])    # outside the schedule
    with pytest.raises(UnknownCanonicalLocationError):
        classify([offer(P(0)), offer(P(1)), offer(P(0), loc=OTHER)], [timeline([0, 1])])


def test_unknown_canonical_locations_fail_closed_rather_than_being_inferred() -> None:
    with pytest.raises(UnknownCanonicalLocationError):
        classify([offer(P(0))], [timeline([0]), timeline([0], loc=OTHER)], locations=[LOC])
    with pytest.raises(CaptureEvidenceError):                                   # approved but no timeline
        classify([offer(P(0))], [timeline([0])], locations=[LOC, OTHER])
    with pytest.raises(UnknownCanonicalLocationError):
        approved_canonical_locations(current_location_authority(),
                                     CanonicalOfferPolicy(CanonicalOfferStatus.NOT_APPROVED))


def test_locations_are_reported_in_authority_order_including_zero_candidate_locations() -> None:
    result = classify([*filler([0, 1]), offer(P(0), loc=OTHER)], [timeline([0, 1]), timeline([0], loc=OTHER)],
                      locations=[OTHER, LOC])
    assert result.report.approved_locations == (OTHER, LOC)
    assert result.report.location(OTHER).counts == OutcomeCounts()               # one capture: baseline only
    assert result.report.location(LOC).counts.candidates == 1


def test_empty_but_valid_populations_return_explicit_zero_counts() -> None:
    result = classify([], [timeline([0, 1, 2]), timeline([0, 1], loc=OTHER)])
    assert result.candidates.empty and tuple(result.candidates.columns) == CANDIDATE_COLUMNS
    assert result.report.overall == OutcomeCounts(intervals=3)
    assert all(s.counts.candidates == 0 for s in result.report.locations)
    assert "mean" not in repr(result.report) and "median" not in repr(result.report)
    nothing = classify([], [])
    assert nothing.report.overall == OutcomeCounts() and nothing.report.locations == ()


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
    reference = classify(rows, timelines, [OTHER, LOC])
    for seed in range(5):
        shuffled = frame(rows).sample(frac=1.0, random_state=seed).set_index(np.arange(len(rows))[::-1])
        candidates, summaries = classify_price_change_candidates(shuffled, list(reversed(timelines)), [OTHER, LOC])
        pd.testing.assert_frame_equal(candidates, reference.candidates)
        assert summaries == reference.report.locations
    assert list(dict.fromkeys(reference.candidates["canonical_location"])) == [OTHER[1], LOC[1]]
    check_invariants(reference)


def test_input_frames_are_never_modified() -> None:
    offers = frame([*filler([0, 1]), offer(P(0), 5000), offer(P(1), 5100)])
    before = offers.copy(deep=True)
    classify_price_change_candidates(offers, [timeline([0, 1])])
    pd.testing.assert_frame_equal(offers, before)


# ============================================================================ result invariants


def test_candidate_and_location_counts_reconcile() -> None:
    rows = [*filler(range(5)), offer(P(0), 5000), offer(P(1), 5100), offer(P(2), 5100), offer(P(3), 4000),
            offer(P(1), 10, name="SYNTH Car B"), offer(P(1), 11, name="SYNTH Car B"), offer(P(2), 12, name="SYNTH Car B"),
            offer(P(4), 0, name="SYNTH Car C"), *filler([0, 1], OTHER), offer(P(1), 0, loc=OTHER),
            offer(P(0), 0, loc=OTHER)]
    result = classify(rows, [timeline(range(5)), timeline([0, 1], loc=OTHER)])
    check_invariants(result)
    loc = result.report.location(LOC).counts
    # filler: 4 unchanged; Car A: increase, unchanged, decrease, disappeared;
    # Car B: ambiguous, ambiguous, disappeared; Car C: appeared.
    assert (loc.intervals, loc.candidates, loc.unchanged, loc.increase, loc.decrease, loc.appeared,
            loc.disappeared, loc.ambiguous, loc.comparable, loc.changed, loc.percent_valid, loc.zero_denominator) == (
        4, 12, 5, 1, 1, 1, 2, 2, 7, 2, 7, 0)
    other = result.report.location(OTHER).counts
    assert (other.intervals, other.unchanged, other.zero_denominator, other.percent_valid) == (1, 2, 1, 1)
    assert result.report.overall == loc + other


def test_outcome_counts_and_reports_enforce_accounting() -> None:
    for bad in (dict(intervals=1, candidates=2, unchanged=1), dict(intervals=1, candidates=1, unchanged=1),
                dict(intervals=1, candidates=1, appeared=1, zero_denominator=1), dict(candidates=-1),
                dict(candidates=1, appeared=1)):
        with pytest.raises(PriceChangeContractError):
            OutcomeCounts(**bad)
    counts = OutcomeCounts.of([T.INCREASE, T.APPEARED], intervals=1) + OutcomeCounts.of([T.AMBIGUOUS], intervals=1)
    assert counts == OutcomeCounts(2, 3, increase=1, appeared=1, ambiguous=1, percent_valid=1)
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateReport(status=PriceChangeStatus.BLOCKED)
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateReport(status=PriceChangeStatus.COMPLETED, blockers=(PB.PRICING_NOT_READY,),
                                   overall=OutcomeCounts())
    blocked = PriceChangeCandidateReport(status=PriceChangeStatus.BLOCKED, blockers=(PB.PRICING_NOT_READY,))
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateResult(blocked, pd.DataFrame(columns=list(CANDIDATE_COLUMNS)))
    good = classify([offer(P(0)), offer(P(1))], [timeline([0, 1])])
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateResult(good.report, None, good.timelines)            # completed needs its frame


def _set(column: str, value: object):  # type: ignore[no-untyped-def]
    def tamper(f: pd.DataFrame) -> pd.DataFrame:
        f[column] = pd.Series([value] * len(f), index=f.index, dtype=object)
        return f
    return tamper


@pytest.mark.parametrize("tamper", [
    _set("outcome", "increase"),                                               # wrong outcome for the prices
    _set("previous_price_cents", None),
    _set(CUR, P(9)),                            # not one hour / not an interval
    _set("change_cents", 0),
    _set("change_dollars", -10.5),
    _set("change_percent", -20.000001),                                       # not the exact percentage
    _set("change_percent", math.inf),
    _set("change_percent", None),
    _set("previous_price", 50.01),
    _set("zero_denominator", True),
    _set("percent_valid", False),
    lambda f: pd.concat([f, f], ignore_index=True),                            # duplicate event keys
    lambda f: f.drop(columns="zero_denominator"),
    lambda f: f[list(reversed(CANDIDATE_COLUMNS))],                            # column order is the contract
    _set("car_name", None),
])
def test_tampered_candidate_frames_are_refused(tamper) -> None:  # type: ignore[no-untyped-def]
    good = classify([offer(P(0), 5000), offer(P(1), 4000)], [timeline([0, 1])])
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateResult(good.report, tamper(good.candidates.copy()), good.timelines)


def test_ambiguous_candidates_never_expose_a_selected_price() -> None:
    good = classify([offer(P(0), 5000), offer(P(0), 5100), offer(P(1), 4000)], [timeline([0, 1])])
    assert good.candidates.loc[0, "outcome"] == "ambiguous"
    for column, value in (("previous_price_cents", 5000), ("current_price_cents", 4000), ("change_cents", -1000),
                          ("previous_price", 50.0), ("current_price", 40.0), ("change_percent", -20.0)):
        tampered = good.candidates.copy()
        tampered[column] = pd.Series([value], dtype=object)
        with pytest.raises(PriceChangeContractError):
            PriceChangeCandidateResult(good.report, tampered, good.timelines)


def test_a_candidate_outside_the_timelines_is_refused() -> None:
    good = classify([offer(P(0), 5000), offer(P(1), 4000)], [timeline([0, 1])])
    with pytest.raises(PriceChangeContractError):
        PriceChangeCandidateResult(good.report, good.candidates, (timeline([0, 1], states={1: CS.MISSING_CAPTURE}),))


# ============================================================================ gated engine (synthetic world)


CONTRACT = current_expected_stream_contract()
AUTHORITY = current_location_authority()
STREAMS = tuple(tuple(k) for k in CONTRACT.expected_keys)
ZONES = {s.city: s.timezone for s in current_per_stream_schedule().schedules}
CAL_DOWN, CAL_AIR = ("calgary", "Calgary Downtown"), ("calgary", "Calgary Int Airport")
TOR_DOWN, TOR_AIR = ("toronto", "Toronto Downtown"), ("toronto", "Toronto Int Airport")
VAN_DOWN, VAN_THUR = ("vancouver", "Vancouver Downtown"), ("vancouver", "Vancouver Thurlow")
START = dt.datetime(2030, 3, 4, 8)          # synthetic local start (no DST transition nearby)
VERSION = "synth_v1"


def local_hour(city: str, h: int) -> tuple[dt.datetime, dt.timedelta, str]:
    local = START + dt.timedelta(hours=h)
    aware = local.replace(tzinfo=ZoneInfo(ZONES[city]))
    return local, aware.utcoffset(), format_utc_instant(aware)


def synthetic_schedule(hours: int, exceptions=None) -> PerStreamSchedule:  # type: ignore[no-untyped-def]
    schedules = tuple(StreamSchedule(
        schedule_version=VERSION, stream=s, city=s[0], timezone=ZONES[s[0]], local_start=START,
        local_end=START + dt.timedelta(hours=hours - 1), end_inclusive=True, cadence="PT1H",
        phase="LOCAL_TOP_OF_HOUR", capture_field="jobs.finished_at", record_id="pricing-authorities-synthetic",
        references=("SYNTH-REFERENCE",)) for s in STREAMS)
    return PerStreamSchedule(
        status=ScheduleAuthorityStatus.AVAILABLE, expected_streams=STREAMS, record_id="pricing-authorities-synthetic",
        schedule_version=VERSION, capture_field="jobs.finished_at", detail_copy_field="cars.job_finished_at",
        detail_observation_field="cars.scraped_at", sharing_mode=SharingMode.PER_STREAM,
        timezones=CityTimezoneMap(tuple(sorted(ZONES.items()))), schedules=schedules,
        exceptions=exceptions or ScheduleExceptions.none(), references=("SYNTH-REFERENCE",))


def car_row(stream, job, finished, name, price, currency="CA$", basis="day"):  # type: ignore[no-untyped-def]
    return {"job_id": job, "city": stream[0], "location": stream[1], "job_finished_at": finished,
            "scraped_at": "SYNTH-OBSERVATION", "car_name": name, "car_type": "SYNTH Compact",
            "transmission": "SYNTH Automatic", "seats": "5", "bags": "2", "pickup_date": "2030-04-01",
            "return_date": "2030-04-03", "price_num": price, "price_per_day": f"{currency}{price:,.2f}/{basis}"}


def synthetic_world(hours=3, products=None, absent_jobs=(), excluded=None, excused=None, drop_streams=()):  # type: ignore[no-untyped-def]
    """Fabricated frames through the real schedule, exclusion, canonical-offer and readiness objects.

    ``products`` maps ``(stream, hour)`` to extra ``(name, price[, (field, value)...])``
    offers; every present stream-period also carries a constant filler product.
    """
    jobs, cars = [], []
    for city in dict.fromkeys(s[0] for s in STREAMS):
        for h in range(hours):
            if (city, h) in absent_jobs:
                continue
            job = f"SYNTH-JOB-{city}-{h}"
            finished = (START + dt.timedelta(hours=h, minutes=7, seconds=13)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            jobs.append({"job_id": job, "city": city, "finished_at": finished})
            for stream in STREAMS:
                if stream[0] != city or (stream, h) in drop_streams:
                    continue
                cars.append(car_row(stream, job, finished, "SYNTH Filler", 10.0))
                for name, price, *unit in (products or {}).get((stream, h), ()):
                    cars.append(car_row(stream, job, finished, name, price, **dict(unit)))
    exclusions, stream_exceptions = (), ()
    if excluded is not None:
        city, h = excluded
        local, offset, text = local_hour(city, h)
        exclusions = (ParentCaptureExclusion(
            city=city, streams=tuple(s for s in STREAMS if s[0] == city), period_start_utc=text, local_start=local,
            utc_offset=offset, reason="SYNTH incomplete capture", authority_kind=AuthorityKind.COLLECTION_OWNER,
            reference="SYNTH-REFERENCE", schedule_version=VERSION),)
    if excused is not None:
        city, h = excused
        local, offset, text = local_hour(city, h)
        stream_exceptions = tuple(StreamScheduleException(
            stream=s, period_start_utc=text, local_start=local, utc_offset=offset,
            failure=ScheduleFailureKind.PARENT_JOB_ABSENT, reason="SYNTH outage",
            authority_kind=AuthorityKind.COLLECTION_OWNER, reference="SYNTH-REFERENCE", schedule_version=VERSION)
            for s in STREAMS if s[0] == city)
    exceptions = (ScheduleExceptions(model=ExceptionsModel.LISTED_EXCEPTIONS, exceptions=stream_exceptions,
                                     parent_capture_exclusions=exclusions)
                  if exclusions or stream_exceptions else None)
    jobs_df, cars_df = pd.DataFrame(jobs), pd.DataFrame(cars)
    scheduled = assess_per_stream_scheduled_coverage(jobs_df, cars_df, schedule=synthetic_schedule(hours, exceptions),
                                                     contract=CONTRACT, relationship=REL)

    def status(excluded_mask: np.ndarray, assigned: pd.Series) -> tuple[str, ...]:
        return tuple(E.GOVERNED_EXCLUSION.value if x else E.ELIGIBLE.value if a else E.CAPTURE_PERIOD_UNASSIGNED.value
                     for x, a in zip(excluded_mask, assigned.notna()))

    population = PricingPopulation(
        binding=frame_binding(jobs_df, cars_df),
        parent_status=status(scheduled.capture_exclusions.parent_mask(jobs_df),
                             scheduled.capture_periods.parent_periods(jobs_df)),
        detail_status=status(scheduled.capture_exclusions.detail_mask(cars_df),
                             scheduled.capture_periods.detail_periods(cars_df)))
    offers = assess_canonical_offers(jobs_df, cars_df, population=population, scheduled=scheduled, policy=POLICY)
    readiness = readiness_for(cars_df, scheduled, offers)
    return dict(jobs=jobs_df, cars=cars_df, readiness=readiness, population=population, scheduled=scheduled,
                canonical_offers=offers, location_authority=AUTHORITY)


def readiness_for(cars, scheduled, offers, blockers=()):  # type: ignore[no-untyped-def]
    return PricingReadinessReport(
        blocking_reasons=tuple(blockers), location_policy=assess_location_policy(
            VANCOUVER_LOCATION_POLICY, None, apply_location_policy(cars, VANCOUVER_LOCATION_POLICY)),
        scheduled_coverage=scheduled, location_authority=AUTHORITY, canonical_offers=offers)


def run(w: dict, **overrides) -> PriceChangeCandidateResult:  # type: ignore[no-untyped-def]
    args = {**w, **overrides}
    return assess_price_change_candidates(args.pop("jobs"), args.pop("cars"), **args)


def pipeline_result(w: dict) -> PricingPipelineResult:
    return PricingPipelineResult(
        record=None, contract=CONTRACT, relationship=REL, jobs=w["jobs"], cars=w["cars"], job_linkage=None,
        unique_keys=None, scheduled=w["scheduled"], temporal=None, temporal_authority=None, reporting_days=None,
        population=w["population"], vehicle_stability=None, canonical_offers=w["canonical_offers"],
        location_authority=w["location_authority"], pricing=w["readiness"])


def at(city: str, h: int) -> str:
    return local_hour(city, h)[2]


def test_the_pipeline_world_completes_in_authority_order_with_every_location() -> None:
    w = synthetic_world(products={(TOR_DOWN, 0): [("SYNTH Car A", 50.0)], (TOR_DOWN, 1): [("SYNTH Car A", 55.0)],
                                  (TOR_DOWN, 2): [("SYNTH Car A", 52.5)]})
    before = (w["jobs"].copy(deep=True), w["cars"].copy(deep=True))
    result = price_change_candidates_from_pipeline(pipeline_result(w))
    assert result.completed and result.report.blockers == ()
    check_invariants(result)
    approved = approved_canonical_locations(AUTHORITY, POLICY)
    assert result.report.approved_locations == approved and VAN_THUR not in approved and len(approved) == 6
    assert outcomes(result) == [(at("toronto", 0), at("toronto", 1), "increase"),
                                (at("toronto", 1), at("toronto", 2), "decrease")]
    assert product(result)["change_cents"].tolist() == [500, -250]
    assert product(result)["change_percent"].tolist() == [10.0, float(Fraction(-25000, 5500))]
    assert all((s.scheduled_periods, s.eligible_periods, s.intervals) == (3, 3, 2) for s in result.report.locations)
    pd.testing.assert_frame_equal(w["jobs"], before[0]), pd.testing.assert_frame_equal(w["cars"], before[1])


def test_canonical_offers_not_matched_pairs_and_one_sided_products_participate(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import ql2_sixt_canada_analysis.matched_location_pricing as mlp

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("matched-location pairs must not be used")

    monkeypatch.setattr(mlp, "assess_matched_location_pricing", forbidden)
    monkeypatch.setattr(mlp, "_match", forbidden)
    w = synthetic_world(products={(TOR_AIR, 0): [("SYNTH Airport Only", 70.0)],
                                  (TOR_AIR, 1): [("SYNTH Airport Only", 77.0)]})
    result = run(w)
    assert result.completed
    rows = product(result, "SYNTH Airport Only")
    assert set(rows["canonical_location"]) == {TOR_AIR[1]}
    assert rows["outcome"].tolist() == ["increase", "disappeared"]


def test_vancouver_alias_offers_are_counted_once_and_keep_their_provenance() -> None:
    both = {(s, h): [("SYNTH Car V", 60.0 + h)] for s in (VAN_DOWN, VAN_THUR) for h in range(3)}
    result = run(synthetic_world(products=both))
    check_invariants(result)
    rows = product(result, "SYNTH Car V")
    assert len(rows) == 2 and set(rows["canonical_location"]) == {VAN_DOWN[1]}
    assert rows["outcome"].tolist() == ["increase", "increase"]                # never ambiguous, never doubled
    assert set(rows["previous_source_labels"]) == {"Vancouver Downtown|Vancouver Thurlow"}
    assert result.report.location(VAN_DOWN).source_streams == (VAN_DOWN, VAN_THUR)
    thurlow_only = run(synthetic_world(products={(VAN_THUR, 0): [("SYNTH Car T", 61.0)],
                                                 (VAN_THUR, 1): [("SYNTH Car T", 61.0)]}))
    row = product(thurlow_only, "SYNTH Car T").iloc[0]
    assert (row["canonical_location"], row["outcome"], row["current_source_labels"]) == (
        VAN_DOWN[1], "unchanged", "Vancouver Thurlow")


def test_governed_excluded_middle_capture_is_a_hard_break_without_mass_turnover() -> None:
    """An eligible capture, the governed INCOMPLETE_PARENT_CAPTURE, a later eligible capture."""
    products = {(s, h): [("SYNTH Car A", 50.0 + h), ("SYNTH Car B", 80.0)] for s in STREAMS for h in (0, 2)}
    products[(CAL_AIR, 1)] = [("SYNTH Car A", 999.0)]                           # the incomplete capture's rows
    w = synthetic_world(products=products, excluded=("calgary", 1), drop_streams={(CAL_DOWN, 1)})
    scheduled = w["scheduled"]
    assert scheduled.is_valid and scheduled.excluded_periods == 2 and scheduled.excluded_parent_captures == 1
    assert w["canonical_offers"].out_of_scope_rows > 0                          # excluded rows never enter
    result = run(w)
    assert result.completed
    check_invariants(result)
    excluded_period = at("calgary", 1)
    for stream in (CAL_DOWN, CAL_AIR):
        s = result.report.location(stream)
        assert (s.scheduled_periods, s.eligible_periods, s.excluded_periods, s.intervals) == (3, 2, 1, 0)
        assert s.breaks == ((IB.GOVERNED_EXCLUSION.value, 2),)
        assert s.counts == OutcomeCounts()                                       # no appearances or disappearances
    assert result.candidates[result.candidates["canonical_city"] == "calgary"].empty
    for t in (t for t in result.timelines if t.canonical_location[0] == "calgary"):
        assert t.periods(CS.GOVERNED_EXCLUSION) == (excluded_period,)
        assert all(excluded_period not in (i.previous_period, i.current_period) for i in t.intervals)
    assert result.report.location(TOR_DOWN).counts.candidates > 0               # other cities are unaffected


def test_an_excused_missing_capture_breaks_and_a_returning_product_is_never_a_change() -> None:
    products = {(TOR_DOWN, 0): [("SYNTH Car A", 50.0)], (TOR_DOWN, 2): [("SYNTH Car A", 70.0)],
                (TOR_DOWN, 3): [("SYNTH Car A", 70.0)]}
    w = synthetic_world(hours=4, products=products, absent_jobs={("toronto", 1)}, excused=("toronto", 1))
    assert w["scheduled"].is_valid
    result = run(w)
    check_invariants(result)
    s = result.report.location(TOR_DOWN)
    assert (s.missing_periods, s.intervals) == (1, 1) and s.breaks == ((IB.MISSING_CAPTURE.value, 2),)
    assert outcomes(result) == [(at("toronto", 2), at("toronto", 3), "unchanged")]


def test_unit_changes_end_to_end_are_never_price_movements() -> None:
    products = {(TOR_DOWN, 0): [("SYNTH Car A", 50.0, ("currency", "CA$")), ("SYNTH Car B", 9.0, ("basis", "day"))],
                (TOR_DOWN, 1): [("SYNTH Car A", 50.0, ("currency", "US$")), ("SYNTH Car B", 9.0, ("basis", "week"))]}
    result = run(synthetic_world(hours=2, products=products))
    assert sorted(product(result)["outcome"]) == ["appeared", "disappeared"]
    assert sorted(product(result, "SYNTH Car B")["outcome"]) == ["appeared", "disappeared"]


def test_pricing_readiness_blockers_produce_a_blocked_report_without_events() -> None:
    w = synthetic_world()
    not_ready = readiness_for(w["cars"], w["scheduled"], w["canonical_offers"],
                              (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE,))
    for blocked in (run(w, readiness=not_ready), price_change_candidates_from_pipeline(
            dataclasses.replace(pipeline_result(w), pricing=not_ready))):
        assert blocked.report.blockers == (PB.PRICING_NOT_READY,) and blocked.candidates is None
        assert blocked.report.readiness_blockers == (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE.value,)
        assert blocked.report.overall is None and blocked.report.locations == () and blocked.timelines is None
    missing = price_change_candidates_from_pipeline(dataclasses.replace(pipeline_result(w), population=None))
    assert missing.report.blockers == (PB.PRICING_NOT_READY,)
    assert missing.report.readiness_blockers == ("required_assessment_unavailable",)


def test_stale_or_mismatched_evidence_fails_closed() -> None:
    w, other = synthetic_world(), synthetic_world(hours=2)
    assert PB.FRAME_BINDING_MISMATCH in run(w, population=other["population"]).report.blockers
    assert PB.FRAME_BINDING_MISMATCH in run(w, cars=w["cars"].assign(price_num=1.0)).report.blockers
    assert PB.FRAME_BINDING_MISMATCH in run(w, jobs=other["jobs"]).report.blockers
    mixed = run(w, canonical_offers=other["canonical_offers"])
    assert {PB.READINESS_EVIDENCE_MISMATCH, PB.FRAME_BINDING_MISMATCH} <= set(mixed.report.blockers)
    assert PB.READINESS_EVIDENCE_MISMATCH in run(w, scheduled=other["scheduled"]).report.blockers
    for result in (mixed, run(w, population=other["population"])):
        assert not result.completed and result.candidates is None
    with pytest.raises(TypeError):
        run(w, scheduled=None)


@pytest.mark.parametrize("field", ["capture_periods", "capture_exclusions"])
def test_missing_capture_evidence_fails_closed(field) -> None:  # type: ignore[no-untyped-def]
    w = synthetic_world()
    scheduled = dataclasses.replace(w["scheduled"], **{field: None})
    result = run(w, scheduled=scheduled, readiness=readiness_for(w["cars"], scheduled, w["canonical_offers"]))
    assert result.report.blockers == (PB.SCHEDULE_EVIDENCE_INVALID,) and result.candidates is None
    with pytest.raises(CaptureEvidenceError):
        capture_timelines(scheduled, POLICY)


def test_invalid_schedule_and_inconsistent_capture_evidence_fail_closed() -> None:
    invalid = synthetic_world(absent_jobs={("toronto", 1)})                     # unexcused missing capture
    assert not invalid["scheduled"].is_valid
    assert PB.SCHEDULE_EVIDENCE_INVALID in run(invalid).report.blockers
    scheduled = synthetic_world(excluded=("calgary", 1), drop_streams={(CAL_DOWN, 1)})["scheduled"]
    stream = scheduled.streams[0]
    tampered = dataclasses.replace(scheduled, streams=(dataclasses.replace(stream, covered=stream.covered - 1),
                                                       *scheduled.streams[1:]))
    with pytest.raises(CaptureEvidenceError):
        capture_timelines(tampered, POLICY)
    unresolved = dataclasses.replace(scheduled, capture_exclusions=dataclasses.replace(
        scheduled.capture_exclusions, entries=()))
    with pytest.raises(CaptureEvidenceError):                                   # exclusion without resolved capture
        capture_timelines(unresolved, POLICY)
    with pytest.raises(TypeError):
        capture_timelines(scheduled, None)


def test_population_disagreement_fails_closed() -> None:
    w = synthetic_world()
    statuses = list(w["population"].parent_status)
    statuses[0] = E.REPORTING_DAY_FAILED.value                                  # an eligible capture's parent
    population = dataclasses.replace(w["population"], parent_status=tuple(statuses))
    assert run(w, population=population).report.blockers == (PB.CAPTURE_EVIDENCE_INCONSISTENT,)


def test_unknown_canonical_locations_and_unavailable_authority_fail_closed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    w = synthetic_world()
    monkeypatch.setattr(pce, "approved_canonical_locations", lambda authority, policy: (TOR_DOWN, TOR_AIR))
    assert run(w).report.blockers == (PB.UNKNOWN_CANONICAL_LOCATION,)
    monkeypatch.undo()

    def unavailable(authority, policy):  # type: ignore[no-untyped-def]
        raise UnknownCanonicalLocationError("synthetic")

    monkeypatch.setattr(pce, "approved_canonical_locations", unavailable)
    assert run(w).report.blockers == (PB.LOCATION_AUTHORITY_UNAVAILABLE,)


def test_run_price_change_events_fails_closed_on_unready_synthetic_raw_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from conftest import contract_columns, write_synthetic_csv

    from ql2_sixt_canada_analysis.schemas import DatasetKey

    for key in DatasetKey:
        write_synthetic_csv(tmp_path / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    before = sorted(os.listdir(tmp_path))
    result = run_price_change_events(tmp_path)
    assert result.report.blockers == (PB.PRICING_NOT_READY,) and result.report.readiness_blockers
    assert result.candidates is None and result.timelines is None
    assert sorted(os.listdir(tmp_path)) == before                               # nothing written
    assert str(tmp_path) not in repr(result) and "synthetic_" not in repr(result)
    with pytest.raises(TypeError):
        price_change_candidates_from_pipeline(object())


# ============================================================================ confidentiality and packaging


def test_no_raw_job_identifier_is_required_or_exposed() -> None:
    assert not set(FORBIDDEN_TIMESTAMP_SOURCES) & set(CANDIDATE_COLUMNS)
    rows = frame([offer(P(0)), offer(P(1))])
    assert "job_id" not in rows.columns and len(classify_price_change_candidates(rows, [timeline([0, 1])])[0]) == 1
    result = run(synthetic_world(products={(TOR_DOWN, 1): [("SYNTH Car A", 50.0)]}))
    shown = result.candidates.to_string() + repr(result)
    assert "SYNTH-JOB" not in shown and "job_id" not in result.candidates.columns


def test_event_level_frames_stay_out_of_repr_and_reports_and_are_never_written(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    result = run(synthetic_world(products={(TOR_DOWN, 0): [("SYNTH Car A", 50.0)],
                                           (TOR_DOWN, 1): [("SYNTH Car A", 61.0)]}))
    assert result.completed and len(result.candidates)
    shown = repr(result) + repr(result.report) + str(result.report) + repr(result.timelines)
    assert "SYNTH Car" not in shown and "SYNTH Filler" not in shown and "SYNTH-JOB" not in shown
    assert not re.search(r"2030\d{4}T\d{6}Z|2030-0[34]-\d\d", shown)            # no periods or rental dates
    assert "DataFrame" not in repr(result) and "price_cents" not in repr(result) and "6100" not in shown
    assert "61.0" not in shown and "22.0" not in shown and str(tmp_path) not in shown
    assert os.listdir(tmp_path) == []
    fields = {f.name: f for f in dataclasses.fields(PriceChangeCandidateResult)}
    assert not fields["candidates"].repr and not fields["candidates"].compare and not fields["timelines"].repr


def test_importing_the_module_performs_no_io_pipeline_or_plotting() -> None:
    """No project file is read, nothing is written, no pipeline runs and no plotting library is loaded."""
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
            "import ql2_sixt_canada_analysis.price_change_events as m\n"
            "print('matplotlib' in sys.modules, 'scipy' in sys.modules, 'seaborn' in sys.modules,\n"
            "      'ql2_sixt_canada_analysis.pricing_pipeline' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False", "False", "False"]


def test_public_package_exports_are_complete() -> None:
    assert len(pce.__all__) == len(set(pce.__all__))
    public = {n for n in vars(pce) if not n.startswith("_") and getattr(getattr(pce, n), "__module__", None)
              == pce.__name__}
    assert public <= set(pce.__all__)
    for name in pce.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(pce, name)
    assert {o.value for o in T} == {"unchanged", "increase", "decrease", "appeared", "disappeared", "ambiguous"}
    assert {"run_price_change_events", "assess_price_change_candidates", "classify_price_change_candidates",
            "PriceChangeBlocker", "PriceChangeStatus", "PriceChangeCandidateResult"} <= set(pce.__all__)
