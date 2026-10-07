"""Matched location pricing: same-job, same-car airport/downtown premiums.

Every frame is fabricated (``SYNTH-*`` jobs and products, synthetic prices and
dates). The only real values are approved configuration read from the
committed authority record (stream keys, comparison pairs, the Vancouver
alias policy and the per-stream schedule definition).
"""

from __future__ import annotations

import dataclasses
import math
import re
from fractions import Fraction

import matplotlib
import numpy as np
import pandas as pd
import pytest

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis import matched_location_pricing as mlp
from ql2_sixt_canada_analysis.canonical_offers import (
    APPROVED_PRODUCT_COLUMNS,
    CanonicalOfferReport,
    assess_canonical_offers,
    current_canonical_offer_policy,
)
from ql2_sixt_canada_analysis.collection_schedule import (
    CapturePeriodIndex,
    PerStreamScheduledCoverageReport,
    StreamPeriodCoverage,
    current_per_stream_schedule,
)
from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
from ql2_sixt_canada_analysis.location_authority import current_location_authority
from ql2_sixt_canada_analysis.matched_location_pricing import (
    MATCH_IDENTITY_COLUMNS,
    MIN_PAIRS_PER_TESTED_VEHICLE_TYPE,
    OVERALL,
    PAIR_COLUMNS,
    PAIR_KEY_COLUMNS,
    CityMatchSummary,
    DistributionSummary,
    MatchCounts,
    MatchedLocationPricingBlocker as MB,
    MatchedLocationPricingError,
    MatchedLocationPricingReport,
    MatchedLocationPricingResult,
    MatchedLocationPricingStatus as MS,
    PremiumMetric,
    SignCounts,
    ComparisonUnit,
    VehicleTypeTest,
    VehicleTypeTestStatus as VS,
    assess_matched_location_pricing,
    city_summary_frame,
    compare_vehicle_types,
    holm_adjust,
    match_count_frame,
    plot_matched_location_premiums,
    render_matched_location_pricing_markdown,
    vehicle_type_summary_frame,
    vehicle_type_test_frame,
    write_matched_location_pricing_deliverables,
)
from ql2_sixt_canada_analysis.pricing_population import DetailEligibility as E, PricingPopulation, frame_binding
from ql2_sixt_canada_analysis.readiness import (
    PricingBlocker as PB,
    PricingReadinessReport,
    apply_location_policy,
    assess_location_policy,
)
from ql2_sixt_canada_analysis.schemas import VANCOUVER_LOCATION_POLICY
from ql2_sixt_canada_analysis.stability import VehicleStabilityReport, VehicleStabilityStatus

matplotlib.use("Agg")

CONTRACT = current_expected_stream_contract()
AUTHORITY = current_location_authority()
POLICY = current_canonical_offer_policy()
CAL_AIR, CAL_DOWN = ("calgary", "Calgary Int Airport"), ("calgary", "Calgary Downtown")
TOR_AIR, TOR_DOWN = ("toronto", "Toronto Int Airport"), ("toronto", "Toronto Downtown")
VAN_AIR, VAN_DOWN, VAN_THUR = (("vancouver", "Vancouver Int Airport"), ("vancouver", "Vancouver Downtown"),
                               ("vancouver", "Vancouver Thurlow"))
CITY_ORDER = ("calgary", "toronto", "vancouver")


def car(stream, job, price=50.0, name="SYNTH Vehicle A", car_type="SYNTH Compact", text=None, **changes):  # type: ignore[no-untyped-def]
    row = {"job_id": job, "city": stream[0], "location": stream[1], "car_name": name, "car_type": car_type,
           "transmission": "SYNTH Automatic", "seats": "5", "bags": "2", "pickup_date": "2030-01-10",
           "return_date": "2030-01-12", "price_num": price,
           "price_per_day": text if text is not None else f"CA${price:,.2f}/day"}
    row.update(changes)
    return row


def period(k: int) -> str:
    return f"20300101T{k:02d}0000Z"


def world(rows, statuses=None, periods=None, stability_observations=None):  # type: ignore[no-untyped-def]
    """Every input of the assessment for synthetic rows (each job its own scheduled period unless stated)."""
    cars = pd.DataFrame(rows)
    cars.insert(1, "row_index", [str(i) for i in range(len(cars))])
    job_ids = list(dict.fromkeys(cars["job_id"]))
    jobs = pd.DataFrame({"job_id": job_ids,
                         "city": [cars.loc[cars["job_id"] == j, "city"].iloc[0] for j in job_ids]})
    periods = periods if periods is not None else {j: period(i) for i, j in enumerate(job_ids)}
    scheduled = PerStreamScheduledCoverageReport(
        schedule=current_per_stream_schedule(), coverage=CONTRACT.coverage,
        streams=tuple(StreamPeriodCoverage(stream=k, expected=1, covered=1, missing=()) for k in CONTRACT.expected_keys),
        jobs_assessed=len(jobs), jobs_assigned=len(jobs),
        capture_periods=CapturePeriodIndex(("job_id",), ("job_id",), {(j,): p for j, p in periods.items()}))
    statuses = tuple(statuses or [E.ELIGIBLE.value] * len(cars))
    population = PricingPopulation(binding=frame_binding(jobs, cars), parent_status=(E.ELIGIBLE.value,) * len(jobs),
                                   detail_status=statuses)
    offers = assess_canonical_offers(jobs, cars, population=population, scheduled=scheduled, policy=POLICY)
    readiness = PricingReadinessReport(
        blocking_reasons=(), location_policy=assess_location_policy(
            VANCOUVER_LOCATION_POLICY, None, apply_location_policy(cars, VANCOUVER_LOCATION_POLICY)),
        scheduled_coverage=scheduled, location_authority=AUTHORITY, canonical_offers=offers)
    n = population.eligible_detail_rows if stability_observations is None else stability_observations
    return dict(jobs=jobs, cars=cars, readiness=readiness, population=population, scheduled=scheduled,
                canonical_offers=offers, location_authority=AUTHORITY, vehicle_stability=stability(n))


def stability(observations: int, status=VehicleStabilityStatus.PASSED) -> VehicleStabilityReport:  # type: ignore[no-untyped-def]
    passed = status is VehicleStabilityStatus.PASSED
    return VehicleStabilityReport(
        status=status, observations_assessed=observations, distinct_entities=1, complete_identity_entities=1,
        incomplete_identity_entities=0, incomplete_identity_observations=0, temporally_unassessable_entities=0,
        sufficient_history_entities=1 if passed else 0, insufficient_history_entities=0 if passed else 1,
        fully_stable_entities=1 if passed else 0, value_unstable_only_entities=0, presence_unstable_only_entities=0,
        value_and_presence_unstable_entities=0, entities_with_value_conflicts=0, entities_with_presence_instability=0,
        same_capture_conflict_entities=0, attributes=())


def assess(w: dict, **overrides) -> MatchedLocationPricingResult:  # type: ignore[no-untyped-def]
    args = {**w, **overrides}
    return assess_matched_location_pricing(args.pop("jobs"), args.pop("cars"), **args)


def run(rows, **kwargs) -> MatchedLocationPricingResult:  # type: ignore[no-untyped-def]
    return assess(world(rows, **kwargs))


def counts(result: MatchedLocationPricingResult, city: str = OVERALL) -> MatchCounts:
    return result.report.city(city).counts


def outcome_tuple(c: MatchCounts) -> tuple[int, ...]:
    return (c.candidate_groups, c.matched, c.airport_only, c.downtown_only, c.ambiguous, c.currency_mismatch,
            c.basis_mismatch, c.percent_valid, c.zero_denominator)


def check_invariants(result: MatchedLocationPricingResult) -> None:
    """The strict accounting invariants every completed result must satisfy."""
    report, pairs = result.report, result.pairs
    for s in (*report.cities, report.overall):
        c = s.counts
        assert c.candidate_groups == (c.matched + c.airport_only + c.downtown_only + c.ambiguous
                                      + c.currency_mismatch + c.basis_mismatch)
        assert s.signs.total == c.matched == s.dollars.n
        assert c.percent_valid + c.zero_denominator == c.matched == s.percent.n + c.zero_denominator
    assert len(pairs) == report.overall.counts.matched
    assert sum(s.counts.matched for s in report.cities) == report.overall.counts.matched
    for name in ("candidate_groups", "airport_only", "downtown_only", "ambiguous", "currency_mismatch",
                 "basis_mismatch", "percent_valid", "zero_denominator"):
        assert sum(getattr(s.counts, name) for s in report.cities) == getattr(report.overall.counts, name)
    assert not pairs.duplicated(list(PAIR_KEY_COLUMNS)).any()
    approved = {(a[0], a[1], d[1]) for a, d in report.approved_pairs}
    assert set(map(tuple, pairs[["canonical_city", "airport_location", "downtown_location"]].to_numpy())) <= approved
    assert (pairs["premium_cents"] == pairs["airport_price_cents"] - pairs["downtown_price_cents"]).all()


# ============================================================================ basic pairs and formulas


def test_a_same_job_same_car_pair_is_matched_with_exact_formulas() -> None:
    result = run([car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_DOWN, "SYNTH-JOB-T1", 64.0)])
    assert result.completed and result.report.blockers == ()
    check_invariants(result)
    assert outcome_tuple(counts(result, "toronto")) == (1, 1, 0, 0, 0, 0, 0, 1, 0)
    row = result.pairs.iloc[0]
    assert (row["airport_price_cents"], row["downtown_price_cents"], row["premium_cents"]) == (8000, 6400, 1600)
    assert row["premium_dollars"] == 16.0 and row["premium_percent"] == 25.0
    assert row["premium_sign"] == "positive" and bool(row["percent_valid"])
    assert (row["currency"], row["price_basis"]) == ("CA$", "day")
    assert tuple(result.pairs.columns) == PAIR_COLUMNS
    assert counts(result, "calgary").candidate_groups == 0 and result.report.city("calgary").dollars.n == 0


@pytest.mark.parametrize(("airport", "downtown", "sign", "percent"), [
    (60.0, 40.0, "positive", 50.0), (40.0, 40.0, "zero", 0.0), (30.0, 40.0, "negative", -25.0),
    (100.01, 33.33, "positive", float(Fraction(100 * (10001 - 3333), 3333)))])
def test_premium_direction_is_airport_minus_downtown(airport, downtown, sign, percent) -> None:  # type: ignore[no-untyped-def]
    result = run([car(CAL_AIR, "SYNTH-JOB-C1", airport), car(CAL_DOWN, "SYNTH-JOB-C1", downtown)])
    row = result.pairs.iloc[0]
    assert row["premium_cents"] == round(airport * 100) - round(downtown * 100)
    assert row["premium_sign"] == sign and row["premium_percent"] == percent
    signs = result.report.city("calgary").signs
    assert signs.total == 1 and getattr(signs, sign) == 1


def test_positive_zero_and_negative_premiums_are_counted_and_shared() -> None:
    rows = []
    for i, (a, d) in enumerate([(60, 40), (50, 50), (45, 50), (70, 35)]):
        job = f"SYNTH-JOB-V{i}"
        rows += [car(VAN_AIR, job, a), car(VAN_DOWN, job, d)]
    result = run(rows)
    signs = result.report.city("vancouver").signs
    assert (signs.positive, signs.zero, signs.negative) == (2, 1, 1)
    assert signs.share("positive") == 0.5 and signs.share("zero") == 0.25
    check_invariants(result)


# ============================================================================ same job, same car, same city


def test_the_same_car_in_different_jobs_never_matches() -> None:
    result = run([car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_DOWN, "SYNTH-JOB-T2", 64.0)])
    assert outcome_tuple(counts(result, "toronto")) == (2, 0, 1, 1, 0, 0, 0, 0, 0)
    assert result.pairs.empty


@pytest.mark.parametrize("change", [{"car_name": "SYNTH Vehicle B"}, {"car_type": "SYNTH Van"},
                                    {"transmission": "SYNTH Manual"}, {"seats": "7"}, {"bags": "3"},
                                    {"pickup_date": "2030-01-11"}, {"return_date": "2030-01-13"},
                                    {"car_name": "synth vehicle a"}, {"seats": "5.0"}])
def test_different_cars_or_rental_periods_in_the_same_job_never_match(change) -> None:  # type: ignore[no-untyped-def]
    result = run([car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_DOWN, "SYNTH-JOB-T1", 64.0, **change)])
    assert outcome_tuple(counts(result, "toronto")) == (2, 0, 1, 1, 0, 0, 0, 0, 0)


def test_price_is_never_part_of_the_car_identity() -> None:
    result = run([car(TOR_AIR, "SYNTH-JOB-T1", 99.0), car(TOR_DOWN, "SYNTH-JOB-T1", 11.0)])
    assert counts(result, "toronto").matched == 1


def test_cross_city_offers_never_match() -> None:
    # Same product, same scheduled period, different cities (each city has its own parent job).
    rows = [car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0)]
    result = run(rows, periods={"SYNTH-JOB-T1": period(1), "SYNTH-JOB-C1": period(1)})
    assert counts(result).matched == 0
    assert counts(result, "toronto").airport_only == 1 and counts(result, "calgary").downtown_only == 1


def test_airport_to_airport_and_downtown_to_downtown_never_pair() -> None:
    result = run([car(VAN_DOWN, "SYNTH-JOB-V1", 50.0), car(VAN_THUR, "SYNTH-JOB-V1", 40.0)])
    assert counts(result).matched == 0                       # Downtown vs Thurlow is never a comparison
    assert counts(result, "vancouver").downtown_only == 1     # price-distinct canonical offers on one side only


def test_unapproved_or_reversed_pairs_are_rejected() -> None:
    from stream_contract_fixtures import synthetic_location_authority

    roles = {k: ("AIRPORT" if "Airport" in k[1] else "DOWNTOWN") for k in CONTRACT.expected_keys}
    reversed_pairs = ((CAL_DOWN, CAL_AIR), (TOR_AIR, TOR_DOWN), (VAN_AIR, VAN_DOWN))
    authority = synthetic_location_authority(CONTRACT, roles, reversed_pairs)
    assert not authority.pairs_valid and authority.effective_pairs == ()
    w = world([car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0)])
    w["readiness"] = dataclasses.replace(w["readiness"], location_authority=authority)
    result = assess(w, location_authority=authority)
    assert not result.completed and MB.LOCATION_PAIRS_INVALID in result.report.blockers and result.pairs is None
    cross = synthetic_location_authority(CONTRACT, roles, ((CAL_AIR, TOR_DOWN),))
    assert not cross.pairs_valid


def test_pairs_follow_the_approved_authority_not_labels() -> None:
    result = run([car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0)])
    assert result.report.approved_pairs == tuple((p.airport, p.downtown) for p in AUTHORITY.effective_pairs)
    assert tuple(s.city for s in result.report.cities) == CITY_ORDER
    assert result.report.city("vancouver").downtown_location == "Vancouver Downtown"


def test_vancouver_thurlow_is_combined_and_never_double_counted() -> None:
    # The same downtown offer observed in both aliased source streams is one canonical offer.
    rows = [car(VAN_AIR, "SYNTH-JOB-V1", 90.0), car(VAN_DOWN, "SYNTH-JOB-V1", 60.0), car(VAN_THUR, "SYNTH-JOB-V1", 60.0)]
    result = run(rows)
    assert outcome_tuple(counts(result, "vancouver")) == (1, 1, 0, 0, 0, 0, 0, 1, 0)
    assert len(result.pairs) == 1 and result.pairs.iloc[0]["downtown_location"] == "Vancouver Downtown"
    # A Thurlow-only downtown offer still pairs through the canonical location, once.
    only_thurlow = run([car(VAN_AIR, "SYNTH-JOB-V1", 90.0), car(VAN_THUR, "SYNTH-JOB-V1", 60.0)])
    assert counts(only_thurlow, "vancouver").matched == 1
    assert set(only_thurlow.pairs["downtown_location"]) == {"Vancouver Downtown"}


def test_price_distinct_aliased_downtown_offers_are_ambiguous_not_cartesian() -> None:
    rows = [car(VAN_AIR, "SYNTH-JOB-V1", 90.0), car(VAN_DOWN, "SYNTH-JOB-V1", 60.0), car(VAN_THUR, "SYNTH-JOB-V1", 61.0)]
    result = run(rows)
    assert outcome_tuple(counts(result, "vancouver")) == (1, 0, 0, 0, 1, 0, 0, 0, 0)
    assert result.pairs.empty


# ============================================================================ population controls


def test_the_governed_exclusion_never_enters_the_analysis() -> None:
    rows = [car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0),
            car(CAL_AIR, "SYNTH-JOB-C2", 85.0), car(CAL_DOWN, "SYNTH-JOB-C2", 60.0)]
    statuses = [E.ELIGIBLE.value] * 2 + [E.GOVERNED_EXCLUSION.value] * 2
    w = world(rows, statuses=statuses)
    assert w["canonical_offers"].out_of_scope_rows == 2      # kept in the source frames, auditable
    result = assess(w)
    assert outcome_tuple(counts(result, "calgary")) == (1, 1, 0, 0, 0, 0, 0, 1, 0)
    assert result.pairs.iloc[0]["airport_price_cents"] == 8000
    assert len(w["cars"]) == 4                               # nothing removed from the source


def test_pricing_ineligible_rows_never_enter_and_block_readiness() -> None:
    rows = [car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0)]
    w = world(rows, statuses=[E.ELIGIBLE.value, E.RENTAL_DATES_FAILED.value])
    assert not w["canonical_offers"].ready
    result = assess(w)
    assert not result.completed and MB.CANONICAL_OFFERS_NOT_READY in result.report.blockers


def test_readiness_false_fails_closed_with_the_existing_blockers() -> None:
    w = world([car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0)])
    blocked = dataclasses.replace(w["readiness"], blocking_reasons=(PB.DATA_INCOMPLETE, PB.KEY_CONTRACTS_INVALID))
    result = assess(w, readiness=blocked)
    assert result.report.status is MS.BLOCKED and result.pairs is None
    assert result.report.blockers == (MB.PRICING_NOT_READY,)
    assert result.report.readiness_blockers == ("data_incomplete", "key_contracts_invalid")
    assert result.report.cities == () and result.report.overall is None
    text = render_matched_location_pricing_markdown(result)
    assert "BLOCKED" in text and "data_incomplete" in text and "Dollar premium" not in text
    with pytest.raises(MatchedLocationPricingError):
        city_summary_frame(result)
    with pytest.raises(MatchedLocationPricingError):
        plot_matched_location_premiums(result)


def test_stale_or_mismatched_frames_and_reports_are_rejected() -> None:
    w = world([car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0)])
    tampered = w["cars"].copy()
    tampered.loc[0, "price_num"] = 1.0
    stale = assess(w, cars=tampered)
    assert MB.FRAME_BINDING_MISMATCH in stale.report.blockers and not stale.completed
    other = world([car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 63.0)])
    swapped = assess(w, canonical_offers=other["canonical_offers"])
    assert {MB.READINESS_EVIDENCE_MISMATCH, MB.FRAME_BINDING_MISMATCH} <= set(swapped.report.blockers)
    other_schedule = assess(w, scheduled=other["scheduled"])
    assert MB.READINESS_EVIDENCE_MISMATCH in other_schedule.report.blockers


def test_vehicle_stability_must_pass_on_the_pricing_population() -> None:
    w = world([car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0)])
    failed = assess(w, vehicle_stability=stability(2, VehicleStabilityStatus.PARTIALLY_ASSESSABLE))
    assert failed.report.blockers == (MB.VEHICLE_STABILITY_NOT_PASSED,)
    other_population = assess(w, vehicle_stability=stability(5))
    assert other_population.report.blockers == (MB.VEHICLE_STABILITY_POPULATION_MISMATCH,)


def test_an_invalid_schedule_assessment_blocks() -> None:
    w = world([car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C1", 64.0)])
    incomplete = dataclasses.replace(w["scheduled"], streams=w["scheduled"].streams[1:])
    w["readiness"] = dataclasses.replace(w["readiness"], scheduled_coverage=incomplete)
    result = assess(w, scheduled=incomplete)
    assert MB.SCHEDULE_EVIDENCE_INVALID in result.report.blockers


def test_a_city_period_with_two_parent_captures_is_not_proven_same_job() -> None:
    rows = [car(CAL_AIR, "SYNTH-JOB-C1", 80.0), car(CAL_DOWN, "SYNTH-JOB-C2", 64.0)]
    result = run(rows, periods={"SYNTH-JOB-C1": period(3), "SYNTH-JOB-C2": period(3)})
    assert result.report.blockers == (MB.SAME_JOB_NOT_PROVEN,) and result.pairs is None


# ============================================================================ cardinality and units


def test_missing_airport_and_missing_downtown_are_unmatched() -> None:
    rows = [car(TOR_AIR, "SYNTH-JOB-T1", 80.0, name="SYNTH Only Airport"),
            car(TOR_DOWN, "SYNTH-JOB-T1", 60.0, name="SYNTH Only Downtown"),
            car(TOR_AIR, "SYNTH-JOB-T1", 70.0), car(TOR_DOWN, "SYNTH-JOB-T1", 60.0)]
    result = run(rows)
    assert outcome_tuple(counts(result, "toronto")) == (3, 1, 1, 1, 0, 0, 0, 1, 0)
    assert counts(result).match_rate == pytest.approx(1 / 3)


def test_duplicate_price_distinct_offers_are_ambiguous_without_expansion() -> None:
    rows = [car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_AIR, "SYNTH-JOB-T1", 81.0),
            car(TOR_DOWN, "SYNTH-JOB-T1", 60.0), car(TOR_DOWN, "SYNTH-JOB-T1", 61.0)]
    result = run(rows)
    assert outcome_tuple(counts(result, "toronto")) == (1, 0, 0, 0, 1, 0, 0, 0, 0)
    assert len(result.pairs) == 0                          # never 2 x 2 = 4 manufactured pairs
    exact_duplicates = run([car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_AIR, "SYNTH-JOB-T1", 80.0),
                            car(TOR_DOWN, "SYNTH-JOB-T1", 60.0)])
    assert counts(exact_duplicates, "toronto").matched == 1     # one canonical offer, deduplicated upstream


def test_currency_and_price_basis_mismatches_are_incompatible() -> None:
    currency = run([car(TOR_AIR, "SYNTH-JOB-T1", 80.0, text="US$80.00/day"), car(TOR_DOWN, "SYNTH-JOB-T1", 60.0)])
    assert outcome_tuple(counts(currency, "toronto")) == (1, 0, 0, 0, 0, 1, 0, 0, 0)
    basis = run([car(TOR_AIR, "SYNTH-JOB-T1", 80.0, text="CA$80.00/week"), car(TOR_DOWN, "SYNTH-JOB-T1", 60.0)])
    assert outcome_tuple(counts(basis, "toronto")) == (1, 0, 0, 0, 0, 0, 1, 0, 0)
    assert currency.pairs.empty and basis.pairs.empty


def test_pairs_in_different_units_are_never_aggregated_together() -> None:
    rows = [car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_DOWN, "SYNTH-JOB-T1", 60.0),
            car(CAL_AIR, "SYNTH-JOB-C1", 80.0, text="US$80.00/day"),
            car(CAL_DOWN, "SYNTH-JOB-C1", 60.0, text="US$60.00/day")]
    result = run(rows)
    assert result.report.blockers == (MB.MIXED_PRICE_UNITS,)


def test_a_zero_downtown_price_keeps_the_dollar_premium_but_no_percentage() -> None:
    rows = [car(TOR_AIR, "SYNTH-JOB-T1", 15.0), car(TOR_DOWN, "SYNTH-JOB-T1", 0.0),
            car(TOR_AIR, "SYNTH-JOB-T2", 30.0), car(TOR_DOWN, "SYNTH-JOB-T2", 20.0)]
    result = run(rows)
    c = counts(result, "toronto")
    assert (c.matched, c.percent_valid, c.zero_denominator) == (2, 1, 1)
    zero = result.pairs.loc[result.pairs["downtown_price_cents"] == 0].iloc[0]
    assert zero["premium_dollars"] == 15.0 and math.isnan(zero["premium_percent"]) and not zero["percent_valid"]
    assert np.isfinite(result.pairs.loc[result.pairs["percent_valid"], "premium_percent"]).all()
    summary = result.report.city("toronto")
    assert summary.dollars.n == 2 and summary.percent.n == 1 and summary.percent.median == 50.0
    check_invariants(result)


@pytest.mark.parametrize("change", [{"price_num": float("nan")}, {"price_num": -5.0}, {"price_num": 12.345},
                                    {"price_per_day": "CA$ 12.00/day"}, {"price_per_day": "CA$99.00/day"},
                                    {"car_type": None}, {"seats": float("nan")}, {"car_name": " SYNTH Vehicle A"}])
def test_invalid_prices_or_product_identity_fail_closed(change) -> None:  # type: ignore[no-untyped-def]
    result = run([car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_DOWN, "SYNTH-JOB-T1", 60.0, **change)])
    assert not result.completed and result.report.blockers == (MB.CANONICAL_OFFERS_NOT_READY,)


def test_a_malformed_offer_table_is_rejected() -> None:
    w = world([car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_DOWN, "SYNTH-JOB-T1", 60.0)])
    report = w["canonical_offers"]
    offers = report.offers.copy()
    offers.loc[0, "car_type"] = None
    forged = dataclasses.replace(report, offers=offers)
    w["readiness"] = dataclasses.replace(w["readiness"], canonical_offers=forged)
    result = assess(w, canonical_offers=forged)
    assert result.report.blockers == (MB.OFFER_CONTRACT_INVALID,)


def test_empty_input_is_explicit_not_zero() -> None:
    w = world([car(TOR_AIR, "SYNTH-JOB-T1", 80.0)], statuses=[E.GOVERNED_EXCLUSION.value])
    result = assess(w)
    assert result.completed and result.pairs.empty
    for s in (*result.report.cities, result.report.overall):
        assert s.counts.candidate_groups == 0 and s.counts.match_rate is None
        assert s.dollars == DistributionSummary(0) and s.dollars.median is None and s.signs.share("positive") is None
    assert all(t.status is VS.NOT_TESTABLE for t in result.report.vehicle_type_tests)
    assert "n/a" in render_matched_location_pricing_markdown(result)


def test_inputs_are_not_modified() -> None:
    w = world([car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_DOWN, "SYNTH-JOB-T1", 60.0)])
    before = (w["jobs"].copy(), w["cars"].copy(), w["canonical_offers"].offers.copy())
    assess(w)
    pd.testing.assert_frame_equal(before[0], w["jobs"])
    pd.testing.assert_frame_equal(before[1], w["cars"])
    pd.testing.assert_frame_equal(before[2], w["canonical_offers"].offers)


def test_wrong_argument_types_raise() -> None:
    w = world([car(TOR_AIR, "SYNTH-JOB-T1", 80.0), car(TOR_DOWN, "SYNTH-JOB-T1", 60.0)])
    for name in ("readiness", "population", "scheduled", "canonical_offers", "location_authority",
                 "vehicle_stability"):
        with pytest.raises(TypeError):
            assess(w, **{name: object()})


# ============================================================================ summaries and statistics


def big_world_rows(seed: int = 7) -> list[dict]:
    """Several jobs per city, several products and vehicle types, mixed signs and some attrition."""
    rng = np.random.default_rng(seed)
    rows = []
    pairs = ((CAL_AIR, CAL_DOWN, 1.5), (TOR_AIR, TOR_DOWN, 1.2), (VAN_AIR, VAN_DOWN, 1.0))
    for c, (air, down, markup) in enumerate(pairs):
        for j in range(30):
            job = f"SYNTH-JOB-{c}-{j:02d}"
            for p, car_type in enumerate(("SYNTH Compact", "SYNTH Compact", "SYNTH SUV", "SYNTH Van")):
                base = 40.0 + 10 * p
                name = f"SYNTH Vehicle {p}"
                if (j + p) % 9 != 0:
                    rows.append(car(down, job, base, name=name, car_type=car_type))
                if (j + 2 * p) % 11 != 0:
                    factor = markup * (1.3 if car_type == "SYNTH SUV" else 1.0) + float(rng.integers(-3, 4)) / 20
                    rows.append(car(air, job, round(base * factor, 2), name=name, car_type=car_type))
    return rows


@pytest.fixture(scope="module")
def big() -> MatchedLocationPricingResult:
    return run(big_world_rows())


def test_city_summaries_reconcile_exactly_to_pair_outcomes(big) -> None:  # type: ignore[no-untyped-def]
    check_invariants(big)
    report = big.report
    for s in report.cities:
        subset = big.pairs.loc[big.pairs["canonical_city"] == s.city]
        assert s.counts.matched == len(subset)
        assert s.signs.positive == int((subset["premium_cents"] > 0).sum())
        assert s.signs.negative == int((subset["premium_cents"] < 0).sum())
    frame = match_count_frame(big)
    assert list(frame["city"]) == [*CITY_ORDER, OVERALL]
    assert frame.loc[frame["city"] == OVERALL, "matched_pairs"].item() == len(big.pairs)
    assert (frame["match_rate"] == frame["matched_pairs"] / frame["candidate_groups"]).all()


def test_quantiles_signs_and_match_rate_denominators_are_correct(big) -> None:  # type: ignore[no-untyped-def]
    for s in big.report.cities:
        subset = big.pairs.loc[big.pairs["canonical_city"] == s.city]
        dollars = subset["premium_dollars"].to_numpy()
        assert s.dollars.median == pytest.approx(np.median(dollars))
        assert (s.dollars.q25, s.dollars.q75) == pytest.approx(tuple(np.percentile(dollars, [25, 75])))
        assert (s.dollars.minimum, s.dollars.maximum) == (dollars.min(), dollars.max())
        assert s.dollars.std == pytest.approx(np.std(dollars, ddof=1))
        assert s.percent.mean == pytest.approx(subset["premium_percent"].mean())
        assert s.counts.match_rate == s.counts.matched / s.counts.candidate_groups
        assert s.counts.airport_only > 0 and s.counts.downtown_only > 0
    table = city_summary_frame(big)
    assert (table["positive_share"] + table["zero_share"] + table["negative_share"]).to_numpy() == pytest.approx(1.0)


def test_distribution_summary_of_known_values() -> None:
    d = DistributionSummary.of([1.0, 2.0, 3.0, 4.0])
    assert (d.n, d.mean, d.median, d.q25, d.q75, d.minimum, d.maximum) == (4, 2.5, 2.5, 1.75, 3.25, 1.0, 4.0)
    assert d.std == pytest.approx(math.sqrt(5 / 3))
    single = DistributionSummary.of([7.0])
    assert single.std is None and single.median == 7.0
    with pytest.raises(MatchedLocationPricingError):
        DistributionSummary(0, mean=0.0)
    with pytest.raises(MatchedLocationPricingError):
        DistributionSummary(2, 1.0, 1.0, 0.0, 2.0, 0.5, 0.0, 3.0)   # q25 above the median


def test_result_objects_reject_inconsistent_counts(big) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(MatchedLocationPricingError):
        MatchCounts(3, 1, 1, 0, 0, 0, 0, 1, 0)                     # outcomes do not sum to candidates
    with pytest.raises(MatchedLocationPricingError):
        MatchCounts(1, 1, 0, 0, 0, 0, 0, 0, 0)                     # matched != percent-valid + zero
    with pytest.raises(MatchedLocationPricingError):
        SignCounts(-1, 0, 0)
    city = big.report.cities[0]
    with pytest.raises(MatchedLocationPricingError):
        dataclasses.replace(city, signs=SignCounts(0, 0, 0))
    with pytest.raises(MatchedLocationPricingError):
        dataclasses.replace(big.report, cities=big.report.cities[1:])
    wrong_overall = dataclasses.replace(big.report.overall, counts=big.report.cities[0].counts,
                                        signs=big.report.cities[0].signs, dollars=big.report.cities[0].dollars,
                                        percent=big.report.cities[0].percent)
    with pytest.raises(MatchedLocationPricingError):
        dataclasses.replace(big.report, overall=wrong_overall)
    with pytest.raises(MatchedLocationPricingError):
        MatchedLocationPricingResult(big.report, big.pairs.iloc[1:])
    swapped = big.pairs.copy()
    swapped.loc[swapped.index[0], "canonical_city"] = "toronto" if swapped.iloc[0]["canonical_city"] != "toronto" \
        else "calgary"
    with pytest.raises(MatchedLocationPricingError):
        MatchedLocationPricingResult(big.report, swapped)
    with pytest.raises(MatchedLocationPricingError):
        MatchedLocationPricingReport(status=MS.BLOCKED)
    with pytest.raises(MatchedLocationPricingError):
        MatchedLocationPricingResult(MatchedLocationPricingReport(status=MS.BLOCKED, blockers=(MB.PRICING_NOT_READY,)),
                                     big.pairs)


def test_no_pair_spans_jobs_periods_products_or_units(big) -> None:  # type: ignore[no-untyped-def]
    # Every pair key is the single identity both offers shared; both offers exist in the canonical offers.
    offers = run(big_world_rows()).pairs
    assert set(offers["currency"]) == {"CA$"} and set(offers["price_basis"]) == {"day"}
    assert big.pairs.groupby(list(MATCH_IDENTITY_COLUMNS)).size().max() == 1


# ============================================================================ vehicle type


def test_vehicle_type_summaries_cover_every_pair(big) -> None:  # type: ignore[no-untyped-def]
    frame = vehicle_type_summary_frame(big)
    for city in CITY_ORDER:
        assert frame.loc[frame["city"] == city, "matched_pairs"].sum() == big.report.city(city).counts.matched
    assert set(frame["car_type"]) == {"SYNTH Compact", "SYNTH SUV", "SYNTH Van"}


def test_vehicle_type_tests_are_deterministic_with_holm_and_effect_sizes(big) -> None:  # type: ignore[no-untyped-def]
    from scipy.stats import kruskal

    again = run(big_world_rows())
    assert again.report.vehicle_type_tests == big.report.vehicle_type_tests
    primary = [t for t in big.report.vehicle_type_tests
               if t.metric is PremiumMetric.PERCENT and t.unit is ComparisonUnit.PAIR]
    assert [t.city for t in primary] == list(CITY_ORDER)
    assert all(t.status is VS.TESTED and t.minimum_per_type == MIN_PAIRS_PER_TESTED_VEHICLE_TYPE for t in primary)
    t = primary[0]
    subset = big.pairs.loc[big.pairs["canonical_city"] == t.city]
    groups = [g["premium_percent"].to_numpy() for _, g in subset.groupby("car_type")
              if len(g) >= MIN_PAIRS_PER_TESTED_VEHICLE_TYPE]
    h, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    assert (t.statistic, t.p_value, t.df) == (pytest.approx(h), pytest.approx(p), len(groups) - 1)
    assert t.epsilon_squared == pytest.approx(h / (n - 1))
    assert [x.p_holm for x in primary] == pytest.approx(list(holm_adjust([x.p_value for x in primary])))
    assert not vehicle_type_test_frame(big).empty


def test_holm_adjustment() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx((0.03, 0.06, 0.06))
    assert holm_adjust([0.5, 0.6]) == pytest.approx((1.0, 1.0))
    assert holm_adjust([]) == ()
    assert holm_adjust([0.02, 0.02]) == pytest.approx((0.04, 0.04))


def test_vehicle_type_tests_are_not_testable_with_insufficient_groups() -> None:
    rows = []
    for j in range(MIN_PAIRS_PER_TESTED_VEHICLE_TYPE + 2):
        job = f"SYNTH-JOB-T{j:02d}"
        rows += [car(TOR_AIR, job, 70.0 + j % 3), car(TOR_DOWN, job, 50.0)]
        if j < 3:
            rows += [car(TOR_AIR, job, 90.0, name="SYNTH Rare", car_type="SYNTH Rare"),
                     car(TOR_DOWN, job, 50.0, name="SYNTH Rare", car_type="SYNTH Rare")]
    result = run(rows)
    test = next(t for t in result.report.vehicle_type_tests
                if t.city == "toronto" and t.metric is PremiumMetric.PERCENT and t.unit is ComparisonUnit.PAIR)
    assert test.status is VS.NOT_TESTABLE and test.reason == "fewer_than_two_types_with_minimum_observations"
    assert (test.types_observed, test.types_tested) == (2, 1)
    assert test.statistic is None and test.p_value is None and test.p_holm is None
    flat = compare_vehicle_types(result.pairs.assign(premium_percent=10.0), "toronto", minimum=3)
    assert flat.status is VS.NOT_TESTABLE and flat.reason == "no_variation"
    with pytest.raises(MatchedLocationPricingError):
        VehicleTypeTest(city="toronto", metric=PremiumMetric.PERCENT, unit=ComparisonUnit.PAIR, status=VS.NOT_TESTABLE,
                        minimum_per_type=20, types_observed=1, types_tested=1, observations=3, reason="x",
                        p_value=0.5)


def test_vehicle_narrative_uses_the_percentage_test_population() -> None:
    rows = []
    for j in range(MIN_PAIRS_PER_TESTED_VEHICLE_TYPE):
        job = f"SYNTH-JOB-T{j:02d}"
        rows += [car(TOR_AIR, job, 10.0, name="SYNTH A", car_type="SYNTH A"),
                 car(TOR_DOWN, job, 10.0 if j == 0 else 0.0, name="SYNTH A", car_type="SYNTH A"),
                 car(TOR_AIR, job, 20.0, name="SYNTH B", car_type="SYNTH B"),
                 car(TOR_DOWN, job, 10.0, name="SYNTH B", car_type="SYNTH B")]
    result = run(rows)
    narrative = mlp._vehicle_sentence(result.report)
    assert "tested-type median premiums range +100.0% to +100.0%" in narrative
    assert "tested-type median premiums range 0.0% to +100.0%" not in narrative


# ============================================================================ visualization


def test_plot_structure(big) -> None:  # type: ignore[no-untyped-def]
    from matplotlib.figure import Figure

    pairs_before = big.pairs.copy()
    fig, (dollars, percent) = plot_matched_location_premiums(big)
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    for ax in (dollars, percent):
        assert [t.get_text() for t in ax.get_xticklabels()] == [c.title() for c in CITY_ORDER]
        zero = [line for line in ax.get_lines() if line.get_label() == "zero premium"]
        assert len(zero) == 1 and set(zero[0].get_ydata()) == {0.0}
        texts = [t.get_text() for t in ax.texts]
        assert "▲ airport premium" in texts and "▼ airport discount" in texts
        assert ax.get_ylabel()
    for s, text in zip(big.report.cities, [t.get_text() for t in dollars.texts if t.get_text().startswith("n =")]):
        assert text.startswith(f"n = {s.counts.matched:,}") and "median" in text
    assert "Dollar" in dollars.get_title(loc="left") and "Percentage" in percent.get_title(loc="left")
    assert "CA$" in dollars.get_yticklabels()[0].get_text() or any(
        "CA$" in t.get_text() for t in dollars.get_yticklabels())
    pd.testing.assert_frame_equal(pairs_before, big.pairs)
    # Deterministic: identical scatter offsets on a second call.
    fig2, (dollars2, _) = plot_matched_location_premiums(big)
    np.testing.assert_array_equal(dollars.collections[0].get_offsets(), dollars2.collections[0].get_offsets())
    # Every pair is drawn (no silent outlier removal).
    assert sum(len(c.get_offsets()) for c in dollars.collections) == len(big.pairs)


def test_plot_never_calls_show(big, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: pytest.fail("plt.show called"))
    figures = plt.get_fignums()
    plot_matched_location_premiums(big)
    assert plt.get_fignums() == figures                 # no pyplot-managed figure was created


# ============================================================================ reports, privacy, exports


def test_reports_and_plots_hold_no_confidential_values(big, tmp_path) -> None:  # type: ignore[no-untyped-def]
    text = render_matched_location_pricing_markdown(big, figure_path="figures/x.png")
    shown = text + repr(big.report) + repr(big)
    for frame in (match_count_frame(big), city_summary_frame(big), vehicle_type_summary_frame(big),
                  vehicle_type_test_frame(big)):
        shown += frame.to_string()
    fig, _ = plot_matched_location_premiums(big)
    shown += " ".join(t.get_text() for ax in fig.axes for t in (*ax.texts, *ax.get_xticklabels()))
    assert "SYNTH-JOB" not in shown and "job_id" not in shown and "SYNTH Vehicle" not in shown
    assert not re.search(r"20300101T\d{6}Z|2030-01-1\d", shown)          # no capture periods or rental dates
    assert "DataFrame" not in repr(big) and "premium_cents" not in repr(big)
    report_path, figure_path = write_matched_location_pricing_deliverables(
        big, report_path=tmp_path / "out" / "r.md", figure_path=tmp_path / "out" / "figures" / "f.png")
    assert figure_path.read_bytes()[:4] == b"\x89PNG"
    assert "`figures/f.png`" in report_path.read_text(encoding="utf-8")
    assert "Associational" in text or "associational" in text


def test_public_package_exports() -> None:
    for name in mlp.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(mlp, name)
    for name in ("run_pricing_pipeline", "PricingPipelineResult"):
        assert name in package.__all__
    assert isinstance(OVERALL, str) and PAIR_KEY_COLUMNS[-2:] == ("currency", "price_basis")
    assert set(APPROVED_PRODUCT_COLUMNS) <= set(PAIR_KEY_COLUMNS)


def test_importing_the_module_performs_no_io_or_plotting() -> None:
    import subprocess
    import sys

    code = ("import sys, ql2_sixt_canada_analysis.matched_location_pricing as m; "
            "print('matplotlib.pyplot' in sys.modules, 'scipy.stats' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_run_blocks_on_synthetic_raw_files_that_are_not_pricing_ready(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from conftest import contract_columns, write_synthetic_csv

    from ql2_sixt_canada_analysis.schemas import DatasetKey

    for key in DatasetKey:
        write_synthetic_csv(tmp_path / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    result = mlp.run_matched_location_pricing(tmp_path)
    assert not result.completed and result.report.blockers == (MB.PRICING_NOT_READY,)
    assert result.report.readiness_blockers and result.pairs is None
    assert mlp.main(["--raw-dir", str(tmp_path), "--reports-dir", str(tmp_path / "reports")]) == 2
    assert not (tmp_path / "reports").exists()


def test_city_summary_rejects_wrong_counts() -> None:
    d = DistributionSummary.of([1.0])
    with pytest.raises(MatchedLocationPricingError):
        CityMatchSummary("toronto", "a", "b", MatchCounts(1, 1, 0, 0, 0, 0, 0, 1, 0), SignCounts(1, 0, 0), d,
                         DistributionSummary(0))
    assert isinstance(CanonicalOfferReport, type)
