"""Authority-backed canonical offer combination (CANONICAL_OFFER_COMBINATION, pricing-authorities-v8).

Frames are fabricated (``SYNTH-*`` products, synthetic prices and dates); the
only real values are approved configuration (stream keys) read from the record.
"""

from __future__ import annotations

import dataclasses
import random
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stream_contract_fixtures import passing_canonical_report, synthetic_offer_policy

from ql2_sixt_canada_analysis.authority_decisions import (
    CANONICAL_OFFER_IDENTITY_COMPONENTS,
    DecisionId,
    DecisionRecordError,
    DecisionStatus,
    load_decision_record,
    parse_decision_record,
    required_decisions,
)
from ql2_sixt_canada_analysis.canonical_offers import (
    APPROVED_PRODUCT_COLUMNS,
    CanonicalOfferBlocker as CB,
    CanonicalOfferPolicy,
    CanonicalOfferReport,
    CanonicalOfferStatus as CS,
    UnassessableOfferReason as UR,
    assess_canonical_offers,
    canonical_offer_policy_from_record,
    current_canonical_offer_policy,
    parse_price_text,
)
from ql2_sixt_canada_analysis.collection_schedule import (
    CapturePeriodIndex,
    PerStreamSchedule,
    PerStreamScheduledCoverageReport,
    ScheduleAuthorityStatus as SS,
)
from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
from ql2_sixt_canada_analysis.pricing_population import (
    DetailEligibility as E,
    PricingPopulation,
    PricingPopulationError,
    frame_binding,
)
from ql2_sixt_canada_analysis.readiness import PricingBlocker as PB
from ql2_sixt_canada_analysis.schemas import ANALYSIS_LOCATION_STREAM_COMPARISON

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V7, V8 = RECORD_DIR / "v7.toml", RECORD_DIR / "v8.toml"
GOVERNANCE = "docs/decisions/governance/vancouver-canonical-offer-combination-governance-v1-2026-10-06.md"
D = DecisionId
VAN_DOWN, VAN_THUR, VAN_AIR = (("vancouver", "Vancouver Downtown"), ("vancouver", "Vancouver Thurlow"),
                               ("vancouver", "Vancouver Int Airport"))
TOR_DOWN = ("toronto", "Toronto Downtown")
STREAMS = (VAN_DOWN, VAN_THUR, VAN_AIR, TOR_DOWN)
POLICY = canonical_offer_policy_from_record(load_decision_record(V8), current_expected_stream_contract())
PERIOD = "20260828T170000Z"


def v8() -> dict:
    return tomllib.loads(V8.read_text(encoding="utf-8"))


def entry(data: dict, decision: DecisionId) -> dict:
    return next(e for e in data["decisions"] if e["id"] == decision.value)


def fails(data: dict) -> str:
    statuses = [e["status"] for e in data["decisions"]]
    data["summary"] = {s.value.lower(): statuses.count(s.value) for s in DecisionStatus}
    data["external_inputs"] = [e["id"] for e in data["decisions"] if e["blocking_external_input"]]
    with pytest.raises(DecisionRecordError) as info:
        parse_decision_record(data)
    return str(info.value)


def offer(stream=VAN_DOWN, job="SYNTH-JOB-1", name="SYNTH Vehicle A", price=42.5, text=None, **changes):  # type: ignore[no-untyped-def]
    row = {"job_id": job, "city": stream[0], "location": stream[1], "car_name": name, "car_type": "SYNTH Class",
           "transmission": "SYNTH Automatic", "seats": 5.0, "bags": 2.0, "pickup_date": "2026-09-01",
           "return_date": "2026-09-04", "price_num": price,
           "price_per_day": text if text is not None else f"CA${price:,.2f}/day"}
    row.update(changes)
    return row


def world(rows, statuses=None, periods=None):  # type: ignore[no-untyped-def]
    """(jobs, cars, population, scheduled) for synthetic offers (every row eligible unless stated)."""
    cars = pd.DataFrame(rows)
    cars.insert(1, "row_index", [str(i) for i in range(len(cars))])
    job_ids = list(dict.fromkeys(cars["job_id"]))
    jobs = pd.DataFrame({"job_id": job_ids, "city": [cars.loc[cars["job_id"] == j, "city"].iloc[0] for j in job_ids]})
    periods = periods if periods is not None else {j: PERIOD for j in job_ids}
    scheduled = PerStreamScheduledCoverageReport(
        schedule=PerStreamSchedule(status=SS.NOT_APPROVED), coverage=None, jobs_assessed=len(jobs),
        capture_periods=CapturePeriodIndex(("job_id",), ("job_id",), {(j,): p for j, p in periods.items()}))
    statuses = statuses or [E.ELIGIBLE.value] * len(cars)
    population = PricingPopulation(binding=frame_binding(jobs, cars), parent_status=(E.ELIGIBLE.value,) * len(jobs),
                                   detail_status=tuple(statuses))
    return jobs, cars, population, scheduled


def combine(rows, statuses=None, policy=POLICY, periods=None):  # type: ignore[no-untyped-def]
    jobs, cars, population, scheduled = world(rows, statuses, periods)
    return assess_canonical_offers(jobs, cars, population=population, scheduled=scheduled, policy=policy,
                                   expected_streams=STREAMS), jobs, cars


# ======================================================================= authority


def test_v8_approves_the_supplied_combination_policy() -> None:
    record = load_decision_record(V8)
    item = record.decision(D.CANONICAL_OFFER_COMBINATION)
    assert item.joint and {a.kind.value for a in item.authority} == {"COLLECTION_OWNER", "BUSINESS_OWNER"}
    assert {a.reference for a in item.authority} == {GOVERNANCE}
    resolution = record.approved_resolution(D.CANONICAL_OFFER_COMBINATION)
    assert (resolution["alias_policy"], resolution["validation"], resolution["union"], resolution["exact_duplicates"],
            resolution["price_variation"], resolution["unassessable"]) == (
        "SEPARATE_REQUIRED_SOURCE_STREAMS", "ALL_FOUNDATIONAL_CONTROLS_BEFORE_COMBINATION",
        "ORDER_INDEPENDENT_NO_STREAM_PRIORITY", "ONE_CANONICAL_ROW_WITH_PROVENANCE", "RETAIN_SEPARATE_AND_FLAG",
        "FAIL_CLOSED")
    assert tuple(resolution["identity"]) == CANONICAL_OFFER_IDENTITY_COMPONENTS
    assert POLICY is not None and POLICY == current_canonical_offer_policy()
    assert (POLICY.status, POLICY.source_streams, POLICY.canonical_location) == (CS.APPROVED, (VAN_DOWN, VAN_THUR),
                                                                                 VAN_DOWN)
    assert POLICY.references == (GOVERNANCE,) and POLICY.record_id == "pricing-authorities-v8"
    contract = current_expected_stream_contract()
    assert {VAN_DOWN, VAN_THUR} <= set(contract.expected_keys)          # both stay separately required


def test_the_decision_exists_only_from_schema_4() -> None:
    v7 = load_decision_record(V7)
    assert not v7.has_decision(D.CANONICAL_OFFER_COMBINATION) and v7.approved_resolution(
        D.CANONICAL_OFFER_COMBINATION) is None
    with pytest.raises(KeyError):
        v7.decision(D.CANONICAL_OFFER_COMBINATION)
    assert D.CANONICAL_OFFER_COMBINATION not in required_decisions(3)
    assert D.CANONICAL_OFFER_COMBINATION in required_decisions(4)
    assert canonical_offer_policy_from_record(v7).status is CS.NOT_APPROVED
    assert canonical_offer_policy_from_record(None).status is CS.RECORD_UNAVAILABLE
    data = v8()
    data["schema_version"] = 3
    assert "not defined in this schema_version" in fails(data)
    data = v8()
    data["decisions"] = [e for e in data["decisions"] if e["id"] != D.CANONICAL_OFFER_COMBINATION.value]
    assert "missing required decisions: CANONICAL_OFFER_COMBINATION" in fails(data)


@pytest.mark.parametrize("change, needle", [
    ({"identity": ["canonical_location", "scheduled_capture_period", "pickup_date", "return_date",
                   "approved_product_identity", "normalized_price", "currency"]}, "identity components"),
    ({"identity": [*CANONICAL_OFFER_IDENTITY_COMPONENTS, "row_index"]}, "identity components"),
    ({"provenance": ["source_location_labels"]}, "provenance fields"),
    ({"source_streams": [list(VAN_DOWN), list(VAN_AIR)]}, "approved alias"),
    ({"canonical_location": list(VAN_THUR)}, "approved alias"),
    ({"union": "DOWNTOWN_FIRST"}, "unsupported value"),
    ({"exact_duplicates": "KEEP_FIRST_ROW"}, "unsupported value"),
    ({"price_variation": "AVERAGE"}, "unsupported value"),
    ({"price_variation": "MINIMUM"}, "unsupported value"),
    ({"unassessable": "DROP"}, "unsupported value"),
    ({"alias_policy": "MERGE_STREAMS"}, "unsupported value"),
    ({"validation": "COMBINE_THEN_VALIDATE"}, "unsupported value"),
])
def test_combination_shape_fails_closed(change, needle) -> None:
    data = v8()
    entry(data, D.CANONICAL_OFFER_COMBINATION)["resolution"].update(change)
    assert needle in fails(data)


def test_a_policy_that_drifts_from_the_alias_or_contract_is_invalid() -> None:
    from stream_contract_fixtures import synthetic_contract

    from ql2_sixt_canada_analysis.schemas import EXPECTED_LOCATION_COVERAGE as COV

    narrower = synthetic_contract(dataclasses.replace(
        COV, expected_locations=tuple(k for k in COV.expected_locations if k != VAN_THUR)))
    assert canonical_offer_policy_from_record(load_decision_record(V8), narrower).status is CS.INVALID
    with pytest.raises(ValueError):
        CanonicalOfferPolicy(CS.APPROVED, "pricing-authorities-v8", (VAN_DOWN, VAN_THUR), VAN_AIR,
                             CANONICAL_OFFER_IDENTITY_COMPONENTS, (GOVERNANCE,))
    with pytest.raises(ValueError):
        CanonicalOfferPolicy(CS.APPROVED, "pricing-authorities-v8", (VAN_DOWN, VAN_THUR), VAN_DOWN, (), (GOVERNANCE,))


def test_the_approved_product_identity_is_the_established_comparison_definition() -> None:
    dates = ("pickup_date", "return_date")
    assert APPROVED_PRODUCT_COLUMNS == tuple(c for c in ANALYSIS_LOCATION_STREAM_COMPARISON.product_columns
                                             if c not in dates)


# ===================================================================== combination


def test_exact_duplicates_across_the_aliased_streams_become_one_canonical_offer() -> None:
    report, jobs, cars = combine([offer(VAN_DOWN), offer(VAN_THUR), offer(VAN_AIR), offer(TOR_DOWN, job="SYNTH-JOB-2")])
    assert (report.source_rows, report.combined_observations, report.canonical_offers, report.unique_offers,
            report.deduplicated_offers, report.duplicate_groups, report.collapsed_observations,
            report.cross_stream_offers) == (4, 4, 3, 2, 1, 1, 1, 1)
    assert report.ready and report.blocking_reasons == () and report.unassessable_rows == 0
    assert dict(report.source_counts) == {VAN_DOWN: 1, VAN_THUR: 1, VAN_AIR: 1, TOR_DOWN: 1}   # both still counted
    assert dict(report.canonical_counts) == {VAN_DOWN: 1, VAN_AIR: 1, TOR_DOWN: 1}            # no Thurlow key
    offers = report.offers_for(jobs, cars)
    merged = offers[offers["canonical_location"] == "Vancouver Downtown"].iloc[0]
    assert (merged["source_location_labels"], merged["observation_count"], merged["provenance"],
            merged["price_variation"]) == ("Vancouver Downtown|Vancouver Thurlow", 2, "deduplicated", False)
    assert merged["price_cents"] == 4250 and merged["currency"] == "CA$" and merged["price_basis"] == "day"
    airport = offers[offers["canonical_location"] == "Vancouver Int Airport"].iloc[0]
    assert (airport["observation_count"], airport["provenance"]) == (1, "unique")


def test_the_union_is_order_independent_with_no_stream_priority_or_row_index() -> None:
    rows = [offer(VAN_DOWN), offer(VAN_THUR), offer(VAN_THUR, name="SYNTH Vehicle B", price=50.0),
            offer(VAN_DOWN, name="SYNTH Vehicle C", price=60.25), offer(VAN_THUR, name="SYNTH Vehicle C", price=60.25),
            offer(VAN_DOWN, price=43.0)]
    first, *_ = combine(rows)
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    second, *_ = combine(shuffled)
    reversed_, *_ = combine(rows[::-1])
    for other in (second, reversed_):
        pd.testing.assert_frame_equal(first.offers, other.offers)
        assert dataclasses.replace(first, offers=None) == dataclasses.replace(other, offers=None, binding=first.binding)
    jobs, cars, population, scheduled = world(rows)
    renumbered = cars.assign(row_index=[str(100 - i) for i in range(len(cars))])
    population = PricingPopulation(binding=frame_binding(jobs, renumbered), parent_status=population.parent_status,
                                   detail_status=population.detail_status)
    again = assess_canonical_offers(jobs, renumbered, population=population, scheduled=scheduled, policy=POLICY,
                                    expected_streams=STREAMS)
    pd.testing.assert_frame_equal(first.offers, again.offers)


def test_different_prices_are_kept_separate_and_flagged_never_averaged() -> None:
    report, jobs, cars = combine([offer(VAN_DOWN, price=42.5), offer(VAN_THUR, price=45.0)])
    assert (report.canonical_offers, report.variation_groups, report.variation_offers, report.deduplicated_offers) == (
        2, 1, 2, 0)
    offers = report.offers_for(jobs, cars)
    assert sorted(offers["price_cents"]) == [4250, 4500] and offers["price_variation"].all()
    assert set(offers["observation_count"]) == {1}                     # nothing discarded, nothing averaged
    assert report.ready                                                # variation is flagged, not a blocker


@pytest.mark.parametrize("change", [
    {"pickup_date": "2026-09-02"}, {"return_date": "2026-09-05"}, {"car_name": "SYNTH Vehicle Z"},
    {"car_type": "SYNTH Other Class"}, {"transmission": "SYNTH Manual"}, {"seats": 4.0}, {"bags": 3.0},
])
def test_every_identity_component_distinguishes_offers(change) -> None:
    report, *_ = combine([offer(VAN_DOWN), offer(VAN_THUR, **change)])
    assert report.canonical_offers == 2 and report.duplicate_groups == 0 and report.variation_groups == 0


def test_capture_periods_and_currencies_are_never_assumed_equal() -> None:
    report, *_ = combine([offer(VAN_DOWN, job="SYNTH-JOB-1"), offer(VAN_THUR, job="SYNTH-JOB-2")],
                         periods={"SYNTH-JOB-1": PERIOD, "SYNTH-JOB-2": "20260828T180000Z"})
    assert report.canonical_offers == 2 and report.duplicate_groups == 0
    report, jobs, cars = combine([offer(VAN_DOWN), offer(VAN_THUR, text="US$42.50/day")])
    assert report.canonical_offers == 2 and report.duplicate_groups == 0 and report.variation_groups == 0
    assert set(report.offers_for(jobs, cars)["currency"]) == {"CA$", "US$"}
    report, *_ = combine([offer(VAN_DOWN), offer(VAN_THUR, text="CA$42.50/week")])
    assert report.canonical_offers == 2                                 # a different price basis


@pytest.mark.parametrize("change, reason", [
    ({"seats": np.nan}, UR.PRODUCT_INCOMPLETE), ({"car_name": None}, UR.PRODUCT_INCOMPLETE),
    ({"car_name": "  "}, UR.PRODUCT_INCOMPLETE), ({"bags": True}, UR.PRODUCT_INCOMPLETE),
    ({"price_num": np.nan}, UR.PRICE_INVALID), ({"price_num": -1.0}, UR.PRICE_INVALID),
    ({"price_num": 42.505}, UR.PRICE_INVALID), ({"price_num": float("inf")}, UR.PRICE_INVALID),
    ({"price_per_day": "42.50"}, UR.PRICE_TEXT_UNPARSED), ({"price_per_day": " CA$42.50/day"}, UR.PRICE_TEXT_UNPARSED),
    ({"price_per_day": None}, UR.PRICE_TEXT_UNPARSED), ({"price_per_day": "CA$42.51/day"}, UR.PRICE_TEXT_DISAGREES),
    ({"pickup_date": "2026-9-01"}, UR.RENTAL_DATE_UNPARSED), ({"return_date": None}, UR.RENTAL_DATE_UNPARSED),
    ({"location": "Vancouver downtown"}, UR.LOCATION_NOT_APPROVED),
])
def test_unassessable_rows_fail_closed_and_never_compare_equal(change, reason) -> None:
    report, jobs, cars = combine([offer(VAN_DOWN, **change), offer(VAN_THUR, **change), offer(VAN_AIR)])
    assert report.unassessable_groups == ((reason.value, 2),) and report.unassessable_rows == 2
    assert (report.combined_observations, report.canonical_offers, report.duplicate_groups) == (1, 1, 0)
    assert not report.ready and CB.UNASSESSABLE_OFFERS in report.blocking_reasons
    assert len(cars) == 3                                              # nothing dropped from the source


def test_the_governed_exclusion_is_out_of_scope_and_failed_controls_are_unassessable() -> None:
    statuses = [E.GOVERNED_EXCLUSION.value, E.ELIGIBLE.value, E.REPORTING_DAY_FAILED.value,
                E.RENTAL_DATES_FAILED.value, E.CAPTURE_PERIOD_UNASSIGNED.value]
    rows = [offer(VAN_DOWN), offer(VAN_THUR), offer(VAN_AIR), offer(VAN_AIR, name="SYNTH Vehicle B"),
            offer(TOR_DOWN, job="SYNTH-JOB-2")]
    report, *_ = combine(rows, statuses)
    assert (report.source_rows, report.out_of_scope_rows, report.combined_observations, report.unassessable_rows) == (
        5, 1, 1, 3)
    assert dict(report.unassessable_groups) == {"reporting_day_failed": 1, "rental_dates_failed": 1,
                                                "capture_period_unassigned": 1}
    assert report.duplicate_groups == 0 and not report.ready           # the excluded row never deduplicates
    excluded_only, *_ = combine(rows[:2], statuses[:2])
    assert excluded_only.ready and excluded_only.out_of_scope_rows == 1 and excluded_only.canonical_offers == 1


def test_an_unapproved_policy_combines_nothing() -> None:
    for policy in (canonical_offer_policy_from_record(load_decision_record(V7)),
                   canonical_offer_policy_from_record(None)):
        report, *_ = combine([offer(VAN_DOWN), offer(VAN_THUR)], policy=policy)
        assert report.offers is None and report.canonical_offers == 0
        assert report.blocking_reasons == (CB.POLICY_UNAVAILABLE,) and not report.ready


# ============================================================ binding and tampering


def test_stale_or_tampered_evidence_is_refused() -> None:
    jobs, cars, population, scheduled = world([offer(VAN_DOWN), offer(VAN_THUR)])
    report = assess_canonical_offers(jobs, cars, population=population, scheduled=scheduled, policy=POLICY,
                                     expected_streams=STREAMS)
    changed = cars.copy()
    changed.loc[0, "price_num"] = 99.0
    with pytest.raises(PricingPopulationError):
        assess_canonical_offers(jobs, changed, population=population, scheduled=scheduled, policy=POLICY,
                                expected_streams=STREAMS)
    with pytest.raises(PricingPopulationError):
        report.offers_for(jobs, changed)
    with pytest.raises(PricingPopulationError):
        population.detail_mask(jobs, changed)
    other_jobs = jobs.iloc[:0]
    with pytest.raises(PricingPopulationError):
        assess_canonical_offers(other_jobs, cars, population=population, scheduled=scheduled, policy=POLICY,
                                expected_streams=STREAMS)
    for forged in ({"canonical_offers": 5}, {"unassessable_rows": 0, "out_of_scope_rows": 1},
                   {"collapsed_observations": 0}, {"source_counts": ()}):
        with pytest.raises(ValueError):
            dataclasses.replace(report, **forged)
    with pytest.raises(PricingPopulationError):
        PricingPopulation(binding=population.binding, parent_status=population.parent_status,
                          detail_status=population.detail_status[:1])


def test_reports_hold_no_offer_level_values() -> None:
    report, *_ = combine([offer(VAN_DOWN), offer(VAN_THUR)])
    text = repr(report) + repr(report.policy)
    assert "SYNTH" not in text and "42.5" not in text and "2026-09" not in text
    assert parse_price_text("CA$1,234.50/day") == ("CA$", 123450, "day")
    for bad in ("CA$1234.5/day", "CA$12,34.50/day", "CA$ 12.50/day", "12.50", 12.5, None):
        assert parse_price_text(bad) is None


# ====================================================================== readiness


def readiness(canonical_offers, policy_report=None):  # type: ignore[no-untyped-def]
    from test_collection_schedule import pricing_with, real_frames, real_report

    j, c = real_frames()
    return pricing_with(j, c, real_report(j, c), canonical_offers=canonical_offers)[1]


def test_readiness_clears_the_merged_stream_blocker_only_with_a_ready_matching_report() -> None:
    assert readiness(passing_canonical_report()).ready
    missing = readiness(None)
    assert missing.blocking_reasons == (PB.CANONICAL_OFFER_COMBINATION_UNRESOLVED,
                                        PB.CANONICAL_OFFER_ASSESSMENT_MISSING)
    unapproved = CanonicalOfferReport(policy=CanonicalOfferPolicy(CS.NOT_APPROVED, "pricing-authorities-v7"),
                                      binding=None)
    assert set(readiness(unapproved).blocking_reasons) == {PB.CANONICAL_OFFER_COMBINATION_UNRESOLVED,
                                                           PB.CANONICAL_OFFER_POLICY_UNAVAILABLE}
    unassessable = passing_canonical_report(source_rows=1, unassessable_rows=1,
                                            unassessable_groups=(("price_invalid", 1),))
    assert set(readiness(unassessable).blocking_reasons) == {PB.CANONICAL_OFFER_COMBINATION_UNRESOLVED,
                                                             PB.CANONICAL_OFFERS_UNASSESSABLE}
    other_alias = passing_canonical_report(synthetic_offer_policy(streams=(VAN_DOWN, VAN_AIR)))
    assert set(readiness(other_alias).blocking_reasons) == {PB.CANONICAL_OFFER_COMBINATION_UNRESOLVED,
                                                            PB.CANONICAL_OFFER_POLICY_MISMATCH}
    with pytest.raises(TypeError):
        readiness({"canonical_offers": 1})
    assert {b.value for b in CB} <= {b.value for b in PB}


# ===================================================================== governance


def test_governance_document_records_the_decision_only() -> None:
    raw = (ROOT / GOVERNANCE).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in ("2026-10-06", "Collection owner and business owner (joint", "separately required source streams",
                   "`vancouver` / `Vancouver Downtown`", "`Vancouver Thurlow`", "order-independent",
                   "no stream priority", "row indexes are never used", "currency", "never assumed",
                   "price variation", "no averaging", "Fail closed", "missing values never compare equal",
                   "pricing-authorities-v8", "Not supplied", "roles were not named"):
        assert phrase in text, phrase
    assert "SYNTH" not in raw and "@" not in raw and not re.search(r"\$\s?\d|\b\d+\.\d{2}\b|\d{6,}", raw)
