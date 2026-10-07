"""The pricing-eligible population: governed exclusion first, then every foundational control (synthetic rows)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from test_collection_schedule import CAL_AIR, CAL_DOWN, EXCLUDED_HOUR, real_frames, real_report

from ql2_sixt_canada_analysis.authority_decisions import load_decision_record
from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
from ql2_sixt_canada_analysis.pricing_population import (
    DetailEligibility as E,
    PricingPopulationError,
    build_pricing_population,
    frame_binding,
)
from ql2_sixt_canada_analysis.rental_dates import DerivedRentalPeriods
from ql2_sixt_canada_analysis.schemas import TEMPORAL_RECONCILIATION
from ql2_sixt_canada_analysis.temporal import derive_reporting_days
from ql2_sixt_canada_analysis.temporal_authority import temporal_authority_from_record

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "docs" / "decisions" / "pricing_authorities" / "v8.toml"
DEF = temporal_authority_from_record(load_decision_record(V8), TEMPORAL_RECONCILIATION,
                                     current_expected_stream_contract()).definition


def decided_world(rental_ok=None):  # type: ignore[no-untyped-def]
    """The decided situation on synthetic rows: the governed Calgary capture has Airport rows only."""
    j, c = real_frames(skip={(CAL_DOWN, EXCLUDED_HOUR)})
    j = j.assign(scrape_date=j["finished_at"].str[:10])                 # the parent-local calendar date
    c = c.assign(scrape_date=c["job_finished_at"].str[:10], scraped_at="2026-01-01 00:00:00 MST")
    scheduled = real_report(j, c)
    days = derive_reporting_days(j, c, DEF)
    rental = DerivedRentalPeriods(jobs=pd.DataFrame({"pricing_eligible": True}, index=j.index),
                                  cars=pd.DataFrame({"pricing_eligible": rental_ok if rental_ok is not None
                                                     else True}, index=c.index))
    return j, c, scheduled, days, rental


def test_the_governed_capture_is_excluded_and_counted_never_deleted() -> None:
    j, c, scheduled, days, rental = decided_world()
    before = (j.copy(deep=True), c.copy(deep=True))
    population = build_pricing_population(j, c, scheduled=scheduled, reporting_days=days, rental_periods=rental)
    assert (population.nominal_parent_captures, population.excluded_parent_captures,
            population.ineligible_parent_captures, population.eligible_parent_captures) == (270, 1, 0, 269)
    assert (population.nominal_detail_rows, population.excluded_detail_rows, population.ineligible_detail_rows,
            population.eligible_detail_rows) == (629, 1, 0, 628)
    eligible = population.eligible_details(j, c)
    excluded_job = j.loc[scheduled.capture_exclusions.parent_mask(j), "job_id"].iloc[0]
    assert excluded_job not in set(eligible["job_id"]) and excluded_job in set(c["job_id"])   # raw row kept
    assert c.loc[c["job_id"] == excluded_job, ["city", "location"]].apply(tuple, axis=1).tolist() == [CAL_AIR]
    pd.testing.assert_frame_equal(j, before[0]), pd.testing.assert_frame_equal(c, before[1])
    assert "SYNTH" not in repr(population) and population.detail_counts()[E.GOVERNED_EXCLUSION.value] == 1


def test_the_governed_exclusion_wins_and_other_failures_are_counted_by_control() -> None:
    j, c, scheduled, days, _ = decided_world()
    excluded_rows = scheduled.capture_exclusions.detail_mask(c)
    rental_ok = ~excluded_rows                                          # the excluded row also fails rental
    rental_ok[[0, 1]] = False                                            # two ordinary rows fail rental dates
    rental = DerivedRentalPeriods(jobs=pd.DataFrame({"pricing_eligible": True}, index=j.index),
                                  cars=pd.DataFrame({"pricing_eligible": rental_ok}, index=c.index))
    c2 = c.copy()
    c2.loc[c2.index[2], "scrape_date"] = "2001-01-01"                  # a reporting-day mismatch
    days2 = derive_reporting_days(j, c2, DEF)
    population = build_pricing_population(j, c2, scheduled=real_report(j, c2), reporting_days=days2,
                                          rental_periods=rental)
    counts = population.detail_counts()
    assert counts == {"eligible": 625, "governed_exclusion": 1, "reporting_day_failed": 1,
                      "rental_dates_failed": 2, "capture_period_unassigned": 0}
    assert population.ineligible_detail_rows == 3                      # failing rows are never dropped silently


def test_stale_misaligned_or_unmatched_inputs_fail_closed() -> None:
    j, c, scheduled, days, rental = decided_world()
    population = build_pricing_population(j, c, scheduled=scheduled, reporting_days=days, rental_periods=rental)
    assert population.binding == frame_binding(j, c)
    changed = c.copy()
    changed.loc[changed.index[0], "location"] = "SYNTH Branch"
    with pytest.raises(PricingPopulationError):
        population.detail_mask(j, changed)
    with pytest.raises(PricingPopulationError):
        build_pricing_population(j, c, scheduled=scheduled, reporting_days=days,
                                 rental_periods=DerivedRentalPeriods(jobs=rental.jobs, cars=rental.cars.iloc[:-1]))
    with pytest.raises(PricingPopulationError):
        build_pricing_population(j, c, scheduled=scheduled, reporting_days=None, rental_periods=rental)
    without = j[j["job_id"] != j.loc[scheduled.capture_exclusions.parent_mask(j), "job_id"].iloc[0]]
    rest = c[c["job_id"].isin(without["job_id"])]
    unmatched = real_report(without, rest)
    assert unmatched.unmatched_exclusions == 1
    with pytest.raises(PricingPopulationError):
        build_pricing_population(without, rest, scheduled=unmatched, reporting_days=derive_reporting_days(
            without, rest, DEF), rental_periods=DerivedRentalPeriods(
            jobs=pd.DataFrame({"pricing_eligible": True}, index=without.index),
            cars=pd.DataFrame({"pricing_eligible": True}, index=rest.index)))


def test_the_baseline_reports_population_reporting_day_offers_and_authority_in_aggregate() -> None:
    import json
    import re

    from stream_contract_fixtures import passing_canonical_report
    from test_collection_schedule import pricing_with

    from ql2_sixt_canada_analysis.authority_decisions import load_current_decision_record
    from ql2_sixt_canada_analysis.pricing_baseline import (
        UnsafeBaselineValueError,
        build_pricing_baseline,
        render_baseline_markdown,
    )
    from ql2_sixt_canada_analysis.stability import assess_vehicle_attribute_stability
    from ql2_sixt_canada_analysis.temporal import assess_temporal_reconciliation

    j, c, scheduled, days, rental = decided_world()
    population = build_pricing_population(j, c, scheduled=scheduled, reporting_days=days, rental_periods=rental)
    _, pricing = pricing_with(j, c, scheduled, canonical_offers=passing_canonical_report())
    authority = temporal_authority_from_record(load_decision_record(V8), TEMPORAL_RECONCILIATION,
                                               current_expected_stream_contract())
    baseline = build_pricing_baseline(
        pricing=pricing, jobs=j, cars=c, temporal=assess_temporal_reconciliation(j, c, DEF),
        vehicle_stability=assess_vehicle_attribute_stability(c.assign(car_name="SYNTH Vehicle")),
        temporal_contract=DEF, temporal_authority=authority, reporting_days=days, pricing_population=population,
        authority_record=load_current_decision_record())
    pp, rd, co = baseline.pricing_population, baseline.reporting_day, baseline.canonical_offers
    assert (pp.status, pp.excluded_parent_captures, pp.eligible_parent_captures, pp.excluded_detail_rows,
            pp.eligible_detail_rows) == ("available", 1, 269, 1, 628)
    assert (rd.status, rd.source_field, rd.timezone_mode, rd.scrape_date_derivation, rd.date_clean_status,
            rd.date_clean_rule_unavailable) == ("available", "jobs.finished_at", "parent_city", "reporting_day",
                                                "retired_from_pricing", False)
    assert dict(rd.parent_status_counts)["agrees"] == 270 and dict(rd.detail_status_counts)["agrees"] == 629
    assert (co.policy_status, co.stream, co.blockers) == ("approved", ("vancouver", "Vancouver Downtown"), ())
    assert len(baseline.authority_statuses) == 23 and all(s == "approved" for _, s in baseline.authority_statuses)
    schedule = baseline.collection_schedule
    assert (schedule.nominal_stream_periods, schedule.excluded_stream_periods, schedule.required_stream_periods,
            schedule.covered_stream_periods, schedule.excluded_parent_captures) == (630, 2, 628, 628, 1)
    markdown = render_baseline_markdown(baseline, commit="abc1234", date="2026-10-06")
    for heading in ("Pricing-eligible population", "Reporting day and source dates", "Canonical offers",
                    "Authority decision statuses", "| `CANONICAL_OFFER_COMBINATION` | `approved` |",
                    "| calgary / Calgary Downtown | 90 | 1 | 89 | 89 | 0 | 0 | none |", "Overall state:"):
        assert heading in markdown, heading
    assert "SYNTH" not in markdown and "job_id" not in markdown
    assert not re.search(r"\d{8}T\d{6}Z|\d{4}-\d{2}-\d{2}(?! \(| \|)|\$\s?\d|\b\d+\.\d{2}\b", markdown.replace(
        "2026-10-06", ""))
    json.dumps(baseline.to_dict())
    import dataclasses
    forged = dataclasses.replace(co, unassessable_groups=(("SYNTH value 1", 1),))
    with pytest.raises(UnsafeBaselineValueError):
        dataclasses.replace(baseline, canonical_offers=forged).to_dict()
