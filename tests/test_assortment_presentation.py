"""Visible-assortment presentation: sanitized tables, timeline view, narrative and the persistence gate.

Every observation is fabricated (``SYNTH-*`` products, synthetic branches,
synthetic 2030 capture periods, prices and policies). The only committed
values read are approved configuration (stream keys, time zones, location roles
and the canonical-offer policy) through the shared synthetic pipeline world.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_price_change_analysis import merge, path
from test_price_change_events import (
    CAL_DOWN,
    LOC,
    OTHER,
    TOR_AIR,
    TOR_DOWN,
    VAN_DOWN,
    VAN_THUR,
    P,
    at,
    pipeline_result,
    readiness_for,
    synthetic_world,
    timeline,
)
from test_visible_assortment import approved, build, car, sets, synthetic_floor_rule

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis import assortment_presentation as ap
from ql2_sixt_canada_analysis import visible_assortment as va
from ql2_sixt_canada_analysis.assortment_contract import (
    ASSORTMENT_TIMELINE_COLUMNS,
    AnomalyPolicyStatus as APS,
    AssessabilityStatus as AS,
)
from ql2_sixt_canada_analysis.assortment_presentation import (
    ASSORTMENT_COLUMN_ALLOWLIST,
    ASSORTMENT_COLUMN_KINDS,
    ASSORTMENT_RECONCILIATION_CHECKS,
    ASSORTMENT_TABLE_SCHEMAS,
    FORBIDDEN_ASSORTMENT_COLUMNS,
    NARRATIVE_SECTIONS,
    TIMELINE_SOURCE_COLUMNS,
    AssortmentPresentationBlocker as PB,
    AssortmentPresentationReport,
    AssortmentPresentationResult,
    AssortmentPresentationStatus,
    AssortmentPresentationTables,
    AssortmentPrivacyError,
    DropPattern,
    Simultaneity,
    assortment_persistence_approved,
    assortment_presentation_from_pipeline,
    assortment_timeline_png,
    assortment_timeline_source_frame,
    build_assortment_narrative,
    build_assortment_presentation_tables,
    run_assortment_presentation,
    validate_assortment_table,
)
from ql2_sixt_canada_analysis.price_change_events import CaptureState as CS, IntervalBreak as IB
from ql2_sixt_canada_analysis.readiness import PricingBlocker
from ql2_sixt_canada_analysis.visible_assortment import AssortmentReconciliationError

ROOT = Path(__file__).resolve().parents[1]
PREV, CUR = "previous_scheduled_capture_period", "scheduled_capture_period"

#: Toronto Downtown loses car B at hour 1 while car A's price rises (a falling assortment with a price
#: increase); Toronto Airport's hour-1 capture is an eligible empty capture (a simultaneous drop); Vancouver
#: Downtown and its Thurlow alias both carry car V, and both lose it at hour 2 (one canonical drop).
RICH = merge(path(TOR_DOWN, (50.0, 55.0, 55.0)), path(TOR_DOWN, (60.0, None, None), "SYNTH Car B"),
             path(TOR_AIR, (50.0, 50.0, 52.0)), path(TOR_AIR, (None, None, 1.0), "SYNTH Car D"),
             path(VAN_DOWN, (70.0, 70.0, None), "SYNTH Car V"), path(VAN_THUR, (70.0, 70.0, None), "SYNTH Car V"),
             path(VAN_DOWN, (40.0, 40.0, 36.0), "SYNTH Car W"))
FORBIDDEN_WORDS = re.compile(r"\b(proved|proves|caused|causes|supplier removed|collection failed|statistically "
                             r"significant|anomal\w*|outlier\w*|alert\w*)\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def world():  # type: ignore[no-untyped-def]
    return synthetic_world(products=RICH, withheld={(TOR_AIR, 1)})


@pytest.fixture(scope="module")
def presentation(world):  # type: ignore[no-untyped-def]
    result = assortment_presentation_from_pipeline(pipeline_result(world))
    assert result.completed
    return result


@pytest.fixture(scope="module")
def tables(presentation):  # type: ignore[no-untyped-def]
    return presentation.tables


def records(frame: pd.DataFrame) -> list[dict]:
    return [{k: (None if v is pd.NA or (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
            for r in frame.astype(object).to_dict("records")]


def located(frame: pd.DataFrame, key) -> list[dict]:  # type: ignore[no-untyped-def]
    return [r for r in records(frame) if (r["canonical_city"], r["canonical_location"]) == key]


def tables_of(rows, tls, **kwargs) -> AssortmentPresentationTables:  # type: ignore[no-untyped-def]
    return build_assortment_presentation_tables(build(rows, tls, **kwargs))


def unusual_free(text: str) -> str:
    return text.replace("unusual-drop", "").replace("unusual_drop", "")


# ============================================================================ construction


def test_a_completed_result_produces_every_validated_table(presentation, tables) -> None:  # type: ignore[no-untyped-def]
    report = presentation.report
    assert report.status is AssortmentPresentationStatus.COMPLETED and report.blockers == ()
    assert report.reconciled and report.reconciliation_checks == len(ASSORTMENT_RECONCILIATION_CHECKS)
    assert report.anomaly_policy_status is APS.UNAVAILABLE and report.persistence_approved is False
    assert dict(report.table_rows) == {name: len(frame) for name, frame in tables.items()}
    for name, frame in tables.items():
        assert tuple(frame.columns) == ASSORTMENT_TABLE_SCHEMAS[name]
        validate_assortment_table(name, frame)                                 # project location authority
    assert (tables.reconciliation_summary["status"] == "reconciled").all()
    assert tuple(tables.reconciliation_summary["check"]) == ASSORTMENT_RECONCILIATION_CHECKS
    assert len(tables.evidence_id) == 16 and "SYNTH" not in repr(presentation)


def test_blocked_upstream_produces_no_tables_figure_or_findings(world) -> None:  # type: ignore[no-untyped-def]
    not_ready = readiness_for(world["cars"], world["scheduled"], world["canonical_offers"],
                              (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE,))
    blocked = assortment_presentation_from_pipeline(dataclasses.replace(pipeline_result(world), pricing=not_ready))
    assert not blocked.completed and blocked.tables is None and blocked.assortment is None
    assert blocked.report.blockers == (PB.ASSORTMENT_NOT_COMPLETED,)
    assert set(blocked.report.upstream_blockers) >= {"pricing_not_ready",
                                                     PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE.value}
    assert blocked.report.table_rows == () and not blocked.report.reconciled
    narrative = build_assortment_narrative(blocked)
    assert not narrative.completed and not re.search(r"\d", narrative.text)
    assert "assortment_not_completed" in narrative.text and "SYNTH" not in narrative.text
    with pytest.raises(TypeError):
        assortment_timeline_png(blocked.tables)                               # type: ignore[arg-type]
    with pytest.raises(AssortmentReconciliationError):
        AssortmentPresentationResult(blocked.report, tables=None, assortment=build(sets({0: "A"}), [timeline([0])]))


def test_the_pipeline_runs_exactly_once_and_nothing_is_written(world, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis import pricing_pipeline

    calls = []
    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline",
                        lambda raw_dir=None: calls.append(raw_dir) or pipeline_result(world))
    monkeypatch.chdir(tmp_path)
    result = run_assortment_presentation(tmp_path)
    assert result.completed and calls == [tmp_path] and os.listdir(tmp_path) == []


def test_the_presentation_never_recalculates_sets_ratios_or_price_outcomes(presentation, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("the presentation must not recalculate")

    from ql2_sixt_canada_analysis import assortment_contract, price_change_events

    for module, name in ((va, "calculate_visible_assortment"), (va, "compare_assortment"), (va, "_product_sets"),
                         (assortment_contract, "compare_assortment"),
                         (price_change_events, "classify_price_change_candidates"),
                         (price_change_events, "capture_timelines")):
        monkeypatch.setattr(module, name, forbidden)
    again = build_assortment_presentation_tables(presentation.assortment)
    pd.testing.assert_frame_equal(again.assortment_timeline, presentation.assortment.timeline)
    source = (ROOT / "src" / "ql2_sixt_canada_analysis" / "assortment_presentation.py").read_text(encoding="utf-8")
    for name in ("compare_assortment(", "calculate_visible_assortment(", "classify_price_change", "capture_timelines(",
                 "frozenset(products", ".membership"):
        assert name not in source


def test_evidence_from_a_different_run_is_rejected(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    other = va.visible_assortment_from_pipeline(pipeline_result(synthetic_world(hours=2)))
    monkeypatch.setattr(va, "visible_assortment_from_pipeline", lambda run: other)
    result = assortment_presentation_from_pipeline(pipeline_result(world))
    assert result.report.blockers == (PB.EVIDENCE_MISMATCH,) and result.tables is None


def test_a_failed_reconciliation_blocks_the_presentation(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real = ap._reconciliation

    def failing(*args):  # type: ignore[no-untyped-def]
        frame = real(*args)
        frame.loc[0, ["observed", "status"]] = [frame.loc[0, "observed"] + 1, "failed"]
        return frame

    monkeypatch.setattr(ap, "_reconciliation", failing)
    result = assortment_presentation_from_pipeline(pipeline_result(world))
    assert result.report.blockers == (PB.RECONCILIATION_FAILED,) and result.tables is None


def test_inputs_are_not_mutated_and_output_is_deterministic(presentation) -> None:  # type: ignore[no-untyped-def]
    result = presentation.assortment
    before = result.timeline.copy(deep=True)
    first, second = build_assortment_presentation_tables(result), build_assortment_presentation_tables(result)
    pd.testing.assert_frame_equal(result.timeline, before)
    for (name, a), (_, b) in zip(first.items(), second.items()):
        pd.testing.assert_frame_equal(a, b, obj=name)
    assert first.evidence_id == second.evidence_id
    with pytest.raises(TypeError):
        build_assortment_presentation_tables(object())                        # type: ignore[arg-type]


# ============================================================================ schema and privacy


def test_every_table_has_an_exact_schema_and_every_column_one_kind() -> None:
    assert ASSORTMENT_TABLE_SCHEMAS["assortment_timeline"] == ASSORTMENT_TIMELINE_COLUMNS
    assert set(ASSORTMENT_COLUMN_KINDS) == ASSORTMENT_COLUMN_ALLOWLIST
    assert set(ASSORTMENT_COLUMN_KINDS.values()) <= {"count", "signed_count", "flag", "share", "statistic", "period",
                                                     "city", "location", "enum"}
    for name, schema in ASSORTMENT_TABLE_SCHEMAS.items():
        assert len(schema) == len(set(schema)), name
    every_forbidden = set().union(*FORBIDDEN_ASSORTMENT_COLUMNS.values())
    assert not every_forbidden & ASSORTMENT_COLUMN_ALLOWLIST
    assert {"car_name", "pickup_date", "currency", "price_cents", "job_id", "source_location_labels", "finished_at",
            "membership", "path"} <= every_forbidden


@pytest.mark.parametrize(("rule", "column"), [(rule, column) for rule, cols in FORBIDDEN_ASSORTMENT_COLUMNS.items()
                                              for column in sorted(cols)])
def test_forbidden_columns_are_refused_by_rule(tables, rule, column) -> None:  # type: ignore[no-untyped-def]
    frame = tables.location_summary.assign(**{column: "SYNTH-SECRET-VALUE"})
    with pytest.raises(AssortmentPrivacyError, match=f"rule {rule}") as info:
        validate_assortment_table("location_summary", frame, approved_locations=tables.approved_locations)
    assert "SYNTH-SECRET-VALUE" not in str(info.value)


@pytest.mark.parametrize(("table", "column", "value", "rule"), [
    ("location_summary", "scheduled_captures", -1, "value_domain"),
    ("location_summary", "scheduled_captures", True, "value_domain"),
    ("location_summary", "retention_median", 1.5, "value_domain"),
    ("location_summary", "anomaly_policy_status", "SYNTH free text", "value_domain"),
    ("location_summary", "canonical_city", "synth-unapproved", "value_domain"),
    ("location_summary", "canonical_location", "SYNTH Unapproved", "value_domain"),
    ("location_summary", "eligible_captures", None, "missing_value"),
    ("assortment_timeline", "scheduled_capture_period", "2030-03-04 08:00", "value_domain"),
    ("assortment_timeline", "capture_state", "SYNTH free text", "value_domain"),
    ("assortment_timeline", "retention", float("inf"), "value_domain"),
    ("assortment_timeline", "net_change", 0.5, "value_domain"),
    ("cross_location_drops", "simultaneity", "SYNTH widespread", "value_domain"),
    ("reconciliation_summary", "status", "SYNTH ok", "value_domain"),
])
def test_semantic_domains_and_missingness_fail_closed(tables, table, column, value, rule) -> None:  # type: ignore[no-untyped-def]
    frame = getattr(tables, table).astype(object).copy()
    if frame.empty:
        pytest.skip("the fixture has no row in this table")
    frame.at[frame.index[0], column] = value
    with pytest.raises(AssortmentPrivacyError, match=f"rule {rule}") as info:
        validate_assortment_table(table, frame, approved_locations=tables.approved_locations)
    assert "SYNTH" not in str(info.value)


def test_unknown_columns_schema_order_and_duplicate_keys_fail(tables) -> None:  # type: ignore[no-untyped-def]
    approved_keys = tables.approved_locations
    with pytest.raises(AssortmentPrivacyError, match="rule allowlist"):
        validate_assortment_table("location_summary", tables.location_summary.assign(note="x"),
                                  approved_locations=approved_keys)
    with pytest.raises(AssortmentPrivacyError, match="rule schema"):
        validate_assortment_table("location_summary", tables.location_summary.iloc[:, ::-1],
                                  approved_locations=approved_keys)
    with pytest.raises(AssortmentPrivacyError, match="rule duplicate_record"):
        validate_assortment_table("location_summary", pd.concat([tables.location_summary,
                                                                 tables.location_summary.iloc[[0]]]),
                                  approved_locations=approved_keys)
    with pytest.raises(AssortmentPrivacyError, match="rule unknown_table"):
        validate_assortment_table("membership_detail", tables.location_summary, approved_locations=approved_keys)
    with pytest.raises(AssortmentPrivacyError, match="rule not_a_table"):
        validate_assortment_table("location_summary", records(tables.location_summary),
                                  approved_locations=approved_keys)
    membership = pd.DataFrame(columns=list(va.MEMBERSHIP_COLUMNS))
    with pytest.raises(AssortmentPrivacyError, match="rule detailed_frame"):
        validate_assortment_table("location_summary", membership, approved_locations=approved_keys)


@pytest.mark.parametrize("change", [
    lambda t, i: t.__setitem__("retained_count", t["retained_count"].mask(t.index == i, 1)),     # break metric
    lambda t, i: t.__setitem__("retention_denominator_status",
                               t["retention_denominator_status"].mask(t.index == i, "defined")),
])
def test_break_rows_cannot_carry_interval_metrics(tables, change) -> None:  # type: ignore[no-untyped-def]
    frame = tables.assortment_timeline.copy()
    seed = frame.index[frame["assessability_status"] == AS.SEED_CAPTURE.value][0]
    change(frame, seed)
    with pytest.raises(AssortmentPrivacyError):
        validate_assortment_table("assortment_timeline", frame, approved_locations=tables.approved_locations)


def test_ratio_presence_and_unusual_drop_follow_their_statuses(tables) -> None:  # type: ignore[no-untyped-def]
    frame = tables.assortment_timeline.copy()
    assessed = frame.index[frame["assessability_status"] == AS.ASSESSED.value][0]
    broken = frame.copy()
    broken.loc[assessed, "retention"] = pd.NA
    with pytest.raises(AssortmentPrivacyError, match="rule denominator_status"):
        validate_assortment_table("assortment_timeline", broken, approved_locations=tables.approved_locations)
    certain = frame.copy()
    certain["unusual_drop"] = certain["unusual_drop"].fillna(False)
    with pytest.raises(AssortmentPrivacyError, match="rule unusual_drop_policy"):
        validate_assortment_table("assortment_timeline", certain, approved_locations=tables.approved_locations)


# ============================================================================ timeline


def test_the_timeline_covers_every_status_and_equals_the_engine() -> None:
    rows = [*sets({0: "AB", 1: "A", 3: "A", 4: "AC"}), car(5, "SYNTH Car C")]
    tls = [timeline([0, 1, 2, 3, 4, 5, 6, 7], states={2: CS.MISSING_CAPTURE, 6: CS.GOVERNED_EXCLUSION})]
    result = build(rows, tls)
    t = build_assortment_presentation_tables(result)
    pd.testing.assert_frame_equal(t.assortment_timeline, result.timeline)
    statuses = [r["assessability_status"] for r in records(t.assortment_timeline)]
    assert statuses == ["seed_capture", "assessed", "capture_not_eligible", "interval_break", "assessed",
                        "assessed", "capture_not_eligible", "interval_break"]
    state = {r[CUR]: r for r in records(t.assortment_timeline)}
    assert state[P(2)]["returned_product_count"] is None and state[P(7)]["returned_product_count"] == 0
    assert state[P(7)]["interval_break_reason"] == IB.GOVERNED_EXCLUSION.value
    assert state[P(5)]["returned_product_count"] == 1 and state[P(5)]["retention"] == 0.5


def test_the_timeline_source_never_crosses_a_break_or_zero_fills(tables) -> None:  # type: ignore[no-untyped-def]
    rows = [*sets({0: "AB", 1: "", 3: "A", 4: "A"})]
    t = tables_of(rows, [timeline([0, 1, 2, 3, 4], states={2: CS.MISSING_CAPTURE})])
    source = assortment_timeline_source_frame(t)
    assert tuple(source.columns) == TIMELINE_SOURCE_COLUMNS
    by_period = {r["period"]: r for r in records(source)}
    assert by_period[P(1)]["returned_product_count"] == 0.0                    # an eligible empty capture
    assert by_period[P(2)]["returned_product_count"] is None and not by_period[P(2)]["eligible"]
    assert by_period[P(0)]["segment"] == by_period[P(1)]["segment"] != by_period[P(3)]["segment"]
    assert by_period[P(3)]["segment"] == by_period[P(4)]["segment"]
    assert int(source["observed_drop"].sum()) == len(t.observed_drop_review) == 1
    tampered = dataclasses.replace(t)                                          # re-validated copy
    object.__setattr__(tampered, "observed_drop_review", t.observed_drop_review.iloc[0:0])
    with pytest.raises(AssortmentReconciliationError):
        assortment_timeline_source_frame(tampered)
    full = assortment_timeline_source_frame(tables)
    assert len(full) == len(tables.assortment_timeline)


def test_the_figure_renders_in_memory_closes_and_writes_nothing(tables, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    before = plt.get_fignums()
    png = assortment_timeline_png(tables, dpi=60)
    assert png[:4] == b"\x89PNG" and plt.get_fignums() == before and os.listdir(tmp_path) == []
    assert b"SYNTH" not in png


# ============================================================================ location summary


def test_location_totals_reconcile_to_the_timeline_and_engine(presentation, tables) -> None:  # type: ignore[no-untyped-def]
    overall = presentation.assortment.report.overall
    summary = tables.location_summary
    assert int(summary["assessed_intervals"].sum()) == overall.assessed_intervals
    assert int(summary["total_additions"].sum()) == overall.additions
    assert int(summary["total_removals"].sum()) == overall.removals
    assert int(summary["observed_drop_intervals"].sum()) == overall.drop_intervals == len(tables.observed_drop_review)
    assert int(summary["empty_captures"].sum()) == overall.empty_captures == 1
    tor = located(summary, TOR_DOWN)[0]
    assert (tor["returned_count_min"], tor["returned_count_median"], tor["returned_count_max"]) == (2.0, 2.0, 3.0)
    assert (tor["retention_contributors"], tor["retention_zero_denominator"]) == (2, 0)
    assert tor["retention_median"] == (2 / 3 + 1.0) / 2                       # median of 2/3 and 1
    for row in records(summary):
        assert row["retention_contributors"] + row["retention_zero_denominator"] == row["assessed_intervals"]


def test_contributor_counts_govern_every_statistic_and_zero_denominators_stay_explicit() -> None:
    t = tables_of(sets({0: "", 1: "", 2: "A"}), [timeline([0, 1, 2])])
    row = records(t.location_summary)[0]
    assert (row["retention_contributors"], row["retention_zero_denominator"], row["retention_median"]) == (0, 2, None)
    assert (row["jaccard_contributors"], row["jaccard_zero_denominator"], row["jaccard_median"]) == (1, 1, 0.0)
    assert row["empty_captures"] == 2 and row["returned_count_median"] == 0.0
    only_breaks = tables_of(sets({0: "A", 2: "A"}), [timeline([0, 1, 2], states={1: CS.MISSING_CAPTURE})])
    quiet = records(only_breaks.location_summary)[0]
    assert quiet["assessed_intervals"] == 0 and quiet["retention_median"] is None and quiet["interval_breaks"] == 2
    tampered = t.location_summary.copy()
    tampered["retention_median"] = 0.5
    with pytest.raises(AssortmentPrivacyError, match="rule contributors"):
        validate_assortment_table("location_summary", tampered, approved_locations=t.approved_locations)


# ============================================================================ observed-drop review


def test_only_assessed_drops_appear_in_deterministic_review_order() -> None:
    spec = {0: "ABCD", 1: "ABCDE", 2: "AB", 3: "A", 5: "", 6: "AB"}
    t = tables_of(sets(spec), [timeline([0, 1, 2, 3, 4, 5, 6], states={4: CS.MISSING_CAPTURE})])
    review = records(t.observed_drop_review)
    assert [(r[CUR], r["absolute_drop"]) for r in review] == [(P(2), 3), (P(3), 1)]   # increases and breaks absent
    assert [r["drop_pattern"] for r in review] == [DropPattern.NET_CONTRACTION.value] * 2
    assert all(r["unusual_drop"] is None and r["review_status"] == "observed_drop_review_candidate" for r in review)
    ties = tables_of([*sets({0: "AB", 1: "A"}), *sets({0: "ABCD", 1: "ABC"}, loc=OTHER)],
                     [timeline([0, 1]), timeline([0, 1], loc=OTHER)], locations=[LOC, OTHER])
    assert [r["canonical_location"] for r in records(ties.observed_drop_review)] == [LOC[1], OTHER[1]]  # rate first


def test_empty_current_captures_and_turnover_are_patterns_not_failures() -> None:
    t = tables_of(sets({0: "AB", 1: "", 2: "C", 3: "D"}), [timeline([0, 1, 2, 3])])
    patterns = {r[CUR]: r["drop_pattern"] for r in records(t.observed_drop_review)}
    assert patterns == {P(1): DropPattern.EMPTY_CURRENT_CAPTURE.value}
    turnover = tables_of(sets({0: "AB", 1: "C"}), [timeline([0, 1])])
    assert records(turnover.observed_drop_review)[0]["drop_pattern"] == DropPattern.COMPLETE_TURNOVER.value
    text = build_assortment_narrative(t).text
    assert "failure" not in text.lower() and "withdraw" not in text.lower()


def test_an_approved_synthetic_policy_fills_unusual_drop_only_through_the_engine() -> None:
    result = build(sets({0: "ABCD", 1: "AB", 2: "A"}), [timeline([0, 1, 2])],
                   policy=approved(synthetic_floor_rule(2, 0.5)))
    t = build_assortment_presentation_tables(result)
    assert [r["unusual_drop"] for r in records(t.observed_drop_review)] == [True, False]
    recon = dict(zip(t.reconciliation_summary["check"], t.reconciliation_summary["observed"]))
    assert recon["unusual_drop_matches_policy_status"] == 1


# ============================================================================ cross-location review


def test_isolated_simultaneous_and_separate_periods(tables) -> None:  # type: ignore[no-untyped-def]
    t = tables_of([*sets({0: "AB", 1: "A", 2: "A"}), *sets({0: "AB", 1: "A", 2: "", 3: ""}, loc=OTHER)],
                  [timeline([0, 1, 2]), timeline([0, 1, 2, 3], loc=OTHER)])
    cross = records(t.cross_location_drops)
    assert [(r[CUR], r["locations_with_observed_drop"], r["simultaneity"]) for r in cross] == [
        (P(1), 2, Simultaneity.SIMULTANEOUS.value), (P(2), 1, Simultaneity.ISOLATED.value)]
    assert [r["assessed_locations"] for r in cross] == [2, 2]
    assert int(t.cross_location_drops["locations_with_observed_drop"].sum()) == len(t.observed_drop_review)
    rich = {r[CUR]: r for r in records(tables.cross_location_drops)}
    assert rich[at("toronto", 1)]["simultaneity"] == Simultaneity.SIMULTANEOUS.value
    vancouver = [r for r in records(tables.observed_drop_review) if r["canonical_city"] == "vancouver"]
    assert len(vancouver) == 1 and vancouver[0]["canonical_location"] == VAN_DOWN[1]   # aliases counted once
    assert vancouver[0]["absolute_drop"] == 1 and VAN_THUR[1] not in tables.observed_drop_review.to_string()


# ============================================================================ price coincidence


def test_coincidence_uses_exact_location_intervals_and_keeps_directions_apart(tables) -> None:  # type: ignore[no-untyped-def]
    tor = located(tables.price_coincidence_summary, TOR_DOWN)[0]
    assert (tor["price_increase_intervals"], tor["price_decrease_intervals"], tor["coincident_intervals"]) == (1, 0, 1)
    assert tor["falling_with_increase_intervals"] == tor["drop_with_price_increase_intervals"] == 1
    assert tor["coincidence_share_of_assortment_changes"] == 1.0 and tor["coincidence_share_of_price_changes"] == 1.0
    air = located(tables.price_coincidence_summary, TOR_AIR)[0]
    assert (air["price_increase_intervals"], air["coincident_intervals"]) == (0, 0)  # filler appeared/disappeared
    van = located(tables.price_coincidence_summary, VAN_DOWN)[0]
    assert (van["price_decrease_intervals"], van["drop_with_price_decrease_intervals"]) == (1, 1)
    calm = located(tables.price_coincidence_summary, CAL_DOWN)[0]
    assert calm["coincidence_share_of_price_changes"] is None and calm["coincidence_share_of_assortment_changes"] is None


def test_unchanged_appeared_disappeared_and_ambiguous_never_coincide() -> None:
    rows = [car(0, "SYNTH Car A", 1), car(0, "SYNTH Car A", 2), car(1, "SYNTH Car A", 3),     # ambiguous
            car(0, "SYNTH Car B"), car(1, "SYNTH Car B"),                                   # unchanged
            car(1, "SYNTH Car C"), car(0, "SYNTH Car D"),                                   # appeared, disappeared
            car(1, "SYNTH Car E", currency="CA$"), car(2, "SYNTH Car E", currency="US$")]   # unit change
    t = tables_of(rows, [timeline([0, 1, 2])])
    row = records(t.price_coincidence_summary)[0]
    assert (row["price_change_intervals"], row["coincident_intervals"]) == (0, 0)
    assert row["assortment_change_intervals"] == 2                            # C/D turnover, then A/B/C removed


def test_changes_in_different_intervals_never_coincide() -> None:
    rows = [car(0, "SYNTH Car A", 50), car(1, "SYNTH Car A", 60), car(2, "SYNTH Car A", 60), car(0, "SYNTH Car B"),
            car(1, "SYNTH Car B")]
    t = tables_of(rows, [timeline([0, 1, 2])])
    row = records(t.price_coincidence_summary)[0]
    assert (row["price_change_intervals"], row["observed_drop_intervals"], row["coincident_intervals"]) == (1, 1, 0)
    assert row["falling_with_increase_intervals"] == 0


# ============================================================================ narrative


def test_the_narrative_has_every_section_and_reconciles_to_the_tables(tables) -> None:  # type: ignore[no-untyped-def]
    narrative = build_assortment_narrative(tables)
    assert narrative.completed and tuple(h for h, _ in narrative.sections) == NARRATIVE_SECTIONS
    text = narrative.text
    review = tables.observed_drop_review
    assert f"Observed drops (absolute drop above zero): {len(review)} assessed intervals" in text
    assert f"All {len(ASSORTMENT_RECONCILIATION_CHECKS)} reconciliation checks passed" in text
    assert "unusual-drop policy is unavailable" in text and "not causation" in text
    assert "zero denominator" in text.lower() or "Zero denominators" in text
    assert "short observation window" in text and "not confirmation of what the supplier offered" in text
    assert "Timeline persistence is not approved" in text
    for column in ("car_name", "SYNTH", "$", "job_id", "/"):
        assert column not in text
    assert not FORBIDDEN_WORDS.search(text) and "unusual" not in unusual_free(text)
    assert build_assortment_narrative(tables) == narrative                       # deterministic


@pytest.mark.parametrize(("spec", "phrases"), [
    ({0: "AB", 1: "AB", 2: "ABC"}, ("No assessed interval shows an observed drop.",)),
    ({0: "", 1: "", 2: ""}, ("Eligible captures with no returned product: 3.", "Zero denominators: 2 for retention",
                             "No assessed interval had an assortment change.")),
])
def test_narrative_edge_cases_are_described_accurately(spec, phrases) -> None:  # type: ignore[no-untyped-def]
    text = build_assortment_narrative(tables_of(sets(spec), [timeline([0, 1, 2])])).text
    for phrase in phrases:
        assert phrase in text, phrase
    assert "0 had a price change" in text and "not causation" in text
    assert not FORBIDDEN_WORDS.search(text) and "unusual" not in unusual_free(text)


def test_narrative_rejects_unvalidated_input() -> None:
    with pytest.raises(TypeError):
        build_assortment_narrative(object())


# ============================================================================ reconciliation


def test_every_reconciliation_check_passes_and_corruption_fails(tables) -> None:  # type: ignore[no-untyped-def]
    recon = tables.reconciliation_summary
    assert (recon["expected"] == recon["observed"]).all()
    corrupted = recon.copy()
    corrupted.loc[0, "observed"] = corrupted.loc[0, "observed"] + 1
    with pytest.raises(AssortmentPrivacyError, match="rule reconciliation_status"):
        validate_assortment_table("reconciliation_summary", corrupted, approved_locations=tables.approved_locations)
    failed = corrupted.copy()
    failed.loc[0, "status"] = "failed"
    with pytest.raises(AssortmentReconciliationError):
        dataclasses.replace(tables, reconciliation_summary=failed)
    with pytest.raises(AssortmentReconciliationError):
        dataclasses.replace(tables, reconciliation_summary=recon.iloc[1:].reset_index(drop=True))
    mutated = tables.location_summary.copy()
    mutated.loc[0, "scheduled_captures"] = mutated.loc[0, "scheduled_captures"] + 1
    with pytest.raises(AssortmentPrivacyError):
        dataclasses.replace(tables, location_summary=mutated)
    with pytest.raises(AssortmentReconciliationError):
        dataclasses.replace(tables, approved_locations=())


def test_reconciliation_detects_engine_disagreement(presentation, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real = ap._location_summary

    def shifted(result, rows):  # type: ignore[no-untyped-def]
        frame = real(result, rows)
        frame.loc[0, "total_additions"] = frame.loc[0, "total_additions"] + 1
        return frame

    monkeypatch.setattr(ap, "_location_summary", shifted)
    with pytest.raises(AssortmentReconciliationError):
        build_assortment_presentation_tables(presentation.assortment)


def test_reports_and_results_enforce_their_states() -> None:
    with pytest.raises(AssortmentReconciliationError):
        AssortmentPresentationReport(AssortmentPresentationStatus.BLOCKED)
    with pytest.raises(AssortmentReconciliationError):
        AssortmentPresentationReport(AssortmentPresentationStatus.COMPLETED)
    with pytest.raises(AssortmentReconciliationError):
        AssortmentPresentationReport(AssortmentPresentationStatus.BLOCKED, blockers=(PB.EVIDENCE_MISMATCH,),
                                     reconciled=True)
    blocked = AssortmentPresentationReport(AssortmentPresentationStatus.BLOCKED, blockers=(PB.EVIDENCE_MISMATCH,))
    with pytest.raises(AssortmentReconciliationError):
        AssortmentPresentationResult(blocked, tables=tables_of(sets({0: "A", 1: "A"}), [timeline([0, 1])]))
    with pytest.raises(TypeError):
        AssortmentPresentationResult("blocked")                                  # type: ignore[arg-type]


# ============================================================================ persistence gate


@pytest.mark.parametrize("target", ["reports/visible_assortment", "data/raw", "data/processed", ".", "tmp_output"])
def test_any_output_directory_fails_closed_without_touching_disk(world, target, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis import pricing_pipeline

    def forbidden(raw_dir=None):  # type: ignore[no-untyped-def]
        raise AssertionError("the pipeline must not run when an output is refused")

    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline", forbidden)
    repo_before = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if ".git" not in p.parts
                         and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)
    for output in (ROOT / target, tmp_path / target, str(tmp_path / target)):
        for result in (run_assortment_presentation(output_dir=output),
                       assortment_presentation_from_pipeline(pipeline_result(world), output_dir=output)):
            assert result.report.blockers == (PB.PERSISTENCE_NOT_APPROVED,) and result.tables is None
    assert not (tmp_path / target).exists() or target == "."
    assert list(tmp_path.iterdir()) == []
    assert sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if ".git" not in p.parts
                  and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts) == repo_before
    with pytest.raises(TypeError):
        run_assortment_presentation(output_dir=3)                              # type: ignore[arg-type]


def test_no_environment_variable_enables_export(world, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for name in ("QL2_SIXT_ASSORTMENT_OUTPUT_DIR", "QL2_SIXT_PRICE_CHANGE_OUTPUT_DIR", "QL2_SIXT_ASSORTMENT_PERSIST",
                 "QL2_SIXT_PRICE_CHANGE_WRITE_DETAIL"):
        monkeypatch.setenv(name, str(tmp_path / "out") if "DIR" in name else "1")
    monkeypatch.chdir(tmp_path)
    result = assortment_presentation_from_pipeline(pipeline_result(world))
    assert result.completed and os.listdir(tmp_path) == [] and not assortment_persistence_approved()
    source = (ROOT / "src" / "ql2_sixt_canada_analysis" / "assortment_presentation.py").read_text(encoding="utf-8")
    for forbidden in ("os.environ", "getenv", "to_csv", "to_parquet", "to_json", "write_text", "write_bytes",
                      "open(", "mkdir", "print("):
        assert forbidden not in source, forbidden


# ============================================================================ packaging


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
            "import ql2_sixt_canada_analysis.assortment_presentation as m\n"
            "print('ql2_sixt_canada_analysis.pricing_pipeline' in sys.modules, 'matplotlib' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_public_package_exports_are_complete() -> None:
    public = {n for n in vars(ap) if not n.startswith("_") and getattr(getattr(ap, n), "__module__", None)
              == ap.__name__}
    assert public <= set(ap.__all__) and len(ap.__all__) == len(set(ap.__all__))
    for name in ap.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(ap, name)
