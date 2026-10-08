"""Visible-assortment definition contract (data-plan Section 5, definition phase).

Synthetic sets and fabricated ``SYNTH-*`` products only; the only committed
values read are approved configuration (stream keys, roles, schedule zones and
the canonical-offer policy) through the shared synthetic pipeline world.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest
from test_price_change_events import P, pipeline_result, readiness_for, synthetic_world

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis import assortment_contract as ac
from ql2_sixt_canada_analysis.assortment_contract import (
    ASSORTMENT_BREAK_REASONS,
    ASSORTMENT_TIMELINE_COLUMNS,
    DEFAULT_ASSORTMENT_DEFINITION as D,
    DEFAULT_UNUSUAL_DROP_POLICY,
    AnomalyPolicyStatus,
    AnomalyPolicyUnavailableError,
    AssortmentBlocker as B,
    AssortmentComparison,
    AssortmentContractError,
    DefinitionStatus,
    DenominatorStatus as DS,
    UnusualDropPolicy,
    VisibleAssortmentDefinition,
    assortment_evidence_blockers,
    classify_unusual_drop,
    compare_assortment,
    distinct_product_count,
)
from ql2_sixt_canada_analysis.canonical_offers import APPROVED_PRODUCT_COLUMNS
from ql2_sixt_canada_analysis.price_change_events import (
    CANDIDATE_COLUMNS,
    EVENT_IDENTITY_COLUMNS,
    EVENT_TIMESTAMP_COLUMN,
    EVENT_UNIT_COLUMNS,
    FORBIDDEN_TIMESTAMP_SOURCES,
    CaptureInterval,
    CaptureState,
    IntervalBreak,
    LocationCaptureTimeline,
    ScheduledCapture,
)
from ql2_sixt_canada_analysis.readiness import PricingBlocker
from ql2_sixt_canada_analysis.schemas import ANALYSIS_LOCATION_STREAM_COMPARISON, CONFIDENTIAL_TECHNICAL_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "decisions" / "governance" / "visible-assortment-contract-proposal-v1-2026-10-08.md"
LOC = ("synth-city", "SYNTH Downtown")


def product(name: str) -> tuple:
    return (name, "SYNTH Compact", "SYNTH Automatic", "5", "2")


def products(*names: str) -> frozenset:
    return frozenset(product(n) for n in names)


def interval(h: int = 1) -> CaptureInterval:
    return CaptureInterval(LOC, P(h - 1), P(h))


def changed(**changes) -> dict:  # type: ignore[no-untyped-def]
    fields = {f.name: getattr(D, f.name) for f in dataclasses.fields(D)}
    fields.update(changes)
    return fields


# ============================================================================ the contract itself


def test_the_default_definition_is_immutable_and_reuses_established_constants() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        D.product_columns = ("car_name",)  # type: ignore[misc]
    with pytest.raises(TypeError):
        D.statuses["population"] = DefinitionStatus.PROPOSED  # type: ignore[index]
    assert D.product_columns == tuple(APPROVED_PRODUCT_COLUMNS)
    assert D.capture_column == EVENT_TIMESTAMP_COLUMN == "scheduled_capture_period"
    assert D.location_columns == ("canonical_city", "canonical_location") == EVENT_IDENTITY_COLUMNS[:2]
    assert D.unit_columns == tuple(EVENT_UNIT_COLUMNS)
    assert set(D.product_columns) <= set(ANALYSIS_LOCATION_STREAM_COMPARISON.product_columns)
    assert set(D.context_columns) == set(ANALYSIS_LOCATION_STREAM_COMPARISON.product_columns) - set(D.product_columns)
    assert D.population_source == "pricing_eligible_canonical_offers"
    assert D.interval_step_seconds == 3600
    assert D.set_key_columns == (*D.location_columns, D.capture_column, *D.context_columns)


def test_column_groups_are_nonempty_unique_disjoint_canonical_offer_columns() -> None:
    groups = [D.location_columns, (D.capture_column,), D.product_columns, D.context_columns, D.unit_columns]
    flat = [c for g in groups for c in g]
    assert all(groups[:3]) and len(flat) == len(set(flat))
    assert set(flat) <= set(EVENT_IDENTITY_COLUMNS) | {EVENT_TIMESTAMP_COLUMN}


def test_product_identity_excludes_price_identifiers_timestamps_units_context_and_provenance() -> None:
    forbidden = {"price_cents", "price_num", "price_per_day", *CONFIDENTIAL_TECHNICAL_COLUMNS,
                 *FORBIDDEN_TIMESTAMP_SOURCES, EVENT_TIMESTAMP_COLUMN, "source_location_labels", "provenance",
                 "observation_count", "currency", "price_basis", "pickup_date", "return_date", "row_index"}
    assert not set(D.product_columns) & forbidden
    assert {*FORBIDDEN_TIMESTAMP_SOURCES, *CONFIDENTIAL_TECHNICAL_COLUMNS, "price_cents"} <= D.prohibited_columns


@pytest.mark.parametrize(("changes", "message"), [
    (dict(product_columns=()), "product_columns must not be empty"),
    (dict(product_columns=("car_name", "car_name")), "repeat"),
    (dict(product_columns=(*APPROVED_PRODUCT_COLUMNS, "price_cents")), "prohibited"),
    (dict(product_columns=(*APPROVED_PRODUCT_COLUMNS, "pickup_date")), "both"),          # overlaps context
    (dict(product_columns=(*APPROVED_PRODUCT_COLUMNS, "job_id")), "not a canonical-offer column"),
    (dict(product_columns=(*APPROVED_PRODUCT_COLUMNS, "scheduled_capture_period")), "both"),
    (dict(product_columns=(*APPROVED_PRODUCT_COLUMNS, "source_location_labels")), "prohibited"),
    (dict(capture_column="scrape_date"), "not a canonical-offer column"),
    (dict(capture_column="price_cents"), "prohibited"),
    (dict(prohibited_columns=frozenset({"price_cents"})), "cover"),
    (dict(interval_step_seconds=7200), "one-hour"),
    (dict(break_reasons=("missing_capture",)), "every event-contract interval break"),
    (dict(price_change_outcomes=("increase", "decrease", "appeared")), "only increase and decrease"),
    (dict(timeline_columns=(*ASSORTMENT_TIMELINE_COLUMNS, "car_name")), "aggregate-only"),
    (dict(timeline_columns=(*ASSORTMENT_TIMELINE_COLUMNS, "previous_price_cents")), "aggregate-only"),
    (dict(timeline_columns=(*ASSORTMENT_TIMELINE_COLUMNS, "outcome")), "aggregate-only"),
    (dict(timeline_columns=ASSORTMENT_TIMELINE_COLUMNS[1:]), "grain"),
    (dict(population_source="raw_cars_rows"), "pricing-eligible canonical offers"),
    (dict(statuses={}), "DefinitionStatus"),
])
def test_malformed_definitions_fail_closed(changes, message) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(AssortmentContractError, match=message):
        VisibleAssortmentDefinition(**changed(**changes))


def test_timeline_schema_is_fixed_ordered_unique_and_aggregate_safe() -> None:
    timeline = ASSORTMENT_TIMELINE_COLUMNS
    assert len(timeline) == len(set(timeline)) and timeline[:3] == D.set_key_columns[:3]
    assert not set(timeline) & (D.prohibited_columns | set(D.product_columns) | set(D.context_columns)
                                | set(D.unit_columns))
    assert not set(timeline) & (set(CANDIDATE_COLUMNS) - {"canonical_city", "canonical_location",
                                                          "previous_scheduled_capture_period"})
    for required in ("returned_product_count", "retention", "retention_denominator_status", "jaccard_similarity",
                     "jaccard_denominator_status", "absolute_drop", "drop_rate", "interval_break_reason",
                     "price_increase_count", "price_decrease_count", "assortment_price_coincidence",
                     "anomaly_policy_status", "unusual_drop", "assessability_status", "capture_state"):
        assert required in timeline


def test_authority_statuses_are_honest() -> None:
    assert D.proposed_definitions == ("multiple_rental_context_stratification", "timeline_persistence",
                                      "unusual_drop_policy")
    for derived in ("population", "location", "capture", "consecutive_interval", "product_identity"):
        assert D.statuses[derived] is DefinitionStatus.DERIVED
    assert set(ASSORTMENT_BREAK_REASONS) == {*(b.value for b in IntervalBreak), "rental_context_changed"}


# ============================================================================ set formulas


def test_returned_products_count_distinct_identities_not_offer_rows() -> None:
    offers = [product("SYNTH Car A"), product("SYNTH Car A"), product("SYNTH Car B")]   # two prices of car A
    assert distinct_product_count(offers) == 2
    assert distinct_product_count([]) == 0                                     # an eligible empty capture
    for bad in ([("SYNTH Car A", None, "x", "5", "2")], [("SYNTH Car A ", "c", "t", "5", "2")],
                [("SYNTH Car A", "c", "t", "5")], [("SYNTH Car A", "", "t", "5", "2")],
                [("SYNTH Car A", float("nan"), "t", "5", "2")]):
        with pytest.raises(AssortmentContractError):
            distinct_product_count(bad)


def test_retained_additions_and_removals_partition_the_union() -> None:
    previous, current = products("A", "B", "C"), products("B", "C", "D", "E")
    c = compare_assortment(previous, current, interval=interval())
    assert (c.retained_count, c.addition_count, c.removal_count) == (2, 2, 1)
    assert c.retained_count + c.addition_count + c.removal_count == len(previous | current)
    assert (c.net_change, c.absolute_drop, c.drop_rate) == (1, 0, 0.0)
    assert c.assessable and c.break_reason is None


def test_retention_is_directional_and_jaccard_symmetric() -> None:
    a, b = products("A", "B", "C", "D"), products("A", "B")
    forward, backward = compare_assortment(a, b, interval=interval()), compare_assortment(b, a, interval=interval())
    assert (forward.retention, backward.retention) == (0.5, 1.0)              # denominator = previous set
    assert forward.jaccard == backward.jaccard == 0.5
    assert (forward.absolute_drop, forward.drop_rate, forward.net_change) == (2, 0.5, -2)
    assert (backward.absolute_drop, backward.drop_rate) == (0, 0.0)
    for value in (forward.retention, forward.jaccard, forward.drop_rate, backward.retention, backward.jaccard):
        assert 0.0 <= value <= 1.0


def test_empty_sets_use_explicit_zero_denominators_never_full_retention() -> None:
    first = compare_assortment(frozenset(), products("A"), interval=interval())
    assert first.retention is None and first.retention_status is DS.ZERO_DENOMINATOR
    assert first.drop_rate is None and first.drop_rate_status is DS.ZERO_DENOMINATOR
    assert first.jaccard == 0.0 and first.jaccard_status is DS.DEFINED and first.addition_count == 1
    both = compare_assortment(frozenset(), frozenset(), interval=interval())
    assert both.jaccard is None and both.jaccard_status is DS.ZERO_DENOMINATOR and both.retention is None
    emptied = compare_assortment(products("A", "B"), frozenset(), interval=interval())
    assert (emptied.retention, emptied.jaccard, emptied.drop_rate, emptied.removal_count) == (0.0, 0.0, 1.0, 2)


@pytest.mark.parametrize("reason", [b.value for b in IntervalBreak] + ["rental_context_changed", None])
def test_breaks_and_seed_captures_are_never_compared(reason) -> None:  # type: ignore[no-untyped-def]
    previous = None if reason is None else products("A", "B")
    c = compare_assortment(previous, products("C"), interval=None, break_reason=reason)
    assert not c.assessable and c.break_reason == reason
    assert (c.addition_count, c.removal_count, c.retention, c.jaccard, c.drop_rate) == (None,) * 5
    assert c.retention_status is DS.NOT_ASSESSABLE
    with pytest.raises(AssortmentContractError):
        AssortmentComparison(False, reason, 2, 1, addition_count=1)            # no additions across a break
    with pytest.raises(AssortmentContractError):
        compare_assortment(products("A"), products("B"), interval=None, break_reason="SYNTH-unknown")
    with pytest.raises(AssortmentContractError):
        compare_assortment(products("A"), products("B"), interval=interval(), break_reason="missing_capture")


def test_missing_captures_and_governed_exclusions_create_breaks_and_empty_captures_stay_visible() -> None:
    def timeline(states):  # type: ignore[no-untyped-def]
        return LocationCaptureTimeline(LOC, tuple(ScheduledCapture(P(h), s, (LOC,)) for h, s in enumerate(states)))

    excluded = timeline([CaptureState.ELIGIBLE, CaptureState.GOVERNED_EXCLUSION, CaptureState.ELIGIBLE])
    missing = timeline([CaptureState.ELIGIBLE, CaptureState.MISSING_CAPTURE, CaptureState.ELIGIBLE])
    assert excluded.intervals == () and dict(excluded.break_counts) == {IntervalBreak.GOVERNED_EXCLUSION.value: 2}
    assert missing.intervals == () and dict(missing.break_counts) == {IntervalBreak.MISSING_CAPTURE.value: 2}
    whole = timeline([CaptureState.ELIGIBLE] * 3)
    c = compare_assortment(products("A"), frozenset(), interval=whole.intervals[0])   # eligible zero-offer capture
    assert c.assessable and c.current_count == 0 and c.removal_count == 1
    with pytest.raises(Exception):
        CaptureInterval(LOC, P(0), P(2))                                       # never across a two-hour gap


def test_unusual_drop_classification_fails_closed_without_approved_authority() -> None:
    c = compare_assortment(products("A", "B", "C", "D"), products("A"), interval=interval())
    assert DEFAULT_UNUSUAL_DROP_POLICY.status is AnomalyPolicyStatus.UNAVAILABLE
    for policy in (DEFAULT_UNUSUAL_DROP_POLICY, UnusualDropPolicy(AnomalyPolicyStatus.PROPOSED, description="SYNTH")):
        with pytest.raises(AnomalyPolicyUnavailableError):
            classify_unusual_drop(c, policy)
    with pytest.raises(AssortmentContractError):
        UnusualDropPolicy(AnomalyPolicyStatus.APPROVED)                       # approval needs recorded authority
    with pytest.raises(AnomalyPolicyUnavailableError):                        # approved, but no executable rule
        classify_unusual_drop(c, UnusualDropPolicy(AnomalyPolicyStatus.APPROVED, "SYNTH-record", "SYNTH-ref"))


# ============================================================================ evidence gate


def test_the_evidence_gate_passes_on_a_ready_synthetic_pipeline_world() -> None:
    w = synthetic_world(products={})
    assert assortment_evidence_blockers(pipeline_result(w)) == ()


def test_the_evidence_gate_fails_closed() -> None:
    w = synthetic_world()
    other = synthetic_world(hours=2)
    run = pipeline_result(w)
    not_ready = dataclasses.replace(run, pricing=readiness_for(w["cars"], w["scheduled"], w["canonical_offers"],
                                                               (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE,)))
    assert B.PRICING_NOT_READY in assortment_evidence_blockers(not_ready)
    assert B.EVIDENCE_BINDING_MISMATCH in assortment_evidence_blockers(dataclasses.replace(
        run, population=other["population"]))
    assert B.PRICING_NOT_READY in assortment_evidence_blockers(dataclasses.replace(
        run, canonical_offers=other["canonical_offers"]))
    missing = dataclasses.replace(run, scheduled=dataclasses.replace(w["scheduled"], capture_exclusions=None))
    assert B.SCHEDULE_EVIDENCE_INVALID in assortment_evidence_blockers(missing)
    assert B.LOCATION_AUTHORITY_UNAVAILABLE in assortment_evidence_blockers(dataclasses.replace(
        run, location_authority=None))
    assert B.CANONICAL_OFFERS_NOT_READY in assortment_evidence_blockers(dataclasses.replace(run, canonical_offers=None))
    with pytest.raises(TypeError):
        assortment_evidence_blockers(object())


def test_more_than_one_rental_context_per_location_capture_fails_closed() -> None:
    from test_price_change_events import TOR_DOWN

    w = synthetic_world(products={(TOR_DOWN, 1): [("SYNTH Car A", 50.0)]})
    offers = w["canonical_offers"]
    frame = offers.offers.copy()
    i = frame.index[frame["car_name"] == "SYNTH Car A"][0]
    import datetime as dt

    frame.at[i, "return_date"] = dt.date(2030, 4, 9)                           # a second search context
    tampered = dataclasses.replace(offers, offers=frame)
    run = dataclasses.replace(pipeline_result(w), canonical_offers=tampered,
                              pricing=readiness_for(w["cars"], w["scheduled"], tampered))
    assert B.MULTIPLE_RENTAL_CONTEXTS in assortment_evidence_blockers(run)


# ============================================================================ documentation, packaging, import


def test_the_governance_proposal_and_readme_reference_the_contract() -> None:
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    for phrase in ("PROPOSED", "Not approved", "retention", "Jaccard", "absolute_drop", "zero_denominator",
                   "rental search context", "unusual", "coincid", "ASSORTMENT_TIMELINE_COLUMNS",
                   "Count returned products by location and capture", "Produce an assortment timeline"):
        assert phrase in text, phrase
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert DOC.name in readme and "assortment_contract.py" in readme
    assert "Visible assortment stability and monitoring work are not started here" not in readme


def test_importing_the_contract_performs_no_io_or_pipeline_execution() -> None:
    code = ("import builtins, io, os, sys\n"
            f"ROOT = {str(ROOT)!r}\n"
            "real = builtins.open\n"
            "def guarded(file, mode='r', *a, **k):\n"
            "    path = os.path.abspath(os.fspath(file)) if isinstance(file, (str, bytes, os.PathLike)) else ''\n"
            "    if any(c in mode for c in 'wax+') or str(path).startswith(ROOT):\n"
            "        raise AssertionError('I/O during import')\n"
            "    return real(file, mode, *a, **k)\n"
            "builtins.open = io.open = guarded\n"
            "import ql2_sixt_canada_analysis.assortment_contract as m\n"
            "print('ql2_sixt_canada_analysis.pricing_pipeline' in sys.modules, 'matplotlib' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_public_package_exports_are_complete() -> None:
    public = {n for n in vars(ac) if not n.startswith("_") and getattr(getattr(ac, n), "__module__", None)
              == ac.__name__}
    assert public <= set(ac.__all__) and len(ac.__all__) == len(set(ac.__all__))
    for name in ac.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(ac, name)
