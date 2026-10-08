"""Visible assortment: the calculation engine of data-plan Section 5 (implements the Prompt 1 contract).

The definitions are fixed by :mod:`ql2_sixt_canada_analysis.assortment_contract`
(:data:`~ql2_sixt_canada_analysis.assortment_contract.DEFAULT_ASSORTMENT_DEFINITION`)
and ``docs/decisions/governance/visible-assortment-contract-proposal-v1-2026-10-08.md``;
this module computes them and redefines nothing.

Entry points: :func:`run_visible_assortment` (runs the pricing pipeline once,
then every gate), :func:`visible_assortment_from_pipeline` (one existing
:class:`~ql2_sixt_canada_analysis.pricing_pipeline.PricingPipelineResult`),
:func:`assess_visible_assortment` (the gated assessment of supplied evidence)
and :func:`calculate_visible_assortment` (the pure engine).

Population and grid
-------------------
Product sets are built only from the pricing-eligible canonical offers
(``CanonicalOfferReport.offers_for``), one set per canonical location and
scheduled capture. The capture grid is the validated price-change result's
schedule-derived :class:`~ql2_sixt_canada_analysis.price_change_events.LocationCaptureTimeline`
objects, never the observed offers: an eligible capture without offers is an
empty set, and every scheduled capture of every approved canonical location
(eligible, governed exclusion or missing) has exactly one timeline row.

Product and context
-------------------
A product is the exact, type-aware value of the approved product identity
(``car_name, car_type, transmission, seats, bags``): no trimming, recasing,
fuzzy matching, imputation or sentinel. Price, currency, price basis,
identifiers, offer position, raw timestamps, source labels and provenance never
enter. The rental dates are the search context of one set; exactly one context
per location capture is required, and a context change across a schedule
interval is the typed break ``rental_context_changed``.

Calculations (``P`` previous set, ``C`` current set, valid intervals only)
--------------------------------------------------------------------------
``returned_product_count = |C|``; ``retained = |P & C|``; additions ``|C - P|``;
removals ``|P - C|``; ``retention = |P & C| / |P|`` (``zero_denominator`` when
``|P| = 0``); ``jaccard_similarity = |P & C| / |P | C|`` (``zero_denominator``
for an empty union); ``net_change = |C| - |P|``;
``absolute_drop = max(|P| - |C|, 0)``; ``drop_rate = absolute_drop / |P|``
(``zero_denominator`` when ``|P| = 0``). Nothing is rounded. Nothing is compared
across a governed exclusion, missing capture, non-hourly adjacency, change of
contributing source streams or rental-context change; the first capture of a
location is a seed without changes.

Unusual drops
-------------
No approved unusual-drop policy exists. With the default policy every row
reports ``anomaly_policy_status = unavailable`` and ``unusual_drop`` is null
(never ``False``). Only an explicitly supplied, approved and executable
:class:`~ql2_sixt_canada_analysis.assortment_contract.UnusualDropPolicy` fills
``unusual_drop`` on assessed rows. No threshold is estimated from the data, and
no drop is called a collection failure or a supplier withdrawal.

Price coincidence
-----------------
From the validated price-change candidates of the same evidence, never from
raw prices: per assessed interval (same canonical city, location, previous and
current scheduled period) the ``increase`` and ``decrease`` candidates are
counted. ``unchanged``, ``appeared``, ``disappeared`` and ``ambiguous`` are never
price changes. The candidates' identities, projected onto location, context
and product, must equal the union of the two endpoint sets exactly, and every
increase or decrease must belong to a retained product; any disagreement fails
closed. Price counts are candidate-level (the price-comparison identity adds
currency and price basis to the product), so one retained product may carry
several unit-specific increases or decreases and ``price_increase_count +
price_decrease_count`` may exceed ``retained_count``; assortment counts stay
distinct visible products. A coincidence is temporal association only, never
causation.

Confidentiality
---------------
The aggregate timeline holds approved canonical keys, scheduled periods,
counts, ratios and enum statuses only. The product-level membership detail
(:attr:`VisibleAssortmentResult.membership`) is proprietary: it stays in
memory, is excluded from ``repr`` and equality, and is never printed or written
by this module. Nothing here writes files. Importing this module performs no I/O.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.assortment_contract import (
    ASSORTMENT_BREAK_REASONS,
    ASSORTMENT_TIMELINE_COLUMNS,
    DEFAULT_ASSORTMENT_DEFINITION,
    DEFAULT_UNUSUAL_DROP_POLICY,
    AnomalyPolicyStatus,
    AssessabilityStatus,
    AssortmentBlocker,
    AssortmentComparison,
    AssortmentContractError,
    DenominatorStatus,
    UnusualDropPolicy,
    assortment_evidence_blockers,
    classify_unusual_drop,
    compare_assortment,
)
from ql2_sixt_canada_analysis.price_change_events import (
    EVENT_IDENTITY_COLUMNS,
    EVENT_INTERVAL_COLUMNS,
    CaptureInterval,
    CaptureState,
    IntervalBreak,
    LocationCaptureTimeline,
    PriceChangeCandidateResult,
    PriceChangeContractError,
    TerminalOutcome,
    parse_scheduled_period,
)

__all__ = [
    "MEMBERSHIP_COLUMNS",
    "RENTAL_CONTEXT_CHANGED",
    "AssortmentCaptureError",
    "AssortmentCounts",
    "AssortmentIdentityError",
    "AssortmentLocationError",
    "AssortmentPriceEvidenceError",
    "AssortmentReconciliationError",
    "AssortmentReport",
    "AssortmentStatus",
    "LocationAssortmentSummary",
    "ProductMembership",
    "RentalContextError",
    "VisibleAssortmentResult",
    "assess_visible_assortment",
    "calculate_visible_assortment",
    "run_visible_assortment",
    "validate_assortment_timeline",
    "validate_price_coincidence",
    "visible_assortment_from_pipeline",
]

Key = tuple[str, ...]
_D = DEFAULT_ASSORTMENT_DEFINITION
_LOCATION: tuple[str, ...] = _D.location_columns
_CAPTURE: str = _D.capture_column
_CONTEXT: tuple[str, ...] = _D.context_columns
_PRODUCT: tuple[str, ...] = _D.product_columns
_PREVIOUS, _CURRENT = EVENT_INTERVAL_COLUMNS
#: The assortment-only interval break (a change of rental search context between two eligible captures).
RENTAL_CONTEXT_CHANGED = ASSORTMENT_BREAK_REASONS[-1]
_PRICE_CHANGES = frozenset(_D.price_change_outcomes)

#: Columns of the proprietary in-memory membership detail: one row per product per assessed interval.
MEMBERSHIP_COLUMNS: tuple[str, ...] = (*_LOCATION, *_CONTEXT, *_PRODUCT, _PREVIOUS, _CURRENT, "membership")
_MEMBERSHIP_KEY = MEMBERSHIP_COLUMNS[:-1]

_INT = ("contributing_stream_count",)
_NULLABLE_INT = ("returned_product_count", "previous_product_count", "retained_count", "addition_count",
                 "removal_count", "net_change", "absolute_drop", "price_increase_count", "price_decrease_count")
_NULLABLE_FLOAT = ("retention", "jaccard_similarity", "drop_rate")
_NULLABLE_BOOL = ("assortment_change", "price_change", "assortment_price_coincidence",
                  "falling_assortment_with_price_increase", "unusual_drop")
_INTERVAL_FIELDS = (_PREVIOUS, "previous_product_count", "retained_count", "addition_count", "removal_count",
                    "retention", "jaccard_similarity", "net_change", "absolute_drop", "drop_rate",
                    "price_increase_count", "price_decrease_count", "assortment_change", "price_change",
                    "assortment_price_coincidence", "falling_assortment_with_price_increase")
if {*_INT, *_NULLABLE_INT, *_NULLABLE_FLOAT, *_NULLABLE_BOOL, *_INTERVAL_FIELDS} - set(
        ASSORTMENT_TIMELINE_COLUMNS):                                           # pragma: no cover - import guard
    raise AssortmentContractError("the engine names a column outside the Prompt 1 timeline schema")


class AssortmentIdentityError(AssortmentContractError):
    """A product, location or context value is missing or malformed (never repaired or imputed)."""


class RentalContextError(AssortmentContractError):
    """One location capture holds more than one rental search context (stratification is not approved)."""


class AssortmentLocationError(AssortmentContractError):
    """An offer or timeline names a canonical location outside the approved configuration."""


class AssortmentCaptureError(AssortmentContractError):
    """An offer lies outside every eligible scheduled capture, or the capture grid is malformed."""


class AssortmentPriceEvidenceError(AssortmentContractError):
    """The price-change evidence is missing, stale, duplicated or disagrees with the product sets."""


class AssortmentReconciliationError(AssortmentContractError):
    """A timeline, membership detail or report violates the assortment accounting."""


class AssortmentStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class ProductMembership(StrEnum):
    """The terminal membership of one product in one assessed interval."""

    RETAINED = "retained"
    ADDED = "added"
    REMOVED = "removed"


# ------------------------------------------------------------------ aggregate accounting


def _count(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise AssortmentReconciliationError(f"{name} must be a non-negative int")


@dataclass(frozen=True, slots=True)
class AssortmentCounts:
    """Summable capture and interval accounting (counts only; no identities, prices or periods)."""

    scheduled_captures: int = 0
    eligible_captures: int = 0
    excluded_captures: int = 0
    missing_captures: int = 0
    #: Eligible captures with no product (observed empty sets, never presumed withdrawals).
    empty_captures: int = 0
    seed_captures: int = 0
    assessed_intervals: int = 0
    #: Eligible captures preceded by a typed break (no comparison).
    break_captures: int = 0
    #: Sum of ``returned_product_count`` over every eligible capture.
    returned_products: int = 0
    #: Sums over assessed intervals.
    previous_products: int = 0
    current_products: int = 0
    retained: int = 0
    additions: int = 0
    removals: int = 0
    price_increases: int = 0
    price_decreases: int = 0
    assortment_change_intervals: int = 0
    price_change_intervals: int = 0
    coincident_intervals: int = 0
    drop_intervals: int = 0
    falling_with_increase_intervals: int = 0
    retention_zero_denominator: int = 0
    jaccard_zero_denominator: int = 0

    def __post_init__(self) -> None:
        for name in self.__slots__:
            _count(getattr(self, name), name)
        rules = (
            (self.scheduled_captures == self.eligible_captures + self.excluded_captures + self.missing_captures,
             "every scheduled capture is eligible, excluded or missing"),
            (self.eligible_captures == self.seed_captures + self.assessed_intervals + self.break_captures,
             "every eligible capture is a seed, assessed or follows a break"),
            (self.empty_captures <= self.eligible_captures, "only eligible captures hold a set"),
            (self.current_products <= self.returned_products, "assessed counts are part of the returned counts"),
            (self.retained + self.removals == self.previous_products, "retained plus removed is the previous count"),
            (self.retained + self.additions == self.current_products, "retained plus added is the current count"),
            (self.drop_intervals <= self.assortment_change_intervals <= self.assessed_intervals,
             "a drop is an assortment change of an assessed interval"),
            (self.price_change_intervals <= self.assessed_intervals, "price changes are counted on assessed intervals"),
            (self.price_change_intervals <= self.price_increases + self.price_decreases,
             "a price-change interval holds an increase or decrease"),
            (self.coincident_intervals <= min(self.assortment_change_intervals, self.price_change_intervals),
             "a coincidence needs both an assortment and a price change"),
            (self.falling_with_increase_intervals <= min(self.drop_intervals, self.price_change_intervals),
             "a falling assortment with a price increase needs both"),
            (self.jaccard_zero_denominator <= self.retention_zero_denominator <= self.assessed_intervals,
             "an empty union implies an empty previous set"),
        )
        for ok, message in rules:
            if not ok:
                raise AssortmentReconciliationError(message)

    def __add__(self, other: AssortmentCounts) -> AssortmentCounts:
        return AssortmentCounts(*(getattr(self, n) + getattr(other, n) for n in self.__slots__))

    @property
    def price_changes(self) -> int:
        """``price_increases + price_decreases`` over assessed intervals."""
        return self.price_increases + self.price_decreases


@dataclass(frozen=True, slots=True)
class LocationAssortmentSummary:
    """One approved canonical location: its typed breaks and its counts."""

    canonical_location: Key
    breaks: tuple[tuple[str, int], ...]
    counts: AssortmentCounts

    def __post_init__(self) -> None:
        key = self.canonical_location
        if not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(v, str) and v for v in key):
            raise AssortmentReconciliationError("a canonical location is an exact (city, location) key")
        if not isinstance(self.counts, AssortmentCounts):
            raise AssortmentReconciliationError("counts must be AssortmentCounts")
        reasons = [r for r, _ in self.breaks]
        if (any(r not in ASSORTMENT_BREAK_REASONS for r in reasons) or len(set(reasons)) != len(reasons)
                or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for _, n in self.breaks)
                or reasons != [r for r in ASSORTMENT_BREAK_REASONS if r in reasons]):
            raise AssortmentReconciliationError("breaks are distinct typed positive counts in contract order")
        c = self.counts
        if c.assessed_intervals + sum(n for _, n in self.breaks) != max(c.scheduled_captures - 1, 0):
            raise AssortmentReconciliationError("every adjacent scheduled pair is an interval or a break")
        if c.seed_captures > 1:
            raise AssortmentReconciliationError("only the first scheduled capture of a location is a seed")

    @property
    def price_changes(self) -> int:
        return self.counts.price_changes


@dataclass(frozen=True, slots=True)
class AssortmentReport:
    """Aggregate, print-safe result: status, blockers, approved location keys and counts only."""

    status: AssortmentStatus
    blockers: tuple[AssortmentBlocker, ...] = ()
    #: Upstream blocker categories (pricing readiness or price-change engine), as plain values.
    upstream_blockers: tuple[str, ...] = ()
    offers_assessed: int = 0
    locations: tuple[LocationAssortmentSummary, ...] = ()
    overall: AssortmentCounts | None = None
    anomaly_policy_status: AnomalyPolicyStatus = AnomalyPolicyStatus.UNAVAILABLE
    #: Assessed intervals the approved policy classified as unusual (``None`` unless the policy is executable).
    unusual_drop_intervals: int | None = None

    def __post_init__(self) -> None:
        _count(self.offers_assessed, "offers_assessed")
        if not isinstance(self.status, AssortmentStatus):
            raise AssortmentReconciliationError("status must be an AssortmentStatus")
        if not isinstance(self.anomaly_policy_status, AnomalyPolicyStatus):
            raise AssortmentReconciliationError("anomaly_policy_status must be an AnomalyPolicyStatus")
        if not all(isinstance(b, str) for b in self.upstream_blockers):
            raise AssortmentReconciliationError("upstream blockers are plain category values")
        if self.status is AssortmentStatus.BLOCKED:
            if not self.blockers or self.locations or self.overall is not None \
                    or self.unusual_drop_intervals is not None:
                raise AssortmentReconciliationError("a blocked report has blockers and no result")
            if not all(isinstance(b, AssortmentBlocker) for b in self.blockers):
                raise AssortmentReconciliationError("blockers must be AssortmentBlocker values")
            return
        if self.blockers or self.upstream_blockers or not isinstance(self.overall, AssortmentCounts):
            raise AssortmentReconciliationError("a completed report has no blockers and overall counts")
        keys = [s.canonical_location for s in self.locations]
        if not keys or len(set(keys)) != len(keys):
            raise AssortmentReconciliationError("one summary per approved canonical location")
        if sum((s.counts for s in self.locations), AssortmentCounts()) != self.overall:
            raise AssortmentReconciliationError("location counts must sum to the overall counts")
        if self.anomaly_policy_status is AnomalyPolicyStatus.APPROVED:
            _count(self.unusual_drop_intervals, "unusual_drop_intervals")
            if self.unusual_drop_intervals > self.overall.drop_intervals:
                raise AssortmentReconciliationError("only an observed drop can be classified as unusual")
        elif self.unusual_drop_intervals is not None:
            raise AssortmentReconciliationError("without an approved policy no drop is classified")

    @property
    def completed(self) -> bool:
        return self.status is AssortmentStatus.COMPLETED

    @property
    def approved_locations(self) -> tuple[Key, ...]:
        return tuple(s.canonical_location for s in self.locations)

    def location(self, key: Key) -> LocationAssortmentSummary:
        return next(s for s in self.locations if s.canonical_location == tuple(key))


# ------------------------------------------------------------------ validation of the published frames


def _na(value: object) -> bool:
    return value is None or value is pd.NA or value is pd.NaT or (isinstance(value, float) and math.isnan(value))


def _int(value: object, *, signed: bool = False) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
            and (signed or value >= 0))


def _ratio(value: object, numerator: int, denominator: int) -> bool:
    return (isinstance(value, (float, np.floating)) and math.isfinite(value) and 0.0 <= value <= 1.0
            and float(value) == numerator / denominator)


def _flag(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _row_counts(row: dict, assessed: bool) -> dict[str, int]:
    """The counts one validated timeline row contributes (assessed rows contribute interval sums)."""
    eligible = row["capture_state"] == CaptureState.ELIGIBLE.value
    status = row["assessability_status"]
    out = {"scheduled_captures": 1, "eligible_captures": int(eligible),
           "excluded_captures": int(row["capture_state"] == CaptureState.GOVERNED_EXCLUSION.value),
           "missing_captures": int(row["capture_state"] == CaptureState.MISSING_CAPTURE.value),
           "empty_captures": int(eligible and row["returned_product_count"] == 0),
           "seed_captures": int(status == AssessabilityStatus.SEED_CAPTURE.value),
           "assessed_intervals": int(assessed),
           "break_captures": int(status == AssessabilityStatus.INTERVAL_BREAK.value),
           "returned_products": int(row["returned_product_count"]) if eligible else 0}
    if assessed:
        out.update(
            previous_products=int(row["previous_product_count"]), current_products=int(row["returned_product_count"]),
            retained=int(row["retained_count"]), additions=int(row["addition_count"]),
            removals=int(row["removal_count"]), price_increases=int(row["price_increase_count"]),
            price_decreases=int(row["price_decrease_count"]),
            assortment_change_intervals=int(bool(row["assortment_change"])),
            price_change_intervals=int(bool(row["price_change"])),
            coincident_intervals=int(bool(row["assortment_price_coincidence"])),
            drop_intervals=int(row["absolute_drop"] > 0),
            falling_with_increase_intervals=int(bool(row["falling_assortment_with_price_increase"])),
            retention_zero_denominator=int(row["retention_denominator_status"]
                                           == DenominatorStatus.ZERO_DENOMINATOR.value),
            jaccard_zero_denominator=int(row["jaccard_denominator_status"]
                                         == DenominatorStatus.ZERO_DENOMINATOR.value))
    return out


def _check_assessed(row: dict) -> None:
    D = DenominatorStatus
    p, c = row["previous_product_count"], row["returned_product_count"]
    r, a, m = row["retained_count"], row["addition_count"], row["removal_count"]
    inc, dec = row["price_increase_count"], row["price_decrease_count"]
    net, drop = row["net_change"], row["absolute_drop"]
    if not all(_int(v) for v in (p, c, r, a, m, inc, dec, drop)) or not _int(net, signed=True):
        raise AssortmentReconciliationError("assessed counts are integers")
    union = r + a + m
    rules = (r + m == p, r + a == c, net == a - m == c - p, drop == max(-net, 0),
             _na(row["interval_break_reason"]), _flag(row["has_previous_interval"]) and bool(
                 row["has_previous_interval"]), _exact_text(row[_PREVIOUS]))
    if not all(rules):
        raise AssortmentReconciliationError("an interval's counts do not partition its endpoint sets")
    for value, status, numerator, denominator in (
            (row["retention"], row["retention_denominator_status"], r, p),
            (row["jaccard_similarity"], row["jaccard_denominator_status"], r, union),
            (row["drop_rate"], row["drop_rate_denominator_status"], drop, p)):
        if denominator:
            if status != D.DEFINED.value or not _ratio(value, numerator, denominator):
                raise AssortmentReconciliationError("a ratio differs from its exact formula")
        elif status != D.ZERO_DENOMINATOR.value or not _na(value):
            raise AssortmentReconciliationError("a zero denominator has no ratio")
    flags = {"assortment_change": a + m > 0, "price_change": inc + dec > 0,
             "assortment_price_coincidence": a + m > 0 and inc + dec > 0,
             "falling_assortment_with_price_increase": net < 0 and inc > 0}
    for name, expected in flags.items():
        if not _flag(row[name]) or bool(row[name]) is not expected:
            raise AssortmentReconciliationError(f"{name} contradicts its definition")
    # Price counts are candidate-level (price-comparison identity: product plus currency and price basis), while
    # ``retained_count`` counts visible products, so one retained product may carry several unit-specific
    # increases or decreases and ``inc + dec`` may exceed ``r``. The only aggregate bound is that a price change
    # needs a retained product; the per-candidate proof that every change projects to a retained product needs
    # candidate identities and is made by ``validate_price_coincidence``.
    if inc + dec > 0 and r == 0:
        raise AssortmentReconciliationError("only retained products can change price")


def validate_price_coincidence(timeline: pd.DataFrame, membership: pd.DataFrame,
                               price_changes: PriceChangeCandidateResult) -> None:
    """Prove the timeline's price counts from the attached candidate identities (candidate grain, not product grain).

    For every assessed row the ``increase`` and ``decrease`` candidates of the
    exact location interval must equal ``price_increase_count`` and
    ``price_decrease_count``; each such candidate, projected onto location,
    rental context and product (dropping currency and price basis), must be a
    ``retained`` membership row of that interval. Several unit-specific
    candidates may therefore project to one retained product. An interval
    without an assessed row (a rental-context break) holds no price change.

    Raises:
        AssortmentReconciliationError: Any disagreement.
    """
    R = AssortmentReconciliationError
    candidates = price_changes.candidates
    needed = (*EVENT_IDENTITY_COLUMNS, *EVENT_INTERVAL_COLUMNS, "outcome")
    if not isinstance(candidates, pd.DataFrame) or any(c not in candidates.columns for c in needed):
        raise R("the attached price-change candidates lack a contract column")
    retained = {tuple(r[:-1]) for r in membership.astype(object).itertuples(index=False, name=None)
                if r[-1] == ProductMembership.RETAINED.value}
    counted: dict[tuple, Counter] = {}
    position = {c: i for i, c in enumerate(needed)}
    member_at = [position[c] for c in _MEMBERSHIP_KEY]
    for values in candidates.loc[:, list(needed)].astype(object).itertuples(index=False, name=None):
        outcome = values[position["outcome"]]
        if outcome not in _PRICE_CHANGES:
            continue
        member = tuple(values[i] for i in member_at)
        if member not in retained:
            raise R("a price change projects to a product that is not retained in its interval")
        key = ((values[position["canonical_city"]], values[position["canonical_location"]]),
               values[position[_PREVIOUS]], values[position[_CURRENT]])
        counted.setdefault(key, Counter())[outcome] += 1
    assessed = set()
    for values in timeline.astype(object).itertuples(index=False, name=None):
        row = dict(zip(ASSORTMENT_TIMELINE_COLUMNS, values))
        if row["assessability_status"] != AssessabilityStatus.ASSESSED.value:
            continue
        key = ((row["canonical_city"], row["canonical_location"]), row[_PREVIOUS], row[_CAPTURE])
        assessed.add(key)
        tally = counted.get(key, Counter())
        if (tally[TerminalOutcome.INCREASE.value], tally[TerminalOutcome.DECREASE.value]) != (
                row["price_increase_count"], row["price_decrease_count"]):
            raise R("the price counts differ from the attached price-change candidates")
    if set(counted) - assessed:
        raise R("a price change lies outside every assessed interval")


def validate_assortment_timeline(timeline: pd.DataFrame, membership: pd.DataFrame, report: AssortmentReport,
                                 timelines: Sequence[LocationCaptureTimeline]) -> None:
    """Enforce every invariant of a completed assortment result (rows, formulas, membership and report).

    Raises:
        AssortmentReconciliationError: Any violation.
    """
    R = AssortmentReconciliationError
    if not isinstance(timeline, pd.DataFrame) or tuple(timeline.columns) != ASSORTMENT_TIMELINE_COLUMNS:
        raise R("the timeline has exactly the contract columns, in order")
    if not isinstance(membership, pd.DataFrame) or tuple(membership.columns) != MEMBERSHIP_COLUMNS:
        raise R("the membership detail has exactly the membership columns, in order")
    if not report.completed:
        raise R("only a completed report has a timeline")
    by_key = {t.canonical_location: t for t in timelines}
    order = report.approved_locations
    if len(by_key) != len(timelines) or set(by_key) != set(order):
        raise R("one capture timeline per reported canonical location")
    expected = [(key, capture, i, by_key[key]) for key in order for i, capture in enumerate(by_key[key].captures)]
    if len(timeline) != len(expected):
        raise R("one timeline row per canonical location and scheduled capture")
    if timeline.duplicated([*_LOCATION, _CAPTURE]).any():
        raise R("the timeline key must be unique")
    for column in ASSORTMENT_TIMELINE_COLUMNS:
        if column in _NULLABLE_FLOAT:
            values = timeline[column].astype(object)
            if any(not _na(v) and not (isinstance(v, (float, np.floating)) and math.isfinite(v)) for v in values):
                raise R("published ratios are finite or null")
    approved = report.anomaly_policy_status is AnomalyPolicyStatus.APPROVED
    per_location: dict[Key, Counter] = {k: Counter() for k in order}
    breaks: dict[Key, Counter] = {k: Counter() for k in order}
    assessed_rows: dict[tuple[Key, str, str], tuple[int, int, int]] = {}
    unusual = 0
    previous_row: dict | None = None
    A = AssessabilityStatus
    for values, (key, capture, i, tl) in zip(timeline.astype(object).itertuples(index=False, name=None), expected):
        row = dict(zip(ASSORTMENT_TIMELINE_COLUMNS, values))
        if (row["canonical_city"], row["canonical_location"]) != key or row[_CAPTURE] != capture.period \
                or row["capture_state"] != capture.state.value \
                or row["contributing_stream_count"] != len(capture.source_streams) \
                or not _int(row["contributing_stream_count"]):
            raise R("timeline rows follow the schedule grid in authority and period order")
        if row["anomaly_policy_status"] != report.anomaly_policy_status.value:
            raise R("every row carries the report's anomaly-policy status")
        status = row["assessability_status"]
        if status not in {s.value for s in A} or status == A.BLOCKED.value:
            raise R("a completed timeline row has a completed assessability status")
        eligible = capture.state is CaptureState.ELIGIBLE
        if eligible != _int(row["returned_product_count"]) or (not eligible and not _na(row["returned_product_count"])):
            raise R("only eligible captures return a product count")
        pair = tl.adjacent_pairs[i - 1] if i else None
        reason = row["interval_break_reason"]
        assessed = status == A.ASSESSED.value
        if i == 0:
            ok = _na(reason) and status == (A.SEED_CAPTURE.value if eligible else A.CAPTURE_NOT_ELIGIBLE.value)
        elif isinstance(pair, IntervalBreak):
            ok = reason == pair.value and status == (A.INTERVAL_BREAK.value if eligible
                                                     else A.CAPTURE_NOT_ELIGIBLE.value)
        else:
            ok = (assessed and _na(reason)) or (reason == RENTAL_CONTEXT_CHANGED and status == A.INTERVAL_BREAK.value)
        if not ok:
            raise R("the assessability status and break reason follow the schedule")
        if assessed:
            _check_assessed(row)
            if previous_row is None or row[_PREVIOUS] != previous_row[_CAPTURE] \
                    or row["previous_product_count"] != previous_row["returned_product_count"]:
                raise R("an interval compares the immediately preceding eligible capture")
            assessed_rows[(key, row[_PREVIOUS], row[_CAPTURE])] = (
                int(row["retained_count"]), int(row["addition_count"]), int(row["removal_count"]))
        else:
            if not _flag(row["has_previous_interval"]) or bool(row["has_previous_interval"]):
                raise R("only an assessed row has a previous interval")
            if any(not _na(row[c]) for c in _INTERVAL_FIELDS):
                raise R("no difference, ratio or coincidence is attributed across a break or to a seed")
            if any(row[c] != DenominatorStatus.NOT_ASSESSABLE.value for c in (
                    "retention_denominator_status", "jaccard_denominator_status", "drop_rate_denominator_status")):
                raise R("a row without an interval has not-assessable denominators")
        drop_flag = row["unusual_drop"]
        if approved and assessed:
            if not _flag(drop_flag) or (bool(drop_flag) and not row["absolute_drop"] > 0):
                raise R("an approved policy classifies every assessed interval, and only drops are unusual")
            unusual += bool(drop_flag)
        elif not _na(drop_flag):
            raise R("unusual_drop is null unless an approved policy classified an assessed interval")
        if not _na(reason):
            breaks[key][reason] += 1
        per_location[key].update(_row_counts(row, assessed))
        previous_row = row
    for summary in report.locations:
        key = summary.canonical_location
        if AssortmentCounts(**per_location[key]) != summary.counts:
            raise R("the timeline contradicts the location counts")
        if dict(summary.breaks) != dict(breaks[key]):
            raise R("the timeline contradicts the location breaks")
    if approved and unusual != report.unusual_drop_intervals:
        raise R("the timeline contradicts the unusual-drop count")
    _validate_membership(membership, assessed_rows)


def _validate_membership(membership: pd.DataFrame, assessed: dict[tuple[Key, str, str], tuple[int, int, int]]) -> None:
    R = AssortmentReconciliationError
    if membership.duplicated(list(_MEMBERSHIP_KEY)).any():
        raise R("the membership key must be unique")
    found: dict[tuple[Key, str, str], Counter] = {}
    allowed = {m.value for m in ProductMembership}
    for values in membership.astype(object).itertuples(index=False, name=None):
        row = dict(zip(MEMBERSHIP_COLUMNS, values))
        if any(_na(v) for v in values) or row["membership"] not in allowed:
            raise R("every membership row carries its full key and a terminal membership")
        found.setdefault(((row["canonical_city"], row["canonical_location"]), row[_PREVIOUS], row[_CURRENT]),
                         Counter())[row["membership"]] += 1
    if set(found) - set(assessed):
        raise R("membership rows lie only within assessed intervals")
    for key, (retained, added, removed) in assessed.items():
        got = found.get(key, Counter())
        if (got[ProductMembership.RETAINED.value], got[ProductMembership.ADDED.value],
                got[ProductMembership.REMOVED.value]) != (retained, added, removed):
            raise R("the membership detail contradicts the interval counts")


@dataclass(frozen=True)
class VisibleAssortmentResult:
    """The aggregate report, the fixed-schema timeline and the proprietary membership detail.

    State-dependent invariant, enforced on construction:

    * **blocked** - no timeline, membership, capture timelines, price-change
      evidence, binding or location authority (every field ``None``);
    * **completed** - the capture timelines and the completed, validated
      :class:`~ql2_sixt_canada_analysis.price_change_events.PriceChangeCandidateResult`
      on the same grid are mandatory, and both the aggregate accounting
      (:func:`validate_assortment_timeline`) and the candidate-level price proof
      (:func:`validate_price_coincidence`) always run. Price counts are
      candidate-level and may exceed ``retained_count`` (several unit-specific
      candidates of one retained product), so the attached candidates are the
      only proof of those counts; ``completed`` is never true without them.
    """

    report: AssortmentReport
    timeline: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    #: Proprietary product-level membership (in memory only; never printed or written).
    membership: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    timelines: tuple[LocationCaptureTimeline, ...] | None = field(default=None, repr=False, compare=False)
    #: The validated price-change result the coincidence counts were reconciled to (mandatory when completed;
    #: ``None`` only for a blocked result).
    price_changes: PriceChangeCandidateResult | None = field(default=None, repr=False, compare=False)
    binding: object = field(default=None, repr=False, compare=False)
    location_authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.report, AssortmentReport):
            raise TypeError("report must be an AssortmentReport")
        held = (self.timeline, self.membership, self.timelines, self.price_changes, self.binding,
                self.location_authority)
        if not self.report.completed:
            if any(v is not None for v in held):
                raise AssortmentReconciliationError("a blocked result holds no calculations")
            return
        if not isinstance(self.timelines, tuple) or not all(isinstance(t, LocationCaptureTimeline)
                                                             for t in self.timelines):
            raise AssortmentReconciliationError("a completed result holds its capture timelines")
        if not isinstance(self.price_changes, PriceChangeCandidateResult):
            raise AssortmentReconciliationError("a completed result holds its validated price-change evidence")
        if not self.price_changes.completed or self.price_changes.candidates is None:
            raise AssortmentReconciliationError("a completed result needs completed price-change evidence")
        if self.price_changes.timelines != self.timelines:
            raise AssortmentReconciliationError("the price-change evidence shares the assortment capture grid")
        validate_assortment_timeline(self.timeline, self.membership, self.report, self.timelines)
        validate_price_coincidence(self.timeline, self.membership, self.price_changes)

    @property
    def completed(self) -> bool:
        return self.report.completed


# ------------------------------------------------------------------ the pure engine


def _exact_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _product_sets(offers: pd.DataFrame, by_location: dict[Key, LocationCaptureTimeline]
                  ) -> dict[tuple[Key, str], tuple[tuple, frozenset]]:
    """``(location, period) -> (rental context, frozenset of product identities)`` over eligible captures."""
    if not isinstance(offers, pd.DataFrame):
        raise TypeError("offers must be a DataFrame")
    columns = (*_LOCATION, _CAPTURE, *_CONTEXT, *_PRODUCT)
    if any(c not in offers.columns for c in columns):
        raise AssortmentIdentityError("the canonical offers lack a product, context, location or capture column")
    eligible = {(k, p) for k, t in by_location.items() for p in t.periods(CaptureState.ELIGIBLE)}
    contexts: dict[tuple[Key, str], set[tuple]] = {}
    products: dict[tuple[Key, str], set[tuple]] = {}
    n_loc, n_ctx = len(_LOCATION), len(_CONTEXT)
    for values in offers.loc[:, list(columns)].astype(object).itertuples(index=False, name=None):
        location, period = values[:n_loc], values[n_loc]
        context, product = values[n_loc + 1:n_loc + 1 + n_ctx], values[n_loc + 1 + n_ctx:]
        if not all(_exact_text(v) for v in (*location, *product)):
            raise AssortmentIdentityError("location and product values are exact non-empty text")
        if any(type(v) is not dt.date for v in context):
            raise AssortmentIdentityError("rental dates are parsed dates")
        try:
            parse_scheduled_period(period)
        except PriceChangeContractError:
            raise AssortmentCaptureError("an offer has no canonical scheduled capture period") from None
        if location not in by_location:
            raise AssortmentLocationError("an offer names an unapproved canonical location")
        if (location, period) not in eligible:
            raise AssortmentCaptureError("an offer lies outside every eligible scheduled capture")
        contexts.setdefault((location, period), set()).add(context)
        products.setdefault((location, period), set()).add(product)
    out = {}
    for key, seen in contexts.items():
        if len(seen) != 1:
            raise RentalContextError("one location capture holds more than one rental search context")
        out[key] = (next(iter(seen)), frozenset(products[key]))
    return out


_IntervalKey = tuple[Key, str, str]


def _price_evidence(candidates: pd.DataFrame, intervals: set[_IntervalKey]) -> tuple[
        dict[_IntervalKey, set[tuple]], dict[_IntervalKey, Counter], dict[_IntervalKey, set[tuple]]]:
    """Per interval: projected ``(context, product)`` identities, outcome counts and price-changed members."""
    P = AssortmentPriceEvidenceError
    needed = (*EVENT_IDENTITY_COLUMNS, *EVENT_INTERVAL_COLUMNS, "outcome")
    if not isinstance(candidates, pd.DataFrame) or any(c not in candidates.columns for c in needed):
        raise P("the price-change candidates lack a contract column")
    if candidates.duplicated([*EVENT_IDENTITY_COLUMNS, *EVENT_INTERVAL_COLUMNS]).any():
        raise P("price-change candidates are unique per identity and interval")
    position = {c: i for i, c in enumerate(needed)}
    context_at = [position[c] for c in _CONTEXT]
    product_at = [position[c] for c in _PRODUCT]
    projected: dict[_IntervalKey, set[tuple]] = {}
    outcomes: dict[_IntervalKey, Counter] = {}
    changed: dict[_IntervalKey, set[tuple]] = {}
    for values in candidates.loc[:, list(needed)].astype(object).itertuples(index=False, name=None):
        key = ((values[position["canonical_city"]], values[position["canonical_location"]]),
               values[position[_PREVIOUS]], values[position[_CURRENT]])
        if key not in intervals:
            raise P("a price-change candidate lies outside every capture interval of the grid")
        try:
            outcome = TerminalOutcome(values[position["outcome"]])
        except ValueError:
            raise P("a price-change candidate has no terminal outcome") from None
        member = (tuple(values[i] for i in context_at), tuple(values[i] for i in product_at))
        projected.setdefault(key, set()).add(member)
        outcomes.setdefault(key, Counter())[outcome.value] += 1
        if outcome.value in _PRICE_CHANGES:
            changed.setdefault(key, set()).add(member)
    return projected, outcomes, changed


def calculate_visible_assortment(
        offers: pd.DataFrame, price_changes: PriceChangeCandidateResult, *,
        policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[LocationAssortmentSummary, ...]]:
    """The pure engine: product sets, interval comparisons, price coincidence and the aggregate timeline.

    ``offers`` are the pricing-eligible canonical offers; ``price_changes`` is a
    completed, validated :class:`~ql2_sixt_canada_analysis.price_change_events.PriceChangeCandidateResult`
    of the same evidence, whose capture timelines are the grid and whose report
    gives the approved canonical-location order. Inputs are never modified; the
    result is independent of row order and of every non-identity column.

    Returns:
        The timeline (:data:`~ql2_sixt_canada_analysis.assortment_contract.ASSORTMENT_TIMELINE_COLUMNS`),
        the proprietary membership detail (:data:`MEMBERSHIP_COLUMNS`) and one summary per approved location.

    Raises:
        AssortmentIdentityError: A product, context or location value is missing or malformed.
        RentalContextError: A location capture holds several rental search contexts.
        AssortmentLocationError: An offer names an unapproved canonical location.
        AssortmentCaptureError: An offer lies outside every eligible capture.
        AssortmentPriceEvidenceError: The price-change evidence is incomplete or disagrees with the sets.
        AnomalyPolicyUnavailableError: An approved policy without an executable rule was supplied.
    """
    if not isinstance(price_changes, PriceChangeCandidateResult):
        raise TypeError("price_changes must be a PriceChangeCandidateResult")
    if not isinstance(policy, UnusualDropPolicy):
        raise TypeError("policy must be an UnusualDropPolicy")
    if not price_changes.completed or not isinstance(price_changes.timelines, tuple) \
            or price_changes.candidates is None:
        raise AssortmentPriceEvidenceError("a completed price-change result is required")
    timelines = price_changes.timelines
    order = price_changes.report.approved_locations
    by_location = {t.canonical_location: t for t in timelines}
    if not order or len(by_location) != len(timelines) or set(by_location) != set(order):
        raise AssortmentCaptureError("one capture timeline per approved canonical location")
    sets = _product_sets(offers, by_location)
    interval_keys = {(t.canonical_location, i.previous_period, i.current_period) for t in timelines
                     for i in t.intervals}
    projected, price_outcomes, price_changed = _price_evidence(price_changes.candidates, interval_keys)
    classify = policy.status is AnomalyPolicyStatus.APPROVED
    empty: tuple[tuple | None, frozenset] = (None, frozenset())
    rows: list[dict] = []
    members: list[tuple] = []
    summaries = []
    A, D = AssessabilityStatus, DenominatorStatus
    for key in order:
        timeline = by_location[key]
        counts: Counter = Counter()
        breaks: Counter = Counter()
        for i, capture in enumerate(timeline.captures):
            eligible = capture.state is CaptureState.ELIGIBLE
            context, current = sets.get((key, capture.period), empty) if eligible else empty
            pair = timeline.adjacent_pairs[i - 1] if i else None
            reason: str | None = pair.value if isinstance(pair, IntervalBreak) else None
            comparison: AssortmentComparison | None = None
            if isinstance(pair, CaptureInterval):
                before = timeline.captures[i - 1].period
                previous_context, previous = sets.get((key, before), empty)
                ikey = (key, before, capture.period)
                union = {(previous_context, p) for p in previous} | {(context, c) for c in current}
                if projected.get(ikey, set()) != union:
                    raise AssortmentPriceEvidenceError("price-change identities differ from the endpoint product sets")
                changed = price_changed.get(ikey, set())
                if previous_context is not None and context is not None and previous_context != context:
                    if changed:
                        raise AssortmentPriceEvidenceError("a price change cannot span a rental-context change")
                    reason = RENTAL_CONTEXT_CHANGED
                else:
                    # ``changed`` holds distinct (context, product) projections: several unit-specific
                    # candidates of one retained product collapse to one member here and are counted below.
                    if any(c not in previous & current for _, c in changed):
                        raise AssortmentPriceEvidenceError("only a retained product can change price")
                    comparison = compare_assortment(previous, current, interval=pair)
                    shared = previous_context if previous_context is not None else context
                    for product in sorted(previous | current):
                        state = (ProductMembership.RETAINED if product in previous and product in current
                                 else ProductMembership.ADDED if product in current else ProductMembership.REMOVED)
                        members.append((*key, *shared, *product, before, capture.period, state.value))
            row = dict.fromkeys(ASSORTMENT_TIMELINE_COLUMNS)
            row.update({"canonical_city": key[0], "canonical_location": key[1], _CAPTURE: capture.period,
                        "capture_state": capture.state.value,
                        "contributing_stream_count": len(capture.source_streams),
                        "returned_product_count": len(current) if eligible else None,
                        "has_previous_interval": comparison is not None, "interval_break_reason": reason,
                        "anomaly_policy_status": policy.status.value,
                        "retention_denominator_status": D.NOT_ASSESSABLE.value,
                        "jaccard_denominator_status": D.NOT_ASSESSABLE.value,
                        "drop_rate_denominator_status": D.NOT_ASSESSABLE.value})
            if comparison is not None:
                tally = price_outcomes.get((key, timeline.captures[i - 1].period, capture.period), Counter())
                inc, dec = tally[TerminalOutcome.INCREASE.value], tally[TerminalOutcome.DECREASE.value]
                change = comparison.addition_count + comparison.removal_count > 0
                row.update({
                    _PREVIOUS: timeline.captures[i - 1].period, "previous_product_count": comparison.previous_count,
                    "retained_count": comparison.retained_count, "addition_count": comparison.addition_count,
                    "removal_count": comparison.removal_count, "retention": comparison.retention,
                    "retention_denominator_status": comparison.retention_status.value,
                    "jaccard_similarity": comparison.jaccard,
                    "jaccard_denominator_status": comparison.jaccard_status.value,
                    "net_change": comparison.net_change, "absolute_drop": comparison.absolute_drop,
                    "drop_rate": comparison.drop_rate,
                    "drop_rate_denominator_status": comparison.drop_rate_status.value,
                    "price_increase_count": inc, "price_decrease_count": dec, "assortment_change": change,
                    "price_change": inc + dec > 0, "assortment_price_coincidence": change and inc + dec > 0,
                    "falling_assortment_with_price_increase": comparison.net_change < 0 and inc > 0,
                    "unusual_drop": classify_unusual_drop(comparison, policy) if classify else None,
                    "assessability_status": A.ASSESSED.value})
            else:
                row["assessability_status"] = (A.CAPTURE_NOT_ELIGIBLE.value if not eligible
                                               else A.SEED_CAPTURE.value if i == 0 else A.INTERVAL_BREAK.value)
            if reason is not None:
                breaks[reason] += 1
            counts.update(_row_counts(row, comparison is not None))
            rows.append(row)
        summaries.append(LocationAssortmentSummary(
            canonical_location=key, breaks=tuple((r, breaks[r]) for r in ASSORTMENT_BREAK_REASONS if breaks[r]),
            counts=AssortmentCounts(**counts)))
    return _timeline_frame(rows), _membership_frame(members, order), tuple(summaries)


def _timeline_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(ASSORTMENT_TIMELINE_COLUMNS), dtype=object)
    for column in _INT:
        frame[column] = frame[column].astype("int64")
    for column in _NULLABLE_INT:
        frame[column] = frame[column].astype("Int64")
    for column in _NULLABLE_FLOAT:
        frame[column] = frame[column].astype("Float64")
    for column in _NULLABLE_BOOL:
        frame[column] = frame[column].astype("boolean")
    frame["has_previous_interval"] = frame["has_previous_interval"].astype(bool)
    return frame


def _membership_frame(members: list[tuple], order: Sequence[Key]) -> pd.DataFrame:
    rank = {k: i for i, k in enumerate(order)}
    width = len(_LOCATION)
    members.sort(key=lambda r: (rank[r[:width]], r[-3], r[-2], r[width:-3]))
    return pd.DataFrame(members, columns=list(MEMBERSHIP_COLUMNS), dtype=object)


# ------------------------------------------------------------------ gated assessment


def _blocked(blockers: Sequence[AssortmentBlocker], upstream: Sequence[str] = (), offers_assessed: int = 0,
             policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY) -> VisibleAssortmentResult:
    return VisibleAssortmentResult(AssortmentReport(
        status=AssortmentStatus.BLOCKED, blockers=tuple(dict.fromkeys(blockers)),
        upstream_blockers=tuple(dict.fromkeys(upstream)), offers_assessed=offers_assessed,
        anomaly_policy_status=policy.status))


def assess_visible_assortment(jobs: pd.DataFrame, cars: pd.DataFrame, *, readiness: object, population: object,
                              scheduled: object, canonical_offers: object, location_authority: object,
                              price_changes: object, policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY,
                              ) -> VisibleAssortmentResult:
    """Gated visible assortment of the pricing-eligible canonical offers (see the module docstring).

    Gates (any failure returns a ``BLOCKED`` result with typed blockers and no
    calculations): pricing readiness is ready and is the assessment of exactly
    the supplied schedule, canonical-offer and location-authority reports; the
    population and canonical offers are bound to these frames; the canonical
    offers are ready with no unassessable row; the schedule assessment is valid
    with capture evidence; the approved canonical-location configuration is
    available; ``price_changes`` is a completed price-change result bound to the
    same frames and location authority, built on the same capture grid and
    offer count. The pure engine then runs, and its result is validated.
    Inputs are never modified; raw ``job_id`` is never read.

    Raises:
        TypeError: An argument has the wrong type.
    """
    from ql2_sixt_canada_analysis.canonical_offers import CanonicalOfferReport
    from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport
    from ql2_sixt_canada_analysis.location_authority import LocationAuthorityReport
    from ql2_sixt_canada_analysis.price_change_events import (
        CaptureEvidenceError,
        UnknownCanonicalLocationError,
        approved_canonical_locations,
        capture_timelines,
    )
    from ql2_sixt_canada_analysis.pricing_population import PricingPopulation, PricingPopulationError, frame_binding
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport

    for value, kind, name in ((readiness, PricingReadinessReport, "readiness"),
                              (population, PricingPopulation, "population"),
                              (scheduled, PerStreamScheduledCoverageReport, "scheduled"),
                              (canonical_offers, CanonicalOfferReport, "canonical_offers"),
                              (location_authority, LocationAuthorityReport, "location_authority"),
                              (price_changes, PriceChangeCandidateResult, "price_changes"),
                              (policy, UnusualDropPolicy, "policy")):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}")
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be DataFrames")
    B = AssortmentBlocker
    if not readiness.ready:
        return _blocked([B.PRICING_NOT_READY], [b.value for b in readiness.blocking_reasons], policy=policy)
    blockers: list[AssortmentBlocker] = []
    binding = frame_binding(jobs, cars)
    if (readiness.canonical_offers is not canonical_offers or readiness.scheduled_coverage is not scheduled
            or readiness.location_authority is not location_authority or population.binding != binding
            or canonical_offers.binding != binding or scheduled.jobs_assessed != len(jobs)):
        blockers.append(B.EVIDENCE_BINDING_MISMATCH)
    if not canonical_offers.ready or canonical_offers.unassessable_rows or canonical_offers.offers is None:
        blockers.append(B.CANONICAL_OFFERS_NOT_READY)
    if (not scheduled.is_valid or scheduled.capture_periods is None or scheduled.capture_exclusions is None
            or scheduled.unmatched_exclusions):
        blockers.append(B.SCHEDULE_EVIDENCE_INVALID)
    approved: tuple[Key, ...] = ()
    try:
        approved = approved_canonical_locations(location_authority, canonical_offers.policy)
    except UnknownCanonicalLocationError:
        blockers.append(B.LOCATION_AUTHORITY_UNAVAILABLE)
    upstream: tuple[str, ...] = ()
    if not price_changes.completed:
        blockers.append(B.PRICE_CHANGE_EVIDENCE_INVALID)
        upstream = tuple(b.value for b in price_changes.report.blockers)
    elif (price_changes.binding is None or price_changes.binding != binding
          or price_changes.location_authority is not location_authority
          or (approved and price_changes.report.approved_locations != approved)):
        blockers.append(B.PRICE_CHANGE_EVIDENCE_INVALID)
    if blockers:
        return _blocked(blockers, upstream, policy=policy)
    try:
        offers = canonical_offers.offers_for(jobs, cars)
    except PricingPopulationError:
        return _blocked([B.EVIDENCE_BINDING_MISMATCH], policy=policy)
    try:
        grid = capture_timelines(scheduled, canonical_offers.policy)
    except CaptureEvidenceError:
        return _blocked([B.SCHEDULE_EVIDENCE_INVALID], offers_assessed=len(offers), policy=policy)
    if grid != price_changes.timelines or price_changes.report.offers_assessed != len(offers):
        return _blocked([B.PRICE_CHANGE_EVIDENCE_INVALID], offers_assessed=len(offers), policy=policy)
    errors = ((AssortmentIdentityError, B.PRODUCT_IDENTITY_INCOMPLETE), (RentalContextError, B.MULTIPLE_RENTAL_CONTEXTS),
              (AssortmentLocationError, B.UNKNOWN_CANONICAL_LOCATION),
              (AssortmentCaptureError, B.CAPTURE_EVIDENCE_INCONSISTENT),
              (AssortmentPriceEvidenceError, B.PRICE_CHANGE_EVIDENCE_INVALID),
              (AssortmentReconciliationError, B.RECONCILIATION_FAILED))
    try:
        timeline, membership, summaries = calculate_visible_assortment(offers, price_changes, policy=policy)
        overall = sum((s.counts for s in summaries), AssortmentCounts())
        unusual = (int(timeline["unusual_drop"].fillna(False).astype(bool).sum())
                   if policy.status is AnomalyPolicyStatus.APPROVED else None)
        report = AssortmentReport(status=AssortmentStatus.COMPLETED, offers_assessed=len(offers),
                                  locations=summaries, overall=overall, anomaly_policy_status=policy.status,
                                  unusual_drop_intervals=unusual)
        return VisibleAssortmentResult(report=report, timeline=timeline, membership=membership,
                                       timelines=price_changes.timelines, price_changes=price_changes,
                                       binding=binding, location_authority=location_authority)
    except tuple(e for e, _ in errors) as error:
        blocker = next(b for e, b in errors if isinstance(error, e))
        return _blocked([blocker], offers_assessed=len(offers), policy=policy)


def visible_assortment_from_pipeline(run: object, *, policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY
                                     ) -> VisibleAssortmentResult:
    """Visible assortment of one :func:`~ql2_sixt_canada_analysis.pricing_pipeline.run_pricing_pipeline` result.

    Never reruns ingestion or the pipeline. Fails closed (``PRICING_NOT_READY``
    with the readiness blocker categories) unless the central pricing gate
    passed and every required assessment exists; then applies the Prompt 1
    evidence gate (:func:`~ql2_sixt_canada_analysis.assortment_contract.assortment_evidence_blockers`),
    derives the price-change candidates from the same object and runs
    :func:`assess_visible_assortment`.
    """
    from ql2_sixt_canada_analysis.price_change_events import price_change_candidates_from_pipeline
    from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport

    if not isinstance(run, PricingPipelineResult):
        raise TypeError("run must be a PricingPipelineResult")
    if not isinstance(policy, UnusualDropPolicy):
        raise TypeError("policy must be an UnusualDropPolicy")
    pricing = run.pricing
    if not isinstance(pricing, PricingReadinessReport):
        raise TypeError("the pipeline produced no readiness report")
    required = (run.population, run.scheduled, run.canonical_offers, run.location_authority)
    if not pricing.ready or any(v is None for v in required):
        return _blocked([AssortmentBlocker.PRICING_NOT_READY],
                        [b.value for b in pricing.blocking_reasons] or ["required_assessment_unavailable"],
                        policy=policy)
    gate = assortment_evidence_blockers(run)
    if gate:
        return _blocked(gate, policy=policy)
    price_changes = price_change_candidates_from_pipeline(run)
    return assess_visible_assortment(
        run.jobs, run.cars, readiness=pricing, population=run.population, scheduled=run.scheduled,
        canonical_offers=run.canonical_offers, location_authority=run.location_authority,
        price_changes=price_changes, policy=policy)


def run_visible_assortment(raw_dir: str | Path | None = None, *,
                           policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY) -> VisibleAssortmentResult:
    """Run ``run_pricing_pipeline`` exactly once, then the gated assortment engine (read-only; nothing is written)."""
    from ql2_sixt_canada_analysis import pricing_pipeline

    return visible_assortment_from_pipeline(pricing_pipeline.run_pricing_pipeline(raw_dir), policy=policy)
