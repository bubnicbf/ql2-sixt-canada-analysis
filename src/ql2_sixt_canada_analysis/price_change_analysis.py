"""Higher-order price-change analysis (issue #4): synchronization, cross-location movement, persistence.

Built only on a completed, validated
:class:`~ql2_sixt_canada_analysis.price_change_events.PriceChangeCandidateResult`
and its capture timelines; events are never rebuilt from raw rows, offers or
matched-location pairs, and the event contract is not redefined. Every result
describes **observed price-change candidates** and **synchronized observed
movements**: descriptive evidence that may be *consistent with* a repricing or
with an extraction anomaly, never proof of either.

Entry points: :func:`analyze_price_change_events` (pure, on one event result
and the exact location-authority report it was validated with),
:func:`price_change_analysis_from_pipeline` (one
:class:`~ql2_sixt_canada_analysis.pricing_pipeline.PricingPipelineResult`) and
:func:`run_price_change_analysis` (runs ``run_pricing_pipeline`` exactly once).

Evidence binding
----------------
A gated event result carries the frame binding and the location-authority
report it was validated with. The pure analysis refuses any other authority
(identity, not equality); the pipeline-aware function also requires the event
binding to equal the pipeline's frames. A mismatch is a typed blocker.

Interval grain and synchronization
----------------------------------
One row per eligible :class:`~ql2_sixt_canada_analysis.price_change_events.CaptureInterval`
of every approved canonical location, taken from the validated timelines, so
zero-event intervals stay visible and hard breaks are absent. Per interval:
the six outcome counts, comparable, ``price_change_count`` (increase +
decrease), ``assortment_event_count`` (appeared + disappeared; ambiguous is
neither), percent-valid and zero-denominator counts and a
:class:`MovementClass`. Three synchronization notions are kept separate, each
requiring at least two changed offers:

* **direction** - every changed offer moved in the same direction (unchanged
  offers do not prevent it and stay in the comparable denominator);
* **exact cent** - every changed offer has the same signed ``change_cents``;
* **exact percentage** - every changed offer has a nonzero previous price and
  the same exact rational ``100 * change_cents / previous_price_cents``
  (reduced :class:`fractions.Fraction`, never float equality).

The largest same-cent and same-percentage cohorts keep partial
synchronization visible.

Cross-location comparison
-------------------------
Only through the approved location authority: the effective comparison pairs
and the canonical roles (never role words in labels). Within one city and one
exact interval, airport and downtown candidates join on the product key
(:data:`CROSS_LOCATION_PRODUCT_COLUMNS`: the event identity without city and
location, so rental dates, product and unit must match). Each matched product
gets one :class:`CrossLocationOutcome` plus separate same-direction,
same-cent and same-percentage flags. A unit change is a disappearance plus an
appearance and never a cross-location price comparison. A city without both
roles is reported ``role_unavailable``; no counterpart is inferred.

Persistence
-----------
Exactly one record per increase or decrease. The follow-up is the candidate of
the same full identity in the immediately following eligible interval of the
same timeline (previous period = the event's current period); nothing is
searched beyond it. Outcomes (:class:`PersistenceOutcome`): ``held``,
``continued``, ``reverted`` (with ``returned_to_prior_price`` and
``overshot_prior_price`` flags), ``disappeared``, ``ambiguous`` and
``not_testable`` with a :class:`NotTestableReason` (right censoring at the
final capture, a governed exclusion, a missing capture, a non-one-hour gap or
a source-stream change). A missing follow-up candidate, an appearance of an
identity present at the shared endpoint or a broken price chain is a
:class:`PriceChangeReconciliationError`. Rates use explicit denominators and
never include not-testable events.

Vancouver
---------
The aliased Vancouver downtown location is one canonical location; source
labels are provenance only and never duplicate a candidate, count,
denominator or comparison. The final Vancouver decrease case is derived: the
latest eligible interval of the governed alias city with at least one
decrease (no hard-coded timestamp or product).

Outputs
-------
Frames stay in memory (excluded from ``repr`` and equality); reports hold
counts, approved configuration keys, enums and summary statistics.
:func:`write_price_change_analysis_outputs` writes the aggregate event table
and heatmap only into an explicitly supplied directory, atomically, after
reconciliation. Nothing here sets thresholds, alerts or causal conclusions.
"""

from __future__ import annotations

import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

import pandas as pd

from ql2_sixt_canada_analysis.price_change_events import (
    CAPTURE_STEP,
    EVENT_IDENTITY_COLUMNS,
    EVENT_INTERVAL_COLUMNS,
    CaptureState,
    PriceChangeCandidateResult,
    PriceChangeContractError,
    TerminalOutcome,
    parse_scheduled_period,
)

__all__ = [
    "CROSS_LOCATION_COLUMNS",
    "CROSS_LOCATION_PRODUCT_COLUMNS",
    "EVENT_TABLE_COLUMNS",
    "PERSISTENCE_COLUMNS",
    "CrossLocationOutcome",
    "CrossLocationStatus",
    "CrossLocationSummary",
    "FinalDecreaseCase",
    "FinalDecreaseIndicator",
    "FinalDecreaseStatus",
    "IntervalFlag",
    "LocationAnalysisSummary",
    "MovementClass",
    "NotTestableReason",
    "PersistenceOutcome",
    "PersistenceSummary",
    "PriceChangeAnalysisBlocker",
    "PriceChangeAnalysisReport",
    "PriceChangeAnalysisResult",
    "PriceChangeAnalysisStatus",
    "PriceChangeReconciliationError",
    "analyze_price_change_events",
    "event_heatmap_source",
    "exact_change_percent",
    "plot_price_change_heatmap",
    "price_change_analysis_from_pipeline",
    "run_price_change_analysis",
    "write_price_change_analysis_outputs",
]

Key = tuple[str, ...]
PREV, CUR = EVENT_INTERVAL_COLUMNS
#: The cross-location product key: the event identity without canonical city and location.
CROSS_LOCATION_PRODUCT_COLUMNS: tuple[str, ...] = tuple(
    c for c in EVENT_IDENTITY_COLUMNS if c not in ("canonical_city", "canonical_location"))

_OUTCOMES = tuple(o.value for o in TerminalOutcome)


class PriceChangeReconciliationError(PriceChangeContractError):
    """Higher-order evidence contradicts the validated events (broken chain, missing candidate, bad totals)."""


class PriceChangeAnalysisStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class PriceChangeAnalysisBlocker(StrEnum):
    """Why no higher-order result was produced (categories only)."""

    EVENTS_NOT_COMPLETED = "events_not_completed"
    EVENT_EVIDENCE_UNBOUND = "event_evidence_unbound"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    ROLE_AUTHORITY_UNAVAILABLE = "role_authority_unavailable"
    RECONCILIATION_FAILED = "reconciliation_failed"


class MovementClass(StrEnum):
    """Price-movement class of one location interval (from increase and decrease counts only)."""

    NO_PRICE_MOVEMENT = "no_price_movement"
    ISOLATED_INCREASE = "isolated_increase"
    ISOLATED_DECREASE = "isolated_decrease"
    SYNCHRONIZED_INCREASE = "synchronized_increase"
    SYNCHRONIZED_DECREASE = "synchronized_decrease"
    MIXED_DIRECTION = "mixed_direction"


class IntervalFlag(StrEnum):
    """Descriptive interval-quality flag (first applicable wins; never a conclusion)."""

    EMPTY_ENDPOINT = "empty_endpoint"                 # an endpoint holds no offers: possible extraction anomaly
    AMBIGUITY_PRESENT = "ambiguity_present"
    PRICE_AND_ASSORTMENT = "price_and_assortment_change"
    ASSORTMENT_ONLY = "assortment_change_only"
    PRICE_ONLY = "price_change_only"
    QUIET = "quiet"


class CrossLocationStatus(StrEnum):
    AVAILABLE = "available"
    ROLE_UNAVAILABLE = "role_unavailable"


class CrossLocationOutcome(StrEnum):
    """One matched airport/downtown product in one interval (first applicable wins)."""

    AMBIGUOUS = "ambiguous"
    SIMULTANEOUS_APPEARANCE = "simultaneous_appearance"
    SIMULTANEOUS_DISAPPEARANCE = "simultaneous_disappearance"
    MIXED_ASSORTMENT = "mixed_assortment"            # appeared on one side, disappeared on the other
    ONE_SIDED_ASSORTMENT = "one_sided_assortment"    # appeared or disappeared on one side only
    BOTH_UNCHANGED = "both_unchanged"
    AIRPORT_ONLY_CHANGE = "airport_only_change"
    DOWNTOWN_ONLY_CHANGE = "downtown_only_change"
    SAME_DIRECTION = "same_direction"
    OPPOSITE_DIRECTION = "opposite_direction"


class PersistenceOutcome(StrEnum):
    HELD = "held"
    CONTINUED = "continued"
    REVERTED = "reverted"
    DISAPPEARED = "disappeared"
    AMBIGUOUS = "ambiguous"
    NOT_TESTABLE = "not_testable"


class NotTestableReason(StrEnum):
    RIGHT_CENSORED_FINAL_CAPTURE = "right_censored_final_capture"
    GOVERNED_EXCLUSION_BREAK = "governed_exclusion_break"
    MISSING_CAPTURE_BREAK = "missing_capture_break"
    NOT_ONE_HOUR_BREAK = "not_one_hour_break"
    SOURCE_STREAMS_CHANGED = "source_streams_changed"


class FinalDecreaseStatus(StrEnum):
    DERIVED = "derived"
    NO_DECREASE = "no_decrease"
    CITY_UNAVAILABLE = "city_unavailable"


class FinalDecreaseIndicator(StrEnum):
    """Descriptive evidence indicators of the final decrease (never proof of repricing or of extraction error)."""

    DIRECTION_SYNCHRONIZED = "direction_synchronized"
    BROAD_CHANGE_SHARE = "majority_of_comparable_offers_changed"
    STABLE_ASSORTMENT = "no_assortment_events"
    ASSORTMENT_DISCONTINUITY = "assortment_events_present"
    EMPTY_ENDPOINT = "empty_endpoint_present"
    AMBIGUITY_PRESENT = "ambiguity_present"
    PROVENANCE_CHANGE_PRESENT = "source_provenance_changed"
    CROSS_LOCATION_SAME_DIRECTION = "airport_downtown_same_direction"
    PERSISTENCE_RIGHT_CENSORED = "persistence_not_testable_right_censored"


#: The aggregate event table: one row per approved canonical location and eligible interval (no product values).
EVENT_TABLE_COLUMNS: tuple[str, ...] = (
    "canonical_city", "canonical_location", "role", PREV, CUR, "candidates", *_OUTCOMES, "comparable",
    "price_change_count", "assortment_event_count", "percent_valid", "zero_denominator", "previous_offers",
    "current_offers", "changed_share_of_comparable", "movement_class", "direction_synchronized",
    "exact_cent_synchronized", "exact_percent_synchronized", "largest_same_cent_cohort",
    "largest_same_percent_cohort", "min_change_cents", "max_change_cents", "median_abs_change_percent",
    "max_abs_change_percent", "multi_source_candidates", "provenance_changed_candidates",
    *(f"persistence_{o.value}" for o in PersistenceOutcome), "interval_flag")
#: In-memory matched airport/downtown products (proprietary product keys; never exported).
CROSS_LOCATION_COLUMNS: tuple[str, ...] = (
    "canonical_city", "airport_location", "downtown_location", PREV, CUR, *CROSS_LOCATION_PRODUCT_COLUMNS,
    "airport_outcome", "downtown_outcome", "cross_outcome", "same_direction", "same_cent_change",
    "same_percent_change")
#: In-memory persistence records, one per increase or decrease (proprietary; never exported).
PERSISTENCE_COLUMNS: tuple[str, ...] = (
    *EVENT_IDENTITY_COLUMNS, PREV, CUR, "direction", "next_current_period", "persistence", "not_testable_reason",
    "returned_to_prior_price", "overshot_prior_price")


# ------------------------------------------------------------------ exact arithmetic


def exact_change_percent(previous_cents: object, change_cents: object) -> Fraction | None:
    """``100 * change / previous`` as a reduced exact rational (``None`` for a zero or missing denominator)."""
    if not isinstance(previous_cents, int) or isinstance(previous_cents, bool) or previous_cents <= 0:
        return None
    if not isinstance(change_cents, int) or isinstance(change_cents, bool):
        raise PriceChangeReconciliationError("a change must be exact integer cents")
    return Fraction(100 * change_cents, previous_cents)


def _movement(increase: int, decrease: int) -> MovementClass:
    M = MovementClass
    if increase + decrease == 0:
        return M.NO_PRICE_MOVEMENT
    if increase + decrease == 1:
        return M.ISOLATED_INCREASE if increase else M.ISOLATED_DECREASE
    if increase and decrease:
        return M.MIXED_DIRECTION
    return M.SYNCHRONIZED_INCREASE if increase else M.SYNCHRONIZED_DECREASE


def _synchronization(changed: list[dict]) -> dict:
    """The three separate synchronization notions and the largest identical cohorts of changed rows."""
    cents = [r["change_cents"] for r in changed]
    percents = [exact_change_percent(r["previous_price_cents"], r["change_cents"]) for r in changed]
    valid = [p for p in percents if p is not None]
    n = len(changed)
    return {
        "direction_synchronized": n >= 2 and (all(c > 0 for c in cents) or all(c < 0 for c in cents)),
        "exact_cent_synchronized": n >= 2 and len(set(cents)) == 1,
        "exact_percent_synchronized": n >= 2 and len(valid) == n and len(set(valid)) == 1,
        "largest_same_cent_cohort": max(Counter(cents).values(), default=0),
        "largest_same_percent_cohort": max(Counter(valid).values(), default=0),
    }


def _share(numerator: int, denominator: int) -> float | None:
    return float(Fraction(numerator, denominator)) if denominator else None


def _labels(row: dict, side: str) -> frozenset:
    text = row[f"{side}_source_labels"]
    return frozenset(p for p in text.split("|") if p) if isinstance(text, str) else frozenset()


# ------------------------------------------------------------------ reports


def _count(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PriceChangeReconciliationError(f"{name} must be a non-negative int")


@dataclass(frozen=True, slots=True)
class LocationAnalysisSummary:
    """Synchronization accounting of one approved canonical location (counts and enums only)."""

    canonical_location: Key
    role: str
    intervals: int
    candidates: int
    price_change_count: int
    assortment_event_count: int
    ambiguous: int
    movement_classes: tuple[tuple[str, int], ...]
    direction_synchronized_intervals: int
    exact_cent_synchronized_intervals: int
    exact_percent_synchronized_intervals: int
    empty_endpoint_intervals: int
    multi_source_candidates: int
    provenance_changed_candidates: int

    def __post_init__(self) -> None:
        for name in ("intervals", "candidates", "price_change_count", "assortment_event_count", "ambiguous",
                     "direction_synchronized_intervals", "exact_cent_synchronized_intervals",
                     "exact_percent_synchronized_intervals", "empty_endpoint_intervals",
                     "multi_source_candidates", "provenance_changed_candidates"):
            _count(getattr(self, name), name)
        if sum(n for _, n in self.movement_classes) != self.intervals:
            raise PriceChangeReconciliationError("every interval has exactly one movement class")
        if self.price_change_count + self.assortment_event_count + self.ambiguous > self.candidates:
            raise PriceChangeReconciliationError("price, assortment and ambiguous events are disjoint candidates")


@dataclass(frozen=True, slots=True)
class CrossLocationSummary:
    """Airport/downtown comparison of one city under the approved authority (counts only)."""

    canonical_city: str
    status: CrossLocationStatus
    airport: Key | None = None
    downtown: Key | None = None
    intervals_compared: int = 0
    matched_products: int = 0
    airport_only_products: int = 0
    downtown_only_products: int = 0
    outcomes: tuple[tuple[str, int], ...] = ()
    same_direction: int = 0
    same_cent_change: int = 0
    same_percent_change: int = 0

    def __post_init__(self) -> None:
        if self.status is CrossLocationStatus.ROLE_UNAVAILABLE and (self.matched_products or self.outcomes):
            raise PriceChangeReconciliationError("an unavailable comparison holds no result")
        if sum(n for _, n in self.outcomes) != self.matched_products:
            raise PriceChangeReconciliationError("every matched product has exactly one cross-location outcome")
        if not self.same_cent_change <= self.same_direction or not self.same_percent_change <= self.same_direction:
            raise PriceChangeReconciliationError("exact matches are a subset of same-direction matches")


@dataclass(frozen=True, slots=True)
class PersistenceSummary:
    """Persistence of every increase and decrease, with explicit denominators."""

    changed_events: int = 0
    with_following_interval: int = 0
    held: int = 0
    continued: int = 0
    reverted: int = 0
    disappeared: int = 0
    ambiguous: int = 0
    not_testable: int = 0
    returned_to_prior_price: int = 0
    overshot_prior_price: int = 0
    not_testable_reasons: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.changed_events != (self.held + self.continued + self.reverted + self.disappeared + self.ambiguous
                                   + self.not_testable):
            raise PriceChangeReconciliationError("persistence outcomes partition every changed event")
        if self.with_following_interval != self.changed_events - self.not_testable:
            raise PriceChangeReconciliationError("testable events are those with a following interval")
        if self.returned_to_prior_price + self.overshot_prior_price > self.reverted:
            raise PriceChangeReconciliationError("full returns and overshoots are reversions")
        if sum(n for _, n in self.not_testable_reasons) != self.not_testable:
            raise PriceChangeReconciliationError("every not-testable event has one reason")

    @property
    def comparable_following(self) -> int:
        """Denominator of the price-path rates: held + continued + reverted (excludes disappeared and ambiguous)."""
        return self.held + self.continued + self.reverted

    def rates(self) -> dict[str, float | None]:
        """Shares with their denominators named in the key (not-testable events are never in a denominator)."""
        c, t = self.comparable_following, self.with_following_interval
        return {"held_of_comparable_following": _share(self.held, c),
                "continued_of_comparable_following": _share(self.continued, c),
                "reverted_of_comparable_following": _share(self.reverted, c),
                "held_of_testable": _share(self.held, t),
                "disappeared_of_testable": _share(self.disappeared, t),
                "ambiguous_of_testable": _share(self.ambiguous, t)}


@dataclass(frozen=True)
class FinalDecreaseCase:
    """The derived final decrease of the governed alias city (aggregates only; the interval is not printed)."""

    status: FinalDecreaseStatus
    canonical_city: str | None = None
    previous_period: str | None = field(default=None, repr=False)
    current_period: str | None = field(default=None, repr=False)
    locations: tuple[tuple[Key, str, bool], ...] = ()         # (canonical location, role, ends at final capture)
    counts: tuple[tuple[str, int], ...] = ()
    comparable: int = 0
    price_change_count: int = 0
    assortment_event_count: int = 0
    changed_share_of_comparable: float | None = None
    direction_synchronized: bool = False
    exact_cent_synchronized: bool = False
    exact_percent_synchronized: bool = False
    largest_same_cent_cohort: int = 0
    largest_same_percent_cohort: int = 0
    decrease_cents: tuple[tuple[str, float], ...] = ()       # min / median / max of signed decrease cents
    decrease_percent: tuple[tuple[str, float], ...] = ()     # min / median / max exact percentage (as floats)
    cross_location: tuple[tuple[str, int], ...] = ()
    provenance: tuple[tuple[str, int], ...] = ()
    persistence: tuple[tuple[str, int], ...] = ()
    not_testable_reasons: tuple[tuple[str, int], ...] = ()
    indicators: tuple[FinalDecreaseIndicator, ...] = ()

    def __post_init__(self) -> None:
        derived = self.status is FinalDecreaseStatus.DERIVED
        if derived != (self.current_period is not None and bool(self.locations)):
            raise PriceChangeReconciliationError("only a derived case names its interval and locations")
        if derived and dict(self.counts).get(TerminalOutcome.DECREASE.value, 0) < 1:
            raise PriceChangeReconciliationError("a derived final decrease holds at least one decrease")
        if sum(n for _, n in self.persistence) != self.price_change_count:
            raise PriceChangeReconciliationError("the case's persistence covers its changed events")

    @property
    def persistence_testable(self) -> bool:
        return any(k != PersistenceOutcome.NOT_TESTABLE.value and n for k, n in self.persistence)

    def describe(self) -> str:
        """A disciplined, metric-free sentence (descriptive evidence only; never proof)."""
        if self.status is not FinalDecreaseStatus.DERIVED:
            return "No final decrease was derived for the governed alias city."
        parts = ["The final observed decrease interval is descriptive evidence only."]
        indicators = set(self.indicators)
        if {FinalDecreaseIndicator.DIRECTION_SYNCHRONIZED, FinalDecreaseIndicator.STABLE_ASSORTMENT} <= indicators:
            parts.append("A directionally synchronized observed movement with stable assortment is more "
                         "consistent with an observed repricing candidate than with an isolated extraction error, "
                         "but it is not proof of repricing.")
        anomaly = {FinalDecreaseIndicator.ASSORTMENT_DISCONTINUITY, FinalDecreaseIndicator.EMPTY_ENDPOINT,
                   FinalDecreaseIndicator.AMBIGUITY_PRESENT, FinalDecreaseIndicator.PROVENANCE_CHANGE_PRESENT}
        if indicators & anomaly:
            parts.append("Assortment, ambiguity or provenance indicators are present; they are possible "
                         "extraction-anomaly indicators, not proof of extraction failure.")
        if FinalDecreaseIndicator.PERSISTENCE_RIGHT_CENSORED in indicators:
            parts.append("Persistence is not testable due to right censoring at the final capture.")
        return " ".join(parts)


@dataclass(frozen=True)
class PriceChangeAnalysisReport:
    """Print-safe higher-order report: status, blockers, counts, enums, approved keys and summary statistics."""

    status: PriceChangeAnalysisStatus
    blockers: tuple[PriceChangeAnalysisBlocker, ...] = ()
    event_blockers: tuple[str, ...] = ()
    intervals: int = 0
    candidates: int = 0
    price_change_count: int = 0
    assortment_event_count: int = 0
    ambiguous: int = 0
    movement_classes: tuple[tuple[str, int], ...] = ()
    locations: tuple[LocationAnalysisSummary, ...] = ()
    cross_location: tuple[CrossLocationSummary, ...] = ()
    persistence: PersistenceSummary | None = None
    final_decrease: FinalDecreaseCase | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PriceChangeAnalysisStatus):
            raise PriceChangeReconciliationError("status must be a PriceChangeAnalysisStatus")
        if self.status is PriceChangeAnalysisStatus.BLOCKED:
            if (not self.blockers or self.locations or self.persistence is not None
                    or self.final_decrease is not None or self.cross_location):
                raise PriceChangeReconciliationError("a blocked report has blockers and no result")
            return
        if self.blockers or self.event_blockers or self.persistence is None or self.final_decrease is None:
            raise PriceChangeReconciliationError("a completed report has no blockers and every section")
        for name in ("intervals", "candidates", "price_change_count", "assortment_event_count", "ambiguous"):
            if sum(getattr(s, name) for s in self.locations) != getattr(self, name):
                raise PriceChangeReconciliationError(f"location {name} must sum to the overall count")
        if sum(n for _, n in self.movement_classes) != self.intervals:
            raise PriceChangeReconciliationError("every interval has exactly one movement class")
        if self.persistence.changed_events != self.price_change_count:
            raise PriceChangeReconciliationError("persistence must cover every increase and decrease")

    @property
    def completed(self) -> bool:
        return self.status is PriceChangeAnalysisStatus.COMPLETED

    def location(self, key: Key) -> LocationAnalysisSummary:
        return next(s for s in self.locations if s.canonical_location == tuple(key))


# ------------------------------------------------------------------ construction


def _records(frame: pd.DataFrame) -> list[dict]:
    return [dict(zip(frame.columns, row)) for row in frame.itertuples(index=False, name=None)]


def _identity(row: dict) -> tuple:
    return tuple(row[c] for c in EVENT_IDENTITY_COLUMNS)


def _check_chains(rows_by_interval: dict, timelines) -> None:  # type: ignore[no-untyped-def]
    """Adjacent intervals agree at their shared endpoint: same identities, offer counts and exposed prices."""
    for timeline in timelines:
        key = timeline.canonical_location
        following = {i.previous_period: i for i in timeline.intervals}
        for interval in timeline.intervals:
            nxt = following.get(interval.current_period)
            if nxt is None:
                continue
            before = {_identity(r): r for r in rows_by_interval.get((key, interval.previous_period,
                                                                     interval.current_period), [])}
            after = {_identity(r): r for r in rows_by_interval.get((key, nxt.previous_period, nxt.current_period), [])}
            present = {i: r["current_offer_count"] for i, r in before.items() if r["current_offer_count"]}
            seen = {i: r["previous_offer_count"] for i, r in after.items() if r["previous_offer_count"]}
            if present != seen:
                raise PriceChangeReconciliationError("adjacent intervals disagree at their shared capture "
                                                     "(missing follow-up candidate or impossible appearance)")
            for identity, row in before.items():
                if row["current_price_cents"] is None or identity not in after:
                    continue
                price = after[identity]["previous_price_cents"]
                if price is not None and price != row["current_price_cents"]:
                    raise PriceChangeReconciliationError("broken price chain between adjacent intervals")


def _not_testable_reason(timeline, period: str) -> NotTestableReason:  # type: ignore[no-untyped-def]
    captures = timeline.captures
    index = next(i for i, c in enumerate(captures) if c.period == period)
    if index == len(captures) - 1:
        return NotTestableReason.RIGHT_CENSORED_FINAL_CAPTURE
    here, nxt = captures[index], captures[index + 1]
    if nxt.state is CaptureState.GOVERNED_EXCLUSION:
        return NotTestableReason.GOVERNED_EXCLUSION_BREAK
    if nxt.state is CaptureState.MISSING_CAPTURE:
        return NotTestableReason.MISSING_CAPTURE_BREAK
    if parse_scheduled_period(nxt.period) - parse_scheduled_period(here.period) != CAPTURE_STEP:
        return NotTestableReason.NOT_ONE_HOUR_BREAK
    if nxt.source_streams != here.source_streams:
        return NotTestableReason.SOURCE_STREAMS_CHANGED
    raise PriceChangeReconciliationError("an eligible adjacent capture without an interval")


def _persistence(changed: list[dict], index: dict, timelines_by_key: dict) -> list[tuple]:
    P = PersistenceOutcome
    out = []
    for row in changed:
        key = (row["canonical_city"], row["canonical_location"])
        timeline = timelines_by_key[key]
        following = {i.previous_period: i for i in timeline.intervals}.get(row[CUR])
        increase = row["outcome"] == TerminalOutcome.INCREASE.value
        reason, nxt_period, returned, overshot = None, None, False, False
        if following is None:
            outcome, reason = P.NOT_TESTABLE, _not_testable_reason(timeline, row[CUR]).value
        else:
            nxt_period = following.current_period
            nxt = index.get((key, following.previous_period, following.current_period, _identity(row)))
            if nxt is None:
                raise PriceChangeReconciliationError("a following eligible interval lacks the changed identity")
            result = TerminalOutcome(nxt["outcome"])
            if result is TerminalOutcome.APPEARED:
                raise PriceChangeReconciliationError("an identity present at the shared capture cannot appear")
            if result is TerminalOutcome.AMBIGUOUS:
                outcome = P.AMBIGUOUS
            elif result is TerminalOutcome.DISAPPEARED:
                outcome = P.DISAPPEARED
            else:
                if nxt["previous_price_cents"] != row["current_price_cents"]:
                    raise PriceChangeReconciliationError("broken price chain at the following interval")
                price, changed_to, prior = nxt["current_price_cents"], row["current_price_cents"], \
                    row["previous_price_cents"]
                if price == changed_to:
                    outcome = P.HELD
                elif (price > changed_to) == increase:
                    outcome = P.CONTINUED
                else:
                    outcome = P.REVERTED
                    returned = price == prior
                    overshot = price < prior if increase else price > prior
        out.append((*_identity(row), row[PREV], row[CUR], row["outcome"], nxt_period, outcome.value, reason,
                    returned, overshot))
    return out


def _summary_stats(values: Sequence) -> tuple[tuple[str, float], ...]:  # type: ignore[type-arg]
    if not values:
        return ()
    return (("min", float(min(values))), ("median", float(statistics.median(values))), ("max", float(max(values))))


def _build(events: PriceChangeCandidateResult, authority) -> tuple:  # type: ignore[no-untyped-def]
    """Every higher-order table and the report (deterministic; raises on any reconciliation failure)."""
    report = events.report
    order = report.approved_locations
    timelines = {t.canonical_location: t for t in events.timelines}
    roles = {k: authority.canonical_role(k) for k in order}
    if any(r is None for r in roles.values()):
        raise PriceChangeReconciliationError("every approved location needs an authority-backed role")
    rows = _records(events.candidates)
    by_interval: dict[tuple, list[dict]] = {}
    index: dict[tuple, dict] = {}
    allowed = {(k, i.previous_period, i.current_period) for k, t in timelines.items() for i in t.intervals}
    for row in rows:
        key = ((row["canonical_city"], row["canonical_location"]), row[PREV], row[CUR])
        if key not in allowed:
            raise PriceChangeReconciliationError("a candidate belongs to no eligible location interval")
        by_interval.setdefault(key, []).append(row)
        index[(*key, _identity(row))] = row
    if len(index) != len(rows):
        raise PriceChangeReconciliationError("a candidate belongs to more than one interval record")
    _check_chains(by_interval, events.timelines)

    # Persistence: exactly one record per increase or decrease.
    changed = [r for r in rows if r["outcome"] in (TerminalOutcome.INCREASE.value, TerminalOutcome.DECREASE.value)]
    persistence = pd.DataFrame(_persistence(changed, index, timelines), columns=list(PERSISTENCE_COLUMNS),
                               dtype=object)
    for column in ("returned_to_prior_price", "overshot_prior_price"):
        persistence[column] = persistence[column].astype(bool)
    by_origin: dict[tuple, Counter] = {}
    for rec in persistence.itertuples(index=False, name=None):
        values = dict(zip(PERSISTENCE_COLUMNS, rec))
        origin = ((values["canonical_city"], values["canonical_location"]), values[PREV], values[CUR])
        by_origin.setdefault(origin, Counter())[values["persistence"]] += 1

    # Interval table.
    table_rows, summaries = [], []
    for key in order:
        timeline = timelines[key]
        role = roles[key].value
        loc_rows = []
        for interval in timeline.intervals:
            ikey = (key, interval.previous_period, interval.current_period)
            group = by_interval.get(ikey, [])
            counts = Counter(r["outcome"] for r in group)
            moved = [r for r in group if r["outcome"] in (TerminalOutcome.INCREASE.value,
                                                          TerminalOutcome.DECREASE.value)]
            comparable = sum(counts[o] for o in ("unchanged", "increase", "decrease"))
            sync = _synchronization(moved)
            percents = [abs(p) for r in moved
                        if (p := exact_change_percent(r["previous_price_cents"], r["change_cents"])) is not None]
            prev_offers = sum(r["previous_offer_count"] for r in group)
            cur_offers = sum(r["current_offer_count"] for r in group)
            price, assortment = counts["increase"] + counts["decrease"], counts["appeared"] + counts["disappeared"]
            multi = sum(1 for r in group if len(_labels(r, "previous") | _labels(r, "current")) > 1)
            provenance_changed = sum(1 for r in group if _labels(r, "previous") and _labels(r, "current")
                                     and _labels(r, "previous") != _labels(r, "current"))
            if prev_offers == 0 or cur_offers == 0:
                flag = IntervalFlag.EMPTY_ENDPOINT
            elif counts["ambiguous"]:
                flag = IntervalFlag.AMBIGUITY_PRESENT
            elif price and assortment:
                flag = IntervalFlag.PRICE_AND_ASSORTMENT
            elif assortment:
                flag = IntervalFlag.ASSORTMENT_ONLY
            elif price:
                flag = IntervalFlag.PRICE_ONLY
            else:
                flag = IntervalFlag.QUIET
            persisted = by_origin.get(ikey, Counter())
            cents = [r["change_cents"] for r in moved]
            loc_rows.append((
                key[0], key[1], role, interval.previous_period, interval.current_period, len(group),
                *(counts[o] for o in _OUTCOMES), comparable, price, assortment,
                sum(1 for r in group if r["percent_valid"]), sum(1 for r in group if r["zero_denominator"]),
                prev_offers, cur_offers, _share(price, comparable),
                _movement(counts["increase"], counts["decrease"]).value, sync["direction_synchronized"],
                sync["exact_cent_synchronized"], sync["exact_percent_synchronized"],
                sync["largest_same_cent_cohort"], sync["largest_same_percent_cohort"],
                min(cents) if cents else None, max(cents) if cents else None,
                float(statistics.median(percents)) if percents else None,
                float(max(percents)) if percents else None, multi, provenance_changed,
                *(persisted[o.value] for o in PersistenceOutcome), flag.value))
        table_rows.extend(loc_rows)
        loc = pd.DataFrame(loc_rows, columns=list(EVENT_TABLE_COLUMNS), dtype=object)
        summaries.append(LocationAnalysisSummary(
            canonical_location=key, role=role, intervals=len(loc_rows), candidates=int(loc["candidates"].sum()),
            price_change_count=int(loc["price_change_count"].sum()),
            assortment_event_count=int(loc["assortment_event_count"].sum()), ambiguous=int(loc["ambiguous"].sum()),
            movement_classes=tuple((m.value, int((loc["movement_class"] == m.value).sum())) for m in MovementClass),
            direction_synchronized_intervals=int(loc["direction_synchronized"].sum()),
            exact_cent_synchronized_intervals=int(loc["exact_cent_synchronized"].sum()),
            exact_percent_synchronized_intervals=int(loc["exact_percent_synchronized"].sum()),
            empty_endpoint_intervals=int((loc["interval_flag"] == IntervalFlag.EMPTY_ENDPOINT.value).sum()),
            multi_source_candidates=int(loc["multi_source_candidates"].sum()),
            provenance_changed_candidates=int(loc["provenance_changed_candidates"].sum())))
    table = pd.DataFrame(table_rows, columns=list(EVENT_TABLE_COLUMNS), dtype=object)
    for column in ("direction_synchronized", "exact_cent_synchronized", "exact_percent_synchronized"):
        table[column] = table[column].astype(bool)

    # Cross-location comparison through the approved pairs only.
    cross_rows, cross = _cross_location(order, roles, authority, timelines, by_interval)
    cross_frame = pd.DataFrame(cross_rows, columns=list(CROSS_LOCATION_COLUMNS), dtype=object)
    for column in ("same_direction", "same_cent_change", "same_percent_change"):
        cross_frame[column] = cross_frame[column].astype(bool)

    persistence_summary = _persistence_summary(persistence)
    final = _final_decrease(authority, order, roles, timelines, table, by_interval, persistence, cross_frame)
    overall = report.overall
    analysis = PriceChangeAnalysisReport(
        status=PriceChangeAnalysisStatus.COMPLETED, intervals=len(table), candidates=int(table["candidates"].sum()),
        price_change_count=int(table["price_change_count"].sum()),
        assortment_event_count=int(table["assortment_event_count"].sum()), ambiguous=int(table["ambiguous"].sum()),
        movement_classes=tuple((m.value, int((table["movement_class"] == m.value).sum())) for m in MovementClass),
        locations=tuple(summaries), cross_location=cross, persistence=persistence_summary, final_decrease=final)
    _reconcile(analysis, overall, table, persistence, cross_frame, rows)
    return analysis, table, cross_frame, persistence


def _cross_location(order, roles, authority, timelines, by_interval):  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis.authority_decisions import LocationRoleDecision as R

    pairs = {}
    if authority.pairs_valid:
        for pair in authority.effective_pairs:
            airport, downtown = tuple(pair.airport), tuple(pair.downtown)
            if airport in roles and downtown in roles:
                if roles[airport] is not R.AIRPORT or roles[downtown] is not R.DOWNTOWN or airport[0] != downtown[0]:
                    raise PriceChangeReconciliationError("an approved pair contradicts the approved roles")
                pairs[airport[0]] = (airport, downtown)
    rows, summaries = [], []
    for city in dict.fromkeys(k[0] for k in order):
        if city not in pairs:
            summaries.append(CrossLocationSummary(city, CrossLocationStatus.ROLE_UNAVAILABLE))
            continue
        airport, downtown = pairs[city]
        a_int = {(i.previous_period, i.current_period) for i in timelines[airport].intervals}
        d_int = {(i.previous_period, i.current_period) for i in timelines[downtown].intervals}
        shared = sorted(a_int & d_int, key=lambda p: (parse_scheduled_period(p[0]), parse_scheduled_period(p[1])))
        outcomes: Counter = Counter()
        a_only = d_only = same_dir = same_cent = same_pct = 0
        for prev, cur in shared:
            a_rows = {tuple(r[c] for c in CROSS_LOCATION_PRODUCT_COLUMNS): r
                      for r in by_interval.get((airport, prev, cur), [])}
            d_rows = {tuple(r[c] for c in CROSS_LOCATION_PRODUCT_COLUMNS): r
                      for r in by_interval.get((downtown, prev, cur), [])}
            a_only += len(set(a_rows) - set(d_rows))
            d_only += len(set(d_rows) - set(a_rows))
            for product in sorted(set(a_rows) & set(d_rows)):
                a, d = a_rows[product], d_rows[product]
                outcome, flags = _cross_outcome(a, d)
                outcomes[outcome.value] += 1
                same_dir += flags[0]
                same_cent += flags[1]
                same_pct += flags[2]
                rows.append((city, airport[1], downtown[1], prev, cur, *product, a["outcome"], d["outcome"],
                             outcome.value, *flags))
        summaries.append(CrossLocationSummary(
            city, CrossLocationStatus.AVAILABLE, airport, downtown, intervals_compared=len(shared),
            matched_products=sum(outcomes.values()), airport_only_products=a_only, downtown_only_products=d_only,
            outcomes=tuple((o.value, outcomes[o.value]) for o in CrossLocationOutcome if outcomes[o.value]),
            same_direction=same_dir, same_cent_change=same_cent, same_percent_change=same_pct))
    return rows, tuple(summaries)


def _cross_outcome(a: dict, d: dict) -> tuple[CrossLocationOutcome, tuple[bool, bool, bool]]:
    C, T = CrossLocationOutcome, TerminalOutcome
    oa, od = T(a["outcome"]), T(d["outcome"])
    assortment = {T.APPEARED, T.DISAPPEARED}
    none = (False, False, False)
    if T.AMBIGUOUS in (oa, od):
        return C.AMBIGUOUS, none
    if oa in assortment or od in assortment:
        if oa == od == T.APPEARED:
            return C.SIMULTANEOUS_APPEARANCE, none
        if oa == od == T.DISAPPEARED:
            return C.SIMULTANEOUS_DISAPPEARANCE, none
        if oa in assortment and od in assortment:
            return C.MIXED_ASSORTMENT, none
        return C.ONE_SIDED_ASSORTMENT, none
    ca, cd = a["change_cents"], d["change_cents"]
    if ca == 0 and cd == 0:
        return C.BOTH_UNCHANGED, none
    if cd == 0:
        return C.AIRPORT_ONLY_CHANGE, none
    if ca == 0:
        return C.DOWNTOWN_ONLY_CHANGE, none
    if (ca > 0) != (cd > 0):
        return C.OPPOSITE_DIRECTION, none
    pa = exact_change_percent(a["previous_price_cents"], ca)
    pd_ = exact_change_percent(d["previous_price_cents"], cd)
    return C.SAME_DIRECTION, (True, ca == cd, pa is not None and pd_ is not None and pa == pd_)


def _persistence_summary(persistence: pd.DataFrame) -> PersistenceSummary:
    counts = Counter(persistence["persistence"])
    reasons = Counter(r for r in persistence["not_testable_reason"] if r is not None)
    return PersistenceSummary(
        changed_events=len(persistence), with_following_interval=len(persistence) - counts["not_testable"],
        held=counts["held"], continued=counts["continued"], reverted=counts["reverted"],
        disappeared=counts["disappeared"], ambiguous=counts["ambiguous"], not_testable=counts["not_testable"],
        returned_to_prior_price=int(persistence["returned_to_prior_price"].sum()),
        overshot_prior_price=int(persistence["overshot_prior_price"].sum()),
        not_testable_reasons=tuple((r.value, reasons[r.value]) for r in NotTestableReason if reasons[r.value]))


def _final_decrease(authority, order, roles, timelines, table, by_interval, persistence,  # type: ignore[no-untyped-def]
                    cross) -> FinalDecreaseCase:
    S, I = FinalDecreaseStatus, FinalDecreaseIndicator
    policy = getattr(authority, "policy", None)
    city = getattr(policy, "first", (None,))[0] if policy is not None else None
    locations = [k for k in order if k[0] == city]
    if not locations:
        return FinalDecreaseCase(S.CITY_UNAVAILABLE)
    subset = table[(table["canonical_city"] == city) & (table["decrease"] > 0)]
    if subset.empty:
        return FinalDecreaseCase(S.NO_DECREASE, canonical_city=city)
    latest = max(zip(subset[PREV], subset[CUR]),
                 key=lambda p: (parse_scheduled_period(p[1]), parse_scheduled_period(p[0])))
    involved = [k for k in locations
                if any((i.previous_period, i.current_period) == latest for i in timelines[k].intervals)]
    rows = [r for k in involved for r in by_interval.get((k, *latest), [])]
    case_table = table[(table["canonical_city"] == city) & (table[PREV] == latest[0]) & (table[CUR] == latest[1])]
    if len(case_table) != len(involved) or int(case_table["candidates"].sum()) != len(rows):
        raise PriceChangeReconciliationError("the final decrease case is not a subset of the interval table")
    counts = Counter(r["outcome"] for r in rows)
    moved = [r for r in rows if r["outcome"] in ("increase", "decrease")]
    decreases = [r for r in rows if r["outcome"] == "decrease"]
    comparable = counts["unchanged"] + counts["increase"] + counts["decrease"]
    sync = _synchronization(moved)
    mask = ((persistence["canonical_city"] == city) & (persistence[PREV] == latest[0])
            & (persistence[CUR] == latest[1]))
    persisted = persistence[mask]
    if len(persisted) != len(moved):
        raise PriceChangeReconciliationError("the final decrease case is not a subset of the persistence table")
    persist_counts = Counter(persisted["persistence"])
    reasons = Counter(r for r in persisted["not_testable_reason"] if r is not None)
    cross_rows = cross[(cross["canonical_city"] == city) & (cross[PREV] == latest[0]) & (cross[CUR] == latest[1])]
    cross_counts = Counter(cross_rows["cross_outcome"])
    provenance = Counter("|".join(sorted(_labels(r, "previous") | _labels(r, "current"))) for r in rows)
    final_flags = tuple((k, roles[k].value, latest[1] == timelines[k].periods(CaptureState.ELIGIBLE)[-1])
                        for k in involved)
    price, assortment = counts["increase"] + counts["decrease"], counts["appeared"] + counts["disappeared"]
    indicators = []
    if sync["direction_synchronized"]:
        indicators.append(I.DIRECTION_SYNCHRONIZED)
    if comparable and 2 * price > comparable:
        indicators.append(I.BROAD_CHANGE_SHARE)
    indicators.append(I.ASSORTMENT_DISCONTINUITY if assortment else I.STABLE_ASSORTMENT)
    if (case_table["interval_flag"] == IntervalFlag.EMPTY_ENDPOINT.value).any():
        indicators.append(I.EMPTY_ENDPOINT)
    if counts["ambiguous"]:
        indicators.append(I.AMBIGUITY_PRESENT)
    if int(case_table["provenance_changed_candidates"].sum()):
        indicators.append(I.PROVENANCE_CHANGE_PRESENT)
    if cross_counts[CrossLocationOutcome.SAME_DIRECTION.value]:
        indicators.append(I.CROSS_LOCATION_SAME_DIRECTION)
    if reasons[NotTestableReason.RIGHT_CENSORED_FINAL_CAPTURE.value]:
        indicators.append(I.PERSISTENCE_RIGHT_CENSORED)
    return FinalDecreaseCase(
        status=S.DERIVED, canonical_city=city, previous_period=latest[0], current_period=latest[1],
        locations=final_flags, counts=tuple((o, counts[o]) for o in _OUTCOMES), comparable=comparable,
        price_change_count=price, assortment_event_count=assortment,
        changed_share_of_comparable=_share(price, comparable),
        direction_synchronized=sync["direction_synchronized"],
        exact_cent_synchronized=sync["exact_cent_synchronized"],
        exact_percent_synchronized=sync["exact_percent_synchronized"],
        largest_same_cent_cohort=sync["largest_same_cent_cohort"],
        largest_same_percent_cohort=sync["largest_same_percent_cohort"],
        decrease_cents=_summary_stats([r["change_cents"] for r in decreases]),
        decrease_percent=_summary_stats([p for r in decreases if (p := exact_change_percent(
            r["previous_price_cents"], r["change_cents"])) is not None]),
        cross_location=tuple((o.value, cross_counts[o.value]) for o in CrossLocationOutcome
                             if cross_counts[o.value]),
        provenance=tuple(sorted(provenance.items())),
        persistence=tuple((o.value, persist_counts[o.value]) for o in PersistenceOutcome if persist_counts[o.value]),
        not_testable_reasons=tuple(sorted(reasons.items())), indicators=tuple(indicators))


def _reconcile(analysis, overall, table, persistence, cross, rows) -> None:  # type: ignore[no-untyped-def]
    """Generic reconciliation of every higher-order table against the validated event report."""
    if analysis.intervals != overall.intervals or analysis.candidates != overall.candidates:
        raise PriceChangeReconciliationError("interval or candidate totals differ from the event report")
    if analysis.price_change_count != overall.changed:
        raise PriceChangeReconciliationError("interval price changes differ from the event report")
    if analysis.assortment_event_count != overall.appeared + overall.disappeared:
        raise PriceChangeReconciliationError("interval assortment events differ from the event report")
    if analysis.ambiguous != overall.ambiguous:
        raise PriceChangeReconciliationError("interval ambiguity differs from the event report")
    for outcome in _OUTCOMES:
        if int(table[outcome].sum()) != getattr(overall, outcome):
            raise PriceChangeReconciliationError("interval outcome totals differ from the event report")
    if (table["candidates"] != table[list(_OUTCOMES)].sum(axis=1)).any():
        raise PriceChangeReconciliationError("an interval's outcomes do not sum to its candidates")
    if len(persistence) != overall.changed or persistence.duplicated(
            [*EVENT_IDENTITY_COLUMNS, PREV, CUR]).any():
        raise PriceChangeReconciliationError("persistence must partition every changed event exactly once")
    used = Counter()
    for city, airport, downtown, prev, cur, *rest in cross.itertuples(index=False, name=None):
        product = tuple(rest[:len(CROSS_LOCATION_PRODUCT_COLUMNS)])
        used[(city, airport, prev, cur, product)] += 1
        used[(city, downtown, prev, cur, product)] += 1
    if any(n > 1 for n in used.values()):
        raise PriceChangeReconciliationError("a cross-location comparison duplicates a source event")
    canonical = Counter((r["canonical_city"], r["canonical_location"], r[PREV], r[CUR], _identity(r)) for r in rows)
    if any(n > 1 for n in canonical.values()):
        raise PriceChangeReconciliationError("an alias duplicates a canonical event")


# ------------------------------------------------------------------ results and entry points


@dataclass(frozen=True)
class PriceChangeAnalysisResult:
    """The print-safe report plus in-memory tables (``None`` unless completed); tables are re-derived on build."""

    report: PriceChangeAnalysisReport
    events: PriceChangeCandidateResult | None = field(default=None, repr=False, compare=False)
    location_authority: object = field(default=None, repr=False, compare=False)
    event_table: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    cross_location: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    persistence: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.report, PriceChangeAnalysisReport):
            raise TypeError("report must be a PriceChangeAnalysisReport")
        frames = (self.event_table, self.cross_location, self.persistence)
        if not self.report.completed:
            if any(f is not None for f in frames) or self.events is not None:
                raise PriceChangeReconciliationError("a blocked result holds no tables")
            return
        if not isinstance(self.events, PriceChangeCandidateResult) or self.location_authority is None:
            raise PriceChangeReconciliationError("a completed result holds its event evidence")
        report, table, cross, persistence = _build(self.events, self.location_authority)
        if report != self.report:
            raise PriceChangeReconciliationError("the report differs from its re-derived evidence")
        for given, derived in zip(frames, (table, cross, persistence)):
            if not isinstance(given, pd.DataFrame) or tuple(given.columns) != tuple(derived.columns) \
                    or not given.equals(derived):
                raise PriceChangeReconciliationError("a higher-order table differs from its re-derived evidence")

    @property
    def completed(self) -> bool:
        return self.report.completed


def _blocked(blockers: Sequence[PriceChangeAnalysisBlocker], event_blockers: Sequence[str] = ()
             ) -> PriceChangeAnalysisResult:
    return PriceChangeAnalysisResult(PriceChangeAnalysisReport(
        status=PriceChangeAnalysisStatus.BLOCKED, blockers=tuple(dict.fromkeys(blockers)),
        event_blockers=tuple(event_blockers)))


def analyze_price_change_events(events: PriceChangeCandidateResult, *, location_authority: object
                                ) -> PriceChangeAnalysisResult:
    """Pure higher-order analysis of one completed, gated event result (see the module docstring).

    ``location_authority`` must be the exact report the event result was
    validated with. Evidence failures return a ``BLOCKED`` result; a
    contradiction inside the evidence raises.

    Raises:
        TypeError: Wrong argument types.
        PriceChangeReconciliationError: The events contradict themselves (broken chain, missing follow-up).
    """
    from ql2_sixt_canada_analysis.location_authority import LocationAuthorityReport

    if not isinstance(events, PriceChangeCandidateResult):
        raise TypeError("events must be a PriceChangeCandidateResult")
    if not isinstance(location_authority, LocationAuthorityReport):
        raise TypeError("location_authority must be a LocationAuthorityReport")
    B = PriceChangeAnalysisBlocker
    if not events.completed:
        return _blocked([B.EVENTS_NOT_COMPLETED], [b.value for b in events.report.blockers])
    if events.binding is None or events.location_authority is None:
        return _blocked([B.EVENT_EVIDENCE_UNBOUND])
    if events.location_authority is not location_authority:
        return _blocked([B.EVIDENCE_MISMATCH])
    if not location_authority.roles_exact:
        return _blocked([B.ROLE_AUTHORITY_UNAVAILABLE])
    expected = tuple(dict.fromkeys(tuple(location_authority.canonical(k))
                                   for k in location_authority.contract.expected_keys))
    if expected != events.report.approved_locations:
        return _blocked([B.EVIDENCE_MISMATCH])
    report, table, cross, persistence = _build(events, location_authority)
    return PriceChangeAnalysisResult(report=report, events=events, location_authority=location_authority,
                                     event_table=table, cross_location=cross, persistence=persistence)


def price_change_analysis_from_pipeline(run: object) -> PriceChangeAnalysisResult:
    """Events and higher-order analysis from one pipeline result (the same object throughout)."""
    from ql2_sixt_canada_analysis.price_change_events import price_change_candidates_from_pipeline
    from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult

    if not isinstance(run, PricingPipelineResult):
        raise TypeError("run must be a PricingPipelineResult")
    events = price_change_candidates_from_pipeline(run)
    return _analyze_bound(run, events)


def _analyze_bound(run, events: PriceChangeCandidateResult) -> PriceChangeAnalysisResult:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis.pricing_population import frame_binding

    B = PriceChangeAnalysisBlocker
    if not events.completed:
        return _blocked([B.EVENTS_NOT_COMPLETED], [b.value for b in events.report.blockers])
    if events.binding is None or events.binding != frame_binding(run.jobs, run.cars) \
            or events.location_authority is not run.location_authority:
        return _blocked([B.EVIDENCE_MISMATCH])
    try:
        return analyze_price_change_events(events, location_authority=run.location_authority)
    except PriceChangeReconciliationError:
        return _blocked([B.RECONCILIATION_FAILED])


def run_price_change_analysis(raw_dir: str | Path | None = None) -> PriceChangeAnalysisResult:
    """Run ``run_pricing_pipeline`` exactly once, then the event engine and the higher-order analysis on it."""
    from ql2_sixt_canada_analysis import pricing_pipeline

    return price_change_analysis_from_pipeline(pricing_pipeline.run_pricing_pipeline(raw_dir))


# ------------------------------------------------------------------ heatmap and outputs


def _require_completed(result: object) -> PriceChangeAnalysisResult:
    if not isinstance(result, PriceChangeAnalysisResult):
        raise TypeError("expected a PriceChangeAnalysisResult")
    if not result.completed:
        raise PriceChangeReconciliationError("the analysis is blocked: "
                                             + ", ".join(b.value for b in result.report.blockers))
    return result


def event_heatmap_source(result: PriceChangeAnalysisResult) -> dict:
    """Heatmap matrices over every scheduled period of every approved location, reconciled before plotting.

    Returns ``locations`` (authority order), ``periods`` (sorted canonical UTC
    current periods), ``increase``/``decrease`` count matrices and a
    ``state`` matrix (``interval``, ``break`` - a scheduled capture with no
    eligible interval ending there - or ``outside`` the location's schedule).
    Non-interval cells are ``NaN`` so a break never reads as zero activity.
    """
    import numpy as np

    result = _require_completed(result)
    table = result.event_table
    timelines = {t.canonical_location: t for t in result.events.timelines}
    locations = list(result.report.locations)
    keys = [s.canonical_location for s in locations]
    periods = sorted({c.period for k in keys for c in timelines[k].captures}, key=parse_scheduled_period)
    column = {p: j for j, p in enumerate(periods)}
    increase = np.full((len(keys), len(periods)), np.nan)
    decrease = np.full((len(keys), len(periods)), np.nan)
    state = np.full((len(keys), len(periods)), "outside", dtype=object)
    for i, key in enumerate(keys):
        for capture in timelines[key].captures:
            state[i, column[capture.period]] = "break"
        rows = table[(table["canonical_city"] == key[0]) & (table["canonical_location"] == key[1])]
        for cur, inc, dec in zip(rows[CUR], rows["increase"], rows["decrease"]):
            j = column[cur]
            increase[i, j], decrease[i, j], state[i, j] = inc, dec, "interval"
    if (int(np.nansum(increase)) != int(table["increase"].sum())
            or int(np.nansum(decrease)) != int(table["decrease"].sum())
            or int(np.nansum(increase) + np.nansum(decrease)) != result.report.price_change_count
            or int((state == "interval").sum()) != result.report.intervals):
        raise PriceChangeReconciliationError("the heatmap source does not reconcile with the event table")
    labels = [f"{k[0].title()} / {k[1]} ({s.role.lower()})" for k, s in zip(keys, locations)]
    return {"locations": keys, "labels": labels, "periods": periods, "increase": increase, "decrease": decrease,
            "state": state}


def plot_price_change_heatmap(result: PriceChangeAnalysisResult, *, dpi: int = 150):  # type: ignore[no-untyped-def]
    """Two-panel heatmap (increases, decreases) per approved location and eligible interval.

    Separate panels keep increases and decreases from cancelling; cells with no
    eligible interval (hard breaks, outside the schedule) are hatched grey, never
    zero. Object-oriented :class:`matplotlib.figure.Figure` (no pyplot state).
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    source = event_heatmap_source(result)
    periods, labels = source["periods"], source["labels"]
    peak = max(1.0, float(np.nanmax(np.concatenate([source["increase"].ravel(), source["decrease"].ravel(),
                                                     [0.0]]))))
    width = min(max(11.0, 4.0 + 0.16 * len(periods)), 30.0)
    fig = Figure(figsize=(width, 2.2 + 0.9 * len(labels)), dpi=dpi, facecolor="white", layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    panels = ((axes[0], source["increase"], "#2a78d6", "Increases per interval (observed candidates)"),
              (axes[1], source["decrease"], "#eb6834", "Decreases per interval (observed candidates)"))
    step = max(1, len(periods) // 16)
    for ax, matrix, colour, title in panels:
        cmap = LinearSegmentedColormap.from_list("one_hue", ["#f7f6f3", colour])
        ax.set_facecolor("#d9d8d4")
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0, vmax=peak,
                          interpolation="nearest")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if source["state"][i, j] != "interval":
                    ax.add_patch(_hatch(j, i))
                elif matrix[i, j] > 0 and len(periods) <= 120:
                    ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=6,
                            color="#0b0b0b" if matrix[i, j] < 0.6 * peak else "white")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks([i + 0.5 for i in range(len(labels) - 1)], minor=True)
        ax.grid(which="minor", axis="y", color="white", linewidth=2.0)
        ax.tick_params(which="minor", left=False)
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        from matplotlib.ticker import MaxNLocator

        bar = fig.colorbar(image, ax=ax, shrink=0.8, label="count")
        bar.locator = MaxNLocator(integer=True)
        bar.update_ticks()
    axes[1].set_xticks(range(0, len(periods), step), [periods[j] for j in range(0, len(periods), step)],
                       rotation=60, ha="right", fontsize=7)
    axes[1].set_xlabel("Interval current scheduled capture period (UTC)", color="#52514e")
    fig.suptitle("Observed price-change candidates by approved location and eligible interval", x=0.01,
                 ha="left", fontsize=12)
    # (the legend is placed outside the axes, above the panels, by the constrained layout)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(facecolor="#d9d8d4", edgecolor="#9a9994", hatch="///",
                              label="No eligible interval (masked break)"),
                        Patch(facecolor="#f7f6f3", edgecolor="#9a9994", label="Eligible interval, zero changes")],
               loc="outside upper right", ncols=2, fontsize=8, frameon=False)
    import textwrap

    note = ("Both panels share one count scale. Hatched grey: no eligible interval ending at that capture "
            "(governed exclusion, missing capture, non-one-hour gap, source-stream change, first capture or "
            "outside the schedule) - not zero activity.")
    fig.supxlabel("\n".join(textwrap.wrap(note, int(width * 14))), fontsize=8, color="#52514e", x=0.01,
                  ha="left")
    return fig, (axes[0], axes[1])


def _hatch(j: int, i: int):  # type: ignore[no-untyped-def]
    from matplotlib.patches import Rectangle

    return Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#d9d8d4", edgecolor="#9a9994", hatch="///",
                     linewidth=0)


def _atomic_write(path: Path, write, *, mode: int | None = None, create_parents: bool = True) -> None:  # type: ignore[no-untyped-def]
    """Write through a temporary file in the target directory, then replace atomically (optional ``chmod``)."""
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        write(Path(temporary))
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_price_change_analysis_outputs(result: PriceChangeAnalysisResult, output_dir: str | Path
                                        ) -> tuple[Path, Path]:
    """Write the aggregate event table (CSV) and the heatmap (PNG) into ``output_dir`` (explicit; atomic).

    Proprietary local artifacts: only aggregate counts, flags and statistics per
    location interval (no product values or event-level prices). Written only
    after the heatmap source reconciles with the table.
    """
    result = _require_completed(result)
    if output_dir is None:
        raise TypeError("an explicit output directory is required")
    directory = Path(output_dir)
    fig, _ = plot_price_change_heatmap(result)
    table_path = directory / "price_change_event_table.csv"
    figure_path = directory / "price_change_event_heatmap.png"
    _atomic_write(table_path, lambda p: result.event_table.to_csv(p, index=False))
    _atomic_write(figure_path, lambda p: fig.savefig(p, format="png", dpi=fig.dpi, metadata={"Software": None}))
    fig.clear()
    return table_path, figure_path
