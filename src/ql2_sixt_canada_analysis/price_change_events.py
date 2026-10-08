"""The price-change event contract: observed price-change candidates between adjacent scheduled captures.

This module locks the contract of issue #4 (price-change events). It turns the
pricing-eligible canonical offers of :func:`~ql2_sixt_canada_analysis.pricing_pipeline.run_pricing_pipeline`
into **observed price-change candidates**: one candidate per offer identity per
eligible one-hour capture interval, each with exactly one terminal outcome. A
candidate is an observation of the collected feed, never a proven genuine market
event: deciding that requires source and operational corroboration outside this
contract.

Population
----------
Only :attr:`CanonicalOfferReport.offers <ql2_sixt_canada_analysis.canonical_offers.CanonicalOfferReport>`
of the pricing-eligible population (the governed Calgary
``INCOMPLETE_PARENT_CAPTURE`` exclusion and every ineligible row never enter;
the Vancouver ``Downtown``/``Thurlow`` aliases are already one canonical
location). Never ``MatchedLocationPricingResult.pairs``: that table holds only the
airport/downtown shared assortment and would drop one-sided products.

Offer identity (:data:`EVENT_IDENTITY_COLUMNS`)
-----------------------------------------------
``canonical_city, canonical_location, pickup_date, return_date``, every column of
:data:`~ql2_sixt_canada_analysis.canonical_offers.APPROVED_PRODUCT_COLUMNS`
(``car_name, car_type, transmission, seats, bags``), ``currency, price_basis``.
Values are the canonical-offer assessment's parsed values, compared by exact
equality: no trimming, recasing, fuzzy matching, imputation or inference. Price
is never part of the identity. Raw source location labels are provenance only
(kept per endpoint for diagnostics) and never split the identity.

Canonical timestamp (:data:`EVENT_TIMESTAMP_COLUMN`)
----------------------------------------------------
``scheduled_capture_period`` from the authority-backed
:class:`~ql2_sixt_canada_analysis.collection_schedule.CapturePeriodIndex`: the
trusted UTC start of the scheduled hourly capture, ``YYYYMMDDTHHMMSSZ``, parsed
strictly by :func:`parse_scheduled_period`. Never raw ``job_id``, row order,
``scrape_date``, ``date_clean``, raw scrape or finish timestamps, the reporting
day, or the last time a product happened to appear.

Exact one-hour adjacency
------------------------
Capture intervals come from the per-stream schedule and its assessed coverage
(:func:`capture_timelines`), independently of which products are visible. For
each canonical location every scheduled period is ``eligible`` (every source
stream scheduling it is covered), a ``governed_exclusion`` or a
``missing_capture``. A :class:`CaptureInterval` joins two schedule-adjacent
eligible periods exactly one hour apart with the same contributing source
streams; every other adjacent pair is a typed :class:`IntervalBreak`. "Previous"
therefore means the immediately preceding scheduled capture, never the last
earlier observation of the product: a product absent at ``t-1`` and present at
``t`` *appeared* at ``t``; a governed exclusion or a missing capture is a hard
break - nothing is compared across it and no appearance or disappearance is
attributed to it.

Terminal outcomes (:class:`TerminalOutcome`)
-------------------------------------------
For every interval and every identity present at either endpoint, exactly one
outcome, first applicable wins:

1. ``ambiguous`` - more than one canonical offer for the identity at either
   endpoint (price-distinct offers with no authority-backed way to choose; never
   resolved by row order, minimum, maximum, mean, median, first, last or a
   Cartesian product; no comparison price is exposed);
2. ``appeared`` - absent previously, exactly one offer currently;
3. ``disappeared`` - exactly one offer previously, absent currently;
4. ``unchanged`` / ``increase`` / ``decrease`` - exactly one offer at both
   endpoints, by the sign of ``current price_cents - previous price_cents``.

A currency or price-basis change is never a price movement: the unit is part of
the identity, so the old unit disappears and the new unit appears. Nothing is
converted or normalized.

Prices
------
Exact non-negative integer ``price_cents``. Change direction is current minus
previous; ``unchanged`` is a zero-cent difference. The (later) percentage change
uses the previous price as its denominator: a zero previous price makes it
undefined and is counted separately (``zero_baseline``), never infinite.
Display rounding never affects classification.

Confidentiality
---------------
The candidate frame (:attr:`PriceChangeCandidateResult.candidates`) holds
proprietary event-level values: it stays in memory, is excluded from ``repr``
and is never written by this module. :class:`PriceChangeCandidateReport` holds
counts, enum values and approved configuration keys only.

Deliberately deferred to later issue #4 phases: synchronized-movement detection,
persistence, the Vancouver decrease case study, alert thresholds, anomaly
scoring, monitoring rules, heatmaps, final event tables and notebook 03.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.canonical_offers import APPROVED_PRODUCT_COLUMNS

__all__ = [
    "CANDIDATE_COLUMNS",
    "CAPTURE_STEP",
    "EVENT_IDENTITY_COLUMNS",
    "EVENT_INTERVAL_COLUMNS",
    "EVENT_KEY_COLUMNS",
    "EVENT_TIMESTAMP_COLUMN",
    "EVENT_UNIT_COLUMNS",
    "FORBIDDEN_TIMESTAMP_SOURCES",
    "SCHEDULED_PERIOD_FORMAT",
    "CaptureEvidenceError",
    "CaptureInterval",
    "CaptureState",
    "IntervalBreak",
    "LocationCaptureTimeline",
    "LocationPriceChangeSummary",
    "OutcomeCounts",
    "PriceChangeBlocker",
    "PriceChangeCandidateReport",
    "PriceChangeCandidateResult",
    "PriceChangeContractError",
    "PriceChangeStatus",
    "ScheduledCapture",
    "TerminalOutcome",
    "assess_price_change_candidates",
    "capture_timelines",
    "classify_endpoint_offers",
    "classify_price_change_candidates",
    "parse_scheduled_period",
    "price_change_candidates_from_pipeline",
    "price_change_cents",
    "validate_candidate_frame",
]

Key = tuple[str, ...]

#: The canonical event timestamp: the trusted UTC start of the scheduled hourly capture period.
EVENT_TIMESTAMP_COLUMN = "scheduled_capture_period"
#: Exact text form of :data:`EVENT_TIMESTAMP_COLUMN` (UTC, ``YYYYMMDDTHHMMSSZ``).
SCHEDULED_PERIOD_FORMAT = "%Y%m%dT%H%M%SZ"
#: The only comparable distance between two captures.
CAPTURE_STEP = dt.timedelta(hours=1)
#: The price unit: part of the identity, so prices of unlike units are never compared.
EVENT_UNIT_COLUMNS: tuple[str, ...] = ("currency", "price_basis")
#: The exact identity two captures must share before their prices are compared (price is never part of it).
EVENT_IDENTITY_COLUMNS: tuple[str, ...] = ("canonical_city", "canonical_location", "pickup_date", "return_date",
                                           *APPROVED_PRODUCT_COLUMNS, *EVENT_UNIT_COLUMNS)
#: The endpoints of a one-hour capture interval (scheduled periods).
EVENT_INTERVAL_COLUMNS: tuple[str, ...] = ("previous_period", "current_period")
#: The unique key of a candidate: one identity in one capture interval.
EVENT_KEY_COLUMNS: tuple[str, ...] = (*EVENT_IDENTITY_COLUMNS, *EVENT_INTERVAL_COLUMNS)
#: Columns of the in-memory candidate frame (proprietary; never printed or written here).
CANDIDATE_COLUMNS: tuple[str, ...] = (*EVENT_KEY_COLUMNS, "outcome", "previous_offer_count", "current_offer_count",
                                      "previous_price_cents", "current_price_cents", "change_cents", "zero_baseline",
                                      "previous_source_labels", "current_source_labels")
#: Fields that must never order, place or identify a capture (and never appear in a candidate).
FORBIDDEN_TIMESTAMP_SOURCES: tuple[str, ...] = ("job_id", "row_index", "scrape_date", "date_clean", "scraped_at",
                                                "finished_at", "job_finished_at", "reporting_day")

_PERIOD = re.compile(r"[0-9]{8}T[0-9]{6}Z", re.ASCII)
_UTC = dt.timezone.utc
_OFFER_COLUMNS = (*EVENT_IDENTITY_COLUMNS, EVENT_TIMESTAMP_COLUMN, "price_cents", "source_location_labels")


class PriceChangeContractError(ValueError):
    """Inputs or results violate the price-change event contract (never silently repaired)."""


class CaptureEvidenceError(PriceChangeContractError):
    """The schedule, coverage and offer evidence disagree (stale, malformed or inconsistent)."""


class TerminalOutcome(StrEnum):
    """Exactly one per candidate (``ambiguous`` takes precedence over every other outcome)."""

    UNCHANGED = "unchanged"
    INCREASE = "increase"
    DECREASE = "decrease"
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    AMBIGUOUS = "ambiguous"


#: Outcomes that compare exactly one previous and one current price.
_PRICED = frozenset({TerminalOutcome.UNCHANGED, TerminalOutcome.INCREASE, TerminalOutcome.DECREASE})


class CaptureState(StrEnum):
    """What one scheduled period of one canonical location is to the event contract."""

    ELIGIBLE = "eligible"                       # every source stream scheduling it is covered
    GOVERNED_EXCLUSION = "governed_exclusion"   # INCOMPLETE_PARENT_CAPTURE: analytically null
    MISSING_CAPTURE = "missing_capture"         # missing for a source stream (excused or not)


class IntervalBreak(StrEnum):
    """Why two schedule-adjacent periods form no capture interval (first applicable wins)."""

    GOVERNED_EXCLUSION = "governed_exclusion"
    MISSING_CAPTURE = "missing_capture"
    NOT_ONE_HOUR = "not_one_hour"
    SOURCE_STREAMS_CHANGED = "source_streams_changed"


class PriceChangeStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class PriceChangeBlocker(StrEnum):
    """Why no candidates were produced (categories only)."""

    PRICING_NOT_READY = "pricing_not_ready"
    READINESS_EVIDENCE_MISMATCH = "readiness_evidence_mismatch"
    FRAME_BINDING_MISMATCH = "frame_binding_mismatch"
    CANONICAL_OFFERS_NOT_READY = "canonical_offers_not_ready"
    SCHEDULE_EVIDENCE_INVALID = "schedule_evidence_invalid"
    CAPTURE_EVIDENCE_INCONSISTENT = "capture_evidence_inconsistent"
    OFFER_CONTRACT_INVALID = "offer_contract_invalid"


# ------------------------------------------------------------------ timestamps and prices


def parse_scheduled_period(value: object) -> dt.datetime:
    """The aware UTC instant of an exact ``YYYYMMDDTHHMMSSZ`` scheduled period (fails closed).

    Only a ``str`` in the canonical form is accepted: no whitespace, no other
    separators or offsets, no ``Timestamp``/``datetime`` objects, and the text
    must round-trip exactly.

    Raises:
        PriceChangeContractError: Missing, non-string, malformed or non-canonical value.
    """
    if not isinstance(value, str) or not _PERIOD.fullmatch(value):
        raise PriceChangeContractError("a scheduled period must be YYYYMMDDTHHMMSSZ text")
    try:
        instant = dt.datetime.strptime(value, SCHEDULED_PERIOD_FORMAT).replace(tzinfo=_UTC)
    except ValueError:
        raise PriceChangeContractError("a scheduled period must be a valid UTC instant") from None
    if instant.strftime(SCHEDULED_PERIOD_FORMAT) != value:
        raise PriceChangeContractError("a scheduled period must be canonical")
    return instant


def _cents(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise PriceChangeContractError("prices are exact non-negative integer cents")
    return int(value)


def price_change_cents(previous: object, current: object) -> int:
    """Exact ``current - previous`` in integer cents (direction: positive is an increase)."""
    return _cents(current) - _cents(previous)


def classify_endpoint_offers(previous: Sequence[int], current: Sequence[int]) -> TerminalOutcome:
    """The terminal outcome of one identity from its canonical offer prices at both endpoints.

    ``previous``/``current`` are the exact cents of every canonical offer of the
    identity at that endpoint (empty when absent). Ambiguity wins; nothing is
    ever chosen among several offers.

    Raises:
        PriceChangeContractError: The identity is absent at both endpoints, or a price is not exact cents.
    """
    previous, current = [_cents(v) for v in previous], [_cents(v) for v in current]
    if not previous and not current:
        raise PriceChangeContractError("a candidate needs an offer at one endpoint at least")
    if len(previous) > 1 or len(current) > 1:
        return TerminalOutcome.AMBIGUOUS
    if not previous:
        return TerminalOutcome.APPEARED
    if not current:
        return TerminalOutcome.DISAPPEARED
    change = current[0] - previous[0]
    return (TerminalOutcome.UNCHANGED if change == 0
            else TerminalOutcome.INCREASE if change > 0 else TerminalOutcome.DECREASE)


# ------------------------------------------------------------------ capture intervals


def _location(value: object) -> Key:
    if (not isinstance(value, tuple) or len(value) != 2
            or not all(isinstance(v, str) and v and v == v.strip() for v in value)):
        raise PriceChangeContractError("a canonical location is an exact (city, location) key")
    return value


@dataclass(frozen=True, slots=True)
class CaptureInterval:
    """Two consecutive eligible scheduled captures of one canonical location, exactly one hour apart."""

    canonical_location: Key
    previous_period: str
    current_period: str

    def __post_init__(self) -> None:
        _location(self.canonical_location)
        previous, current = parse_scheduled_period(self.previous_period), parse_scheduled_period(self.current_period)
        if current - previous != CAPTURE_STEP:
            raise PriceChangeContractError("an interval joins captures exactly one hour apart, previous first")


@dataclass(frozen=True, slots=True)
class ScheduledCapture:
    """One scheduled period of one canonical location: its state and the source streams scheduling it."""

    period: str
    state: CaptureState
    source_streams: tuple[Key, ...]

    def __post_init__(self) -> None:
        parse_scheduled_period(self.period)
        if not isinstance(self.state, CaptureState):
            raise PriceChangeContractError("state must be a CaptureState")
        streams = self.source_streams
        if not isinstance(streams, tuple) or not streams or tuple(sorted(set(streams))) != streams:
            raise PriceChangeContractError("source streams are a sorted, distinct, non-empty tuple")
        for stream in streams:
            _location(stream)


@dataclass(frozen=True)
class LocationCaptureTimeline:
    """Every scheduled period of one canonical location, in time order, and the intervals they allow.

    Built from the schedule and its assessed coverage (:func:`capture_timelines`),
    never from offers. Schedule-adjacent pairs form a :class:`CaptureInterval`
    only when both are eligible, exactly one hour apart and contributed by the
    same source streams; every other pair is a typed :class:`IntervalBreak`.
    """

    canonical_location: Key
    captures: tuple[ScheduledCapture, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _location(self.canonical_location)
        if not isinstance(self.captures, tuple) or not all(isinstance(c, ScheduledCapture) for c in self.captures):
            raise PriceChangeContractError("captures must be ScheduledCapture objects")
        instants = [parse_scheduled_period(c.period) for c in self.captures]
        if any(b <= a for a, b in zip(instants, instants[1:])):
            raise PriceChangeContractError("captures are distinct and in strict time order")
        if any(s[0] != self.canonical_location[0] for c in self.captures for s in c.source_streams):
            raise PriceChangeContractError("a canonical location combines source streams of its own city only")

    @cached_property
    def _pairs(self) -> tuple[tuple[CaptureInterval, ...], tuple[IntervalBreak, ...]]:
        intervals: list[CaptureInterval] = []
        breaks: list[IntervalBreak] = []
        for a, b in zip(self.captures, self.captures[1:]):
            states = {a.state, b.state}
            if CaptureState.GOVERNED_EXCLUSION in states:
                breaks.append(IntervalBreak.GOVERNED_EXCLUSION)
            elif CaptureState.MISSING_CAPTURE in states:
                breaks.append(IntervalBreak.MISSING_CAPTURE)
            elif parse_scheduled_period(b.period) - parse_scheduled_period(a.period) != CAPTURE_STEP:
                breaks.append(IntervalBreak.NOT_ONE_HOUR)
            elif a.source_streams != b.source_streams:
                breaks.append(IntervalBreak.SOURCE_STREAMS_CHANGED)
            else:
                intervals.append(CaptureInterval(self.canonical_location, a.period, b.period))
        return tuple(intervals), tuple(breaks)

    @property
    def intervals(self) -> tuple[CaptureInterval, ...]:
        return self._pairs[0]

    @property
    def break_counts(self) -> tuple[tuple[str, int], ...]:
        breaks = self._pairs[1]
        return tuple((b.value, breaks.count(b)) for b in IntervalBreak if b in breaks)

    def periods(self, state: CaptureState) -> tuple[str, ...]:
        return tuple(c.period for c in self.captures if c.state is state)

    @property
    def source_streams(self) -> tuple[Key, ...]:
        return tuple(sorted({s for c in self.captures for s in c.source_streams}))


def capture_timelines(scheduled: object, policy: object) -> tuple[LocationCaptureTimeline, ...]:
    """One timeline per canonical location from the per-stream schedule and its assessed coverage.

    Each stream's periods are its materialized schedule; a period is excluded
    or missing exactly as the coverage report recorded it, otherwise covered.
    A canonical location's period is a governed exclusion if any contributing
    stream has it excluded, missing if any has it missing, else eligible.
    Offers are never read, so adjacency is independent of product presence.

    Raises:
        TypeError: Wrong argument types.
        CaptureEvidenceError: The schedule is unavailable or the coverage disagrees with it.
    """
    from ql2_sixt_canada_analysis.canonical_offers import CanonicalOfferPolicy
    from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport

    if not isinstance(scheduled, PerStreamScheduledCoverageReport):
        raise TypeError("scheduled must be a PerStreamScheduledCoverageReport")
    if not isinstance(policy, CanonicalOfferPolicy):
        raise TypeError("policy must be a CanonicalOfferPolicy")
    schedule = scheduled.schedule
    if not schedule.available:
        raise CaptureEvidenceError("the schedule is not available")
    coverage = {c.stream: c for c in scheduled.streams}
    if len(coverage) != len(scheduled.streams) or set(coverage) != set(schedule.expected_streams):
        raise CaptureEvidenceError("coverage must name every scheduled stream exactly once")
    by_location: dict[Key, dict[str, list[tuple[Key, CaptureState]]]] = {}
    for stream_schedule in schedule.schedules:
        stream, report = stream_schedule.stream, coverage[stream_schedule.stream]
        nominal = [p.utc_text for p in stream_schedule.periods]
        excluded = {e.period.utc_text for e in report.excluded}
        missing = {m.period.utc_text for m in report.missing}
        if (report.expected != len(nominal) or len(excluded) != len(report.excluded)
                or len(missing) != len(report.missing) or excluded & missing
                or not (excluded | missing) <= set(nominal)
                or report.covered != len(nominal) - len(excluded) - len(missing)):
            raise CaptureEvidenceError("a stream's coverage disagrees with its schedule")
        location = tuple(policy.canonical(stream))
        for text in nominal:
            state = (CaptureState.GOVERNED_EXCLUSION if text in excluded
                     else CaptureState.MISSING_CAPTURE if text in missing else CaptureState.ELIGIBLE)
            by_location.setdefault(location, {}).setdefault(text, []).append((stream, state))
    timelines = []
    for location in sorted(by_location):
        captures = []
        for text in sorted(by_location[location], key=parse_scheduled_period):
            members = by_location[location][text]
            states = {s for _, s in members}
            state = next(s for s in (CaptureState.GOVERNED_EXCLUSION, CaptureState.MISSING_CAPTURE,
                                     CaptureState.ELIGIBLE) if s in states)
            captures.append(ScheduledCapture(text, state, tuple(sorted({k for k, _ in members}))))
        timelines.append(LocationCaptureTimeline(location, tuple(captures)))
    return tuple(timelines)


# ------------------------------------------------------------------ results


def _count(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise PriceChangeContractError(f"{name} must be a non-negative int")


@dataclass(frozen=True, slots=True)
class OutcomeCounts:
    """Candidate accounting: every candidate has exactly one terminal outcome."""

    candidates: int = 0
    unchanged: int = 0
    increase: int = 0
    decrease: int = 0
    appeared: int = 0
    disappeared: int = 0
    ambiguous: int = 0
    #: Priced candidates whose previous price is zero cents (a future percentage change is undefined).
    zero_baseline: int = 0

    def __post_init__(self) -> None:
        for name in self.__slots__:
            _count(getattr(self, name), name)
        if self.candidates != sum(getattr(self, o.value) for o in TerminalOutcome):
            raise PriceChangeContractError("candidates must equal the sum of terminal outcomes")
        if self.zero_baseline > self.priced:
            raise PriceChangeContractError("a zero baseline needs a priced comparison")

    @property
    def priced(self) -> int:
        """Candidates with exactly one offer at both endpoints (unchanged, increase or decrease)."""
        return self.unchanged + self.increase + self.decrease

    def __add__(self, other: OutcomeCounts) -> OutcomeCounts:
        return OutcomeCounts(*(getattr(self, n) + getattr(other, n) for n in self.__slots__))

    @classmethod
    def of(cls, outcomes: Sequence[TerminalOutcome], zero_baseline: int = 0) -> OutcomeCounts:
        values = [TerminalOutcome(o) for o in outcomes]
        return cls(len(values), *(values.count(o) for o in TerminalOutcome), zero_baseline=zero_baseline)


@dataclass(frozen=True, slots=True)
class LocationPriceChangeSummary:
    """One canonical location: its scheduled-period states, intervals, breaks and outcome counts."""

    canonical_location: Key
    source_streams: tuple[Key, ...]
    scheduled_periods: int
    eligible_periods: int
    excluded_periods: int
    missing_periods: int
    intervals: int
    breaks: tuple[tuple[str, int], ...]
    counts: OutcomeCounts

    def __post_init__(self) -> None:
        _location(self.canonical_location)
        for name in ("scheduled_periods", "eligible_periods", "excluded_periods", "missing_periods", "intervals"):
            _count(getattr(self, name), name)
        if self.scheduled_periods != self.eligible_periods + self.excluded_periods + self.missing_periods:
            raise PriceChangeContractError("every scheduled period is eligible, excluded or missing")
        if any(r not in {b.value for b in IntervalBreak} or n <= 0 for r, n in self.breaks):
            raise PriceChangeContractError("breaks are typed positive counts")
        if self.intervals + sum(n for _, n in self.breaks) != max(self.scheduled_periods - 1, 0):
            raise PriceChangeContractError("every adjacent scheduled pair is an interval or a break")
        if not isinstance(self.counts, OutcomeCounts) or (self.intervals == 0 and self.counts.candidates):
            raise PriceChangeContractError("candidates exist only within intervals")


@dataclass(frozen=True, slots=True)
class PriceChangeCandidateReport:
    """Aggregate, print-safe result: status, blockers, approved location keys and counts only."""

    status: PriceChangeStatus
    blockers: tuple[PriceChangeBlocker, ...] = ()
    readiness_blockers: tuple[str, ...] = ()
    offers_assessed: int = 0
    locations: tuple[LocationPriceChangeSummary, ...] = ()
    overall: OutcomeCounts | None = None

    def __post_init__(self) -> None:
        _count(self.offers_assessed, "offers_assessed")
        if not isinstance(self.status, PriceChangeStatus):
            raise PriceChangeContractError("status must be a PriceChangeStatus")
        if self.status is PriceChangeStatus.BLOCKED:
            if not self.blockers or self.locations or self.overall is not None:
                raise PriceChangeContractError("a blocked report has blockers and no result")
            return
        if self.blockers or self.readiness_blockers or not isinstance(self.overall, OutcomeCounts):
            raise PriceChangeContractError("a completed report has no blockers and overall counts")
        keys = [s.canonical_location for s in self.locations]
        if keys != sorted(set(keys)):
            raise PriceChangeContractError("one summary per canonical location, in key order")
        if sum((s.counts for s in self.locations), OutcomeCounts()) != self.overall:
            raise PriceChangeContractError("location counts must sum to the overall counts")

    @property
    def completed(self) -> bool:
        return self.status is PriceChangeStatus.COMPLETED

    @property
    def intervals(self) -> int:
        return sum(s.intervals for s in self.locations)


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA or value is pd.NaT


def validate_candidate_frame(candidates: pd.DataFrame, report: PriceChangeCandidateReport,
                             timelines: Sequence[LocationCaptureTimeline]) -> None:
    """Enforce every row-level invariant of a candidate frame against its report and timelines.

    Raises:
        PriceChangeContractError: Any violation.
    """
    if not isinstance(candidates, pd.DataFrame) or tuple(candidates.columns) != CANDIDATE_COLUMNS:
        raise PriceChangeContractError("the candidate frame has exactly the contract columns")
    overall = report.overall
    if overall is None or len(candidates) != overall.candidates:
        raise PriceChangeContractError("the candidate frame must equal the candidate count")
    if candidates.duplicated(list(EVENT_KEY_COLUMNS)).any():
        raise PriceChangeContractError("the event key must be unique")
    allowed = {(i.canonical_location, i.previous_period, i.current_period) for t in timelines for i in t.intervals}
    outcomes: list[TerminalOutcome] = []
    zero = 0
    for row in candidates.itertuples(index=False, name=None):
        values = dict(zip(CANDIDATE_COLUMNS, row))
        if any(_missing(values[c]) for c in EVENT_KEY_COLUMNS):
            raise PriceChangeContractError("every candidate carries the full identity and interval")
        location = (values["canonical_city"], values["canonical_location"])
        CaptureInterval(location, values["previous_period"], values["current_period"])
        if (location, values["previous_period"], values["current_period"]) not in allowed:
            raise PriceChangeContractError("a candidate lies outside every eligible capture interval")
        outcome = TerminalOutcome(values["outcome"])
        n_prev, n_cur = values["previous_offer_count"], values["current_offer_count"]
        _count(n_prev, "previous_offer_count"), _count(n_cur, "current_offer_count")
        prev, cur, change = values["previous_price_cents"], values["current_price_cents"], values["change_cents"]
        if outcome is TerminalOutcome.AMBIGUOUS:
            ok = (n_prev > 1 or n_cur > 1) and prev is None and cur is None and change is None
        elif outcome is TerminalOutcome.APPEARED:
            ok = (n_prev, n_cur) == (0, 1) and prev is None and change is None and _cents(cur) >= 0
        elif outcome is TerminalOutcome.DISAPPEARED:
            ok = (n_prev, n_cur) == (1, 0) and cur is None and change is None and _cents(prev) >= 0
        else:
            ok = ((n_prev, n_cur) == (1, 1) and classify_endpoint_offers([prev], [cur]) is outcome
                  and change == price_change_cents(prev, cur))
        if not ok:
            raise PriceChangeContractError("a candidate's offers and prices contradict its outcome")
        baseline = values["zero_baseline"]
        if not isinstance(baseline, (bool, np.bool_)) or bool(baseline) != (outcome in _PRICED and prev == 0):
            raise PriceChangeContractError("zero_baseline marks exactly the priced candidates with a zero baseline")
        zero += bool(baseline)
        outcomes.append(outcome)
    if OutcomeCounts.of(outcomes, zero) != overall:
        raise PriceChangeContractError("the candidate frame contradicts the outcome counts")


@dataclass(frozen=True)
class PriceChangeCandidateResult:
    """The aggregate report plus the proprietary in-memory candidate frame (``None`` unless completed)."""

    report: PriceChangeCandidateReport
    candidates: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    timelines: tuple[LocationCaptureTimeline, ...] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.report, PriceChangeCandidateReport):
            raise TypeError("report must be a PriceChangeCandidateReport")
        if not self.report.completed:
            if self.candidates is not None or self.timelines is not None:
                raise PriceChangeContractError("a blocked result holds no candidates")
            return
        if not isinstance(self.timelines, tuple):
            raise PriceChangeContractError("a completed result holds its capture timelines")
        validate_candidate_frame(self.candidates, self.report, self.timelines)

    @property
    def completed(self) -> bool:
        return self.report.completed


# ------------------------------------------------------------------ classification


def _validate_offers(offers: pd.DataFrame) -> None:
    if not isinstance(offers, pd.DataFrame) or any(c not in offers.columns for c in _OFFER_COLUMNS):
        raise PriceChangeContractError("the canonical offers lack a contract column")
    for column in EVENT_IDENTITY_COLUMNS:
        for value in offers[column].astype(object):
            if column in ("pickup_date", "return_date"):
                if type(value) is not dt.date:
                    raise PriceChangeContractError("rental dates are parsed dates")
            elif not isinstance(value, str) or not value or value != value.strip():
                raise PriceChangeContractError("identity values are exact non-empty text")
    for value in offers[EVENT_TIMESTAMP_COLUMN].astype(object):
        parse_scheduled_period(value)
    for value in offers["price_cents"].astype(object):
        _cents(value)
    if offers.duplicated([*EVENT_IDENTITY_COLUMNS, EVENT_TIMESTAMP_COLUMN, "price_cents"]).any():
        raise PriceChangeContractError("canonical offers are unique by identity, period and price")


def _labels(members: list[tuple[int, str]]) -> str | None:
    labels = sorted({part for _, text in members if isinstance(text, str) for part in text.split("|") if part})
    return "|".join(labels) if labels else None


def classify_price_change_candidates(
        offers: pd.DataFrame, timelines: Sequence[LocationCaptureTimeline],
) -> tuple[pd.DataFrame, tuple[LocationPriceChangeSummary, ...]]:
    """Classify every identity of every eligible interval (exact, order independent; inputs unchanged).

    ``offers`` are canonical offers (:data:`EVENT_IDENTITY_COLUMNS`,
    ``scheduled_capture_period``, ``price_cents``, ``source_location_labels``);
    ``timelines`` come from :func:`capture_timelines` on the same evidence.

    Raises:
        PriceChangeContractError: Malformed offers or timelines.
        CaptureEvidenceError: An offer lies outside every eligible capture, or an eligible capture has no offers.
    """
    if not isinstance(timelines, (tuple, list)) or not all(isinstance(t, LocationCaptureTimeline) for t in timelines):
        raise TypeError("timelines must be LocationCaptureTimeline objects")
    locations = [t.canonical_location for t in timelines]
    if len(set(locations)) != len(locations):
        raise PriceChangeContractError("one timeline per canonical location")
    _validate_offers(offers)
    eligible = {(t.canonical_location, p) for t in timelines for p in t.periods(CaptureState.ELIGIBLE)}
    index: dict[tuple[Key, str], dict[tuple, list[tuple[int, str]]]] = {}
    rows = offers.loc[:, [*EVENT_IDENTITY_COLUMNS, EVENT_TIMESTAMP_COLUMN, "price_cents", "source_location_labels"]]
    for values in rows.astype(object).itertuples(index=False, name=None):
        identity, period, cents, labels = values[:len(EVENT_IDENTITY_COLUMNS)], values[-3], int(values[-2]), values[-1]
        capture = ((identity[0], identity[1]), period)
        if capture not in eligible:
            raise CaptureEvidenceError("an offer lies outside every eligible scheduled capture")
        index.setdefault(capture, {}).setdefault(identity, []).append((cents, labels))
    if set(index) != eligible:
        raise CaptureEvidenceError("an eligible scheduled capture has no canonical offers")

    out: list[tuple] = []
    summaries = []
    for timeline in sorted(timelines, key=lambda t: t.canonical_location):
        outcomes: list[TerminalOutcome] = []
        zero = 0
        for interval in timeline.intervals:
            before = index[(timeline.canonical_location, interval.previous_period)]
            after = index[(timeline.canonical_location, interval.current_period)]
            for identity in set(before) | set(after):
                prev, cur = before.get(identity, []), after.get(identity, [])
                outcome = classify_endpoint_offers([c for c, _ in prev], [c for c, _ in cur])
                priced = outcome in _PRICED
                p = prev[0][0] if len(prev) == 1 and outcome is not TerminalOutcome.AMBIGUOUS else None
                c = cur[0][0] if len(cur) == 1 and outcome is not TerminalOutcome.AMBIGUOUS else None
                baseline = priced and p == 0
                zero += baseline
                outcomes.append(outcome)
                out.append((*identity, interval.previous_period, interval.current_period, outcome.value, len(prev),
                            len(cur), p, c, (c - p) if priced else None, baseline, _labels(prev), _labels(cur)))
        breaks = timeline.break_counts
        summaries.append(LocationPriceChangeSummary(
            canonical_location=timeline.canonical_location, source_streams=timeline.source_streams,
            scheduled_periods=len(timeline.captures),
            eligible_periods=len(timeline.periods(CaptureState.ELIGIBLE)),
            excluded_periods=len(timeline.periods(CaptureState.GOVERNED_EXCLUSION)),
            missing_periods=len(timeline.periods(CaptureState.MISSING_CAPTURE)),
            intervals=len(timeline.intervals), breaks=breaks, counts=OutcomeCounts.of(outcomes, zero)))
    position = {c: i for i, c in enumerate(CANDIDATE_COLUMNS)}
    order = [position[c] for c in ("canonical_city", "canonical_location", "previous_period", "current_period",
                                   *EVENT_IDENTITY_COLUMNS[2:])]
    out.sort(key=lambda r: tuple(r[i] for i in order))
    frame = pd.DataFrame(out, columns=list(CANDIDATE_COLUMNS), dtype=object)
    frame["previous_offer_count"] = frame["previous_offer_count"].astype(int)
    frame["current_offer_count"] = frame["current_offer_count"].astype(int)
    frame["zero_baseline"] = frame["zero_baseline"].astype(bool)
    return frame, tuple(summaries)


# ------------------------------------------------------------------ assessment


def _blocked(blockers: Sequence[PriceChangeBlocker], readiness_blockers: Sequence[str] = (),
             offers_assessed: int = 0) -> PriceChangeCandidateResult:
    return PriceChangeCandidateResult(PriceChangeCandidateReport(
        status=PriceChangeStatus.BLOCKED, blockers=tuple(dict.fromkeys(blockers)),
        readiness_blockers=tuple(readiness_blockers), offers_assessed=offers_assessed))


def assess_price_change_candidates(jobs: pd.DataFrame, cars: pd.DataFrame, *, readiness: object,
                                   scheduled: object, canonical_offers: object) -> PriceChangeCandidateResult:
    """Observed price-change candidates of the pricing-eligible canonical offers (see the module docstring).

    ``jobs``/``cars`` are the analysis-stage frames every report was assessed
    on. Pricing readiness must be ready and be the assessment of exactly the
    supplied schedule and canonical-offer reports; the offers must be bound to
    these frames. Any failure returns a ``BLOCKED`` result with typed blockers
    and no candidates. Inputs are never modified; raw ``job_id`` is never read.

    Raises:
        TypeError: An argument has the wrong type.
    """
    from ql2_sixt_canada_analysis.canonical_offers import CanonicalOfferReport
    from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport
    from ql2_sixt_canada_analysis.pricing_population import PricingPopulationError, frame_binding
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport

    for value, kind, name in ((readiness, PricingReadinessReport, "readiness"),
                              (scheduled, PerStreamScheduledCoverageReport, "scheduled"),
                              (canonical_offers, CanonicalOfferReport, "canonical_offers")):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}")
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be DataFrames")
    B = PriceChangeBlocker
    if not readiness.ready:
        return _blocked([B.PRICING_NOT_READY], [b.value for b in readiness.blocking_reasons])
    blockers: list[PriceChangeBlocker] = []
    if readiness.canonical_offers is not canonical_offers or readiness.scheduled_coverage is not scheduled:
        blockers.append(B.READINESS_EVIDENCE_MISMATCH)
    if canonical_offers.binding != frame_binding(jobs, cars):
        blockers.append(B.FRAME_BINDING_MISMATCH)
    if not canonical_offers.ready or canonical_offers.unassessable_rows or canonical_offers.offers is None:
        blockers.append(B.CANONICAL_OFFERS_NOT_READY)
    if not scheduled.is_valid or scheduled.capture_periods is None or scheduled.jobs_assessed != len(jobs):
        blockers.append(B.SCHEDULE_EVIDENCE_INVALID)
    if blockers:
        return _blocked(blockers)
    try:
        offers = canonical_offers.offers_for(jobs, cars)
    except PricingPopulationError:
        return _blocked([B.FRAME_BINDING_MISMATCH])
    try:
        timelines = capture_timelines(scheduled, canonical_offers.policy)
        frame, summaries = classify_price_change_candidates(offers, timelines)
    except CaptureEvidenceError:
        return _blocked([B.CAPTURE_EVIDENCE_INCONSISTENT], offers_assessed=len(offers))
    except PriceChangeContractError:
        return _blocked([B.OFFER_CONTRACT_INVALID], offers_assessed=len(offers))
    report = PriceChangeCandidateReport(
        status=PriceChangeStatus.COMPLETED, offers_assessed=len(offers), locations=summaries,
        overall=sum((s.counts for s in summaries), OutcomeCounts()))
    return PriceChangeCandidateResult(report=report, candidates=frame, timelines=timelines)


def price_change_candidates_from_pipeline(run: object) -> PriceChangeCandidateResult:
    """Candidates from one :func:`~ql2_sixt_canada_analysis.pricing_pipeline.run_pricing_pipeline` result.

    Fails closed (``PRICING_NOT_READY`` with the readiness blocker categories)
    unless the central pricing gate passed and every required assessment exists.
    """
    from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport

    if not isinstance(run, PricingPipelineResult):
        raise TypeError("run must be a PricingPipelineResult")
    pricing = run.pricing
    if not isinstance(pricing, PricingReadinessReport):
        raise TypeError("the pipeline produced no readiness report")
    if not pricing.ready or run.scheduled is None or run.canonical_offers is None:
        return _blocked([PriceChangeBlocker.PRICING_NOT_READY],
                        [b.value for b in pricing.blocking_reasons] or ["required_assessment_unavailable"])
    return assess_price_change_candidates(run.jobs, run.cars, readiness=pricing, scheduled=run.scheduled,
                                          canonical_offers=run.canonical_offers)
