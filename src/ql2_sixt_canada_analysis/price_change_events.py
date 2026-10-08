"""Price-change events: the locked event contract and the event-construction engine (issue #4).

The engine turns the pricing-eligible canonical offers of
:func:`~ql2_sixt_canada_analysis.pricing_pipeline.run_pricing_pipeline` into
**observed price-change candidates**: one candidate per offer identity per
eligible one-hour capture interval, each with exactly one terminal outcome and
exact-cent change metrics. A candidate is an observation of the collected feed,
never a proven genuine market repricing or extraction anomaly: deciding that
requires source and operational corroboration outside this module.

Entry points: :func:`run_price_change_events` (runs the pipeline, then every
gate), :func:`price_change_candidates_from_pipeline` (one existing pipeline run),
:func:`assess_price_change_candidates` (the gated assessment of supplied
evidence) and :func:`classify_price_change_candidates` (the pure engine).

Population
----------
Only :attr:`CanonicalOfferReport.offers <ql2_sixt_canada_analysis.canonical_offers.CanonicalOfferReport>`
of the pricing-eligible population (the governed Calgary
``INCOMPLETE_PARENT_CAPTURE`` exclusion and every ineligible row never enter;
the Vancouver ``Downtown``/``Thurlow`` aliases are already one canonical
location). Never ``MatchedLocationPricingResult.pairs``, raw detail rows, raw
job identifiers, row order or generated files: the pair table holds only the
airport/downtown shared assortment and would drop one-sided products.

Offer identity (:data:`EVENT_IDENTITY_COLUMNS`)
-----------------------------------------------
``canonical_city, canonical_location, pickup_date, return_date``, every column of
:data:`~ql2_sixt_canada_analysis.canonical_offers.APPROVED_PRODUCT_COLUMNS`
(``car_name, car_type, transmission, seats, bags``), ``currency, price_basis``.
Values are the canonical-offer assessment's parsed values, compared by exact
equality: no trimming, recasing, fuzzy matching, imputation or inference. Price
is never part of the identity. Raw source location labels are provenance only
(kept per endpoint for later alias diagnostics) and never split the identity.

Canonical timestamp (:data:`EVENT_TIMESTAMP_COLUMN`)
----------------------------------------------------
``scheduled_capture_period`` from the authority-backed
:class:`~ql2_sixt_canada_analysis.collection_schedule.CapturePeriodIndex`: the
trusted UTC start of the scheduled hourly capture, ``YYYYMMDDTHHMMSSZ``, parsed
strictly by :func:`parse_scheduled_period`. Never raw ``job_id``, row order,
``scrape_date``, ``date_clean``, raw scrape or finish timestamps, the reporting
day, or the last time a product happened to appear.

Interval grid and exact one-hour adjacency
------------------------------------------
:func:`capture_timelines` builds one timeline per canonical location from the
per-stream schedule, its assessed coverage and the resolved governed exclusions,
never from offers, so the grid exists independently of product presence. Each
scheduled period is ``eligible`` (every contributing source stream is covered),
a ``governed_exclusion`` or a ``missing_capture`` (excused or not). A
:class:`CaptureInterval` joins two schedule-adjacent eligible periods exactly
one hour apart with the same contributing source streams; every other adjacent
pair is a typed :class:`IntervalBreak`. The first eligible capture of a run
seeds state only: it has no preceding interval and creates no appearances.
"Previous" is the immediately preceding scheduled capture, never the last
earlier observation of the product: a product absent at ``t-1`` and present at
``t`` *appeared* at ``t``. A governed exclusion or a missing capture is a hard
break: nothing is compared across it and no appearance or disappearance is
attributed to it. The gated assessment also proves that eligible captures are
eligible parent captures of the pricing population and excluded captures are
its governed exclusions.

Terminal outcomes (:class:`TerminalOutcome`)
-------------------------------------------
For every interval and every identity present at either endpoint, exactly one
outcome, first applicable wins:

1. ``ambiguous`` - more than one canonical offer for the identity at either
   endpoint (never resolved by row order, minimum, maximum, mean, median, first,
   last or a Cartesian product; no price is exposed);
2. ``appeared`` - absent previously, exactly one offer currently;
3. ``disappeared`` - exactly one offer previously, absent currently;
4. ``unchanged`` / ``increase`` / ``decrease`` - exactly one offer at both
   endpoints, by the sign of ``current price_cents - previous price_cents``.

A currency or price-basis change is never a price movement: the unit is part of
the identity, so the old unit disappears and the new unit appears. Nothing is
converted or normalized, and mixed units across different identities never
block anything.

Change metrics
--------------
From exact non-negative integer ``price_cents`` only, for comparable outcomes
(unchanged, increase, decrease): ``change_cents = current - previous``,
``change_dollars = change_cents / 100`` and
``change_percent = 100 * change_cents / previous`` evaluated exactly
(:class:`fractions.Fraction`) and stored as a finite float. A zero previous
price has no percentage (``zero_denominator``; never infinite or NaN);
``percent_valid`` and ``zero_denominator`` partition the comparable candidates.
Nothing is rounded here; rounding is presentation only.

Confidentiality
---------------
The candidate frame (:attr:`PriceChangeCandidateResult.candidates`) holds
proprietary event-level values: it stays in memory, is excluded from ``repr``
and equality and is never written. :class:`PriceChangeCandidateReport` holds
counts, enum values and approved configuration keys only.

Deliberately deferred to later issue #4 phases: synchronized-movement detection,
persistence, the Vancouver decrease case study, anomaly review and conclusions,
alert thresholds, monitoring rules, heatmaps, presentation tables and notebook 03.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from functools import cached_property
from pathlib import Path

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
    "UnknownCanonicalLocationError",
    "approved_canonical_locations",
    "assess_price_change_candidates",
    "capture_timelines",
    "change_percent",
    "classify_endpoint_offers",
    "classify_price_change_candidates",
    "parse_scheduled_period",
    "price_change_candidates_from_pipeline",
    "price_change_cents",
    "run_price_change_events",
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
#: The endpoints of a one-hour capture interval (canonical scheduled periods).
EVENT_INTERVAL_COLUMNS: tuple[str, ...] = ("previous_scheduled_capture_period", "current_scheduled_capture_period")
#: The unique key of a candidate: one identity in one capture interval.
EVENT_KEY_COLUMNS: tuple[str, ...] = (*EVENT_IDENTITY_COLUMNS, *EVENT_INTERVAL_COLUMNS)
#: Columns of the in-memory candidate frame, in this order (proprietary; never printed or written here).
#: Price and change fields are ``None`` where the outcome supports no value.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    *EVENT_KEY_COLUMNS, "previous_offer_count", "current_offer_count", "outcome",
    "previous_price_cents", "current_price_cents", "change_cents", "previous_price", "current_price",
    "change_dollars", "change_percent", "percent_valid", "zero_denominator",
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
    """The schedule, coverage, exclusion, population and offer evidence disagree (stale or inconsistent)."""


class UnknownCanonicalLocationError(PriceChangeContractError):
    """An offer or timeline names a canonical location outside the approved configuration (never inferred)."""


class TerminalOutcome(StrEnum):
    """Exactly one per candidate (``ambiguous`` takes precedence over every other outcome)."""

    UNCHANGED = "unchanged"
    INCREASE = "increase"
    DECREASE = "decrease"
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    AMBIGUOUS = "ambiguous"


#: Outcomes that compare exactly one previous and one current price.
_COMPARABLE = frozenset({TerminalOutcome.UNCHANGED, TerminalOutcome.INCREASE, TerminalOutcome.DECREASE})


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
    LOCATION_AUTHORITY_UNAVAILABLE = "location_authority_unavailable"
    UNKNOWN_CANONICAL_LOCATION = "unknown_canonical_location"
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


def change_percent(previous: object, current: object) -> float | None:
    """``100 * (current - previous) / previous`` evaluated exactly from integer cents (``None`` for a zero baseline).

    The exact rational is converted to a float once; it is always finite.
    """
    base = _cents(previous)
    if base == 0:
        return None
    return float(Fraction(100 * price_change_cents(previous, current), base))


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
    """One timeline per canonical location from the per-stream schedule, its coverage and resolved exclusions.

    Each stream's periods are its materialized schedule; a period is excluded
    or missing exactly as the coverage report recorded it, otherwise covered.
    The excluded periods must be exactly the scheduled periods of the resolved
    governed parent captures of their city. A canonical location's period is a
    governed exclusion if any contributing stream has it excluded, missing if
    any has it missing, else eligible. Offers are never read, so adjacency is
    independent of product presence.

    Raises:
        TypeError: Wrong argument types.
        CaptureEvidenceError: The schedule is unavailable, evidence is absent, or coverage disagrees with it.
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
    if scheduled.capture_periods is None or scheduled.capture_exclusions is None:
        raise CaptureEvidenceError("capture-period and governed-exclusion evidence are required")
    coverage = {c.stream: c for c in scheduled.streams}
    if len(coverage) != len(scheduled.streams) or set(coverage) != set(schedule.expected_streams):
        raise CaptureEvidenceError("coverage must name every scheduled stream exactly once")
    resolved: dict[str, set[str]] = {}
    for entry in scheduled.capture_exclusions.entries:
        period = scheduled.capture_periods.periods.get(entry.parent_key)
        if not isinstance(period, str):
            raise CaptureEvidenceError("a governed exclusion has no scheduled capture period")
        resolved.setdefault(entry.city, set()).add(period)
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
        if excluded != resolved.get(stream_schedule.city, set()) & set(nominal):
            raise CaptureEvidenceError("excluded periods differ from the resolved governed exclusions")
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


def approved_canonical_locations(location_authority: object, policy: object) -> tuple[Key, ...]:
    """The approved canonical locations in authority order (the contract's stream order, aliases merged).

    Requires an exact authority-backed role map and a canonical-offer policy
    that canonicalizes every approved stream exactly as the location authority
    does.

    Raises:
        TypeError: Wrong argument types.
        UnknownCanonicalLocationError: The configuration is unavailable or inconsistent.
    """
    from ql2_sixt_canada_analysis.canonical_offers import CanonicalOfferPolicy
    from ql2_sixt_canada_analysis.location_authority import LocationAuthorityReport

    if not isinstance(location_authority, LocationAuthorityReport):
        raise TypeError("location_authority must be a LocationAuthorityReport")
    if not isinstance(policy, CanonicalOfferPolicy):
        raise TypeError("policy must be a CanonicalOfferPolicy")
    keys = tuple(tuple(k) for k in location_authority.contract.expected_keys)
    if not location_authority.roles_exact or not policy.available or not keys:
        raise UnknownCanonicalLocationError("the approved canonical-location configuration is unavailable")
    if any(tuple(location_authority.canonical(k)) != tuple(policy.canonical(k)) for k in keys):
        raise UnknownCanonicalLocationError("the canonical-offer policy disagrees with the location authority")
    return tuple(dict.fromkeys(tuple(policy.canonical(k)) for k in keys))


# ------------------------------------------------------------------ results


def _count(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise PriceChangeContractError(f"{name} must be a non-negative int")


@dataclass(frozen=True, slots=True)
class OutcomeCounts:
    """Interval and candidate accounting: every candidate has exactly one terminal outcome."""

    intervals: int = 0
    candidates: int = 0
    unchanged: int = 0
    increase: int = 0
    decrease: int = 0
    appeared: int = 0
    disappeared: int = 0
    ambiguous: int = 0
    #: Comparable candidates with a nonzero previous price (the percentage is defined).
    percent_valid: int = 0
    #: Comparable candidates with a zero previous price (the percentage is undefined, never infinite).
    zero_denominator: int = 0

    def __post_init__(self) -> None:
        for name in self.__slots__:
            _count(getattr(self, name), name)
        if self.candidates != sum(getattr(self, o.value) for o in TerminalOutcome):
            raise PriceChangeContractError("candidates must equal the sum of terminal outcomes")
        if self.comparable != self.percent_valid + self.zero_denominator:
            raise PriceChangeContractError("comparable must equal percent-valid plus zero-denominator")
        if self.intervals == 0 and self.candidates:
            raise PriceChangeContractError("candidates exist only within intervals")

    @property
    def comparable(self) -> int:
        """Candidates with exactly one offer at both endpoints (unchanged, increase or decrease)."""
        return self.unchanged + self.increase + self.decrease

    @property
    def changed(self) -> int:
        return self.increase + self.decrease

    def __add__(self, other: OutcomeCounts) -> OutcomeCounts:
        return OutcomeCounts(*(getattr(self, n) + getattr(other, n) for n in self.__slots__))

    @classmethod
    def of(cls, outcomes: Sequence[TerminalOutcome], *, intervals: int, zero_denominator: int = 0) -> OutcomeCounts:
        values = [TerminalOutcome(o) for o in outcomes]
        comparable = sum(1 for o in values if o in _COMPARABLE)
        return cls(intervals, len(values), *(values.count(o) for o in TerminalOutcome),
                   percent_valid=comparable - zero_denominator, zero_denominator=zero_denominator)


@dataclass(frozen=True, slots=True)
class LocationPriceChangeSummary:
    """One approved canonical location: its scheduled-period states, breaks and counts."""

    canonical_location: Key
    source_streams: tuple[Key, ...]
    scheduled_periods: int
    eligible_periods: int
    excluded_periods: int
    missing_periods: int
    breaks: tuple[tuple[str, int], ...]
    counts: OutcomeCounts

    def __post_init__(self) -> None:
        _location(self.canonical_location)
        for name in ("scheduled_periods", "eligible_periods", "excluded_periods", "missing_periods"):
            _count(getattr(self, name), name)
        if self.scheduled_periods != self.eligible_periods + self.excluded_periods + self.missing_periods:
            raise PriceChangeContractError("every scheduled period is eligible, excluded or missing")
        if any(r not in {b.value for b in IntervalBreak} or n <= 0 for r, n in self.breaks):
            raise PriceChangeContractError("breaks are typed positive counts")
        if not isinstance(self.counts, OutcomeCounts):
            raise PriceChangeContractError("counts must be OutcomeCounts")
        if self.counts.intervals + sum(n for _, n in self.breaks) != max(self.scheduled_periods - 1, 0):
            raise PriceChangeContractError("every adjacent scheduled pair is an interval or a break")
        if self.counts.intervals > max(self.eligible_periods - 1, 0):
            raise PriceChangeContractError("intervals join eligible periods only")

    @property
    def intervals(self) -> int:
        return self.counts.intervals


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
            if not all(isinstance(b, PriceChangeBlocker) for b in self.blockers):
                raise PriceChangeContractError("blockers must be PriceChangeBlocker values")
            return
        if self.blockers or self.readiness_blockers or not isinstance(self.overall, OutcomeCounts):
            raise PriceChangeContractError("a completed report has no blockers and overall counts")
        keys = [s.canonical_location for s in self.locations]
        if len(set(keys)) != len(keys):
            raise PriceChangeContractError("one summary per canonical location")
        if sum((s.counts for s in self.locations), OutcomeCounts()) != self.overall:
            raise PriceChangeContractError("location counts must sum to the overall counts")

    @property
    def completed(self) -> bool:
        return self.status is PriceChangeStatus.COMPLETED

    @property
    def approved_locations(self) -> tuple[Key, ...]:
        return tuple(s.canonical_location for s in self.locations)

    def location(self, key: Key) -> LocationPriceChangeSummary:
        return next(s for s in self.locations if s.canonical_location == tuple(key))


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA or value is pd.NaT


def _finite_equal(value: object, expected: float) -> bool:
    return (isinstance(value, float) and not isinstance(value, bool) and math.isfinite(value)
            and value == expected)


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
    if {t.canonical_location for t in timelines} != set(report.approved_locations):
        raise PriceChangeContractError("one timeline per reported canonical location")
    if sum(len(t.intervals) for t in timelines) != overall.intervals:
        raise PriceChangeContractError("the timelines contradict the interval counts")
    per_location: dict[Key, list[TerminalOutcome]] = {k: [] for k in report.approved_locations}
    zero: dict[Key, int] = {k: 0 for k in report.approved_locations}
    previous_key, current_key = EVENT_INTERVAL_COLUMNS
    for row in candidates.itertuples(index=False, name=None):
        values = dict(zip(CANDIDATE_COLUMNS, row))
        if any(_missing(values[c]) for c in EVENT_KEY_COLUMNS):
            raise PriceChangeContractError("every candidate carries the full identity and interval")
        location = (values["canonical_city"], values["canonical_location"])
        CaptureInterval(location, values[previous_key], values[current_key])
        if (location, values[previous_key], values[current_key]) not in allowed:
            raise PriceChangeContractError("a candidate lies outside every eligible capture interval")
        outcome = TerminalOutcome(values["outcome"])
        n_prev, n_cur = values["previous_offer_count"], values["current_offer_count"]
        _count(n_prev, "previous_offer_count"), _count(n_cur, "current_offer_count")
        prev, cur, change = values["previous_price_cents"], values["current_price_cents"], values["change_cents"]
        derived = (values["change_dollars"], values["change_percent"])
        valid, zero_denominator = values["percent_valid"], values["zero_denominator"]
        if not isinstance(valid, (bool, np.bool_)) or not isinstance(zero_denominator, (bool, np.bool_)):
            raise PriceChangeContractError("percent_valid and zero_denominator are booleans")
        if outcome is TerminalOutcome.AMBIGUOUS:
            ok = ((n_prev > 1 or n_cur > 1) and prev is None and cur is None and change is None
                  and values["previous_price"] is None and values["current_price"] is None)
        elif outcome is TerminalOutcome.APPEARED:
            ok = ((n_prev, n_cur) == (0, 1) and prev is None and values["previous_price"] is None
                  and change is None and _finite_equal(values["current_price"], _cents(cur) / 100))
        elif outcome is TerminalOutcome.DISAPPEARED:
            ok = ((n_prev, n_cur) == (1, 0) and cur is None and values["current_price"] is None
                  and change is None and _finite_equal(values["previous_price"], _cents(prev) / 100))
        else:
            percent = change_percent(prev, cur)
            ok = ((n_prev, n_cur) == (1, 1) and classify_endpoint_offers([prev], [cur]) is outcome
                  and change == price_change_cents(prev, cur)
                  and _finite_equal(values["previous_price"], prev / 100)
                  and _finite_equal(values["current_price"], cur / 100)
                  and _finite_equal(derived[0], change / 100)
                  and bool(valid) == (percent is not None) and bool(zero_denominator) == (percent is None)
                  and (derived[1] is None if percent is None else _finite_equal(derived[1], percent)))
        if outcome not in _COMPARABLE:
            ok = ok and derived == (None, None) and not valid and not zero_denominator
        if not ok:
            raise PriceChangeContractError("a candidate's offers, prices and metrics contradict its outcome")
        per_location[location].append(outcome)
        zero[location] += bool(zero_denominator)
    for summary in report.locations:
        key = summary.canonical_location
        if OutcomeCounts.of(per_location[key], intervals=summary.intervals, zero_denominator=zero[key]) \
                != summary.counts:
            raise PriceChangeContractError("the candidate frame contradicts the location counts")


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


# ------------------------------------------------------------------ the pure engine


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


def _labels(members: list[tuple[int, object]]) -> str | None:
    labels = sorted({part for _, text in members if isinstance(text, str) for part in text.split("|") if part})
    return "|".join(labels) if labels else None


def _candidate_row(identity: tuple, interval: CaptureInterval, prev: list, cur: list) -> tuple:
    outcome = classify_endpoint_offers([c for c, _ in prev], [c for c, _ in cur])
    p = prev[0][0] if len(prev) == 1 and outcome is not TerminalOutcome.AMBIGUOUS else None
    c = cur[0][0] if len(cur) == 1 and outcome is not TerminalOutcome.AMBIGUOUS else None
    comparable = outcome in _COMPARABLE
    change = c - p if comparable else None
    percent = change_percent(p, c) if comparable else None
    return (*identity, interval.previous_period, interval.current_period, len(prev), len(cur), outcome.value,
            p, c, change, None if p is None else p / 100, None if c is None else c / 100,
            None if change is None else change / 100, percent, comparable and percent is not None,
            comparable and percent is None, _labels(prev), _labels(cur))


def classify_price_change_candidates(
        offers: pd.DataFrame, timelines: Sequence[LocationCaptureTimeline],
        locations: Sequence[Key] | None = None,
) -> tuple[pd.DataFrame, tuple[LocationPriceChangeSummary, ...]]:
    """The pure engine: classify every identity of every eligible interval (exact, order independent).

    ``offers`` are canonical offers (:data:`EVENT_IDENTITY_COLUMNS`,
    ``scheduled_capture_period``, ``price_cents``, ``source_location_labels``);
    ``timelines`` come from :func:`capture_timelines` on the same evidence.
    ``locations`` is the approved canonical-location authority order (default:
    the timelines' locations in key order); it must name exactly the timelines'
    locations, and every location gets a summary, with zero counts if it has no
    candidates. An eligible capture without offers is a valid empty capture
    (the gated assessment separately proves no pipeline capture is empty).
    Inputs are never modified.

    Raises:
        PriceChangeContractError: Malformed offers or timelines.
        UnknownCanonicalLocationError: An offer or timeline names an unapproved canonical location.
        CaptureEvidenceError: An offer lies outside every eligible capture.
    """
    if not isinstance(timelines, (tuple, list)) or not all(isinstance(t, LocationCaptureTimeline) for t in timelines):
        raise TypeError("timelines must be LocationCaptureTimeline objects")
    by_location = {t.canonical_location: t for t in timelines}
    if len(by_location) != len(timelines):
        raise PriceChangeContractError("one timeline per canonical location")
    order = tuple(tuple(k) for k in locations) if locations is not None else tuple(sorted(by_location))
    for key in order:
        _location(key)
    if len(set(order)) != len(order):
        raise PriceChangeContractError("the approved locations are distinct exact keys")
    if set(by_location) - set(order):
        raise UnknownCanonicalLocationError("a timeline names an unapproved canonical location")
    if set(order) - set(by_location):
        raise CaptureEvidenceError("an approved canonical location has no capture timeline")
    _validate_offers(offers)
    eligible = {(t.canonical_location, p) for t in timelines for p in t.periods(CaptureState.ELIGIBLE)}
    index: dict[tuple[Key, str], dict[tuple, list[tuple[int, object]]]] = {}
    rows = offers.loc[:, [*EVENT_IDENTITY_COLUMNS, EVENT_TIMESTAMP_COLUMN, "price_cents", "source_location_labels"]]
    for values in rows.astype(object).itertuples(index=False, name=None):
        identity, period, cents, labels = values[:len(EVENT_IDENTITY_COLUMNS)], values[-3], int(values[-2]), values[-1]
        location = (identity[0], identity[1])
        if location not in by_location:
            raise UnknownCanonicalLocationError("an offer names an unapproved canonical location")
        if (location, period) not in eligible:
            raise CaptureEvidenceError("an offer lies outside every eligible scheduled capture")
        index.setdefault((location, period), {}).setdefault(identity, []).append((cents, labels))

    rank = {k: i for i, k in enumerate(order)}
    out: list[tuple] = []
    summaries = []
    for key in order:
        timeline = by_location[key]
        outcomes: list[TerminalOutcome] = []
        zero = 0
        for interval in timeline.intervals:
            before = index.get((key, interval.previous_period), {})
            after = index.get((key, interval.current_period), {})
            for identity in set(before) | set(after):
                row = _candidate_row(identity, interval, before.get(identity, []), after.get(identity, []))
                outcomes.append(TerminalOutcome(row[len(EVENT_KEY_COLUMNS) + 2]))
                zero += row[CANDIDATE_COLUMNS.index("zero_denominator")]
                out.append(row)
        summaries.append(LocationPriceChangeSummary(
            canonical_location=key, source_streams=timeline.source_streams,
            scheduled_periods=len(timeline.captures),
            eligible_periods=len(timeline.periods(CaptureState.ELIGIBLE)),
            excluded_periods=len(timeline.periods(CaptureState.GOVERNED_EXCLUSION)),
            missing_periods=len(timeline.periods(CaptureState.MISSING_CAPTURE)),
            breaks=timeline.break_counts,
            counts=OutcomeCounts.of(outcomes, intervals=len(timeline.intervals), zero_denominator=zero)))
    position = {c: i for i, c in enumerate(CANDIDATE_COLUMNS)}
    tail = [position[c] for c in (*EVENT_INTERVAL_COLUMNS, *EVENT_IDENTITY_COLUMNS[2:])]
    out.sort(key=lambda r: (rank[(r[0], r[1])], *(r[i] for i in tail)))
    frame = pd.DataFrame(out, columns=list(CANDIDATE_COLUMNS), dtype=object)
    for column in ("previous_offer_count", "current_offer_count"):
        frame[column] = frame[column].astype(int)
    for column in ("percent_valid", "zero_denominator"):
        frame[column] = frame[column].astype(bool)
    return frame, tuple(summaries)


# ------------------------------------------------------------------ gated assessment


def _blocked(blockers: Sequence[PriceChangeBlocker], readiness_blockers: Sequence[str] = (),
             offers_assessed: int = 0) -> PriceChangeCandidateResult:
    return PriceChangeCandidateResult(PriceChangeCandidateReport(
        status=PriceChangeStatus.BLOCKED, blockers=tuple(dict.fromkeys(blockers)),
        readiness_blockers=tuple(readiness_blockers), offers_assessed=offers_assessed))


def _population_agrees(jobs: pd.DataFrame, cars: pd.DataFrame, population, scheduled,  # type: ignore[no-untyped-def]
                       timelines: Sequence[LocationCaptureTimeline], offers: pd.DataFrame,
                       city_column: str) -> bool:
    """Eligible captures are eligible population parents with offers; excluded ones are its governed exclusions."""
    from ql2_sixt_canada_analysis.pricing_population import DetailEligibility

    if city_column not in jobs.columns:
        return False
    periods = scheduled.capture_periods.parent_periods(jobs).tolist()
    cities = jobs[city_column].astype(object).tolist()
    eligible: set[tuple[str, str]] = set()
    excluded: set[tuple[str, str]] = set()
    for city, period, status in zip(cities, periods, population.parent_status):
        if isinstance(city, str) and isinstance(period, str):
            if status == DetailEligibility.ELIGIBLE.value:
                eligible.add((city, period))
            elif status == DetailEligibility.GOVERNED_EXCLUSION.value:
                excluded.add((city, period))
    offered = set(zip(offers["canonical_city"].astype(object), offers["canonical_location"].astype(object),
                      offers[EVENT_TIMESTAMP_COLUMN].astype(object)))
    for timeline in timelines:
        city, location = timeline.canonical_location
        if any((city, p) not in eligible or (city, location, p) not in offered
               for p in timeline.periods(CaptureState.ELIGIBLE)):
            return False
        if any((city, p) not in excluded for p in timeline.periods(CaptureState.GOVERNED_EXCLUSION)):
            return False
    return True


def assess_price_change_candidates(jobs: pd.DataFrame, cars: pd.DataFrame, *, readiness: object,
                                   population: object, scheduled: object, canonical_offers: object,
                                   location_authority: object, city_column: str = "city",
                                   ) -> PriceChangeCandidateResult:
    """Observed price-change candidates of the pricing-eligible canonical offers (see the module docstring).

    ``jobs``/``cars`` are the analysis-stage frames every report was assessed
    on. Gates, in order (any failure returns a ``BLOCKED`` result with typed
    blockers and no candidates): pricing readiness is ready and is the
    assessment of exactly the supplied schedule, canonical-offer and
    location-authority reports; the population and canonical offers are bound
    to these frames and the schedule was assessed on them; the canonical offers
    are ready with no unassessable row; the schedule assessment is valid with
    capture-period and governed-exclusion evidence; the approved
    canonical-location configuration is available and agrees with the offer
    policy and the schedule. The pure engine runs only after every gate, and
    its capture grid must agree with the population. Inputs are never
    modified; raw ``job_id`` is never read.

    Raises:
        TypeError: An argument has the wrong type.
    """
    from ql2_sixt_canada_analysis.canonical_offers import CanonicalOfferReport
    from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport
    from ql2_sixt_canada_analysis.location_authority import LocationAuthorityReport
    from ql2_sixt_canada_analysis.pricing_population import PricingPopulation, PricingPopulationError, frame_binding
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport

    for value, kind, name in ((readiness, PricingReadinessReport, "readiness"),
                              (population, PricingPopulation, "population"),
                              (scheduled, PerStreamScheduledCoverageReport, "scheduled"),
                              (canonical_offers, CanonicalOfferReport, "canonical_offers"),
                              (location_authority, LocationAuthorityReport, "location_authority")):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}")
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be DataFrames")
    B = PriceChangeBlocker
    if not readiness.ready:
        return _blocked([B.PRICING_NOT_READY], [b.value for b in readiness.blocking_reasons])
    blockers: list[PriceChangeBlocker] = []
    if (readiness.canonical_offers is not canonical_offers or readiness.scheduled_coverage is not scheduled
            or readiness.location_authority is not location_authority):
        blockers.append(B.READINESS_EVIDENCE_MISMATCH)
    binding = frame_binding(jobs, cars)
    if (population.binding != binding or canonical_offers.binding != binding
            or scheduled.jobs_assessed != len(jobs)):
        blockers.append(B.FRAME_BINDING_MISMATCH)
    if not canonical_offers.ready or canonical_offers.unassessable_rows or canonical_offers.offers is None:
        blockers.append(B.CANONICAL_OFFERS_NOT_READY)
    if (not scheduled.is_valid or scheduled.capture_periods is None or scheduled.capture_exclusions is None
            or scheduled.unmatched_exclusions):
        blockers.append(B.SCHEDULE_EVIDENCE_INVALID)
    try:
        approved = approved_canonical_locations(location_authority, canonical_offers.policy)
        if set(location_authority.contract.expected_keys) != set(scheduled.schedule.expected_streams):
            raise UnknownCanonicalLocationError("the schedule and the location authority name other streams")
    except UnknownCanonicalLocationError:
        blockers.append(B.LOCATION_AUTHORITY_UNAVAILABLE)
    if blockers:
        return _blocked(blockers)
    try:
        offers = canonical_offers.offers_for(jobs, cars)
    except PricingPopulationError:
        return _blocked([B.FRAME_BINDING_MISMATCH])
    try:
        timelines = capture_timelines(scheduled, canonical_offers.policy)
        frame, summaries = classify_price_change_candidates(offers, timelines, approved)
    except UnknownCanonicalLocationError:
        return _blocked([B.UNKNOWN_CANONICAL_LOCATION], offers_assessed=len(offers))
    except CaptureEvidenceError:
        return _blocked([B.CAPTURE_EVIDENCE_INCONSISTENT], offers_assessed=len(offers))
    except PriceChangeContractError:
        return _blocked([B.OFFER_CONTRACT_INVALID], offers_assessed=len(offers))
    if not _population_agrees(jobs, cars, population, scheduled, timelines, offers, city_column):
        return _blocked([B.CAPTURE_EVIDENCE_INCONSISTENT], offers_assessed=len(offers))
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
    required = (run.population, run.scheduled, run.canonical_offers, run.location_authority)
    if not pricing.ready or any(v is None for v in required):
        return _blocked([PriceChangeBlocker.PRICING_NOT_READY],
                        [b.value for b in pricing.blocking_reasons] or ["required_assessment_unavailable"])
    return assess_price_change_candidates(
        run.jobs, run.cars, readiness=pricing, population=run.population, scheduled=run.scheduled,
        canonical_offers=run.canonical_offers, location_authority=run.location_authority)


def run_price_change_events(raw_dir: str | Path | None = None) -> PriceChangeCandidateResult:
    """Run the validated pricing pipeline, then the gated event engine (read-only; nothing is written)."""
    from ql2_sixt_canada_analysis.pricing_pipeline import run_pricing_pipeline

    return price_change_candidates_from_pipeline(run_pricing_pipeline(raw_dir))
