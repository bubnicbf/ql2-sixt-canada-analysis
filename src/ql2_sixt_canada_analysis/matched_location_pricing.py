"""Matched location pricing: same-job, same-car airport-versus-downtown premiums.

What is compared
----------------
One **matched pair** is one approved city, one shared trusted collection
event, one rental period, one exact approved vehicle product, one currency and
one price basis, with exactly one airport canonical offer and exactly one
canonical downtown offer. The pair-level unique key (:data:`PAIR_KEY_COLUMNS`)
is::

    canonical_city, scheduled_capture_period, pickup_date, return_date,
    car_name, car_type, transmission, seats, bags, currency, price_basis

(plus the approved airport/downtown pair itself, which the city determines
under the current authority: one approved pair per city).

Rules (all fail closed; nothing is inferred from data or labels):

* **Gates first** - pricing readiness must be ``ready`` and must be the
  assessment of exactly the supplied schedule, canonical-offer and
  location-authority reports; the population, canonical offers and frames must
  share one :class:`~ql2_sixt_canada_analysis.pricing_population.FrameBinding`;
  the per-stream schedule assessment must be valid; the pricing-population
  vehicle-stability report must have ``PASSED`` on exactly the
  pricing-eligible rows. Any failure returns a ``BLOCKED`` report with typed
  blockers and no commercial result.
* **Population** - only the canonical offers of the pricing-eligible
  population (:func:`~ql2_sixt_canada_analysis.canonical_offers.assess_canonical_offers`):
  the governed Calgary ``INCOMPLETE_PARENT_CAPTURE`` exclusion and every
  ineligible row never enter; the Vancouver ``Downtown``/``Thurlow`` aliases
  are already one canonical ``Vancouver Downtown`` offer set, so ``Thurlow``
  can never form a second comparison or be counted twice.
* **Locations** - only the effective approved pairs of the location-authority
  report (canonical keys, authority order). Never across cities.
* **Same job** - the canonical offers carry the parent capture's scheduled
  period. The schedule assigns a period only to the single valid claimant of
  its city-period; this module re-proves, on the eligible detail rows and the
  derived linkage key, that every (city, scheduled period) maps to exactly one
  parent capture, so two offers sharing city and period share one trusted
  collection job. Raw ``job_id``, timestamps and scrape order are never used,
  and no job identifier leaves memory.
* **Same car** - exact equality of the approved product identity
  (:data:`~ql2_sixt_canada_analysis.canonical_offers.APPROVED_PRODUCT_COLUMNS`)
  and of the pickup and return dates, as the canonical-offer contract parsed
  them. No trimming, recasing, fuzzy matching or imputation; price is never
  part of the identity.
* **Cardinality** - a match group is one identity on one approved pair.
  Terminal outcomes, in this order: no airport offer (``downtown_only``), no
  downtown offer (``airport_only``), more than one offer on either side
  (``ambiguous``: price-distinct canonical offers with no authority-backed way
  to pair them; never a Cartesian product, row order, minimum or average),
  different currency markers (``currency_mismatch``), different price bases
  (``basis_mismatch``), otherwise ``matched``.
* **Premiums** - fixed direction airport minus downtown in exact integer
  cents; ``premium_percent = 100 * (airport - downtown) / downtown`` computed
  from exact cents. A zero downtown price has no percentage (counted as
  ``zero_denominator``) but keeps its dollar premium. Rounding is display only.

Reports (:class:`MatchedLocationPricingReport`) hold counts, statistics,
enums and approved configuration keys only. The proprietary pair table stays
in memory on :class:`MatchedLocationPricingResult` and is never printed or
written by this module.

Vehicle type comparisons are associational descriptions of observed matched
offers, never causal claims. Hourly captures of the same product are
repeated measurements, not independent draws, so test p-values are
optimistic; a product-level sensitivity test (one median premium per product
and rental period) is reported alongside.

Generate the local (Git-ignored) deliverables after every gate passes::

    python -m ql2_sixt_canada_analysis.matched_location_pricing
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.canonical_offers import APPROVED_PRODUCT_COLUMNS

__all__ = [
    "MATCH_IDENTITY_COLUMNS",
    "MIN_PAIRS_PER_TESTED_VEHICLE_TYPE",
    "MIN_PRODUCTS_PER_TESTED_VEHICLE_TYPE",
    "OVERALL",
    "PAIR_KEY_COLUMNS",
    "PAIR_COLUMNS",
    "CityMatchSummary",
    "DistributionSummary",
    "MatchCounts",
    "MatchOutcome",
    "MatchedLocationPricingBlocker",
    "MatchedLocationPricingError",
    "MatchedLocationPricingReport",
    "MatchedLocationPricingResult",
    "MatchedLocationPricingStatus",
    "PremiumMetric",
    "SignCounts",
    "ComparisonUnit",
    "VehicleTypeSummary",
    "VehicleTypeTest",
    "VehicleTypeTestStatus",
    "assess_matched_location_pricing",
    "city_summary_frame",
    "compare_vehicle_types",
    "holm_adjust",
    "match_count_frame",
    "plot_matched_location_premiums",
    "render_matched_location_pricing_markdown",
    "run_matched_location_pricing",
    "vehicle_type_summary_frame",
    "vehicle_type_test_frame",
    "write_matched_location_pricing_deliverables",
]

#: Minimum valid matched pairs a vehicle type needs to enter a pair-level test (fixed before any result was seen).
MIN_PAIRS_PER_TESTED_VEHICLE_TYPE = 20
#: Minimum distinct products (product identity and rental period) a vehicle type needs in the product-level
#: sensitivity test (fixed before any result was seen).
MIN_PRODUCTS_PER_TESTED_VEHICLE_TYPE = 5
#: The identity two offers must share exactly (besides the approved location pair).
MATCH_IDENTITY_COLUMNS: tuple[str, ...] = ("canonical_city", "scheduled_capture_period", "pickup_date",
                                           "return_date", *APPROVED_PRODUCT_COLUMNS)
#: The unique key of the pair-level table.
PAIR_KEY_COLUMNS: tuple[str, ...] = (*MATCH_IDENTITY_COLUMNS, "currency", "price_basis")
#: Columns of the in-memory pair-level table (proprietary; never printed or written here).
PAIR_COLUMNS: tuple[str, ...] = (*PAIR_KEY_COLUMNS, "airport_location", "downtown_location",
                                 "airport_price_cents", "downtown_price_cents", "premium_cents", "airport_price",
                                 "downtown_price", "premium_dollars", "premium_percent", "percent_valid",
                                 "premium_sign")
#: Label of the all-city summary row.
OVERALL = "overall"
#: Accessible categorical colours (validated reference palette, first three slots; light surface).
_CITY_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
_INK, _INK_SECONDARY, _GRID = "#0b0b0b", "#52514e", "#e4e3df"
_JITTER_SEED = 20261007


class MatchedLocationPricingError(ValueError):
    """The inputs are malformed (a result can never be built from them)."""


class MatchedLocationPricingStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class MatchedLocationPricingBlocker(StrEnum):
    """Why no commercial result was produced (categories only)."""

    PRICING_NOT_READY = "pricing_not_ready"
    READINESS_EVIDENCE_MISMATCH = "readiness_evidence_mismatch"
    FRAME_BINDING_MISMATCH = "frame_binding_mismatch"
    CANONICAL_OFFERS_NOT_READY = "canonical_offers_not_ready"
    LOCATION_PAIRS_INVALID = "location_pairs_invalid"
    SCHEDULE_EVIDENCE_INVALID = "schedule_evidence_invalid"
    SAME_JOB_NOT_PROVEN = "same_job_not_proven"
    VEHICLE_STABILITY_NOT_PASSED = "vehicle_stability_not_passed"
    VEHICLE_STABILITY_POPULATION_MISMATCH = "vehicle_stability_population_mismatch"
    OFFER_CONTRACT_INVALID = "offer_contract_invalid"
    MIXED_PRICE_UNITS = "mixed_price_units"


class MatchOutcome(StrEnum):
    """Terminal outcome of one match group (mutually exclusive, first applicable wins)."""

    DOWNTOWN_ONLY = "downtown_only"
    AIRPORT_ONLY = "airport_only"
    AMBIGUOUS = "ambiguous"
    CURRENCY_MISMATCH = "currency_mismatch"
    BASIS_MISMATCH = "basis_mismatch"
    MATCHED = "matched"


class PremiumMetric(StrEnum):
    DOLLARS = "premium_dollars"
    PERCENT = "premium_percent"


class ComparisonUnit(StrEnum):
    PAIR = "pair"           # every valid matched pair (hourly repeated measurements)
    PRODUCT = "product"     # one median premium per product identity and rental period (sensitivity)


class VehicleTypeTestStatus(StrEnum):
    TESTED = "tested"
    NOT_TESTABLE = "not_testable"


def _count(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MatchedLocationPricingError(f"{name} must be a non-negative int")


def _finite(value: object) -> bool:
    return isinstance(value, float) and math.isfinite(value)


# ------------------------------------------------------------------ result objects


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Distribution of one premium metric (``n == 0`` means every statistic is ``None``; never a fake zero).

    Quantiles use linear interpolation (numpy default); ``std`` is the sample
    standard deviation (``ddof=1``, ``None`` below two values).
    """

    n: int
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q25: float | None = None
    q75: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        _count(self.n, "n")
        stats = (self.mean, self.median, self.q25, self.q75, self.minimum, self.maximum)
        if self.n == 0:
            if any(v is not None for v in (*stats, self.std)):
                raise MatchedLocationPricingError("an empty distribution has no statistics")
            return
        if not all(_finite(v) for v in stats) or (self.n >= 2) != (self.std is not None):
            raise MatchedLocationPricingError("a non-empty distribution needs every statistic")
        tol = 1e-9 * max(1.0, abs(self.minimum), abs(self.maximum))
        if not (self.minimum - tol <= self.q25 <= self.median + tol and self.median <= self.q75 + tol
                and self.q75 <= self.maximum + tol and self.minimum - tol <= self.mean <= self.maximum + tol):
            raise MatchedLocationPricingError("distribution statistics are out of order")
        if self.std is not None and (not _finite(self.std) or self.std < 0):
            raise MatchedLocationPricingError("std must be finite and non-negative")

    @classmethod
    def of(cls, values: Sequence[float] | np.ndarray) -> DistributionSummary:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return cls(0)
        if not np.isfinite(array).all():
            raise MatchedLocationPricingError("premiums must be finite")
        q25, median, q75 = (float(v) for v in np.percentile(array, [25, 50, 75]))
        return cls(int(array.size), float(array.mean()), median,
                   float(array.std(ddof=1)) if array.size >= 2 else None, q25, q75,
                   float(array.min()), float(array.max()))


@dataclass(frozen=True, slots=True)
class SignCounts:
    """Positive (airport premium), zero and negative (airport discount) dollar premiums."""

    positive: int
    zero: int
    negative: int

    def __post_init__(self) -> None:
        for name in ("positive", "zero", "negative"):
            _count(getattr(self, name), name)

    @property
    def total(self) -> int:
        return self.positive + self.zero + self.negative

    def share(self, which: str) -> float | None:
        """Share of ``positive``/``zero``/``negative`` (``None`` for no pairs)."""
        return getattr(self, which) / self.total if self.total else None


@dataclass(frozen=True, slots=True)
class MatchCounts:
    """Match-group accounting (every candidate group has exactly one terminal outcome)."""

    candidate_groups: int
    matched: int
    airport_only: int
    downtown_only: int
    ambiguous: int
    currency_mismatch: int
    basis_mismatch: int
    percent_valid: int
    zero_denominator: int

    def __post_init__(self) -> None:
        for name in self.__slots__:  # type: ignore[attr-defined]
            _count(getattr(self, name), name)
        if self.candidate_groups != (self.matched + self.airport_only + self.downtown_only + self.ambiguous
                                     + self.currency_mismatch + self.basis_mismatch):
            raise MatchedLocationPricingError("candidate groups must equal the sum of terminal outcomes")
        if self.matched != self.percent_valid + self.zero_denominator:
            raise MatchedLocationPricingError("matched pairs must be percent-valid or zero-denominator")

    @property
    def unmatched(self) -> int:
        return self.airport_only + self.downtown_only

    @property
    def incompatible(self) -> int:
        return self.currency_mismatch + self.basis_mismatch

    @property
    def match_rate(self) -> float | None:
        """Valid matched pairs / candidate match groups (``None`` without candidates)."""
        return self.matched / self.candidate_groups if self.candidate_groups else None

    def __add__(self, other: MatchCounts) -> MatchCounts:
        return MatchCounts(*(getattr(self, n) + getattr(other, n) for n in self.__slots__))  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class CityMatchSummary:
    """Counts, sign and distribution of one approved city (or :data:`OVERALL`)."""

    city: str
    airport_location: str | None
    downtown_location: str | None
    counts: MatchCounts
    signs: SignCounts
    dollars: DistributionSummary
    percent: DistributionSummary

    def __post_init__(self) -> None:
        if self.signs.total != self.counts.matched or self.dollars.n != self.counts.matched:
            raise MatchedLocationPricingError("signs and dollar statistics must cover every matched pair")
        if self.percent.n != self.counts.percent_valid:
            raise MatchedLocationPricingError("percentage statistics must cover every percent-valid pair")


@dataclass(frozen=True, slots=True)
class VehicleTypeSummary:
    """Descriptive premiums of one vehicle type in one city."""

    city: str
    car_type: str
    signs: SignCounts
    dollars: DistributionSummary
    percent: DistributionSummary

    def __post_init__(self) -> None:
        if self.signs.total != self.dollars.n or self.dollars.n == 0 or self.percent.n > self.dollars.n:
            raise MatchedLocationPricingError("a vehicle-type summary covers its matched pairs")


@dataclass(frozen=True, slots=True)
class VehicleTypeTest:
    """Kruskal-Wallis comparison of premium distributions across vehicle types within one city."""

    city: str
    metric: PremiumMetric
    unit: ComparisonUnit
    status: VehicleTypeTestStatus
    minimum_per_type: int
    types_observed: int
    types_tested: int
    observations: int
    reason: str | None = None
    statistic: float | None = None
    df: int | None = None
    p_value: float | None = None
    p_holm: float | None = None
    epsilon_squared: float | None = None

    def __post_init__(self) -> None:
        for name in ("minimum_per_type", "types_observed", "types_tested", "observations"):
            _count(getattr(self, name), name)
        values = (self.statistic, self.df, self.p_value, self.p_holm, self.epsilon_squared)
        if self.status is VehicleTypeTestStatus.TESTED:
            if (self.types_tested < 2 or self.df != self.types_tested - 1 or self.reason is not None
                    or not all(_finite(v) for v in (self.statistic, self.p_value, self.epsilon_squared))
                    or not 0.0 <= self.p_value <= 1.0):
                raise MatchedLocationPricingError("a tested comparison needs its full result")
            if self.p_holm is not None and not self.p_value <= self.p_holm <= 1.0:
                raise MatchedLocationPricingError("an adjusted p-value is at least the raw p-value")
        elif any(v is not None for v in values) or not self.reason:
            raise MatchedLocationPricingError("a not-testable comparison carries a reason only")


@dataclass(frozen=True, slots=True)
class MatchedLocationPricingReport:
    """Aggregate, print-safe result: status, blockers, approved pairs, counts and statistics only."""

    status: MatchedLocationPricingStatus
    blockers: tuple[MatchedLocationPricingBlocker, ...] = ()
    readiness_blockers: tuple[str, ...] = ()
    approved_pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()
    shared_collection_events: int = 0
    offers_assessed: int = 0
    offers_outside_approved_pairs: int = 0
    price_units: tuple[tuple[str, str], ...] = ()
    cities: tuple[CityMatchSummary, ...] = ()
    overall: CityMatchSummary | None = None
    vehicle_types: tuple[VehicleTypeSummary, ...] = ()
    vehicle_type_tests: tuple[VehicleTypeTest, ...] = ()

    def __post_init__(self) -> None:
        for name in ("shared_collection_events", "offers_assessed", "offers_outside_approved_pairs"):
            _count(getattr(self, name), name)
        if self.status is MatchedLocationPricingStatus.BLOCKED:
            if not self.blockers or self.cities or self.overall is not None or self.vehicle_types \
                    or self.vehicle_type_tests:
                raise MatchedLocationPricingError("a blocked report has blockers and no commercial result")
            return
        if self.blockers or self.readiness_blockers or self.overall is None or not self.approved_pairs:
            raise MatchedLocationPricingError("a completed report has no blockers and an overall summary")
        expected = tuple(a[0] for a, _ in self.approved_pairs)
        if tuple(c.city for c in self.cities) != expected or self.overall.city != OVERALL:
            raise MatchedLocationPricingError("one city summary per approved pair, in authority order")
        if sum((c.counts for c in self.cities[1:]), self.cities[0].counts) != self.overall.counts:
            raise MatchedLocationPricingError("city counts must sum to the overall counts")
        signs = [sum(getattr(c.signs, s) for c in self.cities) for s in ("positive", "zero", "negative")]
        if SignCounts(*signs) != self.overall.signs:
            raise MatchedLocationPricingError("city signs must sum to the overall signs")
        by_city = {c.city: c.counts.matched for c in self.cities}
        for city in by_city:
            if sum(v.dollars.n for v in self.vehicle_types if v.city == city) != by_city[city]:
                raise MatchedLocationPricingError("vehicle-type summaries must cover every matched pair")
        if len(self.price_units) > 1:
            raise MatchedLocationPricingError("aggregates never mix currencies or price bases")

    @property
    def completed(self) -> bool:
        return self.status is MatchedLocationPricingStatus.COMPLETED

    def city(self, name: str) -> CityMatchSummary:
        return next(c for c in (*self.cities, *((self.overall,) if self.overall else ())) if c.city == name)


@dataclass(frozen=True)
class MatchedLocationPricingResult:
    """The aggregate report plus the proprietary in-memory pair table (``None`` unless completed)."""

    report: MatchedLocationPricingReport
    pairs: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.report, MatchedLocationPricingReport):
            raise TypeError("report must be a MatchedLocationPricingReport")
        if not self.report.completed:
            if self.pairs is not None:
                raise MatchedLocationPricingError("a blocked result holds no pairs")
            return
        pairs = self.pairs
        if not isinstance(pairs, pd.DataFrame) or tuple(pairs.columns) != PAIR_COLUMNS:
            raise MatchedLocationPricingError("a completed result holds the pair table")
        overall = self.report.overall
        if len(pairs) != overall.counts.matched or int(pairs["percent_valid"].sum()) != overall.counts.percent_valid:
            raise MatchedLocationPricingError("the pair table must equal the matched counts")
        if pairs.duplicated(list(PAIR_KEY_COLUMNS)).any():
            raise MatchedLocationPricingError("the pair key must be unique")
        allowed = {(a[1], d[1]) for a, d in self.report.approved_pairs}
        cities = {(a[1], d[1]): a[0] for a, d in self.report.approved_pairs}
        for city, air, down in pairs[["canonical_city", "airport_location", "downtown_location"]].itertuples(
                index=False, name=None):
            if (air, down) not in allowed or cities[(air, down)] != city:
                raise MatchedLocationPricingError("a pair violates the approved city/location relationship")

    @property
    def completed(self) -> bool:
        return self.report.completed


# ------------------------------------------------------------------ assessment


def _blocked(blockers, readiness_blockers=(), pairs=()) -> MatchedLocationPricingResult:  # type: ignore[no-untyped-def]
    return MatchedLocationPricingResult(MatchedLocationPricingReport(
        status=MatchedLocationPricingStatus.BLOCKED, blockers=tuple(dict.fromkeys(blockers)),
        readiness_blockers=tuple(readiness_blockers), approved_pairs=tuple(pairs)))


def _prove_same_job(jobs: pd.DataFrame, cars: pd.DataFrame, population, scheduled,  # type: ignore[no-untyped-def]
                    city_column: str) -> dict[tuple[str, str], int] | None:
    """(city, scheduled period) -> eligible detail rows, when every such key has exactly one parent capture.

    Uses the derived linkage key of the schedule's capture index (never raw
    ``job_id``). Returns ``None`` when any eligible row lacks a period or a
    city-period is claimed by more than one parent capture.
    """
    index = scheduled.capture_periods
    keys = list(index.detail_key_columns)
    if city_column not in cars.columns or any(k not in cars.columns for k in keys):
        return None
    mask = population.detail_mask(jobs, cars)
    periods = index.detail_periods(cars).to_numpy(dtype=object)[mask]
    eligible = cars.loc[mask, [city_column, *keys]].astype(object)
    parents: dict[tuple[str, str], set] = {}
    rows: dict[tuple[str, str], int] = {}
    for period, values in zip(periods, eligible.itertuples(index=False, name=None)):
        city, key = values[0], tuple(values[1:])
        if not isinstance(period, str) or not isinstance(city, str):
            return None
        parents.setdefault((city, period), set()).add(key)
        rows[(city, period)] = rows.get((city, period), 0) + 1
    if any(len(v) != 1 for v in parents.values()):
        return None
    # Parent side: an eligible parent capture's (city, period) is unique as well.
    parent_mask = population.parent_mask(jobs, cars)
    parent_periods = index.parent_periods(jobs).to_numpy(dtype=object)[parent_mask]
    parent_cities = jobs.loc[parent_mask, city_column].astype(object).tolist() if city_column in jobs.columns else []
    seen = [(c, p) for c, p in zip(parent_cities, parent_periods)]
    if len(seen) != int(parent_mask.sum()) or len(set(seen)) != len(seen):
        return None
    return rows


_OFFER_COLUMNS = (*MATCH_IDENTITY_COLUMNS, "canonical_location", "price_cents", "currency", "price_basis")


def _offers_valid(offers: pd.DataFrame) -> bool:
    if any(c not in offers.columns for c in _OFFER_COLUMNS):
        return False
    for column in (*MATCH_IDENTITY_COLUMNS, "canonical_location", "currency", "price_basis"):
        for value in offers[column].astype(object):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return False
            if isinstance(value, str) and (not value or value != value.strip()):
                return False
    for value in offers["price_cents"].astype(object):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            return False
    return True


def assess_matched_location_pricing(jobs: pd.DataFrame, cars: pd.DataFrame, *, readiness: object,
                                    population: object, scheduled: object, canonical_offers: object,
                                    location_authority: object, vehicle_stability: object,
                                    city_column: str = "city") -> MatchedLocationPricingResult:
    """Build same-job, same-car airport/downtown pairs and their premiums (see the module docstring).

    ``jobs``/``cars`` are the analysis-stage frames (authority-backed linkage)
    every report was assessed on. Inputs are never modified.

    Raises:
        TypeError: An argument has the wrong type.
    """
    from ql2_sixt_canada_analysis.canonical_offers import CanonicalOfferReport
    from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport
    from ql2_sixt_canada_analysis.location_authority import LocationAuthorityReport
    from ql2_sixt_canada_analysis.pricing_population import PricingPopulation, PricingPopulationError, frame_binding
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport
    from ql2_sixt_canada_analysis.stability import VehicleStabilityReport

    for value, kind, name in ((readiness, PricingReadinessReport, "readiness"),
                              (population, PricingPopulation, "population"),
                              (scheduled, PerStreamScheduledCoverageReport, "scheduled"),
                              (canonical_offers, CanonicalOfferReport, "canonical_offers"),
                              (location_authority, LocationAuthorityReport, "location_authority"),
                              (vehicle_stability, VehicleStabilityReport, "vehicle_stability")):
        if not isinstance(value, kind):
            raise TypeError(f"{name} must be a {kind.__name__}")
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be DataFrames")
    B = MatchedLocationPricingBlocker
    approved = tuple((tuple(p.airport), tuple(p.downtown)) for p in location_authority.effective_pairs)
    if not readiness.ready:
        return _blocked([B.PRICING_NOT_READY], [b.value for b in readiness.blocking_reasons], approved)
    blockers: list[MatchedLocationPricingBlocker] = []
    if (readiness.canonical_offers is not canonical_offers or readiness.scheduled_coverage is not scheduled
            or readiness.location_authority is not location_authority):
        blockers.append(B.READINESS_EVIDENCE_MISMATCH)
    binding = frame_binding(jobs, cars)
    if population.binding != binding or canonical_offers.binding != binding:
        blockers.append(B.FRAME_BINDING_MISMATCH)
    if not canonical_offers.ready or canonical_offers.unassessable_rows or canonical_offers.offers is None:
        blockers.append(B.CANONICAL_OFFERS_NOT_READY)
    if not location_authority.pairs_valid or not approved:
        blockers.append(B.LOCATION_PAIRS_INVALID)
    if not scheduled.is_valid or scheduled.capture_periods is None or scheduled.jobs_assessed != len(jobs):
        blockers.append(B.SCHEDULE_EVIDENCE_INVALID)
    if not vehicle_stability.is_valid:
        blockers.append(B.VEHICLE_STABILITY_NOT_PASSED)
    elif vehicle_stability.observations_assessed != population.eligible_detail_rows:
        blockers.append(B.VEHICLE_STABILITY_POPULATION_MISMATCH)
    if blockers:
        return _blocked(blockers, (), approved)
    events = _prove_same_job(jobs, cars, population, scheduled, city_column)
    if events is None:
        return _blocked([B.SAME_JOB_NOT_PROVEN], (), approved)
    try:
        offers = canonical_offers.offers_for(jobs, cars)
    except PricingPopulationError:
        return _blocked([B.FRAME_BINDING_MISMATCH], (), approved)
    if not _offers_valid(offers):
        return _blocked([B.OFFER_CONTRACT_INVALID], (), approved)
    offer_events = set(zip(offers["canonical_city"].astype(object), offers["scheduled_capture_period"].astype(object)))
    if not offer_events <= set(events):
        return _blocked([B.SAME_JOB_NOT_PROVEN], (), approved)
    return _match(offers, approved, len(events))


def _sign(cents: int) -> str:
    return "positive" if cents > 0 else ("negative" if cents < 0 else "zero")


def _match(offers: pd.DataFrame, approved: tuple, events: int) -> MatchedLocationPricingResult:
    """Classify every match group and build the pair table (exact, one-to-one, order independent)."""
    B = MatchedLocationPricingBlocker
    membership: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for i, (airport, downtown) in enumerate(approved):
        membership.setdefault(airport, []).append((i, "airport"))
        membership.setdefault(downtown, []).append((i, "downtown"))
    columns = [*MATCH_IDENTITY_COLUMNS, "canonical_location", "price_cents", "currency", "price_basis"]
    width = len(MATCH_IDENTITY_COLUMNS)
    groups: dict[tuple, dict[str, list[tuple[int, str, str]]]] = {}
    outside = 0
    for row in offers.loc[:, columns].astype(object).itertuples(index=False, name=None):
        identity, location = row[:width], row[width]
        cents, currency, basis = int(row[width + 1]), row[width + 2], row[width + 3]
        targets = membership.get((identity[0], location), [])
        if not targets:
            outside += 1
            continue
        for pair_index, side in targets:
            entry = groups.setdefault((pair_index, identity), {"airport": [], "downtown": []})
            entry[side].append((cents, currency, basis))

    O = MatchOutcome
    outcomes: dict[int, dict[MatchOutcome, int]] = {i: {o: 0 for o in O} for i in range(len(approved))}
    rows = []
    for (pair_index, identity) in sorted(groups, key=lambda k: (k[0], tuple(map(str, k[1])))):
        sides = groups[(pair_index, identity)]
        air, down = sides["airport"], sides["downtown"]
        if not air:
            outcome = O.DOWNTOWN_ONLY
        elif not down:
            outcome = O.AIRPORT_ONLY
        elif len(air) > 1 or len(down) > 1:
            outcome = O.AMBIGUOUS
        elif air[0][1] != down[0][1]:
            outcome = O.CURRENCY_MISMATCH
        elif air[0][2] != down[0][2]:
            outcome = O.BASIS_MISMATCH
        else:
            outcome = O.MATCHED
        outcomes[pair_index][outcome] += 1
        if outcome is not O.MATCHED:
            continue
        (a, currency, basis), (d, _, _) = air[0], down[0]
        premium = a - d
        percent = float(Fraction(100 * premium, d)) if d else math.nan
        airport_key, downtown_key = approved[pair_index]
        rows.append((*identity, currency, basis, airport_key[1], downtown_key[1], a, d, premium, a / 100, d / 100,
                     premium / 100, percent, d != 0, _sign(premium)))
    pairs = pd.DataFrame(rows, columns=list(PAIR_COLUMNS)) if rows else pd.DataFrame(
        {c: pd.Series(dtype=object) for c in PAIR_COLUMNS})
    if rows:
        for column in ("airport_price_cents", "downtown_price_cents", "premium_cents"):
            pairs[column] = pairs[column].astype("int64")
        for column in ("airport_price", "downtown_price", "premium_dollars", "premium_percent"):
            pairs[column] = pairs[column].astype(float)
        pairs["percent_valid"] = pairs["percent_valid"].astype(bool)
    else:
        pairs["percent_valid"] = pairs["percent_valid"].astype(bool)
    units = tuple(sorted(set(zip(pairs["currency"].astype(object), pairs["price_basis"].astype(object)))))
    if len(units) > 1:
        return _blocked([B.MIXED_PRICE_UNITS], (), approved)

    cities = tuple(_city_summary(pairs, approved[i][0][0], approved[i], outcomes[i]) for i in range(len(approved)))
    overall_counts = sum((c.counts for c in cities[1:]), cities[0].counts)
    overall = _summary(OVERALL, None, None, overall_counts, pairs)
    report = MatchedLocationPricingReport(
        status=MatchedLocationPricingStatus.COMPLETED, approved_pairs=approved, shared_collection_events=events,
        offers_assessed=len(offers), offers_outside_approved_pairs=outside, price_units=units, cities=cities,
        overall=overall, vehicle_types=_vehicle_type_summaries(pairs, approved),
        vehicle_type_tests=_all_vehicle_type_tests(pairs, approved))
    return MatchedLocationPricingResult(report=report, pairs=pairs)


def _summary(city, airport, downtown, counts: MatchCounts, pairs: pd.DataFrame) -> CityMatchSummary:  # type: ignore[no-untyped-def]
    signs = pairs["premium_sign"].astype(object).tolist()
    valid = pairs.loc[pairs["percent_valid"].to_numpy(dtype=bool), "premium_percent"]
    return CityMatchSummary(city=city, airport_location=airport, downtown_location=downtown, counts=counts,
                            signs=SignCounts(signs.count("positive"), signs.count("zero"), signs.count("negative")),
                            dollars=DistributionSummary.of(pairs["premium_dollars"].to_numpy(dtype=float)),
                            percent=DistributionSummary.of(valid.to_numpy(dtype=float)))


def _city_summary(pairs: pd.DataFrame, city: str, pair: tuple, outcome: dict) -> CityMatchSummary:  # type: ignore[no-untyped-def]
    O = MatchOutcome
    subset = pairs.loc[(pairs["canonical_city"] == city).to_numpy(dtype=bool)
                       & (pairs["airport_location"] == pair[0][1]).to_numpy(dtype=bool)]
    valid = int(subset["percent_valid"].sum())
    counts = MatchCounts(candidate_groups=sum(outcome.values()), matched=outcome[O.MATCHED],
                         airport_only=outcome[O.AIRPORT_ONLY], downtown_only=outcome[O.DOWNTOWN_ONLY],
                         ambiguous=outcome[O.AMBIGUOUS], currency_mismatch=outcome[O.CURRENCY_MISMATCH],
                         basis_mismatch=outcome[O.BASIS_MISMATCH], percent_valid=valid,
                         zero_denominator=len(subset) - valid)
    return _summary(city, pair[0][1], pair[1][1], counts, subset)


# ------------------------------------------------------------------ vehicle type


def _vehicle_type_summaries(pairs: pd.DataFrame, approved: tuple) -> tuple[VehicleTypeSummary, ...]:
    out = []
    for (airport, _) in approved:
        city = airport[0]
        subset = pairs.loc[(pairs["canonical_city"] == city).to_numpy(dtype=bool)]
        for car_type in sorted(set(subset["car_type"].astype(object))):
            group = subset.loc[(subset["car_type"] == car_type).to_numpy(dtype=bool)]
            s = _summary(city, None, None, MatchCounts(len(group), len(group), 0, 0, 0, 0, 0,
                                                       int(group["percent_valid"].sum()),
                                                       len(group) - int(group["percent_valid"].sum())), group)
            out.append(VehicleTypeSummary(city=city, car_type=car_type, signs=s.signs, dollars=s.dollars,
                                          percent=s.percent))
    return tuple(out)


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm step-down adjusted p-values, in input order (deterministic; ties keep input order)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return tuple(adjusted)


def _unit_values(subset: pd.DataFrame, metric: PremiumMetric, unit: ComparisonUnit) -> pd.DataFrame:
    """(car_type, value) observations for the test unit."""
    data = subset.loc[subset["percent_valid"].to_numpy(dtype=bool)] if metric is PremiumMetric.PERCENT else subset
    columns = list(dict.fromkeys(["car_type", *APPROVED_PRODUCT_COLUMNS, "pickup_date", "return_date", metric.value]))
    frame = data.loc[:, columns].astype(object)
    if unit is ComparisonUnit.PAIR:
        return pd.DataFrame({"car_type": frame["car_type"], "value": frame[metric.value].astype(float)})
    keys = [*APPROVED_PRODUCT_COLUMNS, "pickup_date", "return_date"]
    medians = {}
    for values in frame.itertuples(index=False, name=None):
        record = dict(zip(frame.columns, values))
        medians.setdefault(tuple(str(record[k]) for k in keys), (record["car_type"], []))[1].append(
            float(record[metric.value]))
    rows = [(car_type, float(np.median(v))) for _, (car_type, v) in sorted(medians.items())]
    return pd.DataFrame(rows, columns=["car_type", "value"])


def compare_vehicle_types(pairs: pd.DataFrame, city: str, *, metric: PremiumMetric = PremiumMetric.PERCENT,
                          unit: ComparisonUnit = ComparisonUnit.PAIR, minimum: int | None = None) -> VehicleTypeTest:
    """Kruskal-Wallis test of premium distributions across vehicle types in one city (no Holm adjustment).

    Only vehicle types with at least ``minimum`` observations enter
    (default :data:`MIN_PAIRS_PER_TESTED_VEHICLE_TYPE` for pairs,
    :data:`MIN_PRODUCTS_PER_TESTED_VEHICLE_TYPE` for products); fewer than two
    such types, or no variation at all, is ``NOT_TESTABLE``. The effect size
    is epsilon-squared, ``H / (n - 1)`` (Tomczak & Tomczak, 2014). The result
    is associational, never causal.
    """
    from scipy.stats import kruskal

    if minimum is None:
        minimum = MIN_PAIRS_PER_TESTED_VEHICLE_TYPE if unit is ComparisonUnit.PAIR else MIN_PRODUCTS_PER_TESTED_VEHICLE_TYPE
    subset = pairs.loc[(pairs["canonical_city"] == city).to_numpy(dtype=bool)]
    values = _unit_values(subset, metric, unit)
    by_type = {t: values.loc[values["car_type"] == t, "value"].to_numpy(dtype=float)
               for t in sorted(set(values["car_type"]))}
    eligible = {t: v for t, v in by_type.items() if len(v) >= minimum}
    n = int(sum(len(v) for v in eligible.values()))
    common = dict(city=city, metric=metric, unit=unit, minimum_per_type=minimum, types_observed=len(by_type),
                  types_tested=len(eligible), observations=n)
    S = VehicleTypeTestStatus
    if len(eligible) < 2:
        return VehicleTypeTest(status=S.NOT_TESTABLE, reason="fewer_than_two_types_with_minimum_observations",
                               **common)
    if len({float(x) for v in eligible.values() for x in v}) < 2:
        return VehicleTypeTest(status=S.NOT_TESTABLE, reason="no_variation", **common)
    statistic, p_value = kruskal(*eligible.values())
    statistic, p_value = float(statistic), float(p_value)
    return VehicleTypeTest(status=S.TESTED, statistic=statistic, df=len(eligible) - 1, p_value=p_value,
                           epsilon_squared=statistic / (n - 1), **common)


def _all_vehicle_type_tests(pairs: pd.DataFrame, approved: tuple) -> tuple[VehicleTypeTest, ...]:
    """Every city x metric x unit test; Holm adjustment across cities within each (metric, unit) family."""
    import dataclasses

    out = []
    for metric in (PremiumMetric.PERCENT, PremiumMetric.DOLLARS):
        for unit in (ComparisonUnit.PAIR, ComparisonUnit.PRODUCT):
            family = [compare_vehicle_types(pairs, a[0], metric=metric, unit=unit) for a, _ in approved]
            tested = [i for i, t in enumerate(family) if t.status is VehicleTypeTestStatus.TESTED]
            adjusted = holm_adjust([family[i].p_value for i in tested])
            for i, p in zip(tested, adjusted):
                family[i] = dataclasses.replace(family[i], p_holm=p)
            out.extend(family)
    return tuple(out)


# ------------------------------------------------------------------ print-safe tables


def _require_completed(result: object) -> MatchedLocationPricingReport:
    report = result.report if isinstance(result, MatchedLocationPricingResult) else result
    if not isinstance(report, MatchedLocationPricingReport):
        raise TypeError("expected a MatchedLocationPricingResult or MatchedLocationPricingReport")
    if not report.completed:
        raise MatchedLocationPricingError("matched location pricing is blocked: "
                                          + ", ".join(b.value for b in report.blockers))
    return report


def match_count_frame(result: object) -> pd.DataFrame:
    """Match counts and attrition by city and overall (aggregate counts only)."""
    report = _require_completed(result)
    rows = []
    for s in (*report.cities, report.overall):
        c = s.counts
        rows.append({"city": s.city, "candidate_groups": c.candidate_groups, "matched_pairs": c.matched,
                     "airport_only": c.airport_only, "downtown_only": c.downtown_only, "ambiguous": c.ambiguous,
                     "currency_mismatch": c.currency_mismatch, "basis_mismatch": c.basis_mismatch,
                     "percent_valid": c.percent_valid, "zero_denominator": c.zero_denominator,
                     "match_rate": c.match_rate})
    return pd.DataFrame(rows)


def _stats(prefix: str, d: DistributionSummary) -> dict:
    return {f"{prefix}_{k}": getattr(d, k) for k in ("n", "mean", "median", "std", "q25", "q75", "minimum", "maximum")}


def city_summary_frame(result: object) -> pd.DataFrame:
    """Premium distribution and sign shares by city and overall (aggregates only)."""
    report = _require_completed(result)
    rows = []
    for s in (*report.cities, report.overall):
        rows.append({"city": s.city, **_stats("dollars", s.dollars), **_stats("percent", s.percent),
                     "positive": s.signs.positive, "zero": s.signs.zero, "negative": s.signs.negative,
                     "positive_share": s.signs.share("positive"), "zero_share": s.signs.share("zero"),
                     "negative_share": s.signs.share("negative")})
    return pd.DataFrame(rows)


def vehicle_type_summary_frame(result: object) -> pd.DataFrame:
    """Descriptive premiums by city and vehicle type (aggregates only)."""
    report = _require_completed(result)
    rows = []
    for v in report.vehicle_types:
        rows.append({"city": v.city, "car_type": v.car_type, "matched_pairs": v.dollars.n,
                     **{k: getattr(v.dollars, k.split("_", 1)[1]) for k in
                        ("dollars_mean", "dollars_median", "dollars_q25", "dollars_q75")},
                     "percent_n": v.percent.n,
                     **{k: getattr(v.percent, k.split("_", 1)[1]) for k in
                        ("percent_mean", "percent_median", "percent_q25", "percent_q75")},
                     "positive_share": v.signs.share("positive"), "zero_share": v.signs.share("zero"),
                     "negative_share": v.signs.share("negative")})
    return pd.DataFrame(rows)


def vehicle_type_test_frame(result: object) -> pd.DataFrame:
    """Vehicle-type test results (statistics only)."""
    report = _require_completed(result)
    return pd.DataFrame([{"city": t.city, "metric": t.metric.value, "unit": t.unit.value, "status": t.status.value,
                          "reason": t.reason, "minimum_per_type": t.minimum_per_type,
                          "types_observed": t.types_observed, "types_tested": t.types_tested,
                          "observations": t.observations, "statistic": t.statistic, "df": t.df,
                          "p_value": t.p_value, "p_holm": t.p_holm, "epsilon_squared": t.epsilon_squared}
                         for t in report.vehicle_type_tests])


# ------------------------------------------------------------------ visualization


def _dollar(value: float, symbol: str = "$") -> str:
    sign = "-" if value < 0 else ("+" if value > 0 else "")
    return f"{sign}{symbol}{abs(value):,.2f}"


def _percent(value: float) -> str:
    return f"{value:+.1f}%" if value else "0.0%"


def plot_matched_location_premiums(result: MatchedLocationPricingResult, *, figsize: tuple[float, float] = (12.0, 6.2),
                                   dpi: int = 150):  # type: ignore[no-untyped-def]
    """The main commercial figure: dollar and percentage airport premium distributions by city.

    Two panels (dollars, percent), one box plot per approved city in authority
    order with every matched pair drawn as a deterministic jittered point (no
    axis clipping, so no observation is hidden), a zero-premium reference line,
    and per-city annotations of the valid match count and median. Positive
    values are an airport premium, negative an airport discount. Uses an
    object-oriented :class:`matplotlib.figure.Figure` (no pyplot state, never
    shown); the input is not modified.

    Returns:
        ``(figure, (dollar_axes, percent_axes))``.
    """
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    from matplotlib.transforms import blended_transform_factory

    report = _require_completed(result)
    pairs = result.pairs
    cities = [s.city for s in report.cities]
    symbol = report.price_units[0][0] if report.price_units else "$"
    basis = report.price_units[0][1] if report.price_units else "day"
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white", layout="constrained")
    axes = fig.subplots(1, 2)
    panels = ((axes[0], "premium_dollars", f"Dollar premium ({symbol} per {basis})",
               FuncFormatter(lambda v, _: _dollar(v, symbol)), lambda v: _dollar(v, symbol)),
              (axes[1], "premium_percent", "Percentage premium (% of downtown price)",
               FuncFormatter(lambda v, _: _percent(v)), _percent))
    for ax, column, title, formatter, label in panels:
        data = []
        for city in cities:
            subset = pairs.loc[(pairs["canonical_city"] == city).to_numpy(dtype=bool)]
            if column == "premium_percent":
                subset = subset.loc[subset["percent_valid"].to_numpy(dtype=bool)]
            data.append(subset[column].to_numpy(dtype=float))
        positions = np.arange(1, len(cities) + 1)
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1.6, linestyle="--", zorder=1, label="zero premium")
        for i, (values, position) in enumerate(zip(data, positions)):
            if values.size:
                rng = np.random.default_rng(_JITTER_SEED + i)
                ax.scatter(position + rng.uniform(-0.22, 0.22, values.size), values, s=7, alpha=0.22,
                           color=_CITY_COLOURS[i % len(_CITY_COLOURS)], linewidths=0, zorder=2, rasterized=True)
        non_empty = [(v, p) for v, p in zip(data, positions) if v.size]
        if non_empty:
            box = ax.boxplot([v for v, _ in non_empty], positions=[p for _, p in non_empty], widths=0.5,
                             showfliers=False, whis=(0, 100), patch_artist=True, zorder=3,
                             medianprops={"color": _INK, "linewidth": 2.2},
                             whiskerprops={"color": _INK_SECONDARY}, capprops={"color": _INK_SECONDARY},
                             boxprops={"edgecolor": _INK_SECONDARY, "linewidth": 1.2})
            for patch in box["boxes"]:
                patch.set_facecolor((1, 1, 1, 0.55))
        ax.set_xticks(positions, [c.title() for c in cities])
        ax.set_xlim(0.4, len(cities) + 0.6)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_title(title, loc="left", fontsize=12, color=_INK, pad=34)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=_INK_SECONDARY)
        blend = blended_transform_factory(ax.transData, ax.transAxes)
        for values, position in zip(data, positions):
            text = f"n = {values.size:,}\nmedian {label(float(np.median(values)))}" if values.size else "n = 0"
            ax.text(position, 1.01, text, transform=blend, ha="center", va="bottom", fontsize=9, color=_INK)
        ax.text(1.0, 0.99, "▲ airport premium", transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=_INK_SECONDARY)
        ax.text(1.0, 0.01, "▼ airport discount", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color=_INK_SECONDARY)
    axes[0].set_ylabel(f"Airport minus downtown price ({symbol}/{basis})", color=_INK_SECONDARY)
    axes[1].set_ylabel("(Airport - downtown) / downtown", color=_INK_SECONDARY)
    fig.suptitle("Matched airport vs downtown premiums: same job, same car, same rental dates",
                 fontsize=14, color=_INK, x=0.01, ha="left")
    zero = report.overall.counts.zero_denominator
    fig.supxlabel("Boxes: interquartile range with median; whiskers: full range; points: every matched pair "
             "(jittered horizontally). No axis clipping: all observations shown."
             + (f" {zero:,} zero-denominator pairs omitted from the percentage panel." if zero else ""),
             fontsize=8, color=_INK_SECONDARY, x=0.01, ha="left")
    return fig, (axes[0], axes[1])


# ------------------------------------------------------------------ report


def _fmt(value: float | None, kind: str, symbol: str = "$") -> str:
    if value is None:
        return "n/a"
    if kind == "dollar":
        return _dollar(value, symbol)
    if kind == "percent":
        return _percent(value)
    if kind == "share":
        return f"{100 * value:.1f}%"
    if kind == "p":
        return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"
    return f"{value:.3f}"


def _label(key: tuple[str, ...]) -> str:
    return " / ".join(key)


def render_matched_location_pricing_markdown(result: object, *, figure_path: str | None = None) -> str:
    """Markdown report of the aggregate result (counts and statistics only; no rows, identifiers or timestamps).

    A blocked result renders its status and blockers only.
    """
    report = result.report if isinstance(result, MatchedLocationPricingResult) else result
    if not isinstance(report, MatchedLocationPricingReport):
        raise TypeError("expected a MatchedLocationPricingResult or MatchedLocationPricingReport")
    lines = ["# Matched location pricing", "",
             "> Confidential: derived from proprietary QL2 data. Local, Git-ignored artifact; do not commit or "
             "share outside the engagement.", ""]
    if not report.completed:
        lines += ["## Readiness", "", "- Status: **BLOCKED** - no commercial result was calculated.",
                  "- Blockers: " + ", ".join(f"`{b.value}`" for b in report.blockers)]
        if report.readiness_blockers:
            lines.append("- Pricing-readiness blockers: " + ", ".join(f"`{b}`" for b in report.readiness_blockers))
        return "\n".join(lines) + "\n"
    symbol, basis = report.price_units[0] if report.price_units else ("$", "day")
    o = report.overall
    lines += ["## Readiness", "",
              "- Pricing readiness: **ready** (every central `PricingBlocker` clear); readiness bound to the same "
              "schedule, canonical-offer and location-authority reports; population, canonical offers and frames "
              "share one frame binding.",
              "- Canonical offers ready (no unassessable rows; price amount equals the parsed price text).",
              "- Pricing-population vehicle stability: `passed` on exactly the pricing-eligible rows.",
              f"- Same job proven: every (city, scheduled period) of the pricing-eligible rows maps to exactly one "
              f"trusted parent capture ({report.shared_collection_events:,} shared collection events).", "",
              "## Method and matching grain", "",
              "One matched pair = one approved city, one shared trusted collection event, one rental period, one "
              "exact approved vehicle product, one currency and one price basis, with exactly one airport offer and "
              "exactly one canonical downtown offer.", "",
              "Unique key: `" + ", ".join(PAIR_KEY_COLUMNS) + "`.", "",
              "- Premium direction is fixed: airport minus downtown. Positive = airport premium; negative = airport "
              "discount.",
              "- `premium_dollars = airport_price - downtown_price` (exact cents); "
              "`premium_percent = 100 x (airport_price - downtown_price) / downtown_price`.",
              "- Groups with an offer on one side only are unmatched; groups with more than one price-distinct offer "
              "on either side are ambiguous and excluded (no Cartesian product, row order, minimum or average); "
              "currency or price-basis disagreements are incompatible and excluded.",
              "- Match rate = matched pairs / candidate match groups (identities with an offer on at least one side "
              "of an approved pair).",
              f"- Prices compared in `{symbol}` per `{basis}` only (one currency marker and basis in every pair).", "",
              "## Approved location pairs", ""]
    lines += [f"- {_label(a)} versus {_label(d)}" for a, d in report.approved_pairs]
    lines += ["", "The Vancouver `Thurlow` source stream is a governed alias combined into canonical `Vancouver "
              "Downtown` offers before matching; it never forms a second comparison. The governed Calgary incomplete "
              "capture stays in the source data for audit and is excluded from pricing.", "",
              "## Match counts and attrition", "",
              "| City | Candidate groups | Matched | Airport only | Downtown only | Ambiguous | Currency mismatch | "
              "Basis mismatch | % valid | Zero denominator | Match rate |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in (*report.cities, o):
        c = s.counts
        lines.append(f"| {s.city} | {c.candidate_groups:,} | {c.matched:,} | {c.airport_only:,} | "
                     f"{c.downtown_only:,} | {c.ambiguous:,} | {c.currency_mismatch:,} | {c.basis_mismatch:,} | "
                     f"{c.percent_valid:,} | {c.zero_denominator:,} | {_fmt(c.match_rate, 'share')} |")
    lines += ["", f"Canonical offers assessed: {report.offers_assessed:,}; offers outside every approved pair: "
              f"{report.offers_outside_approved_pairs:,}.", "",
              "## Dollar premium by city", "",
              "| City | n | Mean | Median | Std | P25 | P75 | Min | Max |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in (*report.cities, o):
        d = s.dollars
        lines.append(f"| {s.city} | {d.n:,} | " + " | ".join(
            _fmt(getattr(d, k), "dollar", symbol) for k in ("mean", "median", "std", "q25", "q75", "minimum",
                                                            "maximum")) + " |")
    lines += ["", "## Percentage premium by city", "",
              "| City | n | Mean | Median | P25 | P75 | Min | Max |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in (*report.cities, o):
        p = s.percent
        lines.append(f"| {s.city} | {p.n:,} | " + " | ".join(
            _fmt(getattr(p, k), "percent") for k in ("mean", "median", "q25", "q75", "minimum", "maximum")) + " |")
    lines += ["", "## Sign of the premium", "",
              "| City | Airport higher | Equal | Airport lower |", "| --- | ---: | ---: | ---: |"]
    for s in (*report.cities, o):
        g = s.signs
        lines.append(f"| {s.city} | {g.positive:,} ({_fmt(g.share('positive'), 'share')}) | {g.zero:,} "
                     f"({_fmt(g.share('zero'), 'share')}) | {g.negative:,} ({_fmt(g.share('negative'), 'share')}) |")
    lines += ["", "## Distribution interpretation", ""]
    lines += [_distribution_sentence(s, symbol) for s in report.cities]
    lines += ["", "## Vehicle type", "",
              "Descriptive premiums by city and `car_type` (percentage premiums on percent-valid pairs):", "",
              "| City | Vehicle type | n | Median $ | P25 $ | P75 $ | Median % | P25 % | P75 % | Airport higher | "
              "Equal | Airport lower |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for v in report.vehicle_types:
        lines.append(f"| {v.city} | {v.car_type} | {v.dollars.n:,} | {_fmt(v.dollars.median, 'dollar', symbol)} | "
                     f"{_fmt(v.dollars.q25, 'dollar', symbol)} | {_fmt(v.dollars.q75, 'dollar', symbol)} | "
                     f"{_fmt(v.percent.median, 'percent')} | {_fmt(v.percent.q25, 'percent')} | "
                     f"{_fmt(v.percent.q75, 'percent')} | {_fmt(v.signs.share('positive'), 'share')} | "
                     f"{_fmt(v.signs.share('zero'), 'share')} | {_fmt(v.signs.share('negative'), 'share')} |")
    lines += ["", f"Kruskal-Wallis tests within each city. Pair unit: vehicle types with at least "
              f"{MIN_PAIRS_PER_TESTED_VEHICLE_TYPE} matched pairs; product unit (sensitivity): one median premium per "
              f"product identity and rental period, types with at least {MIN_PRODUCTS_PER_TESTED_VEHICLE_TYPE} "
              "products. Holm adjustment across the city tests of each metric/unit family; effect size "
              "epsilon-squared = H / (n - 1).", "",
              "| City | Metric | Unit | Status | Types tested / observed | n | H | df | p | Holm p | epsilon-squared |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in report.vehicle_type_tests:
        status = t.status.value if t.reason is None else f"{t.status.value} ({t.reason})"
        lines.append(f"| {t.city} | {t.metric.value} | {t.unit.value} | {status} | {t.types_tested} / "
                     f"{t.types_observed} | {t.observations:,} | {_fmt(t.statistic, 'num')} | "
                     f"{t.df if t.df is not None else 'n/a'} | {_fmt(t.p_value, 'p')} | {_fmt(t.p_holm, 'p')} | "
                     f"{_fmt(t.epsilon_squared, 'num')} |")
    lines += ["", _vehicle_sentence(report), "", "## Commercial interpretation", ""]
    lines += _commercial_lines(report, symbol)
    lines += ["", "## Limitations", "",
              "- Observational and associational: listed prices of matched offers, not transactions, demand, "
              "availability or fees beyond the listed price basis. No causal claim is made about why airport "
              "prices differ or why the difference varies by vehicle type.",
              "- One supplier (Sixt), one source mode, three cities and the collection window of this feed; the "
              "rental-period mix is whatever the collection requested. Results do not generalise beyond them.",
              "- Hourly captures of the same product are repeated measurements, so pair-level p-values are "
              "optimistic and must not be read as evidence strength on their own. "
              + ("The product-level sensitivity test and the effect sizes should carry more weight."
                 if _sensitivity_testable(report) else
                 f"The product-level sensitivity test was not testable (no city has two vehicle types with at least "
                 f"{MIN_PRODUCTS_PER_TESTED_VEHICLE_TYPE} distinct products), so vehicle type and product are nearly "
                 "confounded and vehicle-type conclusions rest on the descriptive magnitudes, not on p-values."),
              "- Unmatched and ambiguous groups are excluded; if assortment differs systematically between airport "
              "and downtown, matched premiums describe only the shared assortment.",
              "- `car_type` is the source classification used exactly as published; vehicle types with few pairs "
              "are described but not tested.", "",
              "## Main visualization", "",
              f"`{figure_path}`" if figure_path else "Not generated."]
    return "\n".join(lines) + "\n"


def _distribution_sentence(s: CityMatchSummary, symbol: str) -> str:
    d, p, g = s.dollars, s.percent, s.signs
    if d.n == 0:
        return f"- **{s.city}**: no valid matched pairs; no premium is reported."
    pct = (f" ({_fmt(p.median, 'percent')}; middle half {_fmt(p.q25, 'percent')} to {_fmt(p.q75, 'percent')})"
           if p.n else "")
    skew = "above" if d.mean > d.median else ("below" if d.mean < d.median else "equal to")
    return (f"- **{s.city}**: median dollar premium {_fmt(d.median, 'dollar', symbol)}{pct}; middle half of pairs "
            f"between {_fmt(d.q25, 'dollar', symbol)} and {_fmt(d.q75, 'dollar', symbol)}, full range "
            f"{_fmt(d.minimum, 'dollar', symbol)} to {_fmt(d.maximum, 'dollar', symbol)}; the mean is {skew} the "
            f"median. Airport higher in {_fmt(g.share('positive'), 'share')} of {d.n:,} pairs, equal in "
            f"{_fmt(g.share('zero'), 'share')}, lower in {_fmt(g.share('negative'), 'share')}.")


def _vehicle_sentence(report: MatchedLocationPricingReport) -> str:
    primary = [t for t in report.vehicle_type_tests
               if t.metric is PremiumMetric.PERCENT and t.unit is ComparisonUnit.PAIR]
    sensitivity = {t.city: t for t in report.vehicle_type_tests
                   if t.metric is PremiumMetric.PERCENT and t.unit is ComparisonUnit.PRODUCT}
    parts = []
    for t in primary:
        medians = [v.percent.median for v in report.vehicle_types
                   if v.city == t.city and v.dollars.n >= MIN_PAIRS_PER_TESTED_VEHICLE_TYPE and v.percent.n]
        spread = (f"tested-type median premiums range {_fmt(min(medians), 'percent')} to "
                  f"{_fmt(max(medians), 'percent')}" if medians else "no type meets the minimum")
        if t.status is VehicleTypeTestStatus.TESTED:
            size = ("small" if t.epsilon_squared < 0.06 else "moderate" if t.epsilon_squared < 0.14 else "large")
            sens = sensitivity.get(t.city)
            sens_text = (f"product-level sensitivity Holm p {_fmt(sens.p_holm, 'p')}, epsilon-squared "
                         f"{_fmt(sens.epsilon_squared, 'num')}" if sens and sens.status is VehicleTypeTestStatus.TESTED
                         else f"product-level sensitivity not testable ({sens.reason})" if sens else "")
            parts.append(f"- **{t.city}**: {spread}; Holm p {_fmt(t.p_holm, 'p')}, epsilon-squared "
                         f"{_fmt(t.epsilon_squared, 'num')} ({size} effect by common benchmarks); {sens_text}.")
        else:
            parts.append(f"- **{t.city}**: not testable ({t.reason}); {spread}.")
    return ("Percentage premium by vehicle type (associational, not causal):\n\n" + "\n".join(parts)) if parts else ""


def _commercial_lines(report: MatchedLocationPricingReport, symbol: str) -> list[str]:
    o = report.overall
    out = []
    if o.dollars.n:
        out.append(f"- Across {o.counts.matched:,} same-job, same-car matched pairs, the airport listed price is higher "
                   f"in {_fmt(o.signs.share('positive'), 'share')} of comparisons, with a median premium of "
                   f"{_fmt(o.dollars.median, 'dollar', symbol)} ({_fmt(o.percent.median, 'percent')}).")
    ranked = sorted((s for s in report.cities if s.percent.n), key=lambda s: s.percent.median, reverse=True)
    if ranked:
        out.append("- Ranking by median percentage premium: " + ", ".join(
            f"{s.city} {_fmt(s.percent.median, 'percent')}" for s in ranked) + ".")
    mixed = [s.city for s in report.cities if s.signs.positive and s.signs.negative]
    if mixed:
        out.append("- The premium is not uniform: " + ", ".join(mixed) + " show both airport premiums and airport "
                   "discounts for identical offers, so a single average understates the dispersion a revenue "
                   "manager would see.")
    flat = [s.city for s in report.cities if s.percent.n and s.percent.q75 - s.percent.q25 < 0.5]
    if flat:
        out.append("- " + ", ".join(flat) + ": the middle half of pairs shares one percentage premium (interquartile "
                   "width below 0.5 points), consistent with a uniform proportional airport markup rather than "
                   "product-by-product pricing; worth confirming with the supplier's rate structure.")
    discounts = [f"{v.city} {v.car_type} ({_fmt(v.percent.median, 'percent')})" for v in report.vehicle_types
                 if v.percent.n and v.percent.median < 0]
    if discounts:
        out.append("- Vehicle types where the airport is typically cheaper than downtown (negative median premium): "
                   + ", ".join(discounts) + ". These run against the city pattern and are the first candidates "
                   "for a rate-integrity or positioning review.")
    out.append("- Investigation candidates: the cities and vehicle types whose middle half of pairs sits furthest "
               "from zero, and any city where the sign changes across captures. A pricing decision would also need "
               "booking-volume, conversion and competitor-rate data, all-in prices (fees, taxes) and longer coverage.")
    return out


def _sensitivity_testable(report: MatchedLocationPricingReport) -> bool:
    return any(t.unit is ComparisonUnit.PRODUCT and t.status is VehicleTypeTestStatus.TESTED
               for t in report.vehicle_type_tests)


# ------------------------------------------------------------------ orchestration and deliverables


def run_matched_location_pricing(raw_dir: str | Path | None = None) -> MatchedLocationPricingResult:
    """Run the validated pricing pipeline and the matched-location analysis (fails closed unless ready)."""
    from ql2_sixt_canada_analysis.pricing_pipeline import run_pricing_pipeline
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport

    run = run_pricing_pipeline(raw_dir)
    pricing = run.pricing
    if not isinstance(pricing, PricingReadinessReport):
        raise TypeError("the pipeline produced no readiness report")
    approved = tuple((tuple(p.airport), tuple(p.downtown)) for p in run.location_authority.effective_pairs)
    if not pricing.ready or None in (run.population, run.scheduled, run.canonical_offers, run.vehicle_stability):
        return _blocked([MatchedLocationPricingBlocker.PRICING_NOT_READY],
                        [b.value for b in pricing.blocking_reasons] or ["required_assessment_unavailable"], approved)
    return assess_matched_location_pricing(
        run.jobs, run.cars, readiness=pricing, population=run.population, scheduled=run.scheduled,
        canonical_offers=run.canonical_offers, location_authority=run.location_authority,
        vehicle_stability=run.vehicle_stability)


def write_matched_location_pricing_deliverables(result: MatchedLocationPricingResult, *, report_path: str | Path,
                                                figure_path: str | Path) -> tuple[Path, Path]:
    """Write the Markdown report and the main figure (completed results only; paths must be supplied)."""
    _require_completed(result)
    report_path, figure_path = Path(report_path), Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_matched_location_premiums(result)
    fig.savefig(figure_path, dpi=fig.dpi, metadata={"Software": None})
    try:
        relative = figure_path.resolve().relative_to(report_path.resolve().parent).as_posix()
    except ValueError:
        relative = figure_path.name
    report_path.write_text(render_matched_location_pricing_markdown(result, figure_path=relative), encoding="utf-8")
    return report_path, figure_path


def main(argv: list[str] | None = None) -> int:
    """Generate ``reports/matched_location_pricing.md`` and ``reports/figures/matched_location_pricing.png``."""
    from ql2_sixt_canada_analysis.paths import PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Matched airport/downtown pricing (local, Git-ignored outputs).")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    args = parser.parse_args(argv)
    result = run_matched_location_pricing(args.raw_dir)
    if not result.completed:
        print(render_matched_location_pricing_markdown(result), end="")
        return 2
    reports = Path(args.reports_dir)
    write_matched_location_pricing_deliverables(result, report_path=reports / "matched_location_pricing.md",
                                                figure_path=reports / "figures" / "matched_location_pricing.png")
    print("Matched location pricing completed; report and figure written to the reports directory.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
