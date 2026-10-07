"""Authority-backed canonical offer combination (``CANONICAL_OFFER_COMBINATION``, record v8 onwards).

The approved alias keeps ``vancouver / Vancouver Downtown`` and ``vancouver /
Vancouver Thurlow`` as separately required source streams (each keeps its own
schedule and coverage) that canonicalize to ``vancouver / Vancouver Downtown``.
Every other approved stream canonicalizes to itself.

Combination rules (all from the approved resolution; nothing is inferred
from data):

* **Validation before combination** - only detail rows in the pricing-eligible
  population (:class:`~ql2_sixt_canada_analysis.pricing_population.PricingPopulation`:
  trusted linkage and city integrity, reporting-day agreement, rental-date
  validity, an assigned capture period, outside the governed Calgary
  exclusion) are combined. Rows of the governed exclusion are out of scope
  (counted, kept in the source frames).
* **Union** - order-independent with no stream priority; source row order and
  ``row_index`` are never used. Output order is the sorted identity.
* **Exact semantic identity** - canonical location, scheduled capture period
  (UTC start text), strictly parsed pickup and return dates, the approved
  product identity (:data:`APPROVED_PRODUCT_COLUMNS`, the established
  comparison product definition without its dates), normalized price (exact
  cents), price basis and currency marker. The currency marker and basis are
  parsed strictly from ``price_per_day`` and its amount must equal
  ``price_num``; currencies are never assumed equivalent.
* **Exact duplicates** - one deterministic canonical row with provenance
  (raw source location labels, contributing observation count, unique or
  deduplicated).
* **Price variation** - observations that share every identity component but
  price are separate canonical offers flagged ``price_variation``: no
  averaging, no minimum or maximum choice, nothing discarded.
* **Fail closed** - an in-scope row that fails a foundational control or lacks
  an identity component is *unassessable*: it is counted by reason, never
  dropped silently, and missing values never compare equal. Any unassessable
  row blocks readiness.

Reports hold counts, enums and approved stream keys only; the canonical
offers themselves (``CanonicalOfferReport.offers``) stay in memory, are bound
to the exact frames they were built from and are never printed.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from functools import cache

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.authority_decisions import (
    CANONICAL_OFFER_IDENTITY_COMPONENTS,
    AuthorityDecisionRecord,
    DecisionId,
    load_current_decision_record,
)
from ql2_sixt_canada_analysis.pricing_population import (
    DetailEligibility,
    FrameBinding,
    PricingPopulation,
    PricingPopulationError,
)

__all__ = [
    "APPROVED_PRODUCT_COLUMNS",
    "CanonicalOfferBlocker",
    "CanonicalOfferPolicy",
    "CanonicalOfferReport",
    "CanonicalOfferStatus",
    "UnassessableOfferReason",
    "assess_canonical_offers",
    "canonical_offer_policy_from_record",
    "current_canonical_offer_policy",
    "parse_price_text",
]

Key = tuple[str, ...]
#: The approved product identity: the established comparison product definition
#: (``LocationStreamComparisonDefinition.product_columns``) without its rental dates,
#: which are separate identity components.
APPROVED_PRODUCT_COLUMNS = ("car_name", "car_type", "transmission", "seats", "bags")
_PRICE_TEXT = re.compile(r"(?P<currency>[A-Z]{0,3}\$)(?P<whole>[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.(?P<cents>[0-9]{2})"
                         r"/(?P<basis>[a-z]+)", re.ASCII)
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)
_OFFER_COLUMNS = ("canonical_city", "canonical_location", "scheduled_capture_period", "pickup_date", "return_date",
                  *APPROVED_PRODUCT_COLUMNS, "price_cents", "price_basis", "currency", "source_location_labels",
                  "observation_count", "provenance", "price_variation")


class CanonicalOfferStatus(StrEnum):
    APPROVED = "approved"
    NOT_APPROVED = "not_approved"
    RECORD_UNAVAILABLE = "record_unavailable"
    INVALID = "invalid"          # approved but not implementable against the approved alias or contract


class CanonicalOfferBlocker(StrEnum):
    POLICY_UNAVAILABLE = "canonical_offer_policy_unavailable"
    UNASSESSABLE_OFFERS = "canonical_offers_unassessable"


class UnassessableOfferReason(StrEnum):
    """Why an in-scope detail row cannot be combined (counts only)."""

    REPORTING_DAY_FAILED = "reporting_day_failed"
    RENTAL_DATES_FAILED = "rental_dates_failed"
    CAPTURE_PERIOD_UNASSIGNED = "capture_period_unassigned"
    LOCATION_NOT_APPROVED = "location_not_approved"
    RENTAL_DATE_UNPARSED = "rental_date_unparsed"
    PRODUCT_INCOMPLETE = "product_incomplete"
    PRICE_INVALID = "price_invalid"
    PRICE_TEXT_UNPARSED = "price_text_unparsed"
    PRICE_TEXT_DISAGREES = "price_text_disagrees"


# ------------------------------------------------------------------ policy


@dataclass(frozen=True)
class CanonicalOfferPolicy:
    """The approved combination policy (or why it is unavailable)."""

    status: CanonicalOfferStatus
    record_id: str | None = None
    source_streams: tuple[Key, ...] = ()
    canonical_location: Key | None = None
    identity: tuple[str, ...] = ()
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        approved = self.status is CanonicalOfferStatus.APPROVED
        if approved != (self.canonical_location is not None and len(self.source_streams) == 2):
            raise ValueError("an approved policy names exactly two source streams and a canonical location")
        if approved and (self.canonical_location not in self.source_streams or not self.record_id
                         or not self.references or set(self.identity) != set(CANONICAL_OFFER_IDENTITY_COMPONENTS)):
            raise ValueError("an approved policy is complete and authority-backed")

    @property
    def available(self) -> bool:
        return self.status is CanonicalOfferStatus.APPROVED

    def canonical(self, stream: Key) -> Key:
        """The canonical location of an approved source stream (aliased streams share one)."""
        return self.canonical_location if self.available and tuple(stream) in self.source_streams else tuple(stream)


def canonical_offer_policy_from_record(record: AuthorityDecisionRecord | None,
                                       contract: object = None) -> CanonicalOfferPolicy:
    """Build the policy from the APPROVED decision only (fails closed; schema 1-3 records have none)."""
    S = CanonicalOfferStatus
    if record is None:
        return CanonicalOfferPolicy(S.RECORD_UNAVAILABLE)
    resolution = record.approved_resolution(DecisionId.CANONICAL_OFFER_COMBINATION)
    if resolution is None:
        return CanonicalOfferPolicy(S.NOT_APPROVED, record.record_id)
    try:
        streams = tuple(tuple(k) for k in resolution["source_streams"])
        canonical = tuple(resolution["canonical_location"])
        alias = record.approved_resolution(DecisionId.VANCOUVER_LOCATION_IDENTITY)
        if alias is None or alias.get("state") != "CONFIRMED_ALIAS" or tuple(alias["canonical_location"]) != canonical:
            raise ValueError("the combination must follow the approved alias")
        if contract is not None:
            from ql2_sixt_canada_analysis.expected_stream_contract import ExpectedStreamContract
            if not isinstance(contract, ExpectedStreamContract) or not set(streams) <= set(contract.expected_keys):
                raise ValueError("every source stream must be an approved expected stream")
        entry = record.decision(DecisionId.CANONICAL_OFFER_COMBINATION)
        return CanonicalOfferPolicy(S.APPROVED, record.record_id, tuple(sorted(streams)), canonical,
                                    tuple(resolution["identity"]),
                                    tuple(sorted({a.reference for a in entry.authority})))
    except (ValueError, KeyError, TypeError):
        return CanonicalOfferPolicy(S.INVALID, record.record_id)


@cache
def current_canonical_offer_policy() -> CanonicalOfferPolicy:
    """The policy of the current committed record (cached)."""
    from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
    return canonical_offer_policy_from_record(load_current_decision_record(), current_expected_stream_contract())


# ------------------------------------------------------------------ parsing


def parse_price_text(text: object) -> tuple[str, int, str] | None:
    """``(currency marker, amount in cents, basis)`` of an exact price text, else ``None`` (no trimming)."""
    if not isinstance(text, str):
        return None
    match = _PRICE_TEXT.fullmatch(text)
    if match is None:
        return None
    cents = int(match["whole"].replace(",", "")) * 100 + int(match["cents"])
    return match["currency"], cents, match["basis"]


def _cents(value: object) -> int | None:
    """Exact non-negative cents of a finite number with at most two decimals, else ``None``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    try:
        exact = Decimal(repr(float(value))) if isinstance(value, (float, np.floating)) else Decimal(int(value))
    except InvalidOperation:
        return None
    scaled = exact * 100
    return int(scaled) if scaled == scaled.to_integral_value() else None


def _date(value: object) -> dt.date | None:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _product(value: object) -> object:
    """An exact product component, or ``None`` (missing values never compare equal)."""
    if isinstance(value, str):
        return value if value.strip() and value == value.strip() else None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    if not math.isfinite(float(value)):
        return None
    number = Decimal(repr(float(value))) if isinstance(value, (float, np.floating)) else Decimal(int(value))
    return ("number", str(number.normalize()) if number != number.to_integral_value() else str(int(number)))


# ------------------------------------------------------------------ report


@dataclass(frozen=True)
class CanonicalOfferReport:
    """Aggregate result of the combination (counts and approved keys only; offers stay in memory)."""

    policy: CanonicalOfferPolicy
    binding: FrameBinding | None
    source_rows: int = 0
    out_of_scope_rows: int = 0          # governed parent-capture exclusion (kept in the source frames)
    combined_observations: int = 0
    unassessable_rows: int = 0
    canonical_offers: int = 0
    unique_offers: int = 0
    deduplicated_offers: int = 0
    duplicate_groups: int = 0
    collapsed_observations: int = 0     # observations merged into another canonical row
    variation_groups: int = 0
    variation_offers: int = 0
    cross_stream_offers: int = 0        # canonical offers observed in more than one source stream
    unassessable_groups: tuple[tuple[str, int], ...] = ()
    source_counts: tuple[tuple[Key, int], ...] = ()       # combined observations per source stream
    canonical_counts: tuple[tuple[Key, int], ...] = ()    # canonical offers per canonical location
    offers: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        counts = [v for k, v in vars(self).items() if isinstance(v, int) and not isinstance(v, bool)]
        if any(v < 0 for v in counts):
            raise ValueError("counts are non-negative")
        if self.source_rows != self.out_of_scope_rows + self.combined_observations + self.unassessable_rows:
            raise ValueError("every source row is out of scope, combined or unassessable")
        if self.unassessable_rows != sum(n for _, n in self.unassessable_groups):
            raise ValueError("unassessable rows must equal their grouped counts")
        if (self.canonical_offers != self.unique_offers + self.deduplicated_offers
                or self.combined_observations != self.canonical_offers + self.collapsed_observations
                or self.duplicate_groups != self.deduplicated_offers
                or self.combined_observations != sum(n for _, n in self.source_counts)
                or self.canonical_offers != sum(n for _, n in self.canonical_counts)
                or self.variation_offers < 2 * self.variation_groups):
            raise ValueError("the offer counts are inconsistent")
        if self.offers is not None and (len(self.offers) != self.canonical_offers
                                        or int(self.offers["observation_count"].sum()) != self.combined_observations):
            raise ValueError("the offers do not match the counts")

    @property
    def blocking_reasons(self) -> tuple[CanonicalOfferBlocker, ...]:
        B = CanonicalOfferBlocker
        found = []
        if not self.policy.available or self.binding is None:
            found.append(B.POLICY_UNAVAILABLE)
        if self.unassessable_rows:
            found.append(B.UNASSESSABLE_OFFERS)
        return tuple(found)

    @property
    def ready(self) -> bool:
        return not self.blocking_reasons

    def offers_for(self, jobs: pd.DataFrame, cars: pd.DataFrame) -> pd.DataFrame:
        """The canonical offers, only for the exact frames they were built from (stale evidence is refused)."""
        from ql2_sixt_canada_analysis.pricing_population import frame_binding
        if self.binding is None or self.offers is None or frame_binding(jobs, cars) != self.binding:
            raise PricingPopulationError("the frames differ from the evidence these offers were built from")
        return self.offers.copy()


def assess_canonical_offers(jobs: pd.DataFrame, cars: pd.DataFrame, *, population: PricingPopulation,
                            scheduled: object, policy: CanonicalOfferPolicy,
                            location_columns: tuple[str, str] = ("city", "location"),
                            expected_streams: tuple[Key, ...] | None = None) -> CanonicalOfferReport:
    """Combine the pricing-eligible offers into canonical offers (see the module docstring).

    ``population`` and ``scheduled`` must have been built from exactly these
    frames. ``expected_streams`` (default: the scheduled report's contract
    keys) are the approved source streams; any other location is
    unassessable.

    Raises:
        PricingPopulationError: The population or schedule was built from other frames.
        TypeError: Wrong argument types.
    """
    from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport

    if not isinstance(policy, CanonicalOfferPolicy):
        raise TypeError("policy must be a CanonicalOfferPolicy")
    if not isinstance(population, PricingPopulation):
        raise TypeError("population must be a PricingPopulation")
    if not isinstance(scheduled, PerStreamScheduledCoverageReport) or scheduled.capture_periods is None:
        raise TypeError("scheduled must be an assessed PerStreamScheduledCoverageReport")
    population.check_frames(jobs, cars)
    if scheduled.jobs_assessed != len(jobs):
        raise PricingPopulationError("the scheduled-coverage report was assessed on other frames")
    if not policy.available:
        return CanonicalOfferReport(policy=policy, binding=None)
    needed = (*location_columns, "pickup_date", "return_date", *APPROVED_PRODUCT_COLUMNS, "price_num",
              "price_per_day")
    if any(c not in cars.columns for c in needed):
        raise PricingPopulationError("a required offer column is absent")
    if expected_streams is None:
        coverage = scheduled.coverage
        expected_streams = tuple(tuple(k) for k in (coverage.expected_locations if coverage is not None else ()))
    approved = {tuple(k) for k in expected_streams}

    R, E = UnassessableOfferReason, DetailEligibility
    foundational = {E.REPORTING_DAY_FAILED.value: R.REPORTING_DAY_FAILED,
                    E.RENTAL_DATES_FAILED.value: R.RENTAL_DATES_FAILED,
                    E.CAPTURE_PERIOD_UNASSIGNED.value: R.CAPTURE_PERIOD_UNASSIGNED}
    periods = scheduled.capture_periods.detail_periods(cars).tolist()
    columns = {c: cars[c].astype(object).tolist() for c in needed}
    unassessable: dict[str, int] = {}
    out_of_scope = 0
    groups: dict[tuple, list[tuple[Key, int]]] = {}

    def fail(reason: UnassessableOfferReason) -> None:
        unassessable[reason.value] = unassessable.get(reason.value, 0) + 1

    for i, status in enumerate(population.detail_status):
        if status == E.GOVERNED_EXCLUSION.value:
            out_of_scope += 1
            continue
        if status != E.ELIGIBLE.value:
            fail(foundational[status])
            continue
        stream = tuple(columns[c][i] for c in location_columns)
        if stream not in approved:
            fail(R.LOCATION_NOT_APPROVED)
            continue
        pickup, ret = _date(columns["pickup_date"][i]), _date(columns["return_date"][i])
        if pickup is None or ret is None:
            fail(R.RENTAL_DATE_UNPARSED)
            continue
        product = tuple(_product(columns[c][i]) for c in APPROVED_PRODUCT_COLUMNS)
        if any(p is None for p in product):
            fail(R.PRODUCT_INCOMPLETE)
            continue
        cents = _cents(columns["price_num"][i])
        if cents is None:
            fail(R.PRICE_INVALID)
            continue
        text = parse_price_text(columns["price_per_day"][i])
        if text is None:
            fail(R.PRICE_TEXT_UNPARSED)
            continue
        currency, text_cents, basis = text
        if text_cents != cents:
            fail(R.PRICE_TEXT_DISAGREES)
            continue
        period = periods[i]
        if not isinstance(period, str):      # defensive: eligibility already requires a period
            fail(R.CAPTURE_PERIOD_UNASSIGNED)
            continue
        identity = (policy.canonical(stream), period, pickup, ret, product, cents, basis, currency)
        groups.setdefault(identity, []).append((stream, i))

    def sort_key(identity: tuple) -> tuple:
        location, period, pickup, ret, product, cents, basis, currency = identity
        return (location, period, pickup, ret, tuple(repr(p) for p in product), currency, basis, cents)

    ordered = sorted(groups, key=sort_key)
    variation: dict[tuple, set[int]] = {}
    for identity in ordered:
        variation.setdefault(identity[:5] + identity[6:], set()).add(identity[5])
    rows = []
    source_counts: dict[Key, int] = {}
    canonical_counts: dict[Key, int] = {}
    for identity in ordered:
        location, period, pickup, ret, product, cents, basis, currency = identity
        members = groups[identity]
        labels = tuple(sorted({s[1] for s, _ in members}))
        for s, _ in members:
            source_counts[s] = source_counts.get(s, 0) + 1
        canonical_counts[location] = canonical_counts.get(location, 0) + 1
        rows.append((location[0], location[1], period, pickup, ret,
                     *(p if isinstance(p, str) else p[1] for p in product), cents, basis, currency,
                     "|".join(labels), len(members), "deduplicated" if len(members) > 1 else "unique",
                     len(variation[identity[:5] + identity[6:]]) > 1))
    offers = pd.DataFrame(rows, columns=list(_OFFER_COLUMNS)) if rows else pd.DataFrame(columns=list(_OFFER_COLUMNS))
    offers["observation_count"] = offers["observation_count"].astype(int)
    dedup = sum(1 for g in groups.values() if len(g) > 1)
    varying = [v for v in variation.values() if len(v) > 1]
    return CanonicalOfferReport(
        policy=policy, binding=population.binding, source_rows=len(population.detail_status),
        out_of_scope_rows=out_of_scope, combined_observations=sum(len(g) for g in groups.values()),
        unassessable_rows=sum(unassessable.values()), canonical_offers=len(groups),
        unique_offers=len(groups) - dedup, deduplicated_offers=dedup, duplicate_groups=dedup,
        collapsed_observations=sum(len(g) - 1 for g in groups.values()),
        variation_groups=len(varying), variation_offers=sum(len(v) for v in varying),
        cross_stream_offers=sum(1 for g in groups.values() if len({s for s, _ in g}) > 1),
        unassessable_groups=tuple(sorted(unassessable.items())),
        source_counts=tuple(sorted(source_counts.items())), canonical_counts=tuple(sorted(canonical_counts.items())),
        offers=offers)

