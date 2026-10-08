"""Visible-assortment definition contract (data-plan Section 5, definition phase only).

This module fixes, in code, the definitions the future visible-assortment
engine must use. It computes no assortment from data: it holds an immutable
:class:`VisibleAssortmentDefinition`, typed statuses, an evidence gate over one
pipeline result and small pure helpers that pin the set formulas (see
``docs/decisions/governance/visible-assortment-contract-proposal-v1-2026-10-08.md``).
The calculation engine that implements these definitions is
:mod:`ql2_sixt_canada_analysis.visible_assortment`; notebooks, exports and any
approved unusual-drop rule are out of scope here.

Definitions
-----------
* **Population** - the pricing-eligible canonical offers of
  :func:`~ql2_sixt_canada_analysis.pricing_pipeline.run_pricing_pipeline`
  (``CanonicalOfferReport.offers``): the governed Calgary exclusion and every
  ineligible row never enter, an unassessable offer blocks, and Vancouver
  Downtown/Thurlow are one canonical location. Never raw ``cars`` rows,
  matched airport/downtown pairs, row order or generated files.
* **Location** - the exact canonical pair ``(canonical_city, canonical_location)``.
  Source streams stay a coverage concept; source labels are provenance only.
* **Capture** - ``scheduled_capture_period`` (the authority-backed scheduled UTC
  period), on the schedule grid of
  :func:`~ql2_sixt_canada_analysis.price_change_events.capture_timelines`, so an
  eligible capture with no offers is a visible empty set.
* **Consecutive captures** - the existing
  :class:`~ql2_sixt_canada_analysis.price_change_events.CaptureInterval`:
  schedule-adjacent eligible captures exactly one hour apart with the same
  contributing source streams. Every other adjacent pair is a typed break; in
  addition a change of rental search context between the endpoints is a break.
  The first eligible capture of a run seeds its set.
* **Product** - the approved product identity
  (:data:`~ql2_sixt_canada_analysis.canonical_offers.APPROVED_PRODUCT_COLUMNS`).
  The rental dates are the *search context* of a capture set (one context per
  location capture is required; several fail closed); currency and price basis
  are *price-comparison* identity only. Price, identifiers, offer position,
  capture timestamps and provenance never enter.
* **Returned-product count** - the number of distinct product identities in one
  location capture; duplicates, canonical deduplication, price variants and
  several units of one product never inflate it.
* **Additions / removals / retained** - ``C - P``, ``P - C`` and ``P & C`` over a
  valid interval only.
* **Retention** - ``|P & C| / |P|`` (directional); an empty previous set is
  ``zero_denominator`` with no value.
* **Jaccard** - ``|P & C| / |P | C|`` (symmetric); an empty union is
  ``zero_denominator`` with no value.
* **Drop signals** - ``net_change = |C| - |P|``, ``absolute_drop = max(|P| - |C|, 0)``,
  ``drop_rate = absolute_drop / |P|`` (``zero_denominator`` when ``|P| = 0``).
* **Unusual drop** - no approved policy exists; classification fails closed
  (:class:`AnomalyPolicyStatus`).
* **Coincidence** - temporal association within the same canonical location and
  the same previous/current scheduled periods; only ``increase`` and
  ``decrease`` price outcomes are price changes.

Authority: every definition carries a :class:`DefinitionStatus`. ``derived``
definitions follow from approved records and existing contracts; ``analytical``
definitions are repository formulas that need no business authority;
``proposed`` definitions need a new authority decision and are not treated as
approved. Importing this module performs no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from ql2_sixt_canada_analysis.canonical_offers import APPROVED_PRODUCT_COLUMNS
from ql2_sixt_canada_analysis.price_change_events import (
    CANDIDATE_COLUMNS,
    CAPTURE_STEP,
    EVENT_IDENTITY_COLUMNS,
    EVENT_INTERVAL_COLUMNS,
    EVENT_TIMESTAMP_COLUMN,
    EVENT_UNIT_COLUMNS,
    FORBIDDEN_TIMESTAMP_SOURCES,
    CaptureInterval,
    IntervalBreak,
    TerminalOutcome,
)
from ql2_sixt_canada_analysis.schemas import CONFIDENTIAL_TECHNICAL_COLUMNS

__all__ = [
    "ASSORTMENT_BREAK_REASONS",
    "ASSORTMENT_TIMELINE_COLUMNS",
    "DEFAULT_ASSORTMENT_DEFINITION",
    "DEFAULT_UNUSUAL_DROP_POLICY",
    "AnomalyPolicyStatus",
    "AnomalyPolicyUnavailableError",
    "AssessabilityStatus",
    "AssortmentBlocker",
    "AssortmentComparison",
    "AssortmentContractError",
    "AssortmentInterpretation",
    "DefinitionStatus",
    "DenominatorStatus",
    "UnusualDropPolicy",
    "VisibleAssortmentDefinition",
    "assortment_evidence_blockers",
    "classify_unusual_drop",
    "compare_assortment",
    "distinct_product_count",
]


class AssortmentContractError(ValueError):
    """A definition or helper input violates the visible-assortment contract."""


class AnomalyPolicyUnavailableError(AssortmentContractError):
    """Unusual-drop classification was requested without an approved policy."""


class DefinitionStatus(StrEnum):
    """How authoritative one definition is (never upgraded without a recorded decision)."""

    DERIVED = "derived_from_approved_contracts"
    ANALYTICAL = "analytical_definition"
    PROPOSED = "proposed_requires_authority"


class DenominatorStatus(StrEnum):
    DEFINED = "defined"
    ZERO_DENOMINATOR = "zero_denominator"
    NOT_ASSESSABLE = "not_assessable"          # no valid interval: no ratio at all


class AnomalyPolicyStatus(StrEnum):
    UNAVAILABLE = "unavailable"                # no policy exists (current state)
    PROPOSED = "proposed"                      # documented candidate, not approved
    APPROVED = "approved"                      # requires recorded authority provenance


class AssessabilityStatus(StrEnum):
    """Status of one timeline row (one canonical location and scheduled capture)."""

    ASSESSED = "assessed"                      # valid interval to the preceding capture
    SEED_CAPTURE = "seed_capture"              # first eligible capture of a run: no comparison
    INTERVAL_BREAK = "interval_break"          # eligible capture after a typed break: no comparison
    CAPTURE_NOT_ELIGIBLE = "capture_not_eligible"   # governed exclusion or missing capture: no set
    BLOCKED = "blocked"                        # evidence gate failed


class AssortmentInterpretation(StrEnum):
    """Distinct claims about a drop; the data alone supports only the first two."""

    OBSERVED_DROP = "observed_drop"                              # absolute_drop > 0 over a valid interval
    STATISTICALLY_UNUSUAL_DROP = "statistically_unusual_drop"    # only under an approved policy
    SUSPECTED_COLLECTION_FAILURE = "suspected_collection_failure"  # needs collection-owner corroboration
    SUPPLIER_ASSORTMENT_WITHDRAWAL = "supplier_assortment_withdrawal"  # needs supplier corroboration


class AssortmentBlocker(StrEnum):
    """Why assortment analysis may not start on one pipeline result (categories only)."""

    PRICING_NOT_READY = "pricing_not_ready"
    CANONICAL_OFFERS_NOT_READY = "canonical_offers_not_ready"
    SCHEDULE_EVIDENCE_INVALID = "schedule_evidence_invalid"
    EVIDENCE_BINDING_MISMATCH = "evidence_binding_mismatch"
    LOCATION_AUTHORITY_UNAVAILABLE = "location_authority_unavailable"
    PRODUCT_IDENTITY_INCOMPLETE = "product_identity_incomplete"
    MULTIPLE_RENTAL_CONTEXTS = "multiple_rental_contexts"
    # Added with the calculation engine (implementation correspondence; no new business rule):
    UNKNOWN_CANONICAL_LOCATION = "unknown_canonical_location"
    CAPTURE_EVIDENCE_INCONSISTENT = "capture_evidence_inconsistent"
    PRICE_CHANGE_EVIDENCE_INVALID = "price_change_evidence_invalid"
    RECONCILIATION_FAILED = "reconciliation_failed"


#: Interval break reasons of the assortment contract: every event-contract break plus a rental-context change.
ASSORTMENT_BREAK_REASONS: tuple[str, ...] = (*(b.value for b in IntervalBreak), "rental_context_changed")

_LOCATION = ("canonical_city", "canonical_location")
_CONTEXT = ("pickup_date", "return_date")
#: Every column a canonical offer carries (the only source columns an assortment definition may select).
_OFFER_SOURCE = frozenset({*EVENT_IDENTITY_COLUMNS, EVENT_TIMESTAMP_COLUMN, "price_cents", "source_location_labels",
                           "observation_count", "provenance", "price_variation"})
_PRICE = frozenset({"price_cents", "price_num", "price_per_day", "previous_price_cents", "current_price_cents",
                    "change_cents", "previous_price", "current_price", "change_dollars", "change_percent"})
_PROVENANCE = frozenset({"source_location_labels", "previous_source_labels", "current_source_labels",
                         "observation_count", "provenance", "price_variation", "city", "location"})

#: The proposed fixed, ordered, aggregate-only timeline schema (one row per canonical location and scheduled capture).
ASSORTMENT_TIMELINE_COLUMNS: tuple[str, ...] = (
    "canonical_city", "canonical_location", EVENT_TIMESTAMP_COLUMN, "capture_state", "contributing_stream_count",
    "returned_product_count", "has_previous_interval", EVENT_INTERVAL_COLUMNS[0],
    "previous_product_count", "retained_count", "addition_count", "removal_count", "retention",
    "retention_denominator_status", "jaccard_similarity", "jaccard_denominator_status", "net_change",
    "absolute_drop", "drop_rate", "drop_rate_denominator_status", "interval_break_reason", "price_increase_count",
    "price_decrease_count", "assortment_change", "price_change", "assortment_price_coincidence",
    "falling_assortment_with_price_increase", "anomaly_policy_status", "unusual_drop", "assessability_status")


def _names(value: object, group: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(isinstance(v, str) and v and v == v.strip() for v in value):
        raise AssortmentContractError(f"{group} must be a tuple of exact column names")
    if not value and not allow_empty:
        raise AssortmentContractError(f"{group} must not be empty")
    if len(set(value)) != len(value):
        raise AssortmentContractError(f"{group} must not repeat a column")
    return value


@dataclass(frozen=True)
class VisibleAssortmentDefinition:
    """The immutable visible-assortment definitions (see the module docstring for semantics).

    Raises:
        AssortmentContractError: A group is malformed, groups overlap, a selected column is not a canonical-offer
            column, product identity holds a prohibited column, or the timeline exposes a prohibited field.
    """

    population_source: str
    location_columns: tuple[str, ...]
    capture_column: str
    product_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    unit_columns: tuple[str, ...]
    prohibited_columns: frozenset[str]
    interval_step_seconds: int
    break_reasons: tuple[str, ...]
    price_change_outcomes: tuple[str, ...]
    timeline_columns: tuple[str, ...]
    statuses: Mapping[str, DefinitionStatus] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.population_source != "pricing_eligible_canonical_offers":
            raise AssortmentContractError("the population is the pricing-eligible canonical offers")
        groups = {"location_columns": _names(self.location_columns, "location_columns"),
                  "capture_column": _names((self.capture_column,), "capture_column"),
                  "product_columns": _names(self.product_columns, "product_columns"),
                  "context_columns": _names(self.context_columns, "context_columns", allow_empty=True),
                  "unit_columns": _names(self.unit_columns, "unit_columns", allow_empty=True)}
        seen: dict[str, str] = {}
        for group, columns in groups.items():
            for column in columns:
                if column in seen:
                    raise AssortmentContractError(f"{column} belongs to both {seen[column]} and {group}")
                seen[column] = group
                if column not in _OFFER_SOURCE:
                    raise AssortmentContractError(f"{column} is not a canonical-offer column")
        if not isinstance(self.prohibited_columns, frozenset) or not self.prohibited_columns:
            raise AssortmentContractError("prohibited columns must be a non-empty frozenset")
        required = {*FORBIDDEN_TIMESTAMP_SOURCES, *CONFIDENTIAL_TECHNICAL_COLUMNS, *_PRICE, *_PROVENANCE}
        if not required <= self.prohibited_columns:
            raise AssortmentContractError("prohibited columns must cover timestamps, identifiers, prices, provenance")
        identity = set(self.location_columns) | set(self.product_columns) | set(self.context_columns)
        if identity & self.prohibited_columns or self.capture_column in self.prohibited_columns:
            raise AssortmentContractError("a prohibited column cannot identify a location, capture or product")
        if self.capture_column in self.product_columns:
            raise AssortmentContractError("a capture timestamp cannot be part of product identity")
        if self.interval_step_seconds != int(CAPTURE_STEP.total_seconds()):
            raise AssortmentContractError("consecutive captures are exactly the event contract's one-hour step")
        if not set(b.value for b in IntervalBreak) <= set(_names(self.break_reasons, "break_reasons")):
            raise AssortmentContractError("every event-contract interval break remains a break")
        if set(_names(self.price_change_outcomes, "price_change_outcomes")) != {
                TerminalOutcome.INCREASE.value, TerminalOutcome.DECREASE.value}:
            raise AssortmentContractError("only increase and decrease are price changes")
        timeline = _names(self.timeline_columns, "timeline_columns")
        event_fields = set(CANDIDATE_COLUMNS) - {*self.location_columns, *EVENT_INTERVAL_COLUMNS}
        if set(timeline) & (self.prohibited_columns | set(self.product_columns) | set(self.context_columns)
                            | set(self.unit_columns) | event_fields):
            raise AssortmentContractError("the timeline is aggregate-only: no product, price or provenance fields")
        if tuple(timeline[:3]) != (*self.location_columns, self.capture_column):
            raise AssortmentContractError("the timeline grain is canonical location and scheduled capture")
        statuses = dict(self.statuses)
        if not statuses or not all(isinstance(s, DefinitionStatus) for s in statuses.values()):
            raise AssortmentContractError("every definition carries a DefinitionStatus")
        object.__setattr__(self, "statuses", MappingProxyType(statuses))

    @property
    def set_key_columns(self) -> tuple[str, ...]:
        """The key of one product set: canonical location, scheduled capture and rental search context."""
        return (*self.location_columns, self.capture_column, *self.context_columns)

    @property
    def proposed_definitions(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, s in self.statuses.items() if s is DefinitionStatus.PROPOSED))


#: The default contract, built from the established repository constants.
DEFAULT_ASSORTMENT_DEFINITION = VisibleAssortmentDefinition(
    population_source="pricing_eligible_canonical_offers",
    location_columns=_LOCATION,
    capture_column=EVENT_TIMESTAMP_COLUMN,
    product_columns=tuple(APPROVED_PRODUCT_COLUMNS),
    context_columns=_CONTEXT,
    unit_columns=tuple(EVENT_UNIT_COLUMNS),
    prohibited_columns=frozenset({*FORBIDDEN_TIMESTAMP_SOURCES, *CONFIDENTIAL_TECHNICAL_COLUMNS, *_PRICE,
                                  *_PROVENANCE, "parent_key"}),
    interval_step_seconds=int(CAPTURE_STEP.total_seconds()),
    break_reasons=ASSORTMENT_BREAK_REASONS,
    price_change_outcomes=(TerminalOutcome.INCREASE.value, TerminalOutcome.DECREASE.value),
    timeline_columns=ASSORTMENT_TIMELINE_COLUMNS,
    statuses={
        "population": DefinitionStatus.DERIVED,
        "location": DefinitionStatus.DERIVED,
        "capture": DefinitionStatus.DERIVED,
        "consecutive_interval": DefinitionStatus.DERIVED,
        "product_identity": DefinitionStatus.DERIVED,
        "rental_context_single_per_capture": DefinitionStatus.ANALYTICAL,
        "multiple_rental_context_stratification": DefinitionStatus.PROPOSED,
        "returned_product_count": DefinitionStatus.ANALYTICAL,
        "additions_removals": DefinitionStatus.ANALYTICAL,
        "retention": DefinitionStatus.ANALYTICAL,
        "jaccard": DefinitionStatus.ANALYTICAL,
        "drop_signals": DefinitionStatus.ANALYTICAL,
        "unusual_drop_policy": DefinitionStatus.PROPOSED,
        "price_coincidence": DefinitionStatus.ANALYTICAL,
        "timeline_schema": DefinitionStatus.ANALYTICAL,
        "timeline_persistence": DefinitionStatus.PROPOSED,
    },
)


# ------------------------------------------------------------------ pure helpers


def distinct_product_count(identities: Iterable[tuple]) -> int:
    """The returned-product count: distinct complete product identities (several offers of one product count once).

    Raises:
        AssortmentContractError: An identity has a missing or malformed component (never a sentinel match).
    """
    width = len(DEFAULT_ASSORTMENT_DEFINITION.product_columns)
    seen = set()
    for identity in identities:
        if not isinstance(identity, tuple) or len(identity) != width or any(
                v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and (
                    not v or v != v.strip())) for v in identity):
            raise AssortmentContractError("a product identity must be complete and exact")
        seen.add(identity)
    return len(seen)


@dataclass(frozen=True, slots=True)
class AssortmentComparison:
    """The set comparison of one valid interval (or why there is none). Counts only; no identities."""

    assessable: bool
    break_reason: str | None
    previous_count: int | None
    current_count: int
    retained_count: int | None = None
    addition_count: int | None = None
    removal_count: int | None = None
    retention: float | None = None
    retention_status: DenominatorStatus = DenominatorStatus.NOT_ASSESSABLE
    jaccard: float | None = None
    jaccard_status: DenominatorStatus = DenominatorStatus.NOT_ASSESSABLE
    net_change: int | None = None
    absolute_drop: int | None = None
    drop_rate: float | None = None
    drop_rate_status: DenominatorStatus = DenominatorStatus.NOT_ASSESSABLE

    def __post_init__(self) -> None:
        if not self.assessable:
            if any(v is not None for v in (self.retained_count, self.addition_count, self.removal_count,
                                           self.retention, self.jaccard, self.net_change, self.absolute_drop,
                                           self.drop_rate)):
                raise AssortmentContractError("differences across a break are never additions or removals")
            return
        p, c = self.previous_count, self.current_count
        if self.retained_count + self.removal_count != p or self.retained_count + self.addition_count != c:
            raise AssortmentContractError("retained, added and removed products partition the union")
        for value in (self.retention, self.jaccard, self.drop_rate):
            if value is not None and not 0.0 <= value <= 1.0:
                raise AssortmentContractError("ratios lie between zero and one")


def compare_assortment(previous: frozenset | None, current: frozenset, *, interval: CaptureInterval | None,
                       break_reason: str | None = None) -> AssortmentComparison:
    """Compare two product sets only across a valid :class:`CaptureInterval` (pins the Section 5 formulas).

    ``previous`` is ``None`` for a seed capture. Without an interval the result
    carries the typed ``break_reason`` (or none for a seed) and no differences.
    """
    if not isinstance(current, frozenset) or (previous is not None and not isinstance(previous, frozenset)):
        raise AssortmentContractError("product sets are frozensets of identities")
    if interval is None or previous is None:
        if break_reason is not None and break_reason not in ASSORTMENT_BREAK_REASONS:
            raise AssortmentContractError("an unknown interval break reason")
        return AssortmentComparison(False, break_reason, None if previous is None else len(previous), len(current))
    if not isinstance(interval, CaptureInterval) or break_reason is not None:
        raise AssortmentContractError("a valid interval has no break reason")
    retained, union = previous & current, previous | current
    p, c, r = len(previous), len(current), len(retained)
    D = DenominatorStatus
    drop = max(p - c, 0)
    return AssortmentComparison(
        True, None, p, c, retained_count=r, addition_count=len(current - previous),
        removal_count=len(previous - current),
        retention=(r / p) if p else None, retention_status=D.DEFINED if p else D.ZERO_DENOMINATOR,
        jaccard=(r / len(union)) if union else None, jaccard_status=D.DEFINED if union else D.ZERO_DENOMINATOR,
        net_change=c - p, absolute_drop=drop, drop_rate=(drop / p) if p else None,
        drop_rate_status=D.DEFINED if p else D.ZERO_DENOMINATOR)


@dataclass(frozen=True)
class UnusualDropPolicy:
    """An unusual-drop policy. ``approved`` requires recorded authority provenance (none exists today).

    ``rule`` is the injected, executable decision of an approved policy: it
    receives the :class:`AssortmentComparison` of one valid interval and returns
    ``True`` (unusual) or ``False``. The repository defines no rule, method or
    threshold of its own; a rule is only ever evaluated under ``approved``.
    """

    status: AnomalyPolicyStatus
    record_id: str | None = None
    reference: str | None = None
    description: str = ""
    rule: Callable[[AssortmentComparison], bool] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, AnomalyPolicyStatus):
            raise AssortmentContractError("status must be an AnomalyPolicyStatus")
        if self.status is AnomalyPolicyStatus.APPROVED and not (self.record_id and self.reference):
            raise AssortmentContractError("an approved policy names its authority record and reference")
        if self.rule is not None and not callable(self.rule):
            raise AssortmentContractError("a policy rule must be callable")

    @property
    def executable(self) -> bool:
        """Whether this policy may classify: approved with recorded authority and an injected rule."""
        return self.status is AnomalyPolicyStatus.APPROVED and self.rule is not None


#: No approved unusual-drop policy exists: classification fails closed.
DEFAULT_UNUSUAL_DROP_POLICY = UnusualDropPolicy(AnomalyPolicyStatus.UNAVAILABLE)


def classify_unusual_drop(comparison: AssortmentComparison,
                          policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY) -> bool:
    """Classify one valid interval under an approved, executable policy; fail closed otherwise.

    No approved policy exists in this repository, so with the default policy
    (and with any proposed or rule-less policy) this always raises: an observed
    drop is never called unusual, a collection failure or a supplier
    withdrawal here.

    Raises:
        AnomalyPolicyUnavailableError: The policy is not approved or provides no executable rule.
        AssortmentContractError: The comparison is not a valid interval, or the rule returned a non-boolean.
    """
    if not isinstance(comparison, AssortmentComparison):
        raise TypeError("comparison must be an AssortmentComparison")
    if not isinstance(policy, UnusualDropPolicy):
        raise TypeError("policy must be an UnusualDropPolicy")
    if policy.status is not AnomalyPolicyStatus.APPROVED:
        raise AnomalyPolicyUnavailableError("unusual-drop classification needs an approved policy")
    if policy.rule is None:
        raise AnomalyPolicyUnavailableError("the approved policy provides no executable rule")
    if not comparison.assessable:
        raise AssortmentContractError("only a valid interval can be classified")
    verdict = policy.rule(comparison)
    if not isinstance(verdict, bool):
        raise AssortmentContractError("an unusual-drop rule returns True or False")
    return verdict


# ------------------------------------------------------------------ evidence gate


def assortment_evidence_blockers(run: object,
                                 definition: VisibleAssortmentDefinition = DEFAULT_ASSORTMENT_DEFINITION,
                                 ) -> tuple[AssortmentBlocker, ...]:
    """Every reason one pipeline result may not enter assortment analysis (empty tuple: the evidence is sound).

    Reuses the existing gates: pricing readiness bound to the same reports,
    ready canonical offers without unassessable rows, a valid schedule
    assessment with capture-period and governed-exclusion evidence, one frame
    binding, the approved canonical-location configuration, complete product
    identities and one rental search context per location capture.
    """
    from ql2_sixt_canada_analysis.price_change_events import (
        UnknownCanonicalLocationError,
        approved_canonical_locations,
    )
    from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult
    from ql2_sixt_canada_analysis.pricing_population import frame_binding

    if not isinstance(run, PricingPipelineResult):
        raise TypeError("run must be a PricingPipelineResult")
    B, found = AssortmentBlocker, []
    pricing, offers, scheduled, population = run.pricing, run.canonical_offers, run.scheduled, run.population
    if pricing is None or not getattr(pricing, "ready", False) or getattr(pricing, "canonical_offers", None) is not offers \
            or getattr(pricing, "scheduled_coverage", None) is not scheduled:
        found.append(B.PRICING_NOT_READY)
    if offers is None or not offers.ready or offers.unassessable_rows or offers.offers is None:
        found.append(B.CANONICAL_OFFERS_NOT_READY)
    if (scheduled is None or not scheduled.is_valid or scheduled.capture_periods is None
            or scheduled.capture_exclusions is None):
        found.append(B.SCHEDULE_EVIDENCE_INVALID)
    binding = frame_binding(run.jobs, run.cars)
    if population is None or population.binding != binding or offers is None or offers.binding != binding:
        found.append(B.EVIDENCE_BINDING_MISMATCH)
    try:
        if run.location_authority is None or offers is None:
            raise UnknownCanonicalLocationError("unavailable")
        approved_canonical_locations(run.location_authority, offers.policy)
    except UnknownCanonicalLocationError:
        found.append(B.LOCATION_AUTHORITY_UNAVAILABLE)
    frame = offers.offers if offers is not None else None
    if frame is not None:
        columns = [*definition.product_columns, *definition.location_columns]
        if any(c not in frame.columns for c in (*columns, *definition.context_columns, definition.capture_column)) \
                or frame[columns].isna().any().any():
            found.append(B.PRODUCT_IDENTITY_INCOMPLETE)
        elif len(frame) and frame.groupby([*definition.location_columns, definition.capture_column])[
                list(definition.context_columns)].nunique().gt(1).any().any():
            found.append(B.MULTIPLE_RENTAL_CONTEXTS)
    return tuple(dict.fromkeys(found))
