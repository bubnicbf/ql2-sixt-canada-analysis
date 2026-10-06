"""Sanitized pricing-readiness baseline: a thin reporting layer over the existing assessments.

The baseline records *why* the dataset is (or is not) pricing ready without
re-deciding anything. It has two strictly separate parts:

* **Active blockers** - copied from the typed results the existing
  assessments emitted (:class:`~ql2_sixt_canada_analysis.readiness.PricingReadinessReport`
  and the reports it keeps: completeness, expected streams, scheduled
  coverage, trusted join, location policy; plus the temporal and stability
  reports). Values are the enums' own values; nothing is reconstructed from
  booleans and no new readiness rule is applied.
* **Plan-level gaps** (:class:`PlanReadinessGap`) - prerequisites of the
  analysis plan that the central pricing gate does **not** model as
  ``PricingBlocker`` values (no airport/downtown role map, no rental-period
  date rules). They are derived from configuration only and are never
  presented as active blockers. (The exhaustive expected-stream universe is a
  central blocker - ``expected_stream_authority_unavailable`` /
  ``expected_stream_universe_not_exhaustive`` - not a plan gap.)

Stream populations are reported separately. The configured expected
population is the authority-backed source-stream contract
(:class:`~ql2_sixt_canada_analysis.expected_stream_contract.ExpectedStreamContract`,
approved keys in their exact approved spelling). The observed population is
the distinct complete (city, location) keys in the cleaned detail rows
(:func:`~ql2_sixt_canada_analysis.coverage.location_pair_evidence`, exact and
unnormalised); it is reported as **counts only** - how many observed keys
equal an approved key exactly, how many are unexpected and how many of those
are spelling variants - and only observed keys that equal an approved key are
named. Observed source values outside the approved contract are never printed.
Observations never establish authority and never extend the universe.

Per-stream health is reported for every approved expected stream (its typed
stream status, continuity and time coverage) and, separately and anonymously
(numbered, unlabelled), for every observed stream: whether it is in the
contract or a spelling variant, and its job-based continuity counted exactly
(in-scope jobs of its own city value, jobs carrying it, jobs lacking it,
jobs without linked details). The observed-stream health is a diagnostic; it
never substitutes for the expected-stream assessment.

The investigated stream's continuity is reported from its stream report plus
one aggregate diagnostic: distinct complete capture events (detail
relationship keys) in the stream's scope that carry no row of the stream,
matched exactly through the contract. It is an *apparent* source-continuity
gap; without an authoritative schedule no capture is called "scheduled".

Sanitization is fail closed: :meth:`PricingReadinessBaseline.to_dict` and
:func:`render_baseline_markdown` accept only ints, bools, snake-case codes
(:data:`_CODE`) and digit-free location labels (:data:`_LABEL`); DataFrames,
floats, timestamps, identifiers, prices or any other object raise
:class:`UnsafeBaselineValueError`. No row-level value is ever read except the
location keys of the observed population.

Run against the real files (prints Markdown only)::

    python -m ql2_sixt_canada_analysis.pricing_baseline --commit <sha> --date <YYYY-MM-DD>
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from ql2_sixt_canada_analysis.authority_decisions import (
    AuthorityDecisionRecord,
    AuthorityReference,
    DecisionId,
)
from ql2_sixt_canada_analysis.coverage import location_pair_evidence, spelling_variant_keys
from ql2_sixt_canada_analysis.expected_stream_contract import ExpectedStreamContract
from ql2_sixt_canada_analysis.readiness import PricingReadinessReport
from ql2_sixt_canada_analysis.schemas import (
    PROJECT_DEFAULT,
    project_default,
    TemporalKind,
    INVESTIGATED_LOCATION_STREAM,
    JOB_DETAIL_RELATIONSHIP,
    TEMPORAL_RECONCILIATION,
    JobDetailRelationshipDefinition,
    LocationCoverageDefinition,
    LocationCoverageMode,
    TemporalReconciliationDefinition,
)
from ql2_sixt_canada_analysis.stability import VehicleStabilityReport
from ql2_sixt_canada_analysis.streams import _match
from ql2_sixt_canada_analysis.temporal import TemporalReconciliationReport

__all__ = [
    "ApprovedDateAgreement",
    "baseline_authority_inputs",
    "rental_date_fields",
    "LocationRole",
    "BaselineInputError",
    "ContinuityFinding",
    "PlanReadinessGap",
    "ObservedPopulation",
    "ObservedStreamHealth",
    "PricingReadinessBaseline",
    "StreamHealth",
    "StreamPopulation",
    "UnsafeBaselineValueError",
    "build_pricing_baseline",
    "render_baseline_markdown",
    "run_pricing_baseline",
]

#: Snake-case codes (enum values, rule and gap names); no spaces, no leading digit.
_CODE = re.compile(r"^[a-z][a-z0-9_.:]{0,79}$")
#: Location labels (configured or observed keys): letters, spaces, . ' - only; no digits.
_LABEL = re.compile(r"^[A-Za-z][A-Za-z .'\-]{0,62}$")
#: Rental-period columns of the detail contract the plan needs date rules for.
RENTAL_PERIOD_COLUMNS = ("pickup_date", "return_date")
#: Detail-side copies of the parent job's rental-period columns (distinct from the detail's own).
JOB_RENTAL_PERIOD_COLUMNS = ("job_pickup_date", "job_return_date")


@dataclass(frozen=True, slots=True)
class ApprovedDateAgreement:
    """One authority-approved parent-to-detail date agreement: ``target`` must equal ``source``.

    Exact references only - the authority decision record defines which detail
    field (``job_*`` copy or the detail's own field) agrees with which parent
    field; code never infers it from names.
    """

    source: tuple[object, str]
    target: tuple[object, str]


def rental_date_fields(relationship: JobDetailRelationshipDefinition) -> tuple[tuple[object, str], ...]:
    """The six rental-date fields whose semantics must be distinguished (parent, detail copy, detail own)."""
    parent, detail = relationship.parent, relationship.detail
    return (*((parent, c) for c in RENTAL_PERIOD_COLUMNS), *((detail, c) for c in JOB_RENTAL_PERIOD_COLUMNS),
            *((detail, c) for c in RENTAL_PERIOD_COLUMNS))


def baseline_authority_inputs(
    record: AuthorityDecisionRecord, relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> dict[str, object]:
    """Keyword inputs for :func:`build_pricing_baseline` taken from APPROVED decisions only.

    A PROPOSED or REJECTED decision contributes nothing (``None``), so a record
    without approvals - such as ``v1`` - can never close a plan gap.
    """
    if not isinstance(record, AuthorityDecisionRecord):
        raise BaselineInputError("a validated AuthorityDecisionRecord is required")
    inputs: dict[str, object] = dict(location_role_map=None, location_role_authority=None,
                                     rental_period_rule_authority=None, approved_rental_date_agreements=None)
    roles = record.approved_resolution(DecisionId.LOCATION_ROLE_ASSIGNMENTS)
    if roles is not None:
        inputs["location_role_map"] = {tuple(a["stream"]): LocationRole(a["role"].lower()) for a in roles["assignments"]}
        inputs["location_role_authority"] = record.approved_authority(DecisionId.LOCATION_ROLE_ASSIGNMENTS)
    agreements = record.approved_resolution(DecisionId.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS)
    validity = record.approved_resolution(DecisionId.RENTAL_DATE_VALIDITY)
    if agreements is not None and validity is not None:
        datasets = {str(relationship.parent): relationship.parent, str(relationship.detail): relationship.detail}

        def ref(value: str) -> tuple[object, str]:
            dataset, column = value.split(".", 1)
            return (datasets[dataset], column)

        inputs["approved_rental_date_agreements"] = tuple(
            ApprovedDateAgreement(source=ref(a["source"]), target=ref(a["target"])) for a in agreements["agreements"])
        inputs["rental_period_rule_authority"] = record.approved_authority(
            DecisionId.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS)
    return inputs


class BaselineInputError(ValueError):
    """A required assessment input is missing, empty or malformed (fail closed)."""


class UnsafeBaselineValueError(ValueError):
    """A value outside the sanitized vocabulary would be serialized (message holds no value)."""


class LocationRole(StrEnum):
    """The only roles a location-role map may assign (an allowlist)."""

    AIRPORT = "airport"
    DOWNTOWN = "downtown"
    OTHER = "other"


class PlanReadinessGap(StrEnum):
    """Plan prerequisites NOT modeled as ``PricingBlocker`` values (never active blockers)."""

    LOCATION_ROLE_MAP_UNAVAILABLE = "airport_downtown_role_map_unavailable"
    RENTAL_PERIOD_DATE_RULES_UNAVAILABLE = "rental_period_date_rules_unavailable"


_GAP_TEXT = {
    PlanReadinessGap.LOCATION_ROLE_MAP_UNAVAILABLE:
        "No authoritative airport/downtown role map is configured; central readiness does not check roles.",
    PlanReadinessGap.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE:
        "Pickup/return dates have no temporal-contract rules; central readiness does not check them.",
}


@dataclass(frozen=True, slots=True)
class StreamPopulation:
    """The configured expected population: approved keys in contract order."""

    population: str            # "configured_expected"
    authority: str             # "authoritative_exhaustive" | "authoritative_minimum_required" | "authority_unavailable"
    coverage_mode: str         # "exhaustive" | "minimum_required" | "unconfigured"
    keys: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        return len(self.keys)


@dataclass(frozen=True, slots=True)
class ObservedPopulation:
    """The observed population as counts; only observed keys equal to an approved key are named."""

    population: str            # "observed"
    authority: str             # "observed_not_authoritative"
    count: int
    exact_expected_count: int
    unexpected_count: int
    spelling_variant_count: int
    expected_missing_count: int
    keys: tuple[tuple[str, ...], ...]   # observed keys that exactly equal approved keys


@dataclass(frozen=True, slots=True)
class StreamHealth:
    """Typed health of one approved expected stream (statuses only)."""

    stream: tuple[str, ...]
    stream_status: str
    continuity: str
    time_coverage: str


@dataclass(frozen=True, slots=True)
class ObservedStreamHealth:
    """Anonymous job-based continuity of one observed stream (numbered, never labelled)."""

    ordinal: int
    in_contract: bool
    spelling_variant: bool
    continuity: str            # "complete" | "partial" | "unassessable"
    in_scope_jobs: int
    jobs_with_stream: int
    jobs_lacking_stream: int
    jobs_without_linked_details: int


@dataclass(frozen=True, slots=True)
class ContinuityFinding:
    """Aggregate continuity of the investigated stream (counts and statuses only)."""

    stream: tuple[str, ...]
    stream_status: str
    continuity: str
    time_coverage: str
    in_scope_jobs: int
    jobs_without_linked_details: int
    in_scope_capture_events: int
    capture_events_lacking_stream: int
    schedule_available: bool


@dataclass(frozen=True, slots=True)
class PricingReadinessBaseline:
    """Sanitized baseline: active typed blockers, plan gaps, populations, continuity."""

    pricing_ready: bool
    pricing_blockers: tuple[str, ...]
    subordinate_blockers: tuple[tuple[str, tuple[str, ...]], ...]
    statuses: tuple[tuple[str, str], ...]
    plan_gaps: tuple[PlanReadinessGap, ...]
    expected_population: StreamPopulation
    observed_population: ObservedPopulation
    continuity: ContinuityFinding | None
    expected_stream_health: tuple[StreamHealth, ...] = ()
    observed_stream_health: tuple[ObservedStreamHealth, ...] = ()
    authority_record_version: int | None = None

    def to_dict(self) -> dict:
        """Plain, sanitized structure; raises :class:`UnsafeBaselineValueError` on anything unsafe."""
        return _sanitize(self, "baseline")


# ------------------------------------------------------------------- building


def build_pricing_baseline(
    *,
    pricing: PricingReadinessReport,
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    temporal: TemporalReconciliationReport | None,
    vehicle_stability: VehicleStabilityReport | None,
    coverage: LocationCoverageDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
    temporal_contract: TemporalReconciliationDefinition = TEMPORAL_RECONCILIATION,
    investigated_stream: tuple[str, ...] = INVESTIGATED_LOCATION_STREAM,
    location_role_map: Mapping[tuple[str, ...], LocationRole] | None = None,
    location_role_authority: AuthorityReference | None = None,
    rental_period_rule_authority: AuthorityReference | None = None,
    approved_rental_date_agreements: tuple[ApprovedDateAgreement, ...] | None = None,
) -> PricingReadinessBaseline:
    """Assemble the sanitized baseline from existing assessment results (inputs are not modified).

    Plan gaps close only on sufficient, authority-backed evidence (fail closed):

    * the role-map gap closes only with ``location_role_authority`` and a map
      assigning a typed :class:`LocationRole` to **every** configured expected
      and observed stream (partial maps, untyped roles or malformed keys keep
      it open);
    * the rental-period gap closes only with ``rental_period_rule_authority``
      and ``approved_rental_date_agreements`` (from the authority decision
      record) that give **every** detail rental-date field
      (``job_pickup_date``, ``job_return_date``, ``pickup_date``,
      ``return_date``) exactly one approved parent source, when all six
      rental-date fields (:func:`rental_date_fields`) are required ``DATE``
      fields of the temporal contract and the contract's rental-date
      replication rules are **exactly** the approved source-to-target pairs
      (none missing, none unapproved). A rule from a parent field to an
      unrelated detail field never counts.

    ``coverage`` defaults to the contract the pricing report was assessed
    with (``pricing.expected_stream_contract``); a different contract is
    refused. Without an approved contract the expected population is empty
    and no stream is investigated.

    Raises:
        BaselineInputError: A required report is missing or of the wrong type,
            the frames are empty or lack the location columns, or ``coverage``
            differs from the pricing report's contract.
    """
    contract = pricing.expected_stream_contract if isinstance(pricing, PricingReadinessReport) else None
    if coverage is PROJECT_DEFAULT and contract is not None:
        coverage = contract.coverage
    coverage = project_default(coverage, "EXPECTED_LOCATION_COVERAGE")
    if not isinstance(pricing, PricingReadinessReport):
        raise BaselineInputError("a PricingReadinessReport is required")
    if not isinstance(cars, pd.DataFrame) or cars.empty:
        raise BaselineInputError("a non-empty cleaned detail frame is required")
    if temporal is not None and not isinstance(temporal, TemporalReconciliationReport):
        raise BaselineInputError("temporal must be a TemporalReconciliationReport or None")
    if vehicle_stability is not None and not isinstance(vehicle_stability, VehicleStabilityReport):
        raise BaselineInputError("vehicle_stability must be a VehicleStabilityReport or None")
    if not isinstance(jobs, pd.DataFrame) or jobs.empty:
        raise BaselineInputError("a non-empty cleaned jobs frame is required")
    if not isinstance(coverage, LocationCoverageDefinition):
        raise BaselineInputError("an expected-location contract is required")
    if contract is not None and coverage != contract.coverage:
        raise BaselineInputError("coverage differs from the contract the pricing report was assessed with")
    if not set(coverage.location_columns) <= set(cars.columns):
        raise BaselineInputError("the detail frame lacks the location columns")
    if coverage.is_configured and investigated_stream not in coverage.expected_locations:
        raise BaselineInputError("the investigated stream must be a configured expected stream")
    expected_keys = tuple(tuple(k) for k in (coverage.expected_locations or ()))
    observed_keys = _observed_keys(cars, coverage)

    subordinate, statuses = _subordinate(pricing, temporal, vehicle_stability)
    return PricingReadinessBaseline(
        pricing_ready=pricing.ready,
        pricing_blockers=tuple(dict.fromkeys(b.value for b in pricing.blocking_reasons)),
        subordinate_blockers=subordinate,
        statuses=statuses,
        plan_gaps=_plan_gaps(coverage, relationship, temporal_contract, cars, location_role_map,
                             location_role_authority, rental_period_rule_authority,
                             approved_rental_date_agreements),
        expected_population=StreamPopulation(
            population="configured_expected",
            authority=("authority_unavailable" if not coverage.is_configured
                       else "authoritative_exhaustive" if coverage.mode is LocationCoverageMode.EXHAUSTIVE
                       else "authoritative_minimum_required"),
            coverage_mode=coverage.mode.value if coverage.mode is not None else "unconfigured",
            keys=expected_keys),
        observed_population=_observed_population(observed_keys, expected_keys),
        continuity=(_continuity(pricing, cars, coverage, relationship, investigated_stream)
                    if coverage.is_configured else None),
        expected_stream_health=_expected_health(pricing, expected_keys),
        observed_stream_health=_observed_health(jobs, cars, coverage, relationship, observed_keys, expected_keys),
        authority_record_version=_record_version(contract),
    )


def _record_version(contract: ExpectedStreamContract | None) -> int | None:
    match = re.fullmatch(r"pricing-authorities-v(\d+)", contract.record_id or "") if contract is not None else None
    return int(match.group(1)) if match else None


def _observed_population(observed: tuple[tuple[str, ...], ...],
                         expected: tuple[tuple[str, ...], ...]) -> ObservedPopulation:
    exact = tuple(k for k in observed if k in set(expected))
    unexpected = [k for k in observed if k not in set(expected)]
    return ObservedPopulation(
        population="observed", authority="observed_not_authoritative", count=len(observed),
        exact_expected_count=len(exact), unexpected_count=len(unexpected),
        spelling_variant_count=len(spelling_variant_keys(unexpected, expected)) if expected else 0,
        expected_missing_count=sum(1 for k in expected if k not in set(observed)),
        keys=tuple(k for k in expected if k in set(exact)))


def _expected_health(pricing: PricingReadinessReport,
                     expected: tuple[tuple[str, ...], ...]) -> tuple[StreamHealth, ...]:
    completeness = pricing.completeness
    streams = completeness.expected_streams if completeness is not None else None
    reports = streams.reports if streams is not None else {}
    health = []
    for key in expected:
        report = reports.get(key)
        health.append(StreamHealth(
            stream=key, stream_status=report.status.value if report is not None else "report_unavailable",
            continuity=report.stream_continuity.value if report is not None else "report_unavailable",
            time_coverage=report.time_coverage.value if report is not None else "report_unavailable"))
    return tuple(health)


def _observed_health(jobs: pd.DataFrame, cars: pd.DataFrame, coverage: LocationCoverageDefinition,
                     relationship: JobDetailRelationshipDefinition, observed: tuple[tuple[str, ...], ...],
                     expected: tuple[tuple[str, ...], ...]) -> tuple[ObservedStreamHealth, ...]:
    """Exact job-based continuity of every observed stream (counts only; nothing normalised or repaired)."""
    scope, parent_scope = coverage.stream_scope_columns, coverage.parent_scope_columns
    parent_keys, detail_keys = list(relationship.parent_key_columns), list(relationship.detail_key_columns)
    if (not scope or not parent_scope or not set(scope) <= set(coverage.location_columns)
            or not set(parent_scope) | set(parent_keys) <= set(jobs.columns)
            or not set(detail_keys) <= set(cars.columns)):
        return ()
    variants = set(spelling_variant_keys(list(observed), expected)) if expected else set()
    linked = set(map(tuple, cars.loc[:, detail_keys].dropna().astype(object).itertuples(index=False)))
    health = []
    for ordinal, key in enumerate(observed, start=1):
        key_scope = tuple(key[coverage.location_columns.index(c)] for c in scope)
        in_scope = _match(jobs, parent_scope, (key_scope,)).to_numpy()
        scope_jobs = set(map(tuple, jobs.loc[in_scope, parent_keys].dropna().astype(object).itertuples(index=False)))
        rows = _match(cars, coverage.location_columns, (key,)).to_numpy()
        with_stream = set(map(tuple, cars.loc[rows, detail_keys].dropna().astype(object).itertuples(index=False)))
        with_stream &= scope_jobs
        without_details = scope_jobs - linked
        lacking = scope_jobs - with_stream - without_details
        continuity = ("unassessable" if without_details or not scope_jobs
                      else "partial" if lacking else "complete")
        health.append(ObservedStreamHealth(
            ordinal=ordinal, in_contract=key in set(expected), spelling_variant=key in variants,
            continuity=continuity, in_scope_jobs=len(scope_jobs), jobs_with_stream=len(with_stream),
            jobs_lacking_stream=len(lacking), jobs_without_linked_details=len(without_details)))
    return tuple(health)


def _codes(values) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    return tuple(dict.fromkeys(getattr(v, "value", v) for v in values))


def _subordinate(pricing: PricingReadinessReport, temporal: TemporalReconciliationReport | None,
                 stability: VehicleStabilityReport | None):  # type: ignore[no-untyped-def]
    """Blockers and statuses exactly as the existing typed reports emitted them (fixed source order)."""
    blockers: list[tuple[str, tuple[str, ...]]] = []
    statuses: list[tuple[str, str]] = []
    contract = pricing.expected_stream_contract
    if contract is None:
        blockers.append(("expected_stream_contract", ("expected_stream_authority_unavailable",)))
        statuses.append(("expected_stream_authority", "contract_missing"))
    else:
        statuses.append(("expected_stream_authority", contract.status.value))
        statuses.append(("expected_stream_universe_mode", contract.mode.value if contract.mode else "unconfigured"))
        blockers.append(("expected_stream_contract", _codes(contract.blocking_reasons)))
    completeness = pricing.completeness
    blockers.append(("completeness", _codes(completeness.blocking_reasons) if completeness is not None
                     else ("completeness_report_missing",)))
    streams = completeness.expected_streams if completeness is not None else None
    if streams is not None:
        blockers.append(("expected_streams", _codes(streams.blocking_reasons)))
    city = completeness.city_integrity if completeness is not None else None
    if city is not None:
        blockers.append(("city_integrity", _codes(city.blocking_reasons)))
    scheduled = pricing.scheduled_coverage
    if scheduled is None:
        blockers.append(("scheduled_coverage", ("scheduled_coverage_assessment_missing",)))
    else:
        statuses.append(("collection_schedule", scheduled.schedule_assessment.status.value))
        blockers.append(("scheduled_coverage", _codes(scheduled.blocking_reasons)))
    linkage = pricing.job_linkage
    if linkage is None:
        blockers.append(("job_linkage", ("job_key_normalization_missing",)))
    else:
        statuses.append(("job_linkage_policy", linkage.policy_status.value))
        statuses.append(("job_linkage_valid", "true" if linkage.is_valid else "false"))
        blockers.append(("job_linkage", _codes(linkage.blocking_reasons)))
    join = pricing.job_detail_join
    if join is None:
        blockers.append(("trusted_join", ("trusted_join_assessment_missing",)))
    else:
        statuses.append(("trusted_join_ready", "true" if join.join_ready else "false"))
        blockers.append(("trusted_join", _codes(join.blocking_reasons)))
    if temporal is None:
        blockers.append(("temporal", ("temporal_report_missing",)))
    else:
        statuses.append(("temporal_fields_trusted", "true" if temporal.is_valid else "false"))
        blockers.append(("temporal", _codes(temporal.violations)))
        blockers.append(("temporal_unavailable_rules", _codes(temporal.unavailable_rules)))
    policy = pricing.location_policy
    statuses.append(("location_policy_state", policy.state.value))
    blockers.append(("location_policy", _codes(policy.blocking_reasons)))
    statuses.append(("vehicle_stability", stability.status.value if stability is not None else "unavailable"))
    return tuple(blockers), tuple(statuses)


def _plan_gaps(coverage: LocationCoverageDefinition, relationship: JobDetailRelationshipDefinition,
               temporal_contract: TemporalReconciliationDefinition, cars: pd.DataFrame, role_map: object,
               role_authority: object, rental_authority: object,
               approved_agreements: object) -> tuple[PlanReadinessGap, ...]:
    G = PlanReadinessGap
    gaps = []
    required_keys = {tuple(k) for k in (coverage.expected_locations or ())} | set(_observed_keys(cars, coverage))
    if not _role_map_sufficient(role_map, role_authority, required_keys, len(coverage.location_columns)):
        gaps.append(G.LOCATION_ROLE_MAP_UNAVAILABLE)
    if not _rental_rules_sufficient(temporal_contract, relationship, rental_authority, approved_agreements):
        gaps.append(G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE)
    return tuple(g for g in G if g in gaps)


def _role_map_sufficient(role_map: object, authority: object, required: set, width: int) -> bool:
    """Authority-backed, typed roles for every expected and observed stream; anything less is a gap."""
    if not isinstance(authority, AuthorityReference) or not isinstance(role_map, Mapping) or not role_map:
        return False
    for key, role in role_map.items():
        if not (isinstance(key, tuple) and len(key) == width and all(isinstance(v, str) and v for v in key)):
            return False
        if not isinstance(role, LocationRole):
            return False
    return required <= set(role_map)


def _rental_rules_sufficient(contract: object, relationship: JobDetailRelationshipDefinition, authority: object,
                             approved: object) -> bool:
    """Authority, an exact approved agreement per detail rental field, validity and matching rules."""
    if not isinstance(contract, TemporalReconciliationDefinition) or not isinstance(authority, AuthorityReference):
        return False
    if (not isinstance(approved, tuple) or not approved
            or not all(isinstance(a, ApprovedDateAgreement) for a in approved)):
        return False
    fields = rental_date_fields(relationship)
    parents = set(fields[:len(RENTAL_PERIOD_COLUMNS)])
    details = set(fields[len(RENTAL_PERIOD_COLUMNS):])
    pairs = [(tuple(a.source), tuple(a.target)) for a in approved]
    if len(set(pairs)) != len(pairs) or not all(src in parents and tgt in details for src, tgt in pairs):
        return False
    targets = [tgt for _, tgt in pairs]
    if sorted(targets, key=repr) != sorted(details, key=repr):         # every detail field exactly once
        return False
    for ref in fields:                                                  # validity of all six fields
        field = next((f for f in contract.fields if f.ref == ref), None)
        if field is None or field.kind is not TemporalKind.DATE or field.required is not True:
            return False
    configured = {(tuple(r.source), tuple(r.replica)) for r in contract.replications
                  if tuple(r.source) in set(fields) or tuple(r.replica) in set(fields)}
    return configured == set(pairs)                                     # exactly the approved rules


def _observed_keys(cars: pd.DataFrame, coverage: LocationCoverageDefinition) -> tuple[tuple[str, ...], ...]:
    evidence = location_pair_evidence(cars, coverage)          # exact, source-preserving, complete keys only
    columns = list(coverage.location_columns)
    return tuple(sorted(dict.fromkeys(tuple(row) for row in evidence.loc[:, columns].itertuples(index=False))))


def _continuity(pricing: PricingReadinessReport, cars: pd.DataFrame, coverage: LocationCoverageDefinition,
                relationship: JobDetailRelationshipDefinition, target: tuple[str, ...]) -> ContinuityFinding | None:
    completeness = pricing.completeness
    streams = completeness.expected_streams if completeness is not None else None
    report = streams.reports.get(target) if streams is not None else None
    if report is None:
        return None
    accounting = report.event_accounting
    events, lacking = _apparent_gap(cars, coverage, relationship, target)
    scheduled = pricing.scheduled_coverage
    return ContinuityFinding(
        stream=tuple(target), stream_status=report.status.value, continuity=report.stream_continuity.value,
        time_coverage=report.time_coverage.value,
        in_scope_jobs=accounting.in_scope_jobs if accounting is not None else 0,
        jobs_without_linked_details=accounting.zero_detail_jobs if accounting is not None else 0,
        in_scope_capture_events=events, capture_events_lacking_stream=lacking,
        schedule_available=bool(scheduled is not None and scheduled.schedule_assessment.available))


def _apparent_gap(cars: pd.DataFrame, coverage: LocationCoverageDefinition,
                  relationship: JobDetailRelationshipDefinition, target: tuple[str, ...]) -> tuple[int, int]:
    """(in-scope capture events, events with no row of ``target``) from detail rows - counts only.

    Scope rows match the target's scope components exactly; target rows match
    its contract keys exactly (``coverage.match_keys``); events are complete
    detail relationship keys. Nothing is normalised, filled or repaired.
    """
    scope = coverage.stream_scope_columns
    if not scope or not all(c in coverage.location_columns for c in scope):
        return 0, 0
    target_scope = tuple(target[coverage.location_columns.index(c)] for c in scope)
    in_scope = _match(cars, scope, (target_scope,)).to_numpy()
    is_target = _match(cars, coverage.location_columns, coverage.match_keys(target)).to_numpy()
    keys = cars.loc[:, list(relationship.detail_key_columns)]
    complete = keys.notna().all(axis=1).to_numpy()
    events = set(map(tuple, keys.loc[in_scope & complete].astype(object).itertuples(index=False)))
    with_target = set(map(tuple, keys.loc[in_scope & complete & is_target].astype(object).itertuples(index=False)))
    return len(events), len(events - with_target)


# --------------------------------------------------------------- sanitization


def _sanitize(value: object, where: str):  # type: ignore[no-untyped-def]
    """Fail-closed serialization into plain Python values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < 0:
            raise UnsafeBaselineValueError(f"negative count at {where}")
        return value
    if isinstance(value, StrEnum):
        return _sanitize(value.value, where)
    if isinstance(value, str):
        if re.search(r"\.(keys|stream)(\[\])?$", where):
            if not _LABEL.match(value):
                raise UnsafeBaselineValueError(f"unsafe location label at {where}")
        elif not _CODE.match(value):
            raise UnsafeBaselineValueError(f"unsafe code at {where}")
        return value
    if isinstance(value, tuple):
        return [_sanitize(v, where if where.endswith("[]") else f"{where}[]") for v in value]
    if is_dataclass(value) and type(value) in _SERIALIZABLE:
        out = {f.name: _sanitize(getattr(value, f.name), f"{where}.{f.name}") for f in fields(value)}
        if isinstance(value, StreamPopulation):
            out["count"] = value.count
        return out
    if value is None:
        return None
    raise UnsafeBaselineValueError(f"unsupported value type at {where}")


_SERIALIZABLE = (PricingReadinessBaseline, StreamPopulation, ObservedPopulation, StreamHealth, ObservedStreamHealth,
                 ContinuityFinding)


def render_baseline_markdown(baseline: PricingReadinessBaseline, *, commit: str, date: str) -> str:
    """Deterministic Markdown summary (sanitized via :meth:`PricingReadinessBaseline.to_dict`)."""
    if not isinstance(baseline, PricingReadinessBaseline):
        raise BaselineInputError("a PricingReadinessBaseline is required")
    if not re.fullmatch(r"[0-9a-f]{7,40}|unrecorded", commit) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}|unrecorded", date):
        raise UnsafeBaselineValueError("commit must be a hex hash and date YYYY-MM-DD")
    d = baseline.to_dict()
    key = lambda k: " / ".join(k)  # noqa: E731
    lines = [f"- Baseline date: {date}", f"- Commit: `{commit}`",
             f"- Overall state: **{'PRICING READY' if d['pricing_ready'] else 'NOT PRICING READY'}**", "",
             "### A. Active blockers emitted by the current implementation", "",
             "Central `PricingBlocker` values: " + (", ".join(f"`{b}`" for b in d["pricing_blockers"]) or "none"), "",
             "| Source report | Typed blockers |", "| --- | --- |"]
    lines += [f"| {src} | {', '.join(f'`{b}`' for b in codes) or 'none'} |" for src, codes in d["subordinate_blockers"]]
    lines += ["", "| Status | Value |", "| --- | --- |"]
    lines += [f"| {name} | `{value}` |" for name, value in d["statuses"]]
    lines += ["", "### B. Plan-level prerequisites not modeled as `PricingBlocker` values", ""]
    lines += [f"- `{g}` - {_GAP_TEXT[PlanReadinessGap(g)]}" for g in d["plan_gaps"]] or ["- none"]
    e, o = d["expected_population"], d["observed_population"]
    version = d["authority_record_version"]
    lines += ["", f"### Configured expected streams: {e['count']}", "",
              f"Population: `{e['population']}`; authority: `{e['authority']}`; contract mode: "
              f"`{e['coverage_mode']}`; authority record version: "
              f"{version if version is not None else 'unavailable'}. Keys are the approved source spellings, "
              "matched exactly.", ""]
    lines += [f"- {key(k)}" for k in e["keys"]] or ["- none (no approved contract)"]
    lines += ["", f"### Observed streams (cleaned detail rows): {o['count']}", "",
              f"Population: `{o['population']}`; authority: `{o['authority']}`. Observed source values outside "
              "the approved contract are not printed.", "",
              f"- Exactly equal to an approved key: {o['exact_expected_count']}",
              f"- Unexpected (outside the approved contract): {o['unexpected_count']}",
              f"- Of which spelling variants of approved keys: {o['spelling_variant_count']}",
              f"- Approved keys with no exact observed match: {o['expected_missing_count']}"]
    lines += [f"- Observed approved key: {key(k)}" for k in o["keys"]]
    lines += ["", "### Expected stream health (per approved stream)", "",
              "| Stream | Status | Continuity | Time coverage |", "| --- | --- | --- | --- |"]
    lines += [f"| {key(h['stream'])} | `{h['stream_status']}` | `{h['continuity']}` | `{h['time_coverage']}` |"
              for h in d["expected_stream_health"]] or ["| none | | | |"]
    lines += ["", "### Observed stream health (anonymous, exact job-based continuity)", "",
              "| Observed stream | In contract | Spelling variant | Continuity | In-scope jobs | With stream | "
              "Lacking stream | Without linked details |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {h['ordinal']} | {h['in_contract']} | {h['spelling_variant']} | `{h['continuity']}` | "
              f"{h['in_scope_jobs']} | {h['jobs_with_stream']} | {h['jobs_lacking_stream']} | "
              f"{h['jobs_without_linked_details']} |" for h in d["observed_stream_health"]] or ["| none | | | | | | | |"]
    c = d["continuity"]
    lines += ["", "### Investigated approved stream (aggregate)", ""]
    if c is None:
        lines.append("- No single stream report is available for the investigated stream.")
    else:
        lines += [f"- Stream: {key(c['stream'])}; status `{c['stream_status']}`, continuity "
                  f"`{c['continuity']}`, time coverage `{c['time_coverage']}`.",
                  f"- Job-based continuity: {c['in_scope_jobs']} in-scope jobs, "
                  f"{c['jobs_without_linked_details']} without linked detail rows.",
                  f"- Apparent source-continuity gap: {c['capture_events_lacking_stream']} of "
                  f"{c['in_scope_capture_events']} in-scope capture events (detail rows) carry no row of the stream.",
                  "- " + ("An authoritative schedule is available." if c["schedule_available"] else
                          "No authoritative collection schedule exists, so these captures are not called "
                          "scheduled and the gap is pending a schedule.")]
    lines += ["", "Observations do not establish authority. No source-level values (identifiers, timestamps, "
              "dates, prices, vehicle names, offers or rows) are included."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- running


def run_pricing_baseline(raw_dir: str | Path | None = None) -> PricingReadinessBaseline:
    """Run the existing pipeline (the ingestion notebook's calls, in order) and build the baseline."""
    from ql2_sixt_canada_analysis import (  # local import: the package re-exports this module
        ANALYSIS_DATASET_DEFINITIONS, ANALYSIS_JOB_DETAIL_RELATIONSHIP, ANALYSIS_LOCATION_STREAM_COMPARISON,
        ANALYSIS_TEMPORAL_RECONCILIATION, COLLECTION_SCHEDULE, VANCOUVER_LOCATION_POLICY, VEHICLE_ATTRIBUTE_STABILITY,
        assess_job_linkage, load_job_linkage_policy,
        apply_location_policy, assess_city_integrity, assess_collection_schedule, assess_completeness,
        assess_dataset_location_coverage, assess_expected_location_streams, assess_job_detail_join_readiness,
        assess_job_detail_reconciliation, assess_location_policy, assess_one_to_many_join,
        assess_pricing_readiness, assess_raw_dataset_unique_keys, assess_scheduled_time_coverage,
        assess_temporal_reconciliation, assess_vehicle_attribute_stability, compare_location_streams,
        load_raw_datasets, remove_blank_rows_from_raw_datasets, validate_raw_dataset_identifier_dtypes,
    )
    from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
    from ql2_sixt_canada_analysis.paths import resolve_raw_data_dir
    from ql2_sixt_canada_analysis.relationships import RelationshipPreconditionError
    from ql2_sixt_canada_analysis.stability import VehicleStabilityPreconditionError

    def attempt(assess, *errors):  # type: ignore[no-untyped-def]
        try:
            return assess()
        except errors:
            return None

    # The single authority-backed source-stream contract (the same object EXPECTED_LOCATION_COVERAGE comes from).
    contract = current_expected_stream_contract()
    rel, cov = ANALYSIS_JOB_DETAIL_RELATIONSHIP, contract.coverage
    raw = load_raw_datasets(resolve_raw_data_dir(raw_dir))
    cleaned = remove_blank_rows_from_raw_datasets(raw).cleaned
    validate_raw_dataset_identifier_dtypes(cleaned)
    # Authority-backed linkage: every analytical step below uses the derived keys.
    linkage = assess_job_linkage(cleaned.jobs, cleaned.cars, load_job_linkage_policy())
    analysis = linkage.datasets(cleaned)
    jobs, cars = analysis.jobs, analysis.cars
    keys = assess_raw_dataset_unique_keys(analysis, ANALYSIS_DATASET_DEFINITIONS)
    configured = cov.is_configured          # without an approved contract, coverage-based steps fail closed
    coverage = assess_dataset_location_coverage(analysis, cov) if configured else None
    reconciliation = attempt(lambda: assess_job_detail_reconciliation(jobs, cars, rel), RelationshipPreconditionError)
    relationship = attempt(lambda: assess_one_to_many_join(jobs, cars, rel), RelationshipPreconditionError)
    city = assess_city_integrity(jobs, cars, relationship=rel, coverage=cov if configured else None)
    join = assess_job_detail_join_readiness(jobs, cars, rel, job_linkage=linkage.report)
    streams = (assess_expected_location_streams(jobs, cars, coverage=cov, relationship=rel, loaded=raw,
                                                schedule=COLLECTION_SCHEDULE) if configured else None)
    scheduled = (assess_scheduled_time_coverage(assess_collection_schedule(COLLECTION_SCHEDULE), streams)
                 if streams is not None else None)
    temporal = attempt(lambda: assess_temporal_reconciliation(jobs, cars, ANALYSIS_TEMPORAL_RECONCILIATION),
                       RelationshipPreconditionError)
    comparison = attempt(lambda: compare_location_streams(jobs, cars, ANALYSIS_LOCATION_STREAM_COMPARISON),
                         RelationshipPreconditionError)
    stability = attempt(lambda: assess_vehicle_attribute_stability(cars, VEHICLE_ATTRIBUTE_STABILITY),
                        VehicleStabilityPreconditionError)
    completeness = (assess_completeness(datasets=analysis, coverage=coverage, streams=streams,
                                        reconciliation=reconciliation, city_integrity=city, expected_coverage=cov)
                    if configured else None)
    policy = assess_location_policy(VANCOUVER_LOCATION_POLICY, comparison,
                                    apply_location_policy(cars, VANCOUVER_LOCATION_POLICY))
    pricing = assess_pricing_readiness(
        location_policy=policy, completeness=completeness, key_contracts_valid=bool(keys.all_valid),
        one_to_many_contract_valid=bool(relationship is not None and relationship.is_valid),
        temporal_fields_trusted=bool(temporal is not None and temporal.is_valid),
        vehicle_stability=stability, scheduled_coverage=scheduled, job_detail_join=join,
        job_linkage=linkage.report, expected_stream_contract=contract)
    return build_pricing_baseline(pricing=pricing, jobs=jobs, cars=cars, temporal=temporal, vehicle_stability=stability,
                                  relationship=rel, temporal_contract=ANALYSIS_TEMPORAL_RECONCILIATION)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the sanitized pricing-readiness baseline (Markdown).")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--commit", default="unrecorded")
    parser.add_argument("--date", default="unrecorded")
    args = parser.parse_args(argv)
    print(render_baseline_markdown(run_pricing_baseline(args.raw_dir), commit=args.commit, date=args.date), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
