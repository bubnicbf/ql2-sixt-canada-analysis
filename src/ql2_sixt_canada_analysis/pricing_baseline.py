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
  ``PricingBlocker`` values (now only: no rental-date assessment under an
  approved rental-date policy - the rules themselves are central blockers).
  They are never presented as active blockers. (The exhaustive expected-stream universe, the airport/downtown
  role map and the comparison pairs are central blockers - for example
  ``expected_stream_authority_unavailable``,
  ``branch_role_authority_unavailable``,
  ``comparison_pair_authority_unavailable`` - not plan gaps.)

The location authority (:mod:`~ql2_sixt_canada_analysis.location_authority`:
approved roles, comparison pairs and the Vancouver identity policy) is
reported as aggregate statuses: decision statuses, whether the role map is
exact, the number of streams per role, whether the pairs are valid, the
approved pairs on canonical keys (approved configuration), the Vancouver
policy state and its canonical key.

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
    DecisionId,
)
from ql2_sixt_canada_analysis.coverage import location_pair_evidence, spelling_variant_keys
from ql2_sixt_canada_analysis.expected_stream_contract import ExpectedStreamContract
from ql2_sixt_canada_analysis.readiness import PricingReadinessReport
from ql2_sixt_canada_analysis.rental_dates import (
    INFORMATIONAL_LONG_RENTAL_DAYS,
    AgreementStatus as _AS,
    PeriodStatus as _PS,
)
from ql2_sixt_canada_analysis.schemas import (
    PROJECT_DEFAULT,
    project_default,
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
    "CanonicalOfferSummary",
    "CollectionScheduleSummary",
    "PricingPopulationSummary",
    "ReportingDaySummary",
    "LocationAuthoritySummary",
    "StreamScheduleSummary",
    "TemporalBaselineSummary",
    "RentalDateSummary",
    "RentalFieldSummary",
    "RentalPeriodSummary",
    "RentalAgreementSummary",
    "TemporalFieldSummary",
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
    inputs: dict[str, object] = dict(rental_period_rule_authority=None, approved_rental_date_agreements=None)
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


class PlanReadinessGap(StrEnum):
    """Plan prerequisites NOT modeled as ``PricingBlocker`` values (never active blockers)."""

    RENTAL_PERIOD_DATE_RULES_UNAVAILABLE = "rental_period_date_rules_unavailable"


_GAP_TEXT = {
    PlanReadinessGap.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE:
        "No rental-date assessment ran under an approved rental-date policy (see the central "
        "`rental_date_rules_unavailable` / `rental_date_assessment_missing` blockers).",
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
    #: In-scope jobs whose whole capture is a governed parent-capture exclusion (neither present nor a gap).
    governed_excluded_jobs: int = 0


@dataclass(frozen=True, slots=True)
class LocationAuthoritySummary:
    """Aggregate location authority: statuses, counts and approved canonical keys only."""

    role_map_status: str            # approved | not_approved | record_unavailable | report_missing
    roles_exact: bool
    airport_streams: int
    downtown_streams: int
    other_streams: int
    comparison_pair_status: str
    comparison_pairs_valid: bool
    approved_pair_count: int
    vancouver_policy_state: str
    keys: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]   # approved pairs (airport, canonical downtown)
    stream: tuple[str, ...] | None                                # Vancouver canonical key (approved configuration)


@dataclass(frozen=True, slots=True)
class StreamScheduleSummary:
    """One stream's own scheduled coverage (counts only; exact approved key)."""

    stream: tuple[str, ...]
    expected_periods: int
    covered_periods: int
    unexcused_missing_periods: int
    excused_periods: int
    missing_by_failure: tuple[tuple[str, int], ...]
    excluded_periods: int = 0          # governed parent-capture exclusion (analytically null)
    required_periods: int = 0          # expected (nominal) minus excluded


@dataclass(frozen=True, slots=True)
class CollectionScheduleSummary:
    """Aggregate per-stream schedule and coverage: statuses and counts only (no instants, no identifiers)."""

    status: str                         # available | not_approved | record_unavailable | invalid | report_missing
    schedule_version: str | None
    sharing_model: str | None           # per_stream
    capture_field: str | None           # jobs.finished_at
    period_minutes: int
    exceptions_model: str | None        # no_exceptions | listed_exceptions
    excused_period_count: int
    schedule_count: int
    total_periods: int
    city_periods: tuple[tuple[str, int], ...]
    missing_city_periods: tuple[tuple[str, int], ...]
    jobs_assessed: int
    jobs_assigned: int
    job_failures: tuple[tuple[str, int], ...]
    detail_copy_mismatches: int
    unexcused_missing_periods: int
    streams: tuple[StreamScheduleSummary, ...]
    nominal_stream_periods: int = 0
    excluded_stream_periods: int = 0
    required_stream_periods: int = 0
    covered_stream_periods: int = 0
    excluded_parent_captures: int = 0
    unmatched_exclusions: int = 0


@dataclass(frozen=True, slots=True)
class TemporalFieldSummary:
    """Parse and resolution counts of one temporal field (no values)."""

    field: str                       # e.g. jobs.finished_at
    rows: int
    resolved: int
    missing: int
    invalid: int
    unknown_city: int
    context_unavailable: int
    ambiguous: int
    nonexistent: int


@dataclass(frozen=True, slots=True)
class TemporalBaselineSummary:
    """Authority statuses and aggregate counts of the temporal reconciliation (no timestamps)."""

    timezone_status: str             # approved | not_approved | record_unavailable | invalid | report_missing
    ordering_status: str
    tolerance_status: str
    tolerance_seconds: int | None
    reporting_day_status: str
    date_semantics_status: str
    authority_blockers: tuple[str, ...]
    pricing_date_fields: tuple[str, ...]
    fields: tuple[TemporalFieldSummary, ...]
    replication_passed: int
    replication_failed: int
    replication_unassessable: int
    ordering_rule_status: str        # configured | unavailable | report_missing
    ordering_passed: int
    ordering_failed: int
    ordering_unassessable: int
    unlinked_detail_rows: int
    city_mismatch_detail_rows: int
    unavailable_rules: tuple[str, ...]
    temporal_fields_trusted: bool


@dataclass(frozen=True, slots=True)
class ReportingDaySummary:
    """Reporting-day derivation and source-date agreement in aggregate (no dates)."""

    status: str                         # available | unavailable | report_missing
    source_field: str | None            # jobs.finished_at
    timezone_mode: str | None           # parent_city
    scrape_date_derivation: str | None  # reporting_day
    date_clean_status: str              # retired_from_pricing | not_approved | ...
    date_clean_rule_unavailable: bool   # date_derivation:cars.date_clean still reported unavailable
    parent_status_counts: tuple[tuple[str, int], ...]
    detail_status_counts: tuple[tuple[str, int], ...]
    parent_boundary_crossings: int      # agreeing parent dates whose UTC date differs (legitimate)
    detail_boundary_crossings: int
    date_clean_rows: int
    date_clean_valid: int
    date_clean_missing: int
    date_clean_invalid: int


@dataclass(frozen=True, slots=True)
class PricingPopulationSummary:
    """Nominal, governed-excluded and eligible parent captures and detail rows (counts only)."""

    status: str                         # available | report_missing
    nominal_parent_captures: int
    excluded_parent_captures: int
    ineligible_parent_captures: int
    eligible_parent_captures: int
    nominal_detail_rows: int
    excluded_detail_rows: int
    ineligible_detail_rows: int
    eligible_detail_rows: int
    detail_status_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class CanonicalOfferSummary:
    """Canonical offer combination in aggregate (approved keys and counts; no offers, prices or names)."""

    policy_status: str                  # approved | not_approved | record_unavailable | invalid | report_missing
    source_streams: tuple[tuple[str, ...], ...]
    stream: tuple[str, ...] | None      # canonical location
    source_rows: int
    out_of_scope_rows: int
    combined_observations: int
    unassessable_rows: int
    canonical_offers: int
    unique_offers: int
    deduplicated_offers: int
    duplicate_groups: int
    collapsed_observations: int
    variation_groups: int
    variation_offers: int
    unassessable_group_count: int
    unassessable_groups: tuple[tuple[str, int], ...]
    vancouver_source_counts: tuple[tuple[tuple[str, ...], int], ...]
    vancouver_canonical_offers: int
    cross_stream_offers: int
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RentalFieldSummary:
    field: str
    rows: int
    valid: int
    missing: int
    invalid: int


@dataclass(frozen=True, slots=True)
class RentalPeriodSummary:
    period: str
    rows: int
    valid: int
    return_before_pickup: int
    incomplete_or_invalid: int
    same_day: int
    long_informational: int        # valid and >= INFORMATIONAL_LONG_RENTAL_DAYS: descriptive only


@dataclass(frozen=True, slots=True)
class RentalAgreementSummary:
    target: str
    source: str
    rows: int
    match: int
    mismatch: int
    unassessable: int


@dataclass(frozen=True, slots=True)
class RentalDateSummary:
    """Rental-date authority, validity, agreement and eligibility in aggregate (no dates)."""

    policy_status: str             # approved | not_approved | record_unavailable | invalid | report_missing
    maximum_duration_mode: str | None
    parent_rows: int
    detail_rows: int
    fields: tuple[RentalFieldSummary, ...]
    periods: tuple[RentalPeriodSummary, ...]
    agreements: tuple[RentalAgreementSummary, ...]
    linkage_trusted: bool
    validity_holds: bool
    agreement_holds: bool
    eligible_parent_rows: int
    eligible_detail_rows: int
    blockers: tuple[str, ...]


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
    location_authority: LocationAuthoritySummary | None = None
    collection_schedule: CollectionScheduleSummary | None = None
    temporal: TemporalBaselineSummary | None = None
    rental_dates: RentalDateSummary | None = None
    reporting_day: ReportingDaySummary | None = None
    pricing_population: PricingPopulationSummary | None = None
    canonical_offers: CanonicalOfferSummary | None = None
    #: Every authority decision of the current record (its ``DecisionId``) and its status.
    authority_statuses: tuple[tuple[DecisionId, str], ...] = ()

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
    temporal_authority: object = None,
    reporting_days: object = None,
    pricing_population: object = None,
    authority_record: AuthorityDecisionRecord | None = None,
) -> PricingReadinessBaseline:
    """Assemble the sanitized baseline from existing assessment results (inputs are not modified).

    The rental-period plan gap is reported only while the pricing report holds
    no rental-date assessment under an available, authority-backed
    :class:`~ql2_sixt_canada_analysis.rental_dates.RentalDatePolicy`; the rules
    themselves (validity, agreement) are central ``PricingBlocker`` values.

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
        plan_gaps=_plan_gaps(pricing),
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
        location_authority=_location_summary(pricing),
        collection_schedule=_schedule_summary(pricing),
        temporal=_temporal_summary(temporal, temporal_authority),
        rental_dates=_rental_summary(pricing),
        reporting_day=_reporting_day_summary(temporal, temporal_authority, reporting_days),
        pricing_population=_population_summary(pricing_population),
        canonical_offers=_offer_summary(pricing),
        authority_statuses=(tuple((d.id, d.status.value.lower()) for d in authority_record.decisions)
                            if isinstance(authority_record, AuthorityDecisionRecord) else ()),
    )


def _reporting_day_summary(report: TemporalReconciliationReport | None, authority: object,
                           derived: object) -> ReportingDaySummary:
    """Reporting-day statuses and counts (``report_missing`` without a derived reporting-day structure)."""
    from ql2_sixt_canada_analysis.schemas import DatasetKey
    from ql2_sixt_canada_analysis.temporal import DerivedReportingDays
    from ql2_sixt_canada_analysis.temporal_authority import DATE_CLEAN_FIELD, TemporalAuthority

    available = isinstance(authority, TemporalAuthority) and authority.reporting_day_available
    clean = ("retired_from_pricing" if isinstance(authority, TemporalAuthority) and authority.date_clean_retired
             else authority.date_semantics_status.value if isinstance(authority, TemporalAuthority)
             else "report_missing")
    field = next((f for f in report.field_reports if f.ref == DATE_CLEAN_FIELD), None) if report else None
    unavailable = bool(report is not None and "date_derivation:cars.date_clean" in report.unavailable_rules)
    crossings = {r.name: r.boundary_crossing for r in report.date_checks} if report is not None else {}
    ok = isinstance(derived, DerivedReportingDays)
    return ReportingDaySummary(
        status=("available" if available and ok else "report_missing" if available else "unavailable"),
        source_field=authority.reporting_day_source if available else None,
        timezone_mode="parent_city" if available else None,
        scrape_date_derivation=authority.scrape_date_derivation.lower() if available else None,
        date_clean_status=clean, date_clean_rule_unavailable=unavailable,
        parent_status_counts=tuple(derived.status_counts(DatasetKey.JOBS).items()) if ok else (),
        detail_status_counts=tuple(derived.status_counts(DatasetKey.CARS).items()) if ok else (),
        parent_boundary_crossings=crossings.get("date_derivation:jobs.scrape_date", 0),
        detail_boundary_crossings=crossings.get("date_derivation:cars.scrape_date", 0),
        date_clean_rows=field.row_count if field else 0, date_clean_valid=field.valid_count if field else 0,
        date_clean_missing=field.missing_count if field else 0,
        date_clean_invalid=field.invalid_count if field else 0)


def _population_summary(population: object) -> PricingPopulationSummary:
    from ql2_sixt_canada_analysis.pricing_population import PricingPopulation

    if not isinstance(population, PricingPopulation):
        return PricingPopulationSummary("report_missing", 0, 0, 0, 0, 0, 0, 0, 0, ())
    return PricingPopulationSummary(
        "available", population.nominal_parent_captures, population.excluded_parent_captures,
        population.ineligible_parent_captures, population.eligible_parent_captures,
        population.nominal_detail_rows, population.excluded_detail_rows, population.ineligible_detail_rows,
        population.eligible_detail_rows, tuple(population.detail_counts().items()))


def _offer_summary(pricing: PricingReadinessReport) -> CanonicalOfferSummary:
    from ql2_sixt_canada_analysis.canonical_offers import CanonicalOfferReport

    report = pricing.canonical_offers
    if not isinstance(report, CanonicalOfferReport):
        return CanonicalOfferSummary("report_missing", (), None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (), (), 0, 0,
                                     ("canonical_offer_report_missing",))
    policy = report.policy
    sources = set(policy.source_streams)
    canonical = policy.canonical_location
    return CanonicalOfferSummary(
        policy_status=policy.status.value, source_streams=tuple(policy.source_streams), stream=canonical,
        source_rows=report.source_rows, out_of_scope_rows=report.out_of_scope_rows,
        combined_observations=report.combined_observations, unassessable_rows=report.unassessable_rows,
        canonical_offers=report.canonical_offers, unique_offers=report.unique_offers,
        deduplicated_offers=report.deduplicated_offers, duplicate_groups=report.duplicate_groups,
        collapsed_observations=report.collapsed_observations, variation_groups=report.variation_groups,
        variation_offers=report.variation_offers, unassessable_group_count=len(report.unassessable_groups),
        unassessable_groups=report.unassessable_groups,
        vancouver_source_counts=tuple((k, n) for k, n in report.source_counts if k in sources),
        vancouver_canonical_offers=dict(report.canonical_counts).get(canonical, 0) if canonical else 0,
        cross_stream_offers=report.cross_stream_offers, blockers=_codes(report.blocking_reasons))


def _temporal_summary(report: TemporalReconciliationReport | None, authority: object) -> TemporalBaselineSummary:
    """Statuses from the temporal authority and counts from the temporal report (``report_missing`` otherwise)."""
    from ql2_sixt_canada_analysis.temporal_authority import TemporalAuthority

    missing = "report_missing"
    if isinstance(authority, TemporalAuthority):
        statuses = (authority.timezone_status.value, authority.ordering_status.value,
                    authority.tolerance_status.value, authority.tolerance_seconds,
                    authority.reporting_day_status.value, authority.date_semantics_status.value,
                    _codes(authority.blocking_reasons), tuple(authority.pricing_date_fields))
    else:
        statuses = (missing, missing, missing, None, missing, missing, (), ())
    if report is None:
        return TemporalBaselineSummary(*statuses, fields=(), replication_passed=0, replication_failed=0,
                                       replication_unassessable=0, ordering_rule_status=missing, ordering_passed=0,
                                       ordering_failed=0, ordering_unassessable=0, unlinked_detail_rows=0,
                                       city_mismatch_detail_rows=0, unavailable_rules=(),
                                       temporal_fields_trusted=False)
    fields_ = tuple(TemporalFieldSummary(
        field=f"{f.dataset.value}.{f.column}", rows=f.row_count, resolved=f.resolved_count, missing=f.missing_count,
        invalid=f.invalid_count, unknown_city=f.unknown_city_count, context_unavailable=f.context_unavailable_count,
        ambiguous=f.ambiguous_count, nonexistent=f.nonexistent_count) for f in report.field_reports)
    rep_ = report.replications
    return TemporalBaselineSummary(
        *statuses, fields=fields_, replication_passed=sum(r.passed for r in rep_),
        replication_failed=sum(r.failed for r in rep_), replication_unassessable=sum(r.unassessable for r in rep_),
        ordering_rule_status=report.ordering.status.value, ordering_passed=report.ordering.passed,
        ordering_failed=report.ordering.failed, ordering_unassessable=report.ordering.unassessable,
        unlinked_detail_rows=report.unlinked_detail_row_count,
        city_mismatch_detail_rows=report.city_mismatch_detail_row_count,
        unavailable_rules=tuple(report.unavailable_rules), temporal_fields_trusted=bool(report.is_valid))


def _schedule_summary(pricing: PricingReadinessReport) -> CollectionScheduleSummary:
    """Counts from the per-stream scheduled-coverage report (``report_missing`` for anything else)."""
    from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport

    report = pricing.scheduled_coverage
    if not isinstance(report, PerStreamScheduledCoverageReport):
        return CollectionScheduleSummary(
            status="report_missing", schedule_version=None, sharing_model=None, capture_field=None,
            period_minutes=0, exceptions_model=None, excused_period_count=0, schedule_count=0, total_periods=0,
            city_periods=(), missing_city_periods=(), jobs_assessed=0, jobs_assigned=0, job_failures=(),
            detail_copy_mismatches=0, unexcused_missing_periods=0, streams=())
    schedule = report.schedule
    available = schedule.available
    return CollectionScheduleSummary(
        status=schedule.status.value, schedule_version=schedule.schedule_version if available else None,
        sharing_model=schedule.sharing_mode.value.lower() if available else None,
        capture_field=schedule.capture_field if available else None, period_minutes=60 if available else 0,
        exceptions_model=schedule.exceptions.model.value.lower() if available else None,
        excused_period_count=report.excused_total, schedule_count=len(schedule.schedules),
        total_periods=schedule.total_period_count, city_periods=tuple(schedule.period_count_by_city.items()),
        missing_city_periods=report.missing_city_periods, jobs_assessed=report.jobs_assessed,
        jobs_assigned=report.jobs_assigned, job_failures=tuple(report.job_failure_counts.items()),
        detail_copy_mismatches=report.detail_copy_mismatches,
        unexcused_missing_periods=report.unexcused_missing_total,
        streams=tuple(StreamScheduleSummary(
            stream=tuple(c.stream), expected_periods=c.expected, covered_periods=c.covered,
            unexcused_missing_periods=c.unexcused_missing, excused_periods=c.excused,
            missing_by_failure=tuple((k.lower(), n) for k, n in c.missing_by_failure.items()),
            excluded_periods=len(c.excluded), required_periods=c.required)
            for c in report.streams),
        nominal_stream_periods=report.nominal_periods, excluded_stream_periods=report.excluded_periods,
        required_stream_periods=report.required_periods, covered_stream_periods=report.covered_periods,
        excluded_parent_captures=report.excluded_parent_captures, unmatched_exclusions=report.unmatched_exclusions)


def _location_summary(pricing: PricingReadinessReport) -> LocationAuthoritySummary:
    report = pricing.location_authority
    policy = pricing.location_policy
    canonical = policy.scope.canonical_location if policy.scope is not None else None
    if report is None:
        return LocationAuthoritySummary(
            role_map_status="report_missing", roles_exact=False, airport_streams=0, downtown_streams=0,
            other_streams=0, comparison_pair_status="report_missing", comparison_pairs_valid=False,
            approved_pair_count=0, vancouver_policy_state=policy.state.value, keys=(),
            stream=tuple(canonical) if isinstance(canonical, tuple) else None)
    counts = report.role_counts
    return LocationAuthoritySummary(
        role_map_status=report.role_map.status.value, roles_exact=report.roles_exact,
        airport_streams=counts.get("AIRPORT", 0), downtown_streams=counts.get("DOWNTOWN", 0),
        other_streams=counts.get("OTHER", 0), comparison_pair_status=report.pair_set.status.value,
        comparison_pairs_valid=report.pairs_valid, approved_pair_count=len(report.effective_pairs),
        vancouver_policy_state=policy.state.value,
        keys=tuple((tuple(p.airport), tuple(p.downtown)) for p in report.effective_pairs),
        stream=tuple(canonical) if isinstance(canonical, tuple) else None)


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
    authority = pricing.location_authority
    if authority is None:
        blockers.append(("location_authority", ("branch_role_authority_unavailable",
                                                "comparison_pair_authority_unavailable")))
    else:
        statuses.append(("location_role_map", authority.role_map.status.value))
        statuses.append(("comparison_pairs", authority.pair_set.status.value))
        blockers.append(("location_authority", _codes(authority.blocking_reasons)))
    rental = pricing.rental_dates
    if rental is None:
        blockers.append(("rental_dates", ("rental_date_assessment_missing",)))
    else:
        statuses.append(("rental_date_policy", rental.policy.status.value))
        blockers.append(("rental_dates", _codes(rental.blocking_reasons)))
    offers = pricing.canonical_offers
    if offers is None:
        blockers.append(("canonical_offers", ("canonical_offer_report_missing",)))
    else:
        statuses.append(("canonical_offer_policy", offers.policy.status.value))
        blockers.append(("canonical_offers", _codes(offers.blocking_reasons)))
    policy = pricing.location_policy
    statuses.append(("location_policy_state", policy.state.value))
    blockers.append(("location_policy", _codes(policy.blocking_reasons)))
    statuses.append(("vehicle_stability", stability.status.value if stability is not None else "unavailable"))
    return tuple(blockers), tuple(statuses)


def _plan_gaps(pricing: PricingReadinessReport) -> tuple[PlanReadinessGap, ...]:
    """The rental-period gap stays open until a rental-date assessment ran under an approved policy."""
    G = PlanReadinessGap
    return () if pricing.rental_date_rules_available else (G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE,)


def _rental_summary(pricing: PricingReadinessReport) -> RentalDateSummary:
    """Statuses and aggregate counts of the rental-date assessment (no dates, identifiers or rows)."""
    report = pricing.rental_dates
    if report is None:
        return RentalDateSummary(policy_status="report_missing", maximum_duration_mode=None, parent_rows=0,
                                 detail_rows=0, fields=(), periods=(), agreements=(), linkage_trusted=False,
                                 validity_holds=False, agreement_holds=False, eligible_parent_rows=0,
                                 eligible_detail_rows=0, blockers=("rental_date_assessment_missing",))
    policy = report.policy
    return RentalDateSummary(
        policy_status=policy.status.value,
        maximum_duration_mode=policy.maximum_duration_mode.lower() if policy.maximum_duration_mode else None,
        parent_rows=report.parent_rows, detail_rows=report.detail_rows,
        fields=tuple(RentalFieldSummary(f.field, f.rows, f.valid, f.missing, f.invalid) for f in report.fields),
        periods=tuple(RentalPeriodSummary(p.period, p.rows, p.count(_PS.VALID), p.count(_PS.RETURN_BEFORE_PICKUP),
                                          p.rows - p.count(_PS.VALID) - p.count(_PS.RETURN_BEFORE_PICKUP),
                                          p.same_day, p.long_informational) for p in report.periods),
        agreements=tuple(RentalAgreementSummary(a.target, a.source, a.rows, a.count(_AS.MATCH), a.count(_AS.MISMATCH),
                                                a.rows - a.count(_AS.MATCH) - a.count(_AS.MISMATCH))
                         for a in report.agreements),
        linkage_trusted=report.linkage_trusted, validity_holds=report.validity_holds,
        agreement_holds=report.agreement_holds, eligible_parent_rows=report.eligible_parent_rows,
        eligible_detail_rows=report.eligible_detail_rows, blockers=_codes(report.blocking_reasons))


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
        schedule_available=bool(scheduled is not None and scheduled.schedule_assessment.available),
        governed_excluded_jobs=accounting.governed_excluded_jobs if accounting is not None else 0)


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
    if isinstance(value, DecisionId):
        return value.value          # a fixed decision name from the authority schema, never source data
    if isinstance(value, StrEnum):
        return _sanitize(value.value, where)
    if isinstance(value, str):
        if re.search(r"\.(keys|stream|source_streams|vancouver_source_counts)(\[\])?$", where):
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
                 ContinuityFinding, LocationAuthoritySummary, CollectionScheduleSummary, StreamScheduleSummary,
                 TemporalBaselineSummary, TemporalFieldSummary, RentalDateSummary, RentalFieldSummary,
                 RentalPeriodSummary, RentalAgreementSummary, ReportingDaySummary, PricingPopulationSummary,
                 CanonicalOfferSummary)


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
    la = d["location_authority"]
    if la is not None:
        canonical = key(la["stream"]) if la["stream"] else "none"
        lines += ["", "### Location authority (roles, comparison pairs, Vancouver identity)", "",
                  f"- Role map: `{la['role_map_status']}`, exact for every approved stream: {la['roles_exact']} "
                  f"(AIRPORT {la['airport_streams']}, DOWNTOWN {la['downtown_streams']}, OTHER {la['other_streams']}).",
                  f"- Comparison pairs: `{la['comparison_pair_status']}`, valid: {la['comparison_pairs_valid']}, "
                  f"approved within-city pairs on canonical keys: {la['approved_pair_count']}."]
        lines += [f"  - {key(a)} versus {key(dn)}" for a, dn in la["keys"]]
        lines += [f"- Vancouver identity policy: `{la['vancouver_policy_state']}`; canonical location: {canonical}. "
                  "Both raw Vancouver source streams stay separately required by the source contract."]
    rd = d["rental_dates"]
    if rd is not None:
        lines += ["", "### Rental dates (validity, parent/detail agreement, eligibility)", "",
                  f"- Policy: `{rd['policy_status']}`; maximum duration `{rd['maximum_duration_mode'] or 'none'}`; "
                  f"{rd['parent_rows']} parent rows and {rd['detail_rows']} detail rows assessed; linkage trusted: "
                  f"{rd['linkage_trusted']}.",
                  f"- Data validity holds: {rd['validity_holds']}; parent/detail agreement holds: "
                  f"{rd['agreement_holds']}.",
                  f"- Pricing-eligible rows: {rd['eligible_parent_rows']} parent, {rd['eligible_detail_rows']} detail "
                  "(validity and agreement only; no duration cohort applied).",
                  "- Blockers: " + (", ".join(f"`{b}`" for b in rd["blockers"]) or "none") + ".", "",
                  "| Field | Rows | Valid | Missing | Invalid format |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| `{f['field']}` | {f['rows']} | {f['valid']} | {f['missing']} | {f['invalid']} |"
                  for f in rd["fields"]] or ["| none | | | | |"]
        lines += ["", "| Period | Rows | Valid | Return before pickup | Missing or invalid | Same day | "
                  f">= {INFORMATIONAL_LONG_RENTAL_DAYS} days (informational) |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        lines += [f"| `{p['period']}` | {p['rows']} | {p['valid']} | {p['return_before_pickup']} | "
                  f"{p['incomplete_or_invalid']} | {p['same_day']} | {p['long_informational']} |"
                  for p in rd["periods"]] or ["| none | | | | | | |"]
        lines += ["", "| Detail field | Must equal | Rows | Match | Mismatch | Unassessable |",
                  "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| `{a['target']}` | `{a['source']}` | {a['rows']} | {a['match']} | {a['mismatch']} | "
                  f"{a['unassessable']} |" for a in rd["agreements"]] or ["| none | | | | | |"]
    tp = d["temporal"]
    if tp is not None:
        tol = f"{tp['tolerance_seconds']} seconds" if tp["tolerance_seconds"] is not None else "none"
        lines += ["", "### Temporal policy (finish-time zones, replication, scrape/finish ordering)", "",
                  f"- Authority: finish-time zone `{tp['timezone_status']}`; ordering `{tp['ordering_status']}`; "
                  f"tolerance `{tp['tolerance_status']}` ({tol}); reporting day `{tp['reporting_day_status']}`; "
                  f"date semantics `{tp['date_semantics_status']}`.",
                  "- Authority gaps: " + (", ".join(f"`{b}`" for b in tp["authority_blockers"]) or "none") + ".",
                  "- Trusted pricing date fields: " + (", ".join(f"`{f}`" for f in tp["pricing_date_fields"])
                                                       or "none (reporting day unresolved)") + ".",
                  f"- Replication of the finish time: {tp['replication_passed']} passed, "
                  f"{tp['replication_failed']} failed, {tp['replication_unassessable']} unassessable.",
                  f"- Scrape/finish ordering (`{tp['ordering_rule_status']}`): {tp['ordering_passed']} passed, "
                  f"{tp['ordering_failed']} failed, {tp['ordering_unassessable']} unassessable.",
                  f"- Unlinked detail rows: {tp['unlinked_detail_rows']}; detail rows whose city differs from "
                  f"their parent: {tp['city_mismatch_detail_rows']}.",
                  "- Unavailable rules: " + (", ".join(f"`{r}`" for r in tp["unavailable_rules"]) or "none")
                  + f"; temporal fields trusted: {tp['temporal_fields_trusted']}.", "",
                  "| Field | Rows | Resolved | Missing | Invalid | Unknown city | No parent | Ambiguous | Nonexistent |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        lines += [f"| `{f['field']}` | {f['rows']} | {f['resolved']} | {f['missing']} | {f['invalid']} | "
                  f"{f['unknown_city']} | {f['context_unavailable']} | {f['ambiguous']} | {f['nonexistent']} |"
                  for f in tp["fields"]] or ["| none | | | | | | | | |"]
    cs = d["collection_schedule"]
    if cs is not None:
        lines += ["", "### Collection schedule (authority-backed, per stream)", "",
                  f"- Schedule: `{cs['status']}`; version `{cs['schedule_version'] or 'none'}`; sharing model "
                  f"`{cs['sharing_model'] or 'none'}`; capture timestamp `{cs['capture_field'] or 'none'}`; "
                  f"period length {cs['period_minutes']} minutes; exceptions `{cs['exceptions_model'] or 'none'}` "
                  f"({cs['excused_period_count']} excused).",
                  f"- Stream schedules: {cs['schedule_count']}; expected stream-periods: {cs['total_periods']} ("
                  + (", ".join(f"{c} {n}" for c, n in cs["city_periods"]) or "none") + ").",
                  f"- Parent jobs: {cs['jobs_assessed']} assessed, {cs['jobs_assigned']} assigned to exactly one "
                  "expected city-period; assignment failures: "
                  + (", ".join(f"`{k}` {n}" for k, n in cs["job_failures"]) or "none")
                  + f"; detail copies disagreeing with their parent: {cs['detail_copy_mismatches']}.",
                  "- Expected city-periods without exactly one valid job: "
                  + (", ".join(f"{c} {n}" for c, n in cs["missing_city_periods"]) or "none") + ".",
                  f"- Stream-periods: {cs['nominal_stream_periods']} nominal, {cs['excluded_stream_periods']} "
                  f"excluded by a governed parent-capture exclusion, {cs['required_stream_periods']} required, "
                  f"{cs['covered_stream_periods']} covered; unexcused missing: {cs['unexcused_missing_periods']}.",
                  f"- Governed excluded parent captures: {cs['excluded_parent_captures']} (analytically null for "
                  "pricing; raw rows preserved, nothing deleted); exclusions matching no or several captures: "
                  f"{cs['unmatched_exclusions']}.", "",
                  "| Stream | Nominal periods | Excluded | Required | Covered | Unexcused missing | Excused | "
                  "Missing by failure |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {key(st['stream'])} | {st['expected_periods']} | {st['excluded_periods']} | "
                  f"{st['required_periods']} | {st['covered_periods']} | "
                  f"{st['unexcused_missing_periods']} | {st['excused_periods']} | "
                  + (", ".join(f"`{k}` {n}" for k, n in st["missing_by_failure"]) or "none") + " |"
                  for st in cs["streams"]] or ["| none | | | | | | | |"]
    pp = d["pricing_population"]
    if pp is not None:
        lines += ["", "### Pricing-eligible population (after every foundational control)", "",
                  f"- Status: `{pp['status']}`.",
                  f"- Parent captures: {pp['nominal_parent_captures']} nominal, {pp['excluded_parent_captures']} "
                  f"governed exclusion, {pp['ineligible_parent_captures']} failing another control, "
                  f"{pp['eligible_parent_captures']} eligible.",
                  f"- Detail rows: {pp['nominal_detail_rows']} nominal, {pp['excluded_detail_rows']} governed "
                  f"exclusion, {pp['ineligible_detail_rows']} failing another control, "
                  f"{pp['eligible_detail_rows']} eligible.",
                  "- Detail eligibility: " + (", ".join(f"`{k}` {n}" for k, n in pp["detail_status_counts"])
                                              or "none") + ".",
                  "- Excluded rows stay in ingestion, linkage, reconciliation, audit and exception reporting; "
                  "they never enter price summaries, comparisons, product populations, pricing vehicle "
                  "stability, duplicate calculations, offer counts or reporting-day cohorts."]
    rdy = d["reporting_day"]
    if rdy is not None:
        lines += ["", "### Reporting day and source dates", "",
                  f"- Reporting day: `{rdy['status']}`; source `{rdy['source_field'] or 'none'}`; zone mode "
                  f"`{rdy['timezone_mode'] or 'none'}`; scrape-date derivation "
                  f"`{rdy['scrape_date_derivation'] or 'none'}`.",
                  "- Parent scrape dates: " + (", ".join(f"`{k}` {n}" for k, n in rdy["parent_status_counts"])
                                               or "none") + f"; local-date boundary crossings (agreeing, UTC date "
                  f"differs): {rdy['parent_boundary_crossings']}.",
                  "- Detail scrape dates: " + (", ".join(f"`{k}` {n}" for k, n in rdy["detail_status_counts"])
                                               or "none") + f"; boundary crossings: {rdy['detail_boundary_crossings']}.",
                  f"- `cars.date_clean`: `{rdy['date_clean_status']}`; {rdy['date_clean_rows']} rows, "
                  f"{rdy['date_clean_valid']} valid, {rdy['date_clean_missing']} missing, "
                  f"{rdy['date_clean_invalid']} invalid (presence and parse quality only; never a pricing date); "
                  f"derivation rule still reported unavailable: {rdy['date_clean_rule_unavailable']}."]
    co = d["canonical_offers"]
    if co is not None:
        sources = " and ".join(key(k) for k in co["source_streams"]) or "none"
        lines += ["", "### Canonical offers (Vancouver alias combination)", "",
                  f"- Policy: `{co['policy_status']}`; source streams {sources}; canonical location "
                  f"{key(co['stream']) if co['stream'] else 'none'}.",
                  f"- Detail rows: {co['source_rows']} source, {co['out_of_scope_rows']} out of scope (governed "
                  f"exclusion), {co['combined_observations']} combined, {co['unassessable_rows']} unassessable.",
                  f"- Canonical offers: {co['canonical_offers']} ({co['unique_offers']} unique, "
                  f"{co['deduplicated_offers']} deduplicated); duplicate groups {co['duplicate_groups']} "
                  f"({co['collapsed_observations']} observations collapsed); price-variation groups "
                  f"{co['variation_groups']} ({co['variation_offers']} offers kept and flagged); unassessable "
                  f"groups {co['unassessable_group_count']}"
                  + (" (" + ", ".join(f"`{k}` {n}" for k, n in co["unassessable_groups"]) + ")"
                     if co["unassessable_groups"] else "") + ".",
                  "- Vancouver source observations: " + (", ".join(f"{key(k)} {n}"
                                                                   for k, n in co["vancouver_source_counts"])
                                                         or "none")
                  + f"; Vancouver canonical offers: {co['vancouver_canonical_offers']}; offers observed in more "
                  f"than one source stream: {co['cross_stream_offers']}.",
                  "- Blockers: " + (", ".join(f"`{b}`" for b in co["blockers"]) or "none") + "."]
    au = d["authority_statuses"]
    if au:
        lines += ["", "### Authority decision statuses", "", "| Decision | Status |", "| --- | --- |"]
        lines += [f"| `{k}` | `{v}` |" for k, v in au]
    lines += ["", "### Expected stream health (per approved stream)", "",
              "| Stream | Status | Continuity | Time coverage (legacy shared schedule) |", "| --- | --- | --- | --- |"]
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
                  f"{c['in_scope_capture_events']} in-scope capture events (detail rows) carry no row of the stream; "
                  f"in-scope jobs under a governed parent-capture exclusion: {c['governed_excluded_jobs']}.",
                  "- " + ("An authoritative per-stream schedule is available; scheduled coverage is reported "
                          "in the collection-schedule section." if c["schedule_available"] else
                          "No authoritative collection schedule exists, so these captures are not called "
                          "scheduled and the gap is pending a schedule.")]
    lines += ["", "Observations do not establish authority. No source-level values (identifiers, timestamps, "
              "dates, prices, vehicle names, offers or rows) are included."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- running


def run_pricing_baseline(raw_dir: str | Path | None = None) -> PricingReadinessBaseline:
    """Run the existing pipeline (the ingestion notebook's calls, in order) and build the baseline."""
    from ql2_sixt_canada_analysis.pricing_pipeline import run_pricing_pipeline

    run = run_pricing_pipeline(raw_dir)
    return build_pricing_baseline(pricing=run.pricing, jobs=run.jobs, cars=run.cars, temporal=run.temporal,
                                  vehicle_stability=run.vehicle_stability, relationship=run.relationship,
                                  temporal_contract=run.temporal_authority.definition,
                                  temporal_authority=run.temporal_authority, reporting_days=run.reporting_days,
                                  pricing_population=run.population, authority_record=run.record)


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
