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
  ``PricingBlocker`` values (a ``MINIMUM_REQUIRED`` expected-stream contract,
  no airport/downtown role map, no rental-period date rules). They are derived
  from configuration only and are never presented as active blockers.

Stream populations are reported separately: the configured expected population
comes from the coverage contract; the observed population is the distinct
complete (city, location) keys in the cleaned detail rows
(:func:`~ql2_sixt_canada_analysis.coverage.location_pair_evidence`, exact and
unnormalised). Observations never establish authority.

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

from ql2_sixt_canada_analysis.coverage import location_pair_evidence
from ql2_sixt_canada_analysis.readiness import PricingReadinessReport
from ql2_sixt_canada_analysis.schemas import (
    EXPECTED_LOCATION_COVERAGE,
    LocationPolicyAuthority,
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
    "rental_date_fields",
    "LocationRole",
    "BaselineInputError",
    "ContinuityFinding",
    "PlanReadinessGap",
    "PricingReadinessBaseline",
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


class BaselineInputError(ValueError):
    """A required assessment input is missing, empty or malformed (fail closed)."""


class UnsafeBaselineValueError(ValueError):
    """A value outside the sanitized vocabulary would be serialized (message holds no value)."""


class LocationRole(StrEnum):
    """The only roles a location-role map may assign (an allowlist)."""

    AIRPORT = "airport"
    DOWNTOWN = "downtown"


class PlanReadinessGap(StrEnum):
    """Plan prerequisites NOT modeled as ``PricingBlocker`` values (never active blockers)."""

    EXPECTED_STREAMS_NOT_EXHAUSTIVE = "expected_streams_minimum_required_not_exhaustive"
    LOCATION_ROLE_MAP_UNAVAILABLE = "airport_downtown_role_map_unavailable"
    RENTAL_PERIOD_DATE_RULES_UNAVAILABLE = "rental_period_date_rules_unavailable"


_GAP_TEXT = {
    PlanReadinessGap.EXPECTED_STREAMS_NOT_EXHAUSTIVE:
        "The expected-stream contract is a required minimum, not an authoritative exhaustive stream universe.",
    PlanReadinessGap.LOCATION_ROLE_MAP_UNAVAILABLE:
        "No authoritative airport/downtown role map is configured; central readiness does not check roles.",
    PlanReadinessGap.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE:
        "Pickup/return dates have no temporal-contract rules; central readiness does not check them.",
}


@dataclass(frozen=True, slots=True)
class StreamPopulation:
    """One stream population (configured expected or observed); keys sorted and unique."""

    population: str            # "configured_expected" | "observed"
    authority: str             # "authoritative_minimum_required" | "authoritative_exhaustive" | "observed_not_authoritative"
    coverage_mode: str
    keys: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        return len(self.keys)


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
    observed_population: StreamPopulation
    continuity: ContinuityFinding | None

    def to_dict(self) -> dict:
        """Plain, sanitized structure; raises :class:`UnsafeBaselineValueError` on anything unsafe."""
        return _sanitize(self, "baseline")


# ------------------------------------------------------------------- building


def build_pricing_baseline(
    *,
    pricing: PricingReadinessReport,
    cars: pd.DataFrame,
    temporal: TemporalReconciliationReport | None,
    vehicle_stability: VehicleStabilityReport | None,
    coverage: LocationCoverageDefinition = EXPECTED_LOCATION_COVERAGE,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
    temporal_contract: TemporalReconciliationDefinition = TEMPORAL_RECONCILIATION,
    investigated_stream: tuple[str, ...] = INVESTIGATED_LOCATION_STREAM,
    location_role_map: Mapping[tuple[str, ...], LocationRole] | None = None,
    location_role_authority: LocationPolicyAuthority | None = None,
    rental_period_rule_authority: LocationPolicyAuthority | None = None,
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

    Raises:
        BaselineInputError: A required report is missing or of the wrong type,
            the detail frame is empty or lacks the location columns.
    """
    if not isinstance(pricing, PricingReadinessReport):
        raise BaselineInputError("a PricingReadinessReport is required")
    if not isinstance(cars, pd.DataFrame) or cars.empty:
        raise BaselineInputError("a non-empty cleaned detail frame is required")
    if temporal is not None and not isinstance(temporal, TemporalReconciliationReport):
        raise BaselineInputError("temporal must be a TemporalReconciliationReport or None")
    if vehicle_stability is not None and not isinstance(vehicle_stability, VehicleStabilityReport):
        raise BaselineInputError("vehicle_stability must be a VehicleStabilityReport or None")
    if not isinstance(coverage, LocationCoverageDefinition) or not coverage.is_configured:
        raise BaselineInputError("a configured expected-location contract is required")
    if not set(coverage.location_columns) <= set(cars.columns):
        raise BaselineInputError("the detail frame lacks the location columns")
    if investigated_stream not in coverage.expected_locations:
        raise BaselineInputError("the investigated stream must be a configured expected stream")

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
            authority=("authoritative_exhaustive" if coverage.mode is LocationCoverageMode.EXHAUSTIVE
                       else "authoritative_minimum_required"),
            coverage_mode=coverage.mode.value,
            keys=tuple(sorted(dict.fromkeys(tuple(k) for k in coverage.expected_locations)))),
        observed_population=StreamPopulation(
            population="observed", authority="observed_not_authoritative", coverage_mode=coverage.mode.value,
            keys=_observed_keys(cars, coverage)),
        continuity=_continuity(pricing, cars, coverage, relationship, investigated_stream),
    )


def _codes(values) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
    return tuple(dict.fromkeys(getattr(v, "value", v) for v in values))


def _subordinate(pricing: PricingReadinessReport, temporal: TemporalReconciliationReport | None,
                 stability: VehicleStabilityReport | None):  # type: ignore[no-untyped-def]
    """Blockers and statuses exactly as the existing typed reports emitted them (fixed source order)."""
    blockers: list[tuple[str, tuple[str, ...]]] = []
    statuses: list[tuple[str, str]] = []
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
    if coverage.mode is not LocationCoverageMode.EXHAUSTIVE:
        gaps.append(G.EXPECTED_STREAMS_NOT_EXHAUSTIVE)
    required_keys = {tuple(k) for k in coverage.expected_locations} | set(_observed_keys(cars, coverage))
    if not _role_map_sufficient(role_map, role_authority, required_keys, len(coverage.location_columns)):
        gaps.append(G.LOCATION_ROLE_MAP_UNAVAILABLE)
    if not _rental_rules_sufficient(temporal_contract, relationship, rental_authority, approved_agreements):
        gaps.append(G.RENTAL_PERIOD_DATE_RULES_UNAVAILABLE)
    return tuple(g for g in G if g in gaps)


def _role_map_sufficient(role_map: object, authority: object, required: set, width: int) -> bool:
    """Authority-backed, typed roles for every expected and observed stream; anything less is a gap."""
    if not isinstance(authority, LocationPolicyAuthority) or not isinstance(role_map, Mapping) or not role_map:
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
    if not isinstance(contract, TemporalReconciliationDefinition) or not isinstance(authority, LocationPolicyAuthority):
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


_SERIALIZABLE = (PricingReadinessBaseline, StreamPopulation, ContinuityFinding)


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
    for title, pop in (("Configured expected streams", d["expected_population"]),
                       ("Observed streams (cleaned detail rows)", d["observed_population"])):
        lines += ["", f"### {title}: {pop['count']}", "",
                  f"Population: `{pop['population']}`; authority: `{pop['authority']}`; "
                  f"contract coverage mode: `{pop['coverage_mode']}`.", ""]
        lines += [f"- {key(k)}" for k in pop["keys"]]
    c = d["continuity"]
    lines += ["", "### Investigated stream continuity (aggregate)", ""]
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
        COLLECTION_SCHEDULE, LOCATION_STREAM_COMPARISON, VANCOUVER_LOCATION_POLICY, VEHICLE_ATTRIBUTE_STABILITY,
        apply_location_policy, assess_city_integrity, assess_collection_schedule, assess_completeness,
        assess_dataset_location_coverage, assess_expected_location_streams, assess_job_detail_join_readiness,
        assess_job_detail_reconciliation, assess_location_policy, assess_one_to_many_join,
        assess_pricing_readiness, assess_raw_dataset_unique_keys, assess_scheduled_time_coverage,
        assess_temporal_reconciliation, assess_vehicle_attribute_stability, compare_location_streams,
        load_raw_datasets, remove_blank_rows_from_raw_datasets, validate_raw_dataset_identifier_dtypes,
    )
    from ql2_sixt_canada_analysis.paths import resolve_raw_data_dir
    from ql2_sixt_canada_analysis.relationships import RelationshipPreconditionError
    from ql2_sixt_canada_analysis.stability import VehicleStabilityPreconditionError

    def attempt(assess, *errors):  # type: ignore[no-untyped-def]
        try:
            return assess()
        except errors:
            return None

    rel, cov = JOB_DETAIL_RELATIONSHIP, EXPECTED_LOCATION_COVERAGE
    raw = load_raw_datasets(resolve_raw_data_dir(raw_dir))
    cleaned = remove_blank_rows_from_raw_datasets(raw).cleaned
    jobs, cars = cleaned.jobs, cleaned.cars
    validate_raw_dataset_identifier_dtypes(cleaned)
    keys = assess_raw_dataset_unique_keys(cleaned)
    coverage = assess_dataset_location_coverage(cleaned, cov)
    reconciliation = attempt(lambda: assess_job_detail_reconciliation(jobs, cars, rel), RelationshipPreconditionError)
    relationship = attempt(lambda: assess_one_to_many_join(jobs, cars, rel), RelationshipPreconditionError)
    city = assess_city_integrity(jobs, cars, relationship=rel, coverage=cov)
    join = assess_job_detail_join_readiness(jobs, cars, rel)
    streams = assess_expected_location_streams(jobs, cars, coverage=cov, relationship=rel, loaded=raw,
                                               schedule=COLLECTION_SCHEDULE)
    scheduled = assess_scheduled_time_coverage(assess_collection_schedule(COLLECTION_SCHEDULE), streams)
    temporal = attempt(lambda: assess_temporal_reconciliation(jobs, cars, TEMPORAL_RECONCILIATION),
                       RelationshipPreconditionError)
    comparison = attempt(lambda: compare_location_streams(jobs, cars, LOCATION_STREAM_COMPARISON),
                         RelationshipPreconditionError)
    stability = attempt(lambda: assess_vehicle_attribute_stability(cars, VEHICLE_ATTRIBUTE_STABILITY),
                        VehicleStabilityPreconditionError)
    completeness = assess_completeness(datasets=cleaned, coverage=coverage, streams=streams,
                                       reconciliation=reconciliation, city_integrity=city)
    policy = assess_location_policy(VANCOUVER_LOCATION_POLICY, comparison,
                                    apply_location_policy(cars, VANCOUVER_LOCATION_POLICY))
    pricing = assess_pricing_readiness(
        location_policy=policy, completeness=completeness, key_contracts_valid=bool(keys.all_valid),
        one_to_many_contract_valid=bool(relationship is not None and relationship.is_valid),
        temporal_fields_trusted=bool(temporal is not None and temporal.is_valid),
        vehicle_stability=stability, scheduled_coverage=scheduled, job_detail_join=join)
    return build_pricing_baseline(pricing=pricing, cars=cars, temporal=temporal, vehicle_stability=stability)


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
