"""The validated pricing pipeline: every foundational assessment, in order, on the analysis-stage frames.

:func:`run_pricing_pipeline` is the single orchestration of the existing
assessments (the ingestion notebook's calls, in the same order): raw load,
blank-row removal, identifier typing, the authority-backed job linkage (every
later step uses the derived keys, never raw ``job_id``), keys, coverage,
reconciliation, city integrity, the trusted join, the per-stream collection
schedule (with the governed ``INCOMPLETE_PARENT_CAPTURE`` exclusion), rental
dates, temporal reconciliation, the pricing-eligible population, the
pricing-population comparison and vehicle stability, the canonical offers,
completeness, the Vancouver location policy, the location authority and the
central pricing-readiness gate.

It decides nothing new: a step that cannot be assessed is ``None`` and the
central gate reports it as a blocker. The result keeps the proprietary frames
and in-memory evidence for downstream analyses
(:mod:`~ql2_sixt_canada_analysis.pricing_baseline`,
:mod:`~ql2_sixt_canada_analysis.matched_location_pricing`); it is never
printed (frames are excluded from ``repr``) and nothing is written. The coverage
and reconciliation reports that completeness was decided on are retained too,
so the monitoring controls (:mod:`~ql2_sixt_canada_analysis.monitoring`) read
them instead of recomputing them. Every run carries a
:class:`PipelineEvidenceManifest` (frame bindings and the exact report objects
of that run), so substituted or mixed-run evidence is detectable by
:func:`pipeline_evidence_bound` without reassessing anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

__all__ = ["BOUND_PIPELINE_REPORTS", "PipelineEvidenceManifest", "PricingPipelineResult",
           "bind_pipeline_evidence", "pipeline_evidence_bound", "run_pricing_pipeline"]

#: The retained reports whose exact objects the evidence manifest binds to one run, and the input each was
#: assessed from: ``analysis_frames`` (the run's linked ``jobs`` and ``cars``), ``pricing_eligible_frames``
#: (the eligible rows of the bound population), ``authority_record`` (the decision record and contract) or
#: ``cleaned_frames`` (the pre-linkage frames that the job linkage keyed; never retained, so bound by identity).
BOUND_PIPELINE_REPORTS: tuple[tuple[str, str], ...] = (
    ("record", "authority_record"), ("contract", "authority_record"), ("job_linkage", "cleaned_frames"),
    ("unique_keys", "analysis_frames"), ("coverage", "analysis_frames"), ("reconciliation", "analysis_frames"),
    ("scheduled", "analysis_frames"), ("temporal_authority", "authority_record"), ("temporal", "analysis_frames"),
    ("reporting_days", "analysis_frames"), ("population", "analysis_frames"),
    ("vehicle_stability", "pricing_eligible_frames"), ("canonical_offers", "analysis_frames"),
    ("location_authority", "authority_record"), ("pricing", "analysis_frames"))


@dataclass(frozen=True)
class PricingPipelineResult:
    """Every assessment of one pipeline run (in-memory evidence; never printed or written)."""

    record: object = field(repr=False)
    contract: object = field(repr=False)
    relationship: object = field(repr=False)
    jobs: pd.DataFrame = field(repr=False)
    cars: pd.DataFrame = field(repr=False)
    job_linkage: object = field(repr=False)
    unique_keys: object = field(repr=False)
    scheduled: object = field(repr=False)
    temporal: object = field(repr=False)
    temporal_authority: object = field(repr=False)
    reporting_days: object = field(repr=False)
    population: object = field(repr=False)
    vehicle_stability: object = field(repr=False)
    canonical_offers: object = field(repr=False)
    location_authority: object = field(repr=False)
    pricing: object = field(repr=False)
    #: The expected-location coverage report already computed for completeness (``None`` without a contract).
    coverage: object = field(default=None, repr=False)
    #: The per-job job/detail reconciliation report already computed for completeness (``None`` = unassessable).
    reconciliation: object = field(default=None, repr=False)
    #: Provenance of every retained report (:func:`bind_pipeline_evidence`); ``None`` = unbound.
    evidence: PipelineEvidenceManifest | None = field(default=None, repr=False, compare=False)

    @property
    def pricing_analysis_ready(self) -> bool:
        """The central readiness gate passed (no blocking reason)."""
        return bool(self.pricing.ready)


@dataclass(frozen=True, eq=False)
class PipelineEvidenceManifest:
    """Immutable provenance of one pipeline run: its frame bindings and the exact report objects it produced.

    Captured once, when the run is built. ``frames`` binds the linked analysis frames; ``eligible`` binds the
    pricing-eligible frames that vehicle stability was assessed on (``None`` without a population). ``reports``
    holds each bound report object itself, so a structurally valid or even equal-valued report from another run
    is still foreign: provenance is object identity plus frame bindings, never report values. Nothing here is
    printed: every field is excluded from ``repr`` and from equality.
    """

    frames: object = field(repr=False)
    eligible: object = field(repr=False)
    reports: tuple[tuple[str, object], ...] = field(repr=False)

    def __post_init__(self) -> None:
        from ql2_sixt_canada_analysis.pricing_population import FrameBinding

        if not isinstance(self.frames, FrameBinding) or not (
                self.eligible is None or isinstance(self.eligible, FrameBinding)):
            raise TypeError("a manifest binds frames with FrameBinding objects")
        if tuple(name for name, _ in self.reports) != tuple(name for name, _ in BOUND_PIPELINE_REPORTS):
            raise ValueError("a manifest binds exactly the retained pipeline reports, in order")


def _eligible_binding(run: PricingPipelineResult):  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis.pricing_population import frame_binding

    population = run.population
    if population is None:
        return None
    return frame_binding(population.eligible_parents(run.jobs, run.cars),
                         population.eligible_details(run.jobs, run.cars))


def bind_pipeline_evidence(run: PricingPipelineResult) -> PricingPipelineResult:
    """The same run with its evidence manifest captured (called once, where the run is constructed).

    The manifest records the exact retained report objects and the bindings of the frames they were assessed
    from. Replacing any bound report or frame afterwards (for example with :func:`dataclasses.replace`) breaks
    the binding, which :func:`pipeline_evidence_bound` detects.
    """
    from ql2_sixt_canada_analysis.pricing_population import frame_binding

    if not isinstance(run, PricingPipelineResult):
        raise TypeError("run must be a PricingPipelineResult")
    manifest = PipelineEvidenceManifest(
        frames=frame_binding(run.jobs, run.cars), eligible=_eligible_binding(run),
        reports=tuple((name, getattr(run, name)) for name, _ in BOUND_PIPELINE_REPORTS))
    return replace(run, evidence=manifest)


def pipeline_evidence_bound(run: object) -> bool:
    """Whether every retained report is the exact object bound when the run was built, on the same frames.

    Checks object identity of every bound report, the analysis-frame binding, the pricing-eligible-frame
    binding, and the bindings the population and canonical offers carry themselves. Nothing is reassessed:
    no coverage, reconciliation, temporal or stability assessment runs here.
    """
    from ql2_sixt_canada_analysis.pricing_population import frame_binding

    if not isinstance(run, PricingPipelineResult) or not isinstance(run.evidence, PipelineEvidenceManifest):
        return False
    manifest = run.evidence
    if any(getattr(run, name) is not report for name, report in manifest.reports):
        return False
    if not isinstance(run.jobs, pd.DataFrame) or not isinstance(run.cars, pd.DataFrame) \
            or frame_binding(run.jobs, run.cars) != manifest.frames:
        return False
    population, offers = run.population, run.canonical_offers
    if population is not None and getattr(population, "binding", None) != manifest.frames:
        return False
    if offers is not None and getattr(offers, "binding", None) not in (None, manifest.frames):
        return False
    return _eligible_binding(run) == manifest.eligible


def run_pricing_pipeline(raw_dir: str | Path | None = None) -> PricingPipelineResult:
    """Run every foundational assessment on the raw files of ``raw_dir`` (default: the configured raw directory)."""
    from ql2_sixt_canada_analysis import (  # local import: the package re-exports this module
        ANALYSIS_DATASET_DEFINITIONS, ANALYSIS_JOB_DETAIL_RELATIONSHIP, ANALYSIS_LOCATION_STREAM_COMPARISON,
        ANALYSIS_TEMPORAL_RECONCILIATION, COLLECTION_SCHEDULE, VANCOUVER_LOCATION_POLICY, VEHICLE_ATTRIBUTE_STABILITY,
        assess_job_linkage, load_job_linkage_policy,
        apply_location_policy, assess_city_integrity, assess_completeness,
        assess_dataset_location_coverage, assess_expected_location_streams, assess_job_detail_join_readiness,
        assess_job_detail_reconciliation, assess_location_policy, assess_one_to_many_join,
        assess_pricing_readiness, assess_raw_dataset_unique_keys,
        assess_temporal_reconciliation, assess_vehicle_attribute_stability, compare_location_streams,
        load_raw_datasets, remove_blank_rows_from_raw_datasets, validate_raw_dataset_identifier_dtypes,
    )
    from ql2_sixt_canada_analysis.collection_schedule import (
        ScheduleConfigurationError, assess_per_stream_scheduled_coverage, schedule_from_record,
    )
    from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
    from ql2_sixt_canada_analysis.temporal_authority import temporal_authority_from_record
    from ql2_sixt_canada_analysis.rental_dates import (
        assess_rental_dates, derive_rental_periods, rental_date_policy_from_record,
    )
    from ql2_sixt_canada_analysis.temporal import TemporalConfigurationError, derive_reporting_days
    from ql2_sixt_canada_analysis.pricing_population import PricingPopulationError, build_pricing_population
    from ql2_sixt_canada_analysis.canonical_offers import assess_canonical_offers, canonical_offer_policy_from_record
    from ql2_sixt_canada_analysis.location_authority import location_authority_from_record
    from ql2_sixt_canada_analysis.authority_decisions import load_current_decision_record
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
    # The authority-backed per-stream schedule (one schedule per approved stream; parent jobs.finished_at),
    # including the governed parent-capture exclusions it resolved (raw rows are kept; never deleted).
    record = load_current_decision_record()
    schedule = schedule_from_record(record, contract)
    scheduled = attempt(lambda: assess_per_stream_scheduled_coverage(jobs, cars, schedule=schedule, contract=contract,
                                                                     relationship=rel), ScheduleConfigurationError)
    exclusions = scheduled.capture_exclusions if scheduled is not None else None
    streams = (assess_expected_location_streams(jobs, cars, coverage=cov, relationship=rel, loaded=raw,
                                                schedule=COLLECTION_SCHEDULE, capture_exclusions=exclusions)
               if configured else None)
    # Authority-backed rental-date validity and parent/detail agreement on the linked frames.
    rental_dates = attempt(lambda: assess_rental_dates(jobs, cars, policy=rental_date_policy_from_record(record),
                                                       job_linkage=linkage.report, relationship=rel), ValueError)
    # The authority-backed temporal policy (city-local finish times, zero-tolerance scrape/finish ordering).
    temporal_authority = temporal_authority_from_record(record, ANALYSIS_TEMPORAL_RECONCILIATION, contract)
    temporal = attempt(lambda: assess_temporal_reconciliation(jobs, cars, temporal_authority.definition),
                       RelationshipPreconditionError)
    # After temporal validation: the derived reporting day (raw values kept), rental eligibility, and the
    # pricing-eligible population - the governed Calgary exclusion applies here, before any cohort.
    reporting_days = attempt(lambda: derive_reporting_days(jobs, cars, temporal_authority.definition),
                             TemporalConfigurationError, RelationshipPreconditionError)
    rental_periods = attempt(lambda: derive_rental_periods(jobs, cars, policy=rental_date_policy_from_record(record),
                                                           job_linkage=linkage.report, relationship=rel), ValueError)
    population = attempt(lambda: build_pricing_population(jobs, cars, scheduled=scheduled,
                                                          reporting_days=reporting_days,
                                                          rental_periods=rental_periods), PricingPopulationError)
    eligible_jobs = population.eligible_parents(jobs, cars) if population is not None else None
    eligible_cars = population.eligible_details(jobs, cars) if population is not None else None
    # Comparisons and pricing vehicle stability use only the pricing-eligible population (fail closed without it).
    comparison = (attempt(lambda: compare_location_streams(eligible_jobs, eligible_cars,
                                                           ANALYSIS_LOCATION_STREAM_COMPARISON),
                          RelationshipPreconditionError) if population is not None else None)
    stability = (attempt(lambda: assess_vehicle_attribute_stability(eligible_cars, VEHICLE_ATTRIBUTE_STABILITY),
                         VehicleStabilityPreconditionError) if population is not None else None)
    offers = (attempt(lambda: assess_canonical_offers(
        jobs, cars, population=population, scheduled=scheduled,
        policy=canonical_offer_policy_from_record(record, contract)), PricingPopulationError)
        if population is not None else None)
    completeness = (assess_completeness(datasets=analysis, coverage=coverage, streams=streams,
                                        reconciliation=reconciliation, city_integrity=city, expected_coverage=cov)
                    if configured else None)
    policy = assess_location_policy(VANCOUVER_LOCATION_POLICY, comparison,
                                    apply_location_policy(cars, VANCOUVER_LOCATION_POLICY))
    location_authority = location_authority_from_record(record, contract, VANCOUVER_LOCATION_POLICY)
    pricing = assess_pricing_readiness(
        location_policy=policy, completeness=completeness, key_contracts_valid=bool(keys.all_valid),
        one_to_many_contract_valid=bool(relationship is not None and relationship.is_valid),
        temporal_fields_trusted=bool(temporal is not None and temporal.is_valid),
        vehicle_stability=stability, scheduled_coverage=scheduled, job_detail_join=join,
        job_linkage=linkage.report, expected_stream_contract=contract,
        rental_dates=rental_dates, canonical_offers=offers, location_authority=location_authority)
    return bind_pipeline_evidence(PricingPipelineResult(
        record=record, contract=contract, relationship=rel, jobs=jobs, cars=cars, job_linkage=linkage.report,
        unique_keys=keys, scheduled=scheduled, temporal=temporal, temporal_authority=temporal_authority,
        reporting_days=reporting_days, population=population, vehicle_stability=stability,
        canonical_offers=offers, location_authority=location_authority, pricing=pricing,
        coverage=coverage, reconciliation=reconciliation))
