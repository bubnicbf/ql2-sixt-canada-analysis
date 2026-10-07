"""Trace one expected location stream through the pipeline and find where it fails.

A *location stream* is one expected location key from the central
:data:`~ql2_sixt_canada_analysis.schemas.EXPECTED_LOCATION_COVERAGE` contract,
followed across its collection events. :func:`investigate_location_stream`
checks, in pipeline order, and reports the **earliest failing stage** plus a
single primary :class:`LocationStreamStatus`:

1. ``CONFIGURATION`` - the target is a configured expected key of the right
   arity (otherwise ``EXPECTATION_NOT_CONFIGURED``).
2. ``RAW_SOURCE`` / ``INGESTION`` / ``CLEANING`` - exact presence in the raw
   CSV (optional, header-aware streaming scan), in the loaded frame (optional)
   and in the cleaned frame. Lost between stages -> ``INGESTION_EXCLUSION`` or
   ``CLEANING_EXCLUSION``; never present -> ``RAW_STREAM_ABSENT``.
3. ``LOCATION_MATCHING`` - if the exact key is absent but a case-, space-,
   punctuation- or component-order variant exists, ``UNVERIFIED_ALIAS``: the
   variant is *not* applied (only aliases declared centrally are matched) and
   its value is never reported.
4. ``SOURCE_CONTINUITY`` - for a contract with ``stream_scope_columns``, the
   denominator is every **job** of the target's scope, taken from the jobs
   frame through ``parent_scope_columns`` - jobs with zero detail rows
   included - never from the detail rows alone (:class:`StreamEventAccounting`).
   In-scope jobs whose linked details lack the target -> ``RAW_STREAM_PARTIAL``.
   In-scope jobs with no linked detail rows cannot be assigned to a branch
   (jobs carry only a city): a declared zero-offer capture or a job whose
   details are missing makes branch continuity unprovable ->
   ``CONTINUITY_UNASSESSABLE``; no branch is invented for them. No in-scope
   job, or no ``parent_scope_columns``, is also unassessable. This compares
   actual jobs; it does not infer a cadence. Any job whose scope (city) is
   *unassignable* - missing, blank, whitespace-only or padded, as defined once
   by :func:`~ql2_sixt_canada_analysis.city_integrity.unassignable_scope_mask`
   - fails continuity for **every** scoped stream (``SCOPE_UNASSIGNABLE``):
   the stream it belongs to cannot be proven, so it may not silently drop out
   of all of them, and no city is guessed for it.
5. ``TIME_COVERAGE`` - only with an authoritative
   :class:`~ql2_sixt_canada_analysis.schemas.CollectionScheduleDefinition`;
   otherwise temporal completeness is ``NOT_ASSESSED``. Observed collection
   times are resolved to instants by the central temporal contract
   (:func:`~ql2_sixt_canada_analysis.temporal.parse_temporal_field`) - there
   is no separate timestamp policy here. The schedule fails when no scheduled
   period is observed (``SCHEDULED_TIME_ABSENT``), when only some are
   (``RAW_STREAM_PARTIAL``), and fails closed when any observed target time is
   missing, invalid or lacks timezone authority (``SCHEDULED_TIME_UNASSESSABLE``;
   naive values are never read as UTC).
6. ``IDENTIFIER_TYPES`` / ``PARENT_KEYS`` - relationship keys use the
   identifier dtype and the parent (jobs) key is complete and unique.
   ``SCOPE_AGREEMENT`` - every linked detail row that concerns the stream (a
   target row, a row of the target's scope, or a row of a job in that scope)
   carries the same assignable scope as its parent job
   (``relationship.scope_agreement_columns``); otherwise
   ``PARENT_DETAIL_SCOPE_MISMATCH``. A cross-city row is therefore never
   counted under its own city in place of its parent's.
7. ``RELATIONSHIP`` / ``DETAIL_PRESENCE`` / ``RECONCILIATION`` - target detail
   rows link to exactly one job (``RELATIONSHIP_LINK_FAILURE``), target jobs
   have details (``JOBS_PRESENT_DETAILS_ABSENT``) and the target's jobs
   reconcile declared and observed counts (``JOB_DETAIL_COUNT_MISMATCH``).
8. Otherwise ``STREAM_PRESENT_AND_HEALTHY``.

All expected streams
--------------------
:func:`assess_expected_location_streams` investigates **every** key of the
coverage contract's ``expected_locations`` exactly once (in contract order)
and returns an :class:`ExpectedLocationStreamsReport`. The aggregate pairs
each report with the configured key it was produced for and validates the
population against the contract on every construction: a missing,
duplicated, unexpected or unavailable report, or any unhealthy expected
stream, is a :class:`ExpectedStreamBlocker`. Completeness and pricing
readiness consume this aggregate, never a single stream; adding a pair to the
contract automatically expands the required population.

The investigation never creates, repairs, filters or writes records and never
infers aliases or cadence. The report holds enums and booleans only - no
identifiers, location values (other than none at all), timestamps, rows or
counts.
"""

from __future__ import annotations

import csv
import itertools
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.city_integrity import check_scope_contract, scope_agreement, unassignable_scope_mask
from ql2_sixt_canada_analysis.identifiers import is_identifier_dtype
from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.reconciliation import _VALID, _classify_expected_counts, assess_job_detail_reconciliation
from ql2_sixt_canada_analysis.relationships import assess_one_to_many_join
from ql2_sixt_canada_analysis.schemas import (
    PROJECT_DEFAULT,
    project_default,
    COLLECTION_SCHEDULE,
    JOB_DETAIL_RELATIONSHIP,
    TEMPORAL_RECONCILIATION,
    CollectionScheduleDefinition,
    JobDetailRelationshipDefinition,
    LocationCoverageConfigurationError,
    LocationCoverageDefinition,
    TemporalConfigurationError,
    TemporalFieldDefinition,
    TemporalKind,
    TemporalReconciliationDefinition,
)
from ql2_sixt_canada_analysis.temporal import parse_temporal_field
from ql2_sixt_canada_analysis.unique_keys import assess_unique_key

__all__ = [
    "CollectionScheduleAssessment",
    "CollectionScheduleStatus",
    "ScheduledCoverageBlocker",
    "ScheduledCoverageReport",
    "StreamTimeCoverage",
    "assess_collection_schedule",
    "assess_scheduled_time_coverage",
    "ExpectedLocationStreamsReport",
    "ExpectedStreamBlocker",
    "ExpectedStreamResult",
    "assess_expected_location_streams",
    "LocationStreamError",
    "LocationStreamInvestigationReport",
    "LocationStreamStatus",
    "PipelineStage",
    "StreamContinuity",
    "StreamEventAccounting",
    "TimeCoverageStatus",
    "investigate_location_stream",
    "resolve_expected_location",
    "validate_location_stream",
]


class LocationStreamStatus(StrEnum):
    """Primary categorical outcome of a stream investigation."""

    EXPECTATION_NOT_CONFIGURED = "expectation_not_configured"
    RAW_STREAM_ABSENT = "raw_stream_absent"
    INGESTION_EXCLUSION = "ingestion_exclusion"
    CLEANING_EXCLUSION = "cleaning_exclusion"
    UNVERIFIED_ALIAS = "unverified_alias"
    RAW_STREAM_PARTIAL = "raw_stream_partial"
    IDENTIFIER_TYPE_MISMATCH = "identifier_type_mismatch"
    PARENT_KEY_VIOLATION = "parent_key_violation"
    RELATIONSHIP_LINK_FAILURE = "relationship_link_failure"
    JOBS_PRESENT_DETAILS_ABSENT = "jobs_present_details_absent"
    JOB_DETAIL_COUNT_MISMATCH = "job_detail_count_mismatch"
    CONTINUITY_UNASSESSABLE = "continuity_unassessable"
    SCOPE_UNASSIGNABLE = "parent_scope_unassignable"
    PARENT_DETAIL_SCOPE_MISMATCH = "parent_detail_scope_mismatch"
    SCHEDULED_TIME_ABSENT = "scheduled_time_absent"
    SCHEDULED_TIME_UNASSESSABLE = "scheduled_time_unassessable"
    STREAM_PRESENT_AND_HEALTHY = "stream_present_and_healthy"


class PipelineStage(StrEnum):
    """Pipeline stages in investigation order."""

    CONFIGURATION = "configuration"
    RAW_SOURCE = "raw_source"
    INGESTION = "ingestion"
    CLEANING = "cleaning"
    LOCATION_MATCHING = "location_matching"
    SOURCE_CONTINUITY = "source_continuity"
    TIME_COVERAGE = "time_coverage"
    IDENTIFIER_TYPES = "identifier_types"
    PARENT_KEYS = "parent_keys"
    SCOPE_AGREEMENT = "scope_agreement"
    RELATIONSHIP = "relationship"
    DETAIL_PRESENCE = "detail_presence"
    RECONCILIATION = "reconciliation"


class StreamContinuity(StrEnum):
    """Presence of the target across the collection events of its scope."""

    NOT_APPLICABLE = "not_applicable"   # no scope configured, or target absent
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNASSESSABLE = "unassessable"       # in-scope jobs cannot be assigned to the branch
    SCOPE_UNASSIGNABLE = "scope_unassignable"   # some job's scope (city) is unassignable


@dataclass(frozen=True, slots=True)
class StreamEventAccounting:
    """Job-level continuity denominator for one stream (counts only, no identifiers).

    ``total_jobs = scope_excluded_jobs + scope_unassignable_jobs + in_scope_jobs``
    and ``in_scope_jobs = jobs_with_target_details + jobs_with_other_details_only
    + zero_offer_jobs + missing_detail_jobs``. Zero-offer jobs declare zero
    detail rows (a capture that returned no offers); missing-detail jobs
    declare a positive or invalid count but have no linked detail rows. Both
    have no branch identity and stay in the denominator.
    ``scope_unassignable_jobs`` (missing, blank or non-canonical job scope)
    belong to no proven scope; any of them fails continuity
    (:attr:`StreamContinuity.SCOPE_UNASSIGNABLE`).
    """

    total_jobs: int
    scope_excluded_jobs: int
    scope_unassignable_jobs: int
    in_scope_jobs: int
    jobs_with_target_details: int
    jobs_with_other_details_only: int
    zero_offer_jobs: int
    missing_detail_jobs: int
    #: In-scope jobs whose whole capture is a governed ``INCOMPLETE_PARENT_CAPTURE`` (neither presence nor gap).
    governed_excluded_jobs: int = 0

    def __post_init__(self) -> None:
        assert self.total_jobs == self.scope_excluded_jobs + self.scope_unassignable_jobs + self.in_scope_jobs
        assert self.in_scope_jobs == (self.jobs_with_target_details + self.jobs_with_other_details_only
                                      + self.zero_offer_jobs + self.missing_detail_jobs
                                      + self.governed_excluded_jobs)

    @property
    def zero_detail_jobs(self) -> int:
        return self.zero_offer_jobs + self.missing_detail_jobs

    @property
    def branch_unassignable_jobs(self) -> int:
        """In-scope jobs without linked details: no branch identity is available."""
        return self.zero_detail_jobs


class TimeCoverageStatus(StrEnum):
    """Target presence across an authoritative schedule's periods."""

    NOT_ASSESSED = "not_assessed"       # no authoritative schedule
    NEVER_PRESENT = "never_present"
    PARTIAL = "partial"
    COMPLETE = "complete"
    UNASSESSABLE = "unassessable"       # an observed time could not be reconciled


_REPOSITORY_STATUSES = frozenset({
    LocationStreamStatus.EXPECTATION_NOT_CONFIGURED, LocationStreamStatus.INGESTION_EXCLUSION,
    LocationStreamStatus.CLEANING_EXCLUSION, LocationStreamStatus.IDENTIFIER_TYPE_MISMATCH,
})
_UPSTREAM_STATUSES = frozenset({
    LocationStreamStatus.RAW_STREAM_ABSENT, LocationStreamStatus.RAW_STREAM_PARTIAL,
    LocationStreamStatus.PARENT_KEY_VIOLATION, LocationStreamStatus.RELATIONSHIP_LINK_FAILURE,
    LocationStreamStatus.JOBS_PRESENT_DETAILS_ABSENT, LocationStreamStatus.JOB_DETAIL_COUNT_MISMATCH,
    LocationStreamStatus.SCHEDULED_TIME_ABSENT, LocationStreamStatus.CONTINUITY_UNASSESSABLE,
    LocationStreamStatus.SCOPE_UNASSIGNABLE, LocationStreamStatus.PARENT_DETAIL_SCOPE_MISMATCH,
})


@dataclass(frozen=True, slots=True)
class LocationStreamInvestigationReport:
    """Categorical result of a stream investigation (enums and booleans only).

    ``None`` means a stage was not evaluated (input not supplied, or an
    earlier prerequisite failed). ``failing_stages`` lists every failing stage
    in pipeline order; ``earliest_failing_stage`` is its first entry.
    """

    status: LocationStreamStatus
    earliest_failing_stage: PipelineStage | None
    failing_stages: tuple[PipelineStage, ...]
    target_configured: bool
    present_in_raw_source: bool | None
    present_after_ingestion: bool | None
    present_after_cleaning: bool | None
    matched_via_authoritative_alias: bool | None
    unverified_representation_variant: bool | None
    stream_continuity: StreamContinuity
    schedule_available: bool
    time_coverage: TimeCoverageStatus
    identifier_dtypes_valid: bool | None
    parent_keys_valid: bool | None
    target_details_present: bool | None
    target_details_all_linked: bool | None
    reconciliation_passes: bool | None
    relationship_passes: bool | None
    event_accounting: StreamEventAccounting | None = None
    #: Linked rows concerning the stream agree with their parent's scope
    #: (``None`` = not evaluated: no scope invariant or prerequisites failed).
    parent_detail_scope_agrees: bool | None = None

    @property
    def is_healthy(self) -> bool:
        return self.status is LocationStreamStatus.STREAM_PRESENT_AND_HEALTHY

    @property
    def repository_fix_required(self) -> bool:
        """The primary failure points at repository code or configuration."""
        return self.status in _REPOSITORY_STATUSES

    @property
    def upstream_issue_indicated(self) -> bool:
        """The primary failure is in the source data or its collection."""
        return self.status in _UPSTREAM_STATUSES

    @property
    def authoritative_mapping_required(self) -> bool:
        """A possible alias exists but needs authoritative confirmation."""
        return self.status is LocationStreamStatus.UNVERIFIED_ALIAS


class LocationStreamError(Exception):
    """Strict validation found the stream not present and healthy.

    The message names the status and stage only; the report is on ``report``.
    """

    def __init__(self, report: LocationStreamInvestigationReport) -> None:
        stage = report.earliest_failing_stage.value if report.earliest_failing_stage else "none"
        super().__init__(f"Expected location stream failed at stage '{stage}': {report.status.value}.")
        self.report = report


# ------------------------------------------------------------------- public API


def resolve_expected_location(
    target: tuple[str, ...],
    coverage: LocationCoverageDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
) -> tuple[str, ...]:
    """Return ``target`` if it is a configured expected key, else raise.

    Exact lookup only: no fuzzy matching, no substitution, no data access.
    The error never lists configured or observed locations.

    Raises:
        LocationCoverageConfigurationError: The contract is unconfigured, or
            ``target`` is malformed or not an expected key.
    """
    coverage = project_default(coverage, "EXPECTED_LOCATION_COVERAGE")
    if not isinstance(coverage, LocationCoverageDefinition):
        raise TypeError(f"expected a LocationCoverageDefinition, got {type(coverage).__name__}")
    if not coverage.is_configured:
        raise LocationCoverageConfigurationError("No expected-location contract is configured.")
    if (not isinstance(target, tuple) or len(target) != len(coverage.location_columns)
            or not all(isinstance(v, str) for v in target)):
        raise LocationCoverageConfigurationError("The target is not a well-formed location key.")
    if target not in coverage.expected_locations:
        raise LocationCoverageConfigurationError("The target is not a configured expected location.")
    return target


def investigate_location_stream(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    target: tuple[str, ...],
    *,
    coverage: LocationCoverageDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
    loaded: RawDatasets | None = None,
    raw_source: str | Path | None = None,
    schedule: CollectionScheduleDefinition | None = COLLECTION_SCHEDULE,
    temporal: TemporalReconciliationDefinition = TEMPORAL_RECONCILIATION,
    capture_exclusions: object = None,
) -> LocationStreamInvestigationReport:
    """Trace ``target`` through the pipeline; return a categorical report.

    Args:
        jobs, cars: The *cleaned* frames (after blank-row removal).
        target: An expected location key (see :func:`resolve_expected_location`).
        coverage, relationship: Central definitions (defaults: project ones).
        loaded: Optional frames as loaded, before cleaning, to localise loss.
        raw_source: Optional raw CSV of the coverage dataset, scanned
            header-aware with the default CSV dialect, to localise loss.
        schedule: Authoritative schedule, if one exists (default: project's).
        temporal: Temporal contract that parses the schedule's observed
            timestamp field (default: project's).

    Raises:
        TypeError: Invalid argument types.
        LocationCoverageConfigurationError: Configured columns are absent,
            the schedule/contract datasets are not part of the relationship,
            or the schedule's timestamp is not a timestamp field of the
            temporal contract.
    """
    coverage = project_default(coverage, "EXPECTED_LOCATION_COVERAGE")
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be pandas DataFrames")
    if not isinstance(coverage, LocationCoverageDefinition):
        raise TypeError(f"expected a LocationCoverageDefinition, got {type(coverage).__name__}")
    if loaded is not None and not isinstance(loaded, RawDatasets):
        raise TypeError("loaded must be RawDatasets or None")
    frames = {relationship.parent: jobs, relationship.detail: cars}
    if coverage.dataset not in frames:
        raise LocationCoverageConfigurationError("coverage dataset is not part of the relationship")
    schedule_field = _schedule_field(schedule, temporal, frames)
    check_scope_contract(coverage, relationship)
    frame = frames[coverage.dataset]
    columns = coverage.location_columns
    absent = tuple(c for c in (*columns, *coverage.stream_scope_columns) if c not in frame.columns)
    if absent:
        raise LocationCoverageConfigurationError(
            f"The '{coverage.dataset}' frame lacks {len(absent)} location column(s).", absent
        )

    try:
        resolve_expected_location(target, coverage)
        configured = True
    except LocationCoverageConfigurationError:
        configured = False
    if not configured:
        return _report(LocationStreamStatus.EXPECTATION_NOT_CONFIGURED, [PipelineStage.CONFIGURATION],
                       target_configured=False, schedule_available=schedule is not None)

    keys = coverage.match_keys(target)
    exact_mask = _match(frame, columns, (target,))
    mask = _match(frame, columns, keys)
    in_clean = bool(mask.any())
    via_alias = bool(in_clean and not exact_mask.any())
    in_loaded = (bool(_match(getattr(loaded, coverage.dataset.value), columns, keys).any())
                 if loaded is not None else None)
    in_raw = _raw_contains(Path(raw_source), columns, keys) if raw_source is not None else None
    variant = None if in_clean else _variant_present(frame, columns, target)

    failing: list[PipelineStage] = []
    status: LocationStreamStatus | None = None

    def fail(stage: PipelineStage, outcome: LocationStreamStatus) -> None:
        nonlocal status
        failing.append(stage)
        if status is None:
            status = outcome  # used only on the early-return (target absent) path

    if not in_clean:
        if in_raw and in_loaded is False:
            fail(PipelineStage.INGESTION, LocationStreamStatus.INGESTION_EXCLUSION)
        elif in_loaded or (in_raw and in_loaded is None):
            stage = PipelineStage.CLEANING if in_loaded else PipelineStage.INGESTION
            fail(stage, LocationStreamStatus.CLEANING_EXCLUSION if in_loaded
                 else LocationStreamStatus.INGESTION_EXCLUSION)
        elif variant:
            fail(PipelineStage.LOCATION_MATCHING, LocationStreamStatus.UNVERIFIED_ALIAS)
        else:
            fail(PipelineStage.RAW_SOURCE, LocationStreamStatus.RAW_STREAM_ABSENT)
        return _report(status, failing, target_configured=True, present_in_raw_source=in_raw,
                       present_after_ingestion=in_loaded, present_after_cleaning=False,
                       matched_via_authoritative_alias=False, unverified_representation_variant=variant,
                       schedule_available=schedule is not None,
                       time_coverage=(TimeCoverageStatus.NEVER_PRESENT if schedule is not None
                                      else TimeCoverageStatus.NOT_ASSESSED))

    # --- source continuity across the scope's collection events
    continuity, accounting = _continuity(jobs, frame, mask, coverage, relationship,
                                         _governed_excluded(capture_exclusions, jobs, target))
    continuity_failure = _CONTINUITY_FAILURES.get(continuity)
    if continuity_failure is not None:
        fail(PipelineStage.SOURCE_CONTINUITY, continuity_failure)

    # --- relationship preconditions
    dtypes_ok = all(is_identifier_dtype(jobs[c].dtype) for c in relationship.parent_key_columns
                    if c in jobs.columns) and all(
        is_identifier_dtype(cars[c].dtype) for c in relationship.detail_key_columns if c in cars.columns)
    if not dtypes_ok:
        fail(PipelineStage.IDENTIFIER_TYPES, LocationStreamStatus.IDENTIFIER_TYPE_MISMATCH)
    keys_ok = assess_unique_key(jobs, relationship.parent_definition).is_valid
    if not keys_ok:
        fail(PipelineStage.PARENT_KEYS, LocationStreamStatus.PARENT_KEY_VIOLATION)

    # --- linked rows concerning the stream must carry their parent's scope
    scope_agrees = _scope_agrees(jobs, frame, mask, coverage, relationship) if dtypes_ok else None
    if scope_agrees is False:
        fail(PipelineStage.SCOPE_AGREEMENT, LocationStreamStatus.PARENT_DETAIL_SCOPE_MISMATCH)

    # --- target jobs and details
    target_jobs, target_details, details_present, all_linked = _target_rows(
        jobs, cars, mask, coverage, relationship)
    time_status = _time_coverage(schedule, schedule_field, temporal, frames, mask, coverage, target_jobs)
    time_failure = _TIME_FAILURES.get(time_status)
    if time_failure is not None:
        fail(PipelineStage.TIME_COVERAGE, time_failure)

    reconciled = related = None
    if dtypes_ok and keys_ok:
        if not all_linked:
            fail(PipelineStage.RELATIONSHIP, LocationStreamStatus.RELATIONSHIP_LINK_FAILURE)
        elif not details_present:
            fail(PipelineStage.DETAIL_PRESENCE, LocationStreamStatus.JOBS_PRESENT_DETAILS_ABSENT)
        if len(target_jobs):
            related = assess_one_to_many_join(target_jobs, target_details, relationship).is_valid and all_linked
            reconciled = assess_job_detail_reconciliation(target_jobs, target_details, relationship).is_reconciled
            if not reconciled:
                fail(PipelineStage.RECONCILIATION, LocationStreamStatus.JOB_DETAIL_COUNT_MISMATCH)
        else:
            related = False

    failing.sort(key=list(PipelineStage).index)        # pipeline order
    by_stage = {**_STATUS_BY_STAGE, PipelineStage.TIME_COVERAGE: time_failure,
                PipelineStage.SOURCE_CONTINUITY: continuity_failure}
    status = by_stage[failing[0]] if failing else LocationStreamStatus.STREAM_PRESENT_AND_HEALTHY
    return _report(
        status, failing,
        target_configured=True, present_in_raw_source=in_raw, present_after_ingestion=in_loaded,
        present_after_cleaning=True, matched_via_authoritative_alias=via_alias,
        unverified_representation_variant=None, stream_continuity=continuity,
        schedule_available=schedule is not None, time_coverage=time_status,
        identifier_dtypes_valid=dtypes_ok, parent_keys_valid=keys_ok,
        target_details_present=details_present, target_details_all_linked=all_linked,
        reconciliation_passes=reconciled, relationship_passes=related, event_accounting=accounting,
        parent_detail_scope_agrees=scope_agrees,
    )


def validate_location_stream(
    jobs: pd.DataFrame, cars: pd.DataFrame, target: tuple[str, ...], **kwargs: object,
) -> LocationStreamInvestigationReport:
    """Investigate; return the report if healthy, else raise :class:`LocationStreamError`."""
    report = investigate_location_stream(jobs, cars, target, **kwargs)  # type: ignore[arg-type]
    if not report.is_healthy:
        raise LocationStreamError(report)
    return report


# ------------------------------------------------------------ all expected streams


class ExpectedStreamBlocker(StrEnum):
    """Why the expected-stream population is not proven healthy (fixed order)."""

    EXPECTED_STREAM_REPORT_MISSING = "expected_stream_report_missing"
    DUPLICATE_STREAM_REPORT = "duplicate_stream_report"
    UNEXPECTED_STREAM_REPORT = "unexpected_stream_report"
    EXPECTED_STREAM_REPORT_UNAVAILABLE = "expected_stream_report_unavailable"
    STREAM_CONTINUITY_PARTIAL = "stream_continuity_partial"
    STREAM_CONTINUITY_UNASSESSABLE = "stream_continuity_unassessable"
    STREAM_SCOPE_UNASSIGNABLE = "stream_scope_unassignable"
    STREAM_SCOPE_MISMATCH = "stream_parent_detail_scope_mismatch"
    EXPECTED_STREAM_UNHEALTHY = "expected_stream_unhealthy"


@dataclass(frozen=True, slots=True)
class ExpectedStreamResult:
    """One stream report paired with the configured key it was produced for (``None`` = unavailable)."""

    target: tuple[str, ...]
    report: LocationStreamInvestigationReport | None

    def __post_init__(self) -> None:
        if not isinstance(self.target, tuple) or not all(isinstance(v, str) for v in self.target):
            raise TypeError("target must be a configured location key (tuple of strings)")
        if self.report is not None and not isinstance(self.report, LocationStreamInvestigationReport):
            raise TypeError("report must be a LocationStreamInvestigationReport or None")


@dataclass(frozen=True, slots=True)
class ExpectedLocationStreamsReport:
    """Stream results validated against a coverage contract's expected population.

    Every property is derived from ``coverage.expected_locations`` and
    ``results`` on access, so no construction path can claim a population it
    does not hold: omitted, duplicated, unexpected or unavailable results
    always appear in :attr:`blocking_reasons`. ``results`` are stored in
    contract order (then any unexpected keys, ordered), whatever the input
    order. Reports hold no source values; configured keys are repository
    configuration and are never put into messages.
    """

    coverage: LocationCoverageDefinition
    results: tuple[ExpectedStreamResult, ...]
    #: The collection schedule every stream's time coverage was assessed against
    #: (``None`` = none; a hand-built aggregate records none, which never matches an available schedule).
    schedule: CollectionScheduleDefinition | None = None

    def __post_init__(self) -> None:
        if self.schedule is not None and not isinstance(self.schedule, CollectionScheduleDefinition):
            raise TypeError("schedule must be a CollectionScheduleDefinition or None")
        if not isinstance(self.coverage, LocationCoverageDefinition):
            raise TypeError(f"expected a LocationCoverageDefinition, got {type(self.coverage).__name__}")
        if not self.coverage.is_configured:
            raise LocationCoverageConfigurationError(
                "No expected-location contract is configured; the expected streams cannot be assessed.")
        results = tuple(self.results) if isinstance(self.results, (tuple, list)) else None
        if results is None or not all(isinstance(r, ExpectedStreamResult) for r in results):
            raise TypeError("results must be a tuple of ExpectedStreamResult")
        expected = list(self.coverage.expected_locations)
        position = {key: i for i, key in enumerate(expected)}
        ordered = sorted(results, key=lambda r: (position.get(r.target, len(expected)), repr(r.target)))
        object.__setattr__(self, "results", tuple(ordered))

    @classmethod
    def from_reports(
        cls,
        coverage: LocationCoverageDefinition,
        reports: Iterable[tuple[tuple[str, ...], LocationStreamInvestigationReport | None]]
        | Mapping[tuple[str, ...], LocationStreamInvestigationReport | None],
    ) -> ExpectedLocationStreamsReport:
        """Build from ``(configured key, report)`` pairs (or a mapping); validation is by property."""
        items = reports.items() if isinstance(reports, Mapping) else reports
        return cls(coverage=coverage, results=tuple(ExpectedStreamResult(tuple(k), r) for k, r in items))

    # ----------------------------------------------------------- population
    @property
    def expected_stream_count(self) -> int:
        return len(self.coverage.expected_locations)

    @property
    def assessed_stream_count(self) -> int:
        return len(self.results)

    @property
    def _counts(self) -> Counter:
        return Counter(r.target for r in self.results)

    @property
    def missing_stream_count(self) -> int:
        counts = self._counts
        return sum(1 for key in self.coverage.expected_locations if counts[key] == 0)

    @property
    def duplicate_stream_count(self) -> int:
        counts = self._counts
        return sum(1 for key in self.coverage.expected_locations if counts[key] > 1)

    @property
    def unexpected_stream_count(self) -> int:
        expected = set(self.coverage.expected_locations)
        return sum(1 for r in self.results if r.target not in expected)

    @property
    def all_expected_assessed(self) -> bool:
        """Every configured expected key has at least one result."""
        return self.missing_stream_count == 0

    @property
    def assessed_exactly_once(self) -> bool:
        """The results are exactly the configured population, each key once."""
        return (self.missing_stream_count == 0 and self.duplicate_stream_count == 0
                and self.unexpected_stream_count == 0)

    @property
    def reports(self) -> Mapping[tuple[str, ...], LocationStreamInvestigationReport | None]:
        """Read-only mapping of configured key -> report (keys assessed exactly once)."""
        counts = self._counts
        return MappingProxyType({r.target: r.report for r in self.results
                                 if r.target in set(self.coverage.expected_locations) and counts[r.target] == 1})

    @property
    def all_expected_streams_healthy(self) -> bool:
        return self.assessed_exactly_once and all(
            r.report is not None and r.report.is_healthy for r in self.results)

    @property
    def all_continuity_complete(self) -> bool:
        return self.assessed_exactly_once and all(
            r.report is not None and r.report.stream_continuity is StreamContinuity.COMPLETE
            for r in self.results)

    @property
    def is_valid(self) -> bool:
        return not self.blocking_reasons

    @property
    def blocking_reasons(self) -> tuple[ExpectedStreamBlocker, ...]:
        """Every applicable blocker, in the fixed enum order."""
        B = ExpectedStreamBlocker
        found: set[ExpectedStreamBlocker] = set()
        if self.missing_stream_count:
            found.add(B.EXPECTED_STREAM_REPORT_MISSING)
        if self.duplicate_stream_count:
            found.add(B.DUPLICATE_STREAM_REPORT)
        if self.unexpected_stream_count:
            found.add(B.UNEXPECTED_STREAM_REPORT)
        expected = set(self.coverage.expected_locations)
        for result in self.results:
            if result.target not in expected:
                continue
            report = result.report
            if report is None:
                found.add(B.EXPECTED_STREAM_REPORT_UNAVAILABLE)
                continue
            if report.stream_continuity is StreamContinuity.PARTIAL:
                found.add(B.STREAM_CONTINUITY_PARTIAL)
            elif report.stream_continuity is StreamContinuity.SCOPE_UNASSIGNABLE:
                found.add(B.STREAM_SCOPE_UNASSIGNABLE)
            elif report.stream_continuity is not StreamContinuity.COMPLETE:
                found.add(B.STREAM_CONTINUITY_UNASSESSABLE)
            if report.parent_detail_scope_agrees is False:
                found.add(B.STREAM_SCOPE_MISMATCH)
            if not report.is_healthy:
                found.add(B.EXPECTED_STREAM_UNHEALTHY)
        return tuple(b for b in B if b in found)


def assess_expected_location_streams(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    *,
    coverage: LocationCoverageDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
    loaded: RawDatasets | None = None,
    schedule: CollectionScheduleDefinition | None = COLLECTION_SCHEDULE,
    temporal: TemporalReconciliationDefinition = TEMPORAL_RECONCILIATION,
    capture_exclusions: object = None,
) -> ExpectedLocationStreamsReport:
    """Investigate every configured expected stream once; return the validated aggregate.

    The population comes only from ``coverage.expected_locations`` (never from
    the data), in contract order. Inputs are not modified.

    Raises:
        TypeError: Invalid argument types.
        LocationCoverageConfigurationError: The contract is unconfigured or
            its columns are absent (as for :func:`investigate_location_stream`).
    """
    coverage = project_default(coverage, "EXPECTED_LOCATION_COVERAGE")
    if not isinstance(coverage, LocationCoverageDefinition):
        raise TypeError(f"expected a LocationCoverageDefinition, got {type(coverage).__name__}")
    if not coverage.is_configured:
        raise LocationCoverageConfigurationError(
            "No expected-location contract is configured; the expected streams cannot be assessed.")
    results = tuple(
        ExpectedStreamResult(key, investigate_location_stream(
            jobs, cars, key, coverage=coverage, relationship=relationship, loaded=loaded,
            schedule=schedule, temporal=temporal, capture_exclusions=capture_exclusions))
        for key in coverage.expected_locations
    )
    aggregate = ExpectedLocationStreamsReport(coverage=coverage, results=results, schedule=schedule)
    assert aggregate.assessed_exactly_once            # by construction; guards later refactors
    return aggregate


# ------------------------------------------------ schedule and scheduled coverage


class CollectionScheduleStatus(StrEnum):
    """Whether an authoritative collection schedule can be used (fixed order)."""

    AVAILABLE = "available"          # configured and valid against the contracts
    UNAVAILABLE = "unavailable"      # none configured: temporal completeness cannot be proven
    INVALID = "invalid"              # configured but structurally unusable


@dataclass(frozen=True, slots=True)
class CollectionScheduleAssessment:
    """Typed result of validating the configured collection schedule.

    Absence is never read as "no scheduled times were required": it is
    ``UNAVAILABLE`` and blocks pricing. ``schedule`` is kept only when valid.
    """

    status: CollectionScheduleStatus
    schedule: CollectionScheduleDefinition | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, CollectionScheduleStatus):
            raise TypeError("status must be a CollectionScheduleStatus")
        if (self.status is CollectionScheduleStatus.AVAILABLE) != isinstance(self.schedule,
                                                                           CollectionScheduleDefinition):
            raise ValueError("only an available assessment carries (and must carry) a schedule")

    @property
    def available(self) -> bool:
        return self.status is CollectionScheduleStatus.AVAILABLE


def assess_collection_schedule(
    schedule: object = COLLECTION_SCHEDULE,
    *,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
    temporal: TemporalReconciliationDefinition = TEMPORAL_RECONCILIATION,
) -> CollectionScheduleAssessment:
    """Validate the authoritative schedule (default: the project's ``COLLECTION_SCHEDULE``).

    ``None`` is ``UNAVAILABLE``. Anything else must be a
    :class:`~ql2_sixt_canada_analysis.schemas.CollectionScheduleDefinition`
    on a dataset of the relationship whose timestamp is a timestamp field of
    the temporal contract and whose scheduled instants resolve; otherwise -
    including objects whose construction checks were bypassed - ``INVALID``.
    Never raises for a bad schedule.
    """
    S = CollectionScheduleStatus
    if schedule is None:
        return CollectionScheduleAssessment(S.UNAVAILABLE)
    if not isinstance(schedule, CollectionScheduleDefinition):
        return CollectionScheduleAssessment(S.INVALID)
    try:
        _schedule_field(schedule, temporal, {relationship.parent: None, relationship.detail: None})
        instants = schedule.expected_instants
        valid = len(instants) > 0 and not instants.isna().any() and not instants.has_duplicates
    except Exception:  # noqa: BLE001 - any failure to use the schedule is a typed INVALID, never a pass
        valid = False
    return CollectionScheduleAssessment(S.AVAILABLE, schedule) if valid else CollectionScheduleAssessment(S.INVALID)


class ScheduledCoverageBlocker(StrEnum):
    """Why scheduled-time coverage is not proven complete (fixed order; values avoid column names)."""

    COLLECTION_SCHEDULE_UNAVAILABLE = "collection_schedule_unavailable"
    COLLECTION_SCHEDULE_INVALID = "collection_schedule_invalid"
    STREAM_COVERAGE_ASSESSMENT_MISSING = "scheduled_coverage_streams_unavailable"
    STREAM_POPULATION_NOT_EXACT = "scheduled_coverage_streams_not_exact"
    SCHEDULE_NOT_APPLIED = "scheduled_coverage_schedule_not_applied"
    SCHEDULED_COVERAGE_INCOMPLETE = "scheduled_coverage_incomplete"


#: The only accepted per-stream coverage (an allowlist: every other status fails).
_COMPLETE_COVERAGE = frozenset({TimeCoverageStatus.COMPLETE})


@dataclass(frozen=True, slots=True)
class StreamTimeCoverage:
    """Scheduled-time coverage of one configured expected stream (``None`` = no single report)."""

    target: tuple[str, ...]
    time_coverage: TimeCoverageStatus | None

    @property
    def complete(self) -> bool:
        """Only the typed ``COMPLETE`` member passes (a look-alike string or unknown value fails)."""
        return isinstance(self.time_coverage, TimeCoverageStatus) and any(
            self.time_coverage is accepted for accepted in _COMPLETE_COVERAGE)


@dataclass(frozen=True, slots=True)
class ScheduledCoverageReport:
    """Scheduled-time coverage for the exact expected-stream population.

    Every property is derived from the schedule assessment and the
    all-expected-stream aggregate on access. Coverage is complete only when
    the schedule is available, the aggregate holds exactly one report per
    configured expected stream, every report was assessed against that same
    schedule, and every expected stream's ``time_coverage`` is ``COMPLETE``.
    ``NOT_ASSESSED``, ``NEVER_PRESENT``, ``PARTIAL``, ``UNASSESSABLE``, a
    missing report or any future status fail; stream health is not consulted.
    """

    schedule_assessment: CollectionScheduleAssessment
    expected_streams: ExpectedLocationStreamsReport | None

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_assessment, CollectionScheduleAssessment):
            raise TypeError("schedule_assessment must be a CollectionScheduleAssessment")
        if self.expected_streams is not None and not isinstance(self.expected_streams, ExpectedLocationStreamsReport):
            raise TypeError("expected_streams must be an ExpectedLocationStreamsReport or None")

    @property
    def stream_coverage(self) -> tuple[StreamTimeCoverage, ...]:
        """One entry per configured expected stream, in contract order."""
        if self.expected_streams is None:
            return ()
        reports = self.expected_streams.reports       # keys assessed exactly once only
        return tuple(
            StreamTimeCoverage(key, reports[key].time_coverage if reports.get(key) is not None else None)
            for key in self.expected_streams.coverage.expected_locations)

    @property
    def all_streams_complete(self) -> bool:
        return not self.blocking_reasons

    @property
    def is_valid(self) -> bool:
        return not self.blocking_reasons

    @property
    def blocking_reasons(self) -> tuple[ScheduledCoverageBlocker, ...]:
        B = ScheduledCoverageBlocker
        found: set[ScheduledCoverageBlocker] = set()
        status = self.schedule_assessment.status
        if status is CollectionScheduleStatus.UNAVAILABLE:
            found.add(B.COLLECTION_SCHEDULE_UNAVAILABLE)
        elif status is not CollectionScheduleStatus.AVAILABLE:
            found.add(B.COLLECTION_SCHEDULE_INVALID)
        streams = self.expected_streams
        if streams is None:
            found.add(B.STREAM_COVERAGE_ASSESSMENT_MISSING)
        else:
            if not streams.assessed_exactly_once:
                found.add(B.STREAM_POPULATION_NOT_EXACT)
            if self.schedule_assessment.available and streams.schedule != self.schedule_assessment.schedule:
                found.add(B.SCHEDULE_NOT_APPLIED)
            if not all(entry.complete for entry in self.stream_coverage):
                found.add(B.SCHEDULED_COVERAGE_INCOMPLETE)
        return tuple(b for b in B if b in found)


def assess_scheduled_time_coverage(
    schedule: CollectionScheduleAssessment,
    expected_streams: ExpectedLocationStreamsReport | None,
) -> ScheduledCoverageReport:
    """Combine the schedule assessment with the all-expected-stream aggregate (validated by property)."""
    return ScheduledCoverageReport(schedule_assessment=schedule, expected_streams=expected_streams)


# ---------------------------------------------------------------------- helpers


#: Every scheduled-coverage outcome other than COMPLETE / NOT_ASSESSED fails.
_TIME_FAILURES = {
    TimeCoverageStatus.NEVER_PRESENT: LocationStreamStatus.SCHEDULED_TIME_ABSENT,
    TimeCoverageStatus.PARTIAL: LocationStreamStatus.RAW_STREAM_PARTIAL,
    TimeCoverageStatus.UNASSESSABLE: LocationStreamStatus.SCHEDULED_TIME_UNASSESSABLE,
}

_CONTINUITY_FAILURES = {
    StreamContinuity.PARTIAL: LocationStreamStatus.RAW_STREAM_PARTIAL,
    StreamContinuity.UNASSESSABLE: LocationStreamStatus.CONTINUITY_UNASSESSABLE,
    StreamContinuity.SCOPE_UNASSIGNABLE: LocationStreamStatus.SCOPE_UNASSIGNABLE,
}

_STATUS_BY_STAGE = {
    PipelineStage.SOURCE_CONTINUITY: LocationStreamStatus.RAW_STREAM_PARTIAL,
    PipelineStage.IDENTIFIER_TYPES: LocationStreamStatus.IDENTIFIER_TYPE_MISMATCH,
    PipelineStage.PARENT_KEYS: LocationStreamStatus.PARENT_KEY_VIOLATION,
    PipelineStage.SCOPE_AGREEMENT: LocationStreamStatus.PARENT_DETAIL_SCOPE_MISMATCH,
    PipelineStage.RELATIONSHIP: LocationStreamStatus.RELATIONSHIP_LINK_FAILURE,
    PipelineStage.DETAIL_PRESENCE: LocationStreamStatus.JOBS_PRESENT_DETAILS_ABSENT,
    PipelineStage.RECONCILIATION: LocationStreamStatus.JOB_DETAIL_COUNT_MISMATCH,
}


def _report(status: LocationStreamStatus | None, failing: list[PipelineStage], **fields: object
            ) -> LocationStreamInvestigationReport:
    defaults: dict[str, object] = dict(
        target_configured=False, present_in_raw_source=None, present_after_ingestion=None,
        present_after_cleaning=None, matched_via_authoritative_alias=None,
        unverified_representation_variant=None, stream_continuity=StreamContinuity.NOT_APPLICABLE,
        schedule_available=False, time_coverage=TimeCoverageStatus.NOT_ASSESSED,
        identifier_dtypes_valid=None, parent_keys_valid=None, target_details_present=None,
        target_details_all_linked=None, reconciliation_passes=None, relationship_passes=None,
    )
    defaults.update(fields)
    stages = tuple(dict.fromkeys(failing))
    return LocationStreamInvestigationReport(
        status=status or LocationStreamStatus.STREAM_PRESENT_AND_HEALTHY,
        earliest_failing_stage=stages[0] if stages else None,
        failing_stages=stages, **defaults,  # type: ignore[arg-type]
    )


def _match(frame: pd.DataFrame, columns: tuple[str, ...], keys: tuple[tuple[str, ...], ...]) -> pd.Series:
    """Rows whose complete location tuple equals one of ``keys`` exactly."""
    names = [f"component_{i}" for i in range(len(columns))]
    observed = pd.MultiIndex.from_frame(frame.loc[:, list(columns)].astype(object), names=names)
    wanted = pd.MultiIndex.from_tuples(list(keys), names=names)
    return pd.Series(observed.isin(wanted), index=frame.index)


def _normalise(series: pd.Series) -> pd.Series:
    """Diagnostic-only folding (case, surrounding space, punctuation); never applied to data."""
    return series.astype("string").str.casefold().str.replace(r"[^0-9a-z]+", "", regex=True)


def _variant_present(frame: pd.DataFrame, columns: tuple[str, ...], target: tuple[str, ...]) -> bool:
    """Does a case/space/punctuation/component-order variant of ``target`` exist?"""
    names = [f"component_{i}" for i in range(len(columns))]
    folded = pd.DataFrame({n: _normalise(frame[c]) for n, c in zip(names, columns, strict=True)})
    folded = folded.dropna()
    observed = pd.MultiIndex.from_frame(folded, names=names)
    target_folded = tuple(_normalise(pd.Series([v])).iloc[0] for v in target)
    candidates = set(itertools.permutations(target_folded))
    return bool(observed.isin(pd.MultiIndex.from_tuples(list(candidates), names=names)).any())


def _raw_contains(path: Path, columns: tuple[str, ...], keys: tuple[tuple[str, ...], ...]) -> bool:
    """Header-aware streaming scan of a raw CSV for an exact key (nothing kept or printed)."""
    wanted = set(keys)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(columns) <= set(reader.fieldnames):
            raise LocationCoverageConfigurationError("The raw source lacks the location column(s).")
        return any(tuple(row[c] for c in columns) in wanted for row in reader)


def _governed_excluded(capture_exclusions: object, jobs: pd.DataFrame, target: tuple[str, ...]) -> np.ndarray | None:
    """Jobs whose whole capture is a governed ``INCOMPLETE_PARENT_CAPTURE`` for ``target`` (``None`` = none given)."""
    if capture_exclusions is None:
        return None
    from ql2_sixt_canada_analysis.collection_schedule import CaptureExclusionSet

    if not isinstance(capture_exclusions, CaptureExclusionSet):
        raise TypeError("capture_exclusions must be a CaptureExclusionSet or None")
    return capture_exclusions.parent_mask(jobs, tuple(target))


def _continuity(jobs: pd.DataFrame, frame: pd.DataFrame, mask: pd.Series, coverage: LocationCoverageDefinition,
                relationship: JobDetailRelationshipDefinition, governed_excluded: np.ndarray | None = None,
                ) -> tuple[StreamContinuity, StreamEventAccounting | None]:
    """Presence of the target across every in-scope *job* (zero-detail jobs included).

    In-scope jobs whose whole capture is a governed parent-capture exclusion
    for the target are counted separately (``governed_excluded_jobs``) and are
    neither a presence nor a gap; every other in-scope job is classified.
    """
    scope = coverage.stream_scope_columns
    if not scope or coverage.dataset != relationship.detail:
        return StreamContinuity.NOT_APPLICABLE, None
    parent_scope = coverage.parent_scope_columns
    if not parent_scope:
        return StreamContinuity.UNASSESSABLE, None           # no job-level denominator available
    absent = tuple(c for c in parent_scope if c not in jobs.columns)
    if absent:
        raise LocationCoverageConfigurationError(
            f"The '{relationship.parent}' frame lacks {len(absent)} scope column(s).", absent)
    parent_keys, detail_keys = list(relationship.parent_key_columns), list(relationship.detail_key_columns)
    names = [f"scope_{i}" for i in range(len(scope))]
    target_scope = pd.MultiIndex.from_frame(
        frame.loc[mask.to_numpy(), list(scope)].astype(object).dropna().drop_duplicates(), names=names)
    job_scope = jobs.loc[:, list(parent_scope)].astype(object)
    scope_missing = unassignable_scope_mask(jobs, parent_scope)   # the one definition (blank/padded included)
    in_scope = pd.MultiIndex.from_frame(job_scope, names=names).isin(target_scope) & ~scope_missing

    keys = [f"key_{i}" for i in range(len(parent_keys))]
    job_index = pd.MultiIndex.from_frame(jobs.loc[:, parent_keys].astype(object), names=keys)
    detail_frame = frame.loc[:, detail_keys].astype(object)
    complete = detail_frame.notna().all(axis=1).to_numpy()
    all_events = pd.MultiIndex.from_frame(detail_frame.loc[complete], names=keys)
    target_events = pd.MultiIndex.from_frame(detail_frame.loc[complete & mask.to_numpy()], names=keys)
    has_details = job_index.isin(all_events)
    has_target = job_index.isin(target_events)

    declared_zero = np.ones(len(jobs), dtype=bool)
    for column in relationship.expected_detail_count_columns:
        if column not in jobs.columns:
            declared_zero[:] = False
            break
        category, expected = _classify_expected_counts(jobs[column])
        declared_zero &= (category == _VALID) & (expected == 0)

    governed = (np.zeros(len(jobs), dtype=bool) if governed_excluded is None
                else np.asarray(governed_excluded, dtype=bool)) & in_scope
    counted = in_scope & ~governed
    accounting = StreamEventAccounting(
        total_jobs=len(jobs),
        scope_excluded_jobs=int((~in_scope & ~scope_missing).sum()),
        scope_unassignable_jobs=int(scope_missing.sum()),
        in_scope_jobs=int(in_scope.sum()),
        jobs_with_target_details=int((counted & has_target).sum()),
        jobs_with_other_details_only=int((counted & has_details & ~has_target).sum()),
        zero_offer_jobs=int((counted & ~has_details & declared_zero).sum()),
        missing_detail_jobs=int((counted & ~has_details & ~declared_zero).sum()),
        governed_excluded_jobs=int(governed.sum()),
    )
    if accounting.scope_unassignable_jobs:
        status = StreamContinuity.SCOPE_UNASSIGNABLE          # denominator itself unproven
    elif accounting.jobs_with_other_details_only:
        status = StreamContinuity.PARTIAL                     # proven absence in an observed event
    elif accounting.in_scope_jobs == 0 or accounting.zero_detail_jobs:
        status = StreamContinuity.UNASSESSABLE                # branch presence unprovable
    else:
        status = StreamContinuity.COMPLETE
    return status, accounting


def _scope_agrees(jobs: pd.DataFrame, frame: pd.DataFrame, mask: pd.Series, coverage: LocationCoverageDefinition,
                  relationship: JobDetailRelationshipDefinition) -> bool | None:
    """Do linked rows concerning the target agree with their parent's scope? (``None`` = not applicable).

    Concerning the target: a target row, a row whose own scope is the
    target's, or a row linked to a job of the target's scope. Uses the shared
    :func:`~ql2_sixt_canada_analysis.city_integrity.scope_agreement` rule.
    """
    if not relationship.scope_agreement_columns or coverage.dataset != relationship.detail:
        return None
    agreement = scope_agreement(jobs, frame, relationship)
    if not agreement.detail_mismatch.any():
        return True
    names = [f"scope_{i}" for i in range(len(relationship.scope_detail_columns))]
    target_rows = mask.to_numpy()
    detail_scope = pd.MultiIndex.from_frame(
        frame.loc[:, list(relationship.scope_detail_columns)].astype(object), names=names)
    target_scope = detail_scope[target_rows].unique()
    parent_scope = pd.MultiIndex.from_frame(
        jobs.loc[:, list(relationship.scope_parent_columns)].astype(object), names=names)
    concerned_rows = target_rows | detail_scope.isin(target_scope)
    concerned_jobs = parent_scope.isin(target_scope)
    return not (bool((agreement.detail_mismatch & concerned_rows).any())
                or bool((agreement.parent_mismatch & concerned_jobs).any()))


def _target_rows(jobs: pd.DataFrame, cars: pd.DataFrame, mask: pd.Series,
                 coverage: LocationCoverageDefinition, relationship: JobDetailRelationshipDefinition,
                 ) -> tuple[pd.DataFrame, pd.DataFrame, bool, bool]:
    """(target jobs, their detail rows, details present?, all target details linked?)."""
    parent_keys, detail_keys = list(relationship.parent_key_columns), list(relationship.detail_key_columns)
    parent_index = pd.MultiIndex.from_frame(jobs.loc[:, parent_keys])
    if coverage.dataset == relationship.parent:
        target_jobs = jobs.loc[mask.to_numpy()]
        wanted = pd.MultiIndex.from_frame(target_jobs.loc[:, parent_keys])
        details = cars.loc[pd.MultiIndex.from_frame(cars.loc[:, detail_keys]).isin(wanted)]
        return target_jobs, details, len(details) > 0, True
    target_details = cars.loc[mask.to_numpy()]
    detail_index = pd.MultiIndex.from_frame(target_details.loc[:, detail_keys])
    linked = detail_index.isin(parent_index) & target_details.loc[:, detail_keys].notna().all(axis=1).to_numpy()
    target_jobs = jobs.loc[parent_index.isin(detail_index)]
    wanted = pd.MultiIndex.from_frame(target_jobs.loc[:, parent_keys])
    details = cars.loc[pd.MultiIndex.from_frame(cars.loc[:, detail_keys]).isin(wanted)]
    return target_jobs, details, len(target_details) > 0, bool(linked.all())


def _schedule_field(schedule: CollectionScheduleDefinition | None, temporal: TemporalReconciliationDefinition,
                    frames: dict) -> TemporalFieldDefinition | None:
    """The temporal-contract field behind the schedule's timestamp (fail closed if none)."""
    if schedule is None:
        return None
    if not isinstance(temporal, TemporalReconciliationDefinition):
        raise TypeError(f"expected a TemporalReconciliationDefinition, got {type(temporal).__name__}")
    if schedule.dataset not in frames:
        raise LocationCoverageConfigurationError("schedule dataset is not part of the relationship")
    try:
        field = temporal.field((schedule.dataset, schedule.timestamp_column))
    except TemporalConfigurationError:
        raise LocationCoverageConfigurationError(
            "the schedule timestamp is not defined in the temporal contract") from None
    if field.kind is not TemporalKind.TIMESTAMP:
        raise LocationCoverageConfigurationError("the schedule timestamp must be a timestamp field")
    return field


def _time_coverage(schedule: CollectionScheduleDefinition | None, field: TemporalFieldDefinition | None,
                   temporal: TemporalReconciliationDefinition, frames: dict, mask: pd.Series,
                   coverage: LocationCoverageDefinition, target_jobs: pd.DataFrame) -> TimeCoverageStatus:
    """Compare target periods with an authoritative schedule (never inferred).

    Observed times are parsed only by the temporal contract; any missing,
    invalid or unresolved target time makes coverage UNASSESSABLE.
    """
    if schedule is None or field is None:
        return TimeCoverageStatus.NOT_ASSESSED
    if schedule.dataset == coverage.dataset:
        stamps = frames[schedule.dataset].loc[mask.to_numpy(), schedule.timestamp_column]
    else:
        stamps = target_jobs[schedule.timestamp_column]
    if stamps.empty:                                   # nothing observed for the target
        return TimeCoverageStatus.NEVER_PRESENT
    parsed = parse_temporal_field(stamps, field, temporal.canonical_timezone)
    if bool((parsed.missing | parsed.invalid | parsed.unresolved).any()):
        return TimeCoverageStatus.UNASSESSABLE
    observed = pd.DatetimeIndex(parsed.instants.dt.tz_convert("UTC")).floor(schedule.period)
    hit = schedule.expected_instants.isin(observed.unique())
    if not hit.any():
        return TimeCoverageStatus.NEVER_PRESENT
    return TimeCoverageStatus.COMPLETE if bool(hit.all()) else TimeCoverageStatus.PARTIAL
