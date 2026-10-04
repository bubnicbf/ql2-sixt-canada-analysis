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
   collection events of the target's scope (distinct detail relationship keys
   sharing the target rows' scope values) that lack the target. Some missing
   -> ``RAW_STREAM_PARTIAL``. This compares actual collection events; it does
   not infer a cadence.
5. ``TIME_COVERAGE`` - only with an authoritative
   :class:`~ql2_sixt_canada_analysis.schemas.CollectionScheduleDefinition`;
   otherwise temporal completeness is ``NOT_ASSESSED``.
6. ``IDENTIFIER_TYPES`` / ``PARENT_KEYS`` - relationship keys use the
   identifier dtype and the parent (jobs) key is complete and unique.
7. ``RELATIONSHIP`` / ``DETAIL_PRESENCE`` / ``RECONCILIATION`` - target detail
   rows link to exactly one job (``RELATIONSHIP_LINK_FAILURE``), target jobs
   have details (``JOBS_PRESENT_DETAILS_ABSENT``) and the target's jobs
   reconcile declared and observed counts (``JOB_DETAIL_COUNT_MISMATCH``).
8. Otherwise ``STREAM_PRESENT_AND_HEALTHY``.

The investigation never creates, repairs, filters or writes records and never
infers aliases or cadence. The report holds enums and booleans only - no
identifiers, location values (other than none at all), timestamps, rows or
counts.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from ql2_sixt_canada_analysis.identifiers import is_identifier_dtype
from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.reconciliation import assess_job_detail_reconciliation
from ql2_sixt_canada_analysis.relationships import assess_one_to_many_join
from ql2_sixt_canada_analysis.schemas import (
    COLLECTION_SCHEDULE,
    EXPECTED_LOCATION_COVERAGE,
    JOB_DETAIL_RELATIONSHIP,
    CollectionScheduleDefinition,
    JobDetailRelationshipDefinition,
    LocationCoverageConfigurationError,
    LocationCoverageDefinition,
)
from ql2_sixt_canada_analysis.unique_keys import assess_unique_key

__all__ = [
    "LocationStreamError",
    "LocationStreamInvestigationReport",
    "LocationStreamStatus",
    "PipelineStage",
    "StreamContinuity",
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
    RELATIONSHIP = "relationship"
    DETAIL_PRESENCE = "detail_presence"
    RECONCILIATION = "reconciliation"


class StreamContinuity(StrEnum):
    """Presence of the target across the collection events of its scope."""

    NOT_APPLICABLE = "not_applicable"   # no scope configured, or target absent
    COMPLETE = "complete"
    PARTIAL = "partial"


class TimeCoverageStatus(StrEnum):
    """Target presence across an authoritative schedule's periods."""

    NOT_ASSESSED = "not_assessed"       # no authoritative schedule
    NEVER_PRESENT = "never_present"
    PARTIAL = "partial"
    COMPLETE = "complete"


_REPOSITORY_STATUSES = frozenset({
    LocationStreamStatus.EXPECTATION_NOT_CONFIGURED, LocationStreamStatus.INGESTION_EXCLUSION,
    LocationStreamStatus.CLEANING_EXCLUSION, LocationStreamStatus.IDENTIFIER_TYPE_MISMATCH,
})
_UPSTREAM_STATUSES = frozenset({
    LocationStreamStatus.RAW_STREAM_ABSENT, LocationStreamStatus.RAW_STREAM_PARTIAL,
    LocationStreamStatus.PARENT_KEY_VIOLATION, LocationStreamStatus.RELATIONSHIP_LINK_FAILURE,
    LocationStreamStatus.JOBS_PRESENT_DETAILS_ABSENT, LocationStreamStatus.JOB_DETAIL_COUNT_MISMATCH,
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
    coverage: LocationCoverageDefinition = EXPECTED_LOCATION_COVERAGE,
) -> tuple[str, ...]:
    """Return ``target`` if it is a configured expected key, else raise.

    Exact lookup only: no fuzzy matching, no substitution, no data access.
    The error never lists configured or observed locations.

    Raises:
        LocationCoverageConfigurationError: The contract is unconfigured, or
            ``target`` is malformed or not an expected key.
    """
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
    coverage: LocationCoverageDefinition = EXPECTED_LOCATION_COVERAGE,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
    loaded: RawDatasets | None = None,
    raw_source: str | Path | None = None,
    schedule: CollectionScheduleDefinition | None = COLLECTION_SCHEDULE,
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

    Raises:
        TypeError: Invalid argument types.
        LocationCoverageConfigurationError: Configured columns are absent or
            the schedule/contract datasets are not part of the relationship.
    """
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be pandas DataFrames")
    if not isinstance(coverage, LocationCoverageDefinition):
        raise TypeError(f"expected a LocationCoverageDefinition, got {type(coverage).__name__}")
    if loaded is not None and not isinstance(loaded, RawDatasets):
        raise TypeError("loaded must be RawDatasets or None")
    frames = {relationship.parent: jobs, relationship.detail: cars}
    if coverage.dataset not in frames:
        raise LocationCoverageConfigurationError("coverage dataset is not part of the relationship")
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
    continuity = _continuity(frame, mask, coverage, relationship)
    if continuity is StreamContinuity.PARTIAL:
        fail(PipelineStage.SOURCE_CONTINUITY, LocationStreamStatus.RAW_STREAM_PARTIAL)

    # --- relationship preconditions
    dtypes_ok = all(is_identifier_dtype(jobs[c].dtype) for c in relationship.parent_key_columns
                    if c in jobs.columns) and all(
        is_identifier_dtype(cars[c].dtype) for c in relationship.detail_key_columns if c in cars.columns)
    if not dtypes_ok:
        fail(PipelineStage.IDENTIFIER_TYPES, LocationStreamStatus.IDENTIFIER_TYPE_MISMATCH)
    keys_ok = assess_unique_key(jobs, relationship.parent_definition).is_valid
    if not keys_ok:
        fail(PipelineStage.PARENT_KEYS, LocationStreamStatus.PARENT_KEY_VIOLATION)

    # --- target jobs and details
    target_jobs, target_details, details_present, all_linked = _target_rows(
        jobs, cars, mask, coverage, relationship)
    time_status = _time_coverage(schedule, frames, mask, coverage, target_jobs)
    if time_status is TimeCoverageStatus.PARTIAL:
        fail(PipelineStage.TIME_COVERAGE, LocationStreamStatus.RAW_STREAM_PARTIAL)

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
    status = _STATUS_BY_STAGE[failing[0]] if failing else LocationStreamStatus.STREAM_PRESENT_AND_HEALTHY
    return _report(
        status, failing,
        target_configured=True, present_in_raw_source=in_raw, present_after_ingestion=in_loaded,
        present_after_cleaning=True, matched_via_authoritative_alias=via_alias,
        unverified_representation_variant=None, stream_continuity=continuity,
        schedule_available=schedule is not None, time_coverage=time_status,
        identifier_dtypes_valid=dtypes_ok, parent_keys_valid=keys_ok,
        target_details_present=details_present, target_details_all_linked=all_linked,
        reconciliation_passes=reconciled, relationship_passes=related,
    )


def validate_location_stream(
    jobs: pd.DataFrame, cars: pd.DataFrame, target: tuple[str, ...], **kwargs: object,
) -> LocationStreamInvestigationReport:
    """Investigate; return the report if healthy, else raise :class:`LocationStreamError`."""
    report = investigate_location_stream(jobs, cars, target, **kwargs)  # type: ignore[arg-type]
    if not report.is_healthy:
        raise LocationStreamError(report)
    return report


# ---------------------------------------------------------------------- helpers


_STATUS_BY_STAGE = {
    PipelineStage.SOURCE_CONTINUITY: LocationStreamStatus.RAW_STREAM_PARTIAL,
    PipelineStage.TIME_COVERAGE: LocationStreamStatus.RAW_STREAM_PARTIAL,
    PipelineStage.IDENTIFIER_TYPES: LocationStreamStatus.IDENTIFIER_TYPE_MISMATCH,
    PipelineStage.PARENT_KEYS: LocationStreamStatus.PARENT_KEY_VIOLATION,
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


def _continuity(frame: pd.DataFrame, mask: pd.Series, coverage: LocationCoverageDefinition,
                relationship: JobDetailRelationshipDefinition) -> StreamContinuity:
    """Share of the scope's collection events (detail relationship keys) containing the target."""
    scope = coverage.stream_scope_columns
    if not scope or coverage.dataset != relationship.detail:
        return StreamContinuity.NOT_APPLICABLE
    event_columns = list(relationship.detail_key_columns)
    target_scope = pd.MultiIndex.from_frame(frame.loc[mask.to_numpy(), list(scope)].astype(object))
    in_scope = pd.MultiIndex.from_frame(frame.loc[:, list(scope)].astype(object)).isin(target_scope)
    events = frame.loc[in_scope, event_columns].dropna().drop_duplicates()
    target_events = frame.loc[mask.to_numpy(), event_columns].dropna().drop_duplicates()
    covered = pd.MultiIndex.from_frame(events).isin(pd.MultiIndex.from_frame(target_events))
    return StreamContinuity.COMPLETE if bool(covered.all()) else StreamContinuity.PARTIAL


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


def _time_coverage(schedule: CollectionScheduleDefinition | None, frames: dict, mask: pd.Series,
                   coverage: LocationCoverageDefinition, target_jobs: pd.DataFrame) -> TimeCoverageStatus:
    """Compare target periods with an authoritative schedule (never inferred)."""
    if schedule is None:
        return TimeCoverageStatus.NOT_ASSESSED
    if schedule.dataset not in frames:
        raise LocationCoverageConfigurationError("schedule dataset is not part of the relationship")
    if schedule.dataset == coverage.dataset:
        stamps = frames[schedule.dataset].loc[mask.to_numpy(), schedule.timestamp_column]
    else:
        stamps = target_jobs[schedule.timestamp_column]
    observed = pd.to_datetime(stamps, utc=True, errors="coerce", format="ISO8601").dropna().dt.floor(schedule.period)
    expected = pd.to_datetime(list(schedule.expected_periods), utc=True, format="ISO8601").floor(schedule.period)
    hit = expected.isin(pd.DatetimeIndex(observed.unique()))
    if not hit.any():
        return TimeCoverageStatus.NEVER_PRESENT
    return TimeCoverageStatus.COMPLETE if bool(hit.all()) else TimeCoverageStatus.PARTIAL
