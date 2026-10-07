"""Authority-backed, per-stream hourly collection schedule and scheduled coverage.

The schedule is built only from APPROVED decisions of the current authority
record (schema 3, ``pricing-authorities-v5`` onwards):

* ``SCHEDULE_CAPTURE_TIMESTAMP`` - the parent job's ``jobs.finished_at`` marks
  the scheduled execution. ``cars.job_finished_at`` is a replicated copy that
  must agree with its linked parent; ``cars.scraped_at`` is detail
  observation time only and is never used to place a job in a period.
* ``FINISHED_AT_TIMEZONE`` - an exhaustive ``city -> IANA zone`` map selected
  by the parent job's exact ``jobs.city``. A missing, blank or unknown city
  fails closed.
* ``SCHEDULE_EXPECTED_PERIODS`` - cadence ``PT1H`` phased at the top of every
  local clock hour, one parent job per city per local hour, and an explicit
  local window (inclusive end) per stream.
* ``SCHEDULE_SHARING_MODEL`` - ``PER_STREAM``: every expected stream has its
  own typed :class:`StreamSchedule` and its own expected-period set.
* ``SCHEDULE_EXCEPTIONS`` - :class:`ScheduleExceptions`; ``NO_EXCEPTIONS`` is an
  explicit model, never an empty value.

Expected periods are materialized from the definitions with the IANA database
(never from observed jobs): a nonexistent spring-forward local hour is
skipped; a repeated fall-back local hour yields two periods with distinct UTC
offsets (never deduplicated); nothing is shifted and no fixed offsets are
used. Every period exposes its local start, UTC offset, fold and canonical UTC
text (``YYYYMMDDTHHMMSSZ``).

Observed assignment (:func:`assess_per_stream_scheduled_coverage`) validates
the exact parent city, parses ``finished_at`` as a naive value, localizes it
with the city zone, converts it to UTC, takes the local top-of-hour period and
matches it to the expected periods. Every failure is counted and fails closed.
Per stream-period coverage uses the exact raw ``(city, location)`` key only:
no alias, case or whitespace matching and no cross-stream substitution.

Reports hold typed objects and counts. Nothing here prints source values.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache, cached_property
from types import MappingProxyType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.authority_decisions import (
    SCHEDULE_ANCHOR_FIELDS,
    SCHEDULE_DETAIL_COPY_FIELD,
    SCHEDULE_DETAIL_OBSERVATION_FIELD,
    UTC_INSTANT_PATTERN,
    AuthorityDecisionRecord,
    AuthorityKind,
    DecisionId,
    DecisionStatus,
    city_timezones,
    load_current_decision_record,
)
from ql2_sixt_canada_analysis.expected_stream_contract import (
    ExpectedStreamAuthorityStatus,
    ExpectedStreamContract,
    current_expected_stream_contract,
)
from ql2_sixt_canada_analysis.schemas import (
    CityTimezoneMap,
    JobDetailRelationshipDefinition,
    LocationCoverageDefinition,
    TemporalConfigurationError,
    classify_local_time,
    region_iana_zone,
)

__all__ = [
    "ParentCaptureExclusion",
    "ExcludedStreamPeriod",
    "ExcludedCapture",
    "CaptureExclusionSet",
    "CapturePeriodIndex",
    "FINISHED_AT_SOURCE_FORMAT",
    "SCHEDULE_DECISIONS",
    "CityTimezoneMap",
    "ExceptionsModel",
    "JobAssignmentFailure",
    "MissingStreamPeriod",
    "NormalizedFinish",
    "PerStreamSchedule",
    "PerStreamScheduledCoverageReport",
    "ScheduleAuthorityStatus",
    "ScheduleConfigurationError",
    "ScheduleCoverageBlocker",
    "ScheduleExceptions",
    "ScheduleFailureKind",
    "ScheduledPeriod",
    "SharingMode",
    "StreamPeriodCoverage",
    "StreamSchedule",
    "StreamScheduleException",
    "assess_per_stream_scheduled_coverage",
    "current_per_stream_schedule",
    "format_utc_instant",
    "materialize_local_hours",
    "normalize_finished_at",
    "schedule_from_record",
]

Key = tuple[str, ...]
_UTC = dt.timezone.utc
#: Canonical UTC text format of every instant (``YYYYMMDDTHHMMSSZ``).
UTC_TEXT_FORMAT = "%Y%m%dT%H%M%SZ"
#: Source format of ``jobs.finished_at`` and its copy ``cars.job_finished_at`` (naive local wall clock).
FINISHED_AT_SOURCE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
#: The only cadence and phase this model materializes (any other approved value fails closed).
SUPPORTED_CADENCE = "PT1H"
SUPPORTED_PHASE = "LOCAL_TOP_OF_HOUR"
_STEP = dt.timedelta(hours=1)

#: Every decision the schedule rests on; all must be APPROVED in a schema-3 record.
SCHEDULE_DECISIONS: tuple[DecisionId, ...] = (
    DecisionId.SCHEDULE_CAPTURE_TIMESTAMP,
    DecisionId.FINISHED_AT_TIMEZONE,
    DecisionId.SCHEDULE_EXPECTED_PERIODS,
    DecisionId.SCHEDULE_SHARING_MODEL,
    DecisionId.SCHEDULE_EXCEPTIONS,
)


class ScheduleConfigurationError(ValueError):
    """A schedule definition is incomplete, contradictory or unauthorised (messages carry no source values)."""


class ScheduleAuthorityStatus(StrEnum):
    AVAILABLE = "available"                  # every schedule decision APPROVED and the model validates
    NOT_APPROVED = "not_approved"            # a schedule decision (or the stream contract) is not approved
    RECORD_UNAVAILABLE = "record_unavailable"
    INVALID = "invalid"                      # approved decisions that contradict the model or the contract


class SharingMode(StrEnum):
    PER_STREAM = "PER_STREAM"
    SHARED = "SHARED"


class ExceptionsModel(StrEnum):
    NO_EXCEPTIONS = "NO_EXCEPTIONS"
    LISTED_EXCEPTIONS = "LISTED_EXCEPTIONS"


class ScheduleFailureKind(StrEnum):
    """Why an expected stream-period is not covered (the kind an exception must name)."""

    STREAM_ABSENT_FROM_CAPTURE = "STREAM_ABSENT_FROM_CAPTURE"   # the city's job ran; this stream had no rows
    PARENT_JOB_ABSENT = "PARENT_JOB_ABSENT"                     # no valid job for the city-period
    PARENT_JOB_AMBIGUOUS = "PARENT_JOB_AMBIGUOUS"               # several jobs claim the city-period (never excusable)
    PARENT_JOB_INVALID = "PARENT_JOB_INVALID"                   # the only job failed a check (never excusable)
    #: The whole parent collection execution of one city-period is governed as incomplete: every approved
    #: stream of that city is analytically null for that period, even streams whose rows exist.
    INCOMPLETE_PARENT_CAPTURE = "INCOMPLETE_PARENT_CAPTURE"


#: Failure kinds a governed exception may excuse.
EXCUSABLE_FAILURES = frozenset({ScheduleFailureKind.STREAM_ABSENT_FROM_CAPTURE, ScheduleFailureKind.PARENT_JOB_ABSENT})


class JobAssignmentFailure(StrEnum):
    """Why a parent job could not be assigned to exactly one expected city-period."""

    MISSING_CITY = "missing_city"
    UNKNOWN_CITY = "unknown_city"
    MISSING_FINISHED_AT = "missing_finished_at"
    INVALID_FINISHED_AT = "invalid_finished_at"
    NONEXISTENT_LOCAL_TIME = "nonexistent_local_time"
    AMBIGUOUS_LOCAL_TIME = "ambiguous_local_time"
    OUTSIDE_SCHEDULE_WINDOW = "outside_schedule_window"
    NO_EXPECTED_PERIOD = "no_expected_period"
    DUPLICATE_CITY_PERIOD = "duplicate_city_period"
    DETAIL_COPY_MISMATCH = "detail_copy_mismatch"


class ScheduleCoverageBlocker(StrEnum):
    """Scheduled-coverage blockers (values equal ``PricingBlocker`` values)."""

    COLLECTION_SCHEDULE_UNAVAILABLE = "collection_schedule_unavailable"
    COLLECTION_SCHEDULE_INVALID = "collection_schedule_invalid"
    SCHEDULED_COVERAGE_STREAMS_NOT_EXACT = "scheduled_coverage_streams_not_exact"
    SCHEDULED_JOB_ASSIGNMENT_FAILED = "scheduled_job_assignment_failed"
    SCHEDULED_DETAIL_COPY_MISMATCH = "scheduled_detail_copy_mismatch"
    SCHEDULED_COVERAGE_INCOMPLETE = "scheduled_coverage_incomplete"
    SCHEDULE_EXCLUSION_UNMATCHED = "schedule_exclusion_unmatched"    # an exclusion matches no or several captures


# ------------------------------------------------------------------ instants


def format_utc_instant(instant: dt.datetime) -> str:
    """Canonical ``YYYYMMDDTHHMMSSZ`` text of an aware instant (a naive value is refused, never suffixed)."""
    if not isinstance(instant, dt.datetime) or instant.tzinfo is None or instant.utcoffset() is None:
        raise ScheduleConfigurationError("only an aware instant has a canonical UTC text")
    return instant.astimezone(_UTC).strftime(UTC_TEXT_FORMAT)


def _region_zone(name: object) -> ZoneInfo:
    """The shared region-zone check (:func:`~ql2_sixt_canada_analysis.schemas.region_iana_zone`)."""
    try:
        return region_iana_zone(name)
    except TemporalConfigurationError:
        raise ScheduleConfigurationError("timezone must be a region IANA zone name") from None


#: The shared DST classification (one mechanism for the schedule and the temporal contract).
_local_kind = classify_local_time


@dataclass(frozen=True, slots=True)
class NormalizedFinish:
    """A parent ``finished_at`` value resolved to an instant (the raw value is preserved)."""

    raw: str
    city: str
    timezone: str
    local: dt.datetime                 # naive local wall clock, as parsed
    utc_offset: dt.timedelta
    fold: int
    utc: dt.datetime                   # aware UTC

    @property
    def utc_text(self) -> str:
        return format_utc_instant(self.utc)


class _AssignmentError(Exception):
    def __init__(self, kind: JobAssignmentFailure) -> None:
        super().__init__(kind.value)
        self.kind = kind


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA or value is pd.NaT or (
        isinstance(value, str) and not value.strip())


def normalize_finished_at(raw: object, city: object, timezones: CityTimezoneMap) -> NormalizedFinish:
    """Parse ``raw`` as naive local time in the zone of the exact ``city`` and convert it to UTC.

    Raises ``_AssignmentError`` (a :class:`JobAssignmentFailure`) for a missing
    or unknown city, a missing or unparseable value, or a nonexistent or
    ambiguous local time. ``Z`` is never appended to a local value.
    """
    if _missing(city):
        raise _AssignmentError(JobAssignmentFailure.MISSING_CITY)
    try:
        zone_name = timezones.zone_name(city)
    except (ScheduleConfigurationError, TemporalConfigurationError):
        raise _AssignmentError(JobAssignmentFailure.UNKNOWN_CITY) from None
    if _missing(raw):
        raise _AssignmentError(JobAssignmentFailure.MISSING_FINISHED_AT)
    if not isinstance(raw, str):
        raise _AssignmentError(JobAssignmentFailure.INVALID_FINISHED_AT)
    try:
        local = dt.datetime.strptime(raw, FINISHED_AT_SOURCE_FORMAT)
    except ValueError:
        raise _AssignmentError(JobAssignmentFailure.INVALID_FINISHED_AT) from None
    zone = ZoneInfo(zone_name)
    kind = _local_kind(local, zone)
    if kind == "nonexistent":
        raise _AssignmentError(JobAssignmentFailure.NONEXISTENT_LOCAL_TIME)
    if kind == "ambiguous":
        raise _AssignmentError(JobAssignmentFailure.AMBIGUOUS_LOCAL_TIME)
    aware = local.replace(tzinfo=zone)
    return NormalizedFinish(raw=raw, city=city, timezone=zone_name, local=local, utc_offset=aware.utcoffset(),
                            fold=0, utc=aware.astimezone(_UTC))


# ------------------------------------------------------------------- periods


@dataclass(frozen=True, slots=True)
class ScheduledPeriod:
    """One expected period of one stream: its local start, offset, fold and UTC instant."""

    stream: Key
    local_start: dt.datetime           # naive local wall clock
    utc_offset: dt.timedelta
    fold: int
    utc_start: dt.datetime             # aware UTC

    def __post_init__(self) -> None:
        if self.local_start.tzinfo is not None or self.utc_start.tzinfo is None:
            raise ScheduleConfigurationError("a period needs a naive local start and an aware UTC start")
        if self.fold not in (0, 1):
            raise ScheduleConfigurationError("fold must be 0 or 1")
        if self.utc_start.astimezone(_UTC).replace(tzinfo=None) + self.utc_offset != self.local_start:
            raise ScheduleConfigurationError("local start, UTC offset and UTC start disagree")

    @property
    def utc_text(self) -> str:
        return format_utc_instant(self.utc_start)

    @property
    def local_text(self) -> str:
        """Local start with its explicit offset (ISO 8601)."""
        return self.local_start.replace(tzinfo=dt.timezone(self.utc_offset)).isoformat()


def materialize_local_hours(stream: Key, timezone: str, local_start: dt.datetime, local_end: dt.datetime,
                            end_inclusive: bool) -> tuple[ScheduledPeriod, ...]:
    """Top-of-hour local periods of ``[local_start, local_end]`` (or half-open) in ``timezone``.

    Deterministic, from the IANA database only: nonexistent local hours are
    skipped, repeated local hours yield both offsets (fold 0 then 1). Sorted by
    UTC instant; never deduplicated by local wall-clock value.
    """
    zone = _region_zone(timezone)
    out: list[ScheduledPeriod] = []
    current = local_start
    while current < local_end or (end_inclusive and current == local_end):
        kind = _local_kind(current, zone)
        if kind != "nonexistent":
            for fold in ((0, 1) if kind == "ambiguous" else (0,)):
                aware = current.replace(tzinfo=zone, fold=fold)
                out.append(ScheduledPeriod(stream=stream, local_start=current, utc_offset=aware.utcoffset(),
                                           fold=fold, utc_start=aware.astimezone(_UTC)))
        current += _STEP
    out.sort(key=lambda p: p.utc_start)
    if len({p.utc_start for p in out}) != len(out):
        raise ScheduleConfigurationError("materialized periods collide")
    return tuple(out)


@dataclass(frozen=True)
class StreamSchedule:
    """The typed schedule of one expected stream (``PER_STREAM``)."""

    schedule_version: str
    stream: Key
    city: str
    timezone: str
    local_start: dt.datetime
    local_end: dt.datetime
    end_inclusive: bool
    cadence: str
    phase: str
    capture_field: str
    record_id: str
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (isinstance(self.stream, tuple) and len(self.stream) == 2
                and all(isinstance(v, str) and v and v == v.strip() for v in self.stream)):
            raise ScheduleConfigurationError("stream must be an exact (city, location) key")
        if self.city != self.stream[0]:
            raise ScheduleConfigurationError("a stream schedule's city must be the stream's city")
        _region_zone(self.timezone)
        for bound in (self.local_start, self.local_end):
            if not isinstance(bound, dt.datetime) or bound.tzinfo is not None:
                raise ScheduleConfigurationError("boundaries must be naive local wall-clock values")
            if bound.minute or bound.second or bound.microsecond:
                raise ScheduleConfigurationError("boundaries must be at the top of a local hour")
        if self.local_end < self.local_start:
            raise ScheduleConfigurationError("local end precedes local start")
        if not isinstance(self.end_inclusive, bool):
            raise ScheduleConfigurationError("end_inclusive must be a boolean")
        if self.cadence != SUPPORTED_CADENCE or self.phase != SUPPORTED_PHASE:
            raise ScheduleConfigurationError("only PT1H cadence phased at the local top of hour is supported")
        if self.capture_field not in SCHEDULE_ANCHOR_FIELDS:
            raise ScheduleConfigurationError("the capture field must be the parent job finish time")
        if not self.schedule_version or not self.record_id or not self.references:
            raise ScheduleConfigurationError("a stream schedule needs its version, record and references")

    @cached_property
    def periods(self) -> tuple[ScheduledPeriod, ...]:
        """The materialized expected periods (computed once from the definition)."""
        return materialize_local_hours(self.stream, self.timezone, self.local_start, self.local_end,
                                       self.end_inclusive)

    def materialize(self) -> tuple[ScheduledPeriod, ...]:
        return self.periods

    @property
    def period_count(self) -> int:
        return len(self.periods)


# ----------------------------------------------------------------- exceptions


@dataclass(frozen=True, slots=True)
class StreamScheduleException:
    """One governed exception: exactly one stream, period, failure kind and schedule version."""

    stream: Key
    period_start_utc: str
    local_start: dt.datetime
    utc_offset: dt.timedelta
    failure: ScheduleFailureKind
    reason: str
    authority_kind: AuthorityKind
    reference: str
    schedule_version: str

    def __post_init__(self) -> None:
        if not (isinstance(self.period_start_utc, str) and UTC_INSTANT_PATTERN.fullmatch(self.period_start_utc)):
            raise ScheduleConfigurationError("an exception's period must be YYYYMMDDTHHMMSSZ")
        if not isinstance(self.failure, ScheduleFailureKind) or self.failure not in EXCUSABLE_FAILURES:
            raise ScheduleConfigurationError("unsupported exception failure kind")
        if not isinstance(self.authority_kind, AuthorityKind):
            raise ScheduleConfigurationError("an exception needs a responsible authority kind")
        if not self.reason or not self.reference or not self.schedule_version:
            raise ScheduleConfigurationError("an exception needs a reason, a durable reference and a version")

    def excuses(self, stream: Key, period: ScheduledPeriod, failure: ScheduleFailureKind, version: str) -> bool:
        return (self.stream == stream and self.period_start_utc == period.utc_text
                and self.local_start == period.local_start and self.utc_offset == period.utc_offset
                and self.failure is failure and self.schedule_version == version)


@dataclass(frozen=True, slots=True)
class ParentCaptureExclusion:
    """One governed ``INCOMPLETE_PARENT_CAPTURE``: a whole city-period collection execution is analytically null.

    Names the city, every approved stream of that city, the scheduled UTC
    period with its local start and offset, the reason, the responsible
    authority, the durable reference and the schedule version - never a job
    identifier. It applies to exactly this city, period and version; at
    assessment it must match exactly one parent capture.
    """

    city: str
    streams: tuple[Key, ...]
    period_start_utc: str
    local_start: dt.datetime
    utc_offset: dt.timedelta
    reason: str
    authority_kind: AuthorityKind
    reference: str
    schedule_version: str
    failure: ScheduleFailureKind = ScheduleFailureKind.INCOMPLETE_PARENT_CAPTURE

    def __post_init__(self) -> None:
        if self.failure is not ScheduleFailureKind.INCOMPLETE_PARENT_CAPTURE:
            raise ScheduleConfigurationError("a parent-capture exclusion has the INCOMPLETE_PARENT_CAPTURE kind")
        if not (isinstance(self.period_start_utc, str) and UTC_INSTANT_PATTERN.fullmatch(self.period_start_utc)):
            raise ScheduleConfigurationError("an exclusion's period must be YYYYMMDDTHHMMSSZ")
        if (not isinstance(self.streams, tuple) or not self.streams or len(set(self.streams)) != len(self.streams)
                or any(not isinstance(k, tuple) or len(k) != 2 or k[0] != self.city for k in self.streams)):
            raise ScheduleConfigurationError("an exclusion names distinct streams of its own city")
        if not isinstance(self.authority_kind, AuthorityKind):
            raise ScheduleConfigurationError("an exclusion needs a responsible authority kind")
        if not self.reason or not self.reference or not self.schedule_version:
            raise ScheduleConfigurationError("an exclusion needs a reason, a durable reference and a version")

    def applies(self, stream: Key, period: ScheduledPeriod, version: str) -> bool:
        return (stream in self.streams and self.period_start_utc == period.utc_text
                and self.local_start == period.local_start and self.utc_offset == period.utc_offset
                and self.schedule_version == version)


@dataclass(frozen=True, slots=True)
class ScheduleExceptions:
    """The exceptions model. ``NO_EXCEPTIONS`` is explicit; ``LISTED_EXCEPTIONS`` must list at least one
    stream-level exception or parent-capture exclusion."""

    model: ExceptionsModel
    exceptions: tuple[StreamScheduleException, ...] = ()
    parent_capture_exclusions: tuple[ParentCaptureExclusion, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model, ExceptionsModel):
            raise ScheduleConfigurationError("model must be an ExceptionsModel")
        if not isinstance(self.exceptions, tuple) or not all(
                isinstance(e, StreamScheduleException) for e in self.exceptions):
            raise ScheduleConfigurationError("exceptions must be a tuple of StreamScheduleException")
        if not isinstance(self.parent_capture_exclusions, tuple) or not all(
                isinstance(e, ParentCaptureExclusion) for e in self.parent_capture_exclusions):
            raise ScheduleConfigurationError("parent-capture exclusions must be ParentCaptureExclusion objects")
        listed = bool(self.exceptions) or bool(self.parent_capture_exclusions)
        if (self.model is ExceptionsModel.NO_EXCEPTIONS) == listed:
            raise ScheduleConfigurationError("NO_EXCEPTIONS lists none; LISTED_EXCEPTIONS lists at least one")
        markers = [(e.stream, e.period_start_utc, e.failure) for e in self.exceptions]
        if len(set(markers)) != len(markers):
            raise ScheduleConfigurationError("duplicate exception")
        captures = [(e.city, e.period_start_utc) for e in self.parent_capture_exclusions]
        if len(set(captures)) != len(captures):
            raise ScheduleConfigurationError("duplicate parent-capture exclusion")
        covered = {(k, e.period_start_utc) for e in self.parent_capture_exclusions for k in e.streams}
        if any((e.stream, e.period_start_utc) in covered for e in self.exceptions):
            raise ScheduleConfigurationError("a stream-period cannot be both excused and excluded")

    def excluding(self, stream: Key, period: ScheduledPeriod, version: str) -> ParentCaptureExclusion | None:
        return next((e for e in self.parent_capture_exclusions if e.applies(stream, period, version)), None)

    @classmethod
    def none(cls) -> ScheduleExceptions:
        return cls(model=ExceptionsModel.NO_EXCEPTIONS)

    def excuses(self, stream: Key, period: ScheduledPeriod, failure: ScheduleFailureKind, version: str) -> bool:
        return any(e.excuses(stream, period, failure, version) for e in self.exceptions)


# ------------------------------------------------------------------- schedule


@dataclass(frozen=True)
class PerStreamSchedule:
    """The authority-backed per-stream schedule, or why it is unavailable (no source values).

    When ``status`` is ``AVAILABLE`` it holds exactly one :class:`StreamSchedule`
    per expected stream of the approved contract (unknown, duplicate and
    missing streams are refused), the city timezone map, the capture fields
    and the explicit exceptions model.
    """

    status: ScheduleAuthorityStatus
    expected_streams: tuple[Key, ...] = ()
    record_id: str | None = None
    schedule_version: str | None = None
    capture_field: str | None = None
    detail_copy_field: str | None = None
    detail_observation_field: str | None = None
    sharing_mode: SharingMode | None = None
    timezones: CityTimezoneMap | None = None
    schedules: tuple[StreamSchedule, ...] = ()
    exceptions: ScheduleExceptions | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ScheduleAuthorityStatus):
            raise TypeError("status must be a ScheduleAuthorityStatus")
        if self.status is not ScheduleAuthorityStatus.AVAILABLE:
            if self.schedules or self.exceptions is not None or self.timezones is not None:
                raise ScheduleConfigurationError("only an available schedule carries definitions")
            return
        if self.sharing_mode is not SharingMode.PER_STREAM:
            raise ScheduleConfigurationError("only the PER_STREAM sharing model is implemented")
        if self.capture_field not in SCHEDULE_ANCHOR_FIELDS:
            raise ScheduleConfigurationError("the capture field must be the parent job finish time")
        if (self.detail_copy_field != SCHEDULE_DETAIL_COPY_FIELD
                or self.detail_observation_field != SCHEDULE_DETAIL_OBSERVATION_FIELD):
            raise ScheduleConfigurationError("unexpected detail copy or observation field")
        if not isinstance(self.timezones, CityTimezoneMap) or not isinstance(self.exceptions, ScheduleExceptions):
            raise ScheduleConfigurationError("an available schedule needs its timezone map and exceptions model")
        if not self.record_id or not self.schedule_version or not self.references:
            raise ScheduleConfigurationError("an available schedule needs its record, version and references")
        if not self.expected_streams or len(set(self.expected_streams)) != len(self.expected_streams):
            raise ScheduleConfigurationError("expected streams must be a non-empty set of exact keys")
        if not all(isinstance(s, StreamSchedule) for s in self.schedules):
            raise ScheduleConfigurationError("schedules must be StreamSchedule objects")
        keys = [s.stream for s in self.schedules]
        if len(set(keys)) != len(keys):
            raise ScheduleConfigurationError("duplicate stream schedule")
        if set(keys) - set(self.expected_streams):
            raise ScheduleConfigurationError("a stream schedule names an unknown stream")
        if set(self.expected_streams) - set(keys):
            raise ScheduleConfigurationError("an expected stream has no schedule")
        if set(self.timezones.cities) != {k[0] for k in self.expected_streams}:
            raise ScheduleConfigurationError("the timezone map must cover exactly the expected cities")
        for schedule in self.schedules:
            if schedule.timezone != self.timezones.zone_name(schedule.city):
                raise ScheduleConfigurationError("a stream schedule's zone differs from its city's zone")
            if schedule.schedule_version != self.schedule_version or schedule.capture_field != self.capture_field:
                raise ScheduleConfigurationError("a stream schedule disagrees with the schedule version or anchor")
        for exclusion in self.exceptions.parent_capture_exclusions:
            city_streams = {s.stream for s in self.schedules if s.city == exclusion.city}
            if exclusion.schedule_version != self.schedule_version or set(exclusion.streams) != city_streams:
                raise ScheduleConfigurationError("an exclusion must name every approved stream of its city")
            for stream in exclusion.streams:
                period = next((p for p in self.schedule_for(stream).periods
                               if p.utc_text == exclusion.period_start_utc), None)
                if period is None or (period.local_start, period.utc_offset) != (exclusion.local_start,
                                                                                 exclusion.utc_offset):
                    raise ScheduleConfigurationError("an exclusion names no expected period of its streams")
        for exc in self.exceptions.exceptions:
            if exc.schedule_version != self.schedule_version or exc.stream not in keys:
                raise ScheduleConfigurationError("an exception names another version or an unknown stream")
            period = next((p for p in self.schedule_for(exc.stream).periods if p.utc_text == exc.period_start_utc),
                          None)
            if period is None or period.local_start != exc.local_start or period.utc_offset != exc.utc_offset:
                raise ScheduleConfigurationError("an exception names no expected period of its stream")

    @property
    def available(self) -> bool:
        return self.status is ScheduleAuthorityStatus.AVAILABLE

    @property
    def blocking_reasons(self) -> tuple[ScheduleCoverageBlocker, ...]:
        if self.status is ScheduleAuthorityStatus.AVAILABLE:
            return ()
        if self.status is ScheduleAuthorityStatus.INVALID:
            return (ScheduleCoverageBlocker.COLLECTION_SCHEDULE_INVALID,)
        return (ScheduleCoverageBlocker.COLLECTION_SCHEDULE_UNAVAILABLE,)

    def schedule_for(self, stream: Key) -> StreamSchedule:
        for schedule in self.schedules:
            if schedule.stream == tuple(stream):
                return schedule
        raise KeyError("no schedule for this stream")

    @property
    def schedule_count(self) -> int:
        return len(self.schedules)

    @property
    def period_count_by_stream(self) -> Mapping[Key, int]:
        return MappingProxyType({s.stream: s.period_count for s in self.schedules})

    @property
    def period_count_by_city(self) -> Mapping[str, int]:
        counts: Counter[str] = Counter()
        for schedule in self.schedules:
            counts[schedule.city] += schedule.period_count
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def total_period_count(self) -> int:
        return sum(s.period_count for s in self.schedules)

    @property
    def excused_period_count(self) -> int:
        """Stream-periods named by governed stream exceptions (excused missing periods)."""
        return len(self.exceptions.exceptions) if self.exceptions is not None else 0

    @property
    def excluded_period_count(self) -> int:
        """Stream-periods named by governed parent-capture exclusions (every stream of each excluded capture)."""
        if self.exceptions is None:
            return 0
        return sum(len(e.streams) for e in self.exceptions.parent_capture_exclusions)

    @property
    def parent_capture_exclusion_count(self) -> int:
        return len(self.exceptions.parent_capture_exclusions) if self.exceptions is not None else 0


def _local(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def schedule_from_record(record: AuthorityDecisionRecord | None,
                         contract: ExpectedStreamContract) -> PerStreamSchedule:
    """Build the per-stream schedule from the APPROVED schedule decisions of ``record`` (fail closed)."""
    S = ScheduleAuthorityStatus
    if record is None:
        return PerStreamSchedule(status=S.RECORD_UNAVAILABLE)
    if (record.schema_version < 3 or contract.status is not ExpectedStreamAuthorityStatus.APPROVED
            or any(record.decision(d).status is not DecisionStatus.APPROVED for d in SCHEDULE_DECISIONS)):
        return PerStreamSchedule(status=S.NOT_APPROVED, record_id=record.record_id)
    try:
        capture = record.approved_resolution(DecisionId.SCHEDULE_CAPTURE_TIMESTAMP)
        zones = CityTimezoneMap(tuple(city_timezones(record.approved_resolution(DecisionId.FINISHED_AT_TIMEZONE))
                                      .items()))
        periods = record.approved_resolution(DecisionId.SCHEDULE_EXPECTED_PERIODS)
        sharing = SharingMode(record.approved_resolution(DecisionId.SCHEDULE_SHARING_MODEL)["mode"])
        references = tuple(dict.fromkeys(a.reference for d in SCHEDULE_DECISIONS
                                         for a in record.decision(d).authority))
        version = periods["schedule_version"]
        schedules = tuple(
            StreamSchedule(schedule_version=version, stream=tuple(item["stream"]), city=item["stream"][0],
                           timezone=zones.zone_name(item["stream"][0]), local_start=_local(item["local_start"]),
                           local_end=_local(item["local_end"]), end_inclusive=item["end_inclusive"],
                           cadence=periods["cadence"], phase=periods["phase"], capture_field=capture["field"],
                           record_id=record.record_id, references=references)
            for item in periods["streams"])
        raw_exceptions = record.approved_resolution(DecisionId.SCHEDULE_EXCEPTIONS)
        exceptions = _exceptions(raw_exceptions, schedules, zones)
        return PerStreamSchedule(
            status=S.AVAILABLE, expected_streams=contract.expected_keys, record_id=record.record_id,
            schedule_version=version, capture_field=capture["field"], detail_copy_field=capture["detail_copy"],
            detail_observation_field=capture["detail_observation"], sharing_mode=sharing, timezones=zones,
            schedules=schedules, exceptions=exceptions, references=references)
    except (ScheduleConfigurationError, KeyError, TypeError, ValueError):
        return PerStreamSchedule(status=S.INVALID, record_id=record.record_id)


def _exceptions(raw: Mapping, schedules: tuple[StreamSchedule, ...], zones: CityTimezoneMap) -> ScheduleExceptions:
    model = ExceptionsModel(raw["model"])
    if model is ExceptionsModel.NO_EXCEPTIONS:
        return ScheduleExceptions.none()
    items, exclusions = [], []
    for item in raw["exceptions"]:
        if item["failure"] == ScheduleFailureKind.INCOMPLETE_PARENT_CAPTURE.value:
            streams = tuple(sorted(tuple(k) for k in item["streams"]))
            schedule = next((s for s in schedules if s.stream == streams[0]), None)
            period = (next((p for p in schedule.periods if p.utc_text == item["period_start_utc"]), None)
                      if schedule is not None else None)
            if period is None:
                raise ScheduleConfigurationError("an exclusion names no expected period")
            exclusions.append(ParentCaptureExclusion(
                city=item["city"], streams=streams, period_start_utc=item["period_start_utc"],
                local_start=period.local_start, utc_offset=period.utc_offset, reason=item["reason"],
                authority_kind=AuthorityKind(item["authority_kind"]), reference=item["reference"],
                schedule_version=item["schedule_version"]))
            continue
        stream = tuple(item["stream"])
        schedule = next((s for s in schedules if s.stream == stream), None)
        period = (next((p for p in schedule.periods if p.utc_text == item["period_start_utc"]), None)
                  if schedule is not None else None)
        if period is None:
            raise ScheduleConfigurationError("an exception names no expected period of its stream")
        items.append(StreamScheduleException(
            stream=stream, period_start_utc=item["period_start_utc"], local_start=period.local_start,
            utc_offset=period.utc_offset, failure=ScheduleFailureKind(item["failure"]), reason=item["reason"],
            authority_kind=AuthorityKind(item["authority_kind"]), reference=item["reference"],
            schedule_version=item["schedule_version"]))
    return ScheduleExceptions(model=model, exceptions=tuple(items), parent_capture_exclusions=tuple(exclusions))


@cache
def current_per_stream_schedule() -> PerStreamSchedule:
    """The project schedule from the current committed authority record (cached)."""
    return schedule_from_record(load_current_decision_record(), current_expected_stream_contract())


# ------------------------------------------------------------------- coverage


@dataclass(frozen=True, slots=True)
class MissingStreamPeriod:
    period: ScheduledPeriod
    failure: ScheduleFailureKind
    excused: bool


@dataclass(frozen=True, slots=True)
class ExcludedCapture:
    """One resolved excluded parent capture (holds a linkage key: in memory only, never reported)."""

    parent_key: tuple = field(repr=False)
    city: str = ""
    streams: tuple[Key, ...] = ()


@dataclass(frozen=True, slots=True)
class CaptureExclusionSet:
    """The governed parent captures excluded from pricing, resolved on the assessed frames.

    Built only by :func:`assess_per_stream_scheduled_coverage` from matched
    ``INCOMPLETE_PARENT_CAPTURE`` exclusions. Raw rows are never removed:
    the masks select the parent and detail rows to keep out of the pricing
    population and out of stream-continuity gaps.
    """

    parent_key_columns: tuple[str, ...]
    detail_key_columns: tuple[str, ...]
    entries: tuple[ExcludedCapture, ...] = ()

    def _mask(self, frame: pd.DataFrame, columns: tuple[str, ...], stream: Key | None) -> np.ndarray:
        keys = {e.parent_key for e in self.entries if stream is None or tuple(stream) in e.streams}
        if not keys or not len(frame):
            return np.zeros(len(frame), dtype=bool)
        if any(c not in frame.columns for c in columns):
            raise ScheduleConfigurationError("the exclusion key columns are absent")
        values = frame.loc[:, list(columns)].astype(object).itertuples(index=False, name=None)
        return np.fromiter((tuple(v) in keys for v in values), dtype=bool, count=len(frame))

    def parent_mask(self, jobs: pd.DataFrame, stream: Key | None = None) -> np.ndarray:
        """Parent rows of an excluded capture (optionally: only exclusions covering ``stream``)."""
        return self._mask(jobs, self.parent_key_columns, stream)

    def detail_mask(self, cars: pd.DataFrame) -> np.ndarray:
        """Detail rows linked to an excluded parent capture (every stream of that capture)."""
        return self._mask(cars, self.detail_key_columns, None)


@dataclass(frozen=True)
class CapturePeriodIndex:
    """The scheduled period (``YYYYMMDDTHHMMSSZ`` UTC start) of every validly assigned parent capture.

    Built only by :func:`assess_per_stream_scheduled_coverage`: a parent job
    has a period only when it is the single valid claimant of its
    city-period with every detail copy agreeing. Held in memory, never
    reported (keys are linkage keys).
    """

    parent_key_columns: tuple[str, ...]
    detail_key_columns: tuple[str, ...]
    periods: Mapping[tuple, str] = field(default_factory=lambda: MappingProxyType({}), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "periods", MappingProxyType(dict(self.periods)))

    def _lookup(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
        if any(c not in frame.columns for c in columns):
            raise ScheduleConfigurationError("the capture-period key columns are absent")
        values = frame.loc[:, list(columns)].astype(object).itertuples(index=False, name=None)
        return pd.Series([self.periods.get(tuple(v)) for v in values], index=frame.index, dtype=object)

    def parent_periods(self, jobs: pd.DataFrame) -> pd.Series:
        """Scheduled period text of each parent row (``None`` when not validly assigned)."""
        return self._lookup(jobs, self.parent_key_columns)

    def detail_periods(self, cars: pd.DataFrame) -> pd.Series:
        """Scheduled period text of each detail row's parent capture (``None`` when not validly assigned)."""
        return self._lookup(cars, self.detail_key_columns)


@dataclass(frozen=True, slots=True)
class ExcludedStreamPeriod:
    """A stream-period that a governed parent-capture exclusion makes analytically null."""

    period: ScheduledPeriod
    exclusion: ParentCaptureExclusion


@dataclass(frozen=True, slots=True)
class StreamPeriodCoverage:
    """Coverage of one stream's own expected periods (exact raw key only).

    ``expected`` is the nominal count; periods excluded by a matched
    parent-capture exclusion are neither covered nor missing, and the
    remaining ``required`` periods must each be covered or excused.
    """

    stream: Key
    expected: int
    covered: int
    missing: tuple[MissingStreamPeriod, ...]
    excluded: tuple[ExcludedStreamPeriod, ...] = ()

    @property
    def required(self) -> int:
        return self.expected - len(self.excluded)

    @property
    def unexcused_missing(self) -> int:
        return sum(1 for m in self.missing if not m.excused)

    @property
    def excused(self) -> int:
        return sum(1 for m in self.missing if m.excused)

    @property
    def missing_by_failure(self) -> Mapping[str, int]:
        counts = Counter(m.failure.value for m in self.missing if not m.excused)
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def complete(self) -> bool:
        return self.unexcused_missing == 0 and self.covered + self.excused + len(self.excluded) == self.expected


@dataclass(frozen=True, slots=True)
class PerStreamScheduledCoverageReport:
    """Per-stream scheduled coverage against the authority-backed schedule (aggregate counts and typed periods)."""

    schedule: PerStreamSchedule
    coverage: LocationCoverageDefinition | None
    streams: tuple[StreamPeriodCoverage, ...] = ()
    jobs_assessed: int = 0
    jobs_assigned: int = 0
    job_failures: tuple[tuple[JobAssignmentFailure, int], ...] = ()
    detail_copy_mismatches: int = 0
    expected_city_periods: tuple[tuple[str, int], ...] = ()
    missing_city_periods: tuple[tuple[str, int], ...] = ()
    #: Parent-capture exclusions that matched no or several parent captures (fail closed).
    unmatched_exclusions: int = 0
    #: The resolved excluded parent captures (in memory only, never reported: they hold linkage keys).
    capture_exclusions: CaptureExclusionSet | None = field(default=None, repr=False)
    #: The scheduled period of every validly assigned parent capture (in memory only).
    capture_periods: CapturePeriodIndex | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, PerStreamSchedule):
            raise TypeError("schedule must be a PerStreamSchedule")

    @property
    def nominal_periods(self) -> int:
        return sum(c.expected for c in self.streams)

    @property
    def excluded_periods(self) -> int:
        return sum(len(c.excluded) for c in self.streams)

    @property
    def required_periods(self) -> int:
        return sum(c.required for c in self.streams)

    @property
    def covered_periods(self) -> int:
        return sum(c.covered for c in self.streams)

    @property
    def excluded_parent_captures(self) -> int:
        return len(self.capture_exclusions.entries) if self.capture_exclusions is not None else 0

    @property
    def schedule_assessment(self) -> PerStreamSchedule:
        """The schedule this coverage was assessed against (``available`` / ``status``)."""
        return self.schedule

    @property
    def jobs_not_assigned(self) -> int:
        """Parent jobs not assigned to exactly one expected city-period (every failure kind)."""
        return self.jobs_assessed - self.jobs_assigned

    @property
    def job_failure_counts(self) -> Mapping[str, int]:
        return MappingProxyType({k.value: n for k, n in self.job_failures})

    @property
    def streams_exact(self) -> bool:
        keys = tuple(c.stream for c in self.streams)
        return (self.schedule.available and self.coverage is not None and self.coverage.is_configured
                and len(set(keys)) == len(keys)
                and set(keys) == set(self.schedule.expected_streams) == set(self.coverage.expected_locations))

    @property
    def blocking_reasons(self) -> tuple[ScheduleCoverageBlocker, ...]:
        B = ScheduleCoverageBlocker
        if not self.schedule.available:
            return self.schedule.blocking_reasons
        found = []
        if not self.streams_exact:
            found.append(B.SCHEDULED_COVERAGE_STREAMS_NOT_EXACT)
        if any(n for _, n in self.job_failures):
            found.append(B.SCHEDULED_JOB_ASSIGNMENT_FAILED)
        if self.detail_copy_mismatches:
            found.append(B.SCHEDULED_DETAIL_COPY_MISMATCH)
        if self.unmatched_exclusions:
            found.append(B.SCHEDULE_EXCLUSION_UNMATCHED)
        if not self.streams or not all(c.complete for c in self.streams):
            found.append(B.SCHEDULED_COVERAGE_INCOMPLETE)
        return tuple(found)

    @property
    def all_streams_complete(self) -> bool:
        return not self.blocking_reasons

    @property
    def is_valid(self) -> bool:
        return not self.blocking_reasons

    @property
    def unexcused_missing_total(self) -> int:
        return sum(c.unexcused_missing for c in self.streams)

    @property
    def excused_total(self) -> int:
        return sum(c.excused for c in self.streams)


def _column(field_ref: str) -> str:
    return field_ref.split(".", 1)[1]


def assess_per_stream_scheduled_coverage(jobs: pd.DataFrame, cars: pd.DataFrame, *, schedule: PerStreamSchedule,
                                         contract: ExpectedStreamContract,
                                         relationship: JobDetailRelationshipDefinition,
                                         parent_city_column: str = "city") -> PerStreamScheduledCoverageReport:
    """Assign every parent job to one city-period and evaluate each stream on its own periods.

    ``jobs`` / ``cars`` are the linked analysis frames for ``relationship``.
    Source frames are never modified. ``cars.scraped_at`` is not read.
    """
    if not isinstance(schedule, PerStreamSchedule):
        raise TypeError("schedule must be a PerStreamSchedule")
    if not isinstance(contract, ExpectedStreamContract):
        raise TypeError("contract must be an ExpectedStreamContract")
    coverage = contract.coverage if contract.coverage.is_configured else None
    if not schedule.available:
        return PerStreamScheduledCoverageReport(schedule=schedule, coverage=coverage)
    finished_col = _column(schedule.capture_field)
    copy_col = _column(schedule.detail_copy_field)
    pkeys, dkeys = list(relationship.parent_key_columns), list(relationship.detail_key_columns)
    location_cols = list(contract.coverage.location_columns)
    needed = ((jobs, pkeys + [parent_city_column, finished_col]), (cars, dkeys + location_cols + [copy_col]))
    for frame, columns in needed:
        if not isinstance(frame, pd.DataFrame) or any(c not in frame.columns for c in columns):
            raise ScheduleConfigurationError("a required schedule column is missing")

    zones = schedule.timezones
    city_periods: dict[str, dict[dt.datetime, ScheduledPeriod]] = {}
    for s in schedule.schedules:
        for p in s.periods:
            city_periods.setdefault(s.city, {})[p.utc_start] = p
    bounds = {c: (min(ps), max(ps)) for c, ps in city_periods.items()}

    failures: Counter[JobAssignmentFailure] = Counter()
    assigned: dict[tuple, tuple[str, dt.datetime, NormalizedFinish]] = {}
    job_rows = jobs.loc[:, pkeys + [parent_city_column, finished_col]]
    for row in job_rows.itertuples(index=False, name=None):
        key, city, raw = tuple(row[:len(pkeys)]), row[len(pkeys)], row[len(pkeys) + 1]
        try:
            finish = normalize_finished_at(raw, city, zones)
            local_period = finish.local.replace(minute=0, second=0, microsecond=0)
            zone = ZoneInfo(finish.timezone)
            if _local_kind(local_period, zone) != "ok":
                raise _AssignmentError(JobAssignmentFailure.NO_EXPECTED_PERIOD)
            period_utc = local_period.replace(tzinfo=zone).astimezone(_UTC)
            if city not in city_periods:
                raise _AssignmentError(JobAssignmentFailure.NO_EXPECTED_PERIOD)
            if period_utc not in city_periods[city]:
                low, high = bounds[city]
                raise _AssignmentError(JobAssignmentFailure.OUTSIDE_SCHEDULE_WINDOW if not low <= period_utc <= high
                                       else JobAssignmentFailure.NO_EXPECTED_PERIOD)
        except _AssignmentError as exc:
            failures[exc.kind] += 1
            continue
        assigned[key] = (city, period_utc, finish)

    # Detail copies must agree with their linked parent (parsed exactly; scraped_at is not read).
    detail = cars.loc[:, dkeys + location_cols + [copy_col]]
    present: dict[tuple, set[Key]] = {}
    mismatched_jobs: set[tuple] = set()
    mismatches = 0
    for row in detail.itertuples(index=False, name=None):
        key = tuple(row[:len(dkeys)])
        if key not in assigned:
            continue
        stream_key = tuple(row[len(dkeys):len(dkeys) + len(location_cols)])
        copy = row[-1]
        parent = assigned[key][2]
        try:
            agrees = isinstance(copy, str) and dt.datetime.strptime(copy, FINISHED_AT_SOURCE_FORMAT) == parent.local
        except ValueError:
            agrees = False
        if not agrees:
            mismatches += 1
            mismatched_jobs.add(key)
        if all(isinstance(v, str) for v in stream_key):
            present.setdefault(key, set()).add(stream_key)
    for key in mismatched_jobs:
        failures[JobAssignmentFailure.DETAIL_COPY_MISMATCH] += 1

    claims: dict[tuple[str, dt.datetime], list[tuple]] = {}
    for key, (city, period_utc, _) in assigned.items():
        claims.setdefault((city, period_utc), []).append(key)
    for jobs_in_period in claims.values():
        if len(jobs_in_period) > 1:
            failures[JobAssignmentFailure.DUPLICATE_CITY_PERIOD] += len(jobs_in_period)

    # Each parent-capture exclusion must resolve to exactly one valid parent capture of its city-period.
    resolved: dict[tuple[str, str], tuple] = {}
    unmatched = 0
    for exclusion in schedule.exceptions.parent_capture_exclusions:
        period = next(p for p in schedule.schedule_for(exclusion.streams[0]).periods
                      if p.utc_text == exclusion.period_start_utc)
        claimants = claims.get((exclusion.city, period.utc_start), [])
        if len(claimants) == 1 and claimants[0] not in mismatched_jobs:
            resolved[(exclusion.city, exclusion.period_start_utc)] = claimants[0]
        else:
            unmatched += 1

    streams = []
    for s in schedule.schedules:
        covered, missing, excluded = 0, [], []
        for p in s.periods:
            claimants = claims.get((s.city, p.utc_start), [])
            exclusion = schedule.exceptions.excluding(s.stream, p, schedule.schedule_version)
            if exclusion is not None and (s.city, p.utc_text) in resolved:
                excluded.append(ExcludedStreamPeriod(period=p, exclusion=exclusion))   # analytically null
                continue
            if len(claimants) > 1:
                failure = ScheduleFailureKind.PARENT_JOB_AMBIGUOUS
            elif not claimants:
                failure = ScheduleFailureKind.PARENT_JOB_ABSENT
            elif claimants[0] in mismatched_jobs:
                failure = ScheduleFailureKind.PARENT_JOB_INVALID
            elif s.stream in present.get(claimants[0], set()):
                covered += 1
                continue
            else:
                failure = ScheduleFailureKind.STREAM_ABSENT_FROM_CAPTURE
            excused = (failure in EXCUSABLE_FAILURES
                       and schedule.exceptions.excuses(s.stream, p, failure, schedule.schedule_version))
            missing.append(MissingStreamPeriod(period=p, failure=failure, excused=excused))
        streams.append(StreamPeriodCoverage(stream=s.stream, expected=s.period_count, covered=covered,
                                            missing=tuple(missing), excluded=tuple(excluded)))

    expected_city = {c: len(ps) for c, ps in sorted(city_periods.items())}
    missing_city = {c: sum(1 for u in ps if len(claims.get((c, u), [])) != 1) for c, ps in sorted(city_periods.items())}
    return PerStreamScheduledCoverageReport(
        schedule=schedule, coverage=coverage, streams=tuple(streams), jobs_assessed=len(job_rows),
        jobs_assigned=sum(1 for v in claims.values() if len(v) == 1 and v[0] not in mismatched_jobs),
        job_failures=tuple((k, failures[k]) for k in JobAssignmentFailure if failures[k]),
        detail_copy_mismatches=mismatches, expected_city_periods=tuple(expected_city.items()),
        missing_city_periods=tuple(missing_city.items()), unmatched_exclusions=unmatched,
        capture_periods=CapturePeriodIndex(
            parent_key_columns=tuple(pkeys), detail_key_columns=tuple(dkeys),
            periods={key: city_periods[city][period_utc].utc_text for key, (city, period_utc, _) in assigned.items()
                     if len(claims[(city, period_utc)]) == 1 and key not in mismatched_jobs}),
        capture_exclusions=CaptureExclusionSet(
            parent_key_columns=tuple(pkeys), detail_key_columns=tuple(dkeys),
            entries=tuple(ExcludedCapture(parent_key=key, city=city, streams=next(
                e.streams for e in schedule.exceptions.parent_capture_exclusions
                if (e.city, e.period_start_utc) == (city, text)))
                for (city, text), key in sorted(resolved.items()))))
