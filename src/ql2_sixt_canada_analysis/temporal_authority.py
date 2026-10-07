"""Authority-backed temporal policy: city-local finish times and scrape/finish ordering.

Built only from APPROVED decisions of the current authority record (schema 3,
``pricing-authorities-v6`` onwards for the ordering):

* ``FINISHED_AT_TIMEZONE`` - the exhaustive :class:`~ql2_sixt_canada_analysis.schemas.CityTimezoneMap`
  (the same map the per-stream schedule uses). ``jobs.finished_at`` and its
  detail copy ``cars.job_finished_at`` are naive wall-clock values in the zone
  of the exact **parent** ``jobs.city``
  (:data:`~ql2_sixt_canada_analysis.schemas.FINISHED_AT_TIMEZONE_SELECTOR`).
* ``SCRAPED_FINISHED_ORDERING`` + ``SCRAPED_FINISHED_TOLERANCE`` - for every
  trusted linked detail row ``cars.scraped_at <= jobs.finished_at`` on
  full-precision UTC instants, equality allowed, with the approved tolerance
  (zero seconds). The ordering is configured only when both are approved
  and the city map is available.

``cars.scraped_at`` keeps its established designator policy (``MST`` = fixed
UTC-07:00); it is never reinterpreted as market-local time.

Reporting day and source dates (schema 4, ``pricing-authorities-v8`` onwards):

* ``REPORTING_DAY_SOURCE`` = ``jobs.finished_at`` and ``REPORTING_DAY_TIMEZONE``
  mode ``PARENT_CITY`` with an exhaustive city map: the reporting day is the
  local calendar date of the parent finish instant in the parent city's zone
  (detail rows: their trusted linked parent's, never ``scraped_at``).
* ``SCRAPE_DATE_SEMANTICS`` = ``REPORTING_DAY``: ``jobs.scrape_date`` and
  ``cars.scrape_date`` are strictly parsed ``ISO_8601_DATE`` values that must
  equal that reporting day (date checks configured; no repair).
* ``DATE_CLEAN_SEMANTICS`` = ``RETIRED_FROM_PRICING``: ``cars.date_clean`` is
  kept, parsed and reported for presence and parse quality only; it has no
  date check, never blocks and is never a pricing date.

Until approved, the date derivation rules stay ``UNAVAILABLE`` and
``cars.date_clean`` is never a pricing date
(:attr:`TemporalAuthority.pricing_date_fields`). Observed equality between
date fields never creates authority.

Nothing here reads source rows; reports hold statuses and names only.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache

from ql2_sixt_canada_analysis.authority_decisions import (
    AuthorityDecisionRecord,
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
    ANALYSIS_TEMPORAL_RECONCILIATION,
    FINISHED_AT_TIMEZONE_SELECTOR,
    ISO_8601_DATE_FORMAT,
    CityTimezoneMap,
    DatasetKey,
    ReportingDateRule,
    TemporalDateCheck,
    TemporalAwareness,
    TemporalConfigurationError,
    TemporalKind,
    TemporalReconciliationDefinition,
    TimestampOrderingRule,
)


__all__ = [
    "DATE_CLEAN_FIELD",
    "FINISHED_AT_FIELD",
    "SCRAPE_DATE_FIELDS",
    "ORDERING_EARLIER_FIELD",
    "ORDERING_LATER_FIELD",
    "TEMPORAL_DECISIONS",
    "TemporalAuthority",
    "TemporalAuthorityBlocker",
    "TemporalDecisionStatus",
    "current_temporal_authority",
    "current_temporal_reconciliation",
    "temporal_authority_from_record",
]

D = DecisionId
#: The parent finish-time field whose zone the city map decides (its replicas follow it).
FINISHED_AT_FIELD = (DatasetKey.JOBS, "finished_at")
ORDERING_EARLIER_FIELD, ORDERING_LATER_FIELD = (DatasetKey.CARS, "scraped_at"), (DatasetKey.JOBS, "finished_at")
#: The source scrape dates that must equal the reporting day, and the retired cleaned date.
SCRAPE_DATE_FIELDS = ((DatasetKey.JOBS, "scrape_date"), (DatasetKey.CARS, "scrape_date"))
DATE_CLEAN_FIELD = (DatasetKey.CARS, "date_clean")
#: The temporal decisions this module reads.
TEMPORAL_DECISIONS = (D.FINISHED_AT_TIMEZONE, D.SCRAPED_FINISHED_ORDERING, D.SCRAPED_FINISHED_TOLERANCE,
                      D.REPORTING_DAY_SOURCE, D.REPORTING_DAY_TIMEZONE, D.SCRAPE_DATE_SEMANTICS,
                      D.DATE_CLEAN_SEMANTICS)
_UNITS = {"SECONDS": "seconds", "MINUTES": "minutes", "HOURS": "hours"}


class TemporalDecisionStatus(StrEnum):
    APPROVED = "approved"                    # approved and implementable
    NOT_APPROVED = "not_approved"            # proposed or rejected (or a prerequisite is not approved)
    RECORD_UNAVAILABLE = "record_unavailable"
    INVALID = "invalid"                      # approved but contradicting the implemented contract


class TemporalAuthorityBlocker(StrEnum):
    """Why the temporal policy is not fully authority-backed (statuses, never values)."""

    FINISHED_AT_TIMEZONE_UNAVAILABLE = "finished_at_timezone_unavailable"
    TIMESTAMP_ORDERING_UNAVAILABLE = "timestamp_ordering_unavailable"
    REPORTING_DAY_UNRESOLVED = "reporting_day_unresolved"
    DATE_SEMANTICS_UNRESOLVED = "date_semantics_unresolved"


@dataclass(frozen=True)
class TemporalAuthority:
    """The effective temporal contract and the authority statuses it rests on."""

    record_id: str | None
    timezone_status: TemporalDecisionStatus
    ordering_status: TemporalDecisionStatus
    tolerance_status: TemporalDecisionStatus
    reporting_day_status: TemporalDecisionStatus
    date_semantics_status: TemporalDecisionStatus
    definition: TemporalReconciliationDefinition
    city_timezones: CityTimezoneMap | None = None
    ordering: TimestampOrderingRule | None = None
    tolerance_seconds: int | None = None
    reporting_day_source: str | None = None
    references: tuple[str, ...] = ()
    #: Per-city reporting-day zones (``REPORTING_DAY_TIMEZONE`` mode ``PARENT_CITY``) when approved.
    reporting_day_timezones: CityTimezoneMap | None = None
    #: Approved derivations (``REPORTING_DAY``, ``RETIRED_FROM_PRICING``...) or ``None``.
    scrape_date_derivation: str | None = None
    date_clean_derivation: str | None = None

    def __post_init__(self) -> None:
        if (self.timezone_status is TemporalDecisionStatus.APPROVED) != (self.city_timezones is not None):
            raise ValueError("a city map exactly when the timezone decision is approved")
        if (self.ordering_status is TemporalDecisionStatus.APPROVED) != (self.ordering is not None):
            raise ValueError("an ordering rule exactly when the ordering is approved")
        if self.definition.ordering != self.ordering:
            raise ValueError("the definition must carry exactly the approved ordering")

    @property
    def finished_at_timezone_available(self) -> bool:
        return self.timezone_status is TemporalDecisionStatus.APPROVED

    @property
    def timestamp_ordering_available(self) -> bool:
        return self.ordering_status is TemporalDecisionStatus.APPROVED

    @property
    def pricing_date_fields(self) -> tuple[str, ...]:
        """Fields usable as a pricing (reporting) date: only an approved reporting-day source.

        Empty while the reporting-day decisions are unresolved, so neither
        ``cars.date_clean`` nor any other date field is a trusted pricing date;
        ``date_clean`` qualifies only if an authority names it the source.
        """
        return (self.reporting_day_source,) if self.reporting_day_source is not None else ()

    @property
    def reporting_day_available(self) -> bool:
        return (self.reporting_day_status is TemporalDecisionStatus.APPROVED
                and self.date_semantics_status is TemporalDecisionStatus.APPROVED)

    @property
    def date_clean_retired(self) -> bool:
        return self.date_clean_derivation == "RETIRED_FROM_PRICING"

    @property
    def blocking_reasons(self) -> tuple[TemporalAuthorityBlocker, ...]:
        B = TemporalAuthorityBlocker
        found = []
        if not self.finished_at_timezone_available:
            found.append(B.FINISHED_AT_TIMEZONE_UNAVAILABLE)
        if not self.timestamp_ordering_available:
            found.append(B.TIMESTAMP_ORDERING_UNAVAILABLE)
        if self.reporting_day_status is not TemporalDecisionStatus.APPROVED:
            found.append(B.REPORTING_DAY_UNRESOLVED)
        if self.date_semantics_status is not TemporalDecisionStatus.APPROVED:
            found.append(B.DATE_SEMANTICS_UNRESOLVED)
        return tuple(found)


def _status(record: AuthorityDecisionRecord, *decisions: DecisionId) -> TemporalDecisionStatus:
    approved = all(record.decision(d).status is DecisionStatus.APPROVED for d in decisions)
    return TemporalDecisionStatus.APPROVED if approved else TemporalDecisionStatus.NOT_APPROVED


def _ref(text: str) -> tuple[DatasetKey, str]:
    dataset, column = text.split(".", 1)
    return (DatasetKey(dataset), column)


def _with_city_zones(template: TemporalReconciliationDefinition, zones: CityTimezoneMap
                     ) -> TemporalReconciliationDefinition:
    """``finished_at`` and every detail replica of it resolve through the parent-city map."""
    targets = {FINISHED_AT_FIELD} | {tuple(r.replica) for r in template.replications
                                    if tuple(r.source) == FINISHED_AT_FIELD}
    fields = []
    for field in template.fields:
        if field.ref in targets:
            if field.awareness is not TemporalAwareness.NAIVE or field.kind is not TemporalKind.TIMESTAMP:
                raise TemporalConfigurationError("the finish time and its replicas must be naive timestamps")
            field = replace(field, source_timezone=None, city_timezones=zones,
                            timezone_selector=FINISHED_AT_TIMEZONE_SELECTOR)
        fields.append(field)
    return replace(template, fields=tuple(fields))


def temporal_authority_from_record(record: AuthorityDecisionRecord | None,
                                   template: TemporalReconciliationDefinition = ANALYSIS_TEMPORAL_RECONCILIATION,
                                   contract: ExpectedStreamContract | None = None) -> TemporalAuthority:
    """Build the effective temporal contract from APPROVED decisions only (fails closed)."""
    S = TemporalDecisionStatus
    if record is None:
        return TemporalAuthority(None, S.RECORD_UNAVAILABLE, S.RECORD_UNAVAILABLE, S.RECORD_UNAVAILABLE,
                                 S.RECORD_UNAVAILABLE, S.RECORD_UNAVAILABLE, replace(template, ordering=None))
    reporting = _status(record, D.REPORTING_DAY_SOURCE, D.REPORTING_DAY_TIMEZONE)
    semantics = _status(record, D.SCRAPE_DATE_SEMANTICS, D.DATE_CLEAN_SEMANTICS)
    reporting_source = (record.approved_resolution(D.REPORTING_DAY_SOURCE)["field"]
                        if reporting is S.APPROVED else None)
    definition, zones = replace(template, ordering=None), None
    references: list[str] = []

    timezone = _status(record, D.FINISHED_AT_TIMEZONE)
    if timezone is S.APPROVED:
        resolution = record.approved_resolution(D.FINISHED_AT_TIMEZONE)
        try:
            if record.schema_version < 3 or "city_timezones" not in resolution:
                raise TemporalConfigurationError("a single global zone cannot resolve city-local finish times")
            zones = CityTimezoneMap(tuple(city_timezones(resolution).items()))
            if contract is not None and contract.status is ExpectedStreamAuthorityStatus.APPROVED and set(
                    zones.cities) != {k[0] for k in contract.expected_keys}:
                raise TemporalConfigurationError("the city map must cover exactly the approved cities")
            definition = _with_city_zones(definition, zones)
            references += [a.reference for a in record.decision(D.FINISHED_AT_TIMEZONE).authority]
        except (TemporalConfigurationError, ValueError, KeyError):
            timezone, zones, definition = S.INVALID, None, replace(template, ordering=None)

    ordering_status = _status(record, D.SCRAPED_FINISHED_ORDERING)
    tolerance_status = _status(record, D.SCRAPED_FINISHED_TOLERANCE)
    rule, tolerance_seconds = None, None
    if ordering_status is S.APPROVED and tolerance_status is S.APPROVED and timezone is S.APPROVED:
        order = record.approved_resolution(D.SCRAPED_FINISHED_ORDERING)
        tol = record.approved_resolution(D.SCRAPED_FINISHED_TOLERANCE)
        try:
            if (_ref(order["earlier"]), _ref(order["later"])) != (ORDERING_EARLIER_FIELD, ORDERING_LATER_FIELD):
                raise TemporalConfigurationError("only scrape time not after finish time is implemented")
            delta = dt.timedelta(**{_UNITS[tol["unit"]]: tol["tolerance"]})
            rule = TimestampOrderingRule(earlier=ORDERING_EARLIER_FIELD, later=ORDERING_LATER_FIELD,
                                         inclusive=bool(order["equal_allowed"]), tolerance=delta)
            definition = replace(definition, ordering=rule)
            tolerance_seconds = int(delta.total_seconds())
            references += [a.reference for d in (D.SCRAPED_FINISHED_ORDERING, D.SCRAPED_FINISHED_TOLERANCE)
                           for a in record.decision(d).authority]
        except (TemporalConfigurationError, ValueError, KeyError):
            ordering_status = tolerance_status = S.INVALID
            rule = None
    elif ordering_status is S.APPROVED:          # approved but not implementable without tolerance and zones
        ordering_status = S.NOT_APPROVED

    day_zones, scrape_derivation, clean_derivation = None, None, None
    if semantics is S.APPROVED:
        scrape_derivation = record.approved_resolution(D.SCRAPE_DATE_SEMANTICS)["derivation"]
        clean_derivation = record.approved_resolution(D.DATE_CLEAN_SEMANTICS)["derivation"]
    if reporting is S.APPROVED and semantics is S.APPROVED:   # configured only when every date decision is approved
        try:
            definition, day_zones = _with_reporting_day(record, definition, timezone, contract)
            references += [a.reference for d in (D.REPORTING_DAY_SOURCE, D.REPORTING_DAY_TIMEZONE,
                                                 D.SCRAPE_DATE_SEMANTICS, D.DATE_CLEAN_SEMANTICS)
                           for a in record.decision(d).authority]
        except (TemporalConfigurationError, ValueError, KeyError, TypeError):
            reporting = semantics = S.INVALID
            reporting_source = None
            day_zones = scrape_derivation = clean_derivation = None
            definition = replace(definition, date_checks=template.date_checks, retired_fields=())
    return TemporalAuthority(
        record_id=record.record_id, timezone_status=timezone, ordering_status=ordering_status,
        tolerance_status=tolerance_status, reporting_day_status=reporting, date_semantics_status=semantics,
        definition=definition, city_timezones=zones, ordering=rule, tolerance_seconds=tolerance_seconds,
        reporting_day_source=reporting_source, references=tuple(dict.fromkeys(references)),
        reporting_day_timezones=day_zones, scrape_date_derivation=scrape_derivation,
        date_clean_derivation=clean_derivation)


def _with_reporting_day(record: AuthorityDecisionRecord, definition: TemporalReconciliationDefinition,
                        timezone: TemporalDecisionStatus, contract: ExpectedStreamContract | None
                        ) -> tuple[TemporalReconciliationDefinition, CityTimezoneMap]:
    """Configure the reporting-day date checks and retire ``date_clean`` (fails closed on anything else)."""
    if timezone is not TemporalDecisionStatus.APPROVED:
        raise TemporalConfigurationError("the reporting day needs the approved finish-time zones")
    if record.schema_version < 4:
        raise TemporalConfigurationError("only the schema-4 per-city reporting day is implemented")
    source = _ref(record.approved_resolution(D.REPORTING_DAY_SOURCE)["field"])
    tz = record.approved_resolution(D.REPORTING_DAY_TIMEZONE)
    if source != FINISHED_AT_FIELD or tz["mode"] != "PARENT_CITY":
        raise TemporalConfigurationError("only the parent finish time in the parent city's zone is implemented")
    day_zones = CityTimezoneMap(tuple(city_timezones(tz).items()))
    if contract is not None and contract.status is ExpectedStreamAuthorityStatus.APPROVED and set(
            day_zones.cities) != {k[0] for k in contract.expected_keys}:
        raise TemporalConfigurationError("the reporting-day map must cover exactly the approved cities")
    if record.approved_resolution(D.SCRAPE_DATE_SEMANTICS)["derivation"] != "REPORTING_DAY":
        raise TemporalConfigurationError("only scrape dates that equal the reporting day are implemented")
    if record.approved_resolution(D.DATE_CLEAN_SEMANTICS)["derivation"] != "RETIRED_FROM_PRICING":
        raise TemporalConfigurationError("only a date_clean retired from pricing is implemented")
    rule = ReportingDateRule(source=FINISHED_AT_FIELD, city_timezones=day_zones,
                             timezone_selector=FINISHED_AT_TIMEZONE_SELECTOR)
    strict = set(SCRAPE_DATE_FIELDS) | {DATE_CLEAN_FIELD}
    fields = tuple(replace(f, source_format=ISO_8601_DATE_FORMAT) if f.ref in strict else f
                   for f in definition.fields)
    checks = tuple(TemporalDateCheck(target, rule) for target in SCRAPE_DATE_FIELDS)
    return replace(definition, fields=fields, date_checks=checks, retired_fields=(DATE_CLEAN_FIELD,)), day_zones


@cache
def current_temporal_authority() -> TemporalAuthority:
    """The analysis-stage temporal policy from the current committed record (cached)."""
    return temporal_authority_from_record(load_current_decision_record(), ANALYSIS_TEMPORAL_RECONCILIATION,
                                          current_expected_stream_contract())


def current_temporal_reconciliation() -> TemporalReconciliationDefinition:
    """The authority-backed analysis-stage temporal contract (``current_temporal_authority().definition``)."""
    return current_temporal_authority().definition
