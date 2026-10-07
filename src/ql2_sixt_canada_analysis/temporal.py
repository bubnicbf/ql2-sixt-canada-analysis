"""Parse and reconcile the source temporal fields against the central contract.

The template contract is :data:`~ql2_sixt_canada_analysis.schemas.TEMPORAL_RECONCILIATION`;
the authority-backed contract (city-local finish times, the approved ordering)
is :func:`~ql2_sixt_canada_analysis.temporal_authority.current_temporal_reconciliation`.

Parsing policy (:func:`parse_temporal_field`)
--------------------------------------------
* **Missing** - a pandas missing value, or an empty / whitespace-only string.
* **Invalid** - any other value that does not match the field's format and
  awareness: unparsable text, impossible dates, text with surrounding
  whitespace, an offset on a naive field, a time on a date field (midnight
  timestamps are *not* accepted as dates), numbers and other objects
  (numeric timestamps are not a supported source format).
* **Unresolved** - a valid timestamp that cannot become an absolute instant
  without guessing: a naive value with no authoritative ``source_timezone``,
  a naive value that is ambiguous or nonexistent in that zone (daylight-time
  transitions), an offset field value without an offset, or an unknown zone
  designator. Unresolved values are never assigned the machine's zone or UTC.
* Valid instants are converted to the canonical zone (UTC); dates are parsed
  to semantic calendar dates (zero padding is optional; no time part).
* **City-local naive fields** (``city_timezones`` + ``timezone_selector``):
  the zone of each row is selected by the exact value of the parent job's
  city (detail rows: their linked parent's city, never their own city or
  location label) through the approved :class:`CityTimezoneMap`. Unresolved
  values are split into *unknown city* (no approved city), *context
  unavailable* (no linked parent), *ambiguous* (a repeated fall-back hour)
  and *nonexistent* (a spring-forward gap); no occurrence is chosen and
  nothing is shifted. Instants keep the full parsed (sub-second) precision;
  the ``YYYYMMDDTHHMMSSZ`` text (:func:`canonical_utc_text`) is a
  presentation form only.

Rules (:func:`assess_temporal_reconciliation`)
---------------------------------------------
* **Ordering** - ``earlier <= later`` within an explicit, non-negative
  tolerance (``later - earlier >= -tolerance``; strict when exclusive).
* **Date derivation** - the date equals the calendar date of the source
  instant *after* conversion to the rule's reporting zone. Rows where that
  date differs from the UTC date but match are counted as legitimate
  boundary crossings.
* **Replication** - a detail-row copy equals its parent's value (instants, or
  wall times when both sides are naive on the same basis). When both sides
  resolve on the same zone basis, the wall times **and** the full-precision
  UTC instants must both agree.
* **City integrity prerequisite** - a linked detail row is *trusted* only when
  its scope columns (``city``) equal its parent's exactly
  (``relationship.scope_agreement_columns``); mismatched rows are counted
  (``city_mismatch_detail_row_count``) and are unassessable for replication
  and ordering. Ordering is assessed only on trusted rows whose replication
  holds.

Detail rows are linked to exactly one parent through the central
relationship (tuple keys, no concatenation, no many-to-many join); missing
and orphan links are counted as unassessable, never dropped. A rule whose
semantics are not established is ``UNAVAILABLE``: reported, never passed,
and strict validation fails closed. Nothing modifies, repairs, writes or logs
data; reports hold names of configured fields/rules, enums, ints and bools
only - never temporal values or identifiers.

Empty-data policy: empty frames have no contradictions (presence is a
separate control); configuration must still be valid and unavailable rules
still fail closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from enum import StrEnum
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.relationships import RelationshipPreconditionError, _check_relationship_inputs
from ql2_sixt_canada_analysis.schemas import (
    ISO_8601_DATE_FORMAT,
    TEMPORAL_RECONCILIATION,
    DatasetKey,
    ReportingDateRule,
    TemporalAwareness,
    TemporalConfigurationError,
    TemporalFieldDefinition,
    TemporalKind,
    TemporalReconciliationDefinition,
    classify_local_time,
)

__all__ = [
    "RuleStatus",
    "TemporalConfigurationError",
    "TemporalFieldReport",
    "TemporalParseError",
    "TemporalParseResult",
    "TemporalPreconditionError",
    "TemporalReconciliationError",
    "TemporalReconciliationReport",
    "TemporalRuleReport",
    "UTC_CANONICAL_FORMAT",
    "DerivedTimestamps",
    "DerivedReportingDays",
    "ScrapeDateStatus",
    "derive_reporting_days",
    "assess_temporal_reconciliation",
    "canonical_utc_text",
    "derive_utc_timestamps",
    "parse_temporal_field",
    "validate_temporal_reconciliation",
]

_OFFSET_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_DESIGNATOR = re.compile(r"^(?P<body>.+) (?P<zone>[A-Z]+)$")
_STRICT_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)
#: Canonical serialized presentation of a UTC instant (whole seconds; never an identity).
UTC_CANONICAL_FORMAT = "%Y%m%dT%H%M%SZ"


# ------------------------------------------------------------------- parsing


@dataclass(frozen=True, slots=True)
class TemporalParseResult:
    """In-memory parse result for one field (never part of a report).

    ``instants`` are canonical-zone timestamps (``NaT`` unless resolved);
    ``wall`` are naive wall times (timestamps) or midnight-normalised dates;
    masks are aligned with the source rows.
    """

    instants: pd.Series
    wall: pd.Series
    missing: np.ndarray
    invalid: np.ndarray
    unresolved: np.ndarray
    #: Unresolved breakdown for city-local fields (all-False otherwise).
    unknown_city: np.ndarray | None = None
    context_unavailable: np.ndarray | None = None
    ambiguous: np.ndarray | None = None
    nonexistent: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("unknown_city", "context_unavailable", "ambiguous", "nonexistent"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.zeros(len(self.missing), dtype=bool))

    @property
    def valid(self) -> np.ndarray:
        return ~(self.missing | self.invalid)

    @property
    def resolved(self) -> np.ndarray:
        return self.instants.notna().to_numpy()


def parse_temporal_field(
    series: pd.Series, field: TemporalFieldDefinition, canonical_timezone: str = "UTC", *,
    city_values: pd.Series | None = None, context_available: np.ndarray | None = None,
) -> TemporalParseResult:
    """Parse ``series`` according to ``field`` without modifying it (see module docstring).

    For a city-local field, ``city_values`` (aligned with ``series``) are the
    exact selector values - the parent city of each row - and
    ``context_available`` marks rows that have that parent context (default:
    all). Without ``city_values`` every value is unresolved (context
    unavailable); a default zone is never assumed.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not isinstance(field, TemporalFieldDefinition):
        raise TypeError("field must be a TemporalFieldDefinition")
    index = series.index
    values = series.astype(object)
    is_text = values.map(lambda v: isinstance(v, str)).to_numpy(dtype=bool)
    text = pd.Series(np.where(is_text, values, None), index=index, dtype=object)
    blank = text.map(lambda v: isinstance(v, str) and not v.strip()).to_numpy(dtype=bool)
    missing = series.isna().to_numpy(dtype=bool) | blank
    candidate = is_text & ~missing
    other = ~is_text & ~missing                          # numbers/objects: unsupported -> invalid

    def done(instants: pd.Series, wall: pd.Series, invalid: np.ndarray, unresolved_: np.ndarray
             ) -> TemporalParseResult:
        return TemporalParseResult(instants, wall, missing, invalid | other, unresolved_)
    nat = pd.Series(pd.NaT, index=index, dtype=f"datetime64[ns, {canonical_timezone}]")
    unresolved = np.zeros(len(series), dtype=bool)
    usable = text.where(candidate)

    if field.kind is TemporalKind.DATE:
        if field.source_format == ISO_8601_DATE_FORMAT:     # exact YYYY-MM-DD text naming a real date
            exact = usable.map(lambda v: isinstance(v, str) and bool(_STRICT_ISO_DATE.fullmatch(v)))
            wall = pd.to_datetime(usable.where(exact.astype(bool)), format="%Y-%m-%d", errors="coerce")
            parsed = wall.notna().to_numpy() & candidate
            return done(nat, wall, candidate & ~parsed, unresolved)
        wall = pd.to_datetime(usable, format=field.source_format, errors="coerce")
        parsed = wall.notna().to_numpy()
        return done(nat, wall, candidate & ~parsed, unresolved)

    if field.awareness is TemporalAwareness.OFFSET:
        has_offset = usable.map(lambda v: isinstance(v, str) and bool(_OFFSET_SUFFIX.search(v))).to_numpy(dtype=bool)
        aware = pd.to_datetime(usable.where(has_offset), format=field.source_format, utc=True, errors="coerce")
        naive_text = usable.where(candidate & ~has_offset)
        naive_ok = pd.to_datetime(naive_text, format=field.source_format, errors="coerce").notna().to_numpy()
        parsed_aware = aware.notna().to_numpy()
        unresolved = candidate & ~has_offset & naive_ok          # mixed awareness: no offset
        invalid = candidate & ~(parsed_aware | unresolved)
        instants = aware.dt.tz_convert(canonical_timezone)
        wall = aware.dt.tz_localize(None)
        return done(instants, wall, invalid, unresolved)

    if field.awareness is TemporalAwareness.DESIGNATOR:
        parts = usable.str.extract(_DESIGNATOR)
        wall = pd.to_datetime(parts["body"], format=field.source_format, errors="coerce")
        parsed = wall.notna().to_numpy() & candidate
        offsets = parts["zone"].map(lambda z: field.designator_offsets.get(z) if isinstance(z, str) else None)
        known = offsets.notna().to_numpy() & parsed
        delta = pd.to_timedelta(offsets.where(known))
        instants = (wall.where(known).dt.tz_localize("UTC") - delta).dt.tz_convert(canonical_timezone)
        return done(instants, wall, candidate & ~parsed, parsed & ~known)

    # NAIVE
    wall = pd.to_datetime(usable, format=field.source_format, errors="coerce")
    parsed = wall.notna().to_numpy() & candidate
    if field.city_timezones is not None:
        return _city_local(field, wall, parsed, canonical_timezone, missing, candidate & ~parsed, other,
                           city_values, context_available)
    if field.source_timezone is None:
        return done(nat, wall, candidate & ~parsed, parsed.copy())
    local = wall.dt.tz_localize(field.source_timezone, ambiguous="NaT", nonexistent="NaT")
    resolved = local.notna().to_numpy()
    instants = local.dt.tz_convert(canonical_timezone)
    return done(instants, wall, candidate & ~parsed, parsed & ~resolved)


def _city_local(field, wall, parsed, canonical_timezone, missing, invalid, other,  # type: ignore[no-untyped-def]
                city_values, context_available) -> TemporalParseResult:
    """Resolve each parsed wall time in the zone of its exact parent city (fails closed per row)."""
    n = len(wall)
    if city_values is None:
        context = np.zeros(n, dtype=bool)
        zones = [None] * n
    else:
        if len(city_values) != n:
            raise ValueError("city_values must align with the timestamp series")
        context = (np.ones(n, dtype=bool) if context_available is None
                   else np.asarray(context_available, dtype=bool).copy())
        zones = [field.city_timezones.zone_or_none(c) if ok else None
                 for c, ok in zip(city_values.astype(object).tolist(), context)]
    unknown, unavailable = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    ambiguous, nonexistent = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    out: list = [pd.NaT] * n
    cache: dict = {}
    walls = wall.tolist()
    for i in np.flatnonzero(parsed):
        if not context[i]:
            unavailable[i] = True
            continue
        zone_name = zones[i]
        if zone_name is None:
            unknown[i] = True
            continue
        key = (walls[i], zone_name)
        if key not in cache:
            local = walls[i].to_pydatetime()
            zone = ZoneInfo(zone_name)
            kind = classify_local_time(local, zone)
            cache[key] = (kind, pd.Timestamp(local.replace(tzinfo=zone)).tz_convert(canonical_timezone)
                          if kind == "ok" else pd.NaT)
        kind, instant = cache[key]
        if kind == "ambiguous":
            ambiguous[i] = True
        elif kind == "nonexistent":
            nonexistent[i] = True
        else:
            out[i] = instant
    instants = pd.to_datetime(pd.Series(out, index=wall.index, dtype=object), utc=True).astype(
        "datetime64[ns, UTC]").dt.tz_convert(canonical_timezone)
    unresolved = unknown | unavailable | ambiguous | nonexistent
    return TemporalParseResult(instants, wall, missing, invalid | other, unresolved, unknown, unavailable,
                               ambiguous, nonexistent)


def canonical_utc_text(instants: pd.Series) -> pd.Series:
    """``YYYYMMDDTHHMMSSZ`` presentation of aware instants (``None`` where unresolved).

    Whole seconds only: it is a serialization, never the timestamp identity -
    two distinct sub-second instants may share it, so comparisons always use
    the full-precision instants.
    """
    if not isinstance(instants, pd.Series) or not isinstance(instants.dtype, pd.DatetimeTZDtype):
        raise TypeError("instants must be a timezone-aware datetime Series")
    utc = instants.dt.tz_convert("UTC")
    return utc.dt.strftime(UTC_CANONICAL_FORMAT).where(utc.notna(), None).astype(object)


# ------------------------------------------------------------------- reports


class RuleStatus(StrEnum):
    CONFIGURED = "configured"
    UNAVAILABLE = "unavailable"   # no authoritative semantics: never passes


@dataclass(frozen=True, slots=True)
class TemporalFieldReport:
    """Parse metrics for one configured field (counts only)."""

    dataset: DatasetKey
    column: str
    kind: TemporalKind
    required: bool
    row_count: int
    valid_count: int
    missing_count: int
    invalid_count: int
    unresolved_count: int
    unknown_city_count: int = 0
    context_unavailable_count: int = 0
    ambiguous_count: int = 0
    nonexistent_count: int = 0
    resolvable: bool = False
    #: Retired from pricing: reported for presence and parse quality only, never blocking.
    retired: bool = False

    def __post_init__(self) -> None:
        assert self.row_count == self.valid_count + self.missing_count + self.invalid_count
        assert 0 <= self.unresolved_count <= self.valid_count
        assert (self.unknown_city_count + self.context_unavailable_count + self.ambiguous_count
                + self.nonexistent_count) <= self.unresolved_count

    @property
    def resolved_count(self) -> int:
        """Values resolved to an instant (timestamps that can resolve); 0 otherwise."""
        return self.valid_count - self.unresolved_count if self.resolvable else 0

    @property
    def resolves(self) -> bool:
        """For a field that can resolve to instants: no valid value is left unresolved."""
        return not self.resolvable or self.unresolved_count == 0

    @property
    def ref(self) -> tuple[DatasetKey, str]:
        return (self.dataset, self.column)

    @property
    def parse_quality_clean(self) -> bool:
        """No invalid values, and no missing values for a required field (reported even when retired)."""
        return self.invalid_count == 0 and (not self.required or self.missing_count == 0)

    @property
    def parses(self) -> bool:
        """The parse-quality gate: a retired field never blocks (its quality is still reported)."""
        return self.retired or self.parse_quality_clean


@dataclass(frozen=True, slots=True)
class TemporalRuleReport:
    """Outcome counts for one rule over its comparison rows.

    ``unassessable`` covers unlinked rows and rows whose inputs are missing,
    invalid or unresolved. For an ``UNAVAILABLE`` rule every comparison row
    is unassessable. ``boundary_crossing`` counts matching date rows whose
    reporting date differs from the UTC date (date rules only).
    """

    name: str
    status: RuleStatus
    row_count: int
    passed: int
    failed: int
    unassessable: int
    boundary_crossing: int = 0

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name not in ("name", "status"):
                assert type(getattr(self, f.name)) is int and getattr(self, f.name) >= 0, f.name
        assert self.row_count == self.passed + self.failed + self.unassessable
        assert self.boundary_crossing <= self.passed
        if self.status is RuleStatus.UNAVAILABLE:
            assert self.passed == self.failed == 0

    @property
    def holds(self) -> bool:
        return self.status is RuleStatus.CONFIGURED and self.failed == 0 and self.unassessable == 0


@dataclass(frozen=True, slots=True)
class TemporalReconciliationReport:
    """Aggregate temporal reconciliation (no values, identifiers or rows)."""

    field_reports: tuple[TemporalFieldReport, ...]
    ordering: TemporalRuleReport
    date_checks: tuple[TemporalRuleReport, ...]
    replications: tuple[TemporalRuleReport, ...]
    parent_row_count: int
    detail_row_count: int
    unlinked_detail_row_count: int
    city_mismatch_detail_row_count: int = 0

    @property
    def all_required_fields_parse(self) -> bool:
        return all(r.parses for r in self.field_reports)

    @property
    def all_resolvable_fields_resolve(self) -> bool:
        """No unknown-city, context-less, ambiguous or nonexistent value in a field that can resolve."""
        return all(r.resolves for r in self.field_reports)

    @property
    def timestamp_ordering_valid(self) -> bool:
        return self.ordering.holds

    @property
    def date_derivation_valid(self) -> bool:
        return all(r.holds for r in self.date_checks)

    @property
    def replication_valid(self) -> bool:
        return all(r.holds for r in self.replications)

    @property
    def replication_failed_count(self) -> int:
        """Detail rows whose finish-time copy disagrees with the parent (wall time or full-precision instant)."""
        return sum(r.failed for r in self.replications)

    @property
    def unresolved_time_count(self) -> int:
        """Values of resolvable fields left without an instant (no approved zone, no parent, ambiguous, nonexistent)."""
        return sum(r.unresolved_count for r in self.field_reports if r.resolvable)
    @property
    def unavailable_rules(self) -> tuple[str, ...]:
        rules = (self.ordering, *self.date_checks, *self.replications)
        return tuple(r.name for r in rules if r.status is RuleStatus.UNAVAILABLE)

    @property
    def is_valid(self) -> bool:
        """Every field parses and every rule is configured and holds for all rows."""
        return (self.all_required_fields_parse and self.all_resolvable_fields_resolve
                and self.timestamp_ordering_valid and self.date_derivation_valid and self.replication_valid
                and self.unlinked_detail_row_count == 0 and self.city_mismatch_detail_row_count == 0)

    @property
    def violations(self) -> tuple[str, ...]:
        rules = (self.ordering, *self.date_checks, *self.replications)
        checks = (
            ("field_parse", not self.all_required_fields_parse),
            ("time_unresolved", not self.all_resolvable_fields_resolve),
            ("city_mismatch", self.city_mismatch_detail_row_count > 0),
            ("rule_unavailable", bool(self.unavailable_rules)),
            ("ordering", self.ordering.failed > 0),
            ("date_derivation", any(r.failed for r in self.date_checks)),
            ("replication", any(r.failed for r in self.replications)),
            ("unassessable", self.unlinked_detail_row_count > 0 or any(
                r.unassessable for r in rules if r.status is RuleStatus.CONFIGURED)),
        )
        return tuple(name for name, failed in checks if failed)


# ---------------------------------------------------------------- exceptions


class TemporalPreconditionError(RelationshipPreconditionError):
    """Relationship preconditions for temporal comparison failed."""

    control = "Temporal reconciliation"


class TemporalReconciliationError(Exception):
    """Strict validation found temporal violations or unavailable rules.

    ``violations`` lists categories (``field_parse``, ``rule_unavailable``,
    ``ordering``, ``date_derivation``, ``replication``, ``unassessable``);
    the aggregate report is on ``report``. No values appear in the message.
    """

    def __init__(self, report: TemporalReconciliationReport) -> None:
        super().__init__("Temporal reconciliation failed: " + ", ".join(report.violations) + ".")
        self.report = report
        self.violations = report.violations


class TemporalParseError(TemporalReconciliationError):
    """Strict validation failed and required temporal fields did not parse."""


# ------------------------------------------------------------------ public API


def assess_temporal_reconciliation(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    definition: TemporalReconciliationDefinition = TEMPORAL_RECONCILIATION,
) -> TemporalReconciliationReport:
    """Parse all configured fields and assess every rule; returns aggregates only.

    Raises:
        TypeError: Invalid argument types.
        TemporalConfigurationError: A configured column is absent.
        TemporalPreconditionError: The parent key is incomplete or duplicated,
            relationship keys are not identifier-typed, or blank rows remain.
    """
    if not isinstance(definition, TemporalReconciliationDefinition):
        raise TypeError("definition must be a TemporalReconciliationDefinition")
    rel = definition.relationship
    _check_relationship_inputs(jobs, cars, rel, TemporalPreconditionError)
    frames = {rel.parent: jobs, rel.detail: cars}
    absent = [f.column for f in definition.fields if f.column not in frames[f.dataset].columns]
    if absent:
        raise TemporalConfigurationError(f"{len(absent)} configured temporal column(s) are absent.")

    parsed, positions = _parse_fields(jobs, cars, definition)
    linked = positions >= 0
    take = np.where(linked, positions, 0)
    parent_values = _parent_lookup(jobs, cars, positions)
    retired_refs = {tuple(r) for r in definition.retired_fields}
    field_reports = tuple(
        TemporalFieldReport(
            dataset=f.dataset, column=f.column, kind=f.kind, required=f.required, row_count=len(frames[f.dataset]),
            valid_count=int(parsed[f.ref].valid.sum()), missing_count=int(parsed[f.ref].missing.sum()),
            invalid_count=int(parsed[f.ref].invalid.sum()), unresolved_count=int(parsed[f.ref].unresolved.sum()),
            unknown_city_count=int(parsed[f.ref].unknown_city.sum()),
            context_unavailable_count=int(parsed[f.ref].context_unavailable.sum()),
            ambiguous_count=int(parsed[f.ref].ambiguous.sum()),
            nonexistent_count=int(parsed[f.ref].nonexistent.sum()),
            resolvable=f.resolvable_to_instant, retired=f.ref in retired_refs,
        )
        for f in definition.fields
    )

    # City integrity prerequisite: a linked detail row is trusted only when its scope equals its parent's.
    trusted = _trusted_rows(cars, rel, linked, parent_values)
    city_mismatch = linked & ~trusted

    def aligned(ref: tuple[DatasetKey, str], base: DatasetKey) -> tuple[pd.Series, pd.Series, np.ndarray]:
        """(instants, wall, row-linked mask) of ``ref`` aligned to ``base`` rows."""
        result = parsed[ref]
        if ref[0] == base:
            n = len(frames[base])
            return (result.instants.reset_index(drop=True), result.wall.reset_index(drop=True),
                    np.ones(n, dtype=bool))
        inst = result.instants.iloc[take].reset_index(drop=True).where(pd.Series(linked))
        wall = result.wall.iloc[take].reset_index(drop=True).where(pd.Series(linked))
        return inst, wall, linked

    def base_of(*refs: tuple[DatasetKey, str]) -> DatasetKey:
        return rel.detail if any(r[0] == rel.detail for r in refs) else rel.parent

    # --- replication (first: ordering is assessed only where every replica agrees)
    replication_reports = []
    replicas_agree = trusted.copy()
    for rule in definition.replications:
        name = f"replication:{rule.replica[0]}.{rule.replica[1]}"
        source_field, replica_field = definition.field(rule.source), definition.field(rule.replica)
        src_inst, src_wall, _ = aligned(rule.source, rel.detail)
        rep_inst, rep_wall, _ = aligned(rule.replica, rel.detail)
        same_basis = source_field.zone_basis == replica_field.zone_basis
        if source_field.resolvable_to_instant and replica_field.resolvable_to_instant:
            assessable = trusted & src_inst.notna().to_numpy() & rep_inst.notna().to_numpy()
            same = (src_inst == rep_inst).to_numpy() & assessable          # full-precision UTC instants
            if same_basis:
                same &= (src_wall == rep_wall).to_numpy()                   # and the same wall clock
        elif same_basis:
            assessable = trusted & src_wall.notna().to_numpy() & rep_wall.notna().to_numpy()
            same = (src_wall == rep_wall).to_numpy() & assessable           # same naive basis: wall times
        else:
            replication_reports.append(_counts(name, len(cars), 0, 0))
            replicas_agree &= False
            continue
        replication_reports.append(_counts(name, len(cars), int(same.sum()), int((assessable & ~same).sum())))
        replicas_agree &= same

    # --- ordering (trusted detail rows whose finish-time replicas agree; full-precision UTC instants)
    if definition.ordering is None:
        ordering = _unavailable("timestamp_ordering", len(cars))
    else:
        rule = definition.ordering
        base = base_of(rule.earlier, rule.later)
        early, _, ok_a = aligned(rule.earlier, base)
        late, _, ok_b = aligned(rule.later, base)
        scope = (trusted & replicas_agree) if base == rel.detail else np.ones(len(frames[base]), dtype=bool)
        assessable = scope & ok_a & ok_b & early.notna().to_numpy() & late.notna().to_numpy()
        gap = (late - early)[assessable]
        tol = pd.Timedelta(rule.tolerance)
        good = (gap >= -tol) if rule.inclusive else (gap > -tol)
        ordering = _counts("timestamp_ordering", len(frames[base]), int(good.sum()), int((~good).sum()))

    # --- date derivation
    date_reports = []
    for check in definition.date_checks:
        name = f"date_derivation:{check.target[0]}.{check.target[1]}"
        if check.rule is None:
            date_reports.append(_unavailable(name, len(frames[check.target[0]])))
            continue
        base = base_of(check.target, check.rule.source)
        _, target_date, ok_t = aligned(check.target, base)
        source, _, ok_s = aligned(check.rule.source, base)
        # Detail rows take their reporting day only through a trusted linked parent (city integrity).
        scope = trusted if base == rel.detail else np.ones(len(frames[base]), dtype=bool)
        zones = _reporting_zones(check.rule, len(frames[base]), base == rel.detail, jobs, parent_values)
        has_zone = np.asarray([z is not None for z in zones], dtype=bool)
        assessable = (scope & ok_t & ok_s & has_zone & target_date.notna().to_numpy()
                      & source.notna().to_numpy())
        expected = _local_dates(source, zones, assessable)[assessable]
        match = (expected == target_date[assessable]).to_numpy()
        utc_date = source[assessable].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        crossing = match & (utc_date != expected).to_numpy()
        date_reports.append(_counts(name, len(frames[base]), int(match.sum()), int((~match).sum()),
                                    int(crossing.sum())))

    return TemporalReconciliationReport(
        field_reports=field_reports, ordering=ordering, date_checks=tuple(date_reports),
        replications=tuple(replication_reports), parent_row_count=len(jobs), detail_row_count=len(cars),
        unlinked_detail_row_count=int((~linked).sum()), city_mismatch_detail_row_count=int(city_mismatch.sum()),
    )


def validate_temporal_reconciliation(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    definition: TemporalReconciliationDefinition = TEMPORAL_RECONCILIATION,
) -> TemporalReconciliationReport:
    """Assess; return the report if valid, else raise (fails closed on unavailable rules).

    Raises:
        TemporalParseError: Required fields contain missing or invalid values.
        TemporalReconciliationError: Any other violation, including
            unavailable rules and unassessable rows.
    """
    report = assess_temporal_reconciliation(jobs, cars, definition)
    if report.is_valid:
        return report
    if not report.all_required_fields_parse:
        raise TemporalParseError(report)
    raise TemporalReconciliationError(report)


# ------------------------------------------------------------------- helpers


def _trusted_rows(cars: pd.DataFrame, rel, linked: np.ndarray, parent_values) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Linked detail rows whose scope columns (``city``) equal their parent's exactly."""
    agree = linked.copy()
    for parent_column, detail_column in rel.scope_agreement_columns:
        if detail_column not in cars.columns:
            raise TemporalConfigurationError("a scope agreement column is absent.")
        mine = cars[detail_column].astype(object).reset_index(drop=True)
        theirs = parent_values(parent_column)
        same = [isinstance(a, str) and isinstance(b, str) and a == b for a, b in zip(mine.tolist(), theirs.tolist())]
        agree &= np.asarray(same, dtype=bool)
    return agree


def _reporting_zones(rule: ReportingDateRule, n: int, detail: bool, jobs: pd.DataFrame,  # type: ignore[no-untyped-def]
                     parent_values) -> list:
    """The reporting zone of every base row (``None`` = no approved zone; never a default)."""
    if not rule.city_local:
        return [rule.reporting_timezone] * n
    column = rule.timezone_selector[1]           # type: ignore[index]
    if column not in jobs.columns:
        raise TemporalConfigurationError("the reporting-day selector column is absent.")
    cities = parent_values(column) if detail else jobs[column].astype(object).reset_index(drop=True)
    return [rule.city_timezones.zone_or_none(c) for c in cities.tolist()]   # type: ignore[union-attr]


def _local_dates(instants: pd.Series, zones: list, mask: np.ndarray) -> pd.Series:
    """Midnight-normalised local calendar date of each instant in its own zone (``NaT`` outside ``mask``)."""
    out = pd.Series(pd.NaT, index=instants.index, dtype="datetime64[ns]")
    zone_array = np.asarray(zones, dtype=object)
    for zone in sorted({z for z, m in zip(zones, mask) if m and z is not None}):
        rows = mask & (zone_array == zone)
        local = instants[rows].dt.tz_convert(zone).dt.tz_localize(None).dt.normalize()
        out[rows] = local.astype("datetime64[ns]")
    return out


def _unavailable(name: str, rows: int) -> TemporalRuleReport:
    return TemporalRuleReport(name, RuleStatus.UNAVAILABLE, rows, 0, 0, rows)


def _counts(name: str, rows: int, passed: int, failed: int, crossing: int = 0) -> TemporalRuleReport:
    return TemporalRuleReport(name, RuleStatus.CONFIGURED, rows, passed, failed, rows - passed - failed, crossing)


def _parent_positions(jobs: pd.DataFrame, cars: pd.DataFrame, rel) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Parent row position for every detail row (-1 = missing key or orphan); tuple keys."""
    parent = pd.MultiIndex.from_frame(jobs.loc[:, list(rel.parent_key_columns)])
    detail_keys = cars.loc[:, list(rel.detail_key_columns)]
    detail = pd.MultiIndex.from_frame(detail_keys.astype(object))
    positions = parent.get_indexer(detail)   # unique parent keys (precondition)
    positions[~detail_keys.notna().all(axis=1).to_numpy()] = -1
    return positions


def _parent_lookup(jobs: pd.DataFrame, cars: pd.DataFrame, positions: np.ndarray):  # type: ignore[no-untyped-def]
    """A function returning the linked parent's value of a column for every detail row (``None`` if unlinked)."""
    linked = positions >= 0
    take = np.where(linked, positions, 0)

    def parent_values(column: str) -> pd.Series:
        if len(jobs) == 0:
            return pd.Series([None] * len(cars), dtype=object)
        values = jobs[column].astype(object).iloc[take].reset_index(drop=True)
        return values.where(pd.Series(linked), None)
    return parent_values


def _parse_fields(jobs: pd.DataFrame, cars: pd.DataFrame, definition: TemporalReconciliationDefinition
                  ) -> tuple[dict, np.ndarray]:
    """Parse every configured field (city-local fields through their parent city); returns (results, positions)."""
    rel = definition.relationship
    frames = {rel.parent: jobs, rel.detail: cars}
    positions = _parent_positions(jobs, cars, rel)
    linked = positions >= 0
    parent_values = _parent_lookup(jobs, cars, positions)

    def parse(f: TemporalFieldDefinition) -> TemporalParseResult:
        frame = frames[f.dataset]
        if f.timezone_selector is None:
            return parse_temporal_field(frame[f.column], f, definition.canonical_timezone)
        selector = f.timezone_selector[1]
        if selector not in jobs.columns:
            raise TemporalConfigurationError("the timezone selector column is absent.")
        if f.dataset == rel.parent:
            cities, context = jobs[selector].reset_index(drop=True), None
        else:
            cities, context = parent_values(selector), linked
        series = frame[f.column].reset_index(drop=True)
        result = parse_temporal_field(series, f, definition.canonical_timezone, city_values=cities,
                                      context_available=context)
        return TemporalParseResult(result.instants.set_axis(frame.index), result.wall.set_axis(frame.index),
                                   result.missing, result.invalid, result.unresolved, result.unknown_city,
                                   result.context_unavailable, result.ambiguous, result.nonexistent)

    return {f.ref: parse(f) for f in definition.fields}, positions


@dataclass(frozen=True)
class DerivedTimestamps:
    """Derived UTC timestamps, separate from the source frames (which are never modified).

    ``jobs`` / ``cars`` are new frames aligned with the source rows holding,
    for every timestamp field, ``<column>_utc`` (timezone-aware UTC, full
    parsed precision, ``NaT`` when unresolved) and ``<column>_canonical``
    (``YYYYMMDDTHHMMSSZ`` presentation, ``None`` when unresolved). The
    canonical text is never an identity; compare the ``_utc`` columns.
    """

    jobs: pd.DataFrame
    cars: pd.DataFrame


def derive_utc_timestamps(jobs: pd.DataFrame, cars: pd.DataFrame,
                          definition: TemporalReconciliationDefinition = TEMPORAL_RECONCILIATION) -> DerivedTimestamps:
    """``finished_at_utc``, ``job_finished_at_utc``, ``scraped_at_utc`` (and ``*_canonical``) as new frames.

    Values come from the same resolution as :func:`assess_temporal_reconciliation`
    (city-local fields through the linked parent city); raw columns stay
    unchanged. Derived values are for analysis only and are never reported.
    """
    if not isinstance(definition, TemporalReconciliationDefinition):
        raise TypeError("definition must be a TemporalReconciliationDefinition")
    rel = definition.relationship
    _check_relationship_inputs(jobs, cars, rel, TemporalPreconditionError)
    parsed, _ = _parse_fields(jobs, cars, definition)
    out = {rel.parent: pd.DataFrame(index=jobs.index), rel.detail: pd.DataFrame(index=cars.index)}
    for f in definition.fields:
        if f.kind is not TemporalKind.TIMESTAMP:
            continue
        instants = parsed[f.ref].instants.dt.tz_convert("UTC")
        out[f.dataset][f"{f.column}_utc"] = instants
        out[f.dataset][f"{f.column}_canonical"] = canonical_utc_text(instants)
    return DerivedTimestamps(jobs=out[rel.parent], cars=out[rel.detail])


# ------------------------------------------------------------- reporting days


class ScrapeDateStatus(StrEnum):
    """Agreement of a source scrape date with the derived reporting day (per row; never repaired)."""

    AGREES = "agrees"
    MISMATCH = "mismatch"
    MISSING = "missing"            # missing or blank scrape date
    INVALID = "invalid"            # not exact ISO_8601_DATE text naming a real date
    UNRESOLVABLE = "unresolvable"  # no reporting day: finish time unresolved or city not approved
    UNLINKED = "unlinked"          # detail row with no linked parent
    UNTRUSTED = "untrusted"        # detail row whose city disagrees with its parent (city integrity)


@dataclass(frozen=True)
class DerivedReportingDays:
    """Derived reporting-day structure, separate from the source frames (never overwritten).

    ``jobs`` (aligned with the parent rows) and ``cars`` (aligned with the
    detail rows) hold: ``finished_at_raw`` (the raw parent finish value),
    ``finished_at_utc`` (full-precision UTC instant), ``reporting_timezone``,
    ``reporting_day`` (``datetime.date`` or ``None``), ``scrape_date_raw``,
    ``scrape_date_parsed`` (``datetime.date`` or ``None``), ``scrape_date_status``
    (:class:`ScrapeDateStatus` value) and ``reporting_day_provenance``. Detail
    rows carry their trusted linked parent's values only - never anything
    derived from ``scraped_at`` or from their own city. Values are for
    analysis only and are never reported.
    """

    jobs: pd.DataFrame
    cars: pd.DataFrame
    rule: ReportingDateRule

    def status_counts(self, dataset: DatasetKey) -> dict[str, int]:
        frame = self.jobs if dataset is DatasetKey.JOBS else self.cars
        counts = frame["scrape_date_status"].value_counts()
        return {s.value: int(counts.get(s.value, 0)) for s in ScrapeDateStatus}

    def eligible(self, dataset: DatasetKey) -> np.ndarray:
        """Rows whose scrape date agrees with a resolved reporting day."""
        frame = self.jobs if dataset is DatasetKey.JOBS else self.cars
        return (frame["scrape_date_status"] == ScrapeDateStatus.AGREES.value).to_numpy()


def _reporting_rule(definition: TemporalReconciliationDefinition) -> ReportingDateRule:
    rules = {c.rule for c in definition.date_checks if c.rule is not None}
    if len(rules) != 1 or any(c.rule is None for c in definition.date_checks):
        raise TemporalConfigurationError("the reporting day is unavailable: no single approved reporting-day rule")
    rule = next(iter(rules))
    if rule.source[0] is not definition.relationship.parent:
        raise TemporalConfigurationError("the reporting day derives from a parent timestamp")
    return rule


def derive_reporting_days(jobs: pd.DataFrame, cars: pd.DataFrame,
                          definition: TemporalReconciliationDefinition) -> DerivedReportingDays:
    """The reporting day of every parent job, and of every detail row through its trusted parent.

    Reporting day = the local calendar date of the parent finish instant in
    the parent city's approved zone. Each configured date check's target is
    compared semantically (strictly parsed) with it; raw values are kept and
    no value is repaired, preferred or overwritten. Missing, invalid,
    mismatched, unlinked, untrusted and unresolvable rows are labelled, never
    dropped.

    Raises:
        TemporalConfigurationError: No single approved reporting-day rule.
    """
    if not isinstance(definition, TemporalReconciliationDefinition):
        raise TypeError("definition must be a TemporalReconciliationDefinition")
    rule = _reporting_rule(definition)
    rel = definition.relationship
    _check_relationship_inputs(jobs, cars, rel, TemporalPreconditionError)
    parsed, positions = _parse_fields(jobs, cars, definition)
    linked = positions >= 0
    take = np.where(linked, positions, 0)
    parent_values = _parent_lookup(jobs, cars, positions)
    trusted = _trusted_rows(cars, rel, linked, parent_values)
    targets = {c.target[0]: c.target for c in definition.date_checks}
    if set(targets) != {rel.parent, rel.detail}:
        raise TemporalConfigurationError("one reporting-day date check per dataset is required")

    finish = parsed[rule.source].instants.reset_index(drop=True).dt.tz_convert("UTC")
    zones = _reporting_zones(rule, len(jobs), False, jobs, parent_values)
    resolved = finish.notna().to_numpy() & np.asarray([z is not None for z in zones], dtype=bool)
    day = _local_dates(finish, zones, resolved)
    provenance = (f"{rule.source[0]}.{rule.source[1]}@" + (
        f"{rule.timezone_selector[0]}.{rule.timezone_selector[1]}" if rule.city_local else "fixed_zone"))

    def _dates(series: pd.Series) -> list:
        return [None if pd.isna(v) else v.date() for v in series.tolist()]

    def frame(base: DatasetKey, n: int, index: pd.Index, raw_finish: pd.Series, utc: pd.Series, zone: list,
              rday: pd.Series, detail_scope: np.ndarray | None, prov: str) -> pd.DataFrame:
        target = parsed[targets[base]]
        source_frame = jobs if base == rel.parent else cars
        target_day = target.wall.reset_index(drop=True)
        missing, invalid = target.missing, target.invalid
        status = np.full(n, ScrapeDateStatus.AGREES.value, dtype=object)
        has_day = rday.notna().to_numpy()
        same = (rday == target_day).to_numpy() & has_day & target_day.notna().to_numpy()
        status[~same] = ScrapeDateStatus.MISMATCH.value
        status[~has_day] = ScrapeDateStatus.UNRESOLVABLE.value
        status[invalid] = ScrapeDateStatus.INVALID.value
        status[missing] = ScrapeDateStatus.MISSING.value
        if detail_scope is not None:
            status[linked & ~trusted] = ScrapeDateStatus.UNTRUSTED.value
            status[~linked] = ScrapeDateStatus.UNLINKED.value
        return pd.DataFrame({
            "finished_at_raw": raw_finish.to_numpy(dtype=object),
            "finished_at_utc": utc.to_numpy(),
            "reporting_timezone": np.asarray(zone, dtype=object),
            "reporting_day": np.asarray(_dates(rday), dtype=object),
            "scrape_date_raw": source_frame[targets[base][1]].astype(object).to_numpy(),
            "scrape_date_parsed": np.asarray(_dates(target_day.where(~pd.Series(invalid | missing))), dtype=object),
            "scrape_date_status": status,
            "reporting_day_provenance": np.full(n, prov, dtype=object),
        }, index=index)

    job_frame = frame(rel.parent, len(jobs), jobs.index, jobs[rule.source[1]].astype(object).reset_index(drop=True),
                      finish, zones, day, None, provenance)
    keep = pd.Series(trusted)
    n = len(cars)
    if len(jobs):
        d_utc = finish.iloc[take].reset_index(drop=True).where(keep)
        d_raw = jobs[rule.source[1]].astype(object).reset_index(drop=True).iloc[take].reset_index(drop=True).where(
            keep, None)
        d_zone = [zones[p] if t else None for p, t in zip(take.tolist(), trusted.tolist())]
        d_day = day.iloc[take].reset_index(drop=True).where(keep)
    else:
        d_utc = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]")
        d_raw = pd.Series([None] * n, dtype=object)
        d_zone = [None] * n
        d_day = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    car_frame = frame(rel.detail, n, cars.index, d_raw, d_utc, d_zone, d_day, trusted, "linked_parent:" + provenance)
    return DerivedReportingDays(jobs=job_frame, cars=car_frame, rule=rule)
