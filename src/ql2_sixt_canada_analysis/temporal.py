"""Parse and reconcile the source temporal fields against the central contract.

The contract is :data:`~ql2_sixt_canada_analysis.schemas.TEMPORAL_RECONCILIATION`.

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

Rules (:func:`assess_temporal_reconciliation`)
---------------------------------------------
* **Ordering** - ``earlier <= later`` within an explicit, non-negative
  tolerance (``later - earlier >= -tolerance``; strict when exclusive).
* **Date derivation** - the date equals the calendar date of the source
  instant *after* conversion to the rule's reporting zone. Rows where that
  date differs from the UTC date but match are counted as legitimate
  boundary crossings.
* **Replication** - a detail-row copy equals its parent's value (instants, or
  wall times when both sides are naive on the same basis).

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

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.relationships import RelationshipPreconditionError, _check_relationship_inputs
from ql2_sixt_canada_analysis.schemas import (
    TEMPORAL_RECONCILIATION,
    DatasetKey,
    TemporalAwareness,
    TemporalConfigurationError,
    TemporalFieldDefinition,
    TemporalKind,
    TemporalReconciliationDefinition,
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
    "assess_temporal_reconciliation",
    "parse_temporal_field",
    "validate_temporal_reconciliation",
]

_OFFSET_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_DESIGNATOR = re.compile(r"^(?P<body>.+) (?P<zone>[A-Z]+)$")


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

    @property
    def valid(self) -> np.ndarray:
        return ~(self.missing | self.invalid)


def parse_temporal_field(
    series: pd.Series, field: TemporalFieldDefinition, canonical_timezone: str = "UTC"
) -> TemporalParseResult:
    """Parse ``series`` according to ``field`` without modifying it (see module docstring)."""
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
    if field.source_timezone is None:
        return done(nat, wall, candidate & ~parsed, parsed.copy())
    local = wall.dt.tz_localize(field.source_timezone, ambiguous="NaT", nonexistent="NaT")
    resolved = local.notna().to_numpy()
    instants = local.dt.tz_convert(canonical_timezone)
    return done(instants, wall, candidate & ~parsed, parsed & ~resolved)


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

    def __post_init__(self) -> None:
        assert self.row_count == self.valid_count + self.missing_count + self.invalid_count
        assert 0 <= self.unresolved_count <= self.valid_count

    @property
    def ref(self) -> tuple[DatasetKey, str]:
        return (self.dataset, self.column)

    @property
    def parses(self) -> bool:
        """No invalid values, and no missing values for a required field."""
        return self.invalid_count == 0 and (not self.required or self.missing_count == 0)


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

    @property
    def all_required_fields_parse(self) -> bool:
        return all(r.parses for r in self.field_reports)

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
    def unavailable_rules(self) -> tuple[str, ...]:
        rules = (self.ordering, *self.date_checks, *self.replications)
        return tuple(r.name for r in rules if r.status is RuleStatus.UNAVAILABLE)

    @property
    def is_valid(self) -> bool:
        """Every field parses and every rule is configured and holds for all rows."""
        return (self.all_required_fields_parse and self.timestamp_ordering_valid
                and self.date_derivation_valid and self.replication_valid
                and self.unlinked_detail_row_count == 0)

    @property
    def violations(self) -> tuple[str, ...]:
        rules = (self.ordering, *self.date_checks, *self.replications)
        checks = (
            ("field_parse", not self.all_required_fields_parse),
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

    parsed = {f.ref: parse_temporal_field(frames[f.dataset][f.column], f, definition.canonical_timezone)
              for f in definition.fields}
    field_reports = tuple(
        TemporalFieldReport(
            dataset=f.dataset, column=f.column, kind=f.kind, required=f.required, row_count=len(frames[f.dataset]),
            valid_count=int(parsed[f.ref].valid.sum()), missing_count=int(parsed[f.ref].missing.sum()),
            invalid_count=int(parsed[f.ref].invalid.sum()), unresolved_count=int(parsed[f.ref].unresolved.sum()),
        )
        for f in definition.fields
    )
    positions = _parent_positions(jobs, cars, rel)
    linked = positions >= 0

    def aligned(ref: tuple[DatasetKey, str], base: DatasetKey) -> tuple[pd.Series, pd.Series, np.ndarray]:
        """(instants, wall, row-linked mask) of ``ref`` aligned to ``base`` rows."""
        result = parsed[ref]
        if ref[0] == base:
            n = len(frames[base])
            return (result.instants.reset_index(drop=True), result.wall.reset_index(drop=True),
                    np.ones(n, dtype=bool))
        take = np.where(linked, positions, 0)
        inst = result.instants.iloc[take].reset_index(drop=True).where(pd.Series(linked))
        wall = result.wall.iloc[take].reset_index(drop=True).where(pd.Series(linked))
        return inst, wall, linked

    def base_of(*refs: tuple[DatasetKey, str]) -> DatasetKey:
        return rel.detail if any(r[0] == rel.detail for r in refs) else rel.parent

    # --- ordering
    if definition.ordering is None:
        ordering = _unavailable("timestamp_ordering", len(cars))
    else:
        rule = definition.ordering
        base = base_of(rule.earlier, rule.later)
        early, _, ok_a = aligned(rule.earlier, base)
        late, _, ok_b = aligned(rule.later, base)
        assessable = ok_a & ok_b & early.notna().to_numpy() & late.notna().to_numpy()
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
        assessable = ok_t & ok_s & target_date.notna().to_numpy() & source.notna().to_numpy()
        local = source[assessable].dt.tz_convert(check.rule.reporting_timezone)
        expected = local.dt.tz_localize(None).dt.normalize()
        match = (expected == target_date[assessable]).to_numpy()
        utc_date = source[assessable].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        crossing = match & (utc_date != expected).to_numpy()
        date_reports.append(_counts(name, len(frames[base]), int(match.sum()), int((~match).sum()),
                                    int(crossing.sum())))

    # --- replication
    replication_reports = []
    for rule in definition.replications:
        name = f"replication:{rule.replica[0]}.{rule.replica[1]}"
        source_field, replica_field = definition.field(rule.source), definition.field(rule.replica)
        src_inst, src_wall, ok = aligned(rule.source, rel.detail)
        rep_inst, rep_wall, _ = aligned(rule.replica, rel.detail)
        if source_field.resolvable_to_instant and replica_field.resolvable_to_instant:
            left, right = src_inst, rep_inst
        elif (source_field.awareness is replica_field.awareness
              and source_field.source_timezone == replica_field.source_timezone):
            left, right = src_wall, rep_wall          # same naive basis: compare wall times
        else:
            replication_reports.append(_counts(name, len(cars), 0, 0))
            continue
        assessable = ok & left.notna().to_numpy() & right.notna().to_numpy()
        same = (left[assessable] == right[assessable]).to_numpy()
        replication_reports.append(_counts(name, len(cars), int(same.sum()), int((~same).sum())))

    return TemporalReconciliationReport(
        field_reports=field_reports, ordering=ordering, date_checks=tuple(date_reports),
        replications=tuple(replication_reports), parent_row_count=len(jobs), detail_row_count=len(cars),
        unlinked_detail_row_count=int((~linked).sum()),
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
