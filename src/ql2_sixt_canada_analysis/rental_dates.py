"""Authority-backed rental-date validity, parent/detail agreement and pricing eligibility.

Built only from the APPROVED schema-3 decisions ``RENTAL_DATE_VALIDITY`` and
``RENTAL_DATE_PARENT_DETAIL_AGREEMENTS`` of the current authority record
(``pricing-authorities-v7`` onwards):

* Six governed fields: parent ``jobs.pickup_date`` / ``jobs.return_date``;
  detail ``cars.job_pickup_date`` / ``cars.job_return_date`` (the job-prefixed
  copies) and ``cars.pickup_date`` / ``cars.return_date``.
* Format ``ISO_8601_DATE``: exactly ``YYYY-MM-DD`` text naming a real calendar
  date (:func:`parse_iso_dates`). Dates are calendar dates - never timestamps,
  never midnight instants, never given a zone. Nothing is trimmed or coerced:
  surrounding whitespace, times, offsets, slashes, month names, numbers
  (including spreadsheet serials) and other objects are *invalid*; nulls,
  empty and whitespace-only text are *missing*.
* Validity of every pickup/return pair: both required; duration (return minus
  pickup, in calendar days) must be at least the approved minimum (zero: a
  same-day rental is valid); the maximum is **UNBOUNDED** - a long, correctly
  ordered rental is valid data. No threshold is ever derived from observed
  values.
* Agreement: each detail field equals its approved parent field (job-prefixed
  and own pickup -> parent pickup; return -> parent return), compared as
  parsed dates on trusted linked rows only. Disagreements are reported, never
  repaired; raw columns are never modified.

**Validity is not analysis eligibility.** :attr:`RentalDateReport` counts every
row and decides *data validity*; a row is *pricing eligible* only when its
dates are valid and agree. A study that wants a duration cohort selects it
afterwards with :func:`analysis_duration_cohort`, which never changes a
validity result and never re-admits an invalid row.

Reports hold counts, statuses and field names only - never dates, identifiers
or rows.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.authority_decisions import (
    RENTAL_DATE_DETAIL_FIELDS,
    RENTAL_DATE_PARENT_FIELDS,
    AuthorityDecisionRecord,
    DecisionId,
    DecisionStatus,
    load_current_decision_record,
)
from ql2_sixt_canada_analysis.schemas import ANALYSIS_JOB_DETAIL_RELATIONSHIP, JobDetailRelationshipDefinition

__all__ = [
    "ISO_DATE_PATTERN",
    "INFORMATIONAL_LONG_RENTAL_DAYS",
    "RENTAL_DECISIONS",
    "AgreementCounts",
    "AgreementStatus",
    "DateValueStatus",
    "DerivedRentalPeriods",
    "FieldDateCounts",
    "PeriodCounts",
    "PeriodStatus",
    "RentalDateBlocker",
    "RentalDatePolicy",
    "RentalDateReport",
    "RentalPolicyStatus",
    "analysis_duration_cohort",
    "assess_rental_dates",
    "current_rental_date_policy",
    "derive_rental_periods",
    "parse_iso_dates",
    "rental_date_policy_from_record",
]

D = DecisionId
RENTAL_DECISIONS = (D.RENTAL_DATE_VALIDITY, D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS)
#: The exact accepted text of an ISO_8601_DATE value (then checked to be a real calendar date).
ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
#: Descriptive, **informational** band only (never a validity or eligibility rule): periods of at
#: least this many days are counted separately so long rentals stay visible in the baseline.
INFORMATIONAL_LONG_RENTAL_DAYS = 30
#: The two pickup/return pairs of a detail row: the job-prefixed copies and the row's own dates.
DETAIL_PERIODS = (("cars.job_pickup_date", "cars.job_return_date"), ("cars.pickup_date", "cars.return_date"))
PARENT_PERIOD = ("jobs.pickup_date", "jobs.return_date")


class RentalPolicyStatus(StrEnum):
    APPROVED = "approved"
    NOT_APPROVED = "not_approved"
    RECORD_UNAVAILABLE = "record_unavailable"
    INVALID = "invalid"


class DateValueStatus(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"


class PeriodStatus(StrEnum):
    """Data validity of one pickup/return pair (first matching state wins, in this order)."""

    VALID = "valid"
    PICKUP_INVALID = "pickup_invalid"
    RETURN_INVALID = "return_invalid"
    BOTH_MISSING = "both_missing"
    PICKUP_MISSING = "pickup_missing"
    RETURN_MISSING = "return_missing"
    RETURN_BEFORE_PICKUP = "return_before_pickup"         # negative duration (or below the minimum)
    ABOVE_MAXIMUM = "above_maximum"                       # only under an approved BOUNDED maximum


class AgreementStatus(StrEnum):
    """One detail field against its approved parent field on one detail row."""

    MATCH = "match"
    MISMATCH = "mismatch"
    PARENT_MISSING = "parent_missing"
    DETAIL_MISSING = "detail_missing"
    BOTH_MISSING = "both_missing"
    PARENT_INVALID = "parent_invalid"
    DETAIL_INVALID = "detail_invalid"
    ORPHAN_DETAIL = "orphan_detail"                       # missing key or no parent
    LINKAGE_UNTRUSTED = "linkage_untrusted"               # no valid linkage report for these frames


class RentalDateBlocker(StrEnum):
    """Rental-date blockers (values equal ``PricingBlocker`` values)."""

    RENTAL_DATE_RULES_UNAVAILABLE = "rental_date_rules_unavailable"
    RENTAL_DATE_REQUIRED_VALUE_MISSING = "rental_date_required_value_missing"
    RENTAL_DATE_FORMAT_INVALID = "rental_date_format_invalid"
    RENTAL_DATE_ORDERING_INVALID = "rental_date_ordering_invalid"
    RENTAL_DATE_DURATION_ABOVE_MAXIMUM = "rental_date_duration_above_maximum"
    RENTAL_DATE_PARENT_DETAIL_MISMATCH = "rental_date_parent_detail_mismatch"
    RENTAL_DATE_AGREEMENT_UNASSESSABLE = "rental_date_agreement_unassessable"


# --------------------------------------------------------------------- policy


@dataclass(frozen=True, slots=True)
class RentalDatePolicy:
    """The approved rental-date policy, or why it is unavailable (no values)."""

    status: RentalPolicyStatus
    parent_fields: tuple[str, ...] = RENTAL_DATE_PARENT_FIELDS
    detail_fields: tuple[str, ...] = RENTAL_DATE_DETAIL_FIELDS
    source_format: str | None = None
    pickup_required: bool | None = None
    return_required: bool | None = None
    ordering: str | None = None                       # RETURN_ON_OR_AFTER_PICKUP | RETURN_AFTER_PICKUP
    equal_dates_allowed: bool | None = None
    minimum_duration_days: int | None = None
    maximum_duration_days: int | None = None          # None with an UNBOUNDED mode: no maximum
    maximum_duration_mode: str | None = None          # UNBOUNDED | BOUNDED
    agreements: tuple[tuple[str, str], ...] = ()      # (detail target, parent source)
    record_id: str | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, RentalPolicyStatus):
            raise TypeError("status must be a RentalPolicyStatus")
        if self.status is not RentalPolicyStatus.APPROVED:
            return
        if self.source_format != "ISO_8601_DATE":
            raise ValueError("only ISO_8601_DATE is implemented")
        if not (isinstance(self.pickup_required, bool) and isinstance(self.return_required, bool)
                and isinstance(self.equal_dates_allowed, bool)):
            raise ValueError("requiredness and equality must be explicit booleans")
        if type(self.minimum_duration_days) is not int or self.minimum_duration_days < 0:
            raise ValueError("the minimum duration must be a non-negative integer")
        if (self.maximum_duration_mode == "UNBOUNDED") != (self.maximum_duration_days is None) or \
                self.maximum_duration_mode not in ("UNBOUNDED", "BOUNDED"):
            raise ValueError("an UNBOUNDED maximum has no value; a BOUNDED one has an integer")
        targets = [t for t, _ in self.agreements]
        if sorted(targets) != sorted(self.detail_fields) or not all(s in self.parent_fields
                                                                  for _, s in self.agreements):
            raise ValueError("every detail field needs exactly one parent source")
        if not self.record_id or not self.references:
            raise ValueError("an approved policy names its record and references")

    @property
    def available(self) -> bool:
        return self.status is RentalPolicyStatus.APPROVED

    def source_of(self, target: str) -> str:
        return dict(self.agreements)[target]


def rental_date_policy_from_record(record: AuthorityDecisionRecord | None) -> RentalDatePolicy:
    """Build the policy from the APPROVED schema-3 rental decisions (fails closed)."""
    S = RentalPolicyStatus
    if record is None:
        return RentalDatePolicy(status=S.RECORD_UNAVAILABLE)
    if any(record.decision(d).status is not DecisionStatus.APPROVED for d in RENTAL_DECISIONS):
        return RentalDatePolicy(status=S.NOT_APPROVED, record_id=record.record_id)
    try:
        if record.schema_version < 3:
            raise ValueError("historical rental shapes are not implementable")
        validity = record.approved_resolution(D.RENTAL_DATE_VALIDITY)
        mapping = record.approved_resolution(D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS)
        references = tuple(dict.fromkeys(a.reference for d in RENTAL_DECISIONS for a in record.decision(d).authority))
        return RentalDatePolicy(
            status=S.APPROVED, source_format=validity["date_format"], pickup_required=validity["pickup_required"],
            return_required=validity["return_required"], ordering=validity["ordering"],
            equal_dates_allowed=validity["equal_dates_allowed"],
            minimum_duration_days=validity["minimum_duration_days"],
            maximum_duration_mode=validity["maximum_duration_mode"],
            maximum_duration_days=validity.get("maximum_duration_days"),
            agreements=tuple(sorted((a["target"], a["source"]) for a in mapping["agreements"])),
            record_id=record.record_id, references=references)
    except (KeyError, TypeError, ValueError):
        return RentalDatePolicy(status=S.INVALID, record_id=record.record_id)


@cache
def current_rental_date_policy() -> RentalDatePolicy:
    """The rental-date policy of the current committed authority record (cached)."""
    return rental_date_policy_from_record(load_current_decision_record())


# -------------------------------------------------------------------- parsing


def _parse_one(value: object) -> tuple[DateValueStatus, dt.date | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA or value is pd.NaT:
        return DateValueStatus.MISSING, None
    if not isinstance(value, str):                      # numbers, serials, datetimes, other objects
        return DateValueStatus.INVALID, None
    if not value.strip():
        return DateValueStatus.MISSING, None
    if not ISO_DATE_PATTERN.fullmatch(value):           # exact text: no trimming or coercion
        return DateValueStatus.INVALID, None
    try:
        return DateValueStatus.VALID, dt.date.fromisoformat(value)
    except ValueError:                                  # impossible month/day, invalid leap day
        return DateValueStatus.INVALID, None


def parse_iso_dates(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(statuses, dates) for ``series``: ``DateValueStatus`` values and ``datetime.date`` or ``None``.

    The source series is not modified; its raw values stay the record of truth.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    parsed = [_parse_one(v) for v in series.astype(object).tolist()]
    statuses = pd.Series([p[0] for p in parsed], index=series.index, dtype=object)
    dates = pd.Series([p[1] for p in parsed], index=series.index, dtype=object)
    return statuses, dates


# --------------------------------------------------------------------- report


@dataclass(frozen=True, slots=True)
class FieldDateCounts:
    field: str
    rows: int
    valid: int
    missing: int
    invalid: int


@dataclass(frozen=True, slots=True)
class PeriodCounts:
    """Validity of one pickup/return pair over every row of its dataset (counts only)."""

    period: str                                    # e.g. jobs.pickup_date..jobs.return_date
    rows: int
    by_status: tuple[tuple[str, int], ...]
    same_day: int                                  # valid, zero-day duration
    long_informational: int                        # valid, >= INFORMATIONAL_LONG_RENTAL_DAYS (informational)

    def count(self, status: PeriodStatus) -> int:
        return dict(self.by_status).get(status.value, 0)


@dataclass(frozen=True, slots=True)
class AgreementCounts:
    """One approved mapping over every detail row (counts only)."""

    target: str
    source: str
    rows: int
    by_status: tuple[tuple[str, int], ...]

    def count(self, status: AgreementStatus) -> int:
        return dict(self.by_status).get(status.value, 0)


@dataclass(frozen=True, slots=True)
class RentalDateReport:
    """Rental-date validity, agreement and pricing eligibility (aggregate; every in-scope row counted)."""

    policy: RentalDatePolicy
    parent_rows: int = 0
    detail_rows: int = 0
    fields: tuple[FieldDateCounts, ...] = ()
    periods: tuple[PeriodCounts, ...] = ()
    agreements: tuple[AgreementCounts, ...] = ()
    linkage_trusted: bool = False
    eligible_parent_rows: int = 0
    eligible_detail_rows: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RentalDatePolicy):
            raise TypeError("policy must be a RentalDatePolicy")
        for p in self.periods:
            assert sum(n for _, n in p.by_status) == p.rows
        for a in self.agreements:
            assert sum(n for _, n in a.by_status) == a.rows == self.detail_rows
        assert self.eligible_parent_rows <= self.parent_rows and self.eligible_detail_rows <= self.detail_rows

    @property
    def assessed(self) -> bool:
        return self.policy.available

    def _period_total(self, *statuses: PeriodStatus) -> int:
        return sum(p.count(s) for p in self.periods for s in statuses)

    def _agreement_total(self, *statuses: AgreementStatus) -> int:
        return sum(a.count(s) for a in self.agreements for s in statuses)

    @property
    def missing_values(self) -> int:
        return sum(f.missing for f in self.fields)

    @property
    def invalid_values(self) -> int:
        return sum(f.invalid for f in self.fields)

    @property
    def ordering_violations(self) -> int:
        return self._period_total(PeriodStatus.RETURN_BEFORE_PICKUP)

    @property
    def mismatches(self) -> int:
        return self._agreement_total(AgreementStatus.MISMATCH)

    @property
    def unassessable_agreements(self) -> int:
        return sum(a.rows - a.count(AgreementStatus.MATCH) - a.count(AgreementStatus.MISMATCH)
                   for a in self.agreements)

    @property
    def validity_holds(self) -> bool:
        """Every governed value present and valid and every pair ordered (no agreement)."""
        return self.assessed and not self.missing_values and not self.invalid_values and all(
            p.count(PeriodStatus.VALID) == p.rows for p in self.periods)

    @property
    def agreement_holds(self) -> bool:
        return self.assessed and self.linkage_trusted and all(
            a.count(AgreementStatus.MATCH) == a.rows for a in self.agreements)

    @property
    def blocking_reasons(self) -> tuple[RentalDateBlocker, ...]:
        B = RentalDateBlocker
        if not self.assessed:
            return (B.RENTAL_DATE_RULES_UNAVAILABLE,)
        required = [f for f in self.fields if f.missing and (
            (self.policy.pickup_required and f.field.endswith("pickup_date"))
            or (self.policy.return_required and f.field.endswith("return_date")))]
        checks = (
            (B.RENTAL_DATE_REQUIRED_VALUE_MISSING, bool(required)),
            (B.RENTAL_DATE_FORMAT_INVALID, self.invalid_values > 0),
            (B.RENTAL_DATE_ORDERING_INVALID, self.ordering_violations > 0),
            (B.RENTAL_DATE_DURATION_ABOVE_MAXIMUM, self._period_total(PeriodStatus.ABOVE_MAXIMUM) > 0),
            (B.RENTAL_DATE_PARENT_DETAIL_MISMATCH, self.mismatches > 0),
            (B.RENTAL_DATE_AGREEMENT_UNASSESSABLE, not self.linkage_trusted or self.unassessable_agreements > 0),
        )
        return tuple(b for b, failed in checks if failed)

    @property
    def is_valid(self) -> bool:
        return not self.blocking_reasons


# ---------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class DerivedRentalPeriods:
    """Derived analytical values in new frames aligned with the sources (never written back).

    ``jobs``: ``rental_duration_days`` (``<NA>`` unless the period parses),
    ``rental_period_status`` and ``pricing_eligible``. ``cars``:
    ``job_rental_duration_days``, ``rental_duration_days``, both period
    statuses, ``rental_dates_agree`` and ``pricing_eligible``.
    """

    jobs: pd.DataFrame
    cars: pd.DataFrame


def _period(statuses_p, dates_p, statuses_r, dates_r, policy):  # type: ignore[no-untyped-def]
    """(PeriodStatus list, duration list) for aligned pickup/return values."""
    out, durations = [], []
    for sp, dp, sr, dr in zip(statuses_p, dates_p, statuses_r, dates_r):
        duration = (dr - dp).days if sp == DateValueStatus.VALID and sr == DateValueStatus.VALID else None
        if sp == DateValueStatus.INVALID:
            status = PeriodStatus.PICKUP_INVALID
        elif sr == DateValueStatus.INVALID:
            status = PeriodStatus.RETURN_INVALID
        elif sp == DateValueStatus.MISSING and sr == DateValueStatus.MISSING:
            status = PeriodStatus.BOTH_MISSING
        elif sp == DateValueStatus.MISSING:
            status = PeriodStatus.PICKUP_MISSING
        elif sr == DateValueStatus.MISSING:
            status = PeriodStatus.RETURN_MISSING
        elif duration < policy.minimum_duration_days:
            status = PeriodStatus.RETURN_BEFORE_PICKUP
        elif policy.maximum_duration_days is not None and duration > policy.maximum_duration_days:
            status = PeriodStatus.ABOVE_MAXIMUM
        else:
            status = PeriodStatus.VALID
        out.append(status)
        durations.append(duration)
    return out, durations


def _trusted(job_linkage, jobs, cars) -> bool:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis.job_linkage import JobLinkageReport

    return (isinstance(job_linkage, JobLinkageReport) and job_linkage.is_valid
            and job_linkage.parent_row_count == len(jobs) and job_linkage.detail_row_count == len(cars))


def _evaluate(jobs, cars, policy, relationship, job_linkage):  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis.temporal import _parent_positions

    frames = {"jobs": jobs, "cars": cars}
    for ref in (*policy.parent_fields, *policy.detail_fields):
        dataset, column = ref.split(".", 1)
        if column not in frames[dataset].columns:
            raise ValueError("a governed rental-date column is absent")
    parsed = {}
    for ref in (*policy.parent_fields, *policy.detail_fields):
        dataset, column = ref.split(".", 1)
        statuses, dates = parse_iso_dates(frames[dataset][column])
        parsed[ref] = (statuses.tolist(), dates.tolist())
    parent_status, parent_duration = _period(*parsed[PARENT_PERIOD[0]], *parsed[PARENT_PERIOD[1]], policy)
    detail_periods = {pair: _period(*parsed[pair[0]], *parsed[pair[1]], policy) for pair in DETAIL_PERIODS}

    trusted = _trusted(job_linkage, jobs, cars)
    positions = np.full(len(cars), -1, dtype=int)
    if trusted and len(cars):
        try:                                            # unique parent keys are a precondition of the link
            positions = _parent_positions(jobs, cars, relationship)
        except (KeyError, ValueError, pd.errors.InvalidIndexError):   # missing key columns or non-unique parents
            trusted = False
    agreement = {}
    for target, source in policy.agreements:
        t_status, t_date = parsed[target]
        s_status, s_date = parsed[source]
        rows = []
        for i, pos in enumerate(positions):
            if not trusted:
                rows.append(AgreementStatus.LINKAGE_UNTRUSTED)
                continue
            if pos < 0:
                rows.append(AgreementStatus.ORPHAN_DETAIL)
                continue
            ps, ds = s_status[pos], t_status[i]
            if ps == DateValueStatus.INVALID:
                rows.append(AgreementStatus.PARENT_INVALID)
            elif ds == DateValueStatus.INVALID:
                rows.append(AgreementStatus.DETAIL_INVALID)
            elif ps == DateValueStatus.MISSING and ds == DateValueStatus.MISSING:
                rows.append(AgreementStatus.BOTH_MISSING)
            elif ps == DateValueStatus.MISSING:
                rows.append(AgreementStatus.PARENT_MISSING)
            elif ds == DateValueStatus.MISSING:
                rows.append(AgreementStatus.DETAIL_MISSING)
            else:                                       # parsed calendar dates, never raw text
                rows.append(AgreementStatus.MATCH if s_date[pos] == t_date[i] else AgreementStatus.MISMATCH)
        agreement[target] = rows

    parent_ok = np.array([s is PeriodStatus.VALID for s in parent_status], dtype=bool)
    agree_all = np.ones(len(cars), dtype=bool)
    for rows in agreement.values():
        agree_all &= np.array([r is AgreementStatus.MATCH for r in rows], dtype=bool)
    detail_ok = agree_all.copy()
    for statuses, _ in detail_periods.values():
        detail_ok &= np.array([s is PeriodStatus.VALID for s in statuses], dtype=bool)
    linked_parent_ok = np.array([pos >= 0 and parent_ok[pos] for pos in positions], dtype=bool)
    detail_eligible = detail_ok & linked_parent_ok & trusted
    return parsed, (parent_status, parent_duration), detail_periods, agreement, trusted, parent_ok, \
        agree_all, detail_eligible


def _period_counts(name: str, statuses, durations) -> PeriodCounts:  # type: ignore[no-untyped-def]
    counts = Counter(s.value for s in statuses)
    valid = [d for s, d in zip(statuses, durations) if s is PeriodStatus.VALID]
    return PeriodCounts(period=name, rows=len(statuses), by_status=tuple(sorted(counts.items())),
                        same_day=sum(1 for d in valid if d == 0),
                        long_informational=sum(1 for d in valid if d >= INFORMATIONAL_LONG_RENTAL_DAYS))


def assess_rental_dates(jobs: pd.DataFrame, cars: pd.DataFrame, *, policy: RentalDatePolicy, job_linkage: object,
                        relationship: JobDetailRelationshipDefinition = ANALYSIS_JOB_DETAIL_RELATIONSHIP
                        ) -> RentalDateReport:
    """Validate every governed rental date and every approved agreement (aggregate report).

    ``jobs`` / ``cars`` are the linked analysis frames for ``relationship``;
    ``job_linkage`` must be the valid :class:`~ql2_sixt_canada_analysis.job_linkage.JobLinkageReport`
    of those same frames (row counts must match), otherwise every agreement is
    ``linkage_untrusted``. No row is dropped, repaired or modified.
    """
    if not isinstance(policy, RentalDatePolicy):
        raise TypeError("policy must be a RentalDatePolicy")
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be DataFrames")
    if not policy.available:
        return RentalDateReport(policy=policy, parent_rows=len(jobs), detail_rows=len(cars))
    parsed, (p_status, p_duration), detail_periods, agreement, trusted, parent_ok, _, detail_eligible = _evaluate(
        jobs, cars, policy, relationship, job_linkage)
    fields_ = tuple(FieldDateCounts(
        field=ref, rows=len(st), valid=sum(1 for s in st if s == DateValueStatus.VALID),
        missing=sum(1 for s in st if s == DateValueStatus.MISSING),
        invalid=sum(1 for s in st if s == DateValueStatus.INVALID)) for ref, (st, _) in parsed.items())
    periods = (_period_counts("..".join(PARENT_PERIOD), p_status, p_duration),
               *(_period_counts("..".join(pair), *detail_periods[pair]) for pair in DETAIL_PERIODS))
    agreements = tuple(AgreementCounts(target=t, source=s, rows=len(agreement[t]),
                                       by_status=tuple(sorted(Counter(r.value for r in agreement[t]).items())))
                       for t, s in policy.agreements)
    return RentalDateReport(policy=policy, parent_rows=len(jobs), detail_rows=len(cars), fields=fields_,
                            periods=periods, agreements=agreements, linkage_trusted=trusted,
                            eligible_parent_rows=int(parent_ok.sum()), eligible_detail_rows=int(detail_eligible.sum()))


def derive_rental_periods(jobs: pd.DataFrame, cars: pd.DataFrame, *, policy: RentalDatePolicy, job_linkage: object,
                          relationship: JobDetailRelationshipDefinition = ANALYSIS_JOB_DETAIL_RELATIONSHIP
                          ) -> DerivedRentalPeriods:
    """Durations, statuses and pricing eligibility as new frames (raw columns untouched)."""
    if not policy.available:
        raise ValueError("rental-date rules are unavailable")
    _, (p_status, p_duration), detail_periods, _, _, parent_ok, agree_all, eligible = _evaluate(
        jobs, cars, policy, relationship, job_linkage)
    out_jobs = pd.DataFrame({"rental_duration_days": pd.array(p_duration, dtype="Int64"),
                             "rental_period_status": [s.value for s in p_status],
                             "pricing_eligible": parent_ok}, index=jobs.index)
    (job_s, job_d), (own_s, own_d) = (detail_periods[pair] for pair in DETAIL_PERIODS)
    out_cars = pd.DataFrame({"job_rental_duration_days": pd.array(job_d, dtype="Int64"),
                             "rental_duration_days": pd.array(own_d, dtype="Int64"),
                             "job_rental_period_status": [s.value for s in job_s],
                             "rental_period_status": [s.value for s in own_s],
                             "rental_dates_agree": agree_all, "pricing_eligible": eligible}, index=cars.index)
    return DerivedRentalPeriods(jobs=out_jobs, cars=out_cars)


def analysis_duration_cohort(derived: pd.DataFrame, *, minimum_days: int, maximum_days: int | None) -> pd.Series:
    """Rows of a study cohort: pricing-eligible rows whose ``rental_duration_days`` lies in the bounds.

    A separate, explicit analysis filter: it never changes validity, never
    re-admits an ineligible row, and its bounds are the caller's study design
    (an authority-backed analysis policy), not data rules.
    """
    if not isinstance(derived, pd.DataFrame) or not {"rental_duration_days", "pricing_eligible"} <= set(derived):
        raise TypeError("derived must be a DerivedRentalPeriods frame")
    if type(minimum_days) is not int or minimum_days < 0 or (
            maximum_days is not None and (type(maximum_days) is not int or maximum_days < minimum_days)):
        raise ValueError("cohort bounds must be integers with 0 <= minimum <= maximum")
    duration = derived["rental_duration_days"]
    inside = duration.ge(minimum_days).fillna(False)
    if maximum_days is not None:
        inside &= duration.le(maximum_days).fillna(False)
    return (derived["pricing_eligible"].astype(bool) & inside.astype(bool)).rename("in_cohort")
