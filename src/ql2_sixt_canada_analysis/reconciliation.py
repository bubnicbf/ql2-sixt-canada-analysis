"""Reconcile each job's declared detail count with the detail rows present.

The relationship is :data:`~ql2_sixt_canada_analysis.schemas.JOB_DETAIL_RELATIONSHIP`
(parent key, detail foreign key, expected-count column). For every job:

* **expected** = the job's declared detail count (if valid, see below);
* **observed** = cleaned detail rows whose complete foreign-key tuple equals
  the job's key (a job with no detail rows has observed ``0``);
* **difference** = observed - expected.

A job is *matched* when they are equal, *under-counted* when observed is
smaller and *over-counted* when larger.

Every declared-count column of the relationship
(``expected_detail_count_columns``: ``record_count`` and ``actual_car_rows``
for the project) is reconciled **independently** against the same observed
count (:class:`DeclaredCountFieldReport` per column); the declarations must
also agree with each other. A job is reconciled only when every declaration
is valid and matches; the contract passes only when every job is reconciled
and every detail row is linked. The original aggregate fields describe the
primary column. :func:`job_detail_count_results` gives the per-job evidence
(in memory only; identifiers are confidential). The comparison is per job, starting
from the jobs side, so equal global totals can never hide offsetting errors.

Scope agreement
---------------
When the relationship declares ``scope_agreement_columns`` (the project: the
city), every linked detail row must carry its parent job's scope, and both
values must be assignable (the single rule in
:mod:`~ql2_sixt_canada_analysis.city_integrity`). A job with a disagreeing
linked row is not reconciled, and the contract fails
(``parent_detail_scope_mismatch``): matching counts are not trusted while a
detail row may belong to another city's job. The comparison is exact; no
value is normalised, aliased or repaired.

Detail rows fall into exactly one of: **linked** (complete key matching a
job), **missing link** (any key component missing; counted, never dropped)
or **orphan** (complete key matching no job). Every detail row present is
counted - duplicates included; nothing is deduplicated.

Expected-count policy (the source column is never modified)
-----------------------------------------------------------
Each job's expected count is classified, in this precedence, as:

1. **missing** - a pandas missing value, or an empty / whitespace-only string;
2. **non-numeric** - booleans (never treated as 1/0), non-numeric text
   (including the text ``"nan"``) and other non-number objects;
3. **non-finite** - positive or negative infinity;
4. **fractional** - a finite number that is not a whole number;
5. **negative** - a whole number below zero;
6. otherwise **valid**. Whole-valued floats (``2.0``) and numeric strings for
   non-negative whole numbers (``"2"``) are valid; they are interpreted in a
   temporary array only.

Missing and invalid counts are counted, never filled, rounded, clamped or
dropped, and they make the overall contract fail.

Preconditions (raise :class:`ReconciliationPreconditionError`)
-------------------------------------------------------------
Required columns exist (otherwise
:class:`~ql2_sixt_canada_analysis.schemas.RelationshipConfigurationError`);
key columns on both sides use the nullable string identifier dtype; the jobs
key satisfies its unique-key contract (complete and unique - duplicate jobs
are never collapsed or picked arbitrarily); and completely blank rows have
already been removed from both frames.

Empty-data policy
-----------------
Empty jobs with empty details are vacuously reconciled (presence and volume
are separate controls). Empty jobs with any detail rows fail (those rows are
orphans or missing links). Jobs with no detail rows pass only if every
expected count is valid and zero.

Pipeline position: load -> schema check -> blank-row removal -> identifier
dtype validation -> unique-key assessment -> this reconciliation.

Reports and errors hold aggregate integers only - no identifiers, keys, rows
or values - and nothing is printed, logged or written. Metrics computed from
the proprietary data are proprietary: keep them in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.city_integrity import scope_agreement
from ql2_sixt_canada_analysis.relationships import (
    RelationshipPreconditionError,
    _check_relationship_inputs,
    _link_detail_rows,
)
from ql2_sixt_canada_analysis.schemas import (
    JOB_DETAIL_RELATIONSHIP,
    JobDetailRelationshipDefinition,
    RelationshipConfigurationError,
)

__all__ = [
    "DeclaredCountFieldReport",
    "job_detail_count_results",
    "JobDetailReconciliationError",
    "JobDetailReconciliationReport",
    "ReconciliationPreconditionError",
    "RelationshipConfigurationError",
    "assess_job_detail_reconciliation",
    "validate_job_detail_reconciliation",
]

_MISSING, _NON_NUMERIC, _NON_FINITE, _FRACTIONAL, _NEGATIVE, _VALID = range(6)


# ----------------------------------------------------------------------- report


@dataclass(frozen=True, slots=True)
class DeclaredCountFieldReport:
    """Reconciliation of one declared-count column (contract name and counts only)."""

    column: str
    valid_job_count: int
    missing_job_count: int
    invalid_job_count: int
    matched_job_count: int
    under_counted_job_count: int
    over_counted_job_count: int

    @property
    def display_name(self) -> str:
        """Human-readable name of the declaration (for reports and notebooks)."""
        return self.column.replace("_", " ")

    @property
    def reconciled(self) -> bool:
        """Every job's declaration is valid and equals its observed detail count."""
        return (self.missing_job_count == 0 and self.invalid_job_count == 0
                and self.under_counted_job_count == 0 and self.over_counted_job_count == 0)


@dataclass(frozen=True, slots=True)
class JobDetailReconciliationReport:
    """Aggregate job-to-detail reconciliation metrics (plain ints, no values).

    Job metrics:
        job_count, valid_expected_count_job_count,
        missing_expected_count_job_count, invalid_expected_count_job_count
        (= non_numeric + non_finite + fractional + negative),
        matched_job_count, under_counted_job_count, over_counted_job_count
        (over valid jobs only), jobs_without_linked_details_count (any job),
        valid_expected_detail_total, absolute_discrepancy_total and
        net_discrepancy (observed - expected, over valid jobs).

    Detail metrics:
        detail_row_count = linked_detail_row_count
        + missing_link_detail_row_count + orphan_detail_row_count;
        distinct_orphan_key_count.

    ``preconditions_satisfied`` is always ``True`` on a returned report:
    assessment raises instead of reporting when preconditions fail.
    """

    job_count: int
    detail_row_count: int
    valid_expected_count_job_count: int
    missing_expected_count_job_count: int
    non_numeric_expected_count_job_count: int
    non_finite_expected_count_job_count: int
    fractional_expected_count_job_count: int
    negative_expected_count_job_count: int
    matched_job_count: int
    under_counted_job_count: int
    over_counted_job_count: int
    jobs_without_linked_details_count: int
    valid_expected_detail_total: int
    linked_detail_row_count: int
    missing_link_detail_row_count: int
    orphan_detail_row_count: int
    distinct_orphan_key_count: int
    absolute_discrepancy_total: int
    net_discrepancy: int
    count_fields: tuple[DeclaredCountFieldReport, ...] = ()
    declared_counts_disagree_job_count: int = 0
    reconciled_job_count: int = 0
    preconditions_satisfied: bool = True
    #: Linked detail rows (and their jobs) whose scope disagrees with the parent's.
    scope_mismatch_detail_row_count: int = 0
    scope_mismatch_job_count: int = 0

    def __post_init__(self) -> None:
        # Programmer invariants; a violation is a bug in this module.
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "preconditions_satisfied":
                assert isinstance(value, bool)
            elif f.name == "count_fields":
                assert isinstance(value, tuple) and value, f.name
            else:
                assert type(value) is int, f.name
                if f.name != "net_discrepancy":
                    assert value >= 0, f.name
        assert self.job_count == (self.valid_expected_count_job_count
                                  + self.missing_expected_count_job_count
                                  + self.invalid_expected_count_job_count)
        assert self.valid_expected_count_job_count == (
            self.matched_job_count + self.under_counted_job_count + self.over_counted_job_count)
        assert self.jobs_without_linked_details_count <= self.job_count
        assert self.detail_row_count == (self.linked_detail_row_count
                                         + self.missing_link_detail_row_count
                                         + self.orphan_detail_row_count)
        assert self.distinct_orphan_key_count <= self.orphan_detail_row_count
        assert (self.distinct_orphan_key_count == 0) == (self.orphan_detail_row_count == 0)
        assert abs(self.net_discrepancy) <= self.absolute_discrepancy_total
        assert (self.absolute_discrepancy_total == 0) == (
            self.under_counted_job_count == 0 and self.over_counted_job_count == 0)
        for field_report in self.count_fields:
            assert self.job_count == (field_report.valid_job_count + field_report.missing_job_count
                                      + field_report.invalid_job_count)
            assert field_report.valid_job_count == (field_report.matched_job_count
                                                    + field_report.under_counted_job_count
                                                    + field_report.over_counted_job_count)
        assert self.reconciled_job_count <= self.job_count
        assert self.scope_mismatch_detail_row_count <= self.linked_detail_row_count
        assert self.scope_mismatch_job_count <= self.job_count
        assert (self.scope_mismatch_detail_row_count == 0) == (self.scope_mismatch_job_count == 0)

    @property
    def invalid_expected_count_job_count(self) -> int:
        return (self.non_numeric_expected_count_job_count + self.non_finite_expected_count_job_count
                + self.fractional_expected_count_job_count + self.negative_expected_count_job_count)

    @property
    def all_expected_counts_valid(self) -> bool:
        return self.valid_expected_count_job_count == self.job_count

    @property
    def all_valid_jobs_reconciled(self) -> bool:
        return self.under_counted_job_count == 0 and self.over_counted_job_count == 0

    @property
    def all_details_linked(self) -> bool:
        return self.linked_detail_row_count == self.detail_row_count

    @property
    def declared_counts_agree(self) -> bool:
        """No job has valid declarations that disagree with each other."""
        return self.declared_counts_disagree_job_count == 0

    @property
    def all_count_fields_reconciled(self) -> bool:
        return all(f.reconciled for f in self.count_fields)

    @property
    def parent_detail_scope_agrees(self) -> bool:
        """Every linked detail row carries its parent job's (assignable) scope."""
        return self.scope_mismatch_detail_row_count == 0

    @property
    def declared_counts_reconciled(self) -> bool:
        """The count part of the contract: every declaration valid, matched and agreeing; all linked."""
        return (self.preconditions_satisfied and self.all_expected_counts_valid
                and self.all_valid_jobs_reconciled and self.all_count_fields_reconciled
                and self.declared_counts_agree and self.all_details_linked)

    @property
    def is_reconciled(self) -> bool:
        """The full contract: counts reconciled for every job and parent/detail scope agreement."""
        return (self.declared_counts_reconciled and self.parent_detail_scope_agrees
                and self.reconciled_job_count == self.job_count)

    @property
    def violations(self) -> tuple[str, ...]:
        """Safe violation categories present across every declared count (empty when reconciled)."""
        fields_ = self.count_fields
        checks = (
            ("missing_expected_count", sum(f.missing_job_count for f in fields_)),
            ("invalid_expected_count", sum(f.invalid_job_count for f in fields_)),
            ("under_count", sum(f.under_counted_job_count for f in fields_)),
            ("over_count", sum(f.over_counted_job_count for f in fields_)),
            ("declared_counts_disagree", self.declared_counts_disagree_job_count),
            ("missing_link", self.missing_link_detail_row_count),
            ("orphan_detail", self.orphan_detail_row_count),
            ("parent_detail_scope_mismatch", self.scope_mismatch_detail_row_count),
        )
        return tuple(name for name, count in checks if count)


# ------------------------------------------------------------------- exceptions


class ReconciliationPreconditionError(RelationshipPreconditionError):
    """Reconciliation would be ambiguous: a structural precondition failed.

    Same attributes as
    :class:`~ql2_sixt_canada_analysis.relationships.RelationshipPreconditionError`
    (``reason``, ``role``, ``unique_key_report``).
    """

    control = "Job-to-detail reconciliation"


class JobDetailReconciliationError(Exception):
    """Strict validation found count mismatches, invalid counts or unlinked details.

    The message lists violation categories only; the aggregate report is on
    ``report`` (do not log it for proprietary data).
    """

    def __init__(self, report: JobDetailReconciliationReport) -> None:
        super().__init__("Job-to-detail reconciliation failed: " + ", ".join(report.violations) + ".")
        self.report = report


# ------------------------------------------------------------------- public API


def assess_job_detail_reconciliation(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> JobDetailReconciliationReport:
    """Reconcile declared vs observed detail counts per job; return aggregates.

    Never raises for ordinary source mismatches. Neither frame is modified,
    sorted or copied in full; only the relationship and count columns are read.

    Raises:
        TypeError: Invalid argument types.
        RelationshipConfigurationError: A configured column is absent.
        ReconciliationPreconditionError: See the module docstring.
    """
    _check_relationship_inputs(jobs, cars, relationship, ReconciliationPreconditionError,
                               require_expected_count=True, require_scope=True)
    count_column = relationship.expected_detail_count_column

    # --- expected counts (temporary arrays; source untouched)
    category, expected = _classify_expected_counts(jobs[count_column])
    valid = category == _VALID

    # --- observed counts per job and detail-row categories (shared, tuple-keyed)
    linkage = _link_detail_rows(jobs, cars, relationship)
    observed = linkage.observed_per_parent  # left from jobs: every job, zero when none
    difference = observed[valid] - expected[valid]
    agreement = scope_agreement(jobs, cars, relationship)
    per_job = _per_job(jobs, observed, relationship, agreement.parent_mismatch)

    return JobDetailReconciliationReport(
        job_count=len(jobs),
        detail_row_count=len(cars),
        valid_expected_count_job_count=int(valid.sum()),
        missing_expected_count_job_count=int((category == _MISSING).sum()),
        non_numeric_expected_count_job_count=int((category == _NON_NUMERIC).sum()),
        non_finite_expected_count_job_count=int((category == _NON_FINITE).sum()),
        fractional_expected_count_job_count=int((category == _FRACTIONAL).sum()),
        negative_expected_count_job_count=int((category == _NEGATIVE).sum()),
        matched_job_count=int((difference == 0).sum()),
        under_counted_job_count=int((difference < 0).sum()),
        over_counted_job_count=int((difference > 0).sum()),
        jobs_without_linked_details_count=int((observed == 0).sum()),
        valid_expected_detail_total=int(expected[valid].sum()),
        linked_detail_row_count=linkage.linked,
        missing_link_detail_row_count=linkage.missing_link,
        orphan_detail_row_count=linkage.orphan,
        distinct_orphan_key_count=linkage.distinct_orphan_keys,
        absolute_discrepancy_total=int(np.abs(difference).sum()),
        net_discrepancy=int(difference.sum()),
        count_fields=per_job.field_reports,
        declared_counts_disagree_job_count=int((~per_job.agree & per_job.all_valid).sum()),
        reconciled_job_count=int(per_job.reconciled.sum()),
        scope_mismatch_detail_row_count=int(agreement.detail_mismatch.sum()),
        scope_mismatch_job_count=int(agreement.parent_mismatch.sum()),
    )


def job_detail_count_results(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> pd.DataFrame:
    """Per-job evidence, in memory only (a new frame on every call).

    One row per job (zero-detail jobs included, observed ``0``), sorted by the
    parent key: the key columns, ``observed_detail_count``, each declared-count
    column as given, ``<column>_valid`` and ``<column>_matches`` per
    declaration, ``declared_counts_agree``, ``scope_agrees`` and ``job_reconciled``. Holds
    confidential identifiers: never print, log or persist it.
    """
    _check_relationship_inputs(jobs, cars, relationship, ReconciliationPreconditionError,
                               require_expected_count=True, require_scope=True)
    observed = _link_detail_rows(jobs, cars, relationship).observed_per_parent
    mismatch = scope_agreement(jobs, cars, relationship).parent_mismatch
    per_job = _per_job(jobs, observed, relationship, mismatch)
    frame = jobs.loc[:, list(relationship.parent_key_columns)].copy()
    frame["observed_detail_count"] = observed
    for column in relationship.expected_detail_count_columns:
        frame[column] = jobs[column].to_numpy(copy=True)
        frame[f"{column}_valid"] = per_job.valid[column]
        frame[f"{column}_matches"] = per_job.matches[column]
    frame["declared_counts_agree"] = per_job.agree
    frame["scope_agrees"] = ~mismatch
    frame["job_reconciled"] = per_job.reconciled
    keys = list(relationship.parent_key_columns)
    return frame.sort_values(keys, kind="mergesort", na_position="last").reset_index(drop=True)


def validate_job_detail_reconciliation(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> JobDetailReconciliationReport:
    """Assess, then return the report if reconciled or raise :class:`JobDetailReconciliationError`."""
    report = assess_job_detail_reconciliation(jobs, cars, relationship)
    if not report.is_reconciled:
        raise JobDetailReconciliationError(report)
    return report


# ---------------------------------------------------------------------- helpers


@dataclass(frozen=True, slots=True, eq=False)
class _PerJob:
    valid: dict
    matches: dict
    all_valid: np.ndarray
    agree: np.ndarray
    reconciled: np.ndarray
    field_reports: tuple[DeclaredCountFieldReport, ...]


def _per_job(jobs: pd.DataFrame, observed: np.ndarray, relationship: JobDetailRelationshipDefinition,
             scope_mismatch: np.ndarray | None = None) -> _PerJob:
    """Independent per-declaration validity/match flags, agreement and job result.

    A job with a scope-disagreeing linked detail row is never reconciled.
    """
    valid, matches, values, reports = {}, {}, [], []
    for column in relationship.expected_detail_count_columns:
        category, expected = _classify_expected_counts(jobs[column])
        ok = category == _VALID
        difference = observed - expected
        valid[column] = ok
        matches[column] = ok & (difference == 0)
        values.append(np.where(ok, expected, -1))
        reports.append(DeclaredCountFieldReport(
            column=column, valid_job_count=int(ok.sum()),
            missing_job_count=int((category == _MISSING).sum()),
            invalid_job_count=int((~ok & (category != _MISSING)).sum()),
            matched_job_count=int((ok & (difference == 0)).sum()),
            under_counted_job_count=int((ok & (difference < 0)).sum()),
            over_counted_job_count=int((ok & (difference > 0)).sum()),
        ))
    all_valid = np.logical_and.reduce([valid[c] for c in valid]) if valid else np.ones(len(jobs), bool)
    stacked = np.vstack(values) if values else np.zeros((1, len(jobs)), dtype=np.int64)
    agree = all_valid & (stacked == stacked[0]).all(axis=0)
    reconciled = np.logical_and.reduce([matches[c] for c in matches]) & agree
    if scope_mismatch is not None:
        reconciled = reconciled & ~scope_mismatch
    return _PerJob(valid=valid, matches=matches, all_valid=all_valid, agree=agree, reconciled=reconciled,
                   field_reports=tuple(reports))


def _classify_expected_counts(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (category codes, int64 expected values valid where category is VALID)."""
    n = len(series)
    if pd.api.types.is_bool_dtype(series.dtype):
        missing = series.isna().to_numpy(dtype=bool)
        non_numeric = ~missing
        numeric = np.full(n, np.nan)
    elif pd.api.types.is_numeric_dtype(series.dtype):
        missing = series.isna().to_numpy(dtype=bool)
        non_numeric = np.zeros(n, dtype=bool)
        numeric = series.to_numpy(dtype="float64", na_value=np.nan)
    else:
        values = series.astype(object)
        is_text = values.map(lambda v: isinstance(v, str)).to_numpy(dtype=bool)
        blank_text = values.map(lambda v: isinstance(v, str) and not v.strip()).to_numpy(dtype=bool)
        is_bool = values.map(lambda v: isinstance(v, (bool, np.bool_))).to_numpy(dtype=bool)
        is_number = values.map(
            lambda v: isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_))
        ).to_numpy(dtype=bool)
        missing = series.isna().to_numpy(dtype=bool) | blank_text
        parsed_text = pd.to_numeric(values.where(is_text & ~blank_text), errors="coerce")
        parsed_text = parsed_text.to_numpy(dtype="float64", na_value=np.nan)
        number_values = pd.to_numeric(values.where(is_number), errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan)
        numeric = np.where(is_number, number_values, parsed_text)
        unparsable_text = is_text & ~blank_text & np.isnan(parsed_text)
        other_object = ~(is_text | is_number | is_bool)
        non_numeric = ~missing & (is_bool | unparsable_text | other_object)

    category = np.full(n, _VALID, dtype=np.int8)
    candidate = ~missing & ~non_numeric
    with np.errstate(invalid="ignore"):
        non_finite = candidate & ~np.isfinite(numeric)
        fractional = candidate & ~non_finite & (np.mod(numeric, 1) != 0)
        negative = candidate & ~non_finite & ~fractional & (numeric < 0)
    for mask, code in ((negative, _NEGATIVE), (fractional, _FRACTIONAL), (non_finite, _NON_FINITE),
                       (non_numeric, _NON_NUMERIC), (missing, _MISSING)):
        category[mask] = code  # later assignments take precedence
    expected = np.where(category == _VALID, np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0), 0)
    return category, expected.astype(np.int64)

