"""Reconcile each job's declared detail count with the detail rows present.

The relationship is :data:`~ql2_sixt_canada_analysis.schemas.JOB_DETAIL_RELATIONSHIP`
(parent key, detail foreign key, expected-count column). For every job:

* **expected** = the job's declared detail count (if valid, see below);
* **observed** = cleaned detail rows whose complete foreign-key tuple equals
  the job's key (a job with no detail rows has observed ``0``);
* **difference** = observed - expected.

A job is *matched* when they are equal, *under-counted* when observed is
smaller and *over-counted* when larger. The comparison is per job, starting
from the jobs side, so equal global totals can never hide offsetting errors.

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

from ql2_sixt_canada_analysis.identifiers import is_identifier_dtype
from ql2_sixt_canada_analysis.quality import completely_blank_row_mask
from ql2_sixt_canada_analysis.schemas import (
    JOB_DETAIL_RELATIONSHIP,
    DatasetKey,
    JobDetailRelationshipDefinition,
    RelationshipConfigurationError,
)
from ql2_sixt_canada_analysis.unique_keys import UniqueKeyReport, assess_unique_key

__all__ = [
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
    preconditions_satisfied: bool = True

    def __post_init__(self) -> None:
        # Programmer invariants; a violation is a bug in this module.
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "preconditions_satisfied":
                assert isinstance(value, bool)
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
    def is_reconciled(self) -> bool:
        """The full contract: valid counts, every job matched, every detail linked."""
        return (self.preconditions_satisfied and self.all_expected_counts_valid
                and self.all_valid_jobs_reconciled and self.all_details_linked)

    @property
    def violations(self) -> tuple[str, ...]:
        """Safe violation categories present (empty when reconciled)."""
        checks = (
            ("missing_expected_count", self.missing_expected_count_job_count),
            ("invalid_expected_count", self.invalid_expected_count_job_count),
            ("under_count", self.under_counted_job_count),
            ("over_count", self.over_counted_job_count),
            ("missing_link", self.missing_link_detail_row_count),
            ("orphan_detail", self.orphan_detail_row_count),
        )
        return tuple(name for name, count in checks if count)


# ------------------------------------------------------------------- exceptions


class ReconciliationPreconditionError(Exception):
    """Reconciliation would be ambiguous: a structural precondition failed.

    Attributes:
        reason: ``"identifier_dtype"``, ``"parent_key_missing"``,
            ``"parent_key_duplicate"`` or ``"blank_rows_present"``.
        role: The dataset concerned.
        unique_key_report: The parent :class:`UniqueKeyReport` (counts only)
            for parent-key failures, else ``None``.
    """

    def __init__(self, reason: str, role: DatasetKey,
                 unique_key_report: UniqueKeyReport | None = None) -> None:
        super().__init__(f"Job-to-detail reconciliation precondition failed for '{role}': {reason}.")
        self.reason = reason
        self.role = role
        self.unique_key_report = unique_key_report


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
    parent_keys, detail_keys = _check_inputs(jobs, cars, relationship)
    count_column = relationship.expected_detail_count_column

    # --- expected counts (temporary arrays; source untouched)
    category, expected = _classify_expected_counts(jobs[count_column])
    valid = category == _VALID

    # --- observed counts per complete detail key tuple
    detail = cars.loc[:, list(detail_keys)]
    complete = detail.notna().all(axis=1).to_numpy()
    counts = detail.loc[complete].value_counts(sort=False, dropna=False)  # tuple-keyed, no concatenation
    counts.index = counts.index.set_names(list(parent_keys))
    parent_index = pd.MultiIndex.from_frame(jobs.loc[:, list(parent_keys)])
    known = counts.index.isin(parent_index)
    observed = counts.reindex(parent_index, fill_value=0).to_numpy(dtype=np.int64)  # left: every job

    difference = observed[valid] - expected[valid]
    linked = int(counts.to_numpy()[known].sum())
    assert linked == int(observed.sum())  # each detail key matches at most one (unique) job

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
        linked_detail_row_count=linked,
        missing_link_detail_row_count=int((~complete).sum()),
        orphan_detail_row_count=int(counts.to_numpy()[~known].sum()),
        distinct_orphan_key_count=int((~known).sum()),
        absolute_discrepancy_total=int(np.abs(difference).sum()),
        net_discrepancy=int(difference.sum()),
    )


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


def _check_inputs(
    jobs: object, cars: object, relationship: object
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be pandas DataFrames")
    if not isinstance(relationship, JobDetailRelationshipDefinition):
        raise TypeError(f"expected a JobDetailRelationshipDefinition, got {type(relationship).__name__}")
    parent_keys, detail_keys = relationship.parent_key_columns, relationship.detail_key_columns
    for frame, columns, role in (
        (jobs, (*parent_keys, relationship.expected_detail_count_column), relationship.parent),
        (cars, detail_keys, relationship.detail),
    ):
        absent = tuple(c for c in columns if c not in frame.columns)
        if absent:
            raise RelationshipConfigurationError(
                f"The '{role}' frame lacks {len(absent)} relationship column(s).", absent
            )
    for frame, columns, role in ((jobs, parent_keys, relationship.parent),
                                 (cars, detail_keys, relationship.detail)):
        if not all(is_identifier_dtype(frame[c].dtype) for c in columns):
            raise ReconciliationPreconditionError("identifier_dtype", role)
    key_report = assess_unique_key(jobs, relationship.parent_definition)
    if not key_report.is_complete:
        raise ReconciliationPreconditionError("parent_key_missing", relationship.parent, key_report)
    if not key_report.is_unique:
        raise ReconciliationPreconditionError("parent_key_duplicate", relationship.parent, key_report)
    for frame, role in ((jobs, relationship.parent), (cars, relationship.detail)):
        if bool(completely_blank_row_mask(frame).any()):
            raise ReconciliationPreconditionError("blank_rows_present", role)
    return parent_keys, detail_keys


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

