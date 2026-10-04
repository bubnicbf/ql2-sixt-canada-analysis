"""Validate the jobs-to-cars one-to-many relationship and perform trusted joins.

The relationship is :data:`~ql2_sixt_canada_analysis.schemas.JOB_DETAIL_RELATIONSHIP`.
``jobs`` is the **one** (parent) side and ``cars`` the **many** (detail) side.

One-to-many contract
--------------------
* Every parent key is present and identifies exactly one jobs row.
* A job may have zero, one or many detail rows; a detail foreign key may
  therefore repeat - that is expected, not a violation.
* Every detail row has a complete foreign key that matches exactly one job.
  A detail row with any missing component is a **missing-link** row; one
  with a complete key absent from jobs is an **orphan** row. The two are
  counted separately and every detail row is in exactly one of linked,
  missing-link or orphan.
* A parent-left merge with pandas ``validate="one_to_many"`` succeeds, and
  **row conservation** holds: the left join has exactly
  ``linked detail rows + parents without details`` rows, every linked detail
  appears once and every detail-less parent appears once (no Cartesian
  multiplication). Equal input totals are never accepted as a substitute.

This control is distinct from :mod:`~ql2_sixt_canada_analysis.unique_keys`
(each dataset's own key) and :mod:`~ql2_sixt_canada_analysis.reconciliation`
(declared vs observed counts per job); it reuses their definitions and the
shared precondition and linkage helpers defined here.

Preconditions (raise :class:`RelationshipPreconditionError`)
-----------------------------------------------------------
Relationship columns exist (else
:class:`~ql2_sixt_canada_analysis.schemas.RelationshipConfigurationError`);
both sides' key columns use the nullable string identifier dtype; the jobs
key is complete and unique (duplicate parents are never collapsed, chosen
arbitrarily or merged); completely blank rows have been removed.

Trusted join
------------
:func:`join_jobs_to_details` strictly validates first, then merges jobs (left)
with cars (right) using ``how="left"``, ``validate="one_to_many"`` and
``sort=False``, and re-checks the row count. Parent order is preserved and,
within a parent, detail rows keep their source order; the result has a fresh
``RangeIndex``. Key columns with the same name on both sides appear once;
when names differ both are kept (the detail key is missing for parents
without details). Same-named non-key columns get the relationship's
``parent_suffix`` / ``detail_suffix`` (``_job`` / ``_detail``); other columns
keep their names. No merge-indicator column is exposed. Failures raise
:class:`OneToManyRelationshipError` or :class:`ValidatedJoinError`; no
partially trusted frame is ever returned.

Empty-data policy
-----------------
Empty jobs and empty cars are vacuously valid (presence/volume is a separate
control). Empty jobs with any detail rows fail (no detail can link). Jobs
with empty cars are valid; each job appears once in the left join (count
reconciliation may still fail if non-zero counts were declared).

Nothing here modifies, sorts, deduplicates, drops, writes or logs data.
Reports and errors hold aggregate integers and booleans only. A frozen
:class:`ValidatedJoinResult` does not make its DataFrame immutable; never
persist joined proprietary data.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
from pandas.errors import MergeError

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
    "OneToManyJoinReport",
    "OneToManyRelationshipError",
    "RelationshipConfigurationError",
    "RelationshipPreconditionError",
    "ValidatedJoinError",
    "ValidatedJoinResult",
    "assess_one_to_many_join",
    "join_jobs_to_details",
    "validate_one_to_many_join",
]

# Internal positional labels for the key-only cardinality probe. They never
# reach a public result.
_PARENT_POS = "__ql2_parent_position__"
_DETAIL_POS = "__ql2_detail_position__"


# ------------------------------------------------------------------- exceptions


class RelationshipPreconditionError(Exception):
    """A structural precondition for using the relationship failed.

    Attributes:
        reason: ``"identifier_dtype"``, ``"parent_key_missing"``,
            ``"parent_key_duplicate"`` or ``"blank_rows_present"``.
        role: The dataset concerned.
        unique_key_report: The parent :class:`UniqueKeyReport` (counts only)
            for parent-key failures, else ``None``.
    """

    control = "Jobs-to-details relationship"

    def __init__(self, reason: str, role: DatasetKey,
                 unique_key_report: UniqueKeyReport | None = None) -> None:
        super().__init__(f"{self.control} precondition failed for '{role}': {reason}.")
        self.reason = reason
        self.role = role
        self.unique_key_report = unique_key_report


class OneToManyRelationshipError(Exception):
    """Strict validation found the one-to-many contract violated.

    The message lists violation categories only; the aggregate report is on
    ``report``.
    """

    def __init__(self, report: OneToManyJoinReport) -> None:
        super().__init__("One-to-many relationship contract failed: " + ", ".join(report.violations) + ".")
        self.report = report


class ValidatedJoinError(Exception):
    """The trusted join could not be produced safely (e.g. a pandas ``MergeError``).

    The original exception, if any, is the ``__cause__``; the message holds
    no identifiers or row contents.
    """


# ----------------------------------------------------------------------- report


@dataclass(frozen=True, slots=True)
class OneToManyJoinReport:
    """Aggregate one-to-many relationship metrics (plain ints and bools only).

    The precondition flags (``parent_key_complete``, ``parent_key_unique``,
    ``relationship_dtypes_compatible``) are always ``True`` on a returned
    report, because assessment raises when they fail.
    """

    parent_row_count: int
    detail_row_count: int
    parents_with_details_count: int
    parents_without_details_count: int
    linked_detail_row_count: int
    missing_link_detail_row_count: int
    orphan_detail_row_count: int
    distinct_orphan_key_count: int
    expected_left_join_row_count: int
    actual_left_join_row_count: int
    parent_key_complete: bool
    parent_key_unique: bool
    relationship_dtypes_compatible: bool
    cardinality_validated: bool
    row_conservation_holds: bool

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith("_count"):
                assert type(value) is int and value >= 0, f.name
            else:
                assert type(value) is bool, f.name
        assert self.parent_row_count == self.parents_with_details_count + self.parents_without_details_count
        assert self.detail_row_count == (self.linked_detail_row_count + self.missing_link_detail_row_count
                                         + self.orphan_detail_row_count)
        assert self.distinct_orphan_key_count <= self.orphan_detail_row_count
        assert (self.distinct_orphan_key_count == 0) == (self.orphan_detail_row_count == 0)
        assert self.parents_with_details_count <= self.linked_detail_row_count
        assert self.expected_left_join_row_count == (self.linked_detail_row_count
                                                     + self.parents_without_details_count)

    @property
    def all_details_linked(self) -> bool:
        return self.linked_detail_row_count == self.detail_row_count

    @property
    def is_valid(self) -> bool:
        """The full one-to-many contract holds."""
        return (self.parent_key_complete and self.parent_key_unique
                and self.relationship_dtypes_compatible and self.all_details_linked
                and self.cardinality_validated and self.row_conservation_holds)

    @property
    def violations(self) -> tuple[str, ...]:
        """Safe violation categories present (empty when valid)."""
        checks = (
            ("missing_link", self.missing_link_detail_row_count > 0),
            ("orphan_detail", self.orphan_detail_row_count > 0),
            ("cardinality", not self.cardinality_validated),
            ("row_conservation", not self.row_conservation_holds),
        )
        return tuple(name for name, failed in checks if failed)


@dataclass(frozen=True, slots=True)
class ValidatedJoinResult:
    """A trusted parent-left join and the report that validated it.

    ``joined`` is an ordinary (mutable) DataFrame held in memory only.
    """

    joined: pd.DataFrame
    report: OneToManyJoinReport


# ----------------------------------------------------- shared relationship helpers


@dataclass(frozen=True, slots=True)
class _DetailLinkage:
    """Per-parent observed detail counts and detail-row categories (internal)."""

    observed_per_parent: np.ndarray  # aligned with the parent frame's rows
    linked: int
    missing_link: int
    orphan: int
    distinct_orphan_keys: int


def _check_relationship_inputs(
    jobs: object,
    cars: object,
    relationship: object,
    error_type: type[RelationshipPreconditionError] = RelationshipPreconditionError,
    require_expected_count: bool = False,
) -> JobDetailRelationshipDefinition:
    """Shared configuration and precondition checks (used by reconciliation too)."""
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be pandas DataFrames")
    if not isinstance(relationship, JobDetailRelationshipDefinition):
        raise TypeError(f"expected a JobDetailRelationshipDefinition, got {type(relationship).__name__}")
    parent_keys, detail_keys = relationship.parent_key_columns, relationship.detail_key_columns
    extra_parent_columns = (relationship.expected_detail_count_column,) if require_expected_count else ()
    if len(parent_keys) != len(detail_keys):  # guards definitions altered after construction
        raise RelationshipConfigurationError("parent and detail keys must have equal length")
    for frame, columns, role in ((jobs, (*parent_keys, *extra_parent_columns), relationship.parent),
                                 (cars, detail_keys, relationship.detail)):
        absent = tuple(c for c in columns if c not in frame.columns)
        if absent:
            raise RelationshipConfigurationError(
                f"The '{role}' frame lacks {len(absent)} relationship column(s).", absent
            )
    for frame, columns, role in ((jobs, parent_keys, relationship.parent),
                                 (cars, detail_keys, relationship.detail)):
        if not all(is_identifier_dtype(frame[c].dtype) for c in columns):
            raise error_type("identifier_dtype", role)
    key_report = assess_unique_key(jobs, relationship.parent_definition)
    if not key_report.is_complete:
        raise error_type("parent_key_missing", relationship.parent, key_report)
    if not key_report.is_unique:
        raise error_type("parent_key_duplicate", relationship.parent, key_report)
    for frame, role in ((jobs, relationship.parent), (cars, relationship.detail)):
        if bool(completely_blank_row_mask(frame).any()):
            raise error_type("blank_rows_present", role)
    return relationship


def _link_detail_rows(
    jobs: pd.DataFrame, cars: pd.DataFrame, relationship: JobDetailRelationshipDefinition
) -> _DetailLinkage:
    """Classify detail rows and count linked details per parent (tuple keys, no concatenation)."""
    parent_keys, detail_keys = relationship.parent_key_columns, relationship.detail_key_columns
    detail = cars.loc[:, list(detail_keys)]
    complete = detail.notna().all(axis=1).to_numpy()
    counts = detail.loc[complete].value_counts(sort=False, dropna=False)
    counts.index = counts.index.set_names(list(parent_keys))
    parent_index = pd.MultiIndex.from_frame(jobs.loc[:, list(parent_keys)])
    known = counts.index.isin(parent_index)
    observed = counts.reindex(parent_index, fill_value=0).to_numpy(dtype=np.int64)
    values = counts.to_numpy()
    linkage = _DetailLinkage(
        observed_per_parent=observed,
        linked=int(values[known].sum()),
        missing_link=int((~complete).sum()),
        orphan=int(values[~known].sum()),
        distinct_orphan_keys=int((~known).sum()),
    )
    assert linkage.linked == int(observed.sum())  # parent key is unique (precondition)
    return linkage


# ------------------------------------------------------------------- public API


def assess_one_to_many_join(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> OneToManyJoinReport:
    """Assess the one-to-many contract; returns a report for ordinary violations.

    Runs a key-only, cardinality-validated parent-left merge to verify row
    conservation without copying non-key data. Inputs are not modified.

    Raises:
        TypeError: Invalid argument types.
        RelationshipConfigurationError: A configured column is absent.
        RelationshipPreconditionError: See the module docstring.
    """
    _check_relationship_inputs(jobs, cars, relationship)
    linkage = _link_detail_rows(jobs, cars, relationship)
    parents_with = int((linkage.observed_per_parent > 0).sum())
    parents_without = len(jobs) - parents_with
    expected_rows = linkage.linked + parents_without
    cardinality_ok, actual_rows, conserved = _probe_left_join(jobs, cars, relationship, linkage, parents_without)
    return OneToManyJoinReport(
        parent_row_count=len(jobs),
        detail_row_count=len(cars),
        parents_with_details_count=parents_with,
        parents_without_details_count=parents_without,
        linked_detail_row_count=linkage.linked,
        missing_link_detail_row_count=linkage.missing_link,
        orphan_detail_row_count=linkage.orphan,
        distinct_orphan_key_count=linkage.distinct_orphan_keys,
        expected_left_join_row_count=expected_rows,
        actual_left_join_row_count=actual_rows,
        parent_key_complete=True,
        parent_key_unique=True,
        relationship_dtypes_compatible=True,
        cardinality_validated=cardinality_ok,
        row_conservation_holds=conserved and actual_rows == expected_rows,
    )


def validate_one_to_many_join(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> OneToManyJoinReport:
    """Assess, then return the report if valid or raise :class:`OneToManyRelationshipError`."""
    report = assess_one_to_many_join(jobs, cars, relationship)
    if not report.is_valid:
        raise OneToManyRelationshipError(report)
    return report


def join_jobs_to_details(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> ValidatedJoinResult:
    """Return the trusted parent-left join of jobs to cars, or raise.

    Strictly validates the relationship first, then merges with
    ``validate="one_to_many"`` and verifies row conservation on the result.
    See the module docstring for ordering, key and collision policies.

    Raises:
        RelationshipPreconditionError, RelationshipConfigurationError: As for
            assessment (also raised for colliding suffixed column names).
        OneToManyRelationshipError: The contract is violated (missing-link or
            orphan details, cardinality or conservation failure).
        ValidatedJoinError: The merge failed or produced an unexpected shape;
            the pandas exception is the ``__cause__``.
    """
    report = validate_one_to_many_join(jobs, cars, relationship)
    _check_output_columns(jobs, cars, relationship)
    try:
        joined = pd.merge(
            jobs, cars, how="left",
            left_on=list(relationship.parent_key_columns),
            right_on=list(relationship.detail_key_columns),
            validate="one_to_many", sort=False,
            suffixes=(relationship.parent_suffix, relationship.detail_suffix),
        )
    except MergeError as exc:
        raise ValidatedJoinError("The validated jobs-to-details merge failed its cardinality check.") from exc
    if len(joined) != report.expected_left_join_row_count or not joined.columns.is_unique:
        raise ValidatedJoinError("The validated jobs-to-details merge did not conserve rows.")
    return ValidatedJoinResult(joined=joined, report=report)


# ---------------------------------------------------------------------- helpers


def _probe_left_join(
    jobs: pd.DataFrame, cars: pd.DataFrame, relationship: JobDetailRelationshipDefinition,
    linkage: _DetailLinkage, parents_without: int,
) -> tuple[bool, int, bool]:
    """Key-only parent-left merge: (cardinality ok, joined rows, conservation ok)."""
    left = jobs.loc[:, list(relationship.parent_key_columns)]
    right = cars.loc[:, list(relationship.detail_key_columns)]
    left.columns = [f"__ql2_key_{i}__" for i in range(left.shape[1])]
    right.columns = list(left.columns)
    left[_PARENT_POS] = np.arange(len(left))
    right[_DETAIL_POS] = np.arange(len(right))
    try:
        probe = pd.merge(left, right, how="left", on=list(left.columns[:-1]),
                         validate="one_to_many", sort=False)
    except MergeError:
        return False, 0, False
    detail_pos = probe[_DETAIL_POS]
    matched = detail_pos.notna().to_numpy()
    parent_pos = probe[_PARENT_POS].to_numpy()
    conserved = (
        int(matched.sum()) == linkage.linked                                   # each linked detail once
        and bool(detail_pos[matched].is_unique)
        and int((~matched).sum()) == parents_without                           # each empty parent once
        and bool(pd.Series(parent_pos[~matched]).is_unique)
        and bool(pd.Series(parent_pos).is_monotonic_increasing)                # parent order kept
        and np.array_equal(np.bincount(parent_pos[matched].astype(np.int64), minlength=len(jobs)),
                           linkage.observed_per_parent)
    )
    return True, len(probe), conserved


def _check_output_columns(
    jobs: pd.DataFrame, cars: pd.DataFrame, relationship: JobDetailRelationshipDefinition
) -> None:
    """Reject suffix choices that would make output column names ambiguous."""
    parent_keys, detail_keys = relationship.parent_key_columns, relationship.detail_key_columns
    shared_keys = {p for p, d in zip(parent_keys, detail_keys, strict=True) if p == d}
    overlap = (set(jobs.columns) & set(cars.columns)) - shared_keys
    names = [f"{c}{relationship.parent_suffix}" if c in overlap else c for c in jobs.columns]
    names += [f"{c}{relationship.detail_suffix}" if c in overlap else c
              for c in cars.columns if c not in shared_keys]
    if len(names) != len(set(names)):
        raise RelationshipConfigurationError("join suffixes would create duplicate column names")
