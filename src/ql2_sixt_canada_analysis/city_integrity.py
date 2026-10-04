"""City (collection-scope) integrity of jobs and their linked detail rows.

Every job is collected for one city, and every detail (cars) row repeats its
parent job's city. Two invariants follow, and both are checked here, once,
for every module that relies on them:

1. **Assignable job scope.** Each job's scope value
   (``coverage.parent_scope_columns`` - the job ``city``) must be assignable.
   :func:`unassignable_scope_mask` is the single definition of an
   unassignable value: missing (``None`` / ``NaN`` / ``pd.NA``), not a
   string, empty, whitespace-only, **or carrying leading/trailing
   whitespace**. Values are never trimmed, case-folded or aliased: no
   normalisation contract for the city exists (the source's derived
   ``city_clean`` column has no documented authority), so a padded value is
   non-canonical and is classified as unassignable rather than being silently
   repaired into - or silently excluded from - a valid scope.

   Why an unassignable job cannot simply be excluded: stream continuity counts
   every job of a city as one collection event of that city's streams. A job
   whose city cannot be read belongs to *some* stream, but which one cannot be
   proven, so excluding it from every scope would hide a possibly missing
   event from all of them. It therefore invalidates the city/stream integrity
   contract as a whole; no city is ever guessed for it.

2. **Parent/detail agreement.** For every detail row whose complete key links
   to a parent job, each ``relationship.scope_agreement_columns`` pair must
   hold the same assignable value on both sides (exact equality: Vancouver
   and Calgary never compare equal). A row fails when the values differ or
   either side is unassignable. A row linked to several parents (a duplicated
   parent key) is compared with each of them. Orphan, missing-link and
   ambiguous-parent rows remain governed by the relationship controls
   (:mod:`~ql2_sixt_canada_analysis.relationships`); this check complements
   them and never repairs or drops a row.

:func:`assess_city_integrity` returns an immutable :class:`CityIntegrityReport`
with counts, typed :class:`CityIntegrityBlocker` values and *bounded*,
deterministic samples (sorted, at most :data:`CITY_INTEGRITY_SAMPLE_LIMIT`).
The samples hold confidential identifiers and source values: they are
excluded from ``repr`` and from every exception message, and must never be
printed, logged or persisted for the proprietary data.

The same rule feeds stream continuity
(:func:`~ql2_sixt_canada_analysis.streams.investigate_location_stream`),
reconciliation, the trusted-join gate, completeness and pricing readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.schemas import (
    EXPECTED_LOCATION_COVERAGE,
    JOB_DETAIL_RELATIONSHIP,
    JobDetailRelationshipDefinition,
    LocationCoverageConfigurationError,
    LocationCoverageDefinition,
    RelationshipConfigurationError,
)

__all__ = [
    "CITY_INTEGRITY_SAMPLE_LIMIT",
    "CityIntegrityBlocker",
    "CityIntegrityError",
    "CityIntegrityReport",
    "ScopeMismatchSample",
    "assess_city_integrity",
    "unassignable_scope_mask",
    "validate_city_integrity",
]

#: Maximum number of entries in each diagnostic sample.
CITY_INTEGRITY_SAMPLE_LIMIT = 10


class CityIntegrityBlocker(StrEnum):
    """Why city integrity fails (values avoid source column names)."""

    CITY_SCOPE_UNASSIGNABLE = "job_scope_unassignable"
    PARENT_DETAIL_CITY_MISMATCH = "parent_detail_scope_mismatch"


@dataclass(frozen=True, slots=True)
class ScopeMismatchSample:
    """One linked detail row whose scope disagrees with its parent (confidential).

    ``None`` stands for a missing value. Scopes are the raw values, unmodified.
    """

    detail_key: tuple[object, ...]
    parent_key: tuple[object, ...]
    detail_scope: tuple[object, ...]
    parent_scope: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class CityIntegrityReport:
    """Immutable city-integrity result (counts, blockers and bounded samples).

    ``coverage`` is the expected-location contract the report was checked
    against (``None`` when assessed for a relationship alone); completeness
    accepts only a report for its own contract. Samples are excluded from
    ``repr`` because they hold identifiers and source values.
    """

    relationship: JobDetailRelationshipDefinition = field(repr=False)
    coverage: LocationCoverageDefinition | None = field(repr=False)
    job_count: int
    scope_unassignable_job_count: int
    detail_row_count: int
    linked_detail_row_count: int
    scope_mismatch_detail_row_count: int
    scope_mismatch_job_count: int
    scope_unassignable_job_sample: tuple[tuple[object, ...], ...] = field(default=(), repr=False)
    scope_mismatch_sample: tuple[ScopeMismatchSample, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.relationship, JobDetailRelationshipDefinition):
            raise TypeError("relationship must be a JobDetailRelationshipDefinition")
        if self.coverage is not None and not isinstance(self.coverage, LocationCoverageDefinition):
            raise TypeError("coverage must be a LocationCoverageDefinition or None")
        counts = (self.job_count, self.scope_unassignable_job_count, self.detail_row_count,
                  self.linked_detail_row_count, self.scope_mismatch_detail_row_count, self.scope_mismatch_job_count)
        if not all(type(c) is int and c >= 0 for c in counts):
            raise ValueError("counts must be non-negative integers")
        if (self.scope_unassignable_job_count > self.job_count or self.scope_mismatch_job_count > self.job_count
                or self.linked_detail_row_count > self.detail_row_count
                or self.scope_mismatch_detail_row_count > self.linked_detail_row_count):
            raise ValueError("inconsistent city-integrity counts")
        if (self.scope_mismatch_detail_row_count == 0) != (self.scope_mismatch_job_count == 0):
            raise ValueError("mismatched rows and mismatched jobs must both be zero or both positive")
        for sample, count in ((self.scope_unassignable_job_sample, self.scope_unassignable_job_count),
                              (self.scope_mismatch_sample, self.scope_mismatch_detail_row_count)):
            if not isinstance(sample, tuple) or len(sample) != min(count, CITY_INTEGRITY_SAMPLE_LIMIT):
                raise ValueError("a sample must hold min(count, limit) entries")

    @property
    def job_scope_assignable(self) -> bool:
        """Every job has an assignable scope (city)."""
        return self.scope_unassignable_job_count == 0

    @property
    def parent_detail_scope_agrees(self) -> bool:
        """Every linked detail row agrees with its parent's assignable scope."""
        return self.scope_mismatch_detail_row_count == 0

    @property
    def blocking_reasons(self) -> tuple[CityIntegrityBlocker, ...]:
        B = CityIntegrityBlocker
        reasons = []
        if not self.job_scope_assignable:
            reasons.append(B.CITY_SCOPE_UNASSIGNABLE)
        if not self.parent_detail_scope_agrees:
            reasons.append(B.PARENT_DETAIL_CITY_MISMATCH)
        return tuple(reasons)

    @property
    def is_valid(self) -> bool:
        return not self.blocking_reasons


class CityIntegrityError(Exception):
    """Strict validation failed; the message lists blocker categories only."""

    def __init__(self, report: CityIntegrityReport) -> None:
        super().__init__("City integrity failed: " + ", ".join(b.value for b in report.blocking_reasons) + ".")
        self.report = report
        self.blocking_reasons = report.blocking_reasons


# ------------------------------------------------------------------- public API


def unassignable_scope_mask(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Rows whose scope (any of ``columns``) is unassignable - the single definition.

    A value is assignable only when it is a non-empty string equal to its
    whitespace-stripped form. Missing values, non-strings, empty and
    whitespace-only strings and strings with leading or trailing whitespace
    are unassignable. Nothing is trimmed or modified.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    absent = tuple(c for c in columns if c not in frame.columns)
    if absent:
        raise RelationshipConfigurationError(f"The frame lacks {len(absent)} scope column(s).", absent)
    bad = np.zeros(len(frame), dtype=bool)
    for column in columns:
        values = frame[column].astype(object).tolist()
        bad |= np.fromiter((not _assignable(v) for v in values), dtype=bool, count=len(values))
    return bad


def assess_city_integrity(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    *,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
    coverage: LocationCoverageDefinition | None = EXPECTED_LOCATION_COVERAGE,
) -> CityIntegrityReport:
    """Assess assignable job scope and parent/detail scope agreement.

    Scope columns come from ``relationship.scope_agreement_columns``. When
    ``coverage`` is given and declares stream scope on the detail dataset, its
    ``parent_scope_columns`` / ``stream_scope_columns`` must be exactly those
    pairs (one definition of the scope), otherwise a configuration error is
    raised. Inputs are not modified; ordinary data defects never raise.

    Raises:
        TypeError: Invalid argument types.
        RelationshipConfigurationError: No scope pairs are configured, or a
            key or scope column is absent.
        LocationCoverageConfigurationError: The coverage scope disagrees with
            the relationship's scope pairs.
    """
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be pandas DataFrames")
    if not isinstance(relationship, JobDetailRelationshipDefinition):
        raise TypeError(f"expected a JobDetailRelationshipDefinition, got {type(relationship).__name__}")
    if coverage is not None:
        check_scope_contract(coverage, relationship)
    if not relationship.scope_agreement_columns:
        raise RelationshipConfigurationError("the relationship declares no scope_agreement_columns")
    parent_keys, detail_keys = relationship.parent_key_columns, relationship.detail_key_columns
    detail_ids = relationship.detail_definition.unique_key_columns or detail_keys
    for frame, columns, role in ((jobs, (*parent_keys, *relationship.scope_parent_columns), relationship.parent),
                                 (cars, (*detail_keys, *detail_ids, *relationship.scope_detail_columns),
                                  relationship.detail)):
        absent = tuple(dict.fromkeys(c for c in columns if c not in frame.columns))
        if absent:
            raise RelationshipConfigurationError(
                f"The '{role}' frame lacks {len(absent)} key or scope column(s).", absent)

    unassignable = unassignable_scope_mask(jobs, relationship.scope_parent_columns)
    agreement = scope_agreement(jobs, cars, relationship)
    unassignable_keys = sorted(
        (_key(row) for row in jobs.loc[unassignable, list(parent_keys)].itertuples(index=False, name=None)),
        key=_sort_key)
    samples = [
        ScopeMismatchSample(detail_key=_key(d), parent_key=_key(p), detail_scope=_key(ds), parent_scope=_key(ps))
        for d, p, ds, ps in agreement.mismatch_rows(jobs, cars, detail_ids, relationship)
    ]
    samples.sort(key=lambda s: (_sort_key(s.detail_key), _sort_key(s.parent_key),
                                _sort_key(s.detail_scope), _sort_key(s.parent_scope)))
    return CityIntegrityReport(
        relationship=relationship, coverage=coverage, job_count=len(jobs),
        scope_unassignable_job_count=int(unassignable.sum()), detail_row_count=len(cars),
        linked_detail_row_count=int(agreement.detail_linked.sum()),
        scope_mismatch_detail_row_count=int(agreement.detail_mismatch.sum()),
        scope_mismatch_job_count=int(agreement.parent_mismatch.sum()),
        scope_unassignable_job_sample=tuple(unassignable_keys[:CITY_INTEGRITY_SAMPLE_LIMIT]),
        scope_mismatch_sample=tuple(samples[:CITY_INTEGRITY_SAMPLE_LIMIT]),
    )


def validate_city_integrity(jobs: pd.DataFrame, cars: pd.DataFrame, **kwargs: object) -> CityIntegrityReport:
    """Assess; return the report if valid, else raise :class:`CityIntegrityError`."""
    report = assess_city_integrity(jobs, cars, **kwargs)  # type: ignore[arg-type]
    if not report.is_valid:
        raise CityIntegrityError(report)
    return report


# ------------------------------------------------------------ shared internals


def check_scope_contract(coverage: LocationCoverageDefinition, relationship: JobDetailRelationshipDefinition) -> None:
    """Fail unless the coverage scope and the relationship scope pairs are one definition.

    Applies only when ``coverage`` declares stream scope on the relationship's
    detail dataset (the configuration whose continuity relies on job scope).
    """
    if not isinstance(coverage, LocationCoverageDefinition):
        raise TypeError(f"expected a LocationCoverageDefinition, got {type(coverage).__name__}")
    if not coverage.stream_scope_columns or coverage.dataset != relationship.detail:
        return
    pairs = tuple(zip(coverage.parent_scope_columns, coverage.stream_scope_columns, strict=False))
    if (len(coverage.parent_scope_columns) != len(coverage.stream_scope_columns)
            or pairs != relationship.scope_agreement_columns):
        raise LocationCoverageConfigurationError(
            "the coverage scope columns must equal the relationship's scope_agreement_columns")


@dataclass(frozen=True, slots=True, eq=False)
class ScopeAgreement:
    """Row-aligned agreement flags (internal; shared by every consumer)."""

    detail_linked: np.ndarray      # per detail row: complete key matching >= 1 parent
    detail_mismatch: np.ndarray    # per detail row: linked and disagreeing with some parent
    parent_mismatch: np.ndarray    # per parent row: some linked detail disagrees
    mismatch_positions: tuple[tuple[int, int], ...]   # (detail row, parent row), sorted

    def mismatch_rows(self, jobs: pd.DataFrame, cars: pd.DataFrame, detail_ids: tuple[str, ...],
                      relationship: JobDetailRelationshipDefinition) -> list[tuple[tuple[object, ...], ...]]:
        """(detail id, parent key, detail scope, parent scope) value tuples of disagreeing pairs."""
        d = np.fromiter((x for x, _ in self.mismatch_positions), dtype=np.int64, count=len(self.mismatch_positions))
        p = np.fromiter((y for _, y in self.mismatch_positions), dtype=np.int64, count=len(self.mismatch_positions))

        def columns(frame: pd.DataFrame, rows: np.ndarray, names: tuple[str, ...]) -> list[tuple[object, ...]]:
            arrays = [frame[c].astype(object).to_numpy()[rows] for c in names]
            return list(zip(*arrays, strict=True)) if arrays else [()] * len(rows)

        return list(zip(columns(cars, d, detail_ids), columns(jobs, p, relationship.parent_key_columns),
                        columns(cars, d, relationship.scope_detail_columns),
                        columns(jobs, p, relationship.scope_parent_columns), strict=True))


def scope_agreement(jobs: pd.DataFrame, cars: pd.DataFrame,
                    relationship: JobDetailRelationshipDefinition) -> ScopeAgreement:
    """Compare every linked (detail, parent) pair on the relationship's scope pairs.

    Rows without a complete key, or whose key matches no parent, are not
    linked and are left to the relationship controls. With no scope pairs
    configured nothing can disagree.
    """
    parent_keys, detail_keys = list(relationship.parent_key_columns), list(relationship.detail_key_columns)
    parent_scope, detail_scope = list(relationship.scope_parent_columns), list(relationship.scope_detail_columns)
    names = [f"k{i}" for i in range(len(parent_keys))]
    left = pd.DataFrame({n: cars[c].astype(object).to_numpy() for n, c in zip(names, detail_keys, strict=True)})
    left["d"] = np.arange(len(cars))
    left = left.loc[left[names].notna().all(axis=1).to_numpy()]
    right = pd.DataFrame({n: jobs[c].astype(object).to_numpy() for n, c in zip(names, parent_keys, strict=True)})
    right["p"] = np.arange(len(jobs))
    right = right.loc[right[names].notna().all(axis=1).to_numpy()]
    pairs = left.merge(right, on=names, how="inner", sort=False)[["d", "p"]]
    detail_ok = ~unassignable_scope_mask(cars, tuple(detail_scope)) if detail_scope else np.ones(len(cars), bool)
    parent_ok = ~unassignable_scope_mask(jobs, tuple(parent_scope)) if parent_scope else np.ones(len(jobs), bool)
    d, p = pairs["d"].to_numpy(dtype=np.int64), pairs["p"].to_numpy(dtype=np.int64)
    agrees = detail_ok[d] & parent_ok[p]
    for dc, pc in zip(detail_scope, parent_scope, strict=True):
        dv = cars[dc].astype(object).to_numpy()[d]
        pv = jobs[pc].astype(object).to_numpy()[p]
        agrees &= np.fromiter((_assignable(a) and _assignable(b) and a == b for a, b in zip(dv, pv, strict=True)),
                              dtype=bool, count=len(d))
    bad = sorted(zip(d[~agrees].tolist(), p[~agrees].tolist(), strict=True))
    detail_linked = np.zeros(len(cars), dtype=bool)
    detail_linked[d] = True
    detail_mismatch = np.zeros(len(cars), dtype=bool)
    detail_mismatch[d[~agrees]] = True
    parent_mismatch = np.zeros(len(jobs), dtype=bool)
    parent_mismatch[p[~agrees]] = True
    return ScopeAgreement(detail_linked=detail_linked, detail_mismatch=detail_mismatch,
                          parent_mismatch=parent_mismatch, mismatch_positions=tuple(bad))


def _assignable(value: object) -> bool:
    return isinstance(value, str) and value != "" and value == value.strip()


def _key(values: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(None if (v is None or (not isinstance(v, str) and bool(pd.isna(v)))) else v for v in values)


def _sort_key(values: tuple[object, ...]) -> tuple[tuple[int, str], ...]:
    """Total, deterministic order: missing values last, others by text."""
    return tuple((1, "") if v is None else (0, str(v)) for v in values)
