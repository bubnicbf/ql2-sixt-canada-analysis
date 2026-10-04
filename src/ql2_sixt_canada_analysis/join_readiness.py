"""Fail-closed readiness for the trusted jobs-to-details analytical join.

A *relationship-valid* join (:func:`~ql2_sixt_canada_analysis.relationships.join_jobs_to_details`)
is not an analytically trusted one: it can be built while a dataset's
business key is duplicated or while declared job-level detail counts do not
match the detail rows. :func:`assess_job_detail_join_readiness` therefore
assesses, on the **same** frames it joins and in one call:

1. the jobs business-key contract (:func:`~ql2_sixt_canada_analysis.unique_keys.assess_unique_key`);
2. the details business-key contract;
3. declared-count reconciliation
   (:func:`~ql2_sixt_canada_analysis.reconciliation.assess_job_detail_reconciliation`);
4. the one-to-many relationship contract, including orphan and missing-link
   detail rows (:func:`~ql2_sixt_canada_analysis.relationships.assess_one_to_many_join`);
5. city integrity, when the relationship declares scope-agreement columns
   (:func:`~ql2_sixt_canada_analysis.city_integrity.assess_city_integrity`):
   every job's city must be assignable and every linked detail row must
   carry its parent job's city. A cross-city or unassignable row would put a
   detail under the wrong job's city in the joined frame, so either defect
   withholds the trusted join (``job_scope_unassignable`` /
   ``parent_detail_scope_mismatch``).

A report that cannot be produced (a precondition error, for example an
incomplete or duplicated parent key) is **unavailable** and blocks the join;
the absence of a violation is never read as a pass. ``join_ready`` is true
only when every report exists and explicitly passes and the cardinality-
validated merge succeeded.

Outputs
-------
* ``trusted_jobs_with_details`` - the analytical join; ``None`` unless
  ``join_ready``. Use only this frame for pricing or aggregation.
* ``diagnostic_jobs_with_details`` - the relationship-valid join built for
  investigation when the relationship contract alone passes but another
  contract fails. **Untrusted**: never use it for pricing, aggregation or
  conclusions. ``None`` when the trusted join exists or the relationship fails
  (orphans are never silently dropped into a frame).

Both frames are returned as fresh copies on every access, so callers cannot
alter the result held by the readiness object. Nothing is deduplicated,
repaired, filtered or written. Join readiness is one prerequisite only; it
does not establish pricing readiness
(:func:`~ql2_sixt_canada_analysis.readiness.assess_pricing_readiness`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from ql2_sixt_canada_analysis.city_integrity import CityIntegrityReport, assess_city_integrity
from ql2_sixt_canada_analysis.reconciliation import JobDetailReconciliationReport, assess_job_detail_reconciliation
from ql2_sixt_canada_analysis.relationships import (
    OneToManyJoinReport,
    RelationshipPreconditionError,
    ValidatedJoinError,
    assess_one_to_many_join,
    join_jobs_to_details,
)
from ql2_sixt_canada_analysis.schemas import JOB_DETAIL_RELATIONSHIP, JobDetailRelationshipDefinition
from ql2_sixt_canada_analysis.unique_keys import UniqueKeyReport, assess_unique_key

__all__ = [
    "JobDetailJoinBlocker",
    "JobDetailJoinReadiness",
    "UntrustedJoinError",
    "assess_job_detail_join_readiness",
    "require_trusted_job_detail_join",
]


class JobDetailJoinBlocker(StrEnum):
    """Why the trusted join is unavailable (values avoid source column names)."""

    REQUIRED_REPORT_UNAVAILABLE = "required_report_unavailable"
    JOBS_KEY_CONTRACT_FAILED = "jobs_key_contract_failed"
    DETAILS_KEY_CONTRACT_FAILED = "details_key_contract_failed"
    DECLARED_COUNTS_NOT_RECONCILED = "declared_counts_not_reconciled"
    RELATIONSHIP_CONTRACT_FAILED = "relationship_contract_failed"
    ORPHAN_DETAILS_PRESENT = "orphan_details_present"
    MISSING_LINK_DETAILS_PRESENT = "missing_link_details_present"
    JOIN_CONSTRUCTION_FAILED = "join_construction_failed"
    CITY_SCOPE_UNASSIGNABLE = "job_scope_unassignable"
    PARENT_DETAIL_CITY_MISMATCH = "parent_detail_scope_mismatch"


@dataclass(frozen=True, slots=True, eq=False)
class JobDetailJoinReadiness:
    """Evidence, decision and outputs of the trusted-join gate.

    Reports are ``None`` when they could not be produced (unavailable, which
    blocks). Frames are exposed only through copying properties.
    """

    jobs_key_report: UniqueKeyReport | None
    details_key_report: UniqueKeyReport | None
    reconciliation_report: JobDetailReconciliationReport | None
    relationship_report: OneToManyJoinReport | None
    blocking_reasons: tuple[JobDetailJoinBlocker, ...]
    _trusted: pd.DataFrame | None
    _diagnostic: pd.DataFrame | None
    #: City-integrity result (``None`` when the relationship declares no scope invariant).
    city_integrity_report: CityIntegrityReport | None = None

    def __post_init__(self) -> None:
        # Programmer invariants: a trusted frame exists iff nothing blocks.
        assert (self._trusted is not None) == (not self.blocking_reasons)
        assert self._trusted is None or self._diagnostic is None

    @property
    def jobs_key_contract_valid(self) -> bool:
        return self.jobs_key_report is not None and self.jobs_key_report.is_valid

    @property
    def details_key_contract_valid(self) -> bool:
        return self.details_key_report is not None and self.details_key_report.is_valid

    @property
    def all_key_contracts_valid(self) -> bool:
        return self.jobs_key_contract_valid and self.details_key_contract_valid

    @property
    def declared_counts_reconciled(self) -> bool:
        return self.reconciliation_report is not None and self.reconciliation_report.is_reconciled

    @property
    def relationship_contract_valid(self) -> bool:
        return self.relationship_report is not None and self.relationship_report.is_valid

    @property
    def city_integrity_valid(self) -> bool:
        """City integrity passed; ``None`` report means the relationship declares no scope invariant."""
        return self.city_integrity_report is None or self.city_integrity_report.is_valid

    @property
    def all_reports_available(self) -> bool:
        return JobDetailJoinBlocker.REQUIRED_REPORT_UNAVAILABLE not in self.blocking_reasons

    @property
    def join_ready(self) -> bool:
        """True only when every prerequisite explicitly passed and the join was built."""
        return not self.blocking_reasons and self._trusted is not None

    @property
    def trusted_jobs_with_details(self) -> pd.DataFrame | None:
        """The analytical join (a copy), or ``None`` unless ``join_ready``."""
        return self._trusted.copy() if self.join_ready and self._trusted is not None else None

    @property
    def diagnostic_jobs_with_details(self) -> pd.DataFrame | None:
        """UNTRUSTED relationship-valid join (a copy) for investigation only, or ``None``."""
        return self._diagnostic.copy() if self._diagnostic is not None else None


class UntrustedJoinError(Exception):
    """The trusted join is unavailable; ``blocking_reasons`` lists categories only."""

    def __init__(self, readiness: JobDetailJoinReadiness) -> None:
        super().__init__("Trusted jobs-to-details join unavailable: "
                         + ", ".join(b.value for b in readiness.blocking_reasons) + ".")
        self.readiness = readiness
        self.blocking_reasons = readiness.blocking_reasons


# ------------------------------------------------------------------- public API


def assess_job_detail_join_readiness(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> JobDetailJoinReadiness:
    """Assess every join prerequisite on ``jobs``/``cars`` and build the joins.

    All reports and the join are computed from the same frames in this call
    (no reuse of reports computed elsewhere). Inputs are not modified.

    Raises:
        TypeError: Invalid argument types.
        RelationshipConfigurationError / KeyConfigurationError: Invalid
            configuration (not a data condition).
    """
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be pandas DataFrames")
    if not isinstance(relationship, JobDetailRelationshipDefinition):
        raise TypeError(f"expected a JobDetailRelationshipDefinition, got {type(relationship).__name__}")
    B = JobDetailJoinBlocker
    jobs_keys = assess_unique_key(jobs, relationship.parent_definition)
    detail_keys = assess_unique_key(cars, relationship.detail_definition)
    reconciliation = _available(lambda: assess_job_detail_reconciliation(jobs, cars, relationship))
    relation = _available(lambda: assess_one_to_many_join(jobs, cars, relationship))

    # Declared on the project relationship; a configuration error here is raised, never a pass.
    city = (assess_city_integrity(jobs, cars, relationship=relationship, coverage=None)
            if relationship.scope_agreement_columns else None)

    reasons: list[JobDetailJoinBlocker] = []
    if reconciliation is None or relation is None:
        reasons.append(B.REQUIRED_REPORT_UNAVAILABLE)
    if not jobs_keys.is_valid:
        reasons.append(B.JOBS_KEY_CONTRACT_FAILED)
    if not detail_keys.is_valid:
        reasons.append(B.DETAILS_KEY_CONTRACT_FAILED)
    if reconciliation is not None and not reconciliation.is_reconciled and (
            not reconciliation.declared_counts_reconciled or reconciliation.parent_detail_scope_agrees):
        reasons.append(B.DECLARED_COUNTS_NOT_RECONCILED)   # counts fail (or any other non-pass)
    if relation is not None and not relation.is_valid:
        reasons.append(B.RELATIONSHIP_CONTRACT_FAILED)
        if relation.orphan_detail_row_count:
            reasons.append(B.ORPHAN_DETAILS_PRESENT)
        if relation.missing_link_detail_row_count:
            reasons.append(B.MISSING_LINK_DETAILS_PRESENT)

    if city is not None and not city.job_scope_assignable:
        reasons.append(B.CITY_SCOPE_UNASSIGNABLE)
    if city is not None and not city.parent_detail_scope_agrees:
        reasons.append(B.PARENT_DETAIL_CITY_MISMATCH)

    joined = None
    if relation is not None and relation.is_valid:
        try:
            joined = join_jobs_to_details(jobs, cars, relationship).joined
        except (ValidatedJoinError, RelationshipPreconditionError):
            reasons.append(B.JOIN_CONSTRUCTION_FAILED)

    trusted = joined if not reasons else None
    diagnostic = joined if reasons else None
    return JobDetailJoinReadiness(
        jobs_key_report=jobs_keys, details_key_report=detail_keys, reconciliation_report=reconciliation,
        relationship_report=relation, blocking_reasons=tuple(reasons), _trusted=trusted, _diagnostic=diagnostic,
        city_integrity_report=city,
    )


def require_trusted_job_detail_join(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    relationship: JobDetailRelationshipDefinition = JOB_DETAIL_RELATIONSHIP,
) -> pd.DataFrame:
    """Return the trusted join, or raise :class:`UntrustedJoinError` (never a diagnostic frame)."""
    readiness = assess_job_detail_join_readiness(jobs, cars, relationship)
    trusted = readiness.trusted_jobs_with_details
    if trusted is None:
        raise UntrustedJoinError(readiness)
    return trusted


def _available(assess):  # type: ignore[no-untyped-def]
    """Run an assessment; a structural precondition failure makes it unavailable (``None``)."""
    try:
        return assess()
    except RelationshipPreconditionError:
        return None
