"""Location identity policy gate and fail-closed pricing readiness.

Behavioural evidence vs authority
---------------------------------
:func:`~ql2_sixt_canada_analysis.comparison.compare_location_streams`
produces *evidence* (for example ``LIKELY_DUPLICATE_STREAMS``). Evidence never
decides whether two labels are one location. The decision comes only from a
configured, authority-backed
:class:`~ql2_sixt_canada_analysis.schemas.LocationIdentityPolicy`
(project default: :data:`~ql2_sixt_canada_analysis.schemas.VANCOUVER_LOCATION_POLICY`,
``UNRESOLVED``).

:func:`assess_location_policy` turns the policy into explicit permissions
(:class:`LocationPolicyReport`). ``locations_are_aliases`` being false is
**not** evidence that the labels are distinct; only
``locations_comparable_independently`` grants that.

:func:`assess_pricing_readiness` combines the policy with every existing
foundational gate. Pricing is ready only when *all* pass; each failure is
reported as a :class:`PricingBlocker`. The location gate never overrides
another gate.

:func:`apply_location_policy` builds analytical location keys: for a
confirmed alias the approved canonical key replaces both labels; source keys
are kept alongside for lineage. Frames are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from ql2_sixt_canada_analysis.comparison import LocationStreamComparisonReport, LocationStreamComparisonStatus
from ql2_sixt_canada_analysis.coverage import LocationCoverageReport
from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.reconciliation import JobDetailReconciliationReport
from ql2_sixt_canada_analysis.stability import VehicleStabilityReport
from ql2_sixt_canada_analysis.streams import LocationStreamInvestigationReport, StreamContinuity
from ql2_sixt_canada_analysis.schemas import (
    VANCOUVER_LOCATION_POLICY,
    LocationIdentityPolicy,
    LocationPolicyAuthority,
    LocationPolicyConfigurationError,
    LocationPolicyState,
)

__all__ = [
    "CompletenessBlocker",
    "CompletenessReport",
    "assess_completeness",
    "AnalyticalLocationKeys",
    "LocationPolicyReport",
    "PricingBlocker",
    "PricingNotReadyError",
    "PricingReadinessReport",
    "apply_location_policy",
    "assess_location_policy",
    "assess_pricing_readiness",
    "validate_pricing_readiness",
]


class PricingBlocker(StrEnum):
    """Why pricing analysis is not ready (values avoid source column names)."""

    SOURCE_NOT_COMPLETE = "source_not_complete"
    KEY_CONTRACTS_INVALID = "key_contracts_invalid"
    EXPECTED_COVERAGE_FAILED = "expected_coverage_failed"
    EXPECTED_STREAM_UNHEALTHY = "expected_stream_unhealthy"
    JOB_DETAIL_COUNTS_UNRECONCILED = "job_detail_counts_unreconciled"
    ONE_TO_MANY_INVALID = "one_to_many_invalid"
    TEMPORAL_FIELDS_UNTRUSTED = "temporal_fields_untrusted"
    VEHICLE_ATTRIBUTES_UNSTABLE = "vehicle_attributes_unstable"
    VEHICLE_HISTORY_INSUFFICIENT = "vehicle_history_insufficient"
    VEHICLE_STABILITY_UNAVAILABLE = "vehicle_stability_unavailable"
    LOCATION_POLICY_UNRESOLVED = "vancouver_policy_unresolved"
    ALIAS_CANONICALIZATION_NOT_APPLIED = "alias_canonicalization_not_applied"
    IDENTITY_EVIDENCE_CONFLICT = "identity_evidence_conflicts_with_policy"


_CONFLICTING_EVIDENCE = {
    LocationPolicyState.CONFIRMED_ALIAS: frozenset({LocationStreamComparisonStatus.CONFIRMED_DISTINCT_LOCATIONS}),
    LocationPolicyState.CONFIRMED_DISTINCT: frozenset({
        LocationStreamComparisonStatus.CONFIRMED_ALIAS,
        LocationStreamComparisonStatus.DUPLICATED_COLLECTION_CONFIGURATION,
    }),
}


@dataclass(frozen=True, slots=True, eq=False)
class AnalyticalLocationKeys:
    """Source and analytical location keys for one frame (copies on access).

    ``alias_mapping_applied`` is true only when the policy is a confirmed
    alias and its approved mapping produced ``analytical_keys``.
    """

    policy: LocationIdentityPolicy
    alias_mapping_applied: bool
    _source: pd.Series
    _analytical: pd.Series

    @property
    def source_keys(self) -> pd.Series:
        """Original location keys (lineage / audit); a copy."""
        return self._source.copy()

    @property
    def analytical_keys(self) -> pd.Series:
        """Keys to group analysis by (canonical for a confirmed alias); a copy."""
        return self._analytical.copy()


@dataclass(frozen=True, slots=True)
class LocationPolicyReport:
    """Explicit identity-policy state and derived permissions (no location values)."""

    state: LocationPolicyState
    authority: LocationPolicyAuthority | None
    behavioral_evidence: LocationStreamComparisonStatus | None
    identity_evidence_conflict: bool
    canonicalization_applied: bool

    @property
    def location_policy_resolved(self) -> bool:
        """True only for an authority-backed CONFIRMED_ALIAS or CONFIRMED_DISTINCT."""
        return self.state is not LocationPolicyState.UNRESOLVED and self.authority is not None

    @property
    def location_policy_authority_sufficient(self) -> bool:
        """Resolved, authority-backed and not contradicted by identity metadata."""
        return self.location_policy_resolved and not self.identity_evidence_conflict

    @property
    def locations_are_aliases(self) -> bool:
        """True only for an authority-backed CONFIRMED_ALIAS. False does not mean distinct."""
        return self.location_policy_authority_sufficient and self.state is LocationPolicyState.CONFIRMED_ALIAS

    @property
    def locations_comparable_independently(self) -> bool:
        """True only for an authority-backed CONFIRMED_DISTINCT."""
        return self.location_policy_authority_sufficient and self.state is LocationPolicyState.CONFIRMED_DISTINCT

    @property
    def canonicalization_required(self) -> bool:
        return self.state is LocationPolicyState.CONFIRMED_ALIAS

    @property
    def blocking_reasons(self) -> tuple[PricingBlocker, ...]:
        reasons = []
        if not self.location_policy_resolved:
            reasons.append(PricingBlocker.LOCATION_POLICY_UNRESOLVED)
        if self.identity_evidence_conflict:
            reasons.append(PricingBlocker.IDENTITY_EVIDENCE_CONFLICT)
        if self.canonicalization_required and not self.canonicalization_applied:
            reasons.append(PricingBlocker.ALIAS_CANONICALIZATION_NOT_APPLIED)
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class PricingReadinessReport:
    """Fail-closed pricing readiness: ``ready`` only with no blocking reasons."""

    blocking_reasons: tuple[PricingBlocker, ...]
    location_policy: LocationPolicyReport

    @property
    def ready(self) -> bool:
        return not self.blocking_reasons


class PricingNotReadyError(Exception):
    """Pricing analysis is blocked; ``blocking_reasons`` lists categories only."""

    def __init__(self, report: PricingReadinessReport) -> None:
        super().__init__("Pricing analysis is not ready: "
                         + ", ".join(r.value for r in report.blocking_reasons) + ".")
        self.report = report
        self.blocking_reasons = report.blocking_reasons


# ------------------------------------------------------------------- public API


def apply_location_policy(
    frame: pd.DataFrame, policy: LocationIdentityPolicy = VANCOUVER_LOCATION_POLICY,
) -> AnalyticalLocationKeys:
    """Analytical location keys for ``frame`` under ``policy`` (frame untouched).

    Only a confirmed alias changes keys, and only through its approved
    mapping; unresolved and distinct policies leave keys as observed.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(policy, LocationIdentityPolicy):
        raise TypeError("policy must be a LocationIdentityPolicy")
    columns = list(policy.coverage.location_columns)
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise LocationPolicyConfigurationError(f"The frame lacks {len(absent)} location column(s).")
    values = frame.loc[:, columns].astype(object)
    values = values.where(values.notna(), None)
    source = pd.Series(list(map(tuple, values.itertuples(index=False))), index=frame.index, dtype=object)
    mapping = policy.alias_mapping
    analytical = source.map(lambda key: mapping.get(key, key)) if mapping else source.copy()
    return AnalyticalLocationKeys(policy=policy, alias_mapping_applied=bool(mapping),
                                  _source=source, _analytical=analytical.astype(object))


def assess_location_policy(
    policy: LocationIdentityPolicy = VANCOUVER_LOCATION_POLICY,
    comparison: LocationStreamComparisonReport | None = None,
    analytical_keys: AnalyticalLocationKeys | None = None,
) -> LocationPolicyReport:
    """Explicit policy permissions; ``comparison`` is recorded as evidence only.

    Behavioural statuses never change the state. Authoritative identity
    metadata in the comparison that contradicts a resolved policy blocks it.
    """
    if not isinstance(policy, LocationIdentityPolicy):
        raise TypeError("policy must be a LocationIdentityPolicy")
    if comparison is not None and not isinstance(comparison, LocationStreamComparisonReport):
        raise TypeError("comparison must be a LocationStreamComparisonReport or None")
    if analytical_keys is not None and not isinstance(analytical_keys, AnalyticalLocationKeys):
        raise TypeError("analytical_keys must be AnalyticalLocationKeys or None")
    evidence = comparison.status if comparison is not None else None
    conflict = evidence in _CONFLICTING_EVIDENCE.get(policy.state, frozenset())
    applied = (analytical_keys is not None and analytical_keys.policy == policy
               and analytical_keys.alias_mapping_applied)
    return LocationPolicyReport(state=policy.state, authority=policy.authority, behavioral_evidence=evidence,
                                identity_evidence_conflict=conflict, canonicalization_applied=applied)


def assess_pricing_readiness(
    *,
    location_policy: LocationPolicyReport,
    source_complete: bool,
    key_contracts_valid: bool,
    expected_coverage_passed: bool,
    expected_stream_healthy: bool,
    job_detail_counts_reconciled: bool,
    one_to_many_contract_valid: bool,
    temporal_fields_trusted: bool,
    vehicle_stability: VehicleStabilityReport | None,
) -> PricingReadinessReport:
    """Combine every foundational gate with the location policy (all must pass).

    Gate values must be real booleans; anything else is a ``TypeError`` so a
    missing result can never be read as a pass. ``vehicle_stability`` is the
    full-population stability report (``None`` = unavailable, which blocks);
    proven violations and insufficient product history are separate blockers
    and both are reported when both apply.
    """
    if not isinstance(location_policy, LocationPolicyReport):
        raise TypeError("location_policy must be a LocationPolicyReport")
    if vehicle_stability is not None and not isinstance(vehicle_stability, VehicleStabilityReport):
        raise TypeError("vehicle_stability must be a VehicleStabilityReport or None")
    gates = (
        (source_complete, PricingBlocker.SOURCE_NOT_COMPLETE),
        (key_contracts_valid, PricingBlocker.KEY_CONTRACTS_INVALID),
        (expected_coverage_passed, PricingBlocker.EXPECTED_COVERAGE_FAILED),
        (expected_stream_healthy, PricingBlocker.EXPECTED_STREAM_UNHEALTHY),
        (job_detail_counts_reconciled, PricingBlocker.JOB_DETAIL_COUNTS_UNRECONCILED),
        (one_to_many_contract_valid, PricingBlocker.ONE_TO_MANY_INVALID),
        (temporal_fields_trusted, PricingBlocker.TEMPORAL_FIELDS_UNTRUSTED),
    )
    if not all(isinstance(value, bool) for value, _ in gates):
        raise TypeError("every readiness gate must be a bool")
    reasons = [blocker for value, blocker in gates if not value]
    reasons.extend(_stability_blockers(vehicle_stability))
    reasons.extend(location_policy.blocking_reasons)
    return PricingReadinessReport(blocking_reasons=tuple(reasons), location_policy=location_policy)


def _stability_blockers(report: VehicleStabilityReport | None) -> list[PricingBlocker]:
    """Stability blockers; only a full-population PASSED report adds none."""
    if report is None:
        return [PricingBlocker.VEHICLE_STABILITY_UNAVAILABLE]
    blockers = []
    if report.violations:
        blockers.append(PricingBlocker.VEHICLE_ATTRIBUTES_UNSTABLE)
    if report.insufficient_history_entities or (report.distinct_entities == 0):
        blockers.append(PricingBlocker.VEHICLE_HISTORY_INSUFFICIENT)
    if not report.is_valid and not blockers:          # fail closed on any unforeseen non-pass
        blockers.append(PricingBlocker.VEHICLE_ATTRIBUTES_UNSTABLE)
    return blockers


class CompletenessBlocker(StrEnum):
    """Why data completeness is not proven (values avoid source column names)."""

    SOURCE_NOT_COMPLETE = "source_not_complete"
    COVERAGE_UNAVAILABLE = "coverage_unavailable"
    EXPECTED_PAIRS_MISSING = "expected_pairs_missing"
    UNEXPECTED_PAIRS = "unexpected_pairs"
    UNASSIGNED_LOCATIONS = "unassigned_pairs"
    CONFLICTING_LOCATION_ASSIGNMENT = "conflicting_pair_assignment"
    STREAM_UNAVAILABLE = "stream_unavailable"
    STREAM_CONTINUITY_PARTIAL = "stream_continuity_partial"
    STREAM_CONTINUITY_UNASSESSABLE = "stream_continuity_unassessable"
    STREAM_UNHEALTHY = "stream_unhealthy"
    RECONCILIATION_UNAVAILABLE = "reconciliation_unavailable"
    DECLARED_COUNT_UNRECONCILED = "declared_count_unreconciled"
    DECLARED_COUNTS_DISAGREE = "declared_counts_disagree"
    DETAIL_ROWS_UNLINKED = "detail_rows_unlinked"


_COVERAGE_BLOCKERS = {
    "missing_expected_location": CompletenessBlocker.EXPECTED_PAIRS_MISSING,
    "unexpected_location": CompletenessBlocker.UNEXPECTED_PAIRS,
    "missing_location_assignment": CompletenessBlocker.UNASSIGNED_LOCATIONS,
    "conflicting_location_assignment": CompletenessBlocker.CONFLICTING_LOCATION_ASSIGNMENT,
}


@dataclass(frozen=True, slots=True)
class CompletenessReport:
    """Fail-closed completeness: ``complete`` only with no blocking reasons.

    Completeness is one prerequisite only; it says nothing about timestamp
    authority, location identity, keys or stability.
    """

    blocking_reasons: tuple[CompletenessBlocker, ...]

    @property
    def complete(self) -> bool:
        return not self.blocking_reasons


def assess_completeness(
    *,
    datasets: RawDatasets,
    coverage: LocationCoverageReport | None,
    streams: tuple[LocationStreamInvestigationReport | None, ...],
    reconciliation: JobDetailReconciliationReport | None,
) -> CompletenessReport:
    """Combine source completeness, city-location coverage, stream continuity and counts.

    Every input must be present and pass; a missing report (``None``) or an
    empty ``streams`` tuple blocks. Each failure is reported.
    """
    B = CompletenessBlocker
    if not isinstance(datasets, RawDatasets):
        raise TypeError("datasets must be RawDatasets")
    if coverage is not None and not isinstance(coverage, LocationCoverageReport):
        raise TypeError("coverage must be a LocationCoverageReport or None")
    if reconciliation is not None and not isinstance(reconciliation, JobDetailReconciliationReport):
        raise TypeError("reconciliation must be a JobDetailReconciliationReport or None")
    if not isinstance(streams, tuple) or not all(
            s is None or isinstance(s, LocationStreamInvestigationReport) for s in streams):
        raise TypeError("streams must be a tuple of LocationStreamInvestigationReport or None")
    reasons: list[CompletenessBlocker] = []
    if datasets.complete_source is not True:
        reasons.append(B.SOURCE_NOT_COMPLETE)
    if coverage is None:
        reasons.append(B.COVERAGE_UNAVAILABLE)
    else:
        reasons.extend(_COVERAGE_BLOCKERS[v] for v in coverage.violations)
        if not coverage.is_valid and not coverage.violations:
            reasons.append(B.COVERAGE_UNAVAILABLE)
    if not streams or any(s is None for s in streams):
        reasons.append(B.STREAM_UNAVAILABLE)
    for stream in (s for s in streams if s is not None):
        if stream.stream_continuity is StreamContinuity.PARTIAL:
            reasons.append(B.STREAM_CONTINUITY_PARTIAL)
        elif stream.stream_continuity is not StreamContinuity.COMPLETE:
            reasons.append(B.STREAM_CONTINUITY_UNASSESSABLE)
        if not stream.is_healthy:
            reasons.append(B.STREAM_UNHEALTHY)
    if reconciliation is None:
        reasons.append(B.RECONCILIATION_UNAVAILABLE)
    else:
        if not all(f.reconciled for f in reconciliation.count_fields) or not reconciliation.count_fields:
            reasons.append(B.DECLARED_COUNT_UNRECONCILED)
        if not reconciliation.declared_counts_agree:
            reasons.append(B.DECLARED_COUNTS_DISAGREE)
        if not reconciliation.all_details_linked:
            reasons.append(B.DETAIL_ROWS_UNLINKED)
        if not reconciliation.is_reconciled and not any(
                r in reasons for r in (B.DECLARED_COUNT_UNRECONCILED, B.DECLARED_COUNTS_DISAGREE,
                                       B.DETAIL_ROWS_UNLINKED)):
            reasons.append(B.DECLARED_COUNT_UNRECONCILED)     # fail closed on any other non-pass
    return CompletenessReport(blocking_reasons=tuple(dict.fromkeys(reasons)))


def validate_pricing_readiness(**gates: object) -> PricingReadinessReport:
    """Return the readiness report if ready; else raise :class:`PricingNotReadyError`."""
    report = assess_pricing_readiness(**gates)  # type: ignore[arg-type]
    if not report.ready:
        raise PricingNotReadyError(report)
    return report
