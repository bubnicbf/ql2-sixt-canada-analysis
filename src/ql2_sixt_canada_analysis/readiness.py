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

Authoritative identity evidence can *contradict* a resolved policy (it never
selects one). ``LOCATION_MAPPING_DEFECT`` - conflicting authoritative
identity values within one stream - contradicts both ``CONFIRMED_ALIAS`` and
``CONFIRMED_DISTINCT``: the configured decision stays recorded
(``location_policy_resolved``) but is not ``location_policy_authority_sufficient``,
neither permission is granted and pricing is blocked until the source mapping
is corrected or authoritatively reconciled. Behavioural similarity cannot
override it.

:func:`assess_pricing_readiness` combines the policy with every existing
foundational gate. Pricing is ready only when *all* pass; each failure is
reported as a :class:`PricingBlocker`. The location gate never overrides
another gate.

:func:`apply_location_policy` builds analytical location keys: for a
confirmed alias the approved canonical key replaces both labels; source keys
are kept alongside for lineage. Frames are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import pandas as pd

from ql2_sixt_canada_analysis.city_integrity import CityIntegrityBlocker, CityIntegrityReport
from ql2_sixt_canada_analysis.comparison import LocationStreamComparisonReport, LocationStreamComparisonStatus
from ql2_sixt_canada_analysis.coverage import LocationCoverageReport
from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.reconciliation import JobDetailReconciliationReport
from ql2_sixt_canada_analysis.stability import VehicleStabilityReport
from ql2_sixt_canada_analysis.streams import ExpectedLocationStreamsReport
from ql2_sixt_canada_analysis.schemas import (
    EXPECTED_LOCATION_COVERAGE,
    VANCOUVER_LOCATION_POLICY,
    LocationCoverageConfigurationError,
    LocationCoverageDefinition,
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

    COMPLETENESS_UNAVAILABLE = "completeness_unavailable"
    DATA_INCOMPLETE = "data_incomplete"
    EXPECTED_STREAMS_NOT_PROVEN = "expected_streams_not_proven"
    SCOPE_INTEGRITY_NOT_PROVEN = "scope_integrity_not_proven"
    KEY_CONTRACTS_INVALID = "key_contracts_invalid"
    ONE_TO_MANY_INVALID = "one_to_many_invalid"
    TEMPORAL_FIELDS_UNTRUSTED = "temporal_fields_untrusted"
    VEHICLE_ATTRIBUTES_UNSTABLE = "vehicle_attributes_unstable"
    VEHICLE_HISTORY_INSUFFICIENT = "vehicle_history_insufficient"
    VEHICLE_STABILITY_UNAVAILABLE = "vehicle_stability_unavailable"
    LOCATION_POLICY_UNRESOLVED = "vancouver_policy_unresolved"
    ALIAS_CANONICALIZATION_NOT_APPLIED = "alias_canonicalization_not_applied"
    IDENTITY_EVIDENCE_CONFLICT = "identity_evidence_conflicts_with_policy"


#: Authoritative identity evidence that contradicts *every* resolved policy.
#: ``LOCATION_MAPPING_DEFECT`` means the authoritative identity columns hold
#: several different physical-location identities *within* one stream. That is
#: evidence for neither aliasing nor distinctness: whichever decision was
#: configured, the source mapping it would be applied to is itself broken, so
#: neither alias grouping nor independent comparison is safe until the mapping
#: is corrected or authoritatively reconciled. Behavioural statuses (likely
#: duplicate / likely distinct / inconclusive / insufficient captures / absent
#: streams) are deliberately absent: they are never authority.
_RESOLVED_POLICY_CONFLICTS: frozenset[LocationStreamComparisonStatus] = frozenset({
    LocationStreamComparisonStatus.LOCATION_MAPPING_DEFECT,
})

#: Authoritative identity evidence that contradicts one specific resolved state
#: (in addition to :data:`_RESOLVED_POLICY_CONFLICTS`). Read-only.
_STATE_SPECIFIC_CONFLICTS: Mapping[LocationPolicyState, frozenset[LocationStreamComparisonStatus]] = MappingProxyType({
    LocationPolicyState.CONFIRMED_ALIAS: frozenset({LocationStreamComparisonStatus.CONFIRMED_DISTINCT_LOCATIONS}),
    LocationPolicyState.CONFIRMED_DISTINCT: frozenset({
        LocationStreamComparisonStatus.CONFIRMED_ALIAS,
        LocationStreamComparisonStatus.DUPLICATED_COLLECTION_CONFIGURATION,
    }),
})


def _identity_evidence_conflicts(state: LocationPolicyState,
                                 evidence: LocationStreamComparisonStatus | None) -> bool:
    """Whether authoritative identity evidence contradicts the configured decision.

    Only a resolved state is a decision that evidence can contradict; an
    ``UNRESOLVED`` policy has nothing to contradict (it already blocks pricing
    as unresolved), so this is ``False`` there even for a mapping defect -
    which stays visible as the report's recorded evidence.
    """
    if state is LocationPolicyState.UNRESOLVED or evidence is None:
        return False
    return evidence in _RESOLVED_POLICY_CONFLICTS or evidence in _STATE_SPECIFIC_CONFLICTS.get(state, frozenset())


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
    """Explicit identity-policy state and derived permissions (no location values).

    ``behavioral_evidence`` is the comparison status recorded as evidence
    (behavioural or authoritative-identity); it never changes ``state``.
    ``identity_evidence_conflict`` must equal the centralised conflict rule
    for ``state`` and that evidence - a report claiming otherwise is rejected,
    so a mapping defect cannot be hidden by constructing or replacing a report.
    """

    state: LocationPolicyState
    authority: LocationPolicyAuthority | None
    behavioral_evidence: LocationStreamComparisonStatus | None
    identity_evidence_conflict: bool
    canonicalization_applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.state, LocationPolicyState):
            raise TypeError("state must be a LocationPolicyState")
        if self.behavioral_evidence is not None and not isinstance(
                self.behavioral_evidence, LocationStreamComparisonStatus):
            raise TypeError("behavioral_evidence must be a LocationStreamComparisonStatus or None")
        if not isinstance(self.identity_evidence_conflict, bool) or not isinstance(self.canonicalization_applied, bool):
            raise TypeError("identity_evidence_conflict and canonicalization_applied must be bools")
        if self.identity_evidence_conflict != _identity_evidence_conflicts(self.state, self.behavioral_evidence):
            raise ValueError("identity_evidence_conflict disagrees with the recorded identity evidence")

    @property
    def location_policy_resolved(self) -> bool:
        """An authority-backed CONFIRMED_ALIAS or CONFIRMED_DISTINCT decision is configured.

        This records only that a decision exists; whether it may be *used* is
        :attr:`location_policy_authority_sufficient`.
        """
        return self.state is not LocationPolicyState.UNRESOLVED and self.authority is not None

    @property
    def mapping_defect_indicated(self) -> bool:
        """The recorded evidence is ``LOCATION_MAPPING_DEFECT`` (any policy state)."""
        return self.behavioral_evidence is LocationStreamComparisonStatus.LOCATION_MAPPING_DEFECT

    @property
    def location_policy_authority_sufficient(self) -> bool:
        """Resolved and not contradicted by authoritative identity evidence.

        A ``LOCATION_MAPPING_DEFECT`` contradicts every resolved state, so a
        configured alias or distinct decision stays recorded (resolved) but is
        not sufficient for analysis while the defect exists.
        """
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
    #: The completeness decision this readiness rests on (kept for audit).
    completeness: CompletenessReport | None = None

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

    The configured state is never changed by evidence. Behavioural statuses
    neither resolve nor contradict a policy. Authoritative identity metadata
    that contradicts a resolved policy sets ``identity_evidence_conflict``,
    which makes the policy authority-insufficient, withdraws both permissions
    and blocks pricing (``IDENTITY_EVIDENCE_CONFLICT``):

    * ``LOCATION_MAPPING_DEFECT`` (conflicting identities within a stream)
      contradicts **both** ``CONFIRMED_ALIAS`` and ``CONFIRMED_DISTINCT``;
    * ``CONFIRMED_DISTINCT_LOCATIONS`` contradicts ``CONFIRMED_ALIAS``;
    * ``CONFIRMED_ALIAS`` and ``DUPLICATED_COLLECTION_CONFIGURATION``
      contradict ``CONFIRMED_DISTINCT``.

    An ``UNRESOLVED`` policy reports no conflict (there is no decision to
    contradict) and is blocked as unresolved; the evidence stays recorded.
    Applied canonicalisation is recorded but never outweighs a conflict.
    """
    if not isinstance(policy, LocationIdentityPolicy):
        raise TypeError("policy must be a LocationIdentityPolicy")
    if comparison is not None and not isinstance(comparison, LocationStreamComparisonReport):
        raise TypeError("comparison must be a LocationStreamComparisonReport or None")
    if analytical_keys is not None and not isinstance(analytical_keys, AnalyticalLocationKeys):
        raise TypeError("analytical_keys must be AnalyticalLocationKeys or None")
    evidence = comparison.status if comparison is not None else None
    conflict = _identity_evidence_conflicts(policy.state, evidence)
    applied = (analytical_keys is not None and analytical_keys.policy == policy
               and analytical_keys.alias_mapping_applied)
    return LocationPolicyReport(state=policy.state, authority=policy.authority, behavioral_evidence=evidence,
                                identity_evidence_conflict=conflict, canonicalization_applied=applied)


def assess_pricing_readiness(
    *,
    location_policy: LocationPolicyReport,
    completeness: CompletenessReport | None,
    key_contracts_valid: bool,
    one_to_many_contract_valid: bool,
    temporal_fields_trusted: bool,
    vehicle_stability: VehicleStabilityReport | None,
) -> PricingReadinessReport:
    """Combine every foundational gate with the location policy (all must pass).

    Gate values must be real booleans; anything else is a ``TypeError`` so a
    missing result can never be read as a pass. ``completeness`` is the
    authoritative :class:`CompletenessReport` (complete source, coverage,
    *every* configured expected stream, declared counts); there are no
    separate booleans that could override it, and ``None`` blocks.
    ``vehicle_stability`` is the
    full-population stability report (``None`` = unavailable, which blocks);
    proven violations and insufficient product history are separate blockers
    and both are reported when both apply.
    """
    if not isinstance(location_policy, LocationPolicyReport):
        raise TypeError("location_policy must be a LocationPolicyReport")
    if vehicle_stability is not None and not isinstance(vehicle_stability, VehicleStabilityReport):
        raise TypeError("vehicle_stability must be a VehicleStabilityReport or None")
    if completeness is not None and not isinstance(completeness, CompletenessReport):
        raise TypeError("completeness must be a CompletenessReport or None")
    gates = (
        (key_contracts_valid, PricingBlocker.KEY_CONTRACTS_INVALID),
        (one_to_many_contract_valid, PricingBlocker.ONE_TO_MANY_INVALID),
        (temporal_fields_trusted, PricingBlocker.TEMPORAL_FIELDS_UNTRUSTED),
    )
    if not all(isinstance(value, bool) for value, _ in gates):
        raise TypeError("every readiness gate must be a bool")
    reasons = list(_completeness_blockers(completeness))
    reasons.extend(blocker for value, blocker in gates if not value)
    reasons.extend(_stability_blockers(vehicle_stability))
    reasons.extend(location_policy.blocking_reasons)
    return PricingReadinessReport(blocking_reasons=tuple(reasons), location_policy=location_policy,
                                  completeness=completeness)


def _completeness_blockers(report: CompletenessReport | None) -> list[PricingBlocker]:
    if report is None:
        return [PricingBlocker.COMPLETENESS_UNAVAILABLE]
    if report.complete:
        return []
    blockers = [PricingBlocker.DATA_INCOMPLETE]
    if any(b in _STREAM_COMPLETENESS_BLOCKERS for b in report.blocking_reasons):
        blockers.append(PricingBlocker.EXPECTED_STREAMS_NOT_PROVEN)
    if any(b in _SCOPE_COMPLETENESS_BLOCKERS for b in report.blocking_reasons):
        blockers.append(PricingBlocker.SCOPE_INTEGRITY_NOT_PROVEN)
    return blockers


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
    EXPECTED_STREAM_ASSESSMENT_UNAVAILABLE = "expected_stream_assessment_unavailable"
    STREAM_CONTRACT_MISMATCH = "stream_contract_mismatch"
    EXPECTED_STREAM_REPORT_MISSING = "expected_stream_report_missing"
    DUPLICATE_STREAM_REPORT = "duplicate_stream_report"
    UNEXPECTED_STREAM_REPORT = "unexpected_stream_report"
    EXPECTED_STREAM_REPORT_UNAVAILABLE = "expected_stream_report_unavailable"
    STREAM_CONTINUITY_PARTIAL = "stream_continuity_partial"
    STREAM_CONTINUITY_UNASSESSABLE = "stream_continuity_unassessable"
    STREAM_SCOPE_UNASSIGNABLE = "stream_scope_unassignable"
    STREAM_SCOPE_MISMATCH = "stream_parent_detail_scope_mismatch"
    STREAM_UNHEALTHY = "expected_stream_unhealthy"
    CITY_INTEGRITY_UNAVAILABLE = "scope_integrity_unavailable"
    CITY_INTEGRITY_CONTRACT_MISMATCH = "scope_integrity_contract_mismatch"
    CITY_SCOPE_UNASSIGNABLE = "job_scope_unassignable"
    PARENT_DETAIL_CITY_MISMATCH = "parent_detail_scope_mismatch"
    COVERAGE_CONTRACT_MISMATCH = "coverage_contract_mismatch"
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


_STREAM_COMPLETENESS_BLOCKERS = frozenset({
    CompletenessBlocker.EXPECTED_STREAM_ASSESSMENT_UNAVAILABLE, CompletenessBlocker.STREAM_CONTRACT_MISMATCH,
    CompletenessBlocker.EXPECTED_STREAM_REPORT_MISSING, CompletenessBlocker.DUPLICATE_STREAM_REPORT,
    CompletenessBlocker.UNEXPECTED_STREAM_REPORT, CompletenessBlocker.EXPECTED_STREAM_REPORT_UNAVAILABLE,
    CompletenessBlocker.STREAM_CONTINUITY_PARTIAL, CompletenessBlocker.STREAM_CONTINUITY_UNASSESSABLE,
    CompletenessBlocker.STREAM_SCOPE_UNASSIGNABLE, CompletenessBlocker.STREAM_SCOPE_MISMATCH,
    CompletenessBlocker.STREAM_UNHEALTHY,
})

#: City (scope) integrity blockers; any of them adds ``SCOPE_INTEGRITY_NOT_PROVEN`` to pricing.
_SCOPE_COMPLETENESS_BLOCKERS = frozenset({
    CompletenessBlocker.CITY_INTEGRITY_UNAVAILABLE, CompletenessBlocker.CITY_INTEGRITY_CONTRACT_MISMATCH,
    CompletenessBlocker.CITY_SCOPE_UNASSIGNABLE, CompletenessBlocker.PARENT_DETAIL_CITY_MISMATCH,
    CompletenessBlocker.STREAM_SCOPE_UNASSIGNABLE, CompletenessBlocker.STREAM_SCOPE_MISMATCH,
})

_CITY_BLOCKERS = {
    CityIntegrityBlocker.CITY_SCOPE_UNASSIGNABLE: CompletenessBlocker.CITY_SCOPE_UNASSIGNABLE,
    CityIntegrityBlocker.PARENT_DETAIL_CITY_MISMATCH: CompletenessBlocker.PARENT_DETAIL_CITY_MISMATCH,
}


@dataclass(frozen=True, slots=True)
class CompletenessReport:
    """Fail-closed completeness: ``complete`` only with no blocking reasons.

    Completeness is one prerequisite only; it says nothing about timestamp
    authority, location identity, keys or stability. ``expected_streams`` and
    ``city_integrity`` keep the all-expected-stream aggregate and the city
    integrity result it was decided on (for diagnostics); a complete report
    cannot exist without both being valid.
    """

    blocking_reasons: tuple[CompletenessBlocker, ...]
    expected_streams: ExpectedLocationStreamsReport | None = None
    city_integrity: CityIntegrityReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.blocking_reasons, tuple) or not all(
                isinstance(b, CompletenessBlocker) for b in self.blocking_reasons):
            raise TypeError("blocking_reasons must be a tuple of CompletenessBlocker")
        if self.expected_streams is not None and not isinstance(self.expected_streams, ExpectedLocationStreamsReport):
            raise TypeError("expected_streams must be an ExpectedLocationStreamsReport or None")
        if self.city_integrity is not None and not isinstance(self.city_integrity, CityIntegrityReport):
            raise TypeError("city_integrity must be a CityIntegrityReport or None")
        # A complete report must rest on a valid all-expected-stream aggregate and valid city integrity.
        if not self.blocking_reasons and (self.expected_streams is None or not self.expected_streams.is_valid):
            raise ValueError("a complete report requires a valid all-expected-stream assessment")
        if not self.blocking_reasons and (self.city_integrity is None or not self.city_integrity.is_valid):
            raise ValueError("a complete report requires valid city integrity")

    @property
    def complete(self) -> bool:
        return not self.blocking_reasons


def assess_completeness(
    *,
    datasets: RawDatasets,
    coverage: LocationCoverageReport | None,
    streams: ExpectedLocationStreamsReport | None,
    reconciliation: JobDetailReconciliationReport | None,
    city_integrity: CityIntegrityReport | None,
    expected_coverage: LocationCoverageDefinition = EXPECTED_LOCATION_COVERAGE,
) -> CompletenessReport:
    """Combine source completeness, city-location coverage, every expected stream, counts and city integrity.

    ``city_integrity`` is required
    (:func:`~ql2_sixt_canada_analysis.city_integrity.assess_city_integrity`
    for ``expected_coverage``): ``None`` blocks, a report for another contract
    blocks, and an unassignable job city or a cross-city parent/detail row
    blocks - coverage, healthy streams and matching counts cannot override it.

    ``streams`` must be the validated all-expected-stream aggregate
    (:func:`~ql2_sixt_canada_analysis.streams.assess_expected_location_streams`)
    for ``expected_coverage``: a missing aggregate, an aggregate built for
    another contract, and any missing, duplicated, unexpected, unavailable,
    partial, unassessable or unhealthy expected stream block. Row-level
    coverage never substitutes for job-level stream continuity. Every input
    must be present and pass; each failure is reported.
    """
    B = CompletenessBlocker
    if not isinstance(datasets, RawDatasets):
        raise TypeError("datasets must be RawDatasets")
    if coverage is not None and not isinstance(coverage, LocationCoverageReport):
        raise TypeError("coverage must be a LocationCoverageReport or None")
    if reconciliation is not None and not isinstance(reconciliation, JobDetailReconciliationReport):
        raise TypeError("reconciliation must be a JobDetailReconciliationReport or None")
    if streams is not None and not isinstance(streams, ExpectedLocationStreamsReport):
        raise TypeError("streams must be an ExpectedLocationStreamsReport (all expected streams) or None")
    if city_integrity is not None and not isinstance(city_integrity, CityIntegrityReport):
        raise TypeError("city_integrity must be a CityIntegrityReport or None")
    if not isinstance(expected_coverage, LocationCoverageDefinition) or not expected_coverage.is_configured:
        raise LocationCoverageConfigurationError("completeness needs a configured expected-location contract")
    reasons: list[CompletenessBlocker] = []
    if datasets.complete_source is not True:
        reasons.append(B.SOURCE_NOT_COMPLETE)
    if coverage is None:
        reasons.append(B.COVERAGE_UNAVAILABLE)
    else:
        reasons.extend(_COVERAGE_BLOCKERS[v] for v in coverage.violations)
        if not coverage.is_valid and not coverage.violations:
            reasons.append(B.COVERAGE_UNAVAILABLE)
        if coverage.expected_pairs != tuple(expected_coverage.expected_locations):
            reasons.append(B.COVERAGE_CONTRACT_MISMATCH)
    if streams is None:
        reasons.append(B.EXPECTED_STREAM_ASSESSMENT_UNAVAILABLE)
    else:
        if streams.coverage != expected_coverage:
            reasons.append(B.STREAM_CONTRACT_MISMATCH)
        reasons.extend(B(b.value) for b in streams.blocking_reasons)
    if reconciliation is None:
        reasons.append(B.RECONCILIATION_UNAVAILABLE)
    else:
        if not all(f.reconciled for f in reconciliation.count_fields) or not reconciliation.count_fields:
            reasons.append(B.DECLARED_COUNT_UNRECONCILED)
        if not reconciliation.declared_counts_agree:
            reasons.append(B.DECLARED_COUNTS_DISAGREE)
        if not reconciliation.all_details_linked:
            reasons.append(B.DETAIL_ROWS_UNLINKED)
        if not reconciliation.parent_detail_scope_agrees:
            reasons.append(B.PARENT_DETAIL_CITY_MISMATCH)
        if not reconciliation.is_reconciled and not any(
                r in reasons for r in (B.DECLARED_COUNT_UNRECONCILED, B.DECLARED_COUNTS_DISAGREE,
                                       B.DETAIL_ROWS_UNLINKED, B.PARENT_DETAIL_CITY_MISMATCH)):
            reasons.append(B.DECLARED_COUNT_UNRECONCILED)     # fail closed on any other non-pass
    if city_integrity is None:
        reasons.append(B.CITY_INTEGRITY_UNAVAILABLE)
    else:
        if city_integrity.coverage != expected_coverage:
            reasons.append(B.CITY_INTEGRITY_CONTRACT_MISMATCH)
        reasons.extend(_CITY_BLOCKERS[b] for b in city_integrity.blocking_reasons)
        if not city_integrity.is_valid and not city_integrity.blocking_reasons:
            reasons.append(B.CITY_INTEGRITY_UNAVAILABLE)      # fail closed on any other non-pass
    return CompletenessReport(blocking_reasons=tuple(dict.fromkeys(reasons)), expected_streams=streams,
                              city_integrity=city_integrity)


def validate_pricing_readiness(**gates: object) -> PricingReadinessReport:
    """Return the readiness report if ready; else raise :class:`PricingNotReadyError`."""
    report = assess_pricing_readiness(**gates)  # type: ignore[arg-type]
    if not report.ready:
        raise PricingNotReadyError(report)
    return report
