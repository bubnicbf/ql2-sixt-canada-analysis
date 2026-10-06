"""Synthetic expected-stream contracts for tests (fabricated authority; never production configuration)."""

from __future__ import annotations

from ql2_sixt_canada_analysis.authority_decisions import DecisionStatus
from ql2_sixt_canada_analysis.expected_stream_contract import ExpectedStreamAuthorityStatus, ExpectedStreamContract
from ql2_sixt_canada_analysis.schemas import LocationCoverageDefinition


def synthetic_contract(coverage: LocationCoverageDefinition) -> ExpectedStreamContract:
    """An approved contract for ``coverage`` exactly as configured (its own mode decides exhaustiveness)."""
    return ExpectedStreamContract(
        status=ExpectedStreamAuthorityStatus.APPROVED, coverage=coverage, record_id="pricing-authorities-synthetic",
        universe_status=DecisionStatus.APPROVED, spelling_status=DecisionStatus.APPROVED,
        references=("SYNTH-GOVERNANCE-REFERENCE",))


def synthetic_location_authority(contract: ExpectedStreamContract, roles: dict, pairs: tuple, policy=None):  # type: ignore[no-untyped-def]
    """Approved synthetic roles and pairs validated against ``contract`` and ``policy`` (default: the project's)."""
    from ql2_sixt_canada_analysis.authority_decisions import LocationRoleDecision
    from ql2_sixt_canada_analysis.location_authority import (
        ComparisonPair,
        ComparisonPairSet,
        LocationAuthorityStatus,
        LocationRoleMap,
        assess_location_authority,
    )
    from ql2_sixt_canada_analysis.schemas import VANCOUVER_LOCATION_POLICY

    provenance = dict(record_id="pricing-authorities-synthetic", references=("SYNTH-GOVERNANCE-REFERENCE",))
    role_map = LocationRoleMap(status=LocationAuthorityStatus.APPROVED, assignments=tuple(
        (tuple(k), LocationRoleDecision(r) if isinstance(r, str) and r in LocationRoleDecision.__members__ else r)
        for k, r in (roles.items() if isinstance(roles, dict) else roles)), **provenance)
    pair_set = ComparisonPairSet(status=LocationAuthorityStatus.APPROVED, pairs=tuple(
        ComparisonPair(airport=tuple(a), downtown=tuple(d)) for a, d in pairs), **provenance)
    return assess_location_authority(role_map, pair_set, contract,
                                     policy if policy is not None else VANCOUVER_LOCATION_POLICY)
