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
