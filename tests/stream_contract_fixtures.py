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


def synthetic_rental_policy():  # type: ignore[no-untyped-def]
    """The approved rental-date policy shape with fabricated provenance (tests only)."""
    from ql2_sixt_canada_analysis.rental_dates import RentalDatePolicy, RentalPolicyStatus

    return RentalDatePolicy(
        status=RentalPolicyStatus.APPROVED, source_format="ISO_8601_DATE", pickup_required=True,
        return_required=True, ordering="RETURN_ON_OR_AFTER_PICKUP", equal_dates_allowed=True,
        minimum_duration_days=0, maximum_duration_mode="UNBOUNDED",
        agreements=(("cars.job_pickup_date", "jobs.pickup_date"), ("cars.job_return_date", "jobs.return_date"),
                    ("cars.pickup_date", "jobs.pickup_date"), ("cars.return_date", "jobs.return_date")),
        record_id="pricing-authorities-synthetic", references=("SYNTH-GOVERNANCE-REFERENCE",))


def with_rental_dates(jobs, cars, pickup="2000-01-01", return_="2000-01-02"):  # type: ignore[no-untyped-def]
    """Copies of raw synthetic frames whose six rental-date fields hold one valid, agreeing period."""
    return (jobs.assign(pickup_date=pickup, return_date=return_),
            cars.assign(job_pickup_date=pickup, job_return_date=return_, pickup_date=pickup, return_date=return_))


def passing_rental_report(jobs, cars):  # type: ignore[no-untyped-def]
    """A rental-date report for linked synthetic frames whose dates are valid and agree."""
    from conftest import link

    from ql2_sixt_canada_analysis.rental_dates import assess_rental_dates

    result = link(*with_rental_dates(jobs, cars))
    return assess_rental_dates(result.jobs, result.cars, policy=synthetic_rental_policy(), job_linkage=result.report)
