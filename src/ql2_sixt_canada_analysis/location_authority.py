"""Authority-backed location roles, comparison pairs and the Vancouver identity policy.

Three external decisions decide how approved source streams take part in
pricing analysis:

* ``LOCATION_ROLE_ASSIGNMENTS`` (business owner) - exactly one role
  (``AIRPORT``, ``DOWNTOWN`` or ``OTHER``) for every key of the exhaustive
  source-stream contract (:mod:`ql2_sixt_canada_analysis.expected_stream_contract`);
* ``VALID_LOCATION_COMPARISON_PAIRS`` (business owner) - the within-city
  airport/downtown pricing comparisons;
* ``VANCOUVER_LOCATION_IDENTITY`` (collection owner or supplier) - whether the
  two governed Vancouver keys are one location (``CONFIRMED_ALIAS`` with a
  canonical key) or two (``CONFIRMED_DISTINCT``).

This module turns the latest valid authority record into typed objects and
validates them against each other, fail closed. There is no hard-coded role
table, pair list or alias mapping: everything comes from APPROVED decisions.

Source keys and analytical identity stay separate. Roles are assigned to the
exact approved *source* keys (no case, whitespace or punctuation folding);
the Vancouver identity is the existing
:class:`~ql2_sixt_canada_analysis.schemas.LocationIdentityPolicy` built from
the record (:func:`vancouver_policy_from_record`), so there is one alias
mechanism only; comparison pairs are validated on the *canonical* analytical
keys that policy produces, so an aliased key can never create a second,
independent comparison, and the two aliases can never be compared with each
other. Canonicalization never changes source coverage: both raw streams stay
required by the source contract.

When a confirmed alias merges two source streams into one analytical
location, no approved rule says how their offers combine into one pricing
population (they may carry the same offers). That question stays explicit as
``canonical_offer_combination_unresolved``; nothing is dropped, deduplicated
or averaged here.

Reports hold enums, booleans, counts and approved configuration keys only -
never source values.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

from ql2_sixt_canada_analysis.authority_decisions import (
    AuthorityDecisionRecord,
    DecisionId,
    DecisionStatus,
    LocationRoleDecision,
    load_current_decision_record,
)
from ql2_sixt_canada_analysis.expected_stream_contract import ExpectedStreamContract
from ql2_sixt_canada_analysis.schemas import (
    COMPARED_LOCATION_STREAMS,
    LocationCoverageDefinition,
    LocationIdentityPolicy,
    LocationPolicyAuthority,
    LocationPolicyState,
    assess_location_policy_scope,
)

__all__ = [
    "LOCATION_DECISIONS",
    "ComparisonPair",
    "ComparisonPairDefect",
    "ComparisonPairSet",
    "LocationAuthorityBlocker",
    "LocationAuthorityReport",
    "LocationAuthorityStatus",
    "LocationRoleMap",
    "RoleMapDefect",
    "assess_location_authority",
    "comparison_pairs_from_record",
    "current_location_authority",
    "location_authority_from_record",
    "role_map_from_record",
    "vancouver_policy_from_record",
]

D = DecisionId
#: The three decisions this module consumes.
LOCATION_DECISIONS: tuple[DecisionId, ...] = (
    D.LOCATION_ROLE_ASSIGNMENTS, D.VALID_LOCATION_COMPARISON_PAIRS, D.VANCOUVER_LOCATION_IDENTITY)

Key = tuple[str, ...]


class LocationAuthorityStatus(StrEnum):
    APPROVED = "approved"
    NOT_APPROVED = "not_approved"               # PROPOSED or REJECTED
    RECORD_UNAVAILABLE = "record_unavailable"   # no valid authority record


class LocationAuthorityBlocker(StrEnum):
    """Why roles or pairs cannot support pricing (values equal ``PricingBlocker`` values)."""

    BRANCH_ROLE_AUTHORITY_UNAVAILABLE = "branch_role_authority_unavailable"
    BRANCH_ROLES_NOT_EXACT = "branch_roles_not_exact"
    COMPARISON_PAIR_AUTHORITY_UNAVAILABLE = "comparison_pair_authority_unavailable"
    COMPARISON_PAIRS_INVALID = "comparison_pairs_invalid"
    COMPARISON_PAIR_IDENTITY_UNRESOLVED = "comparison_pair_identity_unresolved"
    CANONICAL_OFFER_COMBINATION_UNRESOLVED = "canonical_offer_combination_unresolved"


class RoleMapDefect(StrEnum):
    CONTRACT_UNAVAILABLE = "role_contract_unavailable"       # no usable exhaustive source contract
    MISSING_ROLE = "role_missing"
    UNEXPECTED_KEY = "role_key_outside_contract"
    DUPLICATE_ASSIGNMENT = "role_assigned_twice"
    INVALID_ROLE = "role_invalid"
    ALIAS_ROLE_CONFLICT = "aliased_keys_roles_differ"


class ComparisonPairDefect(StrEnum):
    ROLES_UNAVAILABLE = "pair_roles_unavailable"
    UNKNOWN_KEY = "pair_key_outside_contract"
    SELF_PAIR = "pair_self_comparison"
    CROSS_CITY = "pair_crosses_city"
    ROLE_MISMATCH = "pair_not_airport_downtown"
    NON_CANONICAL_MEMBER = "pair_member_not_canonical"
    SAME_CANONICAL_LOCATION = "pair_resolves_to_one_location"
    DUPLICATE_PAIR = "pair_duplicated"
    IDENTITY_UNRESOLVED = "pair_identity_unresolved"
    EMPTY = "pairs_empty"


@dataclass(frozen=True, slots=True)
class LocationRoleMap:
    """Role assignments as approved (exact source keys, record order; ``()`` unless approved)."""

    status: LocationAuthorityStatus
    assignments: tuple[tuple[Key, object], ...] = ()
    record_id: str | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, LocationAuthorityStatus):
            raise TypeError("status must be a LocationAuthorityStatus")
        if self.status is not LocationAuthorityStatus.APPROVED and self.assignments:
            raise ValueError("only an approved role map carries assignments")
        if self.status is LocationAuthorityStatus.APPROVED and (not self.record_id or not self.references):
            raise ValueError("an approved role map needs its record and references")

    def role(self, key: Key) -> LocationRoleDecision | None:
        """The single valid role of an exact key, else ``None`` (missing, duplicated or invalid)."""
        roles = [r for k, r in self.assignments if k == key]
        return roles[0] if len(roles) == 1 and isinstance(roles[0], LocationRoleDecision) else None


@dataclass(frozen=True, slots=True)
class ComparisonPair:
    """One airport/downtown comparison as declared (source keys, airport first)."""

    airport: Key
    downtown: Key


@dataclass(frozen=True, slots=True)
class ComparisonPairSet:
    """Comparison pairs as approved (record order; ``()`` unless approved)."""

    status: LocationAuthorityStatus
    pairs: tuple[ComparisonPair, ...] = ()
    record_id: str | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, LocationAuthorityStatus):
            raise TypeError("status must be a LocationAuthorityStatus")
        if not all(isinstance(p, ComparisonPair) for p in self.pairs):
            raise TypeError("pairs must be ComparisonPair objects")
        if self.status is not LocationAuthorityStatus.APPROVED and self.pairs:
            raise ValueError("only an approved pair set carries pairs")
        if self.status is LocationAuthorityStatus.APPROVED and (not self.record_id or not self.references):
            raise ValueError("an approved pair set needs its record and references")


@dataclass(frozen=True, slots=True)
class LocationAuthorityReport:
    """Validated roles and comparison pairs for one source contract and identity policy (fail closed)."""

    role_map: LocationRoleMap
    pair_set: ComparisonPairSet
    contract: ExpectedStreamContract
    policy: LocationIdentityPolicy
    role_defects: tuple[RoleMapDefect, ...]
    pair_defects: tuple[ComparisonPairDefect, ...]
    #: Validated pricing comparisons on canonical analytical keys (empty unless every pair is valid).
    effective_pairs: tuple[ComparisonPair, ...]
    #: A confirmed alias merges approved source streams of this contract into one analytical location.
    canonicalization_merges_streams: bool

    def __post_init__(self) -> None:
        # Fail closed: a report is only ever the result of the central validation of its own inputs.
        again = assess_location_authority(self.role_map, self.pair_set, self.contract, self.policy, _validating=True)
        if (self.role_defects, self.pair_defects, self.effective_pairs, self.canonicalization_merges_streams) != (
                again["role_defects"], again["pair_defects"], again["effective_pairs"],
                again["canonicalization_merges_streams"]):
            raise ValueError("a location authority report must equal the validation of its inputs")

    @property
    def referenced_keys(self) -> frozenset:
        """Every source key the contract, the role map or the pairs refer to (configuration only)."""
        keys = set(self.contract.expected_keys) | {k for k, _ in self.role_map.assignments}
        keys |= {k for p in self.pair_set.pairs for k in (p.airport, p.downtown)}
        return frozenset(keys)

    @property
    def roles_exact(self) -> bool:
        return self.role_map.status is LocationAuthorityStatus.APPROVED and not self.role_defects

    @property
    def pairs_valid(self) -> bool:
        return (self.pair_set.status is LocationAuthorityStatus.APPROVED and self.roles_exact
                and not self.pair_defects and bool(self.effective_pairs))

    def canonical(self, key: Key) -> Key:
        """The analytical key of an exact source key under the policy (itself unless a valid alias)."""
        return _canonical_mapping(self.policy).get(key, key)

    def canonical_role(self, key: Key) -> LocationRoleDecision | None:
        """The role of a key's canonical analytical location (``None`` unless the role map is exact)."""
        return self.role_map.role(self.canonical(key)) if self.roles_exact else None

    @property
    def role_counts(self) -> Mapping[str, int]:
        """Number of approved source keys per role (exact role maps only)."""
        if not self.roles_exact:
            return {}
        counts = Counter(r.value for _, r in self.role_map.assignments)
        return {role.value: counts.get(role.value, 0) for role in LocationRoleDecision}

    @property
    def blocking_reasons(self) -> tuple[LocationAuthorityBlocker, ...]:
        B = LocationAuthorityBlocker
        found: list[LocationAuthorityBlocker] = []
        if self.role_map.status is not LocationAuthorityStatus.APPROVED:
            found.append(B.BRANCH_ROLE_AUTHORITY_UNAVAILABLE)
        elif self.role_defects:
            found.append(B.BRANCH_ROLES_NOT_EXACT)
        if self.pair_set.status is not LocationAuthorityStatus.APPROVED:
            found.append(B.COMPARISON_PAIR_AUTHORITY_UNAVAILABLE)
        else:
            if ComparisonPairDefect.IDENTITY_UNRESOLVED in self.pair_defects:
                found.append(B.COMPARISON_PAIR_IDENTITY_UNRESOLVED)
            if set(self.pair_defects) - {ComparisonPairDefect.IDENTITY_UNRESOLVED}:
                found.append(B.COMPARISON_PAIRS_INVALID)
        if self.canonicalization_merges_streams:
            found.append(B.CANONICAL_OFFER_COMBINATION_UNRESOLVED)
        return tuple(b for b in B if b in found)


# ---------------------------------------------------------------- from the record


def _status(record: AuthorityDecisionRecord | None, decision: DecisionId) -> LocationAuthorityStatus:
    if record is None:
        return LocationAuthorityStatus.RECORD_UNAVAILABLE
    if not isinstance(record, AuthorityDecisionRecord):
        raise TypeError("record must be a validated AuthorityDecisionRecord or None")
    return (LocationAuthorityStatus.APPROVED if record.decision(decision).is_approved
            else LocationAuthorityStatus.NOT_APPROVED)


def _provenance(record: AuthorityDecisionRecord, decision: DecisionId) -> dict:
    entry = record.decision(decision)
    return dict(record_id=record.record_id, references=tuple(sorted({a.reference for a in entry.authority})))


def role_map_from_record(record: AuthorityDecisionRecord | None) -> LocationRoleMap:
    """The approved role assignments (exact keys as recorded), or an unavailable map."""
    status = _status(record, D.LOCATION_ROLE_ASSIGNMENTS)
    if status is not LocationAuthorityStatus.APPROVED:
        return LocationRoleMap(status=status)
    resolution = record.approved_resolution(D.LOCATION_ROLE_ASSIGNMENTS)
    assignments = tuple((tuple(a["stream"]), LocationRoleDecision(a["role"])) for a in resolution["assignments"])
    return LocationRoleMap(status=status, assignments=assignments, **_provenance(record, D.LOCATION_ROLE_ASSIGNMENTS))


def comparison_pairs_from_record(record: AuthorityDecisionRecord | None) -> ComparisonPairSet:
    """The approved comparison pairs as recorded, or an unavailable set (never generated from data)."""
    status = _status(record, D.VALID_LOCATION_COMPARISON_PAIRS)
    if status is not LocationAuthorityStatus.APPROVED:
        return ComparisonPairSet(status=status)
    resolution = record.approved_resolution(D.VALID_LOCATION_COMPARISON_PAIRS)
    pairs = tuple(ComparisonPair(airport=tuple(p["airport"]), downtown=tuple(p["downtown"]))
                  for p in resolution["pairs"])
    return ComparisonPairSet(status=status, pairs=pairs, **_provenance(record, D.VALID_LOCATION_COMPARISON_PAIRS))


def vancouver_policy_from_record(record: AuthorityDecisionRecord | None,
                                 coverage: LocationCoverageDefinition) -> LocationIdentityPolicy:
    """The Vancouver :class:`LocationIdentityPolicy` from the APPROVED decision; ``UNRESOLVED`` otherwise.

    The governed keys are :data:`COMPARED_LOCATION_STREAMS`; authority
    metadata names the record and its governance reference. A missing,
    proposed or rejected decision (or no valid record) yields ``UNRESOLVED``;
    behavioural evidence is never consulted.
    """
    first, second = COMPARED_LOCATION_STREAMS
    unresolved = LocationIdentityPolicy(first=first, second=second, coverage=coverage,
                                        state=LocationPolicyState.UNRESOLVED)
    if _status(record, D.VANCOUVER_LOCATION_IDENTITY) is not LocationAuthorityStatus.APPROVED:
        return unresolved
    entry = record.decision(D.VANCOUVER_LOCATION_IDENTITY)
    resolution = entry.resolution
    authority = entry.authority[0]
    metadata = LocationPolicyAuthority(
        source=authority.source, reference=authority.reference,
        note=f"{record.record_id}: {authority.kind.value} decision {D.VANCOUVER_LOCATION_IDENTITY.value}")
    if resolution["state"] == "CONFIRMED_ALIAS":
        return LocationIdentityPolicy(first=first, second=second, coverage=coverage,
                                      state=LocationPolicyState.CONFIRMED_ALIAS, authority=metadata,
                                      canonical_location=tuple(resolution["canonical_location"]))
    return LocationIdentityPolicy(first=first, second=second, coverage=coverage,
                                  state=LocationPolicyState.CONFIRMED_DISTINCT, authority=metadata)


# ------------------------------------------------------------------- validation


def _canonical_mapping(policy: LocationIdentityPolicy) -> dict[Key, Key]:
    """Governed key -> canonical key for a resolved, in-scope confirmed alias (else empty)."""
    if (policy.state is not LocationPolicyState.CONFIRMED_ALIAS or policy.authority is None
            or not assess_location_policy_scope(policy).is_valid):
        return {}
    return {policy.first: policy.canonical_location, policy.second: policy.canonical_location}


def _identity_resolved(policy: LocationIdentityPolicy) -> bool:
    return (policy.state is not LocationPolicyState.UNRESOLVED and policy.authority is not None
            and assess_location_policy_scope(policy).is_valid)


def _construct(role_map, pair_set, contract, policy, fields):  # type: ignore[no-untyped-def]
    return LocationAuthorityReport(role_map=role_map, pair_set=pair_set, contract=contract, policy=policy, **fields)


def assess_location_authority(role_map: LocationRoleMap, pair_set: ComparisonPairSet,
                              contract: ExpectedStreamContract,
                              policy: LocationIdentityPolicy, *,
                              _validating: bool = False) -> LocationAuthorityReport:
    """Validate roles against the exhaustive contract and pairs against roles, contract and policy.

    Roles: exactly one valid role per approved source key, compared exactly;
    none for a key outside the contract; aliased keys must share their role.
    Pairs: both members approved source keys of one city, one ``AIRPORT`` and
    one ``DOWNTOWN``, never the same key, both already canonical under the
    identity policy, resolving to two different canonical locations, each
    unordered canonical pair at most once (a reversed copy is a duplicate). A
    pair touching a governed key needs a resolved identity policy. Nothing is
    inferred from data or names; inputs are not modified.
    """
    if not isinstance(role_map, LocationRoleMap) or not isinstance(pair_set, ComparisonPairSet):
        raise TypeError("role_map and pair_set must be a LocationRoleMap and a ComparisonPairSet")
    if not isinstance(contract, ExpectedStreamContract):
        raise TypeError("contract must be an ExpectedStreamContract")
    if not isinstance(policy, LocationIdentityPolicy):
        raise TypeError("policy must be a LocationIdentityPolicy")
    expected = list(contract.expected_keys)
    mapping = _canonical_mapping(policy)

    R = RoleMapDefect
    role_defects: set[RoleMapDefect] = set()
    if role_map.status is LocationAuthorityStatus.APPROVED:
        if not contract.usable:
            role_defects.add(R.CONTRACT_UNAVAILABLE)
        counts = Counter(k for k, _ in role_map.assignments)
        if any(n > 1 for n in counts.values()):
            role_defects.add(R.DUPLICATE_ASSIGNMENT)
        if any(k not in expected for k in counts):
            role_defects.add(R.UNEXPECTED_KEY)
        if any(counts[k] == 0 for k in expected) or not expected:
            role_defects.add(R.MISSING_ROLE)
        if any(not isinstance(r, LocationRoleDecision) for _, r in role_map.assignments):
            role_defects.add(R.INVALID_ROLE)
        if mapping and len({role_map.role(k) for k in mapping}) != 1:
            role_defects.add(R.ALIAS_ROLE_CONFLICT)
    roles_exact = role_map.status is LocationAuthorityStatus.APPROVED and not role_defects

    P = ComparisonPairDefect
    pair_defects: set[ComparisonPairDefect] = set()
    effective: list[ComparisonPair] = []
    if pair_set.status is LocationAuthorityStatus.APPROVED:
        if not pair_set.pairs:
            pair_defects.add(P.EMPTY)
        if not roles_exact:
            pair_defects.add(P.ROLES_UNAVAILABLE)
        governed = {policy.first, policy.second}
        seen: set[frozenset] = set()
        for pair in pair_set.pairs:
            a, d = pair.airport, pair.downtown
            if a not in expected or d not in expected:
                pair_defects.add(P.UNKNOWN_KEY)
            if a == d:
                pair_defects.add(P.SELF_PAIR)
            if not (isinstance(a, tuple) and isinstance(d, tuple) and a[:1] == d[:1] and a[:1]):
                pair_defects.add(P.CROSS_CITY)
            if roles_exact and (role_map.role(a) is not LocationRoleDecision.AIRPORT
                                or role_map.role(d) is not LocationRoleDecision.DOWNTOWN):
                pair_defects.add(P.ROLE_MISMATCH)
            if {a, d} & governed and not _identity_resolved(policy):
                pair_defects.add(P.IDENTITY_UNRESOLVED)
            ca, cd = mapping.get(a, a), mapping.get(d, d)
            if (ca, cd) != (a, d):
                pair_defects.add(P.NON_CANONICAL_MEMBER)
            if ca == cd:
                pair_defects.add(P.SAME_CANONICAL_LOCATION)
            if isinstance(ca, tuple) and isinstance(cd, tuple) and ca[:1] != cd[:1]:
                pair_defects.add(P.CROSS_CITY)
            resolved = frozenset({ca, cd})
            if resolved in seen:
                pair_defects.add(P.DUPLICATE_PAIR)
            seen.add(resolved)
            effective.append(ComparisonPair(airport=ca, downtown=cd))
    fields = dict(role_defects=tuple(d for d in R if d in role_defects),
                  pair_defects=tuple(d for d in P if d in pair_defects),
                  effective_pairs=tuple(effective) if not pair_defects else (),
                  canonicalization_merges_streams=bool(mapping) and set(mapping) <= set(expected))
    if _validating:
        return fields  # type: ignore[return-value]
    return _construct(role_map, pair_set, contract, policy, fields)


def location_authority_from_record(record: AuthorityDecisionRecord | None, contract: ExpectedStreamContract,
                                   policy: LocationIdentityPolicy) -> LocationAuthorityReport:
    """Roles and pairs from the record, validated against ``contract`` and ``policy``."""
    return assess_location_authority(role_map_from_record(record), comparison_pairs_from_record(record),
                                     contract, policy)


@cache
def current_location_authority() -> LocationAuthorityReport:
    """The project's location authority: current record, current contract and the project Vancouver policy."""
    from ql2_sixt_canada_analysis.expected_stream_contract import current_expected_stream_contract
    from ql2_sixt_canada_analysis.schemas import project_default, PROJECT_DEFAULT

    policy = project_default(PROJECT_DEFAULT, "VANCOUVER_LOCATION_POLICY")
    return location_authority_from_record(load_current_decision_record(), current_expected_stream_contract(),
                                          policy)  # type: ignore[arg-type]
