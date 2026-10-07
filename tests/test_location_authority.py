"""Approved location roles, comparison pairs and the Vancouver alias policy (approved in pricing-authorities-v4,
carried into v5 with the corrected exact raw source keys).

The committed records are read as data; negative cases mutate parsed copies
in memory, build synthetic role maps / pair sets, or use a temporary
repository root. Frames are fabricated (``SYNTH-JOB-*``); the only stream
values are approved keys (governance configuration, never observations).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from stream_contract_fixtures import synthetic_contract
from test_completeness import cars as cars_frame, jobs as jobs_frame
from test_expected_stream_contract import DISPLAY_SPELLINGS, SUPPLIED, completeness_of, healthy_frames, pricing_of
from test_readiness import GATES, evidence

from ql2_sixt_canada_analysis.authority_decisions import (
    CURRENT_RECORD_PATH,
    JOB_IDENTIFIER_DECISIONS,
    AuthorityKind,
    DecisionId,
    DecisionRecordError,
    DecisionStatus,
    LocationRoleDecision as R,
    load_current_decision_record,
    load_decision_record,
    parse_decision_record,
    render_authority_request_checklist,
)
from ql2_sixt_canada_analysis.comparison import LocationStreamComparisonStatus as CS
from ql2_sixt_canada_analysis.expected_stream_contract import (
    EXPECTED_STREAM_DECISIONS,
    current_expected_stream_contract,
    expected_stream_contract_from_record,
)
from ql2_sixt_canada_analysis.location_authority import (
    LOCATION_DECISIONS,
    ComparisonPair,
    ComparisonPairDefect as PD,
    ComparisonPairSet,
    LocationAuthorityBlocker as LB,
    LocationAuthorityReport,
    LocationAuthorityStatus as ST,
    LocationRoleMap,
    RoleMapDefect as RD,
    assess_location_authority,
    comparison_pairs_from_record,
    current_location_authority,
    location_authority_from_record,
    role_map_from_record,
    vancouver_policy_from_record,
)
from ql2_sixt_canada_analysis.readiness import (
    CompletenessBlocker as CMP,
    PricingBlocker as PB,
    apply_location_policy,
    assess_location_policy,
    assess_pricing_readiness,
)
from ql2_sixt_canada_analysis.schemas import (
    EXPECTED_LOCATION_COVERAGE as COV,
    VANCOUVER_LOCATION_POLICY as POLICY,
    LocationPolicyScopeDefect,
    LocationPolicyState as PS,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V1, V2, V3, V4, V5 = (RECORD_DIR / f"v{n}.toml" for n in (1, 2, 3, 4, 5))
SCHEDULE_GOVERNANCE = "docs/decisions/governance/collection-schedule-governance-v1-2026-10-06.md"
GOVERNANCE = "docs/decisions/governance/location-roles-and-identity-governance-2026-10-06.md"
HISTORY_SHA256 = {
    "v1.toml": "b13881b099885130e853207f872e62bfde9820f438e34859f3ddef0ebdbf1bd7",
    "v2.toml": "899c20e289d932868b1d430b8fd70edca6ebb2900ca963c4c040cc6f5840cb4c",
    "v3.toml": "431ec36a8d60b058a9d2ede401ab0274dea0bc525f8b025f2b04cbdff0aa2d79",
    "v4.toml": "28a412352884a28c7ebf6874aa1bfc25182c13f210357861933001886a25d552",
}
D = DecisionId
CAL_DOWN, CAL_AIR, TOR_DOWN, TOR_AIR, VAN_DOWN, VAN_AIR, VAN_THUR = SUPPLIED
#: The decisions exactly as supplied (test oracle).
ROLES = {CAL_DOWN: R.DOWNTOWN, CAL_AIR: R.AIRPORT, TOR_DOWN: R.DOWNTOWN, TOR_AIR: R.AIRPORT,
         VAN_DOWN: R.DOWNTOWN, VAN_AIR: R.AIRPORT, VAN_THUR: R.DOWNTOWN}
PAIRS = ((CAL_AIR, CAL_DOWN), (TOR_AIR, TOR_DOWN), (VAN_AIR, VAN_DOWN))
#: History oracle: the same decisions under the display-style spellings recorded in v4.
_DISPLAY = dict(zip(SUPPLIED, DISPLAY_SPELLINGS))
ROLES_V4 = {_DISPLAY[k]: r for k, r in ROLES.items()}
PAIRS_V4 = tuple((_DISPLAY[a], _DISPLAY[d]) for a, d in PAIRS)
CONTRACT = current_expected_stream_contract()
PROVENANCE = dict(record_id="pricing-authorities-synthetic", references=("SYNTH-GOVERNANCE-REFERENCE",))


def v4() -> dict:
    return tomllib.loads(V4.read_text(encoding="utf-8"))


def v5() -> dict:
    return tomllib.loads(V5.read_text(encoding="utf-8"))


def entry(data: dict, decision: DecisionId) -> dict:
    return next(e for e in data["decisions"] if e["id"] == decision.value)


def finish(data: dict) -> dict:
    statuses = [e["status"] for e in data["decisions"]]
    data["summary"] = {s.value.lower(): statuses.count(s.value) for s in DecisionStatus}
    data["external_inputs"] = [e["id"] for e in data["decisions"] if e["blocking_external_input"]]
    return data


def unapprove(data: dict, decision: DecisionId) -> dict:
    item = entry(data, decision)
    item.update(status="PROPOSED", blocking_external_input=True)
    item.pop("authority", None), item.pop("resolution", None)
    return finish(data)


def fails(data: dict, **kwargs: object) -> str:
    with pytest.raises(DecisionRecordError) as info:
        parse_decision_record(finish(data), **kwargs)
    return str(info.value)


def roles(assignments=None) -> LocationRoleMap:  # type: ignore[no-untyped-def]
    items = ROLES.items() if assignments is None else assignments
    return LocationRoleMap(status=ST.APPROVED, assignments=tuple((tuple(k), r) for k, r in items), **PROVENANCE)


def pairs(items=PAIRS) -> ComparisonPairSet:  # type: ignore[no-untyped-def]
    return ComparisonPairSet(status=ST.APPROVED, pairs=tuple(ComparisonPair(tuple(a), tuple(d)) for a, d in items),
                             **PROVENANCE)


def assess(role_map=None, pair_set=None, policy=POLICY, contract=CONTRACT):  # type: ignore[no-untyped-def]
    return assess_location_authority(role_map if role_map is not None else roles(),
                                     pair_set if pair_set is not None else pairs(), contract, policy)


UNDECIDED = dataclasses.replace(POLICY, state=PS.UNRESOLVED, authority=None, canonical_location=None)


# ======================================================= authority record (v4, v5)


def test_history_is_unchanged_and_valid() -> None:
    for name, digest in HISTORY_SHA256.items():
        assert hashlib.sha256((RECORD_DIR / name).read_bytes()).hexdigest() == digest
    for path in (V1, V2, V3, V4):
        load_decision_record(path)
    v3 = load_decision_record(V3)
    assert all(v3.decision(d).status is DecisionStatus.PROPOSED for d in LOCATION_DECISIONS)


def test_v4_is_history_and_approves_exactly_nine_decisions() -> None:
    record = load_decision_record(V4)
    assert (record.schema_version, record.record_version, record.supersedes) == (2, 4, "pricing-authorities-v3")
    assert CURRENT_RECORD_PATH.name == "v7.toml" and load_current_decision_record() != record
    approved = {d.id for d in record.decisions if d.is_approved}
    assert approved == set(JOB_IDENTIFIER_DECISIONS) | set(EXPECTED_STREAM_DECISIONS) | set(LOCATION_DECISIONS)
    counts = record.counts()
    assert (counts[DecisionStatus.APPROVED], counts[DecisionStatus.PROPOSED], counts[DecisionStatus.REJECTED]) == (9, 13, 0)
    for decision in record.decisions:
        if decision.id not in approved:
            assert decision.status is DecisionStatus.PROPOSED and decision.blocking_external_input
    v3 = load_decision_record(V3)
    for decision in (*JOB_IDENTIFIER_DECISIONS, *EXPECTED_STREAM_DECISIONS):     # carried over unchanged
        assert record.decision(decision).resolution == v3.decision(decision).resolution
        assert record.decision(decision).authority == v3.decision(decision).authority


def test_v4_resolutions_are_exactly_the_supplied_decisions() -> None:
    record = load_decision_record(V4)
    assignments = record.approved_resolution(D.LOCATION_ROLE_ASSIGNMENTS)["assignments"]
    assert {tuple(a["stream"]): R(a["role"]) for a in assignments} == ROLES_V4 and len(assignments) == 7
    declared = record.approved_resolution(D.VALID_LOCATION_COMPARISON_PAIRS)["pairs"]
    assert tuple((tuple(p["airport"]), tuple(p["downtown"])) for p in declared) == PAIRS_V4
    identity = record.approved_resolution(D.VANCOUVER_LOCATION_IDENTITY)
    assert identity["state"] == "CONFIRMED_ALIAS" and tuple(identity["canonical_location"]) == _DISPLAY[VAN_DOWN]
    # v4 governs its own (superseded) spellings, which are not keys of the current contract: unresolved.
    assert vancouver_policy_from_record(record, COV).state is PS.UNRESOLVED


def test_v5_carries_the_location_decisions_with_the_corrected_keys() -> None:
    record, v4_record = load_decision_record(V5), load_decision_record(V4)
    assert (record.schema_version, record.record_version, record.supersedes) == (3, 5, "pricing-authorities-v4")
    current = load_current_decision_record()                         # v6 carries the location decisions unchanged
    assert all(current.decision(d) == record.decision(d) for d in LOCATION_DECISIONS)
    assignments = record.approved_resolution(D.LOCATION_ROLE_ASSIGNMENTS)["assignments"]
    assert {tuple(a["stream"]): R(a["role"]) for a in assignments} == ROLES and len(assignments) == 7
    declared = record.approved_resolution(D.VALID_LOCATION_COMPARISON_PAIRS)["pairs"]
    assert tuple((tuple(p["airport"]), tuple(p["downtown"])) for p in declared) == PAIRS
    identity = record.approved_resolution(D.VANCOUVER_LOCATION_IDENTITY)
    assert identity["state"] == "CONFIRMED_ALIAS" and tuple(identity["canonical_location"]) == VAN_DOWN
    assert tuple(map(tuple, identity["governed_locations"])) == (VAN_DOWN, VAN_THUR)
    for decision in LOCATION_DECISIONS:                               # same authority; respelling is traceable
        assert record.decision(decision).authority == v4_record.decision(decision).authority
        assert [e.reference for e in record.decision(decision).evidence] == [SCHEDULE_GOVERNANCE]


@pytest.mark.parametrize("path", [V4, V5])
def test_location_authority_roles_and_governance_reference(path) -> None:
    record = load_decision_record(path)
    for decision, kind in ((D.LOCATION_ROLE_ASSIGNMENTS, AuthorityKind.BUSINESS_OWNER),
                           (D.VALID_LOCATION_COMPARISON_PAIRS, AuthorityKind.BUSINESS_OWNER),
                           (D.VANCOUVER_LOCATION_IDENTITY, AuthorityKind.COLLECTION_OWNER)):
        authority = record.decision(decision).authority
        assert [a.kind for a in authority] == [kind] and authority[0].reference == GOVERNANCE
        assert all(e.kind.value != "RAW_DATA_OBSERVATION" for e in record.decision(decision).evidence)
    assert (ROOT / GOVERNANCE).is_file()


def test_governance_document_records_the_supplied_decisions_only() -> None:
    raw = (ROOT / GOVERNANCE).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in ("supplied by the repository owner", "2026-10-06", "business owner", "collection owner",
                   "CONFIRMED_ALIAS", "canonical location is **`Vancouver / Downtown`**",
                   "Both raw source labels must be preserved", "Neither key may canonicalize into another city",
                   "never establishes, determines or overrides this identity decision",
                   "current analyzed dataset", "requires a new authority-record version",
                   "Not supplied", "pricing-authorities-v4", "collection schedule", "temporal rules",
                   "reporting-day rules", "rental-date rules", "explicit blocker"):
        assert phrase in text, phrase
    for (city, location), role in ROLES_V4.items():                 # as spelled when supplied (history)
        assert f"| `{city}` | `{location}` | `{role.value}` |" in raw
    assert "@" not in raw and not re.search(r"#\d|[A-Z]{2,}-\d+", raw)
    assert not re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\d{6,}|\d+\.\d{2}", raw)


def test_missing_or_external_governance_reference_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    area = root / "docs" / "decisions" / "governance"
    area.mkdir(parents=True)
    for name in ("job-identifier-governance-2026-10-06.md", "expected-stream-governance-2026-10-06.md"):
        (area / name).write_text("# synthetic\n", encoding="utf-8")
    with pytest.raises(DecisionRecordError, match="missing"):
        load_decision_record(V4, repository_root=root)
    for reference in ("docs/decisions/governance/synth-missing.md", "../" + GOVERNANCE, "https://synth.invalid/x.md",
                      "SYNTH-REFERENCE"):
        data = v4()
        entry(data, D.VANCOUVER_LOCATION_IDENTITY)["authority"][0]["reference"] = reference
        assert "governance document" in fails(data) or "missing" in fails(data)
    assert vancouver_policy_from_record(None, COV).state is PS.UNRESOLVED   # unavailable record -> unresolved


@pytest.mark.parametrize("canonical", [CAL_DOWN, TOR_DOWN, VAN_AIR, ("vancouver", "SYNTH Branch"),
                                       ("vancouver", "vancouver downtown")])
def test_canonical_key_outside_the_governed_vancouver_keys_is_rejected(canonical) -> None:
    data = v5()
    entry(data, D.VANCOUVER_LOCATION_IDENTITY)["resolution"]["canonical_location"] = list(canonical)
    assert "canonical key must be one of the two governed keys" in fails(data)


def test_record_rejects_pairs_and_roles_that_contradict_the_alias() -> None:
    data = v5()                                                      # a second comparison through Thurlow
    entry(data, D.VALID_LOCATION_COMPARISON_PAIRS)["resolution"]["pairs"].append(
        {"airport": list(VAN_AIR), "downtown": list(VAN_THUR)})
    assert "canonical locations" in fails(data)
    data = v5()                                                      # a reversed copy fails the role rule
    entry(data, D.VALID_LOCATION_COMPARISON_PAIRS)["resolution"]["pairs"].append(
        {"airport": list(CAL_DOWN), "downtown": list(CAL_AIR)})
    assert "airport/downtown roles" in fails(data)
    data = v5()
    entry(data, D.VALID_LOCATION_COMPARISON_PAIRS)["resolution"]["pairs"].append(
        {"airport": list(CAL_AIR), "downtown": list(TOR_DOWN)})
    assert "within one city" in fails(data)
    data = v5()
    entry(data, D.VALID_LOCATION_COMPARISON_PAIRS)["resolution"]["pairs"].append(copy.deepcopy(
        entry(data, D.VALID_LOCATION_COMPARISON_PAIRS)["resolution"]["pairs"][0]))
    assert "duplicate" in fails(data)
    data = v5()                                                      # aliases need one role
    next(a for a in entry(data, D.LOCATION_ROLE_ASSIGNMENTS)["resolution"]["assignments"]
         if a["stream"] == list(VAN_THUR))["role"] = "OTHER"
    assert "same approved role" in fails(data)


def test_checklist_is_regenerated_from_the_current_record() -> None:
    checklist = render_authority_request_checklist(load_current_decision_record())
    assert (RECORD_DIR / "authority_request_checklist.md").read_text(encoding="utf-8") == checklist
    resolved, requests = checklist.split("## Resolved decisions")[1].split("\n## ", 1)
    for decision in LOCATION_DECISIONS:
        assert f"`{decision.value}`" in resolved and f"`{decision.value}`" not in requests
    assert GOVERNANCE in resolved
    assert len([line for line in requests.splitlines() if line.startswith("| `")]) == 4


def test_record_readme_identifies_v5_as_current() -> None:
    readme = (RECORD_DIR / "README.md").read_text(encoding="utf-8")
    for phrase in ("v4.toml", "v5.toml", "v6.toml", "v7.toml", "current revision", "CONFIRMED_ALIAS", Path(GOVERNANCE).name,
                   Path(SCHEDULE_GOVERNANCE).name):
        assert phrase in readme, phrase


# ==================================================================== role map


def test_project_role_map_comes_from_the_record_and_is_exact() -> None:
    report = current_location_authority()
    assert report.role_map == role_map_from_record(load_current_decision_record())
    assert dict(report.role_map.assignments) == ROLES and report.roles_exact and report.role_defects == ()
    assert report.role_counts == {"AIRPORT": 3, "DOWNTOWN": 4, "OTHER": 0}
    assert report.role_map.role(VAN_DOWN) is report.role_map.role(VAN_THUR) is R.DOWNTOWN   # both source roles
    assert report.canonical_role(VAN_THUR) is R.DOWNTOWN and report.canonical(VAN_THUR) == VAN_DOWN


@pytest.mark.parametrize("change, defect", [
    (lambda r: {k: v for k, v in r.items() if k != TOR_AIR}, RD.MISSING_ROLE),                       # missing
    (lambda r: {**r, ("Toronto", "SYNTH Branch"): R.DOWNTOWN}, RD.UNEXPECTED_KEY),                   # extra
    (lambda r: [*r.items(), (CAL_AIR, R.AIRPORT)], RD.DUPLICATE_ASSIGNMENT),                          # duplicated
    (lambda r: [*((k, v) for k, v in r.items() if k != CAL_AIR), (CAL_AIR, "AIRPORT_ISH")], RD.INVALID_ROLE),
    (lambda r: {**{k: v for k, v in r.items() if k != CAL_AIR}, ("calgary", "Int Airport"): R.AIRPORT}, RD.MISSING_ROLE),
    (lambda r: {**{k: v for k, v in r.items() if k != CAL_AIR}, ("Calgary", "Int Airport "): R.AIRPORT}, RD.MISSING_ROLE),
    (lambda r: {**{k: v for k, v in r.items() if k != CAL_AIR}, ("Calgary", "Int  Airport"): R.AIRPORT}, RD.MISSING_ROLE),
    (lambda r: {**r, VAN_THUR: R.OTHER}, RD.ALIAS_ROLE_CONFLICT),
])
def test_role_map_defects_keep_pricing_closed(change, defect) -> None:
    changed = change(dict(ROLES))
    items = changed.items() if isinstance(changed, dict) else changed
    report = assess(roles(items))
    assert defect in report.role_defects and not report.roles_exact
    assert LB.BRANCH_ROLES_NOT_EXACT in report.blocking_reasons
    assert not report.pairs_valid and PD.ROLES_UNAVAILABLE in report.pair_defects   # pairs need exact roles
    if defect in (RD.MISSING_ROLE, RD.UNEXPECTED_KEY):
        assert RD.UNEXPECTED_KEY in report.role_defects or set(changed) < set(ROLES) or RD.MISSING_ROLE in report.role_defects


def test_role_map_unavailable_states() -> None:
    for status in (ST.NOT_APPROVED, ST.RECORD_UNAVAILABLE):
        report = assess(LocationRoleMap(status=status))
        assert LB.BRANCH_ROLE_AUTHORITY_UNAVAILABLE in report.blocking_reasons and not report.roles_exact
    with pytest.raises(ValueError):
        LocationRoleMap(status=ST.NOT_APPROVED, assignments=((CAL_AIR, R.AIRPORT),))
    with pytest.raises(ValueError):
        LocationRoleMap(status=ST.APPROVED, assignments=((CAL_AIR, R.AIRPORT),))     # no provenance
    unusable = expected_stream_contract_from_record(None)                          # no exhaustive contract
    assert RD.CONTRACT_UNAVAILABLE in assess(contract=unusable).role_defects


def test_roles_never_change_source_values() -> None:
    frame = cars_frame(*(("SYNTH-JOB-001", label, city) for city, label in SUPPLIED))
    before = frame.copy(deep=True)
    current_location_authority()
    keys = apply_location_policy(frame, POLICY)
    pd.testing.assert_frame_equal(frame, before)
    assert keys.source_keys.tolist() == list(SUPPLIED)


# ============================================================ comparison pairs


def test_project_pairs_are_exactly_the_three_canonical_within_city_pairs() -> None:
    report = current_location_authority()
    assert report.pair_set == comparison_pairs_from_record(load_current_decision_record())
    assert report.pairs_valid and report.pair_defects == ()
    assert tuple((p.airport, p.downtown) for p in report.effective_pairs) == PAIRS
    vancouver = [p for p in report.effective_pairs if p.airport == VAN_AIR]
    assert vancouver == [ComparisonPair(VAN_AIR, VAN_DOWN)]                        # canonical Downtown, once
    assert not any(VAN_THUR in (p.airport, p.downtown) for p in report.effective_pairs)


@pytest.mark.parametrize("extra, defects", [
    ((CAL_AIR, TOR_DOWN), {PD.CROSS_CITY}),                                        # cross-city
    ((CAL_AIR, CAL_AIR), {PD.SELF_PAIR, PD.ROLE_MISMATCH, PD.SAME_CANONICAL_LOCATION}),
    ((TOR_AIR, VAN_AIR), {PD.CROSS_CITY, PD.ROLE_MISMATCH}),                      # airport-to-airport
    ((VAN_AIR, VAN_AIR), {PD.SELF_PAIR}),
    ((CAL_DOWN, CAL_DOWN), {PD.SELF_PAIR}),
    ((CAL_DOWN, CAL_AIR), {PD.ROLE_MISMATCH, PD.DUPLICATE_PAIR}),                  # reversed duplicate
    ((CAL_AIR, CAL_DOWN), {PD.DUPLICATE_PAIR}),                                    # duplicate
    ((("Calgary", "SYNTH Branch"), CAL_DOWN), {PD.UNKNOWN_KEY, PD.ROLE_MISMATCH}),  # unknown key
    ((VAN_AIR, VAN_THUR), {PD.NON_CANONICAL_MEMBER, PD.DUPLICATE_PAIR}),           # second pair through Thurlow
    ((VAN_DOWN, VAN_THUR), {PD.ROLE_MISMATCH, PD.NON_CANONICAL_MEMBER, PD.SAME_CANONICAL_LOCATION}),
])
def test_invalid_pairs_are_rejected(extra, defects) -> None:
    report = assess(pair_set=pairs((*PAIRS, extra)))
    assert defects <= set(report.pair_defects), report.pair_defects
    assert not report.pairs_valid and report.effective_pairs == ()
    assert LB.COMPARISON_PAIRS_INVALID in report.blocking_reasons


def test_downtown_to_downtown_and_other_roles_are_rejected() -> None:
    report = assess(pair_set=pairs(((CAL_DOWN, CAL_DOWN),)))
    assert PD.ROLE_MISMATCH in report.pair_defects
    other = {**ROLES, CAL_DOWN: R.OTHER}
    report = assess(roles(other.items()), pairs(((CAL_AIR, CAL_DOWN),)))
    assert PD.ROLE_MISMATCH in report.pair_defects and not report.pairs_valid


def test_pairs_need_their_own_authority_and_an_identity_decision() -> None:
    # A complete role map cannot substitute for comparison-pair authority ...
    no_pairs = assess(pair_set=ComparisonPairSet(status=ST.NOT_APPROVED))
    assert no_pairs.roles_exact and LB.COMPARISON_PAIR_AUTHORITY_UNAVAILABLE in no_pairs.blocking_reasons
    # ... a pair list cannot substitute for the Vancouver identity decision ...
    undecided = assess(policy=UNDECIDED)
    assert PD.IDENTITY_UNRESOLVED in undecided.pair_defects
    assert LB.COMPARISON_PAIR_IDENTITY_UNRESOLVED in undecided.blocking_reasons and not undecided.pairs_valid
    # ... and an approved identity cannot substitute for the role map.
    no_roles = assess(role_map=LocationRoleMap(status=ST.NOT_APPROVED))
    assert LB.BRANCH_ROLE_AUTHORITY_UNAVAILABLE in no_roles.blocking_reasons and not no_roles.pairs_valid
    empty = assess(pair_set=pairs(()))
    assert PD.EMPTY in empty.pair_defects and not empty.pairs_valid


def test_canonicalization_never_crosses_city_boundaries() -> None:
    forged = dataclasses.replace(POLICY)
    object.__setattr__(forged, "canonical_location", CAL_DOWN)                     # bypassed construction
    report = assess(policy=forged)
    assert report.canonical(VAN_THUR) == VAN_THUR and report.canonical(VAN_DOWN) == VAN_DOWN   # refused, unmapped
    keys = apply_location_policy(cars_frame(("SYNTH-JOB-001", VAN_THUR[1], VAN_THUR[0])), forged)
    assert keys.canonicalization_refused and keys.analytical_keys.tolist() == [VAN_THUR]
    assert LocationPolicyScopeDefect.CANONICAL_LOCATION_CITY_MISMATCH in keys.scope_defects


def test_reports_cannot_be_forged() -> None:
    report = current_location_authority()
    with pytest.raises(ValueError):
        dataclasses.replace(report, pair_defects=())  if report.pair_defects else dataclasses.replace(
            report, effective_pairs=())
    with pytest.raises(ValueError):
        LocationAuthorityReport(role_map=roles(), pair_set=pairs(((CAL_AIR, TOR_DOWN),)), contract=CONTRACT,
                                policy=POLICY, role_defects=(), pair_defects=(), effective_pairs=(),
                                canonicalization_merges_streams=True)


# ============================================================== Vancouver policy


def test_project_policy_is_the_approved_alias_with_authority() -> None:
    assert POLICY == vancouver_policy_from_record(load_current_decision_record(), COV)
    assert POLICY.state is PS.CONFIRMED_ALIAS and (POLICY.first, POLICY.second) == (VAN_DOWN, VAN_THUR)
    assert POLICY.canonical_location == VAN_DOWN and POLICY.scope.is_valid
    assert POLICY.authority.reference == GOVERNANCE and "pricing-authorities-v7" in POLICY.authority.note
    assert dict(POLICY.alias_mapping) == {VAN_DOWN: VAN_DOWN, VAN_THUR: VAN_DOWN}


def test_both_raw_labels_map_to_canonical_downtown_and_stay_preserved() -> None:
    j, c = healthy_frames()
    before = c.copy(deep=True)
    keys = apply_location_policy(c, POLICY)
    pd.testing.assert_frame_equal(c, before)
    source, analytical = keys.source_keys.tolist(), keys.analytical_keys.tolist()
    assert VAN_THUR in source and VAN_THUR not in analytical and analytical.count(VAN_DOWN) == 2
    assert [a for s, a in zip(source, analytical) if s != VAN_THUR] == [s for s in source if s != VAN_THUR]
    report = assess_location_policy(POLICY, None, keys)
    assert report.locations_are_aliases and not report.locations_comparable_independently
    assert report.canonicalization_applied and report.governed_sources_present and report.blocking_reasons == ()


@pytest.mark.parametrize("missing", [VAN_DOWN, VAN_THUR])
def test_one_missing_raw_vancouver_stream_fails_coverage_completeness_and_the_alias(missing) -> None:
    j, c = healthy_frames(tuple(k for k in SUPPLIED if k != missing))
    completeness = completeness_of(j, c)
    assert CMP.EXPECTED_PAIRS_MISSING in completeness.blocking_reasons and not completeness.complete
    keys = apply_location_policy(c, POLICY)
    assert not keys.governed_sources_present                       # the other alias never stands in for it
    report = assess_location_policy(POLICY, None, keys)
    assert not report.location_policy_authority_sufficient
    assert PB.GOVERNED_SOURCE_STREAM_MISSING in report.blocking_reasons
    pricing = pricing_of(j, c, completeness)
    assert {PB.EXPECTED_SOURCE_STREAMS_MISSING, PB.GOVERNED_SOURCE_STREAM_MISSING} <= set(pricing.blocking_reasons)


@pytest.mark.parametrize("status", [CS.LIKELY_DUPLICATE_STREAMS, CS.LIKELY_DISTINCT_STREAMS,
                                    CS.COMPARISON_INCONCLUSIVE, CS.INSUFFICIENT_COMPARABLE_CAPTURES])
def test_behaviour_never_changes_the_policy_state(status) -> None:
    j, c = healthy_frames()
    keys = apply_location_policy(c, POLICY)
    report = assess_location_policy(POLICY, evidence(status), keys)
    assert report.state is PS.CONFIRMED_ALIAS and report.locations_are_aliases and report.blocking_reasons == ()
    unresolved = assess_location_policy(UNDECIDED, evidence(status), apply_location_policy(c, UNDECIDED))
    assert unresolved.state is PS.UNRESOLVED and PB.LOCATION_POLICY_UNRESOLVED in unresolved.blocking_reasons


@pytest.mark.parametrize("status", [CS.LOCATION_MAPPING_DEFECT, CS.CONFIRMED_DISTINCT_LOCATIONS])
def test_conflicting_authoritative_identity_blocks_the_alias(status) -> None:
    j, c = healthy_frames()
    report = assess_location_policy(POLICY, evidence(status), apply_location_policy(c, POLICY))
    assert report.location_policy_resolved and not report.location_policy_authority_sufficient
    assert PB.IDENTITY_EVIDENCE_CONFLICT in report.blocking_reasons and not report.canonicalization_permitted


def test_absent_or_proposed_authority_is_unresolved() -> None:
    for record in (None, load_decision_record(V3)):
        assert vancouver_policy_from_record(record, COV).state is PS.UNRESOLVED
    proposed = {}
    data = v5()
    for decision in (D.VALID_LOCATION_COMPARISON_PAIRS, D.VANCOUVER_LOCATION_IDENTITY):
        proposed = unapprove(data, decision)
    record = parse_decision_record(proposed)
    policy = vancouver_policy_from_record(record, COV)
    assert policy.state is PS.UNRESOLVED and policy.authority is None
    report = location_authority_from_record(record, CONTRACT, policy)
    assert LB.COMPARISON_PAIR_AUTHORITY_UNAVAILABLE in report.blocking_reasons and report.roles_exact


def test_invalid_canonical_location_fails_closed() -> None:
    for canonical in (CAL_DOWN, VAN_AIR, ("Vancouver", "SYNTH Branch")):
        forged = dataclasses.replace(POLICY)
        object.__setattr__(forged, "canonical_location", canonical)
        j, c = healthy_frames()
        keys = apply_location_policy(c, forged)
        assert keys.canonicalization_refused and keys.analytical_keys.tolist() == keys.source_keys.tolist()
        report = assess_location_policy(forged, None, keys)
        assert not report.location_policy_authority_sufficient and report.scope_defects


# ==================================================================== readiness


def test_former_blockers_clear_only_with_the_approved_record() -> None:
    j, c = healthy_frames()
    completeness = completeness_of(j, c)
    approved = pricing_of(j, c, completeness)
    cleared = {PB.BRANCH_ROLE_AUTHORITY_UNAVAILABLE, PB.COMPARISON_PAIR_AUTHORITY_UNAVAILABLE,
               PB.LOCATION_POLICY_UNRESOLVED}
    assert not cleared & set(approved.blocking_reasons) and approved.location_roles_and_pairs_ready
    assert approved.blocking_reasons == (PB.CANONICAL_OFFER_COMBINATION_UNRESOLVED,)   # stays explicit
    assert not approved.ready
    v3 = load_decision_record(V3)
    policy3 = vancouver_policy_from_record(v3, COV)
    gates = {"location_authority": location_authority_from_record(v3, CONTRACT, policy3)}
    blocked = assess_pricing_readiness(
        location_policy=assess_location_policy(policy3, None, apply_location_policy(c, policy3)),
        **(_project_gates(j, c, completeness) | gates))
    assert cleared <= set(blocked.blocking_reasons)


def _project_gates(j, c, completeness):  # type: ignore[no-untyped-def]
    from test_readiness import gates_for

    return gates_for(j, c, COV, completeness) | {"expected_stream_contract": CONTRACT,
                                                  "location_authority": current_location_authority()}


@pytest.mark.parametrize("gate, blocker", [
    ("roles", PB.BRANCH_ROLE_AUTHORITY_UNAVAILABLE),
    ("pairs", PB.COMPARISON_PAIR_AUTHORITY_UNAVAILABLE),
    ("identity", PB.LOCATION_POLICY_UNRESOLVED),
])
def test_each_location_gate_fails_independently(gate, blocker) -> None:
    j, c = healthy_frames()
    completeness = completeness_of(j, c)
    role_map = LocationRoleMap(status=ST.NOT_APPROVED) if gate == "roles" else roles()
    pair_set = ComparisonPairSet(status=ST.NOT_APPROVED) if gate == "pairs" else pairs()
    policy = UNDECIDED if gate == "identity" else POLICY
    authority = assess_location_authority(role_map, pair_set, CONTRACT, policy)
    pricing = assess_pricing_readiness(
        location_policy=assess_location_policy(policy, None, apply_location_policy(c, policy)),
        **(_project_gates(j, c, completeness) | {"location_authority": authority}))
    assert blocker in pricing.blocking_reasons and not pricing.ready


def test_authority_validated_under_another_policy_or_contract_is_rejected() -> None:
    j, c = healthy_frames()
    completeness = completeness_of(j, c)
    under_undecided = assess(policy=UNDECIDED)                   # pairs validated without the alias
    pricing = assess_pricing_readiness(
        location_policy=assess_location_policy(POLICY, None, apply_location_policy(c, POLICY)),
        **(_project_gates(j, c, completeness) | {"location_authority": under_undecided}))
    assert PB.LOCATION_AUTHORITY_POLICY_MISMATCH in pricing.blocking_reasons
    narrower = synthetic_contract(dataclasses.replace(COV, expected_locations=SUPPLIED[:4]))
    other = assess(contract=narrower)
    pricing = assess_pricing_readiness(
        location_policy=assess_location_policy(POLICY, None, apply_location_policy(c, POLICY)),
        **(_project_gates(j, c, completeness) | {"location_authority": other}))
    assert PB.LOCATION_AUTHORITY_CONTRACT_MISMATCH in pricing.blocking_reasons


def test_unrelated_blockers_remain_and_nothing_claims_readiness() -> None:
    j, c = healthy_frames()
    completeness = completeness_of(j, c)
    gates = _project_gates(j, c, completeness) | {"scheduled_coverage": None, "temporal_fields_trusted": False,
                                                    "completeness": None}
    pricing = assess_pricing_readiness(
        location_policy=assess_location_policy(POLICY, None, apply_location_policy(c, POLICY)), **gates)
    assert {PB.SCHEDULED_COVERAGE_ASSESSMENT_MISSING, PB.TEMPORAL_FIELDS_UNTRUSTED,
            PB.COMPLETENESS_UNAVAILABLE} <= set(pricing.blocking_reasons) and not pricing.ready
    assert not assess_pricing_readiness(location_policy=assess_location_policy(POLICY), **GATES).ready


def test_location_blocker_values_are_pricing_blocker_values() -> None:
    assert {b.value for b in LB} <= {b.value for b in PB}


# ======================================================= baseline and notebook


def test_baseline_reports_the_location_authority_in_aggregate() -> None:
    from test_expected_stream_contract import baseline_for

    from ql2_sixt_canada_analysis.pricing_baseline import render_baseline_markdown

    j, c = healthy_frames()
    baseline = baseline_for(j, c)
    summary = baseline.location_authority
    assert (summary.role_map_status, summary.roles_exact, summary.airport_streams, summary.downtown_streams,
            summary.other_streams) == ("approved", True, 3, 4, 0)
    assert (summary.comparison_pair_status, summary.comparison_pairs_valid, summary.approved_pair_count) == (
        "approved", True, 3)
    assert summary.keys == PAIRS and summary.stream == VAN_DOWN and summary.vancouver_policy_state == "confirmed_alias"
    assert ("location_role_map", "approved") in baseline.statuses and ("comparison_pairs", "approved") in baseline.statuses
    assert [h.stream for h in baseline.expected_stream_health] == list(SUPPLIED)     # both raw Vancouver streams
    markdown = render_baseline_markdown(baseline, commit="abc1234", date="2026-10-06")
    assert "vancouver / Vancouver Int Airport versus vancouver / Vancouver Downtown" in markdown
    assert "vancouver / Vancouver Thurlow" in markdown and "versus vancouver / Vancouver Thurlow" not in markdown
    assert "SYNTH" not in markdown and "**NOT PRICING READY**" in markdown
    json.dumps(baseline.to_dict())


def test_notebook_consumes_the_authority_without_duplicating_policy_logic() -> None:
    notebook = json.loads((ROOT / "notebooks" / "01_data_ingestion.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "location_authority = current_location_authority()" in code
    assert "location_authority=location_authority" in code
    assert "apply_location_policy(analysis_cars_df, VANCOUVER_LOCATION_POLICY)" in code
    for forbidden in ("Int Airport", "Thurlow", "AIRPORT", "CONFIRMED_ALIAS", "canonical_location="):
        assert forbidden not in code, forbidden


def test_documentation_describes_the_approved_identity() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    for phrase in ("CONFIRMED_ALIAS", "canonical", "vancouver / Vancouver Thurlow", "comparison pairs",
                   "branch_role_authority_unavailable", "canonical_offer_combination_unresolved",
                   "governed_source_stream_missing"):
        assert phrase in readme, phrase
    assert "Until then no Vancouver pricing conclusion" not in readme
    assert "The dataset is **not** pricing ready" in readme
