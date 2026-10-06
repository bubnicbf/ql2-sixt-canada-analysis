"""Tests for pricing-authority revision 2 (schema 2) and the job-linkage policy adapter.

The committed records are read as data; every negative case mutates a parsed
copy in memory. No source-level data is involved.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import re
import tomllib
from pathlib import Path

import pytest
from test_authority_decisions import _finish, approved_record, entry, fails

from ql2_sixt_canada_analysis import authority_decisions as ad
from ql2_sixt_canada_analysis.authority_decisions import (
    CURRENT_RECORD_PATH,
    JOB_IDENTIFIER_DECISIONS,
    LEGACY_DECIMAL_ZERO_REPAIR,
    OPAQUE_TEXT_IDENTIFIER_POLICY,
    AuthorityKind,
    AuthorityReference,
    DecisionId,
    DecisionStatus,
    load_decision_record,
    parse_decision_record,
    render_authority_request_checklist,
)
from ql2_sixt_canada_analysis.job_linkage import (
    JobLinkagePolicy,
    JobLinkagePolicyError,
    job_linkage_policy_from_record,
    load_job_linkage_policy,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V1, V2 = RECORD_DIR / "v1.toml", RECORD_DIR / "v2.toml"
GOVERNANCE = "docs/decisions/governance/job-identifier-governance-2026-10-06.md"
#: Revision 1 is history: its bytes must never change.
V1_SHA256 = "b13881b099885130e853207f872e62bfde9820f438e34859f3ddef0ebdbf1bd7"
D = DecisionId
JOB = set(JOB_IDENTIFIER_DECISIONS)


def v2() -> dict:
    return tomllib.loads(V2.read_text(encoding="utf-8"))


def unapprove(data: dict, decision: DecisionId) -> dict:
    item = entry(data, decision)
    item.update(status="PROPOSED", blocking_external_input=True)
    item.pop("authority", None), item.pop("resolution", None)
    return _finish(data)


def resolution(data: dict, decision: DecisionId) -> dict:
    return entry(data, decision)["resolution"]


# ---------------------------------------------------------------- revisions


def test_v1_is_unchanged_and_still_validates_under_schema_1() -> None:
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == V1_SHA256
    record = load_decision_record(V1)
    assert record.schema_version == 1 and record.counts()[DecisionStatus.PROPOSED] == 22


def test_v2_validates_under_schema_2_and_supersedes_v1() -> None:
    record = load_decision_record(V2)
    assert (record.schema_version, record.record_version, record.record_id) == (2, 2, "pricing-authorities-v2")
    assert record.supersedes == "pricing-authorities-v1" and record.created == dt.date(2026, 10, 6)
    assert re.fullmatch(r"[0-9a-f]{40}", record.source_commit)
    assert CURRENT_RECORD_PATH.as_posix() == "docs/decisions/pricing_authorities/v3.toml"   # superseded by v3
    assert 2 in ad.SUPPORTED_SCHEMA_VERSIONS and 1 in ad.SUPPORTED_SCHEMA_VERSIONS


def test_exactly_the_four_job_identifier_decisions_are_approved() -> None:
    record = load_decision_record(V2)
    counts = record.counts()
    assert (counts[DecisionStatus.APPROVED], counts[DecisionStatus.PROPOSED], counts[DecisionStatus.REJECTED]) == (4, 18, 0)
    assert {d.id for d in record.decisions if d.is_approved} == JOB
    for decision in record.decisions:
        if decision.id in JOB:
            assert not decision.blocking_external_input and decision.resolution is not None
        else:
            assert decision.blocking_external_input and decision.resolution is None and not decision.authority
    assert set(record.external_inputs) == set(DecisionId) - JOB and len(record.external_inputs) == 18
    raw = v2()
    assert raw["summary"] == {"proposed": 18, "approved": 4, "rejected": 0}
    assert not JOB & set(raw["external_inputs"])


def test_approved_decisions_carry_collection_owner_provenance_to_the_governance_reference() -> None:
    record = load_decision_record(V2)
    for decision_id in JOB_IDENTIFIER_DECISIONS:
        authority = record.decision(decision_id).authority
        assert len(authority) == 1 and authority[0].kind is AuthorityKind.COLLECTION_OWNER
        assert authority[0].reference == GOVERNANCE and authority[0].effective_date is None
    assert (ROOT / GOVERNANCE).is_file()


def test_governance_reference_records_only_supplied_provenance() -> None:
    text = (ROOT / GOVERNANCE).read_text(encoding="utf-8")
    for phrase in ("supplied by the repository owner", "collection-governance meeting", "2026-10-06",
                   "Meeting date | Not supplied", "Participants, organizations, tickets | Not supplied",
                   "opaque text identifier", "never be parsed", "separate derived linkage key",
                   "integer offer position", "must not be edited or replaced", "current historical"):
        assert phrase in text, phrase
    assert "@" not in text and not re.search(r"#\d|[A-Z]{2,}-\d+", text)    # no emails or ticket ids
    assert not re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\d{6,}", text)  # no source-like values
    assert "pricing-authorities-v2" in text


def test_v2_checklist_lists_job_decisions_as_resolved_and_asks_the_other_eighteen() -> None:
    checklist = render_authority_request_checklist(load_decision_record(V2))
    assert "(pricing-authorities-v2)" in checklist
    resolved, requests = checklist.split("## Resolved decisions")[1].split("\n## ", 1)
    assert all(f"`{d.value}`" in resolved and f"`{d.value}`" not in requests for d in JOB)
    assert len([line for line in requests.splitlines() if line.startswith("| `")]) == 18


def test_readme_documents_revision_2() -> None:
    readme = (RECORD_DIR / "README.md").read_text(encoding="utf-8")
    for phrase in ("v2.toml", "schema 2", "separately tested", "job_linkage"):
        assert phrase in readme, phrase


# ------------------------------------------------------------ authority rules


def test_approval_requires_attributable_authority() -> None:
    data = v2()
    entry(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE).pop("authority")
    assert "requires authority provenance" in fails(data)
    data = v2()
    entry(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE)["authority"][0]["kind"] = "RAW_DATA_OBSERVATION"
    assert "non-authoritative evidence cannot serve as authority" in fails(data)
    data = v2()
    entry(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE)["authority"][0]["kind"] = "BUSINESS_OWNER"
    assert "not a responsible role" in fails(data)
    data = v2()
    entry(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE)["authority"][0]["reference"] = " "
    assert "reference" in fails(data)


def test_raw_observations_alone_cannot_approve_a_decision() -> None:
    data = v2()
    item = entry(data, D.EXPECTED_STREAM_UNIVERSE)
    item.update(status="APPROVED", blocking_external_input=False)
    assert "requires authority provenance" in fails(data)
    item["authority"] = [{"kind": "RAW_DATA_OBSERVATION", "source": "SYNTH", "reference": "SYNTH"}]
    assert "non-authoritative evidence" in fails(data)


# ---------------------------------------------------- schema-specific shapes


def test_schema_1_and_schema_2_resolution_shapes_never_mix() -> None:
    assert parse_decision_record(_finish(approved_record())).schema_version == 1    # schema-1 booleans
    as_schema_2 = {**approved_record(), "schema_version": 2}
    assert "JOB_ID" in fails(as_schema_2)                                          # booleans rejected in 2
    as_schema_1 = {**v2(), "schema_version": 1}
    assert "JOB_ID" in fails(as_schema_1)                                          # v2 shapes rejected in 1


@pytest.mark.parametrize("field, value", [
    ("semantics", "SYNTH_NUMERIC"), ("trim_whitespace", True), ("case_fold", True), ("numeric_parsing", True),
    ("missing_invalid", False), ("whitespace_only_invalid", False), ("exact_nonblank_match_preserved", False),
    ("unresolved_representations", "SYNTH_BEST_EFFORT"), ("trim_whitespace", 0),
])
def test_opaque_text_policy_contradictions_fail(field: str, value: object) -> None:
    data = v2()
    resolution(data, D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS)[field] = value
    message = fails(data)
    assert "contradicts" in message or "unsupported value" in message
    assert "SYNTH" not in message                                                # values never echoed


@pytest.mark.parametrize("field, value", [
    ("pattern", "ANY_TRAILING_ZERO"), ("detail_field", "jobs.job_id"), ("parent_field", "cars.job_id"),
    ("match_order", "REPAIR_FIRST"), ("parent_match", "FIRST_MATCH"), ("cause", "UNKNOWN"),
    ("offer_position", "FLOAT"),
])
def test_legacy_repair_is_only_the_narrow_definition(field: str, value: str) -> None:
    data = v2()
    resolution(data, D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE)[field] = value
    assert "contradicts" in fails(data)


def test_missing_or_extra_policy_fields_fail() -> None:
    data = v2()
    resolution(data, D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS).pop("case_fold")
    assert "required fields" in fails(data)
    data = v2()
    resolution(data, D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE)["synth_extra"] = True
    assert "required fields" in fails(data)


def test_cross_decision_contradictions_fail() -> None:
    data = v2()
    resolution(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE)["leading_zeros_significant"] = False
    assert "significant leading zeros" in fails(data)
    data = v2()
    resolution(data, D.JOB_ID_RAW_AND_LINKAGE_PRESERVATION)["preserve_raw_identifier"] = False
    assert "preserved raw value" in fails(data)
    data = unapprove(v2(), D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS)
    resolution(data, D.JOB_ID_RAW_AND_LINKAGE_PRESERVATION)["separate_linkage_key"] = False
    assert "separate linkage key" in fails(data)


def test_not_equivalent_is_a_valid_alternative_answer() -> None:
    data = v2()
    entry(data, D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE)["resolution"] = {"equivalence": "NOT_EQUIVALENT"}
    record = parse_decision_record(data)
    policy = job_linkage_policy_from_record(record)
    assert policy is not None and not policy.legacy_decimal_zero_repair and not policy.legacy_offer_position_repair
    entry(data, D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE)["resolution"] = {"equivalence": "NOT_EQUIVALENT",
                                                                    "pattern": LEGACY_DECIMAL_ZERO_REPAIR["pattern"]}
    assert "required fields" in fails(data)


# --------------------------------------------------------------- policy adapter


def test_v2_produces_the_expected_typed_policy() -> None:
    policy = job_linkage_policy_from_record(load_decision_record(V2))
    assert isinstance(policy, JobLinkagePolicy)
    assert (policy.record_id, policy.semantics) == ("pricing-authorities-v2", "OPAQUE_TEXT")
    assert policy.legacy_decimal_zero_repair is True and policy.legacy_offer_position_repair is True
    assert [a.kind for a in policy.authority] == [AuthorityKind.COLLECTION_OWNER]
    assert policy.authority[0].reference == GOVERNANCE
    current = load_job_linkage_policy()                                         # the committed current record (v3)
    assert current is not None and current.record_id == "pricing-authorities-v3"
    assert {f: getattr(current, f) for f in ("authority", "semantics", "legacy_decimal_zero_repair",
                                             "legacy_offer_position_repair")} == {
        f: getattr(policy, f) for f in ("authority", "semantics", "legacy_decimal_zero_repair",
                                        "legacy_offer_position_repair")}            # v3 preserves the four
    assert dict(OPAQUE_TEXT_IDENTIFIER_POLICY)["numeric_parsing"] is False


def test_v1_produces_no_policy() -> None:
    assert job_linkage_policy_from_record(load_decision_record(V1)) is None
    assert load_job_linkage_policy(V1) is None


@pytest.mark.parametrize("decision", JOB_IDENTIFIER_DECISIONS)
def test_partial_approval_produces_no_policy(decision: DecisionId) -> None:
    record = parse_decision_record(unapprove(v2(), decision))
    assert job_linkage_policy_from_record(record) is None


def test_rejected_decision_produces_no_policy() -> None:
    data = v2()
    item = entry(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE)
    item.pop("resolution")
    item.update(status="REJECTED", rejected="SYNTH rejection")
    record = parse_decision_record(_finish(data))
    assert job_linkage_policy_from_record(record) is None


def test_contradictory_but_individually_valid_answers_produce_no_policy() -> None:
    data = unapprove(v2(), D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS)
    resolution(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE)["leading_zeros_significant"] = False
    record = parse_decision_record(data)                                        # valid record...
    assert job_linkage_policy_from_record(record) is None                       # ...but no policy


def test_malformed_inputs_produce_no_policy(tmp_path: Path) -> None:
    assert job_linkage_policy_from_record(None) is None
    assert job_linkage_policy_from_record(v2()) is None                         # parsed data, not a record
    schema_1 = parse_decision_record(_finish(approved_record()))                # every decision approved
    assert job_linkage_policy_from_record(schema_1) is None                     # booleans cannot express it
    broken = tmp_path / "v2.toml"
    broken.write_text(V2.read_text(encoding="utf-8").replace('case_fold = false', 'case_fold = true'),
                      encoding="utf-8")
    assert load_job_linkage_policy(broken) is None
    assert load_job_linkage_policy(tmp_path / "missing.toml") is None


def test_policy_construction_requires_authority_and_supported_semantics() -> None:
    owner = AuthorityReference(kind=AuthorityKind.COLLECTION_OWNER, source="SYNTH", reference="SYNTH")
    business = AuthorityReference(kind=AuthorityKind.BUSINESS_OWNER, source="SYNTH", reference="SYNTH")
    ok = dict(record_id="SYNTH", authority=(owner,), legacy_decimal_zero_repair=True,
              legacy_offer_position_repair=True)
    JobLinkagePolicy(**ok)
    for change in (dict(authority=()), dict(authority=(business,)), dict(authority=["SYNTH"]),
                   dict(record_id=" "), dict(legacy_decimal_zero_repair=1), dict(semantics="NUMERIC")):
        with pytest.raises(JobLinkagePolicyError):
            JobLinkagePolicy(**(ok | change))
    with pytest.raises(TypeError):
        JobLinkagePolicy()  # type: ignore[call-arg]                             # no permissive default


def test_record_input_is_never_modified() -> None:
    data = v2()
    snapshot = copy.deepcopy(data)
    job_linkage_policy_from_record(parse_decision_record(data))
    assert data == snapshot


# ------------------------------------------------- repository-local references


def _governance_root(tmp_path: Path, *, with_document: bool = True) -> Path:
    root = tmp_path / "repo"
    area = root / "docs" / "decisions" / "governance"
    area.mkdir(parents=True)
    if with_document:
        (area / Path(GOVERNANCE).name).write_text("# synthetic governance reference\n", encoding="utf-8")
    return root


def test_approved_decision_with_an_existing_local_reference_passes(tmp_path: Path) -> None:
    assert parse_decision_record(v2(), repository_root=ROOT).counts()[DecisionStatus.APPROVED] == 4
    assert parse_decision_record(v2(), repository_root=_governance_root(tmp_path)).record_id == "pricing-authorities-v2"
    assert load_decision_record(V2, repository_root=ROOT).record_id == "pricing-authorities-v2"


def test_approved_decision_with_a_missing_local_reference_fails_closed(tmp_path: Path) -> None:
    root = _governance_root(tmp_path, with_document=False)
    with pytest.raises(ad.DecisionRecordError) as info:
        parse_decision_record(v2(), repository_root=root)
    assert "missing" in str(info.value) and "governance" not in str(info.value).split(":")[0]
    with pytest.raises(ad.DecisionRecordError):
        load_decision_record(V2, repository_root=root)          # the production loader applies the same rule


@pytest.mark.parametrize("reference", [
    "docs/decisions/governance/../../../README.md", "../docs/decisions/governance/x.md", "/etc/hosts",
    "README.md", "docs/decisions/pricing_authorities/v1.toml", "docs/decisions/governance/sub/x.md",
    "docs\\decisions\\governance\\x.md", "docs/decisions/governance/.hidden.md", "SYNTH-DECISION-REF",
])
def test_references_outside_the_governance_area_are_rejected(tmp_path: Path, reference: str) -> None:
    data = v2()
    entry(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE)["authority"][0]["reference"] = reference
    message = fails(data)
    assert "governance document" in message and reference not in message


def test_symlink_escaping_the_governance_area_is_rejected(tmp_path: Path) -> None:
    root = _governance_root(tmp_path, with_document=False)
    outside = tmp_path / "outside.md"
    outside.write_text("synthetic\n", encoding="utf-8")
    (root / GOVERNANCE).symlink_to(outside)
    with pytest.raises(ad.DecisionRecordError, match="outside the governance area|missing"):
        parse_decision_record(v2(), repository_root=root)


def test_rejected_decisions_also_need_a_local_reference_and_proposed_gain_nothing(tmp_path: Path) -> None:
    data = v2()
    item = entry(data, D.JOB_ID_LEADING_ZERO_SIGNIFICANCE)
    item.pop("resolution")
    item.update(status="REJECTED", rejected="SYNTH rejection")
    data = _finish(data)
    parse_decision_record(data)                                                  # existing reference: valid
    with pytest.raises(ad.DecisionRecordError, match="missing"):
        parse_decision_record(data, repository_root=_governance_root(tmp_path, with_document=False))
    record = load_decision_record(V2)
    for decision in record.decisions:
        if decision.status is DecisionStatus.PROPOSED:
            assert not decision.authority and record.approved_resolution(decision.id) is None


def test_schema_1_records_are_unaffected(tmp_path: Path) -> None:
    root = _governance_root(tmp_path, with_document=False)
    assert load_decision_record(V1, repository_root=root).schema_version == 1
    assert parse_decision_record(_finish(approved_record()), repository_root=root).schema_version == 1
