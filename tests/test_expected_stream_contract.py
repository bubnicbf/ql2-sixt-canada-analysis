"""Approved exhaustive expected-stream contract (pricing-authorities-v3).

The committed records are read as data; every negative case mutates a parsed
copy in memory or uses a temporary repository root. Frames are fabricated
(``SYNTH-JOB-*``); the only stream values are the approved keys, which are
repository governance configuration, never source observations.
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
from test_completeness import cars as cars_frame, jobs as jobs_frame, reconcile
from test_readiness import DISTINCT, GATES, gates_for

from ql2_sixt_canada_analysis import authority_decisions as ad
from ql2_sixt_canada_analysis.authority_decisions import (
    CURRENT_RECORD_PATH,
    JOB_IDENTIFIER_DECISIONS,
    AuthorityKind,
    DecisionId,
    DecisionRecordError,
    DecisionStatus,
    load_current_decision_record,
    load_decision_record,
    parse_decision_record,
    render_authority_request_checklist,
)
from ql2_sixt_canada_analysis.city_integrity import assess_city_integrity
from ql2_sixt_canada_analysis.coverage import assess_expected_location_coverage, spelling_variant_keys
from ql2_sixt_canada_analysis.expected_stream_contract import (
    EXPECTED_STREAM_DECISIONS,
    ExpectedStreamAuthorityStatus as AS,
    ExpectedStreamContract,
    ExpectedStreamContractBlocker as CB,
    current_expected_stream_contract,
    expected_stream_contract_from_record,
    resolve_expected_stream_contract,
)
from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.location_authority import current_location_authority
from ql2_sixt_canada_analysis.pricing_baseline import build_pricing_baseline, render_baseline_markdown
from ql2_sixt_canada_analysis.readiness import (
    CompletenessBlocker as CMP,
    PricingBlocker as PB,
    apply_location_policy,
    assess_completeness,
    assess_location_policy,
    assess_pricing_readiness,
)
from ql2_sixt_canada_analysis.schemas import (
    COMPARED_LOCATION_STREAMS,
    EXPECTED_LOCATION_COVERAGE,
    INVESTIGATED_LOCATION_STREAM,
    JOB_DETAIL_RELATIONSHIP as REL,
    SOURCE_STREAM_COVERAGE_TEMPLATE,
    VANCOUVER_LOCATION_POLICY,
    LocationCoverageMode,
    LocationPolicyState,
)
from ql2_sixt_canada_analysis.stability import assess_vehicle_attribute_stability
from ql2_sixt_canada_analysis.streams import (
    ExpectedLocationStreamsReport,
    LocationStreamStatus,
    assess_expected_location_streams,
    investigate_location_stream,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V1, V2, V3 = (RECORD_DIR / f"v{n}.toml" for n in (1, 2, 3))
GOVERNANCE = "docs/decisions/governance/expected-stream-governance-2026-10-06.md"
JOB_GOVERNANCE = "docs/decisions/governance/job-identifier-governance-2026-10-06.md"
#: Committed history: these bytes must never change.
V1_SHA256 = "b13881b099885130e853207f872e62bfde9820f438e34859f3ddef0ebdbf1bd7"
V2_SHA256 = "899c20e289d932868b1d430b8fd70edca6ebb2900ca963c4c040cc6f5840cb4c"
#: The seven pairs exactly as supplied in the written governance decision (test oracle).
SUPPLIED = (("Calgary", "Downtown"), ("Calgary", "Int Airport"), ("Toronto", "Downtown"),
            ("Toronto", "Int Airport"), ("Vancouver", "Downtown"), ("Vancouver", "Int Airport"),
            ("Vancouver", "Thurlow"))
D = DecisionId
COV = EXPECTED_LOCATION_COVERAGE
CITY_COL, LABEL_COL = COV.location_columns


def v3() -> dict:
    return tomllib.loads(V3.read_text(encoding="utf-8"))


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


def with_streams(streams: list, decisions=EXPECTED_STREAM_DECISIONS) -> dict:  # type: ignore[no-untyped-def]
    data = v3()
    for decision in decisions:
        entry(data, decision)["resolution"]["streams"] = copy.deepcopy(streams)
    return data


# ===================================================== authority record (v3)


def test_v1_and_v2_are_unchanged_and_still_valid() -> None:
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == V1_SHA256
    assert hashlib.sha256(V2.read_bytes()).hexdigest() == V2_SHA256
    assert load_decision_record(V1).counts()[DecisionStatus.APPROVED] == 0
    assert load_decision_record(V2).counts()[DecisionStatus.APPROVED] == 4


def test_v3_is_a_valid_revision_superseding_v2_and_carried_into_v4() -> None:
    record = load_decision_record(V3)
    assert (record.schema_version, record.record_version, record.record_id) == (2, 3, "pricing-authorities-v3")
    assert record.supersedes == "pricing-authorities-v2" and CURRENT_RECORD_PATH.name == "v4.toml"
    current = load_current_decision_record()                       # v4 keeps both expected-stream approvals
    for decision in EXPECTED_STREAM_DECISIONS:
        assert current.decision(decision).resolution == record.decision(decision).resolution
        assert current.decision(decision).authority == record.decision(decision).authority


def test_v3_approves_exactly_the_job_and_expected_stream_decisions() -> None:
    record = load_decision_record(V3)
    approved = {d.id for d in record.decisions if d.is_approved}
    assert approved == set(JOB_IDENTIFIER_DECISIONS) | set(EXPECTED_STREAM_DECISIONS)
    counts = record.counts()
    assert (counts[DecisionStatus.APPROVED], counts[DecisionStatus.PROPOSED], counts[DecisionStatus.REJECTED]) == (6, 16, 0)
    for decision in record.decisions:
        if decision.id not in approved:
            assert decision.status is DecisionStatus.PROPOSED and decision.blocking_external_input
            assert decision.resolution is None and not decision.authority
    assert set(record.external_inputs) == set(DecisionId) - approved
    # The four job-identifier approvals are carried over unchanged from v2.
    v2 = load_decision_record(V2)
    for decision in JOB_IDENTIFIER_DECISIONS:
        assert record.decision(decision).resolution == v2.decision(decision).resolution
        assert record.decision(decision).authority == v2.decision(decision).authority


def test_v3_universe_is_exactly_the_seven_supplied_pairs_exhaustive_and_spelled_identically() -> None:
    record = load_decision_record(V3)
    universe = record.approved_resolution(D.EXPECTED_STREAM_UNIVERSE)
    spelling = record.approved_resolution(D.EXPECTED_STREAM_SOURCE_SPELLING)
    keys = tuple(tuple(k) for k in universe["streams"])
    assert universe["mode"] == "EXHAUSTIVE" and keys == SUPPLIED and len(set(keys)) == 7
    assert tuple(tuple(k) for k in spelling["streams"]) == SUPPLIED


def test_every_approval_names_its_required_roles_and_an_existing_governance_reference() -> None:
    record = load_decision_record(V3)
    universe = record.decision(D.EXPECTED_STREAM_UNIVERSE).authority
    assert {a.kind for a in universe} == {AuthorityKind.COLLECTION_OWNER, AuthorityKind.BUSINESS_OWNER}
    spelling = record.decision(D.EXPECTED_STREAM_SOURCE_SPELLING).authority
    assert [a.kind for a in spelling] == [AuthorityKind.COLLECTION_OWNER]
    for decision in record.decisions:
        if decision.is_approved:
            assert decision.authority
            for authority in decision.authority:
                assert (ROOT / authority.reference).is_file()
                assert authority.reference == (GOVERNANCE if decision.id in EXPECTED_STREAM_DECISIONS
                                               else JOB_GOVERNANCE)


def test_v3_presents_no_observation_as_authority_and_holds_no_source_values() -> None:
    record = load_decision_record(V3)
    for decision in EXPECTED_STREAM_DECISIONS:
        assert record.decision(decision).evidence == ()          # approved from the written decision only
    text = V3.read_text(encoding="utf-8")
    assert not re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", text)
    assert not re.search(r"\d{6,}", text.replace(record.source_commit, "")
                         .replace("f9b1229da91a8bc05260e2b2458cb38312cc0697", "")
                         .replace("8d62bada0e0bdef279b0645848b3af9088a28d71", ""))


def test_governance_document_records_the_supplied_decision_only() -> None:
    raw = (ROOT / GOVERNANCE).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in ("supplied by the repository owner", "2026-10-06", "EXHAUSTIVE",
                   "No other scheduled stream should exist", "Exact source spelling is authoritative",
                   "Source-key comparison is", "Raw source values must be preserved",
                   "Display labels are separate from source keys", "separate approved location-identity policy",
                   "current analyzed dataset", "until it is replaced by a new versioned authority decision",
                   "requires a new authority-record version", "VANCOUVER_LOCATION_IDENTITY",
                   "collection owner and business owner (joint)", "Not supplied", "pricing-authorities-v3"):
        assert phrase in text, phrase
    for city, location in SUPPLIED:
        assert f"| `{city}` | `{location}` |" in raw
    for unresolved in ("location roles", "Vancouver location identity", "schedule periods", "schedule exceptions",
                       "temporal rules", "reporting-day rules", "rental-date rules"):
        assert unresolved in text, unresolved
    assert "@" not in text and not re.search(r"#\d|[A-Z]{2,}-\d+", text)        # no emails or ticket ids
    assert not re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\d{6,}", text)      # no source-like values


def test_missing_or_broken_governance_reference_fails_validation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    area = root / "docs" / "decisions" / "governance"
    area.mkdir(parents=True)
    (area / Path(JOB_GOVERNANCE).name).write_text("# synthetic\n", encoding="utf-8")   # only the job reference
    with pytest.raises(DecisionRecordError, match="missing"):
        load_decision_record(V3, repository_root=root)
    (area / Path(GOVERNANCE).name).write_text("# synthetic\n", encoding="utf-8")
    assert load_decision_record(V3, repository_root=root).record_id == "pricing-authorities-v3"
    data = v3()
    entry(data, D.EXPECTED_STREAM_UNIVERSE)["authority"][1]["reference"] = "docs/decisions/governance/synth-missing.md"
    assert "missing" in fails(data)
    data = v3()
    entry(data, D.EXPECTED_STREAM_SOURCE_SPELLING)["authority"][0]["reference"] = "../" + GOVERNANCE
    assert "governance document" in fails(data)
    # A missing reference makes the current record unavailable, which fails closed.
    assert load_current_decision_record(repository_root=root.parent / "absent") is None
    assert resolve_expected_stream_contract(repository_root=root.parent / "absent").status is AS.RECORD_UNAVAILABLE


@pytest.mark.parametrize("streams, needle", [
    ([*map(list, SUPPLIED), list(SUPPLIED[0])], "duplicate"),                       # duplicate pair
    ([["Calgary", ""], *map(list, SUPPLIED[1:])], "non-blank"),                     # blank component
    ([["Calgary", "   "], *map(list, SUPPLIED[1:])], "non-blank"),
    ([[" Calgary", "Downtown"], *map(list, SUPPLIED[1:])], "non-blank"),            # padded component
    ([["Calgary"], *map(list, SUPPLIED[1:])], "[city, location]"),                  # wrong shape
    ([["Calgary", "Downtown", "SYNTH"], *map(list, SUPPLIED[1:])], "[city, location]"),
    (["Calgary Downtown", *map(list, SUPPLIED[1:])], "[city, location]"),
    ([[1, 2], *map(list, SUPPLIED[1:])], "[city, location]"),
    ([], "non-empty"),
])
def test_malformed_universe_entries_are_rejected(streams: list, needle: str) -> None:
    assert needle in fails(with_streams(streams))


def test_spellings_must_agree_with_the_universe_exactly() -> None:
    data = v3()
    spelled = entry(data, D.EXPECTED_STREAM_SOURCE_SPELLING)["resolution"]["streams"]
    spelled[0] = ["calgary", "Downtown"]                                             # case-only difference
    assert "cover the approved stream universe exactly" in fails(data)
    data = v3()
    entry(data, D.EXPECTED_STREAM_SOURCE_SPELLING)["resolution"]["streams"].pop()   # one spelling missing
    assert "cover the approved stream universe exactly" in fails(data)


def test_an_approved_universe_without_approved_spellings_is_unusable() -> None:
    record = parse_decision_record(unapprove(v3(), D.EXPECTED_STREAM_SOURCE_SPELLING))
    contract = expected_stream_contract_from_record(record)
    assert contract.status is AS.NOT_APPROVED and not contract.coverage.is_configured and not contract.usable
    assert contract.blocking_reasons == (CB.EXPECTED_STREAM_AUTHORITY_UNAVAILABLE,
                                         CB.EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE)
    with pytest.raises(DecisionRecordError):                                      # spellings need the universe
        parse_decision_record(unapprove(v3(), D.EXPECTED_STREAM_UNIVERSE))


def test_a_minimum_required_universe_is_approved_but_not_exhaustive() -> None:
    data = v3()
    entry(data, D.EXPECTED_STREAM_UNIVERSE)["resolution"]["mode"] = "MINIMUM_REQUIRED"
    contract = expected_stream_contract_from_record(parse_decision_record(finish(data)))
    assert contract.status is AS.APPROVED and contract.mode is LocationCoverageMode.MINIMUM_REQUIRED
    assert contract.blocking_reasons == (CB.EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE,)


def test_checklist_is_regenerated_from_v3() -> None:
    record = load_decision_record(V3)
    checklist = render_authority_request_checklist(record)       # the committed file follows v4 (test_location_authority)
    assert "(pricing-authorities-v3)" in checklist
    resolved, requests = checklist.split("## Resolved decisions")[1].split("\n## ", 1)
    for decision in (*JOB_IDENTIFIER_DECISIONS, *EXPECTED_STREAM_DECISIONS):
        assert f"`{decision.value}`" in resolved and f"`{decision.value}`" not in requests
    assert GOVERNANCE in resolved
    assert len([line for line in requests.splitlines() if line.startswith("| `")]) == 16


def test_record_readme_identifies_v3_as_current() -> None:
    readme = (RECORD_DIR / "README.md").read_text(encoding="utf-8")
    for phrase in ("v3.toml", "current revision", "EXHAUSTIVE", "expected-stream-governance-2026-10-06.md",
                   "new revision"):
        assert phrase in readme, phrase


# ============================================== single resolution of the contract


def test_project_coverage_is_the_one_resolution_of_the_current_record() -> None:
    contract = current_expected_stream_contract()
    assert contract.status is AS.APPROVED and contract.usable and contract.blocking_reasons == ()
    assert contract.coverage is EXPECTED_LOCATION_COVERAGE                  # one object, no second list
    assert contract.expected_keys == SUPPLIED and contract.exhaustive
    assert contract.record_id == CURRENT_RECORD_PATH.stem.replace("v", "pricing-authorities-v")
    assert contract.references == (GOVERNANCE,)
    assert COV.mode is LocationCoverageMode.EXHAUSTIVE and COV.aliases == {}
    assert INVESTIGATED_LOCATION_STREAM in COV.expected_locations
    assert set(COMPARED_LOCATION_STREAMS) <= set(COV.expected_locations)  # Downtown and Thurlow stay separate
    assert VANCOUVER_LOCATION_POLICY.coverage is COV


def test_no_source_module_hard_codes_the_universe() -> None:
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "Int Airport" not in text, path.name                       # only the record names those streams
        assert "Toronto" not in text, path.name


def test_contract_without_approval_is_unconfigured_and_fails_closed() -> None:
    for record in (None, load_decision_record(V1), load_decision_record(V2)):
        contract = expected_stream_contract_from_record(record)
        assert not contract.coverage.is_configured and contract.expected_keys == ()
        assert contract.blocking_reasons == (CB.EXPECTED_STREAM_AUTHORITY_UNAVAILABLE,
                                             CB.EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE)
    assert expected_stream_contract_from_record(None).status is AS.RECORD_UNAVAILABLE
    with pytest.raises(ValueError):
        expected_stream_contract_from_record(load_decision_record(V3), template=COV)   # template must be empty
    with pytest.raises(ValueError):                                       # cannot claim approval without one
        ExpectedStreamContract(status=AS.APPROVED, coverage=SOURCE_STREAM_COVERAGE_TEMPLATE, record_id="SYNTH")
    with pytest.raises(ValueError):
        ExpectedStreamContract(status=AS.APPROVED, coverage=COV, record_id="SYNTH")   # no approvals or references


def test_contract_blocker_values_are_pricing_blocker_values() -> None:
    assert {b.value for b in CB} <= {b.value for b in PB}


# ======================================================== exact source matching


def frame_for(keys) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """One synthetic detail row per key (one job)."""
    return cars_frame(*(("SYNTH-JOB-001", label, city) for city, label in keys))


def test_all_seven_exact_pairs_cover_the_contract() -> None:
    report = assess_expected_location_coverage(frame_for(SUPPLIED), COV)
    assert report.is_valid and report.covered_expected_location_count == 7
    assert (report.unexpected_location_count, report.spelling_variant_location_count) == (0, 0)


@pytest.mark.parametrize("index, variant", [
    (0, ("calgary", "Downtown")),             # case-only city difference
    (0, ("CALGARY", "Downtown")),
    (0, ("Calgary", "downtown")),             # case-only location difference
    (0, (" Calgary", "Downtown")),            # leading whitespace
    (0, ("Calgary ", "Downtown")),            # trailing whitespace
    (1, ("Calgary", "Int  Airport")),         # doubled internal whitespace
    (1, ("Calgary", "Int. Airport")),         # punctuation
    (1, ("Calgary", "Int-Airport")),
    (6, ("Vancouver", "Thurlow ")),           # trailing whitespace on the location
    (4, ("Downtown", "Vancouver")),           # component order
])
def test_spelling_variants_never_cover_and_are_reported(index: int, variant: tuple[str, str]) -> None:
    keys = [*SUPPLIED[:index], variant, *SUPPLIED[index + 1:]]
    frame = frame_for(keys)
    before = frame.copy(deep=True)
    report = assess_expected_location_coverage(frame, COV)
    assert not report.is_valid and report.missing_expected_locations == (SUPPLIED[index],)
    assert (report.unexpected_location_count, report.spelling_variant_location_count) == (1, 1)
    assert {"missing_expected_location", "unexpected_location", "source_spelling_mismatch"} <= set(report.violations)
    pd.testing.assert_frame_equal(frame, before)                               # raw values unchanged
    stream = investigate_location_stream(*frame_inputs(frame), SUPPLIED[index], coverage=COV)
    assert not stream.is_healthy and stream.present_after_cleaning is False


def frame_inputs(cars: pd.DataFrame):  # type: ignore[no-untyped-def]
    jobs = jobs_frame(("SYNTH-JOB-001", len(cars), len(cars), cars[CITY_COL].iloc[0]))
    return jobs, cars


def test_display_labels_and_unresolved_aliases_establish_no_coverage() -> None:
    display = [(city, f"{city} {label}") for city, label in SUPPLIED]          # display-style labels
    report = assess_expected_location_coverage(frame_for(display), COV)
    assert report.covered_expected_location_count == 0 and report.missing_expected_location_count == 7
    assert report.unexpected_location_count == 7 and not report.is_valid
    # The approved alias is analytical only: source keys are kept and coverage still needs both raw streams.
    frame = frame_for(SUPPLIED)
    keys = apply_location_policy(frame, VANCOUVER_LOCATION_POLICY)
    assert keys.alias_mapping_applied and keys.source_keys.tolist() == list(SUPPLIED)
    assert COV.aliases == {}                                                  # never a coverage alias
    assert assess_expected_location_coverage(frame, COV).is_valid
    without_thurlow = frame_for(SUPPLIED[:-1])
    assert assess_expected_location_coverage(without_thurlow, COV).missing_expected_locations == (SUPPLIED[-1],)


def test_spelling_variant_detection_is_diagnostic_only() -> None:
    observed = [("calgary", "Calgary Downtown"), ("Calgary", "Downtown"), ("Toronto", "SYNTH Branch")]
    assert spelling_variant_keys(observed, SUPPLIED) == [("calgary", "Calgary Downtown")]
    assert observed[0] == ("calgary", "Calgary Downtown")                     # nothing rewritten


# ====================================================== exhaustiveness & completeness

CITIES = tuple(dict.fromkeys(city for city, _ in SUPPLIED))
JOB_OF = {city: f"SYNTH-JOB-00{i}" for i, city in enumerate(CITIES, start=1)}


def healthy_frames(keys=SUPPLIED, extra_jobs=()):  # type: ignore[no-untyped-def]
    """One job per city carrying a detail row for each of ``keys`` in that city (declared counts match)."""
    rows = [(JOB_OF[city] if city in JOB_OF else "SYNTH-JOB-009", label, city) for city, label in keys]
    counts = {job: sum(1 for j, _, _ in rows if j == job) for job in JOB_OF.values()}
    job_rows = [(job, counts[job], counts[job], city) for city, job in JOB_OF.items()]
    return jobs_frame(*job_rows, *extra_jobs), cars_frame(*rows)


def completeness_of(j, c, streams=None):  # type: ignore[no-untyped-def]
    return assess_completeness(
        datasets=RawDatasets(jobs=j, cars=c, complete_source=True),
        coverage=assess_expected_location_coverage(c, COV),
        streams=streams if streams is not None else assess_expected_location_streams(j, c, coverage=COV),
        reconciliation=reconcile(j, c), city_integrity=assess_city_integrity(j, c, coverage=COV),
        expected_coverage=COV)


def pricing_of(j, c, report, contract=None, policy=None):  # type: ignore[no-untyped-def]
    """Pricing with the project contract, location authority and identity policy (applied to these rows)."""
    policy = policy if policy is not None else VANCOUVER_LOCATION_POLICY
    gates = gates_for(j, c, COV, report) | {
        "expected_stream_contract": contract if contract is not None else current_expected_stream_contract(),
        "location_authority": current_location_authority()}
    return assess_pricing_readiness(
        location_policy=assess_location_policy(policy, None, apply_location_policy(c, policy)), **gates)


def test_exactly_one_healthy_report_per_approved_stream_is_complete():
    j, c = healthy_frames()
    streams = assess_expected_location_streams(j, c, coverage=COV)
    assert streams.expected_stream_count == streams.assessed_stream_count == 7
    assert [r.target for r in streams.results] == list(SUPPLIED) and streams.assessed_exactly_once
    assert streams.all_expected_streams_healthy and streams.is_valid
    report = completeness_of(j, c, streams)
    assert report.complete and report.expected_streams is streams
    pricing = pricing_of(j, c, report)
    # Every gate under test passes; the aliased Vancouver streams' offer combination stays explicitly unresolved.
    assert pricing.blocking_reasons == (PB.CANONICAL_OFFER_COMBINATION_UNRESOLVED,)
    assert pricing.expected_stream_contract_usable and pricing.location_roles_and_pairs_ready


def test_one_missing_expected_stream_fails_completeness_and_pricing():
    j, c = healthy_frames(SUPPLIED[:-1])
    report = completeness_of(j, c)
    assert {CMP.EXPECTED_PAIRS_MISSING, CMP.STREAM_UNHEALTHY} <= set(report.blocking_reasons)
    pricing = pricing_of(j, c, report)
    assert {PB.DATA_INCOMPLETE, PB.EXPECTED_STREAMS_NOT_PROVEN, PB.EXPECTED_SOURCE_STREAMS_MISSING} <= set(
        pricing.blocking_reasons) and not pricing.ready


def test_several_missing_expected_streams_fail():
    j, c = healthy_frames(SUPPLIED[::2])
    report = completeness_of(j, c)
    streams = report.expected_streams
    assert sum(not r.report.is_healthy for r in streams.results) == 3
    assert CMP.EXPECTED_PAIRS_MISSING in report.blocking_reasons and not pricing_of(j, c, report).ready


def test_one_unexpected_stream_fails_pending_contract_review():
    j, c = healthy_frames((*SUPPLIED, ("Toronto", "SYNTH Branch")))
    coverage = assess_expected_location_coverage(c, COV)
    assert coverage.unexpected_location_count == 1 and coverage.spelling_variant_location_count == 0
    report = completeness_of(j, c)
    assert report.blocking_reasons == (CMP.UNEXPECTED_PAIRS, CMP.DECLARED_COUNT_UNRECONCILED) or (
        CMP.UNEXPECTED_PAIRS in report.blocking_reasons)
    pricing = pricing_of(j, c, report)
    assert PB.UNEXPECTED_SOURCE_STREAMS in pricing.blocking_reasons and not pricing.ready
    assert ("Toronto", "SYNTH Branch") not in COV.expected_locations                 # never added to the universe


def test_unexpected_stream_resembling_an_expected_one_is_a_spelling_mismatch():
    resembling = ("Vancouver", "thurlow")
    j, c = healthy_frames((*SUPPLIED[:-1], resembling))
    report = completeness_of(j, c)
    assert {CMP.EXPECTED_PAIRS_MISSING, CMP.UNEXPECTED_PAIRS, CMP.SOURCE_SPELLING_MISMATCH} <= set(report.blocking_reasons)
    pricing = pricing_of(j, c, report)
    assert {PB.EXPECTED_SOURCE_STREAMS_MISSING, PB.UNEXPECTED_SOURCE_STREAMS, PB.SOURCE_SPELLING_MISMATCH} <= set(
        pricing.blocking_reasons)


def test_missing_and_unexpected_streams_are_both_reported():
    j, c = healthy_frames((*SUPPLIED[:-1], ("Vancouver", "SYNTH Branch")))
    report = completeness_of(j, c)
    assert {CMP.EXPECTED_PAIRS_MISSING, CMP.UNEXPECTED_PAIRS} <= set(report.blocking_reasons)
    assert CMP.SOURCE_SPELLING_MISMATCH not in report.blocking_reasons


def _results(mutate):  # type: ignore[no-untyped-def]
    j, c = healthy_frames()
    streams = assess_expected_location_streams(j, c, coverage=COV)
    altered = dataclasses.replace(streams, results=mutate(streams.results))
    return j, c, altered


@pytest.mark.parametrize("mutate, blocker", [
    (lambda rs: (*rs, rs[0]), CMP.DUPLICATE_STREAM_REPORT),                          # duplicate report
    (lambda rs: rs[:-1], CMP.EXPECTED_STREAM_REPORT_MISSING),                         # omitted report
    (lambda rs: (*rs, dataclasses.replace(rs[0], target=("Toronto", "Thurlow"))), CMP.UNEXPECTED_STREAM_REPORT),
    (lambda rs: (dataclasses.replace(rs[0], report=None), *rs[1:]), CMP.EXPECTED_STREAM_REPORT_UNAVAILABLE),
])
def test_the_report_population_must_be_exactly_the_seven_approved_streams(mutate, blocker):
    j, c, streams = _results(mutate)
    assert isinstance(streams, ExpectedLocationStreamsReport) and not streams.is_valid
    report = completeness_of(j, c, streams)
    assert blocker in report.blocking_reasons and not report.complete
    pricing = pricing_of(j, c, report)
    assert PB.EXPECTED_STREAMS_NOT_PROVEN in pricing.blocking_reasons and not pricing.ready


@pytest.mark.parametrize("target", SUPPLIED)
def test_an_unhealthy_report_for_any_approved_stream_blocks(target):
    city, label = target
    others = tuple(k for k in SUPPLIED if k[0] == city and k != target)
    extra = cars_frame(*(("SYNTH-JOB-009", lab, city) for _, lab in others)) if others else None
    j, c = healthy_frames(extra_jobs=(("SYNTH-JOB-009", len(others), len(others), city),))
    if extra is not None:
        c = pd.concat([c, extra], ignore_index=True).astype(dict(REL.detail_definition.identifier_dtypes))
    report = completeness_of(j, c)
    streams = report.expected_streams
    assert not streams.reports[target].is_healthy                       # a capture of its city lacks it
    assert all(streams.reports[k].is_healthy for k in SUPPLIED if k[0] != city)
    assert not report.complete and not pricing_of(j, c, report).ready


def test_completeness_needs_the_contract_pricing_uses():
    j, c = healthy_frames()
    report = completeness_of(j, c)
    narrower = dataclasses.replace(COV, expected_locations=SUPPLIED[:3])
    pricing = pricing_of(j, c, report, contract=synthetic_contract(narrower))
    assert PB.EXPECTED_STREAM_CONTRACT_MISMATCH in pricing.blocking_reasons and not pricing.ready


def test_authority_blockers_clear_only_with_the_approved_record():
    j, c = healthy_frames()
    report = completeness_of(j, c)
    for record in (None, load_decision_record(V1), load_decision_record(V2)):
        blocked = pricing_of(j, c, report, contract=expected_stream_contract_from_record(record))
        assert {PB.EXPECTED_STREAM_AUTHORITY_UNAVAILABLE, PB.EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE} <= set(
            blocked.blocking_reasons)
    approved = pricing_of(j, c, report, contract=expected_stream_contract_from_record(load_decision_record(V3)))
    assert not {PB.EXPECTED_STREAM_AUTHORITY_UNAVAILABLE, PB.EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE} & set(
        approved.blocking_reasons)
    missing = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                       **(GATES | {"expected_stream_contract": None}))
    assert PB.EXPECTED_STREAM_AUTHORITY_UNAVAILABLE in missing.blocking_reasons


def test_unrelated_blockers_remain_with_the_approved_contract():
    j, c = healthy_frames()
    report = completeness_of(j, c)
    gates = gates_for(j, c, COV, report) | {"expected_stream_contract": current_expected_stream_contract(),
                                            "scheduled_coverage": None, "temporal_fields_trusted": False}
    undecided = dataclasses.replace(VANCOUVER_LOCATION_POLICY, state=LocationPolicyState.UNRESOLVED,
                                    authority=None, canonical_location=None)
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(undecided), **gates)
    assert not pricing.ready
    assert {PB.SCHEDULED_COVERAGE_ASSESSMENT_MISSING, PB.TEMPORAL_FIELDS_UNTRUSTED,
            PB.LOCATION_POLICY_UNRESOLVED} <= set(pricing.blocking_reasons)


def test_remaining_governance_decisions_are_still_proposed():
    record = load_decision_record(V3)
    for decision in (D.LOCATION_ROLE_ASSIGNMENTS, D.VALID_LOCATION_COMPARISON_PAIRS, D.VANCOUVER_LOCATION_IDENTITY,
                     D.SCHEDULE_CAPTURE_TIMESTAMP, D.SCHEDULE_EXPECTED_PERIODS, D.SCHEDULE_SHARING_MODEL,
                     D.SCHEDULE_EXCEPTIONS, D.FINISHED_AT_TIMEZONE, D.SCRAPED_FINISHED_ORDERING,
                     D.SCRAPED_FINISHED_TOLERANCE, D.REPORTING_DAY_SOURCE, D.REPORTING_DAY_TIMEZONE,
                     D.SCRAPE_DATE_SEMANTICS, D.DATE_CLEAN_SEMANTICS, D.RENTAL_DATE_VALIDITY,
                     D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS):
        assert record.decision(decision).status is DecisionStatus.PROPOSED, decision


# ============================================================== baseline & notebook


def baseline_for(j, c):  # type: ignore[no-untyped-def]
    report = completeness_of(j, c)
    pricing = pricing_of(j, c, report, policy=VANCOUVER_LOCATION_POLICY)
    stable = assess_vehicle_attribute_stability(c.assign(car_name="SYNTH Vehicle"))
    return build_pricing_baseline(pricing=pricing, jobs=j, cars=c, temporal=None, vehicle_stability=stable)


def test_baseline_reports_seven_expected_streams_and_a_separate_observed_population():
    j, c = healthy_frames((*SUPPLIED[1:], ("calgary", "Downtown")))       # one approved stream misspelled
    baseline = baseline_for(j, c)
    expected, observed = baseline.expected_population, baseline.observed_population
    assert expected.count == 7 and expected.keys == SUPPLIED
    assert (expected.authority, expected.coverage_mode) == ("authoritative_exhaustive", "exhaustive")
    assert (observed.count, observed.exact_expected_count, observed.unexpected_count,
            observed.spelling_variant_count, observed.expected_missing_count) == (7, 6, 1, 1, 1)
    assert ("calgary", "Downtown") not in observed.keys and observed.keys == SUPPLIED[1:]
    assert [h.stream for h in baseline.expected_stream_health] == list(SUPPLIED)
    # The misspelled observation is a case variant: an unverified alias, never applied.
    assert baseline.expected_stream_health[0].stream_status == LocationStreamStatus.UNVERIFIED_ALIAS.value
    assert sum(h.spelling_variant for h in baseline.observed_stream_health) == 1
    assert baseline.authority_record_version == 4
    assert ("expected_stream_authority", "approved") in baseline.statuses
    assert ("expected_stream_universe_mode", "exhaustive") in baseline.statuses
    assert {"source_spelling_mismatch", "unexpected_pairs", "expected_pairs_missing"} <= set(baseline.pricing_blockers)
    markdown = render_baseline_markdown(baseline, commit="abc1234", date="2026-10-06")
    assert "Configured expected streams: 7" in markdown and "Observed streams (cleaned detail rows): 7" in markdown
    assert "calgary" not in markdown and "SYNTH" not in markdown           # unapproved source value withheld
    assert "**NOT PRICING READY**" in markdown
    json.dumps(baseline.to_dict())


def test_observed_stream_health_keeps_a_partial_stream_visible_without_labels():
    extra_job = ("SYNTH-JOB-009", 1, 1, "Calgary")
    j, c = healthy_frames(extra_jobs=(extra_job,))
    c = pd.concat([c, cars_frame(("SYNTH-JOB-009", "Int Airport", "Calgary"))], ignore_index=True).astype(
        dict(REL.detail_definition.identifier_dtypes))
    baseline = baseline_for(j, c)
    partial = [h for h in baseline.observed_stream_health if h.continuity == "partial"]
    assert len(partial) == 1 and (partial[0].in_scope_jobs, partial[0].jobs_lacking_stream) == (2, 1)
    assert partial[0].in_contract and not partial[0].spelling_variant
    health = {h.stream: h for h in baseline.expected_stream_health}
    assert health[INVESTIGATED_LOCATION_STREAM].continuity == "partial"


def test_notebook_uses_the_resolved_contract_not_a_three_stream_minimum():
    notebook = json.loads((ROOT / "notebooks" / "01_data_ingestion.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "expected_stream_contract = current_expected_stream_contract()" in code
    assert "expected_stream_contract=expected_stream_contract" in code             # pricing consumes it
    assert "expected_coverage=expected_stream_contract.coverage" in code           # completeness uses it
    assert "MINIMUM_REQUIRED" not in code and "*COMPARED_LOCATION_STREAMS" not in code
    assert not any(f"'{label}'" in code or f'"{label}"' in code for _, label in SUPPLIED)


def test_documentation_no_longer_describes_a_three_stream_minimum():
    for path in (ROOT / "README.md", RECORD_DIR / "README.md", ROOT / "notebooks" / "README.md"):
        text = path.read_text(encoding="utf-8").lower()
        assert "three minimum" not in text and "minimum-required streams" not in text, path.name
        assert "required minimum" not in text or "not a required minimum" in text, path.name
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in ("seven", "EXHAUSTIVE", "exact", "new authority-record version", "VANCOUVER_LOCATION_IDENTITY",
                   "expected_stream_authority_unavailable"):
        assert phrase in readme, phrase
    assert "is pricing ready" not in readme.lower() or "not pricing ready" in readme.lower()
