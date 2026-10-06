"""Tests for the versioned pricing-authority decision record (synthetic records only)."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import re
import tomllib
from pathlib import Path

import pytest

from ql2_sixt_canada_analysis import authority_decisions as ad
from ql2_sixt_canada_analysis.authority_decisions import (
    AuthorityKind,
    AuthorityReference,
    DecisionId,
    DecisionRecordError,
    DecisionStatus,
    EvidenceKind,
    load_decision_record,
    parse_decision_record,
    render_authority_request_checklist,
    render_status_summary,
)
from ql2_sixt_canada_analysis.pricing_baseline import (
    ApprovedDateAgreement,
    LocationRole,
    PlanReadinessGap,
    baseline_authority_inputs,
    rental_date_fields,
)
from ql2_sixt_canada_analysis.readiness import PricingBlocker
from ql2_sixt_canada_analysis.schemas import JOB_DETAIL_RELATIONSHIP, LocationPolicyAuthority

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V1 = RECORD_DIR / "v1.toml"
D = DecisionId

A_AIR, A_DOWN = ["alpha", "Alpha Airport"], ["alpha", "Alpha Downtown"]
V_DOWN, V_THUR = ["vancouver", "Vancouver Downtown"], ["vancouver", "Vancouver Thurlow"]
UNIVERSE = [A_AIR, A_DOWN, V_DOWN, V_THUR]

RESOLUTIONS = {
    D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE: {"decimal_zero_suffix_equivalent": True},
    D.JOB_ID_LEADING_ZERO_SIGNIFICANCE: {"leading_zeros_significant": True},
    D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS: {k: True for k in (
        "non_zero_fractional_suffix_invalid", "scientific_notation_invalid", "sign_invalid",
        "surrounding_whitespace_invalid", "padding_invalid", "other_numeric_forms_invalid")},
    D.JOB_ID_RAW_AND_LINKAGE_PRESERVATION: {"preserve_raw_identifier": True, "separate_linkage_key": True},
    D.EXPECTED_STREAM_UNIVERSE: {"mode": "EXHAUSTIVE", "streams": UNIVERSE},
    D.EXPECTED_STREAM_SOURCE_SPELLING: {"streams": UNIVERSE},
    D.LOCATION_ROLE_ASSIGNMENTS: {"assignments": [
        {"stream": A_AIR, "role": "AIRPORT"}, {"stream": A_DOWN, "role": "DOWNTOWN"},
        {"stream": V_DOWN, "role": "DOWNTOWN"}, {"stream": V_THUR, "role": "OTHER"}]},
    D.VALID_LOCATION_COMPARISON_PAIRS: {"pairs": [{"airport": A_AIR, "downtown": A_DOWN}]},
    D.VANCOUVER_LOCATION_IDENTITY: {"state": "CONFIRMED_ALIAS", "canonical_location": V_DOWN},
    D.SCHEDULE_CAPTURE_TIMESTAMP: {"field": "cars.scraped_at"},
    D.SCHEDULE_EXPECTED_PERIODS: {"period": "PT1H",
                                  "period_starts": ["2000-01-01T00:00:00+00:00", "2000-01-01T01:00:00Z"]},
    D.SCHEDULE_SHARING_MODEL: {"mode": "SHARED"},
    D.SCHEDULE_EXCEPTIONS: {"model": "NO_EXCEPTIONS"},
    D.FINISHED_AT_TIMEZONE: {"timezone": "UTC"},
    D.SCRAPED_FINISHED_ORDERING: {"earlier": "cars.scraped_at", "later": "jobs.finished_at", "equal_allowed": True},
    D.SCRAPED_FINISHED_TOLERANCE: {"tolerance": 0, "unit": "MINUTES"},
    D.REPORTING_DAY_SOURCE: {"field": "jobs.scrape_date"},
    D.REPORTING_DAY_TIMEZONE: {"timezone": "America/Toronto"},
    D.SCRAPE_DATE_SEMANTICS: {"meaning": "SYNTH meaning", "derivation": "REPORTING_DAY"},
    D.DATE_CLEAN_SEMANTICS: {"meaning": "SYNTH meaning", "derivation": "SOURCE_SUPPLIED_UNDERIVED"},
    D.RENTAL_DATE_VALIDITY: {"date_format": "ISO calendar date", "pickup_before_return_required": True,
                             "equal_dates_allowed": False, "minimum_duration_days": 1,
                             "maximum_duration_days": 30},
    D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS: {"agreements": [
        {"source": "jobs.pickup_date", "target": "cars.job_pickup_date"},
        {"source": "jobs.return_date", "target": "cars.job_return_date"},
        {"source": "jobs.pickup_date", "target": "cars.pickup_date"},
        {"source": "jobs.return_date", "target": "cars.return_date"}]},
}


def _authority(kind: AuthorityKind) -> dict:
    return {"kind": kind.value, "source": f"SYNTH-{kind.value}", "reference": "SYNTH-DECISION-REF"}


def _proposed_entry(decision: DecisionId) -> dict:
    roles, joint = ad.REQUIRED_DECISIONS[decision]
    return {"id": decision.value, "status": "PROPOSED", "responsible_authority": [r.value for r in roles],
            "joint": joint, "question": "SYNTH question?", "blocking_external_input": True,
            "downstream": ["data_incomplete"]}


def _approve(entry: dict) -> dict:
    decision = DecisionId(entry["id"])
    roles, joint = ad.REQUIRED_DECISIONS[decision]
    entry.update(status="APPROVED", blocking_external_input=False,
                 authority=[_authority(r) for r in (roles if joint else roles[:1])],
                 resolution=copy.deepcopy(RESOLUTIONS[decision]))
    return entry


def _finish(data: dict) -> dict:
    statuses = [e["status"] for e in data["decisions"]]
    data["summary"] = {s.value.lower(): statuses.count(s.value) for s in DecisionStatus}
    data["external_inputs"] = [e["id"] for e in data["decisions"] if e["blocking_external_input"]]
    return data


def proposed_record() -> dict:
    return _finish({"schema_version": 1, "record_version": 1, "record_id": "pricing-authorities-v1",
                    "created": dt.date(2000, 1, 1), "scope": "SYNTH scope", "source_commit": "abcdef1",
                    "decisions": [_proposed_entry(d) for d in DecisionId]})


def approved_record() -> dict:
    data = proposed_record()
    data["decisions"] = [_approve(e) for e in data["decisions"]]
    return _finish(data)


def entry(data: dict, decision: DecisionId) -> dict:
    return next(e for e in data["decisions"] if e["id"] == decision.value)


def fails(data: dict, *, finish: bool = True) -> str:
    with pytest.raises(DecisionRecordError) as info:
        parse_decision_record(_finish(data) if finish else data)
    return str(info.value)


def with_resolution(decision: DecisionId, **changes: object) -> dict:
    data = approved_record()
    entry(data, decision)["resolution"].update(changes)
    return data


# ------------------------------------------------------------ committed revision


def test_committed_v1_validates_with_every_decision_proposed_and_blocking() -> None:
    record = load_decision_record(V1)
    assert record.record_id == "pricing-authorities-v1" and record.supersedes is None
    assert [d.id for d in record.decisions] == list(DecisionId)
    assert record.counts()[DecisionStatus.PROPOSED] == len(DecisionId) == 22
    assert record.external_inputs == tuple(DecisionId)
    for decision in record.decisions:
        assert decision.blocking_external_input and not decision.authority
        assert decision.resolution is None and decision.rejected is None
        assert decision.question.endswith("?")
        assert record.approved_resolution(decision.id) is None
        assert record.approved_authority(decision.id) is None


def test_committed_v1_records_observations_only_as_non_authoritative_evidence() -> None:
    record = load_decision_record(V1)
    universe = record.decision(D.EXPECTED_STREAM_UNIVERSE)
    observed = [e for e in universe.evidence if e.observed_candidates]
    assert len(observed) == 1 and observed[0].kind is EvidenceKind.RAW_DATA_OBSERVATION
    assert len(observed[0].observed_candidates) == 7
    assert "does not prove" in observed[0].summary
    assert {e.kind for e in record.decision(D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE).evidence} == {
        EvidenceKind.RAW_DATA_OBSERVATION}
    assert {e.kind for e in record.decision(D.VANCOUVER_LOCATION_IDENTITY).evidence} == {
        EvidenceKind.BEHAVIORAL_ANALYSIS}
    for decision in record.decisions:
        assert all(isinstance(e.kind, EvidenceKind) for e in decision.evidence)


def test_committed_v1_holds_no_source_level_values() -> None:
    text = V1.read_text(encoding="utf-8")
    assert not re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", text)
    assert not re.search(r"\d{6,}", text.replace("f9b1229da91a8bc05260e2b2458cb38312cc0697", ""))
    assert "[[decisions.authority]]" not in text and "resolution" not in text


def test_downstream_codes_are_known_blockers_or_plan_gaps() -> None:
    known = {b.value for b in PricingBlocker} | {g.value for g in PlanReadinessGap}
    for decision in load_decision_record(V1).decisions:
        assert set(decision.downstream) <= known, decision.id


def test_checklist_render_covers_every_v1_request() -> None:
    # The committed checklist follows the current revision (see test_authority_v2); v1 renders all 22.
    record = load_decision_record(V1)
    checklist = render_authority_request_checklist(record)
    for title in ("## Collection owner or supplier", "## Business owner", "## Joint decision"):
        assert title in checklist
    for decision in record.decisions:
        assert checklist.count(f"`{decision.id.value}`") == 1
        assert decision.question in checklist
    rows = [line for line in checklist.splitlines() if line.startswith("| `")]
    assert len(rows) == 22 and all(line.count("|") == 7 for line in rows)


def test_directory_readme_and_project_readme_document_the_record() -> None:
    readme = (RECORD_DIR / "README.md").read_text(encoding="utf-8")
    for phrase in ("Status semantics", "Authority requirements", "Supersession", "Validation",
                   "implemented separately", "Old revisions stay"):
        assert phrase in readme
    assert "docs/decisions/pricing_authorities" in (ROOT / "README.md").read_text(encoding="utf-8")


# --------------------------------------------------------------- valid records


def test_fully_approved_synthetic_record_is_valid_and_immutable() -> None:
    data = approved_record()
    snapshot = copy.deepcopy(data)
    record = parse_decision_record(data)
    assert data == snapshot                                   # input not modified
    assert record.counts()[DecisionStatus.APPROVED] == 22 and record.external_inputs == ()
    resolution = record.approved_resolution(D.EXPECTED_STREAM_UNIVERSE)
    assert resolution["mode"] == "EXHAUSTIVE"
    with pytest.raises(TypeError):
        resolution["mode"] = "MINIMUM_REQUIRED"               # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.record_version = 2                              # type: ignore[misc]
    assert isinstance(record.approved_authority(D.LOCATION_ROLE_ASSIGNMENTS), AuthorityReference)


def test_rejected_decision_needs_authority_and_rejection_statement() -> None:
    data = proposed_record()
    item = entry(data, D.SCHEDULE_SHARING_MODEL)
    item.update(status="REJECTED", blocking_external_input=False,
                authority=[_authority(AuthorityKind.COLLECTION_OWNER)], rejected="SYNTH proposal rejected")
    record = parse_decision_record(_finish(data))
    assert record.decision(D.SCHEDULE_SHARING_MODEL).rejected == "SYNTH proposal rejected"
    assert record.approved_resolution(D.SCHEDULE_SHARING_MODEL) is None
    del item["rejected"]
    assert "rejected" in fails(data)
    item.update(rejected="SYNTH", resolution={"mode": "SHARED"})
    assert "REJECTED carries no resolution" in fails(data)
    item.pop("resolution")
    item["authority"] = []
    assert "requires authority provenance" in fails(data)


# --------------------------------------------------------------- record shape


def test_versions_ids_and_supersession_fail_closed() -> None:
    for change, needle in ((dict(schema_version=3), "schema_version"), (dict(schema_version="1"), "schema_version"),
                           (dict(schema_version=0), "schema_version"),
                           (dict(record_version=0), "record_version"),
                           (dict(record_id="pricing-authorities-v9"), "record_id"),
                           (dict(supersedes="pricing-authorities-v0"), "supersedes nothing"),
                           (dict(created="2000-01-01"), "created"), (dict(source_commit="HEAD"), "source_commit")):
        assert needle in fails({**proposed_record(), **change})
    data = {**proposed_record(), "record_version": 2, "record_id": "pricing-authorities-v2"}
    assert "previous revision" in fails(data)
    assert parse_decision_record({**data, "supersedes": "pricing-authorities-v1"}).supersedes == "pricing-authorities-v1"


def test_every_decision_exactly_once_and_no_unknown_ids() -> None:
    data = proposed_record()
    data["decisions"].pop()
    assert "missing required decisions" in fails(data)
    data = proposed_record()
    data["decisions"].append(copy.deepcopy(data["decisions"][0]))
    assert "duplicate decision" in fails(data)
    data = proposed_record()
    data["decisions"][0]["id"] = "SYNTH_SECRET_DECISION"
    message = fails(data)
    assert "unknown decision id" in message and "SYNTH_SECRET" not in message


def test_status_roles_and_joint_flags_are_exact() -> None:
    for change, needle in ((dict(status="approved"), "status"), (dict(status="DRAFT"), "status"),
                           (dict(responsible_authority=[]), "responsible_authority"),
                           (dict(responsible_authority=["BUSINESS_OWNER"]), "required roles"),
                           (dict(responsible_authority=["RAW_DATA_OBSERVATION"]), "authority roles"),
                           (dict(joint=True), "joint"), (dict(question="  "), "question"),
                           (dict(downstream=[]), "downstream"), (dict(downstream=["Not A Code"]), "downstream"),
                           (dict(blocking_external_input="yes"), "boolean")):
        data = proposed_record()
        entry(data, D.SCHEDULE_SHARING_MODEL).update(change)
        assert needle in fails(data)


def test_proposed_must_block_and_cannot_carry_authority_or_resolution() -> None:
    for change in (dict(blocking_external_input=False), dict(resolution={"mode": "SHARED"}),
                   dict(authority=[_authority(AuthorityKind.SUPPLIER)]), dict(rejected="SYNTH")):
        data = proposed_record()
        entry(data, D.SCHEDULE_SHARING_MODEL).update(change)
        assert "PROPOSED" in fails(data)


def test_status_flip_alone_cannot_import_an_observation() -> None:
    data = tomllib.loads(V1.read_text(encoding="utf-8"))
    item = entry(data, D.EXPECTED_STREAM_UNIVERSE)
    item.update(status="APPROVED", blocking_external_input=False)
    assert "requires authority provenance" in fails(data)
    item["authority"] = [{"kind": "RAW_DATA_OBSERVATION", "source": "SYNTH", "reference": "SYNTH"}]
    assert "non-authoritative evidence cannot serve as authority" in fails(data)
    item["authority"] = [_authority(AuthorityKind.COLLECTION_OWNER), _authority(AuthorityKind.BUSINESS_OWNER)]
    assert "resolution" in fails(data)                         # evidence candidates are not a resolution


def test_authority_provenance_rules() -> None:
    for authority, needle in (
            ([_authority(AuthorityKind.BUSINESS_OWNER)], "not a responsible role"),
            ([{"kind": "COLLECTION_OWNER", "source": "SYNTH"}], "reference"),
            ([{"kind": "COLLECTION_OWNER", "source": " ", "reference": "SYNTH"}], "source"),
            ([{"kind": "SYNTH_KIND", "source": "S", "reference": "R"}], "authority kind"),
            ([{**_authority(AuthorityKind.SUPPLIER), "effective_date": "2000-01-01"}], "effective_date"),
            ([{**_authority(AuthorityKind.SUPPLIER), "extra": 1}], "unsupported fields")):
        data = approved_record()
        entry(data, D.SCHEDULE_SHARING_MODEL)["authority"] = authority
        assert needle in fails(data)
    data = approved_record()
    entry(data, D.RENTAL_DATE_VALIDITY)["authority"] = [_authority(AuthorityKind.COLLECTION_OWNER)]
    assert "joint decision requires every responsible role" in fails(data)


def test_evidence_kinds_cannot_be_authority_kinds() -> None:
    data = proposed_record()
    entry(data, D.SCHEDULE_SHARING_MODEL)["evidence"] = [
        {"kind": "BUSINESS_OWNER", "summary": "SYNTH", "reference": "SYNTH"}]
    assert "non-authoritative evidence kind" in fails(data)
    with pytest.raises(DecisionRecordError):
        ad.EvidenceReference(kind=AuthorityKind.SUPPLIER, summary="S", reference="R")  # type: ignore[arg-type]
    with pytest.raises(DecisionRecordError):
        AuthorityReference(kind=EvidenceKind.RAW_DATA_OBSERVATION, source="S", reference="R")  # type: ignore[arg-type]


def test_summary_and_external_inputs_must_match() -> None:
    data = proposed_record()
    data["summary"]["proposed"] = 21
    assert "summary" in fails(data, finish=False)
    data = proposed_record()
    data["external_inputs"] = data["external_inputs"][1:]
    assert "external_inputs" in fails(data, finish=False)
    data = proposed_record()
    data["external_inputs"].append(data["external_inputs"][0])
    assert "external_inputs" in fails(data, finish=False)


def test_unknown_fields_and_malformed_files_fail_closed(tmp_path: Path) -> None:
    assert "unsupported fields" in fails({**proposed_record(), "synth": 1})
    data = proposed_record()
    data["decisions"][0]["synth"] = 1
    assert "unsupported fields" in fails(data)
    bad = tmp_path / "bad.toml"
    bad.write_text("schema_version = [", encoding="utf-8")
    with pytest.raises(DecisionRecordError, match="not valid TOML"):
        load_decision_record(bad)
    with pytest.raises(DecisionRecordError, match="cannot be read"):
        load_decision_record(tmp_path / "missing.toml")
    with pytest.raises(DecisionRecordError):
        parse_decision_record({**proposed_record(), "decisions": "SYNTH"})


# ----------------------------------------------------- per-decision resolutions


def test_job_id_resolutions_are_complete_booleans() -> None:
    assert "boolean" in fails(with_resolution(D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE, decimal_zero_suffix_equivalent=1))
    data = approved_record()
    entry(data, D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS)["resolution"].pop("sign_invalid")
    assert "required fields" in fails(data)


def test_stream_universe_keys_are_exact() -> None:
    for streams in ([["alpha"]], [["alpha", ""]], [["alpha", " Alpha Airport"]], [A_AIR, A_AIR], []):
        assert "JOB_ID" not in fails(with_resolution(D.EXPECTED_STREAM_UNIVERSE, streams=streams))
    assert "unsupported value" in fails(with_resolution(D.EXPECTED_STREAM_UNIVERSE, mode="SOME"))
    data = approved_record()
    entry(data, D.EXPECTED_STREAM_UNIVERSE)["resolution"].pop("mode")
    assert "EXPECTED_STREAM_UNIVERSE" in fails(data)
    assert "approved stream universe" in fails(with_resolution(D.EXPECTED_STREAM_SOURCE_SPELLING, streams=[A_AIR]))


def test_role_assignments_cover_the_universe_exactly() -> None:
    base = RESOLUTIONS[D.LOCATION_ROLE_ASSIGNMENTS]["assignments"]
    for assignments in (base[:-1], [*base, {"stream": ["beta", "Beta"], "role": "OTHER"}], [*base, base[0]]):
        assert "exactly once" in fails(with_resolution(D.LOCATION_ROLE_ASSIGNMENTS, assignments=assignments))
    bad_role = [{**base[0], "role": "airport"}, *base[1:]]
    assert "unsupported value" in fails(with_resolution(D.LOCATION_ROLE_ASSIGNMENTS, assignments=bad_role))
    data = approved_record()
    entry(data, D.EXPECTED_STREAM_UNIVERSE).update(_proposed_entry(D.EXPECTED_STREAM_UNIVERSE))
    for item in data["decisions"]:
        if item["id"] in {D.EXPECTED_STREAM_SOURCE_SPELLING.value}:
            item.update(_proposed_entry(D.EXPECTED_STREAM_SOURCE_SPELLING))
            item.pop("authority"), item.pop("resolution")
    entry(data, D.EXPECTED_STREAM_UNIVERSE).pop("authority"), entry(data, D.EXPECTED_STREAM_UNIVERSE).pop("resolution")
    assert "requires EXPECTED_STREAM_UNIVERSE to be APPROVED" in fails(data)


def test_comparison_pairs_stay_within_city_and_use_approved_roles() -> None:
    for pairs, needle in (([{"airport": A_AIR, "downtown": V_DOWN}], "within one city"),
                          ([{"airport": ["alpha", "Alpha Other"], "downtown": A_DOWN}], "approved expected"),
                          ([{"airport": A_DOWN, "downtown": A_AIR}], "airport/downtown roles"),
                          ([{"airport": A_AIR, "downtown": A_DOWN}] * 2, "duplicate comparison pair"),
                          ([], "non-empty")):
        assert needle in fails(with_resolution(D.VALID_LOCATION_COMPARISON_PAIRS, pairs=pairs))


def test_vancouver_identity_alias_and_distinct_rules() -> None:
    assert "governed keys" in fails(with_resolution(D.VANCOUVER_LOCATION_IDENTITY, canonical_location=A_AIR))
    data = approved_record()
    entry(data, D.VANCOUVER_LOCATION_IDENTITY)["resolution"] = {"state": "CONFIRMED_DISTINCT"}
    parse_decision_record(_finish(data))
    entry(data, D.VANCOUVER_LOCATION_IDENTITY)["resolution"]["canonical_location"] = V_DOWN
    assert "required fields" in fails(data)
    data = approved_record()
    entry(data, D.VANCOUVER_LOCATION_IDENTITY)["resolution"] = {"state": "CONFIRMED_DISTINCT"}
    universe = [A_AIR, A_DOWN, V_DOWN]
    entry(data, D.EXPECTED_STREAM_UNIVERSE)["resolution"]["streams"] = universe
    entry(data, D.EXPECTED_STREAM_SOURCE_SPELLING)["resolution"]["streams"] = universe
    entry(data, D.LOCATION_ROLE_ASSIGNMENTS)["resolution"]["assignments"].pop()
    assert "approved roles for both keys" in fails(data)
    assert "unsupported value" in fails(with_resolution(D.VANCOUVER_LOCATION_IDENTITY, state="UNRESOLVED"))


def test_schedule_periods_need_explicit_offsets_and_no_duplicates() -> None:
    for starts, needle in ((["2000-01-01T00:00:00"], "explicit UTC offset"),
                           (["2000-01-01T00:00:00Z", "2000-01-01T01:00:00+01:00"], "duplicate period starts"),
                           (["SYNTH+00:00"], "not ISO-8601"), ([], "non-empty")):
        assert needle in fails(with_resolution(D.SCHEDULE_EXPECTED_PERIODS, period_starts=starts))
    assert "ISO-8601 duration" in fails(with_resolution(D.SCHEDULE_EXPECTED_PERIODS, period="hourly"))
    assert "unsupported value" in fails(with_resolution(D.SCHEDULE_SHARING_MODEL, mode=None))
    assert "permitted field" in fails(with_resolution(D.SCHEDULE_CAPTURE_TIMESTAMP, field="jobs.scrape_date"))


def test_schedule_exceptions_must_be_authority_backed() -> None:
    listed = {"period_start": "2000-01-01T00:00:00Z", "reason": "OUTAGE", "authority_reference": "SYNTH-REF"}
    data = with_resolution(D.SCHEDULE_EXCEPTIONS, model="LISTED_EXCEPTIONS", exceptions=[listed])
    parse_decision_record(_finish(data))
    for change, needle in ((dict(authority_reference=" "), "authority_reference"),
                           (dict(period_start="2000-01-01T00:00:00"), "explicit UTC offset"),
                           (dict(reason="SYNTH"), "unsupported value")):
        assert needle in fails(with_resolution(D.SCHEDULE_EXCEPTIONS, model="LISTED_EXCEPTIONS",
                                               exceptions=[{**listed, **change}]))
    assert "non-empty" in fails(with_resolution(D.SCHEDULE_EXCEPTIONS, model="LISTED_EXCEPTIONS", exceptions=[]))


def test_timezones_ordering_and_tolerance() -> None:
    for decision in (D.FINISHED_AT_TIMEZONE, D.REPORTING_DAY_TIMEZONE):
        for zone in ("Mars/Olympus", "", 5):
            assert "IANA" in fails(with_resolution(decision, timezone=zone))
    assert "two different fields" in fails(with_resolution(D.SCRAPED_FINISHED_ORDERING, later="cars.scraped_at"))
    data = approved_record()
    entry(data, D.SCRAPED_FINISHED_ORDERING)["resolution"].pop("equal_allowed")
    assert "required fields" in fails(data)
    for tolerance in (-1, 1.5, True):
        assert "non-negative integer" in fails(with_resolution(D.SCRAPED_FINISHED_TOLERANCE, tolerance=tolerance))
    data = approved_record()
    entry(data, D.SCRAPED_FINISHED_ORDERING).update(_proposed_entry(D.SCRAPED_FINISHED_ORDERING))
    entry(data, D.SCRAPED_FINISHED_ORDERING).pop("authority"), entry(data, D.SCRAPED_FINISHED_ORDERING).pop("resolution")
    assert "requires SCRAPED_FINISHED_ORDERING to be APPROVED" in fails(data)


def test_reporting_day_must_be_approved_before_date_semantics() -> None:
    assert "contract column" in fails(with_resolution(D.REPORTING_DAY_SOURCE, field="jobs.synth"))
    data = approved_record()
    for decision in (D.REPORTING_DAY_TIMEZONE,):
        entry(data, decision).update(_proposed_entry(decision))
        entry(data, decision).pop("authority"), entry(data, decision).pop("resolution")
    assert "requires REPORTING_DAY_TIMEZONE to be APPROVED" in fails(data)
    assert "unsupported value" in fails(with_resolution(D.DATE_CLEAN_SEMANTICS, derivation="GUESS"))


def test_rental_date_validity_is_complete() -> None:
    data = approved_record()
    entry(data, D.RENTAL_DATE_VALIDITY)["resolution"].pop("equal_dates_allowed")
    assert "required fields" in fails(data)
    assert "duration limits" in fails(with_resolution(D.RENTAL_DATE_VALIDITY, minimum_duration_days=40))
    assert "duration limits" in fails(with_resolution(D.RENTAL_DATE_VALIDITY, maximum_duration_days=-1))


def test_rental_agreements_cover_exactly_the_six_fields() -> None:
    base = RESOLUTIONS[D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS]["agreements"]
    for agreements, needle in (
            (base[:-1], "exactly one approved source"),
            ([*base[:-1], {"source": "jobs.return_date", "target": "cars.pickup_date"}], "exactly one"),
            ([*base, {"source": "jobs.return_date", "target": "cars.pickup_date"}], "exactly one"),
            ([*base[:-1], {"source": "jobs.return_date", "target": "cars.scraped_at"}], "permitted field"),
            ([*base[:-1], {"source": "cars.return_date", "target": "cars.return_date"}], "permitted field"),
            ([{"source": "jobs.pickup_date"}, *base[1:]], "required fields")):
        assert needle in fails(with_resolution(D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS, agreements=agreements))
    assert ad.RENTAL_DATE_PARENT_FIELDS + ad.RENTAL_DATE_DETAIL_FIELDS == tuple(
        f"{d}.{c}" for d, c in rental_date_fields(JOB_DETAIL_RELATIONSHIP))


# ------------------------------------------------------------ sanitization


@pytest.mark.parametrize("secret", ["2026-01-02 03:04", "$123", "45.67", "987654321"])
def test_source_like_text_is_rejected_and_never_echoed(secret: str) -> None:
    data = proposed_record()
    entry(data, D.SCHEDULE_SHARING_MODEL)["question"] = f"SYNTH {secret}?"
    message = fails(data)
    assert "source-level data" in message and secret not in message


def test_error_messages_never_echo_resolution_values() -> None:
    secret = "SYNTHSECRETVALUE"
    for decision, change in ((D.SCHEDULE_SHARING_MODEL, dict(mode=secret)),
                             (D.FINISHED_AT_TIMEZONE, dict(timezone=secret)),
                             (D.EXPECTED_STREAM_UNIVERSE, dict(streams=[[secret, " x"]])),
                             (D.REPORTING_DAY_SOURCE, dict(field=f"jobs.{secret}"))):
        assert secret not in fails(with_resolution(decision, **change))


# --------------------------------------------------------------- CLI / summary


def test_cli_prints_sanitized_summary_and_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ad.main([str(V1)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("VALID\n") and "PROPOSED=22" in out
    record = load_decision_record(V1)
    assert out == "VALID\n" + render_status_summary(record)
    assert not any(d.question in out for d in record.decisions)
    assert "Calgary" not in out and "Vancouver" not in out
    bad = tmp_path / "v1.toml"
    bad.write_text(V1.read_text(encoding="utf-8").replace('status = "PROPOSED"', 'status = "APPROVED"', 1),
                   encoding="utf-8")
    assert ad.main([str(bad)]) == 1
    assert capsys.readouterr().out.startswith("INVALID:")
    assert ad.main([]) == 2


# ---------------------------------------------------------- baseline adapter


def test_v1_supplies_no_baseline_authority_inputs() -> None:
    inputs = baseline_authority_inputs(load_decision_record(V1))
    assert inputs == dict(location_role_map=None, location_role_authority=None,
                          rental_period_rule_authority=None, approved_rental_date_agreements=None)


def test_approved_record_supplies_typed_baseline_inputs() -> None:
    inputs = baseline_authority_inputs(parse_decision_record(approved_record()))
    assert isinstance(inputs["location_role_authority"], AuthorityReference)
    assert isinstance(inputs["rental_period_rule_authority"], AuthorityReference)
    assert inputs["location_role_map"][tuple(V_THUR)] is LocationRole.OTHER
    agreements = inputs["approved_rental_date_agreements"]
    assert all(isinstance(a, ApprovedDateAgreement) for a in agreements)
    assert {a.target for a in agreements} == set(rental_date_fields(JOB_DETAIL_RELATIONSHIP)[2:])
    with pytest.raises(Exception):
        baseline_authority_inputs(approved_record())          # type: ignore[arg-type]


def test_location_policy_authority_no_longer_satisfies_baseline_authority() -> None:
    from ql2_sixt_canada_analysis import pricing_baseline as pb

    legacy = LocationPolicyAuthority(source="SYNTH", reference="SYNTH")
    assert not pb._role_map_sufficient({("a", "b"): LocationRole.AIRPORT}, legacy, {("a", "b")}, 2)
    neutral = AuthorityReference(kind=AuthorityKind.BUSINESS_OWNER, source="SYNTH", reference="SYNTH")
    assert pb._role_map_sufficient({("a", "b"): LocationRole.OTHER}, neutral, {("a", "b")}, 2)
