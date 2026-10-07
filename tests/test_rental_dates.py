"""Authority-backed rental-date validity, parent/detail agreement and pricing eligibility (pricing-authorities-v7).

Committed records are read as data; negative cases mutate parsed copies in
memory. Frames are fabricated (``SYNTH-JOB-*``, synthetic dates); nothing here
comes from the source extracts.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from conftest import link
from stream_contract_fixtures import passing_rental_report, synthetic_rental_policy, with_rental_dates
from test_completeness import CARS, JOBS, frame
from test_readiness import DISTINCT, GATES, scheduled_frames

from ql2_sixt_canada_analysis.authority_decisions import (
    CURRENT_RECORD_PATH,
    AuthorityKind,
    DecisionId,
    DecisionRecordError,
    DecisionStatus,
    load_current_decision_record,
    load_decision_record,
    parse_decision_record,
)
from ql2_sixt_canada_analysis.readiness import PricingBlocker as PB, assess_location_policy, assess_pricing_readiness
from ql2_sixt_canada_analysis.rental_dates import (
    AgreementStatus as AS,
    DateValueStatus as V,
    PeriodStatus as P,
    RentalDateBlocker as RB,
    RentalDatePolicy,
    RentalDateReport,
    RentalPolicyStatus as RS,
    analysis_duration_cohort,
    assess_rental_dates,
    current_rental_date_policy,
    derive_rental_periods,
    parse_iso_dates,
    rental_date_policy_from_record,
)
from ql2_sixt_canada_analysis.schemas import DATASET_DEFINITIONS, JOB_LINKAGE_KEY_COLUMN

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = ROOT / "docs" / "decisions" / "pricing_authorities"
V6, V7 = RECORD_DIR / "v6.toml", RECORD_DIR / "v7.toml"
GOVERNANCE = "docs/decisions/governance/rental-date-validity-and-agreement-governance-v1-2026-10-06.md"
HISTORY = ("v1.toml", "v2.toml", "v3.toml", "v4.toml", "v5.toml", "v6.toml")
D = DecisionId
RENTAL = (D.RENTAL_DATE_VALIDITY, D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS)
#: The decisions exactly as supplied (test oracle).
MAPPINGS = {("cars.job_pickup_date", "jobs.pickup_date"), ("cars.job_return_date", "jobs.return_date"),
            ("cars.pickup_date", "jobs.pickup_date"), ("cars.return_date", "jobs.return_date")}
POLICY = current_rental_date_policy()


def v7() -> dict:
    return tomllib.loads(V7.read_text(encoding="utf-8"))


def entry(data: dict, decision: DecisionId) -> dict:
    return next(e for e in data["decisions"] if e["id"] == decision.value)


def finish(data: dict) -> dict:
    statuses = [e["status"] for e in data["decisions"]]
    data["summary"] = {s.value.lower(): statuses.count(s.value) for s in DecisionStatus}
    data["external_inputs"] = [e["id"] for e in data["decisions"] if e["blocking_external_input"]]
    return data


def unapprove(data: dict, *decisions: DecisionId) -> dict:
    for decision in decisions:
        item = entry(data, decision)
        item.update(status="PROPOSED", blocking_external_input=True)
        item.pop("authority", None), item.pop("resolution", None)
    return finish(data)


def fails(data: dict) -> str:
    with pytest.raises(DecisionRecordError) as info:
        parse_decision_record(finish(data))
    return str(info.value)


# ================================================================== authority record


def test_history_is_unchanged_and_valid() -> None:
    import subprocess

    for name in HISTORY:
        committed = subprocess.run(["git", "show", f"HEAD:docs/decisions/pricing_authorities/{name}"], cwd=ROOT,
                                   capture_output=True, check=False).stdout
        if committed:                                           # byte-identical to the committed history
            assert hashlib.sha256((RECORD_DIR / name).read_bytes()).digest() == hashlib.sha256(committed).digest()
        assert all(load_decision_record(RECORD_DIR / name).decision(d).status is DecisionStatus.PROPOSED
                   for d in RENTAL)


def test_v7_approves_exactly_the_rental_decisions_and_v8_carries_them() -> None:
    record, v6 = load_decision_record(V7), load_decision_record(V6)
    assert (record.schema_version, record.record_version, record.record_id, record.supersedes) == (
        3, 7, "pricing-authorities-v7", "pricing-authorities-v6")
    assert CURRENT_RECORD_PATH.name == "v8.toml"
    current = load_current_decision_record()
    for decision in (D.RENTAL_DATE_VALIDITY, D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS):
        assert current.decision(decision) == record.decision(decision)
    counts = record.counts()
    assert (counts[DecisionStatus.APPROVED], counts[DecisionStatus.PROPOSED], counts[DecisionStatus.REJECTED]) == (
        18, 4, 0)
    for decision in record.decisions:
        if decision.id in RENTAL:
            assert decision.status is DecisionStatus.APPROVED
        else:                                            # every other decision unchanged, PROPOSED ones included
            assert decision == v6.decision(decision.id), decision.id


def test_v7_resolutions_are_the_supplied_decisions() -> None:
    record = load_decision_record(V7)
    assert dict(record.approved_resolution(D.RENTAL_DATE_VALIDITY)) == {
        "date_format": "ISO_8601_DATE", "pickup_required": True, "return_required": True,
        "ordering": "RETURN_ON_OR_AFTER_PICKUP", "equal_dates_allowed": True, "minimum_duration_days": 0,
        "maximum_duration_mode": "UNBOUNDED"}
    assert {(a["target"], a["source"]) for a in
            record.approved_resolution(D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS)["agreements"]} == MAPPINGS
    assert [a.kind for a in record.decision(D.RENTAL_DATE_VALIDITY).authority] == [
        AuthorityKind.COLLECTION_OWNER, AuthorityKind.BUSINESS_OWNER]
    assert [a.kind for a in record.decision(D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS).authority] == [
        AuthorityKind.COLLECTION_OWNER]
    for decision in RENTAL:
        assert {a.reference for a in record.decision(decision).authority} == {GOVERNANCE}
    assert (ROOT / GOVERNANCE).is_file()


def test_a_missing_governance_reference_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(DecisionRecordError, match="missing"):
        load_decision_record(V7, repository_root=tmp_path)
    data = v7()
    entry(data, D.RENTAL_DATE_VALIDITY)["authority"][0]["reference"] = "docs/decisions/governance/synth-missing.md"
    assert "missing" in fails(data)


def test_current_policy_is_typed_and_explicitly_unbounded() -> None:
    assert POLICY is current_rental_date_policy() and POLICY.available and POLICY.record_id == "pricing-authorities-v8"
    assert (POLICY.source_format, POLICY.pickup_required, POLICY.return_required, POLICY.ordering,
            POLICY.equal_dates_allowed, POLICY.minimum_duration_days, POLICY.maximum_duration_mode,
            POLICY.maximum_duration_days) == ("ISO_8601_DATE", True, True, "RETURN_ON_OR_AFTER_PICKUP", True, 0,
                                              "UNBOUNDED", None)
    assert set(POLICY.agreements) == MAPPINGS and POLICY.references == (GOVERNANCE,)
    assert POLICY.parent_fields == ("jobs.pickup_date", "jobs.return_date")
    assert POLICY.detail_fields == ("cars.job_pickup_date", "cars.job_return_date", "cars.pickup_date",
                                    "cars.return_date")


# ================================================================== authority schema


def _validity(**change):  # type: ignore[no-untyped-def]
    data = v7()
    resolution = entry(data, D.RENTAL_DATE_VALIDITY)["resolution"]
    for key, value in change.items():
        if value is ...:
            resolution.pop(key)
        else:
            resolution[key] = value
    return data


@pytest.mark.parametrize("change, needle", [
    (dict(date_format=...), "required fields"),
    (dict(date_format="YYYY/MM/DD"), "unsupported value"),
    (dict(date_format="ISO calendar date"), "unsupported value"),
    (dict(ordering=...), "required fields"),
    (dict(pickup_required=...), "required fields"),
    (dict(pickup_required="yes"), "boolean"),
    (dict(equal_dates_allowed=False), "contradicts the ordering"),
    (dict(ordering="RETURN_AFTER_PICKUP"), "contradicts the ordering"),
    (dict(minimum_duration_days=-1), "non-negative"),
    (dict(minimum_duration_days=1), "minimum duration must be zero"),
    (dict(maximum_duration_days=30), "required fields"),                    # arbitrary max under UNBOUNDED
    (dict(maximum_duration_mode=...), "required fields"),
    (dict(maximum_duration_mode="NONE"), "unsupported value"),             # malformed unbounded representation
    (dict(maximum_duration_mode="UNBOUNDED", maximum_duration_days=0), "required fields"),   # zero is not unbounded
    (dict(maximum_duration_mode="BOUNDED"), "required fields"),           # bounded without a value
    (dict(maximum_duration_mode="BOUNDED", maximum_duration_days=0), "bounded maximum"),
])
def test_validity_resolution_must_be_complete_and_consistent(change, needle) -> None:
    assert needle in fails(_validity(**change))


def test_a_bounded_maximum_is_representable_but_not_approved() -> None:
    record = parse_decision_record(finish(_validity(maximum_duration_mode="BOUNDED", maximum_duration_days=60)))
    policy = rental_date_policy_from_record(record)
    assert policy.available and policy.maximum_duration_days == 60     # only a new record could set this


def _agreements(mutate):  # type: ignore[no-untyped-def]
    data = v7()
    mutate(entry(data, D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS)["resolution"]["agreements"])
    return data


@pytest.mark.parametrize("mutate, needle", [
    (lambda a: a.pop(), "exactly one approved source"),                                   # missing mapping
    (lambda a: a.append(dict(a[0])), "exactly one approved source"),                      # duplicate mapping
    (lambda a: a[0].update(source="jobs.scrape_date"), "SCRAPE"),                         # unknown parent field
    (lambda a: a[0].update(target="cars.date_clean"), "RENTAL_DATE_PARENT_DETAIL"),       # unknown detail field
    (lambda a: a[0].update(source="jobs.return_date"), "pickup field must repeat"),        # pickup <- return
    (lambda a: a[1].update(source="jobs.pickup_date"), "pickup field must repeat"),        # return <- pickup
    (lambda a: a[2].update(target="cars.job_pickup_date"), "exactly one approved source"),  # incomplete coverage
    (lambda a: a.clear(), "non-empty"),
])
def test_agreement_mappings_must_be_exact(mutate, needle) -> None:
    message = fails(_agreements(mutate))
    assert needle in message or "RENTAL_DATE_PARENT_DETAIL_AGREEMENTS" in message


# ======================================================================== parsing


@pytest.mark.parametrize("value, status, date", [
    ("2026-08-28", V.VALID, dt.date(2026, 8, 28)),
    ("2024-02-29", V.VALID, dt.date(2024, 2, 29)),           # leap day in a leap year
    ("2026-02-29", V.INVALID, None),                         # invalid leap day
    ("2026-13-01", V.INVALID, None),                         # impossible month
    ("2026-04-31", V.INVALID, None),                         # impossible day
    ("2026-08-28 00:00:00", V.INVALID, None),                # timestamp in a date field
    ("2026-08-28T00:00:00Z", V.INVALID, None),               # timezone-bearing timestamp
    ("2026-08-28T00:00:00-06:00", V.INVALID, None),
    (20260828, V.INVALID, None),                             # numeric value
    (46262, V.INVALID, None),                                # spreadsheet serial number
    (46262.0, V.INVALID, None),
    ("46262", V.INVALID, None),
    ("2026/08/28", V.INVALID, None),                         # slash-separated
    ("28/08/2026", V.INVALID, None),                         # locale-formatted
    ("Aug 28, 2026", V.INVALID, None),                       # month name
    ("20260828", V.INVALID, None),                           # basic ISO form is not the approved text
    (" 2026-08-28", V.INVALID, None),                        # leading whitespace
    ("2026-08-28 ", V.INVALID, None),                        # trailing whitespace
    ("2026-08-28 SYNTH", V.INVALID, None),                   # additional text
    ("", V.MISSING, None),                                   # empty string
    ("   ", V.MISSING, None),                                # whitespace-only
    (None, V.MISSING, None),                                 # null
    (float("nan"), V.MISSING, None),
    (dt.date(2026, 8, 28), V.INVALID, None),                 # unsupported object type
    (pd.Timestamp("2026-08-28"), V.INVALID, None),
])
def test_iso_dates_are_parsed_strictly(value, status, date) -> None:
    source = pd.Series([value], dtype=object)
    statuses, dates = parse_iso_dates(source)
    assert statuses.tolist() == [status] and dates.tolist() == [date]
    assert source.tolist()[0] is value or (isinstance(value, float) and value != value)    # raw value kept
    assert not isinstance(dates.iloc[0], (dt.datetime, pd.Timestamp))                       # never a timestamp


# ======================================================================== frames

DAY = "2026-08-28"
NEXT = "2026-08-29"


def build(rows, parent=None):  # type: ignore[no-untyped-def]
    """rows: detail dicts (job, jp, jr, p, r); parent: {job: (pickup, return)} - defaults DAY/NEXT."""
    parent = parent or {}
    jobs_ = sorted({r["job"] for r in rows} | set(parent))
    counts = {j: sum(1 for r in rows if r["job"] == j) for j in jobs_}
    jframe = frame(JOBS, [{"job_id": j, "record_count": str(counts[j]), "actual_car_rows": str(counts[j]),
                           "pickup_date": parent.get(j, (DAY, NEXT))[0], "return_date": parent.get(j, (DAY, NEXT))[1]}
                          for j in jobs_])
    cframe = frame(CARS, [{"job_id": r["job"], "row_index": str(i), "job_pickup_date": r.get("jp", DAY),
                           "job_return_date": r.get("jr", NEXT), "pickup_date": r.get("p", DAY),
                           "return_date": r.get("r", NEXT)} for i, r in enumerate(rows)])
    return jframe, cframe


def assess(rows, parent=None, policy=POLICY):  # type: ignore[no-untyped-def]
    j, c = build(rows, parent)
    result = link(j, c)
    return assess_rental_dates(result.jobs, result.cars, policy=policy, job_linkage=result.report)


def detail(job="SYNTH-JOB-1", **dates):  # type: ignore[no-untyped-def]
    return {"job": job, **dates}


def agreement(report: RentalDateReport, target: str) -> dict:
    return dict(next(a for a in report.agreements if a.target == target).by_status)


def period(report: RentalDateReport, name: str):  # type: ignore[no-untyped-def]
    return next(p for p in report.periods if p.period == name)


PARENT = "jobs.pickup_date..jobs.return_date"
OWN = "cars.pickup_date..cars.return_date"


# ======================================================================== validity


@pytest.mark.parametrize("pickup, return_, status, days", [
    ("2026-08-28", "2026-08-30", P.VALID, 2),                # return after pickup
    ("2026-08-28", "2026-08-28", P.VALID, 0),                # equal: zero-day duration, same-day rental
    ("2026-08-28", "2026-08-29", P.VALID, 1),                # one-day duration
    ("2026-08-28", "2027-11-30", P.VALID, 459),              # unusually long: still valid (no maximum)
    ("2026-08-28", "2026-08-27", P.RETURN_BEFORE_PICKUP, -1),
    ("", "2026-08-29", P.PICKUP_MISSING, None),
    ("2026-08-28", "", P.RETURN_MISSING, None),
    ("", "", P.BOTH_MISSING, None),
    ("2026/08/28", "2026-08-29", P.PICKUP_INVALID, None),
    ("2026-08-28", "2026-08-29T00:00:00", P.RETURN_INVALID, None),
])
def test_parent_period_validity(pickup, return_, status, days) -> None:
    rows = [detail(jp=pickup, jr=return_, p=pickup, r=return_)]
    report = assess(rows, parent={"SYNTH-JOB-1": (pickup, return_)})
    assert period(report, PARENT).count(status) == 1
    j, c = build(rows, parent={"SYNTH-JOB-1": (pickup, return_)})
    result = link(j, c)
    derived = derive_rental_periods(result.jobs, result.cars, policy=POLICY, job_linkage=result.report)
    duration = derived.jobs["rental_duration_days"].iloc[0]
    assert (pd.isna(duration) and days is None) or duration == days
    assert bool(derived.jobs["pricing_eligible"].iloc[0]) is (status is P.VALID)
    if status is P.VALID:
        assert report.is_valid and report.eligible_detail_rows == 1
    assert period(report, PARENT).same_day == (1 if days == 0 else 0)


def test_missing_and_invalid_values_map_to_their_blockers() -> None:
    missing = assess([detail(p="")])
    assert RB.RENTAL_DATE_REQUIRED_VALUE_MISSING in missing.blocking_reasons and missing.missing_values == 1
    invalid = assess([detail(r="2026-08-29 ")])
    assert RB.RENTAL_DATE_FORMAT_INVALID in invalid.blocking_reasons and invalid.invalid_values == 1
    ordering = assess([detail()], parent={"SYNTH-JOB-1": (NEXT, DAY)})
    assert RB.RENTAL_DATE_ORDERING_INVALID in ordering.blocking_reasons and ordering.ordering_violations >= 1


def test_no_maximum_or_observed_threshold_is_applied() -> None:
    long = "2031-08-28"
    rows = [detail(job=f"SYNTH-JOB-{i}", jr=long, r=long) for i in range(3)] + [detail(job="SYNTH-JOB-9")]
    report = assess(rows, parent={**{f"SYNTH-JOB-{i}": (DAY, long) for i in range(3)}, "SYNTH-JOB-9": (DAY, NEXT)})
    assert report.is_valid and report.blocking_reasons == ()                 # an outlier is still valid data
    assert period(report, PARENT).long_informational == 3 and period(report, PARENT).count(P.ABOVE_MAXIMUM) == 0
    assert report.eligible_detail_rows == 4 and POLICY.maximum_duration_days is None


# ======================================================================== agreement

ALL = ("cars.job_pickup_date", "cars.job_return_date", "cars.pickup_date", "cars.return_date")


def test_all_six_fields_valid_and_agreeing() -> None:
    report = assess([detail(), detail(job="SYNTH-JOB-2")])
    assert report.validity_holds and report.agreement_holds and report.is_valid and report.linkage_trusted
    assert all(agreement(report, t) == {"match": 2} for t in ALL)
    assert (report.eligible_parent_rows, report.eligible_detail_rows) == (2, 2)


@pytest.mark.parametrize("field, key", [("cars.job_pickup_date", "jp"), ("cars.job_return_date", "jr"),
                                        ("cars.pickup_date", "p"), ("cars.return_date", "r")])
def test_each_mapping_detects_its_own_mismatch(field, key) -> None:
    value = "2026-08-27" if key in ("jp", "p") else "2026-08-30"          # still a valid ordered period
    report = assess([detail(**{key: value})])
    assert agreement(report, field) == {"mismatch": 1}
    assert all(agreement(report, t) == {"match": 1} for t in ALL if t != field)
    assert report.blocking_reasons == (RB.RENTAL_DATE_PARENT_DETAIL_MISMATCH,) and report.eligible_detail_rows == 0


def test_multiple_mismatches_are_all_counted() -> None:
    report = assess([detail(jp="2026-08-27", r="2026-08-30")])
    assert report.mismatches == 2 and report.eligible_detail_rows == 0


@pytest.mark.parametrize("parent, row, status", [
    (("", NEXT), detail(), AS.PARENT_MISSING),
    ((DAY, NEXT), detail(jp="", p=""), AS.DETAIL_MISSING),
    (("", NEXT), detail(jp="", p=""), AS.BOTH_MISSING),              # both missing never passes
    (("2026-8-28", NEXT), detail(), AS.PARENT_INVALID),
    ((DAY, NEXT), detail(jp="2026-08-28T00:00:00"), AS.DETAIL_INVALID),
])
def test_missing_and_invalid_sides_are_unassessable_not_matches(parent, row, status) -> None:
    report = assess([row], parent={"SYNTH-JOB-1": parent})
    assert agreement(report, "cars.job_pickup_date") == {status.value: 1}
    assert RB.RENTAL_DATE_AGREEMENT_UNASSESSABLE in report.blocking_reasons and not report.is_valid
    assert report.eligible_detail_rows == 0


def test_parsed_dates_decide_and_non_iso_text_never_agrees() -> None:
    # Identical raw text that is not ISO never agrees, even if another parser could read it.
    report = assess([detail(jp="28/08/2026")], parent={"SYNTH-JOB-1": ("28/08/2026", NEXT)})
    assert agreement(report, "cars.job_pickup_date") == {AS.PARENT_INVALID.value: 1}
    # Similar-looking but non-ISO detail text against a valid parent is invalid, not a match.
    report = assess([detail(p="2026-8-28")])
    assert agreement(report, "cars.pickup_date") == {AS.DETAIL_INVALID.value: 1}
    # Equal parsed dates match; the comparison is on dates (the parent and detail objects are distinct).
    j, c = build([detail()])
    assert c.loc[0, "job_pickup_date"] == j.loc[0, "pickup_date"]
    assert agreement(assess([detail()]), "cars.job_pickup_date") == {"match": 1}


def test_orphans_untrusted_and_stale_linkage_never_pass() -> None:
    rows = [detail(), detail(job="SYNTH-JOB-2")]
    j, c = build(rows)
    result = link(j, c)
    orphaned = result.cars.copy()
    orphaned.loc[orphaned.index[1], JOB_LINKAGE_KEY_COLUMN] = "SYNTH-JOB-404"   # same row count, no parent
    report = assess_rental_dates(result.jobs, orphaned, policy=POLICY, job_linkage=result.report)
    assert agreement(report, "cars.pickup_date") == {"match": 1, "orphan_detail": 1}
    assert RB.RENTAL_DATE_AGREEMENT_UNASSESSABLE in report.blocking_reasons and report.eligible_detail_rows == 1
    stale = link(*build([detail()])).report                                        # evidence of other frames
    from ql2_sixt_canada_analysis.job_linkage import JobLinkageBlocker

    invalid_report = dataclasses.replace(result.report, collision_count=1,               # not a valid linkage
                                         blocking_reasons=(next(iter(JobLinkageBlocker)),))
    assert not invalid_report.is_valid
    for linkage in (None, stale, invalid_report):
        report = assess_rental_dates(result.jobs, result.cars, policy=POLICY, job_linkage=linkage)
        assert not report.linkage_trusted and agreement(report, "cars.pickup_date") == {"linkage_untrusted": 2}
        assert RB.RENTAL_DATE_AGREEMENT_UNASSESSABLE in report.blocking_reasons
        assert report.eligible_detail_rows == 0 and report.validity_holds         # validity is still assessed
    # A raw orphan makes the linkage itself invalid: every agreement is then unassessable.
    j1, c1 = build([detail()])
    orphan_raw = pd.concat([c1, c1.assign(job_id="SYNTH-JOB-404", row_index="1")], ignore_index=True).astype(
        dict(DATASET_DEFINITIONS[CARS].identifier_dtypes))
    raw_orphan = link(j1, orphan_raw)
    assert not raw_orphan.report.is_valid
    report = assess_rental_dates(raw_orphan.jobs, raw_orphan.cars, policy=POLICY, job_linkage=raw_orphan.report)
    assert agreement(report, "cars.return_date") == {"linkage_untrusted": 2} and report.eligible_detail_rows == 0


def test_disagreements_are_not_repaired_and_raw_columns_stay_unchanged() -> None:
    j, c = build([detail(p="2026-08-27"), detail(job="SYNTH-JOB-2", r="")])
    result = link(j, c)
    before = (result.jobs.copy(deep=True), result.cars.copy(deep=True))
    report = assess_rental_dates(result.jobs, result.cars, policy=POLICY, job_linkage=result.report)
    derived = derive_rental_periods(result.jobs, result.cars, policy=POLICY, job_linkage=result.report)
    pd.testing.assert_frame_equal(result.jobs, before[0]), pd.testing.assert_frame_equal(result.cars, before[1])
    assert not set(derived.cars.columns) & set(result.cars.columns)               # derived values are separate
    assert report.mismatches == 1 and report.missing_values == 1
    assert derived.cars["rental_dates_agree"].tolist() == [False, False]


# =================================================================== pricing eligibility


def _derived(rows, parent=None):  # type: ignore[no-untyped-def]
    j, c = build(rows, parent)
    result = link(j, c)
    return derive_rental_periods(result.jobs, result.cars, policy=POLICY, job_linkage=result.report), \
        assess_rental_dates(result.jobs, result.cars, policy=POLICY, job_linkage=result.report)


def test_eligibility_follows_validity_and_agreement_only() -> None:
    rows = [detail(job="SYNTH-JOB-1", jr=DAY, r=DAY),                 # same-day: eligible
            detail(job="SYNTH-JOB-2", jr="2027-08-28", r="2027-08-28"),  # long: eligible
            detail(job="SYNTH-JOB-3"),                                 # parent return before pickup: ineligible
            detail(job="SYNTH-JOB-4", p=""),                           # missing required date: ineligible
            detail(job="SYNTH-JOB-5", jp="2026-08-27")]                # mismatch: ineligible
    parent = {"SYNTH-JOB-1": (DAY, DAY), "SYNTH-JOB-2": (DAY, "2027-08-28"), "SYNTH-JOB-3": (NEXT, DAY),
              "SYNTH-JOB-4": (DAY, NEXT), "SYNTH-JOB-5": (DAY, NEXT)}
    derived, report = _derived(rows, parent)
    assert derived.cars["pricing_eligible"].tolist() == [True, True, False, False, False]
    assert (report.detail_rows, report.eligible_detail_rows) == (5, 2)            # every row still counted
    assert report.parent_rows == 5 and all(p.rows == 5 for p in report.periods)
    assert not report.is_valid                                                     # never a false all-valid


def test_an_analysis_cohort_is_separate_from_validity() -> None:
    rows = [detail(job="SYNTH-JOB-1", jr=DAY, r=DAY), detail(job="SYNTH-JOB-2", jr="2027-08-28", r="2027-08-28"),
            detail(job="SYNTH-JOB-3")]
    parent = {"SYNTH-JOB-1": (DAY, DAY), "SYNTH-JOB-2": (DAY, "2027-08-28"), "SYNTH-JOB-3": (NEXT, DAY)}
    derived, report = _derived(rows, parent)
    blockers = report.blocking_reasons
    one_day = analysis_duration_cohort(derived.cars, minimum_days=0, maximum_days=1)
    assert one_day.tolist() == [True, False, False]                       # long rental outside the study, still valid
    assert analysis_duration_cohort(derived.cars, minimum_days=0, maximum_days=None).tolist() == [True, True, False]
    assert report.blocking_reasons == blockers and derived.cars["pricing_eligible"].tolist() == [True, True, False]
    with pytest.raises(ValueError):
        analysis_duration_cohort(derived.cars, minimum_days=3, maximum_days=1)


# ======================================================================== readiness


def pricing(rental):  # type: ignore[no-untyped-def]
    return assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                    **(GATES | {"rental_dates": rental}))


def test_the_rental_gate_passes_only_with_an_approved_policy_and_clean_dates() -> None:
    assert pricing(GATES["rental_dates"]).ready                            # the synthetic gate set passes
    assert PB.RENTAL_DATE_ASSESSMENT_MISSING in pricing(None).blocking_reasons
    for status in (RS.NOT_APPROVED, RS.RECORD_UNAVAILABLE, RS.INVALID):
        blocked = pricing(RentalDateReport(policy=RentalDatePolicy(status=status)))
        assert blocked.blocking_reasons == (PB.RENTAL_DATE_RULES_UNAVAILABLE,) and not blocked.ready
    for decision in RENTAL:                                                 # either decision PROPOSED keeps it closed
        record = parse_decision_record(unapprove(v7(), decision))
        assert rental_date_policy_from_record(record).status is RS.NOT_APPROVED


@pytest.mark.parametrize("rows, parent, blocker", [
    ([detail(p="2026/08/28")], None, PB.RENTAL_DATE_FORMAT_INVALID),
    ([detail(r="")], None, PB.RENTAL_DATE_REQUIRED_VALUE_MISSING),
    ([detail(jr=DAY, r=DAY)], {"SYNTH-JOB-1": (NEXT, DAY)}, PB.RENTAL_DATE_ORDERING_INVALID),
    ([detail(jp="2026-08-27")], None, PB.RENTAL_DATE_PARENT_DETAIL_MISMATCH),
    ([detail(jp="")], None, PB.RENTAL_DATE_AGREEMENT_UNASSESSABLE),
])
def test_rental_failures_block_central_readiness(rows, parent, blocker) -> None:
    readiness = pricing(assess(rows, parent))
    assert blocker in readiness.blocking_reasons and not readiness.ready


@pytest.mark.parametrize("end", [DAY, "2029-12-31"])
def test_same_day_and_long_rentals_do_not_block(end) -> None:
    readiness = pricing(assess([detail(jr=end, r=end)], {"SYNTH-JOB-1": (DAY, end)}))
    assert readiness.ready and readiness.rental_dates_valid and readiness.rental_date_rules_available


def test_unrelated_blockers_remain_and_the_project_is_not_ready() -> None:
    readiness = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                         **(GATES | {"temporal_fields_trusted": False}))
    assert readiness.blocking_reasons == (PB.TEMPORAL_FIELDS_UNTRUSTED,)
    assert passing_rental_report(*scheduled_frames()).is_valid and synthetic_rental_policy().available
    assert "pickup_date" in DATASET_DEFINITIONS[JOBS].columns and with_rental_dates is not None


# ====================================================================== baseline, docs


def test_baseline_reports_rental_dates_in_aggregate() -> None:
    import json

    from test_collection_schedule import pricing_with, real_frames, real_report

    from ql2_sixt_canada_analysis.pricing_baseline import build_pricing_baseline, render_baseline_markdown
    from ql2_sixt_canada_analysis.stability import assess_vehicle_attribute_stability

    j, c = real_frames()
    j, c = with_rental_dates(j, c, "2026-08-28", "2026-08-28")
    c.loc[c.index[0], "pickup_date"] = "2026-08-27"
    result = link(j, c)
    rental = assess_rental_dates(result.jobs, result.cars, policy=POLICY, job_linkage=result.report)
    _, readiness = pricing_with(j, c, real_report(j, c), rental_dates=rental)
    stable = assess_vehicle_attribute_stability(c.assign(car_name="SYNTH Vehicle"))
    baseline = build_pricing_baseline(pricing=readiness, jobs=j, cars=c, temporal=None, vehicle_stability=stable)
    summary = baseline.rental_dates
    assert (summary.policy_status, summary.maximum_duration_mode, summary.parent_rows, summary.detail_rows) == (
        "approved", "unbounded", 270, len(c))
    assert summary.blockers == ("rental_date_parent_detail_mismatch",) and summary.eligible_detail_rows == len(c) - 1
    assert baseline.plan_gaps == () and "rental_date_parent_detail_mismatch" in baseline.pricing_blockers
    assert next(p for p in summary.periods if p.period == PARENT).same_day == 270
    markdown = render_baseline_markdown(baseline, commit="abc1234", date="2026-10-06")
    assert "### Rental dates" in markdown and "(informational)" in markdown
    assert not re.search(r"\d{4}-\d{2}-\d{2}(?!\b)|SYNTH", markdown.replace("2026-10-06", ""))
    json.dumps(baseline.to_dict())


def test_governance_document_records_the_supplied_decisions_only() -> None:
    raw = (ROOT / GOVERNANCE).read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for phrase in ("2026-10-06", "collection owner and business owner (joint)", "ISO_8601_DATE", "`YYYY-MM-DD`",
                   "Both are required", "greater than or equal to", "Equality is permitted", "minimum is zero",
                   "no maximum", "Raw values are preserved", "parsed calendar dates", "One-sided missing values fail",
                   "No automatic repair", "Data validity versus analysis eligibility",
                   "current dataset and subsequent collections until superseded", "new authority-record version",
                   "Not supplied", "pricing-authorities-v7"):
        assert phrase in text, phrase
    for target, source in MAPPINGS:
        assert f"| `{target}` | `{source}` |" in raw
    assert "@" not in raw and not re.search(r"#\d|[A-Z]{2,}-\d+", raw) and "SYNTH" not in raw
    assert not re.search(r"\d{4}-\d{2}-\d{2}", raw.replace("2026-10-06", ""))


def test_documentation_describes_the_rental_date_policy() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    records = " ".join((RECORD_DIR / "README.md").read_text(encoding="utf-8").split())
    for phrase in ("ISO_8601_DATE", "YYYY-MM-DD", "same-day", "no maximum", "analysis eligibility",
                   "cars.job_pickup_date", "rental_date_rules_unavailable", "rental_date_parent_detail_mismatch",
                   "never repaired", Path(GOVERNANCE).name):
        assert phrase in readme, phrase
    assert "Pickup/return dates have no temporal-contract rules" not in readme
    for phrase in ("v7.toml", "RENTAL_DATE_VALIDITY", "UNBOUNDED"):
        assert phrase in records, phrase
    assert "The dataset is **pricing ready** under the current authority record" in readme
