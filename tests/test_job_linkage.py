"""Unit tests for authority-backed job linkage and offer-position normalization.

Every identifier is fabricated (``SYNTH-...`` or short synthetic digit strings);
no real identifier is copied into tests. Frames follow the raw source contracts.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
from conftest import SYNTH_LINKAGE_POLICY

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis import identifiers, job_linkage
from ql2_sixt_canada_analysis.job_linkage import (
    JobLinkageBlocker as B,
    JobLinkageNotReadyError,
    JobLinkagePolicyStatus,
    JobLinkagePreconditionError,
    JobLinkageReport,
    assess_job_linkage,
    require_job_linkage,
)
from ql2_sixt_canada_analysis.schemas import (
    CARS_DEFINITION,
    CONFIDENTIAL_TECHNICAL_COLUMNS,
    DATASET_DEFINITIONS,
    JOB_LINKAGE_KEY_COLUMN as LK,
    JOBS_DEFINITION,
    OFFER_POSITION_KEY_COLUMN as PK,
    OFFER_POSITION_KEY_DTYPE,
    SOURCE_JOB_IDENTIFIER_COLUMN as JID,
    SOURCE_OFFER_POSITION_COLUMN as POS,
    DatasetKey,
)

POLICY = SYNTH_LINKAGE_POLICY
NO_REPAIR = dataclasses.replace(POLICY, legacy_decimal_zero_repair=False, legacy_offer_position_repair=False)


def jobs(*ids: object) -> pd.DataFrame:
    rows = [{c: (i if c == JID else f"SYNTH-{c}") for c in JOBS_DEFINITION.columns} for i in ids]
    return pd.DataFrame(rows, columns=list(JOBS_DEFINITION.columns)).astype(dict(JOBS_DEFINITION.identifier_dtypes))


def cars(*details: tuple[object, object], position_dtype: object = object) -> pd.DataFrame:
    rows = [{c: (ref if c == JID else pos if c == POS else f"SYNTH-{c}") for c in CARS_DEFINITION.columns}
            for ref, pos in details]
    frame = pd.DataFrame(rows, columns=list(CARS_DEFINITION.columns)).astype(dict(CARS_DEFINITION.identifier_dtypes))
    frame[POS] = pd.Series([pos for _, pos in details], dtype=position_dtype)
    return frame


def run(j: pd.DataFrame, c: pd.DataFrame, policy=POLICY):  # type: ignore[no-untyped-def]
    return assess_job_linkage(j, c, policy)


def keys(result) -> list:  # type: ignore[no-untyped-def]
    return [None if v is pd.NA else v for v in result.cars[LK].astype(object).tolist()]


def offsets(result) -> list:  # type: ignore[no-untyped-def]
    return [None if v is pd.NA else int(v) for v in result.cars[PK].astype(object).tolist()]


# ------------------------------------------------------------- exact matching


def test_exact_digit_only_matches_link_unchanged() -> None:
    r = run(jobs("101", "102"), cars(("101", 0), ("101", 1), ("102", 0)))
    assert r.is_valid and r.report.blocking_reasons == ()
    assert keys(r) == ["101", "101", "102"] and r.report.exact_match_count == 3
    assert r.report.decimal_zero_repair_count == 0 and r.report.linked_detail_count == 3
    assert r.jobs[LK].tolist() == ["101", "102"]


@pytest.mark.parametrize("identifier", ["007", "0", "000123", "SYNTH-ABC", "synth-abc", "A.B/C_D:E", "x y",
                                        "1e5", "+12", "-12", "12.5", "١٢", "9" * 40, "SYNTH.0", "12.0"])
def test_opaque_identifiers_link_only_by_exact_text(identifier: str) -> None:
    r = run(jobs(identifier), cars((identifier, 0)))
    assert r.is_valid and keys(r) == [identifier] and r.report.exact_match_count == 1
    assert r.cars[JID].tolist() == [identifier]                          # raw value untouched


def test_leading_zeros_are_significant() -> None:
    r = run(jobs("7", "007"), cars(("007", 0), ("7", 0)))
    assert r.is_valid and keys(r) == ["007", "7"]
    unmatched = run(jobs("7"), cars(("007", 0)))
    assert unmatched.report.unmatched_identifier_count == 1 and keys(unmatched) == [None]


def test_very_long_identifiers_are_never_numeric() -> None:
    long = "1" + "0" * 30
    r = run(jobs(long, "1" + "0" * 29 + "1"), cars((long + ".0", 0)))
    assert r.is_valid and keys(r) == [long]                               # no float rounding merges them


def test_exact_legitimate_identifier_ending_in_decimal_zero_is_not_rewritten() -> None:
    r = run(jobs("SYNTH-5.0", "55.0"), cars(("SYNTH-5.0", 0), ("55.0", 0)))
    assert r.is_valid and keys(r) == ["SYNTH-5.0", "55.0"]
    assert r.report.exact_match_count == 2 and r.report.decimal_zero_repair_count == 0


# -------------------------------------------------------------- legacy repair


def test_decimal_zero_detail_value_repairs_to_its_unique_parent() -> None:
    r = run(jobs("4321", "8765"), cars(("4321.0", 0), ("4321.0", 1), ("8765.0", 0)))
    assert r.is_valid and keys(r) == ["4321", "4321", "8765"]
    assert r.report.decimal_zero_repair_count == 3 and r.report.exact_match_count == 0
    assert r.cars[JID].tolist() == ["4321.0", "4321.0", "8765.0"]             # raw kept


def test_leading_zero_decimal_zero_repair_keeps_the_zeros() -> None:
    r = run(jobs("00042", "42"), cars(("00042.0", 0)))
    assert r.is_valid and keys(r) == ["00042"]


def test_exact_and_repaired_candidates_for_different_parents_are_ambiguous() -> None:
    r = run(jobs("77", "77.0"), cars(("77.0", 0)))
    assert not r.is_valid and keys(r) == [None]
    assert r.report.ambiguous_identifier_count == 1 and B.DETAIL_REFERENCE_AMBIGUOUS in r.report.blocking_reasons


@pytest.mark.parametrize("detail", ["SYNTH9.0", "9.00", "9.0.0", "9.5", "9.", ".0", " 9.0", "9.0 ", "+9.0",
                                    "-9.0", "9e0", "9E0", "٩.0", "9.0\n"])
def test_only_ascii_digits_plus_one_decimal_zero_are_repaired(detail: str) -> None:
    parents = jobs("9", "SYNTH9", "9.0.", "+9", "-9")
    r = run(parents, cars((detail, 0)))
    assert keys(r) == [None] and r.report.unmatched_identifier_count == 1
    assert B.DETAIL_REFERENCE_UNMATCHED in r.report.blocking_reasons


def test_repair_is_disabled_without_the_approved_equivalence() -> None:
    r = run(jobs("31"), cars(("31.0", 0)), NO_REPAIR)
    assert keys(r) == [None] and r.report.unmatched_identifier_count == 1


def test_mixed_raw_forms_of_one_parent_collide_and_block() -> None:
    r = run(jobs("64"), cars(("64", 0), ("64.0", 1)))
    assert not r.is_valid and keys(r) == [None, None]
    assert r.report.collision_count == 2 and r.report.blocking_reasons == (B.LINKAGE_COLLISION,)


# ---------------------------------------------------------- invalid references


@pytest.mark.parametrize("value", [None, "", " ", "\t", "  \n "])
def test_missing_and_whitespace_only_detail_references_are_invalid(value: object) -> None:
    r = run(jobs("11", " "), cars(("11", 0), (value, 1)))
    assert keys(r)[1] is None and r.report.missing_identifier_count == 1
    assert B.DETAIL_REFERENCE_MISSING in r.report.blocking_reasons and not r.is_valid


@pytest.mark.parametrize("value", [None, " ", ""])
def test_missing_and_whitespace_only_parent_keys_are_invalid(value: object) -> None:
    r = run(jobs("11", value), cars(("11", 0)))
    assert r.jobs[LK].isna().tolist() == [False, True]
    assert r.report.parent_key_invalid_count == 1 and B.PARENT_KEY_INVALID in r.report.blocking_reasons


def test_unmatched_reference_blocks() -> None:
    r = run(jobs("11"), cars(("12", 0)))
    assert r.report.unmatched_identifier_count == 1 and r.report.blocking_reasons == (B.DETAIL_REFERENCE_UNMATCHED,)


def test_duplicate_parent_keys_block_and_make_references_ambiguous() -> None:
    r = run(jobs("21", "21", "22"), cars(("21", 0), ("21.0", 1), ("22", 0)))
    assert r.report.parent_key_duplicate_row_count == 2
    assert {B.PARENT_KEY_NOT_UNIQUE, B.DETAIL_REFERENCE_AMBIGUOUS} <= set(r.report.blocking_reasons)
    assert keys(r) == [None, None, "22"]
    assert r.jobs[LK].tolist() == ["21", "21", "22"]         # kept, so the jobs key contract also fails


@pytest.mark.parametrize("detail, parent", [(" 5", "5"), ("5 ", "5"), ("synth", "SYNTH"), ("+5", "5"),
                                            ("5e0", "5"), ("5.5", "5"), ("05", "5"), ("5", "05")])
def test_no_trimming_case_folding_sign_exponent_or_fraction_rewriting(detail: str, parent: str) -> None:
    r = run(jobs(parent), cars((detail, 0)))
    assert keys(r) == [None] and r.report.unmatched_identifier_count == 1


def test_identifiers_are_never_passed_through_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("numeric conversion of identifiers")
    monkeypatch.setattr(pd, "to_numeric", forbidden)
    r = run(jobs("0099"), cars(("0099.0", 0)))
    assert keys(r) == ["0099"]


# ----------------------------------------------------------- offer positions


@pytest.mark.parametrize("value, expected", [(0, 0), (5, 5), (np.int64(3), 3), ("0", 0), ("12", 12), ("007", 7),
                                             (2.0, 2), (np.float64(4.0), 4), (-0.0, 0)])
def test_valid_offer_positions(value: object, expected: int) -> None:
    r = run(jobs("1"), cars(("1", value)))
    assert r.is_valid and offsets(r) == [expected]
    assert r.report.valid_row_index_count == 1 and r.report.legacy_row_index_repair_count == 0


@pytest.mark.parametrize("value, expected", [("0.0", 0), ("3.0", 3), ("0010.0", 10)])
def test_legacy_decimal_zero_offer_positions(value: str, expected: int) -> None:
    r = run(jobs("1"), cars(("1", value)))
    assert r.is_valid and offsets(r) == [expected]
    assert r.report.legacy_row_index_repair_count == 1 and r.report.valid_row_index_count == 1
    assert r.cars[POS].tolist() == [value]                                   # raw kept
    blocked = run(jobs("1"), cars(("1", value)), NO_REPAIR)
    assert blocked.report.invalid_row_index_count == 1 and not blocked.is_valid


@pytest.mark.parametrize("dtype", ["int64", "Int64", "float64", object])
def test_offer_position_key_is_nullable_integer(dtype: object) -> None:
    r = run(jobs("1"), cars(("1", 0), ("1", 1), position_dtype=dtype))
    assert r.cars[PK].dtype == OFFER_POSITION_KEY_DTYPE and offsets(r) == [0, 1]
    assert r.cars[LK].dtype == "string" and r.jobs[LK].dtype == "string"


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, ""])
def test_missing_offer_positions_block(value: object) -> None:
    r = run(jobs("1"), cars(("1", value)))
    assert offsets(r) == [None] and r.report.missing_row_index_count == 1
    assert r.report.blocking_reasons == (B.OFFER_POSITION_MISSING,)


@pytest.mark.parametrize("value", [-1, -1.0, 1.5, "1.5", "1.00", math.inf, -math.inf, "inf", "nan", "1e2", "1E2",
                                   "+1", "-1", " 1", "1 ", "SYNTH", "one", True, False, "0x1", "١", 2.0 ** 70])
def test_invalid_offer_positions_block(value: object) -> None:
    r = run(jobs("1"), cars(("1", value)))
    assert offsets(r) == [None] and r.report.invalid_row_index_count == 1
    assert r.report.blocking_reasons == (B.OFFER_POSITION_INVALID,)


def test_boolean_offer_position_column_is_invalid() -> None:
    r = run(jobs("1"), cars(("1", True), ("1", False), position_dtype=bool))
    assert r.report.invalid_row_index_count == 2 and not r.is_valid


def test_duplicate_offer_positions_are_left_to_the_key_contract() -> None:
    r = run(jobs("1"), cars(("1", 0), ("1", "0.0")))
    assert r.is_valid and offsets(r) == [0, 0]                 # derived; the analytical key contract fails later


def test_offer_positions_are_never_routed_through_the_identifier_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = job_linkage._resolve
    monkeypatch.setattr(job_linkage, "_resolve", lambda v, *a: calls.append(v) or original(v, *a))
    run(jobs("1"), cars(("1", "5.0")))
    assert calls == ["1"]


# --------------------------------------------------- policy, purity, safety


def test_unavailable_policy_yields_no_keys_and_blocks() -> None:
    r = run(jobs("1"), cars(("1", 0)), None)
    assert r.report.policy_status is JobLinkagePolicyStatus.UNAVAILABLE and not r.report.policy_available
    assert r.report.blocking_reasons == (B.POLICY_UNAVAILABLE,) and not r.is_valid
    assert r.jobs[LK].isna().all() and r.cars[LK].isna().all() and r.cars[PK].isna().all()
    with pytest.raises(JobLinkageNotReadyError) as info:
        require_job_linkage(jobs("1"), cars(("1", 0)), None)
    assert info.value.blocking_reasons == (B.POLICY_UNAVAILABLE,)


def test_policy_argument_is_required_and_typed() -> None:
    with pytest.raises(TypeError):
        assess_job_linkage(jobs("1"), cars(("1", 0)))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        assess_job_linkage(jobs("1"), cars(("1", 0)), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess_job_linkage([], cars(("1", 0)), POLICY)  # type: ignore[arg-type]


def test_inputs_are_never_modified() -> None:
    j, c = jobs("0042", "9"), cars(("0042.0", "1.0"), ("9", 2))
    j_before, c_before = j.copy(deep=True), c.copy(deep=True)
    r = run(j, c)
    pd.testing.assert_frame_equal(j, j_before)
    pd.testing.assert_frame_equal(c, c_before)
    assert list(r.cars.columns) == [*c.columns, LK, PK] and list(r.jobs.columns) == [*j.columns, LK]
    pd.testing.assert_frame_equal(r.cars[list(c.columns)], c)               # source columns unchanged
    pd.testing.assert_index_equal(r.cars.index, c.index)
    leaked = r.cars
    leaked.loc[0, LK] = "SYNTH-LEAK"
    assert r.cars.loc[0, LK] == "0042"                                       # results are copies


def test_repeated_application_is_deterministic_and_idempotent() -> None:
    j, c = jobs("0042", "9"), cars(("0042.0", "1.0"), ("9", 2), ("9", 3))
    first = run(j, c)
    again = run(first.jobs, first.cars)
    pd.testing.assert_frame_equal(first.jobs, again.jobs)
    pd.testing.assert_frame_equal(first.cars, again.cars)
    assert first.report == again.report == run(j, c).report
    shuffled = run(j.iloc[::-1], c.iloc[::-1])
    assert shuffled.report == first.report


def test_preconditions_fail_closed() -> None:
    with pytest.raises(JobLinkagePreconditionError) as info:
        run(jobs("1").drop(columns=[JID]), cars(("1", 0)))
    assert info.value.reason == "source_column_missing"
    with pytest.raises(JobLinkagePreconditionError) as info:
        run(jobs("1"), cars(("1", 0)).astype({JID: object}))
    assert info.value.reason == "identifier_dtype"
    c = cars(("1", 0))
    blank = pd.DataFrame([[None] * c.shape[1]], columns=c.columns, dtype=object).astype(
        dict(CARS_DEFINITION.identifier_dtypes))
    with pytest.raises(JobLinkagePreconditionError) as info:
        run(jobs("1"), pd.concat([c, blank], ignore_index=True))
    assert info.value.reason == "blank_rows_present"


def test_empty_frames_are_vacuously_valid() -> None:
    r = run(jobs(), cars())
    assert r.is_valid and r.report.parent_row_count == 0 and r.report.detail_row_count == 0


def test_reports_and_errors_never_contain_identifier_values() -> None:
    secret = "SYNTHSECRET0042"
    r = run(jobs(secret, secret, "1"), cars((secret, 0), (secret + ".0", "x"), ("SYNTHOTHER", 1)))
    for text in (repr(r.report), str(r.report), str(dataclasses.asdict(r.report))):
        assert "SYNTH" not in text and "0042" not in text
    with pytest.raises(JobLinkageNotReadyError) as info:
        require_job_linkage(jobs(secret), cars((secret + "X", 0)), POLICY)
    assert "SYNTH" not in str(info.value) and not any(ch.isdigit() for ch in str(info.value))
    assert all(type(getattr(r.report, f.name)) in (int, JobLinkagePolicyStatus, tuple)
               for f in dataclasses.fields(JobLinkageReport))


def test_blocker_values_name_no_source_columns() -> None:
    columns = {c for key in DatasetKey for c in DATASET_DEFINITIONS[key].columns}
    assert not any(c in b.value for b in B for c in columns)


def test_strict_api_returns_the_result_when_valid() -> None:
    result = require_job_linkage(jobs("1"), cars(("1.0", "0.0")), POLICY)
    assert result.is_valid and keys(result) == ["1"]


def test_identifier_module_still_owns_typing_only() -> None:
    assert not hasattr(identifiers, "assess_job_linkage")
    assert "linkage" not in identifiers.__doc__.lower() or "never" in identifiers.__doc__.lower()


def test_confidential_technical_fields_and_exports() -> None:
    assert CONFIDENTIAL_TECHNICAL_COLUMNS == (JID, LK, POS, PK)
    for name in ("assess_job_linkage", "require_job_linkage", "job_linkage_policy_from_record",
                 "load_job_linkage_policy", "JobLinkageReport", "JobLinkagePolicy", "JobLinkageBlocker",
                 "ANALYSIS_JOB_DETAIL_RELATIONSHIP", "JOB_LINKAGE_KEY_COLUMN", "OFFER_POSITION_KEY_COLUMN"):
        assert name in ql2_sixt_canada_analysis.__all__


def test_every_linkage_blocker_has_a_pricing_blocker_of_the_same_value() -> None:
    from ql2_sixt_canada_analysis.join_readiness import JobDetailJoinBlocker
    from ql2_sixt_canada_analysis.readiness import PricingBlocker

    pricing = {b.value for b in PricingBlocker}
    assert {b.value for b in B} <= pricing and {b.value for b in JobDetailJoinBlocker} <= pricing


def test_record_validator_runs_as_a_module_without_reimport_warnings() -> None:
    import subprocess
    import sys
    from pathlib import Path

    record = Path(__file__).resolve().parents[1] / "docs" / "decisions" / "pricing_authorities" / "v2.toml"
    done = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-m",
                           "ql2_sixt_canada_analysis.authority_decisions", str(record)],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0 and done.stdout.startswith("VALID\n") and "RuntimeWarning" not in done.stderr
