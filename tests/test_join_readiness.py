"""Tests for the fail-closed trusted jobs-to-details join.

Frames are built from the central contracts with fabricated values
(``SYNTH-JOB-001``, offer positions 0, 1, ...). Each negative fixture breaks
exactly the contract named in the test.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.join_readiness import (
    JobDetailJoinBlocker as B,
    JobDetailJoinReadiness,
    UntrustedJoinError,
    assess_job_detail_join_readiness as assess,
    require_trusted_job_detail_join,
)
from ql2_sixt_canada_analysis.relationships import join_jobs_to_details
from ql2_sixt_canada_analysis.schemas import DATASET_DEFINITIONS, JOB_DETAIL_RELATIONSHIP, DatasetKey

REL = JOB_DETAIL_RELATIONSHIP
JOBS_DEF, CARS_DEF = REL.parent_definition, REL.detail_definition
PK, = REL.parent_key_columns
DK, = REL.detail_key_columns
COUNT = REL.expected_detail_count_column
POSITION, = CARS_DEF.non_identifier_key_columns          # the detail key's offer-position component
J1, J2, J3, ORPHAN = "SYNTH-JOB-001", "SYNTH-JOB-002", "SYNTH-JOB-003", "SYNTH-JOB-999"


def _frame(definition, rows: list[dict]) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    df = pd.DataFrame([{c: r.get(c, f"SYNTH-{c}") for c in definition.columns} for r in rows],
                      columns=list(definition.columns))
    return df.astype(dict(definition.identifier_dtypes))


def jobs(*declared: tuple[object, int]) -> pd.DataFrame:
    """(job key, declared detail count) per job."""
    return _frame(JOBS_DEF, [{PK: key, COUNT: count} for key, count in declared])


def cars(*details: tuple[object, int]) -> pd.DataFrame:
    """(job key, offer position) per detail row."""
    return _frame(CARS_DEF, [{DK: key, POSITION: pos} for key, pos in details])


def valid_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return jobs((J1, 2), (J2, 1), (J3, 0)), cars((J1, 0), (J1, 1), (J2, 0))


def assert_blocked(r: JobDetailJoinReadiness, *expected: B) -> None:
    assert r.join_ready is False and r.trusted_jobs_with_details is None
    assert set(r.blocking_reasons) == set(expected) and len(r.blocking_reasons) == len(expected)


# ------------------------------------------------------------------ valid join


def test_fully_valid_inputs_produce_the_trusted_join():
    j, c = valid_inputs()
    r = assess(j, c)
    assert r.join_ready is True and r.blocking_reasons == ()
    assert (r.jobs_key_contract_valid, r.details_key_contract_valid, r.all_key_contracts_valid,
            r.declared_counts_reconciled, r.relationship_contract_valid, r.all_reports_available) == (True,) * 6
    joined = r.trusted_jobs_with_details
    assert joined is not None and r.diagnostic_jobs_with_details is None
    # 3 detail rows + 1 job without details; each detail exactly once, nothing dropped or duplicated.
    assert len(joined) == 4 and len(joined) == r.relationship_report.expected_left_join_row_count
    linked = joined.loc[joined[POSITION].notna()]
    assert sorted(zip(linked[PK], linked[POSITION].astype(int))) == [(J1, 0), (J1, 1), (J2, 0)]
    assert joined[PK].tolist().count(J3) == 1
    assert {PK, COUNT, POSITION} <= set(joined.columns)
    shared = (set(JOBS_DEF.columns) & set(CARS_DEF.columns)) - {PK}
    assert all(f"{c}{REL.parent_suffix}" in joined and f"{c}{REL.detail_suffix}" in joined for c in shared)
    pd.testing.assert_frame_equal(joined, join_jobs_to_details(j, c).joined)   # same validated merge


def test_require_trusted_join_returns_the_frame_for_valid_inputs():
    joined = require_trusted_job_detail_join(*valid_inputs())
    assert isinstance(joined, pd.DataFrame) and len(joined) == 4


def test_empty_inputs_are_vacuously_ready():
    r = assess(jobs(), cars())
    assert r.join_ready and len(r.trusted_jobs_with_details) == 0


# ------------------------------------------------------------ single failures


def test_duplicate_detail_key_blocks_trust_although_relationship_passes():
    # Reproduces the original defect: the relationship contract alone passes.
    j, c = jobs((J1, 2)), cars((J1, 0), (J1, 0))
    r = assess(j, c)
    assert r.relationship_contract_valid is True and r.declared_counts_reconciled is True
    assert r.details_key_contract_valid is False and r.all_key_contracts_valid is False
    assert r.jobs_key_contract_valid is True
    assert_blocked(r, B.DETAILS_KEY_CONTRACT_FAILED)
    diagnostic = r.diagnostic_jobs_with_details
    assert diagnostic is not None and len(diagnostic) == 2      # untrusted, investigation only


def test_duplicate_jobs_key_blocks_trust_even_though_a_raw_merge_runs():
    j, c = jobs((J1, 1), (J1, 1)), cars((J1, 0))
    assert len(pd.merge(j, c, left_on=PK, right_on=DK, how="left")) == 2   # technically executable
    r = assess(j, c)
    assert r.jobs_key_contract_valid is False and r.all_key_contracts_valid is False
    assert r.relationship_report is None and r.reconciliation_report is None   # unassessable -> unavailable
    assert r.relationship_contract_valid is False and r.declared_counts_reconciled is False
    assert_blocked(r, B.JOBS_KEY_CONTRACT_FAILED, B.REQUIRED_REPORT_UNAVAILABLE)
    assert r.diagnostic_jobs_with_details is None


def test_declared_count_mismatch_blocks_trust():
    j, c = jobs((J1, 3), (J2, 1)), cars((J1, 0), (J1, 1), (J2, 0))
    r = assess(j, c)
    assert r.all_key_contracts_valid and r.relationship_contract_valid
    assert r.declared_counts_reconciled is False
    assert_blocked(r, B.DECLARED_COUNTS_NOT_RECONCILED)
    assert r.diagnostic_jobs_with_details is not None


def test_orphan_detail_blocks_trust_and_is_never_dropped_into_a_frame():
    j, c = jobs((J1, 1)), cars((J1, 0), (ORPHAN, 0))
    r = assess(j, c)
    assert r.relationship_contract_valid is False and r.relationship_report.orphan_detail_row_count == 1
    assert_blocked(r, B.RELATIONSHIP_CONTRACT_FAILED, B.ORPHAN_DETAILS_PRESENT, B.DECLARED_COUNTS_NOT_RECONCILED)
    assert r.diagnostic_jobs_with_details is None


def test_missing_link_detail_blocks_trust():
    j, c = jobs((J1, 1)), cars((J1, 0), (pd.NA, 1))
    r = assess(j, c)
    assert B.MISSING_LINK_DETAILS_PRESENT in r.blocking_reasons
    assert B.RELATIONSHIP_CONTRACT_FAILED in r.blocking_reasons and not r.join_ready


# ---------------------------------------------------------- multiple failures


def test_all_simultaneous_failures_are_reported():
    j, c = jobs((J1, 1)), cars((J1, 0), (J1, 0))                 # duplicate detail key + count mismatch
    assert_blocked(assess(j, c), B.DETAILS_KEY_CONTRACT_FAILED, B.DECLARED_COUNTS_NOT_RECONCILED)
    j, c = jobs((J1, 2)), cars((J1, 0), (J1, 0), (ORPHAN, 0))    # duplicate detail key + orphan
    assert_blocked(assess(j, c), B.DETAILS_KEY_CONTRACT_FAILED, B.RELATIONSHIP_CONTRACT_FAILED,
                   B.ORPHAN_DETAILS_PRESENT, B.DECLARED_COUNTS_NOT_RECONCILED)


# ------------------------------------------------------ unavailable prerequisites


def test_unassessable_identifier_types_fail_closed():
    j, c = valid_inputs()
    r = assess(j, c.astype({DK: object}))
    assert r.relationship_report is None and r.reconciliation_report is None
    assert B.REQUIRED_REPORT_UNAVAILABLE in r.blocking_reasons and not r.all_reports_available
    assert r.trusted_jobs_with_details is None and r.diagnostic_jobs_with_details is None


def test_blank_rows_make_reports_unavailable():
    j, c = valid_inputs()
    blank = pd.DataFrame([[None] * c.shape[1]], columns=c.columns, dtype=object).astype(
        dict(CARS_DEF.identifier_dtypes))
    r = assess(j, pd.concat([c, blank], ignore_index=True))
    assert B.REQUIRED_REPORT_UNAVAILABLE in r.blocking_reasons and not r.join_ready


def test_missing_parent_key_is_unavailable_not_passing():
    j, c = jobs((J1, 2), (pd.NA, 0)), cars((J1, 0), (J1, 1))
    r = assess(j, c)
    assert_blocked(r, B.JOBS_KEY_CONTRACT_FAILED, B.REQUIRED_REPORT_UNAVAILABLE)


# ------------------------------------------------- non-None is not sufficient


def test_executable_join_is_not_a_trusted_join():
    # The former notebook exposed the relationship-checked join whenever the
    # relationship passed; it exists here, but the trusted join does not.
    j, c = jobs((J1, 1)), cars((J1, 0), (J1, 0))
    relationship_join = join_jobs_to_details(j, c).joined
    assert relationship_join is not None and len(relationship_join) == 2
    r = assess(j, c)
    assert r.diagnostic_jobs_with_details is not None
    assert r.trusted_jobs_with_details is None and r.join_ready is False
    with pytest.raises(UntrustedJoinError) as info:
        require_trusted_job_detail_join(j, c)
    assert info.value.blocking_reasons == r.blocking_reasons
    assert "SYNTH" not in str(info.value) and not any(ch.isdigit() for ch in str(info.value))


@pytest.mark.parametrize("inputs", [
    lambda: (jobs((J1, 1)), cars((J1, 0), (J1, 0))),
    lambda: (jobs((J1, 3)), cars((J1, 0))),
    lambda: (jobs((J1, 1), (J1, 1)), cars((J1, 0))),
    lambda: (jobs((J1, 1)), cars((J1, 0), (ORPHAN, 0))),
])
def test_public_path_never_returns_a_frame_when_any_contract_fails(inputs):
    with pytest.raises(UntrustedJoinError):
        require_trusted_job_detail_join(*inputs())


# ------------------------------------------------- consistency and isolation


def test_outputs_are_isolated_from_inputs_and_callers():
    j, c = valid_inputs()
    j_before, c_before = j.copy(deep=True), c.copy(deep=True)
    r = assess(j, c)
    pd.testing.assert_frame_equal(j, j_before)
    pd.testing.assert_frame_equal(c, c_before)
    expected = r.trusted_jobs_with_details
    # Mutating the validated inputs afterwards cannot change the held trusted frame.
    j.loc[0, PK] = ORPHAN
    c.loc[1, POSITION] = 0
    pd.testing.assert_frame_equal(r.trusted_jobs_with_details, expected)
    # Mutating a returned copy cannot change the next one.
    leaked = r.trusted_jobs_with_details
    leaked.loc[0, PK] = ORPHAN
    pd.testing.assert_frame_equal(r.trusted_jobs_with_details, expected)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.blocking_reasons = ()  # type: ignore[misc]
    # Re-assessing the mutated inputs is what reflects the new state.
    assert not assess(j, c).join_ready


def test_reports_come_from_the_joined_frames():
    j, c = jobs((J1, 1)), cars((J1, 0), (J1, 0))
    r = assess(j, c)
    assert r.details_key_report.total_row_count == len(c) and r.jobs_key_report.total_row_count == len(j)
    assert r.relationship_report.detail_row_count == len(c) and r.reconciliation_report.job_count == len(j)


def test_assessment_is_deterministic():
    j, c = jobs((J1, 1)), cars((J1, 0), (J1, 0), (ORPHAN, 0))
    assert assess(j, c).blocking_reasons == assess(j, c).blocking_reasons


def test_type_errors():
    with pytest.raises(TypeError):
        assess([], cars())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess(jobs(), cars(), object())  # type: ignore[arg-type]


def test_blocker_values_name_no_source_columns():
    columns = {c for key in DatasetKey for c in DATASET_DEFINITIONS[key].columns}
    assert not any(c in b.value for b in B for c in columns)


def test_package_exports():
    for name in ("assess_job_detail_join_readiness", "require_trusted_job_detail_join",
                 "JobDetailJoinReadiness", "JobDetailJoinBlocker", "UntrustedJoinError"):
        assert name in ql2_sixt_canada_analysis.__all__
