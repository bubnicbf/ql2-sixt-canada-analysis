"""Completeness controls: complete-source loading, (city, branch) coverage,
job-based stream continuity and reconciliation of every declared count.

All values are fabricated (``SYNTH-JOB-001``, ``SYNTH-CITY-1``, small counts)
except the configured expected pairs, which are read from the central
contract and never written here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
from conftest import contract_columns, write_synthetic_csv

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis import ingestion
from ql2_sixt_canada_analysis.city_integrity import assess_city_integrity
from ql2_sixt_canada_analysis.coverage import assess_expected_location_coverage, location_pair_evidence
from ql2_sixt_canada_analysis.ingestion import (
    SAFE_ON_BAD_LINES,
    IncompleteSourceOptionError,
    IngestionError,
    RawDataLoadError,
    RawDatasets,
    load_raw_datasets,
)
from ql2_sixt_canada_analysis.quality import remove_blank_rows_from_raw_datasets
from ql2_sixt_canada_analysis.readiness import CompletenessBlocker as CB, assess_completeness
from ql2_sixt_canada_analysis.reconciliation import (
    assess_job_detail_reconciliation as reconcile,
    job_detail_count_results,
)
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    EXPECTED_LOCATION_COVERAGE,
    JOB_DETAIL_RELATIONSHIP,
    DatasetKey,
    LocationCoverageMode,
)
from ql2_sixt_canada_analysis.streams import (
    assess_expected_location_streams,
    LocationStreamStatus as S,
    PipelineStage as P,
    StreamContinuity,
    investigate_location_stream,
)

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
REL = JOB_DETAIL_RELATIONSHIP
PK, = REL.parent_key_columns
DK, = REL.detail_key_columns
PRIMARY, SECONDARY = REL.expected_detail_count_columns
POSITION, = REL.detail_definition.non_identifier_key_columns
COV = EXPECTED_LOCATION_COVERAGE
CITY_COL, LABEL_COL = COV.location_columns
PARENT_CITY, = COV.parent_scope_columns
J1, J2, J3, ORPHAN = "SYNTH-JOB-001", "SYNTH-JOB-002", "SYNTH-JOB-003", "SYNTH-JOB-999"
CITY, OTHER_CITY = "SYNTH-CITY-1", "SYNTH-CITY-2"
A, B = "SYNTH-BRANCH-A", "SYNTH-BRANCH-B"
SYNTH_COV = dataclasses.replace(COV, expected_locations=((CITY, A),), mode=LocationCoverageMode.MINIMUM_REQUIRED)


def frame(key: DatasetKey, rows: list[dict]) -> pd.DataFrame:
    definition = DATASET_DEFINITIONS[key]
    df = pd.DataFrame([{c: r.get(c, f"SYNTH-{c}") for c in definition.columns} for r in rows],
                      columns=list(definition.columns))
    return df.astype(dict(definition.identifier_dtypes))


def jobs(*rows: tuple) -> pd.DataFrame:
    """(job key, primary declared, secondary declared[, city])."""
    return frame(JOBS, [{PK: r[0], PRIMARY: r[1], SECONDARY: r[2], PARENT_CITY: r[3] if len(r) > 3 else CITY}
                        for r in rows])


def cars(*rows: tuple) -> pd.DataFrame:
    """(job key, branch label[, city])."""
    return frame(CARS, [{DK: r[0], POSITION: i, LABEL_COL: r[1], CITY_COL: r[2] if len(r) > 2 else CITY}
                        for i, r in enumerate(rows)])


# ===================================================== Part A: complete-source loader


@pytest.fixture
def multi_row_raw(tmp_path: Path) -> Path:
    for key in DatasetKey:
        write_synthetic_csv(tmp_path / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    return tmp_path


@pytest.mark.parametrize("options", [
    {"nrows": 1}, {"nrows": 3}, {"skiprows": [1]}, {"skiprows": 1}, {"skipfooter": 1, "engine": "python"},
    {"chunksize": 1}, {"iterator": True}, {"comment": "#"}, {"usecols": [0]}, {"index_col": 0},
    {"header": 0}, {"names": ["synth"]},
])
def test_row_eliding_options_are_rejected_before_any_read(multi_row_raw, monkeypatch, options):
    calls = []
    monkeypatch.setattr(ingestion.pd, "read_csv", lambda *a, **k: calls.append(1))
    option = next(o for o in options if o != "engine")
    with pytest.raises(IncompleteSourceOptionError) as info:
        load_raw_datasets(multi_row_raw, read_csv_options=options)
    assert info.value.option == option and option in str(info.value) and info.value.reason
    assert calls == []                                            # nothing was read or returned
    assert isinstance(info.value, IngestionError) and isinstance(info.value, ValueError)


@pytest.mark.parametrize("value", ["skip", "warn", lambda bad: None, "ERROR", None])
def test_row_discarding_bad_line_handlers_are_rejected(multi_row_raw, value):
    with pytest.raises(IncompleteSourceOptionError) as info:
        load_raw_datasets(multi_row_raw, read_csv_options={"on_bad_lines": value})
    assert info.value.option == "on_bad_lines"


def test_fail_fast_bad_line_handling_is_allowed_and_loads_everything(multi_row_raw):
    loaded = load_raw_datasets(multi_row_raw, read_csv_options={"on_bad_lines": SAFE_ON_BAD_LINES})
    assert (len(loaded.jobs), len(loaded.cars), loaded.complete_source) == (3, 3, True)


def test_harmless_options_keep_the_complete_source(multi_row_raw):
    loaded = load_raw_datasets(multi_row_raw, read_csv_options={"na_values": ["synthetic_r0_c3"],
                                                                "encoding": "utf-8"})
    assert (len(loaded.jobs), len(loaded.cars)) == (3, 3) and loaded.complete_source is True
    cleaned = remove_blank_rows_from_raw_datasets(loaded).cleaned
    assert cleaned.complete_source is True                         # carried through blank-row removal


def test_dropping_blank_lines_is_not_a_complete_source(multi_row_raw):
    assert load_raw_datasets(multi_row_raw, preserve_blank_lines=False).complete_source is False
    assert RawDatasets(jobs=frame(JOBS, []), cars=frame(CARS, [])).complete_source is False   # not proven


def test_malformed_record_fails_instead_of_being_skipped(multi_row_raw):
    path = multi_row_raw / f"synthetic_{CARS}.csv"
    path.write_text(path.read_text() + ",".join(["x"] * (len(contract_columns(CARS)) + 3)) + "\n")
    with pytest.raises(RawDataLoadError):
        load_raw_datasets(multi_row_raw)


# ================================================ Part B: (city, branch) coverage


def project_pairs_frame(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return frame(CARS, [{DK: J1, POSITION: i, CITY_COL: c, LABEL_COL: label} for i, (c, label) in enumerate(pairs)])


EXPECTED = list(COV.expected_locations)


def test_project_contract_keys_on_city_branch_pairs():
    assert COV.location_columns == (CITY_COL, LABEL_COL) and COV.label_column == LABEL_COL
    assert all(len(key) == 2 for key in COV.expected_locations)


def test_correct_city_branch_pairs_cover_the_contract():
    r = assess_expected_location_coverage(project_pairs_frame(EXPECTED), COV)
    assert r.is_valid and r.missing_expected_locations == () and r.conflicting_location_label_count == 0
    assert r.expected_pairs == tuple(EXPECTED)


def test_vancouver_labels_under_another_city_do_not_cover_vancouver():
    # Regression: labels were the whole key, so any city satisfied coverage.
    calgary = COV.expected_locations[0][0]
    misassigned = [EXPECTED[0]] + [(calgary, label) for _, label in EXPECTED[1:]]
    df = project_pairs_frame(misassigned)
    r = assess_expected_location_coverage(df, COV)
    assert not r.is_valid and "missing_expected_location" in r.violations
    assert r.missing_expected_locations == tuple(EXPECTED[1:])
    assert (r.covered_expected_location_count, r.unexpected_location_count) == (1, 2)
    evidence = location_pair_evidence(df, COV)
    unexpected = evidence.loc[~evidence["expected"], [CITY_COL, LABEL_COL]]
    assert sorted(map(tuple, unexpected.itertuples(index=False))) == sorted(misassigned[1:])
    assert df[CITY_COL].tolist() == [calgary] * 3                  # source values untouched


def test_one_misassigned_pair_leaves_the_others_covered():
    pairs = [EXPECTED[0], EXPECTED[1], ("SYNTH-OTHER-CITY", EXPECTED[2][1])]
    r = assess_expected_location_coverage(project_pairs_frame(pairs), COV)
    assert r.covered_expected_location_count == 2 and r.missing_expected_locations == (EXPECTED[2],)
    assert not r.is_valid


def test_same_label_under_several_cities_is_a_conflicting_assignment():
    pairs = [*EXPECTED, ("SYNTH-OTHER-CITY", EXPECTED[0][1])]
    df = project_pairs_frame(pairs)
    r = assess_expected_location_coverage(df, COV)
    assert r.all_expected_covered and r.conflicting_location_label_count == 1
    assert not r.is_valid and "conflicting_location_assignment" in r.violations
    evidence = location_pair_evidence(df, COV).set_index([CITY_COL, LABEL_COL])
    assert evidence.loc[("SYNTH-OTHER-CITY", EXPECTED[0][1]), "conflicting_label"]
    assert evidence.loc[EXPECTED[0], "conflicting_label"] and evidence.loc[EXPECTED[0], "expected"]


def test_coverage_evidence_is_deterministic():
    pairs = [*EXPECTED, ("SYNTH-OTHER-CITY", "SYNTH-BRANCH-Z")]
    a, b = project_pairs_frame(pairs), project_pairs_frame(pairs[::-1])
    assert assess_expected_location_coverage(a, COV) == assess_expected_location_coverage(b, COV)
    pd.testing.assert_frame_equal(location_pair_evidence(a, COV), location_pair_evidence(b, COV))


# ========================================= Part C: zero-detail jobs in continuity


def stream(j: pd.DataFrame, c: pd.DataFrame):  # type: ignore[no-untyped-def]
    return investigate_location_stream(j, c, (CITY, A), coverage=SYNTH_COV)


def test_zero_detail_job_stays_in_the_continuity_denominator():
    # Regression: continuity events came only from detail rows, so J2 vanished and the stream was healthy.
    j, c = jobs((J1, 2, 2), (J2, 0, 0)), cars((J1, A), (J1, B))
    r = stream(j, c)
    acc = r.event_accounting
    assert (acc.total_jobs, acc.in_scope_jobs, acc.jobs_with_target_details, acc.zero_offer_jobs,
            acc.missing_detail_jobs, acc.branch_unassignable_jobs) == (2, 2, 1, 1, 0, 1)
    assert r.stream_continuity is StreamContinuity.UNASSESSABLE
    assert (r.status, r.earliest_failing_stage) == (S.CONTINUITY_UNASSESSABLE, P.SOURCE_CONTINUITY)
    assert not r.is_healthy and r.reconciliation_passes is True    # counts reconcile; continuity does not
    per_job = job_detail_count_results(j, c).set_index(PK)
    assert per_job.loc[J2, "observed_detail_count"] == 0 and per_job.loc[J2, "job_reconciled"]


def test_job_with_declared_details_but_none_observed_is_missing_not_zero_offer():
    r = stream(jobs((J1, 1, 1), (J2, 3, 3)), cars((J1, A)))
    assert (r.event_accounting.zero_offer_jobs, r.event_accounting.missing_detail_jobs) == (0, 1)
    assert not r.is_healthy


def test_all_jobs_empty_none_disappear_and_nothing_is_healthy():
    j, c = jobs((J1, 0, 0), (J2, 0, 0), (J3, 0, 0)), cars()
    r = stream(j, c)
    assert r.status is S.RAW_STREAM_ABSENT and not r.is_healthy
    per_job = job_detail_count_results(j, c)
    assert per_job[PK].tolist() == [J1, J2, J3] and per_job["observed_detail_count"].tolist() == [0, 0, 0]
    report = reconcile(j, c)
    assert report.jobs_without_linked_details_count == 3 and report.job_count == 3


def test_zero_detail_job_keeps_its_city_and_gets_no_branch():
    j, c = jobs((J1, 1, 1), (J2, 0, 0), (J3, 0, 0, OTHER_CITY)), cars((J1, A))
    acc = stream(j, c).event_accounting
    assert (acc.in_scope_jobs, acc.scope_excluded_jobs, acc.branch_unassignable_jobs) == (2, 1, 1)
    assert acc.jobs_with_target_details == 1                        # not credited to the branch


def test_jobs_without_a_city_are_counted_as_unassignable():
    j, c = jobs((J1, 1, 1), (J2, 0, 0, None)), cars((J1, A))
    acc = stream(j, c).event_accounting
    assert (acc.total_jobs, acc.scope_unassignable_jobs, acc.in_scope_jobs) == (2, 1, 1)


def test_authoritative_job_level_location_assigns_a_zero_offer_event():
    # With a location key on the jobs side the event belongs to that stream (no details: zero offers).
    jobs_cov = dataclasses.replace(COV, dataset=JOBS, location_columns=(PARENT_CITY,), label_column=None,
                                   expected_locations=((CITY,),), stream_scope_columns=(), parent_scope_columns=())
    r = investigate_location_stream(jobs((J1, 0, 0)), cars(), (CITY,), coverage=jobs_cov)
    assert (r.status, r.earliest_failing_stage) == (S.JOBS_PRESENT_DETAILS_ABSENT, P.DETAIL_PRESENCE)


def test_orphan_details_do_not_change_the_job_denominator():
    j, c = jobs((J1, 1, 1)), cars((J1, A), (ORPHAN, A))
    r = stream(j, c)
    assert r.event_accounting.total_jobs == 1 and r.event_accounting.jobs_with_target_details == 1
    assert r.target_details_all_linked is False and S.RELATIONSHIP_LINK_FAILURE is r.status
    assert reconcile(j, c).orphan_detail_row_count == 1


def test_continuity_is_order_independent():
    j, c = jobs((J1, 2, 2), (J2, 0, 0), (J3, 1, 1)), cars((J1, A), (J1, B), (J3, B))
    assert stream(j, c) == stream(j.iloc[::-1], c.iloc[::-1])


def test_complete_continuity_when_every_in_scope_job_has_the_branch():
    r = stream(jobs((J1, 1, 1), (J2, 1, 1)), cars((J1, A), (J2, A)))
    assert r.stream_continuity is StreamContinuity.COMPLETE and r.is_healthy


# ===================================== Part D: every declared count is reconciled


def fields_of(report) -> dict:  # type: ignore[no-untyped-def]
    return {f.column: f for f in report.count_fields}


def test_both_declared_counts_match():
    r = reconcile(jobs((J1, 2, 2)), cars((J1, A), (J1, B)))
    assert r.is_reconciled and r.declared_counts_agree and r.reconciled_job_count == 1
    assert all(f.reconciled for f in r.count_fields) and [f.column for f in r.count_fields] == [PRIMARY, SECONDARY]


def test_secondary_count_mismatch_fails_although_primary_matches():
    # Regression: only the primary declaration was reconciled.
    j, c = jobs((J1, 2, 999)), cars((J1, A), (J1, B))
    r = reconcile(j, c)
    assert not r.is_reconciled and fields_of(r)[PRIMARY].reconciled and not fields_of(r)[SECONDARY].reconciled
    assert fields_of(r)[SECONDARY].under_counted_job_count == 1
    assert r.violations == ("under_count", "declared_counts_disagree")
    row = job_detail_count_results(j, c).iloc[0]
    assert (row["observed_detail_count"], row[PRIMARY], row[SECONDARY]) == (2, 2, 999)
    assert (row[f"{PRIMARY}_matches"], row[f"{SECONDARY}_matches"], row["declared_counts_agree"],
            row["job_reconciled"]) == (True, False, False, False)


def test_primary_count_mismatch_fails_although_secondary_matches():
    r = reconcile(jobs((J1, 999, 2)), cars((J1, A), (J1, B)))
    assert not r.is_reconciled and not fields_of(r)[PRIMARY].reconciled and fields_of(r)[SECONDARY].reconciled


def test_both_mismatches_are_preserved():
    r = reconcile(jobs((J1, 5, 7)), cars((J1, A), (J1, B)))
    assert not fields_of(r)[PRIMARY].reconciled and not fields_of(r)[SECONDARY].reconciled
    assert r.declared_counts_disagree_job_count == 1


def test_declarations_that_disagree_fail_even_without_details_to_compare():
    r = reconcile(jobs((J1, 0, 1)), cars())
    assert not r.declared_counts_agree and not r.is_reconciled


def test_zero_detail_job_with_zero_declarations_reconciles():
    j, c = jobs((J1, 0, 0)), cars()
    r = reconcile(j, c)
    assert r.is_reconciled and r.jobs_without_linked_details_count == 1
    row = job_detail_count_results(j, c).iloc[0]
    assert row["observed_detail_count"] == 0 and row["job_reconciled"]


@pytest.mark.parametrize("bad", [None, "", "two", -1, 1.5, True, float("inf")])
@pytest.mark.parametrize("which", ["primary", "secondary"])
def test_invalid_declarations_never_pass(bad, which):
    primary, secondary = (bad, 1) if which == "primary" else (1, bad)
    j = frame(JOBS, [{PK: J1, PRIMARY: primary, SECONDARY: secondary, PARENT_CITY: CITY}]).astype(
        {PRIMARY: object, SECONDARY: object})
    r = reconcile(j, cars((J1, A)))
    column = PRIMARY if which == "primary" else SECONDARY
    assert not r.is_reconciled and not fields_of(r)[column].reconciled
    assert fields_of(r)[column].valid_job_count == 0
    row = job_detail_count_results(j, cars((J1, A))).iloc[0]
    assert not row[f"{column}_valid"] and not row["job_reconciled"]


def test_one_failed_job_fails_the_aggregate_and_all_jobs_are_reported():
    j, c = jobs((J1, 1, 1), (J2, 1, 2), (J3, 0, 0)), cars((J1, A), (J2, A))
    r = reconcile(j, c)
    assert not r.is_reconciled and r.reconciled_job_count == 2
    per_job = job_detail_count_results(j, c)
    assert per_job[PK].tolist() == [J1, J2, J3] and per_job["job_reconciled"].tolist() == [True, False, True]


def test_duplicate_detail_keys_stay_a_key_failure_not_hidden_by_counts():
    j = jobs((J1, 2, 2))
    c = frame(CARS, [{DK: J1, POSITION: 0, LABEL_COL: A, CITY_COL: CITY}] * 2)
    assert reconcile(j, c).is_reconciled                                   # counts alone agree
    from ql2_sixt_canada_analysis.unique_keys import assess_unique_key
    assert not assess_unique_key(c, REL.detail_definition).is_valid       # key integrity is separate


def test_per_job_results_are_a_fresh_sorted_frame():
    j, c = jobs((J2, 1, 1), (J1, 1, 1)), cars((J1, A), (J2, A))
    first = job_detail_count_results(j, c)
    first.loc[:, "job_reconciled"] = False
    assert job_detail_count_results(j, c)["job_reconciled"].all()
    assert job_detail_count_results(j, c)[PK].tolist() == [J1, J2]


# ================================================== aggregate completeness gate


def complete_inputs():  # type: ignore[no-untyped-def]
    j, c = jobs((J1, 1, 1), (J2, 1, 1)), cars((J1, A), (J2, A))
    datasets = RawDatasets(jobs=j, cars=c, complete_source=True)
    return dict(datasets=datasets, coverage=assess_expected_location_coverage(c, SYNTH_COV),
                streams=assess_expected_location_streams(j, c, coverage=SYNTH_COV), reconciliation=reconcile(j, c),
                city_integrity=assess_city_integrity(j, c, coverage=SYNTH_COV), expected_coverage=SYNTH_COV)


def test_completeness_passes_only_when_every_control_passes():
    report = assess_completeness(**complete_inputs())
    assert report.complete and report.blocking_reasons == ()


def test_completeness_blocks_on_each_failure():
    base = complete_inputs()
    j_bad, c = jobs((J1, 1, 1), (J2, 0, 0)), cars((J1, A))        # zero-detail job
    cases = {
        CB.SOURCE_NOT_COMPLETE: {"datasets": dataclasses.replace(base["datasets"], complete_source=False)},
        CB.COVERAGE_UNAVAILABLE: {"coverage": None},
        CB.EXPECTED_PAIRS_MISSING: {"coverage": assess_expected_location_coverage(cars((J1, B)), SYNTH_COV)},
        CB.EXPECTED_STREAM_ASSESSMENT_UNAVAILABLE: {"streams": None},
        CB.STREAM_CONTINUITY_UNASSESSABLE: {"streams": assess_expected_location_streams(j_bad, c, coverage=SYNTH_COV)},
        CB.RECONCILIATION_UNAVAILABLE: {"reconciliation": None},
        CB.DECLARED_COUNT_UNRECONCILED: {"reconciliation": reconcile(jobs((J1, 1, 9), (J2, 1, 1)),
                                                                     cars((J1, A), (J2, A)))},
        CB.DETAIL_ROWS_UNLINKED: {"reconciliation": reconcile(jobs((J1, 1, 1)), cars((J1, A), (ORPHAN, A)))},
    }
    for blocker, change in cases.items():
        report = assess_completeness(**(base | change))
        assert not report.complete and blocker in report.blocking_reasons, blocker


def test_misassigned_pair_zero_detail_job_and_secondary_mismatch_all_block_together():
    calgary = COV.expected_locations[0][0]
    c = project_pairs_frame([EXPECTED[0]] + [(calgary, label) for _, label in EXPECTED[1:]])
    j = jobs((J1, 3, 999), (J2, 0, 0, calgary))
    report = assess_completeness(
        datasets=RawDatasets(jobs=j, cars=c, complete_source=True),
        coverage=assess_expected_location_coverage(c, COV),
        streams=assess_expected_location_streams(j, c, coverage=COV),
        reconciliation=reconcile(j, c), city_integrity=assess_city_integrity(j, c))
    assert not report.complete
    assert {CB.EXPECTED_PAIRS_MISSING, CB.DECLARED_COUNT_UNRECONCILED, CB.DECLARED_COUNTS_DISAGREE} <= set(
        report.blocking_reasons)
    assert report.blocking_reasons[0] is CB.EXPECTED_PAIRS_MISSING


def test_completeness_type_checks_and_values_name_no_columns():
    with pytest.raises(TypeError):
        assess_completeness(**(complete_inputs() | {"datasets": object()}))
    columns = {c for key in DatasetKey for c in contract_columns(key)}
    assert not any(c in b.value for b in CB for c in columns)


def test_package_exports():
    for name in ("IncompleteSourceOptionError", "SAFE_ON_BAD_LINES", "location_pair_evidence",
                 "job_detail_count_results", "DeclaredCountFieldReport", "StreamEventAccounting",
                 "assess_completeness", "CompletenessReport", "CompletenessBlocker"):
        assert name in ql2_sixt_canada_analysis.__all__
