"""Integration: authority-backed linkage feeds every downstream contract and gate.

Synthetic CSVs reproduce the *shape* of the historical export (digit-only job
identifiers on jobs, the same digits plus a decimal-zero suffix on details,
decimal-zero offer positions) with fabricated values only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
from conftest import SYNTH_LINKAGE_POLICY, link, linked_join
from test_readiness import DISTINCT, GATES, scheduled_frames

from ql2_sixt_canada_analysis import (
    ANALYSIS_DATASET_DEFINITIONS,
    ANALYSIS_JOB_DETAIL_RELATIONSHIP as REL,
    CONFIDENTIAL_TECHNICAL_COLUMNS,
    JOB_DETAIL_RELATIONSHIP as RAW_REL,
    LOCATION_STREAM_COMPARISON,
    VEHICLE_ATTRIBUTE_STABILITY,
    assess_city_integrity,
    assess_job_detail_join_readiness,
    assess_job_detail_reconciliation,
    assess_job_linkage,
    assess_location_policy,
    assess_one_to_many_join,
    assess_pricing_readiness,
    assess_raw_dataset_unique_keys,
    load_job_linkage_policy,
    load_raw_datasets,
    remove_blank_rows_from_raw_datasets,
    validate_raw_dataset_identifier_dtypes,
)
from ql2_sixt_canada_analysis.join_readiness import JobDetailJoinBlocker as JB
from ql2_sixt_canada_analysis.readiness import PricingBlocker as B
from ql2_sixt_canada_analysis.schemas import (
    ANALYSIS_LOCATION_STREAM_COMPARISON,
    ANALYSIS_TEMPORAL_RECONCILIATION,
    DATASET_DEFINITIONS,
    JOB_LINKAGE_KEY_COLUMN as LK,
    OFFER_POSITION_KEY_COLUMN as PK,
    SOURCE_JOB_IDENTIFIER_COLUMN as JID,
    DatasetKey,
)

JOBS, CARS = DATASET_DEFINITIONS[DatasetKey.JOBS], DATASET_DEFINITIONS[DatasetKey.CARS]
CITY = "SYNTH-CITY"
JOB_IDS = ("4101", "04102", "4103")                   # fabricated; one with a significant leading zero
OFFERS = {"4101": 2, "04102": 1, "4103": 0}


def _write(path: Path, columns, rows) -> None:  # type: ignore[no-untyped-def]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([[row.get(c, f"SYNTH-{c}") for c in columns] for row in rows])


@pytest.fixture
def legacy_export(tmp_path: Path) -> Path:
    """Raw CSVs shaped like the historical export (all values fabricated)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    counts = {c: n for c in RAW_REL.expected_detail_count_columns for n in [None]}
    _write(raw / "synthetic_jobs.csv", JOBS.columns,
           [{JID: j, "city": CITY, **{c: str(OFFERS[j]) for c in counts}} for j in JOB_IDS])
    _write(raw / "synthetic_cars.csv", CARS.columns,
           [{JID: f"{j}.0", "city": CITY, "row_index": f"{p}.0"} for j in JOB_IDS for p in range(OFFERS[j])])
    return raw


def cleaned(raw: Path):  # type: ignore[no-untyped-def]
    datasets = remove_blank_rows_from_raw_datasets(load_raw_datasets(raw)).cleaned
    validate_raw_dataset_identifier_dtypes(datasets)
    return datasets


def test_legacy_export_shape_becomes_fully_linkable(legacy_export: Path) -> None:
    source = cleaned(legacy_export)
    raw_ids = source.cars[JID].copy()
    linkage = assess_job_linkage(source.jobs, source.cars, load_job_linkage_policy())   # the committed v2 policy
    report = linkage.report
    assert report.is_valid and report.blocking_reasons == ()
    assert report.decimal_zero_repair_count == len(source.cars) == report.linked_detail_count
    assert report.valid_row_index_count == len(source.cars)
    analysis = linkage.datasets(source)
    pd.testing.assert_series_equal(analysis.cars[JID], raw_ids)                  # raw values unchanged
    assert analysis.jobs[JID].tolist() == list(JOB_IDS)
    assert sorted(set(analysis.cars[LK])) == sorted(j for j in JOB_IDS if OFFERS[j])
    # Before linkage, raw text links nothing (the historical defect is still visible).
    raw_relation = assess_one_to_many_join(source.jobs, source.cars, RAW_REL)
    assert raw_relation.orphan_detail_row_count == len(source.cars)
    # Every analytical contract runs on the derived keys and passes.
    keys = assess_raw_dataset_unique_keys(analysis, ANALYSIS_DATASET_DEFINITIONS)
    assert keys.all_valid
    j, c = analysis.jobs, analysis.cars
    assert assess_job_detail_reconciliation(j, c, REL).is_reconciled
    assert assess_one_to_many_join(j, c, REL).is_valid
    assert assess_city_integrity(j, c, relationship=REL, coverage=None).is_valid
    join = assess_job_detail_join_readiness(j, c, REL, job_linkage=report)
    assert join.join_ready and join.job_linkage_valid
    joined = join.trusted_jobs_with_details
    assert JID not in joined and {f"{JID}{REL.parent_suffix}", f"{JID}{REL.detail_suffix}", LK, PK} <= set(joined)


def test_missing_authority_policy_blocks_the_trusted_join(legacy_export: Path) -> None:
    source = cleaned(legacy_export)
    linkage = assess_job_linkage(source.jobs, source.cars, None)
    join = assess_job_detail_join_readiness(linkage.jobs, linkage.cars, REL, job_linkage=linkage.report)
    assert not join.join_ready and join.trusted_jobs_with_details is None
    assert JB.JOB_LINKAGE_NOT_VALID in join.blocking_reasons
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                       **(GATES | {"job_detail_join": join, "job_linkage": linkage.report}))
    assert not pricing.ready and B.LINKAGE_POLICY_UNAVAILABLE in pricing.blocking_reasons
    assert B.JOB_IDENTIFIER_NORMALIZATION_NOT_READY in pricing.blocking_reasons


def _pricing(j, c):  # type: ignore[no-untyped-def]
    linkage = link(j, c)
    join = assess_job_detail_join_readiness(linkage.jobs, linkage.cars, job_linkage=linkage.report)
    return join, assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                          **(GATES | {"job_detail_join": join, "job_linkage": linkage.report}))


def _busiest(c: pd.DataFrame) -> tuple[str, int]:
    counts = c[JID].value_counts()
    return counts.index[0], int(counts.iloc[0])


def _retarget(j: pd.DataFrame, c: pd.DataFrame, parent_id: str, detail_ids: list[str]):  # type: ignore[no-untyped-def]
    """Give the busiest job ``parent_id`` and its detail rows ``detail_ids`` (fabricated values)."""
    j, c = j.copy(), c.copy()
    target, _ = _busiest(c)
    rows = c.index[c[JID] == target]
    assert len(rows) == len(detail_ids)
    j.loc[j[JID] == target, JID] = parent_id
    c.loc[rows, JID] = detail_ids
    return j, c


def test_ambiguity_blocks_the_trusted_join_and_pricing() -> None:
    j, c = scheduled_frames()
    _, n = _busiest(c)
    j, c = _retarget(j, c, "9100", ["9100.0"] * n)
    rival = j.loc[j[JID] == "9100"].copy()
    rival[JID] = pd.array(["9100.0"], dtype="string")             # a second job whose exact text is the legacy form
    j = pd.concat([j, rival], ignore_index=True)
    join, pricing = _pricing(j, c)
    assert join.job_linkage_report.ambiguous_identifier_count == n
    assert not join.join_ready and JB.JOB_LINKAGE_NOT_VALID in join.blocking_reasons
    assert not pricing.ready
    assert {B.JOB_IDENTIFIER_NORMALIZATION_NOT_READY, B.LINKAGE_DETAIL_REFERENCE_AMBIGUOUS} <= set(pricing.blocking_reasons)


def test_collision_blocks_the_trusted_join_and_pricing() -> None:
    j, c = scheduled_frames()
    _, n = _busiest(c)
    j, c = _retarget(j, c, "9200", ["9200"] * n)
    extra = c.loc[c[JID] == "9200"].iloc[[0]].copy()
    extra[JID] = pd.array(["9200.0"], dtype="string")            # the same job in its legacy representation
    extra[CARS.non_identifier_key_columns[0]] = 99
    c = pd.concat([c, extra], ignore_index=True)
    join, pricing = _pricing(j, c)
    assert join.job_linkage_report.collision_count == n + 1
    assert not join.join_ready
    assert {B.JOB_IDENTIFIER_NORMALIZATION_NOT_READY, B.LINKAGE_COLLISION} <= set(pricing.blocking_reasons)


def test_invalid_offer_position_blocks_the_trusted_join_and_pricing() -> None:
    j, c = scheduled_frames()
    c = c.copy()
    c[CARS.non_identifier_key_columns[0]] = c[CARS.non_identifier_key_columns[0]].astype(object)
    c.loc[c.index[0], CARS.non_identifier_key_columns[0]] = "1e0"
    join, pricing = _pricing(j, c)
    assert not join.join_ready and B.LINKAGE_OFFER_POSITION_INVALID in pricing.blocking_reasons


def test_valid_linkage_never_bypasses_unrelated_blockers() -> None:
    join = linked_join(*scheduled_frames())
    assert join.join_ready and join.job_linkage_valid
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                       **(GATES | {"job_detail_join": join, "job_linkage": join.job_linkage_report,
                                                   "temporal_fields_trusted": False, "key_contracts_valid": False}))
    assert pricing.job_identifier_normalization_ready and not pricing.ready
    assert pricing.blocking_reasons == (B.KEY_CONTRACTS_INVALID, B.TEMPORAL_FIELDS_UNTRUSTED)
    unresolved = assess_pricing_readiness(location_policy=assess_location_policy(),
                                          **(GATES | {"job_detail_join": join, "job_linkage": join.job_linkage_report}))
    assert B.LOCATION_POLICY_UNRESOLVED in unresolved.blocking_reasons and not unresolved.ready


def test_pricing_requires_the_same_linkage_report_as_the_join() -> None:
    join = linked_join(*scheduled_frames())
    other = link(*scheduled_frames()).report                       # equal counts, different assessment
    assert other == join.job_linkage_report and other is not join.job_linkage_report
    pricing = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                       **(GATES | {"job_detail_join": join, "job_linkage": other}))
    assert pricing.blocking_reasons == (B.JOB_IDENTIFIER_NORMALIZATION_NOT_READY, B.JOB_LINKAGE_REPORT_MISMATCH)
    missing = assess_pricing_readiness(location_policy=assess_location_policy(DISTINCT),
                                       **(GATES | {"job_linkage": None}))
    assert missing.blocking_reasons == (B.JOB_IDENTIFIER_NORMALIZATION_MISSING,)


def test_diagnostic_frames_remain_untrusted_when_linkage_fails() -> None:
    linkage = link(*scheduled_frames(), policy=None)
    result = link(*scheduled_frames())
    join = assess_job_detail_join_readiness(result.jobs, result.cars, job_linkage=linkage.report)
    assert join.trusted_jobs_with_details is None and join.diagnostic_jobs_with_details is not None
    assert JB.JOB_LINKAGE_NOT_VALID in join.blocking_reasons


def test_confidential_technical_fields_stay_out_of_identity_and_evidence() -> None:
    confidential = set(CONFIDENTIAL_TECHNICAL_COLUMNS)
    stability = VEHICLE_ATTRIBUTE_STABILITY
    assert not confidential & {*stability.group_columns, *stability.attribute_columns}
    for comparison in (LOCATION_STREAM_COMPARISON, ANALYSIS_LOCATION_STREAM_COMPARISON):
        assert not confidential & {*comparison.product_columns, *comparison.price_columns}
    assert ANALYSIS_TEMPORAL_RECONCILIATION.relationship is REL
    assert ANALYSIS_LOCATION_STREAM_COMPARISON.relationship is REL
    assert RAW_REL.parent_key_columns == (JID,) and REL.parent_key_columns == (LK,)
    assert REL.detail_definition.unique_key_columns == (LK, PK)
    assert DATASET_DEFINITIONS[DatasetKey.CARS].unique_key_columns == (JID, "row_index")   # raw contract kept


def test_pricing_baseline_reports_only_aggregate_linkage(legacy_export: Path) -> None:
    from ql2_sixt_canada_analysis.pricing_baseline import render_baseline_markdown, run_pricing_baseline

    baseline = run_pricing_baseline(legacy_export)
    data = baseline.to_dict()
    statuses = dict(data["statuses"])
    assert statuses["job_linkage_policy"] == "available" and statuses["job_linkage_valid"] == "true"
    assert dict(data["subordinate_blockers"])["job_linkage"] == []
    text = render_baseline_markdown(baseline, commit="unrecorded", date="unrecorded")
    assert not any(value in text for value in (*JOB_IDS, "4101.0", LK, PK, JID))
    assert SYNTH_LINKAGE_POLICY.record_id not in text
