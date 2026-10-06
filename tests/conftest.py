"""Shared helpers for building obviously synthetic raw CSVs from the contracts."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from ql2_sixt_canada_analysis.authority_decisions import AuthorityKind, AuthorityReference
from ql2_sixt_canada_analysis.job_linkage import JobLinkagePolicy, JobLinkageResult, assess_job_linkage
from ql2_sixt_canada_analysis.join_readiness import (
    JobDetailJoinReadiness,
    assess_job_detail_join_readiness,
    require_trusted_job_detail_join,
)
from ql2_sixt_canada_analysis.schemas import ANALYSIS_JOB_DETAIL_RELATIONSHIP, DATASET_DEFINITIONS, DatasetKey

#: A synthetic, fully approved linkage policy (fabricated authority; tests only).
SYNTH_LINKAGE_POLICY = JobLinkagePolicy(
    record_id="pricing-authorities-synthetic",
    authority=(AuthorityReference(kind=AuthorityKind.COLLECTION_OWNER, source="SYNTH-COLLECTION-OWNER",
                                  reference="SYNTH-GOVERNANCE-REFERENCE"),),
    legacy_decimal_zero_repair=True,
    legacy_offer_position_repair=True,
)


def link(jobs, cars, policy=SYNTH_LINKAGE_POLICY) -> JobLinkageResult:  # type: ignore[no-untyped-def]
    """Analysis-stage frames and report for raw synthetic frames under the synthetic policy."""
    return assess_job_linkage(jobs, cars, policy)


def linked_join(jobs, cars, relationship=ANALYSIS_JOB_DETAIL_RELATIONSHIP) -> JobDetailJoinReadiness:  # type: ignore[no-untyped-def]
    """Link raw synthetic frames, then assess the trusted join on the derived keys."""
    result = link(jobs, cars)
    return assess_job_detail_join_readiness(result.jobs, result.cars, relationship, job_linkage=result.report)


def require_linked_join(jobs, cars):  # type: ignore[no-untyped-def]
    """Strict trusted join on the derived keys of raw synthetic frames."""
    result = link(jobs, cars)
    return require_trusted_job_detail_join(result.jobs, result.cars, job_linkage=result.report)


def join_gates(jobs, cars) -> dict:  # type: ignore[no-untyped-def]
    """The trusted-join and job-linkage pricing gates for raw synthetic frames (one consistent pair)."""
    join = linked_join(jobs, cars)
    return {"job_detail_join": join, "job_linkage": join.job_linkage_report}


def write_synthetic_csv(path: Path, columns: Sequence[str], rows: int = 2) -> Path:
    """Write ``rows`` rows of generated placeholder values under ``columns``."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for r in range(rows):
            writer.writerow([f"synthetic_r{r}_c{c}" for c in range(len(columns))])
    return path


def contract_columns(key: DatasetKey) -> tuple[str, ...]:
    return DATASET_DEFINITIONS[key].columns


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A temporary raw directory holding one contract-conforming CSV per dataset."""
    directory = tmp_path / "raw"
    directory.mkdir()
    for key in DatasetKey:
        write_synthetic_csv(directory / f"synthetic_{key}.csv", contract_columns(key))
    return directory
