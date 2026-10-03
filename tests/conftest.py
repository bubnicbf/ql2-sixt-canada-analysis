"""Shared helpers for building obviously synthetic raw CSVs from the contracts."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from ql2_sixt_canada_analysis.schemas import DATASET_DEFINITIONS, DatasetKey


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
