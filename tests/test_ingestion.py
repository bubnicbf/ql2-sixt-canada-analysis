"""Tests for raw-CSV discovery, header validation and loading.

All inputs are small, obviously synthetic CSVs generated in ``tmp_path`` from
the centralized column contracts. The proprietary raw files are never read.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import contract_columns, write_synthetic_csv

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis import ingestion, paths
from ql2_sixt_canada_analysis.ingestion import (
    AmbiguousDatasetError,
    DatasetNotFoundError,
    IngestionError,
    NotARegularFileError,
    RawDataDirectoryError,
    RawDataDiscoveryError,
    RawDataLoadError,
    RawDatasetPaths,
    RawDatasets,
    SourceSchemaError,
    discover_raw_csvs,
    load_raw_datasets,
)
from ql2_sixt_canada_analysis.schemas import DATASET_DEFINITIONS, DatasetKey

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
J = DATASET_DEFINITIONS[JOBS].filename_tokens[0]
C = DATASET_DEFINITIONS[CARS].filename_tokens[0]


def _jobs(directory: Path, name: str, columns: tuple[str, ...] | None = None) -> Path:
    return write_synthetic_csv(directory / name, columns or contract_columns(JOBS))


def _cars(directory: Path, name: str, columns: tuple[str, ...] | None = None) -> Path:
    return write_synthetic_csv(directory / name, columns or contract_columns(CARS))


def _snapshot(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): "<dir>" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
    }


# --------------------------------------------------------------------- discovery


def test_discovers_one_file_per_dataset(raw_dir: Path) -> None:
    found = discover_raw_csvs(raw_dir)
    assert isinstance(found, RawDatasetPaths)
    assert found.jobs == raw_dir / f"synthetic_{JOBS}.csv"
    assert found.cars == raw_dir / f"synthetic_{CARS}.csv"


def test_discovery_uses_centralized_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from types import MappingProxyType

    patched = MappingProxyType({
        JOBS: replace(DATASET_DEFINITIONS[JOBS], filename_tokens=("alpha",)),
        CARS: replace(DATASET_DEFINITIONS[CARS], filename_tokens=("beta",)),
    })
    monkeypatch.setattr(ingestion, "DATASET_DEFINITIONS", patched)
    jobs = _jobs(tmp_path, "export alpha (1).csv")
    cars = _cars(tmp_path, "export beta (1).csv")
    _jobs(tmp_path, f"ignored_{J}.csv")  # old token no longer matches
    assert discover_raw_csvs(tmp_path) == RawDatasetPaths(jobs=jobs, cars=cars)


def test_discovery_is_case_insensitive(tmp_path: Path) -> None:
    jobs = _jobs(tmp_path, f"Synthetic_{J.upper()}.CSV")
    cars = _cars(tmp_path, f"synthetic_{C.capitalize()}.Csv")
    assert discover_raw_csvs(tmp_path) == RawDatasetPaths(jobs=jobs, cars=cars)


@pytest.mark.parametrize(
    ("jobs_name", "cars_name"),
    [
        (f"demo export 01-99 {J} (1).csv", f"demo export 01-99 {C} (1).csv"),
        (f"2001_{J}_feed (12).csv", f"2001-{C}-feed (3).csv"),
        (f"alpha.{J}.v2.csv", f"alpha.{C}.v2.csv"),
        # Shared prefix naming the other dataset: the last token wins.
        (f"demo_{J}_5_6_{J}_extract (1).csv", f"demo_{J}_5_6_{C}_extract (1).csv"),
        (f"demo {C} batch - {J}.csv", f"demo {J} batch - {C}.csv"),
    ],
)
def test_discovery_tolerates_variable_filename_parts(
    tmp_path: Path, jobs_name: str, cars_name: str
) -> None:
    jobs, cars = _jobs(tmp_path, jobs_name), _cars(tmp_path, cars_name)
    assert discover_raw_csvs(tmp_path) == RawDatasetPaths(jobs=jobs, cars=cars)


def test_unrelated_and_non_csv_files_are_ignored(raw_dir: Path) -> None:
    for name in ["notes.csv", f"{J}site_summary.csv", "README.md", ".gitkeep",
                 f"extra_{J}.txt", f"extra_{C}.parquet", f"extra_{J}.csv.bak"]:
        (raw_dir / name).write_text("synthetic\n", encoding="utf-8")
    found = discover_raw_csvs(raw_dir)
    assert (found.jobs.name, found.cars.name) == (f"synthetic_{JOBS}.csv", f"synthetic_{CARS}.csv")


def test_discovery_does_not_recurse(raw_dir: Path) -> None:
    nested = raw_dir / "nested"
    nested.mkdir()
    _jobs(nested, f"other_{J}.csv")
    _cars(nested, f"other_{C}.csv")
    found = discover_raw_csvs(raw_dir)
    assert found.jobs.parent == raw_dir and found.cars.parent == raw_dir


def test_discovery_accepts_str_and_path(raw_dir: Path) -> None:
    assert discover_raw_csvs(str(raw_dir)) == discover_raw_csvs(raw_dir)


def test_default_directory_comes_from_paths_module(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingestion, "RAW_DATA_DIR", raw_dir)
    assert discover_raw_csvs().jobs.parent == raw_dir
    assert ingestion.RAW_DATA_DIR is not paths.RAW_DATA_DIR  # patched, not shared state


def test_caller_directory_overrides_default(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingestion, "RAW_DATA_DIR", raw_dir.parent / "does_not_exist")
    assert isinstance(load_raw_datasets(raw_dir), RawDatasets)


# ----------------------------------------------------------------------- loading


def test_loads_both_datasets_with_contract_columns(raw_dir: Path) -> None:
    datasets = load_raw_datasets(raw_dir)
    assert isinstance(datasets.jobs, pd.DataFrame) and isinstance(datasets.cars, pd.DataFrame)
    assert datasets.jobs is not datasets.cars
    assert tuple(datasets.jobs.columns) == contract_columns(JOBS)
    assert tuple(datasets.cars.columns) == contract_columns(CARS)


def test_loaded_values_match_synthetic_input(raw_dir: Path) -> None:
    jobs = load_raw_datasets(raw_dir).jobs
    columns = contract_columns(JOBS)
    expected = pd.DataFrame(
        [[f"synthetic_r{r}_c{c}" for c in range(len(columns))] for r in range(2)],
        columns=list(columns),
    )
    pd.testing.assert_frame_equal(jobs, expected)


def test_loading_does_not_clean_values(tmp_path: Path) -> None:
    columns = contract_columns(JOBS)
    path = tmp_path / f"x_{J}.csv"
    blank = ",".join([""] * len(columns))
    padded = ",".join([" pad "] * len(columns))
    path.write_text(",".join(columns) + f"\n{padded}\n{padded}\n{blank}\n", encoding="utf-8")
    _cars(tmp_path, f"x_{C}.csv")
    jobs = load_raw_datasets(tmp_path).jobs
    assert len(jobs) == 3  # duplicates kept
    assert jobs.iloc[0, 0] == " pad "  # whitespace kept
    assert jobs.iloc[2].isna().all()


def test_read_csv_options_apply_to_both_files(raw_dir: Path) -> None:
    datasets = load_raw_datasets(raw_dir, read_csv_options={"na_values": ["synthetic_r0_c0"]})
    for frame in (datasets.jobs, datasets.cars):
        assert pd.isna(frame.iloc[0, 0])
        assert frame.iloc[1, 0] == "synthetic_r1_c0"


@pytest.mark.parametrize(
    "option", ["chunksize", "iterator", "filepath_or_buffer", "usecols", "names", "header"]
)
def test_options_that_change_structure_are_rejected(raw_dir: Path, option: str) -> None:
    with pytest.raises(ValueError, match=option):
        load_raw_datasets(raw_dir, read_csv_options={option: 1})


def test_each_file_is_parsed_by_pandas_once(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = pd.read_csv

    def counting(path, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(Path(path).name)
        return real(path, **kwargs)

    monkeypatch.setattr(ingestion.pd, "read_csv", counting)
    load_raw_datasets(raw_dir)
    assert sorted(calls) == sorted([f"synthetic_{CARS}.csv", f"synthetic_{JOBS}.csv"])


def test_loading_writes_nothing_and_leaves_sources_unchanged(raw_dir: Path) -> None:
    root = raw_dir.parent
    before = _snapshot(root)
    mtimes = {p: p.stat().st_mtime_ns for p in raw_dir.iterdir()}
    load_raw_datasets(raw_dir)
    assert _snapshot(root) == before
    assert {p: p.stat().st_mtime_ns for p in raw_dir.iterdir()} == mtimes


def test_loading_creates_no_interim_or_processed_files(raw_dir: Path) -> None:
    def listing() -> dict[str, set[str]]:
        return {
            str(d): {p.name for p in d.iterdir()} if d.is_dir() else set()
            for d in (paths.INTERIM_DATA_DIR, paths.PROCESSED_DATA_DIR)
        }

    before = listing()
    load_raw_datasets(raw_dir)
    assert listing() == before


# ------------------------------------------------------------- header contracts


@pytest.mark.parametrize("key", list(DatasetKey))
def test_missing_column_is_rejected(tmp_path: Path, key: DatasetKey) -> None:
    columns = contract_columns(key)
    _jobs(tmp_path, f"s_{J}.csv", columns[:-1] if key is JOBS else None)
    _cars(tmp_path, f"s_{C}.csv", columns[:-1] if key is CARS else None)
    with pytest.raises(SourceSchemaError, match=f"'{key}'") as info:
        load_raw_datasets(tmp_path)
    assert info.value.role == key
    assert info.value.missing == (columns[-1],)
    assert not info.value.unexpected


@pytest.mark.parametrize("key", list(DatasetKey))
def test_unexpected_column_is_rejected(tmp_path: Path, key: DatasetKey) -> None:
    extended = (*contract_columns(key), "synthetic_extra_column")
    _jobs(tmp_path, f"s_{J}.csv", extended if key is JOBS else None)
    _cars(tmp_path, f"s_{C}.csv", extended if key is CARS else None)
    with pytest.raises(SourceSchemaError, match="1 unexpected") as info:
        load_raw_datasets(tmp_path)
    assert info.value.role == key
    assert info.value.unexpected == ("synthetic_extra_column",)


@pytest.mark.parametrize("key", list(DatasetKey))
def test_column_order_is_enforced(tmp_path: Path, key: DatasetKey) -> None:
    reordered = tuple(reversed(contract_columns(key)))
    _jobs(tmp_path, f"s_{J}.csv", reordered if key is JOBS else None)
    _cars(tmp_path, f"s_{C}.csv", reordered if key is CARS else None)
    with pytest.raises(SourceSchemaError, match="out of order") as info:
        load_raw_datasets(tmp_path)
    assert info.value.order_mismatch
    assert not (info.value.missing or info.value.unexpected or info.value.duplicated)


def test_duplicate_column_is_rejected(tmp_path: Path) -> None:
    columns = contract_columns(JOBS)
    path = tmp_path / f"s_{J}.csv"
    path.write_text(",".join((*columns, columns[0])) + "\n", encoding="utf-8")
    _cars(tmp_path, f"s_{C}.csv")
    with pytest.raises(SourceSchemaError, match="duplicated") as info:
        load_raw_datasets(tmp_path)
    assert info.value.duplicated == (columns[0],)


def test_schema_error_message_contains_no_column_names(tmp_path: Path) -> None:
    _jobs(tmp_path, f"s_{J}.csv", ("synthetic_secret_header",))
    _cars(tmp_path, f"s_{C}.csv")
    with pytest.raises(SourceSchemaError) as info:
        load_raw_datasets(tmp_path)
    message = str(info.value)
    assert "synthetic_secret_header" not in message
    assert not any(column in message for column in contract_columns(JOBS))
    assert str(tmp_path) not in message


def test_header_is_validated_before_full_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _jobs(tmp_path, f"s_{J}.csv", ("synthetic_wrong",))
    _cars(tmp_path, f"s_{C}.csv")
    monkeypatch.setattr(ingestion.pd, "read_csv", lambda *a, **k: pytest.fail("parsed early"))
    with pytest.raises(SourceSchemaError):
        load_raw_datasets(tmp_path)


# ------------------------------------------------------------------------ errors


def test_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(RawDataDirectoryError, match="does not exist") as info:
        load_raw_datasets(missing)
    assert info.value.path == missing


def test_path_that_is_not_a_directory(tmp_path: Path) -> None:
    a_file = tmp_path / "plain.txt"
    a_file.write_text("synthetic\n", encoding="utf-8")
    with pytest.raises(RawDataDirectoryError, match="not a directory"):
        discover_raw_csvs(a_file)


@pytest.mark.parametrize("key", list(DatasetKey))
def test_missing_dataset(tmp_path: Path, key: DatasetKey) -> None:
    if key is not JOBS:
        _jobs(tmp_path, f"s_{J}.csv")
    if key is not CARS:
        _cars(tmp_path, f"s_{C}.csv")
    with pytest.raises(DatasetNotFoundError, match=f"'{key}'") as info:
        discover_raw_csvs(tmp_path)
    assert info.value.role == key


@pytest.mark.parametrize("key", list(DatasetKey))
def test_multiple_candidates_are_ambiguous(raw_dir: Path, key: DatasetKey) -> None:
    write_synthetic_csv(raw_dir / f"synthetic_{key} (1).csv", contract_columns(key))
    with pytest.raises(AmbiguousDatasetError, match=f"2 candidate CSVs for the '{key}'") as info:
        discover_raw_csvs(raw_dir)
    assert info.value.role == key and len(info.value.candidates) == 2


def test_matching_path_that_is_not_a_regular_file(tmp_path: Path) -> None:
    (tmp_path / f"s_{J}.csv").mkdir()
    _cars(tmp_path, f"s_{C}.csv")
    with pytest.raises(NotARegularFileError, match=f"'{JOBS}'"):
        discover_raw_csvs(tmp_path)


def _contract_header() -> bytes:
    return (",".join(contract_columns(JOBS)) + "\n").encode()


def _synthetic_row() -> bytes:
    return (",".join(["synthetic"] * len(contract_columns(JOBS))) + "\n").encode()


@pytest.mark.parametrize(
    ("payload", "cause"),
    [
        (lambda: b"", None),
        (lambda: _contract_header() + _synthetic_row() + b"SECRETTOKEN" + b"," * 200 + b"\n", ValueError),
        (lambda: b"\xff\xfe\xfaSECRETTOKEN\n", ValueError),
    ],
    ids=["empty", "ragged-row", "bad-encoding"],
)
def test_unreadable_csv_raises_load_error(tmp_path: Path, payload, cause) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / f"s_{J}.csv").write_bytes(payload())
    _cars(tmp_path, f"s_{C}.csv")
    with pytest.raises(RawDataLoadError, match=f"'{JOBS}'") as info:
        load_raw_datasets(tmp_path)
    assert info.value.role == JOBS
    if cause is not None:
        assert isinstance(info.value.__cause__, cause)
    assert "SECRETTOKEN" not in str(info.value)
    assert str(tmp_path) not in str(info.value)


def test_exception_hierarchy() -> None:
    for error in (RawDataDirectoryError, DatasetNotFoundError, AmbiguousDatasetError,
                  NotARegularFileError):
        assert issubclass(error, RawDataDiscoveryError)
    for error in (RawDataDiscoveryError, RawDataLoadError, SourceSchemaError):
        assert issubclass(error, IngestionError)


# ------------------------------------------------------------------- import safety


def test_package_exposes_public_api() -> None:
    assert ql2_sixt_canada_analysis.load_raw_datasets is load_raw_datasets
    assert ql2_sixt_canada_analysis.discover_raw_csvs is discover_raw_csvs
    assert ql2_sixt_canada_analysis.DATASET_DEFINITIONS is DATASET_DEFINITIONS


def test_imports_perform_no_data_access_or_writes(tmp_path: Path) -> None:
    """Import every module in a fresh interpreter from another cwd and audit I/O."""
    script = f"""
import json, os, sys
data_dir = {str((PROJECT_ROOT / "data").resolve())!r}
events = []
def under_data(target):
    if not isinstance(target, (str, bytes, os.PathLike)):
        return False
    path = os.path.realpath(os.fsdecode(target))
    return path == data_dir or path.startswith(data_dir + os.sep)
def hook(event, args):
    if event in ("os.mkdir", "os.makedirs", "os.remove", "os.rename", "shutil.rmtree"):
        events.append(event)
    elif event == "open" and args:
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if any(m in mode for m in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            events.append("write-open")
        elif under_data(args[0]):
            events.append("data-read")
    elif event in ("os.listdir", "os.scandir") and args and under_data(args[0]):
        events.append("data-listing")
sys.addaudithook(hook)
import ql2_sixt_canada_analysis
import ql2_sixt_canada_analysis.paths
import ql2_sixt_canada_analysis.schemas
import ql2_sixt_canada_analysis.ingestion
print(json.dumps(events))
"""
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
