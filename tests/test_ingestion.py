"""Tests for raw-CSV discovery and loading.

All inputs are tiny, obviously synthetic CSVs written to pytest's ``tmp_path``.
The proprietary raw files are never read by this suite.
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

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis import ingestion
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
    default_raw_dir,
    discover_raw_csvs,
    load_raw_datasets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_JOBS_CSV = "widget_id,colour\n1,red\n2,blue\n"
SYNTHETIC_CARS_CSV = "gadget_id,size,score\n10,small,0.5\n20,large,1.5\n30,medium,2.5\n"


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "raw"
    directory.mkdir()
    _write(directory, "synthetic_jobs.csv", SYNTHETIC_JOBS_CSV)
    _write(directory, "synthetic_cars.csv", SYNTHETIC_CARS_CSV)
    return directory


def _snapshot(root: Path) -> dict[str, str]:
    """Map every file under ``root`` to a content hash (directories marked)."""
    snapshot = {}
    for path in sorted(root.rglob("*")):
        key = path.relative_to(root).as_posix()
        snapshot[key] = "<dir>" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


# --------------------------------------------------------------------- discovery


def test_discovers_one_jobs_and_one_cars_csv(raw_dir: Path) -> None:
    paths = discover_raw_csvs(raw_dir)
    assert isinstance(paths, RawDatasetPaths)
    assert paths.jobs == raw_dir / "synthetic_jobs.csv"
    assert paths.cars == raw_dir / "synthetic_cars.csv"


def test_discovery_is_case_insensitive(tmp_path: Path) -> None:
    jobs = _write(tmp_path, "Synthetic_JOBS.CSV", SYNTHETIC_JOBS_CSV)
    cars = _write(tmp_path, "synthetic_Cars.Csv", SYNTHETIC_CARS_CSV)
    assert discover_raw_csvs(tmp_path) == RawDatasetPaths(jobs=jobs, cars=cars)


@pytest.mark.parametrize(
    ("jobs_name", "cars_name"),
    [
        ("demo export 01-99 jobs (1).csv", "demo export 01-99 cars (1).csv"),
        ("2001_jobs_feed (12).csv", "2001-cars-feed (3).csv"),
        ("alpha.jobs.v2.csv", "alpha.cars.v2.csv"),
        # Shared prefix mentioning the other role: the last role token wins.
        ("demo_jobs_5_6_jobs_extract (1).csv", "demo_jobs_5_6_cars_extract (1).csv"),
        ("demo cars batch - jobs.csv", "demo jobs batch - cars.csv"),
    ],
)
def test_discovery_tolerates_variable_filename_components(
    tmp_path: Path, jobs_name: str, cars_name: str
) -> None:
    jobs = _write(tmp_path, jobs_name, SYNTHETIC_JOBS_CSV)
    cars = _write(tmp_path, cars_name, SYNTHETIC_CARS_CSV)
    assert discover_raw_csvs(tmp_path) == RawDatasetPaths(jobs=jobs, cars=cars)


def test_unrelated_and_non_csv_files_are_ignored(raw_dir: Path) -> None:
    _write(raw_dir, "notes.csv", "a,b\n1,2\n")
    _write(raw_dir, "jobsite_summary.csv", "a,b\n1,2\n")  # 'jobsite' is not a role token
    _write(raw_dir, "README.md", "synthetic readme\n")
    _write(raw_dir, ".gitkeep", "")
    _write(raw_dir, "extra_jobs.txt", "not a csv\n")
    _write(raw_dir, "extra_cars.parquet", "not a csv\n")
    _write(raw_dir, "extra_jobs.csv.bak", "not a csv\n")
    paths = discover_raw_csvs(raw_dir)
    assert paths.jobs.name == "synthetic_jobs.csv"
    assert paths.cars.name == "synthetic_cars.csv"


def test_discovery_does_not_recurse(raw_dir: Path) -> None:
    nested = raw_dir / "nested"
    nested.mkdir()
    _write(nested, "other_jobs.csv", SYNTHETIC_JOBS_CSV)
    _write(nested, "other_cars.csv", SYNTHETIC_CARS_CSV)
    paths = discover_raw_csvs(raw_dir)
    assert paths.jobs.parent == raw_dir and paths.cars.parent == raw_dir


def test_discovery_accepts_str_and_pathlike(raw_dir: Path) -> None:
    assert discover_raw_csvs(str(raw_dir)) == discover_raw_csvs(raw_dir)


# ----------------------------------------------------------------------- loading


def test_loads_both_datasets_as_separate_dataframes(raw_dir: Path) -> None:
    datasets = load_raw_datasets(raw_dir)
    assert isinstance(datasets, RawDatasets)
    assert isinstance(datasets.jobs, pd.DataFrame)
    assert isinstance(datasets.cars, pd.DataFrame)
    assert datasets.jobs is not datasets.cars


def test_loaded_frames_match_synthetic_inputs(raw_dir: Path) -> None:
    datasets = load_raw_datasets(raw_dir)
    pd.testing.assert_frame_equal(
        datasets.jobs, pd.DataFrame({"widget_id": [1, 2], "colour": ["red", "blue"]})
    )
    pd.testing.assert_frame_equal(
        datasets.cars,
        pd.DataFrame(
            {
                "gadget_id": [10, 20, 30],
                "size": ["small", "large", "medium"],
                "score": [0.5, 1.5, 2.5],
            }
        ),
    )


def test_loading_preserves_source_values_without_cleaning(tmp_path: Path) -> None:
    # Duplicates, padding and blanks must come through as pandas reads them.
    _write(tmp_path, "x_jobs.csv", "code,label\n007, padded \n007, padded \n,\n")
    _write(tmp_path, "x_cars.csv", SYNTHETIC_CARS_CSV)
    jobs = load_raw_datasets(tmp_path).jobs
    assert len(jobs) == 3
    assert jobs["label"].iloc[0] == " padded "
    assert jobs.iloc[2].isna().all()


def test_alternate_raw_directory(tmp_path: Path) -> None:
    alternate = tmp_path / "somewhere" / "else"
    alternate.mkdir(parents=True)
    _write(alternate, "a_jobs.csv", SYNTHETIC_JOBS_CSV)
    _write(alternate, "a_cars.csv", SYNTHETIC_CARS_CSV)
    datasets = load_raw_datasets(alternate)
    assert list(datasets.jobs.columns) == ["widget_id", "colour"]


def test_read_csv_options_are_applied_to_both_files(raw_dir: Path) -> None:
    datasets = load_raw_datasets(raw_dir, read_csv_options={"dtype": str})
    assert datasets.jobs["widget_id"].tolist() == ["1", "2"]
    assert datasets.cars["score"].tolist() == ["0.5", "1.5", "2.5"]


@pytest.mark.parametrize("option", ["chunksize", "iterator", "filepath_or_buffer"])
def test_options_that_break_the_contract_are_rejected(raw_dir: Path, option: str) -> None:
    with pytest.raises(ValueError, match=option):
        load_raw_datasets(raw_dir, read_csv_options={option: 1})


def test_each_file_is_read_exactly_once(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    real_read_csv = pd.read_csv

    def counting_read_csv(path, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(Path(path))
        return real_read_csv(path, **kwargs)

    monkeypatch.setattr(ingestion.pd, "read_csv", counting_read_csv)
    load_raw_datasets(raw_dir)
    assert sorted(p.name for p in calls) == ["synthetic_cars.csv", "synthetic_jobs.csv"]


def test_loader_writes_no_files_and_leaves_sources_unchanged(raw_dir: Path) -> None:
    root = raw_dir.parent
    before = _snapshot(root)
    mtimes = {p: p.stat().st_mtime_ns for p in raw_dir.iterdir()}
    load_raw_datasets(raw_dir)
    assert _snapshot(root) == before
    assert {p: p.stat().st_mtime_ns for p in raw_dir.iterdir()} == mtimes


# ------------------------------------------------------------------------ errors


def test_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(RawDataDirectoryError, match="does not exist") as info:
        load_raw_datasets(missing)
    assert info.value.path == missing


def test_path_that_is_not_a_directory(tmp_path: Path) -> None:
    a_file = _write(tmp_path, "plain.txt", "synthetic\n")
    with pytest.raises(RawDataDirectoryError, match="not a directory"):
        discover_raw_csvs(a_file)


def test_missing_jobs_csv(tmp_path: Path) -> None:
    _write(tmp_path, "synthetic_cars.csv", SYNTHETIC_CARS_CSV)
    with pytest.raises(DatasetNotFoundError, match="'jobs'") as info:
        discover_raw_csvs(tmp_path)
    assert info.value.role == "jobs"


def test_missing_cars_csv(tmp_path: Path) -> None:
    _write(tmp_path, "synthetic_jobs.csv", SYNTHETIC_JOBS_CSV)
    with pytest.raises(DatasetNotFoundError, match="'cars'") as info:
        discover_raw_csvs(tmp_path)
    assert info.value.role == "cars"


@pytest.mark.parametrize("role", ["jobs", "cars"])
def test_multiple_candidates_are_ambiguous(raw_dir: Path, role: str) -> None:
    _write(raw_dir, f"synthetic_{role} (1).csv", "a\n1\n")
    with pytest.raises(AmbiguousDatasetError, match=f"2 candidate CSVs for the '{role}'") as info:
        discover_raw_csvs(raw_dir)
    assert info.value.role == role
    assert len(info.value.candidates) == 2


def test_matching_path_that_is_not_a_regular_file(tmp_path: Path) -> None:
    (tmp_path / "synthetic_jobs.csv").mkdir()
    _write(tmp_path, "synthetic_cars.csv", SYNTHETIC_CARS_CSV)
    with pytest.raises(NotARegularFileError, match="'jobs'"):
        discover_raw_csvs(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        b"",  # empty file
        b"col_a,col_b\n1,2\nSECRETTOKEN,4,5,6\n",  # ragged rows
        b"col_a,col_b\n\xff\xfe\xfa,SECRETTOKEN\n",  # invalid UTF-8
    ],
    ids=["empty", "ragged", "bad-encoding"],
)
def test_unreadable_csv_raises_load_error_with_cause(tmp_path: Path, payload: bytes) -> None:
    (tmp_path / "synthetic_jobs.csv").write_bytes(payload)
    _write(tmp_path, "synthetic_cars.csv", SYNTHETIC_CARS_CSV)
    with pytest.raises(RawDataLoadError, match="'jobs'") as info:
        load_raw_datasets(tmp_path)
    assert info.value.role == "jobs"
    assert isinstance(info.value.__cause__, ValueError)
    # Messages name the role only: no file contents and no paths.
    assert "SECRETTOKEN" not in str(info.value)
    assert str(tmp_path) not in str(info.value)


def test_exception_hierarchy() -> None:
    for error in (RawDataDirectoryError, DatasetNotFoundError, AmbiguousDatasetError,
                  NotARegularFileError):
        assert issubclass(error, RawDataDiscoveryError)
    assert issubclass(RawDataDiscoveryError, IngestionError)
    assert issubclass(RawDataLoadError, IngestionError)


# ------------------------------------------------------- defaults and import safety


def test_default_raw_dir_is_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_raw_dir() == (PROJECT_ROOT / "data" / "raw").resolve()


def test_package_exposes_public_ingestion_api() -> None:
    assert ql2_sixt_canada_analysis.load_raw_datasets is load_raw_datasets
    assert ql2_sixt_canada_analysis.discover_raw_csvs is discover_raw_csvs
    assert ql2_sixt_canada_analysis.RawDatasets is RawDatasets
    assert ql2_sixt_canada_analysis.RawDatasetPaths is RawDatasetPaths


def test_importing_package_does_not_touch_data_directory(tmp_path: Path) -> None:
    """Import in a fresh interpreter and audit every file/directory access."""
    script = f"""
import json, os, sys
data_dir = {str((PROJECT_ROOT / "data").resolve())!r}
hits = []
def hook(event, args):
    if event in ("open", "os.listdir", "os.scandir") and args and args[0] is not None:
        target = args[0]
        if isinstance(target, (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(target))
            if path == data_dir or path.startswith(data_dir + os.sep):
                hits.append(event)
sys.addaudithook(hook)
import ql2_sixt_canada_analysis
import ql2_sixt_canada_analysis.ingestion
print(json.dumps(hits))
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
    )}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
