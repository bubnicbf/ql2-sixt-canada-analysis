"""Validate the repository layout and Git data-handling policy.

These tests are read-only with respect to the repository: they inspect paths
and ask Git how ignore rules apply, but never create, modify, or delete files
under the project root.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ql2_sixt_canada_analysis"

REQUIRED_DIRECTORIES = [
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
    f"src/{PACKAGE_NAME}",
    "tests",
    "reports",
    "reports/figures",
    "docs",
]

RAW_CSV_FILES = [
    "data/raw/sixt_canada_jobs_153_422_jobs_raw (1).csv",
    "data/raw/sixt_canada_jobs_153_422_cars_raw (1).csv",
]

# Hypothetical generated outputs; they need not exist for check-ignore.
GENERATED_PATHS = [
    "data/interim/example_intermediate.parquet",
    "data/interim/example_intermediate.csv",
    "data/processed/example_analysis_ready.parquet",
    "data/processed/example_analysis_ready.csv",
    "reports/example_report.html",
    "reports/example_report.md",
    "reports/figures/example_chart.png",
    "reports/figures/example_chart.svg",
]

# Files that keep intentionally empty or documented directories in Git.
DIRECTORY_PLACEHOLDERS = [
    "data/raw/README.md",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
    "notebooks/README.md",
    "reports/.gitkeep",
    "reports/figures/.gitkeep",
    "docs/.gitkeep",
]

TRACKED_SOURCE_PATHS = [
    f"src/{PACKAGE_NAME}/__init__.py",
    "tests/test_project_structure.py",
    "README.md",
    "pyproject.toml",
    ".gitignore",
]


def _git_is_ignored(relative_path: str) -> bool:
    """Return True if Git's ignore rules match ``relative_path``.

    ``--no-index`` evaluates the ignore rules alone, so the answer does not
    depend on whether the path is currently tracked or exists on disk.
    """
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", relative_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f"git check-ignore failed for {relative_path!r}: {result.stderr}")
    return result.returncode == 0


@pytest.fixture(scope="module")
def require_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip("project root is not inside a Git work tree")


@pytest.mark.parametrize("relative_dir", REQUIRED_DIRECTORIES)
def test_required_directory_exists(relative_dir: str) -> None:
    assert (PROJECT_ROOT / relative_dir).is_dir(), f"missing directory: {relative_dir}"


def test_package_init_exists() -> None:
    assert (PROJECT_ROOT / "src" / PACKAGE_NAME / "__init__.py").is_file()


def test_package_is_importable_from_src() -> None:
    module = importlib.import_module(PACKAGE_NAME)
    module_path = Path(module.__file__).resolve()
    assert module_path.is_relative_to((PROJECT_ROOT / "src").resolve())


@pytest.mark.parametrize("relative_path", RAW_CSV_FILES)
def test_raw_csv_present_and_not_empty(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        pytest.skip(
            f"{relative_path} is proprietary and not distributed with the "
            "repository; place it in data/raw/ to run this check"
        )
    assert path.is_file()
    assert path.stat().st_size > 0, f"raw file is empty: {relative_path}"


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("relative_path", RAW_CSV_FILES)
def test_raw_csv_kept_out_of_git(relative_path: str) -> None:
    # Policy: the supplied raw CSVs are proprietary and must never be committed.
    assert _git_is_ignored(relative_path)


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("relative_path", GENERATED_PATHS)
def test_generated_outputs_are_ignored(relative_path: str) -> None:
    assert _git_is_ignored(relative_path)


@pytest.mark.parametrize("relative_path", DIRECTORY_PLACEHOLDERS)
def test_directory_placeholder_exists(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).is_file()


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("relative_path", DIRECTORY_PLACEHOLDERS + TRACKED_SOURCE_PATHS)
def test_placeholders_and_source_are_not_ignored(relative_path: str) -> None:
    assert not _git_is_ignored(relative_path)
