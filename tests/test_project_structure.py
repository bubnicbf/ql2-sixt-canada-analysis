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

from ql2_sixt_canada_analysis import paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rel(directory: Path) -> str:
    """Repository-relative POSIX path of a directory from the paths module."""
    return directory.relative_to(paths.PROJECT_ROOT).as_posix()


RAW, INTERIM, PROCESSED = (
    _rel(paths.RAW_DATA_DIR), _rel(paths.INTERIM_DATA_DIR), _rel(paths.PROCESSED_DATA_DIR)
)
PACKAGE_NAME = "ql2_sixt_canada_analysis"

REQUIRED_DIRECTORIES = [
    RAW,
    INTERIM,
    PROCESSED,
    "notebooks",
    f"src/{PACKAGE_NAME}",
    "tests",
    "reports",
    "reports/figures",
    "docs",
]


# Hypothetical generated outputs; they need not exist for check-ignore.
GENERATED_PATHS = [
    f"{INTERIM}/example_intermediate.parquet",
    f"{INTERIM}/example_intermediate.csv",
    f"{PROCESSED}/example_analysis_ready.parquet",
    f"{PROCESSED}/example_analysis_ready.csv",
    "reports/example_report.html",
    "reports/example_report.md",
    "reports/figures/example_chart.png",
    "reports/figures/example_chart.svg",
]

# Files that keep intentionally empty or documented directories in Git.
DIRECTORY_PLACEHOLDERS = [
    f"{RAW}/README.md",
    f"{INTERIM}/.gitkeep",
    f"{PROCESSED}/.gitkeep",
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


def test_local_raw_files_are_not_empty() -> None:
    # Raw files are proprietary and supplied locally; check sizes only, never
    # contents or names, and skip when none are present (e.g. a fresh clone).
    raw_files = [
        path for path in paths.RAW_DATA_DIR.iterdir()
        if path.is_file() and path.name not in {"README.md", ".gitkeep"}
    ]
    if not raw_files:
        pytest.skip("no locally supplied raw files in the raw-data directory")
    empty = [path for path in raw_files if path.stat().st_size == 0]
    assert not empty, f"{len(empty)} raw file(s) are empty"


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize(
    "relative_path",
    [f"{RAW}/synthetic_source.csv", f"{RAW}/nested/synthetic_source.csv"],
)
def test_raw_files_kept_out_of_git(relative_path: str) -> None:
    # Policy: supplied raw files are proprietary and must never be committed.
    assert _git_is_ignored(relative_path)


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("relative_path", GENERATED_PATHS)
def test_generated_outputs_are_ignored(relative_path: str) -> None:
    assert _git_is_ignored(relative_path)


QUALITY_OUTPUT_PATHS = [
    f"{INTERIM}/synthetic_blank_rows.csv",
    f"{PROCESSED}/synthetic_cleaned.parquet",
    "reports/synthetic_quality_report.json",
    "notebooks/synthetic_blank_rows_audit.json",
    "synthetic_removed_rows.txt",
    "synthetic_rejected_rows.json",
    "synthetic_quality_metrics.json",
    "notebooks/01_data_ingestion.executed.ipynb",
    "notebooks/01_data_ingestion.nbconvert.ipynb",
    "notebooks/.ipynb_checkpoints/01_data_ingestion-checkpoint.ipynb",
    "synthetic_extract.tmp.csv",
    ".venv/lib/python3.12/site-packages/x.py",
    "build/lib/x.py",
    f"src/{PACKAGE_NAME}/__pycache__/x.cpython-312.pyc",
    ".pytest_cache/v/cache/nodeids",
    "reports/synthetic_type_validation.json",
    "synthetic_dtype_report.md",
    "synthetic_identifier_values.txt",
    "notebooks/synthetic_identifier_extract.csv",
    "synthetic_diagnostics.json",
    "synthetic_run.log",
    f"{INTERIM}/synthetic_duplicate_keys.csv",
    "reports/synthetic_key_report.json",
    "synthetic_missing_key_rows.txt",
    "notebooks/synthetic_duplicate_key_extract.json",
    "synthetic_key_profile.md",
    "reports/synthetic_reconciliation_report.json",
    "synthetic_mismatch_jobs.txt",
    "notebooks/synthetic_orphan_details.json",
    "synthetic_missing_link_rows.md",
    f"{PROCESSED}/synthetic_orphans.parquet",
]


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("relative_path", QUALITY_OUTPUT_PATHS)
def test_quality_outputs_and_tooling_artifacts_are_ignored(relative_path: str) -> None:
    assert _git_is_ignored(relative_path)


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize(
    "relative_path",
    [f"src/{PACKAGE_NAME}/quality.py", "tests/test_quality.py", "notebooks/01_data_ingestion.ipynb",
     "tests/test_removed_rows_synthetic.py", "docs/quality_notes.md",
     f"src/{PACKAGE_NAME}/unique_keys.py", "tests/test_unique_keys.py", "tests/test_missing_key_rows.py",
     f"src/{PACKAGE_NAME}/reconciliation.py", "tests/test_reconciliation.py", "tests/test_orphan_details.py"],
)
def test_quality_source_tests_and_docs_are_not_ignored(relative_path: str) -> None:
    assert not _git_is_ignored(relative_path)


@pytest.mark.usefixtures("require_git")
def test_no_proprietary_data_file_is_tracked_or_staged() -> None:
    # Only placeholder/README files may be tracked or staged under the data
    # directories; file names are checked, never opened.
    for args in (["git", "ls-files", "--", RAW, INTERIM, PROCESSED],
                 ["git", "diff", "--cached", "--name-only", "--", RAW, INTERIM, PROCESSED]):
        listed = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        offending = [line for line in listed.stdout.splitlines()
                     if line and Path(line).name not in {"README.md", ".gitkeep"}]
        assert not offending, f"{len(offending)} data file(s) tracked or staged"


@pytest.mark.parametrize("relative_path", DIRECTORY_PLACEHOLDERS)
def test_directory_placeholder_exists(relative_path: str) -> None:
    assert (PROJECT_ROOT / relative_path).is_file()


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("relative_path", DIRECTORY_PLACEHOLDERS + TRACKED_SOURCE_PATHS)
def test_placeholders_and_source_are_not_ignored(relative_path: str) -> None:
    assert not _git_is_ignored(relative_path)
