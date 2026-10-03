"""Validate the Python environment configuration and data-safety ignore rules.

These tests read ``pyproject.toml`` and ask Git how its ignore rules apply.
They never open, list, or modify files under the protected data directories.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from ql2_sixt_canada_analysis import paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
PROJECT_NAME = "ql2-sixt-canada-analysis"
PACKAGE_NAME = "ql2_sixt_canada_analysis"

RUNTIME_DEPENDENCIES = ["pandas", "numpy", "pyarrow", "matplotlib", "seaborn", "scipy"]
RUNTIME_IMPORTS = ["pandas", "numpy", "pyarrow", "matplotlib", "seaborn", "scipy"]
DEV_DEPENDENCIES = ["pytest", "jupyterlab"]

PROTECTED_DATA_DIRS = [
    d.relative_to(paths.PROJECT_ROOT).as_posix()
    for d in (paths.RAW_DATA_DIR, paths.INTERIM_DATA_DIR, paths.PROCESSED_DATA_DIR)
]
DATA_EXTENSIONS = [
    "csv", "tsv", "parquet", "feather", "json", "xlsx", "pkl", "pickle", "zip", "gz", "csv.gz",
]
SAFE_PLACEHOLDER_NAMES = {"README.md", ".gitkeep"}


# --------------------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(item).name) for item in requirements}


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def require_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")
    probe = _git("rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip("project root is not inside a Git work tree")


def _is_ignored(relative_path: str) -> bool:
    # --no-index applies the ignore rules alone, independent of tracking state
    # and of whether the (hypothetical) path exists on disk.
    result = _git("check-ignore", "--quiet", "--no-index", relative_path)
    if result.returncode not in (0, 1):
        pytest.fail(f"git check-ignore failed for {relative_path!r}: {result.stderr}")
    return result.returncode == 0


# ---------------------------------------------------------------- project metadata


def test_pyproject_is_valid_toml_with_project_table(pyproject: dict) -> None:
    assert isinstance(pyproject.get("project"), dict)


def test_project_name_and_version(pyproject: dict) -> None:
    project = pyproject["project"]
    assert project["name"] == PROJECT_NAME
    assert project.get("version"), "project version must be declared"


def test_requires_python_supports_311_baseline(pyproject: dict) -> None:
    spec = SpecifierSet(pyproject["project"]["requires-python"])
    assert "3.11" in spec, "Python 3.11 must be supported"
    assert "3.12" in spec, "newer 3.x releases should not be excluded"
    assert "3.10" not in spec, "Python older than 3.11 must not be declared supported"


@pytest.mark.parametrize("dependency", RUNTIME_DEPENDENCIES)
def test_runtime_dependency_declared(pyproject: dict, dependency: str) -> None:
    declared = _requirement_names(pyproject["project"].get("dependencies", []))
    assert canonicalize_name(dependency) in declared


@pytest.mark.parametrize("dependency", DEV_DEPENDENCIES)
def test_dev_extra_provides_tool(pyproject: dict, dependency: str) -> None:
    extras = pyproject["project"].get("optional-dependencies", {})
    assert "dev" in extras, "a 'dev' extra is required for pip install -e '.[dev]'"
    assert canonicalize_name(dependency) in _requirement_names(extras["dev"])


def test_dependencies_are_not_declared_twice(pyproject: dict) -> None:
    runtime = _requirement_names(pyproject["project"]["dependencies"])
    dev = _requirement_names(pyproject["project"]["optional-dependencies"]["dev"])
    assert not runtime & dev


# ------------------------------------------------------------- build and packaging


def test_build_system_uses_setuptools_backend(pyproject: dict) -> None:
    build_system = pyproject["build-system"]
    assert build_system["build-backend"] == "setuptools.build_meta"
    assert "setuptools" in _requirement_names(build_system["requires"])


def test_build_requirement_supports_editable_installs(pyproject: dict) -> None:
    requirements = [Requirement(r) for r in pyproject["build-system"]["requires"]]
    setuptools_req = next(r for r in requirements if r.name == "setuptools")
    # PEP 660 editable installs need setuptools >= 64; an unbounded spec would
    # allow older versions that cannot perform `pip install -e .`.
    assert "63.4" not in setuptools_req.specifier
    assert "64" in setuptools_req.specifier


def test_package_discovery_targets_src_layout(pyproject: dict) -> None:
    where = pyproject["tool"]["setuptools"]["packages"]["find"]["where"]
    assert where == ["src"]
    assert pyproject["tool"]["pytest"]["ini_options"]["pythonpath"] == where


def test_package_is_discoverable_under_src(pyproject: dict) -> None:
    where = PROJECT_ROOT / pyproject["tool"]["setuptools"]["packages"]["find"]["where"][0]
    discovered = {init.parent.name for init in where.glob("*/__init__.py")}
    assert PACKAGE_NAME in discovered
    assert PACKAGE_NAME.isidentifier()


def test_package_imports_from_src_checkout() -> None:
    module = importlib.import_module(PACKAGE_NAME)
    expected = (PROJECT_ROOT / "src" / PACKAGE_NAME / "__init__.py").resolve()
    assert Path(module.__file__).resolve() == expected, (
        "package resolved to an unexpected copy instead of this checkout's src/"
    )


def test_installed_distribution_matches_pyproject(pyproject: dict) -> None:
    try:
        installed_version = importlib.metadata.version(PROJECT_NAME)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("project is not installed; run python -m pip install -e '.[dev]'")
    assert installed_version == pyproject["project"]["version"]


@pytest.mark.parametrize("module_name", RUNTIME_IMPORTS)
def test_runtime_library_is_importable(module_name: str) -> None:
    importlib.import_module(module_name)


def test_jupyterlab_is_installed() -> None:
    assert importlib.util.find_spec("jupyterlab") is not None


# ------------------------------------------------------------ Git ignore and safety


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("data_dir", PROTECTED_DATA_DIRS)
@pytest.mark.parametrize("extension", DATA_EXTENSIONS)
def test_data_files_are_ignored(data_dir: str, extension: str) -> None:
    assert _is_ignored(f"{data_dir}/synthetic_example.{extension}")
    assert _is_ignored(f"{data_dir}/nested/synthetic_example.{extension}")


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("location", ["", "notebooks/", "reports/", "src/", "tests/"])
@pytest.mark.parametrize(
    "extension", ["csv", "tsv", "parquet", "feather", "pkl", "pickle", "zip", "gz", "csv.gz"]
)
def test_derived_data_formats_are_ignored_outside_data_dirs(location: str, extension: str) -> None:
    # Defence in depth: derived datasets saved by mistake elsewhere stay out of Git.
    assert _is_ignored(f"{location}synthetic_export.{extension}")


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize("data_dir", PROTECTED_DATA_DIRS)
@pytest.mark.parametrize("placeholder", sorted(SAFE_PLACEHOLDER_NAMES))
def test_data_placeholders_are_not_ignored(data_dir: str, placeholder: str) -> None:
    assert not _is_ignored(f"{data_dir}/{placeholder}")


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize(
    "relative_path",
    [
        ".venv/pyvenv.cfg",
        "venv/pyvenv.cfg",
        "env/pyvenv.cfg",
        "src/__pycache__/module.cpython-311.pyc",
        ".pytest_cache/README.md",
        ".coverage",
        "htmlcov/index.html",
        "build/lib/module.py",
        "dist/package-0.0.0.tar.gz",
        f"src/{PACKAGE_NAME}.egg-info/PKG-INFO",
        "notebooks/.ipynb_checkpoints/example-checkpoint.ipynb",
        ".DS_Store",
        ".idea/workspace.xml",
        ".vscode/settings.json",
    ],
)
def test_local_environment_artifacts_are_ignored(relative_path: str) -> None:
    assert _is_ignored(relative_path)


@pytest.mark.usefixtures("require_git")
@pytest.mark.parametrize(
    "relative_path",
    ["pyproject.toml", "README.md", f"src/{PACKAGE_NAME}/__init__.py", "tests/test_environment.py"],
)
def test_project_files_are_not_ignored(relative_path: str) -> None:
    assert not _is_ignored(relative_path)


@pytest.mark.usefixtures("require_git")
def test_only_placeholders_are_tracked_or_staged_under_data() -> None:
    # `git ls-files` lists both committed and staged paths; names only, never content.
    result = _git("ls-files", "--", *PROTECTED_DATA_DIRS)
    assert result.returncode == 0, result.stderr
    unsafe = [
        line for line in result.stdout.splitlines()
        if Path(line).name not in SAFE_PLACEHOLDER_NAMES
    ]
    assert not unsafe, f"{len(unsafe)} non-placeholder file(s) tracked under data/"
