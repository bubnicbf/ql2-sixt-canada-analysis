"""Tests for the centralized path configuration."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from ql2_sixt_canada_analysis import paths

TEST_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_root_is_this_checkout() -> None:
    assert isinstance(paths.PROJECT_ROOT, Path)
    assert paths.PROJECT_ROOT.resolve() == TEST_PROJECT_ROOT


def test_data_directories_derive_from_single_root() -> None:
    assert paths.DATA_DIR == paths.PROJECT_ROOT / "data"
    for directory, name in [
        (paths.RAW_DATA_DIR, "raw"),
        (paths.INTERIM_DATA_DIR, "interim"),
        (paths.PROCESSED_DATA_DIR, "processed"),
    ]:
        assert isinstance(directory, Path)
        assert directory == paths.DATA_DIR / name
        assert directory.is_relative_to(paths.PROJECT_ROOT)


def test_paths_do_not_depend_on_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = (paths.PROJECT_ROOT, paths.RAW_DATA_DIR)
    monkeypatch.chdir(tmp_path)
    reloaded = importlib.reload(paths)
    assert (reloaded.PROJECT_ROOT, reloaded.RAW_DATA_DIR) == before
    assert reloaded.PROJECT_ROOT.is_absolute()


def test_paths_module_contains_no_hard_coded_locations() -> None:
    tree = ast.parse(inspect.getsource(paths))
    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    code_literals = [s for s in literals if not any(ch.isspace() for ch in s)]  # skip docstrings
    path_literals = [s for s in code_literals if "/" in s or "\\" in s or ":" in s[:3]]
    assert path_literals == [], "paths must be built from PROJECT_ROOT, not literals"
    # Only directory-name segments are allowed as path literals.
    segments = {s for s in literals if " " not in s and "\n" not in s and s.isidentifier()}
    assert {"data", "raw", "interim", "processed"} <= segments
