"""Structural and execution-order tests for the project notebooks.

Execution runs a copy of each notebook from a clean kernel against synthetic
CSVs generated in ``tmp_path``; the tracked notebook files are only read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from conftest import contract_columns, write_synthetic_csv

from ql2_sixt_canada_analysis import notebook_validation, paths
from ql2_sixt_canada_analysis.notebook_validation import (
    NOTEBOOK_FORMAT_MAJOR,
    NotebookExecutionError,
    NotebookOrderError,
    check_execution_order,
    execute_notebook_copy,
    read_notebook,
)
from ql2_sixt_canada_analysis.schemas import DATASET_DEFINITIONS, DatasetKey

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
INGESTION_NOTEBOOK = NOTEBOOKS_DIR / "01_data_ingestion.ipynb"
# Authoritative order: numeric prefix, ascending.
TRACKED_NOTEBOOKS = sorted(
    p for p in NOTEBOOKS_DIR.glob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
)
NOTEBOOK_IDS = [p.name for p in TRACKED_NOTEBOOKS]


def _code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [c for c in notebook.cells if c.cell_type == "code"]


def _code_source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(c.source for c in _code_cells(notebook))


def _snapshot(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        and ".git" not in p.parts and not p.name.endswith(".pyc")
        and ".pytest_cache" not in p.parts and ".egg-info" not in "".join(p.parts)
    }


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "synthetic_raw"
    directory.mkdir()
    for key in DatasetKey:
        write_synthetic_csv(directory / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    return directory


# ------------------------------------------------------------------- inventory


def test_notebooks_directory_and_ingestion_notebook_exist() -> None:
    assert NOTEBOOKS_DIR.is_dir()
    assert INGESTION_NOTEBOOK.is_file()
    assert INGESTION_NOTEBOOK in TRACKED_NOTEBOOKS


def test_notebooks_have_numeric_order_prefixes() -> None:
    assert TRACKED_NOTEBOOKS, "expected at least one notebook"
    prefixes = [re.match(r"(\d{2})_", p.name) for p in TRACKED_NOTEBOOKS]
    assert all(prefixes), "every notebook needs a two-digit order prefix like 01_"
    assert len({m.group(1) for m in prefixes}) == len(prefixes), "order prefixes must be unique"


def test_readme_documents_every_notebook_in_order() -> None:
    readme = (NOTEBOOKS_DIR / "README.md").read_text(encoding="utf-8")
    positions = [readme.find(p.name) for p in TRACKED_NOTEBOOKS]
    assert all(pos >= 0 for pos in positions), "every notebook must be listed in notebooks/README.md"
    assert positions == sorted(positions), "README must list notebooks in execution order"
    assert paths.RAW_DATA_DIR_ENV_VAR in readme


# ------------------------------------------------------------ clean source policy


@pytest.mark.parametrize("notebook_path", TRACKED_NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebook_is_valid_supported_format(notebook_path: Path) -> None:
    raw = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert raw["nbformat"] == NOTEBOOK_FORMAT_MAJOR
    read_notebook(notebook_path)  # validates against the schema


@pytest.mark.parametrize("notebook_path", TRACKED_NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_committed_code_cells_are_clean(notebook_path: Path) -> None:
    notebook = read_notebook(notebook_path)
    cells = _code_cells(notebook)
    assert cells, "notebook has no code cells"
    for index, cell in enumerate(cells):
        assert cell.execution_count is None, f"code cell {index} has an execution count"
        assert cell.outputs == [], f"code cell {index} has outputs"
        assert "execution" not in cell.metadata, f"code cell {index} keeps execution timing"


@pytest.mark.parametrize("notebook_path", TRACKED_NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebook_metadata_is_portable(notebook_path: Path) -> None:
    notebook = read_notebook(notebook_path)
    assert "widgets" not in notebook.metadata
    assert "path" not in notebook.metadata
    kernelspec = notebook.metadata.get("kernelspec", {})
    assert kernelspec.get("language") == "python"
    text = json.dumps(notebook.metadata)
    assert not re.search(r"(?<![\w\"])/(?:Users|home|sessions|tmp)/", text), "local path in metadata"
    assert os.environ.get("USER", "\0") not in text


@pytest.mark.parametrize("notebook_path", TRACKED_NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebook_has_no_machine_specific_or_proprietary_references(notebook_path: Path) -> None:
    text = notebook_path.read_text(encoding="utf-8")
    assert not re.search(r"(?<![\w\"])/(?:Users|home|sessions|tmp|mnt)/", text), "absolute path"
    assert not re.search(r"[A-Za-z]:\\\\", text), "Windows absolute path"
    code = _code_source(read_notebook(notebook_path))
    assert "os.chdir" not in code
    assert not re.search(r"sys\.path\s*\.\s*(append|insert|extend)", code)
    assert not re.search(r"(?m)^\s*[!%]\s*(pip|conda|uv|poetry|mamba)\b", code)
    assert not re.search(r"(?m)^\s*%\s*cd\b", code)
    assert not re.search(r"\.csv['\"]", code), "notebooks must not name CSV files"


@pytest.mark.parametrize("notebook_path", TRACKED_NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebook_uses_package_apis_not_direct_reads(notebook_path: Path) -> None:
    code = _code_source(read_notebook(notebook_path))
    assert "read_csv" not in code, "use the package ingestion API"
    assert not re.search(r"\.to_(csv|parquet|feather|pickle)\(", code)
    assert "RAW_DATA_DIR" not in code or "paths" in code


def test_ingestion_notebook_imports_centralized_apis() -> None:
    code = _code_source(read_notebook(INGESTION_NOTEBOOK))
    assert re.search(r"from ql2_sixt_canada_analysis(\.ingestion)? import .*load_raw_datasets", code)
    assert re.search(r"from ql2_sixt_canada_analysis\.paths import .*resolve_raw_data_dir", code)
    assert "resolve_raw_data_dir(" in code
    assert "load_raw_datasets(" in code


# ----------------------------------------------------------------- execution


@pytest.mark.parametrize("notebook_path", TRACKED_NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebook_runs_top_to_bottom_from_clean_kernel(
    notebook_path: Path, synthetic_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # not the repository root
    workdir = tmp_path / "kernel_cwd"
    workdir.mkdir()
    source_bytes = notebook_path.read_bytes()
    repo_before = _snapshot(PROJECT_ROOT)

    result = execute_notebook_copy(
        notebook_path,
        workdir=workdir,
        env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)},
        output_path=tmp_path / "executed.ipynb",
    )

    code_cells = _code_cells(result.executed)
    assert result.execution_counts == tuple(range(1, len(code_cells) + 1))
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c.outputs)
    assert notebook_path.read_bytes() == source_bytes, "tracked notebook was modified"
    assert _snapshot(PROJECT_ROOT) == repo_before, "execution changed repository files"
    assert [p.name for p in workdir.iterdir()] == [], "kernel wrote into its working directory"
    assert (tmp_path / "executed.ipynb").is_file()


def test_ingestion_notebook_loads_both_datasets_without_showing_contents(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    result = execute_notebook_copy(
        INGESTION_NOTEBOOK, workdir=tmp_path,
        env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)},
    )
    outputs = "\n".join(
        o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs
    )
    for key in DatasetKey:
        assert key.value in outputs  # final cell confirms both logical datasets
    assert "synthetic_r0_c0" not in outputs, "cell output shows data values"
    assert str(synthetic_raw_dir) not in outputs, "cell output shows a path"
    assert not any(column in outputs for key in DatasetKey for column in contract_columns(key))
    assert not any(
        o.get("output_type") in {"execute_result", "display_data"}
        for c in _code_cells(result.executed) for o in c.outputs
    ), "notebook displays objects"


def test_notebook_fails_clearly_without_raw_data(tmp_path: Path) -> None:
    empty = tmp_path / "empty_raw"
    empty.mkdir()
    with pytest.raises(NotebookExecutionError) as info:
        execute_notebook_copy(
            INGESTION_NOTEBOOK, workdir=tmp_path, env={paths.RAW_DATA_DIR_ENV_VAR: str(empty)}
        )
    assert info.value.cell_index >= 0
    assert info.value.__cause__ is not None
    assert str(empty) not in str(info.value)


# ------------------------------------------------------------- validator unit


def _notebook_with_counts(counts: list[int | None]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    for count in counts:
        cell = nbformat.v4.new_code_cell("pass")
        cell.execution_count = count
        nb.cells.append(cell)
    return nb


@pytest.mark.parametrize("counts", [[1, 2, 3], [1], []])
def test_check_execution_order_accepts_consecutive(counts: list[int]) -> None:
    assert check_execution_order(_notebook_with_counts(counts)) == tuple(counts)


@pytest.mark.parametrize("counts", [[1, None], [2, 1], [1, 3], [2, 3], [1, 1]])
def test_check_execution_order_rejects_out_of_order(counts: list[int | None]) -> None:
    with pytest.raises(NotebookOrderError):
        check_execution_order(_notebook_with_counts(counts))


def test_execution_error_reports_failing_cell_only(tmp_path: Path) -> None:
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("ok = 1"), nbformat.v4.new_code_cell("raise RuntimeError('SECRETTOKEN')")]
    )
    nb.metadata["kernelspec"] = {"name": "python3", "language": "python", "display_name": "Python 3"}
    path = tmp_path / "failing.ipynb"
    nbformat.write(nb, path)
    with pytest.raises(NotebookExecutionError) as info:
        execute_notebook_copy(path, workdir=tmp_path)
    assert info.value.cell_index == 1
    assert "SECRETTOKEN" not in str(info.value)
    assert "SECRETTOKEN" in str(info.value.__cause__)  # preserved for diagnosis


# --------------------------------------------------------------- override API


def test_resolve_raw_data_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.RAW_DATA_DIR_ENV_VAR, raising=False)
    assert paths.resolve_raw_data_dir() == paths.RAW_DATA_DIR
    monkeypatch.setenv(paths.RAW_DATA_DIR_ENV_VAR, "")
    assert paths.resolve_raw_data_dir() == paths.RAW_DATA_DIR
    monkeypatch.setenv(paths.RAW_DATA_DIR_ENV_VAR, str(tmp_path))
    assert paths.resolve_raw_data_dir() == tmp_path
    assert paths.resolve_raw_data_dir(tmp_path / "explicit") == tmp_path / "explicit"
    assert isinstance(paths.resolve_raw_data_dir(), Path)


def test_env_var_name_is_project_specific() -> None:
    assert paths.RAW_DATA_DIR_ENV_VAR.startswith("QL2_")
    assert paths.RAW_DATA_DIR_ENV_VAR.isupper()


def test_importing_notebook_support_performs_no_data_access(tmp_path: Path) -> None:
    script = f"""
import json, os, sys
data_dir = {str((PROJECT_ROOT / "data").resolve())!r}
events = []
def under_data(t):
    if not isinstance(t, (str, bytes, os.PathLike)):
        return False
    p = os.path.realpath(os.fsdecode(t))
    return p == data_dir or p.startswith(data_dir + os.sep)
def hook(event, args):
    if event in ("open", "os.listdir", "os.scandir") and args and under_data(args[0]):
        events.append(event)
    if event in ("os.mkdir", "os.makedirs", "subprocess.Popen"):
        events.append(event)
sys.addaudithook(hook)
import ql2_sixt_canada_analysis.notebook_validation
import ql2_sixt_canada_analysis.paths
print(json.dumps(events))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
    assert notebook_validation.NOTEBOOK_FORMAT_MAJOR == 4
