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
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    EXPECTED_LOCATION_COVERAGE,
    INVESTIGATED_LOCATION_STREAM,
    JOB_DETAIL_RELATIONSHIP,
    DatasetKey,
)

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
    assert re.search(r"from ql2_sixt_canada_analysis(\.ingestion)? import\s*\(?[^)]*?\bload_raw_datasets\b", code, re.S)
    assert re.search(r"from ql2_sixt_canada_analysis\.paths import .*resolve_raw_data_dir", code)
    assert "resolve_raw_data_dir(" in code
    assert "load_raw_datasets(" in code


# ---------------------------------------------------- blank-row quality step


def test_ingestion_notebook_imports_reusable_quality_api() -> None:
    code = _code_source(read_notebook(INGESTION_NOTEBOOK))
    assert re.search(
        r"from ql2_sixt_canada_analysis(\.quality)? import\s*\(?[^)]*?\bremove_blank_rows_from_raw_datasets\b",
        code, re.S,
    )
    assert "remove_blank_rows_from_raw_datasets(" in code


def test_ingestion_notebook_applies_quality_step_after_loading() -> None:
    cells = _code_cells(read_notebook(INGESTION_NOTEBOOK))
    load_index = next(i for i, c in enumerate(cells) if "load_raw_datasets(" in c.source)
    quality_index = next(i for i, c in enumerate(cells) if "remove_blank_rows_from_raw_datasets(" in c.source)
    assert quality_index > load_index
    later = "\n".join(c.source for c in cells[quality_index:])
    # Per-dataset and total counts are kept in named in-memory objects and the
    # cleaned frames feed subsequent variables.
    assert re.search(r"\bblank_rows\s*=\s*remove_blank_rows_from_raw_datasets\(", later)
    assert "total_removed_blank_row_count" in later
    assert re.search(r"\bjobs_df\s*=.*\.cleaned", later) and re.search(r"\bcars_df\s*=.*\.cleaned", later)


def test_ingestion_notebook_does_not_reimplement_blank_row_detection() -> None:
    code = _code_source(read_notebook(INGESTION_NOTEBOOK))
    for pattern in (r"\.dropna\(", r"\.isna\(\)\.all\(", r"\.isnull\(", r"\.str\.strip\(", r"\.strip\(\)\s*==", r"\.replace\("):
        assert not re.search(pattern, code), f"notebook reimplements blank-row logic: {pattern}"


def test_ingestion_notebook_never_displays_counts_or_frames() -> None:
    code = _code_source(read_notebook(INGESTION_NOTEBOOK))
    assert not re.search(r"print\([^\n]*(row_count|len\(|\.shape|_df\b|\.cleaned\b)", code)
    assert not re.search(r"(?m)^\s*(raw|cleaned|jobs_df|cars_df|blank_rows|\w+\.cleaned)\s*$", code), "bare expression would display"
    assert not re.search(r"\.(head|tail|sample|describe|info|to_string|to_markdown)\(", code)


def test_ingestion_notebook_quality_step_runs_with_synthetic_blank_rows(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic_raw"
    directory.mkdir()
    for key in DatasetKey:
        columns = contract_columns(key)
        record = ",".join(f"synthetic_{c}" for c in range(len(columns)))
        (directory / f"synthetic_{key}.csv").write_bytes(
            (",".join(columns) + f"\n{record}\n\n{record}\n").encode("utf-8")
        )
    result = execute_notebook_copy(
        INGESTION_NOTEBOOK, workdir=tmp_path, env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)}
    )
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "blank" in outputs.lower()
    assert not re.search(r"\b[0-9]+\b", outputs), "cell output shows a number"
    assert "synthetic_0" not in outputs
    assert not any(o.get("output_type") in {"execute_result", "display_data"}
                   for c in _code_cells(result.executed) for o in c.outputs)
    assert not list(directory.parent.glob("**/*blank*")), "notebook wrote a blank-row artifact"


# ---------------------------------------------------------- identifier typing


def test_ingestion_notebook_has_no_identifier_lists_or_manual_casts() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    code = _code_source(notebook)
    text = code + "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    identifiers = {c for d in DATASET_DEFINITIONS.values() for c in d.identifier_columns}
    assert not any(re.search(rf"\b{re.escape(c)}\b", text) for c in identifiers), \
        "identifier names belong in the schema module only"
    assert ".astype(" not in code, "no manual casts; the loader types identifiers"
    assert "dtype=" not in code and "identifier_columns" not in code


def test_ingestion_notebook_validates_identifier_dtypes_after_cleaning() -> None:
    cells = _code_cells(read_notebook(INGESTION_NOTEBOOK))
    sources = [c.source for c in cells]
    load = next(i for i, s in enumerate(sources) if "load_raw_datasets(" in s)
    clean = next(i for i, s in enumerate(sources) if "remove_blank_rows_from_raw_datasets(" in s)
    validate = next(i for i, s in enumerate(sources) if "validate_raw_dataset_identifier_dtypes(" in s)
    assert load < clean < validate
    assert re.search(r"validate_raw_dataset_identifier_dtypes\(\s*cleaned\s*\)", sources[validate])
    code = "\n".join(sources)
    assert re.search(
        r"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\bvalidate_raw_dataset_identifier_dtypes\b",
        code, re.S,
    )
    assert not re.search(r"\.(dtypes|info|value_counts|nunique|isna)\b", code), "no type/null/distinct summaries"


def test_ingestion_notebook_runs_with_risky_synthetic_identifiers(tmp_path: Path) -> None:
    risky = ["000123", "123456789012345678901234567890", "SYNTHETIC-ID-001", "A-001-B", "0", ""]
    directory = tmp_path / "synthetic_raw"
    directory.mkdir()
    for key in DatasetKey:
        definition = DATASET_DEFINITIONS[key]
        lines = [",".join(v if c in definition.identifier_columns else f"synthetic_{i}"
                          for i, c in enumerate(definition.columns)) for v in risky]
        (directory / f"synthetic_{key}.csv").write_bytes(
            (",".join(definition.columns) + "\n" + "\n".join(lines) + "\n\n").encode("utf-8")
        )
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    result = execute_notebook_copy(
        INGESTION_NOTEBOOK, workdir=tmp_path, env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)}
    )
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "nullable string" in outputs
    assert not any(v in outputs for v in risky if v and v != "0")
    assert not re.search(r"\b[0-9]+\b", outputs)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before, \
        "notebook wrote files"


# ------------------------------------------------------------------ unique keys


def test_ingestion_notebook_assesses_unique_keys_with_reusable_api() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    cells = _code_cells(notebook)
    sources = [c.source for c in cells]
    code = "\n".join(sources)
    assert re.search(
        r"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\bassess_raw_dataset_unique_keys\b", code, re.S
    )
    clean = next(i for i, s in enumerate(sources) if "remove_blank_rows_from_raw_datasets(" in s)
    dtypes = next(i for i, s in enumerate(sources) if "validate_raw_dataset_identifier_dtypes(" in s)
    assess = next(i for i, s in enumerate(sources) if "assess_raw_dataset_unique_keys(" in s)
    assert clean < dtypes < assess
    assert re.search(r"\bkey_reports\s*=\s*assess_raw_dataset_unique_keys\(\s*cleaned\s*\)", sources[assess])
    # Assessment, not strict validation, so real violations cannot stop the workflow.
    assert "validate_raw_dataset_unique_keys" not in code and "validate_unique_key" not in code


def test_ingestion_notebook_has_no_key_lists_or_own_key_algorithm() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    code = _code_source(notebook)
    text = code + "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    keys = {c for d in DATASET_DEFINITIONS.values() for c in d.unique_key_columns}
    assert not any(re.search(rf"\b{re.escape(c)}\b", text) for c in keys), "key names live in schemas only"
    for pattern in (r"\.duplicated\(", r"\.groupby\(", r"\.drop_duplicates\(", r"\.nunique\(",
                    r"\.value_counts\(", r"unique_key_columns", r"\.notna\(", r"\.isna\("):
        assert not re.search(pattern, code), f"notebook reimplements key logic: {pattern}"
    for attribute in ("row_count", "is_valid", "all_valid", "violations"):
        assert not re.search(rf"print\([^\n]*{attribute}", code), "key results must not be displayed"


def test_ingestion_notebook_key_step_runs_on_synthetic_violations(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic_raw"
    directory.mkdir()
    for key in DatasetKey:
        definition = DATASET_DEFINITIONS[key]
        def line(job: str) -> str:
            return ",".join(
                job if c in definition.identifier_columns
                else ("1" if c in definition.unique_key_columns else f"synthetic_{i}")
                for i, c in enumerate(definition.columns)
            )
        # duplicate key, missing key and a blank line: the notebook must not stop.
        lines = [line("SYNTH-JOB-001"), line("SYNTH-JOB-001"), line(""), "", line("000001")]
        (directory / f"synthetic_{key}.csv").write_bytes(
            (",".join(definition.columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8")
        )
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(
        INGESTION_NOTEBOOK, workdir=tmp_path, env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)}
    )
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "Unique-key assessment completed" in outputs
    assert "SYNTH-JOB-001" not in outputs and "000001" not in outputs
    assert not re.search(r"\b[0-9]+\b", outputs)
    assert not re.search(r"\b(True|False|valid|invalid|duplicate)\b", outputs, re.I)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before
    assert _snapshot(PROJECT_ROOT) == repo_before


# ---------------------------------------------------- job-detail reconciliation


def test_ingestion_notebook_reconciles_with_reusable_api_after_key_assessment() -> None:
    sources = [c.source for c in _code_cells(read_notebook(INGESTION_NOTEBOOK))]
    code = "\n".join(sources)
    for name in ("assess_job_detail_reconciliation", "JOB_DETAIL_RELATIONSHIP"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    keys = next(i for i, s in enumerate(sources) if "assess_raw_dataset_unique_keys(" in s)
    reconcile = next(i for i, s in enumerate(sources) if "assess_job_detail_reconciliation(" in s)
    assert keys < reconcile
    assert re.search(
        r"reconciliation_report\s*=\s*assess_job_detail_reconciliation\(\s*jobs_df\s*,\s*cars_df\s*,"
        r"\s*JOB_DETAIL_RELATIONSHIP\s*\)", sources[reconcile])
    assert "validate_job_detail_reconciliation" not in code  # assessment keeps the workflow running


def test_ingestion_notebook_has_no_relationship_fields_or_own_reconciliation() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    code = _code_source(notebook)
    text = code + "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    rel = JOB_DETAIL_RELATIONSHIP
    fields = {*rel.parent_key_columns, *rel.detail_key_columns, rel.expected_detail_count_column}
    assert not any(re.search(rf"\b{re.escape(f)}\b", text) for f in fields), "fields live in schemas only"
    for pattern in (r"\.merge\(", r"(_df|\.jobs|\.cars)\.join\(", r"\.groupby\(", r"\.value_counts\(", r"\.size\(\)",
                    r"\.reindex\(", r"\.isin\(", r"\.sum\(", r"parent_key_columns", r"detail_key_columns",
                    r"expected_detail_count_column"):
        assert not re.search(pattern, code), f"notebook reimplements reconciliation: {pattern}"
    assert not re.search(r"print\([^\n]*(reconciliation_report|_count|discrepancy|reconciled)", code)


def test_ingestion_notebook_reconciliation_runs_on_synthetic_mismatches(tmp_path: Path) -> None:
    rel = JOB_DETAIL_RELATIONSHIP
    directory = tmp_path / "synthetic_raw"
    directory.mkdir()
    jobs_rows = [{rel.parent_key_columns[0]: "SYNTH-JOB-001", rel.expected_detail_count_column: "2"},
                 {rel.parent_key_columns[0]: "000001", rel.expected_detail_count_column: "1"}]
    cars_rows = [{rel.detail_key_columns[0]: "SYNTH-JOB-001"}, {rel.detail_key_columns[0]: "SYNTH-JOB-404"},
                 {rel.detail_key_columns[0]: ""}]
    for key, rows in ((DatasetKey.JOBS, jobs_rows), (DatasetKey.CARS, cars_rows)):
        columns = DATASET_DEFINITIONS[key].columns
        lines = [",".join(r.get(c, f"synthetic_{i}") for i, c in enumerate(columns)) for r in rows]
        (directory / f"synthetic_{key}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n\n").encode("utf-8"))
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(
        INGESTION_NOTEBOOK, workdir=tmp_path, env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)}
    )
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "reconciliation step completed" in outputs
    assert "SYNTH-JOB" not in outputs and "000001" not in outputs
    assert not re.search(r"\b[0-9]+\b", outputs)
    assert not re.search(r"\b(True|False|orphan|mismatch|under|over)\b", outputs, re.I)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before
    assert _snapshot(PROJECT_ROOT) == repo_before


# ------------------------------------------------- one-to-many relationship


def test_ingestion_notebook_validates_relationship_before_trusted_join() -> None:
    sources = [c.source for c in _code_cells(read_notebook(INGESTION_NOTEBOOK))]
    code = "\n".join(sources)
    for name in ("assess_one_to_many_join", "join_jobs_to_details", "JOB_DETAIL_RELATIONSHIP"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    reconcile = next(i for i, s in enumerate(sources) if "assess_job_detail_reconciliation(" in s)
    relate = next(i for i, s in enumerate(sources) if "assess_one_to_many_join(" in s)
    assert reconcile < relate
    cell = sources[relate]
    assert re.search(r"relationship_report\s*=\s*assess_one_to_many_join\(\s*jobs_df\s*,\s*cars_df\s*,"
                     r"\s*JOB_DETAIL_RELATIONSHIP\s*\)", cell)
    # The join only runs when the contract passes; otherwise no joined frame.
    assert re.search(r"join_jobs_to_details\([^)]*\)\.joined\s*if\s+one_to_many_contract_valid\s+else\s+None", cell, re.S)
    assert not re.search(r"(pd\.merge|\.merge\(|\.join\(\s*(jobs|cars))", code), "no unvalidated direct merge"
    keys = {*JOB_DETAIL_RELATIONSHIP.parent_key_columns, *JOB_DETAIL_RELATIONSHIP.detail_key_columns}
    assert not any(re.search(rf"\b{re.escape(k)}\b", code) for k in keys)
    assert not re.search(r"print\([^\n]*(relationship_report|jobs_with_details|_count\b|_valid\b|is_valid)", code)


def test_ingestion_notebook_relationship_step_runs_on_synthetic_inputs(tmp_path: Path) -> None:
    rel = JOB_DETAIL_RELATIONSHIP
    for case, cars_keys in (("valid", ["SYNTH-JOB-001", "SYNTH-JOB-001"]),
                            ("orphan", ["SYNTH-JOB-001", "SYNTH-JOB-404", ""])):
        directory = tmp_path / case
        directory.mkdir()
        rows = {DatasetKey.JOBS: [{rel.parent_key_columns[0]: "SYNTH-JOB-001", rel.expected_detail_count_column: "2"},
                                  {rel.parent_key_columns[0]: "000001", rel.expected_detail_count_column: "0"}],
                DatasetKey.CARS: [{rel.detail_key_columns[0]: k} for k in cars_keys]}
        for key, key_rows in rows.items():
            columns = DATASET_DEFINITIONS[key].columns
            lines = [",".join(r.get(c, f"synthetic_{i}_{n}") for i, c in enumerate(columns))
                     for n, r in enumerate(key_rows)]
            (directory / f"synthetic_{key}.csv").write_bytes(
                (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
        workdir = tmp_path / f"kernel_{case}"
        workdir.mkdir()
        repo_before = _snapshot(PROJECT_ROOT)
        result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                       env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)})
        outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
        assert "One-to-many relationship validation step completed." in outputs
        assert "SYNTH" not in outputs and not re.search(r"\b[0-9]+\b", outputs)
        assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


# ---------------------------------------------------- expected location coverage


def test_ingestion_notebook_checks_coverage_on_cleaned_frames_before_join() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    for name in ("assess_dataset_location_coverage", "EXPECTED_LOCATION_COVERAGE"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    keys = next(i for i, s in enumerate(sources) if "assess_raw_dataset_unique_keys(" in s)
    cover = next(i for i, s in enumerate(sources) if "assess_dataset_location_coverage(" in s)
    reconcile = next(i for i, s in enumerate(sources) if "assess_job_detail_reconciliation(" in s)
    join = next(i for i, s in enumerate(sources) if "join_jobs_to_details(" in s)
    assert keys < cover < reconcile < join
    assert re.search(r"location_coverage_report\s*=\s*assess_dataset_location_coverage\(\s*cleaned\s*,"
                     r"\s*EXPECTED_LOCATION_COVERAGE\s*\)", sources[cover])
    assert "validate_expected_location_coverage" not in code
    # No location column literals or own distinct/set logic. (Column names
    # may coincide with ordinary words, so only quoted literals are checked.)
    assert not any(re.search(rf"[\"']{re.escape(c)}[\"']", code)
                   for c in (*EXPECTED_LOCATION_COVERAGE.location_columns,
                             *EXPECTED_LOCATION_COVERAGE.stream_scope_columns))
    for pattern in (r"\.unique\(", r"\.nunique\(", r"\.drop_duplicates\(", r"\bset\(", r"\.difference\(",
                    r"\.isin\(", r"expected_locations", r"location_columns", r"\.str\.strip"):
        assert not re.search(pattern, code), f"notebook reimplements coverage: {pattern}"
    assert not re.search(r"print\([^\n]*(location_coverage_report|_ratio|_count\b|_passed)", code)


def test_ingestion_notebook_coverage_step_fails_closed_on_synthetic_inputs(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "Expected-coverage step completed." in outputs
    assert "synthetic_r" not in outputs and not re.search(r"\b[0-9]+\b", outputs)
    assert not re.search(r"\b(True|False|None|missing|unexpected|configured)\b", outputs, re.I)
    assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


# ---------------------------------------------- expected stream investigation


def test_ingestion_notebook_investigates_stream_via_central_target() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    for name in ("investigate_location_stream", "resolve_expected_location", "INVESTIGATED_LOCATION_STREAM"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    relate = next(i for i, s in enumerate(sources) if "assess_one_to_many_join(" in s)
    investigate = next(i for i, s in enumerate(sources) if "investigate_location_stream(" in s)
    assert relate < investigate
    cell = sources[investigate]
    assert re.search(r"stream_target\s*=\s*resolve_expected_location\(\s*INVESTIGATED_LOCATION_STREAM\b", cell)
    assert re.search(r"location_stream_report\s*=\s*investigate_location_stream\(", cell)
    # The target literal lives only in the central contract.
    text = code + "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    assert not any(part in text for part in INVESTIGATED_LOCATION_STREAM)
    for pattern in (r"\.query\(", r"\.loc\[", r"\.merge\(", r"\.groupby\(", r"==\s*stream_target",
                    r"\.isin\(", r"\.str\.", r"validate_location_stream"):
        assert not re.search(pattern, code), f"one-off stream logic in notebook: {pattern}"
    assert not re.search(r"print\([^\n]*(location_stream_report|stream_target|_healthy|status)", code)


def test_ingestion_notebook_stream_step_runs_on_synthetic_inputs(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "Expected-stream investigation step completed." in outputs
    assert not any(part in outputs for part in INVESTIGATED_LOCATION_STREAM)
    assert not re.search(r"\b(absent|partial|healthy|alias|raw_stream|True|False)\b", outputs, re.I)
    assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


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
