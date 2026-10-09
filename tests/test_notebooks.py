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
    TEMPORAL_RECONCILIATION,
    VANCOUVER_LOCATION_POLICY,
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


def _step_output(result: object, marker: str) -> str:
    """Printed output of the one executed code cell whose source contains ``marker``."""
    cells = [c for c in _code_cells(result.executed) if marker in c.source]  # type: ignore[attr-defined]
    assert len(cells) == 1, marker
    return "".join(o.get("text", "") for o in cells[0].outputs)


def _code_source(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(c.source for c in _code_cells(notebook))


# Cells that are required to report categorical gate results and aggregate
# counts (never values or identifiers); the per-step "stay quiet" checks
# exclude them and each has its own focused tests.
REPORTING_STEPS = ("current_expected_stream_contract(", "current_location_authority(", "current_temporal_authority(", "assess_rental_dates(", "assess_job_linkage(", "assess_per_stream_scheduled_coverage(", "assess_city_integrity(", "assess_expected_location_streams(", "assess_vehicle_attribute_stability(", "assess_job_detail_join_readiness(",
                   "assess_pricing_readiness(", "compare_location_streams(", "load_raw_datasets(raw_dir)",
                   "assess_dataset_location_coverage(", "assess_job_detail_reconciliation(",
                   "investigate_location_stream(", "assess_completeness(", "build_pricing_population(",
                   "assess_canonical_offers(")


def _is_reporting(cell: nbformat.NotebookNode) -> bool:
    return any(marker in cell.source for marker in REPORTING_STEPS)


def _quiet_code(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(c.source for c in _code_cells(notebook) if not _is_reporting(c))


def _quiet_outputs(result: object) -> str:
    return "\n".join(o.get("text", "") for c in _code_cells(result.executed)  # type: ignore[attr-defined]
                      if not _is_reporting(c) for o in c.outputs)


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
    outputs = _quiet_outputs(result)
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
    outputs = _quiet_outputs(result)
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
    assert re.search(r"\bkey_reports\s*=\s*assess_raw_dataset_unique_keys\(\s*analysis_datasets\s*,"
                     r"\s*ANALYSIS_DATASET_DEFINITIONS\s*\)", sources[assess])
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
        assert not re.search(rf"print\([^\n]*{attribute}", _quiet_code(notebook)), "key results must not be displayed"


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
    outputs = _step_output(result, "assess_raw_dataset_unique_keys(")  # this step's own cell only
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
    for name in ("assess_job_detail_reconciliation", "ANALYSIS_JOB_DETAIL_RELATIONSHIP"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    keys = next(i for i, s in enumerate(sources) if "assess_raw_dataset_unique_keys(" in s)
    reconcile = next(i for i, s in enumerate(sources) if "assess_job_detail_reconciliation(" in s)
    assert keys < reconcile
    assert re.search(
        r"reconciliation_report\s*=\s*assess_job_detail_reconciliation\(\s*analysis_jobs_df\s*,\s*analysis_cars_df\s*,"
        r"\s*ANALYSIS_JOB_DETAIL_RELATIONSHIP\s*\)", sources[reconcile])
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
    # Reporting cells print aggregate results only; the quiet steps print none.
    assert not re.search(r"print\([^\n]*(reconciliation_report|_count|discrepancy|reconciled)",
                         _quiet_code(notebook))


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
    outputs = _step_output(result, "assess_job_detail_reconciliation(")  # this step's own cell only
    assert "reconciliation step completed" in outputs
    assert "SYNTH-JOB" not in outputs and "000001" not in outputs        # no identifiers, aggregates only
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    for column in rel.expected_detail_count_columns:
        assert lines[f"Declared {column.replace('_', ' ')} reconciled"].split("|")[0].strip() == "False"
    assert lines["Combined reconciliation passed"].split("|")[0].strip() == "False"
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before
    assert _snapshot(PROJECT_ROOT) == repo_before


# ------------------------------------------------- one-to-many relationship


def test_ingestion_notebook_validates_relationship_before_trusted_join() -> None:
    sources = [c.source for c in _code_cells(read_notebook(INGESTION_NOTEBOOK))]
    code = "\n".join(sources)
    for name in ("assess_one_to_many_join", "assess_job_detail_join_readiness", "ANALYSIS_JOB_DETAIL_RELATIONSHIP"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    reconcile = next(i for i, s in enumerate(sources) if "assess_job_detail_reconciliation(" in s)
    relate = next(i for i, s in enumerate(sources) if "assess_one_to_many_join(" in s)
    assert reconcile < relate
    cell = sources[relate]
    assert re.search(r"relationship_report\s*=\s*assess_one_to_many_join\(\s*analysis_jobs_df\s*,\s*analysis_cars_df\s*,"
                     r"\s*ANALYSIS_JOB_DETAIL_RELATIONSHIP\s*\)", cell)
    # The relationship alone never yields a joined frame: the notebook does not
    # call the relationship-checked join directly (the join gate does).
    assert "join_jobs_to_details" not in code and not re.search(r"^jobs_with_details\s*=", code, re.M)
    assert not re.search(r"(pd\.merge|\.merge\(|\.join\(\s*(jobs|cars))", code), "no unvalidated direct merge"
    keys = {*JOB_DETAIL_RELATIONSHIP.parent_key_columns, *JOB_DETAIL_RELATIONSHIP.detail_key_columns}
    assert not any(re.search(rf"\b{re.escape(k)}\b", code) for k in keys)
    assert not re.search(r"print\([^\n]*(relationship_report|jobs_with_details|_count\b|_valid\b|is_valid)", cell)


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
        outputs = _quiet_outputs(result)
        assert "One-to-many relationship validation step completed." in outputs
        assert "SYNTH" not in outputs and not re.search(r"\b[0-9]+\b", outputs)
        assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


# ---------------------------------------------------- trusted job-detail join


def test_ingestion_notebook_gates_the_trusted_join_through_the_api() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    relate = next(i for i, s in enumerate(sources) if "assess_one_to_many_join(" in s)
    gate = next(i for i, s in enumerate(sources) if "assess_job_detail_join_readiness(" in s)
    assert relate < gate
    cell = sources[gate]
    assert re.search(r"job_detail_join\s*=\s*assess_job_detail_join_readiness\(\s*analysis_jobs_df\s*,"
                     r"\s*analysis_cars_df\s*,\s*ANALYSIS_JOB_DETAIL_RELATIONSHIP\s*,"
                     r"\s*job_linkage=job_linkage_report\s*\)", cell)
    assert re.search(r"^trusted_jobs_with_details\s*=\s*job_detail_join\.trusted_jobs_with_details", cell, re.M)
    assert re.search(r"^job_detail_join_ready\s*=\s*job_detail_join\.join_ready", cell, re.M)
    # The trust decision lives in production code, not in notebook boolean logic.
    assert not re.search(r"trusted_jobs_with_details\s*=.*\bif\b", cell)
    for later in sources[gate + 1:]:
        assert "diagnostic_jobs_with_details" not in later, "diagnostic join consumed downstream"
    guidance = next(c.source for c in notebook.cells if c.source.startswith("## Next step"))
    assert "trusted_jobs_with_details" in guidance and "job_detail_join_ready" in guidance
    assert "only when it is not `None`" not in guidance and "not proof" in guidance


def test_ingestion_notebook_reports_blocked_join_on_duplicate_detail_keys(tmp_path: Path) -> None:
    from ql2_sixt_canada_analysis.join_readiness import JobDetailJoinBlocker

    rel = JOB_DETAIL_RELATIONSHIP
    position, = rel.detail_definition.non_identifier_key_columns
    directory = tmp_path / "raw"
    directory.mkdir()
    (parent_city, detail_city), = rel.scope_agreement_columns              # linked rows share their job's city
    rows = {DatasetKey.JOBS: [{rel.parent_key_columns[0]: "SYNTH-JOB-001", parent_city: "SYNTH-CITY-1",
                               **{c: "2" for c in rel.expected_detail_count_columns}}],
            DatasetKey.CARS: [{rel.detail_key_columns[0]: "SYNTH-JOB-001", position: "0",
                               detail_city: "SYNTH-CITY-1"}] * 2}  # duplicate key
    for key, key_rows in rows.items():
        columns = DATASET_DEFINITIONS[key].columns
        lines = [",".join(r.get(c, f"synthetic_{i}_{n}") for i, c in enumerate(columns))
                 for n, r in enumerate(key_rows)]
        (directory / f"synthetic_{key}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)})
    outputs = _step_output(result, "assess_job_detail_join_readiness(")
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Relationship contract passed"].strip() == "True"      # the original defect's trigger
    assert lines["Detail business-key contract passed"].strip() == "False"
    assert lines["Jobs business-key contract passed"].strip() == "True"
    assert lines["Declared counts reconciled"].strip() == "True"
    assert lines["Trusted join ready"].strip() == "False"
    assert JobDetailJoinBlocker.DETAILS_KEY_CONTRACT_FAILED.value in lines["Trusted join blocked by"]
    assert lines["Joined frame held"].strip().startswith("diagnostic only - UNTRUSTED")
    assert "SYNTH" not in outputs and "synthetic_" not in outputs and not re.search(r"\d", outputs)
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
    join = next(i for i, s in enumerate(sources) if "assess_job_detail_join_readiness(" in s)
    assert keys < cover < reconcile < join
    assert re.search(r"location_coverage_report\s*=\s*assess_dataset_location_coverage\(\s*analysis_datasets\s*,"
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
    assert not re.search(r"print\([^\n]*(location_coverage_report|_ratio|_count\b|_passed)", _quiet_code(notebook))


def test_ingestion_notebook_coverage_step_fails_closed_on_synthetic_inputs(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    outputs = _step_output(result, "assess_dataset_location_coverage(")  # this step's own cell only
    assert "Expected-coverage step completed." in outputs
    assert "synthetic_r" not in outputs                                   # observed values never shown
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Observed pair values"].strip() == "withheld (confidential)"
    assert lines["Coverage contract passed"].strip() == "False"
    assert lines["Missing expected pairs"].strip() != "none"           # configured pairs, not source values
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
    assert INVESTIGATED_LOCATION_STREAM[-1] not in text   # branch label (the city is a common word)
    for pattern in (r"\.query\(", r"\.loc\[", r"\.merge\(", r"\.groupby\(", r"==\s*stream_target",
                    r"\.isin\(", r"\.str\.", r"validate_location_stream"):
        assert not re.search(pattern, code), f"one-off stream logic in notebook: {pattern}"
    quiet = "\n".join(s for s in sources if not any(m in s for m in REPORTING_STEPS))
    assert not re.search(r"print\([^\n]*(location_stream_report|stream_target|_healthy|status)", quiet)


def test_ingestion_notebook_stream_step_runs_on_synthetic_inputs(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    outputs = _step_output(result, "investigate_location_stream(")  # this step's own cell only
    assert "Expected-stream investigation step completed." in outputs
    assert INVESTIGATED_LOCATION_STREAM[-1] not in outputs
    assert "Stream continuity:" in outputs and "synthetic_r" not in outputs
    assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


# ------------------------------------------------------ temporal reconciliation


def test_ingestion_notebook_reconciles_temporal_fields_through_the_api() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    for name in ("assess_temporal_reconciliation", "current_temporal_reconciliation", "current_temporal_authority"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    relate = next(i for i, s in enumerate(sources) if "assess_one_to_many_join(" in s)
    temporal = next(i for i, s in enumerate(sources) if "assess_temporal_reconciliation(" in s)
    assert relate < temporal
    assert re.search(r"temporal_report\s*=\s*assess_temporal_reconciliation\(\s*analysis_jobs_df\s*,\s*analysis_cars_df\s*,"
                     r"\s*current_temporal_reconciliation\(\)\s*\)", sources[temporal])
    assert "ANALYSIS_TEMPORAL_RECONCILIATION" not in code                  # the authority-backed contract only
    summary = next(s for s in sources if "temporal_authority = current_temporal_authority()" in s)
    assert sources.index(summary) == temporal + 1
    assert re.search(r"temporal_fields_trusted\s*=", sources[temporal])
    # No duplicated field lists, parsing rules or repairs in the notebook.
    fields = {f.column for f in TEMPORAL_RECONCILIATION.fields}
    assert not any(re.search(rf"[\"']{re.escape(f)}[\"']", code) for f in fields)
    for pattern in (r"to_datetime", r"tz_localize", r"tz_convert", r"strptime", r"\.dt\.", r"fillna", r"ZoneInfo",
                    r"America/", r"astimezone", r"utcoffset", r"Timedelta", r"timedelta", r"<=", r">=",
                    r"validate_temporal_reconciliation", r"parse_temporal_field"):
        assert not re.search(pattern, code), f"notebook duplicates temporal logic: {pattern}"
    assert not re.search(r"print\([^\n]*(temporal_report|_trusted|_count)", sources[temporal])


def test_ingestion_notebook_temporal_step_runs_on_synthetic_inputs(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    outputs = _step_output(result, "assess_temporal_reconciliation(")  # this step's own cell only
    assert "Temporal reconciliation step completed." in outputs
    assert not re.search(r"\b(\d{4}-\d{2}-\d{2}|unavailable|invalid|True|False)\b", outputs)
    assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


# ---------------------------------------------------- related stream comparison


def test_ingestion_notebook_compares_related_streams_through_the_api() -> None:
    from ql2_sixt_canada_analysis.schemas import COMPARED_LOCATION_STREAMS, LOCATION_STREAM_COMPARISON

    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    for name in ("compare_location_streams", "ANALYSIS_LOCATION_STREAM_COMPARISON"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    temporal = next(i for i, s in enumerate(sources) if "assess_temporal_reconciliation(" in s)
    compare = next(i for i, s in enumerate(sources) if "compare_location_streams(" in s)
    assert temporal < compare
    assert re.search(r"location_comparison_report\s*=\s*compare_location_streams\(\s*analysis_jobs_df\s*,"
                     r"\s*analysis_cars_df\s*,\s*ANALYSIS_LOCATION_STREAM_COMPARISON\s*\)", sources[compare])
    # The ambiguous boolean handoff is gone: identity comes from the policy gate.
    assert "location_alias_confirmed" not in code and "alias_authority_sufficient" not in code
    # Pair, columns and aliasing live in the central definition only.
    names = [key[-1] for key in COMPARED_LOCATION_STREAMS]   # branch labels
    full = "\n".join(c.source for c in notebook.cells)
    assert not any(n in full for n in names)
    columns = {*LOCATION_STREAM_COMPARISON.product_columns, *LOCATION_STREAM_COMPARISON.price_columns}
    assert not any(re.search(rf"[\"']{re.escape(c)}[\"']", code) for c in columns)
    for pattern in (r"canonical_location_keys", r"validate_confirmed_location_alias", r"drop_duplicates",
                    r"\.replace\(", r"\.merge\(", r"to_csv", r"to_parquet", r"Counter"):
        assert not re.search(pattern, code), f"notebook duplicates or merges streams: {pattern}"
    quiet = "\n".join(s for s in sources if not any(m in s for m in REPORTING_STEPS))
    assert not re.search(r"print\([^\n]*(location_comparison_report|_confirmed|status)", quiet)


def test_ingestion_notebook_comparison_step_runs_on_synthetic_inputs(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    from ql2_sixt_canada_analysis.schemas import COMPARED_LOCATION_STREAMS

    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    outputs = _step_output(result, "compare_location_streams(")  # this step's own cell only
    assert "Related-stream comparison step completed." in outputs
    assert not any(key[-1] in outputs for key in COMPARED_LOCATION_STREAMS)
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Behavioural comparison result"].strip() == "both_streams_absent"
    assert lines["Duplicate inference blocked by"].strip().startswith("no_paired_captures")
    assert "not authoritative alias confirmation" in outputs
    assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


def test_ingestion_notebook_reports_inconclusive_single_pair_partial_overlap(tmp_path: Path) -> None:
    # Regression: one identical shared capture plus an unpaired capture and no baseline
    # was reported as likely_duplicate_streams.
    from ql2_sixt_canada_analysis.comparison import DuplicateInferenceBlocker as DB
    from ql2_sixt_canada_analysis.schemas import COMPARED_LOCATION_STREAMS, LOCATION_STREAM_COMPARISON as D

    rel = JOB_DETAIL_RELATIONSHIP
    loc = D.coverage.label_column
    scope, = D.coverage.stream_scope_columns
    position, = rel.detail_definition.non_identifier_key_columns
    (city, first), (_, second) = COMPARED_LOCATION_STREAMS
    same = ({c: f"SYNTH-{c.upper()}" for c in (*D.product_columns, *D.price_columns)}
            | {c: "10.00" for c in D.numeric_columns} | {scope: city})   # valid offers
    cars = [same | {rel.detail_key_columns[0]: "SYNTH-JOB-001", loc: first, position: "0"},
            same | {rel.detail_key_columns[0]: "SYNTH-JOB-001", loc: second, position: "1"},
            same | {rel.detail_key_columns[0]: "SYNTH-JOB-002", loc: first, position: "0"}]   # unpaired
    jobs = [{rel.parent_key_columns[0]: "SYNTH-JOB-001", rel.expected_detail_count_column: "2"},
            {rel.parent_key_columns[0]: "SYNTH-JOB-002", rel.expected_detail_count_column: "1"}]
    directory = tmp_path / "raw"
    directory.mkdir()
    for dataset, dataset_rows in ((DatasetKey.JOBS, jobs), (DatasetKey.CARS, cars)):
        columns = DATASET_DEFINITIONS[dataset].columns
        lines = [",".join(r.get(c, f"synthetic_{i}") for i, c in enumerate(columns)) for r in dataset_rows]
        (directory / f"synthetic_{dataset}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)})
    outputs = _step_output(result, "compare_location_streams(")
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Behavioural comparison result"].strip() == "comparison_inconclusive"
    assert lines["Paired captures"].strip() == "1 | minimum required: 2"
    assert lines["Unpaired captures - first stream"].strip() == "1 | second stream: 0"
    assert lines["Matching paired captures"].strip() == "1 | differing: 0"
    assert lines["Eligible paired captures (valid offers on both sides)"].strip() == "1 | invalid: 0"
    assert lines["Target evidence threshold passed"].strip() == "False"
    assert lines["Duplicate-stream inference permitted"].strip() == "False"
    assert lines["Temporal overlap"].strip() == "partial" and lines["Scope baseline"].strip() == "unavailable"
    assert lines["Duplicate inference blocked by"].strip() == ", ".join(
        b.value for b in (DB.INSUFFICIENT_PAIRED_CAPTURES, DB.INCOMPLETE_TEMPORAL_OVERLAP, DB.BASELINE_UNAVAILABLE))
    assert first not in outputs and second not in outputs and "SYNTH" not in outputs
    policy = _step_output(result, "assess_pricing_readiness(")
    # The approved decision comes from the record; inconclusive behaviour never changes it.
    assert "Vancouver identity policy state: confirmed_alias" in policy and "Pricing analysis ready: False" in policy


def test_ingestion_notebook_reports_missing_prices_as_invalid_not_duplicate(tmp_path: Path) -> None:
    # Regression: two fully paired captures with missing prices compared equal and,
    # with one weak comparator capture, were reported as likely_duplicate_streams.
    from ql2_sixt_canada_analysis.comparison import DuplicateInferenceBlocker as DB
    from ql2_sixt_canada_analysis.schemas import COMPARED_LOCATION_STREAMS, LOCATION_STREAM_COMPARISON as D

    rel = JOB_DETAIL_RELATIONSHIP
    loc = D.coverage.label_column
    scope, = D.coverage.stream_scope_columns
    position, = rel.detail_definition.non_identifier_key_columns
    (city, first), (_, second) = COMPARED_LOCATION_STREAMS
    no_price = ({c: f"SYNTH-{c.upper()}" for c in D.product_columns} | {c: "" for c in D.price_columns}
                | {scope: city})
    cars = [no_price | {rel.detail_key_columns[0]: job, loc: label, position: str(i)}
            for job in ("SYNTH-JOB-001", "SYNTH-JOB-002") for i, label in enumerate((first, second))]
    jobs = [{rel.parent_key_columns[0]: job, rel.expected_detail_count_column: "2"}
            for job in ("SYNTH-JOB-001", "SYNTH-JOB-002")]
    directory = tmp_path / "raw"
    directory.mkdir()
    for dataset, dataset_rows in ((DatasetKey.JOBS, jobs), (DatasetKey.CARS, cars)):
        columns = DATASET_DEFINITIONS[dataset].columns
        lines = [",".join(r.get(c, f"synthetic_{i}") for i, c in enumerate(columns)) for r in dataset_rows]
        (directory / f"synthetic_{dataset}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)})
    outputs = _step_output(result, "compare_location_streams(")
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Behavioural comparison result"].strip() == "comparison_unassessable"
    assert lines["Paired captures"].strip() == "2 | minimum required: 2"
    assert lines["Eligible paired captures (valid offers on both sides)"].strip() == "0 | invalid: 2"
    assert lines["Invalid offers found in"].strip() == "first_target_stream, second_target_stream"
    assert lines["Invalid offer reasons"].strip() == "missing_value"
    assert lines["Offer validity sufficient"].strip() == "False"
    assert lines["Target evidence threshold passed"].strip() == "False"
    assert lines["Baseline evidence threshold passed"].strip() == "False"
    assert lines["Duplicate-stream inference permitted"].strip() == "False"
    assert DB.INVALID_OFFER_EVIDENCE.value in lines["Duplicate inference blocked by"]
    assert first not in outputs and second not in outputs and "SYNTH" not in outputs
    assert "Pricing analysis ready: False" in _step_output(result, "assess_pricing_readiness(")
    assert list(workdir.iterdir()) == []


# ---------------------------------------------------- vehicle-attribute stability


def test_ingestion_notebook_assesses_vehicle_stability_through_the_api() -> None:
    from ql2_sixt_canada_analysis.schemas import VEHICLE_ATTRIBUTE_STABILITY as V

    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    for name in ("assess_vehicle_attribute_stability", "VEHICLE_ATTRIBUTE_STABILITY"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    relate = next(i for i, s in enumerate(sources) if "assess_one_to_many_join(" in s)
    temporal = next(i for i, s in enumerate(sources) if "assess_temporal_reconciliation(" in s)
    stability = next(i for i, s in enumerate(sources) if "assess_vehicle_attribute_stability(" in s)
    assert relate < temporal < stability
    assert re.search(r"vehicle_stability_report\s*=\s*assess_vehicle_attribute_stability\(\s*analysis_cars_df\s*,"
                     r"\s*VEHICLE_ATTRIBUTE_STABILITY\s*\)", sources[stability])
    assert re.search(r"vehicle_attributes_stable\s*=", sources[stability])
    # No duplicated field lists or stability logic in the notebook.
    fields = {*V.entity_key_columns, *V.context_columns, *V.attribute_columns}
    assert not any(re.search(rf"[\"']{re.escape(f)}[\"']", code) for f in fields)
    for pattern in (r"groupby", r"nunique", r"drop_duplicates", r"\.shift\(", r"fillna", r"factorize",
                    r"hash", r"\.unique\(", r"value_counts", r"to_csv", r"to_parquet", r"to_json",
                    r"validate_vehicle_attribute_stability"):
        assert not re.search(pattern, code), f"notebook duplicates stability logic: {pattern}"
    # The full-population result is reported (aggregates only); identifiers are never displayed.
    cell = sources[stability]
    for field in ("status", "is_valid", "distinct_entities", "sufficient_history_entities",
                  "insufficient_history_entities", "violations", "blocking_reasons"):
        assert re.search(rf"print\([^\n]*vehicle_stability_report\.{field}\b", cell), field
    assert "classify_vehicle_entities" not in code and "withheld" in cell
    # The gate is the report's own validity, not an empty violations list.
    assert re.search(r"vehicle_attributes_stable\s*=\s*vehicle_stability_report is not None and "
                     r"vehicle_stability_report\.is_valid", cell)
    assert not re.search(r"vehicle_attributes_stable\s*=.*violations", code)
    guidance = next(c.source for c in notebook.cells if c.source.startswith("## Next step"))
    assert '"No violations" is not "stable"' in guidance and "pricing analysis must not proceed" in guidance


def test_ingestion_notebook_stability_step_runs_on_synthetic_inputs(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    code_cells = _code_cells(result.executed)
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c.outputs)
    outputs = _step_output(result, "assess_vehicle_attribute_stability(")  # this step's own cell only
    assert "Vehicle-attribute stability step completed." in outputs
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Full product population valid"].strip() == "False"
    total, sufficient, insufficient = (int(lines[k]) for k in (
        "In-scope entities", "Sufficient-history entities", "Insufficient-history entities"))
    assert sufficient + insufficient <= total
    assert lines["Insufficient-history entity identifiers"].strip() == "withheld (confidential)"
    assert "SYNTH" not in outputs and "synthetic_" not in outputs
    assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


def test_ingestion_notebook_reports_partially_assessable_product_history(tmp_path: Path) -> None:
    from ql2_sixt_canada_analysis.readiness import PricingBlocker
    from ql2_sixt_canada_analysis.schemas import VEHICLE_ATTRIBUTE_STABILITY as V

    rel = JOB_DETAIL_RELATIONSHIP
    key, = V.entity_key_columns
    scope, = V.context_columns
    capture = V.temporal.field(V.observation_time_field).column
    category, transmission, seats, bags = V.attribute_columns
    position, = rel.detail_definition.non_identifier_key_columns
    base = {scope: "SYNTH-LOCATION-001", category: "SYNTH-CLASS-A", transmission: "SYNTH-AUTOMATIC",
            seats: "5", bags: "2", rel.detail_key_columns[0]: "SYNTH-JOB-001"}
    cars = [base | {key: "SYNTH-VEHICLE-001", capture: "2025-01-15 05:00:00 MST", position: "0"},
            base | {key: "SYNTH-VEHICLE-001", capture: "2025-01-15 06:00:00 MST", position: "1"},
            base | {key: "SYNTH-VEHICLE-002", capture: "2025-01-15 05:00:00 MST", position: "2"}]
    rows = {DatasetKey.JOBS: [{rel.parent_key_columns[0]: "SYNTH-JOB-001", rel.expected_detail_count_column: "3"}],
            DatasetKey.CARS: cars}
    directory = tmp_path / "raw"
    directory.mkdir()
    for dataset, dataset_rows in rows.items():
        columns = DATASET_DEFINITIONS[dataset].columns
        lines = [",".join(r.get(c, f"synthetic_{i}_{n}") for i, c in enumerate(columns))
                 for n, r in enumerate(dataset_rows)]
        (directory / f"synthetic_{dataset}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)})
    stability = dict(line.split(":", 1) for line in
                     _step_output(result, "assess_vehicle_attribute_stability(").splitlines() if ":" in line)
    assert stability["Overall stability result"].strip() == "partially_assessable"   # formerly "passed"
    assert stability["Full product population valid"].strip() == "False"
    assert [int(stability[k]) for k in ("In-scope entities", "Sufficient-history entities",
                                        "Insufficient-history entities")] == [2, 1, 1]
    assert stability["Violations"].split("|")[0].strip() == "none observed"
    assert stability["Stability blocked by"].strip() == "insufficient_history"
    pricing = _step_output(result, "assess_pricing_readiness(")
    assert "Pricing analysis ready: False" in pricing
    assert PricingBlocker.VEHICLE_HISTORY_INSUFFICIENT.value in pricing
    assert PricingBlocker.VEHICLE_ATTRIBUTES_UNSTABLE.value not in pricing
    outputs = _step_output(result, "assess_vehicle_attribute_stability(") + pricing
    assert "SYNTH" not in outputs and "synthetic_" not in outputs


# ------------------------------------------ Vancouver policy and pricing readiness


def test_ingestion_notebook_gates_pricing_on_the_vancouver_policy() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    for name in ("VANCOUVER_LOCATION_POLICY", "assess_location_policy", "assess_pricing_readiness",
                 "apply_location_policy"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    order = [next(i for i, s in enumerate(sources) if call in s) for call in (
        "assess_temporal_reconciliation(", "compare_location_streams(", "assess_vehicle_attribute_stability(",
        "assess_location_policy(", "assess_pricing_readiness(")]
    assert order == sorted(order)
    cell = sources[order[-1]]
    for name in ("vancouver_location_policy_state", "vancouver_location_policy_resolved",
                 "vancouver_location_policy_authority_sufficient", "vancouver_locations_are_aliases",
                 "vancouver_policy_scope", "vancouver_policy_scope_valid", "vancouver_canonicalization_permitted", "vancouver_locations_comparable_independently",
                 "pricing_readiness", "pricing_analysis_ready"):
        assert re.search(rf"^{name}\s*=", cell, re.M), name
    # The single-stream health and separate completeness booleans no longer gate pricing:
    # one stream could pass while another expected stream failed.
    assert "location_stream_healthy" not in cell and "expected_location_coverage_passed" not in cell
    assert re.search(r"completeness=completeness\b", cell)
    for gate in ("all_key_contracts_valid", "one_to_many_contract_valid", "temporal_fields_trusted",
                 "vehicle_stability_report"):
        assert gate in cell, f"pricing readiness ignores {gate}"
    # No policy decision may be derived from comparison evidence in the notebook.
    assert not re.search(r"(LIKELY_DUPLICATE|likely_duplicate|\.status\s*(==|is))", code)
    assert not re.search(r"LocationPolicyState\.|CONFIRMED_(ALIAS|DISTINCT)", code)
    guidance = next(c.source for c in notebook.cells if c.source.startswith("## Next step"))
    assert "pricing_analysis_ready" in guidance and "unresolved" in guidance
    assert "No Vancouver pricing" in guidance and "airport-versus-downtown" in guidance


def test_ingestion_notebook_gates_pricing_on_schedule_coverage_and_trusted_join() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    for name in ("current_per_stream_schedule", "assess_per_stream_scheduled_coverage"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)
    for legacy in ("COLLECTION_SCHEDULE", "assess_collection_schedule", "assess_scheduled_time_coverage",
                   "scraped_at"):                                   # no shared schedule, no observation-time key
        assert legacy not in code, legacy
    streams = next(s for s in sources if "assess_expected_location_streams(" in s)
    coverage = next(s for s in sources if "assess_per_stream_scheduled_coverage(" in s)
    assert "collection_schedule = current_per_stream_schedule()" in coverage
    assert "schedule=collection_schedule" in coverage and "contract=expected_stream_contract" in coverage
    assert "relationship=ANALYSIS_JOB_DETAIL_RELATIONSHIP" in coverage
    assert "for position, entry in enumerate(scheduled_coverage_report.streams" in coverage   # every stream
    # Production APIs only: no timezone, period or matching logic in the notebook.
    assert not re.search(r"ZoneInfo|tz_localize|astimezone|strptime|floor\(|timedelta|\.hour\b", code)
    pricing = next(s for s in sources if "assess_pricing_readiness(" in s)
    assert "scheduled_coverage=scheduled_coverage_report" in pricing
    assert "job_detail_join=job_detail_join" in pricing
    assert "trusted_jobs_with_details" not in pricing and "location_stream_healthy" not in pricing
    # The decision is the central API's: no readiness is computed in the notebook itself.
    assert re.search(r"^pricing_analysis_ready = pricing_readiness\.ready$", pricing, re.M)
    assert len(re.findall(r"pricing_analysis_ready\s*=", code)) == 1
    # The scheduled coverage runs first: the streams consume its resolved governed exclusions.
    assert "capture_exclusions=scheduled_coverage_report.capture_exclusions" in streams
    order = [sources.index(s) for s in (coverage, streams, pricing)]
    assert order == sorted(order)


def test_ingestion_notebook_reports_the_per_stream_schedule_and_blocks_on_coverage(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    # The committed record supplies the per-stream schedule; synthetic rows match no approved city or period,
    # so every stream-period stays missing and pricing is blocked on scheduled coverage (never "unavailable").
    from ql2_sixt_canada_analysis.readiness import PricingBlocker

    workdir = tmp_path / "kernel"
    workdir.mkdir()
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    coverage = _step_output(result, "assess_per_stream_scheduled_coverage(")
    lines = dict(line.split(":", 1) for line in coverage.splitlines() if ":" in line)
    assert lines["Collection schedule"].strip() == "available"
    assert lines["Stream schedules"].split("|")[0].strip() == str(len(EXPECTED_LOCATION_COVERAGE.expected_locations))
    streams = [v for k, v in lines.items() if k.startswith("Expected stream ")]
    assert len(streams) == len(EXPECTED_LOCATION_COVERAGE.expected_locations)
    assert all(v.strip().startswith("0 of 90 periods covered") for v in streams)
    assert lines["All expected streams complete against their own schedules"].strip() == "False"
    assert "scheduled_coverage_incomplete" in lines["Scheduled coverage blocked by"]
    assert "collection_schedule_unavailable" not in lines["Scheduled coverage blocked by"]
    pricing = _step_output(result, "assess_pricing_readiness(")
    plines = dict(line.split(":", 1) for line in pricing.splitlines() if ":" in line)
    assert plines["Authoritative schedule available"].strip() == "True"
    assert plines["Scheduled coverage complete for every expected stream"].strip() == "False"
    assert plines["Pricing analysis ready"].strip() == "False"
    for blocker in (PricingBlocker.SCHEDULED_COVERAGE_INCOMPLETE, PricingBlocker.TRUSTED_JOIN_NOT_READY):
        assert blocker.value in plines["Pricing blocked by"]
    assert PricingBlocker.COLLECTION_SCHEDULE_UNAVAILABLE.value not in plines["Pricing blocked by"]
    assert "SYNTH" not in coverage + pricing and list(workdir.iterdir()) == []


def test_ingestion_notebook_reports_the_approved_alias_and_blocked_pricing(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    from ql2_sixt_canada_analysis.readiness import PricingBlocker

    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    outputs = _step_output(result, "assess_pricing_readiness(")
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    # Record v4: the approved alias, canonical Vancouver / Downtown. The synthetic rows carry neither governed
    # raw Vancouver stream, so the alias is recorded but not usable and pricing stays blocked.
    assert lines["Vancouver identity policy state"].strip() == "confirmed_alias"
    assert lines["Policy authority-backed and resolved"].strip() == "True"
    assert lines["Canonical scope validation passed"].strip() == "True"
    assert lines["Supplied canonical key"].strip() == " / ".join(VANCOUVER_LOCATION_POLICY.canonical_location)
    assert lines["Governed raw Vancouver streams both present"].strip() == "False"
    assert lines["Canonicalization permitted"].strip() == "False"
    assert lines["Identity evidence conflicts with policy"].strip() == "False"
    assert lines["Policy authority sufficient for analysis"].strip() == "False"
    assert lines["Vancouver labels confirmed aliases"].strip() == "False"
    assert lines["Vancouver labels comparable independently"].strip() == "False"
    assert lines["Pricing analysis ready"].strip() == "False"
    blocked = lines["Pricing blocked by"]
    assert PricingBlocker.GOVERNED_SOURCE_STREAM_MISSING.value in blocked
    assert PricingBlocker.LOCATION_POLICY_UNRESOLVED.value not in blocked
    for cleared in ("branch_role_authority_unavailable", "comparison_pair_authority_unavailable"):
        assert cleared not in blocked
    authority = _step_output(result, "current_location_authority(")
    assert "Location-role map: approved | exact for every approved stream: True" in authority
    assert "Comparison pairs: approved | valid: True" in authority
    assert "SYNTH" not in outputs and "synthetic_r" not in outputs and not re.search(r"\d", outputs)
    assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


_POLICY_OVERRIDE = """\
# Test-only kernel configuration (synthetic authority, never a real decision):
# resolve the policy and give the comparison an authoritative identity column.
import dataclasses

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis.schemas import LocationPolicyAuthority, LocationPolicyState

state = LocationPolicyState({state!r})
package.VANCOUVER_LOCATION_POLICY = dataclasses.replace(
    package.VANCOUVER_LOCATION_POLICY, state=state,
    authority=LocationPolicyAuthority(source="SYNTH-AUTHORITY"),
    canonical_location={canonical!r} if state is LocationPolicyState.CONFIRMED_ALIAS else None)
package.ANALYSIS_LOCATION_STREAM_COMPARISON = dataclasses.replace(
    package.ANALYSIS_LOCATION_STREAM_COMPARISON, identity_columns=({identity!r},))
"""


_SCOPE_OVERRIDE = """\
# Test-only kernel configuration (synthetic authority, never a real decision):
# a confirmed alias for the governed Vancouver keys whose canonical key is another city's stream.
import dataclasses

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis.schemas import LocationPolicyAuthority, LocationPolicyState

package.VANCOUVER_LOCATION_POLICY = dataclasses.replace(
    package.VANCOUVER_LOCATION_POLICY, state=LocationPolicyState.CONFIRMED_ALIAS,
    authority=LocationPolicyAuthority(source="SYNTH-AUTHORITY"), canonical_location={canonical!r})
"""


def test_ingestion_notebook_blocks_alias_canonicalised_into_another_city(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    # Regression: a confirmed Vancouver alias with the Calgary stream as canonical key was
    # authority-sufficient and its mapping rewrote both Vancouver labels to Calgary.
    from ql2_sixt_canada_analysis.readiness import PricingBlocker

    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        _SCOPE_OVERRIDE.format(canonical=INVESTIGATED_LOCATION_STREAM), encoding="utf-8")
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    pythonpath = os.pathsep.join(p for p in (str(site), os.environ.get("PYTHONPATH", "")) if p)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir), "PYTHONPATH": pythonpath})
    outputs = _step_output(result, "assess_pricing_readiness(")
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Vancouver identity policy state"].strip() == "confirmed_alias"   # recorded for audit
    assert lines["Policy authority-backed and resolved"].strip() == "True"
    assert lines["Canonical scope validation passed"].strip() == "False"
    assert lines["Supplied canonical key"].strip() == " / ".join(INVESTIGATED_LOCATION_STREAM)
    assert "canonical_key_crosses_governed_scope" in lines["Policy scope blocked by"]
    assert lines["Policy authority sufficient for analysis"].strip() == "False"
    assert lines["Canonicalization permitted"].strip() == "False"
    assert lines["Vancouver labels confirmed aliases"].strip() == "False"
    assert lines["Canonicalization required"].strip() == "True | applied: False"
    assert lines["Pricing analysis ready"].strip() == "False"
    assert PricingBlocker.CANONICAL_LOCATION_CITY_MISMATCH.value in lines["Pricing blocked by"]
    assert "SYNTH" not in outputs and list(workdir.iterdir()) == []


@pytest.mark.parametrize("state", ["confirmed_alias", "confirmed_distinct"])
def test_ingestion_notebook_blocks_resolved_policy_on_mapping_defect(state: str, tmp_path: Path) -> None:
    # Regression (P1): a resolved policy stayed sufficient - and pricing could become
    # ready - although the comparison reported a location mapping defect. The notebook
    # source is unchanged; only the kernel's configuration is overridden.
    from test_comparison import ID_COL

    from ql2_sixt_canada_analysis.readiness import PricingBlocker
    from ql2_sixt_canada_analysis.schemas import COMPARED_LOCATION_STREAMS

    rel = JOB_DETAIL_RELATIONSHIP
    city_col, label_col = EXPECTED_LOCATION_COVERAGE.location_columns
    parent_city, = EXPECTED_LOCATION_COVERAGE.parent_scope_columns
    position, = rel.detail_definition.non_identifier_key_columns
    (city, first), (_, second) = COMPARED_LOCATION_STREAMS
    jobs = [{rel.parent_key_columns[0]: job, parent_city: city,
             **{c: "2" for c in rel.expected_detail_count_columns}} for job in ("SYNTH-JOB-001", "SYNTH-JOB-002")]
    # The first stream carries two different authoritative identities: a within-stream conflict.
    identities = {("SYNTH-JOB-001", first): "SYNTH-SITE-1", ("SYNTH-JOB-002", first): "SYNTH-SITE-3",
                  ("SYNTH-JOB-001", second): "SYNTH-SITE-2", ("SYNTH-JOB-002", second): "SYNTH-SITE-2"}
    cars = [{rel.detail_key_columns[0]: job, position: str(i), city_col: city, label_col: label, ID_COL: site}
            for i, ((job, label), site) in enumerate(identities.items())]
    directory = tmp_path / "raw"
    directory.mkdir()
    for dataset, dataset_rows in ((DatasetKey.JOBS, jobs), (DatasetKey.CARS, cars)):
        columns = DATASET_DEFINITIONS[dataset].columns
        lines = [",".join(r.get(c, f"synthetic_{i}") for i, c in enumerate(columns)) for r in dataset_rows]
        (directory / f"synthetic_{dataset}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_POLICY_OVERRIDE.format(
        state=state, canonical=(city, first), identity=ID_COL), encoding="utf-8")
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    pythonpath = os.pathsep.join(p for p in (str(site), os.environ.get("PYTHONPATH", "")) if p)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(directory), "PYTHONPATH": pythonpath})
    comparison = _step_output(result, "compare_location_streams(")
    assert "Behavioural comparison result: location_mapping_defect" in comparison
    outputs = _step_output(result, "assess_pricing_readiness(")
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Vancouver identity policy state"].strip() == state       # decision stays recorded
    assert lines["Policy authority-backed and resolved"].strip() == "True"
    assert lines["Identity evidence conflicts with policy"].strip() == "True"
    assert lines["Policy authority sufficient for analysis"].strip() == "False"
    assert lines["Vancouver labels confirmed aliases"].strip() == "False"
    assert lines["Vancouver labels comparable independently"].strip() == "False"
    assert lines["Pricing analysis ready"].strip() == "False"
    blocked = lines["Pricing blocked by"]
    assert PricingBlocker.IDENTITY_EVIDENCE_CONFLICT.value in blocked
    assert PricingBlocker.LOCATION_POLICY_UNRESOLVED.value not in blocked
    # Governed keys / canonical key are repository configuration and are shown on their own lines.
    shown = comparison + "\n".join(line for line in outputs.splitlines()
                                   if not line.startswith(("Governed", "Supplied canonical")))
    # Approved keys are ordinary words ("Vancouver", "Downtown"), so check the printed values, not the labels.
    values = " ".join(line.split(":", 1)[1] for line in shown.splitlines() if ":" in line)
    assert "SYNTH" not in shown and first not in values and second not in values and city not in values
    assert list(workdir.iterdir()) == []


# ------------------------------------------------------------ data completeness


def test_ingestion_notebook_gates_completeness_through_the_api() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    assert re.search(r"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\bassess_completeness\b", code, re.S)
    cell = next(s for s in sources if "assess_completeness(" in s)
    # The all-expected-stream aggregate, never a hand-picked subset (a single Calgary
    # report used to be passed, so other expected streams could fail unnoticed).
    for argument in ("datasets=analysis_datasets", "coverage=location_coverage_report", "streams=expected_streams_report",
                     "reconciliation=reconciliation_report", "city_integrity=city_integrity_report"):
        assert argument in cell
    assert "streams=(location_stream_report,)" not in code
    expected = next(s for s in sources if "assess_expected_location_streams(" in s)
    assert re.search(r"expected_streams_report\s*=\s*assess_expected_location_streams\(", expected)
    assert "coverage=EXPECTED_LOCATION_COVERAGE" in expected
    assert sources.index(expected) < sources.index(cell)
    pricing = next(s for s in sources if "assess_pricing_readiness(" in s)
    assert "completeness=completeness" in pricing and "source_complete" not in pricing
    for forbidden in ("nrows", "skipfooter", "skiprows", "on_bad_lines", "chunksize", "usecols"):
        assert forbidden not in code
    guidance = next(c.source for c in notebook.cells if c.source.startswith("## Next step"))
    assert "data_complete" in guidance


def test_ingestion_notebook_reports_completeness_on_synthetic_inputs(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    from ql2_sixt_canada_analysis.readiness import CompletenessBlocker

    workdir = tmp_path / "kernel"
    workdir.mkdir()
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)})
    loaded = _step_output(result, "load_raw_datasets(raw_dir)")
    assert "Complete-source ingestion rules enforced: True" in loaded
    outputs = _step_output(result, "assess_completeness(")
    lines = dict(line.split(":", 1) for line in outputs.splitlines() if ":" in line)
    assert lines["Overall completeness"].strip() == "not proven"
    blocked = lines["Completeness blocked by"]
    assert CompletenessBlocker.EXPECTED_PAIRS_MISSING.value in blocked
    assert CompletenessBlocker.SOURCE_NOT_COMPLETE.value not in blocked
    reconcile = _step_output(result, "assess_job_detail_reconciliation(")
    assert len(re.findall(r"^Declared .* reconciled:", reconcile, re.M)) == len(
        JOB_DETAIL_RELATIONSHIP.expected_detail_count_columns)
    assert list(workdir.iterdir()) == []


def test_ingestion_notebook_blocks_when_one_expected_stream_is_partial(tmp_path: Path) -> None:
    # Regression: only the investigated stream fed completeness and pricing, so the
    # notebook could report complete data while another expected stream was partial.
    from ql2_sixt_canada_analysis.readiness import CompletenessBlocker, PricingBlocker
    from ql2_sixt_canada_analysis.schemas import COMPARED_LOCATION_STREAMS

    rel = JOB_DETAIL_RELATIONSHIP
    cov = EXPECTED_LOCATION_COVERAGE
    city_col, label_col = cov.location_columns
    parent_city, = cov.parent_scope_columns
    position, = rel.detail_definition.non_identifier_key_columns
    (city1, label1) = INVESTIGATED_LOCATION_STREAM
    (city2, label2), (_, label3) = COMPARED_LOCATION_STREAMS
    counts = {"SYNTH-JOB-001": "1", "SYNTH-JOB-002": "2", "SYNTH-JOB-003": "1"}
    jobs = [{rel.parent_key_columns[0]: job, parent_city: city1 if job.endswith("1") else city2,
             **{c: n for c in rel.expected_detail_count_columns}} for job, n in counts.items()]
    cars = [{rel.detail_key_columns[0]: "SYNTH-JOB-001", position: "0", city_col: city1, label_col: label1},
            {rel.detail_key_columns[0]: "SYNTH-JOB-002", position: "0", city_col: city2, label_col: label2},
            {rel.detail_key_columns[0]: "SYNTH-JOB-002", position: "1", city_col: city2, label_col: label3},
            {rel.detail_key_columns[0]: "SYNTH-JOB-003", position: "0", city_col: city2, label_col: label2}]
    directory = tmp_path / "raw"
    directory.mkdir()
    for dataset, dataset_rows in ((DatasetKey.JOBS, jobs), (DatasetKey.CARS, cars)):
        columns = DATASET_DEFINITIONS[dataset].columns
        lines = [",".join(r.get(c, f"synthetic_{i}") for i, c in enumerate(columns)) for r in dataset_rows]
        (directory / f"synthetic_{dataset}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)})
    calgary = _step_output(result, "investigate_location_stream(")
    assert "Stream continuity: complete" in calgary                       # the investigated stream is fine
    streams = _step_output(result, "assess_expected_location_streams(")
    lines = dict(line.split(":", 1) for line in streams.splitlines() if ":" in line)
    population = str(len(cov.expected_locations))                         # every approved stream, not three
    assert lines["Configured expected streams"].strip() == population and lines["Assessed streams"].strip() == population
    assert lines["Every expected stream assessed exactly once"].strip() == "True"
    assert lines["All expected streams healthy"].strip() == "False"
    assert "stream_continuity_partial" in lines["Expected streams blocked by"]
    assert not any(label in streams for label in (label1, label2, label3)) and "SYNTH" not in streams
    complete = _step_output(result, "assess_completeness(")
    assert "Overall completeness: not proven" in complete
    assert CompletenessBlocker.STREAM_CONTINUITY_PARTIAL.value in complete
    pricing = _step_output(result, "assess_pricing_readiness(")
    assert "Pricing analysis ready: False" in pricing
    assert PricingBlocker.EXPECTED_STREAMS_NOT_PROVEN.value in pricing
    assert list(workdir.iterdir()) == []


def test_ingestion_notebook_assesses_city_integrity_through_the_api() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    assert re.search(r"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\bassess_city_integrity\b", code, re.S)
    cell = next(s for s in sources if "assess_city_integrity(" in s)
    assert re.search(r"^city_integrity_report\s*=\s*assess_city_integrity\(", cell, re.M)
    assert "coverage=EXPECTED_LOCATION_COVERAGE" in cell and "city_integrity_valid" in cell
    assert "_sample" not in code                                  # identifiers never printed
    order = [next(i for i, s in enumerate(sources) if call in s) for call in (
        "assess_city_integrity(", "assess_job_detail_join_readiness(", "assess_completeness(",
        "assess_pricing_readiness(")]
    assert order == sorted(order)
    join = sources[order[1]]
    assert "job_detail_join.city_integrity_valid != city_integrity_valid" in join
    assert "city_integrity=city_integrity_report" in sources[order[2]]
    guidance = next(c.source for c in notebook.cells if c.source.startswith("## Next step"))
    assert "city_integrity_valid" in guidance


@pytest.mark.parametrize("defect", ["cross_city_row", "blank_job_city"])
def test_ingestion_notebook_blocks_on_city_integrity_defects(defect: str, tmp_path: Path) -> None:
    # Regression: coverage, all three streams and the declared counts passed while a
    # detail row sat under another city than its job, or a job's city was blank.
    from ql2_sixt_canada_analysis.city_integrity import CityIntegrityBlocker
    from ql2_sixt_canada_analysis.join_readiness import JobDetailJoinBlocker
    from ql2_sixt_canada_analysis.readiness import PricingBlocker
    from ql2_sixt_canada_analysis.schemas import COMPARED_LOCATION_STREAMS

    rel = JOB_DETAIL_RELATIONSHIP
    city_col, label_col = EXPECTED_LOCATION_COVERAGE.location_columns
    parent_city, = EXPECTED_LOCATION_COVERAGE.parent_scope_columns
    position, = rel.detail_definition.non_identifier_key_columns
    (city1, label1) = INVESTIGATED_LOCATION_STREAM
    (city2, label2), (_, label3) = COMPARED_LOCATION_STREAMS
    job_rows = [("SYNTH-JOB-001", city1, "1"), ("SYNTH-JOB-002", city2, "2")]
    detail_rows = [("SYNTH-JOB-001", city1, label1), ("SYNTH-JOB-002", city2, label2),
                   ("SYNTH-JOB-002", city2, label3)]
    if defect == "cross_city_row":
        job_rows.append(("SYNTH-JOB-003", city1, "2"))
        detail_rows += [("SYNTH-JOB-003", city1, label1), ("SYNTH-JOB-003", city2, label2)]
        expected = CityIntegrityBlocker.PARENT_DETAIL_CITY_MISMATCH
    else:
        job_rows.append(("SYNTH-JOB-003", "   ", "0"))
        expected = CityIntegrityBlocker.CITY_SCOPE_UNASSIGNABLE
    jobs = [{rel.parent_key_columns[0]: job, parent_city: city, **{c: n for c in rel.expected_detail_count_columns}}
            for job, city, n in job_rows]
    cars = [{rel.detail_key_columns[0]: job, position: str(i), city_col: city, label_col: label}
            for i, (job, city, label) in enumerate(detail_rows)]
    directory = tmp_path / "raw"
    directory.mkdir()
    for dataset, dataset_rows in ((DatasetKey.JOBS, jobs), (DatasetKey.CARS, cars)):
        columns = DATASET_DEFINITIONS[dataset].columns
        lines = [",".join(r.get(c, f"synthetic_{i}") for i, c in enumerate(columns)) for r in dataset_rows]
        (directory / f"synthetic_{dataset}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)})
    integrity = _step_output(result, "assess_city_integrity(")
    lines = dict(line.split(":", 1) for line in integrity.splitlines() if ":" in line)
    assert lines["Scope integrity passed"].strip() == "False"
    assert expected.value in lines["Scope integrity blocked by"]
    assert lines["Jobs with unassignable scope"].strip() == ("1" if defect == "blank_job_city" else "0")
    assert lines["Linked detail rows disagreeing with their parent scope"].strip() == (
        "1" if defect == "cross_city_row" else "0")
    join = _step_output(result, "assess_job_detail_join_readiness(")
    assert "Trusted join ready: False" in join and "Scope integrity passed: False" in join
    assert JobDetailJoinBlocker(expected.value).value in join
    assert "Joined frame held: diagnostic only - UNTRUSTED" in join
    complete = _step_output(result, "assess_completeness(")
    assert "Overall completeness: not proven" in complete and expected.value in complete
    pricing = _step_output(result, "assess_pricing_readiness(")
    assert "Pricing analysis ready: False" in pricing
    assert PricingBlocker.SCOPE_INTEGRITY_NOT_PROVEN.value in pricing
    shown = integrity + join + complete + pricing
    assert "SYNTH" not in shown and " / ".join(INVESTIGATED_LOCATION_STREAM) not in shown   # governed keys are configuration
    assert list(workdir.iterdir()) == []


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


# ------------------------------------------------------ authority-backed job linkage


def test_ingestion_notebook_links_jobs_through_the_api_in_pipeline_order() -> None:
    notebook = read_notebook(INGESTION_NOTEBOOK)
    sources = [c.source for c in _code_cells(notebook)]
    code = "\n".join(sources)
    for name in ("assess_job_linkage", "load_job_linkage_policy", "ANALYSIS_DATASET_DEFINITIONS",
                 "ANALYSIS_JOB_DETAIL_RELATIONSHIP"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S)

    def index(marker: str) -> int:
        return next(i for i, s in enumerate(sources) if marker in s)

    link = index("assess_job_linkage(")
    assert index("remove_blank_rows_from_raw_datasets(") < index("validate_raw_dataset_identifier_dtypes(") < link
    for later in ("assess_raw_dataset_unique_keys(", "assess_job_detail_reconciliation(", "assess_one_to_many_join(",
                  "investigate_location_stream(", "assess_expected_location_streams(", "assess_city_integrity(",
                  "assess_job_detail_join_readiness(", "assess_pricing_readiness("):
        assert link < index(later), later
    cell = sources[link]
    assert re.search(r"job_linkage_policy\s*=\s*load_job_linkage_policy\(\s*\)", cell)
    assert re.search(r"job_linkage\s*=\s*assess_job_linkage\(\s*jobs_df\s*,\s*cars_df\s*,\s*job_linkage_policy\s*\)", cell)
    assert re.search(r"analysis_datasets\s*=\s*job_linkage\.datasets\(\s*cleaned\s*\)", cell)
    pricing = sources[index("assess_pricing_readiness(")]
    assert "job_linkage=job_linkage_report" in pricing
    # After linkage, analytical steps never use the raw frames or the raw-source relationship.
    after = "\n".join(sources[link + 1:])
    assert not re.search(r"\b(jobs_df|cars_df)\b", after)
    assert not re.search(r"(?<!ANALYSIS_)\bJOB_DETAIL_RELATIONSHIP\b", code)
    assert not re.search(r"[\"'](job_id|row_index|job_id_linkage_key|row_index_key)[\"']", code)


def _legacy_export(directory: Path, *, ambiguous: bool = False) -> None:
    """Fabricated CSVs shaped like the historical export (digit ids; decimal-zero detail forms)."""
    rel = JOB_DETAIL_RELATIONSHIP
    jobs = [{rel.parent_key_columns[0]: "5101", rel.expected_detail_count_column: "2"},
            {rel.parent_key_columns[0]: "05102", rel.expected_detail_count_column: "1"}]
    if ambiguous:
        jobs.append({rel.parent_key_columns[0]: "5101.0", rel.expected_detail_count_column: "0"})
    cars = [{rel.detail_key_columns[0]: "5101.0", "row_index": "0.0"},
            {rel.detail_key_columns[0]: "5101.0", "row_index": "1.0"},
            {rel.detail_key_columns[0]: "05102.0", "row_index": "0.0"}]
    for key, key_rows in ((DatasetKey.JOBS, jobs), (DatasetKey.CARS, cars)):
        columns = DATASET_DEFINITIONS[key].columns
        lines = [",".join(r.get(c, f"synthetic_{i}_{n}") for i, c in enumerate(columns)) for n, r in enumerate(key_rows)]
        (directory / f"synthetic_{key}.csv").write_bytes(
            (",".join(columns) + "\n" + "\n".join(lines) + "\n").encode("utf-8"))


@pytest.mark.parametrize("ambiguous", [False, True])
def test_ingestion_notebook_linkage_step_reports_aggregates_only(tmp_path: Path, ambiguous: bool) -> None:
    directory = tmp_path / "raw"
    directory.mkdir()
    _legacy_export(directory, ambiguous=ambiguous)
    workdir = tmp_path / "kernel"
    workdir.mkdir()
    repo_before = _snapshot(PROJECT_ROOT)
    result = execute_notebook_copy(INGESTION_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(directory)})
    linkage = _step_output(result, "assess_job_linkage(")
    join = _step_output(result, "assess_job_detail_join_readiness(")
    pricing = _step_output(result, "assess_pricing_readiness(")
    assert "Linkage policy: available" in linkage
    if ambiguous:
        assert "Job linkage valid: False" in linkage and "detail_job_reference_ambiguous" in linkage
        assert "Job linkage valid: False" in join and "Trusted join ready: False" in join
        assert "job_key_normalization_not_ready" in pricing and "Pricing analysis ready: False" in pricing
    else:
        assert "Job linkage valid: True" in linkage and "approved legacy repair: 3" in linkage
        assert "Job linkage blocked by: nothing" in linkage and "Job linkage valid: True" in join
        assert "Job linkage ready: True" in pricing
    for value in ("5101", "05102", "SYNTH", "synthetic_"):
        assert value not in linkage + join + pricing
    assert not any(column in linkage for key in DatasetKey for column in contract_columns(key))
    assert list(workdir.iterdir()) == [] and _snapshot(PROJECT_ROOT) == repo_before


# ------------------------------------------------------- matched location pricing (02)

MATCHED_PRICING_NOTEBOOK = NOTEBOOKS_DIR / "02_matched_location_pricing.ipynb"


def test_matched_pricing_notebook_uses_package_apis_only() -> None:
    notebook = read_notebook(MATCHED_PRICING_NOTEBOOK)
    code = _code_source(notebook)
    for name in ("run_matched_location_pricing", "match_count_frame", "city_summary_frame",
                 "vehicle_type_summary_frame", "vehicle_type_test_frame", "plot_matched_location_premiums"):
        assert re.search(rf"from ql2_sixt_canada_analysis import\s*\(?[^)]*?\b{name}\b", code, re.S), name
        assert f"{name}(" in code
    # No re-implemented matching, aggregation, statistics or plotting, and no row-level display.
    for pattern in (r"\.merge\(", r"\.groupby\(", r"\.pivot", r"kruskal", r"scipy", r"\.boxplot\(",
                    r"\.scatter\(", r"\.violinplot\(", r"pyplot", r"plt\.", r"\.median\(", r"\.mean\(",
                    r"\.quantile\(", r"\.percentile\(", r"result\.pairs", r"\.pairs\b", r"\.offers\b",
                    r"\.(head|tail|sample|describe|info|to_string|to_markdown)\(", r"job_id", r"\bjobs\b",
                    r"\bcars\b", r"assess_matched_location_pricing\("):
        assert not re.search(pattern, code), f"notebook re-implements or exposes: {pattern}"
    assert "plt.show" not in code and "savefig(buffer" in code
    assert re.search(r"REPORTS_DIR\s*=\s*None", code) and re.search(r"RAW_DIR_OVERRIDE\s*=\s*None", code)


def test_matched_pricing_notebook_is_documented_and_independent_of_notebook_01() -> None:
    notebook = read_notebook(MATCHED_PRICING_NOTEBOOK)
    code = _code_source(notebook)
    assert "%run" not in code and "01_data_ingestion" not in code and "import ipynb" not in code
    text = " ".join("\n".join(c.source for c in notebook.cells if c.cell_type == "markdown").lower().split())
    assert "confidential" in text and "associational" in text and "airport minus downtown" in text
    readme = (NOTEBOOKS_DIR / "README.md").read_text(encoding="utf-8")
    assert MATCHED_PRICING_NOTEBOOK.name in readme


def test_matched_pricing_notebook_runs_clean_and_stops_when_not_ready(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    repo_before = _snapshot(PROJECT_ROOT)
    workdir = tmp_path / "outside_repository"
    workdir.mkdir()
    result = execute_notebook_copy(MATCHED_PRICING_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)}, timeout_seconds=300)
    assert result.execution_counts == tuple(range(1, len(_code_cells(result.executed)) + 1))
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "Matched location pricing status: blocked" in outputs
    assert "pricing_not_ready" in outputs and "no commercial result" in outputs
    assert "Skipped: matched location pricing is blocked." in outputs and "No files written." in outputs
    assert "synthetic_" not in outputs and str(synthetic_raw_dir) not in outputs and str(tmp_path) not in outputs
    assert not re.search(r"\b[0-9]+\b", outputs), "a blocked run shows no numbers"
    assert not any(o.get("output_type") in {"execute_result", "display_data"}
                   for c in _code_cells(result.executed) for o in c.outputs), "no tables or figures when blocked"
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")
                  if "outside_repository" not in p.parts) == before
    assert not any(workdir.iterdir()), "notebook wrote files"
    assert _snapshot(PROJECT_ROOT) == repo_before


# ------------------------------------------------------- price-change events (03)

PRICE_CHANGE_NOTEBOOK = NOTEBOOKS_DIR / "03_price_change_events.ipynb"
PRICE_CHANGE_SECTIONS = ("# 03 — Price-change events", "## Data-plan scope", "## Pipeline and readiness",
                         "## Event population and reconciliation", "## Synchronized movements",
                         "## Airport/downtown comparisons", "## Persistence", "## Final Vancouver decrease",
                         "## Event heatmap", "## Limitations and interpretation", "## Local detailed export")


def test_price_change_notebook_has_the_required_sections_in_order() -> None:
    notebook = read_notebook(PRICE_CHANGE_NOTEBOOK)
    headings = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == "markdown"]
    positions = [next(i for i, h in enumerate(headings) if h.startswith(s)) for s in PRICE_CHANGE_SECTIONS]
    assert positions == sorted(positions)
    text = " ".join("\n".join(c.source for c in notebook.cells if c.cell_type == "markdown").split())
    for phrase in ("observed price-change candidates", "roughly 90 hours", "not proof of intentional repricing",
                   "Right-censored events provide no persistence evidence", "Visible assortment stability",
                   "Monitoring and actionability", "Clear all outputs", "operational corroboration"):
        assert phrase in text, phrase


def test_price_change_notebook_delegates_to_package_functions_and_never_shows_event_rows() -> None:
    notebook = read_notebook(PRICE_CHANGE_NOTEBOOK)
    code = _code_source(notebook)
    for name in ("run_price_change_presentation", "presentation_settings", "heatmap_png", "resolve_raw_data_dir"):
        assert f"{name}(" in code, name
    assert code.count("run_price_change_presentation(") == 1, "the pipeline runs once"
    for pattern in (r"\.merge\(", r"\.groupby\(", r"\.pivot", r"\.sum\(", r"\.median\(", r"\.mean\(",
                    r"\.quantile\(", r"Fraction", r"pyplot", r"plt\.", r"imshow", r"\.candidates\b",
                    r"\.event_table\b", r"\.cross_location\b", r"analysis\.persistence\b", r"\.offers\b",
                    r"build_detailed_event_table", r"write_detailed_event_table", r"read_parquet",
                    r"\.(head|tail|sample|info|to_string|to_markdown|to_html)\(", r"(?<!case)\.describe\(",
                    r"\bHTML\(", r"job_id", r"\bjobs\b", r"\bcars\b", r"run_pricing_pipeline", r"assess_price_change",
                    r"analyze_price_change_events", r"classify_price_change", r"capture_timelines",
                    r"event_heatmap_source", r"savefig", r"magnitude_disclosable", r"magnitude_suppressed\(",
                    r"MINIMUM_MAGNITUDE_CONTRIBUTORS", r"contributor_count\s*[<>=]", r"\.fillna\("):
        assert not re.search(pattern, code), f"notebook re-implements or exposes: {pattern}"
    assert re.search(r"OUTPUT_DIR\s*=\s*None", code) and re.search(r"WRITE_DETAIL\s*=\s*None", code)
    raw = PRICE_CHANGE_NOTEBOOK.read_text(encoding="utf-8")
    assert "image/png" not in raw and "attachments" not in raw and "base64" not in raw


def test_price_change_notebook_stops_clearly_on_blocked_synthetic_data(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    repo_before = _snapshot(PROJECT_ROOT)
    workdir = tmp_path / "outside_repository"
    workdir.mkdir()
    result = execute_notebook_copy(PRICE_CHANGE_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)}, timeout_seconds=300)
    assert result.execution_counts == tuple(range(1, len(_code_cells(result.executed)) + 1))
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "Price-change presentation status: blocked" in outputs and "analysis_not_completed" in outputs
    assert "Skipped: the presentation is blocked." in outputs and "No files written." in outputs
    assert "synthetic_" not in outputs and str(synthetic_raw_dir) not in outputs and str(tmp_path) not in outputs
    assert not any(o.get("output_type") in {"execute_result", "display_data"}
                   for c in _code_cells(result.executed) for o in c.outputs), "no tables or figures when blocked"
    assert not any(workdir.iterdir()) and _snapshot(PROJECT_ROOT) == repo_before


def test_price_change_notebook_runs_top_to_bottom_on_synthetic_ready_data(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed cells, in order, against a ready synthetic pipeline result (in process; nothing written)."""
    import pandas as pd
    from test_price_change_events import pipeline_result, synthetic_world
    from test_price_change_presentation import RICH

    from ql2_sixt_canada_analysis import pricing_pipeline
    from ql2_sixt_canada_analysis.price_change_presentation import validate_sanitized_frame

    world = synthetic_world(products=RICH)
    calls: list[object] = []
    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline",
                        lambda raw_dir=None: calls.append(raw_dir) or pipeline_result(world))
    monkeypatch.delenv("QL2_SIXT_PRICE_CHANGE_OUTPUT_DIR", raising=False)
    monkeypatch.setenv(paths.RAW_DATA_DIR_ENV_VAR, str(tmp_path / "synthetic_raw"))
    monkeypatch.chdir(tmp_path)
    repo_before = _snapshot(PROJECT_ROOT)
    shown: list[object] = []
    printed: list[str] = []
    namespace = {"__name__": "__main__", "print": lambda *a, **k: printed.append(" ".join(map(str, a)))}
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", lambda obj, *a, **k: shown.append(obj))
    for cell in _code_cells(read_notebook(PRICE_CHANGE_NOTEBOOK)):
        exec(compile(cell.source, "<notebook-cell>", "exec"), namespace)   # noqa: S102 - the committed cells
    assert len(calls) == 1
    text = "\n".join(printed)
    assert "Price-change presentation status: completed" in text and "No files written." in text
    assert "SYNTH" not in text and str(tmp_path) not in text
    frames = [o for o in shown if isinstance(o, pd.DataFrame)]
    names = [n for n, f in namespace["tables"].items()]
    assert len(frames) == 7 and len(names) == 7
    for frame in frames:
        name = next(n for n, f in namespace["tables"].items() if f is frame)
        validate_sanitized_frame(name, frame)                                # only sanitized tables are shown
        assert "SYNTH" not in frame.to_csv(index=False)
    images = [o for o in shown if type(o).__name__ == "Image"]
    assert len(images) == 1 and images[0].data[:4] == b"\x89PNG"
    assert os.listdir(tmp_path) == [] and _snapshot(PROJECT_ROOT) == repo_before


# ------------------------------------------------------- visible assortment (04)

ASSORTMENT_NOTEBOOK = NOTEBOOKS_DIR / "04_visible_assortment.ipynb"
ASSORTMENT_SECTIONS = ("# 04 — Visible assortment", "## Question and scope", "## Definitions", "## Confidentiality",
                       "## Readiness and evidence", "## Assortment timeline",
                       "## Additions, removals, retention, and Jaccard", "## Observed-drop review",
                       "## Price-change coincidence", "## Interpretation and limitations",
                       "## Monitoring and actionability", "## Data-plan reconciliation",
                       "## Clear all outputs before committing")


def test_assortment_notebook_has_the_required_sections_and_limitations() -> None:
    notebook = read_notebook(ASSORTMENT_NOTEBOOK)
    headings = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == "markdown"]
    positions = [next(i for i, h in enumerate(headings) if h == s) for s in ASSORTMENT_SECTIONS]
    assert positions == sorted(positions)
    text = " ".join("\n".join(c.source for c in notebook.cells if c.cell_type == "markdown").split())
    for phrase in ("unusual-drop policy is unavailable", "review candidates, not proven anomalies",
                   "price coincidence is not causation", "Timeline persistence is not approved",
                   "roughly 90 hours", "previous set as denominator", "zero_denominator", "Clear all outputs",
                   "Count returned products by location and capture: presented in the aggregate timeline",
                   "Produce an assortment timeline: implemented as a validated in-memory aggregate table",
                   "statistical unusualness and alerting remain unavailable pending approved policy"):
        assert phrase in text, phrase


def test_assortment_notebook_delegates_to_package_functions_only() -> None:
    notebook = read_notebook(ASSORTMENT_NOTEBOOK)
    code = _code_source(notebook)
    for name in ("run_assortment_presentation", "build_assortment_narrative", "assortment_timeline_png",
                 "resolve_raw_data_dir"):
        assert f"{name}(" in code, name
    assert code.count("run_assortment_presentation(") == 1, "the pipeline runs once"
    for pattern in (r"\bdef\b", r"\.merge\(", r"\.groupby\(", r"\.pivot", r"\.sum\(", r"\.median\(", r"\.mean\(",
                    r"\.quantile\(", r"\.std\(", r"pyplot", r"plt\.", r"savefig", r"\.membership\b", r"\.offers\b",
                    r"\.candidates\b", r"\.assortment\b", r"compare_assortment", r"calculate_visible_assortment",
                    r"run_pricing_pipeline", r"visible_assortment_from_pipeline", r"to_csv", r"to_parquet",
                    r"to_json", r"output_dir", r"OUTPUT_DIR", r"open\(", r"\.(head|tail|sample|info|to_string|"
                    r"to_markdown|to_html)\(", r"job_id", r"\bjobs\b", r"\bcars\b", r"\.fillna\(", r"threshold"):
        assert not re.search(pattern, code), f"notebook re-implements or exposes: {pattern}"
    raw = ASSORTMENT_NOTEBOOK.read_text(encoding="utf-8")
    assert "image/png" not in raw and "attachments" not in raw and "base64" not in raw


def test_assortment_notebook_stops_clearly_on_blocked_synthetic_data(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    repo_before = _snapshot(PROJECT_ROOT)
    workdir = tmp_path / "outside_repository"
    workdir.mkdir()
    result = execute_notebook_copy(ASSORTMENT_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)}, timeout_seconds=300)
    assert result.execution_counts == tuple(range(1, len(_code_cells(result.executed)) + 1))
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    assert "Visible-assortment presentation status: blocked" in outputs and "assortment_not_completed" in outputs
    assert "Skipped: the presentation is blocked." in outputs and "No files written." in outputs
    assert "No tables, figure or findings are produced" in outputs
    assert not re.search(r"\d", outputs), "a blocked notebook shows no numbers"
    assert "synthetic_" not in outputs and str(synthetic_raw_dir) not in outputs and str(tmp_path) not in outputs
    assert not any(o.get("output_type") in {"execute_result", "display_data"}
                   for c in _code_cells(result.executed) for o in c.outputs), "no tables or figures when blocked"
    assert not any(workdir.iterdir()) and _snapshot(PROJECT_ROOT) == repo_before


def test_assortment_notebook_runs_top_to_bottom_on_synthetic_ready_data(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed cells, in order, against a ready synthetic pipeline result (in process; nothing written)."""
    import pandas as pd
    from test_assortment_presentation import RICH
    from test_price_change_events import TOR_AIR, pipeline_result, synthetic_world

    from ql2_sixt_canada_analysis import pricing_pipeline
    from ql2_sixt_canada_analysis.assortment_presentation import validate_assortment_table

    world = synthetic_world(products=RICH, withheld={(TOR_AIR, 1)})
    calls: list[object] = []
    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline",
                        lambda raw_dir=None: calls.append(raw_dir) or pipeline_result(world))
    monkeypatch.setenv(paths.RAW_DATA_DIR_ENV_VAR, str(tmp_path / "synthetic_raw"))
    monkeypatch.chdir(tmp_path)
    repo_before = _snapshot(PROJECT_ROOT)
    shown: list[object] = []
    printed: list[str] = []
    namespace = {"__name__": "__main__", "print": lambda *a, **k: printed.append(" ".join(map(str, a)))}
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", lambda obj, *a, **k: shown.append(obj))
    for cell in _code_cells(read_notebook(ASSORTMENT_NOTEBOOK)):
        exec(compile(cell.source, "<notebook-cell>", "exec"), namespace)   # noqa: S102 - the committed cells
    assert len(calls) == 1
    text = "\n".join(printed)
    assert "Visible-assortment presentation status: completed" in text and "No files written." in text
    assert "Unusual-drop policy status: unavailable" in text and "Timeline persistence approved: False" in text
    assert "Observed-drop review" in text and "not causation" in text
    assert "SYNTH" not in text and str(tmp_path) not in text and "$" not in text
    frames = [o for o in shown if isinstance(o, pd.DataFrame)]
    tables = dict(namespace["tables"].items())
    assert len(frames) == len(tables) == 6
    for frame in frames:
        name = next(n for n, f in tables.items() if f is frame)
        validate_assortment_table(name, frame)                              # only sanitized tables are shown
        assert "SYNTH" not in frame.to_csv(index=False)
    images = [o for o in shown if type(o).__name__ == "Image"]
    assert len(images) == 1 and images[0].data[:4] == b"\x89PNG"
    assert os.listdir(tmp_path) == [] and _snapshot(PROJECT_ROOT) == repo_before


# ------------------------------------------------------- monitoring and actionability (05)

MONITORING_NOTEBOOK = NOTEBOOKS_DIR / "05_monitoring_actionability.ipynb"
MONITORING_SECTIONS = ("# 05 — Monitoring and actionability", "## Data-plan scope", "## Severity scale",
                       "## Evaluation statuses", "## Confidentiality and side effects", "## Pipeline evidence",
                       "## Control definitions and sample evaluation",
                       "## Current sample evidence versus proposed operational controls",
                       "## Candidate rules and right-censored observations",
                       "## Limitations and production calibration", "## Data-plan reconciliation",
                       "## Clear all outputs before committing")


def test_monitoring_notebook_has_the_required_sections_in_order_and_its_reconciliation() -> None:
    from ql2_sixt_canada_analysis.monitoring import DATA_PLAN_RECONCILIATION, MONITORING_CONTROLS

    notebook = read_notebook(MONITORING_NOTEBOOK)
    headings = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == "markdown"]
    positions = [next(i for i, h in enumerate(headings) if h == s) for s in MONITORING_SECTIONS]
    assert positions == sorted(positions)
    text = " ".join("\n".join(c.source for c in notebook.cells if c.cell_type == "markdown").split())
    for control in MONITORING_CONTROLS:
        assert control.name in text and DATA_PLAN_RECONCILIATION[control.control_id] in text
    for phrase in ("roughly 90 hours", "not a statistical confidence level", "never a pass", "candidate-only",
                   "confirmation_required", "Right-censored events provide no persistence evidence",
                   "never bridged", "never establishes an alias", "Vancouver Downtown and Thurlow",
                   "not proof of intentional repricing", "not a scheduler", "Nothing is written",
                   "Clear all outputs", "exactly once"):
        assert phrase in text, phrase


def test_monitoring_notebook_delegates_to_package_functions_only() -> None:
    notebook = read_notebook(MONITORING_NOTEBOOK)
    code = _code_source(notebook)
    for name in ("run_monitoring", "monitoring_control_table", "monitoring_summary_lines", "severity_scale_table",
                 "status_legend_table", "resolve_raw_data_dir"):
        assert f"{name}(" in code, name
    assert code.count("run_monitoring(") == 1, "the pipeline runs once"
    for pattern in (r"\bdef\b", r"\blambda\b", r"\.merge\(", r"\.groupby\(", r"\.pivot", r"\.sum\(", r"\.median\(",
                    r"\.mean\(", r"\.quantile\(", r"\.std\(", r"\.max\(", r"\.min\(", r"pyplot", r"plt\.",
                    r"savefig", r"\.evidence\b", r"\.event_table\b", r"\.timeline\b", r"\.persistence\b",
                    r"\.membership\b", r"\.offers\b", r"\.candidates\b", r"run_pricing_pipeline",
                    r"monitoring_from_pipeline", r"evaluate_monitoring_controls", r"ControlEvaluation",
                    r"MonitoringFinding", r"ControlStatus\.", r"SynchronizedMovementPolicy", r"UnusualDropPolicy",
                    r"policy\s*=", r"to_csv", r"to_parquet", r"to_json", r"open\(", r"output_dir", r"os\.environ",
                    r"getenv",
                    r"\.(head|tail|sample|info|to_string|to_markdown|to_html)\(", r"job_id", r"\bjobs\b",
                    r"\bcars\b", r"\.fillna\(", r"threshold", r"alert\(", r"notify", r"requests", r"smtp"):
        assert not re.search(pattern, code), f"notebook re-implements or exposes: {pattern}"
    raw = MONITORING_NOTEBOOK.read_text(encoding="utf-8")
    assert "image/png" not in raw and "attachments" not in raw and "base64" not in raw


def _run_monitoring_cells(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: object) -> tuple:
    from ql2_sixt_canada_analysis import pricing_pipeline

    calls: list[object] = []
    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline", lambda raw_dir=None: calls.append(raw_dir) or run)
    monkeypatch.setenv(paths.RAW_DATA_DIR_ENV_VAR, str(tmp_path / "synthetic_raw"))
    monkeypatch.chdir(tmp_path)
    shown: list[object] = []
    printed: list[str] = []
    namespace = {"__name__": "__main__", "print": lambda *a, **k: printed.append(" ".join(map(str, a)))}
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", lambda obj, *a, **k: shown.append(obj))
    for cell in _code_cells(read_notebook(MONITORING_NOTEBOOK)):
        exec(compile(cell.source, "<notebook-cell>", "exec"), namespace)   # noqa: S102 - the committed cells
    return calls, shown, "\n".join(printed), namespace


def test_monitoring_notebook_runs_top_to_bottom_on_synthetic_ready_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed cells, in order, against a ready synthetic pipeline result (in process; nothing written)."""
    import pandas as pd
    from test_monitoring import FINAL, full_run
    from test_price_change_events import synthetic_world

    from ql2_sixt_canada_analysis.monitoring import validate_monitoring_table

    repo_before = _snapshot(PROJECT_ROOT)
    calls, shown, text, namespace = _run_monitoring_cells(monkeypatch, tmp_path,
                                                          full_run(synthetic_world(products=FINAL)))
    assert len(calls) == 1
    assert "Monitoring evaluation status: evaluated" in text and "No files written." in text
    assert "candidate_only: abrupt_assortment_changes, large_synchronized_price_movements" in text
    assert "confirmation_required: unconfirmed_end_of_window_anomalies" in text
    assert "SYNTH" not in text and str(tmp_path) not in text and "$" not in text
    frames = [o for o in shown if isinstance(o, pd.DataFrame)]
    assert len(frames) == 3
    validate_monitoring_table(frames[2])
    assert frames[2] is namespace["controls"] and len(frames[2]) == 8
    assert all("SYNTH" not in f.to_csv(index=False) for f in frames)
    assert os.listdir(tmp_path) == [] and _snapshot(PROJECT_ROOT) == repo_before


def test_monitoring_notebook_shows_only_definitions_and_categories_when_blocked(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    import pandas as pd
    from test_monitoring import full_run
    from test_price_change_events import synthetic_world

    from ql2_sixt_canada_analysis.monitoring import MONITORING_CONTROLS

    repo_before = _snapshot(PROJECT_ROOT)
    run = dataclasses.replace(full_run(synthetic_world()), pricing=None)
    calls, shown, text, namespace = _run_monitoring_cells(monkeypatch, tmp_path, run)
    assert len(calls) == 1
    assert "Monitoring evaluation status: blocked" in text and "pipeline_evidence_unavailable" in text
    assert "the control definitions remain available" in text and "No files written." in text
    controls = namespace["controls"]
    assert isinstance(controls, pd.DataFrame) and set(controls["evaluation_status"]) == {"not_assessable"}
    assert controls["condition"].tolist() == [c.condition for c in MONITORING_CONTROLS]
    assert set(controls["unavailable_evidence"]) == {"monitoring_evidence_blocked"}
    assert set(controls["findings"]) == {""}
    assert not re.search(r"\d", text) and "SYNTH" not in text
    assert os.listdir(tmp_path) == [] and _snapshot(PROJECT_ROOT) == repo_before


def test_monitoring_notebook_executes_from_a_clean_kernel_on_synthetic_data(synthetic_raw_dir: Path,
                                                                            tmp_path: Path) -> None:
    repo_before = _snapshot(PROJECT_ROOT)
    workdir = tmp_path / "outside_repository"
    workdir.mkdir()
    result = execute_notebook_copy(MONITORING_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)}, timeout_seconds=300)
    assert result.execution_counts == tuple(range(1, len(_code_cells(result.executed)) + 1))
    outputs = "\n".join(o.get("text", "") for c in _code_cells(result.executed) for o in c.outputs)
    rendered = json.dumps([o for c in _code_cells(result.executed) for o in c.outputs])
    assert "Monitoring evaluation status: partially_evaluated" in outputs and "No files written." in outputs
    assert "triggered: missing_expected_locations" in outputs           # expectations come from the contract
    assert "Pricing-readiness blockers:" in outputs
    assert "synthetic_" not in rendered and str(synthetic_raw_dir) not in rendered and str(tmp_path) not in rendered
    assert "image/png" not in rendered
    assert not any(workdir.iterdir()) and _snapshot(PROJECT_ROOT) == repo_before


# ------------------------------------------------------- result / interpretation convention (all notebooks)

INTERPRETATION_HELPERS = ("interpret_section(", "final_conclusions(", "build_assortment_narrative(")
MAX_INTERPRETATION_WORDS = 120


def _tags(cell: nbformat.NotebookNode) -> list[str]:
    return list(cell.metadata.get("tags", []))


@pytest.mark.parametrize("notebook_path", TRACKED_NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_every_result_is_immediately_followed_by_an_interpretation(notebook_path: Path) -> None:
    cells = read_notebook(notebook_path).cells
    results = [i for i, c in enumerate(cells) if "result" in _tags(c)]
    assert results, "every notebook tags its result cells"
    for index in results:
        assert cells[index].cell_type == "code", f"cell {index}: only code cells produce results"
        assert index + 1 < len(cells), f"cell {index}: a result needs an interpretation after it"
        following = cells[index + 1]
        assert "interpretation" in _tags(following), f"cell {index}: the next cell must be the interpretation"
        if following.cell_type == "markdown":
            assert following.source.startswith("**Interpretation.**"), f"cell {index + 1}: unlabelled interpretation"
            assert len(following.source.split()) <= MAX_INTERPRETATION_WORDS, f"cell {index + 1}: not concise"
        else:
            assert any(h in following.source for h in INTERPRETATION_HELPERS), \
                f"cell {index + 1}: runtime interpretation must come from a tested package helper"


@pytest.mark.parametrize("notebook_path", TRACKED_NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_every_displayed_table_or_figure_is_a_tagged_result(notebook_path: Path) -> None:
    for index, cell in enumerate(read_notebook(notebook_path).cells):
        if cell.cell_type == "code":
            body = "\n".join(line for line in cell.source.splitlines()
                             if not re.match(r"\s*(from|import)\s", line))
            if re.search(r"\bdisplay\(", body):
                assert "result" in _tags(cell), f"cell {index} displays a result without the result tag"
        assert set(_tags(cell)) <= {"result", "interpretation"}, f"cell {index}: unknown tag"


def test_matched_premium_figure_interpretation_makes_no_product_attribution() -> None:
    """Regression: the sign spread of matched-pair premiums must not be attributed to product identity."""
    cells = read_notebook(MATCHED_PRICING_NOTEBOOK).cells
    plots = [i for i, c in enumerate(cells)
             if c.cell_type == "code" and "plot_matched_location_premiums(" in c.source]
    assert len(plots) == 1, "exactly one result cell renders the matched-premium figure"
    index = plots[0]
    assert "result" in _tags(cells[index])
    assert index + 1 < len(cells) and "interpretation" in _tags(cells[index + 1])
    text = " ".join(cells[index + 1].source.split())
    lowered = text.lower()
    assert len(text.split()) <= MAX_INTERPRETATION_WORDS
    # What the figure shows: the observed premium sign varies across matched pairs.
    assert re.search(r"premium sign varies across matched pairs|sign of the (observed )?premium varies across "
                     r"matched pairs", lowered)
    # Unresolved contributors: repeated measurements plus capture or rental context.
    assert "repeated measurement" in lowered
    assert "capture timing" in lowered or "rental context" in lowered
    assert "descriptive" in lowered or "associational" in lowered
    # No product attribution: the reviewed phrase is gone, and any sentence linking the variation to the product
    # does so only to deny it.
    assert "premium depends on the product" not in lowered
    attributing = re.compile(r"\b(product|products|product identity)\b[^.]*\b(cause[sd]?|explains?|drives?|"
                             r"depends?|determines?|due to)\b|\b(caused|explained|driven|determined) by "
                             r"(the )?product")
    for sentence in re.split(r"(?<=[.;])\s+", lowered):
        if attributing.search(sentence):
            assert re.search(r"\b(not|no|never|cannot)\b", sentence), f"unsupported product attribution: {sentence}"
