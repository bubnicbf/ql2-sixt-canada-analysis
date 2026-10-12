"""Data-plan Section 7: the final report notebook, its package module and the final documentation.

Synthetic data only: the ready pipeline run is fabricated through the real
schedule, readiness and canonical-offer objects (as in the price-change,
assortment and monitoring tests), and notebook execution uses synthetic CSVs
generated in ``tmp_path``. The proprietary files are never read.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from conftest import contract_columns, write_synthetic_csv

from ql2_sixt_canada_analysis import final_report as fr
from ql2_sixt_canada_analysis import paths
from ql2_sixt_canada_analysis.authority_decisions import CURRENT_RECORD_PATH, load_current_decision_record
from ql2_sixt_canada_analysis.data_dictionary import (
    RAW_FIELD_NOTES,
    DataDictionaryError,
    RawFieldNote,
    field_key_role,
    render_generated_sections,
    replace_generated_sections,
    validate_raw_field_notes,
)
from ql2_sixt_canada_analysis.notebook_validation import execute_notebook_copy, read_notebook
from ql2_sixt_canada_analysis.schemas import (
    ANALYSIS_DATASET_DEFINITIONS,
    DATASET_DEFINITIONS,
    JOB_DETAIL_RELATIONSHIP,
    DatasetKey,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FINAL_NOTEBOOK = NOTEBOOKS_DIR / "06_final_report.ipynb"
ASSUMPTIONS_DOC = PROJECT_ROOT / "docs" / "assumptions_exclusions_and_open_questions.md"
DICTIONARY_DOC = PROJECT_ROOT / "docs" / "data_dictionary.md"
README = PROJECT_ROOT / "README.md"
NOTEBOOKS_README = NOTEBOOKS_DIR / "README.md"
DOCUMENTS = (ASSUMPTIONS_DOC, DICTIONARY_DOC, README, NOTEBOOKS_README)

FINAL_SECTIONS = ("# 06 — Final report", "## Purpose and analytical questions", "## Scope and confidentiality",
                  "## Data and pipeline readiness", "## Matched-location pricing findings",
                  "## Price-change findings", "## Visible-assortment findings", "## Monitoring and actionability",
                  "## Assumptions", "## Exclusions", "## Limitations", "## Unanswered questions",
                  "## Requested additional data", "## Final conclusions", "## Data-plan Section 7 reconciliation")

#: Claims the final interpretation must never make (affirmative wording only).
FORBIDDEN_CLAIMS = re.compile(
    r"\b(proves?|proven|caused by|causes|is intentional|intentionally repriced|supplier (removed|withdrew)|"
    r"availability changed|collection failed|alert raised|statistically significant|confirmed alias|"
    r"persisted beyond)\b", re.IGNORECASE)


def _code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [c for c in notebook.cells if c.cell_type == "code"]


def _snapshot(root: Path) -> dict[str, float]:
    return {p.relative_to(root).as_posix(): p.stat().st_mtime for p in root.rglob("*")
            if p.is_file() and not {"__pycache__", ".git", ".pytest_cache"} & set(p.parts)
            and ".egg-info" not in "".join(p.parts)}


# ============================================================================ synthetic pipeline runs


def ready_run():  # type: ignore[no-untyped-def]
    """A bound, pricing-ready synthetic run on which every section completes."""
    from test_monitoring import full_run, stability_report
    from test_price_change_events import synthetic_world
    from test_price_change_presentation import RICH

    world = synthetic_world(products=RICH)
    return full_run(world, vehicle_stability=stability_report(
        observations_assessed=world["population"].eligible_detail_rows))


@pytest.fixture(scope="module")
def ready_report() -> fr.FinalReportResult:
    return fr.final_report_from_pipeline(ready_run())


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "synthetic_raw"
    directory.mkdir()
    for key in DatasetKey:
        write_synthetic_csv(directory / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    return directory


# ============================================================================ final_report module


def test_ready_run_completes_every_section(ready_report: fr.FinalReportResult) -> None:
    assert ready_report.completed and ready_report.blockers == ()
    assert ready_report.pricing_ready and ready_report.evidence_bound
    assert all(s.findings_valid and s.blockers == () for s in ready_report.sections)
    table = fr.final_section_table(ready_report)
    assert tuple(table.columns) == fr.FINAL_SECTION_TABLE_COLUMNS
    assert table["section"].tolist() == [s.value for s in fr.ReportSection]
    assert table["findings_valid"].tolist() == [True] * len(fr.ReportSection)


def test_every_section_uses_the_same_single_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from ql2_sixt_canada_analysis import pricing_pipeline
    from ql2_sixt_canada_analysis.pricing_population import frame_binding

    run = ready_run()
    calls: list[object] = []
    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline", lambda raw_dir=None: calls.append(raw_dir) or run)
    report = fr.run_final_report("synthetic")
    assert calls == ["synthetic"], "the pipeline runs exactly once"
    binding = frame_binding(run.jobs, run.cars)
    assert report.price_changes.analysis.events.binding == binding
    assert report.assortment.assortment.binding == binding
    assert report.assortment.assortment.location_authority is run.location_authority
    assert report.price_changes.analysis.location_authority is run.location_authority
    assert report.monitoring.evidence.pricing is run.pricing


def test_unbound_evidence_produces_no_commercial_section() -> None:
    run = ready_run()
    tampered = dataclasses.replace(run, scheduled=dataclasses.replace(run.scheduled))   # evidence kept from before
    tampered = dataclasses.replace(tampered, evidence=run.evidence)
    report = fr.final_report_from_pipeline(tampered)
    assert report.blockers == (fr.FinalReportBlocker.EVIDENCE_BINDING_MISMATCH,)
    assert report.matched is None and report.price_changes is None and report.assortment is None
    assert report.monitoring.report.blocked
    assert not any(report.findings_valid(s) for s in fr.ReportSection)
    for section in fr.ReportSection:
        text = fr.interpret_section(report, section)
        assert "evidence_binding_mismatch" in text and not re.search(r"\d", text)


def test_missing_readiness_evidence_fails_closed() -> None:
    report = fr.final_report_from_pipeline(dataclasses.replace(ready_run(), pricing=None))
    assert report.status is fr.FinalReportStatus.BLOCKED
    assert report.blockers == (fr.FinalReportBlocker.PIPELINE_EVIDENCE_UNAVAILABLE,)
    table = fr.final_section_table(report)
    assert set(table["blockers"]) == {"pipeline_evidence_unavailable"} and not table["findings_valid"].any()
    with pytest.raises(fr.FinalReportError):
        fr.matched_summary_table(report)
    with pytest.raises(fr.FinalReportError):
        fr.matched_premium_png(report)


def test_a_blocked_analysis_is_reported_with_its_own_categories() -> None:
    from test_monitoring import full_run
    from test_price_change_events import synthetic_world
    from test_price_change_presentation import RICH

    report = fr.final_report_from_pipeline(full_run(synthetic_world(products=RICH)))   # stability population differs
    assert fr.FinalReportBlocker.MATCHED_PRICING_BLOCKED in report.blockers
    outcome = report.section(fr.ReportSection.MATCHED_PRICING)
    assert outcome.status == "blocked" and outcome.blockers == ("vehicle_stability_population_mismatch",)
    text = fr.interpret_section(report, fr.ReportSection.MATCHED_PRICING)
    assert text.startswith("Blocked (vehicle_stability_population_mismatch)")
    assert "no airport-versus-downtown premium can be stated" in text
    assert report.findings_valid(fr.ReportSection.PRICE_CHANGES)


def test_interpretations_are_deterministic_qualified_and_sanitized(ready_report: fr.FinalReportResult) -> None:
    texts = [fr.interpret_section(ready_report, s) for s in fr.ReportSection] + list(
        fr.final_conclusions(ready_report))
    assert texts == [fr.interpret_section(ready_report, s) for s in fr.ReportSection] + list(
        fr.final_conclusions(ready_report))
    joined = " ".join(texts)
    assert not FORBIDDEN_CLAIMS.search(joined)
    assert "SYNTH" not in joined and "Thurlow" not in joined and "/" not in joined.replace("/day", "")
    assert "not causal" in joined and "not proof of intentional repricing" in joined
    assert "not proof of supplier availability" in joined and "not production alerts" in joined
    assert "right-censored" in joined and "excluded from persistence conclusions" in joined
    assert len(fr.final_conclusions(ready_report)) == 4


def test_section_table_validation_refuses_anything_else(ready_report: fr.FinalReportResult) -> None:
    table = fr.final_section_table(ready_report)
    for change in ({"blockers": ["Value 1", "", "", "", ""]}, {"status": ["passed"] * 5},
                   {"findings_valid": ["yes"] * 5}):
        with pytest.raises(fr.FinalReportError):
            fr.validate_final_section_table(table.assign(**change))
    with pytest.raises(fr.FinalReportError):
        fr.validate_final_section_table(table.iloc[::-1].reset_index(drop=True))
    with pytest.raises(fr.FinalReportError):
        fr.validate_final_section_table(table.assign(extra=1))
    with pytest.raises(fr.FinalReportError):
        fr.SectionOutcome(fr.ReportSection.MATCHED_PRICING, "completed", True, ("pricing_not_ready",))


def test_result_invariants_and_types(ready_report: fr.FinalReportResult) -> None:
    with pytest.raises(TypeError):
        fr.final_report_from_pipeline(object())
    with pytest.raises(fr.FinalReportError):
        dataclasses.replace(ready_report, blockers=(fr.FinalReportBlocker.PRICING_NOT_READY,))
    with pytest.raises(fr.FinalReportError):
        dataclasses.replace(ready_report, evidence_bound=False)
    with pytest.raises(TypeError):
        fr.final_section_table(object())  # type: ignore[arg-type]


def test_displayed_tables_are_sanitized(ready_report: fr.FinalReportResult) -> None:
    from ql2_sixt_canada_analysis.monitoring import MONITORING_TABLE_COLUMNS

    summary = fr.matched_summary_table(ready_report)
    assert not {"car_name", "car_type", "job_id", "price_num", "pickup_date"} & set(summary.columns)
    overview = fr.monitoring_overview_table(ready_report)
    assert set(overview.columns) <= set(MONITORING_TABLE_COLUMNS) and len(overview) == 8
    assert fr.matched_premium_png(ready_report)[:4] == b"\x89PNG"
    for frame in (summary, overview, fr.open_questions_table(), fr.data_requests_table()):
        assert "SYNTH" not in frame.to_csv(index=False)


def test_matched_pricing_from_pipeline_equals_the_run_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    from ql2_sixt_canada_analysis import matched_location_pricing as mlp
    from ql2_sixt_canada_analysis import pricing_pipeline

    run = ready_run()
    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline", lambda raw_dir=None: run)
    assert mlp.run_matched_location_pricing().report == mlp.matched_location_pricing_from_pipeline(run).report
    with pytest.raises(TypeError):
        mlp.matched_location_pricing_from_pipeline(object())


def test_importing_the_new_modules_performs_no_data_or_document_access(tmp_path: Path) -> None:
    import subprocess
    import sys

    watched = [str((PROJECT_ROOT / name).resolve()) for name in ("data", "docs", "notebooks", "reports")]
    script = f"""
import json, os, sys
watched = {watched!r}
events = []
def touched(t):
    if not isinstance(t, (str, bytes, os.PathLike)):
        return False
    p = os.path.realpath(os.fsdecode(t))
    return any(p == w or p.startswith(w + os.sep) for w in watched)
def hook(event, args):
    if event in ("open", "os.listdir", "os.scandir") and args and touched(args[0]):
        events.append(event)
    if event in ("os.mkdir", "os.makedirs", "subprocess.Popen"):
        events.append(event)
sys.addaudithook(hook)
import ql2_sixt_canada_analysis.final_report
import ql2_sixt_canada_analysis.data_dictionary
print(json.dumps(events))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True,
                            check=True)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


# ============================================================================ the final notebook


def test_final_notebook_has_the_next_numeric_prefix_and_is_listed_in_order() -> None:
    notebooks = sorted(p.name for p in NOTEBOOKS_DIR.glob("*.ipynb"))
    prefixes = [int(n[:2]) for n in notebooks]
    assert FINAL_NOTEBOOK.name == notebooks[-1] and prefixes == list(range(1, len(notebooks) + 1))
    for readme, anchor in ((NOTEBOOKS_README, "## Execution order"), (README, "### 3. Run the notebooks")):
        text = readme.read_text(encoding="utf-8")
        text = text[text.index(anchor):]
        positions = [text.find(n) for n in notebooks]
        assert all(p >= 0 for p in positions) and positions == sorted(positions), readme.name


def test_final_notebook_is_clean_portable_nbformat_4() -> None:
    raw = json.loads(FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    assert raw["nbformat"] == 4
    notebook = read_notebook(FINAL_NOTEBOOK)
    assert "widgets" not in notebook.metadata and set(notebook.metadata) <= {"kernelspec", "language_info"}
    for cell in _code_cells(notebook):
        assert cell.outputs == [] and cell.execution_count is None and "execution" not in cell.metadata
    text = FINAL_NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"(?<![\w\"])/(?:Users|home|sessions|tmp|mnt|private|var)/", text)
    assert not re.search(r"[A-Za-z]:\\\\", text) and "image/png" not in text and "base64" not in text


def test_final_notebook_sections_are_in_the_required_order() -> None:
    notebook = read_notebook(FINAL_NOTEBOOK)
    headings = [c.source.splitlines()[0] for c in notebook.cells if c.cell_type == "markdown"]
    positions = [headings.index(h) for h in FINAL_SECTIONS]
    assert positions == sorted(positions)
    record = load_current_decision_record()
    text = " ".join("\n".join(c.source for c in notebook.cells if c.cell_type == "markdown").split())
    assert f"record v{record.record_version}" in text
    for phrase in ("exactly once", "roughly 90 hours", "not supplier availability", "confirmation-required",
                   "analytically null", "completely blank rows are removed and counted", "not an approved threshold",
                   "Decisions already resolved in record v"):
        assert phrase in text, phrase


def test_final_notebook_is_a_thin_package_layer() -> None:
    code = "\n".join(c.source for c in _code_cells(read_notebook(FINAL_NOTEBOOK)))
    for name in ("run_final_report", "resolve_raw_data_dir", "interpret_section", "final_conclusions",
                 "final_section_table", "open_questions_table", "data_requests_table"):
        assert f"{name}(" in code, name
    assert code.count("run_final_report(") == 1
    for pattern in (r"read_csv", r"\.csv\b", r"os\.chdir", r"sys\.path", r"(?m)^\s*[!%]", r"\bopen\(",
                    r"\.to_(csv|parquet|json|pickle|feather)\(", r"savefig", r"output_dir", r"OUTPUT_DIR",
                    r"run_pricing_pipeline", r"_from_pipeline", r"\bdef\b", r"\blambda\b", r"\.merge\(",
                    r"\.groupby\(", r"\.(head|tail|sample|to_string|to_markdown|to_html)\(", r"\.pairs\b",
                    r"\.offers\b", r"\.candidates\b", r"\.membership\b", r"\.evidence\b", r"job_id", r"\bjobs\b",
                    r"\bcars\b", r"write_detail", r"__import__", r"data/raw"):
        assert not re.search(pattern, code), f"notebook re-implements or exposes: {pattern}"


def _exec_final_cells(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: object) -> tuple:
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
    for cell in _code_cells(read_notebook(FINAL_NOTEBOOK)):
        exec(compile(cell.source, "<notebook-cell>", "exec"), namespace)   # noqa: S102 - the committed cells
    return calls, shown, "\n".join(printed)


def test_final_notebook_runs_top_to_bottom_on_ready_synthetic_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ql2_sixt_canada_analysis.assortment_presentation import validate_assortment_table
    from ql2_sixt_canada_analysis.price_change_presentation import validate_sanitized_frame

    repo_before = _snapshot(PROJECT_ROOT)
    calls, shown, text = _exec_final_cells(monkeypatch, tmp_path, ready_run())
    assert len(calls) == 1
    assert "Final report status: completed" in text and "No files written." in text
    assert "Skipped" not in text and "Blocked" not in text
    assert "SYNTH" not in text and str(tmp_path) not in text and not FORBIDDEN_CLAIMS.search(text)
    frames = [o for o in shown if isinstance(o, pd.DataFrame)]
    images = [o for o in shown if type(o).__name__ == "Image"]
    assert len(images) == 2 and all(i.data[:4] == b"\x89PNG" for i in images)
    fr.validate_final_section_table(frames[0])
    validate_sanitized_frame("persistence_summary", frames[2])
    validate_assortment_table("location_summary", frames[3])
    validate_assortment_table("cross_location_drops", frames[4])
    assert len(frames) == 8 and all("SYNTH" not in f.to_csv(index=False) for f in frames)
    assert os.listdir(tmp_path) == [] and _snapshot(PROJECT_ROOT) == repo_before


def test_final_notebook_shows_only_categories_for_unbound_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls, shown, text = _exec_final_cells(monkeypatch, tmp_path, dataclasses.replace(ready_run(), pricing=None))
    assert len(calls) == 1 and "Final report status: blocked" in text
    assert text.count("Skipped:") == 3 and "pipeline_evidence_unavailable" in text
    assert not re.search(r"\d", text) and "SYNTH" not in text
    assert not any(type(o).__name__ == "Image" for o in shown)
    frames = [o for o in shown if isinstance(o, pd.DataFrame)]
    assert set(frames[1]["evaluation_status"]) == {"not_assessable"}


def test_final_notebook_fails_closed_from_a_clean_kernel_outside_the_repository(
        synthetic_raw_dir: Path, tmp_path: Path) -> None:
    from ql2_sixt_canada_analysis.readiness import PricingBlocker

    repo_before = _snapshot(PROJECT_ROOT)
    source_bytes = FINAL_NOTEBOOK.read_bytes()
    workdir = tmp_path / "outside_repository"
    workdir.mkdir()
    result = execute_notebook_copy(FINAL_NOTEBOOK, workdir=workdir,
                                   env={paths.RAW_DATA_DIR_ENV_VAR: str(synthetic_raw_dir)}, timeout_seconds=600)
    cells = _code_cells(result.executed)
    assert result.execution_counts == tuple(range(1, len(cells) + 1))
    outputs = "\n".join(o.get("text", "") for c in cells for o in c.outputs)
    rendered = json.dumps([o for c in cells for o in c.outputs])
    assert "Final report status: blocked" in outputs and "Pricing readiness: blocked" in outputs
    assert outputs.count("Skipped:") == 3 and "No files written." in outputs
    assert "no matched-location, price-change or visible-assortment finding is valid" in outputs.lower()
    # Only allowed blocker categories are revealed (snake_case codes from the typed enums).
    blockers = re.search(r"Pricing-readiness blockers: (.*)", outputs).group(1).split(", ")
    assert blockers and set(blockers) <= {b.value for b in PricingBlocker}
    assert "image/png" not in rendered and "synthetic_" not in rendered
    assert str(synthetic_raw_dir) not in rendered and str(tmp_path) not in rendered
    assert not any(column in rendered for key in DatasetKey for column in contract_columns(key)
                   if column not in {"city", "status", "mode", "location"})
    assert FINAL_NOTEBOOK.read_bytes() == source_bytes, "tracked notebook was modified"
    assert not any(workdir.iterdir()) and _snapshot(PROJECT_ROOT) == repo_before


# ============================================================================ assumptions and open questions


def _section(text: str, heading: str) -> str:
    start = text.index(f"\n{heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end >= 0 else None]


ASSUMPTION_SECTIONS = ("## Approved external decisions", "## Analytical assumptions", "## Data limitations",
                       "## Governed exclusions", "## Mechanical quality exclusions",
                       "## Ambiguous and unassessable evidence", "## Presentation and privacy exclusions",
                       "## Open analytical questions", "## Additional data requests")


def test_assumptions_document_has_the_distinct_sections() -> None:
    text = ASSUMPTIONS_DOC.read_text(encoding="utf-8")
    positions = [text.index(f"\n{h}\n") for h in ASSUMPTION_SECTIONS]
    assert positions == sorted(positions)
    for row in _section(text, "## Analytical assumptions").splitlines()[4:]:
        if row.startswith("| "):
            assert row.count(" | ") >= 2 and all(cell.strip() for cell in row.strip("|").split(" | ")), row


def test_approved_decisions_match_the_current_record_and_are_not_reopened() -> None:
    record = load_current_decision_record()
    assert record is not None and CURRENT_RECORD_PATH.name == f"v{record.record_version}.toml"
    text = ASSUMPTIONS_DOC.read_text(encoding="utf-8")
    approved = _section(text, "## Approved external decisions")
    assert record.record_id in approved and CURRENT_RECORD_PATH.as_posix() in approved
    ids = [str(getattr(d.id, 'value', d.id)) for d in record.decisions]
    assert all(d.is_approved for d in record.decisions)
    for decision in ids:
        assert f"`{decision}`" in approved, decision
    open_part = _section(text, "## Open analytical questions") + _section(text, "## Additional data requests")
    assert not any(decision in open_part for decision in ids), "a resolved v8 decision is relisted as open"
    catalogs = " ".join(q.question + q.why_it_matters for q in fr.OPEN_QUESTIONS)
    assert not any(decision in catalogs for decision in ids)


def test_governed_exclusions_and_censoring_are_described_accurately() -> None:
    text = " ".join(ASSUMPTIONS_DOC.read_text(encoding="utf-8").split())
    governed = " ".join(_section(ASSUMPTIONS_DOC.read_text(encoding="utf-8"), "## Governed exclusions").split())
    for phrase in ("**stay** in the raw and audit populations", "**excluded from the pricing-eligible population**",
                   "analytically null", "not a general rule", "never a job identifier"):
        assert phrase in governed, phrase
    for phrase in ("Completely blank rows", "Partially populated rows are retained",
                   "never becomes infinity or an invented zero", "`confirmation_required`",
                   "excluded from persistence conclusions", "neither persistent nor disproven",
                   "not proof of supplier availability", "roughly 90 hourly captures",
                   "no scheduler, persistence, alerting, notification or dashboard"):
        assert phrase in text, phrase


def test_open_questions_and_requests_mirror_the_catalogs() -> None:
    text = ASSUMPTIONS_DOC.read_text(encoding="utf-8")
    assert replace_generated_sections(text, fr.render_question_catalogs()) == text, "regenerate the catalogs"
    for item in fr.OPEN_QUESTIONS:
        assert f"`{item.question_id}`" in _section(text, "## Open analytical questions")
    for item in fr.DATA_REQUESTS:
        assert f"`{item.request_id}`" in _section(text, "## Additional data requests")
    ids = [q.question_id for q in fr.OPEN_QUESTIONS] + [d.request_id for d in fr.DATA_REQUESTS]
    assert len(ids) == len(set(ids)) and all(re.fullmatch(r"[a-z_]+", i) for i in ids)
    owners = {q.likely_owner for q in fr.OPEN_QUESTIONS} | {d.likely_owner for d in fr.DATA_REQUESTS}
    assert all(re.match(r"(business owner|collection owner|not recorded)", o) for o in owners)
    required = ("longer_history", "next_eligible_collections", "collection_logs_and_capture_completeness",
                "source_snapshots_or_supplier_evidence", "labelled_incidents_and_normal_periods")
    assert set(required) <= {d.request_id for d in fr.DATA_REQUESTS}
    assert {"unusual_assortment_drop_policy", "synchronized_movement_policy",
            "production_monitoring_operations"} <= {q.question_id for q in fr.OPEN_QUESTIONS}


# ============================================================================ data dictionary


def test_dictionary_generated_blocks_are_current() -> None:
    text = DICTIONARY_DOC.read_text(encoding="utf-8")
    assert replace_generated_sections(text) == text, "run python -m ql2_sixt_canada_analysis.data_dictionary"


def test_raw_field_notes_cover_the_schema_exactly() -> None:
    validate_raw_field_notes()
    for key, definition in DATASET_DEFINITIONS.items():
        assert tuple(n.field for n in RAW_FIELD_NOTES if n.dataset is key) == definition.columns
    with pytest.raises(DataDictionaryError):
        validate_raw_field_notes(RAW_FIELD_NOTES[1:])
    with pytest.raises(DataDictionaryError):
        validate_raw_field_notes((*RAW_FIELD_NOTES, RAW_FIELD_NOTES[0]))
    extra = RawFieldNote(DatasetKey.JOBS, "unknown_field", "m", "r", "h", RAW_FIELD_NOTES[0].use)
    with pytest.raises(DataDictionaryError):
        validate_raw_field_notes((*RAW_FIELD_NOTES, extra))


def test_every_raw_field_appears_once_per_dataset_in_the_document() -> None:
    text = DICTIONARY_DOC.read_text(encoding="utf-8")
    for key, definition in DATASET_DEFINITIONS.items():
        section = _section(text, "## Raw fields").split(f"### `{key.value}`")[1].split("###")[0]
        rows = [r for r in section.splitlines() if re.match(r"\| \d+ \| `", r)]
        fields = [re.match(r"\| \d+ \| `([^`]+)`", r).group(1) for r in rows]
        assert fields == list(definition.columns), key
        for row, column in zip(rows, definition.columns):
            identifier = "yes (nullable string)" if column in definition.identifier_columns else "| no |"
            assert identifier in row and field_key_role(key, column) in row


def test_dictionary_states_grains_keys_and_the_relationship() -> None:
    text = DICTIONARY_DOC.read_text(encoding="utf-8")
    assert "One row per scrape (collection) job." in text
    assert "One row per offer position within one scrape job's result list." in text
    rel = JOB_DETAIL_RELATIONSHIP
    for definitions in (DATASET_DEFINITIONS, ANALYSIS_DATASET_DEFINITIONS):
        for definition in definitions.values():
            assert ", ".join(f"`{c}`" for c in definition.unique_key_columns) in text
    assert "Parent: one job has zero, one or many `cars` rows." in text
    assert "Detail: every row links to exactly one `jobs` row." in text
    assert f"`{rel.expected_detail_count_column}`" in text
    assert all(f"`{c}`" in text for c in rel.additional_expected_count_columns)
    assert text.count("One row per") >= 20, "derived and aggregate tables state their grain"


REQUIRED_DERIVED_FIELDS = ("job_id_linkage_key", "row_index_key", "canonical_city", "canonical_location",
                           "scheduled_capture_period", "reporting_day", "rental_duration_days", "price_cents",
                           "currency", "price_basis", "premium_cents", "premium_percent", "premium_dollars",
                           "change_cents", "change_percent", "movement_class", "persistence", "not_testable_reason",
                           "returned_product_count", "retention", "jaccard_similarity", "net_change",
                           "absolute_drop", "drop_rate", "addition_count", "removal_count", "evaluation_status",
                           "severity", "findings", "unavailable_evidence", "notes", "findings_valid")
REQUIRED_FORMULAS = ("`airport_price_cents - downtown_price_cents`", "`100 * premium_cents / downtown_price_cents`",
                     "`n(P ∩ C) / n(P)`", "`n(P ∩ C) / n(P ∪ C)`", "`n(C) − n(P)`", "`max(n(P) − n(C), 0)`",
                     "`absolute_drop / n(P)`", "`current_price_cents - previous_price_cents`",
                     "`100 * change_cents / previous_price_cents`", "`matched_pairs / candidate_groups`")


def test_derived_fields_formulas_and_table_schemas_are_documented() -> None:
    from ql2_sixt_canada_analysis.assortment_presentation import ASSORTMENT_TABLE_SCHEMAS
    from ql2_sixt_canada_analysis.canonical_offers import APPROVED_PRODUCT_COLUMNS
    from ql2_sixt_canada_analysis.price_change_presentation import SANITIZED_TABLE_SCHEMAS

    text = DICTIONARY_DOC.read_text(encoding="utf-8")
    for name in REQUIRED_DERIVED_FIELDS:
        assert f"`{name}`" in text, name
    for formula in REQUIRED_FORMULAS:
        assert formula in text, formula
    for table in (*SANITIZED_TABLE_SCHEMAS, *ASSORTMENT_TABLE_SCHEMAS, "monitoring_control_table",
                  "final_section_table", "match_count_frame", "city_summary_frame"):
        assert f"`{table}`" in text, table
    assert all(f"`{c}`" in text for c in APPROVED_PRODUCT_COLUMNS)
    for kind in ("Raw value", "Derived value", "Configuration / authority value", "Presentation-only value"):
        assert kind in text


# ============================================================================ READMEs and confidentiality


def test_readmes_contain_the_execution_instructions() -> None:
    readme = README.read_text(encoding="utf-8")
    for phrase in ("Python **3.11 or newer**", "python3 -m venv .venv", "source .venv/bin/activate",
                   'python -m pip install -e ".[dev]"', "data/raw/", "Never edit, rename, reformat, commit or "
                   "publish these files", paths.RAW_DATA_DIR_ENV_VAR, "top to bottom after restarting the kernel",
                   "independent", "06_final_report.ipynb", "execute_notebook_copy",
                   "python -m pytest tests/test_final_documentation.py", "python -m pytest tests/test_notebooks.py",
                   "python -m pytest\n", "git diff --check", "python -m ql2_sixt_canada_analysis.matched_location_pricing",
                   "Git-ignored", "python -m ql2_sixt_canada_analysis.notebook_validation --clear notebooks/*.ipynb",
                   "Expected fail-closed behaviour", "Troubleshooting", "No such kernel named python3",
                   "requires a different Python", "Every section prints `blocked`"):
        assert phrase in readme, phrase
    notebooks = NOTEBOOKS_README.read_text(encoding="utf-8")
    for phrase in ("top to bottom after\nrestarting the kernel", "independent", "06_final_report.ipynb",
                   "execute_notebook_copy", paths.RAW_DATA_DIR_ENV_VAR, "`result`", "`interpretation`",
                   "python -m ql2_sixt_canada_analysis.notebook_validation --clear", "python -m pytest tests/test_notebooks.py"):
        assert phrase in notebooks, phrase


def test_documented_commands_name_real_entry_points() -> None:
    import importlib

    text = README.read_text(encoding="utf-8") + NOTEBOOKS_README.read_text(encoding="utf-8")
    for module in set(re.findall(r"python -m (ql2_sixt_canada_analysis\.\w+)", text)):
        assert hasattr(importlib.import_module(module), "main"), module
    for test_file in set(re.findall(r"python -m pytest ((?:tests/\S+\.py ?)+)", text)):
        for name in test_file.split():
            assert (PROJECT_ROOT / name).is_file(), name
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11"' in pyproject and "dev = [" in pyproject


@pytest.mark.parametrize("document", DOCUMENTS + (FINAL_NOTEBOOK,), ids=lambda p: p.name)
def test_documentation_holds_no_paths_filenames_or_real_values(document: Path) -> None:
    text = document.read_text(encoding="utf-8")
    if document.suffix == ".ipynb":
        text = "\n".join(c.source for c in read_notebook(document).cells)
    assert not re.search(r"(?<![\w\"])/(?:Users|home|sessions|tmp|mnt|private)/", text), "absolute path"
    assert not re.search(r"\b[A-Za-z]:\\[A-Za-z]", text), "Windows path"
    assert not re.search(r"\w+_(jobs|cars)_raw\b|\(1\)\.csv", text, re.IGNORECASE), "source file name"
    if document in (ASSUMPTIONS_DOC, DICTIONARY_DOC, FINAL_NOTEBOOK):
        # The new Section 7 documents name no governed period, price, timestamp or synthetic value either
        # (the older README sections quote the approved governance configuration itself).
        assert not re.search(r"\b\d{8}T\d{6}Z\b", text), "a scheduled period value"
        assert not re.search(r"\$\s?\d", text), "a price"
        assert not re.search(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", text), "a timestamp"
        assert "SYNTH" not in text


def test_clear_notebook_state_removes_session_state_and_keeps_tags(tmp_path: Path) -> None:
    from ql2_sixt_canada_analysis.notebook_validation import clear_notebook_state, main

    notebook = read_notebook(FINAL_NOTEBOOK)
    notebook.metadata["widgets"] = {"application/vnd.jupyter.widget-state+json": {"state": {}}}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            cell.execution_count = index
            cell.outputs = [nbformat.v4.new_output("stream", text="value")]
            cell.metadata["execution"] = {"iopub.status.busy": "2030-01-01T00:00:00.000000Z"}
    dirty = tmp_path / "dirty.ipynb"
    nbformat.write(notebook, dirty)
    assert clear_notebook_state(dirty) is True
    cleaned = read_notebook(dirty)
    tracked = read_notebook(FINAL_NOTEBOOK)
    assert cleaned.metadata == tracked.metadata
    assert [(c.source, c.metadata, c.get("outputs"), c.get("execution_count")) for c in cleaned.cells] == \
        [(c.source, c.metadata, c.get("outputs"), c.get("execution_count")) for c in tracked.cells]
    assert clear_notebook_state(dirty) is False                      # idempotent: a clean file is not rewritten
    assert main(["--clear", str(dirty)]) == 0
