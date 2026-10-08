"""Price-change presentation: sanitized aggregate tables, heatmap, local detail export and orchestration.

Every observation is fabricated (``SYNTH-*`` jobs and products, synthetic 2030
capture periods and prices). The only committed values read are approved
configuration (source-stream keys, time zones, location roles and pairs, and the
canonical-offer policy of the current authority record).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_price_change_analysis import analyze, merge, path
from test_price_change_events import (
    CAL_AIR,
    CAL_DOWN,
    PREV,
    CUR,
    TOR_AIR,
    TOR_DOWN,
    VAN_DOWN,
    VAN_THUR,
    pipeline_result,
    synthetic_world,
)

import ql2_sixt_canada_analysis as package
from ql2_sixt_canada_analysis import price_change_presentation as pcp
from ql2_sixt_canada_analysis.price_change_analysis import PriceChangeReconciliationError
from ql2_sixt_canada_analysis.price_change_events import CANDIDATE_COLUMNS, EVENT_KEY_COLUMNS
from ql2_sixt_canada_analysis.price_change_presentation import (
    DETAIL_EXPORT_ENV_VAR,
    DETAILED_EVENT_TABLE_COLUMNS,
    DETAILED_EVENT_TABLE_FILENAME,
    FORBIDDEN_SANITIZED_COLUMNS,
    HEATMAP_FILENAME,
    MANIFEST_FILENAME,
    PRESENTATION_OUTPUT_DIR_ENV_VAR,
    SANITIZED_COLUMN_ALLOWLIST,
    SANITIZED_TABLE_SCHEMAS,
    PresentationTables,
    PriceChangePresentationBlocker as PB,
    PriceChangePresentationResult,
    PrivacyViolationError,
    build_detailed_event_table,
    build_presentation_tables,
    export_sanitized_tables,
    heatmap_png,
    heatmap_source_frame,
    presentation_from_pipeline,
    presentation_settings,
    render_price_change_heatmap,
    run_price_change_presentation,
    validate_sanitized_frame,
    write_detailed_event_table,
)

ROOT = Path(__file__).resolve().parents[1]
RICH = merge(path(TOR_DOWN, (50.0, 55.0, 60.0)), path(TOR_DOWN, (80.0, 88.0, 88.0), "SYNTH Car B"),
             path(TOR_AIR, (50.0, 55.0, 55.0)), path(TOR_AIR, (90.0, 85.0, None), "SYNTH Car C"),
             path(VAN_DOWN, (60.0, 60.0, 54.0)), path(VAN_THUR, (60.0, 60.0, 54.0)),
             path(VAN_DOWN, (70.0, 70.0, 63.0), "SYNTH Car B"))


@pytest.fixture(scope="module")
def rich():  # type: ignore[no-untyped-def]
    return analyze(RICH)


@pytest.fixture(scope="module")
def tables(rich):  # type: ignore[no-untyped-def]
    return build_presentation_tables(rich)


def all_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False)


# ============================================================================ sanitized schemas and privacy


def test_sanitized_tables_use_the_exact_allowlisted_schemas(tables) -> None:  # type: ignore[no-untyped-def]
    for name, frame in tables.items():
        assert tuple(frame.columns) == SANITIZED_TABLE_SCHEMAS[name], name
        assert set(frame.columns) <= SANITIZED_COLUMN_ALLOWLIST
        for forbidden in FORBIDDEN_SANITIZED_COLUMNS.values():
            assert not set(frame.columns) & forbidden
    assert [n for n, _ in tables.items()] == list(SANITIZED_TABLE_SCHEMAS)


@pytest.mark.parametrize(("column", "rule"), [
    ("car_name", "product_identity"), ("pickup_date", "product_identity"), ("seats", "product_identity"),
    ("previous_price_cents", "individual_price"), ("current_price", "individual_price"),
    ("change_percent", "individual_price"), ("previous_source_labels", "source_provenance"),
    ("source_location_labels", "source_provenance"), ("job_id", "raw_identifier"), ("row_index", "raw_identifier"),
    ("unexpected_metric", "allowlist")])
def test_forbidden_and_unexpected_columns_are_rejected_without_echoing_values(tables, column, rule) -> None:  # type: ignore[no-untyped-def]
    frame = tables.event_interval_summary.assign(**{column: "SYNTH-SECRET-VALUE"})
    with pytest.raises(PrivacyViolationError) as info:
        validate_sanitized_frame("event_interval_summary", frame)
    assert f"rule {rule}" in str(info.value) and "SYNTH-SECRET-VALUE" not in str(info.value)
    assert column not in str(info.value)


def test_provenance_text_and_detailed_frames_are_rejected(rich, tables) -> None:  # type: ignore[no-untyped-def]
    summary = tables.event_interval_summary.copy()
    summary.loc[0, "interval_flag"] = "Vancouver Downtown|Vancouver Thurlow"
    with pytest.raises(PrivacyViolationError, match="rule source_provenance"):
        validate_sanitized_frame("event_interval_summary", summary)
    with pytest.raises(PrivacyViolationError, match="rule detailed_frame"):
        validate_sanitized_frame("event_interval_summary", rich.events.candidates)
    with pytest.raises(PrivacyViolationError, match="rule detailed_frame"):
        export_sanitized_tables(build_detailed_event_table(rich), "unused")  # type: ignore[arg-type]
    with pytest.raises(PrivacyViolationError, match="rule unknown_table"):
        validate_sanitized_frame("detail", tables.event_interval_summary)
    with pytest.raises(PrivacyViolationError):
        dataclasses.replace(tables, persistence_summary=tables.persistence_summary.assign(car_name="x"))


def test_no_product_values_prices_or_identifiers_in_any_sanitized_table(tables) -> None:  # type: ignore[no-untyped-def]
    text = "".join(all_text(f) for _, f in tables.items())
    assert "SYNTH" not in text and "2030-04" not in text and "job_id" not in text
    assert "|" not in "".join(all_text(f) for n, f in tables.items() if n != "final_vancouver_decrease")
    for price in ("55.0", "88.0", "5500", "8800"):
        assert price not in all_text(tables.event_interval_summary)


def test_single_change_magnitudes_are_suppressed(tables) -> None:  # type: ignore[no-untyped-def]
    s = tables.event_interval_summary
    single = s[s["price_change_count"] == 1]
    assert len(single) and single["magnitude_suppressed"].all()
    assert single[["min_change_cents", "max_change_cents", "median_abs_change_percent"]].isna().all().all()
    multi = s[s["price_change_count"] >= 2]
    assert len(multi) and multi["min_change_cents"].notna().all() and not multi["magnitude_suppressed"].any()


def test_tables_are_deterministic_and_row_order_independent(rich, tables) -> None:  # type: ignore[no-untyped-def]
    again = build_presentation_tables(rich)
    for (name, a), (_, b) in zip(tables.items(), again.items()):
        pd.testing.assert_frame_equal(a, b, obj=name)
    events = rich.events
    shuffled = dataclasses.replace(events, candidates=events.candidates.sample(frac=1.0, random_state=11)
                                   .reset_index(drop=True))
    from ql2_sixt_canada_analysis.price_change_analysis import analyze_price_change_events

    other = build_presentation_tables(analyze_price_change_events(shuffled, location_authority=rich.location_authority))
    for (name, a), (_, b) in zip(tables.items(), other.items()):
        pd.testing.assert_frame_equal(a, b, obj=name)


# ============================================================================ reconciliation


def test_tables_reconcile_to_the_validated_reports(rich, tables) -> None:  # type: ignore[no-untyped-def]
    overall, report = rich.events.report.overall, rich.report
    s = tables.event_interval_summary
    assert int(s["price_change_count"].sum()) == overall.increase + overall.decrease == report.price_change_count
    assert int(s["assortment_event_count"].sum()) == overall.appeared + overall.disappeared
    assert int(s["candidates"].sum()) == overall.candidates and len(s) == overall.intervals
    recon = tables.reconciliation_summary
    assert (recon["status"] == "reconciled").all() and (recon["expected"] == recon["observed"]).all()
    assert {"persistence_partitions_price_changes", "selected_plus_excluded_equal_price_changes",
            "heatmap_increases_equal_interval_summary", "canonical_events_unique_across_aliases"} <= set(recon["check"])


def test_selected_and_excluded_movements_reconcile_to_all_price_changes(rich, tables) -> None:  # type: ignore[no-untyped-def]
    selection, material = tables.material_selection_reconciliation, tables.material_synchronized_movements
    assert int(selection["price_change_count"].sum()) == rich.report.price_change_count
    assert int(selection.loc[selection["selected"], "price_change_count"].sum()) == int(
        material["price_change_count"].sum())
    assert len(material) and (material["price_change_count"] >= 2).all()
    assert set(material["movement_class"]) <= {"synchronized_increase", "synchronized_decrease"}
    assert (material["selection_rule"] == pcp.MATERIAL_SELECTION_RULE).all()
    excluded = selection[~selection["selected"]]
    assert int(excluded["price_change_count"].sum()) > 0                        # isolated movements stay visible


def test_persistence_presentation_partitions_changes_and_excludes_censored_denominators(rich, tables) -> None:  # type: ignore[no-untyped-def]
    p = tables.persistence_summary
    outcomes = ["held", "continued", "reverted", "disappeared", "ambiguous", "not_testable"]
    assert int(p[outcomes].to_numpy().sum()) == int(p["changed_events"].sum()) == rich.report.price_change_count
    assert (p["testable"] == p["changed_events"] - p["not_testable"]).all()
    assert (p["comparable_following"] == p["held"] + p["continued"] + p["reverted"]).all()
    censored = p[p["testable"] == 0]
    assert len(censored) and censored["held_share_of_comparable_following"].isna().all()
    rated = p[p["comparable_following"] > 0].iloc[0]
    assert rated["held_share_of_comparable_following"] == rated["held"] / rated["comparable_following"]
    assert int(p["not_testable_right_censored_final_capture"].sum()) == int(p["not_testable"].sum())


def test_airport_downtown_totals_reconcile_without_duplication(rich, tables) -> None:  # type: ignore[no-untyped-def]
    a = tables.airport_downtown_summary
    for summary in rich.report.cross_location:
        rows = a[a["canonical_city"] == summary.canonical_city]
        assert int(rows["matched_products"].sum()) == summary.matched_products
        assert int(rows["airport_only_products"].sum()) == summary.airport_only_products
        assert int(rows["downtown_only_products"].sum()) == summary.downtown_only_products
    outcome_columns = [c for c in a.columns if c.startswith("cross_")]
    assert (a[outcome_columns].sum(axis=1) == a["matched_products"]).all()
    assert len(a) == sum(s.intervals_compared for s in rich.report.cross_location)


def test_vancouver_alias_never_creates_two_presentation_events(rich, tables) -> None:  # type: ignore[no-untyped-def]
    s = tables.event_interval_summary
    van = s[s["canonical_location"] == VAN_DOWN[1]]
    assert int(van["price_change_count"].sum()) == 2 and VAN_THUR[1] not in set(s["canonical_location"])
    detail = build_detailed_event_table(rich)
    assert not detail.duplicated(list(EVENT_KEY_COLUMNS)).any()
    assert int(van["multi_source_candidates"].sum()) > 0


def test_final_vancouver_table_is_a_subset_of_the_validated_case(rich, tables) -> None:  # type: ignore[no-untyped-def]
    case = rich.report.final_decrease
    frame = tables.final_vancouver_decrease
    assert tuple(frame.columns) == SANITIZED_TABLE_SCHEMAS["final_vancouver_decrease"] and len(frame) == 1
    f = frame.iloc[0].to_dict()
    assert f["status"] == "derived" and f[CUR] == case.current_period and f[PREV] == case.previous_period
    assert f["decrease"] == dict(case.counts)["decrease"] and f["price_change_count"] == case.price_change_count == 2
    assert f["assortment_event_count"] == case.assortment_event_count
    assert f["persistence_testable"] is False and f["not_testable_right_censored_final_capture"] == 2
    assert f["persistence_not_testable"] == 2 and f["indicator_persistence_not_testable_right_censored"] is True
    assert (f["provenance_dual_alias_source"], f["provenance_primary_alias_only"]) == (2, 1)   # filler + A; B
    assert sum(f[c] for c in ("provenance_dual_alias_source", "provenance_primary_alias_only",
                              "provenance_secondary_alias_only", "provenance_other_canonical_location")) == sum(
        n for _, n in case.counts)
    assert (f["participating_locations"], f["airport_involved"], f["downtown_involved"]) == (2, True, True)
    assert f["all_locations_end_at_final_capture"] is True
    assert f["decrease_cents_min"] is not None and f["magnitude_suppressed"] is False
    s = tables.event_interval_summary
    sub = s[(s["canonical_city"] == "vancouver") & (s[CUR] == case.current_period)]
    assert int(sub["price_change_count"].sum()) == case.price_change_count
    recon = tables.reconciliation_summary.set_index("check")
    for check in ("final_case_assortment_events_equal_case", "final_case_persistence_partitions_case_changes",
                  "final_case_cross_location_equal_cross_table", "final_case_provenance_equal_case_candidates"):
        assert recon.loc[check, "status"] == "reconciled"
    text = all_text(frame)
    assert "|" not in text and "Thurlow" not in text and "SYNTH" not in text and "not proof" not in text


# ============================================================================ heatmap


def heatmap_world(**extra):  # type: ignore[no-untyped-def]
    return analyze(merge(path(TOR_DOWN, (50.0, 55.0, 50.0, 45.0)), path(TOR_DOWN, (80.0, 70.0, 70.0, 75.0), "SYNTH Car B"),
                         path(CAL_AIR, (50.0, 55.0, 99.0, 60.0))), hours=4, excluded=("calgary", 2),
                   drop_streams={(CAL_DOWN, 2)}, **extra)


def test_heatmap_source_keeps_quiet_intervals_and_masks_breaks() -> None:
    result = heatmap_world()
    heat = heatmap_source_frame(result)
    interval = heat[heat["cell_state"] == "interval"]
    assert len(interval) == result.report.intervals
    quiet = interval[(interval["increase"] == 0) & (interval["decrease"] == 0)]
    assert len(quiet) > 0                                                       # quiet intervals are present
    calgary = heat[heat["canonical_city"] == "calgary"]
    breaks = calgary[calgary["cell_state"] == "break"]
    assert len(breaks) == 2 * 3 and breaks[["increase", "decrease"]].isna().all().all()   # never zero
    assert heat.loc[heat["cell_state"] != "interval", ["increase", "decrease"]].isna().all().all()


@pytest.mark.parametrize("world", [dict(absent_jobs={("toronto", 2)}, excused=("toronto", 2)),
                                   dict(short={VAN_THUR})])
def test_missing_capture_and_source_stream_breaks_are_masked(world) -> None:  # type: ignore[no-untyped-def]
    result = analyze(path(TOR_DOWN, (50.0, 55.0, None, 60.0)) if "excused" in world else None,
                     hours=4 if "excused" in world else 3, **world)
    heat = heatmap_source_frame(result)
    key = TOR_DOWN if "excused" in world else VAN_DOWN
    mine = heat[(heat["canonical_city"] == key[0]) & (heat["canonical_location"] == key[1])]
    assert (mine["cell_state"] == "break").sum() >= 2 and mine.loc[mine["cell_state"] == "break",
                                                                   "increase"].isna().all()


def test_increases_and_decreases_stay_separate_in_mixed_intervals() -> None:
    result = heatmap_world()
    heat = heatmap_source_frame(result)
    mixed = heat[(heat["increase"] > 0) & (heat["decrease"] > 0)]
    assert len(mixed) == 2 and (mixed["increase"] == 1).all() and (mixed["decrease"] == 1).all()
    s = build_presentation_tables(result).event_interval_summary
    assert int(heat["increase"].sum()) == int(s["increase"].sum())
    assert int(heat["decrease"].sum()) == int(s["decrease"].sum())


def test_rendering_writes_a_valid_image_only_to_the_requested_directory(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    result = heatmap_world()
    before = plt.get_fignums()
    out = tmp_path / "out"
    figure = render_price_change_heatmap(result, out)
    assert figure == out / HEATMAP_FILENAME and os.listdir(out) == [HEATMAP_FILENAME] and os.listdir(cwd) == []
    data = figure.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000
    from PIL import Image as PILImage

    with PILImage.open(figure) as image:
        width, height = image.size
    assert width > 600 and height > 400
    assert plt.get_fignums() == before                                         # the figure is closed
    assert heatmap_png(result)[:4] == b"\x89PNG" and os.listdir(out) == [HEATMAP_FILENAME]
    with pytest.raises(TypeError):
        render_price_change_heatmap(result, None)  # type: ignore[arg-type]


def test_rendering_refuses_tampered_or_unreconciled_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    result = heatmap_world()
    object.__setattr__(result, "event_table", result.event_table.assign(increase=0))
    with pytest.raises(PriceChangeReconciliationError):
        render_price_change_heatmap(result, tmp_path / "out")
    with pytest.raises(PriceChangeReconciliationError):
        heatmap_png(result)
    assert not (tmp_path / "out").exists()


# ============================================================================ exports


def test_aggregate_export_writes_only_expected_sanitized_files(tables, tmp_path) -> None:  # type: ignore[no-untyped-def]
    written = export_sanitized_tables(tables, tmp_path / "out")
    names = sorted(os.listdir(tmp_path / "out"))
    assert names == sorted(f"price_change_{n}.csv" for n in SANITIZED_TABLE_SCHEMAS)
    assert [p.name for p in written] == [f"price_change_{n}.csv" for n in SANITIZED_TABLE_SCHEMAS]
    for name, file in zip(SANITIZED_TABLE_SCHEMAS, written):
        frame = pd.read_csv(file)
        assert tuple(frame.columns) == SANITIZED_TABLE_SCHEMAS[name]
        text = file.read_text(encoding="utf-8")
        assert "SYNTH" not in text and "2030-04" not in text
        for forbidden in FORBIDDEN_SANITIZED_COLUMNS.values():
            assert not set(frame.columns) & forbidden
    with pytest.raises(FileNotFoundError):
        export_sanitized_tables(tables, tmp_path / "missing" / "nested")         # parents are never created


def test_detailed_export_is_opt_in_typed_complete_and_owner_only(rich, tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "local"
    file, rows = write_detailed_event_table(rich, out)
    assert file.name == DETAILED_EVENT_TABLE_FILENAME and os.listdir(out) == [DETAILED_EVENT_TABLE_FILENAME]
    assert rows == len(rich.events.candidates)
    if os.name == "posix":
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
    frame = pd.read_parquet(file)
    assert tuple(frame.columns) == DETAILED_EVENT_TABLE_COLUMNS and len(frame) == rows
    assert set(CANDIDATE_COLUMNS) <= set(frame.columns)
    changed = frame["outcome"].isin(["increase", "decrease"])
    assert frame.loc[changed, "persistence"].notna().all() and frame.loc[~changed, "persistence"].isna().all()
    assert str(frame["pickup_date"].dtype) in ("object", "date32[day][pyarrow]")
    assert frame["previous_offer_count"].dtype.kind == "i"
    assert frame.loc[frame["outcome"] == "appeared", "previous_price_cents"].isna().all()
    assert frame["final_vancouver_decrease_case"].sum() > 0
    assert not any(isinstance(v, str) and str(tmp_path) in v for v in frame["evidence_id"])


def test_detailed_export_never_appears_in_results_or_reports(rich, tmp_path) -> None:  # type: ignore[no-untyped-def]
    w = synthetic_world(products=RICH)
    result = presentation_from_pipeline(pipeline_result(w), output_dir=tmp_path / "out", write_detail=True)
    shown = repr(result) + repr(result.report) + str(result.tables)
    assert "SYNTH" not in shown and "DataFrame" not in shown and str(tmp_path) not in shown
    assert "detailed_event_table" in result.report.artifacts_written
    fields = {f.name: f for f in dataclasses.fields(PriceChangePresentationResult)}
    assert all(not fields[n].repr and not fields[n].compare for n in ("analysis", "tables", "paths"))
    manifest = json.loads((tmp_path / "out" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["reconciled"] is True and "SYNTH" not in json.dumps(manifest)
    assert str(tmp_path) not in json.dumps(manifest)
    detail = next(a for a in manifest["artifacts"] if a["artifact"] == "detailed_event_table")
    assert detail["classification"] == "local_detail_confidential"
    assert detail["rows"] == len(result.analysis.events.candidates)
    assert sorted(os.listdir(tmp_path / "out")) == sorted(
        [f"price_change_{n}.csv" for n in SANITIZED_TABLE_SCHEMAS]
        + [HEATMAP_FILENAME, DETAILED_EVENT_TABLE_FILENAME, MANIFEST_FILENAME])


def test_detailed_export_refuses_raw_data_destinations(rich, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis import paths

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for target in (raw, raw / "nested", paths.RAW_DATA_DIR):
        with pytest.raises(PrivacyViolationError, match="rule source_data_directory"):
            write_detailed_event_table(rich, target)
    custom = tmp_path / "custom_source"
    custom.mkdir()
    monkeypatch.setenv(paths.RAW_DATA_DIR_ENV_VAR, str(custom))
    with pytest.raises(PrivacyViolationError):
        write_detailed_event_table(rich, custom / "exports")
    other = tmp_path / "read_dir"
    other.mkdir()
    with pytest.raises(PrivacyViolationError):
        write_detailed_event_table(rich, other, source_dirs=(other,))
    assert list(raw.iterdir()) == [] and list(custom.iterdir()) == [] and list(other.iterdir()) == []


def test_settings_require_an_explicit_directory_for_any_write(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv(PRESENTATION_OUTPUT_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(DETAIL_EXPORT_ENV_VAR, "1")
    assert presentation_settings() == (None, False)                            # detail needs a directory
    monkeypatch.setenv(PRESENTATION_OUTPUT_DIR_ENV_VAR, str(tmp_path))
    assert presentation_settings() == (tmp_path, True)
    monkeypatch.setenv(DETAIL_EXPORT_ENV_VAR, "yes")
    assert presentation_settings() == (tmp_path, False)
    assert presentation_settings(tmp_path / "x", False) == (tmp_path / "x", False)


# ============================================================================ orchestration and evidence


def test_the_runner_calls_the_pricing_pipeline_exactly_once(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis import pricing_pipeline

    w = synthetic_world(products=RICH)
    calls = []

    def fake(raw_dir=None):  # type: ignore[no-untyped-def]
        calls.append(raw_dir)
        return pipeline_result(w)

    monkeypatch.setattr(pricing_pipeline, "run_pricing_pipeline", fake)
    result = run_price_change_presentation(tmp_path / "synthetic_raw", output_dir=tmp_path / "out", write_detail=True)
    assert result.completed and len(calls) == 1
    assert set(result.report.artifacts_written) == {*SANITIZED_TABLE_SCHEMAS, "event_heatmap",
                                                    "detailed_event_table", "manifest"}
    nothing = run_price_change_presentation(tmp_path / "synthetic_raw")
    assert nothing.completed and nothing.report.artifacts_written == () and len(calls) == 2
    with pytest.raises(TypeError):
        run_price_change_presentation(tmp_path, write_detail=True)
    assert len(calls) == 2                                                     # refused before running the pipeline


def test_evidence_from_another_run_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    w, other = synthetic_world(products=RICH), synthetic_world()
    from stream_contract_fixtures import synthetic_location_authority
    from test_price_change_events import AUTHORITY, CONTRACT

    foreign = synthetic_location_authority(CONTRACT, dict(AUTHORITY.role_map.assignments),
                                           tuple((p.airport, p.downtown) for p in AUTHORITY.effective_pairs))
    mixed = dataclasses.replace(pipeline_result(w), location_authority=foreign)
    result = presentation_from_pipeline(mixed, output_dir=tmp_path / "out", write_detail=True)
    assert result.report.blockers == (PB.ANALYSIS_NOT_COMPLETED,) and result.tables is None
    stale = dataclasses.replace(pipeline_result(w), jobs=other["jobs"], cars=other["cars"])
    assert presentation_from_pipeline(stale).report.blockers == (PB.ANALYSIS_NOT_COMPLETED,)
    assert not (tmp_path / "out").exists()


def test_a_blocked_upstream_produces_no_tables_heatmap_or_detail(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from conftest import contract_columns, write_synthetic_csv

    from ql2_sixt_canada_analysis.schemas import DatasetKey

    raw = tmp_path / "raw_csv"
    raw.mkdir()
    for key in DatasetKey:
        write_synthetic_csv(raw / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    out = tmp_path / "out"
    result = run_price_change_presentation(raw, output_dir=out, write_detail=True)
    assert not result.completed and result.report.blockers == (PB.ANALYSIS_NOT_COMPLETED,)
    assert result.tables is None and result.paths == {} and result.report.artifacts_written == ()
    assert not out.exists() and str(raw) not in repr(result)
    blocked_analysis = result.analysis
    with pytest.raises(PriceChangeReconciliationError):
        build_presentation_tables(blocked_analysis)
    with pytest.raises(PriceChangeReconciliationError):
        write_detailed_event_table(blocked_analysis, tmp_path / "detail")
    assert not (tmp_path / "detail").exists()


def test_inputs_are_never_mutated(rich, tmp_path) -> None:  # type: ignore[no-untyped-def]
    frames = {n: getattr(rich, n).copy(deep=True) for n in ("event_table", "cross_location", "persistence")}
    candidates = rich.events.candidates.copy(deep=True)
    build_presentation_tables(rich)
    build_detailed_event_table(rich)
    render_price_change_heatmap(rich, tmp_path / "out")
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(getattr(rich, name), frame)
    pd.testing.assert_frame_equal(rich.events.candidates, candidates)


def test_presentation_tables_reject_unreconciled_construction(tables) -> None:  # type: ignore[no-untyped-def]
    broken = tables.reconciliation_summary.assign(status="failed")
    with pytest.raises(PriceChangeReconciliationError):
        dataclasses.replace(tables, reconciliation_summary=broken)
    assert isinstance(tables, PresentationTables) and "DataFrame" not in repr(tables)


def test_full_synthetic_workflow_is_independent_of_input_row_order(tmp_path) -> None:  # type: ignore[no-untyped-def]
    w = synthetic_world(products=RICH)
    reference = presentation_from_pipeline(pipeline_result(w))
    shuffled = dict(w)
    from test_price_change_events import readiness_for
    from ql2_sixt_canada_analysis.canonical_offers import assess_canonical_offers
    from ql2_sixt_canada_analysis.collection_schedule import assess_per_stream_scheduled_coverage
    from ql2_sixt_canada_analysis.pricing_population import PricingPopulation, frame_binding

    order = np.random.default_rng(5).permutation(len(w["cars"]))
    cars = w["cars"].iloc[order].reset_index(drop=True)
    jobs = w["jobs"]
    scheduled = assess_per_stream_scheduled_coverage(jobs, cars, schedule=w["scheduled"].schedule,
                                                     contract=w["location_authority"].contract,
                                                     relationship=__import__("ql2_sixt_canada_analysis.schemas",
                                                                             fromlist=["x"]).JOB_DETAIL_RELATIONSHIP)
    population = PricingPopulation(binding=frame_binding(jobs, cars), parent_status=w["population"].parent_status,
                                   detail_status=tuple(np.array(w["population"].detail_status, dtype=object)[order]))
    offers = assess_canonical_offers(jobs, cars, population=population, scheduled=scheduled,
                                     policy=w["canonical_offers"].policy)
    shuffled.update(cars=cars, scheduled=scheduled, population=population, canonical_offers=offers,
                    readiness=readiness_for(cars, scheduled, offers))
    other = presentation_from_pipeline(pipeline_result(shuffled))
    assert other.completed
    for (name, a), (_, b) in zip(reference.tables.items(), other.tables.items()):
        if name == "reconciliation_summary":
            continue
        pd.testing.assert_frame_equal(a, b, obj=name)
    pd.testing.assert_frame_equal(build_detailed_event_table(reference.analysis).drop(columns="evidence_id"),
                                  build_detailed_event_table(other.analysis).drop(columns="evidence_id"))


# ============================================================================ packaging, import, repository hygiene


def test_importing_the_module_performs_no_io_pipeline_or_plotting() -> None:
    code = ("import builtins, io, os, sys\n"
            f"ROOT = {str(ROOT)!r}\n"
            "real = builtins.open\n"
            "def guarded(file, mode='r', *a, **k):\n"
            "    path = os.path.abspath(os.fspath(file)) if isinstance(file, (str, bytes, os.PathLike)) else ''\n"
            "    if any(c in mode for c in 'wax+') or str(path).startswith(ROOT):\n"
            "        raise AssertionError('I/O during import')\n"
            "    return real(file, mode, *a, **k)\n"
            "builtins.open = io.open = guarded\n"
            "before = sorted(os.listdir('.'))\n"
            "import ql2_sixt_canada_analysis.price_change_presentation as m\n"
            "print('matplotlib' in sys.modules, 'ql2_sixt_canada_analysis.pricing_pipeline' in sys.modules,\n"
            "      sorted(os.listdir('.')) == before)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False", "True"]


def test_public_package_exports_are_complete() -> None:
    assert len(pcp.__all__) == len(set(pcp.__all__))
    public = {n for n in vars(pcp) if not n.startswith("_") and getattr(getattr(pcp, n), "__module__", None)
              == pcp.__name__}
    assert public <= set(pcp.__all__)
    for name in pcp.__all__:
        assert name in package.__all__ and getattr(package, name) is getattr(pcp, name)
    for name in ("run_price_change_presentation", "build_presentation_tables", "heatmap_source_frame",
                 "render_price_change_heatmap", "export_sanitized_tables", "write_detailed_event_table"):
        assert name in pcp.__all__


def test_generated_presentation_artifacts_are_ignored_and_untracked() -> None:
    names = ["reports/price_change_events/price_change_event_interval_summary.csv",
             f"reports/price_change_events/{HEATMAP_FILENAME}",
             f"reports/price_change_events/{DETAILED_EVENT_TABLE_FILENAME}",
             f"elsewhere/{DETAILED_EVENT_TABLE_FILENAME}", f"reports/price_change_events/{MANIFEST_FILENAME}"]
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    ignored = subprocess.run(["git", "check-ignore", "--no-index", *names], cwd=ROOT, capture_output=True, text=True)
    assert sorted(ignored.stdout.split()) == sorted(names)
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert not re.search(r"price_change_event_(detail|heatmap)|price_change_presentation_manifest|"
                         r"reports/price_change_events/", tracked)


def test_readme_documents_privacy_and_data_plan_alignment() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    for text in ("03_price_change_events.ipynb", "run_price_change_presentation", "Sanitized aggregate tables",
                 DETAILED_EVENT_TABLE_FILENAME, PRESENTATION_OUTPUT_DIR_ENV_VAR, DETAIL_EXPORT_ENV_VAR,
                 "right-censored", "observed price-change candidates", "roughly 90 hours",
                 "reports/price_change_events/", "Clear All Outputs"):
        assert text in readme, text


# ============================================================================ semantic allowlist (final case)

SECRET = "SYNTH-SECRET-PRODUCT"
FINAL = "final_vancouver_decrease"


def final_row(tables, **changes):  # type: ignore[no-untyped-def]
    """A copy of the generated final-case record with ``changes`` applied (object dtype keeps exact types)."""
    frame = tables.final_vancouver_decrease.copy().astype(object)
    for column, value in changes.items():
        frame[column] = pd.Series([value], dtype=object)
    return frame


def rejected(name: str, frame: pd.DataFrame, rule: str) -> str:
    with pytest.raises(PrivacyViolationError) as info:
        validate_sanitized_frame(name, frame)
    message = str(info.value)
    assert f"rule {rule}" in message, message
    assert SECRET not in message and "5500" not in message
    return message


def non_derived(tables, status: str, city: object) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    frame = tables.final_vancouver_decrease.copy().astype(object)
    for column in frame.columns:
        frame[column] = pd.Series([None], dtype=object)
    frame["status"] = pd.Series([status], dtype=object)
    frame["canonical_city"] = pd.Series([city], dtype=object)
    return frame


def test_the_generated_final_case_has_exactly_the_fixed_schema_and_passes(tables) -> None:  # type: ignore[no-untyped-def]
    frame = tables.final_vancouver_decrease
    assert tuple(frame.columns) == SANITIZED_TABLE_SCHEMAS[FINAL] and len(frame) == 1
    assert {"section", "metric", "value"}.isdisjoint(frame.columns)
    validate_sanitized_frame(FINAL, frame)
    kinds = pcp.SANITIZED_COLUMN_KINDS
    assert set(SANITIZED_TABLE_SCHEMAS[FINAL]) <= set(kinds) and set(kinds) == SANITIZED_COLUMN_ALLOWLIST


def test_no_decrease_and_city_unavailable_cases_pass_with_only_their_fields(tables) -> None:  # type: ignore[no-untyped-def]
    generated = build_presentation_tables(analyze(path(VAN_DOWN, (50.0, 55.0)))).final_vancouver_decrease
    assert generated.iloc[0]["status"] == "no_decrease" and generated.iloc[0]["canonical_city"] == "vancouver"
    assert all(v is None for c, v in generated.iloc[0].items() if c not in ("status", "canonical_city"))
    validate_sanitized_frame(FINAL, generated)
    validate_sanitized_frame(FINAL, non_derived(tables, "city_unavailable", None))
    rejected(FINAL, non_derived(tables, "city_unavailable", "vancouver"), "status_fields")
    rejected(FINAL, non_derived(tables, "no_decrease", None), "status_fields")
    leaked = non_derived(tables, "no_decrease", "vancouver")
    leaked["price_change_count"] = pd.Series([3], dtype=object)
    rejected(FINAL, leaked, "status_fields")                                   # derived-only field on a non-derived case


@pytest.mark.parametrize(("column", "rule"), [
    ("car_name", "product_identity"), ("car_type", "product_identity"), ("transmission", "product_identity"),
    ("previous_price_cents", "individual_price"), ("current_price", "individual_price"),
    ("change_percent", "individual_price"), ("source_location_labels", "source_provenance"),
    ("job_id", "raw_identifier"), ("row_index", "raw_identifier"), ("section", "allowlist"),
    ("metric", "allowlist"), ("value", "allowlist"), ("free_text", "allowlist")])
def test_forbidden_or_unknown_columns_are_rejected_from_the_final_case(tables, column, rule) -> None:  # type: ignore[no-untyped-def]
    message = rejected(FINAL, final_row(tables, **{column: SECRET}), rule)
    assert column not in message or rule == "allowlist"


def test_missing_or_reordered_final_case_columns_are_rejected(tables) -> None:  # type: ignore[no-untyped-def]
    frame = tables.final_vancouver_decrease
    rejected(FINAL, frame.drop(columns="persistence_testable"), "schema")
    rejected(FINAL, frame[list(reversed(frame.columns))], "schema")


@pytest.mark.parametrize(("column", "value", "rule"), [
    # The demonstrated vulnerability: approved schema, product- or price-level content in its fields.
    ("status", "car_name", "value_domain"),
    ("status", SECRET, "value_domain"),
    ("canonical_city", SECRET, "value_domain"),
    (CUR, SECRET, "value_domain"),
    (PREV, "20300304T0800Z", "value_domain"),                                 # malformed scheduled period
    ("price_change_count", 5500.0, "value_domain"),                           # a price in a count field
    ("comparable", "previous_price_cents", "value_domain"),
    ("decrease_cents_min", "5500", "value_domain"),
    ("airport_involved", "true", "value_domain"),                             # non-boolean flag
    ("airport_involved", 1, "value_domain"),
    ("price_change_count", -1, "value_domain"),                               # negative count
    ("price_change_count", True, "value_domain"),                             # boolean used as a count
    ("changed_share_of_comparable", 1.5, "value_domain"),
    ("changed_share_of_comparable", -0.1, "value_domain"),
    ("changed_share_of_comparable", float("inf"), "value_domain"),
    ("decrease_percent_max", float("-inf"), "value_domain"),
    ("status", "proven_repricing", "value_domain"),                           # unknown enum value
    ("provenance_dual_alias_source", "Vancouver Downtown|Vancouver Thurlow", "source_provenance"),
    ("comparable", None, "status_fields"),                                    # a derived case needs its fields
])
def test_product_price_and_malformed_values_in_approved_fields_are_rejected(tables, column, value, rule) -> None:  # type: ignore[no-untyped-def]
    rejected(FINAL, final_row(tables, **{column: value}), rule)


def test_nan_shares_are_missing_values_never_valid_numbers(tables) -> None:  # type: ignore[no-untyped-def]
    validate_sanitized_frame(FINAL, final_row(tables, changed_share_of_comparable=float("nan")))  # optional share
    summary = tables.event_interval_summary.copy().astype(object)
    summary.loc[summary.index[0], "candidates"] = float("nan")
    rejected("event_interval_summary", summary, "missing_value")
    summary = tables.event_interval_summary.copy().astype(object)
    summary.loc[summary.index[0], "changed_share_of_comparable"] = float("inf")
    rejected("event_interval_summary", summary, "value_domain")


def test_duplicate_records_and_extra_final_rows_are_rejected(tables) -> None:  # type: ignore[no-untyped-def]
    rejected(FINAL, pd.concat([tables.final_vancouver_decrease] * 2, ignore_index=True), "cardinality")
    summary = tables.event_interval_summary
    rejected("event_interval_summary", pd.concat([summary, summary.iloc[:1]], ignore_index=True), "duplicate_record")
    recon = tables.reconciliation_summary
    rejected("reconciliation_summary", pd.concat([recon, recon.iloc[:1]], ignore_index=True), "duplicate_record")


@pytest.mark.parametrize(("table", "column", "value"), [
    ("event_interval_summary", "canonical_location", SECRET),                 # unapproved location label
    ("event_interval_summary", "role", "Airport"),
    ("event_interval_summary", "movement_class", SECRET),
    ("event_interval_summary", "interval_flag", "car_name"),
    ("persistence_summary", "direction", "unchanged"),
    ("reconciliation_summary", "check", "previous_price_cents"),
    ("reconciliation_summary", "status", "ok"),
    ("material_selection_reconciliation", "selected", "yes"),
    ("airport_downtown_summary", "airport_location", "Vancouver Thurlow"),     # a source alias, not canonical
])
def test_every_sanitized_table_is_semantically_validated(tables, table, column, value) -> None:  # type: ignore[no-untyped-def]
    frame = getattr(tables, table).copy().astype(object)
    frame.loc[frame.index[0], column] = value
    rejected(table, frame, "value_domain")


def test_tables_mutated_after_construction_are_rejected_at_export(tables, tmp_path) -> None:  # type: ignore[no-untyped-def]
    mutated = build_presentation_tables(analyze(RICH))
    mutated.final_vancouver_decrease.loc[0, "canonical_city"] = SECRET          # in-place, after validation
    with pytest.raises(PrivacyViolationError) as info:
        export_sanitized_tables(mutated, tmp_path / "out")
    assert SECRET not in str(info.value) and not (tmp_path / "out").exists()
    priced = build_presentation_tables(analyze(RICH))
    priced.event_interval_summary.loc[0, "interval_flag"] = SECRET
    with pytest.raises(PrivacyViolationError):
        export_sanitized_tables(priced, tmp_path / "out")
    unreconciled = build_presentation_tables(analyze(RICH))
    unreconciled.reconciliation_summary.loc[0, "observed"] = 10 ** 6
    with pytest.raises(PriceChangeReconciliationError):
        export_sanitized_tables(unreconciled, tmp_path / "out")
    assert not (tmp_path / "out").exists()                                     # nothing is written on refusal


def test_exported_final_case_csv_holds_only_the_fixed_aggregate_fields(tables, tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = export_sanitized_tables(tables, tmp_path / "out")
    final = next(p for p in paths if p.name == f"price_change_{FINAL}.csv")
    frame = pd.read_csv(final)
    assert tuple(frame.columns) == SANITIZED_TABLE_SCHEMAS[FINAL] and len(frame) == 1
    text = final.read_text(encoding="utf-8")
    assert "SYNTH" not in text and "|" not in text and "Thurlow" not in text and "not proof" not in text
