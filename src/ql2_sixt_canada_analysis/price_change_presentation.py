"""Price-change event presentation: sanitized aggregate tables, the event heatmap and the local detail export.

The presentation layer is subordinate to the validated chain
``run_pricing_pipeline`` -> :mod:`~ql2_sixt_canada_analysis.price_change_events`
-> :mod:`~ql2_sixt_canada_analysis.price_change_analysis`. It never rebuilds
events, never redefines a population and calculates nothing that the
validated reports and tables do not already establish: every table is a
projection or aggregation of a completed
:class:`~ql2_sixt_canada_analysis.price_change_analysis.PriceChangeAnalysisResult`
and is reconciled back to it before anything is returned, rendered or written.
All results describe **observed price-change candidates**; nothing here proves
a genuine repricing or an extraction anomaly.

Entry points
------------
* :func:`run_price_change_presentation` - calls ``run_pricing_pipeline`` exactly
  once and passes that one pipeline result through the event engine, the
  higher-order analysis and the presentation (optionally writing artifacts).
* :func:`build_presentation_tables` / :func:`heatmap_source_frame` - pure,
  in-memory.
* :func:`render_price_change_heatmap`, :func:`export_sanitized_tables`,
  :func:`write_detailed_event_table`, :func:`write_presentation_manifest` -
  write only into an explicitly supplied output directory, atomically.

Sanitized aggregate tables
--------------------------
"Sanitized" is defined by an explicit allowlist
(:data:`SANITIZED_COLUMN_ALLOWLIST`, exact per-table schemas in
:data:`SANITIZED_TABLE_SCHEMAS`) and enforced by :func:`validate_sanitized_frame`:
no product identity, rental dates, individual prices or changes, source
provenance strings, raw job or row identifiers or file paths. Magnitude summaries
of an interval with fewer than two changed offers are suppressed (they would
equal one offer's change). Sanitized tables are still **confidential local
artifacts**: they protect against product-level disclosure in the notebook; they
are not approved for committing or distribution.

Material synchronized movement
------------------------------
Selection is the descriptive event contract, not a threshold: a location
interval is selected when it is direction-synchronized (at least two changed
offers, all in one direction). Every other interval with a price change is
reported as excluded (isolated or mixed direction), and selected plus excluded
price changes reconcile to all price changes.

Local detailed event table
--------------------------
:func:`write_detailed_event_table` (explicit opt-in) writes one Parquet row per
validated candidate with its interval, cross-location, persistence and case
attributes. It refuses any raw or source-data directory, writes atomically with
owner-only permissions where supported, and returns only its path and row count.
It is never displayed, printed, held in a report or included in ``repr``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd

from ql2_sixt_canada_analysis.price_change_analysis import (
    CROSS_LOCATION_PRODUCT_COLUMNS,
    EVENT_TABLE_COLUMNS,
    CrossLocationOutcome,
    FinalDecreaseStatus,
    IntervalFlag,
    MovementClass,
    NotTestableReason,
    PersistenceOutcome,
    PriceChangeAnalysisResult,
    PriceChangeReconciliationError,
)
from ql2_sixt_canada_analysis.price_change_events import (
    CANDIDATE_COLUMNS,
    EVENT_IDENTITY_COLUMNS,
    EVENT_INTERVAL_COLUMNS,
    EVENT_KEY_COLUMNS,
    FORBIDDEN_TIMESTAMP_SOURCES,
    TerminalOutcome,
)

__all__ = [
    "DETAILED_EVENT_TABLE_COLUMNS",
    "DETAILED_EVENT_TABLE_FILENAME",
    "FORBIDDEN_SANITIZED_COLUMNS",
    "DETAIL_EXPORT_ENV_VAR",
    "HEATMAP_FILENAME",
    "HEATMAP_SOURCE_COLUMNS",
    "MANIFEST_FILENAME",
    "MATERIAL_SELECTION_RULE",
    "PRESENTATION_OUTPUT_DIR_ENV_VAR",
    "SANITIZED_COLUMN_ALLOWLIST",
    "SANITIZED_TABLE_SCHEMAS",
    "InterpretationStatus",
    "PresentationTables",
    "PriceChangePresentationBlocker",
    "PriceChangePresentationReport",
    "PriceChangePresentationResult",
    "PriceChangePresentationStatus",
    "PrivacyViolationError",
    "build_detailed_event_table",
    "build_presentation_tables",
    "export_sanitized_tables",
    "heatmap_png",
    "heatmap_source_frame",
    "presentation_settings",
    "presentation_from_pipeline",
    "render_price_change_heatmap",
    "run_price_change_presentation",
    "validate_sanitized_frame",
    "write_detailed_event_table",
    "write_presentation_manifest",
]

PREV, CUR = EVENT_INTERVAL_COLUMNS
HEATMAP_FILENAME = "price_change_event_heatmap.png"
DETAILED_EVENT_TABLE_FILENAME = "price_change_event_detail.local.parquet"
MANIFEST_FILENAME = "price_change_presentation_manifest.json"
#: Environment variable naming an explicit local output directory for the presentation artifacts.
PRESENTATION_OUTPUT_DIR_ENV_VAR = "QL2_SIXT_PRICE_CHANGE_OUTPUT_DIR"
#: Environment variable that opts in to the local detailed export (exactly ``"1"``; needs the output directory).
DETAIL_EXPORT_ENV_VAR = "QL2_SIXT_PRICE_CHANGE_WRITE_DETAIL"
#: The documented material-movement selection rule (descriptive contract; not an alert threshold).
MATERIAL_SELECTION_RULE = "direction_synchronized: at least two changed offers, all in one direction"
_MIN_CHANGES_FOR_MAGNITUDES = 2
_OUTCOMES = tuple(o.value for o in TerminalOutcome)
_PERSIST = tuple(f"persistence_{o.value}" for o in PersistenceOutcome)


class PrivacyViolationError(ValueError):
    """A presentation output would breach the sanitized-output contract (messages name the rule, not values)."""


class PriceChangePresentationStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class PriceChangePresentationBlocker(StrEnum):
    ANALYSIS_NOT_COMPLETED = "analysis_not_completed"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    RECONCILIATION_FAILED = "reconciliation_failed"


class InterpretationStatus(StrEnum):
    """Descriptive interpretation of one location interval (first applicable wins; never proof)."""

    EMPTY_ENDPOINT_REVIEW = "anomaly_indicator_empty_endpoint"
    MATERIAL_SYNCHRONIZED = "material_synchronized_candidate"
    MIXED_DIRECTION = "mixed_direction_movement"
    ISOLATED = "isolated_movement"
    ASSORTMENT_OR_AMBIGUITY_ONLY = "assortment_or_ambiguity_without_price_movement"
    NO_MOVEMENT = "no_price_movement"


# ------------------------------------------------------------------ schemas and privacy

_INTERVAL_KEY = ("canonical_city", "canonical_location", "role", PREV, CUR)
_SUMMARY_SCHEMA = (*_INTERVAL_KEY, "candidates", *_OUTCOMES, "comparable", "price_change_count",
                   "assortment_event_count", "percent_valid", "zero_denominator", "previous_offers", "current_offers",
                   "changed_share_of_comparable", "movement_class", "direction_synchronized",
                   "exact_cent_synchronized", "exact_percent_synchronized", "largest_same_cent_cohort",
                   "largest_same_percent_cohort", "magnitude_suppressed", "min_change_cents", "max_change_cents",
                   "median_abs_change_percent", "max_abs_change_percent", "multi_source_candidates",
                   "provenance_changed_candidates", *_PERSIST, "interval_flag", "material_synchronized",
                   "interpretation_status")
_CROSS_OUTCOMES = tuple(f"cross_{o.value}" for o in CrossLocationOutcome)
#: Exact column order of every sanitized aggregate table.
SANITIZED_TABLE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "event_interval_summary": _SUMMARY_SCHEMA,
    "material_synchronized_movements": (
        *_INTERVAL_KEY, "selection_rule", "movement_class", "comparable", "price_change_count", "increase",
        "decrease", "changed_share_of_comparable", "exact_cent_synchronized", "exact_percent_synchronized",
        "largest_same_cent_cohort", "largest_same_percent_cohort", "min_change_cents", "max_change_cents",
        "median_abs_change_percent", "max_abs_change_percent", "assortment_event_count", "ambiguous",
        *_PERSIST, "interval_flag"),
    "material_selection_reconciliation": ("movement_class", "selected", "intervals", "price_change_count"),
    "airport_downtown_summary": (
        "canonical_city", "airport_location", "downtown_location", PREV, CUR, "matched_products",
        "airport_only_products", "downtown_only_products", *_CROSS_OUTCOMES, "same_direction",
        "same_cent_change", "same_percent_change"),
    "persistence_summary": (
        "canonical_city", "canonical_location", "role", "direction", "changed_events", "testable",
        "comparable_following", *(o.value for o in PersistenceOutcome),
        *(f"not_testable_{r.value}" for r in NotTestableReason), "returned_to_prior_price", "overshot_prior_price",
        "held_share_of_comparable_following", "continued_share_of_comparable_following",
        "reverted_share_of_comparable_following", "disappeared_share_of_testable", "ambiguous_share_of_testable"),
    "final_vancouver_decrease": ("section", "metric", "value"),
    "reconciliation_summary": ("check", "expected", "observed", "status"),
}
#: Every column a sanitized aggregate table may contain (the union of the exact schemas).
SANITIZED_COLUMN_ALLOWLIST: frozenset[str] = frozenset(c for cols in SANITIZED_TABLE_SCHEMAS.values() for c in cols)
#: Columns that must never appear in a sanitized table, by privacy rule.
FORBIDDEN_SANITIZED_COLUMNS: Mapping[str, frozenset[str]] = {
    "product_identity": frozenset({"car_name", "car_type", "transmission", "seats", "bags", "pickup_date",
                                   "return_date", "currency", "price_basis"}),
    "individual_price": frozenset({"price_cents", "price_num", "price_per_day", "previous_price_cents",
                                   "current_price_cents", "change_cents", "previous_price", "current_price",
                                   "change_dollars", "change_percent"}),
    "source_provenance": frozenset({"previous_source_labels", "current_source_labels", "source_location_labels",
                                    "city", "location"}),
    "raw_identifier": frozenset({*FORBIDDEN_TIMESTAMP_SOURCES, "job_id", "row_index", "parent_key"}),
}
_FORBIDDEN_TEXT = ("|",)                    # joined provenance label sets never appear in an aggregate cell


def validate_sanitized_frame(name: str, frame: object) -> None:
    """Refuse a frame that is not exactly an allowlisted sanitized table (messages name the rule only).

    Raises:
        PrivacyViolationError: Unknown table, detailed frame, forbidden or unexpected columns, provenance text.
    """
    if name not in SANITIZED_TABLE_SCHEMAS:
        raise PrivacyViolationError("rule unknown_table: only allowlisted sanitized tables can be exported")
    if not isinstance(frame, pd.DataFrame):
        raise PrivacyViolationError("rule not_a_table: sanitized exports accept DataFrames only")
    columns = set(map(str, frame.columns))
    if set(CANDIDATE_COLUMNS) <= columns or set(EVENT_KEY_COLUMNS) <= columns:
        raise PrivacyViolationError("rule detailed_frame: a detailed event frame cannot be exported as sanitized")
    for rule, forbidden in FORBIDDEN_SANITIZED_COLUMNS.items():
        if columns & forbidden:
            raise PrivacyViolationError(f"rule {rule}: a forbidden column is present")
    if columns - SANITIZED_COLUMN_ALLOWLIST:
        raise PrivacyViolationError("rule allowlist: a column outside the sanitized allowlist is present")
    if tuple(map(str, frame.columns)) != SANITIZED_TABLE_SCHEMAS[name]:
        raise PrivacyViolationError("rule schema: columns differ from the table's exact sanitized schema")
    if name != "final_vancouver_decrease":
        for column in frame.columns:
            values = frame[column]
            if (values.dtype == object or pd.api.types.is_string_dtype(values)) and any(isinstance(v, str) and any(t in v for t in _FORBIDDEN_TEXT)
                                              for v in values):
                raise PrivacyViolationError("rule source_provenance: provenance label text in an aggregate cell")


# ------------------------------------------------------------------ tables


@dataclass(frozen=True)
class PresentationTables:
    """The sanitized aggregate tables of one analysis (in memory; confidential local results)."""

    event_interval_summary: pd.DataFrame = field(repr=False, compare=False)
    material_synchronized_movements: pd.DataFrame = field(repr=False, compare=False)
    material_selection_reconciliation: pd.DataFrame = field(repr=False, compare=False)
    airport_downtown_summary: pd.DataFrame = field(repr=False, compare=False)
    persistence_summary: pd.DataFrame = field(repr=False, compare=False)
    final_vancouver_decrease: pd.DataFrame = field(repr=False, compare=False)
    reconciliation_summary: pd.DataFrame = field(repr=False, compare=False)
    evidence_id: str = ""

    def __post_init__(self) -> None:
        for name in SANITIZED_TABLE_SCHEMAS:
            validate_sanitized_frame(name, getattr(self, name))
        recon = self.reconciliation_summary
        if recon.empty or not (recon["status"] == "reconciled").all():
            raise PriceChangeReconciliationError("presentation tables are not reconciled")

    def items(self) -> tuple[tuple[str, pd.DataFrame], ...]:
        return tuple((name, getattr(self, name)) for name in SANITIZED_TABLE_SCHEMAS)


def _require_analysis(analysis: object) -> PriceChangeAnalysisResult:
    if not isinstance(analysis, PriceChangeAnalysisResult):
        raise TypeError("expected a PriceChangeAnalysisResult")
    if not analysis.completed:
        raise PriceChangeReconciliationError("rule reconciliation: the analysis is not completed")
    return analysis


def _evidence_id(analysis: PriceChangeAnalysisResult) -> str:
    """A short non-reversible identifier of the bound evidence (no paths, values or keys)."""
    binding = analysis.events.binding
    digest = getattr(binding, "digest", "")
    text = f"{getattr(binding, 'parent_rows', '')}:{getattr(binding, 'detail_rows', '')}:{digest}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _interpretation(row: dict) -> str:
    I = InterpretationStatus
    if row["interval_flag"] == IntervalFlag.EMPTY_ENDPOINT.value:
        return I.EMPTY_ENDPOINT_REVIEW.value
    movement = row["movement_class"]
    if movement in (MovementClass.SYNCHRONIZED_INCREASE.value, MovementClass.SYNCHRONIZED_DECREASE.value):
        return I.MATERIAL_SYNCHRONIZED.value
    if movement == MovementClass.MIXED_DIRECTION.value:
        return I.MIXED_DIRECTION.value
    if movement in (MovementClass.ISOLATED_INCREASE.value, MovementClass.ISOLATED_DECREASE.value):
        return I.ISOLATED.value
    if row["assortment_event_count"] or row["ambiguous"]:
        return I.ASSORTMENT_OR_AMBIGUITY_ONLY.value
    return I.NO_MOVEMENT.value


def _interval_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in table.to_dict("records"):
        out = {c: record[c] for c in EVENT_TABLE_COLUMNS}
        suppress = record["price_change_count"] < _MIN_CHANGES_FOR_MAGNITUDES
        out["magnitude_suppressed"] = bool(suppress and record["price_change_count"] > 0)
        if suppress:
            for column in ("min_change_cents", "max_change_cents", "median_abs_change_percent",
                           "max_abs_change_percent"):
                out[column] = None
        out["material_synchronized"] = bool(record["direction_synchronized"])
        out["interpretation_status"] = _interpretation(record)
        rows.append(out)
    frame = pd.DataFrame(rows, columns=list(_SUMMARY_SCHEMA))
    if not len(frame):
        frame = pd.DataFrame(columns=list(_SUMMARY_SCHEMA))
    return frame.astype({c: bool for c in ("direction_synchronized", "exact_cent_synchronized",
                                           "exact_percent_synchronized", "magnitude_suppressed",
                                           "material_synchronized")})


def _material(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = SANITIZED_TABLE_SCHEMAS["material_synchronized_movements"]
    selected = summary[summary["material_synchronized"].astype(bool)].copy()
    selected["selection_rule"] = MATERIAL_SELECTION_RULE
    selected = selected.loc[:, list(schema)].reset_index(drop=True)
    recon = []
    for movement in MovementClass:
        rows = summary[summary["movement_class"] == movement.value]
        recon.append((movement.value, movement in (MovementClass.SYNCHRONIZED_INCREASE,
                                                   MovementClass.SYNCHRONIZED_DECREASE),
                      int(len(rows)), int(rows["price_change_count"].sum())))
    return selected, pd.DataFrame(recon, columns=list(SANITIZED_TABLE_SCHEMAS["material_selection_reconciliation"]))


def _airport_downtown(analysis: PriceChangeAnalysisResult) -> pd.DataFrame:
    cross, table = analysis.cross_location, analysis.event_table
    schema = SANITIZED_TABLE_SCHEMAS["airport_downtown_summary"]
    candidates = {(r["canonical_city"], r["canonical_location"], r[PREV], r[CUR]): r["candidates"]
                  for r in table.to_dict("records")}
    rows = []
    for summary in analysis.report.cross_location:
        if summary.airport is None:
            continue
        city, airport, downtown = summary.canonical_city, summary.airport[1], summary.downtown[1]
        intervals = sorted({(p, c) for (k, loc, p, c) in candidates if k == city and loc == airport}
                           & {(p, c) for (k, loc, p, c) in candidates if k == city and loc == downtown})
        mine = cross[cross["canonical_city"] == city]
        groups = {key: g for key, g in mine.groupby([PREV, CUR], sort=False)}
        for prev, cur in intervals:
            g = groups.get((prev, cur), mine.iloc[:0])
            counts = Counter(g["cross_outcome"])
            matched = len(g)
            rows.append((city, airport, downtown, prev, cur, matched,
                         candidates[(city, airport, prev, cur)] - matched,
                         candidates[(city, downtown, prev, cur)] - matched,
                         *(counts[o.value] for o in CrossLocationOutcome), int(g["same_direction"].sum()),
                         int(g["same_cent_change"].sum()), int(g["same_percent_change"].sum())))
    return pd.DataFrame(rows, columns=list(schema))


def _persistence_table(analysis: PriceChangeAnalysisResult) -> pd.DataFrame:
    schema = SANITIZED_TABLE_SCHEMAS["persistence_summary"]
    persistence = analysis.persistence
    roles = {s.canonical_location: s.role for s in analysis.report.locations}
    rows = []
    for key in (s.canonical_location for s in analysis.report.locations):
        for direction in (TerminalOutcome.INCREASE.value, TerminalOutcome.DECREASE.value):
            g = persistence[(persistence["canonical_city"] == key[0]) & (persistence["canonical_location"] == key[1])
                            & (persistence["direction"] == direction)]
            if not len(g):
                continue
            outcomes = Counter(g["persistence"])
            reasons = Counter(r for r in g["not_testable_reason"] if r is not None)
            comparable = outcomes["held"] + outcomes["continued"] + outcomes["reverted"]
            testable = len(g) - outcomes["not_testable"]

            def share(n: int, d: int) -> float | None:
                return n / d if d else None

            rows.append((key[0], key[1], roles[key], direction, len(g), testable, comparable,
                         *(outcomes[o.value] for o in PersistenceOutcome),
                         *(reasons[r.value] for r in NotTestableReason), int(g["returned_to_prior_price"].sum()),
                         int(g["overshot_prior_price"].sum()), share(outcomes["held"], comparable),
                         share(outcomes["continued"], comparable), share(outcomes["reverted"], comparable),
                         share(outcomes["disappeared"], testable), share(outcomes["ambiguous"], testable)))
    return pd.DataFrame(rows, columns=list(schema))


def _final_case(analysis: PriceChangeAnalysisResult) -> pd.DataFrame:
    case = analysis.report.final_decrease
    policy = getattr(analysis.location_authority, "policy", None)
    primary = getattr(policy, "first", (None, None))[1]
    secondary = getattr(policy, "second", (None, None))[1]
    rows: list[tuple[str, str, str]] = [("case", "status", case.status.value)]
    if case.status is FinalDecreaseStatus.DERIVED:
        rows += [("case", "canonical_city", case.canonical_city), ("case", PREV, case.previous_period),
                 ("case", CUR, case.current_period)]
        for key, role, final in case.locations:
            rows += [("location", f"{key[1]}:role", role), ("location", f"{key[1]}:ends_at_final_capture",
                                                             str(bool(final)).lower())]
        rows += [("outcomes", name, str(n)) for name, n in case.counts]
        rows += [("movement", "comparable", str(case.comparable)),
                 ("movement", "price_change_count", str(case.price_change_count)),
                 ("movement", "changed_share_of_comparable",
                  "" if case.changed_share_of_comparable is None else repr(case.changed_share_of_comparable)),
                 ("synchronization", "direction_synchronized", str(case.direction_synchronized).lower()),
                 ("synchronization", "exact_cent_synchronized", str(case.exact_cent_synchronized).lower()),
                 ("synchronization", "exact_percent_synchronized", str(case.exact_percent_synchronized).lower()),
                 ("synchronization", "largest_same_cent_cohort", str(case.largest_same_cent_cohort)),
                 ("synchronization", "largest_same_percent_cohort", str(case.largest_same_percent_cohort))]
        if case.price_change_count >= _MIN_CHANGES_FOR_MAGNITUDES:
            rows += [("decrease_magnitude", f"cents_{k}", repr(v)) for k, v in case.decrease_cents]
            rows += [("decrease_magnitude", f"percent_{k}", repr(v)) for k, v in case.decrease_percent]
        rows += [("assortment", "assortment_event_count", str(case.assortment_event_count))]
        rows += [("cross_location", name, str(n)) for name, n in case.cross_location]
        composition: Counter = Counter()
        for labels, n in case.provenance:
            parts = set(labels.split("|")) if labels else set()
            if {primary, secondary} <= parts:
                composition["dual_alias_source"] += n
            elif parts == {primary}:
                composition["primary_alias_only"] += n
            elif parts == {secondary}:
                composition["secondary_alias_only"] += n
            else:
                composition["other_canonical_location"] += n
        rows += [("provenance", name, str(composition[name])) for name in
                 ("dual_alias_source", "primary_alias_only", "secondary_alias_only", "other_canonical_location")]
        rows += [("persistence", name, str(n)) for name, n in case.persistence]
        rows += [("persistence", f"not_testable_{name}", str(n)) for name, n in case.not_testable_reasons]
        rows += [("persistence", "persistence_testable", str(case.persistence_testable).lower())]
        rows += [("indicators", "indicator", i.value) for i in case.indicators]
        rows += [("interpretation", "statement", case.describe())]
    return pd.DataFrame(rows, columns=list(SANITIZED_TABLE_SCHEMAS["final_vancouver_decrease"]))


def _check(rows: list, name: str, expected: int, observed: int) -> None:
    rows.append((name, int(expected), int(observed), "reconciled" if int(expected) == int(observed) else "failed"))


def _reconciliation(analysis, summary, material, selection, cross, persistence, heat):  # type: ignore[no-untyped-def]
    overall = analysis.events.report.overall
    report = analysis.report
    rows: list = []
    _check(rows, "event_candidates_equal_outcome_sum", overall.candidates,
           sum(getattr(overall, o) for o in _OUTCOMES))
    _check(rows, "candidate_frame_rows_equal_event_candidates", overall.candidates, len(analysis.events.candidates))
    _check(rows, "interval_candidates_equal_event_candidates", overall.candidates, int(summary["candidates"].sum()))
    _check(rows, "intervals_equal_event_intervals", overall.intervals, len(summary))
    _check(rows, "price_changes_equal_increase_plus_decrease", overall.increase + overall.decrease,
           int(summary["price_change_count"].sum()))
    _check(rows, "assortment_events_equal_appeared_plus_disappeared", overall.appeared + overall.disappeared,
           int(summary["assortment_event_count"].sum()))
    _check(rows, "persistence_partitions_price_changes", overall.changed,
           int(persistence[[o.value for o in PersistenceOutcome]].to_numpy().sum()) if len(persistence) else 0)
    _check(rows, "persistence_records_equal_price_changes", overall.changed, len(analysis.persistence))
    _check(rows, "selected_plus_excluded_equal_price_changes", overall.changed,
           int(selection["price_change_count"].sum()))
    _check(rows, "selected_price_changes_equal_material_table",
           int(selection.loc[selection["selected"], "price_change_count"].sum()),
           int(material["price_change_count"].sum()))
    matched = sum(s.matched_products for s in report.cross_location)
    _check(rows, "cross_location_matched_equal_analysis", matched, int(cross["matched_products"].sum()))
    _check(rows, "cross_location_rows_unique", len(analysis.cross_location),
           len(analysis.cross_location.drop_duplicates(["canonical_city", PREV, CUR,
                                                         *CROSS_LOCATION_PRODUCT_COLUMNS])))
    for side, column in (("airport", "airport_only_products"), ("downtown", "downtown_only_products")):
        _check(rows, f"cross_location_{side}_only_equal_analysis",
               sum(getattr(s, column) for s in report.cross_location), int(cross[column].sum()))
    keys = analysis.events.candidates[list(EVENT_KEY_COLUMNS)]
    _check(rows, "canonical_events_unique_across_aliases", len(keys), len(keys.drop_duplicates()))
    _check(rows, "heatmap_increases_equal_interval_summary", int(summary["increase"].sum()),
           int(heat["increase"].fillna(0).sum()))
    _check(rows, "heatmap_decreases_equal_interval_summary", int(summary["decrease"].sum()),
           int(heat["decrease"].fillna(0).sum()))
    _check(rows, "heatmap_interval_cells_equal_intervals", len(summary), int((heat["cell_state"] == "interval").sum()))
    case = report.final_decrease
    if case.status is FinalDecreaseStatus.DERIVED:
        sub = summary[(summary["canonical_city"] == case.canonical_city) & (summary[PREV] == case.previous_period)
                      & (summary[CUR] == case.current_period)]
        _check(rows, "final_case_price_changes_subset_of_intervals", case.price_change_count,
               int(sub["price_change_count"].sum()))
        _check(rows, "final_case_locations_subset_of_intervals", len(case.locations), len(sub))
    return pd.DataFrame(rows, columns=list(SANITIZED_TABLE_SCHEMAS["reconciliation_summary"]))


#: Long-form heatmap source: one cell per approved location and scheduled period.
HEATMAP_SOURCE_COLUMNS: tuple[str, ...] = ("canonical_city", "canonical_location", "role", "period", "cell_state",
                                           "increase", "decrease")


def heatmap_source_frame(analysis: PriceChangeAnalysisResult) -> pd.DataFrame:
    """The heatmap cells (``interval`` cells carry counts; ``break``/``outside`` cells are ``NaN``, never zero).

    Built by the analysis module's reconciled heatmap source and checked again
    against the event interval table.

    Raises:
        PriceChangeReconciliationError: The analysis is not completed or the totals disagree.
    """
    from ql2_sixt_canada_analysis.price_change_analysis import event_heatmap_source

    analysis = _require_analysis(analysis)
    source = event_heatmap_source(analysis)
    roles = {s.canonical_location: s.role for s in analysis.report.locations}
    rows = []
    for i, key in enumerate(source["locations"]):
        for j, period in enumerate(source["periods"]):
            state = str(source["state"][i, j])
            rows.append((key[0], key[1], roles[key], period, state,
                         float(source["increase"][i, j]), float(source["decrease"][i, j])))
    frame = pd.DataFrame(rows, columns=list(HEATMAP_SOURCE_COLUMNS))
    table = analysis.event_table
    interval = frame["cell_state"] == "interval"
    if (int(frame.loc[interval, "increase"].sum()) != int(table["increase"].sum())
            or int(frame.loc[interval, "decrease"].sum()) != int(table["decrease"].sum())
            or int(interval.sum()) != len(table) or frame.loc[~interval, ["increase", "decrease"]].notna().any().any()):
        raise PriceChangeReconciliationError("rule reconciliation: the heatmap source disagrees with the event table")
    return frame


def build_presentation_tables(analysis: PriceChangeAnalysisResult) -> PresentationTables:
    """Every sanitized aggregate table of one completed analysis (pure; nothing is written).

    Raises:
        PriceChangeReconciliationError: The analysis is not completed or a table does not reconcile.
        PrivacyViolationError: A table breaches the sanitized allowlist.
    """
    analysis = _require_analysis(analysis)
    summary = _interval_summary(analysis.event_table)
    material, selection = _material(summary)
    cross = _airport_downtown(analysis)
    persistence = _persistence_table(analysis)
    heat = heatmap_source_frame(analysis)
    recon = _reconciliation(analysis, summary, material, selection, cross, persistence, heat)
    if not (recon["status"] == "reconciled").all():
        raise PriceChangeReconciliationError("rule reconciliation: a presentation table does not reconcile")
    return PresentationTables(
        event_interval_summary=summary, material_synchronized_movements=material,
        material_selection_reconciliation=selection, airport_downtown_summary=cross,
        persistence_summary=persistence, final_vancouver_decrease=_final_case(analysis),
        reconciliation_summary=recon, evidence_id=_evidence_id(analysis))


# ------------------------------------------------------------------ writing


def _output_directory(output_dir: object, *, refuse_source_data: bool = False,
                      refuse_also: Sequence[str | os.PathLike] = ()) -> Path:
    """An explicit output directory (created itself only; its parent must exist)."""
    if output_dir is None or not isinstance(output_dir, (str, os.PathLike)) or not str(output_dir):
        raise TypeError("an explicit output directory is required")
    directory = Path(output_dir)
    if refuse_source_data:
        from ql2_sixt_canada_analysis.paths import DATA_DIR, RAW_DATA_DIR, resolve_raw_data_dir

        target = directory.resolve()
        sources = {RAW_DATA_DIR.resolve(), resolve_raw_data_dir().resolve(), (DATA_DIR / "raw").resolve(),
                   *(Path(extra).resolve() for extra in refuse_also)}
        parts = target.parts
        if (any(target == s or s in target.parents for s in sources)
                or any(a == "data" and b == "raw" for a, b in zip(parts, parts[1:]))):
            raise PrivacyViolationError("rule source_data_directory: detailed output cannot go into source data")
    if not directory.parent.exists():
        raise FileNotFoundError("the output directory's parent must already exist")
    directory.mkdir(exist_ok=True)
    return directory


def _write(path: Path, write, mode: int = 0o600) -> None:  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis.price_change_analysis import _atomic_write

    _atomic_write(path, write, mode=mode, create_parents=False)


def presentation_settings(output_dir: str | os.PathLike | None = None, write_detail: bool | None = None
                          ) -> tuple[Path | None, bool]:
    """``(output directory or None, write detail)`` from explicit arguments, else the documented variables.

    Nothing is written unless an output directory is set; the detailed export
    additionally needs ``write_detail`` (or ``DETAIL_EXPORT_ENV_VAR=1``).
    """
    if output_dir is None:
        output_dir = os.environ.get(PRESENTATION_OUTPUT_DIR_ENV_VAR, "") or None
    if write_detail is None:
        write_detail = os.environ.get(DETAIL_EXPORT_ENV_VAR, "") == "1"
    return (Path(output_dir) if output_dir is not None else None), bool(write_detail and output_dir is not None)


def heatmap_png(analysis: PriceChangeAnalysisResult, *, dpi: int = 110) -> bytes:
    """The reconciled heatmap as PNG bytes in memory (for notebook display; nothing is written; figure closed)."""
    import io

    from ql2_sixt_canada_analysis.price_change_analysis import plot_price_change_heatmap

    analysis = _require_analysis(analysis)
    build_presentation_tables(analysis)                   # every reconciliation gate before rendering
    fig, _ = plot_price_change_heatmap(analysis, dpi=dpi)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=fig.dpi, metadata={"Software": None})
    finally:
        fig.clear()
    return buffer.getvalue()


def render_price_change_heatmap(analysis: PriceChangeAnalysisResult, output_dir: str | os.PathLike,
                                *, dpi: int = 150) -> Path:
    """Reconcile, render and write the event heatmap to ``output_dir`` (closes its figure; returns the path).

    Raises:
        PriceChangeReconciliationError: Rendering before (or without) successful reconciliation.
    """
    from ql2_sixt_canada_analysis.price_change_analysis import plot_price_change_heatmap

    analysis = _require_analysis(analysis)
    tables = build_presentation_tables(analysis)          # every reconciliation gate, immediately before rendering
    heat = heatmap_source_frame(analysis)
    summary = tables.event_interval_summary
    if (int(heat["increase"].fillna(0).sum()) != int(summary["increase"].sum())
            or int(heat["decrease"].fillna(0).sum()) != int(summary["decrease"].sum())):
        raise PriceChangeReconciliationError("rule reconciliation: heatmap source differs from the interval summary")
    directory = _output_directory(output_dir)
    fig, _ = plot_price_change_heatmap(analysis, dpi=dpi)
    try:
        path = directory / HEATMAP_FILENAME
        _write(path, lambda p: fig.savefig(p, format="png", dpi=fig.dpi, metadata={"Software": None}), mode=0o644)
    finally:
        fig.clear()
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:  # pragma: no cover - the figure was never registered with pyplot
            pass
    return path


def export_sanitized_tables(tables: PresentationTables, output_dir: str | os.PathLike) -> tuple[Path, ...]:
    """Write every sanitized aggregate table as CSV into ``output_dir`` (validated again; atomic)."""
    if not isinstance(tables, PresentationTables):
        raise PrivacyViolationError("rule detailed_frame: only PresentationTables can be exported as sanitized")
    directory = _output_directory(output_dir)
    paths = []
    for name, frame in tables.items():
        validate_sanitized_frame(name, frame)
        path = directory / f"price_change_{name}.csv"
        _write(path, lambda p, f=frame: f.to_csv(p, index=False), mode=0o644)
        paths.append(path)
    return tuple(paths)


#: Columns of the local-only detailed event table (validated candidates plus higher-order attributes).
DETAILED_EVENT_TABLE_COLUMNS: tuple[str, ...] = (
    *CANDIDATE_COLUMNS, "role", "movement_class", "direction_synchronized", "exact_cent_synchronized",
    "exact_percent_synchronized", "interval_flag", "cross_counterpart_location", "cross_outcome",
    "cross_same_direction", "cross_same_cent_change", "cross_same_percent_change", "persistence",
    "not_testable_reason", "returned_to_prior_price", "overshot_prior_price", "final_vancouver_decrease_case",
    "evidence_id")


def build_detailed_event_table(analysis: PriceChangeAnalysisResult) -> pd.DataFrame:
    """The confidential detailed event table (one row per validated candidate). Never display or print it.

    Raises:
        PriceChangeReconciliationError: The analysis is not completed or a join duplicates or loses a row.
    """
    analysis = _require_analysis(analysis)
    events = analysis.events.candidates.copy()
    roles = {s.canonical_location[1]: s.role for s in analysis.report.locations}
    table = analysis.event_table[["canonical_city", "canonical_location", PREV, CUR, "movement_class",
                                  "direction_synchronized", "exact_cent_synchronized", "exact_percent_synchronized",
                                  "interval_flag"]]
    detail = events.merge(table, on=["canonical_city", "canonical_location", PREV, CUR], how="left",
                          validate="many_to_one")
    detail.insert(len(CANDIDATE_COLUMNS), "role", detail["canonical_location"].map(roles))
    cross = analysis.cross_location
    sides = []
    for own, other in (("airport_location", "downtown_location"), ("downtown_location", "airport_location")):
        side = cross.rename(columns={own: "canonical_location", other: "cross_counterpart_location",
                                     "same_direction": "cross_same_direction",
                                     "same_cent_change": "cross_same_cent_change",
                                     "same_percent_change": "cross_same_percent_change"})
        sides.append(side[["canonical_city", "canonical_location", PREV, CUR, *CROSS_LOCATION_PRODUCT_COLUMNS,
                           "cross_counterpart_location", "cross_outcome", "cross_same_direction",
                           "cross_same_cent_change", "cross_same_percent_change"]])
    detail = detail.merge(pd.concat(sides, ignore_index=True), how="left", validate="one_to_one",
                          on=["canonical_city", "canonical_location", PREV, CUR, *CROSS_LOCATION_PRODUCT_COLUMNS])
    persistence = analysis.persistence.drop(columns=["direction", "next_current_period"])
    detail = detail.merge(persistence, how="left", validate="one_to_one", on=[*EVENT_IDENTITY_COLUMNS, PREV, CUR])
    case = analysis.report.final_decrease
    detail["final_vancouver_decrease_case"] = (
        (detail["canonical_city"] == case.canonical_city) & (detail[PREV] == case.previous_period)
        & (detail[CUR] == case.current_period)) if case.status is FinalDecreaseStatus.DERIVED else False
    detail["evidence_id"] = _evidence_id(analysis)
    detail = detail.loc[:, list(DETAILED_EVENT_TABLE_COLUMNS)]
    changed = detail["outcome"].isin([TerminalOutcome.INCREASE.value, TerminalOutcome.DECREASE.value])
    if (len(detail) != len(events) or detail["persistence"].notna().sum() != changed.sum()
            or detail.loc[changed, "persistence"].isna().any()):
        raise PriceChangeReconciliationError("rule reconciliation: the detailed table does not match the candidates")
    return detail


def write_detailed_event_table(analysis: PriceChangeAnalysisResult, output_dir: str | os.PathLike, *,
                               source_dirs: Sequence[str | os.PathLike] = ()) -> tuple[Path, int]:
    """Explicit opt-in: write the confidential detailed table as Parquet (atomic, owner-only).

    Refuses the project raw-data directory, the configured raw-data override,
    any ``data/raw`` path and every directory in ``source_dirs`` (the raw directory
    the pipeline actually read). Returns ``(path, row count)`` only.

    Raises:
        PrivacyViolationError: The destination is a source-data directory.
        PriceChangeReconciliationError: The analysis is not completed or does not reconcile.
    """
    analysis = _require_analysis(analysis)
    directory = _output_directory(output_dir, refuse_source_data=True, refuse_also=source_dirs)
    detail = build_detailed_event_table(analysis)
    path = directory / DETAILED_EVENT_TABLE_FILENAME
    _write(path, lambda p: detail.to_parquet(p, index=False), mode=0o600)
    return path, len(detail)


def write_presentation_manifest(output_dir: str | os.PathLike, tables: PresentationTables,
                                artifacts: Mapping[str, Path], detail_rows: int | None = None) -> Path:
    """A local manifest: artifact names, relative file names, schemas, row counts and reconciliation status."""
    directory = _output_directory(output_dir)
    entries = []
    frames = dict(tables.items())
    for name, path in sorted(artifacts.items()):
        entry = {"artifact": name, "file": Path(path).name}
        if name in frames:
            entry.update(classification="sanitized_aggregate", columns=list(frames[name].columns),
                         rows=int(len(frames[name])))
        elif name == "detailed_event_table":
            entry.update(classification="local_detail_confidential", columns=list(DETAILED_EVENT_TABLE_COLUMNS),
                         rows=int(detail_rows or 0))
        else:
            entry.update(classification="aggregate_figure")
        entries.append(entry)
    manifest = {"evidence_id": tables.evidence_id, "reconciled": True,
                "reconciliation_checks": int(len(tables.reconciliation_summary)), "artifacts": entries}
    path = directory / MANIFEST_FILENAME
    _write(path, lambda p: p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"),
           mode=0o644)
    return path


# ------------------------------------------------------------------ results and orchestration


@dataclass(frozen=True)
class PriceChangePresentationReport:
    """Print-safe presentation status: blockers, table row counts, artifact names and reconciliation."""

    status: PriceChangePresentationStatus
    blockers: tuple[PriceChangePresentationBlocker, ...] = ()
    upstream_blockers: tuple[str, ...] = ()
    table_rows: tuple[tuple[str, int], ...] = ()
    reconciliation_checks: int = 0
    reconciled: bool = False
    artifacts_written: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is PriceChangePresentationStatus.BLOCKED:
            if not self.blockers or self.table_rows or self.reconciled or self.artifacts_written:
                raise PriceChangeReconciliationError("a blocked presentation has blockers and no outputs")
        elif self.blockers or not self.reconciled:
            raise PriceChangeReconciliationError("a completed presentation is reconciled without blockers")

    @property
    def completed(self) -> bool:
        return self.status is PriceChangePresentationStatus.COMPLETED


@dataclass(frozen=True)
class PriceChangePresentationResult:
    """The presentation report plus in-memory analysis, sanitized tables and written paths (none in ``repr``)."""

    report: PriceChangePresentationReport
    analysis: PriceChangeAnalysisResult | None = field(default=None, repr=False, compare=False)
    tables: PresentationTables | None = field(default=None, repr=False, compare=False)
    paths: Mapping[str, Path] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.report.completed and (self.tables is not None or self.paths):
            raise PriceChangeReconciliationError("a blocked presentation holds no tables or outputs")

    @property
    def completed(self) -> bool:
        return self.report.completed


def _blocked(blockers: Sequence[PriceChangePresentationBlocker], upstream: Sequence[str] = (),
             analysis: PriceChangeAnalysisResult | None = None) -> PriceChangePresentationResult:
    return PriceChangePresentationResult(PriceChangePresentationReport(
        status=PriceChangePresentationStatus.BLOCKED, blockers=tuple(blockers), upstream_blockers=tuple(upstream)),
        analysis=analysis)


def presentation_from_pipeline(run: object, *, output_dir: str | os.PathLike | None = None,
                               write_detail: bool = False,
                               source_dirs: Sequence[str | os.PathLike] = ()) -> PriceChangePresentationResult:
    """Events, analysis and presentation from one pipeline result (the same evidence throughout).

    Writes nothing unless ``output_dir`` is given; the detailed table only with ``write_detail=True``.
    """
    from ql2_sixt_canada_analysis.price_change_analysis import price_change_analysis_from_pipeline
    from ql2_sixt_canada_analysis.pricing_population import frame_binding

    B = PriceChangePresentationBlocker
    analysis = price_change_analysis_from_pipeline(run)
    if not analysis.completed:
        return _blocked([B.ANALYSIS_NOT_COMPLETED],
                        [b.value for b in analysis.report.blockers] + list(analysis.report.event_blockers), analysis)
    if (analysis.events.binding != frame_binding(run.jobs, run.cars)                     # type: ignore[attr-defined]
            or analysis.location_authority is not run.location_authority):              # type: ignore[attr-defined]
        return _blocked([B.EVIDENCE_MISMATCH])
    try:
        tables = build_presentation_tables(analysis)
    except PriceChangeReconciliationError:
        return _blocked([B.RECONCILIATION_FAILED])
    paths: dict[str, Path] = {}
    detail_rows = None
    if output_dir is not None:
        for name, path in zip(SANITIZED_TABLE_SCHEMAS, export_sanitized_tables(tables, output_dir)):
            paths[name] = path
        paths["event_heatmap"] = render_price_change_heatmap(analysis, output_dir)
        if write_detail:
            paths["detailed_event_table"], detail_rows = write_detailed_event_table(
                analysis, output_dir, source_dirs=tuple(source_dirs))
        paths["manifest"] = write_presentation_manifest(output_dir, tables, dict(paths), detail_rows)
    elif write_detail:
        raise TypeError("the detailed export needs an explicit output directory")
    report = PriceChangePresentationReport(
        status=PriceChangePresentationStatus.COMPLETED,
        table_rows=tuple((name, len(frame)) for name, frame in tables.items()),
        reconciliation_checks=len(tables.reconciliation_summary), reconciled=True,
        artifacts_written=tuple(sorted(paths)))
    return PriceChangePresentationResult(report=report, analysis=analysis, tables=tables, paths=paths)


def run_price_change_presentation(raw_dir: str | os.PathLike | None = None, *,
                                  output_dir: str | os.PathLike | None = None,
                                  write_detail: bool = False) -> PriceChangePresentationResult:
    """Run ``run_pricing_pipeline`` exactly once, then events, analysis and presentation on that one result."""
    from ql2_sixt_canada_analysis import pricing_pipeline

    from ql2_sixt_canada_analysis.paths import resolve_raw_data_dir

    if write_detail and output_dir is None:
        raise TypeError("the detailed export needs an explicit output directory")
    run = pricing_pipeline.run_pricing_pipeline(raw_dir)
    return presentation_from_pipeline(run, output_dir=output_dir, write_detail=write_detail,
                                      source_dirs=(resolve_raw_data_dir(raw_dir),))
