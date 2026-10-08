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
"Sanitized" is defined by allowlists at two levels and enforced by
:func:`validate_sanitized_frame`. Structurally, every table has an exact fixed
schema (:data:`SANITIZED_TABLE_SCHEMAS`, union :data:`SANITIZED_COLUMN_ALLOWLIST`);
semantically, every column has one value kind (:data:`SANITIZED_COLUMN_KINDS`:
counts, flags, shares, aggregate statistics, canonical periods, approved
cities/locations/roles, enum members, the documented rule text). There is no
generic section/metric/value table: the final Vancouver decrease is one
fixed-schema record whose provenance is four category counts. Unknown or
malformed fields fail closed, and validation is repeated immediately before
export. No product identity, rental dates, individual prices or changes, source
provenance strings, raw job or row identifiers or file paths can pass.

Every magnitude statistic is governed by its own contributing population
(:func:`magnitude_disclosable`, minimum :data:`MINIMUM_MAGNITUDE_CONTRIBUTORS`):
decrease cents by all decreases, decrease percentages by percent-valid decreases,
interval cents by changed offers and interval percentages by changed offers with
a nonzero previous price. Total price-change counts are never a denominator for
a narrower statistic. Suppressed values are absent (never zero) and are withheld
before any table exists. Magnitude summaries
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
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.price_change_analysis import (
    CROSS_LOCATION_PRODUCT_COLUMNS,
    EVENT_TABLE_COLUMNS,
    CrossLocationOutcome,
    FinalDecreaseIndicator,
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
    "MINIMUM_MAGNITUDE_CONTRIBUTORS",
    "PRESENTATION_OUTPUT_DIR_ENV_VAR",
    "RECONCILIATION_CHECKS",
    "SANITIZED_COLUMN_ALLOWLIST",
    "SANITIZED_COLUMN_KINDS",
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
    "final_case_table",
    "heatmap_png",
    "heatmap_source_frame",
    "magnitude_disclosable",
    "magnitude_suppressed",
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
#: The approved minimum number of contributing observations behind any presented magnitude statistic.
MINIMUM_MAGNITUDE_CONTRIBUTORS = 2


def _validated_threshold(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("the magnitude contributor threshold is an integer of at least two")
    return value


_MIN_CONTRIBUTORS = _validated_threshold(MINIMUM_MAGNITUDE_CONTRIBUTORS)


def magnitude_disclosable(contributors: object) -> bool:
    """Whether a magnitude statistic over exactly ``contributors`` observations may be presented.

    The count must be the statistic's own contributing population (for example
    percent-valid decreases for a decrease percentage), never a broader total.
    """
    if isinstance(contributors, (bool, np.bool_)) or not isinstance(contributors, (int, np.integer)) \
            or contributors < 0:
        raise ValueError("a contributor count is a non-negative integer")
    return int(contributors) >= _MIN_CONTRIBUTORS


def magnitude_suppressed(contributors: object) -> bool:
    """Contributors exist but are too few: the statistic exists and is withheld (zero contributors: no statistic)."""
    return not magnitude_disclosable(contributors) and int(contributors) > 0  # type: ignore[call-overload]
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
                   "largest_same_percent_cohort", "magnitude_suppressed", "change_percent_contributor_count",
                   "percent_magnitude_suppressed", "min_change_cents", "max_change_cents",
                   "median_abs_change_percent", "max_abs_change_percent", "multi_source_candidates",
                   "provenance_changed_candidates", *_PERSIST, "interval_flag", "material_synchronized",
                   "interpretation_status")
_CROSS_OUTCOMES = tuple(f"cross_{o.value}" for o in CrossLocationOutcome)
_PROVENANCE_CATEGORIES = ("provenance_dual_alias_source", "provenance_primary_alias_only",
                          "provenance_secondary_alias_only", "provenance_other_canonical_location")
_FINAL_CASE_CORE = ("status", "canonical_city")
#: Fixed aggregate fields of a derived final decrease (all missing unless the case is derived).
_FINAL_CASE_DERIVED = (
    PREV, CUR, "participating_locations", "airport_involved", "downtown_involved",
    "all_locations_end_at_final_capture", *_OUTCOMES, "comparable", "price_change_count", "assortment_event_count",
    "changed_share_of_comparable", "direction_synchronized", "exact_cent_synchronized",
    "exact_percent_synchronized", "largest_same_cent_cohort", "largest_same_percent_cohort",
    "decrease_cent_contributor_count", "decrease_percent_contributor_count", "decrease_zero_denominator_count",
    "decrease_cent_magnitude_suppressed", "decrease_percent_magnitude_suppressed",
    "decrease_cents_min", "decrease_cents_median", "decrease_cents_max", "decrease_percent_min",
    "decrease_percent_median", "decrease_percent_max", *_CROSS_OUTCOMES, *_PROVENANCE_CATEGORIES,
    *(f"persistence_{o.value}" for o in PersistenceOutcome),
    *(f"not_testable_{r.value}" for r in NotTestableReason), "persistence_testable",
    *(f"indicator_{i.value}" for i in FinalDecreaseIndicator))
#: Reconciliation checks a reconciliation summary may report (an explicit allowlist of check names).
RECONCILIATION_CHECKS: tuple[str, ...] = (
    "event_candidates_equal_outcome_sum", "candidate_frame_rows_equal_event_candidates",
    "interval_candidates_equal_event_candidates", "intervals_equal_event_intervals",
    "price_changes_equal_increase_plus_decrease", "assortment_events_equal_appeared_plus_disappeared",
    "persistence_partitions_price_changes", "persistence_records_equal_price_changes",
    "selected_plus_excluded_equal_price_changes", "selected_price_changes_equal_material_table",
    "cross_location_matched_equal_analysis", "cross_location_rows_unique",
    "cross_location_airport_only_equal_analysis", "cross_location_downtown_only_equal_analysis",
    "canonical_events_unique_across_aliases", "heatmap_increases_equal_interval_summary",
    "heatmap_decreases_equal_interval_summary", "heatmap_interval_cells_equal_intervals",
    "final_case_price_changes_subset_of_intervals", "final_case_locations_subset_of_intervals",
    "final_case_assortment_events_equal_case", "final_case_persistence_partitions_case_changes",
    "final_case_cross_location_equal_cross_table", "final_case_provenance_equal_case_candidates",
    "final_case_decreases_equal_outcome_counts", "final_case_cent_contributors_equal_decrease_rows",
    "final_case_percent_contributors_equal_percent_valid_decreases",
    "final_case_zero_denominator_decreases_equal_remaining_decreases",
    "final_case_presented_magnitudes_equal_validated_case",
    "interval_percent_contributors_equal_percent_valid_changes")
#: Exact column order of every sanitized aggregate table.
SANITIZED_TABLE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "event_interval_summary": _SUMMARY_SCHEMA,
    "material_synchronized_movements": (
        *_INTERVAL_KEY, "selection_rule", "movement_class", "comparable", "price_change_count", "increase",
        "decrease", "changed_share_of_comparable", "exact_cent_synchronized", "exact_percent_synchronized",
        "largest_same_cent_cohort", "largest_same_percent_cohort", "magnitude_suppressed",
        "change_percent_contributor_count", "percent_magnitude_suppressed", "min_change_cents", "max_change_cents",
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
    "final_vancouver_decrease": (*_FINAL_CASE_CORE, *_FINAL_CASE_DERIVED),
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


class _Kind(StrEnum):
    """Semantic value kinds of sanitized columns (every allowlisted column has exactly one)."""

    COUNT = "count"                  # non-negative integer, never a boolean
    FLAG = "flag"                    # boolean
    SHARE = "share"                  # finite, 0..1
    PERIOD = "period"                # canonical YYYYMMDDTHHMMSSZ scheduled period
    CITY = "city"                    # an approved canonical city
    LOCATION = "location"            # an approved canonical location name (paired with its city)
    ROLE = "role"                    # an approved location-role enum value
    SIGNED_CENTS = "signed_cents"    # finite aggregate cent statistic (may be negative)
    PERCENT = "percent"              # finite aggregate percentage statistic
    ENUM = "enum"                    # a member of the column's enum
    RULE_TEXT = "rule_text"          # exactly the documented selection rule


def _enum_values(enum: type[StrEnum]) -> frozenset[str]:
    return frozenset(m.value for m in enum)


_ENUMS: Mapping[str, frozenset[str]] = {
    "movement_class": _enum_values(MovementClass), "interval_flag": _enum_values(IntervalFlag),
    "interpretation_status": _enum_values(InterpretationStatus), "status": frozenset(),   # per table, below
    "direction": frozenset({TerminalOutcome.INCREASE.value, TerminalOutcome.DECREASE.value}),
    "check": frozenset(RECONCILIATION_CHECKS),
}
_TABLE_ENUMS: Mapping[tuple[str, str], frozenset[str]] = {
    ("final_vancouver_decrease", "status"): _enum_values(FinalDecreaseStatus),
    ("reconciliation_summary", "status"): frozenset({"reconciled", "failed"}),
}
_FLAGS = frozenset({"direction_synchronized", "exact_cent_synchronized", "exact_percent_synchronized",
                    "magnitude_suppressed", "percent_magnitude_suppressed", "decrease_cent_magnitude_suppressed",
                    "decrease_percent_magnitude_suppressed", "material_synchronized", "selected", "airport_involved",
                    "downtown_involved", "all_locations_end_at_final_capture", "persistence_testable",
                    *(f"indicator_{i.value}" for i in FinalDecreaseIndicator)})
_SHARES = frozenset({"changed_share_of_comparable", "held_share_of_comparable_following",
                     "continued_share_of_comparable_following", "reverted_share_of_comparable_following",
                     "disappeared_share_of_testable", "ambiguous_share_of_testable"})
_SIGNED = frozenset({"min_change_cents", "max_change_cents", "decrease_cents_min", "decrease_cents_median",
                     "decrease_cents_max"})
_PERCENTS = frozenset({"median_abs_change_percent", "max_abs_change_percent", "decrease_percent_min",
                       "decrease_percent_median", "decrease_percent_max"})
_LOCATIONS = frozenset({"canonical_location", "airport_location", "downtown_location"})
#: Columns that may be missing (``None``/``NaN``); every other value must be present.
_NULLABLE = _SHARES | _SIGNED | _PERCENTS | frozenset(_FINAL_CASE_DERIVED)


def _kind(column: str) -> _Kind:
    if column in _FLAGS:
        return _Kind.FLAG
    if column in _SHARES:
        return _Kind.SHARE
    if column in _SIGNED:
        return _Kind.SIGNED_CENTS
    if column in _PERCENTS:
        return _Kind.PERCENT
    if column in (PREV, CUR):
        return _Kind.PERIOD
    if column == "canonical_city":
        return _Kind.CITY
    if column in _LOCATIONS:
        return _Kind.LOCATION
    if column == "role":
        return _Kind.ROLE
    if column in _ENUMS:
        return _Kind.ENUM
    if column == "selection_rule":
        return _Kind.RULE_TEXT
    return _Kind.COUNT


#: The semantic kind of every allowlisted column (an explicit, complete type allowlist).
SANITIZED_COLUMN_KINDS: Mapping[str, str] = {c: _kind(c).value for c in sorted(SANITIZED_COLUMN_ALLOWLIST)}


def _missing(value: object) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _is_count(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) >= 0
    return False


def _is_number(value: object) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value)))


def _approved_locations() -> frozenset[tuple[str, str]]:
    from ql2_sixt_canada_analysis.location_authority import current_location_authority

    authority = current_location_authority()
    return frozenset(tuple(authority.canonical(k)) for k in authority.contract.expected_keys)


def _value_ok(table: str, column: str, value: object, approved: frozenset[tuple[str, str]]) -> bool:
    from ql2_sixt_canada_analysis.authority_decisions import LocationRoleDecision
    from ql2_sixt_canada_analysis.price_change_events import parse_scheduled_period

    kind = _kind(column)
    if kind is _Kind.COUNT:
        return _is_count(value)
    if kind is _Kind.FLAG:
        return isinstance(value, (bool, np.bool_))
    if kind is _Kind.SHARE:
        return _is_number(value) and 0.0 <= float(value) <= 1.0
    if kind in (_Kind.SIGNED_CENTS, _Kind.PERCENT):
        return _is_number(value) and (kind is _Kind.SIGNED_CENTS or column.startswith("decrease_")
                                      or float(value) >= 0.0)
    if not isinstance(value, str):
        return False
    if kind is _Kind.PERIOD:
        try:
            parse_scheduled_period(value)
        except ValueError:
            return False
        return True
    if kind is _Kind.CITY:
        return value in {city for city, _ in approved}
    if kind is _Kind.LOCATION:
        return value in {location for _, location in approved}
    if kind is _Kind.ROLE:
        return value in _enum_values(LocationRoleDecision)
    if kind is _Kind.RULE_TEXT:
        return value == MATERIAL_SELECTION_RULE
    return value in _TABLE_ENUMS.get((table, column), _ENUMS[column])


def _semantic(name: str, frame: pd.DataFrame, approved: frozenset[tuple[str, str]]) -> None:
    columns = list(frame.columns)
    for record in frame.itertuples(index=False, name=None):
        row = dict(zip(columns, record))
        for column, value in row.items():
            if isinstance(value, str) and "|" in value:
                raise PrivacyViolationError("rule source_provenance: provenance label text in an aggregate cell")
            if _missing(value):
                if column not in _NULLABLE and not (name == "final_vancouver_decrease"
                                                    and column == "canonical_city"):
                    raise PrivacyViolationError(f"rule missing_value: column {column} requires a value")
                continue
            if not _value_ok(name, column, value, approved):
                raise PrivacyViolationError(
                    f"rule value_domain: column {column} holds a value outside its {_kind(column).value} domain")
        city = row.get("canonical_city")
        for column in _LOCATIONS & set(columns):
            if (city, row[column]) not in approved:
                raise PrivacyViolationError(f"rule value_domain: column {column} is not an approved location")
        if name in ("event_interval_summary", "material_synchronized_movements"):
            _check_interval_suppression(row)
        if name == "final_vancouver_decrease":
            derived = row["status"] == FinalDecreaseStatus.DERIVED.value
            if derived:
                _check_final_suppression(row)
            if derived and any(_missing(row[c]) for c in _FINAL_CASE_DERIVED if c not in _SHARES | _SIGNED
                               | _PERCENTS):
                raise PrivacyViolationError("rule status_fields: a derived case needs every derived field")
            if not derived and any(not _missing(row[c]) for c in _FINAL_CASE_DERIVED):
                raise PrivacyViolationError("rule status_fields: a non-derived case holds derived-case fields")
            if (row["status"] == FinalDecreaseStatus.CITY_UNAVAILABLE.value) != _missing(row["canonical_city"]):
                raise PrivacyViolationError("rule status_fields: only an unavailable city has no canonical city")
    if name == "final_vancouver_decrease" and len(frame) != 1:
        raise PrivacyViolationError("rule cardinality: the final decrease table holds exactly one record")
    key = {"event_interval_summary": ["canonical_city", "canonical_location", PREV, CUR],
           "material_synchronized_movements": ["canonical_city", "canonical_location", PREV, CUR],
           "material_selection_reconciliation": ["movement_class"],
           "airport_downtown_summary": ["canonical_city", PREV, CUR],
           "persistence_summary": ["canonical_city", "canonical_location", "direction"],
           "reconciliation_summary": ["check"]}.get(name)
    if key and frame.duplicated(key).any():
        raise PrivacyViolationError("rule duplicate_record: a sanitized record is repeated")


def _present(row: dict, columns: Sequence[str]) -> bool:
    values = [not _missing(row[c]) for c in columns]
    if any(values) and not all(values):
        raise PrivacyViolationError("rule magnitude_suppression: a magnitude summary is partially present")
    return all(values)


def _check_interval_suppression(row: dict) -> None:
    """Interval magnitudes follow their own contributors: changes for cents, percent-valid changes for percents."""
    changes, percents = row["price_change_count"], row["change_percent_contributor_count"]
    if percents > changes:
        raise PrivacyViolationError("rule magnitude_suppression: percent contributors exceed changed offers")
    if _present(row, ("min_change_cents", "max_change_cents")) != magnitude_disclosable(changes) \
            or bool(row["magnitude_suppressed"]) != magnitude_suppressed(changes):
        raise PrivacyViolationError("rule magnitude_suppression: cent magnitudes disagree with their contributors")
    if _present(row, ("median_abs_change_percent", "max_abs_change_percent")) != magnitude_disclosable(percents) \
            or bool(row["percent_magnitude_suppressed"]) != magnitude_suppressed(percents):
        raise PrivacyViolationError("rule magnitude_suppression: percent magnitudes disagree with their contributors")


def _check_final_suppression(row: dict) -> None:
    """Decrease magnitudes follow decrease contributors: all decreases for cents, percent-valid ones for percents."""
    decreases = row["decrease"]
    cents, percents = row["decrease_cent_contributor_count"], row["decrease_percent_contributor_count"]
    if cents != decreases or percents + row["decrease_zero_denominator_count"] != decreases or percents > cents:
        raise PrivacyViolationError("rule magnitude_suppression: contributor counts disagree with the decreases")
    cent_columns = ("decrease_cents_min", "decrease_cents_median", "decrease_cents_max")
    percent_columns = ("decrease_percent_min", "decrease_percent_median", "decrease_percent_max")
    if _present(row, cent_columns) != magnitude_disclosable(cents) \
            or bool(row["decrease_cent_magnitude_suppressed"]) != magnitude_suppressed(cents):
        raise PrivacyViolationError("rule magnitude_suppression: cent magnitudes disagree with their contributors")
    if _present(row, percent_columns) != magnitude_disclosable(percents) \
            or bool(row["decrease_percent_magnitude_suppressed"]) != magnitude_suppressed(percents):
        raise PrivacyViolationError("rule magnitude_suppression: percent magnitudes disagree with their contributors")


def validate_sanitized_frame(name: str, frame: object, *,
                             approved_locations: Sequence[tuple[str, str]] | None = None) -> None:
    """Refuse a frame that is not exactly an allowlisted, semantically valid sanitized table.

    Structural: approved table name, a DataFrame, no detailed-frame schema, no
    forbidden or unexpected columns, exactly the table's column order.
    Semantic: every value has its column's kind (non-negative integer counts,
    booleans, finite 0..1 shares, finite statistics, canonical periods,
    approved cities/locations/roles, enum members, the documented rule text),
    missing values only where allowed, no provenance label text, no duplicate
    records and status-consistent final-case fields. Messages name the rule
    (and at most a column name), never a value.

    Raises:
        PrivacyViolationError: Any structural or semantic violation.
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
    approved = (frozenset(tuple(k) for k in approved_locations) if approved_locations is not None
                else _approved_locations())
    _semantic(name, frame, approved)


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


def _percent_contributors(candidates: pd.DataFrame) -> Counter:
    """Changed candidates with a nonzero previous price per location interval (counts only)."""
    changed = candidates["outcome"].isin([TerminalOutcome.INCREASE.value, TerminalOutcome.DECREASE.value]) \
        & candidates["percent_valid"].astype(bool)
    rows = candidates.loc[changed, ["canonical_city", "canonical_location", PREV, CUR]]
    return Counter(tuple(r) for r in rows.itertuples(index=False, name=None))


def _interval_summary(table: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    contributors = _percent_contributors(candidates)
    rows = []
    for record in table.to_dict("records"):
        out = {c: record[c] for c in EVENT_TABLE_COLUMNS}
        changes = int(record["price_change_count"])
        percents = contributors[(record["canonical_city"], record["canonical_location"], record[PREV], record[CUR])]
        out["magnitude_suppressed"] = magnitude_suppressed(changes)
        out["change_percent_contributor_count"] = percents
        out["percent_magnitude_suppressed"] = magnitude_suppressed(percents)
        if not magnitude_disclosable(changes):
            out["min_change_cents"] = out["max_change_cents"] = None
        if not magnitude_disclosable(percents):
            out["median_abs_change_percent"] = out["max_abs_change_percent"] = None
        out["material_synchronized"] = bool(record["direction_synchronized"])
        out["interpretation_status"] = _interpretation(record)
        rows.append(out)
    frame = pd.DataFrame(rows, columns=list(_SUMMARY_SCHEMA))
    if not len(frame):
        frame = pd.DataFrame(columns=list(_SUMMARY_SCHEMA))
    return frame.astype({c: bool for c in ("direction_synchronized", "exact_cent_synchronized",
                                           "exact_percent_synchronized", "magnitude_suppressed",
                                           "percent_magnitude_suppressed", "material_synchronized")})


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


def _provenance_categories(case, authority: object) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Fixed aggregate provenance categories of the case (raw label sets never leave this function)."""
    policy = getattr(authority, "policy", None)
    primary = getattr(policy, "first", (None, None))[1]
    secondary = getattr(policy, "second", (None, None))[1]
    composition = dict.fromkeys(_PROVENANCE_CATEGORIES, 0)
    for labels, n in case.provenance:
        parts = set(labels.split("|")) if labels else set()
        if {primary, secondary} <= parts:
            composition["provenance_dual_alias_source"] += n
        elif parts == {primary}:
            composition["provenance_primary_alias_only"] += n
        elif parts == {secondary}:
            composition["provenance_secondary_alias_only"] += n
        else:
            composition["provenance_other_canonical_location"] += n
    return composition


def _final_case(analysis: PriceChangeAnalysisResult) -> pd.DataFrame:
    return final_case_table(analysis.report.final_decrease, analysis.location_authority)


def final_case_table(case: object, location_authority: object = None) -> pd.DataFrame:
    """The sanitized one-record table of a validated ``FinalDecreaseCase`` (no free text, labels or values).

    Each magnitude summary is governed by its own contributing population and
    suppressed here, before the table exists: decrease cents by the decrease
    cent contributors, decrease percentages by the percent-valid decreases.
    ``location_authority`` supplies the governed alias labels for the provenance
    categories (raw labels never leave this function).
    """
    from ql2_sixt_canada_analysis.authority_decisions import LocationRoleDecision
    from ql2_sixt_canada_analysis.price_change_analysis import FinalDecreaseCase

    if not isinstance(case, FinalDecreaseCase):
        raise TypeError("case must be a FinalDecreaseCase")
    row: dict[str, object] = dict.fromkeys(SANITIZED_TABLE_SCHEMAS["final_vancouver_decrease"])
    row["status"] = case.status.value
    row["canonical_city"] = case.canonical_city
    if case.status is FinalDecreaseStatus.DERIVED:
        counts = dict(case.counts)
        roles = {role for _, role, _ in case.locations}
        cents_ok = magnitude_disclosable(case.decrease_cent_contributors)
        percent_ok = magnitude_disclosable(case.decrease_percent_contributors)
        row.update({
            PREV: case.previous_period, CUR: case.current_period, "participating_locations": len(case.locations),
            "airport_involved": LocationRoleDecision.AIRPORT.value in roles,
            "downtown_involved": LocationRoleDecision.DOWNTOWN.value in roles,
            "all_locations_end_at_final_capture": all(bool(final) for _, _, final in case.locations),
            **{o: int(counts.get(o, 0)) for o in _OUTCOMES},
            "comparable": case.comparable, "price_change_count": case.price_change_count,
            "assortment_event_count": case.assortment_event_count,
            "changed_share_of_comparable": case.changed_share_of_comparable,
            "direction_synchronized": case.direction_synchronized,
            "exact_cent_synchronized": case.exact_cent_synchronized,
            "exact_percent_synchronized": case.exact_percent_synchronized,
            "largest_same_cent_cohort": case.largest_same_cent_cohort,
            "largest_same_percent_cohort": case.largest_same_percent_cohort,
            "decrease_cent_contributor_count": case.decrease_cent_contributors,
            "decrease_percent_contributor_count": case.decrease_percent_contributors,
            "decrease_zero_denominator_count": case.decrease_zero_denominator,
            "decrease_cent_magnitude_suppressed": magnitude_suppressed(case.decrease_cent_contributors),
            "decrease_percent_magnitude_suppressed": magnitude_suppressed(case.decrease_percent_contributors),
            # Suppression happens here, before the table exists: withheld values are never emitted.
            **{f"decrease_cents_{k}": float(v) for k, v in case.decrease_cents if cents_ok},
            **{f"decrease_percent_{k}": float(v) for k, v in case.decrease_percent if percent_ok},
            **{f"cross_{o.value}": int(dict(case.cross_location).get(o.value, 0)) for o in CrossLocationOutcome},
            **_provenance_categories(case, location_authority),
            **{f"persistence_{o.value}": int(dict(case.persistence).get(o.value, 0)) for o in PersistenceOutcome},
            **{f"not_testable_{r.value}": int(dict(case.not_testable_reasons).get(r.value, 0))
               for r in NotTestableReason},
            "persistence_testable": case.persistence_testable,
            **{f"indicator_{i.value}": i in case.indicators for i in FinalDecreaseIndicator},
        })
    return pd.DataFrame([row], columns=list(SANITIZED_TABLE_SCHEMAS["final_vancouver_decrease"]), dtype=object)


def _check(rows: list, name: str, expected: int, observed: int) -> None:
    if name not in RECONCILIATION_CHECKS:
        raise PriceChangeReconciliationError("an unregistered reconciliation check")
    rows.append((name, int(expected), int(observed), "reconciled" if int(expected) == int(observed) else "failed"))


def _reconciliation(analysis, summary, material, selection, cross, persistence, heat, final):  # type: ignore[no-untyped-def]
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
    contributors = _percent_contributors(analysis.events.candidates)
    _check(rows, "interval_percent_contributors_equal_percent_valid_changes", sum(contributors.values()),
           int(summary["change_percent_contributor_count"].sum()))
    _check(rows, "heatmap_interval_cells_equal_intervals", len(summary), int((heat["cell_state"] == "interval").sum()))
    case = report.final_decrease
    if case.status is FinalDecreaseStatus.DERIVED:
        sub = summary[(summary["canonical_city"] == case.canonical_city) & (summary[PREV] == case.previous_period)
                      & (summary[CUR] == case.current_period)]
        _check(rows, "final_case_price_changes_subset_of_intervals", case.price_change_count,
               int(sub["price_change_count"].sum()))
        _check(rows, "final_case_locations_subset_of_intervals", len(case.locations), len(sub))
        counts = dict(case.counts)
        _check(rows, "final_case_assortment_events_equal_case",
               counts.get("appeared", 0) + counts.get("disappeared", 0), int(sub["assortment_event_count"].sum()))
        _check(rows, "final_case_persistence_partitions_case_changes", case.price_change_count,
               sum(n for _, n in case.persistence))
        rows_cross = analysis.cross_location[
            (analysis.cross_location["canonical_city"] == case.canonical_city)
            & (analysis.cross_location[PREV] == case.previous_period)
            & (analysis.cross_location[CUR] == case.current_period)]
        _check(rows, "final_case_cross_location_equal_cross_table", sum(n for _, n in case.cross_location),
               len(rows_cross))
        _check(rows, "final_case_provenance_equal_case_candidates", int(sub["candidates"].sum()),
               sum(_provenance_categories(case, analysis.location_authority).values()))
        cand = analysis.events.candidates
        decreases = cand[(cand["canonical_city"] == case.canonical_city) & (cand[PREV] == case.previous_period)
                         & (cand[CUR] == case.current_period) & (cand["outcome"] == TerminalOutcome.DECREASE.value)]
        _check(rows, "final_case_decreases_equal_outcome_counts", counts.get("decrease", 0), int(sub["decrease"].sum()))
        _check(rows, "final_case_cent_contributors_equal_decrease_rows", case.decrease_cent_contributors,
               int(decreases["change_cents"].notna().sum()))
        _check(rows, "final_case_percent_contributors_equal_percent_valid_decreases",
               case.decrease_percent_contributors, int(decreases["percent_valid"].astype(bool).sum()))
        _check(rows, "final_case_zero_denominator_decreases_equal_remaining_decreases",
               case.decrease_zero_denominator, int(decreases["zero_denominator"].astype(bool).sum()))
        presented = final.iloc[0].to_dict()
        expected_values = {**{f"decrease_cents_{k}": float(v) for k, v in case.decrease_cents
                              if magnitude_disclosable(case.decrease_cent_contributors)},
                           **{f"decrease_percent_{k}": float(v) for k, v in case.decrease_percent
                              if magnitude_disclosable(case.decrease_percent_contributors)}}
        magnitude_columns = [f"decrease_{kind}_{stat}" for kind in ("cents", "percent")
                             for stat in ("min", "median", "max")]
        agrees = all((presented[c] == expected_values[c]) if c in expected_values else _missing(presented[c])
                     for c in magnitude_columns)
        _check(rows, "final_case_presented_magnitudes_equal_validated_case", 1, int(agrees))
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
    summary = _interval_summary(analysis.event_table, analysis.events.candidates)
    material, selection = _material(summary)
    cross = _airport_downtown(analysis)
    persistence = _persistence_table(analysis)
    heat = heatmap_source_frame(analysis)
    final = _final_case(analysis)
    recon = _reconciliation(analysis, summary, material, selection, cross, persistence, heat, final)
    if not (recon["status"] == "reconciled").all():
        raise PriceChangeReconciliationError("rule reconciliation: a presentation table does not reconcile")
    return PresentationTables(
        event_interval_summary=summary, material_synchronized_movements=material,
        material_selection_reconciliation=selection, airport_downtown_summary=cross,
        persistence_summary=persistence, final_vancouver_decrease=final,
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
    for name, frame in tables.items():                    # complete validation immediately before export
        validate_sanitized_frame(name, frame)
    recon = tables.reconciliation_summary
    if not (recon["status"] == "reconciled").all() or not (recon["expected"] == recon["observed"]).all():
        raise PriceChangeReconciliationError("rule reconciliation: tables are not reconciled at export")
    directory = _output_directory(output_dir)
    paths = []
    for name, frame in tables.items():
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
