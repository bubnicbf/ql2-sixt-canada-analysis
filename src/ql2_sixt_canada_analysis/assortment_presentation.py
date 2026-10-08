"""Visible-assortment presentation: sanitized aggregate tables, the timeline view and a bounded narrative.

The presentation layer is subordinate to the validated chain
``run_pricing_pipeline`` -> :mod:`~ql2_sixt_canada_analysis.price_change_events`
-> :mod:`~ql2_sixt_canada_analysis.visible_assortment`. It never rebuilds
product sets, never recalculates additions, removals, ratios or price
outcomes, and never builds a second capture grid: every table is a projection,
selection or aggregation of one completed
:class:`~ql2_sixt_canada_analysis.visible_assortment.VisibleAssortmentResult`
and is reconciled back to it before anything is returned, rendered or narrated.

Entry points
------------
* :func:`run_assortment_presentation` - calls ``run_pricing_pipeline`` exactly
  once and passes that one result through the price-change engine, the
  visible-assortment engine and the presentation.
* :func:`assortment_presentation_from_pipeline` - the same for one existing
  pipeline result.
* :func:`build_assortment_presentation_tables`, :func:`build_assortment_narrative`,
  :func:`assortment_timeline_source_frame` and :func:`assortment_timeline_png` -
  pure and in memory.

Sanitized tables
----------------
Two allowlist levels, enforced by :func:`validate_assortment_table`: every
table has an exact ordered schema (:data:`ASSORTMENT_TABLE_SCHEMAS`, union
:data:`ASSORTMENT_COLUMN_ALLOWLIST`), and every allowlisted column has exactly
one semantic kind (:data:`ASSORTMENT_COLUMN_KINDS`): counts, signed counts,
flags, 0..1 shares, non-negative statistics, canonical periods, approved
cities and locations, or members of a fixed enum. There is no free-text kind.
Product identity, rental dates, units, prices, identifiers, provenance, raw
timestamps and paths cannot pass. Ratios and medians appear only with an
explicit contributor count; unavailable values stay missing, never zero.

Observed drops
--------------
The repository default has no approved unusual-drop policy, so default
results carry no classification. ``observed_drop_review`` lists every assessed
interval with ``absolute_drop > 0`` as a **review candidate**, ordered for
review only (largest drop first). It is not an alert table and no row is
classified as a collection failure or as a supplier withdrawal. If an
explicitly approved, executable policy was supplied to the engine, its
classifications (``unusual_drop``) are reported as they are; the presentation
never creates, approves or reruns a policy. The narrative and the figure take
their policy wording from one summary of the validated tables
(:func:`_policy_wording`), so they cannot contradict the tables or each other.
``cross_location_drops`` groups candidates by exact current scheduled period:
one location is *isolated in this extract*, several are *simultaneous in this
extract*. Simultaneity is not proof of a common cause.

Persistence
-----------
Timeline persistence and format are **proposed, not approved** in the
visible-assortment contract. Nothing here writes a file: supplying an output
directory returns a ``persistence_not_approved`` blocked result before any
directory is touched, and no environment variable enables export.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.assortment_contract import (
    ASSORTMENT_BREAK_REASONS,
    ASSORTMENT_TIMELINE_COLUMNS,
    DEFAULT_ASSORTMENT_DEFINITION,
    AnomalyPolicyStatus,
    AssessabilityStatus,
    DefinitionStatus,
    DenominatorStatus,
)
from ql2_sixt_canada_analysis.canonical_offers import APPROVED_PRODUCT_COLUMNS
from ql2_sixt_canada_analysis.price_change_events import (
    CAPTURE_STEP,
    EVENT_INTERVAL_COLUMNS,
    FORBIDDEN_TIMESTAMP_SOURCES,
    CaptureState,
    parse_scheduled_period,
)
from ql2_sixt_canada_analysis.schemas import CONFIDENTIAL_TECHNICAL_COLUMNS
from ql2_sixt_canada_analysis.visible_assortment import (
    MEMBERSHIP_COLUMNS,
    AssortmentReconciliationError,
    VisibleAssortmentResult,
)

__all__ = [
    "ASSORTMENT_COLUMN_ALLOWLIST",
    "ASSORTMENT_COLUMN_KINDS",
    "ASSORTMENT_RECONCILIATION_CHECKS",
    "ASSORTMENT_TABLE_SCHEMAS",
    "FORBIDDEN_ASSORTMENT_COLUMNS",
    "NARRATIVE_SECTIONS",
    "TIMELINE_SOURCE_COLUMNS",
    "AssortmentNarrative",
    "AssortmentPresentationBlocker",
    "AssortmentPresentationReport",
    "AssortmentPresentationResult",
    "AssortmentPresentationStatus",
    "AssortmentPresentationTables",
    "AssortmentPrivacyError",
    "DropPattern",
    "ReviewStatus",
    "Simultaneity",
    "assortment_persistence_approved",
    "assortment_presentation_from_pipeline",
    "assortment_timeline_png",
    "assortment_timeline_source_frame",
    "build_assortment_narrative",
    "build_assortment_presentation_tables",
    "run_assortment_presentation",
    "validate_assortment_table",
]

PREV, CUR = EVENT_INTERVAL_COLUMNS
_CAPTURE = DEFAULT_ASSORTMENT_DEFINITION.capture_column


class AssortmentPrivacyError(ValueError):
    """A presentation table breaches the sanitized allowlist (messages name the rule, never a value)."""


class AssortmentPresentationStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class AssortmentPresentationBlocker(StrEnum):
    ASSORTMENT_NOT_COMPLETED = "assortment_not_completed"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    RECONCILIATION_FAILED = "reconciliation_failed"
    PERSISTENCE_NOT_APPROVED = "persistence_not_approved"


class DropPattern(StrEnum):
    """The shape of one observed drop (descriptive; never a cause)."""

    NET_CONTRACTION = "net_contraction"              # some products retained, the set shrank
    COMPLETE_TURNOVER = "complete_turnover"          # nothing retained, the current set is not empty
    EMPTY_CURRENT_CAPTURE = "empty_current_capture"  # an eligible capture returned no product


class Simultaneity(StrEnum):
    """How many approved locations show an observed drop in the same scheduled period (this extract only)."""

    ISOLATED = "isolated_in_extract"
    SIMULTANEOUS = "simultaneous_in_extract"


class ReviewStatus(StrEnum):
    """The only interpretation the data supports without an approved policy and corroboration."""

    OBSERVED_DROP_REVIEW_CANDIDATE = "observed_drop_review_candidate"


# ------------------------------------------------------------------ schemas

_LOCATION_KEY = ("canonical_city", "canonical_location")
_INTERVAL_METRICS = ("previous_product_count", "retained_count", "addition_count", "removal_count", "retention",
                     "jaccard_similarity", "net_change", "absolute_drop", "drop_rate", "price_increase_count",
                     "price_decrease_count", "assortment_change", "price_change", "assortment_price_coincidence",
                     "falling_assortment_with_price_increase")
_RATIOS = (("retention", "retention_denominator_status"), ("jaccard_similarity", "jaccard_denominator_status"),
           ("drop_rate", "drop_rate_denominator_status"))

_LOCATION_SUMMARY = (
    *_LOCATION_KEY, "scheduled_captures", "eligible_captures", "excluded_captures", "missing_captures",
    "empty_captures", "seed_captures", "assessed_intervals", "break_captures", "interval_breaks",
    "returned_count_contributors", "returned_count_min", "returned_count_median", "returned_count_max",
    "addition_intervals", "removal_intervals", "observed_drop_intervals", "total_additions", "total_removals",
    "retention_contributors", "retention_zero_denominator", "retention_median", "jaccard_contributors",
    "jaccard_zero_denominator", "jaccard_median", "price_change_intervals", "price_increase_intervals",
    "price_decrease_intervals", "coincident_intervals", "falling_with_increase_intervals", "anomaly_policy_status")
_DROP_REVIEW = (
    *_LOCATION_KEY, PREV, _CAPTURE, "previous_product_count", "returned_product_count", "retained_count",
    "addition_count", "removal_count", "retention", "retention_denominator_status", "jaccard_similarity",
    "jaccard_denominator_status", "net_change", "absolute_drop", "drop_rate", "drop_rate_denominator_status",
    "price_increase_count", "price_decrease_count", "assortment_price_coincidence",
    "falling_assortment_with_price_increase", "drop_pattern", "locations_with_drop_in_period", "simultaneity",
    "anomaly_policy_status", "unusual_drop", "review_status")
_CROSS = (PREV, _CAPTURE, "assessed_locations", "locations_with_observed_drop", "observed_drop_total",
          "simultaneity")
_COINCIDENCE = (
    *_LOCATION_KEY, "assessed_intervals", "assortment_change_intervals", "price_change_intervals",
    "price_increase_intervals", "price_decrease_intervals", "coincident_intervals", "observed_drop_intervals",
    "drop_with_price_increase_intervals", "drop_with_price_decrease_intervals", "falling_with_increase_intervals",
    "coincidence_share_of_assortment_changes", "coincidence_share_of_price_changes")
_RECONCILIATION = ("check", "expected", "observed", "status")

#: The exact, ordered schema of every sanitized presentation table.
ASSORTMENT_TABLE_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "assortment_timeline": ASSORTMENT_TIMELINE_COLUMNS,
    "location_summary": _LOCATION_SUMMARY,
    "observed_drop_review": _DROP_REVIEW,
    "cross_location_drops": _CROSS,
    "price_coincidence_summary": _COINCIDENCE,
    "reconciliation_summary": _RECONCILIATION,
}
ASSORTMENT_COLUMN_ALLOWLIST: frozenset[str] = frozenset(c for cols in ASSORTMENT_TABLE_SCHEMAS.values() for c in cols)

#: Columns refused in any sanitized table, by rule (checked before the allowlist).
FORBIDDEN_ASSORTMENT_COLUMNS: Mapping[str, frozenset[str]] = {
    "product_identity": frozenset(APPROVED_PRODUCT_COLUMNS),
    "rental_context": frozenset({"pickup_date", "return_date"}),
    "price_unit": frozenset({"currency", "price_basis"}),
    "price": frozenset({"price_cents", "price_num", "price_per_day", "previous_price_cents", "current_price_cents",
                        "change_cents", "previous_price", "current_price", "change_dollars", "change_percent"}),
    "identifier": frozenset({*CONFIDENTIAL_TECHNICAL_COLUMNS, "parent_key", "offer_position"}),
    "provenance": frozenset({"source_location_labels", "previous_source_labels", "current_source_labels",
                             "provenance", "observation_count", "price_variation", "city", "location"}),
    "raw_timestamp": frozenset(FORBIDDEN_TIMESTAMP_SOURCES) - frozenset(CONFIDENTIAL_TECHNICAL_COLUMNS),
    "product_membership": frozenset({"membership"}),
    "path": frozenset({"path", "file", "filename", "raw_dir", "output_dir"}),
}


class _Kind(StrEnum):
    COUNT = "count"                 # non-negative integer, never a boolean
    SIGNED_COUNT = "signed_count"   # integer (net change)
    FLAG = "flag"                   # boolean
    SHARE = "share"                 # finite, 0..1
    STATISTIC = "statistic"         # finite, non-negative aggregate statistic (a median of counts)
    PERIOD = "period"               # canonical YYYYMMDDTHHMMSSZ scheduled period
    CITY = "city"                   # an approved canonical city
    LOCATION = "location"           # an approved canonical location (paired with its city)
    ENUM = "enum"                   # a member of the column's enum


def _values(enum: type[StrEnum]) -> frozenset[str]:
    return frozenset(m.value for m in enum)


RECONCILIATION_STATUS = frozenset({"reconciled", "failed"})

#: Every reconciliation check, in order (expected versus observed; all must reconcile).
ASSORTMENT_RECONCILIATION_CHECKS: tuple[str, ...] = (
    "timeline_rows_equal_scheduled_captures",
    "timeline_keys_unique",
    "assessed_rows_equal_engine_intervals",
    "previous_count_equals_retained_plus_removed",
    "current_count_equals_retained_plus_added",
    "jaccard_union_equals_retained_plus_added_plus_removed",
    "net_change_equals_additions_minus_removals",
    "absolute_drop_equals_negative_net_change_floor",
    "observed_drop_rows_equal_assessed_drops",
    "retention_contributors_plus_zero_denominators_equal_assessed",
    "jaccard_contributors_plus_zero_denominators_equal_assessed",
    "price_changes_equal_increases_plus_decreases",
    "coincidence_rows_subset_of_assortment_and_price_changes",
    "falling_rows_subset_of_drops_and_price_increases",
    "location_summaries_equal_engine_counts",
    "location_assessed_totals_equal_overall",
    "location_addition_totals_equal_overall",
    "location_removal_totals_equal_overall",
    "location_drop_totals_equal_overall",
    "non_assessed_rows_hold_no_interval_metrics",
    "unusual_drop_matches_policy_status",
    "cross_location_drops_equal_review_rows",
    "price_coincidence_totals_equal_timeline",
    "single_evidence_run",
)

_ENUMS: Mapping[str, frozenset[str]] = {
    "capture_state": _values(CaptureState),
    "assessability_status": _values(AssessabilityStatus) - {AssessabilityStatus.BLOCKED.value},
    "interval_break_reason": frozenset(ASSORTMENT_BREAK_REASONS),
    "anomaly_policy_status": _values(AnomalyPolicyStatus),
    "retention_denominator_status": _values(DenominatorStatus),
    "jaccard_denominator_status": _values(DenominatorStatus),
    "drop_rate_denominator_status": _values(DenominatorStatus),
    "drop_pattern": _values(DropPattern),
    "simultaneity": _values(Simultaneity),
    "review_status": _values(ReviewStatus),
    "check": frozenset(ASSORTMENT_RECONCILIATION_CHECKS),
    "status": RECONCILIATION_STATUS,
}
_FLAGS = frozenset({"has_previous_interval", "assortment_change", "price_change", "assortment_price_coincidence",
                    "falling_assortment_with_price_increase", "unusual_drop"})
_SHARES = frozenset({"retention", "jaccard_similarity", "drop_rate", "coincidence_share_of_assortment_changes",
                     "coincidence_share_of_price_changes"})
_STATISTICS = frozenset({"returned_count_min", "returned_count_median", "returned_count_max", "retention_median",
                         "jaccard_median"})
_PERIODS = frozenset({PREV, _CAPTURE})


def _kind(column: str) -> _Kind:
    if column in _FLAGS:
        return _Kind.FLAG
    if column in _SHARES:
        return _Kind.SHARE
    if column in _STATISTICS:
        return _Kind.STATISTIC
    if column in _PERIODS:
        return _Kind.PERIOD
    if column == "canonical_city":
        return _Kind.CITY
    if column == "canonical_location":
        return _Kind.LOCATION
    if column in _ENUMS:
        return _Kind.ENUM
    if column == "net_change":
        return _Kind.SIGNED_COUNT
    return _Kind.COUNT


#: The semantic kind of every allowlisted column (one kind per column, across every table).
ASSORTMENT_COLUMN_KINDS: Mapping[str, str] = {c: _kind(c).value for c in sorted(ASSORTMENT_COLUMN_ALLOWLIST)}

#: Columns that may be missing, per table (every other value must be present).
_NULLABLE: Mapping[str, frozenset[str]] = {
    "assortment_timeline": frozenset({"returned_product_count", PREV, "interval_break_reason", "unusual_drop",
                                      *_INTERVAL_METRICS}),
    "location_summary": frozenset({"returned_count_min", "returned_count_median", "returned_count_max",
                                   "retention_median", "jaccard_median"}),
    "observed_drop_review": frozenset({"jaccard_similarity", "unusual_drop"}),
    "cross_location_drops": frozenset(),
    "price_coincidence_summary": frozenset({"coincidence_share_of_assortment_changes",
                                            "coincidence_share_of_price_changes"}),
    "reconciliation_summary": frozenset(),
}
_KEYS: Mapping[str, tuple[str, ...]] = {
    "assortment_timeline": (*_LOCATION_KEY, _CAPTURE), "location_summary": _LOCATION_KEY,
    "observed_drop_review": (*_LOCATION_KEY, _CAPTURE), "cross_location_drops": (_CAPTURE,),
    "price_coincidence_summary": _LOCATION_KEY, "reconciliation_summary": ("check",),
}


def _missing(value: object) -> bool:
    return value is None or value is pd.NA or value is pd.NaT or (isinstance(value, float) and math.isnan(value))


def _int(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))


def _finite(value: object) -> bool:
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value)))


def _value_ok(column: str, value: object, approved: frozenset[tuple[str, str]]) -> bool:
    kind = _kind(column)
    if kind is _Kind.COUNT:
        return _int(value) and int(value) >= 0
    if kind is _Kind.SIGNED_COUNT:
        return _int(value)
    if kind is _Kind.FLAG:
        return isinstance(value, (bool, np.bool_))
    if kind is _Kind.SHARE:
        return _finite(value) and 0.0 <= float(value) <= 1.0
    if kind is _Kind.STATISTIC:
        return _finite(value) and float(value) >= 0.0
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
    return value in _ENUMS[column]


def _fail(rule: str, column: str | None = None) -> AssortmentPrivacyError:
    return AssortmentPrivacyError(f"rule {rule}" + (f": column {column}" if column else ""))


def _check_ratio_status(row: dict) -> None:
    for metric, status in _RATIOS:
        if status not in row:
            continue
        defined = row[status] == DenominatorStatus.DEFINED.value
        if defined == _missing(row[metric]):
            raise _fail("denominator_status", metric)


def _check_unusual(row: dict, assessed: bool) -> None:
    approved = row["anomaly_policy_status"] == AnomalyPolicyStatus.APPROVED.value
    if (approved and assessed) == _missing(row["unusual_drop"]):
        raise _fail("unusual_drop_policy", "unusual_drop")


def _row_rules(name: str, row: dict) -> None:
    if name == "assortment_timeline":
        assessed = row["assessability_status"] == AssessabilityStatus.ASSESSED.value
        eligible = row["capture_state"] == CaptureState.ELIGIBLE.value
        if eligible == _missing(row["returned_product_count"]):
            raise _fail("eligible_count", "returned_product_count")
        if assessed:
            if any(_missing(row[c]) for c in (PREV, *_INTERVAL_METRICS) if c not in dict(_RATIOS)):
                raise _fail("assessed_metrics")
            if not bool(row["has_previous_interval"]) or not _missing(row["interval_break_reason"]):
                raise _fail("assessed_metrics")
            if row["retained_count"] + row["removal_count"] != row["previous_product_count"] \
                    or row["retained_count"] + row["addition_count"] != row["returned_product_count"] \
                    or row["net_change"] != row["addition_count"] - row["removal_count"] \
                    or row["absolute_drop"] != max(-row["net_change"], 0):
                raise _fail("interval_accounting")
            _check_ratio_status(row)
        else:
            if bool(row["has_previous_interval"]) or any(not _missing(row[c]) for c in (PREV, *_INTERVAL_METRICS)):
                raise _fail("break_metrics")
            if any(row[s] != DenominatorStatus.NOT_ASSESSABLE.value for _, s in _RATIOS):
                raise _fail("break_metrics")
        _check_unusual(row, assessed)
    elif name == "observed_drop_review":
        if not row["absolute_drop"] > 0 or row["previous_product_count"] <= 0 \
                or row["retained_count"] + row["removal_count"] != row["previous_product_count"] \
                or row["retained_count"] + row["addition_count"] != row["returned_product_count"] \
                or row["net_change"] != row["addition_count"] - row["removal_count"] \
                or row["absolute_drop"] != -row["net_change"]:
            raise _fail("observed_drop_accounting")
        _check_ratio_status(row)
        if row["retention_denominator_status"] != DenominatorStatus.DEFINED.value \
                or row["drop_rate_denominator_status"] != DenominatorStatus.DEFINED.value:
            raise _fail("denominator_status")
        if row["drop_pattern"] != _drop_pattern(row).value:
            raise _fail("drop_pattern")
        expected = Simultaneity.ISOLATED if row["locations_with_drop_in_period"] == 1 else Simultaneity.SIMULTANEOUS
        if row["locations_with_drop_in_period"] < 1 or row["simultaneity"] != expected.value:
            raise _fail("simultaneity")
        _check_unusual(row, True)
    elif name == "location_summary":
        if row["scheduled_captures"] != row["eligible_captures"] + row["excluded_captures"] + row["missing_captures"]:
            raise _fail("capture_partition")
        if row["eligible_captures"] != row["seed_captures"] + row["assessed_intervals"] + row["break_captures"]:
            raise _fail("eligible_partition")
        if row["returned_count_contributors"] != row["eligible_captures"]:
            raise _fail("contributors", "returned_count_contributors")
        for columns, contributors in ((("returned_count_min", "returned_count_median", "returned_count_max"),
                                       row["returned_count_contributors"]),
                                      (("retention_median",), row["retention_contributors"]),
                                      (("jaccard_median",), row["jaccard_contributors"])):
            if any(_missing(row[c]) == (contributors > 0) for c in columns):
                raise _fail("contributors", columns[0])
        for metric in ("retention", "jaccard"):
            if row[f"{metric}_contributors"] + row[f"{metric}_zero_denominator"] != row["assessed_intervals"]:
                raise _fail("contributors", f"{metric}_contributors")
        if row["coincident_intervals"] > min(row["addition_intervals"] + row["removal_intervals"],
                                             row["price_change_intervals"]):
            raise _fail("coincidence_subset")
        for share in ("retention_median", "jaccard_median"):
            if not _missing(row[share]) and not 0.0 <= float(row[share]) <= 1.0:
                raise _fail("value_domain", share)
    elif name == "cross_location_drops":
        n = row["locations_with_observed_drop"]
        expected = Simultaneity.ISOLATED if n == 1 else Simultaneity.SIMULTANEOUS
        if n < 1 or n > row["assessed_locations"] or row["simultaneity"] != expected.value \
                or row["observed_drop_total"] < n:
            raise _fail("simultaneity")
        if parse_scheduled_period(row[_CAPTURE]) - parse_scheduled_period(row[PREV]) != CAPTURE_STEP:
            raise _fail("interval")
    elif name == "price_coincidence_summary":
        for share, numerator, denominator in (
                ("coincidence_share_of_assortment_changes", "coincident_intervals", "assortment_change_intervals"),
                ("coincidence_share_of_price_changes", "coincident_intervals", "price_change_intervals")):
            if row[denominator]:
                if _missing(row[share]) or float(row[share]) != row[numerator] / row[denominator]:
                    raise _fail("share_formula", share)
            elif not _missing(row[share]):
                raise _fail("share_formula", share)
        if row["coincident_intervals"] > min(row["assortment_change_intervals"], row["price_change_intervals"]) \
                or row["falling_with_increase_intervals"] > min(row["drop_with_price_increase_intervals"],
                                                                row["observed_drop_intervals"]) \
                or row["drop_with_price_increase_intervals"] > row["price_increase_intervals"] \
                or row["drop_with_price_decrease_intervals"] > row["price_decrease_intervals"] \
                or row["price_change_intervals"] > row["assessed_intervals"] \
                or row["assortment_change_intervals"] > row["assessed_intervals"]:
            raise _fail("coincidence_subset")
    elif name == "reconciliation_summary":
        if (row["expected"] == row["observed"]) != (row["status"] == "reconciled"):
            raise _fail("reconciliation_status")


def validate_assortment_table(name: str, frame: object, *,
                              approved_locations: Sequence[tuple[str, str]] | None = None) -> None:
    """Refuse a frame that is not exactly an allowlisted, semantically valid sanitized assortment table.

    Structural: an approved table name, a DataFrame, no forbidden column, no
    column outside the allowlist, exactly the table's column order.
    Semantic: every value has its column's kind; missing values only where the
    table allows them; approved city/location pairs; enum members; ratio
    presence agrees with the denominator status; break and ineligible rows hold
    no interval metrics; ``unusual_drop`` is empty unless an approved policy
    classified an assessed interval; table accounting rules; unique keys.
    Messages name the rule and at most a column, never a value.

    Raises:
        AssortmentPrivacyError: Any violation.
    """
    if name not in ASSORTMENT_TABLE_SCHEMAS:
        raise _fail("unknown_table")
    if not isinstance(frame, pd.DataFrame):
        raise _fail("not_a_table")
    columns = [str(c) for c in frame.columns]
    if set(MEMBERSHIP_COLUMNS) <= set(columns):
        raise _fail("detailed_frame")
    for rule, forbidden in FORBIDDEN_ASSORTMENT_COLUMNS.items():
        if set(columns) & forbidden:
            raise _fail(rule)
    if set(columns) - ASSORTMENT_COLUMN_ALLOWLIST:
        raise _fail("allowlist")
    if tuple(columns) != ASSORTMENT_TABLE_SCHEMAS[name]:
        raise _fail("schema")
    if approved_locations is None:
        from ql2_sixt_canada_analysis.location_authority import current_location_authority

        authority = current_location_authority()
        approved = frozenset(tuple(authority.canonical(k)) for k in authority.contract.expected_keys)
    else:
        approved = frozenset(tuple(k) for k in approved_locations)
    nullable = _NULLABLE[name]
    for record in frame.astype(object).itertuples(index=False, name=None):
        row = dict(zip(columns, record))
        for column, value in row.items():
            if _missing(value):
                if column not in nullable:
                    raise _fail("missing_value", column)
                continue
            if not _value_ok(column, value, approved):
                raise _fail("value_domain", column)
        if "canonical_location" in row and (row["canonical_city"], row["canonical_location"]) not in approved:
            raise _fail("value_domain", "canonical_location")
        _row_rules(name, row)
    if frame.duplicated(list(_KEYS[name])).any():
        raise _fail("duplicate_record")


# ------------------------------------------------------------------ tables


@dataclass(frozen=True)
class AssortmentPresentationTables:
    """The sanitized aggregate tables of one assortment result (in memory; confidential local results)."""

    assortment_timeline: pd.DataFrame = field(repr=False, compare=False)
    location_summary: pd.DataFrame = field(repr=False, compare=False)
    observed_drop_review: pd.DataFrame = field(repr=False, compare=False)
    cross_location_drops: pd.DataFrame = field(repr=False, compare=False)
    price_coincidence_summary: pd.DataFrame = field(repr=False, compare=False)
    reconciliation_summary: pd.DataFrame = field(repr=False, compare=False)
    approved_locations: tuple[tuple[str, str], ...] = ()
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if not self.approved_locations:
            raise AssortmentReconciliationError("presentation tables name their approved locations")
        for name in ASSORTMENT_TABLE_SCHEMAS:
            validate_assortment_table(name, getattr(self, name), approved_locations=self.approved_locations)
        recon = self.reconciliation_summary
        if tuple(recon["check"]) != ASSORTMENT_RECONCILIATION_CHECKS or not (recon["status"] == "reconciled").all():
            raise AssortmentReconciliationError("presentation tables are not reconciled")
        _policy_summary(self)                                   # cross-table policy and classification agreement

    def items(self) -> tuple[tuple[str, pd.DataFrame], ...]:
        return tuple((name, getattr(self, name)) for name in ASSORTMENT_TABLE_SCHEMAS)


def _require(result: object) -> VisibleAssortmentResult:
    if not isinstance(result, VisibleAssortmentResult):
        raise TypeError("expected a VisibleAssortmentResult")
    if not result.completed:
        raise AssortmentReconciliationError("rule reconciliation: the assortment result is not completed")
    return result


def _evidence_id(result: VisibleAssortmentResult) -> str:
    """A short non-reversible identifier of the bound evidence (no paths, values or keys)."""
    binding = result.binding
    text = f"{getattr(binding, 'parent_rows', '')}:{getattr(binding, 'detail_rows', '')}:" \
           f"{getattr(binding, 'digest', '')}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: (None if _missing(v) else v) for k, v in zip(frame.columns, r)}
            for r in frame.astype(object).itertuples(index=False, name=None)]


def _drop_pattern(row: dict) -> DropPattern:
    if row["returned_product_count"] == 0:
        return DropPattern.EMPTY_CURRENT_CAPTURE
    if row["retained_count"] == 0:
        return DropPattern.COMPLETE_TURNOVER
    return DropPattern.NET_CONTRACTION


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _location_summary(result: VisibleAssortmentResult, rows: list[dict]) -> pd.DataFrame:
    out = []
    policy = result.report.anomaly_policy_status.value
    for summary in result.report.locations:
        key = summary.canonical_location
        mine = [r for r in rows if (r["canonical_city"], r["canonical_location"]) == key]
        assessed = [r for r in mine if r["assessability_status"] == AssessabilityStatus.ASSESSED.value]
        returned = [r["returned_product_count"] for r in mine if r["returned_product_count"] is not None]
        retention = [r["retention"] for r in assessed if r["retention"] is not None]
        jaccard = [r["jaccard_similarity"] for r in assessed if r["jaccard_similarity"] is not None]
        c = summary.counts
        out.append({
            "canonical_city": key[0], "canonical_location": key[1],
            "scheduled_captures": sum(1 for _ in mine),
            "eligible_captures": sum(r["capture_state"] == CaptureState.ELIGIBLE.value for r in mine),
            "excluded_captures": sum(r["capture_state"] == CaptureState.GOVERNED_EXCLUSION.value for r in mine),
            "missing_captures": sum(r["capture_state"] == CaptureState.MISSING_CAPTURE.value for r in mine),
            "empty_captures": sum(v == 0 for v in returned),
            "seed_captures": sum(r["assessability_status"] == AssessabilityStatus.SEED_CAPTURE.value for r in mine),
            "assessed_intervals": len(assessed),
            "break_captures": sum(r["assessability_status"] == AssessabilityStatus.INTERVAL_BREAK.value
                                  for r in mine),
            "interval_breaks": sum(r["interval_break_reason"] is not None for r in mine),
            "returned_count_contributors": len(returned),
            "returned_count_min": float(min(returned)) if returned else None,
            "returned_count_median": _median(returned),
            "returned_count_max": float(max(returned)) if returned else None,
            "addition_intervals": sum(r["addition_count"] > 0 for r in assessed),
            "removal_intervals": sum(r["removal_count"] > 0 for r in assessed),
            "observed_drop_intervals": sum(r["absolute_drop"] > 0 for r in assessed),
            "total_additions": sum(r["addition_count"] for r in assessed),
            "total_removals": sum(r["removal_count"] for r in assessed),
            "retention_contributors": len(retention),
            "retention_zero_denominator": sum(r["retention_denominator_status"]
                                              == DenominatorStatus.ZERO_DENOMINATOR.value for r in assessed),
            "retention_median": _median(retention),
            "jaccard_contributors": len(jaccard),
            "jaccard_zero_denominator": sum(r["jaccard_denominator_status"]
                                            == DenominatorStatus.ZERO_DENOMINATOR.value for r in assessed),
            "jaccard_median": _median(jaccard),
            "price_change_intervals": sum(bool(r["price_change"]) for r in assessed),
            "price_increase_intervals": sum(r["price_increase_count"] > 0 for r in assessed),
            "price_decrease_intervals": sum(r["price_decrease_count"] > 0 for r in assessed),
            "coincident_intervals": sum(bool(r["assortment_price_coincidence"]) for r in assessed),
            "falling_with_increase_intervals": sum(bool(r["falling_assortment_with_price_increase"])
                                                   for r in assessed),
            "anomaly_policy_status": policy})
        if out[-1]["assessed_intervals"] != c.assessed_intervals:                  # pragma: no cover - guarded twice
            raise AssortmentReconciliationError("rule reconciliation: location summary disagrees with the engine")
    return _typed(pd.DataFrame(out, columns=list(_LOCATION_SUMMARY)), "location_summary")


def _drop_review(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assessed = [r for r in rows if r["assessability_status"] == AssessabilityStatus.ASSESSED.value]
    drops = [r for r in assessed if r["absolute_drop"] > 0]
    per_period: dict[str, list[dict]] = {}
    for r in drops:
        per_period.setdefault(r[_CAPTURE], []).append(r)
    assessed_per_period: dict[str, int] = {}
    for r in assessed:
        assessed_per_period[r[_CAPTURE]] = assessed_per_period.get(r[_CAPTURE], 0) + 1
    review = []
    for r in drops:
        n = len(per_period[r[_CAPTURE]])
        record = {c: r.get(c) for c in _DROP_REVIEW if c in r}
        record.update(drop_pattern=_drop_pattern(r).value, locations_with_drop_in_period=n,
                      simultaneity=(Simultaneity.ISOLATED if n == 1 else Simultaneity.SIMULTANEOUS).value,
                      review_status=ReviewStatus.OBSERVED_DROP_REVIEW_CANDIDATE.value)
        review.append(record)
    order = {k: i for i, k in enumerate(dict.fromkeys((r["canonical_city"], r["canonical_location"]) for r in rows))}
    review.sort(key=lambda r: (-r["absolute_drop"], -r["drop_rate"], order[(r["canonical_city"],
                                                                            r["canonical_location"])],
                               parse_scheduled_period(r[_CAPTURE])))
    cross = []
    for period in sorted(per_period, key=parse_scheduled_period):
        members = per_period[period]
        cross.append({PREV: members[0][PREV], _CAPTURE: period, "assessed_locations": assessed_per_period[period],
                      "locations_with_observed_drop": len(members),
                      "observed_drop_total": sum(m["absolute_drop"] for m in members),
                      "simultaneity": (Simultaneity.ISOLATED if len(members) == 1
                                       else Simultaneity.SIMULTANEOUS).value})
    return (_typed(pd.DataFrame(review, columns=list(_DROP_REVIEW)), "observed_drop_review"),
            _typed(pd.DataFrame(cross, columns=list(_CROSS)), "cross_location_drops"))


def _coincidence(result: VisibleAssortmentResult, rows: list[dict]) -> pd.DataFrame:
    out = []
    for summary in result.report.locations:
        key = summary.canonical_location
        a = [r for r in rows if (r["canonical_city"], r["canonical_location"]) == key
             and r["assessability_status"] == AssessabilityStatus.ASSESSED.value]
        changes = sum(bool(r["assortment_change"]) for r in a)
        prices = sum(bool(r["price_change"]) for r in a)
        both = sum(bool(r["assortment_price_coincidence"]) for r in a)
        out.append({
            "canonical_city": key[0], "canonical_location": key[1], "assessed_intervals": len(a),
            "assortment_change_intervals": changes, "price_change_intervals": prices,
            "price_increase_intervals": sum(r["price_increase_count"] > 0 for r in a),
            "price_decrease_intervals": sum(r["price_decrease_count"] > 0 for r in a),
            "coincident_intervals": both,
            "observed_drop_intervals": sum(r["absolute_drop"] > 0 for r in a),
            "drop_with_price_increase_intervals": sum(r["absolute_drop"] > 0 and r["price_increase_count"] > 0
                                                      for r in a),
            "drop_with_price_decrease_intervals": sum(r["absolute_drop"] > 0 and r["price_decrease_count"] > 0
                                                      for r in a),
            "falling_with_increase_intervals": sum(bool(r["falling_assortment_with_price_increase"]) for r in a),
            "coincidence_share_of_assortment_changes": both / changes if changes else None,
            "coincidence_share_of_price_changes": both / prices if prices else None})
    return _typed(pd.DataFrame(out, columns=list(_COINCIDENCE)), "price_coincidence_summary")


def _typed(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Fixed dtypes: integers for counts, nullable floats for statistics and shares, strings as objects."""
    frame = frame.astype(object)
    for column in frame.columns:
        kind = _kind(column)
        if kind in (_Kind.COUNT, _Kind.SIGNED_COUNT):
            frame[column] = frame[column].astype("Int64" if column in _NULLABLE[name] else "int64")
        elif kind in (_Kind.SHARE, _Kind.STATISTIC):
            frame[column] = frame[column].astype("Float64")
        elif kind is _Kind.FLAG:
            frame[column] = frame[column].astype("boolean" if column in _NULLABLE[name] else bool)
    return frame


def _check(rows: list, name: str, expected: int, observed: int) -> None:
    if name not in ASSORTMENT_RECONCILIATION_CHECKS:
        raise AssortmentReconciliationError("an unregistered reconciliation check")
    rows.append((name, int(expected), int(observed), "reconciled" if int(expected) == int(observed) else "failed"))


def _reconciliation(result: VisibleAssortmentResult, rows: list[dict], summary: pd.DataFrame,
                    review: pd.DataFrame, cross: pd.DataFrame, coincidence: pd.DataFrame) -> pd.DataFrame:
    report, overall = result.report, result.report.overall
    timeline = result.timeline
    assessed = [r for r in rows if r["assessability_status"] == AssessabilityStatus.ASSESSED.value]
    others = [r for r in rows if r["assessability_status"] != AssessabilityStatus.ASSESSED.value]
    out: list = []
    _check(out, "timeline_rows_equal_scheduled_captures", overall.scheduled_captures, len(rows))
    _check(out, "timeline_keys_unique", len(timeline),
           len(timeline.drop_duplicates(list(_KEYS["assortment_timeline"]))))
    _check(out, "assessed_rows_equal_engine_intervals", overall.assessed_intervals, len(assessed))
    _check(out, "previous_count_equals_retained_plus_removed", len(assessed),
           sum(r["previous_product_count"] == r["retained_count"] + r["removal_count"] for r in assessed))
    _check(out, "current_count_equals_retained_plus_added", len(assessed),
           sum(r["returned_product_count"] == r["retained_count"] + r["addition_count"] for r in assessed))
    defined = [r for r in assessed if r["jaccard_similarity"] is not None]
    _check(out, "jaccard_union_equals_retained_plus_added_plus_removed", len(defined),
           sum(r["jaccard_similarity"] == r["retained_count"] / (r["retained_count"] + r["addition_count"]
                                                                 + r["removal_count"]) for r in defined))
    _check(out, "net_change_equals_additions_minus_removals", len(assessed),
           sum(r["net_change"] == r["addition_count"] - r["removal_count"] for r in assessed))
    _check(out, "absolute_drop_equals_negative_net_change_floor", len(assessed),
           sum(r["absolute_drop"] == max(-r["net_change"], 0) for r in assessed))
    _check(out, "observed_drop_rows_equal_assessed_drops", sum(r["absolute_drop"] > 0 for r in assessed),
           len(review))
    D = DenominatorStatus
    for metric, status in (("retention", "retention_denominator_status"),
                           ("jaccard", "jaccard_denominator_status")):
        _check(out, f"{metric}_contributors_plus_zero_denominators_equal_assessed", len(assessed),
               sum(r[status] in (D.DEFINED.value, D.ZERO_DENOMINATOR.value) for r in assessed))
    events = result.price_changes.report.overall if result.price_changes is not None else None
    _check(out, "price_changes_equal_increases_plus_decreases",
           events.increase + events.decrease if events is not None else -1,
           sum(r["price_increase_count"] + r["price_decrease_count"] for r in assessed))
    coincident = [r for r in assessed if r["assortment_price_coincidence"]]
    _check(out, "coincidence_rows_subset_of_assortment_and_price_changes", len(coincident),
           sum(bool(r["assortment_change"]) and bool(r["price_change"]) for r in coincident))
    falling = [r for r in assessed if r["falling_assortment_with_price_increase"]]
    _check(out, "falling_rows_subset_of_drops_and_price_increases", len(falling),
           sum(r["absolute_drop"] > 0 and r["price_increase_count"] > 0 for r in falling))
    by_key = {(r["canonical_city"], r["canonical_location"]): r for r in _records(summary)}
    matches = 0
    for s in report.locations:
        p, c = by_key[s.canonical_location], s.counts
        matches += all((p["scheduled_captures"] == c.scheduled_captures, p["eligible_captures"] == c.eligible_captures,
                        p["excluded_captures"] == c.excluded_captures, p["missing_captures"] == c.missing_captures,
                        p["empty_captures"] == c.empty_captures, p["seed_captures"] == c.seed_captures,
                        p["assessed_intervals"] == c.assessed_intervals, p["break_captures"] == c.break_captures,
                        p["interval_breaks"] == sum(n for _, n in s.breaks), p["total_additions"] == c.additions,
                        p["total_removals"] == c.removals, p["observed_drop_intervals"] == c.drop_intervals,
                        p["coincident_intervals"] == c.coincident_intervals,
                        p["retention_zero_denominator"] == c.retention_zero_denominator,
                        p["jaccard_zero_denominator"] == c.jaccard_zero_denominator,
                        p["falling_with_increase_intervals"] == c.falling_with_increase_intervals))
    _check(out, "location_summaries_equal_engine_counts", len(report.locations), matches)
    _check(out, "location_assessed_totals_equal_overall", overall.assessed_intervals,
           int(summary["assessed_intervals"].sum()))
    _check(out, "location_addition_totals_equal_overall", overall.additions, int(summary["total_additions"].sum()))
    _check(out, "location_removal_totals_equal_overall", overall.removals, int(summary["total_removals"].sum()))
    _check(out, "location_drop_totals_equal_overall", overall.drop_intervals,
           int(summary["observed_drop_intervals"].sum()))
    _check(out, "non_assessed_rows_hold_no_interval_metrics", len(others),
           sum(all(r[c] is None for c in (PREV, *_INTERVAL_METRICS)) for r in others))
    if report.anomaly_policy_status is AnomalyPolicyStatus.APPROVED:
        _check(out, "unusual_drop_matches_policy_status", report.unusual_drop_intervals or 0,
               sum(bool(r["unusual_drop"]) for r in assessed))
    else:
        _check(out, "unusual_drop_matches_policy_status", len(rows), sum(r["unusual_drop"] is None for r in rows))
    _check(out, "cross_location_drops_equal_review_rows", len(review),
           int(cross["locations_with_observed_drop"].sum()))
    _check(out, "price_coincidence_totals_equal_timeline", overall.coincident_intervals,
           int(coincidence["coincident_intervals"].sum()))
    same = (result.price_changes is not None and result.price_changes.binding == result.binding
            and result.price_changes.timelines == result.timelines
            and result.price_changes.location_authority is result.location_authority)
    _check(out, "single_evidence_run", 1, int(same))
    return pd.DataFrame(out, columns=list(_RECONCILIATION)).astype(
        {"expected": "int64", "observed": "int64"})


def build_assortment_presentation_tables(result: VisibleAssortmentResult) -> AssortmentPresentationTables:
    """Every sanitized aggregate table of one completed assortment result (pure; nothing is written).

    Raises:
        AssortmentReconciliationError: The result is not completed or a table does not reconcile.
        AssortmentPrivacyError: A table breaches the sanitized allowlist.
    """
    result = _require(result)
    timeline = result.timeline.copy()
    rows = _records(timeline)
    summary = _location_summary(result, rows)
    review, cross = _drop_review(rows)
    coincidence = _coincidence(result, rows)
    recon = _reconciliation(result, rows, summary, review, cross, coincidence)
    if not (recon["status"] == "reconciled").all():
        raise AssortmentReconciliationError("rule reconciliation: a presentation table does not reconcile")
    return AssortmentPresentationTables(
        assortment_timeline=timeline, location_summary=summary, observed_drop_review=review,
        cross_location_drops=cross, price_coincidence_summary=coincidence, reconciliation_summary=recon,
        approved_locations=tuple(result.report.approved_locations), evidence_id=_evidence_id(result))


# ------------------------------------------------------------------ timeline view

#: Long-form plotting source: one point per canonical location and scheduled capture.
TIMELINE_SOURCE_COLUMNS: tuple[str, ...] = (
    "canonical_city", "canonical_location", "period", "eligible", "returned_product_count", "segment",
    "observed_drop", "drop_with_price_increase")


def assortment_timeline_source_frame(tables: AssortmentPresentationTables) -> pd.DataFrame:
    """The plotting source, reconciled to the validated timeline.

    Ineligible captures have no count (``NaN``, never zero); eligible empty
    captures are zero. ``segment`` numbers the continuous runs of each location:
    it changes at every break or ineligible capture, so lines never cross one.

    Raises:
        AssortmentReconciliationError: The source disagrees with the timeline.
    """
    if not isinstance(tables, AssortmentPresentationTables):
        raise TypeError("expected AssortmentPresentationTables")
    rows = _records(tables.assortment_timeline)
    out = []
    segment, last = 0, None
    for r in rows:
        key = (r["canonical_city"], r["canonical_location"])
        eligible = r["capture_state"] == CaptureState.ELIGIBLE.value
        if key != last or r["assessability_status"] != AssessabilityStatus.ASSESSED.value:
            segment += 1
        last = key
        assessed = r["assessability_status"] == AssessabilityStatus.ASSESSED.value
        drop = assessed and r["absolute_drop"] > 0
        out.append((key[0], key[1], r[_CAPTURE], eligible,
                    float(r["returned_product_count"]) if eligible else float("nan"),
                    segment if eligible else 0, bool(drop), bool(drop and r["price_increase_count"] > 0)))
    frame = pd.DataFrame(out, columns=list(TIMELINE_SOURCE_COLUMNS))
    review = tables.observed_drop_review
    eligible = frame["eligible"]
    if (len(frame) != len(tables.assortment_timeline) or int(frame["observed_drop"].sum()) != len(review)
            or int(frame["drop_with_price_increase"].sum()) != int((review["price_increase_count"] > 0).sum())
            or frame.loc[~eligible, "returned_product_count"].notna().any()
            or frame.loc[eligible, "returned_product_count"].isna().any()
            or int(frame.loc[eligible, "returned_product_count"].sum())
            != int(tables.assortment_timeline["returned_product_count"].sum())):
        raise AssortmentReconciliationError("rule reconciliation: the timeline source disagrees with the timeline")
    return frame


_SERIES, _DROP, _INK, _MUTED, _BAND = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#ecebe8"


def _render(source: pd.DataFrame, dpi: int, footer: str):  # type: ignore[no-untyped-def]
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    keys = list(dict.fromkeys(zip(source["canonical_city"], source["canonical_location"])))
    periods = sorted(set(source["period"]), key=parse_scheduled_period)
    x_of = {p: i for i, p in enumerate(periods)}
    top = max(1.0, float(np.nanmax(source["returned_product_count"].to_numpy(dtype=float)))
              if source["eligible"].any() else 1.0)
    fig = Figure(figsize=(11, 1.6 * len(keys) + 1.2), dpi=dpi)
    axes = fig.subplots(len(keys), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    for ax, key in zip(axes, keys):
        mine = source[(source["canonical_city"] == key[0]) & (source["canonical_location"] == key[1])]
        for _, cell in mine[~mine["eligible"]].iterrows():
            ax.axvspan(x_of[cell["period"]] - 0.5, x_of[cell["period"]] + 0.5, color=_BAND, lw=0, zorder=0)
        for _, run in mine[mine["eligible"]].groupby("segment", sort=True):
            ax.plot([x_of[p] for p in run["period"]], run["returned_product_count"], color=_SERIES, lw=2, zorder=2)
            if len(run) == 1:
                ax.plot([x_of[p] for p in run["period"]], run["returned_product_count"], "o", color=_SERIES, ms=4,
                        zorder=2)
        drops = mine[mine["observed_drop"]]
        ax.plot([x_of[p] for p in drops["period"]], drops["returned_product_count"], "v", color=_DROP, ms=8,
                zorder=3, linestyle="none")
        both = mine[mine["drop_with_price_increase"]]
        ax.plot([x_of[p] for p in both["period"]], both["returned_product_count"], "o", mfc="none", mec=_INK,
                ms=13, mew=1.5, zorder=4, linestyle="none")
        ax.set_ylim(0, top * 1.15)
        ax.set_title(f"{key[1]} ({key[0]})", loc="left", fontsize=9, color=_INK)
        ax.grid(axis="y", color=_BAND, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=_MUTED, labelsize=8)
    step = max(1, len(periods) // 12)
    axes[-1].set_xticks(range(0, len(periods), step))
    axes[-1].set_xticklabels([periods[i] for i in range(0, len(periods), step)], rotation=45, ha="right",
                             fontsize=7)
    axes[-1].set_xlabel("Scheduled capture period (UTC)", color=_MUTED, fontsize=8)
    fig.supylabel("Returned products (distinct)", color=_MUTED, fontsize=9)
    fig.legend(handles=[
        Line2D([], [], color=_SERIES, lw=2, label="Returned products (line breaks at every interval break)"),
        Line2D([], [], marker="v", color=_DROP, linestyle="none", ms=8, label="Observed drop (review candidate)"),
        Line2D([], [], marker="o", mfc="none", mec=_INK, linestyle="none", ms=10,
               label="Observed drop with a same-interval price increase"),
        Patch(color=_BAND, label="Governed exclusion or missing capture (no count)")],
        loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.text(0.01, 0.005, footer, fontsize=7, color=_MUTED)
    fig.subplots_adjust(top=1 - 0.9 / fig.get_figheight(), bottom=0.9 / fig.get_figheight() + 0.02, hspace=0.45)
    return fig


def _timeline_figure(tables: AssortmentPresentationTables, dpi: int):  # type: ignore[no-untyped-def]
    """The reconciled figure object; its footer is the shared policy wording of the same tables."""
    source = assortment_timeline_source_frame(tables)
    return _render(source, dpi, _policy_wording(tables).figure_footer)


def assortment_timeline_png(tables: AssortmentPresentationTables, *, dpi: int = 110) -> bytes:
    """The reconciled timeline figure as PNG bytes in memory (nothing is written; the figure is closed)."""
    fig = _timeline_figure(tables, dpi)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=dpi, metadata={"Software": None})
    finally:
        fig.clear()
    return buffer.getvalue()


# ------------------------------------------------------------------ policy wording (narrative and figure)


@dataclass(frozen=True)
class _PolicySummary:
    """Policy status and classification counts, computed only from validated, mutually consistent tables."""

    status: AnomalyPolicyStatus
    assessed_intervals: int
    classified_intervals: int        # assessed rows carrying a boolean classification
    unusual_intervals: int           # assessed rows classified ``unusual_drop = True``
    review_rows: int                 # observed-drop review candidates
    review_unusual: int              # review rows classified ``unusual_drop = True``


def _policy_summary(tables: AssortmentPresentationTables) -> _PolicySummary:
    """Summarize the engine's classifications (never re-applies a policy) and prove the tables agree.

    One policy status across the timeline, location summary and review; every
    review row equals the timeline's assessed drop row of the same key,
    including its ``unusual_drop`` value; classifications exist exactly under
    an approved policy (missing otherwise, never ``False``).

    Raises:
        AssortmentReconciliationError: The tables disagree.
    """
    R = AssortmentReconciliationError
    rows = _records(tables.assortment_timeline)
    statuses = ({r["anomaly_policy_status"] for r in rows}
                | {r["anomaly_policy_status"] for r in _records(tables.location_summary)}
                | {r["anomaly_policy_status"] for r in _records(tables.observed_drop_review)})
    if len(statuses) != 1:
        raise R("rule policy_status: the tables disagree on the anomaly-policy status")
    status = AnomalyPolicyStatus(next(iter(statuses)))
    assessed = [r for r in rows if r["assessability_status"] == AssessabilityStatus.ASSESSED.value]
    drops = {(r["canonical_city"], r["canonical_location"], r[_CAPTURE]): r["unusual_drop"]
             for r in assessed if r["absolute_drop"] > 0}
    review = {(r["canonical_city"], r["canonical_location"], r[_CAPTURE]): r["unusual_drop"]
              for r in _records(tables.observed_drop_review)}
    if review != drops:
        raise R("rule policy_status: review classifications differ from the timeline")
    classified = [r["unusual_drop"] for r in assessed if r["unusual_drop"] is not None]
    approved = status is AnomalyPolicyStatus.APPROVED
    if len(classified) != (len(assessed) if approved else 0):
        raise R("rule policy_status: classifications exist exactly under an approved policy")
    return _PolicySummary(status, len(assessed), len(classified), sum(bool(v) for v in classified), len(review),
                          sum(v is True or v is np.True_ for v in review.values()))


@dataclass(frozen=True)
class _PolicyWording:
    """The policy sentences shared by the narrative and the figure (one source of truth)."""

    summary: _PolicySummary
    drop_sentences: tuple[str, ...]
    limitation: str
    figure_footer: str


def _policy_wording(tables: AssortmentPresentationTables) -> _PolicyWording:
    """Wording derived only from the validated policy status and classification counts (no method or threshold)."""
    s = _policy_summary(tables)
    causal = "the data alone does not establish a cause"
    if s.status is AnomalyPolicyStatus.APPROVED:
        drops = (f"An approved unusual-drop policy, supplied explicitly to the engine, was applied: "
                 f"{s.unusual_intervals} of {s.assessed_intervals} assessed intervals were classified as unusual "
                 f"({s.review_unusual} of {s.review_rows} observed drops).",
                 "The classification reports the supplied policy's result only; its method and thresholds are not "
                 f"shown here, and {causal}.")
        limitation = ("Unusual-drop classification used the explicitly supplied approved policy; it is not the "
                      "repository default, and no monitoring rule beyond that classification is established.")
        footer = (f"Unusual-drop classification: approved policy applied; {s.unusual_intervals} of "
                  f"{s.assessed_intervals} assessed intervals classified as unusual. Coincidence is not causation.")
    else:
        state = ("is unavailable" if s.status is AnomalyPolicyStatus.UNAVAILABLE
                 else "is proposed but not approved")
        reason = ("no approved policy" if s.status is AnomalyPolicyStatus.UNAVAILABLE
                  else "policy proposed, not approved")
        if s.review_rows:
            drops = (f"The unusual-drop policy {state}, so no drop is classified; these are review candidates, "
                     f"not classified events, and {causal}.",)
        else:
            drops = (f"The unusual-drop policy {state}, so no drop is classified in any case.",)
        limitation = "Unusual-drop monitoring awaits an approved method, minimum history, grouping and thresholds."
        footer = f"Unusual-drop classification: unavailable ({reason}). Coincidence is not causation."
    return _PolicyWording(s, drops, limitation, footer)


# ------------------------------------------------------------------ narrative

#: The narrative sections of a completed presentation, in order.
NARRATIVE_SECTIONS: tuple[str, ...] = (
    "Scope and readiness", "Visible assortment over time", "Consecutive-capture stability",
    "Additions and removals", "Observed-drop review", "Price coincidence", "Limitations and next actions")


@dataclass(frozen=True)
class AssortmentNarrative:
    """A deterministic, bounded narrative built only from validated aggregate tables."""

    completed: bool
    sections: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def text(self) -> str:
        return "\n\n".join(f"{heading}\n" + "\n".join(f"- {s}" for s in sentences)
                           for heading, sentences in self.sections)


def _ratio_text(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def build_assortment_narrative(presentation: object) -> AssortmentNarrative:
    """The narrative of a presentation result or tables (blocked: categories only, no numbers or findings).

    Every number comes from the validated presentation tables. The wording is
    descriptive: observed drops are review candidates, coincidence is temporal
    association, and no cause or statistical unusualness is claimed.
    """
    if isinstance(presentation, AssortmentPresentationResult):
        if not presentation.completed:
            categories = ", ".join(b.value for b in presentation.report.blockers)
            return AssortmentNarrative(False, (("Status", (
                f"The visible-assortment presentation is blocked ({categories}).",
                "No tables, figure or findings are produced while it is blocked."),),))
        tables = presentation.tables
    else:
        tables = presentation
    if not isinstance(tables, AssortmentPresentationTables):
        raise TypeError("expected an AssortmentPresentationResult or AssortmentPresentationTables")
    wording = _policy_wording(tables)
    rows = _records(tables.assortment_timeline)
    summary = _records(tables.location_summary)
    review = _records(tables.observed_drop_review)
    cross = _records(tables.cross_location_drops)
    coincidence = _records(tables.price_coincidence_summary)
    periods = sorted({r[_CAPTURE] for r in rows}, key=parse_scheduled_period)
    hours = int((parse_scheduled_period(periods[-1]) - parse_scheduled_period(periods[0])) / dt.timedelta(hours=1)) + 1
    total = lambda column: sum(r[column] for r in summary)  # noqa: E731
    locations, scheduled, eligible = len(summary), len(rows), total("eligible_captures")
    per_location = max(r["scheduled_captures"] for r in summary)
    excluded, missing, breaks = total("excluded_captures"), total("missing_captures"), total("interval_breaks")
    assessed_rows = [r for r in rows if r["assessability_status"] == AssessabilityStatus.ASSESSED.value]
    assessed = len(assessed_rows)
    reasons = sorted({r["interval_break_reason"] for r in rows if r["interval_break_reason"] is not None})
    scope = (
        f"The visible-assortment analysis completed for {_plural(locations, 'approved canonical location')} and "
        f"{_plural(scheduled, 'scheduled location capture')} ({eligible} eligible, {excluded} governed "
        f"exclusions, {missing} missing).",
        f"All {len(tables.reconciliation_summary)} reconciliation checks passed before anything was presented.",
        f"Each location was scheduled for at most {per_location} hourly captures (a UTC span of {hours} hours "
        "across cities), a short observation window.",
        (f"Typed interval breaks: {_plural(breaks, 'adjacent capture pair')} ({', '.join(reasons)}); nothing is "
         "compared across a break." if breaks else "No interval break occurred between scheduled captures."))
    returned = [r["returned_product_count"] for r in rows if r["returned_product_count"] is not None]
    empty = total("empty_captures")
    changed_locations = sum(1 for r in summary if r["total_additions"] or r["total_removals"])
    over_time = (
        (f"Returned-product counts (distinct products per location capture) ranged from {min(returned)} to "
         f"{max(returned)} across {eligible} eligible captures." if returned
         else "No eligible capture was available to count."),
        (f"Eligible captures with no returned product: {empty}. They are observed empty sets, not evidence about "
         "supplier availability." if empty else "No eligible capture returned zero products."),
        f"{changed_locations} of {locations} locations had at least one addition or removal across valid intervals.")
    retention = [r["retention"] for r in assessed_rows if r["retention"] is not None]
    jaccard = [r["jaccard_similarity"] for r in assessed_rows if r["jaccard_similarity"] is not None]
    stability = (
        f"Retention (retained products divided by the previous capture's distinct products) is defined for "
        f"{len(retention)} of {assessed} assessed intervals; median {_ratio_text(_median(retention))}"
        + (f", range {min(retention):.3f} to {max(retention):.3f}." if retention else "."),
        f"Jaccard similarity (retained products divided by the union of both captures) is defined for "
        f"{len(jaccard)} of {assessed} assessed intervals; median {_ratio_text(_median(jaccard))}"
        + (f", range {min(jaccard):.3f} to {max(jaccard):.3f}." if jaccard else "."),
        f"Zero denominators: {assessed - len(retention)} for retention (empty previous capture) and "
        f"{assessed - len(jaccard)} for Jaccard (both captures empty); seed captures and captures after a break "
        "are in no denominator.")
    change_periods: dict[str, int] = {}
    for r in assessed_rows:
        if r["assortment_change"]:
            change_periods[r[_CAPTURE]] = change_periods.get(r[_CAPTURE], 0) + 1
    multi = sum(1 for n in change_periods.values() if n > 1)
    additions = (
        f"Across assessed intervals: {_plural(total('total_additions'), 'product addition')} in "
        f"{_plural(total('addition_intervals'), 'interval')} and {_plural(total('total_removals'), 'removal')} in "
        f"{_plural(total('removal_intervals'), 'interval')}; product identities are not shown.",
        (f"Assortment changes occurred in {_plural(len(change_periods), 'scheduled period')}: "
         f"{len(change_periods) - multi} at a single location and {multi} at several locations in the same "
         "period, within this extract only." if change_periods else "No assessed interval had an assortment change."),
        "Simultaneous observations are not evidence of a common cause.")
    isolated = sum(1 for c in cross if c["simultaneity"] == Simultaneity.ISOLATED.value)
    patterns = {p.value: sum(1 for r in review if r["drop_pattern"] == p.value) for p in DropPattern}
    if review:
        top = review[0]
        drops = (
            f"Observed drops (absolute drop above zero): {_plural(len(review), 'assessed interval')}, in "
            f"{_plural(len(cross), 'scheduled period')}: {isolated} isolated to one location and "
            f"{len(cross) - isolated} simultaneous across locations, in this extract only.",
            f"The largest observed drop within this extract was {top['absolute_drop']} products (drop rate "
            f"{top['drop_rate']:.3f}, retention {top['retention']:.3f}).",
            f"Drop patterns: net contraction {patterns['net_contraction']}, complete turnover "
            f"{patterns['complete_turnover']}, empty current capture {patterns['empty_current_capture']}; "
            f"intervals that also had additions: {sum(1 for r in review if r['addition_count'] > 0)}.",
            *wording.drop_sentences)
    else:
        drops = ("No assessed interval shows an observed drop.", *wording.drop_sentences)
    price = (
        f"Of {assessed} assessed intervals, {sum(c['assortment_change_intervals'] for c in coincidence)} "
        f"had an assortment change and {sum(c['price_change_intervals'] for c in coincidence)} had a price change "
        f"({sum(c['price_increase_intervals'] for c in coincidence)} with increases, "
        f"{sum(c['price_decrease_intervals'] for c in coincidence)} with decreases).",
        f"Intervals with both in the same location interval: {sum(c['coincident_intervals'] for c in coincidence)}; "
        f"observed drops with a same-interval price increase: "
        f"{sum(c['drop_with_price_increase_intervals'] for c in coincidence)}, with a price decrease: "
        f"{sum(c['drop_with_price_decrease_intervals'] for c in coincidence)}; falling assortment with a price "
        f"increase: {sum(c['falling_with_increase_intervals'] for c in coincidence)}.",
        "Coincidence is temporal association within the same location interval, not causation.")
    limits = (
        f"At most {per_location} hourly captures per location are insufficient for seasonal or long-term "
        "baselines, and hourly "
        "captures are repeated measurements.",
        "Visible assortment is what the collection returned, not confirmation of what the supplier offered.",
        "Operational and source corroboration would be required before any causal interpretation.",
        wording.limitation,
        "Timeline persistence is not approved, so these results stay in memory and nothing is written.")
    return AssortmentNarrative(True, tuple(zip(NARRATIVE_SECTIONS, (scope, over_time, stability, additions, drops,
                                                                      price, limits))))


# ------------------------------------------------------------------ results and entry points


def assortment_persistence_approved() -> bool:
    """Whether timeline persistence is approved (only a derived, approved status counts; today it is proposed)."""
    return DEFAULT_ASSORTMENT_DEFINITION.statuses.get("timeline_persistence") is DefinitionStatus.DERIVED


@dataclass(frozen=True)
class AssortmentPresentationReport:
    """Print-safe presentation status: blockers, table row counts and reconciliation."""

    status: AssortmentPresentationStatus
    blockers: tuple[AssortmentPresentationBlocker, ...] = ()
    upstream_blockers: tuple[str, ...] = ()
    table_rows: tuple[tuple[str, int], ...] = ()
    reconciliation_checks: int = 0
    reconciled: bool = False
    anomaly_policy_status: AnomalyPolicyStatus = AnomalyPolicyStatus.UNAVAILABLE
    persistence_approved: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, AssortmentPresentationStatus):
            raise AssortmentReconciliationError("status must be an AssortmentPresentationStatus")
        if self.status is AssortmentPresentationStatus.BLOCKED:
            if not self.blockers or self.table_rows or self.reconciled or self.reconciliation_checks:
                raise AssortmentReconciliationError("a blocked presentation has blockers and no outputs")
            if not all(isinstance(b, AssortmentPresentationBlocker) for b in self.blockers):
                raise AssortmentReconciliationError("blockers must be AssortmentPresentationBlocker values")
        elif self.blockers or self.upstream_blockers or not self.reconciled:
            raise AssortmentReconciliationError("a completed presentation is reconciled without blockers")

    @property
    def completed(self) -> bool:
        return self.status is AssortmentPresentationStatus.COMPLETED


@dataclass(frozen=True)
class AssortmentPresentationResult:
    """The presentation report plus the in-memory assortment result and sanitized tables (none in ``repr``)."""

    report: AssortmentPresentationReport
    assortment: VisibleAssortmentResult | None = field(default=None, repr=False, compare=False)
    tables: AssortmentPresentationTables | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.report, AssortmentPresentationReport):
            raise TypeError("report must be an AssortmentPresentationReport")
        if self.report.completed != isinstance(self.tables, AssortmentPresentationTables):
            raise AssortmentReconciliationError("only a completed presentation holds tables")
        if not self.report.completed and self.assortment is not None:
            raise AssortmentReconciliationError("a blocked presentation holds no assortment result")

    @property
    def completed(self) -> bool:
        return self.report.completed


def _blocked(blockers: Sequence[AssortmentPresentationBlocker], upstream: Sequence[str] = ()
             ) -> AssortmentPresentationResult:
    return AssortmentPresentationResult(AssortmentPresentationReport(
        status=AssortmentPresentationStatus.BLOCKED, blockers=tuple(dict.fromkeys(blockers)),
        upstream_blockers=tuple(dict.fromkeys(upstream))))


def _refuse_output(output_dir: object) -> AssortmentPresentationResult | None:
    """Fail closed for any requested output while persistence is not approved (nothing is touched)."""
    if output_dir is None:
        return None
    if not isinstance(output_dir, (str, os.PathLike)):
        raise TypeError("output_dir must be a path or None")
    if not assortment_persistence_approved():
        return _blocked([AssortmentPresentationBlocker.PERSISTENCE_NOT_APPROVED])
    raise NotImplementedError("no approved persistence format exists")                 # pragma: no cover


def assortment_presentation_from_pipeline(run: object, *, output_dir: str | os.PathLike | None = None
                                          ) -> AssortmentPresentationResult:
    """Price changes, visible assortment and presentation from one pipeline result (the same evidence throughout).

    ``output_dir`` is refused (``persistence_not_approved``) because timeline
    persistence is not approved; nothing is written either way.
    """
    from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult
    from ql2_sixt_canada_analysis.pricing_population import frame_binding
    from ql2_sixt_canada_analysis.visible_assortment import visible_assortment_from_pipeline

    refused = _refuse_output(output_dir)
    if refused is not None:
        return refused
    if not isinstance(run, PricingPipelineResult):
        raise TypeError("run must be a PricingPipelineResult")
    B = AssortmentPresentationBlocker
    assortment = visible_assortment_from_pipeline(run)
    if not assortment.completed:
        return _blocked([B.ASSORTMENT_NOT_COMPLETED], [b.value for b in assortment.report.blockers]
                        + list(assortment.report.upstream_blockers))
    if assortment.binding != frame_binding(run.jobs, run.cars) or assortment.location_authority is not \
            run.location_authority:
        return _blocked([B.EVIDENCE_MISMATCH])
    try:
        tables = build_assortment_presentation_tables(assortment)
    except AssortmentReconciliationError:
        return _blocked([B.RECONCILIATION_FAILED])
    report = AssortmentPresentationReport(
        status=AssortmentPresentationStatus.COMPLETED,
        table_rows=tuple((name, len(frame)) for name, frame in tables.items()),
        reconciliation_checks=len(tables.reconciliation_summary), reconciled=True,
        anomaly_policy_status=assortment.report.anomaly_policy_status,
        persistence_approved=assortment_persistence_approved())
    return AssortmentPresentationResult(report=report, assortment=assortment, tables=tables)


def run_assortment_presentation(raw_dir: str | os.PathLike | None = None, *,
                                output_dir: str | os.PathLike | None = None) -> AssortmentPresentationResult:
    """Run ``run_pricing_pipeline`` exactly once, then price changes, assortment and presentation on that result.

    Nothing is written. An ``output_dir`` is refused before the pipeline runs.
    """
    from ql2_sixt_canada_analysis import pricing_pipeline

    refused = _refuse_output(output_dir)
    if refused is not None:
        return refused
    return assortment_presentation_from_pipeline(pricing_pipeline.run_pricing_pipeline(raw_dir))
