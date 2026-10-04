"""Vehicle-attribute stability: do structural attributes stay put per vehicle?

Driven by :data:`~ql2_sixt_canada_analysis.schemas.VEHICLE_ATTRIBUTE_STABILITY`.

Counting policy
---------------
* **Entities** are distinct ``(*context, *entity key)`` tuples (pandas
  grouping on the columns themselves - no string keys). Tuples with a missing
  component are *incomplete-identity* entities; they are counted, never
  merged into complete ones, and never assessed for stability.
* **Observation time** is the reconciled instant of the configured temporal
  field (``parse_temporal_field``). A complete entity with any row lacking a
  valid instant is *temporally unassessable*: set-based value conflicts are
  still detected, but no ordered transition or history claim is made.
* **History** is the number of distinct valid instants. Entities below
  ``minimum_observations`` have *insufficient history*: a single capture
  never proves stability. ``complete = unassessable + sufficient +
  insufficient``.
* **Value conflict**: more than one distinct non-missing value under the
  attribute's comparison policy (exact, type-aware). A change that later
  reverts is still a conflict. **Same-capture conflict**: distinct values
  (or mixed presence under a presence policy) at one instant.
* **Presence violation** follows each attribute's
  :class:`~ql2_sixt_canada_analysis.schemas.MissingValueStabilityPolicy`;
  missingness is reported separately from value conflicts and missing values
  are never replaced by sentinels.
* Per attribute, every complete entity falls in exactly one category, in
  precedence order: value conflict, temporally unassessable, insufficient
  history, always missing, intermittently missing, stable.
* **Empty / no history:** no observed conflict cannot establish stability;
  the status is ``UNASSESSABLE`` (strict validation fails) unless a violation
  is proven.

Source frames are never sorted, mutated or written; the report holds
contract field names, enums and integer counts only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.comparison import canonical_location_keys
from ql2_sixt_canada_analysis.identifiers import is_identifier_dtype
from ql2_sixt_canada_analysis.quality import completely_blank_row_mask
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    VEHICLE_ATTRIBUTE_STABILITY,
    AttributeComparisonPolicy,
    MissingValueStabilityPolicy,
    VehicleAttributeDefinition,
    VehicleStabilityConfigurationError,
    VehicleStabilityDefinition,
)
from ql2_sixt_canada_analysis.temporal import parse_temporal_field

__all__ = [
    "VehicleAttributeStabilityError",
    "VehicleAttributeStabilityReport",
    "VehicleStabilityPreconditionError",
    "VehicleStabilityReport",
    "VehicleStabilityStatus",
    "assess_vehicle_attribute_stability",
    "validate_vehicle_attribute_stability",
]


class VehicleStabilityStatus(StrEnum):
    PASSED = "passed"
    VIOLATIONS = "violations"
    UNASSESSABLE = "unassessable"


@dataclass(frozen=True, slots=True)
class VehicleAttributeStabilityReport:
    """Aggregate result for one stable attribute (contract name and counts only).

    Exclusive categories (sum to ``entities_assessed``): ``value_conflict``,
    ``temporally_unassessable``, ``insufficient_history``, ``always_missing``,
    ``intermittently_missing``, ``stable``. The ``entities_with_*`` and
    transition counts are overlapping measures.
    """

    column: str
    missing_policy: MissingValueStabilityPolicy
    comparison: AttributeComparisonPolicy
    entities_assessed: int
    value_conflict: int
    temporally_unassessable: int
    insufficient_history: int
    always_missing: int
    intermittently_missing: int
    stable: int
    entities_with_missing: int
    entities_with_presence_violation: int
    present_to_missing: int
    missing_to_present: int
    same_capture_conflicts: int

    @property
    def passes(self) -> bool:
        return self.value_conflict == 0 and self.entities_with_presence_violation == 0

    @property
    def categories_reconcile(self) -> bool:
        return self.entities_assessed == (self.value_conflict + self.temporally_unassessable
                                          + self.insufficient_history + self.always_missing
                                          + self.intermittently_missing + self.stable)


@dataclass(frozen=True, slots=True)
class VehicleStabilityReport:
    """Aggregate vehicle-stability report (counts, enums, contract names only)."""

    status: VehicleStabilityStatus
    observations_assessed: int
    distinct_entities: int
    complete_identity_entities: int
    incomplete_identity_entities: int
    incomplete_identity_observations: int
    temporally_unassessable_entities: int
    sufficient_history_entities: int
    insufficient_history_entities: int
    fully_stable_entities: int
    value_unstable_only_entities: int
    presence_unstable_only_entities: int
    value_and_presence_unstable_entities: int
    entities_with_value_conflicts: int
    entities_with_presence_instability: int
    same_capture_conflict_entities: int
    attributes: tuple[VehicleAttributeStabilityReport, ...]

    @property
    def required_attributes_pass(self) -> bool:
        return all(a.passes for a in self.attributes if a.missing_policy is MissingValueStabilityPolicy.REQUIRED)

    @property
    def all_attributes_pass(self) -> bool:
        return all(a.passes for a in self.attributes)

    @property
    def is_valid(self) -> bool:
        return self.status is VehicleStabilityStatus.PASSED

    @property
    def violations(self) -> tuple[str, ...]:
        """Violation categories, in a fixed order (no values)."""
        found = []
        if self.incomplete_identity_entities:
            found.append("incomplete_identity")
        if self.temporally_unassessable_entities:
            found.append("temporally_unassessable")
        if self.entities_with_value_conflicts:
            found.append("value_conflict")
        if self.same_capture_conflict_entities:
            found.append("same_capture_conflict")
        if self.entities_with_presence_instability:
            found.append("presence_instability")
        if not found and self.status is VehicleStabilityStatus.UNASSESSABLE:
            found.append("insufficient_history")
        return tuple(found)

    @property
    def invariants_hold(self) -> bool:
        exclusive = (self.fully_stable_entities + self.value_unstable_only_entities
                     + self.presence_unstable_only_entities + self.value_and_presence_unstable_entities)
        return (self.distinct_entities == self.complete_identity_entities + self.incomplete_identity_entities
                and self.complete_identity_entities == (self.temporally_unassessable_entities
                                                        + self.sufficient_history_entities
                                                        + self.insufficient_history_entities)
                and exclusive == self.sufficient_history_entities
                and self.same_capture_conflict_entities <= self.complete_identity_entities
                and all(a.categories_reconcile and a.entities_assessed == self.complete_identity_entities
                        for a in self.attributes))


class VehicleStabilityPreconditionError(Exception):
    """A structural precondition for the assessment failed (category on ``reason``)."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Vehicle-stability precondition failed: {reason}.")
        self.reason = reason


class VehicleAttributeStabilityError(Exception):
    """Strict validation failed; ``violations`` lists categories, ``report`` the aggregates."""

    def __init__(self, report: VehicleStabilityReport) -> None:
        super().__init__("Vehicle-attribute stability contract failed: " + ", ".join(report.violations) + ".")
        self.report = report
        self.violations = report.violations


# ------------------------------------------------------------------- public API


def assess_vehicle_attribute_stability(
    cars: pd.DataFrame,
    definition: VehicleStabilityDefinition = VEHICLE_ATTRIBUTE_STABILITY,
) -> VehicleStabilityReport:
    """Assess stability on the cleaned, identifier-typed detail frame.

    Raises only for configuration or structural problems
    (:class:`VehicleStabilityConfigurationError`,
    :class:`VehicleStabilityPreconditionError`, ``TypeError``); unstable
    source values are reported, never raised or repaired.
    """
    if not isinstance(definition, VehicleStabilityDefinition):
        raise TypeError("definition must be a VehicleStabilityDefinition")
    if not isinstance(cars, pd.DataFrame):
        raise TypeError("cars must be a pandas DataFrame")
    time_field = definition.temporal.field(definition.observation_time_field)
    needed = (*definition.group_columns, *definition.attribute_columns, time_field.column)
    absent = tuple(c for c in needed if c not in cars.columns)
    if absent:
        raise VehicleStabilityConfigurationError(f"The frame lacks {len(absent)} configured column(s).", absent)
    identifiers = DATASET_DEFINITIONS[definition.dataset].identifier_columns
    if not all(is_identifier_dtype(cars[c].dtype) for c in identifiers if c in cars.columns):
        raise VehicleStabilityPreconditionError("identifier_dtype")
    if bool(completely_blank_row_mask(cars).any()):
        raise VehicleStabilityPreconditionError("blank_rows_present")

    # Temporary working frame on a fresh RangeIndex; the source is never touched.
    group = cars.loc[:, list(definition.group_columns)].reset_index(drop=True)
    if definition.canonical_location_grouping:
        cov = definition.location_coverage
        canonical = canonical_location_keys(cars, cov).reset_index(drop=True)   # tuples, source untouched
        for i, column in enumerate(cov.location_columns):
            group[column] = canonical.map(lambda k, i=i: k[i]).astype(object)
    complete = group.notna().all(axis=1).to_numpy()
    entity = group.groupby(list(group.columns), dropna=False, sort=False).ngroup().to_numpy()
    instants = parse_temporal_field(cars[time_field.column], time_field,
                                    definition.temporal.canonical_timezone).instants.reset_index(drop=True)
    valid_time = instants.notna().to_numpy()

    n_entities = int(np.unique(entity).size)
    incomplete_entities = int(np.unique(entity[~complete]).size)
    work = pd.DataFrame({"e": entity[complete], "t": instants[complete].to_numpy(),
                         "ok": valid_time[complete]})
    by_e = work.groupby("e", sort=True)
    unassessable = ~by_e["ok"].all().astype(bool)
    history = by_e["t"].nunique()          # distinct valid instants
    sufficient = ~unassessable & (history >= definition.minimum_observations)
    insufficient = ~unassessable & ~sufficient
    entities = unassessable.index

    attribute_reports, conflict_any, presence_any, same_any = [], _false(entities), _false(entities), _false(entities)
    for attribute in definition.attributes:
        codes = _value_codes(cars[attribute.column].reset_index(drop=True), attribute)[complete]
        rep, conflict, presence, same = _assess_attribute(attribute, work, codes, unassessable, insufficient)
        attribute_reports.append(rep)
        conflict_any |= conflict
        presence_any |= presence
        same_any |= same

    stable = sufficient & ~conflict_any & ~presence_any
    report = dict(
        observations_assessed=len(cars), distinct_entities=n_entities,
        complete_identity_entities=int(len(entities)), incomplete_identity_entities=incomplete_entities,
        incomplete_identity_observations=int((~complete).sum()),
        temporally_unassessable_entities=int(unassessable.sum()),
        sufficient_history_entities=int(sufficient.sum()), insufficient_history_entities=int(insufficient.sum()),
        fully_stable_entities=int(stable.sum()),
        value_unstable_only_entities=int((sufficient & conflict_any & ~presence_any).sum()),
        presence_unstable_only_entities=int((sufficient & ~conflict_any & presence_any).sum()),
        value_and_presence_unstable_entities=int((sufficient & conflict_any & presence_any).sum()),
        entities_with_value_conflicts=int(conflict_any.sum()),
        entities_with_presence_instability=int(presence_any.sum()),
        same_capture_conflict_entities=int(same_any.sum()) if definition.same_capture_conflicts_reported else 0,
        attributes=tuple(attribute_reports),
    )
    violated = (incomplete_entities or report["temporally_unassessable_entities"]
                or report["entities_with_value_conflicts"] or report["entities_with_presence_instability"])
    if violated:
        status = VehicleStabilityStatus.VIOLATIONS
    elif report["sufficient_history_entities"] == 0:
        status = VehicleStabilityStatus.UNASSESSABLE
    else:
        status = VehicleStabilityStatus.PASSED
    return VehicleStabilityReport(status=status, **report)


def validate_vehicle_attribute_stability(
    cars: pd.DataFrame,
    definition: VehicleStabilityDefinition = VEHICLE_ATTRIBUTE_STABILITY,
) -> VehicleStabilityReport:
    """Return the report if the contract passes; else raise :class:`VehicleAttributeStabilityError`."""
    report = assess_vehicle_attribute_stability(cars, definition)
    if not report.is_valid:
        raise VehicleAttributeStabilityError(report)
    return report


# ---------------------------------------------------------------------- helpers


def _false(index: pd.Index) -> pd.Series:
    return pd.Series(False, index=index)


def _value_codes(values: pd.Series, attribute: VehicleAttributeDefinition) -> np.ndarray:
    """Integer code per row for exact (type-aware) comparison; -1 = missing."""
    missing = values.isna().to_numpy()
    if attribute.comparison is AttributeComparisonPolicy.AUTHORITATIVE_MAPPING:
        mapping = attribute.mapping
        values = values.astype(object).map(lambda v: mapping.get(v, v) if not pd.isna(v) else v)
    if values.dtype == object or attribute.comparison is AttributeComparisonPolicy.AUTHORITATIVE_MAPPING:
        # (type, value) keys so that 0, 0.0 and False stay distinct values.
        values = pd.Series([None if m else (type(v).__name__, v) for v, m in zip(values, missing)], dtype=object)
    codes, _ = pd.factorize(values, use_na_sentinel=True)
    codes = codes.copy()
    codes[missing] = -1
    return codes


def _assess_attribute(attribute: VehicleAttributeDefinition, work: pd.DataFrame, codes: np.ndarray,
                      unassessable: pd.Series, insufficient: pd.Series):  # type: ignore[no-untyped-def]
    frame = work.assign(c=codes, m=codes < 0)
    by_e = frame.groupby("e", sort=True)
    distinct = frame.loc[~frame["m"]].groupby("e")["c"].nunique().reindex(unassessable.index, fill_value=0)
    has_missing = by_e["m"].any().astype(bool)
    all_missing = by_e["m"].all().astype(bool)
    intermittent = has_missing & ~all_missing
    conflict = distinct > 1

    policy = attribute.missing_policy
    if policy is MissingValueStabilityPolicy.REQUIRED:
        presence = has_missing
    elif policy is MissingValueStabilityPolicy.PRESENCE_STABLE:
        presence = intermittent
    else:
        presence = _false(unassessable.index)

    # Per-capture view on rows with a valid instant (temporary, sorted copy).
    timed = frame.loc[frame["ok"]]
    per_capture = timed.groupby(["e", "t"], sort=True).agg(
        n=("c", lambda s: s[s >= 0].nunique()), anym=("m", "any"), allm=("m", "all")
    ).astype({"n": "int64", "anym": bool, "allm": bool})
    mixed = per_capture["anym"] & ~per_capture["allm"]
    same_value = per_capture["n"] > 1
    same_presence = mixed if policy is not MissingValueStabilityPolicy.MISSING_IGNORED else mixed & False
    same = (same_value | same_presence).groupby(level="e").any().reindex(unassessable.index, fill_value=False)

    # Ordered transitions between single-state captures of assessable entities.
    states = per_capture.loc[~mixed, ["allm"]].reset_index()
    states = states.loc[~states["e"].map(unassessable).astype(bool).to_numpy()]
    previous = states.groupby("e")["allm"].shift()
    p2m = states.loc[(previous == False) & states["allm"], "e"].nunique()   # noqa: E712
    m2p = states.loc[(previous == True) & ~states["allm"], "e"].nunique()   # noqa: E712

    remaining = ~conflict
    c_unassessable = remaining & unassessable
    remaining &= ~unassessable
    c_insufficient = remaining & insufficient
    remaining &= ~insufficient
    c_always = remaining & all_missing
    c_intermittent = remaining & intermittent
    c_stable = remaining & ~has_missing
    report = VehicleAttributeStabilityReport(
        column=attribute.column, missing_policy=policy, comparison=attribute.comparison,
        entities_assessed=int(len(unassessable)), value_conflict=int(conflict.sum()),
        temporally_unassessable=int(c_unassessable.sum()), insufficient_history=int(c_insufficient.sum()),
        always_missing=int(c_always.sum()), intermittently_missing=int(c_intermittent.sum()),
        stable=int(c_stable.sum()), entities_with_missing=int(has_missing.sum()),
        entities_with_presence_violation=int(presence.sum()), present_to_missing=int(p2m),
        missing_to_present=int(m2p), same_capture_conflicts=int(same.sum()),
    )
    return report, conflict, presence, same
