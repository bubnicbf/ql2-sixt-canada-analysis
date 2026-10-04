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
* **Full-population status** (:class:`VehicleStabilityStatus`):
  ``PASSED`` only when *every* in-scope entity has a complete identity,
  valid observation times and sufficient history, and no entity violates
  the contract. Proven violations give ``VIOLATIONS`` (precedence: they are
  facts, not missing evidence); otherwise some sufficient and some
  insufficient entities give ``PARTIALLY_ASSESSABLE`` and no sufficient
  entity gives ``UNASSESSABLE``. An empty violations list never implies
  stability: ``blocking_reasons`` adds ``insufficient_history`` whenever any
  entity lacks history (also alongside violations) and ``empty_population``
  for empty input. No partial-coverage tolerance exists.
* **Empty input:** no entity, nothing assessed - ``UNASSESSABLE``.
* :func:`classify_vehicle_entities` returns, in memory only, each entity's
  key and category (identifiers are confidential: never print or persist).

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
    "VehicleEntityHistory",
    "assess_vehicle_attribute_stability",
    "classify_vehicle_entities",
    "validate_vehicle_attribute_stability",
]


class VehicleStabilityStatus(StrEnum):
    """Status of the full in-scope entity population (see module docstring).

    Precedence: ``VIOLATIONS`` > ``PARTIALLY_ASSESSABLE`` > ``UNASSESSABLE`` >
    ``PASSED``. Only ``PASSED`` is valid.
    """

    PASSED = "passed"
    VIOLATIONS = "violations"
    PARTIALLY_ASSESSABLE = "partially_assessable"
    UNASSESSABLE = "unassessable"


class VehicleEntityHistory(StrEnum):
    """Exactly one assessment category per in-scope entity."""

    INCOMPLETE_IDENTITY = "incomplete_identity"
    TEMPORALLY_UNASSESSABLE = "temporally_unassessable"
    INSUFFICIENT_HISTORY = "insufficient_history"
    SUFFICIENT_HISTORY = "sufficient_history"


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

    def __post_init__(self) -> None:
        # Programmer invariants: PASSED means the full population was assessed and stable.
        if self.status is VehicleStabilityStatus.PASSED:
            assert self.sufficient_history_entities == self.distinct_entities > 0
            assert self.insufficient_history_entities == 0 and not self.violations

    @property
    def is_valid(self) -> bool:
        """True only for ``PASSED``: every in-scope entity assessed and stable."""
        return self.status is VehicleStabilityStatus.PASSED

    @property
    def fully_assessed(self) -> bool:
        """Every in-scope entity had a complete identity, valid times and sufficient history."""
        return self.distinct_entities > 0 and self.sufficient_history_entities == self.distinct_entities

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        """Every reason the population is not ``PASSED``: proven violations plus missing evidence."""
        found = list(self.violations)
        if self.insufficient_history_entities:
            found.append("insufficient_history")
        if self.distinct_entities == 0:
            found.append("empty_population")
        return tuple(found)

    @property
    def violations(self) -> tuple[str, ...]:
        """Proven violation categories, in a fixed order (no values).

        Empty does **not** mean stable: see :attr:`blocking_reasons`.
        """
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
    """Strict validation failed.

    ``blocking_reasons`` lists every category (violations and missing
    evidence), ``violations`` the proven violations only, ``report`` the
    aggregates. No values appear in the message.
    """

    def __init__(self, report: VehicleStabilityReport) -> None:
        super().__init__(f"Vehicle-attribute stability contract failed ({report.status.value}): "
                         + ", ".join(report.blocking_reasons) + ".")
        self.report = report
        self.violations = report.violations
        self.blocking_reasons = report.blocking_reasons


# ------------------------------------------------------------------- public API


def assess_vehicle_attribute_stability(
    cars: pd.DataFrame,
    definition: VehicleStabilityDefinition = VEHICLE_ATTRIBUTE_STABILITY,
) -> VehicleStabilityReport:
    """Assess stability on the cleaned, identifier-typed detail frame.

    Every in-scope entity is assessed; the status describes the full
    population. Raises only for configuration or structural problems
    (:class:`VehicleStabilityConfigurationError`,
    :class:`VehicleStabilityPreconditionError`, ``TypeError``); unstable
    source values are reported, never raised or repaired.
    """
    return _run(cars, definition).report


def classify_vehicle_entities(
    cars: pd.DataFrame,
    definition: VehicleStabilityDefinition = VEHICLE_ATTRIBUTE_STABILITY,
) -> pd.DataFrame:
    """In-memory per-entity classification (a new frame on every call).

    Columns: the entity's scope and key columns (canonical scope if the
    contract enables it), ``history`` (:class:`VehicleEntityHistory` value),
    ``value_conflict``, ``presence_violation``, ``same_capture_conflict``,
    ``stable`` (sufficient history and no violation) and
    ``unstable_attributes`` (contract attribute names, contract order). Rows
    are sorted by key. Holds confidential identifiers: never print, log or
    persist it.
    """
    return _run(cars, definition).entities()


@dataclass(frozen=True, slots=True, eq=False)
class _Run:
    report: VehicleStabilityReport
    keys: pd.DataFrame                   # one row per entity code (first occurrence), index = code
    complete_codes: pd.Index
    unassessable: pd.Series
    sufficient: pd.Series
    conflict: dict
    presence: dict
    same: pd.Series

    def entities(self) -> pd.DataFrame:
        frame = self.keys.copy()
        idx = frame.index
        history = pd.Series(VehicleEntityHistory.INCOMPLETE_IDENTITY.value, index=idx, dtype=object)
        complete = idx.isin(self.complete_codes)
        c = self.complete_codes
        history.loc[c] = np.where(self.unassessable.reindex(c), VehicleEntityHistory.TEMPORALLY_UNASSESSABLE.value,
                                  np.where(self.sufficient.reindex(c), VehicleEntityHistory.SUFFICIENT_HISTORY.value,
                                           VehicleEntityHistory.INSUFFICIENT_HISTORY.value))
        def flag(series: pd.Series) -> np.ndarray:
            return series.reindex(idx, fill_value=False).astype(bool).to_numpy() & complete
        columns = list(self.conflict)
        value = np.zeros(len(idx), dtype=bool)
        presence = np.zeros(len(idx), dtype=bool)
        unstable = [[] for _ in range(len(idx))]
        for column in columns:
            vc, pv = flag(self.conflict[column]), flag(self.presence[column])
            value |= vc
            presence |= pv
            for i in np.flatnonzero(vc | pv):
                unstable[i].append(column)
        frame["history"] = history.to_numpy()
        frame["value_conflict"] = value
        frame["presence_violation"] = presence
        frame["same_capture_conflict"] = flag(self.same)
        frame["stable"] = (frame["history"] == VehicleEntityHistory.SUFFICIENT_HISTORY.value).to_numpy() \
            & ~value & ~presence
        frame["unstable_attributes"] = [tuple(u) for u in unstable]
        key_columns = list(self.keys.columns)
        frame = frame.sort_values(key_columns, kind="mergesort", na_position="last") if key_columns else frame
        return frame.reset_index(drop=True)


def _run(cars: pd.DataFrame, definition: VehicleStabilityDefinition) -> _Run:
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
    conflicts, presences = {}, {}
    for attribute in definition.attributes:
        codes = _value_codes(cars[attribute.column].reset_index(drop=True), attribute)[complete]
        rep, conflict, presence, same = _assess_attribute(attribute, work, codes, unassessable, insufficient)
        attribute_reports.append(rep)
        conflicts[attribute.column], presences[attribute.column] = conflict, presence
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
    # Full-population status: absence of violations is not proof while any entity lacks history.
    if violated:
        status = VehicleStabilityStatus.VIOLATIONS
    elif report["sufficient_history_entities"] == 0:
        status = VehicleStabilityStatus.UNASSESSABLE
    elif report["insufficient_history_entities"]:
        status = VehicleStabilityStatus.PARTIALLY_ASSESSABLE
    else:
        status = VehicleStabilityStatus.PASSED
    first = pd.Series(np.arange(len(entity))).groupby(entity).min()
    keys = group.iloc[first.to_numpy()].astype(object).set_axis(first.index)
    keys = keys.where(keys.notna(), None)
    return _Run(report=VehicleStabilityReport(status=status, **report), keys=keys,
                complete_codes=entities, unassessable=unassessable, sufficient=sufficient,
                conflict=conflicts, presence=presences,
                same=same_any if definition.same_capture_conflicts_reported else _false(entities))


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
