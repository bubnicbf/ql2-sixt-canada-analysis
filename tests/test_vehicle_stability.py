"""Tests for vehicle-attribute stability.

All values are fabricated (``SYNTH-VEHICLE-001``, ``SYNTH-CLASS-A``,
``SYNTH-LOCATION-001``, small capacities, ``2025-01-15 05:00:00 MST``).
Column roles come from ``VEHICLE_ATTRIBUTE_STABILITY``; no field list is
duplicated here.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import contract_columns

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.quality import remove_completely_blank_rows
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    EXPECTED_LOCATION_COVERAGE,
    JOB_DETAIL_RELATIONSHIP,
    LOCATION_STREAM_COMPARISON,
    TEMPORAL_RECONCILIATION,
    VEHICLE_ATTRIBUTE_STABILITY,
    AttributeComparisonPolicy,
    DatasetKey,
    MissingValueStabilityPolicy as MP,
    TemporalKind,
    VehicleAttributeDefinition,
    VehicleStabilityConfigurationError,
    VehicleStabilityDefinition,
)
from ql2_sixt_canada_analysis.stability import (
    VehicleAttributeStabilityError,
    VehicleAttributeStabilityReport,
    VehicleStabilityPreconditionError,
    VehicleStabilityReport,
    VehicleEntityHistory as H,
    VehicleStabilityStatus as St,
    assess_vehicle_attribute_stability as assess,
    classify_vehicle_entities as classify,
    validate_vehicle_attribute_stability as validate,
)

CARS = DatasetKey.CARS
V = VEHICLE_ATTRIBUTE_STABILITY
KEY, = V.entity_key_columns
CTX, = V.context_columns
TIME = V.temporal.field(V.observation_time_field).column
CAT, TRANS, SEATS, BAGS = V.attribute_columns     # roles by policy checked in test_project_policies
PRICES = V.price_columns
JOB = JOB_DETAIL_RELATIONSHIP.detail_key_columns[0]
SEARCH = tuple(c for c in V.volatile_columns
               if c not in (*PRICES, TIME, *DATASET_DEFINITIONS[CARS].identifier_columns))
V1, V2 = "SYNTH-VEHICLE-001", "SYNTH-VEHICLE-002"
L1, L2 = "SYNTH-LOCATION-001", "SYNTH-LOCATION-002"
T = [f"2025-01-15 {h:02d}:00:00 MST" for h in range(5, 12)]


def obs(vehicle=V1, t=T[0], loc=L1, **values) -> dict:
    row = {KEY: vehicle, CTX: loc, TIME: t, CAT: "SYNTH-CLASS-A", TRANS: "SYNTH-AUTOMATIC", SEATS: 5, BAGS: 2,
           JOB: "SYNTH-JOB-001"}
    row.update(values)
    return row


def frame(rows: list[dict]) -> pd.DataFrame:
    records = [{c: r.get(c, f"SYNTH-{c}") for c in contract_columns(CARS)} for r in rows]
    df = pd.DataFrame(records, columns=list(contract_columns(CARS)))
    return df.astype(dict(DATASET_DEFINITIONS[CARS].identifier_dtypes))


def attr(report: VehicleStabilityReport, column: str) -> VehicleAttributeStabilityReport:
    return next(a for a in report.attributes if a.column == column)


def two(**changes) -> pd.DataFrame:
    return frame([obs(t=T[0]), obs(t=T[1], **changes)])


# ------------------------------------------------------------- configuration


def test_definition_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        V.minimum_observations = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        V.attributes[0].mapping["x"] = "y"  # type: ignore[index]
    assert isinstance(V.attributes, tuple) and isinstance(V.entity_key_columns, tuple)


def test_entity_and_attributes_non_empty_and_in_schema():
    columns = DATASET_DEFINITIONS[V.dataset].columns
    assert V.entity_key_columns and V.attributes
    assert set((*V.group_columns, *V.attribute_columns, *V.volatile_columns)) == set(columns)


def test_roles_disjoint_and_every_column_classified_once():
    roles = (*V.entity_key_columns, *V.context_columns, *V.attribute_columns, *V.volatile_columns)
    assert len(roles) == len(set(roles)) == len(DATASET_DEFINITIONS[CARS].columns)


def test_volatile_fields_excluded_from_identity_and_attributes():
    structural = set(V.group_columns) | set(V.attribute_columns)
    assert not structural & set(PRICES)
    assert not structural & set(DATASET_DEFINITIONS[CARS].identifier_columns)
    assert not structural & {f.column for f in TEMPORAL_RECONCILIATION.fields if f.dataset == CARS}
    assert not structural & set(DATASET_DEFINITIONS[CARS].unique_key_columns)


def test_prices_are_central_and_volatile():
    assert V.price_columns == LOCATION_STREAM_COMPARISON.price_columns
    assert set(V.price_columns) <= set(V.volatile_columns)


def test_observation_time_is_central_timestamp():
    field = V.temporal.field(V.observation_time_field)
    assert V.temporal is TEMPORAL_RECONCILIATION and field.kind is TemporalKind.TIMESTAMP
    assert field.dataset == V.dataset and field.column in V.volatile_columns


def test_project_policies():
    assert V.minimum_observations >= 2
    assert all(isinstance(a.missing_policy, MP) for a in V.attributes)
    assert all(a.comparison is AttributeComparisonPolicy.EXACT and not a.mapping for a in V.attributes)
    assert V.context_columns == EXPECTED_LOCATION_COVERAGE.location_columns   # scope: source location
    assert V.canonical_location_grouping is False and not EXPECTED_LOCATION_COVERAGE.aliases
    assert {a.missing_policy for a in V.attributes} <= set(MP)


def _replace(**changes):
    return dataclasses.replace(V, **changes)


@pytest.mark.parametrize("changes", [
    {"entity_key_columns": ()},
    {"attributes": ()},
    {"entity_key_columns": ("synth_missing",)},
    {"entity_key_columns": (KEY, KEY)},
    {"entity_key_columns": (KEY, CAT)},                 # identity overlaps a stable attribute
    {"entity_key_columns": (KEY, PRICES[0]), "volatile_columns": tuple(c for c in V.volatile_columns if c != PRICES[0])},
    {"entity_key_columns": (JOB,), "context_columns": (CTX, KEY),
     "volatile_columns": tuple(c for c in V.volatile_columns if c != JOB)},
    {"volatile_columns": V.volatile_columns[1:]},       # unclassified column
    {"volatile_columns": (*V.volatile_columns, CAT)},   # stable attribute also volatile
    {"attributes": (*V.attributes, V.attributes[0])},
    {"minimum_observations": 1},
    {"minimum_observations": True},
    {"minimum_observations": 2.0},
    {"observation_time_field": (CARS, "synth_missing")},
    {"observation_time_field": next(f.ref for f in V.temporal.fields if f.dataset == CARS
                                    and f.kind is TemporalKind.DATE)},
    {"canonical_location_grouping": True},
    {"location_coverage": EXPECTED_LOCATION_COVERAGE},
    {"same_capture_conflicts_reported": "yes"},
])
def test_invalid_configuration_raises_typed_error(changes):
    with pytest.raises(VehicleStabilityConfigurationError):
        _replace(**changes)


def test_job_identifier_entity_rejected():
    # Moving the job identifier into identity is refused even if classified.
    vol = tuple(c for c in V.volatile_columns if c != JOB)
    with pytest.raises(VehicleStabilityConfigurationError):
        _replace(entity_key_columns=(KEY, JOB), volatile_columns=vol)


def test_time_field_in_identity_rejected():
    vol = tuple(c for c in V.volatile_columns if c != TIME)
    with pytest.raises(VehicleStabilityConfigurationError):
        _replace(entity_key_columns=(KEY, TIME), volatile_columns=vol)


@pytest.mark.parametrize("kwargs", [
    {"column": ""}, {"missing_policy": "required"}, {"comparison": "exact"},
    {"comparison": AttributeComparisonPolicy.AUTHORITATIVE_MAPPING},
    {"mapping": {"a": "b"}},
    {"comparison": AttributeComparisonPolicy.AUTHORITATIVE_MAPPING, "mapping": {"a": None}},
    {"mapping": ["a"]},
])
def test_invalid_attribute_definition(kwargs):
    base = {"column": CAT, "missing_policy": MP.REQUIRED}
    with pytest.raises(VehicleStabilityConfigurationError):
        VehicleAttributeDefinition(**(base | kwargs))


def test_missing_frame_column_is_configuration_error_without_values():
    with pytest.raises(VehicleStabilityConfigurationError) as info:
        assess(two().drop(columns=[SEATS]))
    assert info.value.columns == (SEATS,) and "SYNTH" not in str(info.value)


def test_type_errors():
    with pytest.raises(TypeError):
        assess([])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess(two(), object())  # type: ignore[arg-type]


# ------------------------------------------------------------- stable data


def test_repeated_identical_observations_are_stable():
    r = assess(two())
    assert r.status is St.PASSED and r.fully_stable_entities == 1 and r.distinct_entities == 1
    assert all(a.stable == 1 and a.passes for a in r.attributes)


def test_multiple_stable_vehicles():
    rows = [obs(v, t, **({CAT: "SYNTH-CLASS-B"} if v == V2 else {})) for v in (V1, V2) for t in T[:3]]
    r = assess(frame(rows))
    assert r.fully_stable_entities == 2 and r.entities_with_value_conflicts == 0


def test_row_order_irrelevant():
    rows = [obs(t=T[0]), obs(t=T[1], **{CAT: "SYNTH-CLASS-B"}), obs(V2, T[0]), obs(V2, T[2])]
    assert assess(frame(rows)) == assess(frame(rows[::-1]))


def test_duplicates_do_not_create_instability():
    r = assess(frame([obs(t=T[0])] * 3 + [obs(t=T[1])] * 2))
    assert r.fully_stable_entities == 1 and r.same_capture_conflict_entities == 0


def test_price_variation_ignored():
    rows = [obs(t=t, **{p: f"{9 + i}.99" for p in PRICES}) for i, t in enumerate(T[:3])]
    assert assess(frame(rows)).fully_stable_entities == 1


@pytest.mark.parametrize("column", SEARCH)
def test_other_volatile_variation_ignored(column):
    rows = [obs(t=t, **{column: f"SYNTH-VARIES-{i}"}) for i, t in enumerate(T[:3])]
    assert assess(frame(rows)).fully_stable_entities == 1


# ------------------------------------------------------------- value conflicts


def test_single_changed_attribute_conflicts():
    r = assess(two(**{TRANS: "SYNTH-MANUAL"}))
    assert attr(r, TRANS).value_conflict == 1 and attr(r, CAT).stable == 1
    assert r.value_unstable_only_entities == 1 and r.status is St.VIOLATIONS


def test_multiple_attributes_measured_independently():
    r = assess(two(**{TRANS: "SYNTH-MANUAL", SEATS: 4}))
    assert [attr(r, c).value_conflict for c in (CAT, TRANS, SEATS, BAGS)] == [0, 1, 1, 0]
    assert r.entities_with_value_conflicts == 1


def test_change_and_revert_is_still_conflict():
    rows = [obs(t=T[0]), obs(t=T[1], **{CAT: "SYNTH-CLASS-B"}), obs(t=T[2])]
    assert attr(assess(frame(rows)), CAT).value_conflict == 1


@pytest.mark.parametrize("variant", ["synth-class-a", "SYNTH-CLASS-A ", " SYNTH-CLASS-A", "SYNTH-CLASS-A.",
                                     "SYNTH_CLASS_A"])
def test_exact_comparison_counts_representation_drift(variant):
    assert attr(assess(two(**{CAT: variant})), CAT).value_conflict == 1


def test_zero_is_a_value():
    r = assess(frame([obs(t=T[0], **{BAGS: 0}), obs(t=T[1], **{BAGS: 0})]))
    assert attr(r, BAGS).stable == 1 and attr(r, BAGS).entities_with_missing == 0
    assert attr(assess(frame([obs(t=T[0], **{BAGS: 0}), obs(t=T[1], **{BAGS: 1})])), BAGS).value_conflict == 1


def test_false_is_a_value_distinct_from_zero():
    r = assess(frame([obs(t=T[0], **{BAGS: False}), obs(t=T[1], **{BAGS: False})]))
    assert attr(r, BAGS).stable == 1
    assert attr(assess(frame([obs(t=T[0], **{BAGS: False}), obs(t=T[1], **{BAGS: 0})])), BAGS).value_conflict == 1


# ------------------------------------------------------------- missingness


def test_always_missing_required_fails():
    r = assess(frame([obs(t=T[0], **{SEATS: None}), obs(t=T[1], **{SEATS: None})]))
    a = attr(r, SEATS)
    assert a.always_missing == 1 and a.entities_with_presence_violation == 1 and not a.passes
    assert not r.required_attributes_pass and r.status is St.VIOLATIONS


def test_always_missing_presence_stable_passes():
    r = assess(frame([obs(t=T[0], **{BAGS: None}), obs(t=T[1], **{BAGS: None})]))
    a = attr(r, BAGS)
    assert a.always_missing == 1 and a.passes and r.status is St.PASSED


def test_optional_policy_never_fails_on_missing():
    optional = _replace(attributes=tuple(dataclasses.replace(a, missing_policy=MP.MISSING_IGNORED)
                                         if a.column == BAGS else a for a in V.attributes))
    r = assess(frame([obs(t=T[0]), obs(t=T[1], **{BAGS: None})]), optional)
    a = attr(r, BAGS)
    assert a.intermittently_missing == 1 and a.passes and a.present_to_missing == 1 and r.status is St.PASSED


def test_present_to_missing_transition():
    a = attr(assess(frame([obs(t=T[0]), obs(t=T[1], **{BAGS: None})])), BAGS)
    assert a.present_to_missing == 1 and a.missing_to_present == 0 and not a.passes


def test_missing_to_present_transition():
    a = attr(assess(frame([obs(t=T[1]), obs(t=T[0], **{BAGS: None})])), BAGS)
    assert a.missing_to_present == 1 and a.present_to_missing == 0 and not a.passes


def test_missing_is_not_a_sentinel_value():
    df = frame([obs(t=T[0], **{CAT: None}), obs(t=T[1], **{CAT: "None"}), obs(t=T[2], **{CAT: "None"})])
    a = attr(assess(df), CAT)
    assert a.value_conflict == 0 and a.entities_with_missing == 1   # "None" text vs real missing
    assert df[CAT].isna().sum() == 1


def test_missingness_separate_from_conflict():
    rows = [obs(t=T[0]), obs(t=T[1], **{SEATS: None}), obs(t=T[2], **{SEATS: 7})]
    a = attr(assess(frame(rows)), SEATS)
    assert a.value_conflict == 1 and a.entities_with_missing == 1 and a.intermittently_missing == 0


def test_intermittent_missing_creates_no_alternate_value():
    rows = [obs(t=T[0]), obs(t=T[1], **{BAGS: None}), obs(t=T[2])]
    a = attr(assess(frame(rows)), BAGS)
    assert a.value_conflict == 0 and a.intermittently_missing == 1


# ------------------------------------------------------------- history and time


def test_single_observation_insufficient():
    r = assess(frame([obs()]))
    assert r.insufficient_history_entities == 1 and r.fully_stable_entities == 0
    assert r.status is St.UNASSESSABLE and attr(r, CAT).insufficient_history == 1
    with pytest.raises(VehicleAttributeStabilityError) as info:
        validate(frame([obs()]))
    # Missing evidence is a blocking reason, not a proven violation.
    assert info.value.blocking_reasons == ("insufficient_history",) and info.value.violations == ()


def test_same_instant_duplicates_are_one_capture():
    r = assess(frame([obs(), obs()]))
    assert r.insufficient_history_entities == 1


def test_two_captures_meet_default_minimum():
    assert assess(two()).sufficient_history_entities == 1


def test_higher_minimum_respected():
    assert assess(two(), _replace(minimum_observations=3)).insufficient_history_entities == 1


@pytest.mark.parametrize("bad", [None, "", "SYNTH-NOT-A-TIME", "2025-02-30 05:00:00 MST", "2025-01-15 05:00:00 XYZ",
                                 "2025-01-15 05:00:00"])
def test_missing_or_invalid_time_unassessable(bad):
    r = assess(frame([obs(t=T[0]), obs(t=bad), obs(t=T[2])]))
    assert r.temporally_unassessable_entities == 1 and r.sufficient_history_entities == 0
    assert r.status is St.VIOLATIONS and "temporally_unassessable" in r.violations


def test_set_based_conflict_detected_without_valid_time():
    r = assess(frame([obs(t=None), obs(t=None, **{CAT: "SYNTH-CLASS-B"})]))
    assert attr(r, CAT).value_conflict == 1 and attr(r, CAT).present_to_missing == 0


def test_same_capture_identical_is_not_conflict():
    r = assess(frame([obs(t=T[0]), obs(t=T[0]), obs(t=T[1])]))
    assert r.same_capture_conflict_entities == 0 and r.status is St.PASSED


def test_same_capture_conflict_detected():
    r = assess(frame([obs(t=T[0]), obs(t=T[0], **{CAT: "SYNTH-CLASS-B"}), obs(t=T[1])]))
    assert r.same_capture_conflict_entities == 1 and attr(r, CAT).same_capture_conflicts == 1
    assert "same_capture_conflict" in r.violations and r.entities_with_value_conflicts == 1


def test_same_capture_reporting_flag():
    df = frame([obs(t=T[0]), obs(t=T[0], **{CAT: "SYNTH-CLASS-B"})])
    r = assess(df, _replace(same_capture_conflicts_reported=False))
    assert r.same_capture_conflict_entities == 0 and r.entities_with_value_conflicts == 1


def test_out_of_order_input_ordered_by_reconciled_time():
    rows = [obs(t=T[2], **{BAGS: None}), obs(t=T[0]), obs(t=T[1])]   # present, present, missing in time
    a = attr(assess(frame(rows)), BAGS)
    assert a.present_to_missing == 1 and a.missing_to_present == 0


def test_designator_time_is_reconciled_not_textual():
    # Same instant in text order differs from instant order: 23:00 MST on the 14th < 05:00 MST on the 15th.
    rows = [obs(t="2025-01-15 05:00:00 MST"), obs(t="2025-01-14 23:00:00 MST", **{BAGS: None})]
    a = attr(assess(frame(rows)), BAGS)
    assert a.missing_to_present == 1


# ------------------------------------------------------------- identity scope


def test_distinct_vehicles_not_combined():
    r = assess(frame([obs(V1, T[0]), obs(V1, T[1]), obs(V2, T[0], **{CAT: "SYNTH-CLASS-B"}),
                      obs(V2, T[1], **{CAT: "SYNTH-CLASS-B"})]))
    assert r.distinct_entities == 2 and r.entities_with_value_conflicts == 0


def test_location_scope_keeps_locations_separate():
    rows = [obs(t=T[0]), obs(t=T[1]), obs(loc=L2, t=T[0], **{BAGS: 3}), obs(loc=L2, t=T[1], **{BAGS: 3})]
    r = assess(frame(rows))
    assert r.distinct_entities == 2 and r.status is St.PASSED


def test_global_scope_combines_locations_only_when_configured():
    rows = [obs(t=T[0]), obs(t=T[1]), obs(loc=L2, t=T[0], **{BAGS: 3}), obs(loc=L2, t=T[1], **{BAGS: 3})]
    global_scope = _replace(context_columns=(), volatile_columns=(*V.volatile_columns, CTX))
    r = assess(frame(rows), global_scope)
    assert r.distinct_entities == 1 and attr(r, BAGS).value_conflict == 1


def test_job_and_time_do_not_split_entities():
    rows = [obs(t=t, **{JOB: f"SYNTH-JOB-00{i}"}) for i, t in enumerate(T[:3])]
    assert assess(frame(rows)).distinct_entities == 1


def test_price_does_not_split_entities():
    rows = [obs(t=t, **{p: f"{i}.00" for p in PRICES}) for i, t in enumerate(T[:3])]
    assert assess(frame(rows)).distinct_entities == 1


@pytest.mark.parametrize("missing", ["key", "ctx", "both"])
def test_incomplete_identity_counted_separately(missing):
    bad = {"key": {KEY: None}, "ctx": {CTX: None}, "both": {KEY: None, CTX: None}}[missing]
    r = assess(frame([obs(t=T[0]), obs(t=T[1]), obs(t=T[0], **bad), obs(t=T[1], **bad)]))
    assert r.incomplete_identity_entities == 1 and r.incomplete_identity_observations == 2
    assert r.complete_identity_entities == 1 and r.distinct_entities == 2
    assert "incomplete_identity" in r.violations


def test_composite_keys_do_not_collide():
    # Concatenation would make ("SYNTH-AB", "C") equal to ("SYNTH-A", "BC").
    rows = [obs("C", T[0], loc="SYNTH-AB"), obs("C", T[1], loc="SYNTH-AB"),
            obs("BC", T[0], loc="SYNTH-A", **{CAT: "SYNTH-CLASS-B"}), obs("BC", T[1], loc="SYNTH-A", **{CAT: "SYNTH-CLASS-B"})]
    r = assess(frame(rows))
    assert r.distinct_entities == 2 and r.entities_with_value_conflicts == 0


def test_delimiter_like_values_do_not_collide():
    rows = [obs("SYNTH|1", T[0], loc="SYNTH|A"), obs("SYNTH|1", T[1], loc="SYNTH|A"),
            obs("1", T[0], loc="SYNTH|A|SYNTH", **{CAT: "SYNTH-CLASS-B"}),
            obs("1", T[1], loc="SYNTH|A|SYNTH", **{CAT: "SYNTH-CLASS-B"})]
    assert assess(frame(rows)).distinct_entities == 2


# ------------------------------------------------------------- aliases


ALIAS_COV = dataclasses.replace(EXPECTED_LOCATION_COVERAGE, expected_locations=((L1,),), aliases={(L1,): ((L2,),)})
SPLIT = [obs(loc=L1, t=T[0]), obs(loc=L2, t=T[1])]


def test_no_confirmed_alias_keeps_source_locations():
    r = assess(frame(SPLIT))
    assert r.distinct_entities == 2 and r.insufficient_history_entities == 2


def test_unverified_alias_not_applied():
    # An alias is not used unless the stability contract enables canonical grouping.
    unconfigured = dataclasses.replace(EXPECTED_LOCATION_COVERAGE, expected_locations=((L1,),))
    assert not unconfigured.aliases
    assert assess(frame(SPLIT)).distinct_entities == 2


def test_confirmed_alias_groups_only_when_configured():
    canonical = _replace(canonical_location_grouping=True, location_coverage=ALIAS_COV)
    df = frame(SPLIT)
    before = df.copy(deep=True)
    r = assess(df, canonical)
    assert r.distinct_entities == 1 and r.sufficient_history_entities == 1 and r.status is St.PASSED
    pd.testing.assert_frame_equal(df, before)                 # raw labels preserved
    assert df[CTX].tolist() == [L1, L2]


def test_alias_handling_keeps_linkage_columns():
    canonical = _replace(canonical_location_grouping=True, location_coverage=ALIAS_COV)
    df = frame(SPLIT)
    assess(df, canonical)
    assert df[JOB].tolist() == ["SYNTH-JOB-001"] * 2 and str(df[JOB].dtype) == str(frame(SPLIT)[JOB].dtype)


# ------------------------------------------------------------- invariants and safety


def _mixed() -> pd.DataFrame:
    return frame([
        obs(V1, T[0]), obs(V1, T[1]),                                         # stable
        obs(V2, T[0]), obs(V2, T[1], **{CAT: "SYNTH-CLASS-B"}),               # value
        obs("SYNTH-VEHICLE-003", T[0]), obs("SYNTH-VEHICLE-003", T[1], **{BAGS: None}),  # presence
        obs("SYNTH-VEHICLE-004", T[0]), obs("SYNTH-VEHICLE-004", T[1], **{SEATS: 9, BAGS: None}),  # both
        obs("SYNTH-VEHICLE-005", T[0]),                                       # insufficient
        obs("SYNTH-VEHICLE-006", None), obs("SYNTH-VEHICLE-006", T[1]),       # unassessable
        obs(None, T[0]),                                                      # incomplete
    ])


def test_aggregate_categories_reconcile():
    r = assess(_mixed())
    assert r.invariants_hold
    assert (r.fully_stable_entities, r.value_unstable_only_entities, r.presence_unstable_only_entities,
            r.value_and_presence_unstable_entities) == (1, 1, 1, 1)
    assert (r.insufficient_history_entities, r.temporally_unassessable_entities,
            r.incomplete_identity_entities) == (1, 1, 1)
    assert r.distinct_entities == 7 and r.observations_assessed == 12


def test_attribute_categories_reconcile_and_exclusive():
    r = assess(_mixed())
    for a in r.attributes:
        assert a.categories_reconcile and a.entities_assessed == r.complete_identity_entities
        assert min(a.value_conflict, a.stable, a.entities_with_missing, a.present_to_missing) >= 0
    assert r.fully_stable_entities + r.entities_with_value_conflicts <= r.complete_identity_entities


def test_overall_validity_follows_policies():
    assert not assess(_mixed()).is_valid
    r = assess(frame([obs(t=T[0], **{BAGS: None}), obs(t=T[1], **{BAGS: None})]))
    assert r.required_attributes_pass and r.all_attributes_pass and r.is_valid


def test_report_contains_no_frames_keys_values_or_times():
    r = assess(_mixed())
    for item in (r, *r.attributes):
        for field in dataclasses.fields(item):
            value = getattr(item, field.name)
            assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
    text = repr(r)
    assert "SYNTH" not in text and "2025" not in text and "MST" not in text
    assert {a.column for a in r.attributes} == set(V.attribute_columns)


def test_strict_validation_passes_on_stable():
    assert validate(two()).is_valid


@pytest.mark.parametrize("df, category", [
    (lambda: two(**{CAT: "SYNTH-CLASS-B"}), "value_conflict"),
    (lambda: frame([obs(t=T[0]), obs(t=T[1], **{SEATS: None})]), "presence_instability"),
    (lambda: frame([obs(t=T[0]), obs(t="SYNTH-NOT-A-TIME")]), "temporally_unassessable"),
    (lambda: frame([obs(t=T[0]), obs(t=T[1]), obs(None, T[0])]), "incomplete_identity"),
])
def test_strict_validation_raises_safe_categories(df, category):
    with pytest.raises(VehicleAttributeStabilityError) as info:
        validate(df())
    assert category in info.value.violations
    assert "SYNTH" not in str(info.value) and "2025" not in str(info.value)
    assert isinstance(info.value.report, VehicleStabilityReport)


def test_strict_validation_raises_for_invalid_configuration():
    with pytest.raises(VehicleStabilityConfigurationError):
        validate(two(), _replace(minimum_observations=0))


def test_assessment_does_not_raise_for_instability():
    assert assess(two(**{CAT: "SYNTH-CLASS-B"})).status is St.VIOLATIONS


def test_empty_dataset_is_unassessable_not_proven():
    r = assess(frame([]))
    assert r.status is St.UNASSESSABLE and r.distinct_entities == 0 and r.invariants_hold
    with pytest.raises(VehicleAttributeStabilityError):
        validate(frame([]))


# ------------------------------------------------------------- non-mutation


def test_source_unchanged_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _mixed()
    df.index = pd.Index([f"SYNTH-IDX-{i}" for i in range(len(df))])
    before, dtypes, index = df.copy(deep=True), df.dtypes.copy(), df.index.copy()
    first = assess(df)
    assert assess(df) == first
    pd.testing.assert_frame_equal(df, before)
    assert df.index.equals(index) and df.dtypes.equals(dtypes) and list(df.columns) == list(before.columns)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------- pipeline


def test_blank_rows_must_be_removed_first():
    df = frame([obs(t=T[0]), obs(t=T[1])])
    blank = pd.DataFrame([[None] * df.shape[1]], columns=df.columns, dtype=object).astype(
        dict(DATASET_DEFINITIONS[CARS].identifier_dtypes))
    with_blank = pd.concat([df, blank], ignore_index=True)
    with pytest.raises(VehicleStabilityPreconditionError) as info:
        assess(with_blank)
    assert info.value.reason == "blank_rows_present"
    cleaned = remove_completely_blank_rows(with_blank).cleaned
    r = assess(cleaned)
    assert r.incomplete_identity_entities == 0 and r.status is St.PASSED


def test_partially_populated_rows_are_assessed():
    partial = {c: None for c in contract_columns(CARS)} | {CTX: L1, TIME: T[2]}
    r = assess(frame([obs(t=T[0]), obs(t=T[1]), partial]))
    assert r.observations_assessed == 3 and r.incomplete_identity_observations == 1


def test_identifier_dtype_precondition():
    with pytest.raises(VehicleStabilityPreconditionError):
        assess(two().astype({JOB: object}))


def test_duplicate_detail_keys_not_removed():
    rows = [obs(t=T[0], **{"row_index": 1}), obs(t=T[0], **{"row_index": 1}), obs(t=T[1])] \
        if "row_index" in contract_columns(CARS) else [obs(t=T[0]), obs(t=T[0]), obs(t=T[1])]
    assert assess(frame(rows)).observations_assessed == 3


def test_package_exports():
    for name in ("assess_vehicle_attribute_stability", "validate_vehicle_attribute_stability",
                 "VEHICLE_ATTRIBUTE_STABILITY", "VehicleStabilityDefinition", "VehicleStabilityReport",
                 "VehicleAttributeStabilityError", "MissingValueStabilityPolicy"):
        assert name in ql2_sixt_canada_analysis.__all__


def test_authoritative_mapping_applied_non_destructively():
    mapped = _replace(attributes=tuple(
        dataclasses.replace(a, comparison=AttributeComparisonPolicy.AUTHORITATIVE_MAPPING,
                            mapping={"SYNTH-CLASS-A-OLD": "SYNTH-CLASS-A"}) if a.column == CAT else a
        for a in V.attributes))
    df = two(**{CAT: "SYNTH-CLASS-A-OLD"})
    before = df.copy(deep=True)
    assert attr(assess(df, mapped), CAT).value_conflict == 0
    assert attr(assess(df), CAT).value_conflict == 1           # exact by default
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------- full-population validity (history)

V3 = "SYNTH-VEHICLE-003"


def history_of(df: pd.DataFrame) -> dict[str, str]:
    entities = classify(df)
    return dict(zip(entities[KEY], entities["history"]))


def vehicle_attributes_stable(report: VehicleStabilityReport) -> bool:
    """The notebook-facing gate: the report's own validity, nothing else."""
    return report is not None and report.is_valid


def test_mixed_sufficient_and_insufficient_history_is_not_passed():
    # Original defect: one vehicle with two captures, one with one capture -> was PASSED / valid.
    df = frame([obs(V1, T[0]), obs(V1, T[1]), obs(V2, T[0])])
    r = assess(df)
    assert r.status is St.PARTIALLY_ASSESSABLE and r.is_valid is False
    assert (r.distinct_entities, r.sufficient_history_entities, r.insufficient_history_entities) == (2, 1, 1)
    assert r.violations == () and r.blocking_reasons == ("insufficient_history",)
    assert r.fully_stable_entities == 1 and not r.fully_assessed and r.invariants_hold
    assert history_of(df) == {V1: H.SUFFICIENT_HISTORY.value, V2: H.INSUFFICIENT_HISTORY.value}
    assert vehicle_attributes_stable(r) is False
    with pytest.raises(VehicleAttributeStabilityError) as info:
        validate(df)
    assert info.value.report.status is St.PARTIALLY_ASSESSABLE
    assert info.value.blocking_reasons == ("insufficient_history",)
    assert "partially_assessable" in str(info.value) and "SYNTH" not in str(info.value)


def test_full_population_with_sufficient_stable_history_passes():
    df = frame([obs(v, t) for v in (V1, V2, V3) for t in T[:2]])
    r = assess(df)
    assert r.status is St.PASSED and r.is_valid and r.fully_assessed
    assert (r.sufficient_history_entities, r.insufficient_history_entities, r.fully_stable_entities) == (3, 0, 3)
    assert r.violations == () and r.blocking_reasons == ()
    entities = classify(df)
    assert entities["stable"].all() and set(entities["history"]) == {H.SUFFICIENT_HISTORY.value}
    assert vehicle_attributes_stable(r) is True and validate(df) == r


def test_entirely_insufficient_population_is_unassessable():
    df = frame([obs(V1, T[0]), obs(V2, T[1]), obs(V3, T[2])])
    r = assess(df)
    assert r.status is St.UNASSESSABLE and not r.is_valid
    assert (r.sufficient_history_entities, r.insufficient_history_entities, r.fully_stable_entities) == (0, 3, 0)
    assert r.blocking_reasons == ("insufficient_history",)
    entities = classify(df)
    assert set(entities["history"]) == {H.INSUFFICIENT_HISTORY.value} and not entities["stable"].any()


def test_fully_assessed_population_with_violation_fails_and_names_entity_and_attribute():
    df = frame([obs(V1, T[0]), obs(V1, T[1], **{CAT: "SYNTH-CLASS-B"}), obs(V2, T[0]), obs(V2, T[1])])
    r = assess(df)
    assert r.status is St.VIOLATIONS and not r.is_valid and r.insufficient_history_entities == 0
    assert r.violations == ("value_conflict",) and r.blocking_reasons == ("value_conflict",)
    assert attr(r, CAT).value_conflict == 1
    entities = classify(df).set_index(KEY)
    assert entities.loc[V1, "unstable_attributes"] == (CAT,) and entities.loc[V1, "value_conflict"]
    assert entities.loc[V2, "stable"] and entities.loc[V2, "unstable_attributes"] == ()


def test_violations_and_insufficient_history_are_both_preserved():
    df = frame([obs(V1, T[0]), obs(V1, T[1], **{TRANS: "SYNTH-MANUAL"}),   # sufficient, unstable
                obs(V2, T[0]), obs(V2, T[1]),                              # sufficient, stable
                obs(V3, T[0])])                                            # insufficient
    r = assess(df)
    assert r.status is St.VIOLATIONS and not r.is_valid           # proven violations take precedence
    assert r.violations == ("value_conflict",)
    assert r.blocking_reasons == ("value_conflict", "insufficient_history")
    assert (r.sufficient_history_entities, r.insufficient_history_entities, r.fully_stable_entities,
            r.entities_with_value_conflicts) == (2, 1, 1, 1)
    entities = classify(df).set_index(KEY)
    assert entities.loc[V1, "unstable_attributes"] == (TRANS,)
    assert entities.loc[V2, "stable"] and entities.loc[V3, "history"] == H.INSUFFICIENT_HISTORY.value
    with pytest.raises(VehicleAttributeStabilityError) as info:
        validate(df)
    assert info.value.blocking_reasons == ("value_conflict", "insufficient_history")


def test_minimum_history_boundary_is_inclusive():
    three = _replace(minimum_observations=3)
    df = frame([obs(V1, T[0]), obs(V1, T[1]),                     # one below the threshold
                obs(V2, T[0]), obs(V2, T[1]), obs(V2, T[2])])     # exactly at the threshold
    r = assess(df, three)
    assert r.status is St.PARTIALLY_ASSESSABLE
    assert classify(df, three).set_index(KEY)["history"].to_dict() == {
        V1: H.INSUFFICIENT_HISTORY.value, V2: H.SUFFICIENT_HISTORY.value}


def test_duplicate_rows_in_one_capture_do_not_satisfy_history():
    df = frame([obs(V1, T[0])] * 3 + [obs(V2, T[0]), obs(V2, T[1])])
    r = assess(df)
    assert r.status is St.PARTIALLY_ASSESSABLE and r.insufficient_history_entities == 1
    assert history_of(df)[V1] == H.INSUFFICIENT_HISTORY.value


def test_empty_population_is_never_a_validated_stable_population():
    r = assess(frame([]))
    assert r.status is St.UNASSESSABLE and not r.is_valid and not r.fully_assessed
    assert r.blocking_reasons == ("empty_population",) and r.violations == ()
    assert len(classify(frame([]))) == 0 and vehicle_attributes_stable(r) is False
    with pytest.raises(VehicleAttributeStabilityError):
        validate(frame([]))


def test_row_order_does_not_change_status_or_classification():
    rows = [obs(V1, T[0]), obs(V1, T[1], **{CAT: "SYNTH-CLASS-B"}), obs(V2, T[0]), obs(V2, T[1]), obs(V3, T[0])]
    a, b = frame(rows), frame(rows[::-1])
    assert assess(a) == assess(b)
    pd.testing.assert_frame_equal(classify(a), classify(b))


@pytest.mark.parametrize("rows", [
    [obs(V1, T[0]), obs(V1, T[1])],
    [obs(V1, T[0]), obs(V1, T[1]), obs(V2, T[0])],
    [obs(V1, T[0]), obs(V2, T[0])],
    [obs(V1, T[0]), obs(V1, T[1], **{CAT: "SYNTH-CLASS-B"}), obs(V2, T[0])],
    [obs(V1, None), obs(V1, T[1]), obs(None, T[0]), obs(V2, T[0])],
    [],
])
def test_status_invariants_hold_for_every_population(rows):
    df = frame(rows)
    r = assess(df)
    assert r.invariants_hold and r.is_valid == (r.status is St.PASSED)
    if r.status is St.PASSED:
        assert r.insufficient_history_entities == 0 and r.violations == () and r.blocking_reasons == ()
    else:
        assert r.blocking_reasons, "a non-passing population always says why"
    entities = classify(df)
    counts = entities["history"].value_counts().to_dict()
    assert len(entities) == r.distinct_entities                      # nobody dropped
    assert counts.get(H.SUFFICIENT_HISTORY.value, 0) == r.sufficient_history_entities
    assert counts.get(H.INSUFFICIENT_HISTORY.value, 0) == r.insufficient_history_entities
    assert counts.get(H.TEMPORALLY_UNASSESSABLE.value, 0) == r.temporally_unassessable_entities
    assert counts.get(H.INCOMPLETE_IDENTITY.value, 0) == r.incomplete_identity_entities
    assert int(entities["stable"].sum()) == r.fully_stable_entities


def test_status_and_history_values_are_stable_strings():
    assert [s.value for s in St] == ["passed", "violations", "partially_assessable", "unassessable"]
    assert [h.value for h in H] == ["incomplete_identity", "temporally_unassessable",
                                    "insufficient_history", "sufficient_history"]
    assert str(St.PARTIALLY_ASSESSABLE) == "partially_assessable"
    assert "PARTIALLY_ASSESSABLE" not in repr(assess(frame([obs(V1, T[0])])).blocking_reasons)


def test_entity_classification_is_isolated_from_callers():
    df = frame([obs(V1, T[0]), obs(V1, T[1]), obs(V2, T[0])])
    before = df.copy(deep=True)
    first = classify(df)
    first.loc[:, "history"] = H.SUFFICIENT_HISTORY.value
    first.loc[:, "stable"] = True
    second = classify(df)
    assert second["history"].tolist() == [H.SUFFICIENT_HISTORY.value, H.INSUFFICIENT_HISTORY.value]
    assert assess(df).status is St.PARTIALLY_ASSESSABLE
    pd.testing.assert_frame_equal(df, before)
    assert isinstance(second.loc[0, "unstable_attributes"], tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        assess(df).status = St.PASSED  # type: ignore[misc]


def test_classification_withholds_nothing_from_memory_but_report_holds_no_keys():
    df = frame([obs(V1, T[0]), obs(V2, T[0])])
    assert set(classify(df)[KEY]) == {V1, V2}
    assert "SYNTH" not in repr(assess(df))
