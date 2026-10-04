"""Tests for the generic location-stream comparison.

Every value is fabricated (``SYNTH-BRANCH-A``, ``SYNTH-JOB-001``,
``SYNTH-CAR-X``, ``10.00``, ...). Column names come from the central
definitions; the real compared pair is referenced only through
``COMPARED_LOCATION_STREAMS`` / ``LOCATION_STREAM_COMPARISON``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from conftest import contract_columns

import ql2_sixt_canada_analysis
from ql2_sixt_canada_analysis.comparison import (
    OfferDefect,
    ComparisonPreconditionError,
    DuplicateInferenceBlocker as DB,
    IdentityEvidence,
    LocationAliasNotConfirmedError,
    LocationStreamComparisonReport,
    LocationStreamComparisonStatus as S,
    OfferSetResult as O,
    ScopeBaseline,
    TemporalOverlap,
    canonical_location_keys,
    compare_location_streams,
    validate_confirmed_location_alias,
)
from ql2_sixt_canada_analysis.coverage import assess_expected_location_coverage
from ql2_sixt_canada_analysis.schemas import (
    COMPARED_LOCATION_STREAMS,
    EXPECTED_LOCATION_COVERAGE,
    INVESTIGATED_LOCATION_STREAM,
    JOB_DETAIL_RELATIONSHIP,
    LOCATION_STREAM_COMPARISON,
    CapturePairing,
    DatasetKey,
    LocationCoverageConfigurationError,
    LocationStreamComparisonDefinition,
    TemporalAwareness,
)

JOBS, CARS = DatasetKey.JOBS, DatasetKey.CARS
REL = JOB_DETAIL_RELATIONSHIP
PK, DK, COUNT = REL.parent_key_columns[0], REL.detail_key_columns[0], REL.expected_detail_count_column
LOC = EXPECTED_LOCATION_COVERAGE.label_column        # synthetic single-label contracts below
SCOPE = EXPECTED_LOCATION_COVERAGE.stream_scope_columns[0]
D0 = LOCATION_STREAM_COMPARISON
PRODUCT, PRICE = D0.product_columns, D0.price_columns
A, B, C, D = "SYNTH-BRANCH-A", "SYNTH-BRANCH-B", "SYNTH-BRANCH-C", "SYNTH-BRANCH-D"
CITY, CITY2 = "SYNTH-CITY-1", "SYNTH-CITY-2"
J1, J2, J3 = "SYNTH-JOB-001", "SYNTH-JOB-002", "SYNTH-JOB-003"
TIME_FIELD = next(f for f in D0.temporal.fields
                  if f.dataset == CARS and f.awareness is TemporalAwareness.DESIGNATOR)
TECHNICAL = {*REL.detail_key_columns, *REL.detail_definition.identifier_columns,
             *REL.detail_definition.unique_key_columns, LOC, SCOPE,
             *(f.column for f in D0.temporal.fields if f.dataset == CARS)}
ID_COL = next(c for c in contract_columns(CARS) if c not in TECHNICAL | set(PRODUCT) | set(PRICE))
COV = dataclasses.replace(EXPECTED_LOCATION_COVERAGE, location_columns=(LOC,), label_column=LOC,
                          expected_locations=((A,), (B,), (C,)))
DEF = dataclasses.replace(D0, first=(A,), second=(B,), coverage=COV)


def offer(car: str = "SYNTH-CAR-X", price: str = "10.00") -> dict:
    row = {c: f"SYNTH-{c}" for c in PRODUCT}
    row[PRODUCT[0]] = car
    row.update({c: price for c in PRICE})
    return row


def _jobs(keys: tuple[str, ...] = (J1, J2, J3)) -> pd.DataFrame:
    frame = pd.DataFrame({c: [f"SYNTH-{c}"] * len(keys) for c in contract_columns(JOBS)},
                         columns=list(contract_columns(JOBS)))
    frame[PK] = list(keys)
    frame[COUNT] = pd.Series([1] * len(keys), dtype="int64")
    return frame.astype(dict(REL.parent_definition.identifier_dtypes))


def _cars(rows: list[tuple[str, str, dict]], scope: str = CITY) -> pd.DataFrame:
    """rows: (job key, location, offer fields)."""
    records = []
    for job, loc, fields in rows:
        rec = {c: f"SYNTH-{c}" for c in contract_columns(CARS)}
        rec.update({DK: job, LOC: loc, SCOPE: fields.get(SCOPE, scope)})
        rec.update({k: v for k, v in fields.items() if k != SCOPE})
        records.append(rec)
    frame = pd.DataFrame(records, columns=list(contract_columns(CARS)))
    return frame.astype(dict(REL.detail_definition.identifier_dtypes))


def same_both(jobs=(J1, J2), offers=(offer(),), extra=()):
    rows = [(j, loc, o) for j in jobs for loc in (A, B) for o in offers]
    return _cars(rows + list(extra))


# ------------------------------------------------------------- configuration


def test_project_pair_defined_once_and_expected():
    assert LOCATION_STREAM_COMPARISON.first == COMPARED_LOCATION_STREAMS[0]
    assert LOCATION_STREAM_COMPARISON.second == COMPARED_LOCATION_STREAMS[1]
    for key in COMPARED_LOCATION_STREAMS:
        assert key in EXPECTED_LOCATION_COVERAGE.expected_locations
    assert INVESTIGATED_LOCATION_STREAM in EXPECTED_LOCATION_COVERAGE.expected_locations


def test_project_definition_is_immutable_and_has_no_alias_or_identity():
    with pytest.raises(dataclasses.FrozenInstanceError):
        LOCATION_STREAM_COMPARISON.first = ("x",)  # type: ignore[misc]
    assert LOCATION_STREAM_COMPARISON.identity_columns == ()
    assert not EXPECTED_LOCATION_COVERAGE.aliases
    assert LOCATION_STREAM_COMPARISON.pairing is CapturePairing.SHARED_COLLECTION_EVENT
    assert LOCATION_STREAM_COMPARISON.pairing_tolerance is None


def test_target_literals_appear_only_in_schemas():
    root = Path(__file__).resolve().parents[1]
    # Branch labels identify the streams; the city component is a common word (e.g. in names).
    names = [key[-1] for key in COMPARED_LOCATION_STREAMS]
    hits = {p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")
            if any(n in p.read_text(encoding="utf-8") for n in names)}
    assert hits == {"src/ql2_sixt_canada_analysis/schemas.py"}
    schemas = (root / "src/ql2_sixt_canada_analysis/schemas.py").read_text(encoding="utf-8")
    assert all(schemas.count(n) == 1 for n in names)
    assert not any(n in (root / "tests" / "test_comparison.py").read_text(encoding="utf-8") for n in names)


@pytest.mark.parametrize("changes", [
    {"first": ("SYNTH-UNEXPECTED",)},
    {"second": (A,)},
    {"first": (A, B)},
    {"product_columns": ()},
    {"product_columns": (PRICE[0],)},
    {"product_columns": (DK,)},
    {"product_columns": (LOC,)},
    {"product_columns": (SCOPE,)},
    {"product_columns": (TIME_FIELD.column,)},
    {"product_columns": ("synth_missing_column",)},
    {"product_columns": (PRODUCT[0], PRODUCT[0])},
    {"price_columns": (DK,)},
    {"identity_columns": (PRODUCT[0],)},
    {"pairing": "shared_collection_event"},
    {"pairing": CapturePairing.CAPTURE_TIME},
    {"pairing_tolerance": dt.timedelta(0)},
    {"pairing": CapturePairing.CAPTURE_TIME, "capture_time_field": TIME_FIELD.ref,
     "pairing_tolerance": dt.timedelta(seconds=-1)},
    {"pairing": CapturePairing.CAPTURE_TIME, "capture_time_field": next(
        f.ref for f in D0.temporal.fields if f.dataset == CARS and f.kind.value != "timestamp"),
     "pairing_tolerance": dt.timedelta(0)},
    {"pairing": CapturePairing.CAPTURE_TIME, "capture_time_field": next(
        f.ref for f in D0.temporal.fields if f.dataset == JOBS and f.kind.value == "timestamp"),
     "pairing_tolerance": dt.timedelta(0)},
])
def test_invalid_definitions_rejected(changes):
    with pytest.raises(LocationCoverageConfigurationError):
        dataclasses.replace(DEF, **changes)


def test_unconfigured_coverage_rejected():
    with pytest.raises(LocationCoverageConfigurationError):
        dataclasses.replace(DEF, coverage=dataclasses.replace(COV, expected_locations=()))


def test_type_checks():
    with pytest.raises(TypeError):
        compare_location_streams(_jobs(), same_both(), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compare_location_streams([], same_both(), DEF)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        canonical_location_keys([], COV)  # type: ignore[arg-type]


def test_missing_configured_column_reported_by_name_only():
    with pytest.raises(LocationCoverageConfigurationError) as info:
        compare_location_streams(_jobs(), same_both().drop(columns=[PRODUCT[1]]), DEF)
    assert "SYNTH" not in str(info.value)


# ------------------------------------------------------------- preconditions


def test_identifier_dtype_precondition():
    cars = same_both().astype({DK: object})
    with pytest.raises(ComparisonPreconditionError):
        compare_location_streams(_jobs(), cars, DEF)


def test_duplicate_parent_key_precondition():
    with pytest.raises(ComparisonPreconditionError):
        compare_location_streams(_jobs((J1, J1)), same_both(), DEF)


def test_blank_rows_precondition():
    cars = same_both()
    blank = pd.DataFrame([[pd.NA] * cars.shape[1]], columns=cars.columns).astype(cars.dtypes.to_dict())
    with pytest.raises(ComparisonPreconditionError):
        compare_location_streams(_jobs(), pd.concat([cars, blank], ignore_index=True), DEF)


def test_precondition_error_has_no_source_values():
    with pytest.raises(ComparisonPreconditionError) as info:
        compare_location_streams(_jobs((J1, J1)), same_both(), DEF)
    assert "SYNTH" not in str(info.value)


# ---------------------------------------------------------------- presence


def test_both_absent():
    r = compare_location_streams(_jobs(), _cars([(J1, C, offer())]), DEF)
    assert r.status is S.BOTH_STREAMS_ABSENT
    assert not r.first_present and not r.second_present
    assert r.first_details_linked is None and r.product_sets is O.UNAVAILABLE
    assert r.upstream_review_required


@pytest.mark.parametrize("present", [A, B])
def test_one_absent(present):
    r = compare_location_streams(_jobs(), _cars([(J1, present, offer()), (J1, C, offer())]), DEF)
    assert r.status is S.ONE_STREAM_ABSENT
    assert r.first_present is (present == A) and r.second_present is (present == B)
    assert r.comparable_captures_exist is False and r.synchronized_prices is None


def test_label_matching_is_exact():
    cars = _cars([(J1, A.lower(), offer()), (J1, f" {B}", offer()), (J1, f"{A}.", offer())])
    assert compare_location_streams(_jobs(), cars, DEF).status is S.BOTH_STREAMS_ABSENT


def test_linkage_flags():
    cars = same_both(jobs=(J1, "SYNTH-ORPHAN"))
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.first_details_linked is False and r.second_details_linked is False
    assert compare_location_streams(_jobs(), same_both(), DEF).first_details_linked is True


# ------------------------------------------------- behavioural conclusions


def test_identical_offers_discriminative_scope_likely_duplicate():
    extra = [(J1, C, offer("SYNTH-CAR-Y")), (J2, C, offer("SYNTH-CAR-Y"))]
    r = compare_location_streams(_jobs(), same_both(extra=extra), DEF)
    assert r.status is S.LIKELY_DUPLICATE_STREAMS
    assert r.product_sets is O.IDENTICAL and r.price_aware_offers is O.IDENTICAL
    assert r.synchronized_prices is True and r.scope_baseline is ScopeBaseline.DISCRIMINATIVE
    assert r.temporal_overlap is TemporalOverlap.COMPLETE and r.shares_collection_events
    assert not r.confirmed_by_authority and not r.alias_authority_sufficient
    assert r.upstream_review_required and r.identity_evidence is IdentityEvidence.UNAVAILABLE


def test_identical_offers_without_a_baseline_are_inconclusive():
    # Previously LIKELY_DUPLICATE_STREAMS: "baseline is not NON_DISCRIMINATIVE" accepted an
    # UNAVAILABLE baseline, so identical offers counted as evidence although nothing showed
    # that identical behaviour is unusual for the source. Only DISCRIMINATIVE qualifies.
    r = compare_location_streams(_jobs(), same_both(), DEF)
    assert r.scope_baseline is ScopeBaseline.UNAVAILABLE
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert r.duplicate_inference_blockers == (DB.BASELINE_UNAVAILABLE,)


def test_identical_offers_typical_in_scope_is_inconclusive():
    extra = [(J1, C, offer()), (J2, C, offer())]   # C identical to A and B as well
    r = compare_location_streams(_jobs(), same_both(extra=extra), DEF)
    assert r.scope_baseline is ScopeBaseline.NON_DISCRIMINATIVE
    assert r.status is S.COMPARISON_INCONCLUSIVE


def test_baseline_ignores_other_scopes():
    extra = [(J1, C, offer() | {SCOPE: CITY2}), (J1, D, offer() | {SCOPE: CITY2})]
    r = compare_location_streams(_jobs(), same_both(extra=extra), DEF)
    assert r.scope_baseline is ScopeBaseline.UNAVAILABLE


def test_distinct_products_likely_distinct():
    cars = _cars([(J1, A, offer("SYNTH-CAR-X")), (J1, B, offer("SYNTH-CAR-Y")),
                  (J2, A, offer("SYNTH-CAR-X")), (J2, B, offer("SYNTH-CAR-Z"))])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.status is S.LIKELY_DISTINCT_STREAMS and r.product_sets is O.DISTINCT
    assert r.synchronized_prices is False and not r.confirmed_by_authority


def test_same_products_different_prices_inconclusive():
    cars = _cars([(J1, A, offer(price="10.00")), (J1, B, offer(price="11.00")),
                  (J2, A, offer()), (J2, B, offer())])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.product_sets is O.IDENTICAL and r.price_aware_offers is O.SAME_PRODUCTS
    assert r.status is S.COMPARISON_INCONCLUSIVE and r.synchronized_prices is False


def test_partial_product_match_inconclusive():
    cars = _cars([(J1, A, offer()), (J1, B, offer()),
                  (J2, A, offer("SYNTH-CAR-X")), (J2, B, offer("SYNTH-CAR-Y"))])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.product_sets is O.PARTIAL and r.status is S.COMPARISON_INCONCLUSIVE


def test_disjoint_events_insufficient_captures():
    cars = _cars([(J1, A, offer()), (J2, B, offer())])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.status is S.INSUFFICIENT_COMPARABLE_CAPTURES
    assert r.temporal_overlap is TemporalOverlap.DISJOINT and r.shares_collection_events is False
    assert r.comparable_captures_exist is False and r.synchronized_prices is None


def test_partial_overlap_compares_only_shared_events_and_is_not_duplicate_evidence():
    # Previously LIKELY_DUPLICATE_STREAMS from a single shared capture with an unpaired one.
    cars = same_both(jobs=(J1,), extra=[(J2, A, offer("SYNTH-CAR-ONLY-A"))])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.temporal_overlap is TemporalOverlap.PARTIAL
    assert r.price_aware_offers is O.IDENTICAL and r.paired_capture_count == 1   # only shared events compared
    assert r.status is S.COMPARISON_INCONCLUSIVE


def test_missing_event_key_rows_not_paired():
    cars = _cars([(pd.NA, A, offer()), (pd.NA, B, offer())])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.status is S.INSUFFICIENT_COMPARABLE_CAPTURES
    assert r.temporal_overlap is TemporalOverlap.UNAVAILABLE


# ---------------------------------------------------------- offer multisets


def test_row_order_ignored():
    o1, o2 = offer("SYNTH-CAR-X"), offer("SYNTH-CAR-Y")
    cars = _cars([(J1, A, o1), (J1, A, o2), (J1, B, o2), (J1, B, o1)])
    assert compare_location_streams(_jobs(), cars, DEF).price_aware_offers is O.IDENTICAL


def test_multiplicity_matters():
    cars = _cars([(J1, A, offer()), (J1, A, offer()), (J1, B, offer())])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.product_sets is O.DISTINCT


def test_technical_fields_do_not_affect_offers():
    cars = same_both()
    cars.loc[cars[LOC] == B, ID_COL] = "SYNTH-OTHER-VALUE"
    cars.loc[cars[LOC] == B, TIME_FIELD.column] = "SYNTH-OTHER-TIME"
    assert compare_location_streams(_jobs(), cars, DEF).price_aware_offers is O.IDENTICAL


def test_label_difference_alone_is_not_an_offer_difference():
    # Rows differ only in the location label: offers identical.
    r = compare_location_streams(_jobs(), same_both(), DEF)
    assert r.product_sets is O.IDENTICAL


def test_price_does_not_identify_products():
    cars = _cars([(J1, A, offer("SYNTH-CAR-X", "10.00")), (J1, B, offer("SYNTH-CAR-Y", "10.00"))])
    assert compare_location_streams(_jobs(), cars, DEF).product_sets is O.DISTINCT


def test_tuple_comparison_is_collision_safe():
    # Concatenation would make these equal ("ab"+"c" == "a"+"bc").
    x = offer() | {PRODUCT[0]: "SYNTH-ab", PRODUCT[1]: "c"}
    y = offer() | {PRODUCT[0]: "SYNTH-a", PRODUCT[1]: "bc"}
    cars = _cars([(J1, A, x), (J1, B, y)])
    assert compare_location_streams(_jobs(), cars, DEF).product_sets is O.DISTINCT


def test_missing_values_are_unassessable_not_equal():
    # Previously two missing values compared equal (IDENTICAL). A missing signature
    # field is no evidence at all - neither matching nor differing.
    x, y = offer() | {PRODUCT[1]: pd.NA}, offer() | {PRODUCT[1]: "SYNTH-NA"}
    both = compare_location_streams(_jobs(), _cars([(J1, A, x), (J1, B, x)]), DEF)
    assert both.product_sets is O.UNAVAILABLE and both.status is S.COMPARISON_UNASSESSABLE
    one = compare_location_streams(_jobs(), _cars([(J1, A, x), (J1, B, y)]), DEF)
    assert one.product_sets is O.UNAVAILABLE and one.invalid_paired_capture_count == 1


def test_price_text_compared_exactly():
    cars = _cars([(J1, A, offer(price="10.0")), (J1, B, offer(price="10.00"))])
    assert compare_location_streams(_jobs(), cars, DEF).price_aware_offers is O.SAME_PRODUCTS


# --------------------------------------------------------- identity evidence


IDDEF = dataclasses.replace(DEF, identity_columns=(ID_COL,))


def _with_ids(cars: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    cars = cars.copy()
    cars.loc[cars[LOC] == A, ID_COL] = a
    cars.loc[cars[LOC] == B, ID_COL] = b
    return cars


def test_different_identity_confirms_distinct_even_if_offers_identical():
    r = compare_location_streams(_jobs(), _with_ids(same_both(), "SYNTH-ST-1", "SYNTH-ST-2"), IDDEF)
    assert r.status is S.CONFIRMED_DISTINCT_LOCATIONS and r.confirmed_by_authority
    assert r.identity_evidence is IdentityEvidence.DIFFERENT and not r.upstream_review_required


def test_same_identity_shared_events_is_duplicated_configuration():
    r = compare_location_streams(_jobs(), _with_ids(same_both(), "SYNTH-ST-1", "SYNTH-ST-1"), IDDEF)
    assert r.status is S.DUPLICATED_COLLECTION_CONFIGURATION and r.upstream_review_required
    assert not r.alias_authority_sufficient


def test_same_identity_disjoint_events_is_alias():
    cars = _with_ids(_cars([(J1, A, offer()), (J2, B, offer("SYNTH-CAR-Y"))]), "SYNTH-ST-1", "SYNTH-ST-1")
    r = compare_location_streams(_jobs(), cars, IDDEF)
    assert r.status is S.CONFIRMED_ALIAS and r.alias_authority_sufficient
    assert validate_confirmed_location_alias(_jobs(), cars, IDDEF) == r


def test_conflicting_identity_is_mapping_defect():
    cars = _with_ids(same_both(), "SYNTH-ST-1", "SYNTH-ST-2")
    cars.loc[(cars[LOC] == A) & (cars[DK] == J2), ID_COL] = "SYNTH-ST-3"
    r = compare_location_streams(_jobs(), cars, IDDEF)
    assert r.status is S.LOCATION_MAPPING_DEFECT and r.mapping_defect_indicated
    assert r.identity_evidence is IdentityEvidence.CONFLICTING


def test_missing_identity_values_fall_back_to_behaviour():
    cars = _with_ids(same_both(extra=[(J1, C, offer("SYNTH-CAR-Y")), (J2, C, offer("SYNTH-CAR-Y"))]), "SYNTH-ST-1", "SYNTH-ST-1")
    cars.loc[cars[LOC] == B, ID_COL] = pd.NA
    r = compare_location_streams(_jobs(), cars, IDDEF)
    assert r.identity_evidence is IdentityEvidence.UNAVAILABLE
    assert r.status is S.LIKELY_DUPLICATE_STREAMS


# ----------------------------------------------------------- capture time


TDEF = dataclasses.replace(DEF, pairing=CapturePairing.CAPTURE_TIME, capture_time_field=TIME_FIELD.ref,
                           pairing_tolerance=dt.timedelta(minutes=5))


def _timed(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows: (job key, location, capture text)."""
    return _cars([(j, loc, offer() | {TIME_FIELD.column: t}) for j, loc, t in rows])


def test_capture_time_pairs_within_tolerance_across_events():
    cars = _timed([(J1, A, "2025-01-15 05:00:00 MST"), (J2, B, "2025-01-15 05:03:00 MST")])
    r = compare_location_streams(_jobs(), cars, TDEF)
    assert r.temporal_overlap is TemporalOverlap.COMPLETE and r.paired_capture_count == 1
    assert r.shares_collection_events is False
    # One pair is below the evidence minimum (and no baseline exists): not a likely duplicate.
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert r.duplicate_inference_blockers == (DB.INSUFFICIENT_PAIRED_CAPTURES, DB.BASELINE_UNAVAILABLE)


@pytest.mark.parametrize("second, paired", [("05:05:00", True), ("05:05:01", False)])
def test_capture_time_tolerance_boundary(second, paired):
    cars = _timed([(J1, A, "2025-01-15 05:00:00 MST"), (J2, B, f"2025-01-15 {second} MST")])
    assert compare_location_streams(_jobs(), cars, TDEF).comparable_captures_exist is paired


def test_zero_tolerance_requires_exact_instant():
    zero = dataclasses.replace(TDEF, pairing_tolerance=dt.timedelta(0))
    cars = _timed([(J1, A, "2025-01-15 05:00:00 MST"), (J2, B, "2025-01-15 05:00:00 MST")])
    assert compare_location_streams(_jobs(), cars, zero).comparable_captures_exist


def test_ambiguous_capture_time_fails_closed():
    cars = _timed([(J1, A, "2025-01-15 05:00:00 MST"), (J2, B, "2025-01-15 05:01:00 MST"),
                   (J3, B, "2025-01-15 04:59:00 MST")])
    r = compare_location_streams(_jobs(), cars, TDEF)
    assert r.ambiguous_pairing and not r.comparable_captures_exist
    assert r.status is S.INSUFFICIENT_COMPARABLE_CAPTURES


def test_unresolved_capture_time_fails_closed():
    cars = _timed([(J1, A, "2025-01-15 05:00:00 XYZ"), (J2, B, "2025-01-15 05:00:00 MST")])
    r = compare_location_streams(_jobs(), cars, TDEF)
    assert r.temporal_overlap is TemporalOverlap.UNAVAILABLE and not r.comparable_captures_exist


def test_inconsistent_times_within_event_fail_closed():
    cars = _timed([(J1, A, "2025-01-15 05:00:00 MST"), (J1, A, "2025-01-15 06:00:00 MST"),
                   (J2, B, "2025-01-15 05:00:00 MST")])
    assert not compare_location_streams(_jobs(), cars, TDEF).comparable_captures_exist


# ---------------------------------------------------------------- alias API


def test_validate_alias_raises_without_authority():
    with pytest.raises(LocationAliasNotConfirmedError) as info:
        validate_confirmed_location_alias(_jobs(), same_both(extra=[(J1, C, offer("SYNTH-CAR-Y")), (J2, C, offer("SYNTH-CAR-Y"))]), DEF)
    assert info.value.report.status is S.LIKELY_DUPLICATE_STREAMS     # behaviour is never alias authority
    assert "SYNTH" not in str(info.value)


def test_canonical_keys_are_opt_in_and_preserve_raw_labels():
    cars = _cars([(J1, A, offer()), (J1, D, offer())])
    before = cars.copy(deep=True)
    assert canonical_location_keys(cars, COV).tolist() == [(A,), (D,)]      # no aliases: identity
    aliased = dataclasses.replace(COV, aliases={(A,): ((D,),)})            # authority-declared only
    keys = canonical_location_keys(cars, aliased)
    assert keys.tolist() == [(A,), (A,)] and keys.index.equals(cars.index)
    pd.testing.assert_frame_equal(cars, before)
    assert cars[LOC].tolist() == [A, D]                                    # source labels preserved


def test_source_coverage_is_computed_from_raw_labels():
    cars = _cars([(J1, A, offer()), (J1, D, offer())])
    before = cars.copy(deep=True)
    raw = assess_expected_location_coverage(cars, COV)
    canonical_location_keys(cars, dataclasses.replace(COV, aliases={(B,): ((D,),)}))
    assert assess_expected_location_coverage(cars, COV) == raw
    pd.testing.assert_frame_equal(cars, before)


# --------------------------------------------------------- report safety


def test_report_holds_only_enums_bools_counts_and_none():
    r = compare_location_streams(_jobs(), same_both(), DEF)
    assert isinstance(r, LocationStreamComparisonReport)
    for field in dataclasses.fields(r):
        value = getattr(r, field.name)
        if field.name == "invalid_offer_sample":            # bounded, confidential, excluded from repr
            assert not field.repr and isinstance(value, tuple)
        elif field.name == "invalid_offer_fields":          # contract column names only
            assert all(v in (*PRODUCT, *PRICE) for v in value)
        elif field.name == "invalid_offer_reasons":
            assert all(isinstance(v, OfferDefect) for v in value)
        elif field.name == "baseline_evidence":
            assert all(type(getattr(value, f.name)) is int for f in dataclasses.fields(value))
        elif isinstance(value, tuple):
            assert all(isinstance(v, DB) for v in value)
        else:
            assert value is None or isinstance(value, (bool, int)) or hasattr(value, "value")
    assert "SYNTH" not in repr(r)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.status = S.CONFIRMED_ALIAS  # type: ignore[misc]


def test_inputs_not_mutated():
    jobs, cars = _jobs(), same_both(extra=[(J1, C, offer("SYNTH-CAR-Y"))])
    jb, cb = jobs.copy(deep=True), cars.copy(deep=True)
    compare_location_streams(jobs, cars, DEF)
    pd.testing.assert_frame_equal(jobs, jb)
    pd.testing.assert_frame_equal(cars, cb)


def test_deterministic():
    cars = same_both(extra=[(J1, C, offer("SYNTH-CAR-Y"))])
    assert compare_location_streams(_jobs(), cars, DEF) == compare_location_streams(_jobs(), cars.iloc[::-1], DEF)


def test_default_definition_is_project_comparison():
    r = compare_location_streams(_jobs(), _cars([(J1, C, offer())]))
    assert r.status is S.BOTH_STREAMS_ABSENT


def test_package_exports():
    for name in ("compare_location_streams", "validate_confirmed_location_alias", "canonical_location_keys",
                 "LocationStreamComparisonStatus", "LocationStreamComparisonReport",
                 "LOCATION_STREAM_COMPARISON", "LocationStreamComparisonDefinition", "CapturePairing"):
        assert name in ql2_sixt_canada_analysis.__all__
    assert isinstance(ql2_sixt_canada_analysis.LOCATION_STREAM_COMPARISON, LocationStreamComparisonDefinition)


# ------------------------------------------- duplicate-inference evidence rules

# The baseline branch C behaves differently from A/B in every capture it shares,
# which is what makes the scope DISCRIMINATIVE.
def _baseline_rows(jobs: tuple[str, ...]) -> list:
    return [(j, C, offer("SYNTH-CAR-Y")) for j in jobs]


def _evidence(r) -> tuple:  # type: ignore[no-untyped-def]
    return (r.first_capture_count, r.second_capture_count, r.paired_capture_count,
            r.first_unpaired_capture_count, r.second_unpaired_capture_count,
            r.matching_paired_capture_count, r.differing_paired_capture_count, r.minimum_paired_captures)


def test_single_shared_capture_with_unpaired_capture_and_no_baseline_is_not_duplicate():
    # Original defect: one identical shared capture + one unpaired capture + no baseline.
    cars = _cars([(J1, A, offer()), (J1, B, offer()), (J2, A, offer())])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.status is not S.LIKELY_DUPLICATE_STREAMS and r.status is S.COMPARISON_INCONCLUSIVE
    assert r.temporal_overlap is TemporalOverlap.PARTIAL and r.scope_baseline is ScopeBaseline.UNAVAILABLE
    assert _evidence(r) == (2, 1, 1, 1, 0, 1, 0, 2)
    assert r.duplicate_inference_blockers == (DB.INSUFFICIENT_PAIRED_CAPTURES, DB.INCOMPLETE_TEMPORAL_OVERLAP,
                                              DB.BASELINE_UNAVAILABLE)
    assert not r.duplicate_evidence_sufficient and r.price_aware_offers is O.IDENTICAL


def test_one_pair_is_below_the_minimum_even_with_full_overlap_and_baseline():
    # The one-capture comparator no longer qualifies the baseline either (same standard).
    cars = _cars([(J1, A, offer()), (J1, B, offer()), *_baseline_rows((J1,))])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.temporal_overlap is TemporalOverlap.COMPLETE and r.scope_baseline is ScopeBaseline.INSUFFICIENT
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert r.duplicate_inference_blockers == (DB.INSUFFICIENT_PAIRED_CAPTURES, DB.BASELINE_EVIDENCE_INSUFFICIENT)
    assert _evidence(r) == (1, 1, 1, 0, 0, 1, 0, 2)


def test_enough_matching_pairs_with_partial_overlap_are_not_duplicate():
    cars = same_both(jobs=(J1, J2), extra=[(J3, B, offer()), *_baseline_rows((J1, J2))])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.temporal_overlap is TemporalOverlap.PARTIAL and r.scope_baseline is ScopeBaseline.DISCRIMINATIVE
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert r.duplicate_inference_blockers == (DB.INCOMPLETE_TEMPORAL_OVERLAP,)
    assert (r.paired_capture_count, r.first_unpaired_capture_count, r.second_unpaired_capture_count) == (2, 0, 1)


def test_enough_matching_pairs_with_unavailable_baseline_are_not_duplicate():
    r = compare_location_streams(_jobs(), same_both(jobs=(J1, J2)), DEF)
    assert r.temporal_overlap is TemporalOverlap.COMPLETE and r.paired_capture_count == 2
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert r.duplicate_inference_blockers == (DB.BASELINE_UNAVAILABLE,)


def test_enough_matching_pairs_with_non_discriminative_baseline_are_not_duplicate():
    cars = same_both(jobs=(J1, J2), extra=[(J1, C, offer()), (J2, C, offer())])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.scope_baseline is ScopeBaseline.NON_DISCRIMINATIVE
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert r.duplicate_inference_blockers == (DB.BASELINE_NON_DISCRIMINATIVE,)


def test_complete_affirmative_evidence_allows_likely_duplicate_but_never_authority():
    cars = same_both(jobs=(J1, J2), extra=_baseline_rows((J1, J2)))
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.status is S.LIKELY_DUPLICATE_STREAMS and r.duplicate_evidence_sufficient
    assert r.duplicate_inference_blockers == () and _evidence(r) == (2, 2, 2, 0, 0, 2, 0, 2)
    assert not r.confirmed_by_authority and not r.alias_authority_sufficient and r.upstream_review_required
    with pytest.raises(LocationAliasNotConfirmedError):
        validate_confirmed_location_alias(_jobs(), cars, DEF)
    from ql2_sixt_canada_analysis.readiness import assess_location_policy
    from ql2_sixt_canada_analysis.schemas import VANCOUVER_LOCATION_POLICY, LocationPolicyState
    policy = assess_location_policy(VANCOUVER_LOCATION_POLICY, r)
    assert policy.state is LocationPolicyState.UNRESOLVED and not policy.locations_are_aliases
    assert not policy.location_policy_resolved


def test_one_contradictory_pair_blocks_duplicate_and_is_retained():
    cars = _cars([(J1, A, offer()), (J1, B, offer()),
                  (J2, A, offer()), (J2, B, offer(price="99.00")),           # materially differs
                  (J3, A, offer()), (J3, B, offer()), *_baseline_rows((J1, J2, J3))])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert r.duplicate_inference_blockers == (DB.DIFFERING_PAIRED_CAPTURES,)
    assert (r.matching_paired_capture_count, r.differing_paired_capture_count) == (2, 1)
    assert r.price_aware_offers is O.SAME_PRODUCTS


def test_duplicate_rows_and_many_offers_in_one_capture_count_once():
    o1, o2 = offer("SYNTH-CAR-X"), offer("SYNTH-CAR-Z")
    cars = _cars([(J1, A, o1), (J1, A, o1), (J1, A, o2), (J1, B, o1), (J1, B, o1), (J1, B, o2),
                  *_baseline_rows((J1,))])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert (r.first_capture_count, r.second_capture_count, r.paired_capture_count) == (1, 1, 1)
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert DB.INSUFFICIENT_PAIRED_CAPTURES in r.duplicate_inference_blockers


def test_rows_without_a_capture_key_are_counted_not_dropped():
    # Without the unidentifiable row the two streams would look fully overlapping.
    cars = same_both(jobs=(J1, J2), extra=[(pd.NA, A, offer("SYNTH-CAR-Q")), *_baseline_rows((J1, J2))])
    r = compare_location_streams(_jobs(), cars, DEF)
    assert r.temporal_overlap is TemporalOverlap.COMPLETE
    assert (r.first_unassessable_row_count, r.second_unassessable_row_count) == (1, 0)
    assert r.status is S.COMPARISON_INCONCLUSIVE
    assert r.duplicate_inference_blockers == (DB.UNASSESSABLE_OBSERVATIONS,)


def test_unresolvable_capture_time_cannot_create_duplicate_evidence():
    two_pairs = dataclasses.replace(TDEF, pairing_tolerance=dt.timedelta(0))
    cars = _timed([(J1, A, "2025-01-15 05:00:00 MST"), (J1, B, "2025-01-15 05:00:00 MST"),
                   (J2, A, "2025-01-15 06:00:00 MST"), (J2, B, "2025-01-15 06:00:00 XYZ")])
    r = compare_location_streams(_jobs(), cars, two_pairs)
    assert r.status is not S.LIKELY_DUPLICATE_STREAMS
    assert DB.UNASSESSABLE_OBSERVATIONS in r.duplicate_inference_blockers
    assert DB.TEMPORAL_OVERLAP_UNAVAILABLE in r.duplicate_inference_blockers
    assert r.second_unassessable_row_count == 1


@pytest.mark.parametrize("cars_fn", [
    lambda: _cars([(J1, A, offer()), (J1, B, offer()), (J2, A, offer())]),
    lambda: same_both(jobs=(J1, J2), extra=_baseline_rows((J1, J2))),
    lambda: same_both(jobs=(J1, J2), extra=[(J3, B, offer()), *_baseline_rows((J1, J2))]),
])
def test_classification_is_symmetric_under_stream_swap(cars_fn):
    forward = compare_location_streams(_jobs(), cars_fn(), DEF)
    swapped = compare_location_streams(_jobs(), cars_fn(), dataclasses.replace(DEF, first=(B,), second=(A,)))
    assert forward.status is swapped.status
    assert forward.duplicate_inference_blockers == swapped.duplicate_inference_blockers
    assert forward.paired_capture_count == swapped.paired_capture_count
    assert (forward.first_capture_count, forward.first_unpaired_capture_count) == (
        swapped.second_capture_count, swapped.second_unpaired_capture_count)
    assert (forward.temporal_overlap, forward.scope_baseline) == (swapped.temporal_overlap, swapped.scope_baseline)


@pytest.mark.parametrize("pairs, expected", [(2, S.COMPARISON_INCONCLUSIVE), (3, S.LIKELY_DUPLICATE_STREAMS)])
def test_minimum_paired_capture_boundary(pairs, expected):
    three = dataclasses.replace(DEF, minimum_paired_captures=3)
    jobs = (J1, J2, J3)[:pairs]
    r = compare_location_streams(_jobs(), same_both(jobs=jobs, extra=_baseline_rows(jobs)), three)
    assert r.status is expected and r.minimum_paired_captures == 3
    assert (DB.INSUFFICIENT_PAIRED_CAPTURES in r.duplicate_inference_blockers) is (pairs < 3)


@pytest.mark.parametrize("value", [0, -1, 1, 2.0, "2", True, None])
def test_minimum_paired_captures_must_be_an_integer_of_at_least_two(value):
    with pytest.raises(LocationCoverageConfigurationError):
        dataclasses.replace(DEF, minimum_paired_captures=value)


def test_project_minimum_is_central_and_conservative():
    from ql2_sixt_canada_analysis.schemas import MINIMUM_DUPLICATE_PAIRED_CAPTURES
    assert LOCATION_STREAM_COMPARISON.minimum_paired_captures == MINIMUM_DUPLICATE_PAIRED_CAPTURES >= 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        LOCATION_STREAM_COMPARISON.minimum_paired_captures = 1  # type: ignore[misc]


def test_blocker_values_are_stable_and_name_no_source_columns():
    assert [b.value for b in DB] == [
        "no_paired_captures", "insufficient_paired_captures", "incomplete_temporal_overlap",
        "temporal_overlap_unavailable", "ambiguous_pairing", "unassessable_observations",
        "invalid_offer_evidence", "baseline_unavailable", "baseline_evidence_insufficient",
        "baseline_non_discriminative", "differing_paired_captures"]
    columns = set(contract_columns(CARS)) | set(contract_columns(JOBS))
    assert not any(c in b.value for b in DB for c in columns)
