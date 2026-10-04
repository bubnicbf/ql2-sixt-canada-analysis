"""Compare two expected location streams: identity first, behaviour second.

The comparison is defined by a
:class:`~ql2_sixt_canada_analysis.schemas.LocationStreamComparisonDefinition`
(the project pair: :data:`~ql2_sixt_canada_analysis.schemas.LOCATION_STREAM_COMPARISON`).

Evidence rules (fixed in code, not tuned to any data)
-----------------------------------------------------
1. **Presence** - if either stream has no rows: ``ONE_STREAM_ABSENT`` or
   ``BOTH_STREAMS_ABSENT``; nothing is fabricated or compared.
2. **Authoritative identity** (``identity_columns``; empty when the source has
   none). Each stream must carry exactly one complete identity tuple:
   * different identities -> ``CONFIRMED_DISTINCT_LOCATIONS`` (whatever the
     offers look like);
   * the same identity -> ``DUPLICATED_COLLECTION_CONFIGURATION`` when both
     labels are captured in the same collection events (one place collected
     twice), else ``CONFIRMED_ALIAS`` (one place, a label change);
   * a stream with several identities -> ``LOCATION_MAPPING_DEFECT``.
3. **Behaviour** (only when identity is unavailable) on *paired* captures:
   * no unambiguous pairs -> ``INSUFFICIENT_COMPARABLE_CAPTURES``;
   * offers (products *and* prices, as multisets) identical in every paired
     capture, and identical behaviour is **not** typical of other location
     pairs in the same scope -> ``LIKELY_DUPLICATE_STREAMS``;
   * no paired capture with equal product multisets -> ``LIKELY_DISTINCT_STREAMS``;
   * anything else -> ``COMPARISON_INCONCLUSIVE``.
   Behaviour never confirms an alias or distinctness.

Pairing: ``SHARED_COLLECTION_EVENT`` pairs exactly by collection event;
``CAPTURE_TIME`` pairs reconciled instants within an explicit tolerance and
fails closed on ambiguity or unresolved times. Offer multisets are compared
as tuples of the configured columns (``collections.Counter``: hashing plus
exact equality - no lossy fingerprint), preserving multiplicity and ignoring
row order. Prices are compared as exact source values and never identify
products.

The scope **baseline** applies the same pairing and comparison to every
other location pair in the targets' scope: ``NON_DISCRIMINATIVE`` if any is
identical in all its paired captures (identical offers are then normal for
the source and prove nothing), ``DISCRIMINATIVE`` if none is, ``UNAVAILABLE``
if there are no such pairs.

The report holds enums and booleans only. Nothing merges, rewrites, drops,
canonicalises or writes rows; :func:`canonical_location_keys` is opt-in and
applies only aliases declared in the coverage contract.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.relationships import RelationshipPreconditionError, _check_relationship_inputs
from ql2_sixt_canada_analysis.schemas import (
    LOCATION_STREAM_COMPARISON,
    CapturePairing,
    LocationCoverageConfigurationError,
    LocationCoverageDefinition,
    LocationStreamComparisonDefinition,
)
from ql2_sixt_canada_analysis.temporal import _parent_positions, parse_temporal_field

__all__ = [
    "ComparisonPreconditionError",
    "IdentityEvidence",
    "LocationAliasNotConfirmedError",
    "LocationStreamComparisonReport",
    "LocationStreamComparisonStatus",
    "OfferSetResult",
    "ScopeBaseline",
    "TemporalOverlap",
    "canonical_location_keys",
    "compare_location_streams",
    "validate_confirmed_location_alias",
]


class LocationStreamComparisonStatus(StrEnum):
    CONFIRMED_DISTINCT_LOCATIONS = "confirmed_distinct_locations"
    CONFIRMED_ALIAS = "confirmed_alias"
    DUPLICATED_COLLECTION_CONFIGURATION = "duplicated_collection_configuration"
    LOCATION_MAPPING_DEFECT = "location_mapping_defect"
    LIKELY_DUPLICATE_STREAMS = "likely_duplicate_streams"
    LIKELY_DISTINCT_STREAMS = "likely_distinct_streams"
    ONE_STREAM_ABSENT = "one_stream_absent"
    BOTH_STREAMS_ABSENT = "both_streams_absent"
    INSUFFICIENT_COMPARABLE_CAPTURES = "insufficient_comparable_captures"
    COMPARISON_INCONCLUSIVE = "comparison_inconclusive"


class IdentityEvidence(StrEnum):
    UNAVAILABLE = "unavailable"
    SAME = "same"
    DIFFERENT = "different"
    CONFLICTING = "conflicting"


class TemporalOverlap(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    DISJOINT = "disjoint"
    UNAVAILABLE = "unavailable"


class OfferSetResult(StrEnum):
    IDENTICAL = "identical"                      # products and prices equal in every paired capture
    SAME_PRODUCTS = "same_products"              # products equal everywhere, prices differ somewhere
    PARTIAL = "partial"                          # mixed / partially overlapping
    DISTINCT = "distinct"                        # no paired capture with equal products
    UNAVAILABLE = "unavailable"


class ScopeBaseline(StrEnum):
    DISCRIMINATIVE = "discriminative"
    NON_DISCRIMINATIVE = "non_discriminative"
    UNAVAILABLE = "unavailable"


_CONFIRMED = frozenset({
    LocationStreamComparisonStatus.CONFIRMED_DISTINCT_LOCATIONS, LocationStreamComparisonStatus.CONFIRMED_ALIAS,
    LocationStreamComparisonStatus.DUPLICATED_COLLECTION_CONFIGURATION,
    LocationStreamComparisonStatus.LOCATION_MAPPING_DEFECT,
})


@dataclass(frozen=True, slots=True)
class LocationStreamComparisonReport:
    """Categorical comparison of two location streams (enums and booleans only)."""

    status: LocationStreamComparisonStatus
    targets_configured: bool
    first_present: bool
    second_present: bool
    first_details_linked: bool | None
    second_details_linked: bool | None
    identity_evidence: IdentityEvidence
    shares_collection_events: bool | None
    temporal_overlap: TemporalOverlap
    ambiguous_pairing: bool
    comparable_captures_exist: bool
    product_sets: OfferSetResult
    price_aware_offers: OfferSetResult
    synchronized_prices: bool | None
    scope_baseline: ScopeBaseline

    @property
    def confirmed_by_authority(self) -> bool:
        """The status rests on authoritative identity metadata."""
        return self.status in _CONFIRMED

    @property
    def alias_authority_sufficient(self) -> bool:
        return self.status is LocationStreamComparisonStatus.CONFIRMED_ALIAS

    @property
    def mapping_defect_indicated(self) -> bool:
        return self.status is LocationStreamComparisonStatus.LOCATION_MAPPING_DEFECT

    @property
    def upstream_review_required(self) -> bool:
        """Collection configuration or identity needs confirmation by the source owner."""
        return self.status in {
            LocationStreamComparisonStatus.DUPLICATED_COLLECTION_CONFIGURATION,
            LocationStreamComparisonStatus.LIKELY_DUPLICATE_STREAMS,
            LocationStreamComparisonStatus.LOCATION_MAPPING_DEFECT,
            LocationStreamComparisonStatus.ONE_STREAM_ABSENT,
            LocationStreamComparisonStatus.BOTH_STREAMS_ABSENT,
        }


class ComparisonPreconditionError(RelationshipPreconditionError):
    """Relationship preconditions for the comparison failed."""

    control = "Location-stream comparison"


class LocationAliasNotConfirmedError(Exception):
    """The comparison does not confirm an alias. ``report`` holds the categories."""

    def __init__(self, report: LocationStreamComparisonReport) -> None:
        super().__init__(f"Location alias not confirmed: {report.status.value}.")
        self.report = report


# ------------------------------------------------------------------- public API


def compare_location_streams(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    definition: LocationStreamComparisonDefinition = LOCATION_STREAM_COMPARISON,
) -> LocationStreamComparisonReport:
    """Compare the two configured streams; returns a categorical report.

    Raises:
        TypeError: Invalid argument types.
        LocationCoverageConfigurationError: Configured columns are absent.
        ComparisonPreconditionError: Parent key, identifier dtype or
            blank-row preconditions fail.
    """
    if not isinstance(definition, LocationStreamComparisonDefinition):
        raise TypeError("definition must be a LocationStreamComparisonDefinition")
    rel, cov = definition.relationship, definition.coverage
    _check_relationship_inputs(jobs, cars, rel, ComparisonPreconditionError)
    needed = {*cov.location_columns, *cov.stream_scope_columns, *definition.product_columns,
              *definition.price_columns, *definition.identity_columns}
    absent = sorted(c for c in needed if c not in cars.columns)
    if absent:
        raise LocationCoverageConfigurationError(f"The comparison frame lacks {len(absent)} column(s).", tuple(absent))

    keys = _location_keys(cars, cov.location_columns)
    first_mask, second_mask = _key_mask(keys, definition.first), _key_mask(keys, definition.second)
    first_present, second_present = bool(first_mask.any()), bool(second_mask.any())
    positions = _parent_positions(jobs, cars, rel)
    linked = lambda m: bool((positions[m] >= 0).all()) if m.any() else None  # noqa: E731

    base = dict(targets_configured=True, first_present=first_present, second_present=second_present,
                first_details_linked=linked(first_mask), second_details_linked=linked(second_mask))
    if not (first_present and second_present):
        status = (LocationStreamComparisonStatus.BOTH_STREAMS_ABSENT if not (first_present or second_present)
                  else LocationStreamComparisonStatus.ONE_STREAM_ABSENT)
        return LocationStreamComparisonReport(
            status=status, **base, identity_evidence=IdentityEvidence.UNAVAILABLE,
            shares_collection_events=None, temporal_overlap=TemporalOverlap.UNAVAILABLE, ambiguous_pairing=False,
            comparable_captures_exist=False, product_sets=OfferSetResult.UNAVAILABLE,
            price_aware_offers=OfferSetResult.UNAVAILABLE, synchronized_prices=None,
            scope_baseline=ScopeBaseline.UNAVAILABLE)

    identity = _identity(cars, first_mask, second_mask, definition.identity_columns)
    events_first = _events(cars, first_mask, rel)
    events_second = _events(cars, second_mask, rel)
    shares_events = bool(set(events_first) & set(events_second))
    overlap, pairs, ambiguous = _pair(cars, first_mask, second_mask, definition)
    offers = _Offers(cars, definition)
    products, priced = offers.compare(first_mask, second_mask, pairs)
    baseline = _baseline(cars, keys, first_mask | second_mask, definition, offers)

    S = LocationStreamComparisonStatus
    if identity is IdentityEvidence.CONFLICTING:
        status = S.LOCATION_MAPPING_DEFECT
    elif identity is IdentityEvidence.DIFFERENT:
        status = S.CONFIRMED_DISTINCT_LOCATIONS
    elif identity is IdentityEvidence.SAME:
        status = S.DUPLICATED_COLLECTION_CONFIGURATION if shares_events else S.CONFIRMED_ALIAS
    elif not pairs:
        status = S.INSUFFICIENT_COMPARABLE_CAPTURES
    elif priced is OfferSetResult.IDENTICAL and baseline is not ScopeBaseline.NON_DISCRIMINATIVE:
        status = S.LIKELY_DUPLICATE_STREAMS
    elif products is OfferSetResult.DISTINCT:
        status = S.LIKELY_DISTINCT_STREAMS
    else:
        status = S.COMPARISON_INCONCLUSIVE
    return LocationStreamComparisonReport(
        status=status, **base, identity_evidence=identity, shares_collection_events=shares_events,
        temporal_overlap=overlap, ambiguous_pairing=ambiguous, comparable_captures_exist=bool(pairs),
        product_sets=products, price_aware_offers=priced,
        synchronized_prices=(priced is OfferSetResult.IDENTICAL) if pairs else None,
        scope_baseline=baseline)


def validate_confirmed_location_alias(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    definition: LocationStreamComparisonDefinition = LOCATION_STREAM_COMPARISON,
) -> LocationStreamComparisonReport:
    """Return the report only if an alias is confirmed by authority; else raise."""
    report = compare_location_streams(jobs, cars, definition)
    if not report.alias_authority_sufficient:
        raise LocationAliasNotConfirmedError(report)
    return report


def canonical_location_keys(frame: pd.DataFrame, coverage: LocationCoverageDefinition) -> pd.Series:
    """Opt-in: each row's location key with authoritative aliases mapped to their expected key.

    Uses only ``coverage.aliases`` (none are confirmed for the project
    contract). Returns a new Series of tuples aligned to ``frame``; source
    labels and rows are untouched, and source-stream identity stays available
    from the original columns.
    """
    if not isinstance(frame, pd.DataFrame) or not isinstance(coverage, LocationCoverageDefinition):
        raise TypeError("frame must be a DataFrame and coverage a LocationCoverageDefinition")
    reverse = {alias: key for key, aliases in coverage.aliases.items() for alias in aliases}
    return _location_keys(frame, coverage.location_columns).map(lambda k: reverse.get(k, k))


# ---------------------------------------------------------------------- helpers


def _key_mask(keys: pd.Series, target: tuple) -> np.ndarray:
    return np.fromiter((k == target for k in keys), dtype=bool, count=len(keys))


def _location_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = frame.loc[:, list(columns)].astype(object).where(frame.loc[:, list(columns)].notna(), None)
    return pd.Series(list(map(tuple, values.itertuples(index=False))), index=frame.index, dtype=object)


def _rows(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple]:
    values = frame.loc[:, list(columns)].astype(object)
    return list(map(tuple, values.where(values.notna(), None).itertuples(index=False)))


def _events(cars: pd.DataFrame, mask: np.ndarray, rel) -> pd.Series:  # type: ignore[no-untyped-def]
    """Collection-event key (complete detail relationship tuple) per stream row; NaN if incomplete."""
    keys = cars.loc[mask, list(rel.detail_key_columns)]
    complete = keys.notna().all(axis=1)
    return pd.Series(_rows(keys.loc[complete], tuple(keys.columns)), dtype=object)


def _identity(cars: pd.DataFrame, a: np.ndarray, b: np.ndarray, columns: tuple[str, ...]) -> IdentityEvidence:
    if not columns:
        return IdentityEvidence.UNAVAILABLE
    ids = []
    for mask in (a, b):
        values = set(_rows(cars.loc[mask], columns))
        values = {v for v in values if all(x is not None for x in v)}
        ids.append(values)
    if any(len(v) > 1 for v in ids):
        return IdentityEvidence.CONFLICTING
    if any(len(v) == 0 for v in ids):
        return IdentityEvidence.UNAVAILABLE
    return IdentityEvidence.SAME if ids[0] == ids[1] else IdentityEvidence.DIFFERENT


def _pair(cars: pd.DataFrame, a: np.ndarray, b: np.ndarray, d: LocationStreamComparisonDefinition
          ) -> tuple[TemporalOverlap, list[tuple[tuple, tuple]], bool]:
    """(overlap, list of (event_a, event_b) pairs, ambiguous?)."""
    rel = d.relationship
    ev_a, ev_b = set(_events(cars, a, rel)), set(_events(cars, b, rel))
    if not ev_a or not ev_b:
        return TemporalOverlap.UNAVAILABLE, [], False
    if d.pairing is CapturePairing.SHARED_COLLECTION_EVENT:
        common = ev_a & ev_b
        overlap = (TemporalOverlap.COMPLETE if ev_a == ev_b else
                   TemporalOverlap.PARTIAL if common else TemporalOverlap.DISJOINT)
        return overlap, [(e, e) for e in sorted(common, key=repr)], False
    # CAPTURE_TIME: one reconciled instant per event, pairs within tolerance.
    field = d.temporal.field(d.capture_time_field)
    times = {}
    for name, mask in (("a", a), ("b", b)):
        part = cars.loc[mask]
        parsed = parse_temporal_field(part[field.column], field, d.temporal.canonical_timezone)
        frame = pd.DataFrame({"event": _rows(part, rel.detail_key_columns), "t": parsed.instants.to_numpy()})
        per_event = frame.groupby("event", sort=False)["t"].agg(lambda s: s.iloc[0] if s.nunique() == 1 and s.notna().all() else pd.NaT)
        if per_event.isna().any() or per_event.duplicated().any():
            return TemporalOverlap.UNAVAILABLE, [], bool(per_event.duplicated().any())
        times[name] = per_event
    tol = pd.Timedelta(d.pairing_tolerance)
    pairs, ambiguous = [], False
    matched_b: set = set()
    for ea, ta in times["a"].items():
        candidates = [eb for eb, tb in times["b"].items() if abs(tb - ta) <= tol]
        if len(candidates) > 1:
            ambiguous = True
        elif candidates:
            if candidates[0] in matched_b:
                ambiguous = True
            matched_b.add(candidates[0])
            pairs.append((ea, candidates[0]))
    if ambiguous:
        return TemporalOverlap.UNAVAILABLE, [], True
    overlap = (TemporalOverlap.COMPLETE if len(pairs) == len(times["a"]) == len(times["b"]) else
               TemporalOverlap.PARTIAL if pairs else TemporalOverlap.DISJOINT)
    return overlap, pairs, False


class _Offers:
    """Per-row offer tuples grouped by collection event (built once per comparison)."""

    def __init__(self, cars: pd.DataFrame, d: LocationStreamComparisonDefinition) -> None:
        self.products = _rows(cars, d.product_columns)
        self.full = _rows(cars, (*d.product_columns, *d.price_columns))
        self.by_event: dict[tuple, list[int]] = {}
        for position, event in enumerate(_rows(cars, d.relationship.detail_key_columns)):
            self.by_event.setdefault(event, []).append(position)

    def _multiset(self, rows: list[tuple], mask: np.ndarray, event: tuple) -> Counter:
        # Exact tuple equality; multiplicity kept; row order irrelevant.
        return Counter(rows[p] for p in self.by_event.get(event, ()) if mask[p])

    def compare(self, a: np.ndarray, b: np.ndarray, pairs: list) -> tuple[OfferSetResult, OfferSetResult]:
        if not pairs:
            return OfferSetResult.UNAVAILABLE, OfferSetResult.UNAVAILABLE
        product_equal = [self._multiset(self.products, a, ea) == self._multiset(self.products, b, eb)
                         for ea, eb in pairs]
        full_equal = [self._multiset(self.full, a, ea) == self._multiset(self.full, b, eb) for ea, eb in pairs]
        if all(product_equal):
            products = OfferSetResult.IDENTICAL
        elif not any(product_equal):
            products = OfferSetResult.DISTINCT
        else:
            products = OfferSetResult.PARTIAL
        if all(full_equal):
            priced = OfferSetResult.IDENTICAL
        elif all(product_equal):
            priced = OfferSetResult.SAME_PRODUCTS
        elif not any(full_equal):
            priced = OfferSetResult.DISTINCT
        else:
            priced = OfferSetResult.PARTIAL
        return products, priced


def _baseline(cars: pd.DataFrame, keys: pd.Series, target_mask: np.ndarray,
              d: LocationStreamComparisonDefinition, offers: _Offers) -> ScopeBaseline:
    """Is fully identical behaviour typical of other location pairs in the targets' scope?"""
    scope = d.coverage.stream_scope_columns
    if not scope:
        return ScopeBaseline.UNAVAILABLE
    scope_keys = pd.Series(_rows(cars, scope), index=cars.index, dtype=object)
    target_scopes = set(scope_keys[target_mask])
    in_scope = np.fromiter((k in target_scopes for k in scope_keys), dtype=bool, count=len(scope_keys))
    others = sorted({k for k in keys[in_scope & ~target_mask] if all(v is not None for v in k)}, key=repr)
    candidates = sorted({*others, d.first, d.second}, key=repr)
    results = []
    for x, y in combinations(candidates, 2):
        if {x, y} == {d.first, d.second}:
            continue
        mx = _key_mask(keys, x) & in_scope
        my = _key_mask(keys, y) & in_scope
        _, pairs, _ = _pair(cars, mx, my, d)
        if pairs:
            _, priced = offers.compare(mx, my, pairs)
            results.append(priced is OfferSetResult.IDENTICAL)
    if not results:
        return ScopeBaseline.UNAVAILABLE
    return ScopeBaseline.NON_DISCRIMINATIVE if any(results) else ScopeBaseline.DISCRIMINATIVE
