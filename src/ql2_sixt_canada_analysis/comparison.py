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
   * ``LIKELY_DUPLICATE_STREAMS`` only with affirmative evidence for every
     prerequisite (each gap is a :class:`DuplicateInferenceBlocker`):
     at least ``minimum_paired_captures`` independent *eligible* paired
     capture events (events, not offer rows; see "Valid evidence" below); no
     invalid offer in either stream; ``COMPLETE`` temporal overlap (no unpaired
     capture on either side); the ``DISCRIMINATIVE`` scope baseline (an
     allowlist - ``UNAVAILABLE`` or ``NON_DISCRIMINATIVE`` never suffice);
     identical price-aware offers in *every* paired capture; no stream row
     whose capture cannot be identified; unambiguous pairing;
   * pairs exist but none is eligible -> ``COMPARISON_UNASSESSABLE``;
   * no eligible paired capture with equal product multisets (and no invalid
     evidence) -> ``LIKELY_DISTINCT_STREAMS``;
   * anything else -> ``COMPARISON_INCONCLUSIVE`` (insufficient duplicate
     evidence is never read as distinctness).
   Behaviour never confirms an alias or distinctness, never resolves a
   location policy and never enables pricing.

Pairing: ``SHARED_COLLECTION_EVENT`` pairs exactly by collection event;
``CAPTURE_TIME`` pairs reconciled instants within an explicit tolerance and
fails closed on ambiguity or unresolved times. Offer multisets are compared
as tuples of the configured columns (``collections.Counter``: hashing plus
exact equality - no lossy fingerprint), preserving multiplicity and ignoring
row order. Prices are compared as exact source values and never identify
products.

Valid evidence
--------------
:func:`offer_signature` is the single offer-validity contract. Every product
and price column is required; a missing (``None``/``NaN``/``pd.NA``/``NaT``),
blank or - for ``numeric_columns`` - non-finite, negative or non-numeric
value leaves the offer without a signature. Missing values cannot support
identity inference: two unknown prices are not an observation that the
prices agree. A paired capture is *eligible* only when every offer on both
sides has a signature; otherwise it is neither matching nor differing
evidence and does not count toward the minimum. Invalid offers are counted,
never dropped, and multisets are built from valid signatures only.

The scope **baseline** applies the same pairing, validity and sufficiency
routine (:func:`_assess_pair`, :func:`_sufficiency_gaps`) to every other
location pair in the targets' scope: ``NON_DISCRIMINATIVE`` if any comparator
is identical in all its eligible captures (identical offers are then normal
for the source and prove nothing); ``DISCRIMINATIVE`` only if some comparator
pair meets the target's evidence standard (minimum eligible pairs, complete
unambiguous overlap, no invalid or unidentifiable rows) and has a differing
eligible capture; ``INSUFFICIENT`` if comparators exist but none qualifies;
``UNAVAILABLE`` if there are no paired comparators.

The report holds enums, booleans and aggregate evidence counts (captures per
stream, paired, eligible, invalid, unpaired, matching and differing paired
captures, the required minimum, baseline evidence) and a bounded,
repr-excluded sample of invalid offers (offer key, fields, reasons) - no
other values. Nothing merges, rewrites, drops,
canonicalises or writes rows; :func:`canonical_location_keys` is opt-in and
applies only aliases declared in the coverage contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.relationships import RelationshipPreconditionError, _check_relationship_inputs
from ql2_sixt_canada_analysis.schemas import (
    PROJECT_DEFAULT,
    project_default,
    CapturePairing,
    LocationCoverageConfigurationError,
    LocationCoverageDefinition,
    LocationStreamComparisonDefinition,
)
from ql2_sixt_canada_analysis.temporal import _parent_positions, parse_temporal_field

__all__ = [
    "INVALID_OFFER_SAMPLE_LIMIT",
    "BaselineEvidence",
    "InvalidOfferSample",
    "OfferDefect",
    "OfferSignature",
    "OfferStreamRole",
    "offer_signature",
    "ComparisonPreconditionError",
    "DuplicateInferenceBlocker",
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
    COMPARISON_UNASSESSABLE = "comparison_unassessable"   # paired, but no pair has valid offers on both sides


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
    INSUFFICIENT = "insufficient"        # comparators exist but none meets the evidence standard
    UNAVAILABLE = "unavailable"


class DuplicateInferenceBlocker(StrEnum):
    """A missing prerequisite for ``LIKELY_DUPLICATE_STREAMS`` (stable, ordered)."""

    NO_PAIRED_CAPTURES = "no_paired_captures"
    INSUFFICIENT_PAIRED_CAPTURES = "insufficient_paired_captures"
    INCOMPLETE_TEMPORAL_OVERLAP = "incomplete_temporal_overlap"
    TEMPORAL_OVERLAP_UNAVAILABLE = "temporal_overlap_unavailable"
    AMBIGUOUS_PAIRING = "ambiguous_pairing"
    UNASSESSABLE_OBSERVATIONS = "unassessable_observations"
    INVALID_OFFER_EVIDENCE = "invalid_offer_evidence"
    BASELINE_UNAVAILABLE = "baseline_unavailable"
    BASELINE_EVIDENCE_INSUFFICIENT = "baseline_evidence_insufficient"
    BASELINE_NON_DISCRIMINATIVE = "baseline_non_discriminative"
    DIFFERING_PAIRED_CAPTURES = "differing_paired_captures"


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
    first_capture_count: int
    second_capture_count: int
    paired_capture_count: int
    first_unpaired_capture_count: int
    second_unpaired_capture_count: int
    matching_paired_capture_count: int
    differing_paired_capture_count: int
    first_unassessable_row_count: int
    second_unassessable_row_count: int
    minimum_paired_captures: int
    duplicate_inference_blockers: tuple[DuplicateInferenceBlocker, ...]
    #: Paired captures with an invalid offer on at least one side (no evidence either way).
    invalid_paired_capture_count: int = 0
    first_side_invalid_pair_count: int = 0
    second_side_invalid_pair_count: int = 0
    first_invalid_offer_row_count: int = 0
    second_invalid_offer_row_count: int = 0
    #: Invalid offers in comparator (baseline-only) streams.
    baseline_invalid_offer_row_count: int = 0
    #: Invalid signature fields seen in the target streams (contract column order).
    invalid_offer_fields: tuple[str, ...] = ()
    #: Why target offers were invalid (fixed order).
    invalid_offer_reasons: tuple[OfferDefect, ...] = ()
    baseline_evidence: BaselineEvidence | None = None
    #: Bounded, sorted invalid-offer sample (target and baseline); confidential identifiers.
    invalid_offer_sample: tuple[InvalidOfferSample, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        # Programmer invariants for the evidence denominator.
        assert 0 <= self.invalid_paired_capture_count <= self.paired_capture_count
        assert self.matching_paired_capture_count + self.differing_paired_capture_count == (
            self.paired_capture_count - self.invalid_paired_capture_count)
        assert len(self.invalid_offer_sample) <= INVALID_OFFER_SAMPLE_LIMIT
        assert self.first_capture_count == self.paired_capture_count + self.first_unpaired_capture_count
        assert self.second_capture_count == self.paired_capture_count + self.second_unpaired_capture_count
        if self.status is LocationStreamComparisonStatus.LIKELY_DUPLICATE_STREAMS:
            assert not self.duplicate_inference_blockers

    @property
    def eligible_paired_capture_count(self) -> int:
        """Paired captures with valid signatures on both sides (the only evidence counted)."""
        return self.paired_capture_count - self.invalid_paired_capture_count

    @property
    def temporal_pairing_sufficient(self) -> bool:
        B = DuplicateInferenceBlocker
        return not ({B.NO_PAIRED_CAPTURES, B.INCOMPLETE_TEMPORAL_OVERLAP, B.TEMPORAL_OVERLAP_UNAVAILABLE,
                     B.AMBIGUOUS_PAIRING, B.UNASSESSABLE_OBSERVATIONS} & set(self.duplicate_inference_blockers))

    @property
    def offer_validity_sufficient(self) -> bool:
        return DuplicateInferenceBlocker.INVALID_OFFER_EVIDENCE not in self.duplicate_inference_blockers

    @property
    def target_threshold_passed(self) -> bool:
        """Eligible (not merely temporal) paired captures reach the minimum."""
        return self.eligible_paired_capture_count >= self.minimum_paired_captures

    @property
    def baseline_threshold_passed(self) -> bool:
        return self.baseline_evidence is not None and self.baseline_evidence.threshold_passed

    @property
    def baseline_discriminative(self) -> bool:
        return self.scope_baseline is ScopeBaseline.DISCRIMINATIVE

    @property
    def duplicate_inference_permitted(self) -> bool:
        """Behavioural duplicate evidence holds (still evidence, never authority)."""
        return self.status is LocationStreamComparisonStatus.LIKELY_DUPLICATE_STREAMS

    @property
    def invalid_evidence_streams(self) -> tuple[OfferStreamRole, ...]:
        """Where invalid offers occurred (fixed order): first/second target and/or baseline stream."""
        roles = []
        if self.first_invalid_offer_row_count:
            roles.append(OfferStreamRole.FIRST)
        if self.second_invalid_offer_row_count:
            roles.append(OfferStreamRole.SECOND)
        if self.baseline_invalid_offer_row_count:
            roles.append(OfferStreamRole.BASELINE)
        return tuple(roles)

    @property
    def duplicate_evidence_sufficient(self) -> bool:
        """Every prerequisite of a behavioural likely-duplicate holds (still not authority)."""
        return not self.duplicate_inference_blockers

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
    definition: LocationStreamComparisonDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
) -> LocationStreamComparisonReport:
    """Compare the two configured streams; returns a categorical report.

    Raises:
        TypeError: Invalid argument types.
        LocationCoverageConfigurationError: Configured columns are absent.
        ComparisonPreconditionError: Parent key, identifier dtype or
            blank-row preconditions fail.
    """
    definition = project_default(definition, "LOCATION_STREAM_COMPARISON")
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
            scope_baseline=ScopeBaseline.UNAVAILABLE,
            **_evidence(_assess_pair(cars, first_mask, second_mask, definition, _Offers(cars, definition),
                                     _unassessable_rows(cars, definition)), definition, ScopeBaseline.UNAVAILABLE),
            baseline_evidence=BaselineEvidence(**_EMPTY_BASELINE,
                                               minimum_paired_captures=definition.minimum_paired_captures))

    identity = _identity(cars, first_mask, second_mask, definition.identity_columns)
    events_first = _events(cars, first_mask, rel)
    events_second = _events(cars, second_mask, rel)
    shares_events = bool(set(events_first) & set(events_second))
    offers = _Offers(cars, definition)
    unassessable = _unassessable_rows(cars, definition)
    target = _assess_pair(cars, first_mask, second_mask, definition, offers, unassessable)
    overlap, pairs, ambiguous = target.overlap, target.pairs, target.ambiguous
    products, priced = _offer_sets(target)
    baseline, baseline_evidence, comparator_rows = _baseline(
        cars, keys, first_mask | second_mask, definition, offers, unassessable)
    evidence = _evidence(target, definition, baseline)
    columns = (*definition.product_columns, *definition.price_columns)
    invalid_rows = np.flatnonzero((first_mask | second_mask) & ~offers.valid)
    bad = {c for p in invalid_rows for c in offers.invalid_fields[p]}
    bad_reasons = {r for p in invalid_rows for r in offers.reasons[p]}

    S = LocationStreamComparisonStatus
    if identity is IdentityEvidence.CONFLICTING:
        status = S.LOCATION_MAPPING_DEFECT
    elif identity is IdentityEvidence.DIFFERENT:
        status = S.CONFIRMED_DISTINCT_LOCATIONS
    elif identity is IdentityEvidence.SAME:
        status = S.DUPLICATED_COLLECTION_CONFIGURATION if shares_events else S.CONFIRMED_ALIAS
    elif not pairs:
        status = S.INSUFFICIENT_COMPARABLE_CAPTURES
    elif not target.eligible_count:
        status = S.COMPARISON_UNASSESSABLE                     # paired, but nothing valid to compare
    elif not evidence["duplicate_inference_blockers"]:
        status = S.LIKELY_DUPLICATE_STREAMS                    # every prerequisite affirmatively met
    elif target.invalid_count or target.x_invalid_rows or target.y_invalid_rows:
        status = S.COMPARISON_INCONCLUSIVE                     # invalid evidence supports no conclusion
    elif products is OfferSetResult.DISTINCT:
        status = S.LIKELY_DISTINCT_STREAMS
    else:
        status = S.COMPARISON_INCONCLUSIVE
    return LocationStreamComparisonReport(
        status=status, **base, identity_evidence=identity, shares_collection_events=shares_events,
        temporal_overlap=overlap, ambiguous_pairing=ambiguous, comparable_captures_exist=bool(target.full_equal),
        product_sets=products, price_aware_offers=priced,
        synchronized_prices=(priced is OfferSetResult.IDENTICAL) if target.full_equal else None,
        scope_baseline=baseline, **evidence,
        invalid_offer_fields=tuple(c for c in columns if c in bad),
        invalid_offer_reasons=tuple(r for r in OfferDefect if r in bad_reasons), baseline_evidence=baseline_evidence,
        baseline_invalid_offer_row_count=int((comparator_rows & ~first_mask & ~second_mask & ~offers.valid).sum()),
        invalid_offer_sample=_invalid_sample(offers, first_mask, second_mask, comparator_rows))


def validate_confirmed_location_alias(
    jobs: pd.DataFrame,
    cars: pd.DataFrame,
    definition: LocationStreamComparisonDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
) -> LocationStreamComparisonReport:
    """Return the report only if an alias is confirmed by authority; else raise."""
    definition = project_default(definition, "LOCATION_STREAM_COMPARISON")
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


# ------------------------------------------------------------ offer validity

#: Maximum number of entries in each invalid-offer diagnostic sample.
INVALID_OFFER_SAMPLE_LIMIT = 10


class OfferDefect(StrEnum):
    """Why a signature field is invalid (fixed order; values name no columns)."""

    MISSING_VALUE = "missing_value"          # None, NaN, pd.NA, NaT or another missing marker
    BLANK_TEXT = "blank_text"                # empty or whitespace-only text
    INVALID_NUMBER = "invalid_number"        # numeric field that is not a finite, non-negative number


class OfferStreamRole(StrEnum):
    """Which stream an invalid offer belongs to (fixed order)."""

    FIRST = "first_target_stream"
    SECOND = "second_target_stream"
    BASELINE = "baseline_stream"


@dataclass(frozen=True, slots=True)
class OfferSignature:
    """Comparison signature of one offer, or why none exists.

    ``values`` is the exact tuple of signature values when every required
    field is valid, else ``None`` - and an invalid signature is never
    compared: two missing prices are not equal evidence, they are no evidence.
    """

    values: tuple[object, ...] | None
    invalid_fields: tuple[str, ...]
    reasons: tuple[OfferDefect, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.invalid_fields


@dataclass(frozen=True, slots=True)
class InvalidOfferSample:
    """One offer that cannot form a signature (confidential identifiers; bounded, sorted)."""

    role: OfferStreamRole
    offer_key: tuple[object, ...]
    invalid_fields: tuple[str, ...]
    reasons: tuple[OfferDefect, ...]


def _missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _field_defect(value: object, numeric: bool) -> OfferDefect | None:
    """One signature field: present, non-blank text, and a finite non-negative number when numeric.

    Truthiness is never used: ``0`` is a valid number, ``""`` is blank. Missing
    values cannot support identity inference because "unknown" equals
    "unknown" only as a token, not as an observed product or price.
    """
    if isinstance(value, str):
        if not value.strip():
            return OfferDefect.BLANK_TEXT
        if not numeric:
            return None
        try:
            number = float(value)
        except ValueError:
            return OfferDefect.INVALID_NUMBER
        ok = value == value.strip() and bool(np.isfinite(number)) and number >= 0
        return None if ok else OfferDefect.INVALID_NUMBER
    if _missing(value):
        return OfferDefect.MISSING_VALUE
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return OfferDefect.INVALID_NUMBER
    if not numeric:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return OfferDefect.INVALID_NUMBER
    return None if bool(np.isfinite(float(value))) and float(value) >= 0 else OfferDefect.INVALID_NUMBER


def _field_valid(value: object, numeric: bool) -> bool:
    return _field_defect(value, numeric) is None


def offer_signature(row: Mapping[str, object], definition: LocationStreamComparisonDefinition) -> OfferSignature:
    """The single offer-validity contract: a price-aware signature, or the invalid fields.

    Every product and price column of ``definition`` is required. A value
    that is missing (``None``, ``NaN``, ``pd.NA``, ``NaT``), blank or
    whitespace-only text, or - for ``numeric_columns`` - not a finite,
    non-negative number makes the field invalid. Values are never imputed,
    trimmed or converted: valid signatures hold the exact source values.
    """
    columns = (*definition.product_columns, *definition.price_columns)
    numeric = set(definition.numeric_columns)
    defects = {c: _field_defect(row.get(c), c in numeric) for c in columns}
    invalid = tuple(c for c in columns if defects[c] is not None)
    reasons = tuple(r for r in OfferDefect if r in defects.values())
    return OfferSignature(values=None if invalid else tuple(row.get(c) for c in columns), invalid_fields=invalid,
                          reasons=reasons)


class _Offers:
    """Per-row signatures grouped by collection event (built once per comparison)."""

    def __init__(self, cars: pd.DataFrame, d: LocationStreamComparisonDefinition) -> None:
        columns = (*d.product_columns, *d.price_columns)
        numeric = set(d.numeric_columns)
        n = len(cars)
        raw = {c: cars[c].astype(object).tolist() for c in columns}
        invalid = [[] for _ in range(n)]
        reasons: list[set] = [set() for _ in range(n)]
        for c in columns:
            for position, value in enumerate(raw[c]):
                defect = _field_defect(value, c in numeric)
                if defect is not None:
                    invalid[position].append(c)
                    reasons[position].add(defect)
        self.invalid_fields: list[tuple[str, ...]] = [tuple(v) for v in invalid]
        self.reasons: list[tuple[OfferDefect, ...]] = [tuple(r for r in OfferDefect if r in rs) for rs in reasons]
        self.valid = np.fromiter((not v for v in invalid), dtype=bool, count=n)
        width = len(d.product_columns)
        rows = list(zip(*(raw[c] for c in columns), strict=True)) if columns else [()] * n
        self.full = [r if ok else None for r, ok in zip(rows, self.valid, strict=True)]
        self.products = [r[:width] if r is not None else None for r in self.full]
        id_columns = d.relationship.detail_definition.unique_key_columns or d.relationship.detail_key_columns
        self.offer_keys = _rows(cars, tuple(id_columns))
        self.by_event: dict[tuple, list[int]] = {}
        for position, event in enumerate(_rows(cars, d.relationship.detail_key_columns)):
            self.by_event.setdefault(event, []).append(position)

    def rows(self, mask: np.ndarray, event: tuple) -> list[int]:
        return [p for p in self.by_event.get(event, ()) if mask[p]]

    def capture_valid(self, mask: np.ndarray, event: tuple) -> bool:
        """Capture-level fail-closed rule: offers present and every one has a signature."""
        positions = self.rows(mask, event)
        return bool(positions) and all(self.valid[p] for p in positions)

    def multiset(self, signatures: list, mask: np.ndarray, event: tuple) -> Counter:
        # Only called for valid captures: no None ever enters a multiset.
        values = [signatures[p] for p in self.rows(mask, event)]
        assert all(v is not None for v in values)
        return Counter(values)

    def invalid_samples(self, mask: np.ndarray, role: OfferStreamRole) -> list[InvalidOfferSample]:
        return [InvalidOfferSample(role, self.offer_keys[p], self.invalid_fields[p], self.reasons[p])
                for p in np.flatnonzero(mask & ~self.valid)]


@dataclass(frozen=True, slots=True, eq=False)
class _PairEvidence:
    """Evidence for one stream pair (target or baseline comparator), from one shared routine."""

    overlap: TemporalOverlap
    ambiguous: bool
    pairs: list
    eligible: list            # per pair: both sides valid
    x_invalid_pairs: int      # pairs whose first side is invalid
    y_invalid_pairs: int
    product_equal: list       # per *eligible* pair
    full_equal: list          # per *eligible* pair
    x_captures: int
    y_captures: int
    x_unpaired: int
    y_unpaired: int
    x_unassessable_rows: int
    y_unassessable_rows: int
    x_invalid_rows: int
    y_invalid_rows: int

    @property
    def eligible_count(self) -> int:
        return sum(self.eligible)

    @property
    def invalid_count(self) -> int:
        return len(self.pairs) - self.eligible_count

    @property
    def matching(self) -> int:
        return sum(self.full_equal)


def _unassessable_rows(cars: pd.DataFrame, d: LocationStreamComparisonDefinition) -> np.ndarray:
    """Rows whose capture cannot be identified (counted, never dropped)."""
    rel = d.relationship
    unassessable = ~cars.loc[:, list(rel.detail_key_columns)].notna().all(axis=1).to_numpy()
    if d.pairing is CapturePairing.CAPTURE_TIME:
        field = d.temporal.field(d.capture_time_field)
        parsed = parse_temporal_field(cars[field.column], field, d.temporal.canonical_timezone)
        unassessable = unassessable | parsed.instants.isna().to_numpy()
    return unassessable


def _assess_pair(cars: pd.DataFrame, x: np.ndarray, y: np.ndarray, d: LocationStreamComparisonDefinition,
                 offers: _Offers, unassessable: np.ndarray) -> _PairEvidence:
    """Pair captures and classify every pair as eligible or invalid (same rule for target and baseline)."""
    rel = d.relationship
    overlap, pairs, ambiguous = _pair(cars, x, y, d)
    ev_x, ev_y = set(_events(cars, x, rel)), set(_events(cars, y, rel))
    eligible, product_equal, full_equal = [], [], []
    x_bad = y_bad = 0
    for ex, ey in pairs:
        ok_x, ok_y = offers.capture_valid(x, ex), offers.capture_valid(y, ey)
        x_bad += not ok_x
        y_bad += not ok_y
        eligible.append(ok_x and ok_y)
        if ok_x and ok_y:
            product_equal.append(offers.multiset(offers.products, x, ex) == offers.multiset(offers.products, y, ey))
            full_equal.append(offers.multiset(offers.full, x, ex) == offers.multiset(offers.full, y, ey))
    paired_x, paired_y = {ex for ex, _ in pairs}, {ey for _, ey in pairs}
    return _PairEvidence(
        overlap=overlap, ambiguous=ambiguous, pairs=pairs, eligible=eligible, x_invalid_pairs=x_bad,
        y_invalid_pairs=y_bad, product_equal=product_equal, full_equal=full_equal,
        x_captures=len(ev_x | paired_x), y_captures=len(ev_y | paired_y),
        x_unpaired=len(ev_x - paired_x), y_unpaired=len(ev_y - paired_y),
        x_unassessable_rows=int((x & unassessable).sum()), y_unassessable_rows=int((y & unassessable).sum()),
        x_invalid_rows=int((x & ~offers.valid).sum()), y_invalid_rows=int((y & ~offers.valid).sum()),
    )


def _sufficiency_gaps(ev: _PairEvidence, d: LocationStreamComparisonDefinition) -> list[DuplicateInferenceBlocker]:
    """The one evidence-sufficiency standard, applied to the target pair and every baseline comparator.

    Counts only *eligible* paired captures against ``minimum_paired_captures``
    and requires complete, unambiguous overlap, identifiable rows and no
    invalid offer in either stream.
    """
    B = DuplicateInferenceBlocker
    gaps: list[DuplicateInferenceBlocker] = []
    if not ev.pairs:
        gaps.append(B.NO_PAIRED_CAPTURES)
    elif ev.eligible_count < d.minimum_paired_captures:
        gaps.append(B.INSUFFICIENT_PAIRED_CAPTURES)
    if ev.overlap is TemporalOverlap.UNAVAILABLE:
        gaps.append(B.TEMPORAL_OVERLAP_UNAVAILABLE)
    elif ev.overlap is not TemporalOverlap.COMPLETE:
        gaps.append(B.INCOMPLETE_TEMPORAL_OVERLAP)
    if ev.ambiguous:
        gaps.append(B.AMBIGUOUS_PAIRING)
    if ev.x_unassessable_rows or ev.y_unassessable_rows:
        gaps.append(B.UNASSESSABLE_OBSERVATIONS)
    if ev.invalid_count or ev.x_invalid_rows or ev.y_invalid_rows:
        gaps.append(B.INVALID_OFFER_EVIDENCE)
    return gaps


def _offer_sets(ev: _PairEvidence) -> tuple[OfferSetResult, OfferSetResult]:
    """(products, price-aware offers) over *eligible* pairs only."""
    if not ev.full_equal:
        return OfferSetResult.UNAVAILABLE, OfferSetResult.UNAVAILABLE
    product_equal, full_equal = ev.product_equal, ev.full_equal
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


@dataclass(frozen=True, slots=True)
class BaselineEvidence:
    """Aggregate evidence behind the scope baseline (counts summed over comparator pairs).

    A comparator pair *qualifies* only under the same standard as the target
    pair (:func:`_sufficiency_gaps`: at least ``minimum_paired_captures``
    eligible pairs, complete unambiguous overlap, no invalid or unidentifiable
    rows). The baseline is ``DISCRIMINATIVE`` only when some qualified pair has
    at least one differing eligible capture and no comparator pair is
    identical in all its eligible captures.
    """

    comparator_pair_count: int
    paired_comparator_pair_count: int
    qualified_comparator_pair_count: int
    capture_count: int
    paired_capture_count: int
    eligible_paired_capture_count: int
    invalid_paired_capture_count: int
    identical_eligible_capture_count: int
    discriminatory_eligible_capture_count: int
    minimum_paired_captures: int

    @property
    def threshold_passed(self) -> bool:
        """At least one comparator pair meets the target's evidence standard."""
        return self.qualified_comparator_pair_count > 0


_EMPTY_BASELINE = dict(comparator_pair_count=0, paired_comparator_pair_count=0, qualified_comparator_pair_count=0,
                       capture_count=0, paired_capture_count=0, eligible_paired_capture_count=0,
                       invalid_paired_capture_count=0, identical_eligible_capture_count=0,
                       discriminatory_eligible_capture_count=0)


_AFFIRMATIVE_BASELINES = frozenset({ScopeBaseline.DISCRIMINATIVE})       # allowlist, not "anything but"
_BASELINE_GAPS = {ScopeBaseline.UNAVAILABLE: DuplicateInferenceBlocker.BASELINE_UNAVAILABLE,
                  ScopeBaseline.INSUFFICIENT: DuplicateInferenceBlocker.BASELINE_EVIDENCE_INSUFFICIENT,
                  ScopeBaseline.NON_DISCRIMINATIVE: DuplicateInferenceBlocker.BASELINE_NON_DISCRIMINATIVE}


def _evidence(ev: _PairEvidence, d: LocationStreamComparisonDefinition, baseline: ScopeBaseline) -> dict:
    """Auditable target evidence counts and the ordered gaps blocking a likely-duplicate inference.

    Counts are independent capture events (complete detail relationship
    keys), never offer rows; unidentifiable rows and invalid offers are
    counted, not dropped.
    """
    gaps = _sufficiency_gaps(ev, d)
    if baseline not in _AFFIRMATIVE_BASELINES:
        gaps.append(_BASELINE_GAPS[baseline])
    if ev.matching < ev.eligible_count:
        gaps.append(DuplicateInferenceBlocker.DIFFERING_PAIRED_CAPTURES)
    order = list(DuplicateInferenceBlocker)
    return dict(
        first_capture_count=ev.x_captures,
        second_capture_count=ev.y_captures,
        paired_capture_count=len(ev.pairs),
        first_unpaired_capture_count=ev.x_unpaired,
        second_unpaired_capture_count=ev.y_unpaired,
        matching_paired_capture_count=ev.matching,
        differing_paired_capture_count=ev.eligible_count - ev.matching,
        first_unassessable_row_count=ev.x_unassessable_rows,
        second_unassessable_row_count=ev.y_unassessable_rows,
        minimum_paired_captures=d.minimum_paired_captures,
        duplicate_inference_blockers=tuple(sorted(dict.fromkeys(gaps), key=order.index)),
        invalid_paired_capture_count=ev.invalid_count,
        first_side_invalid_pair_count=ev.x_invalid_pairs,
        second_side_invalid_pair_count=ev.y_invalid_pairs,
        first_invalid_offer_row_count=ev.x_invalid_rows,
        second_invalid_offer_row_count=ev.y_invalid_rows,
    )


def _baseline(cars: pd.DataFrame, keys: pd.Series, target_mask: np.ndarray, d: LocationStreamComparisonDefinition,
              offers: _Offers, unassessable: np.ndarray) -> tuple[ScopeBaseline, BaselineEvidence, np.ndarray]:
    """Is fully identical behaviour typical of other location pairs in the targets' scope?

    Every comparator pair is assessed with :func:`_assess_pair` and judged by
    :func:`_sufficiency_gaps` - the target's own standard. Invalid offers are
    neither matching nor differing evidence. Returns (baseline, evidence,
    mask of comparator rows).
    """
    empty = np.zeros(len(cars), dtype=bool)
    scope = d.coverage.stream_scope_columns
    if not scope:
        return ScopeBaseline.UNAVAILABLE, BaselineEvidence(**_EMPTY_BASELINE,
                                                           minimum_paired_captures=d.minimum_paired_captures), empty
    scope_keys = pd.Series(_rows(cars, scope), index=cars.index, dtype=object)
    target_scopes = set(scope_keys[target_mask])
    in_scope = np.fromiter((k in target_scopes for k in scope_keys), dtype=bool, count=len(scope_keys))
    others = sorted({k for k in keys[in_scope & ~target_mask] if all(v is not None for v in k)}, key=repr)
    candidates = sorted({*others, d.first, d.second}, key=repr)
    totals = dict(_EMPTY_BASELINE)
    identical_any = discriminating_qualified = False
    comparator_rows = empty.copy()
    for x, y in combinations(candidates, 2):
        if {x, y} == {d.first, d.second}:
            continue
        mx, my = _key_mask(keys, x) & in_scope, _key_mask(keys, y) & in_scope
        ev = _assess_pair(cars, mx, my, d, offers, unassessable)
        totals["comparator_pair_count"] += 1
        if not ev.pairs:
            continue
        comparator_rows |= (mx | my) & ~target_mask
        qualified = not _sufficiency_gaps(ev, d)
        differing = ev.eligible_count - ev.matching
        totals["paired_comparator_pair_count"] += 1
        totals["qualified_comparator_pair_count"] += qualified
        totals["capture_count"] += ev.x_captures + ev.y_captures
        totals["paired_capture_count"] += len(ev.pairs)
        totals["eligible_paired_capture_count"] += ev.eligible_count
        totals["invalid_paired_capture_count"] += ev.invalid_count
        totals["identical_eligible_capture_count"] += ev.matching
        totals["discriminatory_eligible_capture_count"] += differing
        if ev.eligible_count and differing == 0:
            identical_any = True             # identical behaviour occurs in scope (conservative: any evidence)
        if qualified and differing:
            discriminating_qualified = True
    evidence = BaselineEvidence(**totals, minimum_paired_captures=d.minimum_paired_captures)
    if not totals["paired_comparator_pair_count"]:
        baseline = ScopeBaseline.UNAVAILABLE
    elif identical_any:
        baseline = ScopeBaseline.NON_DISCRIMINATIVE
    elif discriminating_qualified:
        baseline = ScopeBaseline.DISCRIMINATIVE
    else:
        baseline = ScopeBaseline.INSUFFICIENT
    return baseline, evidence, comparator_rows


def _invalid_sample(offers: _Offers, first: np.ndarray, second: np.ndarray,
                    comparator: np.ndarray) -> tuple[InvalidOfferSample, ...]:
    """Bounded, deterministic sample of invalid offers in the target and baseline streams."""
    samples = [*offers.invalid_samples(first, OfferStreamRole.FIRST),
               *offers.invalid_samples(second, OfferStreamRole.SECOND),
               *offers.invalid_samples(comparator & ~first & ~second, OfferStreamRole.BASELINE)]
    order = list(OfferStreamRole)
    samples.sort(key=lambda s: (order.index(s.role), tuple((v is None, str(v)) for v in s.offer_key),
                                s.invalid_fields))
    return tuple(samples[:INVALID_OFFER_SAMPLE_LIMIT])
