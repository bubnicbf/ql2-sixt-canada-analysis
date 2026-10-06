"""Check that the configured cleaned dataset covers the expected locations.

The contract is :data:`~ql2_sixt_canada_analysis.schemas.EXPECTED_LOCATION_COVERAGE`
(a :class:`~ql2_sixt_canada_analysis.schemas.LocationCoverageDefinition`).
Expected locations come from an independent authority, never from the frame
being assessed; there is deliberately no helper that builds them from data.
An unconfigured contract fails closed with
:class:`~ql2_sixt_canada_analysis.schemas.LocationCoverageConfigurationError`.

Semantics
---------
Each source row's location key (one component per configured column) is:

* **unassigned** - any component is missing, empty or whitespace-only (the
  row creates no observed location; the source value is not modified);
* **expected** - the complete key equals an expected key exactly; or
* **unexpected** - the complete key matches no expected key.

Coverage is measured on *distinct* complete observed keys: an expected
location is covered when at least one source row has its exact key, and
repeated rows at one location never compensate for another missing location.
Comparison is exact (case-sensitive, no stripping, no fuzzy matching) and
composite keys are compared as tuples, never concatenated. Only aliases
declared in the central contract (``aliases``) also count, matched exactly.

Composite keys are pairs such as ``(city, location)``: a location label
observed under a different city is a different, *unexpected* key and never
covers the expected pair. With ``label_column`` configured, a label observed
under more than one combination of the other components, at least one of
them outside the contract, is a **conflicting assignment**
(``conflicting_location_label_count``) and fails the contract; a label the
contract itself uses in several cities (``Downtown``) is not a conflict while
every observed combination is expected. City is never inferred from the label.

**Source spelling mismatch** (diagnostic only): an unexpected observed key
that is a case/space/punctuation variant of an expected key - either the
whole key (components in any order) or any single component compared with
the same component of the expected keys - is counted in
``spelling_variant_location_count`` and fails the contract
(``source_spelling_mismatch``). The folding (case-folding, dropping
everything but ASCII letters and digits) exists only to *detect* the
mismatch; it never makes a variant cover an expected key, and no value is
rewritten. Only approved aliases (``aliases``) cover, matched exactly.

The contract passes when every expected location is covered, no label has a
conflicting assignment, no spelling variant is observed and every source row
has a complete location; in
``EXHAUSTIVE`` mode it additionally requires zero unexpected locations,
while ``MINIMUM_REQUIRED`` mode only reports them.

Coverage is checked on the cleaned frame of the contract's dataset (after
blank-row removal) and before any one-to-many join. Branch-level locations
live in the detail dataset; a location counts once however many rows show it. Assessment
never modifies, sorts or removes rows and writes nothing; reports and errors
hold aggregate numbers and booleans plus the *configured* expected keys that
are missing (configuration, not source values) - never observed values.
:func:`location_pair_evidence` returns observed keys in memory only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

import pandas as pd

from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.schemas import (
    PROJECT_DEFAULT,
    project_default,
    LocationCoverageConfigurationError,
    LocationCoverageDefinition,
    LocationCoverageMode,
)

__all__ = [
    "LocationCoverageConfigurationError",
    "assess_dataset_location_coverage",
    "LocationCoverageError",
    "LocationCoverageReport",
    "assess_expected_location_coverage",
    "location_pair_evidence",
    "spelling_variant_keys",
    "validate_expected_location_coverage",
]


@dataclass(frozen=True, slots=True)
class LocationCoverageReport:
    """Aggregate expected-location coverage metrics (no location values)."""

    mode: LocationCoverageMode
    row_count: int
    expected_location_count: int
    observed_location_count: int
    covered_expected_location_count: int
    missing_expected_location_count: int
    unexpected_location_count: int
    missing_location_row_count: int
    conflicting_location_label_count: int = 0
    #: Unexpected observed keys that are spelling variants of expected keys (diagnostic count).
    spelling_variant_location_count: int = 0
    missing_expected_locations: tuple[tuple[str, ...], ...] = ()
    expected_pairs: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        # Programmer invariants; a violation is a bug in this module.
        assert isinstance(self.mode, LocationCoverageMode)
        assert len(self.missing_expected_locations) == self.missing_expected_location_count
        for f in fields(self):
            if f.name not in ("mode", "missing_expected_locations", "expected_pairs"):
                value = getattr(self, f.name)
                assert type(value) is int and value >= 0, f.name
        assert self.expected_location_count > 0
        assert self.covered_expected_location_count + self.missing_expected_location_count == \
            self.expected_location_count
        # Without aliases, observed = covered + unexpected; authoritative
        # aliases can let several observed keys cover one expected location.
        assert self.unexpected_location_count <= self.observed_location_count
        assert self.spelling_variant_location_count <= self.unexpected_location_count
        assert self.missing_location_row_count <= self.row_count
        assert self.observed_location_count <= self.row_count - self.missing_location_row_count

    @property
    def coverage_ratio(self) -> float:
        """Covered expected locations / expected locations (0.0 to 1.0)."""
        return self.covered_expected_location_count / self.expected_location_count

    @property
    def all_expected_covered(self) -> bool:
        return self.missing_expected_location_count == 0

    @property
    def unexpected_locations_acceptable(self) -> bool:
        return self.mode is LocationCoverageMode.MINIMUM_REQUIRED or self.unexpected_location_count == 0

    @property
    def all_rows_assigned(self) -> bool:
        return self.missing_location_row_count == 0

    @property
    def missing_pairs(self) -> tuple[tuple[str, ...], ...]:
        """Configured expected keys not observed (configuration values, not source values)."""
        return self.missing_expected_locations

    @property
    def no_conflicting_assignments(self) -> bool:
        return self.conflicting_location_label_count == 0

    @property
    def no_spelling_variants(self) -> bool:
        return self.spelling_variant_location_count == 0

    @property
    def is_valid(self) -> bool:
        """The configured coverage contract holds."""
        return (self.all_expected_covered and self.all_rows_assigned and self.unexpected_locations_acceptable
                and self.no_conflicting_assignments and self.no_spelling_variants)

    @property
    def violations(self) -> tuple[str, ...]:
        """Safe violation categories present (empty when valid)."""
        checks = (
            ("missing_expected_location", not self.all_expected_covered),
            ("unexpected_location", not self.unexpected_locations_acceptable),
            ("missing_location_assignment", not self.all_rows_assigned),
            ("conflicting_location_assignment", not self.no_conflicting_assignments),
            ("source_spelling_mismatch", not self.no_spelling_variants),
        )
        return tuple(name for name, failed in checks if failed)


class LocationCoverageError(Exception):
    """Strict validation found the coverage contract violated.

    The message lists violation categories only; the aggregate report is on
    ``report``.
    """

    def __init__(self, report: LocationCoverageReport) -> None:
        super().__init__("Expected-location coverage contract failed: " + ", ".join(report.violations) + ".")
        self.report = report


def assess_expected_location_coverage(
    jobs: pd.DataFrame,
    coverage: LocationCoverageDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
) -> LocationCoverageReport:
    """Compare distinct cleaned-jobs locations with the expected contract.

    Never raises for coverage violations; ``jobs`` is not modified.

    Raises:
        TypeError: Invalid argument types.
        LocationCoverageConfigurationError: A location column is absent, or
            no authoritative expected locations are configured (fail closed).
    """
    coverage = project_default(coverage, "EXPECTED_LOCATION_COVERAGE")
    if not isinstance(jobs, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(jobs).__name__}")
    if not isinstance(coverage, LocationCoverageDefinition):
        raise TypeError(f"expected a LocationCoverageDefinition, got {type(coverage).__name__}")
    columns = coverage.location_columns
    absent = tuple(c for c in columns if c not in jobs.columns)
    if absent:
        raise LocationCoverageConfigurationError(
            f"The '{coverage.dataset}' frame lacks {len(absent)} location column(s).", absent
        )
    if not coverage.is_configured:
        raise LocationCoverageConfigurationError(
            "No authoritative expected-location contract is configured; coverage cannot be "
            "assessed. Supply expected_locations and mode from an independent source."
        )
    assert coverage.mode is not None and coverage.expected_locations is not None

    keys, assigned = _assigned_keys(jobs, columns)
    names = [f"component_{i}" for i in range(len(columns))]
    observed = pd.MultiIndex.from_frame(keys.loc[assigned].drop_duplicates(), names=names)
    expected = pd.MultiIndex.from_tuples(list(coverage.expected_locations), names=names)
    # An expected location is covered by its own key or an authoritative alias.
    covered_flags = [
        bool(pd.MultiIndex.from_tuples(list(coverage.match_keys(key)), names=names).isin(observed).any())
        for key in coverage.expected_locations
    ]
    covered = sum(covered_flags)
    missing = tuple(key for key, hit in zip(coverage.expected_locations, covered_flags) if not hit)
    accepted = pd.MultiIndex.from_tuples(
        [k for key in coverage.expected_locations for k in coverage.match_keys(key)], names=names
    )
    conflicts = len(_conflicting_labels(keys.loc[assigned].drop_duplicates(), coverage))
    unexpected_mask = ~observed.isin(accepted)
    unexpected = int(unexpected_mask.sum())
    variants = _spelling_variant_count(observed[unexpected_mask], coverage.expected_locations)

    return LocationCoverageReport(
        mode=coverage.mode,
        row_count=len(jobs),
        expected_location_count=len(expected),
        observed_location_count=len(observed),
        covered_expected_location_count=covered,
        missing_expected_location_count=len(expected) - covered,
        unexpected_location_count=unexpected,
        missing_location_row_count=int((~assigned).sum()),
        conflicting_location_label_count=conflicts,
        spelling_variant_location_count=variants,
        missing_expected_locations=missing,
        expected_pairs=tuple(coverage.expected_locations),
    )


def location_pair_evidence(
    frame: pd.DataFrame,
    coverage: LocationCoverageDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
) -> pd.DataFrame:
    """Distinct observed complete keys with ``expected`` and ``conflicting_label`` flags.

    In memory only (a new frame on every call), sorted by key; observed source
    values are confidential - never print, log or persist them.
    """
    coverage = project_default(coverage, "EXPECTED_LOCATION_COVERAGE")
    if not isinstance(frame, pd.DataFrame) or not isinstance(coverage, LocationCoverageDefinition):
        raise TypeError("frame must be a DataFrame and coverage a LocationCoverageDefinition")
    columns = coverage.location_columns
    absent = tuple(c for c in columns if c not in frame.columns)
    if absent:
        raise LocationCoverageConfigurationError(
            f"The '{coverage.dataset}' frame lacks {len(absent)} location column(s).", absent)
    keys, assigned = _assigned_keys(frame, columns)
    observed = keys.loc[assigned].drop_duplicates().astype(object).reset_index(drop=True)
    accepted = {k for key in (coverage.expected_locations or ()) for k in coverage.match_keys(key)}
    tuples = list(map(tuple, observed.itertuples(index=False)))
    observed["expected"] = [t in accepted for t in tuples]
    conflicting = _conflicting_labels(observed.loc[:, list(columns)], coverage)
    observed["conflicting_label"] = (observed[coverage.label_column].isin(conflicting)
                                     if coverage.label_column else False)
    return observed.sort_values(list(columns), kind="mergesort").reset_index(drop=True)


def _fold(value: object) -> str:
    """Diagnostic-only folding (case, spacing, punctuation); never applied to data or matching."""
    return re.sub(r"[^0-9a-z]+", "", str(value).casefold())


def spelling_variant_keys(observed: object, expected: tuple[tuple[str, ...], ...]) -> list[tuple[object, ...]]:
    """Observed keys (not exactly expected) that are spelling variants of the expected keys.

    In memory only: a key is a variant when its folded components equal the
    folded components of an expected key in some order, or when any component
    differs from every exact expected value at that position but folds to one
    of them. Exact keys are never variants. Nothing is rewritten.
    """
    exact = set(expected)
    folded_keys = {tuple(sorted(_fold(v) for v in key)) for key in expected}
    width = len(expected[0]) if expected else 0
    exact_parts = [{key[i] for key in expected} for i in range(width)]
    folded_parts = [{_fold(key[i]) for key in expected} for i in range(width)]
    variants = []
    for key in observed:
        key = tuple(key)
        if key in exact or len(key) != width:
            continue
        whole = tuple(sorted(_fold(v) for v in key)) in folded_keys
        part = any(key[i] not in exact_parts[i] and _fold(key[i]) in folded_parts[i] for i in range(width))
        if whole or part:
            variants.append(key)
    return variants


def _spelling_variant_count(observed: pd.MultiIndex, expected: tuple[tuple[str, ...], ...]) -> int:
    return len(spelling_variant_keys(list(observed), tuple(expected)))


def _assigned_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[pd.DataFrame, "pd.Series"]:
    """Location columns and a mask of rows whose every component is present and non-blank."""
    keys = frame.loc[:, list(columns)]                             # location columns only
    unassigned = keys.isna().to_numpy(dtype=bool, copy=True)
    for position, (_, column) in enumerate(keys.items()):
        # Only text can be empty/whitespace-only; other scalars never render
        # as blank text, so a vectorised strip on a temporary string view
        # suffices. The source column is untouched.
        blank = column.astype("string").str.strip().eq("").fillna(False)
        unassigned[:, position] |= blank.to_numpy(dtype=bool)
    return keys, ~unassigned.any(axis=1)


def _conflicting_labels(distinct_keys: pd.DataFrame, coverage: LocationCoverageDefinition) -> set:
    """Labels observed under more than one combination of the other key components, at least one unexpected.

    The approved contract may itself use one label in several cities (for
    example a ``Downtown`` branch per city); a label observed only under
    expected (or approved-alias) keys is therefore not a conflict. As soon as
    one of its observed combinations is outside the contract, the label is
    conflicting. City is never inferred from the label.
    """
    label = coverage.label_column
    others = [c for c in coverage.location_columns if c != label]
    if label is None or not others or distinct_keys.empty:
        return set()
    accepted = {k for key in (coverage.expected_locations or ()) for k in coverage.match_keys(key)}
    columns = list(coverage.location_columns)
    rows = distinct_keys.loc[:, columns].astype(object)
    keyed = rows.assign(_accepted=[tuple(r) in accepted for r in rows.itertuples(index=False)])
    grouped = keyed.groupby(label, dropna=False, sort=False)["_accepted"].agg(["size", "all"])
    return set(grouped.index[(grouped["size"] > 1) & ~grouped["all"]])


def assess_dataset_location_coverage(
    datasets: RawDatasets,
    coverage: LocationCoverageDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
) -> LocationCoverageReport:
    """Assess coverage on the frame of ``datasets`` that the contract names.

    Selects ``datasets.<coverage.dataset>`` (cleaned frames expected) so
    callers never hard-code which dataset carries the locations.
    """
    coverage = project_default(coverage, "EXPECTED_LOCATION_COVERAGE")
    if not isinstance(datasets, RawDatasets):
        raise TypeError(f"expected RawDatasets, got {type(datasets).__name__}")
    return assess_expected_location_coverage(getattr(datasets, coverage.dataset.value), coverage)


def validate_expected_location_coverage(
    jobs: pd.DataFrame,
    coverage: LocationCoverageDefinition = PROJECT_DEFAULT,  # type: ignore[assignment]
) -> LocationCoverageReport:
    """Assess, then return the report if valid or raise :class:`LocationCoverageError`."""
    coverage = project_default(coverage, "EXPECTED_LOCATION_COVERAGE")
    report = assess_expected_location_coverage(jobs, coverage)
    if not report.is_valid:
        raise LocationCoverageError(report)
    return report
