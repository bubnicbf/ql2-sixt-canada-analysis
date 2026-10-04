"""Check that cleaned jobs cover the independently configured expected locations.

The contract is :data:`~ql2_sixt_canada_analysis.schemas.EXPECTED_LOCATION_COVERAGE`
(a :class:`~ql2_sixt_canada_analysis.schemas.LocationCoverageDefinition`).
Expected locations come from an independent authority, never from the frame
being assessed; there is deliberately no helper that builds them from data.
An unconfigured contract fails closed with
:class:`~ql2_sixt_canada_analysis.schemas.LocationCoverageConfigurationError`.

Semantics
---------
Each jobs row's location key (one component per configured column) is:

* **unassigned** - any component is missing, empty or whitespace-only (the
  row creates no observed location; the source value is not modified);
* **expected** - the complete key equals an expected key exactly; or
* **unexpected** - the complete key matches no expected key.

Coverage is measured on *distinct* complete observed keys: an expected
location is covered when at least one job has its exact key, and repeated
jobs at one location never compensate for another missing location.
Comparison is exact (case-sensitive, no stripping, no fuzzy matching) and
composite keys are compared as tuples, never concatenated. Only aliases
declared in the central contract (``aliases``) also count, matched exactly.

The contract passes when every expected location is covered and every job has
a complete location; in ``EXHAUSTIVE`` mode it additionally requires zero
unexpected locations, while ``MINIMUM_REQUIRED`` mode only reports them.

Coverage is checked on the cleaned frame of the contract's dataset (after
blank-row removal) and before any one-to-many join. Branch-level locations
live in the detail dataset; a location counts once however many rows show it. Assessment
never modifies, sorts or removes rows and writes nothing; reports and errors
hold aggregate numbers and booleans only - never location values.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd

from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.schemas import (
    EXPECTED_LOCATION_COVERAGE,
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

    def __post_init__(self) -> None:
        # Programmer invariants; a violation is a bug in this module.
        assert isinstance(self.mode, LocationCoverageMode)
        for f in fields(self):
            if f.name != "mode":
                value = getattr(self, f.name)
                assert type(value) is int and value >= 0, f.name
        assert self.expected_location_count > 0
        assert self.covered_expected_location_count + self.missing_expected_location_count == \
            self.expected_location_count
        # Without aliases, observed = covered + unexpected; authoritative
        # aliases can let several observed keys cover one expected location.
        assert self.unexpected_location_count <= self.observed_location_count
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
    def is_valid(self) -> bool:
        """The configured coverage contract holds."""
        return self.all_expected_covered and self.all_rows_assigned and self.unexpected_locations_acceptable

    @property
    def violations(self) -> tuple[str, ...]:
        """Safe violation categories present (empty when valid)."""
        checks = (
            ("missing_expected_location", not self.all_expected_covered),
            ("unexpected_location", not self.unexpected_locations_acceptable),
            ("missing_location_assignment", not self.all_rows_assigned),
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
    coverage: LocationCoverageDefinition = EXPECTED_LOCATION_COVERAGE,
) -> LocationCoverageReport:
    """Compare distinct cleaned-jobs locations with the expected contract.

    Never raises for coverage violations; ``jobs`` is not modified.

    Raises:
        TypeError: Invalid argument types.
        LocationCoverageConfigurationError: A location column is absent, or
            no authoritative expected locations are configured (fail closed).
    """
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

    keys = jobs.loc[:, list(columns)]                              # location columns only
    unassigned = keys.isna().to_numpy(dtype=bool, copy=True)
    for position, (_, column) in enumerate(keys.items()):
        # Only text can be empty/whitespace-only; other scalars never render
        # as blank text, so a vectorised strip on a temporary string view
        # suffices. The source column is untouched.
        blank = column.astype("string").str.strip().eq("").fillna(False)
        unassigned[:, position] |= blank.to_numpy(dtype=bool)
    assigned = ~unassigned.any(axis=1)

    names = [f"component_{i}" for i in range(len(columns))]
    observed = pd.MultiIndex.from_frame(keys.loc[assigned].drop_duplicates(), names=names)
    expected = pd.MultiIndex.from_tuples(list(coverage.expected_locations), names=names)
    # An expected location is covered by its own key or an authoritative alias.
    covered = sum(
        bool(pd.MultiIndex.from_tuples(list(coverage.match_keys(key)), names=names).isin(observed).any())
        for key in coverage.expected_locations
    )
    accepted = pd.MultiIndex.from_tuples(
        [k for key in coverage.expected_locations for k in coverage.match_keys(key)], names=names
    )
    unexpected = int((~observed.isin(accepted)).sum())

    return LocationCoverageReport(
        mode=coverage.mode,
        row_count=len(jobs),
        expected_location_count=len(expected),
        observed_location_count=len(observed),
        covered_expected_location_count=covered,
        missing_expected_location_count=len(expected) - covered,
        unexpected_location_count=unexpected,
        missing_location_row_count=int((~assigned).sum()),
    )


def assess_dataset_location_coverage(
    datasets: RawDatasets,
    coverage: LocationCoverageDefinition = EXPECTED_LOCATION_COVERAGE,
) -> LocationCoverageReport:
    """Assess coverage on the frame of ``datasets`` that the contract names.

    Selects ``datasets.<coverage.dataset>`` (cleaned frames expected) so
    callers never hard-code which dataset carries the locations.
    """
    if not isinstance(datasets, RawDatasets):
        raise TypeError(f"expected RawDatasets, got {type(datasets).__name__}")
    return assess_expected_location_coverage(getattr(datasets, coverage.dataset.value), coverage)


def validate_expected_location_coverage(
    jobs: pd.DataFrame,
    coverage: LocationCoverageDefinition = EXPECTED_LOCATION_COVERAGE,
) -> LocationCoverageReport:
    """Assess, then return the report if valid or raise :class:`LocationCoverageError`."""
    report = assess_expected_location_coverage(jobs, coverage)
    if not report.is_valid:
        raise LocationCoverageError(report)
    return report
