"""The pricing-eligible population: which parent captures and detail rows may enter pricing analysis.

Built after every foundational control has run, from their typed results:

1. **Governed exclusion** - a detail row (or parent job) of a parent capture
   that the approved ``INCOMPLETE_PARENT_CAPTURE`` exclusion makes
   analytically null (:class:`~ql2_sixt_canada_analysis.collection_schedule.CaptureExclusionSet`,
   resolved by the per-stream scheduled-coverage assessment).
2. **Reporting day** - the row's scrape date must agree with the reporting day
   derived from its (trusted linked) parent
   (:func:`~ql2_sixt_canada_analysis.temporal.derive_reporting_days`); this
   also requires linkage and city integrity for detail rows.
3. **Rental dates** - the row must be rental-date eligible
   (:func:`~ql2_sixt_canada_analysis.rental_dates.derive_rental_periods`).
4. **Capture period** - the parent capture must be validly assigned to one
   scheduled period.

The exclusion is applied after temporal validation and before any cohort,
summary, comparison, product population, vehicle-stability, duplicate or
offer-count calculation. **Nothing is removed from the source frames**:
excluded and ineligible rows stay available for ingestion, linkage,
reconciliation, audit and exception reporting; this module only returns masks
and aggregate counts. A population is bound to the exact frames it was built
from (row counts and a content digest) and refuses any other frame, so stale
or tampered evidence can never be reused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

__all__ = [
    "DetailEligibility",
    "FrameBinding",
    "PricingPopulation",
    "PricingPopulationError",
    "build_pricing_population",
    "frame_binding",
]


class PricingPopulationError(ValueError):
    """The population cannot be built, or a frame does not match the bound evidence."""


class DetailEligibility(StrEnum):
    """Why a row is (not) in the pricing population (first failing control wins; counts only)."""

    ELIGIBLE = "eligible"
    GOVERNED_EXCLUSION = "governed_exclusion"          # approved INCOMPLETE_PARENT_CAPTURE (analytically null)
    REPORTING_DAY_FAILED = "reporting_day_failed"      # unlinked, untrusted, unresolvable, missing/invalid/mismatched
    RENTAL_DATES_FAILED = "rental_dates_failed"
    CAPTURE_PERIOD_UNASSIGNED = "capture_period_unassigned"


@dataclass(frozen=True, slots=True)
class FrameBinding:
    """Row counts and a content digest of the frames a result was computed from (no values)."""

    parent_rows: int
    detail_rows: int
    digest: str = field(repr=False)


def _digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame.astype(object), index=True).to_numpy()
    columns = "\x1f".join(map(str, frame.columns)).encode()
    return hashlib.sha256(columns + hashed.tobytes()).hexdigest()


def frame_binding(jobs: pd.DataFrame, cars: pd.DataFrame) -> FrameBinding:
    """Bind a result to the exact parent and detail frames (every column, index and value)."""
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be DataFrames")
    return FrameBinding(len(jobs), len(cars), hashlib.sha256(
        (_digest(jobs) + _digest(cars)).encode()).hexdigest())


@dataclass(frozen=True)
class PricingPopulation:
    """Pricing eligibility of every parent capture and detail row (aggregate counts; statuses in memory)."""

    binding: FrameBinding
    parent_status: tuple[str, ...] = field(repr=False)
    detail_status: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, FrameBinding):
            raise PricingPopulationError("a frame binding is required")
        if len(self.parent_status) != self.binding.parent_rows or len(self.detail_status) != self.binding.detail_rows:
            raise PricingPopulationError("the eligibility statuses do not match the bound frames")
        allowed = {s.value for s in DetailEligibility}
        if not set(self.parent_status) | set(self.detail_status) <= allowed:
            raise PricingPopulationError("unknown eligibility status")

    # ---- aggregate counts (never values)
    @property
    def nominal_parent_captures(self) -> int:
        return self.binding.parent_rows

    @property
    def excluded_parent_captures(self) -> int:
        return self.parent_status.count(DetailEligibility.GOVERNED_EXCLUSION.value)

    @property
    def eligible_parent_captures(self) -> int:
        return self.parent_status.count(DetailEligibility.ELIGIBLE.value)

    @property
    def nominal_detail_rows(self) -> int:
        return self.binding.detail_rows

    @property
    def excluded_detail_rows(self) -> int:
        return self.detail_status.count(DetailEligibility.GOVERNED_EXCLUSION.value)

    @property
    def eligible_detail_rows(self) -> int:
        return self.detail_status.count(DetailEligibility.ELIGIBLE.value)

    @property
    def ineligible_detail_rows(self) -> int:
        """Rows outside the governed exclusion that fail a foundational control (fail closed)."""
        return self.nominal_detail_rows - self.excluded_detail_rows - self.eligible_detail_rows

    @property
    def ineligible_parent_captures(self) -> int:
        return self.nominal_parent_captures - self.excluded_parent_captures - self.eligible_parent_captures

    def detail_counts(self) -> dict[str, int]:
        return {s.value: self.detail_status.count(s.value) for s in DetailEligibility}

    def parent_counts(self) -> dict[str, int]:
        return {s.value: self.parent_status.count(s.value) for s in DetailEligibility}

    # ---- frame access (bound)
    def check_frames(self, jobs: pd.DataFrame, cars: pd.DataFrame) -> None:
        """Raise unless ``jobs``/``cars`` are exactly the frames this population was built from."""
        if frame_binding(jobs, cars) != self.binding:
            raise PricingPopulationError("the frames differ from the evidence this population was built from")

    def parent_mask(self, jobs: pd.DataFrame, cars: pd.DataFrame) -> np.ndarray:
        self.check_frames(jobs, cars)
        return np.asarray([s == DetailEligibility.ELIGIBLE.value for s in self.parent_status], dtype=bool)

    def detail_mask(self, jobs: pd.DataFrame, cars: pd.DataFrame) -> np.ndarray:
        self.check_frames(jobs, cars)
        return np.asarray([s == DetailEligibility.ELIGIBLE.value for s in self.detail_status], dtype=bool)

    def eligible_details(self, jobs: pd.DataFrame, cars: pd.DataFrame) -> pd.DataFrame:
        """A new frame of the pricing-eligible detail rows (the source frame is unchanged)."""
        return cars.loc[self.detail_mask(jobs, cars)].copy()

    def eligible_parents(self, jobs: pd.DataFrame, cars: pd.DataFrame) -> pd.DataFrame:
        return jobs.loc[self.parent_mask(jobs, cars)].copy()


def _bool(values: object, n: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (n,) or array.dtype != bool:
        raise PricingPopulationError(f"{name} must be a boolean mask aligned with the frame")
    return array


def build_pricing_population(jobs: pd.DataFrame, cars: pd.DataFrame, *, scheduled: object, reporting_days: object,
                             rental_periods: object) -> PricingPopulation:
    """Classify every parent capture and detail row (governed exclusion first, then the first failing control).

    Args:
        scheduled: The :class:`~ql2_sixt_canada_analysis.collection_schedule.PerStreamScheduledCoverageReport`
            assessed on these frames under the available schedule (supplies the governed exclusions and the
            capture periods; a capture that is not validly assigned makes its rows ineligible, and an
            exclusion that matched no or several captures fails the whole build).
        reporting_days: :class:`~ql2_sixt_canada_analysis.temporal.DerivedReportingDays` for these frames.
        rental_periods: :class:`~ql2_sixt_canada_analysis.rental_dates.DerivedRentalPeriods` for these frames.

    Raises:
        PricingPopulationError: A required result is missing, invalid or not aligned with the frames.
    """
    from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport
    from ql2_sixt_canada_analysis.rental_dates import DerivedRentalPeriods
    from ql2_sixt_canada_analysis.schemas import DatasetKey
    from ql2_sixt_canada_analysis.temporal import DerivedReportingDays

    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be DataFrames")
    if not isinstance(scheduled, PerStreamScheduledCoverageReport) or not scheduled.schedule_assessment.available:
        raise PricingPopulationError("a per-stream scheduled-coverage report under the approved schedule is required")
    if scheduled.capture_exclusions is None or scheduled.capture_periods is None:
        raise PricingPopulationError("the scheduled-coverage report holds no resolved captures")
    if scheduled.unmatched_exclusions:
        raise PricingPopulationError("a governed exclusion matched no or several parent captures")
    if scheduled.jobs_assessed != len(jobs):
        raise PricingPopulationError("the scheduled-coverage report was assessed on other frames")
    if not isinstance(reporting_days, DerivedReportingDays):
        raise PricingPopulationError("derived reporting days are required")
    if not isinstance(rental_periods, DerivedRentalPeriods):
        raise PricingPopulationError("derived rental periods are required")
    for derived, frame, name in ((reporting_days.jobs, jobs, "reporting days"),
                                 (reporting_days.cars, cars, "reporting days"),
                                 (rental_periods.jobs, jobs, "rental periods"),
                                 (rental_periods.cars, cars, "rental periods")):
        if len(derived) != len(frame) or not derived.index.equals(frame.index):
            raise PricingPopulationError(f"{name} are not aligned with the frames")

    E = DetailEligibility

    def classify(n: int, excluded: np.ndarray, day_ok: np.ndarray, rental_ok: np.ndarray,
                 period_ok: np.ndarray) -> tuple[str, ...]:
        status = np.full(n, E.ELIGIBLE.value, dtype=object)
        status[~period_ok] = E.CAPTURE_PERIOD_UNASSIGNED.value
        status[~rental_ok] = E.RENTAL_DATES_FAILED.value
        status[~day_ok] = E.REPORTING_DAY_FAILED.value
        status[excluded] = E.GOVERNED_EXCLUSION.value
        return tuple(status.tolist())

    exclusions, periods = scheduled.capture_exclusions, scheduled.capture_periods
    parent = classify(
        len(jobs), _bool(exclusions.parent_mask(jobs), len(jobs), "parent exclusion"),
        _bool(reporting_days.eligible(DatasetKey.JOBS), len(jobs), "parent reporting day"),
        _bool(rental_periods.jobs["pricing_eligible"].to_numpy(dtype=bool), len(jobs), "parent rental"),
        periods.parent_periods(jobs).notna().to_numpy())
    detail = classify(
        len(cars), _bool(exclusions.detail_mask(cars), len(cars), "detail exclusion"),
        _bool(reporting_days.eligible(DatasetKey.CARS), len(cars), "detail reporting day"),
        _bool(rental_periods.cars["pricing_eligible"].to_numpy(dtype=bool), len(cars), "detail rental"),
        periods.detail_periods(cars).notna().to_numpy())
    return PricingPopulation(binding=frame_binding(jobs, cars), parent_status=parent, detail_status=detail)
