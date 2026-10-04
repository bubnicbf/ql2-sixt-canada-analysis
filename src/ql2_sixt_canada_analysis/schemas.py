"""Authoritative logical dataset definitions for the raw source files.

Each :class:`DatasetDefinition` records the stable logical key of a raw
dataset, the filename tokens used to discover its file, and the ordered
column contract of its CSV header. Column names are copied exactly from the
source headers and describe structure only: no renaming, types, nullability
or key constraints are implied, and the contract does not make the data
safe to publish.

The column contract is order-sensitive: a raw file's header must equal
``columns`` exactly (same names, same order, no extras, no duplicates).

Identifier columns
------------------
``identifier_columns`` names, in contract order, the columns that are
*labels of an entity's identity* rather than measurements. They are read and
represented as :data:`IDENTIFIER_DTYPE` (pandas nullable string, missing
values stay ``pd.NA``) so numeric inference can never drop leading zeros,
round long integers or turn missing values into text. Every other column keeps
normal pandas inference. Identifier content is never normalised here.
These definitions are the only identifier registry; ingestion, quality
helpers, notebooks and tests all read them from here.

Unique keys and row grain
-------------------------
``unique_key_columns`` is the dataset's business-key contract: the smallest
ordered set of columns that identifies one row of the documented grain. A
valid key is *complete* (no missing component) and *unique* (no two rows share
the full component tuple); :mod:`ql2_sixt_canada_analysis.unique_keys`
measures both without changing data. Components should be identifier columns;
any other component is listed by ``non_identifier_key_columns`` and must be
justified next to the definition. Keys are chosen from the grain's semantics,
never from whatever happens to be unique in one extract.

Job-to-detail relationship
--------------------------
:data:`JOB_DETAIL_RELATIONSHIP` is the single definition of how detail rows
(``cars``) belong to a parent job (``jobs``): the parent key (the jobs unique
key), the matching detail foreign-key components in the same order, and the
parent column that declares how many detail rows the job should have.
:mod:`ql2_sixt_canada_analysis.reconciliation` reconciles that declaration
against the detail rows actually present.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import pandas as pd

__all__ = [
    "CARS_DEFINITION",
    "DATASET_DEFINITIONS",
    "IDENTIFIER_DTYPE",
    "JOBS_DEFINITION",
    "JOB_DETAIL_RELATIONSHIP",
    "JobDetailRelationshipDefinition",
    "KeyConfigurationError",
    "RelationshipConfigurationError",
    "SHARED_IDENTIFIER_COLUMNS",
    "DatasetDefinition",
    "DatasetKey",
    "get_dataset_definition",
]


#: The one dtype used for every identifier column in every dataset: pandas'
#: nullable string dtype (``"string"``), whose missing value is ``pd.NA``.
#: Shared logical identifiers therefore have identical types in both datasets.
IDENTIFIER_DTYPE: Final[pd.StringDtype] = pd.StringDtype()


class KeyConfigurationError(ValueError):
    """A unique-key definition is invalid or cannot be applied to a frame.

    Distinct from data-quality violations (see
    :class:`ql2_sixt_canada_analysis.unique_keys.UniqueKeyViolationError`).
    Messages carry the dataset key and counts only; column names are on
    ``columns``.
    """

    def __init__(self, message: str, key: object = None, columns: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.role = key
        self.columns = tuple(columns)


class RelationshipConfigurationError(ValueError):
    """A job-to-detail relationship definition is invalid or cannot be applied.

    Messages carry dataset keys and counts only; column names are on
    ``columns``.
    """

    def __init__(self, message: str, columns: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.columns = tuple(columns)


class DatasetKey(StrEnum):
    """Stable logical identifiers for the raw datasets."""

    JOBS = "jobs"
    CARS = "cars"


@dataclass(frozen=True, slots=True)
class DatasetDefinition:
    """Immutable description of one raw source dataset.

    Attributes:
        key: Logical dataset identifier.
        filename_tokens: Lowercase words that identify this dataset's file.
            Discovery assigns a CSV to the dataset whose token appears last in
            the filename (see :mod:`ql2_sixt_canada_analysis.ingestion`).
        columns: Exact, ordered CSV header expected for this dataset.
        identifier_columns: Columns (a subset of ``columns``, in the same
            relative order) holding identity labels; loaded as
            :data:`IDENTIFIER_DTYPE`. Empty means the dataset has none.
        unique_key_columns: Ordered business-key components for one row of
            the dataset's grain (see the module docstring). Empty means no
            key contract is declared; assessment then raises
            :class:`KeyConfigurationError`.
    """

    key: DatasetKey
    filename_tokens: tuple[str, ...]
    columns: tuple[str, ...]
    identifier_columns: tuple[str, ...] = ()
    unique_key_columns: tuple[str, ...] = ()

    @property
    def non_identifier_key_columns(self) -> tuple[str, ...]:
        """Key components that are not identifier columns (documented exceptions)."""
        return tuple(c for c in self.unique_key_columns if c not in self.identifier_columns)

    @property
    def identifier_dtypes(self) -> Mapping[str, pd.StringDtype]:
        """Read-only ``{identifier column: IDENTIFIER_DTYPE}`` in contract order."""
        return MappingProxyType({column: IDENTIFIER_DTYPE for column in self.identifier_columns})

    def __post_init__(self) -> None:
        for field_name in ("filename_tokens", "columns"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{self.key}: {field_name} must be a non-empty tuple")
            if not all(isinstance(v, str) and v for v in values):
                raise ValueError(f"{self.key}: {field_name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{self.key}: {field_name} must not contain duplicates")
        if not all(t.isalnum() and t == t.lower() for t in self.filename_tokens):
            raise ValueError(f"{self.key}: filename tokens must be lowercase alphanumeric")
        identifiers = self.identifier_columns
        if not isinstance(identifiers, tuple):
            raise ValueError(f"{self.key}: identifier_columns must be a tuple")
        if not all(isinstance(v, str) and v for v in identifiers):
            raise ValueError(f"{self.key}: identifier_columns must contain non-empty strings")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{self.key}: identifier_columns must not contain duplicates")
        if not set(identifiers) <= set(self.columns):
            raise ValueError(f"{self.key}: identifier_columns must be listed in columns")
        if identifiers != tuple(c for c in self.columns if c in identifiers):
            raise ValueError(f"{self.key}: identifier_columns must follow column order")
        key = self.unique_key_columns
        if not isinstance(key, tuple):
            raise KeyConfigurationError(f"{self.key}: unique_key_columns must be a tuple", self.key)
        if not all(isinstance(v, str) and v for v in key):
            raise KeyConfigurationError(
                f"{self.key}: unique_key_columns must contain non-empty strings", self.key
            )
        if len(set(key)) != len(key):
            raise KeyConfigurationError(
                f"{self.key}: unique_key_columns must not contain duplicates", self.key,
                tuple(dict.fromkeys(c for c in key if key.count(c) > 1)),
            )
        unknown = tuple(c for c in key if c not in self.columns)
        if unknown:
            raise KeyConfigurationError(
                f"{self.key}: {len(unknown)} unique_key_columns not listed in columns",
                self.key, unknown,
            )


JOBS_DEFINITION: Final = DatasetDefinition(
    key=DatasetKey.JOBS,
    filename_tokens=("jobs",),
    columns=(
        'job_id',
        'city',
        'mode',
        'status',
        'record_count',
        'pickup_date',
        'return_date',
        'finished_at',
        'scrape_date',
        'actual_car_rows',
    ),
    # Scrape-job identity; referenced by every cars row (shared key).
    identifier_columns=('job_id',),
    # Grain: one row per scrape (collection) job. The job identifier alone
    # identifies that row; no other column is needed.
    unique_key_columns=('job_id',),
)

CARS_DEFINITION: Final = DatasetDefinition(
    key=DatasetKey.CARS,
    filename_tokens=("cars",),
    columns=(
        'job_id',
        'city',
        'mode',
        'status',
        'job_finished_at',
        'scrape_date',
        'job_pickup_date',
        'job_return_date',
        'row_index',
        'pickup_date',
        'return_date',
        'car_name',
        'car_type',
        'price_per_day',
        'transmission',
        'seats',
        'bags',
        'location',
        'scraped_at',
        'price_num',
        'city_clean',
        'date_clean',
    ),
    # Classification notes (structure only, no source values):
    # * job_id: the parent scrape job's identity, the logical link to jobs.
    #   In this export it can be serialised upstream in a different textual
    #   form than on the jobs side (e.g. a float-style suffix). Reading it as a
    #   string preserves that text verbatim; reconciling the two forms is a
    #   separate, explicit normalisation step, not part of typing.
    # * row_index: deliberately NOT an identifier. It is the ordinal position
    #   of an offer within its job's result list, so it carries order/rank
    #   meaning for assortment analysis and keeps numeric inference.
    # * city/mode/status/location/car_name/car_type and *_clean: descriptive
    #   attributes (names, categories), not identities; already text and left
    #   to normal inference. Dates, timestamps, prices, seats and bags are
    #   measures or calendar values and must never be cast as identifiers.
    identifier_columns=('job_id',),
    # Grain: one row per offer position within one scrape job's result list.
    # A job returns many offers, so the job identifier alone cannot identify a
    # row, and the ordinal restarts in every job, so it cannot either; the
    # pair (parent job identifier, ordinal position) is the smallest key for
    # this grain. row_index is the one documented non-identifier component:
    # it is a positional locator kept numeric for rank analysis (see above),
    # so it appears in non_identifier_key_columns. Offer attributes (vehicle,
    # price, location, timestamps) are deliberately excluded: they describe
    # the offer and may legitimately repeat within a job.
    unique_key_columns=('job_id', 'row_index'),
)

DATASET_DEFINITIONS: Final[Mapping[DatasetKey, DatasetDefinition]] = MappingProxyType(
    {definition.key: definition for definition in (JOBS_DEFINITION, CARS_DEFINITION)}
)


#: Identifier columns present in both datasets (jobs order). Both datasets give
#: them :data:`IDENTIFIER_DTYPE`, so their types always match.
SHARED_IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in JOBS_DEFINITION.identifier_columns
    if column in CARS_DEFINITION.identifier_columns
)


@dataclass(frozen=True, slots=True)
class JobDetailRelationshipDefinition:
    """Immutable parent-to-detail relationship used for count reconciliation.

    Attributes:
        parent: Logical dataset holding one row per parent (job).
        detail: Logical dataset holding the detail rows.
        parent_key_columns: Parent key components; must equal the parent's
            ``unique_key_columns`` so each detail key matches at most one job.
        detail_key_columns: Detail foreign-key components, positionally
            matching ``parent_key_columns``.
        expected_detail_count_column: Parent column declaring how many detail
            rows the parent should have.
        definitions: Registry the columns are validated against (the project
            registry by default; tests may pass a synthetic one).
    """

    parent: DatasetKey
    detail: DatasetKey
    parent_key_columns: tuple[str, ...]
    detail_key_columns: tuple[str, ...]
    expected_detail_count_column: str
    definitions: Mapping[DatasetKey, DatasetDefinition] = dataclass_field(
        default=None, compare=False, repr=False  # type: ignore[arg-type]
    )

    def __post_init__(self) -> None:
        if self.definitions is None:
            object.__setattr__(self, "definitions", DATASET_DEFINITIONS)
        if self.parent not in self.definitions or self.detail not in self.definitions:
            raise RelationshipConfigurationError("relationship datasets must be registered")
        if self.parent == self.detail:
            raise RelationshipConfigurationError("parent and detail datasets must differ")
        for name in ("parent_key_columns", "detail_key_columns"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value:
                raise RelationshipConfigurationError(f"{name} must be a non-empty tuple")
            if not all(isinstance(v, str) and v for v in value):
                raise RelationshipConfigurationError(f"{name} must contain non-empty strings")
            if len(set(value)) != len(value):
                raise RelationshipConfigurationError(f"{name} must not contain duplicates")
        if len(self.parent_key_columns) != len(self.detail_key_columns):
            raise RelationshipConfigurationError("parent and detail keys must have equal length")
        parent, detail = self.parent_definition, self.detail_definition
        for columns, definition in ((self.parent_key_columns, parent), (self.detail_key_columns, detail)):
            unknown = tuple(c for c in columns if c not in definition.columns)
            if unknown:
                raise RelationshipConfigurationError(
                    f"{len(unknown)} '{definition.key}' relationship column(s) not in its contract",
                    unknown,
                )
            untyped = tuple(c for c in columns if c not in definition.identifier_columns)
            if untyped:  # identifier dtype on both sides guarantees compatible types
                raise RelationshipConfigurationError(
                    f"{len(untyped)} '{definition.key}' relationship column(s) are not identifiers",
                    untyped,
                )
        if self.parent_key_columns != parent.unique_key_columns:
            raise RelationshipConfigurationError(
                f"parent key must equal the '{parent.key}' unique key", self.parent_key_columns
            )
        count = self.expected_detail_count_column
        if not isinstance(count, str) or count not in parent.columns:
            raise RelationshipConfigurationError(
                f"expected-count column must be a '{parent.key}' column", (str(count),)
            )
        if count in parent.identifier_columns or count in self.parent_key_columns:
            raise RelationshipConfigurationError(
                "expected-count column must be a measure, not a key or identifier", (count,)
            )

    @property
    def parent_definition(self) -> DatasetDefinition:
        return self.definitions[self.parent]

    @property
    def detail_definition(self) -> DatasetDefinition:
        return self.definitions[self.detail]


#: The jobs -> cars relationship. Each cars row carries its parent job's
#: identifier. Evidence: matching identifier semantics on both sides and the
#: cars contract's ``job_*`` columns, which repeat parent-job attributes.
#: Expected count: ``record_count`` is the number of offer records the job
#: itself reports, i.e. the declaration to reconcile. ``actual_car_rows`` is
#: a downstream tally of rows written, not a declaration, so it is not used
#: here (comparing the two fields would be a separate consistency control).
#: Identifiers are compared verbatim, so the textual-form difference noted on
#: the cars identifier above is reported as orphans/under-counts until an
#: explicit normalisation step reconciles the two forms.
JOB_DETAIL_RELATIONSHIP: Final = JobDetailRelationshipDefinition(
    parent=DatasetKey.JOBS,
    detail=DatasetKey.CARS,
    parent_key_columns=('job_id',),
    detail_key_columns=('job_id',),
    expected_detail_count_column='record_count',
)


def get_dataset_definition(key: DatasetKey | str) -> DatasetDefinition:
    """Return the definition for ``key`` (a :class:`DatasetKey` or its value).

    Raises:
        KeyError: ``key`` is not a known logical dataset.
    """
    try:
        return DATASET_DEFINITIONS[DatasetKey(key)]
    except ValueError as exc:
        raise KeyError(f"Unknown dataset key: {key!r}") from exc
