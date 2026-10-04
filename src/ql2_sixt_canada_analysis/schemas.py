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
against the detail rows actually present, and
:mod:`ql2_sixt_canada_analysis.relationships` validates the one-to-many
cardinality (jobs = one side, cars = many side) before any join is trusted.

Expected location coverage
--------------------------
:data:`EXPECTED_LOCATION_COVERAGE` is the single contract of which locations
the collection is *supposed* to cover. Expected locations must come from an
independent authority (a schedule, assignment or documented market scope),
never from the extract being validated - a list derived from observed rows
would always pass. Locations are branch-level pickup locations (e.g. an
airport or downtown branch of a city), which only the detail dataset carries;
jobs are city-level collection runs. An unconfigured contract fails closed
with :class:`LocationCoverageConfigurationError`.

:data:`COLLECTION_SCHEDULE` is the authoritative collection cadence used to
judge temporal completeness of a stream; it is ``None`` because no
authoritative schedule exists, so temporal completeness is reported as not
assessable rather than inferred from observed rows.
"""

from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
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
    "COLLECTION_SCHEDULE",
    "TEMPORAL_RECONCILIATION",
    "ReportingDateRule",
    "TemporalAwareness",
    "TemporalConfigurationError",
    "TemporalDateCheck",
    "TemporalFieldDefinition",
    "TemporalKind",
    "TemporalReconciliationDefinition",
    "TemporalReplicationRule",
    "TimestampOrderingRule",
    "CollectionScheduleDefinition",
    "EXPECTED_LOCATION_COVERAGE",
    "INVESTIGATED_LOCATION_STREAM",
    "KeyConfigurationError",
    "LocationCoverageConfigurationError",
    "LocationCoverageDefinition",
    "LocationCoverageMode",
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


class LocationCoverageConfigurationError(ValueError):
    """The expected-location contract is missing, invalid or cannot be applied.

    Distinct from coverage violations in the data (see
    :class:`ql2_sixt_canada_analysis.coverage.LocationCoverageError`). Messages
    never contain location values; offending column names are on ``columns``.
    """

    def __init__(self, message: str, columns: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.columns = tuple(columns)


class LocationCoverageMode(StrEnum):
    """How unexpected observed locations are treated.

    * ``EXHAUSTIVE`` - the expected set is the complete universe: every
      expected location must appear and any other observed location fails.
    * ``MINIMUM_REQUIRED`` - the expected set is a required minimum: every
      expected location must appear; extra locations are reported only.

    In both modes jobs with a missing location assignment fail the contract.
    """

    EXHAUSTIVE = "exhaustive"
    MINIMUM_REQUIRED = "minimum_required"


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
        parent_suffix, detail_suffix: Stable suffixes a validated join adds
            to same-named non-key columns from each side (never to columns
            whose names do not collide).
        definitions: Registry the columns are validated against (the project
            registry by default; tests may pass a synthetic one).
    """

    parent: DatasetKey
    detail: DatasetKey
    parent_key_columns: tuple[str, ...]
    detail_key_columns: tuple[str, ...]
    expected_detail_count_column: str
    parent_suffix: str = "_job"
    detail_suffix: str = "_detail"
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
        suffixes = (self.parent_suffix, self.detail_suffix)
        if not all(isinstance(x, str) and x for x in suffixes) or len(set(suffixes)) != 2:
            raise RelationshipConfigurationError("join suffixes must be distinct non-empty strings")
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


@dataclass(frozen=True, slots=True)
class LocationCoverageDefinition:
    """Immutable expected-location contract for one dataset.

    Attributes:
        dataset: Logical dataset whose rows carry the scheduled location.
        location_columns: Ordered source columns forming one location key.
        expected_locations: Authoritative expected location keys, each a tuple
            with one component per location column, or ``None`` when no
            authoritative list has been supplied (the contract is then
            unconfigured and assessment fails closed). Never derived from data.
        mode: :class:`LocationCoverageMode`, or ``None`` while unconfigured.
            Expected locations and mode are configured together.
        aliases: Authoritative, explicit alias keys per expected key (read-only
            mapping ``expected key -> tuple of alias keys``). Empty unless an
            authority confirms an alias; never built from string similarity.
            Aliases are matched exactly and never rewrite source values.
        stream_scope_columns: Columns of the same dataset that identify the
            collection scope a location belongs to (e.g. its city). Used only
            by stream investigation to group collection events; they add
            nothing to the expectations.
        definitions: Registry the columns are validated against (the project
            registry by default; tests may pass a synthetic one).

    Comparison policy: **exact**. Components are compared as given -
    case-sensitive, no stripping, punctuation, alias, abbreviation or fuzzy
    handling - and composite keys are compared as tuples, never as
    concatenated strings. A missing, empty or whitespace-only observed
    component makes that row an *unassigned* location; source values are
    never modified.
    """

    dataset: DatasetKey
    location_columns: tuple[str, ...]
    expected_locations: tuple[tuple[str, ...], ...] | None = None
    mode: LocationCoverageMode | None = None
    aliases: Mapping[tuple[str, ...], tuple[tuple[str, ...], ...]] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )
    stream_scope_columns: tuple[str, ...] = ()
    definitions: Mapping[DatasetKey, DatasetDefinition] = dataclass_field(
        default=None, compare=False, repr=False  # type: ignore[arg-type]
    )

    def __post_init__(self) -> None:
        if self.definitions is None:
            object.__setattr__(self, "definitions", DATASET_DEFINITIONS)
        if self.dataset not in self.definitions:
            raise LocationCoverageConfigurationError("coverage dataset must be registered")
        scope = self.stream_scope_columns
        if not isinstance(scope, tuple) or not all(isinstance(c, str) and c for c in scope) \
                or len(set(scope)) != len(scope):
            raise LocationCoverageConfigurationError("stream_scope_columns must be a tuple of unique names")
        unknown_scope = tuple(c for c in scope if c not in self.definitions[self.dataset].columns)
        if unknown_scope:
            raise LocationCoverageConfigurationError(
                f"{len(unknown_scope)} scope column(s) are not in the '{self.dataset}' contract", unknown_scope
            )
        columns = self.location_columns
        if not isinstance(columns, tuple) or not columns:
            raise LocationCoverageConfigurationError("location_columns must be a non-empty tuple")
        if not all(isinstance(c, str) and c for c in columns) or len(set(columns)) != len(columns):
            raise LocationCoverageConfigurationError("location_columns must be unique non-empty strings")
        unknown = tuple(c for c in columns if c not in self.source_definition.columns)
        if unknown:
            raise LocationCoverageConfigurationError(
                f"{len(unknown)} location column(s) are not in the '{self.dataset}' contract", unknown
            )
        if (self.expected_locations is None) != (self.mode is None):
            raise LocationCoverageConfigurationError(
                "expected_locations and mode must be configured together"
            )
        if not isinstance(self.aliases, Mapping):
            raise LocationCoverageConfigurationError("aliases must be a mapping")
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))
        if self.expected_locations is None:
            if self.aliases:
                raise LocationCoverageConfigurationError("aliases require configured expected locations")
            return
        if not isinstance(self.mode, LocationCoverageMode):
            raise LocationCoverageConfigurationError("mode must be a LocationCoverageMode")
        expected = self.expected_locations
        if not isinstance(expected, tuple) or not expected:
            raise LocationCoverageConfigurationError("expected_locations must be a non-empty tuple")
        for key in expected:
            if not isinstance(key, tuple) or len(key) != len(columns):
                raise LocationCoverageConfigurationError(
                    "each expected location must be a tuple with one component per location column"
                )
            if not all(isinstance(v, str) and v.strip() for v in key):
                raise LocationCoverageConfigurationError(
                    "expected location components must be non-missing, non-blank strings"
                )
        if len(set(expected)) != len(expected):
            raise LocationCoverageConfigurationError("expected_locations must not contain duplicates")
        seen: set[tuple[str, ...]] = set()
        for key, alias_keys in self.aliases.items():
            if key not in expected:
                raise LocationCoverageConfigurationError("aliases may only be given for expected locations")
            if not isinstance(alias_keys, tuple) or not alias_keys:
                raise LocationCoverageConfigurationError("alias keys must be a non-empty tuple")
            for alias in alias_keys:
                if (not isinstance(alias, tuple) or len(alias) != len(columns)
                        or not all(isinstance(v, str) and v.strip() for v in alias)):
                    raise LocationCoverageConfigurationError("each alias must be a well-formed location key")
                if alias in expected or alias in seen:
                    raise LocationCoverageConfigurationError(
                        "an alias must not be an expected key or shared between expected keys"
                    )
                seen.add(alias)

    def match_keys(self, target: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        """The target key followed by its authoritative aliases (exact keys only)."""
        return (target, *self.aliases.get(target, ()))

    @property
    def is_configured(self) -> bool:
        """True once an authoritative expected set and mode are supplied."""
        return self.expected_locations is not None

    @property
    def source_definition(self) -> DatasetDefinition:
        return self.definitions[self.dataset]


#: An expected branch-level location stream, identified as expected by the
#: project owner (the authority for this entry). Defined once here; code,
#: notebooks and tests refer to this constant, never to the literal.
INVESTIGATED_LOCATION_STREAM: Final[tuple[str, ...]] = ('Calgary Downtown',)

#: The expected-location contract. Locations are branch-level pickup
#: locations, carried only by detail rows (``location``); jobs are city-level
#: collection runs, so a jobs-level contract cannot represent a branch stream
#: (an earlier jobs-``city`` structure would have reported a permanent false
#: absence). Expected keys come only from an authority - here the project
#: owner's statement for the one stream above - so the set is a required
#: MINIMUM, not an exhaustive universe. Add further locations only from an
#: authoritative list, never from the observed extract. No aliases are
#: authoritatively confirmed. ``city`` scopes a branch to its collection runs
#: for stream-continuity investigation only.
EXPECTED_LOCATION_COVERAGE: Final = LocationCoverageDefinition(
    dataset=DatasetKey.CARS,
    location_columns=('location',),
    expected_locations=(INVESTIGATED_LOCATION_STREAM,),
    mode=LocationCoverageMode.MINIMUM_REQUIRED,
    stream_scope_columns=('city',),
)


@dataclass(frozen=True, slots=True)
class CollectionScheduleDefinition:
    """Authoritative collection cadence for temporal-completeness checks.

    Attributes:
        dataset: Dataset whose ``timestamp_column`` dates each collection.
        timestamp_column: Column holding ISO-8601 collection timestamps.
        expected_periods: Authoritative ISO-8601 period starts (UTC) that must
            each contain the stream. Never inferred from observed rows.
        period: pandas offset alias used to floor timestamps (e.g. ``"h"``).
        definitions: Registry used for validation (tests may pass their own).
    """

    dataset: DatasetKey
    timestamp_column: str
    expected_periods: tuple[str, ...]
    period: str
    definitions: Mapping[DatasetKey, DatasetDefinition] = dataclass_field(
        default=None, compare=False, repr=False  # type: ignore[arg-type]
    )

    def __post_init__(self) -> None:
        if self.definitions is None:
            object.__setattr__(self, "definitions", DATASET_DEFINITIONS)
        if self.dataset not in self.definitions:
            raise LocationCoverageConfigurationError("schedule dataset must be registered")
        if self.timestamp_column not in self.definitions[self.dataset].columns:
            raise LocationCoverageConfigurationError("schedule timestamp column is not in the contract")
        periods = self.expected_periods
        if not isinstance(periods, tuple) or not periods or not all(isinstance(p, str) and p for p in periods):
            raise LocationCoverageConfigurationError("expected_periods must be a non-empty tuple of strings")
        try:
            parsed = pd.to_datetime(list(periods), utc=True, format="ISO8601")
            pd.tseries.frequencies.to_offset(self.period)
        except (ValueError, TypeError) as exc:
            raise LocationCoverageConfigurationError("schedule periods or period alias are invalid") from exc
        if parsed.has_duplicates:
            raise LocationCoverageConfigurationError("expected_periods must not contain duplicates")


#: No authoritative collection schedule exists in the repository or project
#: documentation, so temporal completeness cannot be proven. Do not infer one
#: from observed rows.
COLLECTION_SCHEDULE: Final[CollectionScheduleDefinition | None] = None


def get_dataset_definition(key: DatasetKey | str) -> DatasetDefinition:
    """Return the definition for ``key`` (a :class:`DatasetKey` or its value).

    Raises:
        KeyError: ``key`` is not a known logical dataset.
    """
    try:
        return DATASET_DEFINITIONS[DatasetKey(key)]
    except ValueError as exc:
        raise KeyError(f"Unknown dataset key: {key!r}") from exc


# ------------------------------------------------------------------ temporal


class TemporalConfigurationError(ValueError):
    """A temporal definition is invalid, incomplete or cannot be applied."""


class TemporalKind(StrEnum):
    """Whether a field holds an instant/wall time or a calendar date."""

    TIMESTAMP = "timestamp"
    DATE = "date"


class TemporalAwareness(StrEnum):
    """How a timestamp field states its time zone in the source text.

    * ``NAIVE`` - no zone information; resolvable to an instant only through
      an authoritative ``source_timezone`` (never the machine's zone).
    * ``OFFSET`` - each value carries a numeric UTC offset or ``Z``.
    * ``DESIGNATOR`` - each value ends with a zone abbreviation that is
      mapped to a fixed UTC offset by ``designator_offsets``.
    * ``NOT_APPLICABLE`` - calendar dates.
    """

    NAIVE = "naive"
    OFFSET = "offset"
    DESIGNATOR = "designator"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class TemporalFieldDefinition:
    """One source temporal field and how to parse it.

    Attributes:
        dataset: Logical dataset holding the column.
        column: Source column name.
        kind: Timestamp or calendar date.
        required: Whether a missing value is a completeness failure.
        source_format: ``strftime``-style format of the value *without* any
            offset or designator, or ``"ISO8601"`` (offset-aware fields).
        awareness: See :class:`TemporalAwareness`.
        source_timezone: Authoritative IANA zone for ``NAIVE`` values, or
            ``None`` when no authority exists (values are then unresolved for
            instant comparisons, never guessed).
        designator_offsets: For ``DESIGNATOR`` fields, the authoritative
            fixed UTC offset of each accepted abbreviation.
    """

    dataset: DatasetKey
    column: str
    kind: TemporalKind
    required: bool
    source_format: str
    awareness: TemporalAwareness
    source_timezone: str | None = None
    designator_offsets: Mapping[str, dt.timedelta] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TemporalKind) or not isinstance(self.awareness, TemporalAwareness):
            raise TemporalConfigurationError("kind and awareness must be enum members")
        if not isinstance(self.column, str) or not self.column:
            raise TemporalConfigurationError("column must be a non-empty string")
        if not isinstance(self.source_format, str) or not self.source_format:
            raise TemporalConfigurationError("source_format must be a non-empty string")
        if (self.kind is TemporalKind.DATE) != (self.awareness is TemporalAwareness.NOT_APPLICABLE):
            raise TemporalConfigurationError("dates (and only dates) have no time-zone awareness")
        if self.source_timezone is not None:
            if self.awareness is not TemporalAwareness.NAIVE:
                raise TemporalConfigurationError("source_timezone applies to naive timestamps only")
            _zone(self.source_timezone)
        offsets = dict(self.designator_offsets)
        if (self.awareness is TemporalAwareness.DESIGNATOR) != bool(offsets):
            raise TemporalConfigurationError("designator offsets are required for (and only for) DESIGNATOR fields")
        for name, offset in offsets.items():
            if not (isinstance(name, str) and name.isalpha() and name.isupper()):
                raise TemporalConfigurationError("designators must be upper-case letters")
            if not isinstance(offset, dt.timedelta) or abs(offset) > dt.timedelta(hours=14):
                raise TemporalConfigurationError("designator offsets must be timedeltas within +/-14h")
        object.__setattr__(self, "designator_offsets", MappingProxyType(offsets))

    @property
    def ref(self) -> tuple[DatasetKey, str]:
        return (self.dataset, self.column)

    @property
    def resolvable_to_instant(self) -> bool:
        """True when values can become absolute instants without guessing."""
        if self.kind is not TemporalKind.TIMESTAMP:
            return False
        return self.awareness is not TemporalAwareness.NAIVE or self.source_timezone is not None


@dataclass(frozen=True, slots=True)
class TimestampOrderingRule:
    """``earlier`` must not be after ``later`` (per linked detail row).

    ``later - earlier`` must be ``>= -tolerance`` (inclusive) or
    ``> -tolerance`` (exclusive). ``tolerance`` must be explicitly authorised,
    is non-negative and is never derived from observed data.
    """

    earlier: tuple[DatasetKey, str]
    later: tuple[DatasetKey, str]
    inclusive: bool = True
    tolerance: dt.timedelta = dt.timedelta(0)

    def __post_init__(self) -> None:
        if not isinstance(self.tolerance, dt.timedelta) or self.tolerance < dt.timedelta(0):
            raise TemporalConfigurationError("tolerance must be a non-negative timedelta")
        if self.earlier == self.later:
            raise TemporalConfigurationError("ordering compares two different fields")


@dataclass(frozen=True, slots=True)
class ReportingDateRule:
    """The date equals the calendar date of ``source`` in ``reporting_timezone``.

    The source instant is converted to the reporting zone *before* its date is
    taken. Dates are compared semantically (parsed), not as strings.
    """

    source: tuple[DatasetKey, str]
    reporting_timezone: str

    def __post_init__(self) -> None:
        _zone(self.reporting_timezone)


@dataclass(frozen=True, slots=True)
class TemporalDateCheck:
    """A date field and its derivation rule; ``rule=None`` means unavailable.

    An unavailable rule is reported and makes strict validation fail closed.
    """

    target: tuple[DatasetKey, str]
    rule: ReportingDateRule | None


@dataclass(frozen=True, slots=True)
class TemporalReplicationRule:
    """``replica`` (a copy carried on detail rows) must equal ``source`` on its parent.

    Instants are compared when both sides resolve; two naive fields with the
    same time-zone basis are compared as wall times.
    """

    source: tuple[DatasetKey, str]
    replica: tuple[DatasetKey, str]


@dataclass(frozen=True, slots=True)
class TemporalReconciliationDefinition:
    """The single temporal contract for the jobs/cars datasets.

    Attributes:
        fields: Every temporal field definition.
        canonical_timezone: Zone in which instants are compared (``UTC``).
        ordering: The ordering rule, or ``None`` when no authority defines it
            (reported as unavailable; strict validation fails closed).
        date_checks: One entry per date field with its derivation rule or
            ``None`` (unavailable).
        replications: Detail-row copies of parent temporal fields.
        relationship: Parent/detail relationship used to link rows.
    """

    fields: tuple[TemporalFieldDefinition, ...]
    canonical_timezone: str
    ordering: TimestampOrderingRule | None
    date_checks: tuple[TemporalDateCheck, ...]
    replications: tuple[TemporalReplicationRule, ...]
    relationship: JobDetailRelationshipDefinition

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple) or not self.fields:
            raise TemporalConfigurationError("fields must be a non-empty tuple")
        refs = [f.ref for f in self.fields]
        if len(set(refs)) != len(refs):
            raise TemporalConfigurationError("each field may be defined once")
        registry = self.relationship.definitions
        for field in self.fields:
            if field.dataset not in (self.relationship.parent, self.relationship.detail):
                raise TemporalConfigurationError("temporal fields must belong to the relationship datasets")
            if field.column not in registry[field.dataset].columns:
                raise TemporalConfigurationError(
                    f"a '{field.dataset}' temporal column is not in its contract"
                )
        _zone(self.canonical_timezone)
        if self.ordering is not None:
            for ref in (self.ordering.earlier, self.ordering.later):
                if self.field(ref).kind is not TemporalKind.TIMESTAMP:
                    raise TemporalConfigurationError("ordering compares timestamp fields")
        targets = [c.target for c in self.date_checks]
        if len(set(targets)) != len(targets):
            raise TemporalConfigurationError("each date field may be checked once")
        for check in self.date_checks:
            if self.field(check.target).kind is not TemporalKind.DATE:
                raise TemporalConfigurationError("date checks target date fields")
            if check.rule is not None and self.field(check.rule.source).kind is not TemporalKind.TIMESTAMP:
                raise TemporalConfigurationError("reporting dates derive from timestamp fields")
        for rule in self.replications:
            source, replica = self.field(rule.source), self.field(rule.replica)
            if (source.dataset, replica.dataset) != (self.relationship.parent, self.relationship.detail):
                raise TemporalConfigurationError("replicas are detail-row copies of parent fields")
            if source.kind is not replica.kind:
                raise TemporalConfigurationError("a replica has the same temporal kind as its source")

    def field(self, ref: tuple[DatasetKey, str]) -> TemporalFieldDefinition:
        for candidate in self.fields:
            if candidate.ref == tuple(ref):
                return candidate
        raise TemporalConfigurationError("a rule refers to an undefined temporal field")

    @property
    def unavailable_rules(self) -> tuple[str, ...]:
        """Safe names of required rules that lack authoritative semantics."""
        names = [] if self.ordering is not None else ["timestamp_ordering"]
        names += [f"date_derivation:{c.target[0]}.{c.target[1]}" for c in self.date_checks if c.rule is None]
        return tuple(names)


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise TemporalConfigurationError("unknown IANA time zone") from exc


_ISO_DATE: Final = "%Y-%m-%d"

#: The temporal contract. Established facts (source formats and provenance):
#:
#: * ``finished_at`` (jobs) - when the collection job finished; a *naive*
#:   timestamp. No authoritative time zone is documented, so it is NOT
#:   resolved to an instant (never assumed UTC or machine-local).
#: * ``job_finished_at`` (cars) - the parent job's ``finished_at`` repeated on
#:   each detail row (``job_*`` columns repeat parent-job attributes), so it
#:   must equal its parent's value (replication rule).
#: * ``scraped_at`` (cars) - when each detail row was scraped; every value
#:   ends with the designator ``MST``. Per the IANA/POSIX definition ``MST``
#:   is fixed UTC-07:00 (no daylight time). It labels the collector's clock -
#:   it appears year-round and for markets in other zones - so it is *not* a
#:   market-local time. If the source meant daylight-adjusted Mountain time,
#:   this mapping must be corrected by the data owner.
#: * ``scrape_date`` (jobs and cars) and ``date_clean`` (cars) - ISO calendar
#:   dates supplied by the source (not generated by repository code).
#:
#: Not established (no repository documentation or source authority), hence
#: unavailable and failing closed: the ordering between ``finished_at`` and
#: ``scraped_at`` (and a time zone for ``finished_at``), which timestamp and
#: reporting time zone define ``scrape_date``, and which define ``date_clean``.
#: The markets span several time zones and no authoritative location-to-zone
#: mapping exists. No tolerance is authorised.
TEMPORAL_RECONCILIATION: Final = TemporalReconciliationDefinition(
    fields=(
        TemporalFieldDefinition(DatasetKey.JOBS, 'finished_at', TemporalKind.TIMESTAMP, True,
                                "%Y-%m-%d %H:%M:%S.%f", TemporalAwareness.NAIVE),
        TemporalFieldDefinition(DatasetKey.JOBS, 'scrape_date', TemporalKind.DATE, True,
                                _ISO_DATE, TemporalAwareness.NOT_APPLICABLE),
        TemporalFieldDefinition(DatasetKey.CARS, 'job_finished_at', TemporalKind.TIMESTAMP, True,
                                "%Y-%m-%d %H:%M:%S.%f", TemporalAwareness.NAIVE),
        TemporalFieldDefinition(DatasetKey.CARS, 'scraped_at', TemporalKind.TIMESTAMP, True,
                                "%Y-%m-%d %H:%M:%S", TemporalAwareness.DESIGNATOR,
                                designator_offsets={"MST": dt.timedelta(hours=-7)}),
        TemporalFieldDefinition(DatasetKey.CARS, 'scrape_date', TemporalKind.DATE, True,
                                _ISO_DATE, TemporalAwareness.NOT_APPLICABLE),
        TemporalFieldDefinition(DatasetKey.CARS, 'date_clean', TemporalKind.DATE, True,
                                _ISO_DATE, TemporalAwareness.NOT_APPLICABLE),
    ),
    canonical_timezone="UTC",
    ordering=None,
    date_checks=(
        TemporalDateCheck((DatasetKey.JOBS, 'scrape_date'), None),
        TemporalDateCheck((DatasetKey.CARS, 'scrape_date'), None),
        TemporalDateCheck((DatasetKey.CARS, 'date_clean'), None),
    ),
    replications=(
        TemporalReplicationRule((DatasetKey.JOBS, 'finished_at'), (DatasetKey.CARS, 'job_finished_at')),
    ),
    relationship=JOB_DETAIL_RELATIONSHIP,
)
