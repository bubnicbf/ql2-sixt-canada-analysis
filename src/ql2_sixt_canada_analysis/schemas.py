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
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import pandas as pd

__all__ = [
    "VANCOUVER_LOCATION_POLICY",
    "LocationIdentityPolicy",
    "LocationPolicyAuthority",
    "LocationPolicyConfigurationError",
    "LocationPolicyState",
    "AttributeComparisonPolicy",
    "MissingValueStabilityPolicy",
    "VEHICLE_ATTRIBUTE_STABILITY",
    "VehicleAttributeDefinition",
    "VehicleStabilityConfigurationError",
    "VehicleStabilityDefinition",
    "CARS_DEFINITION",
    "DATASET_DEFINITIONS",
    "IDENTIFIER_DTYPE",
    "JOBS_DEFINITION",
    "JOB_DETAIL_RELATIONSHIP",
    "JobDetailRelationshipDefinition",
    "COLLECTION_SCHEDULE",
    "COMPARED_LOCATION_STREAMS",
    "LOCATION_STREAM_COMPARISON",
    "CapturePairing",
    "LocationStreamComparisonDefinition",
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

#: Two expected branch-level streams the project owner asked to compare
#: (authority for their expectation). Defined once; referenced by constant.
COMPARED_LOCATION_STREAMS: Final[tuple[tuple[str, ...], tuple[str, ...]]] = (
    ('Vancouver Downtown',),
    ('Vancouver Thurlow',),
)

#: The expected-location contract. Locations are branch-level pickup
#: locations, carried only by detail rows (``location``); jobs are city-level
#: collection runs, so a jobs-level contract cannot represent a branch stream
#: (an earlier jobs-``city`` structure would have reported a permanent false
#: absence). Expected keys come only from an authority - here the project
#: owner's statements for the streams above - so the set is a required
#: MINIMUM, not an exhaustive universe. Add further locations only from an
#: authoritative list, never from the observed extract. No aliases are
#: authoritatively confirmed. ``city`` scopes a branch to its collection runs
#: for stream-continuity investigation only.
EXPECTED_LOCATION_COVERAGE: Final = LocationCoverageDefinition(
    dataset=DatasetKey.CARS,
    location_columns=('location',),
    expected_locations=(INVESTIGATED_LOCATION_STREAM, *COMPARED_LOCATION_STREAMS),
    mode=LocationCoverageMode.MINIMUM_REQUIRED,
    stream_scope_columns=('city',),
)


#: An ISO-8601 instant ends with ``Z`` or an explicit UTC offset.
_EXPLICIT_OFFSET: Final = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


@dataclass(frozen=True, slots=True)
class CollectionScheduleDefinition:
    """Authoritative collection cadence for temporal-completeness checks.

    Attributes:
        dataset: Dataset whose ``timestamp_column`` dates each collection.
        timestamp_column: Observed collection-time column. It must be a
            timestamp field of the temporal contract, which alone decides how
            observed values are parsed and resolved to instants (see
            :func:`ql2_sixt_canada_analysis.streams.investigate_location_stream`).
        expected_periods: Authoritative ISO-8601 period starts, each with an
            explicit ``Z`` or UTC offset (naive values are rejected), that must
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
        if not all(_EXPLICIT_OFFSET.search(p.strip()) for p in periods):
            raise LocationCoverageConfigurationError("expected_periods must state an explicit UTC offset")
        if parsed.has_duplicates:
            raise LocationCoverageConfigurationError("expected_periods must not contain duplicates")

    @property
    def expected_instants(self) -> pd.DatetimeIndex:
        """Scheduled period starts as UTC instants, floored to ``period``."""
        return pd.to_datetime(list(self.expected_periods), utc=True, format="ISO8601").floor(self.period)


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


# ---------------------------------------------------- location-stream comparison


class CapturePairing(StrEnum):
    """How captures of two streams are paired for comparison.

    * ``SHARED_COLLECTION_EVENT`` - exact: captures pair when they belong to
      the same collection event (the same complete detail relationship key),
      so no timestamp tolerance is involved.
    * ``CAPTURE_TIME`` - captures pair when their reconciled instants differ
      by at most an explicit, authorised tolerance; ambiguous pairings fail
      closed.
    """

    SHARED_COLLECTION_EVENT = "shared_collection_event"
    CAPTURE_TIME = "capture_time"


@dataclass(frozen=True, slots=True)
class LocationStreamComparisonDefinition:
    """Immutable definition for comparing two expected location streams.

    Attributes:
        first, second: Expected location keys (from ``coverage``).
        coverage: The expected-location contract (dataset, location columns,
            scope columns, authoritative aliases).
        relationship: Parent/detail relationship (collection events, linkage).
        temporal: Temporal contract used to parse capture times.
        pairing: :class:`CapturePairing`.
        capture_time_field: Temporal field reference for ``CAPTURE_TIME``.
        pairing_tolerance: Explicit, authorised, non-negative tolerance for
            ``CAPTURE_TIME`` (never derived from observed data).
        identity_columns: Authoritative physical-location identity fields
            (station/branch ID, address, coordinates). Empty when the source
            has none - identity can then never be confirmed.
        product_columns: Stable offer identity (no prices, identifiers,
            location labels or capture timestamps).
        price_columns: Price fields, compared only in the price-aware offer.
    """

    first: tuple[str, ...]
    second: tuple[str, ...]
    coverage: LocationCoverageDefinition
    relationship: JobDetailRelationshipDefinition
    temporal: TemporalReconciliationDefinition
    pairing: CapturePairing
    product_columns: tuple[str, ...]
    price_columns: tuple[str, ...]
    identity_columns: tuple[str, ...] = ()
    capture_time_field: tuple[DatasetKey, str] | None = None
    pairing_tolerance: dt.timedelta | None = None

    def __post_init__(self) -> None:
        cov = self.coverage
        if not cov.is_configured:
            raise LocationCoverageConfigurationError("comparison requires a configured coverage contract")
        for key in (self.first, self.second):
            if not isinstance(key, tuple) or len(key) != len(cov.location_columns):
                raise LocationCoverageConfigurationError("comparison targets must be well-formed location keys")
            if key not in cov.expected_locations:
                raise LocationCoverageConfigurationError("comparison targets must be expected locations")
        if self.first == self.second:
            raise LocationCoverageConfigurationError("comparison targets must differ")
        if cov.dataset != self.relationship.detail:
            raise LocationCoverageConfigurationError("comparison expects detail-level locations")
        if not isinstance(self.pairing, CapturePairing):
            raise LocationCoverageConfigurationError("pairing must be a CapturePairing")
        columns = cov.source_definition.columns
        groups = {"product_columns": self.product_columns, "price_columns": self.price_columns,
                  "identity_columns": self.identity_columns}
        for name, group in groups.items():
            if not isinstance(group, tuple) or not all(isinstance(c, str) and c for c in group) \
                    or len(set(group)) != len(group):
                raise LocationCoverageConfigurationError(f"{name} must be a tuple of unique names")
            if not set(group) <= set(columns):
                raise LocationCoverageConfigurationError(f"{name} must exist in the '{cov.dataset}' contract")
        if not self.product_columns:
            raise LocationCoverageConfigurationError("product_columns must not be empty")
        technical = {*self.relationship.detail_key_columns, *self.relationship.detail_definition.unique_key_columns,
                     *self.relationship.detail_definition.identifier_columns, *cov.location_columns,
                     *cov.stream_scope_columns, *(f.column for f in self.temporal.fields if f.dataset == cov.dataset)}
        if set(self.product_columns) & (technical | set(self.price_columns) | set(self.identity_columns)):
            raise LocationCoverageConfigurationError(
                "product identity must exclude identifiers, location labels, scope, timestamps and prices"
            )
        if set(self.price_columns) & technical:
            raise LocationCoverageConfigurationError("price columns must not be technical fields")
        if self.pairing is CapturePairing.CAPTURE_TIME:
            if self.capture_time_field is None or self.pairing_tolerance is None:
                raise LocationCoverageConfigurationError("time pairing needs a capture field and a tolerance")
            if self.temporal.field(self.capture_time_field).kind is not TemporalKind.TIMESTAMP:
                raise LocationCoverageConfigurationError("capture_time_field must be a timestamp")
            if self.capture_time_field[0] != cov.dataset:
                raise LocationCoverageConfigurationError("capture_time_field must be on the location dataset")
        elif self.capture_time_field is not None or self.pairing_tolerance is not None:
            raise LocationCoverageConfigurationError("capture time and tolerance apply only to CAPTURE_TIME")
        if self.pairing_tolerance is not None and (
                not isinstance(self.pairing_tolerance, dt.timedelta) or self.pairing_tolerance < dt.timedelta(0)):
            raise LocationCoverageConfigurationError("pairing_tolerance must be a non-negative timedelta")


#: Comparison of the two owner-identified streams. Pairing is exact by shared
#: collection event: branch streams of a city are collected inside the same
#: collection job, so their captures share the detail relationship key and
#: no timestamp tolerance is needed (none is authorised). The source carries
#: no physical-identity metadata (no station/branch ID, address or
#: coordinates), so ``identity_columns`` is empty and an alias can never be
#: confirmed from this data. Product identity: vehicle and rental-search
#: attributes; prices are compared only in the price-aware offer, as exact
#: source text (price types are not yet validated).
LOCATION_STREAM_COMPARISON: Final = LocationStreamComparisonDefinition(
    first=COMPARED_LOCATION_STREAMS[0],
    second=COMPARED_LOCATION_STREAMS[1],
    coverage=EXPECTED_LOCATION_COVERAGE,
    relationship=JOB_DETAIL_RELATIONSHIP,
    temporal=TEMPORAL_RECONCILIATION,
    pairing=CapturePairing.SHARED_COLLECTION_EVENT,
    product_columns=('car_name', 'car_type', 'transmission', 'seats', 'bags', 'pickup_date', 'return_date'),
    price_columns=('price_per_day', 'price_num'),
)


# --------------------------------------------------- location identity policy


class LocationPolicyConfigurationError(ValueError):
    """A location identity policy is incomplete, contradictory or unauthorised.

    Messages never contain location values.
    """


class LocationPolicyState(StrEnum):
    """Authoritative identity decision for two related location labels.

    * ``UNRESOLVED`` - no sufficient authoritative decision exists. The labels
      must neither be compared independently nor merged.
    * ``CONFIRMED_ALIAS`` - an authority established that both labels are one
      analytical location; they may be used only through the approved
      canonical location, never as two separate locations.
    * ``CONFIRMED_DISTINCT`` - an authority established that the labels are
      distinct analytical locations; independent comparison is allowed,
      subject to every other readiness gate.

    Behavioural evidence (for example a likely-duplicate comparison result)
    never selects a state; only configuration backed by authority does.
    """

    UNRESOLVED = "unresolved"
    CONFIRMED_ALIAS = "confirmed_alias"
    CONFIRMED_DISTINCT = "confirmed_distinct"


@dataclass(frozen=True, slots=True)
class LocationPolicyAuthority:
    """Provenance of a resolved identity decision (never fabricated).

    Attributes:
        source: Who decided (for example the supplier, the collection owner
            or a named business owner). Required, non-blank.
        reference: Optional ticket, document or decision identifier.
        note: Optional short explanation.
    """

    source: str
    reference: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise LocationPolicyConfigurationError("authority source must be a non-blank string")
        for value in (self.reference, self.note):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise LocationPolicyConfigurationError("authority reference and note must be non-blank or None")


@dataclass(frozen=True, slots=True)
class LocationIdentityPolicy:
    """Authority-backed identity policy for two expected location labels.

    Attributes:
        first, second: The two expected location keys the policy governs.
        coverage: Expected-location contract (both keys must be expected,
            so its aliases can never merge them).
        state: :class:`LocationPolicyState`; ``UNRESOLVED`` by default.
        authority: Required for a resolved state; must be absent otherwise.
        canonical_location: Approved analytical key for ``CONFIRMED_ALIAS``
            (required there, forbidden otherwise). Source labels are kept for
            lineage; only the analytical key is canonicalised.
    """

    first: tuple[str, ...]
    second: tuple[str, ...]
    coverage: LocationCoverageDefinition
    state: LocationPolicyState = LocationPolicyState.UNRESOLVED
    authority: LocationPolicyAuthority | None = None
    canonical_location: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        E = LocationPolicyConfigurationError
        cov = self.coverage
        if not isinstance(cov, LocationCoverageDefinition) or not cov.is_configured:
            raise E("a configured expected-location contract is required")
        width = len(cov.location_columns)

        def well_formed(key: object) -> bool:
            return (isinstance(key, tuple) and len(key) == width
                    and all(isinstance(v, str) and v.strip() for v in key))

        for key in (self.first, self.second):
            if not well_formed(key) or key not in cov.expected_locations:
                raise E("policy labels must be expected location keys")
        if self.first == self.second:
            raise E("policy labels must differ")
        if not isinstance(self.state, LocationPolicyState):
            raise E("state must be a LocationPolicyState")
        if self.authority is not None and not isinstance(self.authority, LocationPolicyAuthority):
            raise E("authority must be a LocationPolicyAuthority")
        resolved = self.state is not LocationPolicyState.UNRESOLVED
        if resolved and self.authority is None:
            raise E("a resolved policy requires authority metadata")
        if not resolved and self.authority is not None:
            raise E("an unresolved policy must not carry decision authority")
        if self.state is LocationPolicyState.CONFIRMED_ALIAS:
            if not well_formed(self.canonical_location):
                raise E("a confirmed alias requires a well-formed canonical location")
        elif self.canonical_location is not None:
            raise E("a canonical location is allowed only for a confirmed alias")

    @property
    def resolved(self) -> bool:
        return self.state is not LocationPolicyState.UNRESOLVED

    @property
    def alias_mapping(self) -> Mapping[tuple[str, ...], tuple[str, ...]]:
        """Read-only source-key -> canonical-key mapping (empty unless a confirmed alias)."""
        if self.state is not LocationPolicyState.CONFIRMED_ALIAS:
            return MappingProxyType({})
        return MappingProxyType({self.first: self.canonical_location, self.second: self.canonical_location})


#: Identity policy for the Vancouver pair in ``COMPARED_LOCATION_STREAMS``.
#: UNRESOLVED: no authoritative decision exists in the repository or project
#: documentation. Behavioural comparison (``LOCATION_STREAM_COMPARISON``) is
#: diagnostic evidence only. Set CONFIRMED_ALIAS (with a canonical location)
#: or CONFIRMED_DISTINCT only with ``LocationPolicyAuthority`` naming the
#: supplier / collection-owner / business decision.
VANCOUVER_LOCATION_POLICY: Final = LocationIdentityPolicy(
    first=COMPARED_LOCATION_STREAMS[0],
    second=COMPARED_LOCATION_STREAMS[1],
    coverage=EXPECTED_LOCATION_COVERAGE,
    state=LocationPolicyState.UNRESOLVED,
)


# ----------------------------------------------------- vehicle-attribute stability


class VehicleStabilityConfigurationError(ValueError):
    """A vehicle-stability definition is invalid or cannot be applied.

    Messages never contain source values; offending column names (from the
    contract) are on ``columns``.
    """

    def __init__(self, message: str, columns: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.columns = columns


class MissingValueStabilityPolicy(StrEnum):
    """How missing values of one stable attribute are judged.

    * ``REQUIRED`` - every observation must carry a value; any missing value
      (always or intermittently missing) is a presence violation.
    * ``PRESENCE_STABLE`` - an attribute may be absent for a vehicle, but
      consistently: alternating between present and missing is a violation;
      always missing is allowed and reported.
    * ``MISSING_IGNORED`` - optional: distinct non-missing values are
      compared; missingness is measured and reported but never fails.
    """

    REQUIRED = "required"
    PRESENCE_STABLE = "presence_stable"
    MISSING_IGNORED = "missing_ignored"


class AttributeComparisonPolicy(StrEnum):
    """How values of a stable attribute are compared.

    * ``EXACT`` - source values as read (type-aware: case, whitespace,
      punctuation and category changes are drift; ``0`` and ``False`` are
      meaningful values; nothing is stripped, rounded or filled).
    * ``AUTHORITATIVE_MAPPING`` - an authority-supplied mapping from source
      value to canonical value is applied to a temporary copy; unmapped
      values compare exactly. Requires a non-empty mapping.
    """

    EXACT = "exact"
    AUTHORITATIVE_MAPPING = "authoritative_mapping"


@dataclass(frozen=True, slots=True)
class VehicleAttributeDefinition:
    """One stable vehicle attribute: column, missing-value and comparison policy."""

    column: str
    missing_policy: MissingValueStabilityPolicy
    comparison: AttributeComparisonPolicy = AttributeComparisonPolicy.EXACT
    mapping: Mapping[object, object] = dataclass_field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise VehicleStabilityConfigurationError("attribute column must be a non-empty string")
        if not isinstance(self.missing_policy, MissingValueStabilityPolicy):
            raise VehicleStabilityConfigurationError("missing_policy must be a MissingValueStabilityPolicy",
                                                     (self.column,))
        if not isinstance(self.comparison, AttributeComparisonPolicy):
            raise VehicleStabilityConfigurationError("comparison must be an AttributeComparisonPolicy",
                                                     (self.column,))
        if not isinstance(self.mapping, Mapping):
            raise VehicleStabilityConfigurationError("mapping must be a mapping", (self.column,))
        object.__setattr__(self, "mapping", MappingProxyType(dict(self.mapping)))
        if (self.comparison is AttributeComparisonPolicy.AUTHORITATIVE_MAPPING) != bool(self.mapping):
            raise VehicleStabilityConfigurationError(
                "a mapping is required for, and only allowed with, AUTHORITATIVE_MAPPING", (self.column,))
        if any(pd.isna(v) for pair in self.mapping.items() for v in pair if not isinstance(v, (list, tuple))):
            raise VehicleStabilityConfigurationError("mapping must not map missing values", (self.column,))


@dataclass(frozen=True, slots=True)
class VehicleStabilityDefinition:
    """Immutable contract for vehicle-attribute stability.

    The logical vehicle entity is ``(*context_columns, *entity_key_columns)``.
    Every column of the dataset contract is classified exactly once as an
    entity key, a context (scope) column, a stable attribute or a volatile /
    non-structural column, so a new source column cannot silently join or
    escape the contract.

    Attributes:
        dataset: Dataset holding the observations (detail rows).
        entity_key_columns: Ordered product-identity columns (non-empty).
        context_columns: Ordered scope columns; empty means global scope.
        attributes: Stable attributes with explicit policies (non-empty).
        price_columns: Price fields (a subset of ``volatile_columns``; never
            identity, scope or stable attributes).
        volatile_columns: Columns never treated as structural attributes
            (prices, search dates, identifiers, capture timestamps, collection
            metadata, grouping labels outside the scope).
        temporal: Temporal contract that parses the observation time.
        observation_time_field: Reference to a TIMESTAMP field on ``dataset``
            whose reconciled instants order observations.
        minimum_observations: Distinct valid observation instants an entity
            needs before stability can be claimed (>= 2).
        same_capture_conflicts_reported: Report conflicts at one instant as a
            separate category (an identity ambiguity, not drift over time).
        canonical_location_grouping: Group the location context by
            authoritative aliases declared in ``location_coverage``; off
            unless an alias is confirmed. Source labels are never changed.
        location_coverage: Coverage contract supplying aliases (required
            when ``canonical_location_grouping`` is on).
    """

    dataset: DatasetKey
    entity_key_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    attributes: tuple[VehicleAttributeDefinition, ...]
    volatile_columns: tuple[str, ...]
    price_columns: tuple[str, ...]
    temporal: TemporalReconciliationDefinition
    observation_time_field: tuple[DatasetKey, str]
    minimum_observations: int = 2
    same_capture_conflicts_reported: bool = True
    canonical_location_grouping: bool = False
    location_coverage: LocationCoverageDefinition | None = None

    def __post_init__(self) -> None:
        E = VehicleStabilityConfigurationError
        if self.dataset not in DATASET_DEFINITIONS:
            raise E("stability dataset must be registered")
        columns = DATASET_DEFINITIONS[self.dataset].columns
        groups = {"entity_key_columns": self.entity_key_columns, "context_columns": self.context_columns,
                  "volatile_columns": self.volatile_columns, "price_columns": self.price_columns}
        for name, group in groups.items():
            if not isinstance(group, tuple) or not all(isinstance(c, str) and c for c in group) \
                    or len(set(group)) != len(group):
                raise E(f"{name} must be a tuple of unique non-empty names")
        if not set(self.price_columns) <= set(self.volatile_columns):
            raise E("price columns are volatile and must not be identity, scope or stable attributes")
        if not self.entity_key_columns:
            raise E("entity_key_columns must not be empty")
        if not isinstance(self.attributes, tuple) or not self.attributes \
                or not all(isinstance(a, VehicleAttributeDefinition) for a in self.attributes):
            raise E("attributes must be a non-empty tuple of VehicleAttributeDefinition")
        stable = tuple(a.column for a in self.attributes)
        if len(set(stable)) != len(stable):
            raise E("stable attributes must be unique")
        classified = (*self.entity_key_columns, *self.context_columns, *stable, *self.volatile_columns)
        unknown = tuple(c for c in classified if c not in columns)
        if unknown:
            raise E(f"{len(unknown)} configured column(s) are not in the '{self.dataset}' contract", unknown)
        if len(set(classified)) != len(classified):
            raise E("identity, context, stable and volatile columns must be disjoint")
        unclassified = tuple(c for c in columns if c not in classified)
        if unclassified:
            raise E(f"{len(unclassified)} column(s) of the '{self.dataset}' contract are unclassified",
                    unclassified)
        if set(DATASET_DEFINITIONS[self.dataset].identifier_columns) & set(self.entity_key_columns):
            raise E("collection identifiers must not define the longitudinal vehicle entity")
        if not isinstance(self.temporal, TemporalReconciliationDefinition):
            raise E("temporal must be a TemporalReconciliationDefinition")
        try:
            time_field = self.temporal.field(self.observation_time_field)
        except TemporalConfigurationError:
            raise E("observation_time_field is not defined in the temporal contract") from None
        if time_field.dataset != self.dataset or time_field.kind is not TemporalKind.TIMESTAMP:
            raise E("observation_time_field must be a timestamp on the stability dataset")
        temporal_columns = {f.column for f in self.temporal.fields if f.dataset == self.dataset}
        if not temporal_columns <= set(self.volatile_columns):
            raise E("timestamps and dates are volatile and must not be identity, context or attributes")
        if isinstance(self.minimum_observations, bool) or not isinstance(self.minimum_observations, int) \
                or self.minimum_observations < 2:
            raise E("minimum_observations must be an integer of at least two")
        for flag in (self.same_capture_conflicts_reported, self.canonical_location_grouping):
            if not isinstance(flag, bool):
                raise E("flags must be booleans")
        if self.canonical_location_grouping:
            cov = self.location_coverage
            if not isinstance(cov, LocationCoverageDefinition) or cov.dataset != self.dataset \
                    or not set(cov.location_columns) <= set(self.context_columns):
                raise E("canonical grouping needs a coverage contract whose location columns are context")
        elif self.location_coverage is not None:
            raise E("location_coverage applies only with canonical_location_grouping")

    @property
    def group_columns(self) -> tuple[str, ...]:
        """Context then entity-key columns: the complete logical entity."""
        return (*self.context_columns, *self.entity_key_columns)

    @property
    def attribute_columns(self) -> tuple[str, ...]:
        return tuple(a.column for a in self.attributes)


#: Vehicle-attribute stability contract.
#:
#: Identity: the source has no product or vehicle identifier. The vehicle
#: product a customer sees is the supplier's vehicle name (model label)
#: offered at a pickup location, so the entity is (source location, vehicle
#: name) within the single supplier in this feed. Location scope is
#: deliberate: rental products and fleets are managed per pickup branch, so a
#: structural difference between branches is not instability. Collection
#: identifiers, offer positions, capture times and prices are excluded from
#: identity. A renamed product becomes a new entity (not drift).
#:
#: Stable attributes: vehicle category, transmission, seat and baggage
#: capacity - structural properties of one product. Category, transmission
#: and seats are core listing fields (REQUIRED); baggage capacity may be
#: unpublished for a product but should then be consistently absent
#: (PRESENCE_STABLE). Comparison is exact: no authoritative normalisation
#: exists.
#:
#: Volatile / non-structural: prices, rental search dates, collection job
#: fields and identifiers, offer position, capture timestamps and reporting
#: dates, collection status and mode, and city labels (the branch is the
#: scope). Observations are ordered by the reconciled capture instant.
#: No location alias is confirmed, so source location labels define scope.
VEHICLE_ATTRIBUTE_STABILITY: Final = VehicleStabilityDefinition(
    dataset=DatasetKey.CARS,
    entity_key_columns=('car_name',),
    context_columns=('location',),
    attributes=(
        VehicleAttributeDefinition('car_type', MissingValueStabilityPolicy.REQUIRED),
        VehicleAttributeDefinition('transmission', MissingValueStabilityPolicy.REQUIRED),
        VehicleAttributeDefinition('seats', MissingValueStabilityPolicy.REQUIRED),
        VehicleAttributeDefinition('bags', MissingValueStabilityPolicy.PRESENCE_STABLE),
    ),
    volatile_columns=(
        'job_id', 'city', 'mode', 'status', 'job_finished_at', 'scrape_date', 'job_pickup_date',
        'job_return_date', 'row_index', 'pickup_date', 'return_date', 'price_per_day', 'scraped_at',
        'price_num', 'city_clean', 'date_clean',
    ),
    price_columns=LOCATION_STREAM_COMPARISON.price_columns,
    temporal=TEMPORAL_RECONCILIATION,
    observation_time_field=(DatasetKey.CARS, 'scraped_at'),
    minimum_observations=2,
    same_capture_conflicts_reported=True,
    canonical_location_grouping=False,
)
