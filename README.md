# ql2-sixt-canada-analysis

## Project structure

```text
data/raw/          Immutable source CSVs (proprietary; kept local, not in Git)
data/interim/      Intermediate generated datasets (ignored by Git)
data/processed/    Final analysis-ready datasets (ignored by Git)
notebooks/         Exploratory and presentation notebooks
src/ql2_sixt_canada_analysis/   Reusable Python source package
tests/             Automated validation (pytest)
reports/           Generated analytical reports (ignored by default)
reports/figures/   Generated charts and figures (ignored by default)
docs/              Supporting documentation
```

- `data/raw/` is immutable input data: never edit, rename, or overwrite it.
- Generated data belongs in `data/interim/` or `data/processed/`.
- Reusable logic belongs under `src/ql2_sixt_canada_analysis/`.
- Automated validation belongs under `tests/`.
- Notebooks should import and call functions from the source package rather
  than contain duplicated production logic.

## Environment setup

Requires Python 3.11 or newer. Run these commands from the repository root.

```bash
python3 --version                      # must report 3.11 or newer
python3 -m venv .venv                  # create a local virtual environment
source .venv/bin/activate              # macOS/Linux (Windows: .venv\Scripts\activate)
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"      # package + runtime deps + pytest/JupyterLab
python -m pytest                       # run the test suite
jupyter lab                            # optional: start JupyterLab
```

Dependencies are declared only in `pyproject.toml`: runtime analysis libraries
under `[project] dependencies`, and test and notebook tools under the `dev`
extra. Recreate the environment by deleting `.venv/` and repeating the steps.

Data and environment safety:

- `.venv/` is local to your machine and must never be committed.
- All QL2 data in `data/raw/`, `data/interim/`, and `data/processed/` is
  proprietary and must never be committed; Git ignores these directories
  except for their placeholder/README files.
- Tests use small synthetic data created inside the tests. Never substitute
  real QL2 data for committed test fixtures.
- This configuration installs code and dependencies only; it does not make
  proprietary datasets safe to publish.

## Raw data ingestion

Configuration lives in exactly one place each:

- `ql2_sixt_canada_analysis.paths`: `PROJECT_ROOT` (derived from the package
  location, not the working directory), `DATA_DIR`, `RAW_DATA_DIR`,
  `INTERIM_DATA_DIR` and `PROCESSED_DATA_DIR`.
- `ql2_sixt_canada_analysis.schemas`: the logical dataset keys
  (`DatasetKey.JOBS`, `DatasetKey.CARS`) and an immutable `DatasetDefinition`
  per dataset holding its filename tokens and exact, ordered CSV column
  contract (`DATASET_DEFINITIONS`, `get_dataset_definition()`).
- `ql2_sixt_canada_analysis.ingestion` consumes both: it discovers one CSV
  per dataset, validates each header against its contract, and loads the
  files as pandas DataFrames.

```python
from ql2_sixt_canada_analysis import load_raw_datasets
from ql2_sixt_canada_analysis.paths import RAW_DATA_DIR

raw = load_raw_datasets(RAW_DATA_DIR)   # the default; pass another directory to override
jobs_df = raw.jobs                      # pandas.DataFrame
cars_df = raw.cars                      # pandas.DataFrame
```

- **Discovery:** only `*.csv` files directly in the directory are considered
  (case-insensitive, no recursion). A file belongs to the dataset whose
  filename token appears last in its name, so prefixes, numbers, spaces and
  download suffixes such as `(1)` are tolerated. Missing or duplicate
  datasets raise an `IngestionError` subclass instead of guessing.
- **Header contract:** each header must match its dataset's `columns`
  exactly, including order; otherwise `SourceSchemaError` reports the
  missing, unexpected, duplicated or reordered columns. Contracts describe
  structure only: they imply no types or keys and do not make the data safe
  to publish.
- **Ingestion only:** each file is parsed once with `pandas.read_csv`
  defaults plus the project read policy `ingestion.RAW_CSV_READ_DEFAULTS`
  (extra options via `read_csv_options=`). Nothing is cleaned, renamed,
  coerced, deduplicated or written to disk.
- **Blank lines are preserved:** pandas drops completely blank physical lines
  by default, which would make them impossible to count. The loader reads with
  `skip_blank_lines=False`, defined once in `RAW_CSV_READ_DEFAULTS`, so every
  physical line after the header becomes a row (an empty line and a
  delimiter-only line become all-missing rows; a whitespace-only line keeps
  the whitespace in its first field). `skip_blank_lines` is rejected inside
  `read_csv_options`; the only opt-out is the explicit, documented
  `preserve_blank_lines=False` argument, after which blank-row counts are not
  meaningful. Because the first physical line must then be the header, a file
  whose header is preceded by blank lines is rejected with `RawDataLoadError`.
- **Read options:** the header check is itself a one-row `pandas.read_csv`,
  so tokenizer options (`sep`, `quotechar`, `escapechar`, `doublequote`,
  `skipinitialspace`, `dialect`, `encoding`, `compression`, ...) apply to it
  exactly as to the full read. Value options (`dtype`, `na_values`, ...) apply
  to the full read only; options that reshape the header or result, and
  unrecognised options, raise `ValueError`.
- **Confidentiality:** raw inputs are read-only and proprietary. Raw, interim
  and processed data are Git-ignored and must never be committed. Tests use
  small synthetic CSVs generated in temporary directories and never read the
  real files.

### Complete-source loader policy

`load_raw_datasets` returns **every** record of each file or fails; it never
samples. Before any file is opened it rejects, with
`IncompleteSourceOptionError` (naming the option and why), every
`read_csv_options` entry that could omit or reshape records:

| Option | Why it is prohibited |
| --- | --- |
| `nrows` | limits the number of records read |
| `skiprows`, `skipfooter` | omit leading/selected or trailing records |
| `comment` | truncates lines and drops comment-only records |
| `chunksize`, `iterator` | return partial readers instead of one complete frame |
| `usecols`, `index_col`, `header`, `names` | project away or reshape contract columns |
| `on_bad_lines` other than `"error"` | `"skip"`, `"warn"` or a callable discard malformed records |

`on_bad_lines="error"` (the parser default) is allowed, so malformed records
fail the load (`RawDataLoadError`). Remaining options only change tokenising
or value interpretation. `RawDatasets.complete_source` is `True` only when the
loader enforced this policy with blank lines preserved (blank-row removal
carries the flag through); frames assembled any other way are not proven
complete.

## Identifier fields

Identifier fields are labels (e.g. the scrape-job identity shared by `jobs`
and `cars`), not measurements. Their names are defined once per logical
dataset in `ql2_sixt_canada_analysis.schemas`
(`DatasetDefinition.identifier_columns`, `.identifier_dtypes`,
`SHARED_IDENTIFIER_COLUMNS`); the source code is the authority, so they are
not repeated here.

- **Read-time typing.** `load_raw_datasets` passes the identifier dtype
  mapping to `pandas.read_csv`, so identifiers are read as pandas' nullable
  string dtype (`IDENTIFIER_DTYPE`, `"string"`) and never parsed as numbers.
  This prevents leading-zero loss, rounding of long integers, scientific
  notation and `.0` artefacts. Shared identifiers get the same dtype in both
  datasets. After the read the loader validates the guarantee.
- **Missing stays missing.** Empty identifier fields are `pd.NA`, never the
  text `"nan"`, `"None"` or `"<NA>"`, so completely blank rows are still
  detected and removed by the blank-row step.
- **No normalisation.** Only the type changes. Identifier text is not
  stripped, re-cased, padded, parsed or validated against any business
  format; reconciling different textual forms is a separate, later step.
- **Other columns** keep normal pandas inference unless a caller configures
  them.
- **Caller options.** `read_csv_options` may add `dtype` rules for other
  columns. A `dtype` entry for an identifier is accepted only if it is a
  nullable string dtype; any other identifier `dtype`, a non-text scalar
  `dtype` (e.g. `int`), or a `converters`/`parse_dates` entry for an
  identifier raises `IdentifierTypeConflictError` before any file is read. A
  text scalar `dtype` (e.g. `str`) applies to the other columns only.
  `engine="pyarrow"` is rejected because it infers types before casting and
  would drop leading zeros. Caller dictionaries are never mutated.
- **Post-load helpers** for frames built elsewhere:
  `cast_identifier_fields(frame, definition)` and
  `cast_identifiers_for_raw_datasets(raw)` return new frames with only the
  identifier columns cast (order, index and other columns unchanged;
  idempotent). They **cannot recover** leading zeros or digits already lost
  if a frame was parsed numerically before reaching them, which is why the
  loader types identifiers at read time.
- **Validation:** `validate_identifier_dtypes(frame, definition)` and
  `validate_raw_dataset_identifier_dtypes(raw)` are silent on success and
  raise `IdentifierDtypeError` / `MissingIdentifierColumnError` with the
  dataset key and a count (column names on attributes, never values).
- Tests use fabricated identifiers only (e.g. `000123`, `SYNTHETIC-ID-001`).
  Raw, interim and processed data remain proprietary and Git-ignored;
  identifier values, extracts and type profiles from real data must never be
  committed.

## Data quality: completely blank rows

`ql2_sixt_canada_analysis.quality` holds the first data-quality step and the
project's single definition of a **completely blank row**: a row in which
every field is a pandas missing value, `None`, an empty string or a
whitespace-only string. Any other value is meaningful, including `0`, `0.0`,
`False`, the strings `"0"` and `"False"`, timestamps and non-empty text, so a
partially blank row is never removed.

```python
from ql2_sixt_canada_analysis import load_raw_datasets, remove_blank_rows_from_raw_datasets

raw = load_raw_datasets()                        # ingestion: blank lines kept as rows
blank_rows = remove_blank_rows_from_raw_datasets(raw)   # quality: remove and count

blank_rows.jobs.cleaned                          # cleaned jobs DataFrame
blank_rows.jobs.original_row_count               # int
blank_rows.jobs.retained_row_count               # int
blank_rows.jobs.removed_blank_row_count          # int
blank_rows.cars                                  # the same BlankRowResult for cars
blank_rows.by_key[DatasetKey.JOBS]               # results keyed by DatasetKey
blank_rows.total_removed_blank_row_count         # jobs + cars
blank_rows.cleaned                               # RawDatasets of both cleaned frames
```

`remove_completely_blank_rows(frame)` does the same for a single DataFrame and
returns a `BlankRowResult`; `completely_blank_row_mask(frame)` exposes the
classification alone.

- Ingestion and cleaning are separate stages: `load_raw_datasets` preserves
  blank physical lines (see above) and removes nothing; the quality function
  removes and counts them.
- The step is applied to `jobs` and `cars` independently; the results are
  typed, so original, retained and removed counts and the two datasets cannot
  be confused. `original == retained + removed` always holds.
- The loaded DataFrames are never mutated; the cleaned frame is a new object.
  Column order, retained-row order and every retained cell value (including
  whitespace) are preserved exactly; nothing is stripped or normalised.
- **Index policy:** the original index labels are kept (no reset) so retained
  rows can be traced back to the loaded frame. Call `reset_index` yourself if
  you need positional labels.
- Nothing is written to disk and nothing is logged or printed. The raw files
  are untouched.
- Counts and cleaned frames derived from the proprietary data are themselves
  proprietary: keep them in memory and never commit them (no quality reports,
  rejected-row extracts or cleaned exports; `.gitignore` guards the common
  locations). Automated tests (`tests/test_quality.py`) use only synthetic
  DataFrames and synthetic temporary CSVs.

## Unique keys

Each logical dataset's business key is defined once, with its row grain, in
`ql2_sixt_canada_analysis.schemas` (`DatasetDefinition.unique_key_columns`;
the source code is the authority, so component names are not repeated here).
Keys are chosen from the grain's meaning, not from what happens to be unique
in one extract: `jobs` has one row per scrape job, keyed by the job
identifier; `cars` has one row per offer position within a job, keyed by the
parent job identifier plus the offer's ordinal position (the one documented
non-identifier component, see `non_identifier_key_columns`).

```python
from ql2_sixt_canada_analysis import assess_raw_dataset_unique_keys, validate_raw_dataset_unique_keys

key_reports = assess_raw_dataset_unique_keys(cleaned)   # always returns reports
key_reports.jobs.is_complete, key_reports.jobs.is_unique, key_reports.jobs.is_valid
key_reports.cars.missing_key_row_count, key_reports.cars.duplicate_key_row_count
key_reports.all_valid
validate_raw_dataset_unique_keys(cleaned)               # raises on violations
```

- **A valid key is complete and unique.** Completeness and uniqueness are
  measured separately; the contract passes only when both hold.
- **Missing keys:** a row with any missing component (pandas `NA`/`NaN`) is a
  missing-key row. Missing-key rows are counted on their own and never form
  duplicate groups. Non-empty text such as `"0"`, `"False"`, `"N/A"` or
  `"null"` is a value, and keys are compared verbatim (never stripped or
  normalised); validating identifier *content* is a separate control.
- **Duplicate keys:** evaluated only among complete rows, with all
  components compared as a tuple (`DataFrame.duplicated(keep=False)`), never
  as concatenated strings, so delimiter-like characters cannot collide. Every
  row in a duplicate group is counted (`duplicate_key_row_count`), and groups
  are counted separately (`duplicate_key_group_count`).
- **Assessment vs strict validation:** `assess_unique_key` /
  `assess_raw_dataset_unique_keys` always return typed, in-memory reports
  (counts only, no key values or rows) and raise only
  `KeyConfigurationError` for configuration problems such as a missing key
  column. `validate_unique_key` / `validate_raw_dataset_unique_keys` raise
  `UniqueKeyViolationError` naming the dataset and violation categories
  (`missing_key`, `duplicate_key`), with the reports on `.reports`.
- **Nothing is changed:** validation never removes, deduplicates, fills,
  sorts or repairs rows, and writes nothing.
- **Pipeline order:** completely blank rows are removed *before* key
  assessment, so a blank physical line is counted by the blank-row control
  and not again as a missing key; partially populated rows with a missing key
  remain and are reported.
- **Empty datasets** are vacuously valid (no missing keys, no duplicates);
  whether records were expected at all is a separate presence/volume control.
- Tests use fabricated keys only. Key values, duplicate or missing-key
  extracts and key reports derived from the proprietary data must never be
  committed.

## Expected location coverage

`ql2_sixt_canada_analysis.schemas.EXPECTED_LOCATION_COVERAGE` is the single
expected-location contract: the dataset and location column(s), an immutable
tuple of expected location keys (one component per column), a mode, optional
authoritative aliases and the scope columns used by stream investigation.
Locations are **branch-level** pickup locations, which only the detail
(`cars`) rows carry; jobs are city-level collection runs, so a jobs-level
contract cannot represent a branch stream. Expected locations must come from
an **independent authority** and are never derived from the extract being
validated. The current contract holds three authority-identified streams:
`INVESTIGATED_LOCATION_STREAM` and the two streams in
`COMPARED_LOCATION_STREAMS`. The authority defines a required minimum rather
than an exhaustive universe, so the mode is `MINIMUM_REQUIRED`; add further
locations only from an authoritative list.
An unconfigured contract fails closed with `LocationCoverageConfigurationError`.

```python
from ql2_sixt_canada_analysis import assess_dataset_location_coverage, validate_expected_location_coverage

report = assess_dataset_location_coverage(cleaned)     # uses the frame the contract names
report.coverage_ratio, report.is_valid, report.violations
validate_expected_location_coverage(cleaned.cars)      # raises LocationCoverageError
```

- **Distinct coverage:** an expected location is covered when at least one
  cleaned row has its exact key (or an authoritative alias); repeated rows at
  one location never compensate for another missing location.
- **Separate signals:** missing expected locations, unexpected observed
  locations and rows with a missing / empty / whitespace-only location
  component are counted separately; unassigned rows create no observed key.
- **Modes:** `EXHAUSTIVE` fails on any unexpected location;
  `MINIMUM_REQUIRED` only reports them. Both fail on missing expected
  locations and on unassigned rows. The mode must be stated by the authority.
- **Exact comparison:** case-sensitive, no stripping, punctuation,
  abbreviation or fuzzy handling; composite keys are compared as tuples.
  Aliases count only when declared in the contract's `aliases` (none are
  confirmed today) and never rewrite source values.
- **Assessment vs strict validation:** assessment returns a frozen report of
  counts, a ratio and booleans (no location values or lists) and raises only
  configuration errors; strict validation raises `LocationCoverageError`
  listing violation categories only.
- Coverage runs on cleaned frames, before the one-to-many join, and never
  removes or alters rows. Tests use fabricated locations. Real location lists
  and coverage reports must not be committed.

### City-branch pairs

The project contract keys expected locations on authoritative
**(city, branch)** pairs (`location_columns=('city', 'location')`,
`label_column='location'`), compared exactly as given:

- a branch label observed under another city is a different, *unexpected*
  pair and never covers the expected pair (e.g. a Vancouver branch label
  under Calgary leaves the Vancouver pair missing);
- a label observed under more than one city is a *conflicting assignment*
  (`conflicting_location_label_count`) and fails the contract; city is never
  inferred from the label and labels are never merged;
- the report lists the configured expected pairs (`expected_pairs`) and the
  missing ones (`missing_expected_locations`); observed pairs are source
  values and are available only in memory through `location_pair_evidence`.

These controls do not decide Vancouver identity (see the location policy
below) and do not establish timestamp authority.

## Expected location stream investigation

`investigate_location_stream(jobs_df, cars_df, target)` traces one expected
location stream (resolved with `resolve_expected_location`, which accepts
only an exact configured key and never builds or substitutes one from data)
through configuration, raw source (optional header-aware scan), ingestion,
cleaning, location matching, continuity across the collection events of its
scope, schedule-based time coverage, identifier types, parent keys,
relationship and reconciliation. It returns a frozen, categorical
`LocationStreamInvestigationReport`: the **earliest failing stage**, every
failing stage in order, one primary `LocationStreamStatus` and booleans such
as `repository_fix_required` and `upstream_issue_indicated` - no
identifiers, location values, timestamps, rows or counts.
`validate_location_stream` raises `LocationStreamError` unless the stream is
present and healthy.

- An absent stream (`RAW_STREAM_ABSENT`) is distinguished from jobs present
  with details absent (`JOBS_PRESENT_DETAILS_ABSENT`), a stream present in
  only some collection events (`RAW_STREAM_PARTIAL`), pipeline losses
  (`INGESTION_EXCLUSION`, `CLEANING_EXCLUSION`), key, link and count failures.
- A case/space/punctuation/order variant is reported as `UNVERIFIED_ALIAS`
  and never applied; only aliases declared in the contract are matched.
- Continuity compares the collection events that actually occurred for the
  target's scope; it does not infer a cadence. Temporal completeness needs an
  authoritative `COLLECTION_SCHEDULE`; none exists, so it is reported as
  `NOT_ASSESSED`.
- When a schedule is configured, its timestamp must be a timestamp field of
  `TEMPORAL_RECONCILIATION`, and observed times are resolved only through
  that contract (e.g. the `MST` designator as fixed UTC-07:00). Scheduled
  instants must state an explicit offset. No scheduled period observed
  (`SCHEDULED_TIME_ABSENT`), some missing (`RAW_STREAM_PARTIAL`) or any
  target time that cannot be reconciled - missing, invalid, or without zone
  authority, never assumed UTC (`SCHEDULED_TIME_UNASSESSABLE`) - all fail at
  `TIME_COVERAGE`, and `validate_location_stream` rejects them.
- It never creates, repairs, filters or writes records. Tests use fabricated
  locations and identifiers. Proprietary diagnostics and extracts must not be
  committed; sanitized, metric-free notes live in `docs/investigations/`.

### Zero-detail jobs and continuity

Continuity is counted over **jobs**, not detail rows: every job whose city
(`parent_scope_columns`) matches the stream's city is in the denominator,
including jobs with no detail rows. `event_accounting`
(`StreamEventAccounting`) reconciles
`total_jobs = scope_excluded + scope_unassignable + in_scope` and
`in_scope = with the branch + other branches only + zero-offer + missing details`.
Jobs carry only a city, so a job without linked detail rows cannot be
assigned to a branch: a **zero-offer** capture (both declared counts valid
and zero - evidence that the capture happened and returned an empty
assortment) and a job whose declared details are **missing** both make
branch continuity `UNASSESSABLE` (`CONTINUITY_UNASSESSABLE`) rather than
disappearing. In-scope jobs whose details lack the branch make it `PARTIAL`.
No branch is invented, and orphan detail rows never enter the job
denominator (they remain relationship failures).

## Job-to-detail count reconciliation

The jobs-to-cars relationship is defined once in
`ql2_sixt_canada_analysis.schemas` (`JOB_DETAIL_RELATIONSHIP`: parent key =
the jobs unique key, the matching detail foreign key, and the jobs column that
declares how many detail rows each job should have). Field names live in the
source code only.

```python
from ql2_sixt_canada_analysis import assess_job_detail_reconciliation, validate_job_detail_reconciliation

report = assess_job_detail_reconciliation(jobs_df, cars_df)   # aggregate report, in memory
report.is_reconciled, report.violations
validate_job_detail_reconciliation(jobs_df, cars_df)          # raises JobDetailReconciliationError
```

- **Per job, not global.** For every job, *expected* is the job's declared
  count and *observed* is the number of cleaned detail rows whose complete
  key matches it (zero when none do). Matched, under-counted and over-counted
  jobs are counted separately, with net and absolute discrepancy; equal
  global totals cannot produce a pass when per-job errors offset.
- **Detail rows** are linked, *missing link* (a relationship key component is
  missing) or *orphan* (complete key matching no job), each counted
  separately, plus the number of distinct orphan keys. Every row present is
  counted, duplicates included.
- **Expected counts are not repaired.** Missing (including empty text),
  non-numeric (including booleans), non-finite, fractional and negative
  counts are categorised and make the contract fail; they are never filled,
  rounded, clamped or dropped. Whole-valued floats and numeric strings for
  non-negative whole numbers are interpreted in a temporary array only.
- **Preconditions** raise `ReconciliationPreconditionError`: relationship
  keys must use the identifier dtype, the jobs key must be complete and
  unique (duplicate jobs are never collapsed), and completely blank rows
  must already be removed. Absent columns raise
  `RelationshipConfigurationError`.
- **Assessment vs strict validation:** assessment returns a frozen report of
  aggregate integers (no identifiers, keys or rows) and does not fail on
  ordinary mismatches; strict validation raises
  `JobDetailReconciliationError` listing violation categories only.
- **Nothing is changed:** no rows are removed, deduplicated or altered, and
  nothing is written.
- **Empty data:** empty jobs and empty details are vacuously reconciled
  (presence/volume is a separate control); empty jobs with any detail rows
  fail; jobs without detail rows pass only if every expected count is a
  valid zero.
- Tests use fabricated identifiers and counts. Real reconciliation reports,
  mismatch, orphan or missing-link extracts must never be committed.

### Every declared count

`JOB_DETAIL_RELATIONSHIP.expected_detail_count_columns` is
`('record_count', 'actual_car_rows')`. Each declaration is reconciled
**independently** against the observed detail rows of the job (jobs with
none are observed `0`) - `count_fields` holds one `DeclaredCountFieldReport`
per column - and the declarations must agree with each other
(`declared_counts_disagree_job_count`). Missing, non-numeric, negative,
fractional or non-finite declarations are invalid and never coerced. A job is
reconciled only when every declaration is valid and matches; `is_reconciled`
(used by the notebook, the stream investigation and the trusted join)
requires every job reconciled and every detail row linked. A job with two
detail rows and declarations `2` / `999` (or `999` / `2`) fails.
`job_detail_count_results` gives the per-job evidence (`observed_detail_count`,
each declaration, `<column>_matches`, `declared_counts_agree`,
`job_reconciled`) in memory only, sorted by key. Duplicate detail keys remain
a separate key-integrity failure; nothing is deduplicated.

### All expected streams

The expected-location contract currently holds **three** authority-identified
(city, branch) pairs (`INVESTIGATED_LOCATION_STREAM` followed by
`COMPARED_LOCATION_STREAMS`) in `MINIMUM_REQUIRED` mode: a required minimum,
not an exhaustive list of every branch or every scheduled city.
`assess_expected_location_streams(jobs_df, cars_df)` investigates **every**
configured pair exactly once, in contract order, and returns an
`ExpectedLocationStreamsReport` that pairs each report with its configured
key (`ExpectedStreamResult`). Its properties are derived from the contract on
every access, so a hand-built aggregate (`from_reports`) is validated too:

- an omitted pair (`expected_stream_report_missing`), a pair reported more
  than once (`duplicate_stream_report`), a key outside the contract
  (`unexpected_stream_report`) or a `None` report
  (`expected_stream_report_unavailable`) fails - an omitted report is a
  failure, never "not applicable";
- every expected stream must be healthy with complete continuity
  (`stream_continuity_partial`, `stream_continuity_unassessable`,
  `expected_stream_unhealthy`);
- `expected_stream_count`, `assessed_stream_count`, `assessed_exactly_once`,
  `all_expected_streams_healthy` and the read-only `reports` mapping support
  diagnostics without exposing source values.

Branch labels live in the detail (`cars`) rows; jobs supply the city-level
collection events that form each stream's continuity denominator. Row-level
coverage therefore never substitutes for stream health: a pair can appear in
detail rows while no job of its city exists. Adding a pair to the contract
automatically adds it to the required assessment.

### Aggregate completeness

`assess_completeness(datasets=..., coverage=..., streams=expected_streams_report, reconciliation=...)`
takes the all-expected-stream aggregate (a plain tuple of reports is
rejected) and is complete only when the source is complete, the city-branch
coverage passes, the aggregate was built for the same contract
(`expected_coverage`, default `EXPECTED_LOCATION_COVERAGE`), every expected
stream is assessed exactly once and is healthy, and every declared count
reconciles; a missing report or aggregate blocks. Each failure is a
`CompletenessBlocker` (`source_not_complete`, `expected_pairs_missing`,
`expected_stream_assessment_unavailable`, `stream_contract_mismatch`,
`expected_stream_report_missing`, `stream_continuity_partial`,
`declared_count_unreconciled`, ...). A `CompletenessReport` with no blockers
cannot be constructed without a valid aggregate. Completeness is one
prerequisite only: it does not resolve timestamp authority, Vancouver
location identity, key validity or stability, and does not make the
airport-premium or any other pricing analysis ready.

## One-to-many relationship and relationship-checked join

Jobs are the **one** side and cars the **many** side of the centrally defined
relationship (`JOB_DETAIL_RELATIONSHIP`). This control answers whether jobs
can be joined to cars safely; it is separate from unique-key validation (each
dataset's own key) and count reconciliation (declared vs observed counts),
and reuses their definitions and helpers.

```python
from ql2_sixt_canada_analysis import assess_one_to_many_join, join_jobs_to_details, validate_one_to_many_join

report = assess_one_to_many_join(jobs_df, cars_df)   # aggregate report, in memory
report.is_valid, report.violations
validate_one_to_many_join(jobs_df, cars_df)          # raises OneToManyRelationshipError
result = join_jobs_to_details(jobs_df, cars_df)      # relationship-checked only - NOT analytical trust
```

- **Contract:** parent keys must be complete and unique (otherwise
  `RelationshipPreconditionError`; duplicate parents are never collapsed or
  chosen arbitrarily). Detail foreign keys may repeat - many cars per job is
  expected - but every detail row must link to exactly one job. Missing-link
  details (a key component missing) and orphan details (complete key absent
  from jobs) are counted separately, and every detail row falls into exactly
  one of linked, missing-link or orphan.
- **Cardinality and conservation:** a parent-left merge with pandas
  `validate="one_to_many"` must succeed, and the left join must have exactly
  `linked detail rows + parents without details` rows, with every linked
  detail once and every detail-less parent once. Equal input totals are never
  accepted as proof.
- **Relationship-checked join:** `join_jobs_to_details` strictly validates
  the relationship first, merges with `how="left"`, `validate="one_to_many"`,
  `sort=False`, re-checks the row count and raises `OneToManyRelationshipError`
  / `ValidatedJoinError` (pandas `MergeError` as the cause) instead of
  returning a relationship-invalid frame. It does **not** check business keys
  or declared counts, so it is not an analytically trusted join; use the
  trusted job-detail join gate below.
  Parent order and, within a parent, detail order are preserved; the result
  has a fresh `RangeIndex`. It never repairs, drops or deduplicates
  violations, and the joined frame stays in memory.
- **Column collisions:** same-named key columns appear once; differently
  named keys are both kept. Same-named non-key columns get stable suffixes
  `_job` (parent) and `_detail` (detail) from the relationship definition;
  other columns keep their names. Source frames are never renamed.
- **Empty data:** empty jobs and cars are vacuously valid (presence/volume is
  separate); empty jobs with any detail rows fail; jobs with no detail rows
  are valid and each appears once in the left join (count reconciliation may
  still fail if they declared non-zero counts).
- Tests use fabricated identifiers. Joined proprietary data and relationship
  reports must never be written to tracked locations or committed.

## Trusted job-detail join

A relationship-valid join is not necessarily analytically trustworthy: a
duplicated detail business key, a duplicated or incomplete jobs key, or
declared job-level counts that disagree with the detail rows can all coexist
with a passing one-to-many relationship, and pandas will still merge. A
non-`None` DataFrame is never proof of analytical validity.

```python
from ql2_sixt_canada_analysis import assess_job_detail_join_readiness, require_trusted_job_detail_join

join = assess_job_detail_join_readiness(jobs_df, cars_df)
join.join_ready, join.blocking_reasons
join.trusted_jobs_with_details        # DataFrame only when join_ready, else None
join.diagnostic_jobs_with_details     # UNTRUSTED investigation frame, or None
require_trusted_job_detail_join(jobs_df, cars_df)   # trusted frame or UntrustedJoinError
```

- **Prerequisites (all must explicitly pass, on the same frames that are
  joined, in one call):** the jobs business-key contract
  (`jobs_key_contract_valid`), the detail business-key contract
  (`details_key_contract_valid`; together `all_key_contracts_valid`),
  declared-count reconciliation (`declared_counts_reconciled`) and the
  one-to-many relationship contract (`relationship_contract_valid`: no orphan
  or missing-link details, validated cardinality and row conservation).
- **Fail closed:** a report that cannot be produced (identifier-type, blank
  row or parent-key preconditions) is unavailable and blocks
  (`required_report_unavailable`); the absence of a violation is never read
  as a pass. `join_ready` is true only with no `JobDetailJoinBlocker`
  (`jobs_key_contract_failed`, `details_key_contract_failed`,
  `declared_counts_not_reconciled`, `relationship_contract_failed`,
  `orphan_details_present`, `missing_link_details_present`,
  `join_construction_failed`). Every applicable reason is reported.
- **Trusted vs diagnostic:** `trusted_jobs_with_details` is the only frame
  downstream analysis may use. `diagnostic_jobs_with_details` is the
  relationship-checked join kept for investigation when the relationship
  passes but another contract fails; it must never feed pricing, aggregation
  or conclusions. Orphans are never dropped silently into either frame.
- Frames are returned as fresh copies; mutating the inputs or a returned
  frame cannot change the held result. Nothing is deduplicated, repaired or
  written.
- **Migration:** the notebook variable `jobs_with_details` (previously set
  whenever the relationship passed) is replaced by `trusted_jobs_with_details`
  / `job_detail_join_ready`, with `diagnostic_jobs_with_details` clearly
  labelled untrusted.
- A trusted join is one prerequisite only; it does not establish pricing
  readiness (`assess_pricing_readiness` keeps every other gate).

## Temporal reconciliation

`ql2_sixt_canada_analysis.schemas.TEMPORAL_RECONCILIATION` is the single
temporal contract for `finished_at`, `scraped_at`, `scrape_date` and
`date_clean` (plus the detail rows' copy of the parent job's finish time).
Each field states its dataset, whether it is a **timestamp** or a
**calendar date**, its format and its time-zone policy.

| Field | Dataset | Kind | Established time-zone basis |
| --- | --- | --- | --- |
| `finished_at` | jobs (copy on detail rows) | timestamp | naive; **no authoritative zone**, so not resolved to an instant |
| `scraped_at` | cars, per detail row | timestamp | source designator mapped explicitly (`MST` = fixed UTC-07:00, the collector's clock label, not market-local time) |
| `scrape_date` | jobs and cars | calendar date | source-supplied |
| `date_clean` | cars | calendar date | source-supplied (not generated by repository code) |

```python
from ql2_sixt_canada_analysis import assess_temporal_reconciliation, validate_temporal_reconciliation

report = assess_temporal_reconciliation(jobs_df, cars_df)   # aggregate report, in memory
report.is_valid, report.violations, report.unavailable_rules
validate_temporal_reconciliation(jobs_df, cars_df)          # raises TemporalParseError / TemporalReconciliationError
```

- **Parsing:** missing (pandas missing, empty or whitespace-only),
  invalid (unparsable, impossible dates, unsupported types, a time on a date
  field) and *unresolved* (no authoritative zone, ambiguous or nonexistent
  local times, unknown designators, mixed offset awareness) are counted
  separately. Naive values are **never** given the machine's zone or UTC.
- **Canonical comparison:** instants are compared in **UTC**. A reporting
  date is derived only *after* converting the source instant to the rule's
  reporting zone; rows whose reporting date differs from the UTC date but
  match are counted as legitimate boundary crossings.
- **Rules:** ordering (direction, inclusive/exclusive, explicit non-negative
  tolerance), reporting-date derivation and replication (detail-row copies
  equal their parent) are assessed independently, per linked detail row via
  the central relationship; missing and orphan links are unassessable, never
  dropped.
- **Established rule:** the detail rows' parent finish-time copy must equal
  the parent job's `finished_at` (wall times; both share the same naive
  basis).
- **Unavailable (fail closed until an authority defines them):** the
  ordering between `finished_at` and `scraped_at`, a time zone for
  `finished_at`, and the source timestamp and reporting zone behind
  `scrape_date` and `date_clean`. Markets span several zones and no
  authoritative location-to-zone mapping exists. **No tolerance is
  authorised.** Strict validation fails while any of these is unavailable.
- **Empty data:** empty frames have no contradictions (presence is a
  separate control), but configuration must be valid and unavailable rules
  still fail closed.
- Assessment returns a frozen report of counts and booleans only; strict
  validation raises a typed exception listing categories only. Source values
  are never rewritten, filled or dropped. Tests use fabricated timestamps and
  dates; real temporal diagnostics must not be committed.

## Related location stream comparison

`compare_location_streams(jobs_df, cars_df, LOCATION_STREAM_COMPARISON)`
compares two expected location streams. The pair is defined once, in
`COMPARED_LOCATION_STREAMS` / `LOCATION_STREAM_COMPARISON`
(`src/ql2_sixt_canada_analysis/schemas.py`), together with the pairing mode,
the product columns (stable offer identity: no identifiers, location labels,
timestamps or prices), the price columns and any authoritative identity
columns. The result is a frozen, categorical
`LocationStreamComparisonReport` (enums and booleans only).

Evidence is kept in three layers and never mixed:

1. **Observation** - presence of each stream, linkage, shared collection
   events, temporal overlap, and whether each paired capture has identical
   product multisets and identical price-aware offer multisets (exact tuple
   comparison with `collections.Counter`: multiplicity kept, row order
   ignored, no lossy fingerprints, prices never used as product identity).
   A **scope baseline** applies the same comparison to every other location
   pair in the same scope, so identical behaviour counts only if it is not
   normal for the source.
2. **Interpretation** - `LIKELY_DUPLICATE_STREAMS`, `LIKELY_DISTINCT_STREAMS`,
   `COMPARISON_INCONCLUSIVE`, `INSUFFICIENT_COMPARABLE_CAPTURES`, or presence
   failures `ONE_STREAM_ABSENT` / `BOTH_STREAMS_ABSENT`.
   `LIKELY_DUPLICATE_STREAMS` needs affirmative evidence for **every**
   prerequisite; each missing one is a `DuplicateInferenceBlocker` in
   `duplicate_inference_blockers`, and the result is then
   `COMPARISON_INCONCLUSIVE` (insufficient duplicate evidence is never read
   as distinctness):
   - at least `minimum_paired_captures` independent paired capture events
     (`MINIMUM_DUPLICATE_PAIRED_CAPTURES = 2`, validated as an integer of at
     least two). A capture is one collection event: duplicate rows and the
     many vehicle offers inside one capture are one observation, so they
     never add temporal evidence (`insufficient_paired_captures`);
   - `COMPLETE` temporal overlap - any capture present in only one stream
     means the streams diverge somewhere, so partial overlap is
     `incomplete_temporal_overlap` (no tolerance policy exists);
   - a `DISCRIMINATIVE` scope baseline (an allowlist): with `UNAVAILABLE`
     nothing shows that identical behaviour is unusual for the source, and
     with `NON_DISCRIMINATIVE` it is normal (`baseline_unavailable` /
     `baseline_non_discriminative`);
   - identical price-aware offers in every paired capture - one differing
     capture outweighs any number of matching ones and is kept as evidence
     (`differing_paired_captures`);
   - every stream row identifiable as a capture (rows without a capture key,
     or with an unresolvable capture time under time pairing, are counted as
     unassessable instead of being dropped) and unambiguous pairing.
   The report exposes the denominator: captures per stream, paired,
   unpaired per stream, matching and differing paired captures, unassessable
   rows and the minimum. The rule is symmetric in the two streams.
3. **Confirmation** - only from authoritative identity columns:
   `CONFIRMED_DISTINCT_LOCATIONS`, `CONFIRMED_ALIAS`,
   `DUPLICATED_COLLECTION_CONFIGURATION` or `LOCATION_MAPPING_DEFECT`.
   Similar names, proximity or identical prices never confirm identity.

Pairing is `SHARED_COLLECTION_EVENT` (exact, by the detail relationship key;
the project setting) or `CAPTURE_TIME` (reconciled instants within an
explicit, authorised tolerance; ambiguous or unresolved times fail closed).

```python
from ql2_sixt_canada_analysis import (
    LOCATION_STREAM_COMPARISON, compare_location_streams, validate_confirmed_location_alias,
)

report = compare_location_streams(jobs_df, cars_df, LOCATION_STREAM_COMPARISON)
report.status, report.upstream_review_required, report.alias_authority_sufficient
validate_confirmed_location_alias(jobs_df, cars_df)   # raises LocationAliasNotConfirmedError
```

- Streams are **never merged, relabelled or deduplicated**. Source-label
  coverage stays the default. `canonical_location_keys(frame, coverage)` is
  an opt-in, non-mutating view that applies only aliases declared in the
  coverage contract (none are confirmed).
- The project source carries no physical-identity metadata, so the project
  comparison can reach at most a *likely* conclusion; confirmation requires
  supplier or collection-configuration evidence. See
  `docs/investigations/location_stream_comparison.md` (sanitized).
- Even a valid `LIKELY_DUPLICATE_STREAMS` is behavioural evidence only: it
  never creates an alias, merges streams, resolves the Vancouver location
  policy or enables pricing.
- The comparison is **evidence only**. `alias_authority_sufficient` refers to
  identity metadata in the data (none exists here) and does not decide
  anything downstream: whether the two labels may be merged or compared is
  decided solely by the location identity policy below.
- Tests use fabricated values only. Comparison tables, offer fingerprints,
  paired-capture and price comparison exports are ignored by Git and must
  not be committed.

## Vancouver location policy and pricing readiness

**Behavioural evidence is not identity authority.** Identical offers, full
temporal overlap, similar names or a `LIKELY_DUPLICATE_STREAMS` /
`LIKELY_DISTINCT_STREAMS` comparison can justify an investigation, but never
decide whether the two Vancouver labels are one analytical location.
That decision is the authority-backed `VANCOUVER_LOCATION_POLICY`
(`LocationIdentityPolicy` in `schemas.py`), with three states:

| State | Meaning | Analysis allowed |
| --- | --- | --- |
| `UNRESOLVED` (**default**) | No sufficient authoritative decision exists. | Neither independent comparison nor merging. Pricing is blocked. |
| `CONFIRMED_ALIAS` | An authority established both labels are one analytical location. | Only through the approved `canonical_location`; never as two separate locations. |
| `CONFIRMED_DISTINCT` | An authority established they are distinct analytical locations. | Independent comparison, subject to every other gate. |

A resolved state requires `LocationPolicyAuthority` (non-blank `source`,
optional `reference` and `note`); an unresolved policy may not carry one.
`CONFIRMED_ALIAS` requires a canonical location; any other state forbids one.
Incomplete or contradictory configuration raises
`LocationPolicyConfigurationError`. Nothing is derived from data.

```python
from ql2_sixt_canada_analysis import (
    VANCOUVER_LOCATION_POLICY, apply_location_policy, assess_location_policy, assess_pricing_readiness,
)

keys = apply_location_policy(cars_df)                    # source_keys + analytical_keys (frame untouched)
policy = assess_location_policy(VANCOUVER_LOCATION_POLICY, location_comparison_report, keys)
readiness = assess_pricing_readiness(location_policy=policy, completeness=completeness,
                                     key_contracts_valid=..., ..., vehicle_stability=vehicle_stability_report)
readiness.ready, readiness.blocking_reasons
```

Derived permissions on `LocationPolicyReport`:

- `location_policy_resolved` - an authority-backed `CONFIRMED_ALIAS` or
  `CONFIRMED_DISTINCT` decision is configured (recorded, not necessarily
  usable).
- `location_policy_authority_sufficient` - resolved and not contradicted by
  authoritative identity metadata in the comparison (a contradiction blocks
  pricing as `identity_evidence_conflicts_with_policy`; see below).
- `locations_are_aliases` - true only for a sufficient `CONFIRMED_ALIAS`.
  **False is not evidence that the locations are distinct.**
- `locations_comparable_independently` - true only for a sufficient
  `CONFIRMED_DISTINCT`.
- `canonicalization_required` / `canonicalization_applied` - a confirmed
  alias needs `apply_location_policy` keys built from that same policy;
  `analytical_keys` use the canonical location while `source_keys` keep the
  original labels for lineage and audit.

**Authoritative evidence that contradicts a resolved policy.** Comparison
evidence never selects a state, but authoritative identity evidence can
contradict a configured one (`identity_evidence_conflict`):

| Comparison evidence | Contradicts |
| --- | --- |
| `LOCATION_MAPPING_DEFECT` | **both** `CONFIRMED_ALIAS` and `CONFIRMED_DISTINCT` |
| `CONFIRMED_DISTINCT_LOCATIONS` | `CONFIRMED_ALIAS` |
| `CONFIRMED_ALIAS`, `DUPLICATED_COLLECTION_CONFIGURATION` | `CONFIRMED_DISTINCT` |

`LOCATION_MAPPING_DEFECT` means the authoritative identity columns hold
conflicting physical-location identities *within* one stream. That is
evidence for neither aliasing nor distinctness - the mapping any decision
would rest on is itself broken - so under it a configured decision stays
recorded (`location_policy_resolved` is true, the state is not changed) but
is **not** `location_policy_authority_sufficient`: neither alias grouping
(`locations_are_aliases`) nor independent comparison
(`locations_comparable_independently`) is permitted, applied canonicalisation
stays recorded but does not help, and pricing is blocked as
`identity_evidence_conflicts_with_policy` (alongside
`alias_canonicalization_not_applied` when that also applies) until the source
identity mapping is corrected or authoritatively reconciled. No identity
value is chosen and no row is dropped. Behavioural statuses (likely
duplicate, likely distinct, inconclusive, insufficient captures, absent
streams) never create or override a conflict. An `UNRESOLVED` policy has no
decision to contradict, so `identity_evidence_conflict` is false there; the
defect stays visible as the recorded evidence (`mapping_defect_indicated`)
and pricing is blocked as `vancouver_policy_unresolved`. A
`LocationPolicyReport` whose conflict flag disagrees with its recorded
evidence cannot be constructed.

`assess_pricing_readiness` is fail closed: the `CompletenessReport`
(complete source, city-branch coverage, **every** configured expected
stream, declared counts - there are no separate booleans that could override
it; `None` blocks as `completeness_unavailable`, a failure as
`data_incomplete`, plus `expected_streams_not_proven` for stream failures)
and every other foundational gate (key contracts, one-to-many relationship,
temporal trust, vehicle stability, passed as the full-population
`VehicleStabilityReport`) must pass, and the location policy must add no blocker. Each failure is a `PricingBlocker` (for example
`vancouver_policy_unresolved`, `alias_canonicalization_not_applied`,
`temporal_fields_untrusted`); `validate_pricing_readiness` raises
`PricingNotReadyError`. A resolved policy never overrides another gate.

**Still required:** an authoritative statement - from the supplier, the
collection owner or the business - of whether the two Vancouver labels are
the same pickup location (with the approved canonical label) or distinct
locations, recorded as `LocationPolicyAuthority`. Until then no Vancouver
pricing conclusion or airport-versus-downtown comparison involving these
labels may proceed.

## Vehicle-attribute stability

`VEHICLE_ATTRIBUTE_STABILITY` (`src/ql2_sixt_canada_analysis/schemas.py`) is
the single contract for whether structural vehicle attributes stay the same
across repeated observations of one vehicle product. Every detail column is
classified exactly once as entity key, scope (context), stable attribute or
volatile/non-structural, so new columns cannot silently join or escape it.

- **Identity and scope:** the source has no product identifier. The entity is
  the supplier's vehicle name offered at a source (pickup) location, within
  the single supplier in the feed. Products and fleets are managed per
  branch, so differences between branches are not instability. Collection
  identifiers, offer positions, capture times and prices never define
  identity; a renamed product is a new entity, not drift.
- **Stable vs volatile:** stable attributes are the vehicle category,
  transmission and seat and baggage capacity. Prices, rental search dates,
  collection fields, timestamps, reporting dates and city labels are
  volatile and never assessed as structural attributes.
- **Comparison:** exact, type-aware source values (case, whitespace,
  punctuation and category changes are drift; `0` and `False` are values).
  An `AUTHORITATIVE_MAPPING` policy exists for authority-supplied mappings,
  applied to a temporary copy; none is configured.
- **Missing values** (per attribute, `MissingValueStabilityPolicy`):
  `REQUIRED` (any missing value violates), `PRESENCE_STABLE` (consistently
  absent is allowed, alternating presence violates) or `MISSING_IGNORED`
  (measured, never fails). Missingness is reported separately from value
  conflicts; missing values are never filled or turned into sentinels.
- **History and time:** observations are ordered by the reconciled capture
  instant from `TEMPORAL_RECONCILIATION`, never by row order. An entity needs
  at least `minimum_observations` (2) distinct valid captures; a single
  capture is *insufficient history*, not proof. Entities with any missing or
  unresolvable capture time are *temporally unassessable* (set-based
  conflicts are still detected). Distinct values at one instant are a
  *same-capture conflict*. A change that later reverts is still a conflict.
- **Aliases:** source location labels define the scope. Canonical grouping
  applies only aliases declared in the coverage contract and only when
  `canonical_location_grouping` is enabled; none is confirmed, so it is off.
- **Full-population status:** the result describes every in-scope entity,
  never only the assessable ones. Each entity is in exactly one category
  (`classify_vehicle_entities`: incomplete identity, temporally
  unassessable, insufficient history, sufficient history), so
  `distinct_entities = incomplete + temporally unassessable + sufficient +
  insufficient`.

  | Status | Meaning | Valid |
  | --- | --- | --- |
  | `PASSED` | Every entity has a complete identity, valid times and at least `minimum_observations` distinct captures, and none violates the contract. | yes |
  | `VIOLATIONS` | At least one proven violation (conflict, presence, same-capture, incomplete identity, unassessable time). Takes precedence; insufficient history is still listed. | no |
  | `PARTIALLY_ASSESSABLE` | No proven violation, but some entities have sufficient and others insufficient history. | no |
  | `UNASSESSABLE` | No entity has sufficient history, or the population is empty. | no |

  `violations` lists proven violations only; **an empty list does not mean
  stable**. `blocking_reasons` lists every reason the population did not
  pass, adding `insufficient_history` (also alongside violations) and
  `empty_population`. There is no partial-coverage tolerance: one
  under-observed product blocks the pass. Duplicate rows in one capture do
  not add history; the boundary is inclusive (exactly the minimum is
  sufficient).
- **Downstream:** `vehicle_attributes_stable` (notebook) is true only for
  `PASSED`. `assess_pricing_readiness` takes the report itself
  (`vehicle_stability=`) and blocks with `vehicle_attributes_unstable` for
  proven violations, `vehicle_history_insufficient` for missing history or an
  empty population (both when both apply) and
  `vehicle_stability_unavailable` when no report exists. Product-level
  aggregation, vehicle matching and comparisons that assume stable product
  identity must not proceed on a non-passing population. Presence and
  volume are separate controls.

```python
from ql2_sixt_canada_analysis import (
    assess_vehicle_attribute_stability, validate_vehicle_attribute_stability,
)

report = assess_vehicle_attribute_stability(cars_df)   # aggregate counts, in memory
report.status, report.violations, report.attributes
validate_vehicle_attribute_stability(cars_df)          # raises VehicleAttributeStabilityError
```

Assessment requires blank rows removed and identifier types applied
(`VehicleStabilityPreconditionError`), raises
`VehicleStabilityConfigurationError` for configuration problems and otherwise
returns a frozen report of counts, enums and contract field names - no
vehicles, values, timestamps or locations. Strict validation raises
`VehicleAttributeStabilityError` for every non-`PASSED` status, with
`blocking_reasons` (`incomplete_identity`, `temporally_unassessable`,
`value_conflict`, `same_capture_conflict`, `presence_instability`,
`insufficient_history`, `empty_population`) and the proven `violations`
separately. `classify_vehicle_entities` returns per-entity keys and
categories in memory only; identifiers are confidential and are never
printed or persisted (the notebook reports aggregate counts only). Source values are never rewritten, filled or
dropped. Tests use fabricated vehicles only; real stability profiles, change
extracts and fingerprints are ignored by Git and must not be committed.

## Notebooks

Notebooks live in `notebooks/` and run in numeric-prefix order, top to bottom
from a restarted kernel; `notebooks/README.md` lists the order and rules.
Committed notebooks must have no outputs. Validate structure and execution
(against synthetic temporary data) with:

```bash
python -m pytest tests/test_notebooks.py
```

## Data trust

- All scheduled cities represented
- All authoritative expected (city, branch) pairs present in detail (`cars`) rows
- Job level counts = detail row counts
- Valid offers duplicated
- Job timestamps consistent
- Date field that defines reporting day
- Location/Product attributes stability

## Airport matched premium

Charge difference between airport and downtown

- City
- Vehicle type
- Capture time
- Price percentile
- Percentage of matches where the airport is higher

## Genuine pricing events

- Matched offer frequency from one hour to the next
- Typical increase/decrease
- Simultaneuous products move count
- Do airport/downtown locations move together
- New price persistence
- Simultaneous assortment change
- Duplicated large moves across aliased locations

## Visible assortment stability

- Offers per location and capture
- Which products appear/disappear
- Isolated/widespread changes
- Consecutive product sets similarity
- Falling assortment vs. price increases

## QL2 controls

- Expected/location completeness
- Job/detail reconciliation
- Duplicate/location/alias detection
- Timestamp offset validation
- Product attribute consistency
- Large synchronized price change alerts
- Assortment drop alerts
- Persistence confirmation
