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
- All expected locations present in jobs
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