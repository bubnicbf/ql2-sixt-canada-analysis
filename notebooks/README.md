# notebooks

Exploratory and presentation notebooks. Reusable logic belongs in
`src/ql2_sixt_canada_analysis`; notebooks import it rather than re-implement it.

## Execution order

Run notebooks in numeric-prefix order, each one **top to bottom after
restarting the kernel** (Kernel ▸ Restart Kernel and Run All Cells). No
notebook may rely on variables, imports or files left behind by an earlier
interactive session, and none depends on the working directory.

| Order | Notebook | Purpose |
| --- | --- | --- |
| 1 | `01_data_ingestion.ipynb` | Load the `jobs` and `cars` raw datasets through the package ingestion API (blank physical lines kept as rows; centrally defined identifier fields read as nullable strings) and confirm both loaded; then apply quality step 1, `remove_blank_rows_from_raw_datasets`, which removes only completely blank rows and keeps the per-dataset and total counts in memory (`blank_rows`); then `validate_raw_dataset_identifier_dtypes` checks identifier types on the cleaned frames; then `assess_raw_dataset_unique_keys` measures unique-key completeness and uniqueness, keeping the reports in memory (`key_reports`, `all_key_contracts_valid`) without changing rows; finally `assess_job_detail_reconciliation` reconciles each job's declared detail count with the detail rows present, keeping the aggregate report in memory (`reconciliation_report`, `job_detail_counts_reconciled`). Subsequent variables (`jobs_df`, `cars_df`) are the cleaned, identifier-typed frames. No analysis or other transformation; counts, identifier values, key results and type summaries are never displayed. |

Later notebooks will be added with the next prefixes (`02_`, `03_`, ...).

## Rules

- Raw, interim and processed data under `data/` are proprietary and
  Git-ignored. Never write into `data/raw/`.
- Real data must not appear in cell outputs, markdown, metadata, test
  fixtures or reports: no DataFrame previews, columns, row counts, file names
  or paths.
- Clear all outputs and execution counts before committing
  (Edit ▸ Clear Outputs of All Cells). `tests/test_notebooks.py` fails if a
  committed notebook has outputs, execution counts, widget state or local
  paths.
- Do not call `os.chdir`, edit `sys.path`, or install packages from a cell;
  do not read CSVs directly or name source files.

## Validation

```bash
python -m pytest tests/test_notebooks.py
```

The tests check each notebook's structure and then execute a **copy** from a
clean kernel, top to bottom, against synthetic CSVs generated in a pytest
temporary directory from the centralized column contracts. The executed copy
is written only to that temporary directory; the tracked notebook is never
modified and the proprietary files are never read.

The synthetic directory reaches the notebook through the
`QL2_SIXT_RAW_DATA_DIR` environment variable, read by
`ql2_sixt_canada_analysis.paths.resolve_raw_data_dir()` (defined once in that
module). Developers do not set it: when it is unset the notebook uses the
centralized default raw directory. The same validator,
`ql2_sixt_canada_analysis.notebook_validation.execute_notebook_copy`, can be
used locally against the real files; it reports execution-order problems and
cell failures by cell index only.
