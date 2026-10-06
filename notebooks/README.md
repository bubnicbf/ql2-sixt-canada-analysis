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
| 1 | `01_data_ingestion.ipynb` | Load the `jobs` and `cars` raw datasets through the package ingestion API (blank physical lines kept as rows; centrally defined identifier fields read as nullable strings) and confirm both loaded; then apply quality step 1, `remove_blank_rows_from_raw_datasets`, which removes only completely blank rows and keeps the per-dataset and total counts in memory (`blank_rows`); then `validate_raw_dataset_identifier_dtypes` checks identifier types on the cleaned frames; then `assess_job_linkage` derives the authority-backed job linkage key and integer offer position under the approved policy from the current pricing-authority record (`load_job_linkage_policy`; raw `job_id`/`row_index` unchanged; exact match first, only the approved legacy decimal-zero repair, ambiguity and collisions block; `job_linkage_report`, `job_identifier_normalization_ready`, aggregate counts only) and every later step uses the analysis-stage frames (`analysis_jobs_df`, `analysis_cars_df`, `analysis_datasets`) and analysis-stage contracts (`ANALYSIS_DATASET_DEFINITIONS`, `ANALYSIS_JOB_DETAIL_RELATIONSHIP`); then `assess_raw_dataset_unique_keys` measures analytical unique-key completeness and uniqueness, keeping the reports in memory (`key_reports`, `all_key_contracts_valid`) without changing rows; then `current_expected_stream_contract` resolves the approved source-stream contract once from the current authority record (`expected_stream_contract`: the seven approved (city, location) keys in their exact source spelling, `EXHAUSTIVE`; its coverage is `EXPECTED_LOCATION_COVERAGE`, no second stream list; status, mode, count and approved keys printed - configuration, not source values); then `assess_dataset_location_coverage` compares that contract with distinct locations in the cleaned frame it names, exactly (missing, unexpected and spelling-variant pairs counted separately; `location_coverage_report`, `expected_location_coverage_passed`); then `assess_job_detail_reconciliation` reconciles each job's declared detail count with the detail rows present, keeping the aggregate report in memory (`reconciliation_report`, `job_detail_counts_reconciled`); finally `assess_one_to_many_join` validates the jobs-to-cars one-to-many relationship (`relationship_report`, `one_to_many_contract_valid`) and `assess_job_detail_join_readiness` gates the trusted join: `trusted_jobs_with_details` is a frame only when `job_detail_join_ready` (jobs and detail business keys, declared counts and the relationship all pass), otherwise `None` with `job_detail_join.blocking_reasons`; `diagnostic_jobs_with_details` is untrusted and for investigation only; then `investigate_location_stream` traces the centrally configured expected stream and keeps its categorical report in `location_stream_report` (earliest failing stage and status; nothing displayed or written); finally `assess_temporal_reconciliation` parses and reconciles the source temporal fields against `TEMPORAL_RECONCILIATION` (`temporal_report`, `temporal_fields_trusted` - the gate for any time-based analysis); then `compare_location_streams` compares the two related streams named in `LOCATION_STREAM_COMPARISON` and keeps the result in `location_comparison_report` as evidence only, printing the behavioural result, the paired, eligible and invalid (valid offers on both sides) and unpaired/matching/differing capture counts, where and why offers were invalid, the target and baseline evidence thresholds, overlap, baseline, whether inference is permitted and the duplicate-inference blockers (missing values are unassessable, never equal; the baseline needs the same eligible-capture minimum as the target) (a likely duplicate needs complete overlap, at least the minimum of independent paired captures, a discriminative baseline and identical offers in every pair; it is never alias confirmation, and streams are never merged); then `assess_vehicle_attribute_stability` assesses structural vehicle attributes against `VEHICLE_ATTRIBUTE_STABILITY` (`vehicle_stability_report`, `vehicle_attributes_stable` - true only when the full product population passed; a partially assessable or unassessable population never enables it); then `current_location_authority` reads the approved role map and comparison pairs from the current authority record and validates them against the source contract and the identity policy (`location_authority`; statuses, streams per role and the approved pairs on canonical keys printed - configuration, not source values); finally `assess_location_policy` reports the authority-backed `VANCOUVER_LOCATION_POLICY` state (record v4: `confirmed_alias`, canonical `Vancouver / Downtown`; raw source keys preserved, canonical analytical keys in `vancouver_analytical_locations`; a missing governed raw stream blocks), whether it is resolved and authority sufficient (a `LOCATION_MAPPING_DEFECT` in the comparison contradicts either resolved decision), its governed scope and supplied canonical key, whether canonical scope validation passed (a confirmed alias may canonicalise only to one of its governed same-city keys - never into Calgary or another stream), and the permissions, and `assess_pricing_readiness` combines it with every gate - including the authoritative schedule with complete time coverage for every expected stream (`assess_collection_schedule` + `assess_scheduled_time_coverage` -> `scheduled_coverage_report`; with no schedule configured this is `unavailable` and blocks) the trusted-join gate (`job_detail_join`, whose blockers such as `join_construction_failed` are propagated; it requires the valid linkage report of the same frames) and the job-linkage report itself (`job_linkage_report`) and the approved, exhaustive source-stream contract (`expected_stream_contract`; missing, unexpected and misspelled source streams block) - (`pricing_readiness`, `pricing_analysis_ready`, blocking reasons) - pricing stays blocked while any gate blocks - including the unresolved combination of the aliased Vancouver streams' offers. The loader enforces the complete-source policy (`raw.complete_source`); coverage checks authoritative (city, branch) pairs; continuity counts every in-scope job including zero-detail jobs; reconciliation checks every declared count; `assess_expected_location_streams` investigates every approved stream of the exhaustive contract - all seven, not a minimum subset (`expected_streams_report`); `assess_city_integrity` checks that every job city is assignable and every linked detail row carries its job's city (`city_integrity_report`, `city_integrity_valid`; counts and blocker categories only - the bounded samples hold identifiers and are never printed), and either defect also withholds the trusted join; `assess_completeness` combines them with that aggregate and the city-integrity result (`completeness`, `data_complete`, blocking reasons) and pricing readiness requires a complete source. `jobs_df` / `cars_df` are the cleaned, identifier-typed raw frames; analysis after the linkage step uses `analysis_jobs_df` / `analysis_cars_df` (raw columns plus the confidential derived keys). No analysis or other transformation; counts, identifier values, key results and type summaries are never displayed. |

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
