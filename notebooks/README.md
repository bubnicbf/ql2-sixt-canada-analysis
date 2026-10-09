# notebooks

Presentation notebooks. Reusable logic belongs in
`src/ql2_sixt_canada_analysis`; notebooks import it rather than re-implement it.

## Execution order

Run notebooks in numeric-prefix order, each one **top to bottom after
restarting the kernel** (Kernel ▸ Restart Kernel and Run All Cells). The
notebooks are **independent**: none reads variables, imports, outputs or files
left behind by another notebook or an earlier session, and none depends on the
working directory. Notebooks 02 to 06 each rebuild every readiness gate through
the package, so any one of them can be run on its own; the order is the reading
order. Notebook 06 is the final report and the one to read first if you only
read one.

| Order | Notebook | Purpose |
| --- | --- | --- |
| 1 | `01_data_ingestion.ipynb` | Load the `jobs` and `cars` raw datasets through the package ingestion API (blank physical lines kept as rows; centrally defined identifier fields read as nullable strings) and confirm both loaded; then apply quality step 1, `remove_blank_rows_from_raw_datasets`, which removes only completely blank rows and keeps the per-dataset and total counts in memory (`blank_rows`); then `validate_raw_dataset_identifier_dtypes` checks identifier types on the cleaned frames; then `assess_job_linkage` derives the authority-backed job linkage key and integer offer position under the approved policy from the current pricing-authority record (`load_job_linkage_policy`; raw `job_id`/`row_index` unchanged; exact match first, only the approved legacy decimal-zero repair, ambiguity and collisions block; `job_linkage_report`, `job_identifier_normalization_ready`, aggregate counts only) and every later step uses the analysis-stage frames (`analysis_jobs_df`, `analysis_cars_df`, `analysis_datasets`) and analysis-stage contracts (`ANALYSIS_DATASET_DEFINITIONS`, `ANALYSIS_JOB_DETAIL_RELATIONSHIP`); then `assess_raw_dataset_unique_keys` measures analytical unique-key completeness and uniqueness, keeping the reports in memory (`key_reports`, `all_key_contracts_valid`) without changing rows; then `current_expected_stream_contract` resolves the approved source-stream contract once from the current authority record (`expected_stream_contract`: the seven approved (city, location) keys in their exact source spelling, `EXHAUSTIVE`; its coverage is `EXPECTED_LOCATION_COVERAGE`, no second stream list; status, mode, count and approved keys printed - configuration, not source values); then `assess_dataset_location_coverage` compares that contract with distinct locations in the cleaned frame it names, exactly (missing, unexpected and spelling-variant pairs counted separately; `location_coverage_report`, `expected_location_coverage_passed`); then `assess_job_detail_reconciliation` reconciles each job's declared detail count with the detail rows present, keeping the aggregate report in memory (`reconciliation_report`, `job_detail_counts_reconciled`); finally `assess_one_to_many_join` validates the jobs-to-cars one-to-many relationship (`relationship_report`, `one_to_many_contract_valid`) and `assess_job_detail_join_readiness` gates the trusted join: `trusted_jobs_with_details` is a frame only when `job_detail_join_ready` (jobs and detail business keys, declared counts and the relationship all pass), otherwise `None` with `job_detail_join.blocking_reasons`; `diagnostic_jobs_with_details` is untrusted and for investigation only; then `investigate_location_stream` traces the centrally configured expected stream and keeps its categorical report in `location_stream_report` (earliest failing stage and status; nothing displayed or written); finally `assess_temporal_reconciliation` parses and reconciles the source temporal fields against the authority-backed contract `current_temporal_reconciliation()` (finish times local to the parent job city's IANA zone, full-precision UTC instants, exact parent/detail replication, scrape time earlier than or equal to the finish time with zero tolerance; `temporal_report`, `temporal_fields_trusted` - the gate for any time-based analysis, true on the real data with the approved reporting day of record v8: both scrape dates equal the local calendar date of the parent finish instant in the parent city's zone, parsed strictly; `date_clean` is retired from pricing and reported for parse quality only) and the next cell prints `current_temporal_authority()` statuses and aggregate counts only; then `compare_location_streams` compares the two related streams named in `LOCATION_STREAM_COMPARISON` and keeps the result in `location_comparison_report` as evidence only, printing the behavioural result, the paired, eligible and invalid (valid offers on both sides) and unpaired/matching/differing capture counts, where and why offers were invalid, the target and baseline evidence thresholds, overlap, baseline, whether inference is permitted and the duplicate-inference blockers (missing values are unassessable, never equal; the baseline needs the same eligible-capture minimum as the target) (a likely duplicate needs complete overlap, at least the minimum of independent paired captures, a discriminative baseline and identical offers in every pair; it is never alias confirmation, and streams are never merged); then `assess_vehicle_attribute_stability` assesses structural vehicle attributes against `VEHICLE_ATTRIBUTE_STABILITY` (`vehicle_stability_report`, `vehicle_attributes_stable` - true only when the full product population passed; a partially assessable or unassessable population never enables it); then `current_location_authority` reads the approved role map and comparison pairs from the current authority record and validates them against the source contract and the identity policy (`location_authority`; statuses, streams per role and the approved pairs on canonical keys printed - configuration, not source values); finally `assess_location_policy` reports the authority-backed `VANCOUVER_LOCATION_POLICY` state (record v5: `confirmed_alias`, canonical `vancouver / Vancouver Downtown`; raw source keys preserved, canonical analytical keys in `vancouver_analytical_locations`; a missing governed raw stream blocks), whether it is resolved and authority sufficient (a `LOCATION_MAPPING_DEFECT` in the comparison contradicts either resolved decision), its governed scope and supplied canonical key, whether canonical scope validation passed (a confirmed alias may canonicalise only to one of its governed same-city keys - never into Calgary or another stream), and the permissions, and `assess_pricing_readiness` combines it with every gate - including the authority-backed per-stream collection schedule with every expected stream-period covered by its own stream (`current_per_stream_schedule` + `assess_per_stream_scheduled_coverage` -> `collection_schedule`, `scheduled_coverage_report`: parent `finished_at` in the city's approved IANA zone, one parent job per city and local hour, each stream judged only on its own exact key and periods, one governed `INCOMPLETE_PARENT_CAPTURE` exclusion from record v8 that makes both Calgary stream-periods of one incomplete capture analytically null while every raw row is kept; this step runs before the stream investigation, which receives `capture_exclusions` and counts the governed capture separately instead of as a gap; any unexcused missing stream-period blocks; counts only) the trusted-join gate (`job_detail_join`, whose blockers such as `join_construction_failed` are propagated; it requires the valid linkage report of the same frames) and the job-linkage report itself (`job_linkage_report`) and the authority-backed rental-date report (`assess_rental_dates` with `current_rental_date_policy()` -> `rental_date_report`: exact ISO dates, return on or after pickup with same-day rentals valid and no maximum, every detail date equal to its parent date as parsed dates on trusted linked rows, never repaired; aggregate counts and pricing-eligible rows only) and the approved, exhaustive source-stream contract (`expected_stream_contract`; missing, unexpected and misspelled source streams block) - (`pricing_readiness`, `pricing_analysis_ready`, blocking reasons) - pricing stays blocked while any gate blocks. After the temporal step, `derive_reporting_days`, `derive_rental_periods` and `build_pricing_population` build the pricing-eligible population (`pricing_population`, `pricing_jobs_df`, `pricing_cars_df`: governed exclusion first, then reporting-day, rental-date and capture-period controls; aggregate counts only); the comparison and the vehicle stability used for pricing run on that population (`pricing_location_comparison_report`, `pricing_vehicle_stability_report`; the full-population reports stay as diagnostics), and `assess_canonical_offers` combines the aliased Vancouver streams' offers under the approved `CANONICAL_OFFER_COMBINATION` (`canonical_offer_report`, counts only; a required readiness input). The loader enforces the complete-source policy (`raw.complete_source`); coverage checks authoritative (city, branch) pairs; continuity counts every in-scope job including zero-detail jobs; reconciliation checks every declared count; `assess_expected_location_streams` investigates every approved stream of the exhaustive contract - all seven, not a minimum subset (`expected_streams_report`); `assess_city_integrity` checks that every job city is assignable and every linked detail row carries its job's city (`city_integrity_report`, `city_integrity_valid`; counts and blocker categories only - the bounded samples hold identifiers and are never printed), and either defect also withholds the trusted join; `assess_completeness` combines them with that aggregate and the city-integrity result (`completeness`, `data_complete`, blocking reasons) and pricing readiness requires a complete source. `jobs_df` / `cars_df` are the cleaned, identifier-typed raw frames; analysis after the linkage step uses `analysis_jobs_df` / `analysis_cars_df` (raw columns plus the confidential derived keys). No analysis or other transformation; counts, identifier values, key results and type summaries are never displayed. |
| 2 | `02_matched_location_pricing.ipynb` | Matched location pricing, independent of notebook 01's kernel state: `run_matched_location_pricing` rebuilds every foundational gate through `run_pricing_pipeline` and stops - printing only blocker categories - unless pricing readiness is true and the readiness, frame binding, location authority, canonical offers, schedule evidence (same job proven via the derived linkage key) and pricing-population vehicle stability all agree. Grain: one approved city, one shared trusted collection job, one rental period, one exact approved product (`car_name`, `car_type`, `transmission`, `seats`, `bags` plus pickup/return dates), one currency and price basis, exactly one airport and one canonical downtown offer (Vancouver `Thurlow` combined into canonical `Vancouver Downtown`). Premium = airport minus downtown (dollars in exact cents; percent of the downtown price; zero denominators counted, never infinite). Displays aggregate match counts and attrition (`match_count_frame`), city distribution summaries (`city_summary_frame`), the main two-panel figure (`plot_matched_location_premiums`, rendered in memory) and vehicle-type summaries and Kruskal-Wallis tests with Holm adjustment and epsilon-squared (`vehicle_type_summary_frame`, `vehicle_type_test_frame`; associational, not causal). Writes files only when `REPORTS_DIR` is set; the local Git-ignored report and figure come from `python -m ql2_sixt_canada_analysis.matched_location_pricing`. |
| 3 | `03_price_change_events.ipynb` | Price-change events, independent of earlier kernel state: `run_price_change_presentation` runs `run_pricing_pipeline` exactly once and passes that one result through the event engine (`price_change_events`), the higher-order analysis (`price_change_analysis`) and the presentation reconciliation (`price_change_presentation`); when anything is blocked only blocker categories are printed and no table, figure or file is produced. Displays only sanitized aggregate tables (reconciliation summary, event interval summary, material synchronized movements and selection reconciliation, airport/downtown summary, persistence summary, final Vancouver decrease summary) and the reconciled two-panel event heatmap rendered in memory (`heatmap_png`). Nothing is written unless an explicit output directory is set (`OUTPUT_DIR` or `QL2_SIXT_PRICE_CHANGE_OUTPUT_DIR`); the confidential detailed Parquet event table additionally needs `WRITE_DETAIL` or `QL2_SIXT_PRICE_CHANGE_WRITE_DETAIL=1` and is never displayed. Observed candidates only - descriptive, not proof of repricing or extraction error. |
| 4 | `04_visible_assortment.ipynb` | Visible assortment, independent of earlier kernel state: `run_assortment_presentation` runs `run_pricing_pipeline` exactly once and passes that one result through the price-change engine, the visible-assortment engine (`visible_assortment`) and the presentation reconciliation (`assortment_presentation`). When anything is blocked, only blocker categories and a generic blocked narrative are printed, and no table, figure or finding is produced. Otherwise the notebook displays only sanitized aggregate tables: the reconciliation summary, the assortment timeline (one row per canonical location and scheduled capture), the location summary, the observed-drop review with cross-location simultaneity, and the price-coincidence summary. It also shows the timeline figure, rendered in memory (`assortment_timeline_png`), and the deterministic narrative (`build_assortment_narrative`). The unusual-drop policy is unavailable, so observed drops are review candidates only, and coincidence is not causation. Timeline persistence is not approved: nothing is written, and an output directory is refused with `persistence_not_approved`. |
| 5 | `05_monitoring_actionability.ipynb` | Monitoring and actionability (data-plan Section 6), independent of earlier kernel state: `run_monitoring` runs `run_pricing_pipeline` exactly once, derives the price-change analysis and the visible assortment from that same result, checks that both are bound to its frames and location authority, and evaluates the eight controls (`monitoring`). It shows the severity scale, the five evaluation statuses, and one sanitized row per control (`monitoring_control_table`: condition, severity, likely business impact, recommended response, calibration status, evaluation status and typed finding, evidence-gap and note codes), then status-only summary lines and the production-calibration requirements. Missing locations, job/detail counts, duplicate or aliased feeds, timestamp offsets and product attributes are contract-based and evaluated now; abrupt assortment changes and large synchronized price movements are candidate-only because no threshold is approved; end-of-window observations are right-censored and need confirmation by another collection. A missing prerequisite gives `not_assessable`, never a pass; a blocked evidence chain shows only blocker categories and the eight definitions as `not_assessable`. No files are written and no alert or notification is produced. The roughly 90-hour sample cannot calibrate production thresholds. |
| 6 | `06_final_report.ipynb` | Final report (data-plan Section 7), independent of earlier kernel state: `run_final_report` runs `run_pricing_pipeline` exactly once and passes that one result to `matched_location_pricing_from_pipeline`, `presentation_from_pipeline`, `assortment_presentation_from_pipeline` and `monitoring_from_pipeline` (`final_report`). Evidence that is not bound to that run, or a missing readiness report, produces no commercial section. It presents, in order: purpose and analytical questions, scope and confidentiality, data and pipeline readiness (`final_status_lines`, `final_section_table`), matched-location pricing (`matched_summary_table`, `matched_premium_png`), price changes (persistence summary and heatmap), visible assortment (location summary and cross-location drops), monitoring (`monitoring_overview_table`), assumptions, exclusions, limitations, unanswered questions (`open_questions_table`), requested additional data (`data_requests_table`), final conclusions (`final_conclusions`) and the Section 7 reconciliation. Every result cell is followed by an interpretation generated by `interpret_section` from the sanitized reports; a blocked section shows only its blocker categories and says what cannot be concluded. Nothing is written. |

New notebooks take the next prefix (`07_`, ...) and must be added to this table in order.

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
- Every cell that displays a result (a table, figure, printed status block or
  summary) is tagged `result` in its cell metadata and is **immediately
  followed** by a cell tagged `interpretation`: a short Markdown cell that says
  what the result means, what it supports and its main limitation, or (in the
  final report) a code cell that prints text generated by
  `interpret_section`/`final_conclusions`. Never hard-code real-data values in
  Markdown; runtime-dependent interpretation comes from a tested package
  helper. `tests/test_notebooks.py` enforces the convention.

## Running a notebook

Interactively: start `jupyter lab` from the activated environment, open the
notebook, then Kernel ▸ Restart Kernel and Run All Cells. With the default
settings each notebook reads the two raw exports from `data/raw/`.

Headless, without touching the tracked file (the executed copy stays in
memory; pass `output_path=` only to a Git-ignored location such as
`reports/`, because an executed copy run on the real data contains
confidential outputs):

```bash
python - <<'PY'
from tempfile import TemporaryDirectory
from ql2_sixt_canada_analysis.notebook_validation import execute_notebook_copy

with TemporaryDirectory() as workdir:
    result = execute_notebook_copy("notebooks/06_final_report.ipynb", workdir=workdir, timeout_seconds=900)
print("Executed code cells:", len(result.execution_counts))
PY
```

To execute a copy against **synthetic** data instead, point the documented
override at a directory of synthetic exports that follow the column contracts
(`QL2_SIXT_RAW_DATA_DIR=<synthetic directory> python - <<'PY' ...`). The test
suite does exactly this with CSVs it generates in a temporary directory.

When readiness gates do not pass (for example on synthetic data), notebooks 02
to 04 print `blocked` with blocker categories only, notebook 05 marks the
dependent controls `not_assessable`, and notebook 06 shows each section's
status and blockers and states that no commercial finding is valid. That is
the expected fail-closed behaviour, not a notebook error.

## Validation

```bash
python -m pytest tests/test_notebooks.py tests/test_final_documentation.py
```

The tests check each notebook's structure (including the `result` /
`interpretation` tags) and then execute a **copy** from a
clean kernel (notebooks 02 and 06 from a working directory outside the repository; on
the synthetic CSVs it is not pricing ready, so the test confirms it stops
cleanly with blocker categories only and writes nothing), top to bottom, against synthetic CSVs generated in a pytest
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

## Clearing outputs before committing

Committed notebooks must have no outputs, execution counts, widget state or
execution timing. In JupyterLab use Edit ▸ Clear Outputs of All Cells and
save; from the command line (keeps sources, cell ids, the `result` /
`interpretation` tags and the kernelspec; removes everything else, including
execution timing that `jupyter nbconvert --clear-output` leaves behind):

```bash
python -m ql2_sixt_canada_analysis.notebook_validation --clear notebooks/*.ipynb
python -m pytest tests/test_notebooks.py -k "clean or portable or machine_specific"
```
