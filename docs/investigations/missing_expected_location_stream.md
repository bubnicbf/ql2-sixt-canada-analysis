# Investigation: missing expected location stream

- **Date:** 2026-10-04
- **Stream:** the expected branch-level location configured as
  `INVESTIGATED_LOCATION_STREAM` in `src/ql2_sixt_canada_analysis/schemas.py`
  (identified as expected by the project owner).
- **Tool:** `investigate_location_stream` (`src/ql2_sixt_canada_analysis/streams.py`).

This note is sanitized: it records stage classifications only - no rows,
identifiers, timestamps, date ranges, counts, rates, source filenames or
alternate location values.

## Stages examined

Expected-location configuration, raw source, ingestion, source-schema
validation, blank-row handling, identifier typing, parent unique keys,
location matching (exact, case, whitespace, punctuation, component order,
airport vs downtown, aliases), continuity across collection events, time
coverage, job-to-detail reconciliation, one-to-many linkage, detail presence
and notebook logic.

## Observations

1. **Configuration (repository defect, fixed).** The expected-location
   contract was unconfigured and keyed on the city-level jobs field. Branch
   locations exist only in detail rows, so the stream could not be expressed
   and no control could flag it. Adding the stream to the jobs-level contract
   would have produced a permanent false absence.
2. **Raw source, ingestion, cleaning, schema, identifier typing:** the stream
   is present with its exact configured key in the raw detail source, and
   every pipeline stage retains it. No filter, query or merge in the code or
   notebook removes it.
3. **Location matching:** exact match; no representation variant or alias was
   involved; airport and downtown branches remain distinct.
4. **Continuity (earliest failing data stage):** the stream is present in only
   some of the collection events recorded for its scope. In events without
   it, other branches of the same scope are present and each event's declared
   detail count matches the rows actually returned, so the rows were not lost
   inside the pipeline. All other branch streams were present in every event
   of their scope.
5. **Time coverage:** no authoritative collection schedule exists, so whether
   the overall cadence was complete cannot be proven.
6. **Linkage:** the stream's detail rows cannot link to their parent jobs
   because of the known textual-form difference of the shared job identifier
   (pipeline-wide, not specific to this stream; see the identifier and
   reconciliation notes).

## Conclusion

- **Earliest failing stage:** configuration (fixed); afterwards, source
  continuity.
- **Primary category:** `RAW_STREAM_PARTIAL` - an upstream collection gap:
  the source intermittently returned no rows for this branch.
- **Repository defect:** yes, in configuration only (contract granularity and
  missing expectation); fixed. No code path removed the stream.
- **Not established (speculation only):** *why* the source returned nothing
  for this branch in some runs (e.g. availability, site behaviour or a
  collection failure). The data does not show which.

## Controls now preventing recurrence

- `EXPECTED_LOCATION_COVERAGE` is branch-level and includes the stream
  (`MINIMUM_REQUIRED`), so coverage fails if it disappears entirely.
- `investigate_location_stream` / `validate_location_stream` detect partial
  streams across collection events and localise the earliest failing stage;
  the ingestion notebook runs it after the relationship step.
- Synthetic regression tests: `tests/test_streams.py`, `tests/test_coverage.py`.

## Open dependencies

- An authoritative collection schedule (`COLLECTION_SCHEDULE`) to assess
  temporal completeness.
- ~~An explicit identifier-normalisation step so detail rows link to jobs.~~ Done: authority-backed
  job linkage (`job_linkage`, pricing-authority record `v2`).
- Upstream confirmation of why the branch was not returned in some runs.

## Update (2026-10-06): approved exact source spelling

Pricing-authority record `v3` approves the exhaustive seven-stream source
contract with exact source spellings
([governance reference](../decisions/governance/expected-stream-governance-2026-10-06.md)).
`INVESTIGATED_LOCATION_STREAM` is now the approved key `Calgary / Downtown`.
The observations above were made with the earlier project-owner key, whose
spelling is the one the extract carries; under exact matching the extract
does not contain the approved spelling, so the approved stream is reported
`raw_stream_absent` and the extract's stream is an unexpected spelling
variant (`source_spelling_mismatch`). The variant is never normalised into
the approved key. The continuity finding is unchanged: the extract's Calgary
Downtown stream is still missing from one of its city's capture events, as
shown (anonymously, counts only) in the observed-stream health table of
[`pricing_readiness_baseline.md`](pricing_readiness_baseline.md).
