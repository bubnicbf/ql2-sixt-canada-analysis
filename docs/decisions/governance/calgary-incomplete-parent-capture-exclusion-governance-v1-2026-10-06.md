# Incomplete Calgary parent capture exclusion governance decision - version 1

| | |
| --- | --- |
| Decision | `SCHEDULE_EXCEPTIONS` - listed exception of failure kind `INCOMPLETE_PARENT_CAPTURE` |
| Responsible authority | Collection owner and business owner (joint) |
| Provenance | Direct written governance decision supplied by the repository owner |
| Decision date | 2026-10-06 (date the decision was supplied and recorded in this repository) |
| Meeting date, participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v8`](../pricing_authorities/v8.toml) (schema 4) |
| Supersedes | The `NO_EXCEPTIONS` resolution of `SCHEDULE_EXCEPTIONS` (records v5 to v7, unchanged as history) |
| Schedule version | `per_stream_hourly_v1` |

This page is the durable, repository-local authority reference for the one
schedule exclusion approved in `pricing-authorities-v8`. It records only the
decision as supplied. No personal name, meeting participant, organization or
ticket was supplied, and none is implied. It contains no job identifier,
price, vehicle or source row; the only values on this page are the governed
city, stream keys and scheduled period themselves.

## Decision - APPROVED

| Field | Value |
| --- | --- |
| Schedule version | `per_stream_hourly_v1` |
| City (exact parent `jobs.city`) | `calgary` |
| Stream keys excluded | `calgary` / `Calgary Downtown` and `calgary` / `Calgary Int Airport` (both streams of the city) |
| Failure kind | `INCOMPLETE_PARENT_CAPTURE` |
| Scheduled period (UTC start) | `20260828T170000Z` (local `2026-08-28 11:00`, `America/Edmonton`, UTC-06:00) |

- **What happened:** the one Calgary parent capture of that scheduled period
  returned rows for the Airport stream but no rows for the Downtown stream.
  The parent collection execution is therefore incomplete.
- **Analytically null:** the whole parent capture is treated as analytically
  null for that one period. **Both** Calgary streams are excluded for that
  period - the Airport stream as well as the Downtown stream - so no partial
  city capture enters pricing.
- **Raw data preserved:** the raw parent job, its raw status and every raw
  detail row linked to it stay in the source files and in ingestion,
  linkage, reconciliation, audit and exception reporting. Nothing is
  deleted, rewritten or repaired, and no Downtown rows are fabricated. The
  capture is *excluded from the pricing population*, not deleted.
- **Pricing boundary:** every detail row linked to that parent capture is
  excluded from price summaries, comparisons, product populations, vehicle
  stability used for pricing, duplicate calculations, offer counts and
  reporting-day cohorts. Exclusions are reported only as aggregate counts.
- **Targeting:** the exclusion is keyed by schedule version, city, scheduled
  period and both stream keys - never by a job identifier. It must match
  exactly one valid parent capture; matching zero or several captures fails
  closed (`schedule_exclusion_unmatched`).
- **Not a general rule:** nothing else is covered. There is no general rule
  that drops incomplete captures, and `STREAM_ABSENT_FROM_CAPTURE` is not
  used for the Airport stream. Any future exclusion needs a new, versioned
  authority decision.

## Expected effect on scheduled coverage

| Measure | Count |
| --- | --- |
| Nominal stream-periods | 630 |
| Excluded stream-periods | 2 |
| Required stream-periods | 628 |
| Covered stream-periods | 628 |
| Unexcused missing stream-periods | 0 |

`calgary` / `Calgary Downtown` and `calgary` / `Calgary Int Airport`: 89
covered periods plus 1 excluded period each; every other stream: 90 covered.
