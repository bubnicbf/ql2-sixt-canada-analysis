# Collection schedule (per-stream, hourly) and exact source-key governance decisions - version 1

| | |
| --- | --- |
| Responsible authority | `SCHEDULE_CAPTURE_TIMESTAMP`, `SCHEDULE_EXPECTED_PERIODS`, `SCHEDULE_SHARING_MODEL`, `FINISHED_AT_TIMEZONE`, `EXPECTED_STREAM_SOURCE_SPELLING` (correction): collection owner (authority role: collection owner or supplier). `SCHEDULE_EXCEPTIONS`: collection owner and business owner (joint) |
| Provenance | Direct written governance decisions supplied by the repository owner |
| Decision date | 2026-10-06 (date the decisions were supplied and recorded in this repository) |
| Effective scope | The current dataset and subsequent schedule versions until superseded |
| Meeting date, participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v5`](../pricing_authorities/v5.toml) (schema 3) |
| Schedule version | `per_stream_hourly_v1` |
| Affected fields | `jobs.finished_at`, `jobs.city`, `cars.job_finished_at`, `cars.city`, `cars.location` |

This page is the durable, repository-local authority reference for the
collection-schedule decisions and the corrected exact source keys approved in
`pricing-authorities-v5`. It records only the decisions as supplied. No
personal name, meeting participant, organization, ticket or incident
reference was supplied, and none is implied. It contains no job identifiers,
source timestamps, prices, rows or offers; the only values on this page are
the approved keys, zones and schedule boundaries themselves.

## Decision: corrected exact source keys (`EXPECTED_STREAM_SOURCE_SPELLING`) - APPROVED

The exact raw source keys (`cars.city`, `cars.location`) of the seven
approved streams are:

| City (exact) | Location (exact) | Role | Schedule stream |
| --- | --- | --- | --- |
| `calgary` | `Calgary Downtown` | DOWNTOWN | own schedule |
| `calgary` | `Calgary Int Airport` | AIRPORT | own schedule |
| `toronto` | `Toronto Downtown` | DOWNTOWN | own schedule |
| `toronto` | `Toronto Int Airport` | AIRPORT | own schedule |
| `vancouver` | `Vancouver Downtown` | DOWNTOWN | own schedule |
| `vancouver` | `Vancouver Int Airport` | AIRPORT | own schedule |
| `vancouver` | `Vancouver Thurlow` | DOWNTOWN | own schedule |

These spellings **supersede** the display-style spellings (`Calgary` /
`Downtown` and so on) recorded as exact source keys in the
[expected-stream decision](expected-stream-governance-2026-10-06.md) and
carried by `pricing-authorities-v3` and `-v4`. Those records stay unchanged as
history; this correction is made only through the superseding record.

- The universe stays **EXHAUSTIVE** with exactly these seven keys.
- Location roles are unchanged: the Downtown keys and `Vancouver Thurlow` are
  DOWNTOWN, the Int Airport keys are AIRPORT.
- Valid comparison pairs are unchanged in meaning: `calgary / Calgary Int
  Airport` vs `calgary / Calgary Downtown`; `toronto / Toronto Int Airport` vs
  `toronto / Toronto Downtown`; `vancouver / Vancouver Int Airport` vs the
  canonical `vancouver / Vancouver Downtown`.
- The Vancouver identity decision is unchanged in meaning: the governed keys
  `vancouver / Vancouver Downtown` and `vancouver / Vancouver Thurlow` are a
  `CONFIRMED_ALIAS` with canonical key `vancouver / Vancouver Downtown`.
- Display names are kept separately and never establish coverage or schedule
  identity. Matching is exact: no case, whitespace or alias matching.

## Decision: `SCHEDULE_CAPTURE_TIMESTAMP` - APPROVED

- The schedule capture timestamp is **`jobs.finished_at`**. The parent job is
  the scheduled execution.
- `cars.job_finished_at` is a replicated copy of the parent value; each detail
  copy must agree with its linked parent, and a disagreement fails closed.
- `cars.scraped_at` is the detail observation time only. It is never the
  schedule key, and no first, last, minimum, maximum or average of it is used
  to place a job in a period.
- Every expected stream of the parent job's city is evaluated against the
  parent job's period.

## Decision: `FINISHED_AT_TIMEZONE` - APPROVED (per-city map)

`jobs.finished_at` (and therefore its copy `cars.job_finished_at`) is a naive
local wall-clock value in the zone of the parent job's city
(`jobs.city`, exact value):

| City (exact) | IANA zone |
| --- | --- |
| `calgary` | `America/Edmonton` |
| `toronto` | `America/Toronto` |
| `vancouver` | `America/Vancouver` |

The map is exhaustive. A city that is missing, blank, unknown, misspelled or
not in this table fails closed. Normalization preserves the raw value, parses
it as a naive value, localizes it with the IANA zone, converts it to an aware
UTC instant and serializes it canonically as `YYYYMMDDTHHMMSSZ`. A `Z` is never
simply appended to the local value, and fixed offsets are never used.

The other temporal decisions (scrape/finish ordering and tolerance,
reporting-day rules, scrape-date and cleaned-date semantics, rental-date rules)
are **not** decided here and remain `PROPOSED`.

## Decision: `SCHEDULE_EXPECTED_PERIODS` - APPROVED

- Cadence `PT1H`, phased at the start of every local clock hour.
- Exactly one parent job per city per local hour.
- Window, for every city and every stream: local start
  `2026-08-27T22:00:00`, local end `2026-08-31T15:00:00`, **end boundary
  inclusive**, in the stream's city zone.
- Streams per city: `calgary` - `Calgary Downtown`, `Calgary Int Airport`;
  `toronto` - `Toronto Downtown`, `Toronto Int Airport`; `vancouver` -
  `Vancouver Downtown`, `Vancouver Int Airport`, `Vancouver Thurlow`.
- Expected periods are materialized from these explicit definitions with the
  IANA time-zone database and stored as UTC instants (`YYYYMMDDTHHMMSSZ`) with
  their local start and UTC offset. Duration, boundaries, timezone, stream key
  and materialized periods are explicit; they are never inferred from
  observed jobs.
- This window yields 90 hourly periods per stream (630 stream-periods; 180 for
  Calgary, 180 for Toronto, 270 for Vancouver). These counts are computed from
  the definitions, not configured.

Daylight-saving behaviour (the current window contains no transition, but the
model implements it):

- A **nonexistent** spring-forward local hour is skipped; it is not an expected
  period.
- A **repeated** fall-back local hour yields two expected periods with
  distinct UTC offsets (and UTC instants); they are never deduplicated.
- Times are never shifted automatically, and the offset (or fold) of every
  period is exposed.
- An observed local finish time that is ambiguous or nonexistent in its zone
  fails closed.

## Decision: `SCHEDULE_SHARING_MODEL` - APPROVED, `PER_STREAM`

Each stream has its own schedule definition and its own expected-period set.
One stream's missing coverage does not make another stream unhealthy, and one
stream's presence does not satisfy another (Calgary Airport never satisfies
Calgary Downtown). A future version may change one stream without silently
changing the others. There is no global list of shared UTC instants.

## Decision: `SCHEDULE_EXCEPTIONS` - APPROVED, `NO_EXCEPTIONS`

No scheduled period is cancelled or exempted for any stream. `NO_EXCEPTIONS`
is an explicit decision, not an empty list.

- The Calgary Downtown coverage gap remains a real blocker. No exception is
  created, inferred or excluded for it, it is not marked healthy, and the
  window is not changed to avoid it.
- A future exception requires a new versioned decision with documented
  operational evidence, and must name the stream, the scheduled UTC period, the
  local period and offset, the failure kind, the reason, the responsible
  authority, a durable reference and the schedule version. An exception never
  applies across streams, periods, failure kinds or schedule versions.

Open operational question for the collection owner:

> For the one Calgary hourly job that produced Calgary Airport rows but no Calgary Downtown rows, was the Downtown branch attempted, and what was the outcome?

## Change control

These decisions apply to the current dataset and subsequent schedule versions
until superseded. Any change to the cadence, phase, window, zones, streams,
sharing model, exceptions or source keys requires a new versioned schedule
decision recorded in a new authority-record version; committed record versions
are never edited.

## Scope

This page does not make the dataset pricing ready. The implementation is
`ql2_sixt_canada_analysis.collection_schedule`, with its own tests.
