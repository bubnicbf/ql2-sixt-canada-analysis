# Pricing-authority decision records

This directory holds the versioned record of every **external decision** the
dataset needs before it can be called pricing ready: job-identifier rules,
the expected stream universe and its spelling, location roles and comparison
pairs, the Vancouver location identity, the collection schedule, timestamp and
reporting-day semantics, and rental-date validity and parent/detail
agreements (22 atomic decisions, `DecisionId` in
`src/ql2_sixt_canada_analysis/authority_decisions.py`).

| File | Purpose |
| --- | --- |
| [`v1.toml`](v1.toml) | Revision 1 (schema 1, historical, unchanged) - every decision `PROPOSED` (no attributable authority found). |
| [`v2.toml`](v2.toml) | Revision 2 (schema 2, historical, unchanged) - supersedes v1; the four job-identifier decisions are `APPROVED` by the collection owner, the other 18 remain `PROPOSED`. |
| [`v3.toml`](v3.toml) | Revision 3 (schema 2, historical, unchanged) - supersedes v2; keeps the four job-identifier approvals and approves `EXPECTED_STREAM_UNIVERSE` (`EXHAUSTIVE`, seven streams; collection owner and business owner) and `EXPECTED_STREAM_SOURCE_SPELLING` (collection owner); the other 16 remain `PROPOSED`. |
| [`v4.toml`](v4.toml) | Revision 4 (schema 2, historical, unchanged) - supersedes v3; keeps the six earlier approvals and approves `LOCATION_ROLE_ASSIGNMENTS` and `VALID_LOCATION_COMPARISON_PAIRS` (business owner) and `VANCOUVER_LOCATION_IDENTITY` (`CONFIRMED_ALIAS`, canonical `Vancouver / Downtown` in the display-style spelling of that revision; collection owner); the other 13 remain `PROPOSED`. |
| [`v5.toml`](v5.toml) | Revision 5 (schema 3, historical, unchanged) - supersedes v4; keeps the four job-identifier approvals, supersedes the display-style source spellings with the exact raw source keys (`calgary / Calgary Downtown`, ...; universe, roles, pairs and alias unchanged in meaning) and approves `SCHEDULE_CAPTURE_TIMESTAMP`, `SCHEDULE_EXPECTED_PERIODS`, `SCHEDULE_SHARING_MODEL` (`PER_STREAM`), `SCHEDULE_EXCEPTIONS` (`NO_EXCEPTIONS`; joint) and the per-city `FINISHED_AT_TIMEZONE`; the other 8 remain `PROPOSED`. |
| [`v6.toml`](v6.toml) | Revision 6 (schema 3, historical, unchanged) - supersedes v5; keeps every earlier approval unchanged (the `FINISHED_AT_TIMEZONE` map gains a confirming reference) and approves `SCRAPED_FINISHED_ORDERING` (`cars.scraped_at` earlier than or equal to `jobs.finished_at`; collection owner) and `SCRAPED_FINISHED_TOLERANCE` (zero seconds; collection owner and business owner); the other 6 remain `PROPOSED`. |
| [`v7.toml`](v7.toml) | Revision 7 (schema 3, **current revision**) - supersedes v6; keeps every earlier approval unchanged and approves `RENTAL_DATE_VALIDITY` (`ISO_8601_DATE`, both dates required, return on or after pickup, minimum 0 days, `UNBOUNDED` maximum; collection owner and business owner) and `RENTAL_DATE_PARENT_DETAIL_AGREEMENTS` (four pickup-to-pickup and return-to-return mappings; collection owner); the other 4 remain `PROPOSED`. |
| [`authority_request_checklist.md`](authority_request_checklist.md) | Generated from the current revision: the resolved decisions with their references, and neutral questions for the 4 decisions still blocked on external input. |
| [`../governance/rental-date-validity-and-agreement-governance-v1-2026-10-06.md`](../governance/rental-date-validity-and-agreement-governance-v1-2026-10-06.md) | Durable authority reference for the rental-date validity and parent/detail agreement approvals (direct written decisions supplied by the repository owner, recorded 2026-10-06). |
| [`../governance/job-identifier-governance-2026-10-06.md`](../governance/job-identifier-governance-2026-10-06.md) | Durable authority reference for the job-identifier approvals (collection-governance outcome supplied by the repository owner, recorded 2026-10-06). |
| [`../governance/expected-stream-governance-2026-10-06.md`](../governance/expected-stream-governance-2026-10-06.md) | Durable authority reference for the expected-stream approvals (direct written decisions supplied by the repository owner, recorded 2026-10-06). |
| [`../governance/location-roles-and-identity-governance-2026-10-06.md`](../governance/location-roles-and-identity-governance-2026-10-06.md) | Durable authority reference for the location-role, comparison-pair and Vancouver identity approvals (direct written decisions supplied by the repository owner, recorded 2026-10-06). |
| [`../governance/finished-at-timezone-and-scrape-ordering-governance-v1-2026-10-06.md`](../governance/finished-at-timezone-and-scrape-ordering-governance-v1-2026-10-06.md) | Durable authority reference for the city-local finish-time policy, parent/detail replication and the zero-tolerance scrape/finish ordering (direct written decisions supplied by the repository owner, recorded 2026-10-06). |
| [`../governance/collection-schedule-governance-v1-2026-10-06.md`](../governance/collection-schedule-governance-v1-2026-10-06.md) | Durable authority reference for the per-stream collection schedule (version `per_stream_hourly_v1`), the per-city finish-time zones and the corrected exact source keys (direct written decisions supplied by the repository owner, recorded 2026-10-06). |

## Revision 2: approved job-identifier decisions

`v2.toml` approves, under schema 2, the four job-identifier decisions:

* `JOB_ID_INVALID_NUMERIC_REPRESENTATIONS` - job identifiers are **opaque
  text**: no trimming, no case folding, no numeric parsing; missing and
  whitespace-only values are invalid linkage keys; exact non-blank matches are
  preserved; unknown, unmatched, colliding or ambiguous representations block
  linkage.
* `JOB_ID_LEADING_ZERO_SIGNIFICANCE` - leading zeros are significant.
* `JOB_ID_RAW_AND_LINKAGE_PRESERVATION` - the raw identifier is preserved and
  a separate derived linkage key is required.
* `JOB_ID_DECIMAL_ZERO_EQUIVALENCE` - approved **only** as the legacy repair of
  the historical detail export: exact match first; the single fallback removes
  a final `.0` from a detail value made entirely of ASCII digits and requires a
  unique parent match. The same defect on the offer position is recorded as
  `offer_position = NONNEGATIVE_INTEGER_WITH_LEGACY_DECIMAL_ZERO`.

Schema 1 expressed these four decisions as booleans, which cannot state
opaque-text semantics. Schema 2 replaces only those four shapes with fixed,
explicit policies (`OPAQUE_TEXT_IDENTIFIER_POLICY`,
`LEGACY_DECIMAL_ZERO_REPAIR` in `authority_decisions.py`) and rejects any
contradiction, within one decision or across them. `v1.toml` keeps validating
under schema 1; the shapes never mix. The approvals are implemented
separately and with their own tests in `ql2_sixt_canada_analysis.job_linkage`
(`job_linkage_policy_from_record` builds the policy only when all four are
approved and consistent).

## Revision 3: approved exhaustive source-stream contract

`v3.toml` (superseded by v4 and v5, unchanged) carries the four
job-identifier approvals over unchanged and approves:

* `EXPECTED_STREAM_UNIVERSE` - mode `EXHAUSTIVE`, exactly seven
  `[city, location]` source streams: `Calgary / Downtown`,
  `Calgary / Int Airport`, `Toronto / Downtown`, `Toronto / Int Airport`,
  `Vancouver / Downtown`, `Vancouver / Int Airport`, `Vancouver / Thurlow`.
  No other scheduled stream should exist under this contract. Joint decision
  of the collection owner and the business owner.
* `EXPECTED_STREAM_SOURCE_SPELLING` - those exact spellings are the source
  keys (collection owner). Case, spacing and punctuation are significant;
  the validator requires the spellings to equal the universe exactly
  (duplicates, blank or padded components and malformed pairs are rejected).

Both reference
[`expected-stream-governance-2026-10-06.md`](../governance/expected-stream-governance-2026-10-06.md).
The approved keys come from the supplied written decisions, not from observed
data; the earlier raw-data observations remain in v1 and v2 as history. The
contract applies to the current analyzed dataset and subsequent collections
until a new revision replaces it: any addition, removal, rename or spelling
change of a stream needs a new revision. It does not resolve location roles,
comparison pairs, the Vancouver location identity, the schedule, temporal,
reporting-day or rental-date decisions.

The approval is implemented separately, with its own tests, in
`ql2_sixt_canada_analysis.expected_stream_contract`: the single resolution of
the effective coverage contract (`EXPECTED_LOCATION_COVERAGE`) from the
latest valid approved record. An approved universe without approved
spellings, a missing or invalid record, or a non-exhaustive universe blocks
pricing (`expected_stream_authority_unavailable`,
`expected_stream_universe_not_exhaustive`). Historical downstream codes that
the implementation replaced are mapped in `RETIRED_DOWNSTREAM_CODES` (for
example `expected_streams_minimum_required_not_exhaustive` ->
`expected_stream_universe_not_exhaustive`); committed revisions are never
edited.

## Revision 4: location roles, comparison pairs and the Vancouver alias

`v4.toml` (superseded by v5, unchanged) carries the six
earlier approvals over unchanged and approves, with reference
[`location-roles-and-identity-governance-2026-10-06.md`](../governance/location-roles-and-identity-governance-2026-10-06.md):

* `LOCATION_ROLE_ASSIGNMENTS` (business owner) - exactly one role per
  approved source stream: `Calgary / Downtown` DOWNTOWN,
  `Calgary / Int Airport` AIRPORT, `Toronto / Downtown` DOWNTOWN,
  `Toronto / Int Airport` AIRPORT, `Vancouver / Downtown` DOWNTOWN,
  `Vancouver / Int Airport` AIRPORT, `Vancouver / Thurlow` DOWNTOWN.
* `VALID_LOCATION_COMPARISON_PAIRS` (business owner) - exactly three
  within-city airport/downtown pairs: Calgary `Int Airport` versus `Downtown`,
  Toronto `Int Airport` versus `Downtown`, Vancouver `Int Airport` versus the
  canonical Vancouver `Downtown`.
* `VANCOUVER_LOCATION_IDENTITY` (collection owner) - `CONFIRMED_ALIAS`:
  Vancouver `Downtown` and Vancouver `Thurlow` are one governed location;
  canonical key `Vancouver / Downtown`.

The validator now also rejects: a canonical key that is not one of the two
governed keys (another city, `Vancouver / Int Airport` or any other key);
governed keys outside the approved universe; aliases with different roles;
self-pairs; pairs whose member is a non-canonical alias (a second comparison
through `Thurlow`); pairs resolving to one canonical location; duplicate and
reversed pairs. The roles, pairs and identity are implemented separately,
with their own tests, in `ql2_sixt_canada_analysis.location_authority`
(`vancouver_policy_from_record` builds the existing `LocationIdentityPolicy`;
there is no second alias mechanism). Source coverage still requires both raw
Vancouver streams; canonicalization is analytical only and never hides a
missing raw stream. The schedule, temporal, reporting-day, scrape/clean-date
and rental-date decisions remain `PROPOSED` in v4.

## Revision 5: per-stream collection schedule and corrected source keys

`v5.toml` (superseded by v6, unchanged; **schema 3**) uses the
production loader's schema-3 rules and the reference
[`collection-schedule-governance-v1-2026-10-06.md`](../governance/collection-schedule-governance-v1-2026-10-06.md).
It keeps the four job-identifier approvals unchanged and:

* **supersedes** the display-style spellings of v3 and v4 with the exact raw
  source keys `calgary / Calgary Downtown`, `calgary / Calgary Int Airport`,
  `toronto / Toronto Downtown`, `toronto / Toronto Int Airport`,
  `vancouver / Vancouver Downtown`, `vancouver / Vancouver Int Airport` and
  `vancouver / Vancouver Thurlow` (`EXPECTED_STREAM_SOURCE_SPELLING`,
  collection owner; an evidence note keeps the traceability to the superseded
  spelling decision). The exhaustive universe, the roles (Downtown keys and
  Thurlow DOWNTOWN, Int Airport keys AIRPORT), the three pairs and the
  Vancouver `CONFIRMED_ALIAS` (canonical `vancouver / Vancouver Downtown`) are
  respelled only; their earlier authorities stay attached;
* approves `SCHEDULE_CAPTURE_TIMESTAMP` - parent `jobs.finished_at`, with
  `cars.job_finished_at` as its detail copy and `cars.scraped_at` as
  observation time only;
* approves `FINISHED_AT_TIMEZONE` as an exhaustive city -> IANA zone map
  (`calgary` `America/Edmonton`, `toronto` `America/Toronto`, `vancouver`
  `America/Vancouver`);
* approves `SCHEDULE_EXPECTED_PERIODS` - schedule version
  `per_stream_hourly_v1`, cadence `PT1H`, phase `LOCAL_TOP_OF_HOUR`, one parent
  job per city period, and for every stream the local window
  `2026-08-27T22:00:00` to `2026-08-31T15:00:00`, end inclusive (90 periods
  per stream, 630 in total, computed from the definitions);
* approves `SCHEDULE_SHARING_MODEL` = `PER_STREAM` and `SCHEDULE_EXCEPTIONS` =
  `NO_EXCEPTIONS` (joint: collection owner and business owner). The Calgary
  Downtown gap is **not** excused; a future exception needs a new revision
  naming the stream, UTC period, failure kind, reason, authority kind,
  governance reference and schedule version.

Schema 3 replaces only the shapes of those five schedule decisions and makes
`VANCOUVER_LOCATION_IDENTITY` name its two `governed_locations` explicitly
(schema 1 and 2 records keep governing the keys as they were spelled then,
`SCHEMA_2_VANCOUVER_GOVERNED_KEYS`). The validator rejects a detail field as
the capture anchor, a missing, extra, misspelled or blank city, `UTC`,
`Etc/` and fixed-offset zones, a global list of instants, any boundary with an
offset or off the top of an hour, an end before the start, a cadence other
than `PT1H`, missing, duplicate or unknown stream schedules, an empty
exceptions list in place of `NO_EXCEPTIONS`, and exceptions naming another
version, stream or an invalid period. The other 8 decisions (scrape/finish
ordering and tolerance, reporting day, scrape-date and cleaned-date semantics,
rental dates) remain `PROPOSED`. The schedule is implemented separately, with
its own tests, in `ql2_sixt_canada_analysis.collection_schedule`. Any schedule
change needs a new versioned record.

## Revision 6: city-local finish times and scrape/finish ordering

`v6.toml` (superseded by v7, unchanged; schema 3) references
[`finished-at-timezone-and-scrape-ordering-governance-v1-2026-10-06.md`](../governance/finished-at-timezone-and-scrape-ordering-governance-v1-2026-10-06.md).
It keeps every earlier approval unchanged and:

* keeps `FINISHED_AT_TIMEZONE` with the same exhaustive map (`calgary`
  `America/Edmonton`, `toronto` `America/Toronto`, `vancouver`
  `America/Vancouver`) - no second, contradictory resolution - and adds the
  confirming collection-owner reference: the naive finish time and its detail
  copy are wall-clock times in the zone of the exact parent `jobs.city`,
  resolved to full-precision UTC instants (`YYYYMMDDTHHMMSSZ` is presentation
  only); ambiguous and nonexistent local times fail closed;
* approves `SCRAPED_FINISHED_ORDERING` (collection owner): `cars.scraped_at`
  earlier than or equal to `jobs.finished_at` for every trusted linked detail
  row, on UTC instants, equality allowed, with `cars.job_finished_at`
  replicating the parent's instant;
* approves `SCRAPED_FINISHED_TOLERANCE` (joint: collection owner and business
  owner): **zero seconds**. Any other tolerance needs a new revision.

Schema 3 now also fixes the ordering shape (earlier `cars.scraped_at`, later
`jobs.finished_at`, a boolean `equal_allowed`, and it requires the per-city
`FINISHED_AT_TIMEZONE` map) and validates every zone with the shared region
rule (`schemas.region_iana_zone`: no fixed offsets, `Etc/`, `UTC`,
abbreviations or legacy links). Schema 1 and 2 records keep their historical
scalar `timezone` shape and are not reinterpreted. `REPORTING_DAY_SOURCE`,
`REPORTING_DAY_TIMEZONE`, `SCRAPE_DATE_SEMANTICS`, `DATE_CLEAN_SEMANTICS` and
the two rental-date decisions remain `PROPOSED`. The policy is implemented
separately, with its own tests, in `ql2_sixt_canada_analysis.temporal_authority`
and `ql2_sixt_canada_analysis.temporal`.

## Revision 7: rental-date validity and parent/detail agreements

`v7.toml` (the current revision, `CURRENT_RECORD_PATH`; schema 3) references
[`rental-date-validity-and-agreement-governance-v1-2026-10-06.md`](../governance/rental-date-validity-and-agreement-governance-v1-2026-10-06.md).
It keeps every earlier approval unchanged and approves:

* `RENTAL_DATE_VALIDITY` (joint: collection owner and business owner) for the
  six fields `jobs.pickup_date`, `jobs.return_date`, `cars.job_pickup_date`,
  `cars.job_return_date`, `cars.pickup_date`, `cars.return_date`:
  `date_format = "ISO_8601_DATE"` (exact `YYYY-MM-DD`), `pickup_required` and
  `return_required` true, `ordering = "RETURN_ON_OR_AFTER_PICKUP"`,
  `equal_dates_allowed = true`, `minimum_duration_days = 0`,
  `maximum_duration_mode = "UNBOUNDED"` (no maximum value);
* `RENTAL_DATE_PARENT_DETAIL_AGREEMENTS` (collection owner):
  `cars.job_pickup_date` and `cars.pickup_date` equal `jobs.pickup_date`;
  `cars.job_return_date` and `cars.return_date` equal `jobs.return_date`.

Schema 3 now fixes both shapes: the validity resolution must state the
format (only `ISO_8601_DATE`), both requiredness flags, an ordering consistent
with the equality flag, a non-negative minimum (zero when equality is allowed)
and an explicit maximum-duration mode - `UNBOUNDED` carries **no** maximum
value (zero, a large number or an observed maximum are not "unbounded"), while
`BOUNDED` requires an integer maximum. The agreements must map every detail
field exactly once, to the parent field of the same kind (pickup to pickup,
return to return); unknown fields, duplicates, swapped kinds and incomplete
coverage are rejected. Schema 1 and 2 shapes are unchanged history. The
policy is implemented separately, with its own tests, in
`ql2_sixt_canada_analysis.rental_dates`. `REPORTING_DAY_SOURCE`,
`REPORTING_DAY_TIMEZONE`, `SCRAPE_DATE_SEMANTICS` and `DATE_CLEAN_SEMANTICS`
remain `PROPOSED`. Any change to the rental-date policy needs a new revision.

## Status semantics

| Status | Meaning |
| --- | --- |
| `PROPOSED` | Unresolved. Names the responsible authority role(s), the exact question, the blocker or gap it affects, and `blocking_external_input = true`. Carries **no** resolution and no authority; it is never consumed as production authority. |
| `APPROVED` | An attributable authority answered. Requires authority provenance and a complete resolution in the shape validated for that decision. |
| `REJECTED` | An attributable authority rejected the proposal. Requires authority provenance and a statement of what was rejected; carries no resolution. |

## Authority requirements

Only three kinds of authority can approve or reject: `SUPPLIER`,
`COLLECTION_OWNER` and `BUSINESS_OWNER`. Each `[[decisions.authority]]` entry
needs the kind, the responsible source (team, organization or role holder),
a **durable reference** (document, ticket or written decision) and optionally
a note and an effective date. The kind must be one of the decision's
responsible roles; a joint decision needs every responsible role.

From schema 2 the durable reference must be **repository-local**: a
sanitized governance document `docs/decisions/governance/<name>.md` that
exists in the repository (relative POSIX path, no `..`, no subdirectories, no
symlink leaving that directory). The production loader
(`load_decision_record` / `parse_decision_record`) and the tests apply the
same check, so an `APPROVED` or `REJECTED` decision whose document is missing
or outside that area fails validation, and no policy is built from it. A
broken reference is fixed by adding the missing document, never by editing a
committed revision; a substantive change of authority content needs a new
revision.

Evidence (`RAW_DATA_OBSERVATION`, `BEHAVIORAL_ANALYSIS`,
`REPOSITORY_IMPLEMENTATION_NOTE`, `ISSUE_OR_REVIEW_NOTE`) may explain why a
question is asked, but it is **never** authority: evidence kinds are refused
in authority entries, and flipping a status to `APPROVED` without authority
and a resolution fails validation. Streams observed in an extract, the
decimal-zero identifier pattern and duplicate-looking Vancouver behaviour are
recorded as evidence only. Never invent a person, organization, document,
ticket, email or approval reference.

The record must not contain source rows, identifiers, timestamps, prices,
vehicle names, offer signatures or other source-level data; free text that
looks like such data is rejected.

## Supersession

Revisions are immutable once committed. To record an answer, copy the
current revision to `v<N+1>.toml`, set `record_version = N+1`,
`record_id = "pricing-authorities-v<N+1>"` and
`supersedes = "pricing-authorities-v<N>"`, update `source_commit`, `created`,
the affected decisions, `summary` and `external_inputs`, point
`CURRENT_RECORD_PATH` at the new file, then regenerate the checklist.
Old revisions stay in place as history.

## Validation

```bash
python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v1.toml
python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v2.toml
python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v3.toml
python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v4.toml
python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v5.toml
python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v6.toml
python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v7.toml
```

prints a sanitized status summary (decision ids, statuses, roles and counts
only) and exits non-zero if the record is invalid. Error messages name
categories, decision ids and field names only, never record values. The test
suite validates every committed revision (v1 to v6 byte-for-byte unchanged)
and checks that the checklist matches `render_authority_request_checklist`
for the current revision.

## Approved decisions are implemented separately

Recording an approval changes no production behaviour by itself: every
approved decision still requires a separately tested production
implementation. Contracts in `schemas.py`, the coverage, schedule and temporal
rules and the pricing gate consume an approved decision only through separate
implementation work with its own tests - for the four job-identifier
decisions, `ql2_sixt_canada_analysis.job_linkage`; for the two expected-stream
decisions, `ql2_sixt_canada_analysis.expected_stream_contract`; for the
location roles, comparison pairs and Vancouver identity,
`ql2_sixt_canada_analysis.location_authority`; for the schedule decisions and
the per-city finish-time zones, `ql2_sixt_canada_analysis.collection_schedule`; for the
finish-time zones, replication and scrape/finish ordering in the temporal contract,
`ql2_sixt_canada_analysis.temporal_authority`; for the rental-date validity and
agreements, `ql2_sixt_canada_analysis.rental_dates`.
`pricing_baseline.baseline_authority_inputs` reads APPROVED decisions only.
