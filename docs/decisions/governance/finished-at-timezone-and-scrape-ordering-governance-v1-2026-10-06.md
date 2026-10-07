# Finished-at time zone and scrape/finish ordering governance decisions - version 1

| | |
| --- | --- |
| Responsible authority | `FINISHED_AT_TIMEZONE`: collection owner (authority role: collection owner or supplier). `SCRAPED_FINISHED_ORDERING`: collection owner (authority role: collection owner or supplier). `SCRAPED_FINISHED_TOLERANCE`: collection owner and business owner (joint) |
| Provenance | Direct written governance decisions supplied by the repository owner |
| Decision date | 2026-10-06 (date the decisions were supplied and recorded in this repository) |
| Effective scope | The current dataset and subsequent collections until superseded by a new versioned authority decision |
| Meeting date, participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v6`](../pricing_authorities/v6.toml) (schema 3) |
| Affected fields | `jobs.finished_at`, `jobs.city`, `cars.job_finished_at`, `cars.city`, `cars.scraped_at` |

This page is the durable, repository-local authority reference for the
city-local finish-time policy and the scrape/finish ordering approved in
`pricing-authorities-v6`. It records only the decisions as supplied. No
personal name, meeting participant, organization or ticket was supplied, and
none is implied. It contains no job identifiers, source timestamps, prices,
rows or offers.

## Decision: `FINISHED_AT_TIMEZONE` - APPROVED (confirmed; unchanged resolution)

The naive `jobs.finished_at` values and their repeated copies
`cars.job_finished_at` are local wall-clock times in the time zone of the
**parent job city** (`jobs.city`, exact value). The exhaustive map is:

| Source city (exact) | IANA zone |
| --- | --- |
| `calgary` | `America/Edmonton` |
| `toronto` | `America/Toronto` |
| `vancouver` | `America/Vancouver` |

- The lowercase source city spellings are intentional and exact. A city that
  is missing, blank, unknown, or differs only by case or whitespace fails
  closed; a map with a missing or an unexpected city, or a zone that is not a
  valid IANA region zone, is invalid.
- Only IANA zone identifiers are used, with the installed IANA time-zone
  database, so the daylight-saving rules of the date being interpreted apply.
  Fixed offsets, abbreviations (EST, EDT, MST, MDT, PST, PDT), the computer's
  local zone, one global Canadian zone and any zone inferred from the detail
  location label are not used.
- The parent `jobs.city` selects the zone for both `jobs.finished_at` and the
  linked `cars.job_finished_at`. A detail row's own city must agree with its
  linked parent's city before its repeated finish time can be trusted.

This is the same map the per-stream schedule decision
([collection-schedule governance](collection-schedule-governance-v1-2026-10-06.md),
record `pricing-authorities-v5`) approved; this page confirms it and adds the
interpretation rules below. It is not a second, different resolution.

### Canonical normalization

For `jobs.finished_at`: preserve the exact raw value; parse it with the
approved source format; read the exact parent `jobs.city`; resolve the city
through the map; interpret the naive value as a wall-clock time in that IANA
zone; resolve it to an absolute instant; convert it to UTC and keep a
timezone-aware UTC value for validation and analysis; serialize the
presentation form as `YYYYMMDDTHHMMSSZ`. `cars.job_finished_at` is interpreted
the same way with the zone of its linked parent's city, after the
parent/detail city agreement has been validated.

- **Raw values are preserved.** Source columns are never overwritten;
  normalized values live in separate derived fields or typed results.
- **Full precision.** The internal UTC value keeps the full parsed precision
  (including fractional seconds); replication and ordering checks use it. The
  whole-second `YYYYMMDDTHHMMSSZ` string is a presentation form only and never
  the sole identity of a timestamp: two different sub-second instants stay
  different even when their strings match.
- **Ambiguous local times** (a repeated fall-back hour, with no offset or fold
  in the source) fail closed; neither the first nor the second occurrence is
  chosen.
- **Nonexistent local times** (a spring-forward gap) fail closed; nothing is
  shifted forward or backward.
- Unresolved times are reported only as aggregate counts.

`cars.scraped_at` keeps its established source-designator policy: the `MST`
designator remains fixed UTC-07:00 unless superseded by a separate authority
decision. It is never reinterpreted as market-local time, and it is converted
to a timezone-aware UTC instant before any comparison.

## Decision: `SCRAPED_FINISHED_ORDERING` - APPROVED

For every trusted linked detail row, `cars.scraped_at` must be **earlier than
or equal to** `jobs.finished_at`. **Equality is allowed.** An individual offer
row is scraped during its parent collection job, and the job cannot finish
before its detail observations. All comparisons use resolved UTC instants -
never formatted strings, canonical strings or naive wall-clock values from
different zone bases.

**Parent/detail replication** is also required: `cars.job_finished_at` must
equal its linked parent's `jobs.finished_at` wall-clock value, both resolved
in the parent city's zone, and both must be the same full-precision UTC
instant. A matching seconds-resolution string is not sufficient. A mismatch
is never repaired by copying the parent value into the detail row.

**City integrity is a prerequisite:** a linked detail row whose city differs
from its parent's breaks city integrity and is not trusted for replication or
ordering; missing, orphaned or untrusted linkage likewise leaves a row
unassessable. Invalid, unresolved or unassessable rows are counted, never
dropped.

## Decision: `SCRAPED_FINISHED_TOLERANCE` - APPROVED, zero seconds

The tolerance is **zero seconds**: the ordering is strict except that equality
is allowed. A `cars.scraped_at` instant later than the linked
`jobs.finished_at` instant by any positive amount fails. No grace period is
applied. If operational evidence later supports clock skew or another
tolerance, it must be approved in a new immutable authority-record version.

## Out of scope

These decisions do not approve `REPORTING_DAY_SOURCE`,
`REPORTING_DAY_TIMEZONE`, `SCRAPE_DATE_SEMANTICS` or `DATE_CLEAN_SEMANTICS`,
which remain `PROPOSED`. `cars.date_clean` is preserved as a raw field and is
not used as a pricing date while its semantics are unresolved.

## Change control

Any change to the city map, the zone selector, the normalization, the
ordering, the equality policy or the tolerance requires a new authority-record
version; committed record versions are never edited. These decisions do not
make the dataset pricing ready.
