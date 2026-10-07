# Reporting day and source-date semantics governance decisions - version 1

| | |
| --- | --- |
| Responsible authority | `REPORTING_DAY_SOURCE`, `REPORTING_DAY_TIMEZONE`: business owner. `SCRAPE_DATE_SEMANTICS`, `DATE_CLEAN_SEMANTICS`: collection owner (authority role: collection owner or supplier) |
| Provenance | Direct written governance decisions supplied by the repository owner |
| Decision date | 2026-10-06 (date the decisions were supplied and recorded in this repository) |
| Effective scope | The current dataset and subsequent collections until superseded by a new versioned authority decision |
| Meeting date, participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v8`](../pricing_authorities/v8.toml) (schema 4) |
| Governed fields | `jobs.finished_at`, `jobs.city`, `jobs.scrape_date`, `cars.scrape_date`, `cars.date_clean` |

This page is the durable, repository-local authority reference for the four
date decisions approved in `pricing-authorities-v8`. It records only the
decisions as supplied. No personal name, meeting participant, organization
or ticket was supplied, and none is implied. It contains no identifiers,
rows, timestamps, prices or offers.

## Decision: `REPORTING_DAY_SOURCE` - APPROVED

The reporting day derives from **`jobs.finished_at`**, the parent job's
finish time (resolved with the approved per-city `FINISHED_AT_TIMEZONE`).

## Decision: `REPORTING_DAY_TIMEZONE` - APPROVED (mode `PARENT_CITY`)

The reporting day is the **local calendar date of the finish instant in the
zone of the parent job's city** (exact `jobs.city`):

| City (exact) | IANA zone |
| --- | --- |
| `calgary` | `America/Edmonton` |
| `toronto` | `America/Toronto` |
| `vancouver` | `America/Vancouver` |

- The map is exhaustive; an unknown, blank or misspelled city has no
  reporting day (fails closed), and no default zone is assumed.
- Detail rows receive the reporting day only through their **trusted linked
  parent** (exact city integrity required). It is never derived from
  `cars.scraped_at` or from a detail row's own city or location.

## Decision: `SCRAPE_DATE_SEMANTICS` - APPROVED (`REPORTING_DAY`)

- `jobs.scrape_date` must equal the parent's reporting day, and
  `cars.scrape_date` must equal the linked parent's reporting day.
- Values are parsed strictly as `ISO_8601_DATE` (exact `YYYY-MM-DD` text
  naming a real calendar date); parsed dates are compared, never raw text.
- Raw values are preserved. Nothing is repaired: a missing, invalid,
  mismatched, unlinked, untrusted or unresolvable row fails closed and is
  reported in aggregate.

## Decision: `DATE_CLEAN_SEMANTICS` - APPROVED (`RETIRED_FROM_PRICING`)

- `cars.date_clean` is **retired from pricing**. The field is preserved and
  its presence and parse quality are reported.
- It is never used for the reporting day, never overrides `scrape_date`, is
  never used for grouping, is not required to agree with any other field,
  never creates trust and is never rewritten.
- The temporal contract therefore no longer reports a
  `date_derivation:cars.date_clean` rule as unavailable.

## Derived reporting-day structure

The derived structure keeps, separately from the source columns: the raw
finish value and its full-precision UTC instant, the zone and reporting day,
the raw and parsed scrape dates and their agreement status, and provenance.
Nothing is overwritten. The approved Calgary incomplete-capture exclusion
(see [`calgary-incomplete-parent-capture-exclusion-governance-v1-2026-10-06.md`](calgary-incomplete-parent-capture-exclusion-governance-v1-2026-10-06.md))
is applied after temporal validation and before reporting-day cohorts are
built.
