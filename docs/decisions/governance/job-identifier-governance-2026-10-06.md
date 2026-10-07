# Job-identifier and offer-position governance decisions

| | |
| --- | --- |
| Responsible authority | Collection owner (collection-governance decision) |
| Provenance | Outcome of a collection-governance meeting, supplied by the repository owner |
| Decision date | 2026-10-06 (date the decisions were recorded in this repository) |
| Meeting date | Not supplied |
| Participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v2`](../pricing_authorities/v2.toml) (schema 2) |
| Affected fields | `jobs.job_id`, `cars.job_id`, `cars.row_index` |

This page is the durable, repository-local authority reference for the four
job-identifier decisions approved in `pricing-authorities-v2`. It records only
the conclusions supplied from the governance meeting. No meeting date,
participant, organization, ticket or other provenance was supplied, and none
is implied. It contains no source rows, identifiers, timestamps, prices or
other source-level values, and deliberately gives no example identifiers.

## Decisions

1. `jobs.job_id` and `cars.job_id` represent the same logical source key.
2. The logical source type of `job_id` is text: for analytical purposes it is
   an opaque text identifier, regardless of the upstream physical storage type.
3. A `cars.job_id` value ending in exactly `.0` is a spreadsheet (Excel)
   serialization defect and may be repaired, for linkage only, by removing
   that suffix.
4. Leading zeros are significant and must be preserved.
5. Legitimate identifiers may contain letters, hyphens and other textual
   characters (punctuation included); they are significant.
6. `job_id` must never be parsed through an integer or floating-point
   representation. Non-zero fractional numeric representations must not be
   silently normalized as equivalent identifiers.
7. The raw source identifier must be preserved, unchanged, separately from
   any derived linkage key; a separate derived linkage key is used for joins,
   reconciliation, uniqueness and downstream trust decisions.
8. `cars.row_index` is conceptually an integer offer position. A value ending
   in exactly `.0` is caused by the same spreadsheet serialization problem and
   may be repaired to its integer representation.
9. The raw CSV files must not be edited or replaced.
10. Future upstream exports should serialize `job_id` as text and `row_index`
    as an integer; the repository must safely support the current historical
    export.
11. Raw-data observations alone are not authority.

## Approved normalization boundaries

- **Job linkage key, exact match first.** A detail value that exactly equals a
  job identifier links to that job; the jobs-side key is the raw value
  unchanged.
- **Single legacy repair.** Only a detail value consisting entirely of ASCII
  digits followed by exactly one `.0` may also be matched with that final
  `.0` removed; every preceding character, including leading zeros, is kept.
  The repaired candidate must identify exactly one job.
- **Offer position.** Non-negative integers, ASCII-digit text and ASCII-digit
  text followed by exactly one `.0` (without floating-point coercion of text)
  yield the integer offer position.

## Invalid and ambiguous cases (never repaired)

- Missing or whitespace-only identifiers are invalid linkage keys.
- Surrounding whitespace, signs, exponent notation, non-zero or multi-digit
  fractions (anything other than exactly one `.0` after ASCII digits), case
  differences and alphanumeric text ending in `.0` are never rewritten; such
  values can match only exactly.
- Missing, negative, fractional, signed, exponent, whitespace-padded,
  non-finite, boolean or arbitrary-text offer positions are invalid.

## Collision and ambiguity behaviour

Linkage blocks - the derived key stays missing and a typed blocker is
raised - when the exact and repaired candidates name different jobs, a
candidate matches a duplicated job key, no job matches, or two different raw
representations resolve to the same job. Nothing is guessed, and duplicate
derived offer positions remain a key-contract failure.

## Scope

This record approves only the four job-identifier decisions above. It does
**not** approve the expected-stream universe or spellings, location roles or
comparison pairs, the Vancouver location identity, the collection schedule or
its exceptions, temporal rules (timezones, ordering, tolerances),
reporting-day rules, scrape-date or cleaned-date semantics, or rental-date
validity and agreement rules; those remain `PROPOSED` and blocking.

Approval does not change production behaviour by itself: the implementation
is `ql2_sixt_canada_analysis.job_linkage`, with its own tests.
