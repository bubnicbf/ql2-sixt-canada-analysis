# Expected source-stream universe and source-spelling governance decisions

| | |
| --- | --- |
| Responsible authority | `EXPECTED_STREAM_UNIVERSE`: collection owner and business owner (joint). `EXPECTED_STREAM_SOURCE_SPELLING`: collection owner |
| Provenance | Direct written governance decisions supplied by the repository owner |
| Decision date | 2026-10-06 (date the decisions were supplied and recorded in this repository) |
| Effective scope | The current analyzed dataset and subsequent collections, until superseded by a new versioned authority decision |
| Meeting date, participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v3`](../pricing_authorities/v3.toml) (schema 2) |
| Affected fields | `cars.city`, `cars.location` (the source-stream key) |

This page is the durable, repository-local authority reference for the two
expected-stream decisions approved in `pricing-authorities-v3`. It records
only the decisions as supplied. No personal name, meeting participant,
organization or ticket was supplied, and none is implied. It contains no
source rows, job identifiers, timestamps, prices or other source-level
observations; the only stream values on this page are the approved keys
themselves.

## Decision: `EXPECTED_STREAM_UNIVERSE` - APPROVED, mode EXHAUSTIVE

The expected source-stream universe is **EXHAUSTIVE** and is exactly these
seven city/location pairs:

| City | Location |
| --- | --- |
| `Calgary` | `Downtown` |
| `Calgary` | `Int Airport` |
| `Toronto` | `Downtown` |
| `Toronto` | `Int Airport` |
| `Vancouver` | `Downtown` |
| `Vancouver` | `Int Airport` |
| `Vancouver` | `Thurlow` |

No other scheduled stream should exist under this contract. A missing
approved stream and an observed stream outside this list both fail the
contract; an observed stream is never added to the universe because it was
observed.

## Decision: `EXPECTED_STREAM_SOURCE_SPELLING` - APPROVED

The exact source spelling of every approved pair is the spelling in the table
above. Exact source spelling is authoritative:

- Case, spacing and punctuation are significant. Source-key comparison is
  exact: city and location must both equal an approved pair, compared as a
  pair. Values are never lowercased, uppercased, trimmed, whitespace-collapsed
  or punctuation-rewritten to establish a match.
- Raw source values must be preserved unchanged, separately from the exact
  approved source key, from any analytical display label and from any
  governed canonical alias.
- Display labels are separate from source keys and never establish source
  coverage.
- Aliases require a separate approved location-identity policy; they are
  never hidden inside general string normalization and never establish source
  coverage on their own.

Vancouver `Downtown` and Vancouver `Thurlow` are two separate expected source
streams. Whether they are one analytical location is a separate question
(`VANCOUVER_LOCATION_IDENTITY`), which this decision does not answer.

## Change control

This contract applies to the current analyzed dataset and remains effective
until it is replaced by a new versioned authority decision. Any future
addition, removal, rename or spelling change of a stream requires a new
authority-record version; committed record versions are never edited.

## Scope

This page approves only `EXPECTED_STREAM_UNIVERSE` and
`EXPECTED_STREAM_SOURCE_SPELLING`. It does **not** resolve location roles or
comparison pairs, the Vancouver location identity, schedule periods or the
other schedule decisions, schedule exceptions, temporal rules (timezones,
ordering, tolerances), reporting-day rules, scrape-date or cleaned-date
semantics, or rental-date rules; those remain `PROPOSED` and blocking. The
four job-identifier decisions keep their own reference
([job-identifier governance](job-identifier-governance-2026-10-06.md)).

Approval does not make the dataset pricing ready. The implementation is
`ql2_sixt_canada_analysis.expected_stream_contract`, with its own tests.
