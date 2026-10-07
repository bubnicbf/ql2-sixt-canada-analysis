# Location roles, comparison pairs and Vancouver location identity governance decisions

| | |
| --- | --- |
| Responsible authority | `LOCATION_ROLE_ASSIGNMENTS`: business owner. `VALID_LOCATION_COMPARISON_PAIRS`: business owner. `VANCOUVER_LOCATION_IDENTITY`: collection owner (location-identity authority) |
| Provenance | Direct written governance decisions supplied by the repository owner |
| Decision date | 2026-10-06 (date the decisions were supplied and recorded in this repository) |
| Effective scope | The current analyzed dataset and subsequent collections, until superseded by a new versioned authority decision |
| Meeting date, participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v4`](../pricing_authorities/v4.toml) (schema 2) |
| Depends on | The approved exhaustive seven-stream source contract ([expected-stream governance](expected-stream-governance-2026-10-06.md)) |

This page is the durable, repository-local authority reference for the three
location decisions approved in `pricing-authorities-v4`. It records only the
decisions as supplied. No personal name, meeting participant, organization or
ticket was supplied, and none is implied. It contains no source rows, job
identifiers, timestamps, prices, offer data or behavioural comparison
results; the only stream values on this page are approved source keys.

## Decision: `LOCATION_ROLE_ASSIGNMENTS` - APPROVED

Every approved expected source stream has exactly one analytical role:

| City | Location | Role |
| --- | --- | --- |
| `Calgary` | `Downtown` | `DOWNTOWN` |
| `Calgary` | `Int Airport` | `AIRPORT` |
| `Toronto` | `Downtown` | `DOWNTOWN` |
| `Toronto` | `Int Airport` | `AIRPORT` |
| `Vancouver` | `Downtown` | `DOWNTOWN` |
| `Vancouver` | `Int Airport` | `AIRPORT` |
| `Vancouver` | `Thurlow` | `DOWNTOWN` |

The roles describe how each approved source stream participates in pricing
analysis. They do not alter the exact source keys or the exhaustive
source-stream contract. Valid roles are `AIRPORT`, `DOWNTOWN` and `OTHER`.

## Decision: `VALID_LOCATION_COMPARISON_PAIRS` - APPROVED

Exactly these within-city airport/downtown pricing comparisons are approved:

| Airport | Downtown |
| --- | --- |
| `Calgary / Int Airport` | `Calgary / Downtown` |
| `Toronto / Int Airport` | `Toronto / Downtown` |
| `Vancouver / Int Airport` | canonical `Vancouver / Downtown` |

Not permitted: cross-city comparisons; airport-to-airport or
downtown-to-downtown comparisons; comparisons involving `OTHER` locations;
self-comparisons; pairs inferred from available rows or naming conventions;
Vancouver `Downtown` versus Vancouver `Thurlow` as an independent comparison;
a separate airport-versus-`Thurlow` comparison once `Thurlow` is
canonicalized to `Downtown`; duplicated or reversed copies of an approved
pair.

## Decision: `VANCOUVER_LOCATION_IDENTITY` - APPROVED, state `CONFIRMED_ALIAS`

Vancouver `Downtown` and Vancouver `Thurlow` represent the same governed
location identity for this project. The canonical location is
**`Vancouver / Downtown`**:

- `Vancouver / Downtown` maps to canonical `Vancouver / Downtown`.
- `Vancouver / Thurlow` maps to canonical `Vancouver / Downtown`.

Raw source streams and analytical canonical identity stay separate:

- Both raw source labels must be preserved unchanged for traceability.
- Both exact source streams remain independently required by the exhaustive
  source-stream contract; one raw stream never satisfies coverage for the
  other, and canonicalization never hides a missing raw stream.
- The two streams are combined only through this approved location-identity
  policy, after exact source-key validation - never by concatenation, string
  rewriting, case normalization or behaviour-based grouping.
- They are not independent locations for pricing comparisons; the pricing
  comparison layer uses the canonical Vancouver `Downtown` identity once.
- Neither key may canonicalize into another city, and neither may
  canonicalize to Vancouver `Int Airport`.
- Behavioural similarity or dissimilarity, including duplicate-offer
  behaviour, may support diagnostics but never establishes, determines or
  overrides this identity decision.
- Conflicting authoritative location metadata (a mapping defect) still blocks
  the policy.

## Change control

These decisions apply to the current analyzed dataset and remain effective
until they are replaced by a new versioned authority decision. Any change to
a role, a comparison pair or the Vancouver identity requires a new
authority-record version; committed record versions are never edited.

## Scope

This page approves only `LOCATION_ROLE_ASSIGNMENTS`,
`VALID_LOCATION_COMPARISON_PAIRS` and `VANCOUVER_LOCATION_IDENTITY`. It does
**not** resolve the collection schedule (capture timestamp, expected
periods, sharing model, exceptions), temporal rules (timezones, ordering,
tolerances), reporting-day rules, scrape-date or cleaned-date semantics, or
rental-date rules; those remain `PROPOSED` and blocking. It also does not
decide how offers from the two aliased Vancouver source streams are combined
into one analytical pricing population; until that is decided, the
combination stays an explicit blocker.

Approval does not make the dataset pricing ready. The implementation is
`ql2_sixt_canada_analysis.location_authority`, with its own tests.
