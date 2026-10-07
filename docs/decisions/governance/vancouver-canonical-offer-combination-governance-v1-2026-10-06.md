# Vancouver canonical offer combination governance decision - version 1

| | |
| --- | --- |
| Decision | `CANONICAL_OFFER_COMBINATION` |
| Responsible authority | Collection owner and business owner (joint; roles recorded by the repository, see below) |
| Provenance | Direct written governance decision supplied by the repository owner |
| Decision date | 2026-10-06 (date the decision was supplied and recorded in this repository) |
| Effective scope | The current dataset and subsequent collections until superseded by a new versioned authority decision |
| Meeting date, participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v8`](../pricing_authorities/v8.toml) (schema 4) |

This page is the durable, repository-local authority reference for the offer
combination decision approved in `pricing-authorities-v8`. It records only the
decision as supplied. No personal name, meeting participant, organization or
ticket was supplied, and none is implied. The responsible roles were not
named with the decision; the record assigns it jointly to the collection
owner and the business owner because it combines a collection fact (the
confirmed alias) with an analytical policy (how offers are combined). It
contains no identifiers, rows, timestamps, prices, vehicles or offers.

## Decision - APPROVED

- **Alias policy:** `vancouver` / `Vancouver Downtown` and `vancouver` /
  `Vancouver Thurlow` stay **separately required source streams** (each keeps
  its own schedule and coverage) and both canonicalize to the canonical
  location `vancouver` / `Vancouver Downtown` (the approved
  `VANCOUVER_LOCATION_IDENTITY` alias).
- **Validation before combination:** only offers that pass every
  foundational control (trusted linkage and city integrity, pricing
  eligibility after the Calgary exclusion, rental-date validity and
  agreement, reporting-day agreement, valid price) are combined.
- **Union:** order-independent, with no stream priority; source row order and
  row indexes are never used.
- **Exact semantic identity:** canonical location; scheduled capture period;
  parsed pickup and return dates; the approved product identity (the
  project's established comparison product definition: `car_name`,
  `car_type`, `transmission`, `seats`, `bags`); normalized price; price basis;
  and currency context.
- **Currency:** never assumed. The currency marker and price basis are parsed
  strictly from the price text and the amount must equal the numeric price;
  when they cannot be established the offer is unassessable.
- **Exact duplicates:** observations with the same identity become one
  deterministic canonical row with provenance.
- **Price variation:** observations that share every identity component
  except price are retained as separate observations and flagged as price
  variation - no averaging, no minimum or maximum choice, no discarding.
- **Provenance:** each canonical row carries its raw source location labels,
  the count of contributing observations and whether it is unique or
  deduplicated (offer-level provenance is never printed in reports).
- **Fail closed:** rows with a missing identity component or untrusted
  linkage are unassessable; they are never dropped silently, and missing
  values never compare equal.
