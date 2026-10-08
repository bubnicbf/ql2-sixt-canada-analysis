# Visible assortment contract - version 1 (definition phase)

| | |
| --- | --- |
| Data-plan scope | Section 5, "Visible assortment" (definitions only) |
| Status | Mixed. Each definition is labelled as derived from approved contracts, analytical, or **PROPOSED - not approved**. Proposed items are Not approved: no stakeholder approval exists or is implied |
| Responsible authority for proposed items | Business owner (unusual-drop policy, timeline persistence); collection owner and business owner jointly (rental-context stratification) |
| Prepared | 2026-10-08, in this repository, from the existing schema and approved records |
| Decision record | None. No authority record (`pricing-authorities-v1` to `v8`) is changed or extended by this document |
| Code contract | [`src/ql2_sixt_canada_analysis/assortment_contract.py`](../../../src/ql2_sixt_canada_analysis/assortment_contract.py) |

This page fixes the definitions the future visible-assortment engine must
use. It is a design proposal, not an approval record. It contains no
identifiers, rows, timestamps, prices, vehicles, offers or real-data values.
Nothing here computes an assortment. The calculation engine, the
unusual-drop detector, the timeline implementation, notebooks and exports
belong to the next phase.

## 1. Purpose and scope

Section 5 of the data plan asks the eventual analysis to:

- count returned products by location and capture;
- identify additions and removals;
- calculate consecutive-capture retention;
- calculate Jaccard similarity;
- identify unusual assortment drops;
- check whether assortment changes coincide with price changes;
- produce an assortment timeline.

This phase settles what each of those terms means, in terms of the
validated objects the repository already produces.

## 2. Source-schema findings

- The raw `cars` rows carry the vehicle attributes (`car_name`,
  `car_type`, `transmission`, `seats`, `bags`), the rental dates, the price
  fields (`price_num`, `price_per_day`), the source location and the
  confidential technical columns (`job_id`, `row_index` and their derived
  keys).
- The established comparison product definition
  (`ANALYSIS_LOCATION_STREAM_COMPARISON.product_columns`) is the five vehicle
  attributes plus `pickup_date` and `return_date`.
- The approved canonical-offer identity (`CANONICAL_OFFER_COMBINATION`,
  record v8) is: canonical location, scheduled capture period, parsed pickup
  and return dates, the approved product identity
  (`APPROVED_PRODUCT_COLUMNS` = the five vehicle attributes), normalized price,
  price basis and currency marker.
- The price-change event identity (`EVENT_IDENTITY_COLUMNS`) is canonical
  city and location, rental dates, the five vehicle attributes, currency and
  price basis. Those semantics belong to price comparison: a unit change
  there is a disappearance plus an appearance.
- The present local snapshot has exactly one rental search context per
  location capture and one price unit. This was checked as an aggregate
  only, and no value is recorded here. The contract does not depend on that
  fact: it fails closed when it is not true.

## 3. Reused approved contracts

| Contract | Reused for |
| --- | --- |
| `run_pricing_pipeline` -> `CanonicalOfferReport.offers` | population (pricing-eligible canonical offers) |
| `PricingPopulation`, governed `INCOMPLETE_PARENT_CAPTURE` exclusion (v8) | rows that never enter; excluded captures are breaks |
| `VANCOUVER_LOCATION_IDENTITY` + `CANONICAL_OFFER_COMBINATION` (v8) | Downtown and Thurlow form one canonical location |
| Per-stream schedule (v5+) and `CapturePeriodIndex` | `scheduled_capture_period`, the capture grid |
| `capture_timelines`, `CaptureInterval`, `IntervalBreak` | consecutive captures and typed breaks |
| `APPROVED_PRODUCT_COLUMNS` | product identity |
| `TerminalOutcome` of the price-change engine | which price outcomes are price changes |
| `approved_canonical_locations`, `frame_binding` | evidence identity |

## 4. Exact definitions

The notation is `P` for the previous product set and `C` for the current
product set, over one valid interval.

| # | Term | Definition | Status |
| --- | --- | --- | --- |
| 1 | Population | The pricing-eligible canonical offers of one `run_pricing_pipeline` result. Never raw `cars` rows, matched airport/downtown pairs, row order, generated CSV or report files. The governed Calgary exclusion and every ineligible row never enter. An unassessable canonical offer blocks the whole analysis, because readiness fails. Vancouver Downtown and Thurlow offers are one canonical location after the approved combination. | derived |
| 2 | Location | The exact canonical pair `(canonical_city, canonical_location)`. Source streams remain a coverage concept. Source labels are provenance only, and they never split or merge a location. Display labels and identifiers are never keys. | derived |
| 3 | Capture | `scheduled_capture_period`, the authority-backed scheduled UTC period. Never `job_id`, `row_index`, source row order, `scrape_date`, `date_clean`, raw scrape timestamps, `finished_at`, the reporting day or "last seen". The timeline grid comes from the schedule (`capture_timelines`), so an eligible capture with zero offers is a visible empty set. | derived |
| 4 | Consecutive captures | A `CaptureInterval`: two schedule-adjacent eligible captures exactly one hour apart with the same contributing source streams. Every other adjacent pair is a typed break: `governed_exclusion`, `missing_capture`, `not_one_hour` or `source_streams_changed`. A rental-context change between the endpoints is the additional break `rental_context_changed`. The first eligible capture of a run seeds its set and has no comparison. | derived (plus the rental-context break: analytical) |
| 5 | Product | One distinct value of `APPROVED_PRODUCT_COLUMNS` (`car_name`, `car_type`, `transmission`, `seats`, `bags`), compared exactly as the canonical-offer assessment parsed them. | derived |
| 6 | Returned-product count | `|S|` for the set `S` of distinct valid product identities in one location capture and rental context. | analytical |
| 7 | Additions, removals, retained | Retained: `P & C`. Additions: `C - P`. Removals: `P - C`. Defined over a valid interval only. | analytical |
| 8 | Retention | `|P & C| / |P|`, a float in [0, 1]. When `|P| = 0` the value is `zero_denominator` with no number. It is directional: it measures how much of the previous set survived. | analytical |
| 9 | Jaccard similarity | `|P & C| / |P | C|`, a float in [0, 1]. When both sets are empty the value is `zero_denominator` with no number. It is symmetric. | analytical |
| 10 | Drop signals | `net_change = |C| - |P|`. `absolute_drop = max(|P| - |C|, 0)`. `drop_rate = absolute_drop / |P|` when `|P| > 0`, else `zero_denominator`. `removal_count`, `addition_count`, whether a valid interval exists, and whether the denominator is usable. | analytical |
| 11 | Unusual drop | No approved baseline, history window, method or threshold exists. Classification fails closed. | **PROPOSED - not approved** |
| 12 | Coincidence | Temporal association, never causation. It requires the same canonical location and the same previous and current scheduled periods as the price-change interval. | analytical |
| 13 | Timeline | One row per canonical location and scheduled capture, with the fixed schema in section 8. | schema: analytical; persistence: **PROPOSED** |

## 5. Product-identity decision

**Adopted.** A product is the approved product identity: the five vehicle
attributes.

- **Rental dates** (`pickup_date`, `return_date`) are the *search context*
  of a product set, not product identity. A set is keyed by location,
  capture and context. Exactly one context per location capture is
  required; several contexts fail closed (`multiple_rental_contexts`). A
  context change between consecutive captures is a break, never a mass
  removal plus addition. Collapsing contexts would mix different searches;
  splitting products by context would make the same car look like two
  products.
- **Currency and price basis** are *price-comparison* identity only. A
  product offered with a different unit is still the same visible product.
  Only the price-change engine treats a unit change as a disappearance plus
  an appearance.
- **Never in the product identity:** price, `job_id`, `row_index`, linkage
  keys, offer position, capture timestamps, provenance labels or source
  ordering.

**Alternatives considered.**

- (a) Reuse `EVENT_IDENTITY_COLUMNS`: rejected, because unit changes would
  inflate additions and removals.
- (b) Use the comparison product definition including dates as the identity:
  rejected, because a context change would read as a full turnover.
- (c) Stratify by context when several exist: kept as the **proposed**
  multi-context policy, pending collection-owner and business-owner
  authority.

**Missing identity.** Canonical offers cannot contain a missing identity
component, because such a row is unassessable and blocks readiness. Missing
components never compare equal through a sentinel or imputation, and the
evidence gate reports `product_identity_incomplete` if one appears.

## 6. Duplicates, price variants, units, zero-offer captures and breaks

- Exact duplicate observations are already one canonical offer (canonical
  deduplication), so they count once.
- Several canonical offers of one product at one capture count as one
  product. These can be price-variation groups, several prices, or several
  units.
- Alias provenance (Downtown, Thurlow or both) never duplicates a product.
- An eligible capture with zero offers has `returned_product_count = 0` and
  stays on the timeline. Its intervals give removals and later additions.
  It is an observed condition, never presumed to be a supplier withdrawal.
- A governed exclusion or missing capture has no set
  (`capture_not_eligible`). Comparisons never span it, and no addition or
  removal is attributed to it.

## 7. Coincidence with price changes

- **Price changes** are only the `increase` and `decrease` outcomes of the
  validated price-change candidates. `unchanged`, `ambiguous`, `appeared`
  and `disappeared` are never price changes.
- **Location-interval coincidence:** within the same canonical location and
  exact interval, an assortment change (`addition_count + removal_count >
  0`) together with at least one price change.
- **Retained-product price change:** every increase or decrease needs one
  comparable offer of the same identity and unit at both endpoints, so its
  product is necessarily retained. Additions and removals are never price
  changes, because they lack comparable prices at both endpoints.
- **Direction** is kept as separate `price_increase_count` and
  `price_decrease_count`. `falling_assortment_with_price_increase` means
  `net_change < 0` together with at least one increase in the same
  interval. It is descriptive evidence only.

## 8. Timeline grain and fixed schema

One row per canonical location and scheduled capture, on the schedule grid
(`ASSORTMENT_TIMELINE_COLUMNS`, in order):

`canonical_city, canonical_location, scheduled_capture_period, capture_state,
contributing_stream_count, returned_product_count, has_previous_interval,
previous_scheduled_capture_period, previous_product_count, retained_count,
addition_count, removal_count, retention, retention_denominator_status,
jaccard_similarity, jaccard_denominator_status, net_change, absolute_drop,
drop_rate, drop_rate_denominator_status, interval_break_reason,
price_increase_count, price_decrease_count, assortment_change, price_change,
assortment_price_coincidence, falling_assortment_with_price_increase,
anomaly_policy_status, unusual_drop, assessability_status`.

- **Status fields.** The denominator statuses are `defined`,
  `zero_denominator` or `not_assessable`. `assessability_status` is
  `assessed`, `seed_capture`, `interval_break`, `capture_not_eligible` or
  `blocked`.
- **Unusual-drop fields.** `anomaly_policy_status` stays `unavailable`, and
  `unusual_drop` stays empty, until an approved policy exists.
- **Aggregate only.** The schema holds no product, price, unit, context or
  provenance field.

## 9. Confidentiality and persistence

- Product identities and per-product additions and removals stay in memory,
  with the same classification as the price-change candidate frame. They are
  never printed, held in a report, or written.
- The aggregate timeline holds counts, ratios, enum statuses and approved
  canonical keys and periods only. Real aggregates remain confidential local
  artifacts.
- Whether, where and in what format the timeline may be persisted is
  **PROPOSED - not approved**: a business-owner decision.

## 10. Fail-closed behaviour

`assortment_evidence_blockers(run)` returns typed blockers. The engine may
start only when it returns an empty tuple. The blockers are:

- `pricing_not_ready`: readiness failed or is bound to other reports;
- `canonical_offers_not_ready`;
- `schedule_evidence_invalid`: invalid, or missing capture-period or
  exclusion evidence;
- `evidence_binding_mismatch`;
- `location_authority_unavailable`;
- `product_identity_incomplete`;
- `multiple_rental_contexts`.

`VisibleAssortmentDefinition` rejects any of the following on construction:

- empty, duplicate or overlapping column groups;
- non-canonical-offer columns;
- prohibited columns in identity;
- a step other than one hour;
- missing event breaks;
- price outcomes other than increase and decrease;
- product, price or provenance fields in the timeline.

`classify_unusual_drop` raises `AnomalyPolicyUnavailableError` without an
approved policy.

## 11. Authority analysis

| Definition | Basis | Status |
| --- | --- | --- |
| population, location, capture, consecutive interval, product identity | records v5-v8 and the existing pipeline, event and canonical-offer contracts | derived from approved contracts |
| single rental context per capture; context-change break; returned-product count; set formulas; retention; Jaccard; drop signals; coincidence; timeline schema | repository analytical definitions (no business judgement) | analytical |
| multi-context stratification | collection owner and business owner | **PROPOSED - not approved** |
| unusual-drop policy | business owner | **PROPOSED - not approved** |
| timeline persistence and format | business owner | **PROPOSED - not approved** |

**Proposed decision fields** for a future versioned record, mirroring the
existing decision shape:

- `UNUSUAL_ASSORTMENT_DROP_POLICY`, from the business owner:
  - method: for example, a per-location robust baseline (median and
    MAD-scaled deviation of `drop_rate` over valid intervals) or a fixed
    materiality floor on `absolute_drop` and `drop_rate`;
  - minimum history: valid intervals before any classification;
  - grouping, for example by location only, given the short window;
  - threshold values;
  - scope and version.
- `ASSORTMENT_RENTAL_CONTEXT_POLICY`, from the collection owner and business
  owner: whether several contexts per capture are stratified or rejected.
- `ASSORTMENT_TIMELINE_PERSISTENCE`, from the business owner: whether a local
  aggregate export is permitted, its location and its retention.

These are candidates only, and no value is approved.

**An observed drop is not proof of a cause.** These four claims are kept
distinct:

- an *observed drop* (`absolute_drop > 0`);
- a *statistically unusual drop*, which requires the approved policy;
- a *suspected collection failure*, which requires collection-owner
  corroboration;
- a *genuine supplier assortment withdrawal*, which requires supplier
  corroboration.

The data alone supports only the first.

## 12. Interpretation limitations

- The collection spans roughly 90 hours, so there is no basis for seasonal
  or long-term baselines.
- Hourly captures are repeated measurements.
- Visible assortment is what the collection returned, which is not
  necessarily what the supplier offered. Empty or shrunken captures are
  anomaly indicators, not proof.

## 13. Implementation boundary for the next phase

The next phase should implement the following, without changing these
definitions:

- the engine (sets per location capture, `compare_assortment` over
  `capture_timelines` intervals);
- the timeline rows in `ASSORTMENT_TIMELINE_COLUMNS` order;
- coincidence counts joined from the validated price-change candidates;
- reconciliation;
- tests.

The detector, presentation and exports wait for the proposed decisions.

## 14. Reconciliation to Section 5

| Data-plan bullet | Definition | State |
| --- | --- | --- |
| Count returned products by location and capture | #2, #3, #5, #6 | defined |
| Identify additions and removals | #7 over #4 | defined |
| Calculate consecutive-capture retention | #8 | defined |
| Calculate Jaccard similarity | #9 | defined |
| Identify unusual assortment drops | #10 signals defined; #11 policy | signals defined; alert policy **PROPOSED - not approved** |
| Check whether assortment changes coincide with price changes | #12, section 7 | defined |
| Produce an assortment timeline | #13, section 8 | schema defined; persistence **PROPOSED - not approved** |

## 15. Implementation correspondence (calculation engine, added 2026-10-08)

This section records how the engine in
[`src/ql2_sixt_canada_analysis/visible_assortment.py`](../../../src/ql2_sixt_canada_analysis/visible_assortment.py)
implements the definitions above. It does not change any definition, status
or authority. The proposed items remain **PROPOSED - not approved**.

- **Grid.** The engine reuses the capture timelines held by the validated
  price-change result, so assortment and price coincidence share one grid
  and the same interval keys. `LocationCaptureTimeline.adjacent_pairs` now
  exposes the existing interval-or-break decision for each adjacent pair, in
  order. Its rules are unchanged.
- **Row semantics.** These follow the schema in section 8:
  - A row that is not assessed has every interval field null and
    `not_assessable` denominators.
  - `interval_break_reason` names the break of the pair ending at that row.
    This includes the ineligible capture's own row.
  - `has_previous_interval` is true only on assessed rows.
- **Membership detail.** The membership detail allowed by section 9 is an
  in-memory frame (`MEMBERSHIP_COLUMNS`) with one `retained`, `added` or
  `removed` row per product per assessed interval. It holds no price.
- **Coincidence evidence.** For each interval, the price-change candidates
  projected onto location, rental context and product must equal the union
  of the two endpoint sets. Every increase or decrease must be a retained
  product. Any disagreement fails closed.
- **Counting grain.** `price_increase_count` and `price_decrease_count` count
  price-change candidates, whose identity adds currency and price basis to
  the product. `returned_product_count`, `retained_count`, `addition_count`
  and `removal_count` count distinct visible products. One retained product
  can carry several unit-specific candidates, for example separate CA$ and
  US$ changes, so the price counts may exceed `retained_count`. The proof
  that every changed candidate projects to a retained product of its interval
  uses candidate identities (`validate_price_coincidence`, run whenever price
  evidence is attached to a result). The aggregate row check only requires a
  retained product whenever a price change is counted. This is an
  implementation correction (2026-10-08), not a new rule.
- **Blocker categories.** `AssortmentBlocker` gains four engine categories:
  `unknown_canonical_location`, `capture_evidence_inconsistent`,
  `price_change_evidence_invalid` and `reconciliation_failed`.
- **Policy interface.** `UnusualDropPolicy` gains an optional injected
  `rule`. `classify_unusual_drop` evaluates it only for an approved policy
  that has recorded authority and a rule. The default policy stays
  `unavailable`, with a null `unusual_drop`. The repository adds no method
  or threshold, and only synthetic tests inject a rule.

## 16. Presentation correspondence (added 2026-10-08)

This section records how
[`src/ql2_sixt_canada_analysis/assortment_presentation.py`](../../../src/ql2_sixt_canada_analysis/assortment_presentation.py)
and `notebooks/04_visible_assortment.ipynb` present the engine result. It
changes no definition, status or authority.

- **What the presentation does.** It validates, selects, aggregates, formats
  and narrates one completed engine result. It never rebuilds product sets,
  recalculates ratios or price outcomes, or builds a second grid. Each table
  has an exact schema, a semantic kind for every column and 24 reconciliation
  checks to the engine.
- **Observed drops.** The observed-drop table lists **review candidates**,
  not unusual drops. `unusual_drop` stays empty while the unusual-drop policy
  remains **PROPOSED - not approved**.
- **Cross-location grouping.** Grouping by exact scheduled period describes
  drops as isolated or simultaneous *in this extract* only. It is never
  evidence of a common cause.
- **Persistence.** Timeline persistence remains **PROPOSED - not approved**.
  The presentation writes nothing, and an output directory returns the
  `persistence_not_approved` blocker before any directory is touched. Ignore
  rules for assortment outputs are defence in depth and authorize nothing.
