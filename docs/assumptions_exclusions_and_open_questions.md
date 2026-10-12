# Assumptions, exclusions and open questions

This document separates what was **decided by an external authority**, what
the analysis **assumes**, what the data **cannot show**, what is **excluded**
and why, and what remains **unanswered**. It supports
`notebooks/06_final_report.ipynb` and data-plan Section 7. It contains no
source values, observed counts, identifiers, prices, file names or paths.

The categories are deliberately distinct:

| Category | Meaning | Can the analysis change it? |
| --- | --- | --- |
| Approved external decision | Recorded in the current pricing-authority record with an attributable authority. | No; only a new record revision can. |
| Analytical assumption | A method choice made by this analysis, with a stated scope and consequence. | Yes, with a documented change and tests. |
| Data limitation | Something the supplied extract cannot show, however it is analysed. | No; needs more or different data. |
| Governed exclusion | Records excluded from pricing by an approved decision; raw rows are kept. | No; needs a new record revision. |
| Mechanical quality exclusion | Rows removed or blocked by a deterministic quality rule. | Only by changing the rule and its tests. |
| Unassessable evidence | A statistic or status that cannot be computed and is reported as unavailable, never as zero or a pass. | No; it follows from the evidence. |
| Open analytical question | A question this repository cannot answer without a decision or new evidence. | No; it is not guessed. |
| Additional data request | Evidence that would answer an open question. | No; it must be supplied. |

## Approved external decisions

The current authority source is pricing-authority record **v8**
(`docs/decisions/pricing_authorities/v8.toml`, record id
`pricing-authorities-v8`, schema 4). All 23 decisions are `APPROVED`; none
is `PROPOSED`, and the generated checklist
(`docs/decisions/pricing_authorities/authority_request_checklist.md`) lists no
open request. These are **not** open questions and are not reopened here.

| Decision | What it fixes | Effect on the analysis |
| --- | --- | --- |
| `JOB_ID_INVALID_NUMERIC_REPRESENTATIONS`, `JOB_ID_LEADING_ZERO_SIGNIFICANCE`, `JOB_ID_RAW_AND_LINKAGE_PRESERVATION`, `JOB_ID_DECIMAL_ZERO_EQUIVALENCE` | Job identifiers are opaque text; leading zeros matter; raw identifiers are preserved; only the legacy decimal-zero repair is allowed. | Raw identifiers are never rewritten; analytical linkage uses the separate derived keys `job_id_linkage_key` and `row_index_key`. |
| `EXPECTED_STREAM_UNIVERSE`, `EXPECTED_STREAM_SOURCE_SPELLING` | Exactly seven (city, location) source streams, `EXHAUSTIVE`, in exact source spelling. | A missing, unexpected or misspelled stream blocks completeness and pricing; observed streams never extend the universe. |
| `LOCATION_ROLE_ASSIGNMENTS`, `VALID_LOCATION_COMPARISON_PAIRS` | One airport or downtown role per stream; three within-city airport/downtown pairs. | Matched pricing and airport/downtown co-movement use only these pairs, never names. |
| `VANCOUVER_LOCATION_IDENTITY` | Vancouver Downtown and Vancouver Thurlow are one governed location (`CONFIRMED_ALIAS`), canonical key Vancouver Downtown. | Both raw streams stay separately required; they are analysed as one canonical location and never compared with each other. |
| `CANONICAL_OFFER_COMBINATION` | How the two Vancouver streams' offers combine: validation first, order-independent union, exact semantic identity, exact duplicates kept once with provenance, price variation kept and flagged, unassessable rows fail closed. | Downtown Vancouver and Thurlow are combined **only** under this policy; no averaging, priority or deduplication beyond it. |
| `SCHEDULE_CAPTURE_TIMESTAMP`, `SCHEDULE_EXPECTED_PERIODS`, `SCHEDULE_SHARING_MODEL`, `FINISHED_AT_TIMEZONE` | Parent `jobs.finished_at` is the capture timestamp; one parent job per city and local hour; `PER_STREAM` schedules; city-local IANA zones (`America/Edmonton`, `America/Toronto`, `America/Vancouver`). | Expected periods come from the schedule, never from observed jobs; each stream is judged on its own periods. |
| `SCHEDULE_EXCEPTIONS` | One listed `INCOMPLETE_PARENT_CAPTURE` exclusion for one Calgary period (both Calgary streams). | See *Governed exclusions*. |
| `SCRAPED_FINISHED_ORDERING`, `SCRAPED_FINISHED_TOLERANCE` | `cars.scraped_at` must be earlier than or equal to `jobs.finished_at`, zero tolerance. | Violations block temporal trust. |
| `REPORTING_DAY_SOURCE`, `REPORTING_DAY_TIMEZONE`, `SCRAPE_DATE_SEMANTICS`, `DATE_CLEAN_SEMANTICS` | The reporting day is the local date of the parent finish instant in the parent city's zone; both scrape dates mean that day; `date_clean` is retired from pricing. | `date_clean` is parse-checked only; reporting-day disagreement makes rows ineligible. |
| `RENTAL_DATE_VALIDITY`, `RENTAL_DATE_PARENT_DETAIL_AGREEMENTS` | Exact ISO dates, both required, return on or after pickup (same day allowed, no maximum); detail dates equal their parent dates. | Invalid or disagreeing dates make rows rental-date ineligible; nothing is repaired. |

Proposed but **not approved** (recorded in
`docs/decisions/governance/visible-assortment-contract-proposal-v1-2026-10-08.md`
and `monitoring.py`): the unusual-assortment-drop policy, timeline
persistence, multi-context rental stratification, and every synchronized-movement
threshold. No newer authoritative decision approves them; they appear below as
open questions.

## Analytical assumptions

| Assumption | Scope | Analytical consequence |
| --- | --- | --- |
| Listed prices are compared as returned (`price_num` in exact cents, with the currency marker and basis parsed from `price_per_day`). | Every price measure. | Results describe listed rates per price basis, not total customer cost, taxes, fees or transacted prices. |
| A product is the approved exact identity (`car_name`, `car_type`, `transmission`, `seats`, `bags`); no fuzzy or similarity matching. | Matching, events, assortment, stability. | Spelling or attribute variants are different products; nothing is merged by resemblance. |
| Matched comparisons require the same city, collection job (scheduled period), rental dates, product, currency and price basis, with exactly one offer per side. | Matched-location pricing. | Unmatched, ambiguous and incompatible groups are counted, never imputed; premiums cover matched pairs only. |
| Hourly captures of one product are repeated measurements, not independent observations. | Statistical summaries and vehicle-type tests. | Tests (Kruskal-Wallis, Holm, epsilon-squared) are associational and descriptive; they establish no cause. |
| A price change is judged only between consecutive eligible captures exactly one hour apart with the same contributing streams. | Price changes and assortment intervals. | Nothing is compared across a break; gaps are never bridged. |
| The material synchronized-movement rule (at least two changed offers, all in one direction) is a documented descriptive selection rule. | Price-change presentation. | It selects candidates for review; it is not an approved alert threshold. |
| Magnitude statistics need at least two contributors (`MINIMUM_MAGNITUDE_CONTRIBUTORS`). | Price-change tables. | Smaller groups show the statistic as unavailable to protect individual prices. |
| Persistence is judged only at the immediately following eligible interval. | Persistence outcomes. | Longer-run persistence is not measured. |
| Visible assortment is the set of products the collection returned at a capture. | Assortment measures. | It is not proof of supplier availability or withdrawal. |
| The supplied window (roughly 90 hourly captures) is analysed as one descriptive sample. | Every finding. | Nothing is extrapolated to other periods, seasons or rental contexts. |

## Data limitations

- **Short window.** Roughly 90 hours of captures: enough to evaluate the
  contract-based controls, insufficient for seasonality, long-term behaviour,
  normal-variation baselines or threshold calibration.
- **One supplier and one apparent source mode.** Supplier, channel and
  win/meet/loss questions cannot be answered from these files.
- **No collection logs or source snapshots.** The extract alone cannot
  separate genuine repricing from extraction or processing defects, or a
  supplier withdrawal from an incomplete capture.
- **Final-window censoring.** Changes at the last capture have no following
  capture, so their persistence is unknown.
- **No authoritative location identity metadata in the source.** Location
  identity comes only from the approved authority record; behavioural
  similarity between feeds never establishes an alias.
- **Collector-clock scrape times.** `scraped_at` carries a fixed `MST`
  designator that labels the collector's clock, not market-local time; it is
  never a schedule key.
- **Undocumented source fields.** `mode`, `status` and `city_clean` have no
  documented authority and are not used analytically.

## Governed exclusions

- **Incomplete Calgary parent capture** (`SCHEDULE_EXCEPTIONS`, record v8,
  `docs/decisions/governance/calgary-incomplete-parent-capture-exclusion-governance-v1-2026-10-06.md`).
  One Calgary parent capture returned Airport rows but no Downtown rows. Both
  Calgary stream-periods of that one scheduled period are analytically null:
  neither covered nor missing, and removed from the required periods. The
  raw parent job and every linked detail row **stay** in the raw and audit
  populations (ingestion, blank-row counts, linkage, keys, reconciliation,
  coverage and exception reporting). They are **excluded from the
  pricing-eligible population** (`governed_exclusion`) and therefore from
  price summaries, matched pairs, events, assortment sets, the vehicle
  stability used for pricing, offer counts and reporting-day cohorts. In the
  interval timelines the excluded capture is a typed `governed_exclusion`
  break, so no interval spans it. The exclusion is keyed by schedule version,
  city, period and streams (never a job identifier) and must match exactly
  one capture. It is not a general rule for incomplete captures.

## Mechanical quality exclusions

- **Completely blank rows** (every field missing, empty or whitespace-only)
  are removed from the analytical frames and counted per dataset. Partially
  populated rows are retained and evaluated by every control; values such as
  `0` or `False` are meaningful.
- **Invalid or blocked records** never enter a conclusion that needs trusted
  pricing evidence. Detail rows fail eligibility in order as
  `governed_exclusion`, `reporting_day_failed` (unlinked, untrusted,
  temporally invalid or scrape-date disagreement), `rental_dates_failed`
  or `capture_period_unassigned` (schedule-ineligible). Unmatched, ambiguous
  or colliding job references, an untrusted join, invalid keys and
  unassessable canonical offers do not drop rows silently: they block the
  relevant gate, and pricing readiness fails closed.

## Ambiguous and unassessable evidence

- **Ambiguous** price outcomes (several price-distinct offers of one identity
  at an endpoint) and ambiguous matched groups are counted separately and are
  never price changes or pairs.
- **Zero or invalid denominators** (a zero downtown price, a zero previous
  price, an empty previous product set or union) make the percentage or ratio
  unavailable with an explicit status; it never becomes infinity or an
  invented zero.
- **Right-censored final-window events** are `confirmation_required`. They are
  excluded from persistence conclusions: neither persistent nor disproven.
- **Not-testable persistence** across a governed exclusion, missing capture,
  non-hourly gap or stream change is kept apart by reason and is in no
  denominator.
- **Seed captures and captures after a break** are in no assortment
  denominator; captures with no count are never zero.
- **Monitoring controls** without prerequisite evidence are `not_assessable`,
  which is never a pass; candidate rules are `candidate_only`, never a pass
  or an alert.

## Presentation and privacy exclusions

- Offer-level rows, product identities, individual prices, raw identifiers,
  raw source location labels, source file names, paths and the detailed event
  table are never displayed or committed. Only sanitized aggregate tables
  with fixed schemas, in-memory figures and generated interpretation text are
  shown.
- Notebook outputs are cleared before committing; generated reports and the
  opt-in detailed event export are written only to Git-ignored local
  directories.

## Scope exclusions

- Supplier, channel and win/meet/loss questions (only one supplier and one
  apparent source mode are present).
- Causal claims, intent behind synchronized movements, and supplier
  availability changes.
- Production monitoring: no scheduler, persistence, alerting, notification
  or dashboard system is implemented, and no candidate rule is a production
  alert.

## Open analytical questions

These remain unanswered; none is guessed. Owners are named only where a
governance record supports them. The table is generated from
`OPEN_QUESTIONS` in `src/ql2_sixt_canada_analysis/final_report.py`.

<!-- BEGIN GENERATED: open-questions -->
| ID | Question | Why it matters | Likely owner |
| --- | --- | --- | --- |
| `unusual_assortment_drop_policy` | Should an unusual-assortment-drop method, minimum history, grouping and threshold be approved? | Until approved, observed drops are review candidates only and the control stays candidate-only. | business owner (visible-assortment contract proposal) |
| `synchronized_movement_policy` | Should synchronized-price-movement thresholds and grouping rules be approved? | Until approved, synchronized movements are descriptive and the control stays candidate-only. | business owner (production calibration requirements) |
| `production_monitoring_operations` | If production monitoring is wanted, which persistence, alert-routing, ownership, escalation and review policies apply? | No scheduler, persistence, notification or dashboard exists; none can be designed without them. | business owner for timeline persistence; not recorded in governance documentation for routing and escalation |
| `rental_context_stratification` | Should captures with several rental search contexts be stratified or rejected? | The assortment contract compares one rental search context per interval. | collection owner and business owner jointly (visible-assortment contract proposal) |
| `repricing_versus_collection_defect` | Do the observed synchronized movements and drops reflect market repricing or extraction and processing defects? | The extract alone cannot tell them apart; any commercial reading depends on the answer. | not recorded in governance documentation |
| `final_window_persistence` | Did the price and assortment changes at the final capture persist? | Right-censored observations are confirmation-required and excluded from persistence conclusions. | collection owner (collection schedule governance) |
| `generalization_beyond_window` | Do the matched premiums and change patterns hold for other periods, rental dates, durations and seasons? | The roughly 90-hour sample is descriptive and cannot establish normal variation or seasonality. | not recorded in governance documentation |
| `competitive_and_channel_context` | Which suppliers, channels or competitor rates should the premiums be compared with for win, meet or loss decisions? | The files contain one supplier and one apparent source mode, so competitive questions are out of scope. | not recorded in governance documentation |
<!-- END GENERATED: open-questions -->

## Additional data requests

Generated from `DATA_REQUESTS` in `src/ql2_sixt_canada_analysis/final_report.py`.

<!-- BEGIN GENERATED: data-requests -->
| ID | Evidence requested | Why it is needed | Likely owner |
| --- | --- | --- | --- |
| `longer_history` | A longer historical period of the same job and detail exports. | Cover normal variation, seasonality, collection changes and known incidents; required before any threshold calibration. | not recorded in governance documentation |
| `next_eligible_collections` | The scheduled collections that follow the supplied window. | Resolve right-censored end-of-window observations (confirm or refute persistence). | collection owner (collection schedule governance) |
| `collection_logs_and_capture_completeness` | Collection logs and capture-completeness evidence for the reviewed intervals. | Separate genuine changes from incomplete captures, retries or parser issues. | collection owner (operational corroboration channel) |
| `source_snapshots_or_supplier_evidence` | Raw source snapshots or supplier-side evidence for selected intervals. | Distinguish market repricing from extraction or processing defects. | not recorded in governance documentation |
| `labelled_incidents_and_normal_periods` | Labelled collection incidents and confirmed normal periods. | Back-test any future monitoring rule; no threshold may be optimized against this sample. | not recorded in governance documentation |
| `additional_search_contexts` | Captures for other pickup dates, rental durations and booking lead times. | Test whether premiums and assortment depend on the rental search context. | not recorded in governance documentation |
| `competitor_and_channel_rates` | Comparable rates from other suppliers or channels for the same searches. | Required for supplier, channel and win/meet/loss questions, which this extract cannot answer. | not recorded in governance documentation |
<!-- END GENERATED: data-requests -->
