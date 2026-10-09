# Data dictionary

This dictionary describes the two proprietary source exports and the analytical
outputs built from them, so that an analyst can read the notebooks without
opening the data. It contains **no source values**: no identifiers, prices,
timestamps, product names, observed counts, file names or paths. Examples of
formats are written as patterns.

Sources of truth (read these when this document and the code disagree; the
tests keep them in step):

- Raw dataset contracts: `DATASET_DEFINITIONS`, `ANALYSIS_DATASET_DEFINITIONS`
  and `JOB_DETAIL_RELATIONSHIP` in `src/ql2_sixt_canada_analysis/schemas.py`.
- Approved external decisions: the current pricing-authority record,
  `docs/decisions/pricing_authorities/v8.toml` (record `pricing-authorities-v8`).
- Derived and presentation contracts: `collection_schedule.py`,
  `canonical_offers.py`, `pricing_population.py`, `temporal.py`,
  `rental_dates.py`, `matched_location_pricing.py`, `price_change_events.py`,
  `price_change_analysis.py`, `price_change_presentation.py`,
  `assortment_contract.py`, `visible_assortment.py`,
  `assortment_presentation.py`, `monitoring.py` and `final_report.py`.

Blocks marked `GENERATED` are rendered from the schema contracts by
`python -m ql2_sixt_canada_analysis.data_dictionary`; do not edit them by hand.

## Value categories

Every field below is one of four kinds, and the kinds are never mixed:

| Kind | Meaning | Examples |
| --- | --- | --- |
| Raw value | Read from a source export and never rewritten. | `job_id`, `finished_at`, `price_num` |
| Derived value | Computed in memory from raw values under an approved contract; raw values are kept beside it. | `job_id_linkage_key`, `scheduled_capture_period`, `price_cents`, `premium_cents` |
| Configuration / authority value | Taken from the approved authority record or a code contract, never from the data. | expected stream keys, location roles, city IANA zones, the governed Calgary exclusion |
| Presentation-only value | Exists only in sanitized aggregate tables, figures or interpretation text. | `interpretation_status`, `review_status`, `findings_valid`, suppression flags |

## Datasets

### Source and analysis-stage datasets

<!-- BEGIN GENERATED: raw-datasets -->
| Dataset | Stage | Grain | Unique key | Identifier fields | Relationship | Status | Confidentiality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `jobs` | raw | One row per scrape (collection) job. | `job_id` | `job_id` | Parent: one job has zero, one or many `cars` rows. | Source (immutable CSV export) | Proprietary; never committed or displayed |
| `cars` | raw | One row per offer position within one scrape job's result list. | `job_id`, `row_index` | `job_id` | Detail: every row links to exactly one `jobs` row. | Source (immutable CSV export) | Proprietary; never committed or displayed |
| `jobs` | analysis | One row per scrape (collection) job. | `job_id_linkage_key` | `job_id`, `job_id_linkage_key` | Parent: one job has zero, one or many `cars` rows. | Derived in memory (raw columns plus derived keys) | Proprietary; never committed or displayed |
| `cars` | analysis | One row per offer position within one scrape job's result list. | `job_id_linkage_key`, `row_index_key` | `job_id`, `job_id_linkage_key` | Detail: every row links to exactly one `jobs` row. | Derived in memory (raw columns plus derived keys) | Proprietary; never committed or displayed |

Relationship keys: raw `job_id` -> `job_id` (source form, reconciliation of the raw extract only); analytical linkage uses the derived key of the analysis stage. Declared counts: `record_count`, `actual_car_rows`. Scope agreement: `jobs.city` = `cars.city`.
<!-- END GENERATED: raw-datasets -->

Completely blank rows are removed from both analytical frames and counted;
partially populated rows are kept and evaluated by every control.

### Derived analytical datasets

All of these exist in memory only. Those marked *confidential* hold offer-level
values and are never displayed, written by default or committed.

| Logical dataset | Grain | Unique key | Parent / child relationship | Status | Confidentiality |
| --- | --- | --- | --- | --- | --- |
| Pricing-eligible population (`PricingPopulation`) | One status per parent job and per detail row of the analysis frames. | The analysis-stage keys. | Selects eligible rows of the analysis `jobs` and `cars`; excluded rows stay in the frames for audit. | Derived | Confidential (statuses only are reported as counts) |
| Canonical offers (`CanonicalOfferReport.offers`) | One row per canonical offer: canonical location, scheduled capture period, rental dates, approved product identity, price in cents, price basis and currency marker. | That full identity. | Built from eligible `cars` rows; the two governed Vancouver streams combine into one canonical location. | Derived | Confidential |
| Matched location pairs (`MatchedLocationPricingResult.pairs`) | One row per approved city, scheduled capture period, rental period, approved product, currency and price basis with exactly one airport and one canonical downtown offer. | `PAIR_KEY_COLUMNS`. | Pairs two canonical offers of the same collection job. | Derived | Confidential |
| Price-change candidates (event table) | One row per canonical location, product and unit identity, and consecutive eligible capture interval. | `EVENT_KEY_COLUMNS`. | Compares canonical offers of two consecutive eligible captures. | Derived | Confidential (local opt-in Parquet only) |
| Persistence records | One row per increase or decrease. | The event key. | Child of a price-change candidate; judged at the next eligible interval only. | Derived | Confidential |
| Assortment membership | One row per canonical location, product (approved identity plus rental dates) and assessed interval. | Location, product and interval. | Child of an assessed timeline interval. | Derived | Confidential |
| Assortment timeline | One row per approved canonical location and scheduled capture period. | `canonical_location`, `scheduled_capture_period`. | Parent of the assessed intervals. | Derived | Sanitized aggregate (displayed in notebook 04) |

### Sanitized aggregate tables (displayed)

These tables have fixed schemas enforced in code; a table that does not match
its schema or allowlist is refused before display.

| Table | Producer | Grain | Key |
| --- | --- | --- | --- |
| `match_count_frame` | `matched_location_pricing` | One row per approved city plus `overall`. | `city` |
| `city_summary_frame` | `matched_location_pricing` | One row per approved city plus `overall`. | `city` |
| `vehicle_type_summary_frame` | `matched_location_pricing` | One row per city and `car_type`. | `city`, `car_type` |
| `vehicle_type_test_frame` | `matched_location_pricing` | One row per city, premium metric and comparison unit. | `city`, `metric`, `unit` |
| `event_interval_summary` | `price_change_presentation` | One row per canonical location and eligible one-hour interval (quiet intervals included). | location, `previous_scheduled_capture_period`, `current_scheduled_capture_period` |
| `material_synchronized_movements` | `price_change_presentation` | One row per interval selected by the descriptive material rule. | as `event_interval_summary` |
| `material_selection_reconciliation` | `price_change_presentation` | One row per movement class. | `movement_class` |
| `airport_downtown_summary` | `price_change_presentation` | One row per approved city pair and interval. | `canonical_city`, interval periods |
| `persistence_summary` | `price_change_presentation` | One row per canonical location and direction. | `canonical_location`, `direction` |
| `final_vancouver_decrease` | `price_change_presentation` | Exactly one fixed-schema record. | none (single record) |
| `reconciliation_summary` (price changes) | `price_change_presentation` | One row per reconciliation check. | `check` |
| `assortment_timeline` | `assortment_presentation` | One row per canonical location and scheduled capture period. | `canonical_location`, `scheduled_capture_period` |
| `location_summary` | `assortment_presentation` | One row per canonical location. | `canonical_location` |
| `observed_drop_review` | `assortment_presentation` | One row per assessed interval with an absolute drop above zero. | location, interval periods |
| `cross_location_drops` | `assortment_presentation` | One row per interval with at least one observed drop. | interval periods |
| `price_coincidence_summary` | `assortment_presentation` | One row per canonical location. | `canonical_location` |
| `reconciliation_summary` (assortment) | `assortment_presentation` | One row per reconciliation check. | `check` |
| `monitoring_control_table` | `monitoring` | Exactly one row per required control, in data-plan order. | `control_id` |
| `final_section_table` | `final_report` | Exactly one row per final-report section, in narrative order. | `section` |
| `open_questions_table` / `data_requests_table` | `final_report` | One row per catalogued question or request (fixed text). | `question_id` / `request_id` |

## Raw fields

Identifier fields are read as pandas' nullable string dtype in the same read
that loads the file. Raw values are never edited; every repair or
normalization lives in a separate derived field.

### `jobs`

<!-- BEGIN GENERATED: raw-fields-jobs -->
| # | Field | Meaning | Representation / parsing | Identifier | Key / relationship role | Missing or invalid handling | Analytical use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `job_id` | Identifier of one scrape (collection) job. | Nullable string (opaque text; leading zeros and full text kept; never trimmed, case-folded or parsed). | yes (nullable string) | raw unique-key component; parent side of the job-detail relationship | Missing or whitespace-only values are invalid linkage keys; duplicates block the key contract. Never rewritten; the derived linkage key is separate. | `linkage_only` |
| 2 | `city` | City whose collection the job belongs to; selects the approved IANA time zone. | Text matched exactly (never trimmed or aliased). | no | parent/detail scope agreement | Missing, blank or padded values make the job scope unassignable and block city integrity, the trusted join, completeness and pricing readiness. | `used_in_pricing` |
| 3 | `mode` | Collection mode label supplied by the source. | Text, preserved as read. | no | none | Not evaluated beyond blank-row detection. | `not_used` |
| 4 | `status` | Collection status label supplied by the source. | Text, preserved as read. | no | none | Not evaluated beyond blank-row detection. | `not_used` |
| 5 | `record_count` | Declared number of detail rows for the job. | Non-negative whole number. | no | declared detail count | Missing or invalid values are counted separately and fail the reconciliation; never repaired. | `validation_only` |
| 6 | `pickup_date` | Rental pickup date searched by the job. | Exact `YYYY-MM-DD` text naming a real calendar date (no trimming, time or zone); return on or after pickup, same-day allowed, no maximum. | no | none | Missing, malformed or out-of-order dates are counted and make the job's rows rental-date ineligible; never repaired. | `validation_only` |
| 7 | `return_date` | Rental return date searched by the job. | Exact `YYYY-MM-DD` text naming a real calendar date (no trimming, time or zone); return on or after pickup, same-day allowed, no maximum. | no | none | As `pickup_date`. | `validation_only` |
| 8 | `finished_at` | Scheduled capture timestamp: when the job finished, in the job city's local wall-clock time. | Naive `%Y-%m-%d %H:%M:%S.%f` resolved in the approved city IANA zone to a full-precision UTC instant; defines the scheduled capture period and the reporting day. | no | none | Missing, invalid, ambiguous (fall-back) and nonexistent (spring-forward) values are counted separately, fail closed and leave the capture period unassigned. | `used_in_pricing` |
| 9 | `scrape_date` | Source-supplied calendar date of the job (approved meaning: the reporting day). | Strict ISO calendar date. | no | none | Must equal the derived reporting day; disagreement or invalid values make rows reporting-day ineligible. | `validation_only` |
| 10 | `actual_car_rows` | Second declared detail-row tally of the job. | Non-negative whole number. | no | declared detail count | Reconciled independently of `record_count`; both must match the observed detail rows. | `validation_only` |
<!-- END GENERATED: raw-fields-jobs -->

### `cars`

<!-- BEGIN GENERATED: raw-fields-cars -->
| # | Field | Meaning | Representation / parsing | Identifier | Key / relationship role | Missing or invalid handling | Analytical use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `job_id` | Parent job identifier on each offer row. | Nullable string (opaque text; leading zeros and full text kept; never trimmed, case-folded or parsed). | yes (nullable string) | raw unique-key component; detail reference to the parent job | The historical export carries a legacy decimal-zero suffix; only the approved repair (exact match first, then removing a final `.0` from an all-digit value with a unique parent) is applied, in the derived key. Missing, unmatched, ambiguous or colliding references block linkage. | `linkage_only` |
| 2 | `city` | City of the offer's collection; first component of the source stream key. | Text matched exactly. | no | parent/detail scope agreement | Must equal the parent job's city; a disagreement blocks city integrity. | `used_in_pricing` |
| 3 | `mode` | Repeated collection mode label. | Text, preserved as read. | no | none | Not evaluated beyond blank-row detection. | `not_used` |
| 4 | `status` | Repeated collection status label. | Text, preserved as read. | no | none | Not evaluated beyond blank-row detection. | `not_used` |
| 5 | `job_finished_at` | Copy of the parent job's `finished_at`. | Naive `%Y-%m-%d %H:%M:%S.%f`, resolved with the parent city's zone. | no | none | Must replicate the parent exactly (wall time and instant); failures are counted and block temporal trust. Never a schedule key. | `validation_only` |
| 6 | `scrape_date` | Copy of the reporting day on the offer row. | Strict ISO calendar date. | no | none | Must equal the parent reporting day; otherwise the row is reporting-day ineligible. | `validation_only` |
| 7 | `job_pickup_date` | Copy of the parent job's pickup date. | Exact `YYYY-MM-DD` text naming a real calendar date (no trimming, time or zone); return on or after pickup, same-day allowed, no maximum. | no | none | Must equal the parent `pickup_date` as a parsed date; mismatches are counted, never repaired. | `validation_only` |
| 8 | `job_return_date` | Copy of the parent job's return date. | Exact `YYYY-MM-DD` text naming a real calendar date (no trimming, time or zone); return on or after pickup, same-day allowed, no maximum. | no | none | Must equal the parent `return_date` as a parsed date; mismatches are counted, never repaired. | `validation_only` |
| 9 | `row_index` | Position of the offer in its job's result list. | Non-negative integer (the approved legacy `.0` form is repaired only in the derived key). | no | raw unique-key component | Missing or invalid values leave the derived position key missing and block the detail key contract. Never part of product identity; source order is never used to combine offers. | `linkage_only` |
| 10 | `pickup_date` | Rental pickup date of the offer. | Exact `YYYY-MM-DD` text naming a real calendar date (no trimming, time or zone); return on or after pickup, same-day allowed, no maximum. | no | none | Must be valid and equal the parent pickup date; otherwise the row is rental-date ineligible. | `used_in_pricing` |
| 11 | `return_date` | Rental return date of the offer. | Exact `YYYY-MM-DD` text naming a real calendar date (no trimming, time or zone); return on or after pickup, same-day allowed, no maximum. | no | none | As `pickup_date`. | `used_in_pricing` |
| 12 | `car_name` | Vehicle product name; part of the approved product identity. | Text compared exactly (no fuzzy matching or normalization). | no | none | A missing identity component makes the offer unassessable, which blocks pricing readiness. | `used_in_pricing` |
| 13 | `car_type` | Vehicle class; product identity and vehicle-type grouping. | Text compared exactly. | no | none | Required by vehicle-attribute stability; a conflict or missing value is reported, never filled. | `used_in_pricing` |
| 14 | `price_per_day` | Listed price text carrying the currency marker, amount and price basis. | Strict `<marker>$<whole>.<cents>/<basis>` text (marker of up to three capital letters, optional thousands separators, lowercase basis); the amount must equal `price_num`. | no | none | Unparseable text or an amount disagreeing with `price_num` makes the offer unassessable. Currencies are never assumed equivalent. | `used_in_pricing` |
| 15 | `transmission` | Transmission; part of the approved product identity. | Text compared exactly. | no | none | Required by vehicle-attribute stability. | `used_in_pricing` |
| 16 | `seats` | Seat count; part of the approved product identity. | Value compared exactly. | no | none | Required by vehicle-attribute stability. | `used_in_pricing` |
| 17 | `bags` | Bag count; part of the approved product identity. | Value compared exactly. | no | none | Presence must be stable for a product; missing values never compare equal. | `used_in_pricing` |
| 18 | `location` | Branch label; second component of the source stream key. | Exact source spelling (case, spacing and punctuation significant). | no | none | A missing, unexpected or misspelled stream blocks coverage and pricing; aliases apply only through the approved Vancouver policy. | `used_in_pricing` |
| 19 | `scraped_at` | When the offer row was scraped (collector clock). | `%Y-%m-%d %H:%M:%S` with the `MST` designator, read as fixed UTC-07:00. | no | none | Must be earlier than or equal to the parent finish instant (zero tolerance). Never a schedule key or reporting-day source; it orders observations for vehicle stability only. | `validation_only` |
| 20 | `price_num` | Numeric listed price. | Finite, non-negative number with at most two decimals, converted to exact integer cents. | no | none | Any other value makes the offer unassessable. | `used_in_pricing` |
| 21 | `city_clean` | Source-supplied cleaned city label. | Text, preserved as read. | no | none | No documented authority: city integrity and stream keys use the raw `city`. | `not_used` |
| 22 | `date_clean` | Source-supplied cleaned date. | Strict ISO calendar date. | no | none | Parse quality is reported; the approved decision retires it from pricing. | `retired_from_pricing` |
<!-- END GENERATED: raw-fields-cars -->

### Analytical use by field

<!-- BEGIN GENERATED: field-use -->
| Analytical use | Meaning | jobs fields | cars fields |
| --- | --- | --- | --- |
| `used_in_pricing` | Enters the pricing-eligible offers (identity, scheduled period, grouping or price). | `city`, `finished_at` | `city`, `pickup_date`, `return_date`, `car_name`, `car_type`, `price_per_day`, `transmission`, `seats`, `bags`, `location`, `price_num` |
| `validation_only` | Checked by a control; never enters a price, product identity or grouping. | `record_count`, `pickup_date`, `return_date`, `scrape_date`, `actual_car_rows` | `job_finished_at`, `scrape_date`, `job_pickup_date`, `job_return_date`, `scraped_at` |
| `linkage_only` | Used only to derive the confidential linkage or offer-position keys. | `job_id` | `job_id`, `row_index` |
| `retired_from_pricing` | Parsed and reported for quality only; approved as never a pricing field. | none | `date_clean` |
| `not_used` | Preserved unchanged for audit; no approved semantics and no analysis reads it. | `mode`, `status` | `mode`, `status`, `city_clean` |
<!-- END GENERATED: field-use -->

## Derived analytical fields

| Field | Kind | Definition | Missing / invalid handling |
| --- | --- | --- | --- |
| `job_id_linkage_key` | Derived, confidential | Derived job linkage key on both datasets under the approved job-identifier policy: exact match of the opaque text first; the only fallback removes a final `.0` from an all-digit detail value when that identifies exactly one job. Raw `job_id` is unchanged. | `<NA>` for missing, unmatched, ambiguous or colliding references; any such row blocks the linkage, key and join contracts. |
| `row_index_key` | Derived, confidential | Derived offer-position key: the non-negative integer form of `row_index` (pandas nullable integer), with the approved legacy `.0` repair. | `<NA>` when missing or invalid; blocks the detail key contract. |
| `canonical_city`, `canonical_location` | Derived from configuration | The approved (city, location) source key, except that `vancouver / Vancouver Thurlow` canonicalizes to `vancouver / Vancouver Downtown` under the approved alias; both raw streams stay separately required. | Rows outside the approved contract never receive a canonical key; they block coverage. |
| Location role | Configuration | `AIRPORT` or `DOWNTOWN` per approved source stream, from the authority record; pairs are the three approved within-city airport/downtown comparisons. | A missing or inconsistent role blocks pricing readiness. |
| `scheduled_capture_period` | Derived | UTC start of the local top-of-hour period that the parent job's `finished_at` (resolved in the city's approved IANA zone) falls in, written `YYYYMMDDTHHMMSSZ`; one parent job per city and local hour, schedule version `per_stream_hourly_v1`. | A parent that cannot be assigned to exactly one expected period leaves its rows `capture_period_unassigned`. |
| Capture state | Derived | Per canonical location and period: `eligible`, `governed_exclusion` or `missing_capture`. | Non-eligible captures have no product set or price and are never zero. |
| `reporting_day` | Derived | Local calendar date of the parent finish instant in the parent city's approved zone (`jobs.finished_at`, record v8); detail rows receive it only through their trusted linked parent. Kept beside `finished_at_raw`, `finished_at_utc`, `reporting_timezone`, `scrape_date_raw`, `scrape_date_parsed`, `scrape_date_status` and `reporting_day_provenance`. | `scrape_date_status` is `missing`, `invalid`, `mismatch`, `unresolvable`, `untrusted` or `unlinked`; any of them makes the row `reporting_day_failed`. |
| `rental_duration_days` | Derived | `return_date - pickup_date` in calendar days (0 for a same-day rental); also `job_rental_duration_days` for the job dates on detail rows. | Missing when either date is invalid; `rental_period_status` names the reason. |
| `rental_period_status`, `rental_dates_agree`, `pricing_eligible` | Derived | Validity of the rental period, parent/detail date agreement and rental-date eligibility. | A failing row is `rental_dates_failed` in the pricing population. |
| Detail eligibility | Derived | First failing control per detail row: `eligible`, `governed_exclusion`, `reporting_day_failed`, `rental_dates_failed`, `capture_period_unassigned`. | Only `eligible` rows enter any pricing calculation; the others are counted and kept for audit. |
| Approved product identity | Configuration | `car_name`, `car_type`, `transmission`, `seats`, `bags`, compared exactly; offer and event identities add `pickup_date` and `return_date`, and price identities add `currency` and `price_basis`. Price is never part of product identity. | A missing component makes the offer unassessable and blocks readiness. |
| `price_cents` | Derived | Exact integer cents of `price_num`. | Unassessable when `price_num` is not a finite, non-negative number with at most two decimals or disagrees with the price text. |
| `currency` | Derived | Currency marker parsed strictly from `price_per_day`. | Different markers are never compared or aggregated together. |
| `price_basis` | Derived | Basis parsed strictly from `price_per_day` (the text after `/`). | Different bases are never compared or aggregated together. |
| `observation_count`, `provenance`, `source_location_labels`, `price_variation` | Derived, confidential | Canonical-offer provenance: contributing observations, `unique` or `deduplicated`, the raw source labels, and whether same-identity offers differ in price (kept separate, never averaged). | Reported only as aggregate counts and fixed categories. |

### Matched-location pricing measures

| Field | Definition |
| --- | --- |
| `airport_price_cents`, `downtown_price_cents` | `price_cents` of the matched airport and canonical downtown offer. |
| `premium_cents` | `airport_price_cents - downtown_price_cents` (positive = airport premium, negative = airport discount). |
| `premium_dollars` | `premium_cents / 100`, in the pair's currency and price basis. |
| `premium_percent` | `100 * premium_cents / downtown_price_cents`; unavailable (`percent_valid = False`) when `downtown_price_cents = 0`, never infinite and never zero by substitution. |
| `premium_sign` | `positive`, `zero` or `negative` from `premium_cents`. |
| Match outcomes | Each candidate group (an identity with an offer on at least one side of an approved pair) has exactly one outcome: matched, airport only, downtown only, ambiguous (several price-distinct offers on one side), currency mismatch or basis mismatch. |
| `match_rate` | `matched_pairs / candidate_groups`; unavailable without candidates. |
| Vehicle-type tests | Kruskal-Wallis across `car_type` within each city, Holm adjustment across cities, epsilon-squared effect size; `not_testable` below the fixed minimum observations per type. Associational only. |

### Price-change measures and statuses

| Field | Definition |
| --- | --- |
| `outcome` | Exactly one of `unchanged`, `increase`, `decrease`, `appeared`, `disappeared`, `ambiguous` per candidate; `ambiguous` (several offers of one identity at an endpoint) takes precedence and is never a price change. |
| `change_cents` | `current_price_cents - previous_price_cents`. |
| `change_percent` | `100 * change_cents / previous_price_cents`, evaluated exactly from integer cents; unavailable (`zero_denominator`) when the previous price is zero. |
| `comparable` | Candidates with exactly one previous and one current price (`unchanged + increase + decrease`). |
| `changed_share_of_comparable` | `(increase + decrease) / comparable`; unavailable when `comparable = 0`. |
| `movement_class` | `no_price_movement`, `isolated_increase`, `isolated_decrease`, `synchronized_increase`, `synchronized_decrease` or `mixed_direction`, from the interval's increase and decrease counts. |
| `direction_synchronized`, `exact_cent_synchronized`, `exact_percent_synchronized` | At least two changed offers, all in one direction / all with the same signed cent change / all with the same exact percentage (nonzero previous price). |
| `material_synchronized` | The documented descriptive selection rule (direction-synchronized). Not an alert threshold. |
| `interval_flag` | First applicable of `empty_endpoint`, `ambiguity_present`, `price_and_assortment_change`, `assortment_change_only`, `price_change_only`, `quiet`. Descriptive only. |
| Interval break | Why two schedule-adjacent periods form no interval: `governed_exclusion`, `missing_capture`, `not_one_hour`, `source_streams_changed`. Nothing is compared across a break. |
| `persistence` | For each increase or decrease, judged only at the immediately following eligible interval: `held` (same price), `continued` (moved further in the same direction), `reverted` (moved back; `returned_to_prior_price` and `overshot_prior_price` refine it), `disappeared`, `ambiguous`, or `not_testable`. |
| `not_testable_reason` | `right_censored_final_capture`, `governed_exclusion_break`, `missing_capture_break`, `not_one_hour_break`, `source_streams_changed`. Not-testable events are in no persistence denominator. |
| Persistence shares | `held`, `continued` and `reverted` shares use `held + continued + reverted`; `disappeared` and `ambiguous` shares use events with a following interval. |
| `cross_outcome` | Airport/downtown comparison of one matched product in one interval: `ambiguous`, `simultaneous_appearance`, `simultaneous_disappearance`, `mixed_assortment`, `one_sided_assortment`, `both_unchanged`, `airport_only_change`, `downtown_only_change`, `same_direction`, `opposite_direction`. |
| Magnitude suppression | `magnitude_suppressed`, `percent_magnitude_suppressed` and the final-case suppression flags are true when a statistic has fewer than `MINIMUM_MAGNITUDE_CONTRIBUTORS` (2) contributors; the statistic is then unavailable, never zero. |
| `interpretation_status` | Presentation-only category of an interval row (`material_synchronized_candidate`, `mixed_direction_movement`, `isolated_movement`, `anomaly_indicator_empty_endpoint`, `assortment_or_ambiguity_without_price_movement`, `no_price_movement`); never a conclusion. |

### Visible-assortment measures and statuses

For the previous product set `P` and the current set `C` of one canonical
location (a product counts once however many offers it has), `n(X)` is the
number of products in set `X`:

| Field | Formula / definition |
| --- | --- |
| `returned_product_count` | `n(C)` for an eligible capture (zero is a valid observed count); no count for excluded or missing captures. |
| `retained_count` | `n(P ∩ C)` |
| `addition_count` | `n(C − P)` (products in `C` but not in `P`) |
| `removal_count` | `n(P − C)` (products in `P` but not in `C`) |
| `retention` | `n(P ∩ C) / n(P)` (previous set as denominator). |
| `jaccard_similarity` | `n(P ∩ C) / n(P ∪ C)` |
| `net_change` | `n(C) − n(P)` |
| `absolute_drop` | `max(n(P) − n(C), 0)` |
| `drop_rate` | `absolute_drop / n(P)` |
| `*_denominator_status` | `defined`, `zero_denominator` (empty previous set or empty union: the ratio is unavailable, never zero or infinite) or `not_assessable` (no valid interval). |
| `capture_state`, `assessability_status` | Capture `eligible`, `governed_exclusion` or `missing_capture`; row `assessed`, `seed_capture`, `interval_break`, `capture_not_eligible` or `blocked`. |
| `interval_break_reason` | The price-change break reasons plus `rental_context_changed`; seed captures and captures after a break are in no denominator. |
| `assortment_change`, `price_change`, `assortment_price_coincidence`, `falling_assortment_with_price_increase` | Interval flags; a coincidence needs the same canonical location and the same previous and current periods. Coincidence is not causation. |
| `anomaly_policy_status`, `unusual_drop` | The unusual-drop policy is `unavailable`, so `unusual_drop` stays empty; observed drops are review candidates. |
| `drop_pattern`, `simultaneity`, `review_status` | Presentation-only: `net_contraction`, `complete_turnover` or `empty_current_capture`; `isolated_in_extract` or `simultaneous_in_extract`; `observed_drop_review_candidate`. |

### Monitoring fields

| Field | Definition |
| --- | --- |
| `control_id`, `control`, `condition`, `likely_business_impact`, `recommended_response` | Fixed catalog text of the eight data-plan Section 6 controls (`MONITORING_CONTROLS`). |
| `severity`, `effective_severity` | `critical`, `high` or `medium` business impact; only two documented escalations change the effective severity. Not a statistical confidence level. |
| `calibration_status` | `contract_based` or `candidate_policy_unapproved`. |
| `evaluation_status` | Exactly one of `triggered`, `passed`, `not_assessable` (never a pass), `candidate_only`, `confirmation_required`. |
| `findings`, `unavailable_evidence`, `notes`, `policy_status` | Typed codes only (`MonitoringFinding`, `EvidenceGap`, `MonitoringNote`, policy status); never data values. |

### Final-report fields

| Field | Definition |
| --- | --- |
| `section` | `data_and_pipeline_readiness`, `matched_location_pricing`, `price_change_events`, `visible_assortment`, `monitoring_and_actionability`. |
| `status` | `ready` / `blocked` for readiness; `completed` / `blocked` / `unavailable` for the analyses; the monitoring report status (`evaluated`, `partially_evaluated`, `blocked`). |
| `findings_valid` | True only when the section's evidence passed every gate on the one bound pipeline run. |
| `blockers` | Snake-case blocker categories; never values. |
| Interpretation text | Generated by `interpret_section` and `final_conclusions` from report counts, enums and summary statistics only. |
