# Investigation: Vancouver Downtown versus Vancouver Thurlow

- **Date:** 2026-10-04
- **Streams:** the pair configured as `COMPARED_LOCATION_STREAMS` /
  `LOCATION_STREAM_COMPARISON` in `src/ql2_sixt_canada_analysis/schemas.py`
  (both identified as expected by the project owner).
- **Tool:** `compare_location_streams` (`src/ql2_sixt_canada_analysis/comparison.py`).

This note is sanitized: it records evidence types and classifications only -
no rows, identifiers, prices, products, timestamps, date ranges, counts,
overlap ratios, addresses, coordinates, fingerprints or source filenames.

## Question

Are the two labels distinct locations, two collection streams for one
location, a confirmed alias, duplicated collection configuration, a
label/mapping defect, or undetermined?

## Stages examined

Expected-location configuration, presence in each dataset, label variants,
scope, collection events, job-to-detail linkage, temporal overlap, product
and price-aware offer multisets per paired capture, scope baseline, and
availability of authoritative identity metadata.

## Observation

1. **Configuration (repository gap, fixed).** Neither label was part of the
   expected-location contract, so no control could detect the loss of either
   stream. Both are now expected locations and the pair is defined once.
2. **Presence and labels.** Both labels occur exactly as configured in the
   detail data only (the jobs data is city-level). No case, whitespace or
   punctuation variants were found. Both belong to the same scope.
3. **Collection events.** The two streams occur in the same collection
   events, with complete overlap; pairing is exact by shared event, so no
   time tolerance was needed.
4. **Offers.** In every paired capture, the product multisets and the
   price-aware offer multisets of the two streams are identical.
5. **Baseline.** No other location pair in the same scope behaves this way,
   so identical behaviour is discriminative rather than normal for the source.
6. **Identity metadata.** The source has no station/branch ID, address or
   coordinates; physical identity cannot be established from the data.
7. **Linkage.** The pipeline-wide parent-key linkage failure (documented in
   the relationship control) also affects these detail rows; it is not
   specific to this pair.

## Interpretation

Behavioural evidence indicates **duplicated collection**: the two labels
return the same results in every shared capture
(`LIKELY_DUPLICATE_STREAMS`). Possible explanations include two search
configurations resolving to one supplier location, or one label mapped to
the other location's search. These cannot be distinguished from the data.

## Confirmed conclusion

None. Neither an alias nor distinct locations are confirmed. Similar names,
identical offers and identical prices do not prove physical identity.

## Decision and follow-up

- No alias is configured; the streams are not merged, relabelled or
  deduplicated, and source-label coverage is unchanged.
- **Upstream review required:** confirm with the collection owner which
  supplier location each label's search targets.
- Until resolved, analyses must not count the two streams as independent
  evidence (for example, in location-level price comparisons or assortment
  breadth) without flagging the duplication.
- Regression control: `compare_location_streams` in the ingestion notebook
  (`location_comparison_report`) and `tests/test_comparison.py`.
- Evidence rule (tightened later): a likely duplicate now requires at least
  two independent paired captures, complete temporal overlap, a
  discriminative baseline and identical offers in every paired capture; the
  report shows the evidence counts and any blocking gaps. The categorical
  conclusion above still holds under the stricter rule.
- Policy gate (added later): the identity decision is recorded only in the
  authority-backed `VANCOUVER_LOCATION_POLICY`, which stays `UNRESOLVED`;
  pricing readiness is blocked until an authority confirms an alias (with a
  canonical location) or distinct locations. This investigation's behavioural
  result does not resolve it.
- Mapping defects (added later): if authoritative identity metadata becomes
  available and a stream shows conflicting identities
  (`LOCATION_MAPPING_DEFECT`), any configured alias or distinct decision stays
  recorded but is not authority sufficient; neither alias grouping nor
  independent comparison is permitted and pricing stays blocked until the
  source mapping is corrected or authoritatively reconciled.
