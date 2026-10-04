# Pricing-authority decision records

This directory holds the versioned record of every **external decision** the
dataset needs before it can be called pricing ready: job-identifier rules,
the expected stream universe and its spelling, location roles and comparison
pairs, the Vancouver location identity, the collection schedule, timestamp and
reporting-day semantics, and rental-date validity and parent/detail
agreements (22 atomic decisions, `DecisionId` in
`src/ql2_sixt_canada_analysis/authority_decisions.py`).

| File | Purpose |
| --- | --- |
| [`v1.toml`](v1.toml) | Revision 1 - every decision `PROPOSED` (no attributable authority found). |
| [`authority_request_checklist.md`](authority_request_checklist.md) | Neutral questions to send to each authority, generated from the current revision. |

## Status semantics

| Status | Meaning |
| --- | --- |
| `PROPOSED` | Unresolved. Names the responsible authority role(s), the exact question, the blocker or gap it affects, and `blocking_external_input = true`. Carries **no** resolution and no authority; it is never consumed as production authority. |
| `APPROVED` | An attributable authority answered. Requires authority provenance and a complete resolution in the shape validated for that decision. |
| `REJECTED` | An attributable authority rejected the proposal. Requires authority provenance and a statement of what was rejected; carries no resolution. |

## Authority requirements

Only three kinds of authority can approve or reject: `SUPPLIER`,
`COLLECTION_OWNER` and `BUSINESS_OWNER`. Each `[[decisions.authority]]` entry
needs the kind, the responsible source (team, organization or role holder),
a **durable reference** (document, ticket or written decision) and optionally
a note and an effective date. The kind must be one of the decision's
responsible roles; a joint decision needs every responsible role.

Evidence (`RAW_DATA_OBSERVATION`, `BEHAVIORAL_ANALYSIS`,
`REPOSITORY_IMPLEMENTATION_NOTE`, `ISSUE_OR_REVIEW_NOTE`) may explain why a
question is asked, but it is **never** authority: evidence kinds are refused
in authority entries, and flipping a status to `APPROVED` without authority
and a resolution fails validation. Streams observed in an extract, the
decimal-zero identifier pattern and duplicate-looking Vancouver behaviour are
recorded as evidence only. Never invent a person, organization, document,
ticket, email or approval reference.

The record must not contain source rows, identifiers, timestamps, prices,
vehicle names, offer signatures or other source-level data; free text that
looks like such data is rejected.

## Supersession

Revisions are immutable once committed. To record an answer, copy the
current revision to `v<N+1>.toml`, set `record_version = N+1`,
`record_id = "pricing-authorities-v<N+1>"` and
`supersedes = "pricing-authorities-v<N>"`, update `source_commit`, `created`,
the affected decisions, `summary` and `external_inputs`, then regenerate the
checklist. Old revisions stay in place as history.

## Validation

```bash
python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v1.toml
```

prints a sanitized status summary (decision ids, statuses, roles and counts
only) and exits non-zero if the record is invalid. Error messages name
categories, decision ids and field names only, never record values. The test
suite validates the committed revision and checks that the checklist matches
`render_authority_request_checklist`.

## Approved decisions are implemented separately

Recording an approval changes no production behaviour. Contracts in
`schemas.py`, the coverage, schedule and temporal rules and the pricing gate
consume an approved decision only through separate implementation work with
its own tests. `pricing_baseline.baseline_authority_inputs` reads APPROVED
decisions only, so revision 1 cannot clear any pricing blocker or plan gap.
