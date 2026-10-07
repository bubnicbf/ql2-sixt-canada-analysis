# Rental-date validity and parent/detail agreement governance decisions - version 1

| | |
| --- | --- |
| Responsible authority | `RENTAL_DATE_VALIDITY`: collection owner and business owner (joint). `RENTAL_DATE_PARENT_DETAIL_AGREEMENTS`: collection owner (authority role: collection owner or supplier) |
| Provenance | Direct written governance decisions supplied by the repository owner |
| Decision date | 2026-10-06 (date the decisions were supplied and recorded in this repository) |
| Effective scope | The current dataset and subsequent collections until superseded by a new versioned authority decision |
| Meeting date, participants, organizations, tickets | Not supplied |
| Decision record | [`pricing-authorities-v7`](../pricing_authorities/v7.toml) (schema 3) |
| Governed fields | `jobs.pickup_date`, `jobs.return_date`, `cars.job_pickup_date`, `cars.job_return_date`, `cars.pickup_date`, `cars.return_date` |

This page is the durable, repository-local authority reference for the two
rental-date decisions approved in `pricing-authorities-v7`. It records only
the decisions as supplied. No personal name, meeting participant,
organization or ticket was supplied, and none is implied. It contains no
dates, identifiers, rows, timestamps, prices, locations or offers.

## Decision: `RENTAL_DATE_VALIDITY` - APPROVED

The contract applies to the parent fields `jobs.pickup_date` and
`jobs.return_date` and the detail fields `cars.job_pickup_date`,
`cars.job_return_date`, `cars.pickup_date` and `cars.return_date`.

- **Format:** `ISO_8601_DATE`. The exact accepted text is `YYYY-MM-DD`, and it
  must name a real calendar date. Values are calendar dates: they are not
  timestamps, carry no time zone and are never read as midnight instants.
  Values with a time, an offset, slashes, month names, extra text or
  surrounding whitespace, numbers and other objects are invalid; nothing is
  trimmed or coerced into a date.
- **Requiredness:** pricing-eligible records require valid pickup and return
  dates. Both are required for every rental period.
- **Ordering:** the return date must be greater than or equal to the pickup
  date. **Equality is permitted**: a same-calendar-day rental is valid.
- **Duration:** the return date minus the pickup date in calendar days. The
  logical **minimum is zero** days. There is **no maximum** duration
  (unbounded): an unusually long but correctly ordered rental is valid data,
  and no maximum is inferred from observed values, percentiles, historical
  frequency or analytical preference.

## Decision: `RENTAL_DATE_PARENT_DETAIL_AGREEMENTS` - APPROVED

All four detail date fields repeat the parent job's search dates:

| Detail field | Must equal |
| --- | --- |
| `cars.job_pickup_date` | `jobs.pickup_date` |
| `cars.job_return_date` | `jobs.return_date` |
| `cars.pickup_date` | `jobs.pickup_date` |
| `cars.return_date` | `jobs.return_date` |

- Comparisons use successfully parsed calendar dates on trusted linked rows,
  never raw text alone; a value that violates the approved format cannot
  agree, even if another parser could read it as the same date.
- **Raw values are preserved** for investigation; no source value is
  overwritten.
- **One-sided missing values fail**, and both values missing is not an
  agreement pass, because the fields are required.
- **No automatic repair:** a disagreement is never resolved by preferring the
  parent value, the detail value, the job-prefixed detail value or the
  non-prefixed detail value.

## Data validity versus analysis eligibility

Data validity asks whether a field is present, has the approved format, is a
real calendar date, has its return on or after its pickup, and agrees with
its parent. Analysis eligibility asks whether a rental duration belongs to a
particular pricing study (for example a same-day, one-day, weekend or weekly
cohort). A record is never declared invalid because its duration is outside
a future analytical cohort, and an analytical filter never hides or repairs
invalid data. A duration-based study population needs its own explicit,
approved pricing-analysis policy.

## Change control

Any change to the format, requiredness, ordering, minimum or maximum
duration, or the parent/detail mappings requires a new authority-record
version; committed record versions are never edited. These decisions do not
make the dataset pricing ready.
