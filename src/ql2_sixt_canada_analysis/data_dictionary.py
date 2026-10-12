"""Data-dictionary metadata for the raw source fields, checked against the central schema contracts.

``docs/data_dictionary.md`` documents every raw ``jobs`` and ``cars`` field. The
facts that can drift - which fields exist, their order, which are identifiers,
which form the unique key, which carry the parent/detail relationship - are
**not** written by hand: :func:`render_generated_sections` derives them from
:data:`~ql2_sixt_canada_analysis.schemas.DATASET_DEFINITIONS`,
:data:`~ql2_sixt_canada_analysis.schemas.ANALYSIS_DATASET_DEFINITIONS` and
:data:`~ql2_sixt_canada_analysis.schemas.JOB_DETAIL_RELATIONSHIP`. Only the
meaning, the expected representation, the missing/invalid handling and the
analytical use of each field are curated here (:data:`RAW_FIELD_NOTES`), and
:func:`validate_raw_field_notes` refuses notes that do not cover the schema
exactly once per dataset.

The generated Markdown sits between ``<!-- BEGIN GENERATED: name -->`` and
``<!-- END GENERATED: name -->`` markers in the document; the tests require the
document to equal the rendered blocks. Regenerate with::

    python -m ql2_sixt_canada_analysis.data_dictionary

which prints the blocks (structure only: no data is read). Importing this
module performs no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ql2_sixt_canada_analysis.schemas import (
    ANALYSIS_DATASET_DEFINITIONS,
    DATASET_DEFINITIONS,
    JOB_DETAIL_RELATIONSHIP,
    DatasetKey,
)

__all__ = [
    "GENERATED_SECTION_NAMES",
    "RAW_FIELD_NOTES",
    "DataDictionaryError",
    "FieldUse",
    "RawFieldNote",
    "field_key_role",
    "render_generated_sections",
    "replace_generated_sections",
    "validate_raw_field_notes",
]


class DataDictionaryError(ValueError):
    """The curated notes do not match the schema contracts (messages name fields, never data)."""


class FieldUse(StrEnum):
    """How a raw field participates in the analysis."""

    PRICING = "used_in_pricing"            # enters eligible offers: identity, period, grouping or price
    VALIDATION = "validation_only"         # checked by a control; never a price, identity or grouping value
    LINKAGE = "linkage_only"               # only used to derive the confidential linkage or position keys
    RETIRED = "retired_from_pricing"       # parse-checked and reported; approved as never a pricing field
    NOT_USED = "not_used"                  # preserved for audit; no approved semantics; no analysis reads it


FIELD_USE_DESCRIPTIONS: Mapping[FieldUse, str] = MappingProxyType({
    FieldUse.PRICING: "Enters the pricing-eligible offers (identity, scheduled period, grouping or price).",
    FieldUse.VALIDATION: "Checked by a control; never enters a price, product identity or grouping.",
    FieldUse.LINKAGE: "Used only to derive the confidential linkage or offer-position keys.",
    FieldUse.RETIRED: "Parsed and reported for quality only; approved as never a pricing field.",
    FieldUse.NOT_USED: "Preserved unchanged for audit; no approved semantics and no analysis reads it.",
})


@dataclass(frozen=True, slots=True)
class RawFieldNote:
    """Curated documentation of one raw field (no source values)."""

    dataset: DatasetKey
    field: str
    meaning: str
    representation: str
    missing_or_invalid: str
    use: FieldUse


_J, _C = DatasetKey.JOBS, DatasetKey.CARS
_ID_TEXT = "Nullable string (opaque text; leading zeros and full text kept; never trimmed, case-folded or parsed)."
_RENTAL = ("Exact `YYYY-MM-DD` text naming a real calendar date (no trimming, time or zone); return on or after "
           "pickup, same-day allowed, no maximum.")

#: Curated notes, in schema order per dataset.
RAW_FIELD_NOTES: tuple[RawFieldNote, ...] = (
    # ---------------------------------------------------------------- jobs
    RawFieldNote(_J, "job_id", "Identifier of one scrape (collection) job.", _ID_TEXT,
                 "Missing or whitespace-only values are invalid linkage keys; duplicates block the key contract. "
                 "Never rewritten; the derived linkage key is separate.", FieldUse.LINKAGE),
    RawFieldNote(_J, "city", "City whose collection the job belongs to; selects the approved IANA time zone.",
                 "Text matched exactly (never trimmed or aliased).",
                 "Missing, blank or padded values make the job scope unassignable and block city integrity, the "
                 "trusted join, completeness and pricing readiness.", FieldUse.PRICING),
    RawFieldNote(_J, "mode", "Collection mode label supplied by the source.", "Text, preserved as read.",
                 "Not evaluated beyond blank-row detection.", FieldUse.NOT_USED),
    RawFieldNote(_J, "status", "Collection status label supplied by the source.", "Text, preserved as read.",
                 "Not evaluated beyond blank-row detection.", FieldUse.NOT_USED),
    RawFieldNote(_J, "record_count", "Declared number of detail rows for the job.", "Non-negative whole number.",
                 "Missing or invalid values are counted separately and fail the reconciliation; never repaired.",
                 FieldUse.VALIDATION),
    RawFieldNote(_J, "pickup_date", "Rental pickup date searched by the job.", _RENTAL,
                 "Missing, malformed or out-of-order dates are counted and make the job's rows rental-date "
                 "ineligible; never repaired.", FieldUse.VALIDATION),
    RawFieldNote(_J, "return_date", "Rental return date searched by the job.", _RENTAL,
                 "As `pickup_date`.", FieldUse.VALIDATION),
    RawFieldNote(_J, "finished_at", "Scheduled capture timestamp: when the job finished, in the job city's local "
                 "wall-clock time.",
                 "Naive `%Y-%m-%d %H:%M:%S.%f` resolved in the approved city IANA zone to a full-precision UTC "
                 "instant; defines the scheduled capture period and the reporting day.",
                 "Missing, invalid, ambiguous (fall-back) and nonexistent (spring-forward) values are counted "
                 "separately, fail closed and leave the capture period unassigned.", FieldUse.PRICING),
    RawFieldNote(_J, "scrape_date", "Source-supplied calendar date of the job (approved meaning: the reporting "
                 "day).", "Strict ISO calendar date.",
                 "Must equal the derived reporting day; disagreement or invalid values make rows "
                 "reporting-day ineligible.", FieldUse.VALIDATION),
    RawFieldNote(_J, "actual_car_rows", "Second declared detail-row tally of the job.", "Non-negative whole number.",
                 "Reconciled independently of `record_count`; both must match the observed detail rows.",
                 FieldUse.VALIDATION),
    # ---------------------------------------------------------------- cars
    RawFieldNote(_C, "job_id", "Parent job identifier on each offer row.", _ID_TEXT,
                 "The historical export carries a legacy decimal-zero suffix; only the approved repair (exact match "
                 "first, then removing a final `.0` from an all-digit value with a unique parent) is applied, in the "
                 "derived key. Missing, unmatched, ambiguous or colliding references block linkage.",
                 FieldUse.LINKAGE),
    RawFieldNote(_C, "city", "City of the offer's collection; first component of the source stream key.",
                 "Text matched exactly.",
                 "Must equal the parent job's city; a disagreement blocks city integrity.", FieldUse.PRICING),
    RawFieldNote(_C, "mode", "Repeated collection mode label.", "Text, preserved as read.",
                 "Not evaluated beyond blank-row detection.", FieldUse.NOT_USED),
    RawFieldNote(_C, "status", "Repeated collection status label.", "Text, preserved as read.",
                 "Not evaluated beyond blank-row detection.", FieldUse.NOT_USED),
    RawFieldNote(_C, "job_finished_at", "Copy of the parent job's `finished_at`.",
                 "Naive `%Y-%m-%d %H:%M:%S.%f`, resolved with the parent city's zone.",
                 "Must replicate the parent exactly (wall time and instant); failures are counted and block "
                 "temporal trust. Never a schedule key.", FieldUse.VALIDATION),
    RawFieldNote(_C, "scrape_date", "Copy of the reporting day on the offer row.", "Strict ISO calendar date.",
                 "Must equal the parent reporting day; otherwise the row is reporting-day ineligible.",
                 FieldUse.VALIDATION),
    RawFieldNote(_C, "job_pickup_date", "Copy of the parent job's pickup date.", _RENTAL,
                 "Must equal the parent `pickup_date` as a parsed date; mismatches are counted, never repaired.",
                 FieldUse.VALIDATION),
    RawFieldNote(_C, "job_return_date", "Copy of the parent job's return date.", _RENTAL,
                 "Must equal the parent `return_date` as a parsed date; mismatches are counted, never repaired.",
                 FieldUse.VALIDATION),
    RawFieldNote(_C, "row_index", "Position of the offer in its job's result list.",
                 "Non-negative integer (the approved legacy `.0` form is repaired only in the derived key).",
                 "Missing or invalid values leave the derived position key missing and block the detail key "
                 "contract. Never part of product identity; source order is never used to combine offers.",
                 FieldUse.LINKAGE),
    RawFieldNote(_C, "pickup_date", "Rental pickup date of the offer.", _RENTAL,
                 "Must be valid and equal the parent pickup date; otherwise the row is rental-date ineligible.",
                 FieldUse.PRICING),
    RawFieldNote(_C, "return_date", "Rental return date of the offer.", _RENTAL,
                 "As `pickup_date`.", FieldUse.PRICING),
    RawFieldNote(_C, "car_name", "Vehicle product name; part of the approved product identity.",
                 "Text compared exactly (no fuzzy matching or normalization).",
                 "A missing identity component makes the offer unassessable, which blocks pricing readiness.",
                 FieldUse.PRICING),
    RawFieldNote(_C, "car_type", "Vehicle class; product identity and vehicle-type grouping.",
                 "Text compared exactly.",
                 "Required by vehicle-attribute stability; a conflict or missing value is reported, never filled.",
                 FieldUse.PRICING),
    RawFieldNote(_C, "price_per_day", "Listed price text carrying the currency marker, amount and price basis.",
                 "Strict `<marker>$<whole>.<cents>/<basis>` text (marker of up to three capital letters, optional "
                 "thousands separators, lowercase basis); the amount must equal `price_num`.",
                 "Unparseable text or an amount disagreeing with `price_num` makes the offer unassessable. "
                 "Currencies are never assumed equivalent.", FieldUse.PRICING),
    RawFieldNote(_C, "transmission", "Transmission; part of the approved product identity.",
                 "Text compared exactly.", "Required by vehicle-attribute stability.", FieldUse.PRICING),
    RawFieldNote(_C, "seats", "Seat count; part of the approved product identity.", "Value compared exactly.",
                 "Required by vehicle-attribute stability.", FieldUse.PRICING),
    RawFieldNote(_C, "bags", "Bag count; part of the approved product identity.", "Value compared exactly.",
                 "Presence must be stable for a product; missing values never compare equal.", FieldUse.PRICING),
    RawFieldNote(_C, "location", "Branch label; second component of the source stream key.",
                 "Exact source spelling (case, spacing and punctuation significant).",
                 "A missing, unexpected or misspelled stream blocks coverage and pricing; aliases apply only "
                 "through the approved Vancouver policy.", FieldUse.PRICING),
    RawFieldNote(_C, "scraped_at", "When the offer row was scraped (collector clock).",
                 "`%Y-%m-%d %H:%M:%S` with the `MST` designator, read as fixed UTC-07:00.",
                 "Must be earlier than or equal to the parent finish instant (zero tolerance). Never a schedule "
                 "key or reporting-day source; it orders observations for vehicle stability only.",
                 FieldUse.VALIDATION),
    RawFieldNote(_C, "price_num", "Numeric listed price.",
                 "Finite, non-negative number with at most two decimals, converted to exact integer cents.",
                 "Any other value makes the offer unassessable.", FieldUse.PRICING),
    RawFieldNote(_C, "city_clean", "Source-supplied cleaned city label.", "Text, preserved as read.",
                 "No documented authority: city integrity and stream keys use the raw `city`.", FieldUse.NOT_USED),
    RawFieldNote(_C, "date_clean", "Source-supplied cleaned date.", "Strict ISO calendar date.",
                 "Parse quality is reported; the approved decision retires it from pricing.", FieldUse.RETIRED),
)

#: Names of the generated blocks in ``docs/data_dictionary.md``, in document order.
GENERATED_SECTION_NAMES: tuple[str, ...] = ("raw-datasets", "raw-fields-jobs", "raw-fields-cars", "field-use")


def validate_raw_field_notes(notes: tuple[RawFieldNote, ...] = RAW_FIELD_NOTES) -> None:
    """Require exactly one note per schema field, per dataset, in schema order.

    Raises:
        DataDictionaryError: A field is missing, duplicated, unknown or out of order.
    """
    for key, definition in DATASET_DEFINITIONS.items():
        fields = tuple(n.field for n in notes if n.dataset is key)
        if fields != definition.columns:
            raise DataDictionaryError(f"{key.value}: notes must cover every schema field exactly once, in order")
    if any(n.dataset not in DATASET_DEFINITIONS for n in notes):
        raise DataDictionaryError("notes name an unknown dataset")
    for note in notes:
        if not isinstance(note.use, FieldUse) or not all((note.meaning, note.representation, note.missing_or_invalid)):
            raise DataDictionaryError(f"{note.dataset.value}.{note.field}: incomplete note")


def field_key_role(dataset: DatasetKey, field: str) -> str:
    """Key and relationship role of one raw field, derived from the schema contracts."""
    definition = DATASET_DEFINITIONS[dataset]
    rel = JOB_DETAIL_RELATIONSHIP
    roles = []
    if field in definition.unique_key_columns:
        roles.append("raw unique-key component")
    if dataset is rel.parent and field in rel.parent_key_columns:
        roles.append("parent side of the job-detail relationship")
    if dataset is rel.detail and field in rel.detail_key_columns:
        roles.append("detail reference to the parent job")
    if dataset is rel.parent and field in (rel.expected_detail_count_column, *rel.additional_expected_count_columns):
        roles.append("declared detail count")
    for parent_column, detail_column in rel.scope_agreement_columns:
        if (dataset is rel.parent and field == parent_column) or (dataset is rel.detail and field == detail_column):
            roles.append("parent/detail scope agreement")
    return "; ".join(roles) or "none"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _codes(columns: tuple[str, ...]) -> str:
    return ", ".join(f"`{c}`" for c in columns) or "none"


def _datasets_block() -> str:
    rel = JOB_DETAIL_RELATIONSHIP
    rows = ["| Dataset | Stage | Grain | Unique key | Identifier fields | Relationship | Status | Confidentiality |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    grains = {DatasetKey.JOBS: "One row per scrape (collection) job.",
              DatasetKey.CARS: "One row per offer position within one scrape job's result list."}
    relation = {DatasetKey.JOBS: f"Parent: one job has zero, one or many `{rel.detail.value}` rows.",
                DatasetKey.CARS: f"Detail: every row links to exactly one `{rel.parent.value}` row."}
    for stage, definitions, status in (("raw", DATASET_DEFINITIONS, "Source (immutable CSV export)"),
                                       ("analysis", ANALYSIS_DATASET_DEFINITIONS,
                                        "Derived in memory (raw columns plus derived keys)")):
        for key, definition in definitions.items():
            rows.append(f"| `{key.value}` | {stage} | {grains[key]} | {_codes(definition.unique_key_columns)} | "
                        f"{_codes(definition.identifier_columns)} | {relation[key]} | {status} | "
                        "Proprietary; never committed or displayed |")
    rows.append("")
    rows.append(f"Relationship keys: raw `{', '.join(rel.parent_key_columns)}` -> "
                f"`{', '.join(rel.detail_key_columns)}` (source form, reconciliation of the raw extract only); "
                f"analytical linkage uses the derived key of the analysis stage. Declared counts: "
                f"{_codes((rel.expected_detail_count_column, *rel.additional_expected_count_columns))}. Scope "
                "agreement: " + ", ".join(f"`{rel.parent.value}.{p}` = `{rel.detail.value}.{d}`"
                                           for p, d in rel.scope_agreement_columns) + ".")
    return "\n".join(rows)


def _fields_block(dataset: DatasetKey) -> str:
    definition = DATASET_DEFINITIONS[dataset]
    notes = {n.field: n for n in RAW_FIELD_NOTES if n.dataset is dataset}
    rows = ["| # | Field | Meaning | Representation / parsing | Identifier | Key / relationship role | "
            "Missing or invalid handling | Analytical use |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for position, column in enumerate(definition.columns, start=1):
        note = notes[column]
        identifier = "yes (nullable string)" if column in definition.identifier_columns else "no"
        rows.append(f"| {position} | `{column}` | {_cell(note.meaning)} | {_cell(note.representation)} | "
                    f"{identifier} | {field_key_role(dataset, column)} | {_cell(note.missing_or_invalid)} | "
                    f"`{note.use.value}` |")
    return "\n".join(rows)


def _use_block() -> str:
    rows = ["| Analytical use | Meaning | jobs fields | cars fields |", "| --- | --- | --- | --- |"]
    for use in FieldUse:
        fields = {key: tuple(n.field for n in RAW_FIELD_NOTES if n.dataset is key and n.use is use)
                  for key in DATASET_DEFINITIONS}
        rows.append(f"| `{use.value}` | {FIELD_USE_DESCRIPTIONS[use]} | {_codes(fields[DatasetKey.JOBS])} | "
                    f"{_codes(fields[DatasetKey.CARS])} |")
    return "\n".join(rows)


def render_generated_sections() -> dict[str, str]:
    """The generated Markdown blocks, by name (structure from the contracts; no data is read)."""
    validate_raw_field_notes()
    return {"raw-datasets": _datasets_block(), "raw-fields-jobs": _fields_block(DatasetKey.JOBS),
            "raw-fields-cars": _fields_block(DatasetKey.CARS), "field-use": _use_block()}


def _block_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->)(.*?)(<!-- END GENERATED: "
                      rf"{re.escape(name)} -->)", re.DOTALL)


def replace_generated_sections(document: str, sections: Mapping[str, str] | None = None) -> str:
    """``document`` with every generated block replaced by its current rendering.

    ``sections`` defaults to :func:`render_generated_sections`; other documents
    pass their own blocks (for example the final-report question catalogs).

    Raises:
        DataDictionaryError: A block's markers are missing or repeated.
    """
    for name, body in (render_generated_sections() if sections is None else sections).items():
        pattern = _block_pattern(name)
        if len(pattern.findall(document)) != 1:
            raise DataDictionaryError(f"generated block {name} must appear exactly once")
        document = pattern.sub(lambda m, b=body: f"{m.group(1)}\n{b}\n{m.group(3)}", document)
    return document


def main() -> int:  # pragma: no cover - thin command-line wrapper
    for name, body in render_generated_sections().items():
        print(f"<!-- BEGIN GENERATED: {name} -->\n{body}\n<!-- END GENERATED: {name} -->\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
