"""Authority-backed job linkage and offer-position normalization.

The raw ``job_id`` is **opaque text** (collection-owner decision, recorded in
``docs/decisions/governance/job-identifier-governance-2026-10-06.md`` and
approved in pricing-authority record ``v2``). The historical detail export
serialises it with a spreadsheet decimal-zero defect (``"123"`` on the job
side, ``"123.0"`` on the detail side), and ``row_index`` likewise. This module
derives **separate** analysis-stage keys under the approved decisions; it
never rewrites, trims, case-folds or numerically parses a raw identifier and
never edits the raw files.

Pipeline position: after source-column validation, blank-row removal and
identifier-dtype validation; before the analytical unique-key,
reconciliation, relationship, stream, city-integrity and trusted-join
assessments, which all use :data:`~ql2_sixt_canada_analysis.schemas.ANALYSIS_JOB_DETAIL_RELATIONSHIP`.

Job linkage key (:data:`~ql2_sixt_canada_analysis.schemas.JOB_LINKAGE_KEY_COLUMN`)
--------------------------------------------------------------------------
* Jobs: the raw value unchanged. Missing and whitespace-only values are
  invalid keys (``<NA>``).
* Details, exact match first: the unchanged raw value is a candidate.
* Legacy repair (only when the approved policy enables it): a detail value
  consisting **entirely** of ASCII digits followed by exactly ``.0`` also
  yields the candidate with that final ``.0`` removed - every preceding
  character, including leading zeros, is kept. Nothing else is ever
  rewritten: alphanumeric or arbitrary text ending in ``.0``, signs,
  exponents, other fractions and surrounding whitespace stay as they are and
  can match only exactly.
* Exactly one candidate must identify exactly one parent. Both candidates
  matching (different parents), a match to a duplicated parent key, no match,
  or several raw representations resolving to the same parent (a collision)
  leave the key ``<NA>`` and block linkage - the code never guesses.

Why numeric parsing is forbidden: converting through ``int``/``float`` drops
leading zeros, rounds long values, accepts signs, exponents and whitespace,
and silently merges distinct identifiers. Matching here is string comparison
only, so ``"007"`` and ``"7"`` remain different jobs.

Offer position key (:data:`~ql2_sixt_canada_analysis.schemas.OFFER_POSITION_KEY_COLUMN`)
-----------------------------------------------------------------------------
``row_index`` is a non-negative integer. Accepted: integer values; text made
of ASCII digits; when the policy enables the legacy repair, text of ASCII
digits followed by exactly ``.0`` (derived without float coercion); and values
pandas already inferred as numbers when finite, non-negative, exactly integral
and not boolean. Rejected: missing values, non-zero fractions, negatives,
signed or exponent text, surrounding whitespace, non-finite values, booleans
and arbitrary text. The derived key is pandas nullable ``Int64``; the raw
column is kept unchanged. Duplicate derived keys are a key-contract failure
downstream, never repaired here.

Outputs and trust
-----------------
:func:`assess_job_linkage` returns a :class:`JobLinkageResult`: an aggregate
:class:`JobLinkageReport` (counts and statuses only) and new frames with the
derived columns appended. Derived keys are ``<NA>`` wherever the approved
policy does not resolve them - everywhere when no policy is available - so
downstream contracts fail closed instead of linking on raw text. There is no
default policy: :func:`job_linkage_policy_from_record` builds one only from a
schema-2 record whose four job-identifier decisions are all APPROVED and
consistent. :func:`require_job_linkage` raises :class:`JobLinkageNotReadyError`
unless the report is valid.

Confidentiality: raw identifiers, candidates, linkage keys and offer
positions are confidential technical fields
(:data:`~ql2_sixt_canada_analysis.schemas.CONFIDENTIAL_TECHNICAL_COLUMNS`).
Reports and errors never contain them; never print, log or export the frames.
"""

from __future__ import annotations

import math
import numbers
import re
from collections import Counter
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from ql2_sixt_canada_analysis.authority_decisions import (
    CURRENT_RECORD_PATH,
    JOB_IDENTIFIER_DECISIONS,
    LEGACY_DECIMAL_ZERO_REPAIR,
    OPAQUE_TEXT_IDENTIFIER_POLICY,
    AuthorityDecisionRecord,
    AuthorityKind,
    AuthorityReference,
    DecisionId,
    DecisionRecordError,
    DecisionStatus,
    load_decision_record,
)
from ql2_sixt_canada_analysis.identifiers import is_identifier_dtype
from ql2_sixt_canada_analysis.ingestion import RawDatasets
from ql2_sixt_canada_analysis.quality import completely_blank_row_mask
from ql2_sixt_canada_analysis.relationships import RelationshipPreconditionError
from ql2_sixt_canada_analysis.schemas import (
    IDENTIFIER_DTYPE,
    JOB_LINKAGE_KEY_COLUMN,
    OFFER_POSITION_KEY_COLUMN,
    OFFER_POSITION_KEY_DTYPE,
    SOURCE_JOB_IDENTIFIER_COLUMN,
    SOURCE_OFFER_POSITION_COLUMN,
    DatasetKey,
)

__all__ = [
    "JobLinkageBlocker",
    "JobLinkageNotReadyError",
    "JobLinkagePolicy",
    "JobLinkagePolicyError",
    "JobLinkagePolicyStatus",
    "JobLinkagePreconditionError",
    "JobLinkageReport",
    "JobLinkageResult",
    "assess_job_linkage",
    "job_linkage_policy_from_record",
    "load_job_linkage_policy",
    "require_job_linkage",
]

#: ASCII digits followed by exactly one ".0" (the only rewritable job-identifier form).
_LEGACY_JOB_IDENTIFIER = re.compile(r"[0-9]+\.0")
_DIGITS = re.compile(r"[0-9]+")
_INT64_MAX = int(np.iinfo(np.int64).max)
_DERIVED = (JOB_LINKAGE_KEY_COLUMN, OFFER_POSITION_KEY_COLUMN)
_AUTHORITY_KINDS = frozenset({AuthorityKind.COLLECTION_OWNER, AuthorityKind.SUPPLIER})


class JobLinkagePolicyError(ValueError):
    """A linkage policy was constructed inconsistently (message holds no values)."""


class JobLinkagePreconditionError(RelationshipPreconditionError):
    """A structural precondition for linkage failed (``reason``: a category, never a value)."""

    control = "Job linkage normalization"


class JobLinkagePolicyStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class JobLinkageBlocker(StrEnum):
    """Why linkage is not trusted (values avoid source column names)."""

    POLICY_UNAVAILABLE = "job_linkage_policy_unavailable"
    PARENT_KEY_INVALID = "parent_linkage_key_invalid"
    PARENT_KEY_NOT_UNIQUE = "parent_linkage_key_not_unique"
    DETAIL_REFERENCE_MISSING = "detail_job_reference_missing"
    DETAIL_REFERENCE_UNMATCHED = "detail_job_reference_unmatched"
    DETAIL_REFERENCE_AMBIGUOUS = "detail_job_reference_ambiguous"
    LINKAGE_COLLISION = "job_linkage_collision"
    OFFER_POSITION_MISSING = "offer_position_missing"
    OFFER_POSITION_INVALID = "offer_position_invalid"


@dataclass(frozen=True, slots=True)
class JobLinkagePolicy:
    """The approved linkage policy (opaque text; exact match first; no permissive default).

    Build it with :func:`job_linkage_policy_from_record`; direct construction
    still requires attributable collection-owner or supplier authority.
    """

    record_id: str
    authority: tuple[AuthorityReference, ...]
    legacy_decimal_zero_repair: bool
    legacy_offer_position_repair: bool
    semantics: str = "OPAQUE_TEXT"

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise JobLinkagePolicyError("a decision-record id is required")
        if (not isinstance(self.authority, tuple) or not self.authority
                or not all(isinstance(a, AuthorityReference) and a.kind in _AUTHORITY_KINDS for a in self.authority)):
            raise JobLinkagePolicyError("collection-owner or supplier authority is required")
        if type(self.legacy_decimal_zero_repair) is not bool or type(self.legacy_offer_position_repair) is not bool:
            raise JobLinkagePolicyError("repair flags must be booleans")
        if self.semantics != "OPAQUE_TEXT":
            raise JobLinkagePolicyError("only opaque-text identifier semantics are supported")


@dataclass(frozen=True, slots=True)
class JobLinkageReport:
    """Aggregate linkage result: counts and statuses only (no identifiers, no rows)."""

    policy_status: JobLinkagePolicyStatus
    parent_row_count: int
    detail_row_count: int
    parent_key_invalid_count: int
    parent_key_duplicate_row_count: int
    exact_match_count: int
    decimal_zero_repair_count: int
    missing_identifier_count: int
    unmatched_identifier_count: int
    ambiguous_identifier_count: int
    collision_count: int
    valid_row_index_count: int
    legacy_row_index_repair_count: int
    missing_row_index_count: int
    invalid_row_index_count: int
    blocking_reasons: tuple[JobLinkageBlocker, ...]

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith("_count") and (type(value) is not int or value < 0):
                raise TypeError(f"{f.name} must be a non-negative int")

    @property
    def policy_available(self) -> bool:
        return self.policy_status is JobLinkagePolicyStatus.AVAILABLE

    @property
    def linked_detail_count(self) -> int:
        return self.exact_match_count + self.decimal_zero_repair_count

    @property
    def is_valid(self) -> bool:
        """True only with an available policy, no blocker and every row keyed."""
        return (self.policy_available and not self.blocking_reasons
                and self.linked_detail_count == self.detail_row_count
                and self.valid_row_index_count == self.detail_row_count)


@dataclass(frozen=True, slots=True, eq=False)
class JobLinkageResult:
    """The report plus analysis-stage frames (exposed only as copies)."""

    report: JobLinkageReport
    _jobs: pd.DataFrame
    _cars: pd.DataFrame

    @property
    def jobs(self) -> pd.DataFrame:
        return self._jobs.copy()

    @property
    def cars(self) -> pd.DataFrame:
        return self._cars.copy()

    @property
    def is_valid(self) -> bool:
        return self.report.is_valid

    def datasets(self, source: RawDatasets | None = None) -> RawDatasets:
        """The analysis frames as :class:`RawDatasets` (``complete_source`` carried from ``source``)."""
        complete = bool(source.complete_source) if isinstance(source, RawDatasets) else False
        return RawDatasets(jobs=self.jobs, cars=self.cars, complete_source=complete)


class JobLinkageNotReadyError(Exception):
    """Strict linkage failed; ``blocking_reasons`` holds categories only."""

    def __init__(self, report: JobLinkageReport) -> None:
        super().__init__("Job linkage not ready: " + ", ".join(b.value for b in report.blocking_reasons) + ".")
        self.report = report
        self.blocking_reasons = report.blocking_reasons


# --------------------------------------------------------------------- policy


def job_linkage_policy_from_record(record: object) -> JobLinkagePolicy | None:
    """The policy from APPROVED, consistent schema-2 decisions; ``None`` otherwise (fail closed).

    ``None`` for anything but a validated :class:`AuthorityDecisionRecord`, a
    schema-1 record (its boolean shapes cannot express opaque-text semantics),
    any of the four job-identifier decisions not APPROVED, or any resolution
    or authority that differs from the supported policy.
    """
    if not isinstance(record, AuthorityDecisionRecord) or record.schema_version < 2:
        return None
    entries = [record.decision(d) for d in JOB_IDENTIFIER_DECISIONS]
    if any(e.status is not DecisionStatus.APPROVED or e.resolution is None for e in entries):
        return None
    res = {e.id: dict(e.resolution) for e in entries}                     # type: ignore[arg-type]
    if res[DecisionId.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS] != dict(OPAQUE_TEXT_IDENTIFIER_POLICY):
        return None
    if res[DecisionId.JOB_ID_LEADING_ZERO_SIGNIFICANCE] != {"leading_zeros_significant": True}:
        return None
    if res[DecisionId.JOB_ID_RAW_AND_LINKAGE_PRESERVATION] != {"preserve_raw_identifier": True,
                                                               "separate_linkage_key": True}:
        return None
    decimal = res[DecisionId.JOB_ID_DECIMAL_ZERO_EQUIVALENCE]
    if decimal == dict(LEGACY_DECIMAL_ZERO_REPAIR):
        repair = True
    elif decimal == {"equivalence": "NOT_EQUIVALENT"}:
        repair = False
    else:
        return None
    authority = tuple(dict.fromkeys(a for e in entries for a in e.authority))
    if not authority or not all(a.kind in _AUTHORITY_KINDS for a in authority):
        return None
    try:
        return JobLinkagePolicy(record_id=record.record_id, authority=authority,
                                legacy_decimal_zero_repair=repair, legacy_offer_position_repair=repair)
    except JobLinkagePolicyError:
        return None


def load_job_linkage_policy(path: str | Path | None = None) -> JobLinkagePolicy | None:
    """Policy from the current committed record (or ``path``); ``None`` if invalid or not approved."""
    from ql2_sixt_canada_analysis.paths import PROJECT_ROOT

    target = Path(path) if path is not None else PROJECT_ROOT / CURRENT_RECORD_PATH
    try:
        return job_linkage_policy_from_record(load_decision_record(target))
    except DecisionRecordError:
        return None


# ------------------------------------------------------------------ assessment


def assess_job_linkage(jobs: pd.DataFrame, cars: pd.DataFrame, policy: JobLinkagePolicy | None) -> JobLinkageResult:
    """Derive the analysis-stage keys under ``policy`` (inputs are never modified).

    ``policy`` has no default: ``None`` (no approved policy) yields an
    ``unavailable`` report and all-``<NA>`` derived keys.

    Raises:
        TypeError: Invalid argument types.
        JobLinkagePreconditionError: A source column is absent, an identifier
            column is not the identifier dtype, or completely blank rows remain.
    """
    if not isinstance(jobs, pd.DataFrame) or not isinstance(cars, pd.DataFrame):
        raise TypeError("jobs and cars must be pandas DataFrames")
    if policy is not None and not isinstance(policy, JobLinkagePolicy):
        raise TypeError("policy must be a JobLinkagePolicy or None")
    for frame, columns, role in ((jobs, (SOURCE_JOB_IDENTIFIER_COLUMN,), DatasetKey.JOBS),
                                 (cars, (SOURCE_JOB_IDENTIFIER_COLUMN, SOURCE_OFFER_POSITION_COLUMN), DatasetKey.CARS)):
        if not all(c in frame.columns for c in columns):
            raise JobLinkagePreconditionError("source_column_missing", role)
        if not is_identifier_dtype(frame[SOURCE_JOB_IDENTIFIER_COLUMN].dtype):
            raise JobLinkagePreconditionError("identifier_dtype", role)
        if bool(completely_blank_row_mask(frame.drop(columns=[c for c in _DERIVED if c in frame.columns])).any()):
            raise JobLinkagePreconditionError("blank_rows_present", role)

    parent_raw = _texts(jobs[SOURCE_JOB_IDENTIFIER_COLUMN])
    detail_raw = _texts(cars[SOURCE_JOB_IDENTIFIER_COLUMN])
    positions = cars[SOURCE_OFFER_POSITION_COLUMN]
    if policy is None:
        report = JobLinkageReport(
            policy_status=JobLinkagePolicyStatus.UNAVAILABLE, parent_row_count=len(jobs), detail_row_count=len(cars),
            parent_key_invalid_count=0, parent_key_duplicate_row_count=0, exact_match_count=0,
            decimal_zero_repair_count=0, missing_identifier_count=0, unmatched_identifier_count=0,
            ambiguous_identifier_count=0, collision_count=0, valid_row_index_count=0,
            legacy_row_index_repair_count=0, missing_row_index_count=0, invalid_row_index_count=0,
            blocking_reasons=(JobLinkageBlocker.POLICY_UNAVAILABLE,))
        return JobLinkageResult(report, _with(jobs, {JOB_LINKAGE_KEY_COLUMN: [None] * len(jobs)}),
                                _with(cars, {JOB_LINKAGE_KEY_COLUMN: [None] * len(cars),
                                             OFFER_POSITION_KEY_COLUMN: [None] * len(cars)}))

    # Parent keys: the raw value unchanged; missing / whitespace-only values are invalid.
    parent_keys = [v if _usable(v) else None for v in parent_raw]
    counts = Counter(k for k in parent_keys if k is not None)
    duplicated = {k for k, n in counts.items() if n > 1}

    # Detail resolution per distinct raw value (exact first, then the legacy repair).
    status: dict[object, tuple[str, str | None]] = {}
    for value in dict.fromkeys(detail_raw):
        status[value] = _resolve(value, counts, duplicated, policy.legacy_decimal_zero_repair)
    by_key: dict[str, set] = {}
    for value, (state, key) in status.items():
        if key is not None:
            by_key.setdefault(key, set()).add(value)
    for key, values in by_key.items():
        if len(values) > 1:                           # several raw forms -> one parent: never guess
            for value in values:
                status[value] = ("collision", None)
    states = [status[v][0] for v in detail_raw]
    detail_keys = [status[v][1] for v in detail_raw]

    offer_keys, offer_states = _offer_positions(positions, policy.legacy_offer_position_repair)
    tally, offer_tally = Counter(states), Counter(offer_states)
    report_counts = dict(
        parent_key_invalid_count=sum(k is None for k in parent_keys),
        parent_key_duplicate_row_count=sum(k in duplicated for k in parent_keys if k is not None),
        exact_match_count=tally["exact"], decimal_zero_repair_count=tally["repaired"],
        missing_identifier_count=tally["missing"], unmatched_identifier_count=tally["unmatched"],
        ambiguous_identifier_count=tally["ambiguous"], collision_count=tally["collision"],
        valid_row_index_count=offer_tally["valid"] + offer_tally["legacy"],
        legacy_row_index_repair_count=offer_tally["legacy"],
        missing_row_index_count=offer_tally["missing"], invalid_row_index_count=offer_tally["invalid"])
    B = JobLinkageBlocker
    blockers = [b for b, n in ((B.PARENT_KEY_INVALID, report_counts["parent_key_invalid_count"]),
                               (B.PARENT_KEY_NOT_UNIQUE, report_counts["parent_key_duplicate_row_count"]),
                               (B.DETAIL_REFERENCE_MISSING, report_counts["missing_identifier_count"]),
                               (B.DETAIL_REFERENCE_UNMATCHED, report_counts["unmatched_identifier_count"]),
                               (B.DETAIL_REFERENCE_AMBIGUOUS, report_counts["ambiguous_identifier_count"]),
                               (B.LINKAGE_COLLISION, report_counts["collision_count"]),
                               (B.OFFER_POSITION_MISSING, report_counts["missing_row_index_count"]),
                               (B.OFFER_POSITION_INVALID, report_counts["invalid_row_index_count"])) if n]
    report = JobLinkageReport(policy_status=JobLinkagePolicyStatus.AVAILABLE, parent_row_count=len(jobs),
                              detail_row_count=len(cars), blocking_reasons=tuple(blockers), **report_counts)
    return JobLinkageResult(report, _with(jobs, {JOB_LINKAGE_KEY_COLUMN: parent_keys}),
                            _with(cars, {JOB_LINKAGE_KEY_COLUMN: detail_keys, OFFER_POSITION_KEY_COLUMN: offer_keys}))


def require_job_linkage(jobs: pd.DataFrame, cars: pd.DataFrame, policy: JobLinkagePolicy | None) -> JobLinkageResult:
    """Strict variant: the result, or :class:`JobLinkageNotReadyError` (policy missing or any blocker)."""
    result = assess_job_linkage(jobs, cars, policy)
    if not result.report.is_valid:
        raise JobLinkageNotReadyError(result.report)
    return result


# --------------------------------------------------------------------- helpers


def _texts(series: pd.Series) -> list[object]:
    """Raw values as Python ``str`` or ``None`` (no conversion of present text)."""
    return [None if v is pd.NA or v is None else v for v in series.astype(object).tolist()]


def _usable(value: object) -> bool:
    """Present, non-empty and not whitespace-only (the value itself is never trimmed)."""
    return isinstance(value, str) and value.strip() != ""


def _resolve(value: object, parents: Counter, duplicated: set, repair: bool) -> tuple[str, str | None]:
    if not _usable(value):
        return "missing", None
    exact = value in parents
    candidate = value[:-2] if repair and _LEGACY_JOB_IDENTIFIER.fullmatch(value) else None   # type: ignore[index]
    repaired = candidate is not None and candidate in parents
    if exact and repaired:
        return "ambiguous", None                      # the two candidates name different parents
    if exact:
        return ("ambiguous", None) if value in duplicated else ("exact", value)    # type: ignore[return-value]
    if repaired:
        return ("ambiguous", None) if candidate in duplicated else ("repaired", candidate)
    return "unmatched", None


def _offer_positions(series: pd.Series, legacy_repair: bool) -> tuple[list[int | None], list[str]]:
    """(derived integers, per-row state in valid/legacy/missing/invalid) - no float coercion of text."""
    if pd.api.types.is_bool_dtype(series.dtype):
        states = ["missing" if v is pd.NA or v is None else "invalid" for v in series.astype(object).tolist()]
        return [None] * len(series), states
    keys: list[int | None] = []
    states: list[str] = []
    for value in series.astype(object).tolist():
        key, state = _offer_position(value, legacy_repair)
        keys.append(key)
        states.append(state)
    return keys, states


def _offer_position(value: object, legacy_repair: bool) -> tuple[int | None, str]:
    if value is None or value is pd.NA or value is pd.NaT:
        return None, "missing"
    if isinstance(value, (bool, np.bool_)):
        return None, "invalid"
    if isinstance(value, numbers.Integral):
        number = int(value)
        return (number, "valid") if 0 <= number <= _INT64_MAX else (None, "invalid")
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isnan(number):
            return None, "missing"
        if not math.isfinite(number) or number < 0 or not number.is_integer() or number > _INT64_MAX:
            return None, "invalid"
        return int(number), "valid"
    if isinstance(value, str):
        if value == "":
            return None, "missing"
        if _DIGITS.fullmatch(value):
            number = int(value)
        elif legacy_repair and _LEGACY_JOB_IDENTIFIER.fullmatch(value):
            number = int(value[:-2])
            return (number, "legacy") if number <= _INT64_MAX else (None, "invalid")
        else:
            return None, "invalid"
        return (number, "valid") if number <= _INT64_MAX else (None, "invalid")
    return None, "invalid"


def _with(frame: pd.DataFrame, derived: dict[str, list]) -> pd.DataFrame:
    """A new frame: source columns unchanged (any stale derived columns dropped), derived ones appended."""
    out = frame.drop(columns=[c for c in _DERIVED if c in frame.columns]).copy()
    for column, values in derived.items():
        dtype = IDENTIFIER_DTYPE if column == JOB_LINKAGE_KEY_COLUMN else OFFER_POSITION_KEY_DTYPE
        out[column] = pd.array([pd.NA if v is None else v for v in values], dtype=dtype)
    return out
