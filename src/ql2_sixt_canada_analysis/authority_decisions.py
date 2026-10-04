"""Versioned pricing-authority decision records: load, validate, summarize.

A decision record (``docs/decisions/pricing_authorities/v<N>.toml``) lists every
external decision needed before the dataset can be pricing ready. Each atomic
decision (:class:`DecisionId`) has exactly one :class:`DecisionStatus`:

* ``PROPOSED`` - unresolved. It names the responsible authority role(s), is
  ``blocking_external_input = true``, states the question the authority must
  answer and carries **no** resolution. It can never be consumed as
  production authority.
* ``APPROVED`` / ``REJECTED`` - terminal authority decisions. Both require an
  :class:`AuthorityReference` (supplier, collection owner or business owner,
  with a responsible source and a durable reference) for the responsible
  role(s); ``APPROVED`` also requires a complete, validated resolution for its
  decision type, ``REJECTED`` must state what was rejected.

Non-authoritative evidence (:class:`EvidenceKind`: raw-data observations,
behavioural analyses, repository notes, issue or review notes) may inform a
proposal but never satisfies authority provenance: authority is a separate,
typed field and evidence kinds are rejected there. Changing a status string
alone therefore cannot import an observation into production - the record
then fails validation.

Validation fails closed: unknown, missing or duplicate decisions, unsupported
versions, malformed values and content that looks like source-level data
(timestamps, prices, long identifiers) raise :class:`DecisionRecordError`,
whose messages name categories, decision ids and field names only - never
record values.

This module records decisions; it implements none of them. Production
contracts consume approved decisions only in separate implementation work.

Validate a revision (prints a sanitized status summary; non-zero exit if invalid)::

    python -m ql2_sixt_canada_analysis.authority_decisions docs/decisions/pricing_authorities/v1.toml
"""

from __future__ import annotations

import datetime as dt
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ql2_sixt_canada_analysis.schemas import (
    COMPARED_LOCATION_STREAMS,
    DATASET_DEFINITIONS,
    JOB_DETAIL_RELATIONSHIP,
    DatasetKey,
)

__all__ = [
    "CURRENT_RECORD_PATH",
    "SUPPORTED_SCHEMA_VERSIONS",
    "AuthorityDecisionRecord",
    "AuthorityKind",
    "AuthorityReference",
    "DecisionEntry",
    "DecisionId",
    "DecisionRecordError",
    "DecisionStatus",
    "EvidenceKind",
    "EvidenceReference",
    "LocationRoleDecision",
    "load_decision_record",
    "parse_decision_record",
    "render_authority_request_checklist",
    "render_status_summary",
    "validate_decision_record",
]

#: Record schema versions this module understands.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
#: The current committed revision (repository-relative).
CURRENT_RECORD_PATH = Path("docs/decisions/pricing_authorities/v1.toml")


class DecisionRecordError(ValueError):
    """The record is invalid. The message names categories, ids and fields - never values."""


class DecisionStatus(StrEnum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class AuthorityKind(StrEnum):
    """Who can decide (the only kinds that satisfy authority provenance)."""

    SUPPLIER = "SUPPLIER"
    COLLECTION_OWNER = "COLLECTION_OWNER"
    BUSINESS_OWNER = "BUSINESS_OWNER"


class EvidenceKind(StrEnum):
    """Non-authoritative evidence kinds: may inform a proposal, never approve or reject it."""

    RAW_DATA_OBSERVATION = "RAW_DATA_OBSERVATION"
    BEHAVIORAL_ANALYSIS = "BEHAVIORAL_ANALYSIS"
    REPOSITORY_IMPLEMENTATION_NOTE = "REPOSITORY_IMPLEMENTATION_NOTE"
    ISSUE_OR_REVIEW_NOTE = "ISSUE_OR_REVIEW_NOTE"


class LocationRoleDecision(StrEnum):
    AIRPORT = "AIRPORT"
    DOWNTOWN = "DOWNTOWN"
    OTHER = "OTHER"


class DecisionId(StrEnum):
    """Every required atomic decision (record order)."""

    JOB_ID_DECIMAL_ZERO_EQUIVALENCE = "JOB_ID_DECIMAL_ZERO_EQUIVALENCE"
    JOB_ID_LEADING_ZERO_SIGNIFICANCE = "JOB_ID_LEADING_ZERO_SIGNIFICANCE"
    JOB_ID_INVALID_NUMERIC_REPRESENTATIONS = "JOB_ID_INVALID_NUMERIC_REPRESENTATIONS"
    JOB_ID_RAW_AND_LINKAGE_PRESERVATION = "JOB_ID_RAW_AND_LINKAGE_PRESERVATION"
    EXPECTED_STREAM_UNIVERSE = "EXPECTED_STREAM_UNIVERSE"
    EXPECTED_STREAM_SOURCE_SPELLING = "EXPECTED_STREAM_SOURCE_SPELLING"
    LOCATION_ROLE_ASSIGNMENTS = "LOCATION_ROLE_ASSIGNMENTS"
    VALID_LOCATION_COMPARISON_PAIRS = "VALID_LOCATION_COMPARISON_PAIRS"
    VANCOUVER_LOCATION_IDENTITY = "VANCOUVER_LOCATION_IDENTITY"
    SCHEDULE_CAPTURE_TIMESTAMP = "SCHEDULE_CAPTURE_TIMESTAMP"
    SCHEDULE_EXPECTED_PERIODS = "SCHEDULE_EXPECTED_PERIODS"
    SCHEDULE_SHARING_MODEL = "SCHEDULE_SHARING_MODEL"
    SCHEDULE_EXCEPTIONS = "SCHEDULE_EXCEPTIONS"
    FINISHED_AT_TIMEZONE = "FINISHED_AT_TIMEZONE"
    SCRAPED_FINISHED_ORDERING = "SCRAPED_FINISHED_ORDERING"
    SCRAPED_FINISHED_TOLERANCE = "SCRAPED_FINISHED_TOLERANCE"
    REPORTING_DAY_SOURCE = "REPORTING_DAY_SOURCE"
    REPORTING_DAY_TIMEZONE = "REPORTING_DAY_TIMEZONE"
    SCRAPE_DATE_SEMANTICS = "SCRAPE_DATE_SEMANTICS"
    DATE_CLEAN_SEMANTICS = "DATE_CLEAN_SEMANTICS"
    RENTAL_DATE_VALIDITY = "RENTAL_DATE_VALIDITY"
    RENTAL_DATE_PARENT_DETAIL_AGREEMENTS = "RENTAL_DATE_PARENT_DETAIL_AGREEMENTS"


_D, _A = DecisionId, AuthorityKind
_CO_SUP = (_A.COLLECTION_OWNER, _A.SUPPLIER)
#: Who is responsible (roles) and whether every listed role must decide jointly.
REQUIRED_DECISIONS: Mapping[DecisionId, tuple[tuple[AuthorityKind, ...], bool]] = MappingProxyType({
    _D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE: (_CO_SUP, False),
    _D.JOB_ID_LEADING_ZERO_SIGNIFICANCE: (_CO_SUP, False),
    _D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS: (_CO_SUP, False),
    _D.JOB_ID_RAW_AND_LINKAGE_PRESERVATION: (_CO_SUP, False),
    _D.EXPECTED_STREAM_UNIVERSE: ((_A.COLLECTION_OWNER, _A.BUSINESS_OWNER), True),
    _D.EXPECTED_STREAM_SOURCE_SPELLING: (_CO_SUP, False),
    _D.LOCATION_ROLE_ASSIGNMENTS: ((_A.BUSINESS_OWNER,), False),
    _D.VALID_LOCATION_COMPARISON_PAIRS: ((_A.BUSINESS_OWNER,), False),
    _D.VANCOUVER_LOCATION_IDENTITY: (_CO_SUP, False),
    _D.SCHEDULE_CAPTURE_TIMESTAMP: (_CO_SUP, False),
    _D.SCHEDULE_EXPECTED_PERIODS: (_CO_SUP, False),
    _D.SCHEDULE_SHARING_MODEL: (_CO_SUP, False),
    _D.SCHEDULE_EXCEPTIONS: ((_A.COLLECTION_OWNER, _A.BUSINESS_OWNER), True),
    _D.FINISHED_AT_TIMEZONE: (_CO_SUP, False),
    _D.SCRAPED_FINISHED_ORDERING: (_CO_SUP, False),
    _D.SCRAPED_FINISHED_TOLERANCE: ((_A.COLLECTION_OWNER, _A.BUSINESS_OWNER), True),
    _D.REPORTING_DAY_SOURCE: ((_A.BUSINESS_OWNER,), False),
    _D.REPORTING_DAY_TIMEZONE: ((_A.BUSINESS_OWNER,), False),
    _D.SCRAPE_DATE_SEMANTICS: (_CO_SUP, False),
    _D.DATE_CLEAN_SEMANTICS: (_CO_SUP, False),
    _D.RENTAL_DATE_VALIDITY: ((_A.COLLECTION_OWNER, _A.BUSINESS_OWNER), True),
    _D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS: (_CO_SUP, False),
})

_PARENT, _DETAIL = JOB_DETAIL_RELATIONSHIP.parent, JOB_DETAIL_RELATIONSHIP.detail
#: The six rental-date fields whose relationships must be approved pair by pair.
RENTAL_DATE_PARENT_FIELDS = (f"{_PARENT}.pickup_date", f"{_PARENT}.return_date")
RENTAL_DATE_DETAIL_FIELDS = (f"{_DETAIL}.job_pickup_date", f"{_DETAIL}.job_return_date",
                             f"{_DETAIL}.pickup_date", f"{_DETAIL}.return_date")
_TIMESTAMP_FIELDS = (f"{_PARENT}.finished_at", f"{_DETAIL}.job_finished_at", f"{_DETAIL}.scraped_at")

_CODE = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_EXPLICIT_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")
#: Content that looks like source-level data is refused in free text.
_SENSITIVE = (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"),  # timestamps
              re.compile(r"[$€£]\s?\d|\b\d+\.\d{2}\b"),       # prices
              re.compile(r"\d{6,}"))                          # long identifiers
_TEXT_MAX = 600


# ------------------------------------------------------------------ value types


@dataclass(frozen=True, slots=True)
class AuthorityReference:
    """An actual authority decision source (never a person or reference invented by code)."""

    kind: AuthorityKind
    source: str
    reference: str
    note: str | None = None
    effective_date: dt.date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AuthorityKind):
            raise DecisionRecordError("authority kind must be SUPPLIER, COLLECTION_OWNER or BUSINESS_OWNER")
        for name in ("source", "reference"):
            _text(getattr(self, name), f"authority {name}")
        if self.note is not None:
            _text(self.note, "authority note")
        if self.effective_date is not None and not isinstance(self.effective_date, dt.date):
            raise DecisionRecordError("authority effective_date must be a date")


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """Non-authoritative evidence (informs a proposal; never authority)."""

    kind: EvidenceKind
    summary: str
    reference: str
    observed_candidates: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceKind):
            raise DecisionRecordError("evidence kind is not a known non-authoritative kind")
        _text(self.summary, "evidence summary")
        _text(self.reference, "evidence reference")
        for key in self.observed_candidates:
            _stream_key(key, "evidence observed_candidates")


@dataclass(frozen=True, slots=True)
class DecisionEntry:
    """One atomic decision. ``resolution`` is a read-only mapping (``None`` unless APPROVED)."""

    id: DecisionId
    status: DecisionStatus
    responsible_authority: tuple[AuthorityKind, ...]
    joint: bool
    question: str
    blocking_external_input: bool
    downstream: tuple[str, ...]
    authority: tuple[AuthorityReference, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()
    resolution: Mapping[str, object] | None = None
    rejected: str | None = None

    @property
    def is_approved(self) -> bool:
        return self.status is DecisionStatus.APPROVED


@dataclass(frozen=True, slots=True)
class AuthorityDecisionRecord:
    """A complete, validated decision record (construct via :func:`parse_decision_record`)."""

    schema_version: int
    record_version: int
    record_id: str
    created: dt.date
    scope: str
    source_commit: str
    supersedes: str | None
    decisions: tuple[DecisionEntry, ...]
    external_inputs: tuple[DecisionId, ...]

    def decision(self, decision_id: DecisionId) -> DecisionEntry:
        return next(d for d in self.decisions if d.id is decision_id)

    def counts(self) -> Mapping[DecisionStatus, int]:
        return MappingProxyType({s: sum(d.status is s for d in self.decisions) for s in DecisionStatus})

    def approved_resolution(self, decision_id: DecisionId) -> Mapping[str, object] | None:
        """The resolution for production use - only for an APPROVED decision; ``None`` otherwise."""
        entry = self.decision(DecisionId(decision_id))
        return entry.resolution if entry.status is DecisionStatus.APPROVED else None

    def approved_authority(self, decision_id: DecisionId) -> AuthorityReference | None:
        """The (first) authority reference of an APPROVED decision; ``None`` otherwise."""
        entry = self.decision(DecisionId(decision_id))
        return entry.authority[0] if entry.status is DecisionStatus.APPROVED and entry.authority else None


# ---------------------------------------------------------------- loading


def load_decision_record(path: str | Path) -> AuthorityDecisionRecord:
    """Load and validate a TOML decision record (fail closed)."""
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        raise DecisionRecordError("the record file cannot be read") from None
    except tomllib.TOMLDecodeError:
        raise DecisionRecordError("the record is not valid TOML") from None
    return parse_decision_record(data)


def validate_decision_record(path: str | Path) -> AuthorityDecisionRecord:
    """Public validation entry point (alias of :func:`load_decision_record`)."""
    return load_decision_record(path)


def parse_decision_record(data: Mapping[str, object]) -> AuthorityDecisionRecord:
    """Build a validated record from parsed TOML data (the input is never modified)."""
    if not isinstance(data, Mapping):
        raise DecisionRecordError("the record must be a table")
    _only_keys(data, {"schema_version", "record_version", "record_id", "created", "scope", "source_commit",
                      "supersedes", "summary", "external_inputs", "decisions"}, "record")
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise DecisionRecordError("unsupported schema_version")
    record_version = data.get("record_version")
    if type(record_version) is not int or record_version < 1:
        raise DecisionRecordError("record_version must be a positive integer")
    record_id = data.get("record_id")
    if not isinstance(record_id, str) or record_id != f"pricing-authorities-v{record_version}":
        raise DecisionRecordError("record_id must be 'pricing-authorities-v<record_version>'")
    created = data.get("created")
    if type(created) is not dt.date:
        raise DecisionRecordError("created must be a TOML date")
    scope = _text(data.get("scope"), "scope")
    source_commit = data.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", source_commit):
        raise DecisionRecordError("source_commit must be a hexadecimal commit hash")
    supersedes = data.get("supersedes")
    if record_version == 1:
        if supersedes is not None:
            raise DecisionRecordError("the first revision supersedes nothing")
    elif supersedes != f"pricing-authorities-v{record_version - 1}":
        raise DecisionRecordError("supersedes must name the previous revision")

    raw = data.get("decisions")
    if not isinstance(raw, list) or not all(isinstance(d, Mapping) for d in raw):
        raise DecisionRecordError("decisions must be an array of tables")
    seen: list[DecisionId] = []
    entries = []
    for item in raw:
        entry = _entry(item)
        if entry.id in seen:
            raise DecisionRecordError(f"duplicate decision {entry.id.value}")
        seen.append(entry.id)
        entries.append(entry)
    missing = [d.value for d in DecisionId if d not in seen]
    if missing:
        raise DecisionRecordError("missing required decisions: " + ", ".join(missing))
    order = list(DecisionId)
    decisions = tuple(sorted(entries, key=lambda e: order.index(e.id)))
    by_id = {d.id: d for d in decisions}
    for entry in decisions:
        if entry.status is DecisionStatus.APPROVED:
            _RESOLVERS[entry.id](entry.resolution, by_id, entry.id)   # type: ignore[arg-type]

    blocking = tuple(d.id for d in decisions if d.blocking_external_input)
    inputs = data.get("external_inputs")
    if not isinstance(inputs, list) or not all(isinstance(i, str) for i in inputs):
        raise DecisionRecordError("external_inputs must be an array of decision ids")
    if len(set(inputs)) != len(inputs) or set(inputs) != {d.value for d in blocking}:
        raise DecisionRecordError("external_inputs must list exactly the decisions blocked on external input")

    summary = data.get("summary")
    record = AuthorityDecisionRecord(
        schema_version=schema_version, record_version=record_version, record_id=record_id, created=created,
        scope=scope, source_commit=source_commit, supersedes=supersedes, decisions=decisions,
        external_inputs=blocking)
    expected = {s.value.lower(): n for s, n in record.counts().items()}
    if not isinstance(summary, Mapping) or dict(summary) != expected:
        raise DecisionRecordError("summary counts must equal the decision statuses")
    return record


def _entry(item: Mapping[str, object]) -> DecisionEntry:
    _only_keys(item, {"id", "status", "responsible_authority", "joint", "question", "blocking_external_input",
                      "downstream", "authority", "evidence", "resolution", "rejected"}, "decision")
    try:
        decision_id = DecisionId(item.get("id"))
    except ValueError:
        raise DecisionRecordError("unknown decision id") from None
    name = decision_id.value
    try:
        status = DecisionStatus(item.get("status"))
    except ValueError:
        raise DecisionRecordError(f"{name}: status must be PROPOSED, APPROVED or REJECTED") from None
    roles_raw = item.get("responsible_authority")
    if not isinstance(roles_raw, list) or not roles_raw:
        raise DecisionRecordError(f"{name}: responsible_authority must name at least one role")
    try:
        roles = tuple(AuthorityKind(r) for r in roles_raw)
    except ValueError:
        raise DecisionRecordError(f"{name}: responsible_authority must be authority roles") from None
    required_roles, joint_required = REQUIRED_DECISIONS[decision_id]
    if len(set(roles)) != len(roles) or set(roles) != set(required_roles):
        raise DecisionRecordError(f"{name}: responsible_authority differs from the required roles")
    joint = item.get("joint")
    if joint is not joint_required:
        raise DecisionRecordError(f"{name}: joint must be {str(joint_required).lower()}")
    question = _text(item.get("question"), f"{name}: question")
    blocking = item.get("blocking_external_input")
    if not isinstance(blocking, bool):
        raise DecisionRecordError(f"{name}: blocking_external_input must be a boolean")
    downstream = item.get("downstream")
    if not isinstance(downstream, list) or not downstream or not all(
            isinstance(c, str) and _CODE.match(c) for c in downstream):
        raise DecisionRecordError(f"{name}: downstream must list blocker or gap codes")
    authority = tuple(_authority(a, name) for a in _array(item.get("authority", []), f"{name}: authority"))
    evidence = tuple(_evidence(e, name) for e in _array(item.get("evidence", []), f"{name}: evidence"))
    resolution = item.get("resolution")
    rejected = item.get("rejected")

    if status is DecisionStatus.PROPOSED:
        if blocking is not True:
            raise DecisionRecordError(f"{name}: a PROPOSED decision must block on external input")
        if resolution is not None or rejected is not None or authority:
            raise DecisionRecordError(f"{name}: a PROPOSED decision carries no resolution or authority")
    else:
        kinds = {a.kind for a in authority}
        if not authority:
            raise DecisionRecordError(f"{name}: {status.value} requires authority provenance")
        if not kinds <= set(required_roles):
            raise DecisionRecordError(f"{name}: authority kind is not a responsible role")
        if joint_required and kinds != set(required_roles):
            raise DecisionRecordError(f"{name}: a joint decision requires every responsible role")
        if blocking is not False:
            raise DecisionRecordError(f"{name}: a terminal decision does not block on external input")
        if status is DecisionStatus.APPROVED:
            if not isinstance(resolution, Mapping) or rejected is not None:
                raise DecisionRecordError(f"{name}: APPROVED requires a resolution table")
        else:
            if resolution is not None:
                raise DecisionRecordError(f"{name}: REJECTED carries no resolution")
            _text(rejected, f"{name}: rejected")
    frozen = MappingProxyType(_freeze(resolution)) if isinstance(resolution, Mapping) else None
    return DecisionEntry(id=decision_id, status=status, responsible_authority=roles, joint=joint,
                         question=question, blocking_external_input=blocking, downstream=tuple(downstream),
                         authority=authority, evidence=evidence, resolution=frozen,
                         rejected=rejected if isinstance(rejected, str) else None)


def _authority(raw: object, name: str) -> AuthorityReference:
    if not isinstance(raw, Mapping):
        raise DecisionRecordError(f"{name}: authority entries must be tables")
    _only_keys(raw, {"kind", "source", "reference", "note", "effective_date"}, f"{name}: authority")
    if raw.get("kind") in {e.value for e in EvidenceKind}:
        raise DecisionRecordError(f"{name}: non-authoritative evidence cannot serve as authority")
    try:
        kind = AuthorityKind(raw.get("kind"))
    except ValueError:
        raise DecisionRecordError(f"{name}: authority kind must be SUPPLIER, COLLECTION_OWNER or BUSINESS_OWNER") from None
    effective = raw.get("effective_date")
    if effective is not None and type(effective) is not dt.date:
        raise DecisionRecordError(f"{name}: authority effective_date must be a TOML date")
    try:
        return AuthorityReference(kind=kind, source=raw.get("source"), reference=raw.get("reference"),  # type: ignore[arg-type]
                                  note=raw.get("note"), effective_date=effective)          # type: ignore[arg-type]
    except DecisionRecordError as exc:
        raise DecisionRecordError(f"{name}: {exc}") from None


def _evidence(raw: object, name: str) -> EvidenceReference:
    if not isinstance(raw, Mapping):
        raise DecisionRecordError(f"{name}: evidence entries must be tables")
    _only_keys(raw, {"kind", "summary", "reference", "observed_candidates"}, f"{name}: evidence")
    if raw.get("kind") in {a.value for a in AuthorityKind}:
        raise DecisionRecordError(f"{name}: evidence must use a non-authoritative evidence kind")
    try:
        kind = EvidenceKind(raw.get("kind"))
    except ValueError:
        raise DecisionRecordError(f"{name}: unknown evidence kind") from None
    candidates = raw.get("observed_candidates", [])
    if not isinstance(candidates, list):
        raise DecisionRecordError(f"{name}: observed_candidates must be an array")
    try:
        return EvidenceReference(kind=kind, summary=raw.get("summary"), reference=raw.get("reference"),  # type: ignore[arg-type]
                                 observed_candidates=tuple(_stream_key(k, f"{name}: observed_candidates")
                                                           for k in candidates))
    except DecisionRecordError as exc:
        raise DecisionRecordError(f"{name}: {exc}") from None


# ------------------------------------------------------- resolution validators


def _keys(res: Mapping, required: set[str], name: str, optional: set[str] = frozenset()) -> None:  # type: ignore[assignment]
    present = set(res)
    if not required <= present or not present <= required | optional:
        raise DecisionRecordError(f"{name}: resolution must contain exactly the required fields")


def _flag(res: Mapping, key: str, name: str) -> bool:
    if not isinstance(res.get(key), bool):
        raise DecisionRecordError(f"{name}: resolution field {key} must be a boolean")
    return res[key]


def _choice(res: Mapping, key: str, allowed: set[str], name: str) -> str:
    if res.get(key) not in allowed:
        raise DecisionRecordError(f"{name}: resolution field {key} has an unsupported value")
    return res[key]


def _streams(res: Mapping, key: str, name: str) -> tuple[tuple[str, ...], ...]:
    raw = res.get(key)
    if not isinstance(raw, tuple) or not raw:
        raise DecisionRecordError(f"{name}: resolution field {key} must be a non-empty array of stream keys")
    keys = tuple(_stream_key(k, f"{name}: {key}") for k in raw)
    if len(set(keys)) != len(keys):
        raise DecisionRecordError(f"{name}: resolution field {key} has duplicate stream keys")
    return keys


def _approved(by_id: Mapping, decision: DecisionId, name: str) -> DecisionEntry:
    entry = by_id[decision]
    if entry.status is not DecisionStatus.APPROVED:
        raise DecisionRecordError(f"{name}: requires {decision.value} to be APPROVED")
    return entry


def _universe(by_id: Mapping, name: str) -> set[tuple[str, ...]]:
    return set(_streams(_approved(by_id, _D.EXPECTED_STREAM_UNIVERSE, name).resolution, "streams", name))


def _job_id_bool(key: str):  # type: ignore[no-untyped-def]
    def check(res, by_id, d):  # type: ignore[no-untyped-def]
        _keys(res, {key}, d.value)
        _flag(res, key, d.value)
    return check


def _job_id_invalid(res, by_id, d):  # type: ignore[no-untyped-def]
    required = {"non_zero_fractional_suffix_invalid", "scientific_notation_invalid", "sign_invalid",
                "surrounding_whitespace_invalid", "padding_invalid", "other_numeric_forms_invalid"}
    _keys(res, required, d.value)
    for key in required:
        _flag(res, key, d.value)


def _job_id_preservation(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"preserve_raw_identifier", "separate_linkage_key"}, d.value)
    _flag(res, "preserve_raw_identifier", d.value)
    _flag(res, "separate_linkage_key", d.value)


def _stream_universe(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"mode", "streams"}, d.value)
    _choice(res, "mode", {"EXHAUSTIVE", "MINIMUM_REQUIRED"}, d.value)
    _streams(res, "streams", d.value)


def _source_spelling(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"streams"}, d.value)
    if set(_streams(res, "streams", d.value)) != _universe(by_id, d.value):
        raise DecisionRecordError(f"{d.value}: spellings must cover the approved stream universe exactly")


def _role_assignments(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"assignments"}, d.value)
    raw = res.get("assignments")
    if not isinstance(raw, tuple) or not raw:
        raise DecisionRecordError(f"{d.value}: assignments must be a non-empty array")
    keys = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise DecisionRecordError(f"{d.value}: each assignment must be a table")
        _keys(item, {"stream", "role"}, d.value)
        keys.append(_stream_key(item.get("stream"), f"{d.value}: assignment stream"))
        _choice(item, "role", {r.value for r in LocationRoleDecision}, d.value)
    if len(set(keys)) != len(keys) or set(keys) != _universe(by_id, d.value):
        raise DecisionRecordError(f"{d.value}: assignments must cover the approved stream universe exactly once")


def _comparison_pairs(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"pairs"}, d.value)
    raw = res.get("pairs")
    if not isinstance(raw, tuple) or not raw:
        raise DecisionRecordError(f"{d.value}: pairs must be a non-empty array")
    universe, seen = _universe(by_id, d.value), set()
    roles = {}
    role_entry = by_id[_D.LOCATION_ROLE_ASSIGNMENTS]
    if role_entry.status is not DecisionStatus.APPROVED:
        raise DecisionRecordError(f"{d.value}: requires {_D.LOCATION_ROLE_ASSIGNMENTS.value} to be APPROVED")
    for item in role_entry.resolution["assignments"]:
        roles[tuple(item["stream"])] = item["role"]
    for item in raw:
        if not isinstance(item, Mapping):
            raise DecisionRecordError(f"{d.value}: each pair must be a table")
        _keys(item, {"airport", "downtown"}, d.value)
        airport = _stream_key(item.get("airport"), f"{d.value}: airport")
        downtown = _stream_key(item.get("downtown"), f"{d.value}: downtown")
        if airport[0] != downtown[0]:
            raise DecisionRecordError(f"{d.value}: a comparison pair must stay within one city")
        if airport not in universe or downtown not in universe:
            raise DecisionRecordError(f"{d.value}: pairs must reference approved expected locations")
        if roles.get(airport) != "AIRPORT" or roles.get(downtown) != "DOWNTOWN":
            raise DecisionRecordError(f"{d.value}: pair members must carry the approved airport/downtown roles")
        if (airport, downtown) in seen:
            raise DecisionRecordError(f"{d.value}: duplicate comparison pair")
        seen.add((airport, downtown))


def _vancouver(res, by_id, d):  # type: ignore[no-untyped-def]
    state = _choice(res, "state", {"CONFIRMED_ALIAS", "CONFIRMED_DISTINCT"}, d.value)
    governed = {tuple(k) for k in COMPARED_LOCATION_STREAMS}
    if state == "CONFIRMED_ALIAS":
        _keys(res, {"state", "canonical_location"}, d.value)
        if _stream_key(res.get("canonical_location"), f"{d.value}: canonical_location") not in governed:
            raise DecisionRecordError(f"{d.value}: the canonical key must be one of the two governed keys")
    else:
        _keys(res, {"state"}, d.value)
        roles = by_id[_D.LOCATION_ROLE_ASSIGNMENTS]
        if roles.status is not DecisionStatus.APPROVED or not governed <= {
                tuple(a["stream"]) for a in roles.resolution["assignments"]}:
            raise DecisionRecordError(f"{d.value}: distinct locations need approved roles for both keys")


def _field_ref(res: Mapping, key: str, allowed: tuple[str, ...] | None, name: str) -> str:
    value = res.get(key)
    if not isinstance(value, str) or "." not in value:
        raise DecisionRecordError(f"{name}: resolution field {key} must be a dataset.column reference")
    if allowed is not None:
        if value not in allowed:
            raise DecisionRecordError(f"{name}: resolution field {key} is not a permitted field")
        return value
    dataset, column = value.split(".", 1)
    try:
        if column not in DATASET_DEFINITIONS[DatasetKey(dataset)].columns:
            raise ValueError
    except ValueError:
        raise DecisionRecordError(f"{name}: resolution field {key} is not a contract column") from None
    return value


def _capture_timestamp(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"field"}, d.value)
    _field_ref(res, "field", _TIMESTAMP_FIELDS, d.value)


def _expected_periods(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"period", "period_starts"}, d.value)
    if not isinstance(res.get("period"), str) or not re.fullmatch(r"PT?\d+[DHM]|P\d+D|PT\d+[HM]", res["period"]):
        raise DecisionRecordError(f"{d.value}: period must be an ISO-8601 duration such as PT1H")
    starts = res.get("period_starts")
    if not isinstance(starts, tuple) or not starts:
        raise DecisionRecordError(f"{d.value}: period_starts must be a non-empty array")
    instants = []
    for start in starts:
        if not isinstance(start, str) or not _EXPLICIT_OFFSET.search(start):
            raise DecisionRecordError(f"{d.value}: every period start needs an explicit UTC offset")
        try:
            instants.append(dt.datetime.fromisoformat(start.replace("Z", "+00:00")))
        except ValueError:
            raise DecisionRecordError(f"{d.value}: a period start is not ISO-8601") from None
    if len(set(instants)) != len(instants):
        raise DecisionRecordError(f"{d.value}: duplicate period starts")


def _sharing(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"mode"}, d.value)
    _choice(res, "mode", {"SHARED", "PER_STREAM"}, d.value)


def _exceptions(res, by_id, d):  # type: ignore[no-untyped-def]
    model = _choice(res, "model", {"NO_EXCEPTIONS", "LISTED_EXCEPTIONS"}, d.value)
    if model == "NO_EXCEPTIONS":
        _keys(res, {"model"}, d.value)
        return
    _keys(res, {"model", "exceptions"}, d.value)
    raw = res.get("exceptions")
    if not isinstance(raw, tuple) or not raw:
        raise DecisionRecordError(f"{d.value}: listed exceptions must be a non-empty array")
    for item in raw:
        if not isinstance(item, Mapping):
            raise DecisionRecordError(f"{d.value}: each exception must be a table")
        _keys(item, {"period_start", "reason", "authority_reference"}, d.value)
        if not isinstance(item.get("period_start"), str) or not _EXPLICIT_OFFSET.search(item["period_start"]):
            raise DecisionRecordError(f"{d.value}: exception period_start needs an explicit UTC offset")
        _choice(item, "reason", {"CANCELLATION", "OUTAGE", "HOLIDAY", "OTHER_AUTHORIZED"}, d.value)
        _text(item.get("authority_reference"), f"{d.value}: exception authority_reference")


def _timezone(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"timezone"}, d.value)
    value = res.get("timezone")
    try:
        if not isinstance(value, str) or not value:
            raise ValueError
        ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError):
        raise DecisionRecordError(f"{d.value}: timezone must be a valid IANA zone name") from None


def _ordering(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"earlier", "later", "equal_allowed"}, d.value)
    earlier = _field_ref(res, "earlier", _TIMESTAMP_FIELDS, d.value)
    later = _field_ref(res, "later", _TIMESTAMP_FIELDS, d.value)
    if earlier == later:
        raise DecisionRecordError(f"{d.value}: ordering compares two different fields")
    _flag(res, "equal_allowed", d.value)


def _tolerance(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"tolerance", "unit"}, d.value)
    if type(res.get("tolerance")) is not int or res["tolerance"] < 0:
        raise DecisionRecordError(f"{d.value}: tolerance must be a non-negative integer")
    _choice(res, "unit", {"SECONDS", "MINUTES", "HOURS"}, d.value)
    _approved(by_id, _D.SCRAPED_FINISHED_ORDERING, d.value)


def _reporting_source(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"field"}, d.value)
    _field_ref(res, "field", None, d.value)


def _date_semantics(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"meaning", "derivation"}, d.value)
    _text(res.get("meaning"), f"{d.value}: meaning")
    if _choice(res, "derivation", {"REPORTING_DAY", "SOURCE_SUPPLIED_UNDERIVED"}, d.value) == "REPORTING_DAY":
        _approved(by_id, _D.REPORTING_DAY_SOURCE, d.value)
        _approved(by_id, _D.REPORTING_DAY_TIMEZONE, d.value)


def _rental_validity(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"date_format", "pickup_before_return_required", "equal_dates_allowed",
                "minimum_duration_days", "maximum_duration_days"}, d.value)
    _text(res.get("date_format"), f"{d.value}: date_format")
    _flag(res, "pickup_before_return_required", d.value)
    _flag(res, "equal_dates_allowed", d.value)
    low, high = res.get("minimum_duration_days"), res.get("maximum_duration_days")
    if type(low) is not int or low < 0 or type(high) is not int or high < low:
        raise DecisionRecordError(f"{d.value}: duration limits must be integers with 0 <= minimum <= maximum")


def _rental_agreements(res, by_id, d):  # type: ignore[no-untyped-def]
    _keys(res, {"agreements"}, d.value)
    raw = res.get("agreements")
    if not isinstance(raw, tuple) or not raw:
        raise DecisionRecordError(f"{d.value}: agreements must be a non-empty array")
    pairs = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise DecisionRecordError(f"{d.value}: each agreement must be a table")
        _keys(item, {"source", "target"}, d.value)
        pairs.append((_field_ref(item, "source", RENTAL_DATE_PARENT_FIELDS, d.value),
                      _field_ref(item, "target", RENTAL_DATE_DETAIL_FIELDS, d.value)))
    targets = [t for _, t in pairs]
    if len(set(pairs)) != len(pairs) or sorted(targets) != sorted(RENTAL_DATE_DETAIL_FIELDS):
        raise DecisionRecordError(f"{d.value}: every detail rental-date field needs exactly one approved source")


_RESOLVERS = {
    _D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE: _job_id_bool("decimal_zero_suffix_equivalent"),
    _D.JOB_ID_LEADING_ZERO_SIGNIFICANCE: _job_id_bool("leading_zeros_significant"),
    _D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS: _job_id_invalid,
    _D.JOB_ID_RAW_AND_LINKAGE_PRESERVATION: _job_id_preservation,
    _D.EXPECTED_STREAM_UNIVERSE: _stream_universe,
    _D.EXPECTED_STREAM_SOURCE_SPELLING: _source_spelling,
    _D.LOCATION_ROLE_ASSIGNMENTS: _role_assignments,
    _D.VALID_LOCATION_COMPARISON_PAIRS: _comparison_pairs,
    _D.VANCOUVER_LOCATION_IDENTITY: _vancouver,
    _D.SCHEDULE_CAPTURE_TIMESTAMP: _capture_timestamp,
    _D.SCHEDULE_EXPECTED_PERIODS: _expected_periods,
    _D.SCHEDULE_SHARING_MODEL: _sharing,
    _D.SCHEDULE_EXCEPTIONS: _exceptions,
    _D.FINISHED_AT_TIMEZONE: _timezone,
    _D.SCRAPED_FINISHED_ORDERING: _ordering,
    _D.SCRAPED_FINISHED_TOLERANCE: _tolerance,
    _D.REPORTING_DAY_SOURCE: _reporting_source,
    _D.REPORTING_DAY_TIMEZONE: _timezone,
    _D.SCRAPE_DATE_SEMANTICS: _date_semantics,
    _D.DATE_CLEAN_SEMANTICS: _date_semantics,
    _D.RENTAL_DATE_VALIDITY: _rental_validity,
    _D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS: _rental_agreements,
}
assert set(_RESOLVERS) == set(DecisionId) == set(REQUIRED_DECISIONS)


# ---------------------------------------------------------------- helpers


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _TEXT_MAX:
        raise DecisionRecordError(f"{name} must be non-blank text of at most {_TEXT_MAX} characters")
    if any(p.search(value) for p in _SENSITIVE):
        raise DecisionRecordError(f"{name} looks like source-level data (timestamp, price or identifier)")
    return value


def _stream_key(value: object, name: str) -> tuple[str, str]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, str) and v and v == v.strip() for v in value)):
        raise DecisionRecordError(f"{name} must be [city, location] with exact non-blank components")
    return (value[0], value[1])


def _array(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise DecisionRecordError(f"{name} must be an array")
    return value


def _only_keys(table: Mapping, allowed: set[str], name: str) -> None:
    if not set(table) <= allowed:
        raise DecisionRecordError(f"{name} has unsupported fields")


def _freeze(value: object) -> object:
    """Immutable deep copy of parsed TOML (tables -> read-only mappings, arrays -> tuples)."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


# --------------------------------------------------------------- rendering


def render_status_summary(record: AuthorityDecisionRecord) -> str:
    """Sanitized summary: id, version, counts, decision ids, roles and blocking categories only."""
    counts = record.counts()
    lines = [f"record: {record.record_id} (version {record.record_version}, schema {record.schema_version})",
             "status counts: " + ", ".join(f"{s.value}={counts[s]}" for s in DecisionStatus)]
    for entry in record.decisions:
        roles = "+".join(r.value for r in entry.responsible_authority) if entry.joint else \
            "|".join(r.value for r in entry.responsible_authority)
        blocking = " BLOCKED_ON_EXTERNAL_INPUT" if entry.blocking_external_input else ""
        lines.append(f"{entry.id.value}: {entry.status.value} [{roles}]{blocking}")
    groups = _groups(record)
    lines.append("blocking external input: " + "; ".join(
        f"{label}={len(ids)}" for label, ids in groups if ids))
    return "\n".join(lines) + "\n"


def _groups(record: AuthorityDecisionRecord):  # type: ignore[no-untyped-def]
    blocked = [d for d in record.decisions if d.blocking_external_input]
    return (("collection_owner_or_supplier", [d for d in blocked if set(d.responsible_authority) == set(_CO_SUP)]),
            ("business_owner", [d for d in blocked if d.responsible_authority == (_A.BUSINESS_OWNER,)]),
            ("joint", [d for d in blocked if d.joint]))


_RESPONSE_SHAPE = {
    _D.JOB_ID_DECIMAL_ZERO_EQUIVALENCE: "`decimal_zero_suffix_equivalent` = true or false",
    _D.JOB_ID_LEADING_ZERO_SIGNIFICANCE: "`leading_zeros_significant` = true or false",
    _D.JOB_ID_INVALID_NUMERIC_REPRESENTATIONS: "true or false for each of: non-zero fractional suffix, scientific "
    "notation, sign, surrounding whitespace, padding, other numeric forms",
    _D.JOB_ID_RAW_AND_LINKAGE_PRESERVATION: "`preserve_raw_identifier` and `separate_linkage_key` = true or false",
    _D.EXPECTED_STREAM_UNIVERSE: "`mode` = EXHAUSTIVE or MINIMUM_REQUIRED and the list of [city, location] keys",
    _D.EXPECTED_STREAM_SOURCE_SPELLING: "the exact source spelling of every [city, location] key in the universe",
    _D.LOCATION_ROLE_ASSIGNMENTS: "one role (AIRPORT or DOWNTOWN or OTHER) for every approved stream key",
    _D.VALID_LOCATION_COMPARISON_PAIRS: "list of within-city {airport, downtown} pairs of approved keys",
    _D.VANCOUVER_LOCATION_IDENTITY: "CONFIRMED_ALIAS with one of the two keys as canonical, or CONFIRMED_DISTINCT",
    _D.SCHEDULE_CAPTURE_TIMESTAMP: "one of jobs.finished_at, cars.job_finished_at, cars.scraped_at",
    _D.SCHEDULE_EXPECTED_PERIODS: "period duration (e.g. PT1H) and every period start with an explicit UTC offset",
    _D.SCHEDULE_SHARING_MODEL: "SHARED or PER_STREAM",
    _D.SCHEDULE_EXCEPTIONS: "NO_EXCEPTIONS, or listed exceptions (period start with offset, reason, reference)",
    _D.FINISHED_AT_TIMEZONE: "an IANA timezone name",
    _D.SCRAPED_FINISHED_ORDERING: "earlier field, later field, whether equality is allowed",
    _D.SCRAPED_FINISHED_TOLERANCE: "non-negative integer and unit (SECONDS or MINUTES or HOURS)",
    _D.REPORTING_DAY_SOURCE: "a dataset.column reference",
    _D.REPORTING_DAY_TIMEZONE: "an IANA timezone name",
    _D.SCRAPE_DATE_SEMANTICS: "meaning and derivation (REPORTING_DAY or SOURCE_SUPPLIED_UNDERIVED)",
    _D.DATE_CLEAN_SEMANTICS: "meaning and derivation (REPORTING_DAY or SOURCE_SUPPLIED_UNDERIVED)",
    _D.RENTAL_DATE_VALIDITY: "date format, pickup-before-return required, equality allowed, min/max duration days",
    _D.RENTAL_DATE_PARENT_DETAIL_AGREEMENTS: "for each of cars.job_pickup_date, cars.job_return_date, "
    "cars.pickup_date, cars.return_date: the jobs field it must equal",
}


def render_authority_request_checklist(record: AuthorityDecisionRecord) -> str:
    """Neutral request checklist for every decision blocked on external input (deterministic)."""
    lines = [f"# Authority request checklist ({record.record_id})", "",
             "Generated by `render_authority_request_checklist` from the decision record; regenerate it",
             "rather than editing by hand. Questions are neutral: observed patterns are not suggested answers.",
             "Every answer needs an attributable source (supplier, collection owner or business owner) and a",
             "durable reference (document, ticket or written decision) recorded in a new record revision.", ""]
    titles = {"collection_owner_or_supplier": "Collection owner or supplier", "business_owner": "Business owner",
              "joint": "Joint decision (collection owner and business owner)"}
    for label, entries in _groups(record):
        if not entries:
            continue
        lines += [f"## {titles[label]}", "",
                  "| Decision | Responsible authority | Question | Response shape | Source / reference required |"
                  " Blocker or gap affected |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for entry in entries:
            roles = (" and " if entry.joint else " or ").join(r.value for r in entry.responsible_authority)
            lines.append(f"| `{entry.id.value}` | {roles} | {entry.question} | {_RESPONSE_SHAPE[entry.id]} | "
                         f"authority kind, responsible source, durable reference | "
                         f"{', '.join(f'`{c}`' for c in entry.downstream)} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m ql2_sixt_canada_analysis.authority_decisions <record.toml>", file=sys.stderr)
        return 2
    try:
        record = validate_decision_record(args[0])
    except DecisionRecordError as exc:
        print(f"INVALID: {exc}")
        return 1
    print("VALID")
    print(render_status_summary(record), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
