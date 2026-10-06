"""The single resolution of the expected source-stream contract from the authority record.

The expected source-stream universe is an external decision
(``EXPECTED_STREAM_UNIVERSE``, collection owner and business owner jointly)
plus its exact source spelling (``EXPECTED_STREAM_SOURCE_SPELLING``). This
module turns the latest valid authority record into one typed
:class:`ExpectedStreamContract`; :data:`~ql2_sixt_canada_analysis.schemas.EXPECTED_LOCATION_COVERAGE`
is its ``coverage``, so ingestion checks, coverage, stream continuity,
completeness, pricing readiness, the baseline and the notebook all use the same
contract. There is no second hard-coded stream list.

Fail closed:

* no valid record (missing, unreadable, invalid - including a broken
  governance reference) -> ``record_unavailable``;
* either decision not ``APPROVED`` (an approved universe without approved
  spellings is unusable) -> ``not_approved``;

both leave the coverage **unconfigured** (every assessment that needs it
fails closed) and block pricing with ``expected_stream_authority_unavailable``.
An approved universe that is not ``EXHAUSTIVE`` keeps its approved keys but
blocks pricing with ``expected_stream_universe_not_exhaustive``.

Keys are the approved source spellings, compared exactly (case, spacing and
punctuation significant; nothing is lowercased, trimmed or collapsed). Display
labels and analytical aliases are separate concepts and never establish
source coverage: aliases exist only through an approved location-identity
policy. Observed streams never extend the universe - a contract change needs a
new approved record version.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from pathlib import Path

from ql2_sixt_canada_analysis.authority_decisions import (
    AuthorityDecisionRecord,
    DecisionId,
    DecisionStatus,
    load_current_decision_record,
)
from ql2_sixt_canada_analysis.schemas import (
    SOURCE_STREAM_COVERAGE_TEMPLATE,
    LocationCoverageDefinition,
    LocationCoverageMode,
)

__all__ = [
    "EXPECTED_STREAM_DECISIONS",
    "ExpectedStreamAuthorityStatus",
    "ExpectedStreamContract",
    "ExpectedStreamContractBlocker",
    "current_expected_stream_contract",
    "expected_stream_contract_from_record",
    "resolve_expected_stream_contract",
]

#: The two decisions that together define the source-stream contract.
EXPECTED_STREAM_DECISIONS: tuple[DecisionId, ...] = (
    DecisionId.EXPECTED_STREAM_UNIVERSE, DecisionId.EXPECTED_STREAM_SOURCE_SPELLING)


class ExpectedStreamAuthorityStatus(StrEnum):
    APPROVED = "approved"                       # universe and spelling both APPROVED
    NOT_APPROVED = "not_approved"               # at least one is PROPOSED or REJECTED
    RECORD_UNAVAILABLE = "record_unavailable"   # no valid authority record


class ExpectedStreamContractBlocker(StrEnum):
    """Why the source-stream contract cannot support pricing (values equal ``PricingBlocker`` values)."""

    EXPECTED_STREAM_AUTHORITY_UNAVAILABLE = "expected_stream_authority_unavailable"
    EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE = "expected_stream_universe_not_exhaustive"


@dataclass(frozen=True, slots=True)
class ExpectedStreamContract:
    """The effective source-stream contract and the authority it rests on (no source values).

    ``coverage`` is configured (approved keys, approved mode) only when the
    status is ``APPROVED``; otherwise it is the unconfigured template.
    ``universe_status`` / ``spelling_status`` are the record's decision
    statuses (``None`` without a record); ``references`` are the governance
    documents of the approving authorities.
    """

    status: ExpectedStreamAuthorityStatus
    coverage: LocationCoverageDefinition
    record_id: str | None = None
    universe_status: DecisionStatus | None = None
    spelling_status: DecisionStatus | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExpectedStreamAuthorityStatus):
            raise TypeError("status must be an ExpectedStreamAuthorityStatus")
        if not isinstance(self.coverage, LocationCoverageDefinition):
            raise TypeError("coverage must be a LocationCoverageDefinition")
        approved = self.status is ExpectedStreamAuthorityStatus.APPROVED
        if approved != self.coverage.is_configured:
            raise ValueError("only an approved contract has configured expected streams")
        if approved and (not self.record_id or not self.references
                         or self.universe_status is not DecisionStatus.APPROVED
                         or self.spelling_status is not DecisionStatus.APPROVED):
            raise ValueError("an approved contract needs its record, both approvals and their references")
        if (self.status is ExpectedStreamAuthorityStatus.RECORD_UNAVAILABLE) != (self.record_id is None):
            raise ValueError("record_unavailable exactly when no record is identified")

    @property
    def mode(self) -> LocationCoverageMode | None:
        return self.coverage.mode

    @property
    def expected_keys(self) -> tuple[tuple[str, ...], ...]:
        """The approved source keys in record order (empty unless approved)."""
        return tuple(self.coverage.expected_locations or ())

    @property
    def expected_stream_count(self) -> int:
        return len(self.expected_keys)

    @property
    def exhaustive(self) -> bool:
        return self.mode is LocationCoverageMode.EXHAUSTIVE

    @property
    def blocking_reasons(self) -> tuple[ExpectedStreamContractBlocker, ...]:
        B = ExpectedStreamContractBlocker
        if self.status is not ExpectedStreamAuthorityStatus.APPROVED:
            return (B.EXPECTED_STREAM_AUTHORITY_UNAVAILABLE, B.EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE)
        return () if self.exhaustive else (B.EXPECTED_STREAM_UNIVERSE_NOT_EXHAUSTIVE,)

    @property
    def usable(self) -> bool:
        """Approved, exhaustive and therefore able to support pricing readiness."""
        return not self.blocking_reasons


def expected_stream_contract_from_record(
    record: AuthorityDecisionRecord | None,
    template: LocationCoverageDefinition = SOURCE_STREAM_COVERAGE_TEMPLATE,
) -> ExpectedStreamContract:
    """Build the contract from APPROVED decisions only (``record`` is not modified).

    ``template`` supplies structure only (dataset, key columns, scope) and must
    be unconfigured: expected keys come from the record, never from code or data.
    """
    if not isinstance(template, LocationCoverageDefinition) or template.is_configured:
        raise ValueError("template must be an unconfigured LocationCoverageDefinition")
    if record is None:
        return ExpectedStreamContract(status=ExpectedStreamAuthorityStatus.RECORD_UNAVAILABLE, coverage=template)
    if not isinstance(record, AuthorityDecisionRecord):
        raise TypeError("record must be a validated AuthorityDecisionRecord or None")
    universe = record.decision(DecisionId.EXPECTED_STREAM_UNIVERSE)
    spelling = record.decision(DecisionId.EXPECTED_STREAM_SOURCE_SPELLING)
    statuses = dict(record_id=record.record_id, universe_status=universe.status, spelling_status=spelling.status)
    if not (universe.is_approved and spelling.is_approved):
        return ExpectedStreamContract(status=ExpectedStreamAuthorityStatus.NOT_APPROVED, coverage=template,
                                      **statuses)
    keys = tuple(tuple(k) for k in universe.resolution["streams"])
    spelled = {tuple(k) for k in spelling.resolution["streams"]}
    if set(keys) != spelled or len(keys) != len(spelled):   # validated by the record; never trusted blindly
        return ExpectedStreamContract(status=ExpectedStreamAuthorityStatus.NOT_APPROVED, coverage=template,
                                      **statuses)
    if any(len(k) != len(template.location_columns) for k in keys):
        raise ValueError("approved stream keys do not match the template's key columns")
    coverage = replace(template, expected_locations=keys,
                       mode=LocationCoverageMode(str(universe.resolution["mode"]).lower()))
    references = tuple(sorted({a.reference for d in (universe, spelling) for a in d.authority}))
    return ExpectedStreamContract(status=ExpectedStreamAuthorityStatus.APPROVED, coverage=coverage,
                                  references=references, **statuses)


def resolve_expected_stream_contract(*, repository_root: str | Path | None = None) -> ExpectedStreamContract:
    """Load the current record (fail closed) and resolve the contract (no caching)."""
    return expected_stream_contract_from_record(load_current_decision_record(repository_root=repository_root))


@cache
def current_expected_stream_contract() -> ExpectedStreamContract:
    """The project's effective contract, resolved once per process from the committed current record."""
    return resolve_expected_stream_contract()
