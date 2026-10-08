"""Monitoring and actionability (data-plan Section 6): the control catalog and its sample evaluation.

This module is an in-repository monitoring **blueprint** for a one-off sample
extract. It defines the eight required QL2 controls, evaluates each one from
the evidence the validated pipeline already produced, and presents the result
as sanitized, categorical tables. It is not a scheduler, daemon, alerting
service or notification integration: it prints nothing, writes nothing,
plots nothing, reads no environment variable and opens no connection.

Control definitions versus evaluations
--------------------------------------
:data:`MONITORING_CONTROLS` is an immutable catalog of exactly eight
:class:`MonitoringControl` definitions in data-plan order. Each names its
condition, :class:`Severity`, likely business impact, recommended response,
evidence source and :class:`CalibrationStatus`. The catalog does not depend on
any data and stays available when the sample evidence is blocked.

A :class:`ControlEvaluation` is the sample result of one control. Its
:class:`ControlStatus` separates five situations that must never be merged:

* ``triggered`` - a defect was observed in the supplied evidence;
* ``passed`` - every prerequisite was available and the control held;
* ``not_assessable`` - prerequisite evidence was unavailable (never a pass);
* ``candidate_only`` - a proposed rule without an approved, calibrated
  threshold (never a pass and never a production alert);
* ``confirmation_required`` - a right-censored observation at the end of the
  window that needs another eligible collection.

Findings, evidence gaps and notes are typed enum members
(:class:`MonitoringFinding`, :class:`EvidenceGap`, :class:`MonitoringNote`),
so a report can never carry source values, identifiers, prices or paths.

Evidence flow
-------------
:func:`run_monitoring` runs :func:`~ql2_sixt_canada_analysis.pricing_pipeline.run_pricing_pipeline`
exactly once. :func:`monitoring_from_pipeline` then derives the higher-order
price-change analysis and the visible assortment from that same
:class:`~ql2_sixt_canada_analysis.pricing_pipeline.PricingPipelineResult`,
verifies that both are bound to its frames and location authority, and
evaluates every control with :func:`evaluate_monitoring_controls`. Nothing is
recalculated: the structural controls read the retained foundational reports,
and the anomaly-style controls read the validated price-change and assortment
results. A binding mismatch or a missing readiness report blocks the whole
report, and every control is then ``not_assessable``.

Thresholds
----------
The sample spans roughly 90 hours and cannot calibrate production thresholds.
The abrupt-assortment control uses the existing
:class:`~ql2_sixt_canada_analysis.assortment_contract.UnusualDropPolicy`, whose
default is unavailable. The synchronized-movement control uses
:class:`SynchronizedMovementPolicy`, whose default
(:data:`DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY`) is unavailable and carries no
parameters. Only an explicitly supplied approved policy, with a recorded
authority and every parameter, can turn either control into an evaluated rule.
No threshold is estimated from the data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from ql2_sixt_canada_analysis.assortment_contract import (
    DEFAULT_UNUSUAL_DROP_POLICY,
    AnomalyPolicyStatus,
    AssessabilityStatus,
    AssortmentContractError,
    UnusualDropPolicy,
)
from ql2_sixt_canada_analysis.price_change_analysis import (
    MovementClass,
    NotTestableReason,
    PriceChangeAnalysisResult,
)
from ql2_sixt_canada_analysis.price_change_events import (
    EVENT_INTERVAL_COLUMNS,
    EVENT_TIMESTAMP_COLUMN,
    PriceChangeContractError,
)
from ql2_sixt_canada_analysis.visible_assortment import VisibleAssortmentResult

__all__ = [
    "DATA_PLAN_RECONCILIATION",
    "DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY",
    "MONITORING_CONTROLS",
    "MONITORING_TABLE_COLUMNS",
    "PRODUCTION_CALIBRATION_REQUIREMENTS",
    "SEVERITY_DESCRIPTIONS",
    "STATUS_DESCRIPTIONS",
    "CalibrationStatus",
    "ControlEvaluation",
    "ControlStatus",
    "EvidenceGap",
    "MonitoringBlocker",
    "MonitoringControl",
    "MonitoringControlId",
    "MonitoringContractError",
    "MonitoringEvidence",
    "MonitoringFinding",
    "MonitoringNote",
    "MonitoringReport",
    "MonitoringReportStatus",
    "MonitoringResult",
    "Severity",
    "SynchronizedMovementPolicy",
    "blocked_monitoring_report",
    "evaluate_monitoring_controls",
    "monitoring_control",
    "monitoring_control_table",
    "monitoring_evidence_from_pipeline",
    "monitoring_from_pipeline",
    "monitoring_summary_lines",
    "run_monitoring",
    "severity_scale_table",
    "status_legend_table",
    "validate_monitoring_table",
]


class MonitoringContractError(ValueError):
    """A monitoring definition, policy, evaluation or table breaks the monitoring contract."""


# ================================================================== taxonomy


class MonitoringControlId(StrEnum):
    """The eight required QL2 controls, in data-plan order."""

    MISSING_EXPECTED_LOCATIONS = "missing_expected_locations"
    JOB_DETAIL_COUNT_MISMATCHES = "job_detail_count_mismatches"
    DUPLICATE_OR_ALIASED_LOCATION_FEEDS = "duplicate_or_aliased_location_feeds"
    UNEXPECTED_TIMESTAMP_OFFSETS = "unexpected_timestamp_offsets"
    INVALID_OR_CHANGING_PRODUCT_ATTRIBUTES = "invalid_or_changing_product_attributes"
    ABRUPT_ASSORTMENT_CHANGES = "abrupt_assortment_changes"
    LARGE_SYNCHRONIZED_PRICE_MOVEMENTS = "large_synchronized_price_movements"
    UNCONFIRMED_END_OF_WINDOW_ANOMALIES = "unconfirmed_end_of_window_anomalies"


class Severity(StrEnum):
    """Business impact and response urgency (never statistical confidence), most severe first."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"

    @property
    def rank(self) -> int:
        return tuple(Severity).index(self)


#: What each severity means for the business and for the response (fixed text; no data).
SEVERITY_DESCRIPTIONS: Mapping[Severity, tuple[str, str]] = MappingProxyType({
    Severity.CRITICAL: (
        "Can invalidate the collection as a whole, because every downstream comparison may be biased.",
        "Stop downstream publication and investigate immediately."),
    Severity.HIGH: (
        "Can bias comparisons or contaminate many prices or products at once.",
        "Withhold the affected analysis and investigate before its next use."),
    Severity.MEDIUM: (
        "A provisional or review-level observation whose meaning is not yet established.",
        "Keep it out of customer-facing conclusions until it is reviewed, corroborated or confirmed."),
})


class CalibrationStatus(StrEnum):
    """Whether a control can be evaluated from approved contracts now, or awaits an approved threshold."""

    CONTRACT_BASED = "contract_based"
    CANDIDATE_POLICY_UNAPPROVED = "candidate_policy_unapproved"


class ControlStatus(StrEnum):
    """Exactly one per evaluated control (see the module docstring)."""

    TRIGGERED = "triggered"
    PASSED = "passed"
    NOT_ASSESSABLE = "not_assessable"
    CANDIDATE_ONLY = "candidate_only"
    CONFIRMATION_REQUIRED = "confirmation_required"


#: The observation category each status stands for (fixed text; no data).
STATUS_DESCRIPTIONS: Mapping[ControlStatus, str] = MappingProxyType({
    ControlStatus.TRIGGERED: "Defect observed in the supplied evidence.",
    ControlStatus.PASSED: "Evaluated and passed: every prerequisite was available and the control held.",
    ControlStatus.NOT_ASSESSABLE: "Not assessable: prerequisite evidence was unavailable. This is never a pass.",
    ControlStatus.CANDIDATE_ONLY: ("Candidate rule only: no approved or calibrated threshold exists. Observations "
                                   "are review candidates, not alerts and not passes."),
    ControlStatus.CONFIRMATION_REQUIRED: ("Right-censored observation: no following eligible capture exists, so it "
                                          "needs another collection before confirmation."),
})


class MonitoringFinding(StrEnum):
    """Typed observations that make a control triggered, a review candidate or confirmation-required."""

    # 1 - missing expected locations
    EXPECTED_LOCATION_MISSING = "expected_location_missing"
    LOCATION_ASSIGNMENT_MISSING = "location_assignment_missing"
    COVERAGE_CONTRACT_MISMATCH = "coverage_contract_mismatch"
    UNEXCUSED_MISSING_STREAM_PERIOD = "unexcused_missing_stream_period"
    SCHEDULED_STREAMS_NOT_EXACT = "scheduled_streams_not_exact"
    GOVERNED_EXCLUSION_UNMATCHED = "governed_exclusion_unmatched"
    # 2 - job/detail count mismatches
    DECLARED_COUNT_MISSING = "declared_count_missing"
    DECLARED_COUNT_INVALID = "declared_count_invalid"
    DETAIL_ROWS_BELOW_DECLARED_COUNT = "detail_rows_below_declared_count"
    DETAIL_ROWS_ABOVE_DECLARED_COUNT = "detail_rows_above_declared_count"
    DECLARED_COUNTS_DISAGREE = "declared_counts_disagree"
    DETAIL_JOB_LINK_MISSING = "detail_job_link_missing"
    ORPHAN_DETAIL_ROWS = "orphan_detail_rows"
    PARENT_DETAIL_SCOPE_MISMATCH = "parent_detail_scope_mismatch"
    RECONCILIATION_NOT_PROVEN = "reconciliation_not_proven"
    JOB_LINKAGE_INVALID = "job_linkage_invalid"
    # 3 - duplicate or aliased location feeds
    UNAPPROVED_LOCATION_STREAM = "unapproved_location_stream"
    SOURCE_SPELLING_VARIANT = "source_spelling_variant"
    CONFLICTING_LOCATION_ASSIGNMENT = "conflicting_location_assignment"
    IDENTITY_EVIDENCE_CONFLICT = "identity_evidence_conflicts_with_decision"
    LOCATION_MAPPING_DEFECT = "location_mapping_defect"
    LOCATION_POLICY_SCOPE_INVALID = "location_policy_scope_invalid"
    ALIAS_CANONICALIZATION_NOT_APPLIED = "alias_canonicalization_not_applied"
    LOCATION_ROLES_NOT_EXACT = "location_roles_not_exact"
    COMPARISON_PAIRS_INVALID = "comparison_pairs_invalid"
    LOCATION_AUTHORITY_MISMATCH = "location_authority_mismatch"
    CANONICAL_OFFER_POLICY_MISMATCH = "canonical_offer_policy_mismatch"
    # 4 - unexpected timestamp offsets
    TIMESTAMP_PARSE_FAILURE = "timestamp_parse_failure"
    TIMESTAMP_UNRESOLVED = "timestamp_unresolved"
    TIMESTAMP_CITY_ZONE_UNKNOWN = "timestamp_city_zone_unknown_or_mismatched"
    TIMESTAMP_ORDERING_VIOLATION = "timestamp_ordering_violation"
    REPORTING_DATE_MISMATCH = "reporting_date_mismatch"
    PARENT_DETAIL_TIMESTAMP_MISMATCH = "parent_detail_timestamp_mismatch"
    FINISH_TIME_UNRESOLVABLE = "finish_time_unresolvable_in_city_zone"
    FINISH_TIME_OFF_SCHEDULE = "finish_time_off_schedule"
    CAPTURE_PERIOD_COLLISION = "capture_period_collision"
    # 5 - invalid or changing product attributes
    REQUIRED_IDENTITY_VALUE_MISSING = "required_identity_value_missing"
    ATTRIBUTE_VALUE_CONFLICT = "attribute_value_conflict"
    SAME_CAPTURE_ATTRIBUTE_CONFLICT = "same_capture_attribute_conflict"
    ATTRIBUTE_PRESENCE_INSTABILITY = "attribute_presence_instability"
    # 6 - abrupt assortment changes
    OBSERVED_DROP_REVIEW_CANDIDATE = "observed_assortment_drop_review_candidate"
    APPROVED_UNUSUAL_DROP_RULE_MET = "approved_unusual_drop_rule_met"
    # 7 - large synchronized price movements
    SYNCHRONIZED_INCREASE_REVIEW_CANDIDATE = "synchronized_increase_review_candidate"
    SYNCHRONIZED_DECREASE_REVIEW_CANDIDATE = "synchronized_decrease_review_candidate"
    APPROVED_SYNCHRONIZED_INCREASE_RULE_MET = "approved_synchronized_increase_rule_met"
    APPROVED_SYNCHRONIZED_DECREASE_RULE_MET = "approved_synchronized_decrease_rule_met"
    # 8 - unconfirmed end-of-window anomalies
    RIGHT_CENSORED_PRICE_CHANGE = "right_censored_price_change_candidate"
    RIGHT_CENSORED_ASSORTMENT_DROP = "right_censored_assortment_drop"


class EvidenceGap(StrEnum):
    """Typed prerequisite evidence that was unavailable (the reasons a control is not assessable)."""

    MONITORING_EVIDENCE_BLOCKED = "monitoring_evidence_blocked"
    EXPECTED_STREAM_CONTRACT_UNAVAILABLE = "expected_stream_contract_unavailable"
    LOCATION_COVERAGE_UNAVAILABLE = "location_coverage_unavailable"
    SCHEDULED_COVERAGE_UNAVAILABLE = "scheduled_coverage_unavailable"
    RECONCILIATION_UNAVAILABLE = "reconciliation_unavailable"
    JOB_LINKAGE_UNAVAILABLE = "job_linkage_unavailable"
    LOCATION_AUTHORITY_UNAVAILABLE = "location_authority_unavailable"
    LOCATION_IDENTITY_UNRESOLVED = "location_identity_unresolved"
    LOCATION_POLICY_UNAVAILABLE = "location_policy_unavailable"
    CANONICAL_OFFER_COMBINATION_UNAVAILABLE = "canonical_offer_combination_unavailable"
    TEMPORAL_AUTHORITY_UNAVAILABLE = "temporal_authority_unavailable"
    TEMPORAL_RECONCILIATION_UNAVAILABLE = "temporal_reconciliation_unavailable"
    TEMPORAL_RULE_UNAVAILABLE = "temporal_rule_unavailable"
    TIMESTAMP_ROWS_UNASSESSABLE = "timestamp_rows_unassessable"
    VEHICLE_STABILITY_UNAVAILABLE = "vehicle_stability_unavailable"
    PRODUCT_HISTORY_INSUFFICIENT = "product_history_insufficient"
    PRODUCT_HISTORY_TEMPORALLY_UNASSESSABLE = "product_history_temporally_unassessable"
    PRODUCT_POPULATION_EMPTY = "product_population_empty"
    VISIBLE_ASSORTMENT_UNAVAILABLE = "visible_assortment_unavailable"
    PRICE_CHANGE_ANALYSIS_UNAVAILABLE = "price_change_analysis_unavailable"


class MonitoringNote(StrEnum):
    """Typed context that never changes a status (governed exceptions, policy state, unbridged breaks)."""

    GOVERNED_EXCLUSION_APPLIED = "governed_exclusion_applied"
    GOVERNED_EXCEPTION_EXCUSED = "governed_exception_excused"
    APPROVED_ALIAS_CANONICALIZED = "approved_alias_canonicalized"
    BEHAVIORAL_SIMILARITY_IS_NOT_AUTHORITY = "behavioral_similarity_is_not_authority"
    OPERATIONAL_THRESHOLD_NOT_APPROVED = "operational_threshold_not_approved"
    NO_REVIEW_CANDIDATES_OBSERVED = "no_review_candidates_observed"
    OBSERVED_DROPS_BELOW_APPROVED_RULE = "observed_drops_below_approved_rule"
    SYNCHRONIZED_MOVEMENTS_BELOW_APPROVED_RULE = "synchronized_movements_below_approved_rule"
    MIXED_DIRECTION_INTERVALS_NOT_SYNCHRONIZED = "mixed_direction_intervals_not_synchronized"
    HARD_BREAKS_NOT_BRIDGED = "hard_breaks_not_bridged"
    EMPTY_CAPTURE_PRESENT = "empty_capture_present"
    COLLECTION_COMPLETENESS_SUSPECT = "collection_completeness_suspect"
    PERSISTENCE_NOT_TESTABLE_ACROSS_BREAK = "persistence_not_testable_across_break"
    DROP_FOLLOW_UP_BLOCKED_BY_BREAK = "drop_follow_up_blocked_by_break"
    RIGHT_CENSORED_SYNCHRONIZED_MOVEMENT = "right_censored_synchronized_movement"


# ================================================================== the control catalog


@dataclass(frozen=True, slots=True)
class MonitoringControl:
    """One required control: its definition only (independent of any sample evaluation)."""

    control_id: MonitoringControlId
    name: str
    condition: str
    severity: Severity
    business_impact: str
    recommended_response: str
    evidence_source: str
    calibration_status: CalibrationStatus

    def __post_init__(self) -> None:
        if not isinstance(self.control_id, MonitoringControlId):
            raise MonitoringContractError("control_id must be a MonitoringControlId")
        if not isinstance(self.severity, Severity):
            raise MonitoringContractError("severity must be a Severity")
        if not isinstance(self.calibration_status, CalibrationStatus):
            raise MonitoringContractError("calibration_status must be a CalibrationStatus")
        for name in ("name", "condition", "business_impact", "recommended_response", "evidence_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise MonitoringContractError(f"{name} must be non-empty text")


_C, _S, _K = MonitoringControlId, Severity, CalibrationStatus

#: The eight required controls in data-plan order (immutable; available even when the evidence is blocked).
MONITORING_CONTROLS: tuple[MonitoringControl, ...] = (
    MonitoringControl(
        _C.MISSING_EXPECTED_LOCATIONS, "Missing expected locations",
        "An approved source stream of the exhaustive expected-stream contract, or one of its required scheduled "
        "stream-periods, has no captured rows and no governed exception or exclusion. Expectations come only from "
        "the authority record, never from the locations observed in the extract, and a repeated location never "
        "compensates for a missing one.",
        _S.CRITICAL,
        "Every downstream comparison, price summary and assortment measure is biased toward the locations that "
        "happened to return data, and a city or airport-versus-downtown comparison can silently lose one side.",
        "Check collection logs and source availability for the affected stream and window; rerun or recollect the "
        "window where possible; withhold comparisons that use the affected location until coverage is restored or "
        "the gap is governed by a recorded exception.",
        "Approved expected-stream contract, expected-location coverage and per-stream scheduled coverage.",
        _K.CONTRACT_BASED),
    MonitoringControl(
        _C.JOB_DETAIL_COUNT_MISMATCHES, "Job/detail count mismatches",
        "For any single parent job, a declared detail count is missing, invalid or different from the detail rows "
        "linked to it, declared counts disagree, or detail rows are orphaned, unlinked or carry another scope than "
        "their parent. Each job is judged on its own, so over-counts and under-counts never offset.",
        _S.HIGH,
        "Lost or duplicated offers distort prices, offer counts and visible assortment for the affected capture "
        "and can mimic genuine assortment or price changes.",
        "Reconcile the parent job with its detail feed; identify ingestion or parsing loss or duplication; "
        "quarantine the affected capture before it enters analysis.",
        "Per-job job/detail reconciliation and the authority-backed job linkage.",
        _K.CONTRACT_BASED),
    MonitoringControl(
        _C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS, "Duplicate or aliased location feeds",
        "A location feed is unapproved, misspelled or assigned inconsistently, a location identity decision is "
        "missing or contradicted by authoritative identity evidence, or an approved alias is not canonicalized "
        "before analysis. Similar prices or matching offers are evidence for review only and never establish an "
        "alias.",
        _S.HIGH,
        "Duplicate feeds inflate location counts, synchronized-event counts and assortment measures, and can create "
        "false market comparisons.",
        "Validate source identifiers and collection configuration with the collection owner; apply only approved "
        "canonical mappings (Vancouver Downtown and Thurlow are one canonical location); prevent double counting "
        "until identity is resolved.",
        "Expected-location coverage, approved location roles and comparison pairs, the Vancouver identity policy "
        "with its recorded comparison evidence, and the approved canonical-offer combination.",
        _K.CONTRACT_BASED),
    MonitoringControl(
        _C.UNEXPECTED_TIMESTAMP_OFFSETS, "Unexpected timestamp offsets",
        "A finish or scrape time does not parse, cannot be resolved in its city's approved time zone, falls outside "
        "its scheduled period, collides with another capture, disagrees between parent and detail, or is later than "
        "its parent's finish time. Offsets come only from the approved city time zones, never from the machine time "
        "zone, and naive times are never reinterpreted without authority.",
        _S.HIGH,
        "Incorrect timestamps misorder captures and can manufacture or hide price and assortment changes.",
        "Validate parser and time-zone configuration; compare raw timestamp text with its canonical UTC derivation; "
        "withhold time-sequenced analysis for the affected captures.",
        "Temporal authority, temporal reconciliation and the per-stream schedule assignment of every parent job.",
        _K.CONTRACT_BASED),
    MonitoringControl(
        _C.INVALID_OR_CHANGING_PRODUCT_ATTRIBUTES, "Invalid or changing product attributes",
        "A required product-identity value is missing, or a product's structural attributes conflict over time or "
        "within one capture. Insufficient history and temporally unassessable products are reported as missing "
        "evidence, never as stable.",
        _S.HIGH,
        "Unstable product identity breaks like-for-like price comparison and can create false assortment additions "
        "and removals.",
        "Check source parsing and attribute mappings; quarantine ambiguous products; rerun product matching only "
        "after identity attributes are trustworthy.",
        "Vehicle-attribute stability of the pricing-eligible population under the product identity contract.",
        _K.CONTRACT_BASED),
    MonitoringControl(
        _C.ABRUPT_ASSORTMENT_CHANGES, "Abrupt assortment changes",
        "Candidate rule: a valid consecutive-capture interval whose observed drop in distinct returned products "
        "meets an approved unusual-drop policy (method, minimum history, grouping and threshold). No such policy is "
        "approved, so observed drops are review candidates only and are never called anomalies, collection "
        "failures or supplier withdrawals.",
        _S.MEDIUM,
        "A genuine contraction changes what customers can book, while a collection gap looks the same in the "
        "extract.",
        "Check capture completeness and collection logs; compare nearby captures; seek business or supplier "
        "corroboration before drawing a commercial conclusion. Escalate to high when an approved rule is met and "
        "collection completeness is suspect.",
        "Visible-assortment timeline, observed-drop metrics, typed interval breaks and the unusual-drop policy "
        "status.",
        _K.CANDIDATE_POLICY_UNAPPROVED),
    MonitoringControl(
        _C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS, "Large synchronized price movements",
        "Candidate rule: within one canonical location and one valid interval, the changed offers all move in the "
        "same direction and the movement meets approved changed-offer, changed-share, magnitude and "
        "cross-location parameters. Increases and decreases are evaluated separately. No parameters are approved, "
        "so synchronized movements are descriptive review candidates only, not proof of repricing or of an "
        "extraction failure.",
        _S.HIGH,
        "A genuine market move may change customer decisions, while a processing defect may contaminate many prices "
        "at once.",
        "Check raw versus processed values, recent parser or deployment changes, the affected dimensions and "
        "collection health; seek independent source or operational corroboration before publication or "
        "escalation.",
        "Observed price-change candidates and the higher-order synchronization analysis on canonical locations.",
        _K.CANDIDATE_POLICY_UNAPPROVED),
    MonitoringControl(
        _C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES, "Unconfirmed anomalies at the end of a collection window",
        "A price-change candidate or an observed assortment drop falls in the final interval of the collection "
        "window, so no following eligible capture exists to test its persistence. It requires confirmation; it is "
        "neither persistent nor failed, and missing captures and hard breaks are never bridged.",
        _S.MEDIUM,
        "A final-window movement may be a lasting change or a one-capture artifact, and acting on it early risks a "
        "false customer-facing claim.",
        "Request another eligible collection; keep the finding provisional; avoid customer-facing or causal "
        "conclusions until it is confirmed or independently corroborated.",
        "Price-change persistence with right-censoring semantics and the visible-assortment timeline.",
        _K.CONTRACT_BASED),
)

_BY_ID: Mapping[MonitoringControlId, MonitoringControl] = MappingProxyType(
    {c.control_id: c for c in MONITORING_CONTROLS})


def monitoring_control(control_id: MonitoringControlId | str) -> MonitoringControl:
    """The catalog definition of one control."""
    return _BY_ID[MonitoringControlId(control_id)]


#: What the sample evaluation does for each control (data-plan reconciliation; fixed text).
DATA_PLAN_RECONCILIATION: Mapping[MonitoringControlId, str] = MappingProxyType({
    _C.MISSING_EXPECTED_LOCATIONS: (
        "Missing expected locations: implemented from the approved expected-stream contract, expected-location "
        "coverage and per-stream scheduled coverage; contract-based and evaluated, with governed exclusions kept "
        "separate from unexplained gaps."),
    _C.JOB_DETAIL_COUNT_MISMATCHES: (
        "Job/detail count mismatches: implemented from the per-job reconciliation and the job linkage; "
        "contract-based and evaluated, so offsetting over-counts and under-counts cannot cancel."),
    _C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS: (
        "Duplicate or aliased location feeds: implemented from coverage, the location authority, the Vancouver "
        "identity policy and the canonical-offer combination; contract-based and evaluated, and behavioral "
        "similarity never establishes an alias."),
    _C.UNEXPECTED_TIMESTAMP_OFFSETS: (
        "Unexpected timestamp offsets: implemented from the temporal authority, temporal reconciliation and the "
        "per-stream schedule assignment; contract-based and evaluated against the approved city time zones."),
    _C.INVALID_OR_CHANGING_PRODUCT_ATTRIBUTES: (
        "Invalid or changing product attributes: implemented from vehicle-attribute stability; contract-based and "
        "evaluated, with missing values, insufficient history and proven conflicts kept distinct."),
    _C.ABRUPT_ASSORTMENT_CHANGES: (
        "Abrupt assortment changes: implemented from the visible-assortment engine; candidate-only because the "
        "unusual-drop policy is unavailable, so observed drops are review candidates and no alert is raised."),
    _C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS: (
        "Large synchronized price movements: implemented from the higher-order price-change analysis; "
        "candidate-only because no synchronized-movement parameters are approved, with increases and decreases "
        "kept separate."),
    _C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES: (
        "Unconfirmed anomalies at the end of a collection window: implemented from right-censored persistence and "
        "the assortment timeline; final-window observations are confirmation-required, never persistent, and "
        "breaks are never bridged."),
})

#: What production monitoring would need beyond this one-off sample (fixed text; no data).
PRODUCTION_CALIBRATION_REQUIREMENTS: tuple[str, ...] = (
    "Longer historical coverage that spans normal periods, known incidents, seasonality and collection changes.",
    "Business-owner approval of every candidate method, minimum history, grouping and threshold, recorded in the "
    "authority record before use.",
    "Back-testing of each candidate rule against labelled incidents and normal periods; no threshold may be "
    "optimized against this sample.",
    "Operational corroboration channels with the collection owner for completeness and identity questions.",
    "An approved persistence decision for evaluations and an approved alert-routing decision; this repository "
    "defines neither.",
    "Periodic review of every threshold after collection, parser, schedule or supplier changes.",
)


# ================================================================== policies


@dataclass(frozen=True)
class SynchronizedMovementPolicy:
    """A large-synchronized-movement policy; the repository default is unavailable and carries no parameters.

    Only ``APPROVED`` with a recorded authority (``record_id`` and
    ``reference``) and every parameter is executable. Parameters are refused on
    any other status, so a proposed or unavailable policy can never carry a
    threshold. The repository approves no policy: an approved one must be
    configured explicitly by its owner.

    Parameters (all inclusive minimums of one canonical location interval):
    ``minimum_changed_offers`` (increase plus decrease candidates),
    ``minimum_changed_share`` (of comparable offers, in ``(0, 1]``),
    ``minimum_median_abs_change_percent`` (median absolute percentage change,
    ``> 0``) and ``minimum_locations`` (distinct canonical locations that meet
    the interval conditions in the same direction and period).
    """

    status: AnomalyPolicyStatus
    record_id: str | None = None
    reference: str | None = None
    minimum_changed_offers: int | None = None
    minimum_changed_share: Fraction | None = None
    minimum_median_abs_change_percent: Fraction | None = None
    minimum_locations: int | None = None

    def __post_init__(self) -> None:
        E = MonitoringContractError
        if not isinstance(self.status, AnomalyPolicyStatus):
            raise E("status must be an AnomalyPolicyStatus")
        params = (self.minimum_changed_offers, self.minimum_changed_share, self.minimum_median_abs_change_percent,
                  self.minimum_locations)
        if self.status is not AnomalyPolicyStatus.APPROVED:
            if any(p is not None for p in params):
                raise E("only an approved policy may carry parameters")
            return
        if not (isinstance(self.record_id, str) and self.record_id.strip()
                and isinstance(self.reference, str) and self.reference.strip()):
            raise E("an approved policy names its authority record and reference")
        for name in ("minimum_changed_offers", "minimum_locations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise E(f"{name} must be a positive integer")
        share = self.minimum_changed_share
        if not isinstance(share, Fraction) or not 0 < share <= 1:
            raise E("minimum_changed_share must be a Fraction in (0, 1]")
        magnitude = self.minimum_median_abs_change_percent
        if not isinstance(magnitude, Fraction) or magnitude <= 0:
            raise E("minimum_median_abs_change_percent must be a positive Fraction")

    @property
    def executable(self) -> bool:
        return self.status is AnomalyPolicyStatus.APPROVED


#: No approved synchronized-movement policy exists: the control stays candidate-only.
DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY = SynchronizedMovementPolicy(AnomalyPolicyStatus.UNAVAILABLE)


# ================================================================== evaluations and reports

#: Controls whose effective severity may rise above the catalog severity (and the only level it may rise to).
_ESCALATION: Mapping[MonitoringControlId, Severity] = MappingProxyType({
    _C.ABRUPT_ASSORTMENT_CHANGES: Severity.HIGH,
    _C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES: Severity.HIGH,
})


def _typed(values: object, enum: type[StrEnum], name: str) -> tuple:
    if not isinstance(values, tuple) or not all(isinstance(v, enum) for v in values):
        raise MonitoringContractError(f"{name} must be a tuple of {enum.__name__} members")
    if len(set(values)) != len(values):
        raise MonitoringContractError(f"{name} must not repeat a member")
    return values


@dataclass(frozen=True, slots=True)
class ControlEvaluation:
    """The sample evaluation of one control: one status plus typed findings, gaps and notes (no data values)."""

    control_id: MonitoringControlId
    status: ControlStatus
    findings: tuple[MonitoringFinding, ...] = ()
    unavailable_evidence: tuple[EvidenceGap, ...] = ()
    notes: tuple[MonitoringNote, ...] = ()
    effective_severity: Severity | None = None
    #: The operational policy status of a candidate control (``None`` for contract-based controls).
    policy_status: AnomalyPolicyStatus | None = None

    def __post_init__(self) -> None:
        E = MonitoringContractError
        if not isinstance(self.control_id, MonitoringControlId):
            raise E("control_id must be a MonitoringControlId")
        if not isinstance(self.status, ControlStatus):
            raise E("status must be a ControlStatus")
        _typed(self.findings, MonitoringFinding, "findings")
        _typed(self.unavailable_evidence, EvidenceGap, "unavailable_evidence")
        _typed(self.notes, MonitoringNote, "notes")
        control = _BY_ID[self.control_id]
        if self.effective_severity is None:
            object.__setattr__(self, "effective_severity", control.severity)
        if not isinstance(self.effective_severity, Severity):
            raise E("effective_severity must be a Severity")
        if self.effective_severity is not control.severity and _ESCALATION.get(self.control_id) is not \
                self.effective_severity:
            raise E("only an approved escalation may change a control's severity")
        candidate = control.calibration_status is CalibrationStatus.CANDIDATE_POLICY_UNAPPROVED
        if candidate != (self.policy_status is not None):
            raise E("exactly the candidate controls carry a policy status")
        if self.policy_status is not None and not isinstance(self.policy_status, AnomalyPolicyStatus):
            raise E("policy_status must be an AnomalyPolicyStatus")
        S = ControlStatus
        rules = {
            S.PASSED: not self.findings and not self.unavailable_evidence,
            S.TRIGGERED: bool(self.findings),
            S.NOT_ASSESSABLE: bool(self.unavailable_evidence) and not self.findings,
            S.CANDIDATE_ONLY: candidate and self.policy_status is not AnomalyPolicyStatus.APPROVED
            and not self.unavailable_evidence,
            S.CONFIRMATION_REQUIRED: bool(self.findings)
            and self.control_id is MonitoringControlId.UNCONFIRMED_END_OF_WINDOW_ANOMALIES,
        }
        if not rules[self.status]:
            raise E(f"evaluation is inconsistent with status {self.status.value}")
        if self.status is S.TRIGGERED and self.control_id is MonitoringControlId.UNCONFIRMED_END_OF_WINDOW_ANOMALIES:
            raise E("a right-censored observation requires confirmation; it is never triggered")
        if self.status in (S.PASSED, S.TRIGGERED) and candidate and self.policy_status is not \
                AnomalyPolicyStatus.APPROVED:
            raise E("a candidate rule without an approved policy can neither pass nor trigger")
        if self.effective_severity is not control.severity and self.status not in (
                S.TRIGGERED, S.CONFIRMATION_REQUIRED):
            raise E("severity escalates only for a triggered or confirmation-required evaluation")

    @property
    def control(self) -> MonitoringControl:
        return _BY_ID[self.control_id]


class MonitoringReportStatus(StrEnum):
    EVALUATED = "evaluated"                          # every control had its prerequisite evidence
    PARTIALLY_EVALUATED = "partially_evaluated"      # at least one control is not assessable
    BLOCKED = "blocked"                              # the evidence chain is unusable: every control not assessable


class MonitoringBlocker(StrEnum):
    """Why no control could be evaluated (categories only)."""

    PIPELINE_EVIDENCE_UNAVAILABLE = "pipeline_evidence_unavailable"
    EVIDENCE_BINDING_MISMATCH = "evidence_binding_mismatch"
    DOWNSTREAM_EVIDENCE_INVALID = "downstream_evidence_invalid"


@dataclass(frozen=True)
class MonitoringReport:
    """Print-safe monitoring report: one evaluation per control, in catalog order, plus categories only."""

    status: MonitoringReportStatus
    evaluations: tuple[ControlEvaluation, ...]
    blockers: tuple[MonitoringBlocker, ...] = ()
    #: Pricing-readiness blocker categories of the run (plain values; context only).
    upstream_blockers: tuple[str, ...] = ()
    assortment_policy_status: AnomalyPolicyStatus = AnomalyPolicyStatus.UNAVAILABLE
    movement_policy_status: AnomalyPolicyStatus = AnomalyPolicyStatus.UNAVAILABLE

    def __post_init__(self) -> None:
        E = MonitoringContractError
        if not isinstance(self.status, MonitoringReportStatus):
            raise E("status must be a MonitoringReportStatus")
        if not isinstance(self.evaluations, tuple) or not all(
                isinstance(e, ControlEvaluation) for e in self.evaluations):
            raise E("evaluations must be a tuple of ControlEvaluation")
        if tuple(e.control_id for e in self.evaluations) != tuple(c.control_id for c in MONITORING_CONTROLS):
            raise E("one evaluation per required control, in catalog order")
        _typed(self.blockers, MonitoringBlocker, "blockers")
        if not all(isinstance(b, str) and b and not isinstance(b, bool) for b in self.upstream_blockers) \
                or not isinstance(self.upstream_blockers, tuple):
            raise E("upstream blockers are plain category values")
        for name in ("assortment_policy_status", "movement_policy_status"):
            if not isinstance(getattr(self, name), AnomalyPolicyStatus):
                raise E(f"{name} must be an AnomalyPolicyStatus")
        blocked = self.status is MonitoringReportStatus.BLOCKED
        if blocked != bool(self.blockers):
            raise E("exactly a blocked report carries blockers")
        if blocked and any(e.status is not ControlStatus.NOT_ASSESSABLE
                           or e.unavailable_evidence != (EvidenceGap.MONITORING_EVIDENCE_BLOCKED,)
                           for e in self.evaluations):
            raise E("a blocked report assesses no control")
        if not blocked:
            partial = any(e.status is ControlStatus.NOT_ASSESSABLE for e in self.evaluations)
            if partial != (self.status is MonitoringReportStatus.PARTIALLY_EVALUATED):
                raise E("partially evaluated exactly when a control is not assessable")
        for control_id, policy in ((_C.ABRUPT_ASSORTMENT_CHANGES, self.assortment_policy_status),
                                   (_C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS, self.movement_policy_status)):
            if self.evaluation(control_id).policy_status is not policy:
                raise E("the report's policy statuses match the candidate evaluations")

    @property
    def blocked(self) -> bool:
        return self.status is MonitoringReportStatus.BLOCKED

    def evaluation(self, control_id: MonitoringControlId | str) -> ControlEvaluation:
        key = MonitoringControlId(control_id)
        return next(e for e in self.evaluations if e.control_id is key)

    def controls_with_status(self, status: ControlStatus) -> tuple[MonitoringControlId, ...]:
        return tuple(e.control_id for e in self.evaluations if e.status is status)


def _candidate_policy(control_id: MonitoringControlId, policy: AnomalyPolicyStatus) -> AnomalyPolicyStatus | None:
    return policy if _BY_ID[control_id].calibration_status is CalibrationStatus.CANDIDATE_POLICY_UNAPPROVED else None


def blocked_monitoring_report(blockers: Sequence[MonitoringBlocker], upstream: Sequence[str] = (), *,
                              assortment_policy: AnomalyPolicyStatus = AnomalyPolicyStatus.UNAVAILABLE,
                              movement_policy: AnomalyPolicyStatus = AnomalyPolicyStatus.UNAVAILABLE
                              ) -> MonitoringReport:
    """A report whose eight controls are all not assessable (the definitions stay available)."""
    policies = {_C.ABRUPT_ASSORTMENT_CHANGES: assortment_policy, _C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS: movement_policy}
    evaluations = tuple(ControlEvaluation(
        c.control_id, ControlStatus.NOT_ASSESSABLE, unavailable_evidence=(EvidenceGap.MONITORING_EVIDENCE_BLOCKED,),
        policy_status=_candidate_policy(c.control_id, policies.get(c.control_id, AnomalyPolicyStatus.UNAVAILABLE)))
        for c in MONITORING_CONTROLS)
    return MonitoringReport(MonitoringReportStatus.BLOCKED, evaluations, blockers=tuple(dict.fromkeys(blockers)),
                            upstream_blockers=tuple(dict.fromkeys(upstream)),
                            assortment_policy_status=assortment_policy, movement_policy_status=movement_policy)


# ================================================================== evidence


@dataclass(frozen=True)
class MonitoringEvidence:
    """The validated reports one evaluation reads (all from one pipeline run; in memory only, never printed).

    Every field is optional: a missing report makes the controls that need it
    not assessable, never passed.
    """

    contract: object = field(default=None, repr=False, compare=False)
    coverage: object = field(default=None, repr=False, compare=False)
    scheduled: object = field(default=None, repr=False, compare=False)
    reconciliation: object = field(default=None, repr=False, compare=False)
    job_linkage: object = field(default=None, repr=False, compare=False)
    location_authority: object = field(default=None, repr=False, compare=False)
    canonical_offers: object = field(default=None, repr=False, compare=False)
    pricing: object = field(default=None, repr=False, compare=False)
    temporal_authority: object = field(default=None, repr=False, compare=False)
    temporal: object = field(default=None, repr=False, compare=False)
    vehicle_stability: object = field(default=None, repr=False, compare=False)
    price_changes: PriceChangeAnalysisResult | None = field(default=None, repr=False, compare=False)
    assortment: VisibleAssortmentResult | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        from ql2_sixt_canada_analysis.canonical_offers import CanonicalOfferReport
        from ql2_sixt_canada_analysis.collection_schedule import PerStreamScheduledCoverageReport
        from ql2_sixt_canada_analysis.coverage import LocationCoverageReport
        from ql2_sixt_canada_analysis.expected_stream_contract import ExpectedStreamContract
        from ql2_sixt_canada_analysis.job_linkage import JobLinkageReport
        from ql2_sixt_canada_analysis.location_authority import LocationAuthorityReport
        from ql2_sixt_canada_analysis.readiness import PricingReadinessReport
        from ql2_sixt_canada_analysis.reconciliation import JobDetailReconciliationReport
        from ql2_sixt_canada_analysis.stability import VehicleStabilityReport
        from ql2_sixt_canada_analysis.temporal import TemporalReconciliationReport
        from ql2_sixt_canada_analysis.temporal_authority import TemporalAuthority

        expected = {"contract": ExpectedStreamContract, "coverage": LocationCoverageReport,
                    "scheduled": PerStreamScheduledCoverageReport, "reconciliation": JobDetailReconciliationReport,
                    "job_linkage": JobLinkageReport, "location_authority": LocationAuthorityReport,
                    "canonical_offers": CanonicalOfferReport, "pricing": PricingReadinessReport,
                    "temporal_authority": TemporalAuthority, "temporal": TemporalReconciliationReport,
                    "vehicle_stability": VehicleStabilityReport, "price_changes": PriceChangeAnalysisResult,
                    "assortment": VisibleAssortmentResult}
        for name, kind in expected.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, kind):
                raise TypeError(f"{name} must be a {kind.__name__} or None")


class _Collector:
    """Accumulates typed findings, gaps and notes in first-seen order."""

    def __init__(self) -> None:
        self.findings: dict[MonitoringFinding, None] = {}
        self.gaps: dict[EvidenceGap, None] = {}
        self.notes: dict[MonitoringNote, None] = {}

    def finding(self, value: MonitoringFinding, when: object = True) -> None:
        if when:
            self.findings[value] = None

    def gap(self, value: EvidenceGap, when: object = True) -> None:
        if when:
            self.gaps[value] = None

    def note(self, value: MonitoringNote, when: object = True) -> None:
        if when:
            self.notes[value] = None

    def evaluation(self, control_id: MonitoringControlId, *, censoring: bool = False,
                   severity: Severity | None = None, policy: AnomalyPolicyStatus | None = None) -> ControlEvaluation:
        if self.findings:
            status = ControlStatus.CONFIRMATION_REQUIRED if censoring else ControlStatus.TRIGGERED
        elif self.gaps:
            status = ControlStatus.NOT_ASSESSABLE
        else:
            status = ControlStatus.PASSED
        return ControlEvaluation(control_id, status, tuple(self.findings), tuple(self.gaps), tuple(self.notes),
                                 effective_severity=severity, policy_status=policy)


def _pricing(evidence: MonitoringEvidence):  # type: ignore[no-untyped-def]
    return evidence.pricing


# ------------------------------------------------------------------ 1. missing expected locations


def _missing_expected_locations(ev: MonitoringEvidence) -> ControlEvaluation:
    c = _Collector()
    contract, coverage, scheduled = ev.contract, ev.coverage, ev.scheduled
    usable = contract is not None and contract.usable
    c.gap(EvidenceGap.EXPECTED_STREAM_CONTRACT_UNAVAILABLE, not usable)
    if coverage is None:
        c.gap(EvidenceGap.LOCATION_COVERAGE_UNAVAILABLE)
    else:
        violations = set(coverage.violations)
        c.finding(MonitoringFinding.EXPECTED_LOCATION_MISSING, "missing_expected_location" in violations)
        c.finding(MonitoringFinding.LOCATION_ASSIGNMENT_MISSING, "missing_location_assignment" in violations)
        if usable:   # the expected universe is the authority contract, never the observed locations
            c.finding(MonitoringFinding.COVERAGE_CONTRACT_MISMATCH,
                      tuple(coverage.expected_pairs) != tuple(contract.coverage.expected_locations))
    if scheduled is None or not scheduled.schedule.available:
        c.gap(EvidenceGap.SCHEDULED_COVERAGE_UNAVAILABLE)
    else:
        incomplete = not scheduled.streams or not all(s.complete for s in scheduled.streams)
        c.finding(MonitoringFinding.UNEXCUSED_MISSING_STREAM_PERIOD, scheduled.unexcused_missing_total > 0
                  or incomplete)
        c.finding(MonitoringFinding.SCHEDULED_STREAMS_NOT_EXACT, not scheduled.streams_exact)
        c.finding(MonitoringFinding.GOVERNED_EXCLUSION_UNMATCHED, scheduled.unmatched_exclusions > 0)
        c.note(MonitoringNote.GOVERNED_EXCLUSION_APPLIED, scheduled.excluded_periods > 0)
        c.note(MonitoringNote.GOVERNED_EXCEPTION_EXCUSED, scheduled.excused_total > 0)
    return c.evaluation(_C.MISSING_EXPECTED_LOCATIONS)


# ------------------------------------------------------------------ 2. job/detail count mismatches

_RECONCILIATION_FINDINGS: Mapping[str, MonitoringFinding] = MappingProxyType({
    "missing_expected_count": MonitoringFinding.DECLARED_COUNT_MISSING,
    "invalid_expected_count": MonitoringFinding.DECLARED_COUNT_INVALID,
    "under_count": MonitoringFinding.DETAIL_ROWS_BELOW_DECLARED_COUNT,
    "over_count": MonitoringFinding.DETAIL_ROWS_ABOVE_DECLARED_COUNT,
    "declared_counts_disagree": MonitoringFinding.DECLARED_COUNTS_DISAGREE,
    "missing_link": MonitoringFinding.DETAIL_JOB_LINK_MISSING,
    "orphan_detail": MonitoringFinding.ORPHAN_DETAIL_ROWS,
    "parent_detail_scope_mismatch": MonitoringFinding.PARENT_DETAIL_SCOPE_MISMATCH,
})


def _job_detail_count_mismatches(ev: MonitoringEvidence) -> ControlEvaluation:
    c = _Collector()
    report = ev.reconciliation
    if report is None:
        c.gap(EvidenceGap.RECONCILIATION_UNAVAILABLE)
    else:
        found = [_RECONCILIATION_FINDINGS[v] for v in report.violations]
        for f in found:
            c.finding(f)
        # Fail closed: a report that is not reconciled for every job never passes, even without a named category.
        c.finding(MonitoringFinding.RECONCILIATION_NOT_PROVEN, not found and not report.is_reconciled)
    if ev.job_linkage is None:
        c.gap(EvidenceGap.JOB_LINKAGE_UNAVAILABLE)
    else:
        c.finding(MonitoringFinding.JOB_LINKAGE_INVALID, not ev.job_linkage.is_valid)
    return c.evaluation(_C.JOB_DETAIL_COUNT_MISMATCHES)


# ------------------------------------------------------------------ 3. duplicate or aliased location feeds


def _duplicate_or_aliased_feeds(ev: MonitoringEvidence) -> ControlEvaluation:
    from ql2_sixt_canada_analysis.comparison import LocationStreamComparisonStatus as CS
    from ql2_sixt_canada_analysis.location_authority import ComparisonPairDefect, LocationAuthorityStatus
    from ql2_sixt_canada_analysis.readiness import PricingBlocker

    c = _Collector()
    if ev.coverage is None:
        c.gap(EvidenceGap.LOCATION_COVERAGE_UNAVAILABLE)
    else:
        violations = set(ev.coverage.violations)
        c.finding(MonitoringFinding.UNAPPROVED_LOCATION_STREAM, "unexpected_location" in violations)
        c.finding(MonitoringFinding.SOURCE_SPELLING_VARIANT, "source_spelling_mismatch" in violations)
        c.finding(MonitoringFinding.CONFLICTING_LOCATION_ASSIGNMENT, "conflicting_location_assignment" in violations)
    authority = ev.location_authority
    if authority is None or authority.role_map.status is not LocationAuthorityStatus.APPROVED \
            or authority.pair_set.status is not LocationAuthorityStatus.APPROVED:
        c.gap(EvidenceGap.LOCATION_AUTHORITY_UNAVAILABLE)
    else:
        c.finding(MonitoringFinding.LOCATION_ROLES_NOT_EXACT, bool(authority.role_defects))
        structural = set(authority.pair_defects) - {ComparisonPairDefect.IDENTITY_UNRESOLVED,
                                                    ComparisonPairDefect.ROLES_UNAVAILABLE}
        c.finding(MonitoringFinding.COMPARISON_PAIRS_INVALID, bool(structural))
        c.gap(EvidenceGap.LOCATION_IDENTITY_UNRESOLVED,
              ComparisonPairDefect.IDENTITY_UNRESOLVED in authority.pair_defects)
        if authority.canonicalization_merges_streams:
            offers = ev.canonical_offers
            c.gap(EvidenceGap.CANONICAL_OFFER_COMBINATION_UNAVAILABLE,
                  offers is None or not offers.policy.available or offers.binding is None)
    pricing = _pricing(ev)
    policy = pricing.location_policy if pricing is not None else None
    if policy is None:
        c.gap(EvidenceGap.LOCATION_POLICY_UNAVAILABLE)
    else:
        behavioural = {CS.LIKELY_DUPLICATE_STREAMS, CS.LIKELY_DISTINCT_STREAMS, CS.COMPARISON_INCONCLUSIVE,
                       CS.INSUFFICIENT_COMPARABLE_CAPTURES}
        c.note(MonitoringNote.BEHAVIORAL_SIMILARITY_IS_NOT_AUTHORITY, policy.behavioral_evidence in behavioural)
        c.gap(EvidenceGap.LOCATION_IDENTITY_UNRESOLVED, not policy.location_policy_resolved)
        c.finding(MonitoringFinding.IDENTITY_EVIDENCE_CONFLICT, policy.identity_evidence_conflict)
        c.finding(MonitoringFinding.LOCATION_MAPPING_DEFECT, policy.mapping_defect_indicated)
        if policy.scope is None:
            c.gap(EvidenceGap.LOCATION_POLICY_UNAVAILABLE)
        else:
            c.finding(MonitoringFinding.LOCATION_POLICY_SCOPE_INVALID, not policy.scope_valid)
        c.finding(MonitoringFinding.ALIAS_CANONICALIZATION_NOT_APPLIED,
                  policy.canonicalization_required and not policy.canonicalization_applied)
        c.note(MonitoringNote.APPROVED_ALIAS_CANONICALIZED,
               policy.locations_are_aliases and policy.canonicalization_applied)
        blockers = set(pricing.blocking_reasons)
        c.finding(MonitoringFinding.CANONICAL_OFFER_POLICY_MISMATCH,
                  PricingBlocker.CANONICAL_OFFER_POLICY_MISMATCH in blockers)
        c.finding(MonitoringFinding.LOCATION_AUTHORITY_MISMATCH, bool(
            {PricingBlocker.LOCATION_AUTHORITY_POLICY_MISMATCH, PricingBlocker.LOCATION_AUTHORITY_CONTRACT_MISMATCH}
            & blockers))
    return c.evaluation(_C.DUPLICATE_OR_ALIASED_LOCATION_FEEDS)


# ------------------------------------------------------------------ 4. unexpected timestamp offsets

_TEMPORAL_FINDINGS: Mapping[str, MonitoringFinding] = MappingProxyType({
    "field_parse": MonitoringFinding.TIMESTAMP_PARSE_FAILURE,
    "time_unresolved": MonitoringFinding.TIMESTAMP_UNRESOLVED,
    "city_mismatch": MonitoringFinding.TIMESTAMP_CITY_ZONE_UNKNOWN,
    "ordering": MonitoringFinding.TIMESTAMP_ORDERING_VIOLATION,
    "date_derivation": MonitoringFinding.REPORTING_DATE_MISMATCH,
    "replication": MonitoringFinding.PARENT_DETAIL_TIMESTAMP_MISMATCH,
})
_TEMPORAL_GAPS: Mapping[str, EvidenceGap] = MappingProxyType({
    "rule_unavailable": EvidenceGap.TEMPORAL_RULE_UNAVAILABLE,
    "unassessable": EvidenceGap.TIMESTAMP_ROWS_UNASSESSABLE,
})
_ASSIGNMENT_FINDINGS: Mapping[str, MonitoringFinding] = MappingProxyType({
    "missing_city": MonitoringFinding.TIMESTAMP_CITY_ZONE_UNKNOWN,
    "unknown_city": MonitoringFinding.TIMESTAMP_CITY_ZONE_UNKNOWN,
    "missing_finished_at": MonitoringFinding.FINISH_TIME_UNRESOLVABLE,
    "invalid_finished_at": MonitoringFinding.FINISH_TIME_UNRESOLVABLE,
    "nonexistent_local_time": MonitoringFinding.FINISH_TIME_UNRESOLVABLE,
    "ambiguous_local_time": MonitoringFinding.FINISH_TIME_UNRESOLVABLE,
    "outside_schedule_window": MonitoringFinding.FINISH_TIME_OFF_SCHEDULE,
    "no_expected_period": MonitoringFinding.FINISH_TIME_OFF_SCHEDULE,
    "duplicate_city_period": MonitoringFinding.CAPTURE_PERIOD_COLLISION,
    "detail_copy_mismatch": MonitoringFinding.PARENT_DETAIL_TIMESTAMP_MISMATCH,
})


def _unexpected_timestamp_offsets(ev: MonitoringEvidence) -> ControlEvaluation:
    c = _Collector()
    authority = ev.temporal_authority
    c.gap(EvidenceGap.TEMPORAL_AUTHORITY_UNAVAILABLE, authority is None or bool(authority.blocking_reasons))
    temporal = ev.temporal
    if temporal is None:
        c.gap(EvidenceGap.TEMPORAL_RECONCILIATION_UNAVAILABLE)
    else:
        violations = temporal.violations
        for v in violations:
            if v in _TEMPORAL_FINDINGS:
                c.finding(_TEMPORAL_FINDINGS[v])
            else:
                c.gap(_TEMPORAL_GAPS[v])
        c.gap(EvidenceGap.TIMESTAMP_ROWS_UNASSESSABLE, not violations and not temporal.is_valid)
    scheduled = ev.scheduled
    if scheduled is None or not scheduled.schedule.available:
        c.gap(EvidenceGap.SCHEDULED_COVERAGE_UNAVAILABLE)
    else:
        for failure, count in scheduled.job_failure_counts.items():
            c.finding(_ASSIGNMENT_FINDINGS[failure], count > 0)
        c.finding(MonitoringFinding.PARENT_DETAIL_TIMESTAMP_MISMATCH, scheduled.detail_copy_mismatches > 0)
    return c.evaluation(_C.UNEXPECTED_TIMESTAMP_OFFSETS)


# ------------------------------------------------------------------ 5. invalid or changing product attributes

_STABILITY_FINDINGS: Mapping[str, MonitoringFinding] = MappingProxyType({
    "incomplete_identity": MonitoringFinding.REQUIRED_IDENTITY_VALUE_MISSING,
    "value_conflict": MonitoringFinding.ATTRIBUTE_VALUE_CONFLICT,
    "same_capture_conflict": MonitoringFinding.SAME_CAPTURE_ATTRIBUTE_CONFLICT,
    "presence_instability": MonitoringFinding.ATTRIBUTE_PRESENCE_INSTABILITY,
})


def _invalid_or_changing_product_attributes(ev: MonitoringEvidence) -> ControlEvaluation:
    c = _Collector()
    report = ev.vehicle_stability
    if report is None:
        c.gap(EvidenceGap.VEHICLE_STABILITY_UNAVAILABLE)
    else:
        for v in report.violations:
            if v == "temporally_unassessable":
                c.gap(EvidenceGap.PRODUCT_HISTORY_TEMPORALLY_UNASSESSABLE)
            else:
                c.finding(_STABILITY_FINDINGS[v])
        c.gap(EvidenceGap.PRODUCT_HISTORY_INSUFFICIENT, report.insufficient_history_entities > 0)
        c.gap(EvidenceGap.PRODUCT_POPULATION_EMPTY, report.distinct_entities == 0)
        # Fail closed: only a PASSED report can pass.
        c.gap(EvidenceGap.VEHICLE_STABILITY_UNAVAILABLE, not report.is_valid and not c.findings and not c.gaps)
    return c.evaluation(_C.INVALID_OR_CHANGING_PRODUCT_ATTRIBUTES)


# ------------------------------------------------------------------ 6. abrupt assortment changes


def _abrupt_assortment_changes(ev: MonitoringEvidence, policy: AnomalyPolicyStatus) -> ControlEvaluation:
    c = _Collector()
    result = ev.assortment
    if result is None or not result.completed:
        c.gap(EvidenceGap.VISIBLE_ASSORTMENT_UNAVAILABLE)
        return c.evaluation(_C.ABRUPT_ASSORTMENT_CHANGES, policy=policy)
    report, overall = result.report, result.report.overall
    c.note(MonitoringNote.HARD_BREAKS_NOT_BRIDGED, any(s.breaks for s in report.locations))
    c.note(MonitoringNote.EMPTY_CAPTURE_PRESENT, overall.empty_captures > 0)
    suspect = overall.missing_captures > 0 or overall.excluded_captures > 0 or overall.empty_captures > 0
    executable = policy is AnomalyPolicyStatus.APPROVED and report.unusual_drop_intervals is not None
    if not executable:
        c.finding(MonitoringFinding.OBSERVED_DROP_REVIEW_CANDIDATE, overall.drop_intervals > 0)
        c.note(MonitoringNote.NO_REVIEW_CANDIDATES_OBSERVED, overall.drop_intervals == 0)
        c.note(MonitoringNote.OPERATIONAL_THRESHOLD_NOT_APPROVED)
        return ControlEvaluation(_C.ABRUPT_ASSORTMENT_CHANGES, ControlStatus.CANDIDATE_ONLY, tuple(c.findings),
                                 (), tuple(c.notes), policy_status=policy)
    met = report.unusual_drop_intervals > 0
    c.finding(MonitoringFinding.APPROVED_UNUSUAL_DROP_RULE_MET, met)
    c.note(MonitoringNote.OBSERVED_DROPS_BELOW_APPROVED_RULE, not met and overall.drop_intervals > 0)
    c.note(MonitoringNote.COLLECTION_COMPLETENESS_SUSPECT, met and suspect)
    severity = Severity.HIGH if met and suspect else None
    return c.evaluation(_C.ABRUPT_ASSORTMENT_CHANGES, severity=severity, policy=policy)


# ------------------------------------------------------------------ 7. large synchronized price movements

_PREV, _CUR = EVENT_INTERVAL_COLUMNS
_DIRECTIONS = (
    (MovementClass.SYNCHRONIZED_INCREASE, MonitoringFinding.SYNCHRONIZED_INCREASE_REVIEW_CANDIDATE,
     MonitoringFinding.APPROVED_SYNCHRONIZED_INCREASE_RULE_MET),
    (MovementClass.SYNCHRONIZED_DECREASE, MonitoringFinding.SYNCHRONIZED_DECREASE_REVIEW_CANDIDATE,
     MonitoringFinding.APPROVED_SYNCHRONIZED_DECREASE_RULE_MET),
)


def _at_least(value: object, minimum: Fraction) -> bool:
    """``value >= minimum`` with exact rational comparison; a missing value never qualifies."""
    if value is None or (isinstance(value, float) and value != value) or value is pd.NA:
        return False
    return Fraction(value) >= minimum


def _rule_met(table: pd.DataFrame, movement: MovementClass, policy: SynchronizedMovementPolicy) -> bool:
    """The approved rule for one direction over canonical location intervals (aliases are one location)."""
    periods: dict[str, set[tuple[str, str]]] = {}
    for row in table.to_dict("records"):
        if row["movement_class"] != movement.value:
            continue
        if row["price_change_count"] < policy.minimum_changed_offers \
                or not _at_least(row["changed_share_of_comparable"], policy.minimum_changed_share) \
                or not _at_least(row["median_abs_change_percent"], policy.minimum_median_abs_change_percent):
            continue
        periods.setdefault(row[_CUR], set()).add((row["canonical_city"], row["canonical_location"]))
    return any(len(locations) >= policy.minimum_locations for locations in periods.values())


def _large_synchronized_movements(ev: MonitoringEvidence, policy: SynchronizedMovementPolicy) -> ControlEvaluation:
    c = _Collector()
    analysis = ev.price_changes
    if analysis is None or not analysis.completed:
        c.gap(EvidenceGap.PRICE_CHANGE_ANALYSIS_UNAVAILABLE)
        return c.evaluation(_C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS, policy=policy.status)
    classes = dict(analysis.report.movement_classes)
    c.note(MonitoringNote.MIXED_DIRECTION_INTERVALS_NOT_SYNCHRONIZED,
           classes.get(MovementClass.MIXED_DIRECTION.value, 0) > 0)
    if not policy.executable:
        for movement, candidate, _ in _DIRECTIONS:      # increases and decreases are never netted
            c.finding(candidate, classes.get(movement.value, 0) > 0)
        c.note(MonitoringNote.NO_REVIEW_CANDIDATES_OBSERVED, not c.findings)
        c.note(MonitoringNote.OPERATIONAL_THRESHOLD_NOT_APPROVED)
        return ControlEvaluation(_C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS, ControlStatus.CANDIDATE_ONLY,
                                 tuple(c.findings), (), tuple(c.notes), policy_status=policy.status)
    any_synchronized = False
    for movement, _, met in _DIRECTIONS:
        any_synchronized |= classes.get(movement.value, 0) > 0
        c.finding(met, _rule_met(analysis.event_table, movement, policy))
    c.note(MonitoringNote.SYNCHRONIZED_MOVEMENTS_BELOW_APPROVED_RULE, any_synchronized and not c.findings)
    return c.evaluation(_C.LARGE_SYNCHRONIZED_PRICE_MOVEMENTS, policy=policy.status)


# ------------------------------------------------------------------ 8. unconfirmed end-of-window anomalies


def _final_window_drops(result: VisibleAssortmentResult) -> tuple[bool, bool]:
    """(an observed drop ends at a location's final scheduled capture, a drop's follow-up crosses a break)."""
    rows = result.timeline.to_dict("records")
    by_location: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_location.setdefault((row["canonical_city"], row["canonical_location"]), []).append(row)
    censored = blocked = False
    for location_rows in by_location.values():
        location_rows = sorted(location_rows, key=lambda r: r[EVENT_TIMESTAMP_COLUMN])
        for i, row in enumerate(location_rows):
            if row["assessability_status"] != AssessabilityStatus.ASSESSED.value or not row["absolute_drop"] > 0:
                continue
            if i == len(location_rows) - 1:
                censored = True
            elif location_rows[i + 1]["assessability_status"] != AssessabilityStatus.ASSESSED.value:
                blocked = True                       # never bridged: the next capture is a break or not eligible
    return censored, blocked


def _right_censored_synchronized(analysis: PriceChangeAnalysisResult) -> bool:
    persistence = analysis.persistence
    censored = persistence[persistence["not_testable_reason"] == NotTestableReason.RIGHT_CENSORED_FINAL_CAPTURE.value]
    keys = set(zip(censored["canonical_city"], censored["canonical_location"], censored[_PREV], censored[_CUR]))
    synchronized = {MovementClass.SYNCHRONIZED_INCREASE.value, MovementClass.SYNCHRONIZED_DECREASE.value}
    table = analysis.event_table
    rows = table[table["movement_class"].isin(synchronized)]
    return any(k in keys for k in zip(rows["canonical_city"], rows["canonical_location"], rows[_PREV], rows[_CUR]))


def _unconfirmed_end_of_window(ev: MonitoringEvidence) -> ControlEvaluation:
    c = _Collector()
    escalate = False
    analysis = ev.price_changes
    if analysis is None or not analysis.completed:
        c.gap(EvidenceGap.PRICE_CHANGE_ANALYSIS_UNAVAILABLE)
    else:
        reasons = dict(analysis.report.persistence.not_testable_reasons)
        right = reasons.get(NotTestableReason.RIGHT_CENSORED_FINAL_CAPTURE.value, 0)
        c.finding(MonitoringFinding.RIGHT_CENSORED_PRICE_CHANGE, right > 0)
        c.note(MonitoringNote.PERSISTENCE_NOT_TESTABLE_ACROSS_BREAK, sum(reasons.values()) - right > 0)
        escalate = right > 0 and _right_censored_synchronized(analysis)
        c.note(MonitoringNote.RIGHT_CENSORED_SYNCHRONIZED_MOVEMENT, escalate)
    result = ev.assortment
    if result is None or not result.completed:
        c.gap(EvidenceGap.VISIBLE_ASSORTMENT_UNAVAILABLE)
    else:
        censored, blocked = _final_window_drops(result)
        c.finding(MonitoringFinding.RIGHT_CENSORED_ASSORTMENT_DROP, censored)
        c.note(MonitoringNote.DROP_FOLLOW_UP_BLOCKED_BY_BREAK, blocked)
    return c.evaluation(_C.UNCONFIRMED_END_OF_WINDOW_ANOMALIES, censoring=True,
                        severity=Severity.HIGH if escalate else None)


# ------------------------------------------------------------------ evaluation entry point


def evaluate_monitoring_controls(
        evidence: MonitoringEvidence, *,
        movement_policy: SynchronizedMovementPolicy = DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY) -> MonitoringReport:
    """Evaluate all eight controls on one evidence bundle (pure: no I/O, no mutation, deterministic).

    The assortment policy status is the one the supplied assortment result was
    computed with (unavailable when there is none).
    """
    if not isinstance(evidence, MonitoringEvidence):
        raise TypeError("evidence must be a MonitoringEvidence")
    if not isinstance(movement_policy, SynchronizedMovementPolicy):
        raise TypeError("movement_policy must be a SynchronizedMovementPolicy")
    assortment_policy = (evidence.assortment.report.anomaly_policy_status if evidence.assortment is not None
                         else AnomalyPolicyStatus.UNAVAILABLE)
    evaluations = (
        _missing_expected_locations(evidence),
        _job_detail_count_mismatches(evidence),
        _duplicate_or_aliased_feeds(evidence),
        _unexpected_timestamp_offsets(evidence),
        _invalid_or_changing_product_attributes(evidence),
        _abrupt_assortment_changes(evidence, assortment_policy),
        _large_synchronized_movements(evidence, movement_policy),
        _unconfirmed_end_of_window(evidence),
    )
    partial = any(e.status is ControlStatus.NOT_ASSESSABLE for e in evaluations)
    pricing = evidence.pricing
    upstream = tuple(b.value for b in pricing.blocking_reasons) if pricing is not None else ()
    return MonitoringReport(
        MonitoringReportStatus.PARTIALLY_EVALUATED if partial else MonitoringReportStatus.EVALUATED, evaluations,
        upstream_blockers=upstream, assortment_policy_status=assortment_policy,
        movement_policy_status=movement_policy.status)


# ------------------------------------------------------------------ pipeline entry points


@dataclass(frozen=True)
class MonitoringResult:
    """The print-safe report plus the in-memory evidence it was evaluated from (never printed or written)."""

    report: MonitoringReport
    evidence: MonitoringEvidence | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.report, MonitoringReport):
            raise TypeError("report must be a MonitoringReport")
        if self.report.blocked != (self.evidence is None):
            raise MonitoringContractError("exactly a blocked result holds no evidence")


def monitoring_evidence_from_pipeline(run: object, *, assortment_policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY
                                      ) -> MonitoringEvidence | MonitoringBlocker:
    """The evidence bundle of one pipeline result, or the blocker that prevents binding it.

    Never reruns the pipeline. The price-change analysis and the visible
    assortment are derived from ``run`` itself and must be bound to its frames
    and its location authority.
    """
    from ql2_sixt_canada_analysis.price_change_analysis import price_change_analysis_from_pipeline
    from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult
    from ql2_sixt_canada_analysis.pricing_population import frame_binding
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport
    from ql2_sixt_canada_analysis.visible_assortment import visible_assortment_from_pipeline

    if not isinstance(run, PricingPipelineResult):
        raise TypeError("run must be a PricingPipelineResult")
    if not isinstance(assortment_policy, UnusualDropPolicy):
        raise TypeError("assortment_policy must be an UnusualDropPolicy")
    pricing = run.pricing
    if not isinstance(pricing, PricingReadinessReport) or not isinstance(run.jobs, pd.DataFrame) \
            or not isinstance(run.cars, pd.DataFrame):
        return MonitoringBlocker.PIPELINE_EVIDENCE_UNAVAILABLE
    if (pricing.location_authority is not None and pricing.location_authority is not run.location_authority) \
            or (pricing.scheduled_coverage is not None and pricing.scheduled_coverage is not run.scheduled):
        return MonitoringBlocker.EVIDENCE_BINDING_MISMATCH
    try:
        analysis = price_change_analysis_from_pipeline(run)
        assortment = visible_assortment_from_pipeline(run, policy=assortment_policy)
    except (PriceChangeContractError, AssortmentContractError):
        return MonitoringBlocker.DOWNSTREAM_EVIDENCE_INVALID
    binding = frame_binding(run.jobs, run.cars)
    if analysis.completed and (analysis.events.binding != binding
                               or analysis.location_authority is not run.location_authority):
        return MonitoringBlocker.EVIDENCE_BINDING_MISMATCH
    if assortment.completed and (assortment.binding != binding
                                 or assortment.location_authority is not run.location_authority
                                 or assortment.price_changes.binding != binding):
        return MonitoringBlocker.EVIDENCE_BINDING_MISMATCH
    return MonitoringEvidence(
        contract=run.contract, coverage=run.coverage, scheduled=run.scheduled, reconciliation=run.reconciliation,
        job_linkage=run.job_linkage, location_authority=run.location_authority, canonical_offers=run.canonical_offers,
        pricing=pricing, temporal_authority=run.temporal_authority, temporal=run.temporal,
        vehicle_stability=run.vehicle_stability, price_changes=analysis, assortment=assortment)


def monitoring_from_pipeline(run: object, *, assortment_policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY,
                             movement_policy: SynchronizedMovementPolicy = DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY
                             ) -> MonitoringResult:
    """Evaluate every control on one :class:`~ql2_sixt_canada_analysis.pricing_pipeline.PricingPipelineResult`."""
    if not isinstance(movement_policy, SynchronizedMovementPolicy):
        raise TypeError("movement_policy must be a SynchronizedMovementPolicy")
    evidence = monitoring_evidence_from_pipeline(run, assortment_policy=assortment_policy)
    if isinstance(evidence, MonitoringBlocker):
        pricing = getattr(run, "pricing", None)
        upstream = tuple(b.value for b in getattr(pricing, "blocking_reasons", ()))
        return MonitoringResult(blocked_monitoring_report(
            [evidence], upstream, assortment_policy=assortment_policy.status,
            movement_policy=movement_policy.status))
    return MonitoringResult(evaluate_monitoring_controls(evidence, movement_policy=movement_policy), evidence)


def run_monitoring(raw_dir: str | Path | None = None, *,
                   assortment_policy: UnusualDropPolicy = DEFAULT_UNUSUAL_DROP_POLICY,
                   movement_policy: SynchronizedMovementPolicy = DEFAULT_SYNCHRONIZED_MOVEMENT_POLICY
                   ) -> MonitoringResult:
    """Run ``run_pricing_pipeline`` exactly once, then evaluate every control on that result (nothing is written)."""
    from ql2_sixt_canada_analysis import pricing_pipeline

    if not isinstance(movement_policy, SynchronizedMovementPolicy):
        raise TypeError("movement_policy must be a SynchronizedMovementPolicy")
    if not isinstance(assortment_policy, UnusualDropPolicy):
        raise TypeError("assortment_policy must be an UnusualDropPolicy")
    return monitoring_from_pipeline(pricing_pipeline.run_pricing_pipeline(raw_dir),
                                    assortment_policy=assortment_policy, movement_policy=movement_policy)


# ================================================================== sanitized presentation

#: The public control table: exactly one row per required control, in catalog order.
MONITORING_TABLE_COLUMNS: tuple[str, ...] = (
    "control_id", "control", "condition", "severity", "effective_severity", "likely_business_impact",
    "recommended_response", "calibration_status", "evaluation_status", "observation", "findings",
    "unavailable_evidence", "notes", "policy_status")

_SEPARATOR = ", "
_ENUM_COLUMNS: Mapping[str, frozenset[str]] = MappingProxyType({
    "control_id": frozenset(MonitoringControlId),
    "severity": frozenset(Severity),
    "effective_severity": frozenset(Severity),
    "calibration_status": frozenset(CalibrationStatus),
    "evaluation_status": frozenset(ControlStatus),
    "policy_status": frozenset({*AnomalyPolicyStatus, ""}),
})
_LIST_COLUMNS: Mapping[str, frozenset[str]] = MappingProxyType({
    "findings": frozenset(MonitoringFinding),
    "unavailable_evidence": frozenset(EvidenceGap),
    "notes": frozenset(MonitoringNote),
})


def _joined(values: Sequence[StrEnum]) -> str:
    return _SEPARATOR.join(v.value for v in values)


def monitoring_control_table(report: MonitoringReport) -> pd.DataFrame:
    """The sanitized control table (catalog text, enum values and typed codes only; validated before return)."""
    if not isinstance(report, MonitoringReport):
        raise TypeError("report must be a MonitoringReport")
    rows = []
    for evaluation in report.evaluations:
        control = evaluation.control
        rows.append({
            "control_id": control.control_id.value, "control": control.name, "condition": control.condition,
            "severity": control.severity.value, "effective_severity": evaluation.effective_severity.value,
            "likely_business_impact": control.business_impact, "recommended_response": control.recommended_response,
            "calibration_status": control.calibration_status.value, "evaluation_status": evaluation.status.value,
            "observation": STATUS_DESCRIPTIONS[evaluation.status], "findings": _joined(evaluation.findings),
            "unavailable_evidence": _joined(evaluation.unavailable_evidence), "notes": _joined(evaluation.notes),
            "policy_status": evaluation.policy_status.value if evaluation.policy_status is not None else ""})
    frame = pd.DataFrame(rows, columns=list(MONITORING_TABLE_COLUMNS))
    validate_monitoring_table(frame)
    return frame


def validate_monitoring_table(frame: object) -> None:
    """Refuse any table that is not exactly the sanitized control table (messages name only the rule).

    Raises:
        MonitoringContractError: Wrong type, schema, row count or order; a value outside the fixed catalog,
            the enums or the typed codes.
    """
    E = MonitoringContractError
    if not isinstance(frame, pd.DataFrame):
        raise E("the control table must be a DataFrame")
    if tuple(frame.columns) != MONITORING_TABLE_COLUMNS:
        raise E("the control table must have the exact sanitized schema")
    if tuple(frame["control_id"]) != tuple(c.control_id.value for c in MONITORING_CONTROLS):
        raise E("the control table holds one row per required control, in catalog order")
    descriptions = frozenset(STATUS_DESCRIPTIONS.values())
    for row in frame.to_dict("records"):
        if not all(isinstance(v, str) for v in row.values()):
            raise E("every control-table value is text")
        control = _BY_ID[MonitoringControlId(row["control_id"])]
        catalog = {"control": control.name, "condition": control.condition, "severity": control.severity.value,
                   "likely_business_impact": control.business_impact,
                   "recommended_response": control.recommended_response,
                   "calibration_status": control.calibration_status.value}
        if any(row[k] != v for k, v in catalog.items()):
            raise E("definition columns must equal the fixed control catalog")
        for column, allowed in _ENUM_COLUMNS.items():
            if row[column] not in allowed:
                raise E(f"value outside its enum in column {column}")
        if row["observation"] not in descriptions or row["observation"] != STATUS_DESCRIPTIONS[
                ControlStatus(row["evaluation_status"])]:
            raise E("the observation must describe the evaluation status")
        for column, allowed in _LIST_COLUMNS.items():
            codes = row[column].split(_SEPARATOR) if row[column] else []
            if any(code not in allowed for code in codes) or len(set(codes)) != len(codes):
                raise E(f"value outside its typed codes in column {column}")


def severity_scale_table() -> pd.DataFrame:
    """The severity scale (fixed text): what each level means and the expected response."""
    return pd.DataFrame([{"severity": s.value, "meaning": SEVERITY_DESCRIPTIONS[s][0],
                          "expected_response": SEVERITY_DESCRIPTIONS[s][1]} for s in Severity],
                        columns=["severity", "meaning", "expected_response"])


def status_legend_table() -> pd.DataFrame:
    """The evaluation statuses (fixed text): the five distinct kinds of observation."""
    return pd.DataFrame([{"evaluation_status": s.value, "observation": STATUS_DESCRIPTIONS[s]} for s in ControlStatus],
                        columns=["evaluation_status", "observation"])


def monitoring_summary_lines(report: MonitoringReport) -> tuple[str, ...]:
    """Deterministic, print-safe summary lines (status values and control identifiers only; no data)."""
    if not isinstance(report, MonitoringReport):
        raise TypeError("report must be a MonitoringReport")
    lines = [f"Monitoring evaluation status: {report.status.value}"]
    if report.blocked:
        lines.append("Blocked - every control is not assessable; the control definitions remain available.")
        lines.append("Blockers: " + _joined(report.blockers))
    for status in ControlStatus:
        controls = report.controls_with_status(status)
        lines.append(f"{status.value}: " + (_joined(controls) if controls else "none"))
    escalated = [e.control_id for e in report.evaluations if e.effective_severity is not e.control.severity]
    lines.append("Escalated severity: " + (_joined(escalated) if escalated else "none"))
    lines.append(f"Unusual-drop policy status: {report.assortment_policy_status.value}")
    lines.append(f"Synchronized-movement policy status: {report.movement_policy_status.value}")
    if report.upstream_blockers:
        lines.append("Pricing-readiness blockers: " + _SEPARATOR.join(report.upstream_blockers))
    lines.append("No alert, notification or file was produced.")
    return tuple(lines)
