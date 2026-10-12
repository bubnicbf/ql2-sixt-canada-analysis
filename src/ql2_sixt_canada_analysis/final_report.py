"""Final report (data-plan Section 7): one pipeline run, every analysis, deterministic sanitized interpretation.

:func:`run_final_report` runs :func:`~ql2_sixt_canada_analysis.pricing_pipeline.run_pricing_pipeline`
exactly once and passes that one result to the existing ``*_from_pipeline``
entry points: matched-location pricing, the price-change presentation, the
visible-assortment presentation and the monitoring controls. Nothing is
recomputed here and no analytical rule is added; this module only
orchestrates, checks that the evidence is bound to the one run, and turns
the existing aggregate reports into short, deterministic interpretation text
for ``notebooks/06_final_report.ipynb``.

Fail-closed rules:

* A run that is not a bound :class:`~ql2_sixt_canada_analysis.pricing_pipeline.PricingPipelineResult`
  (no readiness report, no frames, or reports substituted after the run was
  built) produces no commercial section at all: only blocker categories and
  the blocked monitoring catalog.
* A run whose central readiness gate is blocked still calls each downstream
  entry point, which fails closed itself and reports its typed blockers; no
  commercial finding is shown as valid.
* Interpretation text is generated only from report counts, enum values,
  approved configuration keys and summary statistics already shown by the
  earlier notebooks. It never contains rows, product identities, identifiers,
  source labels, paths or file names, and it never makes causal claims.

The open questions and additional-data requests are fixed catalogs
(:data:`OPEN_QUESTIONS`, :data:`DATA_REQUESTS`). Their identifiers are
mirrored in ``docs/assumptions_exclusions_and_open_questions.md`` and the
tests keep the two in step. Importing this module performs no I/O.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import pandas as pd

__all__ = [
    "DATA_REQUESTS",
    "FINAL_SECTION_TABLE_COLUMNS",
    "OPEN_QUESTIONS",
    "DataRequest",
    "FinalReportBlocker",
    "FinalReportError",
    "FinalReportResult",
    "FinalReportStatus",
    "OpenQuestion",
    "ReportSection",
    "SectionOutcome",
    "data_requests_table",
    "final_conclusions",
    "final_report_from_pipeline",
    "final_section_table",
    "final_status_lines",
    "interpret_section",
    "matched_premium_png",
    "matched_summary_table",
    "monitoring_overview_table",
    "open_questions_table",
    "render_question_catalogs",
    "run_final_report",
    "validate_final_section_table",
]


class FinalReportError(ValueError):
    """A final-report object or table violates its contract (messages name the rule, never values)."""


class FinalReportStatus(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"


class FinalReportBlocker(StrEnum):
    """Why the final report cannot present every section as valid (categories only)."""

    PIPELINE_EVIDENCE_UNAVAILABLE = "pipeline_evidence_unavailable"
    EVIDENCE_BINDING_MISMATCH = "evidence_binding_mismatch"
    PRICING_NOT_READY = "pricing_not_ready"
    MATCHED_PRICING_BLOCKED = "matched_pricing_blocked"
    PRICE_CHANGES_BLOCKED = "price_changes_blocked"
    VISIBLE_ASSORTMENT_BLOCKED = "visible_assortment_blocked"
    MONITORING_BLOCKED = "monitoring_blocked"


class ReportSection(StrEnum):
    """The result sections of the final notebook, in narrative order."""

    READINESS = "data_and_pipeline_readiness"
    MATCHED_PRICING = "matched_location_pricing"
    PRICE_CHANGES = "price_change_events"
    VISIBLE_ASSORTMENT = "visible_assortment"
    MONITORING = "monitoring_and_actionability"


#: Exact schema of :func:`final_section_table`.
FINAL_SECTION_TABLE_COLUMNS: tuple[str, ...] = ("section", "status", "findings_valid", "blockers")

_CODE = re.compile(r"[a-z][a-z0-9_]*")
_SEPARATOR = ", "
_SECTION_STATUSES = frozenset({"ready", "blocked", "completed", "evaluated", "partially_evaluated", "unavailable"})


@dataclass(frozen=True, slots=True)
class SectionOutcome:
    """Status of one report section: enum values and blocker codes only."""

    section: ReportSection
    status: str
    findings_valid: bool
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.section, ReportSection):
            raise FinalReportError("section must be a ReportSection")
        if self.status not in _SECTION_STATUSES:
            raise FinalReportError("section status outside the allowed values")
        if not isinstance(self.findings_valid, bool):
            raise FinalReportError("findings_valid must be a bool")
        if any(not isinstance(b, str) or not _CODE.fullmatch(b) for b in self.blockers):
            raise FinalReportError("blockers must be snake_case codes")
        if self.findings_valid and self.blockers:
            raise FinalReportError("a section with valid findings has no blockers")


@dataclass(frozen=True)
class FinalReportResult:
    """Every section of one pipeline run. Downstream results are in memory only (never printed)."""

    status: FinalReportStatus
    blockers: tuple[FinalReportBlocker, ...]
    pricing_ready: bool
    evidence_bound: bool
    readiness_blockers: tuple[str, ...]
    authority_record: str | None
    monitoring: object = field(repr=False)
    matched: object = field(default=None, repr=False)
    price_changes: object = field(default=None, repr=False)
    assortment: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from ql2_sixt_canada_analysis.monitoring import MonitoringResult

        if not isinstance(self.status, FinalReportStatus):
            raise FinalReportError("status must be a FinalReportStatus")
        if not isinstance(self.monitoring, MonitoringResult):
            raise FinalReportError("monitoring must be a MonitoringResult")
        if (self.status is FinalReportStatus.COMPLETED) == bool(self.blockers):
            raise FinalReportError("exactly a blocked report has blockers")
        if self.status is FinalReportStatus.COMPLETED and not (self.pricing_ready and self.evidence_bound):
            raise FinalReportError("a completed report needs ready, bound evidence")
        downstream = (self.matched, self.price_changes, self.assortment)
        if not self.evidence_bound and any(d is not None for d in downstream):
            raise FinalReportError("unbound evidence produces no commercial section")
        if any(not isinstance(b, str) or not _CODE.fullmatch(b) for b in self.readiness_blockers):
            raise FinalReportError("readiness blockers must be snake_case codes")

    @property
    def completed(self) -> bool:
        return self.status is FinalReportStatus.COMPLETED

    def section(self, section: ReportSection | str) -> SectionOutcome:
        """The status of one section."""
        return _OUTCOMES[ReportSection(section)](self)

    @property
    def sections(self) -> tuple[SectionOutcome, ...]:
        return tuple(self.section(s) for s in ReportSection)

    def findings_valid(self, section: ReportSection | str) -> bool:
        """Whether the section's findings may be shown as valid (never true for blocked evidence)."""
        return self.section(section).findings_valid


# ------------------------------------------------------------------ section outcomes


def _codes(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(getattr(v, "value", v)) for v in values))


def _readiness_outcome(r: FinalReportResult) -> SectionOutcome:
    valid = r.pricing_ready and r.evidence_bound
    blockers = () if valid else _codes([b for b in r.blockers if b in _UPSTREAM_BLOCKERS]) + r.readiness_blockers
    if not valid and not blockers:
        blockers = (FinalReportBlocker.PRICING_NOT_READY.value,)
    return SectionOutcome(ReportSection.READINESS, "ready" if valid else "blocked", valid, blockers)


def _downstream_outcome(section: ReportSection, result: object, upstream: tuple[str, ...]) -> SectionOutcome:
    if result is None:
        return SectionOutcome(section, "unavailable", False, upstream or (FinalReportBlocker.PIPELINE_EVIDENCE_UNAVAILABLE.value,))
    report = result.report  # type: ignore[attr-defined]
    if report.completed:
        return SectionOutcome(section, "completed", True)
    blockers = _codes(report.blockers) + _codes(getattr(report, "upstream_blockers", ()))
    blockers += _codes(getattr(report, "readiness_blockers", ()))
    return SectionOutcome(section, "blocked", False, tuple(dict.fromkeys(blockers)))


def _upstream(r: FinalReportResult) -> tuple[str, ...]:
    return _codes([b for b in r.blockers if b in _UPSTREAM_BLOCKERS])


def _monitoring_outcome(r: FinalReportResult) -> SectionOutcome:
    report = r.monitoring.report  # type: ignore[attr-defined]
    status = report.status.value
    if report.blocked:
        return SectionOutcome(ReportSection.MONITORING, status, False,
                              _codes(report.blockers) + _codes(report.upstream_blockers))
    return SectionOutcome(ReportSection.MONITORING, status, True)


_UPSTREAM_BLOCKERS = frozenset({FinalReportBlocker.PIPELINE_EVIDENCE_UNAVAILABLE,
                                FinalReportBlocker.EVIDENCE_BINDING_MISMATCH})

_OUTCOMES = MappingProxyType({
    ReportSection.READINESS: _readiness_outcome,
    ReportSection.MATCHED_PRICING: lambda r: _downstream_outcome(ReportSection.MATCHED_PRICING, r.matched, _upstream(r)),
    ReportSection.PRICE_CHANGES: lambda r: _downstream_outcome(ReportSection.PRICE_CHANGES, r.price_changes,
                                                               _upstream(r)),
    ReportSection.VISIBLE_ASSORTMENT: lambda r: _downstream_outcome(ReportSection.VISIBLE_ASSORTMENT, r.assortment,
                                                                    _upstream(r)),
    ReportSection.MONITORING: _monitoring_outcome,
})


# ------------------------------------------------------------------ orchestration


def final_report_from_pipeline(run: object) -> FinalReportResult:
    """Every section from one existing pipeline result (the pipeline is never rerun).

    The four downstream entry points receive the same ``run`` object, so no
    report or frame from another run can be mixed in. Evidence that is not
    bound to the run produces no commercial section.
    """
    import ql2_sixt_canada_analysis.matched_location_pricing as matched_pricing
    from ql2_sixt_canada_analysis.assortment_presentation import assortment_presentation_from_pipeline
    from ql2_sixt_canada_analysis.monitoring import monitoring_from_pipeline
    from ql2_sixt_canada_analysis.price_change_presentation import presentation_from_pipeline
    from ql2_sixt_canada_analysis.pricing_pipeline import PricingPipelineResult, pipeline_evidence_bound
    from ql2_sixt_canada_analysis.readiness import PricingReadinessReport

    if not isinstance(run, PricingPipelineResult):
        raise TypeError("run must be a PricingPipelineResult")
    record_id = getattr(run.record, "record_id", None)
    authority = record_id if isinstance(record_id, str) and record_id else None
    monitoring = monitoring_from_pipeline(run)
    pricing = run.pricing
    if not isinstance(pricing, PricingReadinessReport) or not isinstance(run.jobs, pd.DataFrame) \
            or not isinstance(run.cars, pd.DataFrame):
        return FinalReportResult(FinalReportStatus.BLOCKED, (FinalReportBlocker.PIPELINE_EVIDENCE_UNAVAILABLE,),
                                 False, False, (), authority, monitoring)
    readiness_blockers = _codes(pricing.blocking_reasons)
    if not pipeline_evidence_bound(run):
        return FinalReportResult(FinalReportStatus.BLOCKED, (FinalReportBlocker.EVIDENCE_BINDING_MISMATCH,),
                                 bool(pricing.ready), False, readiness_blockers, authority, monitoring)
    matched = matched_pricing.matched_location_pricing_from_pipeline(run)
    price_changes = presentation_from_pipeline(run)
    assortment = assortment_presentation_from_pipeline(run)
    blockers: list[FinalReportBlocker] = []
    if not pricing.ready:
        blockers.append(FinalReportBlocker.PRICING_NOT_READY)
    for result, blocker in ((matched, FinalReportBlocker.MATCHED_PRICING_BLOCKED),
                            (price_changes, FinalReportBlocker.PRICE_CHANGES_BLOCKED),
                            (assortment, FinalReportBlocker.VISIBLE_ASSORTMENT_BLOCKED)):
        if not result.completed:
            blockers.append(blocker)
    if monitoring.report.blocked:
        blockers.append(FinalReportBlocker.MONITORING_BLOCKED)
    status = FinalReportStatus.BLOCKED if blockers else FinalReportStatus.COMPLETED
    return FinalReportResult(status, tuple(blockers), bool(pricing.ready), True, readiness_blockers, authority,
                             monitoring, matched, price_changes, assortment)


def run_final_report(raw_dir: str | Path | None = None) -> FinalReportResult:
    """Run ``run_pricing_pipeline`` exactly once, then every section on that one result (nothing is written)."""
    from ql2_sixt_canada_analysis import pricing_pipeline

    return final_report_from_pipeline(pricing_pipeline.run_pricing_pipeline(raw_dir))


# ------------------------------------------------------------------ sanitized tables and figures


def final_section_table(result: FinalReportResult) -> pd.DataFrame:
    """One row per report section: status, whether its findings are valid and its blocker codes (validated)."""
    if not isinstance(result, FinalReportResult):
        raise TypeError("result must be a FinalReportResult")
    frame = pd.DataFrame([{"section": s.section.value, "status": s.status, "findings_valid": s.findings_valid,
                           "blockers": _SEPARATOR.join(s.blockers)} for s in result.sections],
                         columns=list(FINAL_SECTION_TABLE_COLUMNS))
    validate_final_section_table(frame)
    return frame


def validate_final_section_table(frame: object) -> None:
    """Refuse anything but the sanitized section table (enum values, booleans and snake_case codes only)."""
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != FINAL_SECTION_TABLE_COLUMNS:
        raise FinalReportError("the section table must have the exact sanitized schema")
    if tuple(frame["section"]) != tuple(s.value for s in ReportSection):
        raise FinalReportError("the section table holds one row per section, in narrative order")
    for row in frame.to_dict("records"):
        if row["status"] not in _SECTION_STATUSES or not isinstance(row["findings_valid"], bool):
            raise FinalReportError("status or validity outside the allowed values")
        codes = row["blockers"].split(_SEPARATOR) if row["blockers"] else []
        if any(not _CODE.fullmatch(c) for c in codes) or (row["findings_valid"] and codes):
            raise FinalReportError("blockers must be snake_case codes and absent from valid sections")


def final_status_lines(result: FinalReportResult) -> tuple[str, ...]:
    """Deterministic, print-safe status lines (status values, record identifier and blocker codes only)."""
    if not isinstance(result, FinalReportResult):
        raise TypeError("result must be a FinalReportResult")
    lines = [f"Final report status: {result.status.value}",
             f"Pricing-authority record: {result.authority_record or 'unavailable'}",
             f"Pricing readiness: {'ready' if result.pricing_ready else 'blocked'}",
             f"Evidence bound to one pipeline run: {result.evidence_bound}"]
    if result.blockers:
        lines.append("Final-report blockers: " + _SEPARATOR.join(b.value for b in result.blockers))
    if result.readiness_blockers:
        lines.append("Pricing-readiness blockers: " + _SEPARATOR.join(result.readiness_blockers))
    lines.append("No files written.")
    return tuple(lines)


def matched_premium_png(result: FinalReportResult, *, dpi: int = 100) -> bytes:
    """The existing matched-premium figure, rendered in memory (completed matched pricing only)."""
    import matplotlib.pyplot as plt

    from ql2_sixt_canada_analysis.matched_location_pricing import plot_matched_location_premiums

    if not isinstance(result, FinalReportResult) or not result.findings_valid(ReportSection.MATCHED_PRICING):
        raise FinalReportError("the figure needs completed matched-location pricing")
    figure, _ = plot_matched_location_premiums(result.matched)  # type: ignore[arg-type]
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=dpi, metadata={"Software": None})
    finally:
        plt.close(figure)
    return buffer.getvalue()


def matched_summary_table(result: FinalReportResult) -> pd.DataFrame:
    """The existing per-city premium summary (``city_summary_frame``) of completed matched pricing."""
    from ql2_sixt_canada_analysis.matched_location_pricing import city_summary_frame

    if not isinstance(result, FinalReportResult) or not result.findings_valid(ReportSection.MATCHED_PRICING):
        raise FinalReportError("the summary needs completed matched-location pricing")
    return city_summary_frame(result.matched)


#: Columns of the compact monitoring overview (a subset of the validated control table).
_MONITORING_OVERVIEW_COLUMNS = ("control_id", "severity", "effective_severity", "calibration_status",
                                "evaluation_status", "findings", "unavailable_evidence")


def monitoring_overview_table(result: FinalReportResult) -> pd.DataFrame:
    """Enum and typed-code columns of the validated monitoring control table, one row per control."""
    from ql2_sixt_canada_analysis.monitoring import monitoring_control_table

    if not isinstance(result, FinalReportResult):
        raise TypeError("result must be a FinalReportResult")
    table = monitoring_control_table(result.monitoring.report)  # type: ignore[attr-defined]
    return table.loc[:, list(_MONITORING_OVERVIEW_COLUMNS)].reset_index(drop=True)


# ------------------------------------------------------------------ interpretation


def _n(count: int, singular: str, plural: str | None = None) -> str:
    """``count`` with the singular or plural noun."""
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _share(n: int, d: int) -> str:
    return f"{n} of {d} ({100 * n / d:.0f}%)" if d else f"{n} of 0 (share unavailable)"


def _signed(value: float | None, unit: str) -> str:
    if value is None:
        return "unavailable"
    return f"{value:+.2f}{unit}"


def _money(value: float | None, units: tuple[tuple[str, str], ...]) -> str:
    if value is None:
        return "unavailable"
    marker, basis = units[0] if len(units) == 1 else ("", "")
    sign = "+" if value >= 0 else "-"
    return f"{sign}{marker}{abs(value):.2f}" + (f"/{basis}" if basis else "")


_BLOCKED_TEXT = {
    ReportSection.MATCHED_PRICING: "no airport-versus-downtown premium can be stated for this run",
    ReportSection.PRICE_CHANGES: "no price movement, synchronization or persistence finding can be stated",
    ReportSection.VISIBLE_ASSORTMENT: "no assortment count, retention, similarity or drop finding can be stated",
    ReportSection.MONITORING: "no control can be shown as passed or triggered",
}


def _blocked(section: ReportSection, outcome: SectionOutcome) -> str:
    codes = _SEPARATOR.join(outcome.blockers) or "unavailable"
    return (f"Blocked ({codes}): {_BLOCKED_TEXT[section]}. Resolve the blocker categories and rerun; "
            "nothing in this section is evidence of market behaviour.")


def _interpret_readiness(r: FinalReportResult) -> str:
    record = r.authority_record or "an unavailable authority record"
    if r.findings_valid(ReportSection.READINESS):
        return (f"Every foundational gate passed under {record}, and every section below was derived from this "
                "one pipeline run. Findings describe the pricing-eligible population only: the governed Calgary "
                "capture and any ineligible, unmatched, ambiguous or untrusted records are excluded from them. "
                "Readiness supports descriptive analysis of this window; it is not evidence about other periods.")
    outcome = r.section(ReportSection.READINESS)
    codes = _SEPARATOR.join(outcome.blockers) or "unavailable"
    return (f"Pricing evidence is blocked ({codes}) under {record}. No matched-location, price-change or "
            "visible-assortment finding is valid for this run, and the extract should not support pricing "
            "decisions until these controls pass. Contract-based monitoring controls may still describe the "
            "structural defects.")


def _interpret_matched(r: FinalReportResult) -> str:
    report = r.matched.report  # type: ignore[attr-defined]
    units = report.price_units
    parts = []
    for city in report.cities:
        n = city.counts.matched
        if n == 0:
            parts.append(f"{city.city}: no matched pair (premium unavailable)")
            continue
        parts.append(f"{city.city}: airport higher in {_share(city.signs.positive, n)} matched pairs, equal in "
                     f"{city.signs.zero}, lower in {city.signs.negative}; median premium "
                     f"{_money(city.dollars.median, units)} ({_signed(city.percent.median, '%')} of the downtown "
                     "price)")
    overall = report.overall
    rate = overall.counts.match_rate
    coverage = (f"{overall.counts.matched} matched pairs from {overall.counts.candidate_groups} candidate groups "
                f"(match rate {rate:.0%})" if rate is not None else "no candidate groups")
    return ("; ".join(parts) + f". Overall: {coverage}. These are descriptive differences in listed prices for "
            "the same product, job and rental period; they are associational, not causal, and hourly captures of "
            "one product are repeated measurements. A gap worth investigating needs longer history and the "
            "revenue-management context before any pricing decision.")


_EVENT = "appearance or disappearance event"


def _interpret_price_changes(r: FinalReportResult) -> str:
    analysis = r.price_changes.analysis  # type: ignore[attr-defined]
    report = analysis.report
    overall = analysis.events.report.overall
    classes = dict(report.movement_classes)
    synchronized = classes.get("synchronized_increase", 0) + classes.get("synchronized_decrease", 0)
    p = report.persistence
    reasons = dict(p.not_testable_reasons)
    censored = reasons.get("right_censored_final_capture", 0)
    return (f"{_n(report.price_change_count, 'price-change candidate')} ({_n(overall.increase, 'increase')}, "
            f"{_n(overall.decrease, 'decrease')}) and {_n(report.assortment_event_count, _EVENT)} were observed "
            f"over {_n(report.intervals, 'eligible one-hour location interval')}"
            f"; {_n(synchronized, 'interval')} direction-synchronized. Of "
            f"{_n(p.with_following_interval, 'change')} with a following eligible interval, {p.held} held, "
            f"{p.continued} continued, {p.reverted} reverted, {p.disappeared} disappeared and {p.ambiguous} "
            f"ambiguous; {p.not_testable} not testable ({censored} right-censored at the final capture) and "
            "excluded from persistence conclusions. "
            "Synchronized movements are descriptive candidates, not proof of intentional repricing or of an "
            "extraction error; source or operational corroboration is needed to tell them apart.")


def _interpret_assortment(r: FinalReportResult) -> str:
    assortment = r.assortment.assortment  # type: ignore[attr-defined]
    report = assortment.report
    c = report.overall
    retention = (f"pooled retention {c.retained / c.previous_products:.2f}" if c.previous_products
                 else "pooled retention unavailable (zero denominator)")
    return (f"Across {_n(len(report.locations), 'canonical location')}, "
            f"{_n(c.assessed_intervals, 'consecutive one-hour interval')} assessed ({retention}; "
            f"{_n(c.additions, 'addition')} and {_n(c.removals, 'removal')}). Observed drops: "
            f"{_n(c.drop_intervals, 'interval')}; intervals coinciding with a price change at the same location: "
            f"{c.coincident_intervals}; eligible captures with no returned product: {c.empty_captures}. Visible "
            "assortment is what the collection returned, not proof of supplier availability. With the unusual-drop "
            f"policy {report.anomaly_policy_status.value}, observed drops are review candidates, and coincidence "
            "is not causation.")


def _interpret_monitoring(r: FinalReportResult) -> str:
    from ql2_sixt_canada_analysis.monitoring import ControlStatus

    report = r.monitoring.report  # type: ignore[attr-defined]
    counts = {s: len(report.controls_with_status(s)) for s in ControlStatus}
    triggered = report.controls_with_status(ControlStatus.TRIGGERED)
    names = _SEPARATOR.join(c.value for c in triggered) or "none"
    return (f"Of the eight controls, {counts[ControlStatus.TRIGGERED]} triggered ({names}), "
            f"{counts[ControlStatus.PASSED]} passed, {counts[ControlStatus.NOT_ASSESSABLE]} not assessable, "
            f"{counts[ControlStatus.CANDIDATE_ONLY]} candidate-only and "
            f"{counts[ControlStatus.CONFIRMATION_REQUIRED]} confirmation-required. Triggered contract-based controls "
            "name defects to raise with the collection owner; candidate-only rules list review candidates and are "
            "not production alerts, and not-assessable is never a pass. Confirmation-required observations need "
            "the next eligible collection.")


_INTERPRETERS = MappingProxyType({
    ReportSection.MATCHED_PRICING: _interpret_matched,
    ReportSection.PRICE_CHANGES: _interpret_price_changes,
    ReportSection.VISIBLE_ASSORTMENT: _interpret_assortment,
    ReportSection.MONITORING: _interpret_monitoring,
})


def interpret_section(result: FinalReportResult, section: ReportSection | str) -> str:
    """Deterministic, sanitized interpretation of one section (blocked sections state what cannot be concluded)."""
    if not isinstance(result, FinalReportResult):
        raise TypeError("result must be a FinalReportResult")
    section = ReportSection(section)
    if section is ReportSection.READINESS:
        return _interpret_readiness(result)
    outcome = result.section(section)
    if not outcome.findings_valid:
        return _blocked(section, outcome)
    return _INTERPRETERS[section](result)


def final_conclusions(result: FinalReportResult) -> tuple[str, ...]:
    """One deterministic conclusion per analytical question (status-driven; blocked evidence concludes nothing)."""
    if not isinstance(result, FinalReportResult):
        raise TypeError("result must be a FinalReportResult")
    valid = {s: result.findings_valid(s) for s in ReportSection}
    record = result.authority_record or "the unavailable authority record"
    if valid[ReportSection.READINESS]:
        reliability = (f"Coverage and reliability: the extract passed every foundational gate under {record}, so "
                       "this window can support descriptive pricing analysis; the monitoring table lists any "
                       "contract-based defect still to raise with the collection owner.")
    else:
        reliability = ("Coverage and reliability: the extract is not pricing ready for this run, so it should not "
                       "support customer pricing decisions until the listed blocker categories are resolved.")
    if valid[ReportSection.MATCHED_PRICING]:
        structure = ("Pricing structure: matched airport-versus-downtown premiums are summarized by city above; "
                     "they compare listed prices for identical products in the same collection job and are "
                     "associational, not causal.")
    else:
        structure = "Pricing structure: no matched-location premium can be concluded from this run."
    if valid[ReportSection.PRICE_CHANGES] and valid[ReportSection.VISIBLE_ASSORTMENT]:
        change = ("Change and anomaly detection: price movements and visible-assortment changes are summarized as "
                  "descriptive candidates; without source snapshots or collection logs they cannot be classified "
                  "as genuine repricing or as collection defects, and right-censored final-window observations "
                  "need the next eligible collection.")
    else:
        change = ("Change and anomaly detection: no price-change or visible-assortment conclusion is valid for "
                  "this run.")
    if valid[ReportSection.MONITORING]:
        action = ("Actionability: the contract-based controls can run on future extracts now; the abrupt-assortment "
                  "and synchronized-movement rules stay candidate-only until the business owner approves a "
                  "method, minimum history, grouping and threshold, and the open questions and data requests "
                  "below remain unanswered.")
    else:
        action = ("Actionability: monitoring evidence is blocked, so no control result is available; the open "
                  "questions and data requests below remain unanswered.")
    return reliability, structure, change, action


# ------------------------------------------------------------------ open questions and data requests


@dataclass(frozen=True, slots=True)
class OpenQuestion:
    """An unanswered analytical or governance question (never answered or guessed here)."""

    question_id: str
    question: str
    why_it_matters: str
    likely_owner: str


@dataclass(frozen=True, slots=True)
class DataRequest:
    """Additional evidence requested: what is needed, why, and the owner the governance records support."""

    request_id: str
    request: str
    purpose: str
    likely_owner: str


_NOT_RECORDED = "not recorded in governance documentation"

#: Open questions, in priority order. Owners are named only where a governance record supports them.
OPEN_QUESTIONS: tuple[OpenQuestion, ...] = (
    OpenQuestion("unusual_assortment_drop_policy",
                 "Should an unusual-assortment-drop method, minimum history, grouping and threshold be approved?",
                 "Until approved, observed drops are review candidates only and the control stays candidate-only.",
                 "business owner (visible-assortment contract proposal)"),
    OpenQuestion("synchronized_movement_policy",
                 "Should synchronized-price-movement thresholds and grouping rules be approved?",
                 "Until approved, synchronized movements are descriptive and the control stays candidate-only.",
                 "business owner (production calibration requirements)"),
    OpenQuestion("production_monitoring_operations",
                 "If production monitoring is wanted, which persistence, alert-routing, ownership, escalation and "
                 "review policies apply?",
                 "No scheduler, persistence, notification or dashboard exists; none can be designed without them.",
                 "business owner for timeline persistence; " + _NOT_RECORDED + " for routing and escalation"),
    OpenQuestion("rental_context_stratification",
                 "Should captures with several rental search contexts be stratified or rejected?",
                 "The assortment contract compares one rental search context per interval.",
                 "collection owner and business owner jointly (visible-assortment contract proposal)"),
    OpenQuestion("repricing_versus_collection_defect",
                 "Do the observed synchronized movements and drops reflect market repricing or extraction and "
                 "processing defects?",
                 "The extract alone cannot tell them apart; any commercial reading depends on the answer.",
                 _NOT_RECORDED),
    OpenQuestion("final_window_persistence",
                 "Did the price and assortment changes at the final capture persist?",
                 "Right-censored observations are confirmation-required and excluded from persistence conclusions.",
                 "collection owner (collection schedule governance)"),
    OpenQuestion("generalization_beyond_window",
                 "Do the matched premiums and change patterns hold for other periods, rental dates, durations and "
                 "seasons?",
                 "The roughly 90-hour sample is descriptive and cannot establish normal variation or seasonality.",
                 _NOT_RECORDED),
    OpenQuestion("competitive_and_channel_context",
                 "Which suppliers, channels or competitor rates should the premiums be compared with for win, meet "
                 "or loss decisions?",
                 "The files contain one supplier and one apparent source mode, so competitive questions are out of "
                 "scope.",
                 _NOT_RECORDED),
)

#: Additional evidence requested, in priority order.
DATA_REQUESTS: tuple[DataRequest, ...] = (
    DataRequest("longer_history",
                "A longer historical period of the same job and detail exports.",
                "Cover normal variation, seasonality, collection changes and known incidents; required before any "
                "threshold calibration.",
                _NOT_RECORDED),
    DataRequest("next_eligible_collections",
                "The scheduled collections that follow the supplied window.",
                "Resolve right-censored end-of-window observations (confirm or refute persistence).",
                "collection owner (collection schedule governance)"),
    DataRequest("collection_logs_and_capture_completeness",
                "Collection logs and capture-completeness evidence for the reviewed intervals.",
                "Separate genuine changes from incomplete captures, retries or parser issues.",
                "collection owner (operational corroboration channel)"),
    DataRequest("source_snapshots_or_supplier_evidence",
                "Raw source snapshots or supplier-side evidence for selected intervals.",
                "Distinguish market repricing from extraction or processing defects.",
                _NOT_RECORDED),
    DataRequest("labelled_incidents_and_normal_periods",
                "Labelled collection incidents and confirmed normal periods.",
                "Back-test any future monitoring rule; no threshold may be optimized against this sample.",
                _NOT_RECORDED),
    DataRequest("additional_search_contexts",
                "Captures for other pickup dates, rental durations and booking lead times.",
                "Test whether premiums and assortment depend on the rental search context.",
                _NOT_RECORDED),
    DataRequest("competitor_and_channel_rates",
                "Comparable rates from other suppliers or channels for the same searches.",
                "Required for supplier, channel and win/meet/loss questions, which this extract cannot answer.",
                _NOT_RECORDED),
)


def open_questions_table() -> pd.DataFrame:
    """The open-question catalog (fixed text; no data)."""
    return pd.DataFrame([{"question_id": q.question_id, "question": q.question, "why_it_matters": q.why_it_matters,
                          "likely_owner": q.likely_owner} for q in OPEN_QUESTIONS],
                        columns=["question_id", "question", "why_it_matters", "likely_owner"])


def data_requests_table() -> pd.DataFrame:
    """The additional-data request catalog (fixed text; no data)."""
    return pd.DataFrame([{"request_id": d.request_id, "request": d.request, "purpose": d.purpose,
                          "likely_owner": d.likely_owner} for d in DATA_REQUESTS],
                        columns=["request_id", "request", "purpose", "likely_owner"])


def _md(text: str) -> str:
    return text.replace("|", "\\|")


def render_question_catalogs() -> dict[str, str]:
    """Markdown tables of :data:`OPEN_QUESTIONS` and :data:`DATA_REQUESTS` for the generated documentation blocks."""
    questions = ["| ID | Question | Why it matters | Likely owner |", "| --- | --- | --- | --- |"]
    questions += [f"| `{q.question_id}` | {_md(q.question)} | {_md(q.why_it_matters)} | {_md(q.likely_owner)} |"
                  for q in OPEN_QUESTIONS]
    requests = ["| ID | Evidence requested | Why it is needed | Likely owner |", "| --- | --- | --- | --- |"]
    requests += [f"| `{d.request_id}` | {_md(d.request)} | {_md(d.purpose)} | {_md(d.likely_owner)} |"
                 for d in DATA_REQUESTS]
    return {"open-questions": "\n".join(questions), "data-requests": "\n".join(requests)}
