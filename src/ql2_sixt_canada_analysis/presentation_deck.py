"""Six-slide reveal.js presentation built from one final-report run (sanitized values, exported figures).

The deck is generated, never hand-edited. :func:`build_deck` takes one
completed :class:`~ql2_sixt_canada_analysis.final_report.FinalReportResult`
(one pipeline run) and writes a self-contained directory:

* ``deck_results.json``: the machine-readable results artifact. Every
  quantitative value shown on a slide or in the speaker notes is one entry,
  with its raw value, the exact display string, its kind, the report field it
  comes from and, where the final-report notebook prints it, the exact text the
  notebook output contains (``trace``).
* ``figures/*.png``: figures exported by the authoritative plotting functions
  (:func:`~ql2_sixt_canada_analysis.matched_location_pricing.plot_matched_location_premiums`
  and :func:`~ql2_sixt_canada_analysis.price_change_analysis.plot_price_change_heatmap`)
  at slide size, with their titles, axis labels, legends and colour-bar labels
  recorded in the results artifact.
* ``index.html``: the committed template (``presentation/deck.html``) with each
  ``{{ key }}`` placeholder replaced by the display value, wrapped in
  ``<span data-value="key">`` so the deck can be checked against the artifact.
* the committed stylesheet, initialisation script and vendored reveal.js.

:func:`validate_deck` re-checks a built directory: the HTML must equal a fresh
render of the template from the stored results (no stale or edited deck), the
stored results must equal a fresh computation from the run when one is given
(no stale numbers), every value span must match its display string, figures
must be valid PNGs with labelled axes and units, there must be exactly six
primary slides each with speaker notes, every local asset must exist and no
reference may be remote, absolute or machine-specific.

Confidentiality: the values are the same sanitized aggregates the final-report
notebook prints (counts, medians, shares, enum values and approved
configuration keys). They are still derived from proprietary data, so the
built deck belongs in the Git-ignored ``reports/presentation/`` directory and
is never committed. Fail-closed: a report that is not completed, or a figure
or value that does not reconcile, produces no deck. Importing this module
performs no I/O.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import shutil
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from ql2_sixt_canada_analysis.paths import PROJECT_ROOT

__all__ = [
    "DECK_FIGURES",
    "DECK_OUTPUT_DIR",
    "DECK_RESULTS_FILENAME",
    "DECK_SLIDE_COUNT",
    "DECK_SOURCE_DIR",
    "DeckError",
    "DeckValidationError",
    "ObservationWindow",
    "build_deck",
    "deck_results",
    "main",
    "observation_window",
    "opening_conclusion",
    "render_deck_html",
    "retained_share",
    "validate_deck",
    "verify_traces",
]

#: Committed deck sources: template, stylesheet, initialisation script and vendored reveal.js.
DECK_SOURCE_DIR = PROJECT_ROOT / "presentation"
#: Default (Git-ignored) output directory of the built deck.
DECK_OUTPUT_DIR = PROJECT_ROOT / "reports" / "presentation"
DECK_RESULTS_FILENAME = "deck_results.json"
DECK_SLIDE_COUNT = 6
_TEMPLATE = "deck.html"
_STATIC = ("deck.css", "deck.js")
_VENDOR = "vendor"
_SCHEMA = "ql2-sixt-deck-results/v1"
_FIGURE_DPI = 150

#: Exported figures: file name, slide-size figure and font scale, and the label rules each must meet.
DECK_FIGURES: Mapping[str, Mapping[str, Any]] = {
    "matched_premiums": {
        "file": "figures/matched_premiums.png",
        "figsize": (13.0, 5.0),
        "font_scale": 1.15,
        "required": {
            "title": r"airport vs downtown",
            "axes_titles": (r"^Dollar premium \(\S+ per \w+\)$", r"^Percentage premium \(% of downtown price\)$"),
            "ylabels": (r"\(\S+/\w+\)$", r"\(%\)$"),
        },
    },
    "price_change_heatmap": {
        "file": "figures/price_change_heatmap.png",
        "figsize": (13.0, 6.2),
        "font_scale": 1.1,
        "required": {
            "title": r"price moves",
            "axes_titles": (r"^Price increases per interval$", r"^Price decreases per interval$"),
            "xlabels": (r"\(UTC\)$",),
            "colorbar_labels": (r"\(count\)$", r"\(count\)$"),
            "legend": (r"No comparable prior capture", r"no change"),
        },
    },
}

_PLACEHOLDER = re.compile(r"\{\{\s*([=>]?)\s*([a-z][a-z0-9_]*)\s*\}\}")
_UNRESOLVED = re.compile(r"\{\{|\}\}|\{%|%\}|\$\{")
_FORBIDDEN_TEXT = re.compile(r"\b(TODO|TBD|FIXME|XXX|lorem ipsum|placeholder)\b", re.IGNORECASE)
#: Claims the deck must never make (mirrors the final-report wording rules).
_FORBIDDEN_CLAIMS = re.compile(
    r"\b(proves?|proven|caused by|causes|is intentional|intentionally repriced|supplier (removed|withdrew)|"
    r"availability changed|collection failed|alert raised|statistically significant|confirmed alias|"
    r"persisted beyond)\b", re.IGNORECASE)
_REMOTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)
_MACHINE_PATH = re.compile(r"(?:^|[\"'(=\s])(?:/(?:Users|home|sessions|tmp|mnt|private|var|root)/|[A-Za-z]:\\)")
_MIN_NOTE_WORDS = 60


class DeckError(ValueError):
    """The deck cannot be built from this run (not completed, or a value does not reconcile)."""


class DeckValidationError(AssertionError):
    """A built deck is stale, inconsistent or structurally invalid."""


# ============================================================================ values


@dataclass(frozen=True)
class _Value:
    value: Any
    display: str
    kind: str
    source: str
    trace: str | None = None


@dataclass
class _Values:
    items: dict[str, _Value] = field(default_factory=dict)

    def add(self, key: str, value: Any, display: str, kind: str, source: str, trace: str | None = None) -> None:
        if key in self.items:
            raise DeckError(f"duplicate deck value {key}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise DeckError(f"invalid deck value key {key!r}")
        self.items[key] = _Value(value, display, kind, source, trace)

    def count(self, key: str, value: int, source: str, trace: str | None = None) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DeckError(f"{key} must be a non-negative count")
        self.add(key, value, f"{value:,}", "count", source, trace)

    def text(self, key: str, value: str, source: str, trace: str | None = None) -> None:
        self.add(key, value, value, "text", source, trace)


def _money(value: float, symbol: str, basis: str) -> str:
    """Signed price difference: two decimals, currency marker and price basis (as in the figure)."""
    sign = "+" if value > 0 else ("-" if value < 0 else "")
    return f"{sign}{symbol}{abs(value):,.2f}" + (f"/{basis}" if basis else "")


def _pct(value: float, decimals: int = 1) -> str:
    """Signed percentage of the downtown price (one decimal, as in the figure annotations)."""
    return f"{value:+.{decimals}f}%"


def _whole_share(numerator: int, denominator: int) -> str:
    """A share as a whole percentage (as in the notebook interpretation)."""
    if not denominator:
        raise DeckError("a share needs a positive denominator")
    return f"{100 * numerator / denominator:.0f}%"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or singular + "s")


def _period(value: str) -> dt.datetime:
    from ql2_sixt_canada_analysis.price_change_events import PriceChangeContractError, parse_scheduled_period

    try:
        return parse_scheduled_period(value)
    except PriceChangeContractError as error:
        raise DeckError(f"invalid scheduled capture period in the observation window: {error}") from None


_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                 "twenty")


def _number_word(count: int) -> str:
    """``count`` as an English word up to twenty, otherwise as digits with thousands separators."""
    return _NUMBER_WORDS[count] if 0 <= count < len(_NUMBER_WORDS) else f"{count:,}"


@dataclass(frozen=True)
class ObservationWindow:
    """The observation window of the scheduled captures and its rounded, spoken duration.

    ``elapsed_hours`` is the time from the first to the last scheduled capture
    period. The endpoints are capture instants, so neither endpoint adds an
    hour (exclusive of any time after the last capture); ``capture_periods``
    counts the distinct scheduled periods, both endpoints included. The
    duration phrase uses whole days, rounded half up (``floor(hours / 24 +
    0.5)``), prefixed "Approximately" unless the elapsed time is an exact
    number of days. Windows shorter than 36 hours are stated in exact hours.
    """

    first: dt.datetime
    last: dt.datetime
    elapsed_hours: float
    capture_periods: int
    rounded_days: int | None
    phrase: str


def observation_window(periods: Sequence[str] | Any) -> ObservationWindow:
    """The observation window of scheduled capture periods (``YYYYMMDDTHHMMSSZ``; fails closed).

    Raises:
        DeckError: No period, an invalid period, or a single instant (no duration).
    """
    instants = sorted({_period(p) for p in periods})
    if not instants:
        raise DeckError("the observation window is empty: the assortment timeline has no scheduled capture")
    first, last = instants[0], instants[-1]
    elapsed = (last - first).total_seconds() / 3600
    if elapsed <= 0:
        raise DeckError("the observation window cannot support a duration: it holds a single capture period")
    if elapsed < 36:
        hours = round(elapsed)
        approx = "" if hours == elapsed else "Approximately "
        word = _number_word(hours)
        phrase = f"{approx}{word} {_plural(hours, 'hour')} of hourly observations"
        days = None
    else:
        days = int(elapsed / 24 + 0.5)
        approx = "" if days * 24 == elapsed else "Approximately "
        word = _number_word(days)
        phrase = f"{approx}{word} {_plural(days, 'day')} of hourly observations"
    phrase = phrase[0].upper() + phrase[1:]
    return ObservationWindow(first, last, elapsed, len(instants), days, phrase)


def _require_completed(result: object):  # type: ignore[no-untyped-def]
    from ql2_sixt_canada_analysis.final_report import FinalReportResult, ReportSection

    if not isinstance(result, FinalReportResult):
        raise TypeError("result must be a FinalReportResult")
    if not result.completed or not all(result.findings_valid(s) for s in ReportSection):
        codes = ", ".join(b.value for b in result.blockers) or "unavailable"
        raise DeckError(f"the deck needs a completed final report (blocked: {codes}); no deck is built")
    return result


def retained_share(retained: int, previous: int) -> tuple[float, str]:
    """``(ratio, display)`` of the pooled retained-product share (fails closed).

    Rounding rule: a whole percentage, rounded half up. When that would show
    ``100%`` or ``0%`` for a share that is not exactly one or zero, one decimal
    (half up) is shown instead; if even that would round to ``100.0%`` or
    ``0.0%``, the display is ``>99.9%`` or ``<0.1%``. Exact arithmetic on the
    counts, so the display never depends on binary floating point.

    Raises:
        DeckError: A non-integer, negative or zero denominator, or more retained than previous products.
    """
    from fractions import Fraction

    if (isinstance(retained, bool) or isinstance(previous, bool) or not isinstance(retained, int)
            or not isinstance(previous, int)):
        raise DeckError("retained and previous product counts must be integers")
    if previous <= 0:
        raise DeckError("the retained-product share is unavailable: no previous products (zero denominator)")
    if not 0 <= retained <= previous:
        raise DeckError("retained products must be between zero and the previous products")
    percent = Fraction(100 * retained, previous)

    def half_up(value: Fraction, decimals: int) -> Fraction:
        scale = 10 ** decimals
        return Fraction(int(value * scale + Fraction(1, 2)), scale)

    whole = half_up(percent, 0)
    if (whole == 100 and retained != previous) or (whole == 0 and retained != 0):
        tenth = half_up(percent, 1)
        if tenth == 100:
            display = ">99.9%"
        elif tenth == 0:
            display = "<0.1%"
        else:
            display = f"{float(tenth):.1f}%"
    else:
        display = f"{int(whole)}%"
    return retained / previous, display


def opening_conclusion(controls_clean: bool) -> str:
    """The opening slide's conclusion: decision-ready for monitoring only when no control triggered or went unassessed.

    The deck is built only from a completed (pricing-ready) report, so the one
    remaining condition is the monitoring outcome; a triggered or not-assessable
    control narrows the conclusion instead of keeping the decision-ready wording.
    """
    if controls_clean:
        return ("Decision-ready for descriptive monitoring; more history and operational evidence are needed "
                "before using it for pricing decisions.")
    return "Not yet decision-ready: contract-based controls flag defects to resolve before monitoring or pricing use."


def _city_name(city: str) -> str:
    return city.title()


def deck_results(result: object) -> dict[str, Any]:
    """Every value the deck shows, computed from one completed final report (pure; nothing is written).

    Raises:
        DeckError: The report is not completed or a value does not reconcile with the report.
    """
    from ql2_sixt_canada_analysis import final_report as fr
    from ql2_sixt_canada_analysis.assortment_presentation import _CAPTURE
    from ql2_sixt_canada_analysis.monitoring import ControlStatus, monitoring_control

    r = _require_completed(result)
    v = _Values()

    # ---------------------------------------------------------------- scope (slides 1 and 2)
    record = r.authority_record or "unavailable"
    v.text("authority_record", record, "FinalReportResult.authority_record", f"Pricing-authority record: {record}")
    v.text("readiness_status", "ready", "FinalReportResult.pricing_ready", "Pricing readiness: ready")
    matched = r.matched.report
    cities = list(matched.cities)
    v.count("city_count", len(cities), "matched report: approved cities")
    assortment = r.assortment.assortment.report
    v.count("location_count", len(assortment.locations), "visible-assortment report: canonical locations",
            f"Across {fr._n(len(assortment.locations), 'canonical location')}")
    summary = r.assortment.tables.location_summary
    v.count("hours_per_location", int(summary["scheduled_captures"].max()),
            "assortment location summary: scheduled_captures (maximum over locations)")
    v.count("governed_excluded_captures", int(summary["excluded_captures"].sum()),
            "assortment location summary: excluded_captures (sum over locations)")
    window = observation_window(r.assortment.tables.assortment_timeline[_CAPTURE])
    first, last = window.first, window.last
    v.add("window_start", f"{first:%Y%m%dT%H%M%SZ}", f"{first.day} {first:%b %Y, %H:%M} UTC", "date",
          "assortment timeline: first scheduled capture period")
    v.add("window_end", f"{last:%Y%m%dT%H%M%SZ}", f"{last.day} {last:%b %Y, %H:%M} UTC", "date",
          "assortment timeline: last scheduled capture period")
    same_month = (first.year, first.month) == (last.year, last.month)
    label = (f"{first.day}–{last.day} {last:%b %Y}" if same_month
             else f"{first.day} {first:%b %Y} – {last.day} {last:%b %Y}")
    v.text("window_label", label, "assortment timeline: first and last scheduled capture dates (UTC)")
    elapsed = window.elapsed_hours
    v.add("observation_hours", elapsed, f"{elapsed:,.0f}" if elapsed == int(elapsed) else f"{elapsed:,.1f}",
          "duration", "assortment timeline: hours from the first to the last scheduled capture period "
          "(endpoints are capture instants; no hour added)")
    v.count("observation_periods", window.capture_periods,
            "assortment timeline: distinct scheduled capture periods, both endpoints included")
    v.add("scope_duration", window.rounded_days if window.rounded_days is not None else elapsed, window.phrase,
          "scope", "observation_window(): elapsed hours rounded half up to whole days (exact hours under 36)")

    # Slide 1 scope facts: each display is generated from a report field, never typed in the template.
    city_word = _number_word(len(cities))
    v.add("scope_markets", len(cities),
          f"{city_word[0].upper()}{city_word[1:]} Canadian {_plural(len(cities), 'market')}", "scope",
          "matched report: approved cities (each approved city is one Canadian market)")
    pc = r.price_changes.analysis.report
    roles = {str(location.role).lower() for location in pc.locations}
    if roles == {"airport", "downtown"}:
        roles_phrase = "Airport and downtown locations"
    elif roles and roles <= {"airport", "downtown"}:
        roles_phrase = f"{sorted(roles)[0].title()} locations only"
    else:
        raise DeckError("the approved locations have no airport or downtown role")
    v.add("scope_locations", sorted(roles), roles_phrase, "scope",
          "price-change report: authority roles of the approved canonical locations")
    v.count("interval_count", pc.intervals, "price-change report: eligible one-hour location intervals",
            f"over {fr._n(pc.intervals, 'eligible one-hour location interval')}")
    controls = r.monitoring.report

    # ---------------------------------------------------------------- matched pricing (slide 3)
    if len(matched.price_units) != 1:
        raise DeckError("matched pricing must have exactly one currency and price basis")
    symbol, basis = matched.price_units[0]
    overall = matched.overall
    v.count("matched_pairs", overall.counts.matched, "matched report: overall.counts.matched")
    v.count("candidate_groups", overall.counts.candidate_groups, "matched report: overall.counts.candidate_groups")
    if overall.counts.match_rate is None:
        raise DeckError("the match rate is unavailable")
    v.add("match_rate", overall.counts.match_rate, f"{overall.counts.match_rate:.0%}", "share",
          "matched report: overall.counts.match_rate",
          f"{overall.counts.matched} matched pairs from {overall.counts.candidate_groups} candidate groups "
          f"(match rate {overall.counts.match_rate:.0%})")
    v.add("overall_positive_share", overall.signs.positive / overall.counts.matched,
          _whole_share(overall.signs.positive, overall.counts.matched), "share",
          "matched report: overall.signs.positive / overall.counts.matched")
    v.add("overall_median", overall.dollars.median, _money(overall.dollars.median, symbol, basis), "money",
          "matched report: overall.dollars.median")
    table = fr.matched_summary_table(r).set_index("city")
    medians = []
    for i, city in enumerate(cities, start=1):
        n = city.counts.matched
        if n == 0 or city.dollars.median is None or city.percent.median is None:
            raise DeckError("every approved city needs matched pairs for the deck")
        prefix = f"city{i}"
        v.text(f"{prefix}_name", _city_name(city.city), "matched report: city")
        v.count(f"{prefix}_pairs", n, f"matched report: {city.city} counts.matched")
        v.add(f"{prefix}_median", city.dollars.median, _money(city.dollars.median, symbol, basis), "money",
              f"matched report: {city.city} dollars.median",
              f"median premium {fr._money(city.dollars.median, matched.price_units)}")
        v.add(f"{prefix}_median_pct", city.percent.median, _pct(city.percent.median), "percent",
              f"matched report: {city.city} percent.median",
              f"({fr._signed(city.percent.median, '%')} of the downtown price)")
        v.add(f"{prefix}_higher_share", city.signs.positive / n, _whole_share(city.signs.positive, n), "share",
              f"matched report: {city.city} signs.positive / counts.matched",
              f"airport higher in {fr._share(city.signs.positive, n)} matched pairs")
        q25, q75 = float(table.loc[city.city, "percent_q25"]), float(table.loc[city.city, "percent_q75"])
        v.add(f"{prefix}_pct_iqr", [q25, q75], f"{_pct(q25, 2)} to {_pct(q75, 2)}", "percent",
              f"matched summary table: {city.city} percent_q25 and percent_q75")
        medians.append((city.dollars.median, i, city))
    top_median, top_index, top_city = max(medians, key=lambda t: t[0])
    v.text("top_city_name", _city_name(top_city.city), "matched report: city with the largest median dollar premium")
    v.text("top_city_median", v.items[f"city{top_index}_median"].display, f"city{top_index}_median")
    v.text("top_city_median_pct", v.items[f"city{top_index}_median_pct"].display, f"city{top_index}_median_pct")
    v.text("top_city_pct_iqr", v.items[f"city{top_index}_pct_iqr"].display, f"city{top_index}_pct_iqr")
    top_q25, top_q75 = v.items[f"city{top_index}_pct_iqr"].value
    if top_q75 - top_q25 < 1.0:
        v.text("top_city_spread_phrase", "almost constant across pairs",
               "derived: interquartile range of the top city's percentage premium is under one percentage point")
        v.text("top_city_spread_note",
               "an almost constant ratio, which looks like a systematic pricing rule rather than random variation",
               "derived: interquartile range of the top city's percentage premium is under one percentage point")
    else:
        v.text("top_city_spread_phrase", "with a wide spread across pairs",
               "derived: interquartile range of the top city's percentage premium is at least one percentage point")
        v.text("top_city_spread_note", "a wide spread, so the gap is not a single fixed ratio",
               "derived: interquartile range of the top city's percentage premium is at least one percentage point")
    all_positive = all(m > 0 for m, _, _ in medians)
    v.text("matched_headline",
           ("Airports list higher than downtown in every city — widest in " if all_positive
            else "Airport premiums differ in sign across cities — widest in ") + _city_name(top_city.city),
           "derived: sign of every city median dollar premium; city with the largest median")
    v.text("premium_sign_phrase", "higher in every city" if all_positive else "not higher in every city",
           "derived: sign of every city median dollar premium")

    # ---------------------------------------------------------------- price changes and assortment (slide 4)
    events = r.price_changes.analysis.events.report.overall
    classes = dict(pc.movement_classes)
    synchronized = classes.get("synchronized_increase", 0) + classes.get("synchronized_decrease", 0)
    p = pc.persistence
    censored = dict(p.not_testable_reasons).get("right_censored_final_capture", 0)
    if events.increase + events.decrease != pc.price_change_count:
        raise DeckError("increases and decreases do not add up to the price-change candidates")
    v.count("pc_total", pc.price_change_count, "price-change report: price_change_count",
            fr._n(pc.price_change_count, "price-change candidate"))
    v.count("pc_increases", events.increase, "price-event report: overall.increase", fr._n(events.increase, "increase"))
    v.count("pc_decreases", events.decrease, "price-event report: overall.decrease", fr._n(events.decrease, "decrease"))
    v.count("assortment_events", pc.assortment_event_count, "price-change report: assortment_event_count",
            fr._n(pc.assortment_event_count, "appearance or disappearance event"))
    v.count("sync_intervals", synchronized, "price-change report: synchronized_increase + synchronized_decrease",
            f"{fr._n(synchronized, 'interval')} direction-synchronized")
    v.count("with_following", p.with_following_interval, "persistence: with_following_interval",
            f"Of {fr._n(p.with_following_interval, 'change')} with a following eligible interval")
    v.count("held", p.held, "persistence: held", f"{p.held} held")
    v.count("continued", p.continued, "persistence: continued", f"{p.continued} continued")
    v.count("reverted", p.reverted, "persistence: reverted", f"{p.reverted} reverted")
    v.count("not_testable", p.not_testable, "persistence: not_testable", f"{p.not_testable} not testable")
    v.count("censored", censored, "persistence: right_censored_final_capture",
            f"({censored} right-censored at the final capture)")
    if p.with_following_interval + p.not_testable != pc.price_change_count:
        raise DeckError("persistence outcomes do not add up to the price-change candidates")
    direction = ("mostly decreases" if events.decrease > events.increase
                 else "mostly increases" if events.increase > events.decrease else "evenly split")
    table_events = r.price_changes.analysis.event_table
    changed_intervals = int(((table_events["increase"] + table_events["decrease"]) > 0).sum())
    v.count("changed_intervals", changed_intervals, "price-change event table: intervals with an increase or decrease")
    spread = "in a few intervals" if changed_intervals * 10 <= pc.intervals else "across many intervals"
    v.text("pc_spread_phrase", spread, "derived: intervals with a change, at most a tenth of eligible intervals")
    lead = {"mostly decreases": "Prices mostly fall", "mostly increases": "Prices mostly rise",
            "evenly split": "Price moves split evenly"}[direction]
    v.text("pc_direction_phrase", direction, "derived: price-event report overall.increase versus overall.decrease")
    v.text("pc_direction_label", direction[0].upper() + direction[1:],
           "derived: price-event report overall.increase versus overall.decrease (more decreases: mostly decreases; "
           "more increases: mostly increases; equal: evenly split)")
    c = assortment.overall
    if not c.previous_products:
        raise DeckError("pooled retention is unavailable (zero denominator)")
    v.add("retention", c.retained / c.previous_products, f"{c.retained / c.previous_products:.2f}", "ratio",
          "assortment report: overall.retained / overall.previous_products",
          f"pooled retention {c.retained / c.previous_products:.2f}")
    share, share_display = retained_share(c.retained, c.previous_products)
    v.add("retained_share", share, share_display, "retained_share",
          "assortment report: overall.retained / overall.previous_products, pooled over assessed intervals "
          "(retained_share(): whole percent, half up; one decimal when that would show 0% or 100% for an "
          "inexact share)")
    v.count("retained_products", c.retained, "assortment report: overall.retained")
    v.count("previous_products", c.previous_products, "assortment report: overall.previous_products")
    v.count("assessed_intervals", c.assessed_intervals, "assortment report: overall.assessed_intervals",
            fr._n(c.assessed_intervals, "consecutive one-hour interval"))
    v.count("additions", c.additions, "assortment report: overall.additions", fr._n(c.additions, "addition"))
    v.count("removals", c.removals, "assortment report: overall.removals", fr._n(c.removals, "removal"))
    v.count("drop_intervals", c.drop_intervals, "assortment report: overall.drop_intervals",
            f"Observed drops: {fr._n(c.drop_intervals, 'interval')}")
    v.text("drop_phrase", f"{c.drop_intervals:,} {_plural(c.drop_intervals, 'drop')}",
           "assortment report: overall.drop_intervals")
    v.text("drop_intervals_phrase", f"{c.drop_intervals:,} drop {_plural(c.drop_intervals, 'interval')}",
           "assortment report: overall.drop_intervals")
    v.count("coincident_intervals", c.coincident_intervals, "assortment report: overall.coincident_intervals",
            f"same location: {c.coincident_intervals}")
    drops = r.assortment.tables.cross_location_drops
    shared = int((drops["simultaneity"] != "isolated_in_extract").sum()) if len(drops) else 0
    v.count("shared_drop_periods", shared, "cross-location drop table: periods not isolated_in_extract")
    v.text("drop_sharing_phrase", "none shared across locations" if shared == 0
           else f"{shared:,} shared across locations", "derived: cross-location drop table simultaneity")
    v.text("change_headline",
           f"{lead}, {spread}; assortment stays "
           + ("stable" if c.retained / c.previous_products >= 0.95 else "volatile"),
           "derived: increase versus decrease counts, changed intervals; pooled retention of at least 0.95 "
           "reads as stable")
    v.text("policy_status", assortment.anomaly_policy_status.value, "assortment report: anomaly_policy_status")

    # ---------------------------------------------------------------- monitoring (slide 5)
    counts = {s: len(controls.controls_with_status(s)) for s in ControlStatus}
    overview = fr.monitoring_overview_table(r)
    v.count("control_count", len(overview), "monitoring overview table: rows",
            "Of the eight controls" if len(overview) == 8 else None)
    v.count("controls_triggered", counts[ControlStatus.TRIGGERED], "monitoring: controls triggered",
            f"{counts[ControlStatus.TRIGGERED]} triggered")
    v.count("controls_passed", counts[ControlStatus.PASSED], "monitoring: controls passed",
            f"{counts[ControlStatus.PASSED]} passed")
    v.count("controls_not_assessable", counts[ControlStatus.NOT_ASSESSABLE], "monitoring: controls not assessable",
            f"{counts[ControlStatus.NOT_ASSESSABLE]} not assessable")
    v.count("controls_candidate_only", counts[ControlStatus.CANDIDATE_ONLY], "monitoring: candidate-only controls",
            f"{counts[ControlStatus.CANDIDATE_ONLY]} candidate-only")
    v.count("controls_confirmation", counts[ControlStatus.CONFIRMATION_REQUIRED],
            "monitoring: confirmation-required controls",
            f"{counts[ControlStatus.CONFIRMATION_REQUIRED]} confirmation-required")
    if sum(counts.values()) != len(overview):
        raise DeckError("control statuses do not add up to the monitoring table")
    clean = counts[ControlStatus.TRIGGERED] == 0 and counts[ControlStatus.NOT_ASSESSABLE] == 0
    v.text("monitoring_headline",
           "No contract-based control triggered; the open risks need evidence the extract does not hold"
           if clean else "Contract-based controls flag defects to raise with the collection owner",
           "derived: triggered and not-assessable control counts")
    v.text("conclusion", opening_conclusion(clean),
           "derived: readiness status and triggered or not-assessable controls")
    v.text("candidate_phrase", f"{counts[ControlStatus.CANDIDATE_ONLY]:,} candidate "
           + _plural(counts[ControlStatus.CANDIDATE_ONLY], "rule"), "monitoring: candidate-only controls")
    v.text("confirmation_phrase", f"{counts[ControlStatus.CONFIRMATION_REQUIRED]:,} "
           + _plural(counts[ControlStatus.CONFIRMATION_REQUIRED], "control"),
           "monitoring: confirmation-required controls")

    monitoring_rows = []
    for row in overview.to_dict("records"):
        definition = monitoring_control(row["control_id"])
        monitoring_rows.append({"control_id": row["control_id"], "name": definition.name,
                                "severity": row["effective_severity"], "status": row["evaluation_status"],
                                "calibration": row["calibration_status"]})
    city_rows = [{"index": i, "city": _city_name(c.city)} for i, c in enumerate(cities, start=1)]
    return {"schema": _SCHEMA, "authority_record": r.authority_record, "slide_count": DECK_SLIDE_COUNT,
            "values": {k: {"value": x.value, "display": x.display, "kind": x.kind, "source": x.source,
                           "trace": x.trace} for k, x in v.items.items()},
            "cities": city_rows, "monitoring_controls": monitoring_rows}


def verify_traces(results: Mapping[str, Any], text: str) -> tuple[str, ...]:
    """Value keys whose ``trace`` text is missing from ``text`` (empty when every traced value is present).

    ``text`` is either the in-memory interpretation and status text of the run
    or the printed output of an executed ``06_final_report.ipynb`` copy.
    """
    return tuple(k for k, item in results["values"].items() if item.get("trace") and item["trace"] not in text)


def _report_text(result: object) -> str:
    from ql2_sixt_canada_analysis import final_report as fr

    parts = list(fr.final_status_lines(result)) + [fr.interpret_section(result, s) for s in fr.ReportSection]
    return "\n".join(parts)


# ============================================================================ figures


def _figure_metadata(fig) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Titles, axis labels, tick labels, legends and colour-bar labels of a drawn figure."""
    fig.canvas.draw()
    data_axes, colorbars = [], []
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            colorbars.append(ax.get_ylabel() or ax.get_xlabel())
            continue
        data_axes.append({
            "title": ax.get_title(loc="left") or ax.get_title(),
            "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "xticklabels": [t.get_text() for t in ax.get_xticklabels() if t.get_text()],
            "yticklabels": [t.get_text() for t in ax.get_yticklabels() if t.get_text()],
        })
    legends = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
    for ax in fig.axes:
        if ax.get_legend() is not None:
            legends += [t.get_text() for t in ax.get_legend().get_texts()]
    supx = fig._supxlabel.get_text() if getattr(fig, "_supxlabel", None) is not None else ""
    title = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) is not None else ""
    return {"title": title, "axes": data_axes, "colorbar_labels": colorbars, "legend": legends, "note": supx}


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise DeckValidationError(f"{path.name} is not a valid PNG file")
    return struct.unpack(">II", head[16:24])


def _check_labels(name: str, meta: Mapping[str, Any]) -> list[str]:
    """Label and unit problems of one figure against :data:`DECK_FIGURES` (empty when compliant)."""
    rules = DECK_FIGURES[name]["required"]
    problems = []
    if not re.search(rules["title"], meta["title"], re.IGNORECASE):
        problems.append(f"{name}: title does not describe the measure")
    titles = [a["title"] for a in meta["axes"]]
    for pattern, title in zip(rules["axes_titles"], titles, strict=False):
        if not re.search(pattern, title):
            problems.append(f"{name}: panel title {title!r} does not match {pattern!r}")
    if len(titles) != len(rules["axes_titles"]):
        problems.append(f"{name}: expected {len(rules['axes_titles'])} panels")
    for key, field_name in (("ylabels", "ylabel"), ("xlabels", "xlabel")):
        labelled = [a[field_name] for a in meta["axes"] if a[field_name]]
        for pattern in rules.get(key, ()):
            if not any(re.search(pattern, label) for label in labelled):
                problems.append(f"{name}: no {field_name} with unit {pattern!r}")
    for pattern, label in zip(rules.get("colorbar_labels", ()), meta["colorbar_labels"], strict=False):
        if not re.search(pattern, label):
            problems.append(f"{name}: colour bar label {label!r} has no unit")
    if len(meta["colorbar_labels"]) != len(rules.get("colorbar_labels", ())):
        problems.append(f"{name}: unexpected number of colour bars")
    for pattern in rules.get("legend", ()):
        if not any(re.search(pattern, label) for label in meta["legend"]):
            problems.append(f"{name}: legend entry {pattern!r} missing")
    for axis in meta["axes"]:
        if not axis["ylabel"] and not axis["yticklabels"]:
            problems.append(f"{name}: an axis has neither a label nor category labels")
        if any(label.endswith("…") or label.endswith("...") for label in axis["xticklabels"] + axis["yticklabels"]):
            problems.append(f"{name}: a category label is truncated")
    return problems


def _render_figures(result: object, directory: Path) -> dict[str, Any]:
    """Export each deck figure with its authoritative plotting function; return its metadata."""
    from ql2_sixt_canada_analysis.matched_location_pricing import plot_matched_location_premiums
    from ql2_sixt_canada_analysis.price_change_analysis import plot_price_change_heatmap
    from ql2_sixt_canada_analysis.price_change_presentation import build_presentation_tables

    r = _require_completed(result)
    analysis = r.price_changes.analysis
    build_presentation_tables(analysis)            # every price-change reconciliation gate before rendering
    makers = {
        "matched_premiums": lambda spec: plot_matched_location_premiums(
            r.matched, figsize=spec["figsize"], dpi=_FIGURE_DPI, font_scale=spec["font_scale"])[0],
        "price_change_heatmap": lambda spec: plot_price_change_heatmap(
            analysis, dpi=_FIGURE_DPI, figsize=spec["figsize"], font_scale=spec["font_scale"],
            wording="audience")[0],
    }
    out: dict[str, Any] = {}
    for name, spec in DECK_FIGURES.items():
        fig = makers[name](spec)
        try:
            path = directory / spec["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, format="png", dpi=_FIGURE_DPI, metadata={"Software": None}, facecolor="white")
            meta = _figure_metadata(fig)
        finally:
            fig.clear()
        width, height = _png_size(path)
        problems = _check_labels(name, meta)
        if problems:
            raise DeckError("; ".join(problems))
        out[name] = {"file": spec["file"], "width": width, "height": height, **meta}
    return out


def _alt_texts(results: Mapping[str, Any]) -> dict[str, str]:
    values = results["values"]

    def d(key: str) -> str:
        return values[key]["display"]

    cities = "; ".join(
        "{} median {} ({}), {} pairs".format(d(f"city{c['index']}_name"), d(f"city{c['index']}_median"),
                                             d(f"city{c['index']}_median_pct"), d(f"city{c['index']}_pairs"))
        for c in results["cities"])
    matched = ("Two-panel chart of matched airport-minus-downtown listed-price premiums by city. Left panel: "
               "dollar premium per rental day; right panel: premium as a percentage of the downtown price. "
               f"Boxes show the interquartile range and median, whiskers the full range, points every pair. {cities}.")
    heatmap = ("Heatmap of observed price moves. Rows are the standardized (canonical) locations; columns are "
               "eligible hourly comparisons in UTC. The top panel counts price increases "
               f"({values['pc_increases']['display']} in total) and the bottom panel price decreases "
               f"({values['pc_decreases']['display']} in total). Darker cells mean more price moves in that hour; "
               "pale cells are comparable hours with no change; hatched grey cells have no comparable prior "
               "capture and are not zero activity.")
    return {"matched_alt": matched, "heatmap_alt": heatmap}


# ============================================================================ rendering


def _span(key: str, display: str) -> str:
    return f'<span class="v" data-value="{key}">{html.escape(display)}</span>'


def _fragments(results: Mapping[str, Any]) -> dict[str, str]:
    """Repeated blocks generated from the results (one card per city, one row per control)."""
    values = results["values"]

    def s(key: str) -> str:
        return _span(key, values[key]["display"])

    cards = []
    for c in results["cities"]:
        p = f"city{c['index']}"
        cards.append(
            f'<li class="city-card" data-city-index="{c["index"]}">'
            f'<div class="cc-head"><h3>{s(p + "_name")}</h3><p class="big">{s(p + "_median")}</p></div>'
            f'<p>median premium, {s(p + "_median_pct")} of downtown; airport higher in '
            f'{s(p + "_higher_share")} of {s(p + "_pairs")} pairs</p></li>')
    return {"city_cards": "\n".join(cards), "monitoring_notes_table": _monitoring_notes_table(results)}


#: Inline styles for the speaker-notes control table: the speaker view does not load deck.css.
_NOTES_TABLE_STYLE = ("border-collapse: collapse; width: 100%; table-layout: auto; font-size: 0.8em; "
                      "margin: 0.4em 0 0.8em;")
_NOTES_CELL_STYLE = ("text-align: left; vertical-align: top; padding: 0.2em 0.5em 0.2em 0; "
                     "border-bottom: 1px solid #ccc; overflow-wrap: break-word;")


def _words(code: str) -> str:
    return code.replace("_", " ")


def _monitoring_notes_table(results: Mapping[str, Any]) -> str:
    """The complete monitoring-control table for the speaker notes, in authoritative order (one row per control)."""
    head = "".join(f'<th scope="col" style="{_NOTES_CELL_STYLE}">{label}</th>'
                   for label in ("Control", "Effective severity", "Evaluation status", "Calibration status"))
    rows = []
    for row in results["monitoring_controls"]:
        cells = "".join(f'<td style="{_NOTES_CELL_STYLE}">{html.escape(_words(row[k]))}</td>'
                        for k in ("severity", "status", "calibration"))
        rows.append(f'<tr data-control-id="{html.escape(row["control_id"])}">'
                    f'<th scope="row" style="{_NOTES_CELL_STYLE}">{html.escape(row["name"])}</th>{cells}</tr>')
    return (f'<table class="notes-controls" style="{_NOTES_TABLE_STYLE}">'
            f"<caption>Monitoring controls evaluated on this run</caption>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def render_deck_html(template: str, results: Mapping[str, Any]) -> str:
    """The deck HTML: placeholders replaced from ``results`` (fails on any unknown or unresolved field).

    ``{{ key }}`` becomes a ``data-value`` span, ``{{= key }}`` escaped plain text
    (attributes and the document title) and ``{{> name }}`` a generated block.
    """
    values = results["values"]
    plain = {**{k: item["display"] for k, item in values.items()}, **_alt_texts(results)}
    for name, meta in results.get("figures", {}).items():
        plain[f"{name}_width"], plain[f"{name}_height"] = str(meta["width"]), str(meta["height"])
    fragments = _fragments(results)

    def replace(match: re.Match[str]) -> str:
        mode, key = match.groups()
        if mode == ">":
            if key not in fragments:
                raise DeckError(f"unknown deck fragment {key}")
            return fragments[key]
        if mode == "=":
            if key not in plain:
                raise DeckError(f"unknown deck value {key}")
            return html.escape(plain[key], quote=True)
        if key not in values:
            raise DeckError(f"unknown deck value {key}")
        return _span(key, values[key]["display"])

    rendered = _PLACEHOLDER.sub(replace, template)
    if _UNRESOLVED.search(rendered):
        raise DeckError("the rendered deck contains an unresolved template field")
    return rendered


def _template_text(source_dir: Path) -> str:
    return (source_dir / _TEMPLATE).read_text(encoding="utf-8")


def build_deck(result: object, output_dir: str | Path = DECK_OUTPUT_DIR, *,
               source_dir: str | Path = DECK_SOURCE_DIR) -> Path:
    """Export the figures, write the results artifact and render the deck into ``output_dir``.

    The directory is replaced as a whole (only the deck's own files are written
    into it), so no stale figure survives a rebuild. Returns the ``index.html`` path.

    Raises:
        DeckError: The report is not completed, a traced value disagrees with the
            report text, or a figure lacks a required label or unit.
    """
    source = Path(source_dir)
    target = Path(output_dir)
    results = deck_results(result)
    missing = verify_traces(results, _report_text(result))
    if missing:
        raise DeckError("deck values disagree with the final-report text: " + ", ".join(missing))
    if not target.parent.exists():
        raise FileNotFoundError("the output directory's parent must already exist")
    _refuse_source_data(target)
    staging = target.with_name(f".{target.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        results["figures"] = _render_figures(result, staging)
        for name in _STATIC:
            shutil.copy2(source / name, staging / name)
        shutil.copytree(source / _VENDOR, staging / _VENDOR)
        (staging / DECK_RESULTS_FILENAME).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")
        (staging / "index.html").write_text(render_deck_html(_template_text(source), results), encoding="utf-8")
        validate_deck(staging, source_dir=source)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target / "index.html"


def _refuse_source_data(target: Path) -> None:
    from ql2_sixt_canada_analysis.paths import DATA_DIR

    resolved = target.resolve()
    data = DATA_DIR.resolve()
    if resolved == data or data in resolved.parents:
        raise DeckError("the deck cannot be written into the data directory")


# ============================================================================ validation


class _DeckParser(HTMLParser):
    """Collects slides, notes, value spans, images and asset references of the deck."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.section_depth = 0
        self.in_slides = 0
        self.slides: list[dict[str, Any]] = []
        self.nested_sections = 0
        self.refs: list[str] = []
        self.images: list[dict[str, str]] = []
        self.spans: list[tuple[str, list[str]]] = []
        self.span_depth: list[int] = []
        self.notes_depth = 0
        self.title = ""
        self.in_title = False
        self.headings: list[str] = []
        self.text: list[str] = []
        self.inline_script: list[str] = []
        self.in_script = False
        self.lang = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "html":
            self.lang = a.get("lang", "")
        if tag == "title":
            self.in_title = True
        if tag == "script":
            self.in_script = True
            if a.get("src"):
                self.refs.append(a["src"])
        if tag == "link" and a.get("href"):
            self.refs.append(a["href"])
        if tag == "img":
            self.refs.append(a.get("src", ""))
            self.images.append(a)
        if tag in ("a",) and a.get("href") and not a["href"].startswith("#"):
            self.refs.append(a["href"])
        if "slides" in a.get("class", "").split() and tag == "div":
            self.in_slides = len(self.stack) + 1
        if tag == "section" and self.in_slides:
            if self.section_depth:
                self.nested_sections += 1
            else:
                self.slides.append({"id": a.get("id", ""), "attrs": a, "notes": "", "text": "", "headings": []})
            self.section_depth += 1
        if tag == "aside" and "notes" in a.get("class", "").split() and self.slides:
            self.notes_depth = 1
        elif self.notes_depth and tag not in _VOID:
            self.notes_depth += 1
        if tag == "span" and "data-value" in a:
            self.spans.append((a["data-value"], []))
            self.span_depth.append(1)
        elif self.span_depth and tag not in _VOID:
            self.span_depth[-1] += 1
        if tag in ("h1", "h2") and self.slides:
            self.slides[-1]["headings"].append("")
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "script":
            self.in_script = False
        if tag == "section" and self.section_depth:
            self.section_depth -= 1
        if self.notes_depth:
            self.notes_depth -= 1
        if self.span_depth:
            self.span_depth[-1] -= 1
            if self.span_depth[-1] == 0:
                self.span_depth.pop()
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
            if self.in_slides and len(self.stack) < self.in_slides:
                self.in_slides = 0

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
        if self.in_script:
            self.inline_script.append(data)
            return
        self.text.append(data)
        if self.span_depth:
            self.spans[-1][1].append(data)
        if self.slides and self.section_depth:
            slide = self.slides[-1]
            if self.notes_depth:
                slide["notes"] += data
            else:
                slide["text"] += data
                if {"h1", "h2"} & set(self.stack) and slide["headings"]:
                    slide["headings"][-1] += data


_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track",
                   "wbr"})


def _parse(page: str) -> _DeckParser:
    parser = _DeckParser()
    parser.feed(page)
    parser.close()
    return parser


def _local_asset(directory: Path, ref: str) -> Path:
    return (directory / ref.split("#", 1)[0].split("?", 1)[0]).resolve()


def validate_deck(output_dir: str | Path = DECK_OUTPUT_DIR, result: object = None, *,
                  source_dir: str | Path = DECK_SOURCE_DIR) -> dict[str, Any]:
    """Check a built deck; return its parsed results (raises :class:`DeckValidationError` on any problem).

    With ``result`` (a completed final report) the stored results must also
    equal a fresh computation from that run, which detects stale numbers.
    """
    directory = Path(output_dir)
    problems: list[str] = []
    page_path = directory / "index.html"
    results_path = directory / DECK_RESULTS_FILENAME
    if not page_path.is_file() or not results_path.is_file():
        raise DeckValidationError("the deck has not been built (index.html or the results artifact is missing)")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    page = page_path.read_text(encoding="utf-8")
    if results.get("schema") != _SCHEMA:
        problems.append("the results artifact has an unexpected schema")

    # The page must be exactly the template rendered from the stored results.
    try:
        if render_deck_html(_template_text(Path(source_dir)), results) != page:
            problems.append("index.html is stale: it differs from the template rendered from the stored results")
    except DeckError as error:
        problems.append(f"the stored results do not render: {error}")
    if result is not None:
        fresh = deck_results(result)
        stored = {k: v for k, v in results.items() if k != "figures"}
        if json.loads(json.dumps(fresh)) != stored:
            changed = sorted(k for k in set(fresh["values"]) | set(stored.get("values", {}))
                             if json.loads(json.dumps(fresh["values"].get(k))) != stored.get("values", {}).get(k))
            changed += sorted(k for k in set(fresh) | set(stored)
                              if k != "values" and json.loads(json.dumps(fresh.get(k))) != stored.get(k))
            problems.append("the stored results are stale for this run: " + (", ".join(changed) or "structure"))

    parsed = _parse(page)
    if parsed.lang != "en":
        problems.append("the document has no language")
    if not parsed.title.strip() or _PLACEHOLDER.search(parsed.title):
        problems.append("the document has no informative title")
    if len(parsed.slides) != DECK_SLIDE_COUNT:
        problems.append(f"expected exactly {DECK_SLIDE_COUNT} primary slides, found {len(parsed.slides)}")
    if parsed.nested_sections:
        problems.append("vertical (nested) slides are not allowed")
    for number, slide in enumerate(parsed.slides, start=1):
        words = len(slide["notes"].split())
        if words < _MIN_NOTE_WORDS:
            problems.append(f"slide {number} speaker notes are missing or too short ({words} words)")
        if not any(h.strip() for h in slide["headings"]):
            problems.append(f"slide {number} has no heading")
        if not slide["id"]:
            problems.append(f"slide {number} has no id")
    ids = [s["id"] for s in parsed.slides]
    if len(set(ids)) != len(ids):
        problems.append("slide ids are not unique")

    values = results.get("values", {})
    for key, parts in parsed.spans:
        text = "".join(parts)
        if key not in values:
            problems.append(f"span {key} has no value in the results artifact")
        elif text != values[key]["display"]:
            problems.append(f"span {key} shows {text!r}, the results artifact says {values[key]['display']!r}")

    visible = " ".join(parsed.text)
    if _FORBIDDEN_TEXT.search(visible):
        problems.append("the deck contains placeholder wording (TODO, TBD, lorem ipsum, ...)")
    if _FORBIDDEN_CLAIMS.search(visible):
        problems.append("the deck makes a claim the evidence cannot support: "
                        + _FORBIDDEN_CLAIMS.search(visible).group(0))
    if _UNRESOLVED.search(visible):
        problems.append("the deck contains an unresolved template field")

    for ref in parsed.refs:
        if not ref or _REMOTE.match(ref) or ref.startswith("/") or "\\" in ref:
            problems.append(f"asset reference {ref!r} is not a relative local path")
            continue
        if _MACHINE_PATH.search(" " + ref) or ".." in Path(ref).parts:
            problems.append(f"asset reference {ref!r} is machine-specific or leaves the deck directory")
            continue
        path = _local_asset(directory, ref)
        if not path.is_file():
            problems.append(f"asset {ref!r} does not exist")
    if _MACHINE_PATH.search(page):
        problems.append("the deck contains a machine-specific path")
    for image in parsed.images:
        if len(image.get("alt", "").split()) < 8:
            problems.append(f"image {image.get('src')!r} has no meaningful alternative text")

    script = (directory / "deck.js").read_text(encoding="utf-8") if (directory / "deck.js").is_file() else ""
    if "Reveal.initialize" not in script:
        problems.append("reveal.js is not initialized")
    if not re.search(r"slideNumber\s*:\s*['\"]c/t['\"]", script):
        problems.append("slide numbers are not enabled")
    if not re.search(r"plugins\s*:\s*\[[^\]]*RevealNotes", script):
        problems.append("the notes plugin is not configured")
    refs = set(parsed.refs)
    for required in ("vendor/reveal.js/reveal.js", "vendor/reveal.js/plugin/notes/notes.js",
                     "vendor/reveal.js/reveal.css", "deck.css", "deck.js"):
        if required not in refs:
            problems.append(f"the deck does not load {required}")

    figures = results.get("figures", {})
    if set(figures) != set(DECK_FIGURES):
        problems.append("the results artifact does not list every deck figure")
    for name, meta in figures.items():
        path = directory / meta["file"]
        if meta["file"] not in refs:
            problems.append(f"figure {name} is not shown on any slide")
        if not path.is_file():
            problems.append(f"figure {name} is missing")
            continue
        try:
            width, height = _png_size(path)
        except DeckValidationError as error:
            problems.append(str(error))
            continue
        if (width, height) != (meta["width"], meta["height"]) or width < 1200 or height < 600:
            problems.append(f"figure {name} has unexpected or too small dimensions ({width}x{height})")
        if name in DECK_FIGURES:
            problems += _check_labels(name, meta)
    expected = {meta["file"] for meta in figures.values()}
    on_disk = {p.relative_to(directory).as_posix() for p in (directory / "figures").glob("*")} \
        if (directory / "figures").is_dir() else set()
    if on_disk - expected:
        problems.append("the figures directory holds files the deck does not use: "
                        + ", ".join(sorted(on_disk - expected)))
    if problems:
        raise DeckValidationError("\n".join(problems))
    return results


# ============================================================================ notebook verification


def notebook_output_text(notebook) -> str:  # type: ignore[no-untyped-def]
    """Printed and displayed text of an executed notebook (stream and ``text/plain`` outputs)."""
    parts = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                parts.append(output.get("text", ""))
            else:
                parts.append(output.get("data", {}).get("text/plain", ""))
    return "\n".join(parts)


def verify_against_final_notebook(output_dir: str | Path = DECK_OUTPUT_DIR, *, timeout_seconds: int = 900
                                  ) -> tuple[str, ...]:
    """Execute a clean copy of ``06_final_report.ipynb`` and return the traced values its output lacks."""
    from tempfile import TemporaryDirectory

    from ql2_sixt_canada_analysis.notebook_validation import execute_notebook_copy

    results = json.loads((Path(output_dir) / DECK_RESULTS_FILENAME).read_text(encoding="utf-8"))
    with TemporaryDirectory() as workdir:
        executed = execute_notebook_copy(PROJECT_ROOT / "notebooks" / "06_final_report.ipynb", workdir=workdir,
                                         timeout_seconds=timeout_seconds)
    return verify_traces(results, notebook_output_text(executed.executed))


# ============================================================================ command line


def main(argv: Sequence[str] | None = None) -> int:
    """``build``, ``validate`` or ``verify-notebook`` the deck (exit status 0 on success)."""
    parser = argparse.ArgumentParser(prog="python -m ql2_sixt_canada_analysis.presentation_deck",
                                     description="Build and check the six-slide reveal.js deck.")
    parser.add_argument("command", choices=("build", "validate", "verify-notebook"))
    parser.add_argument("--output-dir", default=None, help="deck directory (default: reports/presentation)")
    parser.add_argument("--raw-dir", default=None, help="raw-data directory (default: the standard resolution)")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else DECK_OUTPUT_DIR
    from ql2_sixt_canada_analysis.final_report import run_final_report
    from ql2_sixt_canada_analysis.paths import resolve_raw_data_dir

    try:
        if args.command == "build":
            report = run_final_report(resolve_raw_data_dir(args.raw_dir))
            page = build_deck(report, output_dir)
            print(f"Deck built: {DECK_SLIDE_COUNT} slides, {len(DECK_FIGURES)} figures, results artifact "
                  f"{DECK_RESULTS_FILENAME} ({_display_path(page)}).")
        elif args.command == "validate":
            report = run_final_report(resolve_raw_data_dir(args.raw_dir))
            results = validate_deck(output_dir, report)
            print(f"Deck valid: {DECK_SLIDE_COUNT} slides, {len(results['values'])} values match the current run.")
        else:
            missing = verify_against_final_notebook(output_dir)
            if missing:
                print("Values missing from the executed final-report notebook: " + ", ".join(missing))
                return 1
            traced = sum(1 for v in json.loads((output_dir / DECK_RESULTS_FILENAME).read_text())["values"].values()
                         if v.get("trace"))
            print(f"Every traced deck value ({traced}) appears in the executed final-report notebook output.")
    except (DeckError, DeckValidationError) as error:
        print(f"Deck {args.command} failed: {error}", file=sys.stderr)
        return 1
    return 0


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
