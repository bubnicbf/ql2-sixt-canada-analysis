"""Presentation QA: the six-slide reveal.js deck, its exported figures and its results artifact.

Synthetic data only: the deck is built from the same fabricated, pricing-ready
pipeline run the final-report tests use, into ``tmp_path``. The proprietary
files are never read and nothing is written into the repository.

An optional headless-browser smoke test runs when Playwright and a Chromium
build are installed (they are not project dependencies); otherwise it skips.
"""

from __future__ import annotations

import dataclasses
import functools
import html as html_module
import http.server
import json
import os
import re
import shutil
import subprocess
import threading
from html.parser import HTMLParser
from pathlib import Path

import nbformat
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from test_final_documentation import ready_run

from ql2_sixt_canada_analysis import final_report as fr
from ql2_sixt_canada_analysis import presentation_deck as pd_
from ql2_sixt_canada_analysis.paths import PROJECT_ROOT

TEMPLATE = PROJECT_ROOT / "presentation" / "deck.html"
VENDOR = PROJECT_ROOT / "presentation" / "vendor" / "reveal.js"
SCREENSHOT_DIR_ENV_VAR = "QL2_SIXT_DECK_SCREENSHOT_DIR"


@pytest.fixture(scope="module")
def ready_report() -> fr.FinalReportResult:
    return fr.final_report_from_pipeline(ready_run())


@pytest.fixture(scope="module")
def deck_dir(ready_report: fr.FinalReportResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("deck") / "presentation"
    page = pd_.build_deck(ready_report, target)
    assert page == target / "index.html"
    return target


@pytest.fixture
def deck_copy(deck_dir: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "presentation"
    shutil.copytree(deck_dir, copy)
    return copy


def _results(directory: Path) -> dict:
    return json.loads((directory / pd_.DECK_RESULTS_FILENAME).read_text(encoding="utf-8"))


class _Balance(HTMLParser):
    """Strict-enough well-formedness check: balanced non-void tags and unique ids."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    OPTIONAL_END = {"p", "li"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.ids: list[str] = []
        self.doctype = ""

    def handle_decl(self, decl: str) -> None:
        self.doctype = decl

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.ids += [v for k, v in attrs if k == "id" and v]
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        while self.stack and self.stack[-1] != tag and self.stack[-1] in self.OPTIONAL_END:
            self.stack.pop()
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}>")
            return
        self.stack.pop()


def _template_text_without_placeholders() -> str:
    """Visible and notes text of the template (tags, comments, head, scripts and placeholders removed)."""
    body = TEMPLATE.read_text(encoding="utf-8").split("<body>", 1)[1].split("<script", 1)[0]
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    body = re.sub(r"\{\{.*?\}\}", " ", body)
    return re.sub(r"<[^>]+>", " ", body)


# ============================================================================ build and structure


def test_build_writes_a_self_contained_deck(deck_dir: Path) -> None:
    files = {p.relative_to(deck_dir).as_posix() for p in deck_dir.rglob("*") if p.is_file()}
    assert files == {"index.html", "deck.css", "deck.js", pd_.DECK_RESULTS_FILENAME,
                     "figures/matched_premiums.png", "figures/price_change_heatmap.png",
                     "vendor/reveal.js/reveal.js", "vendor/reveal.js/reveal.css", "vendor/reveal.js/reset.css",
                     "vendor/reveal.js/plugin/notes/notes.js", "vendor/reveal.js/LICENSE"}
    assert not list(deck_dir.parent.glob(".*.building")), "the staging directory is removed"


def test_entry_point_is_well_formed_html(deck_dir: Path) -> None:
    page = (deck_dir / "index.html").read_text(encoding="utf-8")
    parser = _Balance()
    parser.feed(page)
    parser.close()
    assert parser.doctype.lower() == "doctype html"
    assert not parser.errors and not [t for t in parser.stack if t not in _Balance.OPTIONAL_END], parser.errors
    assert len(parser.ids) == len(set(parser.ids)), "ids are unique"
    assert '<meta charset="utf-8">' in page and 'name="viewport"' in page and '<html lang="en">' in page
    title = re.search(r"<title>(.*?)</title>", page).group(1)
    assert 2 <= len(title.split()) <= 6 and "{{" not in title


def test_validate_accepts_the_built_deck_and_matches_the_run(deck_dir: Path, ready_report) -> None:
    results = pd_.validate_deck(deck_dir, ready_report)
    assert results["slide_count"] == pd_.DECK_SLIDE_COUNT == 6
    assert set(results["figures"]) == set(pd_.DECK_FIGURES)


def test_exactly_six_primary_slides_without_vertical_children(deck_dir: Path) -> None:
    parsed = pd_._parse((deck_dir / "index.html").read_text(encoding="utf-8"))
    assert len(parsed.slides) == 6 and parsed.nested_sections == 0
    assert [s["id"] for s in parsed.slides] == ["question", "approach", "pricing", "changes", "limits", "conclusions"]
    template = TEMPLATE.read_text(encoding="utf-8")
    assert template.count("<section") == 6 and template.count("</section>") == 6


def test_every_primary_slide_has_meaningful_speaker_notes(deck_dir: Path) -> None:
    parsed = pd_._parse((deck_dir / "index.html").read_text(encoding="utf-8"))
    for slide in parsed.slides:
        notes = slide["notes"]
        assert len(notes.split()) >= 60, slide["id"]
        assert "Purpose." in notes and "Say." in notes, slide["id"]
        visible = set(re.findall(r"[a-z]{5,}", slide["text"].lower()))
        spoken = set(re.findall(r"[a-z]{5,}", notes.lower()))
        assert len(spoken - visible) >= 25, f"{slide['id']}: notes must add more than the visible text"
    assert all(s["notes"] for s in parsed.slides[:-1])
    assert all("Transition." in s["notes"] for s in parsed.slides[:-1])


def test_reveal_is_initialized_with_slide_numbers_and_the_notes_plugin(deck_dir: Path) -> None:
    script = (deck_dir / "deck.js").read_text(encoding="utf-8")
    assert "Reveal.initialize(" in script
    assert re.search(r'slideNumber:\s*"c/t"', script) and re.search(r"plugins:\s*\[RevealNotes\]", script)
    assert 'navigationMode: "linear"' in script
    page = (deck_dir / "index.html").read_text(encoding="utf-8")
    order = [page.index(s) for s in ('src="vendor/reveal.js/reveal.js"', 'src="vendor/reveal.js/plugin/notes/notes.js"',
                                     'src="deck.js"')]
    assert order == sorted(order), "reveal.js, then the notes plugin, then the initialisation"
    assert "reveal.js 5.1.0" in (VENDOR / "reveal.js").read_text(encoding="utf-8")[:200]
    assert "RevealNotes" in (VENDOR / "plugin" / "notes" / "notes.js").read_text(encoding="utf-8")
    assert "MIT" in (VENDOR / "LICENSE").read_text(encoding="utf-8") or "Permission is hereby granted" in \
        (VENDOR / "LICENSE").read_text(encoding="utf-8")


def test_local_assets_exist_and_no_reference_is_remote_or_machine_specific(deck_dir: Path) -> None:
    page = (deck_dir / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"]+)"', page)
    assert refs and all(not re.match(r"^(?:[a-z]+:|//|/)", r) and ".." not in r for r in refs), refs
    assert all((deck_dir / r).is_file() for r in refs)
    css = (deck_dir / "deck.css").read_text(encoding="utf-8")
    assert "@import" not in css and "url(" not in css, "no stylesheet downloads (fonts or images)"
    for text in (page, css, (deck_dir / "deck.js").read_text(encoding="utf-8"),
                 (deck_dir / pd_.DECK_RESULTS_FILENAME).read_text(encoding="utf-8")):
        assert not re.search(r"(?:/(?:Users|home|sessions|tmp|mnt|private|var|root)/|[A-Za-z]:\\)", text)
        assert "http://" not in text and "https://" not in text.replace("https://revealjs.com", "")


def test_images_have_alternative_text_and_dimensions(deck_dir: Path) -> None:
    page = (deck_dir / "index.html").read_text(encoding="utf-8")
    images = re.findall(r"<img ([^>]+)>", page)
    assert len(images) == len(pd_.DECK_FIGURES)
    results = _results(deck_dir)
    for attrs in images:
        alt = re.search(r'alt="([^"]+)"', attrs).group(1)
        src = re.search(r'src="([^"]+)"', attrs).group(1)
        meta = next(m for m in results["figures"].values() if m["file"] == src)
        assert len(alt.split()) >= 25 and "{{" not in alt
        assert f'width="{meta["width"]}"' in attrs and f'height="{meta["height"]}"' in attrs


def test_no_placeholder_text_or_unsupported_claims(deck_dir: Path) -> None:
    page = (deck_dir / "index.html").read_text(encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", page.split("<body>", 1)[1].split("<script", 1)[0])
    assert not re.search(r"\b(TODO|TBD|FIXME|lorem ipsum)\b", text, re.IGNORECASE)
    assert not re.search(r"\{\{|\}\}|\$\{", page)
    assert not pd_._FORBIDDEN_CLAIMS.search(text)
    assert pd_._FORBIDDEN_CLAIMS.pattern == pd_._FORBIDDEN_CLAIMS.pattern  # mirrors the final-report rule set
    from test_final_documentation import FORBIDDEN_CLAIMS

    assert not FORBIDDEN_CLAIMS.search(text)


def test_template_types_no_quantity(deck_dir: Path) -> None:
    text = _template_text_without_placeholders().replace("QL2", "QL")      # the company name is not a quantity
    assert not re.search(r"\d", text), "every number on a slide or in the notes must come from a placeholder"
    keys = set(re.findall(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}", TEMPLATE.read_text(encoding="utf-8")))
    assert keys <= set(_results(deck_dir)["values"]), keys - set(_results(deck_dir)["values"])


# ============================================================================ slide 1: the opening answer

OPENING_TITLE = "From hourly rate feeds to decision-ready pricing signals"
OPENING_SUBTITLE = "Assessing whether Sixt Canada data can support pricing analysis and ongoing monitoring"
OPENING_CONCLUSION = ("Decision-ready for descriptive monitoring; more history and operational evidence are needed "
                      "before using it for pricing decisions.")
OLD_QUESTION = "Can QL2 trust the hourly Sixt Canada rate feed?"
SCOPE_KEYS = ("scope_markets", "scope_locations", "scope_duration")


def _opening(directory: Path) -> tuple[str, str, str]:
    """``(slide 1 HTML, its visible text, its speaker-notes text)`` of a built deck."""
    return _slide(directory, "question")


def _slide(directory: Path, slide_id: str) -> tuple[str, str, str]:
    """``(visible HTML, visible text, speaker-notes text)`` of one primary slide of a built deck."""
    page = (directory / "index.html").read_text(encoding="utf-8")
    start = page.index(f'<section id="{slide_id}"')
    section = page[start:page.index("</section>", start)]
    notes = section[section.index('<aside class="notes">'):]
    visible = section[:section.index('<aside class="notes">')]

    def text(fragment: str) -> str:
        return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()

    return visible, text(visible), text(notes)


def _report_spanning(hours: int) -> fr.FinalReportResult:
    """A ready synthetic run whose scheduled captures span ``hours`` from the first to the last period."""
    from test_monitoring import full_run, stability_report
    from test_price_change_events import synthetic_world
    from test_price_change_presentation import RICH

    world = synthetic_world(hours=hours - 1, products=RICH)      # the rich fixture adds one trailing period
    report = fr.final_report_from_pipeline(full_run(world, vehicle_stability=stability_report(
        observations_assessed=world["population"].eligible_detail_rows)))
    assert report.completed
    return report


def test_opening_slide_leads_with_the_answer(deck_dir: Path) -> None:
    visible, text, _ = _opening(deck_dir)
    parsed = pd_._parse((deck_dir / "index.html").read_text(encoding="utf-8"))
    assert [h.strip() for h in parsed.slides[0]["headings"]] == [OPENING_TITLE]
    assert re.search(r'<h1 id="question-title">' + re.escape(OPENING_TITLE) + "</h1>", visible)
    assert re.search(r'<p class="subtitle">' + re.escape(OPENING_SUBTITLE) + "</p>", visible)
    conclusion = re.search(r'<p class="conclusion-text">(.*?)</p>', visible, re.DOTALL).group(1)
    assert conclusion == f'<span class="v" data-value="conclusion">{OPENING_CONCLUSION}</span>'
    assert _results(deck_dir)["values"]["conclusion"]["display"] == OPENING_CONCLUSION
    order = [text.index(t) for t in (OPENING_TITLE, OPENING_SUBTITLE, OPENING_CONCLUSION)]
    assert order == sorted(order), "title, then subtitle, then the conclusion"
    assert 'aria-describedby="question-conclusion"' in visible and 'aria-label="Conclusion"' in visible
    assert OLD_QUESTION not in (deck_dir / "index.html").read_text(encoding="utf-8")
    assert OLD_QUESTION not in TEMPLATE.read_text(encoding="utf-8")


def test_opening_conclusion_is_the_requested_wording_only_when_controls_are_clean(deck_dir: Path) -> None:
    assert pd_.opening_conclusion(True) == OPENING_CONCLUSION
    assert pd_.opening_conclusion(False).startswith("Not yet decision-ready")
    values = _results(deck_dir)["values"]
    clean = values["controls_triggered"]["value"] == 0 and values["controls_not_assessable"]["value"] == 0
    assert values["conclusion"]["display"] == pd_.opening_conclusion(clean)


def test_opening_slide_has_exactly_three_scope_facts_from_results(deck_dir: Path, ready_report) -> None:
    visible, _, _ = _opening(deck_dir)
    values = _results(deck_dir)["values"]
    items = re.findall(r"<li>(.*?)</li>", re.search(r'<ul class="scope"[^>]*>(.*?)</ul>', visible, re.DOTALL).group(1),
                       re.DOTALL)
    assert len(items) == 3 and len(re.findall(r"<li\b", visible)) == 3, "exactly three visible scope facts"
    for item, key in zip(items, SCOPE_KEYS, strict=True):
        assert item == f'<span class="v" data-value="{key}">{values[key]["display"]}</span>', key
    markets, locations, duration = (values[k] for k in SCOPE_KEYS)
    cities = len(ready_report.matched.report.cities)
    assert markets["value"] == cities == values["city_count"]["value"]
    assert markets["display"] == f"{pd_._number_word(cities).capitalize()} Canadian markets"
    assert locations["display"] == "Airport and downtown locations" and locations["value"] == ["airport", "downtown"]
    assert duration["display"].endswith("of hourly observations")
    window = pd_.observation_window(ready_report.assortment.tables.assortment_timeline[
        __import__("ql2_sixt_canada_analysis.assortment_presentation", fromlist=["_CAPTURE"])._CAPTURE])
    assert duration["display"] == window.phrase
    assert values["observation_hours"]["value"] == window.elapsed_hours
    assert values["observation_periods"]["value"] == window.capture_periods
    template_scope = re.search(r'<ul class="scope"[^>]*>(.*?)</ul>', TEMPLATE.read_text(encoding="utf-8"),
                               re.DOTALL).group(1)
    assert re.sub(r"\s+", "", template_scope) == "".join(f"<li>{{{{{k}}}}}</li>" for k in SCOPE_KEYS), \
        "the template holds only placeholders for the scope facts"


def test_the_four_day_window_is_derived_from_the_capture_window(ready_report, tmp_path: Path) -> None:
    four_days = _report_spanning(92)                             # the supplied extract's shape: 92 hours
    values = pd_.deck_results(four_days)["values"]
    assert values["scope_duration"]["display"] == "Approximately four days of hourly observations"
    assert values["scope_duration"]["value"] == 4 and values["observation_hours"]["value"] == 92
    assert values["observation_periods"]["value"] == 93
    assert values["window_start"]["display"] == "4 Mar 2030, 13:00 UTC"
    assert values["window_end"]["display"] == "8 Mar 2030, 09:00 UTC"
    shorter = pd_.deck_results(_report_spanning(60))["values"]
    assert shorter["scope_duration"]["display"] == "Approximately three days of hourly observations"
    native = pd_.deck_results(ready_report)["values"]
    assert native["scope_duration"]["display"] == "Four hours of hourly observations"
    target = tmp_path / "deck"
    pd_.build_deck(four_days, target)
    pd_.validate_deck(target, four_days)
    _, visible, _ = _opening(target)
    assert "Approximately four days of hourly observations" in visible
    with pytest.raises(pd_.DeckValidationError, match="stale for this run: .*scope_duration"):
        pd_.validate_deck(target, ready_report)                  # a changed source window is detected


@pytest.mark.parametrize(("hours", "phrase", "days"), [
    (92, "Approximately four days of hourly observations", 4),
    (96, "Four days of hourly observations", 4),
    (84, "Approximately four days of hourly observations", 4),   # 3.5 days rounds half up
    (83, "Approximately three days of hourly observations", 3),
    (36, "Approximately two days of hourly observations", 2),
    (35, "35 hours of hourly observations", None),
    (12, "Twelve hours of hourly observations", None),
    (1, "One hour of hourly observations", None),
    (24 * 30, "30 days of hourly observations", 30),
])
def test_observation_window_rounding_rule(hours: int, phrase: str, days: int | None) -> None:
    import datetime as dt

    start = dt.datetime(2030, 3, 4, 2, tzinfo=dt.timezone.utc)
    periods = [f"{start + dt.timedelta(hours=h):%Y%m%dT%H%M%SZ}" for h in (0, 1, hours)]
    window = pd_.observation_window(periods + periods[:1])
    assert window.phrase == phrase and window.rounded_days == days
    assert window.elapsed_hours == hours and window.capture_periods == len(set(periods))


@pytest.mark.parametrize("periods", [[], ["20300304T020000Z"], ["20300304T020000Z", "2030-03-04 03:00"],
                                     ["20300304T020000Z", None]], ids=["empty", "single", "invalid", "missing"])
def test_observation_window_fails_closed(periods: list) -> None:
    with pytest.raises(pd_.DeckError, match="observation window"):
        pd_.observation_window(periods)


def test_opening_slide_hides_governance_detail_and_the_notes_keep_it(deck_dir: Path) -> None:
    _, visible, notes = _opening(deck_dir)
    values = _results(deck_dir)["values"]
    for hidden in (values["authority_record"]["display"], "authority", "canonical", "Thurlow", "alias",
                   values["window_start"]["display"], values["window_end"]["display"], "supplier", "source mode",
                   "win/meet/loss", "out of scope", "governance"):
        assert hidden.lower() not in visible.lower(), hidden
    assert not re.search(r"\d", visible.replace("QL2", "QL")), "slide 1 states its scope in words"
    for kept in (OPENING_CONCLUSION, values["authority_record"]["display"], "decision record", "canonical location",
                 "Vancouver Thurlow", "Vancouver Downtown", values["window_start"]["display"],
                 values["window_end"]["display"], f'{values["observation_hours"]["display"]} hours',
                 "one supplier and one apparent source mode", "competitor, channel and win/meet/loss",
                 "not actual supplier availability", "Why monitoring is supported", "Why pricing needs more evidence",
                 "I treated incomplete evidence", "Transition."):
        assert kept in notes, kept
    assert len(notes.split()) >= 150
    assert OPENING_SUBTITLE not in notes, "the notes add to the slide rather than repeat it"


def test_a_tampered_opening_value_is_detected(deck_copy: Path) -> None:
    page = deck_copy / "index.html"
    display = _results(deck_copy)["values"]["scope_markets"]["display"]
    page.write_text(page.read_text(encoding="utf-8").replace(display, "Four Canadian markets", 1), encoding="utf-8")
    with pytest.raises(pd_.DeckValidationError, match="span scope_markets"):
        pd_.validate_deck(deck_copy)


# ============================================================================ slide 2: analytical judgment

METHOD_TITLE = "How I made the comparisons defensible"
OLD_METHOD_TITLE = "Every finding rests on one gated pipeline run"
FLOW_STAGES = ("Raw captures", "Validate coverage", "Resolve location identity", "Match identical offers",
               "Analyze prices and changes")
JUDGMENT_CALLS = (
    ("Compare like with like", "Same city, collection event, rental period, vehicle, currency, and price basis."),
    ("Exclude incomplete evidence", "Do not impute or silently repair incomplete captures."),
    ("Separate observation from explanation",
     "Report synchronized movements as investigation candidates, not proof of repricing or collection failure."),
)
PRIORITY = "My priority was to reduce false conclusions, even when that meant excluding data."
GATES = ("stream contract", "schedule coverage", "job linkage", "reconciliation", "trusted join", "city integrity",
         "timestamp", "rental-date", "attribute stability", "location authority")


def _flat(fragment: str) -> str:
    return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def test_methodology_slide_shows_the_flow_judgment_calls_and_priority(deck_dir: Path) -> None:
    visible, text, notes = _slide(deck_dir, "approach")
    parsed = pd_._parse((deck_dir / "index.html").read_text(encoding="utf-8"))
    assert parsed.slides[1]["id"] == "approach"
    assert [h.strip() for h in parsed.slides[1]["headings"]] == [METHOD_TITLE]
    assert f'<h2 id="approach-title">{METHOD_TITLE}</h2>' in visible and 'aria-labelledby="approach-title"' in visible
    assert OLD_METHOD_TITLE not in (deck_dir / "index.html").read_text(encoding="utf-8")

    flow = re.search(r'<ol class="flow" aria-label="([^"]+)">(.*?)</ol>', visible, re.DOTALL)
    assert flow and "flow" in flow.group(1).lower(), "an ordered list with a descriptive accessible label"
    stages = [_flat(li) for li in re.findall(r"<li>(.*?)</li>", flow.group(2), re.DOTALL)]
    assert stages == list(FLOW_STAGES)
    assert not re.search(r"[→➜➔>»›]", _flat(flow.group(2))), "connectors are decorative CSS, not text"
    css = (deck_dir / "deck.css").read_text(encoding="utf-8")
    assert re.search(r'#approach \.flow li:not\(:last-child\)::after \{\s*content: "";', css)

    group = re.search(r'<ul class="judgments" aria-label="([^"]+)">(.*?)</ul>', visible, re.DOTALL)
    assert group and "judgment" in group.group(1).lower()
    cards = re.findall(r"<li>(.*?)</li>", group.group(2), re.DOTALL)
    assert len(cards) == len(JUDGMENT_CALLS) == 3
    for card, (heading, support) in zip(cards, JUDGMENT_CALLS, strict=True):
        assert re.search(r"<h3>" + re.escape(heading) + "</h3>", card)
        assert _flat(card) == f"{heading} {support}"
        assert text.count(heading) == 1, heading

    assert re.search(r'<p class="priority">' + re.escape(PRIORITY) + "</p>", visible)
    assert text.endswith(PRIORITY) and PRIORITY not in notes, "the priority is visible content, not a note"
    order = [text.index(t) for t in (METHOD_TITLE, FLOW_STAGES[0], JUDGMENT_CALLS[0][0], PRIORITY)]
    assert order == sorted(order), "title, flow, judgment calls, then the priority"


def test_methodology_slide_keeps_internal_controls_in_the_notes(deck_dir: Path) -> None:
    visible, text, notes = _slide(deck_dir, "approach")
    values = _results(deck_dir)["values"]
    lowered = text.lower()
    for term in GATES + ("gate", "readiness", "authority", "blocker", "candidate group", "match rate",
                         "assumption", "eligible", "monitoring control"):
        assert term not in lowered, term
    assert values["authority_record"]["display"] not in text
    assert "data-value" not in visible and not re.search(r"\d", text), "no quantity on the visible slide"
    assert "<code>" not in visible and 'class="pill' not in visible and "<table" not in visible

    for term in GATES:
        assert term in notes.lower(), term
    for kept in (values["authority_record"]["display"], "decision record", "audit", "imput",
                 "investigation candidate", "not proof of repricing, supplier behaviour or a collection failure",
                 "not actual supplier availability", "repeated measurements",
                 "one supplier and one apparent source mode",
                 "name, type, transmission, seats and bags", "similarity matching", "Transition."):
        assert kept in notes, kept
    note_html = _slide_html(deck_dir, "approach").split('<aside class="notes">')[1]
    note_spans = re.findall(r'data-value="([a-z0-9_]+)"', note_html)
    for key in ("matched_pairs", "candidate_groups", "match_rate", "governed_excluded_captures", "interval_count",
                "control_count", "readiness_status", "authority_record"):
        assert key in note_spans, key
        assert values[key]["display"] in notes, key
    assert len(notes.split()) >= 300


def _slide_html(directory: Path, slide_id: str) -> str:
    page = (directory / "index.html").read_text(encoding="utf-8")
    start = page.index(f'<section id="{slide_id}"')
    return page[start:page.index("</section>", start)]


def test_a_tampered_methodology_note_value_is_detected(deck_copy: Path) -> None:
    page = deck_copy / "index.html"
    display = _results(deck_copy)["values"]["matched_pairs"]["display"]
    text = page.read_text(encoding="utf-8")
    start = text.index('<section id="approach"')
    span = f'data-value="matched_pairs">{display}<'
    position = text.index(span, start)
    page.write_text(text[:position] + span.replace(display, display + "0") + text[position + len(span):],
                    encoding="utf-8")
    with pytest.raises(pd_.DeckValidationError, match="span matched_pairs"):
        pd_.validate_deck(deck_copy)


# ============================================================================ slide 5: decision boundaries

BOUNDARY_TITLE = "What I would trust today—and what I would not"
BOUNDARIES = (
    ("Use now", ("Feed coverage and contract monitoring.", "Descriptive airport-versus-downtown benchmarks.",
                 "Identification of patterns requiring investigation.")),
    ("Investigate", ("The largest market premium.", "Synchronized price movements.", "Abrupt assortment drops.")),
    ("Cannot conclude yet", ("Why a price changed.", "Whether a pattern is normal over time.",
                             "Competitive position or win/meet/loss.", "Total customer cost.")),
)
INTERNAL_TERMS = ("candidate-only", "candidate only", "confirmation-required", "confirmation required",
                  "not assessable", "not-assessable", "severity", "calibration", "critical", "high", "medium",
                  "low", "passed", "triggered", "control_id", "contract_based")


def _notes_table_rows(directory: Path) -> list[tuple[str, list[str]]]:
    """``(control id, [name, severity, status, calibration])`` of every row of the slide 5 notes table."""
    notes = _slide_html(directory, "limits").split('<aside class="notes">')[1]
    table = re.search(r'<table class="notes-controls"[^>]*>(.*?)</table>', notes, re.DOTALL)
    assert table, "the control table is in the speaker notes"
    rows = re.findall(r'<tr data-control-id="([a-z_]+)">(.*?)</tr>', table.group(1), re.DOTALL)
    return [(cid, [_flat(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", cells, re.DOTALL)])
            for cid, cells in rows]


def test_decision_boundary_slide_shows_three_audience_sections(deck_dir: Path) -> None:
    visible, text, _ = _slide(deck_dir, "limits")
    parsed = pd_._parse((deck_dir / "index.html").read_text(encoding="utf-8"))
    assert [s["id"] for s in parsed.slides].index("limits") == 4
    assert [h.strip() for h in parsed.slides[4]["headings"]] == [BOUNDARY_TITLE]
    assert f'<h2 id="limits-title">{BOUNDARY_TITLE}</h2>' in visible
    assert _results(deck_dir)["values"]["monitoring_headline"]["display"] not in text

    assert re.search(r'<div class="boundaries" role="group" aria-label="[^"]+">', visible)
    groups = re.findall(r'<div class="boundary [^"]+" role="group" aria-labelledby="([^"]+)">\s*'
                        r'<h3 id="([^"]+)">(.*?)</h3>\s*<ul>(.*?)</ul>', visible, re.DOTALL)
    assert len(groups) == len(BOUNDARIES) == 3
    for (labelled_by, heading_id, heading, items), (expected, expected_items) in zip(groups, BOUNDARIES,
                                                                                      strict=True):
        assert labelled_by == heading_id and heading == expected
        assert tuple(_flat(li) for li in re.findall(r"<li>(.*?)</li>", items, re.DOTALL)) == expected_items
    assert [text.index(h) for h, _ in BOUNDARIES] == sorted(text.index(h) for h, _ in BOUNDARIES)
    note = "Use the feed to monitor and prioritize investigations—not to set prices or explain why they changed."
    assert text.count(note) == 1 and text.endswith(note), "the boundary statement closes the projected slide"


def test_decision_boundary_slide_hides_the_control_framework(deck_dir: Path) -> None:
    visible, text, notes = _slide(deck_dir, "limits")
    assert "<table" not in visible and 'class="pill' not in visible and 'class="sev' not in visible
    for term in INTERNAL_TERMS:
        assert not re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE), term
    assert "data-value" not in visible and not re.search(r"\d", text), "no counts on the projected slide"
    for row in _results(deck_dir)["monitoring_controls"]:
        assert row["name"] not in text and row["control_id"] not in text
    assert "not assessable" in notes.lower(), "the internal terms are expected in the notes"


def test_speaker_notes_hold_the_complete_generated_control_table(deck_dir: Path, ready_report) -> None:
    from ql2_sixt_canada_analysis.monitoring import monitoring_control

    rows = _notes_table_rows(deck_dir)
    overview = fr.monitoring_overview_table(ready_report)
    assert [cid for cid, _ in rows] == list(overview["control_id"]), "every control once, in authoritative order"
    for (cid, cells), record in zip(rows, overview.to_dict("records"), strict=True):
        assert cells == [monitoring_control(cid).name, record["effective_severity"].replace("_", " "),
                         record["evaluation_status"].replace("_", " "),
                         record["calibration_status"].replace("_", " ")]
    values = _results(deck_dir)["values"]
    statuses = [cells[2] for _, cells in rows]
    for key, status in (("controls_passed", "passed"), ("controls_triggered", "triggered"),
                        ("controls_candidate_only", "candidate only"),
                        ("controls_confirmation", "confirmation required"),
                        ("controls_not_assessable", "not assessable")):
        assert values[key]["value"] == statuses.count(status), key
    assert values["control_count"]["value"] == len(rows)
    template = TEMPLATE.read_text(encoding="utf-8")
    start = template.index('<section id="limits"')
    slide = template[start:template.index("</section>", start)]
    assert "{{>monitoring_notes_table}}" in slide and "<table" not in slide and "<tr" not in slide
    for record in overview.to_dict("records"):
        assert monitoring_control(record["control_id"]).name not in slide


def test_a_changed_control_status_changes_the_notes_table(deck_dir: Path, ready_report) -> None:
    results = pd_.deck_results(ready_report)
    results["figures"] = _results(deck_dir)["figures"]
    first = results["monitoring_controls"][0]
    original = first["status"]
    first["status"] = "triggered" if original != "triggered" else "passed"
    page = pd_.render_deck_html(TEMPLATE.read_text(encoding="utf-8"), results)
    start = page.index('<section id="limits"')
    notes = page[start:page.index("</section>", start)].split('<aside class="notes">')[1]
    row = re.search(rf'<tr data-control-id="{first["control_id"]}">(.*?)</tr>', notes, re.DOTALL).group(1)
    assert first["status"].replace("_", " ") in _flat(row) and original.replace("_", " ") not in _flat(row)


def test_a_stale_or_tampered_control_table_is_detected(deck_copy: Path, ready_report) -> None:
    results = _results(deck_copy)
    results["monitoring_controls"][0]["status"] = "not_assessable"
    (deck_copy / pd_.DECK_RESULTS_FILENAME).write_text(json.dumps(results), encoding="utf-8")
    (deck_copy / "index.html").write_text(pd_.render_deck_html(TEMPLATE.read_text(encoding="utf-8"), results),
                                          encoding="utf-8")
    pd_.validate_deck(deck_copy)                                   # internally consistent ...
    with pytest.raises(pd_.DeckValidationError, match="stale for this run: monitoring_controls"):
        pd_.validate_deck(deck_copy, ready_report)                 # ... but not this run's monitoring result
    page = deck_copy / "index.html"
    page.write_text(page.read_text(encoding="utf-8").replace("not assessable</td>", "passed</td>", 1),
                    encoding="utf-8")
    with pytest.raises(pd_.DeckValidationError, match="stale"):
        pd_.validate_deck(deck_copy)                               # an edited page no longer renders from results


def test_decision_boundary_notes_explain_the_framework_and_its_limits(deck_dir: Path) -> None:
    _, _, notes = _slide(deck_dir, "limits")
    values = _results(deck_dir)["values"]
    for kept in ("I separated what the evidence can support today", "operationalize", "descriptive benchmarks",
                 "associational, not causal", "before any alert threshold is approved", "investigation lead, not a "
                 "pricing recommendation", "not evidence of intentional repricing",
                 "do not show that supplier availability fell", "collection logs or source snapshots",
                 "normal variation or seasonality",
                 "win/meet/loss", "not total customer cost", "right-censored", "subsequent captures",
                 "never alerts", "Not assessable is never treated as a pass", "not failed analyses",
                 values["monitoring_headline"]["display"], values["top_city_name"]["display"], "Transition."):
        assert kept in notes, kept
    assert len(notes.split()) >= 400


# ============================================================================ slide 6: prioritized next steps

NEXT_HEADING = "What I would do next"
PRIORITIES = (
    ("Productionize the reliable checks",
     "Run coverage, reconciliation, timestamp, location, and product-identity checks on every delivery."),
    ("Investigate the largest commercial signal",
     "Review the largest airport/downtown premium with revenue management."),
    ("Close the evidence gaps",
     "Request longer history, subsequent captures, collection logs, competitor prices, and total-cost components."),
    ("Calibrate monitoring thresholds", "Establish normal variation before converting candidates into alerts."),
)
WEEK_HEADING = "With one more week, I would…"
CLOSING = ("The feed is reliable enough to tell us where to look, but not yet complete enough to tell us what "
           "commercial action to take.")


def _inline(fragment: str) -> str:
    """Flattened text with inline value spans joined to their neighbours (no space before punctuation)."""
    return _flat(re.sub(r"</?span[^>]*>", "", fragment))


def _leading_market_changed(report: fr.FinalReportResult) -> fr.FinalReportResult:
    """The same run with the last approved city's dollar premiums shifted up, so it leads the ranking."""
    matched = report.matched
    cities = list(matched.report.cities)
    dollars = cities[-1].dollars
    shifted = {f.name: getattr(dollars, f.name) + 1000.0 for f in dataclasses.fields(dollars)
               if f.name in ("mean", "median", "q25", "q75", "minimum", "maximum")}
    cities[-1] = dataclasses.replace(cities[-1], dollars=dataclasses.replace(dollars, **shifted))
    matched = dataclasses.replace(matched, report=dataclasses.replace(matched.report, cities=tuple(cities)))
    return dataclasses.replace(report, matched=matched)


def test_final_slide_presents_a_prioritized_plan(deck_dir: Path) -> None:
    visible, text, notes = _slide(deck_dir, "conclusions")
    parsed = pd_._parse((deck_dir / "index.html").read_text(encoding="utf-8"))
    assert parsed.slides[-1]["id"] == "conclusions" and len(parsed.slides) == 6
    assert re.search(r'<h3 id="next-title">' + re.escape(NEXT_HEADING) + "</h3>", visible)
    assert "Recommendations" not in text

    plan = re.search(r'<ol class="priorities"[^>]*aria-labelledby="next-title"[^>]*>(.*?)</ol>', visible, re.DOTALL)
    assert plan, "the priorities are a semantic ordered list labelled by the section heading"
    items = re.findall(r"<li>(.*?)</li>", plan.group(1), re.DOTALL)
    assert len(items) == len(PRIORITIES) == 4
    for item, (heading, support) in zip(items, PRIORITIES, strict=True):
        assert re.search(r"<h4>" + re.escape(heading) + "</h4>", item), heading
        assert _flat(item).startswith(f"{heading} {support}"), heading
        assert text.count(heading) == 1, heading
    assert PRIORITIES[0][0] in _flat(items[0]) and PRIORITIES[-1][0] in _flat(items[-1])

    values = _results(deck_dir)["values"]
    lead = _inline(items[1])
    assert lead.endswith(f"Largest median premium: {values['top_city_name']['display']}, "
                         f"{values['top_city_median']['display']}.")
    assert 'data-value="top_city_name"' in items[1] and 'data-value="top_city_median"' in items[1]

    week = re.search(r'<div class="one-week" role="group" aria-labelledby="week-title">(.*?)</div>', visible,
                     re.DOTALL)
    assert week and f'<h3 id="week-title">{WEEK_HEADING}</h3>' in week.group(1)
    week_text = _inline(week.group(1))
    for idea in ("longer-window baseline", "Backtest candidate alert thresholds",
                 f"Investigate the {values['top_city_name']['display']} premium with commercial and operational"):
        assert idea in week_text, idea
    assert "Start now" in week_text and "Once history arrives" in week_text, "now versus data-dependent work"

    assert text.count(CLOSING) == 1 and text.endswith(CLOSING), "the closing statement is the final visible element"
    assert re.search(r'<p class="closing">' + re.escape(CLOSING).replace(r"\ ", r"\s+") + "</p>", visible)
    order = [text.index(t) for t in (NEXT_HEADING, PRIORITIES[0][0], PRIORITIES[-1][0], WEEK_HEADING, CLOSING)]
    assert order == sorted(order)
    assert "not alerts" in text and not re.search(r"\b(confirmed|production|triggered) alerts?\b", text + notes)


def test_final_slide_template_names_no_market_or_premium(deck_dir: Path) -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    start = template.index('<section id="conclusions"')
    slide = template[start:template.index("</section>", start)]
    for city in _results(deck_dir)["cities"]:
        assert city["city"] not in slide, city["city"]
    assert "{{top_city_name}}" in slide and "{{top_city_median}}" in slide
    assert not re.search(r"\$|\d", _flat(re.sub(r"\{\{.*?\}\}", "", slide)).replace("QL2", "QL"))


def test_a_different_leading_market_changes_the_final_slide(ready_report, deck_dir: Path) -> None:
    original = _results(deck_dir)["values"]["top_city_name"]["display"]
    changed = _leading_market_changed(ready_report)
    results = pd_.deck_results(changed)
    leader = results["values"]["top_city_name"]["display"]
    assert leader == results["cities"][-1]["city"] != original
    page = pd_.render_deck_html(TEMPLATE.read_text(encoding="utf-8"),
                                {**results, "figures": _results(deck_dir)["figures"]})
    start = page.index('<section id="conclusions"')
    slide = _inline(page[start:page.index("</section>", start)].split('<aside class="notes">')[0])
    assert f"Largest median premium: {leader}, {results['values']['top_city_median']['display']}." in slide
    assert f"Investigate the {leader} premium" in slide and original not in slide
    with pytest.raises(pd_.DeckValidationError, match="stale for this run: .*top_city_name"):
        pd_.validate_deck(deck_dir, changed)                     # the built deck no longer matches this run


def test_final_slide_notes_explain_the_order_and_the_evidence_gaps(deck_dir: Path) -> None:
    _, _, notes = _slide(deck_dir, "conclusions")
    for kept in ("Why this order", "need no new data or threshold", "it is a lead", "revenue management",
                 "collection owner", "normal variation and seasonality", "right-censored",
                 "market behaviour from collection behaviour", "win/meet/loss", "Total-cost components",
                 "customer-relevant", "calibrated and backtested", "With one more week", "depend on others",
                 CLOSING, "what would you like to explore"):
        assert kept in notes, kept
    assert len(notes.split()) >= 400


def test_a_tampered_final_slide_value_is_detected(deck_copy: Path) -> None:
    page = deck_copy / "index.html"
    text = page.read_text(encoding="utf-8")
    display = _results(deck_copy)["values"]["top_city_name"]["display"]
    start = text.index('<section id="conclusions"')
    span = f'data-value="top_city_name">{display}<'
    position = text.index(span, start)
    page.write_text(text[:position] + span.replace(display, "Elsewhere") + text[position + len(span):],
                    encoding="utf-8")
    with pytest.raises(pd_.DeckValidationError, match="span top_city_name"):
        pd_.validate_deck(deck_copy)


# ============================================================================ figures, labels and units


def test_figures_are_valid_presentation_ready_pngs(deck_dir: Path) -> None:
    results = _results(deck_dir)
    for name, spec in pd_.DECK_FIGURES.items():
        meta = results["figures"][name]
        path = deck_dir / meta["file"]
        width, height = pd_._png_size(path)
        assert (width, height) == (meta["width"], meta["height"])
        assert width >= 1800 and height >= 700, name
        assert abs(width / height - spec["figsize"][0] / spec["figsize"][1]) < 0.02, name
        assert path.stat().st_size > 20_000


def test_figure_labels_and_units_are_explicit(deck_dir: Path) -> None:
    results = _results(deck_dir)
    matched, heatmap = results["figures"]["matched_premiums"], results["figures"]["price_change_heatmap"]
    assert pd_._check_labels("matched_premiums", matched) == []
    assert pd_._check_labels("price_change_heatmap", heatmap) == []
    symbol = results["values"]["city1_median"]["display"].lstrip("+-").split("$")[0] + "$"
    assert matched["axes"][0]["ylabel"].endswith(f"({symbol}/day)")
    assert matched["axes"][1]["ylabel"].endswith("(%)") and "%" in matched["axes"][1]["title"]
    cities = [c["city"] for c in results["cities"]]
    assert all(a["xticklabels"] == cities for a in matched["axes"]), "category labels are the full city names"
    assert "IQR" not in json.dumps(matched) and "interquartile range" in matched["note"]
    assert heatmap["colorbar_labels"] == ["Candidates (count)", "Candidates (count)"]
    assert heatmap["axes"][1]["xlabel"].endswith("(UTC)")
    assert all(re.fullmatch(r"\d{8}T\d{6}Z", t) for t in heatmap["axes"][1]["xticklabels"])
    assert "not zero activity" in heatmap["note"]


def test_label_rules_detect_missing_units() -> None:
    meta = {"title": "Matched airport vs downtown premiums", "colorbar_labels": [], "legend": [], "note": "",
            "axes": [{"title": "Dollar premium (CA$ per day)", "xlabel": "", "ylabel": "Airport minus downtown",
                      "xticklabels": ["A"], "yticklabels": ["1"]},
                     {"title": "Percentage premium (% of downtown price)", "xlabel": "",
                      "ylabel": "(Airport - downtown) / downtown", "xticklabels": ["A…"], "yticklabels": ["1"]}]}
    problems = pd_._check_labels("matched_premiums", meta)
    assert any("unit" in p for p in problems) and any("truncated" in p for p in problems)


def test_figure_values_agree_with_the_results(deck_dir: Path, ready_report) -> None:
    results = _results(deck_dir)
    matched = ready_report.matched.report
    for city, axis_label in zip(matched.cities, results["figures"]["matched_premiums"]["axes"][0]["xticklabels"]):
        assert axis_label == city.city.title()
    assert [c["city"] for c in results["cities"]] == [c.city.title() for c in matched.cities]
    table = ready_report.price_changes.analysis.event_table
    assert int(table["increase"].sum()) == results["values"]["pc_increases"]["value"]
    assert int(table["decrease"].sum()) == results["values"]["pc_decreases"]["value"]


# ============================================================================ values: reconciliation and staleness


def test_every_value_span_matches_the_results_artifact(deck_dir: Path) -> None:
    results = _results(deck_dir)
    parsed = pd_._parse((deck_dir / "index.html").read_text(encoding="utf-8"))
    assert parsed.spans
    for key, parts in parsed.spans:
        assert "".join(parts) == results["values"][key]["display"], key
    shown = {k for k, _ in parsed.spans}
    for required in ("matched_pairs", "pc_total", "retention", "controls_passed", "city1_median"):
        assert required in shown


def test_values_trace_to_the_final_report_text(deck_dir: Path, ready_report) -> None:
    results = _results(deck_dir)
    traced = [k for k, v in results["values"].items() if v["trace"]]
    assert len(traced) >= 30
    assert pd_.verify_traces(results, pd_._report_text(ready_report)) == ()
    assert pd_.verify_traces(results, "") == tuple(traced)


def test_values_are_recomputed_from_the_report(deck_dir: Path, ready_report) -> None:
    values = _results(deck_dir)["values"]
    pc = ready_report.price_changes.analysis.report
    matched = ready_report.matched.report
    assert values["pc_total"]["value"] == pc.price_change_count
    assert values["interval_count"]["value"] == pc.intervals
    assert values["matched_pairs"]["value"] == matched.overall.counts.matched
    assert values["match_rate"]["display"] == f"{matched.overall.counts.match_rate:.0%}"
    for i, city in enumerate(matched.cities, start=1):
        assert values[f"city{i}_median"]["value"] == city.dollars.median
        assert values[f"city{i}_median_pct"]["display"] == f"{city.percent.median:+.1f}%"
    assert values["control_count"]["value"] == len(fr.monitoring_overview_table(ready_report))


def test_formatting_is_consistent_by_kind(deck_dir: Path) -> None:
    patterns = {"count": r"\d{1,3}(,\d{3})*", "money": r"[+-]?[A-Z]*\$\d{1,3}(,\d{3})*\.\d{2}/\w+",
                "percent": r"[+-]\d+\.\d{1,2}%( to [+-]\d+\.\d{2}%)?", "share": r"\d{1,3}%",
                "ratio": r"\d\.\d{2}", "date": r"\d{1,2} [A-Z][a-z]{2} \d{4}, \d{2}:\d{2} UTC",
                "duration": r"\d{1,3}(,\d{3})*(\.\d)?",
                "scope": r"[A-Z][a-z]+( [A-Za-z]+)* (markets?|locations( only)?|(days?|hours?) of hourly observations)"}
    for key, item in _results(deck_dir)["values"].items():
        if item["kind"] in patterns:
            assert re.fullmatch(patterns[item["kind"]], item["display"]), (key, item["display"])
        assert item["source"], key


def test_a_tampered_page_is_detected(deck_copy: Path) -> None:
    page = deck_copy / "index.html"
    text = page.read_text(encoding="utf-8")
    key = "pc_total"
    display = _results(deck_copy)["values"][key]["display"]
    page.write_text(text.replace(f'data-value="{key}">{display}<', f'data-value="{key}">{display}9<', 1),
                    encoding="utf-8")
    with pytest.raises(pd_.DeckValidationError, match="stale|span pc_total"):
        pd_.validate_deck(deck_copy)


def test_stale_results_are_detected_against_the_run(deck_copy: Path, ready_report) -> None:
    results = _results(deck_copy)
    results["values"]["matched_pairs"]["value"] += 1
    results["values"]["matched_pairs"]["display"] = f"{results['values']['matched_pairs']['value']:,}"
    (deck_copy / pd_.DECK_RESULTS_FILENAME).write_text(json.dumps(results), encoding="utf-8")
    (deck_copy / "index.html").write_text(pd_.render_deck_html(TEMPLATE.read_text(encoding="utf-8"), results),
                                          encoding="utf-8")
    pd_.validate_deck(deck_copy)                                   # internally consistent ...
    with pytest.raises(pd_.DeckValidationError, match="stale for this run: matched_pairs"):
        pd_.validate_deck(deck_copy, ready_report)                 # ... but stale for the run


def test_structural_defects_are_detected(deck_copy: Path, tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(PROJECT_ROOT / "presentation", source)
    template = (source / "deck.html").read_text(encoding="utf-8")
    broken = re.sub(r'<aside class="notes">.*?</aside>', "", template, count=1, flags=re.DOTALL)
    broken = broken.replace('<section id="conclusions"', '<section id="extra"><h2>Extra</h2></section>\n'
                            '    <section id="conclusions"', 1)
    broken = broken.replace('src="figures/matched_premiums.png"', 'src="https://example.org/x.png"')
    (source / "deck.html").write_text(broken, encoding="utf-8")
    (deck_copy / "index.html").write_text(pd_.render_deck_html(broken, _results(deck_copy)), encoding="utf-8")
    (deck_copy / "figures" / "stale.png").write_bytes(b"not a png")
    with pytest.raises(pd_.DeckValidationError) as error:
        pd_.validate_deck(deck_copy, source_dir=source)
    message = str(error.value)
    for expected in ("exactly 6 primary slides", "slide 1 speaker notes", "not a relative local path",
                     "files the deck does not use", "figure matched_premiums is not shown"):
        assert expected in message, expected


def test_missing_notes_plugin_or_slide_numbers_are_detected(deck_copy: Path) -> None:
    script = deck_copy / "deck.js"
    script.write_text(script.read_text(encoding="utf-8").replace('slideNumber: "c/t"', "slideNumber: false")
                      .replace("plugins: [RevealNotes]", "plugins: []"), encoding="utf-8")
    with pytest.raises(pd_.DeckValidationError, match="slide numbers are not enabled") as error:
        pd_.validate_deck(deck_copy)
    assert "notes plugin is not configured" in str(error.value)


def test_rebuild_replaces_the_directory_without_stale_files(ready_report, tmp_path: Path) -> None:
    target = tmp_path / "presentation"
    pd_.build_deck(ready_report, target)
    (target / "figures" / "old_export.png").write_bytes(b"\x89PNG")
    first = (target / "index.html").read_text(encoding="utf-8")
    pd_.build_deck(ready_report, target)
    assert not (target / "figures" / "old_export.png").exists()
    assert (target / "index.html").read_text(encoding="utf-8") == first, "the build is deterministic"
    assert (target / "figures" / "matched_premiums.png").read_bytes() == \
        (target / "figures" / "matched_premiums.png").read_bytes()


# ============================================================================ fail-closed behaviour


def test_a_blocked_report_builds_no_deck(tmp_path: Path) -> None:
    blocked = fr.final_report_from_pipeline(dataclasses.replace(ready_run(), pricing=None))
    target = tmp_path / "presentation"
    with pytest.raises(pd_.DeckError, match="completed final report"):
        pd_.build_deck(blocked, target)
    assert not target.exists() and not list(tmp_path.iterdir())
    with pytest.raises(TypeError):
        pd_.deck_results(object())


def test_the_deck_is_never_written_into_source_data(ready_report) -> None:
    from ql2_sixt_canada_analysis.paths import RAW_DATA_DIR

    with pytest.raises(pd_.DeckError, match="data directory"):
        pd_.build_deck(ready_report, RAW_DATA_DIR / "deck")
    assert not (RAW_DATA_DIR / "deck").exists()


def test_validate_reports_an_unbuilt_deck(tmp_path: Path) -> None:
    with pytest.raises(pd_.DeckValidationError, match="has not been built"):
        pd_.validate_deck(tmp_path)


# ============================================================================ documented commands


def test_documented_build_and_validate_commands_work(ready_report, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(fr, "run_final_report", lambda raw_dir=None: ready_report)
    target = tmp_path / "deck"
    assert pd_.main(["build", "--output-dir", str(target)]) == 0
    assert "Deck built: 6 slides, 2 figures" in capsys.readouterr().out
    assert pd_.main(["validate", "--output-dir", str(target)]) == 0
    assert "Deck valid: 6 slides" in capsys.readouterr().out


def test_build_command_fails_closed_on_synthetic_csvs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from conftest import contract_columns, write_synthetic_csv

    from ql2_sixt_canada_analysis.schemas import DatasetKey

    raw = tmp_path / "raw"
    raw.mkdir()
    for key in DatasetKey:
        write_synthetic_csv(raw / f"synthetic_{key}.csv", contract_columns(key), rows=3)
    target = tmp_path / "deck"
    assert pd_.main(["build", "--raw-dir", str(raw), "--output-dir", str(target)]) == 1
    assert "completed final report" in capsys.readouterr().err
    assert not target.exists()


def test_readme_documents_the_presentation_workflow() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in ("python -m ql2_sixt_canada_analysis.presentation_deck build",
                   "python -m ql2_sixt_canada_analysis.presentation_deck validate",
                   "python -m ql2_sixt_canada_analysis.presentation_deck verify-notebook",
                   "python -m http.server", "reports/presentation/", "python -m pytest tests/test_presentation_deck.py",
                   "speaker view"):
        assert phrase in readme, phrase


def test_notebook_output_text_reads_streams_and_displays() -> None:
    notebook = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("x")
    cell.outputs = [nbformat.v4.new_output("stream", text="179 price-change candidates"),
                    nbformat.v4.new_output("execute_result", data={"text/plain": "table"}, execution_count=1)]
    notebook.cells.append(cell)
    text = pd_.notebook_output_text(notebook)
    assert "179 price-change candidates" in text and "table" in text
    results = {"values": {"a": {"trace": "179 price-change candidates"}, "b": {"trace": "missing"},
                          "c": {"trace": None}}}
    assert pd_.verify_traces(results, text) == ("b",)


def test_sources_are_tracked_and_the_built_deck_is_git_ignored() -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    def ignored(path: str) -> bool:
        result = subprocess.run(["git", "check-ignore", "--quiet", "--no-index", path], cwd=PROJECT_ROOT,
                                capture_output=True, check=False)
        if result.returncode not in (0, 1):
            pytest.skip("not a Git work tree")
        return result.returncode == 0

    for path in ("reports/presentation/index.html", "reports/presentation/deck_results.json",
                 "reports/presentation/figures/matched_premiums.png", "reports/.presentation.building/index.html"):
        assert ignored(path), path
    for path in ("presentation/deck.html", "presentation/deck.css", "presentation/deck.js",
                 "presentation/README.md", "presentation/vendor/reveal.js/reveal.js",
                 "presentation/vendor/reveal.js/plugin/notes/notes.js", "presentation/vendor/reveal.js/LICENSE"):
        assert not ignored(path), path


# ============================================================================ figure regressions (authoritative plots)


def test_matched_figure_units_and_font_scale(ready_report) -> None:
    from ql2_sixt_canada_analysis.matched_location_pricing import plot_matched_location_premiums

    fig, (dollars, percent) = plot_matched_location_premiums(ready_report.matched)
    assert percent.get_ylabel().endswith("(%)") and re.search(r"\(\S+/\w+\)$", dollars.get_ylabel())
    assert dollars._left_title.get_fontsize() == 12                      # the notebook default is unchanged
    big, (big_dollars, _) = plot_matched_location_premiums(ready_report.matched, font_scale=1.5)
    assert big_dollars._left_title.get_fontsize() == pytest.approx(18)
    assert big_dollars.yaxis.label.get_fontsize() == pytest.approx(15)
    fig.clear()
    big.clear()


def test_heatmap_colour_bars_are_labelled_with_units(ready_report) -> None:
    from ql2_sixt_canada_analysis.price_change_analysis import plot_price_change_heatmap

    fig, _ = plot_price_change_heatmap(ready_report.price_changes.analysis)
    labels = [ax.get_ylabel() for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert labels == ["Candidates (count)", "Candidates (count)"]
    default_width = fig.get_figwidth()
    sized, _ = plot_price_change_heatmap(ready_report.price_changes.analysis, figsize=(13.0, 6.0), font_scale=1.2)
    assert tuple(sized.get_size_inches()) == (13.0, 6.0) and default_width >= 11.0
    fig.clear()
    sized.clear()


def test_heatmap_legend_and_note_do_not_overlap(ready_report) -> None:
    from ql2_sixt_canada_analysis.price_change_analysis import plot_price_change_heatmap

    for kwargs in ({}, {"figsize": (13.0, 6.2), "font_scale": 1.1}):
        fig, _ = plot_price_change_heatmap(ready_report.price_changes.analysis, **kwargs)
        FigureCanvasAgg(fig).draw()
        renderer = fig.canvas.get_renderer()
        legend = fig.legends[0].get_window_extent(renderer)
        note = fig._supxlabel.get_window_extent(renderer)
        title = fig._suptitle.get_window_extent(renderer)
        assert not legend.overlaps(note) and not legend.overlaps(title), kwargs
        fig.clear()


def test_assortment_timeline_footer_does_not_overlap_the_axis_label(ready_report) -> None:
    from ql2_sixt_canada_analysis.assortment_presentation import _timeline_figure

    fig = _timeline_figure(ready_report.assortment.tables, 100)
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()
    xlabel = fig.axes[-1].xaxis.label.get_window_extent(renderer)
    ticks = [t.get_window_extent(renderer) for t in fig.axes[-1].get_xticklabels() if t.get_text()]
    footer = min((t for t in fig.texts if t.get_text() and t is not fig._supylabel),
                 key=lambda t: t.get_position()[1]).get_window_extent(renderer)
    assert footer.y1 <= xlabel.y0, "the footer sits below the axis label, not beside it"
    assert all(footer.y1 <= t.y0 for t in ticks)
    assert footer.y0 >= 0 and xlabel.y0 >= 0, "nothing is cut off at the bottom"
    fig.clear()


# ============================================================================ optional headless-browser smoke test


def _playwright():  # type: ignore[no-untyped-def]
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("Playwright is not installed (optional; see README)")
    return sync_playwright


@pytest.mark.parametrize("viewport", [(1920, 1080), (1024, 640)], ids=["presentation", "small"])
def test_browser_smoke(deck_dir: Path, viewport: tuple[int, int]) -> None:
    sync_playwright = _playwright()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(deck_dir))
    handler.log_message = lambda *args, **kwargs: None  # type: ignore[method-assign]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    shots = os.environ.get(SCREENSHOT_DIR_ENV_VAR)
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as error:  # noqa: BLE001 - no browser build is an environment gap, not a deck defect
                pytest.skip(f"no Chromium build for Playwright: {type(error).__name__}")
            page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            problems: list[str] = []
            page.on("console", lambda m: problems.append(f"console {m.type}: {m.text}")
                    if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
            page.on("requestfailed", lambda r: problems.append(f"failed: {r.url}"))
            page.on("response", lambda r: problems.append(f"HTTP {r.status}: {r.url}") if r.status >= 400 else None)
            page.goto(url)
            page.wait_for_selector("html[data-deck-ready=true]", timeout=15_000)
            assert page.evaluate("Reveal.getTotalSlides()") == 6
            assert page.evaluate("Reveal.getHorizontalSlides().length") == 6
            assert page.evaluate("Reveal.hasPlugin('notes')")
            for index in range(6):
                page.evaluate(f"Reveal.slide({index})")
                page.wait_for_timeout(400)
                state = page.evaluate("""() => {
                  const slide = Reveal.getCurrentSlide();
                  const box = slide.getBoundingClientRect();
                  const outside = [];
                  slide.querySelectorAll('*').forEach(el => {
                    if (el.closest('aside.notes')) return;
                    const b = el.getBoundingClientRect();
                    if (!b.width || !b.height) return;
                    if (b.bottom > box.bottom + 1 || b.right > box.right + 1 || b.left < box.left - 1 ||
                        b.top < box.top - 1) outside.push(el.tagName + '.' + el.className);
                  });
                  const broken = [...slide.querySelectorAll('img')].filter(i => !i.complete || !i.naturalWidth);
                  return {number: document.querySelector('.slide-number').innerText.replace(/\\s/g, ''),
                          outside, broken: broken.length,
                          inView: box.bottom <= innerHeight + 1 && box.right <= innerWidth + 1,
                          notes: (slide.querySelector('aside.notes') || {}).textContent || ''};
                }""")
                assert state["number"] == f"{index + 1}/6"
                assert state["outside"] == [], (index + 1, state["outside"])
                assert state["broken"] == 0 and state["inView"]
                assert len(state["notes"].split()) >= 60
                if shots:
                    Path(shots).mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(Path(shots) / f"{viewport[0]}x{viewport[1]}_slide{index + 1}.png"))
            page.evaluate("Reveal.slide(0)")
            with page.expect_popup(timeout=10_000) as popup:
                page.keyboard.press("s")                     # the notes plugin opens the speaker view
            speaker = popup.value
            speaker.wait_for_load_state()
            assert speaker.locator(".speaker-controls-notes").count() == 1
            speaker.close()
            assert problems == [], problems
            browser.close()
    finally:
        server.shutdown()
