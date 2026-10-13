# presentation

Sources of the six-slide reveal.js deck. The deck itself is **generated** by
`python -m ql2_sixt_canada_analysis.presentation_deck build` into the
Git-ignored `reports/presentation/` directory, because every number on it is
derived from the proprietary extract. Never commit a built deck, its figures or
its results artifact.

| File | Purpose |
| --- | --- |
| `deck.html` | Slide template. Six top-level `<section>` slides, each with an `<aside class="notes">` speaker-notes block. Quantities are never typed here: `{{ key }}` inserts a value from the results artifact (rendered as `<span data-value="key">`), `{{= key }}` inserts escaped plain text (attributes and alternative text) and `{{> name }}` a generated block (city cards, monitoring rows). |
| `deck.css` | Deck styles. System fonts only; colours follow the analysis figures. |
| `deck.js` | `Reveal.initialize` configuration: 1280×720 canvas, `slideNumber: "c/t"`, linear navigation and the notes plugin. |
| `vendor/reveal.js/` | Vendored reveal.js 5.1.0 (MIT, see `vendor/reveal.js/LICENSE`): `reveal.js`, `reveal.css`, `reset.css` and `plugin/notes/notes.js`, copied unchanged from the npm package so the deck needs no CDN. |

Rules for editing the template (enforced by `tests/test_presentation_deck.py`):

- Keep exactly six top-level slides; no vertical (nested) slides.
- Every slide keeps a heading and speaker notes of at least 60 words.
- Write no digits in visible slide or notes text: every quantity comes from a
  placeholder so the deck cannot drift from the analysis.
- Reference assets only by relative paths inside the built deck directory.

Slide 1 leads with the conclusion rather than the method: title, subtitle, the
conclusion (`conclusion`, from `opening_conclusion`) and exactly three scope
facts (`scope_markets`, `scope_locations`, `scope_duration`). The authority
record, canonical locations, the Vancouver alias, the exact window and the
supplier and source-mode limits are in its speaker notes. `scope_duration`
comes from `observation_window`: the elapsed time from the first to the last
scheduled capture period (endpoints are capture instants, so no hour is added),
rounded half up to whole days and prefixed "Approximately" unless exact;
windows under 36 hours are stated in hours. The exact hours, period count and
first and last periods stay in `deck_results.json`.

Slide 2 explains the analytical judgment rather than the control system: a
five-stage flow (an ordered list with CSS-only connectors), three judgment
calls and the closing priority statement. It shows no quantities; the
individual readiness gates, the decision record and the matched-pair,
exclusion, interval and control counts are in its speaker notes, as
placeholders. Its styles are scoped to `#approach`.

Slide 6 closes with a decision plan rather than a results recap: a short
evidence strip, four prioritized actions in a semantic ordered list ("What I
would do next"), a compact one-week follow-on plan that separates work that can
start now from work that needs more data, and the closing statement that marks
the boundary between investigation and commercial action. The leading market
and its premium come from `top_city_name` and `top_city_median`; the per-question
recap and the rationale for each evidence gap are in its speaker notes. Its
styles are scoped to `#conclusions`.

To update reveal.js, replace the four files and `LICENSE` from the new npm
package (`npm pack reveal.js@<version>`), update the version above and rebuild.
