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

To update reveal.js, replace the four files and `LICENSE` from the new npm
package (`npm pack reveal.js@<version>`), update the version above and rebuild.
