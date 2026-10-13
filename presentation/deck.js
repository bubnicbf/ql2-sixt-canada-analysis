/* Initialise the Sixt Canada rate-feed review deck (reveal.js 5, vendored under vendor/reveal.js). */
/* global Reveal, RevealNotes */
(function () {
  "use strict";

  Reveal.initialize({
    // Fixed 16:9 canvas; reveal.js scales it to the window.
    width: 1280,
    height: 720,
    margin: 0.03,
    minScale: 0.2,
    maxScale: 2.0,
    center: false,
    hash: true,
    slideNumber: "c/t",
    showSlideNumber: "all",
    controls: true,
    progress: true,
    transition: "fade",
    transitionSpeed: "fast",
    // Six horizontal slides only; no vertical navigation.
    navigationMode: "linear",
    plugins: [RevealNotes]
  }).then(function () {
    document.documentElement.setAttribute("data-deck-ready", "true");
  });
})();
