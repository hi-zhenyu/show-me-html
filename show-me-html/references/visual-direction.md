# Visual Direction

Use this reference when an artifact needs to feel designed, not merely formatted. The aim is content-first distinction: a clear point of view that helps the material land.

## Choose a point of view

Before writing CSS, choose:

- **Tone:** editorial, utilitarian, research lab, operations console, refined minimal, playful tool, technical notebook, brutalist, luxury, retro, organic, or another deliberate direction.
- **Memory:** one visual idea the reader will remember: a timeline spine, annotated margin, tactile controls, data-ribbon header, blueprint grid, paper dossier, stage-like deck, or custom diagram language.
- **Density:** calm and spacious, dense and operational, or layered and exploratory.
- **Constraint:** what must stay quiet so the content remains usable.

Bold and minimal are both valid. The failure mode is not restraint; the failure mode is default.

## Typography

- Pick type that supports the subject. Editorial explainers can use a distinctive serif; dashboards and tools need compact, legible UI type; code-heavy pages need a strong mono pairing.
- Do not default to the same overused font stack for every artifact. System fonts are acceptable for utilitarian tools, but make that an intentional choice.
- Pair one display voice with one reading voice. Avoid more than three font families.
- Use tabular numbers for metrics, rankings, timers, prices, and comparison tables.
- Use balanced or pretty wrapping for headings when supported.
- Keep line length comfortable: about 60-75 characters for prose, tighter for dense panels.

## Color and theme

- Define CSS variables for background, surface, text, muted text, rules, accent, danger, warning, success, and focus.
- Choose a dominant neutral or field color plus sharp accents. Avoid timid palettes where every color has equal weight.
- Avoid one-note palettes dominated by variations of a single hue.
- Avoid the common generated look: purple-blue gradients, glass panels, random neon glows, emoji headers, and four nearly identical card styles.
- Use color to encode meaning only with text, icons, labels, or shape as backup.
- Support dark mode only when it can be done cleanly; otherwise make one excellent theme.

## Composition

- Let the content shape determine the layout: columns for comparisons, lanes for workflows, canvases for diagrams, full-viewport sections for decks.
- Use asymmetry, overlap, grids, side notes, sticky summaries, and responsive panels when they clarify the reading path.
- Do not nest cards inside cards.
- Do not build a landing page when the user asked for a tool, report, comparison, or artifact. Put the usable work first.
- Give fixed-format elements stable dimensions with `aspect-ratio`, grid tracks, min/max sizes, or container constraints.
- Ensure all text fits inside buttons, tabs, labels, and cards at mobile and desktop widths.

## Motion and atmosphere

- Use motion for orientation, causality, and emphasis: page load reveal, chart transition, control feedback, slide change, preview update.
- Prefer a few memorable, well-timed moments over constant motion.
- Animate only `transform` and `opacity` unless there is a strong reason.
- Always include a reduced-motion path.
- Background detail should match the subject: subtle paper texture, blueprint lines, data grid, editorial rule system, chart traces, or material shadows.
- Do not add decorative blobs, random orbs, or unrelated bokeh.

## Context fit

- Operational SaaS, CRM, admin, and internal tools should be quiet, dense, scannable, and repeat-use friendly.
- Reports and research should feel editorial and navigable, with strong prose rhythm and useful sidebars.
- Technical diagrams should use a consistent visual grammar and crisp labels.
- Games and playful tools can be expressive, animated, and illustrative.
- Executive summaries should emphasize hierarchy, confidence, and fast scanning over decorative complexity.

## Visual self-review

Before shipping:

- Can you name the visual concept in one sentence?
- Does the design feel specific to this content?
- Is there one memorable detail, not ten competing effects?
- Did you avoid generic generated aesthetics?
- Does the aesthetic serve reading, decision-making, or manipulation?
