---
name: show-me-html
description: Create self-contained HTML artifacts when a request benefits from spatial layout, visual hierarchy, diagrams, formulas, data charts, evidence, interaction, presentation, or an exportable one-off tool. Use for non-trivial docs, writeups, plans, specs, reports, explainers, summaries, comparisons, reviews, PR descriptions, mockups, diagrams, flowcharts, decks, slides, status updates, incident reports, research, quantitative analysis, playgrounds, editors, and tools. Stay in markdown for short chat replies, code-only answers, terminal-style instructions, and disposable summaries that are only a few bullets.
---

# Show Me HTML

Use this skill when HTML will make the shape of the work clearer than markdown. The goal is not "always output HTML"; the goal is to choose HTML when layout, hierarchy, color, motion, diagrams, or interaction materially improve understanding or action.

## Decide the medium

Reach for a self-contained `.html` artifact when any of these are true:

- **Comparison:** two or more options must be weighed side by side.
- **Spatial structure:** timelines, diffs, maps, flows, states, before/after, architecture, or call graphs carry meaning.
- **Interaction:** the reader needs to tune, filter, drag, simulate, preview, or feel motion.
- **Reference use:** the result will be navigated non-linearly with tabs, anchors, panels, glossary, or filters.
- **Visual hierarchy:** status, severity, ownership, metrics, code, or design tokens need color and structured layout.
- **Quantitative reasoning:** formulas, statistical claims, charts, plots, or tables must be explained and inspected.
- **Evidence:** claims, research, sources, logs, code references, or data provenance must be distinguishable.
- **Round trip:** the reader needs to manipulate data and export the result back to markdown, JSON, prompt text, config, or CSV.
- **Sharing:** the artifact is meant to be read carefully, presented, reviewed, or sent to others.
- **Length:** the answer would become a long linear document.

Stay in markdown for short conversational answers, code snippets, command sequences, small checklists, and files intended to be hand-edited or diffed repeatedly.

## Artifact brief

Before building, silently lock this brief:

- Purpose: what job should the artifact do?
- Audience: who will read or use it, and how much context do they have?
- Content shape: comparison, timeline, report, review, diagram, deck, editor, prototype, or hybrid.
- Visual direction: the deliberate tone and one memorable design idea.
- Interactions: what state changes, if any, must the reader control?
- Chart strategy: for quantitative artifacts, whether ECharts is needed; if not, why a smaller SVG/CSS/table treatment is enough.
- Export path: for any editor/tool, what output format closes the loop?
- Constraints: offline use, privacy, screen size, accessibility, performance, and user-provided style.

## Universal rules

Every HTML artifact must:

1. Be a single self-contained `.html` file unless the user explicitly asks otherwise.
2. Work offline by default: inline CSS and JS; no required runtime server; avoid external assets unless the user provided them or they are clearly optional.
3. Use real layout, not markdown wrapped in tags. Render comparisons as columns, flows as diagrams, timelines as timelines, diffs as diffs.
4. Be readable within five seconds: title, short framing sentence, and clear primary structure above the fold.
5. Use semantic HTML first: `main`, `section`, `article`, `nav`, `table`, `button`, `a`, `label`, and ordered headings before ARIA patches.
6. Be responsive from phone to desktop with stable dimensions for fixed-format UI like boards, grids, slide canvases, charts, controls, and toolbars.
7. Be keyboard usable: visible `:focus-visible`, sensible tab order, real buttons for actions, real links for navigation.
8. Respect motion and comfort: honor `prefers-reduced-motion`, animate only purposeful moments, and avoid `transition: all`.
9. Handle real content: long text, empty states, narrow screens, large numbers, loading, errors, and destructive actions.
10. Export state for every interactive editor/tool. Without export, an editor is incomplete.
11. For factual or evidence-heavy artifacts, keep source notes close to important claims, distinguish fact from inference when it matters, and surface conflicting, stale, skipped, or uncertain data.
12. For data-heavy artifacts with benchmarks, multiple metrics, trend lines, rates, or comparison charts, default to ECharts per `math-and-data.md`. If using inline SVG/CSS/tables instead, state the reason in the final note and keep the underlying data inspectable.

## Reference selection

Read only the references needed for the task:

- `references/patterns.md`: comparisons, plans, reviews, reports, explainers, diagrams, decks, and other artifact shapes.
- `references/visual-direction.md`: typography, color, composition, motion, background detail, and avoiding generic generated aesthetics.
- `references/interface-quality.md`: accessibility, focus, forms, animation, content handling, performance, dark mode, locale, and interaction checks.
- `references/interactive-tools.md`: one-off editors, tuners, triage boards, state, persistence, reset, export, and failure states.
- `references/math-and-data.md`: formulas, equations, derivations, statistics, metrics, charts, plots, tables, and quantitative explainers.

If the request includes frontend design, UI polish, mockups, prototypes, visual exploration, or public-facing presentation, read `visual-direction.md`.

If the artifact contains any interaction, forms, navigation, custom controls, generated images, animation, or dense responsive layout, read `interface-quality.md`.

If the user will manipulate state or data, read `interactive-tools.md` before designing the controls.

If the artifact includes formulas, equations, statistical claims, charts, plots, tables, metrics, or quantitative reasoning, read `math-and-data.md`.

## Build workflow

1. Choose the artifact shape and visual direction from the brief.
2. For quantitative artifacts, decide the chart strategy before implementation: ECharts by default for substantive data visualization; SVG/CSS/table only when the chart is genuinely tiny or a table is clearer.
3. Sketch the layout around the content's structure, not around generic cards.
4. Implement the full single-file HTML with inline CSS and JS.
5. Add interactions only when they help the user understand, decide, or export.
6. Run a self-review against `interface-quality.md`.
7. Deliver the saved path or artifact, plus a concise note on what it contains and any deliberate chart-library deviations.

Bad-looking HTML is worse than good markdown. If the result looks like a generic template, restart the visual direction before shipping.
