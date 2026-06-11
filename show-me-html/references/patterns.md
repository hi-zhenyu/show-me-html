# Artifact Patterns

Use this reference to choose a structure that matches the content. HTML earns its keep when position, contrast, hierarchy, or interaction carries meaning.

## Comparison

Use for options, vendors, designs, approaches, architecture choices, tradeoffs, or recommendations.

- Layout: one column per option for 2-3 options; responsive grid for 4+ options.
- Normalize sections across options: framing, core idea, example, strengths, risks, cost, fit, verdict.
- Put hard metrics in the same row across options so scanning is fair.
- End with a recommendation or decision matrix; do not leave the reader to infer it.
- Avoid stacked prose sections that force the reader to hold every option in memory.

## Exploration and planning

Use for implementation plans, strategy options, project sequencing, technical approaches, or roadmap work.

- Start with the destination: goal, constraints, success criteria, and non-goals.
- Show phases as a timeline or lanes, not as a wall of bullets.
- Include a data-flow or system-flow diagram when more than two components interact.
- Separate decisions already made from open questions.
- Add acceptance checks near the relevant phase, not only at the end.

## Code review and PR writeup

Use when diffs, modules, ownership, risk, or reviewer focus matters.

- Treat the diff or module map as the spine of the page.
- Put annotations in the margin or side panel so code remains readable.
- Group findings by severity and affected area.
- Include "where to focus review" and "what changed" as short framing blocks.
- For module maps, show boundaries, dependencies, and hotspots visually.
- Avoid repeating entire files with prose between every chunk.

## Reports, research, and explainers

Use for status reports, incident reports, concept explainers, feature deep-dives, or learning material.

- Start with the answer: summary, status, or core insight.
- Use sidebars for glossary, assumptions, data sources, and related links.
- Use timelines for incidents and progress reports.
- Use live demos, sliders, or small simulations when behavior matters.
- Separate "what happened", "why it matters", and "what changes next".
- Avoid decorative charts; every visual must answer a question.

## Diagrams and illustrations

Use when structure is clearer as a drawing.

- Prefer inline SVG for flowcharts, architecture, state machines, process maps, and technical illustrations.
- Keep text selectable and meaningful; use `<title>` and `<desc>` for SVGs.
- Draw the main path boldly and secondary/error paths quietly.
- If there are many details, make the diagram navigational and show detail in a panel.
- Avoid diagrams with so many nodes that layout stops communicating.

## Design systems and prototypes

Use when the request is about UI, tokens, components, motion, or interaction feel.

- Render actual component states, not screenshots or descriptions.
- Show variants in a contact sheet: size, intent, state, density, theme.
- For motion, expose timing/easing controls and show the resulting CSS/JS value.
- For flows, make only the decision points high fidelity; keep supporting screens lightweight.
- Include copy buttons when the artifact generates tokens, config, CSS, or specs.

## Decks and presentations

Use when the artifact will be walked through live.

- Build each slide as a full viewport section with arrow-key navigation.
- Keep one idea per slide.
- Vary slide types: thesis, chart, quote, diagram, before/after, decision, appendix.
- Show progress and current slide number.
- Support fullscreen and readable speaker-mode notes only if useful.
- Avoid making every slide the same card with different text.

## Pattern check

Before shipping, ask:

- Does the layout reveal the structure faster than markdown?
- Is the primary action or conclusion visually obvious?
- Would a reader know where to look first, second, and third?
- Is any interactivity essential, or is it just decoration?
- Can the artifact survive narrow screens and long content?
