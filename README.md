# show-me-html

![Banner reading "Markdown is cheap. Show me HTML." above a Markdown document expanding into a rich HTML artifact dashboard](assets/show-me-html-banner.png)

Create self-contained HTML artifacts when a normal chat answer or Markdown file is not enough.

`show-me-html` is an Agent Skill for turning dense content into readable, navigable, and useful single-file HTML: comparisons, technical explainers, research reports, review notes, diagrams, dashboards, interactive tools, chart-heavy analysis, and presentation-like pages.

The skill is intentionally lightweight. The main `SKILL.md` decides when HTML is worth using, then loads focused references only when the task needs them.

## Install

Install with the `skills` CLI:

```bash
npx skills add hi-zhenyu/show-me-html
```

Then describe the HTML artifact you want. You usually do not need to mention the skill name explicitly; agents that support automatic skill activation should pick it up when you ask for HTML, visual layout, diagrams, charts, or interaction.

```text
Turn the solution comparison into a browsable HTML report.
```

If automatic activation does not happen in your agent, invoke it directly with `$show-me-html`.

If your agent supports local skill folders, you can also install manually (Codex as an example):

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R show-me-html "${CODEX_HOME:-$HOME/.codex}/skills/show-me-html"
```

Restart or refresh your agent after manual installation if it does not pick up new skills automatically.

## When to use it

Use `show-me-html` when the answer benefits from layout, hierarchy, visualization, interaction, or sharing:

- Side-by-side comparisons and decision matrices
- Research summaries, technical explainers, and long reports
- Timelines, flows, architecture diagrams, state maps, and process views
- Data-heavy pages with charts, formulas, tables, or benchmarks
- Interactive one-off tools such as prompt tuners, triage boards, calculators, or configuration editors
- Public-facing summaries, PR writeups, reviews, incident reports, and presentation-style pages

Stay in Markdown for short answers, simple command lists, small checklists, code-only replies, and files that should remain easy to diff by hand.

## Example prompts

```text
Make a visual HTML comparison of these three implementation options.
```

```text
Turn this PDF summary into a technical deep-dive HTML page with diagrams and source notes.
```

```text
Build a small interactive HTML prompt tuner. It should export the final prompt as Markdown.
```

```text
Create a chart-heavy benchmark report as a single HTML file. Keep the source data inspectable.
```

## Important note

This skill improves the form, structure, and readability of an agent's output. It does not replace the user's actual task requirements.

For example, if you ask an agent to explain a paper, this skill can help turn the explanation into a clearer HTML artifact with sections, diagrams, charts, source notes, and navigation. But the depth, rigor, evidence, and usefulness of the explanation still depend on what you ask the model to do.

Use this skill to maximize how well the model presents its work, not as a substitute for asking for the right analysis.

HTML artifacts also cost more than Markdown. They usually require more agent work time and more tokens, so use them when the improved presentation, navigation, or interaction is worth the extra cost.

## What it produces

By default, the skill asks the agent to produce one self-contained `.html` file:

- Inline CSS and JavaScript so the artifact works offline where practical
- Semantic HTML, responsive layout, keyboard-friendly controls, and visible focus states
- Evidence and source notes near important claims when the content is factual
- ECharts by default for substantive data visualization
- KaTeX or MathJax when complex formulas need proper rendering
- Export paths for interactive tools so the user's work can move into the next workflow

External libraries are not required to install this skill. A generated artifact may use ECharts, KaTeX, or MathJax when they are the right tool for the job; the skill instructs the agent to keep fallbacks or dependency notes visible when strict offline use matters.

## Examples

This repository includes an English example in both Markdown and HTML, so you can compare the same content as a linear document and as a richer HTML artifact:

- Markdown source: [`examples/deepseek_v4_deep_dive_en.md`](examples/deepseek_v4_deep_dive_en.md)
- Rendered Markdown on GitHub: [`deepseek_v4_deep_dive_en.md`](https://github.com/hi-zhenyu/show-me-html/blob/main/examples/deepseek_v4_deep_dive_en.md)
- HTML artifact source: [`examples/deepseek_v4_deep_dive_en.html`](examples/deepseek_v4_deep_dive_en.html)
- Live HTML artifact on GitHub Pages: [`deepseek_v4_deep_dive_en.html`](https://hi-zhenyu.github.io/show-me-html/examples/deepseek_v4_deep_dive_en.html)

The examples are illustrative outputs, not required inputs. They show the expected level of structure: an answer-first opening, navigable sections, visual hierarchy, source notes, and a complete HTML file that can be opened locally.

GitHub Pages is deployed from `main` with the workflow in `.github/workflows/pages.yml`. If you fork this repository, enable Pages with the GitHub Actions source to publish your own live example URL.

## Repository structure

```text
.github/workflows/pages.yml
.nojekyll
assets/
show-me-html/
  SKILL.md
  references/
examples/
README.md
LICENSE
.gitignore
```

- `show-me-html/SKILL.md` is the skill entry point and trigger description.
- `show-me-html/references/` contains focused guidance that the agent reads only when needed.
- `assets/` contains README and project media.
- `examples/` contains sample Markdown and HTML outputs.
- `.github/workflows/pages.yml` publishes the live HTML example to GitHub Pages.
- `.nojekyll` keeps GitHub Pages from applying Jekyll processing when publishing static files.

## Acknowledgements

This skill was inspired by Thariq Shihipar's essay [The Unreasonable Effectiveness of HTML](https://thariqs.github.io/html-effectiveness/), especially the observation that Markdown can become "a restricting format" as agents become more capable.

This skill's prompt design also references ideas from the [`dogum/html-artifacts`](https://github.com/dogum/html-artifacts) skill.

## License

MIT. See [`LICENSE`](LICENSE).
