# Math and Data

Use this reference for formulas, derivations, equations, statistics, metrics, tables, charts, plots, quantitative explainers, scientific/financial/engineering reports, and data-backed decisions.

## Formula rendering

- Simple inline formulas can use semantic HTML: `<var>`, `<sub>`, `<sup>`, and concise text.
- Complex formulas should use KaTeX or MathJax rather than hand-built HTML.
- Prefer **KaTeX** for speed, compact rendering, and mostly standard TeX.
- Prefer **MathJax** for broader TeX coverage, accessibility extensions, or unusual notation.
- Preserve the source expression near the rendered formula when the user may edit or verify it.
- Give important formulas a plain-language reading immediately nearby.
- Number equations only when they are referenced later.
- Define variables before or directly after the formula; include units and valid ranges.
- Do not place key meaning only inside a screenshot or non-selectable image.

## Library and offline policy

- Keep the artifact single-file when possible. If using KaTeX, MathJax, or ECharts in a strict offline artifact, inline the needed library assets or provide an inline fallback.
- If a CDN is used because inlining would make the artifact impractically large, make the dependency explicit in the final note and keep the page useful when the script fails.
- Treat library choice as an explicit decision, not a convenience shortcut. For data-heavy artifacts, resolve ECharts vs. SVG/CSS/table before coding, then preserve the source data near the chart either way.
- Prefer progressive enhancement: show the raw formula, table, or summary before the library upgrades it.
- Never let a failed chart/formula library leave a blank content area.

## Quantitative explanation

- Separate formula, intuition, assumptions, and result.
- State what is measured, over what population/window, and with what unit.
- Show sample substitution for important formulas, not just the abstract expression.
- For estimates, show sensitivity or confidence when it changes the decision.
- Mark approximations, rounding, normalization, and omitted variables.
- Do not imply precision the data cannot support.

## Data tables

- Use tables for exact values, comparisons, parameters, and source data.
- Use charts for patterns, relationships, distribution, outliers, and change over time.
- Pair important charts with a compact table or data summary when exact values matter.
- Table headers need units where applicable.
- Use tabular numbers and align numeric columns consistently.
- Handle missing, zero, null, estimated, and not-applicable values distinctly.
- Large tables need filtering, sorting, pagination, or virtualization.

## Charts and plots

- Use **ECharts by default** for data charts, especially when the artifact includes multiple charts, benchmark comparisons, multi-series data, trend lines, rates, distributions, filtering, hover inspection, legend toggles, or responsive resizing.
- Use inline SVG/CSS only for genuinely tiny static charts, simple decorative diagrams, or cases where a compact table communicates the data better than a chart.
- If a data-heavy artifact does not use ECharts, say why in the final note. Valid reasons include strict offline size limits, a single trivial chart, no useful interaction, or a table being more accurate than a chart.
- Every chart must answer a named question.
- Title the chart with the conclusion or question, not just the data type.
- Label axes, units, time windows, baselines, and aggregation.
- Use legends only when they reduce ambiguity; directly label series when possible.
- Do not encode meaning by color alone. Add labels, shapes, line styles, annotations, or patterns.
- Use annotations for thresholds, notable events, outliers, and decision boundaries.
- Avoid chartjunk: decorative 3D, arbitrary gradients, excessive gridlines, and unlabeled effects.
- Use responsive containers with stable height and a resize handler.
- Provide empty, loading, and error states for dynamic chart data.

## ECharts implementation notes

- Initialize charts after the container has a stable size.
- Use `ResizeObserver` or a debounced `resize` listener so charts respond to layout changes.
- Keep the source dataset as JSON near the chart code when data is embedded.
- Prefer dataset/encode when it makes the mapping clearer.
- Set accessible labels or adjacent summaries for screen readers.
- For interaction-heavy charts, explain hover, zoom, brush, legend toggle, or filter behavior visibly.
- Dispose chart instances when replacing or rebuilding them in long-lived tools.

## Statistical claims

- Name the statistic: mean, median, percentile, rate, ratio, index, confidence interval, correlation, regression coefficient.
- State numerator and denominator for rates.
- Distinguish correlation, causation, forecast, and assumption.
- Show sample size and missing-data handling when relevant.
- Flag Simpson's paradox, survivorship bias, selection bias, seasonality, and base-rate problems when they could matter.
- Use `Intl.NumberFormat` for numbers and currency in JavaScript.

## Final check

- Can a reader understand the formula without trusting the rendering?
- Can a reader inspect the data behind the chart?
- For data-heavy artifacts, did you use ECharts or explicitly explain why not?
- Are units, axes, denominators, and assumptions visible?
- Is the chart still useful without color?
- Is any uncertainty or approximation honestly labeled?
