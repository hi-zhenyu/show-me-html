# Interactive Tools

Use this reference for one-off editors, triage boards, prompt tuners, data curators, configurators, simulators, calculators, and playgrounds.

## Non-negotiable loop

Every interactive tool must export the user's work. Pick the output that closes the loop:

- Markdown for triage, plans, summaries, and review notes.
- JSON for structured state, settings, datasets, and config.
- CSV for tabular curation.
- Plain prompt text for prompt tuners and instruction builders.
- CSS, JS, or design-token snippets for visual and motion tools.

If there is no export path, add export before adding extra controls.

## Layout

- The work area is dominant.
- Header explains the task in one sentence, then gets out of the way.
- Controls sit close to the thing they affect.
- Export and reset actions stay easy to find without stealing attention.
- Use panels, lanes, inspectors, split views, or canvas layouts based on the task shape.
- Avoid turning a one-off tool into a general product.

## State model

- Preload user-provided data. Do not make the user paste it again.
- Keep state small, inspectable, and serializable.
- Maintain original input separately so export can show changes or diffs.
- Track dirty state after edits.
- Provide undo or reset when edits are easy to make accidentally.
- Persist locally when the environment permits and the user could lose meaningful work on refresh.
- Make persistence visible enough that the user understands what is saved.

## Common tool shapes

### Triage board

- Columns represent decisions: Now, Next, Later, Backlog, Cut, or user-provided buckets.
- Cards include title, source, tags, risk, estimate, and rationale where available.
- Drag and keyboard move should both work when practical.
- Export grouped markdown plus rationale and any changed metadata.

### Feature flag or config editor

- Group toggles by area.
- Show dependency warnings when a prerequisite is off.
- Highlight changed keys.
- Export only the diff by default, with an option for full JSON when useful.

### Prompt tuner

- Split view: prompt editor, test inputs, output preview, and scoring notes.
- Make variables explicit chips or fields.
- Provide copy as prompt, copy as JSON, and reset to original.
- Preserve evaluation notes separately from prompt text.

### Parameter playground

- Controls should map directly to visible output.
- Sliders need numeric inputs or exact value display.
- Show generated config live.
- Include presets and reset.
- Respect reduced motion for animation playgrounds.

### Dataset curator

- Use filters, sort, tags, and batch operations.
- Surface skipped, invalid, duplicate, or uncertain records.
- Export selected rows and a skipped-record report.
- Avoid silent data loss.

## Feedback and failure

- Copy buttons should show success or failure with `aria-live`.
- Invalid actions should explain what to fix.
- Empty states should say what can be done next.
- Destructive actions need confirmation or undo.
- Loading states need labels and should not freeze unrelated controls.
- If an export excludes anything, say exactly what was skipped.

## Export design

- Put export code in a small, testable function.
- Include metadata when useful: generated time, source count, changed count, skipped count.
- Make the exported text readable and pasteable, not just machine-valid.
- Offer download only when copying is not enough.
- Keep exported formats stable and obvious.

## Final tool check

- Can the user complete the task without reading instructions?
- Can every mouse action be understood from visible controls?
- Can keyboard users reach all important controls?
- Can the user recover from mistakes?
- Can the result be copied into the next workflow?
