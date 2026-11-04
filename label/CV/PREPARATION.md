# Visual preparation and triage protocol

All measurements use the native pixel dimensions of the reference images in
`results/images/`. These have the source geometry, RGB colors, and transparency
composited over white. Model resizing is separate from this reference condition.

Prepare one JSON file per figure at `preparation/figures/<image_id>.json`.
Inspect every image visually. Published chart labels are hints, not ground truth.
Keep each independently interpretable plot panel separate where feasible. Include
non-applicable panels, and explain when a complex section remains unresolved.
Panel boxes should describe data displays; keep captions and legends outside or
add explicit exclusion boxes. Coordinates are absolute original-image pixels,
with exclusive upper bounds. Exemplar boxes tightly surround individual marks,
not entire groups. Do not invent precise boxes for unresolved tiny marks.

Record shape:

```json
{
  "figure_id": "123.png",
  "prepared_by": "gpt-6-luna",
  "notes": "Visual preparation; not a human reference annotation.",
  "panels": [{
    "panel_id": "p01",
    "bbox": [10, 20, 300, 250],
    "panel_type": "chart",
    "applicable": true,
    "status": "prepared",
    "notes": "Three series of discrete bars; error bars are attributes.",
    "exclusions": [],
    "mark_families": [{
      "name": "bar_segment",
      "prompt": "rectangular data bars",
      "count_unit": "one bar or stacked bar segment",
      "exemplar_boxes": [[40, 150, 55, 220]]
    }]
  }]
}
```

Allowed families: `point`, `glyph`, `bar_segment`, `pie_slice`, `network_node`,
`data_icon`, `cell`, `region`, `other_discrete`. Count graphical units encoding
data. Continuous curves and areas, text/numbers, legend symbols, axes, ticks,
gridlines, error bars, decoration, photographs, microscopy, and blots are excluded.
Heatmap cells and discrete choropleth regions can count if separately delineated.
Non-applicable panels have `applicable: false`, empty `mark_families`, and a reason.
Uncertain panels must be marked `status: unresolved`; do not silently omit them.

Candidate triage uses numbered sheets with original context and masked crops.
Every unique candidate receives one of `retain`, `reject`, or `ambiguous`, a
family for retained instances, and a short reason. Retain a single discrete data
mark only. Reject duplicates, text, legends, background, decoration, and irrelevant
continuous marks. A mask spanning several marks is `ambiguous` (reason: merged),
not one retained instance. Suspicious fragments are ambiguous (reason: split).
Subagents' classifications are model-assisted pilot decisions, never human ground
truth. Human post-hoc review is stored separately and does not change counts.
