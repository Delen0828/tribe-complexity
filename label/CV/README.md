# Discrete data-mark segmentation pilot

Open [index.html](index.html) directly in a browser. The viewer works from a local
folder, including image modes, filters, instance inspection and review export.
No build step, server, external fonts, CDN or hosted account is required.

The sample contains **150 training figures**, 15 in each observed human complexity
interval `[0,10)`, …, `[90,100]`. Sampling uses Python's `random.Random(0)` on
candidates ordered by study index. [selection.json](selection.json) freezes IDs,
scores, source checksums, input-table checksums and native dimensions. No selected
figure is replaced because it is difficult, ambiguous or inapplicable.

## Outputs and interpretation

- [results/summary.md](results/summary.md): descriptive results by complexity band.
- [results/counts.csv](results/counts.csv): figure-level numeric counts/statuses.
- [results/manifest.json](results/manifest.json): static viewer index and provenance.
- `results/runs/<base-run-id>/figures/`: detailed JSON/RLE and file-compatible JS.
- `results/runs/<base-run-id>/assets/`: native overlays, transparent marks-only
  images and thumbnails.
- `results/runs/<base-run-id>/candidates/`: original raw mask receipts, including
  duplicates and excluded candidates. These large intermediate files are ignored
  by Git but remain available locally. Static detail exports omit duplicate masks
  and reference the raw receipt for complete auditing.
- `preparation/figures/`: model-assisted panel boxes, discrete mark families,
  exclusions, exemplars, applicability and unresolved-section notes.
- `results/runs/<base-run-id>/triage/`: cheaper-subagent retain/reject/ambiguous
  decisions, reasons, candidate-set identities and numbered visual sheets.

A count is the number of retained masks judged to represent **single discrete
graphical data marks**. Points, glyphs, bar segments, pie slices, network nodes,
data icons, separable heatmap cells and discrete mapped regions can count. Text,
legends, decoration, axes, continuous curves/areas and photographic/microscopy
content are excluded; error bars belong to their associated mark. Counts concern
drawn instances across panels, not unique underlying observations.

`complete` means preparation and candidate triage finished without recorded
ambiguities, not that the extraction is correct. `partial` flags ambiguous masks,
unresolved preparation sections or failed panels. Numeric partial counts exclude
those unresolved items. `not_applicable` means no applicable discrete unit was
prepared; it is **not** proof that the source has no data. `0` means an applicable
processed figure has zero retained masks. Blank counts mean N/A, pending work or
failure; the accompanying status disambiguates them. Model confidence is mask
quality, not a probability of human visibility.

Preparation and triage were performed by `gpt-6-luna` subagents, with schema and
coordinate validation. They are provisional model-assisted annotations. Human
quality assessment happens post hoc in the viewer. There is no human ground
truth, MAE, instance accuracy, color K, or edge E estimate in this pilot. Thresholds
are fixed engineering defaults, not thresholds fitted against human judgments.

## Reproduce

From the repository root:

```sh
bash label/CV/setup.sh
label/CV/.conda/bin/python label/CV/run.py select
label/CV/.conda/bin/python label/CV/benchmark.py
label/CV/.conda/bin/python label/CV/run.py segment
label/CV/.conda/bin/python label/CV/run.py sheets
# Complete model-assisted visual triage following PREPARATION.md.
label/CV/.conda/bin/python label/CV/run.py render
label/CV/.conda/bin/python label/CV/run.py verify
label/CV/.conda/bin/python -m unittest discover -s label/CV -p 'test_*.py' -v
```

The Conda environment, caches, checkpoint and upstream checkout live here. Setup
pins Python 3.12, PyTorch 2.10.0/cu128, TorchVision 0.25.0 and SAM 2 source commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4`. `requirements.lock.txt` records the full
installed environment. The optional connected-component CUDA extension is
disabled; the selected mask pipeline does not require it. GPU processes need
host device access; this workspace's filesystem sandbox cannot see CUDA devices.

Segmentation uses SAM 2.1 Hiera Small: a 64×64 prompt grid, batch size 16,
predicted-IoU cutoff 0.8, stability cutoff 0.95, no crop layers, and geometric
exemplar-box seeds. Box NMS is disabled; masks with IoU ≥0.9 are deduplicated.
Distinct overlapping masks are retained. Further duplicate alternatives are
rejected during visual triage. These settings are frozen in `config.json`.

Reference RGB conversion preserves geometry; transparency is composited over
white. Inference crops panels at native coordinates and records SAM's internal
image size. No additional tiling variant or synthetic upscaling is used. Output
masks are mapped back to original coordinates and use bounding-box-local,
column-major RLE: `origin: [x,y]`, `size: [height,width]`, alternating background
and foreground `counts` starting with background, even when that first run is 0.

Unchanged per-figure receipts resume automatically. Preparation changes invalidate
that figure's receipt; changed candidate sets invalidate their visual triage.
Use `--ids 2225.png,4647.png` for bounded stages, `--watch` to process preparation
as it arrives, or `--force` to recompute masks. Run IDs include selection, model,
configuration and checkpoint; final analysis identities also include preparation,
candidate outputs and triage decisions. Historical run directories are preserved.

After correcting panel/exemplar geometry, run:

```sh
label/CV/.conda/bin/python label/CV/reconcile.py
```

This archives affected raw receipts and old triage, regenerates only changed
figures, and refreshes their numbered sheets. The correction audit is saved in
`results/preparation-corrections.json`; re-triage those new candidate sets before
rendering the final exports.

The one-/two-worker benchmark is in `results/benchmark.json`. The measured pilot
uses one persistent GPU worker because two were less than 5% faster and exceeded
the selected memory budget on this machine. CPU rendering uses four workers.
`segment --workers 2` remains available for machines where that is appropriate.

## Visual preparation and quality assessment

[PREPARATION.md](PREPARATION.md) defines the annotation and triage contracts. Visual
preparation is an explicit stage, not an automatic semantic capability of SAM 2.
SAM 2 accepts geometric prompts; its automatic masks do not exhaustively propagate
an exemplar concept. Cheaper visual subagents classify numbered candidate sheets
using the original context and exemplars. Merged groups and uncertain fragments
remain ambiguous; no count is invented for missing individual instances.

Browser review quality, issue flags and notes are stored separately for each final
analysis ID. Use **Export reviews** to keep a portable JSON copy; import validates
the run and image IDs. Review does not modify masks or counts. Native view uses
one source pixel per CSS pixel; fit view is a browsing convenience. Neither mode
reproduces a controlled physical monitor size or viewing distance.

Browser QA uses an installed Playwright/Chromium:

```sh
NODE_PATH=/home/delen/browser-tool/node_modules node label/CV/test_browser.cjs
```

On another machine, install Playwright and its Chromium browser, or set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE`. Test screenshots and downloads stay in `cache/`;
the verification receipt is in `results/browser-verification.json`.

## Switch to SAM 3 after checkpoint approval

Install the official `facebookresearch/sam3` package into this environment, pin its
Git revision, and obtain the approved checkpoint through your secure local
account. Keep credentials out of project files and exports. Then run:

```sh
label/CV/.conda/bin/python label/CV/run.py segment --model sam3 --checkpoint /path/to/sam3.pt
label/CV/.conda/bin/python label/CV/run.py sheets
# Triage this new backend's candidate set before counting.
label/CV/.conda/bin/python label/CV/run.py render
label/CV/.conda/bin/python label/CV/run.py verify
```

The SAM 3 adapter reuses the selection, panel boxes, text prompts and exemplar
boxes. It uses text-plus-exemplar concept segmentation and native-coordinate
exports, with per-mask CPU interpolation to avoid a full native-resolution GPU
mask stack. The adapter is implemented but cannot be validated with real SAM 3
weights until access is approved. Its results get a separate run and separate
review storage.

Official implementations: [SAM 2](https://github.com/facebookresearch/sam2),
[SAM 3](https://github.com/facebookresearch/sam3). Source figures and published
MASSVIS complexity annotations retain their original rights and provenance.
