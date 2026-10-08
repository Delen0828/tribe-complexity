# TRIBE v2: MASSVIS 751 and 4849

The current experiment follows the stimulus-processing protocol in *Can a Neural
Encoding Model Replicate an fMRI Visualization Study?*, Sections 3.2-3.3. It uses
**3-second silent static videos**, **proportional fitting into a centered 256 × 256 area with outer padding**,
and **the first predicted cortical response (t=0)** instead of a temporal average.

Open [the cortical report](index.html) or [the two-page PDF](outputs/comparison.pdf).
The report includes the original PNG stimuli above cortical response maps, a signed contrast, separate positive
and negative dorsal contrasts, and a whole-brain demeaning sensitivity check.
There are no time-series charts or metric dashboards.

The detailed [processing comparison table](processing_comparison.md) distinguishes
what the paper specifies, what our original run did, and what changed. The original
12-second results and scripts are preserved in `archive/original_12s/` as historical
artifacts; the top-level scripts and `outputs/` are the current protocol.

## Current protocol

| Image | Source | Prepared dimensions |
| --- | --- | --- |
| 751 | `data/science/v488_n7413_3_f2.png` | 292 x 292 canvas |
| 4849 | `data/government/whoH08_2.png` | 292 x 292 canvas |

- Hold the RGB image constant for 3 seconds, with no audio or separate task text.
- Use 16 fps, standard H.264 High profile, YUV420, fixed QP 10, all-intra frames, explicit BT.709 color
  metadata and limited range. The papers do not specify FPS or codec. Chroma
  subsampling introduces small pixel differences; this is not lossless RGB.
- Fit the entire source proportionally within 256 × 256, centered on a 292 × 292
  RGB canvas. The standard V-JEPA2 processor removes the 18-pixel outer border,
  preserving the full stimulus and any letterboxing. Padding defaults to white;
  use `run.py --background "#RRGGBB"` to match the experiment. Normalization
  remains exclusively in the pretrained processor. This changes the paper’s
  spatial preparation and standard center-crop behavior; it does not reconstruct
  original experimental display sizes or positions.
- Run pretrained `facebook/tribev2`, with video-only events and independent
  timelines. Use the native population-level/average-subject prediction.
- Preserve native 5-second hemodynamic compensation, 100-second model window,
  padding, and 2 Hz feature sampling. Output is 1 Hz on fsaverage5 (20,484 vertices).
- Save all three returned samples, but select **only sample 0** for the report.
- Compute `751 - 4849` at every vertex. The sensitivity contrast subtracts the
  spatial mean of that difference, following the supplement's demeaning check.
- Both stimulus maps share symmetric limits. Directional dorsal panels share one
  data-derived positive scale, including the demeaned panels, without saturation
  at the supplement's dataset-specific 0.15 limit.

TRIBE's official visual extractor supports CUDA and CPU, but not Apple's MPS device.
Use `--device cuda` on an NVIDIA GPU; the historical macOS run used CPU.
Byte-identical visual windows are cached by SHA-256; mean-token pooling
matches the native extractor. No pretrained weights or upstream code are changed.
A protocol-specific feature cache prevents accidentally reusing 12-second features.

## Run and verify

Dependencies are maintained in the Conda environment `tribe`. Conda supplies
Python 3.11; the environment's pip installs the verified TRIBE/CUDA packages using
`requirements-linux-cuda-lock.txt`, referenced by the root `environment.yml`.

Conda is installed through Miniforge at `~/miniforge3`. In a new Bash terminal:

```sh
conda activate tribe
python tribe/run.py --device cuda
python tribe/compare.py
python tribe/verify.py
```

If the current terminal does not recognize `conda`, first run
`source ~/miniforge3/etc/profile.d/conda.sh`. For scripts without activation, use
`conda run --no-capture-output -n tribe python tribe/run.py --device cuda`.

To recreate the environment on Linux x86_64, run from the repository root
(clone upstream only if it is not already present):

```sh
git clone https://github.com/facebookresearch/tribev2.git tribe/upstream
git -C tribe/upstream checkout af58661791a351a448a489042a28f6c37e1c14b7
conda env create -f environment.yml
```

After editing dependency pins, update with
`conda env update -n tribe -f environment.yml` from the repository root.
Select `~/miniforge3/envs/tribe/bin/python` as the IDE's Python interpreter.

Use `--device cpu` if CUDA is unavailable. The first inference downloads the model
weights and visual encoder; subsequent runs reuse the local cache.
`requirements-lock.txt` is the historical macOS snapshot; use the Linux/CUDA lock
referenced by `environment.yml` for this environment.

Model downloads and extracted features live under `cache/`; Neuralset also uses `~/.cache/neuralset`. Metadata
records input/video/PDF hashes, stimulus placement and encoder crop boxes, model revisions, and inference settings.
`verify.py` checks every decoded video frame against color-error tolerances for the expected padded frame,
uses AVFoundation on macOS to independently verify native decoding, and checks
absence of audio, duration, predictions, t=0 selection, contrast arithmetic, and
report contents. Rendered PDF pages are also inspected visually.

## Files

- `processing_comparison.md`: source-referenced methods table and scope limits.
- `inputs/manifest.json`, `inputs/{751,4849}.png`: original source copies/provenance.
- `inputs/paper_3s_t0_bt709_center256_v3/<RGB hex>/`: padded PNGs, videos, and preparation metadata.
- `outputs/prediction_*.npz`: raw 3 x 20,484 predictions and times 0, 1, 2.
- `outputs/selected_maps.npz`: t=0 maps, signed contrast, demeaned contrast.
- `outputs/comparison.png`: original stimuli, primary cortical maps, and signed contrast.
- `outputs/directional_contrasts.png`: dorsal direction/de-meaning maps.
- `outputs/comparison.pdf`: both neuroimaging figures in one report.
- `outputs/comparison.json`: descriptive spatial metrics for audit, not displayed.
- `outputs/run_metadata.json`, `outputs/events_*.csv`: run provenance.
- `predict-cuda.log`, `compare-cuda.log`, `verification.log`: local execution and verification logs.

## Interpretation

The paper used 60 matched Bubble/Surface pairs per experiment and evaluated
human-study effects by region. MASSVIS 751 and 4849 are **two unrelated images**,
so this is an adaptation of its processing protocol, not a replication of its
experimental comparison. We cannot estimate its stimulus-group confidence intervals
or claim a complexity effect, anatomical significance, or agreement with measured
human fMRI from this pair. No ROI/Destrieux group analysis is claimed.

Values are native model units, not percent BOLD or complexity scores. Vertex order
is left hemisphere (10,242) then right hemisphere (10,242). The signed and demeaned
maps are descriptive predictions, with no significance threshold.

Sources: [main paper](../Can_a_Neural_Encoding_Model_Replicate_an_fMRI_Visu.pdf),
[supplement](../supplemental_materials.pdf),
[TRIBE code](https://github.com/facebookresearch/tribev2),
[weights](https://huggingface.co/facebook/tribev2).
TRIBE's upstream license is CC BY-NC 4.0.

## MP4 color compatibility fix

The earlier RGB H.264 High 4:4:4 Predictive files decoded accurately in FFmpeg but
failed the macOS AVFoundation decoder. They have been replaced with widely supported
YUV420 H.264 files, and predictions were recomputed using a new feature cache.
The previous RGB run is preserved in `archive/rgb_3s_before_color_fix/`. Older public
MP4 paths link to the corrected clips. Original source PNGs remain unchanged.
`outputs/color_validation.json` records decoder-versus-source pixel error;
`verify_macos.swift` checks the independent macOS decoding path.

## Complexity spectrum

Sample one available original stimulus from each of ten equal-width bins of
`mean_human_complexity_rating` (0–100), predict its response, and render the report:

```sh
python tribe/compare.py --mode spectrum --predict --device cuda --seed 0
```

Use `--device cpu` for CPU inference. The report has ten rows ordered from bin 10
(most complex) to bin 1 (least complex). Each row shows the original stimulus,
left lateral, left medial, right lateral, and right medial cortical views.
All maps use t=0 and the same symmetric color scale. Bins are `[0,10)`,
`[10,20)`, …, `[90,100]`; these are rating intervals, not population deciles.
Sampling is uniform within each bin among images available locally. An empty bin
stops the command rather than silently substituting another bin.

To inspect the sample before inference or render existing predictions:

```sh
python tribe/compare.py --mode spectrum --sample-only --seed 0
python tribe/compare.py --mode spectrum --seed 0
```

Outputs default to `tribe/outputs/spectrum/`: `index.html` (redirect to the shared
`outputs/web/` viewer), `spectrum.png`,
`spectrum.pdf`, `spectrum_maps.npz`, `selection.json`, source image copies,
and prediction/provenance files. `--labels PATH` accepts the same CSV schema as
`label/output/labels.csv`. Use a different `--output-dir PATH` for a different
seed or label selection; reuse the same options when rendering. The original
pair report remains available with `python tribe/compare.py`.

## Centered stimulus preprocessing

The new protocol uses a separate feature cache, including the padding color.
Existing predictions and reports still describe their original preprocessing until
rerun. `--prepare-only` writes frames, clips, and `preparation_metadata.json` without
model inference. Test geometry and the actual encoder processor with:

```sh
python -m unittest discover -s tribe -p test_preprocessing.py
```

Encoder defaults: [V-JEPA2 processor configuration](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256/blob/main/video_preprocessor_config.json).

## Dataset runs and grouped brain explorer

Activate the `tribe` environment. Sample **10 different stimuli in each of the
10 fixed complexity bins** (100 total), predict them, and build the web assets:

```sh
python tribe/dataset.py --scope sample --predict --device cuda --seed 0
```

Predict all **5,800 annotated MASSVIS stimuli**, reusing the completed sample:

```sh
python tribe/dataset.py --scope all --predict --device cuda \
  --reuse-from tribe/outputs/massvis_sample
```

Omit `--reuse-from` for an independent full run. Rerun either command to resume:
each completed prediction has an atomic receipt with source/preprocessing identity,
video/prediction hashes, and timings. Only validated matching results are reused.
Use a new `--output-dir` for a different seed, selection, or background. The
full run covers the published annotated dataset; extra unannotated local images
cannot be assigned complexity bins or feature groups and are excluded.

If inference stops before finishing, the normal report command fails rather than
silently presenting incomplete results. To inspect completed predictions explicitly:

```sh
python tribe/dataset.py --scope all --allow-partial
python tribe/web.py
```

Partial reports show completed/selected counts and record pending indices in
`explorer.json`. Every included prediction still passes identity, hash, shape, and
time checks. Means and contrasts use only completed stimuli, so partial groups
may be biased. Resume inference and rebuild to replace the partial report with the
complete dataset. Report-only generation does not rerun the model.

Other options: `--sample-only` saves selection without inference, `--per-bin N`
changes sample size, `--background '#RRGGBB'` changes padding, and `--labels PATH`
uses another compatible label table. Omit `--predict` to rebuild grouped reports
from existing predictions. Missing source images are listed in selection metadata;
insufficient sample-bin populations fail instead of silently sampling duplicates.

Choose rendered views with `--views four` (left/right lateral and medial),
`--views inferior` (one bottom view showing both hemispheres), or `--views all`
(all five, the default). The **Cortical views** browser selector switches between
the layouts included in the report, and saves the choice in `?views=inferior`,
`?views=four`, or `?views=all`. To add the bottom view to existing predictions:

```sh
python tribe/dataset.py --scope all --allow-partial --views all
```

This rebuilds the maps without running inference. Use `--views inferior` to render
only the single bilateral ventral map per group and map mode. Reports generated
with a single layout offer that layout in the browser; older reports retain their
four standard views until rebuilt. All layouts use the same scale within each map mode.

Serve the repository root with `python -m http.server 8000 --bind 127.0.0.1`, then
open [the output visualization hub](outputs/web/index.html). The hub discovers grouped
reports and spectrum reports in sibling output folders, and links the two-stimulus
comparison. Folders without a supported report are shown as unavailable. The attribute dropdown
shows a grid with **one column per group and one, four, or five cortical-view rows**, keeping
all groups visible together. The wide grid scrolls horizontally to preserve all group columns
and the selected view rows. The [grouped brain explorer](../visualizer/brain.html) uses the same
layout and also lets you switch between sample and full-dataset results.
Both offer all 29 published feature variables, perceived complexity, and source
category (31 grouping attributes). These include counts of charts, chart types,
distinct colors, quantitative variables, and categorical variables; all text,
color, panel, and chart-type presence labels; and Government, Infographic, News,
and Science source categories. Counts are exact published labels, binary features
use No/Yes groups, and complexity uses `[0,10)`, …, `[90,100]`.
Each column shows sample size as an array of squares (one square per stimulus)
above its short attribute label. Select the label to inspect member images below the grid.
The two map modes are the equal-stimulus mean at t=0 and that mean minus the mean
of the entire selected set (including the group). All groups share a symmetric
scale within each mode. These are descriptive model predictions, not measured
fMRI or significance tests. The balanced sample is not population-weighted.

Rebuild existing reports to add these grouping attributes without repeating
inference. Existing selections can gain annotations when their labels, stimuli,
and prediction settings are unchanged:

```sh
python tribe/dataset.py --scope sample --seed 0
python tribe/dataset.py --scope all --allow-partial
```

Outputs go to `tribe/outputs/massvis_sample/` or `massvis_all/`. Each report contains
data and rendered maps. Shared HTML, JavaScript, and CSS live in
`tribe/outputs/web/`; legacy folder `index.html` files redirect to that viewer. `explorer.json`
contains group membership, available view layouts, and links to cortical maps; `aggregate_maps.npz`
contains the numeric group means, contrasts, grand mean, and individual t=0 maps.
`timing.json` estimates full-dataset prediction time from uncached preparation,
inference, and saving, plus measured model setup. Feature-cache hits are excluded
from the per-stimulus estimate; rendering is timed separately. Estimates assume
the same device and software and exclude downloads and interruptions.

For custom output folders, open `visualizer/brain.html?data=../path/to/explorer.json`.
Refresh the hub after adding or moving reports with `python tribe/web.py`.
For standalone static deployment, serve the `outputs/` directory, including `web/`
and each report folder with
`explorer.json`, `brain_maps/`, and `inputs/*.png`, plus the linked downloads
(`aggregate_maps.npz`, `selection.json`, and `timing.json`). No server-side code,
annotation viewer, or original dataset paths are needed. Keep the sibling directory layout. Generated outputs remain ignored
by Git and are not automatically published.

```sh
python -m unittest discover -s tribe -p 'test_*.py'
python tribe/verify_dataset.py tribe/outputs/massvis_sample
```

## CPU workers and GPU batching

`--thread N` now means **CPU stimulus-preparation workers only** (default 8).
GPU concurrency uses `--batch-size N` (default 1); PyTorch's CPU compute pool has
its own `--cpu-threads N` setting (default 8). All three accept positive integers.
These options are available in `dataset.py`, `run.py`, and the legacy spectrum
command. They do not launch multiple GPU model copies.

```sh
python tribe/dataset.py --scope sample --predict --device cuda \
  --thread 8 --batch-size 1 --cpu-threads 8
python tribe/dataset.py --scope all --predict --device cuda \
  --thread 8 --batch-size 1 --reuse-from tribe/outputs/massvis_sample
```

A single persistent V-JEPA encoder processes batches of independent stimuli.
Every prepared video is decoded and checked for identical frames before its
64-frame window is encoded. Features are pooled over tokens immediately after
each encoder block, preserving the original embedding and pre-final-normalization
hidden states. The unused V-JEPA predictor is skipped. Neuralset's original layer
selection, layer aggregation, temporal sampling, and TRIBE brain model remain in
use. Pooled feature storage is bounded to the current batch; a final smaller batch
and mixed resumed/new batches are supported. GPU forward calls remain on one
execution path with shared weights.

FP32 remains the default. Optional `--precision bf16` enables CUDA autocast for the
video encoder only; use a separate output directory, for example:

```sh
python tribe/dataset.py --scope sample --predict --device cuda \
  --batch-size 2 --precision bf16 --output-dir tribe/outputs/massvis_sample_bf16
```

BF16 is approximate and opt-in; its identity and feature cache are separate from
FP32. Existing validated FP32 predictions remain reusable. Do not mix precisions
within one selection. Reduce `--batch-size` if GPU memory is insufficient. Report
plotting remains serial because Matplotlib has shared mutable state.

Run metadata records precision, worker counts, requested and actual encoder batch
sizes, encoder load count, peak CUDA memory, and processing wall time. Runtime
estimates use wall-clock throughput, not a sum of overlapping per-image work.
The encoder is loaded lazily, so a fully resumed run need not load it at all.

Benchmark against a completed FP32 reference run (up to two images per complexity
bin), recording numerical differences, peak memory, and throughput:

```sh
python tribe/benchmark_gpu.py --reference tribe/outputs/thread_benchmark/thread_8
```

The earlier CPU-only concurrency comparison took 84.21 seconds for 20 stimuli with
8 threads versus 84.33 seconds with 1 thread; it used the previous nonpersistent
encoder and should not be used to estimate the new GPU-batched execution path.

Measured on the RTX 4070 Ti SUPER with the same 20 stimuli as the previous benchmark:

| Encoder execution | 20-stimulus wall time | Peak allocated GPU memory | Estimated 5,800-stimulus time |
| --- | --- | --- | --- |
| Persistent FP32, batch 1 (default) | 55.80 s | 5.20 GiB | 4 h 30 m |
| Persistent FP32, batch 2 | 58.70 s | 5.79 GiB | 4 h 44 m |
| Persistent FP32, batch 4 | 58.51 s | 6.98 GiB | 4 h 43 m |
| BF16 autocast, batch 2 (opt-in) | 24.73 s | 7.16 GiB | 1 h 59 m |

All FP32 configurations matched the original predictions within `rtol=1e-5,
atol=1e-6`; maximum absolute error was below 4.8e-7. BF16 had mean absolute error
0.00110 and maximum absolute error 0.02228 model units, with minimum per-stimulus
correlation 0.99970 across all three timepoints. BF16 therefore remains explicitly
opt-in. Autocast retains FP32 weights and can use more memory than FP32 in this
implementation. Estimates exclude report rendering and downloads and are based
on 20 stimuli, not a completed full-dataset run. Raw measurements are in
`outputs/gpu_benchmark/benchmark.json`.
