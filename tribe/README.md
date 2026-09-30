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

Outputs default to `tribe/outputs/spectrum/`: `index.html`, `spectrum.png`,
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

Other options: `--sample-only` saves selection without inference, `--per-bin N`
changes sample size, `--background '#RRGGBB'` changes padding, and `--labels PATH`
uses another compatible label table. Omit `--predict` to rebuild grouped reports
from existing predictions. Missing source images are listed in selection metadata;
insufficient sample-bin populations fail instead of silently sampling duplicates.

Serve the repository root with `python -m http.server 8000 --bind 127.0.0.1`, then
open [the grouped brain explorer](../visualizer/brain.html). It offers five grouping
attributes: perceived complexity, chart count, distinct-color count, quantitative
variable count, and categorical variable count. Counts are exact published labels;
complexity uses `[0,10)`, …, `[90,100]`. Each group shows its size and member images.
The two map modes are the equal-stimulus mean at t=0 and that mean minus the mean
of the entire selected set (including the group). All groups share a symmetric
scale within each mode. These are descriptive model predictions, not measured
fMRI or significance tests. The balanced sample is not population-weighted.

Outputs go to `tribe/outputs/massvis_sample/` or `massvis_all/`. `explorer.json`
contains group membership and links to four-view cortical maps; `aggregate_maps.npz`
contains the numeric group means, contrasts, grand mean, and individual t=0 maps.
`timing.json` estimates full-dataset prediction time from uncached preparation,
inference, and saving, plus measured model setup. Feature-cache hits are excluded
from the per-stimulus estimate; rendering is timed separately. Estimates assume
the same device and software and exclude downloads and interruptions.

For custom output folders, open `visualizer/brain.html?data=../path/to/explorer.json`.
Static deployment must include the report folder, `visualizer/`, annotations, and
original images with their relative paths intact. Generated outputs remain ignored
by Git and are not automatically published.

```sh
python -m unittest discover -s tribe -p 'test_*.py'
python tribe/verify_dataset.py tribe/outputs/massvis_sample
```
