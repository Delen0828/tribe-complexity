# Ashcraft controlled visualization study

The implemented experiment is a 72-scatterplot, four-task Psychtoolbox study.
The authoritative sources are `DataVizThesis_Script/DataVizThesis_HAScript.m`,
its `stimSetPilot_4.xlsx` list and `dataVizSetPilot_4/` PNGs, and the numerical
generator in `DataVizThesis_StimGen/DataVizStim(figure)Gen.ipynb`. The proposal
describes an earlier two-task plan; the executable code defines the conditions
reported here. The pilot response logs are behavioral data, not participant fMRI.

## Condition tables

- [study_conditions.csv](study_conditions.csv): 16 permitted task × direction ×
  outlier combinations, prompts, response mappings, pool sizes, and timing.
  `available_images` counts the image pool, not the number shown in that task.
- [visual_conditions.csv](visual_conditions.csv): four actual visual-condition
  pools and their published correlation ranges.
- [study_parameters.csv](study_parameters.csv): generator, presentation,
  counterbalancing, timing, simulation scope, and source evidence.

Each run shows each of the 72 images once. Each task has three blocks of six
images, including nine positive and nine negative plots and nine with and nine
without an outlier. The joint cells are **not balanced**: the whole image pool
contains 24 positive/outlier, 12 positive/no-outlier, 12 negative/outlier, and 24
negative/no-outlier images. Unadjusted direction or outlier marginal contrasts
therefore mix the two visual factors.

The first run uses `XPQMOPMXQOPQMOX`; later runs reverse that order. X means a
12-second fixation. P is passive detection, Q trend judgment, M motion judgment,
and O outlier judgment. A task block starts with a 3-second prompt and shows six
3-second stimuli, each followed by 0.5 seconds of fixation. A nominal run lasts
324 seconds, excluding scanner wait and frame timing. MATLAB uses seed 1234.
All runs after run 1 use the same one-position rotation of the original task
pools; the code does not keep rotating them across successive runs.

Every image moves during every task, traveling left or right through 0.5% of the
screen width in three seconds. Its width/height fit within 80% of the screen.
The background is white. Direction sequences cannot contain more than three
identical directions consecutively. The movement is not unique to task M.

## Lightweight reproduction

Use the existing `tribe` Python environment. NumPy, SciPy, Matplotlib, and Pillow
are already installed; the exporter itself uses only the standard library.
No MATLAB, Colab, Excel engine, Node build, or new package is necessary.

From the repository root:

```sh
conda activate tribe
python SemanticsOfDataViz_HAshcraftGTUgThesis/reproduce.py
```

This exports all 72 authoritative pilot-4 PNGs, **without changing their bytes**,
to `reproduced/pilot4/stimuli/`. It also writes `stimulus_metadata.csv`, SHA-256
provenance, and the three study tables. Filenames, workbook membership, and the
independently supplied RSA metadata agree. The images also match the pilot-4 ZIP.

The supplied rasters are the best source for reproducing this specific pilot.
Their original point coordinates are not supplied. The current notebook is a
different revision: it accepts an added outlier only at Mahalanobis-distance
z ∈ [6,10], whereas pilot-4 filenames report z ∈ [3.042,7.968]. Its source also
sets seed 1. Inferring the pilot's exact old recipe solely from its folder name
would silently change the stimuli.

For a separately labeled, deterministic generation from the **current notebook**:

```sh
python SemanticsOfDataViz_HAshcraftGTUgThesis/reproduce.py --mode generate --seed 1
```

This writes 72 newly rendered PNGs to `reproduced/generated_seed_1/stimuli/`,
full coordinates to `points.npz`, rounded filename metadata plus exact numerical
metrics to `stimulus_metadata.csv`, and version/seed/source-hash provenance.
Use a new `--output` directory for another generation. It is not substituted for
pilot-4 in the simulation. `--z-min` and `--z-max` can explicitly generate other
outlier variants; changing them does not establish the historical pilot recipe.

The port retains the notebook's legacy NumPy RNG and its exact random-call
order. It generates 250 bivariate Gaussian points, orthogonalizes them, then
mixes them to target 72 correlations evenly spaced from +1 to −1. It appends
one additional point in every plot, for **251 points total**. Half the plots
receive an accepted Cauchy candidate; the rest receive a Gaussian point. Final
correlation must retain the target sign, but may differ substantially in
magnitude. Tags use the final correlation and the ±0.7 strong/weak thresholds;
filename values are rounded to three decimals. There is no exact-zero target.

The renderer retains 2.44 × 2.44 inches at 300 dpi (732 × 732 pixels), blue
markers of area 5 points², bottom/left axes, and no tick marks, title, labels, or
legend. The notebook heading mentions other chart types, but its code generates
only scatterplots. All supplied images have `ink_density=unknown`; the current
code does not implement the high/low ink-density factor anticipated by the RSA
notebook. No such factor or human complexity score is invented here.

## Simulation and viewing

```sh
python tribe/run.py --dataset ashcraft --device cpu
# Or, when sufficient GPU memory is available:
python tribe/run.py --dataset ashcraft --device cuda
./start-visualizer.sh --port 8001
```

Open [the direction × outlier comparison](http://127.0.0.1:8001/visualizer/brain.html?dataset=ashcraft&x=correlation_direction&y=has_outlier).
The home page also links it in a new tab. Correlation class groups the absolute
published correlation: **Low** for |r| < 0.3, **Mid** for 0.3 ≤ |r| < 0.7,
and **High** for |r| ≥ 0.7. Correlation direction remains a separate X/Y variable.
The original signed category tags remain in the source filenames and reproduction
metadata. Existing saved predictions use the updated groups without inference.
Click a cell or axis tick to see the corresponding original images.
Choose mean prediction, difference from the whole study mean, or another
cortical view using the existing controls. The page includes links to the CSVs.

The adapter lives in `tribe/studies/ashcraft.py`. Selection, input copies, clips,
predictions, receipts, events, and runtime metadata use the existing pipeline
under `tribe/outputs/ashcraft/`. `dataset.py --dataset ashcraft --predict` and
`study.py --dataset ashcraft --predict` use that same runner. Reruns resume only
validated predictions. `--prepare-only` creates clips without inference.

The established protocol is preserved: independent 3-second silent static
videos at 16 fps; full image fitted into the central 256 × 256 area of a white
292 × 292 canvas; H.264 all-intra QP 10, YUV420, BT.709 limited range; FP32
pretrained `facebook/tribev2`; all three 1-Hz predictions saved as 3 × 20,484
arrays. The viewer averages t=0 with equal image weight. Contrasts subtract the
72-image study mean, rather than an equal-cell mean. Left/right fsaverage5
vertices and units are unchanged.

This run predicts responses to the images. It omits MATLAB movement, task
prompts, attention, responses, intertrial/block fixation, scanner triggers, and
continuous block context. P/Q/M/O are documented experimental conditions and
are **not** separate model predictions from identical pixels. The result tests
model sensitivity to these controlled visual stimuli; the repository contains
no measured participant fMRI with which to validate the thesis's semantic/task
contrast. Model outputs are not measured fMRI, percent BOLD, or complexity.

## Verification

```sh
python -m unittest discover -s SemanticsOfDataViz_HAshcraftGTUgThesis -p 'test_*.py' -v
python -m unittest discover -s tribe -p 'test_*.py'
NODE_PATH="$HOME/browser-tool/node_modules" node visualizer/test_studies.cjs http://127.0.0.1:8001/

# Optional numerical export and full source/video/aggregation audit:
python tribe/study.py --dataset ashcraft
python tribe/verify_dataset.py tribe/outputs/ashcraft
```

Tests compare all 72 generated coordinate arrays directly against the numerical
part of the actual source notebook, check published metadata and byte-identical
exports, reject corrupted exports, and verify actual joint memberships and
weighted contrasts through the existing aggregation code. Browser regressions
include the Ashcraft selector, resources, examples, maps, counts, and URL state.
