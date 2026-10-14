# TRIBE simulator

The simulator generates prediction arrays and provenance. Cortical plotting and
interactive aggregation run in the [web visualizer](../visualizer/README.md).

## Environment

Use the existing Conda environment:

```sh
conda activate tribe
```

If Conda is not in the shell, run
`source ~/miniforge3/etc/profile.d/conda.sh` first. The environment uses Python
3.11 and `requirements-linux-cuda-lock.txt`, referenced by `environment.yml`.
For an independent Linux setup:

```sh
git clone https://github.com/facebookresearch/tribev2.git tribe/upstream
git -C tribe/upstream checkout af58661791a351a448a489042a28f6c37e1c14b7
conda env create -f environment.yml
```

## Predict

Run from the repository root:

```sh
# MassVis: ten stimuli in each of ten complexity bins
python tribe/dataset.py --scope sample --predict --device cuda

# All 5,800 annotated stimuli; reuse completed sample predictions
python tribe/dataset.py --scope all --predict --device cuda \
  --reuse-from tribe/outputs/massvis_sample

# Controlled studies
python tribe/run.py --dataset kylie --device cuda
python tribe/run.py --dataset hayeong --device cuda
python tribe/run.py --dataset arunkumar --device cuda
```

Rerun these commands to resume validated predictions. `--sample-only` saves the
selection; `run.py --prepare-only` prepares clips. Use `--device cpu` when CUDA
is unavailable. Apple's MPS device is unsupported by the upstream visual extractor.

Outputs include `selection.json`, original stimulus copies, prepared clips,
`prediction_<index>.npz`, atomic validation receipts, and run metadata.
Each array contains three responses with shape `3 × 20,484` and times 0/1/2.
Simulation does not render charts, cortical maps, PDFs, or website pages.
Kylie, Hayeong, and Arunkumar comparisons are selected from parent predictions by the visualizer.
See [study source organization](studies/README.md).

`--thread N` sets CPU preparation workers; `--batch-size N` sets stimuli per GPU
encoder forward; `--cpu-threads N` sets PyTorch CPU threads. Defaults are 8, 1,
and 8. One persistent V-JEPA encoder shares weights across batches. FP32 remains
the default; approximate BF16 is opt-in with `--precision bf16` and a separate
output directory. Prediction identity includes precision, source hash, protocol,
and padding color. `--background '#RRGGBB'` changes padding; changed identities
require separate outputs.

## View

```sh
./start-visualizer.sh
```

The launcher starts the frontend and backend on the same port, checks the
environment, and opens **http://127.0.0.1:8000/visualizer/**. Use `--port 8001`
for another port or `--no-open` on a remote machine. Press Ctrl+C to stop.
See the [visualizer instructions](../visualizer/README.md) for connection checks.
Assign X/Y variables, choose a map
mode and cortical view, then click axis ticks or cells for examples. Partial
runs remain available with completed/selected counts. The backend reads saved
predictions directly, so no report rebuild is needed after inference.

For custom output directories inside the repository, use
`visualizer/brain.html?data=../tribe/outputs/custom/selection.json`.
`python tribe/web.py` refreshes redirects for historical output URLs.

## Protocol

- Three-second silent static clips, 16 fps, H.264 High profile, YUV420, fixed QP
  10, all-intra frames, explicit BT.709 limited-range metadata.
- Full image fitted proportionally within a centered 256 × 256 area on a white
  292 × 292 canvas. The pretrained encoder removes the outer 18-pixel border.
- Pretrained `facebook/tribev2`, video-only independent timelines, native
  average-subject predictions, unchanged compensation/window/sampling settings.
- All three 1 Hz predictions are saved. Web means use only t=0, with equal
  stimulus weight. Contrasts subtract the included-study mean.
- fsaverage5 vertices are ordered left hemisphere, then right hemisphere;
  values are native model units.

[Processing comparison](processing_comparison.md) records protocol decisions.
Historical runs are preserved in `archive/`. Predictions are model outputs,
not measured human fMRI or percent BOLD. The two original MassVis stimuli are
unrelated and do not constitute a matched experimental replication.

## Validation and optional array exports

```sh
python -m unittest discover -s tribe -p 'test_*.py'
NODE_PATH="$HOME/browser-tool/node_modules" node visualizer/test_studies.cjs http://127.0.0.1:8000/
```

Omitting `--predict` in `dataset.py`, or using `run.py --dataset NAME
--report-only`, explicitly exports numerical grouped arrays from existing
predictions. These exports are optional for the web interface and contain no
plots. Use `--allow-partial` with `dataset.py` or `study.py` for incomplete
numerical exports; `verify_dataset.py OUTPUT_DIR` verifies exported arithmetic,
prediction/source hashes, and prepared video fidelity. Legacy pair/spectrum
figures remain an explicit `compare.py` action; `compare.py --mode spectrum
--predict` generates predictions only.
