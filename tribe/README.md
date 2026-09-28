# TRIBE v2: MASSVIS 751 and 4849

The current experiment follows the stimulus-processing protocol in *Can a Neural
Encoding Model Replicate an fMRI Visualization Study?*, Sections 3.2-3.3. It uses
**3-second silent static videos**, **native resolution with even-dimension cropping**,
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
| 751 | `data/science/v488_n7413_3_f2.png` | 946 x 902 (unchanged) |
| 4849 | `data/government/whoH08_2.png` | 920 x 686 (one right column and bottom row removed) |

- Hold the RGB image constant for 3 seconds, with no audio or separate task text.
- Use 16 fps, standard H.264 High profile, YUV420, fixed QP 10, all-intra frames, explicit BT.709 color
  metadata and limited range. The papers do not specify FPS or codec. Chroma
  subsampling introduces small pixel differences; this is not lossless RGB.
- No scaling or padding before encoding; the pretrained vision processor still
  performs its standard spatial preprocessing.
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
records input/video/PDF hashes, crop boxes, model revisions, and inference settings.
`verify.py` checks every decoded video frame against color-error tolerances for the expected crop,
uses AVFoundation on macOS to independently verify native decoding, and checks
absence of audio, duration, predictions, t=0 selection, contrast arithmetic, and
report contents. Rendered PDF pages are also inspected visually.

## Files

- `processing_comparison.md`: source-referenced methods table and scope limits.
- `inputs/manifest.json`, `inputs/{751,4849}.png`: original source copies/provenance.
- `inputs/paper_3s_t0_bt709_v2/`: current cropped PNGs and regenerated videos.
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
