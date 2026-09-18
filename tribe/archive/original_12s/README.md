# TRIBE v2 comparison: MASSVIS 751 and 4849

This experiment uses Meta's pretrained `facebook/tribev2` to predict average-subject
cortical fMRI responses to two static MASSVIS images. It does not train a model or
predict a complexity score.

## Results

Open [`index.html`](index.html) or [`outputs/comparison.png`](outputs/comparison.png).
Both runs passed: 12 time points × 20,484 vertices, with finite and distinct outputs.

The temporal-mean maps have spatial Pearson correlation **0.8237**, mean absolute
difference **0.04676**, and RMS difference **0.07339** in model units. Image 751's
whole-cortex mean is **−0.00765**, versus **−0.03916** for 4849 (difference +0.03151).
The difference figure shows more positive predictions for 751 especially over
posterior cortical surfaces; no anatomical ROI significance test was performed.

The RMS of the temporal-mean map is **0.11259** for 751 and **0.11759** for 4849.
Thus, despite the large human complexity difference (92.5 versus 1.9), the simpler
image has slightly larger overall absolute response magnitude in this protocol.
This illustrates why TRIBE response magnitude should not be treated as a direct
complexity score. The maps share a broad spatial pattern but differ regionally.

## Run

The `tribe` conda environment was created with Python 3.11. TRIBE and its plotting
dependencies were installed using pip from the official repository:

```sh
conda create -n tribe python=3.11 pip -y
git clone https://github.com/facebookresearch/tribev2.git tribe/upstream
git -C tribe/upstream checkout af58661791a351a448a489042a28f6c37e1c14b7
conda run -n tribe python -m pip install -e 'tribe/upstream[plotting]'
conda run --no-capture-output -n tribe python tribe/run.py --device cpu
conda run --no-capture-output -n tribe python tribe/compare.py
conda run --no-capture-output -n tribe python tribe/verify.py
```

Run commands from the repository root. Downloaded weights and feature caches live
in `tribe/cache/`. Neuralset also creates its default `~/.cache/neuralset` directory.
`requirements-lock.txt` records installed versions. `pip-check.log` and
`smoke-test.log` record installation checks; `predict.log` records inference.

## Inputs and protocol

IDs follow `label/output/labels.json`, not filename digits or directory order:

| ID | Original image | Published mean human complexity |
| --- | --- | --- |
| 751 | `data/science/v488_n7413_3_f2.png` | 92.5 / 100 |
| 4849 | `data/government/whoH08_2.png` | 1.9 / 100 |

Original image copies, annotations, and SHA-256 checksums are in `inputs/`.
Each image is displayed for 12 seconds in a lossless, silent 16 fps video, preserving
source dimensions and aspect ratio. The pretrained V-JEPA2 processor applies its
standard spatial preprocessing. Separate timelines avoid cross-image context.

We pass Video events directly to `model.predict`, skipping the notebook's
audio extraction/transcription because these images contain no audio. Missing
audio/text modalities use the model's existing missing-modality behavior. Text
visible inside an image remains part of the visual input; no OCR is supplied.
The released checkpoint uses V-JEPA2 vision features (not the inactive DINOv2
extractor present in its configuration).

`run.py` memoizes byte-identical 64-frame visual windows using their SHA-256 hashes.
It keeps mean-token hidden states, matching the official extractor's mean-token
aggregation, so static frames do not require redundant V-JEPA2 passes. This is an
in-process optimization; upstream files and pretrained weights are unchanged.
The official visual extractor does not accept Apple's `mps` device, so this run
uses CPU inference.

The pretrained model retains its 100-second inference window and 2 Hz feature
sampling. Its helper retains the 12 one-second outputs overlapping each stimulus;
context outside the presentation is padded by the native pipeline. Temporal means
include all retained outputs, including onset effects. Predictions already include
the official 5-second hemodynamic compensation; no additional shift is applied.
The plotting comparison uses the full vertex range with the same symmetric color
limits for both images. The difference (751 minus 4849) has its own labeled scale.

## Outputs

- `outputs/prediction_751.npz`, `prediction_4849.npz`: full time × vertex arrays and times.
- `outputs/mean_maps.npz`: temporal mean maps and signed difference.
- `outputs/comparison.png`, `comparison.pdf`: images, four cortical views per map,
  difference maps, temporal response magnitude, and descriptive comparisons.
- `outputs/comparison.json`: numerical metrics.
- `outputs/events_*.csv`, `run_metadata.json`: input protocol and run details.

Vertex order is left hemisphere (10,242), then right hemisphere (10,242), on
fsaverage5. Output units are native model units, not calibrated percent BOLD.
Spatial correlation and differences are descriptive across vertices; they are not
statistical tests or accuracy estimates. Two images and no measured fMRI cannot
establish a relationship between predicted activity and visual complexity. Static
chart presentations also differ from the naturalistic stimuli used for training.

## Sources

- [Official TRIBE v2 demo](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)
- [Official code](https://github.com/facebookresearch/tribev2)
- [Pretrained weights](https://huggingface.co/facebook/tribev2)

Upstream code and weights are subject to their licenses, including TRIBE's
CC BY-NC 4.0 license. Source code commit is pinned above; cached Hugging Face
snapshot directories record weight revisions.
