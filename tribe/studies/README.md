# Kylie and Hayeong comparisons

These studies use the same TRIBE protocol and static cortical viewer as MassVis:
three-second silent clips, full-image proportional fitting inside the centered
256 × 256 encoder crop, white padding on a 292 × 292 canvas, FP32 predictions,
and the first response at t=0. All three 1 Hz responses are saved. Means give each
stimulus equal weight; contrasts subtract the mean of the stimuli on that page.
Each page shares one symmetric scale across its groups in each map mode.

## Source organization

`python3 tribe/studies/prepare.py` safely extracts both archives to
`data/{Kylie,Hayeong}/extracted/` and writes deterministic manifests to
`tribe/studies/manifests/`. It preserves meaningful supplement files, skips
macOS/editor artifacts, rejects unsafe paths and changed extracted files, and
does not run any scripts from the archives. Manifests contain relative source
paths, SHA-256 hashes, grouping definitions, aliases, and comparison membership.
Source images, videos, and prediction/report artifacts stay outside Git, following
the existing MassVis practice. Only the interpreted manifests and code are tracked.

| Study | Authoritative stimuli | Included | Excluded |
| --- | --- | --- | --- |
| Kylie | `supp-materials-2026/01. main-study/01. stimuli/{covid,penguins}/{bar,area}/` | 24: two topics × two chart types × six designs | Pilot/test images, duplicate feature-extraction copies, plots |
| Hayeong experiment 1 | `tvcg-mm-3634785/visulization stimuli/experiment_1/{cartoon-style,image-style}/` | 16: eight topics × standard/embellished | Eight `response_summary/` plots |
| Hayeong experiment 2 | `tvcg-mm-3634785/visulization stimuli/experimet_2/{emission,ice,medal,trump}/` | 24: four topics × six conditions | Survey/report figures |

The original misspellings `visulization`, `experimet_2`, and `handrawn` are
preserved in source paths. Kylie's metadata uses `penguins_area_*` while its main
area folder uses `penguin_area_*`; six explicit aliases resolve that discrepancy.

Kylie's `stimuli-labels.csv` has 72 rows: each of the 24 images appears under
three hypothetical source-category settings. Those rows do not represent 72
different stimuli. The manifest preserves common feature labels once per image
and checks that repeated annotations agree. It does not assign any hypothetical
source category to the stimuli. Its complexity scores are means of the 759
observations in `cleaned_data/data_compratings.csv`, joined using topic,
counterbalance, and design ID; no participant IDs are exported. The procedure
and cleaning script establish group 1 as COVID bar/penguins area and group 2 as
COVID area/penguins bar. Design IDs 00–05 identify manipulations, rather than a
monotonic ranking by rated or predicted complexity. `predicted-complexity.csv`
also contains three source-category scenarios and is not used as a human rating.

Hayeong's experiment 2 survey names color, fonts, hand-drawn bars, real-object
bars, and standard bars. Its irregular medal filenames and `ice_standard.png`
are mapped explicitly after inspecting their original PNGs:

| Filename | Condition |
| --- | --- |
| `ice_standard.png` | Standard, black and white |
| `handrawn_medal.png` | Hand-drawn, color |
| `medal_blackwhite.png` | Hand-drawn, black and white |
| `medal_color_font.png` | Standard bars, hand-drawn font, color |
| `medal_og_final.png` | Image bars, color |

No human complexity score is assigned to Hayeong. Its supplied survey measures
credibility and related responses, which are not treated as complexity ratings.

## Run and view

Activate the existing `tribe` Conda environment and run from the repository root:

```sh
python tribe/run.py --dataset kylie --device cuda
python tribe/run.py --dataset hayeong --device cuda
```

The runner prepares sources, resumes only validated predictions, runs inference,
and generates all comparison pages. Equivalent dataset commands are:

```sh
python tribe/dataset.py --dataset kylie --predict --device cuda
python tribe/dataset.py --dataset hayeong --predict --device cuda
```

`--dataset` defaults to `massvis` in `dataset.py`; existing MassVis commands retain
their behavior. Study runs include every authoritative stimulus, without bin
sampling. Use `--device cpu` when CUDA is unavailable. The existing `--thread`,
`--batch-size`, `--cpu-threads`, `--precision`, `--background`, and `--views`
options apply; changing prediction identity requires a separate output directory.
`run.py --outputs-dir PATH` or `dataset.py --output-dir PATH` puts the parent study
there and comparison folders beside it. Each comparison imports validated parent
predictions and prepared videos, so no stimulus is inferred again for a page.

To inspect the selection, prepare videos without inference, or rebuild reports:

```sh
python tribe/dataset.py --dataset kylie --sample-only
python tribe/run.py --dataset hayeong --prepare-only
python tribe/run.py --dataset kylie --report-only
python tribe/run.py --dataset hayeong --report-only
```

`python tribe/study.py --dataset NAME` is the report-only entry point, with
`--predict`, `--sample-only`, and `--prepare-only` actions. Incomplete inference
fails by default; `dataset.py` and `study.py` accept explicit `--allow-partial`.

Serve the repository root with `python3 -m http.server 8765 --bind 127.0.0.1`, then
open [the report hub](../outputs/web/index.html) or the
[shared visualizer](../../visualizer/brain.html). Every comparison has its own
static page:

| Page | Stimuli | Comparison |
| --- | --- | --- |
| [Kylie overview](../outputs/web/study-kylie.html) | 24 | Six designs and bar/area chart types |
| [Kylie bar variants](../outputs/web/study-kylie_bar.html) | 12 | Six variants, each containing both topics |
| [Kylie area variants](../outputs/web/study-kylie_area.html) | 12 | Six variants, each containing both topics |
| [Hayeong overview](../outputs/web/study-hayeong.html) | 40 | All study conditions and both experiments |
| [Hayeong color](../outputs/web/study-hayeong_color.html) | 16 | Balanced black-and-white/color comparison, plus four style × color cells |
| [Hayeong hand-drawn](../outputs/web/study-hayeong_hand_drawn.html) | 12 | Plain bars/font, plain bars/hand-drawn font, hand-drawn bars/font; all color |
| [Hayeong cartoon/image](../outputs/web/study-hayeong_cartoon_image.html) | 16 | Cartoon/image topic sets, plus standard/embellished cells |

Cartoon and image styles in experiment 1 use different topics. Its between-style
contrast includes topic differences; standard versus embellished versions are
paired within each topic. The color and hand-drawn pages use matching four-topic
sets in experiment 2. These are descriptive model responses, not measured fMRI,
significance tests, or a scalar TRIBE complexity score.

## Verification

```sh
python -m unittest discover -s tribe -p 'test_*.py'
python tribe/verify_dataset.py tribe/outputs/kylie
python tribe/verify_dataset.py tribe/outputs/hayeong
NODE_PATH="$HOME/browser-tool/node_modules" node visualizer/test_studies.cjs http://127.0.0.1:8765/
```

Run `verify_dataset.py` on each comparison folder as well. It checks source,
prediction, and video hashes; finite 3 × 20,484 arrays and times 0/1/2; source
geometry, video decoding and color fidelity; t=0 selection; exact group
membership, arithmetic, shared limits, and rendered image assets. Browser QA
requires the Playwright Node package, its Firefox browser, and the local HTTP server
(pass its root URL as the script's first argument); it checks all seven pages,
group membership/thumbnails, both map modes, all view layouts, URL restoration,
the root visualizer, and mobile overflow. Generated verification JSON, browser
screenshots, and receipts live with the outputs.

The browser command above uses the existing local Playwright installation.
For a different installation, set `NODE_PATH` to its `node_modules` directory
or install Playwright as a local development dependency and omit `NODE_PATH`.
