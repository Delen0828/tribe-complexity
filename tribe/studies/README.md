# Kylie, Hayeong, and Arunkumar comparisons

These studies use the same TRIBE protocol and interactive cortical viewer as MassVis:
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
| Arunkumar experiment 1 | `data/Arunkumar/Stimuli/Expt 1 - 500 stimuli/` | 500, joined to experiment-1 ratings and all taxonomy fields | Recall examples, demographics, result figures |
| Arunkumar experiment 2 | `data/Arunkumar/Stimuli/Expt 2 - 100 stimuli/` | 100, joined by rank and original stimulus ID | Recall examples, demographics, result figures |

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
Perceived complexity is available alongside designed variant and chart type,
grouped into ten-point bins with 100 included in the final bin.

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
Separate binary variables describe each actual stimulus: `has_color`,
`bar_is_hand_drawn`, `font_is_hand_drawn`, `is_comic`, and `bar_is_image`.
Experiment 1's standard versions have grayscale plain bars and plain fonts;
only its embellished cartoon versions have drawn bars/fonts and comic imagery.
Experiment 2's font-only condition has plain bars and a drawn font; its drawn-bar
conditions have both drawn bars and fonts. `is_comic` includes cartoon imagery
in both experiments, and `bar_is_image` identifies the real-object image bars.
The visualizer uses only experiment 2's 24 stimuli and its color and hand-drawn
comparisons. It adds these fields to older selections when reading them, using
the same saved predictions.

## Run and view

Arunkumar's unpacked supplement needs no archive extraction or additional Excel
dependency. Its three authoritative metadata sheets (`Provenance and Purpose`,
`Visual Elements`, and `Data Elements`) supply all 39 taxonomy fields. The
auxiliary `Sheet1` repeats 12 provenance rows and is excluded. Original values
and headers are preserved in `published_features`; trimmed values, named
categories, and No/Yes features are available as X/Y variables. `Skeumorphic`
is exposed as `Skeuomorphic`, retaining the original header in provenance.

Both experiments' published image/information means and standard deviations
are preserved, including experiment 2's group A/B summaries. Scores remain on
the native −10 (image-like) to +10 (information-like) scale. The published
intervals are I1 `[-10,-6]`, I2 `(-6,-3]`, I3 `(-3,3)`, I4 `[3,6)`, and I5
`[6,10]`; spreadsheet membership is checked against each mean. Experiment 2's
`rank-originalID.ext` filenames join its own ratings to the metadata's original
ID. The GIF (stimulus 128) has one frame and follows the static-image protocol.

The combined Arunkumar view contains 600 experiment-specific observations with
equal observation weight, including 100 repeat stimulus IDs. Its separate
experiment views use 500 and 100 observations and their own contrast baselines.
Only 503 distinct source images need inference: 97 experiment-2 files are
byte-identical to experiment 1 and reuse validated predictions. `5-76.jpeg`,
`9-89.png`, and `29-88.png` differ and are inferred from their actual experiment-2
files. Each observation has an independent receipt, thumbnail, and experiment
rating. Workbook and source hashes establish provenance without exporting
participant demographics.

From the repository root in the existing `tribe` environment:

```sh
python tribe/run.py --dataset kylie --device cuda
python tribe/run.py --dataset hayeong --device cuda
python tribe/run.py --dataset arunkumar --device cuda
python visualizer/server.py --port 8000
```

Open **http://localhost:8000/visualizer/**. The home page links each study.
Simulation saves predictions and provenance only. The web backend reads parent
predictions and selects comparisons on demand, without copying or inferring them
again. Comparison choices are available in the Study selector:

| Study selection | Stimuli | Comparison |
| --- | --- | --- |
| Kylie | 24 | Six designs and bar/area chart types |
| Kylie bar | 12 | Six designs, both topics |
| Kylie area | 12 | Six designs, both topics |
| Hayeong | 24 | Experiment 2 and all six conditions |
| Hayeong color | 16 | Black-and-white/color and style × color |
| Hayeong hand drawn | 12 | Plain bars/font, hand-drawn font, hand-drawn bars/font |
| Arunkumar | 600 | Both experiments' own image/information ratings and all 39 taxonomy fields |
| Arunkumar experiment 1 | 500 | Experiment-1 ratings and all taxonomy fields |
| Arunkumar experiment 2 | 100 | Experiment-2 ratings and all taxonomy fields |

Assign variables to X and Y. Click either axis tick to see examples across that
bin; click a cell to see its intersection. Inferior view is the default, with
lateral/medial and all-view alternatives. URL parameters retain the selection.

Cartoon and image styles in experiment 1 use different topics. Its between-style
contrast includes topic differences; standard versus embellished versions are
paired within each topic. Color and hand-drawn comparisons use matching four-topic
sets from experiment 2. These are descriptive model responses.

`dataset.py --dataset NAME --predict` is equivalent to the simulation commands.
Existing worker, batching, precision, background, and custom-output options apply.
Views are chosen only in the web interface. `--sample-only` and `--prepare-only`
remain available. `--report-only` exports optional numeric arrays without plots.

## Verification

```sh
python -m unittest discover -s tribe -p 'test_*.py'
NODE_PATH="$HOME/browser-tool/node_modules" node visualizer/test_studies.cjs http://127.0.0.1:8000/
```

The browser checks require the running visualizer backend and the installed
Playwright Firefox. They cover all study selections, cell/axis membership,
thumbnails, map modes and views, URL restoration, and mobile layout.
