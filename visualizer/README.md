# TRIBE visualizer

From the repository root:

```sh
./start-visualizer.sh
```

The launcher selects the active Python environment, a local `.venv`, or the
existing `tribe` Conda environment, checks backend/rendering dependencies, builds
missing annotation exports, and opens **http://127.0.0.1:8000/visualizer/**.
Keep the terminal open; **Ctrl+C** stops the server. It can also be called by its
absolute path from another directory.

```sh
./start-visualizer.sh --port 8001           # If port 8000 is already occupied
./start-visualizer.sh --no-open             # Remote/headless machine
./start-visualizer.sh --python /path/to/environment/bin/python
```

The same Python server serves the frontend and `/api/` on the selected port.
There is no separate frontend process or Node build. Open the URL printed by
the launcher: a static server or IDE preview can show the page but cannot answer
analysis requests. On a remote machine, forward the selected port in your IDE
and open that forwarded URL. Missing dependencies are reported before startup;
create the environment with `conda env create -f environment.yml` if needed.

To check communication, open `/api/reports` on the same host and port (for
example, `http://127.0.0.1:8000/api/reports`). It should return a JSON list.
An HTML response or a 404 indicates the wrong server; connection refusal means
the server is stopped or the port is wrong. A JSON `Predictions unavailable`
error from `/api/report` means the backend is reachable but that study has no
usable saved predictions. The launcher uses existing predictions; generate
missing ones using the commands in [tribe/README.md](../tribe/README.md).

For manual startup in an activated environment, use
`python visualizer/server.py --port 8000`.

The home page puts the MassVis and Arunkumar dataset explorers first, followed by
four simulation result links: MassVis, Kylie, Hayeong, and Arunkumar.
The MassVis annotation viewer is at `data.html`; existing `?index=42` bookmarks redirect there.

Arunkumar's dataset explorer is at `arunkumar.html`. It reads the published study
manifest and original images directly, so it also works without predictions.
Choose both experiments (600 observations), experiment 1 (indices 1–500), or
experiment 2 (indices 501–600). It shows each observation's published mean and SD,
all 39 taxonomy fields, and a clickable histogram with one-point bins
`[−10,−9)`, `[−9,−8)`, …, `[9,10]`. Dashed lines mark the separate published
I1–I5 boundaries at −6, −3, 3, and 6; the exact intervals and current group appear
below the chart. The experiment selection and visualization index are saved in
the URL. Histogram bars open the observation closest to their bin midpoint,
with visualization indices breaking ties, as in the MassVis explorer.

## Analysis

Assign published variables to X and Y using the selectors or the variable list.
Y can remain All. Each populated cell shows the equal-stimulus mean of the first
TRIBE prediction (t=0), or its difference from the included-study mean. Empty
intersections show a dash. A shared symmetric scale covers the current grid.

MassVis complexity uses Low, Mid, and High: the bottom, middle, and top thirds of
the selected images ranked by their human complexity scores. Groups have roughly
equal sizes; image IDs break rating ties. Membership uses the whole selection,
so it stays fixed while partial predictions finish. Counts remain exact values
and binary features use No/Yes. Click either axis tick for all examples in that group, or a
cell for its intersection. The example gallery sits above the grid, beside the
variable list, and shows about three thumbnails at a time. Scroll horizontally
to see the rest; more examples load automatically in batches of 24. Axis assignments,
map mode, view, and selected examples are saved in the URL.

Kylie also offers Perceived complexity, using the published human means and
ten-point bins `[0,10)`, …, `[90,100]`. Hayeong uses only experiment 2's 24 stimuli and exposes
`has_color`, `bar_is_hand_drawn`, `font_is_hand_drawn`, `is_comic`, and
`bar_is_image` separately, alongside study condition. These variables also work
with existing saved predictions.

Arunkumar exposes image/information ratings on the published −10 (image-like) to
+10 (information-like) scale, grouped in I1–I5 intervals, plus every one of its
39 stimulus taxonomy fields. The Study selector offers both experiments
together (600 observations) or experiment 1 (500) and experiment 2 (100)
separately. Each uses its own published ratings; separate experiments have
separate contrast baselines. The combined view gives each experiment-specific
observation equal weight. Identical image files reuse predictions, while three
changed experiment-2 images use their actual source files.

Inferior / ventral is the default view. The view selector also offers four
lateral/medial views and all five views. Nilearn renders requested maps in the
web backend, with cached images under `tribe/cache/visualizer/`.

The backend reads `selection.json`, prediction arrays, and validated prediction
receipts directly. Simulation does not build plots or pages. Partial runs show
completed/selected counts. Kylie, Hayeong, and Arunkumar comparisons reuse their parent
study's predictions, without additional inference or duplicate prediction folders.

Custom outputs inside the repository can be opened with
`brain.html?data=../tribe/outputs/custom/selection.json`. Existing `explorer.json`
URLs also work when the folder has its selection and prediction files.
`python tribe/web.py` updates old output-page redirects to this shared interface.
The annotation viewer alone can still be served statically; analysis requires
`visualizer/server.py` and the existing Nilearn/NumPy/Matplotlib dependencies.

## Tests

```sh
python -m unittest discover -s tribe -p 'test_*.py'
python -m unittest discover -s label -p 'test_*.py'
node --check visualizer/brain.js
NODE_PATH="$HOME/browser-tool/node_modules" node visualizer/test_explorers.cjs http://127.0.0.1:8000/
NODE_PATH="$HOME/browser-tool/node_modules" node visualizer/test_studies.cjs http://127.0.0.1:8000/
```

Browser tests use the installed Playwright Firefox and the running backend.
They cover navigation, study comparisons, X/Y aggregation, tick and cell examples,
gallery scrolling, URLs, all cortical layouts, both map modes, and mobile layout.
