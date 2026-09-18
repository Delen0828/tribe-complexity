# MASSVIS complexity explorer

Explore 5,800 MASSVIS visualizations with the annotations from Lin et al.,
[*What Makes a Visualization Visually Complex?* (CHI EA 2025)](https://kylierlin.github.io/src/files/lin-chi-lbw-2025.pdf).

The dataset contains one overall **mean human complexity rating (0–100)** and
feature labels in **Text, Color, Data, and Design**. These feature groups are
not independently rated complexity scores. This project joins the published
annotations to your local images; it does not infer labels from pixels.

## Run locally

From the repository root, using Python 3.9 or later:

```sh
python3 label/build.py
python3 -m http.server 8000 --bind 127.0.0.1
```

Open **http://localhost:8000/visualizer/**. Enter an index from **0 to 5799**,
or use Previous / Next. A specific visualization can be linked with
`http://localhost:8000/visualizer/?index=42`.

Keep the server rooted at this repository so the website can access both
`label/output/` and `data/`. Opening `index.html` directly with `file://` does
not support fetching the annotations. No Node installation, third-party
Python packages, API keys, backend application, or build framework is needed.

## Files

| Path | Purpose |
| --- | --- |
| `data/` | Original MASSVIS images, referenced in place |
| `paper-label/` | Original supplementary materials; not needed to rebuild after extraction |
| `label/source/annotations.csv` | One unchanged copy of the published 5,800-row annotation table |
| `label/source/ratings.csv` | Individual scores without participant identifiers; used for rating variance |
| `label/build.py` | Validate annotations, resolve image paths, generate exports |
| `label/output/labels.json` | Typed, grouped records used by the website |
| `label/output/labels.csv` | Flat export with dimension-prefixed feature columns |
| `label/output/report.json` | Source checksum, filename aliases, missing and unlabeled images |
| `label/README.md` | Definitions, derivations, provenance, optional model dependencies |
| `visualizer/` | Static HTML, CSS, and JavaScript |

All **5,800 annotations match local images**. Six filename aliases are explicitly
documented in the supplement's own cleaning script. The local image collection
contains **162 additional images without published annotations**; their paths
are listed in the report. They are not assigned invented study IDs or ratings.

The index is the numeric part of the paper's `image_id` (for example, `42.png`),
not an image's position in a directory or the digits in its original filename.

## Validation

```sh
python3 -m unittest discover -s label -p 'test_*.py'
node --check visualizer/app.js  # optional JavaScript syntax check
```

The builder validates the complete ID set, binary and count values, rating
ranges, chart-type counts, panel complements, and unique image assignments.
Exports are deterministic. Missing images are reported and displayed as
unavailable; ambiguous matches stop the build.

For static hosting, serve `visualizer/`, `label/output/labels.json`, and `data/`
with their relative directory structure intact. The visualizer itself has no
runtime service dependency. Original data and archive files remain excluded
from Git by the existing `.gitignore`; obtain the dataset separately when
cloning this repository.

The compact viewer shows attributes as tables. Its histogram displays all 5,800
image means, with the current mean and a shaded ±1 standard deviation band
from the image’s 10 individual human ratings. Sample variance is also displayed.
