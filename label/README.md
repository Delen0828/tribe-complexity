# Published complexity annotations

## Minimum dependencies

Rebuilding requires Python 3.9+, `source/annotations.csv`, `source/ratings.csv`, `build.py`, and the
images under `../data/`. The script uses only the standard library. It works
without `paper-label/`, notebooks, R, spreadsheets, survey assets, or trained
models. Outputs are reproducible exports; the two source tables are the source of truth.

```sh
python3 label/build.py
```

Run from the repository root (the script also resolves paths correctly from
other working directories). Outputs are overwritten in `label/output/`.

## Provenance and label dependencies

Source: `paper-label/6. model_applications/data/all_labeled_MASSVIS_5800.csv`.
The corresponding CSVs in supplement sections 3, 4, and 5 are byte-identical.
The unchanged source copy is about 987 KB; `output/report.json` records its
SHA-256 checksum. No annotation values are estimated or corrected here.

The supplement's
`3. checking_labels_process/clean_checked_data/clean_checked_labels.Rmd`
documents this dependency chain:

1. Checked crowdsourced labels supply text types, color attributes, variable
   counts, chart counts, and chart types.
2. Text-type and chart-type selections are encoded into binary columns.
3. Original MASSVIS metadata supplies source category and panel multiplicity.
4. Cleaned slider responses are grouped by filename and image ID and averaged
   to produce `mean_human_complexity_rating`.
5. `chart_types_count` is the sum of the 12 binary chart-type indicators.
6. These inputs are joined into the published CSV, which already includes all
   inputs needed by this project's labeling/export pipeline.

The same cleaning script explicitly strips ` (1)` from six Economist filenames
(100, 102, 103, 108, 110, 113). `build.py` reverses only those documented aliases
when an exact local match is unavailable. It does not use fuzzy matching.

## Export schema

Each JSON record contains:

| Field | Meaning |
| --- | --- |
| `index` | Stable integer study index, 0–5799 |
| `image_id` | Published study ID, such as `0.png` |
| `filename` | Published original filename |
| `image_path` | Repository-relative local path, or null if unavailable |
| `image_match` | `exact`, `documented_alias`, or `missing` |
| `category` | S = science, N = news, G = government, I = infographic |
| `mean_human_complexity_rating` | Overall mean human rating, 0–100 |
| `text`, `color`, `data`, `design` | Feature objects described below |

Binary values are integers: 1 means present/yes and 0 means absent/no.
Counts are nonnegative integers. Zero is a source value, not missing data.
Malformed or missing feature values stop the build rather than becoming zero.
The CSV flattens feature objects with prefixes, such as `text.axes_labels`.

| Dimension | Fields | Interpretation |
| --- | --- | --- |
| Text | `no_text`, `axes_labels`, `axes_text`, `titles`, `annotations`, `captions`, `legend_text`, `legend_title`, `text_only` | Published text-type indicators; selections need not be mutually exclusive |
| Color | `black_and_white`, `background_color` | Binary indicators; background color means a non-white background |
| Color | `color_count` | Approximate distinct color count |
| Data | `quantitative`, `categorical` | Approximate numbers of quantitative and categorical variables |
| Design | `chart_count` | Number of charts |
| Design | `chart_types_count` | Number of distinct chart types, derived from indicators |
| Design | `multi_panel` | MASSVIS panel multiplicity; 0 = single, 1 = multiple |
| Design | `area`, `bar`, `circle`, `diagram`, `distribution`, `grid_or_matrix`, `line`, `map`, `point`, `table`, `text`, `trees_or_networks` | Presence of each of the 12 chart types |

The exported features comprise the model's 29 predictors. Together with the
overall rating, these form the paper's 30-dimensional representation.
Multiplicity is grouped under Design for display, while the paper discusses
it as an additional MASSVIS feature. Descriptive multi-select strings and
redundant one-hot source categories remain available in the unchanged source
CSV but are omitted from the normalized export.

Feature presence is not a per-image explanation of causality. The study's
feature importance values and correlations describe aggregate relationships,
not separate dimension scores. No low/medium/high thresholds are invented.

## Optional predictive-model dependencies

The supplement's `6. model_applications/SVR.ipynb` predicts overall complexity
from **already supplied feature labels**, not from images. Its essential
dependencies are NumPy, pandas, and scikit-learn. Plotting, statistical-analysis,
and notebook dependencies are unnecessary for annotation lookup/export.

The main fitting cells apply `log1p` once to `color_count`, `chart_count`,
`chart_types_count`, `quantitative`, and `categorical`; keep binary predictors
unchanged; split 80/20 with `random_state=0`; and fit
`SVR(kernel='rbf', C=10, gamma='scale', epsilon=1)`. Preserve the source feature
column order if reproducing it. The notebook also contains exploratory cells
that can transform data again and text-input examples with inconsistent label
spellings, so executing the whole notebook sequentially is not this pipeline.
There is no need to retrain or distribute a model to display observed ratings.

## Citation

Kylie Lin, Sean Sheng-Tse Ru, David N. Rapp, Hui Guan, and Cindy Xiong Bearfield.
2025. *What Makes a Visualization Visually Complex?* CHI EA '25.
[Paper](https://kylierlin.github.io/src/files/lin-chi-lbw-2025.pdf) ·
[DOI](https://doi.org/10.1145/3706599.3719983) ·
[Supplement](https://osf.io/k4uta/).

Source annotations and original images retain their authors' rights and terms;
this extraction does not grant a new license to the dataset.

## Human rating spread and histogram

`source/ratings.csv` contains only `image_id` and `response`, extracted from
`paper-label/3. checking_labels_process/clean_checked_data/data/slider_study_cleaned_full_5800.csv`.
Participant identifiers and experimental-condition columns are excluded.
All 5,800 images have 10 responses, whose means match the published ratings.
The report also records a checksum for this reduced ratings table.

The exports include `human_rating_count`, `human_rating_variance` (sample
variance, denominator n−1, in squared score points), and `human_rating_sd`
(the square root of that variance). The viewer histogram counts images by
mean score in 20 five-point bins; the last bin includes 100. The current
mean is overlaid with a shaded mean ±1 SD band, clipped to the 0–100 axis.
This band describes variability among human ratings, not a confidence interval
for the mean or the spread of dataset-wide image means.
