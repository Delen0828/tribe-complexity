# Predicting perceived complexity on MASSVIS

This reproduces the five reported predictive models in Lin et al. (2025),
[*What Makes a Visualization Visually Complex?*](https://doi.org/10.1145/3706599.3719983),
Tables 1–2 and Sections 4–5, using the preprocessed labels in `label/`.
All scripts, configuration, reference extracts, models, plots and results live here.
The primary results and any discrepancies are in [results/summary.md](results/summary.md).

## Run

Use Python 3.11+ and install the pinned analysis dependencies in an environment:

```sh
python3 -m venv /tmp/massvis-stats-env
/tmp/massvis-stats-env/bin/python -m pip install -r label/stats/requirements.txt
/tmp/massvis-stats-env/bin/python label/stats/replicate.py
/tmp/massvis-stats-env/bin/python -m unittest discover -s label/stats -p 'test_*.py' -v
```

On this workspace the existing environment also has the required versions:

```sh
/home/delen/miniforge3/envs/tribe/bin/python label/stats/replicate.py
/home/delen/miniforge3/envs/tribe/bin/python -m unittest discover -s label/stats -p 'test_*.py' -v
```

The default runs **both source configurations**. No download, image processing,
or rebuilding of labels is needed. Full-dataset permutation importance takes
longer than fitting the models; it is calculated once and shared by identical
models across profiles. Reruns overwrite generated files in `results/`.

```sh
python label/stats/replicate.py --profile paper
python label/stats/replicate.py --profile supplement
python label/stats/replicate.py --skip-permutation --no-plots --output label/stats/quick-results
python label/stats/replicate.py --config label/stats/config.json
```

Paths in the config resolve against the config file's directory. CLI `--output`
resolves against the working directory. The default command works from any
working directory when its script is referenced by its absolute path.
The artifact regression tests expect a complete default run with both profiles.

## Why two configurations are necessary

The paper's printed hyperparameters contradict the executable supplementary
notebooks for two models. Applying the printed settings cannot reproduce all
of the printed metrics. This implementation preserves both sources explicitly:

| Model | `paper` profile: Table 2 | `supplement` profile: fitting cells |
|---|---|---|
| Multiple linear | `LinearRegression()` | Same |
| Lasso | `Lasso(alpha=0.0046)` | `LassoCV(cv=5, random_state=0)`, original 100-value alpha grid |
| Ridge | `Ridge(alpha=1)` | `RidgeCV(alphas=[.001,.01,.1,1,10], cv=5)` selects 1 |
| RBF SVR | `SVR(kernel='rbf', C=10, gamma='scale', epsilon=1)` | Same |
| Decision tree | depth 6, minimum split 85, minimum leaf 11 | depth 5, minimum split 65, minimum leaf 8, seed 0 |

The Lasso CV procedure selects **0.016882054461360676** on these training data
and reproduces both its published metrics and coefficients. The saved notebook
output claiming alpha `0.004620054181220275` is inconsistent with those metrics
and with a fresh execution of its fitting cells. No alpha was selected by fitting
against the test scores here. The tree settings in its actual fitting cell
reproduce the reported tree metrics and importances.

RBF SVR reproduces R² **0.6439041917**, MAE **9.8549398191** and MSE
**156.3305622337**. Table 2 prints MAE 9.86, while Section 5.2 prints 9.855:
rounding directly from the full-precision result gives 9.85; rounding through
the three-decimal value gives 9.86. Comparison CSVs expose both rounding paths
and full-precision differences. They accept either rounding path, without a
broad numerical tolerance.
The tree MSE has the same rounding issue: 208.374513 becomes 208.37 directly
or 208.38 through the notebook's 208.375.

## Configuration and fidelity

[config.json](config.json) controls input paths, expected data hashes and counts,
feature names and their order, transformations, split fraction and seed,
estimator classes and parameters, CV settings, permutation scope/repeats/scoring,
statistical tests, optional SVR search and output settings. Profile overrides
replace a model specification in full. Unspecified scikit-learn defaults are
recorded in `manifest.json` alongside fitted alpha, effective gamma, solver
iterations, tree depth, support-vector counts and package versions.

The default protocol is:

- Use 5,800 rows in the published image-ID order and the original **29 predictor
  columns**. The 30th dimension in the paper is the outcome. Filenames, source
  categories, single-panel duplicate, human-rating variance and individual
  responses are excluded from the predictors.
- Check the normalized export against every original annotation value. Verify
  the original CSV's SHA-256, and check the 58,000 individual responses average
  to the supplied outcomes. Fail on missing/nonfinite/malformed features rather
  than dropping rows or inventing zeros.
- Apply `log1p` exactly once to `color_count`, `chart_count`, `chart_types_count`,
  `quantitative`, and `categorical`. Keep binary predictors and the 0–100 outcome
  unchanged. Add no standardization, target transformation, stratification,
  image-derived features, sample weighting or prediction clipping.
- Split with `train_test_split(test_size=.2, random_state=0, shuffle=True)`:
  4,640 training and 1,160 test images. Use this same partition for every model.
  Lasso/Ridge CV uses only the training data, five unshuffled folds, and the
  original default scoring (MSE for LassoCV; R² for RidgeCV with `cv=5`).
- Extract signed linear coefficients, tree impurity importances, and SVR
  permutation importance. The original SVR importance uses **all 5,800 images**,
  `scoring='neg_mean_squared_error'`, `n_repeats=5`, `random_state=0`.
  Thus it includes training observations, as the original does. A test-only
  alternative is configurable and will produce different scores.
- Reproduce Section 4's two-sided independent t-tests, using equal variances
  only for multiplicity, no text, legend title and line; use Welch elsewhere.
  Compute Pearson correlations on the **raw**, unlogged counts. Apply no
  multiple-comparison correction, matching the supplementary R code.
  T-statistic signs follow R's factor order: multipanel minus single for
  multiplicity, and absent minus present for the other binary features.

`requirements.txt` pins the environment used here. The supplement does not
provide its original dependency lockfile, so its historical environment cannot
be claimed to be identical. Numerical results are checked against the paper.
Changing the feature order, parameters, partition or dataset can change them.

## Predict from supplied feature labels

```sh
python label/stats/predict.py
python label/stats/predict.py \
  --model label/stats/results/models/supplement/rbf_svr.joblib \
  --input label/output/labels.csv \
  --output label/stats/results/predicted_complexity.csv
```

Input CSVs can use either the normalized export names (`design.multi_panel`,
`color.color_count`, etc.) or the canonical predictor names in the config.
All configured features must be supplied as raw counts/binary values. Each
serialized model contains its preprocessing pipeline, feature mapping and
training settings, so inference applies the transformation exactly once.
This estimates complexity from feature labels; it does not extract labels from
images. `results/predictions.csv` already includes predictions for every image,
with train/test membership clearly marked. Only its test rows support the
reported generalization metrics.

## Supplementary provenance and optional search

The public [OSF archive](https://osf.io/download/2d8jg/) has SHA-256
`f2c33a07db7f6212fed6608f904174a24f783bf504581b1a549b8e92c17c9da3`.
[reference/provenance.json](reference/provenance.json) records hashes and member
paths. The `*.cells.json` files retain exact notebook cell source and textual
outputs, omitting raster plots. The R analysis and original SVR search CSV are
retained unchanged. Original materials retain their authors' rights and terms.

To regenerate these reference extracts (not needed for running the analysis):

```sh
python label/stats/fetch_reference.py
python label/stats/fetch_reference.py --archive /path/to/'Supplemental Materials 2025.zip'
```

The main run refits the five reported models with the selected parameters.
It does not rerun every exploratory notebook cell. For the original 72-fit
SVR grid over three kernels, six C values, two gamma values and two epsilon
values, use:

```sh
python label/stats/replicate.py --svr-grid-search --output label/stats/search-results
```

That search cell applies an **additional log1p to all columns** after the
initial count transformation, including binary columns, and compares models
using the same test split. The optional implementation preserves those choices
and records them in the config; its results do not replace the five main fits.
This exploratory procedure is not nested CV. The original CSV is available in
`reference/svr_full_gridsearch.csv` for inspection without repeating the search.
Its saved scores are not a numerical benchmark for the main fits: a smoke check
of the first search combination does not reproduce the stored score from a
fresh execution of the published cells, indicating additional notebook-state
or source-data differences. This does not affect the reproduced Tables 1–2.
The later 500-run sample-size experiment and the cognition pilot in Section 7.2
are outside this prediction replication.
