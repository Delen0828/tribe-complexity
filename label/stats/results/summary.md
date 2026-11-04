# MASSVIS perceived-complexity replication

Source: Lin et al. (2025), *What Makes a Visualization Visually Complex?*, Tables 1–2 and Sections 4–5.
[Paper](https://doi.org/10.1145/3706599.3719983) · [Author supplement](https://osf.io/k4uta/)

5,800 images; 29 predictors; 20% held out; split seed 0.

Count predictors receive log1p exactly once. Binary features and the outcome remain on their original scales. Predictions are not clipped. All fitted models use only the training partition.

| Profile | Model | R² | MAE | MSE | RMSE |
|---|---|---:|---:|---:|---:|
| paper | linear | 0.590673 | 10.731143 | 179.699701 | 13.405212 |
| paper | lasso | 0.590315 | 10.737540 | 179.856800 | 13.411070 |
| paper | ridge | 0.590475 | 10.734235 | 179.786611 | 13.408453 |
| paper | rbf_svr | 0.643904 | 9.854940 | 156.330562 | 12.503222 |
| paper | decision_tree | 0.531868 | 11.159474 | 205.516082 | 14.335832 |
| supplement | linear | 0.590673 | 10.731143 | 179.699701 | 13.405212 |
| supplement | lasso | 0.589296 | 10.753831 | 180.304219 | 13.427741 |
| supplement | ridge | 0.590475 | 10.734235 | 179.786611 | 13.408453 |
| supplement | rbf_svr | 0.643904 | 9.854940 | 156.330562 | 12.503222 |
| supplement | decision_tree | 0.525356 | 11.193945 | 208.374513 | 14.435183 |

## Agreement with the publication

- `paper`: 9/15 Table 2 metric comparisons match at printed precision; 17/25 available Table 1 feature importance comparisons match.
- `supplement`: 15/15 Table 2 metric comparisons match at printed precision; 25/25 available Table 1 feature importance comparisons match.

- rbfSVR MAE is 9.855 in Section 5.2 and 9.86 in Table 2; full-precision 9.854939819 rounds directly to 9.85. Similarly, tree MSE 208.374513 rounds directly to 208.37 versus the table's 208.38. Both table entries are consistent with rounding the notebooks' three-decimal values a second time. Comparisons record both direct rounding and rounding through three decimals, accepting either; raw values and differences are always retained.
- Lasso alpha 0.0046 in Table 2 (0.004620054181220275 in a saved notebook cell) does not produce its printed metrics. Running LassoCV(cv=5, random_state=0) gives alpha 0.016882054461360676 and reproduces the metrics and coefficients.
- Decision tree depth=6, split=85, leaf=11 in Table 2 does not reproduce its printed metrics. Notebook fitting cell 11 uses depth=5, split=65, leaf=8, random_state=0 and reproduces the metrics and importance.

The two profiles expose contradictory source settings; neither is tuned against the reported numbers. The supplement profile executes the authors' fitting procedure. Its CV searches use only the training partition.

## Interpretation and outputs

Permutation importance uses neg_mean_squared_error scoring on the **all** partition, 5 permutations per feature, random_state=0. The original uses the full dataset, including training observations; its importance is not an independent held-out estimate. The estimator is not refitted during permutation.

`feature_statistics.csv` (enabled=True) uses none transformation for Pearson correlations and feature-specific Student/Welch two-sided t-tests, without multiplicity correction. The default settings follow feature_analysis.Rmd. T statistics and their CIs use the contrast column: present minus absent for multiplicity (R orders multipanel before single), absent minus present for the other binary factors. The mean difference column always uses present minus absent. Associations do not establish causality.

- `metrics.csv`, `metric_comparison.csv`: full-precision evaluation and comparison with Table 2.
- `feature_importance.csv`, `importance_comparison.csv`: every predictor and Table 1 comparison.
- `permutation_repeats.csv`: individual permutation scores, when enabled.
- `predictions.csv`: observed scores, train/test membership and model predictions for every image.
- `split.csv`: exact partition and order used by train_test_split.
- `manifest.json`: input/code hashes, package versions, warnings, estimator defaults and fitted parameters.
- `config.resolved.json`: effective configuration for this execution.
- `models/`: serialized pipelines that accept untransformed predictor labels.
- `figures/`: prediction, importance and feature-mean plots.
