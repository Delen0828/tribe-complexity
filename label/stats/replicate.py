#!/usr/bin/env python3
"""Replicate Lin et al. (CHI EA 2025) from the preprocessed MASSVIS labels."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
ESTIMATORS = {c.__name__: c for c in
              [LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, SVR, DecisionTreeRegressor]}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=json_default, allow_nan=False) + "\n")


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    dataset = config["dataset"]
    features = dataset["features"]
    if len(features) != len(set(features.values())) or not features:
        raise ValueError("Feature mapping must be nonempty and one-to-one")
    forbidden = {dataset["target"], "image_id", "filename", "index", "category",
                 "human_rating_count", "human_rating_sd", "human_rating_variance"}
    if forbidden.intersection(features) or forbidden.intersection(features.values()):
        raise ValueError("Outcome, identifiers, categories and rating spread cannot be predictors")
    if not set(config["preprocessing"]["count_features"]).issubset(features):
        raise ValueError("Every count feature must be a configured predictor")
    if config["preprocessing"]["transform"] not in {"log1p", "none"}:
        raise ValueError("Supported transforms: log1p, none")
    for profile in config["profiles"].values():
        if not set(profile["model_overrides"]).issubset(config["models"]):
            raise ValueError("Profile overrides must name configured models")
    return config, path.parent


def validate_predictors(frame, count_features):
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictors must be finite; missing values are never imputed")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("Predictors must be integer counts or binary indicators")
    for feature in frame:
        column = frame[feature]
        if feature in count_features:
            if (column < 0).any():
                raise ValueError(f"Negative count in {feature}")
        elif not column.isin([0, 1]).all():
            raise ValueError(f"Non-binary value in {feature}")
    chart_types = ["area", "bar", "circle", "diagram", "distribution", "grid_or_matrix",
                   "line", "map", "point", "table", "text", "trees_or_networks"]
    if set(chart_types + ["chart_types_count"]).issubset(frame.columns):
        if not np.equal(frame[chart_types].sum(axis=1), frame.chart_types_count).all():
            raise ValueError("chart_types_count differs from the sum of chart indicators")


def select_predictors(frame, mapping, count_features):
    names = list(mapping)
    if set(names).issubset(frame.columns):
        predictors = frame[names].astype(float).copy()
    elif set(mapping.values()).issubset(frame.columns):
        predictors = frame[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()}).astype(float)
    else:
        missing = [name for name, source in mapping.items() if name not in frame and source not in frame]
        raise ValueError(f"Missing predictors or mixed schemas: {missing}")
    validate_predictors(predictors, count_features)
    return predictors


def load_dataset(config, base):
    options = config["dataset"]
    paths = {key: (base / options[key]).resolve() for key in
             ["labels_csv", "annotations_csv", "ratings_csv"]}
    frame = pd.read_csv(paths["labels_csv"])
    if len(frame) != options["expected_rows"]:
        raise ValueError(f"Expected {options['expected_rows']} images; got {len(frame)}")
    if frame.image_id.duplicated().any() or frame.filename.duplicated().any():
        raise ValueError("Image IDs and filenames must be unique")
    expected_ids = [f"{index}.png" for index in range(len(frame))]
    if frame.image_id.tolist() != expected_ids:
        raise ValueError("Preserve published image-ID row order for the seeded split")
    X = select_predictors(frame, options["features"], config["preprocessing"]["count_features"])
    y = frame[options["target"]].astype(float)
    low, high = options["rating_range"]
    if not np.isfinite(y).all() or not y.between(low, high).all():
        raise ValueError("Invalid mean complexity rating")
    verified_paths = {"labels_csv": paths["labels_csv"]}
    if options["check_annotation_equivalence"]:
        if options["expected_annotations_sha256"] and sha256(paths["annotations_csv"]) != options["expected_annotations_sha256"]:
            raise ValueError("Published annotation checksum mismatch")
        source = pd.read_csv(paths["annotations_csv"])
        if source.image_id.tolist() != frame.image_id.tolist() or source.filename.tolist() != frame.filename.tolist():
            raise ValueError("Preprocessed labels and published annotations have different rows")
        pd.testing.assert_frame_equal(X, source[list(X)].astype(float), check_exact=True)
        np.testing.assert_allclose(y, source[options["target"]], rtol=0, atol=1e-9)
        verified_paths["annotations_csv"] = paths["annotations_csv"]
    if options["check_individual_ratings"]:
        ratings = pd.read_csv(paths["ratings_csv"])
        if not np.isfinite(ratings.response).all() or not ratings.response.between(low, high).all():
            raise ValueError("Invalid individual rating")
        grouped = ratings.groupby("image_id").response.agg(["size", "mean"])
        if set(grouped.index) != set(frame.image_id):
            raise ValueError("Individual ratings must cover exactly the labeled images")
        grouped = grouped.loc[frame.image_id]
        if not (grouped["size"] == options["expected_ratings_per_image"]).all():
            raise ValueError("Unexpected number of individual ratings")
        np.testing.assert_allclose(y, grouped["mean"], atol=1e-9, rtol=0)
        verified_paths["ratings_csv"] = paths["ratings_csv"]
    return frame, X, y, verified_paths


def make_pipeline(spec, features, preprocessing):
    transforms = []
    for index, feature in enumerate(features):
        transformer = "passthrough"
        if preprocessing["transform"] == "log1p" and feature in preprocessing["count_features"]:
            transformer = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        transforms.append((f"f{index:02}", transformer, [feature]))
    # One transformer per feature preserves the exact published column order.
    prepare = ColumnTransformer(transforms, verbose_feature_names_out=False).set_output(transform="pandas")
    try:
        estimator = ESTIMATORS[spec["class"]](**spec["params"])
    except KeyError as error:
        raise ValueError(f"Unsupported estimator: {spec['class']}") from error
    return Pipeline([("preprocess", prepare), ("model", estimator)])


def metrics(y, predictions):
    return {"r2": r2_score(y, predictions), "mae": mean_absolute_error(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "rmse": float(np.sqrt(mean_squared_error(y, predictions)))}


def fitted_details(pipeline):
    model = pipeline["model"]
    details = {"class": type(model).__name__, "params": model.get_params(deep=False)}
    for attribute in ["alpha_", "_gamma", "n_support_", "n_iter_", "intercept_", "dual_gap_"]:
        if hasattr(model, attribute):
            details[attribute] = getattr(model, attribute)
    if isinstance(model, DecisionTreeRegressor):
        details.update(actual_depth=model.get_depth(), n_leaves=model.get_n_leaves())
    return details


def feature_importances(pipeline, X, y, train, test, config, model_name):
    model = pipeline["model"]
    importance_sd = np.zeros(X.shape[1])
    repeats = None
    if hasattr(model, "coef_"):
        values = model.coef_
        method, scope = "signed_coefficient", "training"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        method, scope = "normalized_impurity_reduction", "training"
    elif config["enabled"] and model_name in config["models"]:
        scope = config["dataset"]
        indices = {"all": np.arange(len(X)), "train": train, "test": test}.get(scope)
        if indices is None:
            raise ValueError("Permutation dataset must be all, train or test")
        print(f"  Permutation importance: {len(indices)} images, {config['n_repeats']} repeats", flush=True)
        parameters = {key: config[key] for key in
                      ["scoring", "n_repeats", "random_state", "n_jobs", "max_samples"]}
        # The original calls permutation_importance on the already transformed X.
        transformed = pipeline["preprocess"].transform(X.iloc[indices])
        result = permutation_importance(model, transformed, y.iloc[indices], **parameters)
        values, importance_sd, repeats = result.importances_mean, result.importances_std, result.importances
        method = "permutation_increase_in_mse" if config["scoring"] == "neg_mean_squared_error" else "permutation_decrease_in_score"
    else:
        return None, None
    denominator = np.max(np.abs(values))
    frame = pd.DataFrame({"feature": X.columns, "importance": values, "importance_sd": importance_sd,
                          "scaled": values / denominator if denominator else values,
                          "method": method, "dataset": scope})
    frame["rank"] = frame.importance.abs().rank(method="first", ascending=False).astype(int)
    return frame, repeats


def feature_statistics(X, y, config, count_features):
    if not config["enabled"]:
        return pd.DataFrame()
    if config["multiple_comparison_correction"] != "none":
        raise ValueError("The original analysis has no multiplicity correction")
    if config["correlation"] != "pearson" or config["correlation_transform"] not in {"none", "log1p"}:
        raise ValueError("Expected Pearson correlations on raw or log1p counts")
    rows = []
    for feature in X:
        column = X[feature]
        row = {"feature": feature, "n": len(y)}
        if feature in count_features:
            values = np.log1p(column) if config["correlation_transform"] == "log1p" else column
            result = stats.pearsonr(values, y, alternative=config["alternative"])
            interval = result.confidence_interval(config["confidence_level"])
            row.update(test="pearson", statistic=result.statistic, p_value=result.pvalue,
                       ci_low=interval.low, ci_high=interval.high,
                       direction="higher" if result.statistic > 0 else "lower")
        else:
            absent, present = y[column == 0], y[column == 1]
            if len(absent) < 2 or len(present) < 2:
                raise ValueError(f"Insufficient observations for t-test: {feature}")
            equal_var = feature in config["equal_variance_features"]
            # R orders panel_s as multipanel/single; binary factors as 0/1.
            present_first = feature in config["present_first_features"]
            first, second = (present, absent) if present_first else (absent, present)
            result = stats.ttest_ind(first, second, equal_var=equal_var, alternative=config["alternative"])
            interval = result.confidence_interval(config["confidence_level"])
            row.update(test="student_t" if equal_var else "welch_t", statistic=result.statistic,
                       contrast="present_minus_absent" if present_first else "absent_minus_present",
                       df=result.df, p_value=result.pvalue, ci_low=interval.low, ci_high=interval.high,
                       n_absent=len(absent), n_present=len(present),
                       mean_absent=absent.mean(), mean_present=present.mean(),
                       sd_absent=absent.std(ddof=1), sd_present=present.std(ddof=1),
                       mean_present_minus_absent=present.mean() - absent.mean(),
                       direction="higher" if present.mean() > absent.mean() else "lower")
        row["significant"] = row["p_value"] < config["alpha"]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_metrics(observed, benchmarks):
    rows = []
    for record in observed.to_dict("records"):
        if record["model"] not in benchmarks["table_2"]:
            continue
        for metric, expected in benchmarks["table_2"][record["model"]].items():
            actual = record[metric]
            decimals = benchmarks["metrics_decimals"][metric]
            direct, via_three = round(actual, decimals), round(round(actual, 3), decimals)
            rows.append({"profile": record["profile"], "model": record["model"], "metric": metric,
                         "observed": actual, "paper": expected, "difference": actual - expected,
                         "rounded_directly": direct, "rounded_via_3dp": via_three,
                         "matches_printed_precision": direct == expected or via_three == expected})
    return pd.DataFrame(rows)


def compare_importances(observed, benchmarks):
    rows = []
    for record in observed.to_dict("records"):
        expected = benchmarks["table_1"].get(record["model"], {}).get(record["feature"])
        if expected is not None:
            rows.append({"profile": record["profile"], "model": record["model"],
                         "feature": record["feature"], "observed": record["importance"],
                         "paper": expected, "difference": record["importance"] - expected,
                         "matches_printed_precision": round(record["importance"], 2) == expected})
    return pd.DataFrame(rows)


def run_grid_search(config, X, y, train, test):
    settings = config["svr_grid_search"]
    if not settings["enabled"]:
        return pd.DataFrame()
    if settings["selection"] != "test_split_as_in_original_notebook":
        raise ValueError("Only the original notebook's exploratory selection protocol is supported")
    prep = make_pipeline(config["models"]["rbf_svr"], list(X), config["preprocessing"])["preprocess"]
    prepared = prep.fit_transform(X)
    if settings["extra_log1p_all_features"]:
        prepared = np.log1p(prepared)
    rows = []
    for C, gamma, epsilon, kernel in itertools.product(settings["C"], settings["gamma"], settings["epsilon"], settings["kernels"]):
        params = deepcopy(config["models"]["rbf_svr"]["params"])
        params.update(C=C, gamma=gamma, epsilon=epsilon, kernel=kernel, degree=settings["degree"])
        model = SVR(**params).fit(prepared.iloc[train], y.iloc[train])
        rows.append({"kernel": kernel, "C": C, "gamma": gamma, "epsilon": epsilon,
                     **metrics(y.iloc[test], model.predict(prepared.iloc[test]))})
        print(f"  SVR grid: {len(rows)} fits completed", flush=True)
    return pd.DataFrame(rows)


def make_plots(out, predictions, importance, statistics, config):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/lin-complexity-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = out / "figures"
    destination.mkdir(exist_ok=True)

    def save(fig, name):
        for extension in config["plot_formats"]:
            if extension not in {"png", "pdf", "svg"}:
                raise ValueError("Plot format must be png, pdf or svg")
            fig.savefig(destination / f"{name}.{extension}", dpi=config["dpi"], bbox_inches="tight")
        plt.close(fig)

    svr_columns = [name for name in predictions if name.endswith(".rbf_svr")]
    if svr_columns:
        column = svr_columns[0]
        heldout = predictions[predictions.split == "test"]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(heldout.observed, heldout[column], alpha=0.3, s=12, linewidths=0)
        ax.plot([0, 100], [0, 100], color="black", linewidth=1)
        score = metrics(heldout.observed, heldout[column])
        ax.set(xlabel="Observed mean complexity", ylabel="Predicted complexity",
               title=f"RBF SVR · held-out images (n={len(heldout)})\nR²={score['r2']:.3f}, MAE={score['mae']:.3f}")
        save(fig, "rbf_svr_predictions")
    if not importance.empty:
        selected = importance[importance.model == "rbf_svr"]
        if not selected.empty:
            selected = selected[selected.profile == selected.profile.iloc[0]].sort_values("importance")
            fig, ax = plt.subplots(figsize=(7, 8))
            ax.barh(selected.feature, selected.scaled, color="#397b97")
            scope = selected.dataset.iloc[0]
            count = len(predictions) if scope == "all" else int((predictions.split == scope).sum())
            ax.set(xlabel="Permutation importance / maximum absolute importance",
                   title=f"RBF SVR feature importance · {scope} images (n={count:,})")
            ax.grid(axis="x", alpha=0.2)
            save(fig, "rbf_svr_importance")
    if not statistics.empty:
        binary = statistics[statistics.test != "pearson"].sort_values("mean_present_minus_absent")
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(binary.mean_absent, binary.feature, label="Absent", color="#999999", s=25)
        ax.scatter(binary.mean_present, binary.feature, label="Present", color="#397b97", s=25)
        ax.set(xlabel="Mean human complexity rating", title="Binary feature comparisons · full dataset")
        ax.legend()
        save(fig, "feature_means")


def write_summary(out, metric_frame, comparisons, importance_comparisons, config, profiles):
    lines = ["# MASSVIS perceived-complexity replication", "",
             "Source: Lin et al. (2025), *What Makes a Visualization Visually Complex?*, Tables 1–2 and Sections 4–5.",
             "[Paper](https://doi.org/10.1145/3706599.3719983) · [Author supplement](https://osf.io/k4uta/)", "",
             f"{config['dataset']['expected_rows']:,} images; {len(config['dataset']['features'])} predictors; "
             f"{config['split']['test_size']:.0%} held out; split seed {config['split']['random_state']}.", "",
             ("Count predictors receive log1p exactly once. " if config["preprocessing"]["transform"] == "log1p" else "Count predictors are untransformed. ") +
             "Binary features and the outcome remain on their original scales. "
             "Predictions are not clipped. All fitted models use only the training partition.", "",
             "| Profile | Model | R² | MAE | MSE | RMSE |", "|---|---|---:|---:|---:|---:|"]
    for row in metric_frame.to_dict("records"):
        lines.append(f"| {row['profile']} | {row['model']} | {row['r2']:.6f} | {row['mae']:.6f} | {row['mse']:.6f} | {row['rmse']:.6f} |")
    lines.extend(["", "## Agreement with the publication", ""])
    for profile in profiles:
        comp = comparisons[comparisons.profile == profile]
        imp = importance_comparisons[importance_comparisons.profile == profile] if not importance_comparisons.empty else pd.DataFrame()
        lines.append(f"- `{profile}`: {int(comp.matches_printed_precision.sum())}/{len(comp)} Table 2 metric comparisons match at printed precision; "
                     f"{int(imp.matches_printed_precision.sum()) if len(imp) else 0}/{len(imp)} available Table 1 feature importance comparisons match.")
    benchmarks = json.loads((out / "benchmarks.json").read_text())
    lines.append("")
    lines.extend(f"- {note}" for note in benchmarks["notes"])
    perm = config["permutation_importance"]
    permutation_note = (f"Permutation importance uses {perm['scoring']} scoring on the **{perm['dataset']}** partition, "
                        f"{perm['n_repeats']} permutations per feature, random_state={perm['random_state']}. "
                        "The original uses the full dataset, including training observations; its importance is not an independent held-out estimate. "
                        "The estimator is not refitted during permutation.") if perm["enabled"] else "Permutation importance was disabled for this execution."
    lines.extend(["", "The two profiles expose contradictory source settings; neither is tuned against the reported numbers. "
                  "The supplement profile executes the authors' fitting procedure. Its CV searches use only the training partition.", "",
                  "## Interpretation and outputs", "",
                  permutation_note, "",
                  f"`feature_statistics.csv` (enabled={config['statistics']['enabled']}) uses "
                  f"{config['statistics']['correlation_transform']} transformation for Pearson correlations and feature-specific Student/Welch "
                  f"{config['statistics']['alternative']} t-tests, without multiplicity correction. The default settings follow feature_analysis.Rmd. T statistics and their CIs "
                  "use the contrast column: present minus absent for multiplicity (R orders multipanel before single), "
                  "absent minus present for the other binary factors. The mean difference column always uses present minus absent. "
                  "Associations do not establish causality.", "",
                  "- `metrics.csv`, `metric_comparison.csv`: full-precision evaluation and comparison with Table 2.",
                  "- `feature_importance.csv`, `importance_comparison.csv`: every predictor and Table 1 comparison.",
                  "- `permutation_repeats.csv`: individual permutation scores, when enabled.",
                  "- `predictions.csv`: observed scores, train/test membership and model predictions for every image.",
                  "- `split.csv`: exact partition and order used by train_test_split.",
                  "- `manifest.json`: input/code hashes, package versions, warnings, estimator defaults and fitted parameters.",
                  "- `config.resolved.json`: effective configuration for this execution.",
                  "- `models/`: serialized pipelines that accept untransformed predictor labels.",
                  "- `figures/`: prediction, importance and feature-mean plots.", ""])
    (out / "summary.md").write_text("\n".join(lines))


def run(config_path, profile="all", skip_permutation=False, no_plots=False, output=None, svr_grid_search=False):
    started = time.monotonic()
    config, base = load_config(config_path)
    if skip_permutation:
        config["permutation_importance"]["enabled"] = False
    if no_plots:
        config["outputs"]["plots"] = False
    if svr_grid_search:
        config["svr_grid_search"]["enabled"] = True
    profiles = list(config["profiles"]) if profile == "all" else [profile]
    if not set(profiles).issubset(config["profiles"]):
        raise ValueError(f"Unknown profile {profile}")
    out = Path(output).resolve() if output else (base / config["outputs"]["directory"]).resolve()
    frame, X, y, paths = load_dataset(config, base)
    train, test = train_test_split(np.arange(len(X)), **config["split"])
    out.mkdir(parents=True, exist_ok=True)
    benchmark_path = (base / config["benchmarks"]).resolve()
    benchmarks = json.loads(benchmark_path.read_text())
    write_json(out / "config.resolved.json", config)
    write_json(out / "benchmarks.json", benchmarks)
    manifest = {"config_source": str(Path(config_path).resolve()), "selected_profiles": profiles,
                "versions": {name: importlib.metadata.version(name) for name in
                             ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]},
                "python": sys.version, "platform": platform.platform(),
                "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
                "code_sha256": {path.name: sha256(path) for path in sorted(HERE.glob("*.py"))},
                "config_sha256": sha256(config_path), "benchmarks_sha256": sha256(benchmark_path),
                "reference_sha256": {path.name: sha256(path) for path in sorted((base / "reference").glob("*")) if path.is_file()},
                "provenance": {name: {"path": str((base / path).resolve()), "sha256": sha256(base / path)}
                               for name, path in config["provenance"].items()},
                "n_images": len(X), "n_features": X.shape[1], "feature_order": list(X),
                "n_train": len(train), "n_test": len(test), "models": {}, "warnings": []}
    predictions = frame[["image_id", "filename", "image_path", "category"]].copy()
    predictions["observed"] = y
    predictions["split"] = "train"
    predictions.loc[test, "split"] = "test"
    split_rows = [{"image_id": frame.image_id.iloc[index], "source_row": index,
                   "split": name, "partition_order": order}
                  for name, indices in [("train", train), ("test", test)] for order, index in enumerate(indices)]
    pd.DataFrame(split_rows).to_csv(out / "split.csv", index=False)
    all_metrics, all_importance, all_repeats, cache = [], [], [], {}
    for profile_name in profiles:
        specifications = {**config["models"], **config["profiles"][profile_name]["model_overrides"]}
        manifest["models"][profile_name] = {}
        for model_name, spec in specifications.items():
            key = model_name + json.dumps(spec, sort_keys=True)
            print(f"{profile_name}/{model_name}: {spec['class']}", flush=True)
            if key not in cache:
                pipeline = make_pipeline(spec, list(X), config["preprocessing"])
                fit_started = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(X.iloc[train], y.iloc[train])
                manifest["warnings"].extend({"profile": profile_name, "model": model_name,
                                             "category": warning.category.__name__, "message": str(warning.message)}
                                            for warning in caught)
                prediction = pipeline.predict(X)
                values = metrics(y.iloc[test], prediction[test])
                importance, repeats = feature_importances(pipeline, X, y, train, test,
                                                          config["permutation_importance"], model_name)
                cache[key] = pipeline, prediction, values, importance, repeats, time.monotonic() - fit_started
            pipeline, prediction, values, importance, repeats, elapsed = cache[key]
            print(f"  R²={values['r2']:.6f}, MAE={values['mae']:.6f}, MSE={values['mse']:.6f}", flush=True)
            all_metrics.append({"profile": profile_name, "model": model_name, **values})
            predictions[f"{profile_name}.{model_name}"] = prediction
            details = fitted_details(pipeline)
            details["fit_and_importance_seconds"] = elapsed
            if config["outputs"]["save_models"]:
                model_path = out / "models" / profile_name / f"{model_name}.joblib"
                model_path.parent.mkdir(exist_ok=True, parents=True)
                joblib.dump({"pipeline": pipeline, "feature_mapping": config["dataset"]["features"],
                             "count_features": config["preprocessing"]["count_features"],
                             "profile": profile_name, "model": model_name,
                             "input_scale": "raw labeled counts and binary indicators", "details": details},
                            model_path, compress=3)
                details["artifact_sha256"] = sha256(model_path)
            manifest["models"][profile_name][model_name] = details
            if importance is not None:
                all_importance.append(importance.assign(profile=profile_name, model=model_name))
            if repeats is not None:
                all_repeats.extend({"profile": profile_name, "model": model_name, "feature": feature,
                                    "repeat": repeat, "importance": repeats[index, repeat]}
                                   for index, feature in enumerate(X) for repeat in range(repeats.shape[1]))
    metric_frame = pd.DataFrame(all_metrics)
    importance_frame = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
    comparisons = compare_metrics(metric_frame, benchmarks)
    importance_comparisons = compare_importances(importance_frame, benchmarks)
    statistics = feature_statistics(X, y, config["statistics"], config["preprocessing"]["count_features"])
    grid = run_grid_search(config, X, y, train, test)
    for name, table in [("metrics", metric_frame), ("metric_comparison", comparisons),
                        ("feature_importance", importance_frame), ("importance_comparison", importance_comparisons),
                        ("feature_statistics", statistics), ("predictions", predictions),
                        ("permutation_repeats", pd.DataFrame(all_repeats)), ("svr_grid_search", grid)]:
        table.to_csv(out / f"{name}.csv", index=False)
    if config["outputs"]["plots"]:
        make_plots(out, predictions, importance_frame, statistics, config["outputs"])
    write_summary(out, metric_frame, comparisons, importance_comparisons, config, profiles)
    manifest["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "manifest.json", manifest)
    print(f"Results written to {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--profile", default="all", help="all, paper, supplement, or another configured profile")
    parser.add_argument("--output", type=Path, help="Output directory; default resolves relative to the config")
    parser.add_argument("--skip-permutation", action="store_true", help="Skip the expensive SVR importance step")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--svr-grid-search", action="store_true", help="Also repeat the exploratory notebook search")
    args = parser.parse_args()
    run(args.config, args.profile, args.skip_permutation, args.no_plots, args.output, args.svr_grid_search)


if __name__ == "__main__":
    main()
