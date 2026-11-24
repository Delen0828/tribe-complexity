"""Compare metadata, simulated fMRI, and matched V-JEPA on MASSVIS."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from complexity_features import (
    ENGINE, PROTOCOL, REVISIONS, TRIBE, WHITE, WORKSPACE, VisualCache,
    cortical_features, datasets, load_embedding, load_prediction, prediction_records, source_rows,
)
from replicate import ESTIMATORS, HERE, load_config, load_dataset, make_pipeline, metrics, sha256, write_json

SEED = 0
FOLDS = 5
MODELS = ("ridge", "rbf_svr")
MAIN_FEATURES = ("metadata", "tribe_parcels", "vjepa2")
FEATURE_PAIRS = (("tribe_parcels", "vjepa2"), ("tribe_parcels", "metadata"), ("vjepa2", "metadata"))
GRIDS = {
    "ridge": {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000, 10000]},
    "rbf_svr": {"model__C": [1, 10, 100], "model__epsilon": [0.1, 1, 2]},
}
FIXED_METRICS = HERE / "results" / "metrics.csv"
SPLIT = HERE / "results" / "split.csv"


def paper_split(frame):
    saved = pd.read_csv(SPLIT).sort_values("partition_order")
    if len(saved) != len(frame) or set(saved["source_row"]) != set(frame.index):
        raise ValueError("Published split must cover the labeled cohort exactly")
    if not np.array_equal(frame.loc[saved["source_row"], "image_id"], saved["image_id"]):
        raise ValueError("Published split IDs do not match the labels")
    if set(saved["split"]) != {"train", "test"}:
        raise ValueError("Unexpected published split names")
    partitions = tuple(saved.loc[saved["split"].eq(tag), "source_row"].to_numpy()
                       for tag in ("train", "test"))
    for tag, expected in zip(("train", "test"), train_test_split(frame.index.to_numpy(),
                                                                  test_size=.2, random_state=SEED)):
        actual = saved.loc[saved["split"].eq(tag)]
        np.testing.assert_array_equal(actual["partition_order"], np.arange(len(actual)))
        np.testing.assert_array_equal(actual["source_row"], expected)
    return partitions


def collect_features(indices, rows, records, workspace, cache):
    first, average, embeddings, provenance = [], [], [], []
    for i in indices:
        record = records[i]
        embedding, feature_source = load_embedding(record, workspace, cache)
        if embedding is None:
            raise ValueError(f"No matched V-JEPA embedding for {i}")
        values, _ = load_prediction(rows[i], record.directory, WHITE)
        first.append(values[0].copy())
        average.append(values.mean(axis=0))
        embeddings.append(embedding)
        provenance.append({"image_id": f"{i}.png", "source_sha256": record.source_hash,
                           "prediction_dir": str(record.directory), "video_sha256": record.video_hash,
                           "prediction_sha256": record.prediction_hash,
                           "vjepa_source": feature_source, "vjepa_sha256": hashlib.sha256(
                               embedding.tobytes()).hexdigest()})
    return np.asarray(first), np.asarray(average), np.asarray(embeddings), provenance


def make_splits(frame, groups):
    train, test = paper_split(frame)
    grouped = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
                   .split(frame, groups=groups))
    if len(np.intersect1d(groups[grouped[0]], groups[grouped[1]])):
        raise ValueError("Grouped holdout has a duplicate image across partitions")
    return {"paper": (train, test), "grouped": grouped}


def inner_folds(train, groups=None):
    if groups is None:
        return list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(train))
    return list(GroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
                .split(train, groups=groups[train]))


def build_pipeline(config, X, feature, model):
    spec = config["models"][model]
    if feature == "metadata":
        return make_pipeline(spec, list(X.columns), config["preprocessing"])
    steps = [("scale", StandardScaler())]
    if feature == "tribe_vertices_pca":
        steps.append(("pca", PCA(n_components=128, svd_solver="randomized", random_state=SEED)))
    steps.append(("model", ESTIMATORS[spec["class"]](**spec["params"])))
    return Pipeline(steps)


def subset(X, indices):
    return X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]


def fit_grid(config, X, y, train, folds, feature, model, jobs, memory=None):
    pipeline = build_pipeline(config, X, feature, model)
    if memory is not None:
        pipeline.set_params(memory=memory)
    search = GridSearchCV(pipeline, GRIDS[model], cv=folds, scoring="neg_mean_absolute_error",
                          n_jobs=jobs, pre_dispatch=jobs, error_score="raise")
    with parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(limits=1):
        search.fit(subset(X, train), y[train])
    return search


def paired_mae(y, left, right, groups, repeats=2000):
    differences = np.abs(y - left) - np.abs(y - right)
    _, group_ids = np.unique(groups, return_inverse=True)
    sums = np.bincount(group_ids, weights=differences)
    counts = np.bincount(group_ids)
    draws = np.random.default_rng(SEED).integers(len(counts), size=(repeats, len(counts)))
    bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(bootstrap, [.025, .975])
    return {"delta_mae": float(differences.mean()), "ci_low": float(low), "ci_high": float(high)}


def evaluate_full(config, frame, matrices, y, groups, jobs, out):
    metric_rows, prediction_rows, delta_rows, split_rows, fold_rows, tuning_rows = [], [], [], [], [], []
    splits = make_splits(frame, groups)
    for protocol, (train, test) in splits.items():
        folds = inner_folds(train, groups if protocol == "grouped" else None)
        for tag, positions in (("train", train), ("test", test)):
            split_rows.extend({"protocol": protocol, "split": tag, "partition_order": order,
                               "image_id": frame.at[i, "image_id"], "source_sha256": groups[i]}
                              for order, i in enumerate(positions))
        for fold, (_, validation) in enumerate(folds):
            fold_rows.extend({"protocol": protocol, "fold": fold, "image_id": frame.at[i, "image_id"]}
                             for i in train[validation])
        paired = {}
        features = matrices if protocol == "paper" else {key: matrices[key] for key in MAIN_FEATURES}
        for feature, X in features.items():
            with TemporaryDirectory(dir=out) as work:
                memory = work if feature == "tribe_vertices_pca" else None
                for model in MODELS:
                    print(f"{protocol} {feature} {model}", flush=True)
                    search = fit_grid(config, X, y, train, folds, feature, model, jobs, memory)
                    for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
                        tuning_rows.append({"protocol": protocol, "features": feature, "regressor": model,
                                            "params": json.dumps(params), "cv_mae": -score})
                    prediction = search.predict(subset(X, test))
                    paired[feature, model] = prediction
                    record_scores(metric_rows, prediction_rows, protocol, feature, model, "tuned",
                                  search.best_params_, -search.best_score_, frame, test, y, prediction)
        for model in MODELS:
            for left, right in FEATURE_PAIRS:
                delta_rows.append({"protocol": protocol, "regressor": model, "left": left, "right": right,
                                   **paired_mae(y[test], paired[left, model], paired[right, model], groups[test])})
    expected = pd.read_csv(FIXED_METRICS).query("profile == 'paper'").set_index("model")
    train, test = splits["paper"]
    X = matrices["metadata"]
    for model in MODELS:
        fixed = build_pipeline(config, X, "metadata", model)
        fixed.fit(subset(X, train), y[train])
        prediction = fixed.predict(subset(X, test))
        for key, value in metrics(y[test], prediction).items():
            np.testing.assert_allclose(value, expected.at[model, key], atol=1e-8, rtol=1e-8)
        record_scores(metric_rows, prediction_rows, "paper", "metadata", model, "published",
                      config["models"][model]["params"], None, frame, test, y, prediction)

    predictions = pd.DataFrame(prediction_rows)
    for (protocol, *_), scored in predictions.groupby(["protocol", "features", "regressor", "setting"]):
        np.testing.assert_array_equal(scored["image_id"], frame.iloc[splits[protocol][1]]["image_id"])
    for filename, records in (("metrics", metric_rows), ("predictions", predictions),
                              ("paired_mae", delta_rows), ("split", split_rows),
                              ("cv_folds", fold_rows), ("cv_results", tuning_rows)):
        pd.DataFrame(records).to_csv(out / f"{filename}.csv", index=False)
    return metric_rows, delta_rows, split_rows


def record_scores(metric_rows, prediction_rows, protocol, feature, model, setting, params,
                  cv_mae, frame, test, y, prediction):
    keys = {"protocol": protocol, "features": feature, "regressor": model, "setting": setting}
    metric_rows.append({**keys, "params": json.dumps(params), "cv_mae": cv_mae,
                        "n_test": len(test), **metrics(y[test], prediction)})
    prediction_rows.extend({**keys, "image_id": frame.at[i, "image_id"],
                            "observed": y[i], "predicted": value}
                           for i, value in zip(test, prediction, strict=True))


def write_report(metric_rows, deltas, split_rows, groups, out):
    lines = ["# MASSVIS perceived complexity", "",
             f"The outcome is the mean human rating on the 0–100 scale for {len(groups):,} images. "
             "The image pipelines use the same V-JEPA 2 encoder and static stimulus. "
             "TRIBE responses represent a simulated population-level cortical pattern.", ""]
    tables = {
        "Published holdout": [r for r in metric_rows if r["protocol"] == "paper" and r["features"] in MAIN_FEATURES],
        "Duplicate-grouped holdout": [r for r in metric_rows if r["protocol"] == "grouped"],
        "TRIBE feature sensitivity on the published holdout": [r for r in metric_rows
                                                            if r["features"] not in MAIN_FEATURES],
    }
    for title, rows in tables.items():
        lines.extend([f"## {title}", "", "| Features | Regressor | Fit | MAE | RMSE | R² |",
                      "|---|---|---|---:|---:|---:|"])
        for row in rows:
            lines.append(f"| {row['features']} | {row['regressor']} | {row['setting']} | "
                         f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.4f} |")
        lines.append("")
    lines.extend(["", "## TRIBE versus V-JEPA 2", "",
                  "| Holdout | Regressor | ΔMAE (TRIBE − V-JEPA 2) | 95% interval |",
                  "|---|---|---:|---:|"])
    for row in deltas:
        if (row["left"], row["right"]) == ("tribe_parcels", "vjepa2"):
            lines.append(f"| {row['protocol']} | {row['regressor']} | {row['delta_mae']:.3f} | "
                         f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")
    split = pd.DataFrame(split_rows)
    paper = split.query("protocol == 'paper'")
    both = paper.groupby("source_sha256").split.nunique().gt(1).sum()
    grouped = split.query("protocol == 'grouped'").split.value_counts()
    lines.extend(["", f"The published split has {both} byte-identical image groups across the "
                  f"train and test partitions. The grouped split uses {grouped['train']:,} train "
                  f"and {grouped['test']:,} test images with zero hash overlap.",
                  f"The {len(np.unique(groups)):,} distinct source hashes are the resampling units.",
                  "See `metrics.csv` for the full grouped and sensitivity results. "
                  "The intervals describe paired test-set errors for the fitted models."])
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def evaluate_smoke(config, frame, matrices, y, indices, jobs, out):
    train, _ = paper_split(frame)
    positions = {index: i for i, index in enumerate(indices)}
    train = np.array([positions[i] for i in train if i in positions])[:256]
    folds = inner_folds(train)
    rows = []
    for feature, X in matrices.items():
        for model in MODELS:
            search = fit_grid(config, X, y, train, folds, feature, model, jobs)
            rows.append({"features": feature, "regressor": model,
                         "selected_params": json.dumps(search.best_params_), "n_train": len(train),
                         "cv_mae": -search.best_score_})
    pd.DataFrame(rows).to_csv(out / "smoke_cv.csv", index=False)


def run(workspace, cohort, jobs, atlas_dir):
    config, base = load_config(HERE / "config.json")
    frame, X, y, source_paths = load_dataset(config, base)
    rows = source_rows(frame, workspace)
    records, rejected = prediction_records(rows, workspace)
    cache = VisualCache()
    available = np.array([i for i in sorted(records)
                          if load_embedding(records[i], workspace, cache)[0] is not None])
    print(f"Coverage: {len(records)} fMRI, {len(available)} paired, {len(frame)} labeled", flush=True)
    if cohort == "full" and (missing := np.setdiff1d(frame.index, available)).size:
        raise ValueError(f"Evaluation requires both representations for every published image. "
                         f"Missing {len(missing)} pairs (first IDs: {missing[:10].tolist()})")
    out = workspace / cohort
    out.mkdir(parents=True, exist_ok=True)
    # Use only cached training images for smoke fitting. Materialize the full matrices once.
    paper = paper_split(frame)
    indices = available if cohort == "full" else np.intersect1d(available, paper[0])
    first, average, embeddings, provenance = collect_features(indices, rows, records, workspace, cache)
    parcels, temporal, valid_vertices, parcel_names, atlas_hash = cortical_features(first, average, atlas_dir)
    matrices = {"metadata": X.iloc[indices], "tribe_parcels": parcels.astype(np.float64, copy=False),
                "vjepa2": embeddings.astype(np.float64, copy=False),
                "tribe_mean_parcels": temporal.astype(np.float64, copy=False),
                "tribe_vertices_pca": first[:, valid_vertices]}
    pd.DataFrame(provenance).to_csv(out / "features.csv", index=False)
    checkpoint_metadata = {}
    for directory in datasets(workspace):
        path = directory / "run_metadata.json"
        revisions = json.loads(path.read_text()).get("weight_revisions") if path.is_file() else None
        checkpoint_metadata[directory.name] = revisions
    manifest = {"cohort": cohort, "seed": SEED, "folds": FOLDS, "n_labels": len(frame),
                "n_paired": len(available), "n_features": {key: value.shape[1] for key, value in matrices.items()},
                "vertex_pca_components": 128,
                "fMRI_dimensions": {"time_samples": 3, "fsaverage5_vertices": first.shape[1]},
                "parcel_names": parcel_names, "atlas_annotations_sha256": atlas_hash,
                "recovered_sources": [row["recovery"] for row in rows.values() if "recovery" in row],
                "protocol": PROTOCOL, "engine": ENGINE, "checkpoint_revisions": REVISIONS,
                "inference_receipt_counts": dict(Counter(r.directory.name for r in records.values())),
                "run_checkpoint_metadata": checkpoint_metadata, "rejected_predictions": rejected,
                "inputs_sha256": {name: sha256(path) for name, path in
                                  {**source_paths, "published_split": SPLIT, "config": HERE / "config.json",
                                   "massvis_selection": datasets(workspace)[0] / "selection.json"}.items()},
                "implementation_sha256": {path.name: sha256(path) for path in
                                          (Path(__file__), HERE / "complexity_features.py",
                                           TRIBE / "dataset.py", TRIBE / "preprocessing.py",
                                           TRIBE / "gpu_encoder.py", TRIBE / "run.py")},
                "versions": {name: importlib.metadata.version(name) for name in
                             ["numpy", "scikit-learn", "nilearn", "torch", "neuralset", "exca"]}}
    replay = workspace / "replay_verification.json"
    if replay.is_file():
        manifest["replay_verification"] = json.loads(replay.read_text())
    if cohort == "full":
        groups = np.array([rows[i]["sha256"] for i in indices])
        metric_rows, deltas, split_rows = evaluate_full(config, frame, matrices, y.to_numpy(), groups, jobs, out)
        write_report(metric_rows, deltas, split_rows, groups, out)
    else:
        evaluate_smoke(config, frame, matrices, y.to_numpy()[indices], indices, jobs, out)
    write_json(out / "manifest.json", manifest)
    print(f"Results saved in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--cohort", choices=["full", "smoke"], default="full")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--atlas-dir", type=Path, default=TRIBE / "cache" / "nilearn")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run(args.workspace.resolve(), args.cohort, args.jobs, args.atlas_dir.resolve())


if __name__ == "__main__":
    main()
