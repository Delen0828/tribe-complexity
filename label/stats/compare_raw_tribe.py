"""Compare kernel regressors and a cortical Transformer on all native TRIBE values."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
import torch

from compare_representations import GRIDS, SEED, inner_folds, make_splits, paired_mae, record_scores
from complexity_features import WHITE, WORKSPACE, load_prediction, prediction_records, source_rows
from cortical_transformer import (
    BATCH_SIZE, DEPTH, DROPOUT, EPOCHS, HEADS, LATENTS, LEARNING_RATE, WEIGHT_DECAY, WIDTH,
    cortical_patches, fit_transformer,
)
from replicate import HERE, load_config, load_dataset, sha256, write_json

RAW_FEATURES = "tribe_raw"
RAW_SHAPE = (3, 20484)
ALPHAS = (*GRIDS["ridge"]["model__alpha"], 100000, 1000000)
GAMMA_MULTIPLIERS = (0.25, 1.0, 4.0)
RIDGE_SETTINGS = tuple({"alpha": alpha} for alpha in ALPHAS)
SVR_SETTINGS = tuple({"C": cost, "epsilon": epsilon, "gamma_multiplier": factor}
                     for factor in GAMMA_MULTIPLIERS for cost in GRIDS["rbf_svr"]["model__C"]
                     for epsilon in GRIDS["rbf_svr"]["model__epsilon"])
MODEL_SETTINGS = {
    "ridge": RIDGE_SETTINGS,
    "rbf_svr": SVR_SETTINGS,
    "cortical_transformer": tuple({"epochs": epoch} for epoch in EPOCHS),
}


def standardized_data(responses, fit, evaluate, device):
    train = torch.as_tensor(responses[fit], dtype=torch.float64, device=device)
    evaluation = torch.as_tensor(responses[evaluate], dtype=torch.float64, device=device)
    variance, center = torch.var_mean(train, dim=0, correction=0)
    scale = torch.where(variance > 0, variance.sqrt(), 1.0)
    train.sub_(center).div_(scale)
    evaluation.sub_(center).div_(scale)
    return train, evaluation


def gram_matrices(train, evaluation):
    train = train.flatten(1)
    evaluation = evaluation.flatten(1)
    gram = (train @ train.T).cpu().numpy()
    cross = (evaluation @ train.T).cpu().numpy()
    return gram, cross, evaluation.square().sum(dim=1).cpu().numpy()


def ridge_predictions(gram, cross, target, alphas):
    """Solve the same linear Ridge objective in sample space, with an intercept."""
    col_mean = gram.mean(axis=0)
    centered = gram - col_mean[None, :] - col_mean[:, None] + col_mean.mean()
    cross = cross - cross.mean(axis=1, keepdims=True) - col_mean + col_mean.mean()
    eigenvalues, eigenvectors = eigh(centered, driver="evd", overwrite_a=True)
    eigenvalues = np.maximum(eigenvalues, 0)
    y_mean = target.mean()
    projections = eigenvectors.T @ (target - y_mean)
    weights = eigenvectors @ (projections[:, None] / (eigenvalues[:, None] + np.asarray(alphas)))
    return cross @ weights + y_mean


def rbf_predictions(gram, cross, evaluation_norms, target, settings, jobs):
    norms = gram.diagonal()
    train_dist = np.maximum(norms[:, None] + norms[None, :] - 2 * gram, 0)
    evaluation_dist = np.maximum(evaluation_norms[:, None] + norms[None, :] - 2 * cross, 0)
    # With zero-mean standardized columns, mean squared row norm is d * X.var().
    gamma_scale = 1 / norms.mean() if norms.mean() > 0 else 1.0
    predictions = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for multiplier in sorted({settings[i]["gamma_multiplier"] for i in settings}):
            gamma = gamma_scale * multiplier
            train_kernel = np.exp(-gamma * train_dist)
            evaluation_kernel = np.exp(-gamma * evaluation_dist)

            def fit(i):
                spec = settings[i]
                model = SVR(kernel="precomputed", C=spec["C"], epsilon=spec["epsilon"])
                return i, model.fit(train_kernel, target).predict(evaluation_kernel)

            indices = [i for i, spec in settings.items() if spec["gamma_multiplier"] == multiplier]
            predictions.update(executor.map(fit, indices))
    return predictions


def fit_models(responses, ratings, fit, evaluate, selected, patches, seed, device, jobs):
    train, evaluation = standardized_data(responses, fit, evaluate, device)
    gram, cross, norms = gram_matrices(train, evaluation)
    train, evaluation = train.float(), evaluation.float()
    target = ratings[fit]
    with threadpool_limits(limits=jobs):
        alphas = [RIDGE_SETTINGS[i]["alpha"] for i in selected["ridge"]]
        values = ridge_predictions(gram, cross, target, alphas)
    ridge = dict(zip(selected["ridge"], values.T, strict=True))
    svr = rbf_predictions(gram, cross, norms, target,
                          {i: SVR_SETTINGS[i] for i in selected["rbf_svr"]}, jobs)
    epochs = [EPOCHS[i] for i in selected["cortical_transformer"]]
    deep = fit_transformer(train, evaluation, target, epochs, patches, seed)
    return {"ridge": ridge, "rbf_svr": svr,
            "cortical_transformer": {i: deep[EPOCHS[i]] for i in selected["cortical_transformer"]}}


def score_candidates(responses, ratings, train, folds, patches, device, jobs, smoke=False):
    selected = {model: tuple(range(len(specs))) for model, specs in MODEL_SETTINGS.items()}
    if smoke:
        selected = {"ridge": (2, 5), "rbf_svr": (9, 13), "cortical_transformer": (0,)}
    errors = {model: {i: [] for i in settings} for model, settings in selected.items()}
    for fold, (fit, validation) in enumerate(folds, start=1):
        start = time.monotonic()
        preds = fit_models(responses, ratings, train[fit], train[validation], selected, patches,
                           SEED + fold, device, jobs)
        for model, settings in preds.items():
            for i, predicted in settings.items():
                if not np.isfinite(predicted).all():
                    raise ValueError(f"Non-finite cross-validation predictions for {model}")
                errors[model][i].append(np.abs(ratings[train[validation]] - predicted).mean())
        best = {model: round(min(values[-1] for values in settings.values()), 3)
                for model, settings in errors.items()}
        print(f"  Fold {fold}/{len(folds)} best MAE {best} ({time.monotonic() - start:.1f}s)", flush=True)
    return {model: {i: float(np.mean(scores)) for i, scores in settings.items()}
            for model, settings in errors.items()}


def load_reference(workspace, frame, ratings, groups, splits, inputs):
    reference = workspace / "full"
    manifest = json.loads((reference / "manifest.json").read_text())
    if manifest["n_paired"] != len(frame):
        raise ValueError("The reference comparison must cover all labeled images")
    for name, path in inputs.items():
        if sha256(path) != manifest["inputs_sha256"][name]:
            raise ValueError(f"The reference comparison used a different {name}")
    saved_split = pd.read_csv(reference / "split.csv")
    saved_cv = pd.read_csv(reference / "cv_folds.csv")
    for protocol, (train, test) in splits.items():
        for name, indices in (("train", train), ("test", test)):
            part = saved_split[(saved_split.protocol == protocol) & (saved_split.split == name)]
            part = part.sort_values("partition_order")
            np.testing.assert_array_equal(part.image_id, frame.image_id.iloc[indices])
            np.testing.assert_array_equal(part.source_sha256, groups[indices])
        folds = inner_folds(train, groups if protocol == "grouped" else None)
        for i, (_, validation) in enumerate(folds):
            order = saved_cv[(saved_cv.protocol == protocol) & (saved_cv.fold == i)]
            np.testing.assert_array_equal(order.image_id, frame.image_id.iloc[train[validation]])
    predictions = pd.read_csv(reference / "predictions.csv")
    for (protocol, *_), subset in predictions.groupby(["protocol", "features", "regressor", "setting"]):
        test = splits[protocol][1]
        np.testing.assert_array_equal(subset.image_id, frame.image_id.iloc[test])
        np.testing.assert_allclose(subset.observed, ratings[test], rtol=0, atol=1e-9)
    return reference, pd.read_csv(reference / "metrics.csv"), predictions


def load_responses(indices, rows, records):
    responses = np.empty((len(indices), *RAW_SHAPE), dtype=np.float32)
    for j, i in enumerate(indices):
        values, _ = load_prediction(rows[i], records[i].directory, WHITE)
        if values.dtype != np.float32:
            raise ValueError(f"Unexpected cached precision for {i}")
        responses[j] = values
    return responses


def write_report(out, metrics, deltas):
    lines = ["# MASSVIS complexity from full-resolution TRIBE v2", "",
             "Each chart contributes all three time points and 20,484 fsaverage5 vertices. "
             "The classical regressors receive 61,452 columns per image. "
             "The cortical Transformer receives the native 3 × 20,484 tensor and arranges vertices "
             "into spatial neighborhoods with the three time points in separate channels.", "",
             "Ridge solves the linear problem in sample space. RBF-SVR reuses exact pairwise distances "
             "across the regularization grid. The Perceiver-style Transformer learns spatial and "
             "temporal interactions through cross-attention into 16 latents followed by two "
             "self-attention layers.", ""]
    for protocol, label in (("paper", "Published holdout"), ("grouped", "Duplicate-grouped holdout")):
        lines.extend([f"## {label}", "", "| Input | Regressor | MAE | RMSE | R² |",
                      "|---|---|---:|---:|---:|"])
        table = metrics.loc[metrics.protocol.eq(protocol) & metrics.setting.eq("tuned")]
        table = table[table.features.isin(["metadata", "vjepa2", "tribe_parcels", RAW_FEATURES])]
        for row in table.itertuples():
            lines.append(f"| {row.features} | {row.regressor} | {row.mae:.3f} | {row.rmse:.3f} | {row.r2:.4f} |")
        lines.append("")
    lines.extend(["## Paired test errors", "",
                  "Negative ΔMAE favors the model on the left. Intervals resample held-out source hashes.", "",
                  "| Holdout | Left | Right | ΔMAE | 95% interval |",
                  "|---|---|---|---:|---:|"])
    for row in deltas:
        lines.append(f"| {row['protocol']} | {row['left']} | {row['right']} | {row['delta_mae']:.3f} | "
                     f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def compare(workspace, device, jobs, smoke):
    config, base = load_config(HERE / "config.json")
    frame, _, y, inputs = load_dataset(config, base)
    y = y.to_numpy()
    rows = source_rows(frame, workspace)
    records, errors = prediction_records(rows, workspace)
    groups = np.array([rows[i]["sha256"] for i in frame.index])
    splits = make_splits(frame, groups)
    out = workspace / ("raw_smoke" if smoke else "raw_comparison")
    out.mkdir(parents=True, exist_ok=True)
    patches = cortical_patches()
    if smoke:
        indices = np.array([i for i in splits["paper"][0] if i in records][:96])
        responses = load_responses(indices, rows, records)
        folds = inner_folds(indices)[:1]
        scores = score_candidates(responses, y[indices], np.arange(len(indices)), folds,
                                  patches, device, jobs, smoke=True)
        write_json(out / "cv_mae.json", scores)
        return

    if missing := set(frame.index) - records.keys():
        raise ValueError(f"Missing {len(missing)} TRIBE predictions, including {sorted(missing)[:5]}")
    reference, old_metrics, old_predictions = load_reference(workspace, frame, y, groups, splits, inputs)
    reference_features = pd.read_csv(reference / "features.csv").set_index("image_id")
    for i, record in records.items():
        if (record.prediction_hash, record.source_hash) != tuple(
            reference_features.loc[f"{i}.png", ["prediction_sha256", "source_sha256"]]
        ):
            raise ValueError(f"The reference used a different TRIBE artifact for {i}")
    print(f"Loading {len(frame):,} full-resolution TRIBE tensors on {device}", flush=True)
    responses = load_responses(frame.index, rows, records)
    metric_rows, prediction_rows, deltas, cv = [], [], [], []
    comparison_pairs = [((RAW_FEATURES, model), (feature, model))
                        for model in ("ridge", "rbf_svr") for feature in ("tribe_parcels", "vjepa2", "metadata")]
    comparison_pairs += [((RAW_FEATURES, "cortical_transformer"), right)
                         for right in ((RAW_FEATURES, "ridge"), (RAW_FEATURES, "rbf_svr"), ("vjepa2", "ridge"))]
    for protocol, (train, test) in splits.items():
        print(f"{protocol}: choosing settings within {len(train):,} training images", flush=True)
        folds = inner_folds(train, groups if protocol == "grouped" else None)
        scores = score_candidates(responses, y, train, folds, patches, device, jobs)
        best = {model: min(settings, key=settings.get) for model, settings in scores.items()}
        print(f"{protocol}: refitting " + ", ".join(f"{key} {MODEL_SETTINGS[key][i]}" for key, i in best.items()),
              flush=True)
        predictions = fit_models(responses, y, train, test, {key: (i,) for key, i in best.items()},
                                 patches, SEED, device, jobs)
        previous = old_predictions[(old_predictions.protocol == protocol) & old_predictions.setting.eq("tuned")]
        paired = {key: values.predicted.to_numpy() for key, values in
                  previous.groupby(["features", "regressor"])}
        for model, settings in scores.items():
            for i, error in settings.items():
                cv.append({"protocol": protocol, "regressor": model, "params": json.dumps(MODEL_SETTINGS[model][i]),
                           "cv_mae": error})
            index = best[model]
            value = predictions[model][index]
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite held-out predictions for {model}")
            paired[RAW_FEATURES, model] = value
            record_scores(metric_rows, prediction_rows, protocol, RAW_FEATURES, model, "tuned",
                          MODEL_SETTINGS[model][index], settings[index], frame, test, y, value)
        for left, right in comparison_pairs:
            deltas.append({"protocol": protocol, "left": " / ".join(left), "right": " / ".join(right),
                           **paired_mae(y[test], paired[left], paired[right], groups[test])})

    metrics = pd.concat([old_metrics, pd.DataFrame(metric_rows)], ignore_index=True)
    predictions = pd.concat([old_predictions, pd.DataFrame(prediction_rows)], ignore_index=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "paired_mae.csv", index=False)
    pd.DataFrame(cv).to_csv(out / "cv_results.csv", index=False)
    write_report(out, metrics, deltas)
    write_json(out / "manifest.json", {
        "images": len(frame), "raw_shape": responses.shape, "flattened_shape": (len(frame), int(np.prod(RAW_SHAPE))),
        "time_indices": [0, 1, 2], "vertex_order": "fsaverage5 left then right", "dtype": str(responses.dtype),
        "device": str(device), "seed": SEED, "classical_compute_dtype": "float64",
        "models": MODEL_SETTINGS, "patch_count": len(patches[0]), "vertices_per_padded_patch": patches[0].shape[1],
        "transformer": {"width": WIDTH, "latents": LATENTS, "heads": HEADS, "self_attention_layers": DEPTH,
                        "dropout": DROPOUT, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY, "loss": "Smooth L1, beta=0.2 on standardized ratings",
                        "amp_bfloat16": device.type == "cuda" and torch.cuda.is_bf16_supported()},
        "rejected_predictions": errors,
        "reference_sha256": {name: sha256(reference / name) for name in
                            ("manifest.json", "features.csv", "metrics.csv", "predictions.csv", "split.csv", "cv_folds.csv")},
        "code_sha256": {name: sha256(HERE / name)
                        for name in ("compare_raw_tribe.py", "cortical_transformer.py", "compare_representations.py")},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "scikit-learn", "nilearn", "torch")},
    })
    print(f"Results saved in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or run where the GPU is accessible.")
    torch.set_num_threads(args.jobs)
    compare(args.workspace.resolve(), torch.device(device), args.jobs, args.smoke)


if __name__ == "__main__":
    main()
