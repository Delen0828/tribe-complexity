"""Compare standard ResNet-50 and ViT-B/16 on full-resolution simulated cortex."""
import argparse
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_mesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, vit_b_16

from compare_raw_tribe import RAW_FEATURES, RAW_SHAPE, load_responses, standardized_data
from compare_representations import SEED, inner_folds, make_splits, record_scores
from complexity_features import WORKSPACE, prediction_records, source_rows
from cortical_transformer import cortical_patches
from replicate import HERE, load_config, load_dataset, sha256, write_json
from report_model_comparison import save_report

SIZE = 160
BATCH_SIZE = 32
EPOCHS = (4, 8, 16)
LEARNING_RATES = {"resnet50": 0.0005, "vit_b_16": 0.0003}
BUILDERS = {"resnet50": resnet50, "vit_b_16": partial(vit_b_16, image_size=SIZE, dropout=0.1)}
WEIGHT_DECAY = 0.02
REPORT_NOTE = ("ResNet-50 and ViT-B/16 were trained from random initialization on the same TRIBE values. "
               "Their shared 160 × 160 cortical raster places every vertex exactly once per time channel. "
               "Unoccupied cells are set to zero after training-fold standardization.")


def assign_cells(points, height, width):
    y, x = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
    cells = np.column_stack([x.ravel(), y.ravel()])
    index, pixel = linear_sum_assignment(cdist(points, cells, "sqeuclidean"))
    np.testing.assert_array_equal(index, np.arange(len(points)))
    return np.unravel_index(pixel, (height, width))


def cortical_layout():
    """Map all vertices to distinct pixels in a two-hemisphere spatial atlas."""
    # A 17 × 10 grid of 9 × 8 neighborhoods per hemisphere fits in a 160 square.
    tile_rows, tile_cols, tile_height, tile_width = 17, 10, 9, 8
    y_pad = (SIZE - tile_rows * tile_height) // 2
    layout = np.full((SIZE, SIZE), RAW_SHAPE[1], dtype=np.int64)
    groups, positions = cortical_patches()
    sphere = fetch_surf_fsaverage("fsaverage5")
    count = len(groups) // 2
    offset = 0
    for j, hemi in enumerate(("left", "right")):
        vertices = load_surf_mesh(sphere[f"sphere_{hemi}"]).coordinates.astype(np.float64)
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
        centers = positions[j * count:(j + 1) * count, :3].astype(np.float64)
        angles = np.arctan2(centers[:, 1], centers[:, 0]) / np.pi
        tile_y, tile_x = assign_cells(np.column_stack([angles, centers[:, 2]]), tile_rows, tile_cols)
        for i, (center, group) in enumerate(zip(centers, groups[j * count:(j + 1) * count], strict=True)):
            valid = group[group != RAW_SHAPE[1]]
            # Project vertices into their patch center's tangent plane before assigning pixels.
            anchor = np.eye(3)[np.argmin(np.abs(center))]
            across = np.cross(center, anchor)
            across /= np.linalg.norm(across)
            up = np.cross(center, across)
            projected = vertices[valid - offset] @ np.column_stack([across, up])
            scale = np.max(np.abs(projected), axis=0)
            y, x = assign_cells(projected / scale, tile_height, tile_width)
            y = y + y_pad + tile_height * tile_y[i]
            x = x + tile_width * (tile_x[i] + j * tile_cols)
            layout[y, x] = valid
        offset += len(vertices)
    np.testing.assert_array_equal(np.sort(layout[layout != RAW_SHAPE[1]]), np.arange(RAW_SHAPE[1]))
    return layout


def rasterize(responses, layout):
    indices = torch.as_tensor(layout.ravel(), device=responses.device)
    return F.pad(responses.float(), (0, 1)).index_select(-1, indices).reshape(-1, 3, SIZE, SIZE)


def train_model(name, train, test, y, epochs, seed):
    torch.manual_seed(seed)
    device = train.device
    model = BUILDERS[name](weights=None, num_classes=1).to(device)
    if name == "resnet50":
        model = model.to(memory_format=torch.channels_last)
    rate = LEARNING_RATES[name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=WEIGHT_DECAY,
                                 fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS[-1])
    mean, deviation = y.mean(), max(y.std(), 1e-7)
    target = torch.as_tensor((y - mean) / deviation, dtype=torch.float32, device=device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    predictions = {}
    for epoch in range(1, max(epochs) + 1):
        start = time.monotonic()
        model.train()
        for batch in torch.randperm(len(train), device=device).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = F.smooth_l1_loss(model(train[batch]).flatten(), target[batch], beta=0.2)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch in epochs:
            model.eval()
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                values = torch.cat([model(batch).flatten().float() for batch in test.split(BATCH_SIZE)])
            values = values.cpu().numpy().astype(np.float64) * deviation + mean
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite predictions for {name} at epoch {epoch}")
            predictions[epoch] = values
        print(f"  {name}: epoch {epoch}/{max(epochs)} ({time.monotonic() - start:.1f}s)", flush=True)
    return predictions


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    write_json(temporary, data)
    temporary.replace(path)


def cached_predictions(path, identity, name, responses, labels, train, validation, epochs, seed, layout, device):
    if path.is_file():
        cache = json.loads(path.read_text())
        if cache["identity"] == identity:
            value = {int(key): np.asarray(values) for key, values in cache["predictions"].items()}
            if set(value) == set(epochs) and all(v.shape == validation.shape and np.isfinite(v).all()
                                                for v in value.values()):
                return value
    training, evaluation = standardized_data(responses, train, validation, device)
    training = rasterize(training, layout).contiguous(memory_format=torch.channels_last)
    evaluation = rasterize(evaluation, layout).contiguous(memory_format=torch.channels_last)
    value = train_model(name, training, evaluation, labels[train], epochs, seed)
    atomic_json(path, {"identity": identity, "predictions": value})
    return value


def compare(workspace, device, smoke):
    config, base = load_config(HERE / "config.json")
    frame, _, ratings, _ = load_dataset(config, base)
    ratings = ratings.to_numpy()
    rows = source_rows(frame, workspace)
    records, rejected = prediction_records(rows, workspace)
    groups = np.array([rows[i]["sha256"] for i in frame.index])
    splits = make_splits(frame, groups)
    layout = cortical_layout()
    out = workspace / ("vision_smoke" if smoke else "vision_models")
    cache = out / "fold_cache"
    cache.mkdir(parents=True, exist_ok=True)
    if smoke:
        train = np.array([i for i in splits["paper"][0] if i in records][:128])
        values = load_responses(train, rows, records)
        fit, val = standardized_data(values, np.arange(96), np.arange(96, 128), device)
        fit, val = rasterize(fit, layout), rasterize(val, layout)
        for name in BUILDERS:
            prediction = train_model(name, fit, val, ratings[train[:96]], (1,), SEED)[1]
            print(f"{name} smoke MAE: {np.abs(ratings[train[96:]] - prediction).mean():.3f}", flush=True)
        return

    if missing := set(frame.index) - records.keys():
        raise ValueError(f"Missing {len(missing)} raw TRIBE predictions")
    reference = workspace / "raw_comparison"
    raw_manifest = json.loads((reference / "manifest.json").read_text())
    prior_split = workspace / "full" / "split.csv"
    prior_features = workspace / "full" / "features.csv"
    for path in (prior_split, prior_features):
        if raw_manifest["reference_sha256"][path.name] != sha256(path):
            raise ValueError(f"Reference changed: {path}")
    saved = pd.read_csv(prior_features).set_index("image_id")
    for i, record in records.items():
        if (record.prediction_hash, record.source_hash) != tuple(
                saved.loc[f"{i}.png", ["prediction_sha256", "source_sha256"]]):
            raise ValueError(f"Reference used different TRIBE inputs for image {i}")
    existing_metrics = pd.read_csv(reference / "metrics.csv")
    existing_predictions = pd.read_csv(reference / "predictions.csv")
    for (protocol, *_), subset in existing_predictions.groupby(["protocol", "features", "regressor", "setting"]):
        np.testing.assert_array_equal(subset["image_id"], frame.image_id.iloc[splits[protocol][1]])
        np.testing.assert_allclose(subset["observed"], ratings[splits[protocol][1]], atol=1e-9, rtol=0)
    identity = {
        "python": sys.executable, "conda_env": Path(sys.prefix).name, "device": str(device),
        "input_sha256": sha256(prior_features), "split_sha256": sha256(prior_split),
        "reference_manifest_sha256": sha256(reference / "manifest.json"),
        "implementation_sha256": sha256(Path(__file__)),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "scipy")},
        "epochs": EPOCHS, "learning_rates": LEARNING_RATES, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "image_size": SIZE, "seed": SEED,
        "initialization": "torchvision random (weights=None)",
        "amp_bfloat16": device.type == "cuda" and torch.cuda.is_bf16_supported(),
    }
    identity = json.loads(json.dumps(identity))
    responses = load_responses(frame.index, rows, records)
    metric_rows, prediction_rows, candidates = [], [], []
    for protocol, (train, test) in splits.items():
        folds = inner_folds(train, groups if protocol == "grouped" else None)
        for name in BUILDERS:
            print(f"{protocol} {name} training-fold selection", flush=True)
            errors = {epoch: [] for epoch in EPOCHS}
            for j, (fit, val) in enumerate(folds):
                print(f"  fold {j + 1}/{len(folds)}", flush=True)
                value = cached_predictions(cache / f"{protocol}-{name}-fold{j}.json",
                                           identity, name, responses, ratings, train[fit], train[val],
                                           EPOCHS, SEED + j + 1, layout, device)
                for epoch, predictions in value.items():
                    errors[epoch].append(np.abs(predictions - ratings[train[val]]).mean())
            means = {epoch: float(np.mean(values)) for epoch, values in errors.items()}
            selected = min(means, key=means.get)
            candidates.extend({"protocol": protocol, "regressor": name, "epochs": epoch, "cv_mae": value}
                              for epoch, value in means.items())
            print(f"{protocol} {name} refit for {selected} epochs", flush=True)
            value = cached_predictions(cache / f"{protocol}-{name}-test.json",
                                       {**identity, "selected_epochs": selected}, name, responses, ratings,
                                       train, test, (selected,), SEED, layout, device)
            record_scores(metric_rows, prediction_rows, protocol, RAW_FEATURES, name, "tuned",
                          {"epochs": selected}, means[selected], frame, test, ratings, value[selected])

    all_metrics = pd.concat([existing_metrics, pd.DataFrame(metric_rows)], ignore_index=True)
    all_predictions = pd.concat([existing_predictions, pd.DataFrame(prediction_rows)], ignore_index=True)
    all_metrics.to_csv(out / "metrics.csv", index=False)
    all_predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "cv_results.csv", index=False)
    save_report(all_metrics, out, notes=[REPORT_NOTE])
    atomic_json(out / "manifest.json", {**identity, "rejected_predictions": rejected,
                                        "shape": [len(frame), 3, SIZE, SIZE],
                                        "active_pixels_per_channel": RAW_SHAPE[1]})
    print(f"Results saved in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(6)
    compare(args.workspace.resolve(), torch.device(args.device), args.smoke)


if __name__ == "__main__":
    main()
