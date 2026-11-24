"""Fine-tune image models end to end to predict MASSVIS perceived complexity."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

from compare_raw_tribe import load_reference
from compare_representations import SEED, inner_folds, make_splits, record_scores
from compare_vision_models import atomic_json
from complexity_features import TRIBE, WORKSPACE, source_rows
from replicate import HERE, load_config, load_dataset, sha256

SIZE = 224
MODELS = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2, "fc", 2048, 1e-4),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1, "heads", 768, 3e-5),
}
EPOCHS = (2, 4, 8, 12)
BATCH_SIZE = 32
HEAD_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01


class Charts(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as source:
            rgba = source.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        fitted = ImageOps.contain(rgb, (SIZE, SIZE), method=Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (SIZE, SIZE), "white")
        canvas.paste(fitted, ((SIZE - fitted.width) // 2, (SIZE - fitted.height) // 2))
        return torch.from_numpy(np.array(canvas)).permute(2, 0, 1)


def load_images(paths, device):
    loader = DataLoader(Charts(paths), batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                        pin_memory=device.type == "cuda")
    images = torch.empty((len(paths), 3, SIZE, SIZE), device=device, dtype=torch.uint8,
                         memory_format=torch.channels_last)
    offset = 0
    for batch in loader:
        images[offset:offset + len(batch)].copy_(batch, non_blocking=True)
        offset += len(batch)
    if offset != len(paths):
        raise ValueError("Image loader returned an incomplete cohort")
    return images


def checkpoint_info(name):
    _, weights, _, _, backbone_rate = MODELS[name]
    state_dict = weights.get_state_dict(progress=False, check_hash=True)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    processor = weights.transforms()
    info = {"weights": str(weights), "checkpoint_sha256": sha256(checkpoint),
            "backbone_learning_rate": backbone_rate, "rgb_mean": processor.mean, "rgb_std": processor.std}
    return state_dict, info


def build_regressor(name, state_dict):
    build, _, head_name, width, _ = MODELS[name]
    model = build(weights=None)
    model.load_state_dict(state_dict)
    head = nn.Linear(width, 1)
    setattr(model, head_name, head)
    return model, head


def train_model(name, state_dict, info, images, ratings, train, test, epochs, seed):
    torch.manual_seed(seed)
    device = images.device
    model, head = build_regressor(name, state_dict)
    model.to(device=device, memory_format=torch.channels_last)
    head_parameters = set(head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p not in head_parameters],
         "lr": info["backbone_learning_rate"]},
        {"params": list(head.parameters()), "lr": HEAD_LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS[-1])
    train_rating = ratings[train]
    center, scale = train_rating.mean(), max(train_rating.std(), 1e-7)
    target = torch.as_tensor((train_rating - center) / scale, dtype=torch.float32, device=device)
    train = torch.as_tensor(train, device=device)
    test = torch.as_tensor(test, device=device)
    mean = torch.as_tensor(info["rgb_mean"], device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(info["rgb_std"], device=device).view(1, 3, 1, 1)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def predict(indices):
        x = images[indices].to(dtype=torch.float32, memory_format=torch.channels_last)
        return model(x.div_(255).sub_(mean).div_(std)).flatten()

    predictions = {}
    for epoch in range(1, max(epochs) + 1):
        start = time.monotonic()
        model.train()
        for batch_number, batch in enumerate(torch.randperm(len(train), device=device).split(BATCH_SIZE)):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = F.smooth_l1_loss(predict(train[batch]), target[batch], beta=0.2)
            loss.backward()
            if epoch == 1 and batch_number == 0 and any(p.grad is None for p in model.parameters()):
                raise ValueError(f"Disconnected parameters during {name} fine-tuning")
            optimizer.step()
        scheduler.step()
        if epoch in epochs:
            model.eval()
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                values = torch.cat([predict(indices).float() for indices in test.split(BATCH_SIZE)])
            predictions[epoch] = values.cpu().numpy().astype(np.float64) * scale + center
            if not np.isfinite(predictions[epoch]).all():
                raise ValueError(f"Non-finite predictions for {name} at epoch {epoch}")
        print(f"  {name}: epoch {epoch}/{max(epochs)} ({time.monotonic() - start:.1f}s)", flush=True)
    return predictions


def cached_predictions(path, identity, name, state_dict, images, ratings, train, test, epochs, seed):
    identity = {**identity, "seed": seed, "selected_epochs": list(epochs),
                "partition_sha256": hashlib.sha256(np.r_[len(train), train, test].astype("<i8").tobytes()).hexdigest()}
    if path.is_file():
        cache = json.loads(path.read_text())
        if cache["identity"] == identity:
            predictions = {int(key): np.asarray(value) for key, value in cache["predictions"].items()}
            if set(predictions) == set(epochs) and all(value.shape == test.shape and np.isfinite(value).all()
                                                     for value in predictions.values()):
                return predictions
    predictions = train_model(name, state_dict, identity["models"][name], images, ratings, train, test, epochs, seed)
    atomic_json(path, {"identity": identity, "predictions": predictions})
    return predictions


def run(workspace, device, smoke):
    torch.hub.set_dir(str(TRIBE / "cache" / "torch" / "hub"))
    config, base = load_config(HERE / "config.json")
    frame, _, ratings, inputs = load_dataset(config, base)
    ratings = ratings.to_numpy()
    rows = source_rows(frame, workspace)
    hashes = np.array([rows[i]["sha256"] for i in frame.index])
    splits = make_splits(frame, hashes)
    if smoke:
        indices = splits["paper"][0][:96]
    else:
        load_reference(workspace, frame, ratings, hashes, splits, inputs)
        indices = frame.index
    print(f"Preparing {len(indices):,} original images", flush=True)
    images = load_images([rows[i]["input_path"] for i in indices], device)
    if smoke:
        for name in MODELS:
            state_dict, info = checkpoint_info(name)
            value = train_model(name, state_dict, info, images, ratings[indices], np.arange(64),
                                np.arange(64, 96), (1,), SEED)[1]
            print(f"{name}: smoke produced {len(value)} direct predictions", flush=True)
        return

    out = workspace / "chart_models"
    cache = out / "fold_cache"
    cache.mkdir(parents=True, exist_ok=True)
    identity = {
        "python": sys.executable, "conda_env": Path(sys.prefix).name, "device": str(device),
        "stimuli_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
        "inputs_sha256": {key: sha256(path) for key, path in inputs.items()},
        "split_sha256": sha256(workspace / "full" / "split.csv"),
        "cv_folds_sha256": sha256(workspace / "full" / "cv_folds.csv"),
        "code_sha256": sha256(Path(__file__)),
        "image_shape": [len(frame), 3, SIZE, SIZE],
        "resize": "BICUBIC contain; white letterbox; RGB",
        "epochs": list(EPOCHS), "batch_size": BATCH_SIZE,
        "head_learning_rate": HEAD_LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "loss": "Smooth L1, beta=0.2 on standardized training ratings",
        "schedule": "cosine over the maximum epoch budget",
        "training": "all backbone and scalar head parameters updated",
        "amp_bfloat16": device.type == "cuda" and torch.cuda.is_bf16_supported(),
        "versions": {key: importlib.metadata.version(key) for key in ("torch", "torchvision", "pillow")},
        "models": {},
    }
    metric_rows, prediction_rows, candidates = [], [], []
    for name in MODELS:
        state_dict, identity["models"][name] = checkpoint_info(name)
        model_identity = {**identity, "models": {name: identity["models"][name]}}
        for protocol, (train, test) in splits.items():
            folds = inner_folds(train, hashes if protocol == "grouped" else None)
            errors = {epoch: [] for epoch in EPOCHS}
            print(f"{protocol} {name}: five-fold epoch selection", flush=True)
            for j, (fit, val) in enumerate(folds):
                print(f"  fold {j+1}/{len(folds)}", flush=True)
                value = cached_predictions(cache / f"{protocol}-{name}-fold{j}.json", model_identity,
                                           name, state_dict, images, ratings, train[fit], train[val], EPOCHS,
                                           SEED + j + 1)
                for epoch, predicted in value.items():
                    errors[epoch].append(np.abs(predicted - ratings[train[val]]).mean())
            means = {epoch: float(np.mean(scores)) for epoch, scores in errors.items()}
            selected = min(means, key=means.get)
            candidates.extend({"protocol": protocol, "regressor": name, "epochs": epoch, "cv_mae": error}
                              for epoch, error in means.items())
            print(f"{protocol} {name}: refit through epoch {selected}", flush=True)
            value = cached_predictions(cache / f"{protocol}-{name}-test.json", model_identity,
                                       name, state_dict, images, ratings, train, test, (selected,), SEED)
            record_scores(metric_rows, prediction_rows, protocol, "chart_image", name, "tuned",
                          {"epochs": selected}, means[selected], frame, test, ratings, value[selected])
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "cv_results.csv", index=False)
    atomic_json(out / "manifest.json", identity)
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
    run(args.workspace.resolve(), torch.device(args.device), args.smoke)


if __name__ == "__main__":
    main()
