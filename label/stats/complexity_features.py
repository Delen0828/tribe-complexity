"""Resolve individual MASSVIS predictions and extract matched image features."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from replicate import HERE, load_config, load_dataset, sha256, write_json

TRIBE = HERE.parents[1] / "tribe"
OUTPUTS = TRIBE / "outputs"
WORKSPACE = OUTPUTS / "complexity_comparison"
sys.path.insert(0, str(TRIBE))
from dataset import load_prediction
from gpu_encoder import BatchedEncoder, ENGINE, MODEL_NAME, static_window
from preprocessing import PROTOCOL

WHITE = [255, 255, 255]
LAYERS = [0.5, 0.75, 1.0]
CACHE_SHAPE = (20, 1408, 6)
EMBEDDING_SIZE = 2816
REPLAY_IDS = (0, 751, 4258)
HF_CACHE = TRIBE / "cache" / "huggingface" / "hub"
REVISIONS = {
    "facebook/tribev2": "f894e783020944dcd96e5568550afe2aa9743f9f",
    MODEL_NAME: "875c192b7b704b87d1e1d99345769632dd5f739a",
}


def datasets(workspace):
    return (
        OUTPUTS / "massvis_all",
        OUTPUTS / "massvis_sample",
        OUTPUTS / "gpu_smoke",
        workspace / "completed",
        workspace / "recovered_run",
    )


@dataclass(frozen=True)
class Prediction:
    index: int
    source_hash: str
    directory: Path
    video: Path
    video_hash: str
    prediction_hash: str


def source_rows(frame, workspace):
    selection_path = OUTPUTS / "massvis_all" / "selection.json"
    selection = json.loads(selection_path.read_text())
    if (selection["protocol"], selection["background_rgb"]) != (PROTOCOL, WHITE):
        raise ValueError("MASSVIS selection has a different protocol or background")
    rows = {row["index"]: row for row in selection["rows"]}
    if len(rows) != len(selection["rows"]) or set(rows) != set(range(len(frame))):
        raise ValueError("MASSVIS selection must contain each published image once")
    for i, row in rows.items():
        path = TRIBE.parent / frame.at[i, "image_path"]
        if frame.at[i, "image_id"] != f"{i}.png" or frame.at[i, "image_path"] != row["image_path"]:
            raise ValueError(f"Source identity mismatch for {i}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Source hash mismatch for {i}")
        row["input_path"] = path
        recovery = workspace / "recovered" / f"{i}.json"
        if recovery.is_file():
            info = json.loads(recovery.read_text())
            recovered = recovery.with_suffix(".png")
            if (info["original_sha256"] != row["sha256"] or info["recovered_sha256"] != sha256(recovered)
                or not recovered.read_bytes().startswith(path.read_bytes())):
                raise ValueError(f"Recovered source provenance mismatch for {i}")
            row.update(input_path=recovered, sha256=info["recovered_sha256"], recovery=info)
    return rows


def prediction_records(rows, workspace):
    """Find compatible artifacts in priority order, with corrupt artifacts reported."""
    records = {}
    rejected = []
    for directory in datasets(workspace):
        for path in sorted(directory.glob("prediction_*.json")):
            name = path.stem.removeprefix("prediction_")
            if not name.isdecimal() or int(name) not in rows or int(name) in records:
                continue
            i = int(name)
            try:
                _, receipt = load_prediction(rows[i], directory, WHITE)
                prepared = receipt["prepared"]
                video = (TRIBE / prepared["video"]).resolve()
                if prepared["index"] != i or prepared["source_sha256"] != rows[i]["sha256"]:
                    raise ValueError("Prepared stimulus identity differs")
                if sha256(video) != prepared["video_sha256"]:
                    raise ValueError("Prepared video hash differs")
                records[i] = Prediction(i, rows[i]["sha256"], directory, video,
                                        prepared["video_sha256"], receipt["prediction_sha256"])
            except (OSError, ValueError, KeyError, AssertionError) as error:
                rejected.append({"index": i, "path": str(path), "reason": str(error)})
    return records, rejected


def cache_suffix(video):
    # Neuralset uses an absolute path; the repository-relative suffix survives a move.
    path = Path(video).as_posix()
    marker = "/tribe/outputs/"
    return path.split(marker, 1)[1] if marker in path else path


def image_extractor():
    from neuralset.extractors.image import HuggingFaceImage

    return HuggingFaceImage(model_name=MODEL_NAME, device="cpu", layers=LAYERS,
                            layer_aggregation="group_mean", token_aggregation="mean",
                            cache_n_layers=CACHE_SHAPE[0])


class VisualCache:
    def __init__(self):
        from exca.cachedict.dumpcontext import DumpContext

        self.extractor = image_extractor()
        self.entries = {}
        root = TRIBE / "cache" / "features" / PROTOCOL / "ffffff" / "intra_qp10" / ENGINE / "fp32"
        for config in sorted(root.glob("neuralset.extractors.video.HuggingFaceVideo._get_data,*/*/config.yaml")):
            spec = yaml.safe_load(config.read_text())
            image = spec["image"]
            if (image["model_name"], image["cache_n_layers"], image["cache_all_layers"],
                image["token_aggregation"], spec["frequency"], spec["num_frames"],
                spec["clip_duration"], spec["use_audio"], spec["max_imsize"]) != (
                    MODEL_NAME, CACHE_SHAPE[0], False, "mean", 2.0, None, 4.0, False, None):
                continue
            reader = DumpContext(config.parent)
            for index in sorted(config.parent.glob("*-info.jsonl")):
                with index.open() as stream:
                    for line in stream:
                        # Exca blanks removed records. A writer can leave an unfinished last line.
                        if not line.startswith("{") or not line.endswith("\n"):
                            continue
                        info = json.loads(line)
                        key = info.pop("#key", "")
                        if key.endswith(".mp4_0.00_3.00"):
                            suffix = cache_suffix(key.removesuffix("_0.00_3.00"))
                            self.entries.setdefault(suffix, []).append((reader, info, index, key))

    def embedding(self, video):
        for reader, info, index, key in self.entries.get(cache_suffix(video), ()):
            try:
                timed = reader.load(info)
                values = np.asarray(timed.data)
                if (values.shape != CACHE_SHAPE or values.dtype != np.float32
                    or timed.frequency != 2 or timed.duration != 3 or not np.isfinite(values).all()):
                    raise ValueError(f"Invalid V-JEPA cache array for {video}")
                # The verified input window is the same at all six sample times.
                tiled = np.broadcast_to(values[..., :1], values.shape)
                if not np.array_equal(values, tiled):
                    np.testing.assert_allclose(values, tiled, atol=1e-6, rtol=1e-5)
                grouped = self.extractor._aggregate_layers(values)
                return grouped.mean(axis=-1).reshape(EMBEDDING_SIZE), f"{index}:{key}"
            except (OSError, ValueError, KeyError, AssertionError):
                continue
        return None, None

    def encode(self, encoder, videos):
        windows = [static_window(video) for video in videos]
        for pooled in encoder.encode(windows):
            sampled = self.extractor._aggregate_tokens(pooled).numpy()
            yield self.extractor._aggregate_layers(sampled).reshape(EMBEDDING_SIZE)


def embedding_identity(record):
    return {"index": record.index, "source_sha256": record.source_hash,
            "video_sha256": record.video_hash, "protocol": PROTOCOL, "engine": ENGINE,
            "model": MODEL_NAME, "revision": REVISIONS[MODEL_NAME], "precision": "fp32"}


def load_embedding(record, workspace, cache):
    path = workspace / "embeddings" / f"{record.index}.npz"
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as saved:
                value = saved["embedding"]
                identity = json.loads(saved["identity"].item())
                if (identity == embedding_identity(record) and value.shape == (EMBEDDING_SIZE,)
                    and value.dtype == np.float32 and np.isfinite(value).all()):
                    return value, f"{path} (sha256={sha256(path)})"
        except (OSError, ValueError, KeyError):
            pass
    return cache.embedding(record.video)


def write_embedding(record, embedding, workspace):
    output = workspace / "embeddings"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{record.index}.npz"
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, embedding=embedding,
                        identity=json.dumps(embedding_identity(record), sort_keys=True))
    temporary.replace(path)


def prepare(rows, output, indices):
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    manifest = []
    rejected = []
    for i in sorted(indices):
        source = rows[i]["input_path"].resolve()
        try:
            with Image.open(source) as image:
                image.load()
        except (OSError, ValueError) as error:
            rejected.append({"image_id": f"{i}.png", "path": str(source), "reason": str(error)})
            continue
        destination = inputs / f"{i}.png"
        if not destination.exists():
            destination.symlink_to(source)
        elif sha256(destination) != rows[i]["sha256"]:
            original = (TRIBE.parent / rows[i]["image_path"]).resolve()
            if "recovery" not in rows[i] or not destination.is_symlink() or destination.resolve() != original:
                raise ValueError(f"Stale staged input for {i}: {destination}")
            temporary = destination.with_suffix(".tmp")
            temporary.symlink_to(source)
            temporary.replace(destination)
        manifest.append({"index": i, "sha256": rows[i]["sha256"]})
    write_json(output / "manifest.json", manifest)
    write_json(output / "unreadable_sources.json", rejected)
    print(f"Prepared {len(manifest)} entries in {output / 'manifest.json'}", flush=True)
    if rejected:
        print(f"Unable to decode {len(rejected)} source images. See {output / 'unreadable_sources.json'}",
              flush=True)


def register_recovery(index, row, candidate, url, workspace):
    original = TRIBE.parent / row["image_path"]
    original_bytes = original.read_bytes()
    candidate_bytes = candidate.read_bytes()
    if len(candidate_bytes) <= len(original_bytes) or not candidate_bytes.startswith(original_bytes):
        raise ValueError("The complete image must preserve every byte of the local source")
    with Image.open(candidate) as image:
        image.load()
    directory = workspace / "recovered"
    directory.mkdir(parents=True, exist_ok=True)
    # Keep the downloaded original separate from the dataset archive and labels.
    staged = directory / f"{index}.tmp.png"
    shutil.copyfile(candidate, staged)
    staged.replace(directory / f"{index}.png")
    write_json(directory / f"{index}.json", {"image_id": f"{index}.png", "archive_url": url,
                                            "original_sha256": sha256(original),
                                            "recovered_sha256": sha256(candidate)})


def check_replay(rows, records, workspace, cache):
    result = {}
    for i in REPLAY_IDS:
        old = records[i]
        directory = workspace / "replay"
        current, receipt = load_prediction(rows[i], directory, WHITE)
        original, _ = load_prediction(rows[i], old.directory, WHITE)
        np.testing.assert_allclose(current, original, atol=1e-5, rtol=1e-4)
        expected, _ = cache.embedding(old.video)
        current_embedding, _ = cache.embedding((TRIBE / receipt["prepared"]["video"]).resolve())
        if expected is None or current_embedding is None:
            raise ValueError(f"Missing V-JEPA data for replay image {i}")
        np.testing.assert_allclose(current_embedding, expected, atol=1e-5, rtol=1e-4)
        result[i] = {"original_video_sha256": old.video_hash,
                     "replay_video_sha256": receipt["prepared"]["video_sha256"],
                     "max_abs_fmri_difference": float(np.max(np.abs(current - original))),
                     "max_abs_embedding_difference": float(np.max(np.abs(current_embedding - expected)))}
    metadata = json.loads((workspace / "replay" / "run_metadata.json").read_text())
    revisions = {key.replace("/", "--"): revision for key, revision in REVISIONS.items()}
    if any(metadata["weight_revisions"].get(f"models--{key}") != revision
           for key, revision in revisions.items()):
        raise ValueError("Replay model revisions differ from the comparison configuration")
    write_json(workspace / "replay_verification.json", result)
    print(f"Replay verified on {len(result)} images", flush=True)


def fill_embeddings(records, workspace, cache, device, cpu_threads):
    import torch

    torch.set_num_threads(cpu_threads)
    os.environ["HF_HOME"] = str(HF_CACHE.parent)
    # Transformers can already have imported huggingface_hub while reading the cache.
    import huggingface_hub.constants
    huggingface_hub.constants.HF_HUB_CACHE = str(HF_CACHE)
    model_cache = HF_CACHE / f"models--{MODEL_NAME.replace('/', '--')}"
    if (model_cache / "refs" / "main").read_text().strip() != REVISIONS[MODEL_NAME]:
        raise ValueError("V-JEPA checkpoint revision differs from the comparison configuration")
    encoder = BatchedEncoder(device, "fp32")
    missing = [record for record in records.values()
               if load_embedding(record, workspace, cache)[0] is None]
    print(f"Extracting embeddings for {len(missing)} images with saved fMRI", flush=True)
    for record in missing:
        embedding, = cache.encode(encoder, [record.video])
        write_embedding(record, embedding, workspace)
        print(f"Saved V-JEPA embedding for {record.index}", flush=True)


def cortical_regions(atlas_dir):
    from nilearn.datasets import fetch_atlas_surf_destrieux

    atlas = fetch_atlas_surf_destrieux(data_dir=atlas_dir)
    labels = {}
    names = {0: "Background"}
    for hemi, mapping in (("left", atlas.map_left), ("right", atlas.map_right)):
        if mapping.shape != (10242,):
            raise ValueError(f"Unexpected fsaverage5 vertex count for {hemi}")
        labels[hemi] = np.zeros_like(mapping)
        for index in np.unique(mapping):
            if index < 0 or atlas.labels[index].lower().replace("_", "") in {"unknown", "medialwall", "background"}:
                continue
            label_id = len(names)
            names[label_id] = f"{hemi}:{atlas.labels[index]}"
            labels[hemi][mapping == index] = label_id
    return labels, names


def cortical_features(t0, mean, atlas_dir):
    from nilearn.datasets import load_fsaverage
    from nilearn.maskers import SurfaceLabelsMasker
    from nilearn.surface import SurfaceImage

    labels, names = cortical_regions(atlas_dir)
    mesh = load_fsaverage(mesh="fsaverage5", data_dir=atlas_dir)["pial"]
    lut = pd.DataFrame({"index": list(names), "name": list(names.values())})
    atlas_image = SurfaceImage(mesh, labels)
    masker = SurfaceLabelsMasker(labels_img=atlas_image, lut=lut, strategy="mean",
                                 standardize=None, detrend=False, reports=False).fit()

    def transform(values):
        return masker.transform(SurfaceImage(mesh, {"left": values[:, :10242].T,
                                                     "right": values[:, 10242:].T}))

    mask = np.r_[labels["left"], labels["right"]] != 0
    names = [names[i] for i in masker.labels_ if i != masker.background_label]
    atlas_hash = {path.name: sha256(path) for path in sorted(
        (atlas_dir / "destrieux_surface").glob("*.annot"))}
    return transform(t0), transform(mean), mask, names, atlas_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "prepare-replay", "check-replay", "embed", "recover"])
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--index", type=int, help="Published index for source recovery")
    parser.add_argument("--image", type=Path, help="Complete archived copy of a truncated source")
    parser.add_argument("--source-url", help="Archive source URL for the recovered image")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    config, base = load_config(HERE / "config.json")
    frame, _, _, _ = load_dataset(config, base)
    rows = source_rows(frame, workspace)
    if args.action == "recover":
        if args.index not in rows or args.image is None or not args.source_url:
            parser.error("recover requires --index, --image, and --source-url")
        register_recovery(args.index, rows[args.index], args.image, args.source_url, workspace)
        rows = source_rows(frame, workspace)
        prepare(rows, workspace / "recovered_run", [args.index])
        return
    if args.action == "prepare-replay":
        prepare(rows, workspace / "replay", REPLAY_IDS)
        return
    records, rejected = prediction_records(rows, workspace)
    print(f"Validated {len(records)} fMRI files: {dict(Counter(r.directory.name for r in records.values()))}",
          flush=True)
    if rejected:
        workspace.mkdir(parents=True, exist_ok=True)
        write_json(workspace / "rejected_predictions.json", rejected)
    if args.action == "prepare":
        prepare(rows, workspace / "completed", set(rows) - records.keys())
    elif args.action == "check-replay":
        check_replay(rows, records, workspace, VisualCache())
    else:
        fill_embeddings(records, workspace, VisualCache(), args.device, args.cpu_threads)


if __name__ == "__main__":
    main()
