"""Freeze a complexity-balanced sample from the existing training partition."""
import csv
import random
from PIL import Image
from common import CV, ROOT, file_hash, digest, read_json, write_json, now


def select():
    config = read_json(CV / "config.json")
    labels_path = ROOT / "label/output/labels.json"
    split_path = ROOT / "label/stats/results/split.csv"
    labels = {r["image_id"]: r for r in read_json(labels_path)}
    with split_path.open() as f:
        train_ids = [r["image_id"] for r in csv.DictReader(f) if r["split"] == "train"]
    if len(train_ids) != 4640 or len(set(train_ids)) != 4640:
        raise ValueError("Expected the frozen 4,640-image training partition")
    rng = random.Random(config["seed"])
    records = []
    for band in range(10):
        pool = sorted((labels[i] for i in train_ids if min(9, int(labels[i]["mean_human_complexity_rating"] // 10)) == band), key=lambda r: r["index"])
        for r in rng.sample(pool, config["sample_per_band"]):
            source = ROOT / r["image_path"]
            with Image.open(source) as image:
                width, height = image.size
                target = CV / "results/images" / r["image_id"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                        rgba = image.convert("RGBA")
                        bg = Image.new("RGBA", rgba.size, "white")
                        bg.alpha_composite(rgba)
                        bg.convert("RGB").save(target)
                    else:
                        image.convert("RGB").save(target)
            records.append({k: r[k] for k in ("index", "image_id", "filename", "image_path", "category", "mean_human_complexity_rating", "design")} | {"band": band, "width": width, "height": height, "source_sha256": file_hash(source), "reference_sha256": file_hash(target), "eligible_in_band": len(pool)})
    identity = {"seed": config["seed"], "per_band": config["sample_per_band"], "labels_sha256": file_hash(labels_path), "split_sha256": file_hash(split_path), "records": records}
    manifest = identity | {"selection_id": digest(identity), "created_at": now(), "binning": "[0,10), [10,20), ..., [90,100]", "reference": "Original dimensions, RGB conversion, transparency composited over white; no geometric resizing"}
    path = CV / "selection.json"
    if path.exists() and read_json(path)["selection_id"] != manifest["selection_id"]:
        raise ValueError("Frozen selection changed; preserve or explicitly archive the previous selection first")
    if not path.exists():
        write_json(path, manifest)
    for shard in range(3):
        write_json(CV / f"preparation/batch_{shard}.json", {"selection_id": manifest["selection_id"], "records": records[shard::3]})
    print(f"Selection {manifest['selection_id'][:12]}: {len(records)} training figures, 15 per score band", flush=True)
    return read_json(path)


if __name__ == "__main__":
    select()
