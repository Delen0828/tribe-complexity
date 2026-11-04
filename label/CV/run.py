#!/usr/bin/env python3
"""Reproducible stage CLI; paths resolve relative to this project."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
import traceback
import subprocess
import concurrent.futures
from PIL import Image
from common import CV, ROOT, digest, file_hash, read_json, write_json, now, validate_preparation, deduplicate, rle_bbox, decode_mask


def base_run(config, model, checkpoint):
    selection = read_json(CV / "selection.json")
    checkpoint = Path(checkpoint or CV / ("checkpoints/sam2.1_hiera_small.pt" if model == "sam2.1-hiera-small" else "checkpoints/sam3.pt"))
    identity = {"selection_id": selection["selection_id"], "model": model, "config": config, "checkpoint_sha256": file_hash(checkpoint), "pipeline_version": 1}
    run_id = digest(identity)[:20]
    directory = CV / "results/runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    if not path.exists():
        write_json(path, identity | {"run_id": run_id, "created_at": now()})
    write_json(CV / "results/active_run.json", {"run_id": run_id})
    return directory, selection


def active_run(run_id=None):
    return CV / "results/runs" / (run_id or read_json(CV / "results/active_run.json")["run_id"])


def load_preparation(row):
    path = CV / "preparation/figures" / f"{row['image_id']}.json"
    if not path.exists():
        return None
    prep = read_json(path)
    validate_preparation(prep, row)
    return prep


def process_figure(row, prep, backend, directory, config, force=False):
    path = directory / "candidates" / f"{row['image_id']}.json"
    prep_hash = digest(prep)
    if path.exists() and not force:
        previous = read_json(path)
        if previous["preparation_sha256"] == prep_hash and not previous.get("errors"):
            return previous
    started = time.perf_counter()
    panels, candidates, errors = [], [], []
    image = Image.open(CV / "results/images" / row["image_id"]).convert("RGB")
    for panel in prep["panels"]:
        p = {"panel_id": panel["panel_id"], "bbox": panel["bbox"], "applicable": panel["applicable"], "mark_families": [f["name"] for f in panel.get("mark_families", [])], "notes": panel.get("notes", ""), "preparation_status": panel.get("status", "prepared")}
        if not panel["applicable"]:
            p.update(status="not_applicable", candidate_count=0)
            panels.append(p)
            continue
        try:
            panel_candidates = []
            for c in backend.segment(image, panel):
                if not c["area"]:
                    continue
                c.update(candidate_id=f"{panel['panel_id']}-{len(panel_candidates)+1:05d}", panel_id=panel["panel_id"], families=p["mark_families"])
                bx0, by0, bx1, by1 = rle_bbox(c["rle"])
                for ex in panel.get("exclusions", []):
                    if ex[0] <= (bx0+bx1)/2 <= ex[2] and ex[1] <= (by0+by1)/2 <= ex[3]:
                        c.update(automatic_rejection="excluded_region")
                        break
                panel_candidates.append(c)
            deduplicate(panel_candidates, config["dedup_mask_iou"])
            candidates.extend(panel_candidates)
            p.update(status="segmented", candidate_count=len(panel_candidates))
        except Exception as exc:
            p.update(status="failed", error=str(exc), candidate_count=0)
            errors.append({"panel_id": panel["panel_id"], "error": str(exc), "traceback": traceback.format_exc()})
            if "out of memory" in str(exc).lower():
                import torch
                torch.cuda.empty_cache()
        panels.append(p)
    status = "partial" if errors and any(p["status"] == "segmented" for p in panels) else "failed" if errors else "segmented" if any(p["applicable"] for p in panels) else "not_applicable"
    result = {"figure_id": row["image_id"], "source_sha256": row["source_sha256"], "preparation_sha256": prep_hash, "status": status, "panels": panels, "candidates": candidates, "errors": errors, "runtime_seconds": time.perf_counter()-started, "completed_at": now()}
    result["candidate_set_id"] = digest({k: result[k] for k in ["figure_id", "source_sha256", "preparation_sha256", "panels", "candidates"]})
    write_json(path, result)
    print(f"{row['image_id']}: {status}; {len(candidates)} masks, {result['runtime_seconds']:.1f}s", flush=True)
    return result


def segment(args):
    if args.workers > 1 and args.worker_index is None:
        commands = []
        for index in range(args.workers):
            command = [sys.executable, str(CV/"run.py"), *sys.argv[1:], "--worker-index", str(index)]
            commands.append(subprocess.Popen(command))
        codes = [p.wait() for p in commands]
        if any(codes):
            raise SystemExit(1)
        return
    from backends import create_backend
    config = read_json(CV / "config.json")
    model = args.model or config["model"]
    directory, selection = base_run(config, model, args.checkpoint)
    rows = [r for r in selection["records"] if not args.ids or r["image_id"] in args.ids.split(",")]
    if args.worker_index is not None:
        rows = rows[args.worker_index::args.workers]
    backend = create_backend(model, config, args.checkpoint)
    metadata = read_json(directory / "run.json")
    metadata["provenance"] = backend.provenance()
    write_json(directory / "run.json", metadata)
    import torch
    torch.set_num_threads(4)
    done, started = set(), time.monotonic()
    while len(done) < len(rows):
        progressed = False
        for row in rows:
            if row["image_id"] in done:
                continue
            prep = load_preparation(row)
            if prep is None:
                continue
            process_figure(row, prep, backend, directory, config, args.force)
            done.add(row["image_id"])
            progressed = True
        if not args.watch:
            break
        if len(done) < len(rows) and not progressed:
            if time.monotonic() - started > args.timeout:
                raise TimeoutError("Timed out waiting for visual preparation")
            time.sleep(5)
    print(f"Run {directory.name}: visited {len(done)}/{len(rows)} figures", flush=True)


def verify(args):
    from collections import Counter
    selection = read_json(CV / "selection.json")
    rows = selection["records"]
    errors = []
    if len(rows) != 150 or len({r["image_id"] for r in rows}) != 150 or Counter(r["band"] for r in rows) != Counter({i:15 for i in range(10)}):
        errors.append("Selection is not 150 unique images with 15 per band")
    labels_path, split_path = ROOT / "label/output/labels.json", ROOT / "label/stats/results/split.csv"
    if file_hash(labels_path) != selection["labels_sha256"] or file_hash(split_path) != selection["split_sha256"]:
        errors.append("Source labels/split changed")
    with split_path.open() as f:
        train = {r["image_id"] for r in csv.DictReader(f) if r["split"] == "train"}
    directory = active_run(args.run_id) if (CV / "results/active_run.json").exists() else None
    for row in rows:
        try:
            assert row["image_id"] in train, "Not a training image"
            assert file_hash(ROOT / row["image_path"]) == row["source_sha256"], "Source changed"
            assert file_hash(CV / "results/images" / row["image_id"]) == row["reference_sha256"], "Reference changed"
            prep = load_preparation(row)
            assert prep is not None, "Missing preparation"
            if directory:
                path = directory / "figures" / f"{row['image_id']}.json"
                assert path.exists(), "Missing rendered result"
                result = read_json(path)
                assert result.get("preparation_sha256") == digest(prep), "Rendered preparation is stale"
                raw = read_json(directory / "candidates" / f"{row['image_id']}.json")
                assert result.get("candidate_set_id") == raw["candidate_set_id"], "Rendered candidate set is stale"
                retained = [c for c in result["instances"] if c["decision"] == "retain"]
                expected = len(retained) if any(p["status"] == "segmented" for p in result["panels"]) and not result.get("pending_count", 0) else None
                assert result["count"] == expected, "Count disagrees with masks"
                assert not result.get("pending_count", 0), "Candidate triage is incomplete"
                for c in result["instances"]:
                    mask = decode_mask(c["rle"])
                    assert int(mask.sum()) == c["area"], "Mask area mismatch"
                    b = rle_bbox(c["rle"])
                    assert 0 <= b[0] <= b[2] <= row["width"] and 0 <= b[1] <= b[3] <= row["height"], "Mask outside native image"
                for asset in result["assets"].values():
                    assert (CV / asset).exists(), f"Missing asset {asset}"
                    with Image.open(CV / asset) as image:
                        if "thumbnail" not in asset:
                            assert image.size == (row["width"],row["height"]), "Export changed reference dimensions"
        except Exception as exc:
            errors.append(f"{row['image_id']}: {exc}")
    report = {"verified_at": now(), "figures": len(rows), "passed": not errors, "errors": errors}
    write_json(CV / "results/verification.json", report)
    print(f"Verification: {'PASS' if not errors else 'FAIL'}; {len(errors)} errors", flush=True)
    for error in errors[:20]:
        print(error)
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("select")
    p = sub.add_parser("segment")
    p.add_argument("--model", choices=["sam2.1-hiera-small", "sam3"])
    p.add_argument("--checkpoint")
    p.add_argument("--ids", help="Comma-separated study IDs, including .png")
    p.add_argument("--force", action="store_true")
    p.add_argument("--watch", action="store_true", help="Process preparation files as they arrive")
    p.add_argument("--timeout", type=int, default=14400)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    for stage in ["sheets", "render", "verify"]:
        p = sub.add_parser(stage)
        p.add_argument("--run-id")
        p.add_argument("--ids")
    args = parser.parse_args()
    if args.stage == "select":
        from sampling import select
        select()
    elif args.stage == "segment":
        segment(args)
    elif args.stage == "verify":
        verify(args)
    else:
        from render import sheets, render
        (sheets if args.stage == "sheets" else render)(args)


if __name__ == "__main__":
    main()
