"""Shared, model-independent data contracts and original-coordinate masks."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from datetime import datetime, timezone

CV = Path(__file__).resolve().parent
ROOT = CV.parent.parent
FAMILIES = {"point", "glyph", "bar_segment", "pie_slice", "network_node", "data_icon", "cell", "region", "other_discrete"}


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def encode_mask(mask):
    """Compact bounding-box-local COCO-order RLE with original-image offset."""
    import numpy as np
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return {"origin": [0, 0], "size": [0, 0], "counts": [0]}
    x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
    flat = np.asarray(mask[y0:y1, x0:x1], dtype=np.uint8).flatten(order="F")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.r_[0, changes, len(flat)]).tolist()
    if flat[0]:
        runs.insert(0, 0)
    return {"origin": [x0, y0], "size": [y1 - y0, x1 - x0], "counts": runs}


def decode_mask(rle):
    import numpy as np
    h, w = rle["size"]
    if sum(rle["counts"]) != h * w:
        raise ValueError("RLE length does not match dimensions")
    flat = np.zeros(h * w, dtype=bool)
    cursor = 0
    for i, count in enumerate(rle["counts"]):
        if count < 0:
            raise ValueError("Negative RLE run")
        if i % 2:
            flat[cursor:cursor + count] = True
        cursor += count
    return flat.reshape((h, w), order="F")


def rle_bbox(rle):
    x, y = rle["origin"]
    h, w = rle["size"]
    return [x, y, x + w, y + h]


def mask_iou(a, b):
    import numpy as np
    ax0, ay0, ax1, ay1 = rle_bbox(a)
    bx0, by0, bx1, by1 = rle_bbox(b)
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ma, mb = decode_mask(a), decode_mask(b)
    intersection = int(np.count_nonzero(ma[y0-ay0:y1-ay0, x0-ax0:x1-ax0] & mb[y0-by0:y1-by0, x0-bx0:x1-bx0]))
    union = int(ma.sum()) + int(mb.sum()) - intersection
    return intersection / union if union else 0.0


def deduplicate(candidates, threshold=0.9):
    retained = []
    for candidate in sorted(candidates, key=lambda c: (-c["score"], c["candidate_id"])):
        a = candidate["area"]
        b0 = rle_bbox(candidate["rle"])
        def possible(other):
            b = other["area"]
            if not max(a, b) or min(a, b) / max(a, b) < threshold:
                return False
            b1 = rle_bbox(other["rle"])
            overlap = max(0, min(b0[2], b1[2]) - max(b0[0], b1[0])) * max(0, min(b0[3], b1[3]) - max(b0[1], b1[1]))
            cap = min(a, b, overlap)
            return cap / max(1, a + b - cap) >= threshold
        duplicate = next((c for c in retained if possible(c) and mask_iou(c["rle"], candidate["rle"]) >= threshold), None)
        if duplicate:
            candidate["duplicate_of"] = duplicate["candidate_id"]
        else:
            retained.append(candidate)
    return candidates


def validate_preparation(record, selected):
    if record["figure_id"] != selected["image_id"]:
        raise ValueError("Preparation figure ID mismatch")
    if record.get("source_sha256", selected["source_sha256"]) != selected["source_sha256"]:
        raise ValueError("Preparation source hash mismatch")
    w, h = selected["width"], selected["height"]
    def box(b, container=None):
        if len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
            raise ValueError(f"Invalid box {b}")
        x0, y0, x1, y1 = b
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"Box outside {w}x{h}: {b}")
        if container and not (container[0] <= x0 < x1 <= container[2] and container[1] <= y0 < y1 <= container[3]):
            raise ValueError(f"Box {b} outside panel {container}")
    ids = set()
    for panel in record["panels"]:
        if panel["panel_id"] in ids:
            raise ValueError("Duplicate panel ID")
        ids.add(panel["panel_id"])
        box(panel["bbox"])
        for b in panel.get("exclusions", []):
            box(b, panel["bbox"])
        families = panel.get("mark_families", [])
        if panel["applicable"] and not families:
            raise ValueError("Applicable panel needs mark families")
        for family in families:
            if family["name"] not in FAMILIES or not family.get("prompt"):
                raise ValueError("Invalid family or missing prompt")
            if not family.get("exemplar_boxes"):
                raise ValueError("A mark family needs an exemplar")
            for b in family["exemplar_boxes"]:
                box(b, panel["bbox"])
    if not record["panels"]:
        raise ValueError("Every figure needs at least one panel, including N/A figures")
    return True
