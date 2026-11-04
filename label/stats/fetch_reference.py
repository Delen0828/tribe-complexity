#!/usr/bin/env python3
"""Extract small, auditable code/text references from the public OSF supplement."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
URL = "https://osf.io/download/2d8jg/"
ARCHIVE_SHA256 = "f2c33a07db7f6212fed6608f904174a24f783bf504581b1a549b8e92c17c9da3"
PREFIX = "Supplemental Materials 2025/"
FILES = [
    "4. modeling_and_featureimportance/Linear Regression.ipynb",
    "4. modeling_and_featureimportance/SVR.ipynb",
    "4. modeling_and_featureimportance/Decision-Tree.ipynb",
    "4. modeling_and_featureimportance/svr_cv/svr_full_gridsearch.csv",
    "5. combined_analysis/feature_analysis.Rmd",
]


def extract(archive, destination):
    with archive.open("rb") as archive_file:
        digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"OSF archive checksum mismatch: {digest}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"url": URL, "archive_sha256": digest, "members": []}
    with zipfile.ZipFile(archive) as handle:
        for member in FILES:
            raw = handle.read(PREFIX + member)
            name = Path(member).name
            if name.endswith(".ipynb"):
                notebook = json.loads(raw)
                # Retain exact cell source and text output. Omit embedded raster plots.
                cells = []
                for index, cell in enumerate(notebook["cells"]):
                    outputs = []
                    for output in cell.get("outputs", []):
                        value = output.get("text", output.get("data", {}).get("text/plain", []))
                        if value:
                            outputs.append("".join(value))
                    cells.append({"index": index, "type": cell["cell_type"],
                                  "source": "".join(cell["source"]), "text_outputs": outputs})
                name = name.replace(".ipynb", ".cells.json")
                content = (json.dumps({"original_member": PREFIX + member, "cells": cells}, indent=2) + "\n").encode()
            else:
                content = raw
            (destination / name).write_bytes(content)
            manifest["members"].append({"member": PREFIX + member,
                                        "original_sha256": hashlib.sha256(raw).hexdigest(),
                                        "extracted_file": name,
                                        "extracted_sha256": hashlib.sha256(content).hexdigest()})
        original_data = handle.read(PREFIX + "4. modeling_and_featureimportance/data/all_labeled_MASSVIS_5800.csv")
        manifest["original_annotations_sha256"] = hashlib.sha256(original_data).hexdigest()
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified archive; extracted {len(FILES)} references to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded ZIP, without network access")
    parser.add_argument("--output", type=Path, default=ROOT / "reference")
    args = parser.parse_args()
    if args.archive:
        extract(args.archive, args.output)
    else:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "supplement.zip"
            with urlopen(URL, timeout=180) as response, archive.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            extract(archive, args.output)


if __name__ == "__main__":
    main()
