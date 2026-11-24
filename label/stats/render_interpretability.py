"""Render the five signed maps before publishing the browser report."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from complexity_features import TRIBE
sys.path.insert(0, str(TRIBE.parent))
from visualizer.cortical import brain


def publish(folder, report, pattern, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5", data_dir=str(TRIBE / "cache/nilearn"))
    maps = {key: (np.r_[np.nan, [row[key] for row in report["regions"]]][labels], key)
            for key in ("ablation", "permutation")}
    for t, values in enumerate(pattern):
        maps[f"haufe-{t}"] = (values, "haufe")
    panels = [("both", "ventral"), ("left", "lateral"), ("right", "lateral"),
              ("left", "medial"), ("right", "medial")]
    for name, (values, metric) in maps.items():
        for hemi, view in panels:
            fig = plt.figure(figsize=(3.7, 3.1), facecolor="white")
            try:
                brain(fig, 111, fs, values, hemi, view, max(report["limits"][metric], 1e-12))
                fig.subplots_adjust(left=0, right=1, top=.93, bottom=0)
                with tempfile.NamedTemporaryFile(dir=folder, suffix=".png", delete=False) as tmp:
                    fig.savefig(tmp, format="png", dpi=125)
                Path(tmp.name).replace(folder / f"{name}-{hemi}-{view}.png")
            finally:
                plt.close(fig)
    report["run"] = folder.name
    with tempfile.NamedTemporaryFile(dir=folder.parent, mode="w", delete=False) as tmp:
        json.dump(report, tmp, allow_nan=False)
    Path(tmp.name).replace(folder.parent / "report.json")
