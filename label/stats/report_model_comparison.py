"""Render one model-by-representation table per test metric."""
import argparse
from pathlib import Path

import pandas as pd

from complexity_features import WORKSPACE

SOURCES = {"metadata": "Metadata", "vjepa2": "V-JEPA 2", "tribe_raw": "TRIBE v2"}
MODELS = {
    ("ridge", "tuned"): "Ridge",
    ("rbf_svr", "tuned"): "RBF-SVR",
    ("cortical_transformer", "tuned"): "Cortical Transformer",
    ("resnet50", "tuned"): "ResNet-50",
    ("vit_b_16", "tuned"): "ViT-B/16",
    ("ridge", "published"): "Ridge (published)",
    ("rbf_svr", "published"): "RBF-SVR (published)",
}
METRICS = {"mae": ("MAE", "lower"), "rmse": ("RMSE", "lower"),
           "mse": ("MSE", "lower"), "r2": ("R²", "higher")}
PROTOCOLS = ("paper", "grouped")
IMAGE_NOTE = ("For original chart images, ResNet-50 and ViT-B/16 are fine-tuned end to end from "
              "ImageNet initialization and directly predict a complexity score. The TRIBE v2 cells "
              "for those architectures use end-to-end regressors trained from random initialization "
              "on the cortical tensor.")


def sources_for(results):
    if results.features.eq("chart_image").any():
        return {**SOURCES, "chart_image": "Original chart image"}
    return SOURCES


def metric_table(results, metric):
    sources = sources_for(results)
    indexed = results.set_index(["protocol", "features", "regressor", "setting"])
    if not indexed.index.is_unique:
        raise ValueError("Duplicate rows in the model comparison")
    rows = {}
    for (regressor, setting), name in MODELS.items():
        if not ((results.regressor == regressor) & (results.setting == setting)).any():
            continue
        values = []
        for feature in sources:
            keys = [(split, feature, regressor, setting) for split in PROTOCOLS]
            scores = [indexed.at[key, metric] if key in indexed.index else None for key in keys]
            values.append(" / ".join("—" if score is None else f"{score:.3f}"
                                     for score in scores) if any(score is not None for score in scores) else "—")
        rows[name] = values
    return pd.DataFrame.from_dict(rows, orient="index", columns=sources.values()).rename_axis("Model")


def save_report(results, output, notes=()):
    output.mkdir(parents=True, exist_ok=True)
    results = results.loc[results.features.isin(sources_for(results)) & results.protocol.isin(PROTOCOLS)]
    lines = ["# MASSVIS model comparison", "",
             "Each cell shows **published split / SHA-256-grouped split** test scores. "
             "An em dash indicates that the experiment has no score for that combination.", "",
             "The TRIBE v2 column uses all three time points and 20,484 fsaverage5 vertices per image. "
             "The Ridge and RBF-SVR rows compare V-JEPA 2 and TRIBE v2 under the same regressor families. "
             "The cortical Transformer was trained from random initialization. It uses a Perceiver-style "
             "attention layout.", ""]
    for note in notes:
        lines.extend([note, ""])
    if results.features.eq("chart_image").any():
        lines.extend([IMAGE_NOTE, ""])
    for metric, (label, direction) in METRICS.items():
        table = metric_table(results, metric)
        table.to_csv(output / f"{metric}.csv")
        lines.extend([f"## {label} ({direction} is better)", "",
                      "| Model | " + " | ".join(table.columns) + " |",
                      "|---|" + "|".join("---:" for _ in table.columns) + "|"])
        lines.extend("| " + " | ".join([name, *values]) + " |" for name, values in table.iterrows())
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=WORKSPACE / "vision_models" / "metrics.csv")
    parser.add_argument("--image-metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = pd.read_csv(args.metrics)
    if args.image_metrics:
        image_scores = pd.read_csv(args.image_metrics)
        if (not image_scores.features.eq("chart_image").all()
                or not image_scores.regressor.isin(["resnet50", "vit_b_16"]).all()):
            parser.error("--image-metrics requires direct chart-image scores from ResNet-50 and ViT-B/16")
        results = pd.concat([results, image_scores], ignore_index=True)
    save_report(results, args.output or args.metrics.parent)


if __name__ == "__main__":
    main()
