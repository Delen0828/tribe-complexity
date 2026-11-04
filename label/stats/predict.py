#!/usr/bin/env python3
"""Predict perceived complexity from raw canonical or exported MASSVIS labels."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from replicate import HERE, select_predictors


def predict(model_path, input_path, output_path):
    artifact = joblib.load(model_path)
    frame = pd.read_csv(input_path)
    predictors = select_predictors(frame, artifact["feature_mapping"], artifact["count_features"])
    result = frame[[column for column in ["image_id", "filename", "image_path"] if column in frame]].copy()
    result["predicted_complexity"] = artifact["pipeline"].predict(predictors)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=HERE / "results/models/supplement/rbf_svr.joblib")
    parser.add_argument("--input", type=Path, default=HERE.parent / "output/labels.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results/predicted_complexity.csv")
    args = parser.parse_args()
    result = predict(args.model, args.input, args.output)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
