"""Data-integrity, preprocessing, inference and publication regression checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from predict import predict
from replicate import (HERE, load_config, load_dataset, make_pipeline,
                       metrics, select_predictors, sha256, validate_predictors)


class DataAndPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.base = load_config(HERE / "config.json")
        cls.frame, cls.X, cls.y, cls.paths = load_dataset(cls.config, cls.base)

    def test_data_cover_all_published_images_and_ratings(self):
        self.assertEqual(self.X.shape, (5800, 29))
        self.assertEqual(sha256(self.paths["annotations_csv"]), self.config["dataset"]["expected_annotations_sha256"])
        self.assertEqual(list(self.X), list(self.config["dataset"]["features"]))
        self.assertNotIn("human_rating_sd", self.X)
        self.assertNotIn("mean_human_complexity_rating", self.X)

    def test_preprocessing_is_once_per_call_without_mutating_raw_labels(self):
        original = self.X.iloc[:40].copy(deep=True)
        pipeline = make_pipeline(self.config["models"]["linear"], list(self.X), self.config["preprocessing"])
        preprocess = pipeline["preprocess"]
        first = preprocess.fit_transform(original)
        second = preprocess.transform(original)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(original, self.X.iloc[:40])
        self.assertEqual(list(first), list(original))
        for column in original:
            expected = np.log1p(original[column]) if column in self.config["preprocessing"]["count_features"] else original[column]
            np.testing.assert_array_equal(first[column], expected)

    def test_missing_and_malformed_features_fail(self):
        for value in [np.nan, np.inf, -1, 1.5]:
            bad = self.X.iloc[:3].copy()
            bad.loc[0, "color_count"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_predictors(bad, self.config["preprocessing"]["count_features"])
        bad = self.X.iloc[:3].copy()
        bad.loc[0, "multi_panel"] = 2
        with self.assertRaises(ValueError):
            validate_predictors(bad, self.config["preprocessing"]["count_features"])
        with self.assertRaises(ValueError):
            select_predictors(self.X.drop(columns="color_count"), self.config["dataset"]["features"],
                              self.config["preprocessing"]["count_features"])

    def test_target_cannot_be_configured_as_predictor(self):
        config = deepcopy(self.config)
        config["dataset"]["features"]["mean_human_complexity_rating"] = "mean_human_complexity_rating"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "cannot be predictors"):
                load_config(path)

    def test_reference_archive_extracts_are_unchanged(self):
        manifest = json.loads((HERE / "reference/provenance.json").read_text())
        self.assertEqual(manifest["original_annotations_sha256"], sha256(self.paths["annotations_csv"]))
        for member in manifest["members"]:
            self.assertEqual(sha256(HERE / "reference" / member["extracted_file"]), member["extracted_sha256"])


@unittest.skipUnless((HERE / "results/manifest.json").exists(), "Run replicate.py with both default profiles first")
class CompletedReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = HERE / "results"
        cls.predictions = pd.read_csv(cls.results / "predictions.csv")
        cls.manifest = json.loads((cls.results / "manifest.json").read_text())
        cls.observed_metrics = pd.read_csv(cls.results / "metrics.csv")

    def test_exact_published_partition_is_disjoint_and_complete(self):
        split = pd.read_csv(self.results / "split.csv")
        train, test = train_test_split(np.arange(5800), test_size=0.2, random_state=0)
        np.testing.assert_array_equal(split[split.split == "train"].source_row, train)
        np.testing.assert_array_equal(split[split.split == "test"].source_row, test)
        self.assertEqual(split.image_id.nunique(), 5800)
        self.assertEqual(set(train).intersection(test), set())
        self.assertEqual(len(test), 1160)

    def test_saved_predictions_reproduce_every_metric(self):
        test = self.predictions[self.predictions.split == "test"]
        for row in self.observed_metrics.to_dict("records"):
            computed = metrics(test.observed, test[f"{row['profile']}.{row['model']}"])
            for name, value in computed.items():
                self.assertAlmostEqual(value, row[name], places=10)

    def test_supplement_matches_all_published_metrics_and_importances(self):
        for filename, count in [("metric_comparison.csv", 15), ("importance_comparison.csv", 25)]:
            comparison = pd.read_csv(self.results / filename)
            supplement = comparison[comparison.profile == "supplement"]
            self.assertEqual(len(supplement), count)
            self.assertTrue(supplement.matches_printed_precision.all(), supplement.to_string())

    def test_paper_hyperparameter_discrepancies_are_visible(self):
        comparison = pd.read_csv(self.results / "metric_comparison.csv")
        for model in ["lasso", "decision_tree"]:
            rows = comparison[(comparison.profile == "paper") & (comparison.model == model)]
            self.assertFalse(rows.matches_printed_precision.all())
        fitted = self.manifest["models"]["supplement"]
        self.assertAlmostEqual(fitted["lasso"]["alpha_"], 0.016882054461360676)
        self.assertEqual(fitted["decision_tree"]["params"]["max_depth"], 5)
        self.assertEqual(fitted["decision_tree"]["params"]["min_samples_split"], 65)
        self.assertEqual(fitted["decision_tree"]["params"]["min_samples_leaf"], 8)

    def test_model_reload_and_both_input_schemas_preserve_predictions(self):
        model = self.results / "models/supplement/rbf_svr.joblib"
        artifact = joblib.load(model)
        labels = pd.read_csv(HERE.parent / "output/labels.csv").iloc[:12]
        annotations = pd.read_csv(HERE.parent / "source/annotations.csv").iloc[:12]
        with tempfile.TemporaryDirectory() as directory:
            for name, frame in [("labels", labels), ("annotations", annotations)]:
                source = Path(directory) / f"{name}.csv"
                frame.to_csv(source, index=False)
                result = predict(model, source, Path(directory) / "predictions.csv")
                np.testing.assert_allclose(result.predicted_complexity, self.predictions["supplement.rbf_svr"].iloc[:12], atol=1e-11)
        self.assertEqual(list(artifact["pipeline"].feature_names_in_), list(artifact["feature_mapping"]))

    def test_original_permutation_scope_and_scores_are_preserved(self):
        importance = pd.read_csv(self.results / "feature_importance.csv")
        selected = importance[(importance.profile == "supplement") & (importance.model == "rbf_svr")]
        self.assertEqual(len(selected), 29)
        self.assertEqual(set(selected.dataset), {"all"})
        top = selected.sort_values("rank").iloc[0]
        self.assertEqual(top.feature, "chart_count")
        self.assertAlmostEqual(top.importance, 72.354430, places=5)
        self.assertEqual(top.scaled, 1)
        repeats = pd.read_csv(self.results / "permutation_repeats.csv")
        self.assertEqual(len(repeats[repeats.profile == "supplement"]), 29 * 5)

    def test_no_fitting_warnings(self):
        self.assertEqual(self.manifest["warnings"], [])

    def test_recorded_code_and_model_checksums_match_artifacts(self):
        for filename, checksum in self.manifest["code_sha256"].items():
            self.assertEqual(sha256(HERE / filename), checksum, filename)
        for profile, models in self.manifest["models"].items():
            for name, details in models.items():
                self.assertEqual(sha256(self.results / "models" / profile / f"{name}.joblib"), details["artifact_sha256"])

    def test_statistics_match_source_contrasts_and_untransformed_correlations(self):
        statistics = pd.read_csv(self.results / "feature_statistics.csv").set_index("feature")
        self.assertEqual(statistics.loc["multi_panel", "contrast"], "present_minus_absent")
        self.assertGreater(statistics.loc["multi_panel", "statistic"], 0)
        self.assertAlmostEqual(statistics.loc["no_text", "p_value"], 0.07539719, places=7)
        self.assertAlmostEqual(statistics.loc["chart_count", "statistic"], 0.433749, places=6)


if __name__ == "__main__":
    unittest.main()
