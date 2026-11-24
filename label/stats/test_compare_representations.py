import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from nilearn.surface import InMemoryMesh, PolyMesh

from compare_representations import (
    FIXED_METRICS, MODELS, build_pipeline, inner_folds, make_splits, paired_mae, paper_split,
)
from complexity_features import cortical_features
from replicate import HERE, load_config, load_dataset, metrics


class RepresentationComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, base = load_config(HERE / "config.json")
        cls.frame, cls.metadata, cls.y, _ = load_dataset(cls.config, base)
        cls.paper = paper_split(cls.frame)

    def test_fixed_metadata_baseline_reproduces_published_metrics(self):
        train, test = self.paper
        published = pd.read_csv(FIXED_METRICS).query("profile == 'paper'").set_index("model")
        for model in MODELS:
            with self.subTest(model=model):
                fit = build_pipeline(self.config, self.metadata, "metadata", model)
                fit.fit(self.metadata.iloc[train], self.y.iloc[train])
                actual = metrics(self.y.iloc[test], fit.predict(self.metadata.iloc[test]))
                np.testing.assert_allclose(list(actual.values()), published.loc[model, list(actual)],
                                           rtol=1e-8, atol=1e-8)

    def test_paper_and_grouped_folds_preserve_membership(self):
        groups = np.arange(len(self.frame))
        groups[1] = groups[0]
        splits = make_splits(self.frame, groups)
        train, test = splits["paper"]
        np.testing.assert_array_equal(train, self.paper[0])
        np.testing.assert_array_equal(test, self.paper[1])
        grouped_train, grouped_test = splits["grouped"]
        self.assertEqual(len(np.intersect1d(groups[grouped_train], groups[grouped_test])), 0)
        for fit, validation in inner_folds(grouped_train, groups):
            self.assertEqual(len(np.intersect1d(groups[grouped_train[fit]],
                                                groups[grouped_train[validation]])), 0)

    def test_paired_mae_uses_shared_image_groups(self):
        y = np.zeros(3)
        result = paired_mae(y, np.array([2., 0., 0.]), np.zeros(3), np.array(["a", "a", "b"]))
        self.assertAlmostEqual(result["delta_mae"], 2 / 3)
        self.assertEqual(result["ci_low"], 0)
        self.assertEqual(result["ci_high"], 1)

    def test_surface_labels_preserve_hemisphere_and_signed_mean(self):
        mapping = np.ones(10242, dtype=np.int32)
        mapping[-1] = 2
        mapping[-2] = 0
        atlas = SimpleNamespace(map_left=mapping, map_right=mapping,
                                labels=["Unknown", "cortex", "Medial_wall"])
        coords = np.zeros((10242, 3))
        faces = np.array([[0, 1, 2]])
        mesh = PolyMesh(left=InMemoryMesh(coords, faces), right=InMemoryMesh(coords, faces))
        left = np.repeat([[-4.], [-9.]], 10242, axis=1)
        right = np.repeat([[6.], [8.]], 10242, axis=1)
        left[:, -2:] = 100
        right[:, -2:] = 100
        raw = np.hstack([left, right])
        with (patch("nilearn.datasets.fetch_atlas_surf_destrieux", return_value=atlas),
              patch("nilearn.datasets.load_fsaverage", return_value={"pial": mesh}),
              tempfile.TemporaryDirectory() as work):
            main, mean, mask, names, _ = cortical_features(raw, raw + 2, Path(work))
        np.testing.assert_array_equal(main, [[-4, 6], [-9, 8]])
        np.testing.assert_array_equal(mean, main + 2)
        self.assertEqual(names, ["left:cortex", "right:cortex"])
        self.assertEqual(mask.sum(), 20480)


if __name__ == "__main__":
    unittest.main()
