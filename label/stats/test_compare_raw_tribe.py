import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch

from compare_raw_tribe import gram_matrices, rbf_predictions, ridge_predictions, standardized_data
from cortical_transformer import CorticalTransformer, cortical_patches


class RawTribeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.responses = rng.normal(size=(43, 3, 31)).astype(np.float32)
        self.responses[:31, 1, 6] = 2
        self.responses[31:, 1, 6] = 8
        self.ratings = rng.normal(50, 15, 31)
        self.train, self.test = standardized_data(self.responses, np.arange(31), np.arange(31, 43),
                                                 torch.device("cpu"))

    def test_scaling_uses_training_rows_and_keeps_constant_coordinate(self):
        array = self.responses.reshape(len(self.responses), -1).astype(np.float64)
        reference = StandardScaler().fit(array[:31])
        np.testing.assert_allclose(self.train.flatten(1), reference.transform(array[:31]), atol=1e-10)
        np.testing.assert_allclose(self.test.flatten(1), reference.transform(array[31:]), atol=1e-10)
        np.testing.assert_array_equal(self.test[:, 1, 6], np.full(12, 6.0))

    def test_sample_space_ridge_matches_primal_objective(self):
        gram, cross, _ = gram_matrices(self.train, self.test)
        actual = ridge_predictions(gram, cross, self.ratings, [0.01, 1, 10])
        for index, alpha in enumerate([0.01, 1, 10]):
            reference = Ridge(alpha=alpha).fit(self.train.flatten(1), self.ratings)
            np.testing.assert_allclose(actual[:, index], reference.predict(self.test.flatten(1)), atol=1e-8)

    def test_reused_kernel_matches_direct_svr(self):
        settings = {0: {"C": 10, "epsilon": 1, "gamma_multiplier": 1.0},
                    1: {"C": 100, "epsilon": 0.1, "gamma_multiplier": 0.25}}
        gram, cross, norms = gram_matrices(self.train, self.test)
        predictions = rbf_predictions(gram, cross, norms, self.ratings, settings, 2)
        train = self.train.flatten(1).numpy()
        for i, setting in settings.items():
            gamma = setting["gamma_multiplier"] / (train.shape[1] * train.var())
            reference = SVR(C=setting["C"], epsilon=setting["epsilon"], gamma=gamma)
            reference.fit(train, self.ratings)
            np.testing.assert_allclose(predictions[i], reference.predict(self.test.flatten(1)), atol=1e-6)

    def test_transformer_tokenization_preserves_each_time_vertex_value(self):
        patches, positions = cortical_patches()
        self.assertEqual(len(patches), 324)
        model = CorticalTransformer(patches, positions)
        rows = torch.arange(3 * 20484, dtype=torch.float32).reshape(1, 3, 20484)
        tokens = model.patch_values(rows).reshape(324, 3, patches.shape[1])
        for t in range(3):
            np.testing.assert_array_equal(tokens[:, t].numpy()[patches != 20484],
                                          rows[0, t].numpy()[patches[patches != 20484]])
            self.assertTrue(torch.all(tokens[:, t][patches == 20484] == 0))
        self.assertEqual(model(rows).shape, (1,))


if __name__ == "__main__":
    unittest.main()
