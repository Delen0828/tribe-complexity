import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from interpret_ridge import ALPHAS, fit_split, permutation_predictions


class RidgeInterpretationTests(unittest.TestCase):
    def test_against_explicit_fits_and_joint_permutations(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(68, 3, 9)).astype(np.float32)
        x[:, 1, 3] = 3  # A constant feature and a strongly correlated pair.
        x[:, 1, 4] = x[:, 2, 2] * .99 + .01 * x[:, 1, 4]
        y = 8 + 4 * x[:, 0, 1].astype(float) - 3 * x[:, 2, 7]
        labels = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3])
        train, test = np.arange(46), np.arange(46, 68)
        tuning = fit_split(x, y, train, test, labels, "cpu")["errors"]
        selected = tuning.argmin(axis=1)
        output = fit_split(x, y, train, test, labels, "cpu", selected)
        raw = x.reshape(len(y), -1).astype(float)
        scaled = StandardScaler().fit(raw[train]).transform(raw)
        order = rng.permutation(len(test))
        permuted = permutation_predictions(output["full"], output["parts"], order)
        for region in range(4):
            mask = np.tile(labels != region, 3) if region else np.ones(raw.shape[1], bool)
            for j, alpha in enumerate(ALPHAS):
                model = Ridge(alpha=alpha).fit(scaled[train][:, mask], y[train])
                prediction = model.predict(scaled[test][:, mask])
                np.testing.assert_allclose(tuning[region, j], abs(y[test] - prediction).mean(), atol=1e-7)
                if j == selected[region]:
                    actual = output["removed"][:, region-1] if region else output["full"]
                    np.testing.assert_allclose(actual, prediction, atol=1e-7)
                    if not region:
                        complete = model
            if region:
                shuffled = scaled[test].copy()
                shuffled[:, ~mask] = shuffled[order][:, ~mask]
                np.testing.assert_allclose(permuted[:, region-1], complete.predict(shuffled), atol=1e-7)
        pred = complete.predict(scaled[train])
        pattern = (raw[train] - raw[train].mean(0)).T @ (pred - pred.mean()) / ((pred - pred.mean())**2).sum()
        np.testing.assert_allclose(output["pattern"].ravel(), pattern, atol=1e-8)
        self.assertGreater(output["pattern"][0, 1], 0)
        self.assertLess(output["pattern"][2, 7], 0)
