import unittest

import numpy as np

from mdu.unc.additive_baseline import AdditiveUncertaintyOrdering


class AdditiveUncertaintyOrderingTest(unittest.TestCase):
    def test_predict_returns_row_sums(self):
        scores_cal = np.array([[0.0, 1.0], [2.0, 3.0]])
        scores_test = np.array([[1.0, 2.0], [3.0, 5.0], [-1.0, 4.0]])

        model = AdditiveUncertaintyOrdering().fit(scores_cal)
        additive_scores = model.predict(scores_test)

        np.testing.assert_allclose(additive_scores, np.array([3.0, 8.0, 3.0]))

    def test_1d_additive_preserves_input_scores(self):
        scores_cal = np.array([[0.0], [1.0], [2.0]])
        scores_test = np.array([[3.0], [-1.0], [0.5]])

        model = AdditiveUncertaintyOrdering().fit(scores_cal)
        additive_scores = model.predict(scores_test)

        np.testing.assert_allclose(additive_scores, scores_test.ravel())

    def test_rejects_feature_mismatch(self):
        model = AdditiveUncertaintyOrdering().fit(np.ones((3, 2)))

        with self.assertRaises(ValueError):
            model.predict(np.ones((3, 3)))


if __name__ == "__main__":
    unittest.main()
