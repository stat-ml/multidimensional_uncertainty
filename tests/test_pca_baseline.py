import unittest

import numpy as np

from mdu.unc.pca_baseline import PCAUncertaintyOrdering


class PCAUncertaintyOrderingTest(unittest.TestCase):
    def test_predict_returns_finite_1d_scores(self):
        scores_cal = np.array(
            [
                [0.1, 2.0],
                [0.3, 1.7],
                [0.5, 1.2],
                [0.7, 0.8],
                [0.9, 0.4],
            ]
        )
        scores_test = np.array([[0.2, 1.8], [0.6, 1.0], [1.0, 0.2]])

        model = PCAUncertaintyOrdering().fit(scores_cal)
        scores = model.predict(scores_test)

        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_1d_pca_preserves_input_ordering(self):
        scores_cal = np.array([[0.0], [1.0], [2.0], [3.0]])
        scores_test = np.array([[2.5], [-1.0], [1.5], [4.0]])

        model = PCAUncertaintyOrdering().fit(scores_cal)
        pca_scores = model.predict(scores_test)

        np.testing.assert_array_equal(
            np.argsort(pca_scores), np.argsort(scores_test.ravel())
        )

    def test_sign_orientation_makes_positive_signal_increasing(self):
        scores_cal = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.2],
                [2.0, 1.9],
                [3.0, 3.1],
            ]
        )

        model = PCAUncertaintyOrdering().fit(scores_cal)
        pca_scores = model.predict(scores_cal)

        self.assertTrue(np.all(np.diff(pca_scores) > 0.0))

    def test_constant_calibration_is_deterministic(self):
        scores_cal = np.ones((5, 3))
        scores_test = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        model = PCAUncertaintyOrdering().fit(scores_cal)
        pca_scores = model.predict(scores_test)

        np.testing.assert_allclose(pca_scores, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
