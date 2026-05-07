import unittest

import numpy as np

from mdu.unc.constants import OTTarget, SamplingMethod, ScalingType
from mdu.unc.entropic_ot import EntropicOTOrdering


def make_model(target=OTTarget.BETA, scaling_type=ScalingType.FEATURE_WISE):
    return EntropicOTOrdering(
        target=target,
        sampling_method=SamplingMethod.GRID,
        scaling_type=scaling_type,
        grid_size=0,
        target_params={},
        eps=0.2,
        n_targets_multiplier=1,
        max_iters=300,
        random_state=0,
        tol=1e-9,
    )


class EntropicOTOrdering1DTest(unittest.TestCase):
    def test_1d_predictions_are_non_decreasing(self):
        scores_cal = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        scores_test = np.linspace(-0.2, 1.2, 15).reshape(-1, 1)

        for target in [OTTarget.EXP, OTTarget.BETA]:
            with self.subTest(target=target):
                model = make_model(target=target).fit(scores_cal)
                ranks = model.predict(scores_test)

                self.assertTrue(np.all(np.diff(ranks) >= -1e-8))

    def test_1d_order_matches_empirical_rank_order(self):
        scores_cal = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        scores_test = np.array([[0.7], [0.1], [1.1], [-0.1], [0.4]])

        model = make_model(target=OTTarget.BETA).fit(scores_cal)
        ranks = model.predict(scores_test)

        np.testing.assert_array_equal(
            np.argsort(ranks), np.argsort(scores_test.ravel())
        )

    def test_feature_wise_and_global_scaling_agree_in_1d(self):
        scores_cal = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        scores_test = np.linspace(-0.1, 1.1, 6).reshape(-1, 1)

        feature_wise = make_model(scaling_type=ScalingType.FEATURE_WISE).fit(
            scores_cal
        )
        global_scaled = make_model(scaling_type=ScalingType.GLOBAL).fit(scores_cal)

        np.testing.assert_allclose(
            feature_wise.predict(scores_test),
            global_scaled.predict(scores_test),
            atol=1e-8,
        )

    def test_positive_affine_invariance_under_minmax_scaling(self):
        scores_cal = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        scores_test = np.array([[-0.2], [0.1], [0.5], [1.2]])

        base = make_model().fit(scores_cal)
        shifted = make_model().fit(3.0 * scores_cal + 2.0)

        np.testing.assert_allclose(
            base.predict(scores_test),
            shifted.predict(3.0 * scores_test + 2.0),
            atol=1e-8,
        )

    def test_outputs_are_finite_for_range_and_mild_extrapolation(self):
        scores_cal = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
        scores_test = np.array([[-0.25], [0.0], [0.5], [1.0], [1.25]])

        model = make_model(target=OTTarget.EXP).fit(scores_cal)
        ranks = model.predict(scores_test)

        self.assertTrue(np.all(np.isfinite(ranks)))


if __name__ == "__main__":
    unittest.main()
