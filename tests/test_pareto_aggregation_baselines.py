import unittest

import pandas as pd

from mdu.eval.table_analysis_utils import analyze_composite_pareto_performance


COMPOSITION = "COMPOSITE BAYES ALL OUTER"


class ParetoAggregationBaselinesTest(unittest.TestCase):
    def test_pareto_analysis_includes_additive(self):
        transformed = _transformed_composition_table(include_baselines=True)

        results = analyze_composite_pareto_performance(
            transformed,
            {COMPOSITION: object()},
        )

        self.assertIn(COMPOSITION, results)
        self.assertIn(f"Additive {COMPOSITION}", results)
        self.assertEqual(results[f"Additive {COMPOSITION}"]["total_problems"], 3)

    def test_pareto_baselines_can_be_disabled_for_old_behavior(self):
        transformed = _transformed_composition_table(include_baselines=True)

        results = analyze_composite_pareto_performance(
            transformed,
            {COMPOSITION: object()},
            include_baseline_aggregations=False,
        )

        self.assertEqual(set(results), {COMPOSITION})


def _transformed_composition_table(include_baselines):
    index = pd.MultiIndex.from_tuples(
        [
            ("cifar10", "cifar100 [ood]"),
            ("cifar10", "svhn [ood]"),
            ("cifar100", "cifar10 [ood]"),
        ],
        names=["ind_dataset", "eval"],
    )
    data = {
        "R_b 1 (Logscore)": [0.40, 0.40, 0.40],
        "R_b 1 (Brier)": [0.41, 0.41, 0.41],
        "R_b 1 (Spherical)": [0.42, 0.42, 0.42],
        "R_b 1 (Zero-one)": [0.43, 0.43, 0.43],
        COMPOSITION.lower(): [0.50, 0.50, 0.50],
    }
    if include_baselines:
        data[f"additive {COMPOSITION}".lower()] = [0.30, 0.30, 0.30]
    return pd.DataFrame(data, index=index)


if __name__ == "__main__":
    unittest.main()
