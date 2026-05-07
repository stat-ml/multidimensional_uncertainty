import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mdu.eval.aggregation_contrasts import (
    build_aggregation_contrast_report,
    build_and_write_aggregation_contrast_report,
)


COMPOSITION = "COMPOSITE BAYES ALL OUTER"
MEASURES = [
    "Risk_LogScore_BayesRisk_outer",
    "Risk_BrierScore_BayesRisk_outer",
    "Risk_SphericalScore_BayesRisk_outer",
    "Risk_ZeroOneScore_BayesRisk_outer",
    COMPOSITION,
    f"Additive {COMPOSITION}",
]


def _fake_full_evaluation_df():
    rows = []
    scores = {
        "Risk_LogScore_BayesRisk_outer": 0.74,
        "Risk_BrierScore_BayesRisk_outer": 0.75,
        "Risk_SphericalScore_BayesRisk_outer": 0.76,
        "Risk_ZeroOneScore_BayesRisk_outer": 0.77,
        COMPOSITION: 0.93,
        f"Additive {COMPOSITION}": 0.72,
    }
    for group_idx in range(2):
        for measure, score in scores.items():
            rows.append(
                {
                    "ind_dataset": "cifar10",
                    "ood_dataset": "svhn",
                    "measure": measure,
                    "uncertainty_type": "EntropicOT",
                    "ensemble_group": group_idx,
                    "problem_type": "ood_detection",
                    "roc_auc": score + 0.01 * group_idx,
                    "average_precision": np.nan,
                    "accuracy": np.nan,
                    "aurc": np.nan,
                    "acc_cov_auc": np.nan,
                    "coverage_at_1pct_error": np.nan,
                    "coverage_at_2pct_error": np.nan,
                    "coverage_at_5pct_error": np.nan,
                    "n_ind_samples": 100,
                    "n_ood_samples": 50,
                    "n_correct": np.nan,
                    "n_incorrect": np.nan,
                }
            )
    return pd.DataFrame(rows)


class AggregationContrastsTest(unittest.TestCase):
    def test_finds_clear_aggregation_contrast(self):
        report = build_aggregation_contrast_report(
            _fake_full_evaluation_df(),
            composition_names=[COMPOSITION],
            min_gap=0.05,
            min_broken=1,
        )

        self.assertEqual(len(report.cases), 1)
        case = report.cases.iloc[0]
        self.assertEqual(case["winner"], "EntropicOT")
        self.assertEqual(case["problem_type"], "ood_detection")
        self.assertIn("Additive", case["broken_aggregations"])
        self.assertFalse(report.summary_by_winner.empty)
        self.assertFalse(report.failure_counts.empty)
        self.assertIn(
            "composite_bayes_all_outer_ood_detection",
            report.selected_mean_tables,
        )

    def test_writes_contrast_report_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "results.csv"
            output_dir = Path(tmp) / "aggregation_contrasts"
            _fake_full_evaluation_df().to_csv(input_csv, index=False)

            report = build_and_write_aggregation_contrast_report(
                input_csv,
                output_dir,
                composition_names=[COMPOSITION],
                min_gap=0.05,
                min_broken=1,
            )

            self.assertEqual(len(report.cases), 1)
            self.assertTrue((output_dir / "contrast_cases.csv").exists())
            self.assertTrue((output_dir / "summary_by_winner.csv").exists())
            self.assertTrue((output_dir / "failure_counts.csv").exists())
            self.assertTrue((output_dir / "README.md").exists())
            self.assertTrue(
                (
                    output_dir
                    / "selected_tables"
                    / "composite_bayes_all_outer_ood_detection_mean.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "selected_latex"
                    / "composite_bayes_all_outer_ood_detection.tex"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
