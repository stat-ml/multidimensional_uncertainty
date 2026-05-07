import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mdu.eval.paper_tables import (
    DEFAULT_AVERAGE_ROW,
    build_and_write_paper_tables,
    build_article_pareto_table,
    build_paper_tables,
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
    for group_idx in range(2):
        for measure_idx, measure in enumerate(MEASURES):
            value = 0.70 + 0.02 * measure_idx + 0.01 * group_idx
            rows.append(
                _row(
                    group_idx,
                    measure,
                    problem_type="ood_detection",
                    ood_dataset="svhn",
                    roc_auc=value,
                )
            )
            rows.append(
                _row(
                    group_idx,
                    measure,
                    problem_type="misclassification_detection",
                    ood_dataset="cifar10",
                    roc_auc=value + 0.05,
                    average_precision=value + 0.04,
                    accuracy=0.9,
                    n_correct=90,
                    n_incorrect=10,
                )
            )
            rows.append(
                _row(
                    group_idx,
                    measure,
                    problem_type="selective_prediction",
                    ood_dataset="cifar10",
                    accuracy=0.9,
                    aurc=0.1,
                    acc_cov_auc=value + 0.10,
                    coverage_at_1pct_error=0.2,
                    coverage_at_2pct_error=0.4,
                    coverage_at_5pct_error=0.8,
                )
            )
    return pd.DataFrame(rows)


def _row(
    group_idx,
    measure,
    *,
    problem_type,
    ood_dataset,
    roc_auc=np.nan,
    average_precision=np.nan,
    accuracy=np.nan,
    aurc=np.nan,
    acc_cov_auc=np.nan,
    coverage_at_1pct_error=np.nan,
    coverage_at_2pct_error=np.nan,
    coverage_at_5pct_error=np.nan,
    n_correct=np.nan,
    n_incorrect=np.nan,
):
    return {
        "ind_dataset": "cifar10",
        "ood_dataset": ood_dataset,
        "measure": measure,
        "uncertainty_type": "Risk",
        "ensemble_group": group_idx,
        "problem_type": problem_type,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy": accuracy,
        "aurc": aurc,
        "acc_cov_auc": acc_cov_auc,
        "coverage_at_1pct_error": coverage_at_1pct_error,
        "coverage_at_2pct_error": coverage_at_2pct_error,
        "coverage_at_5pct_error": coverage_at_5pct_error,
        "n_ind_samples": 100,
        "n_ood_samples": 50 if problem_type == "ood_detection" else np.nan,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
    }


class PaperTablesTest(unittest.TestCase):
    def test_builds_problem_and_composition_tables(self):
        bundle = build_paper_tables(
            _fake_full_evaluation_df(),
            composition_names=[COMPOSITION],
        )

        self.assertEqual(
            set(bundle.problem_mean_tables),
            {
                "ood_detection",
                "misclassification_detection",
                "selective_prediction",
            },
        )
        self.assertIn(DEFAULT_AVERAGE_ROW, bundle.problem_mean_tables["ood_detection"].index)
        self.assertIn("average_rank", bundle.average_ranks.columns)
        self.assertFalse(bundle.measure_summary.empty)
        self.assertFalse(bundle.pareto_summary.empty)
        self.assertFalse(bundle.article_pareto_table.empty)
        self.assertEqual(
            set(bundle.pareto_summary["aggregation"]),
            {"EntropicOT", "Additive"},
        )
        self.assertIn("Ours", set(bundle.article_pareto_table["method"]))
        self.assertIn("Additive", set(bundle.article_pareto_table["method"]))

        composition_table = bundle.composition_mean_tables[COMPOSITION][
            "ood_detection"
        ]
        self.assertIn(COMPOSITION.lower(), composition_table.columns)
        self.assertIn(f"additive {COMPOSITION}".lower(), composition_table.columns)

        formatted_value = bundle.problem_latex_tables["ood_detection"].iloc[0, 0]
        self.assertTrue(formatted_value.startswith("\\valvar{"))

    def test_writes_csv_and_latex_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "results.csv"
            output_dir = Path(tmp) / "paper_tables"
            _fake_full_evaluation_df().to_csv(input_csv, index=False)

            bundle = build_and_write_paper_tables(
                input_csv,
                output_dir,
                composition_names=[COMPOSITION],
            )

            self.assertFalse(bundle.average_ranks.empty)
            self.assertTrue((output_dir / "all_tasks_mean.csv").exists())
            self.assertTrue((output_dir / "average_ranks.csv").exists())
            self.assertTrue((output_dir / "pareto_summary.csv").exists())
            self.assertTrue((output_dir / "article_pareto_table.csv").exists())
            self.assertTrue(
                (output_dir / "latex" / "article_pareto_table.tex").exists()
            )
            self.assertTrue(
                (output_dir / "problem_tables" / "ood_detection_mean.csv").exists()
            )
            self.assertTrue((output_dir / "latex" / "ood_detection.tex").exists())
            self.assertTrue(
                (
                    output_dir
                    / "composition_tables"
                    / "composite_bayes_all_outer_ood_detection_mean.csv"
                ).exists()
            )

    def test_article_pareto_table_compares_all_methods_together(self):
        index = pd.MultiIndex.from_tuples(
            [
                ("cifar10", "svhn [ood]"),
                ("cifar10", "tiny_imagenet [ood]"),
                ("cifar100", "cifar10 [ood]"),
            ],
            names=["ind_dataset", "eval"],
        )
        transformed = pd.DataFrame(
            {
                "R_b 1 (Logscore)": [0.4, 0.4, 0.4],
                "R_b 1 (Brier)": [0.5, 0.5, 0.5],
                "R_b 1 (Spherical)": [0.6, 0.6, 0.6],
                "R_b 1 (Zero-one)": [0.7, 0.7, 0.7],
                COMPOSITION.lower(): [0.8, 0.8, 0.8],
                f"additive {COMPOSITION}".lower(): [0.3, 0.3, 0.3],
            },
            index=index,
        )

        table = build_article_pareto_table(transformed, COMPOSITION)

        scores = dict(zip(table["method"], table["pareto_percentage"]))
        self.assertEqual(scores["Ours"], 100.0)
        self.assertEqual(scores["Additive"], 0.0)


if __name__ == "__main__":
    unittest.main()
