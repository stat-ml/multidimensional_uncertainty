import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mdu.eval.aggregation_winners import (
    build_image_winner_records,
    build_llm_winner_records_from_dir,
    load_llm_results_csv,
    summarize_winners,
)


COMPOSITION = "COMPOSITE BAYES ALL OUTER"


class AggregationWinnersTest(unittest.TestCase):
    def test_image_records_cover_tasks_and_split_ties(self):
        records = build_image_winner_records(
            _fake_image_results(),
            composition_names=[COMPOSITION],
        )

        self.assertEqual(
            set(records["problem_type"]),
            {
                "ood_detection",
                "misclassification_detection",
                "selective_prediction",
            },
        )
        self.assertEqual(set(records["aggregation"]), {"Ours", "PCA", "Additive"})

        comparison_credit = records.groupby(
            ["source", "problem_type", "composition", "context"]
        )["winner_credit"].sum()
        self.assertTrue(np.allclose(comparison_credit.to_numpy(), 1.0))
        self.assertTrue((records.groupby("context").size() == 3).all())

        selective = records[
            records["problem_type"].eq("selective_prediction")
            & records["aggregation"].isin(["Ours", "PCA"])
        ]
        self.assertTrue(np.allclose(selective["winner_credit"].to_numpy(), 0.5))
        self.assertTrue(np.allclose(selective["rank"].to_numpy(), 1.5))

        summary = summarize_winners(records).set_index("aggregation")
        self.assertEqual(summary.loc["Ours", "n_comparisons"], 3)
        self.assertAlmostEqual(summary.loc["Ours", "win_rate"], 0.5)
        self.assertAlmostEqual(summary.loc["PCA", "win_rate"], 0.5)
        self.assertAlmostEqual(summary.loc["Additive", "win_rate"], 0.0)

    def test_llm_records_match_triplets_and_ignore_rank_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llama8b_results.csv"
            pd.DataFrame(
                {
                    "Method": [
                        "IMBA_5_Beta_FeatureWise",
                        "PCA_IMBA_5_Beta_FeatureWise",
                        "Additive_IMBA_5_Beta_FeatureWise",
                        "Incomplete",
                        "PCA_Incomplete",
                    ],
                    "trivia": [0.5, 0.6, 0.4, 0.99, 0.01],
                    "trivia_rank": [3, 1, 2, 1, 2],
                    "mmlu": [0.7, 0.7, 0.1, 0.88, 0.02],
                    "mean": [0.6, 0.65, 0.25, 0.935, 0.015],
                    "mean_rank": [2, 1, 3, 1, 2],
                }
            ).to_csv(path, index=False)

            records = build_llm_winner_records_from_dir(tmp)

        self.assertEqual(len(records), 6)
        self.assertEqual(set(records["source"]), {"llm"})
        self.assertEqual(set(records["problem_type"]), {"selective_generation"})
        self.assertEqual(
            set(records["context"]),
            {"llama8b | trivia", "llama8b | mmlu"},
        )
        self.assertFalse(records["context"].str.contains("rank").any())
        self.assertFalse(records["composition"].str.contains("Incomplete").any())

        mmlu = records[records["context"].eq("llama8b | mmlu")]
        credits = dict(zip(mmlu["aggregation"], mmlu["winner_credit"]))
        self.assertEqual(credits["Ours"], 0.5)
        self.assertEqual(credits["PCA"], 0.5)
        self.assertEqual(credits["Additive"], 0.0)

    def test_llm_score_rank_csv_uses_score_columns_only(self):
        csv_text = "\n".join(
            [
                ",Method,trivia,trivia,mmlu,mmlu,mean,mean",
                ",,score,rank,score,rank,score,rank",
                "0,IMBA_5_Beta_FeatureWise,0.5,3,0.7,1,0.6,1",
                "1,PCA_IMBA_5_Beta_FeatureWise,0.6,1,0.7,1,0.65,1",
                "2,Additive_IMBA_5_Beta_FeatureWise,0.4,2,0.1,3,0.25,3",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "falcon7b_results.csv"
            path.write_text(csv_text)

            normalized = load_llm_results_csv(path)
            records = build_llm_winner_records_from_dir(tmp)

        self.assertEqual(normalized.columns.tolist(), ["Method", "trivia", "mmlu"])
        self.assertEqual(len(records), 6)
        self.assertEqual(
            set(records["context"]),
            {"falcon7b | trivia", "falcon7b | mmlu"},
        )
        self.assertFalse(records["context"].str.contains("rank").any())

        trivia = records[records["context"].eq("falcon7b | trivia")]
        credits = dict(zip(trivia["aggregation"], trivia["winner_credit"]))
        self.assertEqual(credits["PCA"], 1.0)
        self.assertEqual(credits["Ours"], 0.0)
        self.assertEqual(credits["Additive"], 0.0)


def _fake_image_results():
    rows = []
    scores = {
        COMPOSITION: {
            "ood_detection": 0.90,
            "misclassification_detection": 0.70,
            "selective_prediction": 0.80,
        },
        f"PCA {COMPOSITION}": {
            "ood_detection": 0.80,
            "misclassification_detection": 0.90,
            "selective_prediction": 0.80,
        },
        f"Additive {COMPOSITION}": {
            "ood_detection": 0.70,
            "misclassification_detection": 0.60,
            "selective_prediction": 0.40,
        },
    }
    for measure, by_problem in scores.items():
        rows.append(
            _row(
                measure,
                problem_type="ood_detection",
                ood_dataset="svhn",
                roc_auc=by_problem["ood_detection"],
            )
        )
        rows.append(
            _row(
                measure,
                problem_type="misclassification_detection",
                ood_dataset="cifar10",
                roc_auc=by_problem["misclassification_detection"],
            )
        )
        rows.append(
            _row(
                measure,
                problem_type="selective_prediction",
                ood_dataset="cifar10",
                acc_cov_auc=by_problem["selective_prediction"],
            )
        )
    return pd.DataFrame(rows)


def _row(
    measure,
    *,
    problem_type,
    ood_dataset,
    roc_auc=np.nan,
    acc_cov_auc=np.nan,
):
    return {
        "ind_dataset": "cifar10",
        "ood_dataset": ood_dataset,
        "measure": measure,
        "uncertainty_type": "EntropicOT",
        "ensemble_group": 0,
        "problem_type": problem_type,
        "roc_auc": roc_auc,
        "average_precision": np.nan,
        "accuracy": np.nan,
        "aurc": np.nan,
        "acc_cov_auc": acc_cov_auc,
        "coverage_at_1pct_error": np.nan,
        "coverage_at_2pct_error": np.nan,
        "coverage_at_5pct_error": np.nan,
        "n_ind_samples": 100,
        "n_ood_samples": 50 if problem_type == "ood_detection" else np.nan,
        "n_correct": np.nan,
        "n_incorrect": np.nan,
    }


if __name__ == "__main__":
    unittest.main()
