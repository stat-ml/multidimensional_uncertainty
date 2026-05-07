import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mdu.data.constants import DatasetName
from mdu.eval.eval_utils import (
    get_results_path,
    process_additive_composition,
    process_multidimensional_composition,
)
from mdu.unc.constants import UncertaintyType


COMPOSITION_CONFIGS = [
    {"type": "MAHALANOBIS", "kwargs": {}},
    {"type": "GMM", "kwargs": {}},
]


def make_args(results_root):
    return SimpleNamespace(
        results_root=str(results_root),
        verbose=False,
        entropic_target="Beta",
        entropic_sampling_method="Grid",
        entropic_scaling_type="FeatureWise",
        entropic_grid_size=0,
        entropic_eps=0.5,
        entropic_n_targets_multiplier=1,
        entropic_max_iters=200,
        entropic_random_state=0,
        entropic_tol=1e-8,
    )


def make_prediction_data():
    return {
        DatasetName.CIFAR10: {
            "group_0": {
                "y_pred": np.array([0, 0, 1, 1]),
                "y_test": np.array([0, 1, 1, 1]),
                "ensemble_accuracy": 0.75,
            }
        }
    }


def write_uncertainty_file(results_root, ind_dataset, ood_dataset, uncertainty_type):
    offset = 0.0 if uncertainty_type == UncertaintyType.MAHALANOBIS else 0.25
    ind_test = np.array([[[0.1, 0.3, 0.7, 0.9]]]) + offset
    ind_calib = np.array([[[0.0, 0.2, 0.6, 1.0]]]) + offset
    ood = np.array([[[0.4, 0.8, 1.1, 1.4]]]) + offset

    path = Path(
        get_results_path(
            ind_dataset,
            ood_dataset,
            uncertainty_type,
            results_root=str(results_root),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ind_test=ind_test, ind_calib=ind_calib, ood=ood)


def write_composition_files(results_root, ind_dataset, ood_dataset):
    write_uncertainty_file(
        results_root, ind_dataset, ood_dataset, UncertaintyType.MAHALANOBIS
    )
    write_uncertainty_file(results_root, ind_dataset, ood_dataset, UncertaintyType.GMM)


class CompositionEvaluatorTest(unittest.TestCase):
    def test_entropic_and_additive_rows_share_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp)
            write_composition_files(
                results_root, DatasetName.CIFAR10, DatasetName.CIFAR100
            )
            args = make_args(results_root)
            prediction_data = make_prediction_data()

            entropic_results = []
            additive_results = []
            process_multidimensional_composition(
                "TEST COMPOSITION",
                COMPOSITION_CONFIGS,
                DatasetName.CIFAR10,
                DatasetName.CIFAR100,
                prediction_data,
                entropic_results,
                args,
                set(),
            )
            process_additive_composition(
                "TEST COMPOSITION",
                COMPOSITION_CONFIGS,
                DatasetName.CIFAR10,
                DatasetName.CIFAR100,
                prediction_data,
                additive_results,
                args,
                set(),
            )

            self.assertEqual(len(entropic_results), 3)
            self.assertEqual(len(additive_results), 3)
            self.assertEqual(set(entropic_results[0]), set(additive_results[0]))
            self.assertTrue(
                all(
                    row["uncertainty_type"] == "Additive"
                    for row in additive_results
                )
            )
            self.assertTrue(
                all(
                    row["measure"] == "Additive TEST COMPOSITION"
                    for row in additive_results
                )
            )

    def test_same_dataset_rows_are_not_duplicated_across_ood_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_root = Path(tmp)
            write_composition_files(
                results_root, DatasetName.CIFAR10, DatasetName.CIFAR100
            )
            write_composition_files(results_root, DatasetName.CIFAR10, DatasetName.SVHN)
            args = make_args(results_root)
            prediction_data = make_prediction_data()
            processed_same_dataset = set()
            results = []

            for ood_dataset in [DatasetName.CIFAR100, DatasetName.SVHN]:
                process_additive_composition(
                    "TEST COMPOSITION",
                    COMPOSITION_CONFIGS,
                    DatasetName.CIFAR10,
                    ood_dataset,
                    prediction_data,
                    results,
                    args,
                    processed_same_dataset,
                )

            problem_types = [row["problem_type"] for row in results]
            self.assertEqual(problem_types.count("ood_detection"), 2)
            self.assertEqual(problem_types.count("misclassification_detection"), 1)
            self.assertEqual(problem_types.count("selective_prediction"), 1)
            self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
