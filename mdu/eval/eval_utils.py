import numpy as np
import torch.nn.functional as F
import torch
import pickle
from typing import Any, Callable
from mdu.unc.risk_metrics import RiskType, GName, ApproximationType
from mdu.unc.constants import UncertaintyType
from mdu.data.constants import DatasetName
from typing import Optional
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from mdu.data.data_utils import split_dataset_indices
from mdu.unc.entropic_ot import EntropicOTOrdering
from mdu.unc.additive_baseline import AdditiveUncertaintyOrdering
from mdu.unc.pca_baseline import PCAUncertaintyOrdering
from mdu.unc.constants import ScalingType, OTTarget, SamplingMethod


def load_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_ensemble_predictions(
    ensemble: list[torch.nn.Module],
    input_tensor: torch.Tensor,
    return_logits: bool = True,
) -> np.ndarray:
    """
    Evaluates the ensemble on the input_tensor and returns the softmax probabilities.
    Returns: numpy array of shape (n_models, num_points, n_classes)
    """
    pred_list = []
    for model in ensemble:
        model.eval()
        with torch.no_grad():
            logits: torch.Tensor = model(input_tensor)
            if return_logits:
                pred_list.append(logits.cpu().numpy())
            else:
                probs = F.softmax(logits, dim=1)
                pred_list.append(probs.cpu().numpy())
    pred_stack = np.stack(pred_list, axis=0)  # shape: (n_models, num_points, n_classes)
    return pred_stack


def get_results_path(
    ind_dataset_: DatasetName,
    ood_dataset_: DatasetName,
    uncertainty_type_: UncertaintyType,
    gname_: Optional[GName] = None,
    risk_type_: Optional[RiskType] = None,
    gt_approximation_: Optional[ApproximationType] = None,
    pred_approximation_: Optional[ApproximationType] = None,
    results_root: str = "./resources/results_cleaned",
):
    """Get path to uncertainty measure results file"""
    ind_dataset = ind_dataset_.value.lower()
    ood_dataset = ood_dataset_.value.lower()
    uncertainty_type = uncertainty_type_.value.lower()
    T = 1.0

    if uncertainty_type_ is UncertaintyType.RISK:
        proper_scoring_rule = gname_.value.lower()
        risk_type = risk_type_.value.lower()
        gt_approximation = gt_approximation_.value.lower()
        if pred_approximation_ is not None:
            pred_approximation = pred_approximation_.value.lower()
            folder_path = f"{results_root}/{ind_dataset}/{uncertainty_type}_{proper_scoring_rule}_{risk_type}_{gt_approximation}_{pred_approximation}_T_{T}/{ood_dataset}"
            file_name = f"{ind_dataset}_{ood_dataset}_{uncertainty_type}_{proper_scoring_rule}_{risk_type}_{gt_approximation}_{pred_approximation}_T_{T}.npz"
        else:
            folder_path = f"{results_root}/{ind_dataset}/{uncertainty_type}_{proper_scoring_rule}_{risk_type}_{gt_approximation}_T_{T}/{ood_dataset}"
            file_name = f"{ind_dataset}_{ood_dataset}_{uncertainty_type}_{proper_scoring_rule}_{risk_type}_{gt_approximation}_T_{T}.npz"
    else:
        folder_path = f"{results_root}/{ind_dataset}/{uncertainty_type}/{ood_dataset}"
        file_name = f"{ind_dataset}_{ood_dataset}_{uncertainty_type}.npz"

    return f"{folder_path}/{file_name}"


def load_predictions_and_split(
    ind_dataset_: DatasetName,
    weights_root: str = "./resources/model_weights",
    ensemble_groups: list = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
    ],
    train_ratio: float = 0.0,
    calib_ratio: float = 0.1,
    test_ratio: float = 0.8,
    random_state: int = 42,
):
    """Load predictions from pickle files and split data"""
    ind_dataset = ind_dataset_.value

    results_by_group = {}

    for group_idx, group in enumerate(ensemble_groups):
        # Load logits from ensemble models in this group
        all_ind_logits = []
        for model_id in group:
            if ind_dataset == DatasetName.TINY_IMAGENET.value:
                eval_res = load_pickle(
                    f"{weights_root}/{ind_dataset}/{model_id}/{ind_dataset}.pkl"
                )
            else:
                eval_res = load_pickle(
                    f"{weights_root}/{ind_dataset}/checkpoints/resnet18/CrossEntropy/{model_id}/{ind_dataset}.pkl"
                )
            all_ind_logits.append(eval_res["embeddings"][None])

        y_true = eval_res["labels"]
        logits_ind = eval_res["embeddings"]  # Use last model's logits for splitting

        # Split dataset indices (same split for all groups)
        _, train_cond_idx, calib_idx, test_idx = split_dataset_indices(
            logits_ind,
            y_true,
            train_ratio=train_ratio,
            calib_ratio=calib_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

        # Split labels
        y_train_cond = y_true[train_cond_idx]
        y_calib = y_true[calib_idx]
        y_test = y_true[test_idx]

        # Split features (logits from ensemble)
        ensemble_logits = np.vstack(
            all_ind_logits
        )  # Shape: (n_models_in_group, n_samples, n_classes)
        X_train_cond = ensemble_logits[:, train_cond_idx, :]
        X_calib = ensemble_logits[:, calib_idx, :]
        X_test = ensemble_logits[:, test_idx, :]

        # Compute ensemble predictions for this group
        y_pred = np.argmax(np.mean(X_test, axis=0), axis=-1)
        ensemble_accuracy = np.mean(y_pred == y_test)

        results_by_group[f"group_{group_idx}"] = {
            "group_models": group,
            "X_train_cond": X_train_cond,
            "X_calib": X_calib,
            "X_test": X_test,
            "y_train_cond": y_train_cond,
            "y_calib": y_calib,
            "y_test": y_test,
            "y_pred": y_pred,
            "ensemble_accuracy": ensemble_accuracy,
        }

    return results_by_group


def compute_ood_detection_metrics(ind_scores, ood_scores):
    """Compute OOD detection metrics using ROC AUC"""
    # Combine scores and labels (0 for in-distribution, 1 for OOD)
    all_scores = np.concatenate([ind_scores, ood_scores])
    all_labels = np.concatenate(
        [
            np.zeros_like(ind_scores),  # 0: in-distribution
            np.ones_like(ood_scores),  # 1: OOD
        ]
    )

    # Compute ROC AUC
    roc_auc = roc_auc_score(all_labels, all_scores)

    return {
        "roc_auc": roc_auc,
        "n_ind_samples": len(ind_scores),
        "n_ood_samples": len(ood_scores),
    }


def compute_misclassification_detection_metrics(uncertainty_scores, y_pred, y_true):
    """Compute misclassification detection metrics"""
    # Create binary labels: 0 for correct, 1 for incorrect
    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    correct_scores = uncertainty_scores[correct_mask]
    incorrect_scores = uncertainty_scores[incorrect_mask]

    # Combine scores and labels
    all_scores = np.concatenate([correct_scores, incorrect_scores])
    all_labels = np.concatenate(
        [
            np.zeros_like(correct_scores),  # 0: correct predictions
            np.ones_like(incorrect_scores),  # 1: incorrect predictions
        ]
    )

    # Compute ROC AUC
    roc_auc = roc_auc_score(all_labels, all_scores)

    # Compute Average Precision (AP) for error detection
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)

    return {
        "roc_auc": roc_auc,
        "average_precision": ap,
        "accuracy": np.mean(correct_mask),
        "n_correct": len(correct_scores),
        "n_incorrect": len(incorrect_scores),
    }


def compute_selective_prediction_metrics(uncertainty_scores, y_pred, y_true):
    """Compute selective prediction metrics including AURC"""
    n = len(uncertainty_scores)

    # Sort by uncertainty (low uncertainty first for selective prediction)
    order = np.argsort(uncertainty_scores)
    correct = (y_pred == y_true).astype(int)[order]

    # Compute coverage and accuracy curves
    coverage = np.arange(1, n + 1) / n
    accuracy = np.cumsum(correct) / np.arange(1, n + 1)
    risk = 1 - accuracy  # Risk = 1 - Accuracy

    # Compute AURC (Area Under Risk-Coverage curve) - lower is better
    aurc = np.trapezoid(risk, coverage)

    # Compute AUC for accuracy-coverage curve - higher is better
    acc_cov_auc = np.trapezoid(accuracy, coverage)

    # Compute coverage at different error rates
    coverage_at_error = {}
    for error_rate in [0.01, 0.02, 0.05]:  # 1%, 2%, 5% error rates
        target_accuracy = 1 - error_rate
        # Find first point where accuracy >= target_accuracy
        valid_indices = np.where(accuracy >= target_accuracy)[0]
        if len(valid_indices) > 0:
            coverage_at_error[f"{int(error_rate * 100)}%err"] = coverage[
                valid_indices[0]
            ]
        else:
            coverage_at_error[f"{int(error_rate * 100)}%err"] = 1.0  # Need all data

    overall_accuracy = np.mean(y_pred == y_true)

    return {
        "aurc": aurc,
        "acc_cov_auc": acc_cov_auc,
        "overall_accuracy": overall_accuracy,
        "coverage_at_1pct_error": coverage_at_error.get("1%err", 1.0),
        "coverage_at_2pct_error": coverage_at_error.get("2%err", 1.0),
        "coverage_at_5pct_error": coverage_at_error.get("5%err", 1.0),
        "n_samples": n,
    }


def create_measure_identifier(
    uncertainty_type, gname=None, risk_type=None, gt_approx=None, pred_approx=None
):
    """Create a unique identifier for each uncertainty measure"""
    if uncertainty_type == UncertaintyType.RISK:
        if pred_approx is not None:
            return f"{uncertainty_type.value}_{gname.value}_{risk_type.value}_{gt_approx.value}_{pred_approx.value}"
        else:
            return f"{uncertainty_type.value}_{gname.value}_{risk_type.value}_{gt_approx.value}"
    elif uncertainty_type == UncertaintyType.MAHALANOBIS:
        return uncertainty_type.value
    elif uncertainty_type == UncertaintyType.GMM:
        return uncertainty_type.value
    else:
        return "combination"


def create_output_filename(base_filename, args):
    """Create output filename with EntropicOT hyperparameters"""
    # Extract base name and extension
    if base_filename.endswith(".csv"):
        base_name = base_filename[:-4]
        extension = ".csv"
    else:
        base_name = base_filename
        extension = ""

    # Create hyperparameter string
    entropic_params = []
    entropic_params.append(f"target_{args.entropic_target}")
    entropic_params.append(f"eps_{args.entropic_eps}")
    entropic_params.append(f"scaling_type_{args.entropic_scaling_type}")
    entropic_params.append(f"iters_{args.entropic_max_iters}")
    entropic_params.append(f"tol_{args.entropic_tol}")
    entropic_params.append(f"rs_{args.entropic_random_state}")
    entropic_params.append(f"grid_size_{args.entropic_grid_size}")
    entropic_params.append(f"n_targets_multiplier_{args.entropic_n_targets_multiplier}")

    # Add scaler information
    if args.entropic_scaling_type == "global":
        entropic_params.append("global_scaler")
    elif args.entropic_scaling_type == "mahalanobis":
        entropic_params.append("mahalanobis")
    # feature_wise is default, no need to add to filename

    entropic_str = "_".join(entropic_params)

    return f"{base_name}_entropic_{entropic_str}{extension}"


def config_to_enum_params(config):
    """Convert config dict to enum parameters for get_results_path"""
    if config["type"] == "RISK":
        kwargs = config["kwargs"]
        uncertainty_type = UncertaintyType.RISK
        gname = getattr(GName, kwargs["g_name"])
        risk_type = getattr(RiskType, kwargs["risk_type"])
        gt_approx = getattr(ApproximationType, kwargs["gt_approx"])
        pred_approx = None
        if kwargs.get("pred_approx", None) is not None:
            pred_approx = getattr(ApproximationType, kwargs.get("pred_approx", None))
        return uncertainty_type, gname, risk_type, gt_approx, pred_approx
    elif config["type"] == "MAHALANOBIS":
        return UncertaintyType.MAHALANOBIS, None, None, None, None
    elif config["type"] == "GMM":
        return UncertaintyType.GMM, None, None, None, None
    else:
        raise ValueError(f"Unknown config type: {config['type']}")


def load_uncertainty_data_for_config(config, ind_dataset, ood_dataset, results_root):
    """Load uncertainty data for a single config"""
    uncertainty_type, gname, risk_type, gt_approx, pred_approx = config_to_enum_params(
        config
    )
    path = get_results_path(
        ind_dataset,
        ood_dataset,
        uncertainty_type,
        gname,
        risk_type,
        gt_approx,
        pred_approx,
        results_root,
    )
    return np.load(path)


def build_composition_uncertainty_matrices(uncertainty_datasets, group_idx):
    """Stack component 1D scores into calibration, ID-test, and OOD matrices."""
    uncertainty_matrix_ind = []
    uncertainty_matrix_calib = []
    uncertainty_matrix_ood = []

    for uncertainty_data in uncertainty_datasets:
        uncertainty_matrix_ind.append(uncertainty_data["ind_test"][group_idx, 0, :])
        uncertainty_matrix_calib.append(uncertainty_data["ind_calib"][group_idx, 0, :])
        uncertainty_matrix_ood.append(uncertainty_data["ood"][group_idx, 0, :])

    return (
        np.column_stack(uncertainty_matrix_ind),
        np.column_stack(uncertainty_matrix_calib),
        np.column_stack(uncertainty_matrix_ood),
    )


def _composition_row_metadata():
    return {
        "gname": None,
        "risk_type": None,
        "gt_approximation": None,
        "pred_approximation": None,
    }


def _base_result_row(
    ind_dataset,
    ood_dataset,
    measure,
    uncertainty_type,
    group_idx,
    problem_type,
    group_data,
    metadata,
):
    return {
        "ind_dataset": ind_dataset.value,
        "ood_dataset": ood_dataset.value,
        "measure": measure,
        "uncertainty_type": uncertainty_type,
        "gname": metadata["gname"],
        "risk_type": metadata["risk_type"],
        "gt_approximation": metadata["gt_approximation"],
        "pred_approximation": metadata["pred_approximation"],
        "ensemble_group": group_idx,
        "problem_type": problem_type,
        "ensemble_accuracy": group_data["ensemble_accuracy"],
    }


def _empty_metric_fields():
    return {
        "roc_auc": None,
        "average_precision": None,
        "accuracy": None,
        "aurc": None,
        "acc_cov_auc": None,
        "coverage_at_1pct_error": None,
        "coverage_at_2pct_error": None,
        "coverage_at_5pct_error": None,
        "n_ind_samples": None,
        "n_ood_samples": None,
        "n_correct": None,
        "n_incorrect": None,
    }


def _evaluation_row(
    ind_dataset,
    ood_dataset,
    measure,
    uncertainty_type,
    group_idx,
    problem_type,
    group_data,
    metadata,
    metrics,
):
    row = _base_result_row(
        ind_dataset,
        ood_dataset,
        measure,
        uncertainty_type,
        group_idx,
        problem_type,
        group_data,
        metadata,
    )
    row.update(_empty_metric_fields())
    row.update(metrics)
    return row


def append_uncertainty_evaluation_rows(
    ind_dataset,
    ood_dataset,
    measure,
    uncertainty_type,
    uncertainty_scores_ind,
    uncertainty_scores_ood,
    group_idx,
    group_data,
    results,
    processed_same_dataset,
    metadata=None,
):
    """Append OOD, misclassification, and selective-prediction rows."""
    metadata = metadata or _composition_row_metadata()
    y_pred = group_data["y_pred"]
    y_test = group_data["y_test"]

    if ind_dataset != ood_dataset:
        ood_metrics = compute_ood_detection_metrics(
            uncertainty_scores_ind, uncertainty_scores_ood
        )
        results.append(
            _evaluation_row(
                ind_dataset,
                ood_dataset,
                measure,
                uncertainty_type,
                group_idx,
                "ood_detection",
                group_data,
                metadata,
                {
                    "roc_auc": ood_metrics["roc_auc"],
                    "n_ind_samples": ood_metrics["n_ind_samples"],
                    "n_ood_samples": ood_metrics["n_ood_samples"],
                },
            )
        )

    same_dataset_key = (ind_dataset, measure, group_idx)
    if same_dataset_key in processed_same_dataset:
        return
    processed_same_dataset.add(same_dataset_key)

    misc_metrics = compute_misclassification_detection_metrics(
        uncertainty_scores_ind, y_pred, y_test
    )
    results.append(
        _evaluation_row(
            ind_dataset,
            ind_dataset,
            measure,
            uncertainty_type,
            group_idx,
            "misclassification_detection",
            group_data,
            metadata,
            {
                "roc_auc": misc_metrics["roc_auc"],
                "average_precision": misc_metrics["average_precision"],
                "accuracy": misc_metrics["accuracy"],
                "n_ind_samples": len(uncertainty_scores_ind),
                "n_correct": misc_metrics["n_correct"],
                "n_incorrect": misc_metrics["n_incorrect"],
            },
        )
    )

    sel_metrics = compute_selective_prediction_metrics(
        uncertainty_scores_ind, y_pred, y_test
    )
    results.append(
        _evaluation_row(
            ind_dataset,
            ind_dataset,
            measure,
            uncertainty_type,
            group_idx,
            "selective_prediction",
            group_data,
            metadata,
            {
                "accuracy": sel_metrics["overall_accuracy"],
                "aurc": sel_metrics["aurc"],
                "acc_cov_auc": sel_metrics["acc_cov_auc"],
                "coverage_at_1pct_error": sel_metrics["coverage_at_1pct_error"],
                "coverage_at_2pct_error": sel_metrics["coverage_at_2pct_error"],
                "coverage_at_5pct_error": sel_metrics["coverage_at_5pct_error"],
                "n_ind_samples": sel_metrics["n_samples"],
            },
        )
    )


def process_uncertainty_measure(
    ind_dataset,
    ood_dataset,
    uncertainty_type,
    gname,
    risk_type,
    gt_approx,
    pred_approx,
    prediction_data,
    results,
    args,
    pbar,
    processed_same_dataset,
):
    """Process a single uncertainty measure configuration"""

    try:
        # Load uncertainty measure data
        path = get_results_path(
            ind_dataset,
            ood_dataset,
            uncertainty_type,
            gname,
            risk_type,
            gt_approx,
            pred_approx,
            args.results_root,
        )
        uncertainty_data = np.load(path)

        # Create measure identifier
        measure_id = create_measure_identifier(
            uncertainty_type, gname, risk_type, gt_approx, pred_approx
        )

        # Get prediction data
        pred_data = prediction_data.get(ind_dataset)

        if pred_data is None:
            if args.verbose:
                print(
                    f"Skipping {measure_id} for {ind_dataset.value}->{ood_dataset.value}: No prediction data"
                )
            pbar.update(1)
            return

        # Process each ensemble group
        for group_key, group_data in pred_data.items():
            group_idx = int(group_key.split("_")[1])  # Extract group index

            # Extract uncertainty scores for this group
            ind_test_scores = uncertainty_data["ind_test"][
                group_idx, 0, :
            ]  # Shape: (n_test_samples,)
            ind_calib_scores = uncertainty_data["ind_calib"][
                group_idx, 0, :
            ]  # Shape: (n_calib_samples,)
            ood_scores = uncertainty_data["ood"][
                group_idx, 0, :
            ]  # Shape: (n_ood_samples,)

            # Get predictions and labels
            y_pred = group_data["y_pred"]
            y_test = group_data["y_test"]

            # Determine problem types to evaluate
            if ind_dataset != ood_dataset:
                # Case 1: OOD Detection (different datasets)
                ood_metrics = compute_ood_detection_metrics(ind_test_scores, ood_scores)

                results.append(
                    {
                        "ind_dataset": ind_dataset.value,
                        "ood_dataset": ood_dataset.value,
                        "measure": measure_id,
                        "uncertainty_type": uncertainty_type.value,
                        "gname": gname.value if gname else None,
                        "risk_type": risk_type.value if risk_type else None,
                        "gt_approximation": gt_approx.value if gt_approx else None,
                        "pred_approximation": pred_approx.value
                        if pred_approx
                        else None,
                        "ensemble_group": group_idx,
                        "problem_type": "ood_detection",
                        "roc_auc": ood_metrics["roc_auc"],
                        "average_precision": None,
                        "accuracy": None,
                        "aurc": None,
                        "acc_cov_auc": None,
                        "coverage_at_1pct_error": None,
                        "coverage_at_2pct_error": None,
                        "coverage_at_5pct_error": None,
                        "n_ind_samples": ood_metrics["n_ind_samples"],
                        "n_ood_samples": ood_metrics["n_ood_samples"],
                        "n_correct": None,
                        "n_incorrect": None,
                        "ensemble_accuracy": group_data["ensemble_accuracy"],
                    }
                )

            # Case 2 & 3: Same dataset evaluation - Use ind_test scores for misclassification and selective prediction
            # Only do this once per (ind_dataset, measure_id, group_idx) combination to avoid duplicates across OOD datasets
            same_dataset_key = (ind_dataset, measure_id, group_idx)
            if same_dataset_key not in processed_same_dataset:
                processed_same_dataset.add(same_dataset_key)

                # Misclassification detection using ind_test scores
                misc_metrics = compute_misclassification_detection_metrics(
                    ind_test_scores, y_pred, y_test
                )

                results.append(
                    {
                        "ind_dataset": ind_dataset.value,
                        "ood_dataset": ind_dataset.value,  # Same dataset for both
                        "measure": measure_id,
                        "uncertainty_type": uncertainty_type.value,
                        "gname": gname.value if gname else None,
                        "risk_type": risk_type.value if risk_type else None,
                        "gt_approximation": gt_approx.value if gt_approx else None,
                        "pred_approximation": pred_approx.value
                        if pred_approx
                        else None,
                        "ensemble_group": group_idx,
                        "problem_type": "misclassification_detection",
                        "roc_auc": misc_metrics["roc_auc"],
                        "average_precision": misc_metrics["average_precision"],
                        "accuracy": misc_metrics["accuracy"],
                        "aurc": None,
                        "acc_cov_auc": None,
                        "coverage_at_1pct_error": None,
                        "coverage_at_2pct_error": None,
                        "coverage_at_5pct_error": None,
                        "n_ind_samples": len(ind_test_scores),
                        "n_ood_samples": None,
                        "n_correct": misc_metrics["n_correct"],
                        "n_incorrect": misc_metrics["n_incorrect"],
                        "ensemble_accuracy": group_data["ensemble_accuracy"],
                    }
                )

                # Selective prediction using ind_test scores
                sel_metrics = compute_selective_prediction_metrics(
                    ind_test_scores, y_pred, y_test
                )

                results.append(
                    {
                        "ind_dataset": ind_dataset.value,
                        "ood_dataset": ind_dataset.value,  # Same dataset for both
                        "measure": measure_id,
                        "uncertainty_type": uncertainty_type.value,
                        "gname": gname.value if gname else None,
                        "risk_type": risk_type.value if risk_type else None,
                        "gt_approximation": gt_approx.value if gt_approx else None,
                        "pred_approximation": pred_approx.value
                        if pred_approx
                        else None,
                        "ensemble_group": group_idx,
                        "problem_type": "selective_prediction",
                        "roc_auc": None,
                        "average_precision": None,
                        "accuracy": sel_metrics["overall_accuracy"],
                        "aurc": sel_metrics["aurc"],
                        "acc_cov_auc": sel_metrics["acc_cov_auc"],
                        "coverage_at_1pct_error": sel_metrics["coverage_at_1pct_error"],
                        "coverage_at_2pct_error": sel_metrics["coverage_at_2pct_error"],
                        "coverage_at_5pct_error": sel_metrics["coverage_at_5pct_error"],
                        "n_ind_samples": sel_metrics["n_samples"],
                        "n_ood_samples": None,
                        "n_correct": None,
                        "n_incorrect": None,
                        "ensemble_accuracy": group_data["ensemble_accuracy"],
                    }
                )

        if args.verbose:
            pass
            # print(f"✓ Processed {measure_id} for {ind_dataset.value}->{ood_dataset.value}")

    except Exception as e:
        if args.verbose:
            if ind_dataset.value != ood_dataset.value:
                print(
                    f"✗ Failed to process {measure_id if 'measure_id' in locals() else 'unknown'} for {ind_dataset.value}->{ood_dataset.value}: {e}"
                )

    if pbar is not None:
        pbar.update(1)


def process_multidimensional_composition(
    composition_name,
    configs,
    ind_dataset,
    ood_dataset,
    prediction_data,
    results,
    args,
    processed_same_dataset,
):
    """Process one multidimensional composition using EntropicOTOrdering"""
    _process_composition_with_ordering(
        composition_name=composition_name,
        configs=configs,
        ind_dataset=ind_dataset,
        ood_dataset=ood_dataset,
        prediction_data=prediction_data,
        results=results,
        args=args,
        processed_same_dataset=processed_same_dataset,
        model_factory=lambda: EntropicOTOrdering(
            target=OTTarget(args.entropic_target),
            sampling_method=SamplingMethod(args.entropic_sampling_method),
            scaling_type=ScalingType(args.entropic_scaling_type),
            grid_size=args.entropic_grid_size,
            target_params={},
            eps=args.entropic_eps,
            n_targets_multiplier=args.entropic_n_targets_multiplier,
            max_iters=args.entropic_max_iters,
            random_state=args.entropic_random_state,
            tol=args.entropic_tol,
        ),
        measure_name=composition_name,
        uncertainty_type="EntropicOT",
    )


def process_pca_composition(
    composition_name,
    configs,
    ind_dataset,
    ood_dataset,
    prediction_data,
    results,
    args,
    processed_same_dataset,
):
    """Process one multidimensional composition using PCA aggregation."""
    _process_composition_with_ordering(
        composition_name=composition_name,
        configs=configs,
        ind_dataset=ind_dataset,
        ood_dataset=ood_dataset,
        prediction_data=prediction_data,
        results=results,
        args=args,
        processed_same_dataset=processed_same_dataset,
        model_factory=PCAUncertaintyOrdering,
        measure_name=f"PCA {composition_name}",
        uncertainty_type="PCA",
    )


def process_additive_composition(
    composition_name,
    configs,
    ind_dataset,
    ood_dataset,
    prediction_data,
    results,
    args,
    processed_same_dataset,
):
    """Process one multidimensional composition using raw score summation."""
    _process_composition_with_ordering(
        composition_name=composition_name,
        configs=configs,
        ind_dataset=ind_dataset,
        ood_dataset=ood_dataset,
        prediction_data=prediction_data,
        results=results,
        args=args,
        processed_same_dataset=processed_same_dataset,
        model_factory=AdditiveUncertaintyOrdering,
        measure_name=f"Additive {composition_name}",
        uncertainty_type="Additive",
    )


def _process_composition_with_ordering(
    composition_name,
    configs,
    ind_dataset,
    ood_dataset,
    prediction_data,
    results,
    args,
    processed_same_dataset,
    model_factory: Callable[[], Any],
    measure_name: str,
    uncertainty_type: str,
):
    """Process one multidimensional composition with a fit/predict aggregator."""

    try:
        uncertainty_datasets = []
        for config in configs:
            uncertainty_data = load_uncertainty_data_for_config(
                config, ind_dataset, ood_dataset, args.results_root
            )
            uncertainty_datasets.append(uncertainty_data)

        pred_data = prediction_data.get(ind_dataset)
        if pred_data is None:
            if args.verbose:
                print(
                    f"Skipping composition {composition_name} for {ind_dataset.value}->{ood_dataset.value}: No prediction data"
                )
            return

        for group_key, group_data in pred_data.items():
            group_idx = int(group_key.split("_")[1])
            (
                uncertainty_matrix_ind,
                uncertainty_matrix_calib,
                uncertainty_matrix_ood,
            ) = build_composition_uncertainty_matrices(
                uncertainty_datasets, group_idx
            )

            try:
                model = model_factory()
                model.fit(uncertainty_matrix_calib)
                uncertainty_scores_ind = model.predict(uncertainty_matrix_ind)
                uncertainty_scores_ood = model.predict(uncertainty_matrix_ood)

                append_uncertainty_evaluation_rows(
                    ind_dataset=ind_dataset,
                    ood_dataset=ood_dataset,
                    measure=measure_name,
                    uncertainty_type=uncertainty_type,
                    uncertainty_scores_ind=uncertainty_scores_ind,
                    uncertainty_scores_ood=uncertainty_scores_ood,
                    group_idx=group_idx,
                    group_data=group_data,
                    results=results,
                    processed_same_dataset=processed_same_dataset,
                )

            except Exception as e:
                if args.verbose:
                    print(
                        f"✗ Failed {uncertainty_type} training for {composition_name} group {group_idx}: {e}"
                    )
                continue

        if args.verbose:
            pass

    except Exception as e:
        if args.verbose:
            if ind_dataset.value != ood_dataset.value:
                print(
                    f"✗ Failed to process composition {composition_name} for {ind_dataset.value}->{ood_dataset.value}: {e}"
                )
