from __future__ import annotations

import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# save as scripts/combo_from_npz.py
import itertools
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from mdu.unc.entropic_ot import EntropicOTOrdering
from mdu.unc.constants import OTTarget, SamplingMethod, ScalingType
from tqdm.auto import tqdm


def find_measures(results_root: str | Path, ind: str, ood: str) -> dict[str, Path]:
    base = Path(results_root) / ind
    out = {}
    if not base.exists():
        return out
    for mdir in sorted(p for p in base.iterdir() if p.is_dir()):
        ood_dir = mdir / ood
        if ood_dir.is_dir():
            files = sorted(ood_dir.glob("*.npz"))
            if files:
                out[mdir.name] = files[-1]  # pick one
    return out


def run(n: int, ind: str, ood: str, results_root: str | Path, verbose: bool = False):
    out_csv = f".resources/results/{ind}_{ood}_{n}_combo_results.csv"

    measures = find_measures(results_root, ind, ood)
    if len(measures) < n:
        raise ValueError(
            f"Only {len(measures)} measures found for ({ind},{ood}), need n={n}"
        )

    rows = []
    all_combinations = list(itertools.combinations(sorted(measures.keys()), n))
    np.random.shuffle(all_combinations)
    all_combinations = all_combinations[: min(int(0.15 * len(all_combinations)), 300)]

    for count, combo_names in enumerate(tqdm(all_combinations)):
        # combo_names is a tuple of size n
        individual_measures_results = [np.load(measures[name]) for name in combo_names]

        for group_idx in range(4):  # 4 groups in the dataset
            row = {}
            row["ind_dataset"] = ind
            row["ood_dataset"] = ood
            row["group_id"] = group_idx

            uncertainty_matrix_ind = []
            uncertainty_matrix_calib = []
            uncertainty_matrix_ood = []

            for i, res in enumerate(individual_measures_results):
                ind_scores = res["ind_test"][group_idx][0]
                ood_scores = res["ood"][group_idx][0]
                calib_scores = res["ind_calib"][group_idx][0]

                # Concatenate scores and labels
                all_scores = np.concatenate([ind_scores, ood_scores])
                all_labels = np.concatenate(
                    [
                        np.zeros_like(ind_scores),  # class 0: in-distribution
                        np.ones_like(ood_scores),  # class 1: OOD
                    ]
                )
                auc = roc_auc_score(all_labels, all_scores)
                row[f"{combo_names[i]}"] = auc

                uncertainty_matrix_ind.append(ind_scores)
                uncertainty_matrix_calib.append(calib_scores)
                uncertainty_matrix_ood.append(ood_scores)

            model = EntropicOTOrdering(
                target=OTTarget.EXP,
                sampling_method=SamplingMethod.GRID,
                scaling_type=ScalingType.FEATURE_WISE,
                grid_size=5,
                target_params={},
                eps=0.25,
                n_targets_multiplier=1,
                max_iters=150,
                random_state=42,
                tol=1e-6,
            )

            uncertainty_matrix_ind = np.column_stack(uncertainty_matrix_ind)
            uncertainty_matrix_calib = np.column_stack(uncertainty_matrix_calib)
            uncertainty_matrix_ood = np.column_stack(uncertainty_matrix_ood)

            try:
                model.fit(uncertainty_matrix_calib)

                uncertainty_scores_ind = model.predict(uncertainty_matrix_ind)
                uncertainty_scores_ood = model.predict(uncertainty_matrix_ood)

                all_scores = np.concatenate(
                    [uncertainty_scores_ind, uncertainty_scores_ood]
                )
                all_labels = np.concatenate(
                    [
                        np.zeros_like(
                            uncertainty_scores_ind
                        ),  # class 0: in-distribution
                        np.ones_like(uncertainty_scores_ood),  # class 1: OOD
                    ]
                )
                auc = roc_auc_score(all_labels, all_scores)
            except KeyboardInterrupt:
                df = pd.DataFrame(rows)
                out_csv = Path(out_csv)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
                df.to_csv(out_csv, mode="a", index=False, header=write_header)
                print(f"Interrupted! Saved {len(rows)} rows to {out_csv}")
                exit()
            except Exception as e:
                if verbose:
                    print(
                        f"Failed combo={combo_names} group={group_idx} "
                        f"for {ind}->{ood}: {e}"
                    )
                continue
            row["multidimensional_score"] = auc

            rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
    df.to_csv(out_csv, mode="a", index=False, header=write_header)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--ind_dataset", type=str, required=True)
    p.add_argument("--ood_dataset", type=str, required=True)
    p.add_argument("--results_root", type=str, default=".resources/results")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    df = run(
        args.n,
        args.ind_dataset,
        args.ood_dataset,
        args.results_root,
        verbose=args.verbose,
    )
