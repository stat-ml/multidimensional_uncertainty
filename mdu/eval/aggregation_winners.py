from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from configs.interesting_compositions import INTERESTING_COMPOSITIONS
from mdu.eval.aggregation_contrasts import (
    aggregation_columns_for,
    problem_type_from_eval,
)
from mdu.eval.table_analysis_utils import transform_by_tasks

AGGREGATION_ORDER = ("Ours", "PCA", "Additive")
_IMAGE_AGGREGATION_NAMES = {
    "Ours": "EntropicOT",
    "PCA": "PCA",
    "Additive": "Additive",
}
_WINNER_COLUMNS = [
    "source",
    "problem_type",
    "composition",
    "context",
    "aggregation",
    "score",
    "rank",
    "winner_credit",
]
_SUMMARY_COLUMNS = [
    "aggregation",
    "win_rate",
    "n_comparisons",
    "mean_rank",
    "mean_score",
]


def empty_winner_records() -> pd.DataFrame:
    """Return an empty long winner-record table with the public schema."""
    return pd.DataFrame(columns=_WINNER_COLUMNS)


def empty_winner_summary() -> pd.DataFrame:
    """Return an empty winner summary with the public schema."""
    return pd.DataFrame(columns=_SUMMARY_COLUMNS)


def build_image_winner_records_from_csv(
    input_csv: str | Path,
    *,
    selective_metric: str = "acc_cov_auc",
    composition_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load full_evaluation output and compare image aggregations."""
    return build_image_winner_records(
        pd.read_csv(input_csv),
        selective_metric=selective_metric,
        composition_names=composition_names,
    )


def build_image_winner_records(
    df: pd.DataFrame,
    *,
    selective_metric: str = "acc_cov_auc",
    composition_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare Ours/PCA/Additive for every image composition and task row."""
    transformed = transform_by_tasks(df, selective_metric=selective_metric)
    if composition_names is None:
        composition_names = list(INTERESTING_COMPOSITIONS)

    records = []
    for composition_name in composition_names:
        available = aggregation_columns_for(composition_name, transformed.columns)
        columns = {
            display_name: available.get(internal_name)
            for display_name, internal_name in _IMAGE_AGGREGATION_NAMES.items()
        }
        if any(column is None for column in columns.values()):
            continue

        for index, row in transformed.iterrows():
            ind_dataset, eval_label = index
            values = {
                aggregation: row[column]
                for aggregation, column in columns.items()
            }
            records.extend(
                _comparison_records(
                    source="image",
                    problem_type=problem_type_from_eval(eval_label),
                    composition=composition_name,
                    context=f"{ind_dataset} | {eval_label}",
                    values=values,
                )
            )

    if not records:
        return empty_winner_records()
    return pd.DataFrame(records, columns=_WINNER_COLUMNS)


def build_llm_winner_records_from_dir(
    results_dir: str | Path,
    *,
    pattern: str = "*_results.csv",
) -> pd.DataFrame:
    """Compare Ours/PCA/Additive for every LLM result CSV in a directory."""
    paths = sorted(Path(results_dir).glob(pattern))
    return build_llm_winner_records_from_csvs(paths)


def build_llm_winner_records_from_csvs(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Compare Ours/PCA/Additive for every provided LLM result CSV."""
    frames = {
        _model_name_from_path(Path(path)): load_llm_results_csv(path)
        for path in paths
    }
    return build_llm_winner_records(frames)


def load_llm_results_csv(path: str | Path) -> pd.DataFrame:
    """Load LLM result CSVs, keeping metric score columns and dropping ranks."""
    path = Path(path)
    multi_header = pd.read_csv(path, header=[0, 1])
    if _looks_like_score_rank_header(multi_header):
        return normalize_llm_results_dataframe(multi_header)
    return normalize_llm_results_dataframe(pd.read_csv(path))


def build_llm_winner_records(
    dfs_by_model: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare matching LLM aggregation triplets in wide result tables."""
    records = []
    for model_name, df in dfs_by_model.items():
        df = normalize_llm_results_dataframe(df)
        if "Method" not in df.columns:
            continue

        score_cols = llm_score_columns(df)
        if not score_cols:
            continue

        scores = df[["Method", *score_cols]].copy()
        for col in score_cols:
            scores[col] = pd.to_numeric(scores[col], errors="coerce")
        scores = scores.groupby("Method", sort=False)[score_cols].mean()

        for composition in llm_matching_compositions(scores.index):
            methods = {
                "Ours": composition,
                "PCA": f"PCA_{composition}",
                "Additive": f"Additive_{composition}",
            }
            for score_col in score_cols:
                values = {
                    aggregation: scores.loc[method, score_col]
                    for aggregation, method in methods.items()
                }
                records.extend(
                    _comparison_records(
                        source="llm",
                        problem_type="selective_generation",
                        composition=composition,
                        context=f"{model_name} | {score_col}",
                        values=values,
                    )
                )

    if not records:
        return empty_winner_records()
    return pd.DataFrame(records, columns=_WINNER_COLUMNS)


def llm_score_columns(df: pd.DataFrame) -> list[str]:
    """Return real score columns from an LLM wide table, excluding ranks."""
    df = normalize_llm_results_dataframe(df)
    score_cols = []
    for col in df.columns:
        col_name = str(col)
        if col_name == "Method":
            continue
        if col_name == "mean" or col_name == "mean_rank":
            continue
        if col_name.endswith("_rank"):
            continue
        if col_name.startswith("Unnamed:"):
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            score_cols.append(col_name)
    return score_cols


def normalize_llm_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize simple or score/rank LLM tables to Method + metric score columns."""
    if isinstance(df.columns, pd.MultiIndex):
        return _normalize_multiindex_llm_results(df)
    if _looks_like_default_score_rank_read(df):
        return _normalize_default_score_rank_read(df)
    return df.copy()


def llm_matching_compositions(methods: Sequence[str]) -> list[str]:
    """Return unprefixed LLM methods that have PCA_ and Additive_ partners."""
    method_set = set(map(str, methods))
    compositions = []
    for method in sorted(method_set):
        if method.startswith("PCA_") or method.startswith("Additive_"):
            continue
        if f"PCA_{method}" in method_set and f"Additive_{method}" in method_set:
            compositions.append(method)
    return compositions


def _looks_like_score_rank_header(df: pd.DataFrame) -> bool:
    if not isinstance(df.columns, pd.MultiIndex):
        return False
    second_level = {str(value).lower() for value in df.columns.get_level_values(1)}
    first_level = {str(value) for value in df.columns.get_level_values(0)}
    return "score" in second_level and "rank" in second_level and "Method" in first_level


def _normalize_multiindex_llm_results(df: pd.DataFrame) -> pd.DataFrame:
    method_col = None
    for col in df.columns:
        if str(col[0]) == "Method":
            method_col = col
            break
    if method_col is None:
        return pd.DataFrame()

    normalized = pd.DataFrame({"Method": df[method_col]})
    for col in df.columns:
        dataset = str(col[0])
        subcolumn = str(col[1]).lower()
        if dataset == "Method" or dataset.startswith("Unnamed:"):
            continue
        if dataset == "mean":
            continue
        if subcolumn != "score":
            continue
        normalized[dataset] = df[col]
    return normalized


def _looks_like_default_score_rank_read(df: pd.DataFrame) -> bool:
    if df.empty or "Method" not in df.columns:
        return False
    first_row = df.iloc[0].astype(str).str.lower()
    return first_row.isin(["score", "rank"]).any()


def _normalize_default_score_rank_read(df: pd.DataFrame) -> pd.DataFrame:
    data = df.iloc[1:].reset_index(drop=True)
    normalized = pd.DataFrame({"Method": data["Method"]})

    for col in df.columns:
        if col == "Method" or str(col).startswith("Unnamed:"):
            continue
        dataset = _strip_duplicate_suffix(str(col))
        if dataset == "mean":
            continue
        subcolumn = str(df.iloc[0][col]).lower()
        if subcolumn != "score":
            continue
        normalized[dataset] = data[col]
    return normalized


def _strip_duplicate_suffix(column: str) -> str:
    return re.sub(r"\.\d+$", "", column)


def summarize_winners(records: pd.DataFrame) -> pd.DataFrame:
    """Summarize long winner records by aggregation."""
    if records.empty:
        return empty_winner_summary()

    summary = (
        records.groupby("aggregation", as_index=False)
        .agg(
            win_rate=("winner_credit", "mean"),
            n_comparisons=("score", "size"),
            mean_rank=("rank", "mean"),
            mean_score=("score", "mean"),
        )
    )
    order = pd.DataFrame({"aggregation": list(AGGREGATION_ORDER)})
    summary = order.merge(summary, on="aggregation", how="left")
    summary["n_comparisons"] = summary["n_comparisons"].fillna(0).astype(int)
    return summary


def _comparison_records(
    *,
    source: str,
    problem_type: str,
    composition: str,
    context: str,
    values: Mapping[str, object],
) -> list[dict[str, object]]:
    scores = pd.Series(
        {
            aggregation: _to_float(values.get(aggregation))
            for aggregation in AGGREGATION_ORDER
        },
        dtype=float,
    )
    if scores.isna().any():
        return []

    ranks = scores.rank(ascending=False, method="average")
    max_score = scores.max()
    winners = [
        aggregation
        for aggregation, score in scores.items()
        if np.isclose(score, max_score, rtol=1e-12, atol=1e-12)
    ]
    winner_credit = {aggregation: 0.0 for aggregation in AGGREGATION_ORDER}
    for aggregation in winners:
        winner_credit[aggregation] = 1.0 / len(winners)

    return [
        {
            "source": source,
            "problem_type": problem_type,
            "composition": composition,
            "context": context,
            "aggregation": aggregation,
            "score": scores[aggregation],
            "rank": ranks[aggregation],
            "winner_credit": winner_credit[aggregation],
        }
        for aggregation in AGGREGATION_ORDER
    ]


def _to_float(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _model_name_from_path(path: Path) -> str:
    return path.stem.removesuffix("_results")
