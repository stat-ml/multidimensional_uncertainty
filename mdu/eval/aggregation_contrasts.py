from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from configs.interesting_compositions import INTERESTING_COMPOSITIONS
from mdu.eval.paper_tables import (
    PROBLEM_PATTERNS,
    format_mean_std_table,
    select_composition_columns,
    slugify,
)
from mdu.eval.table_analysis_utils import transform_by_tasks

AGGREGATIONS = ("EntropicOT", "PCA", "Additive")


@dataclass
class AggregationContrastReport:
    """Tables describing cases where aggregation methods strongly disagree."""

    cases: pd.DataFrame
    summary_by_winner: pd.DataFrame
    failure_counts: pd.DataFrame
    summary_by_problem: pd.DataFrame
    selected_mean_tables: Dict[str, pd.DataFrame]
    selected_std_tables: Dict[str, pd.DataFrame]
    selected_latex_tables: Dict[str, pd.DataFrame]


def build_aggregation_contrast_report(
    df: pd.DataFrame,
    *,
    selective_metric: str = "acc_cov_auc",
    composition_names: Sequence[str] | None = None,
    min_gap: float = 0.05,
    min_broken: int = 2,
    min_winner_score: float | None = None,
    max_loser_score: float | None = None,
    top_n_per_table: int = 25,
    mean_decimals: int = 3,
    std_decimals: int = 3,
) -> AggregationContrastReport:
    """Find composition/task rows where one aggregation beats the others."""
    mean_table, std_table = transform_by_tasks(
        df,
        selective_metric=selective_metric,
        include_std=True,
    )

    if composition_names is None:
        composition_names = list(INTERESTING_COMPOSITIONS)

    cases = find_aggregation_contrasts(
        mean_table,
        std_table=std_table,
        composition_names=composition_names,
        min_gap=min_gap,
        min_broken=min_broken,
        min_winner_score=min_winner_score,
        max_loser_score=max_loser_score,
    )
    selected_mean_tables, selected_std_tables, selected_latex_tables = (
        build_selected_tables(
            mean_table,
            std_table,
            cases,
            top_n_per_table=top_n_per_table,
            mean_decimals=mean_decimals,
            std_decimals=std_decimals,
        )
    )

    return AggregationContrastReport(
        cases=cases,
        summary_by_winner=summarize_by_winner(cases),
        failure_counts=count_failures(cases),
        summary_by_problem=summarize_by_problem(cases),
        selected_mean_tables=selected_mean_tables,
        selected_std_tables=selected_std_tables,
        selected_latex_tables=selected_latex_tables,
    )


def build_and_write_aggregation_contrast_report(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    selective_metric: str = "acc_cov_auc",
    composition_names: Sequence[str] | None = None,
    min_gap: float = 0.05,
    min_broken: int = 2,
    min_winner_score: float | None = None,
    max_loser_score: float | None = None,
    top_n_per_table: int = 25,
    write_latex: bool = True,
    mean_decimals: int = 3,
    std_decimals: int = 3,
) -> AggregationContrastReport:
    """Load full_evaluation output, find contrast cases, and write report files."""
    df = pd.read_csv(input_csv)
    report = build_aggregation_contrast_report(
        df,
        selective_metric=selective_metric,
        composition_names=composition_names,
        min_gap=min_gap,
        min_broken=min_broken,
        min_winner_score=min_winner_score,
        max_loser_score=max_loser_score,
        top_n_per_table=top_n_per_table,
        mean_decimals=mean_decimals,
        std_decimals=std_decimals,
    )
    write_aggregation_contrast_report(
        report,
        output_dir,
        write_latex=write_latex,
        heuristic={
            "min_gap": min_gap,
            "min_broken": min_broken,
            "min_winner_score": min_winner_score,
            "max_loser_score": max_loser_score,
            "selective_metric": selective_metric,
        },
    )
    return report


def find_aggregation_contrasts(
    mean_table: pd.DataFrame,
    *,
    std_table: pd.DataFrame | None = None,
    composition_names: Sequence[str],
    min_gap: float = 0.05,
    min_broken: int = 2,
    min_winner_score: float | None = None,
    max_loser_score: float | None = None,
) -> pd.DataFrame:
    """Return rows where one aggregation is separated from its competitors."""
    records = []

    for composition_name in composition_names:
        agg_columns = aggregation_columns_for(composition_name, mean_table.columns)
        if len(agg_columns) < 2:
            continue

        for index, row in mean_table.iterrows():
            values = {
                aggregation: row[column]
                for aggregation, column in agg_columns.items()
                if column in row and pd.notna(row[column])
            }
            if len(values) < 2:
                continue

            ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
            winner, winner_score = ordered[0]
            second_best, second_best_score = ordered[1]
            worst, worst_score = ordered[-1]

            if min_winner_score is not None and winner_score < min_winner_score:
                continue

            broken = []
            for aggregation, score in values.items():
                if aggregation == winner:
                    continue
                is_far_behind = (winner_score - score) >= min_gap
                is_low_enough = max_loser_score is None or score <= max_loser_score
                if is_far_behind and is_low_enough:
                    broken.append(aggregation)

            if len(broken) < min_broken:
                continue

            ind_dataset, eval_label = index
            record = {
                "composition": composition_name,
                "problem_type": problem_type_from_eval(eval_label),
                "ind_dataset": ind_dataset,
                "eval": eval_label,
                "winner": winner,
                "winner_score": winner_score,
                "second_best": second_best,
                "second_best_score": second_best_score,
                "worst": worst,
                "worst_score": worst_score,
                "gap_to_second": winner_score - second_best_score,
                "gap_to_worst": winner_score - worst_score,
                "broken_aggregations": ", ".join(sorted(broken)),
                "n_broken": len(broken),
            }

            for aggregation in AGGREGATIONS:
                score = values.get(aggregation)
                record[f"{aggregation.lower()}_score"] = score
                if score is not None:
                    record[f"{aggregation.lower()}_gap_from_winner"] = (
                        winner_score - score
                    )
                record[f"{aggregation.lower()}_std"] = lookup_std(
                    std_table,
                    index,
                    agg_columns.get(aggregation),
                )

            record["gap_to_second_over_std"] = gap_over_combined_std(
                record["gap_to_second"],
                record.get(f"{winner.lower()}_std"),
                record.get(f"{second_best.lower()}_std"),
            )
            records.append(record)

    if not records:
        return empty_cases_frame()

    return pd.DataFrame(records).sort_values(
        ["gap_to_second", "gap_to_worst"],
        ascending=[False, False],
    )


def aggregation_columns_for(
    composition_name: str,
    available_columns,
) -> Dict[str, str]:
    """Map aggregation labels to transformed-table columns for one composition."""
    candidate_columns = {
        "EntropicOT": composition_name.lower(),
        "PCA": f"pca {composition_name}".lower(),
        "Additive": f"additive {composition_name}".lower(),
    }
    return {
        aggregation: column
        for aggregation, column in candidate_columns.items()
        if column in available_columns
    }


def problem_type_from_eval(eval_label: str) -> str:
    for problem_type, marker in PROBLEM_PATTERNS.items():
        if marker in str(eval_label):
            return problem_type
    return "unknown"


def lookup_std(
    std_table: pd.DataFrame | None,
    index,
    column: str | None,
) -> float:
    if std_table is None or column is None:
        return np.nan
    if column not in std_table.columns or index not in std_table.index:
        return np.nan
    return std_table.loc[index, column]


def gap_over_combined_std(gap: float, std_a: float, std_b: float) -> float:
    if pd.isna(std_a) or pd.isna(std_b):
        return np.nan
    denom = np.sqrt(std_a**2 + std_b**2)
    if denom == 0:
        return np.inf if gap > 0 else 0.0
    return gap / denom


def summarize_by_winner(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(
            columns=[
                "winner",
                "problem_type",
                "n_cases",
                "mean_gap_to_second",
                "mean_gap_to_worst",
            ]
        )

    return (
        cases.groupby(["winner", "problem_type"], as_index=False)
        .agg(
            n_cases=("winner", "size"),
            mean_gap_to_second=("gap_to_second", "mean"),
            mean_gap_to_worst=("gap_to_worst", "mean"),
            median_gap_to_second=("gap_to_second", "median"),
        )
        .sort_values(["n_cases", "mean_gap_to_second"], ascending=[False, False])
    )


def count_failures(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(
            columns=[
                "broken_aggregation",
                "winner",
                "problem_type",
                "n_cases",
                "mean_gap_from_winner",
            ]
        )

    rows = []
    for _, case in cases.iterrows():
        for aggregation in AGGREGATIONS:
            if aggregation == case["winner"]:
                continue
            gap = case.get(f"{aggregation.lower()}_gap_from_winner")
            if pd.notna(gap) and gap > 0:
                rows.append(
                    {
                        "broken_aggregation": aggregation,
                        "winner": case["winner"],
                        "problem_type": case["problem_type"],
                        "gap_from_winner": gap,
                        "composition": case["composition"],
                    }
                )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .groupby(["broken_aggregation", "winner", "problem_type"], as_index=False)
        .agg(
            n_cases=("broken_aggregation", "size"),
            mean_gap_from_winner=("gap_from_winner", "mean"),
            n_compositions=("composition", "nunique"),
        )
        .sort_values(["n_cases", "mean_gap_from_winner"], ascending=[False, False])
    )


def summarize_by_problem(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(columns=["problem_type", "winner", "n_cases"])
    return (
        cases.pivot_table(
            index="problem_type",
            columns="winner",
            values="composition",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def build_selected_tables(
    mean_table: pd.DataFrame,
    std_table: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    top_n_per_table: int = 25,
    mean_decimals: int = 3,
    std_decimals: int = 3,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Build small per-composition tables for the selected contrast cases."""
    selected_mean_tables: Dict[str, pd.DataFrame] = {}
    selected_std_tables: Dict[str, pd.DataFrame] = {}
    selected_latex_tables: Dict[str, pd.DataFrame] = {}

    if cases.empty:
        return selected_mean_tables, selected_std_tables, selected_latex_tables

    for (composition_name, problem_type), group in cases.groupby(
        ["composition", "problem_type"],
        sort=False,
    ):
        group = group.head(top_n_per_table)
        base_mean = select_composition_columns(mean_table, composition_name)
        if base_mean.empty:
            continue

        index = pd.MultiIndex.from_frame(group[["ind_dataset", "eval"]])
        mean_subset = base_mean.reindex(index=index)
        std_subset = std_table.reindex(
            index=mean_subset.index,
            columns=mean_subset.columns,
        )

        key = f"{slugify(composition_name)}_{problem_type}"
        selected_mean_tables[key] = mean_subset
        selected_std_tables[key] = std_subset
        selected_latex_tables[key] = format_mean_std_table(
            mean_subset,
            std_subset,
            mean_decimals=mean_decimals,
            std_decimals=std_decimals,
        )

    return selected_mean_tables, selected_std_tables, selected_latex_tables


def write_aggregation_contrast_report(
    report: AggregationContrastReport,
    output_dir: str | Path,
    *,
    write_latex: bool = True,
    heuristic: Mapping[str, object] | None = None,
) -> None:
    """Write contrast cases, summaries, and selected mini-tables to disk."""
    output_path = Path(output_dir)
    selected_dir = output_path / "selected_tables"
    latex_dir = output_path / "selected_latex"

    output_path.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    report.cases.to_csv(output_path / "contrast_cases.csv", index=False)
    report.summary_by_winner.to_csv(output_path / "summary_by_winner.csv", index=False)
    report.failure_counts.to_csv(output_path / "failure_counts.csv", index=False)
    report.summary_by_problem.to_csv(output_path / "summary_by_problem.csv", index=False)

    for key, table in report.selected_mean_tables.items():
        table.to_csv(selected_dir / f"{key}_mean.csv")
        report.selected_std_tables[key].to_csv(selected_dir / f"{key}_std.csv")

    if write_latex:
        latex_dir.mkdir(parents=True, exist_ok=True)
        for key, table in report.selected_latex_tables.items():
            (latex_dir / f"{key}.tex").write_text(
                table.to_latex(escape=False),
                encoding="utf-8",
            )

    write_markdown_summary(report, output_path / "README.md", heuristic=heuristic)


def write_markdown_summary(
    report: AggregationContrastReport,
    path: Path,
    *,
    heuristic: Mapping[str, object] | None = None,
) -> None:
    lines = [
        "# Aggregation Contrast Report",
        "",
        "This folder contains cases where one aggregation method is clearly ahead of the others.",
        "",
    ]
    if heuristic:
        lines.append("## Heuristic")
        lines.append("")
        for key, value in heuristic.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    lines.extend(
        [
            "## Files",
            "",
            "- `contrast_cases.csv`: one row per selected composition/task case.",
            "- `summary_by_winner.csv`: how often each aggregation is the clear winner.",
            "- `failure_counts.csv`: how often each aggregation trails the winner.",
            "- `summary_by_problem.csv`: selected-case counts by problem type and winner.",
            "- `selected_tables/`: small CSV tables for manual inspection.",
            "- `selected_latex/`: LaTeX versions of the same selected tables.",
            "",
            f"Selected cases: {len(report.cases)}",
            "",
        ]
    )

    if not report.summary_by_winner.empty:
        lines.append("## Top Winner Patterns")
        lines.append("")
        lines.append("```text")
        lines.append(report.summary_by_winner.head(10).to_string(index=False))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def empty_cases_frame() -> pd.DataFrame:
    columns = [
        "composition",
        "problem_type",
        "ind_dataset",
        "eval",
        "winner",
        "winner_score",
        "second_best",
        "second_best_score",
        "worst",
        "worst_score",
        "gap_to_second",
        "gap_to_worst",
        "broken_aggregations",
        "n_broken",
        "gap_to_second_over_std",
    ]
    for aggregation in AGGREGATIONS:
        prefix = aggregation.lower()
        columns.extend(
            [
                f"{prefix}_score",
                f"{prefix}_gap_from_winner",
                f"{prefix}_std",
            ]
        )
    return pd.DataFrame(columns=columns)
