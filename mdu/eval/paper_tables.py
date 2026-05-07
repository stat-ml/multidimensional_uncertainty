from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
import re
from typing import Dict, Mapping, Sequence

import pandas as pd

from configs.interesting_compositions import INTERESTING_COMPOSITIONS
from mdu.eval.table_analysis_utils import (
    analyze_composite_pareto_performance,
    compute_average_ranks,
    pareto_depth,
    pareto_front,
    select_composite_and_components,
    transform_by_tasks,
)

PROBLEM_PATTERNS: Mapping[str, str] = {
    "ood_detection": "[ood]",
    "misclassification_detection": "[miscls]",
    "selective_prediction": "[selective]",
}

DEFAULT_AVERAGE_ROW = ("AVG", "[all rows]")
DEFAULT_ARTICLE_PARETO_COMPOSITION = "COMPOSITE EAT LOGSCORE OUTER OUTER + M"


@dataclass
class PaperTableBundle:
    """Container for all postprocessed article tables."""

    all_tasks_mean: pd.DataFrame
    all_tasks_std: pd.DataFrame
    problem_mean_tables: Dict[str, pd.DataFrame]
    problem_std_tables: Dict[str, pd.DataFrame]
    problem_latex_tables: Dict[str, pd.DataFrame]
    average_ranks: pd.DataFrame
    measure_summary: pd.DataFrame
    pareto_summary: pd.DataFrame
    article_pareto_table: pd.DataFrame
    article_pareto_latex_table: pd.DataFrame
    composition_mean_tables: Dict[str, Dict[str, pd.DataFrame]]
    composition_std_tables: Dict[str, Dict[str, pd.DataFrame]]
    composition_latex_tables: Dict[str, Dict[str, pd.DataFrame]]


def build_paper_tables(
    df: pd.DataFrame,
    *,
    selective_metric: str = "acc_cov_auc",
    composition_names: Sequence[str] | None = None,
    article_pareto_composition: str | None = DEFAULT_ARTICLE_PARETO_COMPOSITION,
    include_composition_tables: bool = True,
    add_average_row: bool = True,
    mean_decimals: int = 3,
    std_decimals: int = 3,
) -> PaperTableBundle:
    """Build the tables that used to be assembled manually in notebooks."""
    mean_table, std_table = transform_by_tasks(
        df,
        selective_metric=selective_metric,
        include_std=True,
    )

    problem_mean_tables = split_problem_tables(
        mean_table,
        add_average_row=add_average_row,
    )
    problem_std_tables = split_problem_tables(
        std_table,
        add_average_row=add_average_row,
    )
    problem_latex_tables = {
        problem_type: format_mean_std_table(
            problem_mean_tables[problem_type],
            problem_std_tables[problem_type],
            mean_decimals=mean_decimals,
            std_decimals=std_decimals,
        )
        for problem_type in PROBLEM_PATTERNS
    }

    average_ranks = (
        compute_average_ranks(mean_table)
        .rename_axis("measure")
        .reset_index(name="average_rank")
    )
    average_ranks.insert(0, "rank_position", range(1, len(average_ranks) + 1))

    measure_summary = summarize_problem_tables(problem_mean_tables)
    if composition_names is None:
        composition_names = list(INTERESTING_COMPOSITIONS)
    else:
        composition_names = list(composition_names)
    pareto_summary = build_pareto_summary(mean_table, composition_names)
    article_pareto_composition = resolve_article_pareto_composition(
        article_pareto_composition,
        composition_names,
    )
    if article_pareto_composition is None:
        article_pareto_table = empty_article_pareto_table()
        article_pareto_latex_table = empty_article_pareto_latex_table()
    else:
        article_pareto_table = build_article_pareto_table(
            mean_table,
            article_pareto_composition,
        )
        article_pareto_latex_table = format_article_pareto_table(
            article_pareto_table,
            decimals=mean_decimals,
        )

    composition_mean_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    composition_std_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    composition_latex_tables: Dict[str, Dict[str, pd.DataFrame]] = {}

    if include_composition_tables:
        for composition_name in composition_names:
            mean_subset = select_composition_columns(mean_table, composition_name)
            if mean_subset.empty:
                continue

            std_subset = std_table.reindex(
                index=mean_subset.index,
                columns=mean_subset.columns,
            )
            composition_mean_tables[composition_name] = split_problem_tables(
                mean_subset,
                add_average_row=add_average_row,
            )
            composition_std_tables[composition_name] = split_problem_tables(
                std_subset,
                add_average_row=add_average_row,
            )
            composition_latex_tables[composition_name] = {
                problem_type: format_mean_std_table(
                    composition_mean_tables[composition_name][problem_type],
                    composition_std_tables[composition_name][problem_type],
                    mean_decimals=mean_decimals,
                    std_decimals=std_decimals,
                )
                for problem_type in PROBLEM_PATTERNS
            }

    return PaperTableBundle(
        all_tasks_mean=mean_table,
        all_tasks_std=std_table,
        problem_mean_tables=problem_mean_tables,
        problem_std_tables=problem_std_tables,
        problem_latex_tables=problem_latex_tables,
        average_ranks=average_ranks,
        measure_summary=measure_summary,
        pareto_summary=pareto_summary,
        article_pareto_table=article_pareto_table,
        article_pareto_latex_table=article_pareto_latex_table,
        composition_mean_tables=composition_mean_tables,
        composition_std_tables=composition_std_tables,
        composition_latex_tables=composition_latex_tables,
    )


def build_and_write_paper_tables(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    selective_metric: str = "acc_cov_auc",
    composition_names: Sequence[str] | None = None,
    article_pareto_composition: str | None = DEFAULT_ARTICLE_PARETO_COMPOSITION,
    include_composition_tables: bool = True,
    write_latex: bool = True,
    mean_decimals: int = 3,
    std_decimals: int = 3,
) -> PaperTableBundle:
    """Load a full_evaluation CSV, build article tables, and write them to disk."""
    df = pd.read_csv(input_csv)
    bundle = build_paper_tables(
        df,
        selective_metric=selective_metric,
        composition_names=composition_names,
        article_pareto_composition=article_pareto_composition,
        include_composition_tables=include_composition_tables,
        mean_decimals=mean_decimals,
        std_decimals=std_decimals,
    )
    write_paper_tables(bundle, output_dir, write_latex=write_latex)
    return bundle


def split_problem_tables(
    table: pd.DataFrame,
    *,
    add_average_row: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Split a transformed table into OOD, misclassification, and selective tables."""
    if table.empty:
        return {problem_type: pd.DataFrame() for problem_type in PROBLEM_PATTERNS}

    eval_index = table.index.get_level_values("eval").astype(str)
    problem_tables = {}
    for problem_type, pattern in PROBLEM_PATTERNS.items():
        problem_table = table[eval_index.str.contains(pattern, regex=False)].copy()
        if add_average_row:
            problem_table = with_average_row(problem_table)
        problem_tables[problem_type] = problem_table
    return problem_tables


def with_average_row(
    table: pd.DataFrame,
    *,
    label: tuple[str, str] = DEFAULT_AVERAGE_ROW,
) -> pd.DataFrame:
    """Append a mean-over-rows line while preserving the table MultiIndex."""
    if table.empty:
        return table
    average = table.mean(axis=0, numeric_only=True)
    average_row = pd.DataFrame(
        [average],
        index=pd.MultiIndex.from_tuples([label], names=table.index.names),
    )
    return pd.concat([table, average_row])


def format_mean_std_table(
    mean_table: pd.DataFrame,
    std_table: pd.DataFrame,
    *,
    mean_decimals: int = 3,
    std_decimals: int = 3,
    leading_dot_std: bool = True,
    na: str = "--",
) -> pd.DataFrame:
    """Return a LaTeX-ready table with cells like \\valvar{0.931}{.004}."""
    std_table = std_table.reindex(index=mean_table.index, columns=mean_table.columns)
    formatted = pd.DataFrame(index=mean_table.index, columns=mean_table.columns)

    for row_idx in range(len(mean_table.index)):
        for col_idx in range(len(mean_table.columns)):
            mean_value = mean_table.iat[row_idx, col_idx]
            std_value = std_table.iat[row_idx, col_idx]
            if pd.isna(mean_value) or pd.isna(std_value):
                formatted.iat[row_idx, col_idx] = na
                continue

            mean_str = f"{mean_value:.{mean_decimals}f}"
            std_str = f"{std_value:.{std_decimals}f}"
            if leading_dot_std and std_str.startswith("0"):
                std_str = std_str[1:]
            formatted.iat[row_idx, col_idx] = f"\\valvar{{{mean_str}}}{{{std_str}}}"

    return formatted


def summarize_problem_tables(
    problem_mean_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create a long table with per-problem average performance for every measure."""
    rows = []
    for problem_type, table in problem_mean_tables.items():
        if table.empty:
            continue

        if DEFAULT_AVERAGE_ROW in table.index:
            averages = table.loc[DEFAULT_AVERAGE_ROW]
        else:
            averages = table.mean(axis=0, numeric_only=True)

        for measure, value in averages.items():
            rows.append(
                {
                    "problem_type": problem_type,
                    "measure": measure,
                    "mean_score": value,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["problem_type", "measure", "mean_score"])
    return pd.DataFrame(rows).sort_values(
        ["problem_type", "mean_score"],
        ascending=[True, False],
    )


def select_composition_columns(
    transformed_table: pd.DataFrame,
    composition_name: str,
) -> pd.DataFrame:
    """Select component, EntropicOT, and additive columns for a composition."""
    try:
        selected = select_composite_and_components(transformed_table, composition_name)
    except Exception:
        return pd.DataFrame(index=transformed_table.index)

    baseline_columns = [
        f"additive {composition_name}".lower(),
    ]
    for column in baseline_columns:
        if column in transformed_table.columns and column not in selected.columns:
            selected[column] = transformed_table[column]

    return selected


def build_pareto_summary(
    transformed_table: pd.DataFrame,
    composition_names: Sequence[str],
) -> pd.DataFrame:
    """Build article Pareto-front summary for EntropicOT and additive."""
    selected_compositions = {
        name: INTERESTING_COMPOSITIONS[name]
        for name in composition_names
        if name in INTERESTING_COMPOSITIONS
    }
    pareto_results = analyze_composite_pareto_performance(
        transformed_table,
        selected_compositions,
    )

    rows = []
    for result_name, result in pareto_results.items():
        aggregation_type, composition_name = split_pareto_result_name(result_name)
        rows.append(
            {
                "aggregation": aggregation_type,
                "composition": composition_name,
                "result_name": result_name,
                "pareto_count": result["pareto_count"],
                "total_problems": result["total_problems"],
                "pareto_percentage": result["pareto_percentage"],
                "average_pareto_depth": result["average_pareto_depth"],
                "median_pareto_depth": result["median_pareto_depth"],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "aggregation",
                "composition",
                "result_name",
                "pareto_count",
                "total_problems",
                "pareto_percentage",
                "average_pareto_depth",
                "median_pareto_depth",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["pareto_percentage", "pareto_count"],
        ascending=[False, False],
    )


def resolve_article_pareto_composition(
    requested_composition: str | None,
    composition_names: Sequence[str],
) -> str | None:
    """Resolve the composition used for the Table 2-style Pareto table."""
    if requested_composition is None:
        return None
    if requested_composition in composition_names:
        return requested_composition
    if composition_names:
        return composition_names[0]
    return None


def build_article_pareto_table(
    transformed_table: pd.DataFrame,
    composition_name: str,
    *,
    include_baselines: bool = True,
) -> pd.DataFrame:
    """
    Build the right-hand Table 2-style Pareto-front table.

    Unlike `pareto_summary`, this evaluates all displayed methods in one Pareto
    comparison: EntropicOT/Ours, optional additive baseline, and the
    individual components of the selected composition.
    """
    try:
        selected = select_composite_and_components(transformed_table, composition_name)
    except Exception:
        return empty_article_pareto_table()

    composition_col = composition_name.lower()
    aggregation_cols = [composition_col]
    if include_baselines:
        aggregation_cols.extend(
            [
                f"additive {composition_name}".lower(),
            ]
        )

    aggregation_cols = [
        col
        for col in aggregation_cols
        if col in transformed_table.columns or col in selected.columns
    ]
    for col in aggregation_cols:
        if col not in selected.columns and col in transformed_table.columns:
            selected[col] = transformed_table[col]

    component_cols = [col for col in selected.columns if col not in aggregation_cols]
    method_cols = [*aggregation_cols, *component_cols]
    if len(method_cols) < 2:
        return empty_article_pareto_table()

    stats = {
        col: {"pareto_count": 0, "total_pairs": 0, "depths": []}
        for col in method_cols
    }
    problems = list(selected.index)

    for problem1, problem2 in itertools.combinations(problems, 2):
        row1 = selected.loc[problem1]
        row2 = selected.loc[problem2]
        points = []
        valid_cols = []

        for col in method_cols:
            value1 = row1[col]
            value2 = row2[col]
            if pd.notna(value1) and pd.notna(value2):
                points.append((value1, value2))
                valid_cols.append(col)

        if len(points) < 2:
            continue

        front_indices = set(pareto_front(points))
        depths = pareto_depth(points)
        for idx, col in enumerate(valid_cols):
            stats[col]["total_pairs"] += 1
            stats[col]["depths"].append(depths[idx])
            if idx in front_indices:
                stats[col]["pareto_count"] += 1

    rows = []
    for order, col in enumerate(method_cols):
        total_pairs = stats[col]["total_pairs"]
        if total_pairs == 0:
            continue

        depths = pd.Series(stats[col]["depths"], dtype=float)
        pareto_count = stats[col]["pareto_count"]
        rows.append(
            {
                "method": article_pareto_method_label(col, composition_name),
                "measure": col,
                "pareto_count": pareto_count,
                "total_pairs": total_pairs,
                "pareto_percentage": (pareto_count / total_pairs) * 100,
                "average_pareto_depth": depths.mean(),
                "median_pareto_depth": depths.median(),
                "_order": order,
            }
        )

    if not rows:
        return empty_article_pareto_table()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["pareto_percentage", "pareto_count", "_order"],
            ascending=[False, False, True],
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def article_pareto_method_label(measure: str, composition_name: str) -> str:
    """Map internal columns to the compact labels used in the paper table."""
    if measure == composition_name.lower():
        return "Ours"
    if measure == f"additive {composition_name}".lower():
        return "Additive"
    if measure == "mahalanobis":
        return "MahS"

    match = re.match(r"R_([bet])\s+(\d+)(?:\s+(\d+))?\s+\(([^)]*)\)", measure)
    if not match:
        return measure

    risk_type, first_idx, second_idx, _score = match.groups()
    superscript = {
        "b": "b",
        "e": r"\mathrm{exc}",
        "t": r"\mathrm{tot}",
    }[risk_type]
    subscript = first_idx if second_idx is None else f"{first_idx},{second_idx}"
    return rf"$R_{{{subscript}}}^{{{superscript}}}$"


def format_article_pareto_table(
    table: pd.DataFrame,
    *,
    decimals: int = 3,
    highlight_top_two: bool = True,
) -> pd.DataFrame:
    """Format the Table 2-style Pareto table for LaTeX export."""
    if table.empty:
        return empty_article_pareto_latex_table()

    values = table["pareto_percentage"]
    unique_values = sorted(values.dropna().unique(), reverse=True)
    best = unique_values[0] if unique_values else None
    second = unique_values[1] if len(unique_values) > 1 else None

    formatted_values = []
    for value in values:
        if pd.isna(value):
            formatted_values.append("--")
            continue

        value_str = f"{value:.{decimals}f}\\%"
        if highlight_top_two and best is not None and value == best:
            value_str = f"\\textbf{{{value_str}}}"
        elif highlight_top_two and second is not None and value == second:
            value_str = f"\\underline{{{value_str}}}"
        formatted_values.append(value_str)

    return pd.DataFrame(
        {
            "Method": table["method"].tolist(),
            "At Pareto Front (\\%)": formatted_values,
        }
    )


def empty_article_pareto_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "method",
            "measure",
            "pareto_count",
            "total_pairs",
            "pareto_percentage",
            "average_pareto_depth",
            "median_pareto_depth",
        ]
    )


def empty_article_pareto_latex_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["Method", "At Pareto Front (\\%)"])


def split_pareto_result_name(result_name: str) -> tuple[str, str]:
    if result_name.startswith("Additive "):
        return "Additive", result_name.removeprefix("Additive ")
    return "EntropicOT", result_name


def write_paper_tables(
    bundle: PaperTableBundle,
    output_dir: str | Path,
    *,
    write_latex: bool = True,
) -> None:
    """Write CSV and optional LaTeX versions of all generated tables."""
    output_path = Path(output_dir)
    problem_dir = output_path / "problem_tables"
    latex_dir = output_path / "latex"
    composition_dir = output_path / "composition_tables"
    composition_latex_dir = output_path / "composition_latex"

    problem_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    bundle.all_tasks_mean.to_csv(output_path / "all_tasks_mean.csv")
    bundle.all_tasks_std.to_csv(output_path / "all_tasks_std.csv")
    bundle.average_ranks.to_csv(output_path / "average_ranks.csv", index=False)
    bundle.measure_summary.to_csv(output_path / "measure_summary.csv", index=False)
    bundle.pareto_summary.to_csv(output_path / "pareto_summary.csv", index=False)
    bundle.article_pareto_table.to_csv(
        output_path / "article_pareto_table.csv",
        index=False,
    )

    for problem_type, table in bundle.problem_mean_tables.items():
        table.to_csv(problem_dir / f"{problem_type}_mean.csv")
        bundle.problem_std_tables[problem_type].to_csv(
            problem_dir / f"{problem_type}_std.csv"
        )

    if write_latex:
        latex_dir.mkdir(parents=True, exist_ok=True)
        for problem_type, table in bundle.problem_latex_tables.items():
            (latex_dir / f"{problem_type}.tex").write_text(
                table.to_latex(escape=False),
                encoding="utf-8",
            )
        (latex_dir / "article_pareto_table.tex").write_text(
            bundle.article_pareto_latex_table.to_latex(index=False, escape=False),
            encoding="utf-8",
        )

    if bundle.composition_mean_tables:
        composition_dir.mkdir(parents=True, exist_ok=True)
        for composition_name, tables_by_problem in bundle.composition_mean_tables.items():
            slug = slugify(composition_name)
            for problem_type, table in tables_by_problem.items():
                table.to_csv(composition_dir / f"{slug}_{problem_type}_mean.csv")
                bundle.composition_std_tables[composition_name][problem_type].to_csv(
                    composition_dir / f"{slug}_{problem_type}_std.csv"
                )

        if write_latex:
            composition_latex_dir.mkdir(parents=True, exist_ok=True)
            for composition_name, tables_by_problem in (
                bundle.composition_latex_tables.items()
            ):
                slug = slugify(composition_name)
                for problem_type, table in tables_by_problem.items():
                    (composition_latex_dir / f"{slug}_{problem_type}.tex").write_text(
                        table.to_latex(escape=False),
                        encoding="utf-8",
                    )


def slugify(value: str) -> str:
    """Create a stable filename stem for a measure or composition name."""
    keep_chars = []
    for char in value.lower():
        if char.isalnum():
            keep_chars.append(char)
        elif keep_chars and keep_chars[-1] != "_":
            keep_chars.append("_")
    return "".join(keep_chars).strip("_")
