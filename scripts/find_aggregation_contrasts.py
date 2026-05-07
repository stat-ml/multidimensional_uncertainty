import argparse
import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdu.eval.aggregation_contrasts import (
    build_and_write_aggregation_contrast_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find table rows where one aggregation baseline works while the "
            "other aggregation method trails clearly."
        )
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="./resources/refactored/results.csv",
        help="CSV produced by scripts/full_evaluation.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./resources/paper_tables/aggregation_contrasts",
        help="Directory for selected contrast cases and tables",
    )
    parser.add_argument(
        "--selective_metric",
        type=str,
        default="acc_cov_auc",
        help="Metric column to use for selective prediction rows",
    )
    parser.add_argument(
        "--min_gap",
        type=float,
        default=0.05,
        help="Minimum score gap between the winner and a broken aggregation",
    )
    parser.add_argument(
        "--min_broken",
        type=int,
        default=1,
        help="How many non-winning aggregations must trail by at least min_gap",
    )
    parser.add_argument(
        "--min_winner_score",
        type=float,
        default=None,
        help="Optional absolute minimum score for the winning aggregation",
    )
    parser.add_argument(
        "--max_loser_score",
        type=float,
        default=None,
        help="Optional absolute maximum score for an aggregation to count as broken",
    )
    parser.add_argument(
        "--top_n_per_table",
        type=int,
        default=25,
        help="Maximum selected rows to keep in each per-composition inspection table",
    )
    parser.add_argument(
        "--no_latex",
        action="store_true",
        help="Write CSV files only",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_and_write_aggregation_contrast_report(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        selective_metric=args.selective_metric,
        min_gap=args.min_gap,
        min_broken=args.min_broken,
        min_winner_score=args.min_winner_score,
        max_loser_score=args.max_loser_score,
        top_n_per_table=args.top_n_per_table,
        write_latex=not args.no_latex,
    )

    print(f"Aggregation contrast report written to {args.output_dir}")
    print(f"Selected cases: {len(report.cases)}")
    print(f"Selected inspection tables: {len(report.selected_mean_tables)}")
    if not report.summary_by_winner.empty:
        print("\nTop winner patterns:")
        print(report.summary_by_winner.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
