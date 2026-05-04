import argparse
import sys
from pathlib import Path

# project root = parent of "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdu.eval.paper_tables import build_and_write_paper_tables


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build article-ready tables from full_evaluation.py output."
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
        default="./resources/paper_tables",
        help="Directory for generated CSV and LaTeX tables",
    )
    parser.add_argument(
        "--selective_metric",
        type=str,
        default="acc_cov_auc",
        help="Metric column to use for selective prediction tables",
    )
    parser.add_argument(
        "--no_latex",
        action="store_true",
        help="Write CSV files only",
    )
    parser.add_argument(
        "--no_composition_tables",
        action="store_true",
        help="Skip per-composition component/baseline tables",
    )
    parser.add_argument(
        "--mean_decimals",
        type=int,
        default=3,
        help="Number of decimals for formatted means",
    )
    parser.add_argument(
        "--std_decimals",
        type=int,
        default=3,
        help="Number of decimals for formatted standard deviations",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = build_and_write_paper_tables(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        selective_metric=args.selective_metric,
        include_composition_tables=not args.no_composition_tables,
        write_latex=not args.no_latex,
        mean_decimals=args.mean_decimals,
        std_decimals=args.std_decimals,
    )

    print(f"Paper tables written to {args.output_dir}")
    print(f"All-task table shape: {bundle.all_tasks_mean.shape}")
    for problem_type, table in bundle.problem_mean_tables.items():
        print(f"{problem_type}: {table.shape}")
    print(f"Average-rank rows: {len(bundle.average_ranks)}")
    print(f"Composition table groups: {len(bundle.composition_mean_tables)}")


if __name__ == "__main__":
    main()
