"""Command-line interface for the XR Linter."""

import sys
import argparse
from rich.console import Console
from joblib import Parallel, delayed

from xinter.core import (
    lint_dataset_with_error_handling,
    reports_to_dataframe,
)


def gather_results(console, results):
    output = []
    for result in results:
        file_path, reports, error = result
        if error is not None:
            console.print(f"[red]Error linting {file_path}: {error}[/red]")
        else:
            console.print(f"[green]Successfully linted {file_path}[/green]")
        output.append(result)
    return output


def main():
    """Command-line interface for the XR Linter."""
    parser = argparse.ArgumentParser(description="XR Linter CLI")
    parser.add_argument(
        "files",
        metavar="FILE",
        type=str,
        nargs="+",
        help="Files to be linted",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Group name for xarray",
    )
    parser.add_argument(
        "--coords",
        action="store_true",
        help="Also check coordinates in addition to data variables",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="linting_report.parquet",
        help="Location to save output file",
    )
    parser.add_argument(
        "-n",
        "--num-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run (default: use all available cores)",
    )

    args = parser.parse_args()

    console = Console()

    # Run linting in parallel for all files with a spinner
    jobs = [
        delayed(lint_dataset_with_error_handling)(
            file, group=args.group, check_coords=args.coords
        )
        for file in args.files
    ]
    results = Parallel(n_jobs=args.num_jobs, return_as="generator_unordered")(jobs)
    results = gather_results(console, results)
    console.print("\nLinting completed.")

    dfs = reports_to_dataframe(results)
    if dfs.empty:
        console.print(
            "[red]No files were successfully linted. No report generated.[/red]"
        )
        sys.exit(0)

    dfs = dfs.sort_values(
        by=["file_path", "group", "target_type", "variable_name", "checker_name"]
    )

    dfs = dfs.pivot(
        index=["file_path", "group", "variable_name", "target_type"],
        columns="checker_name",
        values="value",
    )

    if args.output.endswith(".parquet"):
        dfs.to_parquet(args.output, index=False)
    elif args.output.endswith(".csv"):
        dfs.to_csv(args.output, index=False)
    else:
        console.print(
            "[red]Unsupported output format. Please use .parquet or .csv extension.[/red]"
        )
        sys.exit(1)

    console.print(f"Combined linting report saved to {args.output}")


if __name__ == "__main__":
    main()
