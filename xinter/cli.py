"""Command-line interface for the XR Linter."""

import argparse
import tempfile
import multiprocessing as mp
import shutil
import sys
from pathlib import Path
from functools import partial

import pandas as pd
from rich.console import Console

from xinter.core import lint_dataset_with_error_handling


def gather_results(console, results):
    """Gather results from parallel linting and print summary to console."""
    for result in results:
        file_path, error = result
        if error is not None:
            console.print(f"[red]Error linting {file_path}: {error}[/red]")
        else:
            console.print(f"[green]Successfully linted {file_path}[/green]")


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

    output_file = Path(args.output)
    tmp_dir = Path(tempfile.mkdtemp(dir="."))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    console = Console()

    # Run linting in parallel for all files
    linting_func = partial(
        lint_dataset_with_error_handling,
        group=args.group,
        check_coords=args.coords,
        output_dir=tmp_dir,
    )
    with mp.Pool(processes=args.num_jobs if args.num_jobs > 0 else None) as pool:
        results = pool.imap_unordered(linting_func, args.files, chunksize=1)
        gather_results(console, results)

    console.print("\nLinting completed.")

    dfs = pd.concat(
        [
            pd.read_parquet(tmp_dir / f"{Path(file).stem}_linting_report.parquet")
            for file in args.files
        ],
        ignore_index=True,
    )

    if dfs.empty:
        console.print(
            "[red]No files were successfully linted. No report generated.[/red]"
        )
        sys.exit(0)

    dfs = dfs.sort_values(
        by=["file_path", "group", "target_type", "variable_name", "checker_name"]
    )

    type_lookup = dfs[["checker_name", "value_type"]]

    dfs = dfs.pivot(
        index=["file_path", "group", "variable_name", "target_type"],
        columns="checker_name",
        values="value",
    )

    type_map = {
        "int": int,
        "float": float,
        "bool": lambda x: x == "True",
        "str": str,
    }

    for col in dfs.columns:
        # get the type for this column from the metadata
        dtype = type_lookup[type_lookup["checker_name"] == col]["value_type"].values[0]
        dfs[col] = dfs[col].apply(lambda x: type_map[dtype](x))

    if output_file.suffix == ".parquet":
        dfs.to_parquet(output_file, index=True)
    elif output_file.suffix == ".csv":
        dfs.to_csv(output_file, index=True)
    else:
        console.print(
            "[red]Unsupported output format. Please use .parquet or .csv.[/red]"
        )
        sys.exit(1)
    shutil.rmtree(tmp_dir)  # Clean up individual parquet files
    console.print(f"Combined linting report saved to [green]{output_file}[/green]")


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
