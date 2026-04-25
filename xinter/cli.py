"""Command-line interface for the xinter."""

import os
import argparse
import tempfile
import multiprocessing as mp
import shutil
import sys
from pathlib import Path
from functools import partial

import pandas as pd
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from xinter.core import lint_dataset_with_error_handling

console = Console()

WORKER_TIMEOUT = int(
    os.environ.get("XINTER_WORKER_TIMEOUT", 60 * 5)
)  # Default to 5 minutes


def gather_results(futures):
    """Gather results from parallel linting and print summary to console."""
    success_count = 0
    error_count = 0
    timeout_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Linting files", total=len(futures))

        for file_path, future in futures:
            try:
                _, error = future.get(
                    timeout=WORKER_TIMEOUT
                )  # Wait up to WORKER_TIMEOUT seconds for each file to be linted
            except mp.TimeoutError:
                console.print(f"[yellow]⌛ Timeout linting {file_path}[/yellow]")
                timeout_count += 1
                progress.advance(task)
                continue

            if error is not None:
                console.print(f"[red]❌ Error linting {file_path}:[/red] {error}")
                error_count += 1
            else:
                console.print(f"[green]✅ Successfully linted {file_path}[/green]")
                success_count += 1

            progress.advance(task)

    # Create summary table
    table = Table(title="Linting Summary", show_header=False, box=None, padding=(0, 1))
    table.add_column(justify="left", no_wrap=True)  # Emoji column
    table.add_column(justify="left", no_wrap=True)  # Label column
    table.add_column(justify="right", no_wrap=True)  # Number column
    table.add_row(
        "[green]✅[/green]",
        "[green]Successful[/green]",
        f"[bold]{success_count:>6}[/bold]",
    )
    table.add_row(
        "[red]❌[/red]", "[red]Errors[/red]", f"[bold]{error_count:>6}[/bold]"
    )
    table.add_row(
        "[yellow]⌛[/yellow]",
        "[yellow]Timeouts[/yellow]",
        f"[bold]{timeout_count:>6}[/bold]",
    )
    table.add_row(
        "[blue]📊[/blue]", "[blue]Total[/blue]", f"[bold]{len(futures):>6}[/bold]"
    )

    console.print()
    console.print(table)
    console.print()

    return success_count, error_count, timeout_count


def main():
    """Command-line interface for the xinter."""
    parser = argparse.ArgumentParser(description="xinter CLI")
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
        default=None,
        help="Number of parallel jobs to run (default: use all available cores)",
    )
    parser.add_argument(
        "--channel-wise",
        action="store_true",
        help="Perform linting in a channel-wise manner on datasets where one axis is time",
    )

    args = parser.parse_args()

    output_file = Path(args.output)
    tmp_dir = Path(tempfile.mkdtemp(dir="."))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Run linting in parallel for all files
    linting_func = partial(
        lint_dataset_with_error_handling,
        group=args.group,
        check_coords=args.coords,
        channel_wise=args.channel_wise,
        output_dir=tmp_dir,
    )
    console.print()
    console.print("[bold cyan]🔍 Starting XR Linter[/bold cyan]")
    console.print(f"[dim]Files to process: {len(args.files)}[/dim]")
    console.print(f"[dim]Workers: {args.num_jobs if args.num_jobs else 'auto'}[/dim]")
    console.print()

    with mp.Pool(processes=args.num_jobs, maxtasksperchild=1) as pool:
        futures = [
            (item, pool.apply_async(linting_func, (item,))) for item in args.files
        ]
        success_count, error_count, timeout_count = gather_results(futures)

    if error_count > 0 or timeout_count > 0:
        console.print(
            f"[yellow]⚠️  Linting completed with {error_count} errors "
            f"and {timeout_count} timeouts.[/yellow]"
        )
    else:
        console.print(
            f"[green bold]✅ Linting completed successfully! "
            f"{success_count} files processed.[/green bold]"
        )

    parquet_files = list(Path(tmp_dir).glob("*.parquet"))

    if not parquet_files:
        console.print(
            "[yellow]⚠️  No files were successfully linted. No report generated.[/yellow]"
        )
        sys.exit(1)

    console.print(
        f"[cyan]📦 Combining results from {len(parquet_files)} files...[/cyan]"
    )
    dfs = pd.concat(
        [pd.read_parquet(file) for file in parquet_files],
        ignore_index=True,
    )

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
        dfs[col] = dfs[col].apply(type_map[dtype])

    dfs.drop("variable_name", axis=1, inplace=True)
    dfs = dfs.reset_index()

    if output_file.suffix == ".parquet":
        dfs.to_parquet(output_file)
    elif output_file.suffix == ".csv":
        dfs.to_csv(output_file)
    else:
        console.print(
            "[red]❌ Error: Unsupported output format. Please use .parquet or .csv.[/red]"
        )
        sys.exit(1)
    shutil.rmtree(tmp_dir)  # Clean up individual parquet files

    console.print()
    console.print(f"[green]💾 Report saved to:[/green] [bold]{output_file}[/bold]")
    console.print(f"[dim]   Format: {output_file.suffix}[/dim]")
    console.print(f"[dim]   Rows: {len(dfs):,}[/dim]")
    console.print()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
