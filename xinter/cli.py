"""Command-line interface for the XR Linter."""

from typing import Optional
import argparse
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.table import Table
from rich.spinner import Spinner
from rich.live import Live
from joblib import Parallel, delayed
from xinter.linters import LinterRegistry


def lint_dataset(
    file_path: str, group: Optional[str] = None, check_coords: bool = False
):
    """Lint an xarray dataset using all registered checkers.

    Args:
        file_path: Path to the xarray-compatible file
        group: Optional group name for datasets with groups
        check_coords: Whether to also check coordinates in addition to data variables

    Returns:
        A dict summarizing the linting results.
    """

    dataset = xr.open_dataset(file_path, group=group)

    # Get all registered checkers and instantiate them
    checkers = [checker_class() for checker_class in LinterRegistry.get_checkers()]

    targets = ["data_vars"]
    if check_coords:
        targets.append("coords")

    reports = {}
    for target in targets:
        checks = {}
        checks["type"] = target
        checks["file_path"] = file_path
        for checker in checkers:
            result = checker.check_dataset(dataset, target=target)
            checks[checker.name] = result

        reports[target] = checks

    return reports


def lint_dataset_with_error_handling(
    file_path: str, group: Optional[str] = None, check_coords: bool = False
):
    """Wrapper around lint_dataset that catches exceptions.

    Args:
        file_path: Path to the xarray-compatible file
        group: Optional group name for datasets with groups
        check_coords: Whether to also check coordinates in addition to data variables

    Returns:
        A tuple of (file_path, reports_dict, error_message).
        If successful, error_message is None. If failed, reports_dict is None.
    """
    try:
        reports = lint_dataset(file_path, group=group, check_coords=check_coords)
        return (file_path, reports, None)
    except (ValueError, RuntimeError, IOError) as e:
        return (file_path, None, str(e))


def reports_to_dataframe(reports: dict) -> pd.DataFrame:
    """Convert nested reports structure to a flat pandas DataFrame.

    Args:
        reports: Dict with target types as keys and check results as values

    Returns:
        A pandas DataFrame with columns: file_path, target_type, variable_name,
        checker_name, value, message, success
    """
    rows = []

    for target_type, checks in reports.items():
        file_path = checks.get("file_path", "unknown")
        for checker_name, results in checks.items():
            if checker_name in ("type", "file_path"):
                continue
            for var_name, result in results.items():
                rows.append(
                    {
                        "file_path": file_path,
                        "target_type": target_type,
                        "variable_name": var_name,
                        "checker_name": checker_name,
                        "value": result.value,
                        "message": result.message,
                        "success": result.success,
                    }
                )

    return pd.DataFrame(rows)


def print_linting_reports(console: Console, results: list) -> list:
    """Print linting reports to console and collect dataframes.

    Args:
        console: Rich Console instance for printing
        results: List of tuples (file_path, reports_dict, error_message)

    Returns:
        List of pandas DataFrames containing the linting results
    """
    dfs = []

    for file_path, reports, error in results:
        if error is not None:
            # Handle error case
            console.print(f"[red]Error linting {file_path}: {error}[/red]")
            continue

        reports_df = reports_to_dataframe(reports)
        dfs.append(reports_df)

        console.print(f"Linting report for {file_path}:")

        for target_type, checks in reports.items():
            console.print(f"\nResults for {target_type}:")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Variable", style="dim", width=30)
            table.add_column("Check", style="dim", width=20)
            table.add_column("Message")

            for check_name, items in checks.items():
                if check_name in ("type", "file_path"):
                    continue
                for var_name, value in items.items():
                    if not value.success:
                        table.add_row(var_name, check_name, value.message)

            console.print(table)

    return dfs


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
        default="linting_report.csv",
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
    spinner = Spinner("dots", text=f"Linting {len(args.files)} file(s)...")
    with Live(spinner, console=console, transient=True):
        results = Parallel(n_jobs=args.num_jobs)(
            delayed(lint_dataset_with_error_handling)(
                file, group=args.group, check_coords=args.coords
            )
            for file in args.files
        )

    # Process and display results after parallel execution
    dfs = print_linting_reports(console, results)

    console.print("\nLinting completed.")

    if dfs:
        dfs_combined = pd.concat(dfs, ignore_index=True)
        # Optionally, save the combined report to a CSV file
        dfs_combined.to_csv(args.output, index=False)
        console.print(f"Combined linting report saved to {args.output}")
    else:
        console.print(
            "[red]No files were successfully linted. No report generated.[/red]"
        )


if __name__ == "__main__":
    main()
