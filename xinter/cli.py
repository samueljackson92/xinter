"""Command-line interface for the XR Linter."""

from typing import Optional
import argparse
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.table import Table
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
    args = parser.parse_args()

    console = Console()

    # Placeholder for linting logic
    dfs = []
    for file in args.files:
        reports = lint_dataset(file, group=args.group, check_coords=args.coords)

        reports_df = reports_to_dataframe(reports)
        dfs.append(reports_df)

        console.print(f"Linting report for {file}:")

        for target_type, checks in reports.items():
            console.print(f"\nResults for {target_type}:")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Variable", style="dim", width=30)
            table.add_column("Check", style="dim", width=20)
            table.add_column("Message")

            for check_name, results in checks.items():
                if check_name in ("type", "file_path"):
                    continue
                for var_name, value in results.items():
                    if not value.success:
                        table.add_row(var_name, check_name, value.message)

            console.print(table)

    console.print("\nLinting completed.")

    dfs_combined = pd.concat(dfs, ignore_index=True)
    # Optionally, save the combined report to a CSV file
    dfs_combined.to_csv(args.output, index=False)
    console.print(f"Combined linting report saved to {args.output}")


if __name__ == "__main__":
    main()
