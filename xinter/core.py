"""
Core functionality for xinter: loading datasets, applying linters, and processing results.
"""

from typing import Optional, Literal
from pydantic import BaseModel
import xarray as xr
import pandas as pd

from xinter.linters import LinterRegistry, LinterResult


TargetType = Literal["data_vars", "coords"]


class Report(BaseModel):
    """Data model for storing linting results for a single file and target type."""

    file_path: str
    group: Optional[str]
    type: TargetType
    results: dict[str, dict[str, LinterResult]]


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
    except (ValueError, RuntimeError, IOError, KeyError) as e:
        return (file_path, None, str(e))


def lint_dataset(
    obj: str | xr.Dataset,
    group: Optional[str] = None,
    check_coords: bool = False,
    engine: Optional[str] = None,
) -> list[Report]:
    """Lint an xarray dataset using all registered checkers.

    Args:
        obj: Path to the xarray-compatible file or an xarray.Dataset
        group: Optional group name for datasets with groups
        check_coords: Whether to also check coordinates in addition to data variables

    Returns:
        A list summarizing the linting results.
    """

    if isinstance(obj, str):
        dataset = xr.open_dataset(obj, group=group, engine=engine)
    else:
        dataset = obj

    # Get all registered checkers and instantiate them
    checkers = [checker_class() for checker_class in LinterRegistry.get_checkers()]

    targets: list[TargetType] = ["data_vars"]
    if check_coords:
        targets.append("coords")

    reports = []
    for target in targets:
        checks = {}
        for checker in checkers:
            result = checker.check_dataset(dataset, target=target)
            checks[checker.name] = result

        report = Report(
            file_path=obj if isinstance(obj, str) else "N/A",
            group=group if group else "N/A",
            type=target,
            results=checks,
        )

        reports.append(report)

    return reports


def reports_to_dataframe(results: list[list[Report]]) -> pd.DataFrame:
    """Convert a list of lists of Report objects into a single pandas DataFrame.
    Args:
        results: A list of lists of Report objects, where each inner list corresponds
        to a file's reports.
    Returns:
        A pandas DataFrame with columns: file_path, target_type, variable_name,
        checker_name, value, message, success
    """
    # Process and display results after parallel execution
    dfs = []
    for reports in results:
        reports_df = _report_to_dataframe(reports)
        dfs.append(reports_df)

    if not dfs:
        return pd.DataFrame()  # Return empty DataFrame if no reports

    dfs_combined = pd.concat(dfs, ignore_index=True)
    return dfs_combined


def _report_to_dataframe(reports: list[Report]) -> pd.DataFrame:
    """Convert nested reports structure to a flat pandas DataFrame.

    Args:
        reports: List of Report objects

    Returns:
        A pandas DataFrame with columns: file_path, target_type, variable_name,
        checker_name, value, message, success
    """
    rows = []

    for report in reports:
        file_path = report.file_path
        group = report.group
        target_type = report.type
        for checker_name, result in report.results.items():
            for var_name, item in result.items():
                rows.append(
                    {
                        "file_path": file_path,
                        "group": group,
                        "target_type": target_type,
                        "variable_name": var_name,
                        "checker_name": checker_name,
                        "value": item.value,
                        "message": item.message,
                        "success": item.success,
                    }
                )

    return pd.DataFrame(rows)
