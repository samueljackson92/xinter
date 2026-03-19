from typing import Optional
import xarray as xr
import pandas as pd

from xinter.linters import LinterRegistry


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
        checks["group"] = group if group else "N/A"
        for checker in checkers:
            result = checker.check_dataset(dataset, target=target)
            checks[checker.name] = result

        reports[target] = checks

    return reports


def reports_to_dataframe(results: list) -> pd.DataFrame:
    # Process and display results after parallel execution
    dfs = []
    for file_path, reports, error in results:
        if error is not None:
            # Handle error case
            continue

        reports_df = _report_to_dataframe(reports)
        dfs.append(reports_df)

    if dfs:
        dfs_combined = pd.concat(dfs, ignore_index=True)
        return dfs_combined
    else:
        return pd.DataFrame()  # Return empty DataFrame if no successful reports


def _report_to_dataframe(reports: dict) -> pd.DataFrame:
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
        group = checks.get("group", "N/A")
        for checker_name, results in checks.items():
            if checker_name in ("type", "file_path", "group"):
                continue
            for var_name, result in results.items():
                rows.append(
                    {
                        "file_path": file_path,
                        "group": group,
                        "target_type": target_type,
                        "variable_name": var_name,
                        "checker_name": checker_name,
                        "value": result.value,
                        "message": result.message,
                        "success": result.success,
                    }
                )

    return pd.DataFrame(rows)
