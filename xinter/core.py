from typing import Optional
import xarray as xr

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
        checks["group"] = group if group else "N/A"
        for checker in checkers:
            result = checker.check_dataset(dataset, target=target)
            checks[checker.name] = result

        reports[target] = checks

    return reports
