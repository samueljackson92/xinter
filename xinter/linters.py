""" "Module defining linters (checkers) for xarray DataArrays and Datasets."""

from abc import ABC, abstractmethod
from scipy.stats import entropy
import numpy as np
import xarray as xr
import pint_xarray  # pylint: disable=unused-import
from pydantic import BaseModel


class CheckerResult(BaseModel):
    """Result of a checker applied to a DataArray."""

    value: bool | str | float | int
    message: str
    success: bool


class DataArrayChecker(ABC):
    """Abstract base class for checking xarray DataArrays."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the checker."""
        raise NotImplementedError("Subclasses must implement this property.")

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the checker."""
        raise NotImplementedError("Subclasses must implement this property.")

    @abstractmethod
    def check(self, var: xr.DataArray) -> CheckerResult:
        """Check a single xarray DataArray."""
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def is_coord(var: xr.DataArray) -> bool:
        """Check if a DataArray is a coordinate dimension.

        A coordinate dimension is defined as a 1-dimensional array where
        the dimension name matches the variable name.

        Args:
            var: The DataArray to check

        Returns:
            True if the DataArray is a coordinate dimension, False otherwise
        """
        return len(var.dims) == 1 and var.dims[0] == var.name

    def check_dataset(self, dataset: xr.Dataset, target: str = "data_vars") -> dict:
        """Apply the check method to each variable in the dataset.

        Args:
            dataset: The xarray Dataset to process
            target: Either 'data_vars' or 'coords' to specify what to iterate over

        Returns:
            A dict with variable names as keys and check results as values.
        """
        report = {}
        items = dataset.data_vars if target == "data_vars" else dataset.coords
        for var in items:
            report[var] = self.check(dataset[var])
        return report


class LinterRegistry:
    """Registry for DataArrayChecker classes.

    Allows users to register custom checkers using the @LinterRegistry.register() decorator.
    """

    _checkers = []

    @classmethod
    def register(cls):
        """Decorator to register a DataArrayChecker class.

        Usage:
            @LinterRegistry.register()
            class MyCustomChecker(DataArrayChecker):
                name = "My Check"
                description = "My custom check"

                def check(self, var: xr.DataArray) -> Any:
                    return some_value
        """

        def decorator(checker_class):
            cls._checkers.append(checker_class)
            return checker_class

        return decorator

    @classmethod
    def get_checkers(cls):
        """Get all registered checker classes."""
        return cls._checkers

    @classmethod
    def clear(cls):
        """Clear all registered checkers. Useful for testing."""
        cls._checkers = []


@LinterRegistry.register()
class NaNsChecker(DataArrayChecker):
    """Check for the proportion of NaN (Not a Number) values in the array.

    Returns a float between 0 and 1, where 0 means no NaN values and 1 means all values are NaN.
    High proportions of NaN values may indicate data quality issues or missing measurements.
    """

    name = "NaNs"
    description = "Proportion of NaN values"
    tolerance: float = 0.2

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = var.isnull().sum().item() / var.size
        success = value <= self.tolerance
        return CheckerResult(
            value=value, message=f"{value * 100:.2f}% NaNs found.", success=success
        )


@LinterRegistry.register()
class MeanChecker(DataArrayChecker):
    """Calculate the arithmetic mean of all values in the array.

    The mean provides a measure of central tendency and can help identify
    if values are centered around expected ranges.
    """

    name = "Mean"
    description = "Mean value"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = var.mean().item()
        return CheckerResult(value=value, message=f"Mean value: {value}", success=True)


@LinterRegistry.register()
class StdChecker(DataArrayChecker):
    """Calculate the standard deviation of all values in the array.

    Standard deviation measures the spread or dispersion of values around the mean.
    Low values indicate data points are close to the mean, while high values indicate
    more variability in the data.
    """

    name = "Standard deviation"
    description = "Standard deviation"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = var.std().item()
        return CheckerResult(
            value=value, message=f"Standard deviation: {value}", success=True
        )


@LinterRegistry.register()
class IQROutliersChecker(DataArrayChecker):
    """Detect outliers using the Interquartile Range (IQR) method.

    Values are considered outliers if they fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR,
    where Q1 and Q3 are the 25th and 75th percentiles respectively.
    Returns the proportion of values identified as outliers (0 to 1).
    """

    name = "IQR outliers"
    description = "Proportion of values outside IQR range"

    def check(self, var: xr.DataArray) -> CheckerResult:
        q1 = var.quantile(0.25).item()
        q3 = var.quantile(0.75).item()
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((var < lower_bound) | (var > upper_bound)).sum().item()
        value = outliers / var.size
        return CheckerResult(
            value=value, message=f"Proportion of outliers: {value}", success=True
        )


@LinterRegistry.register()
class RangeChecker(DataArrayChecker):
    """Calculate the range of values (maximum minus minimum).

    The range provides a simple measure of the spread of data. Large ranges may indicate
    high variability or the presence of outliers.
    """

    name = "Range"
    description = "Range of values (max - min)"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = var.max().item() - var.min().item()
        return CheckerResult(
            value=value, message=f"Range of values: {value}", success=True
        )


@LinterRegistry.register()
class MaxChecker(DataArrayChecker):
    """Find the maximum value in the array.

    Useful for identifying the upper bound of the data and detecting potentially
    unrealistic values that exceed expected limits.
    """

    name = "Max"
    description = "Maximum value"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = var.max().item()
        return CheckerResult(
            value=value, message=f"Maximum value: {value}", success=True
        )


@LinterRegistry.register()
class MinChecker(DataArrayChecker):
    """Find the minimum value in the array.

    Useful for identifying the lower bound of the data and detecting potentially
    unrealistic values that fall below expected limits.
    """

    name = "Min"
    description = "Minimum value"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = var.min().item()
        return CheckerResult(
            value=value, message=f"Minimum value: {value}", success=True
        )


@LinterRegistry.register()
class DuplicateValuesChecker(DataArrayChecker):
    """Calculate the proportion of values that appear more than once.

    High proportions of duplicates may indicate constant or repeated values,
    which could suggest sensor issues, data recording problems, or expected
    patterns in the data.
    """

    name = "Duplicate values"
    description = "Proportion of duplicate values"
    tolerance: float = 0.95

    def check(self, var: xr.DataArray) -> CheckerResult:
        _, counts = np.unique(var.values, return_counts=True)
        duplicates = counts[counts > 1].sum().item()
        value = duplicates / var.size
        success = value < self.tolerance
        message = f"High proportion of duplicate values (> {self.tolerance * 100}): {value * 100}%"
        return CheckerResult(
            value=value,
            message=message,
            success=success,
        )


@LinterRegistry.register()
class NegativeValuesChecker(DataArrayChecker):
    """Calculate the proportion of negative values in the array.

    Useful for identifying variables that should be strictly non-negative
    (e.g., physical quantities like distance, count, or absolute measurements).
    """

    name = "Negative values"
    description = "Proportion of negative values"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = (var < 0).sum().item() / var.size
        return CheckerResult(
            value=value,
            message=f"Proportion of negative values: {value * 100}%",
            success=True,
        )


@LinterRegistry.register()
class ZeroValuesChecker(DataArrayChecker):
    """Calculate the proportion of exact zero values in the array.

    High proportions of zeros may indicate missing data encoded as zeros,
    sparse data, or measurement periods with no activity.
    """

    name = "Zero values"
    description = "Proportion of zero values"
    tolerance: float = 0.95

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = (var == 0).sum().item() / var.size
        success = value < self.tolerance
        return CheckerResult(
            value=value,
            message=f"Proportion of zeros is greater than {self.tolerance * 100}: {value * 100}%",
            success=success,
        )


@LinterRegistry.register()
class ConstantValuesChecker(DataArrayChecker):
    """Check if all values in the array are identical.

    Returns True if the array contains only one unique value, False otherwise.
    Constant arrays may indicate stuck sensors, configuration issues, or
    legitimately constant parameters.
    """

    name = "Constant values"
    description = "Whether all values are constant"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = len(np.unique(var.values)) == 1
        return CheckerResult(
            value=value,
            message=f"All values are constant: {value}",
            success=not value,
        )


@LinterRegistry.register()
class InfiniteValuesChecker(DataArrayChecker):
    """Calculate the proportion of infinite values (both +inf and -inf).

    Infinite values often result from division by zero or numerical overflow
    in calculations, and typically indicate data quality or processing issues.
    """

    name = "Infinite values"
    description = "Proportion of infinite values"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = np.isinf(var.values).sum().item() / var.size
        success = value == 0
        return CheckerResult(
            value=value,
            message=f"Array contains infinite values: {value * 100}%",
            success=success,
        )


@LinterRegistry.register()
class SkewnessChecker(DataArrayChecker):
    """Calculate the skewness (third standardized moment) of the distribution.

    Skewness measures the asymmetry of the distribution:
    - 0: symmetric (like normal distribution)
    - Positive: right-skewed (tail extends to the right)
    - Negative: left-skewed (tail extends to the left)
    Returns 0 if standard deviation is 0 or sample size < 3.
    """

    name = "Skewness"
    description = "Skewness of the distribution"

    def check(self, var: xr.DataArray) -> CheckerResult:
        mean = var.mean().item()
        std = var.std().item()
        n = var.size
        if std == 0 or n < 3:
            return CheckerResult(
                value=0.0,
                message="Skewness is 0 due to zero std or insufficient sample size",
                success=False,
            )
        skewness = ((var - mean) ** 3).mean().item() / (std**3)
        return CheckerResult(
            value=skewness,
            message=f"Skewness: {skewness}",
            success=True,
        )


@LinterRegistry.register()
class KurtosisChecker(DataArrayChecker):
    """Calculate the excess kurtosis (fourth standardized moment) of the distribution.

    Kurtosis measures the "tailedness" of the distribution:
    - 0: normal distribution (mesokurtic)
    - Positive: heavy tails, more outliers (leptokurtic)
    - Negative: light tails, fewer outliers (platykurtic)
    Returns 0 if standard deviation is 0 or sample size < 4.
    """

    name = "Kurtosis"
    description = "Kurtosis of the distribution"

    def check(self, var: xr.DataArray) -> CheckerResult:
        mean = var.mean().item()
        std = var.std().item()
        n = var.size
        if std == 0 or n < 4:
            return CheckerResult(
                value=0.0,
                message="Kurtosis is 0 due to zero std or insufficient sample size",
                success=True,
            )
        kurtosis = ((var - mean) ** 4).mean().item() / (std**4) - 3
        return CheckerResult(
            value=kurtosis,
            message=f"Kurtosis: {kurtosis}",
            success=True,
        )


@LinterRegistry.register()
class EntropyChecker(DataArrayChecker):
    """Calculate the Shannon entropy of the value distribution.

    Entropy measures the randomness or unpredictability of the data:
    - Low entropy: values are concentrated in few states (more predictable)
    - High entropy: values are spread across many states (more random)
    Useful for assessing data diversity and information content.
    """

    name = "Entropy"
    description = "Shannon entropy of the distribution"

    def check(self, var: xr.DataArray) -> CheckerResult:
        _, counts = np.unique(var.values, return_counts=True)
        probabilities = counts / counts.sum()
        entropy_value = entropy(probabilities)
        return CheckerResult(
            value=entropy_value,
            message=f"Shannon entropy: {entropy_value}",
            success=True,
        )


@LinterRegistry.register()
class VariableTypesChecker(DataArrayChecker):
    """Report the data type (dtype) of the array.

    Returns the numpy dtype as a string (e.g., 'float64', 'int32', 'object').
    Useful for verifying expected data types and identifying type inconsistencies.
    """

    name = "Data type"
    description = "Data type of the variable"

    def check(self, var: xr.DataArray) -> CheckerResult:
        return CheckerResult(
            value=str(var.dtype),
            message=f"Data type: {var.dtype}",
            success=True,
        )


@LinterRegistry.register()
class UnitsChecker(DataArrayChecker):
    """Extract the units attribute from the variable's metadata.

    Returns the 'units' attribute if present, otherwise returns 'unknown'.
    Helps verify that variables have proper unit documentation for physical quantities.
    """

    name = "Units"
    description = "Units attribute"

    def check(self, var: xr.DataArray) -> CheckerResult:
        value = var.attrs.get("units", "unknown")
        success = value != "unknown"
        return CheckerResult(
            value=value,
            message=f"Units: {value}",
            success=success,
        )


@LinterRegistry.register()
class UnitsParsableChecker(DataArrayChecker):
    """Check if the units attribute can be parsed by pint_xarray.

    Returns True if the units can be successfully parsed by
    [pint](https://pint.readthedocs.io/en/stable/)'s unit registry, False if parsing fails or
    if no units attribute is present. This helps identify malformed or non-standard unit
    strings that may cause issues in unit-aware computations.
    """

    name = "Units parsable"
    description = "Whether units can be parsed by pint"

    def check(self, var: xr.DataArray) -> CheckerResult:
        units = var.attrs.get("units", None)
        if units is None or units == "unknown":
            return CheckerResult(
                value=False,
                message="No units attribute present.",
                success=False,
            )

        try:
            var.pint.quantify(units)
            return CheckerResult(
                value=True,
                message="Units are parsable.",
                success=True,
            )
        except (ValueError, TypeError, AttributeError):
            return CheckerResult(
                value=False,
                message="Units are not parsable.",
                success=False,
            )


@LinterRegistry.register()
class DiffChecker(DataArrayChecker):
    """Calculate the mean of first differences along the first dimension.

    Useful for detecting trends or average rate of change in sequential data.
    Positive values indicate increasing trends, negative values indicate decreasing trends.
    """

    name = "Diff"
    description = "Mean of first differences"

    def check(self, var: xr.DataArray) -> CheckerResult:
        return CheckerResult(
            value=var.diff(dim=list(var.dims)[0]).mean().item(),
            message=f"Mean of first differences: {var.diff(dim=list(var.dims)[0]).mean().item()}",
            success=True,
        )


@LinterRegistry.register()
class DiffConstantChecker(DataArrayChecker):
    """Check if first differences along the first dimension are constant.

    This checker only runs on coordinate dimensions (1-dimensional arrays where the
    dimension name matches the variable name). Returns True if all consecutive
    differences are identical, indicating a linear relationship with constant slope.
    Useful for identifying uniformly-spaced coordinate arrays.
    """

    name = "Diff constant"
    description = "Whether differences are constant (coordinates only)"

    def check(self, var: xr.DataArray) -> CheckerResult:
        # Only check if this is a coordinate dimension
        if not self.is_coord(var):
            return CheckerResult(
                value=False,
                message="Not a coordinate dimension; check skipped.",
                success=True,
            )

        diffs = var.diff(dim=var.dims[0])
        constant = np.unique(diffs).size == 1
        return CheckerResult(
            value=constant,
            message=f"Differences are {'constant' if constant else 'not constant'}.",
            success=constant,
        )
