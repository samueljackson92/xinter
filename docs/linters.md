# Available Linters

xinter includes 25+ built-in linters (checkers) that automatically assess data quality in your xarray datasets. Each linter examines a specific aspect of your data and returns a result with a value, message, and success status.

## Understanding Linter Results

Each linter returns a `LinterResult` with three components:

- **value**: The computed metric (e.g., proportion of NaNs, mean value)
- **message**: A human-readable description of the result
- **success**: Boolean indicating whether the check passed (some checks always pass, reporting informational metrics)

## Statistical Checkers

### Mean
**Name:** `mean`  
**Applies to:** Numeric arrays

Calculates the arithmetic mean of all values in the array. The mean provides a measure of central tendency and can help identify if values are centered around expected ranges.

**Example output:** `Mean value: 273.15`

---

### Standard Deviation
**Name:** `std`  
**Applies to:** Numeric arrays

Calculates the standard deviation of all values. Standard deviation measures the spread or dispersion of values around the mean. Low values indicate data points are close to the mean, while high values indicate more variability.

**Example output:** `Standard deviation: 12.34`

---

### Range
**Name:** `range`  
**Applies to:** Numeric arrays

Calculates the range of values (maximum minus minimum). Provides a simple measure of data spread. Large ranges may indicate high variability or the presence of outliers.

**Example output:** `Range of values: 100.5`

---

### Maximum
**Name:** `max`  
**Applies to:** Numeric arrays

Finds the maximum value in the array. Useful for identifying the upper bound of the data and detecting potentially unrealistic values that exceed expected limits.

**Example output:** `Maximum value: 350.2`

---

### Minimum
**Name:** `min`  
**Applies to:** Numeric arrays

Finds the minimum value in the array. Useful for identifying the lower bound and detecting potentially unrealistic values that fall below expected limits.

**Example output:** `Minimum value: -50.1`

---

### Skewness
**Name:** `skewness`  
**Applies to:** Numeric arrays

Calculates the skewness (third standardized moment) of the distribution. Skewness measures the asymmetry of the distribution:

- **0**: Symmetric (like normal distribution)
- **Positive**: Right-skewed (tail extends to the right)
- **Negative**: Left-skewed (tail extends to the left)

Returns 0 if standard deviation is 0 or sample size < 3.

**Example output:** `Skewness: 0.45`

---

### Kurtosis
**Name:** `kurtosis`  
**Applies to:** Numeric arrays

Calculates the excess kurtosis (fourth standardized moment) of the distribution. Kurtosis measures the "tailedness" of the distribution:

- **0**: Normal distribution (mesokurtic)
- **Positive**: Heavy tails, more outliers (leptokurtic)
- **Negative**: Light tails, fewer outliers (platykurtic)

Returns 0 if standard deviation is 0 or sample size < 4.

**Example output:** `Kurtosis: 1.2`

---

### Shannon Entropy
**Name:** `entropy`  
**Applies to:** Numeric arrays

Calculates the Shannon entropy of the value distribution. Entropy measures the randomness or unpredictability of the data:

- **Low entropy**: Values are concentrated in few states (more predictable)
- **High entropy**: Values are spread across many states (more random)

Useful for assessing data diversity and information content.

**Example output:** `Shannon entropy: 3.45`

## Data Quality Checkers

### NaN Percentage
**Name:** `nan_percent`  
**Applies to:** Numeric arrays  
**Success threshold:** ≤ 20%

Checks for the proportion of NaN (Not a Number) values. Returns a float between 0 and 1, where 0 means no NaNs and 1 means all values are NaN. High proportions may indicate data quality issues or missing measurements.

**Example output:** `5.00% NaNs found.`

---

### NaN Count
**Name:** `nan_count`  
**Applies to:** Numeric arrays

Returns the total count of NaN values in the array. Complementary to `nan_percent` for absolute counts.

**Example output:** `123 NaN values found.`

---

### Infinite Values Percentage
**Name:** `infinite_percent`  
**Applies to:** Numeric arrays  
**Success threshold:** = 0%

Calculates the proportion of infinite values (both +inf and -inf). Infinite values often result from division by zero or numerical overflow and typically indicate data quality or processing issues.

This check **fails** if any infinite values are found.

**Example output:** `Array contains infinite values: 0.01%`

---

### Infinite Values Count
**Name:** `infinite_count`  
**Applies to:** Numeric arrays  
**Success threshold:** = 0

Returns the total count of infinite values. This check **fails** if any infinite values are found.

**Example output:** `Total count of infinite values: 5`

---

### Duplicate Values
**Name:** `duplicate_values`  
**Applies to:** All arrays  
**Success threshold:** < 95%

Calculates the proportion of values that appear more than once. High proportions may indicate constant or repeated values, which could suggest sensor issues, data recording problems, or expected patterns.

**Example output:** `High proportion of duplicate values (> 95): 98.2%`

---

### Constant Values
**Name:** `constant_values`  
**Applies to:** All arrays

Checks if all values in the array are identical. Returns True if the array contains only one unique value. Constant arrays may indicate stuck sensors, configuration issues, or legitimately constant parameters.

This check **fails** if all values are constant.

**Example output:** `All values are constant: True`

---

### Constant Values Count
**Name:** `constant_values_count`  
**Applies to:** All arrays

Returns the count of the most common value in the array. High counts of a single value may indicate constant or repeated values.

**Example output:** `Count of the most common value: 1500`

## Outlier Detection

### IQR Outliers
**Name:** `iqr`  
**Applies to:** Numeric arrays

Detects outliers using the Interquartile Range (IQR) method. Values are considered outliers if they fall below Q1 - 1.5×IQR or above Q3 + 1.5×IQR, where Q1 and Q3 are the 25th and 75th percentiles.

Returns the proportion of values identified as outliers (0 to 1).

**Example output:** `Proportion of outliers: 0.03`

## Value-Specific Checkers

### Negative Values Percentage
**Name:** `negative_percent`  
**Applies to:** Numeric arrays

Calculates the proportion of negative values. Useful for identifying variables that should be strictly non-negative (e.g., distance, count, absolute measurements).

**Example output:** `Proportion of negative values: 5.2%`

---

### Negative Values Count
**Name:** `negative_count`  
**Applies to:** Numeric arrays

Returns the total count of negative values. Complementary to `negative_percent` for absolute counts.

**Example output:** `Total count of negative values: 42`

---

### Zero Values Percentage
**Name:** `zero_percent`  
**Applies to:** Numeric arrays  
**Success threshold:** < 95%

Calculates the proportion of exact zero values. High proportions may indicate missing data encoded as zeros, sparse data, or measurement periods with no activity.

**Example output:** `Proportion of zeros is greater than 95: 97.1%`

---

### Zero Values Count
**Name:** `zero_count`  
**Applies to:** Numeric arrays

Returns the total count of exact zero values.

**Example output:** `Total count of zero values: 234`

## Metadata & Structure Checkers

### Data Type
**Name:** `data_type`  
**Applies to:** All arrays

Reports the numpy data type (dtype) of the array (e.g., 'float64', 'int32', 'object'). Useful for verifying expected data types and identifying type inconsistencies.

**Example output:** `Data type: float64`

---

### Units
**Name:** `units`  
**Applies to:** All arrays  
**Success threshold:** Units present

Extracts the 'units' attribute from the variable's metadata. Returns the units if present, otherwise returns 'unknown'. This check **fails** if no units are found.

**Example output:** `Units: K` or `Units: unknown`

---

### Units Parsable
**Name:** `units_parsable`  
**Applies to:** All arrays  
**Success threshold:** Units can be parsed

Checks if the units attribute can be parsed by [pint_xarray](https://pint-xarray.readthedocs.io/). Returns True if units can be successfully parsed by pint's unit registry, False if parsing fails or if no units attribute is present.

This helps identify malformed or non-standard unit strings that may cause issues in unit-aware computations.

**Example output:** `Units are parsable.` or `Units are not parsable.`

---

### Shape
**Name:** `shape`  
**Applies to:** All arrays

Reports the shape of the array as a tuple. Useful for verifying expected dimensions and identifying shape inconsistencies.

**Example output:** `Shape: (100, 50, 25)`

---

### Size
**Name:** `size`  
**Applies to:** All arrays

Reports the total number of elements in the array. Useful for verifying expected data sizes and identifying empty or unusually large arrays.

**Example output:** `Size: 125000`

---

### Variable Name
**Name:** `variable_name`  
**Applies to:** All arrays

Reports the name of the variable. Useful for verifying expected variable names and identifying naming inconsistencies.

**Example output:** `Variable name: temperature`

---

### Dimension Names
**Name:** `dimension_names`  
**Applies to:** All arrays

Reports the names of the dimensions of the variable. Useful for verifying expected dimension names and identifying naming inconsistencies.

**Example output:** `Dimension names: ('time', 'lat', 'lon')`

## Temporal & Sequential Checkers

### Mean Difference
**Name:** `diff`  
**Applies to:** Numeric arrays

Calculates the mean of first differences along the first dimension. Useful for detecting trends or average rate of change in sequential data. Positive values indicate increasing trends, negative values indicate decreasing trends.

**Example output:** `Mean of first differences: 0.12`

---

### Constant Differences (Coordinates)
**Name:** `diff_constant`  
**Applies to:** Numeric coordinate arrays only  
**Success threshold:** Differences are constant

Checks if first differences along the first dimension are constant. This checker only runs on coordinate dimensions (1-dimensional arrays where the dimension name matches the variable name).

Returns True if all consecutive differences are identical, indicating a linear relationship with constant slope. Useful for identifying uniformly-spaced coordinate arrays.

**Example output:** `Differences are constant.` or `Differences are not constant.`

## Multi-dimensional Checkers

### Constant Along Dimension
**Name:** `constant_along_dimension`  
**Applies to:** Numeric arrays with 2+ dimensions

Checks if values are constant along any dimension. Useful for identifying problematic data where entire slices contain the same value.

**Example output:** `Values are constant along dimension 'time'.` or `Values are not constant along any dimension.`

---

### NaN Along Dimension
**Name:** `nan_along_dimension`  
**Applies to:** Numeric arrays with 2+ dimensions

Checks if values are entirely NaN along any dimension. Helps identify complete data loss along specific axes.

**Example output:** `Values are NaN along dimension 'lat'.` or `Values are not NaN along any dimension.`

## Custom Checkers

You can create custom checkers using the `@LinterRegistry.register()` decorator:

```python
from xinter.linters import LinterRegistry, DataArrayChecker, LinterResult
import xarray as xr

@LinterRegistry.register()
class MyCustomChecker(DataArrayChecker):
    """Check for values within a specific range."""
    
    name = "in_range"
    description = "Values within expected range"
    
    def check(self, var: xr.DataArray) -> LinterResult:
        min_val, max_val = 0, 100
        in_range = ((var >= min_val) & (var <= max_val)).all().item()
        
        return LinterResult(
            value=in_range,
            message=f"All values in range [{min_val}, {max_val}]: {in_range}",
            success=in_range
        )
```

Once registered, your custom checker will automatically run on all applicable variables when you use `lint_dataset()`.

## Checker Categories Summary

| Category | Checkers |
|----------|----------|
| **Statistical** | mean, std, range, max, min, skewness, kurtosis, entropy |
| **Data Quality** | nan_percent, nan_count, infinite_percent, infinite_count, duplicate_values, constant_values |
| **Outliers** | iqr |
| **Value-Specific** | negative_percent, negative_count, zero_percent, zero_count |
| **Metadata** | data_type, units, units_parsable, shape, size, variable_name, dimension_names |
| **Sequential** | diff, diff_constant |
| **Multi-dimensional** | constant_along_dimension, nan_along_dimension |
