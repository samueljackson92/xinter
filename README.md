# xinter

A comprehensive linting and data quality checking tool for xarray datasets.

## Overview

xinter provides automated data quality checks for xarray datasets, helping you identify issues like missing values, outliers, incorrect units, and other data anomalies. It features an extensible architecture that allows you to easily add custom checkers for your specific data validation needs.

## Features

- **20+ Built-in Checkers**: Comprehensive checks for data quality including:
  - Missing values (NaNs)
  - Statistical properties (mean, std, skewness, kurtosis)
  - Outlier detection (IQR method)
  - Data type validation
  - Units verification and parsing
  - Coordinate uniformity checks
  - And many more...

- **Extensible Architecture**: Easily add custom checkers using a simple decorator pattern
- **Rich CLI Output**: Beautiful terminal output with tables showing results
- **DataFrame Export**: Convert results to pandas DataFrames for further analysis
- **Coordinate Checking**: Optionally check coordinate arrays in addition to data variables
- **Group Support**: Handle datasets with hierarchical groups (e.g., Zarr, NetCDF4)

## Installation

```bash
pip install xinter
```

Or install from source:

```bash
git clone https://github.com/samueljackson92/xinter.git
cd xinter
pip install -e .
```

## Quick Start

### Command Line Interface

Lint a single file:

```bash
xl mydata.zarr
```

Lint multiple files:

```bash
xl file1.nc file2.zarr file3.nc
```

Check coordinates in addition to data variables:

```bash
xl mydata.zarr --coords
```

Specify a group within the dataset:

```bash
xl mydata.zarr --group=/equilibrium
```

### Python API

```python
from xinter.cli import lint_dataset, reports_to_dataframe

# Lint a dataset
reports = lint_dataset("mydata.zarr", check_coords=True)

# Convert to DataFrame for analysis
df = reports_to_dataframe(reports)

# Filter for failed checks
failures = df[~df["success"]]
print(failures)

# Export to CSV
df.to_csv("lint_report.csv", index=False)
```

## Built-in Checkers

| Checker | Description |
|---------|-------------|
| **NaNs** | Proportion of NaN values |
| **Mean** | Arithmetic mean |
| **Standard deviation** | Data spread measure |
| **IQR outliers** | Outliers using interquartile range method |
| **Range** | Maximum minus minimum |
| **Max** | Maximum value |
| **Min** | Minimum value |
| **Duplicate values** | Proportion of repeated values |
| **Negative values** | Proportion of negative values |
| **Zero values** | Proportion of zero values |
| **Constant values** | Whether all values are identical |
| **Infinite values** | Proportion of infinite values |
| **Skewness** | Distribution asymmetry measure |
| **Kurtosis** | Distribution tailedness measure |
| **Zero inflation** | Zero-inflation metric |
| **Entropy** | Shannon entropy of distribution |
| **Data type** | Variable dtype |
| **Units** | Units attribute |
| **Units parsable** | Whether units can be parsed by pint |
| **Diff** | Mean of first differences |
| **Diff constant** | Whether differences are constant (coordinates only) |

## Creating Custom Checkers

You can easily extend xinter with custom checkers:

```python
from xinter.linters import DataArrayChecker, LinterRegistry, CheckerResult
import xarray as xr

@LinterRegistry.register()
class MyCustomChecker(DataArrayChecker):
    """Check if values are within expected range."""
    
    name = "Value range check"
    description = "Checks if values fall within [0, 100]"
    
    def check(self, var: xr.DataArray) -> CheckerResult:
        min_val = var.min().item()
        max_val = var.max().item()
        in_range = 0 <= min_val and max_val <= 100
        
        return CheckerResult(
            value=in_range,
            message=f"Range: [{min_val}, {max_val}]",
            success=in_range,
        )
```

Your custom checker will automatically be included in all linting operations.

## Output Format

The `reports_to_dataframe()` function produces a DataFrame with the following columns:

- **file_path**: Path to the dataset file
- **target_type**: Either "data_vars" or "coords"
- **variable_name**: Name of the variable
- **checker_name**: Name of the checker
- **value**: The check result value
- **message**: Descriptive message about the result
- **success**: Boolean indicating if the check passed

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/xinter.git
cd xinter

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .

# Lint code
ruff check .
```

## Requirements

- Python >= 3.11
- xarray >= 2024.7.0
- pandas >= 2.3.3
- pint-xarray >= 0.6.0
- pydantic >= 2.12.5
- rich >= 14.2.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Samuel Jackson (samuel.jackson@ukaea.uk)

## Acknowledgments

xinter builds on the excellent work of the xarray, pandas, and pint communities.
