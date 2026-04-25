# Xinter Documentation

Welcome to **xinter** - a comprehensive linting and data quality checking tool for xarray datasets.

## Overview

xinter provides automated data quality checks for xarray datasets, helping you identify issues like missing values, outliers, incorrect units, and other data anomalies. It features an extensible architecture that allows you to easily add custom checkers for your specific data validation needs.

## Features

:material-check-circle: **25+ Built-in Checkers** - Comprehensive checks for data quality  
:material-puzzle: **Extensible Architecture** - Easily add custom checkers  
:material-console: **Rich CLI Output** - Beautiful terminal output with tables  
:material-chart-line: **Interactive Dashboard** - Web-based GUI for exploring results  
:material-file-export: **DataFrame Export** - Convert results to pandas DataFrames  
:material-coordinate-map: **Coordinate Checking** - Check coordinate arrays in addition to data variables  
:material-group: **Group Support** - Handle datasets with hierarchical groups

## Installation

Install xinter using pip:

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

Use xinter programmatically in your Python code:

```python
from xinter.core import lint_dataset
from xinter.cli import reports_to_dataframe

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

### Basic Usage Example

Here's a complete example showing how to lint a dataset and analyze the results:

```python
import xarray as xr
from xinter.core import lint_dataset
from xinter.cli import reports_to_dataframe

# Load your dataset
ds = xr.open_dataset("mydata.nc")

# Run linting
reports = lint_dataset("mydata.nc", check_coords=False)

# Convert to DataFrame
df = reports_to_dataframe(reports)

# Show statistics
print(f"Total checks: {len(df)}")
print(f"Passed: {df['success'].sum()}")
print(f"Failed: {(~df['success']).sum()}")

# View failed checks
print("\nFailed checks:")
print(df[~df["success"]][["variable_name", "checker", "message"]])
```

## Output Format

The linting results are returned as a dictionary where each key is a variable name and each value is another dictionary of checker results. You can easily convert this to a pandas DataFrame for further analysis:

```python
# Example output structure
{
    'temperature': {
        'nan_percent': LinterResult(value=0.05, message="5.00% NaNs found.", success=True),
        'mean': LinterResult(value=273.15, message="Mean value: 273.15", success=True),
        # ... more checks
    },
    'pressure': {
        # ... checks for pressure variable
    }
}
```

## CLI Options

The `xl` command supports various options:

| Option | Description |
|--------|-------------|
| `--coords` | Check coordinate arrays in addition to data variables |
| `--group <path>` | Specify a group within the dataset (e.g., `/equilibrium`) |
| `--output <file>` | Save results to a parquet file for use with the GUI |
| `-h, --help` | Show help message |

## What Gets Checked?

By default, xinter runs all registered checkers on every data variable in your dataset:

- **Statistical properties**: mean, standard deviation, min, max, range
- **Data quality**: NaN values, infinite values, duplicates, constants
- **Distribution metrics**: skewness, kurtosis, entropy
- **Outlier detection**: IQR-based outlier proportion
- **Type validation**: data types, shape, size
- **Metadata checks**: units, dimension names
- **Coordinate checks**: uniformity, constant spacing (when `--coords` is used)

See the [Available Linters](linters.md) page for a complete list and detailed descriptions.

## Next Steps

- Explore all [Available Linters](linters.md) and what they check for
- Learn about the interactive [GUI Dashboard](gui.md) for visualizing results
- Create custom checkers for your specific validation needs

## Support

For issues, questions, or contributions, visit the [GitHub repository](https://github.com/samueljackson92/xinter).
