# Quick Start Guide for XR-Linter Dashboard

## Installation

1. **Install the package with GUI dependencies:**

```bash
cd /Users/rt2549/projects/xr-linter
pip install -e ".[gui]"
```

This installs:
- dash (web framework)
- dash-bootstrap-components (UI components)
- plotly (interactive charts)

## Quick Test

You already have example parquet files! Try the dashboard now:

```bash
# Using the command (after installation)
xl-gui linting_report.parquet

# Or directly with Python
python -m xinter.gui linting_report.parquet

# Try the thomson_scattering example
xl-gui thomson_scattering.parquet --port 8080
```

## What You'll See

The dashboard opens at `http://localhost:8050` and shows:

1. **Summary Metrics** at the top:
   - Total Variables
   - Files Analyzed
   - Groups
   - Average NaN %

2. **Interactive Filters**:
   - Filter by File
   - Filter by Group

3. **Visualizations**:
   - Data Quality Overview (NaN distribution histogram)
   - Data Types Distribution (pie chart)
   - Statistical Distribution (box plots)
   - Entropy Analysis (scatter plot)
   - Kurtosis vs Skewness (scatter plot)
   - Detailed Data Table (first 100 rows)

## Customization

### Change Port

```bash
xl-gui linting_report.parquet --port 8888
```

### Custom Title

```bash
xl-gui thomson_scattering.parquet --title "My Analysis"
```

### Debug Mode

```bash
xl-gui linting_report.parquet --debug
```

## Programmatic Usage

```python
from xinter.gui import launch_dashboard

launch_dashboard(
    parquet_file="linting_report.parquet",
    title="Custom Dashboard",
    port=8050,
    debug=False
)
```

## Design Features

- **Modern Typography**: Inter font family from Google Fonts
- **Color Palette**: 
  - Primary: #2D3561 (dark blue)
  - Secondary: #5B7C99 (blue-grey)
  - Accent: #FF6B6B (coral red)
  - Background: #F8F9FA (light grey)
- **Clean Layout**: Card-based design with subtle shadows
- **Responsive**: Works on different screen sizes

## Examples

Run the example script to see different usage patterns:

```bash
python example_gui.py
```

Choose from:
1. Basic usage
2. Custom settings
3. Programmatic control
4. Multiple files combined

## Troubleshooting

### If you get "ModuleNotFoundError: No module named 'dash'"

```bash
pip install -e ".[gui]"
```

### If port 8050 is already in use

```bash
xl-gui linting_report.parquet --port 8888
```

### If the dashboard doesn't load

1. Check your parquet file exists
2. Verify it has the expected columns
3. Try with `--debug` flag for more info

## Next Steps

- Read [GUI_README.md](GUI_README.md) for complete documentation
- Check [example_gui.py](example_gui.py) for code examples
- Modify [xinter/gui.py](xinter/gui.py) to customize the dashboard

## Performance Tips

- The dashboard filters data client-side for fast interactions
- Tables are limited to 100 rows for performance
- Large datasets (>10,000 variables) work fine but may take a moment to load initially

Enjoy exploring your linting data! 🎉
