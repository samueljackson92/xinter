# XR-Linter Dashboard GUI

A modern, minimalist web-based dashboard for exploring and visualizing xarray dataset linting results.

## Features

- **Interactive Visualizations**: Beautiful charts and graphs using Plotly
- **Modern Design**: Clean, minimal interface with Inter font and carefully chosen color palette
- **Real-time Filtering**: Filter data by file and group
- **Comprehensive Analytics**:
  - Data quality overview (NaN distribution)
  - Data type distribution
  - Statistical distributions (mean, std, min, max, entropy)
  - Entropy analysis across variables
  - Kurtosis vs Skewness scatter plots
  - Detailed data table view

## Installation

Install the GUI dependencies:

```bash
pip install -e ".[gui]"
```

This will install:
- `dash` - Web application framework
- `dash-bootstrap-components` - Bootstrap components for Dash
- `plotly` - Interactive plotting library

## Usage

### Command Line

After installation, you can launch the dashboard using:

```bash
xl-gui path/to/linting_report.parquet
```

Or directly with Python:

```bash
python -m xinter.gui path/to/linting_report.parquet
```

### Options

```bash
xl-gui <parquet_file> [OPTIONS]

Options:
  --port PORT        Port to run the dashboard on (default: 8050)
  --host HOST        Host to run the dashboard on (default: 127.0.0.1)
  --title TITLE      Custom title for the dashboard
  --debug            Enable debug mode
```

### Example

```bash
# Launch dashboard on default port (8050)
xl-gui linting_report.parquet

# Launch on custom port
xl-gui linting_report.parquet --port 8080

# With custom title
xl-gui thomson_scattering.parquet --title "Thomson Scattering Analysis"

# Enable debug mode (useful for development)
xl-gui linting_report.parquet --debug
```

## Programmatic Usage

You can also use the GUI module in your own Python scripts:

```python
from xinter.gui import launch_dashboard

# Launch dashboard
launch_dashboard(
    parquet_file="linting_report.parquet",
    title="My Custom Dashboard",
    port=8050,
    host="127.0.0.1",
    debug=False
)
```

Or create a custom dashboard:

```python
from xinter.gui import load_parquet_file, create_dashboard

# Load data
df = load_parquet_file("linting_report.parquet")

# Create dashboard app
app = create_dashboard(df, title="Custom Dashboard")

# Run with custom settings
app.run(host="0.0.0.0", port=8080, debug=True)
```

## Dashboard Sections

### Summary Metrics
- **Total Variables**: Count of all analyzed variables
- **Files Analyzed**: Number of unique files in the report
- **Groups**: Number of unique groups
- **Avg NaN %**: Average percentage of NaN values across all variables

### Filters
- **File Filter**: Filter results by source file
- **Group Filter**: Filter results by data group

### Visualizations

1. **Data Quality Overview**: Histogram showing distribution of NaN percentages
2. **Data Types Distribution**: Pie chart showing breakdown of data types
3. **Statistical Distribution**: Box plots for key metrics (mean, std, min, max, entropy)
4. **Entropy Analysis**: Scatter plot showing entropy values across variables
5. **Kurtosis vs Skewness**: Scatter plot for distribution shape analysis
6. **Detailed Data View**: Interactive table with key metrics (limited to 100 rows)

## Design

The dashboard features:
- **Modern Typography**: Inter font family with carefully tuned weights and spacing
- **Minimal Color Palette**: Professional blue and red accents on white background
- **Responsive Layout**: Works on different screen sizes
- **Clean Charts**: Plotly visualizations with consistent styling
- **Smooth Interactions**: Fast filtering and updates

## Technical Details

- Built with Dash (Plotly's Python framework)
- Uses Bootstrap for responsive grid layout
- All rendering is Python-based (no JavaScript required)
- Data filtering handled client-side for fast updates
- Supports large datasets through pagination and sampling

## Browser Support

The dashboard works in all modern browsers:
- Chrome/Edge (recommended)
- Firefox
- Safari

## Tips

- Use the file and group filters to focus on specific subsets of data
- Hover over chart elements for detailed tooltips
- The data table shows a maximum of 100 rows for performance
- Color coding in charts helps distinguish between data_vars and coords

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'dash'`, install GUI dependencies:

```bash
pip install -e ".[gui]"
```

### Port Already in Use

If port 8050 is busy, specify a different port:

```bash
xl-gui linting_report.parquet --port 8888
```

### Dashboard Not Loading

1. Check that the parquet file exists and is readable
2. Verify the file contains the expected columns
3. Try running with `--debug` flag for more information

## Development

To contribute to the GUI:

1. Install in development mode with GUI extras:
   ```bash
   pip install -e ".[gui,dev]"
   ```

2. Make changes to `xinter/gui.py`

3. Test your changes:
   ```bash
   python -m xinter.gui linting_report.parquet --debug
   ```

4. The dashboard will auto-reload when files change (in debug mode)
