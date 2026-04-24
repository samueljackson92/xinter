"""
Modern web-based dashboard for visualizing xinter linting results.

This module provides an interactive dashboard for exploring and analyzing
xarray dataset linting reports stored in Parquet format.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Output, Input, dash_table
import dash_bootstrap_components as dbc
from pathlib import Path
from typing import Optional


# Modern, minimal color palette
COLORS = {
    "primary": "#2D3561",
    "secondary": "#5B7C99",
    "accent": "#FF6B6B",
    "success": "#51CF66",
    "warning": "#FFD93D",
    "background": "#F8F9FA",
    "surface": "#FFFFFF",
    "text": "#212529",
    "text_secondary": "#6C757D",
    "border": "#DEE2E6",
}


def load_parquet_file(file_path: str) -> pd.DataFrame:
    """Load a linting report from a Parquet file."""
    return pd.read_parquet(file_path)


def create_dashboard(df: pd.DataFrame, title: str = "XR-Linter Dashboard") -> Dash:
    """Create the Dash application with the data dashboard.

    Args:
        df: DataFrame containing linting results
        title: Title for the dashboard

    Returns:
        Dash application instance
    """
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.title = title

    # Inject custom CSS for modern typography and styling
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                body {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background-color: #F8F9FA;
                    color: #212529;
                }
                
                h1, h2, h3, h4, h5, h6 {
                    font-weight: 600;
                    letter-spacing: -0.02em;
                }
                
                .metric-card {
                    background: white;
                    border-radius: 12px;
                    padding: 24px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    border: 1px solid #DEE2E6;
                    height: 100%;
                }
                
                .metric-value {
                    font-size: 2.5rem;
                    font-weight: 700;
                    color: #2D3561;
                    line-height: 1;
                }
                
                .metric-label {
                    font-size: 0.875rem;
                    color: #6C757D;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    margin-top: 8px;
                    font-weight: 500;
                }
                
                .chart-container {
                    background: white;
                    border-radius: 12px;
                    padding: 24px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    border: 1px solid #DEE2E6;
                    margin-bottom: 24px;
                }
                
                .header {
                    background: white;
                    border-bottom: 1px solid #DEE2E6;
                    padding: 32px 0;
                    margin-bottom: 32px;
                }
                
                .header h1 {
                    font-size: 2rem;
                    margin: 0;
                    color: #2D3561;
                }
                
                .header p {
                    color: #6C757D;
                    margin: 8px 0 0 0;
                }
                
                .Select-control {
                    border-radius: 8px !important;
                }
                
                .table-container {
                    max-height: 600px;
                    overflow-y: auto;
                    border-radius: 8px;
                }
                
                table {
                    font-size: 0.875rem;
                }
                
                th {
                    background-color: #F8F9FA !important;
                    font-weight: 600 !important;
                    color: #2D3561 !important;
                    text-transform: uppercase;
                    font-size: 0.75rem;
                    letter-spacing: 0.05em;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """

    # Calculate summary statistics
    total_variables = len(df)
    unique_variable_names = (
        df["variable_name"].nunique() if "variable_name" in df.columns else 0
    )

    # Handle file_path from either column or index
    unique_files = df["file_path"].nunique()

    # Count unique (file_path, group) combinations from index or columns
    unique_groups = df["group"].nunique()

    # Get list of numeric columns for scatter plot axes
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Get all columns for color options
    all_cols = df.columns.tolist()
    # Get list of unique variable names for stats dropdown
    variable_names = sorted(df["variable_name"].unique().tolist()) if "variable_name" in df.columns else []

    # Layout
    app.layout = dbc.Container(
        [
            # Header
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H1(title),
                                    html.P(
                                        "Interactive visualization of xarray dataset linting results"
                                    ),
                                ]
                            )
                        ]
                    )
                ],
                className="header",
            ),
            # Summary Metrics
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        f"{total_variables:,}", className="metric-value"
                                    ),
                                    html.Div(
                                        "Total Datasets", className="metric-label"
                                    ),
                                ],
                                className="metric-card",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        f"{unique_variable_names:,}",
                                        className="metric-value",
                                    ),
                                    html.Div(
                                        "Total Variables", className="metric-label"
                                    ),
                                ],
                                className="metric-card",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        f"{unique_files:,}", className="metric-value"
                                    ),
                                    html.Div("No. Files", className="metric-label"),
                                ],
                                className="metric-card",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        f"{unique_groups:,}", className="metric-value"
                                    ),
                                    html.Div("Groups", className="metric-label"),
                                ],
                                className="metric-card",
                            )
                        ],
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # Filters
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Filter by File",
                                        style={
                                            "fontWeight": "600",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="file-filter",
                                        options=[{"label": "All Files", "value": "all"}]
                                        + [
                                            {"label": f, "value": f}
                                            for f in df["file_path"].unique()
                                        ]
                                        if "file_path" in df.columns
                                        else [],
                                        value="all",
                                        clearable=False,
                                        style={"borderRadius": "8px"},
                                    ),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Filter by Group",
                                        style={
                                            "fontWeight": "600",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="group-filter",
                                        options=[
                                            {"label": "All Groups", "value": "all"}
                                        ]
                                        + [
                                            {"label": g, "value": g}
                                            for g in df["group"].unique()
                                            if pd.notna(g)
                                        ]
                                        if "group" in df.columns
                                        else [],
                                        value="all",
                                        clearable=False,
                                        style={"borderRadius": "8px"},
                                    ),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Filter by Variable Name",
                                        style={
                                            "fontWeight": "600",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    dcc.Input(
                                        id="variable-filter",
                                        type="text",
                                        placeholder="Type to filter variables...",
                                        value="",
                                        style={
                                            "width": "100%",
                                            "borderRadius": "8px",
                                            "border": "1px solid #ced4da",
                                            "padding": "8px 12px",
                                        },
                                    ),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            # Charts Row 1 - Data Health
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5(
                                        "Data Health",
                                        style={
                                            "marginBottom": "20px",
                                            "marginTop": "20px",
                                            "color": COLORS["primary"],
                                        },
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Checklist(
                                        id="log-scale-toggle",
                                        options=[
                                            {
                                                "label": " Log Y-Axis",
                                                "value": "log",
                                            }
                                        ],
                                        value=[],
                                        inline=True,
                                        style={
                                            "marginTop": "25px",
                                            "fontSize": "0.9rem",
                                        },
                                    ),
                                ],
                                width=6,
                            ),
                        ]
                    )
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "NaN Values",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Graph(id="nan-distribution"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Infinite Values",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Graph(id="inf-distribution"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Zero Values",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Graph(id="zero-distribution"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Negative Values",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Graph(id="negative-distribution"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=3,
                    ),
                ]
            ),
            # Data Health Row 2
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Data Types",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Graph(id="data-types"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Units",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Graph(id="units-pie"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Units Parsable",
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Graph(id="units-parsable-pie"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=4,
                    ),
                ]
            ),
            # Charts Row 2
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H5(
                                                        "Statistical Distribution",
                                                        style={
                                                            "marginBottom": "20px",
                                                            "color": COLORS["primary"],
                                                        },
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Variable Name",
                                                        style={
                                                            "fontWeight": "600",
                                                            "marginBottom": "8px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="stats-variable",
                                                        options=[
                                                            {"label": v, "value": v}
                                                            for v in variable_names
                                                        ],
                                                        value=variable_names[0] if variable_names else None,
                                                        clearable=False,
                                                        style={"marginBottom": "10px"},
                                                    ),
                                                ],
                                                width=6,
                                            ),
                                        ]
                                    ),
                                    dcc.Graph(id="stats-overview"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=12,
                    ),
                ]
            ),
            # Charts Row 3 - Scatter Plot Explorer
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        "Scatter Plot Explorer",
                                        style={
                                            "marginBottom": "20px",
                                            "color": COLORS["primary"],
                                        },
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Variable Name",
                                                        style={
                                                            "fontWeight": "600",
                                                            "marginBottom": "8px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="scatter-variable",
                                                        options=[
                                                            {"label": "All", "value": "all"}
                                                        ]
                                                        + [
                                                            {"label": v, "value": v}
                                                            for v in variable_names
                                                        ],
                                                        value="all",
                                                        clearable=False,
                                                        style={"marginBottom": "10px"},
                                                    ),
                                                ],
                                                width=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "X-Axis",
                                                        style={
                                                            "fontWeight": "600",
                                                            "marginBottom": "8px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="scatter-x",
                                                        options=[
                                                            {
                                                                "label": "Index",
                                                                "value": "__index__",
                                                            }
                                                        ]
                                                        + [
                                                            {"label": col, "value": col}
                                                            for col in numeric_cols
                                                        ],
                                                        value="__index__",
                                                        clearable=False,
                                                        style={"marginBottom": "10px"},
                                                    ),
                                                ],
                                                width=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Y-Axis",
                                                        style={
                                                            "fontWeight": "600",
                                                            "marginBottom": "8px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="scatter-y",
                                                        options=[
                                                            {"label": col, "value": col}
                                                            for col in numeric_cols
                                                        ],
                                                        value="mean"
                                                        if "mean" in numeric_cols
                                                        else (
                                                            numeric_cols[1]
                                                            if len(numeric_cols) > 1
                                                            else numeric_cols[0]
                                                            if numeric_cols
                                                            else None
                                                        ),
                                                        clearable=False,
                                                        style={"marginBottom": "10px"},
                                                    ),
                                                ],
                                                width=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Color By",
                                                        style={
                                                            "fontWeight": "600",
                                                            "marginBottom": "8px",
                                                        },
                                                    ),
                                                    dcc.Dropdown(
                                                        id="scatter-color",
                                                        options=[
                                                            {
                                                                "label": "None",
                                                                "value": "none",
                                                            }
                                                        ]
                                                        + [
                                                            {"label": col, "value": col}
                                                            for col in all_cols
                                                        ],
                                                        value="target_type"
                                                        if "target_type" in all_cols
                                                        else "none",
                                                        clearable=False,
                                                        style={"marginBottom": "10px"},
                                                    ),
                                                ],
                                                width=3,
                                            ),
                                        ]
                                    ),
                                    dcc.Graph(id="scatter-plot"),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=12,
                    ),
                ]
            ),
            # Data Table
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.H5(
                                        "Detailed Data View",
                                        style={
                                            "marginBottom": "20px",
                                            "color": COLORS["primary"],
                                        },
                                    ),
                                    html.Div(
                                        id="data-table", className="table-container"
                                    ),
                                ],
                                className="chart-container",
                            )
                        ],
                        width=12,
                    ),
                ]
            ),
        ],
        fluid=True,
        style={"padding": "0 48px"},
    )

    # Helper function to apply filters
    def get_filtered_df(file_val, group_val, variable_val=""):
        """Apply filters to the dataframe based on filter values."""
        filtered_df = df.copy()

        if file_val != "all" and "file_path" in df.columns:
            filtered_df = filtered_df[filtered_df["file_path"] == file_val]

        if group_val != "all" and "group" in df.columns:
            filtered_df = filtered_df[filtered_df["group"] == group_val]

        if variable_val and "variable_name" in df.columns:
            filtered_df = filtered_df[
                filtered_df["variable_name"].str.contains(
                    variable_val, case=False, na=False
                )
            ]

        return filtered_df

    # Callbacks - all now take filter values directly instead of serialized data
    @app.callback(
        Output("nan-distribution", "figure"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
        Input("log-scale-toggle", "value"),
    )
    def update_nan_dist(file_val, group_val, variable_val, log_scale):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if "nan_percent" not in dff.columns or len(dff) == 0:
            return go.Figure()

        fig = px.histogram(
            dff,
            x="nan_percent",
            nbins=max(10, min(30, len(dff) // 10)) if len(dff) > 0 else 10,
            labels={"nan_percent": "NaN %"},
            color_discrete_sequence=[COLORS["primary"]],
        )

        yaxis_config = dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            title="Count",
            type="log" if log_scale and "log" in log_scale else "linear",
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(
                showgrid=True, gridcolor=COLORS["border"], title="NaN %", range=[0, 1]
            ),
            yaxis=yaxis_config,
            height=250,
        )
        return fig

    @app.callback(
        Output("inf-distribution", "figure"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
        Input("log-scale-toggle", "value"),
    )
    def update_inf_dist(file_val, group_val, variable_val, log_scale):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if "infinite_percent" not in dff.columns or len(dff) == 0:
            return go.Figure()

        fig = px.histogram(
            dff,
            x="infinite_percent",
            nbins=max(10, min(30, len(dff) // 10)) if len(dff) > 0 else 10,
            labels={"infinite_percent": "Inf %"},
            color_discrete_sequence=[COLORS["accent"]],
        )

        yaxis_config = dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            title="Count",
            type="log" if log_scale and "log" in log_scale else "linear",
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(
                showgrid=True, gridcolor=COLORS["border"], title="Inf %", range=[0, 1]
            ),
            yaxis=yaxis_config,
            height=250,
        )
        return fig

    @app.callback(
        Output("zero-distribution", "figure"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
        Input("log-scale-toggle", "value"),
    )
    def update_zero_dist(file_val, group_val, variable_val, log_scale):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if "zero_percent" not in dff.columns or len(dff) == 0:
            return go.Figure()

        fig = px.histogram(
            dff,
            x="zero_percent",
            nbins=max(10, min(30, len(dff) // 10)) if len(dff) > 0 else 10,
            labels={"zero_percent": "Zero %"},
            color_discrete_sequence=[COLORS["secondary"]],
        )

        yaxis_config = dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            title="Count",
            type="log" if log_scale and "log" in log_scale else "linear",
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(
                showgrid=True, gridcolor=COLORS["border"], title="Zero %", range=[0, 1]
            ),
            yaxis=yaxis_config,
            height=250,
        )
        return fig

    @app.callback(
        Output("negative-distribution", "figure"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
        Input("log-scale-toggle", "value"),
    )
    def update_negative_dist(file_val, group_val, variable_val, log_scale):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if "negative_percent" not in dff.columns or len(dff) == 0:
            return go.Figure()

        fig = px.histogram(
            dff,
            x="negative_percent",
            nbins=max(10, min(30, len(dff) // 10)) if len(dff) > 0 else 10,
            labels={"negative_percent": "Negative %"},
            color_discrete_sequence=[COLORS["warning"]],
        )

        yaxis_config = dict(
            showgrid=True,
            gridcolor=COLORS["border"],
            title="Count",
            type="log" if log_scale and "log" in log_scale else "linear",
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(
                showgrid=True,
                gridcolor=COLORS["border"],
                title="Negative %",
                range=[0, 1],
            ),
            yaxis=yaxis_config,
            height=250,
        )
        return fig

    @app.callback(
        Output("units-pie", "figure"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
    )
    def update_units_pie(file_val, group_val, variable_val):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if "units" not in dff.columns or len(dff) == 0:
            return go.Figure()

        # Count units, handling NaN/None as "No Units"
        units_counts = dff["units"].fillna("No Units").value_counts()

        # Limit to top 10 for readability
        if len(units_counts) > 10:
            units_counts = units_counts.head(10)

        fig = px.pie(
            values=units_counts.values,
            names=units_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            margin=dict(l=10, r=10, t=10, b=10),
            height=250,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
            ),
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    @app.callback(
        Output("units-parsable-pie", "figure"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
    )
    def update_units_parsable_pie(file_val, group_val, variable_val):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if "units_parsable" not in dff.columns or len(dff) == 0:
            return go.Figure()

        parsable_counts = dff["units_parsable"].value_counts()

        fig = px.pie(
            values=parsable_counts.values,
            names=parsable_counts.index,
            color_discrete_map={
                True: COLORS["success"],
                False: COLORS["accent"],
                "True": COLORS["success"],
                "False": COLORS["accent"],
            },
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            margin=dict(l=10, r=10, t=10, b=10),
            height=250,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
            ),
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    @app.callback(
        Output("data-types", "figure"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
    )
    def update_data_types(file_val, group_val, variable_val):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if "data_type" not in dff.columns or len(dff) == 0:
            return go.Figure()

        type_counts = dff["data_type"].value_counts()

        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            margin=dict(l=10, r=10, t=10, b=30),
            showlegend=True,
            height=250,
        )
        return fig

    @app.callback(
        Output("stats-overview", "figure"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
        Input("stats-variable", "value"),
    )
    def update_stats(file_val, group_val, variable_val, selected_variable):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if len(dff) == 0 or not selected_variable:
            return go.Figure()

        # Filter to selected variable_name
        if "variable_name" in dff.columns:
            dff = dff[dff["variable_name"] == selected_variable]
        
        if len(dff) == 0:
            return go.Figure()

        # Calculate range if min and max are available
        if "min" in dff.columns and "max" in dff.columns:
            dff = dff.copy()
            dff["range"] = dff["max"] - dff["min"]

        # Check if we have the group column
        if "group" not in dff.columns:
            return go.Figure()

        # Define all metrics to show
        all_metrics = ["mean", "std", "min", "max", "range", "skewness", "kurtosis", "entropy"]
        available_metrics = [m for m in all_metrics if m in dff.columns]

        if not available_metrics:
            return go.Figure()

        # Create subplots with separate y-axes for each metric
        num_metrics = len(available_metrics)
        fig = make_subplots(
            rows=1,
            cols=num_metrics,
            subplot_titles=[m.capitalize() for m in available_metrics],
            horizontal_spacing=0.05,
        )

        for i, metric in enumerate(available_metrics, start=1):
            fig.add_trace(
                go.Box(
                    y=dff[metric],
                    name=metric.capitalize(),
                    marker_color=COLORS["secondary"],
                    showlegend=False,
                ),
                row=1,
                col=i,
            )
            # Update individual y-axis
            fig.update_yaxes(
                showgrid=True,
                gridcolor=COLORS["border"],
                row=1,
                col=i,
            )
            fig.update_xaxes(
                showticklabels=False,
                row=1,
                col=i,
            )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            margin=dict(l=0, r=0, t=40, b=20),
            height=400,
            title_text=f"Statistics for {selected_variable}",
            title_x=0.5,
        )
        return fig

    @app.callback(
        Output("scatter-plot", "figure"),
        Input("scatter-variable", "value"),
        Input("scatter-x", "value"),
        Input("scatter-y", "value"),
        Input("scatter-color", "value"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
    )
    def update_scatter(scatter_var, x_col, y_col, color_col, file_val, group_val, variable_val):
        dff = get_filtered_df(file_val, group_val, variable_val)
        
        # Apply variable name filter from scatter plot dropdown
        if scatter_var and scatter_var != "all" and "variable_name" in dff.columns:
            dff = dff[dff["variable_name"] == scatter_var]

        if not x_col or not y_col or len(dff) == 0:
            return go.Figure()

        # Handle index as x-axis
        if x_col == "__index__":
            x_data = list(range(len(dff)))
            x_label = "Index"
        elif x_col not in dff.columns:
            return go.Figure()
        else:
            x_data = dff[x_col]
            x_label = x_col.replace("_", " ").title()

        # Handle y-axis
        if y_col not in dff.columns:
            return go.Figure()
        y_data = dff[y_col]
        y_label = y_col.replace("_", " ").title()

        # Build hover data to show file, group, and variable name
        hover_cols = []
        if "file_path" in dff.columns:
            hover_cols.append("file_path")
        if "group" in dff.columns:
            hover_cols.append("group")
        if "variable_name" in dff.columns:
            hover_cols.append("variable_name")
        hover_data_dict = {col: True for col in hover_cols} if hover_cols else None

        # Determine if color column is categorical or continuous
        color_param = None if color_col == "none" else color_col
        color_continuous_scale = None
        color_discrete_map = None

        if color_param and color_param in dff.columns:
            # Check if it's numeric (continuous) or categorical
            if pd.api.types.is_numeric_dtype(dff[color_param]):
                color_continuous_scale = "Viridis"
            else:
                # For target_type, use our custom colors
                if color_param == "target_type":
                    color_discrete_map = {
                        "data_vars": COLORS["primary"],
                        "coords": COLORS["accent"],
                    }

        fig = px.scatter(
            dff,
            x=x_data,
            y=y_data,
            color=color_param,
            hover_data=hover_data_dict,
            labels={"x": x_label, "y": y_label},
            color_continuous_scale=color_continuous_scale,
            color_discrete_map=color_discrete_map,
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor=COLORS["border"], zeroline=True),
            yaxis=dict(showgrid=True, gridcolor=COLORS["border"], zeroline=True),
            height=500,
        )
        return fig

    @app.callback(
        Output("data-table", "children"),
        Input("file-filter", "value"),
        Input("group-filter", "value"),
        Input("variable-filter", "value"),
        Input("scatter-x", "value"),
        Input("scatter-y", "value"),
    )
    def update_table(file_val, group_val, variable_val, x_col, y_col):
        dff = get_filtered_df(file_val, group_val, variable_val)

        if len(dff) == 0:
            return html.P(
                "No data available", style={"color": COLORS["text_secondary"]}
            )

        # Create a copy and add index as a column if needed
        table_df = dff.head(500).copy()  # Limit to 500 rows for performance

        # Add row number/index column
        if isinstance(dff.index, pd.MultiIndex):
            # For MultiIndex, try to reset only if names don't conflict
            index_names = [name for name in dff.index.names if name]
            conflicting = [name for name in index_names if name in table_df.columns]
            if conflicting:
                # Add a simple row index instead
                table_df.insert(0, "row_index", range(len(table_df)))
            else:
                table_df = table_df.reset_index()
        else:
            # For regular index, add as row number
            table_df.insert(0, "row_index", range(len(table_df)))

        # Build priority columns list: index columns first, then file_path, group, variable_name
        priority_cols = []

        # Add index-related columns
        if "row_index" in table_df.columns:
            priority_cols.append("row_index")

        # Add file_path, group, variable_name
        priority_cols.extend(["file_path", "group", "variable_name"])

        # Add other important columns
        other_cols = [
            "target_type",
            "data_type",
            "mean",
            "std",
            "nan_percent",
            "entropy",
            "kurtosis",
            "skewness",
        ]
        for col in other_cols:
            if col in table_df.columns and col not in priority_cols:
                priority_cols.append(col)

        # Add any remaining columns
        for col in table_df.columns:
            if col not in priority_cols:
                priority_cols.append(col)

        # Reorder columns
        table_df = table_df[priority_cols]

        # Prepare column styling to highlight scatter plot axes
        columns = []
        for col in table_df.columns:
            col_style = {"name": col, "id": col}
            columns.append(col_style)

        # Create conditional styling for highlighted columns
        style_data_conditional = []
        if x_col and x_col != "__index__" and x_col in table_df.columns:
            style_data_conditional.append(
                {
                    "if": {"column_id": x_col},
                    "backgroundColor": "rgba(45, 53, 97, 0.1)",
                    "fontWeight": "600",
                }
            )
        if y_col and y_col in table_df.columns:
            style_data_conditional.append(
                {
                    "if": {"column_id": y_col},
                    "backgroundColor": "rgba(91, 124, 153, 0.1)",
                    "fontWeight": "600",
                }
            )

        return dash_table.DataTable(
            data=table_df.to_dict("records"),
            columns=columns,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "8px",
                "fontSize": "0.875rem",
                "fontFamily": "Inter",
            },
            style_header={
                "backgroundColor": COLORS["background"],
                "fontWeight": "600",
                "borderBottom": f"2px solid {COLORS['border']}",
            },
            style_data_conditional=style_data_conditional,
            sort_action="native",
            sort_mode="multi",
            page_size=100,
        )

    return app


def launch_dashboard(
    parquet_file: str,
    title: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
):
    """Launch the dashboard for a given parquet file.

    Args:
        parquet_file: Path to the parquet file containing linting results
        title: Optional custom title for the dashboard
        host: Host to run the server on
        port: Port to run the server on
        debug: Enable debug mode
    """
    # Load data
    df = load_parquet_file(parquet_file)
    if "variable_name" in df.columns:
        df = df.drop("variable_name", axis=1)
    df = df.reset_index()

    # Create title from filename if not provided
    if title is None:
        filename = Path(parquet_file).stem
        title = f"XR-Linter Dashboard - {filename}"

    # Create and run app
    app = create_dashboard(df, title)

    print(f"\n{'=' * 60}")
    print(f"  XR-Linter Dashboard")
    print(f"{'=' * 60}")
    print(f"  📊 Loaded: {len(df):,} variables")
    print(f"  🌐 Running at: http://{host}:{port}")
    print(f"  💡 Press Ctrl+C to stop")
    print(f"{'=' * 60}\n")

    app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point for the CLI."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch XR-Linter Dashboard for visualizing linting results"
    )
    parser.add_argument(
        "parquet_file", help="Path to the parquet file containing linting results"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to run the dashboard on (default: 127.0.0.1)",
    )
    parser.add_argument("--title", help="Custom title for the dashboard")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    launch_dashboard(
        parquet_file=args.parquet_file,
        title=args.title,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m xinter.gui <parquet_file> [--port PORT]")
        sys.exit(1)

    file_path = sys.argv[1]
    port = 8050

    if "--port" in sys.argv:
        port_idx = sys.argv.index("--port")
        if port_idx + 1 < len(sys.argv):
            port = int(sys.argv[port_idx + 1])

    launch_dashboard(file_path, port=port, debug=True)
