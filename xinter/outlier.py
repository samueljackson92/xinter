"""Outlier detection command for xinter linting reports."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

_IDENTITY_COLS = {"file_path", "group", "variable_name", "target_type"}


def load_report(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: report file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.reset_index(drop=True)
    elif path.suffix == ".csv":
        # cli.py saves CSV with index=True (default), so first column is the saved RangeIndex
        df = pd.read_csv(path, index_col=0)
        return df.reset_index(drop=True)
    else:
        print(
            f"Error: unsupported format '{path.suffix}'. Use .parquet or .csv.",
            file=sys.stderr,
        )
        sys.exit(1)


def identify_numeric_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    all_cols = set(df.columns)
    numeric = set(df.select_dtypes(include="number", exclude=["bool"]).columns)
    candidates = numeric - _IDENTITY_COLS

    analysable: list[str] = []
    constant: list[str] = []
    for col in sorted(candidates):
        if df[col].nunique(dropna=True) >= 2:
            analysable.append(col)
        else:
            constant.append(col)

    non_numeric = sorted(all_cols - numeric - _IDENTITY_COLS)
    skipped = constant + non_numeric
    return analysable, skipped


def _make_finding(
    row: pd.Series, metric: str, value: float, score: float, severity: str
) -> dict:
    return {
        "file_path": str(row.get("file_path", "")),
        "group": row.get("group", None),
        "variable_name": str(row.get("variable_name", "")),
        "target_type": str(row.get("target_type", "")),
        "metric": metric,
        "value": float(value) if pd.notna(value) else None,
        "score": round(float(score), 4),
        "severity": severity,
    }


def detect_zscore(
    df: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float = 3.0,
) -> list[dict]:
    issues: list[dict] = []
    for col in numeric_cols:
        vals = df[col].to_numpy(dtype=float)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std == 0 or np.isnan(std):
            continue
        z = np.abs((vals - mean) / std)
        for idx in np.where(z > threshold)[0]:
            if np.isnan(vals[idx]):
                continue
            abs_z = float(z[idx])
            severity = "high" if abs_z > 5 else "medium"
            issues.append(_make_finding(df.iloc[idx], col, vals[idx], abs_z, severity))
    return issues


def detect_iqr(
    df: pd.DataFrame,
    numeric_cols: list[str],
    k: float = 1.5,
) -> list[dict]:
    issues: list[dict] = []
    for col in numeric_cols:
        vals = df[col].to_numpy(dtype=float)
        if np.sum(np.isfinite(vals)) < 2:
            continue
        q1, q3 = np.nanpercentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        for idx, v in enumerate(vals):
            if not np.isfinite(v):
                continue
            if v < lower:
                score = (lower - v) / iqr
            elif v > upper:
                score = (v - upper) / iqr
            else:
                continue
            severity = "high" if score > 3 else "medium"
            issues.append(_make_finding(df.iloc[idx], col, v, score, severity))
    return issues


def detect_isolation_forest(
    df: pd.DataFrame,
    numeric_cols: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
    console: Console | None = None,
) -> list[dict]:
    from sklearn.ensemble import IsolationForest

    clean_mask = df[numeric_cols].notna().all(axis=1)
    df_clean = df[clean_mask].reset_index(drop=True)

    if len(df_clean) < 10:
        if console:
            console.print(
                "[yellow]⚠️  Fewer than 10 rows have complete data across all numeric "
                "metrics. Isolation Forest skipped.[/yellow]"
            )
        return []

    X = df_clean[numeric_cols].to_numpy(dtype=float)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    issues: list[dict] = []
    for local_idx, (pred, score) in enumerate(zip(predictions, scores)):
        if pred == -1:
            row = df_clean.iloc[local_idx]
            issues.append(
                _make_finding(row, "__multivariate__", float("nan"), float(score), "medium")
            )
    return issues


def build_summary(issues: list[dict]) -> dict:
    affected = {i["variable_name"] for i in issues}
    metrics = sorted({i["metric"] for i in issues})
    severity_counts: dict[str, int] = {}
    for issue in issues:
        sev = issue["severity"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    return {
        "total_findings": len(issues),
        "affected_variables": len(affected),
        "metrics_with_outliers": metrics,
        "severity_counts": severity_counts,
    }


def build_report(
    report_path: Path,
    method: str,
    threshold: float,
    numeric_cols: list[str],
    skipped_cols: list[str],
    total_variables: int,
    issues: list[dict],
) -> dict:
    return {
        "metadata": {
            "report_path": str(report_path.resolve()),
            "method": method,
            "threshold": threshold,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics_analyzed": numeric_cols,
            "metrics_skipped": skipped_cols,
            "total_variables": total_variables,
        },
        "summary": build_summary(issues),
        "issues": issues,
    }


def write_output(
    report: dict,
    output_path: Path,
    fmt: Literal["json", "jsonl", "csv"],
) -> None:
    if fmt == "json":
        output_path.write_text(json.dumps(report, indent=2, default=str))
    elif fmt == "jsonl":
        lines = [json.dumps(issue, default=str) for issue in report["issues"]]
        meta_line = json.dumps(
            {"metadata": report["metadata"], "summary": report["summary"]}, default=str
        )
        lines.append(meta_line)
        output_path.write_text("\n".join(lines) + "\n")
    elif fmt == "csv":
        pd.DataFrame(report["issues"]).to_csv(output_path, index=False)
        meta_path = output_path.with_name(output_path.stem + "_meta.json")
        meta_path.write_text(
            json.dumps(
                {"metadata": report["metadata"], "summary": report["summary"]},
                indent=2,
                default=str,
            )
        )


def print_summary(report: dict, console: Console) -> None:
    meta = report["metadata"]
    summary = report["summary"]

    table = Table(
        title="Outlier Detection Summary", show_header=False, box=None, padding=(0, 1)
    )
    table.add_column(justify="left", no_wrap=True)
    table.add_column(justify="left", no_wrap=True)

    table.add_row("[dim]Method[/dim]", f"[bold]{meta['method']}[/bold]")
    table.add_row("[dim]Threshold[/dim]", f"[bold]{meta['threshold']}[/bold]")
    table.add_row("[dim]Variables scanned[/dim]", f"[bold]{meta['total_variables']:,}[/bold]")
    table.add_row("[dim]Metrics analysed[/dim]", f"[bold]{len(meta['metrics_analyzed'])}[/bold]")
    table.add_row("[dim]Total findings[/dim]", f"[bold]{summary['total_findings']}[/bold]")
    table.add_row(
        "[dim]Affected variables[/dim]", f"[bold]{summary['affected_variables']}[/bold]"
    )

    for sev, count in summary.get("severity_counts", {}).items():
        colour = "red" if sev == "high" else "yellow"
        table.add_row(
            f"[{colour}]{sev.capitalize()} severity[/{colour}]",
            f"[bold {colour}]{count}[/bold {colour}]",
        )

    if summary["metrics_with_outliers"]:
        top = summary["metrics_with_outliers"][:5]
        table.add_row("[dim]Top metrics[/dim]", ", ".join(top))

    console.print()
    console.print(table)
    console.print()


def main() -> None:
    """Entry point for the xl-outlier CLI command."""
    parser = argparse.ArgumentParser(
        description="Run outlier detection on an xinter linting report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "report",
        type=str,
        help="Path to the linting report (.parquet or .csv) produced by xl",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["zscore", "iqr", "isolation_forest"],
        default="zscore",
        help="Outlier detection method",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=3.0,
        help="Detection threshold (z-score cutoff or IQR k multiplier)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected fraction of outliers, used only by isolation_forest",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="outlier_report.json",
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "csv"],
        default=None,
        help="Output format (inferred from -o suffix if omitted)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress rich console output",
    )

    args = parser.parse_args()

    console = Console(quiet=args.quiet)
    report_path = Path(args.report)
    output_path = Path(args.output)

    fmt_map = {".json": "json", ".jsonl": "jsonl", ".csv": "csv"}
    fmt: Literal["json", "jsonl", "csv"] = args.format or fmt_map.get(output_path.suffix, "json")

    console.print()
    console.print("[bold cyan]🔍 XR Outlier Detection[/bold cyan]")
    console.print(f"[dim]Report:  {report_path}[/dim]")
    console.print(f"[dim]Method:  {args.method}[/dim]")
    console.print(f"[dim]Output:  {output_path}[/dim]")
    console.print()

    df = load_report(report_path)

    if df.empty:
        console.print("[red]❌ Report is empty. Nothing to analyse.[/red]")
        sys.exit(1)

    numeric_cols, skipped_cols = identify_numeric_columns(df)

    if not numeric_cols:
        console.print("[red]❌ No numeric columns found in report.[/red]")
        sys.exit(1)

    if args.method == "zscore":
        issues = detect_zscore(df, numeric_cols, threshold=args.threshold)
    elif args.method == "iqr":
        issues = detect_iqr(df, numeric_cols, k=args.threshold)
    else:
        issues = detect_isolation_forest(
            df,
            numeric_cols,
            contamination=args.contamination,
            console=console,
        )

    report = build_report(
        report_path=report_path,
        method=args.method,
        threshold=args.threshold,
        numeric_cols=numeric_cols,
        skipped_cols=skipped_cols,
        total_variables=len(df),
        issues=issues,
    )

    write_output(report, output_path, fmt)
    print_summary(report, console)

    console.print(f"[green]💾 Outlier report saved to:[/green] [bold]{output_path}[/bold]")
    console.print(f"[dim]   Format:   {fmt}[/dim]")
    console.print(f"[dim]   Findings: {len(issues)}[/dim]")
    console.print()
