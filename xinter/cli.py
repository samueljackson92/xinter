"""Command-line interface for the XR Linter."""

import os
import argparse
import tempfile
import multiprocessing as mp
import shutil
import sys
from pathlib import Path
from functools import partial

import pandas as pd
from loguru import logger

from xinter.core import lint_dataset_with_error_handling

WORKER_TIMEOUT = int(
    os.environ.get("XINTER_WORKER_TIMEOUT", 60 * 5)
)  # Default to 5 minutes


def gather_results(futures):
    """Gather results from parallel linting and print summary to console."""
    for file_path, future in futures:
        try:
            _, error = future.get(
                timeout=WORKER_TIMEOUT
            )  # Wait up to WORKER_TIMEOUT seconds for each file to be linted
        except mp.TimeoutError:
            logger.error(f"Timeout linting {file_path}")
            continue

        if error is not None:
            logger.error(f"Error linting {file_path}: {error}")
        else:
            logger.info(f"Successfully linted {file_path}")


def main():
    """Command-line interface for the XR Linter."""
    parser = argparse.ArgumentParser(description="XR Linter CLI")
    parser.add_argument(
        "files",
        metavar="FILE",
        type=str,
        nargs="+",
        help="Files to be linted",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Group name for xarray",
    )
    parser.add_argument(
        "--coords",
        action="store_true",
        help="Also check coordinates in addition to data variables",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="linting_report.parquet",
        help="Location to save output file",
    )
    parser.add_argument(
        "-n",
        "--num-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs to run (default: use all available cores)",
    )
    parser.add_argument(
        "--channel-wise",
        action="store_true",
        help="Perform linting in a channel-wise manner on datasets where one axis is time",
    )

    args = parser.parse_args()

    output_file = Path(args.output)
    tmp_dir = Path(tempfile.mkdtemp(dir="."))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Run linting in parallel for all files
    linting_func = partial(
        lint_dataset_with_error_handling,
        group=args.group,
        check_coords=args.coords,
        channel_wise=args.channel_wise,
        output_dir=tmp_dir,
    )
    with mp.Pool(processes=args.num_jobs, maxtasksperchild=1) as pool:
        futures = [
            (item, pool.apply_async(linting_func, (item,))) for item in args.files
        ]
        gather_results(futures)

    logger.info("Linting completed.")

    dfs = pd.concat(
        [pd.read_parquet(file) for file in Path(tmp_dir).glob("*.parquet")],
        ignore_index=True,
    )

    if dfs.empty:
        logger.warning("No files were successfully linted. No report generated.")
        sys.exit(0)

    dfs = dfs.sort_values(
        by=["file_path", "group", "target_type", "variable_name", "checker_name"]
    )

    type_lookup = dfs[["checker_name", "value_type"]]

    dfs = dfs.pivot(
        index=["file_path", "group", "variable_name", "target_type"],
        columns="checker_name",
        values="value",
    )

    type_map = {
        "int": int,
        "float": float,
        "bool": lambda x: x == "True",
        "str": str,
    }

    for col in dfs.columns:
        # get the type for this column from the metadata
        dtype = type_lookup[type_lookup["checker_name"] == col]["value_type"].values[0]
        dfs[col] = dfs[col].apply(type_map[dtype])

    dfs.drop("variable_name", axis=1, inplace=True)
    dfs = dfs.reset_index()

    if output_file.suffix == ".parquet":
        dfs.to_parquet(output_file)
    elif output_file.suffix == ".csv":
        dfs.to_csv(output_file)
    else:
        logger.error("Unsupported output format. Please use .parquet or .csv.")
        sys.exit(1)
    shutil.rmtree(tmp_dir)  # Clean up individual parquet files
    logger.info(f"Combined linting report saved to {output_file}")


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
