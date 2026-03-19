import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def check_consistent_dimension_names(df: pd.DataFrame) -> pd.DataFrame:
    """Check if dimension names are consistent across all files for each variable.

    Args:
        df: DataFrame containing linting results with columns:
            file_path, group, variable_name, checker_name, value
    Returns:
        DataFrame with columns: variable_name, consistent (bool)
    """
    dim_names = df.loc[df.checker_name == "dimension_names"]
    results = (
        dim_names.groupby(["file_path", "group", "variable_name"])
        .value.unique()
        .apply(lambda x: len(x) == 1)
    )
    for var_name, consistent in results.items():
        if not consistent:
            print(f"Variable: {var_name}, Consistent Dimension Names: {consistent}")


def check_consistent_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Check if data types are consistent across all files for each variable.

    Args:
        df: DataFrame containing linting results with columns:
            file_path, group, variable_name, checker_name, value
    Returns:
        DataFrame with columns: variable_name, consistent (bool)
    """
    dtypes = df.loc[df.checker_name == "data_type"]
    results = (
        dtypes.groupby(["file_path", "group", "variable_name"])
        .value.unique()
        .apply(lambda x: len(x) == 1)
    )
    for var_name, consistent in results.items():
        if not consistent:
            print(f"Variable: {var_name}, Consistent Data Types: {consistent}")


def check_constant_dimension_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.pivot(
        index=["file_path", "group", "variable_name", "target_type"],
        columns="checker_name",
        values="value",
    )
    df = df["constant_along_dimension"].dropna()
    df = df.reset_index()
    df = df.loc[df.target_type != "coords"]  # Exclude coordinate variables
    result = (
        df.groupby(["group", "variable_name"])
        .constant_along_dimension.unique()
        .apply(len)
        > 1
    )
    print(result)


def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    dtypes = df["data_type"].apply(lambda x: np.dtype(x))
    dtypes = dtypes.apply(
        lambda dt: np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.integer)
    )
    df = df.loc[dtypes]
    return df


def check_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Check if mean values are outliers across all files for each variable.

    Args:
        df: DataFrame containing linting results with columns:
            file_path, group, variable_name, checker_name, value
    Returns:
        DataFrame with columns: variable_name, mean_value, is_outlier (bool)
    """
    df = df.pivot(
        index=["file_path", "group", "variable_name", "target_type"],
        columns="checker_name",
        values="value",
    )
    features = ["mean", "std", "min", "max", "range", "iqr"]
    df = df[features + ["data_type"]]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = _select_numeric(df)
    df = df.reset_index()
    df = df.loc[df.target_type != "coords"]  # Exclude coordinate variables
    print(df)

    groups = df.groupby(["group", "variable_name", "target_type"])

    def _check_outliers(group, features):
        features = group[features]
        features = features.astype(float)  # Ensure all features are numeric
        features = features.fillna(features.mean())  # Handle NaN values

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        isoforest = IsolationForest(contamination=0.1, random_state=42)
        group["is_outlier"] = isoforest.fit_predict(features) == -1
        return group

    outliers = groups.apply(_check_outliers, features)
    outliers = outliers.loc[outliers.target_type != "coords"]
    df = outliers.loc[outliers.variable_name == "horizontal_cam_upper"]
    for _, row in df.iterrows():
        if row["is_outlier"]:
            print(
                f"File: {row['file_path']}, Variable: {row['variable_name']}, Group: {row['group']}, Is Outlier: {row['is_outlier']}"
            )


def main():
    parser = argparse.ArgumentParser(description="XR Cross Check Linter CLI")
    parser.add_argument(
        "linter_output", type=str, help="Path to the CSV file containing linter results"
    )

    args = parser.parse_args()
    df = pd.read_csv(args.linter_output)

    check_consistent_dimension_names(df)
    check_consistent_dtypes(df)
    check_constant_dimension_values(df)
    # check_outliers(df)


if __name__ == "__main__":
    main()
