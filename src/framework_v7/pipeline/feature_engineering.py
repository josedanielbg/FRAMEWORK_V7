"""Feature-engineering helpers extracted from notebook C09."""

from __future__ import annotations

import pandas as pd

from framework_v7.profiling import missing_profile


def add_temporal_features(df: pd.DataFrame, date_col: str = "Fecha") -> pd.DataFrame:
    """Add temporal features derived from a date column.

    This function mirrors the C09 feature-engineering notebook by deriving
    calendar fields that can support grouped imputations, diagnostics and
    modeling.

    Args:
        df (pd.DataFrame): Dataset containing a date-like column.
        date_col (str): Date column used to derive temporal attributes.

    Returns:
        pd.DataFrame: Copy with ``Anio``, ``Mes`` and ``Trimestre`` when the
        date column exists. Returns an unchanged copy otherwise.
    """

    output = df.copy()
    if date_col not in output.columns:
        return output
    dates = pd.to_datetime(output[date_col], errors="coerce")
    output["Anio"] = dates.dt.year
    output["Mes"] = dates.dt.month
    output["Trimestre"] = dates.dt.quarter
    return output


def add_missing_flags(df: pd.DataFrame, columns: list[str] | None = None, suffix: str = "_faltante") -> pd.DataFrame:
    """Create binary missing-value indicators.

    Args:
        df (pd.DataFrame): Input dataset.
        columns (list[str] | None): Columns to inspect. Defaults to all
            columns in ``df``.
        suffix (str): Suffix appended to generated flag columns.

    Returns:
        pd.DataFrame: Copy with one binary flag per selected column.
    """

    output = df.copy()
    selected = columns or list(output.columns)
    for column in selected:
        if column in output.columns:
            output[f"{column}{suffix}"] = output[column].isna().astype(int)
    return output


def impute_numeric_by_group(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Impute numeric columns using grouped and global medians.

    When group columns are supplied, missing values are first filled with the
    median inside each group. Remaining missing values are filled with the
    global median for the variable.

    Args:
        df (pd.DataFrame): Input dataset.
        group_cols (list[str] | None): Optional grouping columns, such as
            ``Nodo`` or ``Mes``.

    Returns:
        pd.DataFrame: Copy with numeric missing values imputed.
    """

    output = df.copy()
    numeric_cols = list(output.select_dtypes(include="number").columns)
    groups = [column for column in (group_cols or []) if column in output.columns]
    for column in numeric_cols:
        if groups:
            grouped = output.groupby(groups, dropna=False)[column].transform("median")
            output[column] = output[column].fillna(grouped)
        output[column] = output[column].fillna(output[column].median())
    return output


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute variable-level coverage for a dataset.

    Args:
        df (pd.DataFrame): Dataset to profile.

    Returns:
        pd.DataFrame: Table with variable name, null count and coverage
        percentage for every column.
    """

    return missing_profile(df, limit=len(df.columns) if not df.empty else 0)


def build_engineered_master(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run the default C09 feature-engineering flow.

    The current flow adds temporal features and imputes numeric values. It is
    intentionally lightweight so notebooks can compose it with additional
    domain-specific transformations when needed.

    Args:
        df (pd.DataFrame): Master dataset.
        date_col (str): Date column used for temporal features.
        group_cols (list[str] | None): Optional group columns used during
            numeric imputation.

    Returns:
        pd.DataFrame: Engineered master dataset.
    """

    engineered = add_temporal_features(df, date_col=date_col)
    return impute_numeric_by_group(engineered, group_cols=group_cols)
