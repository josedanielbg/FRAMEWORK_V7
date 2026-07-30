"""Feature-engineering helpers extracted from notebook C09."""

from __future__ import annotations

import pandas as pd

from framework_v7.profiling import missing_profile


def add_temporal_features(df: pd.DataFrame, date_col: str = "Fecha") -> pd.DataFrame:
    """Add common temporal features to a master dataset.

    Args:
        df: Dataset with a date column.
        date_col: Date column name.

    Returns:
        Copy with ``Anio``, ``Mes`` and ``Trimestre`` when dates are valid.
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
    """Create binary indicators for missing values.

    Args:
        df: Input dataset.
        columns: Columns to inspect. Defaults to all columns.
        suffix: Suffix for generated flag columns.

    Returns:
        Copy with missing-value flags.
    """

    output = df.copy()
    selected = columns or list(output.columns)
    for column in selected:
        if column in output.columns:
            output[f"{column}{suffix}"] = output[column].isna().astype(int)
    return output


def impute_numeric_by_group(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Impute numeric columns using grouped medians and global medians.

    Args:
        df: Input dataset.
        group_cols: Optional group columns, for example ``Nodo`` or ``Mes``.

    Returns:
        Copy with numeric missing values imputed.
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
    """Compute complete coverage for every variable.

    Args:
        df: Dataset to profile.

    Returns:
        DataFrame with variable, null count and coverage percentage.
    """

    return missing_profile(df, limit=len(df.columns) if not df.empty else 0)


def build_engineered_master(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run the default C09 engineering flow.

    Args:
        df: Master dataset.
        date_col: Date column name.
        group_cols: Optional group columns for imputation.

    Returns:
        Engineered master dataset.
    """

    engineered = add_temporal_features(df, date_col=date_col)
    return impute_numeric_by_group(engineered, group_cols=group_cols)
