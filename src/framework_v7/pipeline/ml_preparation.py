"""Machine-learning dataset preparation helpers extracted from notebook C12."""

from __future__ import annotations

import pandas as pd

from .utils import existing_columns


def select_model_columns(
    df: pd.DataFrame,
    predictors: list[str],
    target: str,
    identity_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Select identity, predictor and target columns for modeling.

    Args:
        df: Source dataset.
        predictors: Candidate predictor columns.
        target: Target variable.
        identity_columns: Optional identifier/date columns to keep.

    Returns:
        Prepared modeling DataFrame.

    Raises:
        ValueError: If the target variable is missing.
    """

    if target not in df.columns:
        raise ValueError(f"Target variable not found: {target}")
    identity = existing_columns(df, identity_columns or ["Fecha", "Nodo", "Anio", "Mes"])
    available_predictors = existing_columns(df, predictors)
    return df[identity + available_predictors + [target]].copy()


def drop_rows_without_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Remove rows where the target variable is missing.

    Args:
        df: Modeling dataset.
        target: Target variable.

    Returns:
        Filtered dataset.
    """

    if target not in df.columns:
        raise ValueError(f"Target variable not found: {target}")
    return df.dropna(subset=[target]).copy()


def modeling_dataset_diagnostic(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Build a structural diagnostic for a modeling dataset.

    Args:
        df: Modeling dataset.
        target: Target variable.

    Returns:
        Diagnostic table.
    """

    numeric_predictors = [column for column in df.select_dtypes(include="number").columns if column != target]
    return pd.DataFrame(
        [
            {"Indicador": "filas", "Valor": len(df)},
            {"Indicador": "columnas", "Valor": df.shape[1]},
            {"Indicador": "predictoras_numericas", "Valor": len(numeric_predictors)},
            {"Indicador": "nulos", "Valor": int(df.isna().sum().sum())},
            {"Indicador": "target_nulos", "Valor": int(df[target].isna().sum()) if target in df.columns else None},
        ]
    )


def build_modeling_dataset(df: pd.DataFrame, predictors: list[str], target: str) -> pd.DataFrame:
    """Run the default C12 modeling dataset preparation flow.

    Args:
        df: Source master dataset.
        predictors: Predictor columns.
        target: Target variable.

    Returns:
        Dataset ready for sequence construction or tabular modeling.
    """

    selected = select_model_columns(df, predictors, target)
    return drop_rows_without_target(selected, target)
