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

    This function keeps the C12 notebook focused on experiment decisions while
    centralizing column selection and validation.

    Args:
        df (pd.DataFrame): Source master or engineered dataset.
        predictors (list[str]): Candidate predictor columns requested by the
            experiment design.
        target (str): Target variable to predict.
        identity_columns (list[str] | None): Optional identifier/date columns to
            keep. Defaults to ``Fecha``, ``Nodo``, ``Anio`` and ``Mes`` when
            present.

    Returns:
        pd.DataFrame: DataFrame with available identity columns, available
        predictors and the target column.

    Raises:
        ValueError: If the target variable is missing.
    """

    if target not in df.columns:
        raise ValueError(f"Target variable not found: {target}")
    identity = existing_columns(df, identity_columns or ["Fecha", "Nodo", "Anio", "Mes"])
    available_predictors = existing_columns(df, predictors)
    return df[identity + available_predictors + [target]].copy()


def drop_rows_without_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Remove observations where the target variable is missing.

    Args:
        df (pd.DataFrame): Modeling dataset.
        target (str): Target variable used by the experiment.

    Returns:
        pd.DataFrame: Filtered dataset with non-null target values.

    Raises:
        ValueError: If ``target`` is not present.
    """

    if target not in df.columns:
        raise ValueError(f"Target variable not found: {target}")
    return df.dropna(subset=[target]).copy()


def modeling_dataset_diagnostic(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Build structural diagnostics for a modeling dataset.

    Args:
        df (pd.DataFrame): Modeling dataset.
        target (str): Target variable used by the experiment.

    Returns:
        pd.DataFrame: Diagnostic table with rows, columns, numeric predictors,
        total nulls and target nulls.
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

    The flow selects available modeling columns and removes rows without the
    target variable. It does not scale values or build temporal sequences; those
    steps belong to C13.

    Args:
        df (pd.DataFrame): Source master dataset.
        predictors (list[str]): Predictor columns requested by the experiment.
        target (str): Target variable.

    Returns:
        pd.DataFrame: Dataset ready for C13 transformation or sequence
        construction.
    """

    selected = select_model_columns(df, predictors, target)
    return drop_rows_without_target(selected, target)
