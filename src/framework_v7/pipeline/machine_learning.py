"""Machine-learning transformation helpers extracted from notebook C13."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_minmax(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fit min-max parameters for selected columns.

    Args:
        df: Source dataset.
        columns: Numeric columns to scale.

    Returns:
        DataFrame with ``Variable``, ``Min`` and ``Max``.
    """

    rows = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append({"Variable": column, "Min": values.min(), "Max": values.max()})
    return pd.DataFrame(rows)


def apply_minmax(df: pd.DataFrame, scaler: pd.DataFrame) -> pd.DataFrame:
    """Apply min-max scaling from a parameter table.

    Args:
        df: Source dataset.
        scaler: Output from ``fit_minmax``.

    Returns:
        Scaled dataset copy.
    """

    output = df.copy()
    for row in scaler.to_dict("records"):
        column = row["Variable"]
        if column not in output.columns:
            continue
        span = row["Max"] - row["Min"]
        if pd.isna(span) or span == 0:
            output[column] = 0.0
        else:
            output[column] = (pd.to_numeric(output[column], errors="coerce") - row["Min"]) / span
    return output


def create_temporal_sequences(
    df: pd.DataFrame,
    predictors: list[str],
    target: str,
    window: int = 12,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create temporal tensors for sequence models.

    Args:
        df: Ordered modeling dataset.
        predictors: Predictor columns.
        target: Target column.
        window: Number of past rows per sequence.
        horizon: Forecast horizon measured in rows.

    Returns:
        Tuple ``(X, y)`` where X has shape ``samples x window x features``.
    """

    missing = [column for column in predictors + [target] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for sequences: {missing}")
    values = df[predictors].to_numpy(dtype=float)
    target_values = df[target].to_numpy(dtype=float)
    x_rows = []
    y_rows = []
    last_start = len(df) - window - horizon + 1
    for start in range(max(last_start, 0)):
        end = start + window
        target_index = end + horizon - 1
        x_rows.append(values[start:end])
        y_rows.append(target_values[target_index])
    return np.asarray(x_rows), np.asarray(y_rows)


def transformation_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """Build a diagnostic table for transformed ML data.

    Args:
        df: Transformed modeling dataset.

    Returns:
        Diagnostic table.
    """

    numeric_cols = df.select_dtypes(include="number").columns
    constant_cols = [column for column in numeric_cols if df[column].nunique(dropna=True) <= 1]
    return pd.DataFrame(
        [
            {"Indicador": "filas", "Valor": len(df)},
            {"Indicador": "columnas", "Valor": df.shape[1]},
            {"Indicador": "variables_numericas", "Valor": len(numeric_cols)},
            {"Indicador": "variables_constantes", "Valor": len(constant_cols)},
            {"Indicador": "nulos", "Valor": int(df.isna().sum().sum())},
        ]
    )
