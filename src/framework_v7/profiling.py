"""Profiling and lightweight transformation helpers.

The functions in this module are independent from Streamlit and can be reused
by notebooks, tests or future scripts. They describe dataset quality, detect
basic temporal columns and create normalized values for visual categories.
"""

from __future__ import annotations

import pandas as pd

from .catalog import LAYER_CATALOG
from .data_access import load_excel
from .paths import rel


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns from a DataFrame.

    Args:
        df: Dataset to inspect.

    Returns:
        List of numeric column names.
    """

    return list(df.select_dtypes(include="number").columns)


def normalize_01(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to the [0, 1] range.

    Args:
        series: Numeric pandas Series.

    Returns:
        Series scaled between 0 and 1. Constant or invalid inputs return zeros.
    """

    values = pd.to_numeric(series, errors="coerce")
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        return pd.Series([0.0] * len(values), index=series.index)
    return (values - values.min()) / span


def quality_badge(value: float) -> str:
    """Map a normalized numeric value to a qualitative category.

    Args:
        value: Normalized value between 0 and 1.

    Returns:
        Text label: Bajo, Medio, Alto, Critico or Sin dato.
    """

    if pd.isna(value):
        return "Sin dato"
    if value <= 0.25:
        return "Bajo"
    if value <= 0.50:
        return "Medio"
    if value <= 0.75:
        return "Alto"
    return "Critico"


def find_date_column(df: pd.DataFrame) -> str | None:
    """Find the first recognizable date-like column.

    Args:
        df: Dataset to inspect.

    Returns:
        Column name when detected, otherwise None.
    """

    for candidate in ["Fecha", "fecha", "fechaobservacion", "A\xf1o_Mes", "Ano_Mes", "mes_id"]:
        if candidate in df.columns:
            return candidate
    return None


def find_node_column(df: pd.DataFrame) -> str | None:
    """Find the first recognizable location or node column.

    Args:
        df: Dataset to inspect.

    Returns:
        Column name when detected, otherwise None.
    """

    for candidate in ["Nodo", "nodo", "Municipio", "municipio", "Estacion", "estacion"]:
        if candidate in df.columns:
            return candidate
    return None


def dataset_profile(df: pd.DataFrame) -> dict[str, object]:
    """Compute high-level quality indicators for a dataset.

    Args:
        df: Dataset to profile.

    Returns:
        Dictionary with rows, columns, nulls, duplicates and numeric count.
    """

    if df.empty:
        return {"filas": 0, "columnas": 0, "nulos": 0, "duplicados": 0, "numericas": 0}
    return {
        "filas": len(df),
        "columnas": df.shape[1],
        "nulos": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "numericas": len(numeric_columns(df)),
    }


def missing_profile(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Compute missing-value coverage by variable.

    Args:
        df: Dataset to profile.
        limit: Maximum number of rows to return.

    Returns:
        DataFrame with variable name, null count and coverage percentage.
    """

    if df.empty:
        return pd.DataFrame(columns=["Variable", "Nulos", "Cobertura_%"])
    missing = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "Variable", 0: "Nulos"})
        .sort_values("Nulos", ascending=False)
    )
    missing["Cobertura_%"] = (1 - missing["Nulos"] / max(len(df), 1)) * 100
    return missing.head(limit)


def layer_summary() -> pd.DataFrame:
    """Build a summary table for the system layers.

    Returns:
        DataFrame with layer name, role, main file, shape and availability.
    """

    rows = []
    for name, config in LAYER_CATALOG.items():
        path = config["folder"] / config["main"]
        df = load_excel(path)
        profile = dataset_profile(df)
        rows.append(
            {
                "Capa": name,
                "Rol sistemico": config["role"],
                "Archivo": rel(path),
                "Filas": profile["filas"],
                "Columnas": profile["columnas"],
                "Nulos": profile["nulos"],
                "Disponible": path.exists(),
            }
        )
    return pd.DataFrame(rows)
