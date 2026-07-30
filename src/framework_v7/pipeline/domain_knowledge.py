"""Domain-knowledge catalog helpers extracted from notebook C10."""

from __future__ import annotations

import pandas as pd


def classify_variable(series: pd.Series) -> str:
    """Classify a variable by dtype and cardinality.

    Args:
        series: Variable values.

    Returns:
        Variable type label.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return "fecha"
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        return "numerica_continua" if unique_ratio > 0.05 else "numerica_discreta"
    return "categorica"


def build_variable_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Build a reusable catalog of variables.

    Args:
        df: Dataset to catalog.

    Returns:
        DataFrame with type, nulls, coverage and distinct values per variable.
    """

    rows = []
    for column in df.columns:
        series = df[column]
        nulls = int(series.isna().sum())
        rows.append(
            {
                "Variable": column,
                "Tipo_Dato": str(series.dtype),
                "Tipo_Variable": classify_variable(series),
                "Nulos": nulls,
                "Cobertura_%": (1 - nulls / max(len(series), 1)) * 100,
                "Valores_Unicos": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def assign_domain_layer(catalog: pd.DataFrame, layer_keywords: dict[str, list[str]]) -> pd.DataFrame:
    """Assign a domain layer to catalog variables using keywords.

    Args:
        catalog: Variable catalog with a ``Variable`` column.
        layer_keywords: Mapping of layer names to lowercase keywords.

    Returns:
        Catalog copy with ``Capa_Dominio``.
    """

    output = catalog.copy()

    def resolve_layer(variable: str) -> str:
        normalized = variable.lower()
        for layer, keywords in layer_keywords.items():
            if any(keyword.lower() in normalized for keyword in keywords):
                return layer
        return "Sin clasificar"

    output["Capa_Dominio"] = output["Variable"].astype(str).apply(resolve_layer)
    return output


def catalog_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    """Summarize a variable catalog by type and domain layer.

    Args:
        catalog: Output from ``build_variable_catalog``.

    Returns:
        Summary table.
    """

    if catalog.empty:
        return pd.DataFrame(columns=["Capa_Dominio", "Tipo_Variable", "Variables"])
    group_cols = [column for column in ["Capa_Dominio", "Tipo_Variable"] if column in catalog.columns]
    return catalog.groupby(group_cols).size().reset_index(name="Variables")
