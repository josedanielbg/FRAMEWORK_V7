"""Domain-knowledge catalog helpers extracted from notebook C10."""

from __future__ import annotations

import pandas as pd


def classify_variable(series: pd.Series) -> str:
    """Classify a variable by dtype and cardinality.

    This classification supports the C10 catalog of domain knowledge by
    separating date, continuous numeric, discrete numeric and categorical
    variables.

    Args:
        series (pd.Series): Values for one dataset variable.

    Returns:
        str: One of ``fecha``, ``numerica_continua``,
        ``numerica_discreta`` or ``categorica``.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return "fecha"
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        return "numerica_continua" if unique_ratio > 0.05 else "numerica_discreta"
    return "categorica"


def build_variable_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Build a reusable catalog of dataset variables.

    Args:
        df (pd.DataFrame): Dataset to catalog.

    Returns:
        pd.DataFrame: Catalog with variable name, dtype, inferred variable type,
        null count, coverage percentage and number of unique values.
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
    """Assign a domain layer to catalog variables using keyword rules.

    Args:
        catalog (pd.DataFrame): Variable catalog with a ``Variable`` column.
        layer_keywords (dict[str, list[str]]): Mapping of domain-layer labels to
            keywords searched in variable names.

    Returns:
        pd.DataFrame: Catalog copy with a ``Capa_Dominio`` column.
    """

    output = catalog.copy()

    def resolve_layer(variable: str) -> str:
        """Resolve a domain layer label for one variable name.

        Args:
            variable (str): Variable name to classify.

        Returns:
            str: Matching domain layer, or ``Sin clasificar`` when no keyword
            matches.
        """

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
        catalog (pd.DataFrame): Output from ``build_variable_catalog`` or a
            compatible catalog table.

    Returns:
        pd.DataFrame: Summary table with counts by available grouping columns.
    """

    if catalog.empty:
        return pd.DataFrame(columns=["Capa_Dominio", "Tipo_Variable", "Variables"])
    group_cols = [column for column in ["Capa_Dominio", "Tipo_Variable"] if column in catalog.columns]
    return catalog.groupby(group_cols).size().reset_index(name="Variables")
