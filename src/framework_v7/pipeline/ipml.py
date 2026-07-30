"""IPML helpers extracted from notebook C11."""

from __future__ import annotations

import pandas as pd


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale a numeric series to a 0-1 score.

    Args:
        series: Numeric values.
        higher_is_better: Whether high raw values should receive high scores.

    Returns:
        Scaled score series.
    """

    values = pd.to_numeric(series, errors="coerce")
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        score = pd.Series([1.0] * len(values), index=series.index)
    else:
        score = (values - values.min()) / span
    return score if higher_is_better else 1 - score


def compute_ipml(
    variables: pd.DataFrame,
    variable_col: str = "Variable",
    coverage_col: str = "Cobertura_%",
    vif_col: str | None = None,
    domain_weight_col: str | None = None,
) -> pd.DataFrame:
    """Compute a lightweight IPML score for candidate variables.

    Args:
        variables: Candidate variables table.
        variable_col: Variable-name column.
        coverage_col: Coverage percentage column.
        vif_col: Optional VIF column. Lower VIF receives higher score.
        domain_weight_col: Optional domain-relevance weight column.

    Returns:
        Variables table with IPML components and final score.
    """

    output = variables.copy()
    if variable_col not in output.columns:
        raise ValueError(f"Missing variable column: {variable_col}")

    if coverage_col in output.columns:
        output["Puntaje_Cobertura"] = minmax_score(output[coverage_col])
    else:
        output["Puntaje_Cobertura"] = 1.0

    if vif_col and vif_col in output.columns:
        output["Puntaje_Multicolinealidad"] = minmax_score(output[vif_col], higher_is_better=False)
    else:
        output["Puntaje_Multicolinealidad"] = 1.0

    if domain_weight_col and domain_weight_col in output.columns:
        output["Puntaje_Dominio"] = minmax_score(output[domain_weight_col])
    else:
        output["Puntaje_Dominio"] = 1.0

    output["IPML"] = (
        output["Puntaje_Cobertura"] * 0.4
        + output["Puntaje_Multicolinealidad"] * 0.3
        + output["Puntaje_Dominio"] * 0.3
    )
    return output.sort_values("IPML", ascending=False)


def select_variables_by_ipml(ipml_table: pd.DataFrame, threshold: float = 0.6) -> list[str]:
    """Select variables with IPML above a threshold.

    Args:
        ipml_table: Output from ``compute_ipml``.
        threshold: Minimum score.

    Returns:
        Ordered variable names.
    """

    if "Variable" not in ipml_table.columns or "IPML" not in ipml_table.columns:
        return []
    selected = ipml_table[ipml_table["IPML"] >= threshold]
    return selected["Variable"].astype(str).tolist()
