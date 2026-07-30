"""IPML helpers extracted from notebook C11."""

from __future__ import annotations

import pandas as pd


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale a numeric series to a 0-1 score.

    The helper is used by the IPML stage to turn heterogeneous diagnostic
    measures into comparable scores.

    Args:
        series (pd.Series): Numeric values to scale.
        higher_is_better (bool): Whether high raw values should receive high
            scores. Set to ``False`` for penalties such as VIF.

    Returns:
        pd.Series: Score values between 0 and 1. Constant or invalid inputs
        return a score of 1.0 for every row.
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
    """Compute an IPML score for candidate modeling variables.

    IPML combines data coverage, multicollinearity and domain relevance into a
    single ranking metric. Missing optional components receive neutral scores
    so the function can run with partial notebook artifacts.

    Args:
        variables (pd.DataFrame): Candidate variables table.
        variable_col (str): Column containing variable names.
        coverage_col (str): Column containing coverage percentages.
        vif_col (str | None): Optional VIF column. Lower VIF receives a higher
            score.
        domain_weight_col (str | None): Optional column with domain-relevance
            weights.

    Returns:
        pd.DataFrame: Candidate variables sorted by descending ``IPML`` with
        component score columns.

    Raises:
        ValueError: If ``variable_col`` is not present.
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
    """Select variables whose IPML score meets a threshold.

    Args:
        ipml_table (pd.DataFrame): Output from ``compute_ipml``.
        threshold (float): Minimum accepted IPML score.

    Returns:
        list[str]: Ordered variable names. Returns an empty list when required
        columns are absent.
    """

    if "Variable" not in ipml_table.columns or "IPML" not in ipml_table.columns:
        return []
    selected = ipml_table[ipml_table["IPML"] >= threshold]
    return selected["Variable"].astype(str).tolist()
