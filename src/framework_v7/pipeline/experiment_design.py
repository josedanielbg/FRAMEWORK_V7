"""Experimental-design helpers extracted from the design notebook."""

from __future__ import annotations

import pandas as pd

from framework_v7.catalog import EXPERIMENT_DESIGN_FILES
from framework_v7.data_access import load_csv


def load_design_artifacts() -> dict[str, pd.DataFrame]:
    """Load every experimental-design artifact.

    Returns:
        Dictionary keyed by artifact label.
    """

    return {label: load_csv(path) for label, path in EXPERIMENT_DESIGN_FILES.items()}


def active_predictors(predictors: pd.DataFrame, variable_col: str = "Variable") -> list[str]:
    """Return ordered predictor variable names.

    Args:
        predictors: Predictors table.
        variable_col: Variable column name.

    Returns:
        Ordered predictor names.
    """

    if variable_col not in predictors.columns:
        return []
    ordered = predictors.copy()
    if "Orden" in ordered.columns:
        ordered = ordered.sort_values("Orden")
    return ordered[variable_col].dropna().astype(str).tolist()


def experiment_plan(catalog: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    """Join experiment catalog and configuration.

    Args:
        catalog: Experiment catalog table.
        config: Experiment configuration table.

    Returns:
        Combined experiment plan.
    """

    if catalog.empty:
        return pd.DataFrame()
    if config.empty or "Experimento" not in config.columns:
        return catalog.copy()
    return catalog.merge(config, on="Experimento", how="left", suffixes=("", "_Config"))


def experiment_status_summary(status: pd.DataFrame) -> pd.DataFrame:
    """Summarize experiments by execution state and problem type.

    Args:
        status: Experiment status table.

    Returns:
        Summary table.
    """

    if status.empty:
        return pd.DataFrame(columns=["Tipo_Problema", "Estado", "Experimentos"])
    group_cols = [column for column in ["Tipo_Problema", "Estado"] if column in status.columns]
    return status.groupby(group_cols).size().reset_index(name="Experimentos")


def design_summary(artifacts: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Build a high-level summary of design artifacts.

    Args:
        artifacts: Optional artifact dictionary. Loads files when omitted.

    Returns:
        Summary table.
    """

    artifacts = artifacts or load_design_artifacts()
    rows = []
    for label, df in artifacts.items():
        rows.append({"Artefacto": label, "Filas": len(df), "Columnas": df.shape[1], "Disponible": not df.empty})
    return pd.DataFrame(rows)
