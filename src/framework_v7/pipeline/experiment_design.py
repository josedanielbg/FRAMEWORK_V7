"""Experimental-design helpers extracted from the design notebook."""

from __future__ import annotations

import pandas as pd

from framework_v7.catalog import EXPERIMENT_DESIGN_FILES
from framework_v7.data_access import load_csv


def load_design_artifacts() -> dict[str, pd.DataFrame]:
    """Load every experimental-design artifact declared in the catalog.

    Returns:
        dict[str, pd.DataFrame]: Dictionary keyed by artifact label. Missing
        files are represented as empty DataFrames by ``load_csv``.
    """

    return {label: load_csv(path) for label, path in EXPERIMENT_DESIGN_FILES.items()}


def active_predictors(predictors: pd.DataFrame, variable_col: str = "Variable") -> list[str]:
    """Return ordered predictor variable names from a design table.

    Args:
        predictors (pd.DataFrame): Predictors table, usually
            ``variables_predictoras.csv``.
        variable_col (str): Column containing predictor names.

    Returns:
        list[str]: Ordered predictor names. Returns an empty list when
        ``variable_col`` is absent.
    """

    if variable_col not in predictors.columns:
        return []
    ordered = predictors.copy()
    if "Orden" in ordered.columns:
        ordered = ordered.sort_values("Orden")
    return ordered[variable_col].dropna().astype(str).tolist()


def experiment_plan(catalog: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    """Join the experiment catalog with model configuration.

    Args:
        catalog (pd.DataFrame): Experiment catalog table.
        config (pd.DataFrame): Model and training configuration table.

    Returns:
        pd.DataFrame: Combined experiment plan. If configuration is missing,
        returns a copy of the catalog.
    """

    if catalog.empty:
        return pd.DataFrame()
    if config.empty or "Experimento" not in config.columns:
        return catalog.copy()
    return catalog.merge(config, on="Experimento", how="left", suffixes=("", "_Config"))


def experiment_status_summary(status: pd.DataFrame) -> pd.DataFrame:
    """Summarize experiments by execution state and problem type.

    Args:
        status (pd.DataFrame): Experiment status table.

    Returns:
        pd.DataFrame: Count table grouped by available ``Tipo_Problema`` and
        ``Estado`` columns.
    """

    if status.empty:
        return pd.DataFrame(columns=["Tipo_Problema", "Estado", "Experimentos"])
    group_cols = [column for column in ["Tipo_Problema", "Estado"] if column in status.columns]
    return status.groupby(group_cols).size().reset_index(name="Experimentos")


def design_summary(artifacts: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Build a high-level summary of experimental-design artifacts.

    Args:
        artifacts (dict[str, pd.DataFrame] | None): Optional artifact
            dictionary. When omitted, files are loaded from
            ``EXPERIMENT_DESIGN_FILES``.

    Returns:
        pd.DataFrame: Table with artifact name, row count, column count and
        availability flag.
    """

    artifacts = artifacts or load_design_artifacts()
    rows = []
    for label, df in artifacts.items():
        rows.append({"Artefacto": label, "Filas": len(df), "Columnas": df.shape[1], "Disponible": not df.empty})
    return pd.DataFrame(rows)
