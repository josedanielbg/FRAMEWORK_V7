"""Modeling helpers extracted from notebook C14."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one modeling experiment.

    Attributes:
        experiment (str): Experiment identifier, for example ``Exp01``.
        target (str): Target variable to predict.
        model (str): Model family or algorithm name.
        window (int): Number of time steps used as input.
        horizon (int): Forecast horizon measured in rows.
        problem_type (str): Problem type, such as classification or regression.
    """

    experiment: str
    target: str
    model: str = "LSTM"
    window: int = 12
    horizon: int = 1
    problem_type: str = "Clasificacion"


def config_from_row(row: pd.Series) -> ExperimentConfig:
    """Create an experiment configuration from a catalog row.

    Args:
        row (pd.Series): Row from ``catalogo_experimentos.csv`` or a compatible
            experiment catalog.

    Returns:
        ExperimentConfig: Parsed configuration with defaults for missing
        fields.
    """

    return ExperimentConfig(
        experiment=str(row.get("Experimento", "Exp01")),
        target=str(row.get("Variable_Objetivo", "irca")),
        model=str(row.get("Modelo", "LSTM")),
        window=int(row.get("Ventana", 12)),
        horizon=int(row.get("Horizonte", 1)),
        problem_type=str(row.get("Tipo_Problema", "Clasificacion")),
    )


def validate_tensors(x_values: np.ndarray, y_values: np.ndarray) -> pd.DataFrame:
    """Validate tensor shapes before model training.

    Args:
        x_values (np.ndarray): Predictor tensor, usually shaped as
            ``samples x window x features``.
        y_values (np.ndarray): Target tensor or vector.

    Returns:
        pd.DataFrame: Diagnostic table with tensor dimensions, sample counts
        and alignment status.
    """

    return pd.DataFrame(
        [
            {"Indicador": "X_dimensiones", "Valor": "x".join(map(str, x_values.shape))},
            {"Indicador": "y_dimensiones", "Valor": "x".join(map(str, y_values.shape))},
            {"Indicador": "muestras", "Valor": int(len(x_values))},
            {"Indicador": "muestras_y", "Valor": int(len(y_values))},
            {"Indicador": "alineado", "Valor": bool(len(x_values) == len(y_values))},
        ]
    )


def temporal_split(
    x_values: np.ndarray,
    y_values: np.ndarray,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split tensors preserving temporal order.

    Unlike random splits, this function keeps older observations in train and
    newer observations in test, which is appropriate for forecasting workflows.

    Args:
        x_values (np.ndarray): Predictor tensor.
        y_values (np.ndarray): Target tensor or vector.
        train_fraction (float): Fraction assigned to the training partition.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple
        ``X_train, X_test, y_train, y_test``.
    """

    split_index = int(len(x_values) * train_fraction)
    return x_values[:split_index], x_values[split_index:], y_values[:split_index], y_values[split_index:]


def prediction_frame(
    predictions: np.ndarray | list[float],
    start_index: int = 1,
    column: str = "Prediccion",
) -> pd.DataFrame:
    """Create a standard prediction output table.

    Args:
        predictions (np.ndarray | list[float]): Predicted values from a model.
        start_index (int): First record identifier assigned to predictions.
        column (str): Prediction column name.

    Returns:
        pd.DataFrame: Prediction table with ``Registro`` and the configured
        prediction column.
    """

    values = np.asarray(predictions).reshape(-1)
    return pd.DataFrame({"Registro": range(start_index, start_index + len(values)), column: values})
