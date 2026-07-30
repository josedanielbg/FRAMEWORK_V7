"""Modeling helpers extracted from notebook C14."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one modeling experiment."""

    experiment: str
    target: str
    model: str = "LSTM"
    window: int = 12
    horizon: int = 1
    problem_type: str = "Clasificacion"


def config_from_row(row: pd.Series) -> ExperimentConfig:
    """Create an experiment config from a catalog row.

    Args:
        row: Row from ``catalogo_experimentos.csv``.

    Returns:
        ExperimentConfig instance.
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
    """Validate tensor shapes before training.

    Args:
        x_values: Predictor tensor.
        y_values: Target tensor.

    Returns:
        Diagnostic table.
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

    Args:
        x_values: Predictor tensor.
        y_values: Target tensor.
        train_fraction: Fraction assigned to train.

    Returns:
        Tuple ``X_train, X_test, y_train, y_test``.
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
        predictions: Predicted values.
        start_index: First record id.
        column: Prediction column name.

    Returns:
        Prediction DataFrame.
    """

    values = np.asarray(predictions).reshape(-1)
    return pd.DataFrame({"Registro": range(start_index, start_index + len(values)), column: values})
