"""Evaluation helpers extracted from notebook C15."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> pd.DataFrame:
    """Compute binary-classification metrics without external dependencies.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Metrics table.
    """

    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    total = max(len(true), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = (recall + specificity) / 2

    return pd.DataFrame(
        [
            {"Indicador": "Accuracy", "Valor": accuracy},
            {"Indicador": "Precision", "Valor": precision},
            {"Indicador": "Recall", "Valor": recall},
            {"Indicador": "F1 Score", "Valor": f1},
            {"Indicador": "Especificidad", "Valor": specificity},
            {"Indicador": "Accuracy Balanceada", "Valor": balanced_accuracy},
            {"Indicador": "TP", "Valor": tp},
            {"Indicador": "TN", "Valor": tn},
            {"Indicador": "FP", "Valor": fp},
            {"Indicador": "FN", "Valor": fn},
        ]
    )


def regression_metrics(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> pd.DataFrame:
    """Compute common regression metrics without external dependencies.

    Args:
        y_true: True numeric values.
        y_pred: Predicted numeric values.

    Returns:
        Metrics table.
    """

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    residuals = true - pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    denominator = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - np.sum(residuals**2) / denominator if denominator else 0.0
    return pd.DataFrame(
        [
            {"Indicador": "MAE", "Valor": mae},
            {"Indicador": "RMSE", "Valor": rmse},
            {"Indicador": "R2", "Valor": r2},
        ]
    )


def threshold_predictions(predictions: list[float] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities or scores into binary labels.

    Args:
        predictions: Numeric scores.
        threshold: Decision threshold.

    Returns:
        Binary numpy array.
    """

    return (np.asarray(predictions, dtype=float) >= threshold).astype(int)


def recommendations_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Generate basic recommendations from diagnostic metrics.

    Args:
        metrics: Metrics table with ``Indicador`` and ``Valor`` columns.

    Returns:
        Recommendation table.
    """

    values = dict(zip(metrics["Indicador"], pd.to_numeric(metrics["Valor"], errors="coerce")))
    recommendations = []
    if values.get("Recall", 1) == 0:
        recommendations.append("Aplicar balanceo de clases y ajustar el umbral de decision.")
    if values.get("Precision", 1) == 0:
        recommendations.append("Revisar variables predictoras y comparar con modelos base.")
    if values.get("Accuracy Balanceada", 1) < 0.6:
        recommendations.append("Usar metricas balanceadas para seleccionar el mejor modelo.")
    if not recommendations:
        recommendations.append("Mantener el modelo como linea base y validar con nuevos periodos.")
    return pd.DataFrame({"Recomendacion": recommendations})
