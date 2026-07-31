"""Interpretation helpers extracted from notebook C16."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from framework_v7.paths import INTERPRETATION_DIR

from .evaluation import classification_metrics, regression_metrics
from .utils import artifact_inventory, discover_experiments, read_table


DIMENSION_KEYWORDS = {
    "Climatica": ["precipitacion", "temp", "humedad", "radiacion", "viento", "oni"],
    "Hidrologica": ["caudal", "nivel", "volumen", "hidrolog"],
    "Calidad de agua": ["irca", "dbo", "dqo", "oxigeno", "ph", "turbiedad", "conductividad"],
    "Percepcion": ["percepcion", "social", "comunidad", "encuesta"],
    "Gobernanza": ["gobernanza", "institucional", "gestion", "norma", "cobertura"],
}


def calculate_mape(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    """Calculate mean absolute percentage error.

    Args:
        y_true (list[float] | np.ndarray): Observed numeric values.
        y_pred (list[float] | np.ndarray): Predicted numeric values.

    Returns:
        float: MAPE value in percentage units. Zero true values are ignored to
        avoid division by zero. Returns ``nan`` when no comparable values
        remain.

    Raises:
        ValueError: If the input arrays have different lengths.
    """

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    mask = true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100)


def load_interpretation_summary(experiment: str, interpretation_dir: Path | None = None) -> pd.DataFrame:
    """Load the C16 interpretation summary for one experiment.

    Args:
        experiment (str): Experiment identifier, for example ``Exp01-V3``.
        interpretation_dir (Path | None): Optional root directory containing
            interpretation outputs.

    Returns:
        pd.DataFrame: Summary exported by notebook C16. Returns an empty
        DataFrame when the artifact is missing.
    """

    root = interpretation_dir or INTERPRETATION_DIR
    return read_table(root / experiment / "resumen_experimento.csv")


def infer_problem_type(summary: pd.DataFrame | pd.Series | dict[str, object]) -> str:
    """Infer whether an experiment is classification or regression.

    Args:
        summary (pd.DataFrame | pd.Series | dict[str, object]): Summary table,
            row or dictionary containing metric columns.

    Returns:
        str: ``Clasificacion`` when classification metrics are present,
        ``Regresion`` when regression metrics are present, otherwise
        ``Desconocido``.
    """

    if isinstance(summary, pd.DataFrame):
        columns = set(summary.columns)
    else:
        columns = set(summary.keys())
    if {"Accuracy", "Precision", "Recall", "F1"}.intersection(columns):
        return "Clasificacion"
    if {"MAE", "RMSE", "MAPE", "R2"}.intersection(columns):
        return "Regresion"
    return "Desconocido"


def interpret_classification_performance(metrics: pd.Series | dict[str, object]) -> str:
    """Create a short interpretation for classification performance.

    Args:
        metrics (pd.Series | dict[str, object]): Row or dictionary with
            classification metrics.

    Returns:
        str: Human-readable interpretation for notebook reports.
    """

    values = pd.Series(metrics)
    accuracy = pd.to_numeric(values.get("Accuracy"), errors="coerce")
    f1 = pd.to_numeric(values.get("F1"), errors="coerce")
    recall = pd.to_numeric(values.get("Recall"), errors="coerce")

    if pd.notna(f1) and f1 >= 0.75 and pd.notna(recall) and recall >= 0.75:
        return "Modelo solido para clasificacion, con senales utiles para decision operativa."
    if pd.notna(accuracy) and accuracy >= 0.85 and pd.notna(f1) and f1 < 0.7:
        return "Modelo con buena exactitud global, pero sensible a desbalance o umbral de decision."
    return "Modelo base con capacidad parcial; requiere validacion de clases, umbrales y variables."


def interpret_regression_performance(metrics: pd.Series | dict[str, object]) -> str:
    """Create a short interpretation for regression performance.

    Args:
        metrics (pd.Series | dict[str, object]): Row or dictionary with
            regression metrics.

    Returns:
        str: Human-readable interpretation for notebook reports.
    """

    values = pd.Series(metrics)
    r2_score = pd.to_numeric(values.get("R2"), errors="coerce")
    mape = pd.to_numeric(values.get("MAPE"), errors="coerce")

    if pd.notna(r2_score) and r2_score >= 0.7 and (pd.isna(mape) or mape <= 25):
        return "Modelo con capacidad predictiva aceptable para explorar escenarios hidricos."
    if pd.notna(r2_score) and r2_score >= 0.4:
        return "Modelo con aprendizaje parcial; conviene ampliar variables y validar estabilidad temporal."
    return "Modelo exploratorio; requiere mas datos, ajuste de arquitectura o comparacion con modelos base."


def interpret_experiment_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Add problem type and technical interpretation to C16 summaries.

    Args:
        summary (pd.DataFrame): C16 summary table. Expected columns depend on
            the experiment problem type.

    Returns:
        pd.DataFrame: Copy of ``summary`` with ``Tipo_Problema`` and
        ``Interpretacion_Tecnica`` columns.
    """

    if summary.empty:
        return pd.DataFrame()

    output = summary.copy()
    problem_type = infer_problem_type(output)
    output["Tipo_Problema"] = problem_type
    if problem_type == "Clasificacion":
        output["Interpretacion_Tecnica"] = output.apply(interpret_classification_performance, axis=1)
    elif problem_type == "Regresion":
        output["Interpretacion_Tecnica"] = output.apply(interpret_regression_performance, axis=1)
    else:
        output["Interpretacion_Tecnica"] = "No hay metricas suficientes para interpretar el experimento."
    return output


def summarize_interpretation_experiments(interpretation_dir: Path | None = None) -> pd.DataFrame:
    """Combine C16 summaries from every available experiment.

    Args:
        interpretation_dir (Path | None): Optional interpretation artifact root.

    Returns:
        pd.DataFrame: Consolidated interpretation table with one or more rows
        per experiment.
    """

    root = interpretation_dir or INTERPRETATION_DIR
    frames = []
    for experiment in discover_experiments(root):
        summary = load_interpretation_summary(experiment, root)
        interpreted = interpret_experiment_summary(summary)
        if interpreted.empty:
            continue
        if "Experimento" not in interpreted.columns:
            interpreted.insert(0, "Experimento", experiment)
        frames.append(interpreted)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare_predictions_with_actuals(
    predictions: pd.DataFrame,
    actual_values: pd.Series | list[float] | np.ndarray,
    prediction_column: str = "Prediccion",
) -> pd.DataFrame:
    """Align prediction outputs with observed values.

    Args:
        predictions (pd.DataFrame): Prediction table exported by C15.
        actual_values (pd.Series | list[float] | np.ndarray): Observed values
            aligned to the same horizon as the predictions.
        prediction_column (str): Name of the prediction column.

    Returns:
        pd.DataFrame: Comparison table with ``Registro``, prediction, actual
        value and residual. Returns an empty DataFrame when predictions are
        missing.
    """

    if predictions.empty or prediction_column not in predictions.columns:
        return pd.DataFrame(columns=["Registro", prediction_column, "Real", "Residual"])

    length = min(len(predictions), len(actual_values))
    comparison = predictions.head(length).copy()
    comparison["Real"] = np.asarray(actual_values, dtype=float)[:length]
    comparison["Residual"] = comparison["Real"] - pd.to_numeric(
        comparison[prediction_column],
        errors="coerce",
    )
    return comparison


def metrics_from_comparison(
    comparison: pd.DataFrame,
    problem_type: str,
    prediction_column: str = "Prediccion",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute metrics from an aligned prediction-vs-actual table.

    Args:
        comparison (pd.DataFrame): Output from
            ``compare_predictions_with_actuals``.
        problem_type (str): ``Clasificacion`` or ``Regresion``.
        prediction_column (str): Name of the prediction column.
        threshold (float): Threshold used to binarize classification scores.

    Returns:
        pd.DataFrame: Metric table compatible with C15/C16 reporting.
    """

    if comparison.empty or prediction_column not in comparison.columns or "Real" not in comparison.columns:
        return pd.DataFrame()

    true_values = pd.to_numeric(comparison["Real"], errors="coerce")
    pred_values = pd.to_numeric(comparison[prediction_column], errors="coerce")
    valid = true_values.notna() & pred_values.notna()
    if problem_type.lower().startswith("clas"):
        return classification_metrics(true_values[valid].astype(int), (pred_values[valid] >= threshold).astype(int))

    metrics = regression_metrics(true_values[valid], pred_values[valid])
    mape_row = pd.DataFrame([{"Indicador": "MAPE", "Valor": calculate_mape(true_values[valid], pred_values[valid])}])
    return pd.concat([metrics, mape_row], ignore_index=True)


def classify_variable_dimension(variable_name: str) -> str:
    """Classify a variable into a systemic framework dimension.

    Args:
        variable_name (str): Predictor or target variable name.

    Returns:
        str: Dimension label inferred from keyword rules.
    """

    normalized = variable_name.lower()
    for dimension, keywords in DIMENSION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return dimension
    return "Otra"


def dimension_coverage(variables: list[str]) -> pd.DataFrame:
    """Count selected variables by systemic dimension.

    Args:
        variables (list[str]): Variable names used by an experiment.

    Returns:
        pd.DataFrame: Count table with dimensions and variable totals.
    """

    rows = [{"Variable": variable, "Dimension": classify_variable_dimension(variable)} for variable in variables]
    if not rows:
        return pd.DataFrame(columns=["Dimension", "Variables"])
    return pd.DataFrame(rows).groupby("Dimension").size().reset_index(name="Variables")


def interpretation_artifact_inventory(experiment: str | None = None) -> pd.DataFrame:
    """Inventory C16 interpretation artifacts.

    Args:
        experiment (str | None): Optional experiment identifier used to limit
            the recursive scan.

    Returns:
        pd.DataFrame: File inventory for interpretation outputs.
    """

    return artifact_inventory(INTERPRETATION_DIR, experiment)
