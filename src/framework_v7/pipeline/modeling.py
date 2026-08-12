"""Modeling helpers extracted from notebook C14."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from framework_v7.paths import MODELING_DIR

from .utils import artifact_inventory, discover_experiments, read_key_value_table, read_table


TENSORS_DIR = MODELING_DIR / "Tensores"
MODELS_DIR = MODELING_DIR / "Modelos"
DIAGNOSTICS_DIR = MODELING_DIR / "Diagnosticos"
LOGS_DIR = MODELING_DIR / "Bitacoras"


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


def load_tensor_metadata(experiment: str, tensors_dir: Path | None = None) -> dict[str, str]:
    """Load tensor metadata exported by notebook C14.

    Args:
        experiment (str): Experiment identifier.
        tensors_dir (Path | None): Optional tensor artifact root.

    Returns:
        dict[str, str]: Tensor metadata as key-value pairs. Returns an empty
        dictionary when the metadata file is absent.
    """

    root = tensors_dir or TENSORS_DIR
    return read_key_value_table(root / experiment / "metadata_tensor.csv")


def load_model_record(experiment: str, models_dir: Path | None = None) -> pd.DataFrame:
    """Load the modeling record for one experiment.

    Args:
        experiment (str): Experiment identifier.
        models_dir (Path | None): Optional model artifact root.

    Returns:
        pd.DataFrame: Modeling record table, usually one row. Returns an empty
        DataFrame when the CSV is missing.
    """

    root = models_dir or MODELS_DIR
    return read_table(root / experiment / f"registro_{experiment}.csv")


def load_model_diagnostic(experiment: str, diagnostics_dir: Path | None = None) -> pd.DataFrame:
    """Load the model diagnostic table for one experiment.

    Args:
        experiment (str): Experiment identifier.
        diagnostics_dir (Path | None): Optional diagnostics artifact root.

    Returns:
        pd.DataFrame: Diagnostic indicators exported by C14/C15.
    """

    root = diagnostics_dir or DIAGNOSTICS_DIR
    return read_table(root / experiment / f"diagnostico_modelo_{experiment}.csv")


def load_model_recommendations(experiment: str, diagnostics_dir: Path | None = None) -> pd.DataFrame:
    """Load model recommendations for one experiment.

    Args:
        experiment (str): Experiment identifier.
        diagnostics_dir (Path | None): Optional diagnostics artifact root.

    Returns:
        pd.DataFrame: Recommendation table exported by the modeling notebook.
    """

    root = diagnostics_dir or DIAGNOSTICS_DIR
    return read_table(root / experiment / f"recomendaciones_{experiment}.csv")


def load_experiment_log(experiment: str, logs_dir: Path | None = None) -> pd.DataFrame:
    """Load the modeling experiment log.

    Args:
        experiment (str): Experiment identifier.
        logs_dir (Path | None): Optional bitacora artifact root.

    Returns:
        pd.DataFrame: Experiment log table for the selected experiment.
    """

    root = logs_dir or LOGS_DIR
    return read_table(root / experiment / "bitacora_maestra_experimentos.csv")


def summarize_modeling_experiments(modeling_dir: Path | None = None) -> pd.DataFrame:
    """Summarize model artifacts across available experiments.

    Args:
        modeling_dir (Path | None): Optional ``DATA/MODELADO`` root.

    Returns:
        pd.DataFrame: One row per experiment with tensor availability, model
        record fields and diagnostic status.
    """

    root = modeling_dir or MODELING_DIR
    experiments = sorted(
        set(discover_experiments(root / "Tensores"))
        | set(discover_experiments(root / "Modelos"))
        | set(discover_experiments(root / "Diagnosticos"))
    )
    rows = []
    for experiment in experiments:
        metadata = load_tensor_metadata(experiment, root / "Tensores")
        record = load_model_record(experiment, root / "Modelos")
        diagnostic = load_model_diagnostic(experiment, root / "Diagnosticos")
        record_row = record.iloc[0].to_dict() if not record.empty else {}
        rows.append(
            {
                "Experimento": experiment,
                "Variable_Objetivo": metadata.get("Variable Objetivo", record_row.get("Variable_Objetivo", "")),
                "Modelo": metadata.get("Modelo", record_row.get("Modelo", "")),
                "Tensores": bool(metadata),
                "Registro": not record.empty,
                "Diagnostico": not diagnostic.empty,
                "Muestras": metadata.get("Numero de Secuencias", record_row.get("Muestras", "")),
                "Estado_General": record_row.get("Estado_General", ""),
            }
        )
    return pd.DataFrame(rows)


def modeling_artifact_inventory(experiment: str | None = None) -> pd.DataFrame:
    """Inventory C14 modeling artifacts.

    Args:
        experiment (str | None): Optional experiment identifier used to limit
            the recursive scan.

    Returns:
        pd.DataFrame: File inventory for modeling artifacts.
    """

    return artifact_inventory(MODELING_DIR, experiment)
