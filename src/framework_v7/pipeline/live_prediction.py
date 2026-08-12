"""Live prediction helpers for the Streamlit simulator.

The main live workflow loads persisted ``.keras`` models, the fitted
``scaler.pkl`` artifact and the selected variables from
``variables_machine_learning.csv``. The older lightweight baseline helpers are
kept for compatibility with notebooks and smoke tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from framework_v7.paths import (
    FRAMEWORK_STREAMLIT_DIR,
    MACHINE_LEARNING_DIR,
    MODELING_DIR,
    VARIABLES_MACHINE_LEARNING_PATH,
)

from .utils import read_key_value_table, read_table


IRCA_TARGET_LABEL = "IRCA / riesgo de calidad del agua"
VOLUME_TARGET_LABEL = "Volumen util diario de masa hidrica"
TRANSFORMATIONS_DIR = MACHINE_LEARNING_DIR / "Transformaciones"
TENSORS_DIR = MODELING_DIR / "Tensores"
MODELS_DIR = MODELING_DIR / "Modelos"
LIVE_MODELS_DIR = FRAMEWORK_STREAMLIT_DIR / "live_models"

IRCA_FEATURES = [
    "Precipitacion_mm",
    "Temp_Media_C",
    "Humedad_Relativa",
    "Velocidad_Viento",
    "Radiacion_Solar",
    "ONI",
    "VolumenUtilDiarioMasa",
    "INDICE DE DESEMPENO INSTITUCIONAL",
    "ACCESO A AGUA POTABLE ADECUADO",
    "PORCENTAJE DE LA POBLACION CON ACCESO A METODOS DE SANEAMIENTO ADECUADOS",
    "COBERTURA DE ACUEDUCTO URBANO",
    "COBERTURA DE ALCANTARILLADO URBANO",
    "CONTINUIDAD DE ACUEDUCTO URBANO",
    "AGUAS RESIDUALES TRATADAS",
    "CONDUCTIVIDAD ELECTRICA",
    "DEMANDA BIOQUIMICA DE OXIGENO (DBO5)",
    "DEMANDA QUIMICA DE OXIGENO (DQO)",
    "FOSFORO TOTAL",
    "NITROGENO TOTAL",
    "OXIGENO DISUELTO (OD)",
    "SOLIDOS SUSPENDIDOS TOTALES",
    "TURBIDEZ",
    "pH",
]

VOLUME_FEATURES = [
    "Precipitacion_mm",
    "Temp_Max_C",
    "Temp_Min_C",
    "Temp_Media_C",
    "Humedad_Relativa",
    "Velocidad_Viento",
    "Radiacion_Solar",
    "ONI",
    "Nivel_Minimo",
    "POBLACION TOTAL",
    "DENSIDAD POBLACIONAL",
    "INDICE DE DESEMPENO INSTITUCIONAL",
    "ACCESO A AGUA POTABLE ADECUADO",
    "PORCENTAJE DE LA POBLACION CON ACCESO A METODOS DE SANEAMIENTO ADECUADOS",
    "COBERTURA DE ACUEDUCTO URBANO",
    "COBERTURA DE ACUEDUCTO RURAL",
    "COBERTURA DE ALCANTARILLADO URBANO",
    "COBERTURA DE ALCANTARILLADO RURAL",
    "CONTINUIDAD DE ACUEDUCTO URBANO",
    "AGUAS RESIDUALES TRATADAS",
]

FEATURE_GROUPS = {
    "Clima y variabilidad": [
        "Precipitacion_mm",
        "Temp_Max_C",
        "Temp_Min_C",
        "Temp_Media_C",
        "Humedad_Relativa",
        "Velocidad_Viento",
        "Radiacion_Solar",
        "ONI",
    ],
    "Hidrologia e hidraulica": [
        "VolumenUtilDiarioMasa",
        "Nivel_Minimo",
    ],
    "Gobernanza y servicios": [
        "POBLACION TOTAL",
        "DENSIDAD POBLACIONAL",
        "INDICE DE DESEMPENO INSTITUCIONAL",
        "ACCESO A AGUA POTABLE ADECUADO",
        "PORCENTAJE DE LA POBLACION CON ACCESO A METODOS DE SANEAMIENTO ADECUADOS",
        "COBERTURA DE ACUEDUCTO URBANO",
        "COBERTURA DE ACUEDUCTO RURAL",
        "COBERTURA DE ALCANTARILLADO URBANO",
        "COBERTURA DE ALCANTARILLADO RURAL",
        "CONTINUIDAD DE ACUEDUCTO URBANO",
        "AGUAS RESIDUALES TRATADAS",
    ],
    "Calidad y percepcion": [
        "CONDUCTIVIDAD ELECTRICA",
        "DEMANDA BIOQUIMICA DE OXIGENO (DBO5)",
        "DEMANDA QUIMICA DE OXIGENO (DQO)",
        "FOSFORO TOTAL",
        "NITROGENO TOTAL",
        "OXIGENO DISUELTO (OD)",
        "SOLIDOS SUSPENDIDOS TOTALES",
        "TURBIDEZ",
        "pH",
    ],
}


@dataclass(frozen=True)
class LiveTarget:
    """Configuration for one live prediction target.

    Attributes:
        label: Human-readable target name displayed in the app.
        column: Dataset column used as the response variable.
        features: Candidate predictor columns for the baseline model.
        unit: Unit displayed next to predictions.
        description: Short methodological explanation for the app.
    """

    label: str
    column: str
    features: list[str]
    unit: str
    description: str


@dataclass(frozen=True)
class LivePredictionModel:
    """Fitted ridge-regression baseline for interactive predictions.

    Attributes:
        target: Target configuration used for training.
        features: Predictor columns retained after quality checks.
        coefficients: Ridge coefficients in standardized feature space.
        intercept: Model intercept.
        means: Training means used for standardization.
        stds: Training standard deviations used for standardization.
        defaults: Median values used when an input is missing.
        metrics: Validation and training diagnostics.
    """

    target: LiveTarget
    features: list[str]
    coefficients: np.ndarray
    intercept: float
    means: pd.Series
    stds: pd.Series
    defaults: pd.Series
    metrics: dict[str, float]


@dataclass(frozen=True)
class KerasLiveModel:
    """Loaded Keras experiment artifacts for live inference.

    Attributes:
        experiment: Experiment identifier.
        target: Target variable predicted by the model.
        problem_type: Classification or regression label from metadata.
        features: Ordered predictor variables used by the Keras model.
        window: Temporal window expected by the LSTM.
        model_path: Local ``.keras`` artifact.
        scaler_path: Local fitted scaler artifact.
        target_scaler_path: Optional scaler artifact for inverse-transforming
            regression predictions back to the original target scale.
        transformed_dataset_path: Transformed dataset used as temporal context.
        model: Loaded Keras model object.
        scaler: Loaded scikit-learn scaler object.
        target_scaler: Optional fitted scaler for the target variable.
        transformed_dataset: Historical transformed dataset.
        metadata: Consolidated model metadata.
    """

    experiment: str
    target: str
    problem_type: str
    features: list[str]
    window: int
    model_path: Path
    scaler_path: Path
    target_scaler_path: Path | None
    transformed_dataset_path: Path
    model: object
    scaler: object
    target_scaler: object | None
    transformed_dataset: pd.DataFrame
    metadata: dict[str, str]


LIVE_TARGETS = {
    IRCA_TARGET_LABEL: LiveTarget(
        label=IRCA_TARGET_LABEL,
        column="irca",
        features=IRCA_FEATURES,
        unit="IRCA",
        description=(
            "Baseline para estimar riesgo de calidad del agua combinando clima, "
            "calidad fisicoquimica, disponibilidad y condiciones institucionales."
        ),
    ),
    VOLUME_TARGET_LABEL: LiveTarget(
        label=VOLUME_TARGET_LABEL,
        column="VolumenUtilDiarioMasa",
        features=VOLUME_FEATURES,
        unit="m3",
        description=(
            "Baseline para explorar disponibilidad hidrica a partir de clima, "
            "variabilidad ONI y variables de gobernanza territorial."
        ),
    ),
}


def load_machine_learning_variables(path: Path | None = None) -> pd.DataFrame:
    """Load variables approved for machine-learning inference.

    Args:
        path: Optional path to ``variables_machine_learning.csv``.

    Returns:
        DataFrame with the variables catalog. Empty when the file is missing.
    """

    return read_table(path or VARIABLES_MACHINE_LEARNING_PATH)


def selected_machine_learning_variables(path: Path | None = None) -> list[str]:
    """Return variables marked as selected in ``variables_machine_learning.csv``.

    Args:
        path: Optional path to the variables catalog.

    Returns:
        Ordered variable names suitable for model inputs.
    """

    variables = load_machine_learning_variables(path)
    if variables.empty or "Variable" not in variables.columns:
        return []
    selected = variables.copy()
    if "Estado_Final" in selected.columns:
        selected = selected[
            selected["Estado_Final"]
            .astype(str)
            .str.lower()
            .str.contains("seleccionada", na=False)
        ]
    return selected["Variable"].dropna().astype(str).tolist()


def keras_experiment_options() -> pd.DataFrame:
    """Discover Keras experiments available for live inference.

    Returns:
        DataFrame with experiment, target, model path and selected predictors.
    """

    selected_variables = set(selected_machine_learning_variables())
    rows = []
    if MODELS_DIR.exists():
        model_dirs = sorted(path for path in MODELS_DIR.iterdir() if path.is_dir())
    else:
        model_dirs = []
    for experiment_dir in model_dirs:
        model_path = _model_path(experiment_dir.name)
        if model_path is None:
            continue
        metadata = _experiment_metadata(experiment_dir.name)
        target = metadata.get("Variable Objetivo", metadata.get("Variable Objetivo ", ""))
        features = _metadata_features(metadata)
        features = [
            feature
            for feature in features
            if feature in selected_variables and feature != target
        ]
        scaler_path = _scaler_path(experiment_dir.name)
        if not target or not features:
            continue
        rows.append(
            {
                "Experimento": experiment_dir.name,
                "Variable objetivo": target,
                "Tipo problema": metadata.get("Tipo Problema", ""),
                "Ventana": int(float(metadata.get("Ventana", 12))),
                "Variables": len(features),
                "Variables modelo": ";".join(features),
                "Modelo": model_path.name,
                "Scaler": scaler_path.name if scaler_path else "",
            }
        )
    return pd.DataFrame(rows)


def keras_experiment_artifact_paths(experiment: str) -> dict[str, Path | None]:
    """Return inference artifact paths without loading heavy dependencies.

    Args:
        experiment: Experiment identifier.

    Returns:
        Dictionary with model, scaler and transformed dataset paths.
    """

    return {
        "model": _model_path(experiment),
        "scaler": _scaler_path(experiment),
        "target_scaler": _target_scaler_path(experiment),
        "transformed_dataset": _transformed_dataset_path(experiment),
    }


def load_keras_live_model(experiment: str) -> KerasLiveModel:
    """Load Keras model, scaler and temporal context for one experiment.

    Args:
        experiment: Experiment identifier.

    Returns:
        KerasLiveModel ready to score scenarios.

    Raises:
        RuntimeError: If optional inference dependencies are unavailable.
        ValueError: If required artifacts or variables are missing.
    """

    model_path = _model_path(experiment)
    scaler_path = _scaler_path(experiment)
    target_scaler_path = _target_scaler_path(experiment)
    dataset_path = _transformed_dataset_path(experiment)
    metadata = _experiment_metadata(experiment)
    target = metadata.get("Variable Objetivo", "")
    selected_variables = set(selected_machine_learning_variables())
    features = [
        feature
        for feature in _metadata_features(metadata)
        if feature in selected_variables and feature != target
    ]
    window = int(float(metadata.get("Ventana", 12)))

    if model_path is None:
        raise ValueError(f"No se encontro modelo .keras para {experiment}.")
    if scaler_path is None:
        raise ValueError(f"No se encontro scaler.pkl para {experiment}.")
    if dataset_path is None:
        raise ValueError(f"No se encontro dataset transformado para {experiment}.")
    if not features:
        raise ValueError(
            "No hay variables aptas para inferencia segun variables_machine_learning.csv."
        )

    transformed_dataset = read_table(dataset_path)
    missing_context = [
        feature for feature in features if feature not in transformed_dataset.columns
    ]
    if missing_context:
        raise ValueError(
            f"El dataset transformado no contiene variables requeridas: {missing_context}"
        )

    return KerasLiveModel(
        experiment=experiment,
        target=target,
        problem_type=metadata.get("Tipo Problema", ""),
        features=features,
        window=window,
        model_path=model_path,
        scaler_path=scaler_path,
        target_scaler_path=target_scaler_path,
        transformed_dataset_path=dataset_path,
        model=_load_keras_model(model_path),
        scaler=_load_scaler(scaler_path),
        target_scaler=_load_scaler(target_scaler_path) if target_scaler_path else None,
        transformed_dataset=transformed_dataset,
        metadata=metadata,
    )


def predict_with_keras_model(
    artifacts: KerasLiveModel,
    values: dict[str, float],
    sequence_mode: str = "last_step",
) -> float:
    """Predict a single scenario with a loaded Keras model.

    Args:
        artifacts: Loaded model, scaler and transformed temporal context.
        values: Raw user-provided values keyed by predictor variable.
        sequence_mode: Temporal scenario mode. ``last_step`` changes only the
            final timestep; ``full_window`` repeats the scenario across the
            complete LSTM window.

    Returns:
        Numeric prediction returned by the Keras model.
    """

    sequence = build_keras_sequence(artifacts, values, sequence_mode)
    prediction = artifacts.model.predict(sequence, verbose=0)
    value = float(np.asarray(prediction).reshape(-1)[0])
    if artifacts.target_scaler is not None:
        value = float(artifacts.target_scaler.inverse_transform([[value]]).reshape(-1)[0])
    return value


def build_keras_sequence(
    artifacts: KerasLiveModel,
    values: dict[str, float],
    sequence_mode: str = "last_step",
) -> np.ndarray:
    """Build the LSTM input tensor for one live scenario.

    ``last_step`` uses the last ``window - 1`` transformed historical rows and
    the transformed live scenario as the final time step. ``full_window`` uses
    the live scenario in every timestep, useful for persistent scenario tests.

    Args:
        artifacts: Loaded model artifacts.
        values: Raw user-provided values keyed by predictor variable.
        sequence_mode: Temporal scenario mode: ``last_step`` or
            ``full_window``.

    Returns:
        Numpy tensor with shape ``1 x window x features``.
    """

    raw_row = pd.DataFrame([{feature: float(values[feature]) for feature in artifacts.features}])
    transformed_row = _transform_live_row(raw_row, artifacts)
    if sequence_mode == "full_window":
        sequence = pd.concat(
            [transformed_row[artifacts.features]] * artifacts.window,
            ignore_index=True,
        )
        return sequence.to_numpy(dtype=float).reshape(1, artifacts.window, len(artifacts.features))
    if sequence_mode != "last_step":
        raise ValueError(f"Modo de secuencia no soportado: {sequence_mode}")

    history = artifacts.transformed_dataset[artifacts.features].tail(max(artifacts.window - 1, 0))
    sequence = pd.concat(
        [history, transformed_row[artifacts.features]],
        ignore_index=True,
    ).tail(artifacts.window)
    if len(sequence) < artifacts.window:
        raise ValueError("No hay suficientes filas historicas para construir la ventana temporal.")
    return sequence.to_numpy(dtype=float).reshape(1, artifacts.window, len(artifacts.features))


def _transform_live_row(raw_row: pd.DataFrame, artifacts: KerasLiveModel) -> pd.DataFrame:
    """Transform one raw row using the fitted scaler."""

    scaler_features = list(getattr(artifacts.scaler, "feature_names_in_", artifacts.features))
    missing = [feature for feature in scaler_features if feature not in raw_row.columns]
    if missing:
        raise ValueError(f"Faltan variables para el scaler: {missing}")
    transformed = artifacts.scaler.transform(raw_row[scaler_features])
    transformed_df = pd.DataFrame(transformed, columns=scaler_features, index=raw_row.index)
    return transformed_df[artifacts.features]


def _load_keras_model(model_path: Path) -> object:
    """Load a ``.keras`` model with Keras using an inference backend."""

    os.environ.setdefault("KERAS_BACKEND", "openvino")
    try:
        from keras.models import load_model
    except Exception as error:
        raise RuntimeError(
            "No se pudo importar Keras con el backend de inferencia `openvino`. "
            "Verifica que `keras` y `openvino` esten instalados en Streamlit Cloud. "
            f"Detalle original: {error}"
        ) from error
    return load_model(model_path, compile=False)


def _load_scaler(scaler_path: Path) -> object:
    """Load a fitted scikit-learn scaler from disk."""

    try:
        import joblib
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "No se pudo cargar scaler.pkl porque falta `joblib` o `scikit-learn` en el entorno."
        ) from error
    return joblib.load(scaler_path)


def _model_path(experiment: str) -> Path | None:
    """Return the first Keras model path for an experiment."""

    for root in [LIVE_MODELS_DIR / experiment, MODELS_DIR / experiment]:
        if not root.exists():
            continue
        models = sorted(root.glob("modelo_*.keras"))
        if models:
            return models[0]
    return None


def _scaler_path(experiment: str) -> Path | None:
    """Return scaler path for an experiment, falling back to base experiment."""

    for candidate in _experiment_candidates(experiment):
        for path in [
            LIVE_MODELS_DIR / candidate / "scaler.pkl",
            TRANSFORMATIONS_DIR / candidate / "scaler.pkl",
            MODELS_DIR / candidate / "scaler.pkl",
        ]:
            if path.exists():
                return path
    return None


def _target_scaler_path(experiment: str) -> Path | None:
    """Return optional target scaler path for transformed regression outputs."""

    for candidate in _experiment_candidates(experiment):
        live_root = LIVE_MODELS_DIR / candidate
        if live_root.exists() and list(live_root.glob("modelo_*.keras")):
            live_scaler = live_root / "scaler_y.pkl"
            return live_scaler if live_scaler.exists() else None
        for path in [
            MODELS_DIR / candidate / "scaler_y.pkl",
        ]:
            if path.exists():
                return path
    return None


def _transformed_dataset_path(experiment: str) -> Path | None:
    """Return transformed dataset path, falling back to base experiment."""

    for candidate in _experiment_candidates(experiment):
        for name in [
            "dataset_machine_learning_transformado.csv",
            "dataset_machine_learning_transformado.parquet",
        ]:
            path = LIVE_MODELS_DIR / candidate / name
            if path.exists():
                return path
        for name in [
            "dataset_machine_learning_transformado.csv",
            "dataset_machine_learning_transformado.parquet",
        ]:
            path = TRANSFORMATIONS_DIR / candidate / name
            if path.exists():
                return path
    return None


def _experiment_metadata(experiment: str) -> dict[str, str]:
    """Load model metadata from tensor or sequence artifacts."""

    metadata: dict[str, str] = {}
    for candidate in _experiment_candidates(experiment):
        for path in [
            LIVE_MODELS_DIR / candidate / "metadata_modelo.csv",
            LIVE_MODELS_DIR / candidate / "metadata_tensor.csv",
            LIVE_MODELS_DIR / candidate / "metadata_secuencias.csv",
            MODELS_DIR / candidate / "metadata_modelo.csv",
            TENSORS_DIR / candidate / "metadata_tensor.csv",
            TRANSFORMATIONS_DIR / candidate / "metadata_secuencias.csv",
        ]:
            for key, value in read_key_value_table(path).items():
                metadata.setdefault(key, value)
    metadata.setdefault("Experimento", experiment)
    return metadata


def _metadata_features(metadata: dict[str, str]) -> list[str]:
    """Extract ordered model variables from experiment metadata."""

    raw = metadata.get("Variables Modelo", "")
    if not raw or raw.replace(".", "", 1).isdigit():
        raw = metadata.get("Variables Predictoras", "")
    return [feature.strip() for feature in str(raw).split(";") if feature.strip()]


def _experiment_candidates(experiment: str) -> list[str]:
    """Return experiment and base-experiment fallback candidates."""

    candidates = [experiment]
    if "-" in experiment:
        base = experiment.split("-")[0]
        if base not in candidates:
            candidates.append(base)
    return candidates


def available_live_targets(df: pd.DataFrame) -> list[str]:
    """Return target labels that can be trained with the given dataset.

    Args:
        df: Dataset maestro or compatible modeling table.

    Returns:
        Labels for targets with a numeric response column and enough rows.
    """

    labels = []
    for label, target in LIVE_TARGETS.items():
        if target.column not in df.columns:
            continue
        values = pd.to_numeric(df[target.column], errors="coerce")
        if values.notna().sum() >= 30:
            labels.append(label)
    return labels


def feature_group(feature: str) -> str:
    """Return the systemic layer associated with a feature.

    Args:
        feature: Predictor column name.

    Returns:
        Layer label used to organize Streamlit controls.
    """

    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "Otras variables"


def feature_profile(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build descriptive ranges for live input controls.

    Args:
        df: Dataset containing the predictor columns.
        features: Predictor columns retained by the fitted model.

    Returns:
        DataFrame with min, quartiles, median, mean and max per feature.
    """

    rows = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "Variable": feature,
                "Capa": feature_group(feature),
                "Min": float(values.min()),
                "Q25": float(values.quantile(0.25)),
                "Mediana": float(values.median()),
                "Media": float(values.mean()),
                "Q75": float(values.quantile(0.75)),
                "Max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def fit_live_ridge_model(
    df: pd.DataFrame,
    target_label: str,
    alpha: float = 1.0,
    train_fraction: float = 0.8,
) -> LivePredictionModel:
    """Fit a lightweight ridge-regression model for live simulation.

    Args:
        df: Dataset maestro or compatible modeling table.
        target_label: Key from ``LIVE_TARGETS``.
        alpha: Ridge regularization strength.
        train_fraction: Fraction of ordered rows used for training.

    Returns:
        LivePredictionModel with coefficients, defaults and validation metrics.

    Raises:
        ValueError: If the target is unknown or insufficient data is available.
    """

    if target_label not in LIVE_TARGETS:
        raise ValueError(f"Unknown live target: {target_label}")

    target = LIVE_TARGETS[target_label]
    features = [column for column in target.features if column in df.columns]
    if not features:
        raise ValueError("No predictor columns are available for live training.")

    columns = [target.column, *features]
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(subset=[target.column]).reset_index(drop=True)
    if len(numeric) < 30:
        raise ValueError("Not enough rows are available for live training.")

    x_values = numeric[features].copy()
    defaults = x_values.median(numeric_only=True)
    x_values = x_values.fillna(defaults)
    valid_features = [
        column
        for column in features
        if x_values[column].notna().all() and float(x_values[column].std(ddof=0)) > 0
    ]
    if not valid_features:
        raise ValueError("No valid numeric predictors remain after cleaning.")

    x_values = x_values[valid_features]
    y_values = numeric[target.column].astype(float)
    split_index = max(1, min(len(x_values) - 1, int(len(x_values) * train_fraction)))

    x_train = x_values.iloc[:split_index]
    x_test = x_values.iloc[split_index:]
    y_train = y_values.iloc[:split_index].to_numpy(dtype=float)
    y_test = y_values.iloc[split_index:].to_numpy(dtype=float)

    means = x_train.mean()
    stds = x_train.std(ddof=0).replace(0, 1)
    x_train_scaled = (x_train - means) / stds
    x_test_scaled = (x_test - means) / stds

    design = np.column_stack([np.ones(len(x_train_scaled)), x_train_scaled.to_numpy(dtype=float)])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train

    train_pred = design @ weights
    test_design = np.column_stack([np.ones(len(x_test_scaled)), x_test_scaled.to_numpy(dtype=float)])
    test_pred = test_design @ weights
    metrics = _regression_metrics(y_train, train_pred, y_test, test_pred)
    metrics["filas_entrenamiento"] = float(len(x_train))
    metrics["filas_validacion"] = float(len(x_test))
    metrics["variables"] = float(len(valid_features))

    return LivePredictionModel(
        target=target,
        features=valid_features,
        coefficients=weights[1:],
        intercept=float(weights[0]),
        means=means,
        stds=stds,
        defaults=defaults[valid_features],
        metrics=metrics,
    )


def predict_live(model: LivePredictionModel, values: dict[str, float]) -> float:
    """Predict one scenario with a fitted live baseline.

    Args:
        model: Fitted live prediction model.
        values: Mapping from feature name to user-provided value.

    Returns:
        Numeric prediction in the target's original scale.
    """

    row = pd.Series({feature: values.get(feature, model.defaults[feature]) for feature in model.features})
    row = row.astype(float).fillna(model.defaults)
    scaled = (row - model.means[model.features]) / model.stds[model.features]
    prediction = float(model.intercept + np.dot(scaled.to_numpy(dtype=float), model.coefficients))
    return prediction


def feature_influence(model: LivePredictionModel, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest standardized coefficients for explanation.

    Args:
        model: Fitted live prediction model.
        top_n: Number of predictors to return.

    Returns:
        DataFrame sorted by absolute standardized coefficient.
    """

    influence = pd.DataFrame(
        {
            "Variable": model.features,
            "Capa": [feature_group(feature) for feature in model.features],
            "Coeficiente": model.coefficients,
        }
    )
    influence["Impacto_abs"] = influence["Coeficiente"].abs()
    return influence.sort_values("Impacto_abs", ascending=False).head(top_n)


def irca_risk_label(value: float) -> str:
    """Classify an IRCA value according to common Colombian risk intervals.

    Args:
        value: Predicted or observed IRCA value.

    Returns:
        Human-readable risk level.
    """

    if value <= 5:
        return "Sin riesgo"
    if value <= 14:
        return "Riesgo bajo"
    if value <= 35:
        return "Riesgo medio"
    if value <= 80:
        return "Riesgo alto"
    return "Inviable sanitariamente"


def _regression_metrics(
    y_train: np.ndarray,
    train_pred: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
) -> dict[str, float]:
    """Compute train and validation metrics for regression baselines.

    Args:
        y_train: Observed training target values.
        train_pred: Training predictions.
        y_test: Observed validation target values.
        test_pred: Validation predictions.

    Returns:
        Dictionary with MAE, RMSE and R2 metrics.
    """

    return {
        "mae_train": _mae(y_train, train_pred),
        "rmse_train": _rmse(y_train, train_pred),
        "r2_train": _r2(y_train, train_pred),
        "mae_validacion": _mae(y_test, test_pred),
        "rmse_validacion": _rmse(y_test, test_pred),
        "r2_validacion": _r2(y_test, test_pred),
    }


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""

    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination."""

    if len(y_true) < 2:
        return float("nan")
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0:
        return float("nan")
    numerator = float(np.sum((y_true - y_pred) ** 2))
    return 1 - numerator / denominator
