from __future__ import annotations

from pathlib import Path


"""Canonical repository paths for the FRAMEWORK V7 project.

This module centralizes every filesystem location used by the Streamlit app and
the helper modules. Keeping paths in one place makes it easier to reorganize the
repository without hunting for hard-coded strings across the interface.
"""


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "DATA"
MASTER_DIR = DATA_DIR / "MASTER"
NOTEBOOKS_DIR = BASE_DIR / "NOTEBOOKS"

PREDICTIONS_PATH = DATA_DIR / "EVALUACIONES" / "Exp01" / "predicciones.csv"
METADATA_PATH = DATA_DIR / "EVALUACIONES" / "Exp01" / "metadata_prediccion.csv"
ML_DATASET_PATH = (
    DATA_DIR
    / "MACHINE_LEARNING"
    / "C13_MACHINE_LEARNING"
    / "Transformaciones"
    / "Exp01"
    / "dataset_machine_learning_transformado.csv"
)
DIAGNOSTIC_PATH = (
    DATA_DIR
    / "MACHINE_LEARNING"
    / "C13_MACHINE_LEARNING"
    / "Diagnostico"
    / "diagnostico_estadistico_ml.csv"
)
MASTER_PATH = MASTER_DIR / "C09_MASTER" / "Dataset_Maestro_Framework_v03_Con_Imputaciones.csv"
COVERAGE_PATH = MASTER_DIR / "C09_MASTER" / "Dataset_Maestro_Framework_v03_Resumen_Cobertura_Variables.csv"


def rel(path: Path) -> str:
    """Return a repository-relative path for UI labels.

    Args:
        path: Absolute or repository-local path to display.

    Returns:
        A normalized POSIX-like path relative to the repository root.
    """

    return str(path.relative_to(BASE_DIR)).replace("\\", "/")
