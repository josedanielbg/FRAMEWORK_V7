"""C01 climate layer extracted from the Colab workflow."""

from __future__ import annotations

import pandas as pd

from .base import available_key_variables as _available_key_variables
from .base import feature_frame as _feature_frame
from .base import load_dataset as _load_dataset
from .base import summary as _summary


LAYER_NAME = "C01 - Climatica"
KEY_VARIABLES = [
    "Precipitacion_mm",
    "Temp_Max_C",
    "Temp_Min_C",
    "Temp_Media_C",
    "Humedad_Relativa",
    "Velocidad_Viento",
    "Radiacion_Solar",
]


def load_dataset() -> pd.DataFrame:
    """Load the climate layer dataset.

    Returns:
        Climate layer DataFrame.
    """

    return _load_dataset(LAYER_NAME)


def available_key_variables() -> list[str]:
    """Return climate variables present in the dataset.

    Returns:
        List of available climate feature names.
    """

    return _available_key_variables(LAYER_NAME, KEY_VARIABLES)


def feature_frame() -> pd.DataFrame:
    """Return a compact climate feature frame.

    Returns:
        DataFrame with temporal/node identifiers and climate variables.
    """

    return _feature_frame(LAYER_NAME, KEY_VARIABLES)


def summary() -> dict[str, object]:
    """Summarize the climate layer.

    Returns:
        Dictionary with profile, variables and coverage information.
    """

    return _summary(LAYER_NAME, KEY_VARIABLES)

