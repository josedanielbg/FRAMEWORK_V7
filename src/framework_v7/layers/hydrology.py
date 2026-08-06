"""C02 hydrology layer extracted from the Colab workflow."""

from __future__ import annotations

import pandas as pd

from .base import available_key_variables as _available_key_variables
from .base import feature_frame as _feature_frame
from .base import load_dataset as _load_dataset
from .base import summary as _summary


LAYER_NAME = "C02 - Hidrologica"
KEY_VARIABLES = ["Nivel_Minimo", "valorobservado", "valorobservado_norm", "nivel_rio_m"]


def load_dataset() -> pd.DataFrame:
    """Load the hydrology layer dataset.

    Returns:
        Hydrology layer DataFrame.
    """

    return _load_dataset(LAYER_NAME)


def available_key_variables() -> list[str]:
    """Return hydrology variables present in the dataset.

    Returns:
        List of available hydrology feature names.
    """

    return _available_key_variables(LAYER_NAME, KEY_VARIABLES)


def feature_frame() -> pd.DataFrame:
    """Return a compact hydrology feature frame.

    Returns:
        DataFrame with identifiers and hydrology variables.
    """

    return _feature_frame(LAYER_NAME, KEY_VARIABLES)


def summary() -> dict[str, object]:
    """Summarize the hydrology layer.

    Returns:
        Dictionary with profile, variables and coverage information.
    """

    return _summary(LAYER_NAME, KEY_VARIABLES)

