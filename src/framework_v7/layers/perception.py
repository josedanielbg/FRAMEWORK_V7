"""C06 perception layer extracted from the Colab workflow."""

from __future__ import annotations

import pandas as pd

from .base import available_key_variables as _available_key_variables
from .base import feature_frame as _feature_frame
from .base import load_dataset as _load_dataset
from .base import summary as _summary


LAYER_NAME = "C06 - Percepcion"
KEY_VARIABLES = [
    "irca",
    "nivel_de_riesgo",
    "municipio",
    "codigoestacion",
]


def load_dataset() -> pd.DataFrame:
    """Load the perception layer dataset.

    Returns:
        Perception layer DataFrame.
    """

    return _load_dataset(LAYER_NAME)


def available_key_variables() -> list[str]:
    """Return perception variables present in the dataset.

    Returns:
        List of available perception feature names.
    """

    return _available_key_variables(LAYER_NAME, KEY_VARIABLES)


def feature_frame() -> pd.DataFrame:
    """Return a compact perception feature frame.

    Returns:
        DataFrame with identifiers and perception variables.
    """

    return _feature_frame(LAYER_NAME, KEY_VARIABLES)


def summary() -> dict[str, object]:
    """Summarize the perception layer.

    Returns:
        Dictionary with profile, variables and coverage information.
    """

    return _summary(LAYER_NAME, KEY_VARIABLES)
