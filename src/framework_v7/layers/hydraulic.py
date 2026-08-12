"""C05 hydraulic layer extracted from the Colab workflow."""

from __future__ import annotations

import pandas as pd

from .base import available_key_variables as _available_key_variables
from .base import feature_frame as _feature_frame
from .base import load_dataset as _load_dataset
from .base import summary as _summary


LAYER_NAME = "C05 - Hidraulica"
KEY_VARIABLES = ["VolumenUtilDiarioMasa", "Volumen_Imputado", "Nivel_Minimo"]


def load_dataset() -> pd.DataFrame:
    """Load the hydraulic layer dataset.

    Returns:
        Hydraulic layer DataFrame.
    """

    return _load_dataset(LAYER_NAME)


def available_key_variables() -> list[str]:
    """Return hydraulic variables present in the dataset.

    Returns:
        List of available hydraulic feature names.
    """

    return _available_key_variables(LAYER_NAME, KEY_VARIABLES)


def feature_frame() -> pd.DataFrame:
    """Return a compact hydraulic feature frame.

    Returns:
        DataFrame with identifiers and hydraulic variables.
    """

    return _feature_frame(LAYER_NAME, KEY_VARIABLES)


def summary() -> dict[str, object]:
    """Summarize the hydraulic layer.

    Returns:
        Dictionary with profile, variables and coverage information.
    """

    return _summary(LAYER_NAME, KEY_VARIABLES)
