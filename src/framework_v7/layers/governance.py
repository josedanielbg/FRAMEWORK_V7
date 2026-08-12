"""C07 governance layer extracted from the Colab workflow."""

from __future__ import annotations

import pandas as pd

from .base import available_key_variables as _available_key_variables
from .base import feature_frame as _feature_frame
from .base import load_dataset as _load_dataset
from .base import summary as _summary


LAYER_NAME = "C07 - Gobernanza"
KEY_VARIABLES = [
    "INDICE DE DESEMPENO INSTITUCIONAL",
    "ACCESO A AGUA POTABLE ADECUADO",
    "PORCENTAJE DE LA POBLACION CON ACCESO A METODOS DE SANEAMIENTO ADECUADOS",
    "COBERTURA DE ACUEDUCTO URBANO",
    "COBERTURA DE ACUEDUCTO RURAL",
    "COBERTURA DE ALCANTARILLADO URBANO",
    "COBERTURA DE ALCANTARILLADO RURAL",
    "AGUAS RESIDUALES TRATADAS",
]


def load_dataset() -> pd.DataFrame:
    """Load the governance layer dataset.

    Returns:
        Governance layer DataFrame.
    """

    return _load_dataset(LAYER_NAME)


def available_key_variables() -> list[str]:
    """Return governance variables present in the dataset.

    Returns:
        List of available governance feature names.
    """

    return _available_key_variables(LAYER_NAME, KEY_VARIABLES)


def feature_frame() -> pd.DataFrame:
    """Return a compact governance feature frame.

    Returns:
        DataFrame with identifiers and governance variables.
    """

    return _feature_frame(LAYER_NAME, KEY_VARIABLES)


def summary() -> dict[str, object]:
    """Summarize the governance layer.

    Returns:
        Dictionary with profile, variables and coverage information.
    """

    return _summary(LAYER_NAME, KEY_VARIABLES)
