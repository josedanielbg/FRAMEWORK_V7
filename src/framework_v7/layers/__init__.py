"""Layer modules extracted from the Colab workflow.

Each module represents one business/system layer from notebooks C01-C07. The
modules expose a common interface so notebooks, scripts and the Streamlit app
can reuse the same loading and profiling logic.
"""

from __future__ import annotations

from . import climate, governance, hydraulic, hydrology, oni, perception, water_quality


LAYER_MODULES = {
    climate.LAYER_NAME: climate,
    hydrology.LAYER_NAME: hydrology,
    water_quality.LAYER_NAME: water_quality,
    oni.LAYER_NAME: oni,
    hydraulic.LAYER_NAME: hydraulic,
    perception.LAYER_NAME: perception,
    governance.LAYER_NAME: governance,
}


def get_layer_module(layer_name: str):
    """Return the module that implements a system layer.

    Args:
        layer_name: Layer display name, for example ``C01 - Climatica``.

    Returns:
        Python module implementing the requested layer.

    Raises:
        KeyError: If the layer name is not registered.
    """

    return LAYER_MODULES[layer_name]

