"""Common helpers for C01-C07 layer modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from framework_v7.catalog import LAYER_CATALOG, SUPPORT_FILES
from framework_v7.data_access import load_excel, read_text
from framework_v7.profiling import dataset_profile, missing_profile


def layer_config(layer_name: str) -> dict[str, object]:
    """Return catalog configuration for a layer.

    Args:
        layer_name: Layer display name.

    Returns:
        Layer configuration dictionary from ``catalog.py``.
    """

    return LAYER_CATALOG[layer_name]


def layer_folder(layer_name: str) -> Path:
    """Return the folder where a layer stores its artifacts.

    Args:
        layer_name: Layer display name.

    Returns:
        Path to the layer folder.
    """

    return layer_config(layer_name)["folder"]


def main_dataset_path(layer_name: str) -> Path:
    """Return the main dataset path for a layer.

    Args:
        layer_name: Layer display name.

    Returns:
        Path to the main Excel dataset.
    """

    config = layer_config(layer_name)
    return config["folder"] / config["main"]


def load_dataset(layer_name: str) -> pd.DataFrame:
    """Load the main dataset for a layer.

    Args:
        layer_name: Layer display name.

    Returns:
        DataFrame with the main layer dataset.
    """

    return load_excel(main_dataset_path(layer_name))


def load_support(layer_name: str, artifact_name: str) -> pd.DataFrame:
    """Load a support artifact for a layer.

    Args:
        layer_name: Layer display name.
        artifact_name: Support artifact label, such as ``Metadata``.

    Returns:
        DataFrame with the support artifact, or an empty DataFrame if missing.
    """

    folder = layer_folder(layer_name)
    return load_excel(folder / SUPPORT_FILES[artifact_name])


def read_layer_readme(layer_name: str) -> str:
    """Read the README attached to a layer.

    Args:
        layer_name: Layer display name.

    Returns:
        README content, or an empty string when absent.
    """

    return read_text(layer_folder(layer_name) / "08_README.md")


def available_key_variables(layer_name: str, key_variables: list[str]) -> list[str]:
    """Return declared key variables that are present in the dataset.

    Args:
        layer_name: Layer display name.
        key_variables: Candidate variables expected for the layer.

    Returns:
        Variables found in the layer dataset.
    """

    df = load_dataset(layer_name)
    return [column for column in key_variables if column in df.columns]


def feature_frame(layer_name: str, key_variables: list[str]) -> pd.DataFrame:
    """Build a compact feature frame for a layer.

    Args:
        layer_name: Layer display name.
        key_variables: Candidate variables expected for the layer.

    Returns:
        DataFrame with identity/date columns plus available key variables.
    """

    df = load_dataset(layer_name)
    base_columns = [column for column in ["Fecha", "fecha", "Nodo", "Municipio", "Anio", "Mes"] if column in df.columns]
    selected = base_columns + available_key_variables(layer_name, key_variables)
    return df[selected].copy() if selected else df.copy()


def summary(layer_name: str, key_variables: list[str]) -> dict[str, object]:
    """Compute a reusable summary for a layer.

    Args:
        layer_name: Layer display name.
        key_variables: Candidate variables expected for the layer.

    Returns:
        Dictionary with profile, key variables and missing-value information.
    """

    df = load_dataset(layer_name)
    return {
        "layer": layer_name,
        "profile": dataset_profile(df),
        "key_variables": available_key_variables(layer_name, key_variables),
        "missing": missing_profile(df),
    }

