"""Integration helpers extracted from notebook C08."""

from __future__ import annotations

from functools import reduce
from typing import Mapping

import pandas as pd

from .utils import ensure_columns


def build_node(df: pd.DataFrame, municipality_col: str = "municipio", station_col: str | None = None) -> pd.Series:
    """Build the integration node used by layer datasets.

    Args:
        df: Input dataset.
        municipality_col: Municipality column name.
        station_col: Optional station column name.

    Returns:
        Series with uppercase, stripped node labels.
    """

    ensure_columns(df, [municipality_col], "layer dataset")
    node = df[municipality_col].astype(str).str.strip().str.upper()
    if station_col and station_col in df.columns:
        station = df[station_col].astype(str).str.strip().str.upper()
        node = node.where(station.eq("") | station.eq("NAN"), node + " - " + station)
    return node


def standardize_date_column(df: pd.DataFrame, date_col: str = "Fecha") -> pd.DataFrame:
    """Convert a date-like column to monthly timestamps.

    Args:
        df: Input dataset.
        date_col: Date column name.

    Returns:
        Copy with standardized dates.
    """

    ensure_columns(df, [date_col], "layer dataset")
    output = df.copy()
    output[date_col] = pd.to_datetime(output[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return output


def standardize_layer_keys(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    node_col: str = "Nodo",
    municipality_col: str | None = None,
    station_col: str | None = None,
) -> pd.DataFrame:
    """Ensure a dataset has the common keys used by the master dataset.

    Args:
        df: Input layer dataset.
        date_col: Date column name.
        node_col: Output node column name.
        municipality_col: Optional municipality column for node creation.
        station_col: Optional station column for node creation.

    Returns:
        DataFrame with standardized ``Fecha`` and ``Nodo`` keys when possible.
    """

    output = df.copy()
    if date_col in output.columns:
        output = standardize_date_column(output, date_col)
        if date_col != "Fecha":
            output = output.rename(columns={date_col: "Fecha"})
    if node_col not in output.columns and municipality_col:
        output[node_col] = build_node(output, municipality_col, station_col)
    if node_col in output.columns and node_col != "Nodo":
        output = output.rename(columns={node_col: "Nodo"})
    if "Nodo" in output.columns:
        output["Nodo"] = output["Nodo"].astype(str).str.strip().str.upper()
    return output


def merge_layer_frames(frames: Mapping[str, pd.DataFrame], keys: list[str] | None = None) -> pd.DataFrame:
    """Merge multiple layer frames into one master dataset.

    Args:
        frames: Mapping of layer labels to DataFrames.
        keys: Merge keys. Defaults to ``Fecha`` and ``Nodo``.

    Returns:
        Merged master DataFrame.
    """

    keys = keys or ["Fecha", "Nodo"]
    prepared = []
    for label, frame in frames.items():
        if frame.empty:
            continue
        ensure_columns(frame, keys, label)
        prepared.append(frame.copy())
    if not prepared:
        return pd.DataFrame()
    return reduce(lambda left, right: pd.merge(left, right, on=keys, how="outer"), prepared)


def integration_quality(master: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Build a simple quality report for the integrated master dataset.

    Args:
        master: Integrated dataset.
        keys: Key columns used for duplicate checks.

    Returns:
        DataFrame with structural quality indicators.
    """

    keys = keys or ["Fecha", "Nodo"]
    rows = [
        {"Indicador": "filas", "Valor": len(master)},
        {"Indicador": "columnas", "Valor": master.shape[1]},
        {"Indicador": "nulos", "Valor": int(master.isna().sum().sum())},
        {"Indicador": "duplicados_totales", "Valor": int(master.duplicated().sum())},
    ]
    if all(column in master.columns for column in keys):
        rows.append({"Indicador": "duplicados_llave", "Valor": int(master.duplicated(subset=keys).sum())})
    return pd.DataFrame(rows)
